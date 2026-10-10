//! Port of libdeflate `lib/matchfinder_common.h` — shared matchfinder definitions.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/matchfinder_common.h
//!
//! # What is ported
//! - `MATCHFINDER_WINDOW_ORDER` / `MATCHFINDER_WINDOW_SIZE` constants
//! - `mf_pos_t` type alias (`i16`, mirrors C `s16`)
//! - `MATCHFINDER_INITVAL` sentinel
//! - `matchfinder_init` — fill a position table with `MATCHFINDER_INITVAL`
//! - `matchfinder_rebase` — slide/saturate all position entries by one window
//! - `lz_hash` — multiplicative hash (multiply-shift, 32-bit)
//! - `load_u24_unaligned` / `loaded_u32_to_u24` — 3-byte LE load helpers
//! - `lz_extend` — safe byte-then-word match extension
//!
//! # N/A in safe Rust (documented here, not ported)
//! - `MATCHFINDER_MEM_ALIGNMENT` (32 bytes, needed by AVX-2 SIMD): heap allocations
//!   via `Vec`/`Box` do not require explicit alignment annotations in safe Rust.
//!   The SIMD init/rebase tiers use unaligned loads, so alignment only affects speed.
//! - `MATCHFINDER_SIZE_ALIGNMENT` (1024 bytes): same reason — the scalar `matchfinder_init`
//!   and `matchfinder_rebase` iterate over every element and do not need size alignment.
//! - `MATCHFINDER_ALIGNED` attribute: no raw-pointer structs; `Vec` fields are heap-allocated.
//! - `prefetchw`: Rust stable has no prefetch intrinsic; these are performance hints only
//!   and have no effect on correctness.
//! - Architecture-specific `matchfinder_init`/`matchfinder_rebase` overrides (ARM/x86/RISC-V):
//!   only the scalar (generic) path is ported here; SIMD variants are a later phase.
//! - `load_word_unaligned` / `machine_word_t`: `lz_extend` uses `usize`-wide loads below.

use crate::common::get_unaligned_le32;

// ---------------------------------------------------------------------------
// Window constants — mirrors `MATCHFINDER_WINDOW_ORDER` / `MATCHFINDER_WINDOW_SIZE`.
// libdeflate sets these per compression format; for DEFLATE the window is 32 KiB.
// ---------------------------------------------------------------------------

/// `log2(MATCHFINDER_WINDOW_SIZE)`. For DEFLATE: 15 → 32 KiB window.
pub const MATCHFINDER_WINDOW_ORDER: u32 = 15;

/// Sliding window size in bytes. Mirrors `MATCHFINDER_WINDOW_SIZE`.
pub const MATCHFINDER_WINDOW_SIZE: usize = 1 << MATCHFINDER_WINDOW_ORDER;

// ---------------------------------------------------------------------------
// Position type and sentinel.
// C: `typedef s16 mf_pos_t;`  →  Rust: `i16`.
// Positions are stored relative to the current window base.
// Valid in-window positions are in [0, MATCHFINDER_WINDOW_SIZE).
// Out-of-range / uninitialized entries hold MATCHFINDER_INITVAL = -32768 = i16::MIN.
// ---------------------------------------------------------------------------

/// Position type for matchfinder tables.  Mirrors C `mf_pos_t` (`s16` / `i16`).
pub type MfPos = i16;

/// Sentinel value written into uninitialised or slid-out table entries.
/// Mirrors `MATCHFINDER_INITVAL = (mf_pos_t)-MATCHFINDER_WINDOW_SIZE`.
/// For MATCHFINDER_WINDOW_SIZE == 32768: `-32768` == `i16::MIN`.
/// We use `i16::MIN` directly because `-(32768i16)` would overflow in const context.
pub const MATCHFINDER_INITVAL: MfPos = i16::MIN;

/// Byte alignment of every matchfinder table, C's `MATCHFINDER_MEM_ALIGNMENT`
/// (`matchfinder_common.h:66`). The SIMD init/rebase use unaligned loads and
/// stores, so this is for speed only (measured), not soundness.
const TAB_ALIGN: usize = 32;
/// Extra entries allocated so a `TAB_ALIGN`-aligned start always fits.
const TAB_PAD: usize = TAB_ALIGN / core::mem::size_of::<MfPos>();

/// A fixed-size `N`-entry table whose first entry is `TAB_ALIGN`-aligned,
/// matching C's `MATCHFINDER_ALIGNED`. Shared by all three matchfinders
/// (Hc/Ht/Bt), each with its own `N`. `Deref`/`DerefMut` to `[MfPos]` keep
/// every `self.table[..]` call site unchanged.
///
/// Built straight on the heap: the earlier `Box::new(AlignedTab([0; N]))`
/// assembled the hundreds-of-KiB array on the stack first in debug builds
/// and overflowed a 1 MiB main-thread stack (Windows) on the first ZIP
/// write. The table is a padded heap slice that starts at an aligned offset,
/// so no `unsafe` allocation is needed.
pub(crate) struct AlignedTab<const N: usize> {
    buf: Box<[MfPos]>,
    off: usize,
}

impl<const N: usize> AlignedTab<N> {
    /// A zeroed table, allocated directly on the heap.
    pub(crate) fn zeroed() -> Self {
        let buf = vec![0 as MfPos; N + TAB_PAD].into_boxed_slice();
        // `align_offset` counts elements; for a 2-byte-aligned pointer it is
        // always < TAB_PAD. If an implementation ever reports "can't", start
        // at 0: slower, still correct.
        let off = match buf.as_ptr().align_offset(TAB_ALIGN) {
            off if off < TAB_PAD => off,
            _ => 0,
        };
        AlignedTab { buf, off }
    }
}

impl<const N: usize> core::ops::Deref for AlignedTab<N> {
    type Target = [MfPos];
    #[inline(always)]
    fn deref(&self) -> &[MfPos] {
        &self.buf[self.off..self.off + N]
    }
}

impl<const N: usize> core::ops::DerefMut for AlignedTab<N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [MfPos] {
        &mut self.buf[self.off..self.off + N]
    }
}

// ---------------------------------------------------------------------------
// matchfinder_init — fill hash/chain/tree table with MATCHFINDER_INITVAL.
// Mirrors the scalar `matchfinder_init(mf_pos_t *data, size_t size)`.
// In C, `size` is byte-count; here `data` is already the right slice.
// ---------------------------------------------------------------------------

/// Initialise (or re-initialise) a position table by filling every entry with
/// `MATCHFINDER_INITVAL`.  Mirrors `matchfinder_init`.
///
/// Dispatches to the AVX2/SSE2/NEON SIMD fill when available (byte-identical to the
/// scalar `fill`, since both write the same constant to every entry).
#[inline]
pub fn matchfinder_init(data: &mut [MfPos]) {
    match crate::cpu::matchfinder_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::MatchfinderTier::Avx2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("avx2")`.
            unsafe { crate::matchfinder_simd::matchfinder_init_avx2(data) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::MatchfinderTier::Sse2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("sse2")`.
            unsafe { crate::matchfinder_simd::matchfinder_init_sse2(data) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::cpu::MatchfinderTier::Neon => {
            // SAFETY: NEON is a baseline feature of every aarch64 target.
            unsafe { crate::matchfinder_neon::matchfinder_init_neon(data) }
        }
        crate::cpu::MatchfinderTier::Scalar => data.fill(MATCHFINDER_INITVAL),
    }
}

// ---------------------------------------------------------------------------
// matchfinder_rebase — slide all position entries by one window.
// Mirrors the scalar `matchfinder_rebase(mf_pos_t *data, size_t size)`.
//
// For MATCHFINDER_WINDOW_SIZE == 32768 the C code uses a branchless trick:
//   data[i] = 0x8000 | (data[i] & ~(data[i] >> 15));
// which is equivalent to: if data[i] >= 0, subtract 32768; else set to -32768.
// We replicate the branchless form exactly so the bit pattern matches.
// ---------------------------------------------------------------------------

/// Slide all entries in a position table by `MATCHFINDER_WINDOW_SIZE` bytes.
/// Entries that would go below `MATCHFINDER_INITVAL` are clamped to it.
/// Must be called after each full `MATCHFINDER_WINDOW_SIZE` bytes processed.
/// Mirrors `matchfinder_rebase`.
#[inline]
pub fn matchfinder_rebase(data: &mut [MfPos]) {
    // SIMD dispatch: AVX2/SSE2 use PADDSW (NEON: vqaddq_s16) (signed-saturating add of
    // -WINDOW_SIZE), which is byte-identical to the branchless scalar form
    // below for WINDOW_SIZE == 32768. The scalar path remains the fallback and
    // the source of truth.
    match crate::cpu::matchfinder_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::MatchfinderTier::Avx2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("avx2")`.
            unsafe { crate::matchfinder_simd::matchfinder_rebase_avx2(data) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::MatchfinderTier::Sse2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("sse2")`.
            unsafe { crate::matchfinder_simd::matchfinder_rebase_sse2(data) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::cpu::MatchfinderTier::Neon => {
            // SAFETY: NEON is a baseline feature of every aarch64 target.
            unsafe { crate::matchfinder_neon::matchfinder_rebase_neon(data) }
        }
        crate::cpu::MatchfinderTier::Scalar => matchfinder_rebase_scalar(data),
    }
}

/// Scalar branchless `matchfinder_rebase` for WINDOW_SIZE == 32768:
///
/// ```text
/// data[i] = 0x8000 | (data[i] & ~(data[i] >> 15))
/// ```
///
/// Reinterpreting as bit ops on i16: `data[i] >> 15` is 0xFFFF (negative) or
/// 0x0000 (non-negative); `~(data[i] >> 15)` is its complement; `data[i] & ~(…)`
/// clears the value if negative and keeps it if non-negative; `0x8000 | …` sets
/// the sign bit, yielding INITVAL if it was negative else `val - 32768`. I.e.
/// signed-saturating subtraction of 32768. This is the source of truth.
#[inline]
pub(crate) fn matchfinder_rebase_scalar(data: &mut [MfPos]) {
    for v in data.iter_mut() {
        let x = *v;
        let mask = x >> 15; // arithmetic shift: all-ones if negative
        *v = (0x8000u16 as i16) | (x & !mask);
    }
}

// ---------------------------------------------------------------------------
// lz_hash — multiplicative hash.
// Mirrors `lz_hash(u32 seq, unsigned num_bits)`:
//   return (u32)(seq * 0x1E35A7BD) >> (32 - num_bits);
// The multiply uses wrapping arithmetic (same as C unsigned 32-bit overflow).
// ---------------------------------------------------------------------------

/// Multiplicative LZ hash.  `seq` is the 4-byte (or 3-byte) sequence value;
/// `num_bits` is the desired hash table order.  Mirrors `lz_hash`.
#[inline(always)]
pub fn lz_hash(seq: u32, num_bits: u32) -> u32 {
    seq.wrapping_mul(0x1E35A7BDu32) >> (32 - num_bits)
}

// ---------------------------------------------------------------------------
// load_u24_unaligned / loaded_u32_to_u24 — 3-byte load helpers.
// C: on LE (which is our target), loaded_u32_to_u24(v) = v & 0xFFFFFF.
//    load_u24_unaligned(p) = loaded_u32_to_u24(load_u32_unaligned(p)).
// We always target little-endian; the BE path is unreachable for exr-rs.
// The caller must ensure at least 4 bytes are available at the slice start.
// ---------------------------------------------------------------------------

/// Mask a u32 down to its 3 low-order bytes (LE order).
/// Mirrors `loaded_u32_to_u24(v)` on little-endian platforms.
#[inline(always)]
pub fn loaded_u32_to_u24(v: u32) -> u32 {
    v & 0x00FF_FFFF
}

/// Load 3 bytes from the start of `p` into the 24 low bits of a `u32`.
/// Reads 4 bytes (caller must ensure at least 4 bytes available).
/// Mirrors `load_u24_unaligned(p)` on little-endian platforms.
#[inline(always)]
pub fn load_u24_unaligned(p: &[u8]) -> u32 {
    loaded_u32_to_u24(get_unaligned_le32(p))
}

// ---------------------------------------------------------------------------
// lz_extend — byte-exact match extension.
// Mirrors `lz_extend(strptr, matchptr, start_len, max_len)`.
//
// The C version has a fast word-at-a-time path gated on UNALIGNED_ACCESS_IS_FAST.
// We implement the same optimisation using `usize`-wide reads via safe
// `from_ne_bytes` (native-endian, for the XOR-then-count-zeros trick).
// The scalar fallback is the correctness source of truth; the word path
// produces byte-identical results because it finds the same first differing byte.
// ---------------------------------------------------------------------------

/// Extend a match: starting from `start_len`, advance while bytes agree, up to
/// `max_len`.  Returns the total match length.  Mirrors `lz_extend`.
///
/// `in_buf[str_pos..]` is the position being matched *against* (newer).
/// `in_buf[match_pos..]` is the candidate match (older/earlier in input).
///
/// Like the C original, callers are responsible for ensuring both
/// `str_pos + max_len` and `match_pos + max_len` are within `in_buf`.
/// We additionally clamp to the true buffer size as a safety measure.
#[inline]
pub fn lz_extend(in_buf: &[u8], str_pos: usize, match_pos: usize, start_len: u32, max_len: u32) -> u32 {
    // C's `lz_extend` (matchfinder_common.h) uses `max_len`
    // directly with no bounds clamp at all -- it trusts the caller to have
    // already guaranteed `str_pos + max_len <= in_buf.len()` and `match_pos +
    // max_len <= in_buf.len()` (every real caller does, via
    // `adjust_max_and_nice_len`/`BT_MATCHFINDER_REQUIRED_NBYTES`-style
    // preconditions, the same invariant chain the matchfinders rely on). This
    // port used to silently clamp `max` to `in_buf.len() - str_pos` /
    // `in_buf.len() - match_pos` instead -- masking a caller bug (a `max_len`
    // that actually would run past the buffer) by quietly returning a
    // shorter, still-technically-safe match length instead of surfacing the
    // violation. `debug_assert!` the precondition and use `max_len` directly,
    // matching C exactly and the project's "assert, don't silently degrade"
    // rule for latent invariant violations (same fix shape as
    // `bt_matchfinder`'s `can_extend`).
    debug_assert!(
        str_pos + max_len as usize <= in_buf.len() && match_pos + max_len as usize <= in_buf.len(),
        "lz_extend: max_len ({max_len}) runs past the buffer end from str_pos ({str_pos}) or \
         match_pos ({match_pos}) (in_buf.len()={}) — caller must guarantee enough trailing bytes",
        in_buf.len()
    );
    let mut len = start_len as usize;
    let max = max_len as usize;

    // Word-at-a-time fast path: read usize-wide chunks, XOR, find first differing byte.
    const W: usize = core::mem::size_of::<usize>();
    while len + W <= max {
        let s_word = usize::from_ne_bytes(
            in_buf[str_pos + len..str_pos + len + W].try_into().unwrap(),
        );
        let m_word = usize::from_ne_bytes(
            in_buf[match_pos + len..match_pos + len + W].try_into().unwrap(),
        );
        let diff = s_word ^ m_word;
        if diff != 0 {
            // Find the first differing byte within this word.
            // On LE: first byte is LSB → trailing zeros / 8.
            // On BE: first byte is MSB → leading zeros / 8.
            #[cfg(target_endian = "little")]
            let byte_idx = diff.trailing_zeros() as usize / 8;
            #[cfg(target_endian = "big")]
            let byte_idx = diff.leading_zeros() as usize / 8;
            return (len + byte_idx) as u32;
        }
        len += W;
    }

    // Scalar tail.
    while len < max && in_buf[str_pos + len] == in_buf[match_pos + len] {
        len += 1;
    }
    len as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A position-table test vector for the SIMD init/rebase tiers: every relevant i16
/// class (INITVAL, negatives, zero, small/large positives, i16::MAX), at lengths the
/// caller picks off the lane count to exercise each tier's scalar remainder.
#[cfg(test)]
pub(crate) fn rebase_sample(len: usize) -> Vec<MfPos> {
    let mut v = Vec::with_capacity(len);
    let mut x: u32 = 0xABCD_1234;
    for _ in 0..len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.push((x & 0xFFFF) as i16);
    }
    if len > 5 {
        v[0] = MATCHFINDER_INITVAL;
        v[1] = 0;
        v[2] = -1;
        v[3] = i16::MAX;
        v[4] = 100;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table starts `TAB_ALIGN`-aligned and has exactly `N` entries.
    #[test]
    fn aligned_tab_is_aligned_and_sized() {
        fn check<const N: usize>() {
            let t = AlignedTab::<N>::zeroed();
            assert_eq!(t.len(), N);
            assert_eq!(t.as_ptr() as usize % TAB_ALIGN, 0, "N={N}");
            assert!(t.iter().all(|&v| v == 0));
        }
        check::<17>();
        check::<{ 1 << 17 }>();
        check::<{ 3 * (1 << 16) + 2 * 32768 }>();
    }

    /// Building a compressor at every level must not put the matchfinder
    /// tables (up to ~512 KiB) on the stack: it has to succeed on a thread
    /// with a 128 KiB stack. The old `Box::new(AlignedTab([0; N]))` overflowed
    /// here in debug builds.
    #[test]
    fn compressor_tables_never_touch_the_stack() {
        std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(|| {
                for level in 0..=12 {
                    let mut c = crate::deflate_compress::Compressor::new(level).expect("valid level");
                    let input = [7u8; 64];
                    let mut out = vec![0u8; crate::deflate_compress::deflate_compress_bound(input.len())];
                    assert!(c.deflate_compress(&input, &mut out) > 0, "level {level}");
                }
            })
            .expect("spawn small-stack thread")
            .join()
            .expect("compressor construction overflowed a 128 KiB stack");
    }

    #[test]
    fn window_constants() {
        assert_eq!(MATCHFINDER_WINDOW_SIZE, 32768);
        assert_eq!(MATCHFINDER_INITVAL, i16::MIN);
        assert_eq!(MATCHFINDER_INITVAL, -32768);
    }

    #[test]
    fn matchfinder_init_fills_initval() {
        let mut tab = vec![0i16; 8];
        matchfinder_init(&mut tab);
        assert!(tab.iter().all(|&v| v == MATCHFINDER_INITVAL));
    }

    #[test]
    fn matchfinder_rebase_positive_entry() {
        // A positive entry (in-window) should be decremented by WINDOW_SIZE (32768).
        let mut tab = [0i16; 1];
        tab[0] = 100;
        matchfinder_rebase(&mut tab);
        // 100 - 32768 = -32668, but via the bit trick: 0x8000 | (100 & 0xFFFF) = 0x8064 = -32668
        // 100 - 32768 = -32668; bit-trick: 0x8000 | (100 & 0xFFFF) = 0x8064 = -32668
        assert_eq!(tab[0], -32668i16);
    }

    #[test]
    fn matchfinder_rebase_negative_entry_stays_initval() {
        // A negative entry should be clamped to MATCHFINDER_INITVAL.
        let mut tab = [MATCHFINDER_INITVAL; 4];
        tab[1] = -1i16;
        matchfinder_rebase(&mut tab);
        // -1 is negative → masked to 0 → 0x8000 = INITVAL
        // INITVAL itself is negative → clamped too
        assert_eq!(tab[0], MATCHFINDER_INITVAL);
        assert_eq!(tab[1], MATCHFINDER_INITVAL);
    }

    #[test]
    fn matchfinder_rebase_zero_entry() {
        // 0 is non-negative → 0x8000 | (0 & 0xFFFF) = 0x8000 = -32768 = INITVAL
        let mut tab = [0i16; 1];
        matchfinder_rebase(&mut tab);
        assert_eq!(tab[0], MATCHFINDER_INITVAL);
    }

    #[test]
    fn lz_hash_known_values() {
        // The hash must match the C formula: (seq * 0x1E35A7BD) >> (32 - num_bits)
        let seq: u32 = 0x61626364; // "abcd" LE
        let bits = 16u32;
        let expected = seq.wrapping_mul(0x1E35A7BDu32) >> (32 - bits);
        assert_eq!(lz_hash(seq, bits), expected);

        // Check 15-bit hash
        let expected15 = seq.wrapping_mul(0x1E35A7BDu32) >> 17;
        assert_eq!(lz_hash(seq, 15), expected15);
    }

    #[test]
    fn load_u24_and_loaded_u32_to_u24() {
        let buf = [0xAAu8, 0xBB, 0xCC, 0xDD];
        // LE: u32 = 0xDDCCBBAA; low 24 bits = 0x00CCBBAA
        assert_eq!(load_u24_unaligned(&buf), 0x00CC_BBAA);
        assert_eq!(loaded_u32_to_u24(0xDDCCBBAA), 0x00CC_BBAA);
    }

    #[test]
    fn lz_extend_no_match() {
        let buf = b"abcXYZ";
        // str at 0 ("abcXYZ"), match at 0 itself: start_len=0, max=6 → full 6
        assert_eq!(lz_extend(buf, 0, 0, 0, 6), 6);
        // str at 0 "abcXYZ", match at 3 "XYZ": differ at first byte
        assert_eq!(lz_extend(buf, 0, 3, 0, 3), 0);
    }

    #[test]
    fn lz_extend_partial_match() {
        // "aabbaabb" — str=0 "aabbaabb", match=4 "aabb": match 4 then end.
        // max_len must respect BOTH str_pos and match_pos reaching the
        // buffer end (`min(buf.len() - str_pos, buf.len() - match_pos)` =
        // `min(8, 4)` = 4) -- a max_len of 8 here would violate that
        // precondition (`lz_extend` no longer clamps it; bt_matchfinder's
        // tests follow the analogous precondition).
        let buf = b"aabbaabb";
        assert_eq!(lz_extend(buf, 0, 4, 0, 4), 4);
    }

    #[test]
    fn lz_extend_respects_start_len() {
        // str=0 "abcabc", match=3 "abcabc": already known 3 match, extend to 6.
        // Buffer must be long enough so both str_pos+max and match_pos+max are in bounds.
        // str=0, match=3, max=6 → need buf.len() >= max(0+6, 3+6) = 9.
        let buf = b"abcabcabc"; // 9 bytes; buf[0..6]="abcabc", buf[3..9]="abcabc"
        assert_eq!(lz_extend(buf, 0, 3, 3, 6), 6);
    }

    #[test]
    fn lz_extend_respects_max_len() {
        let buf = b"aaaaaaaaaa"; // 10 identical bytes
        // max_len = 5 → cannot exceed 5
        assert_eq!(lz_extend(buf, 0, 0, 0, 5), 5);
    }
}
