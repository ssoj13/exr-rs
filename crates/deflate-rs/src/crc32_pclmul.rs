//! PCLMULQDQ (carryless-multiply) CRC-32 folding — x86 SIMD tier.
//!
//! Port of libdeflate `lib/x86/crc32_pclmul_template.h` instantiated with the
//! **VL=16, USE_AVX512=0** parameters (the `crc32_x86_pclmulqdq` variant from
//! `lib/x86/crc32_impl.h`, target `pclmul,sse4.1`).
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/crc32_pclmul_template.h>
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/crc32_impl.h>
//!
//! The folding constants come verbatim from `lib/crc32_multipliers.h`.
//!
//! # Why only the VL=16 tier
//! libdeflate also ships VL=32 (VPCLMULQDQ/AVX2) and VL=64 (AVX-512) variants.
//! Those are *performance-only* refinements: they compute the **identical**
//! CRC-32 value as this VL=16 tier (and as the scalar slice-by-8 path). Porting
//! the 128-bit tier alone gives us the SIMD speedup while keeping the `unsafe`
//! surface minimal; the scalar path in `crc32.rs` remains the source of truth
//! and the dispatcher (`cpu::crc32_tier`) falls back to it when PCLMULQDQ/SSE4.1
//! are unavailable.
//!
//! # Correctness
//! Output is byte-identical to `crc32::crc32` (verified by `tests/compare_simd.rs`
//! and the libdeflate parity gate). For `len < 16` this tier delegates to the
//! scalar slice-by-1 path exactly as the C template does (`return crc32_slice1`).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Folding constants — from lib/crc32_multipliers.h (verbatim).
// All are 64-bit literals as pclmulqdq consumes 64-bit halves; only the low 32
// bits are nonzero since this is CRC-32.
// ---------------------------------------------------------------------------

const CRC32_X159_MODG: i64 = 0xae689191; // x^159 mod G(x)
const CRC32_X95_MODG: i64 = 0xccaa009e; // x^95 mod G(x)
const CRC32_X223_MODG: i64 = 0x81256527; // x^223 mod G(x)
const CRC32_X287_MODG: i64 = 0xf1da05aa; // x^287 mod G(x)
const CRC32_X479_MODG: i64 = 0x1d9513d7; // x^479 mod G(x)
const CRC32_X543_MODG: i64 = 0x8f352d95; // x^543 mod G(x)
const CRC32_X991_MODG: i64 = 0x910eeec1; // x^991 mod G(x)
const CRC32_X1055_MODG: i64 = 0x33fff533; // x^1055 mod G(x)

const CRC32_BARRETT_CONSTANT_1: i64 = 0xb4e5b025f7011641u64 as i64; // floor(x^95 / G(x))
const CRC32_BARRETT_CONSTANT_2: i64 = 0x00000001db710641; // G(x)

/// `shift_tab[48]` from `crc32_impl.h`.
/// `pshufb(x, shift_tab[len..len+15])` left-shifts `x` by `16-len` bytes;
/// `pshufb(x, shift_tab[len+16..len+31])` right-shifts `x` by `len` bytes.
const SHIFT_TAB: [u8; 48] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

/// Fold the 128-bit `src` into `dst` using the two 64-bit multipliers in `mults`.
/// Mirrors `fold_vec128`.
///
/// # Safety
/// Requires the `pclmul,sse4.1` target features (carryless multiply + SSE).
/// Operates entirely on register values; no memory is touched.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn fold_vec128(src: __m128i, dst: __m128i, mults: __m128i) -> __m128i {
    // No `unsafe` block is needed: these are register-only intrinsics with no
    // memory access, and since Rust 1.87 such intrinsics are *safe* to call
    // inside a fn whose #[target_feature] set implies theirs. The `unsafe fn`
    // marker stays because the caller must still guarantee pclmulqdq + sse4.1.
    let a = _mm_xor_si128(dst, _mm_clmulepi64_si128(src, mults, 0x00));
    _mm_xor_si128(a, _mm_clmulepi64_si128(src, mults, 0x11))
}

/// Fold the trailing `1..=15` bytes at `p[..len]` into the running polynomial
/// `x`.  Mirrors `fold_lessthan16bytes`.  Assumes `p + len - 16` is in-bounds,
/// i.e. there are at least 16 bytes available ending at `p[len]` (the caller —
/// here only reached when `total_len >= 16` — guarantees this by reading the
/// last 16 bytes of the buffer).
///
/// # Safety
/// Requires `pclmul,sse4.1`.  `p` must point to at least `len` readable bytes
/// AND the 16-byte window `p[len-16 .. len]` must be in-bounds (the C contract:
/// "p + len - 16 is in-bounds"). `len` must be in `1..=15`.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn fold_lessthan16bytes(
    x: __m128i,
    p: *const u8,
    len: usize,
    mults_128b: __m128i,
) -> __m128i {
    // SAFETY: SHIFT_TAB has 48 bytes; `&SHIFT_TAB[len]` and `&SHIFT_TAB[len+16]`
    // are valid 16-byte loads for len in 1..=15 (max index 31, +16 bytes = 47).
    // The data load `p + len - 16` is in-bounds by the caller's contract.
    unsafe {
        let lshift = _mm_loadu_si128(SHIFT_TAB.as_ptr().add(len) as *const __m128i);
        let rshift = _mm_loadu_si128(SHIFT_TAB.as_ptr().add(len + 16) as *const __m128i);

        // x0 = x left-shifted by (16 - len) bytes.
        let x0 = _mm_shuffle_epi8(x, lshift);

        // x1 = (x right-shifted by len bytes) blended with the final 16 data
        // bytes, selecting per byte by the msb of rshift.
        let data = _mm_loadu_si128(p.add(len).sub(16) as *const __m128i);
        let x1 = _mm_blendv_epi8(_mm_shuffle_epi8(x, rshift), data, rshift);

        fold_vec128(x0, x1, mults_128b)
    }
}

/// PCLMULQDQ CRC-32 (VL=16). Mirrors `crc32_x86` with VL=16, USE_AVX512=0.
///
/// `crc` is the raw running CRC (already inverted by the public wrapper).
/// Returns the updated raw CRC. For `len < 16`, delegates to the scalar
/// slice-by-1 path (identical to the C `return crc32_slice1(...)`).
///
/// # Safety
/// Requires the `pclmul,sse4.1` target features. `p` must point to `len`
/// readable bytes. Reached only via `cpu::crc32_tier() == Pclmul`, which checks
/// `is_x86_feature_detected!` for both features.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub(crate) unsafe fn crc32_pclmul(crc: u32, p: &[u8]) -> u32 {
    let len = p.len();
    // C: lengths below a 128-bit vector go straight to the scalar path.
    if len < 16 {
        return crate::crc32::crc32_slice1(crc, p);
    }

    // SAFETY: feature-gated by #[target_feature] + the runtime dispatch check.
    // All loads below stay within p[..len]; offsets are bounded by the same
    // length arithmetic libdeflate uses.
    unsafe {
        let mut ptr = p.as_ptr();
        let mut len = len;

        let mults_1v = _mm_set_epi64x(CRC32_X95_MODG, CRC32_X159_MODG);
        let mults_2v = _mm_set_epi64x(CRC32_X223_MODG, CRC32_X287_MODG);
        let mults_4v = _mm_set_epi64x(CRC32_X479_MODG, CRC32_X543_MODG);
        let mults_8v = _mm_set_epi64x(CRC32_X991_MODG, CRC32_X1055_MODG);
        let mults_128b = _mm_set_epi64x(CRC32_X95_MODG, CRC32_X159_MODG);
        let barrett =
            _mm_set_epi64x(CRC32_BARRETT_CONSTANT_2, CRC32_BARRETT_CONSTANT_1);

        // For VL=16, M128I_TO_VEC and VLOADU/VXOR are the 128-bit ops directly.
        let x0 = _mm_cvtsi32_si128(crc as i32);

        let (mut v0, mut v1, mut v2, mut v3);
        let (mut v4, mut v5, mut v6, mut v7);

        // The control flow mirrors the C template's goto structure with nested
        // helper closures replaced by explicit blocks.
        //
        // Tail reductions are factored out so each `goto` target becomes a
        // labelled-block-style early entry.
        const VL: usize = 16;

        // --- Entry decision: len < 8*VL ? ---
        if len < 8 * VL {
            if len < VL {
                // 16 <= len < VL is impossible for VL=16; len<16 already handled.
                // (For VL=16, len < VL == len < 16, so this branch is unreachable
                // here. Kept structurally to mirror C; do scalar to be safe.)
                return crate::crc32::crc32_slice1(_mm_cvtsi128_si32(x0) as u32, p);
            }
            v0 = _mm_xor_si128(_mm_loadu_si128(ptr as *const __m128i), x0);
            if len < 2 * VL {
                ptr = ptr.add(VL);
                // goto less_than_vl_remaining
                return reduce_tail(v0, ptr, len, mults_128b, barrett);
            }
            v1 = _mm_loadu_si128(ptr.add(VL) as *const __m128i);
            if len < 4 * VL {
                ptr = ptr.add(2 * VL);
                // goto less_than_2vl_remaining
                v0 = fold_vec128(v0, v1, mults_1v);
                if len & VL != 0 {
                    v0 = fold_vec128(v0, _mm_loadu_si128(ptr as *const __m128i), mults_1v);
                    ptr = ptr.add(VL);
                }
                return reduce_tail(v0, ptr, len, mults_128b, barrett);
            }
            v2 = _mm_loadu_si128(ptr.add(2 * VL) as *const __m128i);
            v3 = _mm_loadu_si128(ptr.add(3 * VL) as *const __m128i);
            ptr = ptr.add(4 * VL);
        } else {
            // len >= 8*VL. (We skip the >64KiB alignment optimisation: it only
            // changes performance, never the result. Plain misaligned loads are
            // correct and, per the C comment, just as fast on modern x86.)
            v0 = _mm_xor_si128(_mm_loadu_si128(ptr as *const __m128i), x0);
            v1 = _mm_loadu_si128(ptr.add(VL) as *const __m128i);
            v2 = _mm_loadu_si128(ptr.add(2 * VL) as *const __m128i);
            v3 = _mm_loadu_si128(ptr.add(3 * VL) as *const __m128i);
            v4 = _mm_loadu_si128(ptr.add(4 * VL) as *const __m128i);
            v5 = _mm_loadu_si128(ptr.add(5 * VL) as *const __m128i);
            v6 = _mm_loadu_si128(ptr.add(6 * VL) as *const __m128i);
            v7 = _mm_loadu_si128(ptr.add(7 * VL) as *const __m128i);
            ptr = ptr.add(8 * VL);

            // Main loop: 8*VL bytes per iteration.
            while len >= 16 * VL {
                v0 = fold_vec128(v0, _mm_loadu_si128(ptr as *const __m128i), mults_8v);
                v1 = fold_vec128(v1, _mm_loadu_si128(ptr.add(VL) as *const __m128i), mults_8v);
                v2 = fold_vec128(v2, _mm_loadu_si128(ptr.add(2 * VL) as *const __m128i), mults_8v);
                v3 = fold_vec128(v3, _mm_loadu_si128(ptr.add(3 * VL) as *const __m128i), mults_8v);
                v4 = fold_vec128(v4, _mm_loadu_si128(ptr.add(4 * VL) as *const __m128i), mults_8v);
                v5 = fold_vec128(v5, _mm_loadu_si128(ptr.add(5 * VL) as *const __m128i), mults_8v);
                v6 = fold_vec128(v6, _mm_loadu_si128(ptr.add(6 * VL) as *const __m128i), mults_8v);
                v7 = fold_vec128(v7, _mm_loadu_si128(ptr.add(7 * VL) as *const __m128i), mults_8v);
                ptr = ptr.add(8 * VL);
                len -= 8 * VL;
            }

            // Fewer than 8*VL remain: fold the 8 accumulators down to 4.
            v0 = fold_vec128(v0, v4, mults_4v);
            v1 = fold_vec128(v1, v5, mults_4v);
            v2 = fold_vec128(v2, v6, mults_4v);
            v3 = fold_vec128(v3, v7, mults_4v);
            if len & (4 * VL) != 0 {
                v0 = fold_vec128(v0, _mm_loadu_si128(ptr as *const __m128i), mults_4v);
                v1 = fold_vec128(v1, _mm_loadu_si128(ptr.add(VL) as *const __m128i), mults_4v);
                v2 = fold_vec128(v2, _mm_loadu_si128(ptr.add(2 * VL) as *const __m128i), mults_4v);
                v3 = fold_vec128(v3, _mm_loadu_si128(ptr.add(3 * VL) as *const __m128i), mults_4v);
                ptr = ptr.add(4 * VL);
            }
        }

        // Fewer than 4*VL bytes remain: fold 4 accumulators down to 2.
        v0 = fold_vec128(v0, v2, mults_2v);
        v1 = fold_vec128(v1, v3, mults_2v);
        if len & (2 * VL) != 0 {
            v0 = fold_vec128(v0, _mm_loadu_si128(ptr as *const __m128i), mults_2v);
            v1 = fold_vec128(v1, _mm_loadu_si128(ptr.add(VL) as *const __m128i), mults_2v);
            ptr = ptr.add(2 * VL);
        }

        // less_than_2vl_remaining: fold 2 accumulators to 1.
        v0 = fold_vec128(v0, v1, mults_1v);
        if len & VL != 0 {
            v0 = fold_vec128(v0, _mm_loadu_si128(ptr as *const __m128i), mults_1v);
            ptr = ptr.add(VL);
        }

        reduce_tail(v0, ptr, len, mults_128b, barrett)
    }
}

/// Common reduction path: takes the final 128-bit accumulator `x0` (held in
/// `v0`), folds in the trailing `len & 15` bytes, then reduces 128→32 bits via
/// the Barrett constants and returns the CRC.  This corresponds to the
/// `less_than_vl_remaining` … return tail of `crc32_x86` for VL=16 (where the
/// 256/512-bit reduction blocks vanish and `x0 = v0`).
///
/// # Safety
/// Requires `pclmul,sse4.1`. `ptr` must point at the start of the remaining
/// `len & 15` bytes, and the buffer must have ≥16 bytes ending at `ptr[len&15]`
/// (guaranteed because this tier only runs for total length ≥ 16).
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn reduce_tail(
    x0: __m128i,
    ptr: *const u8,
    len: usize,
    mults_128b: __m128i,
    barrett: __m128i,
) -> u32 {
    // SAFETY: feature-gated; see callers. fold_lessthan16bytes' window
    // requirement holds for total length >= 16.
    unsafe {
        let mut x0 = x0;
        let rem = len & 15;
        if rem != 0 {
            x0 = fold_lessthan16bytes(x0, ptr, rem, mults_128b);
        }
        // Multiply by x^32 then reduce mod G via Barrett (n=32, LSB_CRC=1).
        x0 = _mm_xor_si128(
            _mm_clmulepi64_si128(x0, mults_128b, 0x10),
            _mm_bsrli_si128(x0, 8),
        );
        let mut x1 = _mm_clmulepi64_si128(x0, barrett, 0x00);
        x1 = _mm_clmulepi64_si128(x1, barrett, 0x10);
        x0 = _mm_xor_si128(x0, x1);
        _mm_extract_epi32(x0, 2) as u32
    }
}

// ===========================================================================
// Tests: the PCLMULQDQ tier must be byte-identical to the scalar slice-by-8
// path over varied + edge-case data (all lengths that exercise every branch:
// < 16, the 16..8*VL ladder, the 8*VL main loop, and every tail mask).
// ===========================================================================
#[cfg(test)]
mod tests {
    use crate::crc32::crc32 as crc32_public;

    /// Raw scalar reference: `!slice8(!crc)` via the public wrapper with the
    /// SIMD tier forced off is awkward (env is process-global), so we compare
    /// the PCLMULQDQ tier's RAW output against the scalar RAW slice-by-8.
    fn scalar_raw(crc: u32, p: &[u8]) -> u32 {
        crate::crc32::crc32_slice8(crc, p)
    }

    #[test]
    fn pclmul_matches_scalar_varied() {
        if !std::is_x86_feature_detected!("pclmulqdq")
            || !std::is_x86_feature_detected!("sse4.1")
        {
            eprintln!("skip: pclmulqdq/sse4.1 not available on this host");
            return;
        }
        // xorshift32 pseudo-random data, then test every prefix length.
        let mut x: u32 = 0x9E37_79B9;
        let mut buf = vec![0u8; 4100];
        for b in buf.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x & 0xFF) as u8;
        }
        for &init in &[0u32, 0xFFFF_FFFF, 0x1234_5678] {
            for len in 0..=buf.len() {
                let data = &buf[..len];
                // SAFETY: feature presence checked above.
                let simd = unsafe { super::crc32_pclmul(init, data) };
                assert_eq!(simd, scalar_raw(init, data), "len={len} init={init:#x}");
            }
        }
    }

    #[test]
    fn public_crc32_known_vector_under_active_tier() {
        // Whatever tier the dispatcher picks, the public API must still match
        // the canonical CRC-32 of "123456789".
        assert_eq!(crc32_public(0, b"123456789"), 0xCBF43926);
    }
}
