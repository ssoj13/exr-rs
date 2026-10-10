//! Port of libdeflate `common_defs.h` + `lib/lib_common.h` — portable definitions.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/common_defs.h
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/lib_common.h
//!
//! # What is ported (portable symbols only)
//! - LE/BE unaligned load/store helpers (`get_unaligned_le*`, `put_unaligned_le*`, etc.)
//! - Bit-scan helpers (`bsr32`, `bsf32`, `bsr64`, `bsf64`, `bsrw`, `bsfw`)
//! - Arithmetic helpers (`div_round_up`, `align_up`, `round_up`)
//! - Word-size constant (`WORDBYTES`)
//!
//! # N/A in Rust (documented here, not ported)
//! - `forceinline` / `MAYBE_UNUSED` / `NORETURN` / `restrict` — compiler attributes,
//!   not needed; Rust's `#[inline]` is sufficient.
//! - `LIBDEFLATEAPI` / `LIBDEFLATE_EXPORT_SYM` — DLL export decorators, no FFI here.
//! - `LIBDEFLATE_ALIGN_STACK` — i386 ABI workaround, irrelevant in Rust.
//! - `CPU_IS_LITTLE_ENDIAN()` macro — Rust's `cfg!(target_endian)` covers this; we
//!   use `u16/u32/u64::from_le_bytes` / `to_le_bytes` which are always correct.
//! - `UNALIGNED_ACCESS_IS_FAST` — the `from_le_bytes` path is always safe and the
//!   compiler optimises it identically on platforms where unaligned access is fast.
//! - `bswap16/32/64` — handled internally by `u*::swap_bytes()`; not exposed separately
//!   because libdeflate's public callers only use them via the LE/BE load helpers.
//! - `prefetchr` / `prefetchw` — Rust does not expose prefetch in stable; N/A for
//!   correctness, only a hint.
//! - `_aligned_attribute` / `_target_attribute` — attribute macros, handled by Rust
//!   proc-macro / `#[target_feature]` per module; not a crate-wide concern.
//! - `rbit32` — ARM-only bit-reverse; no portable fallback in C either; skipped.
//! - `FREESTANDING` / `memset` / `memcpy` / `memmove` / `memcmp` shims — irrelevant;
//!   Rust targets always have an allocator and std mem ops.
//! - `LIBDEFLATE_ENABLE_ASSERTIONS` / `libdeflate_assertion_failed` — mapped to
//!   `debug_assert!` / `panic!` in Rust naturally.
//! - `STATIC_ASSERT` — Rust has `const { assert!(...) }`.
//! - `ARRAY_LEN` — Rust slices/arrays expose `.len()` directly.
//! - `GCC_PREREQ` / `CLANG_PREREQ` / `MSVC_PREREQ` — compiler-version guards; N/A.
//! - `load_word_unaligned` / `store_word_unaligned` (`machine_word_t`) — word-size
//!   variants; callers use the explicitly-sized forms below.

/// Number of bytes in a machine word (`usize`), mirroring `WORDBYTES`.
pub const WORDBYTES: usize = core::mem::size_of::<usize>();

// ============================================================================
// Unaligned LE/BE load helpers
// Mirrors: get_unaligned_le16 / get_unaligned_le32 / get_unaligned_le64
//          get_unaligned_be16 / get_unaligned_be32
// Using from_le_bytes / from_be_bytes so no unsafe is needed — the compiler
// emits a single load instruction on LE platforms with fast unaligned access.
//
// These are `pub(crate)` internal codec primitives: they slice `p[..N]` and so
// PANIC on a slice shorter than N. Every in-crate call site already establishes
// the length before calling, so the panic is unreachable there — keeping them
// crate-private prevents an external caller from tripping the panic on a short
// slice (there is no external user; verified by grep).
// ============================================================================

/// Load a little-endian `u16` from an unaligned byte slice at offset 0.
/// Mirrors `get_unaligned_le16(p)`.
#[inline(always)]
pub(crate) fn get_unaligned_le16(p: &[u8]) -> u16 {
    u16::from_le_bytes(p[..2].try_into().unwrap())
}

/// Load a little-endian `u32` from an unaligned byte slice at offset 0.
/// Mirrors `get_unaligned_le32(p)`.
#[inline(always)]
pub(crate) fn get_unaligned_le32(p: &[u8]) -> u32 {
    u32::from_le_bytes(p[..4].try_into().unwrap())
}

/// Load a little-endian `u64` from an unaligned byte slice at offset 0.
/// Mirrors `get_unaligned_le64(p)`.
#[inline(always)]
pub(crate) fn get_unaligned_le64(p: &[u8]) -> u64 {
    u64::from_le_bytes(p[..8].try_into().unwrap())
}

/// Load a big-endian `u16` from an unaligned byte slice at offset 0.
/// Mirrors `get_unaligned_be16(p)`.
#[inline(always)]
pub(crate) fn get_unaligned_be16(p: &[u8]) -> u16 {
    u16::from_be_bytes(p[..2].try_into().unwrap())
}

/// Load a big-endian `u32` from an unaligned byte slice at offset 0.
/// Mirrors `get_unaligned_be32(p)`.
#[inline(always)]
pub(crate) fn get_unaligned_be32(p: &[u8]) -> u32 {
    u32::from_be_bytes(p[..4].try_into().unwrap())
}

// ============================================================================
// Unaligned LE store helpers
// Mirrors: put_unaligned_le16 / put_unaligned_le32 / put_unaligned_le64
//          put_unaligned_be16 / put_unaligned_be32
// ============================================================================

/// Store a `u16` as little-endian bytes into slice at offset 0.
/// Mirrors `put_unaligned_le16(v, p)`.
#[inline(always)]
pub(crate) fn put_unaligned_le16(v: u16, p: &mut [u8]) {
    p[..2].copy_from_slice(&v.to_le_bytes());
}

/// Store a `u32` as little-endian bytes into slice at offset 0.
/// Mirrors `put_unaligned_le32(v, p)`.
#[inline(always)]
pub(crate) fn put_unaligned_le32(v: u32, p: &mut [u8]) {
    p[..4].copy_from_slice(&v.to_le_bytes());
}

/// Store a `u64` as little-endian bytes into slice at offset 0.
/// Mirrors `put_unaligned_le64(v, p)`. Used by the decompressor fastloop's
/// word-at-a-time match copy (`deflate_decompress.rs`).
#[inline(always)]
pub(crate) fn put_unaligned_le64(v: u64, p: &mut [u8]) {
    p[..8].copy_from_slice(&v.to_le_bytes());
}

/// Store a `u16` as big-endian bytes into slice at offset 0.
/// Mirrors `put_unaligned_be16(v, p)`.
#[inline(always)]
pub(crate) fn put_unaligned_be16(v: u16, p: &mut [u8]) {
    p[..2].copy_from_slice(&v.to_be_bytes());
}

/// Store a `u32` as big-endian bytes into slice at offset 0.
/// Mirrors `put_unaligned_be32(v, p)`.
#[inline(always)]
pub(crate) fn put_unaligned_be32(v: u32, p: &mut [u8]) {
    p[..4].copy_from_slice(&v.to_be_bytes());
}

// ============================================================================
// Bit-scan helpers
// Mirrors: bsr32, bsr64, bsrw, bsf32, bsf64, bsfw
//
// C semantics (INPUT MUST BE NONZERO — same precondition in Rust):
//   bsr32(v) = 31 - __builtin_clz(v)    →  31 - v.leading_zeros()
//   bsf32(v) = __builtin_ctz(v)         →  v.trailing_zeros()
// ============================================================================

/// Bit Scan Reverse for `u32`: index of the most-significant set bit (0-based from LSB).
/// Input must be nonzero. Mirrors `bsr32(v)`.
#[inline(always)]
pub fn bsr32(v: u32) -> u32 {
    debug_assert!(v != 0, "bsr32: input must be nonzero");
    31 - v.leading_zeros()
}

/// Bit Scan Reverse for `u64`: index of the most-significant set bit (0-based from LSB).
/// Input must be nonzero. Mirrors `bsr64(v)`.
#[inline(always)]
pub fn bsr64(v: u64) -> u32 {
    debug_assert!(v != 0, "bsr64: input must be nonzero");
    63 - v.leading_zeros()
}

/// Bit Scan Reverse for machine word (`usize`). Mirrors `bsrw(v)`.
#[inline(always)]
pub fn bsrw(v: usize) -> u32 {
    debug_assert!(v != 0, "bsrw: input must be nonzero");
    (usize::BITS - 1) - v.leading_zeros()
}

/// Bit Scan Forward for `u32`: index of the least-significant set bit (0-based from LSB).
/// Input must be nonzero. Mirrors `bsf32(v)`.
#[inline(always)]
pub fn bsf32(v: u32) -> u32 {
    debug_assert!(v != 0, "bsf32: input must be nonzero");
    v.trailing_zeros()
}

/// Bit Scan Forward for `u64`: index of the least-significant set bit (0-based from LSB).
/// Input must be nonzero. Mirrors `bsf64(v)`.
#[inline(always)]
pub fn bsf64(v: u64) -> u32 {
    debug_assert!(v != 0, "bsf64: input must be nonzero");
    v.trailing_zeros()
}

/// Bit Scan Forward for machine word (`usize`). Mirrors `bsfw(v)`.
#[inline(always)]
pub fn bsfw(v: usize) -> u32 {
    debug_assert!(v != 0, "bsfw: input must be nonzero");
    v.trailing_zeros()
}

// ============================================================================
// Arithmetic helpers
// Mirrors: DIV_ROUND_UP, ALIGN (same formula as libdeflate's ALIGN macro),
//          ROUND_UP, MIN, MAX
// Note: MIN/MAX are not exposed as functions — use Rust's built-in
// std::cmp::min / std::cmp::max or the `.min()` / `.max()` methods instead.
// ============================================================================

/// Integer ceil division. Mirrors `DIV_ROUND_UP(n, d)`.
#[inline(always)]
pub const fn div_round_up(n: usize, d: usize) -> usize {
    n.div_ceil(d)
}

/// Align `n` up to the nearest multiple of `a` (which must be a power of two).
/// Mirrors `ALIGN(n, a)` — uses the bitmask formula `(n + a - 1) & !(a - 1)`.
#[inline(always)]
pub const fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

/// Round `n` up to the nearest multiple of `d`. Mirrors `ROUND_UP(n, d)`.
/// Unlike `align_up`, `d` need not be a power of two.
#[inline(always)]
pub const fn round_up(n: usize, d: usize) -> usize {
    d * n.div_ceil(d)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- LE load/store round-trips ---

    #[test]
    fn le16_roundtrip() {
        let v: u16 = 0xABCD;
        let mut buf = [0u8; 2];
        put_unaligned_le16(v, &mut buf);
        assert_eq!(get_unaligned_le16(&buf), v);
        // Also verify byte layout is truly little-endian
        assert_eq!(buf, [0xCD, 0xAB]);
    }

    #[test]
    fn le32_roundtrip() {
        let v: u32 = 0x12345678;
        let mut buf = [0u8; 4];
        put_unaligned_le32(v, &mut buf);
        assert_eq!(get_unaligned_le32(&buf), v);
        assert_eq!(buf, [0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn le64_roundtrip() {
        let v: u64 = 0xDEADBEEF_CAFEBABE;
        let mut buf = [0u8; 8];
        put_unaligned_le64(v, &mut buf);
        assert_eq!(get_unaligned_le64(&buf), v);
        assert_eq!(buf, [0xBE, 0xBA, 0xFE, 0xCA, 0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn be16_roundtrip() {
        let v: u16 = 0xABCD;
        let mut buf = [0u8; 2];
        put_unaligned_be16(v, &mut buf);
        assert_eq!(get_unaligned_be16(&buf), v);
        assert_eq!(buf, [0xAB, 0xCD]);
    }

    #[test]
    fn be32_roundtrip() {
        let v: u32 = 0x12345678;
        let mut buf = [0u8; 4];
        put_unaligned_be32(v, &mut buf);
        assert_eq!(get_unaligned_be32(&buf), v);
        assert_eq!(buf, [0x12, 0x34, 0x56, 0x78]);
    }

    // --- Bit-scan known values ---

    #[test]
    fn bsr32_known() {
        // 1 << 0  → MSB at bit 0
        assert_eq!(bsr32(1), 0);
        // 1 << 15 → MSB at bit 15
        assert_eq!(bsr32(1 << 15), 15);
        // 0x80000000 → MSB at bit 31
        assert_eq!(bsr32(0x8000_0000), 31);
        // 0b1010 → MSB at bit 3
        assert_eq!(bsr32(0b1010), 3);
    }

    #[test]
    fn bsf32_known() {
        // lowest set bit of 8 (0b1000) is bit 3
        assert_eq!(bsf32(8), 3);
        assert_eq!(bsf32(1), 0);
        assert_eq!(bsf32(0x8000_0000), 31);
        // 0b1100 → LSB at bit 2
        assert_eq!(bsf32(0b1100), 2);
    }

    #[test]
    fn bsr64_known() {
        assert_eq!(bsr64(1), 0);
        assert_eq!(bsr64(1 << 32), 32);
        assert_eq!(bsr64(0x8000_0000_0000_0000), 63);
    }

    #[test]
    fn bsf64_known() {
        assert_eq!(bsf64(1), 0);
        assert_eq!(bsf64(0x8000_0000_0000_0000), 63);
        assert_eq!(bsf64(0x0000_0001_0000_0000), 32);
    }

    // --- Arithmetic helpers ---

    #[test]
    fn div_round_up_known() {
        assert_eq!(div_round_up(5, 3), 2);
        assert_eq!(div_round_up(6, 3), 2);
        assert_eq!(div_round_up(0, 3), 0);
        assert_eq!(div_round_up(1, 1), 1);
    }

    #[test]
    fn align_up_known() {
        // align to 4
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        // align to 1 is a no-op
        assert_eq!(align_up(7, 1), 7);
    }

    #[test]
    fn round_up_known() {
        assert_eq!(round_up(5, 3), 6);
        assert_eq!(round_up(6, 3), 6);
        assert_eq!(round_up(0, 5), 0);
        assert_eq!(round_up(1, 5), 5);
    }
}
