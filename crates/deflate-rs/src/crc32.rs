//! Port of libdeflate `lib/crc32.c` + `lib/crc32_tables.h` — CRC-32 checksum.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/crc32.c
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/crc32_tables.h
//!
//! CRC-32 used in the gzip format (RFC 1952): polynomial 0xEDB88320 (reflected).
//!
//! ## Algorithm
//! Ports `crc32_slice8()`, the default scalar implementation in libdeflate.
//! Processes 8 bytes per inner-loop iteration using 8 × 256 precomputed tables
//! (`crc32_slice8_table`).
//!
//! `crc32_slice1()` (byte-at-a-time fallback) is also ported; it is used for
//! head/tail alignment around the 8-byte inner loop, exactly as in the C source.
//!
//! SIMD paths (x86 PCLMUL, ARM) are a LATER wave — not included here.
//!
//! ## Public API
//! `libdeflate_crc32(crc, buffer, len)` mirrors the C public function exactly:
//! - Pass `crc = 0` to start a new checksum.
//! - Pass `buffer = &[]` (empty slice) to retrieve the initial value (returns 0).
//! - The return value can be passed back in for incremental hashing.

use crate::crc32_tables::{CRC32_SLICE1_TABLE, CRC32_SLICE8_TABLE};
use crate::common::get_unaligned_le32;
use crate::cpu::{crc32_tier, Crc32Tier};

// ---------------------------------------------------------------------------
// crc32_slice8  — 8-bytes-at-a-time inner loop
// Mirrors the C `crc32_slice8()` in crc32.c.
// ---------------------------------------------------------------------------

/// Compute or update a CRC-32 starting from an initial state `crc`.
///
/// This is the raw state-passing form used internally; it does NOT invert `crc`
/// on entry/exit. Used by `crc32_slice8_aligned` and `crc32_slice1`.
///
/// Mirrors C `crc32_slice8(u32 crc, const u8 *p, size_t len)`.
#[inline]
pub(crate) fn crc32_slice8(mut crc: u32, p: &[u8]) -> u32 {
    let mut i = 0usize;
    let len = p.len();

    // -- Head: byte-at-a-time until 8-byte alignment (or end) --
    // C: `for (; ((uintptr_t)p & 7) && p != end; p++)`
    // We mimic alignment by processing bytes until `i` is on an 8-byte boundary.
    while i < len && (i & 7) != 0 {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[((crc as u8) ^ p[i]) as usize];
        i += 1;
    }

    // -- Main loop: 8 bytes at a time (slice-by-8) --
    // C: `v1 = le32_bswap(*(const u32 *)(p + 0))`
    // `le32_bswap(v)` is defined as `(v)` on LE and `bswap32(v)` on BE:
    //   common_defs.h: `#define le32_bswap(v) (CPU_IS_LITTLE_ENDIAN() ? (v) : bswap32(v))`
    // So on x86/LE this is just a plain u32 load (no swap).
    // In Rust: `get_unaligned_le32` = `u32::from_le_bytes(...)` = identical plain load on LE.
    let end8 = i + ((len - i) & !7usize);
    while i < end8 {
        // Plain LE u32 loads (mirrors C `le32_bswap` = identity on LE)
        let v1 = get_unaligned_le32(&p[i..]);
        let v2 = get_unaligned_le32(&p[i + 4..]);

        let xv1 = crc ^ v1;
        crc =
            CRC32_SLICE8_TABLE[0x700 + ( xv1        & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x600 + ((xv1 >>  8) & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x500 + ((xv1 >> 16) & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x400 + ((xv1 >> 24) & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x300 + ( v2         & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x200 + ((v2  >>  8) & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[0x100 + ((v2  >> 16) & 0xFF) as usize] ^
            CRC32_SLICE8_TABLE[      ( (v2  >> 24) & 0xFF) as usize];
        i += 8;
    }

    // -- Tail: byte-at-a-time for any remaining bytes --
    while i < len {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[((crc as u8) ^ p[i]) as usize];
        i += 1;
    }

    crc
}

// ---------------------------------------------------------------------------
// crc32_slice1 — byte-at-a-time over raw CRC state.
// Mirrors the C `crc32_slice1(u32 crc, const u8 *p, size_t len)`.
// Used by the PCLMULQDQ SIMD tier for short inputs (len < 16), exactly as the
// C template does (`return crc32_slice1(...)`), and as the head/tail of the
// scalar slice-by-8 loop.
// ---------------------------------------------------------------------------

/// Byte-at-a-time CRC-32 over raw (non-inverted) state `crc`.
/// Mirrors C `crc32_slice1`, which C declares `MAYBE_UNUSED`: only the x86
/// PCLMULQDQ tier calls it, so on other architectures it is dead by design.
#[inline]
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    expect(dead_code, reason = "C's MAYBE_UNUSED: only the x86 PCLMULQDQ tier calls it")
)]
pub(crate) fn crc32_slice1(mut crc: u32, p: &[u8]) -> u32 {
    for &b in p {
        crc = (crc >> 8) ^ CRC32_SLICE1_TABLE[((crc as u8) ^ b) as usize];
    }
    crc
}

// ---------------------------------------------------------------------------
// Public API — mirrors `libdeflate_crc32`
// ---------------------------------------------------------------------------

/// Compute or update a CRC-32 checksum.
///
/// Pass `crc = 0` to start a new checksum.  The return value is the updated
/// checksum, which can be passed back in for incremental hashing.
///
/// Passing an empty slice returns `0` (the initial value), mirroring the
/// `buffer == NULL` behaviour of `libdeflate_crc32`.
///
/// Mirrors `libdeflate_crc32(u32 crc, const void *buffer, size_t len)`.
pub fn crc32(crc: u32, data: &[u8]) -> u32 {
    // C: `if (p == NULL) return 0;` handles the NULL/initial-value sentinel.
    // In C, a non-NULL pointer with len==0 still goes through `~crc32_impl(~crc, p, 0)`
    // which returns `~(~crc)` = `crc` — correct for chaining through an empty slice.
    // An empty Rust slice corresponds to len==0 with a non-NULL pointer, so we must
    // NOT short-circuit it to 0 (that would break incremental chaining).
    // `crc32(0, &[])` still returns 0 correctly: `!crc32_impl(!0, &[])` = `!(0xFFFF_FFFF)` = 0.
    //
    // Runtime SIMD dispatch (mirrors libdeflate's `crc32_impl` function pointer):
    // the PCLMULQDQ tier is byte-identical to the scalar slice-by-8 path; the
    // dispatcher falls back to scalar when the feature is unavailable.
    !crc32_impl(!crc, data)
}

/// Compute the raw (pre/post-inversion handled by the caller) CRC-32 over
/// `data`, dispatching to the best available implementation. Mirrors the role
/// of libdeflate's `crc32_impl` indirect call.
#[inline]
fn crc32_impl(crc: u32, data: &[u8]) -> u32 {
    match crc32_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Crc32Tier::Pclmul => {
            // SAFETY: `crc32_tier()` returned `Pclmul` only after verifying
            // `is_x86_feature_detected!("pclmulqdq")` and `("sse4.1")`, so the
            // `#[target_feature(enable = "pclmul,sse4.1")]` precondition holds.
            unsafe { crate::crc32_pclmul::crc32_pclmul(crc, data) }
        }
        Crc32Tier::Scalar => crc32_slice8(crc, data),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::crc32;

    /// The canonical CRC-32 test vector: CRC-32 of "123456789" == 0xCBF43926.
    #[test]
    fn test_vector_123456789() {
        assert_eq!(crc32(0, b"123456789"), 0xCBF43926);
    }

    /// Empty data with crc=0 returns 0 (initial value, mirrors C NULL-buffer path).
    /// Empty data with non-zero crc returns that crc unchanged (chaining semantics).
    #[test]
    fn test_empty_returns_zero() {
        assert_eq!(crc32(0, b""), 0);
        // Chaining through empty slice must be identity (not reset to 0).
        assert_eq!(crc32(0xFFFF_FFFF, b""), 0xFFFF_FFFF);
        assert_eq!(crc32(0x1234_5678, b""), 0x1234_5678);
    }

    /// Incremental == one-shot: split at every position must agree.
    #[test]
    fn test_incremental_equals_oneshot() {
        let data = b"hello, gzip world! 0123456789";
        let expected = crc32(0, data);
        for split in 0..=data.len() {
            let left  = crc32(0, &data[..split]);
            let right = crc32(left, &data[split..]);
            assert_eq!(right, expected, "split@{split}: incremental != one-shot");
        }
    }

    /// CRC-32 of a single zero byte is 0xD202EF8D (well-known value).
    #[test]
    fn test_single_zero_byte() {
        assert_eq!(crc32(0, &[0u8]), 0xD202EF8D);
    }

    /// All-zeros and all-0xFF buffers — exercise boundary paths.
    #[test]
    fn test_boundary_buffers() {
        // CRC-32 of 32 zero bytes (exercises the 8-byte main loop)
        let zeros = [0u8; 32];
        let r = crc32(0, &zeros);
        // Verify incremental match
        assert_eq!(crc32(crc32(0, &zeros[..16]), &zeros[16..]), r);

        // CRC-32 of 7 bytes (head/tail only, no main loop)
        let short = b"abcdefg";
        assert_eq!(crc32(crc32(0, &short[..3]), &short[3..]), crc32(0, short));
    }
}
