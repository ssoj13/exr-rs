//! Port of libdeflate `lib/gzip_compress.c` — compress with a gzip wrapper.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/gzip_compress.c
//!
//! Wraps a raw DEFLATE stream with a 10-byte gzip header and an 8-byte footer
//! (CRC32 + ISIZE), forming a complete RFC-1952 gzip member.  This is what
//! `libdeflate_gzip_compress` produces.
//!
//! ## Header layout (byte-exact to libdeflate)
//! | Offset | Field  | Value                                          |
//! |--------|--------|------------------------------------------------|
//! | 0      | ID1    | 0x1F                                           |
//! | 1      | ID2    | 0x8B                                           |
//! | 2      | CM     | 8 (DEFLATE)                                    |
//! | 3      | FLG    | 0 (no optional fields)                         |
//! | 4–7   | MTIME  | 0 (unavailable), LE u32                        |
//! | 8      | XFL    | 0, 0x02 (slowest, lvl≥18), or 0x04 (fastest) |
//! | 9      | OS     | 0xFF (unknown)                                 |
//!
//! ## Footer
//! CRC-32 of the uncompressed data (LE u32) followed by ISIZE = uncompressed
//! size mod 2³² (LE u32).

use crate::common::{put_unaligned_le32};
use crate::crc32::crc32;
use crate::deflate_compress::{deflate_compress_bound, Compressor};
use crate::gzip_constants::{
    GZIP_CM_DEFLATE, GZIP_ID1, GZIP_ID2, GZIP_MIN_OVERHEAD, GZIP_MTIME_UNAVAILABLE,
    GZIP_OS_UNKNOWN, GZIP_XFL_FASTEST_COMPRESSION, GZIP_XFL_SLOWEST_COMPRESSION,
};

/// Compress `input` into `out` as a complete gzip member (header + DEFLATE body + footer).
///
/// Returns the number of bytes written to `out`, or **0** if the output buffer is
/// too small (mirrors the `return 0` contract of `libdeflate_gzip_compress`).
///
/// Mirrors `libdeflate_gzip_compress`.
pub fn gzip_compress(c: &mut Compressor, input: &[u8], out: &mut [u8]) -> usize {
    if out.len() <= GZIP_MIN_OVERHEAD {
        return 0;
    }

    let mut pos = 0usize;

    // --- 10-byte gzip header ---
    out[pos] = GZIP_ID1;           pos += 1;
    out[pos] = GZIP_ID2;           pos += 1;
    out[pos] = GZIP_CM_DEFLATE;    pos += 1;
    out[pos] = 0;                  pos += 1; // FLG: no optional fields
    put_unaligned_le32(GZIP_MTIME_UNAVAILABLE, &mut out[pos..pos + 4]);
    pos += 4;

    // XFL: hint the decompressor about the effort level used.
    let level = c.compression_level();
    let xfl: u8 = if level < 2 {
        GZIP_XFL_FASTEST_COMPRESSION
    } else if level >= 8 {
        GZIP_XFL_SLOWEST_COMPRESSION
    } else {
        0
    };
    out[pos] = xfl;                pos += 1;
    out[pos] = GZIP_OS_UNKNOWN;    pos += 1;

    // --- Raw DEFLATE body (between header and the reserved 8-byte footer) ---
    // Pre-compute end bound before taking a mutable borrow (borrow checker).
    let deflate_end = out.len() - 8;
    let deflate_out = &mut out[pos..deflate_end];
    let deflate_size = c.deflate_compress(input, deflate_out);
    if deflate_size == 0 {
        return 0;
    }
    pos += deflate_size;

    // --- 8-byte footer: CRC32 (LE u32) + ISIZE (LE u32) ---
    put_unaligned_le32(crc32(0, input), &mut out[pos..pos + 4]);
    pos += 4;
    put_unaligned_le32(input.len() as u32, &mut out[pos..pos + 4]);
    pos += 4;

    pos
}

/// Return the worst-case output size for compressing `n` bytes with [`gzip_compress`].
///
/// Mirrors `libdeflate_gzip_compress_bound`:
/// ```text
/// GZIP_MIN_OVERHEAD + libdeflate_deflate_compress_bound(n)
/// ```
pub fn gzip_compress_bound(n: usize) -> usize {
    GZIP_MIN_OVERHEAD + deflate_compress_bound(n)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate_compress::Compressor;
    use crate::gzip_decompress::gzip_decompress;

    /// Round-trip test for all levels and a variety of inputs.
    #[test]
    fn test_round_trip_all_levels() {
        let inputs: &[&[u8]] = &[
            b"",
            b"a",
            b"hello, gzip!",
            &[0u8; 256],
            &[0xABu8; 1024],
            b"the quick brown fox jumps over the lazy dog",
        ];
        for &input in inputs {
            for level in 1..=12_i32 {
                let mut c = Compressor::new(level).expect("valid level");
                let mut out = vec![0u8; gzip_compress_bound(input.len())];
                let n = gzip_compress(&mut c, input, &mut out);
                assert!(n > 0, "level={level} input_len={}: compress returned 0", input.len());

                let mut recovered = vec![0u8; input.len()];
                let r = gzip_decompress(&out[..n], &mut recovered);
                assert!(
                    r.is_ok(),
                    "level={level} input_len={}: decompress failed: {:?}",
                    input.len(),
                    r
                );
                assert_eq!(r.unwrap(), input.len());
                assert_eq!(&recovered[..input.len()], input);
            }
        }
    }

    /// Verify the fixed header bytes are exactly right.
    #[test]
    fn test_header_bytes() {
        let mut c = Compressor::new(6).unwrap();
        let input = b"header check";
        let mut out = vec![0u8; gzip_compress_bound(input.len())];
        let n = gzip_compress(&mut c, input, &mut out);
        assert!(n >= 18, "output too short: {n}");
        assert_eq!(out[0], 0x1F, "ID1");
        assert_eq!(out[1], 0x8B, "ID2");
        assert_eq!(out[2], 8,    "CM = DEFLATE");
        assert_eq!(out[3], 0,    "FLG = 0 (no optional fields)");
        // MTIME = 0 (4 LE bytes)
        assert_eq!(&out[4..8], &[0, 0, 0, 0], "MTIME");
        // OS = 0xFF
        assert_eq!(out[9], 0xFF, "OS = unknown");
    }

    /// XFL is 0x04 for level 1 (fastest), 0x02 for levels >= 8 (slowest),
    /// 0x00 for levels 2..7.
    #[test]
    fn test_xfl_values() {
        let input = b"xfl test";
        for (level, expected_xfl) in [(1i32, 0x04u8), (2, 0), (6, 0), (7, 0), (8, 0x02), (12, 0x02)] {
            let mut c = Compressor::new(level).unwrap();
            let mut out = vec![0u8; gzip_compress_bound(input.len())];
            let n = gzip_compress(&mut c, input, &mut out);
            assert!(n > 0);
            assert_eq!(out[8], expected_xfl, "level={level}: XFL mismatch");
        }
    }

    /// Too-small output buffer returns 0.
    #[test]
    fn test_too_small_returns_zero() {
        let mut c = Compressor::new(6).unwrap();
        let mut out = vec![0u8; GZIP_MIN_OVERHEAD];
        let n = gzip_compress(&mut c, b"hello", &mut out);
        assert_eq!(n, 0);
    }
}
