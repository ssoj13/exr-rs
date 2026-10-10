//! Port of libdeflate `lib/zlib_constants.h` — constants for the zlib wrapper format.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/zlib_constants.h
//!
//! Used by both [`crate::zlib_compress`] and [`crate::zlib_decompress`].

/// Size of the zlib header in bytes (CMF + FLG).
/// Mirrors `ZLIB_MIN_HEADER_SIZE`.
pub const ZLIB_MIN_HEADER_SIZE: usize = 2;

/// Size of the zlib trailer in bytes (Adler-32 checksum, big-endian).
/// Mirrors `ZLIB_FOOTER_SIZE`.
pub const ZLIB_FOOTER_SIZE: usize = 4;

/// Minimum total overhead = header + footer.
/// Mirrors `ZLIB_MIN_OVERHEAD`.
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_MIN_HEADER_SIZE + ZLIB_FOOTER_SIZE;

/// CM field value for DEFLATE compression method (bits 0..3 of CMF byte).
/// Mirrors `ZLIB_CM_DEFLATE`.
pub const ZLIB_CM_DEFLATE: u8 = 8;

/// CINFO value for a 32 KiB LZ77 window size (bits 4..7 of CMF byte = log2(window)-8).
/// log2(32768) - 8 = 7.  Mirrors `ZLIB_CINFO_32K_WINDOW`.
pub const ZLIB_CINFO_32K_WINDOW: u8 = 7;

/// FLEVEL hint: fastest compression (bits 6..7 of FLG byte).
/// Mirrors `ZLIB_FASTEST_COMPRESSION`.
pub const ZLIB_FASTEST_COMPRESSION: u8 = 0;

/// FLEVEL hint: fast compression.
/// Mirrors `ZLIB_FAST_COMPRESSION`.
pub const ZLIB_FAST_COMPRESSION: u8 = 1;

/// FLEVEL hint: default compression.
/// Mirrors `ZLIB_DEFAULT_COMPRESSION`.
pub const ZLIB_DEFAULT_COMPRESSION: u8 = 2;

/// FLEVEL hint: slowest (best) compression.
/// Mirrors `ZLIB_SLOWEST_COMPRESSION`.
pub const ZLIB_SLOWEST_COMPRESSION: u8 = 3;

/// Bit position of the FDICT flag within the 16-bit header word
/// (bit 5 of the FLG byte = bit 5 of the low byte = bit 5 of the u16).
/// We store the header as a big-endian u16 where the high byte is CMF and
/// the low byte is FLG, so FDICT is at bit 5 of the low byte.
pub const ZLIB_FDICT_BIT: u16 = 1 << 5;
