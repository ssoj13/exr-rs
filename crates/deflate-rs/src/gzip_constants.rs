//! Port of libdeflate `lib/gzip_constants.h` — gzip format constants.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/gzip_constants.h
//!
//! Byte-exact layout of the gzip format (RFC 1952) header and footer fields.
//! Used by both `gzip_compress` and `gzip_decompress`.

/// Fixed 10-byte gzip header: ID1 ID2 CM FLG MTIME[4] XFL OS.
pub const GZIP_MIN_HEADER_SIZE: usize = 10;
/// Fixed 8-byte gzip footer: CRC32[4] ISIZE[4].
pub const GZIP_FOOTER_SIZE: usize = 8;
/// Total minimum overhead = header + footer (no optional fields).
pub const GZIP_MIN_OVERHEAD: usize = GZIP_MIN_HEADER_SIZE + GZIP_FOOTER_SIZE;

// --- Header magic bytes ---
pub const GZIP_ID1: u8 = 0x1F;
pub const GZIP_ID2: u8 = 0x8B;

/// CM field: compression method DEFLATE.
pub const GZIP_CM_DEFLATE: u8 = 8;

// --- FLG bits ---
pub const GZIP_FTEXT: u8     = 0x01; // file is ASCII text
pub const GZIP_FHCRC: u8     = 0x02; // CRC16 for the gzip header present
pub const GZIP_FEXTRA: u8    = 0x04; // optional extra field present
pub const GZIP_FNAME: u8     = 0x08; // original file name (NUL-terminated) present
pub const GZIP_FCOMMENT: u8  = 0x10; // file comment (NUL-terminated) present
/// Reserved FLG bits — must be zero in a valid gzip stream.
pub const GZIP_FRESERVED: u8 = 0xE0;

/// MTIME value used when modification time is unavailable (= 0).
pub const GZIP_MTIME_UNAVAILABLE: u32 = 0;

// --- XFL compression-hint values ---
/// XFL: slowest compression (levels 8–12 in libdeflate).
pub const GZIP_XFL_SLOWEST_COMPRESSION: u8  = 0x02;
/// XFL: fastest compression (levels 0–1 in libdeflate).
pub const GZIP_XFL_FASTEST_COMPRESSION: u8  = 0x04;

// --- OS field values (libdeflate always writes 0xFF = unknown) ---
pub const GZIP_OS_UNKNOWN: u8 = 255;
