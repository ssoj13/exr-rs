//! `deflate-rs` — pure-Rust 1:1 port of libdeflate (v1.26) for exr-rs.
//!
//! Port targets:
//! - libdeflate: <https://github.com/ebiggers/libdeflate/tree/v1.26>
//!
//! Byte-parity is verified against the vcpkg libdeflate **1.26** binary
//! (`cargo xtask parity-deflate`); xtask pins that version and fails on drift.
//!
//! # Foundation modules
//! - [`common`]  — portable helpers: LE/BE load/store, bit-scan, alignment maths.
//! - [`deflate_constants`] — all DEFLATE format constants.
//! - [`utils`]   — allocator indirection (mirrors `libdeflate_set_memory_allocator`).
//! - [`adler32`] — scalar Adler-32 checksum (byte-exact to libdeflate's scalar path).
//!
//! # Inflater
//! - [`deflate_decompress`] — raw DEFLATE decompressor (1:1 port of `deflate_decompress.c`).
//!
//! # Matchfinders (back-reference search for the compressor)
//! - [`matchfinder_common`] — shared defs: window constants, `MfPos`, `lz_hash`, `lz_extend`.
//! - [`hc_matchfinder`]     — hash-chain matchfinder (greedy/lazy levels).
//! - [`bt_matchfinder`]     — binary-tree matchfinder (near-optimal levels).
//! - [`ht_matchfinder`]     — hash-table matchfinder (fastest level).
//!
//! # Compressor
//! - [`deflate_compress`] — raw DEFLATE compressor, all 12 levels (1:1 port of
//!   `deflate_compress.c`).
//!
//! # zlib wrapper + public facade
//! - [`zlib_constants`] — header/footer sizes, CM/CINFO/FLEVEL constants.
//! - [`zlib_compress`]  — zlib stream compressor (header + DEFLATE + Adler-32).
//! - [`zlib_decompress`] — zlib stream decompressor (validate header, DEFLATE, verify Adler-32).
//!
//! # CRC-32 + gzip wrapper (complete libdeflate public API)
//! - [`crc32_tables`] — precomputed CRC-32 lookup tables (ported from `crc32_tables.h`).
//! - [`crc32`]        — CRC-32 checksum, slice-by-8 scalar path (`crc32.c`).
//! - [`gzip_constants`] — gzip header/footer constants (`gzip_constants.h`).
//! - [`gzip_compress`]  — gzip stream compressor (header + DEFLATE + CRC32/ISIZE footer).
//! - [`gzip_decompress`] — gzip stream decompressor (validate, DEFLATE, verify footer).
//!
//! # x86 SIMD acceleration tiers (performance only)
//! Each tier is byte-identical to the scalar source-of-truth above and is
//! reached only via the safe runtime dispatcher in [`cpu`] (with the
//! `DEFLATE_RS_SIMD=scalar|sse2|avx2|pclmul` env override). `unsafe` lives
//! exclusively in these dedicated modules, each behind a `#[target_feature]`:
//! - [`crc32_pclmul`]   — PCLMULQDQ+SSE4.1 carryless-multiply CRC-32 folding.
//! - [`adler32_sse2`] / [`adler32_avx2`] — vectorized Adler-32 (USE_VNNI=0).
//! - [`matchfinder_simd`] — AVX2/SSE2 `matchfinder_init`/`rebase`.
//! - [`matchfinder_neon`] — ARM NEON init/rebase (aarch64; tested on the macOS CI runner).
//! - [`cpu`] — safe runtime feature detection + env override (no `unsafe`).
//!
//! # Public facade (what `exr-core` calls)
//!
//! For **compression**:
//! ```ignore
//! use deflate_rs::Compressor;
//! let mut c = Compressor::new(6)?;
//! let compressed: Vec<u8> = c.zlib_compress(data);
//! // or raw DEFLATE:
//! let compressed: Vec<u8> = c.deflate_compress_vec(data);
//! // or gzip:
//! let compressed: Vec<u8> = c.gzip_compress(data);
//! ```
//!
//! For **decompression**:
//! ```ignore
//! use deflate_rs::Decompressor;
//! let mut d = Decompressor::new();
//! let plain: Vec<u8> = d.zlib_decompress(compressed, expected_size)?;
//! // or raw DEFLATE:
//! let plain: Vec<u8> = d.deflate_decompress_vec(compressed, expected_size)?;
//! // or gzip:
//! let plain: Vec<u8> = d.gzip_decompress(compressed, expected_size)?;
//! ```
//!
//! For **checksums**:
//! ```ignore
//! use deflate_rs::{adler32, crc32};
//! let a = adler32(1, data);          // mirrors libdeflate_adler32
//! let c = crc32::crc32(0, data);     // mirrors libdeflate_crc32
//! ```

pub mod adler32;
pub mod common;
pub mod cpu;
pub mod crc32;
pub mod crc32_pclmul;
pub mod crc32_tables;
pub mod deflate_constants;
pub mod deflate_decompress;
pub mod deflate_decompress_bmi2;
pub mod gzip_compress;
pub mod gzip_constants;
pub mod gzip_decompress;
mod probe;
pub mod adler32_sse2;
pub mod adler32_avx2;
pub mod matchfinder_common;
pub mod matchfinder_simd;
pub mod matchfinder_neon;
pub mod hc_matchfinder;
pub mod bt_matchfinder;
pub mod ht_matchfinder;
pub mod deflate_compress;
pub mod utils;
pub mod zlib_constants;
pub mod zlib_compress;
pub mod zlib_decompress;

// ---------------------------------------------------------------------------
// Re-exports (low-level / building-block API)
// ---------------------------------------------------------------------------

/// Compute or update an Adler-32 checksum.
///
/// Pass `adler = 1` to start a new checksum.  The return value is the updated
/// checksum, and can be passed back in for incremental hashing.
/// Mirrors `libdeflate_adler32`.
pub use adler32::adler32;

/// Compute or update a CRC-32 checksum (gzip polynomial 0xEDB88320 reflected).
///
/// Pass `crc = 0` to start a new checksum.  Pass an empty slice to get the
/// initial value (returns 0).  Mirrors `libdeflate_crc32`.
pub use crc32::crc32;

/// Error type for raw DEFLATE decompression.  Mirrors `libdeflate_result`.
pub use deflate_decompress::DecompressError;

/// Error type for zlib stream decompression.  Extends [`DecompressError`] with
/// header-validation and Adler-32 variants.
pub use zlib_decompress::ZlibDecompressError;

/// Error type for gzip stream decompression.  Extends [`DecompressError`] with
/// gzip header-validation and CRC32/ISIZE mismatch variants.
pub use gzip_decompress::GzipDecompressError;

/// Worst-case compressed size for raw DEFLATE.  Mirrors `libdeflate_deflate_compress_bound`.
pub use deflate_compress::deflate_compress_bound;

/// Worst-case compressed size for a zlib stream.  Mirrors `libdeflate_zlib_compress_bound`.
pub use zlib_compress::zlib_compress_bound;

/// Worst-case compressed size for a gzip stream.  Mirrors `libdeflate_gzip_compress_bound`.
pub use gzip_compress::gzip_compress_bound;

// ---------------------------------------------------------------------------
// High-level facade: Compressor
// ---------------------------------------------------------------------------

/// zlib / DEFLATE / gzip compressor.
///
/// Create one with [`Compressor::new`], then call the appropriate compress method.
/// Internally wraps [`deflate_compress::Compressor`] so all 12 libdeflate levels
/// are available.  Mirrors `libdeflate_alloc_compressor`.
pub use deflate_compress::Compressor as DeflateCompressor;

// Provide the idiomatic Rust wrapper type directly in this crate so that
// exr-core does not need to reach into the sub-module.

/// Compressor facade used by exr-core.
///
/// Thin wrapper around [`DeflateCompressor`] that adds `Vec<u8>`-returning
/// helpers for zlib, gzip, and raw-DEFLATE paths.
pub struct Compressor(deflate_compress::Compressor);

impl Compressor {
    /// Create a new compressor at the given level (1–12; 0 = pass-through;
    /// -1 = alias for the default level, 6, since libdeflate 1.26).
    ///
    /// Returns an error if `level` is outside the range accepted by libdeflate.
    /// Mirrors `libdeflate_alloc_compressor`.
    pub fn new(level: i32) -> Result<Self, deflate_compress::InvalidLevel> {
        deflate_compress::Compressor::new(level).map(Compressor)
    }

    /// Compress `input` as a complete **zlib stream** into `out`, reusing its
    /// allocation across calls.
    ///
    /// `out` is grown (never shrunk) to at least [`zlib_compress::zlib_compress_bound`]
    /// bytes if it is currently smaller; a caller that reuses one `out` buffer
    /// across many calls (e.g. one strip/scanline buffer streamed repeatedly)
    /// pays for at most one realloc, not one per call. The stream is written
    /// into `out[..]` starting at offset 0; the return value is the number of
    /// valid bytes at the front of `out` — bytes beyond it (if `out` was
    /// already longer than needed) are leftover and must be ignored by the
    /// caller. Mirrors `libdeflate_zlib_compress` (2-byte header + raw DEFLATE
    /// body + 4-byte Adler-32 trailer).
    pub fn zlib_compress_into(&mut self, input: &[u8], out: &mut Vec<u8>) -> usize {
        let bound = zlib_compress::zlib_compress_bound(input.len());
        if out.len() < bound {
            out.resize(bound, 0);
        }
        zlib_compress::zlib_compress(&mut self.0, input, out)
    }

    /// Compress `input` as a complete **zlib stream** into a new `Vec<u8>`.
    ///
    /// The stream is: 2-byte header + raw DEFLATE body + 4-byte Adler-32 trailer.
    /// This is what `libdeflate_zlib_compress` produces and what OpenEXR calls.
    ///
    /// Returns an empty `Vec` if compression fails (shouldn't happen with a
    /// correctly-sized allocation; see [`zlib_compress_bound`]). Thin wrapper
    /// over [`Compressor::zlib_compress_into`] — see that method for the
    /// buffer-reusing path.
    pub fn zlib_compress(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let n = self.zlib_compress_into(input, &mut out);
        out.truncate(n);
        out
    }

    /// Compress `input` as a raw **DEFLATE stream** (no zlib header/footer)
    /// into `out`, reusing its allocation across calls. See
    /// [`Compressor::zlib_compress_into`] for the grow-only/leftover-tail
    /// contract. Mirrors `libdeflate_deflate_compress`.
    pub fn deflate_compress_vec_into(&mut self, input: &[u8], out: &mut Vec<u8>) -> usize {
        let bound = deflate_compress::deflate_compress_bound(input.len());
        if out.len() < bound {
            out.resize(bound, 0);
        }
        self.0.deflate_compress(input, out)
    }

    /// Compress `input` as a raw **DEFLATE stream** (no zlib header/footer) into
    /// a new `Vec<u8>`.  Mirrors `libdeflate_deflate_compress`. Thin wrapper
    /// over [`Compressor::deflate_compress_vec_into`].
    pub fn deflate_compress_vec(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let n = self.deflate_compress_vec_into(input, &mut out);
        out.truncate(n);
        out
    }

    /// Compress `input` as a complete **gzip stream** into `out`, reusing its
    /// allocation across calls. See [`Compressor::zlib_compress_into`] for the
    /// grow-only/leftover-tail contract. Mirrors `libdeflate_gzip_compress`.
    pub fn gzip_compress_into(&mut self, input: &[u8], out: &mut Vec<u8>) -> usize {
        let bound = gzip_compress::gzip_compress_bound(input.len());
        if out.len() < bound {
            out.resize(bound, 0);
        }
        gzip_compress::gzip_compress(&mut self.0, input, out)
    }

    /// Compress `input` as a complete **gzip stream** into a new `Vec<u8>`.
    ///
    /// The stream is: 10-byte gzip header + raw DEFLATE body + 8-byte footer
    /// (CRC32 + ISIZE).  This is what `libdeflate_gzip_compress` produces.
    ///
    /// Returns an empty `Vec` if compression fails. Thin wrapper over
    /// [`Compressor::gzip_compress_into`].
    pub fn gzip_compress(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let n = self.gzip_compress_into(input, &mut out);
        out.truncate(n);
        out
    }

    /// Returns the compression level this compressor was created with.
    pub fn compression_level(&self) -> u32 {
        self.0.compression_level()
    }
}

// ---------------------------------------------------------------------------
// High-level facade: Decompressor
// ---------------------------------------------------------------------------

/// Decompressor facade used by exr-core.
///
/// Thin wrapper around [`deflate_decompress::Decompressor`] that adds
/// `Vec<u8>`-returning helpers for zlib, gzip, and raw-DEFLATE paths.
pub struct Decompressor(deflate_decompress::Decompressor);

impl Decompressor {
    /// Create a new decompressor.  Mirrors `libdeflate_alloc_decompressor`.
    pub fn new() -> Self {
        Decompressor(deflate_decompress::Decompressor::new())
    }

    /// Decompress a **zlib stream** into `out`, reusing its allocation across
    /// calls, growing it (never shrinking) to at least `expected_size` bytes
    /// if it is currently smaller. Only `out[..expected_size]` is used as the
    /// working buffer regardless of any extra capacity `out` already carries
    /// from a previous, larger call — so `expected_size` still bounds the
    /// decompressor's `InsufficientSpace` check exactly as before. Returns
    /// the actual decompressed length (`<= expected_size`); bytes beyond it
    /// in `out` are leftover from a prior call and must be ignored.
    ///
    /// Validates the header, decompresses the DEFLATE body, and verifies the
    /// Adler-32 trailer. This is libdeflate's *counted* form
    /// (`libdeflate_zlib_decompress` with a non-NULL `actual_out_nbytes_ret`):
    /// a stream that decompresses to fewer than `expected_size` bytes still
    /// succeeds. Callers that know the exact uncompressed size want
    /// [`Decompressor::zlib_decompress_exact_into`], which reports that case
    /// as [`ZlibDecompressError::ShortOutput`].
    pub fn zlib_decompress_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<usize, ZlibDecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        zlib_decompress::zlib_decompress(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a **zlib stream** into a new `Vec<u8>` of capacity `expected_size`.
    ///
    /// Validates the header, decompresses the DEFLATE body, and verifies the
    /// Adler-32 trailer.  Returns the decompressed bytes on success.
    ///
    /// This is libdeflate's *counted* form (`libdeflate_zlib_decompress` with a
    /// non-NULL `actual_out_nbytes_ret`): a stream that decompresses to fewer
    /// than `expected_size` bytes succeeds and yields a SHORTER `Vec`. Callers
    /// that know the exact uncompressed size want
    /// [`Decompressor::zlib_decompress_exact`], which reports that case as
    /// [`ZlibDecompressError::ShortOutput`]. Thin wrapper over
    /// [`Decompressor::zlib_decompress_into`].
    pub fn zlib_decompress(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, ZlibDecompressError> {
        let mut out = Vec::new();
        let n = self.zlib_decompress_into(input, expected_size, &mut out)?;
        out.truncate(n);
        Ok(out)
    }

    /// Decompress a raw **DEFLATE stream** (no zlib header/footer) into `out`,
    /// reusing its allocation across calls. See
    /// [`Decompressor::zlib_decompress_into`] for the grow-only/leftover-tail
    /// contract.
    pub fn deflate_decompress_vec_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<usize, DecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        deflate_decompress::deflate_decompress(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a raw **DEFLATE stream** (no zlib header/footer) into a new
    /// `Vec<u8>` of capacity `expected_size`.
    ///
    /// Counted form — see [`Decompressor::zlib_decompress`]; a short stream
    /// returns a short `Vec`, not an error. Use
    /// [`Decompressor::deflate_decompress_exact`] for the exact-size contract.
    /// Thin wrapper over [`Decompressor::deflate_decompress_vec_into`].
    pub fn deflate_decompress_vec(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, DecompressError> {
        let mut out = Vec::new();
        let n = self.deflate_decompress_vec_into(input, expected_size, &mut out)?;
        out.truncate(n);
        Ok(out)
    }

    /// Decompress a **gzip stream** into `out`, reusing its allocation across
    /// calls. See [`Decompressor::zlib_decompress_into`] for the
    /// grow-only/leftover-tail contract.
    pub fn gzip_decompress_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<usize, GzipDecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        gzip_decompress::gzip_decompress_with(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a **gzip stream** into a new `Vec<u8>` of capacity `expected_size`.
    ///
    /// Validates the gzip header and footer (CRC32 + ISIZE), decompresses the
    /// DEFLATE body.  Returns the decompressed bytes on success.
    ///
    /// Counted form — see [`Decompressor::zlib_decompress`]; use
    /// [`Decompressor::gzip_decompress_exact`] for the exact-size contract.
    /// Thin wrapper over [`Decompressor::gzip_decompress_into`].
    pub fn gzip_decompress(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, GzipDecompressError> {
        let mut out = Vec::new();
        let n = self.gzip_decompress_into(input, expected_size, &mut out)?;
        out.truncate(n);
        Ok(out)
    }

    /// Decompress a **zlib stream** that is known to be exactly `expected_size`
    /// bytes uncompressed, into `out`, reusing its allocation across calls
    /// (grow-only; see [`Decompressor::zlib_decompress_into`]). On success,
    /// `out[..expected_size]` holds the decompressed bytes.
    pub fn zlib_decompress_exact_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), ZlibDecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        zlib_decompress::zlib_decompress_exact(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a **zlib stream** that is known to be exactly `expected_size`
    /// bytes uncompressed.
    ///
    /// Mirrors `libdeflate_zlib_decompress` with a NULL `actual_out_nbytes_ret`
    /// (`libdeflate.h:222-229`): a stream that decompresses to fewer bytes is
    /// [`ZlibDecompressError::ShortOutput`], not a silent short buffer. The
    /// returned `Vec` always has length `expected_size`. Thin wrapper over
    /// [`Decompressor::zlib_decompress_exact_into`].
    pub fn zlib_decompress_exact(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, ZlibDecompressError> {
        let mut out = Vec::new();
        self.zlib_decompress_exact_into(input, expected_size, &mut out)?;
        out.truncate(expected_size);
        Ok(out)
    }

    /// Decompress a raw **DEFLATE stream** that is known to be exactly
    /// `expected_size` bytes uncompressed, into `out`, reusing its allocation
    /// across calls. See [`Decompressor::zlib_decompress_exact_into`].
    pub fn deflate_decompress_exact_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), DecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        deflate_decompress::deflate_decompress_exact(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a raw **DEFLATE stream** that is known to be exactly
    /// `expected_size` bytes uncompressed.  See
    /// [`Decompressor::zlib_decompress_exact`]. Thin wrapper over
    /// [`Decompressor::deflate_decompress_exact_into`].
    pub fn deflate_decompress_exact(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, DecompressError> {
        let mut out = Vec::new();
        self.deflate_decompress_exact_into(input, expected_size, &mut out)?;
        out.truncate(expected_size);
        Ok(out)
    }

    /// Decompress a **gzip stream** that is known to be exactly `expected_size`
    /// bytes uncompressed, into `out`, reusing its allocation across calls.
    /// See [`Decompressor::zlib_decompress_exact_into`].
    pub fn gzip_decompress_exact_into(
        &mut self,
        input: &[u8],
        expected_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), GzipDecompressError> {
        if out.len() < expected_size {
            out.resize(expected_size, 0);
        }
        gzip_decompress::gzip_decompress_exact(&mut self.0, input, &mut out[..expected_size])
    }

    /// Decompress a **gzip stream** that is known to be exactly `expected_size`
    /// bytes uncompressed.  See [`Decompressor::zlib_decompress_exact`]. Thin
    /// wrapper over [`Decompressor::gzip_decompress_exact_into`].
    pub fn gzip_decompress_exact(
        &mut self,
        input: &[u8],
        expected_size: usize,
    ) -> Result<Vec<u8>, GzipDecompressError> {
        let mut out = Vec::new();
        self.gzip_decompress_exact_into(input, expected_size, &mut out)?;
        out.truncate(expected_size);
        Ok(out)
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Facade tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// libdeflate distinguishes "decompressed to fewer bytes than you said"
    /// (`LIBDEFLATE_SHORT_OUTPUT`, `libdeflate.h:201-203`) from success. The
    /// counted facade methods are the non-NULL `actual_out_nbytes_ret` form and
    /// must keep returning a short buffer; the `_exact` methods are the NULL
    /// form and must report `ShortOutput`.
    ///
    /// The result codes are also pinned against the real C library in
    /// `tests/parity_libdeflate.rs` (`zlibresult.*` golden keys).
    #[test]
    fn zlib_exact_reports_short_output_while_counted_does_not() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(37);
        let compressed = Compressor::new(6).expect("level 6").zlib_compress(&payload);
        let mut d = Decompressor::new();

        // Exact size: success, and the buffer is fully filled.
        let exact = d
            .zlib_decompress_exact(&compressed, payload.len())
            .expect("exact size decodes");
        assert_eq!(exact, payload);

        // Oversized: the C library returns SHORT_OUTPUT here.
        assert_eq!(
            d.zlib_decompress_exact(&compressed, payload.len() + 1),
            Err(ZlibDecompressError::ShortOutput)
        );

        // Undersized: INSUFFICIENT_SPACE, not SHORT_OUTPUT.
        assert_eq!(
            d.zlib_decompress_exact(&compressed, payload.len() - 1),
            Err(ZlibDecompressError::InsufficientSpace)
        );

        // The counted form is unchanged: oversized succeeds with a short Vec.
        let counted = d
            .zlib_decompress(&compressed, payload.len() + 1)
            .expect("counted form tolerates an oversized buffer");
        assert_eq!(counted, payload);
        assert_eq!(counted.len(), payload.len());
    }

    /// The same contract on the raw-DEFLATE and gzip entry points.
    #[test]
    fn deflate_and_gzip_exact_report_short_output() {
        let payload = vec![0x5Au8; 4096];
        let mut c = Compressor::new(9).expect("level 9");
        let raw = c.deflate_compress_vec(&payload);
        let gz = c.gzip_compress(&payload);
        let mut d = Decompressor::new();

        assert_eq!(
            d.deflate_decompress_exact(&raw, payload.len()).as_deref(),
            Ok(payload.as_slice())
        );
        assert_eq!(
            d.deflate_decompress_exact(&raw, payload.len() + 8),
            Err(DecompressError::ShortOutput)
        );
        assert_eq!(
            d.gzip_decompress_exact(&gz, payload.len()).as_deref(),
            Ok(payload.as_slice())
        );
        assert_eq!(
            d.gzip_decompress_exact(&gz, payload.len() + 8),
            Err(GzipDecompressError::ShortOutput)
        );
    }

    /// The buffer-reusing `_into` API must (a) match the `Vec`-returning
    /// wrappers byte-for-byte and (b) actually reuse a too-large buffer
    /// (grow-only: never shrink, ignore the leftover tail) rather than
    /// silently reallocating every call. Covers zlib/deflate/gzip compress
    /// and all six decompress entry points (counted + exact).
    ///
    /// Mutation checks (both against `zlib_compress_into`, both reverted):
    /// 1. Commented out the `if out.len() < bound { out.resize(bound, 0); }`
    ///    grow entirely, so `out` keeps whatever length the caller passed
    ///    (including empty). RED: the grow-path assertions at the end of this
    ///    test (starting from `Vec::new()`) fail with a byte mismatch against
    ///    the `Vec`-returning method, since an un-grown empty `out` can't hold
    ///    the compressed stream.
    /// 2. First tried asserting `out.capacity()` is unchanged as the
    ///    "didn't reallocate" check, and mutated to `*out = vec![0u8; ...]`
    ///    (an unconditional fresh allocation). This did NOT go red: a fresh
    ///    `Vec` of the same length gets the same `capacity()` value as the
    ///    old one, so a same-size reallocation is invisible to a capacity
    ///    comparison. Fixed by comparing `out.as_ptr()` instead (`ptr_before`
    ///    below), which is unique per allocation and DOES go red under that
    ///    same mutation (confirmed) -- capacity was the wrong signal for "no
    ///    realloc", pointer identity is the real signal.
    ///
    /// The correctness half (writing to `out[..]` and returning the right
    /// count) is covered separately by comparing against the plain
    /// `Vec`-returning method for every call site.
    #[test]
    fn into_variants_match_vec_variants_and_reuse_capacity() {
        let payload = b"the quick brown fox jumps over the lazy dog, repeatedly, for coverage."
            .repeat(50);

        // --- Compress: zlib / deflate / gzip ---
        let mut c1 = Compressor::new(6).expect("level 6");
        let mut c2 = Compressor::new(6).expect("level 6");

        let zlib_vec = c1.zlib_compress(&payload);
        let mut zlib_buf = vec![0xAAu8; 10 * payload.len()]; // oversized vs the input-based bound, deliberately non-zero
        let ptr_before = zlib_buf.as_ptr();
        let n = c2.zlib_compress_into(&payload, &mut zlib_buf);
        assert_eq!(&zlib_buf[..n], zlib_vec.as_slice());
        assert_eq!(zlib_buf.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let raw_vec = c1.deflate_compress_vec(&payload);
        let mut raw_buf = vec![0xAAu8; 10 * payload.len()];
        let ptr_before = raw_buf.as_ptr();
        let n = c2.deflate_compress_vec_into(&payload, &mut raw_buf);
        assert_eq!(&raw_buf[..n], raw_vec.as_slice());
        assert_eq!(raw_buf.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let gz_vec = c1.gzip_compress(&payload);
        let mut gz_buf = vec![0xAAu8; 10 * payload.len()];
        let ptr_before = gz_buf.as_ptr();
        let n = c2.gzip_compress_into(&payload, &mut gz_buf);
        assert_eq!(&gz_buf[..n], gz_vec.as_slice());
        assert_eq!(gz_buf.as_ptr(), ptr_before, "must not reallocate when already large enough");

        // --- Decompress: zlib / deflate / gzip, counted + exact ---
        let mut d1 = Decompressor::new();
        let mut d2 = Decompressor::new();

        let zlib_decoded = d1.zlib_decompress(&zlib_vec, payload.len()).unwrap();
        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        let n = d2.zlib_decompress_into(&zlib_vec, payload.len(), &mut out).unwrap();
        assert_eq!(&out[..n], zlib_decoded.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        d2.zlib_decompress_exact_into(&zlib_vec, payload.len(), &mut out).unwrap();
        assert_eq!(&out[..payload.len()], payload.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let raw_decoded = d1.deflate_decompress_vec(&raw_vec, payload.len()).unwrap();
        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        let n = d2
            .deflate_decompress_vec_into(&raw_vec, payload.len(), &mut out)
            .unwrap();
        assert_eq!(&out[..n], raw_decoded.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        d2.deflate_decompress_exact_into(&raw_vec, payload.len(), &mut out)
            .unwrap();
        assert_eq!(&out[..payload.len()], payload.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let gz_decoded = d1.gzip_decompress(&gz_vec, payload.len()).unwrap();
        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        let n = d2.gzip_decompress_into(&gz_vec, payload.len(), &mut out).unwrap();
        assert_eq!(&out[..n], gz_decoded.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        let mut out = vec![0xBBu8; 10 * payload.len()];
        let ptr_before = out.as_ptr();
        d2.gzip_decompress_exact_into(&gz_vec, payload.len(), &mut out).unwrap();
        assert_eq!(&out[..payload.len()], payload.as_slice());
        assert_eq!(out.as_ptr(), ptr_before, "must not reallocate when already large enough");

        // --- Grow path: start from an empty Vec, confirm it actually grows
        // to (at least) the bound instead of panicking/under-writing. ---
        let mut grow_buf = Vec::new();
        let n = c2.zlib_compress_into(&payload, &mut grow_buf);
        assert_eq!(&grow_buf[..n], zlib_vec.as_slice());
        let mut grow_out = Vec::new();
        let n = d2
            .zlib_decompress_into(&zlib_vec, payload.len(), &mut grow_out)
            .unwrap();
        assert_eq!(&grow_out[..n], payload.as_slice());
    }
}
