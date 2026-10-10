//! Port of OpenEXR `ImfZipCompressor.cpp` / `ImfZipCompressor.h` and the byte
//! algorithm in `OpenEXRCore/internal_zip.c` — the **ZIP** and **ZIPS**
//! scanline codecs (the most common EXR compression).
//!
//! # What this codec does (1:1 with C++)
//!
//! `ZipCompressor` is a `Compressor` whose `numScanLines()` is **16** for ZIP
//! (`ZIP_COMPRESSION`, 16 scanlines per chunk) and **1** for ZIPS
//! (`ZIPS_COMPRESSION`, one scanline per chunk).  Both share the exact same
//! per-chunk byte transform; only the chunk height differs.  The transform is:
//!
//! 1. **reorder + predict** — the shared
//!    [`reorder_and_predict`]
//!    (the EXR first-difference predictor + de-interleave), *identical* to the
//!    pre-processing RLE applies (`internal_zip_deconstruct_bytes` ==
//!    `reorder_and_predict`).  See [`crate::compressor::predictor`] for why this
//!    is shared rather than duplicated.
//! 2. **deflate (zlib)** — `libdeflate_zlib_compress` at the default ZIP level.
//!    Our [`deflate_rs::Compressor::zlib_compress`] is a 1:1 port of libdeflate
//!    (matched 72/72 against the C library), so at the same level it produces
//!    **byte-identical** zlib streams to C++.
//!
//! Decompression reverses the order: `libdeflate_zlib_decompress` then
//! [`unpredict_and_reorder`]
//! (`internal_zip_reconstruct_bytes`, scalar `reconstruct` + `interleave`).
//!
//! # The compression level (byte-affecting; confirmed from C++)
//!
//! `internal_zip.c::apply_zip_impl` calls `exr_get_zip_compression_level`, which
//! resolves to the context's `default_zip_level`.  That default is `-1`
//! (`base.c: sDefaultZipLevel = -1`), and `compression.c::exr_compress_buffer`
//! maps a negative level to `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL`, which is **4**
//! (`compression.c: #define EXR_DEFAULT_ZLIB_COMPRESS_LEVEL 4`).  So unless an
//! application overrides it, every ZIP/ZIPS EXR is written at libdeflate level
//! 4.  [`DEFAULT_ZIP_LEVEL`] encodes this; passing it to `deflate_rs` yields the
//! same deflate stream C++ writes.
//!
//! # The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `apply_zip_impl` (**v3.4.15** `internal_zip.c:392`
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_zip.c#L392>):
//! after deflating, **if `compbufsz >= encode->packed_bytes` it copies
//! `encode->packed_buffer` (the ORIGINAL pre-transform line bytes) and sets
//! `compbufsz = packed_bytes`.** This `>=` only exists from 3.4.x; the
//! v3.3.11 port-base rule is `>` (raw wins only when *strictly* larger). Per
//! the behavioural target (C++ 3.4.15), this module
//! follows 3.4.15's `>=` here, so the byte-affecting citation below is pinned
//! to 3.4.15 rather than the file's structural v3.3.11 port-base pin.
//! We replicate this: when the zlib stream is not **strictly smaller** than the
//! raw input we return a copy of the original (pre-transform) `data` unchanged.
//! Note the comparison is `>=`, i.e. at equality the raw bytes win — a
//! same-size zlib wrapper buys nothing, and on disk `dataSize ==
//! uncompressedSize` must mean "raw", which is how the reader detects the
//! passthrough.
//!
//! This codec-local rule mirrors the C codec exactly; it is **not** the writer's
//! store-raw decision.  That one lives in
//! [`crate::compressor::stored_chunk`] (C++ `ImfOutputFile.cpp:573`) and is
//! applied to every chunk regardless of codec.
//!
//! # Format
//!
//! ZIP/ZIPS use the machine-independent **Xdr** byte format (they do not
//! override `Compressor::format()`), so the line buffer handed to
//! [`compress`](ZipCompressor::compress) is already in the on-disk
//! little-endian layout — no native/Xdr conversion happens here, exactly as in
//! C++ (`_data->format == XDR` for ZIP).
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfZipCompressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfZipCompressor.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_zip.c>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/compression.c>

use crate::compressor::predictor::{reorder_and_predict, unpredict_and_reorder};
use crate::compressor::zlib::{compress_into, new_deflate_compressor, uncompress_into, LowCompressor, LowDecompressor};
use crate::compressor::{Compressor, Format};
use crate::error::Result;

/// Default ZIP/ZIPS libdeflate compression level used by OpenEXR — re-export
/// of [`crate::compressor::zlib::DEFAULT_ZIP_LEVEL`], the SSOT constant
/// (this used to be defined here and duplicated in
/// `IDManifest::compress`).
pub const DEFAULT_ZIP_LEVEL: i32 = crate::compressor::zlib::DEFAULT_ZIP_LEVEL;

// ---------------------------------------------------------------------------
// ZipCompressor — the Compressor seam implementation
// ---------------------------------------------------------------------------

/// ZIP / ZIPS scanline codec — port of `Imf::ZipCompressor`.
///
/// `num_scan_lines()` is **16** (ZIP) or **1** (ZIPS); both run the same
/// predictor + zlib pipeline.  Owns scratch buffers whose contents are returned
/// as `&[u8]` by [`compress`](Compressor::compress) /
/// [`uncompress`](Compressor::uncompress); the slice is valid until the next
/// call (mirroring the C++ member output buffer).
pub struct ZipCompressor {
    /// Number of scan lines per chunk — C++ ctor's `numScanLines` (1 or 16).
    num_scan_lines: i32,
    /// Maximum uncompressed bytes in a whole chunk — `maxScanLineSize *
    /// numScanLines` (the size the C++ base `Compressor` allocates for its
    /// in/out buffers).  Used as the exact decode capacity in `uncompress`, so
    /// a chunk never under-allocates regardless of how well it compressed.
    max_chunk_size: usize,
    /// Scratch buffer holding the result of the last `compress`/`uncompress`
    /// call (the C++ output buffer).
    out: Vec<u8>,
    /// Scratch buffer for the intermediate reordered+predicted bytes (the C
    /// core's `scratch_buffer_1`).
    scratch: Vec<u8>,
    /// Persistent low-level deflate compressor — constructed ONCE here
    /// instead of per `compress()` call (
    /// `deflate_rs::Compressor::new` allocates real matchfinder/Huffman
    /// tables, so a fresh one per chunk was a genuine per-chunk allocation
    /// cost C++ does not pay either — see `compressor::zlib`'s module doc).
    /// Its level: resolved once at construction from
    /// `header.codec_zip_level()` (the per-header level if C++ would accept
    /// it, else the process-global default) — matching
    /// C++ `apply_zip_impl`'s `exr_get_zip_compression_level` call
    /// (`internal_zip.c:368-370`). A negative header-level value (`-1`,
    /// "let the codec pick") resolves to [`DEFAULT_ZIP_LEVEL`], matching
    /// `compression.c`'s `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL` fallback.
    deflate_compressor: LowCompressor,
    /// Persistent low-level zlib decompressor (fixed-size internal tables,
    /// no heap allocation, but reused for symmetry with `deflate_compressor`
    /// and to avoid re-zeroing its tables every call).
    deflate_decompressor: LowDecompressor,
}

impl ZipCompressor {
    /// Create a new ZIP/ZIPS compressor — port of
    /// `ZipCompressor::ZipCompressor(const Header&, size_t maxScanLineSize, int numScanLines)`.
    ///
    /// `max_scan_line_size` is the maximum uncompressed bytes in a single scan
    /// line; `num_scan_lines` is 16 (ZIP) or 1 (ZIPS).
    ///
    /// `header.codec_zip_level()` drives the actual deflate level (was
    /// hardcoded to [`DEFAULT_ZIP_LEVEL`] regardless of the header).
    ///
    /// # Errors
    /// [`ExrError::Overflow`] ("Integer multiplication overflow.") if
    /// `max_scan_line_size * num_scan_lines` overflows `usize` — port of
    /// `_maxRawSize = uiMult(maxScanLineSize, numScanLines)` (`ImfZip.cpp`).
    pub fn new(
        header: &crate::header::Header,
        max_scan_line_size: usize,
        num_scan_lines: i32,
    ) -> Result<Self> {
        // C++ allocates the codec in/out buffers at maxScanLineSize *
        // numScanLines (the largest a single chunk's unpacked data can be).
        let max_chunk_size = crate::checked_arithmetic::ui_mult(
            max_scan_line_size,
            num_scan_lines.max(1) as usize,
        )?;
        let deflate_compressor = new_deflate_compressor(header.codec_zip_level())?;
        Ok(Self {
            num_scan_lines,
            max_chunk_size,
            out: Vec::new(),
            scratch: Vec::new(),
            deflate_compressor,
            deflate_decompressor: LowDecompressor::new(),
        })
    }
}

impl Compressor for ZipCompressor {
    /// 16 scan lines per chunk for ZIP, 1 for ZIPS.
    ///
    /// C++: `Compressor(hdr, (numScanLines == 16) ? ZIPS : ZIP, maxScanLineSize,
    /// numScanLines)` — the trailing `numScanLines` is what is reported here.
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// ZIP/ZIPS use the machine-independent Xdr format (they do not override the
    /// C++ `Compressor::format()` default).
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one chunk — port of `internal_exr_apply_zip` / `apply_zip_impl`.
    ///
    /// Applies reorder+predict, then deflates at the default ZIP level; if the
    /// deflate output is **not strictly smaller** than the input, returns the
    /// raw (pre-transform) input unchanged (the "no-shrink → store raw" rule,
    /// `internal_zip.c:392`).
    ///
    /// # Errors
    /// The deflate compressor's level is validated once at construction
    /// ([`ZipCompressor::new`]); `compress` itself only surfaces
    /// [`crate::error::ExrError::Logic`] in the (unreachable in practice)
    /// case `compress_into` reports `zlib_compress` returning 0 despite a
    /// correctly-sized buffer.
    fn compress(&mut self, data: &[u8], _min_y: i32) -> Result<&[u8]> {
        let srcb = data.len();

        // Empty chunk: nothing to deflate; mirror C++ by storing the (empty)
        // raw bytes (the zlib stream of empty input is larger than 0 bytes).
        if srcb == 0 {
            self.out.clear();
            return Ok(&self.out);
        }

        // C++ `internal_zip_deconstruct_bytes(scratch, packed, packed_bytes)`.
        // `reorder_and_predict` fully overwrites all `srcb` bytes of
        // `scratch` unconditionally, so growing/truncating
        // via `ensure_len` instead of `clear()` + zero-fill is sound — see
        // its doc.
        super::ensure_len(&mut self.scratch, srcb);
        reorder_and_predict(&mut self.scratch, data);

        // C++ `exr_compress_buffer(level, scratch, packed_bytes, ...)`. C
        // allocates a fresh libdeflate compressor per call; we reuse our own
        // persistent one instead (see `compressor::zlib`'s module doc) --
        // "better than C" without changing the format.
        compress_into(&mut self.deflate_compressor, &self.scratch, &mut self.out)?;

        // C++ `apply_zip_impl` (internal_zip.c:392): if (compbufsz >=
        // packed_bytes) { memcpy(out, packed_buffer, packed_bytes);
        // compbufsz = packed_bytes; }  At equality the RAW bytes win.
        if self.out.len() >= srcb {
            self.out.clear();
            self.out.extend_from_slice(data);
        }

        Ok(&self.out)
    }

    /// Uncompress one ZIP/ZIPS chunk — port of `internal_exr_undo_zip` /
    /// `undo_zip_impl`.
    ///
    /// Inflates the zlib stream into the scratch buffer, then runs
    /// [`unpredict_and_reorder`]
    /// into the output buffer.
    ///
    /// The caller only invokes this for genuinely compressed chunks; the raw
    /// passthrough (`dataSize == uncompressedSize`) is handled one level up,
    /// exactly as the C core's `internal_exr_undo_zip` short-circuits when
    /// `comp_buf_size == uncompressed_size`.  Because the zlib stream does not
    /// record the uncompressed length here, the inverse transform is driven by
    /// the decoded byte count.
    ///
    /// # Errors
    /// Returns [`ExrError::Input`] if the zlib stream is malformed or does not
    /// decode (corrupt chunk).
    fn uncompress(&mut self, data: &[u8], _min_y: i32, expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }

        // C `undo_zip_impl` decodes into a scratch buffer of size
        // `uncompressed_size` -- the CALLER's exact per-chunk target
        // (`expected_len`), not just the construction-time
        // `max_chunk_size` upper bound (== maxScanLineSize * numScanLines for
        // a FULL scanline/tiled chunk, always >= any real chunk in a
        // scanline/tiled file, but wrong for deep, where the packed/sample
        // table size genuinely varies chunk to chunk). `.max(self.max_chunk_size)`
        // is a no-op for scanline/tiled (expected_len is always <= it there)
        // and keeps deep's construction-time `max_chunk_size` (often 0, see
        // `deep::scanline_input`/`tiled_input`) from ever narrowing the cap
        // below what the file's own worst case already guarantees.
        // `deflate_rs::zlib_decompress` fills the real decoded length and
        // truncates; the reader re-validates the result against the chunk's
        // exact `expected_uncompressed`.
        let cap = expected_len.max(self.max_chunk_size).max(1);

        // Use the checked entry point so trailing bytes after the zlib
        // stream are rejected, matching `exr_uncompress_buffer` (see
        // `uncompress_into` above). Decodes directly into
        // `self.scratch` -- no intermediate `decoded` Vec + copy.
        uncompress_into(
            &mut self.deflate_decompressor,
            data,
            &mut self.scratch,
            cap,
            "ZIP",
            crate::compressor::zlib::ShortOutputPolicy::RequireExact,
        )?;

        // C++ `internal_zip_reconstruct_bytes(out, scratch, actual_out_bytes)`:
        // reconstruct (un-predict) then interleave (reorder inverse).
        // `unpredict_and_reorder` fully overwrites all `outsz` bytes of
        // `self.out` unconditionally -- same `ensure_len`
        // argument as the compress side above.
        let outsz = self.scratch.len();
        super::ensure_len(&mut self.out, outsz);
        unpredict_and_reorder(&mut self.out, &mut self.scratch);

        Ok(&self.out)
    }

    fn out_buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;

    /// A default header — `zip_level()` resolves to whatever the process
    /// global is at the time (default `-1`, resolved to [`DEFAULT_ZIP_LEVEL`]
    /// by `ZipCompressor::new`), same as the tests that predate per-header levels.
    fn hdr() -> Header {
        Header::new()
    }

    /// Regression test: `compress`/`uncompress` now
    /// reuse ONE persistent `deflate_compressor`/`deflate_decompressor`
    /// across calls, and `uncompress_into`'s destination buffer is grown or
    /// (cheaply, without zero-filling) truncated relative to whatever it
    /// held from the PRIOR call, instead of always being cleared and
    /// zero-filled from scratch. Round-trips a FULL-size chunk, then a
    /// SMALLER (partial-block-shaped) chunk, then a FULL-size chunk again,
    /// all on ONE `ZipCompressor` instance, and checks each result against a
    /// FRESH instance decoding the identical bytes with no prior call at any
    /// other size — a leaked/stale byte, or a buffer left too short after a
    /// shrink-then-grow sequence, would show up as a mismatch here.
    ///
    /// Mutation check performed while writing this test (both tried for
    /// real, not assumed):
    /// - An EARLIER version of this test called `compress()` immediately
    ///   before each `uncompress()` on the same instance, which left
    ///   `self.scratch` at the JUST-COMPRESSED chunk's own exact size as a
    ///   side effect of `compress()`'s OWN `clear(); resize(srcb, 0)` --
    ///   masking `uncompress_into`'s grow-or-truncate logic entirely (it
    ///   never actually had to grow OR meaningfully shrink). Fixed by
    ///   pre-compressing all three chunks with a SEPARATE encoder first, so
    ///   the loop under test calls ONLY `uncompress()`, back to back.
    /// - With that fixed, changing `uncompress_into`'s
    ///   `if out.len() < cap { resize } else { truncate }` to an
    ///   unconditional `out.truncate(cap)` (dropping the growth branch)
    ///   DOES fail this test: after the `partial` chunk leaves
    ///   `self.scratch` shorter than `cap`, `truncate` cannot grow it back
    ///   for the next `full2` chunk, and `zlib_decompress_ex` correctly
    ///   reports `ExrError::Input("... InsufficientSpace")` (not a panic --
    ///   the too-small destination is caught cleanly) — confirmed by
    ///   actually applying that exact mutation and re-running this test
    ///   before reverting it.
    #[test]
    fn compress_uncompress_reused_across_sizes_matches_fresh_instance() {
        // Sequential-u16-LE data (the same shape `zip_round_trip_compressible`
        // uses): after reorder+predict the high-byte half is mostly constant,
        // so this genuinely shrinks under ZIP at every size tried here.
        // `uncompress()` is documented as only ever called for chunks that
        // really compressed (matching the container's own dispatch), so a
        // random/incompressible fixture would exercise the wrong path
        // ("no-shrink -> store raw") instead.
        fn make_data(n_u16: usize, start: u16) -> Vec<u8> {
            let mut data = Vec::with_capacity(n_u16 * 2);
            for i in 0..n_u16 as u16 {
                data.extend_from_slice(&start.wrapping_add(i).to_le_bytes());
            }
            data
        }

        // A "full block" (max_chunk_size-sized) and a "partial trailing
        // block" (much smaller) shape, matching how a real scanline block's
        // LAST chunk is often smaller than every earlier one.
        let max_chunk_size = 20_000usize;
        let full1 = make_data(max_chunk_size / 2, 0);
        let partial = make_data(400, 10_000);
        let full2 = make_data(max_chunk_size / 2, 30_000);

        // num_scan_lines=1 so the struct's own `max_chunk_size` (== cap
        // passed to `uncompress_into`) equals `max_chunk_size` above exactly
        // -- `full1`/`full2` then decode to (close to) `cap` while `partial`
        // decodes to far less, genuinely exercising both the "grow" and the
        // "already >= cap, just truncate" branches as the loop alternates.
        //
        // Pre-compress all three with a SEPARATE encoder instance, so the
        // loop below calls ONLY `uncompress()` on `c` -- back to back, with
        // no intervening `compress()` call to (re)set `self.scratch`'s
        // length to the right size as a side effect. This is what actually
        // exercises `uncompress_into`'s own grow-or-truncate logic across
        // real size changes: `self.scratch`'s length after each `uncompress`
        // is that call's real `actual_out`, which the NEXT call's `cap` (a
        // constant, `self.max_chunk_size`) may be smaller OR larger than.
        let mut encoder = ZipCompressor::new(&hdr(), max_chunk_size, 1).unwrap();
        let compressed: Vec<Vec<u8>> = [&full1, &partial, &full2]
            .iter()
            .map(|data| {
                let c = encoder.compress(data, 0).unwrap().to_vec();
                assert!(c.len() < data.len(), "test setup: fixture must actually compress, not store raw");
                c
            })
            .collect();

        let mut c = ZipCompressor::new(&hdr(), max_chunk_size, 1).unwrap();
        let inputs = [&full1, &partial, &full2];
        let labels = ["full1", "partial", "full2"];
        for i in 0..3 {
            let restored = c.uncompress(&compressed[i], 0, 0).unwrap().to_vec();
            assert_eq!(&restored, inputs[i], "reused instance mismatch for {}", labels[i]);

            let mut fresh = ZipCompressor::new(&hdr(), max_chunk_size, 1).unwrap();
            let fresh_restored = fresh.uncompress(&compressed[i], 0, 0).unwrap().to_vec();
            assert_eq!(
                restored, fresh_restored,
                "reused instance diverged from a fresh one for {}", labels[i]
            );
        }
    }

    /// Full codec round trip: compress → uncompress = identity, for both the
    /// compressible and the incompressible (no-shrink → raw) paths.
    #[test]
    fn zip_round_trip_compressible() {
        // 16-bit-style data with strong runs after reorder (constant high byte).
        let mut data = Vec::new();
        for i in 0..1000u16 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        let mut c = ZipCompressor::new(&hdr(), data.len(), 16).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        assert!(
            compressed.len() < data.len(),
            "expected compression: {} >= {}",
            compressed.len(),
            data.len()
        );
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, data.as_slice());
    }

    #[test]
    fn zips_round_trip_compressible() {
        let data = vec![0xABu8; 4096];
        let mut c = ZipCompressor::new(&hdr(), data.len(), 1).unwrap();
        assert_eq!(c.num_scan_lines(), 1);
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        assert!(compressed.len() < data.len(), "all-same must compress");
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, data.as_slice());
    }

    /// Incompressible (pseudo-random) data must trigger the "no-shrink → store
    /// raw" path: `compress` returns the bytes unchanged.
    #[test]
    fn zip_no_shrink_stores_raw() {
        let mut data = Vec::with_capacity(512);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..512 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            data.push((state >> 16) as u8);
        }
        let mut c = ZipCompressor::new(&hdr(), data.len(), 16).unwrap();
        let compressed = c.compress(&data, 0).unwrap();
        assert_eq!(
            compressed,
            data.as_slice(),
            "incompressible data must be stored raw (== original bytes)"
        );
    }

    #[test]
    fn num_scan_lines_and_format() {
        let c16 = ZipCompressor::new(&hdr(), 1024, 16).unwrap();
        assert_eq!(c16.num_scan_lines(), 16);
        assert_eq!(c16.format(), Format::Xdr);
        let c1 = ZipCompressor::new(&hdr(), 1024, 1).unwrap();
        assert_eq!(c1.num_scan_lines(), 1);
    }

    #[test]
    fn zip_round_trip_empty() {
        let mut c = ZipCompressor::new(&hdr(), 0, 16).unwrap();
        let compressed = c.compress(&[], 0).unwrap();
        assert!(compressed.is_empty());
    }

    /// Multi-line block: 16 lines of distinct data round-trips identically.
    #[test]
    fn zip_round_trip_multiline_block() {
        let line_bytes = 7 * 14; // 7 px * 14 bytes/px (RGBAZ-like)
        let mut data = Vec::new();
        for line in 0..16u32 {
            for b in 0..line_bytes {
                data.push((line.wrapping_mul(31).wrapping_add(b as u32)) as u8);
            }
        }
        let mut c = ZipCompressor::new(&hdr(), line_bytes, 16).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, data.as_slice());
    }

    /// A zlib stream followed by trailing garbage must be rejected as a
    /// corrupt chunk, matching `exr_uncompress_buffer`
    /// (`OpenEXRCore/compression.c:180-197`) — C requires
    /// `actual_in_bytes == in_bytes`. The old code (`Decompressor::zlib_decompress`,
    /// the counted form) only checked the decoded length, so it silently
    /// accepted the trailing byte.
    #[test]
    fn zip_rejects_trailing_bytes_after_zlib_stream() {
        let mut data = Vec::new();
        for i in 0..64u16 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        let mut c = ZipCompressor::new(&hdr(), data.len(), 16).unwrap();
        let mut compressed = c.compress(&data, 0).unwrap().to_vec();
        assert!(
            compressed.len() < data.len(),
            "fixture must actually be zlib-compressed, not stored raw"
        );
        compressed.push(0x00); // trailing byte after the valid zlib stream
        let mut c2 = ZipCompressor::new(&hdr(), data.len(), 16).unwrap();
        assert!(
            c2.uncompress(&compressed, 0, 0).is_err(),
            "trailing bytes after the zlib stream must be CORRUPT_CHUNK"
        );
    }

    /// `Header::set_zip_level` must actually change the deflate level
    /// `ZipCompressor` uses — was previously hardwired to [`DEFAULT_ZIP_LEVEL`]
    /// regardless of the header (`internal_zip.c:368-370`'s
    /// `exr_get_zip_compression_level`, not consulted before). Uses only
    /// per-header overrides (never the process-global default) so this test
    /// has no cross-test-parallelism dependency on that shared state.
    #[test]
    fn zip_level_from_header_changes_deflate_output() {
        // Compressible but not so uniform that different levels degenerate
        // to the same bytes.
        let mut data = Vec::new();
        let mut state: u32 = 0x2468_1357;
        for _ in 0..8000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            data.push(((state >> 24) & 0x0f) as u8);
        }

        let mut hdr1 = hdr();
        hdr1.set_zip_level(1);
        let mut c1 = ZipCompressor::new(&hdr1, data.len(), 16).unwrap();
        let out1 = c1.compress(&data, 0).unwrap().to_vec();

        let mut hdr9 = hdr();
        hdr9.set_zip_level(9);
        let mut c9 = ZipCompressor::new(&hdr9, data.len(), 16).unwrap();
        let out9 = c9.compress(&data, 0).unwrap().to_vec();

        assert_ne!(
            out1, out9,
            "level 1 vs level 9 must produce different deflate bytes for this fixture"
        );

        // Round trip must still work regardless of the level used to compress.
        let restored1 = c1.uncompress(&out1, 0, 0).unwrap().to_vec();
        assert_eq!(restored1, data, "level 1 output must still decode correctly");
        let restored9 = c9.uncompress(&out9, 0, 0).unwrap().to_vec();
        assert_eq!(restored9, data, "level 9 output must still decode correctly");
    }

    /// An out-of-range header level is stored as is (C++'s
    /// `zipCompressionLevel()` reference) and, at the codec, rejected by
    /// `exr_set_zip_compression_level` (`part.c:568-577`) with the failure
    /// ignored (`ImfCompressor.cpp:55`): the stream is the one the process
    /// default level makes, not level 9's. C++ oracle (`exrmetrics -z zip -l 12`
    /// on golden_rgba.exr) gives level 4's size. Uses a per-header default-level
    /// header as the reference, so no process-global state is touched.
    #[test]
    fn out_of_range_zip_level_compresses_at_the_default_level() {
        let mut data = Vec::new();
        let mut state: u32 = 0x1357_2468;
        for _ in 0..8000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            data.push(((state >> 24) & 0x0f) as u8);
        }
        let bytes = |level: i32| {
            let mut h = hdr();
            h.set_zip_level(level);
            let mut c = ZipCompressor::new(&h, data.len(), 16).unwrap();
            let out = c.compress(&data, 0).unwrap().to_vec();
            assert_eq!(c.uncompress(&out, 0, 0).unwrap(), &data[..]);
            (h.zip_level(), out)
        };
        let default = bytes(Header::default_zip_compression_level());
        for level in [10, 12, 99, -2, i32::MIN] {
            let (stored, out) = bytes(level);
            assert_eq!(stored, level, "set_zip_level stores the value as is");
            assert_eq!(out, default.1, "level {level} must compress as the default level");
        }
        assert_ne!(bytes(9).1, default.1, "level 9 is in range and must be applied");
    }
}
