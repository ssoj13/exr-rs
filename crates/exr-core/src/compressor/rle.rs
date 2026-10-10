//! Port of OpenEXR `ImfRleCompressor.cpp` / `ImfRleCompressor.h` and the byte
//! algorithm in `OpenEXRCore/internal_rle.c` — the **RLE** scanline codec.
//!
//! # What this codec does (1:1 with C++)
//!
//! `RleCompressor` (a `Compressor` subclass with `numScanLines() == 1`) is a
//! thin wrapper: the actual byte transform lives in the OpenEXRCore C function
//! `internal_exr_apply_rle` / `internal_exr_undo_rle`, which both
//! `ImfRleCompressor` (the legacy `rleCompress`/`rleUncompress`) and the modern
//! core pipeline call.  The transform is two consecutive steps:
//!
//! 1. **reorder + predict** ([`reorder_and_predict`])
//!    — exactly the byte pre-processing OpenEXR applies before RLE *and* ZIP
//!    (shared with the ZIP codec; see [`crate::compressor::predictor`]):
//!    * **reorder**: de-interleave the bytes into two halves so that even-index
//!      bytes land in the first half and odd-index bytes in the second half
//!      (this groups the high/low bytes of 16-bit samples, the typical EXR
//!      layout, improving run lengths);
//!    * **predict**: replace every byte (after the first) by its signed delta
//!      from the previous byte, biased by `+(128+256)` and truncated to a
//!      byte — the standard OpenEXR first-difference predictor.
//! 2. **run-length encode** (`rle_compress`) — the `internal_rle_compress`
//!    format: a stream of `(count, value)` runs and literal escapes.
//!
//! Decompression reverses the order: `rle_decompress` then
//! [`unpredict_and_reorder`].
//!
//! # The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `internal_exr_apply_rle` compares the RLE output size `outb` to the raw
//! input size `srcb`; **if `outb >= srcb` it discards the RLE result and stores
//! the *raw, un-predicted, un-reordered* `packed` bytes instead**, setting the
//! emitted size to `srcb`.  [`RleCompressor::compress`] does the same: when the
//! RLE encoding is not strictly smaller than the input it returns a copy of the
//! original `data` unchanged (so the on-disk `dataSize` equals the raw size).
//!
//! On the read side the *caller* (the scanline reader, mirroring the C core's
//! `decompress_data`) detects this case purely by size: when a chunk's stored
//! `dataSize` equals the expected uncompressed size, the bytes are raw and the
//! codec is bypassed.  [`RleCompressor::uncompress`] is therefore only invoked
//! for genuinely RLE-encoded chunks and always runs the full inverse transform.
//!
//! # Format
//!
//! RLE uses the machine-independent **Xdr** byte format (it does not override
//! `Compressor::format()`), so the line buffer handed to [`compress`](RleCompressor::compress)
//! is already in the on-disk little-endian layout — no native/Xdr conversion is
//! performed here, exactly as in C++.
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfRleCompressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfRleCompressor.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_rle.c>
//! * legacy `ImfRle.cpp` (`rleCompress`/`rleUncompress`) just forwards to the
//!   same core, confirming the format is identical:
//!   <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfRle.cpp>

use crate::compressor::predictor::{reorder_and_predict, unpredict_and_reorder};
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};

// `internal_rle.c`:
//   #define MIN_RUN_LENGTH 3
//   #define MAX_RUN_LENGTH 127
const MIN_RUN_LENGTH: usize = 3;
const MAX_RUN_LENGTH: usize = 127;

// ---------------------------------------------------------------------------
// RleCompressor — the Compressor seam implementation
// ---------------------------------------------------------------------------

/// RLE scanline codec — port of `Imf::RleCompressor`.
///
/// One scan line per chunk (`num_scan_lines() == 1`), Xdr format.  Owns a
/// scratch output buffer whose contents are returned as `&[u8]` by
/// [`compress`](Compressor::compress) / [`uncompress`](Compressor::uncompress);
/// the slice is valid until the next call (mirroring the C++ member buffer).
pub struct RleCompressor {
    /// Maximum uncompressed bytes per scan line — C++ `_maxScanLineSize`.
    ///
    /// Stored for fidelity with the C++ constructor; the Rust buffers grow on
    /// demand so it is not used to pre-size them.
    #[allow(dead_code)]
    max_scan_line_size: usize,
    /// Scratch buffer holding the result of the last `compress`/`uncompress`
    /// call (the C++ output buffer).
    out: Vec<u8>,
    /// Scratch buffer for the intermediate reordered+predicted bytes (the C
    /// core's `scratch_buffer_1`).
    scratch: Vec<u8>,
}

impl RleCompressor {
    /// Create a new RLE compressor — port of
    /// `RleCompressor::RleCompressor(const Header&, size_t maxScanLineSize)`.
    ///
    /// `max_scan_line_size` is the maximum uncompressed bytes in a single scan
    /// line, as computed by `bytesPerLineTable` and passed by the factory.
    pub fn new(max_scan_line_size: usize) -> Self {
        Self {
            max_scan_line_size,
            out: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl Compressor for RleCompressor {
    /// RLE stores exactly one scan line per chunk.
    ///
    /// C++: `Compressor(hdr, EXR_COMPRESSION_RLE, maxScanLineSize, 1)` — the
    /// trailing `1` is `numScanLines`.
    fn num_scan_lines(&self) -> i32 {
        1
    }

    /// RLE uses the machine-independent Xdr format (it does not override the
    /// C++ `Compressor::format()` default).
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one scan line — port of `internal_exr_apply_rle`.
    ///
    /// Applies reorder+predict, then RLE-encodes; if the encoded size is not
    /// strictly smaller than the input, returns the raw input unchanged (the
    /// "no-shrink → store raw" rule).
    fn compress(&mut self, data: &[u8], _min_y: i32) -> Result<&[u8]> {
        let srcb = data.len();

        // C++ `internal_exr_apply_rle`: reorder_and_predict(scratch, packed, srcb).
        // `reorder_and_predict` fully overwrites all `srcb` bytes,
        // so `ensure_len` (grow/truncate, no unconditional zero-fill)
        // is sound -- see `compressor::ensure_len`'s doc.
        super::ensure_len(&mut self.scratch, srcb);
        reorder_and_predict(&mut self.scratch, data);

        // C++: outb = internal_rle_compress(compressed, cap, scratch, srcb).
        self.out.clear();
        rle_compress(&self.scratch, &mut self.out);

        // C++: if (outb >= srcb) { memcpy(compressed, packed, srcb); outb = srcb; }
        // Store the RAW (original, un-transformed) bytes when RLE did not shrink.
        if self.out.len() >= srcb {
            self.out.clear();
            self.out.extend_from_slice(data);
        }

        Ok(&self.out)
    }

    /// Uncompress one RLE-encoded scan line — port of `internal_exr_undo_rle`.
    ///
    /// Runs `rle_decompress` into the scratch buffer, then
    /// [`unpredict_and_reorder`] into the output buffer.
    ///
    /// The caller only invokes this for genuinely compressed chunks (the raw
    /// passthrough, `dataSize == uncompressedSize`, is handled one level up,
    /// exactly as the C core's `decompress_data` does).  Because the RLE stream
    /// itself does not record the uncompressed length, the inverse transform is
    /// driven entirely by the decoded byte count: `rle_decompress` appends until
    /// the input is exhausted, and that decoded length is the uncompressed size.
    ///
    /// # Errors
    /// Returns [`ExrError::Input`] if the RLE stream is malformed (a literal /
    /// run escape claims more bytes than remain).
    fn uncompress(&mut self, data: &[u8], _min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        // C++ `internal_exr_undo_rle`: rle_decompress into scratch, then
        // unpredict_and_reorder(out, scratch, decoded_len). RLE is
        // self-describing (`rle_decompress` runs until `data` is exhausted),
        // so unlike ZIP/PXR24 it never NEEDED `expected_len` to size its
        // output -- kept unused here (the trait has this
        // parameter uniformly so a caller doesn't need to know per-codec
        // which ones need it).
        self.scratch.clear();
        rle_decompress(data, &mut self.scratch)?;

        // `unpredict_and_reorder` fully overwrites all `outsz` bytes
        // -- same argument as the compress side above.
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
// RLE core encode — port of `internal_rle_compress` (internal_rle.c lines 17-63)
// ---------------------------------------------------------------------------

/// Run-length encode `src` into `out` — port of `internal_rle_compress`.
///
/// The stream alternates between two record kinds, distinguished by the sign of
/// the leading count byte (read back as a signed `i8`):
/// * **run** (`count >= 0`): `count` then one `value` byte → `count + 1` copies
///   of `value` (emitted when a byte repeats at least `MIN_RUN_LENGTH` times);
/// * **literal** (`count < 0`): `-count` then `-count` raw bytes copied
///   verbatim (emitted for incompressible stretches).
///
/// `out` grows as needed.  The C version stops early once its fixed output
/// buffer fills (`if (outb >= outbytes) break;`); here the buffer is unbounded,
/// and the codec's "no-shrink → store raw" check at the call site supersedes
/// that early-out — so we always encode the entire input.  For any input the
/// byte stream produced is identical to C's for the bytes it emits.
///
/// C reference:
/// ```c
/// while (runs < end) {
///     uint8_t curcount = 0;
///     while (rune < end && *runs == *rune && curcount < MAX_RUN_LENGTH) { ++rune; ++curcount; }
///     if (curcount >= (MIN_RUN_LENGTH - 1)) {
///         cbuf[outb++] = (int8_t)curcount; cbuf[outb++] = *runs; runs = rune;
///     } else {
///         ++curcount;
///         while (rune < end &&
///                ((rune+1 >= end || *rune != *(rune+1)) ||
///                 (rune+2 >= end || *(rune+1) != *(rune+2))) &&
///                curcount < MAX_RUN_LENGTH) { ++curcount; ++rune; }
///         cbuf[outb++] = (int8_t)(-((int)curcount));
///         while (runs < rune) cbuf[outb++] = *runs++;
///     }
///     ++rune;
/// }
/// ```
///
/// The C pointers `runs` (run start) and `rune` (run-end scan cursor) are
/// modelled by the `usize` indices `runs` and `rune`.  `rune` starts one past
/// `runs` and is post-incremented at the end of every iteration, exactly as in
/// the C `++rune` after each branch.
pub(crate) fn rle_compress(src: &[u8], out: &mut Vec<u8>) {
    let n = src.len();
    if n == 0 {
        return;
    }

    let mut runs = 0usize; // C `runs` — start of the current run
    let mut rune = 1usize; // C `rune` — scan cursor, starts at runs + 1

    while runs < n {
        let mut curcount: usize = 0;

        // Count a repeat run of src[runs]: advance `rune` while it matches and
        // the run is shorter than MAX_RUN_LENGTH.
        while rune < n && src[runs] == src[rune] && curcount < MAX_RUN_LENGTH {
            rune += 1;
            curcount += 1;
        }

        if curcount >= MIN_RUN_LENGTH - 1 {
            // Compressible run: emit (count, value).  `count` fits in an i8
            // because curcount <= MAX_RUN_LENGTH (127); `as u8` matches the C
            // `(int8_t)curcount` store.
            out.push(curcount as u8);
            out.push(src[runs]);
            runs = rune;
        } else {
            // Incompressible literal: include src[runs] (the `++curcount`) then
            // keep extending while the next bytes do NOT begin a >=3-run.
            curcount += 1;
            while rune < n
                && ((rune + 1 >= n || src[rune] != src[rune + 1])
                    || (rune + 2 >= n || src[rune + 1] != src[rune + 2]))
                && curcount < MAX_RUN_LENGTH
            {
                curcount += 1;
                rune += 1;
            }
            // Negative count byte: C `(int8_t)(-(int)curcount)`.  `curcount` is
            // 1..=127, so `-(curcount as i32) as i8 as u8` reproduces the exact
            // two's-complement byte.
            out.push((-(curcount as i32)) as i8 as u8);
            // Copy the literal bytes [runs, rune).
            while runs < rune {
                out.push(src[runs]);
                runs += 1;
            }
        }

        // C: `++rune;` at the end of every iteration.  After a run branch,
        // `runs == rune` (the matched bytes were consumed) and this sets up the
        // next scan; after a literal branch, `runs == rune` likewise.
        rune += 1;
    }
}

// ---------------------------------------------------------------------------
// RLE core decode — port of `internal_rle_decompress` (internal_rle.c 128-168)
// ---------------------------------------------------------------------------

/// Run-length decode `src` into `out`, appending every decoded byte — port of
/// `internal_rle_decompress`.
///
/// Inverts [`rle_compress`]: a negative leading byte introduces `-count`
/// literal bytes; a non-negative leading byte introduces a run of `count + 1`
/// copies of the following value byte.
///
/// C reference:
/// ```c
/// while (unpackbytes < packsz) {
///     if (*in < 0) {
///         uint64_t count = (uint64_t)(-((int)*in++));
///         ++unpackbytes;
///         if (unpackbytes + count > packsz) return 0;
///         memcpy(dst, in, count); in += count; dst += count;
///         unpackbytes += count;
///     } else {
///         uint64_t count = (uint64_t)(*in++);
///         if (unpackbytes + 2 > packsz) return 0;
///         unpackbytes += 2;
///         ++count;
///         memset(dst, *in, count); dst += count;
///         ++in;
///     }
/// }
/// ```
///
/// The C bounds checks against `packsz` (the *input* size) are reproduced as
/// `ExrError::Input` errors so a corrupt stream is rejected, never a panic.
///
/// # Errors
/// Returns [`ExrError::Input`] when a literal escape or run header claims more
/// input bytes than remain.
pub(crate) fn rle_decompress(src: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let packsz = src.len();
    let mut in_pos = 0usize; // C `in` cursor / `unpackbytes` (they advance in lockstep)

    while in_pos < packsz {
        let lead = src[in_pos] as i8;
        if lead < 0 {
            // Literal escape: -count raw bytes follow the count byte.
            let count = (-(lead as i32)) as usize;
            in_pos += 1; // consumed the count byte (C `++unpackbytes`)
            // C: if (unpackbytes + count > packsz) return 0;
            if in_pos + count > packsz {
                return Err(ExrError::Input(
                    "RLE literal run extends past end of compressed data".into(),
                ));
            }
            out.extend_from_slice(&src[in_pos..in_pos + count]);
            in_pos += count;
        } else {
            // Run: a single value byte follows the count byte; emit count+1 copies.
            // C requires both the count byte AND the value byte to be present.
            if in_pos + 2 > packsz {
                return Err(ExrError::Input(
                    "RLE run header extends past end of compressed data".into(),
                ));
            }
            let count = lead as usize + 1; // C `++count`
            let value = src[in_pos + 1];
            for _ in 0..count {
                out.push(value);
            }
            in_pos += 2;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct port-level round trip of the two byte transforms (no Compressor).
    fn transform_round_trip(data: &[u8]) {
        let n = data.len();
        let mut scratch = vec![0u8; n];
        reorder_and_predict(&mut scratch, data);

        let mut encoded = Vec::new();
        rle_compress(&scratch, &mut encoded);

        let mut decoded = Vec::new();
        rle_decompress(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded.len(), n, "rle decode length mismatch");

        let mut restored = vec![0u8; n];
        unpredict_and_reorder(&mut restored, &mut decoded);
        assert_eq!(restored, data, "reorder/predict round trip mismatch");
    }

    #[test]
    fn reorder_predict_round_trip_rle_friendly() {
        // Long runs (compresses well) — exercises the run branch.
        let data: Vec<u8> = vec![5u8; 200];
        transform_round_trip(&data);
    }

    #[test]
    fn reorder_predict_round_trip_mixed() {
        let data: Vec<u8> = vec![1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 9, 8, 7];
        transform_round_trip(&data);
    }

    #[test]
    fn reorder_predict_round_trip_single_byte() {
        transform_round_trip(&[0x42]);
    }

    #[test]
    fn reorder_predict_round_trip_empty() {
        transform_round_trip(&[]);
    }

    #[test]
    fn reorder_predict_round_trip_odd_length() {
        let data: Vec<u8> = (0..=250u8).step_by(3).collect();
        transform_round_trip(&data);
    }

    /// Full codec round trip: compress → uncompress = identity, for both the
    /// compressible and the incompressible (no-shrink → raw) paths.
    #[test]
    fn codec_round_trip_compressible() {
        // 16-bit-style data with strong runs after reorder: high bytes constant.
        let mut data = Vec::new();
        for i in 0..100u16 {
            // low byte varies, high byte constant 0 → reorder groups them.
            data.extend_from_slice(&i.to_le_bytes());
        }
        let mut c = RleCompressor::new(data.len());
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        // This pattern shrinks (the constant high-byte half RLE-compresses).
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
    fn codec_round_trip_all_same() {
        let data = vec![0xABu8; 512];
        let mut c = RleCompressor::new(data.len());
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        assert!(compressed.len() < data.len(), "all-same must compress");
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, data.as_slice());
    }

    /// Incompressible (pseudo-random) data must trigger the "no-shrink → store
    /// raw" path: `compress` returns the bytes unchanged.  Such a chunk is read
    /// back by the *caller* via the `dataSize == uncompressedSize` raw rule, not
    /// by `uncompress`, so we only assert the raw-store behaviour here.
    #[test]
    fn codec_no_shrink_stores_raw() {
        // A simple LCG gives reproducible high-entropy bytes that RLE cannot
        // shrink; after reorder+predict the deltas are still ~random.
        let mut data = Vec::with_capacity(256);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..256 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            data.push((state >> 16) as u8);
        }
        let mut c = RleCompressor::new(data.len());
        let compressed = c.compress(&data, 0).unwrap();
        assert_eq!(
            compressed,
            data.as_slice(),
            "incompressible data must be stored raw (== original bytes)"
        );
    }

    #[test]
    fn num_scan_lines_is_one_and_format_xdr() {
        let c = RleCompressor::new(1024);
        assert_eq!(c.num_scan_lines(), 1);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn decompress_rejects_truncated_literal() {
        // Negative count byte (-3 = literal of 3) but only 1 byte follows.
        let bad = [(-3i32) as i8 as u8, 0xAA];
        let mut out = Vec::new();
        assert!(rle_decompress(&bad, &mut out).is_err());
    }

    #[test]
    fn decompress_rejects_truncated_run() {
        // Non-negative count (run) but no value byte follows.
        let bad = [5u8];
        let mut out = Vec::new();
        assert!(rle_decompress(&bad, &mut out).is_err());
    }
}
