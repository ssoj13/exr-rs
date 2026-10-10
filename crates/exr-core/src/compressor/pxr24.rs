//! Port of OpenEXR `ImfPxr24Compressor.cpp` / `.h` and the byte algorithm in
//! `OpenEXRCore/internal_pxr24.c` — the **PXR24** scanline codec (Loren
//! Carpenter's lossy 24-bit-float compressor, contributed by Pixar).
//!
//! # What this codec does (1:1 with C++)
//!
//! PXR24 preprocesses one chunk of pixel data per channel, per line, to reduce
//! entropy, then deflates the result with zlib.  Compression of HALF and UINT
//! channels is **lossless**; FLOAT channels are **lossy**: each 32-bit float is
//! converted to 24 bits by rounding the significand to 15 bits
//! ([`float_to_float24`]).  The full per-chunk transform, mirroring
//! `apply_pxr24_impl` (`internal_pxr24.c` lines 89-241):
//!
//! 1. **per-channel, per-line de-interleave + first-difference predict** — the
//!    chunk's pixel data is laid out line-by-line, channel-by-channel within a
//!    line (exactly how [`crate::scanline::OutputFile`] gathers it).  For each
//!    line of each channel the codec converts every sample to an unsigned
//!    integer (UINT: identity; HALF: re-interpret the 16 bits; FLOAT: round to
//!    24 bits), replaces it with the difference from its left neighbour
//!    (`diff = pixel - prevPixel`, wrapping), and **byte-splits** that diff so
//!    that all most-significant bytes land in one contiguous block, the next
//!    bytes in the following block, and so on:
//!    * UINT → 4 byte-planes (`diff>>24`, `>>16`, `>>8`, `diff`);
//!    * HALF → 2 byte-planes (`diff>>8`, `diff`);
//!    * FLOAT → 3 byte-planes (`diff>>16`, `>>8`, `diff`) — note the lowest
//!      8 bits of the 24-bit value (= bits 0-7 of the rounded significand) are
//!      dropped, which is what makes FLOAT lossy.
//!
//!    `prevPixel` resets to 0 at the start of **every channel line**.
//! 2. **deflate (zlib)** — `exr_compress_buffer(context, -1, …)` (
//!    `internal_pxr24.c:219-226`), which resolves the hardcoded `-1` to the
//!    **process-global** default zip level (`exr_get_default_zip_compression_level`,
//!    `compression.c:89-117`; [`crate::header::Header::default_zip_compression_level`]),
//!    falling back further to `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4`
//!    ([`crate::compressor::zip::DEFAULT_ZIP_LEVEL`]) only when that default is
//!    itself unset. Unlike ZIP, PXR24 **never** honours a per-header
//!    `zipCompressionLevel()` override — the C source passes a literal `-1`,
//!    not the header's level. Our
//!    [`deflate_rs::Compressor::zlib_compress`] is a 1:1 port of libdeflate, so
//!    at whichever level this resolves to, it produces **byte-identical**
//!    zlib streams to C++.
//!
//! Decompression reverses the order: zlib inflate, then per-channel-per-line
//! un-byte-split + running-sum reconstruct (`undo_pxr24_impl`, lines 260-401).
//! For FLOAT the dropped low byte is reconstructed as 0 (`diff>>24 | >>16 |
//! >>8` with no `<<0` term), so the decoded float is the 24-bit-quantised value.
//!
//! # PXR24 uses its OWN predictor (NOT the shared ZIP/RLE byte predictor)
//!
//! Unlike RLE and ZIP — which share
//! [`crate::compressor::predictor`] (a flat de-interleave + biased
//! first-difference over the *whole* line buffer) — PXR24 runs a *different*,
//! channel-aware transform: the difference is taken between successive **pixels
//! of the same channel** (not successive bytes), it is unbiased (`pixel -
//! prev`, no `+128`), and the byte-split groups bytes by significance across
//! the whole channel-line.  This codec therefore implements its predictor
//! inline and does not call the shared module.
//!
//! # The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `apply_pxr24_impl`: after deflating, **if `compbufsz > packed_bytes` it
//! copies the original `packed_buffer` and sets `compbufsz = packed_bytes`.**
//! `packed_bytes` is the size of the *original* (pre-transform) line buffer.
//! We replicate this: when the zlib stream is **strictly larger** than the raw
//! input we return a copy of the original `data` unchanged.  On disk this means
//! `dataSize == uncompressedSize`, which is exactly how the reader detects the
//! raw passthrough (same convention as ZIP).
//!
//! # Format
//!
//! PXR24 uses the machine-independent **Xdr** byte format (it does not override
//! `Compressor::format()`), so the line buffer handed to
//! [`compress`](Pxr24Compressor::compress) is already in the on-disk
//! little-endian layout.  The per-sample reads below therefore decode the
//! little-endian wire bytes; the byte-split output is the codec's own
//! representation (independent of host endianness).
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPxr24Compressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPxr24Compressor.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_pxr24.c>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/compression.c>

use crate::attr::PixelType;
use crate::compressor::zip::DEFAULT_ZIP_LEVEL;
use crate::compressor::zlib::{compress_into, new_deflate_compressor, uncompress_into, LowCompressor, LowDecompressor};
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};
use imath_rs::Box2i;

use imath_rs::{divp, modp};

// ---------------------------------------------------------------------------
// float_to_float24 — port of `float_to_float24` (internal_pxr24.c lines 17-85)
// ---------------------------------------------------------------------------

/// Convert a 32-bit float to a 24-bit float (packed in the low 24 bits of a
/// `u32`), rounding the significand to 15 bits — port of `float_to_float24`.
///
/// This is the **lossy** step of PXR24's FLOAT path.  The reverse conversion is
/// a simple 8-bit left shift (`pixel24 << 8` reinterpreted as `f32`), performed
/// implicitly during decompression by leaving the low byte at 0.
///
/// The bit-twiddling mirrors `internal_pxr24.c` exactly (NaN/Inf preservation,
/// round-to-nearest with the FLT_MAX overflow guard).
pub fn float_to_float24(f: f32) -> u32 {
    let i = f.to_bits();

    // Disassemble into sign, exponent, significand.
    let s = i & 0x8000_0000;
    let e = i & 0x7f80_0000;
    let m = i & 0x007f_ffff;

    let result: u32 = if e == 0x7f80_0000 {
        if m != 0 {
            // F is a NaN; preserve the sign bit and the 15 leftmost bits of the
            // significand, with one exception: if the 15 leftmost bits are all
            // zero, the NaN would turn into an infinity, so set at least one bit.
            let m = m >> 8;
            (e >> 8) | m | u32::from(m == 0)
        } else {
            // F is an infinity.
            e >> 8
        }
    } else {
        // F is finite, round the significand to 15 bits.
        let r = ((e | m) + (m & 0x0000_0080)) >> 8;
        if r >= 0x7f8000 {
            // F was close to FLT_MAX and the significand was rounded up,
            // overflowing the exponent.  Avoid it by truncating instead.
            (e | m) >> 8
        } else {
            r
        }
    };

    (s >> 8) | result
}

// ---------------------------------------------------------------------------
// ChannelInfo — the per-channel layout the codec needs (from the Header)
// ---------------------------------------------------------------------------

/// Per-channel layout snapshot captured from the header at construction —
/// mirrors the fields of `exr_coding_channel_info_t` that `apply_pxr24_impl`
/// reads (`data_type`, `width`, `y_samples`).
///
/// Channels are stored in the file's channel-list order (alphabetical), which
/// is the order the line buffer is gathered in, so the codec consumes the line
/// buffer in lock-step with the writer/reader.
#[derive(Clone, Copy, Debug)]
struct ChannelInfo {
    /// Sample type on disk (drives the byte-split width and the conversion).
    type_: PixelType,
    /// Horizontal sub-sampling factor (`curc->x_samples`).
    x_sampling: i32,
    /// Vertical sub-sampling factor (`curc->y_samples`).
    y_sampling: i32,
}

// ---------------------------------------------------------------------------
// Pxr24Compressor — the Compressor seam implementation
// ---------------------------------------------------------------------------

/// PXR24 scanline codec — port of `Imf::Pxr24Compressor`.
///
/// `num_scan_lines()` is **16** (`getCompressionNumScanlines(PXR24) == 16`).
/// Owns scratch buffers whose contents are returned as `&[u8]` by
/// [`compress`](Compressor::compress) / [`uncompress`](Compressor::uncompress);
/// the slice is valid until the next call (mirroring the C++ member buffer).
///
/// Because PXR24's transform is channel-aware, the codec captures the channel
/// list and data-window bounds from the header at construction (the C core gets
/// these from `encode->channels` / `encode->chunk`); `compress`/`uncompress`
/// then drive the per-line, per-channel loop from the chunk's `min_y` and the
/// captured layout.
pub struct Pxr24Compressor {
    /// Scan lines per chunk — `numScanLines()` (16 for PXR24).
    num_scan_lines: i32,
    /// Per-channel layout in file (channel-list) order.
    channels: Vec<ChannelInfo>,
    /// Data-window minimum x — used to compute each channel's per-line width.
    min_x: i32,
    /// Data-window maximum x.
    max_x: i32,
    /// Data-window maximum y — clamps the final (possibly partial) block.
    ///
    /// Note PXR24's per-line loop indexes `cury = min_y + y` where `min_y` is
    /// the *chunk's* start y (passed to `compress`/`uncompress`), not the
    /// data-window minimum, so the data-window `min_y` is not stored here.
    max_y: i32,
    /// Maximum uncompressed bytes in a whole chunk — the decode capacity.
    max_chunk_size: usize,
    /// Scratch buffer holding the result of the last call (the C++ output buffer).
    out: Vec<u8>,
    /// Scratch buffer for the intermediate byte-split bytes (the C core's
    /// `scratch_buffer_1`).
    scratch: Vec<u8>,
    /// Persistent low-level deflate compressor — constructed once here
    /// instead of per `compress()` call (see
    /// `compressor::zlib`'s module doc and `ZipCompressor`'s identical
    /// field). Its level: PXR24 always calls `exr_compress_buffer` with a
    /// hardcoded `level = -1` (`internal_pxr24.c:219-226`, v3.4.15) -- it
    /// NEVER reads a per-header override, only the process-global default
    /// (`exr_get_default_zip_compression_level`, `compression.c:89-117`),
    /// which itself falls back to [`DEFAULT_ZIP_LEVEL`] == 4 when still
    /// unset. So the level resolved in [`Pxr24Compressor::new`] is
    /// [`Header::default_zip_compression_level`], not `header.zip_level()`
    /// (`header.zip_level()` would incorrectly let a *per-header*
    /// override -- which PXR24 must ignore -- leak in).
    deflate_compressor: LowCompressor,
    /// Persistent low-level zlib decompressor, reused for symmetry with
    /// `deflate_compressor`.
    deflate_decompressor: LowDecompressor,
}

impl Pxr24Compressor {
    /// Create a new PXR24 compressor — port of `Pxr24Compressor::Pxr24Compressor`.
    ///
    /// `max_scan_line_size` is the maximum uncompressed bytes in a single scan
    /// line (used to size the decode buffer); `num_scan_lines` is 16.  The
    /// channel layout and data-window bounds are captured from `header`.
    ///
    /// # Errors
    /// Propagates [`crate::header::Header::channels`] / `data_window` errors.
    pub fn new(
        header: &crate::header::Header,
        max_scan_line_size: usize,
        num_scan_lines: i32,
    ) -> Result<Self> {
        let dw = header.data_window()?;
        let channel_list = header.channels()?;

        // Capture the channel layout in file order (alphabetical) — the same
        // order the line buffer is gathered in.
        let channels: Vec<ChannelInfo> = channel_list
            .iter()
            .map(|(_, ch)| ChannelInfo {
                type_: ch.type_,
                x_sampling: ch.x_sampling,
                y_sampling: ch.y_sampling,
            })
            .collect();

        let max_chunk_size = max_scan_line_size.saturating_mul(num_scan_lines.max(1) as usize);
        let level = {
            let process_default = crate::header::Header::default_zip_compression_level();
            if process_default < 0 { DEFAULT_ZIP_LEVEL } else { process_default }
        };
        let deflate_compressor = new_deflate_compressor(level)?;

        Ok(Self {
            num_scan_lines,
            channels,
            min_x: dw.min.x,
            max_x: dw.max.x,
            max_y: dw.max.y,
            max_chunk_size,
            out: Vec::new(),
            scratch: Vec::new(),
            deflate_compressor,
            deflate_decompressor: LowDecompressor::new(),
        })
    }

    /// Number of sampled scanlines in the block that starts at `block_min_y`
    /// (chunk `min_y`) — `min(block_min_y + numScanLines - 1, max_y) -
    /// block_min_y + 1`.  This equals the writer's `block_lines`, so the codec
    /// iterates exactly the lines the gather wrote.
    fn block_height(&self, block_min_y: i32) -> i32 {
        let block_max_y = (block_min_y + self.num_scan_lines - 1).min(self.max_y);
        block_max_y - block_min_y + 1
    }

    /// Per-channel sample count for a channel with horizontal sub-sampling
    /// `x_sampling`, over the data window's x range — `divp(max_x, xs) -
    /// divp(min_x, xs) + 1` (matches the writer's `d_max_x - d_min_x + 1`).
    fn channel_width(&self, x_sampling: i32) -> usize {
        let d_min = divp(self.min_x, x_sampling);
        let d_max = divp(self.max_x, x_sampling);
        (d_max - d_min + 1) as usize
    }
}

impl Compressor for Pxr24Compressor {
    /// 16 scan lines per chunk for PXR24.
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// PXR24 uses the machine-independent Xdr format (it does not override the
    /// C++ `Compressor::format()` default).
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one chunk — port of `internal_exr_apply_pxr24` / `apply_pxr24_impl`.
    ///
    /// Runs the per-channel-per-line byte-split + first-difference predictor
    /// into the scratch buffer, deflates at `self.level` (the process-global
    /// default, usually 4 — see [`Pxr24Compressor::new`]), then applies the
    /// "no-shrink → store raw" rule.
    ///
    /// # Errors
    /// * [`ExrError::Logic`] if the gathered line buffer's length disagrees with
    ///   the channel layout (would indicate a writer/codec mismatch — never a
    ///   panic).
    /// * [`ExrError::Arg`] if the deflate compressor cannot be created.
    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]> {
        let srcb = data.len();
        if srcb == 0 {
            self.out.clear();
            return Ok(&self.out);
        }

        let block_height = self.block_height(min_y);

        // The byte-split output is the same total length as the FLOAT-quantised
        // input: UINT 4B/px, HALF 2B/px, FLOAT 3B/px.  Size the scratch buffer
        // by walking the same loop the transform uses, so it matches the C
        // core's `nOut`. The loop below writes every byte in `[0, n_out)` and
        // only `self.scratch[..n_out]` is ever read afterward (n_out <= srcb,
        // since FLOAT shrinks 4B/px to 3B/px) -- any stale tail beyond n_out
        // is never observed, so `ensure_len` is sound.
        super::ensure_len(&mut self.scratch, srcb);

        // `nOut` is the running length of the byte-split output; `n_in` walks
        // the input line buffer.
        let mut n_out = 0usize;
        let mut n_in = 0usize;

        for y in 0..block_height {
            let cury = min_y + y;
            for ci in &self.channels {
                // C: if (curc->height == 0 || (y_samples>1 && cury%y_samples!=0)) continue;
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let w = self.channel_width(ci.x_sampling);

                match ci.type_ {
                    PixelType::Uint => {
                        // 4 byte-planes: out[0..w], out[w..2w], out[2w..3w], out[3w..4w].
                        let in_bytes = w * 4;
                        if n_in + in_bytes > srcb || n_out + in_bytes > self.scratch.len() {
                            return Err(ExrError::Logic(
                                "PXR24 compress: UINT channel overruns the line buffer".into(),
                            ));
                        }
                        let (p0, p1, p2, p3) =
                            (n_out, n_out + w, n_out + 2 * w, n_out + 3 * w);
                        let mut prev: u32 = 0;
                        for x in 0..w {
                            let off = n_in + x * 4;
                            // Xdr (little-endian) wire bytes → host u32.
                            let pixel = u32::from_le_bytes([
                                data[off],
                                data[off + 1],
                                data[off + 2],
                                data[off + 3],
                            ]);
                            let diff = pixel.wrapping_sub(prev);
                            prev = pixel;
                            self.scratch[p0 + x] = (diff >> 24) as u8;
                            self.scratch[p1 + x] = (diff >> 16) as u8;
                            self.scratch[p2 + x] = (diff >> 8) as u8;
                            self.scratch[p3 + x] = diff as u8;
                        }
                        n_in += in_bytes;
                        n_out += in_bytes;
                    }
                    PixelType::Half => {
                        // 2 byte-planes.
                        let in_bytes = w * 2;
                        if n_in + in_bytes > srcb || n_out + in_bytes > self.scratch.len() {
                            return Err(ExrError::Logic(
                                "PXR24 compress: HALF channel overruns the line buffer".into(),
                            ));
                        }
                        let (p0, p1) = (n_out, n_out + w);
                        let mut prev: u32 = 0;
                        for x in 0..w {
                            let off = n_in + x * 2;
                            let pixel = u16::from_le_bytes([data[off], data[off + 1]]) as u32;
                            let diff = pixel.wrapping_sub(prev);
                            prev = pixel;
                            self.scratch[p0 + x] = (diff >> 8) as u8;
                            self.scratch[p1 + x] = diff as u8;
                        }
                        n_in += in_bytes;
                        n_out += in_bytes;
                    }
                    PixelType::Float => {
                        // 3 byte-planes; input is 4B/px (f32) but output is 3B/px.
                        let in_bytes = w * 4;
                        let out_bytes = w * 3;
                        if n_in + in_bytes > srcb || n_out + out_bytes > self.scratch.len() {
                            return Err(ExrError::Logic(
                                "PXR24 compress: FLOAT channel overruns the line buffer".into(),
                            ));
                        }
                        let (p0, p1, p2) = (n_out, n_out + w, n_out + 2 * w);
                        let mut prev: u32 = 0;
                        for x in 0..w {
                            let off = n_in + x * 4;
                            let bits = u32::from_le_bytes([
                                data[off],
                                data[off + 1],
                                data[off + 2],
                                data[off + 3],
                            ]);
                            let f = f32::from_bits(bits);
                            let pixel24 = float_to_float24(f);
                            let diff = pixel24.wrapping_sub(prev);
                            prev = pixel24;
                            // C drops the low 8 bits (lossy): only >>16, >>8, >>0.
                            self.scratch[p0 + x] = (diff >> 16) as u8;
                            self.scratch[p1 + x] = (diff >> 8) as u8;
                            self.scratch[p2 + x] = diff as u8;
                        }
                        n_in += in_bytes;
                        n_out += out_bytes;
                    }
                    PixelType::NumPixelTypes => {
                        return Err(ExrError::Arg("PXR24 compress: unknown pixel type".into()));
                    }
                }
            }
        }

        // The line buffer must have been fully consumed.
        if n_in != srcb {
            return Err(ExrError::Logic(format!(
                "PXR24 compress: consumed {n_in} of {srcb} line-buffer bytes \
                 (channel layout mismatch)"
            )));
        }

        // Deflate the byte-split output at self.level (process-global default;
        // usually 4, same as ZIP's own default resolution).
        compress_into(&mut self.deflate_compressor, &self.scratch[..n_out], &mut self.out)?;

        // Store raw when the deflate stream does NOT shrink below the
        // uncompressed size.  The C core writes raw on `compbufsz > packed_bytes`
        // (internal_pxr24.c:230) and relies on the entropy coder never producing
        // `compbufsz == packed_bytes`; but our deflate-rs CAN land exactly on
        // `compressed.len() == srcb` (observed: HALF `Noise` 7×17, a 16-row chunk
        // deflates to exactly 896 == 896).  The scanline reader's raw-passthrough
        // rule is `dataSize == uncompressedSize ⇒ raw` (its `==` short-circuit,
        // mirroring the C core), so a compressed chunk MUST be strictly smaller
        // than the uncompressed size or the reader will mis-read the zlib bytes as
        // raw pixels.  Hence `>=`, not `>`: emit compressed only when it genuinely
        // shrinks.  Storing raw at equality is also strictly better (a same-size
        // zlib wrapper buys nothing) and a C++ reader decodes our raw chunk
        // identically (its `==`-is-raw rule).  `srcb` is the original
        // (pre-transform) line-buffer size, i.e. the reader's `uncompressedSize`.
        if self.out.len() >= srcb {
            self.out.clear();
            self.out.extend_from_slice(data);
        }

        Ok(&self.out)
    }

    /// Uncompress one PXR24 chunk — port of `internal_exr_undo_pxr24` /
    /// `undo_pxr24_impl`.
    ///
    /// Inflates the zlib stream, then reverses the per-channel-per-line
    /// byte-split + running-sum reconstruct into the output buffer.
    ///
    /// The caller only invokes this for genuinely compressed chunks; the raw
    /// passthrough (`dataSize == uncompressedSize`) is handled one level up,
    /// exactly as the C core short-circuits.
    ///
    /// # Errors
    /// * [`ExrError::Input`] if the zlib stream is malformed, or the inflated
    ///   byte-split data is too short for the channel layout (corrupt chunk).
    fn uncompress(&mut self, data: &[u8], min_y: i32, expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }

        let block_height = self.block_height(min_y);

        // Inflate into the scratch buffer.  The decode capacity is the byte-split
        // size, which is <= the uncompressed (decoded-pixel) size: FLOAT is
        // 3B/px in the stream but 4B/px decoded.  `max_chunk_size` (decoded
        // size) is therefore a safe upper bound for the inflated stream too --
        // and so is `expected_len` (the caller's exact
        // per-chunk target), so `.max()` of both is a no-op for
        // scanline/tiled (where `max_chunk_size` already covers every real
        // chunk) and gives deep's per-chunk-varying sizes a correct cap too.
        let cap = expected_len.max(self.max_chunk_size).max(1);
        // Checked entry point — reject trailing bytes after the zlib
        // stream, matching `exr_uncompress_buffer`. Decodes directly into
        // `self.scratch` -- no intermediate `decoded` Vec + copy.
        uncompress_into(
            &mut self.deflate_decompressor,
            data,
            &mut self.scratch,
            cap,
            "PXR24",
            crate::compressor::zlib::ShortOutputPolicy::RequireExact,
        )?;
        let stream_len = self.scratch.len();

        // Reconstruct each channel line.  `n_dec` walks the byte-split stream;
        // the output is written contiguously line-by-line, channel-by-channel.
        self.out.clear();
        let mut n_dec = 0usize;

        for y in 0..block_height {
            let cury = min_y + y;
            for ci in &self.channels {
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let w = self.channel_width(ci.x_sampling);

                match ci.type_ {
                    PixelType::Uint => {
                        // 4 byte-planes consumed; 4B/px written.
                        if n_dec + w * 4 > stream_len {
                            return Err(ExrError::Input(
                                "PXR24 uncompress: UINT channel truncated".into(),
                            ));
                        }
                        let (p0, p1, p2, p3) =
                            (n_dec, n_dec + w, n_dec + 2 * w, n_dec + 3 * w);
                        let mut pixel: u32 = 0;
                        for x in 0..w {
                            let diff = ((self.scratch[p0 + x] as u32) << 24)
                                | ((self.scratch[p1 + x] as u32) << 16)
                                | ((self.scratch[p2 + x] as u32) << 8)
                                | (self.scratch[p3 + x] as u32);
                            pixel = pixel.wrapping_add(diff);
                            // Write as Xdr (little-endian) wire bytes.
                            self.out.extend_from_slice(&pixel.to_le_bytes());
                        }
                        n_dec += w * 4;
                    }
                    PixelType::Half => {
                        if n_dec + w * 2 > stream_len {
                            return Err(ExrError::Input(
                                "PXR24 uncompress: HALF channel truncated".into(),
                            ));
                        }
                        let (p0, p1) = (n_dec, n_dec + w);
                        let mut pixel: u32 = 0;
                        for x in 0..w {
                            let diff = ((self.scratch[p0 + x] as u32) << 8)
                                | (self.scratch[p1 + x] as u32);
                            pixel = pixel.wrapping_add(diff);
                            self.out
                                .extend_from_slice(&((pixel as u16).to_le_bytes()));
                        }
                        n_dec += w * 2;
                    }
                    PixelType::Float => {
                        // 3 byte-planes consumed; 4B/px written (low byte = 0).
                        if n_dec + w * 3 > stream_len {
                            return Err(ExrError::Input(
                                "PXR24 uncompress: FLOAT channel truncated".into(),
                            ));
                        }
                        let (p0, p1, p2) = (n_dec, n_dec + w, n_dec + 2 * w);
                        let mut pixel: u32 = 0;
                        for x in 0..w {
                            // C reconstructs diff = b0<<24 | b1<<16 | b2<<8 (no <<0):
                            // the dropped low byte stays 0, so the float is the
                            // 24-bit-quantised value.
                            let diff = ((self.scratch[p0 + x] as u32) << 24)
                                | ((self.scratch[p1 + x] as u32) << 16)
                                | ((self.scratch[p2 + x] as u32) << 8);
                            pixel = pixel.wrapping_add(diff);
                            self.out.extend_from_slice(&pixel.to_le_bytes());
                        }
                        n_dec += w * 3;
                    }
                    PixelType::NumPixelTypes => {
                        return Err(ExrError::Arg(
                            "PXR24 uncompress: unknown pixel type".into(),
                        ));
                    }
                }
            }
        }

        Ok(&self.out)
    }

    /// Compress one tile, encoded against the tile's own pixel rectangle.
    ///
    /// PXR24 derives its per-channel width from `min_x`/`max_x` and clamps the
    /// final block to `max_y`; for a tile these are the tile's X/Y extent.
    /// Mirrors C++ `compressTile` feeding the tile's `Box2i range` into the
    /// encode step.  Sets the rectangle fields from `range`, then runs the
    /// identical compress path at `range.min.y` (fresh each tile call).
    fn compress_tile(&mut self, data: &[u8], range: Box2i) -> Result<&[u8]> {
        self.min_x = range.min.x;
        self.max_x = range.max.x;
        self.max_y = range.max.y;
        self.compress(data, range.min.y)
    }

    /// Uncompress one tile, decoded against the tile's own pixel rectangle —
    /// the exact inverse of [`compress_tile`](Self::compress_tile).
    fn uncompress_tile(&mut self, data: &[u8], range: Box2i, expected_len: usize) -> Result<&[u8]> {
        self.min_x = range.min.x;
        self.max_x = range.max.x;
        self.max_y = range.max.y;
        self.uncompress(data, range.min.y, expected_len)
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
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::header::Header;
    use imath_rs::{Half, V2f};

    /// Reverse of `float_to_float24`: the float a PXR24 decode reproduces for a
    /// given source float (the 24-bit-quantised value).  Used by tests to
    /// predict the lossy FLOAT result deterministically.
    fn f24_quantised_bits(f: f32) -> u32 {
        float_to_float24(f) << 8
    }

    /// Build a header with the given channels (in insertion → sorted order),
    /// `w`x`h`, PXR24, INCREASING_Y.
    fn make_header(w: i32, h: i32, chans: &[(&str, PixelType)]) -> Header {
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Pxr24,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for (name, ty) in chans {
            cl.insert(*name, Channel::new(*ty, 1, 1, false)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr
    }

    /// Build an Xdr (little-endian) line buffer for `block_height` lines of the
    /// given channels, with deterministic per-(channel,x,y) sample values.
    /// Returns the buffer.  HALF stores `Half::from(v).bits()`, FLOAT stores
    /// `v.to_bits()`, UINT stores the value cast to u32.
    fn build_line_buffer(
        w: i32,
        block_height: i32,
        chans: &[(&str, PixelType)],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..block_height {
            for (ci, (_, ty)) in chans.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 1000.0 + (y as f32) * 10.0 + x as f32 + 0.5;
                    match ty {
                        PixelType::Half => {
                            buf.extend_from_slice(&Half::from(v).bits().to_le_bytes());
                        }
                        PixelType::Float => {
                            buf.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                        PixelType::Uint => {
                            buf.extend_from_slice(&((v as u32).to_le_bytes()));
                        }
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// HALF + UINT channels are lossless: compress→uncompress must be identity.
    #[test]
    fn round_trip_half_uint_lossless() {
        let (w, h) = (9i32, 5i32);
        let chans = [("H", PixelType::Half), ("U", PixelType::Uint)];
        let hdr = make_header(w, h, &chans);
        // sorted order is H, U (alphabetical) — matches `chans` here.
        let line = build_line_buffer(w, h, &chans);

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 6, 16).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, line.as_slice(), "HALF/UINT must round-trip exactly");
    }

    /// Regression test (same fix + same test shape as
    /// `zip::tests::compress_uncompress_reused_across_sizes_matches_fresh_instance`):
    /// `compress`/`uncompress` now reuse ONE persistent `deflate_compressor`/
    /// `deflate_decompressor`, and `uncompress_into`'s destination buffer is
    /// grown or truncated relative to the PRIOR call instead of always being
    /// zero-filled from scratch. Pre-compresses a FULL (16-line), a PARTIAL
    /// (3-line, the last block's real shape), then a FULL block again with a
    /// SEPARATE encoder (so the loop under test calls only `uncompress()`,
    /// back to back, with no intervening `compress()` masking the buffer's
    /// real leftover size), then decodes them out of order (full, partial,
    /// full) on one persistent instance and checks each against a FRESH
    /// instance decoding the same bytes.
    #[test]
    fn compress_uncompress_reused_across_sizes_matches_fresh_instance() {
        let (w, max_y) = (50i32, 34i32); // h=35: blocks at min_y 0(16), 16(16), 32(3)
        let chans = [("H", PixelType::Half)];
        let hdr = make_header(w, max_y + 1, &chans);

        let full_a = build_line_buffer(w, 16, &chans);
        let partial = build_line_buffer(w, 3, &chans);
        let full_b = build_line_buffer(w, 16, &chans);

        let mut encoder = Pxr24Compressor::new(&hdr, w as usize * 2, 16).unwrap();
        let c_full_a = encoder.compress(&full_a, 0).unwrap().to_vec();
        let c_partial = encoder.compress(&partial, 32).unwrap().to_vec();
        let c_full_b = encoder.compress(&full_b, 16).unwrap().to_vec();
        for (c, data) in [(&c_full_a, &full_a), (&c_partial, &partial), (&c_full_b, &full_b)] {
            assert!(c.len() < data.len(), "test setup: fixture must actually compress, not store raw");
        }

        // Decode out of order (full, partial, full) on ONE instance, min_y
        // values matching each chunk's own real block height so
        // `block_height(min_y)` reports the right size for each.
        let mut u = Pxr24Compressor::new(&hdr, w as usize * 2, 16).unwrap();
        let calls = [(&c_full_a, &full_a, 0, "full_a"), (&c_partial, &partial, 32, "partial"), (&c_full_b, &full_b, 16, "full_b")];
        for (compressed, expected, min_y, label) in calls {
            let restored = u.uncompress(compressed, min_y, 0).unwrap().to_vec();
            assert_eq!(&restored, expected, "reused instance mismatch for {label}");

            let mut fresh = Pxr24Compressor::new(&hdr, w as usize * 2, 16).unwrap();
            let fresh_restored = fresh.uncompress(compressed, min_y, 0).unwrap().to_vec();
            assert_eq!(restored, fresh_restored, "reused instance diverged from a fresh one for {label}");
        }
    }

    /// FLOAT is lossy: compress→uncompress reproduces the 24-bit-quantised
    /// value (deterministically), NOT the original float.
    #[test]
    fn round_trip_float_is_24bit_quantised() {
        let (w, h) = (8i32, 3i32);
        let chans = [("Z", PixelType::Float)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 4, 16).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 0, 0).unwrap();

        // Build the expected (quantised) line: each f32 → f24 → f32 (low byte 0).
        let mut expected = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = 0.0 * 1000.0 + (y as f32) * 10.0 + x as f32 + 0.5;
                let q = f24_quantised_bits(v);
                expected.extend_from_slice(&q.to_le_bytes());
            }
        }
        assert_eq!(
            restored,
            expected.as_slice(),
            "FLOAT must decode to the deterministic 24-bit-quantised value"
        );
    }

    /// Mixed channel set (the parity image's shape): FLOAT lossy, HALF lossless.
    #[test]
    fn round_trip_mixed_channels() {
        let (w, h) = (7i32, 5i32);
        let chans = [
            ("A", PixelType::Float),
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
            ("Z", PixelType::Float),
        ];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 14, 16).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 0, 0).unwrap();

        // Expected: HALF lossless, FLOAT quantised.
        let mut expected = Vec::new();
        for y in 0..h {
            for (ci, (_, ty)) in chans.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 1000.0 + (y as f32) * 10.0 + x as f32 + 0.5;
                    match ty {
                        PixelType::Half => {
                            expected.extend_from_slice(&Half::from(v).bits().to_le_bytes());
                        }
                        PixelType::Float => {
                            expected.extend_from_slice(&f24_quantised_bits(v).to_le_bytes());
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        assert_eq!(restored, expected.as_slice(), "mixed channels mismatch");
    }

    /// The "no-shrink → store raw" path: incompressible data is returned
    /// unchanged (== the original line buffer), so `dataSize == uncompressedSize`.
    #[test]
    fn no_shrink_stores_raw() {
        let (w, h) = (4i32, 1i32);
        let chans = [("U", PixelType::Uint)];
        let hdr = make_header(w, h, &chans);

        // High-entropy UINT data: after byte-split + diff the deltas are still
        // ~random, so zlib cannot shrink it below the raw size.
        let mut line = Vec::new();
        let mut state: u32 = 0x9E37_79B9;
        for _ in 0..(w as usize) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            line.extend_from_slice(&state.to_le_bytes());
        }

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 4, 16).unwrap();
        let compressed = c.compress(&line, 0).unwrap();
        assert_eq!(
            compressed,
            line.as_slice(),
            "incompressible data must be stored raw (== original bytes)"
        );
    }

    /// Multi-block: a partial last block (height < 16) must compress/decompress
    /// using only the lines it actually spans (driven by `min_y` + data window).
    #[test]
    fn partial_last_block() {
        // 20 lines total, blocks of 16 → block at min_y=16 has height 4.
        let (w, h) = (5i32, 20i32);
        let chans = [("H", PixelType::Half)];
        let hdr = make_header(w, h, &chans);

        // Build only the 4 lines of the second block (y = 16..=19).
        let mut line = Vec::new();
        for y in 16..20 {
            for x in 0..w {
                let v = (y as f32) * 10.0 + x as f32 + 0.5;
                line.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
        }

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 2, 16).unwrap();
        let compressed = c.compress(&line, 16).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 16, 0).unwrap();
        assert_eq!(restored, line.as_slice(), "partial block must round-trip");
    }

    #[test]
    fn num_scan_lines_is_16_and_format_xdr() {
        let hdr = make_header(4, 4, &[("H", PixelType::Half)]);
        let c = Pxr24Compressor::new(&hdr, 8, 16).unwrap();
        assert_eq!(c.num_scan_lines(), 16);
        assert_eq!(c.format(), Format::Xdr);
    }

    /// `float_to_float24` matches known reference values from the C++ algorithm.
    #[test]
    fn float24_known_values() {
        // 0.0 → 0; reverse is 0.
        assert_eq!(float_to_float24(0.0), 0);
        // 1.0 = 0x3F800000 → (s>>8)=0, ((e|m)+(m&0x80))>>8 = 0x3F8000.
        assert_eq!(float_to_float24(1.0), 0x3F8000);
        // -1.0 = 0xBF800000 → sign bit preserved: (0x80000000>>8)|0x3F8000.
        assert_eq!(float_to_float24(-1.0), 0x80_0000 | 0x3F8000);
        // +inf = 0x7F800000 → e>>8 = 0x7F8000.
        assert_eq!(float_to_float24(f32::INFINITY), 0x7F8000);
    }

    /// A zlib stream followed by trailing garbage must be rejected as a
    /// corrupt chunk, matching `exr_uncompress_buffer`
    /// (`OpenEXRCore/compression.c:180-197`), which `undo_pxr24_impl` calls
    /// through. See the identical test in `compressor::zip::tests`.
    #[test]
    fn pxr24_rejects_trailing_bytes_after_zlib_stream() {
        let (w, h) = (9i32, 5i32);
        let chans = [("H", PixelType::Half), ("U", PixelType::Uint)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Pxr24Compressor::new(&hdr, w as usize * 6, 16).unwrap();
        let mut compressed = c.compress(&line, 0).unwrap().to_vec();
        compressed.push(0x00); // trailing byte after the valid zlib stream
        let mut c2 = Pxr24Compressor::new(&hdr, w as usize * 6, 16).unwrap();
        assert!(
            c2.uncompress(&compressed, 0, 0).is_err(),
            "trailing bytes after the zlib stream must be CORRUPT_CHUNK"
        );
    }

    // The two process-global-default zip-level tests
    // (`pxr24_ignores_per_header_zip_level_override`,
    // `pxr24_follows_process_global_default_zip_level`) moved to
    // `tests/global_compression_defaults.rs` -- a separate process from this
    // `--lib` unit-test binary, so mutating the process-global default there
    // can no longer race the many OTHER lib tests (this module's own DWA/ZIP
    // parity-shaped tests included) that read the effective compression
    // level without taking any lock. See that file's module doc for the
    // observed flake this caused and why a lock over the mutators alone
    // could not fix it.
}
