//! Port of OpenEXR `ImfPizCompressor.cpp` / `.h` and the byte algorithm in
//! `OpenEXRCore/internal_piz.c` — the **PIZ** scanline codec (Christian Rouet's
//! wavelet + Huffman lossless compressor).
//!
//! # What this codec does (1:1 with C++)
//!
//! PIZ is lossless for every pixel type.  One chunk is compressed by:
//!
//! 1. **Re-gather to per-channel regions.** The block line buffer this codec
//!    receives is laid out line-by-line, channel-by-channel (exactly how
//!    [`crate::scanline::OutputFile`] gathers it).  PIZ first re-gathers it into
//!    a single `u16` array where each channel's samples are *contiguous* across
//!    all lines of the chunk — the layout `internal_piz.c` builds in
//!    `scratch_buffer_1` (`for y { for c { copy channel line } }`).  Every disk
//!    value is little-endian (Xdr), so each on-disk `u16` is read with
//!    [`u16::from_le_bytes`]; 32-bit channels contribute two interleaved `u16`
//!    samples per pixel (`wcount == bytes_per_element / 2`).
//! 2. **Bitmap of used values → forward LUT (range compression).**
//!    `bitmap_from_data` records which of the 65536 possible `u16` values
//!    occur; `forward_lut_from_bitmap` maps them onto a dense `[0, maxValue]`
//!    range.  `apply_lut` rewrites every sample through the LUT — this shrinks
//!    the value range so the wavelet's 14-bit fast path is used more often.
//! 3. **2-D Haar wavelet** ([`wavelet::encode`]) per channel region, per 16-bit
//!    sample lane (`wcount` lanes for 32-bit channels).
//! 4. **Huffman entropy coding** ([`huffman::compress_with_limit`]) of the whole
//!    range-compressed, wavelet-transformed `u16` array.
//!
//! The on-disk chunk is:
//!
//! ```text
//!   minNonZero  u16 LE          // first non-zero bitmap byte index
//!   maxNonZero  u16 LE          // last  non-zero bitmap byte index
//!   bitmap[minNonZero ..= maxNonZero]   (only when minNonZero <= maxNonZero)
//!   hufLength   u32 LE          // length of the Huffman stream that follows
//!   huffman stream
//! ```
//!
//! ## The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `internal_exr_apply_piz`: after Huffman coding, if the total output is **not
//! smaller** than the original packed bytes (`nOut >= packedbytes`), the codec
//! stores the original `packed_buffer` verbatim and sets the output size to
//! `packedbytes`.  On disk this means `dataSize == uncompressedSize`, which is
//! how the reader detects the raw passthrough (same convention as ZIP/RLE).
//! We replicate this: when the assembled PIZ output is `>=` the input length we
//! return a copy of the original `data` unchanged.
//!
//! # Format
//!
//! Like the other ported scanline codecs, PIZ here consumes and produces the
//! machine-independent **Xdr** (little-endian) line buffer.  The C core works
//! internally in host-native `u16` (via `priv_to_native16` / `priv_from_native16`)
//! purely as an optimisation; on a little-endian host that conversion is a
//! no-op, and reading each on-disk `u16` with [`u16::from_le_bytes`] (and
//! writing it back with [`u16::to_le_bytes`]) is byte-identical on every host.
//! Reporting [`Format::Xdr`] keeps the existing `OutputFile`/`InputFile` gather
//! and scatter (which assume Xdr) correct without any special-casing.
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPizCompressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPizCompressor.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_piz.c>

pub(crate) mod fast_huf;
pub mod huffman;
pub mod wavelet;

use crate::attr::PixelType;
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};
use imath_rs::Box2i;

use imath_rs::{divp, modp};

/// The full 16-bit value range (`USHORT_RANGE` in C).
const USHORT_RANGE: usize = 1 << 16;
/// Bitmap size in bytes (`BITMAP_SIZE = USHORT_RANGE >> 3`).
const BITMAP_SIZE: usize = USHORT_RANGE >> 3;

// ---------------------------------------------------------------------------
// Bitmap + LUT — port of `bitmapFromData` / `forwardLutFromBitmap` /
// `reverseLutFromBitmap` / `applyLut`
// ---------------------------------------------------------------------------

/// Build the "used value" bitmap and its non-zero byte span into a
/// caller-owned, reusable `bitmap` buffer — port of `bitmapFromData`.
/// Returns `(min_non_zero, max_non_zero)`.
///
/// `bitmap` is fully re-zeroed every call (`internal_piz.c:32-33`'s own
/// `for (i) bitmap[i] = 0` before setting bits) — required now that
/// `PizCompressor` reuses ONE `bitmap` field for both directions, matching
/// C's single `scratch_buffer_2`; a stale bit surviving from a PRIOR call
/// (e.g. compress on a big chunk, decode/compress on a smaller one after)
/// would corrupt this call's own bitmap.
///
/// Value `0` is never stored in the bitmap (`bitmap[0] &= ~1`) because the data
/// are assumed always to contain zeroes; the LUTs special-case index 0.
fn bitmap_from_data(data: &[u16], bitmap: &mut Vec<u8>) -> (u16, u16) {
    bitmap.clear();
    bitmap.resize(BITMAP_SIZE, 0);

    for &v in data.iter() {
        bitmap[(v >> 3) as usize] |= 1 << (v & 7);
    }

    bitmap[0] &= !1u8; // zero is implicit

    let mut min_nz: u16 = (BITMAP_SIZE - 1) as u16;
    let mut max_nz: u16 = 0;
    for (i, &byte) in bitmap.iter().enumerate() {
        if byte != 0 {
            let i = i as u16;
            if min_nz > i {
                min_nz = i;
            }
            if max_nz < i {
                max_nz = i;
            }
        }
    }

    (min_nz, max_nz)
}

/// Build the forward range-compression LUT into a caller-owned, reusable
/// `lut` buffer — port of `forwardLutFromBitmap`. Returns `max_value`, where
/// `lut[v]` is the compressed value for `v`.
///
/// Every one of `lut`'s `USHORT_RANGE` entries is written unconditionally by
/// this loop (`lut[i] = k` or `lut[i] = 0`, matching C's own
/// `lut[i] = k++`/`lut[i] = 0` at every `i`) — no separate re-zero needed
/// before or after; `resize` only handles first-call sizing.
fn forward_lut_from_bitmap(bitmap: &[u8], lut: &mut Vec<u16>) -> u16 {
    lut.resize(USHORT_RANGE, 0);
    let mut k: u16 = 0;

    for (i, slot) in lut.iter_mut().enumerate() {
        if i == 0 || (bitmap[i >> 3] & (1 << (i & 7))) != 0 {
            *slot = k;
            k = k.wrapping_add(1);
        } else {
            *slot = 0;
        }
    }

    k.wrapping_sub(1)
}

/// Build the inverse LUT into a caller-owned, reusable `lut` buffer — port of
/// `reverseLutFromBitmap`. Returns `max_value`, where `lut[compressed] ==
/// original_value`.
///
/// Unlike `forward_lut_from_bitmap`, this loop only writes the first `k`
/// entries (one per real value found) — C explicitly zeros the REST
/// (`internal_piz.c`'s own trailing `while (k < USHORT_RANGE) lut[k++] = 0;`)
/// rather than relying on the caller's buffer already being zero. Now that
/// `lut` is a reused, persistent field, that trailing zero-fill is
/// load-bearing: without it, entries past `k` would keep whatever a PRIOR,
/// larger chunk's decode left there.
fn reverse_lut_from_bitmap(bitmap: &[u8], lut: &mut Vec<u16>) -> u16 {
    lut.resize(USHORT_RANGE, 0);
    let mut k: usize = 0;

    for i in 0..USHORT_RANGE {
        if i == 0 || (bitmap[i >> 3] & (1 << (i & 7))) != 0 {
            lut[k] = i as u16;
            k += 1;
        }
    }

    let n = (k - 1) as u16;
    for slot in lut[k..].iter_mut() {
        *slot = 0;
    }
    n
}

/// Rewrite every value through `lut` in place — port of `applyLut`
/// (`internal_piz.c:99-116`), including its 8-wide manual unroll ("partially
/// unrolling is noticeably faster" per C's own comment). `as_chunks_mut::<8>()`
/// processing every complete group of 8 (C's condition is `nData > 8`, i.e. a
/// buffer of EXACTLY 8 remaining falls through to the scalar tail there) is
/// behaviourally identical either way — both branches compute the same
/// `lut[data[i]]` per element, so this is purely which code path computes a
/// given element, never a different byte out.
fn apply_lut(lut: &[u16], data: &mut [u16]) {
    let (chunks, remainder) = data.as_chunks_mut::<8>();
    for chunk in chunks {
        let mut dvals = [0u16; 8];
        dvals.copy_from_slice(chunk);
        for v in &mut dvals {
            *v = lut[*v as usize];
        }
        chunk.copy_from_slice(&dvals);
    }
    for v in remainder {
        *v = lut[*v as usize];
    }
}

// ---------------------------------------------------------------------------
// ChannelInfo — per-channel layout captured from the header
// ---------------------------------------------------------------------------

/// One channel's region within the per-channel-contiguous `tmp` buffer:
/// `(start, nx, ny, wcount)` where `start` is the region's first `u16` index,
/// `nx`/`ny` are the channel's sampled pixel extent within the chunk, and
/// `wcount` is the 16-bit lanes per pixel.
type ChannelRegion = (usize, usize, usize, usize);

/// Per-channel layout snapshot — the fields `internal_piz.c` reads from
/// `exr_coding_channel_info_t` (`bytes_per_element`, `x_samples`, `y_samples`).
#[derive(Clone, Copy, Debug)]
struct ChannelInfo {
    /// Sample type on disk (drives `wcount = bytes_per_element / 2`).
    type_: PixelType,
    /// Horizontal sub-sampling factor.
    x_sampling: i32,
    /// Vertical sub-sampling factor.
    y_sampling: i32,
}

impl ChannelInfo {
    /// 16-bit lanes per pixel — `wcount = bytes_per_element / 2`.
    /// HALF/UINT/FLOAT are 2/4/4 bytes → 1/2/2 lanes.
    fn wcount(&self) -> usize {
        match self.type_ {
            PixelType::Half => 1,
            PixelType::Uint | PixelType::Float => 2,
            PixelType::NumPixelTypes => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PizCompressor
// ---------------------------------------------------------------------------

/// PIZ scanline codec — port of `Imf::PizCompressor`.
///
/// `num_scan_lines()` is **32** (`getCompressionNumScanlines(PIZ) == 32`).
/// Owns scratch buffers whose contents are returned as `&[u8]`; the slice is
/// valid until the next call (mirroring the C++ member output buffer).
///
/// The channel list and data-window bounds are captured from the header at
/// construction; `compress`/`uncompress` drive the per-line, per-channel
/// re-gather/scatter from the chunk's `min_y` and that layout.
pub struct PizCompressor {
    /// Scan lines per chunk — `numScanLines()` (32 for PIZ).
    num_scan_lines: i32,
    /// Per-channel layout in file (channel-list) order.
    channels: Vec<ChannelInfo>,
    /// Data-window minimum x.
    min_x: i32,
    /// Data-window maximum x.
    max_x: i32,
    /// Data-window maximum y — clamps the final (possibly partial) block.
    max_y: i32,
    /// Scratch buffer holding the result of the last call (the C++ output buffer).
    out: Vec<u8>,
    /// Test seam: the entropy-coded bit-count ceiling passed to
    /// `huffman::compress_with_limit`. Always `u32::MAX as u64` outside tests
    /// (matching C's real `hufEncode` `ARGUMENT_OUT_OF_RANGE` threshold,
    /// `internal_huf.c:846-848` — never observable at any size a real test
    /// fixture could reach: `> u32::MAX` bits needs a ~512 MiB+ input).
    /// [`Self::set_huf_bit_limit_for_test`] lowers it so a test can drive the
    /// SAME raw-storage fallback branch `compress()` takes on a real
    /// oversized image, through the real `Compressor::compress` seam, without
    /// needing that multi-hundred-MB fixture.
    huf_bit_limit: u64,
    /// Range-compression bitmap scratch, shared by `compress()` AND
    /// `uncompress()` — mirrors C's single `scratch_buffer_2`
    /// (`internal_piz.c:491-501` on encode, `:656-667` on decode: same
    /// region, reused across the whole file's compression/decompression,
    /// never a fresh allocation per chunk). Compress and decode never run
    /// concurrently on one instance, so one field serves both directions,
    /// exactly like C's one buffer.
    bitmap: Vec<u8>,
    /// Range-compression LUT scratch, shared by `compress()` (the forward
    /// LUT, `forward_lut_from_bitmap`) and `uncompress()` (the reverse LUT,
    /// `reverse_lut_from_bitmap`) — same `scratch_buffer_2` region as
    /// `bitmap` in C (`internal_piz.c:502`/`:666`), reused the same way.
    lut: Vec<u16>,
    /// Per-channel-contiguous gather/wavelet/huffman scratch, shared by
    /// `compress()` and `uncompress()` — mirrors C's single
    /// `scratch_buffer_1` (`internal_piz.c:483-489` on encode, where it's
    /// the gather target; `:648-654` on decode, where it's the huffman
    /// output / `wavbuf`).
    tmp: Vec<u16>,
    /// Per-channel current line counter scratch for the gather (compress)
    /// and scatter (uncompress) loops, shared the same way (small, but
    /// still a per-chunk allocation before this field existed).
    chan_line: Vec<usize>,
    /// [`Self::channel_regions`]'s output buffer, reused across calls
    /// (instead of a `Vec::with_capacity` per call in the old
    /// `channel_regions`) — cleared and re-pushed every call, which is cheap
    /// (no zero-fill) and reuses the allocation once warmed up.
    regions: Vec<ChannelRegion>,
    /// Persistent Huffman ENCODE-side scratch — see
    /// [`huffman::HufEncScratch`]'s doc. Also holds the assembled huffman
    /// stream bytes (`compress`'s `out` parameter), a separate buffer from
    /// `self.out` since the chunk header/bitmap are assembled into `self.out`
    /// only AFTER the huffman stream is known (its own length is part of
    /// that header).
    huf_scratch: huffman::HufEncScratch,
    huf_out: Vec<u8>,
    /// Persistent Huffman DECODE-side accelerated-decoder scratch
    /// — see [`fast_huf::FastHufDecoder`]'s doc.
    fast_huf: fast_huf::FastHufDecoder,
}

impl PizCompressor {
    /// Create a new PIZ compressor — port of `PizCompressor::PizCompressor`.
    ///
    /// `num_scan_lines` is 32.  The channel layout and data-window bounds are
    /// captured from `header`.  `_max_scan_line_size` matches the C++ ctor
    /// signature (the C core sizes its scratch from the chunk's unpacked size,
    /// which this port derives per call from the channel layout instead).
    pub fn new(
        header: &crate::header::Header,
        _max_scan_line_size: usize,
        num_scan_lines: i32,
    ) -> Result<Self> {
        let dw = header.data_window()?;
        let channel_list = header.channels()?;

        let channels: Vec<ChannelInfo> = channel_list
            .iter()
            .map(|(_, ch)| ChannelInfo {
                type_: ch.type_,
                x_sampling: ch.x_sampling,
                y_sampling: ch.y_sampling,
            })
            .collect();

        Ok(Self {
            num_scan_lines,
            channels,
            min_x: dw.min.x,
            max_x: dw.max.x,
            max_y: dw.max.y,
            out: Vec::new(),
            huf_bit_limit: u32::MAX as u64,
            bitmap: Vec::new(),
            lut: Vec::new(),
            tmp: Vec::new(),
            chan_line: Vec::new(),
            regions: Vec::new(),
            huf_scratch: huffman::HufEncScratch::default(),
            huf_out: Vec::new(),
            fast_huf: fast_huf::FastHufDecoder::default(),
        })
    }

    /// Test seam (see the field doc): override the entropy-coded
    /// bit-count ceiling so a test can drive `compress()`'s raw-storage
    /// fallback for an oversized Huffman table through the REAL
    /// `Compressor::compress` method, instead of calling
    /// `huffman::compress_with_limit` directly (which proves the primitive
    /// works but never proves `PizCompressor` actually wires it up and takes
    /// the fallback branch on a real chunk).
    #[cfg(test)]
    pub(crate) fn set_huf_bit_limit_for_test(&mut self, limit: u64) {
        self.huf_bit_limit = limit;
    }

    /// Sampled scanline count of the block starting at `block_min_y` —
    /// `min(block_min_y + numScanLines - 1, max_y) - block_min_y + 1`.
    fn block_height(&self, block_min_y: i32) -> i32 {
        let block_max_y = (block_min_y + self.num_scan_lines - 1).min(self.max_y);
        block_max_y - block_min_y + 1
    }

    /// Per-channel horizontal sample count over the data window's x range —
    /// `divp(max_x, xs) - divp(min_x, xs) + 1` (matches the writer's per-line
    /// pixel count).
    fn channel_width(&self, x_sampling: i32) -> usize {
        let d_min = divp(self.min_x, x_sampling);
        let d_max = divp(self.max_x, x_sampling);
        (d_max - d_min + 1) as usize
    }

    /// Layout of each channel's region inside the per-channel-contiguous `tmp`
    /// buffer, plus the total `u16` count.  `regions[c] = (start, nx, ny, wcount)`.
    ///
    /// `nx`/`ny` are the channel's sampled pixel extent within the chunk; the
    /// region holds `nx * ny * wcount` interleaved `u16` samples.
    fn channel_regions(&mut self, block_min_y: i32, block_height: i32) -> Result<usize> {
        self.regions.clear();
        let mut total = 0usize;

        for ci in &self.channels {
            let nx = self.channel_width(ci.x_sampling);
            // Sampled line count of this channel within the block.
            let mut ny = 0usize;
            for y in 0..block_height {
                let cury = block_min_y + y;
                if modp(cury, ci.y_sampling) == 0 {
                    ny += 1;
                }
            }
            let wcount = ci.wcount();
            if wcount == 0 {
                return Err(ExrError::Arg("PIZ: unknown pixel type".into()));
            }
            self.regions.push((total, nx, ny, wcount));
            total += nx * ny * wcount;
        }

        Ok(total)
    }
}

impl Compressor for PizCompressor {
    /// 32 scan lines per chunk for PIZ.
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// PIZ consumes/produces the machine-independent Xdr (little-endian) format
    /// (see the module docs for why this is byte-identical to the C core's
    /// internal-native processing).
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one chunk — port of `internal_exr_apply_piz`.
    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]> {
        let srcb = data.len();
        if srcb == 0 {
            self.out.clear();
            return Ok(&self.out);
        }
        let block_height = self.block_height(min_y);
        let total_u16 = self.channel_regions(min_y, block_height)?;

        // Re-gather the line-interleaved Xdr line buffer into the per-channel-
        // contiguous `self.tmp` u16 array (C's single `scratch_buffer_1`,
        // shared with `uncompress()` — see the field doc). `n_in` walks the
        // input; each channel line lands in its region at the correct line
        // offset. Every element gets written by the gather loop below (one
        // full pass over `total_u16` slots) -- `ensure_len`
        // (grow/truncate, no unconditional zero-fill) instead of
        // `clear()` + `resize(_, 0)`, which used to re-zero-fill on every
        // single call regardless of what the PREVIOUS call already wrote.
        super::ensure_len(&mut self.tmp, total_u16);
        let mut n_in = 0usize;

        // Per-channel running line index within its region.
        self.chan_line.clear();
        self.chan_line.resize(self.channels.len(), 0);

        for y in 0..block_height {
            let cury = min_y + y;
            for (c, ci) in self.channels.iter().enumerate() {
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let (start, nx, _ny, wcount) = self.regions[c];
                let samples = nx * wcount; // u16s on this channel line
                let in_bytes = samples * 2;
                if n_in + in_bytes > srcb {
                    return Err(ExrError::Logic(
                        "PIZ compress: channel line overruns the input buffer".into(),
                    ));
                }
                // Bulk gather (mirrors C's one `memcpy` + one byte-swap pass
                // per line/channel, `internal_piz.c:530-531`): the shared
                // `le_bytes_to_u16s` instead of one
                // bounds-checked `from_le_bytes` per sample.
                let dst = start + self.chan_line[c] * samples;
                let src_line = &data[n_in..n_in + in_bytes];
                crate::misc::le_bytes_to_u16s(src_line, &mut self.tmp[dst..dst + samples])?;
                n_in += in_bytes;
                self.chan_line[c] += 1;
            }
        }

        if n_in != srcb {
            return Err(ExrError::Logic(format!(
                "PIZ compress: consumed {n_in} of {srcb} line-buffer bytes (layout mismatch)"
            )));
        }

        // Bitmap + forward LUT, then range-compress every sample.
        let (min_nz, max_nz) = bitmap_from_data(&self.tmp, &mut self.bitmap);
        let max_value = forward_lut_from_bitmap(&self.bitmap, &mut self.lut);
        apply_lut(&self.lut, &mut self.tmp);

        // 2-D wavelet per channel region, per 16-bit lane.
        for &(start, nx, ny, wcount) in &self.regions {
            if nx == 0 || ny == 0 {
                continue;
            }
            for j in 0..wcount {
                wavelet::encode(
                    &mut self.tmp[start + j..start + nx * ny * wcount],
                    (nx, ny),
                    (wcount, wcount * nx),
                    max_value,
                )?;
            }
        }

        // Huffman-encode the whole transformed array.
        //
        // `huffman::compress_with_limit` returns `Ok(false)` when the
        // entropy-coded bit count does not fit a `u32` (`hufEncode`'s
        // `ARGUMENT_OUT_OF_RANGE`, `internal_huf.c:846-848`). C's
        // `internal_exr_apply_piz` (`internal_piz.c:596-603`) catches that by
        // storing the chunk raw — the same fallback as the "no-shrink" path
        // below, so we just take it early. `huf_scratch`/`huf_out` are
        // persistent fields instead of the old
        // fresh-tables-and-returned-Vec signature.
        if !huffman::compress_with_limit(
            &self.tmp,
            &mut self.huf_scratch,
            self.huf_bit_limit,
            &mut self.huf_out,
        )? {
            self.out.clear();
            self.out.extend_from_slice(data);
            return Ok(&self.out);
        }
        let huff = &self.huf_out;

        // Assemble the PIZ chunk directly into self.out (the
        // same out/scratch policy every codec's compress AND uncompress path
        // follows) instead of a throwaway local Vec: minNonZero, maxNonZero,
        // bitmap span, len, huff.
        self.out.clear();
        self.out.extend_from_slice(&min_nz.to_le_bytes());
        self.out.extend_from_slice(&max_nz.to_le_bytes());
        if min_nz <= max_nz {
            self.out.extend_from_slice(&self.bitmap[min_nz as usize..=max_nz as usize]);
        }
        self.out.extend_from_slice(&(huff.len() as u32).to_le_bytes());
        self.out.extend_from_slice(huff);

        // "No-shrink → store raw": C sets nOut = packedbytes when nOut >= it.
        if self.out.len() >= srcb {
            self.out.clear();
            self.out.extend_from_slice(data);
        }

        Ok(&self.out)
    }

    /// Uncompress one PIZ chunk — port of `internal_exr_undo_piz`.
    ///
    /// `expected_len` is unused here: PIZ's decode capacity
    /// is already driven entirely by `channel_regions(min_y, block_height)`
    /// -- the channel layout plus the CALLER-SUPPLIED `min_y`/block height,
    /// never a construction-time upper bound -- so unlike ZIP/PXR24 it never
    /// needed an explicit target size to begin with.
    fn uncompress(&mut self, data: &[u8], min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }

        let block_height = self.block_height(min_y);
        let total_u16 = self.channel_regions(min_y, block_height)?;

        // --- Read the range-compression header (minNonZero, maxNonZero, bitmap).
        // Reused, grow-only scratch (`internal_piz.c:648-667`): the buffer
        // itself is kept across calls, but every byte is still zeroed each
        // call exactly like C's own `memset` (:673) — only the allocation is
        // amortised, not the zero-fill.
        self.bitmap.clear();
        self.bitmap.resize(BITMAP_SIZE, 0);
        let mut n = 0usize;
        if data.len() < 4 {
            return Err(ExrError::Input("PIZ uncompress: chunk too short for header".into()));
        }
        let min_nz = u16::from_le_bytes([data[0], data[1]]);
        let max_nz = u16::from_le_bytes([data[2], data[3]]);
        n += 4;

        if max_nz as usize >= BITMAP_SIZE {
            return Err(ExrError::Input("PIZ uncompress: maxNonZero out of range".into()));
        }
        if min_nz <= max_nz {
            let span = (max_nz - min_nz + 1) as usize;
            if n + span > data.len() {
                return Err(ExrError::Input("PIZ uncompress: bitmap span truncated".into()));
            }
            self.bitmap[min_nz as usize..=max_nz as usize].copy_from_slice(&data[n..n + span]);
            n += span;
        }

        let max_value = reverse_lut_from_bitmap(&self.bitmap, &mut self.lut);

        // --- Read the Huffman length and decode.
        if n + 4 > data.len() {
            return Err(ExrError::Input("PIZ uncompress: missing Huffman length".into()));
        }
        let huf_len = u32::from_le_bytes([data[n], data[n + 1], data[n + 2], data[n + 3]]) as usize;
        n += 4;
        if n + huf_len > data.len() {
            return Err(ExrError::Input("PIZ uncompress: Huffman stream truncated".into()));
        }
        let huf = &data[n..n + huf_len];

        huffman::decompress(huf, total_u16, &mut self.tmp, &mut self.fast_huf)?;

        // --- Inverse wavelet per channel region, per lane.
        for &(start, nx, ny, wcount) in &self.regions {
            if nx == 0 || ny == 0 {
                continue;
            }
            for j in 0..wcount {
                wavelet::decode(
                    &mut self.tmp[start + j..start + nx * ny * wcount],
                    (nx, ny),
                    (wcount, wcount * nx),
                    max_value,
                )?;
            }
        }

        // --- Inverse LUT (expand to the original value range).
        apply_lut(&self.lut, &mut self.tmp);

        // --- Scatter back to the line-interleaved Xdr output buffer.
        //
        // `out` is sized ONCE to the exact decoded size and every byte is then
        // written by the scatter below (the sum over rows and sampled channels
        // of `nx * wcount * 2` is `total_u16 * 2` by construction; the final
        // length check turns a layout bug into an error instead of stale
        // bytes). A warmed buffer keeps its contents (`ensure_len`'s
        // grow/truncate, inlined so the probe sits on the keep branch) and
        // skips the per-row zero-fill the old `resize(.., 0)` + overwrite paid
        // for.
        let out_len = total_u16 * 2;
        if self.out.len() >= out_len {
            self.out.truncate(out_len);
            crate::probe::hit(crate::probe::Probe::PizOutKept);
        } else {
            self.out.resize(out_len, 0);
        }
        let mut out_pos = 0usize;
        self.chan_line.clear();
        self.chan_line.resize(self.channels.len(), 0);

        for y in 0..block_height {
            let cury = min_y + y;
            for (c, ci) in self.channels.iter().enumerate() {
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let (start, nx, _ny, wcount) = self.regions[c];
                let samples = nx * wcount;
                let src = start + self.chan_line[c] * samples;
                // Bulk write (mirrors C's one `memcpy` per row/channel,
                // `internal_piz.c:777`) via the shared `u16s_to_le_bytes`
                // instead of one bounds-checked store per
                // 2-byte sample.
                let dst = self.out.get_mut(out_pos..out_pos + samples * 2).ok_or_else(|| {
                    ExrError::Logic("PIZ uncompress: scatter overruns the output buffer".into())
                })?;
                crate::misc::u16s_to_le_bytes(&self.tmp[src..src + samples], dst)?;
                out_pos += samples * 2;
                self.chan_line[c] += 1;
            }
        }
        if out_pos != out_len {
            return Err(ExrError::Logic(format!(
                "PIZ uncompress: scattered {out_pos} of {out_len} bytes (layout mismatch)"
            )));
        }

        Ok(&self.out)
    }

    /// Compress one tile, encoded against the tile's own pixel rectangle.
    ///
    /// PIZ derives its chunk layout from `min_x`/`max_x` (per-channel width) and
    /// `max_y` (final-block clamp).  For a tile these must be the tile's X/Y
    /// extent, not the data window's — mirroring C++ `compressTile`, which feeds
    /// the tile's `Box2i range` into the encode step.  We set the rectangle
    /// fields from `range`, then run the identical compress path at
    /// `range.min.y`.  The compressor is dedicated to tiles in the tiled writer,
    /// so each call overwrites the fields fresh (no restore needed).
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
    use imath_rs::{Half, V2f, V2i};

    fn make_header(w: i32, h: i32, chans: &[(&str, PixelType)]) -> Header {
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Piz,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for (name, ty) in chans {
            cl.insert(*name, Channel::new(*ty, 1, 1, false)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr
    }

    /// Build a line-interleaved Xdr block buffer: for each line, for each
    /// channel (in `chans` order), `w` samples of deterministic values.
    fn build_line_buffer(w: i32, block_height: i32, chans: &[(&str, PixelType)]) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..block_height {
            for (ci, (_, ty)) in chans.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 1000.0 + (y as f32) * 10.0 + x as f32 + 0.5;
                    match ty {
                        PixelType::Half => buf.extend_from_slice(&Half::from(v).bits().to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&((v as u32).to_le_bytes())),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// Drive the codec exactly as the scanline reader does: compress, then only
    /// call `uncompress` when the codec genuinely shrank the data.  When the
    /// "no-shrink → store raw" path fired (`compressed.len() == line.len()`) the
    /// reader uses the raw bytes directly, so the round-trip target is `line`
    /// itself.  Either way the recovered bytes must equal the input.
    fn round_trip(w: i32, h: i32, chans: &[(&str, PixelType)]) {
        let hdr = make_header(w, h, chans);
        // sorted channel order must match `chans`; tests pass already-sorted lists.
        let line = build_line_buffer(w, h, chans);
        let mut c = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        let restored: Vec<u8> = if compressed.len() == line.len() {
            // Raw passthrough (reader detects this by size and bypasses the codec).
            compressed
        } else {
            c.uncompress(&compressed, 0, 0).unwrap().to_vec()
        };
        assert_eq!(restored, line.as_slice(), "PIZ round-trip mismatch {w}x{h} {chans:?}");
    }

    #[test]
    fn round_trip_half_only() {
        round_trip(9, 5, &[("R", PixelType::Half)]);
    }

    #[test]
    fn round_trip_multi_half() {
        round_trip(13, 7, &[("B", PixelType::Half), ("G", PixelType::Half), ("R", PixelType::Half)]);
    }

    /// Value-range-controlled line buffer for the scratch-buffer-reuse test
    /// below: every sample is `base + i * 0.01` for a small, distinct `base`
    /// per call, so two calls built with well-separated `base`s produce
    /// PIZ range-compression bitmaps ([`reverse_lut_from_bitmap`]'s
    /// `min_nz`/`max_nz` span) that do NOT overlap — needed so a stale
    /// `self.bitmap` byte from a PRIOR call actually lands in a bit position
    /// the current call never overwrites (see the test's own doc comment for
    /// why a naive value choice does not reliably exercise this).
    fn build_line_buffer_at(w: i32, h: i32, base: f32) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut i = 0.0f32;
        for _y in 0..h {
            for _x in 0..w {
                buf.extend_from_slice(&Half::from(base + i * 0.01).bits().to_le_bytes());
                i += 1.0;
            }
        }
        buf
    }

    /// Scratch-buffer reuse: `bitmap`/`tmp`/`chan_line`
    /// are persistent fields, filled in place on every `uncompress()` call
    /// instead of freshly allocated — this test exists specifically to catch
    /// stale data leaking from a PRIOR, LARGER chunk into a SMALLER one that
    /// reuses the same buffers on the same compressor instance.
    ///
    /// The two chunks use DELIBERATELY SEPARATED value ranges (`base`s
    /// `0.0002` and `50.0`): PIZ's range-compression bitmap only overwrites
    /// the byte span covering the CURRENT chunk's own `[min_nz, max_nz]`
    /// (`internal_piz.c:686-693`); the rest of a persistent `bitmap` buffer
    /// is assumed zero on the wire. A first pass with values near 0 sets
    /// LOW bitmap bit positions, then a second, smaller/higher-valued chunk
    /// only clears/rewrites ITS OWN higher bit-position span — if the
    /// implementation forgets to re-zero the WHOLE bitmap first, the low
    /// positions from the first chunk remain set and corrupt
    /// `reverse_lut_from_bitmap`'s scan (every stale nonzero bit before a
    /// real one shifts that real entry's `lut[]` index). An earlier version
    /// of this test used two same-magnitude ranges and did NOT catch the
    /// mutation below, because Half preserves ordering for positive floats:
    /// if the stale range sits ENTIRELY ABOVE the second chunk's own real
    /// values, the shift lands in unused `lut[]` tail entries and never
    /// surfaces — this is why the value ranges here are chosen deliberately
    /// (small chunk's values are ALL numerically larger, hence bit-pattern
    /// larger, than the big chunk's, so the big chunk's stale low bits sit
    /// BEFORE the small chunk's own real entries and DO shift them).
    ///
    /// Two consecutive `uncompress_tile` calls on ONE `PizCompressor`: a big
    /// tile first (near-zero values, populates the persistent scratch
    /// buffers), then a strictly SMALLER tile second (higher values). The
    /// small tile's restored bytes are checked byte-for-byte against an
    /// INDEPENDENT reference: a FRESH `PizCompressor` decoding the identical
    /// small chunk with no prior call — which cannot possibly see stale
    /// scratch data, so any divergence between the two proves contamination
    /// from the big call.
    ///
    /// Mutation check performed while writing this test: temporarily
    /// guarding `self.bitmap.resize(BITMAP_SIZE, 0)` with
    /// `if self.bitmap.is_empty()` (i.e. only zero-filling on the FIRST
    /// call, leaving the big call's bitmap bits behind for the second) made
    /// this test fail with a real byte mismatch (confirmed below before
    /// reverting), whereas the naive same-range version of this test did
    /// NOT fail under the identical mutation — recorded here so the same
    /// mistake (a scratch-buffer test that can't actually see stale data)
    /// is not repeated.
    #[test]
    fn uncompress_scratch_buffers_do_not_leak_between_a_big_then_small_chunk() {
        let chans: &[(&str, PixelType)] = &[("R", PixelType::Half)];
        let hdr = make_header(128, 80, chans);

        // PIZ's num_scan_lines is 32 -- a single compress/uncompress call
        // only ever covers up to 32 rows (`block_height` caps at it), so
        // "big" here means the full 32-row block, not an oversized tile.
        let big_range = Box2i::from_min_max(V2i::new(0, 0), V2i::new(127, 31)); // 128x32
        let small_range = Box2i::from_min_max(V2i::new(0, 70), V2i::new(127, 77)); // 128x8, strictly smaller

        let mut c = PizCompressor::new(&hdr, 32768, 32).unwrap();

        // First call: a big (full 32-row) tile with near-zero values (low
        // bitmap bit positions), to populate the persistent scratch buffers.
        // Wide/tall enough that the smooth ramp genuinely compresses instead
        // of falling to the "no-shrink -> raw store" passthrough, which
        // would skip the very bitmap/tmp/chan_line code this test targets.
        let big_line = build_line_buffer_at(128, 32, 0.0002);
        let big_compressed = c.compress_tile(&big_line, big_range).unwrap().to_vec();
        assert!(
            big_compressed.len() < big_line.len(),
            "test setup: the big tile must genuinely compress (got {} >= {}), or this test exercises raw store, not the codec",
            big_compressed.len(),
            big_line.len()
        );
        let big_restored = c.uncompress_tile(&big_compressed, big_range, 0).unwrap().to_vec();
        assert_eq!(big_restored, big_line, "sanity: the big tile itself must round-trip");

        // Second call, same compressor instance: a strictly smaller tile
        // with numerically LARGER values (higher bitmap bit positions) —
        // the big chunk's low-position stale bits, if not cleared, sit
        // BEFORE this chunk's own real entries and shift them.
        let small_line = build_line_buffer_at(128, 8, 50.0);
        let small_compressed = c.compress_tile(&small_line, small_range).unwrap().to_vec();
        assert!(
            small_compressed.len() < small_line.len(),
            "test setup: the small tile must genuinely compress (got {} >= {}), or this test exercises raw store, not the codec",
            small_compressed.len(),
            small_line.len()
        );
        let small_restored_reused = c.uncompress_tile(&small_compressed, small_range, 0).unwrap().to_vec();

        // Independent reference: a FRESH compressor decodes the identical
        // small chunk with no prior call, so it cannot see stale scratch
        // data from the big call above.
        let mut fresh = PizCompressor::new(&hdr, 32768, 32).unwrap();
        let small_restored_fresh = fresh.uncompress_tile(&small_compressed, small_range, 0).unwrap().to_vec();

        assert_eq!(
            small_restored_reused, small_restored_fresh,
            "the reused compressor's small-tile output diverged from a fresh instance's -- stale scratch data leaked from the earlier big tile"
        );
        assert_eq!(small_restored_reused, small_line, "the small tile must also match the real uncompressed input");
    }

    /// Scratch-buffer reuse across BOTH directions: `PizCompressor` now
    /// shares `bitmap`/`lut`/`tmp`/`chan_line` between `compress()` AND
    /// `uncompress()` (`internal_piz.c`'s single `scratch_buffer_1/2`,
    /// shape) — the test above only interleaved two
    /// `uncompress` calls; this one drives the full
    /// compress(A) -> uncompress(A) -> compress(B) -> uncompress(B) sequence
    /// on ONE instance (A big, B strictly smaller, well-separated value
    /// ranges — see [`build_line_buffer_at`]'s doc for why that separation
    /// is required), checking EVERY step byte-for-byte against a fresh
    /// instance: `compress(B)`'s OWN bytes must match a fresh compressor's
    /// `compress(B)` (proves `bitmap_from_data`/`forward_lut_from_bitmap`
    /// correctly re-init after A's compress AND A's decode both touched the
    /// same fields), and `uncompress(B)`'s bytes must match a fresh
    /// compressor's `uncompress(B)` of those same bytes.
    ///
    /// Mutation check performed while writing this test: temporarily
    /// guarding `bitmap.resize(BITMAP_SIZE, 0)` in `bitmap_from_data` with
    /// `if bitmap.is_empty()` (so a stale bit from compress(A)/uncompress(A)
    /// survives into compress(B)) made this test fail -- corrupting the
    /// bitmap/LUT badly enough that compress(B)'s own entropy-coded size no
    /// longer beat `srcb`, tripping the earlier "B must genuinely compress"
    /// assertion rather than the later byte-mismatch one. Still a genuine
    /// catch (the test DOES go red under the mutation), just via a different
    /// assertion than anticipated; reverted before committing.
    #[test]
    fn compress_and_uncompress_scratch_buffers_do_not_leak_across_a_full_sequence() {
        let hdr = make_header(128, 80, &[("R", PixelType::Half)]);
        let big_range = Box2i::from_min_max(V2i::new(0, 0), V2i::new(127, 31)); // 128x32
        let small_range = Box2i::from_min_max(V2i::new(0, 70), V2i::new(127, 77)); // 128x8

        let mut c = PizCompressor::new(&hdr, 32768, 32).unwrap();

        // compress(A) -- big, near-zero values.
        let a_line = build_line_buffer_at(128, 32, 0.0002);
        let a_compressed = c.compress_tile(&a_line, big_range).unwrap().to_vec();
        assert!(a_compressed.len() < a_line.len(), "test setup: A must genuinely compress");

        // uncompress(A) -- same instance.
        let a_restored = c.uncompress_tile(&a_compressed, big_range, 0).unwrap().to_vec();
        assert_eq!(a_restored, a_line, "sanity: A must round-trip");

        // compress(B) -- same instance, strictly smaller, numerically larger
        // values. This is the NEW thing this test covers: compress-side
        // scratch reuse, after BOTH a prior compress AND a prior decode
        // touched the same bitmap/lut/tmp/chan_line fields.
        let b_line = build_line_buffer_at(128, 8, 50.0);
        let b_compressed = c.compress_tile(&b_line, small_range).unwrap().to_vec();
        assert!(b_compressed.len() < b_line.len(), "test setup: B must genuinely compress");

        // Independent reference: a FRESH compressor's compress(B) must
        // produce the IDENTICAL bytes -- compress is a pure function of its
        // input given correctly-reinitialised scratch, so any divergence
        // here proves compress-side staleness.
        let mut fresh_for_compress = PizCompressor::new(&hdr, 32768, 32).unwrap();
        let b_compressed_fresh = fresh_for_compress.compress_tile(&b_line, small_range).unwrap().to_vec();
        assert_eq!(
            b_compressed, b_compressed_fresh,
            "the reused compressor's compress(B) bytes diverged from a fresh instance's -- stale scratch data leaked from A"
        );

        // uncompress(B) -- same instance.
        let b_restored_reused = c.uncompress_tile(&b_compressed, small_range, 0).unwrap().to_vec();

        // Independent reference for decode too.
        let mut fresh_for_decode = PizCompressor::new(&hdr, 32768, 32).unwrap();
        let b_restored_fresh = fresh_for_decode.uncompress_tile(&b_compressed, small_range, 0).unwrap().to_vec();
        assert_eq!(
            b_restored_reused, b_restored_fresh,
            "the reused compressor's uncompress(B) bytes diverged from a fresh instance's -- stale scratch data leaked from A"
        );
        assert_eq!(b_restored_reused, b_line, "B must also match the real uncompressed input");
    }

    /// `uncompress` sizes `out` once and scatters into it with
    /// no zero-fill, so the scatter must cover every byte. Decodes on a
    /// compressor whose EVERY scratch buffer (out, tmp, bitmap, lut, chan_line)
    /// was warmed by a prior call and then filled with 0xAA, on a layout with
    /// mixed sample types and x/y sub-sampling (rows a channel skips, channels
    /// narrower than the data window), and requires the exact input back plus
    /// a fresh instance's output. `PizOutKept` proves the warmed `out` was
    /// reused rather than silently replaced.
    #[test]
    fn uncompress_out_is_reused_and_poison_proof() {
        use crate::probe::{self, Probe};
        let chans: &[(&str, PixelType, i32, i32)] = &[
            ("A", PixelType::Float, 1, 1),
            ("B", PixelType::Half, 2, 1),
            ("G", PixelType::Half, 1, 2),
            ("R", PixelType::Uint, 2, 2),
        ];
        let (w, h) = (64i32, 32i32);
        let mut hdr = Header::with_dimensions(
            w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Piz,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for &(n, ty, xs, ys) in chans {
            cl.insert(n, Channel::new(ty, xs, ys, false)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        // Line buffer: per row, per channel on its sampled rows, `divp(max_x, xs) + 1` samples.
        let mut line = Vec::new();
        for y in 0..h {
            for (ci, &(_, ty, xs, ys)) in chans.iter().enumerate() {
                if y % ys != 0 {
                    continue;
                }
                for x in 0..((w - 1) / xs + 1) {
                    let v = (ci as f32) * 100.0 + (y as f32) + (x as f32) * 0.5 + 1.0;
                    match ty {
                        PixelType::Half => line.extend_from_slice(&Half::from(v).bits().to_le_bytes()),
                        PixelType::Float => line.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => line.extend_from_slice(&(v as u32).to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }

        let mut c = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        assert!(compressed.len() < line.len(), "test setup: fixture must not be stored raw");
        assert_eq!(c.uncompress(&compressed, 0, 0).unwrap(), line.as_slice(), "warm-up decode");

        c.out.fill(0xAA);
        c.tmp.fill(0xAAAA);
        c.bitmap.fill(0xAA);
        c.lut.fill(0xAAAA);
        c.chan_line.fill(0xAAAA);
        probe::reset(Probe::PizOutKept);
        let got = c.uncompress(&compressed, 0, 0).unwrap().to_vec();
        assert_eq!(got, line, "poisoned, reused scratch changed the decoded bytes");
        assert_eq!(probe::get(Probe::PizOutKept), 1, "the warmed `out` was not reused");

        let mut fresh = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        assert_eq!(fresh.uncompress(&compressed, 0, 0).unwrap(), got.as_slice());
    }

    /// A warmed `out` that is LARGER than the next chunk must be truncated to
    /// the exact decoded size (the keep-stale-bytes branch of `uncompress`):
    /// decode a big chunk, then a strictly smaller one on the SAME decoder with
    /// no `compress` in between (compress rewrites `out` and would hide it),
    /// and require the exact length and bytes of a fresh decoder.
    #[test]
    fn uncompress_truncates_a_larger_warm_out_to_the_smaller_chunk() {
        let hdr = make_header(128, 80, &[("R", PixelType::Half)]);
        let big_range = Box2i::from_min_max(V2i::new(0, 0), V2i::new(127, 31));
        let small_range = Box2i::from_min_max(V2i::new(0, 70), V2i::new(127, 77));
        let big_line = build_line_buffer_at(128, 32, 0.0002);
        let small_line = build_line_buffer_at(128, 8, 50.0);

        let mut enc = PizCompressor::new(&hdr, 32768, 32).unwrap();
        let big_c = enc.compress_tile(&big_line, big_range).unwrap().to_vec();
        let small_c = enc.compress_tile(&small_line, small_range).unwrap().to_vec();
        assert!(big_c.len() < big_line.len() && small_c.len() < small_line.len(), "test setup: must not be stored raw");

        let mut dec = PizCompressor::new(&hdr, 32768, 32).unwrap();
        assert_eq!(dec.uncompress_tile(&big_c, big_range, 0).unwrap(), big_line.as_slice());
        let got = dec.uncompress_tile(&small_c, small_range, 0).unwrap().to_vec();
        assert_eq!(got.len(), small_line.len(), "stale tail of the larger warm `out` survived");
        assert_eq!(got, small_line);
        let mut fresh = PizCompressor::new(&hdr, 32768, 32).unwrap();
        assert_eq!(fresh.uncompress_tile(&small_c, small_range, 0).unwrap(), got.as_slice());
    }

    #[test]
    fn round_trip_mixed_half_float() {
        round_trip(
            11,
            6,
            &[
                ("A", PixelType::Float),
                ("B", PixelType::Half),
                ("G", PixelType::Half),
                ("R", PixelType::Half),
                ("Z", PixelType::Float),
            ],
        );
    }

    #[test]
    fn round_trip_with_uint() {
        round_trip(8, 4, &[("H", PixelType::Half), ("U", PixelType::Uint)]);
    }

    #[test]
    fn round_trip_single_pixel() {
        round_trip(1, 1, &[("R", PixelType::Half)]);
    }

    #[test]
    fn round_trip_tall_thin() {
        round_trip(1, 32, &[("R", PixelType::Half)]);
    }

    #[test]
    fn round_trip_wide_short() {
        round_trip(40, 1, &[("R", PixelType::Half)]);
    }

    /// High-entropy data triggers the "no-shrink → store raw" path; round-trip
    /// must still be identity.
    #[test]
    fn round_trip_incompressible_raw_store() {
        let (w, h) = (6i32, 3i32);
        let chans = [("U", PixelType::Uint)];
        let hdr = make_header(w, h, &chans);
        let mut line = Vec::new();
        let mut state: u32 = 0x1357_9BDF;
        for _ in 0..(w * h) as usize {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            line.extend_from_slice(&state.to_le_bytes());
        }
        let mut c = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        // High-entropy data cannot shrink, so the raw-store path must fire.
        assert_eq!(
            compressed,
            line.as_slice(),
            "incompressible data must be stored raw (== original bytes)"
        );
    }

    /// `PizCompressor::compress` must take the raw-storage fallback
    /// when the Huffman entropy-coded bit count exceeds the ceiling passed to
    /// `huffman::compress_with_limit` — mirroring C's `hufEncode`
    /// `ARGUMENT_OUT_OF_RANGE` -> raw-store path (`internal_piz.c:596-603`,
    /// `internal_huf.c:846-848`). `huffman::tests::
    /// compress_with_limit_returns_none_over_ceiling` already proves the
    /// PRIMITIVE returns `None` over a tiny ceiling; this test proves
    /// `PizCompressor` itself is actually WIRED to that primitive and takes
    /// the fallback branch through the real `Compressor::compress` seam --
    /// which the huffman-level unit test, by construction, cannot show (it
    /// never touches `PizCompressor` at all). Uses the SAME genuinely
    /// compressible smooth-gradient image
    /// `round_trip_large_compressible_genuine_path` uses below (so WITHOUT
    /// the ceiling override this would take the real compressed path, not
    /// accidentally hit "no-shrink" for an unrelated reason), then lowers
    /// the ceiling via the test seam
    /// ([`PizCompressor::set_huf_bit_limit_for_test`]) to force the
    /// bit-count-exceeded branch without needing the real ~512 MiB+ fixture
    /// that would naturally trigger it.
    #[test]
    fn huf_bit_limit_exceeded_falls_back_to_raw_through_piz_compressor() {
        let (w, h) = (64i32, 32i32); // one full 32-line PIZ block
        let chans = [
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
        ];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        // Sanity: WITHOUT the override, this image genuinely compresses (the
        // real path, not raw-store for an unrelated reason).
        let mut baseline = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        let baseline_compressed = baseline.compress(&line, 0).unwrap();
        assert!(
            baseline_compressed.len() < line.len(),
            "fixture must genuinely compress without the bit-limit override, or this \
             test cannot distinguish the bit-limit fallback from an unrelated \
             no-shrink raw-store"
        );

        // Same image, same compressor construction, but with the test
        // seam forcing the "entropy-coded bit count doesn't fit" branch --
        // the same tiny ceiling `huffman::tests::
        // compress_with_limit_returns_none_over_ceiling` uses.
        let mut c = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        c.set_huf_bit_limit_for_test(8);
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        assert_eq!(
            compressed,
            line.as_slice(),
            "an over-limit Huffman bit count must fall back to storing the chunk \
             raw (byte-identical to the input), through the REAL PizCompressor \
             seam, not just the huffman primitive"
        );
        // Round-trips: a raw-stored chunk IS the original bytes (the reader
        // detects this by size and bypasses the codec entirely -- same
        // convention `round_trip`/`round_trip_incompressible_raw_store` use
        // above), so the equality assertion above already proves the
        // round-trip. Confirm it explicitly against a fresh compressor too,
        // to rule out any state the first `compress()` call could have left
        // behind affecting the comparison.
        let mut c2 = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        c2.set_huf_bit_limit_for_test(8);
        let compressed2 = c2.compress(&line, 0).unwrap().to_vec();
        assert_eq!(compressed2, line, "fallback must be deterministic across calls");
    }

    /// A genuinely compressible image (large, smooth gradients) must take the
    /// real PIZ path (compressed.len() < input) and `uncompress` must invert it
    /// exactly — this is the test that drives wavelet + huffman + bitmap/LUT
    /// end to end through the decode side.
    #[test]
    fn round_trip_large_compressible_genuine_path() {
        let (w, h) = (64i32, 32i32); // one full 32-line PIZ block
        let chans = [
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
        ];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);
        let mut c = PizCompressor::new(&hdr, line.len() + 64, 32).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        assert!(
            compressed.len() < line.len(),
            "smooth gradient image should compress (got {} >= {})",
            compressed.len(),
            line.len()
        );
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(restored, line.as_slice(), "genuine PIZ path round-trip mismatch");
    }

    #[test]
    fn num_scan_lines_is_32_and_format_xdr() {
        let hdr = make_header(4, 4, &[("R", PixelType::Half)]);
        let c = PizCompressor::new(&hdr, 8, 32).unwrap();
        assert_eq!(c.num_scan_lines(), 32);
        assert_eq!(c.format(), Format::Xdr);
    }
}
