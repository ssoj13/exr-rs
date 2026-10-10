//! Port of OpenEXR `ImfB44Compressor.cpp` / `.h` and the byte algorithm in
//! `OpenEXRCore/internal_b44.c` — the **B44** and **B44A** scanline codecs
//! (fixed-rate, lossy compression of 4×4 HALF pixel blocks).
//!
//! # What this codec does (1:1 with C++)
//!
//! B44 is **lossy for HALF channels** and **lossless (passthrough) for FLOAT
//! and UINT channels**.  Each HALF channel's sampled pixel grid is split into
//! 4×4 pixel blocks; each block (32 raw bytes = sixteen `u16`s) is packed into
//! a fixed **14 bytes** by `pack`.  The B44A variant additionally detects a
//! *flat* block — one where, after the running-difference quantisation, all
//! sixteen pixels are equal — and stores it in only **3 bytes** (`pack`
//! returns 3 with the `0xfc` flat marker in `b[2]`).  FLOAT/UINT channels are
//! copied verbatim, exactly as the C core does per channel by pixel type.
//!
//! ## Chunk processing (mirrors `compress_b44_impl` / `uncompress_b44_impl`)
//!
//! The block line buffer this codec receives is laid out line-by-line,
//! channel-by-channel (how [`crate::scanline::OutputFile`] gathers it).  Like
//! PIZ, B44 first **re-gathers** it into a per-channel-contiguous scratch
//! buffer (`scratch_buffer_1`: `for y { for c { copy channel line } }`).  Then
//! it walks channels in order:
//!
//! * **FLOAT/UINT** → raw-copy the whole channel region into the output.
//! * **HALF** → iterate 4×4 blocks across the channel's `nx × ny` grid.  When
//!   `nx`/`ny` is not a multiple of 4 the data are padded by repeating the
//!   rightmost column and the bottom row (exact C++ edge handling).  Each block
//!   is optionally `convertFromLinear`-mapped (when the channel's `p_linear`
//!   flag is set) through [`table::EXP_TABLE`], then `pack`ed.
//!
//! Decode reverses this: per channel, decode 14- or 3-byte blocks back into the
//! `nx × ny` grid (`convertToLinear` via [`table::LOG_TABLE`] when `p_linear`),
//! then re-interleave the per-channel scratch back into the line buffer.
//!
//! ## The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `compress_b44_impl`: after packing, if the total output is **strictly
//! greater** than the original packed bytes (`nOut > packed_bytes`), the codec
//! stores the original `packed_buffer` verbatim and sets the output size to
//! `packed_bytes`.  On disk this makes `dataSize == uncompressedSize`, which is
//! how the reader detects the raw passthrough (same convention as ZIP/PIZ).
//! We replicate this: when the assembled B44 output is `>` the input length we
//! return a copy of the original `data` unchanged.
//!
//! # Format
//!
//! Like the other ported scanline codecs, B44 consumes and produces the
//! machine-independent **Xdr** (little-endian) line buffer; every on-disk `u16`
//! is read with [`u16::from_le_bytes`] and written with [`u16::to_le_bytes`],
//! which is byte-identical to the C core's internal-native processing on any
//! host.  Reporting [`Format::Xdr`] keeps the existing `OutputFile`/`InputFile`
//! gather/scatter correct.
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfB44Compressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfB44Compressor.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_b44.c>

pub mod table;

use crate::attr::PixelType;
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};
use imath_rs::Box2i;

use imath_rs::modp;

/// Block edge length in pixels: B44 tiles HALF channels in 4×4 blocks.
const BLOCK: usize = 4;

// ---------------------------------------------------------------------------
// convertFromLinear / convertToLinear — table-driven half re-mapping
// ---------------------------------------------------------------------------

/// Map sixteen halves through [`table::EXP_TABLE`] — port of `convertFromLinear`.
/// Applied to a block before packing when the channel's `p_linear` flag is set.
#[inline]
fn convert_from_linear(s: &mut [u16; 16]) {
    for v in s.iter_mut() {
        *v = table::EXP_TABLE[*v as usize];
    }
}

/// Map sixteen halves through [`table::LOG_TABLE`] — port of `convertToLinear`.
/// Applied to a block after unpacking when the channel's `p_linear` flag is set.
#[inline]
fn convert_to_linear(s: &mut [u16; 16]) {
    for v in s.iter_mut() {
        *v = table::LOG_TABLE[*v as usize];
    }
}

// ---------------------------------------------------------------------------
// shiftAndRound — port of `shiftAndRound`
// ---------------------------------------------------------------------------

/// Compute `round(x * 2^-shift)` with round-half-to-even — port of
/// `shiftAndRound` in `internal_b44.c`.  All arithmetic is `i32`, matching the
/// C `int` operations exactly.
#[inline]
fn shift_and_round(x: i32, shift: i32) -> i32 {
    // y = x * pow(2, -shift), rounded to nearest, ties to even.
    let x = x << 1;
    let a = (1 << shift) - 1;
    let shift = shift + 1;
    let b = (x >> shift) & 1;
    (x + a + b) >> shift
}

// ---------------------------------------------------------------------------
// pack — port of `pack`
// ---------------------------------------------------------------------------

/// Pack a 4×4 block of halves (`s`, sixteen `u16`s) into `b`, returning the
/// number of bytes written: **14** for the normal encoding, or **3** for a flat
/// block when `flat_fields` is set (B44A).  Port of `pack` in `internal_b44.c`.
///
/// `b` must be at least 14 bytes long.  `exact_max` adjusts `t[0]` for the
/// maximum pixel (used for non-linear channels, i.e. `!p_linear`).
///
/// Bit layout is byte-identical to C++: `b[0..2]` = `t[0]` (big-endian halves),
/// `b[2]` = `(shift << 2) | (r[0] >> 4)`, then the 15 six-bit running
/// differences `r[0..15]` densely packed across `b[2..14]`.
fn pack(s: &[u16; 16], b: &mut [u8], flat_fields: bool, exact_max: bool) -> usize {
    /// Bias added to every running difference so the packed six-bit values land
    /// in `[0, 63]` (`const int bias = 0x20;` in C).
    const BIAS: i32 = 0x20;

    let mut t = [0u16; 16];
    for i in 0..16 {
        if (s[i] & 0x7c00) == 0x7c00 {
            // NaN/inf → bit pattern for +0.0 in the sign-magnitude `t` space.
            t[i] = 0x8000;
        } else if (s[i] & 0x8000) != 0 {
            t[i] = !s[i];
        } else {
            t[i] = s[i] | 0x8000;
        }
    }

    // find max — C: `tMax = 0; for ... if (tMax < t[i]) tMax = t[i];`
    let mut t_max: u16 = 0;
    for &v in t.iter() {
        if t_max < v {
            t_max = v;
        }
    }

    let mut d = [0i32; 16];
    let mut r = [0i32; 15];
    let mut shift: i32 = -1;
    let mut r_min: i32;
    let mut r_max: i32;

    // Find a shift such that all biased running differences fit in [0, 63].
    loop {
        shift += 1;

        // Absolute differences between tMax and t[i], shifted and rounded.
        for i in 0..16 {
            d[i] = shift_and_round(i32::from(t_max) - i32::from(t[i]), shift);
        }

        // Running differences (see the diagram in ImfB44Compressor.cpp).
        r[0] = d[0] - d[4] + BIAS;
        r[1] = d[4] - d[8] + BIAS;
        r[2] = d[8] - d[12] + BIAS;

        r[3] = d[0] - d[1] + BIAS;
        r[4] = d[4] - d[5] + BIAS;
        r[5] = d[8] - d[9] + BIAS;
        r[6] = d[12] - d[13] + BIAS;

        r[7] = d[1] - d[2] + BIAS;
        r[8] = d[5] - d[6] + BIAS;
        r[9] = d[9] - d[10] + BIAS;
        r[10] = d[13] - d[14] + BIAS;

        r[11] = d[2] - d[3] + BIAS;
        r[12] = d[6] - d[7] + BIAS;
        r[13] = d[10] - d[11] + BIAS;
        r[14] = d[14] - d[15] + BIAS;

        // C: `rMin = rMax = r[0]; for (i = 1; i < 15; ++i) { min/max }`.
        r_min = r[0];
        r_max = r[0];
        for &rv in r.iter().skip(1) {
            if r_min > rv {
                r_min = rv;
            }
            if r_max < rv {
                r_max = rv;
            }
        }

        if !(r_min < 0 || r_max > 0x3f) {
            break;
        }
    }

    if r_min == BIAS && r_max == BIAS && flat_fields {
        // Flat block (all pixels equal): 3-byte encoding with the 0xfc marker
        // in b[2], which cannot occur in the 14-byte form (shift <= 12 there).
        b[0] = (t[0] >> 8) as u8;
        b[1] = t[0] as u8;
        b[2] = 0xfc;
        return 3;
    }

    if exact_max {
        // Represent the max pixel as accurately as possible.
        // C: `t[0] = tMax - (uint16_t)(d[0] << shift);` — wrapping 16-bit sub.
        t[0] = t_max.wrapping_sub((d[0] << shift) as u16);
    }

    // Pack t[0], shift and r[0..15] into 14 bytes.
    b[0] = (t[0] >> 8) as u8;
    b[1] = t[0] as u8;

    b[2] = ((shift << 2) | (r[0] >> 4)) as u8;
    b[3] = ((r[0] << 4) | (r[1] >> 2)) as u8;
    b[4] = ((r[1] << 6) | r[2]) as u8;

    b[5] = ((r[3] << 2) | (r[4] >> 4)) as u8;
    b[6] = ((r[4] << 4) | (r[5] >> 2)) as u8;
    b[7] = ((r[5] << 6) | r[6]) as u8;

    b[8] = ((r[7] << 2) | (r[8] >> 4)) as u8;
    b[9] = ((r[8] << 4) | (r[9] >> 2)) as u8;
    b[10] = ((r[9] << 6) | r[10]) as u8;

    b[11] = ((r[11] << 2) | (r[12] >> 4)) as u8;
    b[12] = ((r[12] << 4) | (r[13] >> 2)) as u8;
    b[13] = ((r[13] << 6) | r[14]) as u8;

    14
}

// ---------------------------------------------------------------------------
// unpack14 / unpack3 — port of `unpack14` / `unpack3`
// ---------------------------------------------------------------------------

/// Six-bit mask used when decoding densely packed running differences.
const SIX_BITS: u32 = 0x3f;

/// Read byte `i` of a block as a `u32` (decode helper, mirrors the C casts).
#[inline]
fn b32(b: &[u8], i: usize) -> u32 {
    u32::from(b[i])
}

/// Unpack a 14-byte block into sixteen halves — port of `unpack14`.
/// `b` must be at least 14 bytes long.  All intermediate arithmetic uses `u32`,
/// matching the C casts, before truncation back to `u16`.
///
/// **Every operation here is deliberately `wrapping_*`.**  The C code is plain
/// `unsigned int` arithmetic (`ImfB44Compressor.cpp` `unpack14`), which wraps by
/// definition, and `shift` comes straight off the wire (`b[2] >> 2`, so 0..=63)
/// where C's `<< shift` on x86 masks the count to 5 bits.  On well-formed data
/// the running differences never underflow and `shift <= 12`, so the wrapping is
/// unobservable; on malformed (or extreme valid) data it is exactly what C
/// produces — verified against OpenEXR 3.4.15, which decodes an all-zero B44
/// chunk to the identical bytes (hash `a31022cf56ee0f83`).  Plain `+`/`-`/`<<`
/// made only overflow-checked builds (debug, test, fuzz) panic while release
/// already matched C; wrapping ops make every profile agree with C.
fn unpack14(b: &[u8], s: &mut [u16; 16]) {
    s[0] = ((b32(b, 0) << 8) | b32(b, 1)) as u16;

    let shift = b32(b, 2) >> 2;
    let bias = 0x20u32.wrapping_shl(shift);

    // `prev + (six_bit_delta << shift) - bias`, all wrapping (see above).
    let step = |prev: u16, six: u32| -> u16 {
        u32::from(prev)
            .wrapping_add((six & SIX_BITS).wrapping_shl(shift))
            .wrapping_sub(bias) as u16
    };

    s[4] = step(s[0], (b32(b, 2) << 4) | (b32(b, 3) >> 4));
    s[8] = step(s[4], (b32(b, 3) << 2) | (b32(b, 4) >> 6));
    s[12] = step(s[8], b32(b, 4));

    s[1] = step(s[0], b32(b, 5) >> 2);
    s[5] = step(s[4], (b32(b, 5) << 4) | (b32(b, 6) >> 4));
    s[9] = step(s[8], (b32(b, 6) << 2) | (b32(b, 7) >> 6));
    s[13] = step(s[12], b32(b, 7));

    s[2] = step(s[1], b32(b, 8) >> 2);
    s[6] = step(s[5], (b32(b, 8) << 4) | (b32(b, 9) >> 4));
    s[10] = step(s[9], (b32(b, 9) << 2) | (b32(b, 10) >> 6));
    s[14] = step(s[13], b32(b, 10));

    s[3] = step(s[2], b32(b, 11) >> 2);
    s[7] = step(s[6], (b32(b, 11) << 4) | (b32(b, 12) >> 4));
    s[11] = step(s[10], (b32(b, 12) << 2) | (b32(b, 13) >> 6));
    s[15] = step(s[14], b32(b, 13));

    // Map the sign-magnitude `t` space back to half bit patterns.
    for v in s.iter_mut() {
        if (*v & 0x8000) != 0 {
            *v &= 0x7fff;
        } else {
            *v = !*v;
        }
    }
}

/// Unpack a 3-byte flat block into sixteen identical halves — port of `unpack3`.
/// `b` must be at least 2 bytes long (only `b[0..2]` carry the value).
fn unpack3(b: &[u8], s: &mut [u16; 16]) {
    let mut v = ((b32(b, 0) << 8) | b32(b, 1)) as u16;
    if (v & 0x8000) != 0 {
        v &= 0x7fff;
    } else {
        v = !v;
    }
    for slot in s.iter_mut() {
        *slot = v;
    }
}

// ---------------------------------------------------------------------------
// ChannelInfo — per-channel layout captured from the header
// ---------------------------------------------------------------------------

/// One channel's region within the per-channel-contiguous scratch buffer:
/// `(byte_start, nx, ny)` where `byte_start` is the region's first byte index,
/// and `nx`/`ny` are the channel's sampled pixel extent within the chunk.  The
/// region holds `nx * ny * bytes_per_element` bytes.
type ChannelRegion = (usize, usize, usize);

/// Per-channel layout snapshot — the fields `internal_b44.c` reads from
/// `exr_coding_channel_info_t` (`bytes_per_element`, `x_samples`, `y_samples`,
/// `p_linear`, `data_type`).
#[derive(Clone, Copy, Debug)]
struct ChannelInfo {
    /// Sample type on disk; HALF is B44-compressed, FLOAT/UINT pass through.
    type_: PixelType,
    /// Horizontal sub-sampling factor.
    x_sampling: i32,
    /// Vertical sub-sampling factor.
    y_sampling: i32,
    /// Lossy-compression linearity hint — drives `convertFromLinear`/`ToLinear`
    /// and the `exact_max = !p_linear` argument to [`pack`].
    p_linear: bool,
}

impl ChannelInfo {
    /// Bytes per sample for this channel (HALF=2, FLOAT/UINT=4).
    fn bytes_per_element(&self) -> usize {
        match self.type_ {
            PixelType::Half => 2,
            PixelType::Uint | PixelType::Float => 4,
            PixelType::NumPixelTypes => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// B44Compressor
// ---------------------------------------------------------------------------

/// B44 / B44A scanline codec — port of `Imf::B44Compressor`.
///
/// `num_scan_lines()` is **32** (`getCompressionNumScanlines(B44/B44A) == 32`).
/// `optimize_flat` selects B44A (3-byte flat blocks) when `true`, plain B44
/// when `false` — exactly the `optFlatFields` ctor argument in C++.
///
/// Owns scratch buffers whose contents are returned as `&[u8]`; the slice is
/// valid until the next call (mirroring the C++ member output buffer).  The
/// channel list and data-window bounds are captured from the header at
/// construction; `compress`/`uncompress` drive the per-line, per-channel
/// re-gather/scatter from the chunk's `min_y` and that layout.
pub struct B44Compressor {
    /// Scan lines per chunk — `numScanLines()` (32 for B44/B44A).
    num_scan_lines: i32,
    /// `true` → B44A (flat-block 3-byte path enabled); `false` → plain B44.
    optimize_flat: bool,
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
    /// Per-channel-contiguous re-gather/decode scratch:
    /// reused across calls instead of a fresh `vec![0u8; total_bytes]` each
    /// time. Grown/truncated via [`super::ensure_len`], never unconditionally
    /// zero-filled -- see `compress`/`uncompress`'s comments for why every
    /// byte position is always fully overwritten before being read back on
    /// both sides (compress: the re-gather loop; uncompress: the raw-copy OR
    /// the 4x4-block decode loop, whose per-block valid extents partition the
    /// channel's full nx*ny grid with no gaps).
    tmp: Vec<u8>,
    /// Per-channel running line index within its region (compress AND
    /// uncompress re-interleave loops) — reused across calls, but always
    /// reset to all-zero at the start of a call (it is a counter, not scratch
    /// data being overwritten in place, so `ensure_len`'s "grow/truncate, no
    /// clear" policy does not apply here).
    chan_line: Vec<usize>,
    /// [`Self::channel_regions`]'s output buffer, reused across calls
    /// (instead of a `Vec::with_capacity` per call in the old
    /// `channel_regions`) — cleared and re-pushed every call, which is cheap
    /// (no zero-fill) and reuses the allocation once warmed up.
    regions: Vec<ChannelRegion>,
}

impl B44Compressor {
    /// Create a new B44/B44A compressor — port of `B44Compressor::B44Compressor`.
    ///
    /// `num_scan_lines` is 32.  `optimize_flat` is the `optFlatFields` flag
    /// (B44A when `true`).  The channel layout and data-window bounds are
    /// captured from `header`.  `_max_scan_line_size` matches the C++ ctor
    /// signature (this port sizes its scratch per call from the channel layout).
    pub fn new(
        header: &crate::header::Header,
        _max_scan_line_size: usize,
        num_scan_lines: i32,
        optimize_flat: bool,
    ) -> Result<Self> {
        let dw = header.data_window()?;
        let channel_list = header.channels()?;

        let channels: Vec<ChannelInfo> = channel_list
            .iter()
            .map(|(_, ch)| ChannelInfo {
                type_: ch.type_,
                x_sampling: ch.x_sampling,
                y_sampling: ch.y_sampling,
                p_linear: ch.p_linear,
            })
            .collect();

        Ok(Self {
            num_scan_lines,
            optimize_flat,
            channels,
            min_x: dw.min.x,
            max_x: dw.max.x,
            max_y: dw.max.y,
            out: Vec::new(),
            tmp: Vec::new(),
            chan_line: Vec::new(),
            regions: Vec::new(),
        })
    }

    /// Sampled scanline count of the block starting at `block_min_y` —
    /// `min(block_min_y + numScanLines - 1, max_y) - block_min_y + 1`.
    fn block_height(&self, block_min_y: i32) -> i32 {
        let block_max_y = (block_min_y + self.num_scan_lines - 1).min(self.max_y);
        block_max_y - block_min_y + 1
    }

    /// Per-channel horizontal sample count over the data window's x range —
    /// `divp(max_x, xs) - divp(min_x, xs) + 1` (matches the writer's per-line
    /// pixel count).  Uses `imath_rs::divp` for floor-division on negative x.
    fn channel_width(&self, x_sampling: i32) -> usize {
        let d_min = imath_rs::divp(self.min_x, x_sampling);
        let d_max = imath_rs::divp(self.max_x, x_sampling);
        (d_max - d_min + 1) as usize
    }

    /// Layout of each channel's region inside the per-channel-contiguous
    /// scratch (in bytes) — written into `self.regions`
    /// (`self.regions[c] = (byte_start, nx, ny)`, reused
    /// across calls instead of a fresh `Vec::with_capacity` each time) —
    /// plus the total byte count, returned directly.
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
            let bpe = ci.bytes_per_element();
            if bpe == 0 {
                return Err(ExrError::Arg("B44: unknown pixel type".into()));
            }
            self.regions.push((total, nx, ny));
            total += nx * ny * bpe;
        }

        Ok(total)
    }
}

impl Compressor for B44Compressor {
    /// 32 scan lines per chunk for B44/B44A.
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// B44 consumes/produces the machine-independent Xdr (little-endian) format.
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one chunk — port of `compress_b44_impl`.
    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]> {
        let srcb = data.len();
        if srcb == 0 {
            self.out.clear();
            return Ok(&self.out);
        }

        let block_height = self.block_height(min_y);
        let total_bytes = self.channel_regions(min_y, block_height)?;

        // --- Re-gather the line-interleaved Xdr line buffer into the per-channel-
        //     contiguous `tmp` byte array (C: `scratch_buffer_1` fill loop).
        // `ensure_len`: the loop below writes every byte of
        // every channel's region exactly once (each (y, channel) pair's
        // `copy_from_slice` covers `[dst, dst+bpl)`, and the regions/lines
        // partition `[0, total_bytes)` with no gaps -- verified by the
        // `n_in != srcb` check below, which fails closed if the layout ever
        // doesn't add up), so no unconditional zero-fill is needed.
        super::ensure_len(&mut self.tmp, total_bytes);
        let mut n_in = 0usize;
        // Per-channel running line index within its region -- a counter
        // reset to zero every call (not scratch data), so `clear` + `resize`
        // (not `ensure_len`) is the correct reuse here.
        self.chan_line.clear();
        self.chan_line.resize(self.channels.len(), 0);

        for y in 0..block_height {
            let cury = min_y + y;
            for (c, ci) in self.channels.iter().enumerate() {
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let (start, nx, _ny) = self.regions[c];
                let bpl = nx * ci.bytes_per_element(); // bytes per channel line
                if n_in + bpl > srcb {
                    return Err(ExrError::Logic(
                        "B44 compress: channel line overruns the input buffer".into(),
                    ));
                }
                let dst = start + self.chan_line[c] * bpl;
                self.tmp[dst..dst + bpl].copy_from_slice(&data[n_in..n_in + bpl]);
                n_in += bpl;
                self.chan_line[c] += 1;
            }
        }

        if n_in != srcb {
            return Err(ExrError::Logic(format!(
                "B44 compress: consumed {n_in} of {srcb} line-buffer bytes (layout mismatch)"
            )));
        }

        // --- Pack channels: FLOAT/UINT raw-copy, HALF 4×4-block pack.
        // Filled directly into self.out instead of a
        // throwaway local Vec later move-assigned — the same out/scratch
        // policy every other codec's compress AND uncompress path follows.
        self.out.clear();

        for (c, ci) in self.channels.iter().enumerate() {
            let (start, nx, ny) = self.regions[c];
            let bpe = ci.bytes_per_element();
            let n_bytes = nx * ny * bpe;
            if n_bytes == 0 {
                continue;
            }

            if ci.type_ != PixelType::Half {
                // Non-HALF channels are preserved exactly (raw copy).
                self.out.extend_from_slice(&self.tmp[start..start + n_bytes]);
                continue;
            }

            // HALF channel: iterate 4×4 blocks over the nx × ny grid.
            let region = &self.tmp[start..start + n_bytes];
            // Stride (in u16 samples) of one channel row.
            let row_stride = nx;

            let mut y = 0usize;
            while y < ny {
                // Row indices (in samples) with bottom-row padding by repetition.
                let r0 = y;
                let r1 = if y + 1 >= ny { r0 } else { y + 1 };
                let r2 = if y + 2 >= ny { r1 } else { y + 2 };
                let r3 = if y + 3 >= ny { r2 } else { y + 3 };
                let rows = [r0, r1, r2, r3];

                let mut x = 0usize;
                while x < nx {
                    let mut s = [0u16; 16];
                    let n = (nx - x).min(BLOCK); // valid columns in this block

                    // Fill the 4×4 block, padding the rightmost column by
                    // repeating column `n-1` (C++ edge handling).
                    for i in 0..BLOCK {
                        let col = x + i.min(n - 1);
                        for (ri, &row) in rows.iter().enumerate() {
                            let off = (row * row_stride + col) * 2;
                            s[i + ri * BLOCK] = u16::from_le_bytes([region[off], region[off + 1]]);
                        }
                    }

                    if ci.p_linear {
                        convert_from_linear(&mut s);
                    }

                    // Append the packed block (14 or 3 bytes) to the output.
                    let mut blk = [0u8; 14];
                    let w = pack(&s, &mut blk, self.optimize_flat, !ci.p_linear);
                    self.out.extend_from_slice(&blk[..w]);

                    x += BLOCK;
                }
                y += BLOCK;
            }
        }

        // --- "No-shrink → store raw": C sets nOut = packed_bytes when nOut > it.
        if self.out.len() > srcb {
            self.out.clear();
            self.out.extend_from_slice(data);
        }

        Ok(&self.out)
    }

    /// Uncompress one B44 chunk — port of `uncompress_b44_impl`.
    ///
    /// `expected_len` is unused: B44's decode size is
    /// already driven entirely by `channel_regions(min_y, block_height)`,
    /// never a construction-time upper bound.
    fn uncompress(&mut self, data: &[u8], min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }

        let block_height = self.block_height(min_y);
        let total_bytes = self.channel_regions(min_y, block_height)?;

        // `internal_exr_undo_b44` itself has no
        // raw-passthrough check (`compression.c:416-421`) — real reader seams
        // detect `dataSize == uncompressedSize` one level up before calling the
        // codec (see `crate::compressor::stored_chunk`'s rustdoc). But this
        // codec's own round-trip tests (`assert_idempotent` et al.) call
        // `compress`/`uncompress` directly without going through a reader seam,
        // and `compress` legitimately returns the raw bytes verbatim on the
        // "no-shrink" path — so `uncompress` must still accept them standalone.
        // Removing this check breaks those tests; kept, with this note so a
        // future cleanup doesn't "just delete it"
        // without re-verifying it first.
        if data.len() == total_bytes {
            self.out.clear();
            self.out.extend_from_slice(data);
            return Ok(&self.out);
        }

        // --- Decode blocks per channel into the per-channel-contiguous `tmp`.
        // `ensure_len`: every byte in `[0, total_bytes)` is
        // written before being read back, on BOTH branches below --
        // non-HALF via a full `copy_from_slice`, HALF via the 4x4-block
        // write-back loop, whose per-block valid extents (`n`, `rows_valid`,
        // both clipped to the channel's remaining width/height) partition the
        // channel's full `nx * ny` grid exactly, with no gaps and no overlap
        // -- so no unconditional zero-fill is needed (verified with a
        // poison test: see `poison_reused_scratch_matches_fresh_instance`
        // below).
        super::ensure_len(&mut self.tmp, total_bytes);
        let mut in_i = 0usize;
        let remaining = data.len();

        for (c, ci) in self.channels.iter().enumerate() {
            let (start, nx, ny) = self.regions[c];
            let bpe = ci.bytes_per_element();
            let n_bytes = nx * ny * bpe;
            if n_bytes == 0 {
                continue;
            }

            if ci.type_ != PixelType::Half {
                // Non-HALF channels are stored raw.
                if in_i + n_bytes > remaining {
                    return Err(ExrError::Input("B44 uncompress: truncated raw channel".into()));
                }
                self.tmp[start..start + n_bytes].copy_from_slice(&data[in_i..in_i + n_bytes]);
                in_i += n_bytes;
                continue;
            }

            // HALF channel: decode 14- or 3-byte blocks back into the grid.
            let row_stride = nx;

            let mut y = 0usize;
            while y < ny {
                let mut x = 0usize;
                while x < nx {
                    if in_i + 3 > remaining {
                        return Err(ExrError::Input("B44 uncompress: not enough data".into()));
                    }

                    let mut s = [0u16; 16];
                    // 3-byte flat block is signalled by b[2] >= (13 << 2).
                    if data[in_i + 2] >= (13 << 2) {
                        unpack3(&data[in_i..in_i + 3], &mut s);
                        in_i += 3;
                    } else {
                        if in_i + 14 > remaining {
                            return Err(ExrError::Input("B44 uncompress: not enough data".into()));
                        }
                        unpack14(&data[in_i..in_i + 14], &mut s);
                        in_i += 14;
                    }

                    if ci.p_linear {
                        convert_to_linear(&mut s);
                    }

                    // Valid extent of this block within the grid (no padding on
                    // write-back — only the in-bounds pixels are stored).
                    let n = (nx - x).min(BLOCK);
                    let rows_valid = (ny - y).min(BLOCK);
                    for ri in 0..rows_valid {
                        let row = y + ri;
                        for i in 0..n {
                            let col = x + i;
                            let off = start + (row * row_stride + col) * 2;
                            let bytes = s[i + ri * BLOCK].to_le_bytes();
                            self.tmp[off] = bytes[0];
                            self.tmp[off + 1] = bytes[1];
                        }
                    }

                    x += BLOCK;
                }
                y += BLOCK;
            }
        }

        // --- Re-interleave the per-channel scratch back into the line buffer.
        self.out.clear();
        self.chan_line.clear();
        self.chan_line.resize(self.channels.len(), 0);

        for y in 0..block_height {
            let cury = min_y + y;
            for (c, ci) in self.channels.iter().enumerate() {
                if modp(cury, ci.y_sampling) != 0 {
                    continue;
                }
                let (start, nx, _ny) = self.regions[c];
                let bpl = nx * ci.bytes_per_element();
                let src = start + self.chan_line[c] * bpl;
                self.out.extend_from_slice(&self.tmp[src..src + bpl]);
                self.chan_line[c] += 1;
            }
        }

        Ok(&self.out)
    }

    /// Compress one tile, encoded against the tile's own pixel rectangle.
    ///
    /// B44/B44A derive their chunk layout from `min_x`/`max_x` (per-channel
    /// width) and `max_y` (final-block clamp); for a tile these are the tile's
    /// X/Y extent.  Mirrors C++ `compressTile` feeding the tile's `Box2i range`
    /// into the encode step.  Sets the rectangle fields from `range`, then runs
    /// the identical compress path at `range.min.y` (fresh each tile call).
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

    fn make_header(w: i32, h: i32, chans: &[(&str, PixelType, bool)]) -> Header {
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::B44,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for (name, ty, lin) in chans {
            cl.insert(*name, Channel::new(*ty, 1, 1, *lin)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr
    }

    /// An all-zero 14-byte block underflows `unpack14`'s running
    /// differences.  That is C's defined-by-wrapping behaviour, not an error —
    /// real OpenEXR 3.4.15 decodes an all-zero B44 chunk to exactly these bits
    /// (output hash `a31022cf56ee0f83`).  Only overflow-checked builds used to
    /// panic here (`b44/mod.rs:261`, "attempt to subtract with overflow"), so
    /// this test only means anything in a debug/test profile — which is what it
    /// runs in.
    #[test]
    fn unpack14_all_zero_block_wraps_like_c() {
        let mut s = [0u16; 16];
        unpack14(&[0u8; 14], &mut s);
        // b[2] == 0 => shift 0, bias 0x20.  s[0] = 0 -> !0 = 0xffff; each
        // running difference wraps by -0x20 and the sign-magnitude fixup masks
        // off the top bit: 0xffe0 -> 0x7fe0, 0xffc0 -> 0x7fc0, ...
        assert_eq!(s[0], 0xffff);
        assert_eq!(s[4], 0x7fe0);
        assert_eq!(s[8], 0x7fc0);
        assert_eq!(s[12], 0x7fa0);
    }

    /// `shift` is read straight off the wire (`b[2] >> 2`, so 0..=63).
    /// `0x20u32 << 63` is a shift-overflow panic in a checked build and a
    /// 5-bit-masked shift in release / in C on x86; `wrapping_shl` makes every
    /// profile agree with C.  Sweeping every shift must not unwind.
    #[test]
    fn unpack14_extreme_shift_does_not_panic() {
        for hi in 0u16..=255 {
            for b2 in [0u8, 0x3f, 0x80, 0xfc, 0xff] {
                let mut b = [0xffu8; 14];
                b[0] = (hi >> 8) as u8;
                b[1] = hi as u8;
                b[2] = b2;
                let mut s = [0u16; 16];
                unpack14(&b, &mut s);
                std::hint::black_box(&s);
            }
        }
    }

    /// End to end: a zeroed B44 chunk must decode (to garbage) rather than
    /// panic, in every build profile.
    #[test]
    fn zeroed_b44_chunk_decodes_without_panicking() {
        let (w, h) = (16i32, 16i32);
        let chans = [("Y", PixelType::Half, false)];
        let hdr = make_header(w, h, &chans);
        let src = build_line_buffer(w, h, &chans);
        let mut c = B44Compressor::new(&hdr, src.len() + 64, 32, false).unwrap();
        let packed = c.compress(&src, 0).unwrap().to_vec();

        let zeroed = vec![0u8; packed.len()];
        let out = c.uncompress(&zeroed, 0, 0).unwrap();
        assert_eq!(out.len(), src.len());
    }

    /// Build a line-interleaved Xdr block buffer: for each line, for each
    /// channel (in `chans` order), `w` samples of deterministic values.
    fn build_line_buffer(w: i32, block_height: i32, chans: &[(&str, PixelType, bool)]) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..block_height {
            for (ci, (_, ty, _)) in chans.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 8.0 + (y as f32) * 0.25 + x as f32 * 0.5 + 0.5;
                    match ty {
                        PixelType::Half => {
                            buf.extend_from_slice(&Half::from_f32(v).bits().to_le_bytes())
                        }
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&((v as u32).to_le_bytes())),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// Drive the codec exactly as the scanline reader does (compress, then
    /// `uncompress`, which handles the raw-passthrough internally), assert the
    /// decode is faithful against an INDEPENDENT expectation (not the codec's
    /// own re-encoding, which an all-zero decoder would trivially satisfy),
    /// then assert decode == re-encode-decode idempotence: B44 is lossy, but
    /// the second round-trip of an already-decoded buffer must reproduce
    /// identical bytes (the quantiser is a fixed point after the first pass).
    fn assert_idempotent(w: i32, h: i32, chans: &[(&str, PixelType, bool)], optimize_flat: bool) {
        let hdr = make_header(w, h, chans);
        let line = build_line_buffer(w, h, chans);
        let mut c = B44Compressor::new(&hdr, line.len() + 64, 32, optimize_flat).unwrap();
        let comp1 = c.compress(&line, 0).unwrap().to_vec();
        let dec1 = c.uncompress(&comp1, 0, 0).unwrap().to_vec();
        assert_eq!(dec1.len(), line.len());

        // Independent expectation, byte-by-byte against the ORIGINAL buffer
        // `build_line_buffer` produced (not something the compressor computed):
        // FLOAT/UINT channels are documented to pass through B44 exactly; HALF
        // channels are B44-quantised but must stay within the codec's known
        // tolerance of the true value. An all-zero (or otherwise broken)
        // decoder fails both checks even though it would pass a bare
        // decode-vs-redecode self-comparison.
        let mut off = 0usize;
        for y in 0..h {
            for (name, ty, _) in chans.iter() {
                for x in 0..w {
                    match ty {
                        PixelType::Half => {
                            let orig = f32::from(Half::from_bits(u16::from_le_bytes([
                                line[off], line[off + 1],
                            ])));
                            let got = f32::from(Half::from_bits(u16::from_le_bytes([
                                dec1[off], dec1[off + 1],
                            ])));
                            let rel = (got - orig).abs() / orig.abs().max(1.0);
                            assert!(
                                rel < 0.05,
                                "B44 HALF chan {name} at ({x},{y}) out of tolerance: \
                                 {orig} vs {got} ({w}x{h} {chans:?})"
                            );
                            off += 2;
                        }
                        PixelType::Float | PixelType::Uint => {
                            assert_eq!(
                                &dec1[off..off + 4],
                                &line[off..off + 4],
                                "B44 chan {name} ({ty:?}) must pass through exactly at \
                                 ({x},{y}) ({w}x{h} {chans:?})"
                            );
                            off += 4;
                        }
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }

        let mut c2 = B44Compressor::new(&hdr, dec1.len() + 64, 32, optimize_flat).unwrap();
        let comp2 = c2.compress(&dec1, 0).unwrap().to_vec();
        let dec2 = c2.uncompress(&comp2, 0, 0).unwrap().to_vec();
        assert_eq!(dec1, dec2, "B44 decode not idempotent ({w}x{h} {chans:?})");
    }

    #[test]
    fn round_trip_half_basic() {
        assert_idempotent(9, 5, &[("R", PixelType::Half, false)], false);
    }

    #[test]
    fn round_trip_b44a_flat_block() {
        // A constant HALF channel must hit the flat 3-byte path under B44A and
        // round-trip idempotently.
        let (w, h) = (8i32, 8i32);
        let chans = [("R", PixelType::Half, false)];
        let hdr = make_header(w, h, &chans);
        // Build a constant-value line buffer (all the same half).
        let val = Half::from_f32(1.5).bits().to_le_bytes();
        let mut line = Vec::new();
        for _ in 0..(w * h) as usize {
            line.extend_from_slice(&val);
        }
        let mut c = B44Compressor::new(&hdr, line.len() + 64, 32, true).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        // 4 blocks (2x2) × 3 bytes = 12 bytes (flat path), well below 14×4.
        assert_eq!(compressed.len(), 12, "B44A flat blocks must be 3 bytes each");
        let restored = c.uncompress(&compressed, 0, 0).unwrap().to_vec();
        assert_eq!(restored, line, "B44A flat block round-trip mismatch");
    }

    #[test]
    fn round_trip_b44_no_flat_uses_14_bytes() {
        // Same constant input under plain B44 (optimize_flat=false) must use the
        // 14-byte encoding for every block.
        let (w, h) = (8i32, 8i32);
        let chans = [("R", PixelType::Half, false)];
        let hdr = make_header(w, h, &chans);
        let val = Half::from_f32(1.5).bits().to_le_bytes();
        let mut line = Vec::new();
        for _ in 0..(w * h) as usize {
            line.extend_from_slice(&val);
        }
        let mut c = B44Compressor::new(&hdr, line.len() + 64, 32, false).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        // 4 blocks × 14 bytes = 56 bytes; raw is 8*8*2 = 128, so no raw fallback.
        assert_eq!(compressed.len(), 56, "B44 blocks must be 14 bytes each");
        let restored = c.uncompress(&compressed, 0, 0).unwrap().to_vec();
        assert_eq!(restored, line, "B44 constant round-trip mismatch");
    }

    /// Before dropping the zero-fill, verify with
    /// a poison test: `self.tmp` used to be a fresh
    /// `vec![0u8; total_bytes]` per call; it is now reused via `ensure_len`
    /// (grow/truncate, no zero-fill). If any byte of a channel's region were
    /// EVER left un-written by the compress/decode loops, this test would
    /// leak stale poison bytes into the round-tripped output. Deliberately
    /// pre-fills `tmp` with `0xAA` (never a value either loop would produce
    /// by construction) on the instance under test, before either `compress`
    /// or `uncompress` has ever been called on it, then diffs both stages
    /// against a fresh instance that never saw poison.
    #[test]
    fn poison_reused_scratch_matches_fresh_instance() {
        let chans: &[(&str, PixelType, bool)] = &[
            ("A", PixelType::Float, false),
            ("H", PixelType::Half, false),
            ("U", PixelType::Uint, false),
        ];
        let (w, h) = (11, 6);
        let hdr = make_header(w, h, chans);
        let line = build_line_buffer(w, h, chans);

        let mut poisoned = B44Compressor::new(&hdr, line.len() + 64, 32, true).unwrap();
        // Poison BEFORE the first real call — bigger than any single call's
        // `total_bytes` will need, so both the compress-side and (after
        // `ensure_len` truncates it back down) the uncompress-side reuse of
        // the same field start from genuinely stale bytes, not zeros.
        poisoned.tmp = vec![0xAAu8; line.len() * 4 + 4096];

        let mut fresh = B44Compressor::new(&hdr, line.len() + 64, 32, true).unwrap();

        let comp_poisoned = poisoned.compress(&line, 0).unwrap().to_vec();
        let comp_fresh = fresh.compress(&line, 0).unwrap().to_vec();
        assert_eq!(
            comp_poisoned, comp_fresh,
            "B44 compress: a poisoned (reused) `tmp` scratch buffer produced \
             different compressed bytes than a fresh instance -- some byte of \
             `tmp` is not fully overwritten before being read"
        );

        // Re-poison between compress and uncompress: `tmp` was already
        // truncated down to `total_bytes` by the compress-side `ensure_len`
        // call above, so re-poison it (padded) again to prove the
        // UNCOMPRESS-side write-back loops (raw copy + 4x4-block decode) are
        // themselves complete, independent of what compress left behind.
        let extra_pad = poisoned.tmp.len() + 4096;
        poisoned.tmp = vec![0x55u8; extra_pad];

        let dec_poisoned = poisoned.uncompress(&comp_poisoned, 0, 0).unwrap().to_vec();
        let dec_fresh = fresh.uncompress(&comp_fresh, 0, 0).unwrap().to_vec();
        assert_eq!(
            dec_poisoned, dec_fresh,
            "B44 uncompress: a poisoned (reused) `tmp` scratch buffer produced \
             different decoded bytes than a fresh instance -- some byte of \
             `tmp` is not fully overwritten before being read"
        );
        // FLOAT/UINT pass through exactly; HALF is intentionally lossy
        // (B44-quantised) -- `round_trip_mixed_half_float_uint` already
        // covers the tolerance check for this exact channel layout. This
        // test's job is only the poisoned-vs-fresh comparison above.
        assert_eq!(dec_poisoned.len(), line.len());
    }

    #[test]
    fn round_trip_multi_half() {
        assert_idempotent(
            13,
            7,
            &[
                ("B", PixelType::Half, false),
                ("G", PixelType::Half, false),
                ("R", PixelType::Half, false),
            ],
            true,
        );
    }

    #[test]
    fn round_trip_mixed_half_float_uint() {
        // FLOAT/UINT must pass through exactly; HALF is B44-coded.
        assert_idempotent(
            11,
            6,
            &[
                ("A", PixelType::Float, false),
                ("H", PixelType::Half, false),
                ("U", PixelType::Uint, false),
            ],
            true,
        );
    }

    #[test]
    fn float_uint_only_keeps_size_matching_cpp() {
        // A FLOAT+UINT-only image has no HALF data: B44 raw-copies each channel
        // into a per-channel-grouped buffer of the SAME total size as the input.
        // C++ `compress_b44_impl` keeps that output whenever `nOut <= packed_bytes`
        // (here nOut == packed_bytes), so `compressed_bytes == unpacked_size`.  On
        // read, the C core's universal `packsz == unpacksz` rule (compression.c)
        // raw-copies the bytes and never calls `undo_b44`.  We therefore only
        // assert the encoder output size equals the input (byte-exact vs C++);
        // the genuine FLOAT/UINT-survival case is covered by
        // `round_trip_mixed_half_float_uint`, where HALF data shrinks the chunk
        // and the real decode path runs.
        let (w, h) = (5i32, 3i32);
        let chans = [("A", PixelType::Float, false), ("U", PixelType::Uint, false)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);
        let mut c = B44Compressor::new(&hdr, line.len() + 64, 32, true).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        assert_eq!(
            compressed.len(),
            line.len(),
            "FLOAT/UINT-only B44 output must equal the unpacked size (C++ parity)"
        );
    }

    #[test]
    fn round_trip_linear_channel() {
        // p_linear=true exercises the EXP/LOG table conversion path.
        assert_idempotent(12, 8, &[("Y", PixelType::Half, true)], true);
    }

    #[test]
    fn round_trip_single_pixel() {
        assert_idempotent(1, 1, &[("R", PixelType::Half, false)], true);
    }

    #[test]
    fn round_trip_tall_thin() {
        assert_idempotent(1, 32, &[("R", PixelType::Half, false)], true);
    }

    #[test]
    fn round_trip_wide_short() {
        assert_idempotent(40, 1, &[("R", PixelType::Half, false)], false);
    }

    #[test]
    fn num_scan_lines_is_32_and_format_xdr() {
        let hdr = make_header(4, 4, &[("R", PixelType::Half, false)]);
        let c = B44Compressor::new(&hdr, 8, 32, false).unwrap();
        assert_eq!(c.num_scan_lines(), 32);
        assert_eq!(c.format(), Format::Xdr);
    }

    /// `pack`/`unpack14` must be exact inverses up to the documented quantiser:
    /// packing then unpacking a flat (all-equal) block reproduces the value.
    #[test]
    fn pack_unpack_flat_value_exact() {
        let v = Half::from_f32(3.25).bits();
        let s = [v; 16];
        let mut b = [0u8; 14];
        let w = pack(&s, &mut b, false, true);
        assert_eq!(w, 14);
        let mut out = [0u16; 16];
        unpack14(&b, &mut out);
        assert_eq!(out, [v; 16], "flat block must survive pack/unpack14");
    }
}
