//! Port of OpenEXR 3.4 `ImfHTCompressor.cpp` + `OpenEXRCore/internal_ht.cpp` +
//! `internal_ht_common.cpp` — the **HTJ2K** (High-Throughput JPEG 2000) scanline
//! codec added in OpenEXR 3.4.
//!
//! # 3.4 extension
//! HTJ2K is *not* present in the v3.3.11 base port target; it is an OpenEXR 3.4
//! addition (`HTJ2K256_COMPRESSION = 10`, `HTJ2K32_COMPRESSION = 11`).  This
//! module is a deliberate, documented step toward 3.4 for this one codec.  The
//! JPEG 2000 codestream is produced/consumed by [`jph_rs`] (our pure-Rust
//! OpenJPH port); the EXR-specific chunk framing + channel mapping is ported 1:1
//! from `internal_ht_common.cpp`.
//!
//! # What this codec does (1:1 with C++)
//!
//! Each EXR chunk (256 scanlines for `HTJ2K256`, 32 for `HTJ2K32`) is encoded as
//! an **independent** HTJ2K codestream that covers only that chunk's scanlines:
//! `internal_ht.cpp` uses `image_height = encode->chunk.height` and
//! `image_width = encode->chunk.width`.  Per the C++:
//!
//! * **channel map** (`make_channel_map`) — detects an `R/G/B` (or
//!   `red/green/blue`) triplet of identical type+sampling so the JPEG 2000
//!   Reversible Colour Transform (RCT) can decorrelate them; the matched
//!   channels become codestream components 0/1/2 and the rest follow in file
//!   order.  Otherwise components map 1:1 to file channels.
//! * **codestream params** — `set_num_components`, per-component
//!   `set_comp_info(downsampling, bit_depth=16|32, signed=type!=UINT)`, a
//!   `OJPH_NLT_BINARY_COMPLEMENT_NLT` non-linear transform for every non-UINT
//!   component, planar mode iff any channel is sub-sampled, RCT colour transform
//!   iff RGB and not planar, reversible wavelet (lossless), `128×32` code-blocks,
//!   5 decomposition levels.
//!
//!   **exr-rs extension (not in OpenEXR):** when the header carries the optional
//!   `htj2kQuantStep` attribute the wavelet switches to the irreversible 9/7
//!   path with that quantization step (LOSSY). OpenEXR's HTJ2K is reversible-only
//!   and defines no quality knob; absent the attribute the encode stays lossless
//!   and byte-faithful to the reference. Decode is unaffected — the JPEG 2000
//!   codestream carries its own quantization (QCD), so the attribute is an
//!   encode-time signal only.
//! * **pixel feed** — HALF samples are read as little-endian `i16` and
//!   sign-extended into the codec's `i32` line buffer; FLOAT/UINT samples are
//!   bit-cast from little-endian `i32`.  The interleaved (non-planar) path walks
//!   the chunk line-by-line, feeding each codestream component from its file
//!   channel's slice (`raster_line_offset`); the planar path walks each
//!   component fully before the next.
//!
//! Decompression reverses this: read the chunk header, decode the codestream,
//! and scatter each component's `i32` line back to little-endian HALF/FLOAT/UINT
//! bytes at the right `raster_line_offset`.
//!
//! # Format (Xdr)
//!
//! HTJ2K does not override [`Compressor::format`], so the line buffer handed in
//! is already in the on-disk little-endian Xdr layout; the per-sample reads
//! below decode those little-endian bytes (matching the C++ `int16_t*`/`int32_t*`
//! reinterpret-casts on a little-endian host).
//!
//! # The "no-shrink → store raw" rule (byte-affecting; replicated exactly)
//!
//! `ht_apply_impl` writes the codestream into a fixed buffer of size
//! `packed_bytes` (the uncompressed chunk size); if the codestream would
//! overflow it, OpenJPH's `staticmem_outfile::write` throws `std::range_error`,
//! which `ht_apply_impl` catches and sets `encode->compressed_bytes =
//! encode->packed_bytes`, i.e. the encode pipeline then stores the **raw**
//! packed bytes.  We replicate: if `header + codestream` is not strictly smaller
//! than the uncompressed input, we return a copy of the original `data`.  On
//! disk this means `dataSize == uncompressedSize`, which the reader's chunk loop
//! already treats as the raw passthrough (same convention as ZIP/PXR24/B44).
//!
//! Port targets:
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.0/src/lib/OpenEXR/ImfHTCompressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.0/src/lib/OpenEXRCore/internal_ht.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.0/src/lib/OpenEXRCore/internal_ht_common.cpp>

use crate::attr::PixelType;
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};
use crate::misc::active_row_count;

use imath_rs::{Box2i, divp};
use jph_rs::{Codestream, MemInfile, MemOutfile, Point, Ui32};

/// `OJPH_NLT_BINARY_COMPLEMENT_NLT` — the non-linear transform OpenEXR applies
/// to every non-UINT component (`internal_ht.cpp` line 377).
const NLT_BINARY_COMPLEMENT: u8 = 3;

/// `HEADER_MARKER = 'H' * 256 + 'T'` (`internal_ht_common.cpp` line 262).
const HEADER_MARKER: u16 = (b'H' as u16) * 256 + b'T' as u16;

/// `HEADER_SZ = 6` — the fixed MAGIC(u16) + PLEN(u32) frame
/// (`internal_ht_common.cpp` line 263).
const HEADER_FRAME_SZ: usize = 6;

// ---------------------------------------------------------------------------
// Per-channel layout snapshot (from the header), in FILE order.
// ---------------------------------------------------------------------------

/// Per-channel layout the codec needs, captured from the header at construction
/// — mirrors the `exr_coding_channel_info_t` fields `internal_ht.cpp` reads
/// (`data_type`, `x_samples`, `y_samples`, `width`, `bytes_per_element`).
///
/// Stored in the file's channel-list (alphabetical) order — the same order the
/// scanline gather lays the line buffer out in.
#[derive(Clone, Copy, Debug)]
struct ChannelInfo {
    /// Sample type on disk (drives bit depth + the HALF vs FLOAT/UINT feed).
    type_: PixelType,
    /// Horizontal sub-sampling factor (`x_samples`).
    x_sampling: i32,
    /// Vertical sub-sampling factor (`y_samples`).
    y_sampling: i32,
}

/// One codestream-component → file-channel mapping entry — port of
/// `CodestreamChannelInfo` (`internal_ht_common.h`).
#[derive(Clone, Copy, Debug, Default)]
struct CodestreamChannelInfo {
    /// EXR (file) channel index for this codestream component.
    file_index: usize,
    /// Byte offset of this file channel within one gathered scanline.  Computed
    /// from the file-order channel widths; used only by the interleaved path.
    raster_line_offset: usize,
}

// ---------------------------------------------------------------------------
// Htj2kCompressor — the Compressor seam implementation
// ---------------------------------------------------------------------------

/// HTJ2K scanline codec — port of `Imf::HTCompressor` / `internal_exr_apply_ht`
/// / `internal_exr_undo_ht`.
///
/// `num_scan_lines()` is **256** (`HTJ2K256`) or **32** (`HTJ2K32`).  Owns
/// scratch buffers whose contents are returned as `&[u8]` by
/// [`compress`](Compressor::compress) / [`uncompress`](Compressor::uncompress);
/// the slice is valid until the next call (mirroring the C++ member buffer).
///
/// The channel list + data-window bounds are captured from the header at
/// construction (the C core gets these from `encode->channels` / `encode->chunk`);
/// each call drives the per-line/per-component loop from the chunk's `min_y` and
/// the captured layout.
pub struct Htj2kCompressor {
    /// Scan lines per chunk — 256 (`HTJ2K256`) or 32 (`HTJ2K32`).
    num_scan_lines: i32,
    /// Per-channel layout in file (channel-list) order.
    channels: Vec<ChannelInfo>,
    /// Data-window minimum x — to compute each channel's per-line width.
    min_x: i32,
    /// Data-window maximum x.
    max_x: i32,
    /// Data-window maximum y — clamps the final (possibly partial) chunk.
    max_y: i32,
    /// File channel names, in channel-list order — needed for the RGB heuristic
    /// in [`make_channel_map`] (which keys off `r/g/b` / `red/green/blue`
    /// suffixes).  Parallel to [`Self::channels`].
    names: Vec<crate::attr::Name>,
    /// Scratch buffer holding the result of the last call (the C++ output buffer).
    out: Vec<u8>,
    /// Chunk-local metadata storage; contents are rebuilt on every call.
    channel_map: Vec<CodestreamChannelInfo>,
    /// File-order byte offsets, recomputed for the current chunk rectangle.
    file_offsets: Vec<usize>,
    /// Duplicate-channel flags, reset before validating each wire map.
    seen: Vec<bool>,
    /// Framing bytes, cleared before each encode without discarding capacity.
    header_buf: Vec<u8>,
    /// `false` selects the lossy (irreversible 9/7) HTJ2K path with [`Self::qstep`];
    /// `true` (the default, OpenEXR-faithful) keeps the lossless reversible 5/3 path.
    reversible: bool,
    /// Quantization step for the lossy path (ignored when [`Self::reversible`] is
    /// `true`). Sourced from the optional `htj2kQuantStep` header attribute.
    qstep: f32,
}

/// Free-function form of [`Htj2kCompressor::channel_width`] — needed inside
/// `decode_chunk` once it writes directly into `self.out`:
/// `self.channel_width(..)` is a whole-`&self` method call, which cannot be
/// used while `self.out` is separately borrowed mutably, even though the two
/// fields are disjoint. Taking `min_x`/`max_x` as plain parameters (read from
/// `self.min_x`/`self.max_x` — direct field projections, not a method call —
/// before/around the `&mut self.out` borrow) sidesteps that.
fn channel_width_of(min_x: i32, max_x: i32, x_sampling: i32) -> usize {
    let d_min = divp(min_x, x_sampling);
    let d_max = divp(max_x, x_sampling);
    (d_max - d_min + 1) as usize
}

impl Htj2kCompressor {
    /// Create a new HTJ2K compressor — port of `HTCompressor::HTCompressor`.
    ///
    /// `max_scan_line_size` is the maximum uncompressed bytes in a single scan
    /// line (used to size the chunk buffer); `num_scan_lines` is 256 or 32.  The
    /// channel layout and data-window bounds are captured from `header`.
    ///
    /// # Errors
    /// Propagates [`crate::header::Header::channels`] / `data_window` errors.
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
        // C-08b: the RGB-heuristic name matching below operates on raw bytes
        // (matching C++ `make_channel_map`'s `strrchr`/`strcasecmp` over
        // `const char*`, which never checks encoding), so a non-UTF-8 channel
        // name is classified exactly like a UTF-8 one.
        let names: Vec<crate::attr::Name> = channel_list.iter().map(|(n, _)| n.clone()).collect();

        // Optional lossy mode: the `htj2kQuantStep` header attribute (an exr-rs
        // extension; OpenEXR's HTJ2K is reversible-only) requests an irreversible
        // 9/7 encode at that quantization step. Absent or <= 0 → lossless.
        let (reversible, qstep) = if header.has_htj2k_quant_step() {
            let q = *header.htj2k_quant_step()?;
            if q > 0.0 { (false, q) } else { (true, 0.0) }
        } else {
            (true, 0.0)
        };

        Ok(Self {
            num_scan_lines,
            channels,
            min_x: dw.min.x,
            max_x: dw.max.x,
            max_y: dw.max.y,
            names,
            out: Vec::new(),
            channel_map: Vec::new(),
            file_offsets: Vec::new(),
            seen: Vec::new(),
            header_buf: Vec::new(),
            reversible,
            qstep,
        })
    }

    /// Number of sampled scanlines in the chunk that starts at `block_min_y`
    /// (chunk `min_y`) — `min(block_min_y + numScanLines - 1, max_y) -
    /// block_min_y + 1`.  This equals the writer's `block_lines`, so the codec
    /// iterates exactly the lines the gather wrote (= `encode->chunk.height`).
    fn block_height(&self, block_min_y: i32) -> i32 {
        let block_max_y = (block_min_y + self.num_scan_lines - 1).min(self.max_y);
        block_max_y - block_min_y + 1
    }

    /// Per-channel sample count for a channel with horizontal sub-sampling
    /// `x_sampling`, over the data window's x range — `divp(max_x, xs) -
    /// divp(min_x, xs) + 1` (matches `encode->channels[c].width`).
    fn channel_width(&self, x_sampling: i32) -> usize {
        channel_width_of(self.min_x, self.max_x, x_sampling)
    }

    /// Bytes per sample for a pixel type (HALF=2, FLOAT/UINT=4) — delegates
    /// to the canonical `crate::misc::pixel_type_size`.
    #[inline]
    fn bytes_per_sample(t: PixelType) -> Result<usize> {
        crate::misc::pixel_type_size(t)
    }

    /// Image width = data-window width (`encode->chunk.width`).
    fn image_width(&self) -> usize {
        (divp(self.max_x, 1) - divp(self.min_x, 1) + 1) as usize
    }
}

// ---------------------------------------------------------------------------
// make_channel_map — port of make_channel_map (internal_ht_common.cpp:35-176)
// ---------------------------------------------------------------------------

/// Refill the codestream-component → file-channel map and return the RGB flag
/// and total row bytes. Storage is reused; geometry is rebuilt for every chunk.
///
/// 1:1 with `make_channel_map`: it scans for an `r/g/b` (or `red/green/blue`)
/// suffix triplet sharing a prefix (case-insensitive, order-independent), and if
/// found with matching type+sampling, maps those file channels to codestream
/// components 0/1/2 with all other channels following in file order; otherwise
/// the map is identity.  `raster_line_offset` is the running byte offset of each
/// file channel within one gathered scanline (file order).
fn make_channel_map(
    channels: &[ChannelInfo],
    names: &[crate::attr::Name],
    min_x: i32,
    max_x: i32,
    map: &mut Vec<CodestreamChannelInfo>,
) -> Result<(bool, usize)> {
    let count = channels.len();

    /// One candidate RGB-suffix triplet being matched.
    struct Candidate<'a> {
        r_suffix: &'static [u8],
        g_suffix: &'static [u8],
        b_suffix: &'static [u8],
        r_index: Option<usize>,
        g_index: Option<usize>,
        b_index: Option<usize>,
        /// The layer prefix (text before the last `.`); recorded on first match.
        prefix: Option<&'a [u8]>,
    }

    let mut cands = [
        Candidate {
            r_suffix: b"r",
            g_suffix: b"g",
            b_suffix: b"b",
            r_index: None,
            g_index: None,
            b_index: None,
            prefix: None,
        },
        Candidate {
            r_suffix: b"red",
            g_suffix: b"green",
            b_suffix: b"blue",
            r_index: None,
            g_index: None,
            b_index: None,
            prefix: None,
        },
    ];

    for (i, name) in names.iter().enumerate().take(count) {
        let name = name.as_bytes();
        // strrchr('.'): prefix = text before last '.', suffix = text after it
        // (or the whole name when there is no '.'). C-08b: raw bytes, matching
        // C++ (never checks encoding).
        let (prefix, suffix) = match name.iter().rposition(|&b| b == b'.') {
            Some(d) => (&name[..d], &name[d + 1..]),
            None => (&name[..0], name),
        };
        for c in cands.iter_mut() {
            if let Some(lp) = c.prefix
                && lp != prefix
            {
                continue;
            }
            let matched = if suffix.eq_ignore_ascii_case(c.r_suffix) && c.r_index.is_none() {
                c.r_index = Some(i);
                true
            } else if suffix.eq_ignore_ascii_case(c.g_suffix) && c.g_index.is_none() {
                c.g_index = Some(i);
                true
            } else if suffix.eq_ignore_ascii_case(c.b_suffix) && c.b_index.is_none() {
                c.b_index = Some(i);
                true
            } else {
                false
            };
            if matched {
                if c.prefix.is_none() {
                    c.prefix = Some(prefix);
                }
                break;
            }
        }
    }

    // A triplet is usable only if all three matched AND share type + sampling.
    let mut rgb: Option<(usize, usize, usize)> = None;
    for c in cands.iter() {
        if let (Some(ri), Some(gi), Some(bi)) = (c.r_index, c.g_index, c.b_index) {
            let (rc, gc, bc) = (&channels[ri], &channels[gi], &channels[bi]);
            if rc.type_ == gc.type_
                && rc.type_ == bc.type_
                && rc.x_sampling == gc.x_sampling
                && rc.x_sampling == bc.x_sampling
                && rc.y_sampling == gc.y_sampling
                && rc.y_sampling == bc.y_sampling
            {
                rgb = Some((ri, gi, bi));
                break;
            }
        }
    }

    map.resize(count, CodestreamChannelInfo::default());
    let mut avail = 3usize;
    let mut offset = 0usize;
    for (file_i, channel) in channels.iter().enumerate() {
        let cs_i = match rgb {
            Some((ri, _, _)) if file_i == ri => 0,
            Some((_, gi, _)) if file_i == gi => 1,
            Some((_, _, bi)) if file_i == bi => 2,
            Some(_) => {
                let next = avail;
                avail += 1;
                next
            }
            None => file_i,
        };
        map[cs_i] = CodestreamChannelInfo {
            file_index: file_i,
            raster_line_offset: offset,
        };
        offset = offset
            .checked_add(channel_byte_width(channel, min_x, max_x)?)
            .ok_or_else(|| ExrError::Overflow("HTJ2K channel layout too large".into()))?;
    }
    Ok((rgb.is_some(), offset))
}

/// One file channel's byte width within a scanline: `width * bytes_per_element`,
/// where `width` is the data-window-relative sample count
/// (`divp(max_x, xs) - divp(min_x, xs) + 1`, matching
/// [`Htj2kCompressor::channel_width`]).
///
/// This used to take a plain `image_width` and compute
/// `divp(image_width, xs)`, i.e. it implicitly assumed `min_x == 0`. For a
/// subsampled channel (`xs > 1`) with an odd/unaligned `min_x`, that undercounts
/// or overcounts the true sample count by up to one relative to
/// `channel_width`, which every per-line loop in this module already uses
/// correctly. The mismatch only matters where `channel_byte_width` sizes a
/// buffer (`bpl`/`uncompressed_size` in the planar path) rather than just
/// walking `xs == 1` channels (the non-planar path, where `divp(_, 1)` cannot
/// diverge, so that path was never affected).
#[inline]
fn channel_byte_width(ch: &ChannelInfo, min_x: i32, max_x: i32) -> Result<usize> {
    let w = (divp(max_x, ch.x_sampling) - divp(min_x, ch.x_sampling) + 1) as usize;
    w.checked_mul(Htj2kCompressor::bytes_per_sample(ch.type_)?)
        .ok_or_else(|| ExrError::Overflow("HTJ2K channel row too large".into()))
}

// ---------------------------------------------------------------------------
// Header read / write — port of write_header / read_header
// (internal_ht_common.cpp:265-310)
// ---------------------------------------------------------------------------

/// Write the HTJ2K chunk header (MAGIC, PLEN, NCH, per-component file index) to
/// the front of `buf` — port of `write_header`.  Returns nothing; the bytes are
/// appended to `buf`, which the caller follows with the codestream.
fn write_ht_header(buf: &mut Vec<u8>, map: &[CodestreamChannelInfo]) {
    // payload = NCH(u16) + NCH * file_index(u16); PLEN = payload length.
    let plen = (2 + 2 * map.len()) as u32;
    buf.extend_from_slice(&HEADER_MARKER.to_be_bytes());
    buf.extend_from_slice(&plen.to_be_bytes());
    buf.extend_from_slice(&(map.len() as u16).to_be_bytes());
    for e in map {
        buf.extend_from_slice(&(e.file_index as u16).to_be_bytes());
    }
}

/// Read the HTJ2K chunk header into reused map storage — port of `read_header`.
/// Every entry is overwritten; `raster_line_offset` is filled in later. Returns
/// the header length so the caller can slice the codestream that follows.
///
/// Validates the magic number, payload length, and channel-map bounds; corrupt
/// input yields [`ExrError`] rather than a panic.
fn read_ht_header(data: &[u8], map: &mut Vec<CodestreamChannelInfo>, expected_channels: usize) -> Result<usize> {
    if data.len() < HEADER_FRAME_SZ {
        return Err(ExrError::Input("HTJ2K header too short".into()));
    }
    if u16::from_be_bytes([data[0], data[1]]) != HEADER_MARKER {
        return Err(ExrError::Input(
            "HTJ2K chunk header does not start with the magic number".into(),
        ));
    }
    let plen = u32::from_be_bytes([data[2], data[3], data[4], data[5]]) as usize;
    if plen < 2 {
        return Err(ExrError::Input(
            "HTJ2K header: channel-map payload too small".into(),
        ));
    }
    let total = HEADER_FRAME_SZ + plen;
    if data.len() < total {
        return Err(ExrError::Input("HTJ2K header: truncated".into()));
    }
    let nch = u16::from_be_bytes([data[6], data[7]]) as usize;
    if plen < 2 + 2 * nch {
        return Err(ExrError::Input("HTJ2K header: channel map truncated".into()));
    }
    // Reject inconsistent counts before growing persistent metadata storage.
    if nch != expected_channels {
        return Err(ExrError::Input("HTJ2K: unexpected number of channels in chunk header".into()));
    }
    map.resize(nch, CodestreamChannelInfo::default());
    for (i, e) in map.iter_mut().enumerate() {
        let o = 8 + i * 2;
        *e = CodestreamChannelInfo {
            file_index: u16::from_be_bytes([data[o], data[o + 1]]) as usize,
            raster_line_offset: 0,
        };
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Pixel byte <-> LineBuf i32 helpers
// ---------------------------------------------------------------------------

/// Write little-endian Xdr pixel bytes into a `LineBuf` i32 slice.
///
/// HALF: sign-extend the 16-bit value to 32 bits (C++ does `cur_line->i32[p] =
/// *channel_pixels++` from an `int16_t*`).  FLOAT/UINT: bit-cast the 32-bit word.
fn bytes_to_linebuf(dst: &mut [i32], src: &[u8], t: PixelType) -> Result<()> {
    match t {
        PixelType::Half => {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                *d = i16::from_le_bytes([c[0], c[1]]) as i32;
            }
        }
        PixelType::Float | PixelType::Uint => {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<4>().0.iter()) {
                *d = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        PixelType::NumPixelTypes => {
            return Err(ExrError::Arg("HTJ2K: unknown pixel type".into()));
        }
    }
    Ok(())
}

/// Write `LineBuf` i32 values back to little-endian Xdr pixel bytes.
///
/// HALF: truncate to the low 16 bits (C++ stores `cur_line->i32[p]` through an
/// `int16_t*`).  FLOAT/UINT: store all 32 bits.
fn linebuf_to_bytes(dst: &mut [u8], src: &[i32], t: PixelType) -> Result<()> {
    match t {
        PixelType::Half => {
            for (c, &s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                c.copy_from_slice(&(s as i16).to_le_bytes());
            }
        }
        PixelType::Float | PixelType::Uint => {
            for (c, &s) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                c.copy_from_slice(&s.to_le_bytes());
            }
        }
        PixelType::NumPixelTypes => {
            return Err(ExrError::Arg("HTJ2K: unknown pixel type".into()));
        }
    }
    Ok(())
}

impl Compressor for Htj2kCompressor {
    /// 256 (`HTJ2K256`) or 32 (`HTJ2K32`) scan lines per chunk.
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// HTJ2K uses the machine-independent Xdr format (it does not override the
    /// C++ `Compressor::format()` default).
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Compress one chunk — port of `ht_apply_impl` (`internal_ht.cpp:351-526`).
    ///
    /// Builds a per-chunk HTJ2K codestream (`image_height = block_height`,
    /// `image_width = data-window width`), prefixes the channel-map header, and
    /// applies the "no-shrink → store raw" rule.
    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }

        // OpenJPH (jph-rs) reports unrecoverable errors as `Err(OjphError)` (jph-rs
        // converted its whole port from panicking to `Result`,
        // so no `catch_unwind` boundary is needed here any more). C++
        // `internal_exr_apply_ht` wraps the encode in `try { … } catch(...)` and
        // returns `EXR_ERR_INCORRECT_CHUNK`; `encode_chunk` mirrors that by mapping
        // every fallible jph-rs call into `ExrError` and propagating with `?` — this
        // is the codec's error boundary, not an `unwrap`/`panic!` in our logic.
        // encode_chunk fills self.out directly (see its own
        // doc) instead of returning a fresh Vec for this to move-assign.
        self.encode_chunk(data, min_y)?;
        Ok(&self.out)
    }

    /// Uncompress one chunk — port of `ht_undo_impl` (`internal_ht.cpp:160-325`).
    ///
    /// Reads the channel-map header, decodes the codestream, and scatters each
    /// component's reconstructed line back to little-endian HALF/FLOAT/UINT bytes.
    ///
    /// OpenJPH (jph-rs) reports corrupt-stream errors as `Err(OjphError)` (jph-rs
    /// no longer panics); `internal_exr_undo_ht` catches `(...)` and returns
    /// `EXR_ERR_CORRUPT_CHUNK`. `decode_chunk` mirrors that by mapping every
    /// fallible jph-rs call into `ExrError` and propagating with `?` — corrupt
    /// input never panics out of this codec, and no `catch_unwind` is needed any
    /// more to guarantee that.
    ///
    /// `expected_len` is unused: `decode_chunk`'s size comes
    /// entirely from `image_width()`/`block_height()` (the caller-supplied
    /// `min_y` plus the rectangle fields), never a construction-time bound.
    fn uncompress(&mut self, data: &[u8], min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }
        // decode_chunk fills self.out directly (same "one
        // out/scratch policy" every other codec's uncompress() follows)
        // instead of returning a fresh Vec — no per-chunk allocation of the
        // whole decoded buffer, and no per-chunk zero-fill either.
        self.decode_chunk(data, min_y)?;
        Ok(&self.out)
    }

    /// Compress one tile, encoded against the tile's own pixel rectangle.
    ///
    /// HTJ2K sizes its chunk from `image_width()` (derived from `min_x`/`max_x`)
    /// and `block_height()` (clamped by `max_y`).  In `internal_ht.cpp` these are
    /// `encode->chunk.width` / `encode->chunk.height` — the TILE's dimensions,
    /// not the full data window's.  For a tile we must therefore set all three
    /// rectangle fields from `range`: `min_x`/`max_x` so `image_width()` yields
    /// the tile width, and `max_y` so `block_height(range.min.y)` yields the
    /// tile height (a partial right/bottom edge tile encodes exactly its own
    /// pixels — `min(min_y + num_scan_lines - 1, max_y) - min_y + 1`).  Then we
    /// run the identical encode path at `range.min.y`.  This mirrors C++
    /// `compressTile`, which feeds the tile's `Box2i range` into the encode step
    /// (ImfCompressor.cpp:118).  The compressor is dedicated to tiles in the
    /// tiled writer, so each call overwrites the fields fresh (no restore).
    fn compress_tile(&mut self, data: &[u8], range: Box2i) -> Result<&[u8]> {
        self.min_x = range.min.x;
        self.max_x = range.max.x;
        self.max_y = range.max.y;
        self.compress(data, range.min.y)
    }

    /// Uncompress one tile, decoded against the tile's own pixel rectangle —
    /// the exact inverse of [`compress_tile`](Self::compress_tile).  Sets the
    /// rectangle fields from `range` so `image_width()` / `block_height()` match
    /// `decode->chunk.width` / `decode->chunk.height` (`internal_ht.cpp:227`),
    /// then runs the identical decode path at `range.min.y`.
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

impl Htj2kCompressor {
    /// Encode one chunk into `self.out` as the framed `header + codestream`
    /// bytes, or the raw bytes when the "no-shrink → store raw" rule fires.
    /// All OpenJPH interaction is confined here — every fallible jph-rs call
    /// is mapped into `ExrError` via `map_e` and propagated with `?` (jph-rs
    /// `ojph_error!` no longer panics, so this is an
    /// ordinary `Result` chain, not a `catch_unwind` boundary).
    ///
    /// Takes `&mut self` and fills `self.out` directly
    /// instead of returning a freshly-allocated `Vec<u8>` — the same
    /// out/scratch policy every other codec's compress AND uncompress path
    /// follows. Unlike `decode_chunk`, no borrow-conflict workaround is
    /// needed: every `self.channel_width(..)` call happens BEFORE `self.out`
    /// is touched at all (only the final assembly at the bottom writes to
    /// it), so the whole-`&self` method calls and the later `&mut self.out`
    /// write never overlap.
    fn encode_chunk(&mut self, data: &[u8], min_y: i32) -> Result<()> {
        let map_e = |e: jph_rs::openjph::ojph_message::OjphError| {
            ExrError::Input(format!("HTJ2K encode: {e}"))
        };
        let channel_count = self.channels.len();
        let image_width = self.image_width();
        let image_height = self.block_height(min_y) as usize;

        // Rebuild geometry in reused storage; layer prefixes borrow captured names.
        if self.channel_map.capacity() >= channel_count
            && self.header_buf.capacity() >= HEADER_FRAME_SZ + 2 + 2 * channel_count
        {
            crate::probe::hit(crate::probe::Probe::Htj2kMetadataReuse);
        }
        let (is_rgb, bpl) = make_channel_map(
            &self.channels,
            &self.names,
            self.min_x,
            self.max_x,
            &mut self.channel_map,
        )?;
        let map = &self.channel_map;

        let is_planar = self
            .channels
            .iter()
            .any(|ch| ch.x_sampling > 1 || ch.y_sampling > 1);

        let mut cs = Codestream::new();
        {
            let siz = cs.access_siz();
            siz.set_num_components(channel_count as Ui32);
        }

        // cs_c is the codestream component ordinal passed to set_comp_info /
        // set_nonlinear_transform — it is not merely an index into `map`.
        #[allow(clippy::needless_range_loop)]
        for cs_c in 0..channel_count {
            let file_c = map[cs_c].file_index;
            let ch = &self.channels[file_c];
            let bit_depth: Ui32 = if ch.type_ == PixelType::Half { 16 } else { 32 };
            {
                let siz = cs.access_siz();
                siz.set_comp_info(
                    cs_c as Ui32,
                    Point::new(ch.x_sampling as Ui32, ch.y_sampling as Ui32),
                    bit_depth,
                    ch.type_ != PixelType::Uint,
                );
            }
            if ch.type_ != PixelType::Uint {
                cs.access_nlt_mut()
                    .set_nonlinear_transform(cs_c as Ui32, NLT_BINARY_COMPLEMENT)
                    .map_err(map_e)?;
            }
        }

        cs.set_planar(is_planar);
        {
            let siz = cs.access_siz();
            siz.set_image_offset(Point::new(0, 0));
            siz.set_image_extent(Point::new(image_width as Ui32, image_height as Ui32));
        }
        {
            let cod = cs.access_cod();
            cod.employ_color_transform(if is_rgb && !is_planar { 1 } else { 0 });
            cod.set_reversible(self.reversible);
            cod.set_block_dims(128, 32).map_err(map_e)?;
            cod.set_num_decompositions(5).map_err(map_e)?;
        }
        // Lossy path: install the quantization step (irreversible 9/7). No-op for
        // the lossless reversible default. Mirrors jph-rs `mem.rs` encode order.
        if !self.reversible {
            cs.access_qcd_mut().set_delta(self.qstep);
        }

        // Header bytes (MAGIC + PLEN + channel map) precede the codestream.
        self.header_buf.clear();
        self.header_buf.reserve(HEADER_FRAME_SZ + 2 + 2 * channel_count);
        write_ht_header(&mut self.header_buf, map);

        // OpenJPH writes into a memory file; cap it at the uncompressed size so
        // that, like the C++ `staticmem_outfile`, an over-large codestream is
        // detectable.  We use a growable MemOutfile and check the size after.
        let mut outfile = MemOutfile::default();
        outfile.open(data.len().max(1), false).map_err(map_e)?;
        cs.write_headers_slice(&mut outfile, &[]).map_err(map_e)?;

        // Bootstrap: get the first LineBuf from the codec.
        let mut next_comp: Ui32 = 0;
        let mut cur_line = cs
            .exchange(std::ptr::null_mut(), &mut next_comp)
            .map_err(map_e)?;

        if is_planar {
            // Planar path (internal_ht.cpp:423-475): each component fully, then
            // the next; the inner loop walks chunk-relative scanlines and feeds
            // only the lines that belong to this component's y-sampling.
            for m in map {
                let file_c = m.file_index;
                let ch = self.channels[file_c];
                if image_height == 0 {
                    continue;
                }
                let w = self.channel_width(ch.x_sampling);
                let bpe = Self::bytes_per_sample(ch.type_)?;

                let mut line_byte_off = 0usize;
                for y in 0..(image_height as i64) {
                    // C++ uses absolute y (chunk.start_y + .. ) only for the
                    // y_samples test; min_y is the chunk start.
                    let cury = min_y as i64 + y;
                    for line_c in 0..channel_count {
                        let lch = self.channels[line_c];
                        if cury % (lch.y_sampling as i64) != 0 {
                            continue;
                        }
                        if line_c == file_c {
                            let src_start = line_byte_off;
                            let src_end = src_start + w * bpe;
                            if src_end > data.len() {
                                return Err(ExrError::Logic(
                                    "HTJ2K compress: planar channel overruns the line buffer"
                                        .into(),
                                ));
                            }
                            let src_slice = &data[src_start..src_end];
                            let t = ch.type_;
                            let mut feed_err = None;
                            cs.fill_and_exchange(&mut cur_line, &mut next_comp, |lb, _| {
                                // `LineBuf::get_i32_mut` asserts `w <= lb.size` — a bare contract,
                                // not an `OjphResult` error (jph-rs made
                                // `ojph_error!` fallible, but this bounds check is a caller
                                // invariant, same as slice indexing). Check it ourselves so a
                                // malformed header (e.g. sub-sampling misaligned with `min_x`,
                                // rejected by `Header::sanity_check` but reachable if a caller
                                // bypasses it) errors cleanly instead of panicking.
                                if w > lb.size {
                                    feed_err = Some(ExrError::Logic(format!(
                                        "HTJ2K compress: planar channel width {w} exceeds the codestream line buffer size {}",
                                        lb.size
                                    )));
                                    return;
                                }
                                if let Err(e) = bytes_to_linebuf(lb.get_i32_mut(w), src_slice, t) {
                                    feed_err = Some(e);
                                }
                            })
                            .map_err(map_e)?;
                            if let Some(e) = feed_err {
                                return Err(e);
                            }
                        }
                        let lw = self.channel_width(lch.x_sampling);
                        line_byte_off += lw * Self::bytes_per_sample(lch.type_)?;
                    }
                }
            }
        } else {
            // Interleaved path (internal_ht.cpp:477-515): line-by-line, feeding
            // each codestream component from its file channel's slice.
            if bpl * image_height != data.len() {
                return Err(ExrError::Logic(format!(
                    "HTJ2K compress: bpl*height ({}*{}) != chunk size {}",
                    bpl,
                    image_height,
                    data.len()
                )));
            }
            for y in 0..image_height {
                let line_start = y * bpl;
                for m in map {
                    let file_c = m.file_index;
                    let ch = self.channels[file_c];
                    let w = self.channel_width(ch.x_sampling);
                    let bpe = Self::bytes_per_sample(ch.type_)?;
                    let src_off = line_start + m.raster_line_offset;
                    let src_end = src_off + w * bpe;
                    if src_end > data.len() {
                        return Err(ExrError::Logic(
                            "HTJ2K compress: channel overruns the line buffer".into(),
                        ));
                    }
                    let src_slice = &data[src_off..src_end];
                    let t = ch.type_;
                    let mut feed_err = None;
                    cs.fill_and_exchange(&mut cur_line, &mut next_comp, |lb, _| {
                        // See the planar-path comment above: bound `w` against the codestream's
                        // own line buffer before calling the panicking `get_i32_mut`.
                        if w > lb.size {
                            feed_err = Some(ExrError::Logic(format!(
                                "HTJ2K compress: channel width {w} exceeds the codestream line buffer size {}",
                                lb.size
                            )));
                            return;
                        }
                        if let Err(e) = bytes_to_linebuf(lb.get_i32_mut(w), src_slice, t) {
                            feed_err = Some(e);
                        }
                    })
                    .map_err(map_e)?;
                    if let Some(e) = feed_err {
                        return Err(e);
                    }
                }
            }
        }

        cs.flush().map_err(map_e)?;

        // `outfile` owns the encoded codestream bytes; borrow them instead of
        // `to_vec()`-ing into a throwaway copy before the real assembly below.
        let codestream = outfile.get_data();
        let total = self.header_buf.len() + codestream.len();

        // "no-shrink → store raw": if the framed codestream is not strictly
        // smaller than the uncompressed input, store the raw bytes (C++ catches
        // range_error and sets compressed_bytes = packed_bytes).
        if total >= data.len() {
            self.out.clear();
            self.out.extend_from_slice(data);
            return Ok(());
        }

        self.out.clear();
        self.out.extend_from_slice(&self.header_buf);
        self.out.extend_from_slice(codestream);
        Ok(())
    }

    /// Decode one framed chunk into `self.out`.  All OpenJPH interaction is
    /// confined here — every fallible jph-rs call is mapped into `ExrError`
    /// via `map_e` and propagated with `?` (see [`Self::encode_chunk`]).
    ///
    /// Fills `self.out` in place (sized with `ensure_len`, NOT
    /// zero-filled: every byte is overwritten, see the note below) instead
    /// of returning a freshly-allocated `Vec<u8>` — the same out/scratch
    /// policy every other codec's decode path follows. Takes
    /// `&mut self` for that reason; the in-loop channel-width lookups use the
    /// free function [`channel_width_of`] with `self.min_x`/`self.max_x` read
    /// as plain fields (not via the `&self` method `channel_width`), since a
    /// method call on the whole struct would conflict with the live
    /// `&mut self.out` borrow even though the fields are disjoint.
    fn decode_chunk(&mut self, data: &[u8], min_y: i32) -> Result<()> {
        let map_e = |e: jph_rs::openjph::ojph_message::OjphError| {
            ExrError::Input(format!("HTJ2K decode: {e}"))
        };
        let channel_count = self.channels.len();
        let image_width = self.image_width();
        let image_height = self.block_height(min_y) as usize;

        // Recompute file-order offsets from the CURRENT x rectangle.
        // Compute row and total packed sizes in the same pass, without temporary
        // collections. Y-subsampled channels contribute only their active rows,
        // matching compute_chunk_unpack_size (chunk.c:677-700), not bpl * height.
        let metadata_reused = self.channel_map.capacity() >= channel_count
            && self.file_offsets.capacity() >= channel_count
            && self.seen.capacity() >= channel_count;
        self.file_offsets.resize(channel_count, 0);
        let mut bpl = 0usize;
        let mut uncompressed_size = 0usize;
        for (i, channel) in self.channels.iter().enumerate() {
            self.file_offsets[i] = bpl;
            let row_bytes = channel_byte_width(channel, self.min_x, self.max_x)?;
            bpl = bpl
                .checked_add(row_bytes)
                .ok_or_else(|| ExrError::Overflow("HTJ2K channel layout too large".into()))?;
            let rows = active_row_count(min_y, min_y + image_height as i32 - 1, channel.y_sampling);
            let channel_bytes = row_bytes
                .checked_mul(rows)
                .ok_or_else(|| ExrError::Overflow("HTJ2K chunk size too large".into()))?;
            uncompressed_size = uncompressed_size
                .checked_add(channel_bytes)
                .ok_or_else(|| ExrError::Overflow("HTJ2K chunk size too large".into()))?;
        }

        let header_len = read_ht_header(data, &mut self.channel_map, channel_count)?;
        // Fill raster_line_offset from file-order channel widths and bounds-check
        // each file index (ht_undo_impl:180-191).
        //
        // Reject a duplicate `file_index` in the channel map, matching
        // `read_header` (`internal_ht_common.cpp:308-317`,
        // <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_ht_common.cpp#L308-L317>),
        // which throws (-> `EXR_ERR_CORRUPT_CHUNK`) on `file_index >= NCH` or on
        // a `file_index` already claimed by an earlier entry. The old code only
        // bounds-checked, so a crafted map `[0, 0, ...]` decoded with two
        // codestream components writing the same channel and another left zero.
        {
            self.seen.resize(channel_count, false);
            self.seen.fill(false);
            for e in &mut self.channel_map {
                if e.file_index >= channel_count {
                    return Err(ExrError::Input(
                        "HTJ2K: channel-map file index out of range".into(),
                    ));
                }
                if self.seen[e.file_index] {
                    return Err(ExrError::Input(
                        "HTJ2K: channel-map has a duplicate file index (corrupt chunk)".into(),
                    ));
                }
                self.seen[e.file_index] = true;
                e.raster_line_offset = self.file_offsets[e.file_index];
            }
        }

        if metadata_reused {
            crate::probe::hit(crate::probe::Probe::Htj2kMetadataReuse);
        }
        let map = &self.channel_map;
        let j2c = &data[header_len..];
        let mut infile = MemInfile::default();
        infile.open(j2c);

        let mut cs = Codestream::new();
        cs.read_headers(&mut infile).map_err(map_e)?;

        // Validate the codestream dimensions against the chunk (ht_undo_impl:
        // 218-220 asserts) — return an error rather than panic on a corrupt file.
        {
            let siz = cs.access_siz();
            let cw = (siz.get_image_extent().x - siz.get_image_offset().x) as usize;
            let ch = (siz.get_image_extent().y - siz.get_image_offset().y) as usize;
            let cc = siz.get_num_components() as usize;
            if cw != image_width || ch != image_height || cc != channel_count {
                infile.close_infile();
                return Err(ExrError::Input(
                    "HTJ2K: codestream dimensions do not match the chunk".into(),
                ));
            }
        }

        // Per-component geometry cross-check — `internal_ht.cpp:232-241`,
        // <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_ht.cpp#L232-L241>.
        // The old code only checked the overall width/height/component count
        // above; a codestream whose PER-COMPONENT downsampling or recon size
        // disagreed with our channel map decoded anyway (garbage or a
        // truncated line via `get_i32(w)` using OUR width, not the
        // codestream's). C rejects the chunk unless, for every component
        // `cs_i` mapped to file channel `file_i`:
        //   channels[file_i].height == siz.get_recon_height(cs_i)
        //   channels[file_i].width  == siz.get_recon_width(cs_i)
        //   channels[file_i].height == image_height / siz.get_downsampling(cs_i).y
        //   channels[file_i].width  == image_width  / siz.get_downsampling(cs_i).x
        {
            let siz = cs.access_siz();
            for (cs_i, e) in map.iter().enumerate() {
                let ech = self.channels[e.file_index];
                let w = self.channel_width(ech.x_sampling) as Ui32;
                // Per-chunk sampled height for this channel, matching C's
                // `image_height / downsampling.y` form (integer division; a
                // channel whose sampling does not evenly divide the chunk
                // height cannot match and is correctly rejected below).
                let y_sampling = ech.y_sampling.max(1) as Ui32;
                let h = image_height as Ui32 / y_sampling;
                let recon_w = siz.get_recon_width(cs_i as Ui32);
                let recon_h = siz.get_recon_height(cs_i as Ui32);
                let downsampling = siz.get_downsampling(cs_i as Ui32);
                if h != recon_h
                    || w != recon_w
                    || downsampling.y == 0
                    || downsampling.x == 0
                    || h != (image_height as Ui32 / downsampling.y)
                    || w != (image_width as Ui32 / downsampling.x)
                {
                    infile.close_infile();
                    return Err(ExrError::Input(
                        "HTJ2K: per-component geometry does not match the codestream (corrupt chunk)"
                            .into(),
                    ));
                }
            }
        }

        let is_planar = self
            .channels
            .iter()
            .any(|c| c.x_sampling > 1 || c.y_sampling > 1);
        cs.set_planar(is_planar);
        cs.create().map_err(map_e)?;

        // `self.out` is fully covered on every SUCCESS
        // path regardless of prior contents: the planar branch's per-channel
        // loop writes every byte its OWN y_sampling includes across every
        // row (and every error path returns `Err` before `self.out` would
        // ever be read as a result); the non-planar branch only runs when
        // NO channel is subsampled, so it writes every (y, channel) pair
        // unconditionally. Verified empirically, not just by tracing --
        // `poison_reused_out_matches_fresh_instance_on_subsampled_planar_decode`
        // poisons `self.out` before the first call on the harder (planar)
        // path and diffs against a fresh instance. `ensure_len` (grow/
        // truncate, no unconditional zero-fill) is therefore sound, unlike
        // DWA's UNKNOWN sub-stream, whose deterministic-zero tail on a short
        // decode genuinely IS load-bearing
        // -- not the same shape, checked separately, not assumed.
        // A dedicated test re-proves it for every chunk shape (full, partial
        // edge, tile, multi-channel, all sample types, subsampled):
        // `decode_overwrites_every_byte_for_every_chunk_shape`.
        // `ensure_len`'s grow/truncate, inlined so the probe sits on the branch
        // that keeps stale bytes (a zero-filling mutation removes it).
        if self.out.len() >= uncompressed_size {
            self.out.truncate(uncompressed_size);
            crate::probe::hit(crate::probe::Probe::Htj2kOutKept);
        } else {
            self.out.resize(uncompressed_size, 0);
        }
        let min_x = self.min_x;
        let max_x = self.max_x;
        let mut next_comp: Ui32 = 0;

        if cs.is_planar() {
            // cs_c is compared against next_comp to validate codestream component
            // order — it cannot be removed by iterating map directly.
            #[allow(clippy::needless_range_loop)]
            for cs_c in 0..channel_count {
                let file_c = map[cs_c].file_index;
                let ech = self.channels[file_c];
                if image_height == 0 {
                    continue;
                }
                let w = channel_width_of(min_x, max_x, ech.x_sampling);
                let bpe = Self::bytes_per_sample(ech.type_)?;

                let mut line_byte_off = 0usize;
                for y in 0..(image_height as i64) {
                    let cury = min_y as i64 + y;
                    for line_c in 0..channel_count {
                        let lch = self.channels[line_c];
                        if cury % (lch.y_sampling as i64) != 0 {
                            continue;
                        }
                        if line_c == file_c {
                            // pull_ref: valid only until the next codec call. Write
                            // straight from the borrowed line buffer into `out` —
                            // no intermediate `Vec<i32>` allocation per line.
                            let lb = match cs.pull_ref(&mut next_comp) {
                                Ok(lb) if w <= lb.size => lb,
                                Ok(lb) => {
                                    let size = lb.size;
                                    infile.close_infile();
                                    return Err(ExrError::Input(format!(
                                        "HTJ2K: planar channel width {w} exceeds the codestream line buffer size {size}"
                                    )));
                                }
                                Err(e) => {
                                    infile.close_infile();
                                    return Err(map_e(e));
                                }
                            };
                            if next_comp as usize != cs_c {
                                infile.close_infile();
                                return Err(ExrError::Input(
                                    "HTJ2K: codestream component order mismatch".into(),
                                ));
                            }
                            let dst_end = line_byte_off + w * bpe;
                            if dst_end > self.out.len() {
                                infile.close_infile();
                                return Err(ExrError::Input(
                                    "HTJ2K: planar channel overruns output".into(),
                                ));
                            }
                            linebuf_to_bytes(&mut self.out[line_byte_off..dst_end], lb.get_i32(w), ech.type_)?;
                        }
                        let lw = channel_width_of(min_x, max_x, lch.x_sampling);
                        line_byte_off += lw * Self::bytes_per_sample(lch.type_)?;
                    }
                }
            }
        } else {
            for y in 0..image_height {
                let line_start = y * bpl;
                // cs_c is compared against next_comp to validate codestream
                // component order — it cannot be removed by iterating map directly.
                #[allow(clippy::needless_range_loop)]
                for cs_c in 0..channel_count {
                    let file_c = map[cs_c].file_index;
                    let ech = self.channels[file_c];
                    let w = channel_width_of(min_x, max_x, ech.x_sampling);
                    let bpe = Self::bytes_per_sample(ech.type_)?;

                    // See the planar-path comment above: write straight from the
                    // borrowed line buffer, no per-line `Vec<i32>` allocation.
                    let lb = match cs.pull_ref(&mut next_comp) {
                        Ok(lb) if w <= lb.size => lb,
                        Ok(lb) => {
                            let size = lb.size;
                            infile.close_infile();
                            return Err(ExrError::Input(format!(
                                "HTJ2K: channel width {w} exceeds the codestream line buffer size {size}"
                            )));
                        }
                        Err(e) => {
                            infile.close_infile();
                            return Err(map_e(e));
                        }
                    };
                    if next_comp as usize != cs_c {
                        infile.close_infile();
                        return Err(ExrError::Input(
                            "HTJ2K: codestream component order mismatch".into(),
                        ));
                    }
                    let dst_off = line_start + map[cs_c].raster_line_offset;
                    let dst_end = dst_off + w * bpe;
                    if dst_end > self.out.len() {
                        infile.close_infile();
                        return Err(ExrError::Input(
                            "HTJ2K: channel overruns output".into(),
                        ));
                    }
                    linebuf_to_bytes(&mut self.out[dst_off..dst_end], lb.get_i32(w), ech.type_)?;
                }
            }
        }

        infile.close_infile();
        Ok(())
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

    /// Build a header with the given channels (sorted by name), `w`x`h`, HTJ2K256,
    /// INCREASING_Y, sub-sampling 1.
    fn make_header(w: i32, h: i32, chans: &[(&str, PixelType)]) -> Header {
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Htj2k256,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for (name, ty) in chans {
            cl.insert(*name, Channel::new(*ty, 1, 1, false)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr
    }

    /// Build an Xdr (little-endian) interleaved line buffer for `block_height`
    /// lines, channels in sorted order, with smooth deterministic values (smooth
    /// data keeps the lossy wavelet error small and predictable).
    fn build_line_buffer(w: i32, block_height: i32, chans: &[(&str, PixelType)]) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..block_height {
            for x in 0..w {
                for (ci, (_, ty)) in chans.iter().enumerate() {
                    let v = (ci as f32) * 8.0 + (y as f32) + (x as f32) * 0.5 + 1.0;
                    match ty {
                        PixelType::Half => {
                            buf.extend_from_slice(&Half::from(v).bits().to_le_bytes());
                        }
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => {
                            buf.extend_from_slice(&((v as u32).to_le_bytes()))
                        }
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// Decode a HALF sample (little-endian bits) to f32.
    fn half_at(buf: &[u8], off: usize) -> f32 {
        Half::from_bits(u16::from_le_bytes([buf[off], buf[off + 1]])).to_f32()
    }
    /// Decode a FLOAT sample (little-endian bits) to f32.
    fn float_at(buf: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    }

    /// Round-trip a chunk and return `(restored, was_raw_passthrough)`.
    ///
    /// HTJ2K has substantial per-codestream overhead, so on small/incompressible
    /// inputs the "no-shrink → store raw" rule (mirrored from `ht_apply_impl`)
    /// returns the bytes unchanged.  In that case the file would carry
    /// `dataSize == uncompressedSize`, and the scanline reader skips the codec —
    /// so the codec's own decode of a real codestream is only meaningful when the
    /// encoder actually produced one.  The helper distinguishes the two outcomes.
    fn round_trip(c: &mut Htj2kCompressor, line: &[u8]) -> (Vec<u8>, bool) {
        let compressed = c.compress(line, 0).unwrap().to_vec();
        if compressed.len() == line.len() {
            // Raw passthrough: identity, no header to decode.
            return (compressed, true);
        }
        let restored = c.uncompress(&compressed, 0, 0).unwrap().to_vec();
        (restored, false)
    }

    /// HTJ2K uses the reversible JPEG 2000 path, so a compress→uncompress round
    /// trip reproduces the input within a small lossy bound for non-RGB HALF data
    /// (or exactly, via the raw passthrough on incompressible chunks).  A large
    /// smooth gradient compresses, exercising the real codec path.
    #[test]
    fn round_trip_half_within_bound() {
        let (w, h) = (256i32, 64i32);
        // Non-RGB names so the RCT colour transform is NOT engaged.
        let chans = [("H1", PixelType::Half), ("H2", PixelType::Half)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        let (restored, raw) = round_trip(&mut c, &line);
        assert_eq!(restored.len(), line.len(), "round-trip size must match");
        if raw {
            assert_eq!(restored, line, "raw passthrough must be identity");
            return;
        }
        let stride = 4usize; // [H1 H2] per pixel
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = (y * w as usize + x) * stride;
                for ci in 0..2 {
                    let off = p + ci * 2;
                    let orig = half_at(&line, off);
                    let dec = half_at(&restored, off);
                    assert!(
                        (orig - dec).abs() <= 1.0,
                        "HTJ2K HALF round-trip error too large at ({x},{y}) ch{ci}: {orig} vs {dec}"
                    );
                }
            }
        }
    }

    /// RGB HALF triplet engages the reversible colour transform; the reversible
    /// path must still round-trip within a small bound (or pass through raw).
    #[test]
    fn round_trip_rgb_within_bound() {
        let (w, h) = (256i32, 32i32);
        let chans = [
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
        ];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 6, 32).unwrap();
        let (restored, raw) = round_trip(&mut c, &line);
        assert_eq!(restored.len(), line.len());
        if raw {
            assert_eq!(restored, line);
            return;
        }
        let stride = 6usize; // B,G,R each HALF
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = (y * w as usize + x) * stride;
                for ci in 0..3 {
                    let off = p + ci * 2;
                    let orig = half_at(&line, off);
                    let dec = half_at(&restored, off);
                    assert!(
                        (orig - dec).abs() <= 1.0,
                        "HTJ2K RGB round-trip error too large at ({x},{y}) ch{ci}: {orig} vs {dec}"
                    );
                }
            }
        }
    }

    /// FLOAT channels survive HTJ2K (32-bit reversible) within a relative bound
    /// (or pass through raw on incompressible chunks).
    #[test]
    fn round_trip_float_within_bound() {
        let (w, h) = (256i32, 64i32);
        let chans = [("Z", PixelType::Float)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        let (restored, raw) = round_trip(&mut c, &line);
        assert_eq!(restored.len(), line.len());
        if raw {
            assert_eq!(restored, line);
            return;
        }
        for i in 0..(w * h) as usize {
            let off = i * 4;
            let orig = float_at(&line, off);
            let dec = float_at(&restored, off);
            assert!(
                (orig - dec).abs() <= orig.abs() * 0.05 + 1.0,
                "HTJ2K FLOAT round-trip error too large at {i}: {orig} vs {dec}"
            );
        }
    }

    /// exr-rs extension: the optional `htj2kQuantStep` header attribute selects a
    /// lossy (irreversible 9/7) encode. The lossy stream must be no larger than the
    /// lossless default and must still decode to the original byte length.
    #[test]
    fn lossy_htj2k_shrinks_and_round_trips() {
        let (w, h) = (256i32, 64i32);
        let chans = [("Y", PixelType::Half)];
        let line = build_line_buffer(w, h, &chans);

        // Lossless (default) compressed size.
        let hdr_ll = make_header(w, h, &chans);
        let mut c_ll = Htj2kCompressor::new(&hdr_ll, w as usize * 2, 256).unwrap();
        let ll_len = c_ll.compress(&line, 0).unwrap().len();

        // Lossy via the extension attribute.
        let mut hdr_lossy = make_header(w, h, &chans);
        hdr_lossy.add_htj2k_quant_step(0.01).unwrap();
        let mut c_lossy = Htj2kCompressor::new(&hdr_lossy, w as usize * 2, 256).unwrap();
        let comp = c_lossy.compress(&line, 0).unwrap().to_vec();
        assert!(
            comp.len() <= ll_len,
            "lossy {} should be <= lossless {}",
            comp.len(),
            ll_len
        );

        // It must decode without error to the same byte length.
        let restored = c_lossy.uncompress(&comp, 0, 0).unwrap();
        assert_eq!(restored.len(), line.len());
    }

    /// Warm metadata must be overwritten, including after a corrupt wire map.
    /// RGB and identity codestream maps may alternate on the same decoder.
    #[test]
    fn metadata_reuse_resets_maps_flags_and_header_after_errors() {
        use crate::probe::{self, Probe};

        let (w, h) = (256, 32);
        let rgb = [
            ("main.B", PixelType::Half),
            ("main.G", PixelType::Half),
            ("main.R", PixelType::Half),
        ];
        let identity = [
            ("X", PixelType::Half),
            ("Y", PixelType::Half),
            ("Z", PixelType::Half),
        ];
        let header = make_header(w, h, &rgb);
        let data = build_line_buffer(w, h, &rgb);
        let mut codec = Htj2kCompressor::new(&header, w as usize * 6, 32).unwrap();
        let encoded = codec.compress(&data, 0).unwrap().to_vec();
        assert!(encoded.len() < data.len());
        assert_eq!(codec.uncompress(&encoded, 0, data.len()).unwrap(), data);

        let mut other =
            Htj2kCompressor::new(&make_header(w, h, &identity), w as usize * 6, 32).unwrap();
        let identity_encoded = other.compress(&data, 0).unwrap().to_vec();
        assert!(identity_encoded.len() < data.len());
        assert_ne!(&encoded[8..14], &identity_encoded[8..14]);
        let storage = (
            (codec.channel_map.as_ptr(), codec.channel_map.capacity()),
            (codec.file_offsets.as_ptr(), codec.file_offsets.capacity()),
            (codec.seen.as_ptr(), codec.seen.capacity()),
            (codec.header_buf.as_ptr(), codec.header_buf.capacity()),
        );

        probe::reset(Probe::Htj2kMetadataReuse);
        codec.header_buf.fill(0xA5);
        for entry in &mut codec.channel_map {
            entry.file_index = usize::MAX;
            entry.raster_line_offset = usize::MAX;
        }
        codec.file_offsets.fill(usize::MAX);
        codec.seen.fill(true);
        assert_eq!(codec.compress(&data, 0).unwrap(), encoded);
        assert_eq!(
            codec.uncompress(&identity_encoded, 0, data.len()).unwrap(),
            data
        );

        let mut corrupt = encoded.clone();
        corrupt[10..12].copy_from_slice(&encoded[8..10]);
        assert!(codec.uncompress(&corrupt, 0, data.len()).is_err());
        // A framed but inconsistent large channel count must not inflate the cache.
        let wrong_count = u16::MAX as usize;
        let payload_len = 2 + 2 * wrong_count;
        let mut oversized_map = vec![0; HEADER_FRAME_SZ + payload_len];
        oversized_map[..2].copy_from_slice(&HEADER_MARKER.to_be_bytes());
        oversized_map[2..6].copy_from_slice(&(payload_len as u32).to_be_bytes());
        oversized_map[6..8].copy_from_slice(&u16::MAX.to_be_bytes());
        assert!(codec.uncompress(&oversized_map, 0, data.len()).is_err());
        assert_eq!(codec.channel_map.capacity(), storage.0.1);
        assert_eq!(codec.uncompress(&encoded, 0, data.len()).unwrap(), data);
        assert_eq!(
            codec.uncompress(&identity_encoded, 0, data.len()).unwrap(),
            data
        );
        assert_eq!(
            storage,
            (
                (codec.channel_map.as_ptr(), codec.channel_map.capacity()),
                (codec.file_offsets.as_ptr(), codec.file_offsets.capacity()),
                (codec.seen.as_ptr(), codec.seen.capacity()),
                (codec.header_buf.as_ptr(), codec.header_buf.capacity()),
            )
        );
        assert_eq!(
            probe::get(Probe::Htj2kMetadataReuse),
            4,
            "encode and three successful decodes must reuse metadata"
        );
    }

    /// Reused metadata describes each codec rectangle, not the preceding chunk.
    #[test]
    fn metadata_reuse_rebuilds_offsets_for_changed_chunk_geometry() {
        let channels = [
            ("main.B", PixelType::Half),
            ("main.G", PixelType::Half),
            ("main.R", PixelType::Half),
        ];
        let header = make_header(256, 32, &channels);
        let large = Box2i::from_min_max(imath_rs::V2i::new(-17, 0), imath_rs::V2i::new(238, 31));
        let small = Box2i::from_min_max(imath_rs::V2i::new(17, -12), imath_rs::V2i::new(80, 3));
        let mut codec = Htj2kCompressor::new(&header, 256 * 6, 32).unwrap();
        for range in [large, small, large] {
            let width = range.max.x - range.min.x + 1;
            let height = range.max.y - range.min.y + 1;
            let data = build_line_buffer(width, height, &channels);
            let mut fresh = Htj2kCompressor::new(&header, 256 * 6, 32).unwrap();
            let expected = fresh.compress_tile(&data, range).unwrap().to_vec();
            let encoded = codec.compress_tile(&data, range).unwrap().to_vec();
            assert_eq!(encoded, expected);
            assert!(encoded.len() < data.len());
            codec.file_offsets.fill(usize::MAX);
            for entry in &mut codec.channel_map {
                entry.raster_line_offset = usize::MAX;
            }
            assert_eq!(
                codec.uncompress_tile(&encoded, range, data.len()).unwrap(),
                data
            );
            assert_eq!(
                codec.file_offsets,
                vec![0, width as usize * 2, width as usize * 4]
            );
        }
    }

    /// Corrupt / truncated input must yield an error, never a panic.
    #[test]
    fn corrupt_input_errors_not_panics() {
        let (w, h) = (8i32, 4i32);
        let chans = [("H1", PixelType::Half), ("H2", PixelType::Half)];
        let hdr = make_header(w, h, &chans);
        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();

        // Too short to even hold the 6-byte frame.
        assert!(c.uncompress(&[0u8; 3], 0, 0).is_err());
        // Bad magic number.
        assert!(c.uncompress(&[0xFF, 0xFF, 0, 0, 0, 2, 0, 2], 0, 0).is_err());
        // Valid magic + PLEN but garbage codestream body.
        let mut bad = Vec::new();
        write_ht_header(
            &mut bad,
            &[
                CodestreamChannelInfo { file_index: 0, raster_line_offset: 0 },
                CodestreamChannelInfo { file_index: 1, raster_line_offset: 0 },
            ],
        );
        bad.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33]);
        assert!(c.uncompress(&bad, 0, 0).is_err());
    }

    /// The factory wires HTJ2K256 (256 lines) and HTJ2K32 (32 lines) correctly.
    #[test]
    fn factory_scanline_counts() {
        let chans = [("R", PixelType::Half)];
        let hdr256 = {
            let mut h = make_header(4, 4, &chans);
            *h.compression_mut().unwrap() = Compression::Htj2k256;
            h
        };
        let c = Htj2kCompressor::new(&hdr256, 8, 256).unwrap();
        assert_eq!(c.num_scan_lines(), 256);
        assert_eq!(c.format(), Format::Xdr);
        let c = Htj2kCompressor::new(&hdr256, 8, 32).unwrap();
        assert_eq!(c.num_scan_lines(), 32);
    }

    /// A channel map with a duplicate `file_index` must be rejected as a
    /// corrupt chunk (`read_header`, `internal_ht_common.cpp:308-317`). The old
    /// code only range-checked each index, so `[0, 0]` decoded — two codestream
    /// components writing the same file channel and another left zero.
    #[test]
    fn htj2k_rejects_duplicate_file_index() {
        let (w, h) = (256i32, 64i32);
        let chans = [("H1", PixelType::Half), ("H2", PixelType::Half)];
        let hdr = make_header(w, h, &chans);
        let line = build_line_buffer(w, h, &chans);

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        let compressed = c.compress(&line, 0).unwrap().to_vec();
        assert!(
            compressed.len() < line.len(),
            "fixture must actually compress, not store raw"
        );

        // Header layout: MAGIC(2) PLEN(4) NCH(2) idx0(2) idx1(2).
        assert_eq!(
            u16::from_be_bytes([compressed[6], compressed[7]]),
            2,
            "nch must be 2 for this fixture"
        );
        let mut corrupt = compressed.clone();
        corrupt[10] = corrupt[8];
        corrupt[11] = corrupt[9]; // idx1 := idx0 -> duplicate file_index

        let mut c2 = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        assert!(
            c2.uncompress(&corrupt, 0, 0).is_err(),
            "duplicate file_index in the channel map must be CORRUPT_CHUNK"
        );
    }

    /// Swapping a channel-map entry between two file channels with
    /// different per-component geometry (here, different horizontal
    /// sub-sampling) must be rejected, matching `internal_ht.cpp:232-241`'s
    /// per-component `recon_width`/`recon_height`/`downsampling` cross-check.
    /// The old code only validated the overall width/height/component count.
    #[test]
    fn htj2k_rejects_swapped_component_geometry() {
        let (w, h) = (32i32, 16i32);
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Htj2k256,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        // "A" full-res, "B" horizontally sub-sampled by 2 -> different
        // per-component recon widths (32 vs 16), so swapping their channel-map
        // entries is a geometry mismatch the cross-check must catch.
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("B", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let a_w = w as usize;
        let b_w = (divp(w - 1, 2) - divp(0, 2) + 1) as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for _ in 0..a_w {
                data.extend_from_slice(&Half::from((y as f32) + 1.0).bits().to_le_bytes());
            }
            for _ in 0..b_w {
                data.extend_from_slice(&Half::from((y as f32) + 100.0).bits().to_le_bytes());
            }
        }

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        assert!(
            compressed.len() < data.len(),
            "fixture must actually compress, not store raw"
        );
        assert_eq!(u16::from_be_bytes([compressed[6], compressed[7]]), 2);

        // Swap the two file-index entries: the codestream component with real
        // (recon) width `a_w` now claims to be file channel "B" (width `b_w`),
        // and vice versa.
        let mut corrupt = compressed.clone();
        corrupt.swap(8, 10);
        corrupt.swap(9, 11);

        let mut c2 = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        assert!(
            c2.uncompress(&corrupt, 0, 0).is_err(),
            "swapped per-component geometry must be rejected (corrupt chunk)"
        );
    }

    /// Mutation-testing gap fix: `decode_chunk`'s per-component geometry
    /// cross-check is a 6-term OR (`h!=recon_h || w!=recon_w ||
    /// downsampling.y==0 || downsampling.x==0 ||
    /// h!=(image_height/downsampling.y) || w!=(image_width/downsampling.x)`).
    /// Every term pairs with another driven by the SAME underlying value
    /// (`recon_h`/`h!=(.../downsampling.y)` both trace back to one
    /// component's `YRsiz`; `recon_w`/`w!=(.../downsampling.x)` both trace
    /// back to `XRsiz`), so an x-only or y-only mismatch normally makes BOTH
    /// of that axis's terms true together - masking any single `||`->`&&`
    /// mutation between them. `downsampling.{x,y}==0` is unreachable at all:
    /// jph-rs itself rejects `XRsiz`/`YRsiz == 0` before this Rust-level
    /// check ever runs ("Wrong SIZ-YRsiz value of 0").
    ///
    /// The ONE genuine decoupling: `h = image_height / y_sampling` (plain
    /// integer division, this function's own doc a few lines up: "a channel
    /// whose sampling does not evenly divide the chunk height cannot match
    /// and is correctly rejected below" - INTENTIONAL, not the y-sampling
    /// buffer-size bug fixed separately in `htj2k_y_subsampled_channel_round_trips`,
    /// which was about `uncompressed_size`, not this comparison). An ODD
    /// image height with a y-subsampled channel makes jph-rs's own
    /// ceiling-rounded `recon_h` disagree with `image_height/downsampling.y`
    /// even though BOTH ultimately derive from the same `y_sampling`/`YRsiz`:
    /// `h!=recon_h` fires while `h!=(image_height/downsampling.y)` does
    /// NOT, decoupling those two terms and letting the `||` between them (and
    /// the ones after `h!=recon_h`) be tested in isolation. Verified by hand
    /// (not guessed) via a debug probe before writing this: h=15/2=7,
    /// recon_h=8 (ceil(15/2)), `image_height/downsampling.y`=15/2=7 - so
    /// `h!=recon_h` (7!=8) is true while `h!=(image_height/downsampling.y)`
    /// (7!=7) is false, and every other term is false (width matches, no
    /// zero downsampling).
    #[test]
    fn htj2k_rejects_component_geometry_mismatch_odd_height_isolates_first_term() {
        let (w, h) = (32i32, 15i32); // ODD height.
        let mut hdr = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Htj2k256).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("C", Channel::new(PixelType::Half, 1, 2, false)).unwrap(); // y_sampling = 2, odd h.
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        let ww = w as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for _ in 0..ww {
                data.extend_from_slice(&Half::from((y as f32) + 1.0).bits().to_le_bytes());
            }
            if y % 2 == 0 {
                for _ in 0..ww {
                    data.extend_from_slice(&Half::from((y as f32) + 100.0).bits().to_le_bytes());
                }
            }
        }

        let mut c = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        let mut c2 = Htj2kCompressor::new(&hdr, w as usize * 4, 256).unwrap();
        assert!(
            c2.uncompress(&compressed, 0, 0).is_err(),
            "an odd image height that does not evenly divide a channel's y_sampling must be \
             rejected (h!=recon_h, per this function's own documented contract), not silently \
             mismatched-but-accepted"
        );
    }

    /// Mutation-testing gap fix, isolating the LAST two `||`s in the same
    /// 6-term geometry guard (`h!=(image_height/downsampling.y)` = T5, and
    /// `w!=(image_width/downsampling.x)` = T6) WITHOUT any byte patch (an
    /// earlier attempt patched `YRsiz` directly, which also desynced the
    /// entropy-coded payload from its declared geometry and made jph-rs's
    /// own decoder fail first, masking the Rust-level check entirely).
    /// Instead: ENCODE with header A (channel `S` at `y_sampling=4`, a real,
    /// fully self-consistent codestream: `YRsiz=4`, `recon_h=ceil(11/4)=3`),
    /// then DECODE THE SAME BYTES with header B, identical except `S` is
    /// declared `y_sampling=3`. The codestream itself is untouched and
    /// internally consistent (jph-rs decodes exactly what it encoded), so no
    /// entropy desync - only the DECODER's own `h` (from header B's
    /// declaration) disagrees with what the codestream reports. Verified by
    /// probe before writing this: `h=3` (`floor(11/3)`), `recon_h=3`
    /// (`ceil(11/4)`, MATCHES - T1 false), `downsampling.y=4` (the real
    /// encoded value) so `image_height/downsampling.y=floor(11/4)=2!=3` -
    /// T5 true, and the resulting error IS the geometry-mismatch message
    /// (not a jph-rs decode error), confirming genuine isolation this time.
    /// Verified by hand-mutation: kills BOTH remaining confirmed operators
    /// (the one before T5 and the one before T6) - with T1..T4 all false and
    /// T5 true, T6 stays false, so weakening either adjoining `||` to `&&`
    /// changes the overall result.
    #[test]
    fn htj2k_rejects_component_geometry_mismatch_mismatched_decode_header() {
        let (w, h) = (32i32, 11i32);
        let mut hdr_a = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Htj2k256).unwrap();
        let mut cl_a = ChannelList::new();
        cl_a.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl_a.insert("S", Channel::new(PixelType::Half, 1, 4, false)).unwrap();
        hdr_a.insert("channels", ChannelListAttribute::new(cl_a)).unwrap();
        let ww = w as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for _ in 0..ww {
                data.extend_from_slice(&Half::from((y as f32) + 1.0).bits().to_le_bytes());
            }
            if y % 4 == 0 {
                for _ in 0..ww {
                    data.extend_from_slice(&Half::from((y as f32) + 100.0).bits().to_le_bytes());
                }
            }
        }
        let mut c = Htj2kCompressor::new(&hdr_a, w as usize * 4, 256).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();

        let mut hdr_b = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Htj2k256).unwrap();
        let mut cl_b = ChannelList::new();
        cl_b.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl_b.insert("S", Channel::new(PixelType::Half, 1, 3, false)).unwrap(); // mismatched y_sampling.
        hdr_b.insert("channels", ChannelListAttribute::new(cl_b)).unwrap();
        let mut c2 = Htj2kCompressor::new(&hdr_b, w as usize * 4, 256).unwrap();
        match c2.uncompress(&compressed, 0, 0) {
            Err(ExrError::Input(msg)) => assert_eq!(
                msg, "HTJ2K: per-component geometry does not match the codestream (corrupt chunk)",
                "expected the geometry-mismatch error specifically, got: {msg}"
            ),
            other => panic!("expected the geometry-mismatch error, got {other:?}"),
        }
    }

    /// A subsampled channel with an odd (unaligned) `min_x` must still
    /// size the planar decode buffer correctly. `channel_byte_width` used to
    /// compute `divp(image_width, xs)`, implicitly assuming `min_x == 0`; for
    /// `min_x` odd and `xs == 2` that undercounts the true per-channel sample
    /// count by one relative to `channel_width` (which every per-line loop
    /// already uses), sizing `bpl`/`uncompressed_size` too small.
    #[test]
    fn htj2k_subsampled_odd_min_x_round_trips() {
        // Data window x in [2, 33] (32 columns, min_x = 2 is ALIGNED to the
        // xs=2 sub-sampling), y in [0, 15]. Before the fix, `channel_byte_width`
        // computed `divp(image_width, xs)` (min_x == 0 assumed), which happens
        // to match `channel_width` here too (both give 16) — this case is a
        // regression guard that a non-zero (but aligned) min_x still round-trips
        // once `bpl`/`uncompressed_size` are computed from the data-window-
        // relative width.
        let (min_x, max_x, min_y, max_y) = (2i32, 33i32, 0i32, 15i32);
        let h = max_y - min_y + 1;
        let hdr = {
            let mut hdr = Header::with_dimensions(
                max_x - min_x + 1,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Htj2k256,
            )
            .unwrap();
            // Override the data window explicitly to get a non-zero min_x (the
            // `with_dimensions` helper always starts at (0, 0)).
            *hdr.data_window_mut().unwrap() = Box2i {
                min: imath_rs::V2i { x: min_x, y: min_y },
                max: imath_rs::V2i { x: max_x, y: max_y },
            };
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            cl.insert("B", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
            hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
            hdr
        };

        let a_w = (divp(max_x, 1) - divp(min_x, 1) + 1) as usize;
        let b_w = (divp(max_x, 2) - divp(min_x, 2) + 1) as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..a_w {
                let v = (y as f32) + (x as f32) * 0.25 + 1.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
            for x in 0..b_w {
                let v = (y as f32) + (x as f32) * 0.5 + 100.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
        }

        let mut c = Htj2kCompressor::new(&hdr, (a_w + b_w) * 2, 256).unwrap();
        let compressed = c.compress(&data, min_y).unwrap().to_vec();
        let restored = c.uncompress(&compressed, min_y, 0).unwrap();
        assert_eq!(
            restored.len(),
            data.len(),
            "planar decode buffer must be sized from the data-window-relative \
             channel width, not divp(image_width, xs)"
        );
    }

    /// `Htj2kCompressor::uncompress`'s `self.out` is a
    /// persistent field (`self.out.clear(); self.out.resize(uncompressed_size,
    /// 0)`), reused across calls -- unlike DWA's UNKNOWN sub-stream (whose
    /// deterministic-zero tail on a short decode is explicitly load-bearing),
    /// HTJ2K's planar assembly loop covers every byte of `self.out` on every
    /// success path (each channel writes its own bytes across every row its
    /// own `y_sampling` includes; every error path returns `Err` before
    /// `self.out` would ever be read as a result). Poison `self.out` with an
    /// implausible value BEFORE the first real call on the SAME subsampled
    /// (planar) fixture `htj2k_subsampled_odd_min_x_round_trips` uses,
    /// then diff against a fresh instance -- proves that empirically rather
    /// than trusting the trace alone.
    #[test]
    fn poison_reused_out_matches_fresh_instance_on_subsampled_planar_decode() {
        let (min_x, max_x, min_y, max_y) = (2i32, 33i32, 0i32, 15i32);
        let h = max_y - min_y + 1;
        let hdr = {
            let mut hdr = Header::with_dimensions(
                max_x - min_x + 1, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Htj2k256,
            )
            .unwrap();
            *hdr.data_window_mut().unwrap() = Box2i {
                min: imath_rs::V2i { x: min_x, y: min_y },
                max: imath_rs::V2i { x: max_x, y: max_y },
            };
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            cl.insert("B", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
            hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
            hdr
        };
        let a_w = (divp(max_x, 1) - divp(min_x, 1) + 1) as usize;
        let b_w = (divp(max_x, 2) - divp(min_x, 2) + 1) as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..a_w {
                let v = (y as f32) + (x as f32) * 0.25 + 1.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
            for x in 0..b_w {
                let v = (y as f32) + (x as f32) * 0.5 + 100.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
        }

        let mut poisoned = Htj2kCompressor::new(&hdr, (a_w + b_w) * 2, 256).unwrap();
        let compressed = poisoned.compress(&data, min_y).unwrap().to_vec();
        poisoned.out = vec![0xAAu8; data.len() * 4]; // poison BEFORE the first uncompress
        let restored_poisoned = poisoned.uncompress(&compressed, min_y, 0).unwrap().to_vec();

        let mut fresh = Htj2kCompressor::new(&hdr, (a_w + b_w) * 2, 256).unwrap();
        let compressed_fresh = fresh.compress(&data, min_y).unwrap().to_vec();
        let restored_fresh = fresh.uncompress(&compressed_fresh, min_y, 0).unwrap().to_vec();

        assert_eq!(
            restored_poisoned, restored_fresh,
            "a poisoned, reused `self.out` produced different decoded bytes than a fresh \
             instance -- some byte is not fully overwritten by the planar assembly loop"
        );
        assert_eq!(restored_poisoned, data, "poisoned decode must still match the original data");
    }

    /// One chunk shape for [`decode_overwrites_every_byte_for_every_chunk_shape`]:
    /// data window `0..w x 0..h`, channels `(name, type, x_sampling, y_sampling)`
    /// in name order, and the chunk's pixel rectangle (`None` = a scanline chunk
    /// starting at `min_y`, `Some(range)` = a tile).
    struct Shape {
        name: &'static str,
        w: i32,
        h: i32,
        chans: &'static [(&'static str, PixelType, i32, i32)],
        min_y: i32,
        tile: Option<Box2i>,
    }

    /// Smooth deterministic line buffer for `rows` x `[min_x, max_x]`, honouring
    /// each channel's sub-sampling (a channel only contributes on rows its
    /// `y_sampling` includes, `divp(max_x, xs) - divp(min_x, xs) + 1` samples).
    fn build_shape_buffer(
        chans: &[(&str, PixelType, i32, i32)],
        (min_x, max_x): (i32, i32),
        rows: std::ops::Range<i32>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in rows {
            for (ci, &(_, ty, xs, ys)) in chans.iter().enumerate() {
                if y % ys != 0 {
                    continue;
                }
                let nx = divp(max_x, xs) - divp(min_x, xs) + 1;
                for x in 0..nx {
                    let v = (ci as f32) * 8.0 + (y as f32) + (x as f32) * 0.5 + 1.0;
                    match ty {
                        PixelType::Half => buf.extend_from_slice(&Half::from(v).bits().to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&(v as u32).to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// HTJ2K decode no longer zero-fills `self.out` — it must
    /// therefore overwrite EVERY byte of the chunk. Proved per chunk shape by
    /// decoding into a buffer pre-filled with 0xAA, then 0x55, at three
    /// lengths (shorter than the chunk: stale prefix + zero-filled growth;
    /// exactly the chunk; longer: truncated), and requiring all decodes to equal
    /// a fresh instance's (whose `out` starts empty). A byte the decoder skipped
    /// would read 0xAA in one run, 0x55 in the other and 0 in the fresh one, so
    /// it cannot hide even when the true value happens to be 0xAA or 0x55.
    /// `Htj2kOutKept` must fire exactly for the two lengths that keep stale
    /// bytes (proof the no-zero-fill path ran).
    ///
    /// Shapes: a full 256-line chunk, the partial edge chunk below it, a
    /// partial right-edge tile at a non-zero origin, all three sample types
    /// mixed (incl. RGB names so the colour transform engages), single
    /// channels of each type, and x/y-subsampled (planar) layouts.
    #[test]
    fn decode_overwrites_every_byte_for_every_chunk_shape() {
        use crate::probe::{self, Probe};
        const RGBA: &[(&str, PixelType, i32, i32)] = &[
            ("A", PixelType::Float, 1, 1),
            ("B", PixelType::Half, 1, 1),
            ("G", PixelType::Half, 1, 1),
            ("R", PixelType::Half, 1, 1),
        ];
        const MIXED: &[(&str, PixelType, i32, i32)] = &[
            ("a", PixelType::Uint, 1, 1),
            ("b", PixelType::Float, 1, 1),
            ("c", PixelType::Half, 1, 1),
            ("d", PixelType::Uint, 1, 1),
        ];
        const SUB: &[(&str, PixelType, i32, i32)] = &[
            ("A", PixelType::Half, 1, 1),
            ("B", PixelType::Float, 2, 2),
            ("C", PixelType::Uint, 2, 1),
            ("D", PixelType::Half, 1, 2),
        ];
        let shapes = [
            Shape { name: "full 256-line chunk", w: 96, h: 256, chans: RGBA, min_y: 0, tile: None },
            Shape { name: "partial edge chunk", w: 96, h: 300, chans: RGBA, min_y: 256, tile: None },
            Shape { name: "mixed uint/float/half", w: 80, h: 48, chans: MIXED, min_y: 0, tile: None },
            Shape { name: "one half", w: 128, h: 64, chans: &[("Y", PixelType::Half, 1, 1)], min_y: 0, tile: None },
            Shape { name: "one float", w: 128, h: 64, chans: &[("Y", PixelType::Float, 1, 1)], min_y: 0, tile: None },
            Shape { name: "one uint", w: 128, h: 64, chans: &[("Y", PixelType::Uint, 1, 1)], min_y: 0, tile: None },
            Shape { name: "subsampled planar", w: 64, h: 64, chans: SUB, min_y: 0, tile: None },
            Shape {
                name: "partial right-edge tile",
                w: 100,
                h: 70,
                chans: RGBA,
                min_y: 0,
                tile: Some(Box2i::from_min_max(imath_rs::V2i::new(64, 0), imath_rs::V2i::new(99, 63))),
            },
        ];

        for sh in &shapes {
            let hdr = {
                let mut hdr = Header::with_dimensions(
                    sh.w, sh.h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Htj2k256,
                )
                .unwrap();
                let mut cl = ChannelList::new();
                for &(n, ty, xs, ys) in sh.chans {
                    cl.insert(n, Channel::new(ty, xs, ys, false)).unwrap();
                }
                hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
                hdr
            };
            // The chunk's pixel rectangle: a tile's own, else the scanline block.
            let rect = sh.tile.unwrap_or_else(|| {
                let max_y = (sh.min_y + 255).min(sh.h - 1);
                Box2i::from_min_max(imath_rs::V2i::new(0, sh.min_y), imath_rs::V2i::new(sh.w - 1, max_y))
            });
            let data = build_shape_buffer(sh.chans, (rect.min.x, rect.max.x), rect.min.y..rect.max.y + 1);

            let mut enc = Htj2kCompressor::new(&hdr, sh.w as usize * 16, 256).unwrap();
            let compressed = match sh.tile {
                Some(r) => enc.compress_tile(&data, r),
                None => enc.compress(&data, sh.min_y),
            }
            .unwrap()
            .to_vec();
            assert!(compressed.len() < data.len(), "{}: fixture must not be stored raw", sh.name);

            let decode = |c: &mut Htj2kCompressor| match sh.tile {
                Some(r) => c.uncompress_tile(&compressed, r, 0).unwrap().to_vec(),
                None => c.uncompress(&compressed, sh.min_y, 0).unwrap().to_vec(),
            };
            let mut fresh = Htj2kCompressor::new(&hdr, sh.w as usize * 16, 256).unwrap();
            let expect = decode(&mut fresh);
            assert_eq!(expect.len(), data.len(), "{}: decoded size", sh.name);

            for poison in [0xAAu8, 0x55] {
                for len in [data.len() / 2, data.len(), data.len() + 13] {
                    let mut c = Htj2kCompressor::new(&hdr, sh.w as usize * 16, 256).unwrap();
                    c.out = vec![poison; len];
                    probe::reset(Probe::Htj2kOutKept);
                    let got = decode(&mut c);
                    assert_eq!(got, expect, "{}: poison {poison:#x} len {len}: a byte was not overwritten", sh.name);
                    let kept = probe::get(Probe::Htj2kOutKept);
                    assert_eq!(kept, u64::from(len >= data.len()), "{}: Htj2kOutKept for len {len}", sh.name);
                }
            }
        }
    }

    /// Verified against C++ 3.4.15 (not a codec-level structural limit):
    /// a subsampled channel whose data window `min_x` is NOT a multiple of its
    /// x-sub-sampling factor is not a case any HTJ2K decoder — ours or C++'s —
    /// ever has to handle, because C++'s `Header::sanityCheck`
    /// (`ImfHeader.cpp:1221-1227`,
    /// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfHeader.cpp#L1221-L1227>)
    /// throws `ArgExc` — *"The minimum x coordinate of the image's data window
    /// is not a multiple of the x subsampling factor..."* — at header-open
    /// time, long before any codec runs; `OpenEXRCore`'s
    /// `compute_sampled_width` (`internal_util.h:50-59`) even comments
    /// *"we require that the start_x % x_sampling == 0"* as its precondition.
    /// `crates/exr-core/src/header.rs:995-999` already ports this exact check
    /// in `Header::sanity_check`, which every real reader/writer path in this
    /// crate calls before any compressor is constructed.
    /// So on any legitimately-opened
    /// file, `Htj2kCompressor::new` never sees this combination — it is C++'s
    /// header validation, not any codestream-shape limit, that rules it out.
    /// This test constructs the `Header` directly (bypassing `sanity_check`,
    /// which the real open path never allows) purely to confirm the codec has
    /// a defence-in-depth `Err` here too — never a raw panic — rather than to
    /// claim this is how a real file could ever reach the codec.
    #[test]
    fn htj2k_subsampled_unaligned_min_x_errors_not_panics() {
        let (min_x, max_x, min_y, max_y) = (1i32, 32i32, 0i32, 15i32);
        let h = max_y - min_y + 1;
        let hdr = {
            let mut hdr = Header::with_dimensions(
                max_x - min_x + 1,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Htj2k256,
            )
            .unwrap();
            *hdr.data_window_mut().unwrap() = Box2i {
                min: imath_rs::V2i { x: min_x, y: min_y },
                max: imath_rs::V2i { x: max_x, y: max_y },
            };
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            cl.insert("B", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
            hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
            hdr
        };

        // The primary check: this is exactly what C++'s `Header::sanityCheck`
        // rejects (ArgExc) at header-open time, before any codec is involved.
        assert!(
            hdr.sanity_check(false, false).is_err(),
            "min_x=1 not a multiple of channel B's x_sampling=2 must fail Header::sanity_check, \
             matching C++'s ImfHeader.cpp:1221-1227 — a real reader/writer path would already \
             have refused this file long before constructing the HTJ2K codec"
        );

        let a_w = (divp(max_x, 1) - divp(min_x, 1) + 1) as usize;
        let b_w = (divp(max_x, 2) - divp(min_x, 2) + 1) as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..a_w {
                let v = (y as f32) + (x as f32) * 0.25 + 1.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
            for x in 0..b_w {
                let v = (y as f32) + (x as f32) * 0.5 + 100.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
        }

        // Defence in depth: even bypassing `sanity_check` (as the real open
        // path never does), the codec itself must still error, never panic.
        let mut c = Htj2kCompressor::new(&hdr, (a_w + b_w) * 2, 256).unwrap();
        assert!(
            c.compress(&data, min_y).is_err(),
            "unaligned min_x + sub-sampling must error cleanly even if sanity_check is bypassed"
        );
    }

    /// Y-subsampling bug (found via a mutation-testing repro): the
    /// planar-path packed buffer size was computed as `bpl * image_height`
    /// (`bpl` = sum of full-height per-channel row bytes), which silently
    /// treats a `y_sampling > 1` channel as if it contributed a row for
    /// EVERY scanline. C++'s `compute_chunk_unpack_size`
    /// (`openexr/src/lib/OpenEXRCore/chunk.c:677-700`,
    /// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/chunk.c#L677-L700>)
    /// instead sums, per channel, `chansz(width, x_sampling) *
    /// chansz(height, y_sampling)` — a y-subsampled channel contributes only
    /// `height / y_sampling` rows, matching `misc::bytes_per_line_table`'s and
    /// `misc::active_row_count`'s (also used by DWA's `channel_dims`)
    /// packed-row convention.
    /// `internal_ht.cpp:253`/`:432`'s `bpl * chunk.height` is only an overflow
    /// SANITY bound (`bpl > INT32_MAX || bpl * height > PTRDIFF_MAX`), never
    /// the actual buffer size — conflating the two undercounted the true
    /// packed size for any y-subsampled channel and, on decode, allocated
    /// (and reported) a buffer padded with the missing channel's full height
    /// worth of extra bytes.
    ///
    /// 32x16 image, channel "A" full-res + channel "C" with `y_sampling=2`
    /// (only even rows carry "C" samples, the crate's standard packed-buffer
    /// convention). Before the fix: `uncompress` returned 2048 bytes
    /// (`bpl=128` * `height=16`) instead of the correct 1536
    /// (`A`: 32*16*2=1024, `C`: 32*8*2=512).
    #[test]
    fn htj2k_y_subsampled_channel_round_trips() {
        let (w, h) = (32i32, 16i32);
        let hdr = {
            let mut hdr = Header::with_dimensions(
                w,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Htj2k256,
            )
            .unwrap();
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            cl.insert("C", Channel::new(PixelType::Half, 1, 2, false)).unwrap();
            hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
            hdr
        };

        let ww = w as usize;
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..ww {
                let v = (y as f32) + (x as f32) * 0.25 + 1.0;
                data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
            }
            if y % 2 == 0 {
                for x in 0..ww {
                    let v = (y as f32) + (x as f32) * 0.5 + 100.0;
                    data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
                }
            }
        }
        assert_eq!(data.len(), 1536, "test fixture packed size sanity check");

        let mut c = Htj2kCompressor::new(&hdr, ww * 4, 256).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        let restored = c.uncompress(&compressed, 0, 0).unwrap();
        assert_eq!(
            restored.len(),
            data.len(),
            "planar decode buffer must be sized per-channel (width * height/y_sampling), \
             not bpl * image_height, once any channel is y-subsampled"
        );
        assert_eq!(restored, data, "y-subsampled channel must round-trip byte-exact");
    }
}
