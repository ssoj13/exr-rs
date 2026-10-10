//! Port of OpenEXR `ImfDwaCompressor` / `OpenEXRCore/internal_dwa*.h` — the
//! **DWAA** and **DWAB** lossy DCT scanline codecs (DreamWorks Animation
//! compression).
//!
//! # What this codec does (1:1 with C++)
//!
//! DWA is a JPEG-like lossy scheme tuned for HDR half-float imagery:
//!
//! 1. **Classify** each channel ([`classifier`]): `R/G/B/Y/RY/BY` → lossy DCT,
//!    `A` → RLE, everything else → zlib ("Unknown").  Exactly one rule set is
//!    active: the encoder uses the **default** table, the decoder uses the
//!    **legacy** table for v0/v1 streams and the rules **stored in the chunk**
//!    for v2 streams.
//! 2. **CSC group** matching `R/G/B` triplets (same name prefix + sampling), in
//!    the order each layer PREFIX first appears in the channel list, and
//!    convert to Y′CbCr ([`csc`]) so chroma can be quantised harder.
//! 3. **Perceptual encode** color channels through the nonlinear LUT
//!    ([`tables`]) unless the channel is `p_linear`.
//! 4. **DCT** each 8×8 block ([`dct`]), **quantise** to the fewest-bits
//!    half-float within tolerance ([`quantize`]), **zigzag + RLE** the AC
//!    coefficients ([`zigzag`]).
//! 5. **Entropy code**: AC via static Huffman (or DEFLATE for tiled DWAA), DC
//!    via byte-deconstruct + zlib, RLE channels via byte-RLE + zlib, Unknown
//!    channels via zlib.
//!
//! The on-disk chunk is an 11×`u64` little-endian header, an optional v2
//! channel-rules section ([`channel_rules`]), then the four compressed sections
//! (Unknown, AC, DC, RLE) in that order.
//!
//! **DWAA** batches 32 scanlines per chunk; **DWAB** batches 256.  The only
//! difference between them is `num_scan_lines()`.
//!
//! # Decoder hardening
//!
//! Every size and count in the chunk header is attacker-controlled.  Before any
//! of them is used to allocate or index, [`decompress_dwa`] rejects the negative
//! (top-bit-set) encodings, bounds each one against the chunk geometry the way
//! `DwaCompressor_initializeBuffers` sizes its buffers, and does all offset
//! arithmetic with `checked_*`.  Every zlib sub-stream must produce **exactly**
//! the declared byte count — that is C++'s behaviour too, because it passes a
//! NULL `actual_out` to `libdeflate_zlib_decompress_ex` (or compares `destLen`),
//! so a short stream is a corrupt chunk rather than a silent zero fill.
//!
//! # Byte-exactness notes (confirmed against the C++ core)
//!
//! * **Parity target is OpenEXR 3.4.15's OpenEXRCore, MSVC-x64 build — not the
//!   generic v3.3.11 IlmImf source these files are ported from.** On MSVC x64
//!   `internal_dwa_simd.h` unconditionally compiles the SSE2 DCT bodies, and the
//!   build host that produced our committed goldens has `avx=0` at runtime, so
//!   `dctInverse8x8_sse2_N`/the SSE2 `dctForward8x8` are what actually ran — see
//!   [`dct`]'s module doc for the full platform-dependence story (GCC x64 + AVX CPU uses AVX asm instead;
//!   ARM uses the plain scalar bodies — none of the three are bit-identical to
//!   each other).
//! * The perceptual LUT is the analytic curve from `internal_dwa_table_init.c`
//!   (3.4.15's rewrite of the v3.3.11 `dwaLookups.cpp` generator — single
//!   precision throughout, not `dwaLookups.cpp`'s double-precision `>1.0`
//!   branch), evaluated through bit-exact [`imath_rs::Half`] rounding (see
//!   [`tables`]).
//! * Unknown, AC-DEFLATE, and RLE sub-streams are zlib level **9** (the
//!   hardcoded `9` in `DwaCompressor_compress`); DC uses the context zip level
//!   (`_zipLevel`), which resolves to the EXR default level **4**.
//! * DC bytes are pre-processed with the shared
//!   [`crate::compressor::predictor::reorder_and_predict`] (== `internal_zip_deconstruct_bytes`).
//! * If the assembled output is **not smaller** than the raw line buffer, the
//!   codec returns the raw bytes unchanged (`nWritten >= packed_bytes →` store
//!   raw), which the scanline seam detects by size — exactly as ZIP/B44 do.
//!
//! # Format / seam
//!
//! DWA consumes and produces the machine-independent **Xdr** (little-endian)
//! line buffer, so the seam's gather/scatter is unchanged.  The codec captures
//! the channel list + data-window bounds from the header at construction and
//! rebuilds each chunk's pixel rectangle from `min_y` and `num_scan_lines`.
//!
//! Port targets: structure/layout ported from v3.3.11 `ImfDwaCompressor.cpp`;
//! **behavioural** parity oracle (the one `xtask parity-exr` cross-checks against)
//! is v3.4.15's `OpenEXRCore/internal_dwa_compressor.h`, which changed real
//! behaviour in several places this module now follows over v3.3.11 (the LUT
//! generator's precision, `validate_size`'s stricter UNKNOWN/RLE size checks,
//! and — always builds the CSC en/decoder with the perceptual LUT regardless
//! of `pLinear`).
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDwaCompressor.cpp>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_dwa_compressor.h>

pub mod channel_rules;
pub mod classifier;
pub mod csc;
pub mod dct;
mod decoder;
mod encoder;
pub mod quantize;
pub mod tables;
pub mod zigzag;

use crate::attr::{ChannelList, PixelType};
use crate::compressor::piz::huffman as huf;
use crate::compressor::predictor::{reorder_and_predict, unpredict_and_reorder};
use crate::compressor::rle::{rle_compress, rle_decompress};
use crate::compressor::zlib::{new_deflate_compressor, LowCompressor, LowDecompressor, ShortOutputPolicy};
use crate::compressor::{Compressor, Format};
use crate::error::{ExrError, Result};
use crate::header::Header;
use crate::probe::{self, Probe};

use classifier::{classify_with, CompressionScheme, CscIndex, Rule};
use dct::BLOCK_SIZE;
use decoder::LossyDctDecoder;
use encoder::LossyDctEncoder;
use tables::ensure_tables_initialized;

use imath_rs::{modp, Box2i, Half, V2i};

/// Default DWA compression level (45.0 = visually lossless).  Base quantisation
/// error is `level / 100000`.
pub const DEFAULT_COMPRESSION_LEVEL: f32 = 45.0;

/// DWAA scanlines per chunk.
pub const DWAA_SCANLINES: i32 = 32;

/// DWAB scanlines per chunk.
pub const DWAB_SCANLINES: i32 = 256;

/// zlib level for the Unknown / AC-DEFLATE / RLE sub-streams (hardcoded `9` in
/// `DwaCompressor_compress`).
const DWA_ZLIB_LEVEL_9: i32 = 9;

/// Fallback zlib level for the DC sub-stream when no `Header` context is
/// available (e.g. plain unit tests calling `compress_dwa` directly) —
/// matches the EXR default `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4`, the value
/// `Header::zip_level()` itself resolves to when unset. Production
/// callers (`DwaCompressor::compress`) pass `header.zip_level()`'s actual
/// resolved value instead of this constant — never read outside `#[cfg(test)]`.
#[cfg(test)]
const DWA_ZLIB_LEVEL_DC_DEFAULT: i32 = 4;

// ---------------------------------------------------------------------------
// Header layout (DataSizesSingle) — 11 u64 fields, little-endian.
// ---------------------------------------------------------------------------

/// Index into the DWA chunk header — port of the `DataSizesSingle` enum.
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
enum Field {
    Version = 0,
    UnknownUncompressedSize = 1,
    UnknownCompressedSize = 2,
    AcCompressedSize = 3,
    DcCompressedSize = 4,
    RleCompressedSize = 5,
    RleUncompressedSize = 6,
    RleRawSize = 7,
    AcUncompressedCount = 8,
    DcUncompressedCount = 9,
    AcCompression = 10,
}

/// Number of header fields (`NUM_SIZES_SINGLE`).
const NUM_HEADER_FIELDS: usize = 11;

/// Header size in bytes.
const HEADER_SIZE: usize = NUM_HEADER_FIELDS * 8;

/// AC entropy coder selector (`AcCompression`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AcCompression {
    StaticHuffman = 0,
    Deflate = 1,
}

impl AcCompression {
    /// C++ switches on the stored value and returns `EXR_ERR_CORRUPT_CHUNK` from
    /// the `default:` arm (`internal_dwa_compressor.h:944`), so anything outside
    /// `{0, 1}` is a corrupt chunk rather than a silent fallback to Huffman.
    fn from_u64(v: u64) -> Result<Self> {
        match v {
            0 => Ok(AcCompression::StaticHuffman),
            1 => Ok(AcCompression::Deflate),
            _ => Err(ExrError::Input(format!(
                "DWA: unknown AC compression method {v}"
            ))),
        }
    }
}

/// Read the 11-field LE header.
fn read_header(data: &[u8]) -> Result<[u64; NUM_HEADER_FIELDS]> {
    if data.len() < HEADER_SIZE {
        return Err(ExrError::Input("DWA data too short for header".into()));
    }
    let mut h = [0u64; NUM_HEADER_FIELDS];
    for (i, slot) in h.iter_mut().enumerate() {
        let o = i * 8;
        *slot = u64::from_le_bytes([
            data[o], data[o + 1], data[o + 2], data[o + 3], data[o + 4], data[o + 5], data[o + 6],
            data[o + 7],
        ]);
    }
    Ok(h)
}

/// Write the 11-field LE header.
fn write_header(h: &[u64; NUM_HEADER_FIELDS], out: &mut Vec<u8>) {
    for &v in h {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Raw zlib (zlib-wrapped DEFLATE) compress using the caller's OWN persistent
/// `comp` — mirrors C++ `exr_compress_buffer` (which our `deflate_rs` matches
/// byte-for-byte). The level is resolved once, at `comp`'s construction
/// (see [`DwaDeflateCompressors::new`]/[`crate::compressor::zlib::new_deflate_compressor`]),
/// not per call (perf follow-up, 2026-09-24 — see `compressor::zlib`'s module doc).
fn zlib_compress(data: &[u8], comp: &mut LowCompressor) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    crate::compressor::zlib::compress_into(comp, data, &mut out)?;
    Ok(out)
}

/// Raw zlib decompress for the AC-DEFLATE/DC/RLE sub-streams, requiring
/// **exactly** `expected` bytes of output.
///
/// C's call sites for these three (`internal_dwa_compressor.h:970-982`/
/// `:1012-1020`/`:1048-1057`) all pass a NON-NULL `&destLen`/`&dstLen` to
/// `exr_uncompress_buffer`. With a non-NULL `actual_out`,
/// `libdeflate_zlib_decompress_ex` can only return `LIBDEFLATE_SUCCESS`
/// (confirmed against `decompress_template.h`: the `SHORT_OUTPUT` return path
/// only exists in the `actual_out_nbytes_ret == NULL` branch), and
/// `exr_uncompress_buffer`'s `SUCCESS` branch always checks
/// `in_bytes == actual_in_bytes`. So for these three sub-streams, routing
/// through [`crate::compressor::zlib::uncompress_into`] with
/// [`ShortOutputPolicy::RequireExact`] (which performs exactly that check)
/// plus this function's own `out.len() != expected` post-check (matching
/// each C call site's own `destLen != uncompBytes` follow-up) is not a
/// divergence: it is what C already does. The UNKNOWN sub-stream is
/// DIFFERENT (NULL `actual_out` in C) — see [`zlib_decompress_unknown`].
fn zlib_decompress_exact(data: &[u8], expected: usize, decompressor: &mut LowDecompressor) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    crate::compressor::zlib::uncompress_into(
        decompressor,
        data,
        &mut out,
        expected.max(1),
        "DWA",
        ShortOutputPolicy::RequireExact,
    )?;
    if out.len() != expected {
        return Err(ExrError::Input(format!(
            "DWA zlib stream produced {} bytes, expected {expected}",
            out.len()
        )));
    }
    Ok(out)
}

/// Raw zlib decompress for DWA's UNKNOWN sub-stream — the one C call site
/// (`internal_dwa_compressor.h:922-930`) that passes NULL for `actual_out`.
///
/// Fills the caller's persistent `out` buffer directly. Unlike [`zlib_decompress_exact`], a decode that
/// produces FEWER than `cap` bytes is NOT an error: `libdeflate` returns
/// `LIBDEFLATE_SHORT_OUTPUT` in that case (reachable only with a NULL
/// `actual_out`), and `exr_uncompress_buffer`'s `SHORT_OUTPUT` branch
/// (`compression.c:144-208`) converts that to `EXR_ERR_SUCCESS`
/// UNCONDITIONALLY -- it does not check `actual_in_bytes` (trailing garbage
/// after a short stream is accepted too), and does not require an exact
/// fill. `out` is truncated to the real decoded length here, so its
/// remaining slots (the tail `decompress_dwa`'s assembly loop would have
/// read from this buffer) come from `output`'s own zero-init -- one
/// deterministic zero-fill.
fn zlib_decompress_unknown(
    data: &[u8],
    cap: usize,
    out: &mut Vec<u8>,
    decompressor: &mut LowDecompressor,
) -> Result<()> {
    crate::compressor::zlib::uncompress_into(
        decompressor,
        data,
        out,
        cap.max(1),
        "DWA-UNKNOWN",
        ShortOutputPolicy::TolerateShort,
    )
}

/// Checked `a + b` for a stream-derived size, as a clean `ExrError`.
fn ck_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| ExrError::Input("DWA: chunk size arithmetic overflow".into()))
}

/// Checked `a * b` for a stream-derived size, as a clean `ExrError`.
fn ck_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| ExrError::Input("DWA: chunk size arithmetic overflow".into()))
}

// ---------------------------------------------------------------------------
// Per-channel subsampling support.
// ---------------------------------------------------------------------------
//
// C++ derives each channel's own `width`/`height` for the chunk from
// `compute_sampled_width`/`compute_sampled_height` (`internal_util.h`) and
// feeds/reads the packed scanline buffer skipping rows a channel's `y_samples`
// excludes (`internal_dwa_compressor.h:347-361`/`:1086-1099`). That is exactly
// the convention `misc::bytes_per_line_table` already implements for the
// generic (non-DWA) packed-buffer layout — [`crate::misc::active_row_count`]
// mirrors its `modp(y, y_sampling) == 0` row test (moved there so `compressor
// ::htj2k` doesn't need to cross-import from this module for the same
// convention), and the per-channel width
// uses the same `width / x_sampling` division (both reduce to the unsampled
// case when `x_sampling`/`y_sampling == 1`, so a single formula serves both
// cases; no separate "subsampled" code path is needed anywhere below).

/// This chunk's own `(width, height)` for one channel — `compute_sampled_width`/
/// `compute_sampled_height` (`internal_util.h`), matching `bytes_per_line_table`'s
/// `width / x_sampling` convention for X and [`crate::misc::active_row_count`] for Y.
fn channel_dims(width: usize, rect_min_y: i32, rect_max_y: i32, x_sampling: i32, y_sampling: i32) -> (usize, usize) {
    let w = width / (x_sampling.max(1) as usize);
    let h = crate::misc::active_row_count(rect_min_y, rect_max_y, y_sampling);
    (w, h)
}

// ---------------------------------------------------------------------------
// Per-channel snapshot captured from the header.
// ---------------------------------------------------------------------------

/// One channel's classification + layout for the codec.
#[derive(Clone, Debug)]
struct ChanInfo {
    scheme: CompressionScheme,
    type_: PixelType,
    bytes_per_sample: usize,
    /// Horizontal sampling (`exr_coding_channel_info_t::x_samples`).
    x_sampling: i32,
    /// Vertical sampling (`exr_coding_channel_info_t::y_samples`).
    y_sampling: i32,
    /// `true` -> skip the perceptual LUT (the C++ `p_linear` flag).
    quantize_linearly: bool,
}

/// One CSC group: file-channel indices of its R/G/B members.
struct CscGroup {
    r: usize,
    g: usize,
    b: usize,
}

/// Result of [`classify_channels`] — the per-channel info, the CSC groups in
/// C++'s order, and a "this channel belongs to a group" flag per channel.
struct Classification {
    infos: Vec<ChanInfo>,
    csc_groups: Vec<CscGroup>,
    in_csc: Vec<bool>,
}

/// Classify every channel with `rules` and build the CSC groups — 1:1 port of
/// `DwaCompressor_classifyChannels` (`internal_dwa_compressor.h:1595-1687`) and
/// its `CscPrefixMap_find` helper (`internal_dwa_channeldata.h:117-142`).
///
/// **Group order is the order in which each layer PREFIX is first seen in the
/// channel list**, not the order of the Red channels.  With `B,G,R,Hair.B,
/// Hair.G,Hair.R` the base layer's prefix (`""`) is seen at channel 0 and
/// `Hair.` at channel 3, so the base group is emitted first.  Ordering by the
/// Red channel instead (the old behaviour) emitted `Hair.` first, which swapped
/// the two layers' pixels in both directions across the C++ boundary.
///
/// This one function serves the encoder and the decoder; only `rules` differs.
fn classify_channels(channels: &ChannelList, rules: &[Rule]) -> Result<Classification> {
    // `CscPrefixMapItem[]`: prefix -> the R/G/B channel filling each slot,
    // kept in first-seen order (C++ appends at the first NULL entry).
    let mut prefix_map: Vec<(&[u8], [Option<usize>; 3])> = Vec::new();
    let mut infos = Vec::new();

    for (c, (name, ch)) in channels.iter().enumerate() {
        // Bytes throughout, matching C++ (`Classifier_find_suffix`/
        // `Classifier_match` never check encoding), so a non-UTF-8 DWA
        // channel name classifies exactly like a UTF-8 one.
        let name = name.as_bytes();
        // C++: prefix_len = Classifier_find_suffix(name) - name.
        let suffix = classifier::find_suffix(name);
        let prefix = &name[..name.len() - suffix.len()];
        let slot = match prefix_map.iter().position(|(p, _)| *p == prefix) {
            Some(i) => i,
            None => {
                prefix_map.push((prefix, [None; 3]));
                prefix_map.len() - 1
            }
        };

        let (scheme, csc) = classify_with(rules, name, ch.type_);
        match csc {
            CscIndex::Red => prefix_map[slot].1[0] = Some(c),
            CscIndex::Green => prefix_map[slot].1[1] = Some(c),
            CscIndex::Blue => prefix_map[slot].1[2] = Some(c),
            CscIndex::None => {}
        }

        infos.push(ChanInfo {
            scheme,
            type_: ch.type_,
            bytes_per_sample: crate::misc::pixel_type_size(ch.type_)?,
            x_sampling: ch.x_sampling,
            y_sampling: ch.y_sampling,
            quantize_linearly: ch.p_linear,
        });
    }

    let mut csc_groups = Vec::new();
    let mut in_csc = vec![false; infos.len()];
    for (_, idx) in &prefix_map {
        let (Some(r), Some(g), Some(b)) = (idx[0], idx[1], idx[2]) else {
            continue;
        };
        // C++ :1666 — a group needs a common sampling pattern.
        let (ri, gi, bi) = (&infos[r], &infos[g], &infos[b]);
        if ri.x_sampling != gi.x_sampling
            || ri.x_sampling != bi.x_sampling
            || ri.y_sampling != gi.y_sampling
            || ri.y_sampling != bi.y_sampling
        {
            continue;
        }
        csc_groups.push(CscGroup { r, g, b });
        in_csc[r] = true;
        in_csc[g] = true;
        in_csc[b] = true;
    }

    Ok(Classification {
        infos,
        csc_groups,
        in_csc,
    })
}

/// Everything `classify_channels` produces PLUS the fixed
/// "which compact slot does file-channel `idx` land in" maps that
/// `compress_dwa`/`decompress_dwa` used to rebuild from scratch every chunk
/// (the `dct_to_info`/`info_to_dct` HashMap and the `dct_buf`->`dct_channels`
/// compaction pass). All of this is a pure function of `(channels, rules)` --
/// it does not depend on the chunk rectangle at all -- so it is computed ONCE
/// and reused for every chunk that shares the same `(channels, rules)`:
/// unconditionally on the encode side (`DwaCompressor::channels` never
/// changes after construction and encode always uses
/// `classifier::default_rules()`), and cache-validated per chunk on the
/// decode side (a v2 stream carries its OWN rules per chunk -- see
/// `DwaScratch::plan`'s doc -- so equality is checked, not assumed).
struct DwaChannelPlan {
    infos: Vec<ChanInfo>,
    csc_groups: Vec<CscGroup>,
    in_csc: Vec<bool>,
    /// Per file-channel index -> its slot in `dct_to_info`/the encoder's
    /// `dct_channels`/decoder's `dct_outputs`, or `usize::MAX` if this
    /// channel is not `LossyDct`.
    dct_slot: Vec<usize>,
    /// Per file-channel index -> its slot among RLE channels, or `usize::MAX`.
    rle_slot: Vec<usize>,
    /// Per file-channel index -> its slot among Unknown channels, or `usize::MAX`.
    unknown_slot: Vec<usize>,
    /// dct slot -> file-channel index (the old `dct_to_info`).
    dct_to_info: Vec<usize>,
    /// file-channel index -> dct slot, for channels that are `LossyDct`
    /// (the old per-chunk-rebuilt `info_to_dct` HashMap).
    info_to_dct: std::collections::HashMap<usize, usize>,
    num_dct: usize,
    num_rle: usize,
    num_unknown: usize,
}

/// Build a [`DwaChannelPlan`] from `classify_channels`'s result -- see the
/// struct doc for why this is cacheable across chunks.
fn build_channel_plan(channels: &ChannelList, rules: &[Rule]) -> Result<DwaChannelPlan> {
    let Classification { infos, csc_groups, in_csc } = classify_channels(channels, rules)?;
    let mut dct_slot = vec![usize::MAX; infos.len()];
    let mut rle_slot = vec![usize::MAX; infos.len()];
    let mut unknown_slot = vec![usize::MAX; infos.len()];
    let mut dct_to_info = Vec::new();
    let mut info_to_dct = std::collections::HashMap::new();
    let mut num_rle = 0usize;
    let mut num_unknown = 0usize;
    for (idx, info) in infos.iter().enumerate() {
        match info.scheme {
            CompressionScheme::LossyDct => {
                dct_slot[idx] = dct_to_info.len();
                info_to_dct.insert(idx, dct_to_info.len());
                dct_to_info.push(idx);
            }
            CompressionScheme::Rle => {
                rle_slot[idx] = num_rle;
                num_rle += 1;
            }
            CompressionScheme::Unknown => {
                unknown_slot[idx] = num_unknown;
                num_unknown += 1;
            }
        }
    }
    let num_dct = dct_to_info.len();
    Ok(DwaChannelPlan {
        infos,
        csc_groups,
        in_csc,
        dct_slot,
        rle_slot,
        unknown_slot,
        dct_to_info,
        info_to_dct,
        num_dct,
        num_rle,
        num_unknown,
    })
}

/// Port of `validate_size` (`internal_dwa_compressor.h:691-726`, OpenEXRCore 3.4.15
/// only — absent from our v3.3.11 port base): for one planar scheme (UNKNOWN or
/// RLE), check the header-declared sizes against `required_size` (the sum of that
/// scheme's channels' planar byte counts, C++'s `planarUncSize`).
///
/// * If `required_size > 0`: `uncompressed_size` must reach it and `compressed_size`
///   must be non-zero (a real compressed section has to be present to fill it).
/// * If `required_size == 0` (no channel of this scheme exists): every size the
///   chunk declares for it — `compressed_size`, `uncompressed_size`, and (for RLE)
///   `extra_size`, the intermediate un-RLE'd byte count before the byte-RLE
///   decode — must be zero.
fn validate_dwa_size(
    compressed_size: usize,
    uncompressed_size: usize,
    extra_size: usize,
    required_size: usize,
) -> Result<()> {
    if required_size > 0 {
        if uncompressed_size < required_size || compressed_size == 0 {
            return Err(ExrError::Input("DWA: planar section too small for its channels".into()));
        }
    } else if uncompressed_size > 0 || compressed_size > 0 || extra_size > 0 {
        return Err(ExrError::Input("DWA: planar section declared with no matching channel".into()));
    }
    Ok(())
}

// ===========================================================================
// decompress_dwa — port of DwaCompressor_uncompress
// ===========================================================================

/// Decompress one DWA chunk into the native-LE scanline-interleaved line
/// buffer.  `rectangle` is the chunk's pixel region and `channels` describes the
/// layout; together they bound every size the chunk header declares.
///
/// See the module docs' "Decoder hardening" section — nothing read from
/// `compressed` reaches an allocation or an index before it has been bounded.
/// Persistent DC/RLE/UNKNOWN decode scratch for [`DwaCompressor`] — mirrors `ac_scratch`'s "filled in place, not reallocated per
/// chunk" contract, one struct instead of three separate fields so the
/// `decompress_dwa` call site reads as the same "&mut self.out, &mut
/// scratch" shape every other codec's out/scratch split uses.
#[derive(Default)]
struct DwaScratch {
    dc: Vec<u16>,
    rle: Vec<u8>,
    unknown: Vec<u8>,
    /// Persistent low-level zlib decompressor shared by all four of DWA's
    /// zlib decode call sites (AC-deflate/DC/RLE/UNKNOWN) — constructed once
    /// per `DwaCompressor` instead of per sub-stream per chunk (perf
    /// follow-up, 2026-09-24; see `compressor::zlib`'s module doc). Safe to
    /// share: each `zlib_decompress_exact`/`zlib_decompress_unknown` call is
    /// a complete, self-contained decode with no state carried between calls.
    decompressor: LowDecompressor,
    /// DC un-predict/reorder scratch, reused instead of a
    /// fresh `vec![0u8; n]` per chunk. [`unpredict_and_reorder`] fully
    /// overwrites every byte (it is the same shared predictor every ZIP/RLE
    /// chunk also runs — see [`crate::compressor::predictor`]'s contract), so
    /// [`super::ensure_len`] is sound here too.
    dc_reorder: Vec<u8>,
    /// Persistent Huffman DECODE-side accelerated-decoder scratch for the AC coefficient stream's static-Huffman path — see
    /// `fast_huf::FastHufDecoder`'s doc.
    fast_huf: crate::compressor::piz::fast_huf::FastHufDecoder,
    /// Cached [`DwaChannelPlan`] plus the exact `channels`
    /// and `rules` it was built from. The plan is a function of BOTH
    /// (`build_channel_plan`), so both are part of the key (keying on
    /// `rules` alone would reuse a stale plan). `channels` is compared by name
    /// AND value, not with `ChannelList::eq`, which ignores names,
    /// while DWA classification matches on names. A v2 stream stores its OWN channel rules per
    /// chunk (`channel_rules::read_channel_rules`), so -- unlike the encode
    /// side, where `classifier::default_rules()` is a compile-time constant --
    /// decode CANNOT assume yesterday classification is still valid: every
    /// call compares the freshly-read `rules` against `plan_rules` (`Rule`
    /// now derives `PartialEq` for exactly this) and only reuses `plan` when
    /// they match byte-for-byte. In the common case (every chunk of a file
    /// shares one rule set) this drops `classify_channels` from "once per
    /// chunk" to "once per file".
    plan: Option<DwaChannelPlan>,
    plan_rules: Vec<Rule>,
    plan_channels: ChannelList,
    /// Per-channel `(width, height)` for the CURRENT chunk
    /// -- genuinely recomputed whenever the rectangle or plan changes, but
    /// the Vec allocation is grow-only: `chan_dims_key` records the
    /// `(width, min_y, max_y)` this buffer was last computed for, and a
    /// repeat (every chunk but the file last is the same shape) skips
    /// recomputation entirely.
    chan_dims: Vec<(usize, usize)>,
    chan_dims_key: Option<(usize, i32, i32)>,
    /// Per-row byte count for the CURRENT chunk -- same grow-only-allocation
    /// / recompute-on-key-change policy as `chan_dims`, keyed identically.
    bytes_per_line: Vec<usize>,
    /// Per-channel decoded DCT
    /// output, keyed by file-channel index -- grow-only, reused across
    /// chunks instead of a fresh `Vec<Option<Vec<u16>>>` (and a fresh
    /// `Vec<u16>` per channel/group inside it) every chunk. Indexed exactly
    /// like the old `dct_outputs` local; `dct_filled[idx]` replaces the old
    /// `Option`'s `None` -- reset to `false` for every channel at the top of
    /// each `decompress_dwa` call, set `true` only for the channels THIS
    /// chunk actually decodes (a channel may not be `LossyDct`, or the
    /// channel list itself can change between calls if the compressor is
    /// reused across files -- unlikely in practice but not assumed away).
    dct_outputs: Vec<Vec<u16>>,
    dct_filled: Vec<bool>,
    /// Row-of-blocks scratch shared by every `decode_channel_into` call this
    /// chunk -- grow-only, `ensure_len`'d to the current
    /// channel's `blocks_x * BLOCK_ELEMENTS` each call (varies with a
    /// subsampled channel's own width).
    row_blocks: Vec<u16>,
    /// Same role as `row_blocks`, for `decode_rgb_into`'s 3-plane (Y/Cb/Cr)
    /// case.
    row_blocks_rgb: [Vec<u16>; 3],
}

/// Decode one DWA chunk into `out` (cleared + resized in place — the same
/// out/scratch policy every other codec's decompress path follows).
fn decompress_dwa(
    channels: &ChannelList,
    compressed: &[u8],
    rectangle: Box2i,
    out: &mut Vec<u8>,
    ac_scratch: &mut Vec<u16>,
    scratch: &mut DwaScratch,
) -> Result<()> {
    ensure_tables_initialized();

    let header = read_header(compressed)?;
    let version = header[Field::Version as usize];
    if version > 2 {
        return Err(ExrError::Input(format!("unsupported DWA version: {version}")));
    }

    // C++ `DwaCompressor_uncompress` (:783-789) rejects any size field whose
    // u64 is negative when read as int64 — i.e. the top bit set.  Doing this
    // first makes every `as usize` below lossless on a 64-bit target and
    // removes the u64::MAX / 2^63 wrap cases outright.
    for (i, &v) in header.iter().enumerate() {
        if i != Field::Version as usize && i != Field::AcCompression as usize && (v >> 63) != 0 {
            return Err(ExrError::Input("DWA: negative size field".into()));
        }
    }

    let unknown_uncompressed = header[Field::UnknownUncompressedSize as usize] as usize;
    let unknown_compressed = header[Field::UnknownCompressedSize as usize] as usize;
    let ac_compressed = header[Field::AcCompressedSize as usize] as usize;
    let dc_compressed = header[Field::DcCompressedSize as usize] as usize;
    let rle_compressed = header[Field::RleCompressedSize as usize] as usize;
    let rle_uncompressed = header[Field::RleUncompressedSize as usize] as usize;
    let rle_raw = header[Field::RleRawSize as usize] as usize;
    let ac_count = header[Field::AcUncompressedCount as usize] as usize;
    let dc_count = header[Field::DcUncompressedCount as usize] as usize;
    let ac_method = AcCompression::from_u64(header[Field::AcCompression as usize])?;

    // C++ :764-770 — both the sum and each individual size are checked, so a
    // wrapped sum cannot hide an over-long section.
    let total = ck_add(
        ck_add(unknown_compressed, ac_compressed)?,
        ck_add(dc_compressed, rle_compressed)?,
    )?;
    if compressed.len() < ck_add(HEADER_SIZE, total)?
        || compressed.len() < unknown_compressed
        || compressed.len() < ac_compressed
        || compressed.len() < dc_compressed
        || compressed.len() < rle_compressed
    {
        return Err(ExrError::Input("DWA data truncated".into()));
    }

    // --- Classify with the rule set this stream's version dictates ---------
    // C++ :791-805: legacy (lowercase) rules for v0/v1, the rules STORED IN THE
    // CHUNK for v2.  Never both, and never the default table for a v2 stream.
    let mut data_offset = HEADER_SIZE;
    let stream_rules: Vec<Rule>;
    let rules: &[Rule] = if version >= 2 {
        stream_rules = channel_rules::read_channel_rules(compressed, &mut data_offset)?;
        &stream_rules
    } else {
        classifier::legacy_rules()
    };

    // Cache-validated classification: a v2 stream's rules
    // are read fresh above every call, so reuse the previous `DwaChannelPlan`
    // only when `rules` compares equal to what built it -- see
    // `DwaScratch::plan`'s doc. Taken out of `scratch` (not borrowed) so the
    // rest of this function can freely take `&mut scratch.<other field>`
    // without fighting the borrow checker; put back just before returning.
    let plan = match scratch.plan.take() {
        Some(plan) if scratch.plan_rules == rules && scratch.plan_channels.iter().eq(channels.iter()) => plan,
        _ => {
            probe::hit(Probe::DwaPlanBuild);
            let plan = build_channel_plan(channels, rules)?;
            scratch.plan_rules = rules.to_vec();
            scratch.plan_channels = channels.clone();
            scratch.chan_dims_key = None; // channels/rules changed -> geometry cache is stale too
            plan
        }
    };
    let infos = &plan.infos;
    let csc_groups = &plan.csc_groups;
    let in_csc = &plan.in_csc;

    // --- Geometry-derived bounds for every stream-declared size ------------
    // C++ derives these from the chunk rectangle + channel classes in
    // `DwaCompressor_initializeBuffers` (:1268-1475) and compares each header
    // field against them before allocating or indexing.
    // An empty or inverted rectangle would make `as usize` wrap into a huge
    // extent, so every bound below would be vacuous -- reject it outright.
    if rectangle.max.x < rectangle.min.x || rectangle.max.y < rectangle.min.y {
        return Err(ExrError::Input("DWA: empty chunk rectangle".into()));
    }
    let width = (rectangle.max.x - rectangle.min.x + 1) as usize;
    let height = (rectangle.max.y - rectangle.min.y + 1) as usize;

    // `chan_dims`/`bytes_per_line`: the VALUES are a real
    // function of this chunk's own rectangle (a subsampled channel's extent
    // genuinely varies with chunk height, and the file's last chunk can be
    // shorter than every other), so they cannot be cached across chunks of
    // different shape -- but the common case (every chunk but the last shares
    // one shape) means the SAME allocation, and often the SAME values, are
    // reused chunk after chunk. `key_changed` gates both recomputations;
    // both Vecs are taken out of `scratch` (not borrowed) for the same
    // borrow-splitting reason as `plan` above, and restored at the end.
    let chan_dims_key = (width, rectangle.min.y, rectangle.max.y);
    let key_changed = scratch.chan_dims_key != Some(chan_dims_key);

    // Each channel's OWN `(width, height)` for this chunk -- reduces to
    // `(width, height)` when it isn't subsampled (see `channel_dims`'s doc).
    let mut chan_dims = std::mem::take(&mut scratch.chan_dims);
    if key_changed {
        chan_dims.clear();
        chan_dims.extend(
            infos
                .iter()
                .map(|c| channel_dims(width, rectangle.min.y, rectangle.max.y, c.x_sampling, c.y_sampling)),
        );
    }

    // `uncompressed_size` in C++ -- the unpacked chunk size (the packed
    // scanline buffer's total byte count, `misc::bytes_per_line_table`'s
    // per-row convention: a subsampled channel contributes bytes only on the
    // rows its `y_sampling` includes).
    let mut bytes_per_line = std::mem::take(&mut scratch.bytes_per_line);
    let uncompressed_size = if height > 0 {
        if key_changed {
            bytes_per_line.clear();
            crate::misc::bytes_per_line_table(rectangle.min.y, rectangle.max.y, width as i32, channels, &mut bytes_per_line)?;
        }
        bytes_per_line.iter().try_fold(0usize, |a, &b| ck_add(a, b))?
    } else {
        0
    };
    scratch.chan_dims_key = Some(chan_dims_key);
    let mut num_lossy = 0usize;
    let mut rle_raw_total = 0usize;
    let mut unknown_raw_total = 0usize;
    // Sum of AC/DC block counts across every LOSSY_DCT unit, each at ITS OWN
    // (possibly subsampled) block geometry — a CSC group counts its shared
    // geometry once for DC and 3x for AC (Y, Cb, Cr interleaved per block);
    // a standalone channel counts its own geometry once. Replaces the old
    // `blocks_per_channel * num_lossy` formula, which assumed every channel
    // shared the chunk's own `(width, height)` (true only when unsampled).
    let mut max_ac_count = 0usize;
    let mut max_dc_count = 0usize;
    for group in csc_groups {
        let (w, h) = chan_dims[group.r]; // uniform across a group by construction
        let nb = ck_mul(w.div_ceil(BLOCK_SIZE), h.div_ceil(BLOCK_SIZE))?;
        max_ac_count = ck_add(max_ac_count, ck_mul(ck_mul(nb, 63)?, 3)?)?;
        max_dc_count = ck_add(max_dc_count, ck_mul(nb, 3)?)?;
    }
    for (idx, info) in infos.iter().enumerate() {
        let (w, h) = chan_dims[idx];
        let chan_bytes = ck_mul(ck_mul(w, h)?, info.bytes_per_sample)?;
        match info.scheme {
            CompressionScheme::LossyDct => {
                num_lossy += 1;
                if !in_csc[idx] {
                    let nb = ck_mul(w.div_ceil(BLOCK_SIZE), h.div_ceil(BLOCK_SIZE))?;
                    max_ac_count = ck_add(max_ac_count, ck_mul(nb, 63)?)?;
                    max_dc_count = ck_add(max_dc_count, nb)?;
                }
            }
            // C++ :1334 — RLE "could double the size of the source data".
            CompressionScheme::Rle => rle_raw_total = ck_add(rle_raw_total, chan_bytes)?,
            CompressionScheme::Unknown => {
                unknown_raw_total = ck_add(unknown_raw_total, chan_bytes)?
            }
        }
    }
    let rle_buffer_max = ck_mul(rle_raw_total, 2)?;

    // C++ :774-779 — the two planar streams cannot exceed the unpacked chunk.
    if unknown_uncompressed > uncompressed_size
        || rle_raw > uncompressed_size
        || ck_add(unknown_uncompressed, rle_raw)? > uncompressed_size
    {
        return Err(ExrError::Input("DWA: planar size exceeds chunk".into()));
    }
    // C++ :876 (`> _planarUncBufferSize[UNKNOWN]`).  Our bound is the exact
    // planar byte count the encoder emits, which is what the reconstruct step
    // below consumes; C++'s buffer is that padded by `exr_compress_max_buffer_size`.
    if unknown_uncompressed > unknown_raw_total {
        return Err(ExrError::Input("DWA: UNKNOWN size exceeds chunk geometry".into()));
    }
    // C++ :903 / :951 — AC / DC counts against the geometry-sized buffers.
    if ac_count > max_ac_count {
        return Err(ExrError::Input("DWA: AC count exceeds chunk geometry".into()));
    }
    if dc_count > max_dc_count {
        return Err(ExrError::Input("DWA: DC count exceeds chunk geometry".into()));
    }
    // C++ :986 — both RLE sizes against `_rleBufferSize` / `_planarUncBufferSize[RLE]`.
    if rle_uncompressed > rle_buffer_max || rle_raw > rle_buffer_max {
        return Err(ExrError::Input("DWA: RLE size exceeds chunk geometry".into()));
    }
    // C++ :977 — a zero-length DC stream must declare a zero count.
    if dc_compressed == 0 && dc_count != 0 {
        return Err(ExrError::Input("DWA: DC count without DC data".into()));
    }

    // `validate_size` (`internal_dwa_compressor.h:691-726`, called at
    // `:907`/`:910`), absent from our v3.3.11 port base but present in the 3.4.15
    // oracle. If a scheme's channels need planar bytes, the stream must actually
    // carry a non-empty compressed section that covers them; if none of the
    // scheme's channels exist, EVERY size the header declares for it must be zero.
    // Without this, an UNKNOWN/RLE channel with `compressed==0` (or a too-short
    // `uncompressed`/`raw`) silently zero-filled instead of being a corrupt chunk —
    // and a stream that declares sizes for a scheme with NO channels of that kind
    // was silently accepted instead of rejected.
    validate_dwa_size(unknown_compressed, unknown_uncompressed, 0, unknown_raw_total)?;
    validate_dwa_size(rle_compressed, rle_raw, rle_uncompressed, rle_raw_total)?;

    // --- Section offsets (C++ :846-856 pointer range checks) ---------------
    let unknown_start = data_offset;
    let ac_start = ck_add(unknown_start, unknown_compressed)?;
    let dc_start = ck_add(ac_start, ac_compressed)?;
    let rle_start = ck_add(dc_start, dc_compressed)?;
    let rle_end = ck_add(rle_start, rle_compressed)?;
    if rle_end > compressed.len() {
        return Err(ExrError::Input("DWA data truncated (sections)".into()));
    }

    // --- Unknown (zlib) --- the one sub-stream where C tolerates a short
    // decode (NULL `actual_out`, see `zlib_decompress_unknown`'s doc).
    scratch.unknown.clear();
    if unknown_compressed > 0 {
        zlib_decompress_unknown(
            &compressed[unknown_start..ac_start],
            unknown_uncompressed,
            &mut scratch.unknown,
            &mut scratch.decompressor,
        )?;
    }
    let unknown_data = &scratch.unknown;

    // --- AC coefficients ---
    // `ac_scratch` is `DwaCompressor`'s persistent AC-decode buffer (mirrors
    // `PizCompressor`'s `bitmap`/`tmp`/`chan_line` fields): filled in place
    // instead of a fresh `Vec` per chunk, matching C's own reused
    // `decode->scratch_buffer_1` scheme.
    ac_scratch.clear();
    if ac_compressed > 0 && ac_count > 0 {
        // C++ :899 — with no LOSSY_DCT channel there is no AC buffer at all.
        if num_lossy == 0 {
            return Err(ExrError::Input("DWA: AC data without a DCT channel".into()));
        }
        match ac_method {
            AcCompression::StaticHuffman => {
                huf::decompress(&compressed[ac_start..dc_start], ac_count, ac_scratch, &mut scratch.fast_huf)?;
            }
            AcCompression::Deflate => {
                let bytes = zlib_decompress_exact(
                    &compressed[ac_start..dc_start],
                    ck_mul(ac_count, 2)?,
                    &mut scratch.decompressor,
                )?;
                ac_scratch.resize(ac_count, 0);
                crate::misc::le_bytes_to_u16s(&bytes, ac_scratch)?;
            }
        }
    }

    // --- DC coefficients (zlib + reconstruct) ---
    scratch.dc.clear();
    if dc_compressed > 0 && dc_count > 0 {
        let mut bytes = zlib_decompress_exact(
            &compressed[dc_start..rle_start],
            ck_mul(dc_count, 2)?,
            &mut scratch.decompressor,
        )?;
        // internal_zip_reconstruct_bytes == unpredict + interleave.
        let mut dc_bytes_scratch = std::mem::take(&mut bytes);
        let n = dc_bytes_scratch.len();
        super::ensure_len(&mut scratch.dc_reorder, n);
        unpredict_and_reorder(&mut scratch.dc_reorder, &mut dc_bytes_scratch);
        scratch.dc.resize(dc_count, 0);
        crate::misc::le_bytes_to_u16s(&scratch.dc_reorder, &mut scratch.dc)?;
    }
    let dc_data = &scratch.dc;

    // --- RLE channel data (zlib then byte-RLE) ---
    // C++ :982-1013 gates on `rleRawSize > 0` and requires BOTH the zlib output
    // and the un-RLE'd output to match the declared sizes exactly.
    scratch.rle.clear();
    if rle_raw > 0 {
        let deflated = zlib_decompress_exact(&compressed[rle_start..rle_end], rle_uncompressed, &mut scratch.decompressor)?;
        rle_decompress(&deflated, &mut scratch.rle)?;
        if scratch.rle.len() != rle_raw {
            return Err(ExrError::Input("DWA: RLE raw size mismatch".into()));
        }
    }
    let rle_data = &scratch.rle;

    // --- Reconstruct pixels ---
    // Decoded DCT channels indexed by channel position, `dct_filled[idx]`
    // false until decoded this chunk (rather than a fresh
    // `Vec<Option<Vec<u16>>>` -- with a fresh `Vec<u16>` inside each `Some`
    // -- every call, `scratch.dct_outputs`/`dct_filled` are grow-only and
    // reused). The channel count is fixed by the header for the
    // compressor's whole life, so after the first call this is a no-op
    // resize plus an `infos.len()`-sized `fill(false)` -- no allocation.
    if scratch.dct_outputs.len() < infos.len() {
        scratch.dct_outputs.resize_with(infos.len(), Vec::new);
    }
    if scratch.dct_filled.len() < infos.len() {
        scratch.dct_filled.resize(infos.len(), false);
    }
    scratch.dct_filled[..infos.len()].fill(false);

    // The per-channel decoders consume `ac_iter` (the shared AC-token stream)
    // DIRECTLY, in block raster order — no per-channel collect/re-encode buffer
    // and no hashing. Each group/channel gets ITS OWN decoder at its own
    // (possibly subsampled) `(width, height)`.
    let mut dc_offset = 0usize;
    let mut ac_iter = ac_scratch.iter().copied();

    // CSC groups first (R,G,B order, AC interleaved per block).
    for group in csc_groups {
        // C++ :1073-1079 — every member of a group must really be LOSSY_DCT.
        if infos[group.r].scheme != CompressionScheme::LossyDct
            || infos[group.g].scheme != CompressionScheme::LossyDct
            || infos[group.b].scheme != CompressionScheme::LossyDct
        {
            return Err(ExrError::Input("DWA: CSC group member is not LOSSY_DCT".into()));
        }
        let (gw, gh) = chan_dims[group.r]; // uniform across the group by construction
        let nb = ck_mul(gw.div_ceil(BLOCK_SIZE), gh.div_ceil(BLOCK_SIZE))?;
        let dc_end = ck_add(dc_offset, ck_mul(3, nb)?)?;
        if dc_end > dc_data.len() {
            return Err(ExrError::Input("DWA: insufficient DC for CSC group".into()));
        }
        let group_dc = &dc_data[dc_offset..dc_end];
        dc_offset = dc_end;

        // C++ ALWAYS builds a CSC decoder with the LUT (`LossyDctDecoderCsc_construct`,
        // `internal_dwa_compressor.h:1113-1126`, unconditional) — `p_linear` is honoured
        // only for single (non-CSC) LOSSY_DCT channels (`:1181`). An R channel with
        // `pLinear=1` inside an RGB CSC group must still be perceptually decoded.
        // Pulls 3 * nb blocks from the shared AC stream (Y,Cb,Cr interleaved per
        // block) and advances `ac_iter` to the next group/channel.
        //
        // `decode_rgb_into` needs three INDEPENDENT `&mut Vec<u16>` borrows out
        // of the same `scratch.dct_outputs` Vec, which plain indexing can't
        // give the borrow checker -- `mem::take` each slot
        // into a local (leaving an empty placeholder), decode into the
        // locals (preserving their existing capacity from a prior chunk),
        // then move them back. On an early `?` return the three slots stay
        // emptied (capacity lost, not correctness) -- decompress_dwa itself
        // is failing that call anyway, so this is a harmless perf-only edge
        // case, not a soundness one.
        let mut r_buf = std::mem::take(&mut scratch.dct_outputs[group.r]);
        let mut g_buf = std::mem::take(&mut scratch.dct_outputs[group.g]);
        let mut b_buf = std::mem::take(&mut scratch.dct_outputs[group.b]);
        LossyDctDecoder::new(gw, gh)
            .decode_rgb_into(
                group_dc,
                &mut ac_iter,
                true,
                &mut r_buf,
                &mut g_buf,
                &mut b_buf,
                &mut scratch.row_blocks_rgb,
            )
            .map_err(|e| ExrError::Input(format!("DWA CSC decode: {e}")))?;
        scratch.dct_outputs[group.r] = r_buf;
        scratch.dct_outputs[group.g] = g_buf;
        scratch.dct_outputs[group.b] = b_buf;
        scratch.dct_filled[group.r] = true;
        scratch.dct_filled[group.g] = true;
        scratch.dct_filled[group.b] = true;
    }

    // Remaining single DCT channels.
    for (idx, info) in infos.iter().enumerate() {
        if info.scheme != CompressionScheme::LossyDct || in_csc[idx] {
            continue;
        }
        let (cw, ch_) = chan_dims[idx];
        let nb = ck_mul(cw.div_ceil(BLOCK_SIZE), ch_.div_ceil(BLOCK_SIZE))?;
        let dc_end = ck_add(dc_offset, nb)?;
        if dc_end > dc_data.len() {
            return Err(ExrError::Input("DWA: insufficient DC".into()));
        }
        let chan_dc = &dc_data[dc_offset..dc_end];
        dc_offset = dc_end;

        let use_linear = !info.quantize_linearly;
        // Pulls `nb` blocks from the shared AC stream, in raster order.
        // `scratch.dct_outputs[idx]` and `scratch.row_blocks` are disjoint
        // fields of `scratch`, so a plain single-index borrow is fine here
        // (unlike the CSC case above, which needs 3 slots of the SAME Vec
        // at once).
        LossyDctDecoder::new(cw, ch_)
            .decode_channel_into(
                chan_dc,
                &mut ac_iter,
                use_linear,
                &mut scratch.dct_outputs[idx],
                &mut scratch.row_blocks,
            )
            .map_err(|e| ExrError::Input(format!("DWA decode: {e}")))?;
        scratch.dct_filled[idx] = true;
    }

    // --- Assemble the scanline-interleaved output buffer ---
    // Row-major over the OUTPUT: for each absolute `y`, place every channel
    // that `y_sampling` includes, in channel-list order, with no gap for a
    // channel `y` excludes (`bytes_per_line_table`'s convention — this is
    // what makes `bytes_per_line[row]` vary by row when any channel is
    // y-subsampled). `local_row[idx]` counts how many times channel `idx` has
    // been active so far — its index into that channel's OWN (subsampled)
    // raster, matching the encoder's row-push order (`internal_dwa_compressor.h:347-361`).
    out.clear();
    out.resize(uncompressed_size, 0);
    let mut local_row = vec![0usize; infos.len()];
    let mut out_off = 0usize;

    // Per-channel start offsets within the shared `rle_data`/`unknown_data`
    // planar buffers — each channel's own section is `chan_dims[idx]` sized,
    // in file-channel order (mirrors the encoder's concatenation order).
    let mut rle_chan_start = vec![0usize; infos.len()];
    let mut unknown_chan_start = vec![0usize; infos.len()];
    {
        let mut rle_acc = 0usize;
        let mut unknown_acc = 0usize;
        for (idx, info) in infos.iter().enumerate() {
            let (w, h) = chan_dims[idx];
            match info.scheme {
                CompressionScheme::Rle => {
                    rle_chan_start[idx] = rle_acc;
                    rle_acc += w * h * info.bytes_per_sample;
                }
                CompressionScheme::Unknown => {
                    unknown_chan_start[idx] = unknown_acc;
                    unknown_acc += w * h * info.bytes_per_sample;
                }
                CompressionScheme::LossyDct => {}
            }
        }
    }

    for y in rectangle.min.y..=rectangle.max.y {
        for (idx, info) in infos.iter().enumerate() {
            if modp(y, info.y_sampling) != 0 {
                continue;
            }
            let (cw, _) = chan_dims[idx];
            let lr = local_row[idx];
            local_row[idx] += 1;
            let bps = info.bytes_per_sample;

            match info.scheme {
                CompressionScheme::LossyDct => {
                    if !scratch.dct_filled[idx] {
                        return Err(ExrError::Input("DWA: missing decoded DCT channel".into()));
                    }
                    let decoded = &scratch.dct_outputs[idx];
                    for x in 0..cw {
                        let v = decoded[lr * cw + x];
                        if info.type_ == PixelType::Float {
                            let f = Half::from_bits(v).to_f32();
                            out[out_off..out_off + 4].copy_from_slice(&f.to_le_bytes());
                            out_off += 4;
                        } else {
                            out[out_off..out_off + 2].copy_from_slice(&v.to_le_bytes());
                            out_off += 2;
                        }
                    }
                }
                CompressionScheme::Rle => {
                    // RLE channels are stored planar within their own section:
                    // all byte0, then all byte1, ... over that channel's own pixels.
                    let chan_pixels = cw * chan_dims[idx].1;
                    for x in 0..cw {
                        let pixel_idx = lr * cw + x;
                        for byte in 0..bps {
                            let rp = rle_chan_start[idx] + byte * chan_pixels + pixel_idx;
                            out[out_off] = rle_data.get(rp).copied().unwrap_or(0);
                            out_off += 1;
                        }
                    }
                }
                CompressionScheme::Unknown => {
                    let line_bytes = cw * bps;
                    let src = unknown_chan_start[idx] + lr * line_bytes;
                    let end = (src + line_bytes).min(unknown_data.len());
                    if src < unknown_data.len() {
                        let n = end - src;
                        out[out_off..out_off + n].copy_from_slice(&unknown_data[src..end]);
                    }
                    out_off += line_bytes;
                }
            }
        }
    }
    // Return the taken-out plan/geometry scratch to
    // `scratch` for the next chunk to (maybe) reuse -- `infos`/`csc_groups`/
    // `in_csc` (borrows of `plan`) and `chan_dims`/`bytes_per_line` (owned
    // locals) have no remaining uses past this point.
    scratch.plan = Some(plan);
    scratch.chan_dims = chan_dims;
    scratch.bytes_per_line = bytes_per_line;

    Ok(())
}

/// Compress the AC coefficient stream — DEFLATE for tiled DWAA
/// (`use_deflate_for_ac`), else static Huffman.
///
/// Returns `Ok(None)` when static Huffman can't encode `all_ac` at all — C++'s
/// `internal_huf_compress` returning `EXR_ERR_ARGUMENT_OUT_OF_RANGE`
/// (`internal_dwa_compressor.h:551-566`), which happens when the entropy-coded
/// bit count would not fit the on-wire `uint32_t`. **C++ does not treat this as
/// an error**: it abandons the whole DWA-encoded chunk and copies the RAW
/// packed line buffer as the "compressed" result, returning success — which is
/// exactly our own "chunk didn't shrink, store raw" convention (the writer
/// seam's `stored_chunk`), just reached from inside the codec instead of via
/// the size comparison `compress_dwa` does at the end. The caller must
/// propagate `None` the same way (`return Ok(None)`), not turn it into an
/// error.
///
/// `huf_limit` is normally [`u32::MAX`] (as `u64`, matching C++'s
/// `UINT32_MAX` bit-count ceiling); tests inject a tiny limit via
/// [`huf::compress_with_limit`] to exercise this path without actually
/// building a multi-gigabit AC stream.
fn compress_ac(
    all_ac: &[u16],
    use_deflate_for_ac: bool,
    huf_limit: u64,
    level9: &mut LowCompressor,
    huf_scratch: &mut huf::HufEncScratch,
    ac_deflate_bytes: &mut Vec<u8>,
) -> Result<Option<(Vec<u8>, AcCompression)>> {
    if all_ac.is_empty() {
        return Ok(Some((Vec::new(), AcCompression::StaticHuffman)));
    }
    if use_deflate_for_ac {
        // `ac_deflate_bytes` is a persistent, reused
        // buffer -- `u16s_to_le_bytes` fully overwrites every byte.
        super::ensure_len(ac_deflate_bytes, all_ac.len() * 2);
        crate::misc::u16s_to_le_bytes(all_ac, ac_deflate_bytes)?;
        return Ok(Some((zlib_compress(ac_deflate_bytes, level9)?, AcCompression::Deflate)));
    }
    // `huf_scratch` is the persistent ~2 MiB encode-table
    // scratch; the compressed bytes themselves land in a plain local `Vec`
    // (NOT a persistent field) since this function returns them by value one
    // level up to `compress_dwa`'s own header/assembly logic, exactly like
    // its sibling `dc_compressed`/`rle_compressed`/`unknown_compressed`
    // results (each a fresh `Vec` returned by `zlib_compress` per call).
    let mut huf_out = Vec::new();
    if huf::compress_with_limit(all_ac, huf_scratch, huf_limit, &mut huf_out)? {
        Ok(Some((huf_out, AcCompression::StaticHuffman)))
    } else {
        Ok(None)
    }
}

/// Persistent low-level zlib compressor state for DWA's compress-side zlib
/// call sites (perf follow-up, 2026-09-24 — see `compressor::zlib`'s module
/// doc). AC-deflate/RLE/Unknown all compress at the fixed
/// [`DWA_ZLIB_LEVEL_9`]; DC compresses at the header's own `dc_zip_level`,
/// which may legitimately differ from 9 — two separate persistent
/// compressors, not one, since `deflate_rs::deflate_compress::Compressor`
/// bakes its level in at construction (no `set_level`).
struct DwaDeflateCompressors {
    level9: LowCompressor,
    dc: LowCompressor,
}

impl DwaDeflateCompressors {
    fn new(dc_zip_level: i32) -> Result<Self> {
        Ok(Self {
            level9: new_deflate_compressor(DWA_ZLIB_LEVEL_9)?,
            dc: new_deflate_compressor(dc_zip_level)?,
        })
    }
}

// ===========================================================================
// compress_dwa — port of DwaCompressor_compress
// ===========================================================================

/// Persistent, grow-only encode-side scratch for
/// `compress_dwa` -- mirrors `DwaScratch`'s "filled in place, not reallocated
/// per chunk" contract on the decode side. Every `Vec<Vec<_>>` here is
/// preallocated to the exact outer length `plan.num_dct`/`num_rle`/
/// `num_unknown` DICTATE (invariant for a `DwaCompressor`'s whole lifetime,
/// since its channel list never changes after construction -- see
/// `DwaChannelPlan`'s doc) by [`DwaCompressScratch::resize_for_plan`], then
/// each inner `Vec` is `.clear()`'d (not reallocated) at the start of every
/// chunk, so capacity only ever grows to the chunk's own high-water mark.
#[derive(Default)]
struct DwaCompressScratch {
    /// Per-DCT-channel native-half samples, row-major (`plan.dct_to_info`
    /// order) -- was `dct_buf`+`dct_channels`' two-tier
    /// extract-then-`mem::take`-compact pass; now filled directly via
    /// `plan.dct_slot`.
    dct_channels: Vec<Vec<u16>>,
    /// Per-RLE-channel bytes, per-pixel-interleaved (pre-reorder) -- was
    /// `rle_interleaved`.
    rle_interleaved: Vec<Vec<u8>>,
    /// Per-RLE-channel bytes, planar (post-reorder: all byte0, then byte1,
    /// ...) -- was `rle_channels`.
    rle_channels: Vec<Vec<u8>>,
    /// Per-Unknown-channel scanline bytes -- was `unknown_buf`+`unknown_channels`.
    unknown_channels: Vec<Vec<u8>>,
    /// Concatenated DC / AC coefficient streams for the whole chunk (CSC
    /// groups first, then singles) -- was `all_dc`/`all_ac`.
    all_dc: Vec<u16>,
    all_ac: Vec<u16>,
    /// Per-CSC-group staging for `LossyDctEncoder::encode_rgb_into` -- Y/Cb/Cr
    /// DC must be gathered plane-by-plane (the encoder computes them
    /// interleaved per block) before appending Y,Cb,Cr in order to `all_dc`;
    /// these three replace `encode_rgb`'s own per-call `Vec::with_capacity`.
    dc_y: Vec<u16>,
    dc_cb: Vec<u16>,
    dc_cr: Vec<u16>,
    /// Concatenated raw RLE / Unknown byte streams -- was `rle_raw_data`/
    /// `unknown_raw_data`.
    rle_raw: Vec<u8>,
    unknown_raw: Vec<u8>,
    /// Same grow-only-allocation / recompute-on-key-change policy as
    /// `DwaScratch::chan_dims` -- see its doc. Encode-side geometry can ALSO
    /// change chunk-to-chunk under tiled DWAA (`compress_tile` moves
    /// `min_x`/`max_x`/`max_y`, so `width` is not fixed the way `channels` is).
    chan_dims: Vec<(usize, usize)>,
    chan_dims_key: Option<(usize, i32, i32)>,
}

impl DwaCompressScratch {
    /// Ensure the outer per-scheme `Vec`s have exactly `plan`'s channel
    /// counts. A no-op once the counts stop changing (true for the whole
    /// life of one `DwaChannelPlan`, i.e. the whole life of one
    /// `DwaCompressor` on the encode side), so this "grows" at most once.
    fn resize_for_plan(&mut self, plan: &DwaChannelPlan) {
        if self.dct_channels.len() != plan.num_dct {
            self.dct_channels.resize_with(plan.num_dct, Vec::new);
        }
        if self.rle_interleaved.len() != plan.num_rle {
            self.rle_interleaved.resize_with(plan.num_rle, Vec::new);
        }
        if self.rle_channels.len() != plan.num_rle {
            self.rle_channels.resize_with(plan.num_rle, Vec::new);
        }
        if self.unknown_channels.len() != plan.num_unknown {
            self.unknown_channels.resize_with(plan.num_unknown, Vec::new);
        }
    }
}

/// Compress the native-LE scanline-interleaved line buffer with DWA into
/// `out` (the same out/scratch policy every codec's compress
/// AND uncompress path follows). `use_deflate_for_ac` selects DEFLATE (tiled
/// DWAA) vs static Huffman (scanline / DWAB) for the AC coefficients.
/// Returns whether the assembled chunk actually SHRANK against `data`
/// ("no-shrink -> store raw"): `true` means `out` holds the DWA chunk; `false`
/// means `out` was cleared and the caller must store `data` raw instead.
#[allow(clippy::too_many_arguments)]
fn compress_dwa(
    channels: &ChannelList,
    data: &[u8],
    rectangle: Box2i,
    level: f32,
    use_deflate_for_ac: bool,
    plan: &DwaChannelPlan,
    deflate: &mut DwaDeflateCompressors,
    huf_scratch: &mut huf::HufEncScratch,
    ac_deflate_bytes: &mut Vec<u8>,
    dc_encode_bytes: &mut Vec<u8>,
    dc_encode_reorder: &mut Vec<u8>,
    scratch: &mut DwaCompressScratch,
    out: &mut Vec<u8>,
) -> Result<bool> {
    ensure_tables_initialized();

    let width = (rectangle.max.x - rectangle.min.x + 1) as usize;
    // C++ `DwaCompressor_compress` (:256-258) installs `sDefaultChannelRules`
    // and NOTHING else; the legacy lowercase table is a decode-only fallback for
    // v0/v1 streams.  Classifying `red/green/blue` or `r/g/b/a` as DCT on encode
    // produced chunks real OpenEXR rejects with EXR_ERR_CORRUPT_CHUNK.
    // `plan` is `classify_channels`'s output, computed ONCE
    // in `DwaCompressor::new` -- see `DwaChannelPlan`'s doc for why this is
    // always valid here (encode always uses `classifier::default_rules()`
    // against a channel list that never changes after construction).
    let infos = &plan.infos;
    let csc_groups = &plan.csc_groups;
    let in_csc = &plan.in_csc;

    scratch.resize_for_plan(plan);

    // Same grow-only-allocation / recompute-on-key-change
    // policy as `DwaScratch::chan_dims` on the decode side -- see its doc.
    // Under tiled DWAA, `compress_tile` moves `min_x`/`max_x`/`max_y` per
    // tile, so (unlike `plan`) this genuinely can change chunk to chunk.
    let chan_dims_key = (width, rectangle.min.y, rectangle.max.y);
    if scratch.chan_dims_key != Some(chan_dims_key) {
        scratch.chan_dims.clear();
        scratch.chan_dims.extend(
            infos
                .iter()
                .map(|c| channel_dims(width, rectangle.min.y, rectangle.max.y, c.x_sampling, c.y_sampling)),
        );
        scratch.chan_dims_key = Some(chan_dims_key);
    }
    let chan_dims = &scratch.chan_dims;

    // --- Extract per-channel data from the scanline-interleaved buffer ---
    // Row-major over the INPUT: for each absolute `y`, every channel that
    // `y_sampling` includes contributes its `cw` samples, in channel-list
    // order, with no gap for a channel `y` excludes -- the same convention
    // `misc::bytes_per_line_table` uses to lay the buffer out. RLE
    // bytes are collected per-pixel-interleaved here and reordered into the
    // planar (all byte0, then byte1, ...) layout afterward, once, from each
    // channel's own (possibly subsampled) pixel count.
    //
    // Written DIRECTLY into `scratch`'s per-scheme slots
    // (`plan.dct_slot`/`rle_slot`/`unknown_slot`) instead of the old
    // per-ALL-channels staging arrays (`dct_buf`/`rle_interleaved`/
    // `unknown_buf`) that were then `mem::take`-compacted into a second,
    // freshly-allocated set of dense arrays every chunk.
    for c in scratch.dct_channels.iter_mut() {
        c.clear();
    }
    for c in scratch.rle_interleaved.iter_mut() {
        c.clear();
    }
    for c in scratch.unknown_channels.iter_mut() {
        c.clear();
    }
    let mut pos = 0usize;

    for y in rectangle.min.y..=rectangle.max.y {
        for (idx, info) in infos.iter().enumerate() {
            if modp(y, info.y_sampling) != 0 {
                continue;
            }
            let (cw, _) = chan_dims[idx];
            let bps = info.bytes_per_sample;

            match info.scheme {
                CompressionScheme::LossyDct => {
                    let buf = &mut scratch.dct_channels[plan.dct_slot[idx]];
                    for x in 0..cw {
                        let p = pos + x * bps;
                        let v = if info.type_ == PixelType::Float {
                            if p + 4 <= data.len() {
                                let f = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                                // Clamp to the half range before conversion to
                                // avoid producing +-Inf (C++ encoder lines ~1019).
                                let f = f.clamp(-65504.0_f32, 65504.0_f32);
                                Half::from_f32(f).bits()
                            } else {
                                0
                            }
                        } else if p + 2 <= data.len() {
                            u16::from_le_bytes([data[p], data[p + 1]])
                        } else {
                            0
                        };
                        buf.push(v);
                    }
                    pos += cw * bps;
                }
                CompressionScheme::Rle => {
                    let buf = &mut scratch.rle_interleaved[plan.rle_slot[idx]];
                    for x in 0..cw {
                        let p = pos + x * bps;
                        for b in 0..bps {
                            buf.push(data.get(p + b).copied().unwrap_or(0));
                        }
                    }
                    pos += cw * bps;
                }
                CompressionScheme::Unknown => {
                    let buf = &mut scratch.unknown_channels[plan.unknown_slot[idx]];
                    let line_bytes = cw * bps;
                    let end = (pos + line_bytes).min(data.len());
                    if pos < data.len() {
                        buf.extend_from_slice(&data[pos..end]);
                        if end - pos < line_bytes {
                            let new_len = buf.len() + line_bytes - (end - pos);
                            buf.resize(new_len, 0);
                        }
                    } else {
                        let new_len = buf.len() + line_bytes;
                        buf.resize(new_len, 0);
                    }
                    pos += line_bytes;
                }
            }
        }
    }

    // Reorder each RLE channel from per-pixel-interleaved to planar (all
    // byte0, then byte1, ...) using its OWN pixel count -- was a fresh
    // `Vec::with_capacity` per channel per chunk (`rle_channels.push(chan)`);
    // now written into the matching persistent `scratch.rle_channels` slot.
    for (idx, info) in infos.iter().enumerate() {
        if info.scheme != CompressionScheme::Rle {
            continue;
        }
        let slot = plan.rle_slot[idx];
        let bps = info.bytes_per_sample.max(1);
        let pixels = scratch.rle_interleaved[slot].len() / bps;
        scratch.rle_channels[slot].clear();
        scratch.rle_channels[slot].reserve(scratch.rle_interleaved[slot].len());
        for byte in 0..bps {
            for p in 0..pixels {
                let v = scratch.rle_interleaved[slot][p * bps + byte];
                scratch.rle_channels[slot].push(v);
            }
        }
    }

    // --- Encode DCT channels (CSC groups first, then singles) ---
    // Each group/channel gets ITS OWN encoder at its own (possibly subsampled)
    // `(width, height)` -- a shared chunk-wide encoder was only correct
    // when every channel shared the chunk's own dimensions (unsampled).
    //
    // `info_to_dct` is `plan`'s (built once, not rebuilt
    // per chunk); `all_dc`/`all_ac` are `scratch`'s persistent, grow-only
    // buffers, cleared (not reallocated) below, and the encoder writes
    // straight into them via `encode_channel_into`/`encode_rgb_into` instead
    // of returning a fresh `(Vec<u16>, Vec<u16>)` per channel/group.
    scratch.all_dc.clear();
    scratch.all_ac.clear();

    for group in csc_groups {
        let (Some(&r_dct), Some(&g_dct), Some(&b_dct)) = (
            plan.info_to_dct.get(&group.r),
            plan.info_to_dct.get(&group.g),
            plan.info_to_dct.get(&group.b),
        ) else {
            // Unreachable for the default rules (a CSC slot implies LOSSY_DCT),
            // but never index-panic on a classification we did not hand-write.
            return Err(ExrError::Logic("DWA: CSC group member is not a DCT channel".into()));
        };
        let (gw, gh) = chan_dims[group.r]; // uniform across the group by construction
        // C++ ALWAYS builds a CSC encoder with the LUT (`LossyDctEncoderCsc_construct`,
        // `internal_dwa_compressor.h:370-381`, unconditional) -- `p_linear` is honoured
        // only for single (non-CSC) LOSSY_DCT channels (`:420`). Mirrors the decode-side
        // fix above: an R channel with `pLinear=1` inside an RGB CSC group is
        // still perceptually encoded.
        LossyDctEncoder::new(gw, gh, Some(level)).encode_rgb_into(
            &scratch.dct_channels[r_dct],
            &scratch.dct_channels[g_dct],
            &scratch.dct_channels[b_dct],
            true,
            &mut scratch.dc_y,
            &mut scratch.dc_cb,
            &mut scratch.dc_cr,
            &mut scratch.all_ac,
        );
        scratch.all_dc.extend_from_slice(&scratch.dc_y);
        scratch.all_dc.extend_from_slice(&scratch.dc_cb);
        scratch.all_dc.extend_from_slice(&scratch.dc_cr);
    }

    for (dct_idx, &info_idx) in plan.dct_to_info.iter().enumerate() {
        if in_csc[info_idx] {
            continue;
        }
        let (cw, ch_) = chan_dims[info_idx];
        let use_nonlinear = !infos[info_idx].quantize_linearly;
        LossyDctEncoder::new(cw, ch_, Some(level)).encode_channel_into(
            &scratch.dct_channels[dct_idx],
            use_nonlinear,
            &mut scratch.all_dc,
            &mut scratch.all_ac,
        );
    }

    // --- RLE + Unknown raw byte streams ---
    // `scratch.rle_raw`/`scratch.unknown_raw` are
    // persistent, grow-only -- not fresh `Vec::new()`s per chunk.
    scratch.rle_raw.clear();
    for slot in 0..plan.num_rle {
        let bytes = std::mem::take(&mut scratch.rle_channels[slot]);
        scratch.rle_raw.extend_from_slice(&bytes);
        scratch.rle_channels[slot] = bytes;
    }
    scratch.unknown_raw.clear();
    for slot in 0..plan.num_unknown {
        let bytes = std::mem::take(&mut scratch.unknown_channels[slot]);
        scratch.unknown_raw.extend_from_slice(&bytes);
        scratch.unknown_channels[slot] = bytes;
    }

    // --- Compress AC (Huffman or DEFLATE level 9) ---
    let (ac_compressed, ac_method) = match compress_ac(
        &scratch.all_ac, use_deflate_for_ac, u32::MAX as u64, &mut deflate.level9, huf_scratch, ac_deflate_bytes,
    )? {
        Some(pair) => pair,
        // The AC Huffman stream exceeded the encodable bit-count range.
        // C++ does not error here -- see `compress_ac`'s doc -- it abandons the whole
        // DWA-encoded chunk, which for us means "cannot shrink this chunk, store raw".
        None => {
            out.clear();
            return Ok(false);
        }
    };

    // --- Compress DC (reorder+predict, then zlib at the DC level) ---
    let dc_compressed = if !scratch.all_dc.is_empty() {
        // `dc_encode_bytes`/`dc_encode_reorder` are
        // persistent, reused buffers -- `u16s_to_le_bytes` and
        // `reorder_and_predict` each fully overwrite every byte (the same
        // shared ZIP/RLE predictor contract that makes `ensure_len` safe
        // elsewhere).
        super::ensure_len(dc_encode_bytes, scratch.all_dc.len() * 2);
        crate::misc::u16s_to_le_bytes(&scratch.all_dc, dc_encode_bytes)?;
        super::ensure_len(dc_encode_reorder, dc_encode_bytes.len());
        reorder_and_predict(dc_encode_reorder, dc_encode_bytes);
        zlib_compress(dc_encode_reorder, &mut deflate.dc)?
    } else {
        Vec::new()
    };

    // --- Compress RLE (byte-RLE, then zlib level 9) ---
    let (rle_compressed, rle_uncompressed_size) = if !scratch.rle_raw.is_empty() {
        let mut encoded = Vec::new();
        rle_compress(&scratch.rle_raw, &mut encoded);
        let uncompressed = encoded.len();
        (zlib_compress(&encoded, &mut deflate.level9)?, uncompressed)
    } else {
        (Vec::new(), 0)
    };

    // --- Compress Unknown (zlib level 9) ---
    let unknown_compressed = if !scratch.unknown_raw.is_empty() {
        zlib_compress(&scratch.unknown_raw, &mut deflate.level9)?
    } else {
        Vec::new()
    };

    // --- Header ---
    let mut header = [0u64; NUM_HEADER_FIELDS];
    header[Field::Version as usize] = 2;
    header[Field::UnknownUncompressedSize as usize] = scratch.unknown_raw.len() as u64;
    header[Field::UnknownCompressedSize as usize] = unknown_compressed.len() as u64;
    header[Field::AcCompressedSize as usize] = ac_compressed.len() as u64;
    header[Field::DcCompressedSize as usize] = dc_compressed.len() as u64;
    header[Field::RleCompressedSize as usize] = rle_compressed.len() as u64;
    header[Field::RleUncompressedSize as usize] = rle_uncompressed_size as u64;
    header[Field::RleRawSize as usize] = scratch.rle_raw.len() as u64;
    header[Field::AcUncompressedCount as usize] = scratch.all_ac.len() as u64;
    header[Field::DcUncompressedCount as usize] = scratch.all_dc.len() as u64;
    header[Field::AcCompression as usize] = ac_method as u64;

    out.clear();
    write_header(&header, out);
    channel_rules::write_for_channels(channels, out)?;
    out.extend_from_slice(&unknown_compressed);
    out.extend_from_slice(&ac_compressed);
    out.extend_from_slice(&dc_compressed);
    out.extend_from_slice(&rle_compressed);

    // C++ "no-shrink -> store raw": if the assembled output is not smaller than
    // the raw line buffer, store raw (the seam detects this by size).
    if out.len() >= data.len() {
        out.clear();
        Ok(false)
    } else {
        Ok(true)
    }
}

// ===========================================================================
// DwaCompressor — the Compressor seam implementation
// ===========================================================================

/// DWAA / DWAB scanline codec — port of `Imf::DwaCompressor`.
///
/// `num_scan_lines()` is **32** (DWAA) or **256** (DWAB); both run the identical
/// pipeline.  `use_deflate_for_ac` mirrors the C++ tiled-DWAA-uses-DEFLATE rule
/// (scanline images always use static Huffman, so this is `false` here).  The
/// channel list + data-window bounds are captured from the header; each chunk's
/// pixel rectangle is rebuilt from `min_y` and `num_scan_lines`.
pub struct DwaCompressor {
    /// Scan lines per chunk (32 = DWAA, 256 = DWAB).
    num_scan_lines: i32,
    /// DWA compression level (base error = level / 100000).
    level: f32,
    /// Use DEFLATE (vs Huffman) for AC — `true` only for tiled DWAA.
    use_deflate_for_ac: bool,
    /// Channel list captured from the header.
    channels: ChannelList,
    /// Data-window min x.
    min_x: i32,
    /// Data-window max x.
    max_x: i32,
    /// Data-window max y — clamps the final (partial) block.
    max_y: i32,
    /// Output scratch (the C++ member output buffer); valid until the next call.
    out: Vec<u8>,
    /// AC-coefficient decode scratch, reused across `uncompress()` calls —
    /// mirrors `PizCompressor`'s `bitmap`/`tmp`/`chan_line` fields and C's
    /// own reused `decode->scratch_buffer_1` (`internal_piz.c:648-667`; DWA
    /// shares the same `internal_huf_decompress` entropy stage as PIZ).
    ac_scratch: Vec<u16>,
    /// DC/RLE/UNKNOWN decode scratch, reused across `uncompress()` calls —
    /// see [`DwaScratch`]'s doc (also owns the persistent zlib decompressor).
    scratch: DwaScratch,
    /// Persistent low-level zlib compressors for the AC-deflate/RLE/Unknown
    /// (level 9) and DC (`Header::zip_level()`; C++ `me->_zipLevel`
    /// via `exr_get_zip_compression_level`, the SAME per-header/
    /// process-global seam the ZIP codec uses) sub-streams — constructed
    /// once here instead of per `compress()` call (perf follow-up,
    /// 2026-09-24; see `compressor::zlib`'s module doc).
    deflate: DwaDeflateCompressors,
    /// Persistent Huffman ENCODE-side scratch for the AC
    /// coefficient stream's static-Huffman path (`compress_ac`) — see
    /// [`huf::HufEncScratch`]'s doc. Unused when `use_deflate_for_ac` is set
    /// (tiled DWAA), but always present since `use_deflate_for_ac` is a
    /// runtime, not compile-time, choice.
    huf_scratch: huf::HufEncScratch,
    /// Persistent AC-DEFLATE-path scratch for
    /// `compress_ac`'s `use_deflate_for_ac` branch (tiled DWAA only) --
    /// `u16s_to_le_bytes` fully overwrites every byte, so `ensure_len` is
    /// sound, matching the same argument as `dc_encode_bytes`/
    /// `dc_encode_reorder` below.
    ac_deflate_bytes: Vec<u8>,
    /// Persistent DC-encode scratch: `dc_encode_bytes`
    /// (fully overwritten by `u16s_to_le_bytes`) and `dc_encode_reorder`
    /// (fully overwritten by `reorder_and_predict` -- the same shared
    /// ZIP/RLE predictor contract that makes `ensure_len` safe in the ZIP
    /// and RLE codecs).
    dc_encode_bytes: Vec<u8>,
    dc_encode_reorder: Vec<u8>,
    /// `classify_channels`'s output for `self.channels`,
    /// computed ONCE here in `new()` -- see [`DwaChannelPlan`]'s doc for why
    /// this is valid for the compressor's whole lifetime (encode always
    /// uses `classifier::default_rules()` against a channel list that never
    /// changes after construction, even across `compress_tile`'s per-tile
    /// `min_x`/`max_x`/`max_y` updates).
    plan: DwaChannelPlan,
    /// Persistent, grow-only per-chunk encode scratch --
    /// see [`DwaCompressScratch`]'s doc.
    compress_scratch: DwaCompressScratch,
}

impl DwaCompressor {
    /// Build a DWA compressor from the header.
    ///
    /// `num_scan_lines` is 32 (DWAA) or 256 (DWAB).  `use_deflate_for_ac` is the
    /// tiled-DWAA AC strategy (`false` for scanline files).
    pub fn new(
        header: &Header,
        num_scan_lines: i32,
        use_deflate_for_ac: bool,
    ) -> Result<Self> {
        let dw = header.data_window()?;
        let channels = header.channels()?.clone();
        // C++ `DwaCompressor_construct` (:152) reads the level through
        // `exr_get_dwa_compression_level`, which yields the header's own
        // `dwaCompressionLevel` attribute when present, else the process-global
        // default — `Header::codec_dwa_level()` (plus
        // `exr_set_dwa_compression_level`'s range rule), the ONE seam for this:
        // it never errors (unlike the raw attribute
        // accessor `header.dwa_compression_level()`, which is a different,
        // narrower thing — see its doc in `header.rs`) and already folds in
        // `Header::set_dwa_level`/`set_default_dwa_compression_level`.
        let level = header.codec_dwa_level();
        // The DC sub-stream's zlib level is `Header::codec_zip_level()` — the
        // SAME per-header override / process-global default the ZIP codec
        // reads (`compressor/zip.rs`), not a DWA-specific value.
        let dc_zip_level = header.codec_zip_level();
        let deflate = DwaDeflateCompressors::new(dc_zip_level)?;
        // See `DwaChannelPlan`'s doc: encode's classification
        // is a pure function of `(channels, classifier::default_rules())`,
        // both fixed for this compressor's whole life, so it is built exactly
        // once, here, rather than once per `compress()`/`compress_tile()` call.
        let plan = build_channel_plan(&channels, classifier::default_rules())?;
        Ok(Self {
            num_scan_lines,
            level,
            use_deflate_for_ac,
            channels,
            min_x: dw.min.x,
            max_x: dw.max.x,
            max_y: dw.max.y,
            out: Vec::new(),
            ac_scratch: Vec::new(),
            scratch: DwaScratch::default(),
            deflate,
            huf_scratch: huf::HufEncScratch::default(),
            ac_deflate_bytes: Vec::new(),
            dc_encode_bytes: Vec::new(),
            dc_encode_reorder: Vec::new(),
            plan,
            compress_scratch: DwaCompressScratch::default(),
        })
    }

    /// Chunk pixel rectangle for a block starting at `min_y`.
    fn rectangle(&self, min_y: i32) -> Box2i {
        let block_max_y = (min_y + self.num_scan_lines - 1).min(self.max_y);
        Box2i::from_min_max(V2i::new(self.min_x, min_y), V2i::new(self.max_x, block_max_y))
    }
}

impl Compressor for DwaCompressor {
    fn num_scan_lines(&self) -> i32 {
        self.num_scan_lines
    }

    /// DWA uses the machine-independent Xdr line buffer.
    fn format(&self) -> Format {
        Format::Xdr
    }

    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }
        let rect = self.rectangle(min_y);
        // compress_dwa fills self.out directly; on "no-shrink"
        // it clears self.out itself and reports `false`, so we store raw here.
        let shrunk = compress_dwa(
            &self.channels, data, rect, self.level, self.use_deflate_for_ac, &self.plan, &mut self.deflate,
            &mut self.huf_scratch, &mut self.ac_deflate_bytes, &mut self.dc_encode_bytes,
            &mut self.dc_encode_reorder, &mut self.compress_scratch, &mut self.out,
        )?;
        if !shrunk {
            self.out.clear();
            self.out.extend_from_slice(data);
        }
        Ok(&self.out)
    }

    /// `expected_len` is unused: `decompress_dwa`'s
    /// `uncompressed_size` comes entirely from `channels`/`rectangle`
    /// (`bytes_per_line_table`), never a construction-time bound.
    fn uncompress(&mut self, data: &[u8], min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        if data.is_empty() {
            self.out.clear();
            return Ok(&self.out);
        }
        let rect = self.rectangle(min_y);
        decompress_dwa(&self.channels, data, rect, &mut self.out, &mut self.ac_scratch, &mut self.scratch)?;
        Ok(&self.out)
    }

    /// Compress one tile, encoded against the tile's own pixel rectangle.
    ///
    /// DWA's channel classification and DCT block layout are driven entirely by
    /// `self.rectangle(min_y)` (width = `max_x-min_x+1`, height = block extent),
    /// so encoding a tile correctly only requires the rectangle to be the tile's
    /// X/Y extent — exactly what C++ `compressTile` achieves by feeding the
    /// tile's `Box2i range` into the encode step.  Sets `min_x`/`max_x`/`max_y`
    /// from `range`, then runs the identical compress path at `range.min.y`
    /// (fresh each tile call); `rectangle(range.min.y)` then yields the tile rect.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::Channel;

    /// Test-only convenience wrapper around the grow-only, caller-owns-the-
    /// buffer `decompress_dwa` (the production signature, matching
    /// `DwaCompressor`'s persistent `ac_scratch` field) — NOT a second
    /// production API, just saving every one-off test call site from
    /// spelling out its own throwaway `Vec::new()`. Production code
    /// (`DwaCompressor::uncompress`) always calls the real function with its
    /// own persistent buffer.
    fn decompress_dwa(channels: &ChannelList, compressed: &[u8], rectangle: Box2i) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        super::decompress_dwa(
            channels,
            compressed,
            rectangle,
            &mut out,
            &mut Vec::new(),
            &mut super::DwaScratch::default(),
        )?;
        Ok(out)
    }

    /// Test-only convenience wrapper offering the
    /// `Result<Option<Vec<u8>>>` shape (`Some` = shrunk chunk, `None` =
    /// "no-shrink → store raw") for the many existing call sites — NOT a
    /// second production API, exactly like `decompress_dwa`'s wrapper above.
    /// Production code (`DwaCompressor::compress`) always calls the real
    /// function with its own persistent `self.out` buffer.
    fn compress_dwa(
        channels: &ChannelList,
        data: &[u8],
        rectangle: Box2i,
        level: f32,
        use_deflate_for_ac: bool,
        dc_zip_level: i32,
    ) -> Result<Option<Vec<u8>>> {
        let mut out = Vec::new();
        let mut deflate = super::DwaDeflateCompressors::new(dc_zip_level)?;
        let mut huf_scratch = super::huf::HufEncScratch::default();
        let mut ac_deflate_bytes = Vec::new();
        let mut dc_encode_bytes = Vec::new();
        let mut dc_encode_reorder = Vec::new();
        // Throwaway plan/scratch, matching the real
        // signature's split (a persistent `DwaChannelPlan` + a persistent
        // `DwaCompressScratch` on `DwaCompressor` in production).
        let plan = super::build_channel_plan(channels, super::classifier::default_rules())?;
        let mut scratch = super::DwaCompressScratch::default();
        let shrunk = super::compress_dwa(
            channels, data, rectangle, level, use_deflate_for_ac, &plan, &mut deflate, &mut huf_scratch,
            &mut ac_deflate_bytes, &mut dc_encode_bytes, &mut dc_encode_reorder, &mut scratch, &mut out,
        )?;
        Ok(if shrunk { Some(out) } else { None })
    }

    /// DWA's UNKNOWN sub-stream tolerates a genuinely SHORT
    /// zlib decode -- C passes NULL `actual_out` there, so `libdeflate`'s
    /// `LIBDEFLATE_SHORT_OUTPUT` converts to `EXR_ERR_SUCCESS`
    /// UNCONDITIONALLY (`compression.c:144-208`), unlike every other zlib
    /// caller in this codec (`zlib_decompress_exact`, `RequireExact` policy),
    /// which always demands the consume-all-input check. Must accept a short
    /// decode (`cap` larger than what the stream actually contains) AND
    /// trailing garbage appended after that short stream (C's
    /// `SHORT_OUTPUT` branch skips the consume-all check
    /// entirely, not just the exact-size one). The result is truncated to
    /// the real decoded length here -- the deterministic zero tail comes
    /// from `decompress_dwa`'s own `output` buffer (already zero-resized
    /// before its assembly loop reads `unknown_data` through
    /// `.min(unknown_data.len())`), not from padding this buffer itself.
    #[test]
    fn zlib_decompress_unknown_tolerates_a_short_decode_and_trailing_garbage() {
        let real = b"a genuinely short payload, shorter than the declared cap";
        let mut comp = new_deflate_compressor(4).unwrap();
        let compressed = zlib_compress(real, &mut comp).unwrap();
        let mut decompressor = LowDecompressor::new();

        // Declare a cap far larger than what this stream will ever decode to.
        let cap = real.len() + 1000;
        let mut out = Vec::new();
        zlib_decompress_unknown(&compressed, cap, &mut out, &mut decompressor).unwrap();
        assert_eq!(&out[..], &real[..], "a short decode must still produce its real bytes");

        // Trailing garbage AFTER the short stream must also be accepted (no
        // consume-all check on the SHORT_OUTPUT path).
        let mut with_garbage = compressed.clone();
        with_garbage.extend_from_slice(&[0xAB; 37]);
        let mut out2 = Vec::new();
        zlib_decompress_unknown(&with_garbage, cap, &mut out2, &mut decompressor).unwrap();
        assert_eq!(&out2[..], &real[..], "trailing garbage after a short stream must be tolerated");
    }

    /// The exact-size AC/DC/RLE sub-streams (and ZIP/PXR24/IDManifest) do NOT
    /// share UNKNOWN's leniency: trailing garbage after their zlib stream is
    /// still a corrupt-chunk error, even when the declared cap is larger
    /// than the real decoded content (the same "short" shape UNKNOWN
    /// tolerates).
    #[test]
    fn zlib_decompress_exact_still_rejects_trailing_garbage() {
        let real = b"a genuinely short payload, shorter than the declared cap";
        let mut comp = new_deflate_compressor(4).unwrap();
        let compressed = zlib_compress(real, &mut comp).unwrap();
        let mut with_garbage = compressed.clone();
        with_garbage.extend_from_slice(&[0xAB; 37]);

        let mut decompressor = LowDecompressor::new();
        let err = zlib_decompress_exact(&with_garbage, real.len() + 1000, &mut decompressor).unwrap_err();
        assert!(err.to_string().contains("trailing byte"), "got: {err}");
    }

    /// The C++ OpenEXR oracle `cargo run -p xtask -- parity-exr` builds, or
    /// `None` when the operator explicitly allowed skipping.
    ///
    /// Duplicated (deliberately small) from `tests/support/mod.rs`'s
    /// `oracle_or_skip`/`default_oracle`: that module is a submodule of the
    /// EXTERNAL `tests/parity_exr.rs` integration-test binary, not reachable
    /// from a `#[cfg(test)]` unit test compiled into the library itself
    /// (Cargo does not share `tests/` modules with `src/`), and this test
    /// needs private DWA header/channel-rules internals `tests/parity_exr.rs`
    /// cannot see. Same policy as the original: panics with an actionable
    /// message unless `EXR_ALLOW_SKIP=1` -- a "C++ cross-read" test that
    /// silently did not ask C++ anything is exactly what that policy exists
    /// to prevent (`tests/support/mod.rs`'s own module doc).
    fn oracle_or_skip_dwa_unit_test(name: &str) -> Option<std::path::PathBuf> {
        let allow_skip = std::env::var_os("EXR_ALLOW_SKIP").is_some_and(|v| v == "1");
        let path = std::env::var_os("EXR_ORACLE_EXE")
            .map(std::path::PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests")
                    .join("parity")
                    .join(if cfg!(windows) { "oracle.exe" } else { "oracle" })
            });
        if path.exists() {
            return Some(path);
        }
        let msg = format!(
            "{name}: the C++ OpenEXR oracle is missing ({}) -- this test did NOT run. \
             Run `cargo run -p xtask -- parity-exr` to build it, or set EXR_ORACLE_EXE. \
             Set EXR_ALLOW_SKIP=1 to skip it deliberately.",
            path.display()
        );
        if allow_skip {
            eprintln!("SKIP (EXR_ALLOW_SKIP=1) {msg}");
            return None;
        }
        panic!("{msg}");
    }

    /// (ORACLE-VERIFIED) A real C++ OpenEXR reader accepts a
    /// DWA chunk whose UNKNOWN sub-stream decodes to FEWER bytes than its
    /// declared `unknownUncompressedSize` -- confirming the
    /// `compression.c:144-208` reading (`LIBDEFLATE_SHORT_OUTPUT ->
    /// EXR_ERR_SUCCESS`, unconditional, no consume-all check) against the
    /// REAL library, not just our own reading of the source.
    ///
    /// Builds a single-channel ("Z", Unknown scheme), single-chunk DWAA
    /// image, compresses it normally, then SHRINKS the UNKNOWN sub-stream to
    /// only the first half of the real pixel bytes while leaving
    /// `unknownUncompressedSize` at the ORIGINAL (now larger) declared value
    /// -- a genuinely short decode, not a corrupted one. Checks both sides
    /// independently: our own reader must accept it with a real prefix and a
    /// deterministically zero tail; the real C++ oracle, reading the SAME
    /// patched file (spliced in via the public `RawImage`/`RawChunk`
    /// passthrough API), must also accept it (exit success) -- C's own tail
    /// content is NOT compared (it is not deterministic).
    #[test]
    fn oracle_dwa_unknown_short_stream_is_accepted_by_real_cpp() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::frame_buffer::{FrameBuffer, Slice};
        use crate::io::{MemIStream, MemOStream};
        use crate::scanline::OutputFile;
        use crate::RawImage;
        use imath_rs::V2f;

        let name = "oracle_dwa_unknown_short_stream_is_accepted_by_real_cpp";
        let Some(oracle) = oracle_or_skip_dwa_unit_test(name) else {
            return;
        };

        let w = 8i32;
        let h = 8i32;
        let channels = cl(&[("Z", PixelType::Float)]);
        let mut hdr = Header::with_dimensions(
            w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Dwaa,
        )
        .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels)).unwrap();

        // Real, distinct-valued raw Z pixel bytes -- a successful decode of
        // the real prefix is trivially distinguishable from a zero tail.
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let v = 1.5f32 + i as f32;
            pixels[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // 1) Compress the chunk normally, then locate its UNKNOWN section by
        //    parsing the SAME header/channel-rules format `decompress_dwa`
        //    itself parses.
        let mut c = DwaCompressor::new(&hdr, 32, false).unwrap();
        let compressed_chunk = c.compress(&pixels, 0).unwrap().to_vec();

        let mut header = read_header(&compressed_chunk).unwrap();
        let version = header[Field::Version as usize];
        let mut data_offset = HEADER_SIZE;
        if version >= 2 {
            channel_rules::read_channel_rules(&compressed_chunk, &mut data_offset).unwrap();
        }
        let unknown_uncompressed = header[Field::UnknownUncompressedSize as usize] as usize;
        let unknown_compressed = header[Field::UnknownCompressedSize as usize] as usize;
        assert!(unknown_compressed > 0, "test setup: the Z-only chunk must have a real UNKNOWN section");
        assert_eq!(
            unknown_uncompressed,
            pixels.len(),
            "test setup: a single-channel chunk's declared UNKNOWN size is the whole plane"
        );
        assert_eq!(
            compressed_chunk.len(),
            data_offset + unknown_compressed,
            "test setup: UNKNOWN must be the chunk's only (and therefore last) section"
        );

        // 2) Replace the UNKNOWN section with a genuinely SHORTER stream
        //    (only the first half of the real bytes), keeping
        //    `unknownUncompressedSize` at the ORIGINAL, now-larger value.
        let half = pixels.len() / 2;
        let mut short_comp = new_deflate_compressor(4).unwrap();
        let short_compressed = zlib_compress(&pixels[..half], &mut short_comp).unwrap();
        header[Field::UnknownCompressedSize as usize] = short_compressed.len() as u64;

        let mut patched_chunk = Vec::new();
        write_header(&header, &mut patched_chunk);
        patched_chunk.extend_from_slice(&compressed_chunk[HEADER_SIZE..data_offset]); // channel-rules blob, if any, verbatim
        patched_chunk.extend_from_slice(&short_compressed);

        // 3) OUR reader: must accept it, real prefix intact, tail exactly zero.
        let mut c2 = DwaCompressor::new(&hdr, 32, false).unwrap();
        let decoded = c2
            .uncompress(&patched_chunk, 0, 0)
            .unwrap_or_else(|e| panic!("{name}: our own reader rejected the short-UNKNOWN chunk: {e}"))
            .to_vec();
        assert_eq!(decoded.len(), pixels.len());
        assert_eq!(&decoded[..half], &pixels[..half], "ours: the real prefix must decode correctly");
        assert!(
            decoded[half..].iter().all(|&b| b == 0),
            "ours: the undecoded tail must be deterministically zero"
        );

        // 4) Splice the patched chunk into a real, otherwise-normal .exr file
        //    (via the public RawImage/RawChunk passthrough) and have the
        //    REAL C++ oracle read it.
        let mut mem = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut mem, &hdr).unwrap();
            let mut fb = FrameBuffer::new();
            fb.insert("Z", Slice::new(PixelType::Float, 0, 4, (w * 4) as usize, 1, 1, 0.0, false, false).unwrap())
                .unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, h).unwrap();
        }
        let plain_bytes = mem.into_data();

        let mut is = MemIStream::new(plain_bytes);
        let mut img = RawImage::read(&mut is).expect("parse our own freshly-written file");
        assert_eq!(img.parts[0].chunks.len(), 1, "test setup: an 8x8 image with a 32-line DWAA block is one chunk");
        img.parts[0].chunks[0].data = patched_chunk;

        let mut patched_os = MemOStream::new();
        img.write(&mut patched_os).expect("write the patched file");
        let patched_bytes = patched_os.into_data();

        let out_path = std::env::temp_dir().join(format!("{name}.exr"));
        std::fs::write(&out_path, &patched_bytes)
            .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));

        let output = std::process::Command::new(&oracle)
            .arg("read")
            .arg(&out_path)
            .output()
            .expect("failed to run oracle read");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{name}: real C++ OpenEXR rejected a short UNKNOWN stream it should accept \
             (status {}):\nstderr:\n{stderr}",
            output.status
        );
        let _ = std::fs::remove_file(&out_path);
    }

    /// Slope-parameterised line buffer for the `ac_scratch` reuse test below
    /// — a fixed-formula smooth gradient (like [`build_buffer`]) is the WRONG
    /// choice here: a pure additive offset between two calls only changes
    /// each 8x8 block's DC term, not its AC content (DCT AC coefficients
    /// depend on within-block VARIATION, not absolute level) — the same
    /// class of mistake documented on the PIZ scratch-reuse test above.
    /// Using a genuinely different `x_slope` between calls changes the
    /// actual AC coefficient magnitudes, so a stale-vs-fresh `ac_scratch`
    /// leak can actually be observed. Still smooth within each block, so it
    /// remains genuinely DWA-compressible (unlike per-pixel noise, which
    /// would defeat both the DCT and the entropy stage and fall back to raw
    /// storage, skipping the very decode path this test targets).
    fn build_buffer_with_slope(w: usize, h: usize, infos: &ChannelList, x_slope: f32) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..h {
            for (ci, (_, ch)) in infos.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 4.0 + (y as f32) * 0.1 + (x as f32) * x_slope;
                    match ch.type_ {
                        PixelType::Half => buf.extend_from_slice(&Half::from_f32(v).bits().to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&(v as u32).to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// Scratch-buffer reuse: `ac_scratch` is a
    /// persistent `DwaCompressor` field, filled in place on every
    /// `uncompress()` call instead of freshly allocated — this test exists
    /// specifically to catch stale AC-coefficient data leaking from a PRIOR,
    /// LARGER chunk into a SMALLER one that reuses the same buffer.
    ///
    /// Uses `DwaCompressor` directly (not the `decompress_dwa` test wrapper
    /// above, which passes a throwaway `Vec::new()` and therefore CANNOT see
    /// this class of bug at all). Two consecutive `Compressor::uncompress`
    /// calls on ONE instance: `min_y = 0` decodes the FULL 32-row DWA block
    /// (populates `ac_scratch` with plenty of AC data from an RGB CSC
    /// group — via the Deflate AC path, `use_deflate_for_ac: true` below;
    /// see the mutation-check note further down for why it has to be this
    /// path and not the default Static-Huffman one);
    /// `min_y = 32` on a 40-row image decodes the final, strictly SMALLER
    /// 8-row partial block on the SAME instance. The small block's restored
    /// bytes are checked byte-for-byte against an INDEPENDENT reference: a
    /// FRESH `DwaCompressor` decoding the identical compressed bytes with no
    /// prior call — decode of a fixed byte stream is deterministic
    /// regardless of DWA's lossy ENCODE, so the two must match exactly, and
    /// any divergence proves contamination from the big call.
    ///
    /// Mutation check performed while writing this test: temporarily
    /// removing the `ac_scratch.clear()` call in `decompress_dwa` (so the
    /// small block's own AC entries get APPENDED after the big block's
    /// stale leftovers instead of replacing them) made this test fail with
    /// a real byte mismatch, confirmed before reverting -- but only once two
    /// earlier mistakes were fixed, recorded here so they are not repeated:
    /// (1) `AcCompression::StaticHuffman` (the default, `use_deflate_for_ac:
    /// false`) is masked by `huf::decompress`'s OWN internal
    /// `out.clear()` regardless of the outer bug, so the mutation is only
    /// observable via the `AcCompression::Deflate` path (`use_deflate_for_ac:
    /// true`), which appends into `ac_scratch` with no such inner guard; (2)
    /// a fixed-slope smooth gradient for both blocks (like [`build_buffer`])
    /// produced IDENTICAL quantised AC coefficients for the big block's
    /// leading rows and the small block's own rows regardless of staleness
    /// (both are 8x8-block-local, and a pure y-offset shift does not change
    /// AC content for a linear ramp), making the leak invisible even though
    /// it was really happening -- [`build_buffer_with_slope`] with distinct
    /// slopes per call exists specifically to avoid that coincidence.
    #[test]
    fn uncompress_ac_scratch_does_not_leak_between_a_big_then_small_block() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use imath_rs::V2f;

        let w = 32i32;
        let h = 40i32; // 32 (full block) + 8 (partial final block)
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
        ]);
        let mut hdr = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Dwaa)
            .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();

        let mut c = DwaCompressor::new(&hdr, 32, true).unwrap();

        // First call: the full 32-row block, to populate `ac_scratch` with
        // plenty of AC data.
        let big_buf = build_buffer_with_slope(w as usize, 32, &channels, 0.2);
        let big_compressed = c.compress(&big_buf, 0).unwrap().to_vec();
        let big_restored = c.uncompress(&big_compressed, 0, 0).unwrap().to_vec();
        assert_eq!(big_restored.len(), big_buf.len(), "sanity: the big block must decode to the right size");

        // Second call, same compressor instance: the final, strictly
        // SMALLER 8-row block.
        let small_compressed = c.compress(&build_buffer_with_slope(w as usize, 8, &channels, 1.7), 32).unwrap().to_vec();
        let small_restored_reused = c.uncompress(&small_compressed, 32, 0).unwrap().to_vec();

        // Independent reference: a FRESH compressor decodes the identical
        // small-block bytes with no prior call, so it cannot see stale
        // ac_scratch data from the big call above.
        let mut fresh = DwaCompressor::new(&hdr, 32, true).unwrap();
        let small_restored_fresh = fresh.uncompress(&small_compressed, 32, 0).unwrap().to_vec();

        assert_eq!(
            small_restored_reused, small_restored_fresh,
            "the reused compressor's small-block output diverged from a fresh instance's -- stale ac_scratch data leaked from the earlier big block"
        );
    }

    /// `DwaCompressor`'s `ac_deflate_bytes` (the
    /// `use_deflate_for_ac` / tiled-DWAA path), `dc_encode_bytes`, and
    /// `dc_encode_reorder` are persistent, reused fields instead of
    /// fresh `vec![...]`s per `compress()` call. Poison all three with
    /// implausible values BEFORE the first real call, then compress a
    /// SEQUENCE of genuinely different-sized/sloped blocks (same
    /// big-then-small methodology as the uncompress-side leak test above)
    /// on that poisoned-then-reused instance, diffing every result against
    /// a completely fresh instance.
    #[test]
    fn compress_scratch_does_not_leak_between_a_big_then_small_block() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use imath_rs::V2f;

        let w = 32i32;
        let h = 40i32; // 32 (full block) + 8 (partial final block)
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
        ]);
        let mut hdr = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Dwaa)
            .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();

        // `use_deflate_for_ac: true` (tiled DWAA) so this exercises
        // `ac_deflate_bytes` too, not just the DC-encode buffers.
        let mut c = DwaCompressor::new(&hdr, 32, true).unwrap();
        c.ac_deflate_bytes = vec![0xAAu8; 8192];
        c.dc_encode_bytes = vec![0x55u8; 8192];
        c.dc_encode_reorder = vec![0xFFu8; 8192];

        let big_compressed = c.compress(&build_buffer_with_slope(w as usize, 32, &channels, 0.2), 0).unwrap().to_vec();
        let small_compressed =
            c.compress(&build_buffer_with_slope(w as usize, 8, &channels, 1.7), 32).unwrap().to_vec();

        let mut fresh = DwaCompressor::new(&hdr, 32, true).unwrap();
        let big_compressed_fresh =
            fresh.compress(&build_buffer_with_slope(w as usize, 32, &channels, 0.2), 0).unwrap().to_vec();
        let small_compressed_fresh =
            fresh.compress(&build_buffer_with_slope(w as usize, 8, &channels, 1.7), 32).unwrap().to_vec();

        assert_eq!(
            big_compressed, big_compressed_fresh,
            "a poisoned, reused DwaCompressor produced different compressed bytes for the \
             big block than a fresh instance"
        );
        assert_eq!(
            small_compressed, small_compressed_fresh,
            "a poisoned, reused DwaCompressor produced different compressed bytes for the \
             small block than a fresh instance -- some encode-side scratch field is not fully \
             overwritten on reuse"
        );
    }

    /// `DwaChannelPlan`'s slot maps (`dct_slot`/`rle_slot`/
    /// `unknown_slot`/`dct_to_info`/`info_to_dct`) and `DwaCompressScratch`'s
    /// grow-only per-scheme Vecs (`dct_channels`/`rle_interleaved`/
    /// `rle_channels`/`unknown_channels`/`all_dc`/`all_ac`/`dc_y`/`dc_cb`/
    /// `dc_cr`/`rle_raw`/`unknown_raw`) are reused across chunks. Exercise a
    /// RICH classification -- TWO CSC groups (`R,G,B` and the second-seen
    /// `Hair.R,Hair.G,Hair.B` layer prefix, both driven by the same
    /// `channel_rules` v2 write-then-read path `decoder_uses_rules_stored_in_the_chunk`
    /// tests directly), an RLE channel (`A`), and an Unknown channel (`Z`)
    /// -- on the SAME persistent, poisoned `DwaCompressor` through a
    /// big-then-small chunk-size sequence (same methodology as
    /// `compress_scratch_does_not_leak_between_a_big_then_small_block` /
    /// `uncompress_ac_scratch_does_not_leak_between_a_big_then_small_block`),
    /// covering both encode's and decode's scratch fields' "clear, not
    /// reallocate" contract at once, with a non-trivial channel SET (not
    /// just a size change) driving the plan/slot maps every call.
    #[test]
    fn compress_and_uncompress_scratch_survive_rich_classification_big_then_small() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use imath_rs::V2f;

        let w = 32i32;
        let h = 40i32; // 32 (full block) + 8 (partial final block)
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
            ("Hair.R", PixelType::Half),
            ("Hair.G", PixelType::Half),
            ("Hair.B", PixelType::Half),
            ("A", PixelType::Half),
            ("Z", PixelType::Float),
        ]);
        // Sanity: this channel list really does hit every scheme + two CSC
        // groups under the default (encode) rules -- if a future edit to
        // `classifier::default_rules()`/`find_suffix` changed that, this
        // test would silently stop exercising what it claims to.
        let plan_check = classify_channels(&channels, classifier::default_rules()).unwrap();
        assert_eq!(plan_check.csc_groups.len(), 2, "test setup: must produce two CSC groups");
        assert!(
            plan_check.infos.iter().any(|i| i.scheme == CompressionScheme::Rle),
            "test setup: must have an RLE channel"
        );
        assert!(
            plan_check.infos.iter().any(|i| i.scheme == CompressionScheme::Unknown),
            "test setup: must have an Unknown channel"
        );

        let mut hdr = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Dwaa)
            .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();

        let mut c = DwaCompressor::new(&hdr, 32, true).unwrap();
        // Poison every new per-chunk scratch field BEFORE the first real call.
        c.compress_scratch.all_dc = vec![0xAAAA; 4096];
        c.compress_scratch.all_ac = vec![0x5555; 4096];
        c.compress_scratch.dc_y = vec![0xAAAA; 512];
        c.compress_scratch.dc_cb = vec![0x5555; 512];
        c.compress_scratch.dc_cr = vec![0xAAAA; 512];
        c.compress_scratch.rle_raw = vec![0xAA; 4096];
        c.compress_scratch.unknown_raw = vec![0x55; 4096];
        c.scratch.dc = vec![0xAAAA; 4096];
        c.scratch.rle = vec![0x55; 4096];
        c.scratch.unknown = vec![0xAA; 4096];

        let big_buf = build_buffer_with_slope(w as usize, 32, &channels, 0.2);
        let big_compressed = c.compress(&big_buf, 0).unwrap().to_vec();
        let big_restored = c.uncompress(&big_compressed, 0, 0).unwrap().to_vec();
        assert_eq!(big_restored.len(), big_buf.len(), "sanity: the big block must decode to the right size");

        let small_buf = build_buffer_with_slope(w as usize, 8, &channels, 1.7);
        let small_compressed = c.compress(&small_buf, 32).unwrap().to_vec();
        let small_restored = c.uncompress(&small_compressed, 32, 0).unwrap().to_vec();

        // Independent reference: a FRESH compressor runs the identical
        // sequence with no prior call, so it cannot see stale scratch data.
        let mut fresh = DwaCompressor::new(&hdr, 32, true).unwrap();
        let big_compressed_fresh = fresh.compress(&big_buf, 0).unwrap().to_vec();
        let big_restored_fresh = fresh.uncompress(&big_compressed_fresh, 0, 0).unwrap().to_vec();
        let small_compressed_fresh = fresh.compress(&small_buf, 32).unwrap().to_vec();
        let small_restored_fresh = fresh.uncompress(&small_compressed_fresh, 32, 0).unwrap().to_vec();

        assert_eq!(
            big_compressed, big_compressed_fresh,
            "poisoned/reused big-block compressed bytes diverged from a fresh instance"
        );
        assert_eq!(
            big_restored, big_restored_fresh,
            "poisoned/reused big-block decoded bytes diverged from a fresh instance"
        );
        assert_eq!(
            small_compressed, small_compressed_fresh,
            "poisoned/reused small-block compressed bytes diverged from a fresh instance"
        );
        assert_eq!(
            small_restored, small_restored_fresh,
            "poisoned/reused small-block decoded bytes diverged from a fresh instance -- \
             a new scratch field is not fully overwritten on reuse"
        );
    }

    /// Decode-side scratch reuse:
    /// `LossyDctDecoder::decode_channel_into`/`decode_rgb_into` and
    /// `DwaScratch::dct_outputs`/`dct_filled`/`row_blocks`/`row_blocks_rgb`
    /// are grow-only buffers reused across chunks. Reuses the EXACT rich-classification fixture from
    /// `compress_and_uncompress_scratch_survive_rich_classification_big_then_small`
    /// above (two CSC groups, an RLE channel, an Unknown channel, a real
    /// v2 stream with channel rules embedded per chunk -- `DwaCompressor`
    /// always writes version 2, `Field::Version` set to `2` at `:1563`, so
    /// this exercises `DwaScratch::plan`/`plan_rules`'s v2 rules-from-chunk
    /// path the same way `decoder_uses_rules_stored_in_the_chunk` does),
    /// this time poisoning ONLY the decode-side buffers with content
    /// LARGER than either chunk needs (so a broken `ensure_len`/truncate
    /// would leak stale, wrong-sized garbage into the decoded pixels) before
    /// the first real decode, through the same big (32-row) then small
    /// (8-row, partial final block) sequence, diffed byte-for-byte against a
    /// completely fresh, unpoisoned instance running the identical sequence.
    /// Also adds one "Y" (luminance) channel to the fixture, which the
    /// `DEFAULT_RULES` table classifies `LossyDct` with no CSC group -- the
    /// R/G/B and Hair.R/G/B groups above only ever exercise
    /// `decode_rgb_into`, so without "Y" this test would never touch
    /// `decode_channel_into`'s (the standalone-channel path) own
    /// `ensure_len` calls at all.
    ///
    /// Mutation check: temporarily commented out
    /// `decode_channel_into`'s `ensure_len(row_blocks, ..)` call -- this test
    /// failed immediately with an index-out-of-bounds panic (a fresh
    /// `DwaScratch::row_blocks` starts as an empty `Vec`, so the very first
    /// write into it panics without the grow) rather than the "diverges from
    /// fresh" byte-comparison this test's assertions check for. That is an
    /// honest, different kind of proof than the encode-side poison
    /// test's (which catches STALE BYTES leaking through): every pixel these
    /// decode buffers hold is fully overwritten every call regardless of
    /// buffer reuse (see the `ensure_len`/"fully overwritten" reasoning in
    /// `decode_channel_into`'s/`decode_rgb_into`'s own doc comments), so
    /// there is no stale-byte scenario to poison FOR on the decode side --
    /// the mutation instead proves the `ensure_len` calls are load-bearing
    /// (the allocation genuinely happens there, not somewhere the mutation
    /// would silently no-op), which is what this test can actually prove.
    /// Reverted after confirming red.
    #[test]
    fn decode_scratch_survives_rich_classification_big_then_small() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use imath_rs::V2f;

        let w = 32i32;
        let h = 40i32; // 32 (full block) + 8 (partial final block)
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
            ("Hair.R", PixelType::Half),
            ("Hair.G", PixelType::Half),
            ("Hair.B", PixelType::Half),
            ("A", PixelType::Half),
            ("Z", PixelType::Float),
            // "Y" (luminance) matches `DEFAULT_RULES`' standalone (CscIndex::
            // None) LOSSY_DCT rule -- never joins a CSC group, so this
            // exercises the STANDALONE `decode_channel_into` path (the two
            // CSC groups above only ever exercise `decode_rgb_into`).
            ("Y", PixelType::Half),
        ]);
        let plan_check = classify_channels(&channels, classifier::default_rules()).unwrap();
        assert_eq!(plan_check.csc_groups.len(), 2, "test setup: must produce two CSC groups");
        assert!(
            plan_check.infos.iter().any(|i| i.scheme == CompressionScheme::Rle),
            "test setup: must have an RLE channel"
        );
        assert!(
            plan_check.infos.iter().any(|i| i.scheme == CompressionScheme::Unknown),
            "test setup: must have an Unknown channel"
        );
        let in_any_csc: std::collections::HashSet<usize> =
            plan_check.csc_groups.iter().flat_map(|g| [g.r, g.g, g.b]).collect();
        assert!(
            plan_check
                .infos
                .iter()
                .enumerate()
                .any(|(i, info)| info.scheme == CompressionScheme::LossyDct && !in_any_csc.contains(&i)),
            "test setup: must have a STANDALONE (non-CSC) LossyDct channel"
        );

        let mut hdr = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Dwaa)
            .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();

        let mut c = DwaCompressor::new(&hdr, 32, true).unwrap();
        let big_buf = build_buffer_with_slope(w as usize, 32, &channels, 0.2);
        let big_compressed = c.compress(&big_buf, 0).unwrap().to_vec();

        // Poison the DECODE-side scratch fields ONLY, and only
        // AFTER the compress above (compress doesn't touch `scratch` at all,
        // but this keeps the poisoning obviously scoped to what this test
        // claims to cover) -- deliberately oversized vs. what either chunk
        // actually needs, so a broken truncate/ensure_len would leave real
        // garbage past the correct length instead of silently "fitting".
        c.scratch.dct_outputs = vec![vec![0xAAAAu16; 4096]; channels.len()];
        c.scratch.dct_filled = vec![true; channels.len()];
        c.scratch.row_blocks = vec![0x5555u16; 4096];
        c.scratch.row_blocks_rgb = [vec![0xAAAAu16; 4096], vec![0x5555u16; 4096], vec![0xAAAAu16; 4096]];

        let big_restored = c.uncompress(&big_compressed, 0, 0).unwrap().to_vec();
        assert_eq!(big_restored.len(), big_buf.len(), "sanity: the big block must decode to the right size");

        let small_buf = build_buffer_with_slope(w as usize, 8, &channels, 1.7);
        let small_compressed = c.compress(&small_buf, 32).unwrap().to_vec();
        let small_restored = c.uncompress(&small_compressed, 32, 0).unwrap().to_vec();

        // Independent reference: a FRESH compressor runs the identical
        // sequence with no prior call, so it cannot see stale scratch data.
        let mut fresh = DwaCompressor::new(&hdr, 32, true).unwrap();
        let big_compressed_fresh = fresh.compress(&big_buf, 0).unwrap().to_vec();
        let big_restored_fresh = fresh.uncompress(&big_compressed_fresh, 0, 0).unwrap().to_vec();
        let small_compressed_fresh = fresh.compress(&small_buf, 32).unwrap().to_vec();
        let small_restored_fresh = fresh.uncompress(&small_compressed_fresh, 32, 0).unwrap().to_vec();

        assert_eq!(
            big_compressed, big_compressed_fresh,
            "compress is unaffected by this test's poisoning (decode-only), sanity check"
        );
        assert_eq!(
            big_restored, big_restored_fresh,
            "poisoned/reused big-block decoded bytes diverged from a fresh instance -- \
             decode_channel_into/decode_rgb_into or DwaScratch's dct_outputs/row_blocks \
             is not fully overwritten on reuse"
        );
        assert_eq!(
            small_compressed, small_compressed_fresh,
            "compress is unaffected by this test's poisoning (decode-only), sanity check"
        );
        assert_eq!(
            small_restored, small_restored_fresh,
            "poisoned/reused small-block decoded bytes diverged from a fresh instance -- \
             decode_channel_into/decode_rgb_into or DwaScratch's dct_outputs/row_blocks \
             is not fully overwritten on reuse (shrink case: big-chunk poison, small-chunk decode)"
        );
    }

    /// Mutation-testing gap fix: hand-derived boundary values against
    /// `validate_dwa_size`'s own documented contract (its doc comment cites
    /// `internal_dwa_compressor.h:691-726` and spells out both branches
    /// exactly), not a round trip through the codec.
    #[test]
    fn validate_dwa_size_matches_documented_boundaries() {
        // required_size > 0: uncompressed_size must reach it (>=), and
        // compressed_size must be nonzero.
        assert!(validate_dwa_size(1, 10, 0, 10).is_ok(), "uncompressed == required");
        assert!(validate_dwa_size(1, 11, 0, 10).is_ok(), "uncompressed > required");
        assert!(validate_dwa_size(1, 9, 0, 10).is_err(), "uncompressed < required must error");
        assert!(validate_dwa_size(0, 10, 0, 10).is_err(), "compressed_size == 0 must error even if uncompressed is enough");

        // required_size == 0: every declared size must be exactly zero.
        assert!(validate_dwa_size(0, 0, 0, 0).is_ok());
        assert!(validate_dwa_size(1, 0, 0, 0).is_err(), "compressed_size > 0 with no channel must error");
        assert!(validate_dwa_size(0, 1, 0, 0).is_err(), "uncompressed_size > 0 with no channel must error");
        assert!(validate_dwa_size(0, 0, 1, 0).is_err(), "extra_size > 0 with no channel must error");
    }

    fn cl(chans: &[(&str, PixelType)]) -> ChannelList {
        let mut c = ChannelList::new();
        for (n, t) in chans {
            c.insert(*n, Channel::new(*t, 1, 1, false)).unwrap();
        }
        c
    }

    fn rect(w: i32, h: i32) -> Box2i {
        Box2i::from_min_max(V2i::new(0, 0), V2i::new(w - 1, h - 1))
    }

    /// Build a scanline-interleaved native-LE buffer: for each line, each
    /// channel (in list order), `w` samples of a deterministic gradient.
    fn build_buffer(w: usize, h: usize, infos: &ChannelList) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..h {
            for (ci, (_, ch)) in infos.iter().enumerate() {
                for x in 0..w {
                    let v = (ci as f32) * 4.0 + (y as f32) * 0.1 + x as f32 * 0.2;
                    match ch.type_ {
                        PixelType::Half => buf.extend_from_slice(&Half::from_f32(v).bits().to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&(v as u32).to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// Like [`build_buffer`], but respects each channel's sampling — a
    /// subsampled channel contributes only its own `w/x_sampling` samples on
    /// the rows its `y_sampling` includes, matching `channel_dims`/
    /// `misc::bytes_per_line_table`'s packed-buffer convention.
    fn build_buffer_sampled(w: usize, h: usize, channels: &ChannelList) -> Vec<u8> {
        let mut buf = Vec::new();
        for y in 0..h {
            for (ci, (_, ch)) in channels.iter().enumerate() {
                if (y as i32) % ch.y_sampling != 0 {
                    continue;
                }
                let cw = w / (ch.x_sampling.max(1) as usize);
                for x in 0..cw {
                    let v = (ci as f32) * 4.0 + (y as f32) * 0.1 + x as f32 * 0.2;
                    match ch.type_ {
                        PixelType::Half => buf.extend_from_slice(&Half::from_f32(v).bits().to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&(v as u32).to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    /// DWA must support sub-sampled
    /// channels rather than reject them (C++ `compute_sampled_width`/
    /// `compute_sampled_height`, `internal_dwa_compressor.h:314-322`/`:1047-1062`).
    /// Exercises a standalone (non-CSC) LOSSY_DCT channel (`BY`, 2x2 — the
    /// standard chroma-subsampling shape), an RLE channel (`A`, 2x1), and an
    /// UNKNOWN channel (`custom`, 1x2), round-tripped through our own
    /// encoder/decoder. Self-consistency only: no C++ oracle fixture exists
    /// yet for a subsampled DWA file (would need an `oracle.cpp` addition,
    /// shared test infra outside `compressor/dwa/`).
    #[test]
    fn round_trip_subsampled_channels() {
        let w = 16usize;
        let h = 16usize;
        let mut channels = ChannelList::new();
        channels.insert("BY", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        channels.insert("A", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
        channels.insert("custom", Channel::new(PixelType::Float, 1, 2, false)).unwrap();

        let buf = build_buffer_sampled(w, h, &channels);
        let rect = rect(w as i32, h as i32);

        let chunk = compress_dwa(&channels, &buf, rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient should shrink under DWA");
        let restored = decompress_dwa(&channels, &chunk, rect).unwrap();
        assert_eq!(restored.len(), buf.len(), "packed buffer size mismatch");

        // Walk both buffers with the SAME row-active/width_c convention the
        // codec uses. RLE/UNKNOWN must be lossless; the DCT channel (BY) only
        // within a lossy bound.
        let mut boff = 0usize;
        let mut roff = 0usize;
        for y in 0..h {
            for (name, ch) in channels.iter() {
                if (y as i32) % ch.y_sampling != 0 {
                    continue;
                }
                let cw = w / (ch.x_sampling.max(1) as usize);
                let bps = crate::misc::pixel_type_size(ch.type_).unwrap();
                for x in 0..cw {
                    if name == "BY" {
                        let ov = u16::from_le_bytes([buf[boff], buf[boff + 1]]);
                        let rv = u16::from_le_bytes([restored[roff], restored[roff + 1]]);
                        let of = Half::from_bits(ov).to_f32();
                        let rf = Half::from_bits(rv).to_f32();
                        assert!((of - rf).abs() < 0.5, "BY lossy bound at (y={y},x={x}): {of} vs {rf}");
                    } else {
                        assert_eq!(
                            &buf[boff..boff + bps],
                            &restored[roff..roff + bps],
                            "{name} must round-trip losslessly at (y={y},x={x})"
                        );
                    }
                    boff += bps;
                    roff += bps;
                }
            }
        }
    }

    /// An UNKNOWN channel whose section the header zeroes out
    /// (`unknownCompressedSize == 0`) while the channel still exists (so
    /// `requiredSize > 0`) must be a corrupt chunk, not a silent zero-fill —
    /// `validate_dwa_size` ports C++'s `validate_size`
    /// (`internal_dwa_compressor.h:691-726`).
    #[test]
    fn validate_size_rejects_zeroed_unknown_section() {
        let w = 8usize;
        let h = 8usize;
        let mut channels = ChannelList::new();
        channels.insert("custom", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let buf = build_buffer(w, h, &channels);
        let rect = rect(w as i32, h as i32);

        let mut chunk = compress_dwa(&channels, &buf, rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient should shrink under DWA");
        let off = Field::UnknownCompressedSize as usize * 8;
        chunk[off..off + 8].fill(0);

        assert!(
            decompress_dwa(&channels, &chunk, rect).is_err(),
            "a zeroed UNKNOWN section with a real UNKNOWN channel present must error"
        );
    }

    /// A CSC group (R/G/B) must always be
    /// quantised through the perceptual LUT, even when a member channel's `pLinear`
    /// flag is set — C++ builds the CSC en/decoder with the table unconditionally
    /// (`LossyDctEncoderCsc_construct`/`LossyDctDecoderCsc_construct`,
    /// `internal_dwa_compressor.h:370-381`/`:1113-1126`); `pLinear` is honoured only
    /// for single non-CSC LOSSY_DCT channels (`:420`/`:1181`). Prove it by checking
    /// that `pLinear=1` on R produces the BYTE-IDENTICAL chunk to `pLinear=0` — if the
    /// LUT were skipped for the `pLinear=1` case (the pre-fix bug), the two chunks
    /// would differ.
    #[test]
    fn csc_group_ignores_p_linear_on_r() {
        let w = 16usize;
        let h = 16usize;

        let mut linear_r = ChannelList::new();
        linear_r.insert("R", Channel::new(PixelType::Half, 1, 1, true)).unwrap();
        linear_r.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        linear_r.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();

        let nonlinear_r = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);

        let buf = build_buffer(w, h, &nonlinear_r);
        let rect = rect(w as i32, h as i32);

        let chunk_p_linear = compress_dwa(&linear_r, &buf, rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient should shrink under DWA");
        let chunk_p_nonlinear = compress_dwa(&nonlinear_r, &buf, rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient should shrink under DWA");
        assert_eq!(
            chunk_p_linear, chunk_p_nonlinear,
            "R's pLinear flag must not change a CSC group's encoding"
        );

        // Decode must also agree byte-for-byte between the two channel lists.
        let restored_p_linear = decompress_dwa(&linear_r, &chunk_p_linear, rect).unwrap();
        let restored_p_nonlinear = decompress_dwa(&nonlinear_r, &chunk_p_nonlinear, rect).unwrap();
        assert_eq!(restored_p_linear, restored_p_nonlinear);
    }

    #[test]
    fn round_trip_rgb_within_lossy_bound() {
        let w = 16usize;
        let h = 16usize;
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);

        let chunk = compress_dwa(&channels, &buf, rect(w as i32, h as i32), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("RGB gradient should shrink under DWA");
        let restored = decompress_dwa(&channels, &chunk, rect(w as i32, h as i32)).unwrap();
        assert_eq!(restored.len(), buf.len());

        // Compare decoded halves to source within a generous DWA lossy bound.
        let per_chan = w * 2;
        for y in 0..h {
            for (ci, _) in channels.iter().enumerate() {
                for x in 0..w {
                    let base = y * (per_chan * 3) + ci * per_chan + x * 2;
                    let o = u16::from_le_bytes([buf[base], buf[base + 1]]);
                    let r = u16::from_le_bytes([restored[base], restored[base + 1]]);
                    let of = Half::from_bits(o).to_f32();
                    let rf = Half::from_bits(r).to_f32();
                    assert!((of - rf).abs() < 0.5, "({x},{y}) chan {ci}: {of} vs {rf}");
                }
            }
        }
    }

    /// The DC-only IDCT is what a flat block actually
    /// decodes through on the real chunk path, for both the per-channel and
    /// the CSC (RGB) decoders, and a textured block takes the full IDCT.
    /// Byte identity alone can't show this, since the full IDCT of a DC-only
    /// block gives the same pixels.
    #[test]
    fn decode_dispatches_flat_blocks_to_dc_only_idct() {
        use crate::probe::{self, Probe};
        let (w, h) = (64usize, 64usize);
        let blocks = (w / 8) * (h / 8);
        let flat = Half::from_f32(0.75).bits().to_le_bytes();
        for (names, chans) in [(&[("Y", PixelType::Half)][..], 1u64), (&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)][..], 3)] {
            let channels = cl(names);
            let decode = |buf: &[u8]| {
                let chunk = compress_dwa(&channels, buf, rect(w as i32, h as i32), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
                    .unwrap()
                    .expect("DWA must shrink this input, or the decoder never runs");
                probe::reset(Probe::DwaDcOnlyBlock);
                probe::reset(Probe::DwaFullIdctBlock);
                let restored = decompress_dwa(&channels, &chunk, rect(w as i32, h as i32)).unwrap();
                assert_eq!(restored.len(), buf.len());
                (restored, probe::get(Probe::DwaDcOnlyBlock), probe::get(Probe::DwaFullIdctBlock))
            };

            let flat_buf: Vec<u8> = flat.iter().copied().cycle().take(w * h * chans as usize * 2).collect();
            let (restored, dc_only, full) = decode(&flat_buf);
            assert_eq!((dc_only, full), (blocks as u64 * chans, 0), "{names:?}: flat blocks must all take the DC-only IDCT");
            assert_eq!(restored, flat_buf, "{names:?}: a flat image must decode exactly");

            let (_, dc_only, full) = decode(&build_buffer(w, h, &channels));
            assert!(full > 0, "{names:?}: a gradient must take the full IDCT (dc_only={dc_only})");
        }
    }

    /// The decode-side plan cache is keyed on
    /// `(channels, rules)`. `x.R/x.G/x.B` and `x.R/y.G/z.B` match the SAME
    /// R/G/B rules, so their chunks embed identical rules, but only the first
    /// forms a CSC group. Both decoded through ONE scratch must each match a
    /// fresh-scratch decode, and the plan must be rebuilt exactly when the
    /// key changes. With the old rules-only key the second decode reused the
    /// CSC plan.
    #[test]
    fn plan_cache_is_keyed_on_channels_and_rules() {
        let (w, h) = (64i32, 16i32);
        let rgb = cl(&[("x.R", PixelType::Half), ("x.G", PixelType::Half), ("x.B", PixelType::Half)]);
        let uvw = cl(&[("x.R", PixelType::Half), ("y.G", PixelType::Half), ("z.B", PixelType::Half)]);
        let encode = |channels: &ChannelList| {
            let buf = build_buffer(w as usize, h as usize, channels);
            compress_dwa(channels, &buf, rect(w, h), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
                .unwrap()
                .expect("gradient must shrink under DWA")
        };
        let (chunk_rgb, chunk_uvw) = (encode(&rgb), encode(&uvw));

        let mut scratch = DwaScratch::default();
        let mut decode = |channels: &ChannelList, chunk: &[u8]| {
            let mut out = Vec::new();
            super::decompress_dwa(channels, chunk, rect(w, h), &mut out, &mut Vec::new(), &mut scratch).unwrap();
            out
        };
        probe::reset(Probe::DwaPlanBuild);
        let rgb_1 = decode(&rgb, &chunk_rgb);
        let rgb_2 = decode(&rgb, &chunk_rgb);
        assert_eq!(probe::get(Probe::DwaPlanBuild), 1, "same (channels, rules) must reuse the plan");
        let uvw_1 = decode(&uvw, &chunk_uvw);
        assert_eq!(probe::get(Probe::DwaPlanBuild), 2, "new channel names with the same rules must rebuild the plan");

        assert!(rgb_1 == rgb_2, "a cached plan must decode identically");
        assert!(rgb_1 == decompress_dwa(&rgb, &chunk_rgb, rect(w, h)).unwrap(), "cached vs fresh plan (CSC)");
        assert!(uvw_1 == decompress_dwa(&uvw, &chunk_uvw, rect(w, h)).unwrap(), "stale CSC plan reused for ungrouped channels");
    }

    #[test]
    fn round_trip_dwab_256_lines() {
        // Exercise the DWAB block height path (a tall single-channel image).
        let w = 8usize;
        let h = 40usize;
        let channels = cl(&[("Y", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);
        let chunk = compress_dwa(&channels, &buf, rect(w as i32, h as i32), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT).unwrap();
        // Either shrunk or raw; decode must succeed when shrunk.
        if let Some(chunk) = chunk {
            let restored = decompress_dwa(&channels, &chunk, rect(w as i32, h as i32)).unwrap();
            assert_eq!(restored.len(), buf.len());
        }
    }

    #[test]
    fn tiled_dwaa_tile_taller_than_block() {
        // Regression (baseline "tiled DWAA tile taller than the DWA block (32)"):
        // a tiled DWAA tile TALLER than the 32-row DWA block must decode ALL its
        // rows.  `new_tile_compressor` used to pass a hardcoded 32 (DWAA) / 256
        // (DWAB) as the DWA scanline count instead of the tile height
        // (`numTileLines`), so a 64-row tile's `rectangle()` capped at 32 rows and
        // only half the tile decoded ("size 20480 ≠ 40960").  C++ `newTileCompressor`
        // (ImfCompressor.cpp:403-416) passes `numTileLines`; we now match it.  This
        // drives the real `new_tile_compressor` → `compress_tile`/`uncompress_tile`
        // seam — the direct `compress_dwa` round-trips above bypass `rectangle()`.
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::compressor::new_tile_compressor;
        use imath_rs::V2f;

        let w = 32i32;
        let h = 64i32; // > 32 (the DWAA block height) — the bug's trigger.
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
        ]);
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Dwaa,
        )
        .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone()))
            .unwrap();

        let buf = build_buffer(w as usize, h as usize, &channels);
        let tile_line_size = w as usize * channels.iter().count() * 2; // 3 HALF channels

        // C++ `newTileCompressor(Dwaa, tileLineSize, numTileLines=h, hdr)`.
        let mut c = new_tile_compressor(Compression::Dwaa, tile_line_size, h as usize, &hdr)
            .unwrap()
            .expect("DWAA must build a tile compressor");
        let range = rect(w, h);
        let compressed = c.compress_tile(&buf, range).unwrap().to_vec();
        let restored = c.uncompress_tile(&compressed, range, 0).unwrap();

        // Before the fix `restored.len()` was 32 rows (20480) not 64 (40960).
        assert_eq!(
            restored.len(),
            buf.len(),
            "a tile taller than the 32-row DWA block must decode all rows"
        );

        // The bottom block band (rows 56..64) must be reconstructed within the DWA
        // lossy bound — proves the extra block rows actually decode, not just that
        // the output buffer is sized correctly.
        let per_chan = w as usize * 2;
        for y in (h as usize - 8)..h as usize {
            for x in 0..w as usize {
                let base = y * (per_chan * 3) + x * 2; // R is channel 0
                let o = Half::from_bits(u16::from_le_bytes([buf[base], buf[base + 1]])).to_f32();
                let r = Half::from_bits(u16::from_le_bytes([restored[base], restored[base + 1]]))
                    .to_f32();
                assert!((o - r).abs() < 0.5, "row {y} col {x}: {o} vs {r}");
            }
        }
    }

    #[test]
    fn corrupt_input_errors_not_panics() {
        let channels = cl(&[("R", PixelType::Half)]);
        // Too short for the header.
        assert!(decompress_dwa(&channels, &[0u8; 4], rect(8, 8)).is_err());
        // Header claims huge sizes but no data follows.
        let mut bad = vec![0u8; HEADER_SIZE];
        bad[Field::AcCompressedSize as usize * 8] = 0xff;
        assert!(decompress_dwa(&channels, &bad, rect(8, 8)).is_err());
    }

    /// CSC groups are ordered by the first appearance of each layer
    /// PREFIX, not by the position of the Red channel.  Channel lists sort
    /// alphabetically, so `B,G,R,Hair.B,Hair.G,Hair.R` puts the base layer's
    /// prefix ("") first and `Hair.` second; ordering by Red would emit `Hair.`
    /// first (its R is at index 5 vs the base R at index 2 — no: alphabetical
    /// order is B,G,Hair.B,Hair.G,Hair.R,R, so the base R is LAST).  That is the
    /// bug: the layers swapped pixels in both directions across C++.
    #[test]
    fn csc_group_order_follows_first_seen_layer_prefix() {
        let channels = cl(&[
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
            ("Hair.B", PixelType::Half),
            ("Hair.G", PixelType::Half),
            ("Hair.R", PixelType::Half),
        ]);
        let names: Vec<&str> = channels.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        let c = classify_channels(&channels, classifier::default_rules()).unwrap();
        assert_eq!(c.csc_groups.len(), 2);
        // Group 0 must be the BASE layer (prefix ""), seen first at channel 0.
        assert_eq!(names[c.csc_groups[0].r], "R");
        assert_eq!(names[c.csc_groups[0].g], "G");
        assert_eq!(names[c.csc_groups[0].b], "B");
        // Group 1 is `Hair.`, whose prefix is first seen at channel 2.
        assert_eq!(names[c.csc_groups[1].r], "Hair.R");
        assert_eq!(names[c.csc_groups[1].g], "Hair.G");
        assert_eq!(names[c.csc_groups[1].b], "Hair.B");
    }

    /// CSC group order end to end: a two-layer RGB image must decode each layer back
    /// to ITS OWN values.  Before the fix the base and `Hair.` pixels swapped.
    #[test]
    fn layered_rgb_round_trip_keeps_layers_apart() {
        let w = 32usize;
        let h = 32usize;
        let names = ["B", "G", "Hair.B", "Hair.G", "Hair.R", "R"]; // channel-list order
        let channels = cl(&names.map(|n| (n, PixelType::Half)));
        // Per-channel constant, distinct per layer AND per component.
        let want = |n: &str| match n {
            "R" => 0.1f32,
            "G" => 0.2,
            "B" => 0.3,
            "Hair.R" => 0.6,
            "Hair.G" => 0.7,
            _ => 0.8,
        };
        let mut buf = Vec::new();
        for _y in 0..h {
            for (n, _) in channels.iter() {
                for _x in 0..w {
                    buf.extend_from_slice(&Half::from_f32(want(n.to_str().unwrap_or(""))).bits().to_le_bytes());
                }
            }
        }
        let r = rect(w as i32, h as i32);
        let chunk = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("flat layered image must shrink");
        let got = decompress_dwa(&channels, &chunk, r).unwrap();

        let per_chan = w * 2;
        let nch = channels.iter().count();
        for (ci, (n, _)) in channels.iter().enumerate() {
            let base = 3 * (per_chan * nch) + ci * per_chan; // row 3, first pixel
            let v = Half::from_bits(u16::from_le_bytes([got[base], got[base + 1]])).to_f32();
            assert!(
                (v - want(n.to_str().unwrap_or(""))).abs() < 0.02,
                "channel {n}: {v} should be ~{}",
                want(n.to_str().unwrap_or(""))
            );
        }
    }

    /// The ENCODER must classify with the default (case-sensitive)
    /// rules only.  `red/green/blue` therefore go down the lossless UNKNOWN
    /// path, so the round-trip is bit-exact — and real OpenEXR accepts the
    /// chunk instead of returning EXR_ERR_CORRUPT_CHUNK.
    #[test]
    fn encoder_uses_default_rules_only() {
        for set in [
            ["red", "green", "blue"].as_slice(),
            ["r", "g", "b"].as_slice(),
        ] {
            let channels = cl(&set.iter().map(|n| (*n, PixelType::Half)).collect::<Vec<_>>());
            let c = classify_channels(&channels, classifier::default_rules()).unwrap();
            assert!(
                c.csc_groups.is_empty(),
                "{set:?} must not form a CSC group on encode"
            );
            assert!(c.infos.iter().all(|i| i.scheme == CompressionScheme::Unknown));

            let (w, h) = (16usize, 16usize);
            let buf = build_buffer(w, h, &channels);
            let r = rect(w as i32, h as i32);
            if let Some(chunk) = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT).unwrap()
            {
                let got = decompress_dwa(&channels, &chunk, r).unwrap();
                assert_eq!(got, buf, "{set:?} is UNKNOWN (zlib) => lossless");
            }
        }
    }

    /// The DECODER classifies with the rules the CHUNK carries, not
    /// with our static default table.  A v2 chunk written by an encoder whose
    /// rule set matched lowercase `r/g/b` must group them as a CSC set here.
    #[test]
    fn decoder_uses_rules_stored_in_the_chunk() {
        let channels = cl(&[
            ("r", PixelType::Half),
            ("g", PixelType::Half),
            ("b", PixelType::Half),
        ]);
        // The default table does not match lowercase names ...
        let d = classify_channels(&channels, classifier::default_rules()).unwrap();
        assert!(d.csc_groups.is_empty());
        // ... but the legacy table (what a v0/v1 stream selects) does, and so
        // would any rule section a v2 chunk stores with these suffixes.
        let l = classify_channels(&channels, classifier::legacy_rules()).unwrap();
        assert_eq!(l.csc_groups.len(), 1);
        assert!(l.infos.iter().all(|i| i.scheme == CompressionScheme::LossyDct));

        // Drive the real wire path: rules serialized and parsed back must
        // classify identically to the table they came from.
        let mut wire = Vec::new();
        let upper = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
        ]);
        channel_rules::write_for_channels(&upper, &mut wire).unwrap();
        let mut off = 0usize;
        let parsed = channel_rules::read_channel_rules(&wire, &mut off).unwrap();
        assert_eq!(off, wire.len());
        let p = classify_channels(&upper, &parsed).unwrap();
        assert_eq!(p.csc_groups.len(), 1);
        assert!(p.infos.iter().all(|i| i.scheme == CompressionScheme::LossyDct));
    }

    /// Every size/count in the chunk header is attacker-controlled.
    /// Patching any of the 11 fields with a hostile value must produce a clean
    /// `Err` — never a debug overflow panic, a reversed slice range, or a
    /// terabyte allocation.  (Confirmed C++ 3.4.15 returns EXR_ERR_CORRUPT_CHUNK
    /// for each of these.)
    #[test]
    fn hostile_header_fields_error_never_panic() {
        let (w, h) = (64usize, 64usize);
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
            ("A", PixelType::Half),
        ]);
        let buf = build_buffer(w, h, &channels);
        let r = rect(w as i32, h as i32);
        let good = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient must shrink");
        // The unpatched chunk still decodes.
        assert_eq!(decompress_dwa(&channels, &good, r).unwrap().len(), buf.len());

        let hostile: [u64; 8] = [
            1,
            2,
            255,
            1 << 20,
            1 << 40,
            u64::MAX / 2,
            1u64 << 63,
            u64::MAX,
        ];
        for field in 0..NUM_HEADER_FIELDS {
            for &v in &hostile {
                let mut bad = good.clone();
                bad[field * 8..field * 8 + 8].copy_from_slice(&v.to_le_bytes());
                // Must return, one way or the other, without unwinding.
                let _ = decompress_dwa(&channels, &bad, r);
            }
        }

        // The specific fields the verifier reproduced must be hard errors.
        for (field, v, what) in [
            (Field::AcCompressedSize as usize, u64::MAX, "acCompressedSize"),
            (Field::AcUncompressedCount as usize, 1 << 40, "acCount"),
            (Field::DcUncompressedCount as usize, 1u64 << 63, "dcCount 2^63"),
            (Field::DcUncompressedCount as usize, 1 << 40, "dcCount 2^40"),
            (Field::RleUncompressedSize as usize, 1 << 40, "rleUncompressed"),
            (Field::RleRawSize as usize, 1 << 40, "rleRaw"),
            (Field::AcCompression as usize, 7, "acCompression"),
        ] {
            let mut bad = good.clone();
            bad[field * 8..field * 8 + 8].copy_from_slice(&v.to_le_bytes());
            assert!(
                decompress_dwa(&channels, &bad, r).is_err(),
                "{what} must be rejected"
            );
        }
    }

    /// Mutation-testing gap fix: the truncation guard is a 5-way OR
    /// (`compressed.len() < HEADER_SIZE+total || .. < unknown_compressed || ..
    /// < ac_compressed || .. < dc_compressed || .. < rle_compressed`). Every
    /// individual field's own comparison is provably REDUNDANT with the first
    /// (sum) term whenever that field alone is oversized — `total` sums all
    /// four fields, so any one field exceeding `compressed.len()` forces
    /// `HEADER_SIZE + total` to exceed it too. That is exactly why
    /// `hostile_header_fields_error_never_panic`'s single-field-oversized
    /// cases cannot distinguish the OR from an AND at any of the 4 internal
    /// operators: whichever field is set huge, the sum term fires anyway.
    /// The only way to expose a `||`->`&&` mutation between the sum term and
    /// an individual term is to make the SUM exceed `compressed.len()` while
    /// EVERY individual field, on its own, still fits comfortably - spread
    /// the excess across all four fields.
    #[test]
    fn decompress_dwa_rejects_sum_exceeding_even_when_every_field_individually_fits() {
        let (w, h) = (64usize, 64usize);
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half), ("A", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);
        let r = rect(w as i32, h as i32);
        let good = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient must shrink");
        assert_eq!(decompress_dwa(&channels, &good, r).unwrap().len(), buf.len(), "unpatched chunk must still decode");

        // Only touch ac/dc (the R/G/B CSC group) and rle (the standalone A
        // channel, RLE scheme by default) - `unknown_compressed` must stay 0
        // because these channels have no UNKNOWN-scheme member, and setting
        // it nonzero trips `validate_dwa_size`'s "no matching channel" error
        // for an unrelated reason before this block's own check ever runs
        // (verified: the original 4-field version of this test passed for
        // the wrong reason - it errored via that unrelated check regardless
        // of whether 484-487's operators were mutated, so it didn't actually
        // kill them).
        let avail = (good.len() - HEADER_SIZE) as u64;
        let per_field = (avail * 2) / 5; // 3 fields x 2/5 = 6/5 > 1x avail, each alone well under avail.
        let mut bad = good.clone();
        for field in [Field::AcCompressedSize, Field::DcCompressedSize, Field::RleCompressedSize] {
            let i = field as usize;
            bad[i * 8..i * 8 + 8].copy_from_slice(&per_field.to_le_bytes());
        }
        assert!(
            decompress_dwa(&channels, &bad, r).is_err(),
            "sum of declared compressed sizes ({per_field} x 3 = {}) exceeds available data \
             ({avail} bytes) even though no single field does - must still be rejected",
            per_field * 3
        );
    }

    /// Mutation-testing gap fix: hand-derived boundary values for
    /// `decompress_dwa`'s inverted/empty-rectangle guard
    /// (`rectangle.max.x < rectangle.min.x || rectangle.max.y <
    /// rectangle.min.y`). `rectangle` is `decompress_dwa`'s own parameter,
    /// independent of whatever rectangle the chunk was encoded with, so a
    /// small all-zero (maximally compressible) chunk decoded against a
    /// deliberately different rectangle exercises this guard directly.
    /// `max == min` (single row/column) is VALID, not empty/inverted -
    /// catches `<` weakened to `<=`.
    #[test]
    fn decompress_dwa_accepts_single_column_and_single_row_rectangles() {
        let channels = cl(&[("R", PixelType::Half)]);

        let col_rect = Box2i::from_min_max(V2i::new(5, 0), V2i::new(5, 511));
        let buf = vec![0u8; 512 * 2];
        let good = compress_dwa(&channels, &buf, col_rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("all-zero column must shrink");
        assert!(decompress_dwa(&channels, &good, col_rect).is_ok(), "max.x == min.x (single column) must not be treated as empty/inverted");

        let row_rect = Box2i::from_min_max(V2i::new(0, 5), V2i::new(511, 5));
        let buf = vec![0u8; 512 * 2];
        let good = compress_dwa(&channels, &buf, row_rect, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("all-zero row must shrink");
        assert!(decompress_dwa(&channels, &good, row_rect).is_ok(), "max.y == min.y (single row) must not be treated as empty/inverted");
    }

    /// Each axis inverted by 3 (not 1, so an off-by-one `==` mutant can't
    /// coincidentally still catch it) while the other axis stays valid, to
    /// isolate which term is doing the rejecting - catches `<` narrowed to
    /// `==` at either term.
    #[test]
    fn decompress_dwa_rejects_inverted_rectangle_per_axis() {
        let channels = cl(&[("R", PixelType::Half)]);
        let buf = build_buffer(64, 64, &channels);
        let r = rect(64, 64);
        let good = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient must shrink");

        let inverted_x = Box2i::from_min_max(V2i::new(10, 0), V2i::new(7, 63));
        assert!(decompress_dwa(&channels, &good, inverted_x).is_err(), "max.x < min.x must be rejected even when y is valid");

        let inverted_y = Box2i::from_min_max(V2i::new(0, 10), V2i::new(63, 7));
        assert!(decompress_dwa(&channels, &good, inverted_y).is_err(), "max.y < min.y must be rejected even when x is valid");
    }

    /// Mutation-testing gap fix: the "planar streams cannot exceed the
    /// unpacked chunk" guard (`unknown_uncompressed > uncompressed_size ||
    /// rle_raw > uncompressed_size || (unknown_uncompressed+rle_raw) >
    /// uncompressed_size`, `582:9`/`583:9`). Needs an UNKNOWN-scheme channel
    /// (`custom`, standalone Float) and an RLE-scheme channel (`A`,
    /// standalone Half) so both `unknown_uncompressed` and `rle_raw` are
    /// legitimately nonzero. `582:9`'s `||`->`&&` mutant (between the two
    /// individual terms) is EQUIVALENT and not tested here: either
    /// individual term exceeding the bound always forces the sum term to
    /// exceed it too (the other addend is always `>= 0`), so
    /// `(T1&&T2)||T3 == T1||T2||T3` identically for every reachable input -
    /// proven by exhaustive case analysis over `{T1,T2}x{true,false}` and
    /// confirmed empirically: hand-
    /// mutating `582:9` left the full 60-test suite (including this one)
    /// green. `583:9` (between the combined term and the sum term) is
    /// structurally the real gap (`T3` can be true while `T1`/`T2` are both
    /// false), but an attempt to isolate it here was masked by
    /// `validate_dwa_size`'s later, unrelated "too small" check firing
    /// instead once the guard is weakened - not yet cleanly killed; left as
    /// an open item rather than claiming a false kill.
    #[test]
    fn decompress_dwa_rejects_planar_sum_exceeding_even_when_neither_field_individually_does() {
        let mut channels = ChannelList::new();
        channels.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        channels.insert("custom", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let buf = vec![0u8; 64 * 64 * (2 + 4)]; // all-zero: guaranteed to shrink.
        let r = rect(64, 64);
        let good = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("all-zero chunk must shrink");
        assert_eq!(decompress_dwa(&channels, &good, r).unwrap().len(), buf.len(), "unpatched chunk must still decode");

        let uncompressed_size = buf.len() as u64;
        let per_field = (uncompressed_size * 3) / 5; // 2 fields x 3/5 = 6/5 > 1x, each alone well under 1x.
        let mut bad = good.clone();
        for field in [Field::UnknownUncompressedSize, Field::RleRawSize] {
            let i = field as usize;
            bad[i * 8..i * 8 + 8].copy_from_slice(&per_field.to_le_bytes());
        }
        assert!(
            decompress_dwa(&channels, &bad, r).is_err(),
            "unknown_uncompressed + rle_raw ({per_field} x 2 = {}) exceeds the unpacked chunk \
             size ({uncompressed_size}) even though neither field alone does - must be rejected",
            per_field * 2
        );
    }

    /// Mutation-testing gap fix (583:9, size-matched retry): the earlier
    /// attempt at this guard (`decompress_dwa_rejects_planar_sum_exceeding_...`,
    /// which drove real `compress_dwa` output) got masked by
    /// `validate_dwa_size`'s "too small" check firing instead, because the
    /// declared uncompressed sizes ALSO drive the real zlib buffer sizing -
    /// any header field that doesn't exactly match the true emitted stream
    /// length trips a DIFFERENT check first. This test sidesteps that by
    /// building the whole chunk BY HAND (no real `compress_dwa` call): the
    /// 581-584 guard runs strictly before any zlib inflate is attempted, so
    /// the "compressed" section bytes can be placeholder garbage as long as
    /// every size field satisfies every check that runs BEFORE it
    /// (truncation 484-490, `validate_dwa_size`'s own "too small" bound -
    /// both `unknown_uncompressed`/`rle_raw` are set ABOVE their true
    /// geometric requirement, computed independently here from the channel
    /// types/dimensions, not read back from the crate) while their SUM
    /// still exceeds the whole chunk's `uncompressed_size`.
    #[test]
    fn decompress_dwa_rejects_planar_sum_size_matched_against_specific_error() {
        let channels = cl(&[("A", PixelType::Half), ("custom", PixelType::Float)]);
        let r = rect(64, 64);
        let uncompressed_size: u64 = 64 * 64 * (2 + 4); // A(2B) + custom(4B) per pixel.
        let unknown_required: u64 = 64 * 64 * 4; // custom (Float, UNKNOWN scheme).
        let rle_required: u64 = 64 * 64 * 2; // A (Half, RLE scheme).
        let unknown_uncompressed = unknown_required + 1000; // >= required, individually < uncompressed_size.
        let rle_raw = rle_required + 1000; // same.
        assert!(unknown_uncompressed < uncompressed_size, "test setup: must not trip the individual term");
        assert!(rle_raw < uncompressed_size, "test setup: must not trip the individual term");
        assert!(unknown_uncompressed + rle_raw > uncompressed_size, "test setup: sum must exceed the chunk");

        let unknown_compressed: u64 = 10;
        let rle_compressed: u64 = 10;
        let mut header = [0u64; NUM_HEADER_FIELDS];
        header[Field::Version as usize] = 0; // legacy rules, no channel-rules table to build.
        header[Field::UnknownUncompressedSize as usize] = unknown_uncompressed;
        header[Field::UnknownCompressedSize as usize] = unknown_compressed;
        header[Field::RleCompressedSize as usize] = rle_compressed;
        header[Field::RleRawSize as usize] = rle_raw;
        header[Field::AcCompression as usize] = 0; // StaticHuffman - a valid value.

        let mut chunk = Vec::with_capacity(HEADER_SIZE + (unknown_compressed + rle_compressed) as usize);
        for field in header {
            chunk.extend_from_slice(&field.to_le_bytes());
        }
        chunk.extend(std::iter::repeat_n(0xAAu8, unknown_compressed as usize)); // unknown section (never read).
        chunk.extend(std::iter::repeat_n(0xBBu8, rle_compressed as usize)); // rle section (never read).

        match decompress_dwa(&channels, &chunk, r) {
            Err(ExrError::Input(msg)) => assert_eq!(
                msg, "DWA: planar size exceeds chunk",
                "expected the specific 581-584 error, got a different Input error: {msg}"
            ),
            other => panic!("expected Err(Input(\"DWA: planar size exceeds chunk\")), got {other:?}"),
        }
    }

    /// Mutation-testing gap fix (388:13/389:13): `classify_channels`'s
    /// CSC-group sampling-mismatch guard (C++ `internal_dwa_compressor.h:1666`)
    /// is a 4-way OR (`ri.x != gi.x || ri.x != bi.x || ri.y != gi.y || ri.y !=
    /// bi.y`) - a group forms only when ALL FOUR terms are false (fully
    /// matching sampling). Each of the 4 terms is checked in isolation (only
    /// ONE mismatches at a time) so an OR->AND mutation between any two
    /// adjacent terms is exposed: with only one term true, `||` correctly
    /// skips the group but `&&` would incorrectly let it form (masking real
    /// per-channel resampling differences into one CSC transform).
    #[test]
    fn classify_channels_rejects_csc_group_on_any_single_sampling_mismatch() {
        let cases: [(i32, i32, i32, i32, i32, i32); 4] = [
            // (r_x, r_y, g_x, g_y, b_x, b_y) - exactly one of the 4 terms true each time.
            (1, 1, 2, 1, 1, 1), // r.x != g.x only
            (1, 1, 1, 1, 2, 1), // r.x != b.x only
            (1, 1, 1, 2, 1, 1), // r.y != g.y only
            (1, 1, 1, 1, 1, 2), // r.y != b.y only
        ];
        for (i, (rx, ry, gx, gy, bx, by)) in cases.into_iter().enumerate() {
            let mut channels = ChannelList::new();
            channels.insert("R", Channel::new(PixelType::Half, rx, ry, false)).unwrap();
            channels.insert("G", Channel::new(PixelType::Half, gx, gy, false)).unwrap();
            channels.insert("B", Channel::new(PixelType::Half, bx, by, false)).unwrap();
            let c = classify_channels(&channels, classifier::default_rules()).unwrap();
            assert!(
                c.csc_groups.is_empty(),
                "case {i}: sampling mismatch (R={rx}x{ry} G={gx}x{gy} B={bx}x{by}) must prevent a CSC group from forming"
            );
        }

        // Control: identical sampling on all three must form the group.
        let mut channels = ChannelList::new();
        channels.insert("R", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        channels.insert("G", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        channels.insert("B", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        let c = classify_channels(&channels, classifier::default_rules()).unwrap();
        assert_eq!(c.csc_groups.len(), 1, "matching sampling on R/G/B must form exactly one CSC group");
    }

    /// Mutation-testing gap fix (590:29): `unknown_uncompressed >
    /// unknown_raw_total` (C++ `internal_dwa_compressor.h:876`) is the UPPER
    /// bound complementing `validate_dwa_size`'s LOWER bound on the same pair:
    /// the encoder's own exact planar byte count for the UNKNOWN-scheme
    /// channels must not be exceeded. Hand-built chunk (no real
    /// `compress_dwa`, same technique as the 583:9 fix): an R/G/B CSC group
    /// (LOSSY_DCT, zero AC/DC data supplied - unreached, this check runs
    /// before AC/DC are ever touched) supplies "slack" bytes in
    /// `uncompressed_size` that are NOT part of `unknown_raw_total` or
    /// `rle_raw_total`, so `unknown_uncompressed` can be inflated past its
    /// true requirement without also tripping the 581-584 "sum exceeds
    /// chunk" guard.
    #[test]
    fn decompress_dwa_rejects_unknown_uncompressed_exceeding_its_own_geometry() {
        let mut channels = ChannelList::new();
        channels.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        channels.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        channels.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        channels.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        channels.insert("custom", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let r = rect(64, 64);

        let unknown_raw_total: u64 = 64 * 64 * 4; // custom (Float, UNKNOWN).
        let rle_raw_total: u64 = 64 * 64 * 2; // A (Half, RLE).
        let rgb_bytes: u64 = 3 * 64 * 64 * 2; // R+G+B (Half, CSC/LOSSY_DCT) - the "slack".
        let uncompressed_size = unknown_raw_total + rle_raw_total + rgb_bytes;
        let unknown_uncompressed = unknown_raw_total + 1000; // exceeds ITS OWN requirement...
        assert!(
            unknown_uncompressed + rle_raw_total < uncompressed_size,
            "test setup: must not trip the 581-584 sum guard (has RGB slack to spare)"
        );

        let unknown_compressed: u64 = 10;
        let rle_compressed: u64 = 10;
        let mut header = [0u64; NUM_HEADER_FIELDS];
        header[Field::Version as usize] = 0;
        header[Field::UnknownUncompressedSize as usize] = unknown_uncompressed;
        header[Field::UnknownCompressedSize as usize] = unknown_compressed;
        header[Field::RleCompressedSize as usize] = rle_compressed;
        header[Field::RleRawSize as usize] = rle_raw_total;
        header[Field::AcCompression as usize] = 0;

        let mut chunk = Vec::with_capacity(HEADER_SIZE + (unknown_compressed + rle_compressed) as usize);
        for field in header {
            chunk.extend_from_slice(&field.to_le_bytes());
        }
        chunk.extend(std::iter::repeat_n(0xAAu8, unknown_compressed as usize));
        chunk.extend(std::iter::repeat_n(0xBBu8, rle_compressed as usize));

        match decompress_dwa(&channels, &chunk, r) {
            Err(ExrError::Input(msg)) => assert_eq!(
                msg, "DWA: UNKNOWN size exceeds chunk geometry",
                "expected the specific 590 error, got a different Input error: {msg}"
            ),
            other => panic!("expected Err(Input(\"DWA: UNKNOWN size exceeds chunk geometry\")), got {other:?}"),
        }
    }

    /// Mutation-testing gap fix (462:46): the generic negative-size-field
    /// guard (C++ `DwaCompressor_uncompress` :783-789) rejects any header
    /// field whose top bit is set, EXCEPT `Version` and `AcCompression`
    /// (neither is a "size"). This feeds a top-bit value into
    /// `DcCompressedSize` - a field this guard is the ONLY thing checking at
    /// the point it runs (`decompress_dwa` bails on the very first loop
    /// iteration where the guard fires, long before `classify_channels`,
    /// the truncation check, or any other size comparison ever executes) -
    /// and asserts the SPECIFIC error string, not just `is_err()`, so a
    /// mutation that narrows the guard to a different field cannot
    /// coincidentally still pass by triggering some other check instead.
    #[test]
    fn decompress_dwa_rejects_negative_size_field_with_specific_error() {
        let channels = ChannelList::new();
        let r = rect(1, 1);
        let mut header = [0u64; NUM_HEADER_FIELDS];
        header[Field::Version as usize] = 0;
        header[Field::AcCompression as usize] = 0;
        header[Field::DcCompressedSize as usize] = 1u64 << 63; // top bit set - "negative".
        let mut chunk = Vec::with_capacity(HEADER_SIZE);
        for field in header {
            chunk.extend_from_slice(&field.to_le_bytes());
        }

        match decompress_dwa(&channels, &chunk, r) {
            Err(ExrError::Input(msg)) => assert_eq!(
                msg, "DWA: negative size field",
                "expected the specific 462 error, got a different Input error: {msg}"
            ),
            other => panic!("expected Err(Input(\"DWA: negative size field\")), got {other:?}"),
        }
    }

    /// A truncated sub-stream is a corrupt chunk, not a silent zero fill —
    /// C++ passes a NULL `actual_out` to libdeflate, which fails unless the
    /// stream yields the full declared size.
    #[test]
    fn truncated_substream_is_an_error() {
        let (w, h) = (32usize, 32usize);
        let channels = cl(&[("R", PixelType::Half), ("A", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);
        let r = rect(w as i32, h as i32);
        let good = compress_dwa(&channels, &buf, r, DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("gradient must shrink");
        for cut in 1..=8 {
            let bad = &good[..good.len() - cut];
            assert!(decompress_dwa(&channels, bad, r).is_err(), "cut {cut}");
        }
    }

    /// When static-Huffman AC encoding
    /// hits the entropy-coded-bit-count ceiling, `compress_ac` must return
    /// `Ok(None)` ("store raw"), not an error — C++'s
    /// `EXR_ERR_ARGUMENT_OUT_OF_RANGE` handling
    /// (`internal_dwa_compressor.h:551-566`) abandons the whole chunk and
    /// stores it raw rather than failing the encode. Inject a tiny
    /// `huf_limit` via [`huf::compress_with_limit`] (the Huffman coder's own test seam)
    /// instead of actually building an AC stream large enough to hit
    /// `u32::MAX` bits for real.
    #[test]
    fn compress_ac_reports_none_not_error_when_huffman_overflows() {
        let all_ac: Vec<u16> = (0..64).map(|i| i as u16).collect();
        let mut level9 = new_deflate_compressor(DWA_ZLIB_LEVEL_9).unwrap();
        let mut huf_scratch = huf::HufEncScratch::default();
        let mut ac_deflate_bytes = Vec::new();
        // A real ceiling: `compress_with_limit`'s own test uses `8` to
        // force `Ok(None)` deterministically (see `huffman.rs`).
        let result = compress_ac(&all_ac, false, 8, &mut level9, &mut huf_scratch, &mut ac_deflate_bytes).unwrap();
        assert!(result.is_none(), "a Huffman stream over the bit-count ceiling must report None, not error");

        // Below the ceiling, behaves exactly like the production path.
        let result =
            compress_ac(&all_ac, false, u32::MAX as u64, &mut level9, &mut huf_scratch, &mut ac_deflate_bytes)
                .unwrap();
        assert!(result.is_some());

        // And `compress_dwa` must propagate that `None` as "store raw"
        // (`Ok(None)`), never surface it as an `Err`. We can't force the
        // real ceiling through the public API without an enormous AC stream,
        // so this only checks `compress_ac` returns cleanly for a normal
        // chunk — the propagation itself is a straight `match ... => return
        // Ok(None)` read at the call site.
        let w = 16usize;
        let h = 16usize;
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);
        assert!(compress_dwa(&channels, &buf, rect(w as i32, h as i32), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT).is_ok());
    }

    /// The header's `dwaCompressionLevel` must reach the codec.  The
    /// hardcoded 45 made every level produce byte-identical chunks (ours 2598 B
    /// at 5/45/200/2000 vs C++ 4983/2535/1731/663).
    #[test]
    fn dwa_compression_level_comes_from_the_header() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::compressor::new_compressor;
        use imath_rs::V2f;

        let (w, h) = (64i32, 64i32);
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
        ]);
        let buf = build_buffer(w as usize, 32, &channels);

        let mut sizes = Vec::new();
        for level in [5.0f32, 45.0, 200.0, 2000.0] {
            let mut hdr = Header::with_dimensions(
                w,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Dwaa,
            )
            .unwrap();
            hdr.insert("channels", ChannelListAttribute::new(channels.clone()))
                .unwrap();
            hdr.add_dwa_compression_level(level).unwrap();

            let mut c = new_compressor(Compression::Dwaa, w as usize * 3 * 2, &hdr)
                .unwrap()
                .expect("DWAA compressor");
            let n = c.compress(&buf, 0).unwrap().len();
            sizes.push(n);

            // Round-trip through the same (header-configured) compressor.
            let chunk = c.compress(&buf, 0).unwrap().to_vec();
            assert_eq!(c.uncompress(&chunk, 0, 0).unwrap().len(), buf.len());
        }
        // Higher level => coarser quantisation => strictly smaller chunks.
        assert!(
            sizes.windows(2).all(|p| p[0] > p[1]),
            "chunk sizes must shrink as the level rises, got {sizes:?}"
        );
        // A header WITHOUT the attribute must behave exactly like level 45.
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Dwaa,
        )
        .unwrap();
        hdr.insert("channels", ChannelListAttribute::new(channels.clone()))
            .unwrap();
        let mut c = new_compressor(Compression::Dwaa, w as usize * 3 * 2, &hdr)
            .unwrap()
            .expect("DWAA compressor");
        assert_eq!(c.compress(&buf, 0).unwrap().len(), sizes[1]);
    }

    /// Item 5 follow-up: `Header::set_dwa_level` (the mutable-reference-style
    /// per-header override, distinct from inserting the `dwaCompressionLevel`
    /// attribute — `dwa_compression_level_comes_from_the_header` already
    /// covers the attribute path) must ALSO reach `DwaCompressor` through the
    /// same `header.dwa_level()` seam.
    ///
    /// (The process-global default path — `Header::set_default_dwa_compression_level`
    /// — is exercised at the `Header` level, with proper cross-test locking,
    /// by `header.rs`'s own `zip_and_dwa_compression_level_override_and_default`;
    /// mutating that process-global here too, without that lock, would race
    /// against it and against every OTHER DWA test in this file that relies on
    /// the standard 45 default.)
    #[test]
    fn dwa_level_setter_also_flows_through() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::compressor::new_compressor;
        use imath_rs::V2f;

        let (w, h) = (64i32, 64i32);
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);
        let buf = build_buffer(w as usize, 32, &channels);

        let mk_header = || {
            let mut hdr = Header::with_dimensions(
                w,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Dwaa,
            )
            .unwrap();
            hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();
            hdr
        };
        let compressed_len = |hdr: &Header| {
            new_compressor(Compression::Dwaa, w as usize * 3 * 2, hdr)
                .unwrap()
                .expect("DWAA compressor")
                .compress(&buf, 0)
                .unwrap()
                .len()
        };

        let mut low = mk_header();
        low.set_dwa_level(5.0);
        let mut high = mk_header();
        high.set_dwa_level(2000.0);
        assert!(
            compressed_len(&low) > compressed_len(&high),
            "Header::set_dwa_level must drive the same quantisation as the attribute              (2000 is applied as is, as in C++, not clamped to 100)"
        );
    }

    /// The DC sub-stream's zlib level must come from `Header::zip_level()`
    /// — the SAME seam the ZIP codec uses — not the hardcoded `4`
    /// this crate used before. A CSC-free single-channel image (so the whole
    /// packed buffer is DC-and-AC LOSSY_DCT, giving the DC stream real content
    /// to compress) at zlib level 1 vs level 9 must differ.
    #[test]
    fn dc_zip_level_comes_from_the_header() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::compressor::new_compressor;
        use imath_rs::V2f;

        let (w, h) = (64i32, 64i32);
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);
        let buf = build_buffer(w as usize, 32, &channels);

        let mk_header = |zip_level: i32| {
            let mut hdr = Header::with_dimensions(
                w,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Dwaa,
            )
            .unwrap();
            hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();
            hdr.set_zip_level(zip_level);
            hdr
        };

        let mut c1 = new_compressor(Compression::Dwaa, w as usize * 3 * 2, &mk_header(1))
            .unwrap()
            .expect("DWAA compressor");
        let mut c9 = new_compressor(Compression::Dwaa, w as usize * 3 * 2, &mk_header(9))
            .unwrap()
            .expect("DWAA compressor");
        let chunk1 = c1.compress(&buf, 0).unwrap().to_vec();
        let chunk9 = c9.compress(&buf, 0).unwrap().to_vec();

        assert_ne!(
            chunk1, chunk9,
            "DC compression must vary with Header::zip_level(), it must not be hardcoded"
        );
        // Both must still decode correctly regardless of the DC zlib level used.
        assert_eq!(c1.uncompress(&chunk1, 0, 0).unwrap().len(), buf.len());
        assert_eq!(c9.uncompress(&chunk9, 0, 0).unwrap().len(), buf.len());
    }

    /// Regression: an unset zip level anywhere (no per-header override, no
    /// process-global default set — `Header::zip_level()` returns `-1`) must
    /// NOT reach `deflate_rs::Compressor::new` as a raw `-1` (it errors on
    /// negative levels). C++ `exr_compress_buffer` (`compression.c:111-117`)
    /// resolves this exact case to `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4`
    /// (`compressor::zip::DEFAULT_ZIP_LEVEL`); a compressor built from a
    /// default header must compress and round-trip identically to one built
    /// from a header with `set_zip_level(4)` explicitly.
    #[test]
    fn dc_zip_level_unset_resolves_to_the_c_plus_plus_default_not_minus_one() {
        use crate::attr::{ChannelListAttribute, Compression, LineOrder};
        use crate::compressor::new_compressor;
        use crate::compressor::zip::DEFAULT_ZIP_LEVEL;
        use imath_rs::V2f;

        let (w, h) = (64i32, 64i32);
        let channels = cl(&[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)]);
        let buf = build_buffer(w as usize, 32, &channels);

        let mk_header = |zip_level: Option<i32>| {
            let mut hdr = Header::with_dimensions(
                w,
                h,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::Dwaa,
            )
            .unwrap();
            hdr.insert("channels", ChannelListAttribute::new(channels.clone())).unwrap();
            if let Some(level) = zip_level {
                hdr.set_zip_level(level);
            }
            hdr
        };

        // Unset: Header::zip_level() == -1, previously fed straight to
        // deflate_rs and rejected outright.
        let mut c_default = new_compressor(Compression::Dwaa, w as usize * 3 * 2, &mk_header(None))
            .unwrap()
            .expect("DWAA compressor");
        let mut c_explicit_4 =
            new_compressor(Compression::Dwaa, w as usize * 3 * 2, &mk_header(Some(DEFAULT_ZIP_LEVEL)))
                .unwrap()
                .expect("DWAA compressor");

        let chunk_default = c_default.compress(&buf, 0).unwrap().to_vec();
        let chunk_explicit_4 = c_explicit_4.compress(&buf, 0).unwrap().to_vec();

        assert_eq!(
            chunk_default, chunk_explicit_4,
            "an unset zip level must resolve to DEFAULT_ZIP_LEVEL (4), matching \
             C++'s exr_compress_buffer fallback, not error or silently diverge"
        );
        assert_eq!(c_default.uncompress(&chunk_default, 0, 0).unwrap().len(), buf.len());
    }

    #[test]
    fn alpha_channel_uses_rle_roundtrip() {
        // 16x16 (not 8x8): matches the size other DWA tests (e.g.
        // `round_trip_rgb_within_lossy_bound`) use to reliably shrink under
        // DWA -- an 8x8 two-channel image was small enough that
        // `compress_dwa` could return `None` (no-shrink -> caller stores
        // raw), which made the old `if let Some(chunk)` silently skip the
        // whole assertion. `.expect()` makes that failure loud instead.
        let w = 16usize;
        let h = 16usize;
        // `cl`/`ChannelList` iterate alphabetically: "A" < "R", so channel
        // index 0 in `build_buffer`'s layout is A, index 1 is R -- the
        // OPPOSITE of what the old comment ("A is channel index 1, after R")
        // claimed. That bug meant this test actually compared R's bytes
        // (DCT-lossy) against the SOURCE and happened to pass only because
        // the assertion was skipped entirely (see above).
        let channels = cl(&[("R", PixelType::Half), ("A", PixelType::Half)]);
        let buf = build_buffer(w, h, &channels);
        let chunk = compress_dwa(&channels, &buf, rect(w as i32, h as i32), DEFAULT_COMPRESSION_LEVEL, false, DWA_ZLIB_LEVEL_DC_DEFAULT)
            .unwrap()
            .expect("R+A gradient at 16x16 should shrink under DWA");
        let restored = decompress_dwa(&channels, &chunk, rect(w as i32, h as i32)).unwrap();
        assert_eq!(restored.len(), buf.len());
        // Alpha is RLE (lossless): compare A channel bytes exactly. A is
        // channel index 0 (alphabetically first), so it occupies the FIRST
        // `per_chan` bytes of each scanline, not the second.
        let per_chan = w * 2;
        for y in 0..h {
            let base = y * (per_chan * 2); // A is channel index 0 (before R)
            for x in 0..per_chan {
                assert_eq!(restored[base + x], buf[base + x], "alpha must be lossless at row {y}");
            }
        }
    }
}
