//! `exv-codec` — transform stages + the codec registry over them.
//!
//! # What and why
//! Implementations of the `[temporal predict?] → [spatial transform] → [entropy]` chain. Depends on
//! `exv-format` (takes `CodecId`/`Dtype`) and `rans-rs` (entropy). See `docs/archive/PLAN.md` §3.3–3.4.
//!
//! # Status
//! - `LosslessFast` = stride-aware reorder+delta ([`reorder`]) → interleaved rANS.
//! - `LosslessWave` = reversible 5/3 wavelet ([`wavelet`], tile-local, integer) → rANS.
//! - Lossy: `WaveletLossy`, `Htj2k` (lossy = [`hdr`] codes coded reversibly), `BlockRt`.
//! - Dispatch is the [`registry`]: one [`registry::TileCodec`] per codec family (`backends.rs`) in a static
//!   table; [`encode_channel`]/[`decode_channel`]/[`encode_tile`]/[`decode_tile`] are thin shared fronts
//!   (dims, RAW, length guarantee) over [`Profile::backend`]. There is no `Stage` trait.
//!
//! # Errors
//! Encoding returns [`CodecError`] (a failing encoder must be loud at encode time, never a silently
//! larger or undecodable file). Decoding returns [`DecodeError`] and never panics on hostile input.
//! Decoders trust the `w`/`h` they are given (they come from a validated `Header`); the memory a decode
//! may materialize is bounded by the caller's [`exv_format::DecodeLimits`] at the frame level
//! ([`crate::frame`]), not re-guessed per channel from the input size.

use std::cell::Cell;
use std::sync::Mutex;

use exv_format::limits::alloc_zeroed;
use exv_format::{AlignedBuf, CodecId, DecodeError, Dtype, TAG_RAW};
// The blob tag scheme lives in exv-format (container format); re-exported here so that codec
// consumers (and its tests) see it via the familiar path `exv_codec::{BlobKind, classify_blob}`.
pub use exv_format::{BlobKind, classify_blob};
pub use registry::{Dst, GpuKind, OpenError, TileCodec, TileLayout, Work};
use rans::RansConfig;

/// The [`registry::TileCodec`] implementations (one per codec family); reached only via the registry.
mod backends;
/// Codec registry: static `CodecId` -> backend table, `open` (typed [`registry::OpenError`]), `GpuKind`, `TileLayout`.
pub mod registry;
/// EXV-RT realtime block codec, endpoint-lerp reference/proxy (BC6H-class, LOSSY). Not feature-gated.
pub mod blockrt;
/// EXV-RT realtime block codec, SHIPPING payload (per-block 5/3+quant, shared rANS table). Not feature-gated.
pub mod blockrt_dwt;
/// Hardware BC6H_SF16 4x4 block codec with optional lossless rANS residual.
pub mod bc6h;
pub use bc6h::Bc6hMode;
/// HTJ2K codec, one multi-component codestream per tile (JPEG2000 P15 via jph-rs). Feature-gated: it uses `jph_rs`, which is a
/// dep ONLY under `htj2k-backend` — so the module MUST be gated too, else `--no-default-features`
/// tries to compile it without `jph_rs` and fails (`cannot find crate jph_rs`).
#[cfg(feature = "htj2k-backend")]
pub mod htj2k;
/// Value-domain (log2) transform for HDR-float lossy — shared by both lossy backends.
pub mod hdr;
pub mod image;
pub mod stream;
/// Tiled per-frame codec (per-tile INTRA/SKIP/PRED) for the `.exv` video container — steps 1b/1c.
pub mod frame;
/// Temporal decorrelation (integer bit-pattern delta/undelta) — the codec-level temporal stage.
pub mod temporal;
pub mod lossy;
pub mod quantize;
pub mod reorder;
pub mod wavelet;

/// EXV-RT block-codec payload variant — the `ver` byte in `codec_params`. `Dwt` is the SHIPPING codec;
/// `Lerp` is the demoted endpoint-lerp reference/proxy (kept for comparison + as a fast lossy proxy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRtParams {
    /// `ver=0` — endpoint-lerp ([`blockrt`]): `[min:f16][max:f16][idx_bits indices]`. NOT the shipped
    /// codec (fails colour quality on real HDR); a fast fixed-rate proxy/reference mode.
    Lerp { block: u8, idx_bits: u8, log: bool },
    /// `ver=1` — SHIPPING ([`blockrt_dwt`]): per-`block`×`block` 5/3 wavelet+quant, per-block rANS under
    /// ONE shared per-plane frame table + offset index. `q` = detail quant (0 = lossless).
    Dwt { block: u8, q: u8 },
}

/// How an HTJ2K tile is coded: the wavelet AND the loss model as ONE typed value, stored on the wire as the
/// codec params (`[]` / `[1, q]` / `[2, q]`, see [`Htj2kMode::params`]).
#[cfg(feature = "htj2k-backend")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Htj2kMode {
    /// Reversible 5/3, every dtype, bit-exact (GPU-decodable through jph-wgpu).
    Lossless,
    /// FLOAT channels are mapped to [`hdr`] log2 codes at the `q`-derived scale (`htj2k_scale`) and the codes
    /// are coded with the reversible 5/3 (the loss is only the hdr step); integer channels stay lossless.
    HdrCodes { q: u8 },
    /// Irreversible 9/7 with a scalar quantizer derived from `q` (`ldr_step`). ONLY tiles whose channels are
    /// ALL U8/U16: a codestream has ONE wavelet, so it cannot mix with reversible float/U32 components, and
    /// float bit patterns must never be quantized. CPU decode; the reconstruction is rounded to the integer
    /// sample, so independent decoders may differ by +-1 LSB (FORMAT.md section 7, "decode contract").
    Irrev97 { q: u8 },
}

/// Params kind byte of [`Htj2kMode::HdrCodes`] (`[KIND_HDR, q]`).
#[cfg(feature = "htj2k-backend")]
const KIND_HDR: u8 = 1;
/// Params kind byte of [`Htj2kMode::Irrev97`] (`[KIND_IRREV97, q]`).
#[cfg(feature = "htj2k-backend")]
const KIND_IRREV97: u8 = 2;
/// Largest `q` of [`Htj2kMode::Irrev97`] (`ldr_step(40) = 2^-20`: finer than one 16-bit sample step). Not feature-gated
/// so hosts can range-check their flag in a build without the HTJ2K backend.
pub const LDR_Q_MAX: u8 = 40;
/// Largest `q` of `Transform::WaveletLossy`: the decoder shifts `1i64 << q`, so `q >= 63` would overflow
/// (`lossy::QMAX` clamps at the same value, so a forged larger `q` stays symmetric). Hosts range-check
/// their flag against it BEFORE narrowing to `u8` (`--lossy-wave 300` must not wrap to 44).
pub const WAVE_Q_MAX: u8 = 62;
/// Largest `q` that still changes the HTJ2K hdr-codes quality: `htj2k_scale(q) = clamp(16 q, 16, 256)` saturates at 16,
/// so a larger `q` is the same stream (hosts reject it instead of accepting a no-op knob).
pub const HDR_Q_MAX: u8 = 16;
/// Largest `q` of EXV-RT Dwt (`q < 16`: the F16 detail quantization width; the GPU kernel's `1 << q` relies on it).
pub const RT_Q_MAX: u8 = 15;

#[cfg(feature = "htj2k-backend")]
impl Htj2kMode {
    /// Wire form: empty = lossless, else `[kind, q]`.
    pub fn params(self) -> Vec<u8> {
        match self {
            Self::Lossless => Vec::new(),
            Self::HdrCodes { q } => vec![KIND_HDR, q],
            Self::Irrev97 { q } => vec![KIND_IRREV97, q],
        }
    }

    /// Parse [`Self::params`]; the error text names what is accepted (`registry::open` wraps it in `BadParams`).
    pub fn from_params(params: &[u8]) -> Result<Self, &'static str> {
        match params {
            [] => Ok(Self::Lossless),
            [KIND_HDR, q] => Ok(Self::HdrCodes { q: *q }),
            [KIND_IRREV97, q] if (1..=LDR_Q_MAX).contains(q) => Ok(Self::Irrev97 { q: *q }),
            [KIND_IRREV97, _] => Err("Irrev97 q must be 1..=40"),
            _ => Err("expects [] (lossless), [1, q] (hdr codes) or [2, q] (irreversible 9/7)"),
        }
    }

    /// Exactly reversible (drift-free temporal PRED, bit-exact verify, GPU-decodable).
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Lossless)
    }

    /// The wavelet of a tile that decodes at a reduced J2K resolution level straight to PIXEL values (the LOD
    /// seam); `None` for [`Self::HdrCodes`], whose codes go through the hdr inverse at full resolution only.
    pub fn lod_wavelet(self) -> Option<htj2k::Wavelet> {
        match self {
            Self::Lossless => Some(htj2k::Wavelet::Rev53),
            Self::Irrev97 { .. } => Some(htj2k::Wavelet::Irrev97),
            Self::HdrCodes { .. } => None,
        }
    }
}

/// [`Htj2kMode::Irrev97`] quality `q` -> jph's `delta` (scalar quantizer step, NORMALISED to the sample range:
/// jph's default is `2^-min(16, bit_depth)`, one LSB). `2^(-q/2)`: q=16 is one 8-bit LSB, q=32 one 16-bit LSB.
/// Encode-only: the step lives in the codestream's QCD, so decoders never need `q`.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn ldr_step(q: u8) -> f32 {
    2f32.powf(-f32::from(q) / 2.0)
}

/// Decorrelating stage (spatial). Most variants are lossless + tile-local; `WaveletLossy`/`BlockRt` are lossy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// stride-aware byte-plane split + byte-wise delta (lossless, type-agnostic).
    ReorderDelta,
    /// reversible integer 5/3 wavelet, single level, tile-local (lossless).
    Wavelet53,
    /// signed 5/3 wavelet + detail quantization to `q` bits (lossy, tile-local).
    WaveletLossy { q: u8 },
    /// HTJ2K (JPEG 2000 Part-15) codestream; the [`Htj2kMode`] picks the wavelet and the loss model
    /// (lossless 5/3, hdr log codes coded reversibly, or irreversible 9/7 for LDR integers).
    /// NOT a decorrelate→entropy stage — a complete codec handled
    /// on its own path in `encode/decode_channel` (the `rans`/`entropy` fields of `Profile` are
    /// ignored for it). Per-tile resolution levels give the LOD mip pyramid; CPU decode via
    /// `jph-rs`, GPU via `jph-wgpu`. Requires `htj2k-backend`.
    #[cfg(feature = "htj2k-backend")]
    Htj2k(Htj2kMode),
    /// EXV-RT realtime block codec (`CodecId::EXV_RT`, blob `TAG_EXV_RT`, LOSSY). Payload variant in
    /// [`BlockRtParams`]: `Dwt` = the SHIPPING codec, `Lerp` = the demoted reference/proxy. Own path in
    /// `encode/decode_channel` (rANS/entropy fields unused). NOT feature-gated.
    BlockRt(BlockRtParams),
    /// Hardware `BC6H_SF16` 4x4 block codec (`CodecId::BC6H`, blob `TAG_BC6H`), in either [`Bc6hMode::Lossy`]
    /// or [`Bc6hMode::LosslessResidual`] (BC6H base + rANS 16-bit residual).
    Bc6h(Bc6hMode),
    // Reserved (lossy, Phase 6+): quantized wavelet (9/7 / quant 5/3).
    // DCT is NOT planned: wavelet also covers lossy with the same family of transforms (no legacy).
    // TemporalDelta — lives in exv-video as a separate stage.
}

/// Entropy-backend selection policy — **encode-time only** (decode reads the per-blob tag, so this is
/// never stored in the container). Resolves the GPU-first tension between ratio and GPU-decodability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EntropyPolicy {
    /// **GPU-first (default).** Prefer rANS (GPU-decodable) on any blob where it shrinks; deflate is
    /// computed only to rescue a blob rANS can't compress; raw otherwise. Keeps the GPU decode path
    /// reachable — without this, deflate's LZ wins almost every blob and decode is CPU-only.
    #[default]
    GpuHot,
    /// **Cold / archive max-ratio.** Pick the smaller of rANS/deflate per blob (deflate usually wins
    /// → the tile decodes on the CPU). Use when file size matters more than GPU decode speed.
    MaxRatio,
}

/// Codec profile: transform + entropy backend + entropy policy. Self-describing via
/// [`Profile::codec_id`] (the policy is NOT part of the on-disk identity — it only steers encoding).
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub transform: Transform,
    pub rans: RansConfig,
    pub entropy: EntropyPolicy,
}

impl Profile {
    /// Fast lossless profile (hot GPU path): reorder+delta → rANS, GPU-hot entropy policy.
    pub fn lossless_fast() -> Self {
        Self { transform: Transform::ReorderDelta, rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// Higher-compression lossless profile (reversible wavelet → rANS), GPU-hot entropy policy.
    pub fn lossless_wave() -> Self {
        Self { transform: Transform::Wavelet53, rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// Lossy profile: signed 5/3 wavelet + detail quantization to `q` bits (rANS-only entropy).
    pub fn lossy_wave(q: u8) -> Self {
        Self { transform: Transform::WaveletLossy { q }, rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// HTJ2K **lossless** (reversible 5/3) profile. The `rans`/`entropy` fields are unused
    /// (HTJ2K carries its own entropy) but kept default so `Profile` stays one uniform struct.
    #[cfg(feature = "htj2k-backend")]
    pub fn htj2k() -> Self {
        Self { transform: Transform::Htj2k(Htj2kMode::Lossless), rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// HTJ2K **lossy** profile; `q` is the quality knob (larger = finer hdr codes =
    /// higher quality/larger; near-lossless at q>=16, see `htj2k_scale`). Float channels go through
    /// the [`hdr`] log codes then reversible HTJ2K; integer channels stay lossless.
    #[cfg(feature = "htj2k-backend")]
    pub fn htj2k_lossy(q: u8) -> Self {
        Self { transform: Transform::Htj2k(Htj2kMode::HdrCodes { q }), rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// HTJ2K **irreversible 9/7** profile for all-U8/U16 (LDR integer) images ([`Htj2kMode::Irrev97`]); `q` in
    /// `1..=LDR_Q_MAX` (larger = finer/larger, see `ldr_step`). Encoding a tile with any other dtype is a
    /// [`CodecError::Dtype`] error, never a silent fallback.
    #[cfg(feature = "htj2k-backend")]
    pub fn htj2k_ldr(q: u8) -> Self {
        Self { transform: Transform::Htj2k(Htj2kMode::Irrev97 { q }), rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// EXV-RT endpoint-lerp profile (`ver=0` reference/proxy). `block`×`block` squares, `idx_bits`
    /// indices/texel, `log` = HDR log2 endpoint space. rANS/entropy fields unused (block codec is complete).
    ///
    /// PRECONDITION (mirrors the registry `open`/CLI guards): `block >= 1`, `idx_bits ∈ 1..=8` — else
    /// `encode_channel` would hit `div_ceil(0)` / a zero index range. `debug_assert`ed (caught in tests);
    /// real ingest (`registry::open`, CLI `--rt`) validates before calling, so release code never sees a bad value.
    pub fn exv_rt(block: u8, idx_bits: u8, log: bool) -> Self {
        debug_assert!(block >= 1 && (1..=8).contains(&idx_bits), "exv_rt: block>=1, idx_bits∈1..=8");
        let t = Transform::BlockRt(BlockRtParams::Lerp { block, idx_bits, log });
        Self { transform: t, rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// EXV-RT **shipping** profile (`ver=1`): per-`block`×`block` 5/3 wavelet + detail quant `q`
    /// (0 = lossless), per-block rANS under one shared per-plane table. The realtime GPU-decodable codec.
    ///
    /// PRECONDITION (mirrors the registry `open`/CLI guards): `block ∈ 1..=64` (GPU dim cap; `div_ceil(0)` else),
    /// `q <= RT_Q_MAX` (F16 detail quant width). `debug_assert`ed; real ingest validates before calling.
    pub fn exv_rt_dwt(block: u8, q: u8) -> Self {
        debug_assert!((1..=64).contains(&block) && q <= RT_Q_MAX, "exv_rt_dwt: block∈1..=64, q<16");
        let t = Transform::BlockRt(BlockRtParams::Dwt { block, q });
        Self { transform: t, rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// Hardware `BC6H_SF16` 4x4 block profile (lossy for F16, raw lossless for non-F16).
    pub fn bc6h_lossy() -> Self {
        Self { transform: Transform::Bc6h(Bc6hMode::Lossy), rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// Hardware `BC6H_SF16` 4x4 block base + rANS 16-bit residual profile (100% bit-exact lossless).
    pub fn bc6h_lossless() -> Self {
        Self { transform: Transform::Bc6h(Bc6hMode::LosslessResidual), rans: RansConfig::default(), entropy: EntropyPolicy::GpuHot }
    }

    /// Switch this profile to the cold/archive max-ratio entropy policy ([`EntropyPolicy::MaxRatio`]).
    pub fn max_ratio(mut self) -> Self {
        self.entropy = EntropyPolicy::MaxRatio;
        self
    }

    /// Whether the transform is exactly reversible (lossless). Temporal PRED (open-loop bit-pattern
    /// residual add-back) is only drift-free for a lossless codec, so a lossy profile must keep its
    /// P-frame tiles INTRA/SKIP — the closed-loop lossy-temporal path is future work (PLAN §3.7).
    pub fn is_lossless(&self) -> bool {
        match self.transform {
            Transform::ReorderDelta | Transform::Wavelet53 => true,
            #[cfg(feature = "htj2k-backend")]
            Transform::Htj2k(m) => m.is_lossless(),
            Transform::Bc6h(m) => m.is_lossless(),
            Transform::WaveletLossy { .. } | Transform::BlockRt(_) => false,
        }
    }

    /// Codec params for StageDesc (stored in the header): `[q]` for lossy, empty otherwise.
    pub fn params(&self) -> Vec<u8> {
        match self.transform {
            Transform::WaveletLossy { q } => vec![q],
            // EXV-RT: [ver, block, {idx_bits|q}, flags]. ver=0 Lerp (flags bit0=log); ver=1 Dwt.
            Transform::BlockRt(BlockRtParams::Lerp { block, idx_bits, log }) => vec![0, block, idx_bits, log as u8],
            Transform::BlockRt(BlockRtParams::Dwt { block, q }) => vec![1, block, q, 0],
            // HTJ2K: `[]` lossless, `[kind, q]` otherwise (see `Htj2kMode::params`).
            #[cfg(feature = "htj2k-backend")]
            Transform::Htj2k(m) => m.params(),
            Transform::Bc6h(m) => m.params(),
            _ => Vec::new(),
        }
    }

    /// Stable backend identifier written into the container.
    pub fn codec_id(&self) -> CodecId {
        self.backend().id()
    }

    /// The registry backend serving this profile. A TOTAL exhaustive `match` (so no lookup can fail and the
    /// compiler forces a new [`Transform`] variant to name its backend); a test pins that the result is the
    /// table entry for [`Self::codec_id`].
    pub fn backend(&self) -> &'static dyn TileCodec {
        match self.transform {
            Transform::ReorderDelta => &backends::FAST,
            Transform::Wavelet53 => &backends::WAVE,
            Transform::WaveletLossy { .. } => &backends::LOSSY_WAVE,
            Transform::BlockRt(_) => &backends::EXV_RT,
            #[cfg(feature = "htj2k-backend")]
            Transform::Htj2k(_) => &backends::HTJ2K,
            Transform::Bc6h(_) => &backends::BC6H,
        }
    }

    /// GPU decode route of this profile (`None` = CPU only). The single capability table for exv-gpu / exv-view.
    pub fn gpu_kind(&self) -> Option<GpuKind> {
        self.backend().gpu_kind(self)
    }
}

/// raw-DEFLATE level for the alternative entropy backend. 6 = speed/ratio balance.
/// libdeflate: the win over EXR-ZIP comes from LZ itself (not level 9 vs 6), and level 6 encodes
/// several times faster (encode runs deflate on EVERY blob — speed is critical). raw-DEFLATE
/// without the zlib wrapper — we own the framing, saving 6 header/trailer bytes per blob.
const DEFLATE_LEVEL: i32 = 6;

/// Why an encode failed. Encoders must be loud: a silent fallback (RAW tile, empty blob) turns a codec
/// bug into a 3-10x larger or undecodable file that still "verifies".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Plane/tile geometry is inconsistent: zero dims, a length that is not `w*h*esz`, a channel/dtype
    /// count mismatch, or a size that does not fit the on-disk integer width.
    Dims(&'static str),
    /// A codec/profile parameter is out of range.
    Param(&'static str),
    /// The codec backend (jph-rs / DEFLATE) failed or panicked; the message carries its diagnostic.
    Backend(String),
    /// The profile cannot code a channel of this dtype (e.g. the HTJ2K 9/7 profile takes U8/U16 only).
    Dtype { codec: &'static str, dtype: Dtype },
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dims(m) => write!(f, "bad dimensions: {m}"),
            Self::Param(m) => write!(f, "bad parameter: {m}"),
            Self::Backend(m) => write!(f, "codec backend failed: {m}"),
            Self::Dtype { codec, dtype } => write!(f, "codec `{codec}` cannot code {dtype:?} channels"),
        }
    }
}

impl std::error::Error for CodecError {}

/// The entropy coder refused the block (oversized plane / lane overflow): a hard encode error, never a
/// silently truncated stream.
impl From<rans::EncodeError> for CodecError {
    fn from(e: rans::EncodeError) -> Self {
        CodecError::Backend(e.to_string())
    }
}

/// Lets `io::Result` writers (the streaming frame-body writer) propagate an encode failure with `?`.
impl From<CodecError> for std::io::Error {
    fn from(e: CodecError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}

/// Capability gate for the GPU HTJ2K decoders (jph-wgpu / `exv-gpu` atlas): can a component of `dtype` be
/// decoded on the GPU? U8/U16 are plain UNSIGNED 8/16-bit J2K components (CPU-verified, [`htj2k`]); the GPU
/// pack paths interpret every 16-bit signed component as an IEEE HALF (jph-wgpu's RGBA16F pack has no BF16
/// unpack: a BF16 file would display silently wrong numbers) and route U32 to CPU planes. So only F16, F32 and
/// U32 pass, and a tile holding any other channel must take the CPU decode — one predicate here instead of
/// per-site dtype lists, and consumers return a hard error / pick the CPU path rather than ever decoding such
/// a tile on the GPU.
pub const fn htj2k_gpu_decodable(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F16 | Dtype::F32 | Dtype::U32)
}

/// Exact byte length of a `w×h` plane of `dtype`; `None` on usize overflow. The single definition of
/// "a plane" used by every encode/decode length check.
pub(crate) fn plane_len(w: u32, h: u32, dtype: Dtype) -> Option<usize> {
    (w as usize).checked_mul(h as usize)?.checked_mul(dtype.size())
}

/// Maximum expansion of any DEFLATE stream (RFC 1951: 258-byte match / 2 bits, rounded up).
const DEFLATE_MAX_RATIO: usize = 1032;

/// Compress decorrelated bytes with raw-DEFLATE ([`DEFLATE_LEVEL`]).
pub(crate) fn deflate_encode(decorrelated: &[u8]) -> Result<Vec<u8>, CodecError> {
    let mut c = deflate_rs::Compressor::new(DEFLATE_LEVEL).map_err(|e| CodecError::Backend(format!("deflate: {e:?}")))?;
    Ok(c.deflate_compress_vec(decorrelated))
}

/// Decompress a raw-DEFLATE payload to exactly `expected` bytes. `None` on a corrupt stream / wrong length.
///
/// DEFLATE cannot expand more than 1032:1 (a 258-byte match costs >= 2 bits), so an `expected` beyond that
/// bound is provably corrupt and is refused BEFORE the decompressor reserves `expected` bytes up front - a
/// 2-byte blob must not be able to demand a plane-sized (or, from a forged size, address-space-sized) buffer.
pub(crate) fn deflate_decode(body: &[u8], expected: usize) -> Option<Vec<u8>> {
    if !deflate_plausible(body.len(), expected) {
        return None;
    }
    let mut d = deflate_rs::Decompressor::new();
    d.deflate_decompress_vec(body, expected).ok()
}

/// `false` when a `body_len`-byte DEFLATE payload provably cannot expand to `expected` bytes (the 1032:1 bound).
fn deflate_plausible(body_len: usize, expected: usize) -> bool {
    expected <= body_len.saturating_mul(DEFLATE_MAX_RATIO).saturating_add(DEFLATE_MAX_RATIO)
}

/// APPEND the smallest candidate to `out` as a blob `[tag][payload]`; if none is smaller than the raw data —
/// append the raw data (`TAG_RAW`), capping expansion at +1 byte (the no-shrink rule of all OpenEXR codecs).
/// On equal lengths the first candidate wins (rANS comes first → we preserve GPU-decodability).
pub(crate) fn pick_into(out: &mut Vec<u8>, plane: &[u8], cands: &[(u8, &[u8])]) {
    match cands.iter().min_by_key(|c| c.1.len()) {
        Some(&(tag, body)) if body.len() < plane.len() => {
            out.reserve(body.len() + 1);
            out.push(tag);
            out.extend_from_slice(body);
        }
        _ => {
            out.reserve(plane.len() + 1);
            out.push(TAG_RAW);
            out.extend_from_slice(plane);
        }
    }
}

/// [`pick_into`] for a codec with ONE candidate that already wrote `[tag][payload]` at `out[start..]` (so the
/// payload was never copied through a temporary `Vec`): keep it iff the payload is strictly smaller than the
/// plane, otherwise replace it by the RAW blob.
pub(crate) fn keep_or_raw(out: &mut Vec<u8>, start: usize, plane: &[u8]) {
    if out.len().saturating_sub(start + 1) < plane.len() {
        return;
    }
    out.truncate(start);
    out.push(TAG_RAW);
    out.extend_from_slice(plane);
}

thread_local! {
    /// The per-thread decode/encode scratch of [`with_bytes`] (taken while in use, so nesting just allocates).
    static BYTES: Cell<AlignedBuf> = const { Cell::new(AlignedBuf::new()) };
    /// The per-thread decoded-channel scratch of [`with_tile_bytes`]: a separate cell because the codec running
    /// inside it takes [`BYTES`] for its own decorrelated stream.
    static TILE: Cell<AlignedBuf> = const { Cell::new(AlignedBuf::new()) };
    /// The per-thread growable encode buffer of [`with_vec`].
    static GROW: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

/// Largest scratch a thread keeps between calls (a bigger one-off tile is freed again).
const BYTES_KEEP: usize = 4 << 20;

/// Run `f` over the reusable buffer of `cell`, EXACTLY `len` bytes with UNSPECIFIED contents (callers must
/// overwrite every byte they read). Fallible, so a hostile `len` is an `Err`. The buffer is an [`AlignedBuf`]: the
/// scratch planes (the decorrelated stream, a decoded channel on its way into plane rows) start on a cache line, so
/// the codec's wide loads/stores and the row copies out of it never split one at the base.
fn with_scratch<T>(
    cell: &'static std::thread::LocalKey<Cell<AlignedBuf>>,
    len: usize,
    f: impl FnOnce(&mut [u8]) -> T,
) -> Result<T, std::collections::TryReserveError> {
    let mut buf = cell.with(Cell::take);
    buf.try_fit(len)?; // no clearing: only fresh lines are zero, a smaller request reuses the high-water allocation
    let out = f(&mut buf);
    if buf.capacity() <= BYTES_KEEP {
        cell.with(|c| c.set(buf));
    }
    Ok(out)
}
/// [`with_scratch`] over the per-thread codec scratch. The decorrelated stream between the entropy decoder and
/// `join_undelta` / the inverse wavelet, and the one the encoders build before rANS, is a plane-sized temporary on
/// every tile of every channel; a fresh zeroed `Vec` per call was a memset + page-fault tax.
pub(crate) fn with_bytes<T>(len: usize, f: impl FnOnce(&mut [u8]) -> T) -> Result<T, std::collections::TryReserveError> {
    with_scratch(&BYTES, len, f)
}

/// Run `f` over a reusable per-thread EMPTY `Vec` it may grow (an encoder's intermediate byte stream whose length is
/// only known once built: the lossy varint stream, the EXV-RT pooled block codes). A `Vec` per channel per tile was
/// most of those encoders' allocation churn (lossy-wave 5.1x, EXV-RT 3.1x raw per tile, docs/OPT.md G3).
pub(crate) fn with_vec<T>(f: impl FnOnce(&mut Vec<u8>) -> T) -> T {
    let mut buf = GROW.with(Cell::take);
    buf.clear();
    let out = f(&mut buf);
    if buf.capacity() <= BYTES_KEEP {
        GROW.with(|c| c.set(buf));
    }
    out
}

/// [`with_scratch`] over the per-thread decoded-channel scratch: a per-channel decoder's contiguous output on its way
/// into a strided [`Dst::Rows`] destination.
pub(crate) fn with_tile_bytes<T>(len: usize, f: impl FnOnce(&mut [u8]) -> T) -> Result<T, std::collections::TryReserveError> {
    with_scratch(&TILE, len, f)
}

/// HTJ2K lossy quality knob `q` -> hdr codes-per-stop (`scale`). Higher q = finer codes = higher
/// quality (and a larger codestream); q>=16 -> scale 256 ≈ near-lossless (the only loss is the hdr
/// log step, the codes themselves are coded losslessly). Both encode and decode derive scale from
/// the stored q, so they agree. Integer scale: see [`hdr`] (deterministic table math).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn htj2k_scale(q: u8) -> u32 {
    (q as u32 * 16).clamp(16, 256)
}

/// Encode a single channel (planar bytes, size `w*h*dtype.size()`) with a profile.
///
/// `w`/`h` are needed by the wavelet stage (2D shape); `ReorderDelta` ignores them.
/// Guarantee: `decode_channel(encode_channel(p, d, w, h, prof), d, w, h, prof) == p` (lossless profiles).
///
/// # Errors
/// [`CodecError::Dims`] when `plane.len() != w*h*esz`; [`CodecError::Backend`] when the entropy coder / HTJ2K
/// backend fails (e.g. a plane over the rANS symbol limit — a hard error, never a silently undecodable
/// blob). The only deliberate RAW fallbacks are the generic no-shrink rule and the documented EXV-RT rule (a lossy
/// block profile stores non-F16 channels raw so ids/ints stay exact).
pub fn encode_channel(plane: &[u8], dtype: Dtype, w: u32, h: u32, profile: &Profile) -> Result<Vec<u8>, CodecError> {
    let mut out = Vec::new();
    encode_channel_into(plane, dtype, w, h, profile, &mut out)?;
    Ok(out)
}

/// Run `f`, which APPENDS to `out`; on `Err` cut `out` back to its length at entry, so a failed encode leaves no
/// half-written blob behind (the contract of every `*_into` encoder).
fn rollback<E>(out: &mut Vec<u8>, f: impl FnOnce(&mut Vec<u8>) -> Result<(), E>) -> Result<(), E> {
    let start = out.len();
    let res = f(out);
    if res.is_err() {
        out.truncate(start);
    }
    res
}

/// [`encode_channel`] APPENDING the blob to `out` (no blob `Vec` of its own: the tile encoder frames channels
/// into one buffer). On `Err`, `out` is unchanged.
pub fn encode_channel_into(plane: &[u8], dtype: Dtype, w: u32, h: u32, profile: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
    // One length contract for every codec: otherwise a mis-sized plane is silently zero-padded by a
    // transform ("lossless" garbage) or indexes out of bounds inside it.
    if plane_len(w, h, dtype) != Some(plane.len()) {
        return Err(CodecError::Dims("plane length != w*h*esz"));
    }
    rollback(out, |out| profile.backend().encode_chan(plane, dtype, w, h, profile, out))
}

/// The tile area every decode entry point accepts: `Header::read` already caps the effective tile at
/// [`exv_format::MAX_TILE_AREA`], and the public fronts re-check it so a caller handing them an unvalidated
/// `w`/`h` (u32::MAX squared used to overflow an allocation inside the deflate decoder) gets a typed error
/// instead of a panic - "never panics" must hold for the signature, not only for validated callers.
fn check_tile_dims(w: u32, h: u32) -> Result<(), DecodeError> {
    if u64::from(w) * u64::from(h) > exv_format::MAX_TILE_AREA {
        return Err(DecodeError::Request("tile area exceeds the format's MAX_TILE_AREA"));
    }
    Ok(())
}

/// Decode a channel encoded by [`encode_channel`]. On success the result is EXACTLY `w*h*esz` bytes — the
/// one length guarantee callers may rely on (they must not re-check it). `w`/`h` must come from a validated
/// header; the memory bound is the caller's ([`exv_format::DecodeLimits`] at the frame level).
///
/// # Errors
/// [`DecodeError::Corrupt`] on a corrupt, truncated or profile-mismatching stream (never a panic),
/// [`DecodeError::Request`] when `w*h*esz` overflows.
pub fn decode_channel(data: &[u8], dtype: Dtype, w: u32, h: u32, profile: &Profile) -> Result<Vec<u8>, DecodeError> {
    let kind = classify_blob(data).ok_or(DecodeError::Corrupt("empty or unknown blob tag"))?;
    check_tile_dims(w, h)?;
    // Checked product (CLASS-A: a forged w*h cannot overflow usize).
    let expected = plane_len(w, h, dtype).ok_or(DecodeError::Request("plane size overflows"))?;
    // The plane is allocated up front (fallibly), so refuse what is provably wrong BEFORE it exists: a RAW blob of
    // the wrong length, a DEFLATE payload too short to ever expand to a plane. (Other codecs cannot be bounded by
    // the blob's size - a blank 20000² tile is a few hundred bytes - so, as for every free front, the memory bound
    // is the caller's: `FrameReader` budgets it ([`exv_format::DecodeLimits`]).)
    match kind {
        BlobKind::Raw(rest) if rest.len() != expected => return Err(DecodeError::Corrupt("raw blob length != plane")),
        BlobKind::Deflate(b) if !deflate_plausible(b.len(), expected) => return Err(DecodeError::Corrupt("deflate payload")),
        _ => {}
    }
    let mut out = alloc_zeroed(expected)?;
    decode_channel_into(data, dtype, w, h, profile, &mut out)?;
    Ok(out)
}

/// [`decode_channel`] into the caller's plane, which MUST be exactly `w*h*esz` bytes (else
/// [`DecodeError::Request`]). Every byte is overwritten on success; on `Err` the contents are unspecified. The
/// hot tile path decodes straight into recycled tile buffers through this, so a decode allocates nothing
/// plane-sized of its own beyond the codec's declared working set ([`TileCodec::work`]).
///
/// # Errors
/// As [`decode_channel`].
pub fn decode_channel_into(data: &[u8], dtype: Dtype, w: u32, h: u32, profile: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
    let kind = classify_blob(data).ok_or(DecodeError::Corrupt("empty or unknown blob tag"))?;
    check_tile_dims(w, h)?;
    // Checked product (CLASS-A: a forged w*h cannot overflow usize).
    let expected = plane_len(w, h, dtype).ok_or(DecodeError::Request("plane size overflows"))?;
    if out.len() != expected {
        return Err(DecodeError::Request("output plane length != w*h*esz"));
    }
    if let BlobKind::Raw(rest) = kind {
        // no-shrink: raw uncompressed plane; must carry exactly `expected` bytes.
        return if rest.len() == expected {
            out.copy_from_slice(rest);
            Ok(())
        } else {
            Err(DecodeError::Corrupt("raw blob length != plane"))
        };
    }
    // The backend decodes into exactly `out.len() == expected` bytes or fails: the length guarantee is the type.
    profile.backend().decode_chan(kind, dtype, w, h, profile, out)
}

// ---------------------------------------------------------------------------
// Tile-level codec (format v2) — the per-tile blob entry points. Two mutually exclusive layouts
// ([`TileLayout`], owned by the backend; no third compatibility shim):
//   * PerChannel (rANS family, EXV-RT): `[chan_len[nch]:u32 LE][ch0_blob]…` — each channel an `encode_channel` blob.
//   * MultiComp (HTJ2K): `[TAG_HTJ2K][one multi-component codestream]` — SIZ carries the component count, NO
//     chan_len table; the tag prefix keeps `classify_blob` and the GPU/residency scan working.
// ---------------------------------------------------------------------------

/// Encode one tile's per-channel raw bytes into the self-describing tile blob.
/// `channels[ci]` is `w*h*dtypes[ci].size()` bytes; `w`/`h` are the tile rect.
/// HTJ2K → `[TAG_HTJ2K][mc codestream]`; per-channel codecs →
/// `[chan_len[nch]][per-channel encode_channel]`. Pure + deterministic. Errors as [`encode_channel`].
pub fn encode_tile(channels: &[&[u8]], dtypes: &[Dtype], w: u32, h: u32, profile: &Profile) -> Result<Vec<u8>, CodecError> {
    let mut out = Vec::new();
    encode_tile_into(channels, dtypes, w, h, profile, &mut out)?;
    Ok(out)
}

/// [`encode_tile`] APPENDING the blob to `out` (a caller can reuse one buffer across tiles). On `Err`, `out` is unchanged.
pub fn encode_tile_into(channels: &[&[u8]], dtypes: &[Dtype], w: u32, h: u32, profile: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
    if channels.len() != dtypes.len() {
        return Err(CodecError::Dims("channel count != dtype count"));
    }
    rollback(out, |out| profile.backend().encode_tile(channels, dtypes, w, h, profile, out))
}

/// Decode one tile blob (from [`encode_tile`]) into per-channel tile planes — each EXACTLY
/// `w*h*dtypes[ci].size()` bytes (callers must not re-check). HTJ2K → one multi-component decode of
/// the codestream; per-channel codecs → split + `decode_channel` per sub-blob. `w`/`h` must come from a
/// validated header (see [`decode_channel`]).
///
/// # Errors
/// [`DecodeError::Corrupt`] on a corrupt blob (never panics), [`DecodeError::Unsupported`] for a blob tag
/// the profile's codec cannot carry.
pub fn decode_tile(blob: &[u8], dtypes: &[Dtype], w: u32, h: u32, profile: &Profile) -> Result<Vec<Vec<u8>>, DecodeError> {
    check_tile_dims(w, h)?;
    let mut planes = alloc_planes(dtypes, w, h)?;
    let mut out: Vec<Dst<'_>> = planes.iter_mut().map(|p| Dst::Plane(p)).collect();
    decode_tile_dst(blob, dtypes, w, h, profile, &mut out)?;
    drop(out);
    Ok(planes)
}

/// Zero-filled `Vec` planes of a `w`x`h` tile of `dtypes` (fallible): the destination of the allocating fronts, which
/// decode straight into the `Vec`s they return (no aligned buffer + copy).
///
/// # Errors
/// [`DecodeError::Request`] when a plane size overflows, [`DecodeError::Alloc`] when the allocator refuses.
pub(crate) fn alloc_planes(dtypes: &[Dtype], w: u32, h: u32) -> Result<Vec<Vec<u8>>, DecodeError> {
    dtypes.iter().map(|&dt| Ok(alloc_zeroed(plane_len(w, h, dt).ok_or(DecodeError::Request("plane size overflows"))?)?)).collect()
}

/// Recycled per-channel tile planes: the destination of [`decode_tile_into`]. One buffer is `fit` to each tile it
/// decodes and handed back to a pool afterwards, so a whole-frame decode allocates (and zero-fills) a few
/// `TileBuf`s per window instead of one `Vec` per channel per tile. Planes are [`AlignedBuf`]s: a 256^2 F16 tile plane
/// (128 KiB) from the system heap is only 16-byte aligned on Windows; the codec output and the row copies out of it
/// (`FrameReader::decode_all_into`) start on a cache line.
#[derive(Default)]
pub struct TileBuf {
    planes: Vec<AlignedBuf>,
}

impl TileBuf {
    /// Size the buffer to `dtypes`' planes of a `w`x`h` tile (fallibly; a plane keeps its capacity across tiles).
    /// Contents are UNSPECIFIED: the decoder overwrites every byte or fails.
    ///
    /// # Errors
    /// [`DecodeError::Request`] when a plane size overflows, [`DecodeError::Alloc`] when the allocator refuses.
    pub fn fit(&mut self, dtypes: &[Dtype], w: u32, h: u32) -> Result<(), DecodeError> {
        self.fit_some(dtypes, w, h, |_| true)
    }

    /// [`Self::fit`] for the channels `want(ci)` only; the others become EMPTY planes (capacity kept), so a subset
    /// decode never sizes (or zero-fills) an unwanted channel.
    ///
    /// # Errors
    /// As [`Self::fit`].
    pub fn fit_some(&mut self, dtypes: &[Dtype], w: u32, h: u32, want: impl Fn(usize) -> bool) -> Result<(), DecodeError> {
        self.planes.resize_with(dtypes.len(), AlignedBuf::new);
        for (ci, (p, &dt)) in self.planes.iter_mut().zip(dtypes).enumerate() {
            // An unwanted channel becomes empty (capacity kept); a wanted one is sized without clearing.
            let len = if want(ci) { plane_len(w, h, dt).ok_or(DecodeError::Request("plane size overflows"))? } else { 0 };
            p.try_fit(len).map_err(|_| DecodeError::Alloc(len as u64))?;
        }
        Ok(())
    }

    /// Heap held by the planes (their capacities).
    fn bytes(&self) -> usize {
        self.planes.iter().map(AlignedBuf::capacity).sum()
    }

    /// Channel `ci`'s tile plane (empty for a bad index).
    pub fn plane(&self, ci: usize) -> &[u8] {
        self.planes.get(ci).map_or(&[], AlignedBuf::as_bytes)
    }

    /// The planes as the mutable slices [`decode_tile_into`] fills.
    pub fn planes_mut(&mut self) -> Vec<&mut [u8]> {
        self.planes.iter_mut().map(AlignedBuf::as_bytes_mut).collect()
    }
}

/// Bytes the shared stash of [`LaneBuf`] keeps while idle, in total over ALL threads (not per worker). A lane
/// `TileBuf` is ~1 MiB at 256^2 x 5 channels, so this holds ~32 of them: a whole frame's lanes on a 24-thread box;
/// a bigger lane buffer (huge tiles / many AOVs) is simply freed again.
const LANE_KEEP: usize = 32 << 20;

/// The idle lane buffers (see [`LaneBuf`]).
static LANES: Mutex<Vec<TileBuf>> = Mutex::new(Vec::new());

/// A lane's [`TileBuf`] on loan from a bounded process-wide stash: the whole-frame decode of a per-channel codec
/// decodes each tile into its lane's buffer before copying the rows out, and a fresh buffer per lane per frame was
/// ~0.1x raw of allocation + first-touch churn (docs/OPT.md R). Taken on construction (`Default`), returned on drop.
///
/// Why not a per-thread scratch (tried, REJECTED: it stayed resident on every rayon worker and broke `mem_gate`): the
/// stash is ONE bound for the whole process ([`LANE_KEEP`]), independent of the thread count, and a lane only ever
/// holds what it borrowed, so the in-flight window the caller budgeted ([`FrameReader`](crate::frame::FrameReader)'s
/// `tile_cost` per lane) is unchanged. A buffer is not returned when the lane unwinds from a panic.
pub(crate) struct LaneBuf(TileBuf);

impl LaneBuf {
    /// The stash, tolerating poison (its contents are plain buffers with unspecified bytes).
    fn stash() -> std::sync::MutexGuard<'static, Vec<TileBuf>> {
        LANES.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for LaneBuf {
    /// The most recently returned buffer (warm in cache), else an empty one that grows on its first `fit`.
    fn default() -> Self {
        Self(Self::stash().pop().unwrap_or_default())
    }
}

impl std::ops::Deref for LaneBuf {
    type Target = TileBuf;
    fn deref(&self) -> &TileBuf {
        &self.0
    }
}

impl std::ops::DerefMut for LaneBuf {
    fn deref_mut(&mut self) -> &mut TileBuf {
        &mut self.0
    }
}

impl Drop for LaneBuf {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let buf = std::mem::take(&mut self.0);
        if buf.bytes() == 0 {
            return;
        }
        let mut stash = Self::stash();
        if stash.iter().map(TileBuf::bytes).sum::<usize>() + buf.bytes() <= LANE_KEEP {
            stash.push(buf);
        }
    }
}

/// [`decode_tile`] into `buf`, which is [`TileBuf::fit`] to the tile first (so the planes are exactly
/// `w*h*dtypes[ci].size()`). Decoders write every byte or fail; on `Err` the contents are unspecified.
///
/// # Errors
/// As [`decode_tile`], plus [`DecodeError::Alloc`] from sizing `buf`.
pub fn decode_tile_into(blob: &[u8], dtypes: &[Dtype], w: u32, h: u32, profile: &Profile, buf: &mut TileBuf) -> Result<(), DecodeError> {
    check_tile_dims(w, h)?;
    buf.fit(dtypes, w, h)?;
    let mut out: Vec<Dst<'_>> = buf.planes_mut().into_iter().map(Dst::Plane).collect();
    decode_tile_dst(blob, dtypes, w, h, profile, &mut out)
}

/// THE tile decode front: decode one tile blob into one [`Dst`] per channel (a tile plane, rows of a larger plane,
/// or [`Dst::Skip`] for an unwanted channel). Every non-skipped destination is fully written or the call fails
/// (contents unspecified on `Err`).
///
/// # Errors
/// [`DecodeError::Request`] when `out` does not match `dtypes` / the `w`x`h` shape, otherwise as [`decode_tile`].
pub fn decode_tile_dst(blob: &[u8], dtypes: &[Dtype], w: u32, h: u32, profile: &Profile, out: &mut [Dst<'_>]) -> Result<(), DecodeError> {
    let _s = tracing::info_span!("decode_tile", nch = dtypes.len(), w, h).entered();
    check_tile_dims(w, h)?;
    if out.len() != dtypes.len() || !out.iter().zip(dtypes).all(|(d, dt)| d.fits(w as usize, h as usize, dt.size())) {
        return Err(DecodeError::Request("tile destinations do not match the channels / tile shape"));
    }
    profile.backend().decode_tile(blob, dtypes, w, h, profile, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The idle stash of [`LaneBuf`] never exceeds [`LANE_KEEP`], however many lanes return at once and however big
    /// one is, and a returned buffer comes back out (invariants hold under the parallel test harness).
    #[test]
    fn lane_stash_is_bounded() {
        let dts = [Dtype::F32; 4];
        let mut lanes: Vec<LaneBuf> = (0..48).map(|_| LaneBuf::default()).collect();
        for l in &mut lanes {
            l.fit(&dts, 256, 256).expect("fit"); // 1 MiB each: 48 MiB offered vs a 32 MiB cap
        }
        lanes.push(LaneBuf::default());
        if let Some(l) = lanes.last_mut() {
            l.fit(&dts, 2048, 2048).expect("fit"); // 64 MiB: over the cap on its own, never kept
        }
        drop(lanes);
        let kept: usize = LaneBuf::stash().iter().map(TileBuf::bytes).sum();
        assert!(kept <= LANE_KEEP, "stash holds {kept} B > {LANE_KEEP}");
        let mut again = LaneBuf::default();
        again.fit(&dts, 256, 256).expect("refit");
        assert_eq!(again.plane(0).len(), 256 * 256 * 4);
    }
    use exv_format::{TAG_DEFLATE, TAG_EXV_RT, TAG_RANS};

    fn round_trip(plane: &[u8], dtype: Dtype, w: u32, h: u32, prof: &Profile) {
        let enc = encode_channel(plane, dtype, w, h, prof).expect("encode");
        let dec = decode_channel(&enc, dtype, w, h, prof).expect("decode");
        assert_eq!(dec, plane, "round-trip {dtype:?} via {:?}", prof.transform);
    }

    #[test]
    fn roundtrip_all_dtypes_both_profiles() {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let u8d: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let u16d: Vec<u8> = (0..n).flat_map(|i| (i as u16).to_le_bytes()).collect();
        let u32d: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
        let f32d: Vec<u8> = (0..n).flat_map(|i| (i as f32 * 0.01).to_le_bytes()).collect();
        for prof in [Profile::lossless_fast(), Profile::lossless_wave()] {
            round_trip(&u8d, Dtype::U8, w, h, &prof);
            round_trip(&u16d, Dtype::U16, w, h, &prof);
            round_trip(&u16d, Dtype::F16, w, h, &prof);
            round_trip(&u32d, Dtype::U32, w, h, &prof);
            round_trip(&f32d, Dtype::F32, w, h, &prof);
        }
    }

    /// HTJ2K profile: every dtype (F16/BF16/F32/U32 and, as plain unsigned components, U8/U16) codes as a
    /// real HTJ2K codestream and round-trips bit-exact through the container blob framing.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn roundtrip_htj2k_profile() {
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let u8d: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();
        let u16d: Vec<u8> = (0..n).flat_map(|i| ((i * 37) as u16).to_le_bytes()).collect();
        let u32d: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
        let f32d: Vec<u8> = (0..n).flat_map(|i| (i as f32 * 0.01).to_le_bytes()).collect();
        let prof = Profile::htj2k();
        round_trip(&u16d, Dtype::F16, w, h, &prof); // HTJ2K codestream
        round_trip(&u16d, Dtype::Bf16, w, h, &prof); // HTJ2K codestream
        round_trip(&f32d, Dtype::F32, w, h, &prof); // HTJ2K codestream
        round_trip(&u32d, Dtype::U32, w, h, &prof); // HTJ2K codestream
        round_trip(&u8d, Dtype::U8, w, h, &prof);
        round_trip(&u16d, Dtype::U16, w, h, &prof);
        // The tile path packs mixed dtypes into ONE multi-component codestream.
        let dts = [Dtype::F16, Dtype::U8, Dtype::U16, Dtype::U32, Dtype::F32];
        let chans: Vec<&[u8]> = vec![&u16d, &u8d, &u16d, &u32d, &f32d];
        let blob = encode_tile(&chans, &dts, w, h, &prof).expect("mixed tile");
        assert_eq!(decode_tile(&blob, &dts, w, h, &prof).expect("mixed decode"), chans.iter().map(|c| c.to_vec()).collect::<Vec<_>>());
        assert!(matches!(encode_tile(&[&u16d], &[Dtype::F16, Dtype::F16], w, h, &prof), Err(CodecError::Dims(_))));
    }

    /// Every codec rejects a mis-sized plane at encode time (it used to be zero-padded or index OOB),
    /// and `registry::open` rejects trailing/extra params.
    #[test]
    fn encode_rejects_bad_dims_and_params() {
        for prof in [Profile::lossless_fast(), Profile::lossless_wave(), Profile::lossy_wave(3), Profile::exv_rt_dwt(16, 2)] {
            let r = encode_channel(&[0u8; 31], Dtype::F16, 4, 4, &prof);
            assert!(matches!(r, Err(CodecError::Dims(_))), "{:?}", prof.transform);
        }
        assert!(crate::registry::open(CodecId::WAVELET53_LOSSY, &[]).is_err());
        assert!(crate::registry::open(CodecId::WAVELET53_LOSSY, &[1, 2]).is_err());
        #[cfg(feature = "htj2k-backend")]
        assert!(crate::registry::open(CodecId::HTJ2K, &[8]).is_err(), "bare [q] is the removed pre-Htj2kMode form");
        // Public constructors are only debug-asserted; a release encode must still fail cleanly.
        let bad = Profile { transform: Transform::BlockRt(BlockRtParams::Dwt { block: 0, q: 0 }), ..Profile::lossless_fast() };
        assert!(matches!(encode_channel(&[0u8; 32], Dtype::F16, 4, 4, &bad), Err(CodecError::Param(_))));
    }

    /// B3: `decode_channel` guarantees `len == w*h*esz` — a valid-but-short rANS stream (0 symbols) is
    /// `None`, not an empty "plane"; a forged huge `w*h` is rejected without allocating.
    #[test]
    fn decode_channel_length_guarantee_and_bomb_guard() {
        let prof = Profile::lossless_fast();
        let mut short = vec![TAG_RANS];
        short.extend_from_slice(&rans::encode(&[], prof.rans).unwrap());
        assert!(decode_channel(&short, Dtype::U8, 16, 16, &prof).is_err(), "short stream must not pass");
        let plane = vec![9u8; 64 * 64];
        let ok = encode_channel(&plane, Dtype::U8, 64, 64, &prof).unwrap();
        assert_eq!(classify_blob(&ok).map(|k| matches!(k, BlobKind::Rans(_))), Some(true), "constant plane must be rANS");
        assert_eq!(decode_channel(&ok, Dtype::U8, 64, 64, &prof), Ok(plane));
        // 65535² U8 plane from a ~600-byte stream: the stream's own symbol count (4096) != the plane, so it
        // is corrupt — rejected after decoding 4096 symbols, no 4 GiB buffer is ever requested.
        assert!(decode_channel(&ok, Dtype::U8, 65535, 65535, &prof).is_err());
        assert!(decode_channel(&ok, Dtype::U8, u32::MAX, u32::MAX, &prof).is_err());
    }

    /// B5: an EXV-RT Dwt blob whose self-described block/q disagree with the profile is rejected (the
    /// GPU reads the profile, so CPU and GPU must never decode a tampered blob differently).
    #[test]
    fn dwt_blob_must_match_profile() {
        let (w, h) = (64u32, 64u32);
        let f16d: Vec<u8> = (0..w * h).flat_map(|i| (0x3c00u16 + ((i % 24) as u16)).to_le_bytes()).collect();
        let prof = Profile::exv_rt_dwt(16, 0);
        let enc = encode_channel(&f16d, Dtype::F16, w, h, &prof).unwrap();
        assert_eq!(enc.first(), Some(&TAG_EXV_RT));
        assert!(decode_channel(&enc, Dtype::F16, w, h, &prof).is_ok());
        assert!(decode_channel(&enc, Dtype::F16, w, h, &Profile::exv_rt_dwt(16, 1)).is_err(), "q mismatch");
        assert!(decode_channel(&enc, Dtype::F16, w, h, &Profile::exv_rt_dwt(8, 0)).is_err(), "block mismatch");
    }

    /// EXV-RT identity (B-RT.1): the profile round-trips through `codec_id`/`params`; an F16 channel is
    /// coded by the block codec (TAG_EXV_RT) and the container path reproduces `blockrt::decode_plane`
    /// bit-for-bit; every non-F16 dtype falls back to RAW and round-trips LOSSLESSLY (a lossy profile
    /// keeps ids/ints exact). Broader float ingest (F32/BF16 → f16) is B-RT.2. NOT feature-gated.
    #[test]
    fn roundtrip_exv_rt_profile() {
        let (w, h) = (40u32, 24u32);
        let n = (w * h) as usize;
        // Arbitrary FINITE f16 bit patterns (mask clears sign+high exponent → never inf/NaN).
        let f16d: Vec<u8> = (0..n).flat_map(|i| ((i as u16) & 0x3fff).to_le_bytes()).collect();
        let u32d: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
        let u8d: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();

        let prof = Profile::exv_rt(4, 4, true);
        // Profile identity: reconstruct from the on-disk (CodecId, params) blob.
        assert_eq!(prof.codec_id(), CodecId::EXV_RT);
        let rt = crate::registry::open(prof.codec_id(), &prof.params()).expect("open");
        assert_eq!(rt.transform, Transform::BlockRt(BlockRtParams::Lerp { block: 4, idx_bits: 4, log: true }));
        // Guard: idx_bits=0 / block=0 must be rejected (else decode divides by zero).
        assert!(crate::registry::open(CodecId::EXV_RT, &[0, 4, 0, 1]).is_err());
        assert!(crate::registry::open(CodecId::EXV_RT, &[0, 0, 4, 1]).is_err());
        assert!(crate::registry::open(CodecId::EXV_RT, &[9, 4, 4, 1]).is_err()); // bad version

        // F16 → block codec: tagged TAG_EXV_RT, and the container decode == the raw blockrt decode.
        let enc = encode_channel(&f16d, Dtype::F16, w, h, &prof).expect("encode");
        assert_eq!(enc.first(), Some(&TAG_EXV_RT), "F16 must pick the block codec");
        let dec = decode_channel(&enc, Dtype::F16, w, h, &prof).expect("decode f16");
        let raw = blockrt::encode_plane_mode(&f16d, w, h, 4, 4, true);
        assert_eq!(dec, blockrt::decode_plane_mode(&raw, w, h, 4, 4, true), "container == raw blockrt");
        assert_eq!(dec.len(), f16d.len());

        // Non-F16 → RAW, LOSSLESS.
        round_trip(&u32d, Dtype::U32, w, h, &prof);
        round_trip(&u8d, Dtype::U8, w, h, &prof);
    }

    /// EXV-RT SHIPPING (Dwt, ver=1) identity: profile round-trips codec_id/params; a COMPRESSIBLE F16
    /// plane routes through the block codec (TAG_EXV_RT) and, at q=0 (lossless lifting), decodes
    /// bit-exact through the container; non-F16 → RAW. Exercises the real blockrt_dwt encode/decode path.
    #[test]
    fn roundtrip_exv_rt_dwt_profile() {
        let (w, h) = (64u32, 64u32);
        // Smooth gradient of finite f16 values just above 1.0 (0x3c00..) → compresses well below raw,
        // so `pick` chooses the block codec (not the RAW fallback) and the Dwt path is truly exercised.
        let f16d: Vec<u8> = (0..w * h)
            .flat_map(|i| (0x3c00u16 + ((i % 24) as u16)).to_le_bytes())
            .collect();
        let u32d: Vec<u8> = (0..(w * h) as usize).flat_map(|i| (i as u32).to_le_bytes()).collect();

        let prof = Profile::exv_rt_dwt(16, 0); // q=0 = lossless lifting
        assert_eq!(prof.codec_id(), CodecId::EXV_RT);
        let rt = crate::registry::open(prof.codec_id(), &prof.params()).expect("open");
        assert_eq!(rt.transform, Transform::BlockRt(BlockRtParams::Dwt { block: 16, q: 0 }));
        assert!(crate::registry::open(CodecId::EXV_RT, &[1, 16, 16, 0]).is_err()); // q=16 invalid (F16)
        assert!(crate::registry::open(CodecId::EXV_RT, &[1, 0, 4, 0]).is_err()); // block=0 invalid

        let enc = encode_channel(&f16d, Dtype::F16, w, h, &prof).expect("encode");
        assert_eq!(enc.first(), Some(&TAG_EXV_RT), "compressible F16 must pick the Dwt block codec");
        let dec = decode_channel(&enc, Dtype::F16, w, h, &prof).expect("decode");
        assert_eq!(dec, f16d, "Dwt q=0 lossless round-trip through the container");

        round_trip(&u32d, Dtype::U32, w, h, &prof); // non-F16 → RAW
    }

    /// A supported-dtype HTJ2K blob carries TAG_HTJ2K, so the GPU path (jph-wgpu) can route it.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_blob_is_tagged() {
        let (w, h) = (32u32, 32u32);
        let f32d: Vec<u8> = (0..(w * h) as usize).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect();
        let enc = encode_channel(&f32d, Dtype::F32, w, h, &Profile::htj2k()).expect("encode");
        assert!(
            matches!(classify_blob(&enc), Some(BlobKind::Htj2k(_))),
            "a supported-dtype HTJ2K blob must carry TAG_HTJ2K"
        );
    }

    /// The 9/7 profile end to end through the shared fronts: per-channel and per-tile blobs are HTJ2K codestreams,
    /// U8/U16 come back within the documented +-1 LSB class at a fine q, the profile is neither lossless nor
    /// GPU-decodable, its params are `[2, q]`, and any float/U32 channel is a typed `Dtype` error (never quantized).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_ldr_profile_paths() {
        let (w, h) = (48u32, 40u32);
        let ramp = |dt: Dtype| -> Vec<u8> {
            (0..w * h)
                .flat_map(|i| {
                    let v = (i % w) * 255 / w + (i / w) % 3;
                    if dt == Dtype::U8 { vec![v as u8] } else { ((v * 257) as u16).to_le_bytes().to_vec() }
                })
                .collect()
        };
        let prof = Profile::htj2k_ldr(32);
        assert_eq!((prof.params(), prof.is_lossless(), prof.gpu_kind()), (vec![2, 32], false, None));
        let (p8, p16) = (ramp(Dtype::U8), ramp(Dtype::U16));
        let worst = |a: &[u8], b: &[u8], dt: Dtype| {
            let v = |p: &[u8], i: usize| if dt == Dtype::U8 { i32::from(p[i]) } else { i32::from(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])) };
            (0..a.len() / dt.size()).map(|i| (v(a, i) - v(b, i)).abs()).max().unwrap_or(0)
        };
        for (plane, dt) in [(&p8, Dtype::U8), (&p16, Dtype::U16)] {
            let enc = encode_channel(plane, dt, w, h, &prof).expect("encode");
            let dec = decode_channel(&enc, dt, w, h, &prof).expect("decode");
            assert!(worst(plane, &dec, dt) <= 1, "{dt:?} channel");
        }
        let (chans, dts) = ([p8.as_slice(), p16.as_slice()], [Dtype::U8, Dtype::U16]);
        let tile = encode_tile(&chans, &dts, w, h, &prof).expect("tile");
        assert!(matches!(classify_blob(&tile), Some(BlobKind::Htj2k(_))));
        let back = decode_tile(&tile, &dts, w, h, &prof).expect("decode tile");
        assert!(worst(&p8, &back[0], Dtype::U8) <= 1 && worst(&p16, &back[1], Dtype::U16) <= 1, "tile");
        // A lossless profile cannot decode a 9/7 stream (wavelet pinned by the header check), and vice versa.
        assert!(decode_tile(&tile, &dts, w, h, &Profile::htj2k()).is_err());
        let lossless = encode_tile(&chans, &dts, w, h, &Profile::htj2k()).expect("lossless tile");
        assert!(decode_tile(&lossless, &dts, w, h, &prof).is_err());
        for dt in [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::U32] {
            let plane = vec![0u8; (w * h) as usize * dt.size()];
            let want = CodecError::Dtype { codec: "HTJ2K 9/7", dtype: dt };
            assert_eq!(encode_channel(&plane, dt, w, h, &prof).unwrap_err(), want);
            assert_eq!(encode_tile(&[&p8, &plane], &[Dtype::U8, dt], w, h, &prof).unwrap_err(), want);
        }
    }

    /// HTJ2K lossy on HDR float now routes through the value-domain log codes (hdr.rs), so the error
    /// is bounded RELATIVE (was ~2900x with naive bit-reinterpret quant). 5-decade ramp, q=50.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn roundtrip_htj2k_lossy() {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        // log-uniform HDR ramp 0.01 .. 1000 (5 decades).
        let f32d: Vec<u8> = (0..n)
            .flat_map(|i| (0.01f32 * 10f32.powf(5.0 * i as f32 / (n as f32 - 1.0))).to_le_bytes())
            .collect();
        // q is the quality knob (higher = finer = better; near-lossless at q>=16). Error must be
        // bounded and must shrink as q grows; the bit-reinterpret catastrophe (~2900x) is gone.
        let mut prev = f32::INFINITY;
        for q in [1u8, 4, 16] {
            let prof = Profile::htj2k_lossy(q);
            let enc = encode_channel(&f32d, Dtype::F32, w, h, &prof).expect("encode");
            assert!(matches!(classify_blob(&enc), Some(BlobKind::Htj2k(_))), "lossy must produce an HTJ2K codestream");
            let dec = decode_channel(&enc, Dtype::F32, w, h, &prof).expect("decode");
            assert_eq!(dec.len(), f32d.len(), "lossy decode length");
            let max_rel = f32d
                .as_chunks::<4>().0.iter()
                .zip(dec.as_chunks::<4>().0.iter())
                .map(|(a, b)| {
                    let av = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let bv = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (av - bv).abs() / av.abs()
                })
                .fold(0.0f32, f32::max);
            assert!(max_rel < 0.5, "q={q}: htj2k lossy relative error {max_rel} out of bound");
            assert!(max_rel <= prev + 1e-3, "q={q}: error {max_rel} did not shrink with finer q (prev {prev})");
            prev = max_rel;
        }
        assert!(prev < 0.05, "htj2k lossy at q=16 should be near-lossless, got {prev}");
    }

    /// Build a mixed-dtype multi-channel image (R,G,B:F16 + Z:F32 + id:U32) of `w`×`h` with `tile`²
    /// tiles — the realistic exv HTJ2K shape (all jph-supported dtypes). Used by the mc tests.
    #[cfg(feature = "htj2k-backend")]
    fn mc_image(w: u32, h: u32, tile: u32) -> crate::image::Image {
        use exv_format::{Channel, Header};
        let n = (w * h) as usize;
        // F16 smooth-ish gradients (different per channel), F32 depth ramp, U32 distinct ids.
        let f16 = |k: u32| -> Vec<u8> {
            (0..n).flat_map(|i| (((i as u32 * 3 + k * 101) & 0x3ff) as u16).to_le_bytes()).collect::<Vec<u8>>()
        };
        let z: Vec<u8> = (0..n).flat_map(|i| (i as f32 * 0.25 + 1.0).to_le_bytes()).collect();
        let id: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: tile,
            tile_h: tile,
            codec_id: 6,
            codec_params: Vec::new(),
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "G".into(), dtype: Dtype::F16 },
                Channel { name: "B".into(), dtype: Dtype::F16 },
                Channel { name: "Z".into(), dtype: Dtype::F32 },
                Channel { name: "id".into(), dtype: Dtype::U32 },
            ],
        };
        crate::image::Image::new(header, vec![f16(0), f16(1), f16(2), z, id]).expect("valid mc image")
    }

    /// HTJ2K mc LOSSLESS: a mixed-dtype image round-trips bit-exact through `write_still`/`read_still`,
    /// AND every tile blob is ONE multi-component codestream (`TAG_HTJ2K`, `num_components == nch`),
    /// not `nch` per-channel sub-blobs.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_mc_lossless_roundtrip() {
        // Lib-internal test -> in-crate frame path (the `exv_video` dev-dep can't be mixed with
        // `crate::` types across the dependency cycle).
        use crate::frame::{FrameReader, read_frame_tiles, write_frame_tiles};
        let prof = Profile::htj2k();
        let img = mc_image(128, 96, 64); // 2x2 tiles
        let nch = img.header.channels.len();
        let hdr = {
            let mut h = img.header.clone();
            h.codec_id = prof.codec_id().0;
            h.codec_params = prof.params();
            h.frame_count = 1;
            h
        };
        let body = write_frame_tiles(&img.planes, None, &hdr, &prof).expect("encode");
        let back = read_frame_tiles(&body, &hdr, &prof, None, &exv_format::DecodeLimits::default()).expect("read");
        assert_eq!(back, img.planes, "htj2k mc lossless round-trip must be bit-exact");

        // Each tile blob is ONE multi-component codestream: TAG_HTJ2K + a codestream whose SIZ carries
        // all `nch` components (proves it is NOT the per-channel `[chan_len][subs]` framing).
        let reader = FrameReader::open(&body, hdr.clone()).expect("open");
        for t in 0..reader.tile_count() {
            let blob = reader.tile_blob(t).expect("tile blob");
            match classify_blob(blob) {
                Some(BlobKind::Htj2k(payload)) => {
                    assert_eq!(
                        htj2k::num_components(payload),
                        Some(nch),
                        "tile {t}: one mc codestream with all {nch} components"
                    );
                }
                other => panic!("tile {t}: expected a TAG_HTJ2K mc blob, got {other:?}"),
            }
            // The per-channel single-blob accessor must refuse an mc tile.
            assert!(matches!(reader.tile_channel_blob(t, 0), Err(DecodeError::Request(_))), "mc tile has no per-channel sub-blob");
        }
    }

    /// HTJ2K mc LOSSY: float channels decode within a bounded relative error (and improve with finer
    /// `q`); the integer id channel is coded REVERSIBLE so it must be bit-exact (ids never quantized).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_mc_lossy_roundtrip() {
        use crate::frame::{read_frame_tiles, write_frame_tiles};
        let img = mc_image(128, 96, 64);
        let id_ci = 4usize; // U32 id channel
        let mut prev = f32::INFINITY;
        for q in [1u8, 4, 16] {
            let prof = Profile::htj2k_lossy(q);
            let hdr = {
                let mut h = img.header.clone();
                h.codec_id = prof.codec_id().0;
                h.codec_params = prof.params();
                h.frame_count = 1;
                h
            };
            let body = write_frame_tiles(&img.planes, None, &hdr, &prof).expect("encode");
            let back = read_frame_tiles(&body, &hdr, &prof, None, &exv_format::DecodeLimits::default()).expect("read lossy");
            assert_eq!(back.len(), img.planes.len());
            // Integer id channel: reversible -> bit-exact even on the lossy profile.
            assert_eq!(back[id_ci], img.planes[id_ci], "q={q}: U32 ids must be lossless");
            // Z (F32) channel relative error, bounded and shrinking with finer q.
            let z_ci = 3usize;
            let max_rel = img.planes[z_ci]
                .as_chunks::<4>().0.iter()
                .zip(back[z_ci].as_chunks::<4>().0.iter())
                .map(|(a, b)| {
                    let av = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let bv = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (av - bv).abs() / av.abs()
                })
                .fold(0.0f32, f32::max);
            assert!(max_rel < 0.5, "q={q}: mc lossy F32 relative error {max_rel} out of bound");
            assert!(max_rel <= prev + 1e-3, "q={q}: error {max_rel} did not shrink (prev {prev})");
            prev = max_rel;
        }
        assert!(prev < 0.05, "mc lossy at q=16 should be near-lossless, got {prev}");
    }

    /// HTJ2K mc tile + LOD: the per-tile mc decode (`decode_tile`) of every tile reassembles to the whole-frame
    /// decode (one mc decode per tile), and `decode_tile_channel_level(.,.,1)` returns the reduced (ceil dim/2) extent
    /// for an mc tile, while level 0 reproduces the full-res channel bytes.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_mc_region_and_level() {
        // In-crate frame path (this is a lib-internal unit test — mixing `crate::` types with the
        // `exv_video` dev-dep would clash, since the cycle builds two exv-codec instances).
        use crate::frame::{FrameReader, read_frame_tiles, write_frame_tiles};
        let prof = Profile::htj2k();
        let img = mc_image(160, 128, 64); // 3x2 tiles (incl. a partial right column)
        // Bake the profile's codec into the header so FrameReader reconstructs it (the body is header-less).
        let hdr = {
            let mut h = img.header.clone();
            h.codec_id = prof.codec_id().0;
            h.codec_params = prof.params();
            h.frame_count = 1;
            h
        };
        let body = write_frame_tiles(&img.planes, None, &hdr, &prof).expect("encode");
        let full = read_frame_tiles(&body, &hdr, &prof, None, &exv_format::DecodeLimits::default()).expect("read");
        let reader = FrameReader::open(&body, hdr.clone()).expect("open");

        // Every tile's mc decode scatters back to exactly the whole-frame decode planes.
        let mut whole: Vec<Vec<u8>> = full.iter().map(|p| vec![0u8; p.len()]).collect();
        for t in 0..reader.tile_count() {
            let (x, y, tw, th) = reader.tile_rect(t).expect("rect");
            let tiles = reader.decode_tile(t, &prof, None).expect("mc tile");
            for (ci, plane) in whole.iter_mut().enumerate() {
                let esz = reader.header().channels[ci].dtype.size();
                for r in 0..th as usize {
                    let (src, dst) = (r * tw as usize * esz, ((y as usize + r) * 160 + x as usize) * esz);
                    plane[dst..dst + tw as usize * esz].copy_from_slice(&tiles[ci][src..src + tw as usize * esz]);
                }
            }
        }
        assert_eq!(whole, full, "mc per-tile decode == whole-frame decode");

        // LOD on tile 0: level 0 == full channel bytes; level 1 == ceil(dim/2) extent.
        let (_, _, tw, th) = reader.header().tile_rect(0, 0);
        for ci in 0..reader.header().channels.len() {
            let full_ch = reader.decode_tile_channel(0, ci).expect("full tile/channel");
            let (l0, rw0, rh0) = reader.decode_tile_channel_level(0, ci, 0).expect("lod level0");
            assert_eq!((rw0, rh0), (tw, th), "mc level0 dims == tile rect (ch {ci})");
            assert_eq!(l0, full_ch, "mc level0 bytes == decode_tile_channel (ch {ci})");
            let (_, rw1, rh1) = reader.decode_tile_channel_level(0, ci, 1).expect("lod level1");
            assert_eq!((rw1, rh1), (tw.div_ceil(2), th.div_ceil(2)), "mc level1 dims = ceil(dim/2) (ch {ci})");
        }
    }

    /// A forged wavelet blob with wrong dimensions → None, NOT a panic in inverse_2d.
    #[test]
    fn decode_channel_rejects_wrong_size_wavelet() {
        let prof = Profile::lossless_wave();
        // compressible 64x64 (tag=1 coded, not no-shrink raw); decode as 128x128 → wavelet size-guard → None
        let plane: Vec<u8> = (0..64u32 * 64).flat_map(|i| ((i % 256) as u16).to_le_bytes()).collect();
        let blob = encode_channel(&plane, Dtype::U16, 64, 64, &prof).expect("encode");
        assert!(decode_channel(&blob, Dtype::U16, 128, 128, &prof).is_err());
    }

    /// No-shrink: incompressible data does not expand by more than +1 byte (not a +512 table).
    #[test]
    fn no_shrink_caps_expansion() {
        let mut seed = 0x00C0_FFEEu64;
        let plane: Vec<u8> = (0..1000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed & 0xff) as u8
            })
            .collect();
        let prof = Profile::lossless_fast();
        let blob = encode_channel(&plane, Dtype::U8, 1000, 1, &prof).expect("encode");
        assert!(blob.len() <= plane.len() + 1, "expansion: {} > {}", blob.len(), plane.len() + 1);
        assert_eq!(decode_channel(&blob, Dtype::U8, 1000, 1, &prof).unwrap(), plane);
    }

    #[test]
    fn both_profiles_compress_smooth_gradient() {
        let (w, h) = (256u32, 256u32);
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n)
            .flat_map(|i| {
                let f = i as f32 / n as f32;
                let bits = f.to_bits();
                let half = (((bits >> 16) & 0x8000) | ((bits >> 13) & 0x3fff)) as u16;
                half.to_le_bytes()
            })
            .collect();
        for prof in [Profile::lossless_fast(), Profile::lossless_wave()] {
            let enc = encode_channel(&plane, Dtype::F16, w, h, &prof).expect("encode");
            assert_eq!(decode_channel(&enc, Dtype::F16, w, h, &prof).unwrap(), plane);
            assert!(enc.len() < plane.len() / 2, "{:?}: {} / {}", prof.transform, enc.len(), plane.len());
        }
    }

    /// Deterministic gate for the deflate path (TAG_DEFLATE): build a blob manually from the
    /// decorrelated bytes and check that decode_channel reconstructs the original plane.
    #[test]
    fn deflate_path_decodes() {
        let (w, h) = (32u32, 8u32);
        let dtype = Dtype::F32;
        let plane: Vec<u8> = (0..(w * h) as usize * dtype.size()).map(|i| (i % 7) as u8).collect();
        let decorrelated = reorder::split_delta(&plane, dtype.size());
        let mut blob = vec![TAG_DEFLATE];
        blob.extend_from_slice(&deflate_encode(&decorrelated).expect("deflate"));
        let prof = Profile::lossless_fast();
        assert_eq!(decode_channel(&blob, dtype, w, h, &prof).unwrap(), plane, "deflate round-trip");
        // corrupt deflate payload → None (no panic)
        assert!(decode_channel(&[TAG_DEFLATE, 0xff, 0xff, 0xff], dtype, w, h, &prof).is_err());
    }

    /// MaxRatio: on data with long repeats, LZ77 in deflate beats order-0 rANS → TAG_DEFLATE is chosen.
    #[test]
    fn deflate_wins_on_repetitive_runs_max_ratio() {
        let (w, h) = (256u32, 256u32);
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v: u32 = if (i / 1024) % 2 == 0 { 0x3f80_0000 } else { 0x4010_0000 };
                v.to_le_bytes()
            })
            .collect();
        let prof = Profile::lossless_fast().max_ratio();
        let blob = encode_channel(&plane, Dtype::F32, w, h, &prof).expect("encode");
        assert_eq!(decode_channel(&blob, Dtype::F32, w, h, &prof).unwrap(), plane);
        assert_eq!(classify_blob(&blob), Some(BlobKind::Deflate(&blob[1..])), "deflate should win under MaxRatio");
    }

    /// GpuHot (default): the SAME repetitive data → rANS shrinks → rANS is kept (GPU-decodable), even
    /// though deflate would be smaller. This is the policy that keeps the GPU decode path reachable.
    #[test]
    fn gpu_hot_prefers_rans_when_it_shrinks() {
        let (w, h) = (256u32, 256u32);
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v: u32 = if (i / 1024) % 2 == 0 { 0x3f80_0000 } else { 0x4010_0000 };
                v.to_le_bytes()
            })
            .collect();
        let prof = Profile::lossless_fast(); // GpuHot
        let blob = encode_channel(&plane, Dtype::F32, w, h, &prof).expect("encode");
        assert_eq!(decode_channel(&blob, Dtype::F32, w, h, &prof).unwrap(), plane);
        assert!(matches!(classify_blob(&blob), Some(BlobKind::Rans(_))), "GpuHot keeps rANS when it shrinks");
    }

    /// GpuHot fallback: when rANS cannot shrink a blob but deflate can (uniform histogram + LZ
    /// repetition), GpuHot still emits TAG_DEFLATE so the blob is at least compressed (CPU decode).
    #[test]
    fn gpu_hot_falls_back_to_deflate_when_rans_cannot_shrink() {
        let (w, h) = (256u32, 256u32);
        let n = (w * h) as usize;
        // split_delta of this plane is a period-256 ramp (0..255 repeating): uniform histogram
        // (rANS can't shrink) but strongly periodic (deflate LZ shrinks). Raw = the running sum.
        let mut acc = 0u8;
        let plane: Vec<u8> = (0..n)
            .map(|i| {
                acc = acc.wrapping_add((i % 256) as u8);
                acc
            })
            .collect();
        let prof = Profile::lossless_fast(); // GpuHot
        let blob = encode_channel(&plane, Dtype::U8, w, h, &prof).expect("encode");
        assert_eq!(decode_channel(&blob, Dtype::U8, w, h, &prof).unwrap(), plane);
        assert!(matches!(classify_blob(&blob), Some(BlobKind::Deflate(_))), "GpuHot rescues with deflate");
    }

    /// N7: the public decode fronts never panic on an unvalidated `w`/`h` (u32::MAX squared used to overflow an
    /// allocation inside the deflate decoder), whatever the codec.
    #[test]
    fn decode_fronts_reject_giant_dims() {
        let mut profs = vec![Profile::lossless_fast().max_ratio(), Profile::lossless_wave(), Profile::lossy_wave(3), Profile::exv_rt_dwt(16, 2)];
        #[cfg(feature = "htj2k-backend")]
        profs.push(Profile::htj2k());
        for prof in &profs {
            for blob in [&[TAG_DEFLATE, 0x03, 0x00][..], &[TAG_RANS, 1, 2, 3][..], &[TAG_EXV_RT, 0][..], &[exv_format::TAG_RAW, 0][..]] {
                for (w, h) in [(u32::MAX, u32::MAX), (1 << 17, 1 << 17), (65536, 65536)] {
                    let e = decode_channel(blob, Dtype::U8, w, h, prof);
                    assert!(matches!(e, Err(DecodeError::Request(_))), "{:?} {w}x{h}: {e:?}", prof.transform);
                    assert!(matches!(decode_tile(blob, &[Dtype::U8], w, h, prof), Err(DecodeError::Request(_))));
                }
            }
        }
    }

    /// A 3-byte DEFLATE blob claiming a plane that no DEFLATE stream of that size could expand to is refused
    /// before the decompressor reserves the plane (the 1032:1 bound), as corruption.
    #[test]
    fn deflate_bomb_is_refused_before_allocating() {
        let prof = Profile::lossless_fast().max_ratio();
        // 16384^2 = 256 MiB claimed from 3 bytes (limit would be ~4 KiB)
        let e = decode_channel(&[TAG_DEFLATE, 0x03, 0x00], Dtype::U8, 16384, 16384, &prof);
        assert!(matches!(e, Err(DecodeError::Corrupt(_))), "{e:?}");
        assert!(deflate_decode(&[0x03, 0x00], 1 << 40).is_none());
    }

    /// N6: a lossy profile keeps INTEGER channels bit-exact (object ids / masks / counters), at a size where the
    /// codec really engages (small tiles store RAW and would hide the quantizer): every lossy backend x
    /// U8/U16/U32 x smooth and hashed content.
    #[test]
    fn lossy_profiles_keep_integer_channels_exact() {
        let (w, h) = (128u32, 128u32);
        let n = (w * h) as usize;
        let smooth = |i: usize| ((i as u32 % w) * 3 + (i as u32 / w) * 5) / 4;
        let hashed = |i: usize| (i as u32).wrapping_mul(2_654_435_761) >> 11;
        let mut profs = vec![Profile::lossy_wave(4), Profile::lossy_wave(30), Profile::exv_rt_dwt(16, 2), Profile::exv_rt(8, 4, true)];
        #[cfg(feature = "htj2k-backend")]
        profs.extend([Profile::htj2k_lossy(8), Profile::htj2k_lossy(1)]);
        for prof in &profs {
            for (name, f) in [("smooth", &smooth as &dyn Fn(usize) -> u32), ("hashed", &hashed)] {
                let u8d: Vec<u8> = (0..n).map(|i| f(i) as u8).collect();
                let u16d: Vec<u8> = (0..n).flat_map(|i| (f(i) as u16).to_le_bytes()).collect();
                let u32d: Vec<u8> = (0..n).flat_map(|i| f(i).to_le_bytes()).collect();
                for (dt, plane) in [(Dtype::U8, &u8d), (Dtype::U16, &u16d), (Dtype::U32, &u32d)] {
                    let enc = encode_channel(plane, dt, w, h, prof).expect("encode");
                    assert_eq!(decode_channel(&enc, dt, w, h, prof).as_deref(), Ok(plane.as_slice()), "{:?} {dt:?} {name}", prof.transform);
                }
            }
        }
    }

    /// N5: only F16/F32/U32 HTJ2K components may take a GPU path; BF16 (the pack would read it as IEEE half) and
    /// the unsigned 8/16-bit components are CPU-only.
    #[test]
    fn htj2k_gpu_gate_truth_table() {
        for (dt, ok) in [(Dtype::F16, true), (Dtype::F32, true), (Dtype::U32, true), (Dtype::Bf16, false), (Dtype::U8, false), (Dtype::U16, false)] {
            assert_eq!(htj2k_gpu_decodable(dt), ok, "{dt:?}");
        }
    }
}
