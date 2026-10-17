//! The [`TileCodec`] implementations behind the [`crate::registry`] table — one per codec family.
//!
//! # What and why
//! Each backend owns exactly its codec's encode/decode arms (formerly a long `if let`/`match` chain with
//! `unreachable!` tails in `encode_channel`/`decode_body`). A backend receives the [`Profile`] and extracts
//! its own [`Transform`] variant; the registry guarantees (and a test pins) that the pairing is right, so a
//! mismatch is a plain error here, never a panic. Accepted tag sets (RAW is always valid, handled by the
//! shared front): FAST/WAVE = RANS|DEFLATE, LOSSY_WAVE = RANS, EXV_RT = EXV_RT, HTJ2K = HTJ2K.

use exv_format::{BlobKind, CodecId, DecodeError, Dtype, OpenError, TAG_BC6H, TAG_DEFLATE, TAG_EXV_RT, TAG_RANS};

use crate::registry::{GpuKind, TileCodec, Work};
use crate::{
    Bc6hMode, BlockRtParams, CodecError, EntropyPolicy, Profile, RT_Q_MAX, Transform, bc6h, blockrt, blockrt_dwt, deflate_decode,
    deflate_encode, keep_or_raw, lossy, pick_into, reorder, wavelet, with_bytes,
};

/// Reject params on a codec that defines none, so `profile(p)?.params() == p` always holds.
fn no_params(id: CodecId, params: &[u8]) -> Result<(), OpenError> {
    if params.is_empty() { Ok(()) } else { Err(OpenError::BadParams { id, why: "this codec takes no params" }) }
}

const MISMATCH_TXT: &str = "profile does not belong to this codec backend";
const MISMATCH: CodecError = CodecError::Param(MISMATCH_TXT);

/// The decorrelating stage of the lossless rANS family.
#[derive(Clone, Copy)]
enum Decor {
    /// stride-aware byte-plane split + byte delta (`LOSSLESS_FAST`).
    ReorderDelta,
    /// reversible single-level 5/3 wavelet (`WAVELET53`).
    Wave53,
}

/// `LOSSLESS_FAST` / `WAVELET53`: decorrelate once, then rANS or raw-DEFLATE per [`EntropyPolicy`].
pub(crate) struct RansFamily {
    id: CodecId,
    decor: Decor,
}

impl RansFamily {
    /// Invert the decorrelation of the entropy-decoded stream `dec` (exactly `out.len()` bytes) into `out`.
    fn undo(&self, dec: &[u8], dtype: Dtype, w: u32, h: u32, out: &mut [u8]) -> Result<(), DecodeError> {
        match self.decor {
            // join_undelta is bijective and length-preserving (`dec` and `out` are the same length by construction).
            Decor::ReorderDelta => {
                reorder::join_undelta_into(dec, dtype.size(), out);
                Ok(())
            }
            // inverse_2d refuses a payload that is not exactly w*h*esz (a forged length cannot index out of bounds).
            Decor::Wave53 => wavelet::inverse_2d_into(dec, dtype, w, h, out),
        }
    }
}

pub(crate) static FAST: RansFamily = RansFamily { id: CodecId::LOSSLESS_FAST, decor: Decor::ReorderDelta };
pub(crate) static WAVE: RansFamily = RansFamily { id: CodecId::WAVELET53, decor: Decor::Wave53 };

impl TileCodec for RansFamily {
    fn id(&self) -> CodecId {
        self.id
    }

    fn profile(&self, params: &[u8]) -> Result<Profile, OpenError> {
        no_params(self.id, params)?;
        Ok(match self.decor {
            Decor::ReorderDelta => Profile::lossless_fast(),
            Decor::Wave53 => Profile::lossless_wave(),
        })
    }

    fn gpu_kind(&self, _p: &Profile) -> Option<GpuKind> {
        Some(match self.decor {
            Decor::ReorderDelta => GpuKind::RansFast,
            Decor::Wave53 => GpuKind::RansWave,
        })
    }

    /// rANS is order-0 (GPU-decodable, the hot path) but has no LZ; deflate catches LZ77 repeats where order-0 is
    /// weak (continuous float AOVs). GpuHot keeps rANS whenever it shrinks (so the GPU decode path stays
    /// reachable); MaxRatio always takes the smaller (deflate usually wins -> CPU decode).
    fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
        // The decorrelated stream is a plane-sized temporary: it lives in the per-thread scratch.
        with_bytes(plane.len(), |dec| {
            match self.decor {
                Decor::ReorderDelta => reorder::split_delta_into(plane, dtype.size(), dec),
                Decor::Wave53 => wavelet::forward_2d_into(plane, dtype, w, h, dec)?,
            }
            let start = out.len();
            out.push(TAG_RANS);
            rans::encode_into(dec, p.rans, out)?;
            // Prefer rANS whenever it shrinks; deflate is only computed to rescue an rANS-incompressible blob.
            if p.entropy == EntropyPolicy::GpuHot && out.len() - start - 1 < plane.len() {
                return Ok(());
            }
            let deflate_coded = deflate_encode(dec)?;
            let rans_blob = out.split_off(start); // [TAG_RANS][body]: rare path, one copy
            pick_into(out, plane, &[(TAG_RANS, rans_blob.get(1..).unwrap_or_default()), (TAG_DEFLATE, &deflate_coded)]);
            Ok(())
        })
        .map_err(|_| CodecError::Backend("scratch allocation failed".into()))?
    }

    fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, _p: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
        // Both transforms produce exactly `out.len()` plane bytes; rANS and deflate are capped at it so the
        // container cannot force allocating more than a plane. Foreign tags (HTJ2K / EXV-RT) = mismatched stream.
        match kind {
            BlobKind::Rans(b) => with_bytes(out.len(), |dec| {
                tracing::info_span!("rans_decode").in_scope(|| rans::decode_into(b, dec)).ok_or(DecodeError::Corrupt("rANS payload"))?;
                tracing::info_span!("undecorrelate").in_scope(|| self.undo(dec, dtype, w, h, out))
            })
            .map_err(|_| DecodeError::Alloc(out.len() as u64))?,
            BlobKind::Deflate(b) => {
                let dec = deflate_decode(b, out.len()).ok_or(DecodeError::Corrupt("deflate payload"))?;
                if dec.len() != out.len() {
                    return Err(DecodeError::Corrupt("decoded length != plane"));
                }
                self.undo(&dec, dtype, w, h, out)
            }
            BlobKind::Htj2k(_) | BlobKind::BlockRt(_) | BlobKind::Bc6h(_) | BlobKind::Raw(_) => {
                Err(DecodeError::Unsupported("blob tag not produced by this codec"))
            }
        }
    }

    /// FAST: the rANS/deflate output (one plane) lives while `join_undelta` builds the result. WAVE: that plus the
    /// `u32` lifting buffer of `inverse_2d`.
    fn work(&self, _p: &Profile, dtype: Dtype) -> Work {
        let esz = dtype.size() as u64;
        Work::per_px(match self.decor {
            Decor::ReorderDelta => esz,
            Decor::Wave53 => esz + 4,
        })
    }
}

/// `WAVELET53_LOSSY`: signed 5/3 wavelet + detail quantization -> rANS (deflate is never emitted => corrupt).
pub(crate) struct LossyWave;
pub(crate) static LOSSY_WAVE: LossyWave = LossyWave;

impl TileCodec for LossyWave {
    fn id(&self) -> CodecId {
        CodecId::WAVELET53_LOSSY
    }

    fn profile(&self, params: &[u8]) -> Result<Profile, OpenError> {
        match params {
            [q] => Ok(Profile::lossy_wave(*q)),
            _ => Err(OpenError::BadParams { id: self.id(), why: "expects exactly [q]" }),
        }
    }

    fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let Transform::WaveletLossy { q } = p.transform else { return Err(MISMATCH) };
        let start = out.len();
        out.push(TAG_RANS);
        lossy::encode_lossy_into(plane, dtype, w, h, q as u32, p.rans, out)?;
        keep_or_raw(out, start, plane);
        Ok(())
    }

    fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
        let Transform::WaveletLossy { q } = p.transform else { return Err(DecodeError::Unsupported(MISMATCH_TXT)) };
        match kind {
            BlobKind::Rans(rest) => {
                lossy::decode_lossy_into(rest, dtype, w, h, q as u32, out).ok_or(DecodeError::Corrupt("lossy-wave payload"))
            }
            _ => Err(DecodeError::Unsupported("blob tag not produced by this codec")),
        }
    }

    /// Worst case of `decode_lossy`: the rANS output (<= 10 varint bytes per coefficient, the decode cap) beside the
    /// `i64` coefficients, before the byte buffer is released and the plane built.
    fn work(&self, _p: &Profile, _dtype: Dtype) -> Work {
        Work::per_px(10 + 8)
    }
}

/// `EXV_RT`: the realtime block codec (F16 -> block payload, every other dtype -> RAW so ids/ints stay exact).
pub(crate) struct ExvRt;
pub(crate) static EXV_RT: ExvRt = ExvRt;

impl TileCodec for ExvRt {
    fn id(&self) -> CodecId {
        CodecId::EXV_RT
    }

    /// `[ver, block, {idx_bits|q}, flags]`. ver=0 Lerp: idx_bits in 1..=8, block >= 1 (decode's
    /// `/(2^idx_bits-1)`/`div_ceil` never hit 0). ver=1 Dwt: block in 1..=64 (GPU dim cap), q < 16 (F16 detail quant).
    fn profile(&self, params: &[u8]) -> Result<Profile, OpenError> {
        let bad = |why| OpenError::BadParams { id: CodecId::EXV_RT, why };
        match params {
            [0, block, idx_bits, flags] if *block >= 1 && (1..=8).contains(idx_bits) => {
                Ok(Profile::exv_rt(*block, *idx_bits, (*flags & 1) != 0))
            }
            [0, ..] => Err(bad("lerp needs block >= 1 and idx_bits in 1..=8")),
            [1, block, q, _] if (1..=64).contains(block) && *q <= RT_Q_MAX => Ok(Profile::exv_rt_dwt(*block, *q)),
            [1, ..] => Err(bad("dwt needs block in 1..=64 and q < 16")),
            _ => Err(bad("expects [ver 0|1, block, idx_bits|q, flags]")),
        }
    }

    fn gpu_kind(&self, _p: &Profile) -> Option<GpuKind> {
        Some(GpuKind::ExvRt)
    }

    /// v0 identity: F16 -> block codec (TAG_EXV_RT); every OTHER dtype -> RAW (a lossy profile keeps ids/ints
    /// exact). Broader float ingest (F32/BF16 -> f16) is deliberately NOT done: it would pre-commit a lossy ingest
    /// before it is measured.
    fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let Transform::BlockRt(bp) = p.transform else { return Err(MISMATCH) };
        if dtype != Dtype::F16 {
            pick_into(out, plane, &[]); // no candidate => the plane is stored raw
            return Ok(());
        }
        // The pub constructors only `debug_assert` these; a release build must not hit div_ceil(0).
        match bp {
            BlockRtParams::Lerp { block, idx_bits, log } => {
                if block == 0 || !(1..=8).contains(&idx_bits) {
                    return Err(CodecError::Param("EXV-RT lerp: block>=1, idx_bits in 1..=8"));
                }
                let coded = blockrt::encode_plane_mode(plane, w, h, block as u32, idx_bits as u32, log);
                pick_into(out, plane, &[(TAG_EXV_RT, &coded)]);
            }
            BlockRtParams::Dwt { block, q } => {
                if !(1..=64).contains(&block) || q > RT_Q_MAX {
                    return Err(CodecError::Param("EXV-RT dwt: block in 1..=64, q<16"));
                }
                // The blob stores the block grid as u16 and substream offsets as u32: a silent wrap would only
                // fail at decode (huge tiles with tiny blocks).
                let grid = |n: u32| n.div_ceil(block as u32) <= u16::MAX as u32;
                if !grid(w) || !grid(h) {
                    return Err(CodecError::Dims("EXV-RT block grid exceeds u16"));
                }
                let start = out.len();
                out.push(TAG_EXV_RT);
                blockrt_dwt::encode_plane_into(plane, w, h, block as u32, q as u32, blockrt_dwt::RT_LANES, out)?;
                if u32::try_from(out.len() - start - 1).is_err() {
                    return Err(CodecError::Dims("EXV-RT payload exceeds u32 offsets"));
                }
                keep_or_raw(out, start, plane);
            }
        }
        Ok(())
    }

    fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
        let Transform::BlockRt(bp) = p.transform else { return Err(DecodeError::Unsupported(MISMATCH_TXT)) };
        let corrupt = DecodeError::Corrupt("EXV-RT payload");
        let overflow = DecodeError::Request("plane size overflows");
        match (kind, bp) {
            // Lerp: fixed-rate; length-validate against block_bytes before decode (no OOB).
            (BlobKind::BlockRt(payload), BlockRtParams::Lerp { block, idx_bits, log }) if dtype == Dtype::F16 => {
                let (b, ib) = (block as u32, idx_bits as u32);
                let nb = (w.div_ceil(b) as usize).checked_mul(h.div_ceil(b) as usize).ok_or(overflow.clone())?;
                let need = nb.checked_mul(blockrt::block_bytes(b, ib)).ok_or(overflow)?;
                if payload.len() != need {
                    return Err(corrupt);
                }
                // The demoted reference codec: its decoder still returns a `Vec`.
                let plane = blockrt::decode_plane_mode(payload, w, h, b, ib, log);
                if plane.len() != out.len() {
                    return Err(DecodeError::Corrupt("decoded length != plane"));
                }
                out.copy_from_slice(&plane);
                Ok(())
            }
            // Dwt: self-describing blob; it must agree with the profile (the GPU reads the profile, so a
            // tampered blob header must not make CPU and GPU decode differently - B5).
            (BlobKind::BlockRt(payload), BlockRtParams::Dwt { block, q }) if dtype == Dtype::F16 => {
                if payload.first() != Some(&block) || payload.get(1) != Some(&q) {
                    return Err(DecodeError::Corrupt("EXV-RT blob header disagrees with the profile"));
                }
                blockrt_dwt::decode_plane_into(payload, w, h, out).ok_or(corrupt)
            }
            _ => Err(DecodeError::Unsupported("blob tag / dtype not produced by this codec")),
        }
    }

    /// F16 blocks decode into a block-padded plane (~one plane) that is then cropped into the output; every other
    /// dtype is stored RAW and has no decoder working set.
    fn work(&self, _p: &Profile, dtype: Dtype) -> Work {
        Work::per_px(if dtype == Dtype::F16 { 2 } else { 0 })
    }
}

/// `BC6H`: hardware `BC6H_SF16` 4x4 block codec (`TAG_BC6H`), with optional lossless rANS 16-bit residual.
pub(crate) struct Bc6h;
pub(crate) static BC6H: Bc6h = Bc6h;

impl TileCodec for Bc6h {
    fn id(&self) -> CodecId {
        CodecId::BC6H
    }

    fn profile(&self, params: &[u8]) -> Result<Profile, OpenError> {
        let mode = Bc6hMode::from_params(params).map_err(|why| OpenError::BadParams { id: self.id(), why })?;
        Ok(match mode {
            Bc6hMode::Lossy => Profile::bc6h_lossy(),
            Bc6hMode::LosslessResidual => Profile::bc6h_lossless(),
        })
    }

    fn gpu_kind(&self, _p: &Profile) -> Option<GpuKind> {
        Some(GpuKind::Bc6h)
    }

    fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let Transform::Bc6h(mode) = p.transform else { return Err(MISMATCH) };
        if dtype != Dtype::F16 {
            return match mode {
                Bc6hMode::Lossy => {
                    pick_into(out, plane, &[]);
                    Ok(())
                }
                Bc6hMode::LosslessResidual => {
                    let max_p = (*p).max_ratio();
                    FAST.encode_chan(plane, dtype, w, h, &max_p, out)
                }
            };
        }
        let start = out.len();
        out.push(TAG_BC6H);
        bc6h::encode_plane_into(plane, w, h, mode, p.rans, out)?;
        if mode == Bc6hMode::LosslessResidual {
            let mut fast_cand = Vec::new();
            let max_p = (*p).max_ratio();
            FAST.encode_chan(plane, dtype, w, h, &max_p, &mut fast_cand)?;
            if fast_cand.len() < out.len() - start {
                out.truncate(start);
                out.extend_from_slice(&fast_cand);
                return Ok(());
            }
        }
        keep_or_raw(out, start, plane);
        Ok(())
    }

    fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
        let Transform::Bc6h(mode) = p.transform else { return Err(DecodeError::Unsupported(MISMATCH_TXT)) };
        match (kind, dtype) {
            (BlobKind::Bc6h(payload), Dtype::F16) => bc6h::decode_plane_into(payload, w, h, mode, out),
            (BlobKind::Rans(_) | BlobKind::Deflate(_), dt) if mode == Bc6hMode::LosslessResidual => {
                FAST.decode_chan(kind, dt, w, h, p, out)
            }
            _ => Err(DecodeError::Unsupported("blob tag / dtype not produced by this codec")),
        }
    }

    fn work(&self, p: &Profile, dtype: Dtype) -> Work {
        let Transform::Bc6h(mode) = p.transform else { return Work::per_px(0) };
        match (dtype, mode) {
            (Dtype::F16, Bc6hMode::Lossy) => Work::per_px(0),
            (Dtype::F16, Bc6hMode::LosslessResidual) => Work::per_px(4),
            (_, Bc6hMode::Lossy) => Work::per_px(0),
            (dt, Bc6hMode::LosslessResidual) => FAST.work(p, dt),
        }
    }
}

#[cfg(feature = "htj2k-backend")]
pub(crate) use htj2k_be::HTJ2K;

/// HTJ2K backend: a complete per-channel / multi-component codestream codec (jph-rs), gated with its module.
#[cfg(feature = "htj2k-backend")]
mod htj2k_be {
    use super::*;
    use crate::Htj2kMode;
    use crate::htj2k;
    use crate::registry::{Dst, TileLayout};
    use exv_format::{DecodeError, TAG_HTJ2K, classify_blob};

    pub(crate) struct Htj2k;
    pub(crate) static HTJ2K: Htj2k = Htj2k;

    impl TileCodec for Htj2k {
        fn id(&self) -> CodecId {
            CodecId::HTJ2K
        }

        /// `[]` lossless, `[1, q]` hdr codes, `[2, q]` irreversible 9/7 ([`Htj2kMode::from_params`]).
        fn profile(&self, params: &[u8]) -> Result<Profile, OpenError> {
            let mode = Htj2kMode::from_params(params).map_err(|why| OpenError::BadParams { id: self.id(), why })?;
            Ok(Profile { transform: Transform::Htj2k(mode), ..Profile::htj2k() })
        }

        fn layout(&self) -> TileLayout {
            TileLayout::MultiComp
        }

        /// Only the lossless profile has a GPU path (jph-wgpu): hdr codes and the 9/7 decode on the CPU.
        fn gpu_kind(&self, p: &Profile) -> Option<GpuKind> {
            matches!(p.transform, Transform::Htj2k(Htj2kMode::Lossless)).then_some(GpuKind::Htj2k)
        }

        /// One channel = a one-component codestream of the SAME encoder as the tile blob ([`htj2k::encode_into`]; HdrCodes
        /// maps floats to value-domain log codes and codes them losslessly - quantizing raw float bits is garbage -
        /// while integer channels, U32 ids/masks, are never quantized; only Irrev97 quantizes, and only U8/U16).
        fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
            let Transform::Htj2k(mode) = p.transform else { return Err(MISMATCH) };
            let start = out.len();
            out.push(TAG_HTJ2K);
            htj2k::encode_into(mode, &[plane], &[dtype], w, h, out)?;
            keep_or_raw(out, start, plane);
            Ok(())
        }

        fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut [u8]) -> Result<(), DecodeError> {
            let Transform::Htj2k(mode) = p.transform else { return Err(DecodeError::Unsupported(MISMATCH_TXT)) };
            let BlobKind::Htj2k(cs) = kind else { return Err(DecodeError::Unsupported("blob tag not produced by this codec")) };
            htj2k::decode_into(mode, cs, &[dtype], w, h, &mut [Dst::Plane(out)])
        }

        /// jph-rs's coefficient/line storage, MEASURED on worst-case (random) tiles in `tests/mem_budget.rs`
        /// (`htj2k_worst_case_working_set_is_covered` prints every shape, 2026-10-01, jph-rs e723285): <= ~1.7 B/px/channel
        /// (charged 4), a per-component block context of ~0.30 MiB (16-bit and below) / ~0.57 MiB (32-bit) (charged 320 /
        /// 640 KiB), and ~2.3 MiB of codec context per tile ([`Self::work_base`], charged 2.5 MiB). Tightened from
        /// 512/768 KiB + 4 MiB: this sum is also the per-lane charge of the whole-frame decode window, and the old
        /// over-count kept ~12 of 24 cores busy at 4k. The bound holds because the header check pins 64x64 blocks /
        /// 5 levels / 1x1 components; the tightest measured shape (F32 512^2 x 2ch) peaks at 0.96x the account.
        fn work(&self, _p: &Profile, dtype: Dtype) -> Work {
            Work { per_px: 4, fixed: if dtype.size() == 4 { 640 << 10 } else { 320 << 10 } }
        }

        fn work_base(&self, _p: &Profile) -> u64 {
            2560 << 10
        }

        /// ALL channels in ONE multi-component codestream (96x fewer codestreams at 65k; the unit jph-wgpu
        /// GPU-decodes per tile): `[TAG_HTJ2K][codestream]`, no chan_len table.
        fn encode_tile(&self, chans: &[&[u8]], dtypes: &[Dtype], w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
            let Transform::Htj2k(mode) = p.transform else { return Err(MISMATCH) };
            out.push(TAG_HTJ2K);
            htj2k::encode_into(mode, chans, dtypes, w, h, out)
        }

        /// The exact-length contract of `decode_channel` holds by construction: the shared front checked `out` and
        /// [`htj2k::decode_into`] fills every byte of every non-skipped destination (rows straight from jph's lines) or fails.
        fn decode_tile(&self, blob: &[u8], dtypes: &[Dtype], w: u32, h: u32, p: &Profile, out: &mut [Dst<'_>]) -> Result<(), DecodeError> {
            let Transform::Htj2k(mode) = p.transform else {
                return Err(DecodeError::Unsupported("profile does not belong to the HTJ2K backend"));
            };
            match classify_blob(blob).ok_or(DecodeError::Corrupt("empty or unknown tile tag"))? {
                BlobKind::Htj2k(payload) => htj2k::decode_into(mode, payload, dtypes, w, h, out),
                _ => Err(DecodeError::Unsupported("non-HTJ2K tile blob under an HTJ2K profile")),
            }
        }
    }
}
