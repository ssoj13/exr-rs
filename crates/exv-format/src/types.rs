//! Stable serializable format types: per-channel dtype and the codec-backend identifier.
//!
//! They live in `exv-format` (not `exv-codec`) because these are **discriminants written
//! into the container** — codec implementations only read them (see `docs/archive/PLAN.md` §2, §3.3). This avoids the
//! `exv-codec → exv-format → exv-codec` cycle.

/// Channel element type. v1 supports the full set (see `docs/archive/PLAN.md` §3.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F16,
    F32,
    Bf16,
    U8,
    U16,
    U32,
}

impl Dtype {
    /// Element size in bytes (= stride of the byte-plane split in the codec).
    pub const fn size(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::F16 | Dtype::Bf16 | Dtype::U16 => 2,
            Dtype::F32 | Dtype::U32 => 4,
        }
    }

    /// Float family (matters for stage selection: bit-plane for float, predictor for int).
    pub const fn is_float(self) -> bool {
        matches!(self, Dtype::F16 | Dtype::F32 | Dtype::Bf16)
    }

    /// Stable code written into the container.
    pub const fn code(self) -> u8 {
        match self {
            Dtype::F16 => 0,
            Dtype::F32 => 1,
            Dtype::Bf16 => 2,
            Dtype::U8 => 3,
            Dtype::U16 => 4,
            Dtype::U32 => 5,
        }
    }

    /// Back from the code (corrupt code → `None`).
    pub const fn from_code(c: u8) -> Option<Dtype> {
        Some(match c {
            0 => Dtype::F16,
            1 => Dtype::F32,
            2 => Dtype::Bf16,
            3 => Dtype::U8,
            4 => Dtype::U16,
            5 => Dtype::U32,
            _ => return None,
        })
    }
}

/// Codec-backend identifier (central registry, see `docs/archive/PLAN.md` §3.3).
///
/// Ranges: `0x0000_0000–0x7FFF_FFFF` — spec; `0x8000_0000+` — vendor/experimental (so
/// third-party IDs never collide with future official ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CodecId(pub u32);

impl CodecId {
    /// Raw storage (no-shrink / incompressible tile).
    pub const RAW: CodecId = CodecId(0);
    /// LosslessFast: stride-aware reorder+delta → interleaved rANS.
    pub const LOSSLESS_FAST: CodecId = CodecId(1);
    /// LosslessWave: reversible 5/3 wavelet (tile-local) → interleaved rANS.
    pub const WAVELET53: CodecId = CodecId(2);
    /// LossyWave: signed 5/3 wavelet + detail quantization (params: [q]).
    pub const WAVELET53_LOSSY: CodecId = CodecId(3);
    /// EXV-RT: realtime, embarrassingly-parallel LOSSY block codec. F16 channels → block codec
    /// (`TAG_EXV_RT`); every other dtype (F32/BF16 included) → `TAG_RAW` (a lossy profile keeps ints lossless). GPU-decodable
    /// (WGSL, the realtime 40+ GB/s path; decoder = B-RT.3). `codec_params = [ver:u8, block:u8, p:u8,
    /// flags:u8]` selects the payload (`exv-codec::BlockRtParams`): `ver=1` **Dwt** (SHIPPING —
    /// per-block signed 5/3 wavelet+quant, shared per-plane rANS table, `blockrt_dwt.rs`; `p`=quant `q`);
    /// `ver=0` **Lerp** (demoted endpoint-lerp reference/proxy, `blockrt.rs`; `p`=idx_bits, flags bit0=log).
    pub const EXV_RT: CodecId = CodecId(4);
    /// HTJ2K: ONE multi-component JPEG 2000 Part-15 codestream per tile carrying every channel (lossless 5/3,
    /// hdr log codes, or irreversible 9/7; `codec_params` pick the mode). The strategic GPU codec (CPU decode via
    /// `jph-rs`); requires the `exv-codec/htj2k-backend` feature. Per-tile resolution levels =
    /// the LOD mip pyramid.
    pub const HTJ2K: CodecId = CodecId(6);
    /// BC6H: Hardware GPU `BC6H_SF16` (`Bc6hRgbFloat`) 4x4 RGB F16 block codec (`TAG_BC6H`), with zero-cost
    /// hardware TMU decompression for 16Kx16K @ 60 fps playback and optional `rANS`-compressed 16-bit residual
    /// (`Bc6hMode::LosslessResidual`) for 100% bit-exact lossless F16 reconstruction. Non-RGB / non-F16 channels
    /// stay lossless (`LosslessFast`).
    pub const BC6H: CodecId = CodecId(7);
    // Reserved (Phase 7+): TEMPORAL_DELTA=5.

    /// True for the vendor/experimental range. Consumed by `exv_codec::registry::open` (classifies an
    /// unlinked vendor id as `OpenError::Vendor`) and by [`Display`](core::fmt::Display).
    pub const fn is_vendor(self) -> bool {
        self.0 >= 0x8000_0000
    }

    /// Spec name of a DEFINED id (incl. `RAW`); `None` for the reserved `5`, undefined and vendor ids.
    /// The ONLY id -> name table (CLI, viewer, error messages); knows `HTJ2K` regardless of any cargo feature.
    pub const fn spec_name(self) -> Option<&'static str> {
        Some(match self {
            Self::RAW => "RAW",
            Self::LOSSLESS_FAST => "LosslessFast",
            Self::WAVELET53 => "LosslessWave",
            Self::WAVELET53_LOSSY => "LossyWave",
            Self::EXV_RT => "EXV-RT",
            Self::HTJ2K => "Htj2k",
            Self::BC6H => "BC6H",
            _ => return None,
        })
    }
}

impl core::fmt::Display for CodecId {
    /// `Htj2k (6)` | `vendor 0x80000001` | `unknown 5`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.spec_name() {
            Some(n) => write!(f, "{n} ({})", self.0),
            None if self.is_vendor() => write!(f, "vendor {:#010x}", self.0),
            None => write!(f, "unknown {}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_names_and_display() {
        let names = [
            (0, "RAW"),
            (1, "LosslessFast"),
            (2, "LosslessWave"),
            (3, "LossyWave"),
            (4, "EXV-RT"),
            (6, "Htj2k"),
            (7, "BC6H"),
        ];
        for (id, n) in names {
            assert_eq!(CodecId(id).spec_name(), Some(n));
            assert_eq!(CodecId(id).to_string(), format!("{n} ({id})"));
        }
        assert_eq!(CodecId(5).spec_name(), None);
        assert_eq!(CodecId(5).to_string(), "unknown 5");
        assert_eq!(CodecId(0x8000_0001).spec_name(), None);
        assert_eq!(CodecId(0x8000_0001).to_string(), "vendor 0x80000001");
        assert!(CodecId(0x8000_0000).is_vendor() && !CodecId(0x7fff_ffff).is_vendor());
    }
}
