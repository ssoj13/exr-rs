//! Codec registry: the ONE table that maps an on-disk [`CodecId`] to the code that encodes/decodes it.
//!
//! # What and why
//! Every codec family is one stateless [`TileCodec`] living in the compile-time [`BACKENDS`] slice. The
//! table owns *behaviour and validation* (params parsing at open, per-channel/per-tile encode+decode,
//! tile layout, GPU capability); [`Profile`] stays the plain *data* threaded through the call sites and
//! reaches its backend through [`Profile::backend`] (a total, exhaustive `match`, so no lookup can fail).
//! [`open`] is the single open-time validator: an unknown / vendor / not-built / bad-params header is a
//! typed [`OpenError`], never a silent fallback.
//!
//! # No runtime registration (on purpose)
//! [`Transform`](crate::Transform) is a closed enum and decode must be deterministic (GPU == CPU gate), so a
//! process-global mutable table would only add non-determinism. A vendor codec (`id >= 0x8000_0000`) is added
//! by (1) picking an id in that range, (2) a `Transform` variant + a [`TileCodec`] impl in `backends.rs`,
//! (3) appending it to [`BACKENDS`]. Until then such ids fail with [`OpenError::Vendor`].
//!
//! # Two dispatch layers (both kept)
//! File level = `CodecId` (this table, resolved once at open). Blob level = the per-blob tag
//! ([`BlobKind`], single source in exv-format; exv-gpu reads it there, below the registry). The shared
//! fronts [`crate::encode_channel`]/[`crate::decode_channel`] do dims / RAW / length-guarantee once and call
//! the backend, which owns which non-RAW tags it accepts (a foreign tag => `None` = corrupt).

use exv_format::{BlobKind, CodecId, DecodeError, Dtype};
pub use exv_format::OpenError;

use crate::backends;
use crate::{CodecError, Profile};

/// Which GPU decode route a profile can take. Plain DATA (not fn pointers): `exv-gpu` maps it to its kernels,
/// so there is no exv-codec -> exv-gpu edge. NOT cfg-gated: consumers match it exhaustively in every build.
/// Device limits stay a runtime `GpuError` (caller falls back to CPU); this is the static capability only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuKind {
    /// `LOSSLESS_FAST`: reorder-delta -> rANS (rANS on GPU, undelta prefix-sum on CPU).
    RansFast,
    /// `WAVELET53`: rANS -> repack -> inverse 5/3 on GPU.
    RansWave,
    /// `EXV_RT`: per-block rANS + inverse 5/3 + dequant (F16 blocks; other dtypes ride as RAW tiles).
    ExvRt,
    /// Lossless HTJ2K (jph-wgpu); lossy HTJ2K has no GPU path.
    Htj2k,
    /// `BC6H`: hardware `BC6H_SF16` 4x4 blocks (direct `Bc6hRgbFloat` texture upload + WGSL compute decode / residual).
    Bc6h,
}

/// How one tile blob is laid out; replaces every `matches!(Transform::Htj2k{..})` used as a layout probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileLayout {
    /// `[chan_len[nch]:u32 LE][ch0_blob][ch1_blob]..` — channels independently decodable.
    PerChannel,
    /// `[tag][one multi-component stream]` — all channels in one stream (HTJ2K): no per-channel sub-blob.
    MultiComp,
}

/// Where ONE channel of a tile decodes to ([`TileCodec::decode_tile`]).
///
/// Why rows: tiles of one tile row share every plane row, so safe Rust cannot hand two decode lanes one `&mut`
/// rectangle each of a row-major plane; it CAN hand each lane its own row sub-slices (`split_at_mut`). With
/// [`Dst::Rows`] the whole-frame decode writes straight into the final planes (no tile buffer, no scatter copy).
#[derive(Debug)]
pub enum Dst<'a> {
    /// A tile-contiguous plane of exactly `w*h*esz` bytes.
    Plane(&'a mut [u8]),
    /// Exactly `h` rows of exactly `w*esz` bytes each (a window into a larger plane; rows need not be adjacent).
    Rows(Vec<&'a mut [u8]>),
    /// Not wanted: never written; the codec does no work for it where its layout allows (per-channel codecs skip
    /// the sub-blob entirely, HTJ2K stops after the last wanted component).
    Skip,
}

impl Dst<'_> {
    /// `true` iff this destination has exactly the `w`x`h` shape of `esz`-byte samples (or is [`Dst::Skip`]).
    pub fn fits(&self, w: usize, h: usize, esz: usize) -> bool {
        let row = w.saturating_mul(esz);
        match self {
            Dst::Plane(p) => h.checked_mul(row) == Some(p.len()),
            Dst::Rows(r) => r.len() == h && r.iter().all(|r| r.len() == row),
            Dst::Skip => true,
        }
    }

    /// Row `y` (`row` bytes) of a [`Self::fits`]-checked destination; `None` for [`Dst::Skip`] or out of range.
    pub fn row(&mut self, y: usize, row: usize) -> Option<&mut [u8]> {
        match self {
            Dst::Plane(p) => p.get_mut(y.checked_mul(row)?..y.checked_add(1)?.checked_mul(row)?),
            Dst::Rows(r) => r.get_mut(y).map(|r| &mut **r),
            Dst::Skip => None,
        }
    }

    /// `true` for [`Dst::Skip`].
    pub fn is_skip(&self) -> bool {
        matches!(self, Dst::Skip)
    }
}

/// Decoder working set of ONE channel of a tile (see [`TileCodec::work`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Work {
    /// Bytes per tile pixel (scales with the tile area).
    pub per_px: u64,
    /// Bytes per channel independent of the area (per-component codec state).
    pub fixed: u64,
}

impl Work {
    /// Area-proportional only.
    pub const fn per_px(per_px: u64) -> Self {
        Self { per_px, fixed: 0 }
    }
}

/// One codec family, identified by exactly one [`CodecId`]. Stateless and object-safe so it lives in a `static`.
pub trait TileCodec: Send + Sync + 'static {
    /// The on-disk id this backend serves.
    fn id(&self) -> CodecId;

    /// Display name (logs, `info`); default = the spec name of [`Self::id`].
    fn name(&self) -> &'static str {
        self.id().spec_name().unwrap_or("?")
    }

    /// Parse + validate `codec_params` into a [`Profile`] (THE replacement of the old `from_codec` arms).
    /// Contract (tested): `profile(p)?.codec_id() == self.id()` and `profile(p)?.params() == p`.
    fn profile(&self, params: &[u8]) -> Result<Profile, OpenError>;

    /// Layout of this codec's tile blobs.
    fn layout(&self) -> TileLayout {
        TileLayout::PerChannel
    }

    /// GPU route for this (already validated) profile; `None` = CPU only.
    fn gpu_kind(&self, _p: &Profile) -> Option<GpuKind> {
        None
    }

    /// Encode ONE plane (dims already validated by the shared front), APPENDING a full tagged blob to `out`
    /// (the front rolls `out` back on `Err`); applies the no-shrink RAW rule via `pick_into` / `keep_or_raw`.
    fn encode_chan(&self, plane: &[u8], dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError>;

    /// Decode ONE non-RAW blob of an already classified `kind` into `out`, which the shared front has sized to
    /// EXACTLY `w*h*esz` (RAW is the front's job; the length guarantee is the slice type). Every byte of `out` is
    /// overwritten or the call fails - a short-but-valid stream is a `Corrupt` error here, not a short plane. A
    /// tag this codec does not accept, or a corrupt payload, is a typed [`DecodeError`] naming the failed
    /// structure - never a panic, never a bare `None`.
    fn decode_chan(&self, kind: BlobKind<'_>, dtype: Dtype, w: u32, h: u32, p: &Profile, out: &mut [u8]) -> Result<(), DecodeError>;

    /// Transient heap one channel of `dtype` holds while decoding a tile, on top of its decoded plane
    /// (entropy-decoder output, widened coefficients, codec scratch), for this validated profile. The frame
    /// readers add `tile_px * sum(per_px) + sum(fixed) + work_base()` to their budget so
    /// [`exv_format::DecodeLimits`] bounds the real peak heap, not just the decoded pixels. Derived from each
    /// decoder's allocations and gated by `tests/mem_budget.rs` (measured peak <= accounted, incl. HTJ2K on
    /// worst-case random data); a decoder that allocates more MUST raise this.
    fn work(&self, p: &Profile, dtype: Dtype) -> Work;

    /// Per-tile-decode heap independent of the channel count and the tile area (a codec's own context, e.g.
    /// jph-rs's allocator chunks); 0 for codecs without one.
    fn work_base(&self, _p: &Profile) -> u64 {
        0
    }

    /// Encode one tile, APPENDING its blob to `out`. Default = the [`TileLayout::PerChannel`] framing (u32 length
    /// table + sub-blobs): each channel blob is appended in place and its length patched into the table, so there
    /// is no per-channel `Vec` and no concatenating copy. A `MultiComp` codec overrides both tile entry points.
    fn encode_tile(&self, chans: &[&[u8]], dtypes: &[Dtype], w: u32, h: u32, p: &Profile, out: &mut Vec<u8>) -> Result<(), CodecError> {
        let table = out.len();
        out.resize(table + 4 * chans.len(), 0);
        let mut lens = Vec::with_capacity(chans.len());
        for (buf, &dt) in chans.iter().zip(dtypes) {
            let start = out.len();
            crate::encode_channel_into(buf, dt, w, h, p, out)?;
            lens.push(u32::try_from(out.len() - start).map_err(|_| CodecError::Dims("channel blob exceeds u32"))?);
        }
        let slots = out.get_mut(table..table + 4 * lens.len()).ok_or(CodecError::Dims("channel length table"))?;
        for (slot, len) in slots.as_chunks_mut::<4>().0.iter_mut().zip(lens) {
            slot.copy_from_slice(&len.to_le_bytes());
        }
        Ok(())
    }

    /// Decode one tile into `out[ci]` (one [`Dst`] per channel, shapes checked by the shared front
    /// [`crate::decode_tile_dst`]); every byte of every non-skipped destination is written or the call fails.
    /// Default = split the length table and decode the wanted sub-blobs IN PARALLEL: each channel is independent
    /// and owns its output, so peak heap equals a serial loop's (the out-of-core mem gate holds) while a 43-AOV tile
    /// decodes ~cores x faster. A per-channel decoder produces a contiguous plane, so a [`Dst::Rows`] destination
    /// costs one row copy out of a per-thread tile scratch (L2-resident, unlike a whole multi-channel tile buffer).
    fn decode_tile(&self, blob: &[u8], dtypes: &[Dtype], w: u32, h: u32, p: &Profile, out: &mut [Dst<'_>]) -> Result<(), DecodeError> {
        use rayon::prelude::*;
        let subs = crate::stream::split_tile_blob(blob, dtypes.len()).ok_or(DecodeError::Corrupt("tile channel-length table"))?;
        out.par_iter_mut().zip(dtypes).enumerate().try_for_each(|(ci, (dst, &dt))| {
            if dst.is_skip() {
                return Ok(());
            }
            if !dst.fits(w as usize, h as usize, dt.size()) {
                return Err(DecodeError::Request("output plane length != w*h*esz"));
            }
            // Trace BEFORE decoding: on a hang the log's last line names the (channel, dtype) that froze.
            tracing::trace!(ci, ?dt, "decode_channel");
            let sub = subs.get(ci).copied().ok_or(DecodeError::Corrupt("tile channel count"))?;
            match dst {
                Dst::Plane(plane) => crate::decode_channel_into(sub, dt, w, h, p, plane),
                Dst::Rows(rows) => {
                    let len = crate::plane_len(w, h, dt).ok_or(DecodeError::Request("plane size overflows"))?;
                    crate::with_tile_bytes(len, |tmp| {
                        crate::decode_channel_into(sub, dt, w, h, p, tmp)?;
                        let row = (w as usize * dt.size()).max(1);
                        for (dst, src) in rows.iter_mut().zip(tmp.chunks_exact(row)) {
                            dst.copy_from_slice(src);
                        }
                        Ok(())
                    })
                    .map_err(|_| DecodeError::Alloc(len as u64))?
                }
                Dst::Skip => Ok(()),
            }
        })
    }
}

/// Every codec linked into this build, in id order. Compile-time constant: no lock, no init, no mutation.
static BACKENDS: &[&dyn TileCodec] = &[
    &backends::FAST,
    &backends::WAVE,
    &backends::LOSSY_WAVE,
    &backends::EXV_RT,
    #[cfg(feature = "htj2k-backend")]
    &backends::HTJ2K,
    &backends::BC6H,
];

/// All linked backends.
pub fn all() -> &'static [&'static dyn TileCodec] {
    BACKENDS
}

/// The backend serving exactly `id`, or `None` (use [`open`] to learn WHY).
pub fn get(id: CodecId) -> Option<&'static dyn TileCodec> {
    BACKENDS.iter().copied().find(|b| b.id() == id)
}

/// THE open-time validator: classify `id`, look the backend up and parse `params`. Every consumer of a
/// header's `(codec_id, codec_params)` (container parse, GPU dispatch, CLI verify, C-ABI) goes through here.
pub fn open(id: CodecId, params: &[u8]) -> Result<Profile, OpenError> {
    match get(id) {
        Some(b) => b.profile(params),
        None if id == CodecId::RAW => Err(OpenError::NoProfile(id)),
        None if id.is_vendor() => Err(OpenError::Vendor(id)),
        // A DEFINED spec id without a linked backend: HTJ2K without `htj2k-backend` is the only gated codec
        // (YAGNI: make the feature name a per-id table when a second gated codec appears).
        None if id.spec_name().is_some() => Err(OpenError::NotBuilt { id, feature: "htj2k-backend" }),
        None => Err(OpenError::Unknown(id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transform;

    /// Every Profile constructible in this build, with an arbitrary valid shape.
    fn all_profiles() -> Vec<Profile> {
        let mut v = vec![
            Profile::lossless_fast(),
            Profile::lossless_wave(),
            Profile::lossy_wave(5),
            Profile::exv_rt(4, 4, true),
            Profile::exv_rt_dwt(16, 2),
            Profile::bc6h_lossy(),
            Profile::bc6h_lossless(),
        ];
        #[cfg(feature = "htj2k-backend")]
        v.extend([Profile::htj2k(), Profile::htj2k_lossy(7), Profile::htj2k_ldr(12)]);
        v
    }

    #[test]
    fn table_complete_and_unique() {
        let ids: Vec<u32> = all().iter().map(|b| b.id().0).collect();
        let mut want = vec![1, 2, 3, 4];
        #[cfg(feature = "htj2k-backend")]
        want.push(6);
        want.push(7);
        assert_eq!(ids, want, "BACKENDS = every defined id, once, in id order");
        for b in all() {
            assert!(get(b.id()).is_some_and(|g| g.id() == b.id()));
            assert_eq!(Some(b.name()), b.id().spec_name(), "name == spec_name");
        }
    }

    /// Drift guard between `Profile::{codec_id, params, backend}` and the table.
    #[test]
    fn profile_roundtrip_and_backend() {
        for p in all_profiles() {
            let b = p.backend();
            assert_eq!(b.id(), p.codec_id());
            assert!(get(p.codec_id()).is_some(), "{:?} is in BACKENDS", p.transform);
            let rt = open(p.codec_id(), &p.params()).expect("open");
            assert_eq!(rt.transform, p.transform);
            assert_eq!(rt.params(), p.params());
        }
    }

    #[test]
    fn open_error_classes() {
        let id = CodecId;
        assert_eq!(open(id(5), &[]).unwrap_err(), OpenError::Unknown(id(5)));
        assert_eq!(open(id(8), &[]).unwrap_err(), OpenError::Unknown(id(8)));
        assert_eq!(open(id(0x8000_0000), &[]).unwrap_err(), OpenError::Vendor(id(0x8000_0000)));
        assert_eq!(open(id(0xFFFF_FFFF), &[]).unwrap_err(), OpenError::Vendor(id(0xFFFF_FFFF)));
        assert_eq!(open(CodecId::RAW, &[]).unwrap_err(), OpenError::NoProfile(CodecId::RAW));
        let bad = |c: CodecId, p: &[u8]| matches!(open(c, p), Err(OpenError::BadParams { .. }));
        assert!(bad(CodecId::LOSSLESS_FAST, &[1]));
        assert!(bad(CodecId::WAVELET53, &[0]));
        assert!(bad(CodecId::WAVELET53_LOSSY, &[]));
        assert!(bad(CodecId::WAVELET53_LOSSY, &[1, 2]));
        assert!(bad(CodecId::EXV_RT, &[0, 4, 0, 1])); // idx_bits 0
        assert!(bad(CodecId::EXV_RT, &[0, 0, 4, 1])); // block 0
        assert!(bad(CodecId::EXV_RT, &[9, 4, 4, 1])); // bad ver
        assert!(bad(CodecId::EXV_RT, &[1, 16, 16, 0])); // q >= 16
        assert!(bad(CodecId::EXV_RT, &[1, 0, 4, 0])); // block 0
        assert!(bad(CodecId::EXV_RT, &[1, 16, 2])); // wrong arity
        assert!(bad(CodecId::BC6H, &[]));
        assert!(bad(CodecId::BC6H, &[2]));
        assert!(bad(CodecId::BC6H, &[0, 1]));
        // HTJ2K params are `[]` / `[1, q]` / `[2, q in 1..=40]`; the pre-`Htj2kMode` bare `[q]` is rejected.
        #[cfg(feature = "htj2k-backend")]
        for p in [&[8][..], &[0, 5], &[3, 5], &[1], &[1, 2, 3], &[2], &[2, 0], &[2, 41]] {
            assert!(bad(CodecId::HTJ2K, p), "{p:?}");
        }
        #[cfg(not(feature = "htj2k-backend"))]
        assert!(matches!(open(CodecId::HTJ2K, &[]), Err(OpenError::NotBuilt { .. })));
        for e in [OpenError::Unknown(id(5)), OpenError::Vendor(id(0x8000_0001)), OpenError::NoProfile(CodecId::RAW)] {
            assert!(e.to_string().contains(&match &e {
                OpenError::Unknown(i) | OpenError::Vendor(i) | OpenError::NoProfile(i) => i.to_string(),
                _ => unreachable!(),
            }));
        }
    }

    #[test]
    fn layout_and_gpu_kind_tables() {
        for p in all_profiles() {
            let multi = p.backend().layout() == TileLayout::MultiComp;
            assert_eq!(multi, matches!(p.codec_id(), CodecId::HTJ2K));
            let want = match p.transform {
                Transform::ReorderDelta => Some(GpuKind::RansFast),
                Transform::Wavelet53 => Some(GpuKind::RansWave),
                Transform::WaveletLossy { .. } => None,
                Transform::BlockRt(_) => Some(GpuKind::ExvRt),
                Transform::Bc6h(_) => Some(GpuKind::Bc6h),
                #[cfg(feature = "htj2k-backend")]
                Transform::Htj2k(m) => m.is_lossless().then_some(GpuKind::Htj2k),
            };
            assert_eq!(p.gpu_kind(), want, "{:?}", p.transform);
        }
    }

    /// A foreign non-RAW tag under each profile is `None` (corrupt), RAW decodes under every profile.
    #[test]
    fn foreign_tags_rejected_raw_accepted() {
        use exv_format::{TAG_BC6H, TAG_DEFLATE, TAG_EXV_RT, TAG_HTJ2K, TAG_RANS, TAG_RAW};
        let (w, h) = (4u32, 4u32);
        let plane = vec![7u8; 32];
        for p in all_profiles() {
            let mut raw = vec![TAG_RAW];
            raw.extend_from_slice(&plane);
            assert_eq!(crate::decode_channel(&raw, Dtype::F16, w, h, &p).as_deref(), Ok(plane.as_slice()), "{:?}", p.transform);
            for tag in [TAG_RANS, TAG_DEFLATE, TAG_HTJ2K, TAG_EXV_RT, TAG_BC6H] {
                let own = matches!(
                    (p.transform, tag),
                    (Transform::ReorderDelta | Transform::Wavelet53, TAG_RANS | TAG_DEFLATE)
                        | (Transform::WaveletLossy { .. }, TAG_RANS)
                        | (Transform::BlockRt(_), TAG_EXV_RT)
                        | (Transform::Bc6h(_), TAG_BC6H)
                ) || (cfg!(feature = "htj2k-backend") && tag == TAG_HTJ2K && p.codec_id() == CodecId::HTJ2K);
                if !own {
                    // junk payload under a tag this codec never emits must be rejected
                    let blob = [tag, 1, 2, 3, 4, 5, 6, 7, 8];
                    assert!(crate::decode_channel(&blob, Dtype::F16, w, h, &p).is_err(), "{:?} accepted tag {tag}", p.transform);
                }
            }
        }
    }
}
