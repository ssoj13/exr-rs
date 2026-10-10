//! Rank and pack Cryptomatte ID/coverage contributions into EXR channel planes.
//!
//! Spec 1.2.0 ranking (file-format half): unique IDs ordered by coverage
//! descending; highest -> slot 0, next -> slot 1, .... Each numbered layer
//! `NameNN` holds two slots as RGBA = (ID0, cov0, ID1, cov1).
//!
//! This module does **not** implement the renderer-side sample filter kernel
//! (weights, beauty kernel match). Callers supply already-aggregated coverages.

use std::borrow::Cow;
use std::collections::HashMap;

use exr_core::{ChannelData, Image};
use imath_rs::{Box2i, V2i};

use crate::channels::{
    ranked_base, ranked_channel_name, ranked_channel_of, Component, RankedLayer, Spelling,
    MAX_RANKED_LAYERS,
};
use crate::error::{Error, Result};
use crate::matte::{channel_as_f32, id_channel_as_f32, require_channel};

/// One ID/coverage contribution before ranking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contribution {
    /// Cryptomatte float ID (Murmur bits / exponent clamp).
    pub id: f32,
    /// Coverage weight in the pixel (typically in `[0, 1]`).
    pub coverage: f32,
}

impl Contribution {
    /// Construct a contribution pair.
    #[inline]
    pub fn new(id: f32, coverage: f32) -> Self {
        Self { id, coverage }
    }
}

/// Merge same float-ID bit patterns (sum coverage), drop non-positive / non-finite
/// coverage, sort by coverage descending (tie-break: lower ID bits first), keep
/// at most `max_slots` entries.
pub fn merge_and_rank(contributions: &[Contribution], max_slots: usize) -> Vec<Contribution> {
    if max_slots == 0 || contributions.is_empty() {
        return Vec::new();
    }

    let mut by_bits: HashMap<u32, f32> = HashMap::new();
    for c in contributions {
        if !c.coverage.is_finite() || c.coverage <= 0.0 {
            continue;
        }
        if !c.id.is_finite() {
            continue;
        }
        *by_bits.entry(c.id.to_bits()).or_insert(0.0) += c.coverage;
    }

    let mut ranked: Vec<Contribution> = by_bits
        .into_iter()
        .map(|(bits, coverage)| Contribution {
            id: f32::from_bits(bits),
            coverage,
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.coverage
            .total_cmp(&a.coverage)
            .then_with(|| a.id.to_bits().cmp(&b.id.to_bits()))
    });
    ranked.truncate(max_slots);
    ranked
}

/// Apply Porter-Duff-style coverage composite of a new matte onto existing ranks.
///
/// - [`MergeOp::Over`]: new is A over existing B -- scale B by `(1 - a)`, add A.
/// - [`MergeOp::Under`]: new is A under existing B -- scale A by `(1 - occupied_B)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeOp {
    /// New matte composites over existing IDs.
    Over,
    /// New matte composites under existing IDs.
    Under,
}

/// Composite `new_id` / `new_cov` onto an existing per-pixel contribution list.
pub fn merge_contribution(
    existing: &[Contribution],
    new_id: f32,
    new_cov: f32,
    op: MergeOp,
) -> Vec<Contribution> {
    let a = if new_cov.is_finite() {
        new_cov.max(0.0)
    } else {
        0.0
    };

    let mut out: Vec<Contribution> = match op {
        MergeOp::Over => existing
            .iter()
            .map(|c| Contribution {
                id: c.id,
                coverage: c.coverage * (1.0 - a),
            })
            .collect(),
        MergeOp::Under => existing.to_vec(),
    };

    let add = match op {
        MergeOp::Over => a,
        MergeOp::Under => {
            let occupied: f32 = existing.iter().map(|c| c.coverage.max(0.0)).sum();
            let room = (1.0 - occupied).max(0.0);
            a * room
        }
    };

    if add > 0.0 && new_id.is_finite() {
        out.push(Contribution::new(new_id, add));
    }
    out
}

/// One ranked slot's ID and coverage samples, borrowed from a FLOAT channel or
/// converted from a HALF one.
type IdCovPlane<'a> = (Cow<'a, [f32]>, Cow<'a, [f32]>);

/// Named float channel planes produced by [`pack_channels`].
pub type PackedChannels = Vec<(String, ChannelData)>;

/// Ranked ID/coverage planes converted to `f32` **once**, for per-pixel reads.
///
/// The per-pixel unpack used by inject would otherwise re-convert a whole HALF
/// plane for every pixel (O(n²); measured 7.3 s for 128² before this type
/// existed). Build one of these per image and index it.
pub struct RankedPlanes<'a> {
    /// (ids, coverages) per ID/coverage pair, in `ranked` slot order.
    pairs: Vec<IdCovPlane<'a>>,
    pixel_count: usize,
}

impl<'a> RankedPlanes<'a> {
    /// Convert every ranked ID/coverage plane of `image` once.
    pub fn new(image: &'a Image, ranked: &[RankedLayer], pixel_count: usize) -> Result<Self> {
        Self::from_channels(ranked, pixel_count, |name| require_channel(image, name))
    }

    /// Plane source form of [`RankedPlanes::new`] — `get_channel` is called
    /// exactly once per ranked channel, whatever the pixel count.
    pub fn from_channels<F>(
        ranked: &[RankedLayer],
        pixel_count: usize,
        mut get_channel: F,
    ) -> Result<Self>
    where
        F: FnMut(&str) -> Result<&'a ChannelData>,
    {
        let mut pairs = Vec::new();
        for layer in ranked {
            for (id_name, cov_name) in layer.pairs() {
                let ids = id_channel_as_f32(id_name, get_channel(id_name)?)?;
                let covs = channel_as_f32(get_channel(cov_name)?)?;
                if ids.len() != pixel_count || covs.len() != pixel_count {
                    return Err(Error::Metadata(format!(
                        "channel length mismatch for `{id_name}`/`{cov_name}` (got {}/{} want {pixel_count})",
                        ids.len(),
                        covs.len()
                    )));
                }
                pairs.push((ids, covs));
            }
        }
        Ok(Self { pairs, pixel_count })
    }

    /// Contributions at linear pixel `i` (non-finite / non-positive dropped).
    ///
    /// # Errors
    /// [`Error::Metadata`] if `i` is outside the plane.
    pub fn contributions_at(&self, i: usize) -> Result<Vec<Contribution>> {
        if i >= self.pixel_count {
            return Err(Error::Metadata(format!(
                "pixel index {i} out of range (pixel count {})",
                self.pixel_count
            )));
        }
        let mut out = Vec::with_capacity(self.pairs.len());
        for (ids, covs) in &self.pairs {
            let (Some(&id), Some(&cov)) = (ids.get(i), covs.get(i)) else {
                continue;
            };
            if cov.is_finite() && cov > 0.0 && id.is_finite() {
                out.push(Contribution::new(id, cov));
            }
        }
        Ok(out)
    }
}

/// Pack ranked slots into named `{name}00.R/G/B/A`, … float channel planes.
///
/// `pixel_contributions(i)` returns the unordered contributions for linear
/// pixel index `i` in `[0, width*height)`. Shared by [`pack_image`] (fresh
/// image) and [`replace_ranked_channels`] (in-place re-encode).
#[allow(clippy::needless_range_loop)] // scatter into multi-plane buffers by pixel index
pub fn pack_channels<F>(
    width: usize,
    height: usize,
    crypto_name: &str,
    ranked_layers: usize,
    spelling: Spelling,
    mut pixel_contributions: F,
) -> Result<(PackedChannels, Vec<RankedLayer>)>
where
    F: FnMut(usize) -> Vec<Contribution>,
{
    if width == 0 || height == 0 {
        return Err(Error::Metadata("encode requires non-zero width/height".into()));
    }
    if ranked_layers == 0 {
        return Err(Error::Metadata(
            "encode requires at least one ranked layer".into(),
        ));
    }
    if ranked_layers > MAX_RANKED_LAYERS {
        return Err(Error::Metadata(format!(
            "encode supports at most {MAX_RANKED_LAYERS} ranked layers (two-digit \
             `Name00`..`Name99` indices), got {ranked_layers}"
        )));
    }
    if crypto_name.is_empty() {
        return Err(Error::Metadata("encode requires a Cryptomatte type name".into()));
    }

    let n = width
        .checked_mul(height)
        .ok_or_else(|| Error::Metadata("image pixel count overflow".into()))?;
    let max_slots = ranked_layers
        .checked_mul(2)
        .ok_or_else(|| Error::Metadata("ranked layer count overflow".into()))?;

    // planes[layer][0]=R id0, [1]=G cov0, [2]=B id1, [3]=A cov1
    let mut planes: Vec<[Vec<f32>; 4]> = (0..ranked_layers)
        .map(|_| {
            [
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
                vec![0.0f32; n],
            ]
        })
        .collect();

    for i in 0..n {
        let ranked = merge_and_rank(&pixel_contributions(i), max_slots);
        for (slot, c) in ranked.iter().enumerate() {
            let layer = slot / 2;
            let pair = slot % 2;
            if pair == 0 {
                planes[layer][0][i] = c.id;
                planes[layer][1][i] = c.coverage;
            } else {
                planes[layer][2][i] = c.id;
                planes[layer][3][i] = c.coverage;
            }
        }
    }

    let mut out = Vec::with_capacity(ranked_layers * 4);
    let mut ranked_out = Vec::with_capacity(ranked_layers);

    for (li, plane) in planes.into_iter().enumerate() {
        let base = ranked_base(crypto_name, li);
        let r = ranked_channel_name(&base, Component::R, spelling);
        let g = ranked_channel_name(&base, Component::G, spelling);
        let b = ranked_channel_name(&base, Component::B, spelling);
        let a = ranked_channel_name(&base, Component::A, spelling);
        let [pr, pg, pb, pa] = plane;
        out.push((r.clone(), ChannelData::Float(pr)));
        out.push((g.clone(), ChannelData::Float(pg)));
        out.push((b.clone(), ChannelData::Float(pb)));
        out.push((a.clone(), ChannelData::Float(pa)));
        ranked_out.push(RankedLayer {
            base,
            id0: Some(r),
            cov0: Some(g),
            id1: Some(b),
            cov1: Some(a),
        });
    }

    Ok((out, ranked_out))
}

/// Data and display window of an encoded image.
///
/// Carried end to end so an offset or overscan matte lands on the pixels it
/// was made for (a `(0,0)`-based window moves it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Windows {
    /// Data window: the pixels the planes hold, row-major.
    pub data: Box2i,
    /// Display window.
    pub display: Box2i,
}

impl Windows {
    /// `width x height` at the origin, display window = data window: the
    /// windows of a plane that carries no placement of its own (a numpy array).
    pub fn origin(width: usize, height: usize) -> Result<Self> {
        let w = i32::try_from(width).map_err(|_| Error::Metadata("width too large".into()))?;
        let h = i32::try_from(height).map_err(|_| Error::Metadata("height too large".into()))?;
        if w == 0 || h == 0 {
            return Err(Error::Metadata("encode requires non-zero width/height".into()));
        }
        let data = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } };
        Ok(Self { data, display: data })
    }

    /// The windows of `image`.
    pub fn of(image: &Image) -> Self {
        Self { data: image.data_window, display: image.display_window }
    }

    /// Data-window `(width, height)` in pixels.
    pub fn size(&self) -> Result<(usize, usize)> {
        let dim = |lo: i32, hi: i32| {
            usize::try_from(i64::from(hi) - i64::from(lo) + 1)
                .map_err(|_| Error::Metadata(format!("empty data window [{lo}, {hi}]")))
        };
        Ok((dim(self.data.min.x, self.data.max.x)?, dim(self.data.min.y, self.data.max.y)?))
    }
}

/// Pack ranked slots into a fresh [`Image`] with `windows` (encode from
/// scratch).
pub fn pack_image<F>(
    windows: Windows,
    crypto_name: &str,
    ranked_layers: usize,
    pixel_contributions: F,
) -> Result<(Image, Vec<RankedLayer>)>
where
    F: FnMut(usize) -> Vec<Contribution>,
{
    let (width, height) = windows.size()?;
    let (channels, ranked_out) = pack_channels(
        width,
        height,
        crypto_name,
        ranked_layers,
        Spelling::default(),
        pixel_contributions,
    )?;

    let mut image = Image::new(windows.data);
    image.display_window = windows.display;
    for (name, data) in channels {
        image.insert_channel(&name, data)?;
    }
    Ok((image, ranked_out))
}

/// Re-encode one Cryptomatte type **inside** an existing image.
///
/// The result keeps `src`'s data and display windows, compression, line order
/// and every channel that does not belong to `crypto_name`; only that type's
/// ranked channels are replaced (stale higher-rank layers are dropped). This is
/// the single read-all/write-all codepath inject needs so that beauty channels,
/// other AOVs and other Cryptomatte types survive.
///
/// The re-encoded channels keep the suffix spelling the source used for this type
/// (`.red` stays `.red`, see [`Spelling`]); a type with no ranked channels yet gets `.R`.
///
/// The re-encoded channels are FLOAT even if the source stored them as HALF: a
/// Cryptomatte ID is a 32-bit float bit pattern (spec 1.2.0 p.6), so HALF IDs
/// cannot round-trip and are not written back.
pub fn replace_ranked_channels<F>(
    src: &Image,
    crypto_name: &str,
    ranked_layers: usize,
    pixel_contributions: F,
) -> Result<(Image, Vec<RankedLayer>)>
where
    F: FnMut(usize) -> Vec<Contribution>,
{
    let width = usize::try_from(src.width())
        .map_err(|_| Error::Metadata(format!("invalid image width {}", src.width())))?;
    let height = usize::try_from(src.height())
        .map_err(|_| Error::Metadata(format!("invalid image height {}", src.height())))?;
    let (channels, ranked_out) = pack_channels(
        width,
        height,
        crypto_name,
        ranked_layers,
        Spelling::detect(crypto_name, src.channels().map(|(n, _)| n)).unwrap_or_default(),
        pixel_contributions,
    )?;

    let mut image = Image::new(src.data_window);
    image.display_window = src.display_window;
    image.compression = src.compression;
    image.line_order = src.line_order;
    for (name, data) in src.channels() {
        if ranked_channel_of(crypto_name, name).is_some() {
            continue; // replaced below (or stale: this type shrank)
        }
        image.insert_channel(name, data.clone())?;
    }
    for (name, data) in channels {
        image.insert_channel(&name, data)?;
    }
    Ok((image, ranked_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::name_to_float_id;

    #[test]
    fn ranks_by_coverage_descending() {
        let a = name_to_float_id("a");
        let b = name_to_float_id("b");
        let c = name_to_float_id("c");
        let ranked = merge_and_rank(
            &[
                Contribution::new(a, 0.1),
                Contribution::new(b, 0.7),
                Contribution::new(c, 0.2),
            ],
            2,
        );
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].id.to_bits(), b.to_bits());
        assert_eq!(ranked[1].id.to_bits(), c.to_bits());
    }

    #[test]
    fn merges_same_id_bits() {
        let id = name_to_float_id("bunny");
        let ranked = merge_and_rank(
            &[Contribution::new(id, 0.25), Contribution::new(id, 0.25)],
            4,
        );
        assert_eq!(ranked.len(), 1);
        assert!((ranked[0].coverage - 0.5).abs() < 1e-6);
    }

    #[test]
    fn over_scales_existing() {
        let id_a = name_to_float_id("a");
        let id_b = name_to_float_id("b");
        let merged = merge_contribution(
            &[Contribution::new(id_b, 1.0)],
            id_a,
            0.25,
            MergeOp::Over,
        );
        let ranked = merge_and_rank(&merged, 4);
        let mut by: HashMap<u32, f32> = HashMap::new();
        for c in ranked {
            by.insert(c.id.to_bits(), c.coverage);
        }
        assert!((by[&id_a.to_bits()] - 0.25).abs() < 1e-6);
        assert!((by[&id_b.to_bits()] - 0.75).abs() < 1e-6);
    }

    /// S3 regression: the ranked planes are converted once, not once per pixel
    /// (the quadratic HALF inject came from a per-pixel conversion).
    #[test]
    fn ranked_planes_convert_each_channel_once() {
        let n = 64 * 64;
        let id = name_to_float_id("bunny");
        // FLOAT IDs (HALF IDs are rejected) and HALF coverage: the plane that
        // needs a real conversion.
        let ids = ChannelData::Float(vec![id; n]);
        let covs = ChannelData::Half(vec![imath_rs::Half::from(1.0f32); n]);
        let ranked = [RankedLayer {
            base: "L00".into(),
            id0: Some("L00.R".into()),
            cov0: Some("L00.G".into()),
            id1: None,
            cov1: None,
        }];

        let mut fetches = 0usize;
        let planes = RankedPlanes::from_channels(&ranked, n, |name| {
            fetches += 1;
            Ok(if name == "L00.R" { &ids } else { &covs })
        })
        .expect("planes");
        for i in 0..n {
            planes.contributions_at(i).expect("pixel");
        }
        assert_eq!(fetches, 2, "one fetch/conversion per ranked channel");
        assert_eq!(planes.contributions_at(0).expect("pixel")[0].id.to_bits(), id.to_bits());
        assert!(planes.contributions_at(n).is_err(), "out-of-range pixel errors");
    }

    #[test]
    fn replace_keeps_other_channels_and_windows() {
        let dw = Box2i {
            min: V2i { x: 3, y: 5 },
            max: V2i { x: 6, y: 6 },
        };
        let n = 8usize;
        let mut src = Image::new(dw);
        src.display_window = Box2i {
            min: V2i { x: 0, y: 0 },
            max: V2i { x: 9, y: 9 },
        };
        src.compression = exr_core::Compression::Rle;
        src.insert_channel("R", ChannelData::Float(vec![0.25; n])).unwrap();
        // A stale second ranked layer that the re-encode must drop.
        for c in ["CryptoObject00.R", "CryptoObject00.G", "CryptoObject01.R"] {
            src.insert_channel(c, ChannelData::Float(vec![0.0; n])).unwrap();
        }

        let id = name_to_float_id("bunny");
        let (out, ranked) =
            replace_ranked_channels(&src, "CryptoObject", 1, |_| vec![Contribution::new(id, 1.0)])
                .unwrap();

        assert_eq!(out.data_window, src.data_window);
        assert_eq!(out.display_window, src.display_window);
        assert_eq!(out.compression, src.compression);
        assert_eq!(ranked.len(), 1);
        assert!(out.channel("R").is_some(), "non-crypto channel survives");
        assert!(out.channel("CryptoObject01.R").is_none(), "stale rank dropped");
        match out.channel("CryptoObject00.R").unwrap() {
            ChannelData::Float(v) => assert_eq!(v[0].to_bits(), id.to_bits()),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn pack_writes_rgba_slots() {
        let id = name_to_float_id("bunny");
        let (image, ranked) = pack_image(Windows::origin(2, 1).unwrap(), "CryptoObject", 1, |i| {
            if i == 0 {
                vec![Contribution::new(id, 1.0)]
            } else {
                vec![]
            }
        })
        .unwrap();
        assert_eq!(ranked.len(), 1);
        match image.channel("CryptoObject00.R").unwrap() {
            ChannelData::Float(v) => {
                assert_eq!(v[0].to_bits(), id.to_bits());
                assert_eq!(v[1].to_bits(), 0);
            }
            _ => panic!("expected float"),
        }
        match image.channel("CryptoObject00.G").unwrap() {
            ChannelData::Float(v) => {
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 0.0);
            }
            _ => panic!("expected float"),
        }
    }
}
