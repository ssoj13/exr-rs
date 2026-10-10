//! Nuke-style channel groups ("layers") for GPU upload.
//!
//! EXR encodes layer membership in the channel NAME via a dot: `R,G,B,A` is the
//! main (unlayered) layer; `diffuse.R/G/B`, `depth.Z`, `N.x/y/z`, `motion.u/v`,
//! `cryptomatte.00.R/G/B/A` are named layers.
//!
//! The grouping RULE (dot-split, main-bucket, canonical `R<G<B<A<X<Y<Z<U<V`
//! component order) lives ONCE in [`exr_core::attr::layers`]; this module is a
//! thin adapter that maps the core's [`LayerGroup`](exr_core::attr::LayerGroup)
//! onto a GPU-side [`Layer`] whose channel indices point into
//! [`SourceImage::channels`]. Keeping the rule in `exr-core` means exr-gpu's
//! packing order and exr-view's layer picker cannot drift.

use exr_core::attr::PixelType;

use crate::source::SourceImage;

/// One logical channel group. `channels` are indices into
/// [`SourceImage::channels`], already in canonical component order.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Layer prefix. Empty string = the main (unlayered) `R/G/B/A/Z/Y` channels.
    pub name: String,
    /// Indices into the source's channel list, in canonical component order
    /// (so index 0 → GPU component 0).
    pub(crate) channels: Vec<usize>,
    /// The (first channel's) pixel type. Uniformity across the layer is verified
    /// at upload time, where the per-channel types are available.
    pub pixel_type: PixelType,
    /// Texture width: every channel of a layer shares one sample grid.
    pub width: u32,
    /// Texture height.
    pub height: u32,
    /// `(x_sampling, y_sampling)` of the layer's channels; `(1, 1)` = full res.
    pub sampling: (i32, i32),
}

impl Layer {
    /// Number of real channels in this layer (before any RGBA padding).
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Whether the layer has no channels (cannot happen for a parsed layer).
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// A human label: `"<main>"` for the unlayered group, else the prefix; a
    /// subsampled layer adds its sampling (`"<main> 2x2"`), since the same prefix
    /// can then name two layers (luminance `Y` at 1x1, chroma `RY/BY` at 2x2).
    pub fn label(&self) -> String {
        let base = if self.name.is_empty() { "<main>" } else { &self.name };
        match self.sampling {
            (1, 1) => base.to_string(),
            (xs, ys) => format!("{base} {xs}x{ys}"),
        }
    }
}

/// Group a source image's channels into layers.
///
/// Channels are first split by sampling (a texture holds one sample grid), full
/// resolution first, then each subsampled class in `(x, y)` order. Within a
/// class the grouping rule + ordering is [`exr_core::attr::group_layers`] (the
/// single source of truth): the main (unlayered) layer first if present, then
/// named layers in byte order; within each layer, canonical component order.
pub(crate) fn group_layers(src: &SourceImage) -> Vec<Layer> {
    let mut classes: Vec<(i32, i32)> = src.channels.iter().map(|c| c.sampling).collect();
    classes.sort_unstable();
    classes.dedup();
    // `(1, 1)` sorts first already: it is the smallest sampling there is.
    let mut out = Vec::new();
    for class in classes {
        let members: Vec<usize> =
            (0..src.channels.len()).filter(|&i| src.channels[i].sampling == class).collect();
        let Some(&first) = members.first() else { continue };
        let (width, height) = (src.channels[first].width, src.channels[first].height);
        let groups = exr_core::attr::group_layers(
            members.iter().map(|&i| (src.channels[i].name.as_str(), src.channels[i].pixel_type)),
        );
        out.extend(groups.into_iter().map(|g| Layer {
            name: g.name,
            // `group_layers` indexes into `members`; map back to source indices.
            channels: g.channels.iter().map(|&k| members[k]).collect(),
            pixel_type: g.pixel_type,
            width,
            height,
            sampling: class,
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceChannel;

    /// Build a 1x1 SourceImage with the given (name, type) channels for grouping tests.
    fn src(chs: &[(&str, PixelType)]) -> SourceImage {
        SourceImage {
            width: 1,
            height: 1,
            channels: chs
                .iter()
                .map(|(n, t)| SourceChannel {
                    name: n.to_string(),
                    pixel_type: *t,
                    bytes: vec![0u8; exr_core::misc::pixel_type_size(*t).unwrap()],
                    width: 1,
                    height: 1,
                    sampling: (1, 1),
                })
                .collect(),
        }
    }

    fn names<'a>(layer: &Layer, s: &'a SourceImage) -> Vec<&'a str> {
        layer.channels.iter().map(|&i| s.channels[i].name.as_str()).collect()
    }

    #[test]
    fn groups_and_orders_like_nuke() {
        // End-to-end through the SourceImage adapter: header order is alphabetical,
        // we expect main-first + canonical RGBA order from the shared core helper.
        let s = src(&[
            ("A", PixelType::Half),
            ("B", PixelType::Half),
            ("G", PixelType::Half),
            ("R", PixelType::Half),
            ("depth.Z", PixelType::Float),
            ("diffuse.B", PixelType::Half),
            ("diffuse.G", PixelType::Half),
            ("diffuse.R", PixelType::Half),
            ("N.x", PixelType::Float),
            ("N.y", PixelType::Float),
            ("N.z", PixelType::Float),
        ]);
        let layers = group_layers(&s);
        // main first, then BTreeMap (byte) order like C++ std::map: "N"(0x4E) before
        // "depth"/"diffuse"(0x64) since uppercase sorts before lowercase.
        assert_eq!(layers[0].name, "");
        assert_eq!(names(&layers[0], &s), ["R", "G", "B", "A"]); // canonical, not A<B<G<R
        assert_eq!(layers[1].name, "N");
        assert_eq!(names(&layers[1], &s), ["N.x", "N.y", "N.z"]);
        assert_eq!(layers[2].name, "depth");
        assert_eq!(names(&layers[2], &s), ["depth.Z"]);
        assert_eq!(layers[3].name, "diffuse");
        assert_eq!(names(&layers[3], &s), ["diffuse.R", "diffuse.G", "diffuse.B"]);
    }

    /// Subsampled channels form their own layers at their own size: a
    /// luminance-chroma file is `<main>` (Y, full res) plus `<main> 2x2`
    /// (RY, BY), never one texture mixing two sample grids.
    #[test]
    fn subsampled_channels_group_by_sampling() {
        let ch = |n: &str, w, h, sampling| SourceChannel {
            name: n.to_string(),
            pixel_type: PixelType::Half,
            bytes: vec![0u8; (w * h * 2) as usize],
            width: w,
            height: h,
            sampling,
        };
        let s = SourceImage {
            width: 8,
            height: 6,
            channels: vec![
                ch("BY", 4, 3, (2, 2)),
                ch("RY", 4, 3, (2, 2)),
                ch("Y", 8, 6, (1, 1)),
                ch("depth.Z", 8, 3, (1, 2)),
            ],
        };
        let layers = group_layers(&s);
        let summary: Vec<(String, Vec<&str>, (u32, u32))> =
            layers.iter().map(|l| (l.label(), names(l, &s), (l.width, l.height))).collect();
        assert_eq!(
            summary,
            [
                ("<main>".to_string(), vec!["Y"], (8, 6)),
                ("depth 1x2".to_string(), vec!["depth.Z"], (8, 3)),
                // `RY`/`BY` are outside the known component set, so the core
                // rule (`component_priority`) orders them alphabetically.
                ("<main> 2x2".to_string(), vec!["BY", "RY"], (4, 3)),
            ]
        );
    }
}
