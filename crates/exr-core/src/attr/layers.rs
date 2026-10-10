//! Channel grouping into Nuke-style "layers" — the shared source of truth.
//!
//! EXR encodes layer membership in the channel NAME via a dot: `R,G,B,A` is the
//! main (unlayered) layer; `diffuse.R/G/B`, `depth.Z`, `N.x/y/z`, `motion.u/v`,
//! `cryptomatte.00.R/G/B/A` are named layers. This is the same `rfind('.')` rule
//! as the 1:1-ported [`ChannelList::layers`](crate::attr::ChannelList::layers) /
//! `channelsInLayer` (a dot that is neither leading nor trailing splits
//! `layer.channel`).
//!
//! `ChannelList::layers()` returns only the *named* layer set (it deliberately
//! omits the unlayered/main channels and gives no per-layer ordering, matching
//! C++). Consumers that need to **enumerate every group, main included, with a
//! stable component order** (the GPU upload layer in `exr-gpu`, the viewer's
//! layer picker in `exr-view`) use [`group_layers`] here instead — so the rule
//! lives in ONE place and the two crates cannot drift.
//!
//! ON TOP of the dot-split we impose a CANONICAL component order within a layer
//! (`R<G<B<A<X<Y<Z<U<V`, unknowns last alphabetically). The EXR header stores
//! channels alphabetically (`A<B<G<R`), which is the wrong order for packing
//! RGBA. This ordering is a presentation/packing convention, NOT something the
//! on-disk format asserts — hence it lives here, beside `ChannelList`, not in the
//! byte-faithful `channel_list` port.

use crate::attr::PixelType;
use std::collections::BTreeMap;

/// One logical channel group ("layer").
///
/// `channels` are indices into the SAME ordered sequence that was passed to
/// [`group_layers`] (e.g. the caller's channel `Vec`), already in canonical
/// component order — so `channels[0]` is the layer's first component (R / X / …).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerGroup {
    /// Layer prefix. Empty string = the main (unlayered) `R/G/B/A/Y/Z` group.
    pub name: String,
    /// Indices into the input channel sequence, in canonical component order.
    pub channels: Vec<usize>,
    /// The first channel's [`PixelType`] (layers are usually type-uniform; the
    /// caller verifies uniformity where the per-channel types matter).
    pub pixel_type: PixelType,
}

impl LayerGroup {
    /// Number of real channels in this layer (before any RGBA padding).
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Whether the layer has no channels (cannot happen for a parsed layer).
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// A human label: `"<main>"` for the unlayered group, else the prefix.
    pub fn label(&self) -> &str {
        if self.name.is_empty() { "<main>" } else { &self.name }
    }
}

/// Group an ordered channel sequence into layers.
///
/// Input is any iterator of `(channel_name, pixel_type)` in the order the caller
/// indexes them (typically the EXR header / `ChannelList` alphabetical order).
/// The returned `Vec` is: the main (unlayered) layer first if present, then named
/// layers in `BTreeMap` (byte-wise) order — matching C++ `std::map` iteration.
/// Within each layer, channels are sorted into canonical component order (see the
/// module docs). Indices in [`LayerGroup::channels`] point back into the input
/// sequence.
pub fn group_layers<'a, I>(channels: I) -> Vec<LayerGroup>
where
    I: IntoIterator<Item = (&'a str, PixelType)>,
{
    // Collect once so we can index by position and read suffixes during the sort.
    let chans: Vec<(&str, PixelType)> = channels.into_iter().collect();

    // prefix → channel indices (BTreeMap = deterministic alphabetical layer order).
    let mut by_prefix: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, (name, _)) in chans.iter().enumerate() {
        by_prefix.entry(layer_prefix(name).to_string()).or_default().push(i);
    }

    let build = |name: String, mut idxs: Vec<usize>| -> LayerGroup {
        // Canonical component order: known suffix priority, then alphabetical.
        idxs.sort_by(|&a, &b| {
            let sa = channel_suffix(chans[a].0);
            let sb = channel_suffix(chans[b].0);
            component_priority(sa)
                .cmp(&component_priority(sb))
                .then_with(|| sa.to_ascii_lowercase().cmp(&sb.to_ascii_lowercase()))
        });
        // The layer's representative pixel type is its first (canonical) channel's.
        let pixel_type = idxs.first().map(|&i| chans[i].1).unwrap_or(PixelType::Half);
        LayerGroup { name, channels: idxs, pixel_type }
    };

    let mut layers = Vec::with_capacity(by_prefix.len());
    // Main (unlayered) first.
    if let Some(idxs) = by_prefix.remove("") {
        layers.push(build(String::new(), idxs));
    }
    // Named layers, alphabetical (BTreeMap iteration order).
    for (name, idxs) in by_prefix {
        layers.push(build(name, idxs));
    }
    layers
}

/// The layer prefix of a channel name — the SAME rule as `ChannelList::layers()`:
/// the substring before the LAST dot, but only when that dot is neither leading
/// (`pos != 0`) nor trailing (`pos + 1 < len`). No valid dot → the empty (main)
/// layer.
pub fn layer_prefix(name: &str) -> &str {
    if let Some(pos) = name.rfind('.')
        && pos != 0 && pos + 1 < name.len()
    {
        return &name[..pos];
    }
    ""
}

/// The channel suffix — the part after the last dot, or the whole name if there
/// is no (valid) dot (`"diffuse.R"` → `"R"`, `"R"` → `"R"`). Paired with
/// [`layer_prefix`]: `layer_prefix` + `'.'` + `channel_suffix` reconstructs the
/// name for a layered channel.
pub fn channel_suffix(name: &str) -> &str {
    if let Some(pos) = name.rfind('.')
        && pos != 0 && pos + 1 < name.len()
    {
        return &name[pos + 1..];
    }
    name
}

/// Canonical component priority for a channel suffix (lower = earlier slot).
/// Colour `R/G/B/A` → 0..3, vector/normal `X/Y/Z` → 4..6, motion `U/V` → 7..8,
/// everything else → 255 (sorted alphabetically among themselves, after the known
/// set). This is the Nuke RGBA-first convention, shared by GPU packing and the
/// viewer so both order identically.
pub fn component_priority(suffix: &str) -> u8 {
    match suffix.to_ascii_lowercase().as_str() {
        "r" | "red" => 0,
        "g" | "green" => 1,
        "b" | "blue" => 2,
        "a" | "alpha" => 3,
        "x" => 4,
        "y" => 5,
        "z" => 6,
        "u" => 7,
        "v" => 8,
        _ => 255,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names<'a>(g: &LayerGroup, chans: &[(&'a str, PixelType)]) -> Vec<&'a str> {
        g.channels.iter().map(|&i| chans[i].0).collect()
    }

    #[test]
    fn groups_and_orders_like_nuke() {
        // Header order is alphabetical; expect main-first + canonical RGBA order.
        let chans = [
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
        ];
        let layers = group_layers(chans.iter().copied());
        // main first; then BTreeMap (byte) order: "N"(0x4E) before "depth"/"diffuse"(0x64)
        // because uppercase sorts before lowercase.
        assert_eq!(layers[0].name, "");
        assert_eq!(names(&layers[0], &chans), ["R", "G", "B", "A"]); // canonical, not A<B<G<R
        assert_eq!(layers[0].pixel_type, PixelType::Half);
        assert_eq!(layers[1].name, "N");
        assert_eq!(names(&layers[1], &chans), ["N.x", "N.y", "N.z"]);
        assert_eq!(layers[1].pixel_type, PixelType::Float);
        assert_eq!(layers[2].name, "depth");
        assert_eq!(names(&layers[2], &chans), ["depth.Z"]);
        assert_eq!(layers[3].name, "diffuse");
        assert_eq!(names(&layers[3], &chans), ["diffuse.R", "diffuse.G", "diffuse.B"]);
    }

    #[test]
    fn single_and_two_component_layers() {
        let chans = [
            ("Z", PixelType::Float),
            ("velocity.x", PixelType::Half),
            ("velocity.y", PixelType::Half),
        ];
        let layers = group_layers(chans.iter().copied());
        // "Z" has no dot → main bucket; "velocity" is a named 2-component layer.
        assert_eq!(layers[0].name, "");
        assert_eq!(names(&layers[0], &chans), ["Z"]);
        assert_eq!(layers[1].name, "velocity");
        assert_eq!(names(&layers[1], &chans), ["velocity.x", "velocity.y"]);
    }

    #[test]
    fn layer_prefix_and_suffix_rules() {
        // A dot inside the layer name is valid (cryptomatte.00).
        assert_eq!(layer_prefix("cryptomatte.00.R"), "cryptomatte.00");
        assert_eq!(channel_suffix("cryptomatte.00.R"), "R");
        assert_eq!(layer_prefix("R"), "");
        assert_eq!(channel_suffix("R"), "R");
        assert_eq!(layer_prefix(".R"), ""); // leading dot → main
        assert_eq!(channel_suffix(".R"), ".R"); // not a valid split → whole name
        assert_eq!(layer_prefix("diffuse."), ""); // trailing dot → main
        assert_eq!(channel_suffix("diffuse."), "diffuse.");
    }

    #[test]
    fn unknown_suffixes_sort_after_known_alphabetically() {
        let chans = [
            ("aov.foo", PixelType::Float),
            ("aov.bar", PixelType::Float),
            ("aov.R", PixelType::Float),
        ];
        let layers = group_layers(chans.iter().copied());
        assert_eq!(layers[0].name, "aov");
        // R (known, prio 0) first; then unknowns alphabetically: bar < foo.
        assert_eq!(names(&layers[0], &chans), ["aov.R", "aov.bar", "aov.foo"]);
    }
}
