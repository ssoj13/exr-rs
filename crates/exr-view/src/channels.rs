//! The channel tree: every layer of the frame with its channels, clickable.
//!
//! A click on a layer shows that layer ([`Selection::layer`]); a click on a channel
//! shows it alone in grey ([`Selection::channel`]). Drawn with `egui-outliner`, the
//! same widget (and look: role icons, dimmed type suffix, filter box, 24 pt rows) as
//! warpbro's world outliner. The widget keeps no tree state of its own and the tree's
//! selection is what the frame actually shows (see [`shown`]), so it can never
//! disagree with what is on screen.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use soluwell_panels::outliner::{ContextItem, OutlinerAction, OutlinerConfig, OutlinerModel, TreeNode};
use soluwell_config::phosphor::regular as ph;
use soluwell_config::icons;
use exr_core::attr::{LayerGroup, PixelType};

use crate::image::{DecodedFrame, Selection};

/// The id of the node showing `layer` (and `channel`, if any). The outliner keys its
/// fold state by id, so an id made from names keeps "depth" folded across files and
/// frames, where a position would hand one file's fold state to another file's layer
/// at the same index. Hashes exactly what `Selection`'s derived `Hash` hashes, from
/// borrowed parts, so the cache-hit path needs no `Selection` (no `String` clones).
fn hash_id(layer: &str, channel: Option<&str>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    layer.hash(&mut h);
    channel.hash(&mut h);
    h.finish()
}

/// The id of the node of `sel`.
fn node_id(sel: &Selection) -> u64 {
    hash_id(&sel.layer, sel.channel.as_deref())
}

/// Split a layer-name segment into lowercase tokens at every non-alphanumeric
/// character, camelCase step (`worldPosition`), acronym end (`ZDepth` is `z`,
/// `depth`) and letter/digit step (`Object00`).
fn tokens(s: &str) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in cs.iter().enumerate() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if i > 0 && !cur.is_empty() {
            let p = cs[i - 1];
            // `ZDepth` ends its acronym before the capital that starts a word, but the
            // plural `IDs` keeps its `s`: that lowercase letter is not a word start.
            let word_starts = cs.get(i + 1).is_some_and(|n| {
                n.is_lowercase() && !(*n == 's' && cs.get(i + 2).is_none_or(|m| !m.is_lowercase()))
            });
            let step = (p.is_lowercase() && c.is_uppercase())
                || (p.is_ascii_digit() != c.is_ascii_digit())
                || (p.is_uppercase() && c.is_uppercase() && word_starts);
            if step {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Words that tie a role keyword to what follows (`depth` `of` `field`): the keyword
/// then only modifies another concept, so it does not name this layer's role.
const LINKS: [&str; 10] = ["of", "for", "to", "from", "in", "on", "with", "and", "or", "per"];

/// The icon of a layer from its role, by name first and by components second.
///
/// Name rules look at the last dot-separated segment (`left.N` is normals) and match
/// whole lowercase tokens of it (see [`tokens`]), so a renderer prefix is harmless
/// (`VRayZDepth` is depth, `VRayMtlID` an id) and a word inside another is not a hit
/// (`normalizedDepth` is depth not normals, `overflow` is not motion). One rule
/// applies to every role: a keyword followed by a link word ([`LINKS`]) names no
/// role, so `depthOfField` is not depth. Vendor names are tokens too (`Pz` depth,
/// `Ng` normals, `Ps` position, `motionvector`, `forward`/`backward` motion), and
/// position is a name role, so a `P.R/.G/.B` is not colour.
///
/// Without a name role the components decide, case-insensitively and with the main
/// layer's leading dot (`.R`) ignored: a lone `A` is alpha, a lone `Z` depth, colour
/// needs all of `R`, `G` and `B`, or a `Y` with nothing but `Y`, `RY`, `BY` and `A`
/// beside it (so `x/y/z` stays generic). View layers (`left`) and beauty passes get
/// the same colour icon.
fn layer_icon(name: &str, comps: &[&str]) -> &'static str {
    let seg = name.rsplit('.').next().unwrap_or("");
    let toks = tokens(seg);
    let named = |words: &[&str]| {
        toks.iter().enumerate().any(|(j, t)| {
            words.contains(&t.as_str()) && !toks.get(j + 1).is_some_and(|n| LINKS.contains(&n.as_str()))
        })
    };
    let comps: Vec<&str> = comps.iter().map(|c| c.trim_start_matches('.')).collect();
    let has = |c: &str| comps.iter().any(|k| k.eq_ignore_ascii_case(c));
    let only = |c: &str| matches!(comps.as_slice(), [k] if k.eq_ignore_ascii_case(c));
    let luma = has("Y")
        && comps.iter().all(|k| ["y", "ry", "by", "a"].iter().any(|w| k.eq_ignore_ascii_case(w)));
    if named(&["crypto", "cryptomatte"]) {
        ph::FINGERPRINT
    } else if named(&["mask", "masks", "matte", "mattes", "multimatte"]) {
        ph::MASK_HAPPY
    } else if named(&[
        "id", "ids", "index", "idx", "objectid", "objectids", "materialid", "materialids", "objid",
        "matid", "primid",
    ]) {
        ph::HASH
    } else if named(&["depth", "zdepth", "z", "pz"]) || only("Z") {
        ph::MOUNTAINS
    } else if named(&["n", "nn", "ng", "normal", "normals"]) {
        ph::COMPASS
    } else if named(&["p", "pref", "ps", "pos", "position", "wp", "pworld"]) {
        ph::CROSSHAIR
    } else if named(&[
        "motion", "motionvector", "motionvectors", "velocity", "vel", "vector", "mv", "flow", "forward",
        "backward",
    ]) {
        ph::WIND
    } else if named(&["alpha"]) || only("A") {
        ph::CIRCLE_HALF
    } else if (has("R") && has("G") && has("B")) || luma {
        ph::IMAGE
    } else {
        ph::STACK
    }
}

/// The selection `frame` actually shows for the requested `view`: a channel absent from
/// the frame falls back to its layer, a layer absent falls back to the main (first)
/// one, exactly as [`DecodedFrame::resolve`] decides what is drawn. `None` only for a
/// frame with no layers.
fn shown(frame: &DecodedFrame, view: &Selection) -> Option<Selection> {
    match frame.resolve(view)? {
        // A solo channel is a one-channel group named after the channel.
        Cow::Owned(g) => {
            let i = *g.channels.first()?;
            let layer = frame.layers.iter().find(|l| l.channels.contains(&i))?;
            Some(Selection::channel(&layer.name, &g.name))
        }
        Cow::Borrowed(g) => Some(Selection::layer(&g.name)),
    }
}

/// The tree of one frame for one requested view, what each node id shows, and the
/// inputs it was built from (so [`Built::fits`] can tell when it is stale).
struct Built {
    names: Vec<String>,
    types: Vec<PixelType>,
    /// The grouping the tree shows (name, channel indices, order), compared as is so
    /// the key never relies on how a `DecodedFrame` constructor happened to group.
    layers: Vec<LayerGroup>,
    view: Selection,
    model: OutlinerModel,
    ids: HashMap<u64, Selection>,
    /// Built once with the tree rather than per frame; the same for every tree. The
    /// widget's default glyphs are already the shared phosphor set.
    config: OutlinerConfig,
}

/// The outliner settings of the channel tree.
fn config() -> OutlinerConfig {
    OutlinerConfig::default()
        .with_filter(true)
        .with_row_height(24.0)
        .with_context_items([ContextItem::new("show", "Show"), ContextItem::new("copy", "Copy name")])
}

/// True when `frame`'s channel types are `types`, compared in place:
/// [`DecodedFrame::channel_types`] would allocate a `Vec` on every frame.
fn same_types(frame: &DecodedFrame, types: &[PixelType]) -> bool {
    match &frame.file_types {
        Some(t) => t == types,
        None => {
            frame.channels.len() == types.len()
                && frame.channels.iter().zip(types).all(|((_, c), t)| c.pixel_type() == *t)
        }
    }
}

impl Built {
    /// The tree of `frame` with what `view` resolves to selected.
    fn new(frame: &DecodedFrame, view: &Selection) -> Self {
        let types = frame.channel_types();
        let shown = shown(frame, view);
        let mut ids = HashMap::new();
        let mut roots = Vec::new();
        let mut selected = None;
        let mut add = |sel: Selection, ids: &mut HashMap<u64, Selection>| {
            let id = node_id(&sel);
            if shown.as_ref() == Some(&sel) {
                selected = Some(id);
            }
            ids.insert(id, sel);
            id
        };
        for group in &frame.layers {
            let layer_id = add(Selection::layer(&group.name), &mut ids);
            let prefix = format!("{}.", group.name);
            let mut children = Vec::new();
            let mut comps = Vec::new();
            for &i in &group.channels {
                let Some(name) = frame.channel_names.get(i) else {
                    continue;
                };
                let id = add(Selection::channel(&group.name, name), &mut ids);
                // The main layer has no prefix to strip: a leading dot (`.R`) belongs to
                // the channel's name, not to a layer separator.
                let short = if group.name.is_empty() {
                    name.as_str()
                } else {
                    name.strip_prefix(&prefix).unwrap_or(name)
                };
                comps.push(short);
                let mut node = TreeNode::new(id, short).with_icon(icons::DOT);
                if let Some(t) = types.get(i) {
                    node = node.with_type(format!("{t:?}"));
                }
                children.push(node);
            }
            roots.push(
                TreeNode::new(layer_id, group.label())
                    .with_icon(layer_icon(&group.name, &comps))
                    .with_children(children)
                    .with_open(shown.as_ref().is_some_and(|s| s.layer == group.name)),
            );
        }
        let mut model = OutlinerModel::with_roots(roots);
        model.selection.extend(selected);
        Self {
            names: frame.channel_names.clone(),
            types,
            layers: frame.layers.clone(),
            view: view.clone(),
            model,
            ids,
            config: config(),
        }
    }

    /// True while this is still exactly the tree of `frame` with `view` requested.
    /// Compares in place, allocating nothing.
    fn fits(&self, frame: &DecodedFrame, view: &Selection) -> bool {
        self.view == *view
            && self.names == frame.channel_names
            && self.layers == frame.layers
            && same_types(frame, &self.types)
    }

    /// What "Copy name" puts on the clipboard for node `id`: a channel's full name, a
    /// named layer's name, and for the unnamed main layer (which has no name) its
    /// channel names in tree order, e.g. `R G B A`. Main-layer labels are full channel
    /// names, so a leading-dot `.R` is copied as `.R`.
    fn copy_text(&self, id: u64) -> Option<String> {
        let sel = self.ids.get(&id)?;
        if let Some(c) = &sel.channel {
            return Some(c.clone());
        }
        if !sel.layer.is_empty() {
            return Some(sel.layer.clone());
        }
        let root = self.model.roots.iter().find(|r| r.id == id)?;
        let names: Vec<&str> = root.children.iter().map(|c| c.label.as_str()).collect();
        (!names.is_empty()).then(|| names.join(" "))
    }
}

/// The tree for `frame` and `view`: `old` itself while it still fits, else a rebuilt
/// one. Sequence frames share their channel set, so playback reuses one tree instead of
/// rebuilding (and re-hashing every node id) per displayed frame.
fn fresh(old: Option<Arc<Built>>, frame: &DecodedFrame, view: &Selection) -> Arc<Built> {
    match old {
        Some(b) if b.fits(frame, view) => b,
        _ => Arc::new(Built::new(frame, view)),
    }
}

/// The key of a node's persisted fold flag: the outliner's own (pinned egui-outliner
/// `render_node`), so there is no second map of expanded branches.
fn open_key(base: egui::Id, id: u64) -> egui::Id {
    base.with(("egui_outliner_open", id))
}

/// Fold state on a layer switch: once, fold every group (also clearing flags left by
/// the earlier all-open policy); then, when the REQUESTED layer changes and the frame
/// has it, open its branch. Keyed on the request, not on what is shown: a frame
/// lacking the layer falls back to main for display, and that must neither re-open
/// main nor undo the user's fold of it. Manual folds are otherwise left alone.
fn reveal(ctx: &egui::Context, base: egui::Id, model: &OutlinerModel, view: &Selection) {
    use soluwell_config::state;
    let (init_key, last_key) = (base.with("channel_tree_init"), base.with("channel_tree_last_layer"));
    if !state::get::<bool>(ctx, init_key).unwrap_or(false) {
        for root in &model.roots {
            state::set(ctx, open_key(base, root.id), false);
        }
        state::set(ctx, init_key, true);
    }
    let want = hash_id(&view.layer, None);
    if state::get::<u64>(ctx, last_key) != Some(want) && model.roots.iter().any(|r| r.id == want) {
        state::set(ctx, open_key(base, want), true);
        state::set(ctx, last_key, want);
    }
}

/// Draw the channel tree of `frame`; the selection the user clicked, if any.
pub fn show(ui: &mut egui::Ui, frame: &DecodedFrame, view: &Selection) -> Option<Selection> {
    // The tree lives in egui memory next to the fold state, as an `Arc`: reading it
    // back is a refcount, not a copy of the tree, and the caller owns nothing.
    let base_id = ui.id();
    let cache_id = base_id.with("channel_tree_cache");
    let old: Option<Arc<Built>> = soluwell_config::state::get(ui.ctx(), cache_id);
    let built = fresh(old.clone(), frame, view);
    if !old.is_some_and(|o| Arc::ptr_eq(&o, &built)) {
        soluwell_config::state::set(ui.ctx(), cache_id, built.clone());
    }
    let (model, ids) = (&built.model, &built.ids);
    reveal(ui.ctx(), base_id, model, view);
    let open_id = |id| open_key(base_id, id);
    let mut expand_all = None;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Channels").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button(ph::ARROWS_IN_LINE_VERTICAL).on_hover_text("Collapse every channel group").clicked() {
                expand_all = Some(false);
            }
            if ui.small_button(ph::ARROWS_OUT_LINE_VERTICAL).on_hover_text("Expand every channel group").clicked() {
                expand_all = Some(true);
            }
        });
    });
    if let Some(open) = expand_all {
        for root in &model.roots {
            soluwell_config::state::set(ui.ctx(), open_id(root.id), open);
        }
    }
    let mut picked = None;
    for action in soluwell_panels::outliner::show(ui, model, &built.config) {
        match action {
            // One shown selection: the row the click / key landed on (the matte and the
            // marquee have no lead and change nothing).
            OutlinerAction::Select { lead: Some(id), .. } => picked = ids.get(&id).cloned(),
            OutlinerAction::Context { id, action } => {
                let Some(sel) = ids.get(&id) else { continue };
                match action.as_str() {
                    "show" => picked = Some(sel.clone()),
                    "copy" => {
                        if let Some(text) = built.copy_text(id) {
                            ui.ctx().copy_text(text);
                        }
                    }
                    other => log::warn!("channel tree: unknown context action {other:?}"),
                }
            }
            _ => {}
        }
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::tests_support::frame_with;

    /// Every node maps back to the selection that shows it, and the shown
    /// selection is the one node selected.
    #[test]
    fn nodes_map_to_their_selection() {
        let frame = frame_with(&["A", "B", "G", "R", "depth.Z", "N.x", "N.y", "N.z"]);
        let view = Selection::channel("N", "N.y");
        let b = Built::new(&frame, &view);
        let (model, ids) = (&b.model, &b.ids);

        let labels: Vec<(&str, Vec<&str>)> = model
            .roots
            .iter()
            .map(|r| (r.label.as_str(), r.children.iter().map(|c| c.label.as_str()).collect()))
            .collect();
        assert_eq!(labels, [("<main>", vec!["R", "G", "B", "A"]), ("N", vec!["x", "y", "z"]), ("depth", vec!["Z"])]);

        assert_eq!(model.selection.len(), 1);
        assert_eq!(ids[&model.selection[0]], view);
        for root in &model.roots {
            assert_eq!(ids[&root.id].channel, None);
            for child in &root.children {
                let sel = &ids[&child.id];
                assert_eq!(sel.layer, ids[&root.id].layer);
                assert!(sel.channel.as_deref().is_some_and(|c| c.ends_with(child.label.as_str())));
            }
        }
    }

    /// The tree selects what the frame actually shows when the request falls back
    /// (`DecodedFrame::resolve`: absent channel -> its layer -> main), and opens that layer.
    #[test]
    fn selection_follows_the_fallback() {
        let frame = frame_with(&["R", "G", "B", "depth.Z"]);
        let selected = |view: &Selection| {
            let b = Built::new(&frame, view);
            assert_eq!(b.model.selection.len(), 1);
            let open: Vec<bool> = b.model.roots.iter().map(|r| r.default_open == Some(true)).collect();
            (b.ids[&b.model.selection[0]].clone(), open)
        };
        let main = Selection::layer("");
        let depth = Selection::channel("depth", "depth.Z");
        // Present: itself.
        assert_eq!(selected(&depth), (depth.clone(), vec![false, true]));
        // Absent channel in a present layer: that layer.
        let got = selected(&Selection::channel("depth", "depth.nope"));
        assert_eq!(got, (Selection::layer("depth"), vec![false, true]));
        // Absent channel in an absent layer, and an absent layer: the main layer.
        let got = selected(&Selection::channel("gone", "gone.R"));
        assert_eq!(got, (main.clone(), vec![true, false]));
        assert_eq!(selected(&Selection::layer("gone")), (main, vec![true, false]));
    }

    /// The reveal is keyed on the REQUESTED layer: a frame lacking it (shown as main)
    /// neither opens main nor undoes the user's fold of it, and the layer is revealed
    /// once a frame has it.
    #[test]
    fn reveal_follows_the_request_not_the_fallback() {
        let ctx = egui::Context::default();
        let base = egui::Id::new("channels-test");
        let flag = |id: u64| soluwell_config::state::get::<bool>(&ctx, open_key(base, id));
        let (main_id, n_id) = (hash_id("", None), hash_id("N", None));
        let full = frame_with(&["R", "G", "B", "N.x"]);
        let lacking = frame_with(&["R", "G", "B"]);
        let show = |frame: &DecodedFrame, layer: &str| {
            let view = Selection::layer(layer);
            reveal(&ctx, base, &Built::new(frame, &view).model, &view);
        };

        show(&full, "");
        assert_eq!((flag(main_id), flag(n_id)), (Some(true), Some(false)));
        // The user folds main.
        soluwell_config::state::set(&ctx, open_key(base, main_id), false);
        // N is requested on a frame without it: main is shown, but stays folded.
        show(&lacking, "N");
        assert_eq!(flag(main_id), Some(false));
        // A frame with N: N opens, main is still the user's.
        show(&full, "N");
        assert_eq!((flag(main_id), flag(n_id)), (Some(false), Some(true)));
        // A manual fold of N survives later frames with the same request.
        soluwell_config::state::set(&ctx, open_key(base, n_id), false);
        show(&full, "N");
        assert_eq!(flag(n_id), Some(false));
    }

    /// The dimmed suffix is each channel's own pixel type; layers carry none.
    #[test]
    fn channels_carry_their_pixel_type() {
        let frame = frame_with(&["R", "G", "B"]);
        let b = Built::new(&frame, &Selection::layer(""));
        let root = &b.model.roots[0];
        assert_eq!(root.type_label, None);
        assert_eq!(root.icon.as_deref(), Some(ph::IMAGE));
        assert!(root.children.iter().all(|c| c.type_label.as_deref() == Some("Float")));
        assert!(root.children.iter().all(|c| c.icon.as_deref() == Some(icons::DOT)));
    }

    /// Node ids from borrowed parts are the derived-`Hash` ids of the `Selection`s.
    #[test]
    fn ids_match_the_derived_hash() {
        let derived = |s: &Selection| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };
        for sel in [Selection::layer(""), Selection::layer("N"), Selection::channel("N", "N.x")] {
            assert_eq!(node_id(&sel), derived(&sel));
        }
        assert_eq!(hash_id("depth", None), derived(&Selection::layer("depth")));
    }

    #[test]
    fn tokenizer() {
        let t = |s: &str| tokens(s).join(",");
        assert_eq!(t("VRayZDepth"), "v,ray,z,depth");
        assert_eq!(t("VRayMtlID"), "v,ray,mtl,id");
        assert_eq!(t("IndexOB"), "index,ob");
        assert_eq!(t("CryptoObject00"), "crypto,object,00");
        assert_eq!(t("worldPosition"), "world,position");
        assert_eq!(t("shadow_matte"), "shadow,matte");
        assert_eq!(t("normalizedDepth"), "normalized,depth");
        assert_eq!(t("N"), "n");
        // An acronym keeps its plural `s`; a word starting with `s` still splits off.
        assert_eq!(t("IDs"), "ids");
        assert_eq!(t("ObjectIDs"), "object,ids");
        assert_eq!(t("IDsFoo"), "ids,foo");
        assert_eq!(t("ZSort"), "z,sort");
        assert_eq!(t("depthOfField"), "depth,of,field");
        assert_eq!(t("motionvector"), "motionvector");
    }

    /// One case per role, plus the precedence the order encodes.
    #[test]
    fn layer_icon_by_role() {
        let rgb = ["R", "G", "B"];
        let rgba = ["R", "G", "B", "A"];
        let xyz = ["x", "y", "z"];
        assert_eq!(layer_icon("", &rgba), ph::IMAGE);
        assert_eq!(layer_icon("", &["Y"]), ph::IMAGE);
        assert_eq!(layer_icon("", &["Y", "A"]), ph::IMAGE);
        assert_eq!(layer_icon("", &["Y", "RY", "BY"]), ph::IMAGE);
        assert_eq!(layer_icon("", &["y", "ry", "by"]), ph::IMAGE);
        assert_eq!(layer_icon("v", &xyz), ph::STACK);
        assert_eq!(layer_icon("", &["A"]), ph::CIRCLE_HALF);
        assert_eq!(layer_icon("", &["a"]), ph::CIRCLE_HALF);
        assert_eq!(layer_icon("alpha", &["A"]), ph::CIRCLE_HALF);
        assert_eq!(layer_icon("depth", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("", &["z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("N", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("normals", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("motion", &["x", "y"]), ph::WIND);
        assert_eq!(layer_icon("Velocity", &xyz), ph::WIND);
        assert_eq!(layer_icon("CryptoObject00", &rgba), ph::FINGERPRINT);
        assert_eq!(layer_icon("cryptomatte", &rgba), ph::FINGERPRINT);
        assert_eq!(layer_icon("mask_hero", &["V"]), ph::MASK_HAPPY);
        assert_eq!(layer_icon("MultiMatte", &rgba), ph::MASK_HAPPY);
        assert_eq!(layer_icon("materialID", &["V"]), ph::HASH);
        assert_eq!(layer_icon("objectID", &["V"]), ph::HASH);
        // A mask stays a mask even with another role's components.
        assert_eq!(layer_icon("mask", &["Z"]), ph::MASK_HAPPY);

        // Renderer prefixes and other conventions.
        assert_eq!(layer_icon("VRayNormals", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("VRayZDepth", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("VRayVelocity", &xyz), ph::WIND);
        assert_eq!(layer_icon("VRayMtlID", &["V"]), ph::HASH);
        assert_eq!(layer_icon("VRayObjectID", &["V"]), ph::HASH);
        assert_eq!(layer_icon("PuzzleMatte", &rgb), ph::MASK_HAPPY);
        assert_eq!(layer_icon("shadow_matte", &["V"]), ph::MASK_HAPPY);
        assert_eq!(layer_icon("mv", &["x", "y"]), ph::WIND);
        assert_eq!(layer_icon("Vector", &["X", "Y", "Z", "W"]), ph::WIND);
        assert_eq!(layer_icon("Op_Id", &["V"]), ph::HASH);
        assert_eq!(layer_icon("Prim_Id", &["V"]), ph::HASH);
        assert_eq!(layer_icon("IndexOB", &["V"]), ph::HASH);
        assert_eq!(layer_icon("IndexMA", &["V"]), ph::HASH);
        assert_eq!(layer_icon("z_depth", &["Z"]), ph::MOUNTAINS);
        // A word inside another is not a hit.
        assert_eq!(layer_icon("normalizedDepth", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("normalized", &["V"]), ph::STACK);
        assert_eq!(layer_icon("overflow", &["x", "y"]), ph::STACK);
        assert_eq!(layer_icon("flow", &["x", "y"]), ph::WIND);
        assert_eq!(layer_icon("grid", &["V"]), ph::STACK);
        assert_eq!(layer_icon("velvet", &["V"]), ph::STACK);

        // More vendor names.
        assert_eq!(layer_icon("motionvector", &rgb), ph::WIND);
        assert_eq!(layer_icon("motionvector", &["x", "y"]), ph::WIND);
        assert_eq!(layer_icon("forward", &["u", "v"]), ph::WIND);
        assert_eq!(layer_icon("backward", &["u", "v"]), ph::WIND);
        assert_eq!(layer_icon("IDs", &["V"]), ph::HASH);
        assert_eq!(layer_icon("ObjectIDs", &["V"]), ph::HASH);
        assert_eq!(layer_icon("primid", &["V"]), ph::HASH);
        assert_eq!(layer_icon("idx", &["V"]), ph::HASH);
        assert_eq!(layer_icon("Ng", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("Ps", &xyz), ph::CROSSHAIR);
        assert_eq!(layer_icon("Pz", &["V"]), ph::MOUNTAINS);
        // A keyword tied to what follows (`depth of field`) names no role.
        assert_eq!(layer_icon("depthOfField", &["V"]), ph::STACK);
        assert_eq!(layer_icon("normalsFor", &xyz), ph::STACK);
        assert_eq!(layer_icon("depthPass", &["Z"]), ph::MOUNTAINS);
        // The main layer's leading dot is ignored for roles (labels keep it).
        assert_eq!(layer_icon("", &[".R", "G", "B"]), ph::IMAGE);
        assert_eq!(layer_icon("", &[".A"]), ph::CIRCLE_HALF);
        // Generic vectors stay generic, lone or mixed with luminance-like names.
        assert_eq!(layer_icon("v", &["x", "y", "z"]), ph::STACK);
        assert_eq!(layer_icon("v", &["y", "x"]), ph::STACK);
        assert_eq!(layer_icon("v", &["R", "G"]), ph::STACK);

        // Position, also when written as colour channels.
        for name in ["P", "Pref", "position", "worldPosition", "wp", "left.P"] {
            assert_eq!(layer_icon(name, &xyz), ph::CROSSHAIR, "{name}");
            assert_eq!(layer_icon(name, &rgb), ph::CROSSHAIR, "{name}");
        }

        // Multiview: only the last segment names the role, in any case.
        assert_eq!(layer_icon("left.N", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("LEFT.n", &xyz), ph::COMPASS);
        assert_eq!(layer_icon("left.Z", &xyz), ph::MOUNTAINS);
        assert_eq!(layer_icon("left", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("left.depth", &["Z"]), ph::MOUNTAINS);
        assert_eq!(layer_icon("right.cryptoObject", &rgba), ph::FINGERPRINT);
        // Colour is colour for every layer: views and beauty passes alike.
        assert_eq!(layer_icon("left", &rgb), ph::IMAGE);
        assert_eq!(layer_icon("left", &["r", "g", "b", "a"]), ph::IMAGE);
        assert_eq!(layer_icon("diffuse", &rgb), ph::IMAGE);
    }

    /// "Copy name" copies a channel's full name, a named layer's name, and for the
    /// unnamed main layer its channel names in tree order.
    #[test]
    fn copy_text_by_node() {
        let frame = frame_with(&["A", "B", "G", "R", "N.x", "N.y"]);
        let b = Built::new(&frame, &Selection::layer(""));
        let id = |sel: Selection| node_id(&sel);
        assert_eq!(b.copy_text(id(Selection::channel("N", "N.y"))).as_deref(), Some("N.y"));
        assert_eq!(b.copy_text(id(Selection::channel("", "R"))).as_deref(), Some("R"));
        assert_eq!(b.copy_text(id(Selection::layer("N"))).as_deref(), Some("N"));
        assert_eq!(b.copy_text(id(Selection::layer(""))).as_deref(), Some("R G B A"));
        assert_eq!(b.copy_text(id(Selection::layer("nope"))), None);
    }

    /// A main-layer channel with a leading dot keeps it: its label and what Copy name
    /// puts on the clipboard are the real channel name.
    #[test]
    fn leading_dot_channel_keeps_its_name() {
        let frame = frame_with(&[".R", "G", "B"]);
        let b = Built::new(&frame, &Selection::layer(""));
        let main = &b.model.roots[0];
        assert!(main.children.iter().any(|c| c.label == ".R"));
        assert!(main.children.iter().all(|c| c.label != "R"));
        let copy = |sel: Selection| b.copy_text(node_id(&sel));
        assert_eq!(copy(Selection::channel("", ".R")).as_deref(), Some(".R"));
        let all = copy(Selection::layer("")).unwrap_or_default();
        assert!(all.split(' ').any(|n| n == ".R"), "{all}");
        assert!(all.split(' ').all(|n| n != "R"), "{all}");
    }

    /// The tree is reused while the channel set and the view stay, and rebuilt when
    /// either changes (a frame with other channels, another shown layer or channel).
    #[test]
    fn model_cache_invalidates_on_channels_or_view() {
        let a = frame_with(&["R", "G", "B", "depth.Z"]);
        let main = Selection::layer("");
        let first = fresh(None, &a, &main);

        // Same channels (the next frame of a sequence) and view: the same tree.
        let next = frame_with(&["R", "G", "B", "depth.Z"]);
        assert!(Arc::ptr_eq(&first, &fresh(Some(first.clone()), &next, &main)));

        // Another view: rebuilt, and the new selection is what is selected.
        let depth = Selection::layer("depth");
        let moved = fresh(Some(first.clone()), &a, &depth);
        assert!(!Arc::ptr_eq(&first, &moved));
        assert_eq!(moved.ids[&moved.model.selection[0]], depth);
        let solo = fresh(Some(moved.clone()), &a, &Selection::channel("", "G"));
        assert!(!Arc::ptr_eq(&moved, &solo));

        // Another channel set: rebuilt.
        let other = frame_with(&["R", "G", "B", "A", "depth.Z"]);
        let rebuilt = fresh(Some(first.clone()), &other, &main);
        assert!(!Arc::ptr_eq(&first, &rebuilt));
        assert_eq!(rebuilt.model.roots[0].children.len(), 4);
    }

    /// Identical names and types but another grouping must rebuild: the key holds the
    /// grouping itself, not the assumption that it follows from names and types.
    #[test]
    fn model_cache_invalidates_on_grouping() {
        let a = frame_with(&["R", "G", "B", "depth.Z"]);
        let main = Selection::layer("");
        let first = fresh(None, &a, &main);

        let mut regrouped = frame_with(&["R", "G", "B", "depth.Z"]);
        regrouped.layers.reverse();
        assert_eq!(regrouped.channel_names, a.channel_names);
        assert_eq!(regrouped.channel_types(), a.channel_types());
        let rebuilt = fresh(Some(first.clone()), &regrouped, &main);
        assert!(!Arc::ptr_eq(&first, &rebuilt));
        assert_eq!(rebuilt.model.roots[0].label, "depth");

        let mut moved = frame_with(&["R", "G", "B", "depth.Z"]);
        moved.layers[0].channels.swap(0, 1);
        assert!(!Arc::ptr_eq(&first, &fresh(Some(first.clone()), &moved, &main)));
    }

    /// The in-place type comparison agrees with `channel_types`, for resident and
    /// file-backed type lists.
    #[test]
    fn same_types_matches_channel_types() {
        let mut f = frame_with(&["R", "G", "B"]);
        let types = f.channel_types();
        assert!(same_types(&f, &types));
        assert!(!same_types(&f, &types[..2]));
        assert!(!same_types(&f, &[PixelType::Half, PixelType::Half, PixelType::Half]));
        f.file_types = Some(vec![PixelType::Half; 3]);
        assert!(same_types(&f, &[PixelType::Half; 3]));
        assert!(!same_types(&f, &types));
    }
}
