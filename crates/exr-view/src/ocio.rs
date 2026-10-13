//! The OCIO display transform of the colour image: the config, the user's choice
//! (display, the view per encoding class, look, a per-file input override), the frame's
//! colour space, the transform the tile shader and the CPU twin both run, and the Colour panel.
//!
//! Everything comes from `vfx-ocio` (the OpenColorIO 2.5 port Playa and vfx-view use):
//! ONE `Processor` per frame gives the WGSL spliced into the tile shader (`wgsl_prelude`,
//! bound through `GpuOcioResources::build_for(.., FRAGMENT)` in `tiles::render`) and the
//! exact CPU twin ([`Transform::display`]) that Save view, the headless screenshot and the
//! GPU test use.
//!
//! # One conversion per frame (`COLOR_PLAN.md`)
//!
//! A frame keeps its pixels as read and a colour tag (`exr_import::ColorTag`). [`Ocio::space`]
//! is THE mapping of a tag to a colour space of the config: the tag by name, alias or role
//! (`Config::colorspace`, case-insensitive; interop IDs are aliases in the ACES configs), the
//! config's file rules for an untagged frame (unless the path only matches their default rule,
//! then the tag's explicit fallback), or the user's per-file override. A name the config does
//! not know is a typed [`SpaceError`], shown with the override offered, never a substitute.
//!
//! The processor input is that space; the views offered are the ones the config's viewing
//! rules allow for it (`Config::get_views_for_colorspace`, OCIO `getNumViews(display,
//! colorspace)`), the default the first of them (OCIO `getDefaultView(display, colorspace)`):
//! a PQ HDR10 still gets `Video (colorimetric)` (PQ -> XYZ -> PQ, an identity on a PQ display)
//! instead of a second ACES rendering. The user's view is remembered per encoding class
//! ([`Class`]), so EXRs and a PQ PNG each keep theirs.
//!
//! The chain in colour mode (`channel_mode == RGBA`) is OCIO's `LegacyViewingPipeline`,
//! as `ociodisplay` builds it: the input to `scene_linear` and the exposure there (the
//! pipeline's `linearCC`, a dynamic `ExposureContrastTransform`: an exposure for log and
//! display-referred inputs too, moved per frame as a uniform / `set_dynamic`, no
//! rebuild), the display/view transform (with its look), then the gamma dial on the
//! display-encoded result (where `ociodisplay` applies its display gamma: an
//! `ExponentTransform` after the display/view transform). No sRGB OETF: the display
//! colour space already encodes. Isolated channels are data and keep the built-in path.
//! Associated colour is unpremultiplied before the processor and premultiplied after it
//! (`transfer::unpremul` / `transfer::premul`).
//!
//! What the transform ends with follows the window's output (`present::Output`):
//!
//! - **SDR output: encoded.** The display's own code values go to the canvas as
//!   they are (the user picks the display that matches the monitor, as in any OCIO
//!   viewer). Only SDR displays are offered: a view is hidden when its display
//!   colour space says `encoding: hdr-video` (one without an encoding, OCIO v1, is
//!   offered), and a display needs a non-data SDR view (in OCIO v2 a shared view
//!   renders into the DISPLAY's space, so a PQ display's "SDR" views are PQ too).
//! - **HDR output: linear.** Every display is offered. One processor runs the
//!   display/view transform, then the display colour space back to OCIO's display
//!   reference (linear CIE XYZ D65, 1.0 = 100 nits, the config's own
//!   `to_display_reference` or inverted `from_display_reference`), then XYZ to
//!   linear Rec.709: linear light with 1.0 = SDR reference white, which the tile
//!   shader encodes like the built-in chain and the present pass maps to the
//!   surface. A data view (`Raw`) passes its values through as that light.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, anyhow, bail};
use exr_import::{Builtin, ColorTag, TagOrigin};
use serde::{Deserialize, Serialize};
use vfx_ocio::builtin::embedded;
use vfx_ocio::color_matrix::{Adaptation, REC709, conversion_matrix_from_xyz_d65};
use vfx_ocio::{
    DisplayViewTransform, DynamicPropertyType, EcDynamic, Encoding, ExposureContrastTransform, GpuLanguage,
    GpuProcessor, GpuShaderCode, GroupTransform, LegacyViewingPipeline, MatrixTransform, Processor,
    ReferenceSpaceType, TransformDirection,
};

use crate::image::DecodedFrame;
use crate::transfer::Src;

/// What the user picked; persisted (`persist::Config::ocio`). Empty names mean the
/// config's defaults ([`Ocio::resolve`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sel {
    /// OCIO is the display transform of the colour image (else the built-in one).
    pub on: bool,
    /// A config file or an `ocio://` URI; empty = `$OCIO`, else `ocio://default`.
    pub config: String,
    /// Display; empty = the first one offered.
    pub display: String,
    /// The view of scene-referred frames (scene-linear, log); empty = the first one the
    /// viewing rules allow for the frame's colour space.
    pub scene_view: String,
    /// The view of display-referred frames (video, display-linear); empty = the first allowed.
    pub video_view: String,
    /// Look of scene-referred frames, applied instead of the view's own looks; empty = the view's
    /// looks. A look is an LMT in a scene-referred process space: it belongs to the class, like
    /// the view.
    pub scene_look: String,
    /// Look of display-referred frames (default none: running a scene LMT on a finished picture
    /// would take it to the scene reference and back, compressing its colours).
    pub video_look: String,
}

impl Sel {
    /// The remembered look of `class` (empty = the view's looks).
    pub fn look(&self, class: Class) -> &str {
        match class {
            Class::Scene => &self.scene_look,
            Class::Video => &self.video_look,
        }
    }

    /// The remembered look of `class`, to change.
    pub fn look_mut(&mut self, class: Class) -> &mut String {
        match class {
            Class::Scene => &mut self.scene_look,
            Class::Video => &mut self.video_look,
        }
    }

    /// The remembered view of `class` (empty = the default).
    pub fn view(&self, class: Class) -> &str {
        match class {
            Class::Scene => &self.scene_view,
            Class::Video => &self.video_view,
        }
    }

    /// The remembered view of `class`, to change.
    pub fn view_mut(&mut self, class: Class) -> &mut String {
        match class {
            Class::Scene => &mut self.scene_view,
            Class::Video => &mut self.video_view,
        }
    }
}

/// The encoding class a view choice is remembered for: scene-referred light (scene-linear,
/// log) wants a rendering (an ACES view), a finished picture (SDR / HDR video, display-linear)
/// wants none (`Video (colorimetric)`), as the configs' viewing rules split them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Scene,
    Video,
}

impl Class {
    /// How the Colour panel names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Scene => "scene-referred",
            Self::Video => "display-referred",
        }
    }
}

/// The config `sel_config` stands for: itself, else `$OCIO` (the standard OCIO
/// variable), else OCIO's default built-in (the ACES 2.0 CG config).
pub fn source(sel_config: &str) -> String {
    if !sel_config.trim().is_empty() {
        return sel_config.trim().to_owned();
    }
    std::env::var("OCIO")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "ocio://default".to_owned())
}

/// The gain from a transform's display light to the canvas (1.0 = SDR white of `target.white`
/// nits). Light is relative to SDR white for SDR displays (its 1.0 IS SDR white, as the OS
/// shows SDR content on an HDR display): 1. It is ABSOLUTE for an HDR display colour space or
/// an absolute source of the built-in decode (OCIO's display reference, 1.0 = 100 nits; a
/// 1000-nit view peaks at 10): 100/white on an HDR output, so the present pass, which scales
/// canvas 1.0 to `white` nits, lands on the nits meant. An SDR output shows 100 nits at white.
///
/// `nits` is the nits of the light's 1.0 when it is absolute (an HDR display colour space: 100,
/// the display reference; a frame with its own white, `exr_import::Builtin::nits`), `None` when
/// relative.
pub fn display_gain(nits: Option<f32>, target: &crate::present::Target) -> f32 {
    match nits {
        Some(n) if target.hdr => n / target.white.max(1.0),
        _ => 1.0,
    }
}

/// The nits of a transform's light 1.0 for a frame tagged `tag`: OCIO's display reference (100)
/// on an absolute (HDR) display, or the frame's own white when it is display light with a stated
/// white (EXR `whiteLuminance`, whose 1.0 is the reference white).
fn nits_of(t: &Transform, tag: &ColorTag) -> Option<f32> {
    let white = match tag.origin {
        TagOrigin::Luminance { nits, .. } => nits,
        _ => 100.0,
    };
    t.absolute().then_some(white)
}

/// A frame's colour space in a config ([`Ocio::space`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Space {
    /// The config's own name of it.
    pub name: String,
    /// How it was found: the tag and its origin, a file rule, the untagged fallback, the
    /// override.
    pub how: String,
    /// Which remembered view it takes.
    pub class: Class,
    /// A matrix from the stored values into the space first (EXR `whiteLuminance`: absolute
    /// CIE XYZ D65, `TagOrigin::Luminance`); row-major.
    pub pre: Option<[[f32; 3]; 3]>,
}

/// Why a frame has no colour space in the config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpaceError {
    /// The config has no colour space, alias or role of that name.
    Unknown { space: String, how: String },
}

impl std::fmt::Display for SpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { space, how } => {
                write!(f, "the config has no colour space \"{space}\" ({how}): pick an input override in Settings > Colour")
            }
        }
    }
}

impl std::error::Error for SpaceError {}

/// Names a [`Sel`] resolves to in one config for one frame's colour space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    /// The frame's colour space (the processor input).
    pub input: String,
    pub display: String,
    pub view: String,
    pub look: String,
}

/// A loaded config.
pub struct Ocio {
    /// What it was loaded from ([`source`]).
    pub src: String,
    /// Which load this is (process-wide counter): part of every transform key, so a
    /// reload of the same source rebuilds the tile pipeline.
    serial: u64,
    cfg: vfx_ocio::Config,
}

/// The load counter behind [`Ocio::serial`].
static LOADS: AtomicU64 = AtomicU64::new(0);

impl Ocio {
    /// Load `src` (a file or an `ocio://` URI).
    pub fn load(src: &str) -> Result<Self> {
        let cfg = vfx_ocio::Config::from_file(src).with_context(|| format!("OCIO config {src}"))?;
        log::info!("OCIO config {src}: {} colour spaces", cfg.colorspaces().len());
        Ok(Self { src: src.to_owned(), serial: LOADS.fetch_add(1, Ordering::Relaxed), cfg })
    }

    /// Every colour space, config order (the override list).
    pub fn inputs(&self) -> Vec<&str> {
        self.cfg.colorspace_names().collect()
    }

    /// The colour space of a frame tagged `tag` at `path`, or `over` (the user's override)
    /// when given. See the module docs for the rules; this is their one implementation.
    pub fn space(&self, tag: &ColorTag, path: &Path, over: Option<&str>) -> Result<Space, SpaceError> {
        if let Some(name) = over {
            return self.find(name, "override".to_owned());
        }
        let how = format!("{}, {}", tag.space, tag.origin.label());
        match tag.origin {
            TagOrigin::Untagged => {
                // A rule other than the default one names the space (studio: EXR -> ACES2065-1);
                // the default rule alone says nothing the explicit fallback does not say better
                // (an older config's `Default: ACES2065-1` would take an 8-bit PNG for AP0).
                let file = path.to_string_lossy();
                match self.cfg.colorspace_from_filepath(&file) {
                    Some(cs) if !self.cfg.filepath_only_matches_default_rule(&file) => {
                        self.find(cs, "untagged, the config's file rules".to_owned())
                    }
                    _ => self.find(&tag.space, format!("{}, untagged fallback", tag.space)),
                }
            }
            TagOrigin::Luminance { to_xyz, .. } => self.find(&tag.space, how).map(|s| Space { pre: Some(to_xyz), ..s }),
            TagOrigin::File | TagOrigin::Chromaticities => self.find(&tag.space, how),
        }
    }

    /// The config's colour space `name` (name, alias or role), classed for the view memory.
    fn find(&self, name: &str, how: String) -> Result<Space, SpaceError> {
        let cs = self.cfg.colorspace(name).ok_or_else(|| SpaceError::Unknown { space: name.to_owned(), how: how.clone() })?;
        let class = match cs.encoding() {
            Encoding::SceneLinear | Encoding::Log => Class::Scene,
            Encoding::Sdr | Encoding::Hdr | Encoding::DisplayLinear => Class::Video,
            // No encoding (OCIO v1, data, `edr-video`): the reference space says which.
            _ if cs.reference_space_type() == ReferenceSpaceType::Display => Class::Video,
            _ => Class::Scene,
        };
        Ok(Space { name: cs.name().to_owned(), how, class, pre: None })
    }

    /// The active displays offered: with `hdr` every one with a picture (non-data)
    /// view; without, those with a non-data SDR view. A data view (`Raw`) does not
    /// count: every display has one.
    pub fn displays(&self, hdr: bool) -> Vec<&str> {
        self.cfg
            .display_names()
            .into_iter()
            .filter(|d| {
                self.cfg.get_views(d).into_iter().any(|v| {
                    let enc = self.encoding(d, v);
                    enc != Some(Encoding::Data) && (hdr || enc != Some(Encoding::Hdr))
                })
            })
            .collect()
    }

    /// The views of `display` offered for colour space `space` (config order): those the
    /// viewing rules allow for it, all of them with `hdr`, else the non-HDR ones (data views
    /// included).
    pub fn views(&self, display: &str, space: &str, hdr: bool) -> Vec<&str> {
        self.cfg
            .get_views_for_colorspace(display, space)
            .into_iter()
            .filter(|v| hdr || self.encoding(display, v) != Some(Encoding::Hdr))
            .map(|v| v.name())
            .collect()
    }

    /// The encoding of the colour space `view` of `display` renders into; `None`
    /// when it names none (OCIO v1 configs), which is offered as SDR.
    fn encoding(&self, display: &str, view: &vfx_ocio::View) -> Option<Encoding> {
        let enc = self.cfg.colorspace(view.effective_colorspace(display))?.encoding();
        (enc != Encoding::Unknown).then_some(enc)
    }

    /// Every look.
    pub fn looks(&self) -> Vec<&str> {
        self.cfg.looks().names().collect()
    }

    /// The display `sel` names in this config (empty = the first offered). A name the config
    /// lacks is an error, not a silent substitute: the panel says which.
    fn display<'a>(&'a self, sel: &'a Sel, hdr: bool) -> Result<&'a str> {
        // What a name is missing as: an HDR display on an SDR output is "no SDR display".
        let kind = if hdr { "" } else { "SDR " };
        let displays = self.displays(hdr);
        if sel.display.is_empty() {
            displays.first().copied().ok_or_else(|| anyhow!("the config has no {kind}display"))
        } else if displays.contains(&sel.display.as_str()) {
            Ok(sel.display.as_str())
        } else {
            bail!("the config has no {kind}display \"{}\"", sel.display)
        }
    }

    /// `sel`'s names in this config for a frame in `space`, empty ones replaced by the defaults:
    /// the view is the one remembered for the space's class, else the first the viewing rules
    /// allow. A name the config lacks (or a view not allowed for the space) is an error.
    pub fn resolve(&self, sel: &Sel, space: &Space, hdr: bool) -> Result<Names> {
        let kind = if hdr { "" } else { "SDR " };
        let display = self.display(sel, hdr)?;
        let views = self.views(display, &space.name, hdr);
        let want = sel.view(space.class);
        let view = if want.is_empty() {
            views.first().copied().ok_or_else(|| anyhow!("\"{display}\" has no {kind}view for \"{}\"", space.name))?
        } else if views.contains(&want) {
            want
        } else {
            bail!("\"{display}\" has no {kind}view \"{want}\" for \"{}\" (its viewing rules)", space.name);
        };
        let look = sel.look(space.class);
        if !look.is_empty() && self.cfg.looks().get(look).is_none() {
            bail!("the config has no look \"{look}\"");
        }
        Ok(Names { input: space.name.clone(), display: display.to_owned(), view: view.to_owned(), look: look.to_owned() })
    }

    /// The viewer transform of `names` for stored values `pre` takes into `names.input` (none:
    /// they are in it), built as OCIO viewers build it (`LegacyViewingPipeline`, as
    /// `ociodisplay` does): the exposure is the pipeline's `linearCC`, a DYNAMIC exposure applied
    /// in `scene_linear` (so it is an exposure for a log or display-referred input too, not a
    /// gain on its codes), moved per frame without a rebuild ([`Transform::display`], the tile
    /// shader's uniform); then the display/view transform (a look replaces the view's own looks,
    /// OCIO's looks override); `linear` continues it to linear Rec.709 light as the pipeline's
    /// `displayCC` (see the module doc). Needs the config's `scene_linear` role, as OCIO's
    /// pipeline does; a data input bypasses everything, the exposure included, as in OCIO.
    pub fn transform(&self, names: &Names, pre: Option<[[f32; 3]; 3]>, linear: bool) -> Result<Transform> {
        let dvt = DisplayViewTransform {
            src: names.input.clone(),
            display: names.display.clone(),
            view: names.view.clone(),
            ..DisplayViewTransform::default()
        };
        let mut pipeline = LegacyViewingPipeline::new();
        pipeline.set_display_view_transform(Some(dvt));
        pipeline.set_looks_override_enabled(!names.look.is_empty());
        pipeline.set_looks_override(&names.look);
        pipeline.set_linear_cc(Some(vfx_ocio::Transform::ExposureContrast(ExposureContrastTransform {
            dynamic: EcDynamic::EXPOSURE,
            ..ExposureContrastTransform::default()
        })));
        if linear {
            let chain = self.to_display_light(&names.display, &names.view)?;
            if !chain.is_empty() {
                pipeline.set_display_cc(Some(vfx_ocio::Transform::Group(GroupTransform {
                    name: String::new(),
                    transforms: chain,
                    direction: TransformDirection::Forward,
                })));
            }
        }
        let mut proc = pipeline.processor(&self.cfg)?;
        if let Some(m) = pre {
            let r = |i: usize| m[i].map(f64::from);
            let [r0, r1, r2] = [r(0), r(1), r(2)];
            let matrix = [r0[0], r0[1], r0[2], 0.0, r1[0], r1[1], r1[2], 0.0, r2[0], r2[1], r2[2], 0.0, 0.0, 0.0, 0.0, 1.0];
            let into = vfx_ocio::Transform::Matrix(MatrixTransform {
                name: String::new(),
                matrix,
                offset: [0.0; 4],
                direction: TransformDirection::Forward,
            });
            proc = Processor::combine(&Processor::from_transform(&into, TransformDirection::Forward)?, &proc)?;
        }
        let key = self.key(names, pre, linear);
        // Absolute light: the view renders into an HDR display colour space.
        let absolute = linear
            && self
                .cfg
                .get_views(&names.display)
                .into_iter()
                .find(|v| v.name() == names.view)
                .is_some_and(|v| self.encoding(&names.display, v) == Some(Encoding::Hdr));
        Transform::new(proc, key, linear, absolute)
    }

    /// What [`Self::transform`] of these arguments is cached by ([`Transform::key`]): the config
    /// load and everything the processor is built from, so a cache can be asked before building.
    pub fn key(&self, names: &Names, pre: Option<[[f32; 3]; 3]>, linear: bool) -> String {
        format!(
            "{}#{}|{}|{}|{}|{}|{:?}|{}",
            self.src,
            self.serial,
            names.input,
            names.display,
            names.view,
            names.look,
            pre,
            if linear { "linear" } else { "encoded" }
        )
    }

    /// The transforms that take `display`/`view`'s output to linear Rec.709 light:
    /// its colour space to the display reference (CIE XYZ D65, 1.0 = 100 nits), then
    /// XYZ to Rec.709. None for a data view (its values are the light). A
    /// scene-referred view colour space (OCIO v1 configs) has no display reference:
    /// an error, not a guess at its curve.
    fn to_display_light(&self, display: &str, view: &str) -> Result<Vec<vfx_ocio::Transform>> {
        let v = self
            .cfg
            .get_views(display)
            .into_iter()
            .find(|v| v.name() == view)
            .ok_or_else(|| anyhow!("\"{display}\" has no view \"{view}\""))?;
        let name = v.effective_colorspace(display);
        let cs = self.cfg.colorspace(name).ok_or_else(|| anyhow!("the config has no colour space \"{name}\""))?;
        if cs.is_data() {
            return Ok(Vec::new());
        }
        if cs.reference_space_type() != ReferenceSpaceType::Display {
            bail!("HDR output needs a display-referred view colour space; \"{name}\" is scene-referred (an OCIO v1 config?)");
        }
        let mut out: Vec<vfx_ocio::Transform> = cs
            .to_display_reference()
            .cloned()
            .or_else(|| cs.from_display_reference().map(|t| t.clone().inverse()))
            .into_iter()
            .collect();
        // OCIO's own matrix, that of its builtin `DISPLAY - CIE-XYZ-D65_to_sRGB`.
        out.push(vfx_ocio::Transform::Matrix(MatrixTransform {
            name: String::new(),
            matrix: conversion_matrix_from_xyz_d65(&REC709, Adaptation::None)?,
            offset: [0.0; 4],
            direction: TransformDirection::Forward,
        }));
        Ok(out)
    }

    /// A CPU processor from the frame's space to the config's `scene_linear` role: what the
    /// pixel inspector shows next to the stored values.
    pub fn to_scene_linear(&self, space: &Space) -> Result<Processor> {
        let proc = self.cfg.processor(&space.name, "scene_linear")?;
        match space.pre {
            None => Ok(proc),
            Some(m) => {
                let r = |i: usize| m[i].map(f64::from);
                let [r0, r1, r2] = [r(0), r(1), r(2)];
                let matrix = [r0[0], r0[1], r0[2], 0.0, r1[0], r1[1], r1[2], 0.0, r2[0], r2[1], r2[2], 0.0, 0.0, 0.0, 0.0, 1.0];
                let into = vfx_ocio::Transform::Matrix(MatrixTransform { name: String::new(), matrix, offset: [0.0; 4], direction: TransformDirection::Forward });
                Ok(Processor::combine(&Processor::from_transform(&into, TransformDirection::Forward)?, &proc)?)
            }
        }
    }
}

/// One built display transform: the processor (CPU twin) and its WGSL.
pub struct Transform {
    /// What built it; the tile pipeline rebuilds when it changes.
    pub key: String,
    /// The CPU twin's processor with the exposure (stops) its dynamic exposure is set to:
    /// moved by `set_dynamic` only when a call asks for another exposure, so a row-by-row
    /// export at one exposure takes the read lock only.
    cpu: std::sync::RwLock<(f32, Processor)>,
    /// `fn ocio_transform(vec4<f32>) -> vec4<f32>` with its bindings (groups 1 and 2).
    pub prelude: String,
    /// The uniforms and LUTs `GpuOcioResources` binds.
    pub code: GpuShaderCode,
    /// The uniforms holding the dynamic exposure in stops (the pipeline's `linearCC`), which
    /// the tile pipeline moves per frame; empty when the transform has none (a data input,
    /// whose pipeline bypasses the exposure as OCIO does).
    pub exposure: Vec<String>,
    /// It ends in linear Rec.709 light (HDR output), not in display code values.
    linear: bool,
    /// Its light is ABSOLUTE (an HDR display colour space, e.g. PQ: display
    /// reference 1.0 = 100 nits), not relative to SDR white: see [`display_gain`].
    absolute: bool,
}

impl Transform {
    /// Compile `proc` for the GPU; `linear` says it ends in linear light,
    /// `absolute` that the light is in absolute nits (see [`display_gain`]).
    pub fn new(proc: Processor, key: String, linear: bool, absolute: bool) -> Result<Self> {
        let gpu = GpuProcessor::from_processor_wgsl(&proc)?;
        let code = gpu.generate_shader(GpuLanguage::Wgsl)?;
        let exposure = gpu.dynamic_uniforms(DynamicPropertyType::Exposure);
        let ev = proc.dynamic(DynamicPropertyType::Exposure).unwrap_or(0.0) as f32;
        Ok(Self {
            key,
            prelude: gpu.wgsl_prelude(),
            code,
            exposure,
            cpu: std::sync::RwLock::new((ev, proc)),
            linear,
            absolute,
        })
    }

    /// The CPU processor with its dynamic exposure at `exposure_ev` stops (moved only when it
    /// differs). A poisoned lock is taken over: the guarded value is a processor and the
    /// exposure it was last set to, consistent whatever a panicking reader did.
    fn cpu_at(&self, exposure_ev: f32) -> std::sync::RwLockReadGuard<'_, (f32, Processor)> {
        use std::sync::PoisonError;
        {
            let cpu = self.cpu.read().unwrap_or_else(PoisonError::into_inner);
            if self.exposure.is_empty() || cpu.0.to_bits() == exposure_ev.to_bits() {
                return cpu;
            }
        }
        {
            let mut cpu = self.cpu.write().unwrap_or_else(PoisonError::into_inner);
            if cpu.0.to_bits() != exposure_ev.to_bits() {
                // Cannot fail: `self.exposure` is non-empty, so the processor has a dynamic
                // exposure (the same op the GPU lists); log rather than discard if it ever does.
                match cpu.1.set_dynamic(DynamicPropertyType::Exposure, f64::from(exposure_ev)) {
                    Ok(()) => cpu.0 = exposure_ev,
                    Err(e) => log::error!("OCIO exposure {exposure_ev}: {e}"),
                }
            }
        }
        self.cpu.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Its light is absolute (see [`display_gain`]).
    pub fn absolute(&self) -> bool {
        self.absolute
    }

    /// It ends in linear Rec.709 light, which the tile shader encodes.
    pub fn linear(&self) -> bool {
        self.linear
    }

    /// Whether it samples LUT textures (`Rgba32Float` with a filtering sampler, which
    /// needs `wgpu::Features::FLOAT32_FILTERABLE`).
    pub fn has_luts(&self) -> bool {
        !self.code.textures().is_empty()
    }

    /// CPU twin of the tile shader's OCIO colour branch over stored RGBA pixels of a frame read
    /// as `src`: unpremultiply, the transform with its dynamic exposure at `exposure_ev` (in
    /// `scene_linear`), for a linear transform `src.gain` ([`display_gain`]), the gamma dial and,
    /// for a linear transform, the canvas encoding, then premultiply in light; alpha stays the
    /// input's.
    pub fn display(&self, px: &mut [[f32; 4]], src: &Src, exposure_ev: f32, gamma: f32) {
        let mut alpha = vec![0.0; px.len()];
        self.display_kernel(px, &mut alpha, src, exposure_ev, gamma);
    }

    /// The exporter supplies admitted reusable alpha storage alongside its row.
    pub(crate) fn display_scratch(
        &self,
        px: &mut [[f32; 4]],
        alpha: &mut [f32],
        src: &Src,
        exposure_ev: f32,
        gamma: f32,
    ) -> Result<(), &'static str> {
        if alpha.len() != px.len() {
            return Err("OCIO alpha scratch must match the pixel row");
        }
        self.display_kernel(px, alpha, src, exposure_ev, gamma);
        Ok(())
    }

    fn display_kernel(&self, px: &mut [[f32; 4]], alpha: &mut [f32], src: &Src, exposure_ev: f32, gamma: f32) {
        use crate::transfer::{dial, oetf, premul, unpremul};
        use rayon::prelude::*;
        let straight = src.straight;
        px.par_iter_mut().zip(alpha.par_iter_mut()).for_each(|(p, a)| {
            *a = p[3];
            [p[0], p[1], p[2]] = unpremul([p[0], p[1], p[2]], *a, straight);
            // Alpha is restored verbatim below. A NaN alpha must not enter RGB
            // matrix operations, where even a zero alpha coefficient propagates it.
            p[3] = 1.0;
        });
        self.cpu_at(exposure_ev).1.apply_rgba(px);
        let (linear, gain) = (self.linear, src.gain);
        px.par_iter_mut().zip(alpha.par_iter()).for_each(|(p, &a)| {
            let d = [p[0], p[1], p[2]].map(|c| if linear { oetf(dial(c * gain, gamma)) } else { dial(c, gamma) });
            [p[0], p[1], p[2]] = premul(d, a, straight);
            p[3] = a;
        });
    }
}

/// The frame on screen as [`State::frame`] resolved it, for the revision of the choice it was
/// resolved with.
pub struct Shown {
    /// `DecodedFrame::id` of the frame.
    id: u64,
    /// [`State`]'s revision it was resolved at.
    rev: u64,
    /// The frame's tag and alpha kind (what the info panel shows).
    pub tag: ColorTag,
    pub straight: bool,
    /// Its colour space in the config, when OCIO is on and the config loaded.
    pub space: Option<Result<Space, SpaceError>>,
    /// The names its transform was built with.
    pub names: Option<Names>,
    /// The views the config allows for its colour space on the chosen display.
    pub views: Vec<String>,
    /// Its transform (OCIO on, everything resolved and built).
    transform: Option<Arc<Transform>>,
    /// Its built-in decode (`exr_import::builtin`), or why there is none.
    pub builtin: Result<Builtin, String>,
    /// Its space to `scene_linear` for the pixel inspector, built on first use (hovering one
    /// frame asks every repaint; stepping a sequence never builds it).
    to_lin: std::sync::OnceLock<Option<Processor>>,
}

/// The Colour panel's state: the choice, the loaded config and the transform shown.
pub struct State {
    /// The choice (persisted by the app).
    pub sel: Sel,
    /// The device filters `Rgba32Float` (LUT transforms need it).
    filterable: bool,
    /// The window output is HDR: every display is offered and the transform is linear.
    hdr: bool,
    /// The config of `source(sel.config)`, or why it does not load.
    cfg: Option<(String, Result<Ocio, String>)>,
    /// The input override of the open file or sequence: a colour space of the config used in
    /// place of the frame's tag. Per file (the app clears it on every new open), not persisted.
    pub over: Option<String>,
    /// Bumped by every change of the choice, config, output or override, so the shown frame
    /// resolves again.
    rev: u64,
    /// Transforms built from this config, by key: frames in one colour space share one, and a
    /// sequence of mixed spaces does not rebuild per frame.
    built: HashMap<String, Arc<Transform>>,
    /// The frame on screen, resolved.
    shown: Option<Shown>,
    /// Why OCIO is on but not shown (the config, a name, the frame's colour space).
    pub err: Option<String>,
    /// The path field's widget state. Only the field's own bookkeeping: where the config dialog
    /// starts is the dialog memory's (`file_pick::DialogMemory`, key `"ocio"`).
    field: egui_file_field::FileFieldState,
}

impl State {
    /// A state for `sel`, its config loaded (the device is assumed to filter until
    /// [`Self::set_filterable`] says otherwise).
    pub fn new(sel: Sel) -> Self {
        let field = egui_file_field::FileFieldState::default();
        let mut s = Self {
            sel,
            filterable: true,
            hdr: false,
            cfg: None,
            over: None,
            rev: 0,
            built: HashMap::new(),
            shown: None,
            err: None,
            field,
        };
        s.rebuild();
        s
    }

    /// Tell what the device can do (after the GPU is up) and rebuild.
    pub fn set_filterable(&mut self, filterable: bool) {
        if self.filterable != filterable {
            self.filterable = filterable;
            self.rebuild();
        }
    }

    /// Tell whether the window output is HDR (`present::Output::is_hdr`) and rebuild.
    pub fn set_hdr(&mut self, hdr: bool) {
        if self.hdr != hdr {
            self.hdr = hdr;
            self.rebuild();
        }
    }

    /// Use `over` in place of the open file's colour tag (`None` = the tag), and rebuild.
    pub fn set_override(&mut self, over: Option<String>) {
        if self.over != over {
            self.over = over;
            self.rebuild();
        }
    }

    /// The transform the shown frame's colour image goes through, `None` = the built-in one.
    pub fn active(&self) -> Option<&Arc<Transform>> {
        self.shown.as_ref().and_then(|s| s.transform.as_ref())
    }

    /// The shown frame as resolved ([`Self::frame`]).
    pub fn shown(&self) -> Option<&Shown> {
        self.shown.as_ref()
    }

    /// Load the config when its source changed and mark the shown frame for resolving again
    /// (its transform is built by [`Self::frame`]); a config that does not load is kept in
    /// `err` and the built-in transform shows.
    pub fn rebuild(&mut self) {
        self.rev += 1;
        self.err = None;
        if !self.sel.on {
            return;
        }
        self.load_config();
        if let Err(e) = self.config() {
            log::warn!("OCIO: {e}");
            self.err = Some(e);
        }
    }

    /// Resolve `frame` (cached until it or the choice changes) and return how the tile shader
    /// reads it on `target`: its built-in decode, alpha kind and display gain. With OCIO on,
    /// [`Self::active`] is then its transform, or `err` says why there is none. `Err` only when
    /// no transform can show the colour image: OCIO is not shown and the tag has no built-in
    /// decode.
    pub fn frame(&mut self, frame: &DecodedFrame, target: &crate::present::Target) -> Result<Src, String> {
        if self.shown.as_ref().is_none_or(|s| s.id != frame.id || s.rev != self.rev) {
            self.resolve(frame);
        }
        let Some(s) = &self.shown else { return Err("the frame did not resolve".to_owned()) };
        match &s.transform {
            // Under OCIO the built-in decode only tells an isolated channel's code from its
            // light; a tag without one shows its channels as light (the data rule).
            Some(t) => Ok(Src {
                builtin: s.builtin.clone().unwrap_or(Builtin::LINEAR),
                straight: s.straight,
                gain: display_gain(nits_of(t, &s.tag), target),
            }),
            None => {
                let builtin = s.builtin.clone()?;
                Ok(Src { builtin, straight: s.straight, gain: display_gain(builtin.nits, target) })
            }
        }
    }

    /// Resolve `frame` into [`Self::shown`]: its colour space, names and transform (OCIO on),
    /// and its built-in decode; a failure of the OCIO part is kept in `err`.
    fn resolve(&mut self, frame: &DecodedFrame) {
        let mut shown = Shown {
            id: frame.id,
            rev: self.rev,
            tag: frame.tag.clone(),
            straight: frame.straight,
            space: None,
            names: None,
            views: Vec::new(),
            transform: None,
            builtin: exr_import::builtin(&frame.tag).map_err(|e| e.to_string()),
            to_lin: std::sync::OnceLock::new(),
        };
        self.err = None;
        if self.sel.on
            && let Err(e) = self.build(frame, &mut shown)
        {
            log::warn!("OCIO {}: {e:#}", frame.path.display());
            self.err = Some(format!("{e:#}"));
        }
        self.shown = Some(shown);
    }

    /// The OCIO part of [`Self::resolve`].
    fn build(&mut self, frame: &DecodedFrame, shown: &mut Shown) -> Result<()> {
        self.load_config();
        let (hdr, filterable) = (self.hdr, self.filterable);
        let ocio = self.config().map_err(|e| anyhow!(e))?;
        let space = ocio.space(&frame.tag, &frame.path, self.over.as_deref());
        shown.space = Some(space.clone());
        let space = space?;
        if let Ok(display) = ocio.display(&self.sel, hdr) {
            shown.views = ocio.views(display, &space.name, hdr).into_iter().map(str::to_owned).collect();
        }
        let names = ocio.resolve(&self.sel, &space, hdr)?;
        shown.names = Some(names.clone());
        // The key first: a sequence of frames in one space builds its processor once.
        let key = ocio.key(&names, space.pre, hdr);
        let t = match self.built.get(&key) {
            Some(had) => had.clone(),
            None => {
                let t = Arc::new(ocio.transform(&names, space.pre, hdr)?);
                log::info!("OCIO display transform: {}", t.key);
                self.built.insert(key, t.clone());
                t
            }
        };
        if t.has_luts() && !filterable {
            bail!("this transform samples LUT textures, and this GPU cannot filter 32-bit float textures");
        }
        shown.transform = Some(t);
        Ok(())
    }

    /// Load the config of `sel.config` when its source changed (the transforms of the old one
    /// are dropped with it).
    fn load_config(&mut self) {
        let src = source(&self.sel.config);
        if self.cfg.as_ref().is_none_or(|(s, _)| *s != src) {
            let loaded = Ocio::load(&src).map_err(|e| format!("{e:#}"));
            self.cfg = Some((src, loaded));
            self.built.clear();
        }
    }

    /// The config [`Self::load_config`] loaded, or why it did not load.
    fn config(&self) -> Result<&Ocio, String> {
        match &self.cfg {
            Some((_, Ok(o))) => Ok(o),
            Some((_, Err(e))) => Err(e.clone()),
            None => Err("no config".to_owned()),
        }
    }

    /// One line for the status bar: the shown frame's colour space in the config and how it was
    /// found (OCIO on), else its tag and origin; `None` before a frame is shown.
    pub fn summary(&self) -> Option<String> {
        let s = self.shown.as_ref()?;
        Some(match &s.space {
            Some(Ok(sp)) => format!("{} ({})", sp.name, sp.how),
            _ => format!("{} ({})", s.tag.space, s.tag.origin.label()),
        })
    }

    /// The Metadata panel's colour rows for the shown frame: the tag and its origin, the alpha
    /// kind, the built-in decode, and with OCIO the colour space, the view and any error.
    pub fn rows(&self) -> Vec<(&'static str, String)> {
        let Some(s) = &self.shown else { return Vec::new() };
        let mut out = vec![
            ("Colour tag", s.tag.space.clone()),
            ("Tag origin", s.tag.origin.label().to_owned()),
            ("Alpha", if s.straight { "straight" } else { "associated (or none)" }.to_owned()),
            ("Built-in decode", match &s.builtin {
                Ok(b) => match b.nits {
                    Some(n) => format!("{:?}, absolute ({n} nits = 1)", b.tf),
                    None => format!("{:?}", b.tf),
                },
                Err(e) => e.clone(),
            }),
        ];
        match &s.space {
            Some(Ok(sp)) => {
                out.push(("OCIO colour space", sp.name.clone()));
                out.push(("Found by", sp.how.clone()));
            }
            Some(Err(e)) => out.push(("OCIO colour space", e.to_string())),
            None => {}
        }
        if let Some(n) = &s.names {
            out.push(("OCIO view", format!("{} / {}", n.display, n.view)));
        }
        if let Some(e) = &self.err {
            out.push(("OCIO error", e.clone()));
        }
        out
    }

    /// The scene-linear values of stored `rgb` of the shown frame (the pixel inspector), `None`
    /// when OCIO is off or the frame has no colour space in the config.
    pub fn scene_linear(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        let shown = self.shown.as_ref()?;
        let proc = shown.to_lin.get_or_init(|| {
            let space = shown.space.as_ref()?.as_ref().ok()?;
            self.config().ok()?.to_scene_linear(space).map_err(|e| log::warn!("OCIO scene_linear of {}: {e:#}", space.name)).ok()
        });
        let mut px = [[rgb[0], rgb[1], rgb[2], 1.0]];
        proc.as_ref()?.apply_rgba(&mut px);
        Some([px[0][0], px[0][1], px[0][2]])
    }

    /// Use another config: the names of the old one mean nothing there, so they go
    /// back to the new config's defaults (the override too).
    pub fn set_config(&mut self, config: String) {
        self.sel = Sel { on: self.sel.on, config, ..Sel::default() };
        self.over = None;
        self.rebuild();
    }

    /// The folder of the current config: where the config file dialog opens when nothing is
    /// remembered; `None` when the config is built in or its folder is gone.
    pub fn start_dir(&self) -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(self.field.start_dir(&self.sel.config));
        dir.is_dir().then_some(dir)
    }

    /// Switch OCIO on or off.
    pub fn set_on(&mut self, on: bool) {
        self.sel.on = on;
        self.rebuild();
    }

    /// Read the config again (it changed on disk, or a failed load was fixed) and
    /// rebuild; the new load serial makes the tile pipeline rebuild too.
    pub fn reload(&mut self) {
        self.cfg = None;
        self.rebuild();
    }

    /// The Colour panel. Returns whether the choice changed; `browse` is set when the
    /// user asks for a config file (the app runs the file dialog).
    pub fn ui(&mut self, ui: &mut egui::Ui, browse: &mut bool) -> bool {
        let before = (self.sel.clone(), self.over.clone());
        let mut reload = false;
        ui.checkbox(&mut self.sel.on, "OCIO display transform")
            .on_hover_text("Show the colour image through the OCIO display / view (isolated channels stay raw).\nOff: the built-in decode of the frame's colour tag, the tone operator and sRGB.");
        ui.add_space(4.0);

        let mut config = self.sel.config.clone();
        egui::Grid::new("ocio.grid").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("Config");
            ui.horizontal(|ui| {
                config_combo(ui, &mut config);
                reload = ui.button("Reload").on_hover_text("Read the config again (after it changed on disk).").clicked();
            });
            ui.end_row();
            // A path (or `ocio://` URI) typed here applies on Enter, not per keystroke: every
            // change loads the whole config. One that does not load shows the error below.
            ui.label("Config file").on_hover_text("An .ocio / .ocioz path or an ocio:// URI. Empty: $OCIO, else the built-in ACES 2.0 CG.");
            let typed = path_edit(ui, &mut self.field, &self.sel.config);
            *browse |= typed.browse;
            if let Some(path) = typed.apply {
                config = path;
            }
            ui.end_row();
            if config != self.sel.config {
                // The names of the old config mean nothing in the new one.
                self.sel = Sel { on: self.sel.on, config, ..Sel::default() };
                self.over = None;
            }
            if reload {
                self.cfg = None;
            }
            self.load_config();

            let Ok(ocio) = self.config() else {
                return;
            };
            let owned = |v: Vec<&str>| v.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let hdr = self.hdr;
            let displays = owned(ocio.displays(hdr));
            let (inputs, looks) = (owned(ocio.inputs()), owned(ocio.looks()));
            let shown = self.shown.as_ref();
            let space = shown.and_then(|s| s.space.as_ref()).and_then(|s| s.as_ref().ok());
            let names = shown.and_then(|s| s.names.as_ref());
            let views = shown.map(|s| s.views.clone()).unwrap_or_default();
            let presets: Vec<_> = PRESETS
                .iter()
                .filter(|p| hdr || !p.hdr)
                .map(|p| {
                    let ok = displays.iter().any(|d| d == p.display)
                        && space.is_none_or(|s| ocio.views(p.display, &s.name, hdr).contains(&p.view));
                    let on = names.is_some_and(|n| n.display == p.display && n.view == p.view);
                    (p, ok, on)
                })
                .collect();

            // Keep the quick choices above the advanced controls, including in a
            // short Settings dock. Both entry points use this same preset table.
            ui.label("Monitor preset").on_hover_text("Choose the image rendering of scene-referred frames; this does not change UI reference white.");
            ui.horizontal_wrapped(|ui| {
                for (p, ok, on) in presets {
                    let tip = if ok {
                        p.hint.to_owned()
                    } else {
                        format!("{}\n\nUnavailable: this config (or its viewing rules for the frame's colour space) has no \"{}\" / \"{}\".", p.hint, p.display, p.view)
                    };
                    let button = egui::Button::new(p.label).selected(self.sel.on && on);
                    if ui.add_enabled(ok, button).on_hover_text(tip).clicked() {
                        self.sel.on = true;
                        self.sel.display = p.display.to_owned();
                        self.sel.scene_view = p.view.to_owned();
                    }
                }
            });
            ui.end_row();
            ui.label("Input").on_hover_text(
                "The colour space of the frame's values: from the file's tag (or the config's file rules for an untagged file), or an override for this file.",
            );
            let tag = shown.map_or_else(
                || "the frame's tag".to_owned(),
                |s| match &s.space {
                    Some(Ok(sp)) if self.over.is_none() => format!("{} ({})", sp.name, sp.how),
                    _ => format!("{} ({})", s.tag.space, s.tag.origin.label()),
                },
            );
            let mut over = self.over.clone().unwrap_or_default();
            combo(ui, "ocio.input", &mut over, &inputs, &format!("From the file: {tag}"));
            self.over = (!over.is_empty()).then_some(over);
            ui.end_row();
            ui.label("Display").on_hover_text(
                "The display the image is rendered for. SDR output: the monitor's own colour space (HDR ones need an HDR output, Settings > Display).",
            );
            let d_before = self.sel.display.clone();
            combo(ui, "ocio.display", &mut self.sel.display, &displays, names.map_or("", |n| n.display.as_str()));
            if self.sel.display != d_before {
                // The view lists belong to the display.
                self.sel.scene_view.clear();
                self.sel.video_view.clear();
            }
            ui.end_row();
            let class = space.map_or(Class::Scene, |s| s.class);
            ui.label("View").on_hover_text(format!(
                "The rendering for the display: the views the config's viewing rules allow for the frame's colour space. Remembered for {} frames.",
                class.label()
            ));
            combo(ui, "ocio.view", self.sel.view_mut(class), &views, names.map_or("", |n| n.view.as_str()));
            ui.end_row();
            ui.label("Look").on_hover_text(format!("A look applied instead of the view's own looks. Remembered for {} frames.", class.label()));
            combo(ui, "ocio.look", self.sel.look_mut(class), &looks, "(the view's looks)");
            ui.end_row();
        });

        // One rebuild for whatever changed this frame.
        let changed = (self.sel.clone(), self.over.clone()) != before || reload;
        if changed {
            self.rebuild();
        }
        if let Some(e) = &self.err {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, format!("OCIO not shown: {e}"));
        } else if let Err(e) = self.config() {
            ui.add_space(4.0);
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        changed
    }
}

/// A display + view pairing worth one click (checked against the config before use).
struct Preset {
    label: &'static str,
    display: &'static str,
    view: &'static str,
    /// Offered only on an HDR output.
    hdr: bool,
    hint: &'static str,
}

/// The pairings of the ACES 2.0 configs (vfx-view's monitor presets; the HDR one
/// only on an HDR output).
const PRESETS: &[Preset] = &[
    Preset {
        label: "SDR · sRGB",
        display: "sRGB - Display",
        view: "ACES 2.0 - SDR 100 nits (Rec.709)",
        hdr: false,
        hint: "A monitor clamped to sRGB (an sRGB picture mode, or Windows colour management).",
    },
    Preset {
        label: "SDR · P3",
        display: "Display P3 - Display",
        view: "ACES 2.0 - SDR 100 nits (P3 D65)",
        hdr: false,
        hint: "A wide-gamut monitor in its native mode: the same rendering, encoded for P3 primaries.",
    },
    Preset {
        label: "sRGB, no rendering",
        display: "sRGB - Display",
        view: "Un-tone-mapped",
        hdr: false,
        hint: "Encode straight to sRGB without a view transform (0.18 stays 0.461).",
    },
    Preset {
        label: "HDR · 1000 nits",
        display: "Rec.2100-PQ - Display",
        view: "ACES 2.0 - HDR 1000 nits (P3 D65)",
        hdr: true,
        hint: "The ACES 2.0 1000-nit HDR rendering; the output (Settings > Display) carries it to the monitor.",
    },
];

/// The config combo: automatic ($OCIO or the default built-in), OCIO's built-in
/// configs, or the file in use.
fn config_combo(ui: &mut egui::Ui, config: &mut String) {
    let name = |c: &str| {
        if c.is_empty() {
            return format!("Auto ({})", source(""));
        }
        embedded::find_ocio_builtin(c).map_or_else(|| c.to_owned(), |e| e.ui_name.to_owned())
    };
    egui::ComboBox::from_id_salt("ocio.config")
        .selected_text(name(config))
        .width(260.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(config, String::new(), name(""))
                .on_hover_text("$OCIO when set, else OCIO's default built-in config (ACES 2.0 CG).");
            for e in embedded::ocio_builtins() {
                ui.selectable_value(config, e.uri(), e.ui_name);
            }
        })
        .response
        .on_hover_text(source(config));
}

/// What [`path_edit`] saw this frame.
struct PathEdit {
    /// Browse was clicked: the app runs the file dialog.
    browse: bool,
    /// Enter was pressed in the field: the path (trimmed; empty = auto) to use.
    apply: Option<String>,
    /// The text the field shows (the draft while typing, else the current config); tests only.
    #[cfg(test)]
    shown: String,
    /// The text field's id (to focus it); tests only.
    #[cfg(test)]
    id: egui::Id,
}

/// The config path field: a typed path applies on Enter, not per keystroke (every change loads
/// the whole config; one that does not load shows the error under the panel). The draft lives
/// in egui's temp memory under the field's own `Ui` id, so each instance keeps its own
/// text; it is dropped when the field loses the keyboard.
fn path_edit(ui: &mut egui::Ui, field: &mut egui_file_field::FileFieldState, current: &str) -> PathEdit {
    let key = ui.id().with("ocio_config_draft");
    let mut text = ui.data(|d| d.get_temp::<String>(key)).unwrap_or_else(|| current.to_owned());
    let shown = egui_file_field::FileField::new(&mut text, field)
        .hint("Auto: $OCIO, else the built-in ACES 2.0 CG")
        .tooltip("Load a config file (.ocio / .ocioz).")
        .desired_width(260.0)
        .show(ui);
    let enter = shown.response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    let out = PathEdit {
        browse: shown.browse_clicked(),
        apply: enter.then(|| text.trim().to_owned()),
        #[cfg(test)]
        shown: text.clone(),
        #[cfg(test)]
        id: shown.response.id,
    };
    // Read the focus BEFORE taking the data lock: egui's context lock is not re-entrant.
    let typing = shown.response.has_focus();
    ui.data_mut(|d| {
        if typing {
            d.insert_temp(key, text);
        } else {
            d.remove::<String>(key);
        }
    });
    out
}

/// A combo over `items` with an empty choice labelled `empty` (the default).
fn combo(ui: &mut egui::Ui, id: &str, cur: &mut String, items: &[String], empty: &str) {
    let text = if cur.is_empty() { format!("{empty} (default)") } else { cur.clone() };
    egui::ComboBox::from_id_salt(id).selected_text(text).width(260.0).show_ui(ui, |ui| {
        ui.selectable_value(cur, String::new(), format!("{empty} (default)"));
        for it in items {
            ui.selectable_value(cur, it.clone(), it);
        }
    });
}

#[cfg(test)]
impl Ocio {
    /// The config's `scene_linear` role as a frame's colour space (an untagged EXR without file
    /// rules): what the tests that are about the view, not the tag, run from.
    pub(crate) fn scene(&self) -> Space {
        self.find("scene_linear", "test".to_owned()).expect("the config has a scene_linear role")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::tests_support::frame_of;
    use crate::present::Target;
    use exr_core::api::ChannelData;

    fn cg() -> Ocio {
        Ocio::load("ocio://default").expect("the default built-in config")
    }

    fn studio() -> Ocio {
        Ocio::load("ocio://studio-config-latest").expect("the Studio built-in config")
    }

    fn tag(space: &str) -> ColorTag {
        ColorTag { space: space.to_owned(), origin: TagOrigin::File }
    }

    /// `sel`'s transform of `space` (encoded unless `linear`).
    fn tr(o: &Ocio, sel: &Sel, space: &Space, linear: bool) -> Transform {
        o.transform(&o.resolve(sel, space, linear).unwrap(), space.pre, linear).unwrap()
    }

    const SDR: Target = Target { hdr: false, white: 80.0, peak: 1000.0 };

    /// A 1x1 RGBA frame tagged `tag`, with `id` (the state's cache key) and `path`.
    fn frame(tag: ColorTag, id: u64, path: &str) -> DecodedFrame {
        let ch = |v: f32| ChannelData::Float(vec![v]);
        let mut f = frame_of(1, 1, vec![("A".into(), ch(1.0)), ("B".into(), ch(0.18)), ("G".into(), ch(0.18)), ("R".into(), ch(0.18))]);
        f.tag = tag;
        f.id = id;
        f.path = std::path::PathBuf::from(path);
        f
    }

    /// A minimal OCIO v2 config file in a fresh temp folder: (folder, file).
    fn good_config(tag: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::Builder::new().prefix(&format!("exr-view-ocio-{tag}-")).tempdir().unwrap();
        let file = dir.path().join("good.ocio");
        std::fs::write(
            &file,
            "ocio_profile_version: 2\n\nsearch_path: \"\"\nroles:\n  default: raw\n  scene_linear: raw\n\n\
             displays:\n  Mon:\n    - !<View> {name: Raw, colorspace: raw}\n\nactive_displays: [Mon]\nactive_views: [Raw]\n\n\
             colorspaces:\n  - !<ColorSpace>\n    name: raw\n    isdata: false\n",
        )
        .unwrap();
        (dir, file)
    }

    /// The path field keeps one draft PER INSTANCE (keyed by its `Ui` id): typing in one neither
    /// leaks into another nor is lost, and Enter applies the trimmed text once.
    #[test]
    fn each_path_field_keeps_its_own_draft() {
        let ctx = egui::Context::default();
        let mut states = [egui_file_field::FileFieldState::default(), egui_file_field::FileFieldState::default()];
        let mut frame = |events: Vec<egui::Event>, focus: Option<usize>, ids: &mut [egui::Id; 2]| {
            let mut out = Vec::new();
            let mut o = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |ui| {
                for (i, state) in states.iter_mut().enumerate() {
                    ui.push_id(i, |ui| {
                        let e = path_edit(ui, state, "current.ocio");
                        ids[i] = e.id;
                        if focus == Some(i) {
                            ui.memory_mut(|m| m.request_focus(e.id));
                        }
                        out.push((e.shown, e.apply));
                    });
                }
            });
            o.textures_delta.clear();
            out
        };
        let mut ids = [egui::Id::NULL; 2];
        let idle = frame(vec![], None, &mut ids);
        assert_eq!(idle[0].0, "current.ocio");
        frame(vec![], Some(0), &mut ids);
        let typed = frame(vec![egui::Event::Text(" a.ocio ".into())], None, &mut ids);
        assert_eq!(typed[0].0, "current.ocio a.ocio ", "the draft of the focused field grows");
        assert_eq!(typed[1].0, "current.ocio", "the other field is untouched");
        let still = frame(vec![], None, &mut ids);
        assert_eq!(still[0].0, "current.ocio a.ocio ", "the draft survives the next frame");
        assert_eq!(still[1].0, "current.ocio");
        let key = |pressed| egui::Event::Key { key: egui::Key::Enter, physical_key: None, pressed, repeat: false, modifiers: Default::default() };
        let done = frame(vec![key(true), key(false)], None, &mut ids);
        assert_eq!(done[0].1.as_deref(), Some("current.ocio a.ocio"), "Enter applies the trimmed draft");
        assert_eq!(done[1].1, None);
        let after = frame(vec![], None, &mut ids);
        assert_eq!(after[0].0, "current.ocio", "the draft is gone once applied");
    }

    /// A typed or browsed config path that does not load is an error on the state (the panel and
    /// the toolbar show it), never a silent fallback to the default config; the dialog then starts
    /// from the dialog memory, not from the bad path.
    #[test]
    fn a_bad_config_path_is_an_error_and_offers_no_folder() {
        let f = frame(ColorTag::scene_linear(), 1, "a.exr");
        let mut s = State::new(Sel { on: true, config: "ocio://default".to_owned(), ..Sel::default() });
        assert!(s.frame(&f, &SDR).is_ok());
        assert!(s.err.is_none() && s.active().is_some());
        s.set_config("no/such/dir/config.ocio".to_owned());
        assert!(s.err.is_some());
        s.frame(&f, &SDR).unwrap();
        assert!(s.err.is_some() && s.active().is_none());
        assert_eq!(s.start_dir(), None, "nothing loaded yet, and a missing folder is not offered");
        let (guard, file) = good_config("good");
        let dir = guard.path().to_path_buf();
        s.set_config(file.display().to_string());
        assert!(s.err.is_none(), "{:?}", s.err);
        assert_eq!(s.start_dir(), Some(dir.clone()));
        s.set_config("no/such/dir/config.ocio".to_owned());
        assert!(s.err.is_some());
        assert_eq!(s.start_dir(), None, "a missing folder is not offered");
        s.set_config("ocio://default".to_owned());
        assert!(s.err.is_none(), "a built-in URI loads");
    }

    /// The defaults of the ACES 2.0 CG config, and scene 18 % grey through them at
    /// the value vfx-ocio's reference points give (0.349 encoded for the SDR sRGB view).
    #[test]
    fn default_transform_renders_mid_grey() {
        let o = cg();
        let n = o.resolve(&Sel::default(), &o.scene(), false).unwrap();
        assert_eq!(n.input, "ACEScg");
        assert_eq!((n.display.as_str(), n.view.as_str()), ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"));
        let t = o.transform(&n, None, false).unwrap();
        let mut px = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut px, &Src::LINEAR, 0.0, 1.0);
        for c in &px[0][..3] {
            assert!((c - 0.349).abs() < 2e-3, "{px:?}");
        }
        eprintln!("default transform: LUT textures = {}", t.has_luts());
    }

    /// Exposure acts before the display transform (for a scene-linear input it equals a
    /// gain on the input) and the dial acts on the display value.
    #[test]
    fn exposure_and_dial_bracket_the_transform() {
        let o = cg();
        let t = tr(&o, &Sel::default(), &o.scene(), false);
        let mut a = [[0.09, 0.09, 0.09, 1.0]];
        let mut b = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut a, &Src::LINEAR, 1.0, 1.0);
        t.display(&mut b, &Src::LINEAR, 0.0, 1.0);
        assert!((a[0][0] - b[0][0]).abs() < 1e-6, "{a:?} vs {b:?}");
        let mut g = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut g, &Src::LINEAR, 0.0, 2.0);
        assert!((g[0][0] - b[0][0].sqrt()).abs() < 1e-6, "{g:?}");
    }

    /// The exposure is OCIO's `linearCC`: it acts in `scene_linear` whatever the input space.
    /// The same light viewed as `scene_linear` values and as ACEScct (log) codes shows the
    /// same picture at +1.5 EV, which a gain on the log codes would not; and moving the
    /// exposure back restores the first picture (the processor is moved, not rebuilt).
    #[test]
    fn exposure_acts_in_scene_linear_for_a_log_input() {
        let o = cg();
        let lin = tr(&o, &Sel::default(), &o.scene(), false);
        let log = tr(&o, &Sel::default(), &o.space(&tag("ACEScct"), Path::new("a.exr"), None).unwrap(), false);
        let light = [[0.18f32, 0.18, 0.18, 1.0], [0.5, 0.2, 0.05, 1.0], [2.0, 1.0, 0.3, 1.0], [0.01, 0.02, 0.005, 1.0]];
        let mut codes = light;
        o.cfg.processor("scene_linear", "ACEScct").unwrap().apply_rgba(&mut codes);
        let ev = 1.5;
        let (mut a, mut b) = (light, codes);
        lin.display(&mut a, &Src::LINEAR, ev, 1.0);
        log.display(&mut b, &Src::LINEAR, ev, 1.0);
        for (a, b) in a.iter().zip(&b) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 2e-4, "linear {a:?} vs log input {b:?}");
            }
        }
        // A gain on the codes (the old rule) is a different, wrong picture.
        let mut gained = codes.map(|p| [p[0] * 2f32.powf(ev), p[1] * 2f32.powf(ev), p[2] * 2f32.powf(ev), p[3]]);
        log.display(&mut gained, &Src::LINEAR, 0.0, 1.0);
        assert!(gained.iter().zip(&b).any(|(g, b)| (g[0] - b[0]).abs() > 0.05), "{gained:?} vs {b:?}");
        // The exposure moves both ways on the one processor: 0 EV, then +1.5 again, then
        // 0 EV gives the first picture back exactly.
        let mut first = codes;
        log.display(&mut first, &Src::LINEAR, 0.0, 1.0);
        assert_ne!(first, b, "the exposure must change the picture");
        let mut up = codes;
        log.display(&mut up, &Src::LINEAR, ev, 1.0);
        assert_eq!(up, b);
        let mut back = codes;
        log.display(&mut back, &Src::LINEAR, 0.0, 1.0);
        assert_eq!(back, first);
    }

    /// HDR views are hidden (the Studio config has PQ displays), SDR ones stay, and a
    /// name the config lacks is an error, not a substitute.
    #[test]
    fn only_sdr_is_offered_and_unknown_names_fail() {
        let o = studio();
        let scene = o.scene();
        let all: Vec<&str> = o.cfg.display_names();
        assert!(all.contains(&"Rec.2100-PQ - Display"));
        assert!(!o.displays(false).contains(&"Rec.2100-PQ - Display"));
        assert!(o.displays(false).contains(&"sRGB - Display"));
        let hdr = Sel { display: "Rec.2100-PQ - Display".into(), ..Sel::default() };
        assert!(o.resolve(&hdr, &scene, false).is_err());
        let bad = Sel { scene_view: "no such view".into(), ..Sel::default() };
        assert!(o.resolve(&bad, &scene, false).unwrap_err().to_string().contains("no such view"));
        // An HDR output offers the PQ display and its HDR views.
        assert!(o.displays(true).contains(&"Rec.2100-PQ - Display"));
        let n = o.resolve(&hdr, &scene, true).unwrap();
        assert!(o.views(&n.display, &scene.name, true).contains(&"ACES 2.0 - HDR 1000 nits (P3 D65)"));
    }

    /// A look replaces the view's looks and changes the picture.
    #[test]
    fn a_look_changes_the_picture() {
        let o = studio();
        let looks = o.looks();
        let look = *looks.first().expect("the Studio config has looks");
        let plain = tr(&o, &Sel::default(), &o.scene(), false);
        let looked = tr(&o, &Sel { scene_look: look.into(), ..Sel::default() }, &o.scene(), false);
        let (mut a, mut b) = ([[0.9, 0.2, 0.05, 1.0]], [[0.9, 0.2, 0.05, 1.0]]);
        plain.display(&mut a, &Src::LINEAR, 0.0, 1.0);
        looked.display(&mut b, &Src::LINEAR, 0.0, 1.0);
        assert_ne!(a, b, "look {look}");
    }

    /// Linear (HDR output) = the encoded transform decoded to light: for the sRGB
    /// display the display's code values run back through its own curve and
    /// primaries, so the linear result is the sRGB EOTF of the encoded one (within
    /// the float noise of two matrices); an HDR view uses the headroom (a 1000-nit
    /// view reaches ~10x SDR white) where the SDR view stays at white.
    #[test]
    fn linear_is_the_light_of_the_display() {
        let o = studio();
        let scene = o.scene();
        let sdr = o.resolve(&Sel::default(), &scene, true).unwrap();
        let enc = o.transform(&sdr, None, false).unwrap();
        let lin = o.transform(&sdr, None, true).unwrap();
        assert!(lin.linear() && !enc.linear() && lin.key != enc.key);
        let hdr_out = Target { hdr: true, white: 240.0, peak: 1000.0 };
        for v in [0.02f32, 0.18, 0.7, 3.0] {
            let (mut a, mut b) = ([[v, v * 0.8, v * 0.5, 1.0]], [[v, v * 0.8, v * 0.5, 1.0]]);
            enc.display(&mut a, &Src::LINEAR, 0.0, 1.0);
            let gain = display_gain(lin.absolute().then_some(100.0), &hdr_out);
            lin.display(&mut b, &Src { gain, ..Src::LINEAR }, 0.0, 1.0);
            for c in 0..3 {
                // `display` returns the canvas encoding: sRGB-encoded both ways.
                assert!((a[0][c] - b[0][c]).abs() < 2e-3, "{v}: encoded {:?} vs linear {:?}", a[0], b[0]);
            }
        }
        // Light in display-reference units (gain 1: 1.0 = 100 nits for an HDR view).
        let peak = |t: &Transform| {
            let mut px = [[64.0, 64.0, 64.0, 1.0]];
            t.display(&mut px, &Src::LINEAR, 0.0, 1.0);
            crate::transfer::eotf(px[0][1])
        };
        let hdr_sel = Sel { display: "Rec.2100-PQ - Display".into(), scene_view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(), ..Sel::default() };
        let hdr = tr(&o, &hdr_sel, &scene, true);
        assert!(peak(&hdr) > 5.0, "1000-nit view peak {}", peak(&hdr));
        assert!(peak(&lin) < 1.05, "SDR view peak {}", peak(&lin));
        // An HDR view is absolute: on a 250-nit SDR white its 100 nits are 0.4 of it;
        // an SDR view (and any encoded transform) is relative, and an SDR output never scales.
        let at = |white| Target { hdr: true, white, peak: 1000.0 };
        let g = |t: &Transform, target: &Target| display_gain(t.absolute().then_some(100.0), target);
        assert_eq!(g(&hdr, &at(250.0)), 0.4);
        assert_eq!((g(&lin, &at(250.0)), g(&enc, &at(250.0))), (1.0, 1.0));
        assert_eq!(display_gain(Some(100.0), &SDR), 1.0);
    }

    /// A config change resets the names; OCIO off shows the built-in transform.
    #[test]
    fn state_follows_the_choice() {
        let f = frame(ColorTag::scene_linear(), 1, "a.exr");
        let mut s = State::new(Sel { on: true, config: "ocio://default".into(), ..Sel::default() });
        s.frame(&f, &SDR).unwrap();
        assert!(s.active().is_some(), "{:?}", s.err);
        s.sel.scene_view = "no such view".into();
        s.rebuild();
        s.frame(&f, &SDR).unwrap();
        assert!(s.active().is_none() && s.err.as_deref().is_some_and(|e| e.contains("no such view")));
        s.set_config("ocio://studio-config-latest".into());
        s.frame(&f, &SDR).unwrap();
        assert!(s.sel.scene_view.is_empty() && s.active().is_some());
        // A reload is a new transform (new key), so the tile pipeline rebuilds too.
        let key = s.active().unwrap().key.clone();
        s.reload();
        s.frame(&f, &SDR).unwrap();
        assert_ne!(s.active().unwrap().key, key);
        // An HDR output makes the transform linear (a new key: the pipeline rebuilds).
        let key = s.active().unwrap().key.clone();
        s.set_hdr(true);
        s.frame(&f, &SDR).unwrap();
        assert!(s.active().unwrap().linear() && s.active().unwrap().key != key);
        s.set_on(false);
        s.frame(&f, &SDR).unwrap();
        assert!(s.active().is_none() && s.err.is_none());
    }

    #[test]
    fn admitted_alpha_scratch_is_reused_and_preserves_alpha_bits() {
        let ocio = cg();
        let transform = tr(&ocio, &Sel::default(), &ocio.scene(), false);
        let alpha_bits = [0x8000_0000, 0x7fc0_1234, 0x3f80_0000, 0x0000_0000];
        let mut scratch = vec![0.0; 4096];
        let allocation = scratch.as_ptr();
        let capacity = scratch.capacity();
        for _ in 0..3 {
            let mut pixels: Vec<[f32; 4]> = (0..4096)
                .map(|i| [0.18, 0.18, 0.18, f32::from_bits(alpha_bits[i % 4])])
                .collect();
            transform.display_scratch(&mut pixels, &mut scratch, &Src::LINEAR, 0.0, 1.0).unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                assert_eq!(pixel[3].to_bits(), alpha_bits[i % 4]);
                // -0, NaN, 1 and 0 alpha: no partial coverage, so the colour is the opaque one.
                assert_eq!(crate::transfer::code8(pixel[0]), 89);
            }
            assert_eq!(scratch.as_ptr(), allocation);
            assert_eq!(scratch.capacity(), capacity);
        }
    }

    #[test]
    fn short_alpha_scratch_refuses_before_changing_pixels() {
        let ocio = cg();
        let transform = tr(&ocio, &Sel::default(), &ocio.scene(), false);
        let mut pixels = [[0.18, 0.5, 1.0, f32::from_bits(0x7fc0_4567)]];
        let before = pixels.map(|p| p.map(f32::to_bits));
        assert!(transform.display_scratch(&mut pixels, &mut [], &Src::LINEAR, 1.0, 1.2).is_err());
        assert_eq!(pixels.map(|p| p.map(f32::to_bits)), before);
    }

    // ── COLOR_PLAN tests ─────────────────────────────────────────────────────

    /// The PQ HDR10 PNG (cICP 9/16/0/1, tagged `pq_rec2020_display`) on a PQ display: the
    /// viewing rules offer `Video (colorimetric)` (and the data view), never an ACES view; that is
    /// the default; and the chain to the HDR10 signal (the canvas light at SDR white, nits, PQ)
    /// gives the file's codes back within one 16-bit code.
    #[test]
    fn pq_png_is_an_identity_on_a_pq_display() {
        let o = studio();
        let space = o.space(&tag("pq_rec2020_display"), Path::new("frame.pq.png"), None).unwrap();
        assert_eq!((space.name.as_str(), space.class), ("Rec.2100-PQ - Display", Class::Video));
        let views = o.views("Rec.2100-PQ - Display", &space.name, true);
        assert!(!views.iter().any(|v| v.starts_with("ACES")), "{views:?}");
        // A scene look saved for EXRs (an LMT in ACES2065-1) is not the PQ frame's.
        let sel = Sel {
            display: "Rec.2100-PQ - Display".into(),
            scene_look: "ACES 1.3 Reference Gamut Compression".into(),
            ..Sel::default()
        };
        let names = o.resolve(&sel, &space, true).unwrap();
        assert_eq!((names.view.as_str(), names.look.as_str()), ("Video (colorimetric)", ""));
        let t = o.transform(&names, None, true).unwrap();
        assert!(t.absolute());
        let target = Target { hdr: true, white: 203.0, peak: 1000.0 };
        let src = Src { gain: display_gain(t.absolute().then_some(100.0), &target), ..Src::LINEAR };
        // f32 light holds ~7 digits per pixel; PQ's toe turns a component 10^8 below its
        // neighbours (code 1 next to 10000 nits) into noise codes, so the triples stay within a
        // range f32 carries.
        let codes: Vec<[u16; 3]> = vec![[0, 0, 0], [20000, 30000, 40000], [32768, 32768, 32768], [41000, 38000, 12000], [65535, 52000, 30000], [8000, 9000, 7000]];
        let mut px: Vec<[f32; 4]> = codes.iter().map(|c| [c[0], c[1], c[2]].map(|k| f32::from(k) / 65535.0)).map(|[r, g, b]| [r, g, b, 1.0]).collect();
        t.display(&mut px, &src, 0.0, 1.0);
        for (p, want) in px.iter().zip(&codes) {
            let light = [p[0], p[1], p[2]].map(crate::transfer::eotf);
            let nits = crate::present::rec2020_nits(light, target.white);
            for (n, w) in nits.iter().zip(want) {
                let got = (crate::present::pq(*n) * 65535.0).round();
                assert!((got - f32::from(*w)).abs() <= 1.0, "{want:?} -> {nits:?} -> {got}");
            }
        }
    }

    /// An sRGB still on the sRGB display, whether tagged as a texture (vfx-io's PNG default,
    /// `srgb_rec709_scene`) or as a display (`srgb_rec709_display`): `Video (colorimetric)` is
    /// the default view and the 8-bit codes come back within one code.
    #[test]
    fn srgb_png_is_an_identity_on_an_srgb_display() {
        let o = studio();
        for t in ["srgb_rec709_scene", "srgb_rec709_display"] {
            let space = o.space(&tag(t), Path::new("a.png"), None).unwrap();
            assert_eq!(space.class, Class::Video, "{t}");
            let names = o.resolve(&Sel::default(), &space, false).unwrap();
            assert_eq!((names.display.as_str(), names.view.as_str()), ("sRGB - Display", "Video (colorimetric)"), "{t}");
            let tr = o.transform(&names, None, false).unwrap();
            for k in [0u8, 1, 17, 64, 128, 200, 254, 255] {
                let v = f32::from(k) / 255.0;
                let mut px = [[v, v, v, 1.0]];
                tr.display(&mut px, &Src::LINEAR, 0.0, 1.0);
                let got = crate::transfer::code8(px[0][1]);
                assert!(got.abs_diff(k) <= 1, "{t}: {k} -> {got}");
            }
        }
    }

    /// An EXR with Rec.709 chromaticities through an ACES view is vfx-ocio's own processor from
    /// `lin_rec709_scene`; an untagged EXR follows the file rules (Studio: EXR -> ACES2065-1).
    #[test]
    fn exr_tags_follow_chromaticities_then_file_rules() {
        let o = studio();
        let rec709 = ColorTag { space: "lin_rec709_scene".into(), origin: TagOrigin::Chromaticities };
        let space = o.space(&rec709, Path::new("a.exr"), None).unwrap();
        assert_eq!(space.class, Class::Scene);
        let names = o.resolve(&Sel::default(), &space, false).unwrap();
        assert!(names.view.starts_with("ACES 2.0"), "{names:?}");
        let t = o.transform(&names, None, false).unwrap();
        let reference = o.cfg.display_processor("lin_rec709_scene", &names.display, &names.view).unwrap();
        let light = [[0.18f32, 0.18, 0.18, 1.0], [0.9, 0.3, 0.05, 1.0], [4.0, 2.0, 0.5, 1.0]];
        let (mut a, mut b) = (light, light);
        t.display(&mut a, &Src::LINEAR, 0.0, 1.0);
        reference.apply_rgba(&mut b);
        for (a, b) in a.iter().zip(&b) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-4, "{a:?} vs {b:?}");
            }
        }
        let untagged = o.space(&ColorTag::scene_linear(), Path::new("/shots/a.0001.exr"), None).unwrap();
        assert_eq!(untagged.name, "ACES2065-1");
        assert!(untagged.how.contains("file rules"), "{}", untagged.how);
    }

    /// An untagged integer still follows the file rules; when only their default rule matches,
    /// the explicit fallback (`srgb_rec709_display`) names it: `sRGB - Display` in Studio, which
    /// is also what that default rule says.
    #[test]
    fn untagged_png_follows_the_file_rules() {
        let o = studio();
        let untagged = ColorTag { space: "srgb_rec709_display".into(), origin: TagOrigin::Untagged };
        let space = o.space(&untagged, Path::new("C:/plates/a.png"), None).unwrap();
        assert_eq!(space.name, "sRGB - Display");
        assert_eq!(o.cfg.colorspace_from_filepath("C:/plates/a.png"), Some("sRGB - Display"));
    }

    /// Frames in one colour space share one transform: the cache is asked by key before
    /// anything is built (a sequence does not rebuild per frame).
    #[test]
    fn frames_of_one_space_share_the_transform() {
        let mut s = State::new(Sel { on: true, config: "ocio://default".into(), ..Sel::default() });
        s.frame(&frame(ColorTag::scene_linear(), 1, "a.0001.exr"), &SDR).unwrap();
        let first = s.active().unwrap().clone();
        s.frame(&frame(ColorTag::scene_linear(), 2, "a.0002.exr"), &SDR).unwrap();
        assert!(Arc::ptr_eq(&first, s.active().unwrap()));
        assert_eq!(s.built.len(), 1);
    }

    /// A tag the config lacks is a typed error, not a substitute: the state shows no OCIO
    /// transform and says why, the built-in decode still knows the curve, and an override fixes
    /// it.
    #[test]
    fn an_unknown_tag_is_a_typed_error() {
        let o = studio();
        let err = o.space(&tag("KodakLog"), Path::new("a.cin"), None).unwrap_err();
        assert!(matches!(&err, SpaceError::Unknown { space, .. } if space == "KodakLog"), "{err}");
        let f = frame(tag("KodakLog"), 7, "a.cin");
        let mut s = State::new(Sel { on: true, config: "ocio://studio-config-latest".into(), ..Sel::default() });
        let src = s.frame(&f, &SDR).unwrap();
        assert!(s.active().is_none());
        assert!(s.err.as_deref().is_some_and(|e| e.contains("KodakLog")), "{:?}", s.err);
        assert_eq!(src.builtin.tf, exr_import::Tf::Cineon);
        s.set_override(Some("ADX10".into()));
        s.frame(&f, &SDR).unwrap();
        assert!(s.active().is_some() && s.err.is_none(), "{:?}", s.err);
        assert!(s.shown().and_then(|x| x.space.clone()).is_some_and(|x| x.is_ok_and(|x| x.how == "override")));
    }

    /// Associated alpha: the OCIO twin's canvas pixel is the transform of the unpremultiplied
    /// colour, premultiplied in light; straight alpha: of the colour as stored.
    #[test]
    fn alpha_is_processed_unpremultiplied() {
        let o = studio();
        let t = tr(&o, &Sel::default(), &o.scene(), false);
        let (c, a) = ([0.4f32, 0.2, 0.1], 0.5f32);
        let opaque = |rgb: [f32; 3]| {
            let mut px = [[rgb[0], rgb[1], rgb[2], 1.0]];
            t.display(&mut px, &Src::LINEAR, 0.0, 1.0);
            [px[0][0], px[0][1], px[0][2]]
        };
        let want = opaque(c).map(|d| crate::transfer::oetf(crate::transfer::eotf(d) * a));
        let mut assoc = [[c[0] * a, c[1] * a, c[2] * a, a]];
        t.display(&mut assoc, &Src::LINEAR, 0.0, 1.0);
        let mut straight = [[c[0], c[1], c[2], a]];
        t.display(&mut straight, &Src { straight: true, ..Src::LINEAR }, 0.0, 1.0);
        for k in 0..3 {
            assert!((assoc[0][k] - want[k]).abs() < 1e-5, "{assoc:?} vs {want:?}");
            assert!((straight[0][k] - want[k]).abs() < 1e-5, "{straight:?} vs {want:?}");
        }
        assert_eq!((assoc[0][3], straight[0][3]), (a, a));
        // Premultiplying the display value directly (the old chain) is a different picture.
        let mut old = [[c[0] * a, c[1] * a, c[2] * a, 1.0]];
        t.display(&mut old, &Src::LINEAR, 0.0, 1.0);
        assert!((old[0][0] - want[0]).abs() > 1e-3);
    }

    /// The built-in decode (no OCIO) of each display tag equals OCIO's own colour space taken to
    /// its display reference and on to linear Rec.709: sRGB, gamma 2.2 / 2.4, Display P3, PQ
    /// (units of 100 nits) and HLG (its 1000-nit reference display).
    #[test]
    fn builtin_decode_matches_the_config() {
        let o = studio();
        let to709 = vfx_ocio::Transform::Matrix(MatrixTransform {
            name: String::new(),
            matrix: conversion_matrix_from_xyz_d65(&REC709, Adaptation::None).unwrap(),
            offset: [0.0; 4],
            direction: TransformDirection::Forward,
        });
        let codes = [0.0f32, 0.02, 0.1, 0.25, 0.5, 0.58, 0.75, 0.9, 1.0];
        for t in ["srgb_rec709_display", "g22_rec709_display", "g24_rec709_display", "srgb_p3d65_display", "pq_rec2020_display", "hlg_rec2020_display"] {
            let cs = o.cfg.colorspace(t).unwrap_or_else(|| panic!("{t}"));
            let to_ref = cs.to_display_reference().cloned().or_else(|| cs.from_display_reference().map(|x| x.clone().inverse())).unwrap();
            let chain = vfx_ocio::Transform::Group(GroupTransform { name: String::new(), transforms: vec![to_ref, to709.clone()], direction: TransformDirection::Forward });
            let proc = o.cfg.processor_from_transform(&chain, TransformDirection::Forward).unwrap();
            let builtin = exr_import::builtin(&tag(t)).unwrap();
            for &r in &codes {
                for rgb in [[r, r, r], [r, r * 0.5, 1.0 - r]] {
                    let mut want = [[rgb[0], rgb[1], rgb[2], 1.0]];
                    proc.apply_rgba(&mut want);
                    let got = crate::transfer::decode(&builtin, rgb);
                    for c in 0..3 {
                        let tol = 2e-4 + 2e-4 * want[0][c].abs();
                        assert!((got[c] - want[0][c]).abs() < tol, "{t} {rgb:?}: built-in {got:?} vs OCIO {:?}", want[0]);
                    }
                }
            }
        }
    }

    /// Our own HDR Save view EXR (display light at SDR white 203 nits: `whiteLuminance` 203, no
    /// `colorInteropID`) is display-referred CIE XYZ through the config's display reference role
    /// (`Video (colorimetric)`), and it reopens as it was saved on both kinds of display: on the
    /// relative sRGB display 1.0 is white again (not 2.03x over it); on a PQ display on an HDR
    /// output 1.0 is 203 nits whatever the SDR white.
    #[test]
    fn white_luminance_exr_is_display_light() {
        let o = studio();
        let mut h = exr_core::header::Header::new();
        h.add_white_luminance(203.0).unwrap();
        let f = frame(exr_import::exr_tag(&h).unwrap(), 9, "shot.exr");
        let space = o.space(&f.tag, Path::new("shot.exr"), None).unwrap();
        assert_eq!((space.name.as_str(), space.class), ("CIE XYZ-D65 - Display-referred", Class::Video));
        let names = o.resolve(&Sel::default(), &space, false).unwrap();
        assert_eq!(names.view, "Video (colorimetric)");
        let t = o.transform(&names, space.pre, false).unwrap();
        let mut px = [[1.0, 1.0, 1.0, 1.0], [0.18, 0.18, 0.18, 1.0]];
        t.display(&mut px, &Src::LINEAR, 0.0, 1.0);
        assert!((px[0][1] - 1.0).abs() < 1e-3, "{px:?}");
        assert!((px[1][1] - crate::transfer::oetf(0.18)).abs() < 1e-3, "{px:?}");
        // PQ display, HDR output with a 250-nit SDR white: 1.0 lands at 203 nits.
        let mut s = State::new(Sel { on: true, config: "ocio://studio-config-latest".into(), display: "Rec.2100-PQ - Display".into(), ..Sel::default() });
        s.set_hdr(true);
        let hdr = Target { hdr: true, white: 250.0, peak: 1000.0 };
        let src = s.frame(&f, &hdr).unwrap();
        let mut px = [[1.0, 1.0, 1.0, 1.0]];
        s.active().unwrap().display(&mut px, &src, 0.0, 1.0);
        let nits = crate::transfer::eotf(px[0][1]) * hdr.white;
        assert!((nits - 203.0).abs() < 0.5, "{nits}");
        // Without OCIO: the same white, relative on SDR, at its nits on HDR.
        let mut off = State::new(Sel::default());
        assert_eq!(off.frame(&f, &SDR).unwrap().gain, 1.0);
        assert!((off.frame(&f, &hdr).unwrap().gain - 203.0 / 250.0).abs() < 1e-6);
    }

    /// The pixel inspector's scene-linear values: the stored value through the config from the
    /// frame's space (an sRGB texture's 0.5 is 0.214 in Rec.709, a different triple in ACEScg).
    #[test]
    fn inspector_shows_scene_linear() {
        let f = frame(tag("srgb_rec709_scene"), 3, "a.png");
        let mut s = State::new(Sel { on: true, config: "ocio://studio-config-latest".into(), ..Sel::default() });
        s.frame(&f, &SDR).unwrap();
        let lin = s.scene_linear([0.5, 0.5, 0.5]).unwrap();
        // Neutral stays neutral; 0.214 Rec.709 is 0.214 grey in ACEScg (same white after CAT).
        assert!(lin.iter().all(|v| (v - 0.214).abs() < 2e-3), "{lin:?}");
    }
}
