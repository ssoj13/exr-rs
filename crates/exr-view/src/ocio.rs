//! The OCIO display transform of the colour image: the config, the user's choice
//! (input colour space, display, view, look), the transform the tile shader and the
//! CPU twin both run, and the Colour panel.
//!
//! Everything comes from `vfx-ocio` (the OpenColorIO 2.5 port Playa and vfx-view use):
//! ONE `Processor` gives the WGSL spliced into the tile shader (`wgsl_prelude`, bound
//! through `GpuOcioResources::build_for(.., FRAGMENT)` in `tiles::render`) and the
//! exact CPU twin (`apply_rgba`) that Save view and the GPU test use.
//!
//! The chain in colour mode (`channel_mode == RGBA`) is OCIO's `LegacyViewingPipeline`,
//! as `ociodisplay` builds it: the input to `scene_linear` and the exposure there (the
//! pipeline's `linearCC`, a dynamic `ExposureContrastTransform`: an exposure for log and
//! display-referred inputs too, moved per frame as a uniform / `set_dynamic`, no
//! rebuild), the display/view transform (with its look), then the gamma dial on the
//! display-encoded result (where `ociodisplay` applies its display gamma: an
//! `ExponentTransform` after the display/view transform). No sRGB OETF: the display
//! colour space already encodes. Isolated channels are data and keep the built-in path.
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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use vfx_ocio::builtin::embedded;
use vfx_ocio::color_matrix::{Adaptation, REC709, conversion_matrix_from_xyz_d65};
use vfx_ocio::{
    DisplayViewTransform, DynamicPropertyType, EcDynamic, Encoding, ExposureContrastTransform, GpuLanguage,
    GpuProcessor, GpuShaderCode, GroupTransform, LegacyViewingPipeline, MatrixTransform, Processor,
    ReferenceSpaceType, TransformDirection,
};

/// What the user picked; persisted (`persist::Config::ocio`). Empty names mean the
/// config's defaults ([`Ocio::resolve`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sel {
    /// OCIO is the display transform of the colour image (else the built-in one).
    pub on: bool,
    /// A config file or an `ocio://` URI; empty = `$OCIO`, else `ocio://default`.
    pub config: String,
    /// Input colour space; empty = the `scene_linear` role.
    pub input: String,
    /// Display; empty = the first one offered.
    pub display: String,
    /// View; empty = the display's first one offered.
    pub view: String,
    /// Look applied instead of the view's own looks; empty = the view's looks.
    pub look: String,
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

/// Names a [`Sel`] resolves to in one config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
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

    /// Every colour space, config order.
    pub fn inputs(&self) -> Vec<&str> {
        self.cfg.colorspace_names().collect()
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

    /// The views of `display` offered (config order): all with `hdr`, else the
    /// non-HDR ones (data views included).
    pub fn views(&self, display: &str, hdr: bool) -> Vec<&str> {
        self.cfg
            .get_views(display)
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

    /// `sel`'s names in this config, empty ones replaced by the defaults. A name the
    /// config lacks is an error, not a silent substitute: the panel says which.
    pub fn resolve(&self, sel: &Sel, hdr: bool) -> Result<Names> {
        let input = if sel.input.is_empty() {
            if !self.cfg.has_role("scene_linear") {
                bail!("the config has no scene_linear role: pick an input colour space");
            }
            "scene_linear".to_owned()
        } else if self.cfg.colorspace(&sel.input).is_some() {
            sel.input.clone()
        } else {
            bail!("the config has no colour space \"{}\"", sel.input);
        };
        // What a name is missing as: an HDR display on an SDR output is "no SDR display".
        let kind = if hdr { "" } else { "SDR " };
        let displays = self.displays(hdr);
        let display = if sel.display.is_empty() {
            displays.first().copied().ok_or_else(|| anyhow!("the config has no {kind}display"))?
        } else if displays.contains(&sel.display.as_str()) {
            sel.display.as_str()
        } else {
            bail!("the config has no {kind}display \"{}\"", sel.display);
        };
        let views = self.views(display, hdr);
        let view = if sel.view.is_empty() {
            views.first().copied().ok_or_else(|| anyhow!("\"{display}\" has no {kind}view"))?
        } else if views.contains(&sel.view.as_str()) {
            sel.view.as_str()
        } else {
            bail!("\"{display}\" has no {kind}view \"{}\"", sel.view);
        };
        if !sel.look.is_empty() && self.cfg.looks().get(&sel.look).is_none() {
            bail!("the config has no look \"{}\"", sel.look);
        }
        Ok(Names { input, display: display.to_owned(), view: view.to_owned(), look: sel.look.clone() })
    }

    /// The viewer transform of `names`, built as OCIO viewers build it
    /// (`LegacyViewingPipeline`, as `ociodisplay` does): the exposure is the pipeline's
    /// `linearCC`, a DYNAMIC exposure applied in `scene_linear` (so it is an exposure for a
    /// log or display-referred input too, not a gain on its codes), moved per frame without
    /// a rebuild ([`Transform::display`], the tile shader's uniform); then the display/view
    /// transform (a look replaces the view's own looks, OCIO's looks override); `linear`
    /// continues it to linear Rec.709 light as the pipeline's `displayCC` (see the module
    /// doc). Needs the config's `scene_linear` role, as OCIO's pipeline does; a data input
    /// bypasses everything, the exposure included, as in OCIO.
    pub fn transform(&self, names: &Names, linear: bool) -> Result<Transform> {
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
        let proc = pipeline.processor(&self.cfg)?;
        let key = format!(
            "{}#{}|{}|{}|{}|{}|{}",
            self.src,
            self.serial,
            names.input,
            names.display,
            names.view,
            names.look,
            if linear { "linear" } else { "encoded" }
        );
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
    /// reference 1.0 = 100 nits), not relative to SDR white: see [`Self::gain`].
    absolute: bool,
}

impl Transform {
    /// Compile `proc` for the GPU; `linear` says it ends in linear light,
    /// `absolute` that the light is in absolute nits (see [`Self::gain`]).
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

    /// The gain from the transform's light to the canvas (1.0 = SDR white of
    /// `white` nits). An SDR view's light is relative (its 1.0 IS SDR white, as the
    /// OS shows SDR content on an HDR display): 1. An HDR view's light is absolute
    /// (display reference 1.0 = 100 nits; a 1000-nit view peaks at 10): 100/white,
    /// so the present pass, which scales canvas 1.0 to `white` nits, lands on the
    /// nits the view meant.
    pub fn gain(&self, white: f32) -> f32 {
        if self.absolute { 100.0 / white.max(1.0) } else { 1.0 }
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

    /// CPU twin of the tile shader's OCIO colour branch over linear RGBA pixels:
    /// the transform with its dynamic exposure at `exposure_ev` (in `scene_linear`), for a
    /// linear transform `gain` ([`Self::gain`]), the gamma dial and, for a linear transform,
    /// the canvas encoding; alpha stays the input's.
    pub fn display(&self, px: &mut [[f32; 4]], exposure_ev: f32, gamma: f32, gain: f32) {
        let mut alpha = vec![0.0; px.len()];
        self.display_kernel(px, &mut alpha, exposure_ev, gamma, gain);
    }

    /// The exporter supplies admitted reusable alpha storage alongside its row.
    pub(crate) fn display_scratch(
        &self,
        px: &mut [[f32; 4]],
        alpha: &mut [f32],
        exposure_ev: f32,
        gamma: f32,
        gain: f32,
    ) -> Result<(), &'static str> {
        if alpha.len() != px.len() {
            return Err("OCIO alpha scratch must match the pixel row");
        }
        self.display_kernel(px, alpha, exposure_ev, gamma, gain);
        Ok(())
    }

    fn display_kernel(
        &self,
        px: &mut [[f32; 4]],
        alpha: &mut [f32],
        exposure_ev: f32,
        gamma: f32,
        gain: f32,
    ) {
        use rayon::prelude::*;
        px.par_iter_mut().zip(alpha.par_iter_mut()).for_each(|(p, a)| {
            *a = p[3];
            // Alpha is restored verbatim below. A NaN alpha must not enter RGB
            // matrix operations, where even a zero alpha coefficient propagates it.
            p[3] = 1.0;
        });
        self.cpu_at(exposure_ev).1.apply_rgba(px);
        let linear = self.linear;
        px.par_iter_mut().zip(alpha.par_iter()).for_each(|(p, &a)| {
            for c in &mut p[..3] {
                *c = if linear {
                    crate::transfer::oetf(crate::transfer::dial(*c * gain, gamma))
                } else {
                    crate::transfer::dial(*c, gamma)
                };
            }
            p[3] = a;
        });
    }
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
    /// The transform in use: OCIO is on and it built.
    active: Option<Arc<Transform>>,
    /// Why OCIO is on but not shown.
    pub err: Option<String>,
    /// The path field's widget state. Only the field's own bookkeeping: where the config dialog
    /// starts is the dialog memory's (`file_pick::DialogMemory`, key `"ocio"`).
    field: egui_file_field::FileFieldState,
}

impl State {
    /// A state for `sel`, built (the device is assumed to filter until
    /// [`Self::set_filterable`] says otherwise).
    pub fn new(sel: Sel) -> Self {
        let field = egui_file_field::FileFieldState::default();
        let mut s = Self { sel, filterable: true, hdr: false, cfg: None, active: None, err: None, field };
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

    /// The transform the colour image goes through, `None` = the built-in one.
    pub fn active(&self) -> Option<&Arc<Transform>> {
        self.active.as_ref()
    }

    /// Load the config when its source changed, then (OCIO on) build the transform
    /// of the choice; a failure is kept in `err` and the built-in transform shows.
    pub fn rebuild(&mut self) {
        self.err = None;
        if !self.sel.on {
            self.active = None;
            return;
        }
        match self.build() {
            Ok(t) => {
                if self.active.as_ref().is_none_or(|a| a.key != t.key) {
                    log::info!("OCIO display transform: {}", t.key);
                    self.active = Some(Arc::new(t));
                }
            }
            Err(e) => {
                log::warn!("OCIO: {e:#}");
                self.err = Some(format!("{e:#}"));
                self.active = None;
            }
        }
    }

    fn build(&mut self) -> Result<Transform> {
        self.load_config();
        let ocio = self.config().map_err(|e| anyhow!(e))?;
        let t = ocio.transform(&ocio.resolve(&self.sel, self.hdr)?, self.hdr)?;
        if t.has_luts() && !self.filterable {
            bail!("this transform samples LUT textures, and this GPU cannot filter 32-bit float textures");
        }
        Ok(t)
    }

    /// Load the config of `sel.config` when its source changed.
    fn load_config(&mut self) {
        let src = source(&self.sel.config);
        if self.cfg.as_ref().is_none_or(|(s, _)| *s != src) {
            let loaded = Ocio::load(&src).map_err(|e| format!("{e:#}"));
            self.cfg = Some((src, loaded));
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

    /// Use another config: the names of the old one mean nothing there, so they go
    /// back to the new config's defaults.
    pub fn set_config(&mut self, config: String) {
        self.sel = Sel { on: self.sel.on, config, ..Sel::default() };
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
        let before = self.sel.clone();
        let mut reload = false;
        ui.checkbox(&mut self.sel.on, "OCIO display transform")
            .on_hover_text("Show the colour image through the OCIO display / view (isolated channels stay raw).\nOff: the built-in sRGB display with the tone operator.");
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
            let resolved = ocio.resolve(&self.sel, hdr).ok();
            let views = resolved.as_ref().map(|n| owned(ocio.views(&n.display, hdr))).unwrap_or_default();
            let (inputs, looks) = (owned(ocio.inputs()), owned(ocio.looks()));
            let presets: Vec<_> = PRESETS
                .iter()
                .filter(|p| hdr || !p.hdr)
                .map(|p| {
                    let ok = displays.iter().any(|d| d == p.display) && ocio.views(p.display, hdr).contains(&p.view);
                    let shown = resolved.as_ref().is_some_and(|n| n.display == p.display && n.view == p.view);
                    (p, ok, shown)
                })
                .collect();
            let dflt = |f: fn(&Names) -> &str| resolved.as_ref().map_or(String::new(), |n| f(n).to_owned());

            // Keep the quick choices above the advanced controls, including in a
            // short Settings dock. Both entry points use this same preset table.
            ui.label("Monitor preset").on_hover_text("Choose the image rendering; this does not change UI reference white.");
            ui.horizontal_wrapped(|ui| {
                for (p, ok, shown) in presets {
                    let tip = if ok {
                        p.hint.to_owned()
                    } else {
                        format!("{}\n\nUnavailable: this config has no \"{}\" / \"{}\".", p.hint, p.display, p.view)
                    };
                    let button = egui::Button::new(p.label).selected(self.sel.on && shown);
                    if ui.add_enabled(ok, button).on_hover_text(tip).clicked() {
                        self.sel.on = true;
                        self.sel.display = p.display.to_owned();
                        self.sel.view = p.view.to_owned();
                    }
                }
            });
            ui.end_row();
            ui.label("Input").on_hover_text("The colour space of the image's values.");
            combo(ui, "ocio.input", &mut self.sel.input, &inputs, "scene_linear (role)");
            ui.end_row();
            ui.label("Display").on_hover_text(
                "The display the image is rendered for. SDR output: the monitor's own colour space (HDR ones need an HDR output, Settings > Display).",
            );
            let d_before = self.sel.display.clone();
            combo(ui, "ocio.display", &mut self.sel.display, &displays, &dflt(|n| &n.display));
            if self.sel.display != d_before {
                // The view list belongs to the display.
                self.sel.view.clear();
            }
            ui.end_row();
            ui.label("View").on_hover_text("The rendering of the scene for the display.");
            combo(ui, "ocio.view", &mut self.sel.view, &views, &dflt(|n| &n.view));
            ui.end_row();
            ui.label("Look").on_hover_text("A look applied instead of the view's own looks.");
            combo(ui, "ocio.look", &mut self.sel.look, &looks, "(the view's looks)");
            ui.end_row();
        });

        // One rebuild for whatever changed this frame.
        let changed = self.sel != before || reload;
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
mod tests {
    use super::*;

    fn cg() -> Ocio {
        Ocio::load("ocio://default").expect("the default built-in config")
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

    /// A typed or browsed config path that does not load is an error on the state (the panel and
    /// the toolbar show it), never a silent fallback to the default config; the dialog then starts
    /// from the dialog memory, not from the bad path.
    #[test]
    fn a_bad_config_path_is_an_error_and_offers_no_folder() {
        let mut s = State::new(Sel { on: true, config: "ocio://default".to_owned(), ..Sel::default() });
        assert!(s.err.is_none() && s.active().is_some());
        s.set_config("no/such/dir/config.ocio".to_owned());
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

    /// The defaults of the ACES 2.0 CG config, and scene 18 % grey through them at
    /// the value vfx-ocio's reference points give (0.349 encoded for the SDR sRGB view).
    #[test]
    fn default_transform_renders_mid_grey() {
        let o = cg();
        let n = o.resolve(&Sel::default(), false).unwrap();
        assert_eq!(n.input, "scene_linear");
        assert_eq!((n.display.as_str(), n.view.as_str()), ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"));
        let t = o.transform(&n, false).unwrap();
        let mut px = [[0.18, 0.18, 0.18, 0.4]];
        t.display(&mut px, 0.0, 1.0, 1.0);
        for c in &px[0][..3] {
            assert!((c - 0.349).abs() < 2e-3, "{px:?}");
        }
        assert_eq!(px[0][3], 0.4, "alpha is the input's");
        eprintln!("default transform: LUT textures = {}", t.has_luts());
    }

    /// Exposure acts before the display transform (for a scene-linear input it equals a
    /// gain on the input) and the dial acts on the display value.
    #[test]
    fn exposure_and_dial_bracket_the_transform() {
        let o = cg();
        let t = o.transform(&o.resolve(&Sel::default(), false).unwrap(), false).unwrap();
        let mut a = [[0.09, 0.09, 0.09, 1.0]];
        let mut b = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut a, 1.0, 1.0, 1.0);
        t.display(&mut b, 0.0, 1.0, 1.0);
        assert!((a[0][0] - b[0][0]).abs() < 1e-6, "{a:?} vs {b:?}");
        let mut g = [[0.18, 0.18, 0.18, 1.0]];
        t.display(&mut g, 0.0, 2.0, 1.0);
        assert!((g[0][0] - b[0][0].sqrt()).abs() < 1e-6, "{g:?}");
    }

    /// The exposure is OCIO's `linearCC`: it acts in `scene_linear` whatever the input space.
    /// The same light viewed as `scene_linear` values and as ACEScct (log) codes shows the
    /// same picture at +1.5 EV, which a gain on the log codes would not; and moving the
    /// exposure back restores the first picture (the processor is moved, not rebuilt).
    #[test]
    fn exposure_acts_in_scene_linear_for_a_log_input() {
        let o = cg();
        let lin = o.transform(&o.resolve(&Sel::default(), false).unwrap(), false).unwrap();
        let log_sel = Sel { input: "ACEScct".into(), ..Sel::default() };
        let log = o.transform(&o.resolve(&log_sel, false).unwrap(), false).unwrap();
        let light = [[0.18f32, 0.18, 0.18, 1.0], [0.5, 0.2, 0.05, 1.0], [2.0, 1.0, 0.3, 1.0], [0.01, 0.02, 0.005, 1.0]];
        let mut codes = light;
        o.cfg.processor("scene_linear", "ACEScct").unwrap().apply_rgba(&mut codes);
        let ev = 1.5;
        let (mut a, mut b) = (light, codes);
        lin.display(&mut a, ev, 1.0, 1.0);
        log.display(&mut b, ev, 1.0, 1.0);
        for (a, b) in a.iter().zip(&b) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 2e-4, "linear {a:?} vs log input {b:?}");
            }
        }
        // A gain on the codes (the old rule) is a different, wrong picture.
        let mut gained = codes.map(|p| [p[0] * 2f32.powf(ev), p[1] * 2f32.powf(ev), p[2] * 2f32.powf(ev), p[3]]);
        log.display(&mut gained, 0.0, 1.0, 1.0);
        assert!(gained.iter().zip(&b).any(|(g, b)| (g[0] - b[0]).abs() > 0.05), "{gained:?} vs {b:?}");
        // The exposure moves both ways on the one processor: 0 EV, then +1.5 again, then
        // 0 EV gives the first picture back exactly.
        let mut first = codes;
        log.display(&mut first, 0.0, 1.0, 1.0);
        assert_ne!(first, b, "the exposure must change the picture");
        let mut up = codes;
        log.display(&mut up, ev, 1.0, 1.0);
        assert_eq!(up, b);
        let mut back = codes;
        log.display(&mut back, 0.0, 1.0, 1.0);
        assert_eq!(back, first);
    }

    /// HDR views are hidden (the Studio config has PQ displays), SDR ones stay, and a
    /// name the config lacks is an error, not a substitute.
    #[test]
    fn only_sdr_is_offered_and_unknown_names_fail() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let all: Vec<&str> = o.cfg.display_names();
        assert!(all.contains(&"Rec.2100-PQ - Display"));
        assert!(!o.displays(false).contains(&"Rec.2100-PQ - Display"));
        assert!(o.displays(false).contains(&"sRGB - Display"));
        let hdr = Sel { display: "Rec.2100-PQ - Display".into(), ..Sel::default() };
        assert!(o.resolve(&hdr, false).is_err());
        let bad = Sel { input: "no such space".into(), ..Sel::default() };
        assert!(o.resolve(&bad, false).unwrap_err().to_string().contains("no such space"));
        // An HDR output offers the PQ display and its HDR views.
        assert!(o.displays(true).contains(&"Rec.2100-PQ - Display"));
        let n = o.resolve(&hdr, true).unwrap();
        assert!(o.views(&n.display, true).contains(&"ACES 2.0 - HDR 1000 nits (P3 D65)"));
    }

    /// A look replaces the view's looks and changes the picture.
    #[test]
    fn a_look_changes_the_picture() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let looks = o.looks();
        let look = *looks.first().expect("the Studio config has looks");
        let plain = o.transform(&o.resolve(&Sel::default(), false).unwrap(), false).unwrap();
        let looked = o.transform(&o.resolve(&Sel { look: look.into(), ..Sel::default() }, false).unwrap(), false).unwrap();
        let (mut a, mut b) = ([[0.9, 0.2, 0.05, 1.0]], [[0.9, 0.2, 0.05, 1.0]]);
        plain.display(&mut a, 0.0, 1.0, 1.0);
        looked.display(&mut b, 0.0, 1.0, 1.0);
        assert_ne!(a, b, "look {look}");
    }

    /// Linear (HDR output) = the encoded transform decoded to light: for the sRGB
    /// display the display's code values run back through its own curve and
    /// primaries, so the linear result is the sRGB EOTF of the encoded one (within
    /// the float noise of two matrices); an HDR view uses the headroom (a 1000-nit
    /// view reaches ~10x SDR white) where the SDR view stays at white.
    #[test]
    fn linear_is_the_light_of_the_display() {
        let o = Ocio::load("ocio://studio-config-latest").unwrap();
        let sdr = o.resolve(&Sel::default(), true).unwrap();
        let enc = o.transform(&sdr, false).unwrap();
        let lin = o.transform(&sdr, true).unwrap();
        assert!(lin.linear() && !enc.linear() && lin.key != enc.key);
        for v in [0.02f32, 0.18, 0.7, 3.0] {
            let (mut a, mut b) = ([[v, v * 0.8, v * 0.5, 1.0]], [[v, v * 0.8, v * 0.5, 1.0]]);
            enc.display(&mut a, 0.0, 1.0, 1.0);
            lin.display(&mut b, 0.0, 1.0, lin.gain(240.0));
            for c in 0..3 {
                // `display` returns the canvas encoding: sRGB-encoded both ways.
                assert!((a[0][c] - b[0][c]).abs() < 2e-3, "{v}: encoded {:?} vs linear {:?}", a[0], b[0]);
            }
        }
        // Light in display-reference units (gain 1: 1.0 = 100 nits for an HDR view).
        let peak = |t: &Transform| {
            let mut px = [[64.0, 64.0, 64.0, 1.0]];
            t.display(&mut px, 0.0, 1.0, 1.0);
            crate::transfer::eotf(px[0][1])
        };
        let hdr_sel = Sel { display: "Rec.2100-PQ - Display".into(), view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(), ..Sel::default() };
        let hdr = o.transform(&o.resolve(&hdr_sel, true).unwrap(), true).unwrap();
        assert!(peak(&hdr) > 5.0, "1000-nit view peak {}", peak(&hdr));
        assert!(peak(&lin) < 1.05, "SDR view peak {}", peak(&lin));
        // An HDR view is absolute: on a 250-nit SDR white its 100 nits are 0.4 of it;
        // an SDR view (and any encoded transform) is relative.
        assert_eq!(hdr.gain(250.0), 0.4);
        assert_eq!((lin.gain(250.0), enc.gain(250.0)), (1.0, 1.0));
    }

    /// A config change resets the names; OCIO off shows the built-in transform.
    #[test]
    fn state_follows_the_choice() {
        let mut s = State::new(Sel { on: true, config: "ocio://default".into(), ..Sel::default() });
        assert!(s.active().is_some(), "{:?}", s.err);
        s.sel.view = "no such view".into();
        s.rebuild();
        assert!(s.active().is_none() && s.err.as_deref().is_some_and(|e| e.contains("no such view")));
        s.set_config("ocio://studio-config-latest".into());
        assert!(s.sel.view.is_empty() && s.active().is_some());
        // A reload is a new transform (new key), so the tile pipeline rebuilds too.
        let key = s.active().unwrap().key.clone();
        s.reload();
        assert_ne!(s.active().unwrap().key, key);
        // An HDR output makes the transform linear (a new key: the pipeline rebuilds).
        let key = s.active().unwrap().key.clone();
        s.set_hdr(true);
        assert!(s.active().unwrap().linear() && s.active().unwrap().key != key);
        s.set_on(false);
        assert!(s.active().is_none() && s.err.is_none());
    }

    #[test]
    fn admitted_alpha_scratch_is_reused_and_preserves_alpha_bits() {
        let ocio = Ocio::load("ocio://default").unwrap();
        let transform = ocio.transform(&ocio.resolve(&Sel::default(), false).unwrap(), false).unwrap();
        let alpha_bits = [0x8000_0000, 0x7fc0_1234, 0x3f80_0000, 0x0000_0000];
        let mut scratch = vec![0.0; 4096];
        let allocation = scratch.as_ptr();
        let capacity = scratch.capacity();
        for _ in 0..3 {
            let mut pixels: Vec<[f32; 4]> = (0..4096)
                .map(|i| [0.18, 0.18, 0.18, f32::from_bits(alpha_bits[i % 4])])
                .collect();
            transform.display_scratch(&mut pixels, &mut scratch, 0.0, 1.0, 1.0).unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                assert_eq!(pixel[3].to_bits(), alpha_bits[i % 4]);
                assert_eq!(crate::transfer::code8(pixel[0]), 89);
            }
            assert_eq!(scratch.as_ptr(), allocation);
            assert_eq!(scratch.capacity(), capacity);
        }
    }

    #[test]
    fn short_alpha_scratch_refuses_before_changing_pixels() {
        let ocio = Ocio::load("ocio://default").unwrap();
        let transform = ocio.transform(&ocio.resolve(&Sel::default(), false).unwrap(), false).unwrap();
        let mut pixels = [[0.18, 0.5, 1.0, f32::from_bits(0x7fc0_4567)]];
        let before = pixels.map(|p| p.map(f32::to_bits));
        assert!(transform.display_scratch(&mut pixels, &mut [], 1.0, 1.2, 1.0).is_err());
        assert_eq!(pixels.map(|p| p.map(f32::to_bits)), before);
    }
}
