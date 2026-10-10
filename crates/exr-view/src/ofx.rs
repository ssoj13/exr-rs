//! OpenFX effects in the viewer (feature `ofx`): one effect applied to the shown image, on ofx-rs's
//! shared application layer (`ofx-runtime`) and UI (`ofx-host-egui`) - the same layer Playa and the
//! ofx-rs viewer use, so this file is only what exr-view itself decides.
//!
//! ```text
//! OfxEffects::new()  Runtime<u32> (host "exr-view": overlays, custom interacts, pages; the built-in
//!                    effects, then OFX_PLUGIN_PATH and the OS plug-in dirs) + UiEvents as observer
//! Effects tab        catalog menu (Filter / General / Generator plug-ins, grouped) -> one Effect,
//!                    its EffectUi panel (parameters, parameter interacts), enable, errors
//! viewport           ImageStage::process: the materialized linear RGBA (top row first) as the
//!                    Source (copied once into a host frame, negative row stride = OFX bottom-left
//!                    origin), the Output a lent Vec<f32> -> uploaded instead of the decoded pixels;
//!                    ImageStage::overlay: the effect's overlay over the image (it takes the LMB)
//! every UI frame     tick: questions, finished edits, dialogs, plug-in writes / redraws -> re-render
//! ```
//!
//! The image is premultiplied linear RGBA (the EXR convention); a Generator makes an image of the
//! shown one's size. Frames at other times are not offered (no `FrameInput`): a temporal plug-in
//! sees missing images there.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ofx_host_egui::{CanonicalView, EffectUi, MessageKind, UiEvents, plugin_menu};
use ofx_runtime::ofx_host::{HostContext, HostInfo};
use ofx_runtime::{Effect, EffectSetup, PluginKey, PluginSummary, Runtime, ScanSpec};
use crate::image::{DecodedFrame, ImageStage, Selection};
use crate::tiles::effect::EffectSource;
use crate::tiles::effect_frames::Inputs;
use crate::tiles::region::RegionBudget;
use crate::tiles::source::TileSource;

/// The runtime key of the viewer's one effect.
const EFFECT: u32 = 0;

/// The contexts an effect over the shown image may take, in preference order.
const CONTEXTS: [HostContext; 3] = [
    HostContext::Filter,
    HostContext::General,
    HostContext::Generator,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InputIdentity {
    pub(crate) primary: u64,
    pub(crate) index: i32,
    pub(crate) source_epoch: u64,
    pub(crate) revision: u64,
    pub(crate) rate_bits: u64,
    pub(crate) range: (i32, i32),
}
/// Immutable facts for one source generation, shared only with tile workers.
pub(crate) struct OfxEffectsSnapshot {
    effect: Arc<Effect<u32>>,
    inputs: Arc<Inputs>,
    time: f64,
    frame_rate: Option<f64>,
    frame_range: Option<[f64; 2]>,
}
impl OfxEffectsSnapshot {
    fn source(self, frame: Arc<DecodedFrame>, view: Selection, budget: RegionBudget)
        -> Result<Arc<dyn TileSource>, String> {
        EffectSource::new(self.effect, self.inputs, frame, view, self.time,
            self.frame_rate, self.frame_range, budget)
            .map(|source| Arc::new(source) as Arc<dyn TileSource>)
            .map_err(|error| error.to_string())
    }
}
/// The viewer's OpenFX effect (see the module doc).
pub struct OfxEffects {
    // `ui` before `effect` and `runtime`: the instance's interacts go first.
    ui: Option<EffectUi>,
    effect: Option<Arc<Effect<u32>>>,
    runtime: Arc<Runtime<u32>>,
    events: Arc<UiEvents>,
    chosen: Option<PluginKey>,
    enabled: bool,
    /// The time the effect renders at (the sequence frame; 0 for a single image).
    time: f64,
    /// Advanced whenever the effect's output may change (an edit, a plug-in write, a redraw).
    revision: u64,
    inputs: Option<Arc<Inputs>>,
    input_identity: Option<InputIdentity>,
    input_error: Option<String>,
    accepted: Option<crate::tiles::view::AcceptedEffect>,
    frame_rate: Option<f64>,
    frame_range: Option<[f64; 2]>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Why the last render or selection failed.
    error: Option<String>,
    /// Why the effect cannot run on the shown image at all (the viewer's
    /// [`ImageStage::set_blocked`]): shown in the panel, cleared when it can.
    blocked: Option<String>,
}

impl OfxEffects {
    /// Start the host and scan for plug-ins: the built-in effects, then `OFX_PLUGIN_PATH` and the
    /// operating system's plug-in directories.
    pub fn new() -> Self {
        Self::with_scan(Self::with_builtins(ScanSpec::standard()))
    }

    /// `base` plus the built-in effects: THE one place this crate opens the process's linked
    /// plug-in registry (`ofx::linked`, which the built-in effects' binary is).
    ///
    /// The host enumerates that binary once, at its first open (ofxCore.h
    /// `OfxGetNumberOfPlugins`): a type registered later is missing for the rest of the process.
    /// A test build therefore registers every test fixture of the crate here, before the first
    /// open, whichever test (or `AppState::new` in an app test) gets here first; before this,
    /// fixtures registered per module came out `PluginMissing` depending on test order.
    fn with_builtins(base: ScanSpec) -> ScanSpec {
        #[cfg(test)]
        {
            static REGISTER: std::sync::Once = std::sync::Once::new();
            REGISTER.call_once(|| {
                crate::image::consumer_tests::register_linked();
                crate::tiles::effect_tests::register_linked();
            });
        }
        base.with_builtin_effects()
    }

    /// Tests only: the built-in effects and the crate's fixtures, no installed plug-ins.
    #[cfg(test)]
    pub(crate) fn test_scan() -> ScanSpec {
        Self::with_builtins(ScanSpec::new())
    }

    /// Start the host with the plug-ins `scan` finds.
    pub fn with_scan(scan: ScanSpec) -> Self {
        let info = Self::host_info();
        let runtime = Runtime::new(info, scan);
        let events = Arc::new(UiEvents::new());
        runtime.inbox().set_observer(Some(events.clone()));
        Self {
            ui: None,
            effect: None,
            runtime,
            events,
            chosen: None,
            enabled: true,
            time: 0.0,
            revision: 0,
            inputs: None,
            input_identity: None,
            input_error: None,
            accepted: None,
            frame_rate: None,
            frame_range: None,
            wake: None,
            error: None,
            blocked: None,
        }
    }

    pub(crate) fn host_info() -> HostInfo {
        let info = HostInfo::cpu_only()
            .with_overlays(true)
            .with_custom_interacts(true)
            .with_paged_parameters(true)
            .with_background(false);
        match info.clone().with_identity(
            "com.ssoj13.exr-view",
            "exr-view",
            [0, 1, 0],
            env!("CARGO_PKG_VERSION"),
        ) {
            Ok(named) => named,
            Err(error) => {
                log::warn!("OpenFX host identity refused ({error:?}); the default stays");
                info
            }
        }

    }

    /// Whether an effect is applied to the shown image.
    pub fn active(&self) -> bool {
        self.enabled && self.effect.is_some()
    }

    /// The effect renders at `time` from now on.
    pub fn set_time(&mut self, time: f64) {
        if self.time != time {
            self.time = time;
            self.revision += 1;
        }
    }

    /// Once per UI frame: the plug-in's questions (windows), finished edits, dialogs (on this, the
    /// UI thread), and whatever makes the effect render again (its writes, its redraw requests).
    pub fn tick(&mut self, ctx: &egui::Context) {
        let repaint = ctx.clone();
        self.wake = Some(Arc::new(move || repaint.request_repaint()));
        self.attach_ui();
        self.events.show(ctx);
        let finished = self.runtime.finish_events();
        self.runtime.run_dialogs();
        let inbox = self.runtime.inbox();
        let changed = !inbox.take_edits().is_empty() || !inbox.take_redraws().is_empty();
        for (_, time) in inbox.take_times() {
            log::debug!("OpenFX effect asked for time {time}; the viewer's playhead stays");
        }
        if finished > 0 || changed {
            self.revision += 1;
            ctx.request_repaint();
        }
    }

    /// The Effects tab: pick a plug-in, enable it, its parameters.
    pub fn panel(&mut self, ui: &mut egui::Ui) {
        let mut chosen = None;
        let current = self
            .chosen
            .as_ref()
            .and_then(|key| self.runtime.catalog().get(key))
            .map_or("None", |plugin| plugin.short_label.as_str());
        ui.horizontal(|ui| {
            ui.menu_button(format!("Effect: {current}"), |ui| {
                if ui.button("None").clicked() {
                    chosen = Some(None);
                    ui.close();
                }
                let usable: Vec<&PluginSummary> = self
                    .runtime
                    .catalog()
                    .plugins()
                    .iter()
                    .filter(|plugin| CONTEXTS.iter().any(|context| plugin.supports(*context)))
                    .collect();
                if let Some(key) = plugin_menu(ui, &usable, self.chosen.as_ref()) {
                    chosen = Some(Some(key));
                }
            });
            if ui.checkbox(&mut self.enabled, "enabled").changed() {
                self.revision += 1;
            }
        });
        if let Some(key) = chosen {
            self.choose(key);
        }
        // The blocked note, the error and the plug-in's messages live in the log docked at the bottom of the tab
        // (drawn first: egui panels claim their edge before the controls), so a failure never shifts the controls.
        let error = self
            .error
            .clone()
            .or_else(|| self.effect.as_ref().and_then(|effect| effect.error()));
        let notes: Vec<(MessageKind, &str)> = self
            .blocked
            .as_deref()
            .map(|text| (MessageKind::Warning, text))
            .into_iter()
            .chain(error.as_deref().map(|text| (MessageKind::Error, text)))
            .collect();
        self.events.log_ui(ui, &notes);
        ui.separator();
        let revision = self.runtime.inbox().revision();
        let time = self.time;
        match (&mut self.ui, &self.effect) {
            (Some(effect_ui), Some(effect)) => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if effect_ui.panel(ui, effect, revision, time).changed() {
                        self.revision += 1;
                    }
                });
            }
            (None, Some(_)) => {
                ui.label("The parameters show once the effect rendered the image.");
            }
            _ => {
                let problems = self.runtime.catalog().problems();
                if !problems.is_empty() {
                    ui.collapsing(format!("{} scan problem(s)", problems.len()), |ui| {
                        for problem in problems {
                            ui.label(problem);
                        }
                    });
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn choose_for_test(&mut self, key: PluginKey) { self.choose(Some(key)); }

    /// Use plug-in `key` (its first context among [`CONTEXTS`]), or none.
    fn choose(&mut self, key: Option<PluginKey>) {
        self.ui = None;
        self.effect = None;
        self.error = None;
        self.inputs = None;
        self.input_identity = None;
        self.runtime.forget_all();
        self.chosen = key.clone();
        self.revision += 1;
        let Some(key) = key else {
            return;
        };
        let described = self
            .runtime
            .catalog()
            .context_among(&key, &CONTEXTS)
            .and_then(|context| self.runtime.describe(&key, context));
        match described {
            Ok(described) => {
                self.effect = Some(
                    self.runtime
                        .effect(EFFECT, &described, EffectSetup::default()),
                );
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }

    pub(crate) fn generation_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.chosen.hash(&mut hasher);
        self.revision.hash(&mut hasher);
        self.time.to_bits().hash(&mut hasher);
        hasher.finish() | 1
    }
    pub(crate) fn inputs_match(&self, identity: InputIdentity) -> bool {
        self.input_identity == Some(identity)
    }
    pub(crate) fn set_inputs(&mut self, identity: InputIdentity, inputs: Arc<Inputs>,
        frame_rate: Option<f64>, frame_range: Option<[f64; 2]>) {
        self.input_error = None;
        self.input_identity = Some(identity);
        self.inputs = Some(inputs);
        self.frame_rate = frame_rate;
        self.frame_range = frame_range;
    }
    pub(crate) fn set_input_error(&mut self, error: String) {
        self.input_error = Some(error.clone());
        self.error = Some(error);
        self.inputs = None;
        self.input_identity = None;
    }
    fn snapshot(&self, primary: Arc<DecodedFrame>) -> Result<OfxEffectsSnapshot, String> {
        if let Some(error) = &self.input_error { return Err(error.clone()); }
        let effect = self.effect.clone().ok_or("no effect")?;
        let inputs = self.inputs.clone().unwrap_or_else(|| Arc::new(Inputs::Static {
            frame: primary, index: 0,
        }));
        Ok(OfxEffectsSnapshot { effect, inputs, time: self.time,
            frame_rate: self.frame_rate, frame_range: self.frame_range })
    }
    fn attach_ui(&mut self) {
        let Some(effect) = &self.effect else { return; };
        if self.ui.as_ref().is_some_and(|ui| ui.shows(effect)) { return; }
        if let Some(attached) = EffectUi::attach(effect) {
            match attached {
                Ok(ui) => {
                    for problem in ui.problems() { log::warn!("OpenFX effect UI: {problem}"); }
                    self.ui = Some(ui);
                }
                Err(error) => self.error = Some(format!("{error:#}")),
            }
        }
    }

}

impl Default for OfxEffects {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageStage for OfxEffects {
    fn key(&self) -> u64 {
        if !self.active() {
            return 0;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.chosen.hash(&mut hasher);
        self.revision.hash(&mut hasher);
        self.time.to_bits().hash(&mut hasher);
        self.input_identity.hash(&mut hasher);
        hasher.finish() | 1
    }

    fn set_blocked(&mut self, reason: Option<String>) {
        self.blocked = reason;
    }

    fn uses_worker(&self) -> bool { self.active() }
    fn worker(&self, frame: Arc<DecodedFrame>, view: Selection, budget: RegionBudget)
        -> Result<Option<Arc<dyn TileSource>>, String> {
        if !self.active() { return Ok(None); }
        self.snapshot(frame.clone())?.source(frame, view, budget).map(Some)
    }
    fn worker_wake(&self) -> Option<Arc<dyn Fn() + Send + Sync>> { self.wake.clone() }
    fn accepted(&mut self, shown: Option<crate::tiles::view::AcceptedEffect>) { self.accepted = shown; }

    fn takes_pointer(&self) -> bool {
        self.active()
            && self.accepted.is_some()
            && self
                .ui
                .as_ref()
                .is_some_and(|ui| ui.overlay_kind().is_some())
    }

    fn overlay(
        &mut self,
        painter: &egui::Painter,
        response: &egui::Response,
        image_rect: egui::Rect,
        _size: (u32, u32),
    ) -> bool {
        let Some(shown) = self.accepted else { return false; };
        if !self.active() {
            return false;
        }
        let (Some(ui), Some(effect)) = (&mut self.ui, &self.effect) else {
            return false;
        };
        let view = CanonicalView::new(image_rect, [0.0, 0.0, f64::from(shown.size.0) * shown.aspect, f64::from(shown.size.1)]);
        let quiet = effect.events_pending() == 0;
        match ui.overlay(painter, response, view, shown.time, None, quiet) {
            Ok(true) => {
                self.revision += 1;
                true
            }
            Ok(false) => false,
            Err(error) => {
                self.error = Some(format!("overlay: {error:#}"));
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in Basic gain chosen in the tab renders the shown pixels through the stage: at its
    /// default scale 1 the image comes back unchanged (rows top first, through the negative
    /// stride), the stage is active with a non-zero upload key, and the effect's UI is attached
    /// once the instance exists; "None" turns the stage off.
    #[test]
    fn a_builtin_effect_processes_the_shown_pixels() {
        // The built-in effects (and the crate's fixtures) only: an installed plug-in has no
        // business in a unit test.
        let mut fx = OfxEffects::with_scan(OfxEffects::test_scan());
        assert!(!fx.active());
        assert_eq!(fx.key(), 0);
        fx.choose(Some(PluginKey::new("net.ofxrs.basic_gain", 1)));
        assert!(fx.active(), "{:?}", fx.error);
        let key = fx.key();
        assert_ne!(key, 0);
        // 2x2: top row red / green, bottom row blue / white, alpha 1.
        let pixels: Vec<f32> = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        .concat();
        let mut frame = crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
        frame.width = 2; frame.height = 2;
        frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(1, 1));
        frame.display_window = frame.data_window;
        for (c, (_, channel)) in frame.channels.iter_mut().enumerate() {
            *channel = exr_core::api::ChannelData::Float(pixels.as_chunks::<4>().0.iter().map(|p| p[c]).collect());
        }
        let frame = Arc::new(frame);
        let budget = RegionBudget::new(4096);
        let source = fx.worker(frame, Selection::default(), budget.clone()).unwrap().unwrap();
        let grid = source.grid();
        let id = crate::tiles::grid::TileId { level: 0, tx: 0, ty: 0 };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let produced = pool.install(|| source.produce(&grid, source.unit(id),
            &crate::tiles::source::ProductionContext::default()));
        let crate::tiles::source::Production::Processed(work) = produced else { panic!("worker did not produce an admitted effect owner"); };
        let shown: Vec<_> = (0..2).flat_map(|y| {
            let work = &work;
            (0..2).flat_map(move |x| work.pixel(x, y).unwrap())
        }).collect();
        assert_eq!(shown, pixels);
        assert_eq!(budget.used(), 64);
        fx.attach_ui();
        assert!(fx.ui.is_some(), "the UI thread attaches after worker instance creation");
        fx.set_time(3.0);
        assert_ne!(fx.key(), key, "another time is another image");
        fx.choose(None);
        assert!(!fx.active());
        assert_eq!(fx.process(&pixels, 2, 2), pixels.as_slice());
    }
}
