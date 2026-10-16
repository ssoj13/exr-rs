//! OpenFX effects in the viewer (feature `ofx`): one effect applied to the shown image, on ofx-rs's
//! shared application layer (`ofx-runtime`) and UI (`ofx-host-egui`) - the same layer Playa and the
//! ofx-rs viewer use, so this file is only what exr-view itself decides.
//!
//! ```text
//! OfxEffects::new()  Runtime<u32> (host "exr-view": overlays, custom interacts, pages; the built-in
//!                    plug-ins of the standard OpenFX search: OFX_PLUGIN_PATH, then the OS plug-in dir)
//!                    + UiEvents as observer
//! Effects tab        ofx-host-egui's EffectStack of one row: the "+" catalog menu (Filter / General /
//!                    Generator plug-ins, grouped) -> one Effect; enable, remove, the site picker
//!                    (ofx-runtime's SiteMenu: Default / Auto / CPU, every GPU site refused with its
//!                    reason - this host opens no GPU site), the status "CPU images, WGPU compute"
//!                    (the delivery site and what the plug-in reported computing with, read from the
//!                    effect after each completed render) and, in the body, its EffectUi panel
//!                    (parameters, parameter interacts) and errors
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

use ofx_host_egui::{
    Backend, CanonicalView, EffectStack, EffectUi, MessageKind, StackAction, StackRow, StackStatus,
    UiEvents, backend_options, plugin_menu,
};
use ofx_runtime::ofx_host::{HostContext, HostInfo, RenderSite};
use ofx_runtime::{
    Effect, EffectSetup, PluginKey, PluginSummary, Runtime, RuntimeError, ScanSpec, SiteMenu,
    Target,
};
use crate::image::{DecodedFrame, ImageStage, Selection};
use crate::tiles::effect::EffectSource;
use crate::tiles::effect_frames::Inputs;
use crate::tiles::region::RegionBudget;
use crate::tiles::source::TileSource;

/// Where users get ofx-rs's effects bundle.
const RELEASES_URL: &str = "https://github.com/ssoj13/ofx-rs/releases/latest";

/// The operating system's OpenFX plug-in directory (the last entry of the standard search).
#[cfg(windows)]
const OS_PLUGIN_DIR: &str = r"C:\Program Files\Common Files\OFX\Plugins";
#[cfg(target_os = "macos")]
const OS_PLUGIN_DIR: &str = "/Library/OFX/Plugins";
#[cfg(not(any(windows, target_os = "macos")))]
const OS_PLUGIN_DIR: &str = "/usr/OFX/Plugins";

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
    fn source(self, frame: Arc<DecodedFrame>, view: Selection, decoder: Arc<crate::ocio::Decoder>, budget: RegionBudget)
        -> Result<Arc<dyn TileSource>, String> {
        EffectSource::new(self.effect, self.inputs, frame, view, decoder, self.time,
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
    /// Whether the effect's row is folded (its parameters hidden).
    collapsed: bool,
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
    /// The dialog behind the parameter panel's browse buttons; installed in every attached UI.
    /// The panel's `PathPicker`: a mailbox the viewer answers from its one file dialog (`file_pick`).
    adapter: Arc<crate::file_pick::OfxPicker>,
    /// Why the last render or selection failed.
    error: Option<String>,
    /// Why the effect cannot run on the shown image at all (the viewer's
    /// [`ImageStage::set_blocked`]): shown in the panel, cleared when it can.
    blocked: Option<String>,
}

impl OfxEffects {
    /// Start the host and scan for plug-ins like any OpenFX application: `OFX_PLUGIN_PATH`, then the
    /// operating system's plug-in directory. The built-in effects of ofx-rs are an ordinary
    /// installed bundle (`ofx-effects.ofx.bundle`), not linked in.
    pub fn new() -> Self {
        Self::with_scan(ScanSpec::standard())
    }

    /// Tests only: the crate's linked fixture plug-ins as a scan with no search directory.
    ///
    /// The host enumerates a linked binary once, at its first open (ofxCore.h
    /// `OfxGetNumberOfPlugins`): a type registered later is missing for the rest of the process.
    /// So every fixture of the crate is registered here, before the first open, whichever test
    /// gets here first.
    #[cfg(test)]
    pub(crate) fn test_scan() -> ScanSpec {
        Self::fixtures(ScanSpec::new())
    }

    #[cfg(test)]
    fn fixtures(base: ScanSpec) -> ScanSpec {
        static REGISTER: std::sync::Once = std::sync::Once::new();
        REGISTER.call_once(|| {
            crate::image::consumer_tests::register_linked();
            crate::tiles::effect_tests::register_linked();
        });
        let points = ofx::linked::entry_points();
        // SAFETY: the discovery functions of the process's own plug-in registry (`ofx::linked`),
        // valid and unchanged for the life of the process (ofxCore.h:611-635).
        let fixtures = unsafe {
            ofx_runtime::ofx_host::LinkedBinary::new(
                "exr-view-fixtures",
                points.get_number_of_plugins,
                points.get_plugin,
                Some(points.set_host),
            )
        };
        match fixtures {
            Ok(linked) => base.with_linked(linked),
            Err(error) => panic!("the fixture registry is not a linked binary: {error:#}"),
        }
    }

    /// Tests only: the fixtures plus the standard OpenFX search, for tests that need a real effect
    /// (ofx-rs's Basic and Gaussian Blur, from the installed `ofx-effects.ofx.bundle`). A missing
    /// bundle FAILS the test loudly, never skips: deploy it from ofx-rs with
    /// `python bootstrap.py u -o "<OFX plug-in dir>" -f` (or put it on `OFX_PLUGIN_PATH`).
    #[cfg(test)]
    pub(crate) fn bundle_scan() -> ScanSpec {
        let probe = Runtime::<u32>::new(Self::host_info(), Self::fixtures(ScanSpec::standard()));
        for id in ["net.ofxrs.basic_gain", "net.ofxrs.gaussian_blur"] {
            assert!(
                probe.catalog().get(&PluginKey::new(id, 1)).is_some(),
                "OpenFX plug-in {id} not found: install ofx-rs's ofx-effects.ofx.bundle with \
                 `python bootstrap.py u -o \"<OFX plug-in dir>\" -f` in ofx-rs, or set OFX_PLUGIN_PATH; \
                 scan problems: {:?}",
                probe.catalog().problems()
            );
        }
        Self::fixtures(ScanSpec::standard())
    }

    /// Start the host with the plug-ins `scan` finds.
    pub fn with_scan(scan: ScanSpec) -> Self {
        let info = Self::host_info();
        let runtime = Runtime::new(info, scan);
        let events = Arc::new(UiEvents::new());
        runtime.inbox().set_observer(Some(events.clone()));
        let adapter = Arc::new(crate::file_pick::OfxPicker::default());
        Self {
            ui: None,
            effect: None,
            runtime,
            events,
            chosen: None,
            enabled: true,
            collapsed: false,
            time: 0.0,
            revision: 0,
            inputs: None,
            input_identity: None,
            input_error: None,
            accepted: None,
            frame_rate: None,
            frame_range: None,
            wake: None,
            adapter,
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

    /// The mailbox of the parameter panel's browse dialogs: the app turns a request into its one
    /// file dialog and posts the answer back.
    pub fn picker(&self) -> &crate::file_pick::OfxPicker {
        &self.adapter
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

    /// Where the effect's last COMPLETED render ran and what computed it, in ofx-runtime's words
    /// ([`ofx_runtime::RenderedOn`]: "CPU images, WGPU compute"). The viewer renders on the CPU site only (its
    /// host opens no GPU site), so the delivery is always the CPU's; the compute backend is what the plug-in
    /// reported - the host settles the report when each render ends, so this is never an earlier render's.
    /// `None` before the first render; `Err` when the report is not in the vocabulary (shown, not guessed).
    fn rendered(&self) -> Option<Result<String, String>> {
        match self.effect.as_ref()?.rendered_on(RenderSite::Cpu) {
            Ok(on) => Some(Ok(on.to_string())),
            Err(RuntimeError::NoInstance) => None,
            Err(error) => Some(Err(error.to_string())),
        }
    }

    /// The effect takes picker entry `option` of `menu` (an index into [`SiteMenu::options`]; `choice` `None`:
    /// follow the runtime's default). A refused entry is never taken - the stack disables it with its reason.
    fn pick_site(&mut self, menu: &SiteMenu, option: usize) {
        let (Some(effect), Some(entry)) = (&self.effect, menu.options.get(option)) else {
            return;
        };
        if entry.refusal.is_some() {
            return;
        }
        effect.set_site_choice(entry.choice);
        self.revision += 1;
    }

    /// The Effects tab: an [`EffectStack`] of the one effect (header: enable, name, site picker and where it last
    /// rendered, remove; "+" picks a plug-in), the status line, then its parameters.
    pub fn panel(&mut self, ui: &mut egui::Ui) {
        let mut chosen = None;
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
        let name = self
            .chosen
            .as_ref()
            .and_then(|key| self.runtime.catalog().get(key))
            .map_or_else(String::new, |plugin| plugin.short_label.clone());
        // The picker is the runtime's one site menu; the status reads the effect, not a copy kept here.
        let menu = self.effect.as_ref().map(|effect| effect.site_menu());
        let options = menu.as_ref().map(backend_options);
        let on = self.rendered();
        let rows: Vec<StackRow<'_>> = match (&menu, &options) {
            (Some(menu), Some(options)) => vec![StackRow {
                key: u64::from(EFFECT),
                label: &name,
                enabled: self.enabled,
                collapsed: self.collapsed,
                backend: Some(Backend {
                    options,
                    selected: menu.selected,
                    tip: Some(
                        "Image delivery site. The plug-in's Compute Backend parameter chooses its compute engine.",
                    ),
                }),
                status: match &on {
                    Some(Ok(text)) if error.is_none() => Some(StackStatus::Ok(text)),
                    Some(Err(text)) => Some(StackStatus::Error(text)),
                    _ => None,
                },
            }],
            _ => Vec::new(),
        };
        let mut add = |ui: &mut egui::Ui| {
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
        };
        let mut changed = false;
        let actions = egui::ScrollArea::vertical()
            .show(ui, |ui| {
                EffectStack::new(&rows)
                    .title("Effect")
                    .add_menu(&mut add)
                    .show(ui, |ui, _| match (&mut self.ui, &self.effect) {
                        (Some(effect_ui), Some(effect)) => {
                            changed = effect_ui.panel(ui, effect, revision, time).changed();
                        }
                        (None, Some(_)) => {
                            ui.label("The parameters show once the effect rendered the image.");
                        }
                        _ => {}
                    })
            })
            .inner;
        if self.effect.is_none() {
            let problems = self.runtime.catalog().problems();
            if self.runtime.catalog().plugins().is_empty() {
                ui.label(format!(
                    "No OpenFX plug-ins found. Searched: the OFX_PLUGIN_PATH directories, then {OS_PLUGIN_DIR}."
                ));
                ui.hyperlink_to(RELEASES_URL, RELEASES_URL);
                ui.label(format!(
                    "Download the ofx-rs effects bundle (ofx-effects.ofx.bundle) and unzip it into {OS_PLUGIN_DIR},                      or point OFX_PLUGIN_PATH at its folder."
                ));
            }
            if !problems.is_empty() {
                ui.collapsing(format!("{} scan problem(s)", problems.len()), |ui| {
                    for problem in problems {
                        ui.label(problem);
                    }
                });
            }
        }
        if changed {
            self.revision += 1;
        }
        for action in actions {
            match action {
                StackAction::Enable(_, enabled) => {
                    self.enabled = enabled;
                    self.revision += 1;
                }
                StackAction::Collapse(_, collapsed) => self.collapsed = collapsed,
                StackAction::Remove(_) => chosen = Some(None),
                StackAction::Backend { option, .. } => {
                    if let Some(menu) = &menu {
                        self.pick_site(menu, option);
                    }
                }
                // One effect: nothing to reorder (the stack offers no move for a single row).
                StackAction::Move { .. } => {}
            }
        }
        if let Some(key) = chosen {
            self.choose(key);
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
        // An explicit site this host cannot honour is an error to show, never a render somewhere else.
        if let Target::Unavailable(why) = effect.target() {
            return Err(why);
        }
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
                    // The path parameters' browse buttons open the viewer's one file dialog.
                    self.ui = Some(ui.with_picker(self.adapter.clone()));
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
    fn worker(&self, frame: Arc<DecodedFrame>, view: Selection, decoder: Arc<crate::ocio::Decoder>, budget: RegionBudget)
        -> Result<Option<Arc<dyn TileSource>>, String> {
        if !self.active() { return Ok(None); }
        self.snapshot(frame.clone())?.source(frame, view, decoder, budget).map(Some)
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
        // The installed ofx-effects bundle (and the crate's fixtures).
        let mut fx = OfxEffects::with_scan(OfxEffects::bundle_scan());
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
        let source = fx.worker(frame, Selection::default(), std::sync::Arc::new(crate::ocio::Decoder::builtin_only()), budget.clone()).unwrap().unwrap();
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

    // -----------------------------------------------------------------------------------------
    // The site picker and the render status (ofx-rs plan0 5.11): the runtime's SiteMenu through
    // ofx-host-egui's EffectStack, the status read from the effect after each completed render.
    // -----------------------------------------------------------------------------------------

    use egui::accesskit::Role;
    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT as _, Queryable as _};
    use ofx_runtime::SiteChoice;

    /// The Effects tab of `fx`, drawn headless.
    fn tab(fx: OfxEffects) -> Harness<'static, OfxEffects> {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(700.0, 500.0))
            .build_ui_state(|ui, fx: &mut OfxEffects| fx.panel(ui), fx);
        harness.run();
        harness
    }

    fn with_basic() -> OfxEffects {
        let mut fx = OfxEffects::with_scan(OfxEffects::bundle_scan());
        fx.choose(Some(PluginKey::new("net.ofxrs.basic_gain", 1)));
        assert!(fx.effect.is_some(), "{:?}", fx.error);
        fx
    }

    /// One tile of a 2x2 image through the effect: a completed render on a worker thread.
    fn render_once(fx: &OfxEffects) {
        let mut frame = crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
        frame.width = 2;
        frame.height = 2;
        frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(1, 1));
        frame.display_window = frame.data_window;
        for (_, channel) in frame.channels.iter_mut() {
            *channel = exr_core::api::ChannelData::Float(vec![0.5; 4]);
        }
        let source = fx
            .worker(Arc::new(frame), Selection::default(), std::sync::Arc::new(crate::ocio::Decoder::builtin_only()), RegionBudget::new(4096))
            .expect("the worker")
            .expect("an active effect");
        let grid = source.grid();
        let id = crate::tiles::grid::TileId { level: 0, tx: 0, ty: 0 };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let produced = pool.install(|| {
            source.produce(&grid, source.unit(id), &crate::tiles::source::ProductionContext::default())
        });
        assert!(
            matches!(produced, crate::tiles::source::Production::Processed(_)),
            "the tile was rendered"
        );
    }

    /// The picker is the runtime's menu: Default first, every site listed, the GPU ones disabled (this host
    /// opens no GPU site) and a click on one changes nothing; CPU is taken and re-renders. RED: a hand-made
    /// option list, or a refused entry that sets the choice.
    #[test]
    fn the_site_picker_is_the_runtime_menu_and_takes_only_what_is_open() {
        let fx = with_basic();
        let mut harness = tab(fx);
        let picker = |h: &Harness<'static, OfxEffects>| {
            h.get_all_by_role(Role::ComboBox).next().expect("site picker").value()
        };
        assert_eq!(picker(&harness).as_deref(), Some("Default (Auto)"));
        let revision = harness.state().revision;
        harness.get_all_by_role(Role::ComboBox).next().unwrap().click();
        harness.run();
        for refused in ["Vulkan", "CUDA", "OpenCL Buffer", "OpenCL Image", "OpenGL"] {
            assert!(
                harness.get_by_label(refused).accesskit_node().is_disabled(),
                "{refused} cannot be taken: no GPU site is open"
            );
        }
        harness.get_by_label("CUDA").click();
        harness.run();
        let effect = harness.state().effect.clone().expect("effect");
        assert_eq!(effect.site_override(), None);
        assert_eq!(harness.state().revision, revision, "a refused entry re-renders nothing");
        // The click on a disabled entry may close the popup: open it again if so.
        if harness.query_by_label("CPU").is_none() {
            harness.get_all_by_role(Role::ComboBox).next().unwrap().click();
            harness.run();
        }
        harness.get_by_label("CPU").click();
        harness.run();
        assert_eq!(effect.site_override(), Some(SiteChoice::Cpu));
        assert!(harness.state().revision > revision, "the pick re-renders");
        assert_eq!(picker(&harness).as_deref(), Some("CPU"));
    }

    /// A saved site name this build does not know stays the selected entry, refused, and the effect is an
    /// explicit error naming it - never rendered on the default. RED: unknown text mapped to Default.
    #[test]
    fn an_unknown_saved_site_is_shown_refused_and_not_rendered() {
        let fx = with_basic();
        fx.effect.as_ref().expect("effect").set_saved_site(Some("metal-2"));
        let error = fx
            .worker(
                Arc::new(crate::image::tests_support::frame_with(&["R", "G", "B", "A"])),
                Selection::default(), std::sync::Arc::new(crate::ocio::Decoder::builtin_only()),
                RegionBudget::new(4096),
            )
            .err()
            .expect("an unavailable site is an error");
        assert!(error.contains("metal-2"), "{error}");
        let harness = tab(fx);
        let picker = harness.get_all_by_role(Role::ComboBox).next().expect("site picker");
        assert_eq!(picker.value().as_deref(), Some("metal-2 (unknown)"));
    }

    /// The status names the delivery site and the compute the plug-in reported, from the effect, after a
    /// completed render; before any render and for a new effect there is nothing - no stale value.
    #[test]
    fn the_status_names_the_delivery_and_the_compute_and_never_a_stale_one() {
        let fx = with_basic();
        assert_eq!(fx.rendered(), None, "no render yet: nothing to say");
        render_once(&fx);
        let on = fx.rendered().expect("a render completed").expect("a valid report");
        assert!(
            on == "CPU images, CPU compute" || on == "CPU images, WGPU compute",
            "{on}"
        );
        let mut harness = tab(fx);
        assert!(
            harness.get_all_by_label_contains(&on).count() >= 1,
            "the tab shows {on}"
        );
        // Another effect has rendered nothing: the old report must not be shown for it.
        harness
            .state_mut()
            .choose(Some(PluginKey::new("net.ofxrs.basic_gain", 1)));
        harness.run();
        assert_eq!(harness.state().rendered(), None);
        assert_eq!(harness.query_all_by_label_contains("images").count(), 0);
    }

    /// The browse button of a path parameter reaches the viewer's picker mailbox (what `app.rs` turns into
    /// its one file dialog), and the path the viewer answers with lands in the parameter (the preset file
    /// every ofx-crate plug-in declares). RED: no picker installed = the click only reports "the application
    /// has installed no file dialog".
    #[test]
    fn a_browse_click_reaches_the_picker_and_the_pick_lands_in_the_parameter() {
        let mut fx = OfxEffects::with_scan(OfxEffects::test_scan());
        fx.choose(Some(PluginKey::new(crate::image::consumer_tests::EFFECT_ID, 1)));
        render_once(&fx);
        fx.attach_ui();
        // As the viewer's frame does: tick (finished edits, redraws), then the panel.
        let mut harness = Harness::builder()
            .with_size(egui::vec2(700.0, 500.0))
            .build_ui_state(|ui, fx: &mut OfxEffects| { fx.tick(ui.ctx()); fx.panel(ui) }, fx);
        harness.run_steps(3);
        eprintln!("DBGTREE {:#?}", harness.root());
        // The "Presets" group is closed by default: open it, then press the Preset File browse icon.
        harness.get_by_label("Presets").click();
        harness.run_steps(3);
        harness
            .get_all_by_role_and_label(Role::Button, egui_widgets_config::icons::FILE)
            .next()
            .expect("a browse button")
            .click();
        // The viewer answers the request like `app.rs` does once its dialog confirms; the edit goes to the
        // instance as an event and comes back on a later frame.
        let mut asked = Vec::new();
        for _ in 0..200 {
            harness.step();
            if let Some(request) = harness.state().picker().take_request() {
                asked.push(request);
                harness.state().picker().answer(ofx_host_egui::PickOutcome::Picked("picked.json".into()));
            }
            if harness.query_all_by_value("picked.json").next().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(asked.len(), 1, "one dialog was asked for");
        assert_eq!(asked[0].kind, ofx_host_egui::PickKind::Save, "the preset file is a path that need not exist");
        assert!(
            harness.query_all_by_value("picked.json").next().is_some(),
            "the picked path is the parameter's value; text inputs: {:?}",
            harness.get_all_by_role(Role::TextInput).map(|n| n.accesskit_node().value()).collect::<Vec<_>>()
        );
    }
}
