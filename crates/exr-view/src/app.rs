//! Application shell: winit `ApplicationHandler` + egui-winit/egui-wgpu glue +
//! egui_dock layout + the flat GPU image viewport.
//!
//! A raw winit window drives a wgpu surface; egui renders each frame through
//! `egui-wgpu` into an `egui_dock` dock space. No eframe: the viewer owns its
//! wgpu device/surface directly, which is why the image backend's `present()`
//! takes device/queue/renderer as plain references rather than a `RenderState`.
//!
//! Layering:
//! - [`AppState`] — the loaded image + view controls + the `ui_*` panel bodies.
//! - [`Viewer`] — wgpu context + egui integration + the GPU colour pass + dock
//!   state; runs one frame and owns the load seam.
//! - [`App`] — winit `ApplicationHandler` (lazy window on `resumed`, pumps events).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use egui_attr_grid::{AttrField, AttrGridState, AttrValue};
use cryptomatte::{Manifest, RankedLayer, discover_path, extract_coverage_with, load_manifest};
use egui_dock::{DockArea, DockState, Style};
use egui_widgets_config::reset;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::dock::{self, DockTab, DockTabViewer, default_dock_state};
use crate::file_pick::{FilePick, Pick};
use crate::gpu_ctx::GpuContext;
use crate::channels;
use crate::commands::{self, Action, COMMANDS, NavCmd};
use crate::image::{ChannelDisplay, DecodedFrame, DisplayTarget, ImageBackend, Selection};
use crate::persist::{Budgets, Config};
use crate::playback::Playback;
use crate::scopes::{
    GpuHistogramPanel, GpuPanel, GpuVectorscopePanel, GpuWaveformPanel, PanelCtx, show_panel,
};
use scancache::Scancache;
use scanseq::{Scanner, Seq};

/// Events the viewer posts to its own event loop from another thread.
///
/// The frame cache decodes on a worker pool; `CacheProgress` is how a finished
/// decode asks the UI thread for exactly one repaint, instead of the UI polling
/// at 60 Hz while anything is in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserEvent {
    CacheProgress,
}

/// Build the frame cache's [`scancache::Wake`] from an event-loop proxy.
///
/// Public because only [`crate::run`] can make the proxy (winit exposes
/// `create_proxy` on `EventLoop`, not on the `ActiveEventLoop` of `resumed`).
///
/// `send_event` fails only once the loop has exited, which is exactly when there
/// is nothing left to wake — so the error is deliberately dropped here rather
/// than logged on every in-flight decode during shutdown.
pub fn proxy_wake(proxy: EventLoopProxy<UserEvent>) -> scancache::Wake {
    Arc::new(move || {
        if proxy.send_event(UserEvent::CacheProgress).is_err() {
            log::trace!("cache wake after the event loop exited");
        }
    })
}

/// Transport-timeline cache-fill tint: a frame already decoded + resident.
const CACHE_LOADED: egui::Color32 = egui::Color32::from_rgb(70, 150, 70);
/// Transport-timeline cache-fill tint: a frame queued / decoding (not yet resident).
const CACHE_INFLIGHT: egui::Color32 = egui::Color32::from_rgb(150, 130, 60);

/// Compact human-readable byte count for the status bar (e.g. `12 GB`).
fn humanize_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} {}", UNITS[u])
    } else {
        format!("{v:.0} {}", UNITS[u])
    }
}

use crate::theme;
use crate::viewport::{self, TabGpu, ViewportGpu};

/// System memory reserved (GiB) as a floor the cache may not cross — OS + GPU
/// driver + the application's own working set.
const CACHE_MEM_RESERVE_GB: f64 = 2.0;
/// Decode-pool width cap. DWAA decode is CPU + memory-bandwidth bound; past ~8
/// threads the gain flattens while the UI/wgpu threads need headroom.
const MAX_DECODE_THREADS: usize = 8;

/// Decode-pool thread count: `available_parallelism` clamped to `[2, MAX]`.
fn decode_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(2, MAX_DECODE_THREADS)
}

/// RAM byte budget: `min(installed * share, installed - CACHE_MEM_RESERVE_GB)` via
/// `scancache::Budget::from_installed_ram` (the playa formula), `share` from
/// `persist::Budgets::ram`. A property of the MACHINE, not of the moment (sizing off
/// *free* RAM makes a cache shrink its own allowance as it fills, and inherit
/// whatever else was running at launch). The frame cache and the tile cache each
/// get it: the tile cache holds one frame's pyramid (a new frame drops the old
/// generation) and a file-backed frame holds no pixels, so the two do not stack.
pub(crate) fn ram_budget(share: f64) -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let installed = sys.total_memory() as usize; // bytes (sysinfo >= 0.30)
    let budget = scancache::Budget::from_installed_ram(installed, share, CACHE_MEM_RESERVE_GB);
    log::info!(
        "RAM budget: {} MiB (installed {} MiB, reserve {} GiB, share {share})",
        budget >> 20,
        installed >> 20,
        CACHE_MEM_RESERVE_GB,
    );
    budget
}

/// Atlas budget when the driver reports no VRAM figure: conservative.
const TILE_VRAM_FALLBACK: u64 = 1 << 30;

/// Display controls fed to the GPU colour pass (wgpu-only — no CPU tonemap).
///
/// `exposure` is in stops (EV), `gamma` is an extra display gamma dial
/// (1.0 = identity), and `channel` selects RGBA / single-channel isolate.
/// `hovered_pixel` is written by the viewport each frame for the Pixel Info panel.
pub struct ViewControls {
    pub exposure: f32,
    /// The exposure the toolbar's bypass button parked (see `exposure_control`); persisted.
    pub exposure_hold: f32,
    pub gamma: f32,
    pub channel: ChannelDisplay,
    /// What is shown: a layer, or one channel alone. Shared by the toolbar combo
    /// and the channel tree; applied to the displayed backend each frame (like
    /// `channel`). Per-file, so it is reset on load and NOT persisted (a layer
    /// name from one file is meaningless for the next).
    pub view: Selection,
    /// Cryptomatte type name ("" = no crypto overlay). Per-file, not persisted.
    pub crypto_layer: String,
    /// Object/material name within the selected Cryptomatte type ("" = none).
    pub crypto_name: String,
    pub hovered_pixel: Option<(u32, u32)>,
    /// Tone operator of the colour image (`transfer::tonemap::*`); persisted.
    pub tonemap: u32,
    /// What shows under the image's alpha (the toolbar's drop-up); persisted.
    pub backdrop: egui_display::Backdrop,
    /// The OCIO display transform (the Colour panel); when active it replaces the
    /// tone operator and the sRGB encode of the colour image. Its choice is persisted.
    pub colour: crate::ocio::State,
    /// The window output as the colour pass needs it; set by the viewer each frame.
    pub target: crate::present::Target,
    /// A navigation asked (menu, key, toolbar), applied by the viewport to the shown
    /// image as soon as one is drawn (a sequence frame may still be decoding); a new
    /// file drops it (`reset_view_state`).
    pub nav: Option<NavCmd>,
}

impl Default for ViewControls {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            exposure_hold: 0.0,
            gamma: 1.0,
            channel: ChannelDisplay::Rgba,
            view: Selection::default(),
            crypto_layer: String::new(),
            crypto_name: String::new(),
            hovered_pixel: None,
            tonemap: crate::transfer::tonemap::NONE,
            backdrop: egui_display::Backdrop::default(),
            colour: crate::ocio::State::new(crate::ocio::Sel::default()),
            target: crate::present::Target::default(),
            nav: None,
        }
    }
}

impl ViewControls {
    /// The display transform of the colour image: `Some(op)` = the built-in tone
    /// operator `op` + sRGB, `None` = OCIO.
    pub fn display(&self) -> Option<u32> {
        (!self.colour.sel.on).then_some(self.tonemap)
    }

    /// Show the colour image through `d` (see [`Self::display`]).
    pub fn set_display(&mut self, d: Option<u32>) {
        match d {
            Some(op) => {
                self.tonemap = crate::transfer::tonemap::known(op);
                self.colour.set_on(false);
            }
            None => self.colour.set_on(true),
        }
    }

    /// The display transform after the current one (T): None, Reinhard, AgX, OCIO.
    pub fn next_display(&self) -> Option<u32> {
        use crate::transfer::tonemap;
        match self.display() {
            None => Some(tonemap::NONE),
            Some(op) if tonemap::next(op) == tonemap::NONE => None,
            Some(op) => Some(tonemap::next(op)),
        }
    }
}

/// Map a [`ChannelDisplay`] to the small integer the config stores (and back via
/// [`channel_from_u8`]). Kept here (not on the enum) so the ported image module
/// carries no persistence concern; the on-disk format owns this numbering.
fn channel_to_u8(c: ChannelDisplay) -> u8 {
    match c {
        ChannelDisplay::Rgba => 0,
        ChannelDisplay::R => 1,
        ChannelDisplay::G => 2,
        ChannelDisplay::B => 3,
        ChannelDisplay::A => 4,
        ChannelDisplay::Luminance => 5,
    }
}

/// Inverse of [`channel_to_u8`]; an unknown value falls back to `Rgba` so a
/// hand-edited or future config can never panic the viewer.
fn channel_from_u8(v: u8) -> ChannelDisplay {
    match v {
        1 => ChannelDisplay::R,
        2 => ChannelDisplay::G,
        3 => ChannelDisplay::B,
        4 => ChannelDisplay::A,
        5 => ChannelDisplay::Luminance,
        _ => ChannelDisplay::Rgba,
    }
}

/// Persistent view navigation, kept on [`AppState`] so it survives switching
/// frames in a sequence (each frame is a separate [`ImageBackend`] in the cache;
/// we copy this nav into whichever backend is displayed before drawing it).
#[derive(Debug, Clone)]
pub struct ViewNav {
    pub pan: egui::Vec2,
    pub zoom: f32,
    pub fit_zoom: f32,
    pub picked_pixel: Option<(u32, u32)>,
}

impl Default for ViewNav {
    fn default() -> Self {
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            fit_zoom: 0.0,
            picked_pixel: None,
        }
    }
}

impl ViewNav {
    /// Copy this nav into `img` before the viewport draws it.
    fn apply_to(&self, img: &mut ImageBackend) {
        img.pan = self.pan;
        img.zoom = self.zoom;
        img.fit_zoom = self.fit_zoom;
        img.picked_pixel = self.picked_pixel;
    }

    /// Read back the nav the viewport just mutated on `img`.
    fn read_from(&mut self, img: &ImageBackend) {
        self.pan = img.pan;
        self.zoom = img.zoom;
        self.fit_zoom = img.fit_zoom;
        self.picked_pixel = img.picked_pixel;
    }
}

/// One Cryptomatte type discovered for the currently open file (viewer pick UI).
#[derive(Clone, Debug)]
struct CryptoLayerCatalog {
    name: String,
    ranked: Vec<RankedLayer>,
    /// Sorted object/material names for the combo.
    names: Vec<String>,
    /// The full `Manifest`, not a bare
    /// `by_name` `HashMap` — `Manifest::id_for_name` is the canonical
    /// name->id resolver (manifest hit, else hash fallback; the hash IS the
    /// id definition per spec 1.2.0 p.3), so the viewer must not reimplement
    /// that fallback against a stripped-down map.
    manifest: Manifest,
}

/// Cryptomatte catalog for the open file (header discover + manifests).
#[derive(Clone, Debug, Default)]
struct CryptoCatalog {
    layers: Vec<CryptoLayerCatalog>,
}

impl CryptoCatalog {
    fn discover(path: &std::path::Path) -> Option<Self> {
        let layers = match discover_path(path) {
            Ok(l) if !l.is_empty() => l,
            Ok(_) => return None,
            Err(e) => {
                log::debug!("cryptomatte discover skipped for {}: {e}", path.display());
                return None;
            }
        };
        let mut out = Vec::with_capacity(layers.len());
        for layer in layers {
            let manif = load_manifest(
                path,
                layer.manif_file.as_deref(),
                layer.inline_manifest.as_deref(),
            )
            .unwrap_or_else(|e| {
                log::warn!(
                    "cryptomatte: layer `{}` manifest unavailable ({e}); \
                     resolving names by hash",
                    layer.name
                );
                Manifest::default()
            });
            let mut names: Vec<String> = manif.by_name.keys().cloned().collect();
            names.sort_unstable();
            out.push(CryptoLayerCatalog {
                name: layer.name,
                ranked: layer.ranked,
                names,
                manifest: manif,
            });
        }
        if out.is_empty() {
            None
        } else {
            Some(Self { layers: out })
        }
    }

    fn layer(&self, name: &str) -> Option<&CryptoLayerCatalog> {
        self.layers.iter().find(|l| l.name == name)
    }
}

/// What a background open produced. Everything expensive — the directory scan,
/// `scanseq` detection, the EXR decode and the Cryptomatte manifest parse — runs
/// on a worker thread and arrives here; the UI thread only installs the result.
enum OpenOutcome {
    /// A `name.####.exr` set was detected (more than one frame present).
    Sequence {
        /// The path the user asked for (what goes into recents / `last_file`).
        requested: PathBuf,
        sequence: Seq,
        /// Every present frame's modification time at the scan (the reload baseline).
        stamps: Stamps,
        crypto: Option<CryptoCatalog>,
    },
    /// A single image, already decoded.
    Single {
        path: PathBuf,
        frame: Box<DecodedFrame>,
        crypto: Option<CryptoCatalog>,
    },
    /// Nothing could be opened. The current view is KEPT — a failed open used to
    /// clear the loaded image first and leave an empty viewport behind.
    Failed { path: PathBuf, reason: String },
}

/// The whole open pipeline, off the UI thread. Pure: it touches no viewer state,
/// so it can be called from a worker and its result applied later.
fn open_blocking(path: PathBuf, detect_sequence: bool, backing: u64) -> OpenOutcome {
    log::info!(
        "open: {} (detect_sequence={detect_sequence})",
        path.display()
    );

    // A directory argument: pick a representative frame inside it to seed
    // detection (the first EXR whose name parses as a sequence frame).
    let seed = if path.is_dir() {
        match first_frame_in_dir(&path) {
            Some(p) => {
                log::info!("open: seed frame in dir = {}", p.display());
                p
            }
            None => {
                return OpenOutcome::Failed {
                    path,
                    reason: "no image frame files in this directory".to_owned(),
                };
            }
        }
    } else {
        path.clone()
    };

    if detect_sequence {
        match Scanner::from_file(&seed) {
            Some(sequence) if sequence.frame_count() > 1 => {
                log::info!(
                    "scanseq: '{}' — {} frame(s) in [{}-{}], padding={}, {} missing",
                    sequence.pattern(),
                    sequence.frame_count(),
                    sequence.start,
                    sequence.end,
                    sequence.padding,
                    sequence.missed.len(),
                );
                let crypto = CryptoCatalog::discover(&seed);
                let stamps = frame_stamps(&sequence);
                return OpenOutcome::Sequence {
                    requested: path,
                    sequence,
                    stamps,
                    crypto,
                };
            }
            _ => log::info!("open: no multi-frame sequence — opening a single image"),
        }
    }

    // Too large to hold (a 65k plate) -> file-backed, decoded per tile band.
    // Half the tile cache's RAM budget: the frame must leave the cache room.
    match DecodedFrame::open(&seed, backing) {
        Ok(frame) => {
            let crypto = CryptoCatalog::discover(&seed);
            OpenOutcome::Single {
                path: seed,
                frame: Box::new(frame),
                crypto,
            }
        }
        Err(e) => OpenOutcome::Failed {
            path: seed,
            reason: e.status(),
        },
    }
}

/// The three GPU scope panels: histogram / waveform / vectorscope.
/// Each owns its compute pipelines + output texture and is driven once per
/// frame from the displayed image's linear `Rgba32Float` texture (zero copy).
/// Persisted on [`AppState`] so the pipelines are built once and reused.
#[derive(Default)]
pub struct Scopes {
    pub histogram: GpuHistogramPanel,
    pub waveform: GpuWaveformPanel,
    pub vectorscope: GpuVectorscopePanel,
}

/// A live frame sequence: the detected model + the playback clock + the decode
/// cache. The displayed frame is `cache.get_mut(playback.current())`; the worker
/// keeps it (and neighbours) decoded so the UI never blocks.
pub struct SequenceState {
    #[cfg(feature = "ofx")]
    source_epoch: Option<u64>,
    /// The detected sequence (frame set + naming), shared with the decode loader and
    /// replaced when a rescan finds new frames (the loader resolves frame -> file here).
    pub sequence: Arc<std::sync::RwLock<Seq>>,
    /// The folder rescan: frames added by a render in progress, frames rewritten.
    watch: SeqWatch,
    /// FPS-based clock over the present frames.
    pub playback: Playback,
    /// Async CPU frame cache of `Arc<DecodedFrame>` (background work-stealing
    /// decode + epoch cancel + byte-budget LRU — the `scancache` engine).
    pub cache: Scancache<DecodedFrame>,
    /// The cache's live byte cap, [`RamPartitions::frames`], re-set every tick by
    /// `tick_sequence` from what the tile cache and the OpenFX inputs hold.
    frame_budget: scancache::SharedBudget,
    /// Frames really presented per second and skipped while playing (the
    /// transport bar's indicator), fed by `ensure_display`.
    pub meter: crate::playback::ShowMeter,
    /// Display object for the CURRENT frame, rebuilt from the cache on a frame
    /// change. Holds GPU textures + per-view nav (the cached `DecodedFrame` is
    /// CPU-only); `None` until the first frame decodes.
    pub display: Option<ImageBackend>,
    /// Which frame `display` currently holds (so we rebuild only on change).
    pub displayed_frame: Option<i32>,
    /// Frame seen on the previous tick — distinguishes a user SEEK (→ epoch-
    /// cancelling `focus`) from a sequential clock ADVANCE (→ read-ahead `advance`).
    pub last_seen: i32,
    /// Set by the viewport draw while the LMB scrub gesture is live (one UI frame
    /// of lag behind the actual drag, since `tick_sequence` runs before the
    /// viewport is drawn). Selects [`scancache::Playhead::scrubbing`] so the cache
    /// admits only the frame under the cursor and defers read-ahead until the
    /// scrub settles (`Scancache::set_scrub_settle`), instead of the play-window
    /// read-ahead thrashing on every drag step.
    pub scrubbing: bool,
}

impl SequenceState {
    /// Collect a finished rescan and apply it; start the next one when due (once per
    /// [`WATCH_PERIOD`], on a thread: listing and `stat`ing thousands of frames must
    /// not stall a frame). `wake` brings the loop back when it lands. Returns whether
    /// the frame set or a frame changed.
    fn poll_rescan(&mut self, now: Instant, wake: &scancache::Wake) -> bool {
        let mut changed = false;
        if let Some(rx) = &self.watch.rx {
            match rx.try_recv() {
                Ok(found) => {
                    self.watch.rx = None;
                    if let Some((seq, stamps)) = found {
                        changed = self.apply_rescan(seq, stamps);
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return false,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    log::warn!("sequence rescan: the worker ended without a result");
                    self.watch.rx = None;
                }
            }
        }
        if now < self.watch.next {
            return changed;
        }
        self.watch.next = now + WATCH_PERIOD;
        let seed = self.sequence.read().ok().and_then(|s| s.get_file(i64::from(self.playback.current())));
        let Some(seed) = seed else { return changed };
        let (tx, rx) = std::sync::mpsc::channel();
        let wake = Arc::clone(wake);
        let spawned = std::thread::Builder::new().name("seq-rescan".into()).spawn(move || {
            let found = Scanner::from_file(Path::new(&seed)).map(|seq| {
                let stamps = frame_stamps(&seq);
                (seq, stamps)
            });
            if tx.send(found).is_ok() {
                wake();
            }
        });
        match spawned {
            Ok(_) => self.watch.rx = Some(rx),
            Err(e) => log::warn!("sequence rescan: could not spawn a worker: {e}"),
        }
        changed
    }

    /// Take a rescan: a new frame set goes to the cache (decoded survivors kept) and
    /// the clock; a rewritten frame is dropped from the cache so it decodes again, and
    /// the shown frame is rebuilt once its new decode lands.
    fn apply_rescan(&mut self, seq: Seq, stamps: Stamps) -> bool {
        let (set_changed, rewritten) = seq_changes(&self.watch.stamps, &stamps);
        self.watch.stamps = stamps;
        if !set_changed && rewritten.is_empty() {
            return false;
        }
        #[cfg(feature = "ofx")]
        { self.source_epoch = self.source_epoch.and_then(|epoch| epoch.checked_add(1)); }
        let indices: Vec<i32> = seq.indices.iter().map(|&f| f as i32).collect();
        // The loader must resolve the new frames before the cache asks for them.
        match self.sequence.write() {
            Ok(mut s) => *s = seq,
            Err(e) => log::warn!("sequence rescan: the sequence lock is poisoned: {e}"),
        }
        if set_changed {
            log::info!("sequence rescan: {} frames present", indices.len());
            self.cache.set_frames(indices.clone());
            self.playback.set_indices(indices);
        }
        for &f in &rewritten {
            self.cache.invalidate(f);
        }
        if !rewritten.is_empty() {
            log::info!("sequence rescan: {} frame(s) rewritten on disk", rewritten.len());
        }
        if rewritten.contains(&self.playback.current()) {
            self.displayed_frame = None;
        }
        true
    }

    /// Ensure `display` holds the current frame and return it, rebuilding from the
    /// cache when the frame changed. Returns `None` while the current frame is
    /// still decoding (the caller shows a "loading" placeholder). The persistent
    /// pan/zoom (`ViewNav`, on `AppState`) is applied by the caller after this.
    fn ensure_display(&mut self, tiles: Option<&DisplayTarget>) -> Option<&mut ImageBackend> {
        let cur = self.playback.current();
        if self.playback.playing && !self.scrubbing
            && self.display.as_ref().is_some_and(|d| tiles.is_some_and(|t| !t.can_advance(d.frame.id)))
        {
            return self.display.as_mut();
        }
        if self.displayed_frame != Some(cur) {
            // Rebuild ONLY once the new frame has decoded. While it is still
            // loading, KEEP the previous frame on screen — so the viewport widget
            // (and the LMB scrubber) stay live: dragging onto a not-yet-loaded
            // frame must NOT stop the scrub or swap in a text-cursor placeholder.
            if let Some(frame) = self.cache.get(cur) {
                // Re-use the existing backend (keeping its ~w*h*4 f32 buffer
                // allocation) instead of building a fresh one per displayed frame.
                match self.display.as_mut() {
                    Some(d) => d.set_frame(frame),
                    None => self.display = Some(ImageBackend::from_frame(frame)),
                }
                self.displayed_frame = Some(cur);
                // The swap happens only once the previous frame was presented
                // (`can_advance` above), so steps beyond one went by unshown.
                let stepped = self.playback.take_stepped();
                if self.playback.playing {
                    self.meter.record(std::time::Instant::now(), stepped.saturating_sub(1));
                } else {
                    self.meter.clear();
                }
            }
        }
        // `None` only before the very first frame has ever decoded.
        self.display.as_mut()
    }
}

/// How the one RAM budget is shared by its three consumers.
///
/// `tiles` (the tile RAM cache) and `holding` (OpenFX input leases) are CEILINGS
/// of caches that fill on demand, not reservations. `decoded` is the frame
/// cache's guaranteed floor (what remains when both are full); it also decides
/// which frames open file-backed. The frame cache's LIVE cap is [`Self::frames`]:
/// the whole budget minus what the other two hold right now. A static quarter
/// left a 64 GB machine with 12 GB of frames while the tile cache, idle on the
/// direct GPU path, sat on 24 GB, so a 20 GB sequence was re-decoded every loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RamPartitions { total: usize, decoded: usize, holding: usize, tiles: usize }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RamConfigurationTooSmall;
impl std::fmt::Display for RamConfigurationTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RAM budget must provide at least one byte each for decoding and frame holding")
    }
}
impl RamPartitions {
    fn new(total: usize) -> Self {
        let decoded = total / 4;
        let holding = total / 4;
        let tiles = total - decoded - holding;
        Self { total, decoded, holding, tiles }
    }
    /// The frame cache's live cap while the tile cache holds `tiles_used` bytes
    /// and the OpenFX inputs `holding_used`; never below `decoded`.
    fn frames(self, tiles_used: usize, holding_used: usize) -> usize {
        self.total - tiles_used.min(self.tiles) - holding_used.min(self.holding)
    }
    fn validate(self) -> std::result::Result<Self, RamConfigurationTooSmall> {
        if self.decoded == 0 || self.holding == 0 { Err(RamConfigurationTooSmall) } else { Ok(self) }
    }
}
/// Shared viewer state surfaced to the dock tabs: the loaded image (single OR a
/// sequence), the view controls, and a status line.
pub struct AppState {
    ram_partitions: RamPartitions,
    #[cfg(feature = "ofx")]
    holding_budget: scancache::HoldingBudget,
    /// Status line text shown in the bottom bar.
    pub status: String,
    /// Path requested on launch (consumed by the load seam on the first frame).
    pub pending_path: Option<PathBuf>,
    /// A single loaded image. `None` for a sequence or before load.
    pub image: Option<ImageBackend>,
    /// A loaded sequence with its playback clock + decode cache.
    pub seq: Option<SequenceState>,
    /// Persistent view navigation (pan/zoom/pick) — survives sequence frame swaps.
    pub nav: ViewNav,
    /// Display controls (exposure / gamma / channel) → GPU colour pass.
    pub controls: ViewControls,
    /// GPU scope panels — histogram / waveform / vectorscope.
    pub scopes: Scopes,
    /// The single, persistent GPU display target (linear + display textures + the
    /// reused egui `TextureId`), shared by the viewport and the scopes and reused
    /// across every frame swap. Lazily built on first display; reallocated only on
    /// a resolution INCREASE. See `DisplayTarget` — this is what makes sequence
    /// playback reuse one set of VRAM textures instead of leaking one per frame.
    pub display_gpu: DisplayTarget,
    /// Persisted viewer config (window geom + controls + recents). Updated on
    /// every successful open (last_file + recents) and saved on exit.
    pub config: Config,
    /// The RAM budget of this run (`persist::Budgets::ram` at launch; the caches are
    /// sized once).
    pub ram_budget: usize,
    /// The Settings panel (selected category, fonts).
    pub prefs: egui_prefs2::PrefsPanelState,
    /// The window output as it is (the viewer fills it; Settings and the status bar read it).
    pub display: DisplayStatus,
    /// The shown single image's file, re-opened when it changes on disk (a render
    /// writing into it, a comp re-saving it).
    watch: Option<Watch>,
    /// The open in flight is a reload of the shown file: keep the view (layer,
    /// channel, pan/zoom, pin) instead of resetting it.
    reloading: bool,
    /// The File > Open dialog.
    pub file_pick: FilePick,
    /// The named-layout bar (keeps its rename field between frames).
    pub layout_bar: egui_dock_layout::DockLayoutBar,
    /// The Metadata grid's column split (session only).
    pub metadata_grid: AttrGridState,
    /// Actions asked from inside a panel (the toolbar's snapshot menu, a tab's context
    /// menu), where the dock is borrowed: `build_ui` runs them once it is released.
    pub queue: Vec<Action>,
    /// The dock tabs that are open this frame (what a tab's context menu leaves out).
    pub open_tabs: Vec<DockTab>,
    /// Panels hidden (toggled with `Shift+Z`): draw ONLY the viewport (image +
    /// scrubber) — no menu / toolbar / status / transport / dock chrome.
    /// Independent of `fullscreen`. Transient (not persisted).
    pub panels_hidden: bool,
    /// A viewport is active in this UI pass; scope-only layouts advance freely.
    viewport_active: bool,
    /// Borderless OS-fullscreen (toggled with `Z`) via `ViewportCommand::
    /// Fullscreen`. Independent of `panels_hidden`. Transient (starts windowed).
    pub fullscreen: bool,
    /// Sequence auto-detection (CLI `--auto`). When false (the default),
    /// `open_path` opens the named frame/file as a SINGLE image and never scans
    /// the directory for a `name.####.exr` sequence.
    pub auto_detect: bool,
    /// Cryptomatte types/names for the open file (toolbar pick). `None` when the
    /// file has no Cryptomatte metadata.
    crypto: Option<CryptoCatalog>,
    /// How a finished decode asks the UI for a repaint. A no-op until the event
    /// loop hands us a proxy (`App::resumed`), so `AppState` stays constructible
    /// in tests without a window.
    wake: scancache::Wake,
    /// An open running on a worker thread. Dropping the receiver cancels our
    /// interest in it (a superseded open), so the newest request always wins.
    pending_open: Option<std::sync::mpsc::Receiver<OpenOutcome>>,
    /// OpenFX controls and the viewport's [`ImageStage`](crate::ImageStage) implementation.
    #[cfg(feature = "ofx")]
    pub effects: crate::ofx::OfxEffects,
}

impl AppState {
    /// Build the state, applying the saved display controls from `config` so
    /// exposure / gamma / channel persist across runs.
    fn new(
        initial: Option<PathBuf>,
        config: Config,
        auto_detect: bool,
        wake: scancache::Wake,
    ) -> Self {
        let config = Config { budgets: config.budgets.clamped(), ..config };
        let ram_budget = ram_budget(config.budgets.ram);
        let partitions = RamPartitions::new(ram_budget);
        let mut display_gpu = DisplayTarget::new(partitions.tiles, config.budgets.upload_bytes());
        // Finished tiles wake the loop, so a frame is presented as soon as its
        // tiles land instead of at the next playback deadline.
        display_gpu.set_wake(Arc::clone(&wake));
        let status = match &initial {
            Some(p) => format!("Loading '{}'…", p.display()),
            None => "Ready — open a file: exr-rs view <file>".to_owned(),
        };
        // Restore the persisted display controls (channel via the u8 mapping).
        let file_pick = FilePick::new(config.open_seq);
        let prefs = prefs_state(&config.fonts);
        let controls = ViewControls {
            exposure: config.exposure,
            exposure_hold: config.exposure_hold,
            gamma: config.gamma,
            channel: channel_from_u8(config.channel),
            tonemap: crate::transfer::tonemap::known(config.tonemap),
            backdrop: config.backdrop,
            colour: crate::ocio::State::with_dir(config.ocio.clone(), config.ocio_dir.clone()),
            target: crate::present::Target::default(),
            // Active layer / crypto pick are per-file (set on load), never persisted.
            view: Selection::default(),
            crypto_layer: String::new(),
            crypto_name: String::new(),
            hovered_pixel: None,
            nav: None,
        };
        Self {
            status,
            pending_path: initial,
            image: None,
            seq: None,
            nav: ViewNav::default(),
            controls,
            scopes: Scopes::default(),
            display_gpu,
            ram_partitions: partitions,
            #[cfg(feature = "ofx")]
            holding_budget: scancache::HoldingBudget::new(partitions.holding),
            ram_budget,
            config,
            prefs,
            display: DisplayStatus::default(),
            watch: None,
            reloading: false,
            file_pick,
            layout_bar: egui_dock_layout::DockLayoutBar::new(),
            metadata_grid: AttrGridState::default(),
            queue: Vec::new(),
            open_tabs: Vec::new(),
            panels_hidden: false,
            viewport_active: false,
            fullscreen: false,
            auto_detect,
            crypto: None,
            wake,
            pending_open: None,
            #[cfg(feature = "ofx")]
            effects: crate::ofx::OfxEffects::new(),
        }
    }

    /// The stage the viewport shows the image through: the OpenFX effect (feature `ofx`), else none.
    #[cfg(feature = "ofx")]
    fn stage(effects: &mut crate::ofx::OfxEffects) -> Option<&mut dyn crate::image::ImageStage> {
        Some(effects)
    }

    /// The Effects tab: the OpenFX effect's plug-in and parameters.
    pub(crate) fn ui_effects(&mut self, ui: &mut egui::Ui) {
        #[cfg(feature = "ofx")]
        self.effects.panel(ui);
        #[cfg(not(feature = "ofx"))]
        ui.label("This build has no OpenFX host (feature `ofx`).");
    }

    /// Record a successfully opened `path` into the config and persist it:
    /// set `last_file`, push it onto the deduplicated recents list, and save
    /// (best-effort) so recents survive even a crash before a clean exit.
    fn record_open(&mut self, path: &std::path::Path, was_sequence: bool) {
        self.config.last_file = Some(path.to_path_buf());
        self.config.last_sequence = was_sequence;
        self.config.push_recent(path);
        self.config.save();
    }

    /// Consume `pending_path` (if any) on the first frame and open it.
    ///
    /// Decides single-image vs sequence: `scanseq::Scanner::from_file` +
    /// `frame_count > 1` opens a sequence (playback clock + decode cache +
    /// worker), else the single image. `view <dir>` detects from any frame
    /// in that dir; `view <frame>` detects from that frame. A bad / deep /
    /// multipart EXR becomes a status message, never a panic.
    fn load_pending(&mut self) {
        let Some(path) = self.pending_path.take() else {
            return;
        };
        // Restore the saved open mode for the reopened last_file: if it was last
        // opened as a sequence, force sequence detection so it reopens as a
        // sequence (not a lone frame). An explicit CLI/menu path that is not the
        // saved last_file follows the normal `--auto` behaviour.
        let restore_sequence =
            self.config.last_sequence && self.config.last_file.as_deref() == Some(path.as_path());
        self.open_path_with(path, restore_sequence);
    }

    /// Open `path` (file / directory / frame pattern), replacing any currently
    /// loaded image or sequence. Used by the launch path (`Self::load_pending`)
    /// and the File menu; sequence auto-detection follows the `--auto` flag. A
    /// bad / deep / multipart EXR becomes a status message, never a panic.
    pub fn open_path(&mut self, path: PathBuf) {
        self.open_path_with(path, false);
    }

    /// Open a path the user chose in the viewer (File > Open, a recent file) by the
    /// "Auto-detect sequences" option (`Config::open_seq`, Settings > UI): on, the frame
    /// opens as its whole sequence; off, like [`Self::open_path`].
    pub fn open_chosen(&mut self, path: PathBuf) {
        self.open_path_with(path, self.config.open_seq);
    }

    /// Open a File > Open pick: its dialog box is the same option, so take it over first.
    pub fn open_picked(&mut self, path: PathBuf) {
        self.config.open_seq = self.file_pick.seq;
        self.open_chosen(path);
    }

    /// Open a drag-and-dropped `path`, always treating it as (potentially) a
    /// sequence: scanseq scans the dropped file's directory, so dropping any one
    /// frame of a `name.####.exr` set opens the WHOLE sequence regardless of the
    /// `--auto` flag. A lone EXR with no numbered siblings still opens as a single
    /// image (there is no sequence to form).
    pub fn open_dropped(&mut self, path: PathBuf) {
        self.open_path_with(path, true);
    }

    /// Shared open implementation. `force_sequence` runs scanseq sequence
    /// detection even when `--auto` is off (drag-and-drop sets it).
    ///
    /// Non-blocking: the directory scan, sequence detection, EXR decode and
    /// Cryptomatte manifest parse all run on a worker thread (`open_blocking`)
    /// and are installed by [`Self::tick_open`]. The current view is left alone
    /// until the new one has actually loaded — an open that fails no longer
    /// leaves an empty viewport behind.
    fn open_path_with(&mut self, path: PathBuf, force_sequence: bool) {
        let detect = self.auto_detect || force_sequence;
        let backing = self.backing_budget();
        self.status = format!("Opening {}…", path.display());
        let (tx, rx) = std::sync::mpsc::channel();
        // Replacing the receiver abandons any previous in-flight open.
        self.pending_open = Some(rx);
        let wake = Arc::clone(&self.wake);
        let for_worker = path.clone();
        let spawned = std::thread::Builder::new()
            .name("exr-view-open".to_owned())
            .spawn(move || {
                let outcome = open_blocking(for_worker, detect, backing);
                // A send error means the UI moved on (a newer open); nothing to do.
                if tx.send(outcome).is_ok() {
                    wake();
                }
            });
        if let Err(e) = spawned {
            // No worker: do it inline rather than silently never opening.
            log::warn!("open: could not spawn a worker ({e}) — opening on the UI thread");
            self.pending_open = None;
            let outcome = open_blocking(path, detect, backing);
            self.apply_open(outcome);
        }
    }

    /// Install a finished background open, if one has arrived. Called once per
    /// UI frame; never blocks.
    fn tick_open(&mut self) {
        let Some(rx) = self.pending_open.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(outcome) => {
                self.pending_open = None;
                self.apply_open(outcome);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // The worker died without sending (a panic in a dependency).
                self.pending_open = None;
                self.status = "open failed: the loader thread stopped".to_owned();
                log::error!("open: worker thread ended without a result");
            }
        }
    }

    /// Replace the current view with a finished open — or, on failure, keep it.
    fn apply_open(&mut self, outcome: OpenOutcome) {
        match outcome {
            OpenOutcome::Sequence {
                requested,
                sequence,
                stamps,
                crypto,
            } => {
                self.reloading = false;
                self.watch = None;
                self.reset_view_state();
                self.record_open(&requested, true);
                self.crypto = crypto;
                self.open_sequence(sequence, stamps);
            }
            OpenOutcome::Single {
                path,
                frame,
                crypto,
            } => {
                let img = ImageBackend::from_frame(Arc::new(*frame));
                self.status = format!(
                    "{}  —  {}×{}  [{}]",
                    img.file_name(),
                    img.width,
                    img.height,
                    img.frame.channel_names.join(" ")
                );
                log::info!(
                    "open: loaded {} — {}x{}",
                    img.file_name(),
                    img.width,
                    img.height
                );
                let mut img = img;
                if std::mem::take(&mut self.reloading)
                    && let Some(old) = self.image.as_ref()
                {
                    // Same file, new pixels: the user keeps looking where they were.
                    img.pan = old.pan;
                    img.zoom = old.zoom;
                    img.fit_zoom = old.fit_zoom;
                    img.picked_pixel = old.picked_pixel.filter(|&(x, y)| x < img.width && y < img.height);
                } else {
                    self.reset_view_state();
                    self.record_open(&path, false);
                }
                self.watch = Some(Watch::new(path));
                self.crypto = crypto;
                self.image = Some(img);
            }
            OpenOutcome::Failed { path, reason } => {
                // A reload of a file still being written fails; the next change retries.
                self.reloading = false;
                // Deliberately NOT clearing the current image/sequence: a typo in
                // a path used to throw away the view the user was working with.
                self.status = format!("{}: {reason}", path.display());
                log::error!("open FAILED for {}: {reason}", path.display());
            }
        }
    }

    /// Drop the currently displayed image/sequence and every per-file selection.
    /// Called only when a replacement has actually loaded.
    fn reset_view_state(&mut self) {
        self.display_gpu.clear_presentation();
        self.image = None;
        self.seq = None;
        self.nav = ViewNav::default();
        // Layers / crypto picks are per-file: start every new load on main / none.
        self.controls.view = Selection::default();
        // A Fit asked while the previous file was shown is not for this one.
        self.controls.nav = None;
        self.controls.crypto_layer.clear();
        self.controls.crypto_name.clear();
        self.crypto = None;
    }

    /// Open a detected sequence: build the clock over the PRESENT frame set
    /// (`scanseq::Seq::indices`), the decode cache, and prime the first frame.
    fn open_sequence(&mut self, sequence: Seq, stamps: Stamps) {
        if let Err(error) = self.ram_partitions.validate() {
            self.status = error.to_string();
            return;
        }
        // User-meaningful milestone (info): a sequence was detected — its frame
        // range + count + padding.
        log::info!(
            "sequence detected: {} frames [{}-{}] count={}",
            sequence.pattern(),
            sequence.start,
            sequence.end,
            sequence.frame_count()
        );
        // Drive playback over the frames actually present (EXR frame numbers fit
        // i32; scanseq tracks them as i64). Gaps are simply absent from this set.
        let indices: Vec<i32> = sequence.indices.iter().map(|&f| f as i32).collect();
        // Seed the clock with the persisted playback prefs (fps + loop).
        let mut playback = Playback::new(indices.clone(), self.config.fps);
        playback.loop_ = self.config.loop_;
        let cur = playback.current();

        // Decode loader (runs on a worker thread): resolve a frame number to its
        // file via the shared sequence (a rescan may add frames), then CPU-decode
        // it. A frame outside the set is a clear error (surfaced on the status line).
        let shared = Arc::new(std::sync::RwLock::new(sequence.clone()));
        let seq_for_loader = Arc::clone(&shared);
        // Frames too large to hold open file-backed (tiles decode the visible
        // rows per frame), with the single-image rule; computed once here.
        let backing_budget = self.backing_budget();
        // `exr_core::api::read` (inside `DecodedFrame::open`) is not interruptible
        // mid-decode, so `cancel` is honoured at the two points that ARE cheap:
        // before starting (skip work the cache no longer wants) and before
        // publishing the result (a decode that finished after going stale must not
        // be reported as success — `Scancache` drops a cancelled `Err` with no
        // failure record and no retry backoff).
        let loader: scancache::Loader<DecodedFrame> =
            Arc::new(move |frame: i32, cancel: &scancache::Cancel| {
                if cancel.is_cancelled() {
                    return Err("cancelled".to_owned());
                }
                let file = seq_for_loader.read().ok().and_then(|s| s.get_file(i64::from(frame)));
                let result = match file {
                    Some(path) => DecodedFrame::open(std::path::Path::new(&path), backing_budget)
                        .map_err(|e| e.status()),
                    None => Err(format!("frame {frame}: not present in the sequence")),
                };
                if cancel.is_cancelled() {
                    return Err("cancelled".to_owned());
                }
                result
            });
        let size_of: scancache::SizeOf<DecodedFrame> = Arc::new(|f: &DecodedFrame| f.byte_size());
        // A finished decode wakes the event loop directly (no UI polling).
        let wake = Arc::clone(&self.wake);
        let frame_budget: scancache::SharedBudget =
            Arc::new(std::sync::atomic::AtomicUsize::new(self.frame_cap()));
        let cache = Scancache::with_shared_budget(
            indices,
            Arc::clone(&frame_budget),
            decode_threads(),
            size_of,
            loader,
            wake,
        );
        // Point the pull pool at the opened frame under the user's strategy; workers
        // fill outward from here and re-prioritise on any later scrub.
        cache.set_strategy(self.config.preload_strategy);
        cache.set_playhead(
            scancache::Playhead::at(cur).with_range((playback.first(), playback.last())),
        );
        log::info!(
            "sequence cache: {} present frames, focus frame {cur} + background fill",
            cache.total()
        );
        self.status = format!(
            "{}  —  {} frames [{}-{}]{}",
            sequence.pattern(),
            sequence.frame_count(),
            sequence.start,
            sequence.end,
            if sequence.is_complete() {
                String::new()
            } else {
                format!("  ({} missing)", sequence.missed.len())
            },
        );
        self.seq = Some(SequenceState {
            #[cfg(feature = "ofx")]
            source_epoch: Some(0),
            sequence: shared,
            watch: SeqWatch::new(stamps),
            playback,
            cache,
            frame_budget,
            meter: Default::default(),
            display: None,
            displayed_frame: None,
            last_seen: cur,
            scrubbing: false,
        });
    }

    /// Per-frame sequence bookkeeping: advance the clock and keep the cache
    /// focused on the playhead. Called once per UI frame before the viewport
    /// draws. The UI NEVER blocks on decode (it happens on the worker pool).
    fn tick_sequence(&mut self) {
        let strategy = self.config.preload_strategy;
        let cap = self.frame_cap();
        let Some(s) = self.seq.as_mut() else {
            return;
        };
        // Rebalance the one RAM budget: the frame cache gets what the tile cache and
        // the OpenFX inputs do not hold right now. A shrink frees the RAM at once; a
        // growth is used by the workers' next pull (every playhead step and decode).
        if s.frame_budget.swap(cap, std::sync::atomic::Ordering::Relaxed) > cap {
            s.cache.trim_to_budget();
        }
        // Advance the clock, then point the cache's pull pool at the current
        // playhead. `set_playhead` is a no-op when the playhead is unchanged, so
        // rebuilding it fresh every tick (rather than gating on `advanced` / a
        // frame-number diff) is both simpler and correct — direction and scrub
        // state matter to the ranker even when the frame number itself has not
        // moved yet.
        s.playback.tick();
        let cur = s.playback.current();
        let range = (s.playback.first(), s.playback.last());
        let ph = if s.scrubbing {
            scancache::Playhead::scrubbing(cur)
        } else if s.playback.playing {
            let dir = if s.playback.reverse { scancache::Dir::Backward } else { scancache::Dir::Forward };
            scancache::Playhead::playing(cur, dir, s.playback.loop_)
        } else {
            scancache::Playhead::at(cur)
        }
        .with_range(range);
        s.cache.set_playhead(ph);
        s.last_seen = cur;
        // Apply the user's preload strategy (no-op when unchanged).
        s.cache.set_strategy(strategy);
        // Service any deferred deadline (scrub settle / retry backoff) that has
        // come due — without this a scrub never resumes read-ahead and a failed
        // frame is never retried, since nothing else calls back into the cache
        // once `has_pending()` goes quiet.
        s.cache.poll();
        // Surface a decode error to the status line (never a panic).
        if let Some((frame, msg)) = s.cache.take_error() {
            self.status = format!("frame {frame}: {msg}");
        }
    }

    /// Apply / clear Cryptomatte coverage overlay on `img` from toolbar picks.
    /// Uses resident decoded channels (no second disk read). Idempotent when the
    /// selection already matches `img.coverage_sel()`.
    fn sync_crypto_overlay(
        crypto: Option<&CryptoCatalog>,
        layer: &str,
        name: &str,
        img: &mut ImageBackend,
    ) {
        let want = if layer.is_empty() || name.is_empty() {
            None
        } else {
            Some((layer.to_owned(), name.to_owned()))
        };
        if img.coverage_sel() == want.as_ref() {
            return;
        }
        // Already tried this selection on this frame and it failed — do NOT
        // retry (each attempt re-materializes and re-uploads the whole image).
        // `set_frame` / a new selection clears the memo.
        if want.is_some() && img.coverage_failed() == want.as_ref() {
            return;
        }
        let Some((layer_name, obj_name)) = want else {
            img.set_coverage_overlay(None, None);
            return;
        };
        let Some(cat) = crypto else {
            log::warn!("cryptomatte overlay: no catalog for this file");
            img.mark_coverage_failed((layer_name, obj_name));
            return;
        };
        let Some(layer_info) = cat.layer(&layer_name) else {
            log::warn!("cryptomatte overlay: unknown layer `{layer_name}`");
            img.mark_coverage_failed((layer_name, obj_name));
            return;
        };
        let id = layer_info.manifest.id_for_name(&obj_name);
        if img.is_file_backed() {
            // No resident channels: the tile source computes the coverage per
            // band from the file (`tiles::file::FileSource::coverage`).
            img.set_coverage_file((layer_name, obj_name), layer_info.ranked.clone(), id);
            return;
        }
        let result = {
            let channels = &img.frame.channels;
            extract_coverage_with(
                img.width,
                img.height,
                &layer_info.ranked,
                &[id],
                |ch_name| {
                    channels
                        .iter()
                        .find(|(n, _)| n == ch_name)
                        .map(|(_, d)| d)
                        .ok_or_else(|| {
                            cryptomatte::Error::Metadata(format!(
                                "missing Cryptomatte channel `{ch_name}`"
                            ))
                        })
                },
            )
        };
        match result {
            Ok(matte) => {
                img.set_coverage_overlay(Some(matte.coverage), Some((layer_name, obj_name)));
            }
            Err(e) => {
                log::warn!("cryptomatte overlay extract failed: {e}");
                img.mark_coverage_failed((layer_name, obj_name));
            }
        }
    }

    /// Draw the viewport tab: routes single-image vs sequence to the SAME
    /// [`viewport::show`] (single source of responsibility for GPU display).
    ///
    /// For a sequence, the current frame's backend comes from the cache; the
    /// persistent [`ViewNav`] is copied in before drawing and read back after,
    /// so pan/zoom/pick stay put as frames swap. A not-yet-decoded frame shows a
    /// "loading frame N…" placeholder (no block).
    pub(crate) fn display_viewport(&mut self, ui: &mut egui::Ui, gpu: &mut dyn TabGpu, strip: viewport::Strip) {
        // On-image playhead-line opacity (a setting; 0 = off). Snapshot before the
        // image/seq borrows so the viewport call can take it by value.
        let playhead_alpha = self.config.playhead_alpha;
        // The shared, persistent GPU display target — split-borrowed before the
        // image/seq borrow so the same textures + egui id are reused every frame.
        let display_gpu = &mut self.display_gpu;
        let crypto_layer = self.controls.crypto_layer.clone();
        let crypto_name = self.controls.crypto_name.clone();
        let crypto = self.crypto.clone();
        #[cfg(feature = "ofx")]
        let effects = &mut self.effects;
        // Single image: the backend owns its own nav.
        if let Some(img) = self.image.as_mut() {
            // Single image: no sequence — no scrub/playhead (LMB pins a pixel).
            Self::sync_crypto_overlay(crypto.as_ref(), &crypto_layer, &crypto_name, img);
            #[cfg(feature = "ofx")]
            {
                effects.set_time(0.0);
                let frame = img.decoded();
                let identity = crate::ofx::InputIdentity {
                    primary: frame.id, index: 0, source_epoch: 0,
                    revision: effects.generation_key(), rate_bits: 0, range: (0, 0),
                };
                if !effects.inputs_match(identity) {
                    effects.set_inputs(identity, Arc::new(crate::tiles::effect_frames::Inputs::Static {
                        frame, index: 0,
                    }), None, Some([0.0, 0.0]));
                }
            }
            #[cfg(feature = "ofx")]
            let stage = Self::stage(effects);
            #[cfg(not(feature = "ofx"))]
            let stage = None;
            // The viewport's outcome is its scrub, which a single image has none of.
            let _single = viewport::show(
                ui,
                img,
                &mut self.controls,
                gpu,
                display_gpu,
                None,
                playhead_alpha,
                None,
                stage,
                strip,
            );
            return;
        }

        // Sequence: pull the current frame's backend from the cache.
        if let Some(s) = self.seq.as_mut() {
            let cur = s.playback.current();
            let pos = s.playback.position();
            // Frame-number range for the scrub cursor's live frame readout.
            let seq_range = (s.playback.first(), s.playback.last());
            let mut scrub_to = None;
            // Rebuild the display backend from the cached `Arc<DecodedFrame>` when
            // the frame changed; `None` = the current frame is still decoding.
            s.ensure_display(Some(display_gpu));
            #[cfg(feature = "ofx")]
            let shown = s.displayed_frame.unwrap_or(cur);
            if let Some(img) = s.display.as_mut() {
                self.nav.apply_to(img);
                Self::sync_crypto_overlay(crypto.as_ref(), &crypto_layer, &crypto_name, img);
                #[cfg(feature = "ofx")]
                {
                    effects.set_time(f64::from(shown));
                    let frame = img.decoded();
                    let rate = f64::from(s.playback.fps);
                    if let Some(source_epoch) = s.source_epoch {
                        let identity = crate::ofx::InputIdentity {
                            primary: frame.id, index: shown, source_epoch,
                            revision: effects.generation_key(), rate_bits: rate.to_bits(), range: seq_range,
                        };
                        if !effects.inputs_match(identity) {
                            match s.cache.open_demand(self.holding_budget.clone(),
                                crate::tiles::effect_frames::MAX_INPUT_FRAMES) {
                                Ok(lease) => effects.set_inputs(identity, Arc::new(
                                    crate::tiles::effect_frames::Inputs::Demand {
                                        lease, primary_index: shown, primary_id: frame.id,
                                    }), Some(rate), Some([f64::from(seq_range.0), f64::from(seq_range.1)])),
                                Err(error) => effects.set_input_error(format!("OpenFX input lease unavailable: {error:?}")),
                            }
                        }
                    } else {
                        effects.set_input_error("sequence source generation exhausted".into());
                    }
                }
                #[cfg(feature = "ofx")]
                let stage = Self::stage(effects);
                #[cfg(not(feature = "ofx"))]
                let stage = None;
                // `Some(pos)` enables the image-relative scrub + playhead line; the
                // LMB drag fraction comes back in `out.scrub_to`.
                let out = viewport::show(
                    ui,
                    img,
                    &mut self.controls,
                    gpu,
                    display_gpu,
                    Some(pos),
                    playhead_alpha,
                    Some(seq_range),
                    stage,
                    strip,
                );
                self.nav.read_from(img);
                scrub_to = out.scrub_to;
                s.scrubbing = out.scrub_to.is_some();
            } else {
                // Only reached before the FIRST frame has ever decoded (after that
                // the previous frame is kept on screen). Non-selectable so it never
                // shows a text cursor over the viewport.
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Label::new(format!("loading frame {cur}…")).selectable(false));
                });
            }
            // Apply the on-screen scrubber's seek (after the cache borrow ends).
            if let Some(frac) = scrub_to {
                s.playback.scrub_fraction(frac);
            }
            return;
        }

        ui.centered_and_justified(|ui| {
            ui.label("No image loaded — File > Open (Ctrl+O), drop a file here, or run: exr-rs view <file>");
        });
    }

    /// Draw a GPU scope tab (histogram / waveform / vectorscope).
    ///
    /// Feeds the scope from the SAME linear `Rgba32Float` texture the viewport
    /// uses (via [`ImageBackend::ensure_linear`] — zero re-upload). The scope
    /// computes into its own texture each frame from the currently displayed
    /// frame, so during sequence playback it automatically follows the frame.
    /// No image (or a not-yet-decoded sequence frame) → a placeholder label,
    /// never a panic.
    pub(crate) fn display_scope(
        &mut self,
        ui: &mut egui::Ui,
        tab: &DockTab,
        gpu: &mut dyn TabGpu,
    ) {
        // Resolve the displayed backend (single image, or the sequence's current
        // decoded frame). Split-borrow `scopes` from the image source so both can
        // be borrowed at once (disjoint fields of `self`).
        let scopes = &mut self.scopes;
        let display_gpu = &mut self.display_gpu;
        let crypto_layer = self.controls.crypto_layer.clone();
        let crypto_name = self.controls.crypto_name.clone();
        let crypto = self.crypto.clone();
        #[cfg(feature = "ofx")]
        let stage = Self::stage(&mut self.effects);
        #[cfg(not(feature = "ofx"))]
        let stage = None;
        let src: Option<&mut ImageBackend> = if let Some(img) = self.image.as_mut() {
            Some(img)
        } else if let Some(s) = self.seq.as_mut() {
            s.ensure_display(self.viewport_active.then_some(display_gpu))
        } else {
            None
        };

        let Some(img) = src else {
            ui.centered_and_justified(|ui| {
                ui.label("No image — scope updates once a frame is loaded.");
            });
            return;
        };
        Self::sync_crypto_overlay(crypto.as_ref(), &crypto_layer, &crypto_name, img);
        let Some(gpu) = gpu.gpu() else {
            ui.centered_and_justified(|ui| {
                ui.label("No GPU adapter — scopes need one.");
            });
            return;
        };

        // Ensure the linear source texture is on the GPU (no-op if present()
        // already uploaded it this frame) and grab its view + dimensions.
        let Some((src_view, src_w, src_h)) =
            img.ensure_linear(gpu.device, gpu.queue, display_gpu, stage)
        else {
            ui.centered_and_justified(|ui| {
                ui.label("Empty image — nothing to analyse.");
            });
            return;
        };

        // Vectorscope is square; histogram / waveform fill the panel.
        let square = matches!(tab, DockTab::Vectorscope);
        let panel: &mut dyn GpuPanel = match tab {
            DockTab::Histogram => &mut scopes.histogram,
            DockTab::Waveform => &mut scopes.waveform,
            DockTab::Vectorscope => &mut scopes.vectorscope,
            // Non-scope tabs never reach here (routed in `DockTabViewer::ui`).
            _ => return,
        };

        // Compute into the scope's output texture sized to the panel rect, then
        // paint it. The panel size is taken from the egui layout each frame.
        let avail = ui.available_size();
        let (out_w, out_h) = if square {
            let s = avail.x.min(avail.y).max(1.0) as u32;
            (s, s)
        } else {
            (avail.x.max(1.0) as u32, avail.y.max(1.0) as u32)
        };
        let mut ctx = PanelCtx {
            device: gpu.device,
            queue: gpu.queue,
            renderer: gpu.renderer,
        };
        panel.tick(&mut ctx, src_view, src_w, src_h, out_w, out_h);
        show_panel(panel, ui, square);
    }

    /// The currently displayed backend (single image, or the sequence's current
    /// frame if it is decoded) — used by the metadata + pixel-info panels.
    fn current_backend(&self) -> Option<&ImageBackend> {
        if let Some(img) = &self.image {
            return Some(img);
        }
        // The last-built sequence display (its metadata/pixels follow the frame).
        self.seq.as_ref()?.display.as_ref()
    }

    /// Frames too large to hold open file-backed, decoded per tile band: above half
    /// the tile cache's RAM budget, so the frame leaves the cache room.
    fn backing_budget(&self) -> u64 {
        self.ram_partitions.decoded as u64
    }

    /// The frame cache's live cap ([`RamPartitions::frames`]) for the RAM the tile
    /// cache and the OpenFX inputs hold now.
    fn frame_cap(&self) -> usize {
        #[cfg(feature = "ofx")]
        let holding = self.holding_budget.used_bytes();
        #[cfg(not(feature = "ofx"))]
        let holding = 0;
        self.ram_partitions.frames(self.display_gpu.tiles_ram().0, holding)
    }

    /// The Colour tab: the OCIO display transform (`ocio::State::ui`); the config path field's
    /// Browse button opens the file dialog for a config.
    pub fn ui_colour(&mut self, ui: &mut egui::Ui) {
        let mut browse = false;
        self.controls.colour.ui(ui, &mut browse);
        if browse {
            self.file_pick.pick_config(self.controls.colour.start_dir());
        }
    }

    /// What the output in use encodes for (SDR white, display peak): the user's,
    /// else the OS's, else the output's default (`persist::DisplayPrefs::target`,
    /// the one rule).
    pub fn target(&self) -> crate::present::Target {
        self.config.display.target(self.display.active, &self.display.info)
    }

    /// The Settings tab: memory, output, UI and the same OCIO controls as the
    /// Colour tab. Both entry points edit one persisted colour selection.
    pub fn ui_settings(&mut self, ui: &mut egui::Ui) {
        use egui_phosphor::regular as icon;
        let target = self.target();
        let config = &mut self.config;
        let display = &self.display;
        let colour = &mut self.controls.colour;
        let mut browse = false;
        let mut open_colour = false;
        let mut open_display = false;
        let mut category = self.prefs.selected;
        let mut reset = false;
        let fonts = config.fonts;
        let categories = [
            egui_prefs2::Category::new(icon::MEMORY, "Memory"),
            egui_prefs2::Category::new(icon::MONITOR, "Display"),
            egui_prefs2::Category::new(icon::TEXT_AA, "UI"),
            egui_prefs2::Category::new(icon::MONITOR, "Colour"),
        ];
        egui_prefs2::draw(
            ui,
            &mut self.prefs,
            &categories,
            |ui, idx| {
                category = idx;
                match idx {
                    0 => budgets_ui(ui, &mut config.budgets),
                    1 => {
                        display_ui(ui, &mut config.display, display, target);
                        ui.add_space(8.0);
                        open_colour = ui.button("Colour management & monitor presets…").clicked();
                    }
                    3 => {
                        egui_prefs2::section_header(ui, "Colour management");
                        ui.label(format!("Window output: {}.", display.active.label()));
                        ui.label("Monitor presets choose the image rendering. HDR 1000 nits is a rendering peak; SDR reference white controls the brightness of the UI and relative-white image values.");
                        open_display = ui.button("Display output & reference white…").clicked();
                        ui.add_space(8.0);
                        colour.ui(ui, &mut browse);
                    }
                    _ => ui_prefs(ui, &mut config.fonts, &mut config.open_seq),
                }
            },
            Some(|| reset = true),
        );
        if reset {
            match category {
                0 => config.budgets = Budgets::default(),
                1 => config.display = crate::persist::DisplayPrefs::default(),
                3 => {
                    colour.sel = crate::ocio::Sel::default();
                    colour.reload();
                }
                _ => {
                    let d = crate::persist::Config::default();
                    (config.fonts, config.open_seq) = (d.fonts, d.open_seq);
                }
            }
        }
        if config.fonts != fonts {
            theme::set(ui.ctx(), &theme::current(ui.ctx()), &config.fonts);
            self.prefs.fonts = prefs_fonts(&config.fonts);
        }
        self.display_gpu.upload = config.budgets.upload_bytes();
        if browse {
            self.file_pick.pick_config(colour.start_dir());
        }
        if open_colour {
            self.prefs.selected = 3;
            ui.ctx().request_repaint();
        } else if open_display {
            self.prefs.selected = 1;
            ui.ctx().request_repaint();
        }
    }

    /// Re-open the shown single image when its file changed on disk (checked at most
    /// once per `WATCH_PERIOD`, one `stat`; no watcher thread). Returns whether a
    /// reload started.
    pub fn poll_watch(&mut self) -> bool {
        if self.pending_open.is_some() {
            return false;
        }
        if let Some(s) = self.seq.as_mut() {
            return s.poll_rescan(Instant::now(), &self.wake);
        }
        let Some(watch) = self.watch.as_mut() else {
            return false;
        };
        if !watch.changed(Instant::now()) {
            return false;
        }
        let path = watch.path.clone();
        log::info!("{} changed on disk: reloading", path.display());
        self.reloading = true;
        self.open_path_with(path, false);
        true
    }

    /// When [`Self::poll_watch`] next wants to run (the event loop sleeps until then).
    pub fn watch_deadline(&self) -> Option<Instant> {
        match &self.seq {
            Some(s) => Some(s.watch.next),
            None => self.watch.as_ref().map(|w| w.next),
        }
    }

    /// Save what is shown (`export.rs`) and say how it went on the status line: an SDR
    /// 8-bit PNG, or with `hdr` the display light as `.exr` / that signal as a `.png`.
    fn save_view(&mut self, path: &Path, max_dim: u32, hdr: Option<crate::export::HdrPng>) {
        let look = crate::export::Look {
            exposure: self.controls.exposure,
            gamma: self.controls.gamma,
            mode: self.controls.channel.to_channel_mode(),
            tonemap: self.controls.tonemap,
            ocio: self.controls.colour.active().cloned(),
            white: self.controls.target.white,
            peak: self.controls.target.peak,
        };
        let display_gpu = &mut self.display_gpu;
        #[cfg(feature = "ofx")]
        let stage = Self::stage(&mut self.effects);
        #[cfg(not(feature = "ofx"))]
        let stage = None;
        let img = match (self.image.as_mut(), self.seq.as_mut()) {
            (Some(img), _) => Some(img),
            (None, Some(s)) => s.display.as_mut(),
            (None, None) => None,
        };
        let Some(img) = img else {
            self.status = "Nothing to save".to_owned();
            return;
        };
        self.status = match crate::export::grab(img, display_gpu, max_dim, stage) {
            Ok(None) => "Save view: the image is still loading, try again".to_owned(),
            Err(crate::tiles::region::RegionError::Pending) =>
                "Save view: pixel admission is busy, try again".to_owned(),
            Err(error) => format!("Save view failed: {error}"),
            Ok(Some(raw)) => {
                let saved = match hdr {
                    Some(png) => crate::export::save_hdr(path, &raw, &look, png),
                    None => crate::export::write_sdr_png(path, &raw, &look),
                };
                match saved {
                    Ok(()) if raw.level == 0 => format!("Saved {} ({}x{})", path.display(), raw.w, raw.h),
                    Ok(()) => format!("Saved {} ({}x{}, pyramid level {})", path.display(), raw.w, raw.h, raw.level),
                    Err(e) => format!("Save view failed: {e:#}"),
                }
            }
        };
    }

    /// What is open, for the command availability rule (`commands::enabled`).
    pub fn avail(&self) -> commands::Avail {
        commands::Avail { seq: self.seq.is_some(), file: self.image.is_some() || self.seq.is_some() }
    }

    /// The EXR `step` items away from what is shown, in its folder by name; the
    /// frames of an open sequence count as one item, so PgDn leaves the sequence.
    fn sibling(&self, step: isize) -> Option<PathBuf> {
        let (cur, own): (PathBuf, HashSet<PathBuf>) = match (&self.image, &self.seq) {
            (Some(img), _) => (img.frame.path.clone(), HashSet::new()),
            (None, Some(s)) => {
                let seq = s.sequence.read().ok()?;
                let file = |f: i32| seq.get_file(i64::from(f)).map(PathBuf::from);
                let own = s.playback.indices().iter().filter_map(|&f| file(f)).collect();
                (file(s.playback.current())?, own)
            }
            (None, None) => return None,
        };
        let mut files: Vec<PathBuf> = std::fs::read_dir(cur.parent()?)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && exr_import::is_image(p))
            .collect();
        files.sort();
        neighbour(&files, &cur, step, &own)
    }

    /// EXR header metadata panel: tinted collapsible sections (`egui-titlebar`) for the channel
    /// tree, the processed presentation and the header attributes.
    pub fn ui_metadata(&mut self, ui: &mut egui::Ui) {
        let colors = theme::current(ui.ctx()).colors;

        // The channel tree: a layer, or one channel alone. It writes the one shared
        // `controls.view`, like the toolbar's layer combo.
        let picked = if self.current_backend().is_some() {
            section(ui, "Channels", colors.status_info, |ui| {
                self.current_backend().and_then(|img| channels::show(ui, &img.frame, &self.controls.view))
            })
            .flatten()
        } else {
            None
        };
        if let Some(view) = picked {
            if view.channel.is_some() {
                self.controls.channel = ChannelDisplay::Rgba;
            }
            self.controls.view = view;
        }

        #[cfg(feature = "ofx")]
        let frozen = self.display_gpu.accepted_pixels();
        #[cfg(feature = "ofx")]
        if let Some(pixels) = frozen.as_ref() {
            section(ui, "Processed", colors.status_warn, |ui| {
                let source = pixels.lineage();
                ui.monospace(format!("frame: {}   time: {}   revision: {}",
                    source.primary_index, source.time, pixels.revision()));
                ui.monospace(format!("Primary data window: {:?}", source.data_window));
                ui.monospace(format!("Primary display window: {:?}", source.display_window));
                ui.monospace(format!("Primary pixel aspect: {}", source.pixel_aspect));
            });
        }
        // Only show raw header attributes belonging to this accepted primary.
        let fields = self.current_backend().filter(|_img| {
            #[cfg(feature = "ofx")]
            if let Some(pixels) = frozen.as_ref() { return pixels.is_primary_frame(&_img.frame); }
            true
        }).map(|img| {
            img.frame
                .metadata
                .iter()
                .enumerate()
                .map(|(i, e)| AttrField::new(&e.key, AttrValue::Label(e.value.clone())).with_order(i as f32))
                .collect::<Vec<_>>()
        });
        let grid = &mut self.metadata_grid;
        section(ui, "Attributes", colors.status_ok, |ui| match fields {
            Some(mut fields) => {
                #[cfg(feature = "ofx")]
                if frozen.is_some() {
                    ui.label("Original primary source header");
                }
                // Header values are text to copy (theme.rs turns label selection off elsewhere).
                ui.style_mut().interaction.selectable_labels = true;
                egui_attr_grid::render_grid(ui, &mut fields, grid, &HashSet::new());
            }
            None => {
                ui.label("EXR header attributes appear here once a file is loaded.");
            }
        });
    }

    /// Pixel inspector panel: the hovered/pinned pixel's RGBA values.
    pub fn ui_pixel_info(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pixel Info");
        ui.separator();
        // Pinned pick: single image keeps it on its backend; a sequence keeps it
        // on the persistent nav (so it survives frame swaps).
        let nav_pick = self.nav.picked_pixel;
        let hovered = self.controls.hovered_pixel;
        #[cfg(feature = "ofx")]
        if let Some(pixels) = self.display_gpu.accepted_pixels() {
            let img = self.current_backend();
            let pinned = if self.image.is_some() { img.and_then(|img| img.picked_pixel) } else { nav_pick };
            if let Some((x, y)) = hovered.or(pinned) {
                if let Some(sample) = pixels.canvas_sample(x, y) {
                    ui.monospace(format!("canvas ({x}, {y})   frame: {}   time: {}   level: {}",
                        pixels.lineage().primary_index, pixels.lineage().time, sample.level));
                    for (name, value) in ["R", "G", "B", "A"].into_iter().zip(sample.pixel) {
                        ui.monospace(format!("{name}: {value:.6}"));
                    }
                    if let Some(img) = img.filter(|img| pixels.is_primary_frame(&img.frame))
                        && let Some((dx, dy)) = pixels.primary_pixel(x, y)
                        && let Some(all) = img.probe_at(dx, dy)
                    {
                        ui.separator();
                        ui.label(egui::RichText::new("Original primary source point — all channels").strong());
                        for (name, value) in all { ui.monospace(format!("{name}: {value}")); }
                    }
                }
            } else {
                ui.label("Hover the image to inspect pixels; click to pin.");
            }
            return;
        }
        match self.current_backend() {
            Some(img) => {
                let pinned = if self.image.is_some() {
                    img.picked_pixel
                } else {
                    nav_pick
                };
                let probe = hovered.or(pinned);
                if let Some((px, py)) = probe {
                    if let Some(rows) = img.readout_at(px, py) {
                        ui.monospace(format!("x: {px}   y: {py}"));
                        for (name, v) in rows {
                            ui.monospace(format!("{name}: {v:.6}"));
                        }
                    }
                    // Every channel of the file here, as stored (AOV debugging:
                    // depth, ids, motion without switching layers).
                    if let Some(all) = img.probe_at(px, py) {
                        ui.separator();
                        ui.label(egui::RichText::new("All channels").strong());
                        for (name, v) in all {
                            ui.monospace(format!("{name}: {v}"));
                        }
                    }
                } else {
                    ui.label("Hover the image to inspect pixels; click to pin.");
                }
            }
            None => {
                ui.label(&self.status);
            }
        }
    }

    /// The display controls of the viewport toolbar strip: exposure + gamma sliders, the
    /// display transform, channel isolate, layer / Cryptomatte pick, fit / 1:1, snapshot.
    /// These feed the GPU colour-pass uniforms (no CPU processing).
    fn ui_toolbar(&mut self, ui: &mut egui::Ui) {
        // Display dials. Ctrl+click or right-click resets one to its default (the shared
        // `egui_widgets_config::reset` gesture): exposure -> 0 EV, gamma -> 1.0, display -> None,
        // channel -> RGBA.
        ui.label("Exposure");
        egui_viewport_toolbar::exposure_control(
            ui,
            &mut self.controls.exposure,
            &mut self.controls.exposure_hold,
            -10.0..=10.0,
        );
        ui.separator();
        ui.label("Gamma");
        // A SlimSlider owns the reset gesture (no Ctrl-press preview of a wrong value).
        ui.add(egui_slider::SlimSlider::new(&mut self.controls.gamma, 0.1..=4.0).default_value(1.0).decimals(2))
            .on_hover_text("Display gamma dial — pow(c, 1/gamma) before the sRGB encode.\n1.0 = off (correct sRGB look).\nCtrl+click or right-click: reset to 1.0.");
        ui.separator();
        // One control for the display transform of the colour image: a built-in
        // tone operator + sRGB, or OCIO (configured in the Colour tab).
        ui.label("Display");
        let c = &mut self.controls;
        let ocio_on = c.colour.sel.on;
        let text = match (ocio_on, &c.colour.err) {
            (true, None) => "OCIO".to_owned(),
            (true, Some(_)) => "OCIO (error)".to_owned(),
            (false, _) => crate::transfer::tonemap::name(c.tonemap).to_owned(),
        };
        let mut pick: Option<Option<u32>> = None;
        let tip = match (&c.colour.err, c.colour.active()) {
            (Some(e), _) => format!("OCIO is not shown: {e}"),
            (None, Some(t)) if ocio_on => format!("OCIO: {}", t.key),
            _ => "Built-in: the tone operator, then the sRGB encode.".to_owned(),
        };
        let display = egui::ComboBox::from_id_salt("display_transform")
            .selected_text(text)
            .show_ui(ui, |ui| {
                for (op, name) in crate::transfer::tonemap::ALL {
                    if ui.selectable_label(!ocio_on && c.tonemap == op, name).clicked() {
                        pick = Some(Some(op));
                    }
                }
                ui.separator();
                if ui.selectable_label(ocio_on, "OCIO").on_hover_text("Configure in Window > Colour (OCIO).").clicked() {
                    pick = Some(None);
                }
            })
            .response
            .on_hover_text(format!(
                "{tip}\nDisplay transform of the colour image (after the exposure; isolated channels stay raw).\nT cycles; Ctrl+click or right-click: reset to None."
            ));
        if reset::requested(ui, &display) {
            pick = Some(Some(crate::transfer::tonemap::NONE));
        }
        if let Some(d) = pick {
            c.set_display(d);
        }
        ui.separator();
        for mode in [
            ChannelDisplay::Rgba,
            ChannelDisplay::R,
            ChannelDisplay::G,
            ChannelDisplay::B,
            ChannelDisplay::A,
            ChannelDisplay::Luminance,
        ] {
            let selected = self.controls.channel == mode;
            let what = match mode {
                ChannelDisplay::Rgba => "Show full colour (RGBA).",
                ChannelDisplay::R => "Isolate the Red channel as greyscale.",
                ChannelDisplay::G => "Isolate the Green channel as greyscale.",
                ChannelDisplay::B => "Isolate the Blue channel as greyscale.",
                ChannelDisplay::A => "Show the Alpha channel as greyscale.",
                ChannelDisplay::Luminance => "Show Rec.709 luminance as greyscale.",
            };
            let resp = ui
                .selectable_label(selected, mode.name())
                .on_hover_text(format!("{what}\nCtrl+click or right-click: reset to RGBA."));
            // The reset wins over the click it rides on.
            if reset::requested(ui, &resp) {
                self.controls.channel = ChannelDisplay::Rgba;
            } else if resp.clicked() {
                self.controls.channel = mode;
            }
        }
        ui.separator();
        // Layer picker: collect (name, label) first so the immutable backend
        // borrow ends before we mutate controls.view in the combo.
        let layers: Vec<(String, String)> = self
            .current_backend()
            .map(|img| {
                img.layers().iter().map(|g| (g.name.clone(), g.label().to_string())).collect()
            })
            .unwrap_or_default();
        if layers.len() > 1 {
            let layer = layers
                .iter()
                .find(|(n, _)| *n == self.controls.view.layer)
                .map(|(_, l)| l.clone())
                .unwrap_or_else(|| layers[0].1.clone());
            // A channel shown alone is named, so the combo never claims the whole layer.
            let cur = match &self.controls.view.channel {
                Some(c) => format!("{layer}: {c}"),
                None => layer,
            };
            let combo = egui::ComboBox::from_label("Layer")
                .selected_text(cur)
                .show_ui(ui, |ui| {
                    for (name, label) in &layers {
                        ui.selectable_value(&mut self.controls.view, Selection::layer(name), label);
                    }
                });
            combo
                .response
                .on_hover_text("Pick which channel group (layer) to display, e.g. beauty, depth, normals.");
            ui.separator();
        }
        // Cryptomatte pick: type + object name → greyscale coverage overlay.
        if let Some(cat) = self.crypto.clone() {
            let type_label = if self.controls.crypto_layer.is_empty() {
                "(off)".to_owned()
            } else {
                self.controls.crypto_layer.clone()
            };
            let type_combo = egui::ComboBox::from_label("Crypto")
                .selected_text(type_label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.controls.crypto_layer,
                        String::new(),
                        "(off)",
                    );
                    for layer in &cat.layers {
                        ui.selectable_value(
                            &mut self.controls.crypto_layer,
                            layer.name.clone(),
                            &layer.name,
                        );
                    }
                });
            type_combo
                .response
                .on_hover_text("Cryptomatte type — pick an object to show its coverage matte.");
            // If the type changed to one that does not contain the current name, clear name.
            if !self.controls.crypto_layer.is_empty() {
                let names = cat
                    .layer(&self.controls.crypto_layer)
                    .map(|l| l.names.clone())
                    .unwrap_or_default();
                if !self.controls.crypto_name.is_empty()
                    && !names.iter().any(|n| n == &self.controls.crypto_name)
                {
                    self.controls.crypto_name.clear();
                }
                let name_label = if self.controls.crypto_name.is_empty() {
                    "(pick)".to_owned()
                } else {
                    self.controls.crypto_name.clone()
                };
                let name_combo = egui::ComboBox::from_label("Matte")
                    .selected_text(name_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.controls.crypto_name,
                            String::new(),
                            "(pick)",
                        );
                        for n in &names {
                            ui.selectable_value(
                                &mut self.controls.crypto_name,
                                n.clone(),
                                n,
                            );
                        }
                    });
                name_combo
                    .response
                    .on_hover_text("Object / material name from the Cryptomatte manifest.");
            } else {
                self.controls.crypto_name.clear();
            }
            ui.separator();
        }
        let ctx = ui.ctx().clone();
        let keys = |id: &str| commands::find(id).map_or(String::new(), |c| commands::shortcut_text(&ctx, c));
        if ui
            .button("Fit")
            .on_hover_text(format!("Fit the whole image to the window ({}).", keys("view.fit")))
            .clicked()
        {
            self.controls.nav = Some(NavCmd::Fit);
        }
        if ui
            .button("1:1")
            .on_hover_text(format!("Zoom to 100% — one image pixel per screen pixel ({}).", keys("view.home")))
            .clicked()
        {
            self.controls.nav = Some(NavCmd::Home);
        }
        // Detect-sequence: expand the open single frame into its sequence
        // (forces scanseq, independent of `--auto`). Shown only for a single
        // image — a sequence is already a sequence. Falls back to the same
        // single image if no numbered siblings are present.
        if self.image.is_some() && self.seq.is_none() {
            ui.separator();
            let clicked = ui
                .button("Detect seq")
                .on_hover_text("Detect & load the image sequence this frame belongs to")
                .clicked();
            if clicked
                && let Some(path) = self.image.as_ref().map(|img| img.frame.path.clone())
            {
                self.open_dropped(path);
            }
        }
        // What shows under the image's alpha: composited by the present pass (`viewport::show`).
        ui.separator();
        if let Some(b) = egui_viewport_toolbar::backdrop_picker(ui, &self.controls.backdrop) {
            self.controls.backdrop = b;
        }
        // One snapshot button keeps the strip narrow: a click is File > Save view as PNG, the
        // right-click menu offers the HDR save. Both are the registry's actions (run by `build_ui`).
        ui.separator();
        let png = commands::find("file.save_view");
        let on = png.is_some_and(|c| commands::enabled(&c.action, self.avail()));
        let snap = ui
            .add_enabled(on, egui::Button::new(egui_phosphor::regular::CAMERA))
            .on_hover_text("Save the view as PNG as displayed · right-click: HDR (EXR / PQ PNG)");
        if snap.clicked() {
            self.queue.push(Action::SaveView);
        }
        snap.context_menu(|ui| {
            for id in ["file.save_view", "file.save_view_hdr"] {
                let Some(cmd) = commands::find(id) else { continue };
                if ui.add_enabled(commands::enabled(&cmd.action, self.avail()), egui::Button::new(cmd.label)).clicked() {
                    self.queue.push(cmd.action.clone());
                    ui.close();
                }
            }
        });
    }

    /// The Viewport tab: the image with the display toolbar strip over its top edge.
    /// The strip claims only its own rectangle (see `egui-viewport-toolbar`), so the viewport's
    /// scrub / pan / zoom bindings never see a press that landed on it. While it is docked at
    /// the top the viewport starts BELOW it, by a band that follows the roll (`Strip::band`):
    /// Fit / 100% centring and the pixel pick work on the area the strip leaves free, and the
    /// image slides with the roll. The wheel and raw right-drag are not routed by hit-testing,
    /// so the viewport is told when the pointer is over the strip (`viewport::Strip`, a value of
    /// this frame only); a strip floating over the image makes the HUD step aside instead.
    pub(crate) fn ui_viewport(&mut self, ui: &mut egui::Ui, gpu: &mut dyn TabGpu) {
        // The strip is drawn first: the band it takes is known before the viewport allocates.
        let rect = ui.available_rect_before_wrap();
        let (strip, band) = self.toolbar_strip(ui, rect);
        if band > 0.0 {
            // The viewport's own grey, so the band reads as part of it.
            let fill = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), band));
            ui.painter().rect_filled(fill, 0.0, viewport::VIEW_BG);
            ui.add_space(band);
        }
        self.display_viewport(ui, gpu, strip);
    }

    /// The display toolbar strip over `rect` (the viewport's area): what the viewport is told
    /// about it, and the band it takes off the top. The strip floats on a layer of its own, so it
    /// is registered as part of the viewport for the keys.
    fn toolbar_strip(&mut self, ui: &mut egui::Ui, rect: egui::Rect) -> (viewport::Strip, f32) {
        let mut state = self.config.toolbar;
        let widget = egui_viewport_toolbar::ViewportToolbar {
            edge: egui_viewport_toolbar::ToolbarEdge::Top,
            margin: 4.0,
            id_salt: "exr_view_toolbar",
            expand_tooltip: "Show the display controls",
            collapse_tooltip: "Hide the display controls",
            ..Default::default()
        };
        let bar = widget.show(ui, rect, &mut state, |ui| self.ui_toolbar(ui));
        commands::strip(ui.ctx(), &bar);
        self.config.toolbar = state;
        // Rolled out / up is saved at once; a drag to another height at exit, with the rest.
        if bar.toggled {
            self.config.save();
        }
        let strip = viewport::Strip { rect: bar.rect, hot: bar.contains_pointer };
        let band = strip.band(rect, widget.margin, widget.button_width);
        (strip, band)
    }

    /// Transport bar for a loaded sequence: play/pause, frame scrub slider,
    /// frame/range readout, fps field, loop toggle, prev/next step buttons.
    /// Hidden when no sequence is open (single-image view is unchanged).
    pub fn ui_transport(&mut self, ui: &mut egui::Ui) {
        let Some(s) = self.seq.as_mut() else {
            return;
        };
        ui.vertical(|ui| {
            // Row 1 — transport controls, frame readout, fps, loop.
            ui.horizontal(|ui| {
                use egui_phosphor::regular as icon;
                let ctx = ui.ctx().clone();
                // "Label (key)" from the registry, so the hover names the real key.
                let tip = |id: &str| {
                    commands::find(id).map_or(String::new(), |c| format!("{} ({})", c.label, commands::shortcut_text(&ctx, c)))
                };
                let pb = &mut s.playback;
                if ui.button(icon::SKIP_BACK).on_hover_text(tip("play.first")).clicked() {
                    pb.go_first();
                }
                if ui.button(icon::CARET_LEFT).on_hover_text(tip("play.prev")).clicked() {
                    pb.step_prev();
                }
                let back = if pb.playing && pb.reverse { icon::PAUSE } else { icon::REWIND };
                if ui.button(back).on_hover_text(tip("play.reverse")).clicked() {
                    pb.play(true);
                }
                let fwd = if pb.playing && !pb.reverse { icon::PAUSE } else { icon::PLAY };
                if ui.button(fwd).on_hover_text(tip("play.toggle")).clicked() {
                    pb.play(false);
                }
                if ui.button(icon::CARET_RIGHT).on_hover_text(tip("play.next")).clicked() {
                    pb.step_next();
                }
                if ui.button(icon::SKIP_FORWARD).on_hover_text(tip("play.last")).clicked() {
                    pb.go_last();
                }
                ui.separator();
                ui.label(format!("{} / [{}–{}]", pb.current(), pb.first(), pb.last()));
                ui.separator();
                ui.label("fps");
                let mut fps = ui
                    .add(
                        egui::DragValue::new(&mut pb.fps)
                            .speed(0.5)
                            .range(0.1..=240.0),
                    )
                    .on_hover_text("Playback rate in frames per second — drag to change.\nCtrl+click or right-click: reset to the default rate.");
                reset::apply(ui, &mut fps, &mut pb.fps, crate::persist::DEFAULT_FPS);
                // What really reaches the screen: green while every frame is shown,
                // red once the clock passes frames unshown; the measured rate in brackets.
                if pb.playing {
                    let now = std::time::Instant::now();
                    if let Some(shown) = s.meter.fps(now) {
                        let skipped = s.meter.skipped(now);
                        let colors = crate::theme::current(ui.ctx()).colors;
                        let col = if skipped == 0 { colors.status_ok } else { colors.status_error };
                        ui.label(egui::RichText::new(format!("{} ({shown:.1})", icon::CIRCLE)).color(col))
                            .on_hover_text(format!(
                                "Frames actually shown per second over the last second: {shown:.1}.\n\
                                 {skipped} skipped (the clock passed them before they could be drawn)."
                            ));
                    }
                }
                ui.separator();
                ui.checkbox(&mut pb.loop_, "loop")
                    .on_hover_text("Loop back to the first frame at the end of the sequence.");
            });

            // Row 2 — the cache-aware timeline (full width). Slots are POSITIONS
            // into the present set (`indices`), so gaps are already collapsed; the
            // tint shows each frame's decode-cache state, the label is the real
            // on-disk frame number, and a scrub seeks by position.
            let pos = s.playback.pos();
            let count = s.playback.frame_count();
            if count > 1 {
                let indices = s.playback.indices().to_vec();
                let seek = {
                    let cache = &s.cache;
                    egui_timeline::Timeline::new()
                        .height(20.0)
                        .show(
                            ui,
                            pos,
                            count,
                            |i| {
                                let f = indices[i];
                                if cache.contains(f) {
                                    Some(CACHE_LOADED)
                                } else if cache.is_in_flight(f) {
                                    Some(CACHE_INFLIGHT)
                                } else {
                                    None
                                }
                            },
                            |i| indices[i].to_string(),
                        )
                        .seek
                };
                if let Some(slot) = seek {
                    s.playback.set_pos(slot);
                }
            }
        });
    }
}

/// Static GPU identity for the status bar (captured once — it never changes for the
/// session): adapter name, backend, and the driver's reported VRAM budget (total).
struct GpuStatus {
    name: String,
    backend: String,
    vram_budget: Option<u64>,
}

/// The window output as it is: what the surface offers, what it shows, what the
/// display reports, and why the chosen output is not in use. Filled by the viewer
/// (`Viewer::apply_output` / `refresh_display`).
#[derive(Default)]
pub struct DisplayStatus {
    /// Outputs the surface can be configured for.
    pub outputs: Vec<crate::present::Output>,
    /// The output in use.
    pub active: crate::present::Output,
    /// The display's HDR figures (OS).
    pub info: wgpu::DisplayHdrInfo,
    /// Why the chosen output is not the one in use.
    pub err: Option<String>,
}

/// The live viewer: GPU context + egui integration + colour pass + dock layout.
struct Viewer {
    window: Arc<Window>,
    gpu: GpuContext,
    gpu_status: GpuStatus,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// The canvas everything is drawn into and the pass that encodes it for the surface.
    present: crate::present::PresentPass,
    /// The output last asked for (`config.display.output`); a new ask re-applies.
    asked: crate::present::Output,
    /// The monitor the window is on (a move to another one re-reads its HDR figures).
    monitor: Option<String>,
    dock_state: DockState<DockTab>,
    app_state: AppState,
    /// True until the first frame has run the pending-path load.
    first_frame: bool,
    /// egui's requested repaint delay from the last frame (root viewport). Drives
    /// on-demand redraw in `about_to_wait`: `ZERO` = repaint now, finite = wake by
    /// then, `MAX`/absent = nothing pending → sleep (`ControlFlow::Wait`).
    repaint_after: Duration,
    /// Deadline of the next playback frame. The pacing clock (`about_to_wait`)
    /// only asks for a redraw once this has passed, then advances it by whole
    /// frame periods — so a slow frame DROPS the frames it missed instead of
    /// stretching the clip, and an idle loop sleeps until the deadline.
    next_frame: Instant,
    /// The window is minimised (a 0-area resize) — there is nothing to draw and
    /// no vsync to throttle us, so redraws must stop entirely.
    minimised: bool,
    /// The compositor reports the window as fully occluded — same reasoning.
    occluded: bool,
}

impl Viewer {
    fn new(
        window: Arc<Window>,
        initial: Option<PathBuf>,
        config: Config,
        auto_detect: bool,
        wake: scancache::Wake,
    ) -> Result<Self> {
        let gpu = GpuContext::new(window.clone())?;
        // Capture the GPU identity once for the status bar (static for the session).
        let gpu_status = {
            let info = gpu.adapter.get_info();
            GpuStatus {
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                vram_budget: wgpu_info::vram_budget_bytes(&gpu.adapter),
            }
        };

        let egui_ctx = egui::Context::default();
        theme::install_fonts(&egui_ctx);
        theme::set(&egui_ctx, &theme::Theme::dark_theme(), &config.fonts);

        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // egui and the tiles draw into the canvas (`present::CANVAS_FORMAT`, not sRGB,
        // so egui stores its sRGB-authored colours as they are); the present pass
        // dithers once, for the surface's bit depth, so egui does not.
        let options = egui_wgpu::RendererOptions { dithering: false, ..Default::default() };
        let mut egui_renderer = egui_wgpu::Renderer::new(&gpu.device, crate::present::CANVAS_FORMAT, options);
        // The tiled display path (atlas + pipeline) lives in egui's callback
        // resources, where its paint callback binds it.
        let share = config.budgets.clamped().vram;
        let vram = gpu_status.vram_budget.map(|v| (v as f64 * share) as u64).unwrap_or_else(|| {
            log::warn!("the driver reports no VRAM budget; tile atlas limited to 1 GiB");
            TILE_VRAM_FALLBACK
        });
        egui_renderer.callback_resources.insert(crate::tiles::render::TileGpu::new(&gpu.device, vram));
        let present = crate::present::PresentPass::new(&gpu.device, gpu.config.format, gpu.output);
        let dock_state = dock::restore(config.dock.as_deref());
        let mut app_state = AppState::new(initial, config, auto_detect, wake);
        app_state
            .controls
            .colour
            .set_filterable(gpu.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE));

        let asked = gpu.output;
        let mut viewer = Self {
            window,
            gpu,
            egui_state,
            egui_renderer,
            present,
            asked,
            monitor: None,
            dock_state,
            gpu_status,
            app_state,
            first_frame: true,
            // Sleep by default until egui / playback / decodes ask for a repaint.
            repaint_after: Duration::MAX,
            next_frame: Instant::now(),
            minimised: false,
            occluded: false,
        };
        viewer.apply_output();
        Ok(viewer)
    }

    /// Configure the surface for the output the user asked for
    /// (`config.display.output`) and rebuild what depends on it (the present pass,
    /// the OCIO transform's HDR mode). An output the surface does not offer is an
    /// error on the status line and in Settings; the surface keeps what it had.
    fn apply_output(&mut self) {
        let ask = self.app_state.config.display.output;
        self.asked = ask;
        let err = self.gpu.set_output(ask).err();
        if err.is_some() && !self.gpu.outputs().contains(&self.gpu.output) {
            // The output in use is gone too (HDR switched off in the OS): the surface
            // cannot stay configured for it, SDR 8-bit is always there.
            if let Err(e) = self.gpu.set_output(crate::present::Output::Sdr8) {
                log::error!("output: even SDR 8-bit failed: {e}");
            }
        }
        if let Some(e) = &err {
            log::warn!("output: {e}");
            self.app_state.status = format!("{e}; showing {}", self.gpu.output.label());
        }
        let mut next = crate::present::PresentPass::new(
            &self.gpu.device,
            self.gpu.config.format,
            self.gpu.output,
        );
        next.inherit_canvas(&self.gpu.device, &mut self.present);
        self.present = next;
        self.app_state.controls.colour.set_hdr(self.gpu.output.is_hdr());
        self.app_state.display.err = err;
        self.refresh_display();
    }

    /// After a move to another monitor or on focus (HDR may have been switched in
    /// the OS): re-read the display, and re-apply the output when the one in use is
    /// no longer offered or the one asked for has become available.
    fn recheck_display(&mut self) {
        self.refresh_display();
        let d = &self.app_state.display;
        let ask = self.app_state.config.display.output;
        let lost = !d.outputs.contains(&d.active);
        let now_there = d.active != ask && d.outputs.contains(&ask);
        if lost || now_there {
            self.apply_output();
        }
    }

    /// Re-read what the surface offers and what the display reports.
    fn refresh_display(&mut self) {
        self.monitor = self.window.current_monitor().and_then(|m| m.name());
        let d = &mut self.app_state.display;
        d.outputs = self.gpu.outputs();
        d.active = self.gpu.output;
        d.info = self.gpu.hdr_info();
        log::debug!("display {:?}: {:?}", self.monitor, d.info);
    }

    /// Nothing of this window is on screen: drawing would be wasted work, and
    /// (with no vsync to block on) would spin the CPU at full speed.
    fn hidden(&self) -> bool {
        self.minimised || self.occluded
    }

    /// Snapshot the LIVE viewer state into the config and persist it.
    ///
    /// Reads the current window outer-position + inner-size into `config.window`
    /// (physical px), copies the display controls (exposure / gamma / channel),
    /// and the sequence playback prefs (fps / loop) when a sequence is open, then
    /// saves (best-effort). Called on Esc and on close so geometry + controls
    /// survive the next launch. Never panics: a failed geometry query just skips
    /// the window field, and `save` swallows IO errors.
    fn save_state(&mut self) {
        // Live window geometry (physical px). `outer_position` can fail on some
        // platforms/window states — keep the previous saved geometry if so. In
        // cinema/fullscreen the live size is the whole monitor, which we must NOT
        // persist (it would relaunch maximised over the screen) — keep the last
        // windowed geometry instead.
        if self.window.fullscreen().is_none()
            && let Ok(pos) = self.window.outer_position()
        {
            let size = self.window.inner_size();
            self.app_state.config.window = Some(crate::persist::WindowGeom {
                x: pos.x,
                y: pos.y,
                w: size.width,
                h: size.height,
            });
        }
        // Display controls.
        let c = &self.app_state.controls;
        self.app_state.config.exposure = c.exposure;
        self.app_state.config.exposure_hold = c.exposure_hold;
        self.app_state.config.ocio_dir = c.colour.dir_state().clone();
        self.app_state.config.gamma = c.gamma;
        self.app_state.config.channel = channel_to_u8(c.channel);
        self.app_state.config.tonemap = c.tonemap;
        self.app_state.config.backdrop = c.backdrop;
        self.app_state.config.ocio = c.colour.sel.clone();
        // Sequence playback prefs (only meaningful when a sequence is loaded).
        if let Some(s) = &self.app_state.seq {
            self.app_state.config.fps = s.playback.fps;
            self.app_state.config.loop_ = s.playback.loop_;
        }
        match egui_dock_layout::to_blob(&self.dock_state) {
            Ok(blob) => self.app_state.config.dock = Some(blob),
            Err(e) => log::warn!("dock layout not saved: {e}"),
        }
        self.app_state.config.save();
    }

    /// Forward a window event to egui; returns egui's response (`consumed` +
    /// `repaint`). The caller wakes the loop on `repaint` so input refreshes the
    /// UI under the on-demand `ControlFlow::Wait` redraw policy.
    fn on_window_event(&mut self, event: &WindowEvent) -> egui_winit::EventResponse {
        self.egui_state.on_window_event(&self.window, event)
    }

    /// Render a single frame: load on the first frame, run egui (menu/toolbar/
    /// dock with the GPU viewport), tessellate, upload, draw to the surface.
    fn render(&mut self) -> Result<()> {
        if self.first_frame {
            self.app_state.load_pending();
            self.first_frame = false;
        }

        // An output asked for in Settings last frame is applied before this frame
        // draws, so tiles and surface never disagree for a frame.
        if self.app_state.config.display.output != self.asked {
            self.apply_output();
        }
        self.app_state.controls.target = self.app_state.target();

        // Install a background open that finished since the last frame.
        self.app_state.tick_open();

        // Per-frame sequence bookkeeping: advance the clock, drain decodes,
        // (re)queue the current frame + neighbours. Never blocks on decode.
        self.app_state.tick_sequence();

        // Free the GPU resources of any scope panel whose tab is currently closed
        // (closed via the dock's tab-close button). egui retains a
        // native texture until `free_texture`, so an idle scope would otherwise
        // keep its output texture + egui id resident. Safe: a closed scope is not
        // ticked this frame, so its id is not referenced by any paint job.
        // Idempotent — a no-op once released; the panel re-registers on reopen.
        for scope in DockTab::SCOPES {
            if self.dock_state.find_tab(&scope).is_none() {
                let panel: &mut dyn GpuPanel = match scope {
                    DockTab::Histogram => &mut self.app_state.scopes.histogram,
                    DockTab::Waveform => &mut self.app_state.scopes.waveform,
                    DockTab::Vectorscope => &mut self.app_state.scopes.vectorscope,
                    _ => continue,
                };
                panel.release(&mut self.egui_renderer);
            }
        }

        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        self.present.prepare_canvas(
            &self.gpu.device,
            &self.gpu.queue,
            self.gpu.config.width,
            self.gpu.config.height,
        );
        if let Some(gpu) = self
            .egui_renderer
            .callback_resources
            .get_mut::<crate::tiles::render::TileGpu>()
        {
            gpu.canvas_bytes = self.present.canvas_reserve(
                &self.gpu.device,
                self.gpu.config.width,
                self.gpu.config.height,
            );
        }
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();

        // Split disjoint borrows so the egui closure can use the GPU bits while
        // egui_state is borrowed separately for input handling.
        let dock_state = &mut self.dock_state;
        let app_state = &mut self.app_state;
        let mut viewport_gpu = ViewportGpu {
            device: &self.gpu.device,
            queue: &self.gpu.queue,
            renderer: &mut self.egui_renderer,
        };
        let gpu_status = &self.gpu_status;

        let mut full_output = ctx.run_ui(raw_input, |ui| {
            build_ui(ui, dock_state, app_state, &mut viewport_gpu, gpu_status);
            // After every panel, on each exit path of `build_ui`: the modal paints on top.
            match app_state.file_pick.poll(ui.ctx()) {
                Some((Pick::Open, path)) => app_state.open_picked(path),
                Some((Pick::OcioConfig, path)) => app_state.controls.colour.set_config(path.display().to_string()),
                Some((pick @ (Pick::SaveView | Pick::SaveViewHdr), path)) => {
                    let max_dim = viewport_gpu.device.limits().max_texture_dimension_2d;
                    let hdr = (pick == Pick::SaveViewHdr).then_some(app_state.file_pick.png);
                    app_state.save_view(&path, max_dim, hdr);
                }
                None => {}
            }

        });

        // Every frame, even one that is skipped below: a frame that showed no backdrop must not
        // present the last one.
        hand_backdrop(&ctx, &mut self.present.backdrop);
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        // Stash egui's requested repaint delay (min across viewports) so
        // `about_to_wait` schedules an on-demand wake instead of busy-spinning:
        // ZERO = an animation wants the next frame now; finite = wake by then;
        // MAX/empty = egui is idle and we may sleep.
        self.repaint_after = full_output
            .viewport_output
            .values()
            .map(|o| o.repaint_delay)
            .min()
            .unwrap_or(Duration::MAX);

        // egui hands the texture deltas over to be CONSUMED: uploads and frees
        // below drain them, and epaint asserts (debug) that none is dropped.
        let mut textures = std::mem::take(&mut full_output.textures_delta);
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        // Apply egui's texture UPLOADS before touching the surface (V3).
        //
        // These deltas are egui's only record of what it has uploaded: a frame
        // that skips them leaves the renderer's texture set behind egui's model,
        // and the NEXT delta — a partial update (`pos: Some(..)`) of a texture
        // that was never allocated, or a write past a font atlas that was never
        // grown — is a panic or a corrupted atlas. They are plain queue writes,
        // so they are valid whether or not a surface frame is acquired below.
        // egui 0.36 batches several deltas per texture id; apply them in order.
        for (id, deltas) in textures.set.drain() {
            for delta in &deltas {
                self.egui_renderer
                    .update_texture(&self.gpu.device, &self.gpu.queue, id, delta);
            }
        }

        // Acquire the next surface frame (wgpu 30 `CurrentSurfaceTexture` enum).
        // Every skip path still frees what egui asked to free — a skipped `free`
        // leaks the texture for the rest of the session.
        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu
                    .surface
                    .configure(&self.gpu.device, &self.gpu.config);
                free_textures(&mut self.egui_renderer, &mut textures);
                // The reconfigured surface has nothing in it: ask for the repaint
                // that this skipped frame owed.
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                free_textures(&mut self.egui_renderer, &mut textures);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::warn!("surface validation error; skipping frame");
                free_textures(&mut self.egui_renderer, &mut textures);
                return Ok(());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let canvas = self
            .present
            .canvas(&self.gpu.device, self.gpu.config.width, self.gpu.config.height)
            .clone();

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("exr-view egui encoder"),
            });

        let cmd_bufs = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exr-view egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &canvas,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // The linear grey the sRGB surface was cleared to, in the
                        // canvas encoding.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(crate::transfer::oetf(0.05)),
                            g: f64::from(crate::transfer::oetf(0.05)),
                            b: f64::from(crate::transfer::oetf(0.06)),
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // egui-wgpu's `render` needs a `'static` render pass.
            let mut pass = pass.forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        // The target the tiles drew with this frame, even if Settings moved it since.
        self.present.draw(&self.gpu.queue, &mut encoder, &view, self.app_state.controls.target);

        free_textures(&mut self.egui_renderer, &mut textures);

        self.gpu.queue.submit(
            cmd_bufs
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        self.gpu.queue.present(frame); // wgpu 30: Queue::present
        Ok(())
    }
}

/// Give the present pass the backdrop the UI showed this frame (`viewport::show` publishes it
/// through `egui_display::show_backdrop`), or clear it when the frame showed none.
fn hand_backdrop(ctx: &egui::Context, slot: &mut Option<egui_display::BackdropUniform>) {
    *slot = egui_display::take_backdrop(ctx);
}

/// Release the egui textures this frame's output retired, draining the list.
/// Called on EVERY exit from `render` (including the skipped-frame paths) — a
/// skipped `free` leaks a texture and its bind group for the rest of the session.
fn free_textures(renderer: &mut egui_wgpu::Renderer, delta: &mut egui::TexturesDelta) {
    for id in delta.free.drain() {
        renderer.free_texture(&id);
    }
}

/// Run the keys pressed this frame. Every bare-letter shortcut is suppressed while a widget wants
/// the keyboard (a text field, a slider being typed into): otherwise typing "z" in a filter box
/// toggled fullscreen and " " played the sequence.
fn read_keys(ctx: &egui::Context, app_state: &mut AppState, dock_state: &mut DockState<DockTab>) {
    let typing = ctx.egui_wants_keyboard_input() || app_state.file_pick.is_open();
    for action in commands::pressed(ctx, typing, app_state.avail()) {
        run(action, app_state, dock_state, ctx);
    }
}

/// The transport bar (sequence playback), a keyboard panel of its own: the frame keys work over
/// it, and a click on it makes it the owner.
fn transport_bar(ui: &mut egui::Ui, app_state: &mut AppState) {
    let bar = egui::Panel::bottom("transport_bar").show(ui, |ui| {
        app_state.ui_transport(ui);
    });
    commands::register(ui, commands::Scope::Timeline, bar.response.rect);
}

/// Build the egui UI for this frame (menu bar, toolbar, status bar, dock).
///
/// Free function (not a method) so the `&mut Viewer` fields can be split into
/// disjoint borrows before the egui closure captures them.
fn build_ui(
    ui: &mut egui::Ui,
    dock_state: &mut DockState<DockTab>,
    app_state: &mut AppState,
    gpu: &mut dyn TabGpu,
    gpu_status: &GpuStatus,
) {
    let ctx = ui.ctx().clone();
    app_state.viewport_active = app_state.panels_hidden || dock_state.iter_all_nodes().any(|(_, node)| {
        node.get_leaf().is_some_and(|leaf| leaf.tabs.get(leaf.active.0) == Some(&DockTab::Viewport))
    });

    // The OpenFX effect's questions, finished edits, dialogs and re-render requests (UI thread).
    #[cfg(feature = "ofx")]
    app_state.effects.tick(&ctx);

    // Every key goes through the command registry, here, before any panel: so the
    // keys work with the panels hidden (cinema mode) and a panel click cannot eat them.
    read_keys(&ctx, app_state, dock_state);
    // Panels hidden → draw ONLY the viewport (independent of fullscreen).
    if app_state.panels_hidden {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                // The dock is not drawn, so the viewport registers itself: the keys keep working.
                commands::panel(ui, commands::Scope::Viewport, |ui| {
                    app_state.display_viewport(ui, gpu, viewport::Strip::NONE);
                });
            });
        return;
    }

    // The menu bar is `commands::MENU`; items look up their label, keys and state in
    // the registry. Its `Dyn` slots are drawn here. Work that needs `app_state`
    // mutably beyond the config (open a recent file, clear the list) is deferred
    // until the bar has been drawn.
    let mut recent_to_open: Option<PathBuf> = None;
    let mut clear_recents = false;
    let tabs: Vec<DockTab> = COMMANDS
        .iter()
        .filter_map(|c| match &c.action {
            Action::Tab(tab) if dock_state.find_tab(tab).is_some() => Some(tab.clone()),
            _ => None,
        })
        .collect();
    // The tab context menu (inside the dock, where `dock_state` is borrowed) lists what is closed.
    app_state.open_tabs.clone_from(&tabs);
    let (fullscreen, panels_hidden, channel) =
        (app_state.fullscreen, app_state.panels_hidden, app_state.controls.channel);
    let avail = app_state.avail();
    // Disjoint borrows: the `Dyn` slots edit these config fields, the layout bar the
    // named layouts.
    let crate::persist::Config { recents, playhead_alpha, preload_strategy, layouts, .. } = &mut app_state.config;
    let layout_bar = &mut app_state.layout_bar;
    let mut layout_pick = None;
    let mut clicked = None;
    egui::Panel::top("menu_bar").show(ui, |ui| {
        let item = |id: &'static str| {
            let Some(cmd) = commands::find(id) else {
                // `menu_and_registry_agree` makes this unreachable; a visible marker
                // beats a row that lies about its name.
                return egui_menubar::ItemState::from_label(format!("?{id}")).enabled(false);
            };
            let checked = match &cmd.action {
                Action::Fullscreen => fullscreen,
                Action::HidePanels => panels_hidden,
                Action::Isolate(mode) => channel == *mode,
                Action::Tab(tab) => tabs.contains(tab),
                _ => false,
            };
            egui_menubar::ItemState::from_label(cmd.label)
                .shortcut(commands::shortcut_text(&ctx, cmd))
                .checked(checked)
                .enabled(commands::enabled(&cmd.action, avail))
        };
        let slot = |id: &'static str, ui: &mut egui::Ui| match id {
            "file.recent" => {
                ui.menu_button("Open Recent", |ui| {
                    if recents.is_empty() {
                        ui.add_enabled(false, egui::Button::new("(none)"));
                        return;
                    }
                    for path in recents.iter() {
                        // The file name; the full path on hover.
                        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("<unnamed>");
                        if ui.button(name).on_hover_text(path.display().to_string()).clicked() {
                            recent_to_open = Some(path.clone());
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button("Clear recents").clicked() {
                        clear_recents = true;
                        ui.close();
                    }
                });
            }
            "view.playhead" => {
                ui.label("Playhead line (over image)");
                ui.add(egui_slider::SlimSlider::new(&mut *playhead_alpha, 0.0..=1.0).text("opacity").default_value(0.0).decimals(2))
                    .on_hover_text("Thin white vertical line at the current frame, over the image. 0 = off.\nCtrl+click or right-click: reset to off.");
            }
            "view.preload" => {
                ui.label("Preload strategy");
                // Which frames the cache loads first around the playhead.
                for strat in scancache::Strategy::ALL {
                    ui.radio_value(&mut *preload_strategy, strat, strat.label());
                }
            }
            other => {
                ui.add_enabled(false, egui::Button::new(format!("?{other}")));
            }
        };
        // The named layouts sit at the right end of the menu row.
        let layouts_ui = |ui: &mut egui::Ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                match layout_bar.show(ui, layouts, dock_state) {
                    Ok(pick) => layout_pick = pick,
                    Err(e) => log::warn!("dock layout: {e}"),
                }
            });
        };
        clicked = egui_menubar::MenuBar::new(commands::MENU).show_full(ui, item, slot, layouts_ui);
    });
    if let Some(state) = layout_pick {
        *dock_state = state;
    }
    if let Some(cmd) = clicked.and_then(commands::find) {
        run(cmd.action.clone(), app_state, dock_state, &ctx);
    }
    if let Some(path) = recent_to_open {
        app_state.open_chosen(path);
    }
    if clear_recents {
        app_state.config.recents.clear();
        app_state.config.save();
    }

    egui::Panel::bottom("status_bar").show(ui, |ui| {
        let text_col = theme::current(&ctx).colors.text_secondary;
        // Snapshot the readouts into owned locals so the section closures don't
        // borrow `app_state` (we also borrow its config mutably for the layout).
        let status = app_state.status.clone();
        let frame_info = app_state.seq.as_ref().map(|s| {
            (
                s.playback.current(),
                s.playback.first(),
                s.playback.last(),
                s.playback.playing,
            )
        });
        // One lock (`Status`) instead of separate `resident_count`/`total`/
        // `playable_from_here` calls that could disagree with each other.
        let cache_info = app_state.seq.as_ref().map(|s| s.cache.status());
        let gpu_name = gpu_status.name.clone();
        let gpu_backend = gpu_status.backend.clone();
        let output = app_state.display.active;
        let target = app_state.target();
        let vram_budget = gpu_status.vram_budget;
        let atlas = gpu.gpu().and_then(|g| {
            g.renderer.callback_resources.get::<crate::tiles::render::TileGpu>().map(|t| t.atlas.bytes())
        });
        let (tiles_ram, tiles_budget) = app_state.display_gpu.tiles_ram();
        let tiles_pending = app_state.display_gpu.tiles_pending();
        let display_path = app_state.display_gpu.path();

        let crate::persist::Config { status_bar_layout: layout, status_bar_locked: locked, .. } = &mut app_state.config;
        // Right-click on the bar: lock the splitters / put the widths back. The sense sits under
        // the sections, so their own widgets and the splitters still take their clicks first.
        let bar = ui.interact(ui.available_rect_before_wrap(), ui.id().with("status_menu"), egui::Sense::click());
        bar.context_menu(|ui| {
            ui.checkbox(locked, "Lock section widths").on_hover_text("Drag-resizing the sections is off while locked.");
            if ui.button("Reset section widths").clicked() {
                *layout = Default::default();
                ui.close();
            }
        });
        let mut sections = vec![
            // Frame / range / play state.
            egui_statusbar::Section::new(210.0, move |ui| match frame_info {
                Some((cur, first, last, playing)) => {
                    let st = if playing { egui_phosphor::regular::PLAY } else { egui_phosphor::regular::PAUSE };
                    ui.label(format!("{st} frame {cur} / [{first}–{last}]"));
                }
                None => {
                    ui.label("single image");
                }
            }),
            // Decode-cache fill N/M and its RAM held / live cap (stays visible at
            // 100% with a ✓). The cap moves: it is the RAM budget minus what the
            // tile cache and the OpenFX inputs hold (`RamPartitions::frames`).
            egui_statusbar::Section::new(300.0, move |ui| match &cache_info {
                Some(st) if st.total > 0 => {
                    let (res, tot) = (st.resident, st.total);
                    let frac = res as f32 / tot as f32;
                    let mark = if res >= tot { format!("{} ", egui_phosphor::regular::CHECK) } else { String::new() };
                    let txt = format!(
                        "cache {mark}{res}/{tot} · {} / {}",
                        humanize_bytes(st.used_bytes as u64),
                        humanize_bytes(st.budget_bytes as u64)
                    );
                    // `playable` is the honest pre-stutter number (contiguous
                    // resident run from the playhead along the play direction) —
                    // `resident`/`total` alone can look full while gaps ahead of
                    // the cursor would still stall playback.
                    ui.add(egui::ProgressBar::new(frac).desired_width(260.0).text(txt))
                        .on_hover_text(format!(
                            "{} frames playable from here\n{} per frame on average\n{} decoding\nRAM held / the cache's live budget",
                            st.playable,
                            humanize_bytes(st.avg_frame_bytes as u64),
                            st.decoding
                        ));
                }
                _ => {
                    ui.label("cache —");
                }
            }),
            // GPU adapter · backend · output (SDR white on an HDR one, the peak on HLG).
            egui_statusbar::Section::new(260.0, move |ui| {
                let out = match output {
                    crate::present::Output::Hlg => {
                        format!("{} · {:.0} nit white · {:.0} nit peak", output.label(), target.white, target.peak)
                    }
                    o if o.is_hdr() => format!("{} · {:.0} nit white", o.label(), target.white),
                    o => o.label().to_owned(),
                };
                ui.label(format!("{gpu_name} · {gpu_backend} · {out}"))
                    .on_hover_text("The window output: Settings > Display.");
            }),
            // The display path of the last frame: resident GPU texture or CPU tiles.
            egui_statusbar::Section::new(110.0, move |ui| match display_path {
                Some(path) => {
                    ui.label(path.label()).on_hover_text(path.hint());
                }
                None => {
                    ui.label("display —");
                }
            }),
            // Tile atlas VRAM: held / budget (the driver's total on hover).
            egui_statusbar::Section::new(150.0, move |ui| {
                let text = match atlas {
                    Some((used, budget)) => format!("VRAM {} / {}", humanize_bytes(used), humanize_bytes(budget)),
                    None => "VRAM —".to_owned(),
                };
                let total = vram_budget.map_or("unknown".to_owned(), humanize_bytes);
                ui.label(text).on_hover_text(format!("tile atlas pages / its budget; driver budget {total}"));
            }),
            // Tile RAM cache and tiles in flight.
            egui_statusbar::Section::new(170.0, move |ui| {
                let busy = if tiles_pending > 0 { format!(" · {tiles_pending} pending") } else { String::new() };
                ui.label(format!(
                    "tiles {} / {}{busy}",
                    humanize_bytes(tiles_ram as u64),
                    humanize_bytes(tiles_budget as u64)
                ))
                .on_hover_text("RAM tile cache held / its budget; tiles still being produced");
            }),
            // File / dimensions / channels (the flex tail).
            egui_statusbar::Section::new(0.0, move |ui| {
                ui.colored_label(text_col, &status);
            }),
        ];
        egui_statusbar::StatusBar::new().locked(*locked).show(ui, layout, &mut sections);
    });

    // Transport bar (sequence playback) sits just above the status bar; only
    // shown when a sequence is loaded — single-image view is unchanged.
    // Playback keys are handled at the top of `build_ui` (so they work in cinema
    // mode where this bar is hidden).
    if app_state.seq.is_some() {
        transport_bar(ui, app_state);
    }

    egui::CentralPanel::default_margins().show(ui, |ui| {
        DockArea::new(dock_state)
            .style({
                let mut style = Style::from_egui(ui.style().as_ref());
                style.tab_bar.height = app_state.config.fonts.clamped().dock_tab;
                style
            })
            .show_inside(
                ui,
                &mut DockTabViewer {
                    state: app_state,
                    gpu,
                },
            );
    });
    // What the panels asked for while the dock was borrowed (snapshot menu, tab context menu).
    for action in std::mem::take(&mut app_state.queue) {
        run(action, app_state, dock_state, &ctx);
    }
}

/// Run `action` (a menu click or a key, `commands::COMMANDS`).
fn run(action: Action, app_state: &mut AppState, dock_state: &mut DockState<DockTab>, ctx: &egui::Context) {
    match action {
        Action::Open => {
            // Start where the file shown last lives, whichever way it was opened.
            let dir = app_state.config.last_file.as_deref().and_then(Path::parent).map(Path::to_path_buf);
            // The dialog's box shows the option as it is now (Settings may have changed it).
            app_state.file_pick.seq = app_state.config.open_seq;
            app_state.file_pick.open(dir);
        }
        Action::SaveView => {
            let dir = app_state.config.last_file.as_deref().and_then(Path::parent).map(Path::to_path_buf);
            let name = app_state
                .current_backend()
                .map(|img| format!("{}.png", img.frame.path.file_stem().and_then(|s| s.to_str()).unwrap_or("view")))
                .unwrap_or_else(|| "view.png".to_owned());
            app_state.file_pick.save_view(dir, &name);
        }
        Action::SaveViewHdr => {
            let dir = app_state.config.last_file.as_deref().and_then(Path::parent).map(Path::to_path_buf);
            let name = app_state
                .current_backend()
                .map(|img| format!("{}.hdr.png", img.frame.path.file_stem().and_then(|s| s.to_str()).unwrap_or("view")))
                .unwrap_or_else(|| "view.hdr.png".to_owned());
            app_state.file_pick.save_view_hdr(dir, &name);
        }
        Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        Action::Nav(cmd) => app_state.controls.nav = Some(cmd),
        // Borderless OS fullscreen through egui-winit (no manual `Window::set_fullscreen`),
        // independent of the panel toggle below.
        Action::Fullscreen => {
            app_state.fullscreen = !app_state.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(app_state.fullscreen));
        }
        Action::HidePanels => app_state.panels_hidden = !app_state.panels_hidden,
        Action::PlayPause | Action::PlayReverse | Action::StepPrev | Action::StepNext | Action::First | Action::Last => {
            if let Some(s) = app_state.seq.as_mut() {
                let pb = &mut s.playback;
                match action {
                    Action::PlayPause => pb.play(false),
                    Action::PlayReverse => pb.play(true),
                    Action::StepPrev => pb.step_prev(),
                    Action::StepNext => pb.step_next(),
                    Action::First => pb.go_first(),
                    _ => pb.go_last(),
                }
            }
        }
        Action::FilePrev | Action::FileNext => {
            let step = if action == Action::FileNext { 1 } else { -1 };
            match app_state.sibling(step) {
                Some(path) => app_state.open_path(path),
                None => app_state.status = "No further EXR in this folder".to_owned(),
            }
        }
        Action::CycleDisplay => {
            let c = &mut app_state.controls;
            c.set_display(c.next_display());
        }
        Action::Isolate(mode) => {
            let c = &mut app_state.controls.channel;
            *c = if *c == mode { ChannelDisplay::Rgba } else { mode };
        }
        Action::ResetLayout => *dock_state = default_dock_state(),
        // Open the tab (in the focused leaf), or bring an open one forward; closing is the tab's X.
        Action::Tab(tab) => match dock_state.find_tab(&tab) {
            Some(path) => {
                let node = path.node_path();
                if let Err(e) = dock_state.set_active_tab(path) {
                    log::warn!("cannot bring {tab:?} forward: {e}");
                }
                dock_state.set_focused_node_and_surface(node);
            }
            None => dock_state.push_to_focused_leaf(tab),
        },
    }
}

/// A tinted collapsible section of a panel (`egui-titlebar`); `Some` of the body's value when it
/// is open. The open flag lives in egui's temp memory under the panel's id: a session-only
/// fold, like the channel tree's.
fn section<R>(ui: &mut egui::Ui, label: &str, tint: egui::Color32, body: impl FnOnce(&mut egui::Ui) -> R) -> Option<R> {
    let key = ui.id().with(("section_open", label));
    let open = ui.data(|d| d.get_temp::<bool>(key)).unwrap_or(true);
    let shown = egui_titlebar::CollapsingSection::new(label).id_salt(label).open(open).tint(tint, 0.16).show(ui, body);
    if shown.header.open != open {
        ui.data_mut(|d| d.insert_temp(key, shown.header.open));
    }
    shown.body_returned
}

/// The file `step` places from `cur` in the sorted `files`, not counting `cur`'s own
/// group `own` (the other frames of its sequence); `None` past either end.
fn neighbour(files: &[PathBuf], cur: &Path, step: isize, own: &HashSet<PathBuf>) -> Option<PathBuf> {
    let at = files.partition_point(|f| f.as_path() < cur);
    let skip = |f: &&PathBuf| f.as_path() != cur && !own.contains(*f);
    if step >= 0 {
        files.get(at..)?.iter().filter(skip).nth(step.unsigned_abs().checked_sub(1)?).cloned()
    } else {
        files.get(..at)?.iter().rev().filter(skip).nth(step.unsigned_abs() - 1).cloned()
    }
}

/// Settings > Memory.
fn budgets_ui(ui: &mut egui::Ui, budgets: &mut Budgets) {
    const RESET: &str = "\nCtrl+click or right-click: reset.";
    let default = Budgets::default();
    egui_prefs2::section_header(ui, "Caches (next launch)");
    egui_prefs2::section_grid(ui, "budgets_launch", |ui| {
        ui.label("RAM share")
            .on_hover_text(format!("Share of installed RAM for the frame cache and for the tile cache, each.{RESET}"));
        ui.add(egui_slider::SlimSlider::new(&mut budgets.ram, Budgets::RAM).default_value(default.ram).decimals(2));
        ui.end_row();
        ui.label("VRAM share").on_hover_text(format!("Share of the GPU's memory for the tile atlas.{RESET}"));
        ui.add(egui_slider::SlimSlider::new(&mut budgets.vram, Budgets::VRAM).default_value(default.vram).decimals(2));
        ui.end_row();
    });
    egui_prefs2::section_header(ui, "Display");
    egui_prefs2::section_grid(ui, "budgets_live", |ui| {
        ui.label("Upload per frame").on_hover_text(format!(
            "Tile bytes sent to the GPU per frame: lower keeps panning smooth, higher fills in faster.{RESET}"
        ));
        ui.add(egui_slider::SlimSlider::new(&mut budgets.upload_mib, Budgets::UPLOAD).default_value(default.upload_mib).suffix(" MiB"));
        ui.end_row();
    });
}

/// Settings > Display: the output (applied at once by the viewer), SDR white, the
/// display peak (HLG), and what the OS reports about the display.
fn display_ui(ui: &mut egui::Ui, prefs: &mut crate::persist::DisplayPrefs, status: &DisplayStatus, target: crate::present::Target) {
    let state = egui_display::DisplayState {
        output: status.active,
        target,
        info: status.info.clone(),
        error: status.err.clone(),
        available: status.outputs.clone(),
    };
    egui_display::settings_ui(ui, prefs, Some(&state));
}

/// The Settings panel drawn with the viewer's text sizes (egui-prefs2 ships larger ones).
fn prefs_state(fonts: &crate::persist::Fonts) -> egui_prefs2::PrefsPanelState {
    egui_prefs2::PrefsPanelState { fonts: prefs_fonts(fonts), ..Default::default() }
}

/// The Settings panel's own sizes from `persist::Fonts`.
fn prefs_fonts(fonts: &crate::persist::Fonts) -> egui_prefs2::PrefsFonts {
    let f = fonts.clamped();
    egui_prefs2::PrefsFonts { sidebar: f.sidebar, heading: f.heading, body: f.body }
}

/// Settings > UI: opening files, then the text sizes per style, the dock tab height and
/// the Settings category list (vfx-view's "Fonts & UI").
fn ui_prefs(ui: &mut egui::Ui, fonts: &mut crate::persist::Fonts, open_seq: &mut bool) {
    use crate::persist::Fonts;
    egui_prefs2::section_header(ui, "Files");
    egui_prefs2::section_grid(ui, "files", |ui| {
        ui.label("Auto-detect sequences").on_hover_text(
            "File > Open and recent files open a frame of a name.####.exr set as the whole sequence              (a lone frame still opens as one image). The Open dialog has the same box.",
        );
        ui.checkbox(open_seq, "");
        ui.end_row();
    });
    egui_prefs2::section_header(ui, "Fonts & UI");
    egui_prefs2::section_grid(ui, "fonts", |ui| {
        const RESET: &str = "\nCtrl+click or right-click: reset.";
        let d = Fonts::default();
        for (label, tip, value, default) in [
            ("Heading", "Size of section titles.", &mut fonts.heading, d.heading),
            ("Body", "Size of ordinary interface text: the one to raise if the viewer is hard to read.", &mut fonts.body, d.body),
            ("Monospace", "Size of fixed-width text: pixel values, paths, attribute values.", &mut fonts.monospace, d.monospace),
            ("Button", "Size of text on buttons and dock tabs.", &mut fonts.button, d.button),
            ("Small", "Size of secondary text: hints, badges, ruler labels.", &mut fonts.small, d.small),
            ("Settings tabs", "Size of the category list down the left of this panel, nothing else.", &mut fonts.sidebar, d.sidebar),
        ] {
            ui.label(label).on_hover_text(format!("{tip}{RESET}"));
            ui.add(egui_slider::SlimSlider::new(&mut *value, Fonts::SIZE).default_value(default).step_by(0.5).max_decimals(1).suffix(" pt"));
            ui.end_row();
        }
        ui.label("Dock tab height").on_hover_text(format!("Height of the tab strip of each dock panel.{RESET}"));
        ui.add(egui_slider::SlimSlider::new(&mut fonts.dock_tab, Fonts::TAB).default_value(d.dock_tab).step_by(1.0).max_decimals(0).suffix(" px"));
        ui.end_row();
    });
}

/// How often the shown file's modification time is checked.
const WATCH_PERIOD: Duration = Duration::from_secs(1);

/// The shown file and the modification time it was loaded at.
struct Watch {
    path: PathBuf,
    stamp: Option<std::time::SystemTime>,
    /// The next check.
    next: Instant,
}

impl Watch {
    fn new(path: PathBuf) -> Self {
        let stamp = modified(&path);
        Self { path, stamp, next: Instant::now() + WATCH_PERIOD }
    }

    /// Whether the file changed since it was loaded (checked when `now` reached the
    /// next check; the stamp moves on, so one change reloads once).
    fn changed(&mut self, now: Instant) -> bool {
        if now < self.next {
            return false;
        }
        self.next = now + WATCH_PERIOD;
        let stamp = modified(&self.path);
        // A file that vanished (being replaced) is not a change yet: its new
        // version is.
        if stamp.is_none() || stamp == self.stamp {
            return false;
        }
        self.stamp = stamp;
        true
    }
}

/// Every present frame's modification time, by frame number.
type Stamps = std::collections::HashMap<i32, std::time::SystemTime>;

/// The sequence rescan's state.
struct SeqWatch {
    /// The frames and their modification times as last seen.
    stamps: Stamps,
    /// The next rescan.
    next: Instant,
    /// A rescan in flight: the new sequence and stamps (`None`: nothing detected).
    rx: Option<std::sync::mpsc::Receiver<Option<(Seq, Stamps)>>>,
}

impl SeqWatch {
    fn new(stamps: Stamps) -> Self {
        Self { stamps, next: Instant::now() + WATCH_PERIOD, rx: None }
    }
}

/// The modification time of every present frame of `seq`.
fn frame_stamps(seq: &Seq) -> Stamps {
    seq.indices
        .iter()
        .filter_map(|&f| {
            let path = seq.get_file(f)?;
            Some((i32::try_from(f).ok()?, modified(Path::new(&path))?))
        })
        .collect()
}

/// Between two scans: whether the frame set differs, and the frames present in both
/// whose file was rewritten (ascending).
fn seq_changes(old: &Stamps, new: &Stamps) -> (bool, Vec<i32>) {
    let set_changed = old.len() != new.len() || old.keys().any(|f| !new.contains_key(f));
    let mut rewritten: Vec<i32> =
        new.iter().filter(|(f, t)| old.get(f).is_some_and(|o| o != *t)).map(|(f, _)| *f).collect();
    rewritten.sort_unstable();
    (set_changed, rewritten)
}

/// The modification time of `path`, `None` when it cannot be read.
fn modified(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Find a representative sequence frame inside `dir`: the first image whose stem
/// contains a digit run. Used when `view <dir>` is given a directory rather than
/// a specific frame — we seed `scanseq::Scanner::from_file` from this, which then
/// does the real grouping/detection.
fn first_frame_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_seed_candidate(p))
        .collect();
    // Sort for determinism (lowest filename first).
    frames.sort();
    frames.into_iter().next()
}

/// A plausible sequence-frame seed: an image whose stem carries a digit run (a
/// `notes.001.txt` was never openable; the formats are `exr_import::families`).
fn is_seed_candidate(path: &std::path::Path) -> bool {
    exr_import::is_image(path) && has_frame_digits(path)
}

/// True when the file's stem contains an ASCII-digit run — i.e. it is a
/// plausible sequence frame for `scanseq` to group. (scanseq itself decides
/// what actually forms a sequence; this is only a cheap seed filter.)
fn has_frame_digits(path: &std::path::Path) -> bool {
    path.file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|stem| stem.bytes().any(|b| b.is_ascii_digit()))
}

/// winit application handler: owns the deferred `Viewer` and pumps the loop.
#[derive(Default)]
pub struct App {
    viewer: Option<Viewer>,
    /// Path to open once the window/GPU come up (file / dir / frame pattern).
    initial: Option<PathBuf>,
    /// Persisted config loaded at startup; moved into the `Viewer`'s `AppState`
    /// in `resumed` (and consulted there for the restored window geometry).
    config: Option<Config>,
    /// Sequence auto-detection toggle (CLI `--auto`); default off via `Default`.
    auto_detect: bool,
    /// How a finished decode asks for a repaint. Built from the event loop's
    /// proxy in [`crate::run`] (a proxy can only be made from the `EventLoop`
    /// itself, not from the `ActiveEventLoop` handed to `resumed`).
    wake: Option<scancache::Wake>,
}

impl App {
    /// Create the handler with an optional path to open on launch, the persisted
    /// [`Config`] (window geom + controls + recents), and the cache wake channel.
    pub fn new(
        initial: Option<PathBuf>,
        config: Config,
        auto_detect: bool,
        wake: scancache::Wake,
    ) -> Self {
        Self {
            viewer: None,
            initial,
            config: Some(config),
            auto_detect,
            wake: Some(wake),
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            return;
        }
        // Restore the saved window geometry (physical px) when present, else
        // fall back to the default 1280×800 logical size.
        let mut attrs = Window::default_attributes().with_title("exr-view");
        attrs = match self.config.as_ref().and_then(|c| c.window) {
            Some(g) => attrs
                .with_position(PhysicalPosition::new(g.x, g.y))
                .with_inner_size(PhysicalSize::new(g.w, g.h)),
            None => attrs.with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0)),
        };
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        // Move the loaded config into the viewer's AppState (defaults if absent,
        // e.g. on a platform that re-`resumed`s after a suspend).
        let config = self.config.take().unwrap_or_default();
        // The cache's wake channel: a finished decode posts `CacheProgress` to
        // this loop, so the UI sleeps instead of polling `has_pending()`. A
        // second `resumed` (after a suspend) reuses it.
        let wake = self.wake.clone().unwrap_or_else(|| Arc::new(|| {}));
        match Viewer::new(window, self.initial.clone(), config, self.auto_detect, wake) {
            Ok(v) => {
                // Kick the first paint: under `ControlFlow::Wait` nothing else
                // would trigger the initial `RedrawRequested`.
                v.window.request_redraw();
                self.viewer = Some(v);
            }
            Err(e) => {
                log::error!("GPU/egui init failed: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(viewer) = self.viewer.as_mut() else {
            return;
        };

        // Feed egui; if it wants a repaint (hover, focus, text caret, drag, …),
        // wake the loop so input stays responsive while we otherwise sleep.
        let response = viewer.on_window_event(&event);
        if response.repaint {
            viewer.window.request_redraw();
        }

        match event {
            // Save state on close (window button / OS request) before exiting.
            WindowEvent::CloseRequested => {
                viewer.save_state();
                // Exit NOW: skip the decode-pool drain in `Pool::Drop` (it waits up
                // to ~1 s for in-flight frame decodes to finish, which is the Esc/close
                // lag). The process is ending, so the OS reclaims the decode threads
                // and GPU at once; config is already persisted by `save_state()`.
                std::process::exit(0);
            }
            // Esc to exit — but only when egui did NOT want the key. Escape is
            // also "close this menu" / "cancel this edit"; quitting the viewer
            // out from under an open menu or a focused text field is not what
            // the user asked for.
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                && key_event.state == ElementState::Pressed
                && !response.consumed
                && !viewer.egui_state.egui_ctx().egui_wants_keyboard_input()
                && !egui::Popup::is_any_open(viewer.egui_state.egui_ctx())
                && !viewer.app_state.file_pick.is_open() =>
            {
                viewer.save_state();
                // Exit NOW: skip the decode-pool drain in `Pool::Drop` (it waits up
                // to ~1 s for in-flight frame decodes to finish, which is the Esc/close
                // lag). The process is ending, so the OS reclaims the decode threads
                // and GPU at once; config is already persisted by `save_state()`.
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                // A 0-area resize is a minimise: stop drawing until it comes back
                // (the surface cannot be configured at 0x0 either).
                viewer.minimised = size.width == 0 || size.height == 0;
                viewer.gpu.resize(size.width, size.height);
                if !viewer.hidden() {
                    viewer.window.request_redraw();
                }
            }
            WindowEvent::Moved(_) => {
                // Another monitor has other HDR figures (and maybe other outputs).
                if viewer.window.current_monitor().and_then(|m| m.name()) != viewer.monitor {
                    viewer.recheck_display();
                    viewer.window.request_redraw();
                }
            }
            WindowEvent::Focused(true) => {
                // HDR may have been switched in the OS meanwhile.
                viewer.recheck_display();
                viewer.window.request_redraw();
            }
            WindowEvent::Occluded(occluded) => {
                viewer.occluded = occluded;
                if !viewer.hidden() {
                    viewer.window.request_redraw();
                }
            }
            // Drag-and-drop: a hover shows a hint; a drop ALWAYS opens as a
            // sequence when one exists — `open_dropped` forces scanseq detection
            // (independent of `--auto`), so dropping any one frame of a
            // `name.####.exr` set opens the whole sequence; a lone EXR opens single.
            WindowEvent::HoveredFile(path) => {
                viewer.app_state.status = format!("Drop to open: {}", path.display());
                viewer.window.request_redraw();
            }
            WindowEvent::HoveredFileCancelled => {
                viewer.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                viewer.app_state.open_dropped(path);
                viewer.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if viewer.hidden() {
                    return;
                }
                if let Err(e) = viewer.render() {
                    log::error!("render error: {e:#}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    /// Decide the next wake instead of spinning (V2).
    ///
    /// `request_redraw` queues a paint message IMMEDIATELY on Windows, which ends
    /// the `WaitUntil` wait at once — so asking for one on every pass made the
    /// timeout decorative and the viewer rendered at the monitor rate (and, with
    /// no vsync while minimised, spun the CPU). The redraw is now requested only
    /// when something is actually due:
    /// - **playing** → when the frame deadline has passed. The deadline advances
    ///   by whole frame periods, so a late frame is DROPPED, never stretched.
    /// - **decodes in flight** → the cache's `Wake` wakes us the moment a frame
    ///   lands (see `UserEvent::CacheProgress`); the slow `WaitUntil` below is
    ///   only a backstop for the progress readout.
    /// - **idle** → egui's own `repaint_delay` (caret / tooltip animations),
    ///   else `Wait` (sleep at ~0% CPU until an OS event).
    ///
    /// While minimised or occluded NOTHING is requested: the loop just waits.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(viewer) = self.viewer.as_mut() else {
            return;
        };
        if viewer.hidden() {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        let (playing, pending, fps, next_wake) = match viewer.app_state.seq.as_ref() {
            Some(s) => (
                s.playback.playing,
                s.cache.has_pending(),
                s.playback.fps,
                s.cache.next_wake(),
            ),
            None => (false, false, 24.0, None),
        };

        if playing {
            let dt = Duration::from_secs_f32(1.0 / fps.max(1.0));
            let now = Instant::now();
            if now >= viewer.next_frame {
                // Skip over every deadline we already missed so playback keeps
                // real time (drop frames) instead of running slow.
                let behind = now.saturating_duration_since(viewer.next_frame);
                let missed = (behind.as_secs_f64() / dt.as_secs_f64()).floor();
                let skip = (missed as u32).saturating_add(1);
                viewer.next_frame += dt * skip;
                viewer.window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(viewer.next_frame));
            return;
        }
        // Not playing: re-arm the clock so resuming starts a fresh period.
        viewer.next_frame = Instant::now();
        // A changed file starts its reload here (the worker wakes the loop when done),
        // so an idle viewer notices without rendering a frame per check.
        viewer.app_state.poll_watch();
        let watch = viewer.app_state.watch_deadline();

        match viewer.repaint_after {
            d if d == Duration::ZERO => {
                viewer.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Poll);
            }
            d if d == Duration::MAX => {
                // The cache wake (`UserEvent::CacheProgress`) drives the repaint
                // while frames decode; this WaitUntil is the backstop that also
                // services `Scancache::poll()` (via `tick_sequence` on the next
                // render): `next_wake()` gives the soonest deferred deadline
                // (scrub settle / retry backoff) so a scrub resumes read-ahead and
                // a failed frame retries even with no in-flight decode to wake us.
                // 250 ms remains the backstop for the cache-progress readout when
                // there is no deadline but decodes are still in flight.
                let backstop = Duration::from_millis(250);
                let wait = match (next_wake, pending) {
                    (Some(d), _) => Some(d.min(backstop)),
                    (None, true) => Some(backstop),
                    (None, false) => None,
                };
                let until = match (wait.map(|d| Instant::now() + d), watch) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                event_loop.set_control_flow(match until {
                    Some(t) => ControlFlow::WaitUntil(t),
                    None => ControlFlow::Wait,
                });
            }
            d => {
                let t = Instant::now() + d;
                event_loop.set_control_flow(ControlFlow::WaitUntil(watch.map_or(t, |w| w.min(t))));
            }
        }
    }

    /// A decode landed (or the cache made progress): repaint once, now. This is
    /// what replaces the old 60 Hz poll while frames are loading.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let Some(viewer) = self.viewer.as_ref() else {
            return;
        };
        match event {
            UserEvent::CacheProgress => {
                if !viewer.hidden() {
                    viewer.window.request_redraw();
                }
            }
        }
    }
}

#[cfg(test)]
mod neighbour_tests {
    use super::*;

    /// PgUp/PgDn step over the folder by name; the other frames of the open
    /// sequence are one item with the shown frame, and the ends are ends.
    #[test]
    fn neighbour_skips_the_own_sequence() {
        let files: Vec<PathBuf> = ["a.exr", "shot.0001.exr", "shot.0002.exr", "shot.0003.exr", "z.exr"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let own: HashSet<PathBuf> = files[1..4].iter().cloned().collect();
        let cur = Path::new("shot.0002.exr");
        assert_eq!(neighbour(&files, cur, 1, &own), Some(PathBuf::from("z.exr")));
        assert_eq!(neighbour(&files, cur, -1, &own), Some(PathBuf::from("a.exr")));
        let none = HashSet::new();
        assert_eq!(neighbour(&files, Path::new("a.exr"), 1, &none), Some(PathBuf::from("shot.0001.exr")));
        assert_eq!(neighbour(&files, Path::new("a.exr"), -1, &none), None);
        assert_eq!(neighbour(&files, Path::new("z.exr"), 1, &none), None);
        // A file not in the listing (deleted meanwhile) still has neighbours.
        assert_eq!(neighbour(&files, Path::new("m.exr"), 1, &none), Some(PathBuf::from("shot.0001.exr")));
    }
}

#[cfg(test)]
mod watch_tests {
    use super::*;

    /// A file rewritten on disk is a change once; an untouched or vanished file is
    /// not, and nothing is checked before the period ends.
    #[test]
    fn watch_sees_each_rewrite_once() {
        let dir = std::env::temp_dir().join(format!("exr-view-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shot.exr");
        std::fs::write(&path, b"v1").unwrap();
        let mut w = Watch::new(path.clone());
        let t0 = Instant::now();
        assert!(!w.changed(t0), "before the period");
        let later = |w: &Watch| w.next + Duration::from_millis(1);
        assert!(!w.changed(later(&w)), "untouched");
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() + Duration::from_secs(5)).unwrap();
        drop(file);
        assert!(w.changed(later(&w)), "rewritten");
        assert!(!w.changed(later(&w)), "the same change reloads once");
        std::fs::remove_file(&path).unwrap();
        assert!(!w.changed(later(&w)), "vanished is not a change");
    }
}

#[cfg(test)]
mod seq_watch_tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    /// A 2x2 EXR frame whose R holds `v`.
    fn frame(path: &Path, v: f32) {
        let win = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(1, 1));
        exr_core::api::Image::new(win)
            .with_channel("R", exr_core::api::ChannelData::Float(vec![v; 4]))
            .write(path, exr_core::Compression::NoCompression)
            .unwrap();
    }

    #[test]
    fn sequence_does_not_cancel_an_unpresented_playback_frame() {
        let dir = std::env::temp_dir().join(format!("exr-view-handoff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = |f| dir.join(format!("shot.{f:04}.exr"));
        for f in 1..=2 { frame(&path(f), f as f32); }
        let mut app = AppState::new(None, Config::default(), true, Arc::new(|| {}));
        app.apply_open(open_blocking(path(1), true, u64::MAX));
        let deadline = Instant::now() + Duration::from_secs(30);
        while !app.seq.as_ref().is_some_and(|s| s.cache.contains(1) && s.cache.contains(2)) {
            app.tick_sequence();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        let s = app.seq.as_mut().unwrap();
        s.ensure_display(Some(&app.display_gpu));
        // The production source setup used by exports/scopes also creates the
        // generation; it has not reached a viewport yet (no GPU needed here).
        s.display.as_mut().unwrap().export_level(&mut app.display_gpu, 16384, None).unwrap();
        let first_id = s.display.as_ref().unwrap().frame.id;
        s.playback.set_pos(1);
        s.playback.playing = true;
        s.ensure_display(Some(&app.display_gpu));
        assert_eq!(s.displayed_frame, Some(1), "clock frame 2 does not cancel frame 1's tiles");
        assert_eq!(s.display.as_ref().unwrap().frame.id, first_id);
        // OFX time is read from this accepted displayed_frame, not the clock.
        assert_eq!((s.displayed_frame, s.playback.current()), (Some(1), 2));
        s.scrubbing = true;
        s.ensure_display(Some(&app.display_gpu));
        assert_eq!(s.displayed_frame, Some(2), "manual scrub can supersede a pending frame");
        s.playback.set_pos(0);
        s.scrubbing = false;
        s.ensure_display(None);
        assert_eq!(s.displayed_frame, Some(1), "scope-only view needs no viewport presentation");
    }

    /// A render in progress: a frame appears and a frame is rewritten. The rescan
    /// grows the clock and the cache to the new frame and drops only the rewritten
    /// frame from the cache; the decode loader resolves the new frame.
    #[test]
    fn rescan_follows_a_sequence_changing_on_disk() {
        let dir = std::env::temp_dir().join(format!("exr-view-seqwatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = |f: i32| dir.join(format!("shot.{f:04}.exr"));
        for f in 1..=3 {
            frame(&name(f), f as f32);
        }
        let wake: scancache::Wake = Arc::new(|| {});
        let mut app = AppState::new(None, Config::default(), true, Arc::clone(&wake));
        app.apply_open(open_blocking(name(1), true, u64::MAX));
        assert_eq!(app.seq.as_ref().expect("opened as a sequence").playback.indices(), [1, 2, 3]);
        // The viewer's per-frame path (clock, playhead with the current range, cache
        // poll) until `ok` holds.
        let run = |app: &mut AppState, what: &str, ok: &dyn Fn(&SequenceState) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(30);
            while !app.seq.as_ref().is_some_and(ok) {
                assert!(Instant::now() < deadline, "{what}");
                app.tick_sequence();
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        // Let every frame decode, so a rewrite has something to drop.
        run(&mut app, "frames decode", &|s| (1..=3).all(|f| s.cache.contains(f)));

        frame(&name(4), 4.0);
        frame(&name(2), 20.0);
        let later = std::time::SystemTime::now() + Duration::from_secs(5);
        std::fs::File::options().write(true).open(name(2)).unwrap().set_modified(later).unwrap();
        let s = app.seq.as_mut().unwrap();
        s.watch.next = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(30);
        while !s.poll_rescan(Instant::now(), &wake) {
            assert!(Instant::now() < deadline, "the rescan lands");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(s.playback.indices(), [1, 2, 3, 4]);
        assert_eq!(s.cache.total(), 4);
        let red = |s: &SequenceState, f: i32| s.cache.get(f).and_then(|d| d.channels[0].1.sample_f32(0));
        assert!(red(s, 2).is_none_or(|v| v == 20.0), "the stale decode of the rewritten frame is gone");
        assert_eq!((red(s, 1), red(s, 3)), (Some(1.0), Some(3.0)), "untouched frames stay decoded");
        run(&mut app, "the new frame decodes", &|s| s.cache.contains(4));
        run(&mut app, "the rewritten frame decodes again", &|s| red(s, 2) == Some(20.0));
    }

    /// "Auto-detect sequences" (`Config::open_seq`) decides how a chosen frame opens,
    /// whatever `--auto` says: on = the whole sequence, off = the one frame.
    #[test]
    fn chosen_paths_follow_the_auto_sequence_option() {
        let dir = std::env::temp_dir().join(format!("exr-view-openseq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = |f: i32| dir.join(format!("shot.{f:04}.exr"));
        for f in 1..=3 {
            frame(&name(f), f as f32);
        }
        let wake: scancache::Wake = Arc::new(|| {});
        for open_seq in [true, false] {
            let config = Config { open_seq, ..Config::default() };
            let mut app = AppState::new(None, config, false, Arc::clone(&wake));
            app.open_chosen(name(2));
            let deadline = Instant::now() + Duration::from_secs(30);
            while app.seq.is_none() && app.image.is_none() {
                assert!(Instant::now() < deadline, "the open lands");
                app.tick_open();
                std::thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(app.seq.is_some(), open_seq, "open_seq = {open_seq}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A `name.####.png` set is detected and opens as a sequence like an EXR one, and its
    /// frames decode through `exr-import`.
    #[test]
    fn a_png_sequence_opens_as_a_sequence() {
        let dir = std::env::temp_dir().join(format!("exr-view-pngseq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = |f: i32| dir.join(format!("shot.{f:04}.png"));
        for f in 1..=3 {
            crate::image::tests::write_png(&name(f), [f as u8 * 40, 0, 0, 255]);
        }
        let wake: scancache::Wake = Arc::new(|| {});
        let mut app = AppState::new(None, Config { open_seq: true, ..Config::default() }, false, wake);
        app.open_chosen(name(2));
        let deadline = Instant::now() + Duration::from_secs(30);
        while app.seq.is_none() && app.image.is_none() {
            assert!(Instant::now() < deadline, "the open lands");
            app.tick_open();
            std::thread::sleep(Duration::from_millis(10));
        }
        let seq = app.seq.as_ref().expect("a .png set opens as a sequence");
        assert_eq!(seq.playback.indices(), &[1, 2, 3]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// New frames change the set; a different time on a surviving frame is a rewrite;
    /// a vanished frame changes the set and is no rewrite; equal scans change nothing.
    #[test]
    fn seq_changes_tells_set_changes_from_rewrites() {
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let t1 = t0 + Duration::from_secs(1);
        let scan = |v: &[(i32, SystemTime)]| v.iter().copied().collect::<Stamps>();
        let old = scan(&[(1, t0), (2, t0), (3, t0)]);
        assert_eq!(seq_changes(&old, &old), (false, vec![]));
        assert_eq!(seq_changes(&old, &scan(&[(1, t0), (2, t0), (3, t0), (4, t0)])), (true, vec![]));
        assert_eq!(seq_changes(&old, &scan(&[(1, t0), (2, t1), (3, t1)])), (false, vec![2, 3]));
        assert_eq!(seq_changes(&old, &scan(&[(1, t0), (3, t1)])), (true, vec![3]));
        assert_eq!(seq_changes(&old, &scan(&[(1, t0), (2, t0), (5, t0)])), (true, vec![]), "same size, other frame");
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;
    use crate::transfer::tonemap;

    /// T walks None, Reinhard, AgX, OCIO and back; picking a tone operator turns
    /// OCIO off, and OCIO keeps the tone operator for when it is turned off.
    #[test]
    fn display_cycle_covers_every_transform() {
        let mut c = ViewControls::default();
        let mut seen = vec![c.display()];
        for _ in 0..4 {
            c.set_display(c.next_display());
            seen.push(c.display());
        }
        assert_eq!(seen, [Some(tonemap::NONE), Some(tonemap::REINHARD), Some(tonemap::AGX), None, Some(tonemap::NONE)]);
        c.set_display(Some(tonemap::AGX));
        c.set_display(None);
        assert!(c.colour.sel.on && c.tonemap == tonemap::AGX);
        c.set_display(Some(tonemap::REINHARD));
        assert!(!c.colour.sel.on && c.colour.active().is_none());
    }
}

#[cfg(test)]
mod ram_partition_tests {
    use super::*;
    #[test]
    fn configured_total_is_partitioned_exactly_without_overflow() {
        for total in [0, 1, 2, 3, 4, 5, 7, 13, usize::MAX - 1, usize::MAX] {
            let p = RamPartitions::new(total);
            assert_eq!(p.decoded, total / 4);
            assert_eq!(p.holding, total / 4);
            assert_eq!(p.decoded.checked_add(p.holding).unwrap().checked_add(p.tiles), Some(total));
            assert_eq!(p.validate().is_ok(), total >= 4);
        }
    }

    /// The frame cache gets the whole budget while the other caches are empty, loses
    /// exactly what they hold, and never drops below its floor (their usage is
    /// clamped to their ceilings, so an over-full cache cannot underflow it).
    #[test]
    fn frame_cap_is_the_budget_minus_what_the_others_hold() {
        for total in [4, 13, 64 << 30, usize::MAX] {
            let p = RamPartitions::new(total);
            assert_eq!(p.frames(0, 0), total, "idle tiles and inputs leave it all to frames");
            assert_eq!(p.frames(1, 0), total - 1);
            assert_eq!(p.frames(0, 1), total - 1);
            assert_eq!(p.frames(p.tiles, p.holding), p.decoded, "both full: the floor");
            assert_eq!(p.frames(usize::MAX, usize::MAX), p.decoded, "over-full: still the floor");
        }
    }
}

#[cfg(test)]
mod dock_action_tests {
    use super::*;

    /// Window menu / tab context menu actions open a tab or bring an open one forward; they
    /// never close it (that is the tab's X), so no entry can remove the main view.
    #[test]
    fn tab_action_opens_and_focuses_but_never_closes() {
        let mut app = AppState::new(None, Config::default(), true, Arc::new(|| {}));
        let mut dock = default_dock_state();
        let ctx = egui::Context::default();
        // Open: a closed tab comes back.
        let path = dock.find_tab(&DockTab::Waveform).expect("default layout has a waveform");
        dock.remove_tab(path);
        assert!(dock.find_tab(&DockTab::Waveform).is_none());
        run(Action::Tab(DockTab::Waveform), &mut app, &mut dock, &ctx);
        assert!(dock.find_tab(&DockTab::Waveform).is_some());
        // Focus: the open Histogram becomes the active tab of its leaf, and stays open.
        run(Action::Tab(DockTab::Histogram), &mut app, &mut dock, &ctx);
        let path = dock.find_tab(&DockTab::Histogram).expect("still open");
        let leaf = dock.leaf(path.node_path()).expect("a leaf");
        assert_eq!(leaf.active, path.tab, "brought forward");
        // Viewport: the same action on the open main view keeps it.
        run(Action::Tab(DockTab::Viewport), &mut app, &mut dock, &ctx);
        assert!(dock.find_tab(&DockTab::Viewport).is_some());
    }

    /// Every `.rs` under `dir`, recursively.
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("source dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// Files that may read Ctrl / Cmd themselves: the keyboard-shortcut registry (a chord is not
    /// a click). Everywhere else a Ctrl read is the reset gesture being spelled locally.
    const CTRL_ALLOWED: &[&str] = &["commands.rs"];

    /// Whether `needle` occurs in `t` as a whole token: the char after it is not part of an
    /// identifier (a Ctrl field read is a hit, a longer name that merely starts with it is not).
    fn has_token(t: &str, needle: &str) -> bool {
        t.match_indices(needle).any(|(at, _)| {
            !t[at + needle.len()..].chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
    }

    /// Where `text` (the source file `name`) spells the reset gesture itself, whole file, with
    /// ALL whitespace removed first so a spelling split over lines or padded still matches. The
    /// needles are joined from pieces, so this guard never matches its own source.
    fn reset_spellings(name: &str, text: &str) -> Vec<String> {
        let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let mut found = Vec::new();
        let secondary = ["secondary", "_clicked"].concat();
        if t.contains(&secondary) {
            found.push(format!("{name}: a bare right click"));
        }
        // `<x>_by(..)` / `button_<x>(..)` taking the secondary button: look inside the call.
        for call in [["clicked", "_by("].concat(), ["button_", "clicked("].concat()] {
            for (at, _) in t.match_indices(&call) {
                let args = t[at + call.len()..].split(')').next().unwrap_or("");
                if args.contains("Secondary") {
                    found.push(format!("{name}: a secondary {call}..)"));
                }
            }
        }
        if !CTRL_ALLOWED.contains(&name) {
            for needle in [
                [".comm", "and"].concat(),
                [".ct", "rl"].concat(),
                ["mac_", "cmd"].concat(),
                ["Modifiers::", "COMMAND"].concat(),
                ["Modifiers::", "CTRL"].concat(),
                ["matches_", "exact("].concat(),
                ["matches_", "logically("].concat(),
            ] {
                if has_token(&t, &needle) {
                    found.push(format!("{name}: a Ctrl / Cmd read ({needle})"));
                }
            }
        }
        found
    }

    /// The reset gesture is spelled once, in `egui-widgets-config::reset`: no local copy of it
    /// may come back anywhere in the viewer's sources.
    #[test]
    fn the_reset_gesture_is_not_spelled_locally() {
        let mut files = Vec::new();
        sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
        assert!(files.len() > 10, "the source scan found {} files", files.len());
        for file in files {
            let name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            let text = std::fs::read_to_string(&file).expect("readable source");
            let found = reset_spellings(name, &text);
            assert!(found.is_empty(), "{}: {found:?}", file.display());
        }
    }

    /// The scanner itself: each spelling it has to catch (built from pieces, so these cases are
    /// not in this file's text), and the legitimate uses it must leave alone.
    #[test]
    fn the_reset_guard_catches_every_spelling() {
        let by = ["clicked", "_by"].concat();
        let button = ["button_", "clicked"].concat();
        let bare = ["secondary", "_clicked"].concat();
        let cmd = ["modifiers.", "command"].concat();
        let ctrl = ["modifiers.", "ctrl"].concat();
        let exact = ["matches_", "exact"].concat();
        let bad = [
            format!("if r.{bare}() {{}}"),
            format!("r.{by}(egui::PointerButton::Secondary)"),
            format!("r.{by}(\n    egui::PointerButton::Secondary,\n)"),
            format!("r . {by} (\n  Secondary )"),
            format!("r.{button}(egui::PointerButton::Secondary)"),
            format!("if ui.input(|i| i.{cmd}) && r.clicked() {{}}"),
            format!("i.{ctrl}"),
            format!("i.modifiers.{exact}(egui::Modifiers::COMM{}) ", "AND"),
            format!("let m = egui::Modifiers::CT{};", "RL"),
        ];
        for text in &bad {
            assert!(!reset_spellings("app.rs", text).is_empty(), "missed: {text}");
        }
        // Not gestures: a primary `clicked_by`, a plain click, Ctrl in the shortcut registry.
        let fine = [
            format!("r.{by}(egui::PointerButton::Primary)"),
            "if r.clicked() {}".to_owned(),
            format!("egui::Modifiers::COMM{}", "AND"),
            // Longer identifiers that merely start with a Ctrl needle are not reads.
            format!("ui.{}_palette(); i.modifiers.{}_x", ["comm", "and"].concat(), ["ct", "rl"].concat()),
        ];
        for text in [&fine[0], &fine[1], &fine[3]] {
            assert!(reset_spellings("app.rs", text).is_empty(), "false alarm: {text}");
        }
        assert!(reset_spellings("commands.rs", &fine[2]).is_empty(), "the shortcut registry may read Ctrl");
        assert!(!reset_spellings("app.rs", &fine[2]).is_empty());
    }
}

/// The keyboard through the PRODUCTION functions `build_ui` is made of (`read_keys`,
/// `transport_bar`, `toolbar_strip`): nothing here registers a panel or reads a key by hand.
#[cfg(test)]
mod key_path_tests {
    use super::*;
    use crate::commands::Scope;
    use crate::viewport::NoGpu;
    use egui::{Event, Key, Modifiers, PointerButton, Pos2, RawInput, pos2, vec2};

    pub(super) fn input(events: Vec<Event>) -> RawInput {
        RawInput { screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))), events, ..Default::default() }
    }

    fn press(key: Key, modifiers: Modifiers) -> Event {
        Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }
    }

    fn click(pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }
    }

    pub(super) fn app() -> AppState {
        AppState::new(None, Config::default(), true, Arc::new(|| {}))
    }

    /// One frame; `ui` draws the panels. No renderer here, so the font upload is dropped.
    pub(super) fn frame(ctx: &egui::Context, events: Vec<Event>, ui: impl FnMut(&mut egui::Ui)) {
        let mut ui = ui;
        let mut out = ctx.run_ui(input(events), |root| ui(root));
        out.textures_delta.clear();
    }

    /// `read_keys` is what runs the keys: R isolates the red channel, and a modal (here the Open
    /// dialog) that has the keyboard leaves bare keys alone while Ctrl chords still fire.
    #[test]
    fn read_keys_runs_the_pressed_keys_and_respects_typing() {
        let (mut app, mut dock, ctx) = (app(), default_dock_state(), egui::Context::default());
        frame(&ctx, vec![press(Key::R, Modifiers::NONE)], |ui| read_keys(ui.ctx(), &mut app, &mut dock));
        assert_eq!(app.controls.channel, ChannelDisplay::R);
        frame(&ctx, vec![Event::Key { key: Key::R, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE }], |_| {});
        app.file_pick.open(None);
        assert!(app.file_pick.is_open());
        frame(&ctx, vec![press(Key::G, Modifiers::NONE)], |ui| read_keys(ui.ctx(), &mut app, &mut dock));
        assert_eq!(app.controls.channel, ChannelDisplay::R, "bare keys are the dialog's while it is open");
        frame(&ctx, vec![press(Key::Z, Modifiers::NONE)], |ui| read_keys(ui.ctx(), &mut app, &mut dock));
        assert!(!app.fullscreen);
    }

    /// The transport bar is the Timeline panel: the pointer over it makes the frame keys its own.
    #[test]
    fn the_transport_bar_is_the_timeline_panel() {
        let (mut app, ctx) = (app(), egui::Context::default());
        let at = pos2(400.0, 599.0);
        for _ in 0..2 {
            frame(&ctx, vec![Event::PointerMoved(at)], |ui| transport_bar(ui, &mut app));
        }
        assert_eq!(commands::active(&ctx), Some(Scope::Timeline));
        // Off the bar, with nothing registered there, the keyboard is the fallback's.
        frame(&ctx, vec![Event::PointerMoved(pos2(400.0, 100.0))], |ui| transport_bar(ui, &mut app));
        assert_eq!(commands::active(&ctx), Some(Scope::Viewport));
    }

    /// The toolbar strip floats on its own layer: over it the keys are the viewport's, and a click
    /// on it makes the viewport the retained owner.
    #[test]
    fn the_toolbar_strip_belongs_to_the_viewport_for_the_keys() {
        let (mut app, ctx) = (app(), egui::Context::default());
        let strip = std::cell::Cell::new(egui::Rect::NOTHING);
        let mut got = Vec::new();
        let mut run = |app: &mut AppState, events: Vec<Event>| {
            let avail = app.avail();
            frame(&ctx, events, |root| {
                got = commands::pressed(root.ctx(), false, avail);
                egui::Panel::left("meta").show(root, |ui| commands::panel(ui, Scope::Metadata, |ui| ui.set_min_size(vec2(150.0, 500.0))));
                egui::CentralPanel::default().show(root, |ui| {
                    let rect = ui.max_rect();
                    strip.set(app.toolbar_strip(ui, rect).0.rect);
                });
            });
            std::mem::take(&mut got)
        };
        let (meta, away) = (pos2(50.0, 300.0), pos2(700.0, 500.0));
        run(&mut app, vec![Event::PointerMoved(meta)]);
        run(&mut app, vec![Event::PointerMoved(meta)]);
        run(&mut app, vec![click(meta, true)]);
        run(&mut app, vec![click(meta, false), Event::PointerMoved(away)]);
        // Metadata owns the keyboard; H is the viewport's key.
        let release = || Event::Key { key: Key::H, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE };
        assert_eq!(run(&mut app, vec![press(Key::H, Modifiers::NONE)]), []);
        run(&mut app, vec![release()]);
        assert!(strip.get().width() > 0.0, "the strip is drawn");
        run(&mut app, vec![Event::PointerMoved(strip.get().center())]);
        run(&mut app, vec![]);
        assert_eq!(run(&mut app, vec![press(Key::H, Modifiers::NONE)]), [Action::Nav(NavCmd::Home)], "over the strip");
        // A click on the strip: the viewport keeps the keys after the pointer leaves it.
        run(&mut app, vec![release()]);
        run(&mut app, vec![click(strip.get().center(), true)]);
        run(&mut app, vec![click(strip.get().center(), false), Event::PointerMoved(meta)]);
        run(&mut app, vec![Event::PointerMoved(away)]);
        assert_eq!(run(&mut app, vec![press(Key::H, Modifiers::NONE)]), [Action::Nav(NavCmd::Home)], "owner after the click");
    }

    /// Every panel of the real `build_ui` registers itself: each dock tab (`DockTabViewer::ui`),
    /// the viewport drawn alone with the panels hidden, and the transport bar. A registered panel
    /// is told apart from the fallback by an overlay on top of it (the overlay makes its pointer
    /// own nothing), and the tabs by the scope they own under the pointer. Runs without an
    /// adapter: the dock viewer holds a `NoGpu`, so the pixel tabs only say so, which is all the
    /// registrations need.
    #[test]
    fn build_ui_registers_every_panel() {
        let mut gpu = NoGpu;
        let status = GpuStatus { name: String::new(), backend: String::new(), vram_budget: None };
        let dir = std::env::temp_dir().join(format!("exr-view-keypath-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = |f: i32| dir.join(format!("shot.{f:04}.exr"));
        let win = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(1, 1));
        for f in 1..=2 {
            exr_core::api::Image::new(win)
                .with_channel("R", exr_core::api::ChannelData::Float(vec![0.5; 4]))
                .write(name(f), exr_core::Compression::NoCompression)
                .unwrap();
        }
        let mut app = app();
        app.apply_open(open_blocking(name(1), true, u64::MAX));
        assert!(app.seq.is_some());
        let mut dock = default_dock_state();
        let ctx = egui::Context::default();
        // The scopes owned under the pointer over a grid of the window, with or without an overlay
        // on top of everything (`None`: the pointer is over a registered panel and an overlay).
        let mut probe = |app: &mut AppState, dock: &mut DockState<DockTab>, overlay: bool| {
            let mut seen = Vec::new();
            for y in (5..600).step_by(30) {
                for x in (5..800).step_by(40) {
                    for _ in 0..2 {
                        frame(&ctx, vec![Event::PointerMoved(pos2(x as f32, y as f32))], |ui| {
                            build_ui(ui, dock, app, &mut gpu, &status);
                            if overlay {
                                egui::Area::new(egui::Id::new("keypath-overlay"))
                                    .order(egui::Order::Foreground)
                                    .fixed_pos(Pos2::ZERO)
                                    .show(ui.ctx(), |ui| {
                                        ui.allocate_exact_size(vec2(800.0, 600.0), egui::Sense::hover());
                                    });
                            }
                        });
                    }
                    seen.push(commands::active(&ctx));
                }
            }
            seen
        };
        // Every tab, brought forward in turn, owns its scope somewhere; the transport bar too.
        for tab in DockTab::ALL {
            if dock.find_tab(&tab).is_none() {
                dock.push_to_focused_leaf(tab.clone());
            }
            run(Action::Tab(tab.clone()), &mut app, &mut dock, &ctx);
            assert!(probe(&mut app, &mut dock, false).contains(&Some(Scope::of(&tab))), "{tab:?} registers its scope");
        }
        assert!(probe(&mut app, &mut dock, false).contains(&Some(Scope::Timeline)), "the transport bar registers");
        // A registered panel hides what is under an overlay: the viewport tab and the viewport
        // alone (panels hidden) leave the pointer unowned there, the fallback does not.
        assert!(probe(&mut app, &mut dock, true).contains(&None), "the viewport tab registers");
        app.panels_hidden = true;
        assert!(probe(&mut app, &mut dock, true).iter().all(Option::is_none), "the lone viewport covers the window");
        app.panels_hidden = false;
        // The keys of the frame go through `read_keys`: R isolates red.
        frame(&ctx, vec![press(Key::R, Modifiers::NONE)], |ui| build_ui(ui, &mut dock, &mut app, &mut gpu, &status));
        assert_eq!(app.controls.channel, ChannelDisplay::R);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// The viewer backdrop (BUG8): what the viewport shows under the image's alpha reaches the
/// present pass through the shared `egui-display` path, never a painter-drawn checker.
#[cfg(test)]
mod backdrop_tests {
    use super::key_path_tests::{app, frame, input};
    use super::*;
    use crate::viewport::NoGpu;
    use egui_display::{Backdrop, BackdropKind, Checker};

    /// A 2x2 single image in a fresh `AppState`.
    fn with_image() -> AppState {
        let dir = std::env::temp_dir().join(format!("exr-view-backdrop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.exr");
        let win = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(1, 1));
        exr_core::api::Image::new(win)
            .with_channel("R", exr_core::api::ChannelData::Float(vec![0.5; 4]))
            .write(&path, exr_core::Compression::NoCompression)
            .unwrap();
        let mut app = app();
        app.apply_open(open_blocking(path, false, u64::MAX));
        assert!(app.image.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
        app
    }

    /// One frame of the viewport tab content (no GPU: the tiles are not drawn, the backdrop is).
    fn show(app: &mut AppState, ctx: &egui::Context) {
        frame(ctx, vec![], |root| {
            egui::CentralPanel::default().show(root, |ui| app.display_viewport(ui, &mut NoGpu, viewport::Strip::NONE));
        });
    }

    /// The viewport shows the persisted backdrop over the image's rect, the host hands it to the
    /// present pass, and a frame that shows none presents none.
    #[test]
    fn the_viewport_shows_its_backdrop_and_the_host_hands_it_over() {
        let (mut app, ctx) = (with_image(), egui::Context::default());
        let mut slot = None;
        show(&mut app, &ctx);
        hand_backdrop(&ctx, &mut slot);
        let checker = slot.expect("the viewport shows the default checker");
        assert_eq!((checker.a, checker.cell_px), (Checker::default().a, 16.0), "the checker, in screen pixels");
        app.controls.backdrop = Backdrop { kind: BackdropKind::Black, ..Backdrop::default() };
        show(&mut app, &ctx);
        hand_backdrop(&ctx, &mut slot);
        let black = slot.expect("black is a backdrop too");
        assert_eq!((black.a, black.cell_px), ([0.0, 0.0, 0.0, 1.0], 0.0), "a solid, not the checker");
        assert!(black.clip_px[2] > black.clip_px[0] && black.clip_px[3] > black.clip_px[1], "over the image's rect");
        // Nothing shown this frame: the pass is told so, not left with the last one.
        hand_backdrop(&ctx, &mut slot);
        assert!(slot.is_none());
    }

    /// No image, no backdrop: it is only ever under the image.
    #[test]
    fn an_empty_viewport_shows_no_backdrop() {
        let (mut app, ctx) = (app(), egui::Context::default());
        let mut slot = None;
        show(&mut app, &ctx);
        hand_backdrop(&ctx, &mut slot);
        assert!(slot.is_none());
    }

    /// The strip carries the backdrop button (the shared picker labels it "Backdrop: <kind>").
    #[test]
    fn the_toolbar_has_the_backdrop_picker() {
        let (mut app, ctx) = (app(), egui::Context::default());
        ctx.enable_accesskit();
        let mut labels = Vec::new();
        for _ in 0..2 {
            let mut out = ctx.run_ui(input(vec![]), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    let rect = ui.max_rect();
                    app.toolbar_strip(ui, rect);
                });
            });
            // No renderer here: drop the font upload, as `frame` does (debug epaint asserts on it).
            out.textures_delta.clear();
            labels.extend(
                out.platform_output
                    .accesskit_update
                    .iter()
                    .flat_map(|u| u.nodes.iter())
                    .filter_map(|(_, n)| n.label().map(str::to_owned)),
            );
        }
        assert!(labels.iter().any(|l| l == "Backdrop: Checker"), "{labels:?}");
    }
}
