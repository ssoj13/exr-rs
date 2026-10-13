//! `exr-view` — a GPU-first OpenEXR viewer with sequence playback and scopes.
//!
//! winit + wgpu 30 + egui/egui_dock drive the window. EXR files are loaded through
//! the workspace `exr-core` reader. Eligible images use resident GPU display;
//! other images use a bounded tile pyramid and atlas, including file-backed input.
//! A shared display transform feeds the float canvas and SDR, PQ, HLG or scRGB output.
//!
//! GPU presentation is mandatory. Decoding and the current OpenFX tile source run
//! on the CPU; the latter retains charged original-precision work in TileState.
//! Processed consumers freeze the accepted presentation's owners, lineage and
//! coverage rather than borrowing a pending effect generation.
//!
//! # Entry point
//! [`run`] creates the winit event loop and drives the viewer to completion.
//! Its optional initial path selects a file, directory or frame pattern.
//! [`screenshot`] exports the neutral view without creating a window.

mod app;
mod channels;
#[cfg(feature = "ofx")]
mod consumers;
mod commands;
mod dock;
mod export;
mod file_pick;
mod frame_loader;
mod gpu_ctx;
mod image;
#[cfg(feature = "ofx")]
mod ofx;
mod ocio;
mod persist;
mod playback;
mod present;
mod scopes;
mod theme;
mod tiles;
mod timing;
mod transfer;
mod viewport;

pub use app::{App, AppState, UserEvent, ViewControls, proxy_wake};
pub use dock::{DockTab, default_dock_state};
pub use export::HdrPng;
pub use gpu_ctx::GpuContext;
pub use image::{ChannelDisplay, DecodedFrame, ImageBackend, ImageStage, LoadError};
pub use persist::{Config, WindowGeom};
pub use playback::Playback;
pub use scopes::{GpuHistogramPanel, GpuVectorscopePanel, GpuWaveformPanel};
// Sequence detection is now provided by the vendored `scanseq` crate
// (`scanseq::{Scanner, Seq}`) — the hand-rolled `sequence.rs` was removed.
pub use scanseq::{Scanner as SeqScanner, Seq};
pub use theme::{ColorTokens, Theme};
pub use transfer::{code8, display_pixel, oetf};

use std::path::PathBuf;

use anyhow::{Context, Result};
use winit::event_loop::{ControlFlow, EventLoop};

/// Launch the viewer: build the winit event loop and run the app to exit.
///
/// `initial` is an optional path to open on launch (a single EXR file, a
/// directory, or a `name.####.exr` frame pattern). The status bar shows it;
/// the load seam reads it on the first frame. Blocks until the window
/// is closed; propagates init/loop errors so the caller (CLI / bin) can report
/// them (it prints + exits non-zero on failure — the lib never panics on setup).
/// `auto` (CLI `--auto`) toggles `name.####.exr` sequence auto-detection. When
/// false (default), the given path opens as a single image with no directory scan.
/// Save `input` as the viewer shows it with the neutral look (0 EV, gamma 1, RGBA)
/// without a window (`exr-rs view <file> --screenshot <out>`): an SDR 8-bit PNG, or with
/// `hdr` HDR: the display light as `.exr`, or the `hdr` signal (HDR10 PQ / HLG) as `.png`
/// (by `output`'s extension). The colour is the saved OCIO choice's (Settings > Colour) through
/// the frame's own colour space, the processor the viewer runs; budgets are the saved Settings
/// ones. Returns the saved size.
pub fn screenshot(input: &std::path::Path, output: &std::path::Path, hdr: Option<export::HdrPng>) -> Result<(u32, u32)> {
    let cfg = Config::load();
    let budgets = cfg.budgets.clamped();
    let ram = app::ram_budget(budgets.ram);
    if let Some(png) = hdr {
        return export::headless_hdr(input, output, png, &cfg.ocio, ram as u64 / 2, ram, budgets.upload_bytes());
    }
    let shot = export::headless(input, output, &cfg.ocio, ram as u64 / 2, ram, budgets.upload_bytes())?;
    Ok((shot.w, shot.h))
}

pub fn run(initial: Option<PathBuf>, auto: bool) -> Result<()> {
    // Load the persisted viewer config (window geom + controls + recents). A
    // missing/corrupt config returns defaults — never fails (see `persist`).
    let cfg = Config::load();
    if let Some(p) = persist::config_path() {
        log::info!("config: {}", p.display());
    }

    // Effective initial path: an explicit CLI `view <file>` ALWAYS wins; only
    // when none was given do we reopen the last file from the config.
    let effective_initial = initial.or_else(|| cfg.last_file.clone());

    // Lifecycle milestone (info): the viewer is starting + what it will open.
    log::info!("exr-view launching, initial={effective_initial:?}");
    // A user-event loop: the frame cache wakes the UI from its decode threads
    // (`UserEvent::CacheProgress`) instead of the UI polling for progress.
    let event_loop = EventLoop::<app::UserEvent>::with_user_event()
        .build()
        .context("failed to create winit event loop")?;
    // On-demand redraw (kills the old 100% idle-CPU spin). Start parked on
    // `Wait`; `App::about_to_wait` re-decides the control flow every iteration:
    // `WaitUntil(fps)` while playing, ~60 Hz while decodes land, egui's
    // `repaint_delay` for animations, otherwise `Wait` (sleep at ~0% CPU).
    event_loop.set_control_flow(ControlFlow::Wait);

    let wake = app::proxy_wake(event_loop.create_proxy());
    let mut app = App::new(effective_initial, cfg, auto, wake);
    log::info!("entering winit event loop");
    let result = event_loop
        .run_app(&mut app)
        .context("winit event loop terminated with error");
    match &result {
        Ok(()) => log::info!("event loop exited cleanly"),
        Err(e) => log::error!("event loop exited with error: {e:#}"),
    }
    result
}
