//! Dock layout (egui_dock 0.19).
//!
//! A central `Viewport` (the flat GPU image) plus `Metadata` and
//! `PixelInfo` side panels. The `Viewport` tab needs GPU access (to run the
//! colour pass + register the display texture), so the tab viewer also carries a
//! [`TabGpu`] handle alongside the shared [`AppState`].

use egui_dock::{DockState, NodeIndex, NodePath, TabViewer};
use serde::{Deserialize, Serialize};

use crate::app::AppState;
use crate::commands;
use crate::viewport::TabGpu;

/// Tab identifiers for dock panels. (De)serializable so layouts can persist.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DockTab {
    /// 2-D image viewport (the flat GPU image).
    Viewport,
    /// EXR header metadata.
    Metadata,
    /// Pixel inspector / readout.
    PixelInfo,
    /// GPU histogram scope.
    Histogram,
    /// GPU waveform scope.
    Waveform,
    /// GPU vectorscope.
    Vectorscope,
    /// OpenFX effect over the shown image: plug-in, parameters (feature `ofx`; without it the tab
    /// says so, so a saved layout loads in either build).
    Effects,
    /// The OCIO display transform: config, input, display, view, look.
    Colour,
    /// Preferences (memory budgets).
    Settings,
}

impl DockTab {
    /// Human-readable tab title.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Viewport => "Viewport",
            Self::Metadata => "Metadata",
            Self::PixelInfo => "Pixel Info",
            Self::Histogram => "Histogram",
            Self::Waveform => "Waveform",
            Self::Vectorscope => "Vectorscope",
            Self::Effects => "Effects",
            Self::Colour => "Colour",
            Self::Settings => "Settings",
        }
    }

    /// The three GPU scope tabs, reconciled each frame so a closed one frees its VRAM.
    pub const SCOPES: [DockTab; 3] = [Self::Histogram, Self::Waveform, Self::Vectorscope];

    /// Every tab, in the order the Window menu lists them (a test pins that each has a command).
    pub const ALL: [DockTab; 9] = [
        Self::Viewport,
        Self::Metadata,
        Self::PixelInfo,
        Self::Histogram,
        Self::Waveform,
        Self::Vectorscope,
        Self::Effects,
        Self::Colour,
        Self::Settings,
    ];
}

/// Default dock layout: central viewport with a right-hand info column.
/// The layout saved at the last exit, else the default one. A saved layout that no
/// longer reads (written by a build with other tabs) is logged and replaced by the
/// default: the viewer must open either way.
pub fn restore(blob: Option<&str>) -> DockState<DockTab> {
    match blob.map(egui_dock_layout::from_blob) {
        Some(Ok(state)) => state,
        Some(Err(e)) => {
            log::warn!("saved dock layout ignored: {e}");
            default_dock_state()
        }
        None => default_dock_state(),
    }
}

pub fn default_dock_state() -> DockState<DockTab> {
    let mut dock = DockState::new(vec![DockTab::Viewport]);

    // Right column: Metadata over Pixel Info.
    let [_center, right] =
        dock.main_surface_mut()
            .split_right(NodeIndex::root(), 0.78, vec![DockTab::Metadata]);
    let [_meta, info] = dock
        .main_surface_mut()
        .split_below(right, 0.4, vec![DockTab::PixelInfo]);
    // Stack the three GPU scopes as tabs below the pixel-info panel so
    // they share one slot by default; drag them out to view side-by-side.
    dock.main_surface_mut().split_below(
        info,
        0.5,
        vec![DockTab::Histogram, DockTab::Waveform, DockTab::Vectorscope],
    );

    dock
}

/// Bridges egui_dock tab rendering to the app's per-tab UI.
///
/// Borrows `AppState` mutably (selected file, view controls) and a
/// [`TabGpu`] handle so the `Viewport` tab can drive the GPU colour pass.
pub struct DockTabViewer<'a> {
    pub state: &'a mut AppState,
    pub gpu: &'a mut dyn TabGpu,
}

impl TabViewer for DockTabViewer<'_> {
    type Tab = DockTab;

    fn id(&mut self, tab: &mut DockTab) -> egui::Id {
        egui::Id::new(&*tab)
    }

    fn title(&mut self, tab: &mut DockTab) -> egui::WidgetText {
        tab.name().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut DockTab) {
        // Every tab is a keyboard panel, keys or not: it takes the keyboard from the previous owner.
        commands::panel(ui, commands::Scope::of(tab), |ui| match tab {
            // Single source of responsibility for GPU display: both the single
            // image and the sequence's current frame route through this.
            DockTab::Viewport => self.state.ui_viewport(ui, self.gpu),
            DockTab::Metadata => self.state.ui_metadata(ui),
            DockTab::PixelInfo => self.state.ui_pixel_info(ui),
            // GPU scopes: each computes into its own texture from the
            // linear display texture and paints it (egui-wgpu native texture).
            scope @ (DockTab::Histogram | DockTab::Waveform | DockTab::Vectorscope) => {
                self.state.display_scope(ui, scope, self.gpu)
            }
            DockTab::Effects => self.state.ui_effects(ui),
            DockTab::Colour => self.state.ui_colour(ui),
            DockTab::Settings => self.state.ui_settings(ui),
        })
    }

    /// Right-click on a tab: every panel, checked when open, to open or bring forward. They are
    /// the Window menu's entries and queue the same `Action::Tab` (run by `build_ui` once the dock
    /// is released). egui_dock appends its own items after ours, hence the separator.
    fn context_menu(&mut self, ui: &mut egui::Ui, tab: &mut DockTab, _path: NodePath) {
        for panel in DockTab::ALL {
            let Some(cmd) = commands::for_tab(&panel) else { continue };
            let on = commands::enabled(&cmd.action, self.state.avail());
            let open = self.state.open_tabs.contains(&panel);
            if ui.add_enabled(on, egui::Button::selectable(open, cmd.label)).clicked() {
                self.state.queue.push(cmd.action.clone());
                ui.close();
            }
        }
        // egui_dock 0.21.1 (`leaf.rs` 431-536) adds, after this, Detach (never for a lone tab of
        // a floating window), Close (`show_close_buttons && is_closeable`) and Hide tab bar
        // (`hidable_tab_bars`). The DockArea keeps its defaults (close buttons on, hidable bars
        // off), so Close is there exactly when the tab is closeable, and it is the one of the
        // three that is always there then (Detach is a subset case): no rule dangles.
        if self.is_closeable(tab) {
            ui.separator();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tab of a layout, in iteration order.
    fn tabs(state: &DockState<DockTab>) -> Vec<DockTab> {
        state.iter_all_tabs().map(|(_, t)| t.clone()).collect()
    }

    /// A saved layout comes back as saved (a scope tab removed stays removed); no
    /// layout or one that no longer reads gives the default.
    #[test]
    fn restore_brings_back_the_saved_layout() {
        let mut state = default_dock_state();
        let loc = state.find_tab(&DockTab::Histogram).expect("default layout has a histogram");
        state.remove_tab(loc);
        let blob = egui_dock_layout::to_blob(&state).unwrap();
        assert_eq!(tabs(&restore(Some(&blob))), tabs(&state));
        assert!(!tabs(&restore(Some(&blob))).contains(&DockTab::Histogram));
        assert_eq!(tabs(&restore(None)), tabs(&default_dock_state()));
        assert_eq!(tabs(&restore(Some("{\"not\": \"a layout\"}"))), tabs(&default_dock_state()));
    }
}
