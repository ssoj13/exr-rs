//! Dock layout (`egui-dock-layout`).
//!
//! A central `Viewport` (the flat GPU image) plus `Metadata` and `PixelInfo` side panels and the
//! GPU scopes. The dock never mutates the layout itself: [`Dock::show`] draws it and applies the
//! actions the user asked for afterwards, each atomically. The `Viewport` tab needs GPU access
//! (to run the colour pass + register the display texture), so the tab content also carries a
//! [`TabGpu`] handle alongside the shared [`AppState`].

use egui_dock_layout::{
    Action as DockAction, DockArea, DockContent, DockError, DockStyle, Layout, Node, SplitAxis, SplitSize,
};
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
            Self::Settings => "Settings",
        }
    }

    /// The three GPU scope tabs, reconciled each frame so a closed one frees its VRAM.
    pub const SCOPES: [DockTab; 3] = [Self::Histogram, Self::Waveform, Self::Vectorscope];

    /// Every tab, in the order the Window menu lists them (a test pins that each has a command).
    pub const ALL: [DockTab; 8] = [
        Self::Viewport,
        Self::Metadata,
        Self::PixelInfo,
        Self::Histogram,
        Self::Waveform,
        Self::Vectorscope,
        Self::Effects,
        Self::Settings,
    ];
}

/// The dock renderer's id: its gesture memory and the maximized panel live under it.
fn area_id() -> egui::Id {
    egui::Id::new("exr_view.dock")
}

/// The live arrangement plus the little view state the host keeps between frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Dock {
    /// The arrangement (persisted as an `egui_dock_layout` blob).
    pub layout: Layout<DockTab>,
    /// The panel the user last clicked into: [`Dock::open`] adds a closed tab to its group.
    focus: Option<DockTab>,
    /// A panel whose floating group comes to the front on the next draw ([`Dock::open`]).
    raise: Option<DockTab>,
}

impl Default for Dock {
    fn default() -> Self {
        Self::new(default_layout())
    }
}

/// Default dock layout: central viewport with a right-hand info column (Metadata over Pixel Info,
/// the three GPU scopes as tabs of one slot below them; drag them out to view side-by-side).
pub fn default_layout() -> Layout<DockTab> {
    let tabs = |panels: Vec<DockTab>| Node::Tabs { panels, active: 0 };
    let split = |axis, ratio, first, second| Node::Split {
        axis,
        size: SplitSize::Ratio(ratio),
        first: Box::new(first),
        second: Box::new(second),
    };
    let scopes = tabs(DockTab::SCOPES.to_vec());
    let lower = split(SplitAxis::Vertical, 0.5, tabs(vec![DockTab::PixelInfo]), scopes);
    let right = split(SplitAxis::Vertical, 0.4, tabs(vec![DockTab::Metadata]), lower);
    Layout { root: Some(split(SplitAxis::Horizontal, 0.78, tabs(vec![DockTab::Viewport]), right)), floating: Vec::new() }
}

impl Dock {
    /// The dock over `layout`, nothing focused.
    pub fn new(layout: Layout<DockTab>) -> Self {
        Self { layout, focus: None, raise: None }
    }

    /// The layout saved at the last exit, else the default one. A saved layout that no longer
    /// reads (another build's tabs, or an `egui_dock`-era blob: there is no compatibility) is
    /// logged and replaced by the default: the viewer must open either way.
    pub fn restore(blob: Option<&str>) -> Self {
        match blob.map(egui_dock_layout::from_blob) {
            Some(Ok(layout)) => Self::new(layout),
            Some(Err(e)) => {
                log::warn!("saved dock layout ignored: {e}");
                Self::default()
            }
            None => Self::default(),
        }
    }

    /// The layout as a blob for the config.
    pub fn blob(&self) -> Result<String, egui_dock_layout::Error> {
        egui_dock_layout::to_blob(&self.layout)
    }

    /// Whether `tab` is docked or floating anywhere.
    pub fn contains(&self, tab: &DockTab) -> bool {
        self.layout.contains(tab)
    }

    /// Replace the whole arrangement (Window > Reset layout, a named layout): a drag in flight
    /// and a maximized panel belong to the old one, so both are dropped.
    pub fn replace(&mut self, ctx: &egui::Context, layout: Layout<DockTab>) {
        egui_dock_layout::cancel_drag::<DockTab>(ctx, area_id());
        DockArea::<DockTab>::set_maximized(ctx, area_id(), None);
        *self = Self::new(layout);
    }

    /// Open `tab` (into the group of the panel last clicked into, else the first docked one), or
    /// bring an open one forward and raise its floating group. Never closes anything.
    pub fn open(&mut self, tab: DockTab) -> Result<(), DockError> {
        let anchor = self.focus.clone().filter(|f| *f != tab && self.contains(f)).or_else(|| {
            self.layout.root.as_ref().and_then(|root| first_panel(root)).filter(|p| **p != tab).cloned()
        });
        self.layout.apply(DockAction::Open { panel: tab.clone(), anchor })?;
        self.raise = Some(tab);
        Ok(())
    }

    /// Whether `tab` is drawn this frame: the active tab (or an open accordion entry) of its
    /// group, floating or docked, and not hidden behind another group's Maximize.
    pub fn shown(&self, ctx: &egui::Context, tab: &DockTab) -> bool {
        if self.layout.floating.iter().any(|g| g.panels.get(g.active) == Some(tab)) {
            return true;
        }
        let Some(root) = &self.layout.root else { return false };
        let Some(group) = group_of(root, tab) else { return false };
        let active = match group {
            Node::Tabs { panels, active } => panels.get(*active) == Some(tab),
            Node::Stack { entries } => entries.iter().any(|e| e.panel == *tab && e.open),
            Node::Split { .. } => false,
        };
        // The renderer falls back to the whole tree when the maximized panel left it.
        let maximized = DockArea::<DockTab>::maximized(ctx, area_id());
        active && maximized.and_then(|m| group_of(root, &m)).is_none_or(|g| std::ptr::eq(g, group))
    }

    /// Draw the dock into the rest of `ui`, then apply what the user asked for (each action is
    /// atomic: a rejected one is logged and changes nothing).
    pub fn show(&mut self, ui: &mut egui::Ui, tab_height: f32, content: &mut impl DockContent<DockTab>) {
        let style = DockStyle { tab_height, ..DockStyle::from_ui(ui) };
        let mut area = DockArea::new(area_id());
        if let Some(tab) = self.raise.take() {
            area = area.raise_panel(tab);
        }
        let out = area.show_with(ui, &self.layout, &style, content);
        if let Some(e) = out.error {
            log::error!("dock not drawn: {e}");
        }
        if let Some(tab) = out.focused {
            self.focus = Some(tab);
        }
        for action in out.actions {
            if let Err(e) = self.layout.apply(action.clone()) {
                log::warn!("dock: {action:?} refused: {e}");
            }
        }
    }
}

/// The first panel of the docked tree, depth first (the tree is validated: bounded depth).
fn first_panel(node: &Node<DockTab>) -> Option<&DockTab> {
    match node {
        Node::Tabs { panels, .. } => panels.first(),
        Node::Stack { entries } => entries.first().map(|e| &e.panel),
        Node::Split { first, second, .. } => first_panel(first).or_else(|| first_panel(second)),
    }
}

/// The docked group (tab group or accordion) holding `tab` (the tree is validated: bounded depth).
fn group_of<'a>(node: &'a Node<DockTab>, tab: &DockTab) -> Option<&'a Node<DockTab>> {
    match node {
        Node::Tabs { panels, .. } => panels.contains(tab).then_some(node),
        Node::Stack { entries } => entries.iter().any(|e| e.panel == *tab).then_some(node),
        Node::Split { first, second, .. } => group_of(first, tab).or_else(|| group_of(second, tab)),
    }
}

/// Bridges the dock's panel rendering to the app's per-tab UI.
///
/// Borrows `AppState` mutably (selected file, view controls) and a
/// [`TabGpu`] handle so the `Viewport` tab can drive the GPU colour pass.
pub struct DockTabViewer<'a> {
    pub state: &'a mut AppState,
    pub gpu: &'a mut dyn TabGpu,
}

impl DockContent<DockTab> for DockTabViewer<'_> {
    fn title(&self, tab: &DockTab) -> String {
        tab.name().to_owned()
    }

    fn body(&mut self, ui: &mut egui::Ui, tab: &DockTab) {
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
            DockTab::Settings => self.state.ui_settings(ui),
        })
    }

    /// The tab's context menu (and ≡ button): every panel, checked when open, to open or bring
    /// forward. They are the Window menu's entries and queue the same `Action::Tab` (run by
    /// `build_ui` once the dock is released). The dock's own items (Float, Move to, Maximize,
    /// Close) follow ours; Close is always among them (every tab is closable), hence the separator.
    fn panel_menu(&mut self, ui: &mut egui::Ui, _tab: &DockTab) {
        for panel in DockTab::ALL {
            let Some(cmd) = commands::for_tab(&panel) else { continue };
            let on = commands::enabled(&cmd.action, self.state.avail());
            let open = self.state.open_tabs.contains(&panel);
            if ui.add_enabled(on, egui::Button::selectable(open, cmd.label)).clicked() {
                self.state.queue.push(cmd.action.clone());
                ui.close();
            }
        }
        ui.separator();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A saved layout comes back as saved (a scope tab removed stays removed); no layout, one
    /// that no longer reads, or an `egui_dock`-era blob gives the default.
    #[test]
    fn restore_brings_back_the_saved_layout() {
        let mut dock = Dock::default();
        dock.layout.apply(DockAction::Close { panel: DockTab::Histogram }).unwrap();
        let blob = dock.blob().unwrap();
        assert_eq!(Dock::restore(Some(&blob)).layout, dock.layout);
        assert!(!Dock::restore(Some(&blob)).contains(&DockTab::Histogram));
        assert_eq!(Dock::restore(None).layout, default_layout());
        assert_eq!(Dock::restore(Some("{\"not\": \"a layout\"}")).layout, default_layout());
        let egui_dock = r#"{"surfaces":[{"Main":{"nodes":[{"Leaf":{"tabs":["Viewport"],"active":0}}]}}]}"#;
        assert_eq!(Dock::restore(Some(egui_dock)).layout, default_layout());
    }

    /// The default layout is valid and holds the viewport, the info column and the scopes once.
    #[test]
    fn default_layout_is_valid() {
        let layout = default_layout();
        layout.validate().unwrap();
        let mut panels: Vec<DockTab> = layout.panels().into_iter().cloned().collect();
        panels.sort_by_key(|t| DockTab::ALL.iter().position(|a| a == t));
        assert_eq!(panels, DockTab::ALL[..6]);
    }

    /// `shown`: the active tab of its group only, and not while another group is maximized.
    #[test]
    fn shown_follows_the_active_tab_and_maximize() {
        let ctx = egui::Context::default();
        let mut dock = Dock::default();
        assert!(dock.shown(&ctx, &DockTab::Viewport) && dock.shown(&ctx, &DockTab::Histogram));
        assert!(!dock.shown(&ctx, &DockTab::Waveform), "behind the histogram tab");
        dock.open(DockTab::Waveform).unwrap();
        assert!(dock.shown(&ctx, &DockTab::Waveform) && !dock.shown(&ctx, &DockTab::Histogram));
        DockArea::<DockTab>::set_maximized(&ctx, area_id(), Some(DockTab::Metadata));
        assert!(!dock.shown(&ctx, &DockTab::Viewport) && dock.shown(&ctx, &DockTab::Metadata));
        dock.replace(&ctx, default_layout());
        assert!(dock.shown(&ctx, &DockTab::Viewport), "a replaced layout is not maximized");
    }
}
