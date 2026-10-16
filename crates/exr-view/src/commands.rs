//! The one command registry of the viewer: every action with its menu label and keys.
//!
//! The menu bar (`egui-menubar`, [`MENU`]) and the keyboard ([`pressed`]) both read
//! [`COMMANDS`], so the shortcut a menu prints is by construction the one that fires,
//! and no action exists on the keyboard only. What an action does is `app::run`'s.
//!
//! Keys are PANEL-SCOPED (`egui-hotkeys`): a command lists where its keys are live
//! ([`Where`]). [`Where::Global`] keys fire over any panel; the others only while their
//! [`Scope`] owns the keyboard (the panel under the pointer, else the last one clicked, else
//! the viewport), so the arrows fold the channel tree over Metadata and step frames over the
//! viewport and the timeline. Fit, the channel isolates and the display transform are global: a
//! viewer wants F / R / G / B / A / T whichever panel the pointer is over.

use std::sync::LazyLock;

use egui::{Key, KeyboardShortcut, Modifiers, Rect, Ui};
use egui_hotkeys::{Binding, Keymap};
use egui_menubar::MenuNode::{self, Dyn, Item, Sep, Sub};

use crate::dock::DockTab;
use crate::image::ChannelDisplay;

/// A keyboard panel: one per dock tab (the three scope tabs share one) plus the transport
/// timeline. EVERY panel is registered each frame, keys or not, so a panel without keys still
/// takes the keyboard from the previous owner (typing in Settings must not frame the image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The image viewport (also drawn alone when the panels are hidden).
    Viewport,
    /// The transport bar's timeline.
    Timeline,
    /// EXR header and the channel tree.
    Metadata,
    /// Pixel readout.
    PixelInfo,
    /// Histogram, waveform and vectorscope.
    Scopes,
    /// The OpenFX tab.
    Effects,
    /// Preferences.
    Settings,
}

impl Scope {
    /// Every scope.
    pub const ALL: [Scope; 7] = [
        Scope::Viewport,
        Scope::Timeline,
        Scope::Metadata,
        Scope::PixelInfo,
        Scope::Scopes,
        Scope::Effects,
        Scope::Settings,
    ];

    /// The scope of a dock tab. Exhaustive on purpose: a new tab must pick its scope.
    pub fn of(tab: &DockTab) -> Self {
        match tab {
            DockTab::Viewport => Scope::Viewport,
            DockTab::Metadata => Scope::Metadata,
            DockTab::PixelInfo => Scope::PixelInfo,
            DockTab::Histogram | DockTab::Waveform | DockTab::Vectorscope => Scope::Scopes,
            DockTab::Effects => Scope::Effects,
            DockTab::Settings => Scope::Settings,
        }
    }
}

/// Where a command's keys are live.
#[derive(Clone, Copy, Debug)]
pub enum Where {
    /// Over any panel: the end of every context chain (a panel table binding the same chord
    /// would override it).
    Global,
    /// Only while one of these panels owns the keyboard.
    In(&'static [Scope]),
}

/// Something the user can ask the viewer to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// File > Open dialog.
    Open,
    /// Save the view as a PNG (`export.rs`).
    SaveView,
    /// Save the view as HDR: display light EXR or HDR10 PQ PNG (`export.rs`).
    SaveViewHdr,
    /// Quit (Esc also quits; it is handled at the window, where egui's use of it is known).
    Quit,
    /// A viewport navigation, applied by the viewport to the shown image.
    Nav(NavCmd),
    /// Borderless fullscreen on/off.
    Fullscreen,
    /// Hide / show every panel but the viewport.
    HidePanels,
    /// Sequence playback forward on/off.
    PlayPause,
    /// Sequence playback backwards on/off.
    PlayReverse,
    /// First frame of the sequence.
    First,
    /// Last frame of the sequence.
    Last,
    /// Open the previous EXR of the folder (by name; a sequence counts as one item).
    FilePrev,
    /// Open the next EXR of the folder.
    FileNext,
    /// Previous frame of the sequence.
    StepPrev,
    /// Next frame of the sequence.
    StepNext,
    /// Show one channel of RGBA alone; again to go back to RGBA.
    Isolate(ChannelDisplay),
    /// The next display transform: the tone operators (none, Reinhard, AgX), then OCIO.
    CycleDisplay,
    /// Open a dock tab, or bring an open one forward (closing is the tab's own X).
    Tab(DockTab),
    /// Put the panels back where a fresh install has them.
    ResetLayout,
}

/// A viewport navigation ([`Action::Nav`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavCmd {
    /// Fit the image to the viewport.
    Fit,
    /// 100%: one image pixel per screen pixel.
    Home,
    /// Zoom in one step about the viewport centre.
    ZoomIn,
    /// Zoom out one step about the viewport centre.
    ZoomOut,
}

/// How a [`Chord`] is matched (see `egui_hotkeys::Binding`).
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// The key as the layout reports it, with exactly the stated modifiers.
    Exact,
    /// The key position, whatever the layout types (digits).
    Physical,
    /// Exactly the stated modifiers, ignoring a held Shift ("+" is Shift+= on most layouts).
    LooseShift,
}

/// One key of a command: the shortcut a menu prints, plus how it is matched.
#[derive(Clone, Copy, Debug)]
pub struct Chord {
    /// What the menu prints and the binding is built from.
    pub shortcut: KeyboardShortcut,
    kind: Kind,
    /// Also fires on the auto-repeat of a held key: stepping and zooming keep going while the
    /// key is down, as `consume_shortcut` did for every key. Toggles (fit, channels, tone,
    /// fullscreen, play, open) do not repeat: holding R would flicker the channel.
    repeat: bool,
}

impl Chord {
    /// This chord, repeating while held.
    const fn rep(mut self) -> Self {
        self.repeat = true;
        self
    }
}

/// One registry entry.
pub struct Command {
    /// Stable id, named by [`MENU`].
    pub id: &'static str,
    /// Menu label.
    pub label: &'static str,
    /// Keys that run it; the first is the one the menu prints.
    pub keys: &'static [Chord],
    /// Where the keys are live.
    pub at: Where,
    /// What it does.
    pub action: Action,
}

const fn chord(modifiers: Modifiers, key: Key, kind: Kind) -> Chord {
    Chord { shortcut: KeyboardShortcut::new(modifiers, key), kind, repeat: false }
}

const fn key(key: Key) -> Chord {
    chord(Modifiers::NONE, key, Kind::Exact)
}

const fn ctrl(key: Key) -> Chord {
    chord(Modifiers::COMMAND, key, Kind::Exact)
}

const fn shift(key: Key) -> Chord {
    chord(Modifiers::SHIFT, key, Kind::Exact)
}

/// The key at this position, e.g. the `0` key whatever it types.
const fn phys(key: Key) -> Chord {
    chord(Modifiers::NONE, key, Kind::Physical)
}

/// Ctrl + `key`, with or without Shift.
const fn ctrl_any(key: Key) -> Chord {
    chord(Modifiers::COMMAND, key, Kind::LooseShift)
}

/// Frame stepping works over the viewport and the timeline, and nowhere else.
const FRAMES: Where = Where::In(&[Scope::Viewport, Scope::Timeline]);
/// 100% and zoom: the viewport's own (F, the channel isolates and the display transform are
/// global).
const VIEW: Where = Where::In(&[Scope::Viewport]);

/// Every command of the viewer.
pub const COMMANDS: &[Command] = &[
    Command { id: "file.open", label: "Open…", keys: &[ctrl(Key::O)], at: Where::Global, action: Action::Open },
    Command { id: "file.save_view", label: "Save view as PNG…", keys: &[ctrl(Key::S)], at: Where::Global, action: Action::SaveView },
    Command { id: "file.save_view_hdr", label: "Save view as HDR (EXR / PQ or HLG PNG)…", keys: &[], at: Where::Global, action: Action::SaveViewHdr },
    Command { id: "file.prev", label: "Previous file in folder", keys: &[key(Key::PageUp)], at: Where::Global, action: Action::FilePrev },
    Command { id: "file.next", label: "Next file in folder", keys: &[key(Key::PageDown)], at: Where::Global, action: Action::FileNext },
    Command { id: "file.quit", label: "Quit", keys: &[], at: Where::Global, action: Action::Quit },
    Command { id: "view.fit", label: "Fit", keys: &[key(Key::F)], at: Where::Global, action: Action::Nav(NavCmd::Fit) },
    Command {
        id: "view.home",
        label: "Actual size (100%)",
        keys: &[key(Key::H), phys(Key::Num0)],
        at: VIEW,
        action: Action::Nav(NavCmd::Home),
    },
    // `Equals` next to `Plus`: `+` is Shift+= on most layouts, and egui reports either.
    Command {
        id: "view.zoom_in",
        label: "Zoom in",
        keys: &[ctrl_any(Key::Plus).rep(), ctrl(Key::Equals).rep()],
        at: VIEW,
        action: Action::Nav(NavCmd::ZoomIn),
    },
    Command { id: "view.zoom_out", label: "Zoom out", keys: &[ctrl(Key::Minus).rep()], at: VIEW, action: Action::Nav(NavCmd::ZoomOut) },
    Command { id: "view.fullscreen", label: "Fullscreen", keys: &[key(Key::Z)], at: Where::Global, action: Action::Fullscreen },
    Command { id: "view.panels", label: "Hide panels", keys: &[shift(Key::Z)], at: Where::Global, action: Action::HidePanels },
    Command { id: "view.red", label: "Red", keys: &[key(Key::R)], at: Where::Global, action: Action::Isolate(ChannelDisplay::R) },
    Command { id: "view.green", label: "Green", keys: &[key(Key::G)], at: Where::Global, action: Action::Isolate(ChannelDisplay::G) },
    Command { id: "view.blue", label: "Blue", keys: &[key(Key::B)], at: Where::Global, action: Action::Isolate(ChannelDisplay::B) },
    Command { id: "view.alpha", label: "Alpha", keys: &[key(Key::A)], at: Where::Global, action: Action::Isolate(ChannelDisplay::A) },
    Command { id: "view.tonemap", label: "Next display transform", keys: &[key(Key::T)], at: Where::Global, action: Action::CycleDisplay },
    Command { id: "tab.viewport", label: "Viewport", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Viewport) },
    Command { id: "tab.metadata", label: "Metadata", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Metadata) },
    Command { id: "tab.pixel_info", label: "Pixel Info", keys: &[], at: Where::Global, action: Action::Tab(DockTab::PixelInfo) },
    Command { id: "tab.histogram", label: "Histogram", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Histogram) },
    Command { id: "tab.waveform", label: "Waveform", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Waveform) },
    Command { id: "tab.vectorscope", label: "Vectorscope", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Vectorscope) },
    Command { id: "tab.effects", label: "Effects", keys: &[], at: Where::Global, action: Action::Tab(DockTab::Effects) },
    Command { id: "window.reset_layout", label: "Reset layout", keys: &[], at: Where::Global, action: Action::ResetLayout },
    Command { id: "tab.settings", label: "Settings", keys: &[ctrl(Key::Comma)], at: Where::Global, action: Action::Tab(DockTab::Settings) },
    Command { id: "play.toggle", label: "Play / pause", keys: &[key(Key::Space)], at: Where::Global, action: Action::PlayPause },
    Command { id: "play.reverse", label: "Play backwards", keys: &[shift(Key::Space)], at: Where::Global, action: Action::PlayReverse },
    Command {
        id: "play.prev",
        label: "Previous frame",
        keys: &[key(Key::ArrowLeft).rep(), key(Key::Comma).rep()],
        at: FRAMES,
        action: Action::StepPrev,
    },
    Command {
        id: "play.next",
        label: "Next frame",
        keys: &[key(Key::ArrowRight).rep(), key(Key::Period).rep()],
        at: FRAMES,
        action: Action::StepNext,
    },
    Command { id: "play.first", label: "First frame", keys: &[key(Key::Home)], at: FRAMES, action: Action::First },
    Command { id: "play.last", label: "Last frame", keys: &[key(Key::End)], at: FRAMES, action: Action::Last },
];

/// The key tables `egui-hotkeys` reads, derived from [`COMMANDS`] once: the global table plus
/// one per [`Scope`]. A command listed in several scopes appears in each.
pub struct Keys {
    global: Vec<Binding<&'static str>>,
    scoped: Vec<Vec<Binding<&'static str>>>,
}

impl Keys {
    fn build() -> Self {
        let mut keys = Keys { global: Vec::new(), scoped: vec![Vec::new(); Scope::ALL.len()] };
        for cmd in COMMANDS {
            for chord in cmd.keys {
                let KeyboardShortcut { modifiers, logical_key } = chord.shortcut;
                let b = match chord.kind {
                    Kind::Exact => Binding::new(cmd.id, logical_key, modifiers),
                    Kind::Physical => Binding::physical(cmd.id, logical_key, modifiers),
                    Kind::LooseShift => Binding::additive(cmd.id, logical_key, modifiers),
                };
                let b = if chord.repeat { b.repeats() } else { b };
                match cmd.at {
                    Where::Global => keys.global.push(b),
                    Where::In(scopes) => scopes.iter().for_each(|s| keys.scoped[*s as usize].push(b)),
                }
            }
        }
        keys
    }
}

impl Keymap for Keys {
    type Scope = Scope;
    type Cmd = &'static str;

    fn global(&self) -> &[Binding<&'static str>] {
        &self.global
    }

    fn bindings(&self, scope: Scope) -> &[Binding<&'static str>] {
        self.scoped.get(scope as usize).map_or(&[], Vec::as_slice)
    }

    /// Before the first click (and after one outside every panel) the viewport owns the keys:
    /// this is a viewer, so F / R / G / B / the arrows work right after launch.
    fn fallback(&self) -> Option<Scope> {
        Some(Scope::Viewport)
    }
}

static KEYS: LazyLock<Keys> = LazyLock::new(Keys::build);

/// Declare a keyboard panel for this frame. Every dock tab, the timeline and the viewport drawn
/// alone (panels hidden) call it every frame, keys or not; see [`Scope`].
pub fn register(ui: &Ui, scope: Scope, rect: Rect) {
    egui_hotkeys::register(&*KEYS, ui, scope, rect);
}

/// One keyboard panel filling `ui`: declare it, then draw `body`. The dock tabs and the viewport
/// drawn alone (panels hidden) both go through here, so a panel cannot be drawn without taking
/// the keyboard.
pub fn panel<R>(ui: &mut Ui, scope: Scope, body: impl FnOnce(&mut Ui) -> R) -> R {
    register(ui, scope, ui.max_rect());
    body(ui)
}

/// The viewport's toolbar strip. It floats on a layer of its own (an `egui::Area`), so the
/// pointer over it would otherwise hit no panel; it is part of the viewport.
pub fn strip(ctx: &egui::Context, bar: &egui_viewport_toolbar::ToolbarResponse) {
    egui_hotkeys::register_at(&*KEYS, ctx, bar.layer, Scope::Viewport, bar.rect);
}

/// Who owns the keyboard right now (what [`pressed`] routes to).
#[cfg(test)]
pub fn active(ctx: &egui::Context) -> Option<Scope> {
    egui_hotkeys::active(&*KEYS, ctx)
}

/// The menu bar. `Dyn` slots are drawn by the host: `file.recent` (the recent files),
/// `view.playhead` (the playhead line opacity), `view.preload` (the cache strategy).
pub const MENU: &[MenuNode] = &[
    Sub(
        "File",
        &[
            Item("file.open"),
            Dyn("file.recent"),
            Item("file.save_view"),
            Item("file.save_view_hdr"),
            Sep,
            Item("file.prev"),
            Item("file.next"),
            Sep,
            Item("file.quit"),
        ],
    ),
    Sub(
        "View",
        &[
            Item("view.fit"),
            Item("view.home"),
            Item("view.zoom_in"),
            Item("view.zoom_out"),
            Sep,
            Item("view.fullscreen"),
            Item("view.panels"),
            Sep,
            Item("view.red"),
            Item("view.green"),
            Item("view.blue"),
            Item("view.alpha"),
            Item("view.tonemap"),
            Sep,
            Dyn("view.playhead"),
            Sep,
            Dyn("view.preload"),
        ],
    ),
    // One entry per dock tab (checked = open; a click opens it or brings it forward), then the layout reset.
    Sub(
        "Window",
        &[
            Item("tab.viewport"),
            Item("tab.metadata"),
            Item("tab.pixel_info"),
            Sep,
            Item("tab.histogram"),
            Item("tab.waveform"),
            Item("tab.vectorscope"),
            Sep,
            Item("tab.effects"),
            Item("tab.settings"),
            Sep,
            Item("window.reset_layout"),
        ],
    ),
    Sub(
        "Playback",
        &[
            Item("play.toggle"),
            Item("play.reverse"),
            Sep,
            Item("play.prev"),
            Item("play.next"),
            Item("play.first"),
            Item("play.last"),
        ],
    ),
];

/// What is open, for [`enabled`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Avail {
    /// A sequence is open.
    pub seq: bool,
    /// A file (image or sequence) is open.
    pub file: bool,
}

/// Whether `action` can run now: playback needs a sequence, folder stepping an open
/// file, the Effects tab a build with the OpenFX host. The menu greys out what this
/// refuses and [`pressed`] leaves such keys to the widgets (the arrows then still move
/// in the channel tree, say).
pub fn enabled(action: &Action, avail: Avail) -> bool {
    match action {
        Action::PlayPause
        | Action::PlayReverse
        | Action::StepPrev
        | Action::StepNext
        | Action::First
        | Action::Last => avail.seq,
        Action::FilePrev | Action::FileNext | Action::SaveView | Action::SaveViewHdr => avail.file,
        Action::Tab(DockTab::Effects) => cfg!(feature = "ofx"),
        _ => true,
    }
}

/// The command named `id`.
pub fn find(id: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.id == id)
}

/// The command that opens / closes `tab` (every dock tab has one: `every_tab_has_a_command`),
/// so the Window menu and the tab context menu list the same labels and run the same action.
pub fn for_tab(tab: &DockTab) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| matches!(&c.action, Action::Tab(t) if t == tab))
}

/// The text a menu prints for `cmd`'s keys (its first key, in the OS's notation).
pub fn shortcut_text(ctx: &egui::Context, cmd: &Command) -> String {
    cmd.keys.first().map_or(String::new(), |k| ctx.format_shortcut(&k.shortcut))
}

impl Command {
    /// Every key of it is a Ctrl/Cmd chord: the only keys that still run while a modal of the
    /// viewer (the file dialog) is open, since bare keys there belong to the dialog.
    fn chorded(&self) -> bool {
        !self.keys.is_empty() && self.keys.iter().all(|k| k.shortcut.modifiers.command)
    }
}

/// The actions whose keys were pressed this frame (the keys are consumed), routed by
/// `egui-hotkeys`: global keys always, the others only for the [`Scope`] that owns the
/// keyboard. While a text field has the keyboard, egui-hotkeys leaves it the keys it acts on
/// itself (Space, arrows, letters, Ctrl+A …); while a `modal` of the viewer is open only the
/// Ctrl/Cmd chord commands fire. Keys of a command that is not [`enabled`] (or held back by the
/// modal) are not consumed at all.
///
/// Chords match exactly, so Shift+Z is "Hide panels" only and plain Z "Fullscreen" only, with
/// no ordering between them. Panel ownership comes from the PREVIOUS frame's geometry, which is
/// why this can run once at the top of the frame, before any panel is drawn.
pub fn pressed(ctx: &egui::Context, modal: bool, avail: Avail) -> Vec<Action> {
    let on = |id| find(id).is_some_and(|c| enabled(&c.action, avail) && (!modal || c.chorded()));
    let ids = egui_hotkeys::pressed(&*KEYS, ctx, on);
    ids.into_iter().filter_map(find).map(|c| c.action.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu_items(nodes: &'static [MenuNode], out: &mut Vec<&'static str>) {
        for n in nodes {
            match n {
                Item(id) => out.push(id),
                Sub(_, children) => menu_items(children, out),
                Sep | Dyn(_) | MenuNode::Header(_) => {}
            }
        }
    }

    /// The menu and the registry name the same commands, each once.
    #[test]
    fn menu_and_registry_agree() {
        let mut items = Vec::new();
        menu_items(MENU, &mut items);
        let mut ids: Vec<&str> = COMMANDS.iter().map(|c| c.id).collect();
        for id in &items {
            assert!(find(id).is_some(), "menu item {id} names no command");
        }
        items.sort_unstable();
        ids.sort_unstable();
        let unique = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), unique, "duplicate command id");
        assert_eq!(items, ids, "every command is in the menu exactly once");
    }

    /// The Window menu lists every dock tab (so none is unreachable once closed) and the
    /// layout reset, and the tab context menu finds each tab's command.
    #[test]
    fn every_tab_has_a_command() {
        let mut items = Vec::new();
        let window = MENU.iter().find_map(|n| match n {
            Sub("Window", children) => Some(*children),
            _ => None,
        });
        menu_items(window.expect("a Window menu"), &mut items);
        for tab in DockTab::ALL {
            let cmd = for_tab(&tab).unwrap_or_else(|| panic!("no command opens {tab:?}"));
            assert!(items.contains(&cmd.id), "{} is not in the Window menu", cmd.id);
        }
        assert!(items.contains(&"window.reset_layout"));
        // The reset is listed once: it left the View menu.
        let mut all = Vec::new();
        menu_items(MENU, &mut all);
        assert_eq!(all.iter().filter(|id| **id == "window.reset_layout").count(), 1);
    }

    /// A tab whose build cannot show it is greyed out in both menus by the one rule.
    #[test]
    fn effects_tab_follows_the_build() {
        let cmd = for_tab(&DockTab::Effects).unwrap();
        assert_eq!(enabled(&cmd.action, Avail::default()), cfg!(feature = "ofx"));
        assert!(enabled(&for_tab(&DockTab::Metadata).unwrap().action, Avail::default()));
    }

    /// What one pressed key did: the actions `pressed` returned and whether the key is still
    /// there for the widgets afterwards.
    #[derive(Debug, PartialEq)]
    struct Pressed {
        actions: Vec<Action>,
        left: bool,
    }

    /// Run `pressed` over a frame in which `key` is pressed with `mods`, with `scope` the panel
    /// the pointer rests over (`None`: no panel registered, nothing clicked). Mirrors the app's
    /// order: the keys are read at the top of the frame, from the geometry of the frame before.
    fn press_in(scope: Option<Scope>, key: Key, mods: Modifiers, typing: bool, has_seq: bool) -> Pressed {
        press_at(scope, key, None, mods, typing, has_seq)
    }

    /// [`press_in`] for a key that reports its position (`physical_key`) apart from what it types.
    fn press_at(scope: Option<Scope>, key: Key, physical_key: Option<Key>, mods: Modifiers, typing: bool, has_seq: bool) -> Pressed {
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let run = |events: Vec<egui::Event>, f: &mut dyn FnMut(&mut Ui)| {
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let mut output = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().show(root, |ui| f(ui));
            });
            // No renderer here: the font atlas upload egui asks for is dropped on purpose.
            output.textures_delta.clear();
        };
        let draw = |ui: &mut Ui| {
            if let Some(scope) = scope {
                panel(ui, scope, |_| {});
            }
        };
        run(vec![egui::Event::PointerMoved(screen.center())], &mut |ui| draw(ui));
        let event = egui::Event::Key { key, physical_key, pressed: true, repeat: false, modifiers: mods };
        let mut got = Pressed { actions: Vec::new(), left: false };
        run(vec![event], &mut |ui| {
            got.actions = pressed(ui.ctx(), typing, Avail { seq: has_seq, file: has_seq });
            got.left = ui.input(|i| i.key_pressed(key));
            draw(ui);
        });
        got
    }

    /// [`press_in`] over the viewport, with a sequence open.
    fn press(key: Key, mods: Modifiers, typing: bool) -> Vec<Action> {
        press_in(Some(Scope::Viewport), key, mods, typing, true).actions
    }

    fn step(scope: Scope, key: Key, has_seq: bool) -> Pressed {
        press_in(Some(scope), key, Modifiers::NONE, false, has_seq)
    }

    /// Every dock tab is a scope (none is a keyboard hole that lets the previous owner keep the
    /// keys), the timeline is one more, and together they are all the scopes there are.
    #[test]
    fn every_tab_is_a_scope() {
        let mut seen: Vec<Scope> = DockTab::ALL.iter().map(Scope::of).collect();
        seen.push(Scope::Timeline);
        for scope in Scope::ALL {
            assert!(seen.contains(&scope), "no panel is {scope:?}");
        }
        // `Scope::ALL` indexes the key tables: its order is the enum's.
        for (i, scope) in Scope::ALL.iter().enumerate() {
            assert_eq!(*scope as usize, i);
        }
    }

    /// No two bindings of one table (global, or one panel's) can match the same key press: the
    /// first listed would silently win. A panel chord equal to a global one is an override and
    /// is not checked here (there is none today).
    #[test]
    fn no_chord_overlaps_inside_a_table() {
        assert_eq!(egui_hotkeys::conflicts(&*KEYS, &Scope::ALL), []);
    }

    /// Each binding resolves to its own command in its own table, and the tables do not shadow
    /// one another: no panel chord is also a global one.
    #[test]
    fn every_binding_resolves_to_itself() {
        let km = &*KEYS;
        for scope in Scope::ALL {
            for b in km.bindings(scope) {
                let hit = egui_hotkeys::resolve(km, Some(scope), b.key, Some(b.key), b.modifiers, false);
                assert_eq!(hit, Some((egui_hotkeys::Route::Scope(scope), b.command)), "{scope:?}: {}", b.command);
            }
        }
        for b in km.global() {
            for scope in Scope::ALL {
                let hit = egui_hotkeys::resolve(km, Some(scope), b.key, Some(b.key), b.modifiers, false);
                assert_eq!(hit, Some((egui_hotkeys::Route::Global, b.command)), "global {} over {scope:?}", b.command);
            }
        }
        // The first-listed key of a command is what its menu prints: it has a binding.
        for cmd in COMMANDS.iter().filter(|c| !c.keys.is_empty()) {
            let listed = |t: &[Binding<&'static str>]| t.iter().any(|b| b.command == cmd.id);
            assert!(listed(km.global()) || Scope::ALL.iter().any(|s| listed(km.bindings(*s))), "{} has no binding", cmd.id);
        }
    }

    /// Shift+Z hides the panels and does NOT also go fullscreen; Z alone goes fullscreen. Exact
    /// chords, over every panel.
    #[test]
    fn exact_chords_over_any_panel() {
        for scope in Scope::ALL {
            let on = |key, mods| press_in(Some(scope), key, mods, false, true).actions;
            assert_eq!(on(Key::Z, Modifiers::SHIFT), [Action::HidePanels], "{scope:?}");
            assert_eq!(on(Key::Z, Modifiers::NONE), [Action::Fullscreen], "{scope:?}");
            assert_eq!(on(Key::Space, Modifiers::SHIFT), [Action::PlayReverse], "{scope:?}");
            assert_eq!(on(Key::Space, Modifiers::NONE), [Action::PlayPause], "{scope:?}");
            assert_eq!(on(Key::O, Modifiers::COMMAND), [Action::Open], "{scope:?}");
            assert_eq!(on(Key::Comma, Modifiers::COMMAND), [Action::Tab(DockTab::Settings)], "{scope:?}");
            assert_eq!(on(Key::PageDown, Modifiers::NONE), [Action::FileNext], "{scope:?}");
        }
        // Ctrl+Shift+Z is neither of them.
        assert_eq!(press(Key::Z, Modifiers::COMMAND.plus(Modifiers::SHIFT), false), []);
    }

    /// Zoom: Ctrl with `=`, `+` (typed with Shift on most layouts) and `-`.
    #[test]
    fn zoom_chords() {
        assert_eq!(press(Key::Equals, Modifiers::COMMAND, false), [Action::Nav(NavCmd::ZoomIn)]);
        assert_eq!(press(Key::Plus, Modifiers::COMMAND, false), [Action::Nav(NavCmd::ZoomIn)]);
        assert_eq!(press(Key::Plus, Modifiers::COMMAND.plus(Modifiers::SHIFT), false), [Action::Nav(NavCmd::ZoomIn)]);
        assert_eq!(press(Key::Minus, Modifiers::COMMAND, false), [Action::Nav(NavCmd::ZoomOut)]);
        assert_eq!(press(Key::Minus, Modifiers::NONE, false), []);
    }

    /// Fit, the channel isolates and the display transform are global: they work over every panel
    /// (and not while typing). 100% and zoom are the viewport's: over another panel the key is
    /// left for that panel's widgets.
    #[test]
    fn view_keys_global_or_viewport() {
        let fired = |scope, key| press_in(Some(scope), key, Modifiers::NONE, false, true);
        for scope in Scope::ALL {
            for key in [Key::F, Key::R, Key::G, Key::B, Key::A, Key::T] {
                assert_eq!(fired(scope, key).actions.len(), 1, "{key:?} over {scope:?}");
                assert_eq!(press_in(Some(scope), key, Modifiers::NONE, true, true), Pressed { actions: vec![], left: true }, "{key:?} while typing");
            }
            let home = fired(scope, Key::H);
            let zoom = press_in(Some(scope), Key::Equals, Modifiers::COMMAND, false, true);
            if scope == Scope::Viewport {
                assert_eq!((home.actions, zoom.actions), (vec![Action::Nav(NavCmd::Home)], vec![Action::Nav(NavCmd::ZoomIn)]));
            } else {
                assert_eq!((home, zoom), (Pressed { actions: vec![], left: true }, Pressed { actions: vec![], left: true }), "{scope:?}");
            }
        }
        assert_eq!(fired(Scope::Metadata, Key::F).actions, [Action::Nav(NavCmd::Fit)]);
        // `0` is the key position: 100% on the plain key only.
        assert_eq!(press(Key::Num0, Modifiers::NONE, false), [Action::Nav(NavCmd::Home)]);
        assert_eq!(press(Key::Num0, Modifiers::SHIFT, false), []);
        // A layout whose unshifted `0` key types another character (AZERTY "a grave") is still that key.
        let azerty = press_at(Some(Scope::Viewport), Key::Backtick, Some(Key::Num0), Modifiers::NONE, false, true);
        assert_eq!(azerty.actions, [Action::Nav(NavCmd::Home)]);
    }

    /// Arrows (and `,` `.` Home End) step frames over the viewport and the timeline; over the
    /// channel tree they are left to the widget.
    #[test]
    fn arrows_step_over_viewport_and_timeline_only() {
        for scope in [Scope::Viewport, Scope::Timeline] {
            assert_eq!(step(scope, Key::ArrowRight, true), Pressed { actions: vec![Action::StepNext], left: false });
            assert_eq!(step(scope, Key::ArrowLeft, true).actions, [Action::StepPrev]);
            assert_eq!(step(scope, Key::Period, true).actions, [Action::StepNext]);
            assert_eq!(step(scope, Key::Comma, true).actions, [Action::StepPrev]);
            assert_eq!(step(scope, Key::Home, true).actions, [Action::First]);
            assert_eq!(step(scope, Key::End, true).actions, [Action::Last]);
        }
        for scope in [Scope::Metadata, Scope::PixelInfo, Scope::Scopes, Scope::Effects, Scope::Settings] {
            for key in [Key::ArrowRight, Key::ArrowLeft, Key::Home, Key::End] {
                assert_eq!(step(scope, key, true), Pressed { actions: vec![], left: true }, "{key:?} over {scope:?}");
            }
        }
    }

    /// The actions fired by the AUTO-REPEAT of a key held down over the viewport: the press
    /// frame, then a frame whose only event is the repeat.
    fn held(key: Key, mods: Modifiers) -> Vec<Action> {
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let mut got = Vec::new();
        for (i, repeat) in [None, Some(false), Some(true)].into_iter().enumerate() {
            let events = match repeat {
                None => vec![egui::Event::PointerMoved(screen.center())],
                Some(repeat) => vec![egui::Event::Key { key, physical_key: None, pressed: true, repeat, modifiers: mods }],
            };
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let mut output = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    if i == 2 {
                        got = pressed(ui.ctx(), false, Avail { seq: true, file: true });
                    }
                    panel(ui, Scope::Viewport, |_| {});
                });
            });
            output.textures_delta.clear();
        }
        got
    }

    /// Holding an arrow, `,` / `.` or Ctrl+- keeps stepping / zooming (as the old key reads did);
    /// holding a toggle does not repeat it.
    #[test]
    fn held_steps_and_zoom_repeat_toggles_do_not() {
        assert_eq!(held(Key::ArrowRight, Modifiers::NONE), [Action::StepNext]);
        assert_eq!(held(Key::ArrowLeft, Modifiers::NONE), [Action::StepPrev]);
        assert_eq!(held(Key::Period, Modifiers::NONE), [Action::StepNext]);
        assert_eq!(held(Key::Comma, Modifiers::NONE), [Action::StepPrev]);
        assert_eq!(held(Key::Minus, Modifiers::COMMAND), [Action::Nav(NavCmd::ZoomOut)]);
        assert_eq!(held(Key::Equals, Modifiers::COMMAND), [Action::Nav(NavCmd::ZoomIn)]);
        for (key, mods) in [(Key::R, Modifiers::NONE), (Key::F, Modifiers::NONE), (Key::Z, Modifiers::NONE), (Key::Space, Modifiers::NONE), (Key::Home, Modifiers::NONE)] {
            assert_eq!(held(key, mods), [], "{key:?} must not repeat");
        }
    }

    /// Before any click the viewport owns the keyboard (the keymap's fallback).
    #[test]
    fn the_viewport_owns_the_keys_at_startup() {
        assert_eq!(press_in(None, Key::H, Modifiers::NONE, false, true).actions, [Action::Nav(NavCmd::Home)]);
        assert_eq!(press_in(None, Key::ArrowRight, Modifiers::NONE, false, true).actions, [Action::StepNext]);
    }

    /// Space plays over any panel, but while typing it is text.
    #[test]
    fn space_plays_everywhere_except_while_typing() {
        for scope in Scope::ALL {
            assert_eq!(press_in(Some(scope), Key::Space, Modifiers::NONE, false, true).actions, [Action::PlayPause], "{scope:?}");
            assert_eq!(
                press_in(Some(scope), Key::Space, Modifiers::NONE, true, true),
                Pressed { actions: vec![], left: true },
                "{scope:?} while typing"
            );
        }
    }

    /// While typing, letters are text; Ctrl chords still work.
    #[test]
    fn typing_blocks_bare_keys_only() {
        assert_eq!(press(Key::R, Modifiers::NONE, true), []);
        assert_eq!(press(Key::O, Modifiers::COMMAND, true), [Action::Open]);
        assert_eq!(press_in(Some(Scope::Metadata), Key::O, Modifiers::COMMAND, true, true).actions, [Action::Open]);
        assert_eq!(press(Key::R, Modifiers::NONE, false), [Action::Isolate(ChannelDisplay::R)]);
    }

    /// A key whose command cannot run is left to the widgets: without a sequence the
    /// arrows are not eaten (the channel tree folds with them); with one they step.
    #[test]
    fn keys_of_disabled_commands_are_not_consumed() {
        let arrow = |has_seq| press_in(Some(Scope::Viewport), Key::ArrowRight, Modifiers::NONE, false, has_seq);
        assert_eq!(arrow(false), Pressed { actions: vec![], left: true });
        assert_eq!(arrow(true), Pressed { actions: vec![Action::StepNext], left: false });
        // Global ones follow the same rule: no file open, no folder stepping.
        assert_eq!(press_in(Some(Scope::Viewport), Key::PageDown, Modifiers::NONE, false, false), Pressed { actions: vec![], left: true });
    }
}
