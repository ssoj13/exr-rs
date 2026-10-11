//! Viewer persistence: window geometry + display controls + the recent-files
//! list, stored as a small JSON config in the OS config directory.
//!
//! # Why
//! A viewer that forgets everything on quit is annoying: relaunching should
//! restore the window where it was, the exposure/gamma/channel the operator
//! left, and reopen the last file. A File→Recents menu needs a deduplicated,
//! bounded history. All of that is one tiny JSON file — no database, no daemon.
//!
//! # Where it is used
//! `exr_view::run` loads [`Config`] at startup (and uses `last_file` as the
//! effective initial path when launched with no argument). `App`/`AppState`
//! carry the live [`Config`]: [`AppState::open_path`] records each successful
//! open into it (`last_file` + `push_recent` + [`Config::save`]), the File menu
//! reads `recents`, and the viewer writes the live window geometry + controls
//! back via [`Config::save`] on Esc and on close.
//!
//! # Robustness
//! Loading is total: a missing or unreadable config returns [`Config::default`] (logged at
//! debug), a corrupt one too (logged at warn: a user's settings were lost), and one bad
//! FIELD resets that field alone ([`Config::parse`], warn); never an error or a panic. Saving is
//! best-effort: any failure is logged at warn and swallowed (a viewer must not
//! die because `%APPDATA%` is read-only). No `unwrap`/`expect`/`panic` here.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Maximum number of entries kept in the recent-files list. Older entries are
/// dropped once this many newer files have been opened.
pub const RECENTS_MAX: usize = 12;

/// Saved window geometry (outer position + inner size, physical pixels).
///
/// Stored so the next launch can place the window where the operator left it.
/// Physical pixels keep it independent of any per-monitor DPI change between
/// runs (winit reports outer position + inner size in physical pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowGeom {
    /// Outer-position x (physical px).
    pub x: i32,
    /// Outer-position y (physical px).
    pub y: i32,
    /// Inner-size width (physical px).
    pub w: u32,
    /// Inner-size height (physical px).
    pub h: u32,
}

/// The persisted viewer configuration.
///
/// Every field is `#[serde(default)]` so a config written by an older build
/// (with fewer fields) still loads — missing keys fall back to the type default
/// instead of failing the whole parse. `channel` is a `u8` (mapped to/from the
/// viewer's `ChannelDisplay` by the caller) so the on-disk format does not
/// depend on the enum's internal representation. `fps`/`loop_` are global
/// playback preferences applied to whatever sequence is opened next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Last window geometry (absent on first run).
    #[serde(default)]
    pub window: Option<WindowGeom>,
    /// Display exposure in stops (EV).
    #[serde(default)]
    pub exposure: f32,
    /// The exposure the toolbar's bypass button parked (0 EV <-> this value).
    #[serde(default)]
    pub exposure_hold: f32,
    /// Display gamma dial (1.0 = identity).
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    /// Channel-isolate selection, encoded as a small integer (see the caller's
    /// `ChannelDisplay` mapping).
    #[serde(default)]
    pub channel: u8,
    /// Preferred sequence playback rate.
    #[serde(default = "default_fps")]
    pub fps: f32,
    /// Preferred sequence loop toggle.
    #[serde(default)]
    pub loop_: bool,
    /// Path opened most recently (reopened on launch when no CLI arg is given).
    #[serde(default)]
    pub last_file: Option<PathBuf>,
    /// Whether `last_file` was opened as a SEQUENCE (vs a single image). Restored
    /// on launch so a sequence reopens as a sequence, not a lone frame.
    #[serde(default)]
    pub last_sequence: bool,
    /// "Auto-detect sequences" (Settings > UI, also File > Open's box): File > Open and recent
    /// files open a picked frame's whole `name.####.exr` set.
    #[serde(default = "default_open_seq")]
    pub open_seq: bool,
    /// Recently opened paths, most-recent first, deduplicated, capped at
    /// `RECENTS_MAX`.
    #[serde(default)]
    pub recents: Vec<PathBuf>,
    /// Where every file dialog last confirmed a pick, by request key (Open, Save, the OCIO
    /// config browser, OpenFX path parameters), and the Open filter.
    #[serde(default)]
    pub dialogs: crate::file_pick::DialogMemory,
    /// Status-bar section widths (draggable splitters). Empty = the bar fills it
    /// with the section defaults on first draw.
    #[serde(default)]
    pub status_bar_layout: egui_statusbar::StatusBarLayout,
    /// The status bar's splitters are locked (right-click menu): the widths stay as they are.
    #[serde(default)]
    pub status_bar_locked: bool,
    /// The viewport toolbar strip: rolled out or up, and its vertical offset.
    #[serde(default)]
    pub toolbar: egui_viewport_toolbar::ToolbarState,
    /// Opacity (0..=1) of the thin white vertical playhead line drawn over the
    /// IMAGE during sequence playback/scrub. `0` (default) = off.
    #[serde(default)]
    pub playhead_alpha: f32,
    /// Frame-cache preload strategy (which frames load first around the playhead).
    #[serde(default)]
    pub preload_strategy: scancache::Strategy,
    /// The dock layout at exit (`egui_dock_layout::to_blob`), restored at launch.
    #[serde(default)]
    pub dock: Option<String>,
    /// The user's named dock layouts (the layout bar in the menu row).
    #[serde(default)]
    pub layouts: egui_dock_layout::LayoutStore,
    /// Memory budgets (the Settings tab).
    #[serde(default)]
    pub budgets: Budgets,
    /// Tone operator (`transfer::tonemap::*`), 0 = none.
    #[serde(default)]
    pub tonemap: u32,
    /// What shows under the image's alpha (toolbar drop-up): the shared viewer backdrop,
    /// composited in the present pass so it is right on every output.
    #[serde(default)]
    pub backdrop: egui_display::Backdrop,
    /// The OCIO display transform choice (the Colour panel).
    #[serde(default)]
    pub ocio: crate::ocio::Sel,
    /// The window output (the Settings tab's Display).
    #[serde(default)]
    pub display: DisplayPrefs,
    /// Text sizes and the dock tab height (the Settings tab's UI).
    #[serde(default)]
    pub fonts: Fonts,
}

/// Text sizes (pt) per egui text style, the dock tab-bar height (px) and the size of
/// the Settings category list: the "Fonts & UI" knobs of vfx-view, applied by `theme::set`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Fonts {
    pub heading: f32,
    pub body: f32,
    pub monospace: f32,
    pub button: f32,
    pub small: f32,
    /// egui_dock `TabBarStyle::height`.
    pub dock_tab: f32,
    /// The Settings panel's category list (egui-prefs2 paints it with its own `FontId`).
    pub sidebar: f32,
}

impl Fonts {
    /// Allowed text size.
    pub const SIZE: std::ops::RangeInclusive<f32> = 6.0..=32.0;
    /// Allowed dock tab-bar height.
    pub const TAB: std::ops::RangeInclusive<f32> = 16.0..=48.0;

    /// The same sizes held inside the allowed ranges (a hand-edited config could say anything,
    /// and a NaN size would break text layout).
    pub fn clamped(self) -> Self {
        let size = |v: f32, d: f32| if v.is_nan() { d } else { v.clamp(*Self::SIZE.start(), *Self::SIZE.end()) };
        let d = Self::default();
        Self {
            heading: size(self.heading, d.heading),
            body: size(self.body, d.body),
            monospace: size(self.monospace, d.monospace),
            button: size(self.button, d.button),
            small: size(self.small, d.small),
            dock_tab: if self.dock_tab.is_nan() { d.dock_tab } else { self.dock_tab.clamp(*Self::TAB.start(), *Self::TAB.end()) },
            sidebar: size(self.sidebar, d.sidebar),
        }
    }
}

impl Default for Fonts {
    /// The viewer's compact look: Small 9, every other style 10, egui_dock's 24 px tab bar.
    fn default() -> Self {
        Self { heading: 10.0, body: 10.0, monospace: 10.0, button: 10.0, small: 9.0, dock_tab: 24.0, sidebar: 10.0 }
    }
}

/// Shared application display preferences; preserves the viewer's config representation.
pub use egui_display::DisplayPrefs;

/// How much memory the viewer may take (the Settings tab). The shares are of the
/// machine's INSTALLED RAM / the adapter's VRAM, not of what is free at the moment
/// (sizing off free memory shrinks a cache's own allowance as it fills). The RAM
/// and VRAM caches are built at launch, so their shares apply at the next launch;
/// the upload limit applies at once.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Budgets {
    /// Share of installed RAM for the frame cache and the tile cache (each).
    pub ram: f64,
    /// Share of VRAM for the tile atlas.
    pub vram: f64,
    /// Tile bytes uploaded to the GPU per frame, in MiB (a bound on frame time).
    pub upload_mib: u32,
}

impl Budgets {
    /// Allowed RAM share.
    pub const RAM: std::ops::RangeInclusive<f64> = 0.25..=0.9;
    /// Allowed VRAM share.
    pub const VRAM: std::ops::RangeInclusive<f64> = 0.1..=0.9;
    /// Allowed upload MiB per frame.
    pub const UPLOAD: std::ops::RangeInclusive<u32> = 8..=1024;

    /// The same budgets held inside the allowed ranges (a hand-edited config
    /// could say anything).
    pub fn clamped(self) -> Self {
        Self {
            ram: self.ram.clamp(*Self::RAM.start(), *Self::RAM.end()),
            vram: self.vram.clamp(*Self::VRAM.start(), *Self::VRAM.end()),
            upload_mib: self.upload_mib.clamp(*Self::UPLOAD.start(), *Self::UPLOAD.end()),
        }
    }

    /// The upload limit in bytes.
    pub fn upload_bytes(&self) -> usize {
        (self.upload_mib as usize) << 20
    }
}

impl Default for Budgets {
    /// Default budgets: 75% RAM,
    /// 50% VRAM, 64 MiB of uploads per frame.
    fn default() -> Self {
        Self { ram: 0.75, vram: 0.5, upload_mib: 64 }
    }
}

/// Default gamma (identity) — used by serde when the key is absent so an old
/// config (or a partial hand-edit) does not silently force gamma to 0.
fn default_gamma() -> f32 {
    1.0
}

/// The default playback rate (also what the fps field resets to).
pub const DEFAULT_FPS: f32 = 24.0;

/// [`DEFAULT_FPS`] for serde when the key is absent, so an old config does not force a
/// 0-fps (stalled) playback.
fn default_fps() -> f32 {
    DEFAULT_FPS
}

/// File > Open opens sequences unless the user unticked it.
fn default_open_seq() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            window: None,
            exposure: 0.0,
            exposure_hold: 0.0,
            gamma: default_gamma(),
            channel: 0,
            fps: default_fps(),
            loop_: false,
            last_file: None,
            last_sequence: false,
            open_seq: default_open_seq(),
            recents: Vec::new(),
            dialogs: Default::default(),
            status_bar_layout: egui_statusbar::StatusBarLayout::default(),
            status_bar_locked: false,
            toolbar: egui_viewport_toolbar::ToolbarState::default(),
            playhead_alpha: 0.0,
            preload_strategy: scancache::Strategy::default(),
            dock: None,
            layouts: egui_dock_layout::LayoutStore::default(),
            budgets: Budgets::default(),
            tonemap: 0,
            backdrop: egui_display::Backdrop::default(),
            ocio: crate::ocio::Sel::default(),
            display: DisplayPrefs::default(),
            fonts: Fonts::default(),
        }
    }
}

/// The on-disk config path: `<config-dir>/exr-view/config.json`.
///
/// `<config-dir>` is the platform user-config directory (`%APPDATA%` on
/// Windows, `~/.config` on Linux, `~/Library/Application Support` on macOS).
/// Returns `None` only when the OS exposes no config directory at all — in
/// which case persistence is simply disabled (load → defaults, save → no-op).
///
/// Unit tests get a per-process file under the temp dir instead: they drive
/// `AppState`, whose opens record recents and save, and wrote the user's real
/// config before (a test sequence's temp path ended up in the user's recents).
pub fn config_path() -> Option<PathBuf> {
    if cfg!(test) {
        let dir = std::env::temp_dir().join(format!("exr-view-test-config-{}", std::process::id()));
        return Some(dir.join("config.json"));
    }
    Some(dirs::config_dir()?.join("exr-view").join("config.json"))
}

impl Config {
    /// Load the config from disk, falling back to [`Config::default`] on ANY
    /// problem (no config dir, missing file, unreadable, corrupt JSON).
    ///
    /// Never errors and never panics: a broken config must not stop the viewer from
    /// starting. The reason for a fallback is logged: at debug for a missing or unreadable
    /// file (normal on a first run), at warn for corrupt JSON (settings were lost); a single
    /// unreadable field is [`Config::parse`]'s, also a warn.
    pub fn load() -> Config {
        let Some(path) = config_path() else {
            log::debug!("persist: no config dir available; using defaults");
            return Config::default();
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                // Missing file on first run is normal — debug, not warn.
                log::debug!(
                    "persist: no config at {} ({e}); using defaults",
                    path.display()
                );
                return Config::default();
            }
        };
        match Config::parse(&text) {
            Ok(cfg) => {
                log::debug!("persist: loaded config from {}", path.display());
                cfg
            }
            Err(e) => {
                log::warn!(
                    "persist: corrupt config at {} ({e}); using defaults",
                    path.display()
                );
                Config::default()
            }
        }
    }

    /// Read a config; a field whose stored value no longer has its type's shape (a foreign
    /// crate changed a struct, or the file was hand-edited) falls back to its default with a
    /// warning, and every other field is kept: one bad field never resets the whole config.
    /// The one mechanism for every field, ours and foreign. Fails only for text that is not
    /// a JSON object at all. It relies on every field being keyed by its own name: a serde
    /// `alias` or `flatten` on `Config` would make a field's key ambiguous and break the slow path.
    pub fn parse(text: &str) -> serde_json::Result<Config> {
        use serde::de::Error as _;
        let stored: serde_json::Value = serde_json::from_str(text)?;
        // The usual case: everything reads.
        if let Ok(cfg) = serde_json::from_value::<Config>(stored.clone()) {
            return Ok(Self::migrated(cfg, &stored));
        }
        let serde_json::Value::Object(stored) = stored else {
            return Err(serde_json::Error::custom("the config is not a JSON object"));
        };
        let serde_json::Value::Object(mut kept) = serde_json::to_value(Config::default())? else {
            return Err(serde_json::Error::custom("a config does not serialize to an object"));
        };
        let stored_all = stored.clone();
        // Admit the stored fields one at a time, each on top of the defaults and what was admitted.
        for (key, value) in stored {
            let mut trial = kept.clone();
            trial.insert(key.clone(), value.clone());
            match serde_json::from_value::<Config>(serde_json::Value::Object(trial)) {
                Ok(_) => {
                    kept.insert(key, value);
                }
                Err(e) => log::warn!("config: `{key}` no longer reads ({e}); using its default"),
            }
        }
        let cfg = serde_json::from_value(serde_json::Value::Object(kept))?;
        Ok(Self::migrated(cfg, &serde_json::Value::Object(stored_all)))
    }

    /// Fold the settings of older configs into today's fields: the Colour panel's own remembered
    /// folder (`ocio_dir.last_dir`) now lives in the dialog memory under the key `"ocio"`.
    fn migrated(mut cfg: Config, stored: &serde_json::Value) -> Config {
        if let Some(dir) = stored.pointer("/ocio_dir/last_dir").and_then(serde_json::Value::as_str) {
            cfg.dialogs.migrate_ocio(dir);
        }
        cfg
    }

    /// Write the config to disk as pretty JSON, creating the parent directory
    /// if needed. Best-effort: any failure is logged at warn and swallowed
    /// (the viewer keeps running). Never panics.
    pub fn save(&self) {
        let Some(path) = config_path() else {
            log::warn!("persist: no config dir available; not saving");
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            log::warn!(
                "persist: cannot create {} ({e}); not saving",
                parent.display()
            );
            return;
        }
        let json = match serde_json::to_string_pretty(self) {
            Ok(j) => j,
            Err(e) => {
                log::warn!("persist: cannot serialize config ({e}); not saving");
                return;
            }
        };
        match std::fs::write(&path, json) {
            Ok(()) => log::debug!("persist: saved config to {}", path.display()),
            Err(e) => log::warn!("persist: cannot write {} ({e}); not saving", path.display()),
        }
    }

    /// Record `path` as the most-recently-opened file.
    ///
    /// The path is canonicalized when possible (so the same file reached via
    /// different relative paths collapses to one entry); if canonicalization
    /// fails (e.g. the file was removed) the path is stored as-is. Any existing
    /// equal entry is removed first (no duplicates), the path is inserted at the
    /// front (most-recent), and the list is truncated to `RECENTS_MAX`.
    pub fn push_recent(&mut self, path: &Path) {
        let entry = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.recents.retain(|p| p != &entry);
        self.recents.insert(0, entry);
        self.recents.truncate(RECENTS_MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An output name this build does not know reads as the default output and
    /// keeps the rest of the config; a known one round-trips.
    #[test]
    fn unknown_output_keeps_the_config() {
        let json = r#"{"recents": ["a.exr"], "display": {"output": "Dolby", "white": 250.0}}"#;
        let c: Config = serde_json::from_str(json).unwrap();
        assert_eq!(c.display.output, crate::present::Output::Sdr8);
        assert_eq!((c.display.white, c.display.peak, c.recents.len()), (Some(250.0), None, 1));
        for output in crate::present::Output::ALL {
            let hdr = Config { display: DisplayPrefs { output, white: None, peak: Some(600.0) }, ..Config::default() };
            let back: Config = serde_json::from_str(&serde_json::to_string(&hdr).unwrap()).unwrap();
            assert_eq!(back.display, hdr.display);
        }
    }

    /// A config written before the "Auto-detect sequences" option opens sequences; an unticked box is kept.
    #[test]
    fn open_seq_defaults_on_and_round_trips() {
        let c: Config = serde_json::from_str(r#"{"recents": []}"#).unwrap();
        assert!(c.open_seq && Config::default().open_seq);
        let off = Config { open_seq: false, ..Config::default() };
        let back: Config = serde_json::from_str(&serde_json::to_string(&off).unwrap()).unwrap();
        assert!(!back.open_seq);
    }

    /// An old config keeps the compact look (Small 9, the rest 10, 24 px tabs); a NaN or
    /// out-of-range size is held to the allowed range or the default.
    #[test]
    fn fonts_default_and_clamp() {
        let c: Config = serde_json::from_str(r#"{"recents": []}"#).unwrap();
        let d = c.fonts;
        assert_eq!((d.heading, d.body, d.monospace, d.button, d.small, d.dock_tab, d.sidebar), (10.0, 10.0, 10.0, 10.0, 9.0, 24.0, 10.0));
        let c: Config = serde_json::from_str(r#"{"fonts": {"body": 1.0, "small": 100.0, "dock_tab": 2.0}}"#).unwrap();
        let f = Fonts { heading: f32::NAN, ..c.fonts }.clamped();
        assert_eq!((f.body, f.small, f.dock_tab, f.heading, f.button), (6.0, 32.0, 16.0, 10.0, 10.0));
    }

    /// A config written before the toolbar, the status bar lock and the exposure hold existed
    /// still loads with them at their defaults; they survive a round trip.
    #[test]
    fn toolbar_status_lock_and_hold_default_and_round_trip() {
        let mut c: Config = serde_json::from_str(r#"{"recents": []}"#).unwrap();
        assert!(c.toolbar.expanded && c.toolbar.vertical_offset == 0.0);
        assert!(!c.status_bar_locked && c.exposure_hold == 0.0);
        c.toolbar.expanded = false;
        c.toolbar.vertical_offset = 12.0;
        c.status_bar_locked = true;
        c.exposure_hold = 1.5;
        let back: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert!(!back.toolbar.expanded && back.toolbar.vertical_offset == 12.0 && back.status_bar_locked);
        assert_eq!(back.exposure_hold, 1.5);
    }

    /// A field whose stored shape changed (a foreign crate's type, or one of ours, or a scalar)
    /// resets THAT field only: everything else the operator saved stays. Text that is not an
    /// object at all is an error (the caller falls back to the defaults and warns).
    #[test]
    fn a_malformed_field_resets_only_itself() {
        let json = r#"{"exposure": 2.0, "recents": ["a.exr"], "gamma": 1.8, "fps": 30.0,
            "toolbar": {"expanded": "maybe", "vertical_offset": [1]},
            "status_bar_layout": 7, "layouts": "x", "preload_strategy": {"nope": 1},
            "display": [], "ocio_dir": 3,
            "fonts": "big", "budgets": "lots", "backdrop": 9, "ocio": 5, "window": "here", "channel": "red",
            "dock": 4, "last_file": {"a": 1}}"#;
        let c = Config::parse(json).unwrap();
        // What was fine is kept...
        assert_eq!((c.exposure, c.gamma, c.fps, c.recents.len()), (2.0, 1.8, 30.0, 1));
        // ...a foreign type, one of ours and scalars each fall back alone.
        assert_eq!(c.toolbar, egui_viewport_toolbar::ToolbarState::default());
        assert_eq!(c.status_bar_layout, egui_statusbar::StatusBarLayout::default());
        assert!(c.dialogs.dirs.is_empty(), "a malformed ocio_dir migrates nothing");
        assert_eq!(c.display, DisplayPrefs::default());
        assert_eq!(c.fonts, Fonts::default());
        assert_eq!(c.budgets, Budgets::default());
        assert_eq!(c.backdrop, egui_display::Backdrop::default());
        assert_eq!(c.ocio, crate::ocio::Sel::default());
        assert!(c.window.is_none() && c.dock.is_none() && c.last_file.is_none());
        assert_eq!(c.channel, 0);
        assert!(Config::parse("[1, 2]").is_err() && Config::parse("not json").is_err());
        assert_eq!(Config::parse(r#"{"gamma": "x"}"#).unwrap().gamma, 1.0, "a bad gamma is the default, not 0");
    }

    /// A config with EVERY field off its default (a new field left at its default fails
    /// `a_full_config_round_trips`'s guard until it is set here).
    fn full_config() -> Config {
        let mut layouts = egui_dock_layout::LayoutStore::default();
        layouts.save("Wide", "{}");
        Config {
            window: Some(WindowGeom { x: -3, y: 7, w: 1200, h: 800 }),
            exposure: 1.5,
            exposure_hold: -2.0,
            gamma: 2.2,
            channel: 3,
            fps: 30.0,
            loop_: true,
            last_file: Some(PathBuf::from("shots/a.0001.exr")),
            last_sequence: true,
            open_seq: false,
            recents: vec![PathBuf::from("a.exr"), PathBuf::from("b.exr")],
            dialogs: crate::file_pick::DialogMemory {
                dirs: [("open", "shots"), ("save", "out"), ("ocio", "C:/cfg")]
                    .into_iter()
                    .map(|(k, d)| (k.to_owned(), PathBuf::from(d)))
                    .collect(),
                filters: [("open".to_owned(), "PNG".to_owned())].into_iter().collect(),
            },
            status_bar_layout: egui_statusbar::StatusBarLayout::new(vec![111.0, 222.0]),
            status_bar_locked: true,
            toolbar: egui_viewport_toolbar::ToolbarState { expanded: false, vertical_offset: 40.0 },
            playhead_alpha: 0.5,
            preload_strategy: scancache::Strategy::Forward,
            dock: Some("blob".to_owned()),
            layouts,
            budgets: Budgets { ram: 0.5, vram: 0.25, upload_mib: 128 },
            tonemap: 2,
            backdrop: egui_display::Backdrop { kind: egui_display::BackdropKind::Color, color: [0.2, 0.4, 0.6, 1.0], ..Default::default() },
            ocio: crate::ocio::Sel { on: true, config: "x.ocio".into(), input: "i".into(), display: "d".into(), view: "v".into(), look: "l".into() },
            display: DisplayPrefs { output: crate::present::Output::Hlg, white: Some(250.0), peak: Some(1200.0) },
            fonts: Fonts { heading: 12.0, body: 13.0, monospace: 11.0, button: 12.5, small: 8.0, dock_tab: 30.0, sidebar: 14.0 },
        }
    }

    /// What `save` writes, `parse` reads back: every field, compared whole (`==`) and as JSON,
    /// and no field of the probe is left at its default.
    #[test]
    fn a_full_config_round_trips() {
        let full = full_config();
        let json = serde_json::to_value(&full).unwrap();
        let default = serde_json::to_value(Config::default()).unwrap();
        let (json_map, default_map) = (json.as_object().unwrap(), default.as_object().unwrap());
        for (key, value) in default_map {
            assert_ne!(json_map.get(key), Some(value), "`{key}` is at its default in the probe config");
        }
        let text = serde_json::to_string_pretty(&full).unwrap();
        let back = Config::parse(&text).unwrap();
        assert_eq!(back, full);
        assert_eq!(serde_json::to_value(&back).unwrap(), json);
    }

    /// The slow path (one field no longer reads): valid object fields next to the broken one
    /// survive whole, foreign (`layouts`, `preload_strategy`) and ours (`fonts`) alike.
    #[test]
    fn valid_fields_survive_next_to_a_broken_one() {
        let full = full_config();
        let mut json = serde_json::to_value(&full).unwrap();
        json["toolbar"] = serde_json::json!({"expanded": "maybe"});
        let back = Config::parse(&json.to_string()).unwrap();
        assert_eq!(back.toolbar, egui_viewport_toolbar::ToolbarState::default());
        assert_eq!(back.fonts, full.fonts);
        assert_eq!(back.layouts, full.layouts);
        assert_eq!(back.preload_strategy, full.preload_strategy);
        let expect = Config { toolbar: egui_viewport_toolbar::ToolbarState::default(), ..full };
        assert_eq!(back, expect, "nothing else changed");
    }

    /// The level rule: the user's value wins (clamped to its range), else the OS's
    /// (clamped too), else the output's default; white and peak independently.
    #[test]
    fn target_levels_user_then_os_then_default() {
        use crate::present::{HDR_REF_WHITE, HLG_REF_PEAK, Output, SCRGB_NITS};
        let os = |white: Option<f32>, peak: Option<f32>| wgpu::DisplayHdrInfo {
            luminance: Some(wgpu::DisplayLuminance { sdr_white_nits: white, max_nits: peak, ..Default::default() }),
            ..Default::default()
        };
        let none = wgpu::DisplayHdrInfo::default();
        let p = |white, peak| DisplayPrefs { output: Output::Hlg, white, peak };
        let t = p(None, None).target(Output::Hlg, &none);
        assert_eq!((t.hdr, t.white, t.peak), (true, HDR_REF_WHITE, HLG_REF_PEAK));
        assert_eq!(p(None, None).target(Output::Scrgb, &none).white, SCRGB_NITS);
        assert!(!p(None, None).target(Output::Sdr8, &none).hdr);
        let t = p(None, None).target(Output::Hlg, &os(Some(240.0), Some(600.0)));
        assert_eq!((t.white, t.peak), (240.0, 600.0));
        let t = p(Some(300.0), Some(1500.0)).target(Output::Hlg, &os(Some(240.0), Some(600.0)));
        assert_eq!((t.white, t.peak), (300.0, 1500.0));
        let t = p(Some(9000.0), Some(10.0)).target(Output::Hlg, &os(Some(f32::NAN), Some(-1.0)));
        assert_eq!((t.white, t.peak), (1000.0, 400.0));
        let t = p(None, None).target(Output::Hlg, &os(Some(f32::NAN), Some(10_000.0)));
        assert_eq!((t.white, t.peak), (HDR_REF_WHITE, 2000.0));
    }

    /// The backdrop defaults to the checker (a config from before it existed shows one), and a
    /// chosen kind and its parameters survive a save.
    #[test]
    fn backdrop_defaults_to_the_checker_and_persists() {
        let c = Config::parse("{}").unwrap();
        assert_eq!(c.backdrop.kind, egui_display::BackdropKind::Checker);
        let mut c = Config::default();
        c.backdrop.kind = egui_display::BackdropKind::Gray50;
        c.backdrop.checker.size_px = 40.0;
        let back = Config::parse(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back.backdrop, c.backdrop);
    }

    /// A test run never reads or writes the user's own config: its path is a
    /// per-process temp file, outside the user config dir.
    #[test]
    fn tests_never_touch_the_user_config() {
        let p = config_path().unwrap();
        assert!(p.starts_with(std::env::temp_dir()), "{}", p.display());
        if let Some(user) = dirs::config_dir() {
            assert!(!p.starts_with(user.join("exr-view")), "{}", p.display());
        }
    }

    /// Round-trip + dedup: pushing the SAME path twice keeps a single entry at
    /// the front, and serialize → deserialize reproduces the config exactly.
    #[test]
    fn round_trip_and_dedup() {
        let mut cfg = Config {
            exposure: 1.5,
            gamma: 2.2,
            channel: 3,
            fps: 30.0,
            loop_: true,
            window: Some(WindowGeom {
                x: 10,
                y: 20,
                w: 1280,
                h: 800,
            }),
            ..Config::default()
        };

        // canonicalize() fails for a non-existent path, so the as-is path is
        // stored verbatim — making the dedup assertion deterministic.
        let p = PathBuf::from("Z:/no/such/frame.0001.exr");
        cfg.push_recent(&p);
        cfg.push_recent(&p);
        assert_eq!(
            cfg.recents.len(),
            1,
            "duplicate push must not grow the list"
        );
        assert_eq!(cfg.recents[0], p, "the deduped entry stays at the front");

        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back, "config must survive a JSON round-trip unchanged");
    }

    /// Truncation + order: pushing 13 distinct paths keeps only the newest 12,
    /// most-recent first (the very first push is evicted).
    #[test]
    fn truncates_to_max_and_orders_newest_first() {
        let mut cfg = Config::default();
        let paths: Vec<PathBuf> = (0..RECENTS_MAX + 1)
            .map(|i| PathBuf::from(format!("Z:/seq/f.{i:04}.exr")))
            .collect();
        for p in &paths {
            cfg.push_recent(p);
        }
        assert_eq!(cfg.recents.len(), RECENTS_MAX, "list capped at RECENTS_MAX");
        // Newest push is first; the oldest (paths[0]) was evicted.
        assert_eq!(cfg.recents[0], paths[RECENTS_MAX], "newest is at the front");
        assert_eq!(
            cfg.recents[RECENTS_MAX - 1],
            paths[1],
            "oldest kept is paths[1]"
        );
        assert!(
            !cfg.recents.contains(&paths[0]),
            "the 13th-oldest was evicted"
        );
    }

    /// Budgets out of range are held in range; a config written before budgets
    /// existed (or naming only some of them) loads the defaults for the rest.
    #[test]
    fn budgets_clamp_and_default() {
        let wild = Budgets { ram: 3.0, vram: 0.0, upload_mib: 1 << 20 }.clamped();
        assert_eq!(wild, Budgets { ram: 0.9, vram: 0.1, upload_mib: 1024 });
        let old: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(old.budgets, Budgets::default());
        let partial: Config = serde_json::from_str(r#"{"budgets": {"ram": 0.5}}"#).unwrap();
        assert_eq!(partial.budgets, Budgets { ram: 0.5, ..Budgets::default() });
    }

    /// A config written before the dialog memory was one map: the Colour panel's `ocio_dir` and the
    /// per-purpose Open / Save fields load into the map under their keys.
    #[test]
    fn an_old_config_migrates_its_dialog_folders() {
        let c = Config::parse(
            r#"{"ocio_dir": {"last_dir": "C:/cfg"},
                "dialogs": {"open_dir": "shots", "open_filter": "PNG", "save_dir": "out"}}"#,
        )
        .unwrap();
        let dir = |k: &str| c.dialogs.dirs[k].clone();
        assert_eq!((dir("ocio"), dir("open"), dir("save")), ("C:/cfg".into(), "shots".into(), "out".into()));
        assert_eq!(c.dialogs.filters["open"], "PNG");
        // Through the field-by-field path (one other field is malformed) as well.
        let c = Config::parse(r#"{"gamma": "x", "ocio_dir": {"last_dir": "C:/cfg"}}"#).unwrap();
        assert_eq!(c.dialogs.dirs["ocio"], std::path::Path::new("C:/cfg"));
    }
}
