//! Every file dialog of the viewer: one in-app picker (`egui-file-dialog`, as in Playa) behind
//! one type, [`FilePick`], with one memory, [`DialogMemory`].
//!
//! Immediate mode: it never blocks the frame, so playback and tile production go on
//! while it is open, and directories are listed on a worker thread. Paths are not
//! canonicalized: on Windows that prefixes the verbatim `\\?\` form, which would
//! then land in recents and the window title.
//!
//! Who asks: File > Open, Save view (SDR / HDR), the OCIO config browser, and - through the thin
//! [`OfxPicker`] adapter - the browse buttons of OpenFX path parameters. Each request carries a
//! memory KEY (`"open"`, `"save"`, `"ocio"`, `"ofx:<kind>"`); the dialog starts in the folder
//! remembered under it and records the folder of a confirmed pick, a cancel changes nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use egui_file_dialog::{DialogState, FileDialog, Filter, OpeningMode, SaveExtension};
use serde::{Deserialize, Serialize};

/// Label of the filter that lists every format at once (slot 0 of the Open filters).
const ALL: &str = "All images";
/// Name of the OCIO config filter.
const OCIO: &str = "OCIO config (*.ocio, *.ocioz)";
/// Name of the PNG save extension.
const PNG: &str = "PNG image";
/// Name of the HDR PNG save extension (PQ or HLG: the dialog's right panel).
const PNG_HDR: &str = "HDR PNG (16-bit; signal at the right)";
/// Name of the display-light EXR save extension.
const EXR_LIGHT: &str = "EXR (display light)";
/// Memory keys of the viewer's own dialogs.
const OPEN: &str = "open";
const SAVE: &str = "save";
const OCIO_KEY: &str = "ocio";

/// What a pick is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// File > Open.
    Open,
    /// File > Save view as PNG.
    SaveView,
    /// File > Save view as HDR (EXR or HDR PNG, by the extension; the PNG signal is FilePick::png).
    SaveViewHdr,
    /// Colour > File…: an OCIO config.
    OcioConfig,
    /// The browse button of an OpenFX path parameter (answered through [`OfxPicker`]).
    Ofx,
}

/// Where the dialogs were last confirmed, persisted in `Config` (`#[serde(default)]`: an older
/// config has none): ONE map from a request's key to the folder of its last confirmed pick, and
/// one from a key to the filter last in force (only Open has a filter family choice).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "StoredMemory")]
pub struct DialogMemory {
    /// Key -> folder of the last file confirmed under it.
    pub dirs: BTreeMap<String, PathBuf>,
    /// Key -> filter last in force: `exr_import::Family::key` for Open; absent = [`ALL`].
    pub filters: BTreeMap<String, String>,
}

/// The stored shape of [`DialogMemory`]: the maps plus the fields of configs written before them
/// (`open_dir`, `open_filter`, `save_dir`), which migrate to the keys `"open"` / `"save"`.
#[derive(Default, Deserialize)]
#[serde(default)]
struct StoredMemory {
    dirs: BTreeMap<String, PathBuf>,
    filters: BTreeMap<String, String>,
    open_dir: Option<PathBuf>,
    open_filter: Option<String>,
    save_dir: Option<PathBuf>,
}

impl From<StoredMemory> for DialogMemory {
    fn from(s: StoredMemory) -> Self {
        let mut m = DialogMemory { dirs: s.dirs, filters: s.filters };
        for (key, old) in [(OPEN, s.open_dir), (SAVE, s.save_dir)] {
            if let Some(dir) = old {
                m.dirs.entry(key.to_owned()).or_insert(dir);
            }
        }
        if let Some(f) = s.open_filter {
            m.filters.entry(OPEN.to_owned()).or_insert(f);
        }
        m
    }
}

impl DialogMemory {
    /// The folder a dialog starts in: the one remembered under `key` while it still exists, else
    /// `fallback`. A vanished folder must not reach the dialog, which would silently open
    /// somewhere else.
    pub fn start(&self, key: &str, fallback: Option<PathBuf>) -> Option<PathBuf> {
        self.dirs.get(key).filter(|d| d.is_dir()).cloned().or(fallback)
    }

    /// Remember the folder of a file confirmed under `key`. Returns whether anything changed.
    pub fn record(&mut self, key: &str, picked: &Path) -> bool {
        let Some(dir) = picked.parent().map(Path::to_path_buf) else { return false };
        self.dirs.insert(key.to_owned(), dir.clone()) != Some(dir)
    }

    /// Remember the filter in force under `key` (`None` = the default, which clears it).
    pub fn record_filter(&mut self, key: &str, filter: Option<String>) -> bool {
        match filter {
            Some(f) => self.filters.insert(key.to_owned(), f.clone()) != Some(f),
            None => self.filters.remove(key).is_some(),
        }
    }

    /// Where the legacy `Config::ocio_dir` (`{"last_dir": "..."}`) pointed: migrated by
    /// `Config::parse` to the key `"ocio"` unless the map already has one.
    pub fn migrate_ocio(&mut self, last_dir: &str) {
        if !last_dir.is_empty() {
            self.dirs.entry(OCIO_KEY.to_owned()).or_insert_with(|| PathBuf::from(last_dir));
        }
    }
}

type Test = Box<dyn Fn(&Path) -> bool + Send + Sync>;

/// A filter test: the file's extension is one of `exts` (case-insensitive).
fn has_ext(exts: Vec<String>) -> Test {
    Box::new(move |p: &Path| {
        p.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.iter().any(|x| e.eq_ignore_ascii_case(x)))
    })
}

/// The Open dialog's file filters, in order: [`ALL`], then one per family. Each test records its
/// slot in `seen` when the dialog runs it over a listing.
fn open_filters(families: &[exr_import::Family], seen: &Arc<AtomicUsize>) -> Vec<(String, Test)> {
    let mut tests: Vec<(String, Test)> = vec![(ALL.to_owned(), Box::new(exr_import::is_image))];
    tests.extend(families.iter().map(|f| (f.label.clone(), has_ext(f.exts.clone()))));
    tests
        .into_iter()
        .enumerate()
        .map(|(slot, (label, test))| {
            let seen = Arc::clone(seen);
            let tracked: Test = Box::new(move |p| {
                seen.store(slot, Ordering::Relaxed);
                test(p)
            });
            (label, tracked)
        })
        .collect()
}

/// The persisted key of filter `slot` (0 = [`ALL`] reads as `None`).
fn key_of(families: &[exr_import::Family], slot: usize) -> Option<String> {
    slot.checked_sub(1).and_then(|i| families.get(i)).map(|f| f.key.clone())
}

/// The file dialogs of the viewer (one on screen at a time).
pub struct FilePick {
    dialog: FileDialog,
    /// The Open filters after [`ALL`], in dialog order (`exr_import::families`).
    families: Vec<exr_import::Family>,
    /// Slot of the Open filter whose test ran last: 0 = [`ALL`], `n` = `families[n - 1]`.
    /// `egui-file-dialog` 0.15 has no getter for the selected filter, but it runs only that
    /// filter over a listing, and "All files" is off for Open, so this is the filter in force
    /// when a file is confirmed (its folder was listed through it).
    seen: Arc<AtomicUsize>,
    /// What the open dialog is for.
    purpose: Pick,
    /// The memory key of the open request.
    key: String,
    /// An OpenFX request is on screen and not yet answered.
    ofx_live: bool,
    /// The OpenFX dialog was cancelled and the adapter is not told yet.
    ofx_cancelled: bool,
    /// The "Auto-detect sequences" box of File > Open (the dialog's right panel): the
    /// `Config::open_seq` option, set from it before the dialog opens.
    pub seq: bool,
    /// The signal of an HDR PNG (Save view as HDR, the dialog's right panel); session only.
    pub png: crate::export::HdrPng,
}

impl FilePick {
    /// A closed picker listing every openable format; `seq` is the saved "Auto-detect sequences"
    /// option.
    pub fn new(seq: bool) -> Self {
        Self {
            dialog: Self::blank("Open"),
            families: exr_import::families(),
            seen: Arc::new(AtomicUsize::new(0)),
            purpose: Pick::Open,
            key: OPEN.to_owned(),
            ofx_live: false,
            ofx_cancelled: false,
            seq,
            png: crate::export::HdrPng::default(),
        }
    }

    /// A fresh dialog: modal, listings on a worker, paths as the OS gives them. A request builds
    /// its own (filters, start folder), so nothing of the previous purpose leaks into it.
    fn blank(title: &str) -> FileDialog {
        FileDialog::new().title(title).as_modal(true).canonicalize_paths(false).load_via_thread(true)
    }

    /// Start `dialog` in `dir`, or where it decides when there is none.
    fn start_in(dialog: &mut FileDialog, dir: Option<PathBuf>) {
        let config = dialog.config_mut();
        match dir {
            Some(dir) => {
                config.opening_mode = OpeningMode::AlwaysInitialDir;
                config.initial_directory = dir;
            }
            None => config.opening_mode = OpeningMode::LastPickedDir,
        }
    }

    /// The Open filter in force, as its persisted key (`None` = [`ALL`]). Valid after an Open
    /// pick; see [`Self::seen`].
    pub fn filter_key(&self) -> Option<String> {
        key_of(&self.families, self.seen.load(Ordering::Relaxed))
    }

    /// Open the picker where `mem` last confirmed an Open (else in `fallback`), with the filter of
    /// the family remembered under the same key (unknown or none = [`ALL`]).
    pub fn open(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>) {
        let label = mem
            .filters
            .get(OPEN)
            .and_then(|k| self.families.iter().find(|f| &f.key == k))
            .map_or(ALL, |f| f.label.as_str())
            .to_owned();
        let mut dialog = Self::blank("Open");
        // No "All files": every Open filter is ours, so `seen` always names the one in force.
        for (name, test) in open_filters(&self.families, &self.seen) {
            dialog = dialog.add_file_filter(&name, Filter::new(test));
        }
        dialog.config_mut().default_file_filter = Some(label);
        dialog.config_mut().show_all_files_filter = false;
        Self::start_in(&mut dialog, mem.start(OPEN, fallback));
        self.seen.store(0, Ordering::Relaxed);
        self.begin(dialog, Pick::Open, OPEN);
        self.dialog.pick_file();
    }

    /// Pick an OCIO config: the dialog lists nothing but `.ocio` / `.ocioz`. It starts where the
    /// last config was picked from, else in `fallback` (the current config's folder).
    pub fn pick_config(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>) {
        let mut dialog = Self::blank("OCIO config")
            .add_file_filter(OCIO, Filter::new(has_ext(vec!["ocio".into(), "ocioz".into()])))
            .default_file_filter(OCIO);
        dialog.config_mut().show_all_files_filter = false;
        Self::start_in(&mut dialog, mem.start(OCIO_KEY, fallback));
        self.begin(dialog, Pick::OcioConfig, OCIO_KEY);
        self.dialog.pick_file();
    }

    /// Ask where to save the view as PNG, proposing `name`.
    pub fn save_view(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>, name: &str) {
        self.save(mem, fallback, name, Pick::SaveView, &[(PNG, "png")]);
    }

    /// Ask where to save the view as HDR: the extension picks the format (`export::save_hdr`),
    /// the right panel the signal of a PNG ([`Self::png`]): the dialog does not say which
    /// save extension was picked, and both PNG signals are `.png`.
    pub fn save_view_hdr(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>, name: &str) {
        self.save(mem, fallback, name, Pick::SaveViewHdr, &[(PNG_HDR, "png"), (EXR_LIGHT, "exr")]);
    }

    /// The save dialog offering only `exts` (name, extension), the first one the default:
    /// the SDR save must not offer a format it cannot write.
    fn save(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>, name: &str, purpose: Pick, exts: &[(&str, &str)]) {
        let mut dialog = Self::blank("Save");
        let config = dialog.config_mut();
        config.default_file_name = name.to_owned();
        config.save_extensions = exts
            .iter()
            .map(|(n, e)| SaveExtension { id: egui::Id::new(n), name: (*n).to_owned(), file_extension: (*e).to_owned() })
            .collect();
        config.default_save_extension = exts.first().map(|(n, _)| (*n).to_owned());
        Self::start_in(&mut dialog, mem.start(SAVE, fallback));
        self.begin(dialog, purpose, SAVE);
        self.dialog.save_file();
    }

    /// An OpenFX path parameter's dialog (`request`: file to open / save / folder, with its own
    /// filters, or every file when it has none - never the image filters). Starts in the folder
    /// the panel names (the current value's, the presets folder), else where `mem` last confirmed
    /// an OpenFX pick of this kind, else in `fallback`.
    #[cfg(feature = "ofx")]
    pub fn ofx(&mut self, mem: &DialogMemory, fallback: Option<PathBuf>, request: ofx_host_egui::PickRequest) {
        use ofx_host_egui::PickKind;
        let key = format!("ofx:{}", match request.kind {
            PickKind::Open => "open",
            PickKind::Save => "save",
            PickKind::Folder => "folder",
        });
        let mut dialog = Self::blank(&request.title);
        if let Some(name) = &request.name {
            dialog = dialog.default_file_name(name);
        }
        let first = request.filters.first().map(|(label, _)| label.clone());
        for (label, exts) in request.filters {
            dialog = dialog.add_file_filter(&label, Filter::new(has_ext(exts)));
        }
        dialog.config_mut().show_all_files_filter = first.is_none();
        dialog.config_mut().default_file_filter = first;
        let dir = request.dir.filter(|d| d.is_dir()).or_else(|| mem.start(&key, fallback));
        Self::start_in(&mut dialog, dir);
        self.begin(dialog, Pick::Ofx, &key);
        self.ofx_live = true;
        match request.kind {
            PickKind::Open => self.dialog.pick_file(),
            PickKind::Save => self.dialog.save_file(),
            PickKind::Folder => self.dialog.pick_directory(),
        }
    }

    /// Install the request's dialog.
    fn begin(&mut self, dialog: FileDialog, purpose: Pick, key: &str) {
        self.dialog = dialog;
        self.purpose = purpose;
        self.key = key.to_owned();
        self.ofx_live = false;
    }

    /// Whether the dialog is on screen (the viewer's bare-key shortcuts are off then).
    pub fn is_open(&self) -> bool {
        matches!(self.dialog.state(), DialogState::Open)
    }

    /// Whether the OpenFX dialog was cancelled since the last call (once).
    pub fn take_ofx_cancel(&mut self) -> bool {
        std::mem::take(&mut self.ofx_cancelled)
    }

    /// Draw the dialog and return the picked file with what it is for, once; a confirmed pick is
    /// recorded in `mem` first (its folder under the request's key, the Open filter). Call after
    /// the rest of the UI so the modal paints on top.
    pub fn poll(&mut self, ctx: &egui::Context, mem: &mut DialogMemory) -> Option<(Pick, PathBuf)> {
        if self.purpose == Pick::Open {
            let seq = &mut self.seq;
            self.dialog.update_with_right_panel_ui(ctx, &mut |ui, _| {
                ui.checkbox(seq, "Auto-detect sequences")
                    .on_hover_text("Open the picked frame's whole name.####.exr set (a lone frame opens as one image); also Settings > UI");
            });
        } else if self.purpose == Pick::SaveViewHdr {
            let png = &mut self.png;
            self.dialog.update_with_right_panel_ui(ctx, &mut |ui, _| {
                ui.label("PNG signal");
                for choice in crate::export::HdrPng::ALL {
                    ui.radio_value(png, choice, choice.encoding().label());
                }
                ui.weak("An .exr holds the display light either way.");
            });
        } else {
            self.dialog.update(ctx);
        }
        if self.ofx_live && matches!(self.dialog.state(), DialogState::Cancelled) {
            self.ofx_live = false;
            self.ofx_cancelled = true;
        }
        let picked = self.dialog.take_picked().map(|p| (self.purpose, p))?;
        self.ofx_live = false;
        mem.record(&self.key, &picked.1);
        if picked.0 == Pick::Open {
            mem.record_filter(&self.key, self.filter_key());
        }
        Some(picked)
    }
}

/// The OpenFX panel's `PathPicker` (`ofx_host_egui`): a mailbox between the parameter panel and
/// the viewer's one [`FilePick`]. The panel's `request` is parked here; the app turns it into
/// [`FilePick::ofx`] after the UI is built and posts the answer back for the panel's next `poll`.
/// There is no dialog of its own.
#[cfg(feature = "ofx")]
#[derive(Debug, Default)]
pub struct OfxPicker {
    asked: std::sync::Mutex<Option<ofx_host_egui::PickRequest>>,
    answer: std::sync::Mutex<Option<ofx_host_egui::PickOutcome>>,
}

#[cfg(feature = "ofx")]
impl OfxPicker {
    /// The dialog the panel asked for since the last call.
    pub fn take_request(&self) -> Option<ofx_host_egui::PickRequest> {
        self.asked.lock().unwrap_or_else(|p| p.into_inner()).take()
    }

    /// How the dialog ended, for the panel's next `poll`.
    pub fn answer(&self, outcome: ofx_host_egui::PickOutcome) {
        *self.answer.lock().unwrap_or_else(|p| p.into_inner()) = Some(outcome);
    }
}

#[cfg(feature = "ofx")]
impl ofx_host_egui::PathPicker for OfxPicker {
    fn request(&self, request: ofx_host_egui::PickRequest) {
        *self.asked.lock().unwrap_or_else(|p| p.into_inner()) = Some(request);
    }

    fn poll(&self, _: &egui::Context) -> Option<ofx_host_egui::PickOutcome> {
        self.answer.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter_names(p: &mut FilePick) -> Vec<String> {
        p.dialog.config_mut().file_filters.iter().map(|f| f.name.clone()).collect()
    }

    #[test]
    fn each_key_starts_in_its_own_remembered_folder_else_the_fallback() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let fallback = Some(PathBuf::from("fallback"));
        let mut m = DialogMemory::default();
        assert_eq!(m.start("open", fallback.clone()), fallback, "nothing remembered");
        m.record("open", &a.path().join("x.exr"));
        m.record("ocio", &b.path().join("c.ocio"));
        assert_eq!(m.start("open", fallback.clone()), Some(a.path().to_path_buf()));
        assert_eq!(m.start("ocio", fallback.clone()), Some(b.path().to_path_buf()));
        assert_eq!(m.start("save", fallback.clone()), fallback, "keys do not share a folder");
        let gone = a.path().join("gone");
        m.dirs.insert("open".into(), gone);
        assert_eq!(m.start("open", fallback.clone()), fallback, "a vanished folder falls back");
        assert_eq!(m.start("open", None), None);
    }

    #[test]
    fn a_confirmed_pick_is_recorded_under_its_key_only() {
        let mut m = DialogMemory::default();
        assert!(m.record("open", Path::new("shots/a.png")));
        assert!(!m.record("open", Path::new("shots/b.png")), "same folder: unchanged");
        assert!(m.record("ofx:folder", Path::new("out/x")));
        assert_eq!(m.dirs.len(), 2);
        assert_eq!(m.dirs["open"], Path::new("shots"));
        // The Open filter: a family is remembered, "All images" is a real choice that clears it.
        assert!(m.record_filter("open", Some("PNG".into())));
        assert!(!m.record_filter("open", Some("PNG".into())));
        assert!(m.record_filter("open", None));
        assert!(m.filters.is_empty());
        assert!(!m.record_filter("open", None));
    }

    /// A cancelled dialog yields no pick, so `poll` records nothing: the memory is touched only by
    /// a path (see `poll`), never by a request.
    #[test]
    fn a_request_alone_changes_no_memory() {
        let mem = DialogMemory::default();
        let mut p = FilePick::new(true);
        p.open(&mem, Some(PathBuf::from("x")));
        p.pick_config(&mem, None);
        p.save_view(&mem, None, "v.png");
        assert_eq!(mem, DialogMemory::default());
    }

    #[test]
    fn memory_round_trips_and_old_configs_migrate() {
        let mut m = DialogMemory::default();
        m.record("open", Path::new("d/a.exr"));
        m.record("ofx:open", Path::new("p/b.json"));
        m.record_filter("open", Some("DPX".into()));
        let back: DialogMemory = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        let none: DialogMemory = serde_json::from_str("{}").unwrap();
        assert_eq!(none, DialogMemory::default());
        // The shape before the maps: per-purpose fields become the keys "open" / "save".
        let old: DialogMemory =
            serde_json::from_str(r#"{"open_dir": "shots", "open_filter": "PNG", "save_dir": "out"}"#).unwrap();
        assert_eq!(old.dirs["open"], Path::new("shots"));
        assert_eq!(old.dirs["save"], Path::new("out"));
        assert_eq!(old.filters["open"], "PNG");
        let mut ocio = DialogMemory::default();
        ocio.migrate_ocio("cfg");
        assert_eq!(ocio.dirs["ocio"], Path::new("cfg"));
        ocio.migrate_ocio("other");
        assert_eq!(ocio.dirs["ocio"], Path::new("cfg"), "an existing entry wins");
        let cfg: crate::persist::Config = serde_json::from_str(r#"{"recents": []}"#).unwrap();
        assert_eq!(cfg.dialogs, DialogMemory::default());
    }

    #[test]
    fn the_filter_in_force_is_the_one_whose_test_ran() {
        let families = exr_import::families();
        let seen = Arc::new(AtomicUsize::new(0));
        let tests = open_filters(&families, &seen);
        assert_eq!(tests.len(), families.len() + 1, "all + families");
        assert_eq!(tests[0].0, ALL);
        let slot = families.iter().position(|f| f.key == "PNG").unwrap() + 1;
        assert!((tests[slot].1)(Path::new("a.PNG")) && !(tests[slot].1)(Path::new("a.dpx")));
        assert_eq!(key_of(&families, seen.load(Ordering::Relaxed)).as_deref(), Some("PNG"));
        assert!((tests[0].1)(Path::new("a.dpx")));
        assert_eq!(key_of(&families, seen.load(Ordering::Relaxed)), None, "ALL reads as no family");
    }

    #[test]
    fn open_restores_the_saved_filter_or_all_and_lists_only_images() {
        let mut p = FilePick::new(true);
        let mut mem = DialogMemory::default();
        let label = |p: &mut FilePick| p.dialog.config_mut().default_file_filter.clone();
        mem.record_filter("open", Some("DPX".into()));
        p.open(&mem, None);
        assert!(label(&mut p).unwrap().starts_with("DPX ("));
        assert!(!p.dialog.config_mut().show_all_files_filter);
        assert!(!filter_names(&mut p).iter().any(|n| n == OCIO), "no OCIO slot in the image dialog");
        mem.filters.insert("open".into(), "NoSuchFormat".into());
        p.open(&mem, None);
        assert_eq!(label(&mut p).as_deref(), Some(ALL), "an unknown key reads as All images");
        p.open(&DialogMemory::default(), None);
        assert_eq!(label(&mut p).as_deref(), Some(ALL));
    }

    #[test]
    fn the_ocio_dialog_shows_only_ocio_configs_and_starts_in_its_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut mem = DialogMemory::default();
        mem.record("ocio", &dir.path().join("c.ocio"));
        let mut p = FilePick::new(true);
        p.pick_config(&mem, Some(PathBuf::from("elsewhere")));
        assert_eq!(filter_names(&mut p), [OCIO]);
        assert_eq!(p.dialog.config_mut().default_file_filter.as_deref(), Some(OCIO));
        assert!(!p.dialog.config_mut().show_all_files_filter);
        assert_eq!(p.dialog.config_mut().initial_directory, dir.path());
        assert_eq!((p.purpose, p.key.as_str()), (Pick::OcioConfig, "ocio"));
    }

    /// An OpenFX path parameter's dialog carries its own filters (never the image ones), or every
    /// file; it starts in the folder the panel names, else in its kind's remembered folder.
    #[cfg(feature = "ofx")]
    #[test]
    fn an_ofx_pick_uses_the_requests_filters_and_its_own_memory() {
        use ofx_host_egui::{PickKind, PickRequest};
        let (named, kept) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let mut mem = DialogMemory::default();
        mem.record("ofx:save", &kept.path().join("p.json"));
        let request = |kind, dir: Option<PathBuf>, filters: Vec<(String, Vec<String>)>| PickRequest {
            kind,
            title: "Preset".into(),
            dir,
            name: Some("p.json".into()),
            filters,
        };
        let mut p = FilePick::new(true);
        let json = vec![("JSON".to_owned(), vec!["json".to_owned()])];
        p.ofx(&mem, None, request(PickKind::Save, Some(named.path().to_path_buf()), json.clone()));
        assert_eq!(filter_names(&mut p), ["JSON"]);
        assert!(!p.dialog.config_mut().show_all_files_filter);
        assert_eq!(p.dialog.config_mut().initial_directory, named.path(), "the panel's folder first");
        assert_eq!((p.purpose, p.key.as_str()), (Pick::Ofx, "ofx:save"));
        p.ofx(&mem, None, request(PickKind::Save, None, json));
        assert_eq!(p.dialog.config_mut().initial_directory, kept.path(), "else the remembered one");
        p.ofx(&mem, Some(PathBuf::from("img")), request(PickKind::Open, None, vec![]));
        assert!(filter_names(&mut p).is_empty() && p.dialog.config_mut().show_all_files_filter, "no filters = every file");
        assert_eq!(p.key, "ofx:open", "files to open and to save remember apart");
        assert_eq!(p.dialog.config_mut().initial_directory, PathBuf::from("img"), "else the caller's fallback");
    }
}
