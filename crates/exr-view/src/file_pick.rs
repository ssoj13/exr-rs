//! File > Open: the in-app picker (`egui-file-dialog`, as in Playa).
//!
//! Immediate mode: it never blocks the frame, so playback and tile production go on
//! while it is open, and directories are listed on a worker thread. Paths are not
//! canonicalized: on Windows that prefixes the verbatim `\\?\` form, which would
//! then land in recents and the window title.

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
}

/// Where the dialogs were last confirmed, persisted in `Config` (`#[serde(default)]`: an older
/// config has none). Open and Save remember their own folder; only Open has a filter.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DialogMemory {
    /// Folder of the last file confirmed in File > Open.
    pub open_dir: Option<PathBuf>,
    /// `exr_import::Family::key` of the Open filter last in force; `None` = [`ALL`].
    pub open_filter: Option<String>,
    /// Folder of the last file confirmed in either Save view dialog.
    pub save_dir: Option<PathBuf>,
}

impl DialogMemory {
    /// The folder a dialog starts in: the remembered one while it still exists, else `fallback`
    /// (the folder of the image on screen). A vanished folder must not reach the dialog, which
    /// would silently open somewhere else.
    pub fn start(saved: Option<&Path>, fallback: Option<PathBuf>) -> Option<PathBuf> {
        saved.filter(|d| d.is_dir()).map(Path::to_path_buf).or(fallback)
    }

    /// Remember a confirmed pick (`None`, a cancelled dialog or no result this frame, changes
    /// nothing); `filter` is the Open filter in force. Returns whether anything changed.
    pub fn record(&mut self, picked: Option<&(Pick, PathBuf)>, filter: Option<String>) -> bool {
        let Some((pick, path)) = picked else { return false };
        let dir = path.parent().map(Path::to_path_buf);
        let before = self.clone();
        match pick {
            Pick::Open => {
                self.open_dir = dir;
                self.open_filter = filter;
            }
            Pick::SaveView | Pick::SaveViewHdr => self.save_dir = dir,
            Pick::OcioConfig => {}
        }
        *self != before
    }
}

type Test = Box<dyn Fn(&Path) -> bool + Send + Sync>;

/// The dialog's file filters, in order: [`ALL`], one per family, then the OCIO config filter
/// (past the families, so `seen` never goes stale on it). Each test records its slot in `seen`
/// when the dialog runs it over a listing.
fn open_filters(families: &[exr_import::Family], seen: &Arc<AtomicUsize>) -> Vec<(String, Test)> {
    let has_ext = |exts: Vec<String>| {
        move |p: &Path| p.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.iter().any(|x| e.eq_ignore_ascii_case(x)))
    };
    let mut tests: Vec<(String, Test)> = vec![(ALL.to_owned(), Box::new(exr_import::is_image))];
    tests.extend(families.iter().map(|f| (f.label.clone(), Box::new(has_ext(f.exts.clone())) as Test)));
    tests.push((OCIO.to_owned(), Box::new(has_ext(vec!["ocio".into(), "ocioz".into()]))));
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

/// The persisted key of filter `slot` (0 = [`ALL`] and the OCIO slot read as `None`).
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
        let families = exr_import::families();
        let seen = Arc::new(AtomicUsize::new(0));
        let mut dialog = FileDialog::new().title("Open").as_modal(true).canonicalize_paths(false).load_via_thread(true);
        for (label, test) in open_filters(&families, &seen) {
            dialog = dialog.add_file_filter(&label, Filter::new(test));
        }
        let dialog = dialog.default_file_filter(ALL);
        Self { dialog, families, seen, purpose: Pick::Open, seq, png: crate::export::HdrPng::default() }
    }

    /// The Open filter in force, as its persisted key (`None` = [`ALL`]). Valid after an Open
    /// pick; see [`Self::seen`].
    pub fn filter_key(&self) -> Option<String> {
        key_of(&self.families, self.seen.load(Ordering::Relaxed))
    }
    /// Open the picker in `dir`, or where the dialog itself starts when there is none, with the
    /// filter of family `filter` (a persisted key; unknown or `None` = [`ALL`]).
    pub fn open(&mut self, dir: Option<PathBuf>, filter: Option<&str>) {
        let label = filter
            .and_then(|k| self.families.iter().find(|f| f.key == k))
            .map_or(ALL, |f| f.label.as_str())
            .to_owned();
        let config = self.dialog.config_mut();
        match dir {
            Some(dir) => {
                config.opening_mode = OpeningMode::AlwaysInitialDir;
                config.initial_directory = dir;
            }
            None => config.opening_mode = OpeningMode::LastPickedDir,
        }
        // The dialog selects its default filter on every open. No "All files": every Open filter
        // is ours, so `seen` always names the one in force.
        config.default_file_filter = Some(label);
        config.show_all_files_filter = false;
        self.seen.store(0, Ordering::Relaxed);
        self.purpose = Pick::Open;
        self.dialog.pick_file();
    }

    /// Pick an OCIO config, starting in `dir` (the current config's folder), else
    /// where the dialog last picked.
    pub fn pick_config(&mut self, dir: Option<PathBuf>) {
        let config = self.dialog.config_mut();
        match dir {
            Some(dir) => {
                config.opening_mode = OpeningMode::AlwaysInitialDir;
                config.initial_directory = dir;
            }
            None => config.opening_mode = OpeningMode::LastPickedDir,
        }
        config.default_file_filter = Some(OCIO.to_owned());
        config.show_all_files_filter = true;
        self.purpose = Pick::OcioConfig;
        self.dialog.pick_file();
    }

    /// Ask where to save the view as PNG, in `dir`, proposing `name`.
    pub fn save_view(&mut self, dir: Option<PathBuf>, name: &str) {
        self.save(dir, name, Pick::SaveView, &[(PNG, "png")]);
    }

    /// Ask where to save the view as HDR: the extension picks the format (`export::save_hdr`),
    /// the right panel the signal of a PNG ([`Self::png`]): the dialog does not say which
    /// save extension was picked, and both PNG signals are `.png`.
    pub fn save_view_hdr(&mut self, dir: Option<PathBuf>, name: &str) {
        self.save(dir, name, Pick::SaveViewHdr, &[(PNG_HDR, "png"), (EXR_LIGHT, "exr")]);
    }

    /// The save dialog offering only `exts` (name, extension), the first one the default:
    /// the SDR save must not offer a format it cannot write.
    fn save(&mut self, dir: Option<PathBuf>, name: &str, purpose: Pick, exts: &[(&str, &str)]) {
        let config = self.dialog.config_mut();
        if let Some(dir) = dir {
            config.opening_mode = OpeningMode::AlwaysInitialDir;
            config.initial_directory = dir;
        }
        config.default_file_name = name.to_owned();
        config.save_extensions = exts
            .iter()
            .map(|(n, e)| SaveExtension { id: egui::Id::new(n), name: (*n).to_owned(), file_extension: (*e).to_owned() })
            .collect();
        config.default_save_extension = exts.first().map(|(n, _)| (*n).to_owned());
        self.purpose = purpose;
        self.dialog.save_file();
    }

    /// Whether the dialog is on screen (the viewer's bare-key shortcuts are off then).
    pub fn is_open(&self) -> bool {
        matches!(self.dialog.state(), DialogState::Open)
    }

    /// Draw the dialog and return the picked file with what it is for, once. Call
    /// after the rest of the UI so the modal paints on top.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<(Pick, PathBuf)> {
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
        self.dialog.take_picked().map(|p| (self.purpose, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_pick(p: &str) -> Option<(Pick, PathBuf)> {
        Some((Pick::Open, PathBuf::from(p)))
    }

    #[test]
    fn start_prefers_a_saved_folder_that_exists() {
        let dir = tempfile::tempdir().unwrap();
        let fallback = Some(PathBuf::from("fallback"));
        assert_eq!(DialogMemory::start(Some(dir.path()), fallback.clone()), Some(dir.path().to_path_buf()));
        let gone = dir.path().join("gone");
        assert_eq!(DialogMemory::start(Some(&gone), fallback.clone()), fallback, "a vanished folder falls back");
        assert_eq!(DialogMemory::start(None, fallback.clone()), fallback);
        assert_eq!(DialogMemory::start(None, None), None);
    }

    #[test]
    fn a_confirmed_pick_is_remembered_a_cancel_is_not() {
        let mut m = DialogMemory::default();
        assert!(!m.record(None, Some("PNG".into())), "cancel / nothing picked");
        assert_eq!(m, DialogMemory::default());

        assert!(m.record(open_pick("shots/a.png").as_ref(), Some("PNG".into())));
        assert_eq!((m.open_dir.as_deref(), m.open_filter.as_deref()), (Some(Path::new("shots")), Some("PNG")));
        assert!(!m.record(open_pick("shots/b.png").as_ref(), Some("PNG".into())), "unchanged");

        // A save has its own folder and leaves the Open memory alone; so does an OCIO pick.
        assert!(m.record(Some(&(Pick::SaveView, PathBuf::from("out/v.png"))), None));
        assert!(m.record(Some(&(Pick::SaveViewHdr, PathBuf::from("hdr/v.exr"))), None));
        assert!(!m.record(Some(&(Pick::OcioConfig, PathBuf::from("cfg/c.ocio"))), None));
        assert_eq!((m.open_dir.as_deref(), m.save_dir.as_deref()), (Some(Path::new("shots")), Some(Path::new("hdr"))));
        assert_eq!(m.open_filter.as_deref(), Some("PNG"));

        // "All images" is a real choice: it clears the remembered family.
        assert!(m.record(open_pick("shots/c.exr").as_ref(), None));
        assert_eq!(m.open_filter, None);
    }

    #[test]
    fn memory_round_trips_and_old_configs_have_none() {
        let m = DialogMemory { open_dir: Some("d".into()), open_filter: Some("DPX".into()), save_dir: Some("s".into()) };
        let back: DialogMemory = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        let old: DialogMemory = serde_json::from_str("{}").unwrap();
        assert_eq!(old, DialogMemory::default());
        let cfg: crate::persist::Config = serde_json::from_str(r#"{"recents": []}"#).unwrap();
        assert_eq!(cfg.dialogs, DialogMemory::default());
    }

    #[test]
    fn the_filter_in_force_is_the_one_whose_test_ran() {
        let families = exr_import::families();
        let seen = Arc::new(AtomicUsize::new(0));
        let tests = open_filters(&families, &seen);
        assert_eq!(tests.len(), families.len() + 2, "all + families + ocio");
        assert_eq!(tests[0].0, ALL);
        let slot = families.iter().position(|f| f.key == "PNG").unwrap() + 1;
        assert!((tests[slot].1)(Path::new("a.PNG")) && !(tests[slot].1)(Path::new("a.dpx")));
        assert_eq!(key_of(&families, seen.load(Ordering::Relaxed)).as_deref(), Some("PNG"));
        assert!((tests[0].1)(Path::new("a.dpx")));
        assert_eq!(key_of(&families, seen.load(Ordering::Relaxed)), None, "ALL reads as no family");
        assert!((tests[families.len() + 1].1)(Path::new("c.ocio")));
        assert_eq!(key_of(&families, seen.load(Ordering::Relaxed)), None, "the OCIO slot reads as none");
    }

    #[test]
    fn open_restores_the_saved_filter_or_all() {
        let mut p = FilePick::new(true);
        let label = |p: &mut FilePick| p.dialog.config_mut().default_file_filter.clone();
        p.open(None, Some("DPX"));
        assert!(label(&mut p).unwrap().starts_with("DPX ("));
        assert!(!p.dialog.config_mut().show_all_files_filter);
        p.open(None, Some("NoSuchFormat"));
        assert_eq!(label(&mut p).as_deref(), Some(ALL), "an unknown key reads as All images");
        p.open(None, None);
        assert_eq!(label(&mut p).as_deref(), Some(ALL));
    }
}
