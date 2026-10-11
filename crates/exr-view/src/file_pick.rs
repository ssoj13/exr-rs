//! File > Open: the in-app picker (`egui-file-dialog`, as in Playa).
//!
//! Immediate mode: it never blocks the frame, so playback and tile production go on
//! while it is open, and directories are listed on a worker thread. Paths are not
//! canonicalized: on Windows that prefixes the verbatim `\\?\` form, which would
//! then land in recents and the window title.

use std::path::{Path, PathBuf};

use egui_file_dialog::{DialogState, FileDialog, Filter, OpeningMode, SaveExtension};

/// Name of the image filter.
const FILTER: &str = "Images (every format the viewer opens)";
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

/// The file dialogs of the viewer (one on screen at a time).
pub struct FilePick {
    dialog: FileDialog,
    /// What the open dialog is for.
    purpose: Pick,
    /// The "Auto-detect sequences" box of File > Open (the dialog's right panel): the
    /// `Config::open_seq` option, set from it before the dialog opens.
    pub seq: bool,
    /// The signal of an HDR PNG (Save view as HDR, the dialog's right panel); session only.
    pub png: crate::export::HdrPng,
}

impl FilePick {
    /// A closed picker listing the openable images; `seq` is the saved "Auto-detect sequences" option.
    pub fn new(seq: bool) -> Self {
        let dialog = FileDialog::new()
            .title("Open")
            .as_modal(true)
            .canonicalize_paths(false)
            .load_via_thread(true)
            .add_file_filter(FILTER, Filter::new(|p: &Path| exr_import::is_image(p)))
            .add_file_filter(OCIO, Filter::new(|p: &Path| {
                p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("ocio") || e.eq_ignore_ascii_case("ocioz"))
            }))
            .default_file_filter(FILTER);
        Self { dialog, purpose: Pick::Open, seq, png: crate::export::HdrPng::default() }
    }

    /// Open the picker in `dir` (the folder of the file shown last), or where the
    /// dialog itself starts when there is none.
    pub fn open(&mut self, dir: Option<PathBuf>) {
        let config = self.dialog.config_mut();
        match dir {
            Some(dir) => {
                config.opening_mode = OpeningMode::AlwaysInitialDir;
                config.initial_directory = dir;
            }
            None => config.opening_mode = OpeningMode::LastPickedDir,
        }
        // The dialog selects its default filter on every open.
        config.default_file_filter = Some(FILTER.to_owned());
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
