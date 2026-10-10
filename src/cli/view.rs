//! `view` / `v` subcommand — launch the GPU viewer (egui + wgpu) on an optional
//! file / directory / `name.####.exr` frame pattern.
//!
//! `--auto` turns on sequence auto-detection: the player scans the directory for a
//! `name.####.exr` sequence and plays it. Without it (the default), the named path
//! opens as a single image — `exr-rs view a.0001.exr` shows exactly that frame.
//!
//! Gated behind the `viewer` cargo feature (default-on). Without it the command
//! still parses but returns a clear error, so a `--no-default-features` build is
//! a pure inspector/converter with no GUI dependencies.

use std::path::PathBuf;

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs v                                  # start the viewer (also: exr-rs view)",
    "exr-rs view --help                         # viewer options",
    "exr-rs view shot.exr                      # one image",
    "exr-rs view renders/beauty.0001.exr --auto  # the whole numbered sequence, playable",
    "exr-rs view renders/                      # a folder",
    "exr-rs view shot.exr --screenshot out.png   # save the view as PNG, no window",
    "exr-rs view shot.exr --screenshot out.png --hdr   # HDR10 PQ PNG (16-bit, cICP), no window",
    "exr-rs view shot.exr --screenshot out.png --hdr --hlg   # HLG PNG (16-bit, cICP), no window",
    "exr-rs view shot.exr --screenshot out.exr --hdr   # EXR of the display light (half, nits = pixel x whiteLuminance)",
);

/// `view` arguments.
#[derive(clap::Args)]
pub struct Args {
    /// A file, a directory or a frame of a `name.####.exr` sequence.
    path: Option<PathBuf>,
    /// Optional compatibility flag for pathless launch; bare view/v already starts the viewer.
    #[arg(short = 'e', long = "empty", conflicts_with_all = ["path", "screenshot", "hdr"])]
    _empty: bool,
    /// Detect the `name.####.exr` sequence around the path and play it.
    #[arg(long, overrides_with = "no_auto")]
    auto: bool,
    /// Open the named path as one image (the default).
    #[arg(long)]
    no_auto: bool,
    /// Save the file as the viewer shows it (neutral look) to this PNG (with --hdr: .png or .exr) and exit, without a window.
    #[arg(long, value_name = "FILE", requires = "path")]
    screenshot: Option<PathBuf>,
    /// With --screenshot: save HDR instead. `.png` = HDR10 (16-bit PQ, BT.2020, cICP + cLLI),
    /// `.exr` = the display light (linear Rec.709, half, whiteLuminance 203 nits).
    #[arg(long, requires = "screenshot")]
    hdr: bool,
    /// With --hdr: a `.png` is HLG (16-bit, BT.2020, cICP, for the BT.2100 1000-nit reference display)
    /// instead of HDR10 PQ.
    #[arg(long, requires = "hdr")]
    hlg: bool,
}

/// `exr-rs view` — open the viewer (optionally on a path), or with `--screenshot` render one
/// file to a PNG. The viewer blocks until its window closes.
pub fn run(args: Args) -> CliResult {
    match (args.screenshot, args.path) {
        (Some(out), Some(input)) => screenshot(&input, &out, args.hdr, args.hlg),
        (_, path) => launch(path, args.auto),
    }
}

#[cfg(feature = "viewer")]
fn screenshot(input: &std::path::Path, out: &std::path::Path, hdr: bool, hlg: bool) -> CliResult {
    let png = if hlg { exr_view::HdrPng::Hlg } else { exr_view::HdrPng::Pq };
    let (w, h) = exr_view::screenshot(input, out, hdr.then_some(png))
        .map_err(|e| CliError::Other(format!("screenshot: {e:#}")))?;
    outln!("{}: {w}x{h}", out.display());
    Ok(())
}

#[cfg(not(feature = "viewer"))]
fn screenshot(_input: &std::path::Path, _out: &std::path::Path, _hdr: bool, _hlg: bool) -> CliResult {
    launch(None, false)
}

/// Launch the viewer (feature present): drive `exr_view::run`, mapping its
/// `anyhow::Error` into the CLI's error type.
#[cfg(feature = "viewer")]
fn launch(path: Option<PathBuf>, detect: bool) -> CliResult {
    exr_view::run(path, detect).map_err(|e| CliError::Other(format!("viewer: {e:#}")))
}

/// Feature absent: report it instead of silently doing nothing.
#[cfg(not(feature = "viewer"))]
fn launch(_path: Option<PathBuf>, _detect: bool) -> CliResult {
    Err(CliError::Other(
        "this build was compiled without the `viewer` feature \
         (it is on by default — rebuild without `--no-default-features`)"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "viewer"))]
    use super::*;
    use clap::Parser;

    #[test]
    fn bare_view_and_alias_open_the_existing_pathless_viewer() {
        for command in ["view", "v"] {
            let cli = super::super::Cli::try_parse_from(["exr-rs", command]).unwrap();
            let super::super::Command::View(args) = cli.command else { panic!("not view") };
            assert!(args.path.is_none());
            assert!(args.screenshot.is_none());
            assert!(!args._empty);
            assert!(!args.auto);
            #[cfg(not(feature = "viewer"))]
            assert!(matches!(run(args), Err(CliError::Other(message)) if message.contains("without the `viewer` feature")));
        }
    }

    #[test]
    fn empty_window_is_explicit_and_uses_the_existing_pathless_launch() {
        for command in ["view", "v"] {
            for flag in ["--empty", "-e"] {
                let cli = super::super::Cli::try_parse_from(["exr-rs", command, flag]).unwrap();
                let super::super::Command::View(args) = cli.command else { panic!("not view") };
                assert!(args._empty);
                assert!(args.path.is_none());
                assert!(args.screenshot.is_none());
                #[cfg(not(feature = "viewer"))]
                assert!(matches!(run(args), Err(CliError::Other(message)) if message.contains("without the `viewer` feature")));
            }
        }
        for words in [
            &["exr-rs", "view", "--empty", "shot.exr"][..],
            &["exr-rs", "view", "--empty", "--screenshot", "out.png"][..],
            &["exr-rs", "view", "--empty", "--hdr"][..],
        ] {
            assert!(super::super::Cli::try_parse_from(words).is_err(), "{words:?}");
        }
    }
}
