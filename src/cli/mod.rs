//! The `exr-rs` command line: one [`clap`] definition per subcommand, so the parser, the help
//! text and the errors come from the same declaration.
//!
//! Tools without arguments print help (`arg_required_else_help`), with usage, options and
//! common command lines ([`examples!`]). The viewer is the launch exception: bare `view` or
//! `v` starts the viewer without requiring a path. Every command has explicit help and a visible single-letter
//! alias. `tests::every_command_helps` checks the help contract throughout the tree.

#[macro_use]
pub mod stdio;

pub mod aces;
pub mod attr;
pub mod check;
pub mod convert;
pub mod cryptomatte;
pub mod envmap;
pub mod info;
pub mod manifest;
pub mod metrics;
pub mod multipart;
pub mod multiview;
pub mod preview;
pub mod tile;
mod tool_io;
pub mod view;

use clap::{Parser, Subcommand};

/// An `Examples:` help section from `"command line  # what it does"` literals, built at compile
/// time so a command's examples live next to its argument definition.
macro_rules! examples {
    ($($line:literal),+ $(,)?) => {
        concat!("Examples:\n", $("  ", $line, "\n"),+)
    };
}
pub(crate) use examples;

const EXAMPLES: &str = examples!(
    "exr-rs info shot.exr                       # every header attribute (like exrheader)",
    "exr-rs convert in.exr out.exr -c dwab      # re-encode with another codec",
    "exr-rs view shot.exr                       # open the GPU viewer",
    "exr-rs view shots/beauty.0001.exr --auto   # play the whole numbered sequence",
    "exr-rs cryptomatte info shot.exr           # list the Cryptomatte layers",
);

/// OpenEXR inspector, converter and viewer (a pure-Rust port of the OpenEXR tools).
#[derive(Parser)]
#[command(name = "exr-rs", bin_name = "exr-rs", version, arg_required_else_help = true, after_help = EXAMPLES)]
pub struct Cli {
    /// Raise the log level: -v debug, -vv trace (RUST_LOG overrides).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print every header attribute of every part (the output of OpenEXR's exrheader).
    #[command(visible_alias = "i", visible_alias = "header", alias = "exrheader", arg_required_else_help = true, after_help = info::EXAMPLES)]
    Info(info::Args),

    /// Re-encode a single-part image, optionally with another compression.
    #[command(visible_alias = "x", arg_required_else_help = true, after_help = convert::EXAMPLES)]
    Convert(convert::Args),

    /// Cryptomatte: list layers, extract mattes, encode from coverage images.
    #[command(visible_alias = "c", arg_required_else_help = true, after_help = cryptomatte::EXAMPLES)]
    Cryptomatte(cryptomatte::Args),

    /// Open the GPU viewer, optionally on a file, directory or numbered sequence.
    #[command(visible_alias = "v", after_help = view::EXAMPLES)]
    View(view::Args),

    /// Check that files open and read without crashes or runaway allocations (OpenEXR's exrcheck).
    #[command(visible_alias = "k", alias = "exrcheck", arg_required_else_help = true, after_help = check::EXAMPLES)]
    Check(check::Args),

    /// Make a tiled (optionally MIP/RIP-mapped) copy of an image (OpenEXR's exrmaketiled).
    #[command(visible_alias = "t", alias = "maketiled", alias = "exrmaketiled", arg_required_else_help = true, after_help = tile::EXAMPLES)]
    Tile(tile::Args),

    /// Add a preview image (thumbnail) to the header (OpenEXR's exrmakepreview).
    #[command(visible_alias = "p", alias = "makepreview", alias = "exrmakepreview", arg_required_else_help = true, after_help = preview::EXAMPLES)]
    Preview(preview::Args),

    /// Set standard or custom header attributes (OpenEXR's exrstdattr).
    #[command(visible_alias = "a", alias = "stdattr", alias = "exrstdattr", arg_required_else_help = true, after_help = attr::EXAMPLES)]
    Attr(attr::Args),

    /// Combine images into a multipart file, separate it, or convert between the two (OpenEXR's exrmultipart).
    #[command(visible_alias = "m", alias = "exrmultipart", arg_required_else_help = true, after_help = multipart::EXAMPLES)]
    Multipart(multipart::Args),

    /// Combine per-view images into one multi-view (stereo) image (OpenEXR's exrmultiview).
    #[command(visible_alias = "s", alias = "exrmultiview", arg_required_else_help = true, after_help = multiview::EXAMPLES)]
    Multiview(multiview::Args),

    /// Convert and filter environment maps (lat-long and cube) (OpenEXR's exrenvmap).
    #[command(visible_alias = "e", alias = "exrenvmap", arg_required_else_help = true, after_help = envmap::EXAMPLES)]
    Envmap(envmap::Args),

    /// Measure read/write time and size per codec (OpenEXR's exrmetrics).
    #[command(visible_alias = "b", alias = "exrmetrics", arg_required_else_help = true, after_help = metrics::EXAMPLES)]
    Metrics(metrics::Args),

    /// Print the ID manifest of files (OpenEXR's exrmanifest).
    #[command(visible_alias = "n", alias = "exrmanifest", arg_required_else_help = true, after_help = manifest::EXAMPLES)]
    Manifest(manifest::Args),

    /// Convert an image to an ACES image container file (OpenEXR's exr2aces).
    #[command(visible_alias = "A", alias = "2aces", alias = "exr2aces", arg_required_else_help = true, after_help = aces::EXAMPLES)]
    Aces(aces::Args),
}

/// Run the parsed command line and return the process exit code; the one place exit codes are
/// decided (`main` leaves with it once everything here has been dropped).
///
/// The tools that print progress take `verbose`: the global `-v` / `-vv`, or a debug log level
/// from `RUST_LOG`, which is what the C++ tools' own `-v` flag stands for.
pub fn run(cli: Cli) -> i32 {
    let verbose = cli.verbose > 0 || log::log_enabled!(log::Level::Debug);
    let result = match cli.command {
        Command::Info(a) => info::run(a),
        Command::Convert(a) => convert::run(a),
        Command::Cryptomatte(a) => cryptomatte::run(a),
        Command::View(a) => view::run(a),
        Command::Check(a) => check::run(a),
        Command::Tile(a) => tile::run(a, verbose),
        Command::Preview(a) => preview::run(a, verbose),
        Command::Attr(a) => attr::run(a),
        Command::Multipart(a) => multipart::run(a),
        Command::Multiview(a) => multiview::run(a, verbose),
        Command::Envmap(a) => envmap::run(a, verbose),
        Command::Metrics(a) => metrics::run(a, verbose),
        Command::Manifest(a) => manifest::run(a),
        Command::Aces(a) => aces::run(a, verbose),
    };
    let code = match result {
        Ok(()) => 0,
        Err(CliError::Exit(code)) => code,
        Err(err) => {
            errln!("error: {err}");
            1
        }
    };
    stdio::flush();
    code
}

/// A runtime error of a subcommand (argument errors are clap's, reported before `run`).
#[derive(Debug)]
pub enum CliError {
    /// An error returned by `exr-core`.
    Exr(exr_core::ExrError),
    /// An error from another subsystem (Cryptomatte, the viewer, file I/O).
    Other(String),
    /// Leave with this exit code, the message (if any) already printed: the C++ tools' `return -1`
    /// after a usage error. Turned into the process exit code by [`run`] alone.
    Exit(i32),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Exr(e) => write!(f, "{e}"),
            CliError::Other(m) => write!(f, "{m}"),
            CliError::Exit(code) => write!(f, "exit code {code}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<exr_core::ExrError> for CliError {
    fn from(e: exr_core::ExrError) -> Self {
        CliError::Exr(e)
    }
}

/// Subcommand result alias.
pub type CliResult = Result<(), CliError>;

/// Print the `--version` text of the OpenEXR tool `tool` (e.g. `exrcheck`), exactly as every
/// `src/bin/<tool>/main.cpp` of OpenEXR v3.4.15 prints it. The C++ adds
/// `(OpenEXR version <lib>)` only when the header and library versions differ; ours are one crate.
pub fn print_version(tool: &str) {
    out!(
        "{tool} (OpenEXR) {v} https://openexr.com\n\
         Copyright (c) Contributors to the OpenEXR Project\n\
         License BSD-3-Clause\n",
        v = exr_core::LIBRARY_VERSION
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// clap's own consistency checks over the whole tree (duplicate names, bad defaults ...).
    #[test]
    fn the_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    /// Tools without arguments show help; the viewer accepts a pathless launch
    /// and shows help only when explicitly requested.
    #[test]
    fn every_command_helps() {
        fn walk(cmd: &clap::Command, path: &str, seen: &mut Vec<String>) {
            let path = format!("{path} {}", cmd.get_name());
            let help = cmd.get_after_help().map(|h| h.to_string()).unwrap_or_default();
            assert!(help.starts_with("Examples:\n"), "{path}: no Examples section");
            assert!(help.lines().count() >= 3, "{path}: fewer than two examples");
            if cmd.get_name() == "view" {
                for flag in ["--help", "-h"] {
                    let err = cmd.clone().try_get_matches_from([cmd.get_name().to_owned(), flag.to_owned()]).unwrap_err();
                    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp, "{path}: {flag}");
                }
                assert!(!cmd.is_arg_required_else_help_set(), "{path}: no arguments must launch");
                assert!(cmd.clone().try_get_matches_from([cmd.get_name().to_owned()]).is_ok(), "{path}");
            } else {
                assert!(cmd.is_arg_required_else_help_set(), "{path}: no arguments must print help");
                let err = cmd.clone().try_get_matches_from([cmd.get_name().to_owned()]).unwrap_err();
                assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand, "{path}");
            }
            for sub in cmd.get_subcommands() {
                assert!(sub.get_visible_aliases().any(|a| a.chars().count() == 1),
                    "{path} {}: missing visible single-letter alias", sub.get_name());
            }
            seen.push(path.clone());
            for sub in cmd.get_subcommands() {
                walk(sub, &path, seen);
            }
        }
        let mut seen = Vec::new();
        walk(&Cli::command(), "", &mut seen);
        assert!(seen.len() >= 8, "{seen:?}");
    }

    /// No arguments at all, and a subcommand with none, both end in the help (clap reports it
    /// as `DisplayHelpOnMissingArgumentOrSubcommand`; `main` prints it to stdout, exit code 0).
    #[test]
    fn no_arguments_is_help() {
        for args in [&["exr-rs"][..], &["exr-rs", "info"], &["exr-rs", "cryptomatte", "extract"]] {
            let err = Cli::try_parse_from(args).err().unwrap_or_else(|| panic!("{args:?} parsed"));
            assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand, "{args:?}");
            assert!(err.to_string().contains("Examples:"), "{args:?}");
        }
    }

    /// Parse every current spelling with real arguments into the canonical subcommand.
    #[test]
    fn aliases_parse_the_same_commands() {
        let cases: [(&str, &[&str], &[&str]); 14] = [
            ("info", &["i", "header", "exrheader"], &["shot.exr"]),
            ("convert", &["x"], &["in.exr", "out.exr"]),
            ("cryptomatte", &["c"], &["info", "shot.exr"]),
            ("view", &["v"], &["--empty"]),
            ("check", &["k", "exrcheck"], &["shot.exr"]),
            ("tile", &["t", "maketiled", "exrmaketiled"], &["in.exr", "out.exr"]),
            ("preview", &["p", "makepreview", "exrmakepreview"], &["in.exr", "out.exr"]),
            ("attr", &["a", "stdattr", "exrstdattr"], &["-owner", "Jane", "in.exr", "out.exr"]),
            ("multipart", &["m", "exrmultipart"], &["-combine", "-i", "in.exr", "-o", "out.exr"]),
            ("multiview", &["s", "exrmultiview"], &["left", "l.exr", "right", "r.exr", "out.exr"]),
            ("envmap", &["e", "exrenvmap"], &["in.exr", "out.exr"]),
            ("metrics", &["b", "exrmetrics"], &["shot.exr"]),
            ("manifest", &["n", "exrmanifest"], &["shot.exr"]),
            ("aces", &["A", "2aces", "exr2aces"], &["in.exr", "out.aces"]),
        ];
        for (name, aliases, words) in cases {
            let long = Cli::try_parse_from(["exr-rs", name].into_iter().chain(words.iter().copied())).unwrap();
            for alias in aliases {
                let argv: Vec<_> = ["exr-rs", alias].into_iter().chain(words.iter().copied()).collect();
                let short = Cli::try_parse_from(&argv).unwrap();
                assert!(std::mem::discriminant(&long.command) == std::mem::discriminant(&short.command), "{name}/{alias}");
                let matches = Cli::command().try_get_matches_from(&argv).unwrap();
                assert_eq!(matches.subcommand_name(), Some(name), "{name}/{alias}");
            }
        }
        for (name, aliases, words) in [
            ("info", &["i"][..], &["shot.exr"][..]),
            ("extract", &["e"][..], &["shot.exr", "-n", "hero"][..]),
            ("encode", &["n", "enc"][..], &["out.exr", "-c", "CryptoObject", "-m", "hero=hero.exr"][..]),
            ("inject", &["j", "inj"][..], &["in.exr", "out.exr", "-n", "hero", "-m", "hero.exr"][..]),
        ] {
            for spelling in [name].into_iter().chain(aliases.iter().copied()) {
                let argv: Vec<_> = ["exr-rs", "c", spelling].into_iter().chain(words.iter().copied()).collect();
                assert!(matches!(Cli::try_parse_from(&argv).unwrap().command, Command::Cryptomatte(_)));
                let matches = Cli::command().try_get_matches_from(&argv).unwrap();
                assert_eq!(matches.subcommand().unwrap().1.subcommand_name(), Some(name));
            }
        }
    }
}
