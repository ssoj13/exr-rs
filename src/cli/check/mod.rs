//! `exr-rs check` — port of OpenEXR's `exrcheck` (`openexr/src/bin/exrcheck/main.cpp`, v3.4.15):
//! read every file through every OpenEXR read path and report `OK` or `bad`.
//!
//! Output and exit codes are exrcheck's: ` file NAME ` then `OK` / `bad` per file on stdout, exit 1 if
//! any file was bad, `No such file: NAME` on stderr and exit -1 (255 on Unix) for an unreadable
//! path. Deviations, all deliberate:
//! * with no arguments the project convention applies (help on stdout, exit 0) instead of exrcheck's
//!   usage on stderr with exit 1;
//! * `-m -t -s -c` apply to every file of the command line; exrcheck applies them only to the files
//!   that come after them on the command line;
//! * `-h` / `--version` are handled before any file is checked (exrcheck handles them where it meets
//!   them).

use exr_util::check_file::{CheckOpts, check_bytes, check_file};

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs check shot.exr                  # OK or bad, exit code 1 if bad",
    "exr-rs check a.exr b.exr c.exr         # several files, one line each",
    "exr-rs check -m -t frames/*.exr        # quick screening of untrusted files (less memory, less time)",
    "exr-rs check -c shot.exr               # the OpenEXRCore-style check (chunk by chunk)",
    "exr-rs check -s shot.exr               # read through an in-memory stream instead of the file",
);

/// `check` arguments (the options of `exrcheck`).
#[derive(clap::Args)]
pub struct Args {
    /// Avoid excessive memory allocation (some files will not be fully checked).
    #[arg(short = 'm', long = "reduce-memory")]
    reduce_memory: bool,

    /// Avoid spending excessive time (some files will not be fully checked).
    #[arg(short = 't', long = "reduce-time")]
    reduce_time: bool,

    /// Use the stream API (the file is read into memory) instead of the file API.
    #[arg(short = 's', long = "stream")]
    stream: bool,

    /// Add the core library checks: only the chunk-level (OpenEXRCore-style) path is run.
    #[arg(short = 'c', long = "core")]
    core: bool,

    /// Print version information.
    #[arg(long)]
    version: bool,

    /// Input files.
    #[arg(required_unless_present = "version")]
    files: Vec<String>,
}

/// exrcheck's `exrCheck`: `true` if the file is bad. `-s` loads the whole file and checks the memory.
fn check_one(path: &str, args: &Args) -> bool {
    let opts = CheckOpts { reduce_memory: args.reduce_memory, reduce_time: args.reduce_time, core: args.core };
    if !args.stream {
        return check_file(path, opts);
    }
    match std::fs::read(path) {
        Ok(data) => check_bytes(&data, opts),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound || e.kind() == std::io::ErrorKind::PermissionDenied => {
            errln!("internal error: bad file '{path}' for in-memory stream");
            true
        }
        Err(_) => {
            errln!("internal error: failed to read file {path}");
            true
        }
    }
}

/// Run `check`.
pub fn run(args: Args) -> CliResult {
    if args.version {
        super::print_version("exrcheck");
        return Ok(());
    }
    let mut bad_found = false;
    for path in &args.files {
        if std::fs::File::open(path).is_err() {
            errln!("No such file: {path}");
            return Err(CliError::Exit(-1));
        }
        out!(" file {path} ");
        super::stdio::flush();
        if check_one(path, &args) {
            outln!("bad");
            bad_found = true;
        } else {
            outln!("OK");
        }
    }
    if bad_found { Err(CliError::Exit(1)) } else { Ok(()) }
}
