//! `exr-rs metrics` — port of OpenEXR's `exrmetrics` (`openexr/src/bin/exrmetrics`, v3.4.15):
//! read an image, write an identical copy (to memory, or to a file), read the copy back, and report
//! the time each step took and the file sizes, as JSON or CSV, for any number of input files,
//! compression methods and pixel data types. This file is `main.cpp`'s `main`; the command line is
//! `opts`, the reports are `report`, the measuring is `exrmetrics`.
//!
//! The grammar is the tool's own (see `opts`): any word that is not an option is an input file.
//! Times are wall-clock seconds and so differ from run to run; every other number is exact and
//! equals the C++ tool's (same writers, same byte counts).
//!
//! Deliberate differences from `main.cpp`, each a defect there:
//! * `-h` and `--version` print their text and stop. The C++ goes on with no input and no options
//!   and prints an empty report (`[` / `]`) after them (`parse` returns 0, `main` only tests for
//!   non-zero);
//! * without arguments the project convention applies (help on stdout, exit 0), not usage on stderr
//!   with exit 1; `-h` prints the clap-rendered option reference;
//! * a number that is not a number is an error (`-t x`, `-p 0abc`, `-l abc`; `main.cpp:656,700,773,
//!   790` take the digits it starts with, else 0); hexadecimal forms are numbers. A lone `-v` is
//!   taken by clap as the global flag, so it ends in clap's usage error (exit 2) rather than ours.
//! * the report fixes what the C++ gets wrong: see `exrmetrics.rs` (mip tile count, deep tiled
//!   bookkeeping, `mixed` suffix) and `report.rs` (`null` for no samples).
//!
//! Every error exits with code 1, as in the C++ (its message is `error from exrmetrics: ...`).

mod exrmetrics;
mod opts;
mod report;

use exr_core::container::version::is_non_image;
use exr_core::io::StdIfStream;
use exr_core::multipart::MultiPartInputFileOwned;
use exr_core::threading::set_global_thread_count;

use super::tool_io::print_help;
use super::{CliError, CliResult, examples};
use exrmetrics::exrmetrics;
use opts::{Cmd, Opts, TIME_REREAD, TIME_WRITE};
use report::{Run, csv_stats, json_stats};

pub const EXAMPLES: &str = examples!(
    "exr-rs metrics shot.exr                          # sizes and read / write / re-read times as JSON",
    "exr-rs metrics -z all --csv shot.exr             # one row per compression method",
    "exr-rs metrics -z piz,zip --pixelmode half,float a.exr b.exr   # every combination, both files",
    "exr-rs metrics --passes 10 --time write shot.exr # ten timed writes, with statistics",
    "exr-rs metrics --convert -z dwaa -l 60 -o out.exr in.exr   # just write a recompressed copy",
);

/// `metrics` arguments: the tool has its own grammar ([`opts::parse`]), so clap only hands over the
/// words and shows the option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs metrics [options] infile [infile2...] [-o outfile]"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = opts::help())]
    args: Vec<String>,
}

/// Run `metrics`.
pub fn run(args: Args, verbose: bool) -> CliResult {
    let o = match opts::parse(&args.args).map_err(CliError::Other)? {
        Cmd::Help => return print_help("metrics"),
        Cmd::Version => {
            super::print_version("exrmetrics");
            return Ok(());
        }
        Cmd::Run(o) => o,
    };
    // `-v` given after the command is the global `exr-rs -v`, which clap takes first: debug level.
    let verbose = o.verbose || verbose;
    let runs = measure(&o, verbose)?;

    // Per-part sizes on disk need the memory stream, so an output file rules them out.
    let part_size = o.output_part_size_on_disk && o.out_file.is_none();
    if o.timing != 0 || o.output_size_data {
        let text = if o.csv { csv_stats(&runs, o.output_size_data, o.timing) } else { json_stats(&runs, o.output_size_data, o.timing, part_size, true, true) };
        out!("{text}");
    }
    Ok(())
}

/// `main`'s measuring loop: every input file through every compression and pixel mode.
fn measure(o: &Opts, verbose: bool) -> Result<Vec<Run>, CliError> {
    let threads = if o.threads < 0 { std::thread::available_parallelism().map_or(1, |n| n.get() as i32) } else { o.threads };
    set_global_thread_count(threads)?;

    let mut runs = Vec::new();
    for (file, name) in o.in_files.iter().enumerate() {
        // With an explicit compression list, the methods a deep file cannot use are skipped (the
        // original method is always measured).
        let mut has_deep = false;
        if o.compressions.len() > 1 && o.compressions[0].is_some() {
            has_deep = is_deep(name)?;
        }
        for &compression in &o.compressions {
            if !has_deep || compression.is_none_or(|c| c.is_valid_deep()) {
                for &mode in &o.pixel_modes {
                    let metrics = exrmetrics(
                        name,
                        o.out_file.as_deref(),
                        o.part,
                        compression,
                        o.level,
                        o.passes,
                        o.out_file.is_some() || o.output_size_data || o.timing & TIME_WRITE != 0,
                        o.timing & TIME_REREAD != 0,
                        mode,
                        verbose,
                    )?;
                    runs.push(Run { file, name: name.clone(), mode, compression, metrics });
                }
            }
        }
    }
    Ok(runs)
}

/// `isNonImage (in.version ())`: whether the file holds deep data, from its version word.
fn is_deep(path: &str) -> Result<bool, CliError> {
    let file = MultiPartInputFileOwned::new_owned(Box::new(StdIfStream::open(path)?))?;
    // The version word is a signed `int` in the C++ and unsigned here; only its flag bits matter.
    Ok(is_non_image(file.version() as u32))
}
