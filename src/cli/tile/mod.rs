//! `exr-rs tile` — port of OpenEXR's `exrmaketiled` (`openexr/src/bin/exrmaketiled`, v3.4.15):
//! produce a tiled, optionally MIP- or RIP-mapped, version of an image. This file is `main.cpp`
//! (the command line and the input checks); the conversion is `make_tiled`. The in-memory image
//! class of the C++ (`Image.cpp/h`) is `exr_util::flat_image_level::FlatImageLevel`, see
//! `src/cli/tool_io.rs` for how its channels meet the file readers and writers.
//!
//! Options are the tool's single-dash ones (`-m`, `-t 32 32`), parsed left to right like the C++.
//! Deliberate differences from `main.cpp`, each a defect there:
//! * an unknown `-option` is an error (the C++ takes it for a file name); `--` ends the options;
//! * a number that is not entirely a number, or does not fit an `int`, is an error where `strtol`
//!   would read its prefix, 0 or saturate (`-p x` silently meant part 0; `-t x 8` stays "Tile size must
//!   be greater than zero", which is what 0 gets in the C++);
//! * "same file" also catches two spellings of one path (`main.cpp:336` compares the strings);
//! * without arguments the project convention applies (help on stdout, exit 0), not usage on
//!   stderr with exit -1; `-h` prints the clap-rendered option reference, not the C++ text;
//! * `-v` is the global `exr-rs -v` as well (it is accepted anywhere, as the C++ flag is): the
//!   messages appear when the log level is `debug` or finer, or when the tool's own `-v` was given.
//!
//! Kept from the C++: `-p` echoes `part number: N` on stdout while parsing, and a missing file name
//! prints the one-line usage on stderr and exits -1.

mod make_tiled;

use std::collections::BTreeSet;

use exr_core::ExrError;
use exr_core::attr::{Compression, LevelMode, LevelRoundingMode};
use exr_core::header;
use exr_core::io::StdIfStream;
use exr_core::multipart::MultiPartInputFile;

use super::tool_io::{c_int, get_compression, print_help, same_file, values};
use super::{CliError, CliResult, examples};
use make_tiled::{Extrapolation, Job, make_tiled};

pub const EXAMPLES: &str = examples!(
    "exr-rs tile in.exr out.exr                 # 64 x 64 tiles, one level",
    "exr-rs tile -m in.exr out.exr              # a MIP-mapped image",
    "exr-rs tile -r -t 128 128 -z piz in.exr out.exr   # RIP-mapped, 128 x 128 tiles, PIZ",
    "exr-rs tile -m -e periodic periodic -f A in.exr out.exr   # wrap around; do not filter A",
    "exr-rs tile -p 1 multi.exr out.exr         # tile part 1 of a multipart file",
);

/// `tile` arguments: the tool has its own grammar ([`parse`]), so clap only hands over the words
/// and shows the option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs tile [options] <infile> <outfile>"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = help())]
    args: Vec<String>,
}

/// `usageMessage (.., verbose = true)`'s text, for the help of the arguments.
fn help() -> String {
    format!(
        "Read an OpenEXR image from infile, produce a tiled version of the image, and save the\n\
         result in outfile.\n\
         \n\
         Options:\n\
         \n  -o            produces a ONE_LEVEL image (default)\n\
         \n  -m            produces a MIPMAP_LEVELS multiresolution image\n\
         \n  -r            produces a RIPMAP_LEVELS multiresolution image\n\
         \n  -f c          when a MIPMAP_LEVELS or RIPMAP_LEVELS image is produced, image channel c\n\
         \x20               will be resampled without low-pass filtering. This option can be\n\
         \x20               specified multiple times to disable low-pass filtering for multiple\n\
         \x20               channels.\n\
         \n  -e x y        when a MIPMAP_LEVELS or RIPMAP_LEVELS image is produced, low-pass\n\
         \x20               filtering takes samples outside the image's data window. This requires\n\
         \x20               extrapolating the image. Option -e specifies how the image is\n\
         \x20               extrapolated horizontally and vertically (black/clamp/periodic/mirror,\n\
         \x20               default is clamp).\n\
         \n  -t x y        sets the tile size in the output image to x by y pixels (default is\n\
         \x20               64 by 64)\n\
         \n  -d            sets level size rounding to ROUND_DOWN (default)\n\
         \n  -u            sets level size rounding to ROUND_UP\n\
         \n  -z x          sets the data compression method to x ({names},\n\
         \x20               default is zip)\n\
         \n  -v            verbose mode\n\
         \n  -h, --help    print this message\n\
         \n      --version print version information\n\
         \n\
         Multipart Options:\n\
         \n  -p i          part number, default is 0\n",
        names = Compression::names_string("/")
    )
}

/// What the command line asks for.
enum Cmd {
    Help,
    Version,
    /// A file name is missing: the C++ prints the usage line and exits -1.
    Usage,
    Make(Box<Job>),
}

/// `getExtrapolation`: the lower- or the upper-case word, nothing in between.
fn get_extrapolation(s: &str) -> Result<Extrapolation, String> {
    match s {
        "black" | "BLACK" => Ok(Extrapolation::Black),
        "clamp" | "CLAMP" => Ok(Extrapolation::Clamp),
        "periodic" | "PERIODIC" => Ok(Extrapolation::Periodic),
        "mirror" | "MIRROR" => Ok(Extrapolation::Mirror),
        _ => Err(format!("Unknown extrapolation method \"{s}\"")),
    }
}

/// `main`'s argument loop: left to right, the first error wins, `-h` / `--version` act when reached.
fn parse(argv: &[String]) -> Result<Cmd, String> {
    let (mut input, mut output): (Option<&str>, Option<&str>) = (None, None);
    let mut job = Job {
        input: String::new(),
        output: String::new(),
        part: 0,
        mode: LevelMode::OneLevel,
        rounding: LevelRoundingMode::RoundDown,
        compression: Compression::Zip,
        tile: (64, 64),
        no_filter: BTreeSet::new(),
        ext: (Extrapolation::Clamp, Extrapolation::Clamp),
        verbose: false,
    };
    let mut only_files = false;
    let mut i = 0;
    while i < argv.len() {
        let tok = argv[i].as_str();
        let mut used = 1;
        if only_files || !tok.starts_with('-') || tok == "-" {
            // The first name is the input, a later one the output (a third replaces the second).
            if input.is_none() {
                input = Some(tok);
            } else {
                output = Some(tok);
            }
            i += 1;
            continue;
        }
        match tok {
            "--" => only_files = true,
            "-o" => job.mode = LevelMode::OneLevel,
            "-m" => job.mode = LevelMode::MipmapLevels,
            "-r" => job.mode = LevelMode::RipmapLevels,
            "-f" => {
                job.no_filter.insert(values(argv, i, 1, "missing filter value with -f option")?[0].clone().into_bytes());
                used = 2;
            }
            "-e" => {
                let v = values(argv, i, 2, "missing extrapolation values with -e option")?;
                job.ext = (get_extrapolation(&v[0])?, get_extrapolation(&v[1])?);
                used = 3;
            }
            "-t" => {
                let v = values(argv, i, 2, "missing tile size with -t option")?;
                // A non-number is 0 to `strtol`, and 0 is rejected just below.
                let (x, y) = (c_int(&v[0]).unwrap_or(0), c_int(&v[1]).unwrap_or(0));
                if x <= 0 || y <= 0 {
                    return Err("Tile size must be greater than zero".into());
                }
                job.tile = (x as u32, y as u32);
                used = 3;
            }
            "-d" => job.rounding = LevelRoundingMode::RoundDown,
            "-u" => job.rounding = LevelRoundingMode::RoundUp,
            "-z" => {
                job.compression = get_compression(&values(argv, i, 1, "Missing compression value with -z option")?[0])?;
                used = 2;
            }
            "-v" => job.verbose = true,
            "-p" => {
                let v = &values(argv, i, 1, "Missing part num with -p option")?[0];
                job.part = c_int(v)?;
                outln!("part number: {}", job.part);
                used = 2;
            }
            "-h" | "--help" => return Ok(Cmd::Help),
            "--version" => return Ok(Cmd::Version),
            _ => return Err(format!("Unknown option {tok}")),
        }
        i += used;
    }
    let (Some(input), Some(output)) = (input, output) else { return Ok(Cmd::Usage) };
    job.input = input.into();
    job.output = output.into();
    Ok(Cmd::Make(Box::new(job)))
}

/// Run `tile`: parse, then the input checks of `main` and the conversion.
pub fn run(args: Args, verbose: bool) -> CliResult {
    match parse(&args.args).map_err(CliError::Other)? {
        Cmd::Help => print_help("tile"),
        Cmd::Version => {
            super::print_version("exrmaketiled");
            Ok(())
        }
        Cmd::Usage => {
            errln!("Usage: exr-rs tile [options] infile outfile");
            Err(CliError::Exit(-1))
        }
        Cmd::Make(mut job) => {
            job.verbose |= verbose;
            make(&job)
        }
    }
}

/// `main`'s tail: the checks on the input, then `makeTiled`.
fn make(job: &Job) -> CliResult {
    if same_file(&job.input, &job.output) {
        return Err(CliError::Other("Input and output cannot be the same file".into()));
    }
    let mut stream = StdIfStream::open(&job.input)?;
    let mut input = MultiPartInputFile::new(&mut stream)?;
    let parts = input.parts();
    if job.part < 0 || job.part >= parts {
        return Err(CliError::Other(format!(
            "You asked for part {} in {}, which only has {parts} parts",
            job.part, job.input
        )));
    }
    if matches!(input.part_type(job.part)?, header::DEEP_TILE | header::DEEP_SCANLINE) {
        return Err(CliError::Exr(ExrError::Arg("Cannot make tile for deep data".into())));
    }
    Ok(make_tiled(&mut input, job)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn job(line: &str) -> Job {
        match parse(&args(line)) {
            Ok(Cmd::Make(j)) => *j,
            Ok(_) => panic!("not a job: {line}"),
            Err(e) => panic!("{line}: {e}"),
        }
    }

    fn err(line: &str) -> String {
        match parse(&args(line)) {
            Err(e) => e,
            Ok(_) => panic!("{line} parsed"),
        }
    }

    #[test]
    fn defaults_are_the_cpp_ones() {
        let j = job("in.exr out.exr");
        assert_eq!((j.input.as_str(), j.output.as_str()), ("in.exr", "out.exr"));
        assert_eq!((j.mode, j.rounding, j.compression, j.tile, j.part), (LevelMode::OneLevel, LevelRoundingMode::RoundDown, Compression::Zip, (64, 64), 0));
        assert_eq!(j.ext, (Extrapolation::Clamp, Extrapolation::Clamp));
        assert!(j.no_filter.is_empty() && !j.verbose);
    }

    #[test]
    fn the_last_of_competing_options_wins() {
        assert_eq!(job("-m -r a b").mode, LevelMode::RipmapLevels);
        assert_eq!(job("-r -o a b").mode, LevelMode::OneLevel);
        assert_eq!(job("-u -d a b").rounding, LevelRoundingMode::RoundDown);
        assert_eq!(job("-d -u a b").rounding, LevelRoundingMode::RoundUp);
        assert_eq!(job("a -m b").mode, LevelMode::MipmapLevels, "options may follow the files");
    }

    #[test]
    fn options_with_values() {
        let j = job("-f R -f Z -e black MIRROR -t 0x20 16 -z PIZ -v -p 2 a b");
        assert_eq!(j.no_filter, BTreeSet::from([b"R".to_vec(), b"Z".to_vec()]));
        assert_eq!(j.ext, (Extrapolation::Black, Extrapolation::Mirror));
        assert_eq!((j.tile, j.compression, j.verbose, j.part), ((32, 16), Compression::Piz, true, 2));
    }

    #[test]
    fn the_cpp_error_messages() {
        assert_eq!(err("-f"), "missing filter value with -f option");
        assert_eq!(err("a b -e clamp"), "missing extrapolation values with -e option");
        assert_eq!(err("-t 8"), "missing tile size with -t option");
        assert_eq!(err("-z"), "Missing compression value with -z option");
        assert_eq!(err("-p"), "Missing part num with -p option");
        assert_eq!(err("-t 0 8 a b"), "Tile size must be greater than zero");
        assert_eq!(err("-t x 8 a b"), "Tile size must be greater than zero");
        assert_eq!(err("-z lzw a b"), "Unknown compression method \"lzw\"");
        assert_eq!(err("-e Clamp clamp a b"), "Unknown extrapolation method \"Clamp\"");
        assert_eq!(err("-p x a b"), "Expected an integer, got \"x\"");
        assert_eq!(err("-x a b"), "Unknown option -x");
    }

    #[test]
    fn help_version_and_usage_act_where_they_are_met() {
        assert!(matches!(parse(&args("a b --version")), Ok(Cmd::Version)));
        assert!(matches!(parse(&args("-m -h")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("a")), Ok(Cmd::Usage)));
        assert!(matches!(parse(&args("-m")), Ok(Cmd::Usage)));
        // An earlier error wins over a later -h.
        assert_eq!(err("-t 0 0 -h"), "Tile size must be greater than zero");
        // `--` ends the options: what follows is a file name.
        assert_eq!(job("-- -a b").input, "-a");
    }

    #[test]
    fn the_help_lists_every_option_and_codec() {
        let h = help();
        for opt in ["-o ", "-m ", "-r ", "-f c", "-e x y", "-t x y", "-d ", "-u ", "-z x", "-v ", "-h, --help", "--version", "-p i"] {
            assert!(h.contains(opt), "help lacks {opt}");
        }
        for c in Compression::ALL {
            assert!(h.contains(c.name()), "help lacks {}", c.name());
        }
    }
}
