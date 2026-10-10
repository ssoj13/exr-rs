//! `exr-rs preview` — port of OpenEXR's `exrmakepreview` (`openexr/src/bin/exrmakepreview`,
//! v3.4.15): insert a preview image (thumbnail) into an image's header. This file is `main.cpp`
//! (the command line); the work is `make_preview`.
//!
//! Options are the tool's single-dash ones (`-w 200`, `-e 1.5`), parsed left to right like the C++.
//! Deliberate differences from `main.cpp`, each a defect there:
//! * an unknown `-option` is an error (the C++ takes it for a file name); `--` ends the options;
//! * an exposure that is not a number is an error (`strtod` read it as 0); a width that is not a
//!   number stays "preview width must be greater than zero" (`strtol` read it as 0, which that
//!   message rejects);
//! * "same file" also catches two spellings of one path (`main.cpp:164` compares the strings);
//! * without arguments the project convention applies (help on stdout, exit 0), not usage on
//!   stderr with exit -1; `-h` prints the clap-rendered option reference, not the C++ text;
//! * `-v` is the global `exr-rs -v` as well (it is accepted anywhere, as the C++ flag is): the
//!   messages appear when the log level is `debug` or finer, or when the tool's own `-v` was given.
//!
//! Kept from the C++: a missing file name prints the one-line usage on stderr and exits -1.

mod make_preview;

use super::tool_io::{c_f32, c_int, print_help, same_file};
use super::{CliError, CliResult, examples};
use make_preview::make_preview;

pub const EXAMPLES: &str = examples!(
    "exr-rs preview in.exr out.exr             # a 100 pixel wide preview",
    "exr-rs preview -w 256 in.exr out.exr      # 256 pixels wide",
    "exr-rs preview -e 1.5 in.exr out.exr      # 1.5 f-stops brighter",
    "exr-rs preview -v in.exr out.exr          # say what is being done",
);

/// `preview` arguments: the tool has its own grammar ([`parse`]), so clap only hands over the words
/// and shows the option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs preview [options] <infile> <outfile>"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = HELP)]
    args: Vec<String>,
}

/// `usageMessage (.., verbose = true)`'s text, for the help of the arguments.
const HELP: &str = "\
Read an OpenEXR image from infile, generate a preview image, add it to the image's header, and
save the result in outfile. Infile and outfile must not refer to the same file (the program
cannot edit an image file \"in place\").

Options:

  -w x          sets the width of the preview image to x pixels (default is 100)

  -e s          adjusts the preview image's exposure by s f-stops (default is 0). Positive
                values make the image brighter, negative values make it darker.

  -v            verbose mode

  -h, --help    print this message

      --version print version information
";

/// What `main` runs: `-w`, `-e`, `-v` and the two file names.
struct Job {
    input: String,
    output: String,
    width: i32,
    exposure: f32,
    verbose: bool,
}

/// What the command line asks for.
enum Cmd {
    Help,
    Version,
    /// A file name is missing: the C++ prints the usage line and exits -1.
    Usage,
    Make(Job),
}

/// The value token after option `i`, or the error the C++ throws for a missing one.
fn value<'a>(argv: &'a [String], i: usize, missing: &str) -> Result<&'a str, String> {
    argv.get(i + 1).map(String::as_str).ok_or_else(|| missing.to_string())
}

/// `main`'s argument loop: left to right, the first error wins, `-h` / `--version` act when reached.
fn parse(argv: &[String]) -> Result<Cmd, String> {
    let (mut input, mut output): (Option<&str>, Option<&str>) = (None, None);
    let (mut width, mut exposure, mut verbose) = (100, 0.0f32, false);
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
            "-w" => {
                // A non-number is 0 to `strtol`, and 0 is rejected just below.
                width = c_int(value(argv, i, "Missing width for -w argument")?).unwrap_or(0);
                if width <= 0 {
                    return Err("preview width must be greater than zero".into());
                }
                used = 2;
            }
            "-e" => {
                let v = value(argv, i, "missing exposure for -e argument")?;
                exposure = c_f32(v)?;
                used = 2;
            }
            "-v" => verbose = true,
            "-h" | "--help" => return Ok(Cmd::Help),
            "--version" => return Ok(Cmd::Version),
            _ => return Err(format!("Unknown option {tok}")),
        }
        i += used;
    }
    let (Some(input), Some(output)) = (input, output) else { return Ok(Cmd::Usage) };
    Ok(Cmd::Make(Job { input: input.into(), output: output.into(), width, exposure, verbose }))
}

/// Run `preview`: parse, check the names, make the preview.
pub fn run(args: Args, verbose: bool) -> CliResult {
    match parse(&args.args).map_err(CliError::Other)? {
        Cmd::Help => print_help("preview"),
        Cmd::Version => {
            super::print_version("exrmakepreview");
            Ok(())
        }
        Cmd::Usage => {
            errln!("Usage: exr-rs preview [options] infile outfile");
            Err(CliError::Exit(-1))
        }
        Cmd::Make(job) => {
            if same_file(&job.input, &job.output) {
                return Err(CliError::Other("Input and output cannot be the same file".into()));
            }
            let verbose = job.verbose || verbose;
            Ok(make_preview(&job.input, &job.output, job.width, job.exposure, verbose)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn job(line: &str) -> Job {
        match parse(&args(line)) {
            Ok(Cmd::Make(j)) => j,
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
    fn defaults_and_options() {
        let j = job("a b");
        assert_eq!((j.width, j.exposure, j.verbose), (100, 0.0, false));
        let j = job("-w 0x20 -e -1.5 -v a b");
        assert_eq!((j.width, j.exposure, j.verbose), (32, -1.5, true));
        assert_eq!((j.input.as_str(), j.output.as_str()), ("a", "b"));
        assert_eq!(job("a -w 7 b").width, 7, "options may sit between the files");
    }

    #[test]
    fn the_cpp_error_messages() {
        assert_eq!(err("-w"), "Missing width for -w argument");
        assert_eq!(err("a b -e"), "missing exposure for -e argument");
        assert_eq!(err("-w 0 a b"), "preview width must be greater than zero");
        assert_eq!(err("-w -3 a b"), "preview width must be greater than zero");
        assert_eq!(err("-w x a b"), "preview width must be greater than zero");
        assert_eq!(err("-e x a b"), "Expected a number, got \"x\"");
        assert_eq!(err("-q a b"), "Unknown option -q");
    }

    #[test]
    fn help_version_and_usage_act_where_they_are_met() {
        assert!(matches!(parse(&args("a b --version")), Ok(Cmd::Version)));
        assert!(matches!(parse(&args("-v -h")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("a")), Ok(Cmd::Usage)));
        assert_eq!(err("-w 0 -h"), "preview width must be greater than zero");
    }
}
