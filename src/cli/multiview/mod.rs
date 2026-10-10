//! `exr-rs multiview` — port of OpenEXR's `exrmultiview` (`openexr/src/bin/exrmultiview`,
//! v3.4.15): combine single-view images into one multi-view (stereo) image. This file is `main.cpp`
//! (the command line); the work is `make_multi_view`. The in-memory image class of the C++
//! (`Image.cpp/h`) is `exr_util::flat_image_level::FlatImageLevel`, see `src/cli/tool_io.rs` for how
//! its channels meet the file readers and writers.
//!
//! The grammar is the tool's own: `viewname1 infile1 viewname2 infile2 ... outfile`, a lone name
//! (or one followed by an option) being the output; `-z`, `-v` may stand anywhere. Every error,
//! usage errors included, leaves with exit code -1, as in the C++.
//!
//! Deliberate differences from `main.cpp`, each a defect there:
//! * an unknown `-option` is an error (the C++ takes it for a name); `--` ends the options;
//! * without arguments the project convention applies (help on stdout, exit 0), not usage on
//!   stderr with exit -1; `-h` prints the clap-rendered option reference, not the C++ text;
//! * `-v` is the global `exr-rs -v` as well (it is accepted anywhere, as the C++ flag is): the
//!   messages appear when the log level is `debug` or finer, or when the tool's own `-v` was given.
//!
//! The output may be one of the inputs, as in the C++: every input is closed, its pixels held in
//! memory, before the output file is created.

mod make_multi_view;

use exr_core::attr::Compression;

use super::tool_io::{get_compression, print_help};
use super::{CliError, CliResult, examples};
use make_multi_view::make_multi_view;

pub const EXAMPLES: &str = examples!(
    "exr-rs multiview left imgL.exr right imgR.exr out.exr     # left is the default view",
    "exr-rs multiview -z zip left L.exr right R.exr center C.exr out.exr   # three views, ZIP",
    "exr-rs multiview -v left L.exr right R.exr out.exr        # say what is being read and written",
);

/// `multiview` arguments: the tool has its own grammar ([`parse`]), so clap only hands over the
/// words and shows the option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs multiview [options] <viewname1> <infile1> <viewname2> <infile2> ... <outfile>"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = help())]
    args: Vec<String>,
}

/// `usageMessage (.., verbose = true)`'s text, for the help of the arguments.
fn help() -> String {
    format!(
        "Combine two or more single-view OpenEXR image files into a single multi-view image file.\n\
         On the command line, each single-view input image is specified together with a\n\
         corresponding view name. The first view on the command line becomes the default view.\n\
         Example:\n\
         \n  exr-rs multiview left imgL.exr right imgR.exr imgLR.exr\n\
         \n\
         Here, imgL.exr and imgR.exr become the left and right views in output file imgLR.exr.\n\
         The left view becomes the default view.\n\
         \n\
         Options:\n\
         \n  -z x          sets the data compression method to x ({names},\n\
         \x20               default is piz)\n\
         \n  -v            verbose mode\n\
         \n  -h, --help    print this message\n\
         \n      --version print version information\n",
        names = Compression::names_string("/")
    )
}

/// What `main` runs: views with their files, the output, the codec.
struct Job {
    views: Vec<String>,
    in_files: Vec<String>,
    output: String,
    compression: Compression,
    verbose: bool,
}

/// What the command line asks for.
enum Cmd {
    Help,
    Version,
    Make(Job),
}

/// `main`'s argument loop: left to right, the first error wins, `-h` / `--version` act when reached.
fn parse(argv: &[String]) -> Result<Cmd, String> {
    let (mut views, mut in_files) = (Vec::new(), Vec::new());
    let mut output: Option<&str> = None;
    let (mut compression, mut verbose) = (Compression::Piz, false);
    let mut only_files = false;
    let mut i = 0;
    while i < argv.len() {
        let tok = argv[i].as_str();
        if !only_files && tok.starts_with('-') && tok != "-" {
            match tok {
                "--" => only_files = true,
                "-z" => {
                    let v = argv.get(i + 1).ok_or("Missing compression value with -z option")?;
                    compression = get_compression(v)?;
                    i += 1;
                }
                "-v" => verbose = true,
                "-h" | "--help" => return Ok(Cmd::Help),
                "--version" => return Ok(Cmd::Version),
                _ => return Err(format!("Unknown option {tok}")),
            }
            i += 1;
            continue;
        }
        // A view name and its file, unless it is the last word or an option follows: the output.
        match argv.get(i + 1) {
            Some(next) if !next.starts_with('-') => {
                views.push(tok.to_string());
                in_files.push(next.clone());
                i += 2;
            }
            _ => {
                if output.is_some() {
                    return Err("improper output file specified".into());
                }
                output = Some(tok);
                i += 1;
            }
        }
    }
    if views.len() < 2 {
        return Err("Must specify at least two views".into());
    }
    let output = output.ok_or("Must specify an output file")?.to_string();
    Ok(Cmd::Make(Job { views, in_files, output, compression, verbose }))
}

/// Run `multiview`; every error leaves with exit code -1.
pub fn run(args: Args, verbose: bool) -> CliResult {
    let job = match parse(&args.args) {
        Ok(Cmd::Help) => return print_help("multiview"),
        Ok(Cmd::Version) => {
            super::print_version("exrmultiview");
            return Ok(());
        }
        Ok(Cmd::Make(job)) => job,
        Err(e) => return Err(fail(&CliError::Other(e))),
    };
    let verbose = job.verbose || verbose;
    make_multi_view(&job.views, &job.in_files, &job.output, job.compression, verbose).map_err(|e| fail(&CliError::Exr(e)))
}

/// The C++ `catch` of `main`: the message on stderr, then exit code -1.
fn fail(err: &CliError) -> CliError {
    errln!("error: {err}");
    CliError::Exit(-1)
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
    fn views_files_and_output() {
        let j = job("left l.exr right r.exr out.exr");
        assert_eq!((j.views, j.in_files), (vec!["left".to_string(), "right".into()], vec!["l.exr".to_string(), "r.exr".into()]));
        assert_eq!((j.output.as_str(), j.compression, j.verbose), ("out.exr", Compression::Piz, false));
    }

    #[test]
    fn options_may_stand_anywhere() {
        let j = job("-v a 1 -z zip b 2 out -v");
        assert_eq!((j.compression, j.verbose, j.views.len(), j.output.as_str()), (Compression::Zip, true, 2, "out"));
        // A name followed by an option is the output, whatever comes after.
        assert_eq!(job("a 1 b 2 out -v").output, "out");
        // `out b` would be a view and its file: a pair is read before anything else.
        assert_eq!(job("a 1 out b 2").output, "2");
    }

    #[test]
    fn the_cpp_error_messages() {
        assert_eq!(err("a 1 out"), "Must specify at least two views");
        assert_eq!(err("a 1 b 2"), "Must specify an output file");
        assert_eq!(err("a 1 b 2 o1 -v o2"), "improper output file specified");
        assert_eq!(err("-z"), "Missing compression value with -z option");
        assert_eq!(err("-z lzw a 1 b 2 o"), "Unknown compression method \"lzw\"");
        assert_eq!(err("-q a 1 b 2 o"), "Unknown option -q");
    }

    #[test]
    fn help_and_version_act_where_they_are_met() {
        assert!(matches!(parse(&args("a 1 --version")), Ok(Cmd::Version)));
        assert!(matches!(parse(&args("-v -h")), Ok(Cmd::Help)));
        assert_eq!(err("-z x -h"), "Unknown compression method \"x\"");
    }

    #[test]
    fn the_help_lists_every_option_and_codec() {
        let h = help();
        for opt in ["-z x", "-v ", "-h, --help", "--version"] {
            assert!(h.contains(opt), "help lacks {opt}");
        }
        for c in Compression::ALL {
            assert!(h.contains(c.name()), "help lacks {}", c.name());
        }
    }
}
