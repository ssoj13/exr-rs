//! `exr-rs multipart` — port of OpenEXR's `exrmultipart` (`openexr/src/bin/exrmultipart`,
//! v3.4.15): combine files into one multipart file, separate a multipart file into single-part
//! files, or convert a multi-view image into a multipart one. This file is the command line
//! (`main`); the three modes are `combine`, `separate` and `convert`, `names` holds the part-name
//! rules they share.
//!
//! The grammar is the tool's own: a mode word first (`-combine`, `-separate`, `-convert`), then
//! `-i` followed by files, `-o` followed by the output, `-override [0/1]`, and `-view name` inside
//! `-i` to give the files after it a view. The last word after an option is its value, and for `-i`
//! every following word is another file, so a file cannot be called `-o`.
//!
//! Deliberate differences from `exrmultipart.cpp`, each a defect there:
//! * the usage line lacks the space after the program name (`exrmultipart-combine`);
//! * without arguments the project convention applies (help on stdout, exit 0), not usage on
//!   stderr; `-h` prints the clap-rendered option reference, not the C++ text;
//! * safety checks that see through spelling (`./a.exr` is `a.exr`), and pixels of unknown part
//!   types are an error: see `combine`, `separate`, `convert`.
//!
//! Kept from the C++: `argv[1]` alone names the mode (a mode word anywhere else is ignored, and an
//! unknown one prints the usage and still exits 0), `-override`'s value is a number (any later word
//! after it is read as the value again; a word that is not a number is an error, where `atoi` says 0), `-view` outside `-i` is an error.

mod combine;
mod convert;
mod names;
mod separate;

use exr_core::io::StdIfStream;
use exr_core::multipart::MultiPartInputFileOwned;

use super::tool_io::{c_int, print_help};
use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs multipart -combine -i beauty.exr diffuse.exr -o out.exr   # one part per file, named after it",
    "exr-rs multipart -combine -i a.exr:1::spec b.exr -o out.exr      # only part 1 of a.exr, called 'spec'",
    "exr-rs multipart -combine -i -view left l.exr -view right r.exr -o lr.exr   # files with views",
    "exr-rs multipart -separate -i in.exr -o part                     # part.1.exr, part.2.exr, ...",
    "exr-rs multipart -convert -i stereo.exr -o stereo2.exr           # multi-view -> multipart",
);

/// `multipart` arguments: the tool has its own grammar ([`parse`]), so clap only hands over the
/// words and shows the option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs multipart -combine -i input.exr[:partnum][::partname] [input2.exr[:partnum]][::partname] [...] -o outfile.exr [options]\n       exr-rs multipart -separate -i infile.exr -o outfileBaseName [options]\n       exr-rs multipart -convert -i infile.exr -o outfile.exr [options]"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = help())]
    args: Vec<String>,
}

/// `usageMessage (.., verbose = true)`'s text, for the help of the arguments.
fn help() -> String {
    "Combine or split multipart data\n\
     \n\
     Options:\n\
     \n  -override [0/1]   0 = do not override conflicting shared attributes [default]\n\
     \x20                   1 = override conflicting shared attributes\n\
     \n  -view name        (after specifying -i) assign following inputs to view 'name'\n\
     \n  -h, --help        print this message\n\
     \n      --version     print version information\n"
        .to_string()
}

/// The three usage lines of `usageMessage` (the C++ prints these on stderr when no input is given).
fn usage() -> String {
    "Usage: exr-rs multipart -combine -i input.exr[:partnum][::partname] [input2.exr[:partnum]][::partname] [...] -o outfile.exr [options]\n\
     \x20 or: exr-rs multipart -separate -i infile.exr -o outfileBaseName [options]\n\
     \x20 or: exr-rs multipart -convert -i infile.exr -o outfile.exr [options]\n"
        .to_string()
}

/// What `main` runs: the mode word, the files with their views, the output, `-override`.
struct Job {
    /// `argv[1]`: decides the mode, whatever else the line holds.
    mode: String,
    files: Vec<String>,
    views: Vec<Option<String>>,
    out: Option<String>,
    over: bool,
}

/// What the command line asks for.
enum Cmd {
    Help,
    Version,
    /// No input file: the usage, exit 1.
    Usage,
    Run(Job),
}

/// What the next plain word is: the C++'s `mode` variable.
#[derive(Clone, Copy, PartialEq)]
enum Take {
    Nothing,
    Files,
    Out,
    Over,
    View,
}

/// `main`'s argument loop: left to right, `-h` / `--version` act when reached, the first error wins.
fn parse(argv: &[String]) -> Result<Cmd, String> {
    let (mut files, mut views) = (Vec::new(), Vec::new());
    let (mut view, mut out, mut over): (Option<&str>, Option<String>, bool) = (None, None, false);
    let mut take = Take::Nothing;
    for tok in argv {
        match tok.as_str() {
            "-h" | "--help" => return Ok(Cmd::Help),
            "--version" => return Ok(Cmd::Version),
            "-i" => take = Take::Files,
            "-o" => take = Take::Out,
            "-override" => take = Take::Over,
            "-view" => {
                if take != Take::Files {
                    return Err("-view requires -i".into());
                }
                take = Take::View;
            }
            word => match take {
                Take::Nothing => {}
                Take::Files => {
                    files.push(word.to_string());
                    views.push(view.map(String::from));
                }
                Take::Out => out = Some(word.to_string()),
                Take::Over => over = c_int(word)? != 0,
                Take::View => {
                    view = Some(word);
                    take = Take::Files;
                }
            },
        }
    }
    let Some(mode) = argv.first().cloned().filter(|_| !files.is_empty()) else {
        return Ok(Cmd::Usage);
    };
    Ok(Cmd::Run(Job { mode, files, views, out, over }))
}

/// Open an input: one reader owning its file, so several inputs can be held at once.
fn open(path: &str) -> Result<MultiPartInputFileOwned<'static>, CliError> {
    Ok(MultiPartInputFileOwned::new_owned(Box::new(StdIfStream::open(path)?))?)
}

/// Run `multipart`; every error exits 1 (`CliError`), as the C++ `catch` does.
pub fn run(args: Args) -> CliResult {
    let job = match parse(&args.args).map_err(CliError::Other)? {
        Cmd::Help => return print_help("multipart"),
        Cmd::Version => {
            super::print_version("exrmultipart");
            return Ok(());
        }
        Cmd::Usage => {
            err!("{}", usage());
            return Err(CliError::Exit(1));
        }
        Cmd::Run(job) => job,
    };

    outln!("input:");
    for (f, v) in job.files.iter().zip(&job.views) {
        match v {
            Some(v) => outln!("      {f} in view {v}"),
            None => outln!("      {f}"),
        }
    }
    let out = job.out.as_deref().ok_or_else(|| CliError::Other("No output file specified".into()))?;
    outln!("output:\n      {out}");
    outln!("override:{}\n", u8::from(job.over));

    match job.mode.as_str() {
        "-combine" => {
            outln!("-combine multipart input ");
            combine::combine(&job.files, &job.views, out, job.over)
        }
        "-separate" => {
            outln!("-separate multipart input ");
            separate::separate(&job.files, out, job.over)
        }
        "-convert" => {
            outln!("-convert input to EXR2 multipart");
            convert::convert(&job.files, out)
        }
        // The C++ prints the usage and falls out of `main` with 0.
        _ => {
            err!("{}", usage());
            Ok(())
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
            Ok(Cmd::Run(j)) => j,
            Ok(_) => panic!("not a job: {line}"),
            Err(e) => panic!("{line}: {e}"),
        }
    }

    #[test]
    fn files_output_and_override() {
        let j = job("-combine -i a.exr b.exr -o out.exr -override 1");
        assert_eq!((j.mode.as_str(), j.files, j.out.as_deref(), j.over), ("-combine", vec!["a.exr".to_string(), "b.exr".into()], Some("out.exr"), true));
        assert_eq!(j.views, [None, None]);
        assert!(!job("-combine -i a -o o -override 0").over);
        assert!(!job("-combine -i a -o o").over);
    }

    /// `-view name` is sticky: it applies to every file after it, across later `-i`s.
    #[test]
    fn views_apply_to_the_files_after_them() {
        let j = job("-combine -i a.exr -view left b.exr c.exr -view right d.exr -o o");
        let v: Vec<_> = j.views.iter().map(|v| v.as_deref()).collect();
        assert_eq!(v, [None, Some("left"), Some("left"), Some("right")]);
        assert_eq!(job("-combine -i -view v a -i b -o o").views.len(), 2);
        assert_eq!(job("-combine -i -view v a -i b -o o").views[1].as_deref(), Some("v"));
    }

    /// The words after an option keep feeding it: the last `-o` word is the output, and the word
    /// after `-override`'s value is read as the value again.
    #[test]
    fn a_mode_keeps_taking_words() {
        assert_eq!(job("-combine -i a -o x y").out.as_deref(), Some("y"));
        assert!(!job("-combine -i a -o o -override 1 0").over, "the last word is the value");
        assert!(job("-combine -i a -o o -override 0 2").over);
        // A word that is not a number is an error (the C++ `atoi` would make it 0).
        for bad in ["-override x", "-override 1 x", "-override 1x", "-override 1.5"] {
            assert!(parse(&args(&format!("-combine -i a -o o {bad}"))).err().is_some_and(|e| e.starts_with("Expected ")), "{bad}");
        }
        // Words before any option are ignored, the mode word included.
        assert_eq!(job("-combine stray -i a -o o").files, ["a"]);
    }

    #[test]
    fn view_outside_files_is_an_error() {
        assert_eq!(parse(&args("-combine -view v -i a")).err().as_deref(), Some("-view requires -i"));
        assert_eq!(parse(&args("-combine -i a -o o -view v")).err().as_deref(), Some("-view requires -i"));
        assert_eq!(parse(&args("-combine -i a -view")).err(), None, "a dangling -view after -i is not an error");
    }

    #[test]
    fn help_and_version_act_where_they_are_met() {
        assert!(matches!(parse(&args("-combine -i a --version")), Ok(Cmd::Version)));
        assert!(matches!(parse(&args("-combine -h")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("-combine -i a -o --help")), Ok(Cmd::Help)), "an option's value never hides -h");
        assert_eq!(parse(&args("-combine -view v -h")).err().as_deref(), Some("-view requires -i"), "the first error wins");
    }

    #[test]
    fn no_input_is_the_usage() {
        assert!(matches!(parse(&args("-combine")), Ok(Cmd::Usage)));
        assert!(matches!(parse(&args("-combine -o out.exr")), Ok(Cmd::Usage)));
        assert!(matches!(parse(&args("-i")), Ok(Cmd::Usage)));
    }

    #[test]
    fn the_help_lists_every_option() {
        let h = help();
        for opt in ["-override [0/1]", "-view name", "-h, --help", "--version"] {
            assert!(h.contains(opt), "help lacks {opt}");
        }
        for mode in ["-combine", "-separate", "-convert"] {
            assert!(usage().contains(mode), "usage lacks {mode}");
        }
    }
}
