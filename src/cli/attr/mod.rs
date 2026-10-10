//! `exr-rs attr` — port of OpenEXR's `exrstdattr` (`openexr/src/bin/exrstdattr/main.cpp`,
//! v3.4.15): copy an image with attributes set, changed or erased in its headers.
//!
//! The pixels are never decoded: every part's compressed chunks are copied through
//! (`copy_pixels`, the C++ `OutputPart::copyPixels`), scanline, tiled and deep alike, so a lossy
//! file loses nothing and the work is one read and one write per chunk. The command line
//! (`opts`) is the tool's own single-dash grammar, see there.
//!
//! Behaviour that differs from the C++, each a `main.cpp` defect:
//! * an unknown `-option`, a non-numeric number and a value that does not fit its field are
//!   errors (`opts`);
//! * a part whose type is none of the four image types is an error, where the C++ silently
//!   writes the part without its pixels (`main.cpp:889-912` has no `else`);
//! * "same file" also catches two spellings of one path (`main.cpp:793` compares the strings, so
//!   `x.exr ./x.exr` would truncate the input);
//! * the missing-output error says "output" (`main.cpp:791` says "input").

mod opts;

use exr_core::header::Header;
use exr_core::io::StdIfStream;
use exr_core::{MultiPartInputFile, MultiPartOutputFile};

use super::tool_io::{print_help, same_file};
use super::{CliError, CliResult, examples};
use opts::{Cmd, Job};

pub const EXAMPLES: &str = examples!(
    "exr-rs attr -owner \"Jane\" -comments \"final\" in.exr out.exr     # standard attributes",
    "exr-rs attr -expTime 0.01 -aperture 8 -focus infinity in.exr out.exr   # camera data",
    "exr-rs attr -part 1 -string note \"fix\" -int take 3 in.exr out.exr    # custom, one part",
    "exr-rs attr -erase comments -erase owner in.exr out.exr            # remove attributes",
    "exr-rs attr -h                                                    # every option",
);

/// `attr` arguments: the tool has its own grammar (`opts`), so clap only hands over the words
/// and shows the generated option reference as the help of this argument.
#[derive(clap::Args)]
#[command(
    disable_help_flag = true,
    allow_hyphen_values = true,
    override_usage = "exr-rs attr [options] <infile> <outfile>"
)]
pub struct Args {
    #[arg(value_name = "ARGS", required = true, trailing_var_arg = true, allow_hyphen_values = true, help = opts::help())]
    args: Vec<String>,
}

/// Run `attr`: parse, then help / version / the edit.
pub fn run(args: Args) -> CliResult {
    match opts::parse(&args.args).map_err(CliError::Other)? {
        Cmd::Help => print_help("attr"),
        Cmd::Version => {
            super::print_version("exrstdattr");
            Ok(())
        }
        Cmd::Edit(job) => edit(&job),
    }
}

/// The header edits, the part copy: the tail of `exrstdattr`'s `main`.
fn edit(job: &Job) -> CliResult {
    if same_file(&job.input, &job.output) {
        return Err(CliError::Other("Input and output cannot be the same file".into()));
    }
    let mut in_stream = StdIfStream::open(&job.input)?;
    let mut input = MultiPartInputFile::new(&mut in_stream)?;
    let parts = input.parts();

    let headers = (0..parts).map(|p| edit_header(job, input.header(p)?, p, parts)).collect::<Result<Vec<_>, _>>()?;

    exr_core::io::write_file(&job.output, |stream| -> exr_core::Result<_> {
        let mut output = MultiPartOutputFile::new(stream, &headers)?;
        for p in 0..parts {
            exr_core::multipart::copy_part(&mut input, &mut output, p, p)?;
        }
        Ok(())
    })?;
    Ok(())
}

/// One part's edited header: the erases first (so an attribute can be re-inserted with another
/// type), then the sets. `parts` validates the part numbers the options selected.
fn edit_header(job: &Job, original: &Header, part: i32, parts: i32) -> Result<Header, CliError> {
    // The attributes a fresh header has are essential; the C++ builds one to know them.
    let essential = Header::new();
    let bad_part = |n: i32| {
        CliError::Other(format!(
            "Invalid part number {n}. Part numbers in file {} go from 0 to {}.",
            job.input,
            parts - 1
        ))
    };
    let mut h = original.clone();
    for e in &job.erases {
        if e.part == -1 || e.part == part {
            if essential.contains(&e.name) {
                return Err(CliError::Other(format!(
                    "Cannot erase attribute {}. It is an essential attribute",
                    e.name
                )));
            }
            h.erase(&e.name)?;
        } else if e.part < 0 || e.part >= parts {
            return Err(bad_part(e.part));
        }
    }
    for s in &job.sets {
        if s.part == -1 || s.part == part {
            h.insert_boxed(s.name.as_str(), s.attr.clone_attr())?;
        } else if s.part < 0 || s.part >= parts {
            return Err(bad_part(s.part));
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// The command definition: no arguments print the help, and every option of the table is
    /// documented in it.
    #[test]
    fn the_help_lists_every_option() {
        let mut root = super::super::Cli::command();
        root.build();
        let help = root.find_subcommand_mut("attr").map(|c| c.render_long_help().to_string()).unwrap_or_default();
        for o in opts::OPTS {
            assert!(help.contains(&format!("-{} {}", o.name, o.args)), "help lacks {}", o.name);
        }
    }
}
