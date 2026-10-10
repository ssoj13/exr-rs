//! `separate`: every part of a multipart file into a single-part file of its own
//! (`-separate -i in.exr -o base` writes `base.1.exr`, `base.2.exr`, ...).
//!
//! The parts are copied as compressed chunks (`copyPixels`). Deliberate differences from the C++:
//! an output that is the input by any spelling of the path is refused (see `combine`); a part of a
//! type that is not one of the four image types is an error, where the C++ writes its header alone;
//! the input is opened once (the C++ opens it a second time "to check that it exists").

use exr_core::header;
use exr_core::io::write_file;
use exr_core::multipart::{copy_part, MultiPartOutputFile};

use super::names::filename_check;
use super::open;
use crate::cli::tool_io::same_file;
use crate::cli::{CliError, CliResult};

fn fail(msg: impl Into<String>) -> CliError {
    CliError::Other(msg.into())
}

/// What `-separate` prints for a part of this type.
fn label(ty: &str) -> Option<&'static str> {
    match ty {
        header::SCANLINE_IMAGE => Some("scanlineimage"),
        header::TILED_IMAGE => Some("tiledimage"),
        header::DEEP_SCANLINE => Some("deepscanline"),
        header::DEEP_TILE => Some("deeptile"),
        _ => None,
    }
}

/// Run `-separate`; `inputs` must hold exactly one file (the parser guarantees at least one).
pub fn separate(inputs: &[String], out: &str, over: bool) -> CliResult {
    let [in_name] = inputs else {
        return Err(fail("-separate only take one input file"));
    };
    let mut infile = open(in_name)?;
    let n = infile.parts();
    outln!("numOutputs: {n}");

    let names: Vec<String> = (1..=n).map(|p| format!("{out}.{p}.exr")).collect();
    for name in &names {
        outln!("outputfilename: {name}");
    }
    filename_check(names.iter().map(String::as_str), in_name, same_file).map_err(fail)?;

    for (p, name) in (0..n).zip(&names) {
        let header = infile.header(p)?.clone();
        write_file(name, |stream| {
            let mut outf = MultiPartOutputFile::new_with_override(stream, std::slice::from_ref(&header), over)?;
            if let Some(text) = label(header.type_()?) {
                outln!("{text}");
            }
            Ok::<(), CliError>(copy_part(&mut infile, &mut outf, p, 0)?)
        })?;
    }
    outln!("\nSeparate Success");
    Ok(())
}
