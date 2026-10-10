//! `combine`: several files, or chosen parts of them, into one multipart file (`-combine`).
//!
//! Every input argument is `<file>[:<partnum>][::<partname>]` (see `names`). Parts are copied as
//! compressed chunks (`copyPixels`), never decoded, so the pixels of a lossy file are untouched.
//!
//! Deliberate differences from the C++: an output that is one of the inputs (by any spelling of
//! the path, not just equal text) is refused before it can empty a file still being read; a part of
//! a type that is not one of the four image types is an error, where the C++ leaves it without
//! pixels; the output is created once (the C++ opens it twice to "bail early", which produces the
//! same file).

use exr_core::header::{self, Header};
use exr_core::io::write_file;
use exr_core::multipart::{copy_part, MultiPartOutputFile};

use super::names::{filename_check, make_unique_names, parse_filename, set_str};
use super::open;
use crate::cli::tool_io::same_file;
use crate::cli::{CliError, CliResult};

fn fail(msg: impl Into<String>) -> CliError {
    CliError::Other(msg.into())
}

/// What `-combine` prints for a part of this type (the C++ spells the deep scanline one out).
fn label(ty: &str) -> Option<&'static str> {
    match ty {
        header::SCANLINE_IMAGE => Some("scanlineimage"),
        header::TILED_IMAGE => Some("tiledimage"),
        header::DEEP_SCANLINE => Some("deepscanlineimage"),
        header::DEEP_TILE => Some("deeptile"),
        _ => None,
    }
}

/// Run `-combine`. `views[i]` is the view of `inputs[i]` (`-view`), if any.
pub fn combine(inputs: &[String], views: &[Option<String>], out: &str, over: bool) -> CliResult {
    let mut files = Vec::with_capacity(inputs.len());
    let mut names = Vec::with_capacity(inputs.len());
    // Per output part: the input file it comes from and its part number there.
    let mut src: Vec<(usize, i32)> = Vec::new();
    let mut headers: Vec<Header> = Vec::new();

    for (arg, view) in inputs.iter().zip(views) {
        let spec = parse_filename(arg).map_err(fail)?;
        let at = files.len();
        files.push(open(&spec.file)?);
        let parts = files[at].parts();
        let wanted = match spec.part {
            None => 0..parts,
            Some(n) if n >= parts => {
                return Err(fail(format!("You asked for part {n} in {arg}, which only has {parts} parts")));
            }
            Some(n) => n..n + 1,
        };
        for j in wanted {
            let mut h = files[at].header(j)?.clone();
            if !h.has_name() || spec.forced {
                set_str(&mut h, "name", spec.name.as_bytes())?;
            }
            if let Some(v) = view {
                set_str(&mut h, "view", v.as_bytes())?;
            }
            headers.push(h);
            src.push((at, j));
        }
        names.push(spec.file);
    }

    filename_check(names.iter().map(String::as_str), out, same_file).map_err(fail)?;
    if inputs.len() > 1 {
        make_unique_names(&mut headers).map_err(fail)?;
    }

    write_file(out, |stream| {
        let mut outf = MultiPartOutputFile::new_with_override(stream, &headers, over)?;
        for (p, ((at, j), h)) in src.iter().zip(&headers).enumerate() {
            if let Some(text) = label(h.type_()?) {
                outln!("part {p}: {text}");
            }
            copy_part(&mut files[*at], &mut outf, *j, i32::try_from(p).map_err(|_| fail("too many parts"))?)?;
        }
        Ok::<(), CliError>(())
    })?;
    drop(files);
    outln!("\nCombine Success");
    Ok(())
}
