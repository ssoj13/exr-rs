//! `exr-rs aces` — port of OpenEXR's `exr2aces` (`openexr/src/bin/exr2aces/main.cpp`,
//! v3.4.15): read an OpenEXR image as ACES RGB and save it as an ACES image container file.
//!
//! The file rules (scanlines only; R,G,B or Y,RY,BY plus optional A; NONE / PIZ / B44A; ACES
//! chromaticities) are `AcesOutputFile`'s, in `exr_core::rgba`. The tool reads the whole image,
//! maps the compression (NONE stays, B44/B44A become B44A, anything else PIZ) and writes it.
//!
//! * **Verbose.** The C++ `-v` prints `Reading file ..` / `Writing file ..` on stdout. Here `-v` is
//!   the global `exr-rs -v` (it is accepted anywhere on the command line, as the C++ flag is):
//!   the messages appear when the log level is `debug` or finer, i.e. with `-v` (or
//!   `RUST_LOG=debug`), so one flag serves both purposes.
//! * **Arguments.** Exactly two file names; the C++ tool silently lets a third replace the second
//!   (`main.cpp:175-178`), clap rejects it.

use std::path::{Path, PathBuf};

use exr_core::ExrError;
use exr_core::attr::{Compression, LineOrder};
use exr_core::io::StdIfStream;
use exr_core::rgba::{AcesInputFile, AcesOutputFile, Rgba};

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs aces in.exr out.aces      # ACES image container file",
    "exr-rs aces -v in.exr out.aces   # say what is being read and written",
    "exr-rs 2aces in.exr out.aces     # the alias (also `exr2aces`)",
);

/// `aces` arguments. Verbose (`-v`) is the global flag, see the module doc.
#[derive(clap::Args)]
pub struct Args {
    /// Input OpenEXR image (a scanline RGB(A) or luminance-chroma file).
    #[arg(required_unless_present = "version")]
    input: Option<PathBuf>,
    /// Output ACES image container file.
    #[arg(required_unless_present = "version")]
    output: Option<PathBuf>,
    /// Print version information (the text of `exr2aces --version`).
    #[arg(long)]
    version: bool,
}

/// The compression an ACES file gets for an image that had `c` (`exr2aces`, `main.cpp:95-103`).
fn aces_compression(c: Compression) -> Compression {
    match c {
        Compression::NoCompression => c,
        Compression::B44 | Compression::B44a => Compression::B44a,
        _ => Compression::Piz,
    }
}

/// Reverse the row order of a raster `width` pixels wide, in place.
fn flip_rows(pixels: &mut [Rgba], width: usize) {
    let rows = pixels.len() / width;
    for y in 0..rows / 2 {
        let (top, bottom) = pixels.split_at_mut((rows - 1 - y) * width);
        top[y * width..(y + 1) * width].swap_with_slice(&mut bottom[..width]);
    }
}

/// The path as the `&str` the exr-core streams take; a name that is not UTF-8 is an error, not a
/// lossy (and so different) path.
fn utf8(p: &Path) -> Result<&str, CliError> {
    p.to_str().ok_or_else(|| CliError::Other(format!("path is not valid UTF-8: {}", p.display())))
}

/// Run `aces`: the whole conversion.
pub fn run(args: Args, verbose: bool) -> CliResult {
    if args.version {
        super::print_version("exr2aces");
        return Ok(());
    }
    // clap requires both names unless `--version` was given.
    let (Some(input), Some(output)) = (&args.input, &args.output) else {
        return Err(CliError::Other("missing file names".into()));
    };
    let (input, output) = (utf8(input)?, utf8(output)?);

    // Read everything first (the input file is closed before the output is created, so a
    // failing read never leaves an output file behind).
    if verbose {
        outln!("Reading file {input}");
    }
    let (mut header, channels, width, height, pixels) = {
        let mut stream = StdIfStream::open(input)?;
        let mut file = AcesInputFile::new(&mut stream)?;
        let header = file.header()?.clone();
        let window = file.data_window()?;
        // One `i32` extent per axis, as in the C++ (`main.cpp:87-88`), but overflow is an error.
        let extent = |lo: i32, hi: i32| {
            hi.checked_sub(lo)
                .and_then(|d| d.checked_add(1))
                .filter(|&n| n > 0)
                .ok_or_else(|| ExrError::Input(format!("invalid data window [{lo}, {hi}]")))
        };
        let width = extent(window.min.x, window.max.x)?;
        let height = extent(window.min.y, window.max.y)?;
        let count =
            (width as usize).checked_mul(height as usize).ok_or_else(|| ExrError::Input("image too large".into()))?;
        // A fallible allocation: an absurd data window is an error, not an abort.
        let mut pixels: Vec<Rgba> = Vec::new();
        pixels
            .try_reserve_exact(count)
            .map_err(|_| CliError::Other(format!("cannot allocate {count} pixels for {input}")))?;
        pixels.resize(count, Rgba::default());
        file.read_pixels(window.min.y, window.max.y, &mut pixels, 1, width as isize)?;
        (header, file.channels()?, width, height, pixels)
    };

    let compression = aces_compression(header.compression()?);
    *header.compression_mut()? = compression;
    // `write_pixels` takes buffer row i as the i-th line WRITTEN (C++ addresses rows by absolute
    // y), and a DECREASING_Y file writes its top line first: hand it the rows in that order.
    let mut pixels = pixels;
    if header.line_order()? == LineOrder::DecreasingY {
        flip_rows(&mut pixels, width as usize);
    }

    if verbose {
        outln!("Writing file {output}");
    }
    exr_core::io::write_file(output, |stream| -> exr_core::Result<_> {
        let mut file = AcesOutputFile::new(stream, &header, channels)?;
        file.set_frame_buffer(1, width as isize)?;
        file.write_pixels(&pixels, height)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Even and odd row counts, one column and several.
    #[test]
    fn flip_reverses_the_rows() {
        let px = |n: usize| Rgba::from_f32(n as f32, 0.0, 0.0, 1.0);
        for (rows, width) in [(1, 3), (2, 1), (3, 2), (4, 3), (5, 1)] {
            let mut a: Vec<Rgba> = (0..rows * width).map(px).collect();
            flip_rows(&mut a, width);
            for y in 0..rows {
                for x in 0..width {
                    assert_eq!(a[y * width + x], px((rows - 1 - y) * width + x), "{rows}x{width} ({x},{y})");
                }
            }
        }
    }

    /// `main.cpp:95-103`, written out per codec: NONE stays, both B44 flavours become B44A, every
    /// other codec (lossless, DWA, HTJ2K alike) becomes PIZ.
    #[test]
    fn compression_map() {
        use Compression::*;
        for (from, to) in [
            (NoCompression, NoCompression),
            (B44, B44a),
            (B44a, B44a),
            (Rle, Piz),
            (Zips, Piz),
            (Zip, Piz),
            (Piz, Piz),
            (Pxr24, Piz),
            (Dwaa, Piz),
            (Dwab, Piz),
            (Htj2k256, Piz),
            (Htj2k32, Piz),
        ] {
            assert_eq!(aces_compression(from), to, "{}", from.name());
        }
        // Every codec that exists is covered above and lands on one AcesOutputFile accepts.
        for c in Compression::ALL {
            assert!(matches!(aces_compression(c), NoCompression | Piz | B44a), "{}", c.name());
        }
    }
}
