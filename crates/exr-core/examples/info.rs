//! Print a header-only summary of any EXR file — no pixels decoded — so it
//! works on every file kind (scanline, tiled, deep, multipart).
//!
//! Run:
//! ```text
//! cargo run -p exr-core --example info -- path/to/image.exr
//! ```
//!
//! This is a minimal illustration of inspecting a file through the low-level
//! [`exr_core::container::version`] flags + [`exr_core::header::Header`]. The
//! full-featured inspector is the workspace CLI: `exr-rs info <file>` (richer
//! per-part output, multipart enumeration, tile/deep labels).

use std::process::ExitCode;

use exr_core::container::version;
use exr_core::header::Header;
use exr_core::io::{IStream, StdIfStream};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: info <file.exr>");
        return ExitCode::FAILURE;
    };

    match dump(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Peek the magic + version word to classify the file, then print the
/// single-part header (multipart enumeration is left to the `exr-rs info` CLI).
fn dump(path: &str) -> exr_core::Result<()> {
    let mut stream = StdIfStream::open(path)?;

    let magic = exr_core::read_i32(&mut stream)?;
    if magic as u32 != version::MAGIC {
        return Err(exr_core::ExrError::Input(format!(
            "{path} is not an OpenEXR file (bad magic number)."
        )));
    }
    let version_word = exr_core::read_i32(&mut stream)? as u32;
    let flags = version::get_flags(version_word);

    println!("file:      {path}");
    println!(
        "version:   {} (tiled={}, deep={}, multipart={})",
        version::get_version(version_word),
        version::is_tiled(flags),
        version::is_non_image(flags),
        version::is_multi_part(flags),
    );

    if version::is_multi_part(flags) {
        println!("note:      multipart — use `exr-rs info {path}` for per-part details.");
        return Ok(());
    }

    // Single-part: one header follows the version word. Rewind to byte 0 so the
    // header reader sees magic + version itself.
    stream.seekg(0)?;
    let (header, _) = Header::read_from(&mut stream)?;

    let dw = header.data_window()?;
    println!(
        "size:      {} x {}",
        dw.max.x - dw.min.x + 1,
        dw.max.y - dw.min.y + 1
    );
    println!("compression: {:?}", header.compression()?);
    println!("channels:");
    for (name, ch) in header.channels()?.iter() {
        println!("  {name} : {:?}", ch.type_);
    }
    Ok(())
}
