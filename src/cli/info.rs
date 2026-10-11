//! `exr-rs info <file>` — a header-only, exrinfo-like dump.
//!
//! Reads only the file's magic, version word, and part header(s) — never the
//! pixels — so it works uniformly on scanline, tiled, deep, and multipart
//! files. The file-level summary comes from the version flags
//! ([`exr_core::container::version`]); every part's attributes are then printed
//! completely, in `exrheader`'s layout, by [`exr_core::api::describe`].

use exr_core::api::describe;
use exr_core::container::version;
use exr_core::header::Header;
use exr_core::io::{IStream, StdIfStream};
use exr_core::{ExrError, MultiPartInputFile};

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs info shot.exr              # file kind, then every attribute of every part",
    "exr-rs info a.exr b.exr c.exr     # several files, one after another",
    "exr-rs info plate.dpx              # another format: a short summary (format, size, channels)",
    "exr-rs header shot.exr            # the same (alias; also `exrheader`)",
);

/// `info` arguments.
#[derive(clap::Args)]
pub struct Args {
    /// EXR files to describe (headers only; pixels are never read).
    #[arg(required = true)]
    files: Vec<String>,
}

/// Run the `info` subcommand: every file in turn, a blank line between them; the first error
/// stops the run (as `exrheader` does).
pub fn run(args: Args) -> CliResult {
    for (i, path) in args.files.iter().enumerate() {
        if i > 0 {
            outln!();
        }
        if exr_import::is_foreign(std::path::Path::new(path)) {
            foreign(path)?;
            continue;
        }
        let mut stream = StdIfStream::open(path)?;
        dump(path, &mut stream)?;
    }
    Ok(())
}

/// A non-EXR still: format, size and channel count from its header, no pixels decoded. It has no
/// OpenEXR header to dump, so this is a summary; `convert` brings it into EXR.
fn foreign(path: &str) -> CliResult {
    let rows = exr_import::probe(std::path::Path::new(path)).map_err(|e| CliError::Other(e.to_string()))?;
    outln!("file: {path}");
    for (k, v) in rows {
        outln!("  {k}: {v}");
    }
    Ok(())
}

/// Print the full header dump for the file behind `stream`.
///
/// Reads the magic + version word once, prints the file-level line, then
/// dispatches to the multipart or single-part header enumeration. The stream
/// is rewound between the peek and the real header read so each reader starts
/// from byte 0.
fn dump(path: &str, stream: &mut StdIfStream) -> CliResult {
    // Peek magic + version word to classify the file before choosing a reader.
    stream.seekg(0)?;
    let magic = exr_core::read_i32(stream)?;
    if magic as u32 != version::MAGIC {
        return Err(CliError::Exr(ExrError::Input(format!(
            "{path} is not an OpenEXR file (bad magic number)."
        ))));
    }
    let version_word = exr_core::read_i32(stream)? as u32;
    let flags = version::get_flags(version_word);

    if !version::supports_flags(flags) {
        return Err(CliError::Exr(ExrError::Input(
            "version flags field contains unknown flags.".into(),
        )));
    }

    let multipart = version::is_multi_part(flags);
    let deep = version::is_non_image(flags);
    let tiled = version::is_tiled(flags);
    let long_names = flags & version::LONG_NAMES_FLAG != 0;

    outln!("file: {path}");
    outln!("  magic:   OK (0x{:08X})", version::MAGIC);
    outln!(
        "  version: {} (flags: 0x{:06X})",
        version::get_version(version_word),
        flags >> 8
    );
    outln!(
        "  type:    {}",
        file_kind(multipart, deep, tiled)
    );
    outln!(
        "  flags:   tiled={} deep={} multipart={} longNames={}",
        yes_no(tiled),
        yes_no(deep),
        yes_no(multipart),
        yes_no(long_names),
    );

    // Rewind for the real header read(s).
    stream.seekg(0)?;

    if multipart {
        // MultiPartInputFile reads every part header + chunk-offset table.
        let mp = MultiPartInputFile::new(stream)?;
        let n = mp.parts();
        outln!("  parts:   {n}");
        for i in 0..n {
            let header = mp.header(i)?;
            outln!();
            outln!("  part {i}:");
            print_header(header);
        }
    } else {
        // Single-part: one header follows the version word.  `read_checked`
        // (not `read_from`) so a crafted header is rejected before we compute
        // with its values — C++ validates every part at open time
        // (`parse_header.c:2813` -> `validation.c:697`).
        let (header, _) = Header::read_checked(stream)?;
        outln!("  parts:   1");
        outln!();
        print_header(&header);
    }

    Ok(())
}

/// Human label for the file as a whole, from the version flags.
fn file_kind(multipart: bool, deep: bool, tiled: bool) -> &'static str {
    match (multipart, deep, tiled) {
        (true, _, _) => "multipart",
        (false, true, true) => "deep tiled",
        (false, true, false) => "deep scanline",
        (false, false, true) => "single-part tiled",
        (false, false, false) => "single-part scanline",
    }
}

/// Print one part's header: every attribute, in `exrheader`'s layout
/// (`name (type t): value`, block values on indented lines) — the same text the
/// viewer shows, from the one formatter, [`exr_core::api::describe`].
///
/// Nothing is derived or hand-picked here: `name`, `type`, `compression`,
/// `tiles` ... are attributes like any other, so they appear exactly when the
/// file carries them.
fn print_header(header: &Header) {
    for row in describe::rows(header) {
        outln!("{}", row.line());
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-EXR still gets a summary from its header; a file the decoder refuses is an error,
    /// not a silent empty dump.
    #[test]
    fn a_foreign_still_is_summarised_and_garbage_is_an_error() {
        let tmp = std::env::temp_dir().join(format!("exr-rs-info-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let png = tmp.join("a.png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&png).unwrap()), 3, 2);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&[0; 18]).unwrap();
        foreign(png.to_str().unwrap()).unwrap();
        let rows = exr_import::probe(&png).unwrap();
        assert!(rows.contains(&("size".to_owned(), "3x2".to_owned())) && rows.contains(&("channels".to_owned(), "3".to_owned())), "{rows:?}");
        let bad = tmp.join("b.png");
        std::fs::write(&bad, b"nope").unwrap();
        assert!(foreign(bad.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}