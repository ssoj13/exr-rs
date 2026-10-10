//! `exr-rs manifest` — port of OpenEXR's `exrmanifest`
//! (`openexr/src/bin/exrmanifest/main.cpp`, v3.4.15): print the ID manifest embedded in the
//! header of every part of every file, as plain text.
//!
//! Headers only: the pixels are never read. The text is byte-identical to the C++ tool's (its
//! `dumpManifest` layout: a table per channel group, columns padded to the widest cell, then the
//! raw / uncompressed / compressed sizes).
//!
//! Deliberate differences from the C++ tool, each an improvement on a `main.cpp` defect:
//! * `--help` is recognised at any position (`main.cpp:179` compares `argv[1]` inside the loop, so
//!   `--help` worked only as the first argument);
//! * no arguments print the help and exit 0, like every `exr-rs` command (the C++ prints a usage
//!   line to stderr and exits with -1);
//! * a table row with more text cells than the group has components widens the column list
//!   instead of indexing past its end (`main.cpp:94`, undefined behaviour in C++).

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use exr_core::io::StdIfStream;
use exr_core::{CompressedIDManifest, IDManifest, IdLifetime, MultiPartInputFile};

use super::stdio::Out;
use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs manifest cryptomatte.exr           # the ID manifest of every part",
    "exr-rs manifest a.exr b.exr               # several files, one after another",
    "exr-rs manifest shot.exr > manifest.txt   # keep the text",
);

/// `manifest` arguments.
#[derive(clap::Args)]
pub struct Args {
    /// EXR files whose ID manifests to print (headers only; pixels are never read).
    #[arg(required_unless_present = "version", value_name = "FILE")]
    files: Vec<PathBuf>,
    /// Print version information (the text of `exrmanifest --version`).
    #[arg(long)]
    version: bool,
}

/// Print the manifest of every file in turn; the first error stops the run (what was already
/// printed stays, as in the C++ tool).
pub fn run(args: Args) -> CliResult {
    if args.version {
        super::print_version("exrmanifest");
        return Ok(());
    }
    let mut out = BufWriter::new(Out);
    let result = args.files.iter().try_for_each(|path| print(path, &mut out));
    // The flush error only matters when nothing else failed first.
    result.and_then(|()| out.flush().map_err(io_err))
}

/// `printManifest`: one file, every part.
fn print(path: &Path, out: &mut impl Write) -> CliResult {
    let name = path.to_str().ok_or_else(|| CliError::Other(format!("path is not valid UTF-8: {}", path.display())))?;
    let mut stream = StdIfStream::open(name)?;
    let file = MultiPartInputFile::new(&mut stream)?;
    for part in 0..file.parts() {
        if file.parts() > 1 {
            writeln!(out, "{} part {part}:", path.display()).map_err(io_err)?;
        }
        // `hasIDManifest`: an `idManifest` attribute of exactly this type.
        let attr = file.header(part)?.find_typed_attribute::<CompressedIDManifest>("idManifest");
        let Some(attr) = attr else {
            out.write_all(b"no manifest found\n").map_err(io_err)?;
            continue;
        };
        let packed = &attr.value;
        let size = dump(&IDManifest::from_compressed(packed)?, out).map_err(io_err)?;
        write!(
            out,
            "raw text size    : {size}\nuncompressed size: {}\ncompressed size  : {}\n",
            packed.uncompressed_data_size,
            packed.compressed_data_size(),
        )
        .map_err(io_err)?;
    }
    Ok(())
}

/// `dumpManifest`: the tables of `m`; returns the "raw text size" the C++ tool reports (the bytes
/// the strings and ids would take as text, one separator each).
fn dump(m: &IDManifest, out: &mut impl Write) -> io::Result<usize> {
    let mut size = 0;
    for (i, group) in m.iter().enumerate() {
        if i > 0 {
            out.write_all(b"\n\n")?;
        }
        out.write_all(b" channels  : ")?;
        for (n, channel) in group.channels().iter().enumerate() {
            if n > 0 {
                out.write_all(b",")?;
            }
            out.write_all(channel.as_bytes())?;
            size += channel.len() + 1;
        }
        writeln!(out, "\n hashScheme: {}", group.hash_scheme())?;
        writeln!(out, " encoding  : {}", group.encoding_scheme())?;
        let lifetime = match group.lifetime() {
            IdLifetime::Frame => "frame",
            IdLifetime::Shot => "shot",
            IdLifetime::Stable => "stable",
        };
        writeln!(out, " lifetime  : {lifetime}")?;

        // Column widths: the widest of the component name and every cell below it.
        let components = group.components();
        let mut width: Vec<usize> = components.iter().map(String::len).collect();
        size += components.iter().map(|c| c.len() + 1).sum::<usize>();
        let mut id_width = 0;
        for (id, text) in group.iter() {
            let id_len = id.to_string().len();
            size += id_len;
            id_width = id_width.max(id_len);
            if text.len() > width.len() {
                width.resize(text.len(), 0);
            }
            for (w, cell) in width.iter_mut().zip(text) {
                size += cell.len() + 1;
                *w = (*w).max(cell.len());
            }
        }

        // Header row, then one row per id; every cell is followed by padding to its column plus one.
        write_row(out, "", id_width, components.iter().map(String::as_str), &width)?;
        for (id, text) in group.iter() {
            write_row(out, &id.to_string(), id_width, text.iter().map(String::as_str), &width)?;
        }
    }
    Ok(size)
}

/// One table row: 5 spaces, the id column, then the cells (each padded to its column width + 1).
/// Padding counts BYTES like the C++ `string::size`, not characters (`{:<w}` would), so non-ASCII
/// cells line up exactly as in the original.
fn write_row<'a>(
    out: &mut impl Write,
    id: &str,
    id_width: usize,
    cells: impl Iterator<Item = &'a str>,
    width: &[usize],
) -> io::Result<()> {
    out.write_all(b"     ")?;
    write_cell(out, id, id_width)?;
    for (cell, &w) in cells.zip(width) {
        write_cell(out, cell, w)?;
    }
    out.write_all(
        b"
",
    )
}

/// `cell` followed by spaces up to `width + 1` bytes.
fn write_cell(out: &mut impl Write, cell: &str, width: usize) -> io::Result<()> {
    out.write_all(cell.as_bytes())?;
    write!(out, "{:1$}", "", (width + 1).saturating_sub(cell.len()))
}

/// stdout failures (a closed pipe) are reported like any other error.
fn io_err(e: io::Error) -> CliError {
    CliError::Other(format!("write failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr_core::{ID_SCHEME, MURMURHASH3_32};

    /// Two groups: ids of unequal length, three cells of unequal width, a component wider than
    /// its cells; the sizes follow the C++ accounting (`main.cpp:53,79,88,93`).
    #[test]
    fn dump_matches_the_cpp_layout() {
        let mut m = IDManifest::new();
        let g = m.add_channel("crypto_a");
        g.set_component("name").unwrap();
        g.set_hash_scheme(MURMURHASH3_32);
        g.set_encoding_scheme(ID_SCHEME);
        g.set_lifetime(IdLifetime::Stable);
        g.insert_single(7, "x").unwrap();
        g.insert_single(1234, "hero").unwrap();
        let g = m.add_channel("crypto_b");
        g.channels_mut().insert("crypto_c".to_string());
        g.set_components(vec!["a".into(), "bb".into()]).unwrap();
        g.set_lifetime(IdLifetime::Shot);
        g.insert(3, vec!["long cell".into(), "y".into()]).unwrap();

        let mut out = Vec::new();
        let size = dump(&m, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let expected = concat!(
            " channels  : crypto_a\n",
            " hashScheme: MurmurHash3_32\n",
            " encoding  : id\n",
            " lifetime  : stable\n",
            "          name \n",
            "     7    x    \n",
            "     1234 hero \n",
            "\n\n",
            " channels  : crypto_b,crypto_c\n",
            " hashScheme: unknown\n",
            " encoding  : unknown\n",
            " lifetime  : shot\n",
            "       a         bb \n",
            "     3 long cell y  \n",
        );
        assert_eq!(text, expected);
        // group 1: channel 9 + component 5 + ids 1+4 + cells 2+5; group 2: channels 9+9,
        // components 2+3, id 1, cells 10+2.
        assert_eq!(size, (9 + 5 + 1 + 4 + 2 + 5) + (9 + 9 + 2 + 3 + 1 + 10 + 2));
    }

    /// A group with no entries prints the header row only, and a manifest of no groups prints
    /// nothing (just the size lines around it).
    #[test]
    fn empty_tables_print_only_headers() {
        let mut m = IDManifest::new();
        m.add_channel("c").set_component("name").unwrap();
        let mut out = Vec::new();
        assert_eq!(dump(&m, &mut out).unwrap(), 2 + 5);
        assert!(String::from_utf8(out).unwrap().ends_with(" lifetime  : stable\n      name \n"));

        let mut out = Vec::new();
        assert_eq!(dump(&IDManifest::new(), &mut out).unwrap(), 0);
        assert!(out.is_empty());
    }

    /// Column padding counts bytes (C++ `string::size`), so a 2-byte e-acute gets one space less
    /// than a 2-character ASCII cell would (`{:<w}` formatting counts chars and gets this wrong).
    #[test]
    fn padding_counts_bytes() {
        let mut m = IDManifest::new();
        let g = m.add_channel("c");
        g.set_component("n").unwrap();
        g.insert_single(1, "\u{e9}").unwrap();
        g.insert_single(2, "abc").unwrap();
        let mut out = Vec::new();
        dump(&m, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.ends_with("     1 \u{e9}  \n     2 abc \n"), "{text:?}");
    }

    /// A row with more cells than components must not panic (undefined behaviour in the C++).
    #[test]
    fn a_wide_row_widens_the_columns() {
        let mut m = IDManifest::new();
        let g = m.add_channel("c");
        g.set_component("a").unwrap();
        // `insert` refuses a mismatched row, so widen through the raw entry accessor.
        g.entry(1).extend(["p".to_string(), "qq".to_string()]);
        let mut out = Vec::new();
        dump(&m, &mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().ends_with("     1 p qq \n"));
    }
}
