//! `exr-rs convert <in> <out> [--compress <codec>]` — re-encode a single-part,
//! non-deep image.
//!
//! Reads the image as stored through [`exr_core::Image::read_all`] and writes it
//! back with the chosen compression, keeping its storage shape: scanline or tiled
//! (tile size, level mode, rounding), every MIP/RIP level, and subsampled
//! channels at their native sampling. Only the compression changes — like
//! OpenImageIO's `iconvert`, which copies every subimage and MIP level
//! (`src/iconvert/iconvert.cpp`), where reading level 0 alone would silently drop
//! the rest. The facade rejects deep and multipart inputs with a clear
//! [`exr_core::ExrError`]; we surface that as a non-zero exit rather than
//! half-converting. A non-EXR input (`exr-import`: PNG, JPEG, TIFF, DPX, HDR, ...) is decoded to
//! scene-linear, associated-alpha channels and written as a plain scanline EXR.

use std::path::PathBuf;

use clap::builder::{PossibleValuesParser, TypedValueParser};
use exr_core::Image;
use exr_core::attr::Compression;

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs convert in.exr out.exr              # keep the source compression",
    "exr-rs convert in.exr out.exr -c zip       # lossless, zlib",
    "exr-rs convert in.exr out.exr -c dwaa      # lossy, small",
    "exr-rs convert in.exr out.exr -c htj2k256  # HTJ2K (OpenEXR 3.4)",
    "exr-rs convert plate.dpx plate.exr         # any other still: scene-linear, associated alpha",
);

/// `convert` arguments.
#[derive(clap::Args)]
pub struct Args {
    /// Input: an EXR (single-part, not deep), or a PNG/JPEG/TIFF/DPX/HDR/... still (read as scene-linear).
    input: PathBuf,
    /// Output EXR (same storage as the input: tiles, levels and sampling are kept).
    output: PathBuf,
    /// Output compression (default: the source file's).
    #[arg(short, long, value_parser = codec_parser(), ignore_case = true)]
    compress: Option<Compression>,
}

/// The accepted codec names, from [`Compression::ALL`] + [`Compression::name`]: the parser and
/// the `[possible values: ...]` help list come from the same place and cannot drift apart.
pub fn codec_parser() -> impl TypedValueParser<Value = Compression> {
    PossibleValuesParser::new(Compression::ALL.map(Compression::name)).try_map(|name: String| {
        Compression::from_name(&name.to_ascii_lowercase()).ok_or_else(|| format!("unknown codec \"{name}\""))
    })
}

/// Run the `convert` subcommand. Deep / multipart inputs come back from the facade as a clear
/// `ExrError` pointing at the low-level API: a non-zero exit, never a half conversion.
pub fn run(args: Args) -> CliResult {
    // Any other still comes in as scene-linear floats with associated alpha (`exr-import`, the
    // viewer's loader), then leaves as a scanline EXR.
    let image = if exr_import::is_foreign(&args.input) {
        exr_import::load(&args.input).map_err(|e| CliError::Other(e.to_string()))?.image
    } else {
        Image::read_all(&args.input)?
    };
    let compression = args.compress.unwrap_or(image.compression);
    image.write(&args.output, compression)?;
    outln!(
        "converted {} -> {} (compression: {})",
        args.input.display(),
        args.output.display(),
        compression.name(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;

    fn codec(v: &str) -> Result<Option<Compression>, clap::Error> {
        let cli = Cli::try_parse_from(["exr-rs", "convert", "a.exr", "b.exr", "--compress", v])?;
        let crate::cli::Command::Convert(a) = cli.command else { unreachable!() };
        Ok(a.compress)
    }

    /// Every codec name of `Compression::ALL` parses to itself, in any case; the help lists
    /// every one (the original bug: HTJ2K writable but not accepted by the CLI).
    #[test]
    fn every_codec_parses_and_is_listed() {
        use clap::CommandFactory;
        let help = Cli::command().find_subcommand_mut("convert").unwrap().render_help().to_string();
        for c in Compression::ALL {
            assert_eq!(codec(c.name()).unwrap(), Some(c));
            assert_eq!(codec(&c.name().to_ascii_uppercase()).unwrap(), Some(c));
            assert!(help.contains(c.name()), "help lacks {}", c.name());
        }
    }

    /// Converting keeps the storage shape and the pixels of every level: a C++ MIP and RIP golden
    /// and a 2x2-subsampled one, re-encoded with another codec, read back level for level and
    /// channel for channel identical (both codecs lossless).
    #[test]
    fn convert_keeps_tiles_levels_and_sampling() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity");
        let tmp = std::env::temp_dir().join(format!("exr-rs-convert-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        for (name, codec) in [
            ("golden_tiled_zip_mipmap.exr", Compression::Piz),
            ("golden_tiled_zip_ripmap.exr", Compression::NoCompression),
            ("golden_sub_zip_x2y2.exr", Compression::Rle),
        ] {
            let (input, output) = (dir.join(name), tmp.join(name));
            run(Args { input: input.clone(), output: output.clone(), compress: Some(codec) }).unwrap();
            let (a, b) = (Image::read_all(&input).unwrap(), Image::read_all(&output).unwrap());
            assert_eq!(b.compression, codec, "{name}");
            assert_eq!(a.tiling(), b.tiling(), "{name}");
            let (nx, ny) = a.num_levels().unwrap();
            assert_eq!((nx, ny), b.num_levels().unwrap(), "{name}");
            let mut levels = 0;
            for ly in 0..ny {
                for lx in 0..nx {
                    let (la, lb) = (a.level(lx, ly), b.level(lx, ly));
                    assert_eq!(la.is_some(), lb.is_some(), "{name} level ({lx}, {ly})");
                    let (Some(la), Some(lb)) = (la, lb) else { continue };
                    levels += 1;
                    assert_eq!(la.window(), lb.window(), "{name} level ({lx}, {ly})");
                    let chans: Vec<_> = la.channels().map(|(n, d)| (n, la.sampling(n), d)).collect();
                    let chans_b: Vec<_> = lb.channels().map(|(n, d)| (n, lb.sampling(n), d)).collect();
                    assert_eq!(chans, chans_b, "{name} level ({lx}, {ly})");
                }
            }
            assert!(levels > 1 || a.channels().any(|(n, _)| a.sampling(n) != Some((1, 1))), "{name}: nothing to keep");
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// A PNG and a Radiance HDR (both written here by hand / by the `png` crate, not by the
    /// importer) come out as scene-linear EXRs that exr-core reads back: sRGB 128 is 0.2158605
    /// under half alpha 128/255 in linear light, and the HDR's 1.0 stays 1.0.
    #[test]
    fn convert_reads_a_png_and_an_hdr_into_exr() {
        let tmp = std::env::temp_dir().join(format!("exr-rs-convert-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let png = tmp.join("a.png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&png).unwrap()), 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&[128, 128, 128, 128]).unwrap();
        let hdr = tmp.join("a.hdr");
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n".to_vec();
        bytes.extend_from_slice(&[128, 128, 128, 129]); // 0.5 * 2^(129 - 128) = 1.0
        std::fs::write(&hdr, bytes).unwrap();

        let sample = |img: &Image, ch: &str| img.channel(ch).and_then(|c| c.sample_f32(0)).unwrap();
        let (out_png, out_hdr) = (tmp.join("a.exr"), tmp.join("b.exr"));
        run(Args { input: png, output: out_png.clone(), compress: None }).unwrap();
        run(Args { input: hdr, output: out_hdr.clone(), compress: None }).unwrap();
        let (p, h) = (Image::read_all(&out_png).unwrap(), Image::read_all(&out_hdr).unwrap());
        let a = 128.0 / 255.0;
        assert!((sample(&p, "R") - 0.215_860_5 * a).abs() < 2e-4, "{}", sample(&p, "R"));
        assert!((sample(&p, "A") - a).abs() < 2e-4);
        assert_eq!((sample(&h, "R"), sample(&h, "G"), sample(&h, "B")), (1.0, 1.0, 1.0));
        assert!(h.channel("A").is_none());
        std::fs::remove_dir_all(&tmp).unwrap();
    }
    #[test]
    fn an_unknown_codec_is_rejected_with_the_list() {
        let err = codec("bogus").unwrap_err().to_string();
        assert!(err.contains("htj2k256") && err.contains("zip"), "{err}");
    }
}
