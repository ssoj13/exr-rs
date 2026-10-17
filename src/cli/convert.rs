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
//! half-converting. A non-EXR input (`exr-import`: PNG, JPEG, TIFF, DPX, HDR, ...) is written as a
//! plain scanline EXR the way `oiiotool in.png -o out.exr` writes it: its values as read, alpha
//! associated, its colour tag as `colorInteropID` (`exr_import::Loaded::to_exr`).

use std::path::PathBuf;

use clap::builder::{PossibleValuesParser, TypedValueParser};
use exr_core::Image;
use exr_core::attr::Compression;

use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs convert in.exr out.exr                        # keep the source compression",
    "exr-rs convert in.exr out.exr -c zip                 # lossless, zlib",
    "exr-rs convert in.exr out.exr -c dwaa                # lossy, small",
    "exr-rs convert in.exr out.exr -c htj2k256            # HTJ2K (OpenEXR 3.4)",
    "exr-rs convert in.exr out.exv                        # EXV video (default: bc6h-lossless)",
    "exr-rs convert in.exr out.exv --exv-codec bc6h-lossy # EXV hardware GPU BC6H",
    "exr-rs convert clip.exv frame.####.exr -c zip        # unpack EXV clip to EXR sequence",
    "exr-rs convert plate.dpx plate.exr                   # any other still: values as read, colorInteropID tag",
);

/// Accepted `--exv-codec` profile names.
pub const EXV_CODECS: [&str; 7] = [
    "bc6h-lossless",
    "bc6h",
    "bc6h-lossy",
    "htj2k",
    "fast",
    "wave",
    "rt",
];

/// `convert` arguments.
#[derive(clap::Args)]
pub struct Args {
    /// Input: an EXR (single-part, not deep), an EXV clip, or a PNG/JPEG/TIFF/DPX/HDR/... still.
    pub input: PathBuf,
    /// Output: an EXR (`out.exr` or `out.####.exr` for multi-frame EXV clips) or an EXV (`out.exv`).
    pub output: PathBuf,
    /// Output OpenEXR compression (default: the source file's).
    #[arg(short, long, value_parser = codec_parser(), ignore_case = true)]
    pub compress: Option<Compression>,
    /// Output EXV codec when writing `.exv` (default: `bc6h-lossless`).
    #[arg(long = "exv-codec", value_parser = PossibleValuesParser::new(EXV_CODECS), ignore_case = true)]
    pub exv_codec: Option<String>,
    /// GOP length when writing `.exv` (`1` = all-intra, `> 1` = temporal P-frames).
    #[arg(long)]
    pub gop: Option<u32>,
}

/// The accepted codec names, from [`Compression::ALL`] + [`Compression::name`]: the parser and
/// the `[possible values: ...]` help list come from the same place and cannot drift apart.
pub fn codec_parser() -> impl TypedValueParser<Value = Compression> {
    PossibleValuesParser::new(Compression::ALL.map(Compression::name)).try_map(|name: String| {
        Compression::from_name(&name.to_ascii_lowercase()).ok_or_else(|| format!("unknown codec \"{name}\""))
    })
}

/// Format a numbered frame path when `pattern` contains `####` in its file name.
fn expand_frame_pattern(pattern: &std::path::Path, frame: usize) -> Option<PathBuf> {
    let name = pattern.file_name()?.to_str()?;
    let start = name.find('#')?;
    let end = start + name[start..].bytes().take_while(|&b| b == b'#').count();
    let width = end - start;
    let formatted = format!("{}{}{}", &name[..start], format_args!("{frame:0width$}"), &name[end..]);
    Some(pattern.with_file_name(formatted))
}

/// Discover all numbered frame files on disk matching a `name.####.ext` pattern, sorted by frame number.
fn collect_pattern_frames(pattern: &std::path::Path) -> Result<Option<Vec<PathBuf>>, CliError> {
    let Some(name) = pattern.file_name().and_then(|n| n.to_str()) else {
        return Ok(None);
    };
    let Some(start) = name.find('#') else {
        return Ok(None);
    };
    let end = start + name[start..].bytes().take_while(|&b| b == b'#').count();
    let prefix = &name[..start];
    let suffix = &name[end..];
    let dir = pattern
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut matched: Vec<(u64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| CliError::Other(e.to_string()))? {
        let entry = entry.map_err(|e| CliError::Other(e.to_string()))?;
        let fname_os = entry.file_name();
        let Some(fname) = fname_os.to_str() else { continue };
        if fname.len() > prefix.len() + suffix.len()
            && fname.starts_with(prefix)
            && fname.ends_with(suffix)
        {
            let mid = &fname[prefix.len()..fname.len() - suffix.len()];
            if let Ok(num) = mid.parse::<u64>() {
                matched.push((num, entry.path()));
            }
        }
    }
    matched.sort_by_key(|(n, _)| *n);
    if matched.is_empty() {
        return Err(CliError::Other(format!(
            "no frames found matching pattern {}",
            pattern.display()
        )));
    }
    Ok(Some(matched.into_iter().map(|(_, p)| p).collect()))
}

/// Run the `convert` subcommand. Deep / multipart inputs come back from the facade as a clear
/// `ExrError` pointing at the low-level API: a non-zero exit, never a half conversion.
pub fn run(args: Args) -> CliResult {
    let other = |e: exr_import::ImportError| CliError::Other(e.to_string());

    if exr_import::is_exv(&args.output) {
        let codec_name = args.exv_codec.as_deref().unwrap_or("bc6h-lossless");
        let gop = args.gop.unwrap_or(1);
        let profile = exr_import::parse_exv_profile(codec_name).map_err(other)?;
        let load_one = |p: &std::path::Path| -> Result<Image, CliError> {
            if exr_import::is_foreign(p) {
                exr_import::load(p).map_err(other)?.to_exr().map_err(other)
            } else {
                Ok(Image::read_all(p)?)
            }
        };
        let frames = if let Some(paths) = collect_pattern_frames(&args.input)? {
            paths.iter().map(|p| load_one(p)).collect::<Result<Vec<_>, _>>()?
        } else {
            vec![load_one(&args.input)?]
        };
        exr_import::write_exr_frames_as_exv(&args.output, &frames, profile, gop).map_err(other)?;
        outln!(
            "converted {} ({} frame(s)) -> {} (exv codec: {})",
            args.input.display(),
            frames.len(),
            args.output.display(),
            codec_name,
        );
        return Ok(());
    }

    if (exr_import::is_exv(&args.input) || exr_import::has_exv_magic(&args.input))
        && expand_frame_pattern(&args.output, 0).is_some()
    {
        let clip = exr_import::ExvClip::open(&args.input, 64 << 20).map_err(other)?;
        let count = clip.frame_count();
        for f in 0..count {
            let out_path = expand_frame_pattern(&args.output, f)
                .ok_or_else(|| CliError::Other("invalid output frame pattern".to_owned()))?;
            let image = clip.load_frame(f).map_err(other)?.to_exr().map_err(other)?;
            let compression = args.compress.unwrap_or(image.compression);
            image.write(&out_path, compression)?;
        }
        outln!(
            "converted {} ({count} frames) -> {}",
            args.input.display(),
            args.output.display(),
        );
        return Ok(());
    }

    // Any other still comes in through the viewer's loader (`exr-import`) and leaves as a
    // scanline EXR holding its values and its colour tag.
    let image = if exr_import::is_foreign(&args.input) {
        exr_import::load(&args.input).map_err(other)?.to_exr().map_err(other)?
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
            run(Args {
                input: input.clone(),
                output: output.clone(),
                compress: Some(codec),
                exv_codec: None,
                gop: None,
            })
            .unwrap();
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
    /// importer) come out as EXRs that exr-core reads back with their values and tags, as
    /// oiiotool writes them: the PNG's code 128 associated with alpha 128 (128/255 squared) and
    /// tagged `srgb_rec709_scene`, the HDR's 1.0 stays 1.0, tagged `lin_rec709_scene`.
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
        run(Args { input: png, output: out_png.clone(), compress: None, exv_codec: None, gop: None }).unwrap();
        run(Args { input: hdr, output: out_hdr.clone(), compress: None, exv_codec: None, gop: None }).unwrap();
        let (p, h) = (Image::read_all(&out_png).unwrap(), Image::read_all(&out_hdr).unwrap());
        let a = 128.0 / 255.0;
        assert!((sample(&p, "R") - a * a).abs() < 2e-4, "{}", sample(&p, "R"));
        assert!((sample(&p, "A") - a).abs() < 2e-4);
        assert_eq!((sample(&h, "R"), sample(&h, "G"), sample(&h, "B")), (1.0, 1.0, 1.0));
        assert!(h.channel("A").is_none());
        let id = |img: &Image| {
            let attr = img.attributes().get("colorInteropID").and_then(|a| a.as_any().downcast_ref::<exr_core::attr::StringAttribute>());
            attr.map(|a| String::from_utf8_lossy(a.value.as_bytes()).into_owned())
        };
        assert_eq!(id(&p).as_deref(), Some("srgb_rec709_scene"));
        assert_eq!(id(&h).as_deref(), Some("lin_rec709_scene"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn convert_roundtrips_exr_through_exv_bc6h_lossless() {
        use exr_core::api::ChannelData;
        use imath_rs::{Box2i, Half, V2i};

        let tmp = std::env::temp_dir().join(format!("exr-rs-convert-exv-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let make_frame = |seed: usize| {
            let mut img = Image::new(Box2i::from_min_max(V2i::new(0, 0), V2i::new(7, 7)));
            let r: Vec<Half> = (0..64).map(|i| Half::from_f32((i + seed) as f32 * 0.125)).collect();
            let g: Vec<Half> = (0..64).map(|i| Half::from_f32((i + seed) as f32 * -0.05)).collect();
            let b: Vec<Half> = (0..64).map(|i| Half::from_f32((i + seed) as f32 * 0.25 + 1.0)).collect();
            img.insert_channel("R", ChannelData::Half(r)).unwrap();
            img.insert_channel("G", ChannelData::Half(g)).unwrap();
            img.insert_channel("B", ChannelData::Half(b)).unwrap();
            img
        };
        let f0 = tmp.join("in.0000.exr");
        let f1 = tmp.join("in.0001.exr");
        make_frame(0).write(&f0, Compression::Zip).unwrap();
        make_frame(7).write(&f1, Compression::Zip).unwrap();

        let exv = tmp.join("clip.exv");
        run(Args {
            input: tmp.join("in.####.exr"),
            output: exv.clone(),
            compress: None,
            exv_codec: Some("bc6h-lossless".to_owned()),
            gop: Some(2),
        })
        .unwrap();
        run(Args {
            input: exv,
            output: tmp.join("out.####.exr"),
            compress: Some(Compression::Zip),
            exv_codec: None,
            gop: None,
        })
        .unwrap();

        for (src, dst) in [(&f0, tmp.join("out.0000.exr")), (&f1, tmp.join("out.0001.exr"))] {
            let (a, b) = (Image::read_all(src).unwrap(), Image::read_all(&dst).unwrap());
            assert_eq!(a.level_window(0, 0).unwrap(), b.level_window(0, 0).unwrap());
            for (name, chan_a) in a.channels() {
                let chan_b = b.channel(name).expect(name);
                assert_eq!(chan_a, chan_b, "channel {name} mismatch after EXR -> EXV -> EXR");
            }
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn an_unknown_codec_is_rejected_with_the_list() {
        let err = codec("bogus").unwrap_err().to_string();
        assert!(err.contains("htj2k256") && err.contains("zip"), "{err}");
    }
}
