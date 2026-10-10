//! `exr-rs cryptomatte` — inspect / extract / encode / inject Cryptomatte EXRs.
//!
//! Thin CLI over the `cryptomatte` crate (SSOT).

use std::fs;
use std::path::Path;

use cryptomatte::{
    discover_path, from_named_mattes, load_manifest, CryptomatteFile, EncodeOptions,
    ManifestPlacement, MergeOp, NamedMatte, Windows,
};
use exr_core::attr::Compression;
use exr_core::{ChannelData, Image};

use super::convert::codec_parser;
use super::tool_io::same_file;
use super::{CliError, CliResult, examples};

pub const EXAMPLES: &str = examples!(
    "exr-rs c info shot.exr                                   # types, ranks, manifest names",
    "exr-rs c extract shot.exr -n hero                        # coverage stats of one object",
    "exr-rs c extract shot.exr -n hero -n cape -o matte.exr   # write the combined matte",
    "exr-rs c encode out.exr -c CryptoObject -m hero=hero.exr -m cape=cape.exr",
    "exr-rs c inject shot.exr out.exr -n prop -m prop.exr     # add an object to a type",
);

/// `cryptomatte` arguments: one of its subcommands.
#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// List the Cryptomatte types of every part: ranked layers, hash, manifest names.
    #[command(visible_alias = "i", arg_required_else_help = true, after_help = INFO_EXAMPLES)]
    Info {
        /// The EXR to inspect (headers only).
        file: String,
    },
    /// Sum the coverage of named objects: print stats, or write the matte.
    #[command(visible_alias = "e", arg_required_else_help = true, after_help = EXTRACT_EXAMPLES)]
    Extract {
        /// The Cryptomatte EXR.
        file: String,
        /// Object name (repeat for several; their coverage is summed).
        #[arg(short, long = "name", required = true)]
        names: Vec<String>,
        /// Cryptomatte type (default: the only one, else an error listing them).
        #[arg(short, long)]
        layer: Option<String>,
        /// Write the matte: `.raw` as little-endian f32 samples, anything else as a
        /// one-channel FLOAT EXR with the source windows.
        #[arg(short, long)]
        out: Option<String>,
    },
    /// Build a Cryptomatte EXR from named single-channel coverage EXRs.
    #[command(visible_alias = "n", visible_alias = "enc", arg_required_else_help = true, after_help = ENCODE_EXAMPLES)]
    Encode {
        /// The EXR to write.
        out: String,
        /// Cryptomatte type name (e.g. CryptoObject).
        #[arg(short, long)]
        crypto: String,
        /// `name=coverage.exr`: the coverage is the file's only channel, else R, else Y. All
        /// mattes must share one data window. Repeat for each object.
        #[arg(short, long = "matte", required = true, value_parser = parse_matte_arg)]
        mattes: Vec<(String, String)>,
        /// Ranked layers (each holds two id/coverage pairs).
        #[arg(short, long, default_value_t = cryptomatte::DEFAULT_RANKED_LAYERS)]
        ranked: usize,
        /// Write the manifest to this relative `.json` (forward slashes) instead of inline.
        #[arg(short, long)]
        sidecar: Option<String>,
    },
    /// Add a named coverage to a Cryptomatte type of an existing EXR (other parts and channels stay).
    #[command(visible_alias = "j", visible_alias = "inj", arg_required_else_help = true, after_help = INJECT_EXAMPLES)]
    Inject {
        /// The Cryptomatte EXR to read (single-part or multipart).
        input: String,
        /// The EXR to write; not the input.
        output: String,
        /// Object name to add (hashed like every Cryptomatte name).
        #[arg(short, long)]
        name: String,
        /// Coverage EXR: its only channel, else R, else Y; same data window as the input part.
        #[arg(short, long)]
        mask: String,
        /// Cryptomatte type (default: the first, by name, as `extract`).
        #[arg(short, long)]
        layer: Option<String>,
        /// Composite the new coverage over or under the existing ids.
        #[arg(short, long, value_enum, default_value_t = Op::Over)]
        op: Op,
        /// Compression of the written part (default: the source's).
        #[arg(short, long, value_parser = codec_parser(), ignore_case = true)]
        compress: Option<Compression>,
    },
}

/// `--op`: the library's [`MergeOp`] as a CLI value.
#[derive(Clone, Copy, clap::ValueEnum)]
enum Op {
    Over,
    Under,
}

impl From<Op> for MergeOp {
    fn from(op: Op) -> Self {
        match op {
            Op::Over => MergeOp::Over,
            Op::Under => MergeOp::Under,
        }
    }
}

const INFO_EXAMPLES: &str = examples!(
    "exr-rs c info shot.exr     # every type in every part",
    "exr-rs c i shot.exr        # the same",
);
const EXTRACT_EXAMPLES: &str = examples!(
    "exr-rs c extract shot.exr -n hero                         # nonzero count, sum, min, max",
    "exr-rs c extract shot.exr -n hero -l CryptoMaterial       # from one type",
    "exr-rs c extract shot.exr -n hero -n cape -o matte.exr    # write the combined matte",
    "exr-rs c extract shot.exr -n hero -o matte.raw            # raw f32 samples",
);
const INJECT_EXAMPLES: &str = examples!(
    "exr-rs c inject shot.exr out.exr -n prop -m prop.exr                 # over, in the first type",
    "exr-rs c inject shot.exr out.exr -n prop -m prop.exr -l CryptoMaterial",
    "exr-rs c inject shot.exr out.exr -n sky -m sky.exr -o under          # behind what is there",
    "exr-rs c inject shot.exr out.exr -n prop -m prop.exr -c zip          # recompress the part",
);
const ENCODE_EXAMPLES: &str = examples!(
    "exr-rs c encode out.exr -c CryptoObject -m hero=hero.exr -m cape=cape.exr",
    "exr-rs c encode out.exr -c CryptoObject -m hero=hero.exr -r 6       # six ranked layers",
    "exr-rs c encode out.exr -c CryptoObject -m hero=hero.exr -s manifest.json",
);

/// Run the `cryptomatte` / `c` subcommand.
pub fn run(args: Args) -> CliResult {
    match args.command {
        Command::Info { file } => cmd_info(&file),
        Command::Extract { file, names, layer, out } => cmd_extract(&file, &names, layer.as_deref(), out.as_deref()),
        Command::Encode { out, crypto, mattes, ranked, sidecar } => cmd_encode(&out, &crypto, &mattes, ranked, sidecar),
        Command::Inject { input, output, name, mask, layer, op, compress } => {
            cmd_inject(&input, &output, &name, &mask, layer.as_deref(), op.into(), compress)
        }
    }
}

fn cmd_info(path: &str) -> CliResult {
    // Header-only, every part: `info` never needs the pixels.
    let layers = discover_path(path).map_err(crypto_err)?;
    outln!("file: {path}");
    if layers.is_empty() {
        outln!("  (no Cryptomatte layers)");
        return Ok(());
    }
    for layer in &layers {
        outln!("  layer: {}", layer.name);
        outln!("    part: {}", layer.part);
        outln!("    metadata_id: {}", layer.metadata_id);
        outln!("    hash: {}", layer.hash);
        outln!("    conversion: {}", layer.conversion);
        outln!("    prefix: {}", layer.md_prefix);
        if let Some(ref s) = layer.manif_file {
            outln!("    manif_file: {s}");
        }
        outln!(
            "    inline_manifest: {}",
            if layer.inline_manifest.is_some() {
                "yes"
            } else {
                "no"
            }
        );
        let ranked: Vec<&str> = layer.ranked.iter().map(|r| r.base.as_str()).collect();
        outln!("    ranked: {}", ranked.join(", "));
        match load_manifest(
            Path::new(path),
            layer.manif_file.as_deref(),
            layer.inline_manifest.as_deref(),
        ) {
            Ok(m) => {
                let mut names: Vec<&str> = m.by_name.keys().map(|s| s.as_str()).collect();
                names.sort_unstable();
                outln!("    names ({}): {}", names.len(), names.join(", "));
            }
            Err(e) => outln!("    manifest: error: {e}"),
        }
    }
    Ok(())
}

fn cmd_extract(path: &str, names: &[String], layer: Option<&str>, out: Option<&str>) -> CliResult {
    let file = CryptomatteFile::open_for_layer(path, layer).map_err(crypto_err)?;
    let layer_info = file.select_layer(layer).map_err(crypto_err)?;

    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let matte = file
        .extract_names(layer_info, &name_refs)
        .map_err(crypto_err)?;

    let nonzero = matte.coverage.iter().filter(|&&v| v > 1e-6).count();
    let sum: f64 = matte.coverage.iter().map(|&v| f64::from(v)).sum();
    let min = matte
        .coverage
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max = matte
        .coverage
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    outln!(
        "extract: layer={} names=[{}] size={}x{}",
        layer_info.name,
        names.join(","),
        matte.width,
        matte.height
    );
    outln!("  nonzero={nonzero} sum={sum:.6} min={min} max={max}");

    if let Some(out_path) = out {
        write_matte_out(out_path, &matte.coverage, Windows::of(&file.image))?;
        outln!("  wrote {out_path}");
    }
    Ok(())
}

/// Write a matte: `.raw` as bare f32 samples, anything else as a one-channel
/// EXR with the source part's data and display windows, so the matte stays
/// registered to the image it came from.
fn write_matte_out(path: &str, coverage: &[f32], windows: Windows) -> CliResult {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "raw" => {
            let mut bytes = Vec::with_capacity(coverage.len() * 4);
            for &v in coverage {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            fs::write(p, bytes).map_err(|e| CliError::Other(e.to_string()))?;
        }
        _ => {
            let mut image = Image::new(windows.data);
            image.display_window = windows.display;
            image
                .insert_channel("R", ChannelData::Float(coverage.to_vec()))
                .map_err(CliError::from)?;
            image.write(p, Compression::Zip).map_err(CliError::from)?;
        }
    }
    Ok(())
}

fn cmd_encode(out: &str, crypto_name: &str, mattes: &[(String, String)], ranked: usize, sidecar: Option<String>) -> CliResult {
    let mut named: Vec<NamedMatte> = Vec::with_capacity(mattes.len());
    let mut windows: Option<Windows> = None;
    for (name, cov_path) in mattes {
        let (w, plane) = load_coverage_exr(cov_path)?;
        match windows {
            None => windows = Some(w),
            Some(first) if first.data != w.data => {
                return Err(CliError::Other(format!(
                    "matte `{name}` data window {:?} does not match {:?}",
                    w.data, first.data
                )));
            }
            Some(_) => {}
        }
        named.push(NamedMatte {
            name: name.clone(),
            coverage: plane,
        });
    }

    let mut opts = EncodeOptions::new(crypto_name);
    opts.ranked_layers = ranked;
    if let Some(rel) = sidecar {
        opts.manifest = ManifestPlacement::Sidecar {
            relative_path: rel,
        };
    }
    let windows = windows.ok_or_else(|| CliError::Other("encode requires a --matte".into()))?;
    let enc = from_named_mattes(windows, &named, &opts).map_err(crypto_err)?;
    enc.write(out, Some(Compression::Zip)).map_err(crypto_err)?;
    outln!(
        "encoded {out} crypto={crypto_name} names={} ranked={ranked} size={}x{}",
        named.len(),
        enc.image.width(),
        enc.image.height()
    );
    Ok(())
}

/// The same `CryptomatteFile::inject` + `EncodedCryptomatte::write` path as the Python binding;
/// the CLI adds only the mask loading and the checks that belong to files (window, output).
fn cmd_inject(
    input: &str,
    output: &str,
    name: &str,
    mask: &str,
    layer: Option<&str>,
    op: MergeOp,
    compress: Option<Compression>,
) -> CliResult {
    if same_file(input, output) {
        return Err(CliError::Other("input and output file names cannot be the same".into()));
    }
    let file = CryptomatteFile::open_for_layer(input, layer).map_err(crypto_err)?;
    let layer_info = file.select_layer(layer).map_err(crypto_err)?;
    let (windows, plane) = load_coverage_exr(mask)?;
    let want = Windows::of(&file.image);
    if windows.data != want.data {
        return Err(CliError::Other(format!(
            "mask data window {:?} does not match the image's {:?}",
            windows.data, want.data
        )));
    }
    // `ranked_layers` 0 keeps the layer's own count (the library's rule, shared with Python).
    let opts = EncodeOptions { ranked_layers: 0, ..EncodeOptions::new(&layer_info.name) };
    let enc = file.inject(layer_info, name, &plane, op, &opts).map_err(crypto_err)?;
    enc.write(output, compress).map_err(crypto_err)?;
    outln!("injected {name} into {} of {input} -> {output} (part {})", layer_info.name, file.part);
    Ok(())
}

/// `name=path.exr` of `--matte`.
fn parse_matte_arg(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((name, path)) if !name.is_empty() && !path.is_empty() => Ok((name.to_string(), path.to_string())),
        _ => Err(format!("invalid matte \"{s}\" (expected name=path.exr)")),
    }
}

/// The coverage plane of a matte EXR and its windows.
///
/// The plane is the file's only channel, else `R`, else `Y` -- never "the first
/// channel", which in channel-name order is `A` for an RGBA file and `B` for RGB.
fn load_coverage_exr(path: &str) -> Result<(Windows, Vec<f32>), CliError> {
    let image = Image::read(path).map_err(CliError::from)?;
    let w = usize::try_from(image.width()).map_err(|_| CliError::Other("bad width".into()))?;
    let h = usize::try_from(image.height()).map_err(|_| CliError::Other("bad height".into()))?;
    let data = coverage_channel(&image)
        .map_err(|e| CliError::Other(format!("{path}: {e}")))?;
    // One channel -> f32 conversion for the whole crate (SSOT).
    let plane = cryptomatte::channel_as_f32(data)
        .map_err(|e| CliError::Other(format!("{path}: {e}")))?
        .into_owned();
    if plane.len() != w * h {
        return Err(CliError::Other(format!(
            "{path}: channel length {} != {w}*{h}",
            plane.len()
        )));
    }
    Ok((Windows::of(&image), plane))
}

/// See [`load_coverage_exr`] for the choice.
fn coverage_channel(image: &Image) -> Result<&ChannelData, String> {
    let names: Vec<&str> = image.channels().map(|(n, _)| n).collect();
    let pick = match names.as_slice() {
        [only] => Some(*only),
        _ => ["R", "Y"].into_iter().find(|c| names.contains(c)),
    };
    pick.and_then(|n| image.channel(n)).ok_or_else(|| {
        format!("no coverage channel: want the only channel, R or Y, found {names:?}")
    })
}

fn crypto_err(e: cryptomatte::Error) -> CliError {
    CliError::Other(e.to_string())
}
