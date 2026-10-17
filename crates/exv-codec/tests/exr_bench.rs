//! Bench: EXV vs OpenEXR on the real golden corpus (sibling `exr-core`, dev-dep).
//!
//! `cargo test -p exv-codec --test exr_bench -- --ignored --nocapture`
//!
//! Reads the uncompressed `golden.exr` → raw planes, compresses with EXV profiles, and compares
//! against the file sizes of `golden_<codec>.exr` (the same image, different OpenEXR codecs). Honest
//! ratio numbers + a check of EXV lossless round-trip.

use exr_core::{ChannelData, read};
use exv_codec::image::Image as ExvImage;
use exv_video::{read_still, write_still};
use exv_codec::Profile;
use exv_format::{Channel, Dtype, Header};

const DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../../exr-rs/crates/exr-core/tests/parity/");

/// Load an EXR into a EXV Image (flat channels) + return the raw byte size.
fn load(path: &str) -> Option<(ExvImage, usize)> {
    let img = read(path).ok()?;
    let (w, h) = (img.width() as u32, img.height() as u32);
    let mut channels = Vec::new();
    let mut planes: Vec<Vec<u8>> = Vec::new();
    let mut raw = 0usize;
    for (name, cd) in img.channels() {
        let (dt, bytes): (Dtype, Vec<u8>) = match cd {
            ChannelData::Half(v) => {
                (Dtype::F16, v.iter().flat_map(|x| x.bits().to_le_bytes()).collect())
            }
            ChannelData::Float(v) => (Dtype::F32, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ChannelData::Uint(v) => (Dtype::U32, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        };
        raw += bytes.len();
        channels.push(Channel { name: name.to_string(), dtype: dt });
        planes.push(bytes);
    }
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels,
    };
    Some((ExvImage::new(header, planes).ok()?, raw))
}

#[test]
#[ignore = "needs sibling exr-core + corpus build"]
fn bench_vs_exr() {
    let golden = format!("{DIR}golden.exr");
    let Some((img, raw)) = load(&golden) else {
        eprintln!("skip: could not load {golden}");
        return;
    };
    eprintln!(
        "\n=== EXV vs OpenEXR — golden.exr {}x{}, {} channels, raw {} bytes ===",
        img.header.width,
        img.header.height,
        img.header.channels.len(),
        raw
    );

    // GpuHot = GPU-decodable (default); MaxRatio = cold/archive (deflate, CPU decode). Show both so
    // the ratio-vs-GPU-decodability tradeoff is explicit against the OpenEXR codecs below.
    for (name, prof) in [
        ("exv Fast (GpuHot)", Profile::lossless_fast()),
        ("exv Fast (MaxRatio)", Profile::lossless_fast().max_ratio()),
        ("exv Wave (GpuHot)", Profile::lossless_wave()),
        ("exv Wave (MaxRatio)", Profile::lossless_wave().max_ratio()),
    ] {
        let bytes = write_still(&img, &prof).expect("encode .exv");
        let dec = read_still(&bytes).expect("exv decode");
        let lossless = dec.planes == img.planes;
        assert!(lossless, "{name} must be lossless");
        eprintln!(
            "{name:<20} {:>9} bytes  {:>6.2}x  lossless={lossless}",
            bytes.len(),
            raw as f64 / bytes.len() as f64
        );
    }
    let lw = write_still(&img, &Profile::lossy_wave(4)).expect("encode .exv");
    eprintln!("{:<20} {:>9} bytes  {:>6.2}x  (lossy q=4)", "exv LossyWave", lw.len(), raw as f64 / lw.len() as f64);

    eprintln!("  -- OpenEXR (same frame, different codecs; * = lossy) --");
    for (codec, lossy) in [
        ("rle", false),
        ("zips", false),
        ("zip", false),
        ("piz", false),
        ("pxr24", true),
        ("b44", true),
        ("dwaa", true),
        ("htj2k256", false),
    ] {
        let p = format!("{DIR}golden_{codec}.exr");
        if let Ok(m) = std::fs::metadata(&p) {
            eprintln!(
                "exr {:<16} {:>9} bytes  {:>6.2}x{}",
                codec,
                m.len(),
                raw as f64 / m.len() as f64,
                if lossy { " *" } else { "" }
            );
        }
    }
}
