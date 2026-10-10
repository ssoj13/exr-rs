//! End-to-end gate for the `exr-rs cryptomatte` CLI:
//! `encode` reads the coverage from R (not "the first channel", `A` for RGBA)
//! and keeps the coverage file's windows; `extract --out` writes the matte with
//! the source's data and display windows. Drives the real binary.

use std::path::{Path, PathBuf};
use std::process::Command;

use cryptomatte::CryptomatteFile;
use exr_core::{ChannelData, Compression, Image};
use imath_rs::{Box2i, V2i};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("exr-rs-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// 4x3 data window at (5, -2) inside a 12x8 display window at (0, -4).
fn windows() -> (Box2i, Box2i) {
    let data = Box2i { min: V2i { x: 5, y: -2 }, max: V2i { x: 8, y: 0 } };
    let display = Box2i { min: V2i { x: 0, y: -4 }, max: V2i { x: 11, y: 3 } };
    (data, display)
}

/// Coverage in R (a gradient), A = 1 everywhere as a decoy, plus G/B.
fn coverage_rgba(path: &Path) -> Vec<f32> {
    let (data, display) = windows();
    let r: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
    let mut image = Image::new(data)
        .with_channel("R", ChannelData::Float(r.clone()))
        .with_channel("G", ChannelData::Float(vec![0.0; 12]))
        .with_channel("B", ChannelData::Float(vec![0.0; 12]))
        .with_channel("A", ChannelData::Float(vec![1.0; 12]));
    image.display_window = display;
    image.write(path, Compression::Zip).expect("write coverage");
    r
}

fn exr_rs(args: &[&str]) -> bool {
    Command::new(env!("CARGO_BIN_EXE_exr-rs"))
        .args(args)
        .status()
        .expect("run exr-rs")
        .success()
}

#[test]
fn encode_reads_r_and_keeps_windows_and_extract_out_keeps_them_too() {
    let dir = temp_dir("cli-crypto");
    let cov = dir.join("cov.exr");
    let out = dir.join("crypto.exr");
    let matte = dir.join("matte.exr");
    let r = coverage_rgba(&cov);
    let (data, display) = windows();

    let bunny = format!("bunny={}", cov.display());
    assert!(exr_rs(&["cryptomatte", "encode", out.to_str().unwrap(), "-c", "CryptoObject", "-m", &bunny]));

    let file = CryptomatteFile::open(&out).expect("open encoded");
    assert_eq!(file.image.data_window, data, "encode keeps the data window");
    assert_eq!(file.image.display_window, display, "encode keeps the display window");
    let layer = file.select_layer(None).expect("layer");
    let got = file.extract_names(layer, &["bunny"]).expect("extract");
    assert_eq!(got.coverage, r, "coverage came from R, not the A decoy");

    assert!(exr_rs(&[
        "cryptomatte", "extract", out.to_str().unwrap(), "-n", "bunny", "-o", matte.to_str().unwrap(),
    ]));
    let m = Image::read(&matte).expect("read matte");
    assert_eq!(m.data_window, data, "extract --out keeps the data window");
    assert_eq!(m.display_window, display, "extract --out keeps the display window");
    match m.channel("R").expect("R") {
        ChannelData::Float(v) => assert_eq!(v, &r),
        other => panic!("expected FLOAT, got {other:?}"),
    }
}

#[test]
fn encode_rejects_a_coverage_file_without_an_unambiguous_channel() {
    let dir = temp_dir("cli-crypto-ambiguous");
    let cov = dir.join("gb.exr");
    let (data, _) = windows();
    Image::new(data)
        .with_channel("G", ChannelData::Float(vec![0.5; 12]))
        .with_channel("B", ChannelData::Float(vec![0.5; 12]))
        .write(&cov, Compression::Zip)
        .expect("write");
    let bunny = format!("bunny={}", cov.display());
    let out = dir.join("crypto.exr");
    assert!(!exr_rs(&["cryptomatte", "encode", out.to_str().unwrap(), "-c", "CryptoObject", "-m", &bunny]));
}

// ---- `inject` ----------------------------------------------------------------

use cryptomatte::{from_named_mattes, EncodeOptions, NamedMatte, Windows};
use exr_core::io::StdIfStream;
use exr_core::RawImage;

/// A one-channel coverage EXR on the shared windows: `first` at pixel 0, zero elsewhere.
fn mask_at_first(path: &Path, first: f32) -> Vec<f32> {
    let (data, display) = windows();
    let mut v = vec![0.0f32; 12];
    v[0] = first;
    let mut image = Image::new(data).with_channel("R", ChannelData::Float(v.clone()));
    image.display_window = display;
    image.write(path, Compression::Zip).expect("write mask");
    v
}

/// Single-part source holding `bunny` (the R gradient, zero at pixel 0) in `CryptoObject`.
fn single_part_source(dir: &Path) -> (PathBuf, Vec<f32>) {
    let cov = dir.join("bunny.exr");
    let r = coverage_rgba(&cov);
    let src = dir.join("src.exr");
    let bunny = format!("bunny={}", cov.display());
    assert!(exr_rs(&["cryptomatte", "encode", src.to_str().unwrap(), "-c", "CryptoObject", "-m", &bunny]));
    (src, r)
}

/// beauty | crypto (bunny) | depth, as a multipart scanline file.
fn multipart_source(dir: &Path) -> (PathBuf, Vec<f32>) {
    let (data, display) = windows();
    let r: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
    let enc = from_named_mattes(
        Windows { data, display },
        &[NamedMatte::new("bunny", &r)],
        &EncodeOptions::new("CryptoObject"),
    )
    .expect("encode");
    let mut crypto = enc.image.clone();
    for (k, a) in &enc.attrs {
        crypto.attributes_mut().insert(k.clone(), a.clone_attr()).expect("attr");
    }
    let plain = |name: &str, v: f32| {
        let mut i = Image::new(data).with_channel(name, ChannelData::Float(vec![v; 12]));
        i.display_window = display;
        i
    };
    let src = dir.join("multi.exr");
    exr_core::api::write_multipart_scanline(
        &src,
        vec![
            (plain("R", 0.5), Compression::Zip, "beauty".to_string()),
            (crypto, Compression::Zip, "crypto".to_string()),
            (plain("depth", 2.0), Compression::Zip, "depth".to_string()),
        ],
    )
    .expect("write multipart");
    (src, r)
}

/// Extract `name` of `CryptoObject` from `path` through the library.
fn matte_of(path: &Path, name: &str) -> Vec<f32> {
    let file = CryptomatteFile::open_for_layer(path, Some("CryptoObject")).expect("open");
    let layer = file.select_layer(Some("CryptoObject")).expect("layer");
    file.extract_names(layer, &[name]).expect("extract").coverage
}

fn cli_inject(src: &Path, out: &Path, extra: &[&str]) -> bool {
    let mut args = vec!["cryptomatte", "inject", src.to_str().unwrap(), out.to_str().unwrap()];
    args.extend_from_slice(extra);
    exr_rs(&args)
}

#[test]
fn inject_single_part_gives_the_mask_back_and_leaves_other_objects_alone() {
    let dir = temp_dir("cli-inject-single");
    let (src, r) = single_part_source(&dir);
    let mask = dir.join("cube.exr");
    let m = mask_at_first(&mask, 0.5);
    let out = dir.join("out.exr");
    assert!(cli_inject(&src, &out, &["-n", "cube", "-m", mask.to_str().unwrap(), "-c", "rle"]));

    assert_eq!(matte_of(&out, "cube"), m, "the mask comes back");
    assert_eq!(matte_of(&out, "bunny"), r, "bunny is zero under the mask, so it is unchanged");
    assert_eq!(Image::read(&out).expect("read").compression, Compression::Rle, "-c applies");
    // The CLI's extract sees it too.
    assert!(exr_rs(&["cryptomatte", "extract", out.to_str().unwrap(), "-n", "cube"]));
}

#[test]
fn inject_multipart_keeps_the_other_parts_byte_for_byte() {
    let dir = temp_dir("cli-inject-multi");
    let (src, r) = multipart_source(&dir);
    let mask = dir.join("cube.exr");
    let m = mask_at_first(&mask, 0.5);
    let out = dir.join("out.exr");
    assert!(cli_inject(&src, &out, &["-n", "cube", "-m", mask.to_str().unwrap()]));

    assert_eq!(matte_of(&out, "cube"), m);
    assert_eq!(matte_of(&out, "bunny"), r);
    let raw = |p: &Path| RawImage::read(&mut StdIfStream::open(p.to_str().unwrap()).unwrap()).unwrap();
    let (a, b) = (raw(&src), raw(&out));
    assert_eq!(b.parts.len(), 3);
    for i in [0, 2] {
        assert_eq!(a.parts[i].chunks, b.parts[i].chunks, "part {i} chunks");
    }
    assert!(CryptomatteFile::open(&out).expect("open").multipart);
}

#[test]
fn inject_over_and_under_differ_where_the_object_is_already_there() {
    let dir = temp_dir("cli-inject-op");
    let (src, r) = single_part_source(&dir);
    let mask = dir.join("full.exr");
    let (data, _) = windows();
    Image::new(data)
        .with_channel("R", ChannelData::Float(vec![1.0; 12]))
        .write(&mask, Compression::Zip)
        .expect("write");
    let (over, under) = (dir.join("over.exr"), dir.join("under.exr"));
    assert!(cli_inject(&src, &over, &["-n", "sky", "-m", mask.to_str().unwrap()]));
    assert!(cli_inject(&src, &under, &["-n", "sky", "-m", mask.to_str().unwrap(), "-o", "under"]));
    // Over a full mask, bunny is gone; under it, bunny keeps its coverage and sky gets the rest.
    assert!(matte_of(&over, "bunny").iter().all(|&v| v == 0.0));
    let bunny = matte_of(&under, "bunny");
    for (got, want) in bunny.iter().zip(&r) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    for (sky, b) in matte_of(&under, "sky").iter().zip(&r) {
        assert!((sky - (1.0 - b)).abs() < 1e-6);
    }
}

#[test]
fn inject_errors_leave_no_output_and_the_input_alone() {
    let dir = temp_dir("cli-inject-errors");
    let (src, _) = single_part_source(&dir);
    let before = std::fs::read(&src).expect("read");
    let mask = dir.join("m.exr");
    mask_at_first(&mask, 0.5);
    let out = dir.join("out.exr");
    let m = mask.to_str().unwrap();

    assert!(!cli_inject(&src, &out, &["-n", "cube", "-m", m, "-l", "Nope"]), "unknown layer");
    // A mask on another data window.
    let wrong = dir.join("wrong.exr");
    Image::new(Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 3, y: 2 } })
        .with_channel("R", ChannelData::Float(vec![0.5; 12]))
        .write(&wrong, Compression::Zip)
        .expect("write");
    assert!(!cli_inject(&src, &out, &["-n", "cube", "-m", wrong.to_str().unwrap()]), "window mismatch");
    // Output is the input, however it is spelled.
    assert!(!cli_inject(&src, &src, &["-n", "cube", "-m", m]), "output == input");
    let dotted = dir.join(".").join("src.exr");
    assert!(!cli_inject(&src, &dotted, &["-n", "cube", "-m", m]), "output == input, spelled differently");

    assert!(!out.exists(), "a refused inject writes nothing");
    assert_eq!(std::fs::read(&src).expect("read"), before, "the input is untouched");
}

#[test]
fn inject_picks_the_part_of_the_named_type() {
    let dir = temp_dir("cli-inject-layer");
    let (data, display) = windows();
    let r: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
    let crypto = |name: &str, obj: &str| {
        let enc = from_named_mattes(Windows { data, display }, &[NamedMatte::new(obj, &r)], &EncodeOptions::new(name))
            .expect("encode");
        let mut image = enc.image.clone();
        for (k, a) in &enc.attrs {
            image.attributes_mut().insert(k.clone(), a.clone_attr()).expect("attr");
        }
        image
    };
    // CryptoMaterial sorts first, so the default type lives in part 2, not part 1.
    let src = dir.join("two.exr");
    exr_core::api::write_multipart_scanline(
        &src,
        vec![
            (crypto("CryptoObject", "bunny"), Compression::Zip, "obj".to_string()),
            (crypto("CryptoMaterial", "porcelain"), Compression::Zip, "mat".to_string()),
        ],
    )
    .expect("write");
    let mask = dir.join("cube.exr");
    let m = mask_at_first(&mask, 0.5);
    let out = dir.join("out.exr");
    assert!(cli_inject(&src, &out, &["-n", "cube", "-m", mask.to_str().unwrap(), "-l", "CryptoObject"]));

    assert_eq!(matte_of(&out, "cube"), m, "injected into CryptoObject");
    let mat = CryptomatteFile::open_for_layer(&out, Some("CryptoMaterial")).expect("open material");
    let layer = mat.select_layer(Some("CryptoMaterial")).expect("layer");
    assert_eq!(mat.extract_names(layer, &["porcelain"]).expect("porcelain").coverage, r);
    assert_eq!(mat.extract_names(layer, &["cube"]).expect("cube").sum(), 0.0, "CryptoMaterial untouched");
}
