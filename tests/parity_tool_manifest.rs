//! `exr-rs manifest` against OpenEXR's `exrmanifest` (v3.4.15): the stdout text and the exit code
//! over the goldens, the private image set, and generated files with several manifests, several
//! parts, wide and non-ASCII cells. `#[ignore]`d: run by `cargo run -p xtask -- parity-tools`.

mod tool_support;

use std::path::Path;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::MemOStream;
use exr_core::{
    ID_SCHEME, ID2_SCHEME, IDManifest, IDManifestAttribute, IdLifetime, MURMURHASH3_32, MURMURHASH3_64,
    MultiPartOutputFile, NOTHASHED, StringAttribute,
};
use imath_rs::V2f;
use tool_support::*;

const W: i32 = 4;
const H: i32 = 2;

/// Run both tools on `args` (file paths) and compare the exit code and the stdout text.
fn compare(cpp_exe: &Path, args: &[&Path]) -> Result<(), String> {
    let (c, o) = (run(cpp_exe, args), run_ours("manifest", args));
    same_run(&c, &o, || match first_diff(c.text().as_bytes(), o.text().as_bytes()) {
        Some(msg) => Err(format!("stdout: {msg}\n--- C++\n{}\n--- ours\n{}", c.text(), o.text())),
        None => Ok(()),
    })
}

/// One scanline part header, `name` making it a multipart member, with `manifest` attached.
fn part_header(name: Option<&str>, manifest: Option<&IDManifest>) -> Header {
    let mut h = Header::with_dimensions(
        W,
        H,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .unwrap();
    let mut channels = ChannelList::new();
    channels.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
    h.insert("channels", ChannelListAttribute::new(channels)).unwrap();
    // `MultiPartOutputFile` wants a `type` even for one part; only multipart members need a name.
    h.insert("type", StringAttribute::new("scanlineimage".into())).unwrap();
    if let Some(name) = name {
        h.insert("name", StringAttribute::new(name.into())).unwrap();
    }
    if let Some(m) = manifest {
        h.insert("idManifest", IDManifestAttribute::new(m.to_compressed().unwrap())).unwrap();
    }
    h
}

/// Write a file with one part per header (a single part is a plain single-part file).
fn write_parts(path: &Path, headers: &[Header]) {
    let pixels: Vec<u8> = (0..W * H).flat_map(|i| (i as f32).to_ne_bytes()).collect();
    let mut fb = FrameBuffer::new();
    fb.insert("Z", Slice::new(PixelType::Float, 0, 4, W as usize * 4, 1, 1, 0.0, false, false).unwrap()).unwrap();
    let mut out = MemOStream::new();
    {
        let mut file = MultiPartOutputFile::new(&mut out, headers).unwrap();
        for p in 0..headers.len() as i32 {
            let mut part = file.output_part(p).unwrap();
            part.set_frame_buffer(&fb).unwrap();
            part.write_pixels(&pixels, H).unwrap();
        }
    }
    std::fs::write(path, out.into_data()).unwrap();
}

/// Groups covering every lifetime, hash scheme and encoding, one to three components, ids of 1 to
/// 20 digits, multi-channel groups, non-ASCII cells (the C++ pads by bytes), empty cells, a group
/// without entries and one without components.
fn rich_manifest() -> IDManifest {
    let mut m = IDManifest::new();
    let g = m.add_channel("crypto_object");
    g.set_component("name").unwrap();
    g.set_hash_scheme(MURMURHASH3_32);
    g.set_encoding_scheme(ID_SCHEME);
    g.set_lifetime(IdLifetime::Stable);
    for name in ["/world/hero", "/world/prop_sword", "/wörld/héro_ünïcode", "", "x"] {
        g.insert_hashed_single(name).unwrap();
    }
    g.insert_single(7, "seven").unwrap();
    g.insert_single(u64::MAX, "max").unwrap();

    let g = m.add_channel("material");
    g.channels_mut().insert("material2".to_string());
    g.channels_mut().insert("Ünicode.channel".to_string());
    g.set_components(vec!["shader".into(), "variant".into(), "n".into()]).unwrap();
    g.set_hash_scheme(MURMURHASH3_64);
    g.set_encoding_scheme(ID2_SCHEME);
    g.set_lifetime(IdLifetime::Shot);
    for i in 0..300u64 {
        g.insert(
            i * 7919 + 1,
            vec![format!("shader_{}", "x".repeat((i % 13) as usize)), format!("v{i}"), String::new()],
        )
        .unwrap();
    }

    let g = m.add_channel("frame_ids");
    g.set_component("id").unwrap();
    g.set_hash_scheme(NOTHASHED);
    g.set_lifetime(IdLifetime::Frame);
    g.insert_single(1, "a").unwrap();
    g.insert_single(1_000_000, "b").unwrap();

    m.add_channel("no_entries").set_component("only a header").unwrap();
    m.add_channel("no_components");
    m
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn stdout_is_identical_over_every_input() {
    let cpp_exe = cpp("exrmanifest");
    let mut report = Report::default();
    for file in inputs() {
        report.check(file.display(), compare(&cpp_exe, &[&file]));
    }
    report.finish("manifest over goldens + image set");
}

/// Generated files: rich and tiny manifests, a multipart file mixing parts with and without one,
/// and several files in one invocation (the second failing, the output up to it kept).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn generated_manifests_are_identical() {
    let cpp_exe = cpp("exrmanifest");
    let dir = TempDir::new("manifest");
    let rich = rich_manifest();
    let mut small = IDManifest::new();
    small.add_channel("id").set_component("name").unwrap();

    let single = dir.join("single.exr");
    write_parts(&single, &[part_header(None, Some(&rich))]);
    let plain = dir.join("plain.exr");
    write_parts(&plain, &[part_header(None, None)]);
    let multi = dir.join("multi.exr");
    write_parts(
        &multi,
        &[part_header(Some("a"), Some(&rich)), part_header(Some("b"), None), part_header(Some("c"), Some(&small))],
    );
    let missing = dir.join("missing.exr");

    let mut report = Report::default();
    for (what, files) in [
        ("rich single part", vec![&single]),
        ("no manifest", vec![&plain]),
        ("multipart", vec![&multi]),
        ("several files", vec![&single, &plain, &multi]),
        ("a missing file stops the run", vec![&single, &missing, &multi]),
        ("a missing file alone", vec![&missing]),
    ] {
        let args: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
        report.check(what, compare(&cpp_exe, &args));
    }
    report.finish("manifest over generated files");

    // The generated output really is non-trivial (the comparison above is not empty vs empty).
    let text = run_ours("manifest", [&single]).text();
    assert!(
        text.contains("crypto_object") && text.contains("héro_ünïcode") && text.contains("raw text size"),
        "{text}"
    );
}

/// `--version`: same text, exit code 0 (anywhere on the command line, with or without files).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrmanifest", "manifest", &[&["--version"], &["nonexistent.exr", "--version"]]);
}
