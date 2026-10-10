//! Structured `IDManifest` cross-check against real C++ OpenEXR.
//!
//! Two directions, both gated on `EXR_ORACLE_EXE` (skip-with-note
//! standalone, per `tests/support/mod.rs`):
//!
//! (a) **C++ -> ours:** the oracle's `write-idmanifest <path>` mode writes a
//!     tiny EXR whose header carries a real C++-built `IDManifestAttribute`.
//!     We read it, decompress + parse the attribute with
//!     [`exr_core::IDManifest::from_compressed`], and assert the decoded
//!     structure matches the KNOWN deterministic manifest below.
//!
//! (b) **ours -> C++:** we build the SAME known manifest with our
//!     [`exr_core::IDManifest`], serialize + compress it with
//!     [`exr_core::IDManifest::to_compressed`], and write it into a header
//!     attribute with our [`exr_core::scanline::OutputFile`]. The oracle's
//!     `dump-idmanifest <path> <out>` mode then decodes it with real C++ and
//!     writes a canonical text dump; we assert that dump matches the same
//!     KNOWN manifest.
//!
//! Comparing DECODED content (not raw attribute bytes) is deliberate: two
//! different zlib implementations need not produce byte-identical compressed
//! output for the same input, so a byte-diff of the attribute would be
//! testing the wrong thing. What must match is what each side's decoder
//! reconstructs from the other side's encoder — the actual interop contract.
//!
//! # KNOWN manifest (must equal `oracle.cpp`'s `do_write_idmanifest`)
//! One channel-group manifest:
//!   * channel: `"id"`
//!   * component: `"name"`
//!   * hash scheme: `MurmurHash3_32`
//!   * encoding scheme: `"id"`
//!   * lifetime: `STABLE`
//!   * entries: `1 -> "asset_one"`, `2 -> "asset_two"`, `1000 -> "asset_big_id"`

use std::path::Path;
use std::process::Command;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, IDManifest, IdLifetime, PixelType};
use exr_core::header::Header;
use exr_core::scanline::OutputFile;
use exr_core::{IDManifestAttribute, StdIfStream};

mod support;

fn known_manifest() -> IDManifest {
    let mut m = IDManifest::new();
    let g = m.add_channel("id");
    g.set_component("name").unwrap();
    g.set_hash_scheme(exr_core::attr::MURMURHASH3_32);
    g.set_encoding_scheme(exr_core::attr::ID_SCHEME);
    g.set_lifetime(IdLifetime::Stable);
    g.insert_single(1, "asset_one").unwrap();
    g.insert_single(2, "asset_two").unwrap();
    g.insert_single(1000, "asset_big_id").unwrap();
    m
}

fn assert_is_known_manifest(m: &IDManifest) {
    assert_eq!(m.len(), 1, "expected exactly one channel-group manifest");
    let g = m.get(0).unwrap();
    assert!(g.channels().contains("id"), "channels: {:?}", g.channels());
    assert_eq!(g.components(), &["name".to_string()]);
    assert_eq!(g.hash_scheme(), exr_core::attr::MURMURHASH3_32);
    assert_eq!(g.lifetime(), IdLifetime::Stable);
    assert_eq!(g.find(1), Some(&["asset_one".to_string()][..]));
    assert_eq!(g.find(2), Some(&["asset_two".to_string()][..]));
    assert_eq!(g.find(1000), Some(&["asset_big_id".to_string()][..]));
    assert_eq!(g.len(), 3);
}

/// Direction (a): C++ writes the file, we read + decode it.
#[test]
fn cpp_written_idmanifest_decodes_correctly_in_rust() {
    let name = "idmanifest_gate::cpp_written_idmanifest_decodes_correctly_in_rust";
    let Some(oracle) = support::oracle_or_skip(name) else {
        return;
    };

    let mut path = std::env::temp_dir();
    path.push("exr_core_idmanifest_from_cpp.exr");

    let output = Command::new(&oracle)
        .arg("write-idmanifest")
        .arg(&path)
        .output()
        .expect("failed to run oracle write-idmanifest");
    assert!(
        output.status.success(),
        "oracle write-idmanifest failed (status {}):\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let mut stream = StdIfStream::open(path.to_str().unwrap()).expect("open oracle-written file");
    let (header, _) = Header::read_checked(&mut stream).expect("read header");
    let attr = header.get("idmanifest").expect("idmanifest attribute present");
    let compressed = attr
        .as_any()
        .downcast_ref::<IDManifestAttribute>()
        .expect("idmanifest attribute has the expected typed shape")
        .value
        .clone();

    let manifest = IDManifest::from_compressed(&compressed)
        .expect("Rust must decode a real C++-written idmanifest attribute");
    assert_is_known_manifest(&manifest);

    let _ = std::fs::remove_file(&path);
}

/// Direction (b): we write the file, C++ reads + decodes it.
#[test]
fn rust_written_idmanifest_decodes_correctly_in_cpp() {
    let name = "idmanifest_gate::rust_written_idmanifest_decodes_correctly_in_cpp";
    let Some(oracle) = support::oracle_or_skip(name) else {
        return;
    };

    let manifest = known_manifest();
    let compressed = manifest.to_compressed().expect("serialize + compress");

    // A minimal 1x1 single-FLOAT-channel header carrying the attribute —
    // matches oracle.cpp's `do_write_idmanifest` header shape closely enough
    // for `dump-idmanifest` (which only reads the header, not pixels).
    let mut header = Header::with_dimensions(
        4,
        4,
        1.0,
        imath_rs::V2f { x: 0.0, y: 0.0 },
        1.0,
        exr_core::attr::LineOrder::IncreasingY,
        exr_core::attr::Compression::NoCompression,
    )
    .unwrap();
    let mut channels = ChannelList::new();
    channels.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    header.insert("channels", ChannelListAttribute::new(channels)).unwrap();
    header.insert("idmanifest", IDManifestAttribute::new(compressed)).unwrap();

    let mut exr_path = std::env::temp_dir();
    exr_path.push("exr_core_idmanifest_to_cpp.exr");
    {
        let mut os = exr_core::io::std_io::StdOfStream::create(exr_path.to_str().unwrap())
            .expect("create output file");
        let mut of = OutputFile::new(&mut os, &header).expect("construct OutputFile");
        let pixels = vec![0u8; 4 * 4 * 4]; // 16 FLOAT pixels, all zero
        let mut fb = exr_core::frame_buffer::FrameBuffer::new();
        fb.insert(
            "Z",
            exr_core::frame_buffer::Slice::make(
                PixelType::Float,
                0,
                imath_rs::V2i { x: 0, y: 0 },
                4,
                0,
                0,
                1,
                1,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();
        of.set_frame_buffer(&fb).expect("set frame buffer");
        of.write_pixels(&pixels, 4).expect("write pixels");
    }

    let mut dump_path = std::env::temp_dir();
    dump_path.push("exr_core_idmanifest_to_cpp.dump.txt");
    let output = Command::new(&oracle)
        .arg("dump-idmanifest")
        .arg(&exr_path)
        .arg(&dump_path)
        .output()
        .expect("failed to run oracle dump-idmanifest");
    assert!(
        output.status.success(),
        "oracle dump-idmanifest failed (status {}):\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let dump = std::fs::read_to_string(&dump_path).expect("read oracle's text dump");
    assert_dump_matches_known_manifest(&dump);

    let _ = std::fs::remove_file(&exr_path);
    let _ = std::fs::remove_file(&dump_path);
}

/// Parses `oracle.cpp`'s `do_dump_idmanifest` text format (see its own
/// rustdoc-style comment there) well enough to check it against
/// [`known_manifest`], without needing a third copy of the format's grammar.
fn assert_dump_matches_known_manifest(dump: &str) {
    let mut lines = dump.lines();
    let group_line = lines.next().expect("dump has a GROUP line");
    assert!(group_line.starts_with("GROUP "), "unexpected line: {group_line}");
    let rest = &group_line["GROUP ".len()..];
    let parts: Vec<&str> = rest.split(" | ").collect();
    assert_eq!(parts.len(), 5, "GROUP line shape: {group_line}");
    assert_eq!(parts[0], "id", "channels");
    assert_eq!(parts[1], "name", "components");
    assert_eq!(parts[2], "2", "lifetime (STABLE == 2)");
    assert_eq!(parts[3], "MurmurHash3_32", "hash scheme");
    assert_eq!(parts[4], "id", "encoding scheme");

    let mut entries: Vec<(u64, String)> = lines
        .map(|line| {
            assert!(line.starts_with("ENTRY "), "unexpected line: {line}");
            let rest = &line["ENTRY ".len()..];
            let (id_str, text) = rest.split_once(' ').expect("ENTRY has an id and text");
            (id_str.parse().expect("numeric id"), text.to_string())
        })
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        vec![
            (1, "asset_one".to_string()),
            (2, "asset_two".to_string()),
            (1000, "asset_big_id".to_string()),
        ]
    );
}

#[allow(dead_code)]
fn _unused_path_type_check(_p: &Path) {}
