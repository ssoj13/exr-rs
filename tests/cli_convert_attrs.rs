//! End-to-end gate: `exr-rs convert` must not lose header metadata.
//!
//! The facade used to rebuild the output header from `Image`'s own fields only,
//! so every comment / chromaticity / custom / opaque attribute a file carried
//! was dropped by a `convert`. This test drives the real CLI binary (not the
//! library) so the whole path — `Image::read` → `Image::write` — is covered.

use std::path::PathBuf;
use std::process::Command;

use exr_core::attr::{
    Attribute, Chromaticities, Compression, ExrString, Name, OpaqueAttribute, StringAttribute,
    TypedAttribute,
};
use exr_core::{ChannelData, Image};
use imath_rs::{Box2i, V2i};

/// A fresh, unique directory under the system temp dir (no external crates).
fn temp_dir(tag: &str) -> PathBuf {
    let unique = format!(
        "exr-rs-{tag}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// 8x4 two-channel source image.
fn source_image() -> Image {
    let dw = Box2i {
        min: V2i { x: 0, y: 0 },
        max: V2i { x: 7, y: 3 },
    };
    let n = 8 * 4;
    let r: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let g: Vec<u32> = (0..n).map(|i| i as u32 * 7).collect();
    Image::new(dw)
        .with_channel("R", ChannelData::Float(r))
        .with_channel("G", ChannelData::Uint(g))
}

#[test]
fn convert_preserves_header_metadata() {
    let dir = temp_dir("convert-attrs");
    let input = dir.join("in.exr");
    let output = dir.join("out.exr");

    let chroma = Chromaticities {
        red: [0.64, 0.33],
        green: [0.30, 0.60],
        blue: [0.15, 0.06],
        white: [0.3127, 0.3290],
    };
    let extra: Vec<(Name, Box<dyn Attribute>)> = vec![
        (
            "comments".into(),
            Box::new(StringAttribute::new(ExrString::from("shot 010, take 3"))),
        ),
        (
            "owner".into(),
            Box::new(StringAttribute::new(ExrString::from("Alex"))),
        ),
        (
            "chromaticities".into(),
            Box::new(TypedAttribute::<Chromaticities>::new(chroma)),
        ),
        (
            "pixelAspectRatio".into(),
            Box::new(TypedAttribute::<f32>::new(2.0)),
        ),
        // Cryptomatte-style key + a type this build does not know: both must
        // survive byte-for-byte (the unknown one as an opaque blob).
        (
            "cryptomatte/0123456/name".into(),
            Box::new(StringAttribute::new(ExrString::from("uCryptoObject"))),
        ),
        (
            "vendorAttr".into(),
            Box::new(OpaqueAttribute::with_data(
                "vendorThing",
                vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01],
            )),
        ),
    ];

    let src = source_image();
    src.write_with_attrs(&input, Compression::Zip, extra)
        .expect("write the source file");

    let status = Command::new(env!("CARGO_BIN_EXE_exr-rs"))
        .arg("convert")
        .arg(&input)
        .arg(&output)
        .args(["--compress", "none"])
        .status()
        .expect("run exr-rs convert");
    assert!(status.success(), "convert exited with {status:?}");

    let before = Image::read(&input).expect("read source");
    let after = Image::read(&output).expect("read converted");

    // The whole carried attribute set must match, compared by wire identity
    // (name + type + serialised bytes), not merely "some attributes present".
    assert_eq!(
        before.attributes(),
        after.attributes(),
        "convert dropped or altered header attributes"
    );
    for name in [
        "comments",
        "owner",
        "chromaticities",
        "pixelAspectRatio",
        "cryptomatte/0123456/name",
        "vendorAttr",
    ] {
        assert!(after.attributes().contains(name), "{name} was lost");
    }
    assert_eq!(
        after
            .attributes()
            .get("vendorAttr")
            .expect("opaque attribute")
            .type_name(),
        b"vendorThing"
    );

    // The requested re-encode really happened, and the pixels are unchanged.
    assert_eq!(after.compression, Compression::NoCompression);
    assert_eq!(before.channel("R"), after.channel("R"));
    assert_eq!(before.channel("G"), after.channel("G"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Re-encoding a **C++-written** file with the same codec must reproduce it
/// byte for byte.
///
/// This is the tightest possible statement of the metadata fix plus the `type` fix: the
/// facade used to (a) drop the source header's attributes and (b) add a
/// `type:string:13` attribute real C++ never writes for a single-part scanline
/// file (C++ only *fixes* an existing type, `ImfOutputFile.cpp:775`). Either
/// bug changes the header bytes, so this comparison catches both.
///
/// `golden.exr` is the parity oracle's single-part **scanline** image; the
/// facade writes scanlines, so if that fixture ever becomes tiled or multipart
/// this assertion is the thing that must be updated (it is not a codec test —
/// the source compression is kept).
#[test]
fn convert_reproduces_a_cpp_written_file_byte_for_byte() {
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/exr-core/tests/parity/golden.exr");
    let dir = temp_dir("convert-identity");
    let output = dir.join("out.exr");

    let status = Command::new(env!("CARGO_BIN_EXE_exr-rs"))
        .arg("convert")
        .arg(&golden)
        .arg(&output)
        .status()
        .expect("run exr-rs convert");
    assert!(status.success(), "convert exited with {status:?}");

    let before = std::fs::read(&golden).expect("read golden");
    let after = std::fs::read(&output).expect("read converted");
    assert_eq!(
        before.len(),
        after.len(),
        "converted file size differs from the C++-written source"
    );
    assert_eq!(before, after, "converted file is not byte-identical to C++");

    let _ = std::fs::remove_dir_all(&dir);
}
