//! Inject / encode regressions (bug-hunt round 3, W4.3: S1, S2, S3 + manifest).
//!
//! S2: inject must produce the ORIGINAL file with only the Cryptomatte type it
//! touches re-encoded. S1: the `conversion` metadata must be the spec value.
//! S3: the HALF path must not be quadratic.

use std::fs;
use std::path::PathBuf;

use cryptomatte::{
    from_named_mattes, CryptomatteFile, EncodeOptions, Error, HeaderAttrs, ManifestPlacement,
    MergeOp, NamedMatte, Windows,
};
use exr_core::attr::ExrString;
use exr_core::{ChannelData, Compression, Image, StringAttribute};
use imath_rs::{Box2i, Half, V2i};

const W: usize = 8;
const H: usize = 5;
const N: usize = W * H;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("exr-rs-cryptomatte-tests")
        .join(format!("{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn string_attr(v: &str) -> Box<dyn exr_core::Attribute> {
    Box::new(StringAttribute::new(ExrString::from(v)))
}

/// Source file: offset data window, its own display window, RLE, R/G/B beauty
/// channels, two Cryptomatte types and a custom header attribute.
fn write_source(path: &std::path::Path) {
    let data_window = Box2i {
        min: V2i { x: 3, y: 5 },
        max: V2i {
            x: 3 + W as i32 - 1,
            y: 5 + H as i32 - 1,
        },
    };
    let mut image = Image::new(data_window);
    image.display_window = Box2i {
        min: V2i { x: 0, y: 0 },
        max: V2i { x: 15, y: 11 },
    };
    image.compression = Compression::Rle;

    for (i, name) in ["R", "G", "B"].iter().enumerate() {
        let plane: Vec<f32> = (0..N).map(|p| (p + i * 100) as f32).collect();
        image.insert_channel(*name, ChannelData::Float(plane)).unwrap();
    }

    let mut attrs: HeaderAttrs = vec![("owner".into(), string_attr("alex"))];
    for (crypto, obj) in [("CryptoObject", "cube"), ("CryptoAsset", "asset_a")] {
        let opts = EncodeOptions {
            ranked_layers: 1,
            ..EncodeOptions::new(crypto)
        };
        let enc = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new(obj, &[1.0f32; N])], &opts).unwrap();
        for (name, data) in enc.image.channels() {
            image.insert_channel(name, data.clone()).unwrap();
        }
        for (key, attr) in &enc.attrs {
            attrs.push((key.clone(), attr.clone_attr()));
        }
    }

    image
        .write_with_attrs(path, Compression::Rle, attrs)
        .expect("write source");
}

#[test]
fn inject_preserves_channels_attrs_and_windows() {
    let dir = scratch("inject-preserve");
    let src = dir.join("src.exr");
    let dst = dir.join("dst.exr");
    write_source(&src);

    let file = CryptomatteFile::open(&src).expect("open src");
    assert_eq!(file.layers.len(), 2, "two Cryptomatte types in the source");
    let layer = file.layer_by_name("CryptoObject").expect("CryptoObject");
    let opts = EncodeOptions {
        ranked_layers: 1,
        ..EncodeOptions::new("CryptoObject")
    };
    let enc = file
        .inject(layer, "bunny", &[0.25f32; N], MergeOp::Over, &opts)
        .expect("inject");
    enc.write(&dst, None).expect("write dst");

    let out = CryptomatteFile::open(&dst).expect("open dst");

    // Every non-Cryptomatte channel survives, bit-exact.
    for name in ["R", "G", "B"] {
        assert_eq!(
            out.image.channel(name),
            file.image.channel(name),
            "channel {name} lost or changed"
        );
    }
    // Windows and compression are the source's, not origin/ZIP defaults.
    assert_eq!(out.image.data_window, file.image.data_window);
    assert_eq!(out.image.display_window, file.image.display_window);
    assert_eq!(out.image.compression, Compression::Rle);

    // Custom header attribute survives.
    assert!(out.header.contains("owner"), "custom attribute lost");

    // The other Cryptomatte type is untouched (channels + metadata + matte).
    assert_eq!(out.layers.len(), 2, "other Cryptomatte type lost");
    let asset = out.layer_by_name("CryptoAsset").expect("CryptoAsset");
    let asset_matte = out.extract_names(asset, &["asset_a"]).expect("asset matte");
    assert!((asset_matte.sum() - N as f32).abs() < 1e-5);

    // The injected type is re-encoded: bunny 0.25 over cube 1.0 -> 0.75.
    let obj = out.layer_by_name("CryptoObject").expect("CryptoObject");
    let bunny = out.extract_names(obj, &["bunny"]).expect("bunny");
    let cube = out.extract_names(obj, &["cube"]).expect("cube");
    assert!((bunny.coverage[0] - 0.25).abs() < 1e-6);
    assert!((cube.coverage[0] - 0.75).abs() < 1e-6);
}

/// S1: the only conformant `conversion` value is `uint32_to_float32` (spec
/// 1.2.0 p.3), on both the encode and the inject path.
#[test]
fn writes_spec_conversion_value() {
    let dir = scratch("conversion");
    let src = dir.join("src.exr");
    write_source(&src);
    let file = CryptomatteFile::open(&src).expect("open");
    for layer in &file.layers {
        assert_eq!(layer.conversion, cryptomatte::CONVERSION_UINT32_TO_FLOAT32);
        assert_eq!(layer.hash, cryptomatte::HASH_MURMUR3_32);
    }

    let layer = file.layer_by_name("CryptoObject").unwrap();
    let opts = EncodeOptions {
        ranked_layers: 1,
        ..EncodeOptions::new("CryptoObject")
    };
    let enc = file
        .inject(layer, "bunny", &[0.5f32; N], MergeOp::Over, &opts)
        .expect("inject");
    let key = layer.metadata_key("conversion");
    let written = enc
        .attrs
        .iter()
        .filter(|(k, _)| *k == key.as_str())
        .count();
    assert_eq!(written, 1, "exactly one conversion attribute for the type");
}

/// A missing manifest must not block extraction by name: the IDs are the name
/// hashes, which is how the reference decoder resolves them.
#[test]
fn extracts_by_name_without_manifest() {
    let dir = scratch("no-manifest");
    let path = dir.join("nomanifest.exr");

    let opts = EncodeOptions {
        ranked_layers: 1,
        manifest: ManifestPlacement::Sidecar {
            relative_path: "does_not_exist.json".into(),
        },
        ..EncodeOptions::new("CryptoObject")
    };
    let enc =
        from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("bunny", &[1.0f32; N])], &opts).unwrap();
    // Write EXR + sidecar, then lose the sidecar: `manif_file` points at nothing.
    enc.write(&path, Some(Compression::Zip)).unwrap();
    let sidecar = path.parent().unwrap().join("does_not_exist.json");
    fs::remove_file(&sidecar).expect("sidecar was written");

    let file = CryptomatteFile::open(&path).expect("open");
    let layer = &file.layers[0];
    assert!(file.load_layer_manifest(layer).is_err(), "sidecar is missing");
    let matte = file.extract_names(layer, &["bunny"]).expect("extract by name");
    assert!((matte.sum() - N as f32).abs() < 1e-5);
}

/// S3: inject over HALF channels must be linear in the pixel count.
///
/// The structural guarantee (one plane conversion per channel) is asserted in
/// `rank::tests::ranked_planes_convert_each_channel_once`; this is the
/// end-to-end guard.
///
/// Rewritten (was a wall-clock threshold: `large_us <= small_us * 8`, which
/// failed in practice under a debug build / a loaded machine -- e.g. seen
/// failing at 128^2 8875us vs 256^2 200758us, a ~22x ratio that a slow debug
/// build alone can produce even WITHOUT the O(n^2) bug, since the wall-clock
/// gate has no way to distinguish "this machine/build is just slow" from
/// "this code is quadratic"). Now counts real operations via
/// [`cryptomatte::matte::channel_as_f32_calls`] -- a
/// `#[cfg(feature = "test-instrumentation")]` thread-local counter
/// incremented once per call to `channel_as_f32`, the ONE per-channel
/// HALF/FLOAT conversion primitive `RankedPlanes::from_channels` is built
/// around ("convert every ranked plane ONCE, not once per pixel"). With that
/// fix in place, `inject_named_matte` calls it exactly `2 * pairs` times
/// (one ID plane + one coverage plane per rank PAIR -- each RGBA
/// `CryptoObjectNN` channel packs 2 pairs, `channels::RankedLayer::pairs`)
/// REGARDLESS of pixel count; the same fix pattern as exr-core's D-10
/// (`sample_count_at_calls`), which is build-mode and load INDEPENDENT --
/// same machine, same debug/release choice, same result every time.
#[test]
fn inject_half_channels_scales_linearly() {
    fn channel_as_f32_calls_during_inject(side: usize) -> u64 {
        let n = side * side;
        let opts = EncodeOptions {
            ranked_layers: 1,
            ..EncodeOptions::new("CryptoObject")
        };
        let base =
            from_named_mattes(Windows::origin(side, side).unwrap(), &[NamedMatte::new("cube", &vec![1.0f32; n])], &opts)
                .unwrap();

        // Same image with HALF coverage (G/A); IDs stay FLOAT, the only
        // storage that can hold a 32-bit ID (HALF IDs are an error, C9).
        let mut image = Image::new(base.image.data_window);
        for (name, data) in base.image.channels() {
            if name.ends_with(".R") || name.ends_with(".B") {
                image.insert_channel(name, data.clone()).unwrap();
                continue;
            }
            let halves: Vec<Half> = cryptomatte::channel_as_f32(data)
                .unwrap()
                .iter()
                .map(|&v| Half::from(v))
                .collect();
            image.insert_channel(name, ChannelData::Half(halves)).unwrap();
        }

        // Reset AFTER the setup conversions above (which are not part of
        // `inject_named_matte` and are O(1) in channel count, not pixel
        // count, anyway) so only the call under test is measured.
        cryptomatte::matte::reset_channel_as_f32_calls();
        let inj = cryptomatte::inject_named_matte(cryptomatte::InjectArgs {
            image: &image,
            crypto_name: "CryptoObject",
            ranked: &base.ranked,
            existing_manifest: &base.manifest,
            name: "bunny",
            coverage: &vec![0.25f32; n],
            op: MergeOp::Over,
            opts: &opts,
        })
        .expect("inject");
        // Read the counter BEFORE the verification `extract_coverage` call
        // below, which also calls `channel_as_f32` (once per channel, O(1)
        // in pixel count) but is no part of the operation under test.
        let calls = cryptomatte::matte::channel_as_f32_calls();

        let bunny = cryptomatte::extract_coverage(
            &inj.image,
            &inj.ranked,
            &[cryptomatte::name_to_float_id("bunny")],
        )
        .unwrap();
        assert!((bunny.coverage[0] - 0.25).abs() < 1e-3, "HALF inject value");
        calls
    }

    let small = channel_as_f32_calls_during_inject(128);
    let large = channel_as_f32_calls_during_inject(256);
    eprintln!("HALF inject channel_as_f32 calls: 128^2 {small}, 256^2 {large}");
    // With `ranked_layers: 1` both sizes must make the SAME small, constant
    // number of calls (2: one ID plane + one coverage plane) -- independent
    // of pixel count entirely, which is a strictly stronger guarantee than
    // "scales linearly" and needs no ratio/threshold at all. An O(n^2)
    // regression (re-converting a plane per pixel) would instead scale this
    // count with `n`, so equality still catches it.
    assert_eq!(
        small, large,
        "channel_as_f32 call count must be pixel-count-independent (constant per \
         ranked layer), not scale with the image size -- a per-pixel plane \
         re-conversion regression would make `large` grow with n"
    );
    // `ranked_layers: 1` packs ONE RGBA `CryptoObjectNN` channel, which the
    // Cryptomatte spec fits TWO rank pairs into (R/G = pair 0, B/A = pair 1
    // -- see `channels::RankedLayer::pairs`), so `RankedPlanes::from_channels`
    // converts 2 pairs x 2 planes (ID + coverage) each = 4 calls total.
    assert_eq!(small, 4, "expected exactly 2 rank pairs x (ID + coverage) plane conversions");
}

/// C9: HALF ID channels are an error for extraction and inject, not a silent
/// all-zero matte (a HALF keeps 16 of an ID's 32 bits and never matches).
#[test]
fn half_id_channels_are_an_error() {
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let base = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("cube", &[1.0f32; N])], &opts)
        .unwrap();
    let mut image = Image::new(base.image.data_window);
    for (name, data) in base.image.channels() {
        let halves: Vec<Half> = cryptomatte::channel_as_f32(data)
            .unwrap()
            .iter()
            .map(|&v| Half::from(v))
            .collect();
        image.insert_channel(name, ChannelData::Half(halves)).unwrap();
    }
    let err = cryptomatte::extract_coverage(&image, &base.ranked, &[cryptomatte::name_to_float_id("cube")])
        .unwrap_err();
    assert!(matches!(&err, Error::Metadata(m) if m.contains("HALF")), "{err:?}");
    let err = cryptomatte::inject_named_matte(cryptomatte::InjectArgs {
        image: &image,
        crypto_name: "CryptoObject",
        ranked: &base.ranked,
        existing_manifest: &base.manifest,
        name: "bunny",
        coverage: &[0.25f32; N],
        op: MergeOp::Over,
        opts: &opts,
    })
    .unwrap_err();
    assert!(matches!(&err, Error::Metadata(m) if m.contains("HALF")), "{err:?}");
}
