//! Integration tests against Psyop sample EXRs.
//!
//! Samples and goldens come from the test image set (`exr-test-data`, run with
//! `--features test-data`; `CRYPTOMATTE_SAMPLES` points at another set). Stats + bit-exact
//! pixel goldens.

mod common;

use cryptomatte::{float_id_to_hex, hex_to_float_id, name_to_float_id, CryptomatteFile, Matte};

use common::{assert_matches_golden, require_sample, samples_root};

struct MatteStats {
    sum: f32,
    nonzero: usize,
    max: f32,
    over_one: usize,
}

fn matte_stats(m: &Matte) -> MatteStats {
    let mut nonzero = 0usize;
    let mut max = 0.0f32;
    let mut over_one = 0usize;
    for &c in &m.coverage {
        assert!(c.is_finite(), "non-finite coverage");
        assert!(c >= 0.0, "negative coverage {c}");
        if c > 1e-6 {
            nonzero += 1;
        }
        if c > max {
            max = c;
        }
        if c > 1.0 + 1e-3 {
            over_one += 1;
        }
    }
    MatteStats {
        sum: m.sum(),
        nonzero,
        max,
        over_one,
    }
}

fn assert_sum_close(got: f32, expected: f32, abs_tol: f32) {
    assert!(
        (got - expected).abs() <= abs_tol,
        "sum {got} not within {abs_tol} of {expected}"
    );
}

fn assert_mattes_bit_eq(a: &Matte, b: &Matte) {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    assert_eq!(a.coverage.len(), b.coverage.len());
    for (i, (x, y)) in a.coverage.iter().zip(b.coverage.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "pixel {i}");
    }
}

fn assert_manifest_hashes_match(file: &CryptomatteFile, layer_name: &str) {
    let layer = file.layer_by_name(layer_name).expect(layer_name);
    let manif = file.load_layer_manifest(layer).expect("manifest");
    assert!(!manif.by_name.is_empty());
    for (name, id) in &manif.by_name {
        assert_eq!(
            name_to_float_id(name).to_bits(),
            id.to_bits(),
            "manifest hex for {name:?} must equal Murmur float id"
        );
        assert_eq!(float_id_to_hex(*id), float_id_to_hex(name_to_float_id(name)));
    }
}

/// Bunny object matte — measured on Psyop `bunny_CryptoObject.exr`.
const BUNNY_W: i64 = 1280;
const BUNNY_H: i64 = 720;
const BUNNY_RANKED: usize = 3;
const BUNNY_SUM: f32 = 60460.645;
const BUNNY_NZ: usize = 68919;
const BUNNY_HEX: &str = "13851a76";

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn bunny_discover_and_extract_are_real() {
    let path = require_sample("bunny_CryptoObject.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    assert_eq!(file.layers.len(), 1);
    let layer = file.layer_by_name("uCryptoObject").expect("uCryptoObject");
    assert!(layer.hash.eq_ignore_ascii_case("MurmurHash3_32"));
    assert!(
        layer.conversion.eq_ignore_ascii_case("float")
            || layer.conversion.eq_ignore_ascii_case("uint32_to_float32")
    );
    assert_eq!(layer.ranked.len(), BUNNY_RANKED);
    assert!(layer.inline_manifest.is_some());
    assert!(layer.manif_file.is_none());
    assert_eq!(file.image.width(), BUNNY_W);
    assert_eq!(file.image.height(), BUNNY_H);

    let manif = file.load_layer_manifest(layer).expect("manifest");
    assert_eq!(manif.by_name.len(), 32166);
    let bunny_id = *manif.by_name.get("bunny").expect("bunny in manifest");
    assert_eq!(bunny_id.to_bits(), name_to_float_id("bunny").to_bits());
    assert_eq!(bunny_id.to_bits(), hex_to_float_id(BUNNY_HEX).unwrap().to_bits());
    assert_eq!(float_id_to_hex(bunny_id), BUNNY_HEX);

    let by_name = file.extract_names(layer, &["bunny"]).expect("by name");
    let by_id = file.extract_ids(layer, &[bunny_id]).expect("by id");
    assert_mattes_bit_eq(&by_name, &by_id);
    assert_eq!(by_name.width as i64, BUNNY_W);
    assert_eq!(by_name.height as i64, BUNNY_H);

    let st = matte_stats(&by_name);
    assert_eq!(st.nonzero, BUNNY_NZ);
    assert_sum_close(st.sum, BUNNY_SUM, 0.05);
    assert_eq!(st.max, 1.0);
    assert_eq!(st.over_one, 0);
    assert_matches_golden(&by_name, &samples_root(), "bunny_CryptoObject.exr", "bunny");

    let junk = file
        .extract_names(layer, &["__no_such_cryptomatte_object__"])
        .expect("junk");
    let junk_st = matte_stats(&junk);
    assert_eq!(junk_st.sum, 0.0);
    assert_eq!(junk_st.nonzero, 0);
}

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn sidecar_bunny_matches_inline_matte_stats() {
    let path = require_sample("sidecar_manifest/bunny_CryptoObject.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    let layer = file.layers.first().expect("layer");
    assert_eq!(
        layer.manif_file.as_deref(),
        Some("bunny_CryptoObject_manifest.json")
    );
    assert!(layer.inline_manifest.is_none());

    let manif = file.load_layer_manifest(layer).expect("sidecar");
    assert_eq!(manif.by_name.len(), 32166);
    assert_eq!(
        manif.by_name["bunny"].to_bits(),
        hex_to_float_id(BUNNY_HEX).unwrap().to_bits()
    );

    let matte = file.extract_names(layer, &["bunny"]).expect("extract");
    let st = matte_stats(&matte);
    assert_eq!(st.nonzero, BUNNY_NZ);
    assert_sum_close(st.sum, BUNNY_SUM, 0.05);
    assert_eq!(st.max, 1.0);
    assert_eq!(st.over_one, 0);
    assert_matches_golden(
        &matte,
        &samples_root(),
        "sidecar_manifest/bunny_CryptoObject.exr",
        "bunny",
    );
}

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn special_chars_utf8_and_punctuation_names() {
    let path = require_sample("debug_images/special_chars.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    assert_eq!(file.layers.len(), 1);
    assert_manifest_hashes_match(&file, "special_chars");

    let layer = file.layer_by_name("special_chars").unwrap();
    let manif = file.load_layer_manifest(layer).unwrap();
    assert_eq!(manif.by_name.len(), 4);
    for key in [
        "равнина",
        "name_containing/slash",
        "name, with, comma",
        "name containing a \"quote\"  ",
    ] {
        assert!(manif.by_name.contains_key(key), "missing {key:?}");
    }

    // Psyop float golden for UTF-8 "равнина".
    assert_eq!(
        manif.by_name["равнина"].to_bits(),
        name_to_float_id("равнина").to_bits()
    );
    assert_eq!(float_id_to_hex(manif.by_name["равнина"]), "9623511e");

    let plain = file.extract_names(layer, &["равнина"]).unwrap();
    let st = matte_stats(&plain);
    assert_eq!(st.nonzero, 147);
    assert_sum_close(st.sum, 108.11782, 0.02);
    assert_eq!(st.max, 1.0);
    assert_matches_golden(
        &plain,
        &samples_root(),
        "debug_images/special_chars.exr",
        "равнина",
    );

    let slash = file
        .extract_names(layer, &["name_containing/slash"])
        .unwrap();
    let slash_st = matte_stats(&slash);
    assert_eq!(slash_st.nonzero, 148);
    assert_sum_close(slash_st.sum, 109.34287, 0.02);
    assert_matches_golden(
        &slash,
        &samples_root(),
        "debug_images/special_chars.exr",
        "name_containing/slash",
    );

    let junk = file
        .extract_names(layer, &["__missing__"])
        .unwrap();
    assert_eq!(matte_stats(&junk).sum, 0.0);
}

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn multichannel_discovers_three_cryptomattes() {
    let path = require_sample("debug_images/multichannel.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    assert_eq!(file.layers.len(), 3);
    let names: Vec<_> = file.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["crypto_asset", "crypto_material", "crypto_object"]);

    for layer in &file.layers {
        assert_eq!(layer.ranked.len(), 3);
        assert_manifest_hashes_match(&file, &layer.name);
        let junk = file
            .extract_names(layer, &["__no_such_cryptomatte_object__"])
            .unwrap();
        assert_eq!(matte_stats(&junk).sum, 0.0);
    }

    let asset = file.layer_by_name("crypto_asset").unwrap();
    let tor = file.extract_names(asset, &["toroidal_asset"]).unwrap();
    let tor_st = matte_stats(&tor);
    assert_eq!(tor_st.nonzero, 6665);
    assert_sum_close(tor_st.sum, 4459.2183, 0.05);
    assert_eq!(tor_st.max, 1.0);
    assert_matches_golden(
        &tor,
        &samples_root(),
        "debug_images/multichannel.exr",
        "toroidal_asset",
    );

    let ball = file.extract_names(asset, &["ball_asset"]).unwrap();
    let ball_st = matte_stats(&ball);
    assert_eq!(ball_st.nonzero, 4069);
    assert_sum_close(ball_st.sum, 2713.3147, 0.05);
    assert_matches_golden(
        &ball,
        &samples_root(),
        "debug_images/multichannel.exr",
        "ball_asset",
    );

    // Selecting both covers at least each alone; total ≤ sum of parts (+float slack).
    let both = file
        .extract_names(asset, &["toroidal_asset", "ball_asset"])
        .unwrap();
    let both_st = matte_stats(&both);
    assert!(both_st.sum > tor_st.sum);
    assert!(both_st.sum > ball_st.sum);
    assert!(both_st.nonzero > tor_st.nonzero);
    assert!(both_st.nonzero > ball_st.nonzero);
    assert!(both_st.sum <= tor_st.sum + ball_st.sum + 0.05);
}

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn debug_sidecar_manifest_extracts() {
    let path = require_sample("debug_images/sidecar_manifest.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    let layer = file.layer_by_name("crypto_asset").unwrap();
    assert_eq!(
        layer.manif_file.as_deref(),
        Some("sidecar_manifest.crypto_asset.json")
    );
    assert_manifest_hashes_match(&file, "crypto_asset");

    let left = file.extract_names(layer, &["leftAsset"]).unwrap();
    let left_st = matte_stats(&left);
    assert_eq!(left_st.nonzero, 147);
    assert_sum_close(left_st.sum, 107.03307, 0.02);
    assert_matches_golden(
        &left,
        &samples_root(),
        "debug_images/sidecar_manifest.exr",
        "leftAsset",
    );

    let right = file.extract_names(layer, &["rightAsset"]).unwrap();
    let right_st = matte_stats(&right);
    assert_eq!(right_st.nonzero, 148);
    assert_sum_close(right_st.sum, 109.34287, 0.02);
    assert_matches_golden(
        &right,
        &samples_root(),
        "debug_images/sidecar_manifest.exr",
        "rightAsset",
    );

    // Distinct objects → different mattes (not a constant false positive).
    assert!(left.coverage.iter().zip(right.coverage.iter()).any(|(a, b)| a.to_bits() != b.to_bits()));
}


#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn synthetic_exact_pixels_against_golden() {
    let rel = "synthetic/synth_CryptoObject.exr";
    let path = require_sample(rel);
    let file = CryptomatteFile::open(&path).expect("open synth");
    let layer = file.layer_by_name("synth").expect("synth layer");
    let a = file.extract_names(layer, &["objA"]).unwrap();
    let b = file.extract_names(layer, &["objB"]).unwrap();
    assert_eq!(a.width, 4);
    assert_eq!(a.height, 4);
    // Known layout from generator.
    assert_eq!(a.coverage[0], 1.0); // (0,0)
    assert_eq!(a.coverage[1], 0.5); // (0,1)
    assert_eq!(a.coverage[4], 0.0); // (1,0) is B
    assert_eq!(a.coverage[5], 0.25); // (1,1)
    assert_eq!(b.coverage[4], 1.0);
    assert_eq!(b.coverage[5], 0.25);
    assert_matches_golden(&a, &samples_root(), rel, "objA");
    assert_matches_golden(&b, &samples_root(), rel, "objB");
}

/// Psyop's `bunny_*` samples spell their ranked channels `.red/.green/.blue/.alpha`: an inject
/// keeps that spelling (no `.R` twins), and the objects already in the file keep their mattes.
#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn inject_keeps_the_real_samples_red_green_blue_alpha_spelling() {
    let path = require_sample("bunny_CryptoObject.exr");
    let file = CryptomatteFile::open(&path).expect("open");
    let layer = file.layer_by_name("uCryptoObject").expect("uCryptoObject");
    assert!(file.image.channels().any(|(n, _)| n == "uCryptoObject00.red"), "the sample spells .red");
    let n = usize::try_from(file.image.width() * file.image.height()).expect("size");
    let before = file.extract_names(layer, &["bunny"]).expect("bunny");

    let opts = cryptomatte::EncodeOptions::new("uCryptoObject");
    let enc = file
        .inject(layer, "nothing_here", &vec![0.0; n], cryptomatte::MergeOp::Over, &opts)
        .expect("inject");
    let out = std::env::temp_dir().join(format!("exr-rs-crypto-red-{}.exr", std::process::id()));
    enc.write(&out, None).expect("write");

    let back = CryptomatteFile::open(&out).expect("reopen");
    let names: Vec<&str> = back.image.channels().map(|(n, _)| n).collect();
    assert!(names.contains(&"uCryptoObject00.red") && names.contains(&"uCryptoObject00.alpha"), "{names:?}");
    assert!(!names.iter().any(|n| n.starts_with("uCryptoObject0") && n.ends_with(".R")), "{names:?}");
    let layer = back.layer_by_name("uCryptoObject").expect("layer");
    let after = back.extract_names(layer, &["bunny"]).expect("bunny");
    assert_mattes_bit_eq(&before, &after);
    std::fs::remove_file(&out).expect("cleanup");
}
