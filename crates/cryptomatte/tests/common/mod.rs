//! The Cryptomatte samples and their pixel-matte goldens (made by
//! `scripts/fetch_cryptomatte_samples.py`), from the test image set (`exr-test-data`).

use std::fs;
use std::path::{Path, PathBuf};

use cryptomatte::Matte;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // fields kept for schema parity with goldens/index.json
pub struct GoldenIndexEntry {
    pub exr: String,
    pub name: String,
    #[serde(default)]
    pub layer: Option<String>,
    pub stem: String,
    pub json: String,
    pub bin: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // sha256/exr/name present for golden metadata round-trip
pub struct GoldenMeta {
    pub exr: String,
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub sha256: String,
}

/// The sample root: `CRYPTOMATTE_SAMPLES` (another sample set, e.g. freshly
/// generated), else the test image set's `cryptomatte/`.
pub fn samples_root() -> PathBuf {
    match std::env::var_os("CRYPTOMATTE_SAMPLES") {
        Some(p) => PathBuf::from(p),
        None => exr_test_data::cryptomatte(),
    }
}

pub fn require_sample(rel: &str) -> PathBuf {
    let p = samples_root().join(rel);
    assert!(p.is_file(), "missing sample {}", p.display());
    p
}

pub fn load_index(root: &Path) -> Vec<GoldenIndexEntry> {
    let path = root.join("goldens").join("index.json");
    let text = fs::read_to_string(&path).unwrap_or_default();
    if text.is_empty() {
        return Vec::new();
    }
    serde_json::from_str(&text).expect("goldens/index.json")
}

pub fn load_golden(root: &Path, exr_rel: &str, name: &str) -> Option<(GoldenMeta, Vec<f32>)> {
    let exr_rel = exr_rel.replace('\\', "/");
    let entry = load_index(root).into_iter().find(|e| e.exr == exr_rel && e.name == name)?;
    let meta_path = root.join(&entry.json);
    let bin_path = root.join(&entry.bin);
    let meta: GoldenMeta = serde_json::from_str(&fs::read_to_string(&meta_path).ok()?).ok()?;
    let bytes = fs::read(&bin_path).ok()?;
    assert_eq!(
        bytes.len(),
        meta.width.checked_mul(meta.height)?.checked_mul(4)?,
        "golden bin size mismatch for {exr_rel}::{name}"
    );
    let mut pixels = Vec::with_capacity(meta.width * meta.height);
    for chunk in bytes.as_chunks::<4>().0 {
        pixels.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some((meta, pixels))
}

pub fn assert_matches_golden(matte: &Matte, root: &Path, exr_rel: &str, name: &str) {
    let (meta, expected) = load_golden(root, exr_rel, name).unwrap_or_else(|| {
        panic!(
            "golden missing for {exr_rel}::{name}; re-run scripts/fetch_cryptomatte_samples.py --write-goldens"
        )
    });
    assert_eq!(matte.width, meta.width);
    assert_eq!(matte.height, meta.height);
    assert_eq!(matte.coverage.len(), expected.len());
    for (i, (got, exp)) in matte.coverage.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "pixel mismatch at {i} for {exr_rel}::{name}: got={got} expected={exp}"
        );
    }
}
