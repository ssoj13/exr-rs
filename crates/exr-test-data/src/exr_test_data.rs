//! Where exr-rs's test image set is: OpenEXR's own `src/test` images (v3.4.15), openexr-images
//! (e38ffb0) and the Cryptomatte samples with their goldens, kept in the separate `exr-rs-tests`
//! checkout (not a cargo dependency: nobody has to download it to build or test the crates).
//!
//! Tests that read it are off by default: each crate gates them behind its `test-data` feature
//! (`#[cfg_attr(not(feature = "test-data"), ignore = ...)]`), and `bootstrap.py t` turns the
//! feature on. With the feature on, missing data is a failure, not a skip.

use std::path::{Path, PathBuf};

/// Environment variable that points at the data checkout.
pub const ENV: &str = "EXR_TEST_DATA";

/// The data root: `EXR_TEST_DATA`, else `exr-rs-tests` next to this workspace. Panics with where
/// it looked when neither exists (only data tests call it).
pub fn root() -> PathBuf {
    let root = match std::env::var_os(ENV) {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../exr-rs-tests"),
    };
    assert!(
        root.join("openexr-images").is_dir(),
        "exr-rs test data not found at {} (set {ENV} to an exr-rs-tests checkout)",
        root.display()
    );
    root
}

/// A file or directory under [`root`], by its `/`-separated relative path.
pub fn path(rel: &str) -> PathBuf {
    root().join(rel)
}

/// OpenEXR's `src/test` images (v3.4.15), the `.exr` files only.
pub fn openexr_test() -> PathBuf {
    path("openexr/src/test")
}

/// The openexr-images repository (e38ffb0).
pub fn openexr_images() -> PathBuf {
    path("openexr-images")
}

/// Cryptomatte samples (Psyop 1.4.0) with the pixel goldens and the synthetic sample.
pub fn cryptomatte() -> PathBuf {
    path("cryptomatte")
}
