//! Integration test: open a REAL multipart EXR (the committed parity oracle
//! golden) and check exr-gpu enumerates its image parts as `Part`s with layers.
//! CPU-only (no GPU) — exercises decode + part split + layer grouping end to end.
//!
//! `golden_multipart.exr` (written by the C++ oracle in the parity gate) has THREE
//! parts: scanline/NONE, tiled/ZIP, deep-scanline/ZIPS. exr-gpu maps the two image
//! parts to `Part`s and skips the deep one.

use std::path::PathBuf;

/// Path to the committed multipart golden, relative to this crate.
fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../exr-core/tests/parity/golden_multipart.exr")
}

#[test]
fn opens_multipart_image_parts_and_skips_deep() {
    let path = golden();
    if !path.exists() {
        eprintln!("skip: {} not present", path.display());
        return;
    }

    let loader = exr_gpu::GpuLoader::open(&path).expect("open multipart golden");

    // Two of the three parts are image parts (scanline + tiled); the deep part is
    // skipped. (A bump here would mean the deep-skip or part split regressed.)
    assert_eq!(
        loader.parts().len(),
        2,
        "expected 2 image parts (deep part skipped), got {}",
        loader.parts().len()
    );

    for part in loader.parts() {
        assert!(part.width() > 0 && part.height() > 0, "part has zero extent");
        assert!(!part.layers().is_empty(), "image part has no layers");
        // The default layer must be uploadable shape-wise (1..=4 same-type channels
        // for the KNOWN B/G/R image); we don't touch the GPU here.
        assert!(part.default_layer().is_some(), "no default layer");
    }
}
