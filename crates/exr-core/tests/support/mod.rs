//! Shared test-support for the integration tests — the SINGLE place that
//! decides whether a missing fixture, oracle or GPU may turn a test into a
//! no-op.
//!
//! # Why this exists
//! A gate that reports success because it never ran is worse than no gate. Most
//! of this suite's C++-parity value used to evaporate silently: a `return` on a
//! missing fixture reads as PASSED in cargo's output, so a green
//! `cargo test --workspace` proved far less than it looked like. Every skip now
//! goes through here and **panics with an actionable message** unless the
//! operator explicitly opted out with `EXR_ALLOW_SKIP=1` (the same env var
//! `imath-rs/tests/parity_golden.rs` and `deflate-rs/tests/parity_libdeflate.rs`
//! use — one name for the whole workspace).
//!
//! # Where it is used
//! `mod support;` from each exr-core integration test, and — because Cargo does
//! not share `tests/` modules across crates — via an explicit
//! `#[path = "../../exr-core/tests/support/mod.rs"] mod support;` from
//! `exr-gpu/tests/`. Keep this module **std-only** so that include keeps
//! working from any crate.

#![allow(dead_code)] // each test binary uses a different subset

use std::path::{Path, PathBuf};

/// The one opt-out env var for every skip in the workspace.
pub const ALLOW_SKIP: &str = "EXR_ALLOW_SKIP";

/// Whether the operator explicitly allowed tests to no-op on missing inputs.
pub fn skip_allowed() -> bool {
    std::env::var_os(ALLOW_SKIP).is_some_and(|v| v == "1")
}

/// Common tail of every skip/failure message.
fn opt_out_note() -> String {
    format!("Set {ALLOW_SKIP}=1 to skip it deliberately.")
}

/// Resolve a required fixture file or directory.
///
/// Returns `Some(path)` when it exists. When it does not, this **panics** with
/// `what` + `how_to_get` — unless `EXR_ALLOW_SKIP=1`, in which case it prints
/// the same text and returns `None` so the caller can `return`.
pub fn fixture_or_skip(what: &str, path: impl AsRef<Path>, how_to_get: &str) -> Option<PathBuf> {
    let path = path.as_ref();
    if path.exists() {
        return Some(path.to_path_buf());
    }
    let msg = format!(
        "{what}: fixture {} not found -- this test did NOT run. {how_to_get} {}",
        path.display(),
        opt_out_note()
    );
    if skip_allowed() {
        eprintln!("SKIP ({ALLOW_SKIP}=1) {msg}");
        return None;
    }
    panic!("{msg}");
}

/// Same as [`fixture_or_skip`] for a condition that is not a path (a GPU
/// adapter, a CPU feature, a tier selection...).
pub fn require_or_skip(what: &str, ok: bool, how_to_get: &str) -> bool {
    if ok {
        return true;
    }
    let msg = format!("{what}: precondition not met -- this test did NOT run. {how_to_get} {}", opt_out_note());
    if skip_allowed() {
        eprintln!("SKIP ({ALLOW_SKIP}=1) {msg}");
        return false;
    }
    panic!("{msg}");
}

/// The C++ OpenEXR oracle executable, or `None` when skipping was allowed.
///
/// Resolution order:
/// 1. `EXR_ORACLE_EXE` (set by `cargo run -p xtask -- parity-exr` and by
///    `python bootstrap.py t`);
/// 2. the oracle `xtask` builds next to the goldens
///    (`crates/exr-core/tests/parity/oracle.exe`) — gitignored, but present on
///    any machine that has run the parity gate once. It links the vcpkg
///    OpenEXR statically on this project's triplet, so it needs no extra PATH.
///
/// Anything else panics: a "C++ cross-read" test that silently did not ask C++
/// anything is exactly the defect this module exists to remove.
pub fn oracle_or_skip(name: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("EXR_ORACLE_EXE").map(PathBuf::from) {
        if p.as_os_str().is_empty() {
            // fall through to discovery
        } else if p.exists() {
            return Some(p);
        } else {
            return fixture_or_skip(
                name,
                &p,
                "EXR_ORACLE_EXE points at a file that does not exist; run \
                 `cargo run -p xtask -- parity-exr` to build it.",
            );
        }
    }
    let discovered = default_oracle();
    fixture_or_skip(
        name,
        &discovered,
        "The C++ OpenEXR oracle is missing. Run `cargo run -p xtask -- parity-exr` \
         (Windows + vcpkg OpenEXR) to build it, or set EXR_ORACLE_EXE.",
    )
}

/// Where `xtask` puts the oracle it builds, resolved from the *consuming*
/// crate so the `#[path]`-included copy in `exr-gpu` also finds it.
fn default_oracle() -> PathBuf {
    let exe = if cfg!(windows) { "oracle.exe" } else { "oracle" };
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let own = manifest.join("tests").join("parity").join(exe);
    if own.exists() {
        return own;
    }
    manifest.join("..").join("exr-core").join("tests").join("parity").join(exe)
}

// ---------------------------------------------------------------------------
// Golden chunk-size assertions
// ---------------------------------------------------------------------------

/// A golden must contain at least one chunk this big, so a trivially small
/// fixture cannot satisfy the gate below.
const SHRINK_FLOOR: usize = 1024;

/// Assert that a committed golden really exercises its codec.
///
/// The reason this exists: an audit found that `golden_piz.exr`,
/// `golden_dwaa.exr`, `golden_dwab.exr` and both HTJ2K goldens stored **every**
/// chunk raw (490 of 490 bytes), because the 7x5 KNOWN image was too small to
/// compress. The writer's "no shrink -> store raw" rule meant the tests named
/// `..._byte_identical_to_cpp_golden` proved only the header and the raw-store
/// path; the codecs had never been compared with C++ at all. Nothing detected
/// that for months, so a golden must now prove it is compressed.
///
/// A chunk counts as compressed when its stored payload is strictly smaller
/// than the unpacked pixel block it stands for — exactly the condition
/// `compressor::stored_chunk` uses to decide whether to store raw.
///
/// The rule is deliberately **not** "every chunk shrank": a small edge tile
/// genuinely does not compress under PIZ (its Huffman table alone outweighs a
/// 4x32 tile), and demanding it would push the fixture towards tile-aligned
/// dimensions, losing the partial-tile coverage. What a working codec on real
/// data always satisfies, and a raw-stored golden never does, is:
///
/// * the file compresses **in total**, and
/// * the **largest** chunk — the bulk of the image — is compressed.
///
/// Supports single-part flat scanline and tiled files: the shapes the goldens
/// use. Deep and multipart goldens are lossless-only and are not gated here.
pub fn assert_chunks_compressed(path: &Path) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("assert_chunks_compressed: read {}: {e}", path.display()));
    let mut is = exr_core::MemIStream::new(bytes);
    let img = exr_core::RawImage::read(&mut is)
        .unwrap_or_else(|e| panic!("assert_chunks_compressed: parse {}: {e}", path.display()));
    assert_eq!(img.parts.len(), 1, "{}: expected a single-part file", path.display());
    let part = &img.parts[0];
    let raws = raw_chunk_sizes(path, part);

    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let stored_total: usize = part.chunks.iter().map(|c| c.data.len()).sum();
    let raw_total: usize = raws.iter().sum();
    let n_raw = part.chunks.iter().zip(&raws).filter(|(c, r)| c.data.len() >= **r).count();

    assert!(
        raws.iter().any(|r| *r >= SHRINK_FLOOR),
        "{name}: every chunk is under {SHRINK_FLOOR} raw bytes -- too small to prove anything"
    );
    assert!(
        stored_total < raw_total,
        "{name}: stored {stored_total} bytes for {raw_total} raw -- the codec never shrank          anything, so this golden tests only the raw-store path. Enlarge the KNOWN image in          tests/parity/oracle.cpp rather than accepting it."
    );

    eprintln!(
        "{name}: {} chunks, {stored_total} stored / {raw_total} raw ({n_raw} stored raw)",
        part.chunks.len()
    );

    // The biggest chunk carries the bulk of the image; if the codec could not
    // shrink THAT, it did not meaningfully run.
    let (i, biggest) = raws
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| **r)
        .unwrap_or_else(|| panic!("{name}: no chunks"));
    assert!(
        part.chunks[i].data.len() < *biggest,
        "{name}: the LARGEST chunk {:?} was stored RAW ({} bytes of {biggest}) -- this golden          does not exercise its codec on real data ({n_raw} of {} chunks are raw).",
        part.chunks[i].coord,
        part.chunks[i].data.len(),
        part.chunks.len()
    );
}

/// Unpacked byte size of each chunk of a single-part flat file, in the same
/// order as `part.chunks`.
fn raw_chunk_sizes(path: &Path, part: &exr_core::RawPart) -> Vec<usize> {
    use exr_core::chunk::ChunkCoord;

    let h = &part.header;
    let chans = h.channels().expect("channels");
    let mut bpp = 0usize;
    for (n, c) in chans.iter() {
        assert_eq!(
            (c.x_sampling, c.y_sampling),
            (1, 1),
            "{}: channel {n} is subsampled; the raw-size math here assumes 1x1",
            path.display()
        );
        bpp += exr_core::misc::pixel_type_size(c.type_).expect("pixel size");
    }
    let dw = h.data_window().expect("dataWindow");
    let width = (dw.max.x - dw.min.x + 1) as usize;

    match part.chunks.first().map(|c| c.coord) {
        Some(ChunkCoord::Scanline { .. }) => {
            // The block height of a chunk is the gap to the next chunk's y
            // (and the remainder of the data window for the last one), so this
            // needs no per-codec "scanline lines" table.
            let ys: Vec<i32> = part
                .chunks
                .iter()
                .map(|c| match c.coord {
                    ChunkCoord::Scanline { y } => y,
                    other => panic!("mixed chunk kinds: {other:?}"),
                })
                .collect();
            let mut sorted = ys.clone();
            sorted.sort_unstable();
            ys.iter()
                .map(|y| {
                    let next = sorted
                        .iter()
                        .copied()
                        .find(|n| *n > *y)
                        .unwrap_or(dw.max.y + 1);
                    (next - y) as usize * width * bpp
                })
                .collect()
        }
        Some(ChunkCoord::Tile { .. }) => {
            let mut is = exr_core::StdIfStream::open(path.to_str().expect("utf-8 path"))
                .expect("open tiled golden");
            let tif = exr_core::tiled::TiledInputFile::new(&mut is).expect("TiledInputFile");
            let (tw, th) = (tif.tile_x_size() as i32, tif.tile_y_size() as i32);
            part.chunks
                .iter()
                .map(|c| match c.coord {
                    ChunkCoord::Tile { dx, dy, lx, ly } => {
                        // Edge tiles are clipped to the level's data window.
                        let lvl = tif.data_window_for_level(lx, ly).expect("level window");
                        let lw = lvl.max.x - lvl.min.x + 1;
                        let lh = lvl.max.y - lvl.min.y + 1;
                        let w = (lw - dx * tw).min(tw).max(0) as usize;
                        let h = (lh - dy * th).min(th).max(0) as usize;
                        w * h * bpp
                    }
                    other => panic!("mixed chunk kinds: {other:?}"),
                })
                .collect()
        }
        None => panic!("{}: file has no chunks", path.display()),
    }
}
