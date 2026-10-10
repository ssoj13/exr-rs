//! Shared test-support for deflate-rs's live-oracle integration tests
//! (`parity_libdeflate_large.rs` / `parity_libdeflate_malformed.rs`).
//!
//! Modelled on `crates/exr-core/tests/support/mod.rs`'s convention: a skip is
//! never silent (`EXR_ALLOW_SKIP=1` opt-out only, same env var the rest of the
//! workspace uses), and the oracle path resolves through one function so both
//! test binaries agree on where to look.
//!
//! # Why this exists
//! `parity_libdeflate_large.rs` and `parity_libdeflate_malformed.rs` used to
//! each carry their own copy of `oracle_path`/`oracle_or_skip`/`lcg_bytes` and
//! the subprocess spawn/write/wait boilerplate. Cargo does not share `tests/` modules
//! across crates, but it DOES share them within one crate's `tests/`
//! directory via `mod support;` from each test binary — there was no reason
//! for the duplication within deflate-rs itself.

#![allow(dead_code)] // each test binary uses a different subset

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The one opt-out env var for every skip in the workspace.
pub const ALLOW_SKIP: &str = "EXR_ALLOW_SKIP";

/// Whether the operator explicitly allowed tests to no-op on missing inputs.
pub fn skip_allowed() -> bool {
    std::env::var_os(ALLOW_SKIP).is_some_and(|v| v == "1")
}

/// Path to the oracle binary `cargo xtask parity-deflate` builds (gitignored).
///
/// Resolution order:
/// 1. `EXR_DEFLATE_ORACLE_EXE` (set by `xtask`'s `parity_deflate` when it
///    wires these tests into the gate) — an empty value falls through to
///    discovery, same convention as exr-core's `EXR_ORACLE_EXE`.
/// 2. `crates/deflate-rs/tests/parity/oracle.exe`, next to the golden file.
pub fn oracle_path() -> PathBuf {
    if let Some(p) = std::env::var_os("EXR_DEFLATE_ORACLE_EXE")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("oracle.exe")
}

/// The oracle binary, or `None` when skipping was allowed.
///
/// Panics with an actionable message unless `EXR_ALLOW_SKIP=1` — a live-oracle
/// parity test that silently did not run proves nothing.
pub fn oracle_or_skip(name: &str) -> Option<PathBuf> {
    let p = oracle_path();
    if p.exists() {
        return Some(p);
    }
    let msg = format!(
        "{name}: {} not found -- run `cargo run -p xtask -- parity-deflate` once \
         (Windows + vcpkg libdeflate) to build it, or set {ALLOW_SKIP}=1 to skip \
         this test deliberately.",
        p.display()
    );
    if skip_allowed() {
        eprintln!("SKIP ({ALLOW_SKIP}=1) {msg}");
        return None;
    }
    panic!("{msg}");
}

/// Deterministic LCG byte stream (avoids a `rand` dependency); shared formula
/// with the in-crate differential tests (`deflate_decompress.rs`'s
/// `lcg_bytes`) though defined separately since those live in the library
/// crate, not `tests/`.
pub fn lcg_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut rng = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
        out.push((rng >> 16) as u8);
    }
    out
}

/// Run `oracle.exe zc <level>` with `input` piped to stdin; returns the raw
/// `libdeflate_zlib_compress` output from stdout.
pub fn oracle_zlib_compress(oracle: &Path, level: i32, input: &[u8]) -> Vec<u8> {
    let mut child = Command::new(oracle)
        .args(["zc", &level.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn oracle.exe");
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(input)
        .expect("write to oracle stdin");
    let output = child.wait_with_output().expect("oracle wait failed");
    assert!(
        output.status.success(),
        "oracle zc {level} failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Run `oracle.exe zdx <out_cap>` with `compressed` piped to stdin. Returns
/// `(result_code, actual_in, actual_out)` per `enum libdeflate_result`; the
/// latter two are `-1` (the oracle's sentinel) unless `result_code ==
/// LIBDEFLATE_SUCCESS` (0).
pub fn oracle_zlib_decompress_ex(oracle: &Path, out_cap: usize, compressed: &[u8]) -> (i64, i64, i64) {
    let mut child = Command::new(oracle)
        .args(["zdx", &out_cap.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn oracle.exe");
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(compressed)
        .expect("write to oracle stdin");
    let output = child.wait_with_output().expect("oracle wait failed");
    assert!(
        output.status.success(),
        "oracle zdx {out_cap} failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8_lossy(&output.stdout);
    let line = line.trim();
    let mut result = None;
    let mut actual_in = None;
    let mut actual_out = None;
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("result=") {
            result = v.parse::<i64>().ok();
        } else if let Some(v) = tok.strip_prefix("actual_in=") {
            actual_in = v.parse::<i64>().ok();
        } else if let Some(v) = tok.strip_prefix("actual_out=") {
            actual_out = v.parse::<i64>().ok();
        }
    }
    (
        result.unwrap_or_else(|| panic!("no result= in oracle output: {line:?}")),
        actual_in.unwrap_or_else(|| panic!("no actual_in= in oracle output: {line:?}")),
        actual_out.unwrap_or_else(|| panic!("no actual_out= in oracle output: {line:?}")),
    )
}

/// One (median_ns, min_ns) timing result from `oracle.exe bench {zc|zd}`.
#[derive(Debug, Clone, Copy)]
pub struct BenchStats {
    pub median_ns: i64,
    pub min_ns: i64,
}

/// Run `oracle.exe bench {zc|zd} <level> <reps>` with `input` piped to
/// stdin. `sub_mode` is `"zc"` or `"zd"`. See `oracle.c`'s `bench` mode doc
/// comment for exactly what each times (compress-only vs. decompress-only,
/// isolated from any exr-core-side predictor/interleave/I-O cost).
pub fn oracle_bench(oracle: &Path, sub_mode: &str, level: i32, reps: u32, input: &[u8]) -> BenchStats {
    let mut child = Command::new(oracle)
        .args(["bench", sub_mode, &level.to_string(), &reps.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn oracle.exe");
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(input)
        .expect("write to oracle stdin");
    let output = child.wait_with_output().expect("oracle wait failed");
    assert!(
        output.status.success(),
        "oracle bench {sub_mode} {level} {reps} failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8_lossy(&output.stdout);
    let line = line.trim();
    let mut median_ns = None;
    let mut min_ns = None;
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("median_ns=") {
            median_ns = v.parse::<i64>().ok();
        } else if let Some(v) = tok.strip_prefix("min_ns=") {
            min_ns = v.parse::<i64>().ok();
        }
    }
    BenchStats {
        median_ns: median_ns.unwrap_or_else(|| panic!("no median_ns= in oracle output: {line:?}")),
        min_ns: min_ns.unwrap_or_else(|| panic!("no min_ns= in oracle output: {line:?}")),
    }
}
