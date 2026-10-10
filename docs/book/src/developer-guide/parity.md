# Parity and testing

The native reference for disputed EXR behaviour is OpenEXR 3.4.15.
Byte-identity claims apply to the covered cases and pinned oracle/toolchain,
rather than every possible input or platform.

## Choose the gate

```sh
cargo test --workspace
python bootstrap.py t
python bootstrap.py t cargo
python bootstrap.py t exr python
```

Plain Cargo tests leave `test-data` feature-gated fixtures and ignored hardware
tests unexecuted. The bootstrap full gate enables private fixtures and runs six
stages: Cargo workspace tests (`cargo`), clippy on the workspace and `openexr-py`
(`clippy`), Imath parity (`parity`), libdeflate parity (`deflate`), OpenEXR parity
(`exr`) and the maturin/pytest Python gate (`python`). A bare `t` runs that gate; it does not print help.
Use `python bootstrap.py help` for the wrapper's command reference.

Tests that require enabled fixtures or an oracle fail without them; the explicit
`EXR_ALLOW_SKIP=1` opt-out turns those failures into skips.

## Native oracles

| Command | Reference | Purpose |
|---|---|---|
| `cargo run -p xtask -- parity` | Imath 3.2.2 | Bit/ULP golden checks for the Imath 3.2.0 port |
| `cargo run -p xtask -- parity-deflate` | libdeflate 1.26 | Byte identity across compression levels |
| `cargo run -p xtask -- parity-exr` | OpenEXR 3.4.15 (108 tests) | C++→Rust and Rust→C++ read/write comparisons |
| `cargo run -p xtask -- parity-header` | C++ exrheader | Header formatter comparisons |
| `cargo run -p xtask -- parity-tools` | C++ image tools | CLI/tool behaviour comparisons |

The native build path uses Windows/MSVC and vcpkg. On other platforms xtask reports
a skip and returns without building the native oracle; that is not a native pass.
Tests backed by committed goldens can still run independently.

`vcv-rs` supplies MSVC/SDK/UCRT settings. One vcpkg resolver handles headers,
static/DLL libraries, CRT choice and DLL search paths:
`VCPKG_INSTALLED` selects the installed triplet directly; otherwise
`VCPKG_ROOT/installed` (default `C:/vcpkg/installed`) is searched for a matching
`x64-windows*` triplet. Do not duplicate that resolver in a new gate.

The EXR oracle covers scanline, tiled, deep and multipart read/write, codecs,
RGBA/YCA, ACES, metadata and geometry boundaries. Compressed goldens assert that
they actually compress. Lossy decoded pixels are compared with the C++ result,
not with the original uncompressed source.

DWA byte identity is pinned to the MSVC-x64 SSE2 DCT operation order; other C++
DCT variants can differ. HTJ2K requires OpenJPH 0.31.0 parity through `jph-rs`,
including the version embedded in the codestream. Set `EXR_ORACLE_EXE` to the
actual oracle when it is outside the normal generated location.
Per-gate details remain in each crate's `tests/parity/README.md`.

## Fixtures

`exr-test-data` resolves `EXR_TEST_DATA`, otherwise a sibling `exr-rs-tests`
checkout. It contains the pinned OpenEXR test images, openexr-images and
Cryptomatte fixtures/goldens. With `test-data` enabled, a missing checkout fails.
The corpus test additionally supports `EXR_CORPUS_DIR` for its OpenEXR corpus.

No gate reads upstream source trees. The oracles link the vcpkg libraries pinned
in `xtask/vcpkg.json`; `parity-header` and `parity-tools`, which compile
OpenEXR's own tools, fetch the pinned v3.4.15 sources into
`target/ref/openexr-v3.4.15` themselves (see the root README's "Reference
sources"). Fixtures come only from the test-data crate.

## GPU and viewer checks

Ignored GPU tests must be run on an actual device:

```sh
cargo test -p exr-gpu -- --ignored
```

Viewer tests additionally cover geometry, readiness, cancellation, frame retention
during playback, budgets, and GPU and fixture cases. Shader validation and readback
checks do not replace looking at the running application.

## Fast paths and measurements

Every parallel, SIMD or reuse fast path has an engagement test (`exr_core::probe`):
output equality alone also passes on a silent fallback. Run gates on a clean checkout
of the commit being tested; compare timings only with identical inputs and thread
counts on a quiet machine.
