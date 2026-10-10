# Crates

Workspace members (plus the root `exr-rs` CLI package):

| Crate | Ports / built on | Role |
|---|---|---|
| `imath-rs` | Imath v3.2.0 | `Half`, vectors/matrices/boxes, color, noise, frustum. Zero `unsafe`. The foundation type layer. |
| `deflate-rs` | libdeflate v1.26 | inflate + the full compressor (all 12 levels) + zlib/gzip wrappers + crc32, with x86 SIMD tiers behind a safe scalar fallback. Byte-identical to C. Workspace-local SIMD `unsafe` is confined here; shared codec SIMD lives in its dependency. |
| `exr-core` | OpenEXR/IlmImf v3.3.11 layout (+ HTJ2K from 3.4); behaviour = C++ 3.4.15 | the container, all 12 codecs, scanline/tiled/deep/multipart readers and writers, deep compositing, the `api/` facade, `rgba/` (RGBA / tiled RGBA / ACES in+out, luminance-chroma, LUTs), `chunk/` (raw passthrough). The heart of the project. |
| `cryptomatte` | Cryptomatte 1.2 spec, on `exr-core` + murmur3 | decode (metadata, manifests, ranked matte extract) and encode (named mattes, inject over/under). |
| `exr-gpu` | `exr-core` + wgpu 30 | upload decoded channel layers into VRAM (textures or a planar storage buffer); readback. |
| `exr-view` | `exr-core` + winit / egui 0.36 / egui_dock 0.21 / wgpu 30 | the GPU viewer: colour pass, scopes, layer selection, Cryptomatte overlay, sequence playback. |
| `exr-util` | OpenEXR whole-image tools | checker, tiled/preview/attribute/multipart/multiview/environment/metrics/ACES helpers. |
| `exr-test-data` | test infrastructure | resolves opt-in private fixtures via `EXR_TEST_DATA`. |

Reusable viewer widgets (git dependencies, not workspace members):

| Crate | Built on | Role |
|---|---|---|
| `egui-timeline` | egui 0.36 | reusable time-track / scrubber widget with per-slot tint (cache state). |
| `egui-statusbar` | egui 0.36 | reusable sectioned, resizable status bar. |

`xtask` is also a workspace member. `cargo xtask {setup, parity, parity-exr,
parity-deflate}` builds the native oracles through one vcpkg resolver.

Outside the workspace:

| Crate / repo | Role |
|---|---|
| `crates/openexr-py` | drop-in `OpenEXR` Python module (PyO3 + numpy, abi3). Built only by maturin — its `extension-module` feature does not link libpython, which would break `cargo test --workspace`. |
| `jph-rs` (git) | OpenJPH 0.31.0 port (`Result`-based, no panics): the HTJ2K codestream behind EXR's HTJ2K256/HTJ2K32 codecs. |
| `scanseq` (git) | filename sequence detection (`name.####.ext`, padding- and gap-aware). |
| `scancache` (git) | frame-cache decode engine: pull pool, byte budget, playhead-ranked prefetch window and eviction, scrub mode, decode cancellation. |
| `gpu-info-rs` (git, imported as `wgpu-info`) | adapter / limit / VRAM query and a max-limits device request. |
| `vcv-rs` (git) | detects the MSVC / Windows SDK / UCRT environment so the oracles build without manual `vcvars`. |
| `codec-simd` (git) | shared safe SIMD codec primitives with scalar references. |
| `oiio-rs` (git) | Rust OCIO/colour-math components used by viewer transforms. |
| `ofx-rs` (git, optional `ofx`) | runtime, host UI and native/built-in OpenFX effect integration. |
| `murmur3` (git) | MurmurHash3 x86_32 for Cryptomatte IDs. |

The root `src/` is the [exr-rs CLI](../user-guide/cli.md), including the whole-image tools.

## Why so many crates

Each upstream library is ported into its own crate so the 1:1 mapping stays obvious
and each piece is independently testable and reusable. Pieces shared with other
projects (HTJ2K, sequence detection, the frame cache, GPU info, the MSVC helper) live
in their own repos and are consumed by git-ref.
