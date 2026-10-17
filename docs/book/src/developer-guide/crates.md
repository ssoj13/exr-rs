# Crates

Workspace members (plus the root `exr-rs` CLI package):

| Crate | Ports / built on | Role |
|---|---|---|
| `imath-rs` | Imath v3.2.0 | `Half`, vectors/matrices/boxes, color, noise, frustum. Zero `unsafe`. The foundation type layer. |
| `deflate-rs` | libdeflate v1.26 | inflate + the full compressor (all 12 levels) + zlib/gzip wrappers + crc32, with x86 SIMD tiers behind a safe scalar fallback. Byte-identical to C. Workspace-local SIMD `unsafe` is confined here; shared codec SIMD lives in its dependency. |
| `exr-core` | OpenEXR/IlmImf v3.3.11 layout (+ HTJ2K from 3.4); behaviour = C++ 3.4.15 | the container, all 12 codecs, scanline/tiled/deep/multipart readers and writers, deep compositing, the `api/` facade, `rgba/` (RGBA / tiled RGBA / ACES in+out, luminance-chroma, LUTs), `chunk/` (raw passthrough). The heart of the project. |
| `cryptomatte` | Cryptomatte 1.2 spec, on `exr-core` + murmur3 | decode (metadata, manifests, ranked matte extract) and encode (named mattes, inject over/under). |
| `exr-gpu` | `exr-core` + wgpu 30 | upload decoded channel layers into VRAM (textures or a planar storage buffer); readback. |
| `exr-view` | `exr-core` + winit / egui 0.36 / egui_dock 0.21 / wgpu 30 | the GPU viewer behind `exr-rs view`: resident and tiled display, OCIO, HDR output, scopes, layer selection, Cryptomatte overlay, sequence playback, OpenFX effects. |
| `exr-util` | OpenEXRUtil | in-memory flat and deep images, whole-file image I/O and `CheckFile`, used by the whole-image CLI tools. |
| `exr-image` | `exr-core` + the `image` crate | EXR decoding for `image` through exr-core (`register()` installs the hooks). |
| `exr-import` | `exr-core`, `vfx-io`, `exv-video` | Non-EXR stills (PNG, DPX, HDR, ...) and `.exv` clips (`ExvClip`, `write_exr_frames_as_exv`) as scene-linear `exr_core::api::Image`; used by the viewer and `convert`. |
| `exr-test-data` | test infrastructure | resolves opt-in private fixtures via `EXR_TEST_DATA`. |
| `rans-rs` | 32-stream interleaved rANS | 12-bit scale rANS entropy codec (zero `unsafe`). |
| `exv-format` | `exr-core`, `imath-rs` | `.exv` wire format (`Header`, `Meta::exr_headers` TLV tag `6`, `TileIndex`, `TAG_BC6H`, `bufpool`, `fsio`, `crc32c`). |
| `exv-tile` | `exv-format` | 2D tile grid, byte-plane `split_delta`/`unsplit_delta`, LeGall 5/3 & CDF 9/7 DWT, reversible colour transform (RCT). |
| `exv-codec` | `exv-tile`, `rans-rs`, `jph-rs` | `.exv` CPU tile codecs: `BC6H` (`Bc6hMode::{Lossy, LosslessResidual}`), `HTJ2K`, `rANS`, `Wavelet`, `EXV-RT`, temporal `PRED`. |
| `exv-video` | `exv-codec` | multi-frame `.exv` container (`FrameIndex`, `FrameToc`, `VideoIndex`, `GopCache`, memory-mapped `open_bytes`). |
| `exv-gpu-kernels` | `exv-format`, `wgpu 30` | WGSL compute kernels for `.exv` (`BC6H`, `rANS`, `DWT`, `EXV-RT`, `PRED`, `RCT`, `F16` texture pack). |
| `exv-gpu` | `exv-gpu-kernels`, `exv-video` | GPU tile/frame decoder & VRAM cache (`GpuTileCache`, `FrameAssembler`, `decode_bc6h_batched`). |
| `exv-capi` | `exv-video`, `exv-gpu` | C ABI (`exv_capi` + `docs/exv-capi.h` via `cbindgen`) for native host plug-ins (`cpp/exv-ae` After Effects `ExvAE.aex`). |

`xtask` is also a workspace member. `cargo xtask <command>` with `setup`,
`parity`, `parity-deflate`, `parity-exr`, `parity-header` or `parity-tools` builds the
native oracles through one vcpkg resolver.

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
| `oiio-rs` (git, package `vfx-ocio`) | Rust OCIO/colour-math components used by viewer transforms. |
| `ofx-rs` (git, optional `ofx`) | runtime and host UI; the built-in effects load as the `ofx-effects.ofx` bundle, not linked. |
| `egui-widgets-rs` (git) | the shared egui widgets of the viewer: timeline, status bar, outliner, attribute grid, HUD, preferences, display/present helpers. |
| `murmur3` (git) | MurmurHash3 x86_32 and x64_128: Cryptomatte IDs and the ID manifest hash schemes in `exr-core`. |

The root `src/` is the [exr-rs CLI](../user-guide/cli.md), including the whole-image tools.

## Why so many crates

Each upstream library is ported into its own crate so the 1:1 mapping stays obvious
and each piece is independently testable and reusable. Pieces shared with other
projects (HTJ2K, sequence detection, the frame cache, GPU info, the MSVC helper) live
in their own repos and are consumed by git-ref.
