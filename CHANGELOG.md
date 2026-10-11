# Changelog

All notable changes to **exr-rs** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **exr-view**: the Effects tab is ofx-host-egui's `EffectStack`; the effect's render-site picker is ofx-runtime's
  `SiteMenu` (GPU sites listed and refused with the reason: the viewer renders on the CPU site) and the status shows
  the delivery site with the compute backend the plug-in reported ("CPU images, WGPU compute") after each render.

## [0.1.0] — 2026-10-10

Initial public release.

- **exr-core**: a pure-Rust port of OpenEXR (behaviour of C++ 3.4.15): scanline, tiled
  (mip/rip), deep, multipart and multiview files, all twelve compression codecs, the RGBA/ACES/YCA
  layers, raw-chunk copy and an idiomatic `api` facade. ZIP, PIZ, PXR24, B44, DWA and HTJ2K writes
  are byte-identical to C++ 3.4.15, and every codec is pixel-exact in both directions against a
  real C++ oracle.
- **imath-rs** (Imath 3.2.0) and **deflate-rs** (libdeflate 1.26, byte-identical, with SSE2/AVX2/
  PCLMUL/NEON tiers) ports; **exr-util** (OpenEXRUtil) and the OpenEXR command-line tools as
  `exr-rs` subcommands.
- **cryptomatte**: Cryptomatte decode, encode and inject.
- **exr-view**: a GPU viewer (wgpu, egui) for images and sequences with SDR, HDR10 PQ, HLG and
  scRGB output, OCIO display transforms, scopes, a tiled display pyramid, OpenFX effects and a
  persisted viewer backdrop.
- **openexr-py**: a drop-in replacement for the `OpenEXR` Python module (CPython 3.8+, abi3),
  passing the official OpenEXR Python tests.
- Release ZIPs for Windows x86_64, Linux x86_64 and macOS aarch64, and Python wheels for five
  platforms. Each ported crate keeps its original's license; `THIRD-PARTY-LICENSES.html` in the
  ZIP lists every crate in the binary.

[Unreleased]: https://github.com/ssoj13/exr-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ssoj13/exr-rs/releases/tag/v0.1.0
