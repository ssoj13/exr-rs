# Changelog

All notable changes to **exr-rs** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### 2026-10-10

- **exr-view**: the Colour dock tab is gone; the OCIO controls live in Settings > Colour only (one home,
  not two copies). A saved layout that still names the tab is replaced by the default once.
- **Any colour tag opens.** `exr-import` resolves a still's colour-space tag through `vfx-io`'s
  table instead of a short list of its own: `g24_rec709_display` (the PNG sequence that failed with
  "unsupported colour space"), `lin_ap1_scene`, `srgb_p3d65_display`, `pq_rec2020_display` and the
  rest of the Color Interop IDs decode with their curve and are rotated into Rec.709.
- **A broken sequence frame fails once.** A frame that does not decode is no longer decoded again on
  every cache retry: it is logged once at `warn` with its path, marked red on the timeline, named in
  the viewport instead of "loading", and tried again only when its file changes.
- **Other image formats.** The viewer and `exr-rs convert` read every still format `vfx-io`
  decodes except HEIC/HEIF and AVIF: PNG, JPEG, TIFF, DPX, Cineon, Radiance HDR, BMP, TGA, SGI,
  ICO, GIF, WebP, PNM/PFM, IFF, RLA, Softimage PIC. New crate `exr-import` loads them as
  scene-linear, associated-alpha frames (OIIO colour rule: reader tag, else sRGB for integers and
  linear for floats; straight alpha premultiplied in linear light); EXR keeps the exr-core paths.
  Sequences of these formats play; `exr-rs info` summarises them. The Open dialog filters come
  from the one format registry.
- **File dialogs remember.** File > Open restores the last folder and format filter (also across
  restarts), Save view its own last folder; stored in the config, older configs load unchanged.
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
