# Changelog

All notable changes to **exr-rs** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### 2026-10-13

- **Decode images to scene-linear light before processing, then display by source class.**
  A finished PQ HDR10 picture previously received another ACES rendering and appeared almost
  black. Frames now retain their source values and colour/alpha tags; one CPU decode feeds
  layer assembly, pyramid reduction, OpenFX, scopes, and Pixel Info. The display stage starts
  from `scene_linear`, while the source class selects and remembers its view and look:
  `Video (colorimetric)` for display-referred pictures. Both stages cache processors by their
  complete inputs. OCIO config, source-space, and transform failures are explicit errors;
  turning OCIO off selects the built-in decode/display pair. Tags come from the reader, EXR
  `colorInteropID` / `whiteLuminance` / `chromaticities`, or config file rules; Input remains a
  per-file override. `convert` writes foreign stills as read with `colorInteropID`. This
  supersedes the absolute-light change below and removes `HDR_REFERENCE_WHITE_NITS`,
  `white_nits`, `light_gain`, and global `ocio.input` / `ocio.view` (now per-class views).
- **Decoded-source changes invalidate cached presentation and temporal inputs.** The primary
  decode key and decoder policy (config load revision and input override) enter source identity
  and OpenFX render stamps. Native file MIPs are reused only with an identity decode. Pixel
  Info prepares scene-linear readout even when the viewport and scopes are hidden. Save view
  and headless screenshots use the display stage's CPU twin over decoded pixels. Disabling
  an effect clears accepted effect pixels for readout, scopes, export, and presentation even
  without a viewport frame.
- **Focused color validation is complete:** the viewer suite reports 335 passed, 0 failed,
  and 23 ignored tests (188.34s). Three GPU checks cover tile/CPU twins and Direct decoded
  profiles (every half component for OCIO/built-in sRGB/PQ); Pixel Info-only preparation also
  passes. A separate singular default-view bridge regression passes (0.01s), verifying typed
  `OcioError::Transform` rejection by worker decode, frame resolution, and the shown decode.
  The real WarpBro PQ PNG round-trips with worst and mean code error 0; the visually inspected
  SDR proof shows the Venice environment and blue fractal. The native/GPU effect-disable
  regression also passes (1.84s, explicit ignored-test run). No native HDR viewport screenshot
  was captured. The six-stage bootstrap gate, integration, and release remain pending.
- **Release preparation for a same-version v0.1.0 rebuild (pending).** Nineteen own Git source
  pins were verified against remote revisions. The inaccessible OFX `fabb` pin was replaced
  manually with `9cf607d`, adding the required `ofx-banner` and `ofx-expiry` packages without
  changing `windows-sys`. Bootstrap's obsolete `u` action was replaced by `r`; it builds the
  exact pinned OpenFX bundle. The local bundle build produced 35 release plug-ins in 4m17s;
  its source hash `03e011bcb9a34e9193a3bb32ea1f9ab71767899c35d9c4bb1c00d60782375a89`
  matches recomputation. Final gates, tag/release replacement, CI, and published assets are
  still pending.

### 2026-10-12

- **HDR stills show at their real brightness.** A PQ (HDR10) still decoded to `nits / 100` and was
  then shown with 1.0 at the display's SDR white, so it came out `white / 100` times too bright
  (twice at 203 nits); HLG decoded with its reference white at ~0.26, about four times too dark.
  Now, per BT.2408, PQ is `nits / 203` and HLG's 75 % reference white is 1.0. Absolute light is a
  property of the frame: an imported PQ still, or an EXR with the standard `whiteLuminance`
  attribute (the nits of RGB 1, 1, 1; OpenEXR `ImfStandardAttributes.h`), is shown on an HDR
  output at the nits it holds whatever the SDR white is set to, and on SDR with 203 nits at
  white. Save view writes the same light, so a saved display-light EXR reopens unchanged.
  Isolated channels are data and are not scaled.
- **PFM opens linear** (oiio-rs d6d5c95): PFM files were tagged `Rec709` like the integer PNM
  formats (OIIO's rule) and showed dark; their samples are linear quantities.

- **Dependencies**: every ssoj13 git dependency at its head (ofx-rs a778569: OpenFX Auto computes
  on the CPU when only a software GPU adapter exists; gpu-info-rs 136386b; render-rs 6901ac1).
- **Tests leave no files behind**: the Cryptomatte and viewer tests write into self-removing
  temporary folders (`tempfile`) instead of `%TEMP%` folders named after the process id, which
  piled up and could hand a later run with the same id stale frames. exr-view no longer declares
  `env_logger` (the `exr-rs` binary owns the log backend).

### 2026-10-11

- **Scrubbing loaded frames is fast.** Switching between resident 3840x2160 half frames costs about
  25 ms instead of about 150 ms: an all-half layer goes to the display from its planes as
  `Rgba16Float` (no f32 materialize, half the upload), the precision profile comes from load-time
  channel spans instead of a rescan, the idle stage shows the materialized pixels without a copy,
  other layers materialize through a typed, cache-blocked fill, and open scopes read the display's
  texture instead of uploading the frame a second time.
- **exr-view**: the viewer no longer links ofx-rs's built-in effects (`ofx-runtime` feature `effects`); they
  are the standard OpenFX bundle `ofx-effects.ofx.bundle`, found by the usual search (`OFX_PLUGIN_PATH`, then
  the OS plug-in directory). Linking `ofx-effects` pulled ofx-fractal -> render-env -> exr-core from the exr-rs
  git repo, so the build carried two copies of exr-core, deflate-rs, imath-rs and exr-image. Plug-in IDs are
  unchanged, so saved effects stay valid. The empty Effects tab names the searched places and links the
  ofx-rs release (`https://github.com/ssoj13/ofx-rs/releases/latest`), where the bundle comes from.
- **exr-view**: ofx-rs bumped to 597457b (and every other ssoj13 git dependency to its head). The OpenFX
  panel's browse buttons now open the viewer's in-app file dialog (`egui-file-dialog`) and write the picked
  path into the parameter; presets live in `~/.ofx-rs/presets/<plug-in ID>/` (`OFX_RS_PRESETS_DIR`).
- **One file dialog.** File > Open, Save view, the OCIO config browser and the OpenFX browse buttons use the
  same in-app dialog, each with its own filters (the OCIO dialog lists only `.ocio` / `.ocioz`; an OpenFX
  dialog the parameter's own filters, else every file) and each remembering the folder of its last
  confirmed pick under its own key. The Colour panel's separate `ocio_dir` memory is folded into the same
  map; older configs migrate (`ocio_dir`, `open_dir`, `save_dir`, `open_filter`).
- **exr-view**: plug-ins see a stable image identity (`kOfxImagePropUniqueIdentifier`) for an unchanged
  input, so their caches hit across renders; it changes with the file (path, modification time, size at
  load), layer or channel, mip level and data window.

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
