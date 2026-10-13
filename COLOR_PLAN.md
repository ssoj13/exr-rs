# COLOR_PLAN: decode to scene-linear light, then display by source class

Status: approved by the operator 2026-10-12; two-stage source implementation reviewed
2026-10-13 on branch `color-plan`. Focused viewer, color, and GPU validation is complete:
335 passed, 0 failed, 23 ignored. The full six-stage bootstrap gate, integration, and
same-version v0.1.0 release replacement remain pending. Replaces the
import-time linearisation and per-frame light gain (`8e7f166`). Keep this file while the work
goes on; remove it only after final gates and integration, with the summary in CHANGELOG.md.

## Why

A PQ HDR10 PNG written by WarpBro (cICP 9/16/0/1, an ACES 2.0 HDR rendering, i.e. a finished
display-referred picture) shows almost black in exr-view with the operator's OCIO setup
(display `Rec.2100-PQ - Display`, view `ACES 2.0 - HDR 1000 nits (P3 D65)`). Causes, verified:

- `exr-import` linearises every still into linear Rec.709 (`exr_import.rs` module doc, `Tf::lin`;
  PQ also divided by 203, `HDR_REFERENCE_WHITE_NITS`), so the frame forgets what it was.
- exr-view feeds every frame to OCIO as ONE global input space (`ocio::Sel::input`, empty = the
  `scene_linear` role, `ocio.rs:57,157`), and runs the chosen scene view on it: the ACES picture
  is tone-mapped a second time.
- EXR frames are always taken as `scene_linear` (ACEScg in the studio config): `chromaticities`
  and `colorInteropID` are never read on load (only written by export, `export.rs:571`).

OCIO itself does not skip the view transform for a display-referred source:
`BuildSourceToDisplay` (OpenColorIO `transforms/DisplayViewTransform.cpp:177-218`) converts a
display-referred source back to scene reference through the inverse default view transform
(`ColorSpaceTransform.cpp:348-392`, studio config `default_view_transform: Un-tone-mapped`) and
then applies the view. The config's answer is the choice of view: `viewing_rules` tie the ACES
views to `encodings: [log, scene-linear]` and `Video (colorimetric)` (a display-referred view
transform, `from_display_reference: MatrixTransform {}`) to `[sdr-video, hdr-video, edr-video,
display-linear]` (studio-config-v4.0.0_aces-v2.0_ocio-v2.5.ocio:31-48, 99, 106-108). A PQ
source viewed with `Video (colorimetric)` on a PQ display is PQ -> XYZ -> PQ: an identity.

## The model (two stages, revised 2026-10-13 by the operator)

- A frame stores its pixels **as read** (integers normalised to 0..1, floats as they are) and a
  **colour tag**: the colour space and where it came from. Nothing is converted at import.
- **Stage 1, DECODE (one, for everything):** frame space -> `scene_linear`. With OCIO the
  config's processor `space -> scene_linear` (a `ColorSpaceTransform` to the role; a display
  space goes through the inverse default view transform, as OCIO does), with the tag's
  pre-matrix in front (EXR `whiteLuminance`, unnamed `chromaticities`); without OCIO the built-in
  curve and primaries (`exr_import::builtin`, `transfer::decode`) to linear Rec.709, the
  built-in scene_linear. Alpha leaves stage 1 associated in linear light (associated colour is
  unpremultiplied before the conversion, straight colour taken as stored; both multiplied by
  alpha after). Its output is the ONLY thing every consumer works on: the pyramid levels (built
  from it, so levels average light), OpenFX effects, scopes, the pixel readout, Save view.
  Stage 1 runs where a layer's RGBA is assembled from the stored channels (`ImageBackend`
  materialize, file-backed bands, effect inputs): one function, `ocio::Decode::apply`.
  Data layers (normals, depth, ids) and data colour spaces are not converted.
- **Stage 2, DISPLAY (one):** `scene_linear` -> display / view (+ look), the exposure as OCIO's
  `linearCC` in `scene_linear`; without OCIO the built-in exposure, tone operator and sRGB encode.
  The view and look are chosen by the SOURCE class (scene-referred vs display-referred, the
  config's viewing rules for the frame's own space), remembered per class; an effect does not
  change the class. The tile shader runs stage 2 only (its input, the pyramid, is already
  scene_linear), so there is no second texture pass and no decode on the GPU.
- Why display-referred stills stay an identity: stage 1 takes a PQ or sRGB picture to scene
  through the inverse default view transform (studio / cg configs: Un-tone-mapped, a matrix),
  stage 2's `Video (colorimetric)` comes back through the same default view transform forward
  (`ColorSpaceTransform.cpp:348-392`, `DisplayViewTransform.cpp` BuildSourceToDisplay). Measured
  with vfx-ocio before coding: sRGB 1.6e-6, P3 EDR 6.4e-7, PQ exact except 10^4-nit primaries next
  to zero components (f32 noise through PQ's toe, 2.7e-3 of a code range). Tested below.
- Both halves are cached by key (stage 1 per frame space + pre-matrix, stage 2 per display, view,
  look, output form), asked before anything is built.

## Work (focused validation complete; full gates and integration pending)

1. [x] **exr-import: no linearisation.** `Loaded` returns the decoded values and a `ColorTag`
   `{ space: String, origin: TagOrigin }` instead of linear Rec.709. Tag order (OIIO's,
   `png_pvt.h:224-234, 326-337`, `color_ocio.cpp:2266-2365`):
   1. `oiio:ColorSpace` from vfx-io (already set from cICP: 9/16 -> `pq_rec2020_display`, tested
      in oiio-rs `vfx-io/tests/png_chunk_facts.rs:114-122`; sRGB / gAMA chunks; DPX transfer);
   2. nothing: the config's **file rules** (`Config::colorspace_from_filepath`,
      `filepath_only_matches_default_rule`); with no config or only the default rule, the
      explicit fallback: integer data `srgb_rec709_display`, float data `lin_rec709_scene`.
   Remove `HDR_REFERENCE_WHITE_NITS`, `Loaded::white_nits` and the PQ / HLG divisors.
   Alpha: keep whether the file's alpha is straight or associated; no premultiplication in
   encoded space and none at import (see 5).
2. [x] **EXR tags.** On load: `colorInteropID` (OpenEXR standard attribute, `exrstdattr.rst:51-57`)
   first; else `chromaticities` matched to `lin_<primaries>_scene` (AP0, AP1, Rec.709,
   Rec.2020, P3-D65; Bradford tolerance like `exr_import`'s white-point check); else file rules
   (studio: EXR -> ACES2065-1); else `scene_linear`. Multipart: per part (`exrinfo.rst:20-28`).
3. [x] **Tag -> OCIO colour space.** `Config::colorspace` resolves names and aliases, case
   insensitive (vfx-ocio `config.rs:478-487`); interop IDs are aliases in the studio / cg
   configs (`Rec.2100-PQ - Display` aliases `pq_rec2020_display`). A tag the config does not know
   is a typed error shown in the status bar and the frame's info, with the manual override
   (below) offered; never a silent substitute.
4. [x] **OCIO per frame.**
   - Stage 1 input = the frame's space; output = `scene_linear`. Stage 2 input =
     `scene_linear`; source space still selects the view and look. Cache each stage before
     building, including config load revision, source pre-matrix, and alpha kind for decode.
   - Views offered = `Config::get_views_for_colorspace(display, space)` (vfx-ocio
     `config.rs:2135-2149`, OCIO `getNumViews(display, colorspace)`); default =
     the first of them (OCIO `getDefaultView(display, colorspace)`).
   - The operator's view choice is remembered per encoding class (scene-linear / log versus
     video / display-linear) so a sequence of EXRs and a PQ PNG each keep their view.
   - `ocio.input` becomes an **override** of the frame tag (per file, shown as such), not the
     input of everything. Data colour spaces stay bypassed (`data_bypass`).
   - Exposure stays OCIO's `linearCC` in `scene_linear` (`ocio.rs:192-207`): it works for any
     input space.
5. [x] **Alpha at both stages.** CPU decode unpremultiplies associated source colour and
   processes straight source colour as stored; both leave stage 1 associated in linear light.
   Stage 2 unpremultiplies that light for display processing and premultiplies for the canvas.
   The tile shader and CPU display twin share the same alpha rule.
6. [x] **Without OCIO** (explicitly off): the tag selects CPU built-in decode to linear Rec.709
   (sRGB, Rec.709, gamma, PQ, HLG, Cineon, and primaries). The shader runs the built-in display
   stage only. With OCIO on, a failed config load or transform is an explicit error.
7. [x] **Absolute light.** Drop `DecodedFrame::light_gain` / `white_nits` for imported stills: PQ
   and HLG are display colour spaces in OCIO (1.0 = 100 nits of the display reference, the same
   unit WarpBro writes with). The HDR output's SDR white stays the present pass's business
   (`present::Target`). EXR `whiteLuminance` without a colour-space tag is resolved as
   display-referred light with a pre-matrix; relative displays retain their reference white,
   and absolute outputs use the stated nits. See the decision below.
8. [x] **Inspector, scopes, export.** Original all-channel probes retain stored values;
   the active pixel readout uses `scene_linear` from stage 1. Scopes switch between
   scene-linear and display values. Save view and headless screenshots apply the same stage-2
   CPU twin to decoded pixels. Pixel Info resolves colour and prepares readout even when viewport and scopes are hidden.
9. [x] **UI.** The frame's tag and its origin in the info panel and status bar; the view combo lists
   only the allowed views and marks the remembered one.

## Progress (2026-10-13)

- [x] 1 `exr_import::{ColorTag, TagOrigin, Loaded::{tag, straight}}`; values as read; tags stored
  by interop ID (`sRGB` -> `srgb_rec709_scene`); `Loaded::to_exr` for `exr-rs convert`
  (oiiotool's behaviour: values, associated alpha, `colorInteropID`).
- [x] 2 `exr_import::exr_tag(header)` (per part by construction; the viewer reads one part).
- [x] 3 `ocio::Ocio::space` (the one mapping) + `SpaceError::Unknown`.
- [x] 4 `ocio::State::frame` (per frame id + choice revision), transforms cached by key,
  `Ocio::views(display, space, hdr)` = `get_views_for_colorspace`, `Sel::{scene_view,
  video_view}` per `Class`, `State::over` (per file, not persisted).
- [x] 5 `transfer::{unpremul, premul, alpha_wgsl}`; stage 1 associates decoded light,
  stage 2 shares the CPU/shader alpha rule.
- [x] 6 `exr_import::{builtin, Builtin, Tf}` + CPU `transfer::decode` (HLG on its 1000-nit
  reference display, PQ in units of 100 nits; absolute light uses `ocio::display_gain`).
  Shader decode and source curve/matrix/straight uniforms are removed.
- [x] 7 `HDR_REFERENCE_WHITE_NITS`, `Loaded::white_nits`, `DecodedFrame::{white_nits,
  light_gain}`, `render::Look::{ocio_gain, light_gain}` removed. Open point decided (below).
- [x] 8 Pixel Info reads stage-1 `scene_linear` and prepares independently of the viewport;
  scopes' "Display values" switch (`Config::scope_display`, `export::shade_flat`); Save view
  and headless screenshots share `export::shade` and `State::frame` over decoded pixels.
- [x] 9 Metadata > Colour (`State::rows`), status bar colour section (`State::summary`), the view
  combo lists the allowed views and says which class it is remembered for.
- [x] OpenFX inputs and every generated pyramid level now consume decoded scene-linear light.
  Native file MIPs are used only for an identity decode. Decode identity includes the primary
  decode key and decoder policy (config load serial and override); temporal neighbour sources
  and `RenderInputs::stamp` change with that identity. Held presentation is discarded when
  the decode identity changes. `DisplayTarget::sync_worker` clears accepted effect pixels
  when the worker becomes inactive, keeping readout, scopes, export, and presentation on the
  current source even without a viewport frame.
- [ ] Separate metadata follow-up: vfx-io tags every chunkless PNG/JPEG
  `srgb_rec709_scene` (OIIO's assumption), so such a still is "tagged", never "untagged"
  (the file-rule path serves formats that never tag: TIFF without EXIF, SGI, IFF, ICO, ...);
  telling an assumed tag from a stated one needs a vfx-io fact.

## Review fixes (2026-10-13)

- [x] HIGH 1 Looks follow the frame class: `Sel::{scene_look, video_look}` (`Sel::look(class)`),
  video default none; test: the PQ PNG with a saved scene LMT keeps no look and stays an identity.
- [x] HIGH 2 oiio-rs bumped by hand in Cargo.lock to fcad4c0 (`Encoding::Edr`; only the rev
  strings change, built with `--locked`). (a) `ocio::extended`: `hdr-video` and `edr-video`
  displays/views are offered on HDR outputs only; EDR light is RELATIVE (OCIO: proportional to an
  SDR signal, values above 1 brighter than SDR white), so its gain is 1 and only `hdr-video` is
  absolute. Consequence: a PQ still viewed on Display P3 HDR on an HDR output shows 100 nits at
  SDR white (2.03x brighter than on a PQ display at a 203-nit white), as that display defines.
  (b) `Ocio::srgb_display` + `State::for_srgb_file`: the 8-bit sRGB PNG (Save view, headless SDR
  shot) is rendered on the config's sRGB display whatever display is saved, HDR views dropped;
  a config without one is the typed `NoSrgbDisplay`. (c) EDR frames (`srgbe_p3d65_display`) are
  display-referred and get `Video (colorimetric)`. Tests `edr_displays_are_extended_and_relative`,
  `edr_frames_get_their_video_view`, `srgb_files_render_on_the_srgb_display`.
- [x] MEDIUM 3 `TagOrigin::Luminance { to_xyz, nits }`: RGB 1 is the display reference's 1.0
  (the white of a relative display), `nits` only sets where an absolute (HDR) output shows it
  (`Builtin::nits`, `ocio::display_gain(nits, target)`); test `white_luminance_exr_is_display_light`
  (203 nits: 1.0 = white on sRGB - Display, 203 nits on a PQ display at any SDR white).
- [x] MEDIUM 4 `Ocio::key(names, linear)` is asked of the cache before `transform` builds;
  the pre-matrix belongs to stage 1's separate cache key;
  test `frames_of_one_space_share_the_transform`.
- [x] MEDIUM 5 `TagOrigin::Chromaticities { to709 }`: primaries no interop ID names are taken
  as they are (their matrix into `lin_rec709_scene`, Bradford to D65; OCIO pre-matrix and the
  built-in decode); degenerate ones are `ImportError::Attr`. Tests in `exr_tag_rules` and
  `exr_tags_follow_chromaticities_then_file_rules` (a P3 matrix equals the named P3 space).
- [ ] Final acceptance (operator): the two-stage model replaces the single frame -> display
  processor. A-D source review and E focused validation are complete; the six-stage bootstrap
  gate and integration remain pending.
  - [x] A `ocio::Decode` (stage 1, CPU, `apply` over RGBA rows) and its resolver per frame.
  - [x] B stage 2 transforms from `scene_linear` (view / look by the source class), pre-matrix
    moved to stage 1, cache keys per half.
  - [x] C stage 1 applied where layers are assembled (materialize, file bands, effect inputs);
    half planes only for an identity decode; tile identity follows the decode key.
  - [x] D the tile shader is stage 2 only (built-in decode, curve / matrix / straight uniforms
    and `decode_wgsl` removed); Save view, scopes and readout on the stage-1 output.
  - [x] E tests: PQ and sRGB PNG identity through both stages with and without an identity
    effect; a 2x2 black/white sRGB checker's level 1 is 0.5 linear.

## Tests

Historical first-pass tests below predate final acceptance of the two-stage implementation.
Final focused receipts (coordinator verified, 2026-10-13):

- The real WarpBro PQ fixture round-trips through both stages with worst code error 0 and
  mean code error 0. Proof images are retained under `target/color-resume-proof/` (HDR
  `pq_round_trip.png` and the SDR result). The coordinator visually inspected the SDR image:
  the Venice environment and blue fractal are visible, without the repeated-ACES near-black
  failure. No native HDR viewport screenshot was captured; the strongest evidence is exact
  real PQ codes, CPU/GPU twins, and the visually inspected SDR artifact.
- GPU checks: 3 passed, including tile display/CPU twins and the Direct decoded-profile
  regression checking every half component for OCIO and built-in sRGB/PQ sources.
- Pixel Info with viewport and scopes hidden: 1 passed.
- Native/GPU `disabling_effect_updates_consumers_without_a_viewport_frame`: 1 passed in
  1.84s (explicit ignored-test run). Disabling the effect updates readout, scopes, export, and
  presentation by clearing accepted effect pixels when the worker becomes inactive.
- A-D source review covers source-class views/looks, explicit invalid-config errors,
  scene-linear effect/pyramid inputs, decode-policy/acceptance cache identity, metadata, and
  identity-only native MIPs.
- [x] Focused viewer suite (358 tests): 335 passed, 0 failed, 23 ignored in 188.34s. This
  replaces the initial run's 332 passes, 1 test defect, and 22 ignored tests.
- [x] Singular default-view bridge regression: 1 passed in 0.01s. The processor returns typed
  `OcioError::Transform("Singular Matrix can’t be inverted")`; worker decode, frame resolution,
  and the shown decode all reject it explicitly.
- [ ] Full six-stage bootstrap gate on the frozen candidate: pending (parity first, then
  cargo, clippy, and Python checks).
- [ ] Integration and same-version v0.1.0 release replacement: pending.

First-pass test inventory (2026-10-13): `ocio::tests::{pq_png_is_an_identity_on_a_pq_display,
srgb_png_is_an_identity_on_an_srgb_display, exr_tags_follow_chromaticities_then_file_rules,
untagged_png_follows_the_file_rules, an_unknown_tag_is_a_typed_error,
alpha_is_processed_unpremultiplied, builtin_decode_matches_the_config,
white_luminance_exr_is_display_light, inspector_shows_scene_linear}`, `transfer::tests::{
builtin_decode_closed_forms, alpha_is_unpremultiplied_then_premultiplied_in_light,
isolated_channels_show_codes_of_encoded_frames}`, `exr_import::tests::{exr_tag_rules,
pq_png_keeps_codes_and_cicp_tag, ...}`, GPU `tiles_render_the_display_transform_of_every_texel`
(decode cases), and `export::tests::real_pq_png_round_trips_through_the_headless_shot`
(`--ignored`, `EXR_VIEW_PQ_PNG`: WarpBro's frame.000000.pq.png comes back bit-exact).


- PQ PNG (cICP 9/16/0/1) -> `Rec.2100-PQ - Display` / `Video (colorimetric)`: identity within
  one 16-bit code; the ACES views are not offered for it.
- sRGB PNG -> `sRGB - Display` / `Video (colorimetric)`: identity within one 8-bit code.
- EXR with Rec.709 chromaticities through an ACES view equals vfx-ocio's CPU processor from
  `lin_rec709_scene`; an untagged EXR follows the file rules.
- Untagged PNG follows the file rules (studio: `sRGB - Display`).
- A tag the config lacks -> a typed error, no substitution.
- Associated / straight alpha: the canvas pixel equals process(unpremult) * alpha.
- No-OCIO path: PQ and sRGB decode equal the reference curves.

## Decisions

- Open point (EXR with `whiteLuminance` and no colour tag), decided 2026-10-13: OpenEXR defines
  `whiteLuminance` as the nits of RGB (1, 1, 1) and says that with `chromaticities` it converts
  the pixels to CIE XYZ (`ImfStandardAttributes.h`, `exrstdattr.rst`). So such a file is absolute
  CIE XYZ light: the tag is the config's `cie_xyz_d65_interchange` role (OCIO's display reference,
  1.0 = 100 nits; `CIE XYZ-D65 - Display-referred`, encoding display-linear in the ACES configs)
  with a pre-matrix (primaries, Bradford to D65, `TagOrigin::Luminance`), combined in front of
  the processor (`Processor::combine`). Revised in review: RGB 1 maps to the reference 1.0 (the
  white of a relative display, so the picture reopens as saved on SDR) and `whiteLuminance` is the
  nits an absolute (HDR) output shows that 1.0 at, not a factor on the pixels. The viewing rules then offer `Video
  (colorimetric)`, so our own Save view HDR EXRs (Rec.709 display light + `whiteLuminance`)
  reopen as they were saved, with no gain special case. A config without the role is a typed
  error (override offered); the no-OCIO path decodes the same matrix as absolute light.
  `colorInteropID` still wins over `whiteLuminance`.
- "Only the default rule" read as: an untagged frame whose path matches only the config's
  default file rule takes the explicit fallback (`srgb_rec709_display`, `lin_rec709_scene`,
  `scene_linear` for EXR); a config with no file rules matches only the default too. In the ACES
  4.0 configs the default rule says `sRGB - Display`, the same as the fallback; older configs'
  `Default: ACES2065-1` would have taken an 8-bit PNG for AP0.

- Untagged integer stills: follow the config's file rules (studio / cg: `Default -> sRGB -
  Display`, i.e. shown as they are). OIIO would take them as `srgb_rec709_scene` (a texture);
  the config decides, not the code.
- No compatibility with the old behaviour (no flag for import-time linearisation).

## Coordination

- `8e7f166` (HDR stills at their real brightness) is superseded by items 1 and 7; its tests
  (`hdr_reference_white_is_one`, `absolute_light_round_trips_and_scales_only_colour`, GPU tile
  vs twin at light gain 2.5) are replaced by the ones above.
- WarpBro writes HDR10 PNGs from OCIO's HDR views with 1.0 = 100 nits (`render_service::hdr_scale`),
  the same unit as the config's PQ display space, so its files round-trip unchanged.
