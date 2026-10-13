# COLOR_PLAN: frames keep their own colour space, one OCIO conversion at display

Status: approved by the operator 2026-10-12; implemented 2026-10-13 on branch `color-plan`
(items 1-9 and the Tests done; see "Progress"). Replaces the import-time linearisation
and the per-frame light gain (`8e7f166`). Keep this file current while the work goes on; when it
is done, its summary goes to CHANGELOG.md and the file is removed.

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

## The model

- A frame stores its pixels **as read** (integers normalised to 0..1, floats as they are) and a
  **colour tag**: the colour space and where it came from.
- The screen gets **one** OCIO processor per frame: frame space -> display / view (+ look).
  The view is chosen among the views the config's viewing rules allow for that space.
- No working space in storage. Not ACEScg, not linear Rec.709: converting into one on load is a
  compositor's model (Nuke Read -> working space); for a viewer it loses the source and forces a
  second rendering of display-referred pictures.

## Work

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
   - Processor input = the frame's space; cache processors (and their WGSL) by (space, display,
     view, looks, config revision).
   - Views offered = `Config::get_views_for_colorspace(display, space)` (vfx-ocio
     `config.rs:2135-2149`, OCIO `getNumViews(display, colorspace)`); default =
     the first of them (OCIO `getDefaultView(display, colorspace)`).
   - The operator's view choice is remembered per encoding class (scene-linear / log versus
     video / display-linear) so a sequence of EXRs and a PQ PNG each keep their view.
   - `ocio.input` becomes an **override** of the frame tag (per file, shown as such), not the
     input of everything. Data colour spaces stay bypassed (`data_bypass`).
   - Exposure stays OCIO's `linearCC` in `scene_linear` (`ocio.rs:192-207`): it works for any
     input space.
5. [x] **Alpha in the shader.** Associated alpha: unpremultiply, run the processor on colour,
   premultiply for the canvas (egui's premultiplied blend). Straight alpha: process, then
   premultiply. One function in the tile shader and its CPU twin.
6. [x] **Without OCIO** (off or no config): the same tag drives a built-in decode in the shader
   (the `Tf` curves move from exr-import to the shader: sRGB, Rec.709, gamma, PQ, HLG, Cineon;
   primaries to the display's). One path, two sources of the transform.
7. [x] **Absolute light.** Drop `DecodedFrame::light_gain` / `white_nits` for imported stills: PQ
   and HLG are display colour spaces in OCIO (1.0 = 100 nits of the display reference, the same
   unit WarpBro writes with). The HDR output's SDR white stays the present pass's business
   (`present::Target`). Open point: an EXR with `whiteLuminance` but no colour space tag (our own
   display-light EXRs) - tag it `display-linear` through a config space if the config has one,
   else keep the gain only for that case; decide when implementing, document it here.
8. [x] **Inspector, scopes, export.** The pixel inspector shows the stored values and their
   `scene_linear` (CPU processor). Scopes run on source or display values (a switch). Save view
   and the headless screenshot use the same per-frame processor (one codepath).
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
- [x] 5 `transfer::{unpremul, premul, alpha_wgsl}`; the shader and every CPU twin use them.
- [x] 6 `exr_import::{builtin, Builtin, Tf}` + `transfer::{decode, decode_wgsl}` (HLG on its
  1000-nit reference display, PQ in units of 100 nits; absolute light uses `ocio::display_gain`).
- [x] 7 `HDR_REFERENCE_WHITE_NITS`, `Loaded::white_nits`, `DecodedFrame::{white_nits,
  light_gain}`, `render::Look::{ocio_gain, light_gain}` removed. Open point decided (below).
- [x] 8 Pixel Info shows `scene_linear` (`State::scene_linear`); scopes' "Display values" switch
  (`Config::scope_display`, `export::shade_flat`); Save view and the headless screenshot share
  `export::shade` and `State::frame` (the screenshot uses the saved OCIO choice).
- [x] 9 Metadata > Colour (`State::rows`), status bar colour section (`State::summary`), the view
  combo lists the allowed views and says which class it is remembered for.
- [ ] Follow-ups: OpenFX effects and the tile pyramid work on stored (possibly encoded) values;
  vfx-io tags every chunkless PNG/JPEG `srgb_rec709_scene` (OIIO's assumption), so such a still is
  "tagged", never "untagged" (the file-rule path serves formats that never tag: TIFF without
  EXIF, SGI, IFF, ICO, ...); telling an assumed tag from a stated one needs a vfx-io fact.

## Review fixes (2026-10-13)

- [x] HIGH 1 Looks follow the frame class: `Sel::{scene_look, video_look}` (`Sel::look(class)`),
  video default none; test: the PQ PNG with a saved scene LMT keeps no look and stays an identity.
- [ ] HIGH 2 EDR displays (`Encoding::Edr` from oiio-rs), the SDR screenshot on a matching SDR
  display, `srgbe_p3d65_display` frames get their video view.
- [ ] MEDIUM 3 whiteLuminance round trip exact on relative displays (203 test).
- [x] MEDIUM 4 `Ocio::key(names, pre, linear)` is asked of the cache before `transform` builds;
  test `frames_of_one_space_share_the_transform`.
- [ ] MEDIUM 5 unmatched EXR chromaticities: pre-matrix or typed error, no silent untagged.
- [ ] Pyramid levels and OpenFX inputs on linear light, or a written design.
- [ ] LOW: premul convention doc, explicit builtin under OCIO, presets for Video frames,
  `vec![Vec::with_capacity(n); c]`.

## Tests

All in place and passing (2026-10-13): `ocio::tests::{pq_png_is_an_identity_on_a_pq_display,
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
  with a pre-matrix (primaries, Bradford to D65, x nits / 100, `TagOrigin::Luminance`), combined in
  front of the processor (`Processor::combine`). The viewing rules then offer `Video
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
