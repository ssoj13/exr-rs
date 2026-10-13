# COLOR_PLAN: frames keep their own colour space, one OCIO conversion at display

Status: approved by the operator 2026-10-12, not started. Replaces the import-time linearisation
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

1. **exr-import: no linearisation.** `Loaded` returns the decoded values and a `ColorTag`
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
2. **EXR tags.** On load: `colorInteropID` (OpenEXR standard attribute, `exrstdattr.rst:51-57`)
   first; else `chromaticities` matched to `lin_<primaries>_scene` (AP0, AP1, Rec.709,
   Rec.2020, P3-D65; Bradford tolerance like `exr_import`'s white-point check); else file rules
   (studio: EXR -> ACES2065-1); else `scene_linear`. Multipart: per part (`exrinfo.rst:20-28`).
3. **Tag -> OCIO colour space.** `Config::colorspace` resolves names and aliases, case
   insensitive (vfx-ocio `config.rs:478-487`); interop IDs are aliases in the studio / cg
   configs (`Rec.2100-PQ - Display` aliases `pq_rec2020_display`). A tag the config does not know
   is a typed error shown in the status bar and the frame's info, with the manual override
   (below) offered; never a silent substitute.
4. **OCIO per frame.**
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
5. **Alpha in the shader.** Associated alpha: unpremultiply, run the processor on colour,
   premultiply for the canvas (egui's premultiplied blend). Straight alpha: process, then
   premultiply. One function in the tile shader and its CPU twin.
6. **Without OCIO** (off or no config): the same tag drives a built-in decode in the shader
   (the `Tf` curves move from exr-import to the shader: sRGB, Rec.709, gamma, PQ, HLG, Cineon;
   primaries to the display's). One path, two sources of the transform.
7. **Absolute light.** Drop `DecodedFrame::light_gain` / `white_nits` for imported stills: PQ
   and HLG are display colour spaces in OCIO (1.0 = 100 nits of the display reference, the same
   unit WarpBro writes with). The HDR output's SDR white stays the present pass's business
   (`present::Target`). Open point: an EXR with `whiteLuminance` but no colour space tag (our own
   display-light EXRs) - tag it `display-linear` through a config space if the config has one,
   else keep the gain only for that case; decide when implementing, document it here.
8. **Inspector, scopes, export.** The pixel inspector shows the stored values and their
   `scene_linear` (CPU processor). Scopes run on source or display values (a switch). Save view
   and the headless screenshot use the same per-frame processor (one codepath).
9. **UI.** The frame's tag and its origin in the info panel and status bar; the view combo lists
   only the allowed views and marks the remembered one.

## Tests

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
