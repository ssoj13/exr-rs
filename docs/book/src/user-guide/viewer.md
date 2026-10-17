# Viewer

The viewer is the `exr-rs view` (`v`) subcommand, an image and sequence player on
winit, egui and wgpu; its code is the `exr-view` library crate.

```sh
exr-rs v                         # start the viewer (also: exr-rs view)
exr-rs view --help                # viewer options
exr-rs view beauty.exr
exr-rs view --auto beauty.0001.exr
```

Bare `v` or `view` starts the viewer without a path and restores the saved
session; `--empty`/`-e` does the same explicitly. `--help`/`-h` shows help. An
explicit file opens alone unless `--auto` is given (`--no-auto` states the default).
Drag-and-drop detects sequences; the Open dialog and recents use the saved
auto-detect preference. Missing frame numbers remain gaps in the detected sequence.

## Workspace

![EXR viewer with the 32-point checker, metadata, histogram and HDR10 settings](../assets/viewer-pq.png)

Actual native-window capture, showing `Desk.exr` on a
Windows Vulkan HDR10 PQ surface. The image is from
[openexr-images](https://github.com/AcademySoftwareFoundation/openexr-images);
its [license and screenshot provenance](../assets/sources.md) accompany the book.
PNG screenshots document the interface; they do not reproduce an HDR monitor's
absolute light output.

The viewport, Metadata, Pixel Info, Settings and scopes are dockable panels.
Metadata contains the Channels tree and original header attributes; the toolbar
selects the layer and RGBA/R/G/B/A/luma display. Click a layer to select it, or a
channel to view it alone. The current layer opens by default; manual folds persist
across redraws, and switching layers reveals the selected layer once. Expand all
and Collapse all sit to the right of Channels.

Metadata also shows the displayed (processed) result's frame, time, data/display
windows and pixel aspect ratio; the original header appears separately when it belongs
to that result's source. Pixel Info shows processed scene-linear RGBA, canvas
coordinates and the sampled pyramid level, plus the original source values of all
channels when that source point is available. Histogram, waveform, vectorscope and
viewport readout keep showing the last completed result while new work is pending.

Under the image's alpha the viewport shows a backdrop picked from the toolbar:
Checker (default), Black, 50% Gray or a custom colour; the choice persists. It is
composited in the present pass in linear light relative to SDR white, so it has
the right brightness on SDR, HDR10 PQ, HLG and scRGB outputs. The default checker
has 16-pixel cells fixed to the viewport; the picker sets the cell size and can pin
the cells to the image so they pan and zoom with it. Exports (Save view) carry no backdrop.

## Navigation and playback

| Input | Action |
|---|---|
| Middle-button drag | Pan |
| Right-button drag | Zoom about the viewport centre |
| Mouse wheel | Zoom toward the cursor |
| `F` | Fit the display window to the current viewport |
| `H` or `0` | Show at 1:1 |
| `Ctrl +` / `Ctrl -` | Step zoom |
| `R` / `G` / `B` / `A` | Isolate a channel |
| `Space` / `Shift+Space` | Play/pause a sequence / play backwards |
| Left/right arrows | Step frames |
| Left-button drag on a sequence | Scrub |
| Left-button click on a still | Pin pixel readout |
| `Z` / `Shift+Z` | Fullscreen / hide panels |

Fit uses the EXR display window; the data window is positioned inside it.
The View menu selects the preload strategy. Worker-thread decoding and the
shared frame cache stay within the configured RAM allowance. During playback the
current image stays on screen until its replacement is ready to draw.

## Display and colour

Source pixels remain scene-linear. The GPU display transform applies exposure,
gamma, channel selection and the selected tone operator or OCIO display/view/look.
A shared extended-sRGB f16 canvas holds the image and UI; the final present pass
encodes the selected output surface.

![Settings showing HDR10 PQ and a 480-nit white level](../assets/display-pq.png)

Settings → Display selects SDR 8/10-bit, HDR10 PQ, HLG or scRGB when the actual
surface supports that format/colour-space pair. SDR white and display peak can use
OS values or explicit overrides. The screenshot shows a saved 480-nit override;
it is not the application default or a calibration recommendation. Reference white sets the brightness of the UI and of relative-white image values on an
HDR output; its manual range is 80–1000 nits, independent of the OCIO rendering peak.
The display-peak override shapes HLG output only.

Settings → Colour holds the OCIO controls (config, input override, display, view, look). Every
frame is converted once, from its own colour space (see below) to the display/view: the views
offered are the ones the config's viewing rules allow for that space, so an HDR10 PNG or an sRGB
screenshot gets `Video (colorimetric)` (shown as it is) while an EXR gets an ACES rendering; the
view is remembered separately for scene-referred and display-referred frames. Input overrides
the open file's colour space (for that file only). Quick
monitor presets appear before the detailed input/display/view/look choices,
including HDR 1000 nits when an HDR output and the loaded config support it (it
selects an ACES HDR rendering, not a 1000-nit UI white). Display and Colour link
directly to each other.

Open Settings with `Ctrl+,` or from the Window menu; its categories are Memory,
Display, UI (sequence auto-detection, fonts, interface sizes) and Colour, and Reset
affects the category shown. Settings persist in the platform's exr-view
configuration directory.

Settings → Memory controls RAM/VRAM shares and per-frame upload allowance.
Eligible images up to 8192 pixels per dimension use a reused resident GPU source,
subject to device support and actual allocation/transition budgets. Other images
use the 512-pixel tiled pyramid with coarser-level fallback; file-backed production
reads bounded bands. Small images are not forced through tile loading.

## Save a view

File → Save view exports the selected display look as SDR PNG, HDR10 PQ PNG, HLG PNG
or linear display-light EXR; there is no scRGB file export. Display-light EXR uses
linear Rec.709; its `whiteLuminance` records the current display reference white. A display export does not preserve every original
channel or source metadata.

Save view exports the displayed result's pixels, geometry and time. The requested
maximum dimension selects a pyramid level; mixed fine/coarse coverage uses the same area reducer as scopes.
PNG and EXR writers stream rows without assembling a full f32 image. The success
message reports dimensions and any nonzero pyramid level. While the pixels are still
loading or busy, Save view asks to retry; budget and writer errors are reported instead
of silently exporting the original input.

The CLI's `--screenshot` is a neutral image export without a window, rather than
a capture of the UI. Its neutral display-light EXR uses a 203-nit reference white.
See [CLI examples](cli.md). Use `convert` or the
[library API](library.md) for source-image read/write workflows.

## Other image formats

Besides EXR the viewer opens [`.exv` video clips](exv.md) (multi-frame `.exv` containers open
directly as sequences with their embedded OpenEXR headers and Cryptomatte manifests) as well as
PNG, JPEG, TIFF, DPX, Cineon, Radiance HDR, BMP, TGA, SGI, ICO, GIF,
WebP, PNM/PFM, IFF, RLA and Softimage PIC (HEIC/HEIF and AVIF are not built in), as sequences
too (`shot.####.png`). EXR keeps its own readers; every other format is decoded by `exr-import`
into the same frame, so exposure, OCIO, scopes, Pixel Info, effects and Save view
work unchanged. The first frame of multi-image stills (GIF, ICO, TIFF pages) is shown, and the
file is always held in RAM (no out-of-core pyramid).

Frames keep their values as the file holds them and a colour tag. A foreign still takes the
format reader's colour-space tag (OpenImageIO's rule: PNG honours its `sRGB`/`gAMA`/`cICP` chunks,
Radiance HDR is linear, DPX maps its transfer code, PNM is Rec.709); an EXR takes
`colorInteropID`, else `whiteLuminance` (absolute light: shown colorimetrically, as display light),
else `chromaticities` (matched to AP0/AP1/Rec.709/Rec.2020/P3-D65, other primaries taken by their own
matrix into linear Rec.709). An untagged file follows the OCIO
config's file rules (ACES configs: EXR is ACES2065-1); when only their default rule matches, an
integer still is `srgb_rec709_display`, a float still `lin_rec709_scene` and an EXR `scene_linear`.
The config resolves the tag by name or alias; a name it does not know is an error shown in the
status bar and the Metadata panel's Colour section, with the input override offered, never a
silent substitute. Without OCIO the same tag drives the built-in decode (sRGB, BT.709, gamma,
Cineon/`KodakLog` printing density, PQ, HLG, primaries into Rec.709); a PQ or HLG frame is absolute
light (100 nits = 1.0) and shows at its nits on an HDR output. Straight alpha (PNG, TGA, WebP,
unassociated TIFF) and associated alpha are both kept: the display converts the unpremultiplied
colour and premultiplies in light. The Metadata panel shows the tag, its origin, the alpha kind
and the resolved colour space; Pixel Info adds the `scene_linear` values; the scopes have a
Display values switch (the colour image as displayed instead of the stored values).
A sequence frame that fails to decode is marked red on the timeline and is not retried until its
file changes; the viewport and the log say why.

File > Open remembers the last folder and the last format filter across openings and restarts
(Save view remembers its own folder); the filter list is the format registry.

## OpenFX and current limits

The Effects dock hosts the installed OpenFX plug-ins (the standard search: `OFX_PLUGIN_PATH`, then the OS
plug-in directory; ofx-rs's effects are its `ofx-effects.ofx.bundle`, from the `ofx-effects-<version>-<platform>.zip` of <https://github.com/ssoj13/ofx-rs/releases/latest>:
unzip it into the plug-in directory or a folder on `OFX_PLUGIN_PATH`; the empty dock names the searched
places and links the release), with
path parameters' browse buttons (they open the viewer's one file dialog, which also remembers the folder per use), presets saved per plug-in in
`~/.ofx-rs/presets/<plug-in ID>/` (`OFX_RS_PRESETS_DIR` moves that folder), parameters, overlays and
plug-in scan diagnostics. Effects read source pixels lazily through checked f32
region providers within a bounded CPU memory budget; their output feeds the scopes,
viewport readout, Metadata, Pixel Info and Save view. If preparing a non-tiled effect
exceeds the budget before rendering begins, the viewer may render it from a coarser
whole-image level; a refusal once rendering has begun does not fall back. Not yet
done: HLG verification on HLG hardware, exposure for log-encoded input, and scRGB
file export.

Deep and multipart files report an unsupported-input error; use the corresponding
low-level readers for those file shapes. Subsampled channels are decoded whole and
scaled to the data window; luminance/chroma inputs also provide reconstructed RGB.
Window geometry, docks, controls, recents and budgets persist between runs.
