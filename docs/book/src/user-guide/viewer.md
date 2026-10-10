# Viewer

`exr-view` is the winit/egui/wgpu image and sequence viewer. The CLI opens the
same application:

```sh
exr-rs v                         # start the viewer (also: exr-rs view)
exr-rs view --help                # viewer options
exr-rs view beauty.exr
exr-rs view --auto beauty.0001.exr
```

Bare `v` or `view` starts the viewer without requiring a path, preserving
persisted-session restoration. `--empty`/`-e` remains an optional legacy
compatibility flag. `--help`/`-h` shows help. An explicit file opens alone unless `--auto` is supplied. Drag-and-drop detects
sequences; the Open dialog and recents use the saved auto-detect preference.
Missing frame numbers remain gaps in the detected sequence.

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

For an accepted processed result, Metadata also identifies its frame, time,
revision, primary data/display windows and pixel aspect ratio. Its original header
is shown separately only when the loaded source matches that accepted primary.
Pixel Info shows processed scene-linear RGBA, canvas coordinates and the actual
sample pyramid level. Original primary-source values for all channels appear
separately when that source point is available. Histogram, waveform, vectorscope
and viewport readout use the accepted processed result while replacement work is
pending.

Under the image's alpha the viewport shows a backdrop picked from the toolbar:
Checker (default), Black, 50% Gray or a custom colour; the choice persists. It is
composited in the present pass in linear light relative to SDR white, so it has
the right brightness on SDR, HDR10 PQ, HLG and scRGB outputs, and the checker
follows the image as it pans and zooms. Exports (Save view) carry no backdrop.

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
| `Space` | Play/pause a sequence |
| Left/right arrows | Step frames |
| Left-button drag on a sequence | Scrub |
| Left-button click on a still | Pin pixel readout |
| `Z` / `Shift+Z` | Fullscreen / hide panels |

Fit uses the EXR display window; the data window is positioned inside it.
The View menu selects the preload strategy. Worker-thread decoding and the
shared frame cache stay within the configured RAM allowance. Playback keeps the
accepted GPU image until replacement pixels are drawable.

## Display and colour

Source pixels remain scene-linear. The GPU display transform applies exposure,
gamma, channel selection and the selected tone operator or OCIO display/view/look.
A shared extended-sRGB f16 canvas holds the image and UI; the final present pass
encodes the selected output surface.

![Settings showing HDR10 PQ and a 480-nit white level](../assets/display-pq.png)

Settings → Display selects SDR 8/10-bit, HDR10 PQ, HLG or scRGB when the actual
surface supports that format/colour-space pair. SDR white and display peak can use
OS values or explicit overrides. The screenshot shows a saved 480-nit override;
it is not the application default or a calibration recommendation. Manual SDR
reference white supports 80–1000 nits; this is independent of the HDR rendering
peak. The display-peak override shapes HLG output only.

Settings → Colour exposes the same OCIO controls as the Colour dock tab. Quick
monitor presets appear before the detailed input/display/view/look choices,
including HDR 1000 nits when an HDR output and the loaded config support it.
Display and Colour link directly to each other.

Settings → Memory controls RAM/VRAM shares and per-frame upload allowance.
Eligible images up to 8192 pixels per dimension use a reused resident GPU source,
subject to device support and actual allocation/transition budgets. Other images
use the 512-pixel tiled pyramid with coarser-level fallback; file-backed production
reads bounded bands. Small images are not forced through tile loading.

## Save a view

File → Save view exports the selected display look. SDR PNG, HDR10 PQ PNG and
linear display-light EXR are implemented; HLG PNG and scRGB file export remain
open. Display-light EXR uses linear Rec.709; its `whiteLuminance` records the
current display reference white. A display export does not preserve every original
channel or source metadata.

When a processed result is accepted, Save view retains that result's pixels,
geometry and time for the export. The requested maximum dimension selects a
pyramid level; mixed fine/coarse coverage uses the same area reducer as scopes.
PNG and EXR writers stream rows without assembling a full f32 image. The success
message reports dimensions and any nonzero pyramid level. Loading or busy pixel
admission produces a retry message; budget and writer errors are reported instead
of silently exporting the original input.

The CLI's `--screenshot` is a neutral image export without a window, rather than
a capture of the UI. Its neutral display-light EXR uses a 203-nit reference white.
See [CLI examples](cli.md). Use `convert` or the
[library API](library.md) for source-image read/write workflows.

## OpenFX and current limits

The Effects dock hosts built-in/native effects, parameters, overlays and scan
diagnostics. Lazy tiled OpenFX uses checked original-f32 ROI providers and bounded
CPU admission. Accepted processed pixels are connected to scopes, viewport
readout, Metadata, Pixel Info and Save view. If host preparation exceeds the budget
before native rendering begins and its temporary owners have retired, non-tile effects
may use a coarser whole-image level. Pending work keeps its selected level; a refusal
after native rendering begins does not trigger this fallback. The bounded fallback
and the bounded geometry/time/consumer contracts are accepted, completing the
approved tile/OpenFX phase after independent host verification. HLG hardware,
log-input exposure and HLG/scRGB export remain separate work.

Deep and multipart files report an unsupported-input error; use the corresponding
low-level readers for those file shapes. Subsampled channels are decoded whole and
scaled to the data window; luminance/chroma inputs also provide reconstructed RGB.
Window geometry, docks, controls, recents and budgets persist between runs.
