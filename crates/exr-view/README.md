# exr-view

GPU OpenEXR viewer with sequence playback, docked panels, colour management and
OpenFX integration. The viewer owns its winit/wgpu surface and uses the shared
egui widget crates also used by Playa. See the [viewer guide](../../docs/book/src/user-guide/viewer.md)
and the [active execution plan](../../plans/20261001-viewer-tiles-openfx.md).

## Preferences

Open Settings with Ctrl+, or the Window menu.

| Category | Controls |
|---|---|
| Memory | RAM/VRAM cache shares and upload budget |
| Display | Supported window outputs, SDR reference white, HLG display peak, OS display information |
| Colour | OCIO config, input space, display, view, look, and monitor presets |
| UI | Sequence auto-detection, fonts and interface sizes |

Settings > Colour and the separate Colour dock tab edit the same selection and use
the same OCIO panel. Config-file browsing and reload work from either entry point.
Display and Colour have direct links to each other. Reset affects the category
currently shown.

Reference white controls UI brightness and relative-white image values on an HDR
output. Its manual range is 80–1000 nits; this range does not change saved values,
OS-derived defaults or the current output. It is independent of the OCIO rendering
peak. The HDR 1000-nit preset selects an ACES HDR rendering, not 1000-nit UI white.
Monitor presets are checked against the loaded OCIO config; HDR presets require an
active HDR output. HLG display peak affects HLG only.

The viewport checker defaults to 32 egui points and stays fixed in screen space.
Config persists at the platform's exr-view configuration directory.

## Implementation status

Resident and tiled display paths, playback, scopes, OCIO controls and supported
surface outputs are implemented. This is an application-specific subset of
Playa's preferences; compositor, project and audio settings do not apply here.
OpenFX source acceptance and processed consumers remain tracked in the active
plan; this UI update does not establish acceptance of those modules.
