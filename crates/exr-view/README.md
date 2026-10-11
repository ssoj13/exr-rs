# exr-view

GPU OpenEXR viewer library behind `exr-rs view`: sequence playback, docked panels,
colour management, HDR output and OpenFX integration. The viewer owns its winit/wgpu
surface and uses the shared egui widget crates (`egui-widgets-rs`) also used by Playa.
Usage, controls and settings are in the [viewer guide](../../docs/book/src/user-guide/viewer.md)
([online](https://ssoj13.github.io/exr-rs/user-guide/viewer.html)).

The `ofx` feature (on by default) links the OpenFX host; effects are installed
OpenFX plug-ins (standard search; the exr-view tests that render a real effect need ofx-rs's
`ofx-effects.ofx.bundle` installed and fail without it); `test-data` enables the tests that read the private image set.
