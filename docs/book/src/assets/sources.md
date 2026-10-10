# Screenshot provenance

These are native window captures of the exr-rs viewer on Windows, taken before the
first release. No image generation or pixel editing was used.

- `viewer-pq.png`: Desk.exr, viewport checker, metadata, scopes and display settings.
- `display-pq.png`: Settings → Display, HDR10 PQ with a saved 480-nit white override.
- `viewer-lens-flare.jpg` (and its 1600-pixel copy `viewer-lens-flare-1600.jpg`, the README
  hero): frame 1 of the `robat` RGBA sequence under the built-in Lens Flare OpenFX effect,
  with the Effects, Metadata and Colour panels and the waveform; captured 2026-10-10,
  HDR10 PQ window surface, saved as JPEG.
- Device/backend: NVIDIA GeForce RTX 3080 Ti, Vulkan, HDR10 PQ window surface.
- These PNG captures document UI state; they do not encode a calibrated HDR reproduction.

The displayed image is `ScanLines/Desk.exr` from
[openexr-images](https://github.com/AcademySoftwareFoundation/openexr-images),
revision `e38ffb0790f62f05a6f083a6fa4cac150b3b7452`.
Copyright © 2004 Industrial Light & Magic, a division of Lucasfilm Entertainment
Company Ltd., with contributor copyrights where indicated.
The complete BSD 3-clause [license](openexr-images-LICENSE.txt) is included.
