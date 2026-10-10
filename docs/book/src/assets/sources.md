# Screenshot provenance

These are actual Windows native-window/region captures made with filesystem-mcp-rs
from exr-rs source commit `57b1062390b5e5c400cc120330970ae59be415cf`.
The CLI/viewer was built from a clean isolated checkout with the checked-in lockfile.
No image generation or pixel editing was used.

- `viewer-pq.png`: Desk.exr, viewport checker, metadata, scopes and display settings.
- `display-pq.png`: Settings → Display, HDR10 PQ with a saved 480-nit white override.
- Device/backend: NVIDIA GeForce RTX 3080 Ti, Vulkan, HDR10 PQ window surface.
- These PNG captures document UI state; they do not encode a calibrated HDR reproduction.

The displayed image is `ScanLines/Desk.exr` from
[openexr-images](https://github.com/AcademySoftwareFoundation/openexr-images),
revision `e38ffb0790f62f05a6f083a6fa4cac150b3b7452`.
Copyright © 2004 Industrial Light & Magic, a division of Lucasfilm Entertainment
Company Ltd., with contributor copyrights where indicated.
The complete BSD 3-clause [license](openexr-images-LICENSE.txt) is included.
