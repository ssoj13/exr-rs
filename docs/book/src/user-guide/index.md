# User Guide

`exr-rs` ships as:

- a **command-line tool** (`exr-rs`) for inspecting and converting EXR files,
  Cryptomatte helpers, whole-image tools and launching the viewer;
- a **viewer** (`exr-view`) — a GPU image/sequence player;
- a set of **libraries** — `exr-core` (read/write), `exr-gpu` (upload to VRAM), and
  `cryptomatte` (decode/encode) and `exr-util` (OpenEXR whole-image tools);
- a drop-in **Python module** (`openexr-py`, imported as `OpenEXR`), built with maturin
  (see the repo-root `README.md`).

The following pages cover each. Most users want either the
[command-line tool](cli.md) or the [library API](library.md).

## Format support at a glance

| Feature | Read | Write (facade) |
|---|---|---|
| Scanline, single-part | ✅ | ✅ |
| Tiled (mip/rip levels) | ✅ (facade: level 0) | low-level API |
| Deep (scanline / tiled) | ✅ (low-level) | low-level API |
| Multipart / multiview | ✅ (low-level) | low-level API |
| Luminance-chroma (subsampled) | ✅ (`RgbaInputFile`) | — |
| Codecs: NONE, RLE, ZIP/ZIPS, PIZ, PXR24, B44/B44A, DWAA/DWAB, HTJ2K256/HTJ2K32 | ✅ | ✅ |

The convenience `exr_core::read`/`Image::write` facade covers the common single-part
full-resolution case; the low-level readers cover everything else (see
[Library API](library.md)).
