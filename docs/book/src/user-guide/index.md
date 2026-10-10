# User Guide

`exr-rs` ships as:

- a **command-line tool** (`exr-rs`) for inspecting and converting EXR files,
  Cryptomatte helpers and whole-image tools;
- a **viewer**, `exr-rs view` — a GPU image/sequence player (library crate `exr-view`);
- a set of **libraries** — `exr-core` (read/write), `exr-gpu` (upload to VRAM), and
  `cryptomatte` (decode/encode) and `exr-util` (OpenEXR whole-image tools);
- a drop-in **Python module** (`openexr-py`, imported as `OpenEXR`), shipped as release
  wheels or built with maturin (see [Installation](installation.md)).

The following pages cover each. Most users want either the
[command-line tool](cli.md) or the [library API](library.md).

## Format support at a glance

| Feature | Read | Write (facade) |
|---|---|---|
| Scanline, single-part | ✅ | ✅ |
| Tiled (mip/rip levels) | ✅ (`read`: level 0; `read_all`: every level) | ✅ (`Image::set_tiling`, every level) |
| Deep (scanline / tiled) | ✅ (low-level) | low-level API |
| Multipart / multiview | ✅ (low-level) | low-level API |
| Luminance-chroma (subsampled) | ✅ (`RgbaInputFile`; `read_all` keeps native sampling) | `RgbaOutputFile`; facade keeps native sampling |
| Codecs: NONE, RLE, ZIP/ZIPS, PIZ, PXR24, B44/B44A, DWAA/DWAB, HTJ2K256/HTJ2K32 | ✅ | ✅ |

The `exr_core::read` / `Image::read_all` / `Image::write` facade covers single-part flat
files; the low-level readers cover deep and multipart (see [Library API](library.md)).
