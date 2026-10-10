# Library API

`exr-core` exposes two layers: a small idiomatic **facade** for the common case, and
the **low-level readers/writers** for everything else.

## The facade

`exr_core::read` / `Image::read_all` / `Image::write` handle single-part flat
(non-deep) files — the common case.

```rust
use exr_core::{read, ChannelData};
use exr_core::attr::Compression;

let img = read("beauty.exr")?;
println!("{}x{}", img.width(), img.height());

for (name, data) in img.channels() {
    match data {
        ChannelData::Half(v)  => { /* &[imath_rs::Half] */ }
        ChannelData::Float(v) => { /* &[f32] */ }
        ChannelData::Uint(v)  => { /* &[u32] */ }
    }
    let _ = name;
}

img.write("out.exr", Compression::Zip)?;
```

`Image` holds the data/display windows, the compression and line order, and an
ordered map of channel name → typed pixel plane (row-major over the data window).
`read` returns level 0 of a tiled file as one flat raster and rejects subsampled
channels. `Image::read_all` reads the file as stored: tile description, every
MIP/RIP level, and subsampled channels at their native sampling. `Image::write`
writes scanline, or tiled with every level when a tiling is set
(`Image::set_tiling`), so `read_all` + `write` keeps the storage shape.

Related helpers: `Image::write_with_attrs` (extra header attributes),
`exr_core::write_rgba_file` (quick RGBA write), and the `chunk` module
(`RawImage` / `RawChunk`) for byte-exact EXR→EXR transcodes without decoding.

`Image` also retains header attributes. Use `write_with_attrs` when deliberately
adding or overriding metadata; source data/display windows and pixel aspect ratio
matter independently of the pixel plane's dimensions.

## Streaming and ownership

Low-level writers retain a `FrameBuffer` layout, not a borrowed pixel snapshot.
Pass native-endian bytes to each `write_pixels(pixels, rows)` or `write_tiles` call;
a row/strip buffer can be refilled and reused between calls. `Slice` validates
sampling, strides and address bounds at the shared boundary.

Call the writer's fallible `finish()` and propagate its error. Scanline output
checks that every row was written and flushes the stream before returning its
borrowed/owned stream holder. Dropping a writer is not a successful finish signal.
The `new_owned` constructors retain owned streams through the same implementation.
Multipart part views reuse persisted per-part state instead of rebuilding writers.

Deep readers first expose sample counts; size the channel heaps from them before
reading samples. The sample heap starts at the first **requested** scanline or tile
rectangle: samples do not reserve a prefix for unrequested rows or tiles, while counts
follow the configured count slice. If successive requests fill one larger heap, set
each channel slice's `base_offset` to its destination before each read. For tiled
reads, channel tile-coordinate flags restart the heap within that tile along the
selected axis; count-slice flags select count storage independently. These rules also
apply to nonzero data windows and multipart readers.

## Channel layers (Nuke-style)

EXR channel names encode layers with dots (`diffuse.R`, `depth.Z`, `N.x`). The
ported `ChannelList` exposes `layers()` and `channels_in_layer()` to group them.

## Low-level readers

The facade returns `ExrError::Unsupported(kind)` for files it does not model — deep
and multipart (and, for `read`, subsampled). `kind` is the structured `exr_core::Unsupported`
(`Multipart` / `Deep` / `Subsampled`; `exr_core::classify_unsupported` extracts it), and
the error message names the reader to use instead — branch on the kind, not the text:

| File shape | Reader |
|---|---|
| Multipart | `MultiPartInputFile` (per-part `input_part` / `tiled_input_part` / `deep_*_input_part`) |
| Deep scanline / tiled | `DeepScanLineInputFile` / `DeepTiledInputFile` |
| Subsampled (luminance-chroma) | `rgba::RgbaInputFile` — reconstructs full-resolution RGBA |
| Arbitrary subsampled planes | `Image::read_all`, or the low-level `FrameBuffer` + `scanline::InputFile` / `TiledInputFile` |

The flat `Image` model cannot represent N independent parts or variable per-pixel
sample counts — hence the split. The low-level readers
expose these file shapes; covered cases are checked against the C++ oracle.

### Luminance-chroma example

```rust
use exr_core::rgba::RgbaInputFile;

let img = RgbaInputFile::read("luma_chroma.exr")?;   // Y + subsampled RY/BY → RGBA
let (w, h) = (img.width(), img.height());
for px in &img.pixels { /* Rgba { r, g, b, a: Half } */ }
let _ = (w, h);
```

## Errors and logging

Fallible library APIs report invalid input through `ExrError`; propagate their results. The libraries log
through the `log` facade; a binary installs a backend (`env_logger`) and sets the
level.
