# Library API

`exr-core` exposes two layers: a small idiomatic **facade** for the common case, and
the **low-level readers/writers** for everything else.

## The facade

`exr_core::read` / `Image::write` handle single-part, full-resolution, flat-channel
files — the common case.

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
Tiled files are read at level 0; the facade always writes scanline files.

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

Deep readers first expose sample counts. Size channel heaps for the requested
row/tile rectangle before reading samples; when writing into a larger reusable
heap, adjust the slice base offset for that requested region.

## Channel layers (Nuke-style)

EXR channel names encode layers with dots (`diffuse.R`, `depth.Z`, `N.x`). The
ported `ChannelList` exposes `layers()` and `channels_in_layer()` to group them.

## Low-level readers

The facade returns `ExrError::Unsupported(kind)` for files it does not model — deep,
multipart, and subsampled. `kind` is the structured `exr_core::Unsupported`
(`Multipart` / `Deep` / `Subsampled`; `exr_core::classify_unsupported` extracts it), and
the error message names the reader to use instead — branch on the kind, not the text:

| File shape | Reader |
|---|---|
| Multipart | `MultiPartInputFile` (per-part `input_part` / `tiled_input_part` / `deep_*_input_part`) |
| Deep scanline / tiled | `DeepScanLineInputFile` / `DeepTiledInputFile` |
| Subsampled (luminance-chroma) | `rgba::RgbaInputFile` — reconstructs full-resolution RGBA |
| Arbitrary subsampled planes | the low-level `FrameBuffer` + `scanline::InputFile` / `TiledInputFile` |

The flat `Image` model cannot represent N independent parts, variable per-pixel
sample counts, or mixed-resolution channels — hence the split. The low-level readers
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
