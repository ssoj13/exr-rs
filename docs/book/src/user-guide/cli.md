# Command-line tool

`exr-rs --help` lists every command; `exr-rs <command> --help` gives options and
examples. Calling the CLI or any subcommand except `view` without arguments prints
help; bare `view` starts the [viewer](viewer.md).

| Command | Alias | Purpose |
|---|---|---|
| `info` | `i`, `header` | Print every header attribute for every part |
| `convert` | `x` | Re-encode a flat single-part image, or read a PNG/DPX/HDR/... still into EXR |
| `cryptomatte` | `c` | Inspect, extract, encode or inject Cryptomatte |
| `view` | `v` | Start the viewer, optionally with an image or sequence |
| `check` | `k` | Check file structure and pixel decoding |
| `tile` | `t` | Create a tiled image with optional MIP/RIP levels |
| `preview` | `p` | Add a header thumbnail |
| `attr` | `a` | Set standard or custom attributes |
| `multipart` | `m` | Combine, separate or convert multipart images |
| `multiview` | `s` | Combine views into a stereo/multiview image |
| `envmap` | `e` | Convert and filter environment maps |
| `metrics` | `b` | Measure codec read/write time and size |
| `manifest` | `n` | Inspect ID manifests |
| `aces` | `A` | Convert to an ACES image container |

Long names work in scripts; aliases are convenient interactively. The OpenEXR tool
names (`exrheader`, `exrcheck`, `exrmaketiled`, `exrmakepreview`, `exrstdattr`,
`exrmultipart`, `exrmultiview`, `exrenvmap`, `exrmetrics`, `exrmanifest`, `exr2aces`)
also work as hidden aliases. A `-v` or `-vv` raises logging to debug or trace;
`RUST_LOG` overrides it.

## Inspect and convert

```sh
exr-rs info beauty.exr
exr-rs check beauty.exr
exr-rs convert beauty.exr beauty-zip.exr --compress zip
exr-rs convert plate.0001.dpx plate.0001.exr
```

`info` reads headers for scanline, tiled, deep and multipart files (a non-EXR still gets a
short format/size/channels summary). `convert` also reads every non-EXR still the viewer opens
(PNG, JPEG, TIFF, DPX, HDR, TGA, ...; the [colour rule](viewer.md#other-image-formats)) and writes
it as a scanline EXR the way `oiiotool` does: the values as read, alpha associated, the colour tag
as the `colorInteropID` attribute. `convert`
re-encodes a flat single-part image and keeps its storage shape: scanline or tiled
(tile size, level mode, rounding), every MIP/RIP level, and subsampled channels at
their native sampling. Without `--compress`, it keeps the input codec. Deep and
multipart inputs are rejected; use the `multipart` tool or the low-level APIs.

Codec names come from `Compression::ALL`: `none`, `rle`, `zips`, `zip`, `piz`,
`pxr24`, `b44`, `b44a`, `dwaa`, `dwab`, `htj2k256`, `htj2k32`. Lossy codecs can change
pixels when recompressing; header inspection is not a pixel-equivalence check.

## Cryptomatte

```text
exr-rs cryptomatte info beauty.exr
exr-rs cryptomatte extract beauty.exr --name Car --out car.exr
exr-rs cryptomatte encode matte.exr --crypto CryptoObject --matte Car=car.exr
exr-rs cryptomatte inject beauty.exr out.exr --name Prop --mask prop.exr
```

`info` lists types, ranked layers and names. `extract` sums selected coverage and
prints statistics or saves an EXR/raw matte. `encode` builds ranked layers from
named coverage planes. `inject` adds a named coverage to an existing type, over or
under the existing ids. Nested help covers layer, rank, sidecar and compression options.

## Save a neutral view

```sh
exr-rs view beauty.exr --screenshot beauty.png
exr-rs view beauty.exr --screenshot beauty-pq.png --hdr
exr-rs view beauty.exr --screenshot beauty-hlg.png --hdr --hlg
exr-rs view beauty.exr --screenshot beauty-light.exr --hdr
```

See [Viewer](viewer.md) for launching the viewer, sequence detection, controls and display.

`--screenshot` saves a neutral view without opening a window; it does not capture
desktop/dock panels. SDR output is PNG. With `--hdr`, PNG is 16-bit HDR10 PQ with
BT.2020/cICP/cLLI metadata (`--hlg`: 16-bit HLG with cICP, for the BT.2100 1000-nit
reference display); EXR stores half-float display light in linear Rec.709
with `whiteLuminance = 203` nits. These are display exports, not a lossless copy of
all original channels. Use `convert` or the [library API](library.md) for that.
