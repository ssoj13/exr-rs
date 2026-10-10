# Command-line tool

`exr-rs --help` lists every command; `exr-rs <command> --help` gives options and
examples. Bare `exr-rs v` or `exr-rs view` starts the viewer without requiring a path;
persisted-session restoration still applies.
Use `exr-rs v --help`/`-h` or `exr-rs view --help`/`-h` for viewer help.
`--empty`/`-e` remains optional legacy compatibility. Calling the CLI or other subcommands
without arguments prints help.

| Command | Alias | Purpose |
|---|---|---|
| `info` | `i`, `header` | Print every header attribute for every part |
| `convert` | `x` | Re-encode a flat single-part image |
| `cryptomatte` | `c` | Inspect, extract or encode Cryptomatte |
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

Long names work in scripts; aliases are convenient interactively. Legacy OpenEXR
tool names are accepted where provided by `--help`. A leading `-v` or `-vv` raises
logging to debug or trace. `RUST_LOG` overrides it.

## Inspect and convert

```sh
exr-rs info beauty.exr
exr-rs check beauty.exr
exr-rs convert beauty.exr beauty-zip.exr --compress zip
```

`info` reads headers for scanline, tiled, deep and multipart files. `convert`
reads flat single-part images, including level zero of tiled files, and writes
scanline output. Without `--compress`, it keeps the input codec. Deep/multipart
conversion uses appropriate low-level APIs or dedicated tools.

Codec names come from `Compression::ALL`: `none`, `rle`, `zips`, `zip`, `piz`,
`pxr24`, `b44`, `b44a`, `dwaa`, `dwab`, `htj2k256`, `htj2k32`. Lossy codecs can change
pixels when recompressing; header inspection is not a pixel-equivalence check.

## Cryptomatte

```text
exr-rs cryptomatte info beauty.exr
exr-rs cryptomatte extract beauty.exr --name Car --out car.exr
exr-rs cryptomatte encode matte.exr --crypto object --matte Car=car.exr
```

`info` lists types, ranked layers and names. `extract` sums selected coverage and
prints statistics or saves an EXR/raw matte. `encode` builds ranked layers from
named coverage planes. Nested help covers layer, rank and sidecar options.

## View and save a neutral look

```sh
exr-rs v
exr-rs view --help
exr-rs view beauty.exr
exr-rs view --auto beauty.0001.exr
exr-rs view beauty.exr --screenshot beauty.png
exr-rs view beauty.exr --screenshot beauty-pq.png --hdr
exr-rs view beauty.exr --screenshot beauty-light.exr --hdr
```

An explicit file opens alone unless `--auto` is set. `--no-auto` states that choice
explicitly. See [Viewer](viewer.md) for detection, controls and display.

`--screenshot` saves a neutral view without opening a window; it does not capture
desktop/dock panels. SDR output is PNG. With `--hdr`, PNG is 16-bit HDR10 PQ with
BT.2020/cICP/cLLI metadata; EXR stores half-float display light in linear Rec.709
with `whiteLuminance = 203` nits. These are display exports, not a lossless copy of
all original channels. Use `convert` or the [library API](library.md) for that.
