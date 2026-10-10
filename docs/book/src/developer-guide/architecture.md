# Architecture

The [shared diagrams](https://github.com/ssoj13/exr-rs/blob/main/docs/DIAGRAMS.md) cover the crate graph, read/write paths,
codec dispatch, deep addressing, viewer pipeline and parity harness.

## Layering

```
imath-rs      deflate-rs      jph-rs (git dep)         foundation ports
     \            |            /
      +------ exr-core -------+                         OpenEXR port + api/ facade + rgba/ + chunk/
             /    |     \
      exr-gpu  cryptomatte  exr-view -- scanseq, scancache, wgpu-info (git deps)
                              |        egui-timeline, egui-statusbar
                        exr-rs CLI     openexr-py (maturin, outside the workspace)
```

Lower crates know nothing of higher ones. `exr-core` is `wgpu`-free; the GPU lives
only in `exr-gpu` and `exr-view`. The viewer reads through the `exr-core` facade
and uses only `exr-gpu`'s checked `UploadLayout` for texture sizing; `exr-gpu` is
otherwise a separate upload library for downstream consumers.

## The port

The whole format stack — the container (header/attributes/IO/chunk layout) and all
codecs — is a fresh 1:1 port written directly from the C++ OpenEXR sources (file
layout from v3.3.11; HTJ2K from 3.4) and converged to byte-exact parity with them.
Behaviour follows **C++ OpenEXR 3.4.15**, the version the parity oracle links: where
3.3.11 and 3.4.15 differ, 3.4.15 wins, and contested behaviour is settled by running the
oracle. `imath-rs` (Imath v3.2.0) supplies the math types.

## Writers and multipart parts

Writers take the pixel bytes on every write call: `set_frame_buffer(&fb)` stores only
the layout (per-channel offsets, strides and types), and `write_pixels(pixels, n)`,
`write_tiles(pixels, ..)` and the deep writers' `write_pixels(counts, samples, n)` get
the bytes each time and bounds-check them. A caller can refill one strip buffer
between calls, which is what streaming very large images needs; there is no lifetime
on the pixel bytes and no snapshot copy. `finish()` returns a fallible result,
checks completeness and propagates final stream errors; callers must handle it.

`MultiPartOutputFile` / `MultiPartInputFile` own the stream and keep one lazily built
writer/reader state per part (C++ caches one object per part). `*output_part(n)` /
`*input_part(n)` return a short-lived view over that state, so interleaved access across
parts resumes where it left off. Only real multipart files carry the `partNumber`
chunk prefix.

## Endianness seam

The low-level `FrameBuffer` is **native-endian**: the gather/scatter in `misc.rs`
converts native ↔ XDR (little-endian on the wire) at the I/O boundary. Every reader
scatters native bytes into the frame buffer and every writer gathers them, so the
`api/` facade lays its planes out native to match. This is the single source of truth
for byte order.

## Codecs

All codecs sit behind the `Compressor` trait; `new_compressor` /
`new_tile_compressor` build one from `Compression` (whose `ALL` list is the single
source of truth for which codecs exist). Chunks are independent, so parallelism is at
the chunk/frame level; the entropy stages inside each codec are serial.

| Codec | Algorithm | Lossy | Lines / chunk | Module |
|---|---|---|---|---|
| NONE | passthrough | — | 1 | — |
| RLE | reorder + delta → run-length | no | 1 | `rle.rs` + `predictor.rs` |
| ZIPS / ZIP | reorder + delta → DEFLATE (deflate-rs; `zipCompressionLevel`, default 4) | no | 1 / 16 | `zip.rs` |
| PIZ | Haar wavelet → Huffman | no | 32 | `piz/` |
| PXR24 | float → 24 bit, delta + byte split → DEFLATE | yes (float) | 16 | `pxr24.rs` |
| B44 / B44A | 4×4 half-block quantisation | yes | 32 | `b44/` |
| DWAA / DWAB | CSC → DCT 8×8 → quantise → Huffman (AC) + zlib (DC) | yes | 32 / 256 | `dwa/` |
| HTJ2K256 / HTJ2K32 | RCT → wavelet → HT block coder (jph-rs) | see note | 256 / 32 | `htj2k.rs` |

HTJ2K is lossless: `Compression::is_lossy` returns false, as in C++ 3.4.15. It
encodes reversibly unless the exr-rs extension attribute `htj2kQuantStep` is set. Deep data accepts only
NONE / RLE / ZIPS. Tiled chunks go through `compress_tile` / `uncompress_tile`, which
PIZ, PXR24, B44, DWA, and HTJ2K override to use the tile's own rectangle.

## The facade vs the low-level readers

`exr_core::api::Image` is a deliberately flat model: one data window, one sample per
pixel per channel, every channel full-resolution. It cannot represent:

- **multipart** — N independent parts, each with its own header and channels;
- **deep** — a variable number of samples per pixel;
- **subsampled** — channels stored at different resolutions (luminance-chroma).

So the facade rejects those with `ExrError::Unsupported(kind)`, whose message points
at the low-level reader (`MultiPartInputFile`, `Deep*InputFile`, `RgbaInputFile`, or
the raw `FrameBuffer` path). Consumers branch on the structured kind, never on the
message text. The low-level readers expose these file shapes; the parity harness checks
covered read/write cases against C++.

## Input hardening

Untrusted sizes flow through single checked boundaries: `api::window_pixel_count`
(window geometry), `Slice::validate_layout` (re-checked at every frame-buffer insert),
`container::xdr::read_bounded_bytes` (attribute sizes vs the stream length),
`deep::sample_table` (cumulative counts, count addresses, chunk sizes before
allocation), and the checked tiled geometry in `tiled/misc.rs`. New code must reuse
these instead of recomputing sizes locally.

## Luminance-chroma (rgba/)

`exr-core/src/rgba/` is a 1:1 port of `ImfRgbaYca` (the 27-tap windowed-sinc chroma
reconstruction filters, `YCAtoRGBA`, and the saturation fix) plus the `FromYca` read
orchestration from `ImfRgbaFile`. `RgbaInputFile` reads a `Y` + subsampled `RY`/`BY`
file and reconstructs full-resolution RGBA. The C++ keeps rotating scanline caches; the
incremental Rust reader uses bounded rotating YCA and RGBA row buffers, reusing
storage as a sequential scan advances. The one-shot bulk decoder instead holds
whole-window planes; these are distinct allocation contracts. `RgbaOutputFile` is the write side
(streaming RGBA → YCA, `set_yc_rounding`); `TiledRgbaInputFile` / `TiledRgbaOutputFile`,
`AcesInputFile` / `AcesOutputFile` and `HalfLut` / `RgbaLut` complete the unit. All are
checked against the C++ oracle.

## exr-gpu

`exr-gpu` decouples *where channels came from* (single-part / multipart / stored subsampled planes, or explicit RGBA reconstruction)
from *channels → GPU* via an internal `SourceImage` (name → pixel type + native
planar bytes + dimensions). Layer grouping mirrors `ChannelList::layers()`. The
load-bearing decision is the **layer → `wgpu::TextureFormat` contract** (see the
[GPU upload](../user-guide/gpu.md) page): arity and sample type pick the format, and
anything that cannot map cleanly is a hard error rather than a silent coercion. One
checked `UploadLayout` sizes upload, readback, and `vram_estimate`, and is validated
against the device's `max_texture_dimension_2d`.

GPU readback uses `copy_texture_to_buffer` into a 256-aligned staging buffer, then
`device.poll(PollType::Wait{..})` (wgpu 30). Texture upload via `write_texture` needs
no row alignment.

## Why decoding stays on the CPU

The serial entropy stages (Huffman, DEFLATE, the HT block coder's passes) are
GPU-hostile, and a GPU decoder would also have to reproduce the C++ output
bit-for-bit. `exr-gpu` therefore uploads CPU-decoded planes; frame-level parallelism
(decoding several frames at once in `scancache`) is the right granularity for
playback. Codec optimization decisions require current profiles and controlled
measurements; historical timings are not a portable throughput guarantee.

## Viewer ownership

The [viewer guide](../user-guide/viewer.md) describes one resident-or-tiled display
pipeline and its shared colour/present path. Immutable decoded frames are cache
owners; viewport state and GPU textures belong to display state. Pending sequence
work keeps the accepted image visible until replacement upload succeeds.

Lazy OpenFX uses checked original-f32 ROI providers and owner-attached CPU admission.
Each physical host allocation retains its lease; copied storage requires new admission.
The viewer prepares an immutable snapshot of geometry, time, primary-source lineage
and deduplicated physical work owners before GPU mutations, then commits it alongside
the accepted drawable GPU overview. Scopes, viewport readout, App metadata, Pixel Info
and export consume that committed snapshot;
external aliases keep its pixel owners alive during delayed export. Charged row
scratch and the shared mixed-LOD area reducer support streaming PNG/EXR without
a full f32 mosaic or display-texture readback.

Consumers attach to that snapshot, and preparation has a bounded budget fallback.
The host's shared request witness records refusal and
monotonic native lifecycle entry in one allocation, inherited by strip and callback
workers. Only a typed intrinsic Budget before BeginSequenceRender/Render and clean
rollback permits a coarser whole level; surviving owners, including zero-byte owners,
retain the original refusal. Pending preserves scale and completed work; successful
identity delivery closes fallback. This does not predict arbitrary in-Render allocation
or guarantee immediate upload. Actual Gaussian/provider, temporal and GPU/native
consumer contracts are covered by host regression tests.
