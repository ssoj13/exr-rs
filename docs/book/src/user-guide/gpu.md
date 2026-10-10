# GPU upload (exr-gpu)

`exr-gpu` lands a decoded EXR in GPU memory (wgpu) so it can be sampled, processed
by a compute/render pass, or read back to the CPU. Decoding happens on the CPU in
`exr-core`; `exr-gpu` only uploads. (Running the codecs themselves on the GPU is a
non-goal — the entropy stages are serial and would break the byte-exact invariant.)

## Model

A file is exposed as one or more **parts** (single-part = one part; multipart = one
per image part). `GpuLoader::open` preserves stored channel samples, including
subsampled planes. Use `GpuLoader::open_rgba` to reconstruct luminance-chroma into
one RGBA part. Each part
groups its channels into **layers** (the Nuke dot-naming convention). You pick a
part + layer and upload it.

```rust
use exr_gpu::GpuLoader;

let loader = GpuLoader::open("comp.exr")?;        // stored channels, single or multipart
for part in loader.parts() {
    for layer in part.layers() {
        println!("{} / {}: {} channels", part.name(), layer.label(), layer.len());
    }
}

let part  = loader.default_part();
let layer = part.default_layer().unwrap();
let img   = part.load_layer(layer, &device, &queue, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC)?;
// img.texture / img.view are ready to bind; img.format follows the source type.
```

## Layer → texture mapping

A layer's channel count and sample type determine the texture format:

| Channels | Format family |
|---|---|
| 1 | `R*` |
| 2 | `Rg*` |
| 3 | `Rgba*` (padded, alpha = 1.0 for colour layers) |
| 4 | `Rgba*` |

`Half → *16Float`, `Float → *32Float`, `Uint → *Uint` — never a lossy
down-conversion. A layer with more than four channels, or with mixed sample types,
returns a `FormatMismatch` error. Pixels are uploaded **linear scene-referred**; any
tone-map/colour management is the consumer's job.

## Readback and compute

```rust
// Process on the GPU, then copy the result back to CPU bytes.
let bytes = img.read_back(&device, &queue)?;   // needs COPY_SRC usage

// Or upload all of a part's channels planar into one storage buffer with a
// channel-offset table (the compute escape hatch — any number of channels).
let buf = part.load_buffer(&device, &queue,
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)?;
let loc = buf.channel("N.x").unwrap();          // { offset, len, pixel_type }
let _ = (bytes, loc);
```

For repeated readbacks, reuse one destination Vec with `GpuImage::read_back_into`
rather than allocating a new pixel buffer per frame. `COPY_SRC` must have been
included when creating the texture. Upload limits and bytes use `UploadLayout`;
a GPU device limit is checked before allocation.

`Part::vram_estimate(layer)` returns the bytes a layer upload would allocate — check
it against `wgpu_info::vram_budget_bytes(adapter)` before a bulk load on multi-AOV
files.
