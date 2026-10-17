# exv-gpu-kernels

Pure **GPU tile/block decode kernels** for EXV, plus the reusable `GpuContext` that owns the wgpu device/queue/adapter
and the shared compute pipelines. Extracted from `exv-gpu` so the raw WGSL decoders are reusable standalone, with
**no** jph-wgpu, atlas, codec routing or `exv-codec` dependency.

## What's here
- **`GpuContext`** — device + queue + adapter + shared `Pipelines` (compiled once per device, disk pipeline cache via
  gpu-info-rs) + a pooled scratch for decode intermediates. Build it on its own device (`GpuContext::new`, refuses a
  software adapter unless `new_sw()` / `EXV_GPU_ALLOW_SW=1`) or share a host's device (`GpuContext::from_wgpu`).
  `check_fits` / `recommended_tile_size` check tiles against this device's limits (via `exv-tile`).
- **`GpuTile` / `TileEnc`** — a VRAM-resident decoded plane and its word packing; `read_back` copies it out
  (test/export only — the hot path keeps data in VRAM).
- **Batch builders** — `RtBatch` (EXV-RT whole-frame), `storage_parts` / `fill_parts` (parallel staging fill),
  `plan_chunks` (split a batch under the device buffer cap), `blit_format` / `blit_record` (decoded plane → texture).
- **Single-tile decoders** (host wrappers over WGSL kernels, bit-for-bit vs the CPU reference):
  - `decode_block_gpu` — single rANS block.
  - `inverse_wavelet_gpu` — inverse 5/3 lifting.
  - `decode_tile_gpu_wave{,_vram}` / `decode_tile_gpu_fast{,_vram}` — rANS wavelet / reorder+delta tile.
  - `decode_tile_exvrt_vram` / `prepare_tile_exvrt_vram` (+ `ExvrtWarmTile`) — EXV-RT block codec (q=0 and q>0).
- **WGSL constants** — `prelude()` / `wgsl(src)` / `dtype_wgsl()`: shared numbers are Rust `const`s rendered once
  into every shader (no bare duplicates; guard tests enforce it).

## What's NOT here
No HTJ2K (jph-rs / jph-wgpu), no resident atlas, no `compose`, no codec routing / `GpuDecoder`. Those live in
`exv-gpu`, which orchestrates over these kernels.

## Errors
Every entry returns `GpuResult<T>`; `GpuError::{Limit, Corrupt, Unsupported, Device, Budget}`. wgpu validation /
OOM errors are captured by error scopes and returned, never panicked.

## wgpu unification (do not break)
wgpu is pulled **only** through `gpu_info::wgpu` (gpu-info-rs, `branch = "main"`) — the same spec `exv-gpu` and
`jph-wgpu` use — so the whole graph resolves to **one** wgpu (v30) and `exv-gpu` can inject this crate's device into
jph-wgpu (`Gpu::from_parts`).

## Tests
`cargo test -p exv-gpu-kernels -p exv-gpu -- --include-ignored --test-threads=1` (GPU tests are `#[ignore]`; shader
validation tests run without a GPU).
