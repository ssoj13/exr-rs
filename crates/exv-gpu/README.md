# exv-gpu

GPU decode **orchestration** for EXV: whole-`.exv` decode, HTJ2K-on-GPU, the out-of-core resident atlas, and the
plane→RGBA compose pass. Built on the pure kernels in [`exv-gpu-kernels`](../exv-gpu-kernels).

## What it is
Routes each tile/channel of an `.exv` to the right decoder (rANS wave/fast, EXV-RT block, HTJ2K), keeps results in
VRAM (no readback in the hot path) and, for the viewer, decodes only the visible tiles and channels into a
byte-budgeted **resident atlas** that a **compose** pass turns into a linear-RGBA texture for the color pass. Tiles a
GPU kernel cannot take fall back to the CPU decoder (`exv-codec`) explicitly, then are uploaded.

## Where used
`exv-view` (feature `gpu-resident`: resident display + video playback) and `exv-cli` (`bench --gpu`, GPU decode).
Depends on `exv-gpu-kernels`, `exv-codec`/`exv-video` (CPU fallback, container + temporal chains), and, under the
default-on `htj2k-backend` feature, `jph-wgpu-rs` (HTJ2K on GPU) + `wgpu-tile-atlas`.

## Structure
```
gpu.rs      lib root: mod decls + pub-use re-exports
dispatch.rs decode_still_gpu · GpuDecoder · GpuImage/GpuFrame · decode_wave_batched · read_back_image
htj2k.rs    (feat) Htj2kGpuDecoder · decode_htj2k_still_planes_gpu · decode_htj2k_resident · DECODE_BUDGET
atlas.rs    (feat) ExvResidentAtlas · Clip · Want · Planes · TileTex · AtlasStats · ExvDec (batched miss decoders)
atlas_job.rs (feat) async decode: Plan · Done · Worker (dedicated thread) · Flight (in-flight/wanted rules) · Wake · Bytes
compose.rs  (feat) PlaneCompose  (resident planes → linear Rgba32Float)
```

## Key API
- `decode_still_gpu(ctx, data, &limits) -> GpuImage` — whole still on GPU (LosslessFast / LosslessWave / EXV-RT);
  HTJ2K returns `GpuDecodeError::NoGpuDecoder` (use the HTJ2K entry points or the atlas).
- `GpuDecoder::decode_still(bytes, &limits) -> GpuFrame` — `channel_buffer`, `to_rgba_texture`, `read_back_rgba`.
- `ExvResidentAtlas` — `supports(profile, dtypes, level)` (capability gate), `request(Clip, frame, Want{tiles, level,
  chans}, content, display)` (synchronous) or the async split `plan` -> `submit` -> `poll` + `install` (+ `with_wake`,
  `cancel_all`; one decode worker thread), `get(level, rx, ry)`, `stats() -> AtlasStats` (`Display` for a status bar). Temporal clips:
  INTRA + SKIP tiles are reused across frames; PRED / motion / B tiles are a typed error.
- `compose::PlaneCompose` — composes resident planes into the color pass's input.
- Re-exported from `exv-gpu-kernels`: `GpuContext`, `GpuError`, `GpuTile`, `read_back`, `decode_tile_gpu_*`, etc.

## Notes
- Every entry point takes the caller's `DecodeLimits` (byte budget + CRC `Verify` policy); over budget is
  `Budget`, not a device limit.
- **wgpu:** only through `gpu_info::wgpu` (one wgpu, v30, shared with jph-wgpu, so the device can be injected).
- **`htj2k-backend`** mirrors `exv-codec`'s; enable HTJ2K through THIS feature. `--no-default-features` drops HTJ2K +
  the atlas/compose modules together.
- Lossless paths are **bit-for-bit == the CPU reference**; EXV-RT q>0 too (table-based log decode, no hardware `exp2`).
- Full-resolution HTJ2K tiles decode on the CPU and are uploaded (measured faster than jph-wgpu today); LOD requests
  and `with_htj2k_gpu(true)` use jph-wgpu. Numbers: `docs/OPT.md` (sections H, I).
