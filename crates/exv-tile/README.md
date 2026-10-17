# exv-tile

EXV out-of-core tile engine: a fallible tile source trait, a byte-budget tile cache, a whole-frame plane recycler,
the memory budget formulas, and the region assembler.

## What it is
Lets the pipeline process 65k² images without materializing the whole image or running out of memory. Tiles are
pulled on demand into reused buffers; caches reserve bytes **before** decoding, so concurrent decodes can never
jointly exceed the budget. RAM/VRAM sizes come from the OS (via `gpu-info-rs`) or an explicit override — an
unanswerable query is an error, never a guessed default.

## Where used
`exv-codec` (windowed parallel encode/decode, plane recycling), `exv-video` (`ExvSource`, clip plane pool, region
reads), `exv-gpu-kernels` (tile-vs-device checks), `exv-view` (RAM/VRAM planning), `exv-capi` (frame cache), `exv-cli`
(budgets, streaming, region reads).

## Key API
- `TileSource` (`Send + Sync`, fallible: `grid()`, `dtypes()`, `read_tile(tx, ty, &mut TileBuf)`), `PlanesSource`
  (foreign linear image), `stream_tiles` (one reused buffer, peak `O(tile)`), `pipe` (ordered windowed parallel loop),
  `par_window` (window size in bytes).
- `TilePool<K, V>` — byte-LRU cache: `Lease` pins, single-flight loads, reserve-before-decode, cancel, prefetch
  (never evicts); over budget with everything pinned = `TileError::OverBudget`.
- `PlanePool` / `PlaneBuf` — recycles 64-byte-aligned whole-frame plane buffers by exact length.
- `Budget` (shared, live-resizable cap; `Budget::auto` = half the RAM, capped by what is free; `limits()` -> the
  `DecodeLimits` decoders enforce), `usable_ram`, `frame_cache`, `frame_share`, `par_for`, `window_spare`, `Stage`,
  `plan_vram` (resident-GPU VRAM split), `heap_slack`.
- `read_region` — the one rectangle assembler (+ processing halo, `Edge::Clamp|Zero`, strict or fault-tolerant `OnErr`).
- `fits` / `pick_tile_size` / `DeviceLimits` — tile size vs GPU limits.

## Notes
- Halo is for **processing only** (resampling, filters): the codec transform is tile-local by definition and never
  uses halo.
- Native `.exv` tiles are read by `exv_video::ExvSource` (a zero-copy blob slice for the GPU); foreign linear images use
  `PlanesSource` (strided gather).
