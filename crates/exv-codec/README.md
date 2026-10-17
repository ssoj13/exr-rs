# exv-codec

EXV codecs (reorder+delta, 5/3 wavelet, lossy wavelet, HTJ2K, EXV-RT) behind a static codec registry, plus the
tiled frame-body reader/writer.

## What it is
Implements the `[temporal predict?] → [spatial transform] → [entropy]` pipeline and the static `registry`
(one `TileCodec` per codec id; `registry::open` validates a header's `codec_id` + `codec_params`).
Depends on `exv-format` (types, blob tags, limits), `exv-tile` (byte windows, plane pool) and `rans-rs` (entropy).

## Why / where used
The codec heart of `exv-rs`. Used by `exv-video` (the `.exv` container), `exv-gpu` (verified blob accessors +
CPU fallback), `exv-capi`, `exv-cli` and `exv-view`.

## Key API
- `Profile` (transform + rANS config + `EntropyPolicy`): `lossless_fast`, `lossless_wave`, `lossy_wave(q)`,
  `htj2k`, `htj2k_lossy(q)`, `htj2k_ldr(q)`, `exv_rt`, `exv_rt_dwt`, `.max_ratio()`.
- `Transform` — `ReorderDelta`, `Wavelet53`, `WaveletLossy{q}`, `Htj2k(Htj2kMode)` (`Lossless` | `HdrCodes{q}` |
  `Irrev97{q}`, feature `htj2k-backend`), `BlockRt(BlockRtParams)`.
- `EntropyPolicy` — `GpuHot` (default: prefer rANS, keeps the GPU path reachable) vs `MaxRatio` (smaller of
  rANS/deflate per blob, CPU decode).
- `encode_channel` / `decode_channel`, `encode_tile` / `decode_tile` (+ `_into` variants writing into caller buffers).
- `frame` — `write_frame_tiles`, `write_intra_frame_body_windowed`, `read_frame_tiles`, `FrameIndex`, `FrameReader`
  (on-demand tile/channel/LOD decode, `decode_all_into` with `Planes` + `Chans` channel subsets, CRC-32C verified blob accessors).
- `registry` — `open`, `TileCodec`, `Dst`, `GpuKind`, `TileLayout`.
- `htj2k` (JPEG 2000 Part 15 via `jph-rs`), `hdr` (log2 value-domain transform for HDR-float lossy), `blockrt` /
  `blockrt_dwt` (EXV-RT realtime block codec), `image` (`Image`, `RgbaPick` display channel pick).
- Re-exports `exv_format::{BlobKind, classify_blob}`.

## Notes / gotchas
- **HTJ2K is the default codec** (feature `htj2k-backend`, on by default). In a workspace build enable it on the top
  crate (`exv-gpu`, `exv-video`, `exv-cli` forward it), not on `exv-codec` alone.
- HTJ2K is a complete codec, not a decorrelate→entropy stage: one multi-component codestream per tile, every dtype
  (U8/U16 as plain unsigned components). `Irrev97` takes all-U8/U16 tiles only; anything else is `CodecError::Dtype`.
- Wavelet transforms are tile-local **without halo**; halo exists only in `exv-tile` for processing.
- Lossy floats go through `hdr`'s log2 codes (quantizing raw float bits corrupts the exponent). Integer channels stay
  lossless under every lossy profile.
- Decode never panics: errors are `DecodeError`, memory is bounded by the caller's `DecodeLimits`, and a jph-rs panic
  is caught at the `catch_unwind` boundary.
