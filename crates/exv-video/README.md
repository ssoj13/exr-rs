# exv-video

The `.exv` container: tiled temporal video (I/P/B) and stills in one format — a still is a 1-frame I-GOP
(`gop=1` is an all-intra sequence).

## What it is
- Per-tile INTRA/SKIP/PRED coding from `exv_codec::frame`; every coded tile blob carries a CRC-32C
  (container version 3).
- P-frames are motion-compensated by a global MV (`motion` module); B-frames use a ref-DAG (refs by
  decode index) with CLOSED GOPs for bounded random access.
- Memory-bounded streaming write/read: windows of `budget / frame_bytes` frames, GOP-parallel decode,
  deterministic bytes for any thread count.
- `open_bytes` over the shared `fsio` crate (ssoj13/fsio-rs: read-through/mmap open, atomic temp-file writes).

Depends on `exv-format`, `exv-codec`, `exv-tile`. Feature `htj2k-backend` (default-on) forwards to
`exv-codec/htj2k-backend`. Used by `exv-cli`, `exv-capi`, `exv-view`, `exv-gpu`.

## Key API
- Write: `ClipWriter` (push frames into any `Write + Seek`, count patched at `finish` — for hosts that
  stream), `write_clip`, `write_clip_pull` (loader callback, bounded RAM), `transcode`,
  `write_video[_meta]`, `write_still`, `write_still_windowed` (out-of-core still). Errors: `VideoError`.
- Read: `VideoIndex::parse` (+ `with_limits` / `with_pool`), `decode_frame` (random access, decodes only the
  frame's ancestors, optional channel subset via `Chans`), `frame_refs` + `decode_with_refs` (one-frame
  decode from cached references), `read_video[_with]`, `read_video_streaming`, `read_windows`,
  `read_still[_with]`, `Decoder` (playback cache).
- Integrity: `verify_checksums` (scan every blob CRC without decoding); policy = `DecodeLimits.verify`.
- Tiles / GPU seams: `ExvSource` (`exv_tile::TileSource` over an INTRA frame), `still_frame_body`,
  `frame_body`, `frame_view`, `temporal_class`, `TileChains` / `TileChain`.
- Re-exports `Chans`, `Planes`, `DecodeError`, `DecodeLimits`, `Meta`, `Verify`, `PlanePool`, `PlaneStats`.

## Notes
- Temporal delta on float channels operates on the **bit pattern**, never float subtraction. Temporal
  coding is lossless-only; a lossy profile emits INTRA/SKIP tiles only (MV 0).
- Encode is open-loop (prediction from source frames), so a window encodes in parallel.
- Layout: `prelude (MAGIC "EXV1", version, gop, frame_count) | header (+Meta) | frames | outer index | index_off:u64`;
  outer index = `n:u32` + per frame `kind:u8, offset:u64, display:u32`. Byte spec: `docs/FORMAT.md` §6.
- Deep data is I-frames only (not implemented yet).
