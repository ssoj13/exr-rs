# exv-format

EXV on-disk types: the header, dtypes, codec ids, per-blob tags, decode limits, checksums.

## What it is
Owns the stable discriminants and structural types every other crate refers to: `Header` (+ `Meta`, `TileGrid`),
`Dtype`, `CodecId`, the per-blob backend tags (`classify_blob` / `BlobKind`), the typed decode errors
(`DecodeError`, `OpenError`), the caller-owned decode budget + checksum policy (`DecodeLimits`, `Verify`), CRC-32C,
the 64-byte aligned buffer `AlignedBuf`, f16/bf16 conversion (`half`) and the bounded LE reader `Rd`.
Codec implementations live in `exv-codec`, the `.exv` container in `exv-video`. The byte-level spec is
`docs/FORMAT.md` (derived from this code; the code wins).

## Why / where used
The format root of the dependency graph: used by every `exv-rs` crate except `rans-rs`.
Dependencies: `crc32c`, `bytemuck` only.

## Key API
- `Header`, `Channel`, `Meta`, `TileGrid`, `WriteError`, `VERSION` (= 1), `MAX_TILE_DIM`, `MAX_TILE_AREA` (`header.rs`).
- `Dtype` (F16/F32/BF16/U8/U16/U32), `CodecId` (+ `spec_name`, `is_vendor`) (`types.rs`).
- `blob::{BlobKind, classify_blob}` + `TAG_RAW=0` / `TAG_RANS=1` / `TAG_DEFLATE=2` / `TAG_HTJ2K=3` / `TAG_EXV_RT=4`.
- `DecodeError`, `OpenError`; `DecodeLimits { max_alloc, verify }`, `Verify { On, Off }`, `limits::{cap, alloc_zeroed}`.
- `crc::{crc32c, check, Mismatch}`, `AlignedBuf`, `half::{half_to_f32, f32_to_half, bf16_to_f32, f32_to_bf16}`, `Rd`.

## Notes / gotchas
- Little-endian everywhere; header version `1`, container version `3` (CRC-32C per coded blob). Pre-stable: no layout
  guarantee before the GPU==CPU bit-exact gate.
- `codec_params` is a flat `u16`-length byte blob in the header; `CodecId` is `u32` (`>= 0x8000_0000` = vendor range).
- Per-blob framing is `[tag:u8][payload]`; decode is **tag-driven** — the encode-time entropy policy is never stored.
- Every `match BlobKind` must handle `Htj2k` and `BlockRt` (no tag is feature-gated).
- `Header::read` caps `tile_w`/`tile_h` at `MAX_TILE_DIM` (65536) and the effective tile area at `MAX_TILE_AREA` (2^30 px):
  a decompression-bomb guard. Image `width`/`height` stay uncapped (65k² tiled is a design goal); decode memory is
  bounded by the caller's `DecodeLimits`.
- The per-frame tile index (with its CRCs) is written and parsed by `exv-codec::frame`; see `docs/FORMAT.md` §5.
