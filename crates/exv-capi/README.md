# exv-capi

C ABI (ABI version 5) over exv-codec / exv-video — the FFI seam for native hosts (After Effects, ...).

## What it is
A flat C ABI (`cdylib` + `staticlib`, built as `exv_capi.dll` / `libexv_capi.so` / `exv_capi.a`) wrapping
`exv-codec`, `exv-video` and `exv-tile`. It is the project's only FFI boundary: the core crates stay native Rust, and
native plugins link this library. The shape mirrors the sibling `apv-rs::apv-capi` (path-session model) without
YCbCr/CICP/color management, since EXV is already linear RGBA-float / arbitrary planar.

## Header
`docs/exv-capi.h` is GENERATED from `src/capi.rs` by cbindgen (`cbindgen.toml`); the `header_in_sync` test
regenerates it and fails on any drift. Regenerate: `EXV_BLESS_HEADER=1 cargo test -p exv-capi header_in_sync`.
Struct sizes are pinned on both sides (Rust `const` asserts + header static asserts).

## Key API
- `exv_version`, `exv_abi_version` (compare with `EXV_ABI_VERSION` at load), `exv_last_error_message` (thread-local).
- `exv_session_open` / `_info` / `_frame_info` / `_meta` / `_decode` / `_channel_count` / `_channel_name` /
  `_decode_mapped_rgbaf32` / `_close` — LAZY decode session over an `.exv` (a still is a 1-frame `.exv`): only the
  seek index is parsed at open; frames decode on demand, only the channels a call reads, through one process-wide
  decoded-frame cache bounded by half of RAM (capped by free RAM). Safe to call concurrently on one handle.
- `ExvPixelFormat::{RgbaF32, RgbaF16}` — output formats, honoring a caller byte stride. Other value sets:
  `ExvDtype`, `ExvCodec`, `ExvCodecMode`, `ExvMetaFlag`, `ExvStatus` (`0` = OK).
- `exv_encode_rgba_f32` — still encode into an opaque `ExvBuffer`; codec by id (1 LosslessFast, 2 Wavelet53,
  3 Wavelet53Lossy, 4 EXV-RT, 6 HTJ2K) + `codec_mode` + `q` (inclusive limits `EXV_Q_WAVE_MAX`, `EXV_Q_HDR_MAX`,
  `EXV_Q_LDR_MAX`, `EXV_Q_RT_MAX`); unknown ids are `UNSUPPORTED`, out-of-range `q` is `INVALID_ARGUMENT`.
- `exv_venc_open(path, params, meta, gop, search)` / `_add_rgba_f32` / `_finish` / `_free` — STREAMING clip
  encode: each frame is encoded and appended to a temp file as it arrives (memory = one frame), `finish` commits
  atomically, `free` of an unfinished encoder aborts and deletes the temp. Clip meta (fps/PAR/...) goes into `open`.
- `exv_buffer_data` / `exv_buffer_len` / `exv_buffer_free` — opaque output buffer.
- `exv_write_file` — Unicode-correct, atomic file write (Rust owns I/O in both directions).

## Notes
- Every status-returning export is panic-safe (`catch_unwind` -> `Internal`).
- Paths may be plain paths or `file://` URLs (percent-decoded, UNC supported).
- Corrupted tile blobs are detected by CRC-32C and reported as `INVALID_FORMAT` naming frame/tile/channel.
- Sessions on files < 64 MiB are read into RAM rather than memory-mapped (no Windows lock when re-exporting over
  them); close a session on a path before overwriting that path.
- Consumer: `cpp/exv-ae` (loads `exv_capi.dll` at runtime, refuses an ABI mismatch).
