//! `exv-video` — the `.exv` container crate: **tiled temporal video (I/P/B)** on the unified container
//! model (a still is a 1-frame I-GOP; `gop=1` is an all-intra sequence).
//!
//! The implementation is the tiled IBP codec in [`container`] (steps 1a–1d):
//! - per-tile **INTRA / SKIP / PRED** ([`exv_codec::frame`]), coded against a reconstructed reference;
//! - **P-frames** motion-compensated by a global MV ([`motion`]);
//! - **B-frames** (bidirectional): general ref-DAG (refs by decode index) + decode/display reorder,
//!   CLOSED GOPs (no ref crosses an I → bounded random access), prediction = the reversible u64
//!   bit-pattern mean of two MC references;
//! - **memory-bounded streaming** write/read (`W = budget/frame_bytes` window; GOP-parallel decode).
//!
//! The temporal bit-pattern delta is a codec stage in `exv_codec::temporal`. This root re-exports the
//! container's public API so consumers use `exv_video::{write_video, decode_frame, …}` directly.
//!
//! v1: temporal is LOSSLESS-only (open-loop residual); a lossy profile stays I/P + INTRA/SKIP, MV=0.
//! Deep = I-frames only.

pub mod cache;
pub mod container;
pub mod error;
pub mod motion;
pub mod source;
pub mod writer;

pub use container::{
    BadBlob, ChainErr, Decoder, Frame, FrameKind, FrameView, MAGIC, Mv, TemporalClass, TileChain, TileChains, VERSION, VideoIndex, Windows,
    decode_frame, decode_with_refs, frame_body, frame_refs, frame_view, read_still, read_still_with, read_video, read_video_streaming, read_video_with,
    read_windows, still_frame_body, temporal_class, verify_checksums,
};
pub use cache::{CacheError, GopCache};
pub use error::VideoError;
pub use source::{Bytes, ExvSource, open_bytes};
pub use exv_codec::frame::{Chans, Planes};
pub use exv_format::{DecodeError, DecodeLimits, Meta, Verify};
pub use exv_tile::{PlanePool, PlaneStats};
pub use writer::{
    ClipWriter, WriteOpts, transcode, write_clip, write_clip_pull, write_still, write_still_windowed, write_video,
    write_video_meta,
};
