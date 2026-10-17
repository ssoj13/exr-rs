//! `exv-format` — EXV container types (on-disk format + stable serializable discriminants).
//!
//! # What it is
//! Owns the **stable discriminants written into the container** and the structural types
//! everything else refers to: [`Header`] (+ [`TileGrid`]), the per-channel [`Dtype`], the codec
//! id [`CodecId`], and the per-blob backend tags ([`classify_blob`]). Codec *implementations*
//! live in `exv-codec` (keeps `exv-codec → exv-format → exv-codec` acyclic).
//!
//! # Authoritative spec
//! The byte-level specification of every structure is **`docs/FORMAT.md`** (kept in lockstep
//! with this code). Short summary of the ACTUAL current layout:
//! - **Little-endian** on the wire, everywhere. Version bytes = pre-stable (layout not
//!   guaranteed until the GPU==CPU gate). All field parsing goes through the bounded reader [`Rd`].
//! - [`Header`] (`header.rs`): version · flags (bit0 = [`Meta`] TLV block) · width · height ·
//!   frame_count · tile_w · tile_h · `codec_id:u32` · `codec_params` (a `u16`-length flat byte blob) ·
//!   channels (`name_len:u8` + name bytes + `dtype:u8`) · optional Meta block.
//! - `.exv` (`exv-video::container`, MAGIC `EXV1`): prelude → embedded `Header` → frames (frame_hdr +
//!   header-less tiled body of `exv-codec::frame`) → outer frame index → `index_off:u64` trailer.
//! - Per blob: `[tag:u8][payload]`, tag ∈ {RAW=0, RANS=1, DEFLATE=2, HTJ2K=3, EXV_RT=4}; decode is tag-driven (the
//!   encode-time `EntropyPolicy` is NOT stored on disk).
//! - [`limits`]: [`DecodeLimits`] (caller-owned decode byte budget + the [`Verify`] checksum policy), plus the re-exported
//!   `bufpool` helpers `cap` (capacity hints) and `alloc_zeroed` (fallible allocation; its `AllocError` converts into
//!   [`DecodeError::Alloc`]); [`DecodeError`] is the one typed decode error.
//! - [`AlignedBuf`]: the 64-byte (cache line) aligned byte buffer of planes and codec scratch, owned by `bufpool` and
//!   re-exported here so every codec path names it through the format crate.
//! - [`crc`]: CRC-32C of every tile blob (stored in the frame tile index), the one checksum helper.

pub mod blob;
pub mod bufpool;
pub mod crc;
pub mod error;
pub mod fsio;
pub mod half;
pub mod header;
pub mod limits;
pub mod rd;
pub mod types;
pub use blob::{BlobKind, TAG_BC6H, TAG_DEFLATE, TAG_EXV_RT, TAG_HTJ2K, TAG_RANS, TAG_RAW, classify_blob};
pub use bufpool::AlignedBuf;
pub use error::{DecodeError, OpenError};
pub use header::{Channel, Header, MAX_TILE_AREA, MAX_TILE_DIM, Meta, TileGrid, VERSION, WriteError};
pub use limits::{DecodeLimits, Verify};
pub use rd::Rd;
pub use types::{CodecId, Dtype};
