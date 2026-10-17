//! Per-blob backend framing: the first byte of each encoded blob is the storage-backend tag.
//!
//! # What and why
//! The container stores one blob `[tag:u8][payload]` per (tile×channel), or one per tile for the
//! multi-component HTJ2K layout (FORMAT.md §4). The tag tells HOW to
//! get the decorrelated/raw bytes back: raw, rANS, raw-DEFLATE, or a complete HTJ2K codestream.
//! This is the **container
//! format**, so it lives here (next to the header), not in `exv-codec` — that way both the codec
//! (`encode_channel`/`decode_channel`) and the independent GPU decoder (`exv-gpu`) read the same
//! tag scheme without duplication and without a exv-gpu → exv-codec dependency.
//!
//! Backend selection at encode time (whichever is smaller — rANS or deflate) is done by `exv-codec`;
//! here we have only the stable discriminants and their parsing.

/// Raw: no transform/entropy applied (no-shrink — data is incompressible). payload = the plane.
pub const TAG_RAW: u8 = 0;
/// rANS container over decorrelated bytes. GPU-decodable (hot path).
pub const TAG_RANS: u8 = 1;
/// raw-DEFLATE over decorrelated bytes. CPU-decode only (LZ77 is not GPU-friendly).
pub const TAG_DEFLATE: u8 = 2;
/// Complete HTJ2K (JPEG 2000 Part-15) codestream — its own DWT+entropy, NOT the `decorrelate→entropy` family.
/// CPU-decodable via `jph-rs`; GPU-decodable via `jph-wgpu` (the strategic GPU codec). In a file it is the
/// TILE-level tag of the multi-component layout (ONE codestream carries every channel of the tile, FORMAT.md
/// §4.6); the single-component form (one channel) exists for the per-channel codec API and tests. Resolution
/// levels inside the codestream give the per-tile mip pyramid (LOD).
pub const TAG_HTJ2K: u8 = 3;
/// EXV-RT block codec (`CodecId::EXV_RT`) payload for ONE F16 channel plane (`exv-codec/src/blockrt_dwt.rs`, the
/// shipping payload; `blockrt.rs` is the demoted Lerp reference). Self-contained + directly block-addressable,
/// GPU-decoded as a unit (WGSL) — its own tag, like HTJ2K, not the `decorrelate→entropy` family. LOSSY.
pub const TAG_EXV_RT: u8 = 4;
/// Hardware GPU `BC6H_SF16` (`Bc6hRgbFloat`) 4x4 RGB block payload (`CodecId::BC6H`), either standalone lossy
/// 16-byte hardware blocks or paired with a lossless rANS residual stream (`Bc6hMode::LosslessResidual`).
pub const TAG_BC6H: u8 = 5;

/// Storage backend of a single blob + its payload (without the tag byte).
///
/// Consumers (e.g. the GPU decoder) need to know whether a blob can be decoded on the GPU (`Rans`, `BlockRt`, `Bc6h`) or
/// requires the CPU (`Deflate`), without hardcoding the tag scheme themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobKind<'a> {
    /// [`TAG_RAW`] — raw plane bytes (no transform applied).
    Raw(&'a [u8]),
    /// [`TAG_RANS`] — rANS container over decorrelated bytes (GPU-decodable).
    Rans(&'a [u8]),
    /// [`TAG_DEFLATE`] — raw-DEFLATE over decorrelated bytes (CPU-decode only).
    Deflate(&'a [u8]),
    /// [`TAG_HTJ2K`] — a complete HTJ2K codestream, multi-component in a file (CPU via jph-rs).
    Htj2k(&'a [u8]),
    /// [`TAG_EXV_RT`] — an EXV-RT block-codec payload for one F16 channel plane (CPU `blockrt_dwt` / GPU via the
    /// WGSL block decoder). LOSSY, directly block-addressable.
    BlockRt(&'a [u8]),
    /// [`TAG_BC6H`] — a hardware GPU `BC6H_SF16` 4x4 RGB block payload (with optional lossless rANS residual).
    Bc6h(&'a [u8]),
}

/// Parse a blob's tag byte into backend + payload. `None` on an empty/unknown tag (corrupt input).
pub fn classify_blob(blob: &[u8]) -> Option<BlobKind<'_>> {
    let (tag, rest) = blob.split_first()?;
    match *tag {
        TAG_RAW => Some(BlobKind::Raw(rest)),
        TAG_RANS => Some(BlobKind::Rans(rest)),
        TAG_DEFLATE => Some(BlobKind::Deflate(rest)),
        TAG_HTJ2K => Some(BlobKind::Htj2k(rest)),
        TAG_EXV_RT => Some(BlobKind::BlockRt(rest)),
        TAG_BC6H => Some(BlobKind::Bc6h(rest)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_tags() {
        assert_eq!(classify_blob(&[0, 9, 9]), Some(BlobKind::Raw(&[9, 9][..])));
        assert_eq!(classify_blob(&[1, 2]), Some(BlobKind::Rans(&[2][..])));
        assert_eq!(classify_blob(&[2, 7]), Some(BlobKind::Deflate(&[7][..])));
        assert_eq!(classify_blob(&[3, 4, 5]), Some(BlobKind::Htj2k(&[4, 5][..])));
        assert_eq!(classify_blob(&[4, 8, 9]), Some(BlobKind::BlockRt(&[8, 9][..])));
        assert_eq!(classify_blob(&[5, 1, 2]), Some(BlobKind::Bc6h(&[1, 2][..])));
        assert_eq!(classify_blob(&[6]), None); // unknown tag
        assert_eq!(classify_blob(&[]), None); // empty
    }
}
