//! Shared tile-blob primitives for the `.exv` tile codec.
//!
//! The small self-contained set the frame-body reader
//! ([`crate::frame::FrameReader`]) builds on: the self-describing per-tile blob splitter and the tile [`Rect`]
//! (the whole-frame decode writes tile rows straight into the planes; region assembly lives in `exv_tile::read_region`). The unified still/video container lives in `exv-video`.
//!
//! A tile blob is `[chan_len[nch]:u32 LE][ch0_blob][ch1_blob]…[ch{nch-1}_blob]` for per-channel codecs
//! (rANS/wavelet), or `[TAG_HTJ2K][one multi-component codestream]` for HTJ2K (no per-channel slice).

use exv_format::limits::cap;
use exv_format::{Header, Rd};

/// A pixel rectangle `(x, y, w, h)` — one tile's extent, or the covered region of a tile range. A named
/// struct (not a 4-tuple + loose args) keeps the gather/scatter helpers under clippy's argument limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    /// The pixel rect of tile `(tx, ty)` (edge tiles clamped by [`Header::tile_rect`]).
    pub(crate) fn of_tile(header: &Header, tx: u32, ty: u32) -> Self {
        let (x, y, w, h) = header.tile_rect(tx, ty);
        Self { x, y, w, h }
    }
}

/// Split a self-describing tile blob (format v2) into its `nch` per-channel sub-blobs. The blob is
/// `[chan_len[nch]:u32 LE][ch0_blob]…[ch{nch-1}_blob]`: read the `nch` lengths from the front, then
/// return the `nch` sub-slices. Fully bounds-checked ([`Rd`]) — a forged length table or a short/truncated
/// blob yields `None`, never a panic or an `nch`-sized allocation the blob cannot back.
pub(crate) fn split_tile_blob(blob: &[u8], nch: usize) -> Option<Vec<&[u8]>> {
    let mut r = Rd::new(blob);
    let mut lens = Vec::with_capacity(cap(nch, blob.len(), 4));
    for _ in 0..nch {
        lens.push(r.u32()? as usize);
    }
    let mut subs = Vec::with_capacity(lens.len());
    for l in lens {
        subs.push(r.take(l)?);
    }
    Some(subs)
}

#[cfg(test)]
mod tests {
    use super::split_tile_blob;

    /// A forged channel count / length table never allocates by it nor panics: `None`.
    #[test]
    fn split_tile_blob_rejects_forged_tables() {
        assert!(split_tile_blob(&[0xff; 8], usize::MAX).is_none());
        assert!(split_tile_blob(&[0xff; 8], 2).is_none(), "lengths past the end");
        assert!(split_tile_blob(&[], 1).is_none());
        let ok = [1u8, 0, 0, 0, 2, 0, 0, 0, 0xaa, 0xbb, 0xcc];
        assert_eq!(split_tile_blob(&ok, 2), Some(vec![&[0xaa][..], &[0xbb, 0xcc][..]]));
    }
}
