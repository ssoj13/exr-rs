//! [`ExvSource`]: a native `.exv` frame as an [`exv_tile::TileSource`].
//!
//! # Why
//! Every out-of-core consumer (tiled-EXR bridge, region reads, the viewer's tile cache, the GPU atlas) needs "decode
//! tile (tx,ty) of this frame". Before, each re-parsed the container and the full tile index per call. An
//! `ExvSource` parses both ONCE, owns the file bytes (an `Arc`, so worker threads share it), and serves tiles
//! through the codec's single [`FrameReader`] path. The raw blob seam [`ExvSource::tile_blob`] is a zero-copy
//! sub-slice of the mapped file, which is what the GPU decoders consume without any CPU decode.
//!
//! Only INTRA frames: a P/B frame's tiles reference another frame and are not independently decodable (FORMAT.md
//! section 5); they stay on the [`crate::decode_frame`] / [`crate::Decoder`] paths.

use std::ops::Range;
use std::sync::Arc;

use exv_codec::Profile;
use exv_codec::frame::{FrameIndex, FrameReader};
use exv_codec::registry::TileLayout;
use exv_format::{DecodeError, DecodeLimits, Dtype, Header, TileGrid};
use exv_tile::{TileBuf, TileError, TileResult, TileSource, check_tile};

use crate::container::{FrameKind, VideoIndex, frame_body};

/// Shared read-only container bytes: a read-through `Vec` below 64 MiB or a file map above
/// ([`exv_format::fsio::open_ro`]), or any in-memory buffer. `Arc` so tile workers share one copy.
pub type Bytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// Open `path` read-only as shared [`Bytes`] ([`exv_format::fsio::open_ro`]: the ONE place a container file is mapped).
///
/// # Errors
/// [`exv_format::fsio::Error::Read`] (names the path) when the file cannot be opened, read or mapped.
pub fn open_bytes(path: &std::path::Path) -> Result<Bytes, exv_format::fsio::Error> {
    Ok(Arc::new(exv_format::fsio::open_ro(path)?))
}

/// One INTRA frame of an `.exv` served tile by tile.
pub struct ExvSource {
    data: Bytes,
    /// The frame body inside `data` (header-less tile blobs + per-tile index).
    body: Range<usize>,
    idx: FrameIndex,
    profile: Profile,
    limits: DecodeLimits,
}

/// A codec failure of tile `(tx, ty)` (channel `ci` if the read was per-channel) as the tile engine's error.
fn dec_err(tx: u32, ty: u32, ci: Option<usize>, e: DecodeError) -> TileError {
    match e {
        DecodeError::Limit { .. } | DecodeError::Alloc(_) => TileError::Limit(e.to_string().into()),
        e => TileError::Decode { tx, ty, ci, why: e.to_string().into() },
    }
}

impl ExvSource {
    /// Frame `frame` (DISPLAY index) of the container `data` already parsed into `vidx`.
    ///
    /// # Errors
    /// [`TileError::NotIntra`] for a P/B frame, [`TileError::Limit`] for an out-of-range frame,
    /// [`TileError::Decode`] when the frame body or its tile index is corrupt or the codec cannot be opened.
    pub fn open(data: Bytes, vidx: &VideoIndex, frame: usize) -> TileResult<Self> {
        let f = frame as u32;
        match vidx.frame_kind(frame) {
            None => return Err(TileError::Limit(format!("frame {frame} out of range").into())),
            Some(FrameKind::Intra) => {}
            Some(_) => return Err(TileError::NotIntra { frame: f }),
        }
        let bytes: &[u8] = (*data).as_ref();
        let (header, body) = frame_body(bytes, vidx, frame)
            .ok_or_else(|| TileError::Decode { tx: 0, ty: 0, ci: None, why: "corrupt frame body".into() })?;
        // `body` is a sub-slice of `bytes`: its offset is pure pointer arithmetic (no self-referential borrow).
        let start = body.as_ptr() as usize - bytes.as_ptr() as usize;
        let range = start..start + body.len();
        let idx = FrameIndex::parse(body, header).map_err(|e| dec_err(0, 0, None, e))?;
        let profile = idx.profile().map_err(|e| dec_err(0, 0, None, DecodeError::Codec(e)))?;
        Ok(Self { data, body: range, idx, profile, limits: vidx.limits() })
    }

    /// The first frame of a still (a 1-frame `.exv`): parses the container, then [`Self::open`].
    ///
    /// # Errors
    /// As [`Self::open`], plus [`TileError::Decode`] for a malformed container.
    pub fn open_still(data: Bytes) -> TileResult<Self> {
        let vidx = {
            let bytes: &[u8] = (*data).as_ref();
            VideoIndex::parse(bytes).map_err(|e| dec_err(0, 0, None, e))?
        };
        Self::open(data, &vidx, 0)
    }

    /// Per-tile decode budget (default [`DecodeLimits::default`]; [`Self::open`] inherits the index's).
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The container header of this frame.
    pub fn header(&self) -> &Header {
        self.idx.header()
    }

    /// The parsed per-tile index (kind / rect per tile).
    pub fn index(&self) -> &FrameIndex {
        &self.idx
    }

    fn body(&self) -> &[u8] {
        // `body` was derived from `data` at open and `data` is immutable: always in range.
        (*self.data).as_ref().get(self.body.clone()).unwrap_or(&[])
    }

    fn reader(&self) -> FrameReader<'_> {
        FrameReader::over(self.body(), &self.idx).with_limits(self.limits)
    }

    /// Row-major tile index of `(tx, ty)` after the bounds check.
    fn tile_no(&self, tx: u32, ty: u32) -> TileResult<u32> {
        check_tile(&self.idx.header().grid(), tx, ty)?;
        Ok(ty * self.idx.header().tiles_x() + tx)
    }

    /// The raw self-describing blob of tile `(tx, ty)`: a ZERO-COPY sub-slice of the container bytes (the unit the
    /// GPU decoders consume without a CPU decode).
    ///
    /// # Errors
    /// [`TileError::Oob`] for a bad index, [`TileError::Decode`] when the entry points outside the body or the blob
    /// fails its CRC (verified under this source's [`DecodeLimits::verify`] policy).
    pub fn tile_blob(&self, tx: u32, ty: u32) -> TileResult<&[u8]> {
        let t = self.tile_no(tx, ty)?;
        self.reader().tile_blob(t).map_err(|e| dec_err(tx, ty, None, e))
    }
}

impl TileSource for ExvSource {
    fn grid(&self) -> TileGrid {
        self.idx.header().grid()
    }

    fn dtypes(&self) -> &[Dtype] {
        self.idx.dtypes()
    }

    fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()> {
        let t = self.tile_no(tx, ty)?;
        out.planes = self.reader().decode_tile(t, &self.profile, None).map_err(|e| dec_err(tx, ty, None, e))?;
        Ok(())
    }

    /// Per-channel codecs (rANS / wavelet / EXV-RT) decode only channel `ci` (the viewer shows 3 of 43 AOVs); HTJ2K
    /// is one multi-component codestream, so the whole tile decodes once and `ci` is picked.
    fn read_chan(&self, tx: u32, ty: u32, ci: usize, out: &mut Vec<u8>) -> TileResult<()> {
        let n = self.idx.dtypes().len();
        if ci >= n {
            return Err(TileError::Chan { ci, n });
        }
        let t = self.tile_no(tx, ty)?;
        let r = self.reader();
        *out = if self.profile.backend().layout() == TileLayout::MultiComp {
            let tiles = r.decode_tile(t, &self.profile, None).map_err(|e| dec_err(tx, ty, Some(ci), e))?;
            tiles.into_iter().nth(ci).ok_or(TileError::Chan { ci, n })?
        } else {
            r.decode_tile_channel(t, ci).map_err(|e| dec_err(tx, ty, Some(ci), e))?
        };
        Ok(())
    }
}
