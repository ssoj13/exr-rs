//! `EXV1` tiled container — **steps 1a–1d** (per-tile INTRA/SKIP/PRED; I/P/B frames, global MV).
//! Byte layout: `docs/FORMAT.md` §6.
//!
//! A video is a sequence of frames in DECODE order on top of ONE authoritative [`Header`] (dims / tile
//! grid / channels / codec). Frames are coded against references BY DECODE INDEX (the general ref-DAG);
//! `display_index` restores presentation order. Each frame is a header-less tiled body
//! ([`exv_codec::frame`]) coded against a reconstructed reference the container builds:
//! - **I** (intra): no reference (all tiles INTRA). Starts a GOP — self-contained (IDR-like); no ref
//!   crosses it, so a GOP decodes independently → bounded random access.
//! - **P**: 1 ref = an earlier decoded frame, motion-compensated by a global MV ([`crate::motion`]).
//! - **B**: 2 refs (a past + a future frame in display order), each MC by its own MV, then AVERAGED
//!   (reversible integer bit-pattern mean); tiles code the residual vs that mean. B is emitted AFTER
//!   both refs in decode order (hence the reorder). Temporal is LOSSLESS-only (open-loop); a lossy
//!   profile stays I/P with MV=0 + INTRA/SKIP tiles.
//!
//! # Container (LE)
//! ```text
//! MAGIC "EXV1" | version:u8 | gop:u32 | frame_count:u32 | Header (authoritative, codec baked in)
//!   | frames[N] in DECODE order, each:
//!       frame_hdr { kind:u8, n_refs:u8, refs:[u32(decode idx); n_refs], mvs:[(i16,i16); n_refs] }
//!       frame_body  (exv_codec::frame::write_frame_tiles — tile blobs + per-tile index + trailer)
//!   | outer index { n_frames:u32, per frame: kind:u8, offset:u64, display_index:u32 }
//!   | index_off:u64 (trailer)
//! ```
//! `n_refs` is fixed by `kind` (Intra 0 / Pred 1 / Bidir 2); every ref is an EARLIER frame of the SAME GOP.
//!
//! # Hostile input + memory budgets
//! [`VideoIndex::parse`] validates the whole outer structure before anything is allocated from it (the
//! index must fill exactly `[index_off, trailer)`), but NEVER judges the declared image size: a legitimate
//! 20000² all-zero image is a ~1.6 KB file, indistinguishable from a bomb by any compression-ratio guess.
//! Memory is bounded where it is actually materialized, by the CALLER's [`DecodeLimits`]
//! ([`VideoIndex::with_limits`], default [`DecodeLimits::default`]): decoded planes of a frame
//! ([`exv_codec::frame`]), the resident GOP of [`decode_frame`]/[`Decoder`], and the output of
//! [`read_video_with`] — each returns [`DecodeError::Limit`] over budget. Per-frame refs are validated where
//! they are used ([`decode_one`]).

use crate::error::VideoError;
use crate::motion;
use exv_format::bufpool::AllocError;
use exv_codec::Profile;
use exv_codec::frame::{self, Chans, Planes, RefPlanes};
use exv_codec::image::Image;
use exv_format::rd::off;
use exv_format::{CodecId, DecodeError, DecodeLimits, Dtype, Header, Meta, Rd};
use exv_tile::{Budget, PlaneBuf, PlanePool};
use rayon::prelude::*;
use tracing::info_span;

/// Signature of the tiled `.exv` container: ASCII "EXV1" in LE.
pub const MAGIC: u32 = u32::from_le_bytes(*b"EXV1");
/// Container version. `3` = the tiled/IBP container whose embedded `Header` has no magic (header layout `VERSION` 1)
/// and whose frame tile index carries a CRC-32C per tile blob (`3` replaced `2` = the same layout without CRCs).
pub const VERSION: u8 = 3;
/// Bytes per outer-index entry: `kind:u8 + offset:u64 + display_index:u32`.
pub(crate) const IDX_ENTRY: usize = 13;
/// Bytes of the prelude before the embedded [`Header`]: `MAGIC(4) | version(1) | gop(4) | frame_count(4)`.
pub(crate) const PRELUDE_LEN: usize = 13;
/// Offset of the prelude's `frame_count` (patched by the streaming writer at `finish`). The header's own
/// copy sits at `PRELUDE_LEN + exv_format::header::FRAME_COUNT_OFF`.
pub(crate) const PRELUDE_COUNT_OFF: usize = 9;

/// Frame prediction kind (FROZEN discriminants).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    /// Intra — self-contained, no references (GOP anchor).
    Intra = 0,
    /// Forward-predicted (P) — 1 reference + a global MV.
    Pred = 1,
    /// Bidirectional (B) — 2 references (past+future) + a global MV each; predicted by their mean.
    Bidir = 2,
}

impl FrameKind {
    fn from_u8(v: u8) -> Option<FrameKind> {
        match v {
            0 => Some(FrameKind::Intra),
            1 => Some(FrameKind::Pred),
            2 => Some(FrameKind::Bidir),
            _ => None, // unknown kind → hard error (no silent fallback)
        }
    }

    /// Number of references a frame of this kind carries (fixed by the kind; see the layout above).
    const fn n_refs(self) -> usize {
        match self {
            FrameKind::Intra => 0,
            FrameKind::Pred => 1,
            FrameKind::Bidir => 2,
        }
    }
}

/// A frame's motion vector `(dx, dy)` (i16 on the wire). A non-zero MV makes a SKIP tile mean "the SHIFTED
/// reference tile", which reads neighbouring reference tiles: such a clip is no longer tile-local ([`TemporalClass::Motion`]).
pub type Mv = (i16, i16);
/// A parsed `frame_hdr`: `(kind, refs as decode indices, one MV per ref)`.
type FrameHdr = (FrameKind, Vec<u32>, Vec<Mv>);

/// Whether two headers share dims + channels — a video's frames must be homogeneous.
pub(crate) fn same_shape(a: &Header, b: &Header) -> bool {
    a.width == b.width && a.height == b.height && a.channels == b.channels
}

/// Serialize a frame header (`kind`, `n_refs`, `refs` as decode indices, one `mv` per ref).
pub(crate) fn write_frame_hdr(out: &mut Vec<u8>, kind: FrameKind, refs: &[u32], mvs: &[Mv]) {
    debug_assert!(refs.len() == kind.n_refs() && mvs.len() == refs.len());
    out.push(kind as u8);
    out.push(refs.len() as u8); // <= 2
    for &r in refs {
        out.extend_from_slice(&r.to_le_bytes());
    }
    for &(dx, dy) in mvs {
        out.extend_from_slice(&dx.to_le_bytes());
        out.extend_from_slice(&dy.to_le_bytes());
    }
}

/// Parse a frame header at the reader; returns `(kind, refs, mvs)`. `n_refs` must equal the kind's
/// fixed count (`None` otherwise — no silent tolerance of extra/missing refs).
fn read_frame_hdr(r: &mut Rd) -> Option<FrameHdr> {
    let kind = FrameKind::from_u8(r.u8()?)?;
    let n_refs = r.u8()? as usize;
    if n_refs != kind.n_refs() {
        return None;
    }
    let refs = (0..n_refs).map(|_| r.u32()).collect::<Option<Vec<_>>>()?;
    let mvs = (0..n_refs).map(|_| r.i16().zip(r.i16())).collect::<Option<Vec<_>>>()?;
    Some((kind, refs, mvs))
}

/// `true` iff `planes` is exactly one plane per channel of `header`, each `w*h*esz` bytes — the shape every
/// motion-compensation helper relies on, validated once here instead of indexing blindly.
pub(crate) fn planes_match(planes: &[Vec<u8>], header: &Header) -> bool {
    planes.len() == header.channels.len()
        && planes.iter().zip(&header.channels).all(|(p, c)| header.plane_len(c.dtype) == Some(p.len()))
}

/// One reference channel, motion-compensated: borrowed for a zero MV, else a shifted / averaged copy in a plane of a
/// [`PlanePool`] (`shift_into` / `avg_into` overwrite every byte; the buffer goes back to the pool when the reference
/// drops, so MV != 0 and B-frame clips allocate no planes per frame in steady state, on the read AND the write side).
pub(crate) enum RefChan<'r> {
    Borrowed(&'r [u8]),
    Pooled(PlaneBuf),
}

impl RefChan<'_> {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            RefChan::Borrowed(b) => b,
            RefChan::Pooled(p) => p,
        }
    }
}

/// `p` shifted by `mv` (edge-clamped): borrowed for `(0,0)`, else a pooled copy. `Ok(None)` = `p` is not a
/// `w*h*esz` plane of `header` (the caller names the error); `Err` = the pool's allocation failed.
fn shift_chan<'a>(pool: &PlanePool, p: &'a [u8], header: &Header, dtype: Dtype, (dx, dy): Mv) -> Result<Option<RefChan<'a>>, AllocError> {
    if (dx, dy) == (0, 0) {
        return Ok(Some(RefChan::Borrowed(p)));
    }
    let mut out = pool.take(p.len())?;
    Ok(motion::shift_into(p, header.width, header.height, dtype, dx as i32, dy as i32, &mut out).map(|()| RefChan::Pooled(out)))
}

/// Per-element bit-pattern MEAN of two channels into a pooled plane (`Ok(None)` = the planes disagree in length /
/// element width).
fn avg_chan(pool: &PlanePool, a: &RefChan<'_>, b: &RefChan<'_>, dtype: Dtype) -> Result<Option<RefChan<'static>>, AllocError> {
    let mut out = pool.take(a.bytes().len())?;
    Ok(motion::avg_into(&mut out, a.bytes(), b.bytes(), dtype.size()).map(|()| RefChan::Pooled(out)))
}

/// Writer side: the motion-compensated reference of `prev`, one [`RefChan`] per channel (lossless for any MV; `mv =
/// (0,0)` BORROWS `prev`, no copy). The planes are validated for BOTH paths (a zero MV must not smuggle a malformed
/// plane past the checks).
///
/// # Errors
/// [`VideoError::Shape`] if the planes are inconsistent with the header, [`VideoError::Alloc`] on a pool failure.
pub(crate) fn mc_shift<'a>(pool: &PlanePool, prev: &'a [Vec<u8>], header: &Header, mv: Mv) -> Result<Vec<RefChan<'a>>, VideoError> {
    if !planes_match(prev, header) {
        return Err(VideoError::Shape(MC_SHAPE));
    }
    let chans = header.channels.iter().zip(prev);
    chans.map(|(ch, p)| shift_chan(pool, p, header, ch.dtype, mv)?.ok_or(VideoError::Shape(MC_SHAPE))).collect()
}

/// Reversible integer bit-pattern MEAN of two (already MC) frames — the B prediction. Per element the
/// value is `(a + b) >> 1` computed in `u64` (so a `U32`/`F32` sum can't overflow). Deterministic ⇒
/// encode and decode compute the identical mean; the tile residual makes reconstruction exact. Per channel the
/// kernel is [`motion::avg_into`] (const-width chunked loops) into a pooled plane.
///
/// # Errors
/// [`VideoError::Shape`] if either input does not match the header, [`VideoError::Alloc`] on a pool failure.
pub(crate) fn mc_avg(pool: &PlanePool, a: &[RefChan<'_>], b: &[RefChan<'_>], header: &Header) -> Result<Vec<RefChan<'static>>, VideoError> {
    let ok = |c: &[RefChan<'_>]| c.len() == header.channels.len() && c.iter().zip(&header.channels).all(|(p, ch)| header.plane_len(ch.dtype) == Some(p.bytes().len()));
    if !ok(a) || !ok(b) {
        return Err(VideoError::Shape(MC_SHAPE));
    }
    let chans = header.channels.iter().zip(a.iter().zip(b));
    chans.map(|(ch, (pa, pb))| avg_chan(pool, pa, pb, ch.dtype)?.ok_or(VideoError::Shape(MC_SHAPE))).collect()
}

/// The shape error of the writer-side MC helpers.
const MC_SHAPE: &str = "reference planes do not match the header";

/// `Some(v)` or the typed "corrupt stream" error naming the structure that was missing/short.
fn need<T>(o: Option<T>, what: &'static str) -> Result<T, DecodeError> {
    o.ok_or(DecodeError::Corrupt(what))
}

/// Parsed `.exv` seek-index: authoritative header + codec + per-frame (kind, display index, offset).
pub struct VideoIndex {
    /// Authoritative clip header (dims / tile grid / channels / codec).
    pub header: Header,
    /// Codec profile reconstructed from `header.codec_id`/`codec_params`.
    profile: Profile,
    /// `gop` cadence hint (`>= 1`).
    pub gop: u32,
    /// Per DECODE-order frame: prediction kind.
    kinds: Vec<FrameKind>,
    /// Per DECODE-order frame: its DISPLAY position.
    displays: Vec<u32>,
    /// Inverse of `displays` (a validated permutation): DISPLAY position → DECODE index.
    decode_of: Vec<usize>,
    /// Per DECODE-order frame: byte offset of its `frame_hdr` (strictly increasing, inside the frames region).
    offsets: Vec<usize>,
    /// End of the frames region (== start of the outer index).
    frames_end: usize,
    /// Optional clip metadata (fps / color / HDR-transfer / timecode); `None` if the file carries none.
    meta: Option<Meta>,
    /// The caller's decode budget (see [`Self::with_limits`]); every decode through this index obeys it.
    limits: DecodeLimits,
    /// THE clip's plane recycler: every frame decoded through this index takes its planes here and gives them back
    /// when dropped (a freed P-chain reference, an evicted GOP / cache entry), so steady-state playback allocates,
    /// zero-fills and page-faults nothing. Shared by clones of the pool ([`Self::with_pool`]).
    pool: PlanePool,
    /// Idle cap of the DEFAULT pool, kept equal to `limits.max_alloc` (the most one decode may hold: an idle shelf
    /// beyond it could never be taken by this index's decodes at once). A caller pool has its own budget.
    pool_cap: Budget,
}

impl VideoIndex {
    /// Parse the header + outer seek-index. Validates the structure only (magic/version, counts that agree,
    /// an index that fills exactly `[index_off, trailer)`, strictly increasing in-range offsets, a display
    /// permutation, an Intra first frame, a known codec) — it does NOT judge the declared image size; that
    /// is the decode budget's job ([`Self::with_limits`]). Nothing is allocated from a file-supplied count
    /// before the index length has been proven to match it.
    ///
    /// # Errors
    /// [`DecodeError::Corrupt`] (naming the structure) or [`DecodeError::Codec`] for a codec that cannot be opened.
    pub fn parse(data: &[u8]) -> Result<VideoIndex, DecodeError> {
        let mut r = Rd::new(data);
        if need(r.u32(), "magic")? != MAGIC || need(r.u8(), "version")? != VERSION {
            return Err(DecodeError::Corrupt("not an EXV1 container (magic/version)"));
        }
        let gop = need(r.u32(), "prelude")?;
        if gop == 0 {
            return Err(DecodeError::Corrupt("gop == 0")); // writers always emit >= 1
        }
        let frame_count = need(r.u32(), "prelude")?;

        let (header, meta, hlen) = Header::read_with_meta(need(data.get(r.pos()..), "header")?)?;
        if header.frame_count != frame_count {
            return Err(DecodeError::Corrupt("frame count stored twice and the copies disagree"));
        }
        need(header.frame_bytes(), "frame byte size overflows")?; // addressable, nothing more
        // Typed open-time codec validation (unknown / vendor / not built / bad params): `DecodeError::Codec`.
        let profile = exv_codec::registry::open(CodecId(header.codec_id), &header.codec_params)?;
        let head_end = need(r.pos().checked_add(hlen), "header length")?;

        let trailer = need(data.len().checked_sub(8), "trailer")?;
        let index_off = need(off(need(Rd::at(data, trailer).u64(), "trailer")?), "index offset")?;
        if index_off < head_end || index_off > trailer {
            return Err(DecodeError::Corrupt("index offset outside the file"));
        }
        let mut ir = Rd::at(data, index_off);
        let n = need(usize::try_from(need(ir.u32(), "index")?).ok(), "index count")?;
        if Some(n) != usize::try_from(frame_count).ok() {
            return Err(DecodeError::Corrupt("index count != frame count"));
        }
        // The index must fill EXACTLY [index_off, trailer): rejects a forged `n` before any allocation
        // (n * 13 bytes must really be present) and any trailing garbage between index and trailer.
        if n.checked_mul(IDX_ENTRY).and_then(|b| b.checked_add(4)) != Some(trailer - index_off) {
            return Err(DecodeError::Corrupt("index length != 4 + 13 * frame count"));
        }
        let pool_cap = Budget::new(DecodeLimits::default().max_alloc);
        let (mut kinds, mut displays, mut offsets) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let mut decode_of = vec![usize::MAX; n];
        for i in 0..n {
            let kind = need(FrameKind::from_u8(need(ir.u8(), "index entry")?), "unknown frame kind")?;
            let frame_off = need(off(need(ir.u64(), "index entry")?), "frame offset")?;
            let disp = need(usize::try_from(need(ir.u32(), "index entry")?).ok(), "display index")?;
            // Frame offsets lie in the frames region and strictly increase (a frame's end is the next start).
            if frame_off < head_end || frame_off >= index_off || offsets.last().is_some_and(|&p| frame_off <= p) {
                return Err(DecodeError::Corrupt("frame offsets not strictly increasing inside the frames region"));
            }
            if i == 0 && kind != FrameKind::Intra {
                return Err(DecodeError::Corrupt("decode must start at an I-frame"));
            }
            // Every display position is produced exactly once (a valid permutation).
            let slot = need(decode_of.get_mut(disp), "display index out of range")?;
            if *slot != usize::MAX {
                return Err(DecodeError::Corrupt("display index repeated"));
            }
            *slot = i;
            kinds.push(kind);
            displays.push(disp as u32);
            offsets.push(frame_off);
        }
        Ok(VideoIndex {
            header,
            profile,
            gop,
            kinds,
            displays,
            decode_of,
            offsets,
            frames_end: index_off,
            meta,
            limits: DecodeLimits::default(),
            pool: PlanePool::new(pool_cap.clone()),
            pool_cap,
        })
    }

    /// Replace the decode budget: the caller's byte ceiling for what any one decode call through this index
    /// may hold at once (a frame's planes + one tile, the resident GOP, a whole clip's output).
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self.pool_cap.set(limits.max_alloc);
        self
    }

    /// Decode into `pool` instead of the index's own (one pool shared by every source of a host, e.g. a viewer's frame
    /// cache whose evicted frames feed the next decodes; its budget caps the idle bytes).
    pub fn with_pool(mut self, pool: PlanePool) -> Self {
        self.pool = pool;
        self
    }

    /// The plane pool every decode through this index uses ([`PlanePool::stats`] proves reuse).
    pub fn pool(&self) -> &PlanePool {
        &self.pool
    }

    /// Number of frames.
    pub fn frame_count(&self) -> usize {
        self.offsets.len()
    }

    /// DECODE-order index of the frame shown at DISPLAY position `k` (`None` if out of range).
    pub fn decode_idx(&self, k: usize) -> Option<usize> {
        self.decode_of.get(k).copied()
    }

    /// The clip's self-describing metadata (fps / color / HDR-transfer / timecode), or `None`.
    pub fn meta(&self) -> Option<&Meta> {
        self.meta.as_ref()
    }

    /// Prediction kind of the frame shown at DISPLAY position `k` (`None` if out of range).
    pub fn frame_kind(&self, k: usize) -> Option<FrameKind> {
        self.kinds.get(self.decode_idx(k)?).copied()
    }

    /// The decode budget every decode through this index obeys ([`Self::with_limits`]).
    pub fn limits(&self) -> DecodeLimits {
        self.limits
    }

    /// Bytes of ONE decoded frame (all channel planes); `parse` proved this representable.
    fn frame_bytes(&self) -> u64 {
        self.header.frame_bytes().unwrap_or(u64::MAX)
    }

    /// Err([`DecodeError::Limit`]) unless `frames` decoded frames fit the budget at once.
    pub(crate) fn check_frames(&self, frames: usize) -> Result<(), DecodeError> {
        self.limits.check(self.frame_bytes().saturating_mul(frames as u64))
    }
}

/// `(kind, refs, mvs, body)` of DECODE-order frame `di`.
fn frame_parts<'a>(data: &'a [u8], idx: &VideoIndex, di: usize) -> Result<(FrameHdr, &'a [u8]), DecodeError> {
    let start = *need(idx.offsets.get(di), "frame index out of range")?;
    let end = idx.offsets.get(di + 1).copied().unwrap_or(idx.frames_end);
    let mut r = Rd::at(data, start);
    let (kind, refs, mvs) = need(read_frame_hdr(&mut r), "frame header")?;
    Ok(((kind, refs, mvs), need(data.get(r.pos()..end), "frame body")?))
}

/// Borrow-by-decode-index lookup over a table of already decoded frames — the `get` of [`decode_one`].
/// Slot 0 of `decoded` is decode index `base` (GOP-relative tables hold only their own frames).
fn by_idx<'a>(decoded: &'a [Option<Planes>], base: usize) -> impl Fn(u32) -> Option<&'a Planes> + 'a {
    move |r| decoded.get((r as usize).checked_sub(base)?)?.as_ref()
}

/// Reconstruct DECODE-order frame `di` of the GOP starting at decode index `gs`, restricted to `chans`, into planes from
/// the clip's [`PlanePool`]. `get(r)` yields the already-decoded planes of decode index `r` (decoded with the same or a
/// wider channel set). Builds the reference per wanted channel (INTRA none; P shifts one ref; B averages two MC refs;
/// unwanted channels stay `None`, so a subset decode never touches them) and decodes. The ONE decode routine of every
/// read path: every ref must be an earlier frame of the same GOP (`gs <= r < di`, the closed-GOP invariant), else a
/// `Corrupt` error.
pub(crate) fn decode_one<'r>(
    data: &[u8],
    idx: &VideoIndex,
    di: usize,
    gs: usize,
    chans: Chans<'_>,
    get: impl Fn(u32) -> Option<&'r Planes>,
) -> Result<Planes, DecodeError> {
    let ((kind, refs, mvs), body) = frame_parts(data, idx, di)?;
    let _sp = tracing::debug_span!("decode_one", di, ?kind).entered();
    if refs.iter().any(|&r| !(gs..di).contains(&(r as usize))) {
        tracing::warn!(di, gs, ?refs, "reference is not an earlier frame of its GOP");
        return Err(DecodeError::Corrupt("reference is not an earlier frame of its GOP"));
    }
    let hdr = &idx.header;
    // The i-th reference, motion-compensated by its own MV, one entry per header channel.
    let mc_ref = |i: usize| -> Result<Vec<Option<RefChan<'r>>>, DecodeError> {
        let r = *need(refs.get(i), "missing reference")?;
        let (dx, dy) = *need(mvs.get(i), "missing motion vector")?;
        let planes = need(get(r), "reference frame not decoded")?;
        const BAD: DecodeError = DecodeError::Corrupt("reference planes do not match the header");
        hdr.channels
            .iter()
            .enumerate()
            .map(|(ci, ch)| {
                if !chans.has(ci) {
                    return Ok(None);
                }
                let p = need(planes.get(ci), "reference channel not decoded")?;
                if hdr.plane_len(ch.dtype) != Some(p.len()) {
                    return Err(BAD);
                }
                Ok(Some(need(shift_chan(&idx.pool, p, hdr, ch.dtype, (dx, dy))?, "reference planes do not match the header")?))
            })
            .collect()
    };
    let reference: Option<Vec<Option<RefChan<'r>>>> = match kind {
        FrameKind::Intra => None,
        FrameKind::Pred => Some(mc_ref(0)?),
        FrameKind::Bidir => {
            let (a, b) = (mc_ref(0)?, mc_ref(1)?);
            let mean = a.iter().zip(&b).zip(&hdr.channels).map(|((a, b), ch)| match (a, b) {
                (Some(a), Some(b)) => Ok(Some(need(avg_chan(&idx.pool, a, b, ch.dtype)?, "B reference planes do not match the header")?)),
                _ => Ok(None),
            });
            Some(mean.collect::<Result<_, DecodeError>>()?)
        }
    };
    let refs: Option<RefPlanes<'_>> = reference.as_ref().map(|r| r.iter().map(|p| p.as_ref().map(RefChan::bytes)).collect());
    let reader = info_span!("open_index").in_scope(|| frame::FrameReader::open(body, hdr.clone()))?.with_limits(idx.limits);
    let mut out = Planes::pooled(idx.pool.clone());
    reader.decode_all_into(&idx.profile, refs.as_ref(), chans, &mut out)?;
    Ok(out)
}

/// One decoded frame: the clip header (`frame_count = 1`) + its planes. The planes come from (and return to, on drop)
/// the clip's [`PlanePool`] ([`VideoIndex::with_pool`]); a channel the decode did not select ([`Chans`]) is `None`,
/// never a zero plane. THE output of the per-frame entry points ([`decode_frame`], [`decode_with_refs`]); the
/// whole-clip readers hand out [`Image`]s ([`Frame::into_image`]).
#[derive(Debug)]
pub struct Frame {
    /// The clip header with `frame_count = 1` (every channel, decoded or not).
    pub header: Header,
    /// One slot per header channel.
    pub planes: Planes,
}

impl Frame {
    /// Wrap decoded planes under `header` (as a single frame).
    pub(crate) fn new(header: &Header, planes: Planes) -> Frame {
        let mut header = header.clone();
        header.frame_count = 1;
        Frame { header, planes }
    }

    /// The frame as an [`Image`]: ONE copy of each plane into a `Vec` (an `Image` owns unaligned `Vec` planes); the
    /// 64-byte aligned pooled planes then go back to the clip's pool, so the next decode reuses them.
    ///
    /// # Errors
    /// [`DecodeError::Request`] when a channel was not decoded (a [`Chans::Only`] frame), [`DecodeError::Corrupt`] when the
    /// planes do not match the header.
    pub fn into_image(self) -> Result<Image, DecodeError> {
        image_of(self.header, &self.planes)
    }

    /// Borrow-convert the frame into an [`Image`] (for [`crate::GopCache`] leases that remain pinned in the pool).
    pub fn to_image(&self) -> Result<Image, DecodeError> {
        image_of(self.header.clone(), &self.planes)
    }
}

/// `planes` copied into an [`Image`] under `header` (all channels or [`DecodeError::Request`]).
fn image_of(header: Header, planes: &Planes) -> Result<Image, DecodeError> {
    let copy = (0..planes.len()).map(|ci| planes.get(ci).map(<[u8]>::to_vec)).collect::<Option<Vec<_>>>();
    let copy = copy.ok_or(DecodeError::Request("frame was decoded with a channel subset"))?;
    // A decoder that returns planes that do not fit its own header is a codec bug or a corrupt stream.
    Image::new(header, copy).map_err(|_| DecodeError::Corrupt("decoded planes do not match the header"))
}

/// An [`Image`] as an (unpooled, all-channel) [`Frame`]: lets image consumers (`exv render`) share the frame code.
impl From<Image> for Frame {
    fn from(img: Image) -> Frame {
        Frame { header: img.header, planes: Planes::from(img.planes) }
    }
}

/// One decoded frame as an [`Image`] (its header has `frame_count = 1`); the planes go back to the pool after the copy.
fn as_frame(header: &Header, planes: Planes) -> Result<Image, DecodeError> {
    image_copy(header, &planes)
}

/// A COPY of still-needed planes as an [`Image`] (the pooled originals stay resident as a GOP cache entry / the next
/// P-frame's reference).
fn image_copy(header: &Header, planes: &Planes) -> Result<Image, DecodeError> {
    let mut header = header.clone();
    header.frame_count = 1;
    image_of(header, planes)
}

/// Decode index of the I-frame that starts the GOP containing decode frame `di`.
pub(crate) fn gop_start(idx: &VideoIndex, di: usize) -> usize {
    (0..=di).rev().find(|&i| idx.kinds.get(i) == Some(&FrameKind::Intra)).unwrap_or(0)
}

/// The decode-order range `[start, end)` of the GOP (I-anchored, self-contained) containing decode
/// frame `di`. GOPs are independent (no ref crosses an I), so this range decodes standalone.
fn gop_range(idx: &VideoIndex, di: usize) -> (usize, usize) {
    let start = gop_start(idx, di);
    let end = ((start + 1)..idx.frame_count()).find(|&i| idx.kinds.get(i) == Some(&FrameKind::Intra)).unwrap_or(idx.frame_count());
    (start, end)
}

/// All GOP ranges `[I .. next I)` in decode order (closed GOPs, so each decodes standalone).
fn gop_ranges(idx: &VideoIndex) -> Vec<(usize, usize)> {
    let n = idx.frame_count();
    let starts: Vec<usize> = (0..n).filter(|&i| idx.kinds.get(i) == Some(&FrameKind::Intra)).collect();
    starts.iter().enumerate().map(|(j, &gs)| (gs, starts.get(j + 1).copied().unwrap_or(n))).collect()
}

/// A decode failure with the DECODE index of the frame that failed.
type GopErr = (usize, DecodeError);

/// Decode the whole GOP `[gs, ge)` (all channels) in decode order; returns the planes of each frame (slot `i` = decode
/// index `gs + i`). Every frame's refs are earlier frames of the GOP, hence already in the table.
fn decode_gop(data: &[u8], idx: &VideoIndex, gs: usize, ge: usize) -> Result<Vec<Planes>, GopErr> {
    let mut table: Vec<Option<Planes>> = (gs..ge).map(|_| None).collect();
    for f in gs..ge {
        let planes = decode_one(data, idx, f, gs, Chans::All, by_idx(&table, gs)).map_err(|e| (f, e))?;
        *table.get_mut(f - gs).ok_or((f, DecodeError::Corrupt("frame slot")))? = Some(planes);
    }
    table
        .into_iter()
        .enumerate()
        .map(|(i, p)| p.ok_or((gs + i, DecodeError::Corrupt("decoded frame missing"))))
        .collect()
}

/// The decode work of ONE frame: its ancestors (the frame, its refs, their refs, ...) in ascending decode
/// order, each with the ancestors that may be freed right after it is decoded, and the peak number of
/// frames resident while doing so. A step the planner's `leaf` predicate accepted ([`plan_chain`]) is already
/// decoded (a [`crate::GopCache`] hit): it still appears as a step (so its consumers find it and its pin is
/// released at the right time) but its own references were not planned.
pub(crate) struct Chain {
    /// `(decode index, ancestors freed after it)`, ascending; the last step is the requested frame.
    pub(crate) steps: Vec<(usize, Vec<usize>)>,
    /// Slots of the GOP-relative table a walk needs: `di + 1 - gs`.
    pub(crate) len: usize,
    /// Most frames resident at once (the budget is checked against it).
    pub(crate) peak: usize,
}

/// Plan the dependency closure of decode frame `di` inside the GOP starting at `gs` (headers only, no
/// decode). An I/P chain needs exactly `gs..=di` but only ever 2 resident; a B GOP skips sibling B-frames
/// and later anchors. Refs are validated like [`decode_one`] does (`gs <= r < f`).
///
/// `leaf(f)` is asked once per NEEDED frame, top-down: `true` = `f` is already available (a cache hit the caller
/// pinned), so its references are not needed on its account and the walk starts above it. The stateless
/// [`decode_frame`] passes `|_| false`.
pub(crate) fn plan_chain(data: &[u8], idx: &VideoIndex, gs: usize, di: usize, leaf: &mut dyn FnMut(usize) -> bool) -> Result<Chain, DecodeError> {
    let len = di + 1 - gs;
    let mut need_f = vec![false; len];
    let mut last_use = vec![0usize; len]; // decode index of the LAST (largest) frame referencing slot r
    let mut refs_of: Vec<Vec<usize>> = vec![Vec::new(); len];
    if let Some(slot) = need_f.last_mut() {
        *slot = true;
    }
    for f in (gs..=di).rev() {
        if !need_f.get(f - gs).copied().unwrap_or(false) || leaf(f) {
            continue;
        }
        let ((_, refs, _), _) = frame_parts(data, idx, f)?;
        for r in refs {
            let r = r as usize;
            if !(gs..f).contains(&r) {
                return Err(DecodeError::Corrupt("reference is not an earlier frame of its GOP"));
            }
            need_f[r - gs] = true;
            if last_use[r - gs] == 0 {
                last_use[r - gs] = f; // first seen while walking down = the largest referencing frame
            }
            refs_of[f - gs].push(r);
        }
    }
    let (mut steps, mut live, mut peak) = (Vec::new(), 0usize, 0usize);
    for f in gs..=di {
        if !need_f[f - gs] {
            continue;
        }
        live += 1; // `f` itself, alongside everything still alive
        peak = peak.max(live);
        let mut free: Vec<usize> = refs_of[f - gs].iter().copied().filter(|&r| last_use[r - gs] == f).collect();
        free.sort_unstable();
        free.dedup();
        live -= free.len();
        steps.push((f, free));
    }
    Ok(Chain { steps, len, peak })
}

/// The frames a [`walk`] holds so far, looked up by decode index: the `get` of [`decode_one`] for one step.
pub(crate) struct Held<'a, H> {
    table: &'a [Option<H>],
    base: usize,
    planes_of: fn(&H) -> &Planes,
}

impl<H> Held<'_, H> {
    /// The planes of decode index `r` (`None` = not held: never planned, or already freed).
    pub(crate) fn get(&self, r: u32) -> Option<&Planes> {
        self.table.get((r as usize).checked_sub(self.base)?)?.as_ref().map(self.planes_of)
    }
}

/// Run a planned [`Chain`] bottom-up: `make(step, held)` produces the holder `H` of one step from the frames held so
/// far; a holder is dropped right after its last user. THE ancestor walk of both the stateless [`decode_frame`]
/// (`H` = [`Planes`]) and the cached [`crate::GopCache::get`] (`H` = a pool lease), so a P chain peaks at 2 resident
/// frames either way. Returns the last step (the requested frame).
pub(crate) fn walk<H, E: From<DecodeError>>(
    chain: &Chain,
    gs: usize,
    planes_of: fn(&H) -> &Planes,
    mut make: impl FnMut(usize, &Held<'_, H>) -> Result<H, E>,
) -> Result<H, E> {
    let mut table: Vec<Option<H>> = (0..chain.len).map(|_| None).collect();
    for (f, free) in &chain.steps {
        let h = make(*f, &Held { table: &table, base: gs, planes_of })?;
        *need(table.get_mut(f - gs), "frame slot")? = Some(h);
        for r in free {
            *need(table.get_mut(r - gs), "frame slot")? = None;
        }
    }
    Ok(need(table.last_mut().and_then(Option::take), "decoded frame missing")?)
}

/// The planes of an owned frame (the `planes_of` of [`walk`] for [`decode_frame`]).
fn own_planes(p: &Planes) -> &Planes {
    p
}

/// Decode the `chans` channels of the frame at DISPLAY position `k` via random access. Only the frame's DEPENDENCY
/// ANCESTORS inside its (independent) GOP are decoded ([`plan_chain`], each restricted to `chans` as well: a channel's
/// reference is the same channel) and a reference goes back to the clip's pool as soon as its last user is decoded,
/// so a P-chain needs `<= 2` resident frames (and allocates no more) and a B GOP never decodes siblings or later
/// anchors. The budget ([`VideoIndex::with_limits`]) is checked against that real peak.
///
/// STATELESS: every call re-decodes the whole chain from the GOP's I-frame (`O(gop)` frames per call, `O(gop^2)` for
/// a sequential loop). A host that fetches frames one by one from a P/B clip uses [`crate::GopCache`], the same walk
/// over a cache.
///
/// # Errors
/// [`DecodeError::Request`] for an out-of-range `k` or channel, [`DecodeError::Limit`] over budget, otherwise the
/// failing frame's error.
pub fn decode_frame(data: &[u8], idx: &VideoIndex, k: usize, chans: Chans<'_>) -> Result<Frame, DecodeError> {
    let di = idx.decode_idx(k).ok_or(DecodeError::Request("frame index out of range"))?;
    let gs = gop_start(idx, di);
    let chain = plan_chain(data, idx, gs, di, &mut |_| false)?;
    let _sp = info_span!("decode_frame", frame = k, di, gs, decoded = chain.steps.len()).entered();
    idx.check_frames(chain.peak)?;
    let planes = walk(&chain, gs, own_planes, |f, held| decode_one(data, idx, f, gs, chans, |r| held.get(r)))?;
    Ok(Frame::new(&idx.header, planes))
}

/// DISPLAY indices of the frames that display frame `k` references DIRECTLY: none for an I-frame, one for a P-frame,
/// two anchors for a B-frame. The one-step dependency of [`decode_with_refs`]; the transitive closure is the chain
/// [`decode_frame`] walks itself.
///
/// # Errors
/// [`DecodeError::Request`] for an out-of-range `k`, [`DecodeError::Corrupt`] for a frame header that is unreadable or
/// names a reference that is not an earlier frame of its GOP.
pub fn frame_refs(data: &[u8], idx: &VideoIndex, k: usize) -> Result<Vec<usize>, DecodeError> {
    let di = idx.decode_idx(k).ok_or(DecodeError::Request("frame index out of range"))?;
    let gs = gop_start(idx, di);
    let ((_, refs, _), _) = frame_parts(data, idx, di)?;
    refs.into_iter()
        .map(|r| {
            let r = r as usize;
            if !(gs..di).contains(&r) {
                return Err(DecodeError::Corrupt("reference is not an earlier frame of its GOP"));
            }
            idx.displays.get(r).map(|&d| d as usize).ok_or(DecodeError::Corrupt("display index out of range"))
        })
        .collect()
}

/// Decode the `chans` channels of display frame `k` from the already-decoded planes of its direct references
/// ([`frame_refs`]): `refs(r)` yields the planes of DISPLAY frame `r` (`None` = not available ->
/// `Corrupt("reference frame not decoded")`; they must hold at least `chans`). One frame of work instead of the whole
/// chain [`decode_frame`] pays; the caller owns the references (a frame cache keeps them resident). Bit-identical to
/// `decode_frame(k, chans)` for the same references (the same `decode_one`).
///
/// # Errors
/// [`DecodeError::Request`] for an out-of-range `k`, [`DecodeError::Limit`] when the output plus its references
/// exceed the budget, otherwise the decode error.
pub fn decode_with_refs<'r>(
    data: &[u8],
    idx: &VideoIndex,
    k: usize,
    chans: Chans<'_>,
    refs: impl Fn(usize) -> Option<&'r Planes>,
) -> Result<Frame, DecodeError> {
    let di = idx.decode_idx(k).ok_or(DecodeError::Request("frame index out of range"))?;
    let gs = gop_start(idx, di);
    let ((_, rs, _), _) = frame_parts(data, idx, di)?;
    idx.check_frames(1 + rs.len())?;
    let planes = decode_one(data, idx, di, gs, chans, |r| refs(idx.displays.get(r as usize).map(|&d| d as usize)?))?;
    Ok(Frame::new(&idx.header, planes))
}

/// Decode all frames in DISPLAY order (decode order forward pass + reorder) under the default budget
/// ([`DecodeLimits::default`]); [`read_video_with`] takes an explicit one.
pub fn read_video(data: &[u8]) -> Result<Vec<Image>, DecodeError> {
    read_video_with(data, &DecodeLimits::default())
}

/// As [`read_video`] with an explicit budget: the whole clip is the output, so `frame_count` frames must fit
/// `limits` ([`DecodeError::Limit`] otherwise — use [`read_video_streaming`] for clips larger than RAM).
pub fn read_video_with(data: &[u8], limits: &DecodeLimits) -> Result<Vec<Image>, DecodeError> {
    let idx = VideoIndex::parse(data)?.with_limits(*limits);
    let n = idx.frame_count();
    idx.check_frames(n)?;
    let mut decoded: Vec<Option<Planes>> = (0..n).map(|_| None).collect();
    let mut gs = 0;
    for f in 0..n {
        if idx.kinds.get(f) == Some(&FrameKind::Intra) {
            gs = f; // a new closed GOP starts here
        }
        decoded[f] = Some(decode_one(data, &idx, f, gs, Chans::All, by_idx(&decoded, 0))?);
    }
    // Reorder decode → display.
    let mut ordered: Vec<Option<Image>> = (0..n).map(|_| None).collect();
    for (f, planes) in decoded.into_iter().enumerate() {
        let planes = need(planes, "decoded frame missing")?;
        *need(ordered.get_mut(idx.displays[f] as usize), "display index out of range")? = Some(as_frame(&idx.header, planes)?);
    }
    ordered.into_iter().map(|i| need(i, "display slot empty")).collect()
}

/// Decode a still = the first frame of a `.exv` (a still is a 1-frame clip) under the default budget.
pub fn read_still(data: &[u8]) -> Result<Image, DecodeError> {
    read_still_with(data, &DecodeLimits::default())
}

/// As [`read_still`] with an explicit decode budget.
///
/// An INTRA first frame (every still) decodes straight into the returned `Vec` planes ([`frame::FrameReader::decode_all`]):
/// the pooled path would decode into aligned planes and then copy them out ([`Frame::into_image`]) for a one-shot read.
pub fn read_still_with(data: &[u8], limits: &DecodeLimits) -> Result<Image, DecodeError> {
    let idx = VideoIndex::parse(data)?.with_limits(*limits);
    let di = idx.decode_idx(0).ok_or(DecodeError::Request("frame index out of range"))?;
    let ((kind, _, _), body) = frame_parts(data, &idx, di)?;
    if kind != FrameKind::Intra {
        return decode_frame(data, &idx, 0, Chans::All)?.into_image();
    }
    let _sp = info_span!("decode_frame", frame = 0, di, gs = di, decoded = 1).entered();
    let reader = info_span!("open_index").in_scope(|| frame::FrameReader::open(body, idx.header.clone()))?.with_limits(idx.limits);
    let planes = reader.decode_all(&idx.profile, None)?;
    let mut header = idx.header.clone();
    header.frame_count = 1;
    Image::new(header, planes).map_err(|_| DecodeError::Corrupt("decoded planes do not match the header"))
}

/// Borrow the authoritative `Header` (cloned) + the header-less FRAME BODY of decode-frame 0 of a
/// `.exv` — the raw-blob seam the GPU decode path (`exv-gpu`) opens a [`frame::FrameReader`] over
/// (`FrameReader::open(body, header)`). For a still (1-frame clip) this IS the image. The returned
/// body borrows `data` (zero-copy). `None` on a corrupt / empty stream.
pub fn still_frame_body(data: &[u8]) -> Option<(Header, &[u8])> {
    let idx = VideoIndex::parse(data).ok()?;
    let (_, body) = frame_parts(data, &idx, 0).ok()?;
    Some((idx.header.clone(), body))
}

/// One frame's parts by DECODE index: its prediction kind, refs (decode indices of the same GOP), one MV per
/// ref and the header-less body a [`frame::FrameReader`] opens. THE accessor of a frame's temporal structure:
/// [`TileChains`] and [`temporal_class`] read it, and the GPU resident path resolves tile dependencies from it.
pub struct FrameView<'a> {
    /// DECODE-order index of the frame.
    pub di: usize,
    /// I / P / B.
    pub kind: FrameKind,
    /// References as decode indices (`gop_start <= r < di`): 0 for I, 1 for P, 2 for B.
    pub refs: Vec<u32>,
    /// One motion vector per reference.
    pub mvs: Vec<Mv>,
    /// Frame body (tile blobs + tile index + trailer), zero-copy borrow of `data`.
    pub body: &'a [u8],
}

/// The [`FrameView`] of DECODE-order frame `di` (`None` when out of range or the frame header is corrupt).
pub fn frame_view<'a>(data: &'a [u8], idx: &VideoIndex, di: usize) -> Option<FrameView<'a>> {
    let ((kind, refs, mvs), body) = frame_parts(data, idx, di).ok()?;
    Some(FrameView { di, kind, refs, mvs, body })
}

/// Header + header-less body of DISPLAY frame `k` iff it is an I-frame (decodes with NO reference): the seam of
/// native-INTRA tile sources ([`crate::ExvSource`]). P/B frames are `None`; the temporal GPU path reads them through
/// [`TileChains`] instead.
pub fn frame_body<'a>(data: &'a [u8], idx: &VideoIndex, k: usize) -> Option<(Header, &'a [u8])> {
    let v = frame_view(data, idx, idx.decode_idx(k)?)?;
    (v.kind == FrameKind::Intra).then(|| (idx.header.clone(), v.body))
}

/// A damaged tile of a clip found by [`verify_checksums`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BadBlob {
    /// DISPLAY index of the frame.
    pub frame: usize,
    /// Row-major tile index within the frame.
    pub tile: u32,
    /// [`DecodeError::Checksum`] (which blob, both CRCs) or the structural error that stopped the check.
    pub err: DecodeError,
}

/// The integrity scan of a whole clip (`exv verify`): every coded blob of every frame hashed against its index CRC,
/// whatever the index's [`exv_format::Verify`] policy, WITHOUT decoding a pixel. Returns at most `max` damaged tiles in
/// display order (frames in display order, tiles row-major); an empty `Vec` = every blob is intact. Tiles of a frame hash
/// in parallel; cost = one pass over the compressed bytes.
///
/// # Errors
/// [`DecodeError::Corrupt`] when a frame header or tile index is structurally broken (named by the rule).
pub fn verify_checksums(data: &[u8], idx: &VideoIndex, max: usize) -> Result<Vec<BadBlob>, DecodeError> {
    let mut bad = Vec::new();
    for k in 0..idx.frame_count() {
        if bad.len() >= max {
            break;
        }
        let di = idx.decode_idx(k).ok_or(DecodeError::Corrupt("frame index permutation"))?;
        let (_, body) = frame_parts(data, idx, di)?;
        let reader = frame::FrameReader::open(body, idx.header.clone())?;
        let room = max - bad.len();
        bad.extend(reader.bad_blobs().into_iter().take(room).map(|(tile, err)| BadBlob { frame: k, tile, err }));
    }
    Ok(bad)
}

/// How much temporal machinery a clip needs to decode on a TILE-LOCAL resident GPU path. Monotone: a higher class
/// needs everything the lower ones do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TemporalClass {
    /// Every tile of every frame is INTRA (gop=1, or P frames that coded no tile against a reference): each tile
    /// decodes standalone.
    Intra,
    /// INTRA + SKIP tiles with a zero MV (what a lossy profile emits): a SKIP tile IS its reference tile's bytes,
    /// so a tile resolves to ONE standalone INTRA tile decode of an ancestor frame — no residual arithmetic.
    SkipOnly,
    /// Some tile is PRED (lossless temporal delta): needs a residual add onto the reference tile.
    Pred,
    /// A SKIP/PRED tile sits in a frame with a non-zero MV: the tile is a SHIFT of its reference, which reads
    /// neighbouring tiles (not tile-local).
    Motion,
    /// B-frames: the prediction is the mean of two motion-compensated anchors (decode order != display order).
    Bidir,
}

impl TemporalClass {
    /// Whether the tile-local resident GPU path decodes this class (INTRA and SKIP resolve to standalone tile decodes).
    pub const fn gpu_tiles(self) -> bool {
        matches!(self, Self::Intra | Self::SkipOnly)
    }
}

impl std::fmt::Display for TemporalClass {
    /// The reason text a host logs when it routes a clip to the CPU path.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Intra => "all tiles INTRA",
            Self::SkipOnly => "INTRA + SKIP tiles (zero motion)",
            Self::Pred => "PRED tiles (lossless temporal delta needs a GPU residual add)",
            Self::Motion => "SKIP/PRED tiles with a non-zero motion vector (shifted reference is not tile-local)",
            Self::Bidir => "B-frames (mean of two motion-compensated anchors)",
        })
    }
}

/// Classify the whole clip ([`TemporalClass`]) from the frame headers and tile kind tables (no pixel decode). Run
/// ONCE at open and keep the result: it opens the tile index of every non-INTRA frame, so the scan stops as soon as
/// the class exceeds `upto` (the result is then a lower bound that already exceeds it; a host that only asks "is it
/// GPU tile-local" passes [`TemporalClass::SkipOnly`], pass [`TemporalClass::Bidir`] for the exact class). It looks
/// at the frame MVs too — a SKIP tile under a non-zero MV is a shifted copy, NOT the co-located bytes
/// ([`TemporalClass::Motion`]).
///
/// # Errors
/// [`DecodeError::Corrupt`] / the frame index's own error for an unreadable frame (the caller reports it; never
/// guess a class).
pub fn temporal_class(data: &[u8], idx: &VideoIndex, upto: TemporalClass) -> Result<TemporalClass, DecodeError> {
    if idx.kinds.contains(&FrameKind::Bidir) {
        return Ok(TemporalClass::Bidir);
    }
    let mut class = TemporalClass::Intra;
    for di in 0..idx.frame_count() {
        let ((kind, _, mvs), body) = frame_parts(data, idx, di)?;
        if kind == FrameKind::Intra {
            continue; // an I-frame has no reference: all its tiles are INTRA by construction
        }
        let moved = mvs.iter().any(|&mv| mv != (0, 0));
        let reader = frame::FrameReader::open(body, idx.header.clone())?;
        for t in 0..reader.tile_count() {
            let c = match need(reader.tile_kind(t), "tile kind")? {
                frame::TILE_INTRA => TemporalClass::Intra,
                _ if moved => TemporalClass::Motion,
                frame::TILE_SKIP => TemporalClass::SkipOnly,
                frame::TILE_PRED => TemporalClass::Pred,
                frame::TILE_BI => return Ok(TemporalClass::Bidir), // undecodable before B support
                _ => return Err(DecodeError::Corrupt("unknown tile kind")),
            };
            class = class.max(c);
            if class > upto {
                return Ok(class);
            }
        }
    }
    Ok(class)
}

/// Why a tile's temporal dependency chain cannot be resolved tile-locally ([`TileChains::chain`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainErr {
    /// The frame (or a tile of it) is bidirectional.
    Bidir,
    /// A SKIP/PRED tile under a non-zero MV (a shifted reference).
    Motion,
    /// Malformed structure (frame/tile out of range, a ref outside its GOP, a non-INTRA tile in an I-frame).
    Corrupt(&'static str),
}

impl std::fmt::Display for ChainErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bidir => f.write_str("bidirectional frame (not tile-local)"),
            Self::Motion => f.write_str("SKIP/PRED tile under a non-zero motion vector (not tile-local)"),
            Self::Corrupt(what) => write!(f, "corrupt temporal structure: {what}"),
        }
    }
}

impl std::error::Error for ChainErr {}

/// Where a tile's pixels come from: tile `t` of frame `di` decodes (bit-exactly) from the INTRA tile `t` of frame
/// `base`, after `preds` residual adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileChain {
    /// Decode index of the frame holding the INTRA tile this tile resolves to (the frame itself when it is INTRA).
    pub base: usize,
    /// Decode indices of the frames whose PRED residual tile is added on top, OLDEST first. Empty for INTRA/SKIP-only
    /// chains: a SKIP tile costs no decode work at all.
    pub preds: Vec<usize>,
}

/// One frame opened for tile walking: kind, MVs, the reference and its tile reader.
struct ChainNode<'a> {
    kind: FrameKind,
    /// `(ref, mv)` of a P-frame (`None` for an I-frame).
    pred: Option<(u32, Mv)>,
    reader: frame::FrameReader<'a>,
}

/// Resolves tile dependency chains of a clip. With a zero MV a tile of a P-frame depends only on the CO-LOCATED tile
/// of its reference (SKIP copies it, PRED adds a residual onto it, INTRA ignores it), so temporal decode is a per-tile
/// pointer chase — compatible with visible-tile culling. Frames are opened (tile index parsed) lazily and cached for
/// the walker's lifetime: build one per batch of tiles, not one per tile.
pub struct TileChains<'a> {
    data: &'a [u8],
    idx: &'a VideoIndex,
    nodes: std::collections::HashMap<usize, ChainNode<'a>>,
}

impl<'a> TileChains<'a> {
    /// A walker over `data` (`idx` = `VideoIndex::parse(data)`).
    pub fn new(data: &'a [u8], idx: &'a VideoIndex) -> Self {
        Self { data, idx, nodes: std::collections::HashMap::new() }
    }

    /// The opened frame `di` (parsed once, cached).
    fn node(&mut self, di: usize) -> Result<&ChainNode<'a>, ChainErr> {
        if !self.nodes.contains_key(&di) {
            let v = frame_view(self.data, self.idx, di).ok_or(ChainErr::Corrupt("frame out of range"))?;
            // The readers obey the caller's budget + CRC policy carried by the index (the GPU atlas decodes through them).
            let reader = frame::FrameReader::open(v.body, self.idx.header.clone())
                .map_err(|_| ChainErr::Corrupt("frame tile index"))?
                .with_limits(self.idx.limits);
            let pred = v.refs.first().copied().zip(v.mvs.first().copied());
            let node = ChainNode { kind: v.kind, pred, reader };
            self.nodes.insert(di, node);
        }
        self.nodes.get(&di).ok_or(ChainErr::Corrupt("frame out of range"))
    }

    /// The tile reader of decode frame `di` (the frame a [`TileChain::base`] names).
    ///
    /// # Errors
    /// [`ChainErr::Corrupt`] for an unreadable frame.
    pub fn reader(&mut self, di: usize) -> Result<&frame::FrameReader<'a>, ChainErr> {
        Ok(&self.node(di)?.reader)
    }

    /// An already-opened frame's reader (by shared reference, so several can be held at once, e.g. by parallel CPU decode
    /// tasks); `None` until [`Self::reader`] / [`Self::chain`] opened `di`.
    pub fn opened(&self, di: usize) -> Option<&frame::FrameReader<'a>> {
        self.nodes.get(&di).map(|n| &n.reader)
    }

    /// Walk tile `t` of decode frame `di` back through its references to the INTRA tile it resolves to.
    ///
    /// # Errors
    /// [`ChainErr::Motion`] / [`ChainErr::Bidir`] when the chain is not tile-local, [`ChainErr::Corrupt`] on a
    /// malformed structure (every ref must satisfy `gop_start <= r < frame`, the closed-GOP invariant).
    pub fn chain(&mut self, di: usize, t: u32) -> Result<TileChain, ChainErr> {
        let (mut cur, mut preds) = (di, Vec::new());
        loop {
            let (kind, pred, tk) = {
                let n = self.node(cur)?;
                (n.kind, n.pred, n.reader.tile_kind(t).ok_or(ChainErr::Corrupt("tile out of range"))?)
            };
            if kind == FrameKind::Bidir {
                return Err(ChainErr::Bidir);
            }
            match tk {
                frame::TILE_INTRA => break,
                _ if kind == FrameKind::Intra => return Err(ChainErr::Corrupt("non-INTRA tile in an I-frame")),
                frame::TILE_SKIP | frame::TILE_PRED => {
                    let (r, mv) = pred.ok_or(ChainErr::Corrupt("P-frame without a reference"))?;
                    if mv != (0, 0) {
                        return Err(ChainErr::Motion);
                    }
                    if tk == frame::TILE_PRED {
                        preds.push(cur);
                    }
                    let r = r as usize;
                    if !(gop_start(self.idx, cur)..cur).contains(&r) {
                        return Err(ChainErr::Corrupt("reference is not an earlier frame of its GOP"));
                    }
                    cur = r;
                }
                frame::TILE_BI => return Err(ChainErr::Bidir),
                _ => return Err(ChainErr::Corrupt("unknown tile kind")),
            }
        }
        preds.reverse();
        Ok(TileChain { base: cur, preds })
    }
}

/// Stateful playback decoder over a `.exv`. Parses the seek index ONCE and caches the
/// current GOP's decoded frames — so `frame(k)` inside the cached
/// GOP and `next()` are O(1), versus the stateless [`decode_frame`] which re-decodes the whole GOP every
/// call. A cached GOP holds ALL its frames (a B references a FUTURE anchor, so a partial decode is wrong);
/// GOPs are closed (no ref crosses an I) so exactly one GOP is resident at a time and a cross-GOP scrub
/// pays one GOP decode. Stateless `decode_frame` stays for one-shots.
pub struct Decoder<'a> {
    data: &'a [u8],
    idx: VideoIndex,
    /// The resident GOP as a decode-order range `[start,end)`, or `None` when nothing valid is resident
    /// (before the first decode, or after a failed GOP switch).
    cached: Option<(usize, usize)>,
    /// Decoded planes of the resident GOP: slot `i` = decode index `cached.0 + i`. Pooled in the index's [`PlanePool`]:
    /// evicting a GOP shelves its planes and the next GOP decodes into them (no allocation, zeroing or page faults).
    frames: Vec<Planes>,
    /// Playback cursor in DISPLAY order for [`Self::next`].
    cursor: usize,
    /// Decoder-level span (stored, not an entered guard: the decoder may move between threads).
    span: tracing::Span,
}

impl<'a> Decoder<'a> {
    /// Open a `.exv` for stateful playback under the default decode budget ([`Self::with_limits`] changes it).
    pub fn open(data: &'a [u8]) -> Result<Decoder<'a>, DecodeError> {
        let idx = VideoIndex::parse(data)?;
        let span = info_span!("decoder", w = idx.header.width, h = idx.header.height, n = idx.frame_count());
        Ok(Decoder { data, idx, cached: None, frames: Vec::new(), cursor: 0, span })
    }

    /// Replace the decode budget (a resident GOP must fit it, see [`VideoIndex::with_limits`]).
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.idx = self.idx.with_limits(limits);
        self
    }

    /// Number of frames (display order).
    pub fn frame_count(&self) -> usize {
        self.idx.frame_count()
    }
    /// The clip header.
    pub fn header(&self) -> &Header {
        &self.idx.header
    }
    /// The clip's self-describing metadata (fps / color / HDR-transfer / timecode), or `None`.
    pub fn meta(&self) -> Option<&Meta> {
        self.idx.meta()
    }

    /// Ensure the GOP containing DECODE index `di` is resident (evict the old, decode the new whole GOP
    /// in decode order so each frame's refs are already present).
    ///
    /// The old GOP is evicted first (one GOP resident at a time — the memory bound), so `cached` is reset
    /// BEFORE decoding: on failure nothing stale claims to be resident and the next request for the old
    /// GOP simply re-decodes it, instead of finding an empty cache marked valid.
    fn ensure_gop(&mut self, di: usize) -> Result<(), DecodeError> {
        let (start, end) = gop_range(&self.idx, di);
        if self.cached == Some((start, end)) {
            tracing::debug!(gs = start, ge = end, "gop cache hit");
            return Ok(());
        }
        self.cached = None;
        self.frames.clear();
        self.idx.check_frames(end - start)?;
        let _sp = info_span!(parent: &self.span, "decode_gop", gs = start, ge = end).entered();
        self.frames = decode_gop(self.data, &self.idx, start, end).map_err(|(_, e)| e)?;
        self.cached = Some((start, end));
        Ok(())
    }

    /// Decode DISPLAY frame `k` (random access; caches its whole GOP). Advances the `next()` cursor past
    /// `k`. Errors as [`decode_frame`].
    pub fn frame(&mut self, k: usize) -> Result<Image, DecodeError> {
        let di = self.idx.decode_idx(k).ok_or(DecodeError::Request("frame index out of range"))?;
        self.ensure_gop(di)?;
        self.cursor = k + 1;
        let (start, _) = need(self.cached, "no resident GOP")?;
        image_copy(&self.idx.header, need(self.frames.get(di - start), "resident frame missing")?)
    }
}

/// The next frame in DISPLAY order — O(1) while it stays in the resident GOP. Ends at the last frame or
/// at the first corrupt one (`None`; use [`Decoder::frame`] to see the error).
impl Iterator for Decoder<'_> {
    type Item = Image;
    fn next(&mut self) -> Option<Image> {
        (self.cursor < self.frame_count()).then(|| self.frame(self.cursor).ok()).flatten()
    }
}

/// `true` when decode frames `[gs, ge)` form a pure I→P→P… chain where every P references exactly the
/// frame before it, so each frame needs only its predecessor in RAM (the 2-frame memory path).
fn is_chain(data: &[u8], idx: &VideoIndex, gs: usize, ge: usize) -> bool {
    (gs..ge).all(|f| match frame_parts(data, idx, f) {
        Ok(((FrameKind::Intra, ..), _)) => true,
        Ok(((FrameKind::Pred, refs, _), _)) => refs.first().is_some_and(|&r| r as usize + 1 == f),
        _ => false,
    })
}

/// How many GOPs may decode at once under `budget_bytes`: worst case a GOP task holds its whole GOP, so
/// bound by the REAL longest GOP, not the file's `gop` hint (which a foreign/forged writer can set to anything).
fn max_par(idx: &VideoIndex, ranges: &[(usize, usize)], budget_bytes: u64) -> usize {
    let gop_len = ranges.iter().map(|&(s, e)| (e - s) as u64).max().unwrap_or(1);
    exv_tile::par_for(budget_bytes, idx.frame_bytes().saturating_mul(gop_len), rayon::current_num_threads())
}

/// Streaming decode: GOPs (I-anchored, independent) decode in PARALLEL; each frame is handed to `sink`
/// by its DISPLAY index, OWNED (a consumer that writes it out can move its planes on without a copy; the
/// price is one frame copy per frame in a pure-chain GOP, kept only while the next P still needs it). A
/// pure I/P chain GOP holds only 2 frames (prev+cur); any other GOP (B, or a P
/// referencing a non-adjacent frame) holds the whole GOP (random ref access) — both through the same
/// [`decode_one`], so the stored reference indices are always honored. Concurrency is bounded by
/// `budget_bytes`; every GOP's resident set must also fit the index's [`DecodeLimits`]
/// ([`VideoIndex::with_limits`]) or the call fails with that limit error before decoding it.
pub fn read_video_streaming(
    data: &[u8],
    idx: &VideoIndex,
    budget_bytes: u64,
    sink: &(dyn Fn(usize, Image) -> Result<(), String> + Sync),
) -> Result<(), String> {
    let n = idx.frame_count();
    if n == 0 {
        return Ok(());
    }
    let ranges = gop_ranges(idx);
    if ranges.first().map(|r| r.0) != Some(0) {
        return Err("stream does not start with an I-frame".into());
    }
    let par = max_par(idx, &ranges, budget_bytes);
    let span = info_span!("read_clip", n, gops = ranges.len(), max_par = par, budget_mb = budget_bytes >> 20);
    let _g = span.enter();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(par).build().map_err(|e| format!("thread pool: {e}"))?;
    pool.install(|| {
        ranges
            .par_iter()
            .map(|&(gs, ge)| {
                let chain = is_chain(data, idx, gs, ge);
                let _sp = info_span!(parent: &span, "decode_gop", gs, ge, chain).entered();
                // Resident frames of this GOP task: prev + cur for a chain, the whole GOP otherwise.
                idx.check_frames(if chain { (ge - gs).min(2) } else { ge - gs }).map_err(|e| e.to_string())?;
                if chain {
                    let mut prev: Option<Planes> = None;
                    for f in gs..ge {
                        let get = |r: u32| if r as usize + 1 == f { prev.as_ref() } else { None };
                        let planes = decode_one(data, idx, f, gs, Chans::All, get).map_err(|e| format!("frame {f}: {e}"))?;
                        // The sink takes ownership: hand it a copy only when the next frame still references these
                        // planes (they stay pooled as `prev` and go back to the pool when the frame after replaces them).
                        let img = if f + 1 < ge {
                            let img = image_copy(&idx.header, &planes);
                            prev = Some(planes);
                            img
                        } else {
                            prev = None;
                            as_frame(&idx.header, planes)
                        };
                        sink(idx.displays[f] as usize, img.map_err(|e| format!("frame {f}: {e}"))?)?;
                    }
                } else {
                    let frames = decode_gop(data, idx, gs, ge).map_err(|(f, e)| format!("frame {f}: {e}"))?;
                    for (f, planes) in (gs..ge).zip(frames) {
                        let img = as_frame(&idx.header, planes).map_err(|e| format!("frame {f}: {e}"))?;
                        sink(idx.displays[f] as usize, img)?;
                    }
                }
                Ok(())
            })
            .collect::<Result<Vec<()>, String>>()
    })?;
    Ok(())
}

/// Iterator over a clip as WINDOWS of whole source GOPs in DISPLAY order — the read side of a
/// `.exv -> .exv` transcode ([`crate::transcode`]): every frame is decoded exactly ONCE (a per-frame
/// [`decode_frame`] loop re-decodes the GOP prefix for each frame, `O(gop^2)`). Created by [`read_windows`].
pub struct Windows<'a> {
    data: &'a [u8],
    idx: &'a VideoIndex,
    ranges: Vec<(usize, usize)>,
    /// GOPs per window (the same budget rule as [`read_video_streaming`]).
    par: usize,
    /// Next GOP range to decode.
    next: usize,
    /// DISPLAY index the next window must start at (windows must tile the display axis).
    disp: usize,
    span: tracing::Span,
}

/// Read a clip as windows of whole GOPs: up to `budget_bytes / (frame_bytes * longest GOP)` GOPs decode in
/// parallel per window, and the window's frames come back in DISPLAY order. A window's display range must be
/// contiguous and follow the previous one ([`VideoError::Shape`] otherwise: a foreign file whose GOPs
/// interleave in display order cannot be streamed; read it with [`read_video_with`]).
pub fn read_windows<'a>(data: &'a [u8], idx: &'a VideoIndex, budget_bytes: u64) -> Windows<'a> {
    let ranges = gop_ranges(idx);
    let par = max_par(idx, &ranges, budget_bytes);
    let span = info_span!("read_clip", n = idx.frame_count(), gops = ranges.len(), max_par = par, budget_mb = budget_bytes >> 20);
    Windows { data, idx, ranges, par, next: 0, disp: 0, span }
}

impl Windows<'_> {
    /// Decode the GOPs `gops` in parallel and return their frames sorted by DISPLAY index.
    fn decode(&mut self, gops: &[(usize, usize)]) -> Result<Vec<Image>, VideoError> {
        let (data, idx, span) = (self.data, self.idx, &self.span);
        let dec = |frame: usize| move |err| VideoError::Decode { frame, err };
        idx.check_frames(gops.iter().map(|&(s, e)| e - s).sum()).map_err(dec(self.disp))?;
        let per_gop = gops
            .par_iter()
            .map(|&(gs, ge)| {
                let _sp = info_span!(parent: span, "decode_gop", gs, ge, chain = false).entered();
                let planes = decode_gop(data, idx, gs, ge).map_err(|(f, e)| dec(idx.displays[f] as usize)(e))?;
                (gs..ge)
                    .zip(planes)
                    .map(|(f, p)| {
                        let d = idx.displays[f] as usize;
                        Ok((d, as_frame(&idx.header, p).map_err(dec(d))?))
                    })
                    .collect::<Result<Vec<_>, VideoError>>()
            })
            .collect::<Result<Vec<_>, VideoError>>()?;
        let mut frames: Vec<(usize, Image)> = per_gop.into_iter().flatten().collect();
        frames.sort_by_key(|f| f.0);
        if frames.iter().enumerate().any(|(i, f)| f.0 != self.disp + i) {
            return Err(VideoError::Shape("GOP display ranges are not contiguous"));
        }
        self.disp += frames.len();
        Ok(frames.into_iter().map(|f| f.1).collect())
    }
}

impl Iterator for Windows<'_> {
    type Item = Result<Vec<Image>, VideoError>;

    /// The next window; after an error the iterator is exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        let first = self.next;
        if first >= self.ranges.len() {
            return None;
        }
        let end = (first + self.par).min(self.ranges.len());
        self.next = end;
        let gops = self.ranges.get(first..end)?.to_vec();
        let r = self.span.clone().in_scope(|| self.decode(&gops));
        if r.is_err() {
            self.next = self.ranges.len();
        }
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{WriteOpts, write_clip_pull, write_still, write_still_windowed, write_video, write_video_meta};
    use exv_format::{Channel, Dtype, Header};
    use std::io;
    use std::io::Cursor;
    use std::sync::Mutex;

    /// A 2-channel (F16,F32) frame with `shift`-dependent content (neighbours differ). `tile` = grid.
    fn frame(w: u32, h: u32, tile: u32, shift: u32) -> Image {
        let n = (w * h) as usize;
        let r: Vec<u8> = (0..n).flat_map(|i| (((i as u32 % w + shift) & 0xff) as u16).to_le_bytes()).collect();
        let z: Vec<u8> = (0..n).flat_map(|i| ((i as u32 + shift) as f32).to_le_bytes()).collect();
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: tile,
            tile_h: tile,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "Z".into(), dtype: Dtype::F32 },
            ],
        };
        Image::new(header, vec![r, z]).unwrap()
    }

    /// A single-channel U16 frame filled with a constant value `v` — for exact-mean B tests.
    fn u16_const(w: u32, h: u32, tile: u32, v: u16) -> Image {
        let n = (w * h) as usize;
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: tile,
            tile_h: tile,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: vec![Channel { name: "V".into(), dtype: Dtype::U16 }],
        };
        Image::new(header, vec![(0..n).flat_map(|_| v.to_le_bytes()).collect()]).unwrap()
    }

    /// Lossless round-trip, differing frames, across tile grids × gop × MV search × B-frames.
    #[test]
    fn roundtrip() {
        for tile in [64u32, 96] {
            for gop in [1u32, 4] {
                for search in [0i32, 4] {
                    for b in [0u32, 1] {
                        let frames: Vec<Image> = (0..7).map(|s| frame(96, 72, tile, s)).collect();
                        let enc = write_video(&frames, &Profile::lossless_fast(), gop, search, b).expect("encode");
                        let dec = read_video(&enc).expect("decode");
                        assert_eq!(dec.len(), frames.len(), "tile={tile} gop={gop} search={search} b={b}");
                        for (a, c) in dec.iter().zip(&frames) {
                            assert_eq!(a.planes, c.planes, "tile={tile} gop={gop} search={search} b={b}");
                        }
                    }
                }
            }
        }
    }

    /// B-frame benefit: an integer clip that is EXACTLY the mean of its neighbours (a linear fade) →
    /// the middle B is SKIP (cur == mean) → the B encode is smaller than the same clip as all-P, and
    /// display order is preserved bit-exact.
    #[test]
    fn bframe_mean_skip_and_exact() {
        // display: [100, 150, 200] — 150 == (100+200)/2, an exact mean.
        let frames = vec![
            u16_const(96, 64, 64, 100),
            u16_const(96, 64, 64, 150),
            u16_const(96, 64, 64, 200),
        ];
        let prof = Profile::lossless_fast();
        let with_b = write_video(&frames, &prof, 8, 0, 1).expect("b"); // gop 8 (I only at 0), 1 B between anchors
        let no_b = write_video(&frames, &prof, 8, 0, 0).expect("nob");
        assert!(with_b.len() < no_b.len(), "B {} should beat P-only {} on an exact mean", with_b.len(), no_b.len());
        let dec = read_video(&with_b).expect("decode");
        assert_eq!(dec.len(), 3);
        for (a, c) in dec.iter().zip(&frames) {
            assert_eq!(a.planes, c.planes, "B clip round-trips in display order");
        }
    }

    /// Global MV: a PANNING clip encodes smaller with MV search and round-trips bit-exact.
    #[test]
    fn global_mv_pan_smaller_and_exact() {
        let base = frame(128, 96, 64, 5);
        let (w, h) = (base.header.width, base.header.height);
        let frames: Vec<Image> = (0..4)
            .map(|i| {
                let planes: Vec<Vec<u8>> = base
                    .planes
                    .iter()
                    .zip(base.header.channels.iter())
                    .map(|(p, ch)| motion::shift_plane(p, w, h, ch.dtype, i * 3, i * 2).expect("shift"))
                    .collect();
                Image::new(base.header.clone(), planes).unwrap()
            })
            .collect();
        let prof = Profile::lossless_fast();
        let with_mv = write_video(&frames, &prof, 4, 8, 0).expect("mv");
        let no_mv = write_video(&frames, &prof, 4, 0, 0).expect("nomv");
        assert!(with_mv.len() < no_mv.len(), "MV {} should beat no-MV {} on a pan", with_mv.len(), no_mv.len());
        let dec = read_video(&with_mv).expect("decode");
        for (a, c) in dec.iter().zip(&frames) {
            assert_eq!(a.planes, c.planes, "panning round-trip exact");
        }
    }

    /// `frame_body(k)` of an I-frame decodes STANDALONE (no reference) and equals `decode_frame(k)`; a P-frame is `None`
    /// (its tiles need a reference — the temporal GPU path reads it through `TileChains`).
    #[test]
    fn frame_body_intra_standalone() {
        let frames = vec![frame(80, 48, 64, 1), frame(80, 48, 64, 2), frame(80, 48, 64, 3)];
        let prof = Profile::lossless_fast();
        // gop=1 → EVERY frame is INTRA → each `frame_body` decodes standalone.
        let enc = write_video(&frames, &prof, 1, 0, 0).expect("encode all-intra");
        let idx = VideoIndex::parse(&enc).expect("index");
        assert_eq!(idx.frame_count(), 3);
        assert_eq!(temporal_class(&enc, &idx, TemporalClass::Bidir), Ok(TemporalClass::Intra), "gop=1 must be all-INTRA");
        for k in 0..idx.frame_count() {
            let (hdr, body) = frame_body(&enc, &idx, k).expect("frame_body");
            let planes = exv_codec::frame::read_frame_tiles(body, &hdr, &idx.profile, None, &idx.limits)
                .expect("decode body standalone");
            let full = decode_frame(&enc, &idx, k, Chans::All).and_then(Frame::into_image).expect("decode_frame");
            assert_eq!(planes, full.planes, "frame_body({k}) standalone == decode_frame({k})");
        }
        // A P-frame clip (gop large, differing frames): frames 1.. are P → `frame_body` returns None.
        let enc_p = write_video(&frames, &prof, 8, 0, 0).expect("encode with P");
        let idx_p = VideoIndex::parse(&enc_p).expect("index P");
        assert!(frame_body(&enc_p, &idx_p, 0).is_some(), "frame 0 is I");
        assert!(frame_body(&enc_p, &idx_p, 1).is_none(), "frame 1 is P → not standalone");
        let v = frame_view(&enc_p, &idx_p, 1).expect("view");
        assert_eq!((v.di, v.kind, v.refs.as_slice(), v.mvs.as_slice()), (1, FrameKind::Pred, &[0u32][..], &[(0i16, 0i16)][..]));
    }

    /// `img` with ONE byte of tile `(tx, ty)`'s F32 channel changed (so exactly that tile differs; sign/exponent
    /// intact, so a lossy float codec still sees a normal number).
    fn poke(img: &Image, tx: u32, ty: u32, salt: u8) -> Image {
        let h = &img.header;
        let (x, y, ..) = h.tile_rect(tx, ty);
        let mut planes = img.planes.clone();
        planes[1][((y * h.width + x) * 4) as usize] ^= salt | 1;
        Image::new(h.clone(), planes).expect("poked image")
    }

    /// Bytes of tile `t` of every channel of full-size `planes`.
    fn tile_of(h: &Header, planes: &[Vec<u8>], t: u32) -> Vec<Vec<u8>> {
        let (x, y, tw, th) = h.tile_rect(t % h.tiles_x(), t / h.tiles_x());
        h.channels
            .iter()
            .zip(planes)
            .map(|(c, p)| {
                let esz = c.dtype.size();
                (y..y + th).flat_map(|r| p[((r * h.width + x) as usize) * esz..((r * h.width + x + tw) as usize) * esz].to_vec()).collect()
            })
            .collect()
    }

    /// A 2x2-tile clip where frame 1 changes tile (1,0) only and frame 2 changes tile (0,1) only: the other tiles are
    /// SKIP, chains cross frames.
    fn partial_skip_clip() -> Vec<Image> {
        let f0 = frame(128, 96, 64, 3);
        let f1 = poke(&f0, 1, 0, 0x10);
        let f2 = poke(&f1, 0, 1, 0x20);
        vec![f0, f1.clone(), f2.clone(), f2]
    }

    /// `temporal_class` truth table: INTRA / SKIP-only (identical + partially identical frames, lossy too) / PRED /
    /// MOTION (a pan: SKIP tiles under a non-zero MV are SHIFTED copies, the old `temporal_gpu_ok` said "ok") / B.
    #[test]
    fn temporal_class_truth_table() {
        let class = |frames: &[Image], prof: &Profile, gop: u32, search: i32, b: u32| {
            let enc = write_video(frames, prof, gop, search, b).expect("encode");
            temporal_class(&enc, &VideoIndex::parse(&enc).expect("index"), TemporalClass::Bidir).expect("class")
        };
        let lossless = Profile::lossless_fast();
        let same = vec![frame(80, 48, 64, 3), frame(80, 48, 64, 3)];
        assert_eq!(class(&same, &lossless, 8, 0, 0), TemporalClass::SkipOnly, "identical frames: INTRA + SKIP");
        assert_eq!(class(&same, &lossless, 1, 0, 0), TemporalClass::Intra, "gop=1");
        let part = partial_skip_clip();
        assert_eq!(class(&part, &Profile::lossy_wave(4), 8, 0, 0), TemporalClass::SkipOnly, "lossy: changed tiles stay INTRA");
        assert_eq!(class(&part, &lossless, 8, 0, 0), TemporalClass::Pred, "lossless: a changed tile is a cheap PRED residual");
        // A pan found by the MV search: every tile is SKIP/PRED against the SHIFTED reference.
        let base = frame(128, 96, 64, 5);
        let (w, h) = (base.header.width, base.header.height);
        let pan: Vec<Image> = (0..4)
            .map(|i| {
                let planes = base
                    .planes
                    .iter()
                    .zip(&base.header.channels)
                    .map(|(p, c)| motion::shift_plane(p, w, h, c.dtype, i * 3, i * 2).expect("shift"))
                    .collect();
                Image::new(base.header.clone(), planes).expect("pan frame")
            })
            .collect();
        assert_eq!(class(&pan, &lossless, 4, 8, 0), TemporalClass::Motion);
        assert_eq!(class(&pan, &Profile::lossy_wave(4), 4, 8, 0), TemporalClass::Intra, "lossy codes every tile INTRA: the MV is moot");
        let seq: Vec<Image> = (0..5).map(|s| frame(64, 48, 64, s)).collect();
        assert_eq!(class(&seq, &lossless, 8, 0, 1), TemporalClass::Bidir);
        // `upto` stops the scan early: a Pred clip asked "is it SkipOnly-or-lower" answers with a class that already exceeds it.
        let enc = write_video(&part, &lossless, 8, 0, 0).expect("encode");
        let early = temporal_class(&enc, &VideoIndex::parse(&enc).expect("index"), TemporalClass::SkipOnly).expect("class");
        assert!(early > TemporalClass::SkipOnly);
        assert!(TemporalClass::Intra < TemporalClass::SkipOnly && TemporalClass::Pred < TemporalClass::Motion);
        assert!(TemporalClass::SkipOnly.gpu_tiles() && !TemporalClass::Pred.gpu_tiles());
    }

    /// `TileChains::chain`: for every frame and tile the tile's bytes equal the INTRA tile of `base` (SKIP-only
    /// chains) or `base` + the listed PRED residual frames; the oracle is `decode_frame`.
    #[test]
    fn tile_chain_resolves_to_the_intra_tile() {
        let clip = partial_skip_clip();
        for prof in [Profile::lossy_wave(4), Profile::lossless_fast()] {
            let enc = write_video(&clip, &prof, 8, 0, 0).expect("encode");
            let idx = VideoIndex::parse(&enc).expect("index");
            let class = temporal_class(&enc, &idx, TemporalClass::Bidir).expect("class");
            let mut walk = TileChains::new(&enc, &idx);
            let n_tiles = idx.header.tile_count() as u32;
            for di in 0..idx.frame_count() {
                let img = decode_frame(&enc, &idx, di, Chans::All).and_then(Frame::into_image).expect("decode");
                for t in 0..n_tiles {
                    let c = walk.chain(di, t).expect("chain");
                    let rd = walk.reader(c.base).expect("base reader");
                    assert_eq!(rd.tile_kind(t), Some(frame::TILE_INTRA), "frame {di} tile {t}: base must be INTRA");
                    assert!(c.base <= di && c.preds.iter().all(|&p| c.base < p && p <= di), "{c:?}");
                    if c.preds.is_empty() {
                        let base = decode_frame(&enc, &idx, c.base, Chans::All).and_then(Frame::into_image).expect("base decode");
                        assert_eq!(tile_of(&idx.header, &img.planes, t), tile_of(&idx.header, &base.planes, t), "frame {di} tile {t}");
                    } else {
                        assert_eq!(class, TemporalClass::Pred, "preds only occur in the lossless PRED class");
                    }
                }
            }
            // The scene of this clip: tile (0,0) never changes -> it resolves to frame 0 from every frame.
            assert!((0..idx.frame_count()).all(|di| walk.chain(di, 0).expect("chain").base == 0));
        }
        // Lossy clip: frame 3 == frame 2 -> tile (0,1) resolves to its INTRA tile in frame 2.
        let enc = write_video(&clip, &Profile::lossy_wave(4), 8, 0, 0).expect("encode");
        let idx = VideoIndex::parse(&enc).expect("index");
        let mut walk = TileChains::new(&enc, &idx);
        assert_eq!(walk.chain(3, 2).expect("chain"), TileChain { base: 2, preds: vec![] });
        assert_eq!(walk.chain(1, 1).expect("chain").base, 1, "changed tile is INTRA in its own frame");
    }

    /// A pan clip's SKIP tiles under a MV are rejected as `Motion` (never a guess), a B clip as `Bidir`.
    #[test]
    fn tile_chain_rejects_motion_and_bidir() {
        let base = frame(128, 96, 64, 5);
        let (w, h) = (base.header.width, base.header.height);
        let pan: Vec<Image> = (0..3)
            .map(|i| {
                let planes = base
                    .planes
                    .iter()
                    .zip(&base.header.channels)
                    .map(|(p, c)| motion::shift_plane(p, w, h, c.dtype, i * 3, i * 2).expect("shift"))
                    .collect();
                Image::new(base.header.clone(), planes).expect("pan frame")
            })
            .collect();
        let enc = write_video(&pan, &Profile::lossless_fast(), 4, 8, 0).expect("encode");
        let idx = VideoIndex::parse(&enc).expect("index");
        assert_eq!(TileChains::new(&enc, &idx).chain(1, 0), Err(ChainErr::Motion));
        let seq: Vec<Image> = (0..5).map(|s| frame(64, 48, 64, s)).collect();
        let enc = write_video(&seq, &Profile::lossless_fast(), 8, 0, 1).expect("encode");
        let idx = VideoIndex::parse(&enc).expect("index");
        let b = (0..idx.frame_count()).find(|&di| idx.kinds[di] == FrameKind::Bidir).expect("a B frame");
        assert_eq!(TileChains::new(&enc, &idx).chain(b, 0), Err(ChainErr::Bidir));
        assert!(matches!(TileChains::new(&enc, &idx).chain(99, 0), Err(ChainErr::Corrupt(_))));
    }

    /// Unification gate: a still IS a 1-frame `.exv`.
    #[test]
    fn still_is_one_frame() {
        let f = frame(80, 48, 64, 3);
        let enc = write_video(std::slice::from_ref(&f), &Profile::lossless_fast(), 1, 0, 0).expect("encode");
        let idx = VideoIndex::parse(&enc).expect("index");
        assert_eq!(idx.frame_count(), 1);
        assert_eq!(decode_frame(&enc, &idx, 0, Chans::All).and_then(Frame::into_image).expect("df").planes, f.planes);
    }

    /// Random-access `decode_frame(k)` == `read_video()[k]` for every display k (fwd/back), with B.
    #[test]
    fn random_access() {
        let frames: Vec<Image> = (0..9).map(|s| frame(80, 48, 64, s)).collect();
        let enc = write_video(&frames, &Profile::lossless_fast(), 4, 4, 1).expect("encode");
        let seq = read_video(&enc).expect("seq");
        let idx = VideoIndex::parse(&enc).expect("index");
        for k in [0usize, 8, 5, 1, 3, 6, 0] {
            assert_eq!(decode_frame(&enc, &idx, k, Chans::All).and_then(Frame::into_image).expect("df").planes, seq[k].planes, "k={k}");
        }
        assert!(matches!(decode_frame(&enc, &idx, frames.len(), Chans::All), Err(DecodeError::Request(_))), "out-of-range → Request");
    }

    /// Truncations never panic (I/P/B streams).
    #[test]
    fn malformed_never_panics() {
        for b in [0u32, 1] {
            let frames: Vec<Image> = (0..5).map(|s| frame(48, 32, 64, s)).collect();
            let good = write_video(&frames, &Profile::lossless_fast(), 4, 4, b).expect("enc");
            for cut in 0..good.len() {
                let _ = read_video(&good[..cut]);
                let _ = VideoIndex::parse(&good[..cut]);
            }
        }
        assert!(VideoIndex::parse(&[0xffu8; 32]).is_err());
    }

    /// Streaming write/read (memory-bounded I/P) round-trips bit-exact, matching the in-RAM path.
    #[test]
    fn streaming_roundtrip() {
        let frames: Vec<Image> = (0..7).map(|s| frame(96, 72, 64, s)).collect();
        let prof = Profile::lossless_fast();
        let get = |fi: usize| frames.get(fi).cloned().ok_or_else(|| format!("no frame {fi}"));
        // budget = 1 byte → window collapses to 1 frame → exercises the carry (window-boundary P).
        let opts = WriteOpts { gop: 3, search: 4, ..WriteOpts::default() };
        let enc = write_clip_pull(Cursor::new(Vec::new()), frames.len(), &prof, &opts, 1, &get, &|_| {})
            .expect("stream enc")
            .0
            .into_inner();
        // B-frames need a whole-clip reorder: the streaming writer refuses them instead of ignoring them.
        let with_b = WriteOpts { b_frames: 1, ..opts.clone() };
        assert!(matches!(
            write_clip_pull(Cursor::new(Vec::new()), frames.len(), &prof, &with_b, 1, &get, &|_| {}),
            Err(crate::VideoError::Unsupported(_))
        ));
        // Decodes via the normal reader…
        for (a, b) in read_video(&enc).expect("read").iter().zip(&frames) {
            assert_eq!(a.planes, b.planes, "streaming enc → in-RAM decode");
        }
        // …and via the streaming reader (frames delivered by display index).
        let idx = VideoIndex::parse(&enc).expect("idx");
        let out: Mutex<Vec<Option<Vec<Vec<u8>>>>> = Mutex::new((0..frames.len()).map(|_| None).collect());
        read_video_streaming(&enc, &idx, 1 << 20, &|di, img| {
            out.lock().unwrap()[di] = Some(img.planes);
            Ok(())
        })
        .expect("stream read");
        for (d, f) in out.into_inner().unwrap().into_iter().zip(&frames) {
            assert_eq!(d.expect("frame present"), f.planes, "streaming read, display order");
        }
    }

    /// Stateful `Decoder`: `frame(k)` random access + `next()` sequence == the stateless `read_video`,
    /// with B-frames (decode≠display). Exercises the GOP cache + display→decode map.
    #[test]
    fn decoder_matches_read_video() {
        let frames: Vec<Image> = (0..9).map(|s| frame(80, 48, 64, s)).collect();
        let enc = write_video(&frames, &Profile::lossless_fast(), 4, 4, 1).expect("encode");
        let seq = read_video(&enc).expect("seq");
        let mut dec = Decoder::open(&enc).expect("decoder");
        assert_eq!(dec.frame_count(), seq.len());
        for k in [0usize, 7, 3, 8, 1, 5, 0] {
            assert_eq!(dec.frame(k).expect("frame").planes, seq[k].planes, "frame({k}) random access");
        }
        let mut dec2 = Decoder::open(&enc).expect("decoder2");
        for (k, f) in seq.iter().enumerate() {
            assert_eq!(dec2.next().expect("next").planes, f.planes, "next() step {k}");
        }
        assert!(dec2.next().is_none(), "next() past end");
    }

    /// B3 wiring: `write_video_meta` records [`Meta`] into the `.exv`; `VideoIndex`/`Decoder` expose it,
    /// frames still decode, and a plain `write_video` carries no meta.
    #[test]
    fn meta_roundtrips_through_container() {
        use exv_format::Meta;
        let frames: Vec<Image> = (0..4).map(|s| frame(64, 48, 64, s)).collect();
        let prof = Profile::lossless_fast();
        let meta = Meta { fps: Some((24, 1)), primaries: 1, transfer: 2, timecode: Some(7), ..Meta::default() };
        let opts = WriteOpts { gop: 2, search: 4, b_frames: 1, meta: meta.clone() };
        let enc = write_video_meta(&frames, &prof, &opts).expect("encode meta");
        let idx = VideoIndex::parse(&enc).expect("index");
        assert_eq!(idx.meta(), Some(&meta));
        let dec = read_video(&enc).expect("decode"); // frames still decode with meta present
        assert_eq!(dec.len(), frames.len());
        for (a, b) in dec.iter().zip(&frames) {
            assert_eq!(a.planes, b.planes);
        }
        assert_eq!(Decoder::open(&enc).expect("dec").meta(), Some(&meta));
        // Plain write_video carries no meta.
        let plain = write_video(&frames, &prof, 2, 4, 1).expect("plain");
        assert_eq!(VideoIndex::parse(&plain).expect("idx").meta(), None);
    }

    /// Lossy-wave GOP video (identical frames → INTRA + SKIP tiles, gop 8) decodes correctly.
    #[test]
    fn lossy_wave_video_decodes() {
        let frames: Vec<Image> = (0..16).map(|_| frame(256, 192, 64, 0)).collect();
        let prof = Profile::lossy_wave(2);
        let enc = write_video(&frames, &prof, 8, 0, 0).expect("encode");
        let dec = read_video(&enc).expect("decode");
        assert_eq!(dec.len(), 16);
    }

    /// REPRO: nested rayon par_iter (decode_tile per-channel) inside read_video_streaming's custom pool
    /// (par_iter over GOPs) must not DEADLOCK. Forces max_par>1 (big budget + >=2 GOPs) while keeping the nesting
    /// (43 channels = more per-channel tasks than a typical pool has threads, several tiles per frame): only the
    /// pixel count and the frame count are cut so it is cheap enough for the normal test run.
    #[test]
    fn streaming_manychan_no_deadlock() {
        use exv_format::{Channel, Dtype, Header};
        let (w, h, tile, nch) = (128u32, 96u32, 64u32, 43usize);
        let n = (w * h) as usize;
        let mk = |s: u32| -> Image {
            let header = Header {
                width: w,
                height: h,
                frame_count: 1,
                tile_w: tile,
                tile_h: tile,
                codec_id: 0,
                codec_params: Vec::new(),
                channels: (0..nch).map(|c| Channel { name: format!("c{c}"), dtype: Dtype::F32 }).collect(),
            };
            let planes: Vec<Vec<u8>> = (0..nch)
                .map(|c| (0..n).flat_map(|i| (c as f32 + s as f32 + (i as f32 * 0.01).sin()).to_le_bytes()).collect())
                .collect();
            Image::new(header, planes).expect("img")
        };
        // 4 GOPs of 2 frames: several concurrent GOP tasks, each one nesting the per-channel par_iter.
        let frames: Vec<Image> = (0..8).map(mk).collect();
        let enc = write_video(&frames, &Profile::lossy_wave(2), 2, 0, 0).expect("enc");
        let idx = VideoIndex::parse(&enc).expect("idx");
        let count = std::sync::atomic::AtomicUsize::new(0);
        // budget 4 GiB → max_par = num_threads → GOP par_iter is multi-threaded → nests my channel par_iter.
        read_video_streaming(&enc, &idx, 4 << 30, &|_fi, _img| {
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })
        .expect("stream");
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 8);
        let ranges = gop_ranges(&idx);
        assert!(ranges.len() >= 2 && max_par(&idx, &ranges, 4 << 30) > 1, "probe must run concurrent GOP tasks (nested rayon)");
    }

    /// TIMING: the VIEWER's exact call `decode_frame(0)` on a 43-channel lossy-wave q12 .exv, in-RAM
    /// (isolates the decode from disk/mmap). Compares to `read_video` and repeats to see warm vs cold.
    /// Run: `cargo test --release -p exv-video timing_decode_frame0 -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn timing_decode_frame0() {
        use exv_format::{Channel, Dtype, Header};
        use std::time::Instant;
        let (w, h, tile, nch) = (1280u32, 720u32, 256u32, 43usize);
        let n = (w * h) as usize;
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: tile,
            tile_h: tile,
            codec_id: 0,
            codec_params: Vec::new(),
            channels: (0..nch).map(|c| Channel { name: format!("ch{c}"), dtype: Dtype::F32 }).collect(),
        };
        // Varied HDR-ish floats (log-distributed) so the entropy/hdr path sees realistic symbols.
        let planes: Vec<Vec<u8>> = (0..nch)
            .map(|c| {
                (0..n)
                    .flat_map(|i| {
                        let v = 0.02f32 * 2f32.powf(((i as f32 * 0.013 + c as f32).sin()) * 5.0);
                        v.to_le_bytes()
                    })
                    .collect()
            })
            .collect();
        let img = Image::new(header, planes).expect("img");
        let enc = write_video(std::slice::from_ref(&img), &Profile::lossy_wave(12), 1, 0, 0).expect("enc");
        let idx = VideoIndex::parse(&enc).expect("idx");
        for pass in 0..3 {
            let t = Instant::now();
            let out = decode_frame(&enc, &idx, 0, Chans::All).and_then(Frame::into_image).expect("decode_frame");
            eprintln!("decode_frame(0) pass {pass}: {:?}  ({} planes)", t.elapsed(), out.planes.len());
        }
    }

    /// TIMING (Robat-like): per-frame encode+decode cost of a 43-channel F32 1280x720 frame.
    /// Run: `cargo test --release -p exv-video timing_robat_like -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe, run explicitly in release"]
    fn timing_robat_like_frame() {
        use exv_format::{Channel, Dtype, Header};
        use std::time::Instant;
        let (w, h, tile, nch) = (1280u32, 720u32, 256u32, 43usize);
        let n = (w * h) as usize;
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: tile,
            tile_h: tile,
            codec_id: 0,
            codec_params: Vec::new(),
            channels: (0..nch).map(|c| Channel { name: format!("ch{c}"), dtype: Dtype::F32 }).collect(),
        };
        let planes: Vec<Vec<u8>> = (0..nch)
            .map(|c| {
                (0..n)
                    .flat_map(|i| (c as f32 * 0.1 + (i as f32 * 0.0007).sin() * 4.0).to_le_bytes())
                    .collect()
            })
            .collect();
        let img = Image::new(header, planes).expect("img");
        for (name, prof) in [("lossy-wave q2", Profile::lossy_wave(2)), ("htj2k lossless", Profile::htj2k())] {
            let t = Instant::now();
            let enc = write_video(std::slice::from_ref(&img), &prof, 1, 0, 0).expect("enc");
            let et = t.elapsed();
            let t = Instant::now();
            let dec = read_video(&enc).expect("dec");
            let dt = t.elapsed();
            eprintln!("[{name}] 43ch F32 1280x720: ENCODE {et:?}  DECODE {dt:?}  ({} bytes)", enc.len());
            assert_eq!(dec.len(), 1);
        }
    }

    /// Byte offset of the embedded `Header` (= prelude length before it): magic4 + ver1 + gop4 + count4.
    const HDR_AT: usize = 13;

    fn set_u32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn index_off_of(enc: &[u8]) -> usize {
        let mut tail = [0u8; 8];
        tail.copy_from_slice(&enc[enc.len() - 8..]);
        u64::from_le_bytes(tail) as usize
    }

    /// A forged frame count (prelude + header + index all `u32::MAX`, ~60-byte-class file) must be a clean
    /// `None`: the index length is proven against `n` BEFORE any `with_capacity(n)` (was a ~52 GB abort).
    #[test]
    fn forged_frame_count_is_none() {
        let mut enc = write_video(&[frame(16, 16, 16, 1)], &Profile::lossless_fast(), 1, 0, 0).expect("enc");
        let index_off = index_off_of(&enc);
        set_u32(&mut enc, 9, u32::MAX); // prelude frame_count
        set_u32(&mut enc, HDR_AT + 10, u32::MAX); // header frame_count
        set_u32(&mut enc, index_off, u32::MAX); // index n
        assert!(VideoIndex::parse(&enc).is_err());
        assert!(read_video(&enc).is_err());
        // Prelude vs header disagreement alone is also rejected.
        let mut enc2 = write_video(&[frame(16, 16, 16, 1)], &Profile::lossless_fast(), 1, 0, 0).expect("enc");
        set_u32(&mut enc2, 9, 2);
        assert!(VideoIndex::parse(&enc2).is_err());
    }

    /// Patch the header's width/height AND tile size (a 1-tile file declaring a huge one-tile image).
    fn forge_one_tile(enc: &mut [u8], side: u32) {
        for at in [HDR_AT + 2, HDR_AT + 6, HDR_AT + 14, HDR_AT + 18] {
            set_u32(enc, at, side);
        }
    }

    /// A single-channel U8 all-zero frame (compresses ~thousands:1: the "legit but tiny file" case).
    fn blank_u8(side: u32) -> Image {
        let header = Header {
            width: side,
            height: side,
            frame_count: 1,
            tile_w: side,
            tile_h: side,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: vec![Channel { name: "M".into(), dtype: Dtype::U8 }],
        };
        Image::new(header, vec![vec![0u8; side as usize * side as usize]]).expect("img")
    }

    /// VID-02: a tiny file declaring a big image must neither be REJECTED at open (a valid highly compressible
    /// file looks exactly like this) nor abort the process at decode: `parse` accepts it, and the decode
    /// budget — the caller's policy — turns the over-budget request into a typed `Limit` error before any
    /// plane is allocated.
    #[test]
    fn tiny_file_with_huge_dims_opens_and_decodes_to_a_typed_limit_error() {
        let mut enc = write_video(&[blank_u8(512)], &Profile::lossless_fast(), 1, 0, 0).expect("enc");
        assert!(enc.len() < 4096, "a blank plane is a tiny file ({} B)", enc.len());
        forge_one_tile(&mut enc, 20000); // 4e8 px, one tile: > 256 MiB and > 65536 x file size
        let idx = VideoIndex::parse(&enc).expect("a self-consistent file opens whatever it declares");
        assert_eq!((idx.header.width, idx.header.height), (20000, 20000));
        for limits in [DecodeLimits::default(), DecodeLimits::new(1 << 20)] {
            // 400 MB fits the 8 GiB default (and then fails as corrupt: the body is a 512² blob, not 20000²);
            // the 1 MiB budget is refused up front.
            let r = read_still_with(&enc, &limits);
            assert!(r.is_err(), "{limits:?}");
        }
        assert!(matches!(read_still_with(&enc, &DecodeLimits::new(1 << 20)), Err(DecodeError::Limit { .. })));
        assert!(matches!(read_video_with(&enc, &DecodeLimits::new(1 << 20)), Err(DecodeError::Limit { .. })));
        assert!(matches!(Decoder::open(&enc).expect("open").with_limits(DecodeLimits::new(1 << 20)).frame(0), Err(DecodeError::Limit { .. })));
        // 65535² F32-ish forgery: 17 GiB per plane — over the default budget, still no abort.
        forge_one_tile(&mut enc, 32768);
        set_u32(&mut enc, HDR_AT + 2, 65535);
        set_u32(&mut enc, HDR_AT + 6, 65535);
        assert!(matches!(read_still(&enc), Err(DecodeError::Limit { .. }) | Err(DecodeError::Corrupt(_))));
    }

    /// VID-02 end to end: a real blank image of a size the OLD ratio guard refused (4e8 px = 381 MiB over
    /// its 256 MiB floor, from a file far below 381 MiB / 65536) opens, decodes bit-exact under a large
    /// budget and reports a typed `Limit` under a small one. ~400 MB planes: run explicitly (`--ignored`,
    /// release recommended).
    #[test]
    #[ignore = "400 MB blank-image round trip; run with --release -- --ignored"]
    fn blank_20000_square_roundtrips_under_a_large_budget() {
        let img = blank_u8(20000);
        let enc = write_video(std::slice::from_ref(&img), &Profile::lossless_fast(), 1, 0, 0).expect("enc");
        assert!(enc.len() < 65536, "~{} B file", enc.len());
        let idx = VideoIndex::parse(&enc).expect("opens");
        assert!(decode_frame(&enc, &idx, 0, Chans::All).and_then(Frame::into_image).is_ok(), "default 8 GiB budget");
        let small = VideoIndex::parse(&enc).expect("opens").with_limits(DecodeLimits::new(1 << 20));
        assert!(matches!(decode_frame(&enc, &small, 0, Chans::All).and_then(Frame::into_image), Err(DecodeError::Limit { .. })));
        assert_eq!(read_still(&enc).expect("decodes").planes, img.planes);
    }

    /// The DEFAULT profile (lossless HTJ2K) takes 8/16-bit integer channels end to end: stills and I/P/B
    /// clips of U8 + U16 + F16 + U32 channels (odd size, tiny tiles, values above 0x7FFF) round-trip
    /// bit-exact, including PRED/SKIP tiles whose residuals wrap in the 8/16-bit element width.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_default_profile_roundtrips_u8_u16_clips() {
        let mk = |shift: u32| -> Image {
            let (w, h) = (37u32, 21u32);
            let n = (w * h) as usize;
            let header = Header {
                width: w,
                height: h,
                frame_count: 1,
                tile_w: 16,
                tile_h: 16,
                codec_id: 6,
                codec_params: Vec::new(),
                channels: vec![
                    Channel { name: "R8".into(), dtype: Dtype::U8 },
                    Channel { name: "G16".into(), dtype: Dtype::U16 },
                    Channel { name: "H".into(), dtype: Dtype::F16 },
                    Channel { name: "id".into(), dtype: Dtype::U32 },
                ],
            };
            let u8p: Vec<u8> = (0..n).map(|i| ((i as u32 * 7 + shift * 9) % 256) as u8).collect();
            let u16p: Vec<u8> = (0..n).flat_map(|i| ((i as u32 * 1777 + shift * 4001) as u16).to_le_bytes()).collect();
            let f16p: Vec<u8> = (0..n).flat_map(|i| (((i as u32 * 13 + shift) & 0x7bff) as u16).to_le_bytes()).collect();
            let u32p: Vec<u8> = (0..n).flat_map(|i| ((i as u32).wrapping_mul(2654435761u32) ^ shift).to_le_bytes()).collect();
            Image::new(header, vec![u8p, u16p, f16p, u32p]).expect("img")
        };
        let prof = Profile::htj2k();
        let still = mk(0);
        let enc = write_still(&still, &prof).expect("encode still");
        assert_eq!(read_still(&enc).expect("decode still").planes, still.planes);
        let frames: Vec<Image> = (0..5).map(mk).collect();
        for (gop, b) in [(1u32, 0u32), (5, 0), (5, 1)] {
            let enc = write_video(&frames, &prof, gop, 2, b).expect("encode clip");
            let dec = read_video(&enc).expect("decode clip");
            for (a, f) in dec.iter().zip(&frames) {
                assert_eq!(a.planes, f.planes, "gop {gop} b {b}");
            }
        }
    }

    /// The budget counts what is resident: a whole clip for `read_video_with`, a GOP for `decode_frame`.
    #[test]
    fn budget_counts_resident_frames() {
        let frames: Vec<Image> = (0..6).map(|s| frame(32, 32, 16, s)).collect();
        let enc = write_video(&frames, &Profile::lossless_fast(), 3, 0, 0).expect("enc");
        let fb = frames[0].header.frame_bytes().expect("fb");
        let tile = 16 * 16 * 6; // F16 + F32 channels
        // One frame's decode needs planes + one tile; a GOP of 3 needs 3 frames, the clip 6.
        assert!(read_video_with(&enc, &DecodeLimits::new(6 * fb + tile)).is_ok());
        assert!(matches!(read_video_with(&enc, &DecodeLimits::new(6 * fb - 1)), Err(DecodeError::Limit { .. })));
        let idx = VideoIndex::parse(&enc).expect("idx").with_limits(DecodeLimits::new(3 * fb + tile));
        assert!(decode_frame(&enc, &idx, 4, Chans::All).and_then(Frame::into_image).is_ok(), "a 3-frame GOP fits 3 frames");
        // decode_frame frees a reference after its last user: a P-chain peaks at 2 resident frames, whatever
        // the GOP length, so 3*fb - 1 still fits and only a budget below 2 frames is refused.
        let two = VideoIndex::parse(&enc).expect("idx").with_limits(DecodeLimits::new(3 * fb - 1));
        assert!(decode_frame(&enc, &two, 5, Chans::All).and_then(Frame::into_image).is_ok(), "a P chain needs 2 frames resident, not the GOP");
        let tight = VideoIndex::parse(&enc).expect("idx").with_limits(DecodeLimits::new(2 * fb - 1));
        assert!(matches!(decode_frame(&enc, &tight, 4, Chans::All).and_then(Frame::into_image), Err(DecodeError::Limit { .. })));
    }

    /// Structural index forgeries: offsets out of order / outside the frames region, gop = 0, bad first kind.
    #[test]
    fn forged_index_entries_are_none() {
        let frames: Vec<Image> = (0..3).map(|s| frame(16, 16, 16, s)).collect();
        let enc = write_video(&frames, &Profile::lossless_fast(), 8, 0, 0).expect("enc");
        let index_off = index_off_of(&enc);
        let entry = |i: usize| index_off + 4 + i * IDX_ENTRY;
        let mut bad = enc.clone();
        bad[entry(1) + 1..entry(1) + 9].copy_from_slice(&(index_off as u64).to_le_bytes()); // offset == index start
        assert!(VideoIndex::parse(&bad).is_err(), "offset outside the frames region");
        let mut bad = enc.clone();
        let prev = bad[entry(0) + 1..entry(0) + 9].to_vec();
        bad[entry(1) + 1..entry(1) + 9].copy_from_slice(&prev); // not strictly increasing
        assert!(VideoIndex::parse(&bad).is_err(), "non-increasing offsets");
        let mut bad = enc.clone();
        bad[entry(0)] = FrameKind::Pred as u8;
        assert!(VideoIndex::parse(&bad).is_err(), "first frame must be Intra");
        let mut bad = enc.clone();
        set_u32(&mut bad, 5, 0);
        assert!(VideoIndex::parse(&bad).is_err(), "gop = 0");
        assert!(VideoIndex::parse(&enc).is_ok());
    }

    /// A corrupt GOP must not poison the previously good one: after the failed switch the old GOP still decodes.
    #[test]
    fn decoder_failed_gop_keeps_old_gop_decodable() {
        let frames: Vec<Image> = (0..4).map(|s| frame(32, 32, 16, s)).collect();
        let mut enc = write_video(&frames, &Profile::lossless_fast(), 2, 0, 0).expect("enc");
        let at = VideoIndex::parse(&enc).expect("idx").offsets[2]; // frame 2 = the second GOP's I-frame
        enc[at] = 0xff; // unknown frame kind → that GOP cannot decode
        let mut dec = Decoder::open(&enc).expect("open");
        let first = dec.frame(0).expect("good GOP");
        assert!(dec.frame(2).is_err(), "corrupt GOP fails");
        assert_eq!(dec.frame(0).expect("old GOP still decodable").planes, first.planes);
        assert_eq!(first.planes, frames[0].planes);
    }

    /// The stored P reference index is honored by EVERY read path: a P patched to reference frame 0
    /// (not its predecessor) decodes identically via `read_video` and `read_video_streaming`; a ref
    /// that is not an earlier frame of the GOP is a hard error in both.
    #[test]
    fn p_reference_index_honored_everywhere() {
        let frames: Vec<Image> = (0..3).map(|s| frame(32, 32, 16, s * 5)).collect();
        let enc = write_video(&frames, &Profile::lossless_fast(), 8, 0, 0).expect("enc");
        let at = VideoIndex::parse(&enc).expect("idx").offsets[2];
        let stream = |data: &[u8]| -> Result<Vec<Vec<Vec<u8>>>, String> {
            let idx = VideoIndex::parse(data).map_err(|e| e.to_string())?;
            let out: Mutex<Vec<Option<Vec<Vec<u8>>>>> = Mutex::new((0..3).map(|_| None).collect());
            read_video_streaming(data, &idx, 1 << 20, &|d, img| {
                let mut g = out.lock().map_err(|e| e.to_string())?;
                *g.get_mut(d).ok_or("display index")? = Some(img.planes);
                Ok(())
            })?;
            out.into_inner().map_err(|e| e.to_string())?.into_iter().map(|f| f.ok_or("missing".to_string())).collect()
        };
        // Patch frame 2's ref (frame_hdr = kind, n_refs, ref:u32 at +2) from 1 to 0.
        let mut redirected = enc.clone();
        set_u32(&mut redirected, at + 2, 0);
        let whole: Vec<_> = read_video(&redirected).expect("whole").into_iter().map(|i| i.planes).collect();
        assert_eq!(stream(&redirected).expect("stream"), whole, "paths agree on the stored ref");
        assert_ne!(whole[2], frames[2].planes, "the redirect really changed the reference");
        // Self-reference (not an EARLIER frame) is rejected by both.
        let mut selfref = enc.clone();
        set_u32(&mut selfref, at + 2, 2);
        assert!(read_video(&selfref).is_err());
        assert!(stream(&selfref).is_err());
        // Untouched stream still round-trips through the streaming chain path.
        assert_eq!(stream(&enc).expect("stream ok"), frames.iter().map(|f| f.planes.clone()).collect::<Vec<_>>());
    }

    /// `n_refs` is fixed by the kind; a mismatched frame header is rejected.
    #[test]
    fn frame_hdr_n_refs_must_match_kind() {
        assert!(read_frame_hdr(&mut Rd::new(&[FrameKind::Pred as u8, 0])).is_none());
        assert!(read_frame_hdr(&mut Rd::new(&[FrameKind::Intra as u8, 1, 0, 0, 0, 0, 0, 0, 0, 0])).is_none());
        let mut ok = Vec::new();
        write_frame_hdr(&mut ok, FrameKind::Bidir, &[0, 1], &[(1, -2), (3, 4)]);
        assert_eq!(read_frame_hdr(&mut Rd::new(&ok)), Some((FrameKind::Bidir, vec![0, 1], vec![(1, -2), (3, 4)])));
    }

    /// A zero MV borrows the reference (no full-frame copy); a non-zero MV builds a shifted copy.
    #[test]
    fn zero_mv_borrows_reference() {
        let f = frame(16, 16, 16, 1);
        let pool = PlanePool::new(Budget::new(1 << 20));
        let all = |c: Result<Vec<RefChan<'_>>, VideoError>, borrowed: bool| {
            c.is_ok_and(|c| c.iter().all(|r| matches!(r, RefChan::Borrowed(_)) == borrowed))
        };
        assert!(all(mc_shift(&pool, &f.planes, &f.header, (0, 0)), true));
        assert!(all(mc_shift(&pool, &f.planes, &f.header, (1, 0)), false));
        let bad = [vec![0u8; 3], vec![0u8; 3]];
        assert!(mc_shift(&pool, &bad, &f.header, (1, 0)).is_err(), "inconsistent planes rejected");
        assert!(mc_shift(&pool, &bad, &f.header, (0, 0)).is_err(), "…and for a zero MV too");
        fn bor(p: &[Vec<u8>]) -> Vec<RefChan<'_>> {
            p.iter().map(|v| RefChan::Borrowed(v)).collect()
        }
        assert!(mc_avg(&pool, &bor(&f.planes), &bor(&f.planes[..1]), &f.header).is_err(), "mean needs matching planes");
        assert!(mc_avg(&pool, &bor(&bad), &bor(&f.planes), &f.header).is_err());
        let u = u16_const(4, 4, 4, 10);
        let v = u16_const(4, 4, 4, 21);
        let m = mc_avg(&pool, &bor(&u.planes), &bor(&v.planes), &u.header).expect("mean");
        assert!(m[0].bytes().as_chunks::<2>().0.iter().all(|e| u16::from_le_bytes(*e) == 15), "(10 + 21) >> 1");
    }

    /// A header the format cannot represent (channel name > 255 bytes) is a failed write, not a corrupt file.
    #[test]
    fn unrepresentable_header_fails_write() {
        let mut f = frame(16, 16, 16, 1);
        f.header.channels[0].name = "x".repeat(300);
        assert!(matches!(
            write_video(std::slice::from_ref(&f), &Profile::lossless_fast(), 1, 0, 0),
            Err(crate::VideoError::Header(_))
        ));
        let mut sink = Vec::new();
        let r = write_still_windowed(&f.header, &Profile::lossless_fast(), &Meta::default(), 1 << 30, &mut sink, |_, _| Ok(vec![]));
        assert_eq!(r.map_err(|e| e.kind()), Err(io::ErrorKind::InvalidInput));
        assert!(sink.is_empty());
    }
}
