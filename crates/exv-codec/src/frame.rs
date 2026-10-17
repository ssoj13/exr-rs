//! Tiled `.exv` FRAME codec (steps 1b/1c) — a video frame as a grid of per-tile INTRA / SKIP / PRED.
//!
//! Header-LESS: the container's `Header` is authoritative (dims / tile grid / channels / codec), so a
//! frame carries only tile blobs + a per-tile index. This is the frozen frame shape from
//! `docs/FORMAT.md` §5 (decision: `docs/DESIGN_LOG.md` D-02), minus the B (bidirectional) kind (added in 1d).
//!
//! ```text
//! frame body = [tile blobs …]                            (SKIP tiles contribute 0 bytes)
//!            | index { n_tiles:u32, n_crc:u16, per tile: kind:u8, off:u64, len:u32, crc:[u32; n_crc] }
//!            | index_off:u64 (trailer)                  (off/len rel. to body[0])
//! ```
//! `crc` = CRC-32C ([`exv_format::crc`]) of every CODED blob of the tile: `n_crc = nch` for a per-channel codec (one per
//! channel sub-blob, so a damaged channel fails alone and a channel subset verifies only what it decodes), `n_crc = 1`
//! for a multi-component (HTJ2K) tile (its one codestream). SKIP tiles carry zeros. Every blob accessor of
//! [`FrameReader`] verifies under the reader's [`exv_format::Verify`] policy before a codec or the GPU sees the bytes.
//! Each tile is coded against a **reference** frame (the previous reconstructed frame, already
//! motion-compensated by the caller — the container applies the global MV; for MV=0 it is the raw prev):
//! - **INTRA** (`kind=0`): a self-describing tile blob of the tile itself ([`crate::encode_tile`]).
//! - **PRED**  (`kind=1`): a blob of the integer bit-pattern **residual** `cur − ref` ([`temporal::delta`])
//!   — used only when it encodes SMALLER than INTRA; decode = `decode + undelta(ref)`.
//! - **SKIP**  (`kind=3`, `len=0`): `cur` is byte-identical to `ref`'s co-located tile → reuse it.
//!
//! Correctness holds for ANY codec (lossy or lossless): the codec is deterministic and SKIP/PRED
//! reference the *reconstructed* previous frame, so a decoded frame is exactly what its tiles encode.
//! (`2`=BI is reserved for 1d.)

use std::borrow::Cow;
use std::io::{self, Write};
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;

use crate::stream::Rect;
use crate::registry::{self, OpenError, TileLayout};
use crate::{CodecError, Dst, LaneBuf, Profile, TileBuf, decode_channel, decode_tile_dst, encode_tile, temporal};
use exv_format::crc;
use exv_format::limits::{alloc_zeroed, cap};
use exv_format::{AlignedBuf, CodecId, DecodeError, DecodeLimits, Dtype, Header, Rd, Verify};
use exv_tile::{PlaneBuf, PlanePool};

/// Per-tile prediction kind on the wire (FROZEN discriminants; 1b/1c emit INTRA / PRED / SKIP).
pub const TILE_INTRA: u8 = 0;
/// Forward-predicted tile: the blob is the `cur − ref` residual ([`temporal::delta`]).
pub const TILE_PRED: u8 = 1;
/// Reserved: bidirectional tile (1d).
pub const TILE_BI: u8 = 2;
/// SKIP: this tile == the reference frame's co-located tile (`len==0`, reuse it).
pub const TILE_SKIP: u8 = 3;

/// Bytes per index entry on the wire before its checksums: `kind:u8, off:u64, len:u32` (then `crc:[u32; n_crc]`).
const ENTRY_HEAD: usize = 13;

/// One per-tile index entry, offsets relative to the frame body start (its CRCs live in the index's flat table).
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// `TILE_*` prediction kind.
    kind: u8,
    /// Blob offset (relative to the frame body start).
    off: u64,
    /// Blob length (`0` for SKIP).
    len: u32,
}

impl Entry {
    /// The entry of a coded tile blob placed at body offset `off`.
    fn coded(kind: u8, off: u64, blob: &[u8]) -> Result<Self, CodecError> {
        Ok(Self { kind, off, len: blob_len(blob)? })
    }

    /// A SKIP entry (no blob).
    const SKIP: Self = Self { kind: TILE_SKIP, off: 0, len: 0 };
}

/// Checksums per tile of a `layout` with `nch` channels: one per channel sub-blob, or one for the multi-component blob.
fn crcs_per_tile(layout: TileLayout, nch: usize) -> usize {
    match layout {
        TileLayout::PerChannel => nch,
        TileLayout::MultiComp => 1,
    }
}

/// The CRC-32C of every coded blob of a tile blob: each channel sub-blob (per-channel layout) or the whole blob
/// (multi-component). Runs on the parallel encode lanes. A per-channel blob that does not split is an encoder bug, so a
/// typed error rather than a checksum of the wrong bytes.
fn blob_crcs(blob: &[u8], layout: TileLayout, nch: usize) -> Result<Vec<u32>, CodecError> {
    Ok(match layout {
        TileLayout::MultiComp => vec![crc::crc32c(blob)],
        TileLayout::PerChannel => crate::stream::split_tile_blob(blob, nch)
            .ok_or(CodecError::Dims("tile blob does not split into its channels"))?
            .into_iter()
            .map(crc::crc32c)
            .collect(),
    })
}

/// Gather a tile rectangle's raw bytes for ONE channel from a full-size plane (strided copy —
/// the inverse of [`scatter`]). `plane` is `full_w*full_h*esz` (callers validate it with
/// [`planes_ok`]); the result is `r.w*r.h*esz`.
fn gather(plane: &[u8], full_w: u32, r: Rect, esz: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(r.w as usize * r.h as usize * esz);
    for src in rect_rows(plane, full_w, r, esz) {
        out.extend_from_slice(src);
    }
    out
}

/// The rows of tile rect `r` inside a full-size plane, top to bottom (`r.w * esz` bytes each). `plane` is
/// `full_w*full_h*esz` (callers validate it with [`planes_ok`]); a row that would fall outside is skipped, never
/// a panic (the callers then see a short tile and fail their length checks).
fn rect_rows(plane: &[u8], full_w: u32, r: Rect, esz: usize) -> impl Iterator<Item = &[u8]> {
    let row = r.w as usize * esz;
    (0..r.h).filter_map(move |y| {
        let s = ((r.y + y) as usize * full_w as usize + r.x as usize) * esz;
        plane.get(s..s + row)
    })
}

/// `true` iff `planes` is exactly one full-size plane per channel of `h` (the precondition of
/// [`gather`]; checked once at the API boundary instead of panicking mid-tile).
fn planes_ok(planes: &[Vec<u8>], h: &Header, dtypes: &[Dtype]) -> bool {
    planes.len() == dtypes.len() && planes.iter().zip(dtypes).all(|(p, &d)| h.plane_len(d) == Some(p.len()))
}

/// Which channels a whole-frame decode materializes ([`FrameReader::decode_all_into`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Chans<'a> {
    /// Every channel of the header.
    #[default]
    All,
    /// Only these header channel indices (any order, duplicates allowed). The others are NOT decoded: per-channel
    /// codecs skip their sub-blobs, HTJ2K stops after the last wanted component.
    Only(&'a [usize]),
}

impl Chans<'_> {
    /// Is channel `ci` wanted?
    pub fn has(&self, ci: usize) -> bool {
        match self {
            Chans::All => true,
            Chans::Only(list) => list.contains(&ci),
        }
    }
}

/// A frame's decoded planes: one slot per header channel, `None` = not decoded (unselected by [`Chans`]), never a
/// zero-filled plane pretending to be data. The REUSABLE destination of [`FrameReader::decode_all_into`]:
///
/// - a slot whose length already matches is decoded over IN PLACE (no allocation, no zeroing, no page faults);
/// - otherwise its buffer comes from the [`PlanePool`] given to [`Planes::pooled`] (else a fresh allocation), and
///   every buffer it drops (a resize, an unselected channel, the `Planes` itself) goes back to that pool.
///
/// So playback / transcode keeps one `Planes` per in-flight frame (or one pool per clip) and decodes into it.
/// After a failed decode the contents are unspecified (the shape stays valid).
#[derive(Debug, Default)]
pub struct Planes {
    slots: Vec<Option<PlaneBuf>>,
    pool: Option<PlanePool>,
}

impl Planes {
    /// Empty planes whose buffers come from and return to `pool`.
    pub fn pooled(pool: PlanePool) -> Self {
        Self { slots: Vec::new(), pool: Some(pool) }
    }

    /// Number of channel slots (the header's channel count after a decode).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// `true` before the first decode.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Channel `ci`'s plane bytes (row-major `w*h*esz`), `None` when not decoded.
    pub fn get(&self, ci: usize) -> Option<&[u8]> {
        self.slots.get(ci)?.as_deref()
    }

    /// Bytes held by the decoded planes (the resident cost a frame cache charges).
    pub fn bytes(&self) -> usize {
        self.slots.iter().flatten().map(|b| b.len()).sum()
    }

    /// The planes as owned `Vec`s (`None` = not decoded): a COPY ([`PlaneBuf::into_vec`]); the aligned buffers go back
    /// to the pool. Hot whole-frame `Vec` paths decode straight into `Vec`s instead ([`FrameReader::decode_all`]).
    pub fn into_vecs(mut self) -> Vec<Option<Vec<u8>>> {
        std::mem::take(&mut self.slots).into_iter().map(|s| s.map(PlaneBuf::into_vec)).collect()
    }

    /// One mutable plane per slot (`None` = not decoded): the destination table of the whole-frame decoder.
    fn slices(&mut self) -> Vec<Option<&mut [u8]>> {
        self.slots.iter_mut().map(|s| s.as_deref_mut()).collect()
    }

    /// Shape the slots for a `w`x`h` decode of `dtypes` restricted to `chans`, reusing what fits.
    fn fit(&mut self, dtypes: &[Dtype], w: u64, h: u64, chans: Chans<'_>) -> Result<(), DecodeError> {
        self.slots.resize_with(dtypes.len(), || None);
        for (ci, (slot, d)) in self.slots.iter_mut().zip(dtypes).enumerate() {
            if !chans.has(ci) {
                *slot = None;
                continue;
            }
            let len = plane_bytes(w, h, *d)?;
            if slot.as_ref().is_some_and(|b| b.len() == len) {
                continue;
            }
            *slot = None; // give the wrong-sized buffer back first: the pool may hand it to another channel
            *slot = Some(match &self.pool {
                Some(pool) => pool.take(len)?,
                None => PlaneBuf::new(AlignedBuf::zeroed(len)?),
            });
        }
        Ok(())
    }
}

/// Bytes of one `w`x`h` plane of `d`, or a typed error when it overflows `usize`.
fn plane_bytes(w: u64, h: u64, d: Dtype) -> Result<usize, DecodeError> {
    w.checked_mul(h)
        .and_then(|n| n.checked_mul(d.size() as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(DecodeError::Request("plane size overflows"))
}

/// Unpooled planes holding every channel (an `Image`'s planes wrapped for the `Planes` consumers; dropping frees them).
/// Copies each plane into aligned storage ([`AlignedBuf`]).
impl From<Vec<Vec<u8>>> for Planes {
    fn from(planes: Vec<Vec<u8>>) -> Self {
        Self { slots: planes.iter().map(|p| Some(PlaneBuf::new(AlignedBuf::from(p.as_slice())))).collect(), pool: None }
    }
}

/// A BORROWED reference frame (the MC'd previous frame of SKIP/PRED tiles): one optional full-size plane per channel. A
/// channel the caller did not decode is `None`; that is only an error when a SKIP/PRED tile of a WANTED channel needs it.
/// Built from either plane holder, so `Vec` planes (motion-compensated copies) and pooled [`Planes`] feed the same decoder:
/// `RefPlanes::from(&planes)` / `RefPlanes::from(vecs.as_slice())`.
#[derive(Clone, Debug, Default)]
pub struct RefPlanes<'r>(Vec<Option<&'r [u8]>>);

impl<'r> From<&'r [Vec<u8>]> for RefPlanes<'r> {
    fn from(planes: &'r [Vec<u8>]) -> Self {
        Self(planes.iter().map(|p| Some(p.as_slice())).collect())
    }
}

impl<'r> From<&'r Planes> for RefPlanes<'r> {
    fn from(planes: &'r Planes) -> Self {
        Self((0..planes.len()).map(|ci| planes.get(ci)).collect())
    }
}

/// One optional plane per channel in header order: how a container builds a reference it motion-compensated per
/// channel (borrowed planes and shifted/averaged copies side by side, unwanted channels `None`).
impl<'r> FromIterator<Option<&'r [u8]>> for RefPlanes<'r> {
    fn from_iter<I: IntoIterator<Item = Option<&'r [u8]>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// One tile row of the output planes during [`FrameReader::decode_all_into`]: first the raw per-channel bands, then
/// (split by the first tile that needs it) one [`Dst`] table per tile of the row, each taken exactly once. Splitting
/// lazily keeps the row tables to the bands in flight (a full table up front is ~1.3 GB of slice headers at 65k^2).
enum Band<'p> {
    /// Per channel: the band's rows `[ty*tile_h, ..)` of the plane, `None` = channel not decoded.
    Raw(Vec<Option<&'p mut [u8]>>),
    /// Per tile `tx`: its destinations, `None` once taken.
    Split(Vec<Option<Vec<Dst<'p>>>>),
}

impl<'p> Band<'p> {
    /// Take tile `tx`'s destinations (splitting the band on first use).
    fn take(&mut self, tx: usize, h: &Header, dtypes: &[Dtype]) -> Result<Vec<Dst<'p>>, DecodeError> {
        if let Band::Raw(chans) = self {
            let tx_n = h.tiles_x() as usize;
            let mut tiles: Vec<Vec<Dst<'p>>> = (0..tx_n).map(|_| Vec::with_capacity(chans.len())).collect();
            for (band, d) in std::mem::take(chans).into_iter().zip(dtypes) {
                let Some(band) = band else {
                    tiles.iter_mut().for_each(|t| t.push(Dst::Skip));
                    continue;
                };
                let (esz, tw) = (d.size(), h.tile_w as usize);
                let mut rows: Vec<Vec<&'p mut [u8]>> = (0..tx_n).map(|_| Vec::with_capacity(h.tile_h as usize)).collect();
                for mut line in band.chunks_mut((h.width as usize * esz).max(1)) {
                    for r in &mut rows {
                        let (piece, rest) = line.split_at_mut((tw * esz).min(line.len()));
                        r.push(piece);
                        line = rest;
                    }
                }
                for (t, r) in tiles.iter_mut().zip(rows) {
                    t.push(Dst::Rows(r));
                }
            }
            *self = Band::Split(tiles.into_iter().map(Some).collect());
        }
        match self {
            Band::Split(tiles) => tiles.get_mut(tx).and_then(Option::take).ok_or(DecodeError::Corrupt("tile decoded twice")),
            Band::Raw(_) => Err(DecodeError::Corrupt("tile row band")),
        }
    }
}

/// Run `work(t, scratch)` for every `t` in `0..n` on `win` parallel LANES: lane `l` owns tiles `l, l+win, l+2·win…`
/// in ascending order and keeps ONE scratch `B` for all of them. That is a sliding window with no barrier: exactly
/// `win` tiles are ever in flight (the byte-budgeted window the caller computed), and a lane that finishes a tile
/// starts its next at once instead of waiting for the slowest tile of a batch. `work` must be order-independent
/// (it writes disjoint places).
///
/// The error returned is the one of the LOWEST failing tile index, whatever the thread count: a lane stops once a
/// lower-index failure is recorded (nothing it could still decode would be the first), and otherwise runs on, so
/// every tile below the final minimum was attempted.
fn par_lanes<B: Default + Send>(n: u64, win: u64, work: impl Fn(u64, &mut B) -> Result<(), DecodeError> + Sync) -> Result<(), DecodeError> {
    let lanes = win.clamp(1, n.max(1));
    let first: Mutex<Option<(u64, DecodeError)>> = Mutex::new(None);
    let lock = || first.lock().unwrap_or_else(PoisonError::into_inner);
    (0..lanes).into_par_iter().for_each(|lane| {
        let (mut t, mut scratch) = (lane, B::default());
        while t < n {
            if lock().as_ref().is_some_and(|&(e, _)| e < t) {
                break;
            }
            if let Err(err) = work(t, &mut scratch) {
                let mut g = lock();
                if g.as_ref().is_none_or(|&(e, _)| t < e) {
                    *g = Some((t, err));
                }
                break;
            }
            t += lanes;
        }
    });
    let failed = first.into_inner().unwrap_or_else(PoisonError::into_inner);
    failed.map_or(Ok(()), |(_, e)| Err(e))
}

/// Encode one tile's per-channel raw bytes into a self-describing blob (thin wrapper over the single
/// tile-codec authority [`crate::encode_tile`]).
fn encode_chans(chans: &[Vec<u8>], dtypes: &[Dtype], tw: u32, th: u32, profile: &Profile) -> Result<Vec<u8>, CodecError> {
    let refs: Vec<&[u8]> = chans.iter().map(Vec::as_slice).collect();
    encode_tile(&refs, dtypes, tw, th, profile)
}

/// Gather every channel of tile `(x,y,tw,th)` from full-size `planes` (owned `Vec` planes or borrowed slices).
fn gather_tile<P: AsRef<[u8]>>(planes: &[P], h: &Header, dtypes: &[Dtype], r: Rect) -> Vec<Vec<u8>> {
    (0..h.channels.len()).map(|ci| gather(planes[ci].as_ref(), h.width, r, dtypes[ci].size())).collect()
}

/// Serialize the per-tile index + `index_off` trailer (shared by both frame-body writers so their
/// layouts cannot drift). `index_off` = byte length of the tile blobs that precede it; `crcs` = `n_crc` checksums per
/// entry, in entry order.
fn index_bytes(entries: &[Entry], crcs: &[u32], n_crc: usize, index_off: u64) -> Result<Vec<u8>, CodecError> {
    let n = u32::try_from(entries.len()).map_err(|_| CodecError::Dims("tile count exceeds u32"))?;
    let n_crc16 = u16::try_from(n_crc).map_err(|_| CodecError::Dims("checksums per tile exceed u16"))?;
    if crcs.len() != entries.len() * n_crc {
        return Err(CodecError::Dims("checksum table does not match the tile entries"));
    }
    let mut idx = Vec::with_capacity(6 + entries.len() * (ENTRY_HEAD + 4 * n_crc) + 8);
    idx.extend_from_slice(&n.to_le_bytes());
    idx.extend_from_slice(&n_crc16.to_le_bytes());
    for (e, c) in entries.iter().zip(crcs.chunks_exact(n_crc.max(1))) {
        idx.push(e.kind);
        idx.extend_from_slice(&e.off.to_le_bytes());
        idx.extend_from_slice(&e.len.to_le_bytes());
        c.iter().for_each(|v| idx.extend_from_slice(&v.to_le_bytes()));
    }
    idx.extend_from_slice(&index_off.to_le_bytes());
    Ok(idx)
}

/// `blob.len()` as the on-disk `u32` (a silently truncated length would only fail at decode).
fn blob_len(blob: &[u8]) -> Result<u32, CodecError> {
    u32::try_from(blob.len()).map_err(|_| CodecError::Dims("tile blob exceeds u32"))
}

/// Encode a frame's full-size planes into a header-less tiled frame body. `reference` = the previous
/// RECONSTRUCTED frame's planes, already motion-compensated by the caller (or `None` for an I-frame).
/// Per tile: SKIP if byte-identical to the reference; else the smaller of PRED (residual) / INTRA.
/// Tiles are row-major. A front over [`write_frame_tiles_ref`] for `Vec` references.
///
/// # Errors
/// [`CodecError::Dims`] when `planes`/`reference` are not one full-size plane per channel, otherwise
/// whatever [`crate::encode_tile`] reports (e.g. U8/U16 under HTJ2K) — never a silently broken body.
pub fn write_frame_tiles(
    planes: &[Vec<u8>],
    reference: Option<&[Vec<u8>]>,
    header: &Header,
    profile: &Profile,
) -> Result<Vec<u8>, CodecError> {
    write_frame_tiles_ref(planes, reference.map(RefPlanes::from).as_ref(), header, profile)
}

/// [`write_frame_tiles`] over a [`RefPlanes`] reference: a container that motion-compensates into POOLED planes (the
/// clip writer) borrows them here without moving them into `Vec`s. Every reference channel must be present
/// (`None` = [`CodecError::Dims`]: an encoder needs the whole reference, unlike a subset decode).
///
/// # Errors
/// As [`write_frame_tiles`].
pub fn write_frame_tiles_ref(
    planes: &[Vec<u8>],
    reference: Option<&RefPlanes<'_>>,
    header: &Header,
    profile: &Profile,
) -> Result<Vec<u8>, CodecError> {
    let h = header;
    let tx_n = h.tiles_x() as u64;
    let dtypes: Vec<Dtype> = h.channels.iter().map(|c| c.dtype).collect();
    let dims = || CodecError::Dims("planes/reference are not one full-size plane per channel");
    // The reference as one full-size slice per channel (a missing / mis-sized channel is the same `Dims` error).
    let reference: Option<Vec<&[u8]>> = reference
        .map(|r| {
            let v = (r.0.len() == dtypes.len())
                .then(|| r.0.iter().zip(&dtypes).map(|(p, &d)| p.filter(|p| h.plane_len(d) == Some(p.len()))).collect())
                .flatten();
            v.ok_or_else(dims)
        })
        .transpose()?;
    if !planes_ok(planes, h, &dtypes) {
        return Err(dims());
    }
    let n_tiles = h.tile_count();
    u32::try_from(n_tiles).map_err(|_| CodecError::Dims("tile count exceeds u32"))?; // checked BEFORE any n-sized allocation
    // The planes are already resident, so the encode's transient (a window of raw tile + gathered reference +
    // encoder scratch + blob) may add only `window_spare` of them, whatever the tile size.
    let frame_bytes: u64 = planes.iter().map(|p| p.len() as u64).sum();
    let win = exv_tile::par_window(exv_tile::window_spare(frame_bytes), tile_encode_cost(h, &dtypes));

    let (layout, nch) = (profile.backend().layout(), dtypes.len());
    let n_crc = crcs_per_tile(layout, nch);
    let mut body = Vec::new();
    let mut entries: Vec<Entry> = Vec::with_capacity(n_tiles as usize);
    let mut crcs: Vec<u32> = Vec::with_capacity(n_tiles as usize * n_crc);
    exv_tile::pipe(
        n_tiles,
        win,
        |t| {
            let (tx, ty) = ((t % tx_n) as u32, (t / tx_n) as u32);
            let r = Rect::of_tile(h, tx, ty);
            let (tw, th) = (r.w, r.h);
            let cur = gather_tile(planes, h, &dtypes, r);
            let (kind, blob) = match reference.as_deref() {
                None => (TILE_INTRA, encode_chans(&cur, &dtypes, tw, th, profile)?),
                Some(refp) => {
                    let refc = gather_tile(refp, h, &dtypes, r);
                    if refc == cur {
                        (TILE_SKIP, Vec::new())
                    } else if profile.is_lossless() {
                        // PRED = integer bit-pattern residual vs the (MC) reference; keep it only if it
                        // encodes smaller than coding the tile raw (INTRA) — else a scene-cut/large-motion
                        // tile just stays INTRA. Both decode to `cur` exactly (lossless residual add-back).
                        // Lossless ONLY: open-loop residual would drift on a lossy codec (guarded above).
                        let resid = (0..h.channels.len())
                            .map(|ci| temporal::delta(&cur[ci], &refc[ci], dtypes[ci]))
                            .collect::<Result<Vec<Vec<u8>>, CodecError>>()?;
                        let pred = encode_chans(&resid, &dtypes, tw, th, profile)?;
                        let intra = encode_chans(&cur, &dtypes, tw, th, profile)?;
                        if pred.len() < intra.len() { (TILE_PRED, pred) } else { (TILE_INTRA, intra) }
                    } else {
                        // Lossy codec: no temporal PRED (drift) — code the tile raw. SKIP above is still safe.
                        (TILE_INTRA, encode_chans(&cur, &dtypes, tw, th, profile)?)
                    }
                }
            };
            // Checksummed here, on the parallel encode lanes, so the ordered sink stays a plain append.
            let sums = if kind == TILE_SKIP { vec![0; n_crc] } else { blob_crcs(&blob, layout, nch)? };
            Ok::<_, CodecError>((kind, blob, sums))
        },
        |_, (kind, blob, sums)| {
            if kind == TILE_SKIP {
                entries.push(Entry::SKIP);
            } else {
                entries.push(Entry::coded(kind, body.len() as u64, &blob)?);
                body.extend_from_slice(&blob);
            }
            crcs.extend_from_slice(&sums);
            Ok(())
        },
    )?;
    let index_off = body.len() as u64;
    body.extend_from_slice(&index_bytes(&entries, &crcs, n_crc, index_off)?);
    Ok(body)
}

/// Bytes ONE in-flight tile holds while it encodes: the raw tile (an edge-clamped `min(tile, image)` one is the
/// worst case), its gathered reference / residual copy, the encoder scratch and the coded blob. That is the
/// measured `Stage::Encode` multiple of the raw tile bytes ([`exv_tile::Stage`]); the tile window is
/// `budget / this` ([`exv_tile::par_window`]), so a user-chosen 16384^2 tile gets a window of 1, not 64 of them.
fn tile_encode_cost(h: &Header, dtypes: &[Dtype]) -> u64 {
    let px = (h.tile_w.min(h.width) as u64) * (h.tile_h.min(h.height) as u64);
    let esz: u64 = dtypes.iter().map(|d| d.size() as u64).sum();
    px.saturating_mul(esz).saturating_mul(exv_tile::Stage::Encode.cost())
}

/// Streaming, windowed **all-INTRA** frame-body writer — the out-of-core WRITE path for a still
/// (= a 1-frame `.exv`). Emits the header-less
/// FRAME-body layout (per-tile `kind` byte + body-relative offsets), so it drops straight under the
/// `.exv` container (`exv-video`). `budget` is the byte budget of the encode window (raw tiles, encoder scratch
/// and blobs in flight, see [`tile_encode_cost`]); `produce(t0, t1)` reads the raw per-channel bytes of tiles
/// `[t0, t1)` IN ORDER (`produce(..)[i][ci]` = `tw*th*esz` bytes of tile `t0+i`, channel `ci`); each
/// window is read serially, encoded in parallel, and written in order — peak heap = one window, NOT
/// `O(image)`. Every tile is INTRA (a still has no temporal reference). Returns the frame-body byte
/// length. The byte layout is IDENTICAL to [`write_frame_tiles`] with `reference == None`.
///
/// # Errors
/// `io::ErrorKind::InvalidData` when `produce` returns the wrong number of tiles, channels or plane
/// sizes, or when a tile fails to encode ([`CodecError`]); otherwise the I/O error.
pub fn write_intra_frame_body_windowed(
    header: &Header,
    profile: &Profile,
    budget: u64,
    out: &mut impl Write,
    mut produce: impl FnMut(u64, u64) -> io::Result<Vec<Vec<Vec<u8>>>>,
) -> io::Result<u64> {
    let tx_n = header.tiles_x() as u64;
    let dtypes: Vec<Dtype> = header.channels.iter().map(|c| c.dtype).collect();
    let n_tiles = header.tile_count();
    let (layout, nch) = (profile.backend().layout(), dtypes.len());
    let n_crc = crcs_per_tile(layout, nch);
    let mut entries: Vec<Entry> = Vec::with_capacity(n_tiles as usize);
    let mut crcs: Vec<u32> = Vec::with_capacity(n_tiles as usize * n_crc);
    let mut pos: u64 = 0; // running frame-body offset (relative to body[0])
    let win = exv_tile::par_window(budget, tile_encode_cost(header, &dtypes));

    let mut t0 = 0u64;
    while t0 < n_tiles {
        let t1 = (t0 + win).min(n_tiles);
        let raw = produce(t0, t1)?; // serial read of the window's raw tiles
        if raw.len() as u64 != t1 - t0 {
            return Err(CodecError::Dims("producer returned the wrong number of tiles").into());
        }
        // Parallel INTRA encode of the pre-read window; ordered write keeps the index offsets correct.
        // (`encode_tile` validates the channel count and every plane length.)
        let batch = raw
            .into_par_iter()
            .enumerate()
            .map(|(i, chans)| {
                let t = t0 + i as u64;
                let (tx, ty) = ((t % tx_n) as u32, (t / tx_n) as u32);
                let (_, _, tw, th) = header.tile_rect(tx, ty);
                let blob = encode_chans(&chans, &dtypes, tw, th, profile)?;
                let sums = blob_crcs(&blob, layout, nch)?;
                Ok((blob, sums))
            })
            .collect::<Result<Vec<(Vec<u8>, Vec<u32>)>, CodecError>>()?;
        for (blob, sums) in batch {
            out.write_all(&blob)?;
            entries.push(Entry::coded(TILE_INTRA, pos, &blob)?);
            crcs.extend_from_slice(&sums);
            pos += blob.len() as u64;
        }
        t0 = t1;
    }

    // Frame-body per-tile index + `index_off` trailer (same framing as `write_frame_tiles`).
    let idx = index_bytes(&entries, &crcs, n_crc, pos)?;
    out.write_all(&idx)?;
    Ok(pos + idx.len() as u64)
}

/// Decode a header-less tiled frame body into full-size planes. `reference` = the previous
/// reconstructed frame's planes (motion-compensated by the caller), required iff the body has any
/// SKIP/PRED tiles. `limits` is the caller's budget for the planes this materializes (plus one tile):
/// the index is validated FIRST and the budget checked BEFORE any plane is allocated.
///
/// # Errors
/// [`DecodeError::Corrupt`] (bad body / index / tile), [`DecodeError::Request`] (a SKIP/PRED tile without a
/// matching reference), [`DecodeError::Limit`] (over budget), [`DecodeError::Alloc`] (allocator refused),
/// [`DecodeError::Checksum`] (a tile blob does not match its index CRC; `limits.verify` = [`exv_format::Verify::On`]).
pub fn read_frame_tiles(
    body: &[u8],
    header: &Header,
    profile: &Profile,
    reference: Option<&[Vec<u8>]>,
    limits: &DecodeLimits,
) -> Result<Vec<Vec<u8>>, DecodeError> {
    let _s = tracing::info_span!("read_frame_tiles", w = header.width, h = header.height, nch = header.channels.len()).entered();
    let reader = tracing::info_span!("open_index").in_scope(|| FrameReader::open(body, header.clone()))?.with_limits(*limits);
    reader.decode_all(profile, reference)
}

/// Parse a frame body's per-tile index (`… [n_tiles:u32][n_crc:u16][kind:u8, off:u64, len:u32, crc:[u32; n_crc]]×n … [index_off:u64]`)
/// against the expected tile count; validates each INTRA/PRED blob lies within the tile-blobs region.
/// `n_crc` must be `1` or `nch` (the codec's layout picks which, checked by [`FrameIndex::parse`]).
/// Returns `(entries, crcs, n_crc)`, or [`DecodeError::Corrupt`] naming the broken rule.
fn parse_frame_index(body: &[u8], expected_tiles: u64, nch: usize) -> Result<(Vec<Entry>, Vec<u32>, usize), DecodeError> {
    const SHORT: DecodeError = DecodeError::Corrupt("frame tile index truncated");
    let tail = body.len().checked_sub(8).ok_or(SHORT)?; // the `index_off` trailer
    let index_off = Rd::at(body, tail).u64().and_then(|v| usize::try_from(v).ok()).ok_or(SHORT)?;
    if index_off > tail {
        return Err(DecodeError::Corrupt("frame tile index offset outside the body"));
    }
    let mut r = Rd::at(body, index_off);
    let n = r.u32().ok_or(SHORT)? as usize;
    if n as u64 != expected_tiles {
        return Err(DecodeError::Corrupt("frame tile count differs from the header's tile grid"));
    }
    let n_crc = r.u16().ok_or(SHORT)? as usize;
    if n_crc != 1 && n_crc != nch {
        return Err(DecodeError::Corrupt("checksums per tile are neither 1 nor the channel count"));
    }
    let entry = ENTRY_HEAD + 4 * n_crc;
    // `expected_tiles` comes from the (hostile) header too, so `n` alone proves nothing: the entries must
    // physically fit between here and the trailer. Reject BEFORE any n-sized allocation (a 100-byte
    // file claiming ~2^32 tiles used to abort in `with_capacity`).
    let avail = tail.saturating_sub(r.pos());
    if n.checked_mul(entry).is_none_or(|b| b > avail) {
        return Err(SHORT);
    }
    let mut entries = Vec::with_capacity(cap(n, avail, entry));
    let mut crcs = Vec::with_capacity(cap(n, avail, entry) * n_crc);
    for _ in 0..n {
        let (kind, off, len) = (r.u8().ok_or(SHORT)?, r.u64().ok_or(SHORT)?, r.u32().ok_or(SHORT)?);
        if kind != TILE_SKIP && off.checked_add(len as u64).is_none_or(|end| end > index_off as u64) {
            return Err(DecodeError::Corrupt("tile blob outside the tile-blob region"));
        }
        entries.push(Entry { kind, off, len });
        for _ in 0..n_crc {
            crcs.push(r.u32().ok_or(SHORT)?);
        }
    }
    Ok((entries, crcs, n_crc))
}

/// The OWNED, parsed per-tile index of a frame body (+ the header, dtypes and codec it decodes under).
///
/// Why separate from [`FrameReader`]: parsing the index is O(n_tiles), so a long-lived tile source (the `.exv`
/// [`TileSource`](exv_tile::TileSource) in `exv-video`, the GPU atlas) parses it ONCE and hands out cheap borrowing
/// readers per call ([`FrameReader::over`]), instead of re-parsing on every request.
#[derive(Clone)]
pub struct FrameIndex {
    /// OWNED so a returned tile blob's lifetime is tied ONLY to `body` (the container data), independent of
    /// where the `Header` came from.
    header: Header,
    dtypes: Vec<Dtype>,
    entries: Vec<Entry>,
    /// `n_crc` CRC-32C per entry, flat in entry order (module docs: one per channel sub-blob or one per tile).
    crcs: Vec<u32>,
    n_crc: usize,
    /// The header's codec resolved ONCE at parse (typed reason kept), not re-parsed on every call.
    profile: Result<Profile, OpenError>,
    /// Some tile is SKIP/PRED/BI (not INTRA): temporal reconstruction holds a gathered reference + the sum beside
    /// the residual, which [`FrameReader::check_budget`] must charge. Computed once (the check runs per tile call).
    temporal: bool,
}

impl FrameIndex {
    /// Parse the body's tile index against `header`'s tile grid. Takes the `Header` BY VALUE (cheap — dims +
    /// channel list) so the index is self-contained.
    ///
    /// # Errors
    /// [`DecodeError::Corrupt`] naming the broken index rule (truncated, tile count != the header's grid, a blob
    /// outside the blob region, a checksum count that does not fit the codec's tile layout). An unopenable codec is
    /// NOT an error here: it is kept and reported by the first decode (so a viewer can still inspect the tile index).
    pub fn parse(body: &[u8], header: Header) -> Result<Self, DecodeError> {
        let nch = header.channels.len();
        let (entries, crcs, n_crc) = parse_frame_index(body, header.tile_count(), nch)?;
        let dtypes = header.channels.iter().map(|c| c.dtype).collect();
        let profile = registry::open(CodecId(header.codec_id), &header.codec_params);
        if let Ok(p) = &profile
            && n_crc != crcs_per_tile(p.backend().layout(), nch)
        {
            return Err(DecodeError::Corrupt("checksums per tile do not match the codec's tile layout"));
        }
        let temporal = entries.iter().any(|e| e.kind != TILE_INTRA);
        Ok(Self { header, dtypes, entries, crcs, n_crc, profile, temporal })
    }

    /// The `n_crc` index checksums of tile `t`.
    fn crcs(&self, t: u32) -> &[u32] {
        let i = t as usize * self.n_crc;
        self.crcs.get(i..i + self.n_crc).unwrap_or(&[])
    }

    /// The authoritative container `Header`.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Per-channel element types (header order).
    pub fn dtypes(&self) -> &[Dtype] {
        &self.dtypes
    }

    /// Number of tiles (row-major); equals `header.tile_count()`.
    pub fn tile_count(&self) -> u32 {
        self.entries.len() as u32
    }

    /// The prediction kind of tile `t` ([`TILE_INTRA`]/[`TILE_PRED`]/[`TILE_SKIP`]).
    pub fn tile_kind(&self, t: u32) -> Option<u8> {
        self.entries.get(t as usize).map(|e| e.kind)
    }

    /// The pixel rect `(x, y, tw, th)` of tile `t`.
    pub fn tile_rect(&self, t: u32) -> Option<(u32, u32, u32, u32)> {
        if t >= self.tile_count() {
            return None;
        }
        let tx_n = self.header.tiles_x();
        Some(self.header.tile_rect(t % tx_n, t / tx_n))
    }

    /// The codec profile reconstructed from the header; the typed [`OpenError`] says why a codec cannot be opened.
    pub fn profile(&self) -> Result<Profile, OpenError> {
        self.profile.clone()
    }
}

/// Lazy reader over a header-less tiled frame body: decodes tiles ON DEMAND (INTRA/PRED/SKIP resolved against a
/// `reference`) through a parsed [`FrameIndex`]. The basis for region / partial decode of a frame (viewer, GPU
/// upload, still = a 1-frame `.exv`). An all-INTRA frame (a still / I-frame) needs no reference.
///
/// Every method that materializes decoded pixels first checks the reader's [`DecodeLimits`] (default
/// [`DecodeLimits::default`]; set with [`Self::with_limits`]) and returns [`DecodeError::Limit`] over budget.
pub struct FrameReader<'a> {
    body: &'a [u8],
    /// Owned by [`Self::open`], borrowed by [`Self::over`] (a source that parsed it once).
    idx: Cow<'a, FrameIndex>,
    limits: DecodeLimits,
}

impl<'a> FrameReader<'a> {
    /// Parse `body`'s tile index against `header` ([`FrameIndex::parse`]) and read through it.
    ///
    /// # Errors
    /// As [`FrameIndex::parse`].
    pub fn open(body: &'a [u8], header: Header) -> Result<Self, DecodeError> {
        let idx = FrameIndex::parse(body, header)?;
        Ok(Self { body, idx: Cow::Owned(idx), limits: DecodeLimits::default() })
    }

    /// Read `body` through an index parsed from THE SAME body (no re-parse; the cheap per-call reader of a
    /// long-lived source).
    pub fn over(body: &'a [u8], idx: &'a FrameIndex) -> Self {
        Self { body, idx: Cow::Borrowed(idx), limits: DecodeLimits::default() }
    }

    /// Replace the decode budget (the caller's byte ceiling for what any one decode call materializes).
    pub fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The authoritative container `Header` this reader decodes against.
    pub fn header(&self) -> &Header {
        &self.idx.header
    }

    /// Number of tiles (row-major); equals `header.tile_count()`.
    pub fn tile_count(&self) -> u32 {
        self.idx.tile_count()
    }

    /// The prediction kind of tile `t` ([`TILE_INTRA`]/[`TILE_PRED`]/[`TILE_SKIP`]).
    pub fn tile_kind(&self, t: u32) -> Option<u8> {
        self.idx.tile_kind(t)
    }

    /// The pixel rect `(x, y, tw, th)` of tile `t`.
    pub fn tile_rect(&self, t: u32) -> Option<(u32, u32, u32, u32)> {
        self.idx.tile_rect(t)
    }

    /// `(kind, blob)` of tile `t` for any kind that carries a blob (INTRA/PRED/BI), NOT yet verified (callers verify
    /// the parts they consume with [`Self::verify`]). THE one place an entry becomes a body slice.
    ///
    /// # Errors
    /// [`DecodeError::Request`] for a bad `t` or a SKIP tile (no blob), [`DecodeError::Corrupt`] for a range outside the
    /// body.
    fn blob(&self, t: u32) -> Result<(u8, &'a [u8]), DecodeError> {
        let e = *self.idx.entries.get(t as usize).ok_or(DecodeError::Request("tile index out of range"))?;
        if e.kind == TILE_SKIP {
            return Err(DecodeError::Request("SKIP tile has no blob"));
        }
        let blob = usize::try_from(e.off)
            .ok()
            .and_then(|start| Some(start..start.checked_add(usize::try_from(e.len).ok()?)?))
            .and_then(|r| self.body.get(r))
            .ok_or(DecodeError::Corrupt("tile blob outside the frame body"))?;
        Ok((e.kind, blob))
    }

    /// The codec's tile layout (an unopenable codec is its typed error).
    fn layout(&self) -> Result<TileLayout, DecodeError> {
        Ok(self.idx.profile.as_ref().map_err(|e| DecodeError::Codec(e.clone()))?.backend().layout())
    }

    /// Check tile `t`'s coded blobs against the index CRCs ([`exv_format::crc`]) under the reader's
    /// [`exv_format::Verify`] policy: a multi-component tile's one blob, or each WANTED channel sub-blob of a
    /// per-channel tile (`want(ci)`), so a channel subset hashes only what it decodes and a damaged channel fails alone.
    /// `blob` is [`Self::blob`]`(t)`. Returns the per-channel sub-blobs for a per-channel tile (empty for multi-component)
    /// so a caller that needs them does not split twice.
    ///
    /// # Errors
    /// [`DecodeError::Checksum`] for the first mismatching blob, [`DecodeError::Corrupt`] for a broken channel-length
    /// table, [`DecodeError::Codec`] for an unopenable codec.
    fn verify(&self, t: u32, blob: &'a [u8], want: impl Fn(usize) -> bool) -> Result<Vec<&'a [u8]>, DecodeError> {
        self.check(t, blob, want, self.limits.verify)
    }

    /// [`Self::verify`] under an explicit policy (the scan of [`Self::bad_blobs`] always verifies).
    fn check(&self, t: u32, blob: &'a [u8], want: impl Fn(usize) -> bool, policy: Verify) -> Result<Vec<&'a [u8]>, DecodeError> {
        const TABLE: DecodeError = DecodeError::Corrupt("tile channel-length table");
        let sums = self.idx.crcs(t);
        match self.layout()? {
            TileLayout::MultiComp => {
                if policy == Verify::On {
                    crc::check(t, None, blob, *sums.first().ok_or(TABLE)?)?;
                }
                Ok(Vec::new())
            }
            TileLayout::PerChannel => {
                let subs = crate::stream::split_tile_blob(blob, self.idx.dtypes.len()).ok_or(TABLE)?;
                if policy == Verify::On {
                    for (ci, (sub, &sum)) in subs.iter().zip(sums).enumerate().filter(|(ci, _)| want(*ci)) {
                        crc::check(t, u16::try_from(ci).ok(), sub, sum)?;
                    }
                }
                Ok(subs)
            }
        }
    }

    /// Every coded blob of the frame checked against its index CRC, whatever the reader's [`exv_format::Verify`]
    /// policy: the integrity scan of `exv verify` (no decode, hashing only, tiles in parallel). Returns `(tile, error)`
    /// in tile order: [`DecodeError::Checksum`] (the first bad blob of the tile) or the structural error that stopped
    /// the check (a broken channel-length table). SKIP tiles carry no blob and always pass.
    pub fn bad_blobs(&self) -> Vec<(u32, DecodeError)> {
        (0..self.tile_count())
            .into_par_iter()
            .filter(|&t| self.tile_kind(t) != Some(TILE_SKIP))
            .filter_map(|t| {
                let res = self.blob(t).and_then(|(_, b)| self.check(t, b, |_| true, Verify::On));
                res.err().map(|e| (t, e))
            })
            .collect()
    }

    /// Channel `ci`'s full-size reference plane, or the typed error a SKIP/PRED tile gets without one (a missing,
    /// wrong-resolution or short reference must not make the row gathers go out of bounds).
    fn ref_plane<'r>(&self, refs: Option<&RefPlanes<'r>>, ci: usize) -> Result<&'r [u8], DecodeError> {
        const NEED: DecodeError = DecodeError::Request("SKIP/PRED tile needs a full-size reference frame");
        let refs = refs.map(|r| &r.0).filter(|r| r.len() == self.idx.dtypes.len()).ok_or(NEED)?;
        let p = refs.get(ci).copied().flatten().ok_or(NEED)?;
        let d = *self.idx.dtypes.get(ci).ok_or(NEED)?;
        if self.idx.header.plane_len(d) == Some(p.len()) { Ok(p) } else { Err(NEED) }
    }

    /// Bytes ONE tile decode holds at its peak: the decoded tile (the EFFECTIVE tile `min(tile, image)` whose area
    /// `Header::read` caps) plus that tile's decoder working set. The working set is each codec's own account
    /// ([`TileCodec::work`], summed over channels because a tile decodes its channels in parallel) or, for
    /// SKIP/PRED tiles, the gathered reference + sum (2 planes) - whichever is larger. Without it the budget
    /// counted decoded pixels only and a lossy-wave tile really peaked at ~9.5x the number.
    ///
    /// `direct` = the whole-frame path, which decodes into plane ROWS ([`Dst::Rows`]): a multi-component (HTJ2K)
    /// tile then has no tile buffer at all (jph's lines go straight into the planes), while a per-channel codec still
    /// holds one decoded channel per concurrently decoding channel (the row-copy scratch), i.e. the same bytes.
    /// This is the per-lane charge of the decode window, so a stale over-count here directly idles cores.
    /// Public so every tile decoder outside this crate (the exv-gpu atlas CPU route) charges the SAME formula as
    /// [`Self::check_budget`] instead of re-deriving it.
    pub fn tile_cost(&self, direct: bool) -> Result<u64, DecodeError> {
        self.tile_cost_of(direct, Chans::All)
    }

    /// [`Self::tile_cost`] restricted to the `chans` a decoder really holds at once: the SAME formula with the element
    /// size and the per-channel working sets summed over the wanted channels only. The atlas per-channel CPU fallback
    /// (one channel per decode call) charges `Chans::Only(&[ci])` instead of a figure that counts every channel.
    pub fn tile_cost_of(&self, direct: bool, chans: Chans<'_>) -> Result<u64, DecodeError> {
        let h = &self.idx.header;
        let tile_px = (h.tile_w.min(h.width) as u64) * (h.tile_h.min(h.height) as u64);
        let wanted = || self.idx.dtypes.iter().enumerate().filter(move |&(ci, _)| chans.has(ci)).map(|(_, &d)| d);
        let esz: u64 = wanted().map(|d| d.size() as u64).sum();
        let prof = self.idx.profile.as_ref().map_err(|e| DecodeError::Codec(e.clone()))?;
        let (mut per_px, mut fixed) = (0u64, prof.backend().work_base(prof));
        for d in wanted() {
            let w = prof.backend().work(prof, d);
            per_px = per_px.saturating_add(w.per_px);
            fixed = fixed.saturating_add(w.fixed);
        }
        let work = per_px.max(if self.idx.temporal { 2 * esz } else { 0 }).saturating_mul(tile_px).saturating_add(fixed);
        let tile = if direct && prof.backend().layout() == TileLayout::MultiComp { 0 } else { esz.saturating_mul(tile_px) };
        Ok(tile.saturating_add(work))
    }

    /// Bytes of `out_px` output pixels over the `chans` channels.
    fn out_bytes(&self, out_px: u64, chans: Chans<'_>) -> u64 {
        let esz: u64 = self.idx.dtypes.iter().enumerate().filter(|&(ci, _)| chans.has(ci)).map(|(_, d)| d.size() as u64).sum();
        out_px.saturating_mul(esz)
    }

    /// Err([`DecodeError::Limit`]) unless the REAL peak of a decode fits the budget: `out_px` output pixels of the
    /// `chans` channels plus one tile's peak ([`Self::tile_cost`] of the `direct` row path or a tile-buffer path; all
    /// channels: an HTJ2K tile holds every component).
    fn check_budget(&self, out_px: u64, chans: Chans<'_>, direct: bool) -> Result<(), DecodeError> {
        self.limits.check(self.out_bytes(out_px, chans).saturating_add(self.tile_cost(direct)?))
    }

    /// Decode every tile into full-size planes (the whole-frame path of [`read_frame_tiles`]): the allocating front of
    /// [`Self::decode_all_into`] (all channels, fresh planes).
    ///
    /// Decodes STRAIGHT into the returned `Vec`s (no aligned buffer + copy): their base alignment is the allocator's
    /// (on Windows a >= 1 MiB block is `page + 64`, i.e. 64-aligned; smaller ones 16), which is why the reusable pooled path
    /// ([`Planes`], always 64-aligned) is the one hosts should run per frame.
    pub fn decode_all(&self, profile: &Profile, reference: Option<&[Vec<u8>]>) -> Result<Vec<Vec<u8>>, DecodeError> {
        let h = &self.idx.header;
        self.precheck(Chans::All)?;
        let mut out = tracing::info_span!("alloc_planes").in_scope(|| {
            self.idx.dtypes.iter().map(|&d| Ok(alloc_zeroed(plane_bytes(h.width as u64, h.height as u64, d)?)?)).collect::<Result<Vec<_>, DecodeError>>()
        })?;
        let planes = out.iter_mut().map(|p| Some(p.as_mut_slice())).collect();
        self.decode_rows(profile, reference.map(RefPlanes::from).as_ref(), Chans::All, planes)?;
        Ok(out)
    }

    /// Decode the `chans` channels of every tile into `out` (reused in place / from its pool, see [`Planes`]); the
    /// other slots become `None`. `reference` = the (MC) reference frame, required iff the body has SKIP/PRED tiles,
    /// and then only its `chans` planes are read.
    ///
    /// Tiles decode on `win` parallel lanes ([`par_lanes`]) STRAIGHT INTO the planes: a tile row owns the rows
    /// `[ty*tile_h, ..)` of every plane, so each plane splits safely into one band per tile row, and the first lane to
    /// reach a band splits its rows into per-tile column slices ([`Dst::Rows`]) that the band's tiles then take. No
    /// tile buffer, no scatter copy, no lock held while decoding; the result is byte-identical for any thread count.
    /// The window `win` is sized in BYTES from what the budget leaves beside the planes ([`exv_tile::par_window`]
    /// over [`Self::tile_cost`]) and capped at [`exv_tile::window_spare`] of them. The budget is checked before any
    /// plane is (re)allocated.
    ///
    /// # Errors
    /// As [`read_frame_tiles`], plus [`DecodeError::Request`] for a channel index out of range. On `Err` the contents
    /// of `out` are unspecified.
    pub fn decode_all_into(&self, profile: &Profile, reference: Option<&RefPlanes<'_>>, chans: Chans<'_>, out: &mut Planes) -> Result<(), DecodeError> {
        let h = &self.idx.header;
        self.precheck(chans)?;
        tracing::info_span!("fit_planes").in_scope(|| out.fit(&self.idx.dtypes, h.width as u64, h.height as u64, chans))?;
        self.decode_rows(profile, reference, chans, out.slices())
    }

    /// The checks of a whole-frame decode that run BEFORE any plane is (re)allocated: channel range and budget.
    fn precheck(&self, chans: Chans<'_>) -> Result<(), DecodeError> {
        if let Chans::Only(list) = chans
            && list.iter().any(|&ci| ci >= self.idx.dtypes.len())
        {
            return Err(DecodeError::Request("channel index out of range"));
        }
        let h = &self.idx.header;
        self.check_budget(h.width as u64 * h.height as u64, chans, true)
    }

    /// The one whole-frame decoder behind [`Self::decode_all`] / [`Self::decode_all_into`]: every tile into `planes`
    /// (one full-size plane per header channel, `None` = not decoded; sized by the caller after [`Self::precheck`]).
    fn decode_rows(&self, profile: &Profile, refs: Option<&RefPlanes<'_>>, chans: Chans<'_>, planes: Vec<Option<&mut [u8]>>) -> Result<(), DecodeError> {
        let h = &self.idx.header;
        let dtypes = &self.idx.dtypes;
        let out_px = h.width as u64 * h.height as u64;
        // The planes already cost `resident`; the window may transiently add `window_spare` on top (the same rule as the
        // encode side), and never more than the budget leaves. The spare is a share of the WHOLE frame, not of the
        // wanted channels: the lane count (speed) must not depend on which channels are viewed, or an RGB-of-5 decode
        // (resident 96 MiB at 4k -> spare 48 MiB) idled half the cores while the same machine accepts a full-frame
        // decode of 2.3x that. The charge per lane stays `tile_cost` (every channel: an HTJ2K subset still decodes the components in front of
        // the last wanted one) and the hard cap stays
        // `limits.max_alloc`, so nothing here can exceed the caller's budget.
        let resident = self.out_bytes(out_px, chans);
        let spare = exv_tile::window_spare(self.out_bytes(out_px, Chans::All));
        let win = exv_tile::par_window(self.limits.max_alloc.saturating_sub(resident).min(spare), self.tile_cost(true)?);
        let (tx_n, th, width) = (h.tiles_x() as usize, h.tile_h as usize, h.width as usize);
        // Per tile row: every channel's band (`None` = unwanted), split lazily into per-tile row tables.
        let mut bands: Vec<Mutex<Band<'_>>> = (0..h.tiles_y()).map(|_| Mutex::new(Band::Raw(Vec::with_capacity(dtypes.len())))).collect();
        for (slot, d) in planes.into_iter().zip(dtypes) {
            let mut chunks = slot.map(|p| p.chunks_mut((th * width * d.size()).max(1)));
            for band in &mut bands {
                if let Band::Raw(chans) = band.get_mut().unwrap_or_else(PoisonError::into_inner) {
                    chans.push(chunks.as_mut().and_then(Iterator::next));
                }
            }
        }
        // Which codecs write plane rows themselves: a multi-component (HTJ2K) decoder emits rows, so its tiles go
        // straight into the planes. A per-channel decoder (rANS / wavelet / EXV-RT) produces a contiguous plane, so it
        // decodes into the lane's tile buffer ([`LaneBuf`]: on loan from a bounded process-wide stash, so a steady-state
        // frame allocates none) and the wanted channels' rows are copied out (budgeted per lane by `tile_cost`; a
        // per-thread scratch instead would stay resident on every rayon worker after the frame).
        let rows_direct = profile.backend().layout() == TileLayout::MultiComp;
        par_lanes(self.tile_count() as u64, win, |t, buf: &mut LaneBuf| {
            let (tx, ty) = (t as usize % tx_n, t as usize / tx_n);
            let band = bands.get(ty).ok_or(DecodeError::Corrupt("tile row outside the plane"))?;
            let mut dst = band.lock().unwrap_or_else(PoisonError::into_inner).take(tx, h, dtypes)?;
            if rows_direct {
                return self.decode_tile_to(t as u32, profile, refs, &mut dst);
            }
            let r = self.rect(t as u32)?;
            buf.fit_some(dtypes, r.w, r.h, |ci| dst.get(ci).is_some_and(|d| !d.is_skip()))?;
            let mut tmp: Vec<Dst<'_>> =
                buf.planes_mut().into_iter().zip(&dst).map(|(p, d)| if d.is_skip() { Dst::Skip } else { Dst::Plane(p) }).collect();
            self.decode_tile_to(t as u32, profile, refs, &mut tmp)?;
            drop(tmp);
            let _s = tracing::info_span!("rows_copy", t).entered();
            for (ci, (d, dt)) in dst.iter_mut().zip(dtypes).enumerate() {
                let row = (r.w as usize * dt.size()).max(1);
                if let Dst::Rows(rows) = d {
                    for (out, src) in rows.iter_mut().zip(buf.plane(ci).chunks_exact(row)) {
                        out.copy_from_slice(src);
                    }
                }
            }
            Ok(())
        })
    }

    /// Decode tile `t` into per-channel tile-rect planes (`tw*th*esz` each, exactly — callers need not
    /// re-check). INTRA decodes standalone; SKIP copies the reference's co-located tile; PRED adds the
    /// residual back onto it. `reference` = the (MC) reference frame's full-size planes, required iff
    /// tile `t` is SKIP/PRED.
    ///
    /// # Errors
    /// [`DecodeError::Request`] for a bad `t` or a SKIP/PRED tile with no (or a wrong-size) reference,
    /// [`DecodeError::Unsupported`] for a BI tile (not decodable before 1d), [`DecodeError::Limit`] over
    /// budget, otherwise the blob's [`DecodeError::Corrupt`].
    pub fn decode_tile(&self, t: u32, profile: &Profile, reference: Option<&[Vec<u8>]>) -> Result<Vec<Vec<u8>>, DecodeError> {
        let r = self.rect(t)?;
        self.check_budget(0, Chans::All, false)?;
        // Straight into the returned `Vec`s (an aligned `TileBuf` would need a copy out).
        let mut planes = crate::alloc_planes(&self.idx.dtypes, r.w, r.h)?;
        let mut dst: Vec<Dst<'_>> = planes.iter_mut().map(|p| Dst::Plane(p)).collect();
        self.decode_tile_to(t, profile, reference.map(RefPlanes::from).as_ref(), &mut dst)?;
        drop(dst);
        Ok(planes)
    }

    /// [`Self::decode_tile`] into a recycled [`TileBuf`] (sized here to the tile, so a tile allocates nothing of its
    /// own once the buffer is warm).
    ///
    /// # Errors
    /// As [`Self::decode_tile`]; on `Err` the contents of `buf` are unspecified.
    pub fn decode_tile_into(
        &self,
        t: u32,
        profile: &Profile,
        reference: Option<&[Vec<u8>]>,
        buf: &mut TileBuf,
    ) -> Result<(), DecodeError> {
        let r = self.rect(t)?;
        self.check_budget(0, Chans::All, false)?; // (the whole-frame path checked its budget once, up front)
        buf.fit(&self.idx.dtypes, r.w, r.h)?;
        let mut dst: Vec<Dst<'_>> = buf.planes_mut().into_iter().map(Dst::Plane).collect();
        self.decode_tile_to(t, profile, reference.map(RefPlanes::from).as_ref(), &mut dst)
    }

    /// Tile `t`'s pixel rect, or a typed error for a bad `t`.
    fn rect(&self, t: u32) -> Result<Rect, DecodeError> {
        let h = &self.idx.header;
        if t >= self.tile_count() {
            return Err(DecodeError::Request("tile index out of range"));
        }
        Ok(Rect::of_tile(h, t % h.tiles_x(), t / h.tiles_x()))
    }

    /// THE tile decoder of every frame path: tile `t` into one [`Dst`] per channel (tile planes or plane rows; skipped
    /// channels are not decoded). INTRA decodes straight into `dst`; PRED decodes the residual into `dst` and adds the
    /// reference's co-located rows in place; SKIP copies them. Only the wanted channels' reference planes are read.
    fn decode_tile_to(&self, t: u32, profile: &Profile, refs: Option<&RefPlanes<'_>>, dst: &mut [Dst<'_>]) -> Result<(), DecodeError> {
        let kind = self.idx.entries.get(t as usize).ok_or(DecodeError::Request("tile index out of range"))?.kind;
        let h = &self.idx.header;
        let r = self.rect(t)?;
        // Verified before the codec runs: only the channels this call decodes (a SKIP/PRED tile's reference rows are
        // already-verified decoded planes).
        let want: Vec<bool> = dst.iter().map(|d| !d.is_skip()).collect();
        let blob = || -> Result<&[u8], DecodeError> {
            let (_, b) = self.blob(t)?;
            self.verify(t, b, |ci| want.get(ci).copied().unwrap_or(false))?;
            Ok(b)
        };
        let dtypes = &self.idx.dtypes;
        // `(channel, reference plane)` of every wanted channel; resolved BEFORE decoding so a missing reference fails fast.
        let wanted_refs = || -> Result<Vec<(usize, &[u8])>, DecodeError> {
            dst.iter().enumerate().filter(|(_, d)| !d.is_skip()).map(|(ci, _)| Ok((ci, self.ref_plane(refs, ci)?))).collect()
        };
        match kind {
            TILE_INTRA => decode_tile_dst(blob()?, dtypes, r.w, r.h, profile, dst),
            TILE_PRED | TILE_SKIP => {
                let rp = wanted_refs()?;
                if kind == TILE_PRED {
                    decode_tile_dst(blob()?, dtypes, r.w, r.h, profile, dst)?;
                } else if dst.len() != dtypes.len() || !dst.iter().zip(dtypes).all(|(d, dt)| d.fits(r.w as usize, r.h as usize, dt.size())) {
                    return Err(DecodeError::Request("tile destinations do not match the channels / tile shape"));
                }
                for (ci, plane) in rp {
                    let (out, &dt) = (dst.get_mut(ci).ok_or(DecodeError::Request("channel index"))?, dtypes.get(ci).ok_or(DecodeError::Request("channel index"))?);
                    let row = r.w as usize * dt.size();
                    // `rect_rows` yields exactly `r.h` rows of a validated reference (`ref_plane`).
                    for (y, prev) in rect_rows(plane, h.width, r, dt.size()).enumerate() {
                        let cur = out.row(y, row).ok_or(DecodeError::Request("tile destination row"))?;
                        if kind == TILE_PRED {
                            temporal::undelta_assign(cur, prev, dt)?;
                        } else {
                            cur.copy_from_slice(prev);
                        }
                    }
                }
                Ok(())
            }
            _ => Err(DecodeError::Unsupported("bidirectional (BI) tiles")),
        }
    }

    /// The raw self-describing tile blob for tile `t` EXACTLY as written — the unit the GPU decode path
    /// (jph-wgpu / rANS kernels) consumes WITHOUT a CPU decode. **INTRA tiles only**: a PRED blob is a
    /// residual (`cur − ref`), not a picture, so handing it to a consumer that treats blobs as pixels
    /// would display garbage; SKIP has no blob; BI is undecodable. The bytes are CRC-verified under the reader's
    /// [`exv_format::Verify`] policy (the GPU path's integrity check happens here, on the CPU, before any upload).
    /// The C0 raw-blob seam shared by the still + frame tile readers.
    ///
    /// # Errors
    /// [`DecodeError::Request`] for a bad `t` or a non-INTRA tile, [`DecodeError::Corrupt`] for a range outside the
    /// body, [`DecodeError::Checksum`] for a blob that does not match its CRC.
    pub fn tile_blob(&self, t: u32) -> Result<&'a [u8], DecodeError> {
        let blob = self.intra(t)?;
        self.verify(t, blob, |_| true)?;
        Ok(blob)
    }

    /// The unverified blob of INTRA tile `t` (the accessors' common front).
    fn intra(&self, t: u32) -> Result<&'a [u8], DecodeError> {
        match self.blob(t)? {
            (TILE_INTRA, blob) => Ok(blob),
            _ => Err(DecodeError::Request("tile is not INTRA (its blob is not a picture)")),
        }
    }

    /// The per-channel sub-blobs of INTRA tile `t` (header channel order), all verified, split once: what a per-tile
    /// consumer of every channel (the GPU job planners) calls. Per-channel codecs (rANS / wavelet / EXV-RT) only — an
    /// HTJ2K tile is ONE multi-component codestream with no per-channel slice.
    ///
    /// # Errors
    /// As [`Self::tile_blob`], plus [`DecodeError::Codec`] for an unopenable codec, [`DecodeError::Request`] for a
    /// multi-component (HTJ2K) layout and [`DecodeError::Corrupt`] for a broken channel-length table.
    pub fn tile_chans(&self, t: u32) -> Result<Vec<&'a [u8]>, DecodeError> {
        self.chans_where(t, |_| true)
    }

    /// The raw per-channel sub-blob for tile `t`, channel `ci`; only THAT sub-blob is verified (a damaged neighbour
    /// channel does not fail it).
    ///
    /// # Errors
    /// As [`Self::tile_chans`], plus [`DecodeError::Request`] for an out-of-range `ci`.
    pub fn tile_channel_blob(&self, t: u32, ci: usize) -> Result<&'a [u8], DecodeError> {
        self.chans_where(t, |c| c == ci)?.get(ci).copied().ok_or(DecodeError::Request("channel index out of range"))
    }

    /// The sub-blobs of per-channel INTRA tile `t`, the `want`ed ones verified.
    fn chans_where(&self, t: u32, want: impl Fn(usize) -> bool) -> Result<Vec<&'a [u8]>, DecodeError> {
        if self.layout()? == TileLayout::MultiComp {
            return Err(DecodeError::Request("multi-component (HTJ2K) tile has no per-channel sub-blobs"));
        }
        let blob = self.intra(t)?;
        self.verify(t, blob, want)
    }

    /// The codec profile reconstructed from the container `Header` (codec id/params) — how the tiles
    /// decode; resolved once at [`Self::open`]. The typed [`OpenError`] says why a codec cannot be opened (unknown /
    /// vendor / not built / bad params). The frame body is header-less; the profile lives in the Header.
    pub fn profile(&self) -> Result<Profile, OpenError> {
        self.idx.profile.clone()
    }

    /// Decode ONE channel `ci` of tile `t` into freshly allocated tile-rect bytes (`tw*th*esz`). The
    /// frame-body analog of the old `stream::ExvTileReader::decode_tile_channel` — STILL/INTRA use
    /// (`reference == None`): HTJ2K tiles are one multi-component codestream (no per-channel slice) so
    /// the whole tile decodes once and channel `ci` is picked; per-channel codecs split the blob and
    /// decode only `ci`.
    ///
    /// # Errors
    /// [`DecodeError::Request`] for an out-of-range `t`/`ci` or a SKIP/PRED tile (they need a reference or
    /// are undecodable standalone — use [`Self::decode_tile`]; a PRED residual must never be returned as
    /// pixels), [`DecodeError::Unsupported`] for an unknown codec, [`DecodeError::Limit`] over budget,
    /// otherwise the blob's [`DecodeError::Corrupt`].
    pub fn decode_tile_channel(&self, t: u32, ci: usize) -> Result<Vec<u8>, DecodeError> {
        let profile = self.profile()?;
        let dtype = self.idx.header.channels.get(ci).ok_or(DecodeError::Request("channel index out of range"))?.dtype;
        // HTJ2K: no per-channel sub-blob — decode the whole tile once and pick `ci`.
        if profile.backend().layout() == TileLayout::MultiComp {
            if self.tile_kind(t) != Some(TILE_INTRA) {
                return Err(DecodeError::Request("tile is not INTRA (needs a reference)"));
            }
            return self.decode_tile(t, &profile, None)?.into_iter().nth(ci).ok_or(DecodeError::Request("channel index out of range"));
        }
        let (_, _, tw, th) = self.tile_rect(t).ok_or(DecodeError::Request("tile index out of range"))?;
        self.check_budget(0, Chans::All, false)?;
        let sub = self.tile_channel_blob(t, ci)?;
        // `decode_channel` guarantees the exact `tw*th*esz` length.
        decode_channel(sub, dtype, tw, th, &profile)
    }

    /// Decode channel `ci` of tile `t` at reduced resolution `level` (lossless or 9/7 HTJ2K INTRA tiles
    /// only, [`crate::Htj2kMode::lod_wavelet`]), returning `(reduced_bytes, reduced_w, reduced_h)` — J2K resolution scalability, the
    /// frame-body analog of the old `stream::ExvTileReader::decode_tile_channel_level`. `level == 0`
    /// reproduces [`Self::decode_tile_channel`] at the tile's full extent.
    ///
    /// # Errors
    /// [`DecodeError::Request`] when `t`/`ci` is out of range or the level exceeds the codestream's
    /// decomposition count, [`DecodeError::Unsupported`] when the profile has no HTJ2K LOD (not HTJ2K, or hdr codes),
    /// [`DecodeError::Limit`] over budget, [`DecodeError::Corrupt`] when the blob is not an HTJ2K codestream
    /// (or the tile is not INTRA) or its extent disagrees with the tile rect. Cfg-gated on
    /// `htj2k-backend` (the free fn it calls is optional).
    #[cfg(feature = "htj2k-backend")]
    pub fn decode_tile_channel_level(&self, t: u32, ci: usize, level: u32) -> Result<(Vec<u8>, u32, u32), DecodeError> {
        use crate::{BlobKind, Transform, classify_blob, htj2k};
        let wav = match self.profile()?.transform {
            Transform::Htj2k(mode) => mode.lod_wavelet(),
            _ => None,
        }
        .ok_or(DecodeError::Unsupported("LOD decode needs lossless or 9/7 HTJ2K"))?;
        if ci >= self.idx.dtypes.len() {
            return Err(DecodeError::Request("channel index out of range"));
        }
        let (_, _, tw, th) = self.tile_rect(t).ok_or(DecodeError::Request("tile index out of range"))?;
        self.check_budget(0, Chans::All, false)?;
        let blob = self.tile_blob(t)?;
        match classify_blob(blob) {
            Some(BlobKind::Htj2k(payload)) => {
                let (planes, rw, rh) = htj2k::decode_htj2k_multi_level(payload, &self.idx.dtypes, tw, th, level, wav)?;
                let plane = planes.into_iter().nth(ci).ok_or(DecodeError::Request("channel index out of range"))?;
                Ok((plane, rw, rh))
            }
            _ => Err(DecodeError::Corrupt("tile is not an HTJ2K codestream")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use exv_format::Channel;

    /// `par_lanes` visits every tile exactly once, keeps at most `win` scratches, and reports the LOWEST failing tile
    /// whatever the lane count or timing.
    #[test]
    fn par_lanes_covers_all_and_reports_lowest_error() {
        use std::sync::atomic::{AtomicU64, Ordering};
        for win in [1u64, 2, 7, 64, 1000] {
            let seen = AtomicU64::new(0);
            par_lanes::<u8>(50, win, |t, _| {
                seen.fetch_add(1 << t, Ordering::Relaxed);
                Ok(())
            })
            .expect("no failure");
            assert_eq!(seen.load(Ordering::Relaxed), (1u64 << 50) - 1, "win {win}: each tile exactly once");
            for _ in 0..20 {
                let err = par_lanes::<u8>(50, win, |t, _| {
                    if t == 13 || t == 30 { Err(DecodeError::Corrupt(if t == 13 { "low" } else { "high" })) } else { Ok(()) }
                });
                assert_eq!(err, Err(DecodeError::Corrupt("low")), "win {win}");
            }
        }
        par_lanes::<u8>(0, 4, |_, _| Err(DecodeError::Request("never"))).expect("no tiles, no work");
    }

    /// Generous budget for tests that are not about the budget.
    const LIM: DecodeLimits = DecodeLimits::new(1 << 30);
    /// Encode-window budget for tests that are not about it.
    const BUDGET: u64 = 1 << 30;

    fn header(w: u32, h: u32, tile: u32) -> Header {
        Header {
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
        }
    }

    fn planes(hd: &Header, seed: u32) -> Vec<Vec<u8>> {
        let n = (hd.width * hd.height) as usize;
        let r: Vec<u8> = (0..n).flat_map(|i| (((i as u32 * 11 + seed) & 0xffff) as u16).to_le_bytes()).collect();
        let z: Vec<u8> = (0..n).flat_map(|i| ((i as u32 + seed) as f32).to_le_bytes()).collect();
        vec![r, z]
    }

    /// All-INTRA frame (no reference) round-trips bit-exact.
    #[test]
    fn intra_roundtrip() {
        for tile in [64u32, 96] {
            let hd = header(96, 72, tile);
            let p = planes(&hd, 3);
            let prof = Profile::lossless_fast();
            let body = write_frame_tiles(&p, None, &hd, &prof).expect("encode");
            let dec = read_frame_tiles(&body, &hd, &prof, None, &LIM).expect("decode");
            assert_eq!(dec, p, "tile={tile} INTRA round-trip");
        }
    }

    /// A frame IDENTICAL to the reference → ALL tiles SKIP → tiny body + reconstructs from ref.
    #[test]
    fn skip_identical_frame() {
        let hd = header(80, 48, 64);
        let prof = Profile::lossless_fast();
        let p = planes(&hd, 5);
        let intra = write_frame_tiles(&p, None, &hd, &prof).expect("encode");
        let skip = write_frame_tiles(&p, Some(&p), &hd, &prof).expect("encode");
        assert!(skip.len() < intra.len() / 4, "all-SKIP body {} << INTRA {}", skip.len(), intra.len());
        assert_eq!(read_frame_tiles(&skip, &hd, &prof, Some(&p), &LIM).expect("dec"), p);
    }

    /// PRED: a frame that is the reference + a tiny per-element bit delta (near-constant residual) →
    /// PRED tiles chosen, body MUCH smaller than all-INTRA, and round-trips bit-exact.
    #[test]
    fn pred_smaller_and_exact() {
        let hd = header(256, 192, 64);
        let prof = Profile::lossless_fast();
        let refp = planes(&hd, 1);
        // cur = ref with +1 added to every element's integer bit-pattern (residual is all 1s → compresses tiny).
        let cur: Vec<Vec<u8>> = refp
            .iter()
            .zip(hd.channels.iter())
            .map(|(plane, ch)| {
                let esz = ch.dtype.size();
                let mut out = plane.clone();
                for e in out.chunks_mut(esz) {
                    let mut v = 0u32;
                    for (i, &b) in e.iter().enumerate() {
                        v |= (b as u32) << (8 * i);
                    }
                    v = v.wrapping_add(1);
                    for (i, b) in e.iter_mut().enumerate() {
                        *b = (v >> (8 * i)) as u8;
                    }
                }
                out
            })
            .collect();
        let intra = write_frame_tiles(&cur, None, &hd, &prof).expect("encode");
        let pred = write_frame_tiles(&cur, Some(&refp), &hd, &prof).expect("encode");
        assert!(pred.len() < intra.len(), "PRED {} should be < INTRA {}", pred.len(), intra.len());
        assert_eq!(read_frame_tiles(&pred, &hd, &prof, Some(&refp), &LIM).expect("dec"), cur, "PRED round-trip");
    }

    /// A scene-cut frame (totally unlike the reference) still round-trips (tiles fall back to INTRA).
    #[test]
    fn scene_cut_falls_back_to_intra() {
        let hd = header(96, 64, 64);
        let prof = Profile::lossless_fast();
        let refp = planes(&hd, 1);
        let cur = planes(&hd, 9999); // unrelated content
        let body = write_frame_tiles(&cur, Some(&refp), &hd, &prof).expect("encode");
        assert_eq!(read_frame_tiles(&body, &hd, &prof, Some(&refp), &LIM).expect("dec"), cur);
    }

    /// SKIP/PRED without a reference → None (no panic); truncations never panic.
    #[test]
    fn no_reference_or_truncation_is_safe() {
        let hd = header(64, 64, 64);
        let prof = Profile::lossless_fast();
        let p = planes(&hd, 2);
        let skip = write_frame_tiles(&p, Some(&p), &hd, &prof).expect("encode");
        assert!(read_frame_tiles(&skip, &hd, &prof, None, &LIM).is_err(), "SKIP needs a reference");
        for cut in 0..skip.len() {
            let _ = read_frame_tiles(&skip[..cut], &hd, &prof, Some(&p), &LIM);
        }
    }

    /// FrameReader decodes tiles individually, matching the full-frame decode (region / partial decode).
    #[test]
    fn frame_reader_matches_full() {
        let hd = header(160, 96, 64); // 3×2 tiles
        let prof = Profile::lossless_fast();
        let refp = planes(&hd, 1);
        // Perturb part of channel 0 → a mix of INTRA/PRED/SKIP tiles.
        let mut cur = refp.clone();
        for y in 0..hd.height {
            for x in 0..70u32 {
                let i = (y * hd.width + x) as usize * 2;
                cur[0][i] ^= 0x5C;
            }
        }
        let body = write_frame_tiles(&cur, Some(&refp), &hd, &prof).expect("encode");
        let full = read_frame_tiles(&body, &hd, &prof, Some(&refp), &LIM).expect("full");
        let reader = FrameReader::open(&body, hd.clone()).expect("open");
        assert_eq!(reader.tile_count(), hd.tile_count() as u32);
        let mut region: Vec<Vec<u8>> =
            hd.channels.iter().map(|c| vec![0u8; hd.plane_len(c.dtype).unwrap()]).collect();
        for t in 0..reader.tile_count() {
            let (x, y, tw, _) = reader.tile_rect(t).unwrap();
            let tiles = reader.decode_tile(t, &prof, Some(&refp)).expect("tile");
            for (ci, plane) in region.iter_mut().enumerate() {
                let row = tw as usize * hd.channels[ci].dtype.size();
                for (yy, src) in tiles[ci].chunks_exact(row).enumerate() {
                    let at = ((y as usize + yy) * hd.width as usize + x as usize) * hd.channels[ci].dtype.size();
                    plane[at..at + row].copy_from_slice(src);
                }
            }
        }
        assert_eq!(region, full, "per-tile reader == full frame decode");
        // A still (all-INTRA) frame's tiles decode with NO reference.
        let still = write_frame_tiles(&cur, None, &hd, &prof).expect("encode");
        let sr = FrameReader::open(&still, hd.clone()).expect("open still");
        assert!((0..sr.tile_count()).all(|t| sr.decode_tile(t, &prof, None).is_ok()), "INTRA tiles need no ref");
    }

    /// The streaming, windowed all-INTRA frame-body writer produces BYTE-IDENTICAL output to the
    /// in-RAM [`write_frame_tiles`] (`reference == None`) — so a streamed still decodes exactly like an
    /// in-RAM one. Also proves the per-window producer yields the right tile order.
    #[test]
    fn intra_streaming_body_matches_in_ram() {
        let hd = header(200, 130, 64); // 4×3 tiles incl. edge tiles (not a multiple of 64)
        let prof = Profile::lossless_fast();
        let p = planes(&hd, 7);
        let in_ram = write_frame_tiles(&p, None, &hd, &prof).expect("encode");

        // A producer that gathers each tile's per-channel bytes on demand (the out-of-core contract).
        let dtypes: Vec<Dtype> = hd.channels.iter().map(|c| c.dtype).collect();
        let tx_n = hd.tiles_x() as u64;
        let mut streamed = Vec::new();
        let n = write_intra_frame_body_windowed(&hd, &prof, BUDGET, &mut streamed, |t0, t1| {
            let mut window = Vec::new();
            for t in t0..t1 {
                let (tx, ty) = ((t % tx_n) as u32, (t / tx_n) as u32);
                let (x, y, tw, th) = hd.tile_rect(tx, ty);
                let chans: Vec<Vec<u8>> =
                    (0..hd.channels.len()).map(|ci| gather(&p[ci], hd.width, Rect { x, y, w: tw, h: th }, dtypes[ci].size())).collect();
                window.push(chans);
            }
            Ok(window)
        })
        .expect("stream");
        assert_eq!(n as usize, streamed.len());
        assert_eq!(streamed, in_ram, "streamed frame body == in-RAM write_frame_tiles(None)");
        // And it decodes back to the source planes.
        assert_eq!(read_frame_tiles(&streamed, &hd, &prof, None, &LIM).expect("dec"), p);
    }

    /// A tiny hostile body claiming ~2^32 tiles is rejected BEFORE any tile-count-sized allocation
    /// (`with_capacity(n)` used to abort the process); truncations never panic.
    #[test]
    fn forged_tile_count_is_rejected_before_alloc() {
        let hd = header(65535, 65535, 1); // tile_count = 65535² (fits u32), declared by the hostile header
        let mut body = (hd.tile_count() as u32).to_le_bytes().to_vec();
        body.extend_from_slice(&0u64.to_le_bytes()); // index_off = 0
        assert!(matches!(FrameReader::open(&body, hd.clone()), Err(DecodeError::Corrupt(_))));
        for cut in 0..body.len() {
            assert!(matches!(FrameReader::open(&body[..cut], hd.clone()), Err(DecodeError::Corrupt(_))));
        }
    }

    /// A valid 1-tile index over a 65535² image: the planes it would materialize (~25 GB) exceed the
    /// caller's budget, so every whole-image / region decode is a typed `Limit` error before any plane is
    /// allocated — never an allocation abort. The SAME file decodes under a large enough budget (the
    /// budget is the caller's policy, not a guess about compression ratios).
    #[test]
    fn over_budget_decode_is_a_typed_error() {
        let hd = header(65535, 65535, 65535);
        // One INTRA tile with an empty blob: n_tiles=1, n_crc=2 (two per-channel blobs), kind, off=0, len=0, crcs, index_off=0.
        let one_empty_tile = || {
            let mut b = 1u32.to_le_bytes().to_vec();
            b.extend_from_slice(&2u16.to_le_bytes());
            b.push(TILE_INTRA);
            b.extend_from_slice(&[0u8; 8 + 4 + 2 * 4 + 8]);
            b
        };
        let body = one_empty_tile();
        let prof = Profile::lossless_fast();
        let small = DecodeLimits::new(1 << 20);
        assert!(matches!(read_frame_tiles(&body, &hd, &prof, None, &small), Err(DecodeError::Limit { .. })));
        let reader = FrameReader::open(&body, hd.clone()).expect("index is well-formed").with_limits(small);
        assert!(matches!(reader.decode_tile(0, &prof, None), Err(DecodeError::Limit { .. })), "one tile is over budget too");
        // A budget that admits the planes gets past the check (this body's tile blob is empty, so the decode
        // then fails as corrupt — NOT as a limit, and without ever allocating 25 GB: planes are only
        // requested after the budget passes, so use a size the machine can actually serve here).
        let ok = header(64, 64, 64);
        let tiny = one_empty_tile();
        assert!(matches!(read_frame_tiles(&tiny, &ok, &prof, None, &LIM), Err(DecodeError::Corrupt(_))));
    }

    /// `tile_cost_of` is the ONE formula: `All` == `tile_cost`, and for FAST (linear, no fixed part) a channel subset
    /// costs exactly its own bytes: per-channel peaks add up to the whole-tile figure.
    #[test]
    fn tile_cost_of_subsets_share_one_formula() {
        let hd = header(96, 72, 64);
        let prof = Profile::lossless_fast();
        let body = write_frame_tiles(&planes(&hd, 1), None, &hd, &prof).expect("encode");
        let rd = FrameReader::open(&body, hd.clone()).expect("open");
        let px = 64 * 64;
        let all = rd.tile_cost(false).expect("cost");
        assert_eq!(rd.tile_cost_of(false, Chans::All), Ok(all));
        // FAST: decoded tile esz B/px + working set esz B/px.
        assert_eq!(rd.tile_cost_of(false, Chans::Only(&[0])), Ok(2 * 2 * px), "F16 channel alone");
        assert_eq!(rd.tile_cost_of(false, Chans::Only(&[1])), Ok(2 * 4 * px), "F32 channel alone");
        assert_eq!(all, 2 * 6 * px);
    }

    /// A budget fits exactly `planes + one tile + the codec's tile working set`: one byte less is a `Limit`, the exact figure decodes.
    #[test]
    fn budget_counts_planes_plus_one_tile() {
        let hd = header(96, 72, 64);
        let prof = Profile::lossless_fast();
        let p = planes(&hd, 3);
        let body = write_frame_tiles(&p, None, &hd, &prof).expect("encode");
        let esz: u64 = hd.channels.iter().map(|c| c.dtype.size() as u64).sum(); // 6 B/px over both channels
        // planes + one tile, plus the codec's working set (FAST: one extra plane's bytes per tile pixel).
        let need = esz * (96 * 72 + 64 * 64) + esz * (64 * 64);
        assert!(matches!(
            read_frame_tiles(&body, &hd, &prof, None, &DecodeLimits::new(need - 1)),
            Err(DecodeError::Limit { .. })
        ));
        assert_eq!(read_frame_tiles(&body, &hd, &prof, None, &DecodeLimits::new(need)), Ok(p));
    }

    /// B1/B10: a PRED tile is a residual — it must never come out as pixels through the standalone
    /// accessors (blob / channel blob / channel decode / lenient region), and SKIP/PRED against a
    /// wrong-size reference is `None`, not an out-of-bounds panic.
    #[test]
    fn pred_tiles_are_never_returned_as_pixels() {
        let hd = header(256, 192, 64);
        let prof = Profile::lossless_fast();
        let refp = planes(&hd, 1);
        let cur: Vec<Vec<u8>> = refp
            .iter()
            .map(|p| p.iter().map(|b| b.wrapping_add(1)).collect()) // every byte +1: a tiny, compressible residual
            .collect();
        let body = write_frame_tiles(&cur, Some(&refp), &hd, &prof).expect("write");
        let reader = FrameReader::open(&body, hd.clone()).expect("open");
        let t = (0..reader.tile_count()).find(|&t| reader.tile_kind(t) == Some(TILE_PRED)).expect("a PRED tile");
        assert!(matches!(reader.tile_blob(t), Err(DecodeError::Request(_))));
        assert!(matches!(reader.tile_channel_blob(t, 0), Err(DecodeError::Request(_))));
        assert!(reader.decode_tile_channel(t, 0).is_err());
        assert!(matches!(reader.decode_tile(t, &prof, None), Err(DecodeError::Request(_))), "PRED needs a reference");
        assert!(reader.decode_tile(t, &prof, Some(&refp)).is_ok());
        // B10: wrong-resolution / short reference planes.
        assert!(reader.decode_tile(t, &prof, Some(&[vec![0u8; 3], vec![0u8; 3]])).is_err());
        assert!(reader.decode_tile(t, &prof, Some(&refp[..1])).is_err());
    }

    /// Writers reject malformed input with a typed error instead of panicking in a rayon worker or
    /// writing a file the reader then rejects.
    #[test]
    fn writers_reject_malformed_input() {
        let hd = header(96, 72, 64);
        let prof = Profile::lossless_fast();
        let p = planes(&hd, 1);
        assert!(matches!(write_frame_tiles(&p[..1], None, &hd, &prof), Err(CodecError::Dims(_))));
        let mut bad = p.clone();
        bad[0].pop();
        assert!(matches!(write_frame_tiles(&bad, None, &hd, &prof), Err(CodecError::Dims(_))));
        assert!(matches!(write_frame_tiles(&p, Some(&bad), &hd, &prof), Err(CodecError::Dims(_))));
        let mut sink = Vec::new();
        let kind = |r: io::Result<u64>| r.expect_err("must fail").kind();
        // too few tiles / too few channels / a short plane
        assert_eq!(kind(write_intra_frame_body_windowed(&hd, &prof, BUDGET, &mut sink, |_, _| Ok(Vec::new()))), io::ErrorKind::InvalidData);
        let one_chan = |t0: u64, t1: u64| Ok((t0..t1).map(|_| vec![vec![0u8; 4]]).collect());
        assert_eq!(kind(write_intra_frame_body_windowed(&hd, &prof, BUDGET, &mut sink, one_chan)), io::ErrorKind::InvalidData);
        let short = |t0: u64, t1: u64| Ok((t0..t1).map(|_| vec![vec![0u8; 4], vec![0u8; 4]]).collect());
        assert_eq!(kind(write_intra_frame_body_windowed(&hd, &prof, BUDGET, &mut sink, short)), io::ErrorKind::InvalidData);
    }

    /// `decode_all_into` == `decode_all` for every profile, channel subset and frame kind (INTRA / PRED+SKIP), into
    /// REUSED planes (the second decode of a frame allocates nothing: same buffers, pool hits only); unselected
    /// channels are `None`, never zero-filled stand-ins.
    #[test]
    fn decode_all_into_reuses_planes_and_decodes_subsets() {
        let mut profs = vec![Profile::lossless_fast(), Profile::lossless_wave(), Profile::lossy_wave(3)];
        #[cfg(feature = "htj2k-backend")]
        profs.extend([Profile::htj2k(), Profile::htj2k_lossy(9)]);
        for prof in profs {
            let mut hd = header(200, 130, 64); // 4x3 tiles, edge tiles on both axes
            hd.codec_id = prof.codec_id().0;
            hd.codec_params = prof.params();
            hd.channels.push(Channel { name: "id".into(), dtype: Dtype::U32 });
            let n = (hd.width * hd.height) as usize;
            let mut cur = planes(&hd, 4);
            cur.push((0..n as u32).flat_map(|i| (i * 7).to_le_bytes()).collect());
            let mut refp = cur.clone();
            for b in refp[0].iter_mut().take(3000) {
                *b ^= 0x21; // a few tiles differ: INTRA/PRED, the rest SKIP
            }
            let lossless = prof.is_lossless();
            let reference = lossless.then_some(refp.as_slice());
            let body = write_frame_tiles(&cur, reference, &hd, &prof).expect("encode");
            let rd = FrameReader::open(&body, hd.clone()).expect("open").with_limits(LIM);
            let full = rd.decode_all(&prof, reference).expect("decode_all");
            let pool = PlanePool::new(exv_tile::Budget::new(1 << 30));
            let mut refs = Planes::default();
            if lossless {
                rd.decode_all_into(&prof, None, Chans::All, &mut refs).expect_err("SKIP/PRED without a reference");
                let ref_body = write_frame_tiles(&refp, None, &hd, &prof).expect("encode ref");
                FrameReader::open(&ref_body, hd.clone()).expect("open").decode_all_into(&prof, None, Chans::All, &mut refs).expect("ref");
            }
            let rr = lossless.then(|| RefPlanes::from(&refs));
            let rr = rr.as_ref();
            for chans in [Chans::All, Chans::Only(&[0]), Chans::Only(&[2, 1]), Chans::Only(&[])] {
                let mut out = Planes::pooled(pool.clone());
                rd.decode_all_into(&prof, rr, chans, &mut out).expect("decode_all_into");
                let ptrs: Vec<_> = (0..out.len()).map(|ci| out.get(ci).map(<[u8]>::as_ptr)).collect();
                assert!(ptrs.iter().flatten().all(|p| (*p as usize).is_multiple_of(64)), "pooled planes are 64-byte aligned");
                let mut tb = TileBuf::default();
                tb.fit(&rd.idx.dtypes, 64, 64).expect("fit");
                assert!((0..out.len()).all(|ci| (tb.plane(ci).as_ptr() as usize).is_multiple_of(64)), "lane TileBuf planes are 64-byte aligned");
                let misses = pool.stats().misses;
                rd.decode_all_into(&prof, rr, chans, &mut out).expect("again");
                assert_eq!(pool.stats().misses, misses, "{:?} {chans:?}: a re-decode allocates nothing", prof.transform);
                for (ci, f) in full.iter().enumerate() {
                    assert_eq!(out.get(ci), chans.has(ci).then_some(f.as_slice()), "{:?} {chans:?} ch {ci}", prof.transform);
                    assert_eq!(out.get(ci).map(<[u8]>::as_ptr), ptrs[ci], "decoded in place");
                }
            }
            assert!(Planes::default().is_empty() && rd.decode_all_into(&prof, rr, Chans::Only(&[9]), &mut Planes::default()).is_err());
        }
    }

    /// U8/U16 are first-class under HTJ2K: a frame mixing U8, U16, F16, U32 and F32 channels round-trips
    /// bit-exact through both writers and the frame reader (odd tile edges included), and every channel is
    /// really an HTJ2K codestream (one multi-component blob per tile).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn htj2k_mixed_int_and_float_channels_roundtrip() {
        let mut hd = header(70, 45, 32);
        hd.codec_id = exv_format::CodecId::HTJ2K.0; // the reader derives the profile from the header
        hd.channels = vec![
            Channel { name: "m".into(), dtype: Dtype::U8 },
            Channel { name: "n".into(), dtype: Dtype::U16 },
            Channel { name: "h".into(), dtype: Dtype::F16 },
            Channel { name: "i".into(), dtype: Dtype::U32 },
            Channel { name: "z".into(), dtype: Dtype::F32 },
        ];
        let n = (hd.width * hd.height) as usize;
        let p: Vec<Vec<u8>> = hd
            .channels
            .iter()
            .enumerate()
            .map(|(ci, c)| (0..n * c.dtype.size()).map(|i| ((i * (7 + ci) + ci * 31) % 251) as u8).collect())
            .collect();
        let prof = Profile::htj2k();
        let body = write_frame_tiles(&p, None, &hd, &prof).expect("encode");
        assert_eq!(read_frame_tiles(&body, &hd, &prof, None, &LIM).expect("decode"), p);
        let mut sink = Vec::new();
        let dtypes: Vec<Dtype> = hd.channels.iter().map(|c| c.dtype).collect();
        let tx_n = hd.tiles_x() as u64;
        write_intra_frame_body_windowed(&hd, &prof, BUDGET, &mut sink, |t0, t1| {
            Ok((t0..t1)
                .map(|t| {
                    let r = Rect::of_tile(&hd, (t % tx_n) as u32, (t / tx_n) as u32);
                    (0..p.len()).map(|ci| gather(&p[ci], hd.width, r, dtypes[ci].size())).collect()
                })
                .collect())
        })
        .expect("stream");
        assert_eq!(sink, body, "streamed == in-RAM");
        let reader = FrameReader::open(&body, hd.clone()).expect("open");
        let blob = reader.tile_blob(0).expect("blob");
        assert_eq!(crate::htj2k::num_components(&blob[1..]), Some(5), "ONE multi-component codestream");
        for ci in 0..5 {
            assert_eq!(reader.decode_tile_channel(0, ci).expect("channel"), gather(&p[ci], hd.width, Rect::of_tile(&hd, 0, 0), dtypes[ci].size()));
        }
    }
}
