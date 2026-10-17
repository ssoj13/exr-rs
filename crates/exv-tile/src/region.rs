//! ONE region assembler: read an arbitrary pixel rectangle (plus an optional processing halo) of any
//! [`TileSource`] into per-channel planes.
//!
//! # Why
//! Whole-frame decode, viewer panning, `bench` and the tiled-EXR bridge all need "the pixels of this rectangle";
//! three hand-written tile-range loops (strict, lenient, scatter) existed in the codec. The codec itself is
//! tile-local and never sees a halo: the halo only widens the rectangle here and edge-fills outside the image,
//! for processing consumers (blur / resample / LOD).

use exv_format::bufpool::alloc_zeroed;

use crate::budget::Budget;
use crate::pool::{Cancel, Lease, TileKey, TilePool};
use crate::source::{TileBuf, TileError, TileResult, TileSource, check_tile, par_window, pipe};

/// A pixel rectangle in image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// What a halo pixel outside the image holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// Replicate the nearest image pixel.
    Clamp,
    /// Zero.
    Zero,
}

/// What a failing tile does to the read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnErr {
    /// Abort with the first error (strict).
    Fail,
    /// Leave what failed zero-filled and list it in [`RegionOut::failed`]: one corrupt tile must not sink a 68 GB
    /// job. A tile whose read fails is retried channel by channel so intact channels of a per-channel codec survive.
    Record,
}

/// The planes of a region read.
#[derive(Debug)]
pub struct RegionOut {
    /// Image coordinates of the first output pixel (`region.x - halo`; negative when the halo leaves the image).
    pub x: i64,
    pub y: i64,
    /// Output size = `region.w + 2*halo` x `region.h + 2*halo`.
    pub w: u32,
    pub h: u32,
    /// One tightly packed row-major plane per channel.
    pub planes: Vec<Vec<u8>>,
    /// Failures recorded under [`OnErr::Record`]: `(tx, ty, channel, error)`. `channel` is `None` when the whole
    /// tile failed (HTJ2K: one codestream for all channels, or every channel failed) and `Some(ci)` when only
    /// that channel of a per-channel codec did - the others decoded and are in the planes.
    pub failed: Vec<(u32, u32, Option<usize>, TileError)>,
}

/// A tile either owned (straight from the source) or pinned in the pool.
enum Got {
    Own(TileBuf),
    Pooled(Lease<TileBuf>),
    /// Salvaged tile: good channels + zero-filled bad ones, with the per-channel failures.
    Partial(TileBuf, Vec<(Option<usize>, TileError)>),
}

impl Got {
    fn buf(&self) -> &TileBuf {
        match self {
            Got::Own(b) | Got::Partial(b, _) => b,
            Got::Pooled(l) => l,
        }
    }
}

/// Re-read a tile that failed as a whole, one channel at a time: intact channels are kept, bad ones are
/// zero-filled and listed. When EVERY channel fails the tile is reported once (`whole`).
fn salvage(src: &dyn TileSource, tx: u32, ty: u32, whole: TileError) -> TileResult<Got> {
    let (_, _, tw, th) = check_tile(&src.grid(), tx, ty)?;
    let (mut buf, mut bad) = (TileBuf::default(), Vec::new());
    for (ci, d) in src.dtypes().iter().enumerate() {
        let mut p = Vec::new();
        match src.read_chan(tx, ty, ci, &mut p) {
            Ok(()) => buf.planes.push(p),
            Err(e) => {
                buf.planes.push(vec![0; tw as usize * th as usize * d.size()]);
                bad.push((Some(ci), e));
            }
        }
    }
    if bad.len() == buf.planes.len() {
        bad = vec![(None, whole)];
    }
    Ok(Got::Partial(buf, bad))
}

/// The in-image part of the output rectangle, `[x0, x1) x [y0, y1)`.
type Clip = (i64, i64, i64, i64);

/// Read `r` (+ `halo` pixels each side, filled per `edge` outside the image) of `src`.
///
/// Tiles come through `pool` when given (a viewer pan re-reads the neighbours; the pool must belong to this
/// source: its keys carry only the tile coordinates). They decode in parallel windows sized from the bytes the
/// budget has left after the output planes. The output allocation is checked against `budget` BEFORE it is made.
///
/// # Errors
/// [`TileError::Limit`] for an empty or out-of-image region, [`TileError::OverBudget`] when the output planes do not
/// fit `budget`, or (under [`OnErr::Fail`]) the first failing tile's error.
pub fn read_region(
    src: &dyn TileSource,
    pool: Option<&TilePool<TileKey, TileBuf>>,
    r: Region,
    halo: u32,
    edge: Edge,
    on_err: OnErr,
    budget: &Budget,
) -> TileResult<RegionOut> {
    let g = src.grid();
    let dtypes = src.dtypes();
    let inside = |a: u32, n: u32, lim: u32| n > 0 && a.checked_add(n).is_some_and(|e| e <= lim);
    if !inside(r.x, r.w, g.width) || !inside(r.y, r.h, g.height) {
        return Err(TileError::Limit(format!("region {r:?} is empty or outside the {}x{} image", g.width, g.height).into()));
    }
    let (ow, oh) = (r.w as u64 + 2 * halo as u64, r.h as u64 + 2 * halo as u64);
    let esz_sum: u64 = dtypes.iter().map(|d| d.size() as u64).sum();
    let out_bytes = ow.saturating_mul(oh).saturating_mul(esz_sum);
    if out_bytes > budget.cap() {
        return Err(TileError::OverBudget {
            need: usize::try_from(out_bytes).unwrap_or(usize::MAX),
            used: 0,
            cap: usize::try_from(budget.cap()).unwrap_or(usize::MAX),
        });
    }
    let alloc = |n: u64| {
        let n = usize::try_from(n).map_err(|_| TileError::Limit("plane size overflows usize".into()))?;
        alloc_zeroed(n).map_err(|e| TileError::Limit(e.to_string().into()))
    };
    let mut planes = dtypes.iter().map(|d| alloc(ow * oh * d.size() as u64)).collect::<TileResult<Vec<_>>>()?;

    // Output origin and the part of it that lies inside the image (the tiles to read).
    let (ox, oy) = (r.x as i64 - halo as i64, r.y as i64 - halo as i64);
    let clip: Clip = (ox.max(0), oy.max(0), (ox + ow as i64).min(g.width as i64), (oy + oh as i64).min(g.height as i64));
    let (tw1, th1) = (g.tile_w.max(1), g.tile_h.max(1));
    let (tx0, ty0) = (clip.0 as u32 / tw1, clip.1 as u32 / th1);
    let (tx1, ty1) = ((clip.2 as u32 - 1) / tw1, (clip.3 as u32 - 1) / th1);
    let ntx = (tx1 - tx0 + 1) as u64;
    let n = ntx * (ty1 - ty0 + 1) as u64;

    // Window = what the budget leaves beside the output planes, over the worst-case tile (a full one).
    let full_tile = tw1 as u64 * th1 as u64 * esz_sum;
    let win = par_window(budget.cap().saturating_sub(out_bytes), full_tile);
    let cancel = Cancel::new();
    let at = |i: u64| (tx0 + (i % ntx) as u32, ty0 + (i / ntx) as u32);
    let mut failed = Vec::new();
    pipe(
        n,
        win,
        |i| {
            let (tx, ty) = at(i);
            let got = match pool {
                Some(p) => {
                    let key = TileKey { frame: 0, level: 0, chan: TileKey::ALL, tx, ty };
                    src.tile_bytes(tx, ty).and_then(|cost| {
                        p.get_or_load(key, cost, &cancel, |_| {
                            let mut b = TileBuf::default();
                            src.read_tile(tx, ty, &mut b).map(|()| b)
                        })
                        .map(Got::Pooled)
                    })
                }
                None => {
                    let mut b = TileBuf::default();
                    src.read_tile(tx, ty, &mut b).map(|()| Got::Own(b))
                }
            };
            Ok::<_, TileError>(match got {
                // Only a corrupt/unreadable tile is salvaged; a budget, limit or cancel error must surface (a
                // per-channel retry would decode outside the pool's reservation).
                Err(e @ (TileError::Decode { .. } | TileError::Io(_))) if on_err == OnErr::Record => salvage(src, tx, ty, e),
                g => g,
            })
        },
        |i, got| {
            let (tx, ty) = at(i);
            let t = got?;
            blit(&mut planes, src, (ox, oy, ow as usize), clip, (tx, ty), t.buf())?;
            if let Got::Partial(_, bad) = t {
                failed.extend(bad.into_iter().map(|(ci, e)| (tx, ty, ci, e)));
            }
            Ok(())
        },
    )?;

    if halo > 0 && edge == Edge::Clamp {
        let esz: Vec<usize> = dtypes.iter().map(|d| d.size()).collect();
        fill_clamp(&mut planes, &esz, (ox, oy, ow as usize, oh as usize), clip);
    }
    let dim = |n: u64| u32::try_from(n).map_err(|_| TileError::Limit("region plus halo exceeds u32".into()));
    Ok(RegionOut { x: ox, y: oy, w: dim(ow)?, h: dim(oh)?, planes, failed })
}

/// Copy the overlap of tile `(tx,ty)` with the in-image output rect `clip` into the output planes.
fn blit(
    planes: &mut [Vec<u8>],
    src: &dyn TileSource,
    (ox, oy, ow): (i64, i64, usize),
    (cx0, cy0, cx1, cy1): Clip,
    (tx, ty): (u32, u32),
    t: &TileBuf,
) -> TileResult<()> {
    let (x, y, tw, th) = check_tile(&src.grid(), tx, ty)?;
    let (ix0, ix1) = ((x as i64).max(cx0), (x as i64 + tw as i64).min(cx1));
    let (iy0, iy1) = ((y as i64).max(cy0), (y as i64 + th as i64).min(cy1));
    if t.planes.len() != planes.len() {
        return Err(TileError::Chan { ci: t.planes.len(), n: planes.len() });
    }
    for ((dst, srcp), d) in planes.iter_mut().zip(&t.planes).zip(src.dtypes()) {
        let esz = d.size();
        let row = (ix1 - ix0) as usize * esz;
        for iy in iy0..iy1 {
            let s = ((iy - y as i64) as usize * tw as usize + (ix0 - x as i64) as usize) * esz;
            let o = ((iy - oy) as usize * ow + (ix0 - ox) as usize) * esz;
            let from = srcp.get(s..s + row).ok_or_else(|| TileError::Io("tile plane shorter than its rect".into()))?;
            dst.get_mut(o..o + row).ok_or_else(|| TileError::Io("region plane overrun".into()))?.copy_from_slice(from);
        }
    }
    Ok(())
}

/// Replicate the nearest in-image pixel into the halo that lies outside the image: columns first (per valid row),
/// then whole rows.
fn fill_clamp(planes: &mut [Vec<u8>], esz: &[usize], (ox, oy, ow, oh): (i64, i64, usize, usize), clip: Clip) {
    let (cx0, cy0, cx1, cy1) = clip;
    let (l, r) = ((cx0 - ox) as usize, (ox + ow as i64 - cx1) as usize);
    let (rows0, rows1) = ((cy0 - oy) as usize, (cy1 - oy) as usize);
    for (p, &e) in planes.iter_mut().zip(esz) {
        for row in rows0..rows1 {
            let base = row * ow * e;
            let (first, last) = (base + l * e, base + (ow - r - 1) * e);
            for i in 0..l {
                p.copy_within(first..first + e, base + i * e);
            }
            for i in 0..r {
                p.copy_within(last..last + e, base + (ow - r + i) * e);
            }
        }
        for row in 0..oh {
            let from = row.clamp(rows0, rows1 - 1);
            if from != row {
                p.copy_within(from * ow * e..(from + 1) * ow * e, row * ow * e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::PlanesSource;
    use exv_format::{Dtype, TileGrid};

    /// A 2-channel (U8 + U16) image whose pixel value encodes its position.
    fn img(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
        let a = (0..w * h).map(|i| (i % 251) as u8).collect();
        let b = (0..w * h).flat_map(|i| ((i * 3 % 65521) as u16).to_le_bytes()).collect();
        (a, b)
    }

    fn src<'a>(a: &'a [u8], b: &'a [u8], w: u32, h: u32, t: u32) -> PlanesSource<'a> {
        PlanesSource::new(TileGrid { width: w, height: h, tile_w: t, tile_h: t }, vec![Dtype::U8, Dtype::U16], vec![a, b]).expect("src")
    }

    /// Image pixel at `(x, y)` clamped into the `w x h` image: `(u8 value, u16 bytes)`.
    fn px(a: &[u8], b: &[u8], (w, h): (u32, u32), x: i64, y: i64) -> (u8, [u8; 2]) {
        let (x, y) = (x.clamp(0, w as i64 - 1) as usize, y.clamp(0, h as i64 - 1) as usize);
        let i = y * w as usize + x;
        (a[i], [b[2 * i], b[2 * i + 1]])
    }

    fn at(o: &RegionOut, i: usize) -> (u8, [u8; 2]) {
        (o.planes[0][i], [o.planes[1][2 * i], o.planes[1][2 * i + 1]])
    }

    #[test]
    fn full_region_equals_the_source_image() {
        let (w, h) = (70u32, 45u32);
        let (a, b) = img(w, h);
        let s = src(&a, &b, w, h, 16);
        let out = read_region(&s, None, Region { x: 0, y: 0, w, h }, 0, Edge::Zero, OnErr::Fail, &Budget::new(1 << 20)).expect("read");
        assert_eq!((out.planes[0].as_slice(), out.planes[1].as_slice()), (a.as_slice(), b.as_slice()));
    }

    #[test]
    fn sub_region_and_pool_agree_with_direct() {
        let (w, h) = (70u32, 45u32);
        let (a, b) = img(w, h);
        let s = src(&a, &b, w, h, 16);
        let bud = Budget::new(1 << 20);
        let pool = TilePool::new(bud.clone());
        let r = Region { x: 13, y: 9, w: 40, h: 30 };
        let direct = read_region(&s, None, r, 0, Edge::Zero, OnErr::Fail, &bud).expect("direct");
        let pooled = read_region(&s, Some(&pool), r, 0, Edge::Zero, OnErr::Fail, &bud).expect("pooled");
        let again = read_region(&s, Some(&pool), r, 0, Edge::Zero, OnErr::Fail, &bud).expect("again");
        assert_eq!(direct.planes, pooled.planes);
        assert_eq!(pooled.planes, again.planes);
        let st = pool.stats();
        assert!(st.hits > 0 && st.loads == st.misses, "{st:?}");
        for yy in 0..r.h {
            for xx in 0..r.w {
                let want = px(&a, &b, (w, h), (r.x + xx) as i64, (r.y + yy) as i64);
                assert_eq!(at(&direct, (yy * r.w + xx) as usize), want);
            }
        }
    }

    #[test]
    fn halo_clamp_replicates_the_edge_and_zero_leaves_zeros() {
        let (w, h) = (20u32, 17u32);
        let (a, b) = img(w, h);
        let s = src(&a, &b, w, h, 8);
        let bud = Budget::new(1 << 20);
        let r = Region { x: 0, y: 0, w: 10, h: 9 };
        let c = read_region(&s, None, r, 3, Edge::Clamp, OnErr::Fail, &bud).expect("clamp");
        assert_eq!((c.x, c.y, c.w, c.h), (-3, -3, 16, 15));
        for oy in 0..c.h as i64 {
            for ox in 0..c.w as i64 {
                let want = px(&a, &b, (w, h), ox - 3, oy - 3);
                assert_eq!(at(&c, (oy * c.w as i64 + ox) as usize), want, "({ox},{oy})");
            }
        }
        let z = read_region(&s, None, r, 3, Edge::Zero, OnErr::Fail, &bud).expect("zero");
        assert_eq!(z.planes[0][0], 0, "halo corner is zero");
        // A halo that also leaves the image on the far side.
        let far = read_region(&s, None, Region { x: 10, y: 8, w: 10, h: 9 }, 3, Edge::Clamp, OnErr::Fail, &bud).expect("far");
        assert_eq!(at(&far, (far.w * far.h) as usize - 1), px(&a, &b, (w, h), w as i64 - 1, h as i64 - 1));
    }

    #[test]
    fn bad_requests_and_budget_are_typed_errors() {
        let (a, b) = img(16, 16);
        let s = src(&a, &b, 16, 16, 8);
        let go = |r, bud: &Budget| read_region(&s, None, r, 0, Edge::Zero, OnErr::Fail, bud).map(drop);
        let big = Budget::new(1 << 20);
        assert!(matches!(go(Region { x: 10, y: 0, w: 10, h: 4 }, &big), Err(TileError::Limit(_))));
        assert!(matches!(go(Region { x: 0, y: 0, w: 0, h: 4 }, &big), Err(TileError::Limit(_))));
        assert!(matches!(go(Region { x: 0, y: 0, w: 16, h: 16 }, &Budget::new(100)), Err(TileError::OverBudget { .. })));
    }

    /// A source whose tile (1,0) is corrupt.
    struct Flaky<'a>(PlanesSource<'a>);

    impl TileSource for Flaky<'_> {
        fn grid(&self) -> TileGrid {
            self.0.grid()
        }
        fn dtypes(&self) -> &[Dtype] {
            self.0.dtypes()
        }
        fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()> {
            if (tx, ty) == (1, 0) {
                return Err(TileError::Decode { tx, ty, ci: None, why: "corrupt".into() });
            }
            self.0.read_tile(tx, ty, out)
        }
    }

    /// A per-channel-codec stand-in: only channel 1 of tile (1,0) is corrupt, so `read_tile` fails as a whole but
    /// `read_chan` of the other channels works.
    struct ChanBad<'a>(PlanesSource<'a>);

    impl TileSource for ChanBad<'_> {
        fn grid(&self) -> TileGrid {
            self.0.grid()
        }
        fn dtypes(&self) -> &[Dtype] {
            self.0.dtypes()
        }
        fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()> {
            if (tx, ty) == (1, 0) {
                return Err(TileError::Decode { tx, ty, ci: Some(1), why: "corrupt".into() });
            }
            self.0.read_tile(tx, ty, out)
        }
        fn read_chan(&self, tx: u32, ty: u32, ci: usize, out: &mut Vec<u8>) -> TileResult<()> {
            if (tx, ty, ci) == (1, 0, 1) {
                return Err(TileError::Decode { tx, ty, ci: Some(ci), why: "corrupt".into() });
            }
            let mut t = TileBuf::default();
            self.0.read_tile(tx, ty, &mut t)?;
            *out = t.planes.swap_remove(ci);
            Ok(())
        }
    }

    #[test]
    fn a_bad_channel_does_not_take_the_good_ones_with_it() {
        let (a, b) = img(16, 16);
        let s = ChanBad(src(&a, &b, 16, 16, 8));
        let full = Region { x: 0, y: 0, w: 16, h: 16 };
        let out = read_region(&s, None, full, 0, Edge::Zero, OnErr::Record, &Budget::new(1 << 20)).expect("lenient");
        assert_eq!(out.failed.len(), 1);
        assert!(matches!(out.failed[0], (1, 0, Some(1), _)));
        assert_eq!(out.planes[0], a, "channel 0 of the failing tile survives");
        assert_eq!(&out.planes[1][..16], &b[..16], "channel 1 outside the bad tile is intact");
        assert_eq!(&out.planes[1][16..32], &[0u8; 16], "bad channel of the bad tile is zero");
    }

    #[test]
    fn corrupt_tile_fails_strict_and_is_recorded_when_lenient() {
        let (a, b) = img(16, 16);
        let s = Flaky(src(&a, &b, 16, 16, 8));
        let bud = Budget::new(1 << 20);
        let full = Region { x: 0, y: 0, w: 16, h: 16 };
        let strict = read_region(&s, None, full, 0, Edge::Zero, OnErr::Fail, &bud);
        assert!(matches!(strict, Err(TileError::Decode { tx: 1, ty: 0, .. })));
        let out = read_region(&s, None, full, 0, Edge::Zero, OnErr::Record, &bud).expect("lenient");
        assert_eq!(out.failed.len(), 1);
        assert_eq!(out.planes[0][8], 0, "failed tile zero-filled");
        assert_eq!(out.planes[0][0], a[0], "good tiles intact");
    }
}
