//! Tile grid of the display pyramid: pure
//! geometry, no GPU and no pixels.
//!
//! Coordinates are data-window pixels with the data window's `min` at `(0, 0)`.
//! Level `k` is the image reduced by `2^k` per axis; its pixel `(x, y)` covers the
//! level-0 block `[x·2^k, (x+1)·2^k) × [y·2^k, (y+1)·2^k)` clipped to the image, so
//! every level has `ceil(w / 2^k) x ceil(h / 2^k)` pixels. The top level fits one
//! tile.

/// Tile edge in pixels (design decision 2026-09-29: 512).
pub const TILE: u32 = 512;

/// One tile of the pyramid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    /// Pyramid level (0 = full resolution).
    pub level: u32,
    /// Tile column within the level.
    pub tx: u32,
    /// Tile row within the level.
    pub ty: u32,
}

/// An axis-aligned pixel rectangle (`w`, `h` >= 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixRect {
    /// Left column.
    pub x: u32,
    /// Top row.
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

/// The pyramid of one `w x h` image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    w: u32,
    h: u32,
    levels: u32,
    aspect_bits: u32,
    lattice: Option<(super::region::LatticeAxis, super::region::LatticeAxis)>,
}

/// `ceil(v / 2^k)`, never below 1.
fn reduce(v: u32, k: u32) -> u32 {
    let v = u64::from(v);
    let d = 1u64 << k.min(40);
    // At most `v`, so it fits back into u32.
    (v.div_ceil(d).max(1)) as u32
}

impl Grid {
    /// The pyramid of a `w x h` image (a zero extent is treated as 1).
    pub fn new(w: u32, h: u32) -> Self {
        let (w, h) = (w.max(1), h.max(1));
        let mut levels = 1;
        while reduce(w, levels - 1) > TILE || reduce(h, levels - 1) > TILE {
            levels += 1;
        }
        Self { w, h, levels, aspect_bits: 1.0f32.to_bits(), lattice: None }
    }

    /// Processed geometry on one immutable signed canonical sampling lattice.
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn on_lattice(w: u32, h: u32, origin: (i64,i64)) -> Result<Self,String> {
        super::region::pixel_count(w,h)?;
        let x=super::region::LatticeAxis::new(origin.0,w)?;
        let y=super::region::LatticeAxis::new(origin.1,h)?;
        let mut grid=Self {w,h,levels:1,aspect_bits:1.0f32.to_bits(),lattice:Some((x,y))};
        while { let (w,h)=grid.level_size(grid.levels-1); w>TILE || h>TILE } {
            grid.levels+=1;
        }
        Ok(grid)
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn with_aspect(mut self, aspect: f64) -> Result<Self, String> {
        let value = aspect as f32;
        if !value.is_finite() || value <= 0.0 || !(self.w as f32 * value).is_finite() {
            return Err("pixel aspect exceeds finite display geometry".into());
        }
        self.aspect_bits = value.to_bits();
        Ok(self)
    }
    pub(crate) fn display_aspect(&self) -> f32 { f32::from_bits(self.aspect_bits) }
    pub(crate) fn display_view(&self, view: [f32; 4]) -> [f32; 4] {
        let aspect = self.display_aspect();
        [view[0] / aspect, view[1], view[2] / aspect, view[3]]
    }
    /// A bounded whole-level effect may choose a proxy beyond the initial overview.
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn include_level(&mut self, level: u32) -> Result<(),String> {
        if level > 40 { return Err("processed proxy exceeds checked pyramid".into()); }
        self.levels = self.levels.max(level + 1);
        Ok(())
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn lattice_bounds(&self, k:u32) -> (i64,i64,i64,i64) {
        let step=1i128<<k.min(40);
        let Some((x,y))=self.lattice else {
            let (w,h)=self.level_size(k);
            return (0,0,i64::from(w),i64::from(h));
        };
        let (x0,x1)=x.bounds(step); let (y0,y1)=y.bounds(step);
        (x0,y0,x1,y1)
    }
    /// Fractional source edges in the requested tile's texels, preserving partial cells.
    pub(crate) fn texture_rect(&self,id:TileId,src:PixRect)->Option<[f32;4]> {
        let r=self.rect(id)?;
        let step=1i128<<id.level;
        let coord=|v:u32,axis:Option<super::region::LatticeAxis>,origin:u32| {
            let bin=axis.map_or(f64::from(v)/step as f64,|axis|axis.texel(v,step));
            (bin-f64::from(origin)) as f32
        };
        Some([coord(src.x,self.lattice.map(|v|v.0),r.x),coord(src.y,self.lattice.map(|v|v.1),r.y),
            coord(src.x.checked_add(src.w)?,self.lattice.map(|v|v.0),r.x),
            coord(src.y.checked_add(src.h)?,self.lattice.map(|v|v.1),r.y)])
    }

    /// Plan consumer reduction on the same signed lattice as displayed pixels.
    /// Reconstructing a zero-origin lattice would change ragged proxy edge weights.
    #[cfg(feature = "ofx")]
    pub(crate) fn region_plan(
        &self,
        level: u32,
        rect: PixRect,
        policy: super::region::RegionPolicy,
    ) -> Result<super::region::RegionPlan, String> {
        if level >= self.levels || level > 40 {
            return Err("consumer level is outside the checked pyramid".into());
        }
        let origin = self.lattice.map_or((0, 0), |(x, y)| (x.start(), y.start()));
        super::region::RegionPlan::on_lattice(
            self.size(), level, rect, policy, super::region::SamplingPhase::TOP_LEFT, origin,
        )
    }

    /// Top-first native cell covering one local canvas pixel.
    #[cfg(feature = "ofx")]
    pub(crate) fn level_index(&self, level: u32, x: u32, y: u32) -> Option<(u32, u32)> {
        if level >= self.levels || level > 40 || x >= self.w || y >= self.h { return None; }
        let step = 1i128 << level;
        let index = |local: u32, axis: Option<super::region::LatticeAxis>| {
            let origin = i128::from(axis.map_or(0, |axis| axis.start()));
            let first = axis.map_or(0, |axis| axis.bounds(step).0);
            u32::try_from((origin + i128::from(local)).div_euclid(step) - i128::from(first)).ok()
        };
        Some((index(x, self.lattice.map(|axis| axis.0))?,
              index(y, self.lattice.map(|axis| axis.1))?))
    }

    /// Checked canvas footprint of a top-first local level cell, including ragged edges.
    #[cfg(feature = "ofx")]
    pub(crate) fn level_cell(&self, level: u32, x: u32, y: u32) -> Option<PixRect> {
        if level >= self.levels || level > 40 { return None; }
        let (w, h) = self.level_size(level);
        if x >= w || y >= h { return None; }
        let step = 1i128 << level;
        let span = |index: u32, extent: u32, axis: Option<super::region::LatticeAxis>| {
            axis.map_or_else(|| {
                let start = (i128::from(index) * step).min(i128::from(extent)) as u32;
                let end = ((i128::from(index) + 1) * step).min(i128::from(extent)) as u32;
                start..end
            }, |axis| axis.span(index, 1, step))
        };
        let xs = span(x, self.w, self.lattice.map(|axis| axis.0));
        let ys = span(y, self.h, self.lattice.map(|axis| axis.1));
        let rect = PixRect { x: xs.start, y: ys.start, w: xs.end - xs.start, h: ys.end - ys.start };
        (rect.w > 0 && rect.h > 0).then_some(rect)
    }

    /// Level-0 extent.
    pub fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    /// Number of levels; the last one fits in a single tile.
    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Pixel extent of level `k`.
    pub fn level_size(&self, k: u32) -> (u32, u32) {
        if let Some((x,y))=self.lattice {
            let step=1i128<<k.min(40);
            let (x0,x1)=x.bounds(step);let (y0,y1)=y.bounds(step);
            // At level zero this is the validated u32 extent; every coarser
            // axis has at most ceil(extent/2)+1 bins, hence still fits u32.
            ((i128::from(x1)-i128::from(x0)) as u32,(i128::from(y1)-i128::from(y0)) as u32)
        } else { (reduce(self.w, k), reduce(self.h, k)) }
    }

    /// Tile columns and rows of level `k`.
    pub fn tiles(&self, k: u32) -> (u32, u32) {
        let (lw, lh) = self.level_size(k);
        (lw.div_ceil(TILE), lh.div_ceil(TILE))
    }

    /// The level pixels tile `id` holds (edge tiles are partial), or `None` for
    /// a tile outside the pyramid.
    pub fn rect(&self, id: TileId) -> Option<PixRect> {
        if id.level >= self.levels {
            return None;
        }
        let (lw, lh) = self.level_size(id.level);
        let (x, y) = (id.tx.checked_mul(TILE)?, id.ty.checked_mul(TILE)?);
        if x >= lw || y >= lh {
            return None;
        }
        Some(PixRect { x, y, w: (lw - x).min(TILE), h: (lh - y).min(TILE) })
    }

    /// The level-0 pixels tile `id` covers (clipped to the image).
    pub fn source_rect(&self, id: TileId) -> Option<PixRect> {
        let r = self.rect(id)?;
        let k = id.level;
        if let Some((x,y))=self.lattice {
            let xs=x.span(r.x,r.w,1i128<<k);
            let ys=y.span(r.y,r.h,1i128<<k);
            return Some(PixRect{x:xs.start,y:ys.start,w:xs.end-xs.start,h:ys.end-ys.start});
        }
        let x0 = u64::from(r.x) << k;
        let y0 = u64::from(r.y) << k;
        let x1 = (u64::from(r.x + r.w) << k).min(u64::from(self.w));
        let y1 = (u64::from(r.y + r.h) << k).min(u64::from(self.h));
        // All four are at most the level-0 extent, a u32.
        Some(PixRect { x: x0 as u32, y: y0 as u32, w: (x1 - x0) as u32, h: (y1 - y0) as u32 })
    }

    /// The level to draw at `zoom` screen pixels per level-0 pixel: the coarsest
    /// level whose pixels are still at least one screen pixel wide.
    pub fn pick_level(&self, zoom: f32) -> u32 {
        let top = self.levels - 1;
        if !(zoom.is_finite() && zoom > 0.0) {
            return top;
        }
        if zoom >= 1.0 {
            return 0;
        }
        let k = (1.0 / zoom).log2().floor();
        // `k` is finite and >= 0 here; clamp before the cast.
        (k.min(top as f32)) as u32
    }

    /// The tiles of level `k` that intersect `view` (level-0 pixel coordinates
    /// `[x0, y0, x1, y1)`, any values), grown by `ring` tiles on each side.
    pub fn visible(&self, k: u32, view: [f32; 4], ring: u32) -> Vec<TileId> {
        if k >= self.levels {
            return Vec::new();
        }
        let (cols, rows) = self.tiles(k);
        let step=1i128<<k;
        let offsets=self.lattice.map(|(x,y)|(
            (i128::from(x.bounds(step).0)*step-i128::from(x.start())) as f32,
            (i128::from(y.bounds(step).0)*step-i128::from(y.start())) as f32,
        )).unwrap_or((0.,0.));
        let view=[view[0]-offsets.0,view[1]-offsets.1,view[2]-offsets.0,view[3]-offsets.1];
        let span = (TILE as f32) * (1u64 << k) as f32; // level-0 pixels per tile
        let range = |lo: f32, hi: f32, n: u32| -> Option<(u32, u32)> {
            if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
                return None;
            }
            let a = (lo / span).floor();
            let b = (hi / span).ceil() - 1.0;
            if b < 0.0 || a >= n as f32 {
                return None;
            }
            let a = (a.max(0.0) as u32).saturating_sub(ring);
            let b = ((b.min((n - 1) as f32)) as u32).saturating_add(ring).min(n - 1);
            Some((a, b))
        };
        let (Some((tx0, tx1)), Some((ty0, ty1))) =
            (range(view[0], view[2], cols), range(view[1], view[3], rows))
        else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(((tx1 - tx0 + 1) * (ty1 - ty0 + 1)) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                out.push(TileId { level: k, tx, ty });
            }
        }
        out
    }

    /// The tile of level `k` that contains level-0 pixel `(x, y)`.
    pub fn tile_at(&self, k: u32, x: u32, y: u32) -> Option<TileId> {
        if k >= self.levels || x >= self.w || y >= self.h {
            return None;
        }
        if let Some((ax,ay))=self.lattice {
            let step=1i128<<k;
            let bin=|axis:super::region::LatticeAxis,v:u32| {
                ((i128::from(axis.start())+i128::from(v)).div_euclid(step)-i128::from(axis.bounds(step).0)) as u32
            };
            Some(TileId{level:k,tx:bin(ax,x)/TILE,ty:bin(ay,y)/TILE})
        } else {Some(TileId { level: k, tx: (x >> k) / TILE, ty: (y >> k) / TILE })}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_lattice_preserves_partial_cells_and_exact_source_edges() {
        for origin in [1,-1] {
            let mut grid=Grid::on_lattice(4,5,(origin,-5)).unwrap();
            grid.include_level(1).unwrap();
            assert_eq!(grid.level_size(1),(3,3));
            let id=TileId{level:1,tx:0,ty:0};
            let source=grid.source_rect(id).unwrap();
            assert_eq!(source,PixRect{x:0,y:0,w:4,h:5});
            assert_eq!(grid.texture_rect(id,source),Some([0.5,0.5,2.5,3.0]));
            assert_eq!(grid.tile_at(1,0,0),Some(id));
            assert_eq!(grid.tile_at(1,3,4),Some(id));
        }
    }
    #[test]
    fn shifted_tile_boundary_clips_without_stretching_the_last_cell() {
        let grid=Grid::on_lattice(1024,5,(1,-5)).unwrap();
        assert_eq!(grid.level_size(1),(513,3));
        let first=TileId{level:1,tx:0,ty:0};
        let last=TileId{level:1,tx:1,ty:0};
        let a=grid.source_rect(first).unwrap();
        let b=grid.source_rect(last).unwrap();
        assert_eq!(a,PixRect{x:0,y:0,w:1023,h:5});
        assert_eq!(b,PixRect{x:1023,y:0,w:1,h:5});
        assert_eq!(grid.texture_rect(first,a),Some([0.5,0.5,512.0,3.0]));
        assert_eq!(grid.texture_rect(last,b),Some([0.0,0.5,0.5,3.0]));
        assert_eq!(grid.tile_at(1,1022,0),Some(first));
        assert_eq!(grid.tile_at(1,1023,0),Some(last));
        assert_eq!(grid.visible(1,[1023.0,0.0,1024.0,5.0],0),[last]);
    }
    #[test]
    fn old_zero_origin_adapter_preserves_uniform_uvs() {
        let grid=Grid::new(1101,600);
        for level in 0..grid.levels() {
            let (cols,rows)=grid.tiles(level);
            for ty in 0..rows {for tx in 0..cols {
                let id=TileId{level,tx,ty};
                let source=grid.source_rect(id).unwrap();
                let rect=grid.rect(id).unwrap();
                let scale=(1u64<<level) as f32;
                assert_eq!(grid.texture_rect(id,source),Some([
                    source.x as f32/scale-rect.x as f32,
                    source.y as f32/scale-rect.y as f32,
                    (source.x+source.w) as f32/scale-rect.x as f32,
                    (source.y+source.h) as f32/scale-rect.y as f32]));
            }}
        }
    }

    #[test]
    fn levels_end_at_one_tile() {
        for (w, h, want) in [
            (1, 1, 1),
            (512, 512, 1),
            (513, 512, 2),
            (1024, 1, 2),
            (1025, 1, 3),
            (16384, 8192, 6),
            (65536, 65536, 8),
            (1, 100_000, 9),
        ] {
            let g = Grid::new(w, h);
            assert_eq!(g.levels(), want, "{w}x{h}");
            let (tw, th) = g.level_size(g.levels() - 1);
            assert!(tw <= TILE && th <= TILE, "{w}x{h}: top level {tw}x{th}");
            if g.levels() > 1 {
                let (pw, ph) = g.level_size(g.levels() - 2);
                assert!(pw > TILE || ph > TILE, "{w}x{h}: one level too many");
            }
        }
    }

    #[test]
    fn level_size_is_ceil_division() {
        let g = Grid::new(1001, 3);
        assert_eq!(g.level_size(0), (1001, 3));
        assert_eq!(g.level_size(1), (501, 2));
        assert_eq!(g.level_size(2), (251, 1));
        assert_eq!(g.level_size(9), (2, 1));
        assert_eq!(g.level_size(10), (1, 1));
    }

    #[test]
    fn tiles_cover_every_level_pixel_exactly_once() {
        for (w, h) in [(1, 1), (512, 512), (513, 1), (1100, 600), (4097, 3)] {
            let g = Grid::new(w, h);
            for k in 0..g.levels() {
                let (lw, lh) = g.level_size(k);
                let (cols, rows) = g.tiles(k);
                let mut area = 0u64;
                for ty in 0..rows {
                    for tx in 0..cols {
                        let r = g.rect(TileId { level: k, tx, ty }).expect("tile in grid");
                        assert!(r.w >= 1 && r.h >= 1 && r.w <= TILE && r.h <= TILE);
                        assert!(r.x + r.w <= lw && r.y + r.h <= lh);
                        area += u64::from(r.w) * u64::from(r.h);
                    }
                }
                assert_eq!(area, u64::from(lw) * u64::from(lh), "{w}x{h} level {k}");
                assert!(g.rect(TileId { level: k, tx: cols, ty: 0 }).is_none());
                assert!(g.rect(TileId { level: k, tx: 0, ty: rows }).is_none());
            }
            assert!(g.rect(TileId { level: g.levels(), tx: 0, ty: 0 }).is_none());
        }
    }

    #[test]
    fn source_rect_scales_and_clips() {
        let g = Grid::new(1100, 600);
        assert_eq!(g.source_rect(TileId { level: 0, tx: 2, ty: 1 }), Some(PixRect { x: 1024, y: 512, w: 76, h: 88 }));
        // Level 1 is 550x300: two tiles across, 1024 + 76 source columns.
        assert_eq!(g.source_rect(TileId { level: 1, tx: 0, ty: 0 }), Some(PixRect { x: 0, y: 0, w: 1024, h: 600 }));
        assert_eq!(g.source_rect(TileId { level: 1, tx: 1, ty: 0 }), Some(PixRect { x: 1024, y: 0, w: 76, h: 600 }));
        // Level 2 is 275x150: one tile covering all of level 0.
        assert_eq!(g.source_rect(TileId { level: 2, tx: 0, ty: 0 }), Some(PixRect { x: 0, y: 0, w: 1100, h: 600 }));
        // An odd edge: level 1 of 1101 wide has 551 px, the last covering one source column.
        let g = Grid::new(1101, 2);
        assert_eq!(g.level_size(1), (551, 1));
        assert_eq!(g.source_rect(TileId { level: 1, tx: 1, ty: 0 }), Some(PixRect { x: 1024, y: 0, w: 77, h: 2 }));
    }

    #[test]
    fn pick_level_keeps_level_pixels_at_least_one_screen_pixel() {
        let g = Grid::new(65536, 65536); // 8 levels
        assert_eq!(g.pick_level(4.0), 0);
        assert_eq!(g.pick_level(1.0), 0);
        assert_eq!(g.pick_level(0.6), 0);
        assert_eq!(g.pick_level(0.5), 1);
        assert_eq!(g.pick_level(0.26), 1);
        assert_eq!(g.pick_level(0.25), 2);
        assert_eq!(g.pick_level(1e-9), 7, "clamped to the top");
        assert_eq!(g.pick_level(0.0), 7);
        assert_eq!(g.pick_level(f32::NAN), 7);
        for k in 0..g.levels() {
            let zoom = 1.0 / (1u32 << k) as f32;
            assert_eq!(g.pick_level(zoom), k);
            assert!(zoom * (1u32 << g.pick_level(zoom)) as f32 >= 1.0 - 1e-6);
        }
    }

    #[test]
    fn visible_tiles_intersect_the_view() {
        let g = Grid::new(4096, 2048); // level 0: 8x4 tiles
        let ids = |v: Vec<TileId>| v.into_iter().map(|t| (t.tx, t.ty)).collect::<Vec<_>>();
        assert_eq!(ids(g.visible(0, [0.0, 0.0, 512.0, 512.0], 0)), [(0, 0)]);
        assert_eq!(ids(g.visible(0, [511.0, 0.0, 513.0, 1.0], 0)), [(0, 0), (1, 0)]);
        assert_eq!(ids(g.visible(0, [-100.0, -100.0, 10.0, 10.0], 0)), [(0, 0)]);
        assert_eq!(g.visible(0, [5000.0, 0.0, 6000.0, 10.0], 0), []);
        assert_eq!(g.visible(0, [10.0, 10.0, 10.0, 20.0], 0), [], "empty view");
        assert_eq!(g.visible(0, [f32::NAN, 0.0, 1.0, 1.0], 0), []);
        assert_eq!(ids(g.visible(0, [600.0, 600.0, 700.0, 700.0], 1)), [
            (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)
        ]);
        // Level 1 tiles span 1024 level-0 pixels.
        assert_eq!(ids(g.visible(1, [0.0, 0.0, 4096.0, 2048.0], 0)), [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]);
        // The whole image at every level is exactly the level's grid.
        for k in 0..g.levels() {
            let (c, r) = g.tiles(k);
            assert_eq!(g.visible(k, [0.0, 0.0, 4096.0, 2048.0], 3).len() as u32, c * r);
        }
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn consumer_proxy_preserves_signed_ragged_cell_boundaries() {
        let mut grid = Grid::on_lattice(5, 7, (-3, -7)).unwrap();
        grid.include_level(2).unwrap();
        assert_eq!(grid.level_size(2), (2, 2));
        assert_eq!(grid.level_cell(2, 0, 0), Some(PixRect { x: 0, y: 0, w: 3, h: 3 }));
        assert_eq!(grid.level_cell(2, 1, 1), Some(PixRect { x: 3, y: 3, w: 2, h: 4 }));
        assert_eq!(grid.level_index(2, 2, 2), Some((0, 0)));
        assert_eq!(grid.level_index(2, 3, 3), Some((1, 1)));
        let plan = grid.region_plan(2, PixRect { x: 1, y: 1, w: 1, h: 1 },
            super::super::region::RegionPolicy::OriginalArea).unwrap();
        assert_eq!(plan.lattice_origin(), (-3, -7));
        assert_eq!(plan.level_bounds(), (-1, -2, 1, 0));
        assert_eq!(plan.rows(), 3..7);
        assert!(grid.level_cell(2, 2, 0).is_none());
        assert!(grid.level_index(2, 5, 0).is_none());
        assert!(grid.region_plan(3, PixRect { x: 0, y: 0, w: 1, h: 1 },
            super::super::region::RegionPolicy::OriginalArea).is_err());
    }

    #[test]
    fn tile_at_finds_the_containing_tile() {
        let g = Grid::new(4096, 2048);
        assert_eq!(g.tile_at(0, 1023, 512), Some(TileId { level: 0, tx: 1, ty: 1 }));
        assert_eq!(g.tile_at(1, 1024, 0), Some(TileId { level: 1, tx: 1, ty: 0 }));
        assert_eq!(g.tile_at(0, 4096, 0), None);
        for (x, y) in [(0, 0), (4095, 2047), (1500, 900)] {
            for k in 0..g.levels() {
                let id = g.tile_at(k, x, y).unwrap();
                let s = g.source_rect(id).unwrap();
                assert!(x >= s.x && x < s.x + s.w && y >= s.y && y < s.y + s.h);
            }
        }
    }
}
