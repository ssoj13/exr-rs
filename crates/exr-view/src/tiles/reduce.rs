//! Tile pixels of any pyramid level from a level-0 RGBA image
//! (`.bughunt/design_tiling.md` §3.2): the ONE reduction rule every tile
//! source uses.
//!
//! Level-`k` pixel `(x, y)` is the mean of the level-0 pixels in
//! `[x·2^k, (x+1)·2^k) × [y·2^k, (y+1)·2^k)` clipped to the image (an area
//! average; an edge pixel averages only the pixels it really covers). Sums are
//! taken in `f64` so a deep overview level (4^k samples) does not lose precision
//! to the order of addition. Tiles are stored as `f16` RGBA — display data;
//! exact values are read from the source (the pixel readout never uses a tile).

use imath_rs::Half;

use super::grid::{Grid, PixRect, TileId};
use super::region::{RegionAccumulator, RegionPlan, RegionPolicy, SamplingPhase};

/// One tile's pixels: `w * h` RGBA texels as `f16` bits, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct TileData {
    /// Width in texels (<= [`super::grid::TILE`]).
    pub w: u32,
    /// Height in texels.
    pub h: u32,
    /// `w * h * 4` half-float bit patterns.
    pub rgba: Vec<u16>,
}

impl TileData {
    /// Heap bytes held (the cache weight).
    pub fn bytes(&self) -> usize {
        self.rgba.len() * 2
    }

    /// Texel `(x, y)` as `f32` RGBA.
    #[cfg(test)]
    pub fn texel(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        if x >= self.w || y >= self.h {
            return None;
        }
        let i = (y as usize * self.w as usize + x as usize) * 4;
        let px = self.rgba.get(i..i + 4)?;
        Some([0, 1, 2, 3].map(|c| f32::from(Half::from_bits(px[c]))))
    }
}

/// Tile `id` of the pyramid `grid` of the level-0 image `src` (`w * h * 4`
/// linear RGBA `f32`, row-major, `(w, h) == grid.size()`). `None` for a tile
/// outside the pyramid or a source of the wrong length.
pub fn tile_from_rgba(src: &[f32], grid: &Grid, id: TileId) -> Option<TileData> {
    let r = grid.rect(id)?;
    let plan = RegionPlan::new(
        grid.size(),
        id.level,
        r,
        RegionPolicy::OriginalArea,
        SamplingPhase::TOP_LEFT,
    )
    .ok()?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(plan.output_bytes() / 4).ok()?;
    super::region::resident_rows(
        src,
        plan,
        || false,
        |row| rgba.extend(row.iter().map(|&v| Half::from(v).bits())),
    )
    .ok()?;
    Some(TileData {
        w: r.w,
        h: r.h,
        rgba,
    })
}

/// The entire level, through the same streaming f64 region kernel as tiles.
pub fn level_from_rgba(src: &[f32], grid: &Grid, k: u32) -> Option<Vec<f32>> {
    if k >= grid.levels() {
        return None;
    }
    let (w, h) = grid.level_size(k);
    let plan = RegionPlan::new(
        grid.size(),
        k,
        PixRect { x: 0, y: 0, w, h },
        RegionPolicy::OriginalArea,
        SamplingPhase::TOP_LEFT,
    )
    .ok()?;
    super::region::resident(src, plan, || false).ok()
}

/// One tile row, backed by the shared f32 region accumulator. A file overview
/// retains several accumulators and feeds them all in one decode pass.
pub struct BandAccumulator {
    grid: Grid,
    level: u32,
    ty: u32,
    acc: RegionAccumulator,
}
impl BandAccumulator {
    pub fn new(grid: &Grid, level: u32, ty: u32) -> Option<Self> {
        let first = grid.rect(TileId { level, tx: 0, ty })?;
        let (w, _) = grid.level_size(level);
        let rect = PixRect {
            x: 0,
            y: first.y,
            w,
            h: first.h,
        };
        let plan = RegionPlan::new(
            grid.size(),
            level,
            rect,
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .ok()?;
        Some(Self {
            grid: *grid,
            level,
            ty,
            acc: RegionAccumulator::new(plan).ok()?,
        })
    }
    pub fn rows(&self) -> std::ops::Range<u32> {
        self.acc.rows()
    }
    pub fn push_row(&mut self, y: u32, row: &[f32]) -> Result<(), String> {
        self.acc.push_row(y, row)
    }
    pub fn finish(self) -> Result<Vec<(TileId, TileData)>, String> {
        let rgba = self.acc.finish()?;
        let (lw, _) = self.grid.level_size(self.level);
        let (cols, _) = self.grid.tiles(self.level);
        let mut out = Vec::with_capacity(cols as usize);
        for tx in 0..cols {
            let id = TileId {
                level: self.level,
                tx,
                ty: self.ty,
            };
            let r = self.grid.rect(id).ok_or("tile row outside grid")?;
            let mut bits = Vec::with_capacity(r.w as usize * r.h as usize * 4);
            for y in 0..r.h {
                let i = (y as usize * lw as usize + r.x as usize) * 4;
                bits.extend(
                    rgba[i..i + r.w as usize * 4]
                        .iter()
                        .map(|&v| Half::from(v).bits()),
                );
            }
            out.push((
                id,
                TileData {
                    w: r.w,
                    h: r.h,
                    rgba: bits,
                },
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::grid::TILE;

    /// Deterministic, non-degenerate test image.
    fn image(w: u32, h: u32) -> Vec<f32> {
        (0..w * h * 4)
            .map(|i| ((i * 37 % 101) as f32) * 0.25 - 3.0)
            .collect()
    }

    /// Reference: the same area average written the obvious way, per pixel.
    fn reference(src: &[f32], w: u32, h: u32, k: u32, lx: u32, ly: u32) -> [f32; 4] {
        let s = 1u32 << k;
        let mut sum = [0f64; 4];
        let mut n = 0u32;
        for y in ly * s..((ly + 1) * s).min(h) {
            for x in lx * s..((lx + 1) * s).min(w) {
                for c in 0..4 {
                    sum[c] += f64::from(src[((y * w + x) * 4 + c as u32) as usize]);
                }
                n += 1;
            }
        }
        sum.map(|v| f32::from(Half::from((v / f64::from(n)) as f32)))
    }

    #[test]
    fn level_zero_is_the_source_rounded_to_half() {
        let (w, h) = (700, 530); // 2 x 2 tiles at level 0, partial edges
        let src = image(w, h);
        let g = Grid::new(w, h);
        let t = tile_from_rgba(
            &src,
            &g,
            TileId {
                level: 0,
                tx: 1,
                ty: 1,
            },
        )
        .unwrap();
        assert_eq!((t.w, t.h), (700 - TILE, 530 - TILE));
        for (x, y) in [(0, 0), (t.w - 1, t.h - 1), (37, 5)] {
            let i = (((y + TILE) * w + x + TILE) * 4) as usize;
            let want = [0, 1, 2, 3].map(|c| f32::from(Half::from(src[i + c])));
            assert_eq!(t.texel(x, y), Some(want));
        }
    }

    #[test]
    fn every_level_matches_the_reference_area_average() {
        for (w, h) in [(1, 1), (3, 5), (513, 3), (1100, 600), (1025, 1)] {
            let src = image(w, h);
            let g = Grid::new(w, h);
            for k in 0..g.levels() {
                let (cols, rows) = g.tiles(k);
                for ty in 0..rows {
                    for tx in 0..cols {
                        let id = TileId { level: k, tx, ty };
                        let t = tile_from_rgba(&src, &g, id).unwrap();
                        let r = g.rect(id).unwrap();
                        assert_eq!((t.w, t.h), (r.w, r.h));
                        for y in 0..t.h {
                            for x in 0..t.w {
                                let want = reference(&src, w, h, k, r.x + x, r.y + y);
                                assert_eq!(t.texel(x, y), Some(want), "{w}x{h} L{k} {x},{y}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn edge_pixels_average_only_what_they_cover() {
        // 1025 x 1: level 1 is 513 wide; its last pixel covers source column 1024
        // alone, its first covers columns 0 and 1.
        let mut src = vec![0.0f32; 1025 * 4];
        src[0] = 1.0;
        src[4] = 3.0;
        src[1024 * 4] = 9.0;
        let g = Grid::new(1025, 1);
        assert_eq!(g.level_size(1), (513, 1));
        let first = tile_from_rgba(
            &src,
            &g,
            TileId {
                level: 1,
                tx: 0,
                ty: 0,
            },
        )
        .unwrap();
        assert_eq!(first.texel(0, 0).map(|p| p[0]), Some(2.0));
        let last = tile_from_rgba(
            &src,
            &g,
            TileId {
                level: 1,
                tx: 1,
                ty: 0,
            },
        )
        .unwrap();
        assert_eq!((last.w, last.h), (1, 1));
        assert_eq!(
            last.texel(0, 0).map(|p| p[0]),
            Some(9.0),
            "not averaged with a phantom column"
        );
    }

    /// The full-level image and the tiles are the same average: every tile
    /// texel equals the level image's texel rounded to half.
    #[test]
    fn level_image_matches_the_tiles() {
        let (w, h) = (1100, 600);
        let src = image(w, h);
        let g = Grid::new(w, h);
        for k in 0..g.levels() {
            let (lw, _) = g.level_size(k);
            let level = level_from_rgba(&src, &g, k).unwrap();
            let (cols, rows) = g.tiles(k);
            for ty in 0..rows {
                for tx in 0..cols {
                    let id = TileId { level: k, tx, ty };
                    let (t, r) = (tile_from_rgba(&src, &g, id).unwrap(), g.rect(id).unwrap());
                    for (x, y) in [(0, 0), (t.w - 1, t.h - 1), (t.w / 2, t.h / 3)] {
                        let i = (((r.y + y) * lw + r.x + x) * 4) as usize;
                        let want = [0, 1, 2, 3].map(|c| f32::from(Half::from(level[i + c])));
                        assert_eq!(
                            t.texel(x, y),
                            Some(want),
                            "L{k} tile {tx},{ty} texel {x},{y}"
                        );
                    }
                }
            }
        }
        assert!(level_from_rgba(&src, &g, g.levels()).is_none());
        assert!(level_from_rgba(&src[1..], &g, 0).is_none());
    }

    /// Streaming the rows through the accumulator gives the tiles
    /// `tile_from_rgba` gives, bit for bit, at every level and tile row.
    #[test]
    fn band_accumulator_matches_tile_from_rgba() {
        for (w, h) in [(1, 1), (513, 3), (1100, 600), (1025, 1030)] {
            let src = image(w, h);
            let g = Grid::new(w, h);
            for k in 0..g.levels() {
                let (_, rows) = g.tiles(k);
                for ty in 0..rows {
                    let mut acc = BandAccumulator::new(&g, k, ty).unwrap();
                    for y in acc.rows() {
                        let row = &src[(y * w * 4) as usize..((y + 1) * w * 4) as usize];
                        acc.push_row(y, row).unwrap();
                    }
                    for (id, tile) in acc.finish().unwrap() {
                        assert_eq!(Some(tile), tile_from_rgba(&src, &g, id), "{w}x{h} {id:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn band_accumulator_rejects_out_of_order_or_short_input() {
        let g = Grid::new(4, 4);
        let src = image(4, 4);
        let mut acc = BandAccumulator::new(&g, 0, 0).unwrap();
        assert!(
            acc.push_row(1, &src[16..32]).is_err(),
            "row 0 must come first"
        );
        assert!(acc.push_row(0, &src[0..8]).is_err(), "short row");
        acc.push_row(0, &src[0..16]).unwrap();
        assert!(acc.finish().is_err(), "stopped after one of four rows");
        assert!(BandAccumulator::new(&g, 0, 1).is_none());
    }

    #[test]
    fn rejects_a_wrong_source_or_tile() {
        let g = Grid::new(4, 4);
        assert!(
            tile_from_rgba(
                &[0.0; 63],
                &g,
                TileId {
                    level: 0,
                    tx: 0,
                    ty: 0
                }
            )
            .is_none()
        );
        assert!(
            tile_from_rgba(
                &[0.0; 64],
                &g,
                TileId {
                    level: 0,
                    tx: 1,
                    ty: 0
                }
            )
            .is_none()
        );
        assert!(
            tile_from_rgba(
                &[0.0; 64],
                &g,
                TileId {
                    level: 1,
                    tx: 0,
                    ty: 0
                }
            )
            .is_none()
        );
    }
}
