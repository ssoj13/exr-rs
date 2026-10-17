//! `exv-tile` - out-of-core tile engine (process 65k^2 without OOM).
//!
//! # What and why
//! One tile SOURCE trait, one byte-budget CACHE, one place that turns RAM/VRAM into numbers:
//! - [`source`]: [`TileSource`] (fallible, thread-safe), [`PlanesSource`] (foreign linear image),
//!   [`stream_tiles`] (one reused buffer, peak `O(tile)`), [`pipe`] (ordered windowed parallel loop).
//! - [`pool`]: [`TilePool`] - byte-LRU with [`Lease`] pins, single-flight loads and reserve-before-decode, so
//!   concurrent decodes can never jointly exceed the budget. RAM and VRAM caches are two instances.
//! - [`budget`]: [`Budget`], the memory formulas (`usable_ram`, `frame_cache`, `par_for`, `window_tiles`).
//!   RAM/VRAM come from `gpu-info-rs` only; an unanswerable OS query is [`BudgetError`], never a guessed default.
//! - [`region`]: [`read_region`], the ONE rectangle assembler (+ processing halo, strict or fault-tolerant).
//! - [`pick_tile_size`] / [`fits`]: tile size vs GPU limits (`docs/archive/PLAN.md` 3.5).
//!
//! Halo is NEVER used by the codec (wavelet/HTJ2K tiles are tile-local); it only widens a [`read_region`] request
//! for processing consumers.

pub mod budget;
pub mod pool;
pub mod region;
pub mod source;

pub use budget::{
    Budget, BudgetError, BudgetExt, Stage, VramError, VramPlan, frame_cache, frame_share, heap_slack, par_for, plan_vram, usable_ram, vram_share,
    window_spare, window_tiles,
};
pub use pool::{Cancel, Job, Lease, Loader, PlaneBuf, PlanePool, PlaneStats, PoolStats, TileKey, TilePool};
pub use region::{Edge, OnErr, Region, RegionOut, read_region};
pub use source::{
    PlanesSource, StreamStats, TileBuf, TileError, TileResult, TileSource, Weigh, check_tile, par_window, pipe, stream_tiles,
    tile_bytes_of,
};

use exv_format::{Dtype, TileGrid};

/// GPU limits a tile must fit (the numbers `gpu_info` reports for the device).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// `max_storage_buffer_binding_size`.
    pub max_binding: u64,
    /// VRAM budget for the decode working set.
    pub vram: u64,
    /// `max_texture_dimension_2d`.
    pub max_tex_dim: u32,
}

/// Largest tile edge a device takes: half the texture limit (halo headroom), hard-capped at 16384.
fn edge_cap(l: &DeviceLimits) -> u32 {
    (l.max_tex_dim / 2).clamp(1, 16384)
}

/// Does a tile of `side` x `side` pixels (halo included) fit the binding and VRAM limits?
///
/// per-binding: the largest single buffer is `b_sym = side^2 * esz * 4` (1 u32 per byte), independent of
/// channels (decode is per-channel) - the 4x blow-up the first model ignored (M1).
/// total VRAM: the full working set `b_sym + b_elem + b_out + staging + streams ~ side^2 * (esz*5 + 12)`,
/// x channels (conservatively all resident) x 2 (double buffer).
fn side_fits(l: &DeviceLimits, side: u64, esz: u64, channels: u64) -> bool {
    let area = side.saturating_mul(side);
    let sym_binding = area.saturating_mul(esz).saturating_mul(4);
    let working = area.saturating_mul(esz.saturating_mul(5) + 12).saturating_mul(channels).saturating_mul(2);
    sym_binding <= l.max_binding && working <= l.vram
}

/// Pick the tile size (power of two) to fit GPU limits: the largest `t` with [`side_fits`] at `t + 2*halo`
/// and `t <= max_texture_dim/2` (hard cap 16384).
pub fn pick_tile_size(
    max_binding_bytes: u64,
    vram_budget_bytes: u64,
    max_texture_dim: u32,
    esz: usize,
    channels: usize,
    halo: u32,
) -> u32 {
    let l = DeviceLimits { max_binding: max_binding_bytes, vram: vram_budget_bytes, max_tex_dim: max_texture_dim };
    let (esz, channels) = (esz as u64, channels.max(1) as u64);
    let cap = edge_cap(&l);
    let mut t = 1u32;
    while t * 2 <= cap && side_fits(&l, (t * 2 + 2 * halo) as u64, esz, channels) {
        t *= 2;
    }
    t
}

/// Validate at open that the grid's tiles (+ `halo`) fit `l` - the same formula as [`pick_tile_size`], so a file
/// whose tiles the device cannot decode is a typed [`TileError::Limit`] up front, not a mid-decode failure.
///
/// # Errors
/// [`TileError::Limit`] naming the violated limit.
pub fn fits(l: &DeviceLimits, g: &TileGrid, dtypes: &[Dtype], halo: u32) -> TileResult<()> {
    let (tw, th) = (g.tile_w.min(g.width), g.tile_h.min(g.height));
    let cap = edge_cap(l);
    if tw > cap || th > cap {
        return Err(TileError::Limit(format!("tile {tw}x{th} exceeds the device tile edge cap {cap}").into()));
    }
    let esz = dtypes.iter().map(|d| d.size() as u64).max().unwrap_or(1);
    let side = tw.max(th) as u64 + 2 * halo as u64;
    if !side_fits(l, side, esz, dtypes.len().max(1) as u64) {
        return Err(TileError::Limit(format!("tile {tw}x{th} (+halo {halo}) does not fit the device binding/VRAM limits").into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generator: computes tile bytes on the fly, NOT holding the whole image (out-of-core).
    struct GenSource {
        grid: TileGrid,
        dtypes: Vec<Dtype>,
    }

    impl TileSource for GenSource {
        fn grid(&self) -> TileGrid {
            self.grid
        }
        fn dtypes(&self) -> &[Dtype] {
            &self.dtypes
        }
        fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()> {
            let (x, y, tw, th) = check_tile(&self.grid, tx, ty)?;
            out.planes.resize_with(self.dtypes.len(), Vec::new);
            for (p, d) in out.planes.iter_mut().zip(&self.dtypes) {
                let esz = d.size();
                p.clear();
                p.resize(tw as usize * th as usize * esz, 0);
                for r in 0..th as usize {
                    for c in 0..tw as usize {
                        // cheap pattern (the remaining element bytes are 0)
                        p[(r * tw as usize + c) * esz] = ((x as usize + c) ^ (y as usize + r)) as u8;
                    }
                }
            }
            Ok(())
        }
    }

    #[test]
    fn planes_source_gathers_tiles_and_rejects_oob() {
        // small u16 plane 8x6, 4x4 tiles -> exact tile gather
        let (w, h) = (8u32, 6u32);
        let plane: Vec<u8> = (0..(w * h) as u16).flat_map(|v| v.to_le_bytes()).collect();
        let grid = TileGrid { width: w, height: h, tile_w: 4, tile_h: 4 };
        let src = PlanesSource::new(grid, vec![Dtype::U16], vec![&plane]).expect("planes");
        let mut buf = TileBuf::default();
        // tile (1,1): x=4,y=4 -> clamped to w=4,h=2 (edge)
        src.read_tile(1, 1, &mut buf).expect("edge tile");
        assert_eq!(buf.planes[0].len(), 4 * 2 * 2);
        // first tile element = pixel (4,4) = index 4*8+4 = 36
        assert_eq!(u16::from_le_bytes([buf.planes[0][0], buf.planes[0][1]]), 36);
        assert_eq!(src.tile_bytes(1, 1), Ok(16));
        // The old source panicked / returned garbage here.
        assert_eq!(src.read_tile(2, 0, &mut buf), Err(TileError::Oob { tx: 2, ty: 0, tiles_x: 2, tiles_y: 2 }));
        assert!(src.read_tile(0, 9, &mut buf).is_err());
        assert!(matches!(src.read_chan(0, 0, 3, &mut Vec::new()), Err(TileError::Chan { ci: 3, n: 1 })));
        let mut one = Vec::new();
        src.read_chan(1, 1, 0, &mut one).expect("chan");
        assert_eq!(one, buf.planes[0]);
    }

    #[test]
    fn planes_source_rejects_mismatched_planes() {
        let grid = TileGrid { width: 4, height: 4, tile_w: 2, tile_h: 2 };
        let short = vec![0u8; 15];
        assert!(PlanesSource::new(grid, vec![Dtype::U8], vec![&short]).is_err(), "short plane");
        assert!(PlanesSource::new(grid, vec![Dtype::U8, Dtype::U8], vec![&short]).is_err(), "plane count");
    }

    #[test]
    fn out_of_core_bounded_memory() {
        // Logical image 8192x8192 f32 = 256 MiB - NOT allocated whole; streamed in 256^2 tiles.
        let grid = TileGrid { width: 8192, height: 8192, tile_w: 256, tile_h: 256 };
        let src = GenSource { grid, dtypes: vec![Dtype::F32] };
        let mut checksum = 0u64;
        let st = stream_tiles(&src, |_, _, b| {
            checksum = checksum.wrapping_add(b.planes[0].iter().map(|&x| x as u64).sum::<u64>());
            Ok(())
        })
        .expect("stream");
        let one_tile = 256 * 256 * 4;
        assert_eq!(st.tiles, 32 * 32);
        assert!(st.peak_buf <= one_tile + 4096, "peak {} > one tile {one_tile} - not out-of-core", st.peak_buf);
        assert!(checksum > 0);
    }

    #[test]
    fn stream_aborts_on_the_first_error() {
        let grid = TileGrid { width: 64, height: 64, tile_w: 16, tile_h: 16 };
        let src = GenSource { grid, dtypes: vec![Dtype::U8] };
        let mut seen = 0;
        let r = stream_tiles(&src, |tx, ty, _| {
            seen += 1;
            if (tx, ty) == (1, 0) { Err(TileError::Cancelled) } else { Ok(()) }
        });
        assert_eq!((r, seen), (Err(TileError::Cancelled), 2));
    }

    #[test]
    fn grid_math_handles_65k() {
        // 65536^2 not allocated - only check geometry (uint32 resolutions); a 1-px-tile index would overflow u32.
        let g = TileGrid { width: 65536, height: 65536, tile_w: 256, tile_h: 256 };
        assert_eq!(g.tiles_x(), 256);
        assert_eq!(g.tile_count(), 256 * 256);
        assert_eq!(check_tile(&g, 255, 255), Ok((65280, 65280, 256, 256)));
        assert!(check_tile(&g, 256, 0).is_err());
        let one = TileGrid { width: 65536, height: 65536, tile_w: 1, tile_h: 1 };
        assert_eq!(one.tile_count(), 1 << 32);
    }

    #[test]
    fn pipe_is_ordered_for_any_window_and_stops_on_error() {
        for win in [1u64, 3, 7, 100] {
            let mut got = Vec::new();
            pipe(
                25,
                win,
                |i| Ok::<_, ()>(i * 2),
                |i, v| {
                    assert_eq!(v, i * 2);
                    got.push(i);
                    Ok(())
                },
            )
            .expect("pipe");
            assert_eq!(got, (0..25).collect::<Vec<_>>(), "window {win}");
        }
        let r = pipe(10, 4, |i| if i == 6 { Err(i) } else { Ok(i) }, |_, _| Ok(()));
        assert_eq!(r, Err(6));
    }

    #[test]
    fn pick_tile_size_respects_limits() {
        let big = u64::MAX; // drop one limit, check the other in isolation
        // per-binding 2 GiB, esz=4: b_sym=side^2*16; side=16384 -> 4.3 GiB > 2 GiB -> take 8192.
        assert_eq!(pick_tile_size(2 << 30, big, 32768, 4, 4, 0), 8192);
        // channel does NOT affect per-binding (b_sym is per-channel) -> also 8192.
        assert_eq!(pick_tile_size(2 << 30, big, 32768, 4, 1, 0), 8192);
        // both limits generous -> bounded by texture-cap/hard cap 16384.
        assert_eq!(pick_tile_size(big, big, 32768, 4, 1, 0), 16384);
        // total VRAM bounds it: 64 MiB, f32, 4 channels -> smaller tile.
        let t = pick_tile_size(big, 64 << 20, 32768, 4, 4, 0);
        assert!((64..=512).contains(&t), "vram-bound got {t}");
        // small binding 8 MiB, f32: b_sym=side^2*16 <= 8 MiB -> side<=512 -> tile 512.
        assert_eq!(pick_tile_size(8 << 20, big, 32768, 4, 1, 0), 512);
    }

    #[test]
    fn fits_agrees_with_pick_tile_size() {
        let l = DeviceLimits { max_binding: 2 << 30, vram: u64::MAX, max_tex_dim: 32768 };
        let dt = [Dtype::F32; 4];
        let ok = TileGrid { width: 65536, height: 65536, tile_w: 8192, tile_h: 8192 };
        let bad = TileGrid { width: 65536, height: 65536, tile_w: 16384, tile_h: 16384 };
        assert_eq!(fits(&l, &ok, &dt, 0), Ok(()));
        assert!(matches!(fits(&l, &bad, &dt, 0), Err(TileError::Limit(_))));
        let wide = DeviceLimits { max_binding: u64::MAX, ..l };
        assert_eq!(fits(&wide, &bad, &dt, 0), Ok(()));
        let tiny = TileGrid { width: 100, height: 100, tile_w: 256, tile_h: 256 };
        assert_eq!(fits(&l, &tiny, &dt, 4), Ok(()), "the effective tile is min(tile, image)");
    }
}
