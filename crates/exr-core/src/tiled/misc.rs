//! Port of OpenEXR `ImfTiledMisc.{cpp,h}` — level/tile math for tiled files.
//!
//! This is the foundation for all tiled I/O: it computes, from a
//! [`TileDescription`] and a data window, how many MIP/RIP levels exist, the
//! pixel size of each level, the number of tiles per level, and the data
//! window covered by a given level or tile.  These formulas are the single
//! most bug-prone part of the tiled format, so every helper is a 1:1 port of
//! the C++ source and the rounding (`ROUND_DOWN` truncates, `ROUND_UP` rounds
//! up) matches it exactly.
//!
//! Unlike C++ this module owns no `new[]`/`delete[]` plumbing: where the C++
//! `precalculateTileInfo` fills caller-supplied `int*` arrays, we return owned
//! `Vec<i32>`s ([`precalculate_tile_info`]).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledMisc.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledMisc.h>

use imath_rs::{Box2i, V2i};

use crate::attr::channel_list::ChannelList;
use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
use crate::error::{ExrError, Result};
use crate::header::Header;

// ---------------------------------------------------------------------------
// levelSize — ImfTiledMisc.cpp lines 26-38
// ---------------------------------------------------------------------------

/// Pixel size of level `l` along one axis — port of `levelSize`.
///
/// `a = max - min + 1` is the full-resolution size; `b = 2^l` is the level's
/// down-scale factor.  The base size is the integer division `a / b`; under
/// `ROUND_UP` it is bumped by one when the division truncated (`size*b < a`).
/// The result is clamped to at least 1, so every level has a non-empty extent.
///
/// Returns [`ExrError::Arg`] when `l < 0`, mirroring the C++ `ArgExc`.
///
/// C++:
/// ```cpp
/// int a = max - min + 1;
/// int b = (1 << l);
/// int size = a / b;
/// if (rmode == ROUND_UP && size * b < a) size += 1;
/// return std::max (size, 1);
/// ```
pub fn level_size(min: i32, max: i32, l: i32, rmode: LevelRoundingMode) -> Result<i32> {
    if l < 0 {
        return Err(ExrError::Arg("Argument not in valid range.".into()));
    }
    // `1 << l` is computed in i64 below, so any
    // `l` that fits an i64 shift (`l <= 62`) is safe; the old `l >= 31` bound
    // assumed an i32 shift and rejected legitimate levels. A `ROUND_UP` image
    // wider/taller than 2^30 needs level 31 (`ceil_log2` returns 31 for a
    // dimension in `(2^30, 2^31)`, so `num_levels = 32`, last level index 31).
    // C++ 3.4.15 `OpenEXRCore/parse_header.c:2326-2338` computes
    // `calc_level_size` in int64 too (`1<<31` does not overflow there), and
    // legacy `ImfTiledMisc.cpp:31-37` computes `1<<l` in a plain `int`
    // (`1<<31` is UB, INT_MIN in practice, giving size 1 — same practical
    // result as the widened i64 path here). Reject only levels that cannot
    // possibly arise from a valid `TileDescription` (i64 shift overflow).
    if !(0..=62).contains(&l) {
        return Err(ExrError::Arg("Argument not in valid range.".into()));
    }

    // i64 intermediates: `max - min + 1` and `size * b` overflow i32 for extreme
    // windows reached via public callers that bypass `Header::sanity_check`.
    // Compute wide, then narrow the final size (valid windows fit i32, so this is
    // identical to the C++ result; an unrepresentable size is a hard error).
    let a = max as i64 - min as i64 + 1;
    let b = 1i64 << l;
    let mut size = a / b;

    if rmode == LevelRoundingMode::RoundUp && size * b < a {
        size += 1;
    }

    i32::try_from(size.max(1))
        .map_err(|_| ExrError::Arg("Argument not in valid range.".into()))
}

/// One axis extent `max - min + 1`, computed wide and range-checked.  An extent
/// that exceeds i32 (reachable via public tiled callers that bypass
/// `Header::sanity_check`) is rejected instead of overflowing i32 (debug panic /
/// release wrap).  For a valid window the result is identical to `max - min + 1`.
fn axis_extent(min: i32, max: i32) -> Result<i32> {
    let n = max as i64 - min as i64 + 1;
    i32::try_from(n).map_err(|_| ExrError::Arg("Argument not in valid range.".into()))
}

// ---------------------------------------------------------------------------
// dataWindowForLevel — ImfTiledMisc.cpp lines 40-58
// ---------------------------------------------------------------------------

/// Data window of level `(lx, ly)` — port of `dataWindowForLevel`.
///
/// The level's min corner equals the full-resolution min corner; its max
/// corner is `min + (levelSize - 1)` along each axis.
///
/// C++:
/// ```cpp
/// V2i levelMin = V2i (minX, minY);
/// V2i levelMax = levelMin + V2i (levelSize (minX, maxX, lx, mode) - 1,
///                                levelSize (minY, maxY, ly, mode) - 1);
/// return Box2i (levelMin, levelMax);
/// ```
pub fn data_window_for_level(
    tile_desc: &TileDescription,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    lx: i32,
    ly: i32,
) -> Result<Box2i> {
    let level_min = V2i { x: min_x, y: min_y };

    let level_max = V2i {
        x: level_min.x + level_size(min_x, max_x, lx, tile_desc.rounding_mode)? - 1,
        y: level_min.y + level_size(min_y, max_y, ly, tile_desc.rounding_mode)? - 1,
    };

    Ok(Box2i::from_min_max(level_min, level_max))
}

// ---------------------------------------------------------------------------
// dataWindowForTile — ImfTiledMisc.cpp lines 60-85
// ---------------------------------------------------------------------------

/// Data window of tile `(dx, dy)` in level `(lx, ly)` — port of `dataWindowForTile`.
///
/// The tile's min corner is offset from the level min by `dx*xSize`, `dy*ySize`.
/// Its max corner is `min + tileSize - 1`, clamped to the level's max corner so
/// edge tiles are not padded past the image.  The clamp uses `int64_t` math to
/// match the C++ overflow-safe arithmetic.
///
/// C++:
/// ```cpp
/// V2i tileMin = V2i (minX + dx * xSize, minY + dy * ySize);
/// int64_t tileMaxX = int64_t (tileMin[0]) + xSize - 1;
/// int64_t tileMaxY = int64_t (tileMin[1]) + ySize - 1;
/// V2i levelMax = dataWindowForLevel (...).max;
/// V2i tileMax = V2i (std::min (tileMaxX, int64_t (levelMax[0])),
///                    std::min (tileMaxY, int64_t (levelMax[1])));
/// return Box2i (tileMin, tileMax);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn data_window_for_tile(
    tile_desc: &TileDescription,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
) -> Result<Box2i> {
    // Widen the tile-origin arithmetic: `dx * xSize` and its sum with `min_x`
    // overflow i32 for extreme tile coordinates reached via public callers that
    // bypass `Header::sanity_check`.  Valid coordinates fit i32 (identical to the
    // C++ result); an unrepresentable origin is a hard error rather than a wrap.
    let tile_min_x = min_x as i64 + dx as i64 * tile_desc.x_size as i64;
    let tile_min_y = min_y as i64 + dy as i64 * tile_desc.y_size as i64;
    let tile_min = V2i {
        x: i32::try_from(tile_min_x)
            .map_err(|_| ExrError::Arg("Tile x coordinate out of range.".into()))?,
        y: i32::try_from(tile_min_y)
            .map_err(|_| ExrError::Arg("Tile y coordinate out of range.".into()))?,
    };

    let tile_max_x = tile_min.x as i64 + tile_desc.x_size as i64 - 1;
    let tile_max_y = tile_min.y as i64 + tile_desc.y_size as i64 - 1;

    let level_max = data_window_for_level(tile_desc, min_x, max_x, min_y, max_y, lx, ly)?.max;

    let tile_max = V2i {
        x: tile_max_x.min(level_max.x as i64) as i32,
        y: tile_max_y.min(level_max.y as i64) as i32,
    };

    Ok(Box2i::from_min_max(tile_min, tile_max))
}

// ---------------------------------------------------------------------------
// calculateBytesPerPixel — ImfTiledMisc.cpp lines 87-101
// ---------------------------------------------------------------------------

/// Sum of all channels' sample sizes in bytes — port of `calculateBytesPerPixel`.
///
/// C++ iterates the header's channel list and accumulates
/// `pixelTypeSize (c.channel ().type)`.
pub fn calculate_bytes_per_pixel(header: &Header) -> Result<usize> {
    let channels: &ChannelList = header.channels()?;

    let mut bytes_per_pixel = 0usize;
    for (_name, channel) in channels.iter() {
        bytes_per_pixel += crate::misc::pixel_type_size(channel.type_)?;
    }

    Ok(bytes_per_pixel)
}

// ---------------------------------------------------------------------------
// floorLog2 / ceilLog2 / roundLog2 — ImfTiledMisc.cpp anon namespace 143-186
// ---------------------------------------------------------------------------

/// `floor(log2(x))` for `x > 0` — port of the file-local `floorLog2`.
///
/// Counts how many times `x` can be halved before reaching 1.
fn floor_log2(mut x: i32) -> i32 {
    let mut y = 0;
    while x > 1 {
        y += 1;
        x >>= 1;
    }
    y
}

/// `ceil(log2(x))` for `x > 0` — port of the file-local `ceilLog2`.
///
/// Like [`floor_log2`] but bumps the result by one if any discarded low bit was
/// set (i.e. `x` was not an exact power of two).
fn ceil_log2(mut x: i32) -> i32 {
    let mut y = 0;
    let mut r = 0;
    while x > 1 {
        if x & 1 != 0 {
            r = 1;
        }
        y += 1;
        x >>= 1;
    }
    y + r
}

/// `floorLog2` for `ROUND_DOWN`, `ceilLog2` for `ROUND_UP` — port of `roundLog2`.
fn round_log2(x: i32, rmode: LevelRoundingMode) -> i32 {
    if rmode == LevelRoundingMode::RoundDown {
        floor_log2(x)
    } else {
        ceil_log2(x)
    }
}

// ---------------------------------------------------------------------------
// calculateNumXLevels / calculateNumYLevels — ImfTiledMisc.cpp 188-252
// ---------------------------------------------------------------------------

/// Number of levels along X — port of `calculateNumXLevels`.
///
/// * `ONE_LEVEL` → 1.
/// * `MIPMAP_LEVELS` → `roundLog2(max(w, h)) + 1` (square pyramid count).
/// * `RIPMAP_LEVELS` → `roundLog2(w) + 1` (independent X reductions).
///
/// Returns [`ExrError::Arg`] for an unknown level mode, mirroring the C++ `ArgExc`.
pub fn num_x_levels(
    tile_desc: &TileDescription,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> Result<i32> {
    let num = match tile_desc.mode {
        LevelMode::OneLevel => 1,
        LevelMode::MipmapLevels => {
            let w = axis_extent(min_x, max_x)?;
            let h = axis_extent(min_y, max_y)?;
            round_log2(w.max(h), tile_desc.rounding_mode) + 1
        }
        LevelMode::RipmapLevels => {
            let w = axis_extent(min_x, max_x)?;
            round_log2(w, tile_desc.rounding_mode) + 1
        }
        LevelMode::Invalid => {
            return Err(ExrError::Arg("Unknown LevelMode format.".into()));
        }
    };
    Ok(num)
}

/// Number of levels along Y — port of `calculateNumYLevels`.
///
/// * `ONE_LEVEL` → 1.
/// * `MIPMAP_LEVELS` → `roundLog2(max(w, h)) + 1` (same count as X).
/// * `RIPMAP_LEVELS` → `roundLog2(h) + 1` (independent Y reductions).
///
/// Returns [`ExrError::Arg`] for an unknown level mode, mirroring the C++ `ArgExc`.
pub fn num_y_levels(
    tile_desc: &TileDescription,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> Result<i32> {
    let num = match tile_desc.mode {
        LevelMode::OneLevel => 1,
        LevelMode::MipmapLevels => {
            let w = axis_extent(min_x, max_x)?;
            let h = axis_extent(min_y, max_y)?;
            round_log2(w.max(h), tile_desc.rounding_mode) + 1
        }
        LevelMode::RipmapLevels => {
            let h = axis_extent(min_y, max_y)?;
            round_log2(h, tile_desc.rounding_mode) + 1
        }
        LevelMode::Invalid => {
            return Err(ExrError::Arg("Unknown LevelMode format.".into()));
        }
    };
    Ok(num)
}

// ---------------------------------------------------------------------------
// calculateNumTiles — ImfTiledMisc.cpp anon namespace 254-269
// ---------------------------------------------------------------------------

/// Number of tiles per level along one axis — port of `calculateNumTiles`.
///
/// For each level `i` in `0..num_levels`, the tile count is
/// `ceil(levelSize / size)` computed as `(l + size - 1) / size` with the level
/// size widened to 64 bits (as in C++) to avoid overflow for large images.
/// Returns an owned `Vec<i32>` rather than filling a caller `int*`.
fn calculate_num_tiles(
    num_levels: i32,
    min: i32,
    max: i32,
    size: i32,
    rmode: LevelRoundingMode,
) -> Result<Vec<i32>> {
    let mut num_tiles = Vec::with_capacity(num_levels.max(0) as usize);
    for i in 0..num_levels {
        // use 64 bits to avoid int overflow if size is large.
        let l = level_size(min, max, i, rmode)? as u64;
        // (l + size - 1) / size in the C++; `div_ceil` is the same ceiling division.
        num_tiles.push(l.div_ceil(size as u64) as i32);
    }
    Ok(num_tiles)
}

// ---------------------------------------------------------------------------
// precalculateTileInfo — ImfTiledMisc.cpp lines 273-306
// ---------------------------------------------------------------------------

/// All level/tile counts for a data window — port of `precalculateTileInfo`.
///
/// Returns `(num_x_tiles, num_y_tiles, num_x_levels, num_y_levels)` where
/// `num_x_tiles[lx]` / `num_y_tiles[ly]` give the per-level tile counts.  The
/// C++ signature returns these through `int*&` out-parameters; here they are
/// owned `Vec<i32>`s so there is no manual allocation to clean up.
pub fn precalculate_tile_info(
    tile_desc: &TileDescription,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> Result<(Vec<i32>, Vec<i32>, i32, i32)> {
    let num_x_levels = num_x_levels(tile_desc, min_x, max_x, min_y, max_y)?;
    let num_y_levels = num_y_levels(tile_desc, min_x, max_x, min_y, max_y)?;

    let num_x_tiles = calculate_num_tiles(
        num_x_levels,
        min_x,
        max_x,
        tile_desc.x_size as i32,
        tile_desc.rounding_mode,
    )?;

    let num_y_tiles = calculate_num_tiles(
        num_y_levels,
        min_y,
        max_y,
        tile_desc.y_size as i32,
        tile_desc.rounding_mode,
    )?;

    Ok((num_x_tiles, num_y_tiles, num_x_levels, num_y_levels))
}

// ---------------------------------------------------------------------------
// getTiledChunkOffsetTableSize — ImfTiledMisc.cpp lines 308-392
// ---------------------------------------------------------------------------

/// Number of entries in a tiled chunk offset table — port of
/// `getTiledChunkOffsetTableSize`.
///
/// For `ONE_LEVEL`/`MIPMAP` it sums `numXTiles[i] * numYTiles[i]` over the
/// shared level index; for `RIPMAP` it sums `numXTiles[i] * numYTiles[j]` over
/// the full `(lx, ly)` grid.  The running total is checked against `i32::MAX`
/// exactly as the C++ does, returning [`ExrError::Logic`] on overflow.
pub fn get_tiled_chunk_offset_table_size(header: &Header) -> Result<i32> {
    let data_window = header.data_window()?;
    let tile_desc = header.tile_description()?;

    let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) = precalculate_tile_info(
        &tile_desc,
        data_window.min.x,
        data_window.max.x,
        data_window.min.y,
        data_window.max.y,
    )?;

    let mut line_offset_size: u64 = 0;

    match tile_desc.mode {
        LevelMode::OneLevel | LevelMode::MipmapLevels => {
            // Sum numXTiles[i] * numYTiles[i] over the shared level index.
            let nx = &num_x_tiles[..num_x_levels as usize];
            let ny = &num_y_tiles[..num_x_levels as usize];
            for (&xt, &yt) in nx.iter().zip(ny.iter()) {
                line_offset_size += xt as u64 * yt as u64;
                if line_offset_size > i32::MAX as u64 {
                    return Err(ExrError::Logic("Maximum number of tiles exceeded".into()));
                }
            }
        }
        LevelMode::RipmapLevels => {
            // Sum numXTiles[i] * numYTiles[j] over the full (lx, ly) grid.
            let nx = &num_x_tiles[..num_x_levels as usize];
            let ny = &num_y_tiles[..num_y_levels as usize];
            for &xt in nx {
                for &yt in ny {
                    line_offset_size += xt as u64 * yt as u64;
                    if line_offset_size > i32::MAX as u64 {
                        return Err(ExrError::Logic("Maximum number of tiles exceeded".into()));
                    }
                }
            }
        }
        LevelMode::Invalid => {
            return Err(ExrError::Logic(
                "Bad level mode getting chunk offset table size".into(),
            ));
        }
    }

    Ok(line_offset_size as i32)
}

// ---------------------------------------------------------------------------
// Tests — values are hand-derived from the C++ formulas (shown in comments)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn td(x: u32, y: u32, mode: LevelMode, rm: LevelRoundingMode) -> TileDescription {
        TileDescription {
            x_size: x,
            y_size: y,
            mode,
            rounding_mode: rm,
        }
    }

    // -- floor/ceil/round log2 -------------------------------------------------
    // floorLog2: counts halvings until <=1.
    //   1->0, 2->1, 3->1, 4->2, 255->7, 256->8.
    // ceilLog2: floor + (1 if any low bit dropped).
    //   1->0, 2->1, 3->2, 4->2, 255->8, 256->8.
    #[test]
    fn log2_helpers() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(4), 2);
        assert_eq!(floor_log2(255), 7);
        assert_eq!(floor_log2(256), 8);

        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(255), 8);
        assert_eq!(ceil_log2(256), 8);
    }

    // -- levelSize -------------------------------------------------------------
    // dataWindow 0..255 → a = 256. b = 2^l.
    //   l=0: 256/1=256; l=1:128; l=2:64; l=3:32; ...; l=8:256/256=1.
    // For a non-power-of-two like 0..254 → a=255:
    //   l=1 ROUND_DOWN: 255/2=127; ROUND_UP: 127*2=254 < 255 → 128.
    #[test]
    fn level_size_pow2_and_rounding() {
        assert_eq!(level_size(0, 255, 0, LevelRoundingMode::RoundDown).unwrap(), 256);
        assert_eq!(level_size(0, 255, 1, LevelRoundingMode::RoundDown).unwrap(), 128);
        assert_eq!(level_size(0, 255, 2, LevelRoundingMode::RoundDown).unwrap(), 64);
        assert_eq!(level_size(0, 255, 8, LevelRoundingMode::RoundDown).unwrap(), 1);

        // a = 255 (0..254): rounding diverges at l=1.
        assert_eq!(level_size(0, 254, 1, LevelRoundingMode::RoundDown).unwrap(), 127);
        assert_eq!(level_size(0, 254, 1, LevelRoundingMode::RoundUp).unwrap(), 128);

        // clamp to >= 1: huge l.
        assert_eq!(level_size(0, 255, 20, LevelRoundingMode::RoundDown).unwrap(), 1);
    }

    #[test]
    fn level_size_negative_level_errs() {
        assert!(level_size(0, 255, -1, LevelRoundingMode::RoundDown).is_err());
    }

    #[test]
    fn level_size_out_of_range_level_errs_no_panic() {
        // `1 << l` is computed in i64, so l up to 62 is valid (a
        // ROUND_UP image wider/taller than 2^30 needs level 31 — see
        // `level_31_reachable_for_round_up_giant_window` below). Only a level
        // that would overflow even the i64 shift is rejected.
        assert!(level_size(0, 255, 63, LevelRoundingMode::RoundDown).is_err());
        assert!(level_size(0, 255, 100, LevelRoundingMode::RoundDown).is_err());
        assert!(level_size(0, 255, -1, LevelRoundingMode::RoundDown).is_err());
        // l = 30 / 31 / 62 are all in range (clamp to 1 for this small window).
        assert_eq!(level_size(0, 255, 30, LevelRoundingMode::RoundDown).unwrap(), 1);
        assert_eq!(level_size(0, 255, 31, LevelRoundingMode::RoundDown).unwrap(), 1);
        assert_eq!(level_size(0, 255, 62, LevelRoundingMode::RoundDown).unwrap(), 1);
    }

    #[test]
    fn level_31_reachable_for_round_up_giant_window() {
        // Regression fixture: a ROUND_UP window of width 2^30+2
        // (min=-536870912, max=536870913) needs 32 levels (0..=31) because
        // `ceil_log2` of a dimension in (2^30, 2^31) is 31. C++ 3.4.15 opens
        // this fine (`OpenEXRCore/parse_header.c:2326-2338`, int64 shift);
        // before the fix, `level_size(.., 31, ..)` errored with
        // "Argument not in valid range", rejecting the header outright.
        let min = -536_870_912i32;
        let max = 536_870_913i32;
        assert_eq!(ceil_log2(axis_extent(min, max).unwrap()), 31);
        assert_eq!(level_size(min, max, 31, LevelRoundingMode::RoundUp).unwrap(), 1);
    }

    #[test]
    fn level_size_extreme_window_errs_no_panic() {
        // max - min + 1 overflows i32; the widened compute rejects the
        // unrepresentable size instead of panicking in debug / wrapping in release.
        assert!(level_size(i32::MIN, i32::MAX, 0, LevelRoundingMode::RoundDown).is_err());
    }

    #[test]
    fn data_window_for_tile_overflow_errs_no_panic() {
        // A huge tile coordinate makes `dx * xSize` overflow i32; reject cleanly.
        let t = td(64, 64, LevelMode::OneLevel, LevelRoundingMode::RoundDown);
        assert!(data_window_for_tile(&t, 0, 255, 0, 255, i32::MAX, 0, 0, 0).is_err());
    }

    #[test]
    fn num_levels_extreme_window_errs_no_panic() {
        // Public level-count helpers must reject an i32-overflowing window extent
        // (reached without Header::sanity_check) rather than panic/wrap.
        let t = td(64, 64, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown);
        assert!(num_x_levels(&t, i32::MIN, i32::MAX, 0, 255).is_err());
        assert!(num_y_levels(&t, 0, 255, i32::MIN, i32::MAX).is_err());
    }

    // -- ONE_LEVEL -------------------------------------------------------------
    // dataWindow 0..255 x 0..255, 64x64 tiles.
    //   numXLevels = numYLevels = 1.
    //   numXTiles[0] = ceil(256/64) = 4; numYTiles[0] = 4.
    #[test]
    fn one_level_counts() {
        let t = td(64, 64, LevelMode::OneLevel, LevelRoundingMode::RoundDown);
        assert_eq!(num_x_levels(&t, 0, 255, 0, 255).unwrap(), 1);
        assert_eq!(num_y_levels(&t, 0, 255, 0, 255).unwrap(), 1);

        let (nxt, nyt, nxl, nyl) = precalculate_tile_info(&t, 0, 255, 0, 255).unwrap();
        assert_eq!((nxl, nyl), (1, 1));
        assert_eq!(nxt, vec![4]);
        assert_eq!(nyt, vec![4]);
    }

    // -- MIPMAP ROUND_DOWN -----------------------------------------------------
    // 256x256 → max(w,h)=256 → roundLog2_down(256)=8 → 9 levels (X and Y).
    // levelSize per level (a=256): 256,128,64,32,16,8,4,2,1.
    // numTiles with 64-tile: ceil(size/64):
    //   256→4,128→2,64→1,32→1,16→1,8→1,4→1,2→1,1→1.
    #[test]
    fn mipmap_round_down() {
        let t = td(64, 64, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown);
        assert_eq!(num_x_levels(&t, 0, 255, 0, 255).unwrap(), 9);
        assert_eq!(num_y_levels(&t, 0, 255, 0, 255).unwrap(), 9);

        // level_width/height per level == levelSize on that axis.
        let widths: Vec<i32> = (0..9)
            .map(|l| level_size(0, 255, l, t.rounding_mode).unwrap())
            .collect();
        assert_eq!(widths, vec![256, 128, 64, 32, 16, 8, 4, 2, 1]);

        let (nxt, nyt, ..) = precalculate_tile_info(&t, 0, 255, 0, 255).unwrap();
        let expect = vec![4, 2, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(nxt, expect);
        assert_eq!(nyt, expect);
    }

    // -- MIPMAP ROUND_UP -------------------------------------------------------
    // Non-pow2 window 0..199 x 0..99 (w=200,h=100), 64-tiles.
    //   max(w,h)=200 → ceilLog2(200)=8 → 9 levels.
    //   X levelSize (a=200) ROUND_UP: l0=200, l1: 200/2=100 (exact) =100,
    //     l2: 200/4=50 =50, l3: 200/8=25 =25, l4:200/16=12,12*16=192<200→13,
    //     l5:200/32=6,6*32=192<200→7, l6:200/64=3,3*64=192<200→4,
    //     l7:200/128=1,1*128=128<200→2, l8:200/256=0→max(...,1)=1.
    //   → 200,100,50,25,13,7,4,2,1.
    #[test]
    fn mipmap_round_up_nonpow2() {
        let t = td(64, 64, LevelMode::MipmapLevels, LevelRoundingMode::RoundUp);
        assert_eq!(num_x_levels(&t, 0, 199, 0, 99).unwrap(), 9);
        assert_eq!(num_y_levels(&t, 0, 199, 0, 99).unwrap(), 9);

        let widths: Vec<i32> = (0..9)
            .map(|l| level_size(0, 199, l, t.rounding_mode).unwrap())
            .collect();
        assert_eq!(widths, vec![200, 100, 50, 25, 13, 7, 4, 2, 1]);

        // numXTiles (64-tile, ceil(size/64)): 200→4,100→2,50→1,25→1,13→1,...→1.
        let (nxt, ..) = precalculate_tile_info(&t, 0, 199, 0, 99).unwrap();
        assert_eq!(nxt, vec![4, 2, 1, 1, 1, 1, 1, 1, 1]);
    }

    // -- RIPMAP ----------------------------------------------------------------
    // 256x128 window (w=256,h=128), 64-tiles, ROUND_DOWN.
    //   numXLevels = roundLog2_down(256)+1 = 9.
    //   numYLevels = roundLog2_down(128)+1 = 8.
    //   X levelSize (a=256): 256,128,64,32,16,8,4,2,1 → tiles 4,2,1,1,1,1,1,1,1.
    //   Y levelSize (a=128): 128,64,32,16,8,4,2,1     → tiles 2,1,1,1,1,1,1,1.
    #[test]
    fn ripmap_counts() {
        let t = td(64, 64, LevelMode::RipmapLevels, LevelRoundingMode::RoundDown);
        assert_eq!(num_x_levels(&t, 0, 255, 0, 127).unwrap(), 9);
        assert_eq!(num_y_levels(&t, 0, 255, 0, 127).unwrap(), 8);

        let (nxt, nyt, nxl, nyl) = precalculate_tile_info(&t, 0, 255, 0, 127).unwrap();
        assert_eq!((nxl, nyl), (9, 8));
        assert_eq!(nxt, vec![4, 2, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(nyt, vec![2, 1, 1, 1, 1, 1, 1, 1]);
    }

    // -- dataWindowForLevel / dataWindowForTile --------------------------------
    // ONE_LEVEL 0..255 x 0..255, 64-tiles:
    //   level (0,0) window = 0..255 x 0..255.
    //   tile (0,0): min(0,0), max=min(0+64-1, 255)=63 → 0..63 x 0..63.
    //   tile (3,3): min(192,192), max=min(192+63,255)=255 → 192..255 x 192..255.
    //   tile (3,0) on a 200-wide level clamps the right edge (see below).
    #[test]
    fn data_window_for_level_and_tile_one_level() {
        let t = td(64, 64, LevelMode::OneLevel, LevelRoundingMode::RoundDown);

        let lvl = data_window_for_level(&t, 0, 255, 0, 255, 0, 0).unwrap();
        assert_eq!((lvl.min.x, lvl.min.y), (0, 0));
        assert_eq!((lvl.max.x, lvl.max.y), (255, 255));

        let t00 = data_window_for_tile(&t, 0, 255, 0, 255, 0, 0, 0, 0).unwrap();
        assert_eq!((t00.min.x, t00.min.y), (0, 0));
        assert_eq!((t00.max.x, t00.max.y), (63, 63));

        let t33 = data_window_for_tile(&t, 0, 255, 0, 255, 3, 3, 0, 0).unwrap();
        assert_eq!((t33.min.x, t33.min.y), (192, 192));
        assert_eq!((t33.max.x, t33.max.y), (255, 255));
    }

    // Edge-tile clamp on a non-multiple level:
    //   window 0..199 (w=200), 64-tiles → 4 tiles: 0..63,64..127,128..191,192..199.
    //   tile (3,0): min=192, raw max=192+63=255, clamped to level max 199.
    #[test]
    fn data_window_for_tile_edge_clamp() {
        let t = td(64, 64, LevelMode::OneLevel, LevelRoundingMode::RoundDown);
        let tile = data_window_for_tile(&t, 0, 199, 0, 199, 3, 0, 0, 0).unwrap();
        assert_eq!((tile.min.x, tile.min.y), (192, 0));
        assert_eq!((tile.max.x, tile.max.y), (199, 63));
    }
}
