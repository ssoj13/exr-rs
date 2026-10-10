//! Port of OpenEXR `ImfTileOffsets.{cpp,h}` — the tiled chunk offset table.
//!
//! A tiled EXR stores, right after the header, a table of `u64` file offsets —
//! one per tile — that lets a reader seek directly to any tile.  This is the
//! tiled analogue of the scanline line-offset table.  The table is laid out as
//! a jagged 3-D array `_offsets[level][dy][dx]` whose **level** index depends
//! on the [`LevelMode`]:
//!
//! * `ONE_LEVEL` / `MIPMAP_LEVELS` — 1-D over levels: `_offsets` has
//!   `numXLevels` entries, indexed `_offsets[lx][dy][dx]` (for ONE_LEVEL only
//!   `lx == 0` exists; for MIPMAP `lx == ly`).
//! * `RIPMAP_LEVELS` — flattened 2-D: `_offsets` has `numXLevels * numYLevels`
//!   entries, indexed `_offsets[lx + ly * numXLevels][dy][dx]`.
//!
//! This mirrors the C++ `std::vector<std::vector<std::vector<uint64_t>>>`
//! exactly, including the flattening formula, so the on-disk byte order written
//! by [`TileOffsets::write_to`] is identical to OpenEXR's.
//!
//! Offsets are `u64` little-endian; tile coordinates in the per-tile chunk
//! headers are `i32`.  Both go through the container [`xdr`] helpers — we never
//! hand-roll endianness.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTileOffsets.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTileOffsets.h>

use crate::attr::tile_description::LevelMode;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

/// Tiled chunk offset table — port of `class TileOffsets`.
///
/// Holds the jagged offset array and the level mode/counts needed to index it.
/// See the module docs for the level-index layout.
#[derive(Clone, Debug)]
pub struct TileOffsets {
    /// Level mode controlling the indexing of `offsets` (`_mode`).
    mode: LevelMode,
    /// Number of X levels (`_numXLevels`).
    num_x_levels: i32,
    /// Number of Y levels (`_numYLevels`).
    num_y_levels: i32,
    /// Jagged offset array `[level][dy][dx]` (`_offsets`); see module docs.
    offsets: Vec<Vec<Vec<u64>>>,
}

impl Default for TileOffsets {
    /// Matches the C++ default ctor `TileOffsets(ONE_LEVEL, 0, 0, 0, 0)`:
    /// an empty table with a single (zero-row) level.
    fn default() -> Self {
        Self {
            mode: LevelMode::OneLevel,
            num_x_levels: 0,
            num_y_levels: 0,
            offsets: vec![Vec::new()],
        }
    }
}

impl TileOffsets {
    // -----------------------------------------------------------------------
    // Constructor — ImfTileOffsets.cpp lines 21-70
    // -----------------------------------------------------------------------

    /// Allocate the offset array for the given level mode and tile counts —
    /// port of the `TileOffsets` constructor.
    ///
    /// `num_x_tiles[lx]` / `num_y_tiles[ly]` are the per-level tile counts from
    /// [`crate::tiled::misc::precalculate_tile_info`].  For `ONE_LEVEL`/`MIPMAP`
    /// the array has `num_x_levels` rows sized `numYTiles[l] x numXTiles[l]`;
    /// for `RIPMAP` it has `num_x_levels * num_y_levels` rows sized
    /// `numYTiles[ly] x numXTiles[lx]`.  Every offset starts at 0.
    ///
    /// Returns [`ExrError::Arg`] for an invalid (`NUM_LEVELMODES`) mode, matching
    /// the C++ `ArgExc`.
    pub fn new(
        mode: LevelMode,
        num_x_levels: i32,
        num_y_levels: i32,
        num_x_tiles: &[i32],
        num_y_tiles: &[i32],
    ) -> Result<Self> {
        let mut offsets: Vec<Vec<Vec<u64>>> = Vec::new();

        match mode {
            LevelMode::OneLevel | LevelMode::MipmapLevels => {
                offsets.resize(num_x_levels.max(0) as usize, Vec::new());
                for (l, level) in offsets.iter_mut().enumerate() {
                    let ny = *num_y_tiles
                        .get(l)
                        .ok_or_else(|| ExrError::Arg("numYTiles index out of range".into()))?;
                    let nx = *num_x_tiles
                        .get(l)
                        .ok_or_else(|| ExrError::Arg("numXTiles index out of range".into()))?;
                    *level = vec![vec![0u64; nx.max(0) as usize]; ny.max(0) as usize];
                }
            }
            LevelMode::RipmapLevels => {
                // i64 product: a corrupt/oversized level count would overflow the
                // i32 `numXLevels * numYLevels`; widen then range-check.
                let level_count = num_x_levels.max(0) as i64 * num_y_levels.max(0) as i64;
                offsets.resize(
                    usize::try_from(level_count)
                        .map_err(|_| ExrError::Arg("Tile level count out of range".into()))?,
                    Vec::new(),
                );
                for ly in 0..num_y_levels {
                    for lx in 0..num_x_levels {
                        let l = (ly as i64 * num_x_levels as i64 + lx as i64) as usize;
                        let ny = *num_y_tiles.get(ly as usize).ok_or_else(|| {
                            ExrError::Arg("numYTiles index out of range".into())
                        })?;
                        let nx = *num_x_tiles.get(lx as usize).ok_or_else(|| {
                            ExrError::Arg("numXTiles index out of range".into())
                        })?;
                        offsets[l] = vec![vec![0u64; nx.max(0) as usize]; ny.max(0) as usize];
                    }
                }
            }
            LevelMode::Invalid => {
                return Err(ExrError::Arg(
                    "Bad initialisation of TileOffsets object".into(),
                ));
            }
        }

        Ok(Self {
            mode,
            num_x_levels,
            num_y_levels,
            offsets,
        })
    }

    // -----------------------------------------------------------------------
    // anyOffsetsAreInvalid — ImfTileOffsets.cpp lines 72-81
    // -----------------------------------------------------------------------

    /// `true` if any offset is 0 (i.e. unwritten) — port of `anyOffsetsAreInvalid`.
    ///
    /// The C++ test is `_offsets[l][dy][dx] <= 0`; offsets are `uint64_t`, so
    /// `<= 0` is exactly `== 0` here.  A zero offset means the table was never
    /// fully written (the file is likely incomplete).
    pub fn any_offsets_are_invalid(&self) -> bool {
        for level in &self.offsets {
            for row in level {
                for &off in row {
                    if off == 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // isEmpty — ImfTileOffsets.cpp lines 394-402
    // -----------------------------------------------------------------------

    /// `true` if every offset is 0 — port of `isEmpty`.
    pub fn is_empty(&self) -> bool {
        for level in &self.offsets {
            for row in level {
                for &off in row {
                    if off != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // isValidTile — ImfTileOffsets.cpp lines 404-449
    // -----------------------------------------------------------------------

    /// Whether `(dx, dy, lx, ly)` addresses an existing tile — port of `isValidTile`.
    ///
    /// Bounds-checks the coordinates against the jagged array for the current
    /// level mode.  Used by the shared reconstruction sweep
    /// ([`crate::chunk::offsets::reconstruct`]) to reject garbage coordinates
    /// read from a corrupt/incomplete file before indexing.
    pub fn is_valid_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> bool {
        if lx < 0 || ly < 0 || dx < 0 || dy < 0 {
            return false;
        }

        match self.mode {
            LevelMode::OneLevel => {
                lx == 0
                    && ly == 0
                    && !self.offsets.is_empty()
                    && self.in_row_bounds(0, dy, dx)
            }
            LevelMode::MipmapLevels => {
                // MIPMAP tiles always have
                // lx == ly; the only real caller of this function is the
                // shared reconstruction sweep's slot oracle
                // (`chunk::offsets::tile_slot`), which validates a chunk
                // leader read off disk — a leader can claim any `(lx, ly)`.
                // C++ 3.4.15's own reconstruction oracle,
                // `OpenEXRCore/chunk.c:62-76`
                // (`validate_and_compute_tile_chunk_off`), rejects `lx !=
                // ly` for ONE_LEVEL and MIPMAP before ever indexing a slot;
                // without this a bogus `(0,0,1,0)` leader was accepted into
                // the level-1 slot for (0,0), silently dropping the genuine
                // `(0,0,1,1)` chunk that followed it (first-wins sweep).
                lx == ly
                    && lx < self.num_x_levels
                    && ly < self.num_y_levels
                    && (lx as usize) < self.offsets.len()
                    && self.in_row_bounds(lx as usize, dy, dx)
            }
            LevelMode::RipmapLevels => {
                if lx < self.num_x_levels && ly < self.num_y_levels {
                    let l = lx as i64 + ly as i64 * self.num_x_levels as i64;
                    l >= 0
                        && (l as usize) < self.offsets.len()
                        && self.in_row_bounds(l as usize, dy, dx)
                } else {
                    false
                }
            }
            LevelMode::Invalid => false,
        }
    }

    /// Helper: is `(dy, dx)` within `offsets[level]`'s 2-D extent?
    ///
    /// Mirrors the C++ checks `int(_offsets[l].size()) > dy &&
    /// int(_offsets[l][dy].size()) > dx`.
    fn in_row_bounds(&self, level: usize, dy: i32, dx: i32) -> bool {
        match self.offsets.get(level).and_then(|rows| rows.get(dy as usize)) {
            Some(row) => (dx as usize) < row.len(),
            None => false,
        }
    }

    // -----------------------------------------------------------------------
    // Flattened level index — the C++ `operator()` switch (lines 451-503)
    // -----------------------------------------------------------------------

    /// Map `(lx, ly)` to the flat `offsets` level index for the current mode.
    ///
    /// Mirrors the C++ `operator()` switch:
    /// * `ONE_LEVEL` → 0
    /// * `MIPMAP_LEVELS` → `lx`
    /// * `RIPMAP_LEVELS` → `lx + ly * numXLevels`
    ///
    /// Returns [`ExrError::Arg`] for an invalid mode (C++ `ArgExc`).
    fn level_index(&self, lx: i32, ly: i32) -> Result<usize> {
        // i64: the RIPMAP `lx + ly * numXLevels` product can overflow i32 for a
        // corrupt/oversized level count; widen then range-check.
        let idx: i64 = match self.mode {
            LevelMode::OneLevel => 0,
            LevelMode::MipmapLevels => lx as i64,
            LevelMode::RipmapLevels => lx as i64 + ly as i64 * self.num_x_levels as i64,
            LevelMode::Invalid => {
                return Err(ExrError::Arg("Unknown LevelMode format.".into()));
            }
        };
        if idx < 0 {
            return Err(ExrError::Arg("Negative tile level index".into()));
        }
        usize::try_from(idx).map_err(|_| ExrError::Arg("Tile level index out of range".into()))
    }

    /// Immutable access to offset `(dx, dy, lx, ly)` — port of the const
    /// `operator() (dx, dy, lx, ly)`.
    ///
    /// Bounds-checked (the C++ indexes unchecked); out-of-range yields
    /// [`ExrError::Arg`].
    pub fn get(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<u64> {
        let l = self.level_index(lx, ly)?;
        self.offsets
            .get(l)
            .and_then(|rows| rows.get(dy as usize))
            .and_then(|row| row.get(dx as usize))
            .copied()
            .ok_or_else(|| ExrError::Arg("Tile offset index out of range".into()))
    }

    /// Mutable access to offset `(dx, dy, lx, ly)` — port of the non-const
    /// `operator() (dx, dy, lx, ly)`.
    ///
    /// Bounds-checked; out-of-range yields [`ExrError::Arg`].
    pub fn get_mut(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<&mut u64> {
        let l = self.level_index(lx, ly)?;
        self.offsets
            .get_mut(l)
            .and_then(|rows| rows.get_mut(dy as usize))
            .and_then(|row| row.get_mut(dx as usize))
            .ok_or_else(|| ExrError::Arg("Tile offset index out of range".into()))
    }

    /// Immutable access with a single level number — port of the const
    /// `operator() (dx, dy, l)`, which forwards to `(dx, dy, l, l)`.
    pub fn get_l(&self, dx: i32, dy: i32, l: i32) -> Result<u64> {
        self.get(dx, dy, l, l)
    }

    /// Mutable access with a single level number — port of the non-const
    /// `operator() (dx, dy, l)`, which forwards to `(dx, dy, l, l)`.
    pub fn get_l_mut(&mut self, dx: i32, dy: i32, l: i32) -> Result<&mut u64> {
        self.get_mut(dx, dy, l, l)
    }

    /// Borrow the raw jagged offset array — port of `getOffsets`.
    pub fn get_offsets(&self) -> &Vec<Vec<Vec<u64>>> {
        &self.offsets
    }

    /// Linear slot index of tile `(dx, dy, lx, ly)` in [`write_to`](Self::write_to)
    /// order (`[level][dy][dx]`).
    ///
    /// Not a C++ method: the C++ writer rewrites the whole table from its
    /// destructor, but our borrowed-stream writer back-patches one slot at a
    /// time, so it needs the `u64` offset of an individual slot.  The returned
    /// index times `8` (the `u64` size) gives the byte offset of the slot
    /// relative to the table start ([`write_to`](Self::write_to)'s return).
    ///
    /// Counts every slot emitted before `(dx, dy, lx, ly)`: all slots in the
    /// preceding levels, then the preceding rows of this level, then `dx`.
    /// Bounds-checked; out-of-range yields [`ExrError::Arg`].
    pub fn slot_index(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<usize> {
        let level = self.level_index(lx, ly)?;
        let rows = self
            .offsets
            .get(level)
            .ok_or_else(|| ExrError::Arg("Tile offset index out of range".into()))?;

        let dy_us =
            usize::try_from(dy).map_err(|_| ExrError::Arg("Negative tile dy".into()))?;
        let dx_us =
            usize::try_from(dx).map_err(|_| ExrError::Arg("Negative tile dx".into()))?;

        // The target row must exist and contain dx.
        let target_row = rows
            .get(dy_us)
            .ok_or_else(|| ExrError::Arg("Tile offset index out of range".into()))?;
        if dx_us >= target_row.len() {
            return Err(ExrError::Arg("Tile offset index out of range".into()));
        }

        // Slots in all levels before this one.
        let mut idx = 0usize;
        for prev_level in &self.offsets[..level] {
            for row in prev_level {
                idx += row.len();
            }
        }
        // Slots in the preceding rows of this level.
        for row in &rows[..dy_us] {
            idx += row.len();
        }
        // Slots in this row before dx.
        idx += dx_us;

        Ok(idx)
    }

    // -----------------------------------------------------------------------
    // writeTo — ImfTileOffsets.cpp lines 268-291
    // -----------------------------------------------------------------------

    /// Write the whole offset table as `u64` LE values — port of `writeTo`.
    ///
    /// Returns the stream position where the table started (so the writer can
    /// back-patch it after the tiles are written), matching the C++ return.
    /// Offsets are emitted in `[level][dy][dx]` order via [`xdr::write_u64`].
    pub fn write_to(&self, os: &mut dyn OStream) -> Result<u64> {
        let pos = os.tellp()?;

        for level in &self.offsets {
            for row in level {
                for &off in row {
                    xdr::write_u64(os, off)?;
                }
            }
        }

        Ok(pos)
    }

    // -----------------------------------------------------------------------
    // readFrom (stream) — ImfTileOffsets.cpp lines 203-241
    // -----------------------------------------------------------------------

    /// Number of offset slots in the table (its flat length).
    pub fn slot_count(&self) -> usize {
        self.offsets
            .iter()
            .flat_map(|level| level.iter())
            .map(|row| row.len())
            .sum()
    }

    /// Read the offset table from a stream — port of the stream `readFrom`.
    ///
    /// Delegates the raw read to the ONE checked offset-table reader
    /// ([`crate::chunk::offsets::read_offset_table`]), which caps the entry
    /// count and range-checks every offset against the file before anything is
    /// allocated (C++ `OpenEXRCore/chunk.c::extract_chunk_table`).  If the
    /// table is incomplete — any offset 0 or outside the chunk region —
    /// `complete` is set `false` and the table is rebuilt by scanning the
    /// chunks ([`reconstruct_from_file`](Self::reconstruct_from_file)).
    pub fn read_from(
        &mut self,
        is: &mut dyn IStream,
        complete: &mut bool,
        is_multi_part: bool,
        is_deep: bool,
    ) -> Result<()> {
        let count = i64::try_from(self.slot_count())
            .map_err(|_| ExrError::Input("Tile chunk count out of range".into()))?;
        let table = crate::chunk::offsets::read_offset_table(is, count)?;
        self.read_from_vec(&table.offsets, complete)?;

        if !table.complete || !*complete {
            *complete = false;
            self.reconstruct_from_file(is, is_multi_part, is_deep)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // readFrom (vector) — ImfTileOffsets.cpp lines 243-266
    // -----------------------------------------------------------------------

    /// Populate the table from a flat slice of offsets — port of the
    /// `readFrom(std::vector<uint64_t>, bool&)` overload (used by the multipart
    /// reader, which collects all chunk offsets up front).
    ///
    /// The slice must hold exactly one entry per tile slot, in `[level][dy][dx]`
    /// order; a length mismatch is [`ExrError::Arg`] (C++ `ArgExc`).  `complete`
    /// is `!any_offsets_are_invalid()`.
    pub fn read_from_vec(&mut self, chunk_offsets: &[u64], complete: &mut bool) -> Result<()> {
        let mut total_size = 0usize;
        for level in &self.offsets {
            for row in level {
                total_size += row.len();
            }
        }

        if chunk_offsets.len() != total_size {
            return Err(ExrError::Arg(
                "Wrong offset count, not able to read from this array".into(),
            ));
        }

        let mut pos = 0usize;
        for level in &mut self.offsets {
            for row in level {
                for off in row {
                    // `pos` cannot exceed `total_size` by construction above.
                    *off = chunk_offsets[pos];
                    pos += 1;
                }
            }
        }

        *complete = !self.any_offsets_are_invalid();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // reconstructFromFile — ImfTileOffsets.cpp lines 174-201
    // -----------------------------------------------------------------------

    /// Rebuild a missing/corrupt table by scanning the file — port of
    /// `reconstructFromFile` / `findTiles`.
    ///
    /// The scan itself is the **shared** chunk sweep in
    /// [`crate::chunk::offsets::reconstruct`] (one codepath for scanline, deep
    /// scanline, tiled and deep tiled): it walks the chunks from the current
    /// stream position, maps each tile chunk leader back to its slot through
    /// this table's geometry (`is_valid_tile` + `slot_index`), swallows a
    /// mid-scan fault (incomplete files are expected to fault — C++
    /// `catch (...) {}`), and restores the stream position.
    pub fn reconstruct_from_file(
        &mut self,
        is: &mut dyn IStream,
        is_multi_part: bool,
        is_deep: bool,
    ) -> Result<()> {
        use crate::chunk::offsets::{reconstruct, Framing, PartOffsets};

        let first_chunk_pos = is.tellg()?;

        // Flat slot-order view of the table; `self` is cloned as the geometry
        // oracle the sweep uses to turn (dx, dy, lx, ly) into a slot index.
        let mut flat = self.flatten();
        let geometry = self.clone();
        let framing = if is_deep {
            Framing::DeepTile(geometry)
        } else {
            Framing::Tile(geometry)
        };

        reconstruct(
            is,
            &mut [PartOffsets {
                framing,
                offsets: &mut flat,
            }],
            is_multi_part,
            first_chunk_pos,
        )?;

        let mut complete = false;
        self.read_from_vec(&flat, &mut complete)?;
        Ok(())
    }

    /// The table's offsets in flat `[level][dy][dx]` slot order — the shape the
    /// shared offset-table reader / reconstruction sweep works in.
    fn flatten(&self) -> Vec<u64> {
        self.offsets
            .iter()
            .flat_map(|level| level.iter())
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    // -----------------------------------------------------------------------
    // getTileOrder — ImfTileOffsets.cpp lines 293-392
    // -----------------------------------------------------------------------

    /// Fill `(dx, dy, lx, ly)` tables with tile coordinates in file order —
    /// port of `getTileOrder`.
    ///
    /// Sorts all `(filePos, dx, dy, l)` records by file position, then writes
    /// the `dx`/`dy` (level-independent) and the per-mode `lx`/`ly`:
    /// * `ONE_LEVEL` → `0, 0`
    /// * `MIPMAP_LEVELS` → `l, l`
    /// * `RIPMAP_LEVELS` → `l % numXLevels, l / numXLevels`
    ///
    /// Each output slice must be at least `total_tiles` long; a short slice is
    /// [`ExrError::Arg`].  Assumes a fully populated table (as the C++ does).
    pub fn get_tile_order(
        &self,
        dx_table: &mut [i32],
        dy_table: &mut [i32],
        lx_table: &mut [i32],
        ly_table: &mut [i32],
    ) -> Result<()> {
        /// One `(filePos, dx, dy, l)` record, sorted by `file_pos`.
        struct TilePos {
            file_pos: u64,
            dx: i32,
            dy: i32,
            l: i32,
        }

        let mut table: Vec<TilePos> = Vec::new();
        for (l, level) in self.offsets.iter().enumerate() {
            for (dy, row) in level.iter().enumerate() {
                for (dx, &file_pos) in row.iter().enumerate() {
                    table.push(TilePos {
                        file_pos,
                        dx: dx as i32,
                        dy: dy as i32,
                        l: l as i32,
                    });
                }
            }
        }

        let entries = table.len();
        if dx_table.len() < entries
            || dy_table.len() < entries
            || lx_table.len() < entries
            || ly_table.len() < entries
        {
            return Err(ExrError::Arg(
                "tile order output array too small".into(),
            ));
        }

        table.sort_by_key(|t| t.file_pos);

        // pass 1: dx and dy, independent of level mode.
        for (i, t) in table.iter().enumerate() {
            dx_table[i] = t.dx;
            dy_table[i] = t.dy;
        }

        // pass 2: levels, which depend on the level mode.
        match self.mode {
            LevelMode::OneLevel => {
                for i in 0..entries {
                    lx_table[i] = 0;
                    ly_table[i] = 0;
                }
            }
            LevelMode::MipmapLevels => {
                for (i, t) in table.iter().enumerate() {
                    lx_table[i] = t.l;
                    ly_table[i] = t.l;
                }
            }
            LevelMode::RipmapLevels => {
                for (i, t) in table.iter().enumerate() {
                    lx_table[i] = t.l % self.num_x_levels;
                    ly_table[i] = t.l / self.num_x_levels;
                }
            }
            LevelMode::Invalid => {
                return Err(ExrError::Logic("Bad level mode getting tile order".into()));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    // ONE_LEVEL 256x256, 64-tiles → 1 level, 4x4 tiles (from misc tests).
    fn one_level_4x4() -> TileOffsets {
        TileOffsets::new(LevelMode::OneLevel, 1, 1, &[4], &[4]).unwrap()
    }

    #[test]
    fn shape_one_level() {
        let t = one_level_4x4();
        // 1 level, 4 rows (dy), 4 cols (dx).
        assert_eq!(t.get_offsets().len(), 1);
        assert_eq!(t.get_offsets()[0].len(), 4);
        assert_eq!(t.get_offsets()[0][0].len(), 4);
        assert!(t.is_empty());
        assert!(t.any_offsets_are_invalid());
    }

    #[test]
    fn shape_mipmap() {
        // 256x256 mipmap round-down: 9 levels, tiles 4,2,1,1,1,1,1,1,1.
        let nxt = vec![4, 2, 1, 1, 1, 1, 1, 1, 1];
        let nyt = nxt.clone();
        let t = TileOffsets::new(LevelMode::MipmapLevels, 9, 9, &nxt, &nyt).unwrap();
        assert_eq!(t.get_offsets().len(), 9);
        // level 0 is 4x4, level 1 is 2x2, level 2.. are 1x1.
        assert_eq!(t.get_offsets()[0].len(), 4);
        assert_eq!(t.get_offsets()[0][0].len(), 4);
        assert_eq!(t.get_offsets()[1].len(), 2);
        assert_eq!(t.get_offsets()[1][0].len(), 2);
        assert_eq!(t.get_offsets()[2].len(), 1);
    }

    #[test]
    fn shape_ripmap_flat_index() {
        // 256x128 ripmap round-down: numXLevels=9, numYLevels=8.
        // numXTiles = 4,2,1,1,1,1,1,1,1 ; numYTiles = 2,1,1,1,1,1,1,1.
        let nxt = vec![4, 2, 1, 1, 1, 1, 1, 1, 1];
        let nyt = vec![2, 1, 1, 1, 1, 1, 1, 1];
        let t = TileOffsets::new(LevelMode::RipmapLevels, 9, 8, &nxt, &nyt).unwrap();
        // flat size = 9 * 8 = 72.
        assert_eq!(t.get_offsets().len(), 72);
        // level (lx=1, ly=0) → flat 1: numYTiles[0]=2 rows, numXTiles[1]=2 cols.
        assert_eq!(t.get_offsets()[1].len(), 2);
        assert_eq!(t.get_offsets()[1][0].len(), 2);
        // level (lx=0, ly=1) → flat 0 + 1*9 = 9: numYTiles[1]=1 row, numXTiles[0]=4 cols.
        assert_eq!(t.get_offsets()[9].len(), 1);
        assert_eq!(t.get_offsets()[9][0].len(), 4);
    }

    #[test]
    fn get_set_one_level() {
        let mut t = one_level_4x4();
        *t.get_mut(3, 2, 0, 0).unwrap() = 0xDEAD_BEEF;
        assert_eq!(t.get(3, 2, 0, 0).unwrap(), 0xDEAD_BEEF);
        assert!(!t.is_empty());
        // out of range → Err, not panic.
        assert!(t.get(4, 0, 0, 0).is_err());
        assert!(t.get_mut(0, 4, 0, 0).is_err());
    }

    #[test]
    fn ripmap_flat_index_matches_formula() {
        let nxt = vec![4, 2, 1, 1, 1, 1, 1, 1, 1];
        let nyt = vec![2, 1, 1, 1, 1, 1, 1, 1];
        let mut t = TileOffsets::new(LevelMode::RipmapLevels, 9, 8, &nxt, &nyt).unwrap();
        // tile (dx=1, dy=0) at level (lx=1, ly=0) → flat level 1, row 0, col 1.
        *t.get_mut(1, 0, 1, 0).unwrap() = 42;
        assert_eq!(t.get_offsets()[1][0][1], 42);
        // tile (dx=0, dy=0) at level (lx=0, ly=1) → flat level 9.
        *t.get_mut(0, 0, 0, 1).unwrap() = 99;
        assert_eq!(t.get_offsets()[9][0][0], 99);
    }

    #[test]
    fn write_read_round_trip() {
        let mut t = one_level_4x4();
        // Fill with offsets that actually address the chunk region: the table
        // occupies bytes 0..128, so a "complete" table's entries must all lie
        // in [128, file_size) — `read_from` range-checks them (C++
        // `chunk.c::extract_chunk_table`), so 1..16 would now (correctly) read
        // back as an INCOMPLETE table.
        const TABLE_BYTES: u64 = 16 * 8;
        let mut v = TABLE_BYTES;
        for dy in 0..4 {
            for dx in 0..4 {
                *t.get_mut(dx, dy, 0, 0).unwrap() = v;
                v += 8;
            }
        }

        let mut os = MemOStream::new();
        let start = t.write_to(&mut os).unwrap();
        assert_eq!(start, 0);
        let mut bytes = os.into_data();
        // 16 tiles * 8 bytes each.
        assert_eq!(bytes.len(), TABLE_BYTES as usize);
        assert_eq!(&bytes[0..8], &TABLE_BYTES.to_le_bytes());
        // Pad out the "chunk region" the offsets point into.
        bytes.resize(TABLE_BYTES as usize + 16 * 8, 0u8);

        let mut t2 = one_level_4x4();
        let mut is = MemIStream::new(bytes);
        let mut complete = false;
        t2.read_from(&mut is, &mut complete, false, false).unwrap();
        assert!(complete);
        for dy in 0..4 {
            for dx in 0..4 {
                assert_eq!(t.get(dx, dy, 0, 0).unwrap(), t2.get(dx, dy, 0, 0).unwrap());
            }
        }
    }

    #[test]
    fn read_from_vec_length_check() {
        let mut t = one_level_4x4();
        let mut complete = false;
        // 16 slots; wrong length errors.
        assert!(t.read_from_vec(&[1, 2, 3], &mut complete).is_err());
        let all: Vec<u64> = (1..=16).collect();
        t.read_from_vec(&all, &mut complete).unwrap();
        assert!(complete);
        assert_eq!(t.get(0, 0, 0, 0).unwrap(), 1);
        assert_eq!(t.get(3, 3, 0, 0).unwrap(), 16);
    }

    #[test]
    fn is_valid_tile_bounds() {
        let t = one_level_4x4();
        assert!(t.is_valid_tile(0, 0, 0, 0));
        assert!(t.is_valid_tile(3, 3, 0, 0));
        assert!(!t.is_valid_tile(4, 0, 0, 0)); // dx out of range
        assert!(!t.is_valid_tile(0, 0, 1, 0)); // lx != 0 for ONE_LEVEL
        assert!(!t.is_valid_tile(-1, 0, 0, 0)); // negative
    }

    /// A MIPMAP leader with `lx != ly` must be
    /// rejected by `is_valid_tile`, the reconstruction sweep's slot oracle
    /// (`chunk::offsets::tile_slot`). Before the fix a bogus `(0, 0, 1, 0)`
    /// leader was accepted into the level-1 (lx=1) slot for tile (0,0),
    /// which would silently displace the genuine `(0, 0, 1, 1)` chunk in a
    /// first-wins reconstruction sweep. C++ 3.4.15's own reconstruction
    /// oracle, `OpenEXRCore/chunk.c:62-76`
    /// (`validate_and_compute_tile_chunk_off`), requires `levelx == levely`
    /// for ONE_LEVEL and MIPMAP before ever computing a slot.
    #[test]
    fn is_valid_tile_mipmap_rejects_lx_ne_ly() {
        let t = TileOffsets::new(LevelMode::MipmapLevels, 2, 2, &[2, 1], &[2, 1]).unwrap();
        assert!(t.is_valid_tile(0, 0, 0, 0)); // level 0, in range
        assert!(t.is_valid_tile(0, 0, 1, 1)); // level 1, lx == ly
        assert!(!t.is_valid_tile(0, 0, 1, 0)); // lx != ly: not a real MIPMAP tile
        assert!(!t.is_valid_tile(0, 0, 0, 1)); // lx != ly, other direction
    }
}
