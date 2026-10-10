//! Tiled image foundation — level/tile math and the chunk offset table.
//!
//! OpenEXR tiled files store the image as a grid of fixed-size tiles, optionally
//! across a MIP-map or RIP-map pyramid of resolution levels.  This module ports
//! the two pieces of machinery that everything tiled depends on, ported fresh
//! from the C++ OpenEXR source:
//!
//! * [`misc`] — `ImfTiledMisc.{cpp,h}`: the level/tile math (how many levels,
//!   how big each level is, how many tiles per level, the data window of a level
//!   or tile, the chunk-offset-table size). The rounding modes (`ROUND_DOWN`
//!   truncates, `ROUND_UP` rounds up) are reproduced bit-for-bit.
//! * [`tile_offsets`] — `ImfTileOffsets.{cpp,h}`: the `[level][dy][dx]` table of
//!   `u64` file offsets, with read/write/reconstruct and bounds-checked accessors.
//!
//! The tiled **writer** ([`output::TiledOutputFile`], `ImfTiledOutputFile`) is
//! implemented and the tiled **reader**
//! ([`input::TiledInputFile`], `ImfTiledInputFile`) is implemented; the
//! reader round-trips the writer pixel-exact.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/>

pub mod input;
pub mod misc;
pub mod output;
pub mod tile_offsets;

pub use input::{GenericTiledReader, TiledInputFile, TiledInputPartReader};
pub(crate) use input::TiledInputFileState;
pub use output::{
    TiledCopySource, TiledOutputFile, TiledOutputFileOwned, TiledOutputFileState,
    TiledOutputPartWriter,
};
pub use tile_offsets::TileOffsets;
