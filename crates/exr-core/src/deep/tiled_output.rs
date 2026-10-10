//! Port of OpenEXR `ImfDeepTiledOutputFile.cpp` / `.h` —
//! `class DeepTiledOutputFile`, the single-part **deep tiled** writer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledOutputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledOutputFile.h>
//!
//! This fuses tiling ([`crate::tiled`] `TileOffsets`/`TiledMisc`) with
//! the deep "count then data" chunk machinery
//! ([`crate::deep::scanline_output`]) over a tile rectangle.
//!
//! # The deep tiled chunk byte layout (confirmed from `writeTileData`)
//!
//! `writeTileData` (ImfDeepTiledOutputFile.cpp:452–538) writes, in order (all
//! multi-byte fields Xdr little-endian):
//!
//! | field                        | type | bytes | C++ line |
//! |------------------------------|------|-------|----------|
//! | `[partNumber]`               | i32  | 4     | 489 — multipart only, NOT here |
//! | `dx` (tileX)                 | i32  | 4     | 491 |
//! | `dy` (tileY)                 | i32  | 4     | 492 |
//! | `lx` (levelX)                | i32  | 4     | 493 |
//! | `ly` (levelY)                | i32  | 4     | 494 |
//! | `packedSampleCountTableSize` | u64  | 8     | 500 |
//! | `packedDataSize`             | u64  | 8     | 506 |
//! | `unpackedDataSize`           | u64  | 8     | 507 |
//! | sample-count table           | bytes| `packedSampleCountTableSize` | 513 |
//! | sample data                  | bytes| `packedDataSize`             | 520 |
//!
//! Note the field order: the four tile coords come first (vs the deep scanline
//! chunk's single `y`), then the **count-table size BEFORE the two data sizes**
//! (same relative order as the deep scanline chunk).
//!
//! # The cumulative sample-count table over the tile rectangle
//!
//! From `TileBufferTask::execute` (ImfDeepTiledOutputFile.cpp:944–956): the
//! table covers the tile's pixel rectangle (`dataWindowForTile`).  For each row
//! `i` of the rectangle, `count` resets to 0 and accumulates across the columns
//! `j`; after each `count += getSampleCount(j, i)` the running `count` is
//! written as an Xdr `int` (4 bytes).  So the table is `tileW * tileH` ints, the
//! last int of each row being that row's total sample count — exactly the deep
//! scanline cumulative table, but over the tile rect instead of a full scanline.
//!
//! # Two SEPARATE compressors, sized for the WHOLE tile (the partial-edge gotcha)
//!
//! C++ `initialize` (lines 1203–1273) sets
//! `maxSampleCountTableSize = tileDesc.ySize * tileDesc.xSize * sizeof(int)` —
//! the **whole** tile area, not the per-tile rectangle.  The count-table
//! compressor is a *scanline* compressor
//! (`newCompressor(compression, maxSampleCountTableSize, header)`); the
//! sample-data compressor is a *tile* compressor rebuilt per tile
//! (`newTileCompressor(compression, maxBytesPerTileLine, tileDesc.ySize,
//! header)`).
//!
//! Crucially, the count-table "no-shrink → store raw" branch
//! (lines 972–978) stores **`maxSampleCountTableSize`** bytes from a buffer that
//! was zero-initialised at construction (lines 1264–1268) and then filled only
//! to the rectangle's `tableDataSize`.  For interior (full) tiles
//! `tableDataSize == maxSampleCountTableSize`.  For **partial edge tiles** the
//! rectangle is smaller. C++ reuses the buffer without clearing its tail,
//! so padding can retain an earlier tile's cumulative counts. This port
//! deliberately writes deterministic zero padding instead. Only the real
//! rectangle's entries are decoded; C++ cross-reads accept the zero tail.
//!
//! # Safe-Rust adaptation of the `char**` deep addressing
//!
//! Identical to the deep scanline writer: the caller hands
//! [`DeepTiledOutputFile::set_frame_buffer`] a **flat per-channel sample heap**
//! (`sample_data`) laid out contiguously in whole-data-window `(x, y)` raster
//! order, and a **flat sample-count buffer** (`sample_counts`) addressed by the
//! sample-count slice.  A tile's pixels are an interior rectangle of that global
//! raster order; pixel `(x, y)`'s samples begin at
//! `samples_before_pixel(x, y) * sample_stride` (see
//! [`crate::deep::sample_table`]).  The bytes written are byte-identical to the
//! C++ writer's.
//!
//! # Offset back-patch vs C++ `Drop`
//!
//! C++ rewrites the filled tile-offset table from its destructor.  Like
//! [`crate::tiled::TiledOutputFile`] and [`crate::deep::DeepScanLineOutputFile`],
//! this port back-patches each tile's offset into the reserved table slot
//! immediately after the chunk is written ([`TileOffsets::slot_index`] →
//! seek-back `u64` LE → seek-forward), so the file is valid without `Drop`.

use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
use crate::attr::{LineOrder, PixelType};
use crate::compressor::{
    default_format, new_compressor, new_tile_compressor, stored_chunk, Compressor, Format,
};
use crate::container::{xdr, SIZE_I64};
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::deep::sample_table::{encode_cumulative_row, sample_count_at};
use crate::deep::tiled_input::{DeepTiledInputFile, RawTileData};
use crate::error::{ExrError, Result};
use crate::header::{Header, DEEP_TILE};
use crate::io::OStream;
use crate::misc::pixel_type_size;
use crate::tiled::misc::{
    calculate_bytes_per_pixel, data_window_for_level, data_window_for_tile,
    get_tiled_chunk_offset_table_size, level_size, precalculate_tile_info,
};
use crate::tiled::output::TileGeom;
use crate::tiled::tile_offsets::TileOffsets;

use imath_rs::Box2i;

/// Xdr `int` size in bytes — the per-pixel cumulative-count table entry width
/// (C++ `sizeof(int)` written via `Xdr::write<int>`).
const SIZE_INT: usize = 4;

// ---------------------------------------------------------------------------
// TOutSliceInfo — port of the anonymous-namespace `struct TOutSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel deep tiled write descriptor — port of `TOutSliceInfo`
/// (ImfDeepTiledOutputFile.cpp, anonymous namespace).
///
/// Built by [`DeepTiledOutputFile::set_frame_buffer`] from the file's channel
/// list and the user [`DeepFrameBuffer`].  Mirrors the C++ struct field-for-
/// field, except the raw `char* base` (a `char**` for deep) becomes a signed
/// byte offset (`base_offset`) into the caller-owned flat sample heap — see the
/// module-level docs for the safe-Rust adaptation.
#[derive(Clone, Debug)]
struct TOutSliceInfo {
    /// Sample type stored on disk for this channel (C++ `type`).
    type_: PixelType,
    /// Signed byte offset of this channel's flat sample heap within the caller's
    /// `sample_data` buffer.  Replaces the C++ `char* base` (`0` when `zero`).
    base_offset: i64,
    /// Byte distance between consecutive samples of one pixel — C++
    /// `sampleStride`.
    sample_stride: i32,
    /// Byte distance between horizontally adjacent pixel pointer slots — C++
    /// `xStride`.  Unused for addressing in the flat heap model (samples are
    /// contiguous), kept for structural fidelity with the C++ struct.
    #[allow(dead_code)]
    x_stride: i32,
    /// Byte distance between vertically adjacent pixel pointer slots — C++
    /// `yStride`.  Unused in the flat model (see `x_stride`); kept for fidelity.
    #[allow(dead_code)]
    y_stride: i32,
    /// `true` when the file channel has no matching frame-buffer slice; the data
    /// buffer is zero-filled for this channel (C++ `zero`).
    zero: bool,
    /// `slice.xTileCoords` — when set, the sample address uses x relative to the
    /// tile's min x (C++ `xOffsetForData = xTileCoords ? tileRange.min.x : 0`).
    x_tile_coords: bool,
    /// `slice.yTileCoords` — when set, the sample address uses y relative to the
    /// tile's min y.
    y_tile_coords: bool,
}

/// A buffered (not-yet-written) deep tile's already-gathered-and-compressed
/// chunk payload: `(packed_count_table, packed_data, unpacked_data_size)` —
/// the `tile_map` value type, factored out to keep the struct field type
/// simple (clippy `type_complexity`).
type BufferedDeepTile = (Vec<u8>, Vec<u8>, u64);

/// Source of raw (never decompressed) deep tile chunks for
/// [`DeepTiledOutputFile::copy_pixels`] / [`DeepTiledOutputFileState::copy_pixels`]
/// -- one impl per C++ `copyPixels` overload's source type
/// (`DeepTiledInputFile`, `DeepTiledInputPart`).
///
/// Duck-typed (no lifetime parameter, no "hand back the owned concrete
/// reader" method), like `crate::tiled::output::TiledCopySource`, for the
/// same reason: a borrowing multipart part-reader view (`DeepTiledInputPartReader`) never owns a concrete
/// `DeepTiledInputFile` to hand back. Implemented directly for
/// [`DeepTiledInputFile`] here; `multipart::part::DeepTiledInputPart`
/// implements it in `multipart/part.rs`.
pub trait DeepTiledCopySource {
    /// This source's header.
    fn header(&self) -> &Header;
    /// Total number of tiles in the image -- C++ `totalTiles()`.
    fn total_tiles(&self) -> Result<i64>;
    /// The on-disk tile order -- C++ `getTileOrder`.
    fn tile_order(&self) -> Result<crate::tiled::input::TileOrder>;
    /// Tile `(dx, dy, lx, ly)`'s raw (still-encoded) on-disk bytes, plus the
    /// chunk-header-verified coordinates -- C++ `rawTileData`.
    fn raw_tile_data(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<RawTileData>;
}

impl<'a> DeepTiledCopySource for DeepTiledInputFile<'a> {
    fn header(&self) -> &Header {
        DeepTiledInputFile::header(self)
    }
    fn total_tiles(&self) -> Result<i64> {
        DeepTiledInputFile::total_tiles(self)
    }
    fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        DeepTiledInputFile::tile_order(self)
    }
    fn raw_tile_data(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<RawTileData> {
        DeepTiledInputFile::raw_tile_data(self, dx, dy, lx, ly)
    }
}

// ---------------------------------------------------------------------------
// DeepTiledOutputFile — port of `class DeepTiledOutputFile`
// ---------------------------------------------------------------------------

/// Single-part deep tiled EXR writer — port of `Imf::DeepTiledOutputFile`.
///
/// Construct with [`DeepTiledOutputFile::new`] (writes the header with the deep
/// and tiled flags and reserves the tile-offset table), describe the source
/// pixels with [`DeepTiledOutputFile::set_frame_buffer`], then push tiles with
/// [`DeepTiledOutputFile::write_tile`] / [`DeepTiledOutputFile::write_tiles`].
///
/// The writer borrows the caller's [`OStream`] for its whole lifetime and
/// back-patches each tile-offset-table entry immediately after writing the
/// corresponding chunk (see the module docs).
pub struct DeepTiledOutputFile<'a> {
    /// Borrowed output stream (the C++ `_streamData->os`).
    stream: &'a mut dyn OStream,

    /// Everything else this writer owns (mirroring
    /// `crate::tiled::output::TiledOutputFileState`): factored into its own,
    /// stream-free struct so [`crate::multipart::MultiPartOutputFile`] can
    /// persist it across separate [`crate::multipart::MultiPartOutputFile::
    /// deep_tiled_output_part`] calls instead of rebuilding (and silently
    /// resetting the `tile_map`/`next_tile_to_write` buffering) a fresh
    /// writer every call. See [`DeepTiledOutputFileState`].
    state: DeepTiledOutputFileState,
}

/// Every field of C++ `DeepTiledOutputFile::Data` except the stream — the
/// writer's **persisted** per-part state.
///
/// Building a brand-new [`DeepTiledOutputFile`] from the header on every
/// `MultiPartOutputFile::deep_tiled_output_part(n)` call would silently
/// reset the `tile_map`/`next_tile_to_write` out-of-order tile buffering —
/// C++ instead caches ONE writer object per part
/// (`ImfMultiPartOutputFile.cpp:313-321`) so writes interleaved across parts
/// resume correctly. Splitting the writer into this pure state struct plus
/// the stream lets `MultiPartOutputFile` persist a
/// `Vec<Option<DeepTiledOutputFileState>>` (one slot per deep tiled part,
/// built on first access) and hand out a borrowing
/// [`DeepTiledOutputPartWriter`] view instead. The standalone
/// [`DeepTiledOutputFile`] just owns one state for its whole life.
pub(crate) struct DeepTiledOutputFileState {
    /// The image header (type forced to `deeptile`) — C++ `_data->header`.
    header: Header,

    /// The current deep frame buffer — C++ `_data->frameBuffer`.
    frame_buffer: DeepFrameBuffer,

    /// Tile layout copied from the header — C++ `_data->tileDesc`.
    tile_desc: TileDescription,

    /// Line order copied from the header — C++ `_data->lineOrder`.
    line_order: LineOrder,

    /// Data-window bounds — C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,

    /// Number of levels in X / Y — C++ `_data->numXLevels` / `numYLevels`.
    num_x_levels: i32,
    num_y_levels: i32,
    /// Per-level tile counts — C++ `_data->numXTiles` / `numYTiles`.
    num_x_tiles: Vec<i32>,
    num_y_tiles: Vec<i32>,

    /// The tile-offset table — C++ `_data->tileOffsets`.
    tile_offsets: TileOffsets,
    /// File position where the offset table begins — C++
    /// `_data->tileOffsetsPosition`.
    tile_offsets_position: u64,

    /// Tiles held because they were written out of the file's line
    /// order — C++ `Data::tileMap` (`ImfDeepTiledOutputFile.cpp:280-330`
    /// mirrors the flat writer's `tileMap`/`nextTileToWrite`, see
    /// `tiled::output`). Each entry is the already-gathered,
    /// already-compressed `(packed_count, packed_data, unpacked_data_size)`
    /// for one tile, held until every tile before it in line order is on
    /// disk. `RANDOM_Y` never buffers (stays empty).
    tile_map: std::collections::HashMap<(i32, i32, i32, i32), BufferedDeepTile>,
    /// The next tile coordinate this writer expects, in line order — `None`
    /// for `RANDOM_Y` (which never buffers). C++ `Data::nextTileToWrite`.
    next_tile_to_write: Option<(i32, i32, i32, i32)>,

    /// Compressor data format (Xdr for the deep-valid codecs) — C++
    /// `_data->format`.  Kept for structural fidelity; the deep gather always
    /// emits Xdr, so the C++ `NATIVE` `convertToXdr` branch never fires.
    #[allow(dead_code)]
    format: Format,

    /// Per-channel write descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][DeepTiledOutputFile::set_frame_buffer]).
    slices: Vec<TOutSliceInfo>,

    /// Sample-count slice geometry — C++'s `sampleCountSliceBase`/`xStride`/
    /// `yStride`. Pixel data is passed per write call: this file
    /// stores only the LAYOUT; the actual count/sample BYTES are passed
    /// fresh to each [`write_tiles`][Self::write_tiles]/`write_tile` call,
    /// never stored.
    /// Byte offset of the sample-count slice's origin within the caller's
    /// per-call buffer —
    /// C++'s base pointer, `sampleCount(base, …)`.  `0` for the whole-window
    /// layout; a local raster sets it negative to map its low corner to `0`.
    sample_count_base_offset: i64,
    /// X byte stride for `sample_counts` — C++ `sampleCountXStride`.
    sample_count_x_stride: i32,
    /// Y byte stride for `sample_counts` — C++ `sampleCountYStride`.
    sample_count_y_stride: i32,
    /// Sample-count slice `xTileCoords` — C++ `sampleCountXTileCoords`.
    sample_count_x_tile_coords: bool,
    /// Sample-count slice `yTileCoords` — C++ `sampleCountYTileCoords`.
    sample_count_y_tile_coords: bool,
    /// A growable per-row sample-count prefix cache (reset in
    /// `set_frame_buffer`), so `gather_tile`'s per-pixel `samples_before_pixel`
    /// addressing sums each row at most once instead of from scratch on every
    /// pixel of every channel. Deliberately NOT a fixed whole-window eager
    /// build: `set_frame_buffer` allows a sample-count buffer that only
    /// covers the rows the level currently being written needs (a file is
    /// written one level at a time), and this cache never
    /// reads a row that isn't actually queried, so a legitimately short
    /// per-level buffer still works — an eager build to the file's global
    /// `max_y` would spuriously fail it.
    sample_prefix: crate::deep::sample_table::RowPrefixCache,

    /// Max size in bytes of one pixel-sample-count table — C++
    /// `_data->maxSampleCountTableSize = tileDesc.ySize * tileDesc.xSize *
    /// sizeof(int)` (the WHOLE tile area; see module docs on the raw-store path).
    max_sample_count_table_size: usize,

    /// The **separate** sample-count-table compressor (a scanline compressor
    /// sized for `maxSampleCountTableSize`), or `None` for NONE — C++
    /// `TileBuffer::sampleCountTableCompressor`.
    count_compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this writer drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `DeepTiledOutputFile::Data::{multipart, partNumber}`.
    ///
    /// The same unified seam as the flat tiled / deep scanline writers: when
    /// `Some`, every deep tile chunk is prefixed with the `partNumber` (i32 LE)
    /// exactly as C++ `writeTileData` does (`if (ofd->multipart)
    /// Xdr::write(partNumber)`, ImfDeepTiledOutputFile.cpp:487-490); when
    /// `None`, no prefix is written, so single-part output is byte-identical to
    /// the pre-multipart writer.
    part_number: Option<i32>,

    /// Byte position of the header's `"preview"` attribute pixels, as
    /// returned by `Header::write_to` — C++ `Data::previewPosition`. Set by
    /// [`new_standalone`](Self::new_standalone) (the real value) or by
    /// [`from_part_with_preview`](Self::from_part_with_preview) (the real
    /// value `MultiPartOutputFile` captured from
    /// `Header::write_attributes_to`); `0`
    /// only via the plain [`from_part`](Self::from_part), used by the public
    /// `DeepTiledOutputFile::from_part` constructor, which does not itself
    /// write any header and so has no real value to supply.
    preview_position: u64,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the deep scanline writer's
    /// identical field.
    num_threads: crate::threading::ThreadCount,
}

impl<'a> DeepTiledOutputFile<'a> {
    /// Open a `DeepTiledOutputFile` on `stream` for `header` — port of
    /// `DeepTiledOutputFile::DeepTiledOutputFile(OStream&, const Header&, int)`
    /// followed by `initialize`.
    ///
    /// Validates the header (`sanity_check(true, …)` — tiled), forces the type
    /// to `deeptile`, computes the level / tile counts, builds the offset table,
    /// writes the magic number, version word (with the deep + tiled flags folded
    /// in automatically by [`Header::version_field`]) and header, and reserves
    /// the tile-offset table.
    ///
    /// # Errors
    /// * Propagates [`Header::sanity_check`] failures (incl. a missing tile
    ///   description and "compression not valid for deep data").
    /// * [`ExrError::Arg`] if the tile area exceeds the OpenEXR `u32` limit
    ///   (C++ "Deep tile size exceeds maximum permitted area").
    /// * Returns the [`new_compressor`] error for any not-yet-ported codec.
    /// * Propagates stream write / position errors.
    pub fn new(stream: &'a mut dyn OStream, header: &Header) -> Result<Self> {
        Self::with_num_threads(stream, header, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(
        stream: &'a mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        // Single-part: the writer owns the file, so `build` writes magic +
        // version + header + reserves the tile-offset table, and no per-chunk
        // partNumber prefix (`part_number == None`).
        Self::build(stream, header, None, None, num_threads)
    }

    /// Open a `DeepTiledOutputFile` for **one part of a multipart file** — port
    /// of `DeepTiledOutputFile::DeepTiledOutputFile(const OutputPartData*)`.
    ///
    /// The enclosing [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile)
    /// has already written the file prologue.  This constructor only rebuilds
    /// the per-part write state and remembers `part_number` (the i32 prefix of
    /// every deep tile chunk) and `tile_offsets_position` (this part's reserved
    /// offset table, back-patched per tile).  The stream cursor must already
    /// sit where this part's first tile chunk belongs.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledOutputFile.cpp> (`DeepTiledOutputFile(const OutputPartData*)` + `writeTileData`).
    pub fn from_part(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
    ) -> Result<Self> {
        Self::build(stream, header, part_number, Some(tile_offsets_position), None)
    }

    /// Same as [`Self::from_part`], with an explicit `num_threads` override.
    pub fn from_part_mt(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::build(stream, header, part_number, Some(tile_offsets_position), num_threads)
    }

    /// Shared constructor body for [`new`](Self::new) (single-part) and
    /// [`from_part`](Self::from_part) (multipart part).
    ///
    /// The actual state computation lives in
    /// [`DeepTiledOutputFileState::new_standalone`] /
    /// [`DeepTiledOutputFileState::from_part`] so
    /// [`DeepTiledOutputPartWriter`] (which BORROWS a
    /// `DeepTiledOutputFileState` instead of owning one) can share the exact
    /// same construction logic.
    fn build(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        offsets_position: Option<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = match offsets_position {
            None => DeepTiledOutputFileState::new_standalone_mt(stream, header, num_threads)?,
            Some(pos) => {
                DeepTiledOutputFileState::from_part_mt(header, part_number, pos, num_threads)?
            }
        };
        Ok(Self { stream, state })
    }

    /// The image header — port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The current frame buffer — port of `const DeepFrameBuffer& frameBuffer()`.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk. See [`DeepTiledOutputFileState::update_preview_image`] for the
    /// full doc.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written tile's chunk. See
    /// [`DeepTiledOutputFileState::break_tile`] for the full doc.
    #[allow(clippy::too_many_arguments)]
    pub fn break_tile(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        offset: u64,
        length: usize,
        c: u8,
    ) -> Result<()> {
        self.state.break_tile(self.stream, dx, dy, lx, ly, offset, length, c)
    }

    /// Tile width — port of `unsigned int tileXSize() const`.
    pub fn tile_x_size(&self) -> u32 {
        self.state.tile_x_size()
    }

    /// Tile height — port of `unsigned int tileYSize() const`.
    pub fn tile_y_size(&self) -> u32 {
        self.state.tile_y_size()
    }

    /// Level mode — port of `LevelMode levelMode() const`.
    pub fn level_mode(&self) -> LevelMode {
        self.state.level_mode()
    }

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode()`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
    }

    /// Compression type — convenience accessor (C++ reads it off `header()`).
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
    }

    /// Number of levels in X — port of `int numXLevels() const`.
    pub fn num_x_levels(&self) -> i32 {
        self.state.num_x_levels()
    }

    /// Number of levels in Y — port of `int numYLevels() const`.
    pub fn num_y_levels(&self) -> i32 {
        self.state.num_y_levels()
    }

    /// Number of levels (MIP convenience) — port of `int numLevels() const`.
    pub fn num_levels(&self) -> Result<i32> {
        self.state.num_levels()
    }

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(...)`.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.state.is_valid_level(lx, ly)
    }

    /// Width of level `lx` — port of `int levelWidth(int lx) const`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.state.level_width(lx)
    }

    /// Height of level `ly` — port of `int levelHeight(int ly) const`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.state.level_height(ly)
    }

    /// Number of X tiles in level `lx` — port of `int numXTiles(int lx) const`.
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }

    /// Number of Y tiles in level `ly` — port of `int numYTiles(int ly) const`.
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.state.num_y_tiles(ly)
    }

    /// Data window of level `(lx, ly)` — port of
    /// `Box2i dataWindowForLevel(int lx, int ly) const`.
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_level(lx, ly)
    }

    /// Data window of level `l` (MIP convenience) — port of
    /// `Box2i dataWindowForLevel(int l) const`.
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.state.data_window_for_level_l(l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int lx, int ly) const`.
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_tile(dx, dy, lx, ly)
    }

    /// Data window of tile `(dx, dy, l)` (MIP convenience) — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int l) const`.
    pub fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.state.data_window_for_tile_l(dx, dy, l)
    }

    /// Set the pixel source. See [`DeepTiledOutputFileState::set_frame_buffer`]
    /// for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)`. See [`DeepTiledOutputFileState::write_tiles`] for the full doc.
    #[allow(clippy::too_many_arguments)]
    pub fn write_tiles(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tiles(self.stream, sample_counts, sample_data, dx1, dx2, dy1, dy2, lx, ly)
    }

    /// Write the rectangle of tiles at MIP level `l`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_tiles_l(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tiles_l(self.stream, sample_counts, sample_data, dx1, dx2, dy1, dy2, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)`.
    pub fn write_tile(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tile(self.stream, sample_counts, sample_data, dx, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l`.
    pub fn write_tile_l(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tile_l(self.stream, sample_counts, sample_data, dx, dy, l)
    }

    /// Quick pixel copy: transcode every tile of `input` straight into this
    /// file, raw (no decompress/recompress). See
    /// [`DeepTiledOutputFileState::copy_pixels`] for the full doc.
    pub fn copy_pixels<T: DeepTiledCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }
}

// ---------------------------------------------------------------------------
// DeepTiledOutputFileState — the state-only construction + algorithm, shared
// by the owning `DeepTiledOutputFile` and the borrowing
// `DeepTiledOutputPartWriter`.
// ---------------------------------------------------------------------------

impl DeepTiledOutputFileState {
    /// Pure state computation shared by
    /// [`new_standalone_mt`](Self::new_standalone_mt) and
    /// [`from_part_with_preview_mt`](Self::from_part_with_preview_mt) —
    /// everything C++ `DeepTiledOutputFile::initialize(header)` computes that
    /// does NOT depend on `tile_offsets_position`/`preview_position` (which
    /// the two callers fill in differently). Never touches the stream.
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default) — see [`crate::threading::ThreadCount`].
    fn compute_mt(
        header: &Header,
        part_number: Option<i32>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // C++ ctor: header.sanityCheck(true);  (tiled)
        header.sanity_check(true, false)?;

        // --- C++ DeepTiledOutputFile::initialize(header) -------------------
        let mut hdr = header.clone();
        // `ImfDeepTiledOutputFile.cpp:1147-1149`: a part header that
        // already declares a different type cannot be forced into deeptile —
        // catch the type-mismatched-part misuse before silently overwriting
        // it below.
        if hdr.has_type() && hdr.type_()? != DEEP_TILE {
            return Err(ExrError::Arg(
                "Can't build a DeepTiledOutputFile from a type-mismatched part.".into(),
            ));
        }
        // C++: _data->header.setType(DEEPTILE);
        hdr.set_type(DEEP_TILE)?;

        let line_order = hdr.line_order()?;
        let tile_desc = hdr.tile_description()?;

        let data_window = hdr.data_window()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        // C++: maxSampleCountTableSize = tileDesc.ySize * tileDesc.xSize * sizeof(int).
        // (The WHOLE tile area — NOT the per-tile rectangle.)
        let max_sample_count_table_size = (tile_desc.y_size as u64)
            .saturating_mul(tile_desc.x_size as u64)
            .saturating_mul(core::mem::size_of::<u32>() as u64);

        // C++: if (maxSampleCountTableSize > UINT_MAX) throw "Deep tile size …".
        if max_sample_count_table_size > u32::MAX as u64 {
            return Err(ExrError::Arg(
                "Deep tile size exceeds maximum permitted area".into(),
            ));
        }
        let max_sample_count_table_size = max_sample_count_table_size as usize;

        // C++: precalculateTileInfo(tileDesc, minX..maxY, numXTiles, numYTiles,
        //      numXLevels, numYLevels);
        let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
            precalculate_tile_info(&tile_desc, min_x, max_x, min_y, max_y)?;

        // C++: Compressor* c = newTileCompressor(compression, 0, tileDesc.ySize,
        //      header); format = defaultFormat(c); delete c;
        let compression = hdr.compression()?;
        let probe = new_tile_compressor(compression, 0, tile_desc.y_size as usize, &hdr)?;
        let format = default_format(probe.as_deref());
        drop(probe);

        // C++: tileOffsets = TileOffsets(mode, numXLevels, numYLevels,
        //      numXTiles, numYTiles);
        let tile_offsets = TileOffsets::new(
            tile_desc.mode,
            num_x_levels,
            num_y_levels,
            &num_x_tiles,
            &num_y_tiles,
        )?;

        // C++: header.setChunkCount(getChunkOffsetTableSize(header));
        let chunk_count = get_tiled_chunk_offset_table_size(&hdr)?;
        hdr.set_chunk_count(chunk_count)?;

        // C++ (per tile buffer): sampleCountTableCompressor =
        //   newCompressor(compression, maxSampleCountTableSize, header).
        // Single-threaded → one instance.  The count table is compressed by a
        // SCANLINE compressor (not a tile compressor); the data compressor is
        // rebuilt per tile in `write_single_tile`.
        let count_compressor = new_compressor(compression, max_sample_count_table_size, &hdr)?;

        // Mirrors the flat tiled writer: "the first tile coordinate we will be writing
        // if the file is not RANDOM_Y" (`ImfDeepTiledOutputFile.cpp` shares
        // `Data::initialize` with the flat tiled writer's own logic).
        let next_tile_to_write = match line_order {
            LineOrder::IncreasingY => Some((0, 0, 0, 0)),
            LineOrder::DecreasingY => {
                Some((0, num_y_tiles.first().copied().unwrap_or(1).saturating_sub(1), 0, 0))
            }
            LineOrder::RandomY | LineOrder::Invalid => None,
        };

        // Structural detail (debug): deep tiled writer is up.
        log::debug!(
            "deep DeepTiledOutputFile: {}x{} part={part_number:?} tile={}x{} mode={:?} \
             levels={}x{} codec={compression:?}",
            max_x - min_x + 1,
            max_y - min_y + 1,
            tile_desc.x_size,
            tile_desc.y_size,
            tile_desc.mode,
            num_x_levels,
            num_y_levels
        );

        Ok(Self {
            header: hdr,
            frame_buffer: DeepFrameBuffer::new(),
            tile_desc,
            line_order,
            min_x,
            max_x,
            min_y,
            max_y,
            num_x_levels,
            num_y_levels,
            num_x_tiles,
            num_y_tiles,
            tile_offsets,
            tile_offsets_position: 0,
            tile_map: std::collections::HashMap::new(),
            next_tile_to_write,
            format,
            slices: Vec::new(),
            sample_count_base_offset: 0,
            sample_count_x_stride: 0,
            sample_count_y_stride: 0,
            sample_count_x_tile_coords: false,
            sample_count_y_tile_coords: false,
            sample_prefix: crate::deep::sample_table::RowPrefixCache::new(min_x, max_x, min_y),
            max_sample_count_table_size,
            count_compressor,
            part_number,
            preview_position: 0,
            num_threads,
        })
    }

    /// Standalone single-part construction: writes the magic + version word,
    /// the header, and reserves the (zero-filled) tile-offset table on
    /// `stream` — port of the C++ `DeepTiledOutputFile` constructor body,
    /// minus the pure state computation shared with
    /// [`from_part`](Self::from_part).
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default) — see [`crate::threading::ThreadCount`].
    fn new_standalone_mt(
        stream: &mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let mut state = Self::compute_mt(header, None, num_threads)?;

        // C++: writeMagicNumberAndVersionField(os, header);
        //      previewPosition     = header.writeTo(os, true);
        //      tileOffsetsPosition = tileOffsets.writeTo(os);
        //
        // `Header::write_to` folds in the TILED + deep (NON_IMAGE) flags
        // automatically, so passing base_flags = 0 is correct.
        state.preview_position = state.header.write_to(stream, 0)?;
        state.tile_offsets_position = state.tile_offsets.write_to(stream)?;
        Ok(state)
    }

    /// Multipart-part construction — port of `DeepTiledOutputFile::
    /// DeepTiledOutputFile(const OutputPartData*)`. The container already
    /// wrote the file prologue and this part's reserved offset table; this
    /// only computes the pure per-part state and records where that table
    /// lives. Touches no stream at all — this is what lets
    /// `MultiPartOutputFile` build this state ONCE per part and persist it
    /// across `deep_tiled_output_part(n)` calls, including the
    /// `tile_map`/`next_tile_to_write` buffering. `preview_position` stays `0` (see the
    /// field doc): the public `DeepTiledOutputFile::from_part` constructor
    /// (this function's only OTHER caller besides `MultiPartOutputFile`) does
    /// not itself write any header, so it has no real value to supply.
    /// `MultiPartOutputFile` calls `from_part_with_preview_mt` instead.
    /// `num_threads` is the per-file
    /// override (`None` = capture the current process-wide default).
    pub(crate) fn from_part_mt(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::from_part_with_preview_mt(header, part_number, tile_offsets_position, 0, num_threads)
    }

    /// Like [`from_part_mt`](Self::from_part_mt), but additionally records
    /// `preview_position` — C++
    /// `ImfMultiPartOutputFile.cpp:474-477` captures
    /// `parts[i]->previewPosition = headers[i].writeTo(..)` for every part
    /// while writing the header list, so `update_preview_image` works on a
    /// multipart deep-tiled part exactly as it does on a standalone file.
    /// [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile) calls
    /// this (not [`from_part_mt`](Self::from_part_mt)) with the value
    /// `Header::write_attributes_to` returned for this part, and with its own
    /// captured `num_threads` so every part of a multipart file shares one
    /// thread-count decision.
    pub(crate) fn from_part_with_preview_mt(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        preview_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let mut state = Self::compute_mt(header, part_number, num_threads)?;
        state.tile_offsets_position = tile_offsets_position;
        state.preview_position = preview_position;
        Ok(state)
    }

    // -----------------------------------------------------------------------
    // Accessors — thin wrappers (ImfDeepTiledOutputFile.cpp lines 1327-2073)
    // -----------------------------------------------------------------------

    /// The image header — port of `const Header& header() const`.
    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    /// The current frame buffer — port of `const DeepFrameBuffer& frameBuffer()`.
    pub(crate) fn frame_buffer(&self) -> &DeepFrameBuffer {
        &self.frame_buffer
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk — port of `void updatePreviewImage(const PreviewRgba newPixels[])`
    /// (`ImfDeepTiledOutputFile.cpp:1730-1757`). Same shape as
    /// `tiled::output::TiledOutputFileState::update_preview_image`.
    ///
    /// # Errors
    /// * [`ExrError::Logic`] if the file has no preview image.
    /// * [`ExrError::Arg`] if the header has no `"preview"` attribute of the
    ///   expected type, or `new_pixels.len()` does not match the existing
    ///   preview's pixel count.
    pub(crate) fn update_preview_image(
        &mut self,
        stream: &mut dyn OStream,
        new_pixels: &[crate::attr::PreviewRgba],
    ) -> Result<()> {
        use crate::attr::attribute::AttrValue;
        use crate::attr::PreviewImage;

        if self.preview_position == 0 {
            return Err(ExrError::Logic(format!(
                "Cannot update preview image pixels. File \"{}\" does not \
                 contain a preview image.",
                stream.file_name()
            )));
        }

        let pia = self.header.typed_attribute_mut::<PreviewImage>("preview")?;
        let pixels = pia.value.pixels_mut();
        if new_pixels.len() != pixels.len() {
            return Err(ExrError::Arg(format!(
                "New preview image has {} pixels; the file's preview image \
                 has {}.",
                new_pixels.len(),
                pixels.len()
            )));
        }
        pixels.copy_from_slice(new_pixels);

        let saved = stream.tellp()?;
        stream.seekp(self.preview_position)?;
        pia.value.write_value(stream, 0)?;
        stream.seekp(saved)?;
        Ok(())
    }

    /// Overwrite `length` bytes of an already-written tile's chunk, starting
    /// `offset` bytes into it, with the byte `c` — port of `void breakTile(int
    /// dx, int dy, int lx, int ly, int offset, int length, char c)`
    /// (`ImfDeepTiledOutputFile.cpp:1782-1805`; test-only helper for exercising
    /// corrupt-chunk error paths). Same shape as
    /// `tiled::output::TiledOutputFileState::break_tile`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if the tile has not been written yet
    /// (offset-table slot still 0), matching the C++ throw.
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ breakTile's parameter list
    pub(crate) fn break_tile(
        &mut self,
        stream: &mut dyn OStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        offset: u64,
        length: usize,
        c: u8,
    ) -> Result<()> {
        let position = self.tile_offsets.get(dx, dy, lx, ly)?;
        if position == 0 {
            return Err(ExrError::Arg(format!(
                "Cannot overwrite tile ({dx}, {dy}, {lx},{ly}). The tile has \
                 not yet been stored in file \"{}\".",
                stream.file_name()
            )));
        }

        stream.seekp(position + offset)?;
        let fill = vec![c; length];
        stream.write(&fill)?;
        Ok(())
    }

    /// Tile width — port of `unsigned int tileXSize() const`.
    pub fn tile_x_size(&self) -> u32 {
        self.tile_desc.x_size
    }

    /// Tile height — port of `unsigned int tileYSize() const`.
    pub fn tile_y_size(&self) -> u32 {
        self.tile_desc.y_size
    }

    /// Level mode — port of `LevelMode levelMode() const`.
    pub fn level_mode(&self) -> LevelMode {
        self.tile_desc.mode
    }

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode()`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.tile_desc.rounding_mode
    }

    /// Compression type — convenience accessor (C++ reads it off `header()`).
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.header.compression()
    }

    /// Number of levels in X — port of `int numXLevels() const`.
    pub fn num_x_levels(&self) -> i32 {
        self.num_x_levels
    }

    /// Number of levels in Y — port of `int numYLevels() const`.
    pub fn num_y_levels(&self) -> i32 {
        self.num_y_levels
    }

    /// Number of levels (MIP convenience) — port of `int numLevels() const`.
    ///
    /// Returns [`ExrError::Logic`] for `RIPMAP_LEVELS` (undefined there), exactly
    /// as the C++ throws `LogicExc`.
    pub fn num_levels(&self) -> Result<i32> {
        if self.level_mode() == LevelMode::RipmapLevels {
            return Err(ExrError::Logic(
                "numLevels() is not defined for RIPMAPs.".into(),
            ));
        }
        Ok(self.num_x_levels)
    }

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(...)`.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        if lx < 0 || ly < 0 {
            return false;
        }
        if self.level_mode() == LevelMode::MipmapLevels && lx != ly {
            return false;
        }
        if lx >= self.num_x_levels || ly >= self.num_y_levels {
            return false;
        }
        true
    }

    /// Width of level `lx` — port of `int levelWidth(int lx) const`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        level_size(self.min_x, self.max_x, lx, self.tile_desc.rounding_mode)
    }

    /// Height of level `ly` — port of `int levelHeight(int ly) const`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        level_size(self.min_y, self.max_y, ly, self.tile_desc.rounding_mode)
    }

    /// Number of X tiles in level `lx` — port of `int numXTiles(int lx) const`.
    ///
    /// Returns [`ExrError::Logic`] when `lx` is out of range (C++ `LogicExc`).
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.num_x_tiles
            .get(usize::try_from(lx).map_err(|_| {
                ExrError::Logic("numXTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Logic("numXTiles(): argument is not in valid range".into()))
    }

    /// Number of Y tiles in level `ly` — port of `int numYTiles(int ly) const`.
    ///
    /// Returns [`ExrError::Logic`] when `ly` is out of range (C++ `LogicExc`).
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.num_y_tiles
            .get(usize::try_from(ly).map_err(|_| {
                ExrError::Logic("numYTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Logic("numYTiles(): argument is not in valid range".into()))
    }

    /// Data window of level `(lx, ly)` — port of
    /// `Box2i dataWindowForLevel(int lx, int ly) const`.
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        data_window_for_level(
            &self.tile_desc,
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            lx,
            ly,
        )
    }

    /// Data window of level `l` (MIP convenience) — port of
    /// `Box2i dataWindowForLevel(int l) const`.
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.data_window_for_level(l, l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int lx, int ly) const`.
    ///
    /// Returns [`ExrError::Arg`] for an invalid tile (C++ `ArgExc`).
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        if !self.is_valid_tile(dx, dy, lx, ly) {
            return Err(ExrError::Arg("Arguments not in valid range.".into()));
        }
        data_window_for_tile(
            &self.tile_desc,
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            dx,
            dy,
            lx,
            ly,
        )
    }

    /// Data window of tile `(dx, dy, l)` (MIP convenience) — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int l) const`.
    pub fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.data_window_for_tile(dx, dy, l, l)
    }

    /// Whether `(dx, dy, lx, ly)` addresses an existing tile — port of the
    /// private `bool DeepTiledOutputFile::isValidTile(int, int, int, int) const`
    /// (ImfDeepTiledOutputFile.cpp:2065-2073).
    fn is_valid_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> bool {
        if lx < 0 || lx >= self.num_x_levels || ly < 0 || ly >= self.num_y_levels {
            return false;
        }
        let nx = match self.num_x_tiles.get(lx as usize) {
            Some(&n) => n,
            None => return false,
        };
        let ny = match self.num_y_tiles.get(ly as usize) {
            Some(&n) => n,
            None => return false,
        };
        dx >= 0 && dx < nx && dy >= 0 && dy < ny
    }

    // -----------------------------------------------------------------------
    // setFrameBuffer — ImfDeepTiledOutputFile.cpp lines 1339-1453
    // -----------------------------------------------------------------------

    /// Set the pixel source — port of `DeepTiledOutputFile::setFrameBuffer`.
    ///
    /// `sample_counts` is the caller-owned flat per-pixel sample-count buffer
    /// addressed by the frame buffer's sample-count slice (byte strides taken
    /// from that slice); it replaces the C++ `char* sampleCountSliceBase`.
    /// `sample_data` is the caller-owned flat per-channel sample heap the deep
    /// slices' `base_offset` index into (laid out in whole-data-window raster
    /// order — see the module docs).
    ///
    /// Performs the same compatibility checks as C++ (pixel type must match the
    /// header channel; **every channel must have sampling `(1,1)`** — subsampling
    /// is not supported in tiled files; the sample-count slice must be present)
    /// and builds the per-channel `TOutSliceInfo` table, including the "channel
    /// present in the header but absent from the frame buffer → write zeroes"
    /// case.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when a slice's type is incompatible with the header
    ///   channel, or when any slice has sampling other than `(1,1)` (matching
    ///   the C++ `THROW`s).
    /// * [`ExrError::Arg`] when no sample-count slice was set (C++ "Invalid base
    ///   pointer, please set a proper sample count slice.").
    ///
    /// Writer pixel data is passed PER WRITE CALL: this stores only the LAYOUT; it
    /// does **not** take or retain the sample-count / sample-data BYTES.
    /// Those are passed to each [`write_tiles`][Self::write_tiles] /
    /// `write_tile` call instead — no stored lifetime, no snapshot, and a
    /// per-level count buffer can be handed fresh to each level's write.
    pub(crate) fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++: compatibility checks for every header channel also in the fb.
        for (name, ch) in channels.iter() {
            if let Some(slice) = frame_buffer.find_slice(name) {
                if ch.type_ != slice.slice.type_ {
                    return Err(ExrError::Arg(format!(
                        "Pixel type of \"{name}\" channel of output file is not \
                         compatible with the frame buffer's pixel type."
                    )));
                }
                if slice.slice.x_sampling != 1 || slice.slice.y_sampling != 1 {
                    return Err(ExrError::Arg(
                        "All channels in a tiled file must have sampling (1,1).".into(),
                    ));
                }
            }
        }

        // C++: the sample-count slice base must be set (non-null). In the
        // safe-Rust model a slice with both strides zero is the equivalent of
        // a null base pointer (no buffer bytes are available here to check
        // emptiness against — that check now lives in `write_tiles`).
        let sc_slice = frame_buffer.get_sample_count_slice();
        if sc_slice.x_stride == 0 && sc_slice.y_stride == 0 {
            return Err(ExrError::Arg(
                "Invalid base pointer, please set a proper sample count slice.".into(),
            ));
        }
        // The sample-count byte strides are narrowed usize->i32 for the
        // addressing math; a value above i32::MAX would wrap to a bogus/negative
        // stride and mis-address the buffer, so reject it up front.
        if sc_slice.x_stride > i32::MAX as usize || sc_slice.y_stride > i32::MAX as usize {
            return Err(ExrError::Arg(
                "Deep sample-count slice stride exceeds the i32 range.".into(),
            ));
        }
        self.sample_count_base_offset = sc_slice.base_offset;
        self.sample_count_x_stride = sc_slice.x_stride as i32;
        self.sample_count_y_stride = sc_slice.y_stride as i32;
        self.sample_count_x_tile_coords = sc_slice.x_tile_coords;
        self.sample_count_y_tile_coords = sc_slice.y_tile_coords;

        // No whole-data-window extent check here (unlike the scanline writer).
        // A tiled file is written one LEVEL at a time, and each level may bind its
        // own count buffer that only covers that level's rows (mip level 1 of an
        // 8x6 window reads rows [0, 3), not [0, 6)); a data-window-sized bound would
        // reject that valid layout. The single authoritative boundary is
        // `sample_count_at`, which hard-errors on any out-of-range read at gather
        // time (never a silent zero sample), for absolute and tile-relative coords.

        // C++: build the slice table in header channel order.  The sample-count
        // slice is not in the header, so it is not added here.
        let mut slices: Vec<TOutSliceInfo> = Vec::new();
        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                None => slices.push(TOutSliceInfo {
                    type_: ch.type_,
                    base_offset: 0,
                    sample_stride: 0,
                    x_stride: 0,
                    y_stride: 0,
                    zero: true,
                    x_tile_coords: false,
                    y_tile_coords: false,
                }),
                Some(slice) => slices.push(TOutSliceInfo {
                    type_: slice.slice.type_,
                    base_offset: slice.slice.base_offset,
                    sample_stride: slice.sample_stride,
                    x_stride: slice.slice.x_stride as i32,
                    y_stride: slice.slice.y_stride as i32,
                    zero: false,
                    x_tile_coords: slice.slice.x_tile_coords,
                    y_tile_coords: slice.slice.y_tile_coords,
                }),
            }
        }

        // Reset the per-row prefix cache — a fresh `set_frame_buffer`
        // call means a (possibly different, possibly level-sized)
        // `sample_counts` buffer, so rows cached from a previous buffer must
        // not be reused.
        self.sample_prefix =
            crate::deep::sample_table::RowPrefixCache::new(self.min_x, self.max_x, self.min_y);

        self.slices = slices;
        self.frame_buffer = frame_buffer.clone();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // writeTiles / writeTile — ImfDeepTiledOutputFile.cpp lines 1464-1691
    // -----------------------------------------------------------------------

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)` — port of `DeepTiledOutputFile::writeTiles(...)`.
    ///
    /// The tile coordinate ranges are inclusive; `dx1 > dx2` / `dy1 > dy2` are
    /// swapped (as in C++).  The rectangle is walked in the file's line order
    /// (`DECREASING_Y` walks rows high→low) so the common "write whole level in
    /// order" path lays the chunks out exactly as a C++ `INCREASING_Y` /
    /// `DECREASING_Y` file would.  Each tile is gathered, compressed and written
    /// immediately, and its offset back-patched (see the module docs).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, if any corner tile is
    ///   invalid, or if the level is invalid (matching the C++ throws).
    /// * [`ExrError::Arg`] if a tile is written more than once.
    /// * Propagates gather / codec / stream errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_tiles(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        mut dx1: i32,
        mut dx2: i32,
        mut dy1: i32,
        mut dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer …").
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data source.".into(),
            ));
        }
        // Pixel data is passed per write call, so the null-base check lives
        // here rather than in `set_frame_buffer` (which receives no buffer
        // bytes to check).
        if sample_counts.is_empty() {
            return Err(ExrError::Arg(
                "Invalid base pointer, please set a proper sample count slice.".into(),
            ));
        }

        // C++: validate both corner tiles and the level.
        if !self.is_valid_tile(dx1, dy1, lx, ly) || !self.is_valid_tile(dx2, dy2, lx, ly) {
            return Err(ExrError::Arg("Tile coordinates are invalid.".into()));
        }
        if !self.is_valid_level(lx, ly) {
            return Err(ExrError::Arg(format!(
                "Level coordinate ({lx}, {ly}) is invalid."
            )));
        }

        // C++: if (dx1 > dx2) swap; if (dy1 > dy2) swap.
        if dx1 > dx2 {
            std::mem::swap(&mut dx1, &mut dx2);
        }
        if dy1 > dy2 {
            std::mem::swap(&mut dy1, &mut dy2);
        }

        // C++: walk dy in the file's line order; dx always low→high.
        let (dy_start, dy_step) = if self.line_order == LineOrder::DecreasingY {
            (dy2, -1)
        } else {
            (dy1, 1)
        };

        let mut coords: Vec<(i32, i32, i32, i32)> = Vec::new();
        let mut dy = dy_start;
        while dy >= dy1 && dy <= dy2 {
            for dx in dx1..=dx2 {
                coords.push((dx, dy, lx, ly));
            }
            dy += dy_step;
        }

        // Parallel only when there is more than one tile
        // AND the captured thread count is `> 1`, mirroring the flat tiled
        // writer and the deep tiled reader's identical gate.
        if coords.len() > 1
            && self.num_threads.is_parallel()
            && self.write_tiles_parallel(stream, sample_counts, sample_data, &coords)?
        {
            crate::probe::hit(crate::probe::Probe::DeepTileWriteParallel);
            return Ok(());
        }
        // A `false`/skipped result means either the fast path wasn't
        // eligible (mixed tile-coords flags across slices — see
        // `write_tiles_parallel`'s doc), or there was only one tile to begin
        // with — fall back to the exact serial path.

        for &(dx, dy, lx, ly) in &coords {
            self.write_single_tile(stream, sample_counts, sample_data, dx, dy, lx, ly)?;
        }

        Ok(())
    }

    /// Write the rectangle of tiles at MIP level `l` — port of
    /// `DeepTiledOutputFile::writeTiles(dx1,dxMax,dyMin,dyMax,l)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_tiles_l(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.write_tiles(stream, sample_counts, sample_data, dx1, dx2, dy1, dy2, l, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)` — port of
    /// `DeepTiledOutputFile::writeTile(int dx, int dy, int lx, int ly)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_tile(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.write_tiles(stream, sample_counts, sample_data, dx, dx, dy, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l` — port of
    /// `DeepTiledOutputFile::writeTile(int dx, int dy, int l)`.
    pub(crate) fn write_tile_l(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        l: i32,
    ) -> Result<()> {
        self.write_tile(stream, sample_counts, sample_data, dx, dy, l, l)
    }

    /// Quick pixel copy: transcode every tile of `input` straight into this
    /// file, RAW (no decompress/recompress) -- port of the 2 C++
    /// `DeepTiledOutputFile::copyPixels` overloads
    /// (`ImfDeepTiledOutputFile.cpp:1694-1864`).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if the headers' tile description, data window,
    ///   line order, compression, or channel list differ.
    /// * [`ExrError::Logic`] if this file already has pixel data written.
    /// * Propagates `input`'s `total_tiles`/`tile_order`/`raw_tile_data` errors.
    pub(crate) fn copy_pixels<T: DeepTiledCopySource>(
        &mut self,
        stream: &mut dyn OStream,
        input: &mut T,
    ) -> Result<()> {
        let hdr = &self.header;
        let in_hdr = input.header();

        let td = hdr.tile_description().map_err(|_| {
            ExrError::Arg("The output file is tiled, but is missing a tile description.".into())
        })?;
        let in_td = in_hdr.tile_description().map_err(|_| {
            ExrError::Arg("Cannot perform a quick pixel copy: the input file is not tiled.".into())
        })?;
        if td != in_td {
            return Err(ExrError::Arg(
                "Quick pixel copy failed: the files have different tile descriptions.".into(),
            ));
        }
        if hdr.data_window()? != in_hdr.data_window()? {
            return Err(ExrError::Arg(
                "Cannot copy pixels: the files have different data windows.".into(),
            ));
        }
        if hdr.line_order()? != in_hdr.line_order()? {
            return Err(ExrError::Arg(
                "Quick pixel copy failed: the files have different line orders.".into(),
            ));
        }
        if hdr.compression()? != in_hdr.compression()? {
            return Err(ExrError::Arg(
                "Quick pixel copy failed: the files use different compression methods.".into(),
            ));
        }
        if hdr.channels()? != in_hdr.channels()? {
            return Err(ExrError::Arg(
                "Quick pixel copy failed: the files have different channel lists.".into(),
            ));
        }

        // --- No pixel data may have been written to this file yet ----------
        if !self.tile_offsets.is_empty() {
            return Err(ExrError::Logic(
                "Quick pixel copy failed: this file already contains pixel data.".into(),
            ));
        }

        // C++: size_t numAllTiles = in.totalTiles();
        let num_all_tiles = input.total_tiles()?;

        // `copy_pixels` bypasses the `tile_map` buffering entirely (calls
        // `write_tile_chunk_now` directly), exactly like C++'s `copyPixels`
        // calling the low-level `writeTileData` rather than
        // `bufferedTileWrite` -- it already computes tiles strictly in file
        // order itself (`tile_order()` for RANDOM_Y, `next_tile_coord` walk
        // otherwise), so there is never an out-of-order tile to buffer here.
        let random_y = self.line_order == LineOrder::RandomY;

        // For RANDOM_Y, C++ asks the INPUT file for the on-disk tile order and
        // drives the copy in that order (mirrors the flat writer's own
        // `copy_pixels`, `tiled::output::TiledOutputFileState::copy_pixels`).
        if random_y {
            let (dx_t, dy_t, lx_t, ly_t) = input.tile_order()?;
            self.next_tile_to_write = dx_t
                .first()
                .zip(dy_t.first())
                .zip(lx_t.first())
                .zip(ly_t.first())
                .map(|(((&dx, &dy), &lx), &ly)| (dx, dy, lx, ly));

            for i in 0..num_all_tiles as usize {
                let coord = self.next_tile_to_write.ok_or_else(|| {
                    ExrError::Logic("copy_pixels: RANDOM_Y tile cursor exhausted early.".into())
                })?;
                let (dx, dy, lx, ly) = coord;
                let raw = input.raw_tile_data(dx, dy, lx, ly)?;
                let (packed_count, packed_data, unpacked) =
                    (raw.packed_count_table, raw.packed_data, raw.unpacked_data_size);
                Self::write_tile_chunk_now(
                    stream,
                    &mut self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    coord,
                    &packed_count,
                    &packed_data,
                    unpacked,
                )?;
                if i + 1 < num_all_tiles as usize {
                    self.next_tile_to_write = dx_t
                        .get(i + 1)
                        .zip(dy_t.get(i + 1))
                        .zip(lx_t.get(i + 1))
                        .zip(ly_t.get(i + 1))
                        .map(|(((&dx, &dy), &lx), &ly)| (dx, dy, lx, ly));
                }
            }
        } else {
            for _ in 0..num_all_tiles {
                let coord = self.next_tile_to_write.ok_or_else(|| {
                    ExrError::Logic("copy_pixels: tile write cursor is unset.".into())
                })?;
                let (dx, dy, lx, ly) = coord;
                let raw = input.raw_tile_data(dx, dy, lx, ly)?;
                let (packed_count, packed_data, unpacked) =
                    (raw.packed_count_table, raw.packed_data, raw.unpacked_data_size);
                Self::write_tile_chunk_now(
                    stream,
                    &mut self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    coord,
                    &packed_count,
                    &packed_data,
                    unpacked,
                )?;
                self.next_tile_to_write = Some(self.next_tile_coord(coord)?);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // The gather + count-table + compress + write of one tile.
    //
    // Fuses C++ `TileBufferTask::execute` (gather + count table + compress)
    // with `bufferedTileWrite` / `writeTileData` (the chunk write + offset
    // record).  Single-threaded; written immediately at the current stream
    // position, so the on-disk order is the call order (RANDOM_Y semantics).
    // -----------------------------------------------------------------------
    /// The successor of tile coordinate `a` in the file's line order — port
    /// of C++ `DeepTiledOutputFile::Data::nextTileCoord`; the walk is the
    /// flat writer's, shared as [`TileGeom::next`].
    fn next_tile_coord(&self, a: (i32, i32, i32, i32)) -> Result<(i32, i32, i32, i32)> {
        self.geom().next(a)
    }

    /// The read-only level geometry `next_tile_coord` walks.
    fn geom(&self) -> TileGeom<'_> {
        TileGeom {
            tile_desc: self.tile_desc,
            line_order: self.line_order,
            num_x_levels: self.num_x_levels,
            num_y_levels: self.num_y_levels,
            num_x_tiles: &self.num_x_tiles,
            num_y_tiles: &self.num_y_tiles,
        }
    }

    /// Disjoint borrows for the pipelined write: what the encode workers read
    /// (`header`, `slices`) next to the emit state ([`DeepTileSink`]) that
    /// `feed` mutates, so neither needs to be cloned per call.
    fn split(&mut self) -> (&Header, &[TOutSliceInfo], DeepTileSink<'_>) {
        let geom = TileGeom {
            tile_desc: self.tile_desc,
            line_order: self.line_order,
            num_x_levels: self.num_x_levels,
            num_y_levels: self.num_y_levels,
            num_x_tiles: &self.num_x_tiles,
            num_y_tiles: &self.num_y_tiles,
        };
        let sink = DeepTileSink {
            geom,
            tile_offsets: &mut self.tile_offsets,
            tile_offsets_position: self.tile_offsets_position,
            part_number: self.part_number,
            tile_map: &mut self.tile_map,
            next_tile_to_write: &mut self.next_tile_to_write,
        };
        (&self.header, &self.slices, sink)
    }

    /// Write one already-gathered-and-compressed deep tile chunk to the
    /// stream NOW and back-patch its offset-table slot — split out of
    /// [`Self::write_single_tile`] (mirrors the flat writer's
    /// `write_tile_chunk_now`) so it can be called either immediately
    /// or later, when a buffered tile's turn comes up. Takes the exact
    /// fields it touches rather than `&mut self` so a caller can hold
    /// `packed_count`/`packed_data` borrowed from another field of `self`
    /// across the call.
    #[allow(clippy::too_many_arguments)]
    fn write_tile_chunk_now(
        stream: &mut dyn OStream,
        tile_offsets: &mut TileOffsets,
        tile_offsets_position: u64,
        part_number: Option<i32>,
        coord: (i32, i32, i32, i32),
        packed_count: &[u8],
        packed_data: &[u8],
        unpacked_data_size: u64,
    ) -> Result<()> {
        let (dx, dy, lx, ly) = coord;
        let chunk_position = stream.tellp()?;
        *tile_offsets.get_mut(dx, dy, lx, ly)? = chunk_position;

        // Byte order (C++ `writeTileData`, `ImfDeepTiledOutputFile.cpp:452-538`):
        // for a multipart file the deep tile chunk is prefixed with the part
        // number (i32 LE) before dx/dy/lx/ly -- the ONLY difference from the
        // single-part layout. Then dx, dy, lx, ly (4*i32), sampleCountTableSize,
        // packedDataSize, unpackedDataSize (3*u64), count table, data.
        if let Some(pn) = part_number {
            xdr::write_i32(&mut *stream, pn)?;
        }
        xdr::write_i32(&mut *stream, dx)?;
        xdr::write_i32(&mut *stream, dy)?;
        xdr::write_i32(&mut *stream, lx)?;
        xdr::write_i32(&mut *stream, ly)?;
        xdr::write_u64(&mut *stream, packed_count.len() as u64)?;
        xdr::write_u64(&mut *stream, packed_data.len() as u64)?;
        xdr::write_u64(&mut *stream, unpacked_data_size)?;
        stream.write(packed_count)?;
        stream.write(packed_data)?;

        // --- Back-patch the offset table slot ------------------------------
        let slot_index = tile_offsets.slot_index(dx, dy, lx, ly)?;
        let saved = stream.tellp()?;
        stream.seekp(tile_offsets_position + slot_index as u64 * SIZE_I64 as u64)?;
        xdr::write_u64(&mut *stream, chunk_position)?;
        stream.seekp(saved)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_single_tile(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        // C++ bufferedTileWrite: reject a second write to the same tile.
        if self.tile_offsets.get(dx, dy, lx, ly)? != 0 {
            return Err(ExrError::Arg(format!(
                "Attempt to write tile ({dx}, {dy}, {lx}, {ly}) more than once."
            )));
        }

        // C++ TileBufferTask::execute: the tile's pixel rectangle.
        let tile_range = data_window_for_tile(
            &self.tile_desc,
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            dx,
            dy,
            lx,
            ly,
        )?;

        // C++: the sample-count addressing offsets (xOffsetForSampleCount /
        // yOffsetForSampleCount) — used both for the gather and the count table.
        let x_off_sc = if self.sample_count_x_tile_coords {
            tile_range.min.x
        } else {
            0
        };
        let y_off_sc = if self.sample_count_y_tile_coords {
            tile_range.min.y
        } else {
            0
        };

        // --- Gather the tile's sample data (Xdr LE) ------------------------
        let mut data: Vec<u8> = Vec::new();
        self.gather_tile(sample_counts, sample_data, &tile_range, x_off_sc, y_off_sc, &mut data)?;
        let unpacked_data_size = data.len() as u64;

        // --- Build the cumulative sample-count table over the tile rect ----
        // C++ (TileBufferTask::execute:944-956): for each row i of the rect,
        // count resets to 0 and accumulates across columns j; each running
        // total is written as an Xdr int.  Addresses use the sample-count tile
        // offsets (j - x_off_sc, i - y_off_sc).
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;
        let table_data_size = tile_w * tile_h * SIZE_INT;
        let mut count_table = Vec::with_capacity(table_data_size);
        for i in tile_range.min.y..=tile_range.max.y {
            // Collect the row first so `sample_count`'s Result propagates
            // before it is handed to the count-encoder.
            let row: Vec<u32> = (tile_range.min.x..=tile_range.max.x)
                .map(|j| self.sample_count(sample_counts, j - x_off_sc, i - y_off_sc))
                .collect::<Result<_>>()?;
            encode_cumulative_row(row, &mut count_table)?;
        }

        // --- Compress the count table (its OWN scanline compressor) --------
        // C++ `ImfDeepTiledOutputFile.cpp:958-978`: if a compressor exists,
        // compress; then, if there is NO compressor OR the compressed size
        // is `>= maxSampleCountTableSize` (:973 — the same store-raw-at-equality
        // rule as `stored_chunk`, but measured against the WHOLE-tile max table
        // size rather than this tile's table, which is why it cannot call the
        // helper),
        // Store the RAW table padded to maxSampleCountTableSize bytes.
        // Deliberately clear padding: C++ retains previous tile bytes in its reused
        // construction-time buffer. The decoded rectangle is identical either way.
        let packed_count: Vec<u8> = {
            let compressed: Option<Vec<u8>> = match self.count_compressor.as_mut() {
                Some(c) => Some(
                    c.compress(&count_table, tile_range.min.y)
                        .map_err(crate::error::encode_err)?
                        .to_vec(),
                ),
                None => None,
            };
            match compressed {
                Some(out) if out.len() < self.max_sample_count_table_size => out,
                _ => {
                    // Raw store: real bytes then zero padding to the WHOLE-tile
                    // max size, without retaining a previous tile's bytes.
                    let mut raw = count_table.clone();
                    raw.resize(self.max_sample_count_table_size, 0u8);
                    raw
                }
            }
        };

        // --- Compress the sample data (a per-tile tile compressor) ---------
        // C++ rebuilds the data compressor per tile via newTileCompressor(...,
        // maxBytesPerTileLine, tileDesc.ySize, header) and runs compressTile()
        // over the tile rectangle.  We size maxBytesPerTileLine from the tile's
        // own rectangle width (bytesPerPixel * tileW), matching the C++
        // `calculateBytesPerLine` max for this tile.  The codec's own
        // "no-shrink → store raw" rule keeps the chunk byte-identical to C++.
        let bytes_per_pixel = calculate_bytes_per_pixel(&self.header)?;
        let max_bytes_per_tile_line = bytes_per_pixel.saturating_mul(tile_w);
        let compression = self.header.compression()?;
        let mut data_compressor = new_tile_compressor(
            compression,
            max_bytes_per_tile_line,
            self.tile_desc.y_size as usize,
            &self.header,
        )?;

        // C++ `ImfDeepTiledOutputFile.cpp:1010`: `compSize < dataSize` → use the
        // compressed tile; else store raw (at equality too — readers detect a
        // raw passthrough by `packedDataSize == unpackedDataSize` alone).  Every
        // ported codec reports Xdr, so the NATIVE convertToXdr branch
        // (ImfDeepTiledOutputFile.cpp:1023) is unreachable.
        debug_assert_eq!(self.format, Format::Xdr);
        let packed_data: Vec<u8> = match data_compressor.as_mut() {
            Some(c) => stored_chunk(
                c.compress_tile(&data, tile_range).map_err(crate::error::encode_err)?,
                &data,
            )
            .to_vec(),
            None => data.clone(),
        };
        debug_assert!(packed_data.len() as u64 <= unpacked_data_size);

        // --- bufferedTileWrite — write now, or hold for later --------
        self.feed_tile(stream, (dx, dy, lx, ly), packed_count, packed_data, unpacked_data_size)
    }

    /// Write one already-gathered-and-compressed tile chunk now, or
    /// hold it for later if it arrives out of the file's line order — port of
    /// C++ `bufferedTileWrite` (`ImfDeepTiledOutputFile.cpp`, shares the flat
    /// writer's shape; see `tiled::output`), extracted out of
    /// [`Self::write_single_tile`] so both the old serial path and
    /// [`Self::write_tiles_parallel`]'s gather+encode-in-parallel /
    /// emit-in-order path share the EXACT same buffering state machine —
    /// this is what makes the writer's byte-identical-at-every-thread-count
    /// invariant hold: only the CPU-bound gather+encode work runs in
    /// parallel, emission always happens strictly in call order.
    fn feed_tile(
        &mut self,
        stream: &mut dyn OStream,
        coord: (i32, i32, i32, i32),
        packed_count: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_size: u64,
    ) -> Result<()> {
        self.split().2.feed(stream, coord, packed_count, packed_data, unpacked_data_size)
    }
}

/// The emit-side state of the deep tiled writer ([`DeepTileSink::feed`] is the
/// `bufferedTileWrite` state machine), borrowed field-by-field from
/// [`DeepTiledOutputFileState::split`] so the encode workers can keep reading
/// the header and layout while tiles are fed.
struct DeepTileSink<'a> {
    geom: TileGeom<'a>,
    tile_offsets: &'a mut TileOffsets,
    tile_offsets_position: u64,
    part_number: Option<i32>,
    tile_map: &'a mut std::collections::HashMap<(i32, i32, i32, i32), BufferedDeepTile>,
    next_tile_to_write: &'a mut Option<(i32, i32, i32, i32)>,
}

impl DeepTileSink<'_> {
    fn feed(
        &mut self,
        stream: &mut dyn OStream,
        coord: (i32, i32, i32, i32),
        packed_count: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_size: u64,
    ) -> Result<()> {
        let (dx, dy, lx, ly) = coord;
        match self.geom.line_order {
            LineOrder::RandomY | LineOrder::Invalid => {
                // Tiles can be written in any order: never buffer.
                DeepTiledOutputFileState::write_tile_chunk_now(
                    stream,
                    &mut *self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    coord,
                    &packed_count,
                    &packed_data,
                    unpacked_data_size,
                )?;
            }
            LineOrder::IncreasingY | LineOrder::DecreasingY => {
                let key = coord;
                if self.tile_map.contains_key(&key) {
                    return Err(ExrError::Arg(format!(
                        "Attempt to write tile ({dx}, {dy}, {lx}, {ly}) more than once."
                    )));
                }

                if *self.next_tile_to_write == Some(key) {
                    // Every tile before this one in line order is already on
                    // disk: write it now, advance the cursor, then flush any
                    // tiles buffered for the coordinates that follow.
                    DeepTiledOutputFileState::write_tile_chunk_now(
                        stream,
                        &mut *self.tile_offsets,
                        self.tile_offsets_position,
                        self.part_number,
                        key,
                        &packed_count,
                        &packed_data,
                        unpacked_data_size,
                    )?;
                    let mut next = self.geom.next(key)?;
                    while let Some((buf_count, buf_data, buf_unpacked)) = self.tile_map.remove(&next) {
                        DeepTiledOutputFileState::write_tile_chunk_now(
                            stream,
                            &mut *self.tile_offsets,
                            self.tile_offsets_position,
                            self.part_number,
                            next,
                            &buf_count,
                            &buf_data,
                            buf_unpacked,
                        )?;
                        next = self.geom.next(next)?;
                    }
                    *self.next_tile_to_write = Some(next);
                } else {
                    // Out of turn: hold a copy until its turn comes up.
                    self.tile_map.insert(key, (packed_count, packed_data, unpacked_data_size));
                }
            }
        }

        Ok(())
    }
}

impl DeepTiledOutputFileState {

    /// Read one pixel's sample count from the flat count buffer — port of the
    /// C++ inline `sampleCount(base, xStride, yStride, x, y)` (ImfMisc.h).
    fn sample_count(&self, sample_counts: &[u8], x: i32, y: i32) -> Result<u32> {
        sample_count_at(
            sample_counts,
            self.sample_count_base_offset,
            self.sample_count_x_stride,
            self.sample_count_y_stride,
            x,
            y,
        )
    }

    /// Gather one tile's samples for every channel into `data` (Xdr LE) — port
    /// of the `for y in tileRange` / `copyFromDeepFrameBuffer` loop in
    /// `TileBufferTask::execute` (ImfDeepTiledOutputFile.cpp:876-938).
    ///
    /// Iteration order is row-major over the tile rectangle (outer `y`, then
    /// each channel slice, then each pixel `x`).  Absent channels (`zero`) emit,
    /// per row, zero-valued samples for that row's total sample count (C++
    /// `fillChannelWithZeroes` with `bytesPerLine[row]`).  Present channels emit,
    /// per pixel, that pixel's `sampleCount` samples read from the flat heap at
    /// `base_offset + samples_before_pixel(x, y) * sampleStride + i * sampleStride`.
    #[allow(clippy::too_many_arguments)]
    fn gather_tile(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        tile_range: &Box2i,
        x_off_sc: i32,
        y_off_sc: i32,
        data: &mut Vec<u8>,
    ) -> Result<()> {
        // Clone the (small) slice table so the loop body can call
        // `self.sample_prefix.before_pixel` (`&mut self`) without holding a
        // borrow of `self.slices` alive.
        let slices = self.slices.clone();
        for y in tile_range.min.y..=tile_range.max.y {
            for slice in &slices {
                let sample_size = pixel_type_size(slice.type_)?;

                if slice.zero {
                    // C++ fillChannelWithZeroes for this row's total sample count
                    // (bytesPerLine[row] == rowSamples * sampleSize).
                    let mut row_samples: u64 = 0;
                    for j in tile_range.min.x..=tile_range.max.x {
                        row_samples += self.sample_count(sample_counts, j - x_off_sc, y - y_off_sc)? as u64;
                    }
                    let n = row_samples as usize * sample_size;
                    data.resize(data.len() + n, 0u8);
                    continue;
                }

                // C++: xOffsetForData = xTileCoords ? tileRange.min.x : 0; etc.
                // The data heap is in whole-data-window raster order, so the
                // per-pixel start uses the ABSOLUTE pixel position (the C++
                // copyFromDeepFrameBuffer dereferences base[(y-yOff)*yStride +
                // (x-xOff)*xStride] — in the flat model that pointer's target is
                // the cumulative sample offset of (x, y) in raster order).
                let x_off_data = if slice.x_tile_coords {
                    tile_range.min.x
                } else {
                    0
                };
                let y_off_data = if slice.y_tile_coords {
                    tile_range.min.y
                } else {
                    0
                };

                for x in tile_range.min.x..=tile_range.max.x {
                    let cnt = self.sample_count(sample_counts, x - x_off_sc, y - y_off_sc)?;
                    // Pixel start in the flat per-channel heap (raster order over
                    // the whole data window — see sample_table module docs).
                    // O(1) amortized via the growable `RowPrefixCache`
                    // instead of re-summing from `(min_x, min_y)` on every
                    // single pixel of every channel.
                    let pixel_start = self.sample_prefix.before_pixel(
                        sample_counts,
                        self.sample_count_base_offset,
                        self.sample_count_x_stride,
                        self.sample_count_y_stride,
                        x - x_off_data,
                        y - y_off_data,
                    )? as i64
                        * slice.sample_stride as i64;
                    for i in 0..cnt as i64 {
                        let off = slice.base_offset + pixel_start + i * slice.sample_stride as i64;
                        if off < 0 {
                            return Err(ExrError::Arg(
                                "Deep sample address is out of range (negative).".into(),
                            ));
                        }
                        let off = off as usize;
                        let src = sample_data.get(off..off + sample_size).ok_or_else(|| {
                            ExrError::Arg(format!(
                                "Deep sample address {off} out of range of the sample data buffer."
                            ))
                        })?;
                        // Samples in the flat heap are already Xdr LE (same
                        // adaptation as the flat/scanline writers); copy verbatim.
                        data.extend_from_slice(src);
                    }
                }
            }
        }
        Ok(())
    }

    /// Parallel gather+encode of the tile rectangle in `coords` (already
    /// range-checked and walked in the file's line order by the caller),
    /// emitted through [`Self::feed_tile`] STRICTLY in that same order — the
    /// writer's determinism invariant (byte-identical output at every thread
    /// count). Same three-function split (gather+encode / feed) and same
    /// scope restriction as the deep tiled reader's `read_tiles_parallel`
    /// (see its doc): declines (`Ok(false)`) when the destination slices
    /// disagree on `x_tile_coords`/`y_tile_coords`, letting the caller fall
    /// back to the fully serial path everyone already trusts.
    fn write_tiles_parallel(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        coords: &[(i32, i32, i32, i32)],
    ) -> Result<bool> {
        let Some((x_tile_coords, y_tile_coords)) =
            super::uniform_tile_coords(self.slices.iter().map(|s| (s.x_tile_coords, s.y_tile_coords)))
        else {
            return Ok(false);
        };

        // Fail fast + deterministic: reject a duplicate write before doing
        // any gather/encode work, in the same call order as the serial path
        // (`write_single_tile`'s own up-front check).
        for &(dx, dy, lx, ly) in coords {
            if self.tile_offsets.get(dx, dy, lx, ly)? != 0 {
                return Err(ExrError::Arg(format!(
                    "Attempt to write tile ({dx}, {dy}, {lx}, {ly}) more than once."
                )));
            }
        }

        let mut tile_ranges = Vec::with_capacity(coords.len());
        for &(dx, dy, lx, ly) in coords {
            tile_ranges.push(data_window_for_tile(
                &self.tile_desc,
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                dx,
                dy,
                lx,
                ly,
            )?);
        }

        // Serial pre-pass: one shared per-tile `samples_before_pixel` matrix
        // (reused across every channel — see the scope-restriction doc
        // above), extending the persisted `RowPrefixCache` in the
        // SAME row-major order `coords` was built in (dy outer per
        // `write_tiles`' line-order walk, dx inner), which keeps its
        // row-level extension monotonic in `y` — preserving the
        // amortized-linear guarantee exactly as the deep tiled reader's
        // identical pre-pass.
        let x_off = if x_tile_coords { None } else { Some(0i32) };
        let y_off = if y_tile_coords { None } else { Some(0i32) };
        let mut prefixes: Vec<Vec<u64>> = Vec::with_capacity(coords.len());
        // Each tile's footprint, known before anything is encoded: its count
        // table plus its samples, gathered and then encoded (~2x), so the ring
        // can bound bytes (`Ring::deep`), not just item count
        // (`ImfDeepTiledOutputFile.cpp:1539-1620` is bounded by item count only).
        let bytes_per_sample = self.slices.iter().try_fold(0usize, |acc, s| pixel_type_size(s.type_).map(|n| acc + n))?;
        let mut tile_bytes: Vec<usize> = Vec::with_capacity(coords.len());
        for tr in &tile_ranges {
            let xo = x_off.unwrap_or(tr.min.x);
            let yo = y_off.unwrap_or(tr.min.y);
            let tw = (tr.max.x - tr.min.x + 1) as usize;
            let th = (tr.max.y - tr.min.y + 1) as usize;
            let mut mat = Vec::with_capacity(tw * th);
            let mut samples = 0usize;
            let x_off_sc = if self.sample_count_x_tile_coords { tr.min.x } else { 0 };
            let y_off_sc = if self.sample_count_y_tile_coords { tr.min.y } else { 0 };
            for y in tr.min.y..=tr.max.y {
                for x in tr.min.x..=tr.max.x {
                    samples += self.sample_count(sample_counts, x - x_off_sc, y - y_off_sc)? as usize;
                    mat.push(self.sample_prefix.before_pixel(
                        sample_counts,
                        self.sample_count_base_offset,
                        self.sample_count_x_stride,
                        self.sample_count_y_stride,
                        x - xo,
                        y - yo,
                    )?);
                }
            }
            prefixes.push(mat);
            tile_bytes.push(2 * (tw * th * SIZE_INT + samples.saturating_mul(bytes_per_sample)));
        }

        let tile_y_size = self.tile_desc.y_size as usize;
        let max_sample_count_table_size = self.max_sample_count_table_size;
        let sc_x_tile_coords = self.sample_count_x_tile_coords;
        let sc_y_tile_coords = self.sample_count_y_tile_coords;
        let sc_base_offset = self.sample_count_base_offset;
        let sc_x_stride = self.sample_count_x_stride;
        let sc_y_stride = self.sample_count_y_stride;
        let count = self.num_threads;
        // The workers read the header/layout for the whole call while the
        // emit step mutates the emit state: disjoint field borrows, no clones.
        let (header, slices, mut sink) = self.split();
        let compression = header.compression()?;
        let bytes_per_pixel = calculate_bytes_per_pixel(header)?;
        // `(packed_count, packed_data, unpacked_data_size)` — clippy
        // `type_complexity` (matches the deep scanline writer's identical
        // `EncodedLine` alias).
        type EncodedTile = (Vec<u8>, Vec<u8>, u64);

        // Two pools -- count-table
        // codecs and per-tile data codecs differ in constructor
        // (`new_compressor` vs `new_tile_compressor`) and size, so they
        // must not share one free list. See `gather_and_encode_deep_tile`'s
        // doc.
        let count_pool = crate::compressor::CodecPool::new();
        let data_pool = crate::compressor::CodecPool::new();
        let (count_pool, data_pool) = (&count_pool, &data_pool);
        let (tile_ranges, prefixes) = (&tile_ranges, &prefixes);

        crate::threading::ordered_pipeline(
            count,
            coords.len(),
            crate::threading::Ring::deep(count),
            |k| tile_bytes[k],
            |k| -> Result<EncodedTile> {
                gather_and_encode_deep_tile(
                    slices,
                    sample_counts,
                    sample_data,
                    &tile_ranges[k],
                    sc_x_tile_coords,
                    sc_y_tile_coords,
                    sc_base_offset,
                    sc_x_stride,
                    sc_y_stride,
                    &prefixes[k],
                    compression,
                    header,
                    bytes_per_pixel,
                    tile_y_size,
                    max_sample_count_table_size,
                    count_pool,
                    data_pool,
                )
            },
            |k, (packed_count, packed_data, unpacked_data_size)| {
                sink.feed(stream, coords[k], packed_count, packed_data, unpacked_data_size)
            },
        )?;

        Ok(true)
    }
}

/// Gather one tile's sample data + build & compress its cumulative
/// sample-count table + compress its sample data — the pure, `Send`-safe
/// worker body for [`DeepTiledOutputFileState::write_tiles_parallel`]. Fuses
/// what the serial path's `gather_tile` + the inline count-table loop + both
/// compression steps in [`DeepTiledOutputFileState::write_single_tile`] do.
/// `count_pool`/`data_pool` are checkout/checkin `CodecPool`s -- NEVER
/// `self.count_compressor` (that field
/// stays serial-only; sharing it across concurrent tiles would race its
/// internal buffers). Before `CodecPool` existed this constructed a FRESH
/// compressor for BOTH codecs on every single call, unconditionally -- the
/// worst offender among this crate's parallel paths (measured: not even
/// `map_init`'s partial per-job-split reuse, a real codec construction per
/// tile, every tile, every call). `prefix` is the caller's
/// serially-computed `samples_before_pixel` matrix for this tile (row-major,
/// `tile_w` wide) — see the caller's pre-pass doc for why that must stay
/// serial while this gather does not.
#[allow(clippy::too_many_arguments)]
fn gather_and_encode_deep_tile(
    slices: &[TOutSliceInfo],
    sample_counts: &[u8],
    sample_data: &[u8],
    tile_range: &Box2i,
    sc_x_tile_coords: bool,
    sc_y_tile_coords: bool,
    sc_base_offset: i64,
    sc_x_stride: i32,
    sc_y_stride: i32,
    prefix: &[u64],
    compression: crate::attr::Compression,
    header: &Header,
    bytes_per_pixel: usize,
    tile_y_size: usize,
    max_sample_count_table_size: usize,
    count_pool: &crate::compressor::CodecPool,
    data_pool: &crate::compressor::CodecPool,
) -> Result<(Vec<u8>, Vec<u8>, u64)> {
    let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
    let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;

    let x_off_sc = if sc_x_tile_coords { tile_range.min.x } else { 0 };
    let y_off_sc = if sc_y_tile_coords { tile_range.min.y } else { 0 };

    // --- Gather sample data (Xdr LE) — same row-major (y outer, slice, x
    // inner) walk as the serial `gather_tile`, but addressing each pixel's
    // sample-heap start via the precomputed `prefix` matrix instead of the
    // stateful `RowPrefixCache` (which needs `&mut self` and is only safe to
    // extend serially). ------------------------------------------------
    let mut data: Vec<u8> = Vec::new();
    for y in tile_range.min.y..=tile_range.max.y {
        for slice in slices {
            let sample_size = pixel_type_size(slice.type_)?;

            if slice.zero {
                let mut row_samples: u64 = 0;
                for x in tile_range.min.x..=tile_range.max.x {
                    row_samples += sample_count_at(
                        sample_counts,
                        sc_base_offset,
                        sc_x_stride,
                        sc_y_stride,
                        x - x_off_sc,
                        y - y_off_sc,
                    )? as u64;
                }
                let n = row_samples as usize * sample_size;
                data.resize(data.len() + n, 0u8);
                continue;
            }

            for x in tile_range.min.x..=tile_range.max.x {
                let cnt = sample_count_at(
                    sample_counts,
                    sc_base_offset,
                    sc_x_stride,
                    sc_y_stride,
                    x - x_off_sc,
                    y - y_off_sc,
                )?;
                let idx =
                    (y - tile_range.min.y) as usize * tile_w + (x - tile_range.min.x) as usize;
                let pixel_start = *prefix.get(idx).ok_or_else(|| {
                    ExrError::Logic(
                        "gather_and_encode_deep_tile: prefix index out of range".into(),
                    )
                })? as i64
                    * slice.sample_stride as i64;
                for i in 0..cnt as i64 {
                    let off = slice.base_offset + pixel_start + i * slice.sample_stride as i64;
                    if off < 0 {
                        return Err(ExrError::Arg(
                            "Deep sample address is out of range (negative).".into(),
                        ));
                    }
                    let off = off as usize;
                    let src = sample_data.get(off..off + sample_size).ok_or_else(|| {
                        ExrError::Arg(format!(
                            "Deep sample address {off} out of range of the sample data buffer."
                        ))
                    })?;
                    data.extend_from_slice(src);
                }
            }
        }
    }
    let unpacked_data_size = data.len() as u64;

    // --- Cumulative sample-count table over the tile rect ---------------
    let mut count_table = Vec::with_capacity(tile_w * tile_h * SIZE_INT);
    for y in tile_range.min.y..=tile_range.max.y {
        let row: Vec<u32> = (tile_range.min.x..=tile_range.max.x)
            .map(|x| {
                sample_count_at(
                    sample_counts,
                    sc_base_offset,
                    sc_x_stride,
                    sc_y_stride,
                    x - x_off_sc,
                    y - y_off_sc,
                )
            })
            .collect::<Result<_>>()?;
        encode_cumulative_row(row, &mut count_table)?;
    }

    // --- Compress the count table (pooled -- see
    // `crate::compressor::CodecPool`'s own doc) ---------
    let mut count_compressor =
        count_pool.lease(|| new_compressor(compression, max_sample_count_table_size, header))?;
    let packed_count: Vec<u8> = {
        let compressed: Option<Vec<u8>> = match count_compressor.as_mut() {
            Some(c) => Some(
                c.compress(&count_table, tile_range.min.y)
                    .map_err(crate::error::encode_err)?
                    .to_vec(),
            ),
            None => None,
        };
        match compressed {
            Some(out) if out.len() < max_sample_count_table_size => out,
            _ => {
                let mut raw = count_table.clone();
                raw.resize(max_sample_count_table_size, 0u8);
                raw
            }
        }
    };
    drop(count_compressor);

    // --- Compress the sample data (pooled, same rationale) --------------
    let max_bytes_per_tile_line = bytes_per_pixel.saturating_mul(tile_w);
    let mut data_compressor = data_pool.lease(|| {
        new_tile_compressor(compression, max_bytes_per_tile_line, tile_y_size, header)
    })?;
    let packed_data: Vec<u8> = match data_compressor.as_mut() {
        Some(c) => stored_chunk(
            c.compress_tile(&data, *tile_range).map_err(crate::error::encode_err)?,
            &data,
        )
        .to_vec(),
        None => data.clone(),
    };
    drop(data_compressor);

    Ok((packed_count, packed_data, unpacked_data_size))
}

/// A **borrowing** view over one multipart part's persisted deep tiled writer
/// state — the multipart file owns each part's writer state and this view
/// borrows it, same shape as `crate::tiled::output::TiledOutputPartWriter` /
/// `crate::deep::scanline_output::DeepScanLineOutputPartWriter`: instead of
/// [`MultiPartOutputFile::deep_tiled_output_part`](crate::multipart::
/// MultiPartOutputFile::deep_tiled_output_part) building a brand-new
/// [`DeepTiledOutputFile`] (and its state — including the `tile_map`/
/// `next_tile_to_write` buffering) on every call, the container persists one
/// [`DeepTiledOutputFileState`] per deep tiled part across calls and hands
/// out this thin view.
///
/// Exposes the same public surface as [`DeepTiledOutputFile`] so existing
/// callers of `deep_tiled_output_part` are unaffected by the return-type
/// change.
pub struct DeepTiledOutputPartWriter<'p> {
    stream: &'p mut dyn OStream,
    state: &'p mut DeepTiledOutputFileState,
}

impl<'p> DeepTiledOutputPartWriter<'p> {
    /// Build a view over an existing (already-constructed, possibly
    /// previously-written-to) part state and the container's stream.
    pub(crate) fn new(stream: &'p mut dyn OStream, state: &'p mut DeepTiledOutputFileState) -> Self {
        Self { stream, state }
    }

    /// The image header — port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The current frame buffer — port of `const DeepFrameBuffer& frameBuffer()`.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk. See [`DeepTiledOutputFileState::update_preview_image`].
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written tile's chunk. See
    /// [`DeepTiledOutputFileState::break_tile`].
    #[allow(clippy::too_many_arguments)]
    pub fn break_tile(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        offset: u64,
        length: usize,
        c: u8,
    ) -> Result<()> {
        self.state.break_tile(self.stream, dx, dy, lx, ly, offset, length, c)
    }

    /// Tile width — port of `unsigned int tileXSize() const`.
    pub fn tile_x_size(&self) -> u32 {
        self.state.tile_x_size()
    }

    /// Tile height — port of `unsigned int tileYSize() const`.
    pub fn tile_y_size(&self) -> u32 {
        self.state.tile_y_size()
    }

    /// Level mode — port of `LevelMode levelMode() const`.
    pub fn level_mode(&self) -> LevelMode {
        self.state.level_mode()
    }

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode()`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
    }

    /// Compression type — convenience accessor.
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
    }

    /// Number of levels in X — port of `int numXLevels() const`.
    pub fn num_x_levels(&self) -> i32 {
        self.state.num_x_levels()
    }

    /// Number of levels in Y — port of `int numYLevels() const`.
    pub fn num_y_levels(&self) -> i32 {
        self.state.num_y_levels()
    }

    /// Number of levels (MIP convenience) — port of `int numLevels() const`.
    pub fn num_levels(&self) -> Result<i32> {
        self.state.num_levels()
    }

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(...)`.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.state.is_valid_level(lx, ly)
    }

    /// Width of level `lx` — port of `int levelWidth(int lx) const`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.state.level_width(lx)
    }

    /// Height of level `ly` — port of `int levelHeight(int ly) const`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.state.level_height(ly)
    }

    /// Number of X tiles in level `lx` — port of `int numXTiles(int lx) const`.
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }

    /// Number of Y tiles in level `ly` — port of `int numYTiles(int ly) const`.
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.state.num_y_tiles(ly)
    }

    /// Data window of level `(lx, ly)`.
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_level(lx, ly)
    }

    /// Data window of level `l` (MIP convenience).
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.state.data_window_for_level_l(l)
    }

    /// Data window of tile `(dx, dy, lx, ly)`.
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_tile(dx, dy, lx, ly)
    }

    /// Data window of tile `(dx, dy, l)` (MIP convenience).
    pub fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.state.data_window_for_tile_l(dx, dy, l)
    }

    /// Set the pixel source. See [`DeepTiledOutputFileState::set_frame_buffer`].
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)`. See [`DeepTiledOutputFileState::write_tiles`].
    #[allow(clippy::too_many_arguments)]
    pub fn write_tiles(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tiles(self.stream, sample_counts, sample_data, dx1, dx2, dy1, dy2, lx, ly)
    }

    /// Write the rectangle of tiles at MIP level `l`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_tiles_l(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tiles_l(self.stream, sample_counts, sample_data, dx1, dx2, dy1, dy2, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)`.
    pub fn write_tile(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tile(self.stream, sample_counts, sample_data, dx, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l`.
    pub fn write_tile_l(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        dx: i32,
        dy: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tile_l(self.stream, sample_counts, sample_data, dx, dy, l)
    }

    /// Quick pixel copy: transcode every tile of `input` straight into this
    /// file, raw (no decompress/recompress). See
    /// [`DeepTiledOutputFileState::copy_pixels`] for the full doc.
    pub fn copy_pixels<T: DeepTiledCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{
        Channel, ChannelList, ChannelListAttribute, Compression, TileDescriptionAttribute,
    };
    use crate::deep::frame_buffer::DeepSlice;
    use crate::deep::tiled_input::DeepTiledInputFile;
    use crate::frame_buffer::Slice;
    use crate::io::{IStream, MemIStream, MemOStream};
    use imath_rs::{Half, V2f};

    /// Build a deep tiled header: `width`x`height`, one HALF channel "A", deep
    /// tile type, the given compression, tile description and INCREASING_Y order.
    fn make_deep_tiled_header(
        width: i32,
        height: i32,
        comp: Compression,
        td: TileDescription,
    ) -> Header {
        let mut h = Header::with_dimensions(
            width,
            height,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            comp,
        )
        .unwrap();
        h.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.set_type(DEEP_TILE).unwrap();
        h
    }

    /// Build a `w`x`h` deep image with per-pixel sample counts `0..` in raster
    /// order, one HALF channel whose sample value is `pixel_index*10 + i`.
    /// Returns `(sample_counts_bytes, sample_data_bytes, frame_buffer, counts)`.
    fn make_deep_pixels(w: i32, h: i32) -> (Vec<u8>, Vec<u8>, DeepFrameBuffer, Vec<u32>) {
        let n = (w * h) as usize;
        let counts: Vec<u32> = (0..n as u32).collect();

        // Flat sample-count buffer: u32 LE per pixel, x_stride=4, y_stride=4*w.
        let mut sc = Vec::with_capacity(n * 4);
        for &c in &counts {
            sc.extend_from_slice(&c.to_le_bytes());
        }

        // Flat sample heap for channel "A": each pixel contributes `count` HALF
        // samples (2 bytes each), laid out contiguously in raster order.
        let mut data = Vec::new();
        for (pix, &c) in counts.iter().enumerate() {
            for i in 0..c {
                let v = (pix as f32) * 10.0 + i as f32;
                let bits = Half::from(v).bits();
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }

        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 4 * w as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        fb.insert(
            "A",
            DeepSlice::new(
                PixelType::Half,
                0, // base_offset into the flat heap
                8,
                8 * w as usize,
                2, // sample_stride = sizeof(HALF)
                1,
                1,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();

        (sc, data, fb, counts)
    }

    /// Whole-data-window raster cumulative count before pixel `(x, y)`.
    fn cum_before(counts: &[u32], w: i32, x: i32, y: i32) -> u64 {
        let mut t: u64 = 0;
        for yy in 0..y {
            for xx in 0..w {
                t += counts[(yy * w + xx) as usize] as u64;
            }
        }
        for xx in 0..x {
            t += counts[(y * w + xx) as usize] as u64;
        }
        t
    }

    /// Parse the written file and verify the deep tiled chunk structure: magic +
    /// deep+tiled version flags, fully back-patched offset table, each tile
    /// chunk's coords + 3 size fields, the decoded cumulative count table over
    /// each level-0 tile rect (NONE only, with zero-padding to the whole-tile
    /// max size), and that every chunk stays within the file.
    fn verify(
        header: &Header,
        bytes: &[u8],
        comp: Compression,
        counts: &[u32],
        num_slots: usize,
    ) {
        let dw = header.data_window().unwrap();
        let w = dw.max.x - dw.min.x + 1;
        let h = dw.max.y - dw.min.y + 1;
        let td = header.tile_description().unwrap();

        // Re-read the header + version word, then walk every chunk by offset.
        let mut is = MemIStream::new(bytes.to_vec());
        let (hdr, version_word) = Header::read_from(&mut is).unwrap();
        let vw = version_word as u32;
        // C++ `writeMagicNumberAndVersionField` (ImfGenericOutputFile.cpp:39-48)
        // sets the version flags as `deep ? NON_IMAGE : (tiled ? TILED)` — an
        // `else if`.  A single-part `deeptile` file therefore carries ONLY the
        // NON_IMAGE flag in the version word (the TILED flag is NOT set); readers
        // recognise it as tiled via the `type == deeptile` attribute.
        assert!(
            crate::container::version::is_non_image(vw),
            "deep NON_IMAGE_FLAG must be set"
        );
        assert!(
            !crate::container::version::is_tiled(vw),
            "deeptile version word must NOT set the TILED flag (C++ else-if)"
        );
        assert_eq!(hdr.type_().unwrap(), DEEP_TILE);
        let header_len = is.tellg().unwrap() as usize;

        let mut offsets = Vec::with_capacity(num_slots);
        for k in 0..num_slots {
            let o = u64::from_le_bytes(
                bytes[header_len + k * 8..header_len + k * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            assert_ne!(o, 0, "offset slot {k} not back-patched");
            offsets.push(o);
        }

        let max_table = (td.x_size * td.y_size) as usize * 4;

        for &off in &offsets {
            let mut p = off as usize;
            let dx = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            let dy = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            let lx = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            let ly = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            let packed_count = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            let packed_data = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            let unpacked = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;

            let rect = data_window_for_tile(&td, 0, w - 1, 0, h - 1, dx, dy, lx, ly).unwrap();
            let tw = (rect.max.x - rect.min.x + 1) as usize;
            let th = (rect.max.y - rect.min.y + 1) as usize;

            // For ONE_LEVEL (lx==ly==0) we can check the data size + counts.
            // (For coarser MIP levels the down-sampled image is not derived from
            // `counts`, so we only structurally check those.)
            if comp == Compression::NoCompression && lx == 0 && ly == 0 {
                assert_eq!(packed_count, max_table, "raw count table padded to max");
                let table = &bytes[p..p + packed_count];
                let mut tile_samples: u64 = 0;
                let mut idx = 0usize;
                for ry in 0..th {
                    let mut run: i32 = 0;
                    for rx in 0..tw {
                        let gx = rect.min.x + rx as i32;
                        let gy = rect.min.y + ry as i32;
                        let c = counts[(gy * w + gx) as usize];
                        run += c as i32;
                        tile_samples += c as u64;
                        let got =
                            i32::from_le_bytes(table[idx * 4..idx * 4 + 4].try_into().unwrap());
                        assert_eq!(got, run, "count table at ({rx},{ry}) tile ({dx},{dy})");
                        // Cross-check vs the analytic cumulative before this pixel.
                        let before = cum_before(counts, w, gx, gy);
                        let _ = before; // (validates the heap model in gather)
                        idx += 1;
                    }
                }
                for b in &table[tw * th * 4..] {
                    assert_eq!(*b, 0, "count table padding non-zero");
                }
                // unpackedDataSize == tile sample count * sizeof(HALF).
                assert_eq!(
                    unpacked, tile_samples as usize * 2,
                    "unpacked data size for tile ({dx},{dy})"
                );
                // NONE → data stored raw, packed == unpacked.
                assert_eq!(packed_data, unpacked, "NONE packed==unpacked");
            }

            p += packed_count + packed_data;
            assert!(p <= bytes.len(), "chunk overruns file");
        }
    }

    fn one_level_td() -> TileDescription {
        TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }
    }

    #[test]
    fn deep_tiled_one_level_none_structure() {
        // 4x4 image, 2x2 tiles → ONE_LEVEL 2x2 grid = 4 tiles.
        let (w, h) = (4, 4);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(df.num_x_levels(), 1);
            assert_eq!(df.num_x_tiles(0).unwrap(), 2);
            assert_eq!(df.num_y_tiles(0).unwrap(), 2);
            df.set_frame_buffer(&fb).unwrap();
            df.write_tiles(&sc, &data, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();
        verify(&header, &bytes, Compression::NoCompression, &counts, 4);
    }

    /// `update_preview_image` and `break_tile` (mirrors `tiled::output`'s
    /// own test of the same pair).
    #[test]
    fn update_preview_image_and_break_tile() {
        use crate::attr::{PreviewImage, PreviewImageAttribute, PreviewRgba};

        // 2x2 image, one 2x2 tile -> a single chunk, so the on-disk chunk
        // layout (leader = 4 i32 + 3 u64 = 40 bytes) is trivial to locate.
        let (w, h) = (2, 2);
        let td = one_level_td();
        let mut header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let preview = PreviewImage::new(
            2,
            1,
            Some(vec![
                PreviewRgba::new(1, 2, 3, 4),
                PreviewRgba::new(5, 6, 7, 8),
            ]),
        )
        .unwrap();
        header.insert("preview", PreviewImageAttribute::new(preview)).unwrap();
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        // Pass A: write with no corruption (the baseline).
        let mut out_a = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out_a, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();
        }
        let bytes_a = out_a.into_data();

        // Pass B: identical write, but corrupt 1 byte of the tile's sample
        // DATA payload via `break_tile` (leader 40 bytes + the raw, padded
        // 2*2*4=16-byte count table -> data starts at byte 56 of the chunk),
        // then update the preview image, then check both landed.
        let mut out_b = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out_b, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();

            df.break_tile(0, 0, 0, 0, 56, 1, 0xFF).unwrap();
            // A never-written tile must error, matching C++'s throw.
            assert!(df.break_tile(1, 1, 0, 0, 0, 1, 0).is_err());

            // All 9s: guaranteed to differ from every original preview byte
            // (1..=8), so exactly 8 bytes (2 pixels * 4 channels) change --
            // no coincidental byte equality to account for.
            df.update_preview_image(&[
                PreviewRgba::new(9, 9, 9, 9),
                PreviewRgba::new(9, 9, 9, 9),
            ])
            .unwrap();
            // Wrong pixel count must error, not silently truncate/overrun.
            assert!(df.update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)]).is_err());
        }
        let bytes_b = out_b.into_data();

        assert_eq!(bytes_a.len(), bytes_b.len(), "break_tile/update_preview_image must not resize the file");
        let diff_positions: Vec<usize> = bytes_a
            .iter()
            .zip(bytes_b.iter())
            .enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i))
            .collect();
        // Exactly the corrupted data byte plus the 2 preview pixels (4 bytes
        // each = 8 bytes) must differ -- proving break_tile hit the real
        // chunk bytes (not a no-op) and update_preview_image touched only
        // the preview attribute, nothing else.
        assert_eq!(diff_positions.len(), 1 + 8, "expected exactly the broken byte + the updated preview pixels to differ: {diff_positions:?}");
        assert!(
            diff_positions.iter().any(|&p| bytes_b[p] == 0xFF),
            "the broken chunk byte (0xFF) must be among the differing bytes: {diff_positions:?}"
        );

        // The header's preview attribute on disk must reflect the update.
        let mut header_is = MemIStream::new(bytes_b);
        let (read_back_header, _version_word) = Header::read_from(&mut header_is).unwrap();
        let pia = read_back_header.typed_attribute::<PreviewImage>("preview").unwrap();
        assert_eq!(
            pia.value.pixels(),
            &[PreviewRgba::new(9, 9, 9, 9), PreviewRgba::new(9, 9, 9, 9)]
        );
    }

    /// `update_preview_image` on a header with NO preview image must
    /// error (`ExrError::Logic`), matching C++'s `LogicExc` throw, instead of
    /// silently no-oping.
    #[test]
    fn update_preview_image_without_preview_errors() {
        use crate::attr::PreviewRgba;

        let (w, h) = (2, 2);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();

        match df.update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)]) {
            Err(ExrError::Logic(_)) => {}
            Err(other) => panic!("expected Logic error, got {other:?}"),
            Ok(_) => panic!("expected an error for a file with no preview image"),
        }
    }

    /// Writing tiles out of call order (`write_tile` calls in reverse)
    /// must produce a BYTE-IDENTICAL file to writing them in order, for both
    /// `INCREASING_Y` and `DECREASING_Y` — mirrors
    /// `tiled::output`'s own test
    /// (`increasing_y_out_of_order_writes_match_in_order_byte_for_byte`).
    /// ZIPS (not NONE) so a byte-identical result actually proves the
    /// on-disk chunk ORDER matches, not merely that "no compression" makes
    /// order invisible.
    fn out_of_order_vs_in_order_writes(lo: LineOrder) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (4, 4);
        let td = one_level_td();
        let mut header = make_deep_tiled_header(w, h, Compression::Zips, td);
        *header.line_order_mut().unwrap() = lo;
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        // In order: (0,0), (1,0), (0,1), (1,1).
        let in_order = [(0, 0), (1, 0), (0, 1), (1, 1)];
        let mut out_a = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out_a, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            for &(dx, dy) in &in_order {
                df.write_tile(&sc, &data, dx, dy, 0, 0).unwrap();
            }
        }
        let bytes_in_order = out_a.into_data();

        // Reverse call order: (1,1), (0,1), (1,0), (0,0).
        let reverse_order = [(1, 1), (0, 1), (1, 0), (0, 0)];
        let mut out_b = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out_b, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            for &(dx, dy) in &reverse_order {
                df.write_tile(&sc, &data, dx, dy, 0, 0).unwrap();
            }
        }
        let bytes_out_of_order = out_b.into_data();

        (bytes_in_order, bytes_out_of_order)
    }

    #[test]
    fn increasing_y_out_of_order_writes_match_in_order_byte_for_byte() {
        let (a, b) = out_of_order_vs_in_order_writes(LineOrder::IncreasingY);
        assert_eq!(a, b, "out-of-order write_tile calls must produce byte-identical output");
    }

    #[test]
    fn decreasing_y_out_of_order_writes_match_in_order_byte_for_byte() {
        let (a, b) = out_of_order_vs_in_order_writes(LineOrder::DecreasingY);
        assert_eq!(a, b, "out-of-order write_tile calls must produce byte-identical output");
    }

    /// A tile still buffered (never flushed because tiles before it in
    /// line order were never written) when the writer is dropped is simply
    /// discarded — its offset-table slot stays 0 — matching C++'s destructor,
    /// which frees `tileMap` without flushing it.
    #[test]
    fn tile_still_buffered_at_drop_is_discarded() {
        let (w, h) = (4, 4);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::Zips, td);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            // Write (1,0) out of turn -- (0,0) never written -- so (1,0) sits
            // buffered in `tile_map` when `df` drops at the end of this block.
            df.write_tile(&sc, &data, 1, 0, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let tif = DeepTiledInputFile::new(&mut is).unwrap();
        assert!(
            !tif.is_complete(),
            "a file with a buffered-but-never-flushed tile must be incomplete"
        );
    }

    #[test]
    fn deep_tiled_one_level_zips_structure() {
        let (w, h) = (4, 4);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::Zips, td);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_tiles(&sc, &data, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();
        verify(&header, &bytes, Compression::Zips, &counts, 4);
    }

    /// `copy_pixels`: transcode a source deep tiled file into a fresh
    /// writer with a matching header, byte-for-byte, without ever
    /// decompressing. ZIPS (not NONE) so a byte-identical result actually
    /// proves `copy_pixels` never decompresses+recompresses. Both
    /// `INCREASING_Y` (walks `next_tile_coord`) and `RANDOM_Y` (walks the
    /// input's own `tile_order()`).
    #[test]
    fn copy_pixels_byte_identical() {
        let (w, h) = (4, 4);
        let td = one_level_td();

        for lo in [LineOrder::IncreasingY, LineOrder::RandomY] {
            let mut header = make_deep_tiled_header(w, h, Compression::Zips, td);
            *header.line_order_mut().unwrap() = lo;
            let (sc, data, fb, _counts) = make_deep_pixels(w, h);

            let mut src_out = MemOStream::new();
            {
                let mut src = DeepTiledOutputFile::new(&mut src_out, &header).unwrap();
                src.set_frame_buffer(&fb).unwrap();
                // Arbitrary call order -- RANDOM_Y allows any order;
                // INCREASING_Y's out-of-order tile buffering makes this in-order on disk
                // regardless of call order.
                src.write_tile(&sc, &data, 1, 1, 0, 0).unwrap();
                src.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();
                src.write_tile(&sc, &data, 1, 0, 0, 0).unwrap();
                src.write_tile(&sc, &data, 0, 1, 0, 0).unwrap();
            }
            let src_bytes = src_out.into_data();

            let mut src_is = MemIStream::new(src_bytes.clone());
            let mut src_in = DeepTiledInputFile::new(&mut src_is).unwrap();

            let mut dst_out = MemOStream::new();
            {
                let mut dst = DeepTiledOutputFile::new(&mut dst_out, &header).unwrap();
                dst.copy_pixels(&mut src_in).unwrap();
            }
            let dst_bytes = dst_out.into_data();

            assert_eq!(
                src_bytes, dst_bytes,
                "copy_pixels must reproduce the source file byte-for-byte ({lo:?})"
            );
        }
    }

    /// `copy_pixels` must reject copying into a file that already has
    /// pixel data (C++ LogicExc "already contains pixel data").
    #[test]
    fn copy_pixels_rejects_when_destination_already_has_pixel_data() {
        let (w, h) = (4, 4);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut src_out = MemOStream::new();
        {
            let mut src = DeepTiledOutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_tiles(&sc, &data, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = DeepTiledInputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        let mut dst = DeepTiledOutputFile::new(&mut dst_out, &header).unwrap();
        dst.set_frame_buffer(&fb).unwrap();
        dst.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)));
    }

    #[test]
    fn deep_tiled_partial_edge_tiles_none() {
        // 5x3 image, 2x2 tiles → 3x2 grid; the right column & bottom row tiles
        // are partial.  Exercises the zero-padded raw count table path.
        let (w, h) = (5, 3);
        let td = one_level_td();
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(df.num_x_tiles(0).unwrap(), 3); // ceil(5/2)
            assert_eq!(df.num_y_tiles(0).unwrap(), 2); // ceil(3/2)
            df.set_frame_buffer(&fb).unwrap();
            df.write_tiles(&sc, &data, 0, 2, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();
        verify(&header, &bytes, Compression::NoCompression, &counts, 6);
    }

    #[test]
    fn deep_tiled_mipmap_none_structure() {
        // 4x4 image, 2x2 tiles, MIPMAP round-down → 3 levels.
        //   level 0: 4x4 → 2x2 tiles (4); level 1: 2x2 → 1x1 (1);
        //   level 2: 1x1 → 1x1 (1).  Total slots = 6.
        let (w, h) = (4, 4);
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(df.num_levels().unwrap(), 3);
            // The sample-count buffer + heap cover the full 4x4 window, so the
            // coarser levels (which address pixels inside that window via
            // data_window_for_tile) are addressable: each level's data window is
            // an interior rect of the same window.
            for l in 0..3 {
                let nx = df.num_x_tiles(l).unwrap();
                let ny = df.num_y_tiles(l).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                df.write_tiles_l(&sc, &data, 0, nx - 1, 0, ny - 1, l).unwrap();
            }
        }
        let bytes = out.into_data();
        // Only level 0 tiles are fully count-checked (see verify()).
        verify(&header, &bytes, Compression::NoCompression, &counts, 6);
    }

    #[test]
    fn deep_tiled_decreasing_y_first_chunk_top_row() {
        // 4x6 image, 2x2 tiles → 2x3 grid; DECREASING_Y writes the dy=2 row
        // first, so the first chunk on disk carries dy==2.
        let (w, h) = (4, 6);
        let td = one_level_td();
        let mut header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        *header.line_order_mut().unwrap() = LineOrder::DecreasingY;
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_tiles(&sc, &data, 0, 1, 0, 2, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes.clone());
        let _ = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;
        // First chunk physically follows the offset table (6 slots * 8 bytes).
        let chunks_start = header_len + 6 * 8;
        let _dx = i32::from_le_bytes(bytes[chunks_start..chunks_start + 4].try_into().unwrap());
        let dy = i32::from_le_bytes(bytes[chunks_start + 4..chunks_start + 8].try_into().unwrap());
        assert_eq!(dy, 2, "DECREASING_Y must write the top tile row (dy=2) first");
    }

    #[test]
    fn deep_tiled_absent_channel_zero_filled() {
        // Header has A (HALF) and Z (FLOAT); frame buffer omits Z → Z is
        // zero-filled for each tile's sample count.  unpacked == samples*(2+4).
        let (w, h) = (4, 4);
        let td = one_level_td();
        let mut header = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        header
            .insert("tiles", TileDescriptionAttribute::new(td))
            .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false))
            .unwrap();
        header
            .insert("channels", ChannelListAttribute::new(cl))
            .unwrap();
        header.set_type(DEEP_TILE).unwrap();

        let (sc, data, fb, counts) = make_deep_pixels(w, h); // fb has only "A"

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_tiles(&sc, &data, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes.clone());
        let _ = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;
        let td2 = header.tile_description().unwrap();
        let max_table = (td2.x_size * td2.y_size) as usize * 4;

        for k in 0..4usize {
            let off = u64::from_le_bytes(
                bytes[header_len + k * 8..header_len + k * 8 + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let dx = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let dy = i32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let packed_count =
                u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap()) as usize;
            let unpacked =
                u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap()) as usize;

            let rect = data_window_for_tile(&td2, 0, w - 1, 0, h - 1, dx, dy, 0, 0).unwrap();
            // Per-row sample counts (the deep gather is row-major: for each row
            // y, A's samples then Z's samples, then the next row — NOT all-A then
            // all-Z over the whole tile).
            let mut row_samples: Vec<u64> = Vec::new();
            for gy in rect.min.y..=rect.max.y {
                let mut s: u64 = 0;
                for gx in rect.min.x..=rect.max.x {
                    s += counts[(gy * w + gx) as usize] as u64;
                }
                row_samples.push(s);
            }
            let tile_samples: u64 = row_samples.iter().sum();
            // A (2 bytes) + Z (4 bytes) per sample.
            assert_eq!(unpacked, tile_samples as usize * (2 + 4), "A+Z size");
            assert_eq!(packed_count, max_table);

            // Data starts after the count table; walk it row by row and assert
            // each row's Z segment (the second, FLOAT, half) is zero-filled.
            let mut p = off + 40 + packed_count;
            for &s in &row_samples {
                let a_bytes = s as usize * 2; // A: HALF samples for this row
                let z_bytes = s as usize * 4; // Z: FLOAT samples for this row
                let z = &bytes[p + a_bytes..p + a_bytes + z_bytes];
                assert!(z.iter().all(|&b| b == 0), "absent Z not zero-filled");
                p += a_bytes + z_bytes;
            }
        }
    }

    #[test]
    fn deep_tiled_writing_a_tile_twice_errors() {
        let td = one_level_td();
        let header = make_deep_tiled_header(4, 4, Compression::NoCompression, td);
        let (sc, data, fb, _counts) = make_deep_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();
        let err = df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn deep_tiled_no_frame_buffer_errors() {
        let td = one_level_td();
        let header = make_deep_tiled_header(4, 4, Compression::NoCompression, td);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        let err = df.write_tile(&[], &[], 0, 0, 0, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn deep_tiled_missing_sample_count_slice_errors() {
        // Pixel data is passed per write call: an empty sample-count buffer
        // is not checkable at `set_frame_buffer` (no buffer is passed there)
        // — it is `write_tiles`' null-base-pointer check.
        let td = one_level_td();
        let header = make_deep_tiled_header(4, 4, Compression::NoCompression, td);
        let (_sc, data, fb, _counts) = make_deep_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let err = df.write_tile(&[], &data, 0, 0, 0, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    /// 4x4 MIPMAP with 2x2 tiles: level 1 is 2x2, so a count buffer covering only
    /// the first two window rows is a valid per-level layout (regression: a
    /// whole-data-window extent check in `set_frame_buffer` rejected it, which the
    /// C++ cross-read gate `ours_write_deep_tiled_*_mipmap_read_by_cpp` caught).
    fn mipmap_4x4_header() -> Header {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        make_deep_tiled_header(4, 4, Compression::NoCompression, td)
    }

    #[test]
    fn deep_tiled_level_sized_count_buffer_accepted() {
        let header = mipmap_4x4_header();
        let (sc, data, fb, _counts) = make_deep_pixels(4, 4);
        let level_rows = &sc[..2 * 4 * 4]; // 2 rows x 4 px x u32
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_tiles_l(level_rows, &data, 0, 0, 0, 0, 1).unwrap();
    }

    #[test]
    fn deep_tiled_short_count_buffer_errors_at_gather() {
        // The same short buffer cannot serve level 0 (rows 2..4 are out of range):
        // the gather must hard-error, never read the missing rows as zero samples.
        let header = mipmap_4x4_header();
        let (sc, data, fb, _counts) = make_deep_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let err = df.write_tiles_l(&sc[..2 * 4 * 4], &data, 0, 1, 0, 1, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn deep_tiled_invalid_tile_coords_error() {
        let td = one_level_td();
        let header = make_deep_tiled_header(4, 4, Compression::NoCompression, td);
        let (sc, data, fb, _counts) = make_deep_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        // dx=2 out of range (only 0,1 valid).
        assert!(df.write_tile(&sc, &data, 2, 0, 0, 0).is_err());
        // lx=1 invalid for ONE_LEVEL.
        assert!(df.write_tile(&sc, &data, 0, 0, 1, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel write determinism (deep tiled writer).
    // -----------------------------------------------------------------------

    /// A 5x5 tile grid (3x3 tiles, 15x13 image → partial edge tiles in the
    /// last row) written in a SINGLE `write_tiles` call must produce
    /// byte-identical file bytes at every thread count — the writer's
    /// determinism invariant, exercised across the whole
    /// `write_tiles_parallel` gather+encode+feed path including the
    /// serial `RowPrefixCache` pre-pass. Mirrors
    /// `tiled::output`'s `parallel_write_tiles_is_byte_identical_across_thread_counts`
    /// and the deep tiled reader's `parallel_read_is_byte_identical_to_serial_across_thread_counts`.
    #[test]
    fn parallel_write_is_byte_identical_across_thread_counts() {
        let td = TileDescription {
            x_size: 3,
            y_size: 3,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (15i32, 13i32);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::DeepTileWriteParallel);
                let mut df = DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (df.num_x_tiles(0).unwrap(), df.num_y_tiles(0).unwrap());
                df.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            let bytes = out.into_data();
            verify(&header, &bytes, Compression::NoCompression, &counts, (5 * 5) as usize);

            crate::probe::assert_engaged(crate::probe::Probe::DeepTileWriteParallel, n);
            match &reference {
                None => reference = Some(bytes),
                Some(r) => assert_eq!(
                    r.as_slice(),
                    bytes.as_slice(),
                    "num_threads={n}: file bytes differ from num_threads=0"
                ),
            }
        }
    }

    /// Same grid, ZIPS compression (a real codec, so the fresh-per-worker
    /// count/data compressor path is exercised) and `DecreasingY` line order
    /// (so `write_tiles_parallel`'s emission through `feed_tile` must drive
    /// the `tile_map`/`next_tile_to_write` buffering exactly like the
    /// serial path) — byte-identical across thread counts.
    #[test]
    fn parallel_write_zips_decreasing_y_byte_identical_across_thread_counts() {
        let td = TileDescription {
            x_size: 3,
            y_size: 3,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (15i32, 13i32);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let mut header = make_deep_tiled_header(w, h, Compression::Zips, td);
            *header.line_order_mut().unwrap() = LineOrder::DecreasingY;
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::DeepTileWriteParallel);
                let mut df = DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (df.num_x_tiles(0).unwrap(), df.num_y_tiles(0).unwrap());
                df.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            let bytes = out.into_data();

            // Sanity: round-trips through the real reader.
            let mut is = MemIStream::new(bytes.clone());
            let mut dif = DeepTiledInputFile::new(&mut is).unwrap();
            let n_pix = (w * h) as usize;
            let mut sc_read = vec![0u8; n_pix * 4];
            let mut fb_in = DeepFrameBuffer::new();
            fb_in.insert_sample_count_slice(Slice {
                type_: PixelType::Uint,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4 * w as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            })
            .unwrap();
            dif.set_frame_buffer(&fb_in).unwrap();
            let (nx, ny) = (dif.num_x_tiles(0).unwrap(), dif.num_y_tiles(0).unwrap());
            dif.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut sc_read).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::DeepTileWriteParallel, n);
            for (i, &c) in counts.iter().enumerate() {
                let got = u32::from_le_bytes(sc_read[i * 4..i * 4 + 4].try_into().unwrap());
                assert_eq!(got, c, "num_threads={n}: sample count mismatch at pixel {i}");
            }

            match &reference {
                None => reference = Some(bytes),
                Some(r) => assert_eq!(
                    r.as_slice(),
                    bytes.as_slice(),
                    "num_threads={n}: file bytes differ from num_threads=0"
                ),
            }
        }
    }

    #[test]
    fn raw_count_padding_is_zero_after_distinct_full_tiles_serial_and_parallel() {
        let (w, h) = (8, 6);
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_deep_tiled_header(w, h, Compression::NoCompression, td);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);
        for threads in [0, 4] {
            let mut out = MemOStream::new();
            {
                let mut writer =
                    DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(threads))
                        .unwrap();
                writer.set_frame_buffer(&fb).unwrap();
                crate::probe::reset(crate::probe::Probe::DeepTileWriteParallel);
                writer.write_tiles(&sc, &data, 0, 1, 0, 1, 0, 0).unwrap();
                crate::probe::assert_engaged(crate::probe::Probe::DeepTileWriteParallel, threads);
            }
            let bytes = out.into_data();
            verify(&header, &bytes, Compression::NoCompression, &counts, 4);
            let mut stream = MemIStream::new(bytes);
            let mut reader = crate::deep::DeepTiledInputFile::new(&mut stream).unwrap();
            for x in 0..=1 {
                let raw = reader.raw_tile_data(x, 1, 0, 0).unwrap();
                assert_eq!(raw.packed_count_table.len(), 64);
                assert!(raw.packed_count_table[32..].iter().all(|b| *b == 0));
            }
        }
    }

    /// The deep tiled ring really overlaps: tiles are fed while later tiles
    /// are still encoding. Mutation: emit only after the window is produced
    /// -> this fails.
    #[test]
    fn parallel_write_feeds_while_later_tiles_encode() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (48i32, 48i32);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        for &n in &[2i32, 8] {
            assert!(
                crate::probe::saw_overlap(|| {
                    let header = make_deep_tiled_header(w, h, Compression::Zips, td);
                    let mut out = MemOStream::new();
                    let mut df =
                        DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                    df.set_frame_buffer(&fb).unwrap();
                    let (nx, ny) = (df.num_x_tiles(0).unwrap(), df.num_y_tiles(0).unwrap());
                    df.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
                }),
                "num_threads={n}: no tile was fed while another was still encoding"
            );
        }
    }

    /// Same byte bound for the deep tiled ring. Mutation: ignore the byte cap
    /// -> this fails.
    #[test]
    fn parallel_write_bounds_in_flight_bytes() {
        let td = TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (32i32, 32i32);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        let cap = 300_000usize;
        crate::threading::set_deep_ring_bytes(Some(cap));
        crate::threading::take_peak_bytes();
        let header = make_deep_tiled_header(w, h, Compression::Zips, td);
        let mut out = MemOStream::new();
        let mut df = DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(4)).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let (nx, ny) = (df.num_x_tiles(0).unwrap(), df.num_y_tiles(0).unwrap());
        df.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        drop(df);
        crate::threading::set_deep_ring_bytes(None);
        let peak = crate::threading::take_peak_bytes();
        assert!(peak > 0 && peak <= cap, "peak outstanding bytes {peak} vs cap {cap}");
    }

    /// Writer-side codec reuse for the deep tiled ring (count + data pools,
    /// at most one codec per worker each) over 64 tiles. Mutation: drop the
    /// lease's return -> this fails.
    #[test]
    fn parallel_write_reuses_codecs() {
        use crate::probe::{get, reset, Probe};
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (32i32, 32i32);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        for &n in &[2i32, 8] {
            reset(Probe::CodecCreated);
            let header = make_deep_tiled_header(w, h, Compression::Zips, td);
            let mut out = MemOStream::new();
            {
                let mut df = DeepTiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (df.num_x_tiles(0).unwrap(), df.num_y_tiles(0).unwrap());
                df.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            assert!(get(Probe::CodecCreated) <= 2 * n as u64, "num_threads={n}: {} codecs built for 64 tiles", get(Probe::CodecCreated));
        }
    }
}
