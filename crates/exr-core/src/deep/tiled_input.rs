//! Port of OpenEXR `ImfDeepTiledInputFile.cpp` / `.h` --
//! `class DeepTiledInputFile`, the single-part **deep tiled** reader.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledInputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledInputFile.h>
//!
//! # A note on the port target
//!
//! In OpenEXR v3.3.11 `DeepTiledInputFile` is a thin C++ wrapper over the
//! OpenEXRCore C decode pipeline (`exr_read_tile_chunk_info`, `exr_decoding_run`,
//! `EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL`, ...): the offset-table read, chunk
//! seek, separate count-table / sample-data decompress and the
//! cumulative-to-per-pixel count conversion all live inside the C core.  A
//! literal token-for-token port of the v3.3.11 `.cpp` would have to port the
//! entire C core first (out of scope) -- exactly the situation of
//! [`crate::tiled::TiledInputFile`] and
//! [`crate::deep::DeepScanLineInputFile`].  This module therefore fuses
//! those two precisely-ported readers: it mirrors the **tiled reader**'s
//! container handling (offset-table read + reconstruction, chunk seek, tile /
//! level validation, chunk-coord sanity check) and the **deep scanline
//! reader**'s two-pass count-then-data algorithm, applied over a tile rectangle.
//! Both are the exact inverse of the
//! [`DeepTiledOutputFile`][crate::deep::DeepTiledOutputFile] writer, so this
//! reader round-trips it pixel-exact and count-exact.
//!
//! The v3.3.11 semantics that *are* observable are matched exactly:
//! * the constructor validates magic + version (the deep / non-image flag set),
//!   reads the header, checks `type == deeptile` and that it is a tiled header
//!   (a tile description is present), reads the tile-offset table (with
//!   reconstruction);
//! * `setFrameBuffer` runs the deep `setFrameBuffer` checks (sampling must be
//!   `(1,1)`; the sample-count slice base must be set) and records the slice
//!   table + sample-count slice;
//! * `readPixelSampleCounts(dx1,dx2,dy1,dy2,lx,ly)` decompresses *only* each
//!   tile's count table, converts the cumulative running totals back to
//!   per-pixel counts and writes them into the sample-count slice;
//! * `readTiles(dx1,dx2,dy1,dy2,lx,ly)` skips the count table, decompresses the
//!   sample data and scatters each channel's samples into the caller's
//!   per-pixel arrays via the [`DeepSlice`][crate::deep::DeepSlice] addressing.
//!
//! # The deep tiled chunk byte layout (the inverse of the writer's `writeTileData`)
//!
//! Each chunk is (all multi-byte fields Xdr little-endian):
//!
//! | field                        | type | bytes |
//! |------------------------------|------|-------|
//! | `dx` (tileX)                 | i32  | 4     |
//! | `dy` (tileY)                 | i32  | 4     |
//! | `lx` (levelX)                | i32  | 4     |
//! | `ly` (levelY)                | i32  | 4     |
//! | `packedSampleCountTableSize` | u64  | 8     |
//! | `packedDataSize`             | u64  | 8     |
//! | `unpackedDataSize`           | u64  | 8     |
//! | sample-count table           | bytes| `packedSampleCountTableSize` |
//! | sample data                  | bytes| `packedDataSize`             |
//!
//! The four tile coords are verified against the request (C++ `readTileData`
//! checks this and throws on mismatch -- a corrupt offset table or file).
//!
//! # The cumulative-to-per-pixel count conversion over the tile rectangle
//!
//! The writer stores, per row `i` of the tile rectangle, a **cumulative running
//! total**: `count` resets to 0 at the start of each row and accumulates across
//! the columns `j`, each running sum written as an Xdr `int`.  The last int of a
//! row is that row's total.  To recover the per-pixel counts the reader takes
//! the first difference **per row**:
//!
//! ```text
//! perPixel[row][0] = cumulative[row][0]
//! perPixel[row][j] = cumulative[row][j] - cumulative[row][j-1]   (j > 0)
//! ```
//!
//! This is the tile-rectangle analogue of the deep scanline reader's per-line
//! first difference, and the inverse of the writer's per-row accumulation.
//!
//! # The raw count-table store (the whole-tile-padding gotcha)
//!
//! The writer's count-table compressor is sized for
//! `maxSampleCountTableSize = tileDesc.ySize * tileDesc.xSize * 4` -- the WHOLE
//! tile area.  Its "no-shrink -> store raw" branch stores **exactly
//! `maxSampleCountTableSize`** bytes: the real `tileW*tileH*4` ints followed by
//! zero padding (the construction-time buffer is zero-initialised).  So on read,
//! when `packedSampleCountTableSize == maxSampleCountTableSize` the table is the
//! raw (zero-padded) table and the first `tileW*tileH*4` bytes are the real
//! cumulative ints; otherwise it is compressed and the count decompressor
//! reproduces exactly `tileW*tileH*4` bytes.  (The interior-full-tile raw case
//! where `tileW*tileH*4 == maxSampleCountTableSize` is covered by the same
//! equality.)
//!
//! # Safe-Rust adaptation of the `char**` deep addressing
//!
//! The C++ deep `base` is a `char**` (per-pixel pointers into a sample heap),
//! impossible in safe Rust (zero `unsafe`, no FFI).  This reader uses the same
//! **flat-buffer model** as the deep tiled writer (see its module docs): the caller
//! provides a flat **sample-count buffer** (one `u32` per pixel, addressed by
//! the sample-count slice's byte strides) and a flat per-channel **sample heap**
//! in which each pixel's sample array is laid out contiguously in `(x, y)`
//! raster order over the tile rectangle of the `read_tiles` call (the level
//! for a whole-level request, which is the writer's layout; one tile for a
//! single-tile request).  A tile's pixels are an interior
//! rectangle of that order, so pixel `(x, y)`'s samples begin at
//! `samples_before_pixel(x, y) * sampleStride` (the shared
//! [`crate::deep::sample_table`] helper) -- the exact inverse of the writer's
//! `gather_tile`.  The bytes consumed are byte-identical to the writer's, so the
//! round-trip is exact.

use crate::attr::name::Name;
use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
use crate::attr::PixelType;
use crate::compressor::{new_compressor, Compressor};
use crate::container::xdr;
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::deep::sample_table::{
    check_chunk_sizes, decode_cumulative_row, sample_count_at,
};
use crate::error::{ExrError, Result};
use crate::header::{Header, DEEP_TILE};
use crate::io::IStream;
use crate::multipart::input::{read_prologue, Storage};
use crate::misc::{pixel_type_size, try_filled_vec};
use crate::tiled::misc::{
    data_window_for_level, data_window_for_tile, level_size, precalculate_tile_info,
};
use crate::tiled::tile_offsets::TileOffsets;

use imath_rs::{Box2i, V2i};

/// Xdr `int` size in bytes -- the per-pixel cumulative-count table entry width
/// (the writer writes each running total via `Xdr::write<int>`).
const SIZE_INT: usize = 4;

/// `raw_tile_data`'s return -- a named struct (not a 7-element tuple) both to
/// keep the signature simple (clippy `type_complexity`) and because a long
/// `Result` tuple makes a mutation-testing tool enumerate the PRODUCT of
/// every element's candidate return values (an even longer tuple in
/// `tiled::output::TiledOutputFileState::compute` made `cargo mutants --list`
/// blow up).
#[derive(Debug)]
pub struct RawTileData {
    /// The chunk leader's own coordinates, verified against the request --
    /// echoed back so a caller can confirm the leader it read really
    /// addresses the tile it asked for (mirrors the flat writer's
    /// `TiledInputFileState::raw_tile_data` contract). Not read by any
    /// caller in this crate today (every call site already trusts the
    /// coordinates it passed in), kept for the public contract.
    #[allow(dead_code)]
    pub dx: i32,
    #[allow(dead_code)]
    pub dy: i32,
    #[allow(dead_code)]
    pub lx: i32,
    #[allow(dead_code)]
    pub ly: i32,
    pub packed_count_table: Vec<u8>,
    pub packed_data: Vec<u8>,
    pub unpacked_data_size: u64,
}

// ---------------------------------------------------------------------------
// TInSliceInfo -- read-side per-channel deep tiled descriptor
// ---------------------------------------------------------------------------

/// Per-channel deep tiled read descriptor -- the read-side counterpart of the
/// deep tiled writer's `TOutSliceInfo` and the inverse of the C++ deep tiled decode
/// channel handling.
///
/// Built by [`DeepTiledInputFile::set_frame_buffer`] from the file's channel
/// list and the user [`DeepFrameBuffer`].  Like the deep scanline reader's
/// `InSliceInfo`, the raw `char**` per-pixel pointer base becomes a signed byte
/// offset (`base_offset`) into the caller-owned flat sample heap (see the
/// module docs).
#[derive(Clone, Debug)]
struct TInSliceInfo {
    /// Channel name -- used to match the file channel order on the read scatter.
    name: Name,
    /// Sample type the frame-buffer slice expects (the scatter target type).
    type_: PixelType,
    /// Signed byte offset of this channel's flat sample heap within the
    /// caller's `sample_data` buffer.  Replaces the C++ `char** base`.
    base_offset: i64,
    /// Byte distance between consecutive samples of one pixel -- C++
    /// `DeepSlice::sampleStride`.
    sample_stride: i32,
    /// `slice.xTileCoords` -- when set, the per-pixel sample address uses x
    /// relative to the tile's min x (C++ `xOffsetForData`).
    x_tile_coords: bool,
    /// `slice.yTileCoords` -- when set, the address uses y relative to the
    /// tile's min y.
    y_tile_coords: bool,
    /// `true` when this frame-buffer slice has no matching file channel -- the
    /// C++ *fill list* entry.  On read its per-pixel arrays are filled with the
    /// slice's fill value (no bytes are consumed from the chunk data).
    fill: bool,
    /// Fill value written into a `fill` slice's samples (C++ `slice.fillValue`).
    fill_value: f64,
}

// ---------------------------------------------------------------------------
// DeepTiledInputFile -- port of `class DeepTiledInputFile`
// ---------------------------------------------------------------------------

/// Single-part deep tiled EXR reader -- port of `Imf::DeepTiledInputFile`.
///
/// Construct with [`DeepTiledInputFile::new`] (validates magic + version, reads
/// the header and the tile-offset table), describe the destination with
/// [`DeepTiledInputFile::set_frame_buffer`], then perform the two-pass read:
/// [`read_pixel_sample_counts`][DeepTiledInputFile::read_pixel_sample_counts]
/// (fills counts) followed by
/// [`read_tiles`][DeepTiledInputFile::read_tiles] (fills the samples).
///
/// The reader borrows the caller's [`IStream`] for its whole lifetime.  Its
/// structure mirrors the
/// [`DeepTiledOutputFile`][crate::deep::DeepTiledOutputFile] writer field-for-
/// field so the read path is the exact inverse of the write path.
pub struct DeepTiledInputFile<'a> {
    /// Borrowed input stream (the C++ `_streamData->is`).
    stream: &'a mut dyn IStream,

    /// Everything else this reader owns (mirrors
    /// `crate::tiled::input::TiledInputFileState`): factored into its own,
    /// stream-free struct so [`crate::multipart::MultiPartInputFile`] can
    /// persist it across separate `deep_tiled_input_part` calls instead of
    /// rebuilding a fresh reader every call.
    state: DeepTiledInputFileState,
}

/// Every field of C++ `DeepTiledInputFile::Data` except the stream — the
/// reader's **persisted** per-part state, owned by the multipart file so a
/// part view resumes where the previous one left off.
pub(crate) struct DeepTiledInputFileState {
    /// The image header -- C++ `_data->header`.
    header: Header,

    /// Raw version word read from the file -- C++ `_data->version`.
    version: i32,

    /// Tile layout copied from the header -- C++ `_data->tileDesc`.
    tile_desc: TileDescription,

    /// Data-window bounds -- C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,

    /// Number of levels in X / Y -- C++ `_data->numXLevels` / `numYLevels`.
    num_x_levels: i32,
    num_y_levels: i32,
    /// Per-level tile counts -- C++ `_data->numXTiles` / `numYTiles`.
    num_x_tiles: Vec<i32>,
    num_y_tiles: Vec<i32>,

    /// The tile-offset table -- C++ `_data->tileOffsets`.
    tile_offsets: TileOffsets,
    /// `true` when the on-disk offset table was complete (no reconstruction was
    /// needed) -- C++ `_data->fileIsComplete`.  Exposed via [`is_complete`].
    file_is_complete: bool,

    /// Per-channel read descriptors -- built by
    /// [`set_frame_buffer`][DeepTiledInputFile::set_frame_buffer] (empty until
    /// then).  C++ `_data->frameBuffer` + the fill list.
    slices: Vec<TInSliceInfo>,
    /// `true` once a valid frame buffer has been set -- C++
    /// `_data->frameBufferValid`.
    frame_buffer_valid: bool,
    /// The caller's frame buffer, kept for [`frame_buffer`][Self::frame_buffer]
    /// -- C++ `const DeepFrameBuffer& frameBuffer() const`. Cloned in
    /// `set_frame_buffer`; `DeepFrameBuffer::new()` (empty) until then.
    caller_frame_buffer: DeepFrameBuffer,

    /// Byte offset of the sample-count slice's origin within the caller's count
    /// buffer -- C++'s base pointer.  `0` for the whole-window layout; a local
    /// raster sets it negative to map its low corner to `0`.
    sample_count_base_offset: i64,
    /// Byte strides + tile-coords of the sample-count slice -- C++
    /// `sampleCountSlice`.  Recorded by `set_frame_buffer`; the read scatters
    /// per-pixel counts into the user buffer using these.
    sample_count_x_stride: i32,
    sample_count_y_stride: i32,
    sample_count_x_tile_coords: bool,
    sample_count_y_tile_coords: bool,
    /// A growable per-row sample-count prefix cache (rebuilt by
    /// `begin_range` at the start of every `read_tiles` call, so it is
    /// addressed from that call's tile rectangle), sourced from the caller's
    /// `sample_counts` buffer, so `scatter_tile`/`fill_tile`'s per-pixel
    /// addressing sums each row at most once across the whole read instead
    /// of recomputing it from scratch on every pixel of every channel of
    /// every tile.
    sample_prefix: [crate::deep::sample_table::RowPrefixCache; 4],
    prefix_domains: [Option<(i32, i32, i32)>; 4],
    heap_range: Box2i,

    /// The image compression -- C++ `header.compression()`.
    compression: crate::attr::Compression,
    /// Max size in bytes of one sample-count table -- C++
    /// `maxSampleCountTableSize = tileDesc.ySize * tileDesc.xSize * 4` (the
    /// WHOLE tile area).  Used to recognise the writer's zero-padded raw store.
    max_sample_count_table_size: usize,
    /// The **separate** sample-count-table decompressor (a scanline compressor
    /// sized for `maxSampleCountTableSize`), or `None` for NONE -- the inverse
    /// of the writer's `count_compressor`.  The count table and the sample data
    /// are compressed independently, so they need separate codec instances.
    /// Persistent (built once at construction) -- see `data_compressor`'s doc.
    count_compressor: Option<Box<dyn Compressor>>,
    /// The **separate** sample-DATA decompressor for the fully SERIAL read
    /// path (`read_tile_data`), or `None` for NONE. A single tile's sample
    /// count is unbounded a priori, so no fixed construction-time bound
    /// fits; because `Compressor::uncompress` takes an explicit
    /// `expected_len` per call, THIS field is built once here (any
    /// construction size works) and reused -- a persistent codec, not one
    /// per chunk. The genuinely
    /// PARALLEL decode path (`decode_and_scatter_tile`, run from several
    /// rayon threads at once) checks codecs out of a `CodecPool` instead:
    /// about one per worker, reused across tiles.
    data_compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this reader drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `DeepTiledInputFile::Data::{multiPartBackwardSupport, partNumber}`.
    ///
    /// The same unified seam as the other readers: when `Some`, each deep tile
    /// chunk's leading `partNumber` (i32 LE) is read and verified against this
    /// part; when `None` there is no such prefix.
    part_number: Option<i32>,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the deep scanline reader's
    /// identical field. Gates [`Self::read_tiles`]'s parallel path; never
    /// touches [`Self::read_pixel_sample_counts`], mirroring C++ (only
    /// `readTiles`/`readPixels` is threaded there, never
    /// `readPixelSampleCounts`).
    num_threads: crate::threading::ThreadCount,
}

impl<'a> DeepTiledInputFile<'a> {
    /// Open a `DeepTiledInputFile` on `stream` -- port of the
    /// `DeepTiledInputFile` constructor followed by `Data::initialize`.
    ///
    /// Validates the magic number + version word (deep / non-image flag set, not
    /// multipart), reads the header, checks the part is a deep tiled part
    /// (`type == deeptile`), runs the tiled `sanity_check` (which requires a
    /// tile description), computes the level / tile counts, and reads the
    /// tile-offset table (reconstructing it from the file when the stored table
    /// is zero/corrupt).  Constructs the count-table decompressor.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] for non-deep files (part 0 of a multipart file is read), or a part whose
    ///   `type` is not `deeptile` (matching the C++ "File part is not a deep
    ///   tiled part").
    /// * Propagates [`Header::read_from`] / [`Header::sanity_check`] failures
    ///   (incl. a missing tile description).
    /// * Returns the [`new_compressor`] error for any not-yet-ported codec.
    /// * Propagates stream read / position errors.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::with_num_threads(stream, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(stream: &'a mut dyn IStream, num_threads: Option<i32>) -> Result<Self> {
        // ONE prologue reader for the whole crate: magic + version word + the
        // header + the chunk offset table (with reconstruction) — C++ likewise
        // has a single `internal_exr_parse_header` behind every reader class.
        //
        // A single-part `deeptile` file sets ONLY NON_IMAGE in the version word
        // (the TILED flag is an `else if` in `writeMagicNumberAndVersionField`);
        // it is identified as tiled by `type == deeptile`, which is exactly what
        // the prologue's storage derivation does for a deep file.
        let prologue = read_prologue(stream, true)?;
        // C++ opens part 0 of a multipart file like a single-part one (its chunk leaders then
        // carry the part number, which the reader checks).
        let part_number = prologue.multipart.then_some(0);
        let version_word = prologue.version;
        let part = prologue
            .parts
            .into_iter()
            .next()
            .ok_or_else(|| ExrError::Input("File contains no parts.".into()))?;

        // C++ `Data::initialize`: storage(partNumber) must be DEEP_TILED.
        if part.storage != Storage::DeepTiled {
            return Err(ExrError::Arg(format!(
                "File part is not a deep tiled part (type must be \"{DEEP_TILE}\")."
            )));
        }

        Self::from_part_mt(stream, part.header, version_word, part_number, part.chunk_offsets, num_threads)
    }

    /// Open a `DeepTiledInputFile` for **one part of a multipart file** -- port
    /// of `DeepTiledInputFile::DeepTiledInputFile(InputPartData*)`.
    ///
    /// The enclosing [`MultiPartInputFile`](crate::multipart::MultiPartInputFile)
    /// has already read the file's magic + version word, every part header, and
    /// every part's chunk offset table.  This constructor takes the
    /// already-parsed `header`, the file `version` word, the part's
    /// `chunk_offsets` (flat, in `[level][dy][dx]` slot order), and the part's
    /// `part_number`, and rebuilds the per-part read state without touching the
    /// stream prologue.  Each deep tile chunk's leading `partNumber` is verified
    /// against `part_number`.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepTiledInputFile.cpp> (`DeepTiledInputFile(InputPartData*)`) + `OpenEXRCore/chunk.c`.
    pub fn from_part(
        stream: &'a mut dyn IStream,
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
    ) -> Result<Self> {
        Self::from_part_mt(stream, header, version, part_number, chunk_offsets, None)
    }

    /// Same as [`Self::from_part`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::from_part`]).
    pub fn from_part_mt(
        stream: &'a mut dyn IStream,
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = DeepTiledInputFileState::from_part_mt(
            header,
            version,
            part_number,
            chunk_offsets,
            num_threads,
        )?;
        Ok(Self { stream, state })
    }

    /// The image header -- port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The currently set frame buffer -- port of
    /// `const DeepFrameBuffer& DeepTiledInputFile::frameBuffer() const`.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// The raw file-format version word -- port of `int version() const`.
    pub fn version(&self) -> i32 {
        self.state.version()
    }

    /// Whether the on-disk offset table was complete -- port of
    /// `bool isComplete() const`.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Compression type -- convenience accessor.
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
    }

    /// Tile width -- port of `unsigned int tileXSize() const`.
    pub fn tile_x_size(&self) -> u32 {
        self.state.tile_x_size()
    }

    /// Tile height -- port of `unsigned int tileYSize() const`.
    pub fn tile_y_size(&self) -> u32 {
        self.state.tile_y_size()
    }

    /// Level mode -- port of `LevelMode levelMode() const`.
    pub fn level_mode(&self) -> LevelMode {
        self.state.level_mode()
    }

    /// Level rounding mode -- port of `LevelRoundingMode levelRoundingMode()`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
    }

    /// Number of levels in X -- port of `int numXLevels() const`.
    pub fn num_x_levels(&self) -> i32 {
        self.state.num_x_levels()
    }

    /// Number of levels in Y -- port of `int numYLevels() const`.
    pub fn num_y_levels(&self) -> i32 {
        self.state.num_y_levels()
    }

    /// Number of levels (MIP convenience) -- port of `int numLevels() const`.
    pub fn num_levels(&self) -> Result<i32> {
        self.state.num_levels()
    }

    /// Whether `(lx, ly)` is a valid level -- port of `bool isValidLevel(...)`.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.state.is_valid_level(lx, ly)
    }

    /// Width of level `lx` -- port of `int levelWidth(int lx) const`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.state.level_width(lx)
    }

    /// Height of level `ly` -- port of `int levelHeight(int ly) const`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.state.level_height(ly)
    }

    /// Number of X tiles in level `lx` -- port of `int numXTiles(int lx) const`.
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }

    /// Number of Y tiles in level `ly` -- port of `int numYTiles(int ly) const`.
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.state.num_y_tiles(ly)
    }

    /// Total number of tiles in the image -- port of `size_t totalTiles() const`.
    pub fn total_tiles(&self) -> Result<i64> {
        self.state.total_tiles()
    }

    /// The on-disk tile order -- port of `void getTileOrder(int[], int[], int[], int[])`.
    pub fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        self.state.tile_order()
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

    /// Set the pixel destination. See
    /// [`DeepTiledInputFileState::set_frame_buffer`] for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Read the per-pixel sample counts for the tile rectangle
    /// `[dx1..=dx2] x [dy1..=dy2]` at level `(lx, ly)`. See
    /// [`DeepTiledInputFileState::read_pixel_sample_counts`] for the full doc.
    #[allow(clippy::too_many_arguments)]
    pub fn read_pixel_sample_counts(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_counts(self.stream, dx1, dx2, dy1, dy2, lx, ly, sample_counts)
    }

    /// Read the per-pixel sample counts for the tile rectangle at MIP level `l`.
    #[allow(clippy::too_many_arguments)]
    pub fn read_pixel_sample_counts_l(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_counts_l(self.stream, dx1, dx2, dy1, dy2, l, sample_counts)
    }

    /// Read the per-pixel sample counts for a single tile.
    pub fn read_pixel_sample_count(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_count(self.stream, dx, dy, lx, ly, sample_counts)
    }

    /// Read the per-pixel sample counts for a single tile at MIP level `l`.
    pub fn read_pixel_sample_count_l(
        &mut self,
        dx: i32,
        dy: i32,
        l: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_count_l(self.stream, dx, dy, l, sample_counts)
    }

    /// Read deep samples for the tile rectangle `[dx1..=dx2] x [dy1..=dy2]`
    /// at level `(lx, ly)`.
    ///
    /// Each channel's sample heap starts at its slice's `base_offset`, with
    /// no prefix outside the requested rectangle. Channel tile-coordinate
    /// flags restart the heap within each clipped tile along the selected
    /// axis; absolute axes span the requested rectangle. Count-slice flags
    /// independently select count storage coordinates.
    ///
    /// To populate a larger heap through successive requests, update channel
    /// base offsets and call [`Self::set_frame_buffer`] before each read.
    /// See [`crate::deep::frame_buffer`] for the shared addressing model.
    #[allow(clippy::too_many_arguments)]
    pub fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_tiles(self.stream, dx1, dx2, dy1, dy2, lx, ly, sample_counts, sample_data)
    }

    /// Read the deep sample data for the tile rectangle at MIP level `l`.
    /// The request-local sample-heap contract is the same as [`Self::read_tiles`].
    #[allow(clippy::too_many_arguments)]
    pub fn read_tiles_l(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_tiles_l(self.stream, dx1, dx2, dy1, dy2, l, sample_counts, sample_data)
    }

    /// Read the deep sample data for a single tile.
    /// The request-local sample-heap contract is the same as [`Self::read_tiles`].
    pub fn read_tile(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_tile(self.stream, dx, dy, lx, ly, sample_counts, sample_data)
    }

    /// Read the deep sample data for a single tile at MIP level `l`.
    /// The request-local sample-heap contract is the same as [`Self::read_tiles`].
    pub fn read_tile_l(
        &mut self,
        dx: i32,
        dy: i32,
        l: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_tile_l(self.stream, dx, dy, l, sample_counts, sample_data)
    }

    /// Read tile `(dx, dy, lx, ly)`'s chunk RAW (still-packed count table +
    /// sample data, no decompression). See
    /// [`DeepTiledInputFileState::raw_tile_data`] for the full doc.
    pub fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<RawTileData> {
        self.state.raw_tile_data(self.stream, dx, dy, lx, ly)
    }
}

// ---------------------------------------------------------------------------
// DeepTiledInputFileState — the state-only construction + algorithm, shared
// by the owning `DeepTiledInputFile` and a borrowing part-reader view (the
// multipart file owns this state so part views resume across calls).
// ---------------------------------------------------------------------------

impl DeepTiledInputFileState {
    /// Multipart-part / standalone construction — port of
    /// `DeepTiledInputFile::DeepTiledInputFile(InputPartData*)`. Pure (no
    /// stream access at all): the prologue reader
    /// (`multipart::input::read_prologue`) already parses the header + chunk
    /// offset table before this ever runs — same as the deep scanline
    /// reader's own `from_part_mt` and
    /// `ScanLineInputFileState::from_part_mt`. `num_threads` is the
    /// per-file override (`None` = capture the current process-wide
    /// default).
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // The part must be a deep tiled part (the container validated the
        // `type`; re-check so this path never mis-reads another kind).
        if !header.has_type() || header.type_()? != DEEP_TILE {
            return Err(ExrError::Arg(format!(
                "File part is not a deep tiled part (type must be \"{DEEP_TILE}\")."
            )));
        }

        // `OpenEXRCore/decoding.c:307-318`: a deep part's `version`
        // attribute (defaulted to 1 when absent — `multipart/input.rs`) must
        // be exactly 1; any other value is rejected at decode-init time in
        // C++. This is the per-part deep `version` ATTRIBUTE, distinct from
        // the file's magic/version WORD (the `version` parameter above).
        if header.has_version() && header.version()? != 1 {
            return Err(ExrError::Arg(format!(
                "Version {} not supported for deeptile images.",
                header.version()?
            )));
        }

        // C++: header.sanityCheck(isTiled = true, isMultipart).
        header.sanity_check_at_open(true, part_number.is_some())?;

        let tile_desc = header.tile_description()?;
        let data_window = header.data_window()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
            precalculate_tile_info(&tile_desc, min_x, max_x, min_y, max_y)?;

        // Match C++ 3.4's actual open-time tile-size bound exactly —
        // `OpenEXRCore/validation.c:593-595` rejects a tile description whose
        // `x_size`/`y_size` EACH exceed `INT_MAX/4` (the legacy-writer-only
        // `tileX*tileY*4 > UINT_MAX` product check this used to run instead
        // was never the real gate real C++ 3.4 readers apply).
        const MAX_TILE_DIM: u32 = (i32::MAX / 4) as u32;
        if tile_desc.x_size == 0
            || tile_desc.y_size == 0
            || tile_desc.x_size > MAX_TILE_DIM
            || tile_desc.y_size > MAX_TILE_DIM
        {
            return Err(ExrError::Arg(format!(
                "Invalid tile description size ({} x {})",
                tile_desc.x_size, tile_desc.y_size
            )));
        }
        let max_sample_count_table_size = (tile_desc.y_size as u64)
            * (tile_desc.x_size as u64)
            * (core::mem::size_of::<u32>() as u64);
        let max_sample_count_table_size = max_sample_count_table_size as usize;

        let compression = header.compression()?;
        let count_compressor = new_compressor(compression, max_sample_count_table_size, &header)?;
        // Any construction size works because
        // `expected_len` is authoritative per `uncompress()` call -- see
        // `data_compressor`'s doc.
        let data_compressor = new_compressor(compression, 0, &header)?;

        // Seed the tile-offset table from the offsets the container collected
        // (C++ `tileOffsets.readFrom(chunkOffsets, fileIsComplete)`).
        let mut tile_offsets = TileOffsets::new(
            tile_desc.mode,
            num_x_levels,
            num_y_levels,
            &num_x_tiles,
            &num_y_tiles,
        )?;
        let mut file_is_complete = false;
        tile_offsets.read_from_vec(&chunk_offsets, &mut file_is_complete)?;

        Ok(Self {
            header,
            version,
            tile_desc,
            min_x,
            max_x,
            min_y,
            max_y,
            num_x_levels,
            num_y_levels,
            num_x_tiles,
            num_y_tiles,
            tile_offsets,
            file_is_complete,
            slices: Vec::new(),
            frame_buffer_valid: false,
            caller_frame_buffer: DeepFrameBuffer::new(),
            sample_count_base_offset: 0,
            sample_count_x_stride: 0,
            sample_count_y_stride: 0,
            sample_count_x_tile_coords: false,
            sample_count_y_tile_coords: false,
            sample_prefix: std::array::from_fn(|_| crate::deep::sample_table::RowPrefixCache::new(min_x, max_x, min_y)),
            prefix_domains: [None; 4],
            heap_range: Box2i { min: V2i { x: min_x, y: min_y }, max: V2i { x: max_x, y: max_y } },
            compression,
            max_sample_count_table_size,
            count_compressor,
            data_compressor,
            part_number,
            num_threads,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors -- thin wrappers (mirror TiledInputFile + the deep tiled writer)
    // -----------------------------------------------------------------------

    /// The image header -- port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The currently set frame buffer -- port of
    /// `const DeepFrameBuffer& DeepTiledInputFile::frameBuffer() const`.
    /// Empty until [`set_frame_buffer`][Self::set_frame_buffer] has
    /// been called.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        &self.caller_frame_buffer
    }

    /// The raw file-format version word -- port of `int version() const`.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Whether the on-disk offset table was complete -- port of
    /// `bool isComplete() const`.
    pub fn is_complete(&self) -> bool {
        self.file_is_complete
    }

    /// Compression type -- convenience accessor (C++ reads it off `header()`).
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.header.compression()
    }

    /// Tile width -- port of `unsigned int tileXSize() const`.
    pub fn tile_x_size(&self) -> u32 {
        self.tile_desc.x_size
    }

    /// Tile height -- port of `unsigned int tileYSize() const`.
    pub fn tile_y_size(&self) -> u32 {
        self.tile_desc.y_size
    }

    /// Level mode -- port of `LevelMode levelMode() const`.
    pub fn level_mode(&self) -> LevelMode {
        self.tile_desc.mode
    }

    /// Level rounding mode -- port of `LevelRoundingMode levelRoundingMode()`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.tile_desc.rounding_mode
    }

    /// Number of levels in X -- port of `int numXLevels() const`.
    pub fn num_x_levels(&self) -> i32 {
        self.num_x_levels
    }

    /// Number of levels in Y -- port of `int numYLevels() const`.
    pub fn num_y_levels(&self) -> i32 {
        self.num_y_levels
    }

    /// Number of levels (MIP convenience) -- port of `int numLevels() const`.
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

    /// Whether `(lx, ly)` is a valid level -- port of `bool isValidLevel(...)`.
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

    /// Width of level `lx` -- port of `int levelWidth(int lx) const`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        level_size(self.min_x, self.max_x, lx, self.tile_desc.rounding_mode)
    }

    /// Height of level `ly` -- port of `int levelHeight(int ly) const`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        level_size(self.min_y, self.max_y, ly, self.tile_desc.rounding_mode)
    }

    /// Number of X tiles in level `lx` -- port of `int numXTiles(int lx) const`.
    ///
    /// Returns [`ExrError::Arg`] when `lx` is out of range (`ImfDeepTiledInputFile.cpp:556`
    /// throws `ArgExc`).
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.num_x_tiles
            .get(usize::try_from(lx).map_err(|_| {
                ExrError::Arg("numXTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Arg("numXTiles(): argument is not in valid range".into()))
    }

    /// Number of Y tiles in level `ly` -- port of `int numYTiles(int ly) const`.
    ///
    /// Returns [`ExrError::Arg`] when `ly` is out of range (`ImfDeepTiledInputFile.cpp:572`
    /// throws `ArgExc`).
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.num_y_tiles
            .get(usize::try_from(ly).map_err(|_| {
                ExrError::Arg("numYTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Arg("numYTiles(): argument is not in valid range".into()))
    }

    /// Total number of tiles across every level -- port of
    /// `DeepTiledInputFile::totalTiles() const`. Mirrors the flat
    /// `TiledInputFile`: Σ over every `(lx, ly)` pair of `numXTiles(lx) *
    /// numYTiles(ly)` (`ONE_LEVEL`/`MIPMAP_LEVELS` pair `lx == ly`;
    /// `RIPMAP_LEVELS` is the full cross product).
    pub fn total_tiles(&self) -> Result<i64> {
        let mut total: i64 = 0;
        match self.tile_desc.mode {
            LevelMode::OneLevel | LevelMode::MipmapLevels => {
                for l in 0..self.num_x_levels {
                    total += self.num_x_tiles(l)? as i64 * self.num_y_tiles(l)? as i64;
                }
            }
            LevelMode::RipmapLevels => {
                for ly in 0..self.num_y_levels {
                    for lx in 0..self.num_x_levels {
                        total += self.num_x_tiles(lx)? as i64 * self.num_y_tiles(ly)? as i64;
                    }
                }
            }
            LevelMode::Invalid => {
                // `ImfDeepTiledInputFile.cpp:749` throws `ArgExc`.
                return Err(ExrError::Arg("Invalid level mode.".into()));
            }
        }
        Ok(total)
    }

    /// The `(dx, dy, lx, ly)` of every tile actually present in the file, in
    /// file (on-disk chunk) order -- port of `DeepTiledInputFile::getTileOrder(...)`
    /// Delegates to the same `TileOffsets::get_tile_order` the flat
    /// `TiledInputFile::tile_order` uses (this port always returns the FULL
    /// list sorted by file offset, per that method's own doc note).
    pub fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        let n = self.tile_offsets.slot_count();
        let mut dx = vec![0i32; n];
        let mut dy = vec![0i32; n];
        let mut lx = vec![0i32; n];
        let mut ly = vec![0i32; n];
        self.tile_offsets.get_tile_order(&mut dx, &mut dy, &mut lx, &mut ly)?;
        Ok((dx, dy, lx, ly))
    }

    /// Data window of level `(lx, ly)` -- port of
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

    /// Data window of level `l` (MIP convenience) -- port of
    /// `Box2i dataWindowForLevel(int l) const`.
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.data_window_for_level(l, l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` -- port of
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

    /// Data window of tile `(dx, dy, l)` (MIP convenience) -- port of
    /// `Box2i dataWindowForTile(int dx, int dy, int l) const`.
    pub fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.data_window_for_tile(dx, dy, l, l)
    }

    /// Whether `(dx, dy, lx, ly)` addresses an existing tile -- port of
    /// `bool DeepTiledInputFile::isValidTile(int, int, int, int) const`.
    fn is_valid_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> bool {
        if lx < 0 || lx >= self.num_x_levels || ly < 0 || ly >= self.num_y_levels {
            return false;
        }
        if self.level_mode() == LevelMode::MipmapLevels && lx != ly {
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
    // setFrameBuffer -- ImfDeepTiledInputFile.cpp `setFrameBuffer`
    // -----------------------------------------------------------------------

    /// Set the pixel destination -- port of `DeepTiledInputFile::setFrameBuffer`.
    ///
    /// Performs the deep tiled `setFrameBuffer` checks (every channel must have
    /// sampling `(1,1)` -- subsampling is not supported in tiled files; the
    /// sample-count slice base must be set) and records the per-channel
    /// `TInSliceInfo` table, including the *fill list* (frame-buffer slices
    /// the file does not provide -> filled with their fill value), plus the
    /// sample-count slice geometry.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when the sample-count slice has zero strides (treated
    ///   as an unset base -- C++ "Invalid base pointer, please set a proper
    ///   sample count slice.").
    /// * [`ExrError::Arg`] when any slice has sampling other than `(1,1)` (C++
    ///   "All channels in a tiled file must have sampling (1,1).").
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++: every frame-buffer slice (deep tiled requires sampling (1,1)).
        for (_name, deep_slice) in frame_buffer.iter() {
            if deep_slice.slice.x_sampling != 1 || deep_slice.slice.y_sampling != 1 {
                return Err(ExrError::Arg(
                    "All channels in a tiled file must have sampling (1,1).".into(),
                ));
            }
        }

        // C++: the sample-count slice base must be set.  In the safe-Rust model
        // the count slice is described by its byte strides; a slice with both
        // strides zero is the equivalent of a null base.
        let sc_slice = frame_buffer.get_sample_count_slice();
        if sc_slice.x_stride == 0 && sc_slice.y_stride == 0 {
            return Err(ExrError::Arg(
                "Invalid base pointer, please set a proper sample count slice.".into(),
            ));
        }
        // `ImfDeepTiledInputFile.cpp:1212`: the sample-count slice's
        // sampling needs its own check (the loop above only walks the named
        // deep slices, which does not include the count slice) — "Tiled data
        // should not have subsampling." applies to it too.
        if sc_slice.x_sampling != 1 || sc_slice.y_sampling != 1 {
            return Err(ExrError::Arg(
                "Tiled data should not have subsampling.".into(),
            ));
        }
        self.sample_count_base_offset = sc_slice.base_offset;
        self.sample_count_x_stride = sc_slice.x_stride as i32;
        self.sample_count_y_stride = sc_slice.y_stride as i32;
        self.sample_count_x_tile_coords = sc_slice.x_tile_coords;
        self.sample_count_y_tile_coords = sc_slice.y_tile_coords;

        // C++: walk the frame buffer; a slice naming a file channel is a normal
        // scatter descriptor, one with no file channel joins the fill list.
        let mut slices: Vec<TInSliceInfo> = Vec::new();
        for (name, deep_slice) in frame_buffer.iter() {
            let fill = channels.find_channel(name).is_none();
            slices.push(TInSliceInfo {
                name: name.clone(),
                type_: deep_slice.slice.type_,
                base_offset: deep_slice.slice.base_offset,
                sample_stride: deep_slice.sample_stride,
                x_tile_coords: deep_slice.slice.x_tile_coords,
                y_tile_coords: deep_slice.slice.y_tile_coords,
                fill,
                fill_value: deep_slice.slice.fill_value,
            });
        }

        self.slices = slices;
        self.caller_frame_buffer = frame_buffer.clone();
        self.frame_buffer_valid = true;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // readPixelSampleCounts -- ImfDeepTiledInputFile.cpp lines 652-707
    // -----------------------------------------------------------------------

    /// Read the per-pixel sample counts for the tile rectangle
    /// `[dx1..=dx2] x [dy1..=dy2]` at level `(lx, ly)` -- port of
    /// `DeepTiledInputFile::readPixelSampleCounts(dx1,dx2,dy1,dy2,lx,ly)`.
    ///
    /// For every tile in range this seeks to the chunk via the offset table,
    /// reads `dx,dy,lx,ly` + the three size fields, verifies the coords match
    /// the request (C++ checks this), decompresses *only* the count table,
    /// converts the per-row cumulative running totals back to per-pixel counts
    /// (first difference per row, see the module docs) and writes the per-pixel
    /// counts into the caller's flat `sample_counts` buffer using the
    /// sample-count slice's byte strides and tile-coords offsets.
    ///
    /// `dx1 > dx2` / `dy1 > dy2` are swapped (as in C++).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the tile / level is
    ///   invalid (matching the C++ throws).
    /// * [`ExrError::Input`] on a corrupt chunk header / count table.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_pixel_sample_counts(
        &mut self,
        stream: &mut dyn IStream,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        if !self.frame_buffer_valid {
            return Err(ExrError::Arg(
                "readPixelSampleCounts called with no valid frame buffer".into(),
            ));
        }

        let (dx1, dx2, dy1, dy2) = self.range_check(dx1, dx2, dy1, dy2, lx, ly)?;

        for ty in dy1..=dy2 {
            for tx in dx1..=dx2 {
                let (rect, per_pixel, _pd, _up) = self.read_count_table(stream, tx, ty, lx, ly)?;
                self.scatter_counts(&rect, &per_pixel, sample_counts)?;
            }
        }
        Ok(())
    }

    /// Read the per-pixel sample counts for the tile rectangle at MIP level `l`
    /// -- port of `readPixelSampleCounts(dx1,dx2,dy1,dy2,l)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_pixel_sample_counts_l(
        &mut self,
        stream: &mut dyn IStream,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.read_pixel_sample_counts(stream, dx1, dx2, dy1, dy2, l, l, sample_counts)
    }

    /// Read the per-pixel sample counts for a single tile -- port of
    /// `readPixelSampleCount(dx, dy, lx, ly)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_pixel_sample_count(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.read_pixel_sample_counts(stream, dx, dx, dy, dy, lx, ly, sample_counts)
    }

    /// Read the per-pixel sample counts for a single tile at MIP level `l` --
    /// port of `readPixelSampleCount(dx, dy, l)`.
    pub(crate) fn read_pixel_sample_count_l(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        l: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.read_pixel_sample_count(stream, dx, dy, l, l, sample_counts)
    }

    // -----------------------------------------------------------------------
    // readTiles -- ImfDeepTiledInputFile.cpp `readTiles`
    // -----------------------------------------------------------------------

    /// Read the deep sample data for the tile rectangle `[dx1..=dx2] x
    /// [dy1..=dy2]` at level `(lx, ly)` into the caller's flat per-channel
    /// sample heap -- port of `DeepTiledInputFile::readTiles(...)`.
    ///
    /// Must be called *after*
    /// [`read_pixel_sample_counts`][DeepTiledInputFile::read_pixel_sample_counts]
    /// (the caller uses the counts to size the per-pixel arrays inside
    /// `sample_data`).  For every tile in range this seeks to the chunk, skips
    /// the count table, decompresses the sample data and de-interleaves each
    /// channel's samples into the caller's per-pixel arrays via the
    /// [`DeepSlice`][crate::deep::DeepSlice] addressing
    /// (`base_offset + pixelStart + i*sampleStride`).  Channels absent from the
    /// file (the fill list) are filled with their fill value.  The chunk's own
    /// coords are verified to equal the requested tile (C++ checks this).
    ///
    /// `sample_counts` is the per-pixel count buffer filled by the first pass;
    /// `sample_data` is the flat per-channel sample heap the deep slices'
    /// `base_offset` index into.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the tile / level is
    ///   invalid.
    /// * [`ExrError::Input`] on a corrupt chunk (bad coords, wrong decompressed
    ///   size) or a sample address outside `sample_data`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tiles(
        &mut self,
        stream: &mut dyn IStream,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        if !self.frame_buffer_valid {
            return Err(ExrError::Arg(
                "readTiles called with no valid frame buffer".into(),
            ));
        }

        let (dx1, dx2, dy1, dy2) = self.range_check(dx1, dx2, dy1, dy2, lx, ly)?;
        self.begin_range(dx1, dx2, dy1, dy2, lx, ly)?;

        let mut coords: Vec<(i32, i32)> = Vec::new();
        for ty in dy1..=dy2 {
            for tx in dx1..=dx2 {
                coords.push((tx, ty));
            }
        }

        // Parallel only when there is more than one tile
        // AND the captured thread count is `> 1`, matching the flat tiled
        // reader and mirroring C++'s asymmetry that only `readTiles`,
        // never `readPixelSampleCounts`, is ever threaded.
        if coords.len() > 1
            && self.num_threads.is_parallel()
            && self.read_tiles_parallel(stream, &coords, lx, ly, sample_counts, sample_data)?
        {
            crate::probe::hit(crate::probe::Probe::DeepTileReadParallel);
            return Ok(());
        }
        // A `false`/skipped result means either the fast path wasn't
        // eligible (see `read_tiles_parallel`'s doc — mixed tile-coords
        // flags across slices, or an aliased/overlapping write pattern), or
        // there was only one tile to begin with — fall back to serial.

        for &(tx, ty) in &coords {
            self.read_tile_data(stream, tx, ty, lx, ly, sample_counts, sample_data)?;
        }
        Ok(())
    }

    /// The parallel read body — same three-function split as the deep
    /// scanline reader's `read_pixels_parallel`, generalised to 2D tile
    /// geometry. Returns `Ok(true)` when it fully serviced the request, or
    /// `Ok(false)` when it declined (see below) — the caller then falls back
    /// to the fully serial loop, producing identical bytes either way.
    ///
    /// **Scope restriction (documented, not silent):** `scatter_tile`'s
    /// per-pixel `samples_before` addressing depends on EACH destination
    /// slice's OWN `x_tile_coords`/`y_tile_coords` flags (a per-channel
    /// setting), not just the sample-count slice's. Precomputing one shared
    /// prefix matrix per tile (reused across every channel, which is what
    /// makes the serial pre-pass cheap and correct) requires every slice to
    /// agree on both flags; if the caller's frame buffer mixes them across
    /// channels (a legal but unusual setup — this crate has found no test
    /// or real file that does), this function declines (`Ok(false)`) rather
    /// than risk a subtly wrong per-slice prefix computation, and the caller
    /// falls back to the exact serial path everyone already trusts.
    fn read_tiles_parallel(
        &mut self,
        stream: &mut dyn IStream,
        coords: &[(i32, i32)],
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<bool> {
        let Some((x_tile_coords, y_tile_coords)) =
            super::uniform_tile_coords(self.slices.iter().map(|s| (s.x_tile_coords, s.y_tile_coords)))
        else {
            return Ok(false);
        };

        let mut tile_ranges = Vec::with_capacity(coords.len());
        for &(tx, ty) in coords {
            tile_ranges.push(data_window_for_tile(
                &self.tile_desc,
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                tx,
                ty,
                lx,
                ly,
            )?);
        }

        // Serial pre-pass: one shared per-tile `samples_before_pixel` matrix
        // (reused across every channel — see the scope-restriction doc
        // above), extending the persisted `RowPrefixCache` in the
        // SAME row-major order `coords` was built in (ty outer, tx inner),
        // which keeps its row-level extension monotonic in `y` — preserving
        // the amortized-linear guarantee exactly as the deep scanline
        // reader's identical pre-pass does.
        let mut prefixes: Vec<Vec<u64>> = Vec::with_capacity(coords.len());
        for tr in &tile_ranges {
            let tw = (tr.max.x - tr.min.x + 1) as usize;
            let th = (tr.max.y - tr.min.y + 1) as usize;
            let mut mat = Vec::with_capacity(tw * th);
            for y in tr.min.y..=tr.max.y {
                for x in tr.min.x..=tr.max.x {
                    mat.push(self.sample_start(sample_counts, tr, x_tile_coords, y_tile_coords, x, y)?);
                }
            }
            prefixes.push(mat);
        }

        // Fetch every tile's chunk (leader + count table decode + packed
        // sample-data bytes) on the calling thread, and verify each tile's
        // counts against the caller's buffer while still serial.
        let mut raws = Vec::with_capacity(coords.len());
        for (&(tx, ty), tr) in coords.iter().zip(&tile_ranges) {
            let raw = self.fetch_tile(stream, tx, ty, lx, ly)?;
            self.verify_counts(tr, &raw.per_pixel, sample_counts)?;
            raws.push(raw);
        }

        let mut intervals = Vec::with_capacity(coords.len());
        for (raw, prefix) in raws.iter().zip(&prefixes) {
            let tile_w = (raw.tile_range.max.x - raw.tile_range.min.x + 1) as usize;
            intervals.push(deep_tile_write_interval(&self.slices, &raw.per_pixel, prefix, tile_w)?);
        }
        let dests = match crate::misc::plan_disjoint(sample_data, &intervals) {
            Some(d) => d,
            None => return Ok(false),
        };

        let compression = self.compression;
        let header = &self.header;
        let slices = &self.slices;
        let count = self.num_threads;

        // A real checkout/checkin `CodecPool` instead of `map_init` (which
        // builds a codec per item) -- see `Compressor`'s trait
        // doc and `crate::compressor::CodecPool`'s own doc.
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;
        let results: Vec<Result<()>> = crate::threading::run_on_global_pool(count, || {
            use rayon::prelude::*;
            raws.par_iter()
                .zip(prefixes.par_iter())
                .zip(dests.into_par_iter())
                .map(|((raw, prefix), mut dest)| -> Result<()> {
                    let mut worker_compressor =
                        codec_pool.checkout(|| new_compressor(compression, 0, header))?;
                    let result = decode_and_scatter_tile(
                        worker_compressor.as_mut(),
                        header,
                        slices,
                        raw,
                        prefix,
                        &mut dest,
                    );
                    codec_pool.checkin(worker_compressor);
                    result
                })
                .collect()
        })?;
        // Deterministic error: the lowest tile index that failed, not
        // whichever finished first in time.
        for r in results {
            r?;
        }
        Ok(true)
    }

    /// Fetch tile `(dx, dy, lx, ly)`'s chunk — leader + count
    /// table + packed sample-data bytes — WITHOUT decompressing the sample
    /// data or scattering. Mirrors the deep scanline reader's `fetch_line` —
    /// see its doc for why the count table is decoded here rather than
    /// deferred. `tile_range` is passed in (already computed by the caller)
    /// to avoid recomputing it.
    ///
    /// `&mut self`, using the
    /// persistent `self.count_compressor` — exactly ONE caller
    /// (`read_tiles_parallel`'s serial pre-pass for-loop, itself `&mut
    /// self`), fetching every tile ONE AFTER ANOTHER on the calling thread
    /// before the genuinely parallel decode step (`decode_and_scatter_tile`,
    /// with `CodecPool` codecs) starts — see `fetch_line`'s identical doc for the
    /// full rationale.
    fn fetch_tile(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<RawDeepTile> {
        let (tile_range, per_pixel, packed_data, unpacked_size) = self.read_count_table(stream, dx, dy, lx, ly)?;
        let mut packed_data_bytes = vec![0u8; packed_data];
        stream.read(&mut packed_data_bytes)?;
        Ok(RawDeepTile { tile_range, per_pixel, packed_data: packed_data_bytes, unpacked_size })
    }

    /// Read the deep sample data for the tile rectangle at MIP level `l` -- port
    /// of `readTiles(dx1,dx2,dy1,dy2,l)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tiles_l(
        &mut self,
        stream: &mut dyn IStream,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.read_tiles(stream, dx1, dx2, dy1, dy2, l, l, sample_counts, sample_data)
    }

    /// Read the deep sample data for a single tile -- port of
    /// `readTile(dx, dy, lx, ly)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tile(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.read_tiles(stream, dx, dx, dy, dy, lx, ly, sample_counts, sample_data)
    }

    /// Read the deep sample data for a single tile at MIP level `l` -- port of
    /// `readTile(dx, dy, l)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tile_l(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        l: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.read_tile(stream, dx, dy, l, l, sample_counts, sample_data)
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    /// Swap `dx1 > dx2` / `dy1 > dy2` and validate the rectangle + level -- the
    /// shared head of C++ `readTiles` / `readPixelSampleCounts`.
    fn range_check(
        &self,
        mut dx1: i32,
        mut dx2: i32,
        mut dy1: i32,
        mut dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32)> {
        if dx1 > dx2 {
            std::mem::swap(&mut dx1, &mut dx2);
        }
        if dy1 > dy2 {
            std::mem::swap(&mut dy1, &mut dy2);
        }
        if !self.is_valid_level(lx, ly) {
            return Err(ExrError::Arg(format!(
                "Level coordinate ({lx}, {ly}) is invalid."
            )));
        }
        if !self.is_valid_tile(dx1, dy1, lx, ly) || !self.is_valid_tile(dx2, dy2, lx, ly) {
            return Err(ExrError::Arg("Tile coordinates are invalid.".into()));
        }
        Ok((dx1, dx2, dy1, dy2))
    }

    /// Begin a range-local heap; channel flags independently choose local axes.
    fn begin_range(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        let tile = |dx, dy| {
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
        };
        let (lo, hi) = (tile(dx1, dy1)?, tile(dx2, dy2)?);
        self.heap_range = Box2i {
            min: lo.min,
            max: hi.max,
        };
        self.prefix_domains.fill(None);
        Ok(())
    }

    /// Channel flags select the heap domain; count flags select storage coordinates.
    /// Both domains are expressed through absolute pixels, including negative windows.
    fn sample_start(
        &mut self,
        counts: &[u8],
        tile: &Box2i,
        channel_x_local: bool,
        channel_y_local: bool,
        x: i32,
        y: i32,
    ) -> Result<u64> {
        let index = usize::from(channel_x_local) + 2 * usize::from(channel_y_local);
        let min_x = if channel_x_local {
            tile.min.x
        } else {
            self.heap_range.min.x
        };
        let max_x = if channel_x_local {
            tile.max.x
        } else {
            self.heap_range.max.x
        };
        let min_y = if channel_y_local {
            tile.min.y
        } else {
            self.heap_range.min.y
        };
        let domain = (min_x, max_x, min_y);
        if self.prefix_domains[index] != Some(domain) {
            self.sample_prefix[index].reset(min_x, max_x, min_y);
            self.prefix_domains[index] = Some(domain);
        }
        let (base, xs, ys) = (
            self.sample_count_base_offset,
            self.sample_count_x_stride,
            self.sample_count_y_stride,
        );
        let (count_x_local, count_y_local) = (
            self.sample_count_x_tile_coords,
            self.sample_count_y_tile_coords,
        );
        let (window_x, window_y) = (i64::from(self.min_x), i64::from(self.min_y));
        let (tile_w, tile_h) = (
            i64::from(self.tile_desc.x_size),
            i64::from(self.tile_desc.y_size),
        );
        self.sample_prefix[index].before_pixel_mapped(x, y, |absolute_x, absolute_y| {
            let count_x = if count_x_local {
                (i64::from(absolute_x) - window_x).rem_euclid(tile_w) as i32
            } else {
                absolute_x
            };
            let count_y = if count_y_local {
                (i64::from(absolute_y) - window_y).rem_euclid(tile_h) as i32
            } else {
                absolute_y
            };
            sample_count_at(counts, base, xs, ys, count_x, count_y)
        })
    }

    /// Seek to tile `(dx, dy, lx, ly)`'s chunk and read its header -- returns
    /// `(packedCountSize, packedDataSize, unpackedDataSize)` with the stream
    /// positioned at the start of the (compressed) count table.  Verifies the
    /// chunk's stored tile coords match the request (C++ `readTileData` checks
    /// this) and that the tile was written (non-zero offset).
    fn seek_chunk(
        &self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(u64, u64, u64)> {
        let chunk_offset = self.tile_offsets.get(dx, dy, lx, ly)?;
        if chunk_offset == 0 {
            return Err(ExrError::Input(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is missing from the file."
            )));
        }

        stream.seekg(chunk_offset)?;
        // For a multipart file the deep tile chunk leader begins with the part
        // number (i32 LE), which must match the part being read (C++
        // `extract_chunk_leader`); single-part files have no such prefix.
        // Then: dx,dy,lx,ly (4*i32), packedCountSize,packedDataSize,
        // unpackedDataSize (3*u64).
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(&mut *stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in deep tile chunk: expected {expected}, found {found}."
                )));
            }
        }
        let r_dx = xdr::read_i32(&mut *stream)?;
        let r_dy = xdr::read_i32(&mut *stream)?;
        let r_lx = xdr::read_i32(&mut *stream)?;
        let r_ly = xdr::read_i32(&mut *stream)?;
        if r_dx != dx || r_dy != dy || r_lx != lx || r_ly != ly {
            return Err(ExrError::Input(format!(
                "Unexpected tile coordinates in deep chunk header: expected \
                 ({dx}, {dy}, {lx}, {ly}), found ({r_dx}, {r_dy}, {r_lx}, {r_ly}) \
                 (corrupt offset table or chunk)."
            )));
        }
        let packed_count = xdr::read_u64(&mut *stream)?;
        let packed_data = xdr::read_u64(&mut *stream)?;
        let unpacked = xdr::read_u64(&mut *stream)?;

        // Tiled sibling of the scanline bound in
        // `scanline_input.rs::seek_chunk`: this is the ONE place all three
        // deep tile chunk leader sizes are parsed, feeding BOTH the normal
        // decode path AND the raw `copy_pixels` path (`raw_tile_data`, which
        // allocates straight from these sizes). Ported C++'s own leader-parse
        // bounds (`OpenEXRCore/chunk.c:1360-1432`,
        // `read_and_validate_chunk_leader`'s deep-tiled branch) exactly:
        // unlike the scanline leader, C++ does NOT cap the sample-table size
        // (`ddata[0]`) at `INT32_MAX` here (a smaller-than-nominal edge
        // tile's table can legitimately be smaller than the base tile size,
        // and the comment there notes "just check it's at least a multiple
        // of int32_t" -- no upper bound beyond the file-size check below);
        // `packed_data`/`unpacked` (`ddata[1]`/`ddata[2]`) ARE capped at
        // `INT32_MAX` ("not all compressors support 64-bit").
        for (name, size) in [("packed data", packed_data), ("unpacked data", unpacked)] {
            if size > i32::MAX as u64 {
                return Err(ExrError::Input(format!(
                    "Deep tile ({dx}, {dy}), level ({lx}, {ly}) has an invalid {name} \
                     size {size} (corrupt leader)."
                )));
            }
        }
        if let Some(file_size) = stream.size() {
            let sample_count_data_offset = stream.tellg()?;
            let data_offset = sample_count_data_offset.saturating_add(packed_count);
            if sample_count_data_offset.saturating_add(packed_count) > file_size
                || data_offset.saturating_add(packed_data) > file_size
            {
                return Err(ExrError::Input(format!(
                    "Deep tile ({dx}, {dy}), level ({lx}, {ly}): sample table size \
                     {packed_count} + packed data size {packed_data} results in access past \
                     the end of the file (corrupt leader)."
                )));
            }
        }

        Ok((packed_count, packed_data, unpacked))
    }

    /// Read tile `(dx, dy, lx, ly)`'s chunk RAW -- the packed (still
    /// compressed, or already-raw for NONE/no-shrink) count-table and
    /// sample-data bytes, without decompressing or scattering into a frame
    /// buffer -- port of `void rawTileData(int& dx, int& dy, int& lx,
    /// int& ly, char* pixelData, uint64_t& pixelDataSize)`
    /// (`ImfDeepTiledInputFile.cpp`). Same exact-coordinate contract as the
    /// flat reader's `TiledInputFileState::raw_tile_data`:
    /// `copyPixels` always supplies coordinates it has itself already
    /// computed (from `tile_order()` or `next_tile_coord`), so there is no
    /// "next tile" cursor semantics to port, only the exact-coordinate form.
    ///
    /// Used by [`DeepTiledOutputFileState::copy_pixels`] to transcode a
    /// chunk without ever decompressing it (C++'s own `copyPixels`,
    /// `ImfDeepTiledOutputFile.cpp:1807-1837`, does exactly this: reads the
    /// raw block, slices the 3 size fields out of its own fixed 40-byte
    /// header layout, and re-writes them via `writeTileData` verbatim).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if `(dx, dy, lx, ly)` is outside the image.
    /// * [`ExrError::Input`] if the tile is missing, or its chunk header's
    ///   coordinates don't match the request (corrupt offset table or file).
    pub(crate) fn raw_tile_data(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<RawTileData> {
        if !self.is_valid_tile(dx, dy, lx, ly) {
            return Err(ExrError::Arg(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is outside the image's data window."
            )));
        }
        // `seek_chunk` already bounds `packed_data`/`unpacked` to `i32::MAX`
        // and both `packed_count`/`packed_data` to the remaining bytes of the
        // file; `try_filled_vec` (fallible
        // `try_reserve`) is still used rather than `vec![0u8; n]`, matching
        // every other attacker-controlled-size allocation in this crate.
        let (packed_count, packed_data, unpacked) = self.seek_chunk(stream, dx, dy, lx, ly)?;
        let packed_count_usize = usize::try_from(packed_count)
            .map_err(|_| ExrError::Input(format!("Count table size {packed_count} too large.")))?;
        let packed_data_usize = usize::try_from(packed_data)
            .map_err(|_| ExrError::Input(format!("Packed data size {packed_data} too large.")))?;
        let mut count_bytes = try_filled_vec(packed_count_usize, 0u8)?;
        stream.read(&mut count_bytes)?;
        let mut data_bytes = try_filled_vec(packed_data_usize, 0u8)?;
        stream.read(&mut data_bytes)?;
        Ok(RawTileData {
            dx,
            dy,
            lx,
            ly,
            packed_count_table: count_bytes,
            packed_data: data_bytes,
            unpacked_data_size: unpacked,
        })
    }

    /// Read + decompress tile `(dx, dy, lx, ly)`'s count table and convert the
    /// per-row cumulative running totals to per-pixel counts.  Leaves the stream
    /// positioned at the start of the (compressed) sample data and returns
    /// `(tile_rect, per_pixel_counts, packed_data_size, unpacked_data_size)`
    /// (per-pixel counts in row-major order over the tile rectangle).
    #[allow(clippy::too_many_arguments)]
    fn read_count_table(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(Box2i, Vec<u32>, usize, usize)> {
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
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;
        let table_data_size = tile_w * tile_h * SIZE_INT;

        let (packed_count, packed_data, unpacked) = self.seek_chunk(stream, dx, dy, lx, ly)?;

        // Reject an implausible count-table size BEFORE allocating a buffer
        // from the untrusted on-wire `packed_count`.  The writer stores the count
        // table either compressed (strictly smaller than `maxSampleCountTableSize`)
        // or raw padded to exactly `maxSampleCountTableSize`, so `packed_count` can
        // never exceed that bound for a valid file; a malformed near-`u64::MAX`
        // value is a memory-exhaustion DoS and becomes a clean error here.
        if packed_count > self.max_sample_count_table_size as u64 {
            return Err(ExrError::Input(format!(
                "Deep count table size {packed_count} exceeds the maximum {} the \
                 writer could store (corrupt chunk).",
                self.max_sample_count_table_size
            )));
        }

        let packed_count_usize = usize::try_from(packed_count).map_err(|_| {
            ExrError::Input(format!("Count table size {packed_count} too large."))
        })?;
        let mut packed = vec![0u8; packed_count_usize];
        stream.read(&mut packed)?;

        // Decompress the count table with its OWN decompressor, mirroring the
        // deep tiled writer's TWO raw-store mechanisms (same as the deep scanline
        // reader's `read_count_table`):
        //   * the writer's OUTER "no-shrink (or no compressor)" branch stores the
        //     raw table padded with zeroes out to EXACTLY `maxSampleCountTableSize`
        //     (the real `table_data_size` cumulative ints + zero padding) ->
        //     `packed == max_sample_count_table_size`.  (This also covers the
        //     interior-full-tile case where table_data_size == max.)
        //   * the ZIP/RLE codec's OWN internal "no-shrink" store returns the raw
        //     `table_data_size` predicted bytes, which the writer keeps as-is
        //     because that is `< max` -> `packed == table_data_size`.  Without
        //     this case a raw-stored ZIPS/RLE partial-tile count table would be
        //     mis-routed to `uncompress` (zlib `BadHeader`).
        // In BOTH raw cases the stored bytes are the uncompressed cumulative
        // table (its first `table_data_size` bytes); only a genuinely smaller
        // `packed` is a real compressed stream.
        // C++ `OpenEXRCore/decoding.c:625-662`, `chunk.c:1379-1382`: a
        // NONE-compressed count table only needs to be a multiple of 4 bytes
        // (it need not cover the full tile rectangle) — a short table is the
        // raw cumulative ints for the leading rows/pixels, zero-padded for
        // the rest (a fully absent, zero-length table included).
        let cumulative_bytes: Vec<u8> = if packed_count_usize == self.max_sample_count_table_size
            || packed_count_usize == table_data_size
        {
            packed.truncate(table_data_size);
            packed
        } else if self.count_compressor.is_none() {
            if packed_count_usize % SIZE_INT != 0 || packed_count_usize > table_data_size {
                return Err(ExrError::Input(format!(
                    "Deep count table size {packed_count_usize} is not a valid \
                     uncompressed table for a {table_data_size}-byte tile (corrupt chunk)."
                )));
            }
            let mut buf = packed;
            buf.resize(table_data_size, 0);
            buf
        } else if packed_count_usize == 0 {
            // An absent (zero-length) compressed count table is an
            // all-zero row, not an error.
            vec![0u8; table_data_size]
        } else if let Some(c) = self.count_compressor.as_mut() {
            let decoded = c
                .uncompress(&packed, tile_range.min.y, table_data_size)
                .map_err(crate::error::decode_err)?;
            if decoded.len() != table_data_size {
                return Err(ExrError::Input(format!(
                    "Decompressed deep count table size {} does not match the \
                     expected size {table_data_size} (corrupt chunk).",
                    decoded.len()
                )));
            }
            decoded.to_vec()
        } else {
            return Err(ExrError::Input(format!(
                "Deep count table size {packed_count_usize} differs from both the \
                 raw size {table_data_size} and maxSampleCountTableSize {} but \
                 the file has no compressor.",
                self.max_sample_count_table_size
            )));
        };

        // Convert per-row cumulative running totals -> per-pixel counts (first
        // difference, reset per row of the tile rectangle -- the inverse of the
        // writer's per-row accumulation).
        // Per-row first difference of the cumulative table (reset to 0 per tile
        // row) via the shared checked decoder — inverse of the writer.
        let mut per_pixel = vec![0u32; tile_w * tile_h];
        for row in 0..tile_h {
            decode_cumulative_row(&cumulative_bytes, row * tile_w, tile_w, &mut per_pixel)?;
        }

        let (packed_data, unpacked) = check_chunk_sizes(
            &per_pixel, self.header.channels()?, self.compression, packed_data, unpacked, stream,
        )?;
        Ok((tile_range, per_pixel, packed_data, unpacked))
    }

    /// Write per-pixel counts into the caller's flat sample-count buffer -- port
    /// of the C++ deep tiled count scatter, addressing
    /// `sampleCount(base, xStride, yStride, x - xOffSC, y - yOffSC)` over the
    /// tile rectangle.
    fn scatter_counts(
        &self,
        tile_range: &Box2i,
        per_pixel: &[u32],
        sample_counts: &mut [u8],
    ) -> Result<()> {
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

        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        for (ry, y) in (tile_range.min.y..=tile_range.max.y).enumerate() {
            for (rx, x) in (tile_range.min.x..=tile_range.max.x).enumerate() {
                let count = *per_pixel.get(ry * tile_w + rx).ok_or_else(|| {
                    ExrError::Input("Per-pixel count index out of range.".into())
                })?;
                let off = self.sample_count_base_offset
                    + (y - y_off_sc) as i64 * self.sample_count_y_stride as i64
                    + (x - x_off_sc) as i64 * self.sample_count_x_stride as i64;
                if off < 0 {
                    return Err(ExrError::Arg(
                        "Sample-count slice address is out of range (negative).".into(),
                    ));
                }
                let off = off as usize;
                let dst = sample_counts.get_mut(off..off + SIZE_INT).ok_or_else(|| {
                    ExrError::Arg(format!(
                        "Sample-count slice address {off} out of range of the count buffer."
                    ))
                })?;
                dst.copy_from_slice(&count.to_le_bytes());
            }
        }
        Ok(())
    }

    /// Read one tile's sample data and scatter it into the caller's flat
    /// per-channel sample heap -- the inverse of the writer's `gather_tile`
    /// and the port of the deep tiled `readTileData` decode scatter.
    #[allow(clippy::too_many_arguments)]
    fn read_tile_data(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        // Read the count table for this tile (drives the per-pixel sizes and
        // positions the stream at the start of the sample data).  It returns the
        // chunk's data sizes too, so no second seek is needed.
        let (tile_range, per_pixel, packed_data_usize, unpacked_usize) =
            self.read_count_table(stream, dx, dy, lx, ly)?;

        // Read the (possibly compressed) sample data bytes.
        let mut packed = vec![0u8; packed_data_usize];
        stream.read(&mut packed)?;

        // Decompress with the SEPARATE, persistent data decompressor
        // (`self.data_compressor`, reused across tiles --
        // see its field doc), same raw-store rule: if the stored bytes
        // already equal the unpacked size they are raw. Deep is NONE/RLE/
        // ZIPS -- all rectangle-insensitive -- so the scanline `uncompress`
        // at `tile_range.min.y` reproduces the bytes the writer's
        // `compress_tile` produced.
        let data: Vec<u8> = if packed_data_usize == unpacked_usize {
            packed
        } else if let Some(c) = self.data_compressor.as_mut() {
            let decoded = c
                .uncompress(&packed, tile_range.min.y, unpacked_usize)
                .map_err(crate::error::decode_err)?;
            if decoded.len() != unpacked_usize {
                return Err(ExrError::Input(format!(
                    "Decompressed deep sample data size {} does not match the \
                     expected size {unpacked_usize} (corrupt chunk).",
                    decoded.len()
                )));
            }
            decoded.to_vec()
        } else {
            return Err(ExrError::Input(format!(
                "Deep sample data size {packed_data_usize} differs from the \
                 unpacked size {unpacked_usize} but the file has no compressor."
            )));
        };

        // Verify the caller's stored per-pixel counts match the chunk's counts
        // (the scatter below relies on `per_pixel`; the writer's `gather_tile`
        // addressed the flat heap via the same sample-count buffer, so it must
        // agree).  A mismatch means the caller did not run readPixelSampleCounts
        // first / sized the heap wrong -- surface it rather than corrupting.
        self.verify_counts(&tile_range, &per_pixel, sample_counts)?;

        self.scatter_tile(&tile_range, &per_pixel, &data, sample_counts, sample_data)
    }

    /// De-interleave one tile's decompressed sample data into the caller's flat
    /// per-channel heap -- the inverse of the writer's `gather_tile`.
    ///
    /// Walks the tile rectangle row-major (outer `y`, then each file channel,
    /// then each pixel `x`), exactly as the writer gathered it.  File channels
    /// present in the frame buffer scatter; file channels absent from it are
    /// skipped (their bytes are still consumed); fill-list channels (in the
    /// frame buffer but not the file) are filled with their fill value.
    ///
    /// The per-pixel heap start is the pixel's position in the raster of the
    /// current `read_tiles` call's tile rectangle (`begin_range`; the whole
    /// level for a whole-level request, which is the writer's addressing),
    /// against the caller's flat `sample_counts` buffer, with the
    /// `xTileCoords`/`yTileCoords` offsets applied as the writer applied them.
    fn scatter_tile(
        &mut self,
        tile_range: &Box2i,
        per_pixel: &[u32],
        data: &[u8],
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        // Clone the (small) channel list + slice table so the loop body can
        // call `self.sample_prefix.before_pixel` (`&mut self`, for the
        // prefix cache) without holding a borrow of `self.header`/`self.slices`
        // alive across it.
        let channels = self.header.channels()?.clone();
        let slices = self.slices.clone();
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;

        let mut read_offset = 0usize;
        for (row, y) in (tile_range.min.y..=tile_range.max.y).enumerate() {
            let row_base = row * tile_w;
            for (name, ch) in channels.iter() {
                let sample_size = pixel_type_size(ch.type_)?;
                // The destination slice for this file channel (skip fill-list
                // slices -- those are handled after the file channels).
                let slice = slices.iter().find(|s| &s.name == name && !s.fill);

                match slice {
                    None => {
                        // File channel with no destination slice: consume this
                        // row's bytes (its total sample count) without scatter.
                        let mut row_samples: u64 = 0;
                        for col in 0..tile_w {
                            row_samples += *per_pixel.get(row_base + col).ok_or_else(|| {
                                ExrError::Input("Per-pixel count index out of range.".into())
                            })? as u64;
                        }
                        read_offset += row_samples as usize * sample_size;
                    }
                    Some(slice) => {
                        // Channel flags choose the heap axes independently of count storage.
                        for (col, x) in (tile_range.min.x..=tile_range.max.x).enumerate() {
                            let cnt = *per_pixel.get(row_base + col).ok_or_else(|| {
                                ExrError::Input("Per-pixel count index out of range.".into())
                            })? as i64;
                            // Pixel start in the flat per-channel heap (raster
                            // order over the whole data window).  O(1)
                            // amortized via the growable `RowPrefixCache`
                            // instead of re-summing from `(min_x, min_y)` on
                            // every single pixel of every channel of every tile.
                            let pixel_start = self.sample_start(sample_counts, tile_range, slice.x_tile_coords, slice.y_tile_coords, x, y)? as i64
                                * slice.sample_stride as i64;

                            if slice.fill {
                                // (Cannot occur: `slice` here is a non-fill file
                                // channel.)  Kept for total fidelity with the
                                // writer's gather branch structure.
                                continue;
                            }

                            // C++ `curchan.user_data_type = fbslice->type` +
                            // `generic_unpack_deep_pointers`
                            // (`OpenEXRCore/unpack.c`): convert file type ->
                            // frame-buffer type per sample, never a raw copy.
                            let dst_size = pixel_type_size(slice.type_)?;
                            for i in 0..cnt {
                                let src_slice =
                                    data.get(read_offset..read_offset + sample_size).ok_or_else(
                                        || {
                                            ExrError::Input(
                                                "Deep sample data underrun during scatter.".into(),
                                            )
                                        },
                                    )?;
                                let dst_off = slice.base_offset
                                    + pixel_start
                                    + i * slice.sample_stride as i64;
                                if dst_off < 0 {
                                    return Err(ExrError::Arg(
                                        "Deep sample address is out of range (negative).".into(),
                                    ));
                                }
                                let dst_off = dst_off as usize;
                                let dst = sample_data
                                    .get_mut(dst_off..dst_off + dst_size)
                                    .ok_or_else(|| {
                                        ExrError::Arg(format!(
                                            "Deep sample address {dst_off} out of range of the \
                                             sample data buffer."
                                        ))
                                    })?;
                                let mut conv = [0u8; 4];
                                crate::misc::copy_into_frame_buffer(
                                    src_slice,
                                    0,
                                    &mut conv,
                                    0,
                                    1,
                                    dst_size,
                                    false,
                                    0.0,
                                    crate::misc::BufferFormat::Xdr,
                                    slice.type_,
                                    ch.type_,
                                )?;
                                dst.copy_from_slice(&conv[..dst_size]);
                                read_offset += sample_size;
                            }
                        }
                    }
                }
            }
        }

        // Fill list: frame-buffer slices with no file channel.  Their per-pixel
        // arrays are filled with the slice's fill value (C++ deep tiled fill),
        // addressed the same way (absolute raster start over the whole window).
        // Iterate the local clone (`slices`), not `self.slices`, so calling
        // `self.fill_tile` (`&mut self`) doesn't conflict with the loop's borrow.
        for slice in slices.iter().filter(|s| s.fill) {
            self.fill_tile(tile_range, per_pixel, sample_counts, slice, sample_data)?;
        }

        Ok(())
    }

    /// Fill one fill-list channel's per-pixel arrays over the tile rectangle
    /// with its fill value -- the deep tiled analogue of the deep scanline
    /// `fill_line`, in the flat-heap model.
    fn fill_tile(
        &mut self,
        tile_range: &Box2i,
        per_pixel: &[u32],
        sample_counts: &[u8],
        slice: &TInSliceInfo,
        sample_data: &mut [u8],
    ) -> Result<()> {
        let sample_size = pixel_type_size(slice.type_)?;
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        for (row, y) in (tile_range.min.y..=tile_range.max.y).enumerate() {
            let row_base = row * tile_w;
            for (col, x) in (tile_range.min.x..=tile_range.max.x).enumerate() {
                let cnt = *per_pixel.get(row_base + col).ok_or_else(|| {
                    ExrError::Input("Per-pixel count index out of range.".into())
                })? as i64;
                // O(1) amortized via the growable `RowPrefixCache`.
                let pixel_start = self.sample_start(sample_counts, tile_range, slice.x_tile_coords, slice.y_tile_coords, x, y)? as i64
                    * slice.sample_stride as i64;
                for i in 0..cnt {
                    let dst_off = slice.base_offset + pixel_start + i * slice.sample_stride as i64;
                    if dst_off < 0 {
                        return Err(ExrError::Arg(
                            "Deep fill address is out of range (negative).".into(),
                        ));
                    }
                    let dst_off = dst_off as usize;
                    let dst = sample_data
                        .get_mut(dst_off..dst_off + sample_size)
                        .ok_or_else(|| {
                            ExrError::Arg(format!(
                                "Deep fill address {dst_off} out of range of the sample data buffer."
                            ))
                        })?;
                    write_fill_value(dst, slice.type_, slice.fill_value)?;
                }
            }
        }
        Ok(())
    }

    /// Verify the caller's stored per-pixel counts match the chunk's counts --
    /// the safe-Rust guard that replaces the C++ assumption that the user
    /// allocated arrays sized exactly by the prior `readPixelSampleCounts`.
    fn verify_counts(
        &self,
        tile_range: &Box2i,
        per_pixel: &[u32],
        sample_counts: &[u8],
    ) -> Result<()> {
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
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        for (ry, y) in (tile_range.min.y..=tile_range.max.y).enumerate() {
            for (rx, x) in (tile_range.min.x..=tile_range.max.x).enumerate() {
                let off = self.sample_count_base_offset
                    + (y - y_off_sc) as i64 * self.sample_count_y_stride as i64
                    + (x - x_off_sc) as i64 * self.sample_count_x_stride as i64;
                // A negative/out-of-range count address is a hard error, not
                // silently skipped, matching the hard-error policy of
                // `sample_count_at` — this is
                // a Rust-only guard with no C++ equivalent, so it must be at
                // least as strict as the rest of the crate, not more lenient.
                if off < 0 {
                    return Err(ExrError::Arg(format!(
                        "Sample-count buffer address for ({x},{y}) is out of range (negative)."
                    )));
                }
                let off = off as usize;
                let b = sample_counts.get(off..off + SIZE_INT).ok_or_else(|| {
                    ExrError::Arg(format!(
                        "Sample-count buffer address {off} for ({x},{y}) is out of range of \
                         the count buffer (len {}).",
                        sample_counts.len()
                    ))
                })?;
                let stored = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let expected = *per_pixel.get(ry * tile_w + rx).unwrap_or(&0);
                if stored != expected {
                    return Err(ExrError::Arg(format!(
                        "Sample-count mismatch at ({x},{y}): the buffer says {stored} \
                         but the chunk holds {expected}; call readPixelSampleCounts first."
                    )));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure per-tile pipeline functions (deep tiled reader, parallel path)
// ---------------------------------------------------------------------------

/// One tile's fetched-but-not-decoded deep chunk: the count table already
/// decoded, the still-possibly-compressed sample DATA bytes, and the
/// chunk's declared unpacked size. Built by
/// [`DeepTiledInputFileState::fetch_tile`] on the calling thread; consumed
/// by [`decode_and_scatter_tile`] on any thread.
struct RawDeepTile {
    tile_range: Box2i,
    per_pixel: Vec<u32>,
    packed_data: Vec<u8>,
    unpacked_size: usize,
}

/// `abs_addr - base`, erroring instead of underflowing when a caller passes
/// an address before its own piece's base — see the flat readers' identical
/// helper for the full rationale.
fn local_deep_tile_addr(abs_addr: usize, base: usize) -> Result<usize> {
    abs_addr.checked_sub(base).ok_or_else(|| {
        ExrError::Base(format!(
            "internal error: scatter address {abs_addr} precedes its destination \
             piece's base {base}"
        ))
    })
}

/// Decode one [`RawDeepTile`]'s sample data and scatter it into `dests` --
/// ONE [`crate::misc::ChunkDest`] PER SLICE, aligned 1:1 (same length, same
/// order) with `slices`. Same logic [`DeepTiledInputFileState::scatter_tile`]
/// runs, using the PRECOMPUTED `prefix` matrix (`prefix[row*tile_w + col]`
/// = that pixel's `samples_before_pixel`, shared across every channel — see
/// `read_tiles_parallel`'s scope-restriction doc for why this is valid only
/// when every slice agrees on its tile-coords flags) instead of calling the
/// stateful `RowPrefixCache` (unsafe to share across threads).
///
/// Not a single combined `dest`/`dest_base` -- see
/// `decode_and_scatter_line`'s identical doc for why that's wrong for 2+
/// active channels.
///
/// Takes the compressor BY REFERENCE
/// (see `decode_and_scatter_line`'s identical doc) — each worker checks one
/// out of the call site's `CodecPool`, not one per tile.
fn decode_and_scatter_tile(
    compressor: Option<&mut Box<dyn Compressor>>,
    header: &Header,
    slices: &[TInSliceInfo],
    raw: &RawDeepTile,
    prefix: &[u64],
    dests: &mut [crate::misc::ChunkDest],
) -> Result<()> {
    let tile_range = &raw.tile_range;
    let packed_data_usize = raw.packed_data.len();
    let unpacked_usize = raw.unpacked_size;

    let data: Vec<u8> = if packed_data_usize == unpacked_usize {
        raw.packed_data.clone()
    } else if let Some(c) = compressor {
        let decoded = c
            .uncompress(&raw.packed_data, tile_range.min.y, unpacked_usize)
            .map_err(crate::error::decode_err)?;
        if decoded.len() != unpacked_usize {
            return Err(ExrError::Input(format!(
                "Decompressed deep sample data size {} does not match the \
                 expected size {unpacked_usize} (corrupt chunk).",
                decoded.len()
            )));
        }
        decoded.to_vec()
    } else {
        return Err(ExrError::Input(format!(
            "Deep sample data size {packed_data_usize} differs from the \
             unpacked size {unpacked_usize} but the file has no compressor."
        )));
    };

    let channels = header.channels()?;
    let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
    let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;
    let mut read_offset = 0usize;
    for row in 0..tile_h {
        let row_base = row * tile_w;
        for (name, ch) in channels.iter() {
            let sample_size = pixel_type_size(ch.type_)?;
            let slice_idx = slices.iter().position(|s| &s.name == name && !s.fill);

            match slice_idx {
                None => {
                    let mut row_samples: u64 = 0;
                    for col in 0..tile_w {
                        row_samples += *raw.per_pixel.get(row_base + col).ok_or_else(|| {
                            ExrError::Input("Per-pixel count index out of range.".into())
                        })? as u64;
                    }
                    read_offset += row_samples as usize * sample_size;
                }
                Some(slice_idx) => {
                    let slice = &slices[slice_idx];
                    let dst_size = pixel_type_size(slice.type_)?;
                    for (col, _x) in (tile_range.min.x..=tile_range.max.x).enumerate() {
                        let idx = row_base + col;
                        let cnt = *raw.per_pixel.get(idx).ok_or_else(|| {
                            ExrError::Input("Per-pixel count index out of range.".into())
                        })? as i64;
                        let pixel_start = *prefix.get(idx).ok_or_else(|| {
                            ExrError::Input("Per-pixel prefix index out of range.".into())
                        })? as i64
                            * slice.sample_stride as i64;
                        for i in 0..cnt {
                            let src_slice =
                                data.get(read_offset..read_offset + sample_size).ok_or_else(|| {
                                    ExrError::Input(
                                        "Deep sample data underrun during scatter.".into(),
                                    )
                                })?;
                            let dst_off_abs =
                                slice.base_offset + pixel_start + i * slice.sample_stride as i64;
                            if dst_off_abs < 0 {
                                return Err(ExrError::Arg(
                                    "Deep sample address is out of range (negative).".into(),
                                ));
                            }
                            // One destination per
                            // (slice, row), matching `deep_tile_write_interval`'s
                            // flat (slice-major, row-minor) emission order.
                            let dest = dests.get_mut(slice_idx * tile_h + row).ok_or_else(|| {
                                ExrError::Base("internal error: no destination for this slice/row".into())
                            })?;
                            let dst_off = local_deep_tile_addr(dst_off_abs as usize, dest.base)?;
                            let dst = dest.piece.get_mut(dst_off..dst_off + dst_size).ok_or_else(|| {
                                ExrError::Arg(format!(
                                    "Deep sample address {dst_off} out of range of the \
                                     sample data buffer."
                                ))
                            })?;
                            let mut conv = [0u8; 4];
                            crate::misc::copy_into_frame_buffer(
                                src_slice,
                                0,
                                &mut conv,
                                0,
                                1,
                                dst_size,
                                false,
                                0.0,
                                crate::misc::BufferFormat::Xdr,
                                slice.type_,
                                ch.type_,
                            )?;
                            dst.copy_from_slice(&conv[..dst_size]);
                            read_offset += sample_size;
                        }
                    }
                }
            }
        }
    }

    for (idx, slice) in slices.iter().enumerate() {
        if !slice.fill {
            continue;
        }
        let start = idx * tile_h;
        let row_dests = dests.get_mut(start..start + tile_h).ok_or_else(|| {
            ExrError::Base("internal error: no destinations for this slice's rows".into())
        })?;
        fill_tile_into(tile_range, &raw.per_pixel, prefix, slice, row_dests)?;
    }

    Ok(())
}

/// Fill one fill-list channel's per-pixel arrays over the tile rectangle
/// with its fill value, using a PRECOMPUTED `prefix` matrix — free-function,
/// `dest_base`-generalised form of
/// [`DeepTiledInputFileState::fill_tile`].
///
/// `dests` is ONE destination PER ROW
/// of the tile for THIS slice (`dests[row]`, `tile_h` entries) instead of a
/// single combined destination -- see `deep_tile_write_interval`'s doc,
/// bug #3, for why a multi-row tile is not one contiguous byte range even
/// for a single channel.
fn fill_tile_into(
    tile_range: &Box2i,
    per_pixel: &[u32],
    prefix: &[u64],
    slice: &TInSliceInfo,
    dests: &mut [crate::misc::ChunkDest],
) -> Result<()> {
    let sample_size = pixel_type_size(slice.type_)?;
    let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
    for row in 0..(tile_range.max.y - tile_range.min.y + 1) as usize {
        let row_base = row * tile_w;
        let dest = dests.get_mut(row).ok_or_else(|| {
            ExrError::Base("internal error: no destination for this row".into())
        })?;
        for col in 0..tile_w {
            let idx = row_base + col;
            let cnt = *per_pixel.get(idx).ok_or_else(|| {
                ExrError::Input("Per-pixel count index out of range.".into())
            })? as i64;
            let pixel_start = *prefix.get(idx).ok_or_else(|| {
                ExrError::Input("Per-pixel prefix index out of range.".into())
            })? as i64
                * slice.sample_stride as i64;
            for i in 0..cnt {
                let dst_off_abs = slice.base_offset + pixel_start + i * slice.sample_stride as i64;
                if dst_off_abs < 0 {
                    return Err(ExrError::Arg(
                        "Deep fill address is out of range (negative).".into(),
                    ));
                }
                let dst_off = local_deep_tile_addr(dst_off_abs as usize, dest.base)?;
                let dst = dest.piece.get_mut(dst_off..dst_off + sample_size).ok_or_else(|| {
                    ExrError::Arg(format!(
                        "Deep fill address {dst_off} out of range of the sample data buffer."
                    ))
                })?;
                write_fill_value(dst, slice.type_, slice.fill_value)?;
            }
        }
    }
    Ok(())
}

/// Compute the EXACT byte range a tile's [`decode_and_scatter_tile`] call
/// would write for EACH (slice, row) pair into the sample-data heap,
/// WITHOUT writing anything — the input to [`crate::misc::plan_disjoint`].
/// Returns `slices.len() * tile_h` ranges, in (slice-major, row-minor)
/// order -- i.e. slice 0's `tile_h` rows, then slice 1's `tile_h` rows,
/// etc. (a row with no non-zero-count pixels in it gets an empty `0..0`
/// placeholder). [`decode_and_scatter_tile`] indexes this same
/// `[slice_idx * tile_h + row]` scheme to find its destination.
///
/// THREE pitfalls this avoids, the first
/// two documented in `deep::scanline_input`'s `deep_line_write_interval`
/// (this is its tiled counterpart, same underlying issues):
/// 1. Despite this doc's own prior "exact, not an over-approximation"
///    claim, the per-pixel upper bound was `start + cnt*stride` (one full
///    stride PAST the last real sample) plus `dst_size` -- the true last
///    sample is at index `cnt - 1` (guaranteed `>= 0`, since `cnt == 0`
///    pixels are skipped), so its own start address (`far`) replaces the
///    one-past-the-end `end`.
/// 2. `lo`/`hi` used to accumulate across EVERY slice into one merged
///    range -- wrong for 2+ active channels: channels are stored in
///    separate, non-adjacent heap regions, so a single merged range spans
///    through every OTHER tile's data sitting between two channels.
/// 3. (TILE-SPECIFIC, found while chasing why even a SINGLE-channel tiled
///    parallel read still fell back to serial after fixing 1-2) A tile's
///    OWN pixels are NOT one contiguous byte range even for one channel,
///    unless the tile happens to be exactly one row tall: `before_pixel`
///    addresses samples in WHOLE-IMAGE raster order, so a multi-row tile's
///    row 0 and row 1 are separated in address space by every OTHER
///    tile's row-0 data in between (the tiles to its right, on the same
///    image row). Merging across rows into one range per slice (as bug #2's
///    fix alone still did) reproduces the exact same "spans through
///    unrelated data" failure bug #2 fixed for channels, just for ROWS
///    instead. Fixed by resetting `lo`/`hi` per ROW too, yielding one range
///    per (slice, row) pair -- each row of a tile really IS one contiguous
///    span (`before_pixel` is monotonic within a row), so this is finally
///    tight in every dimension.
fn deep_tile_write_interval(
    slices: &[TInSliceInfo],
    per_pixel: &[u32],
    prefix: &[u64],
    tile_w: usize,
) -> Result<crate::misc::RangeSet> {
    let tile_h = per_pixel.len().checked_div(tile_w).unwrap_or(0);
    let mut out = crate::misc::RangeSet::with_capacity(slices.len() * tile_h);
    for slice in slices {
        let dst_size = pixel_type_size(slice.type_)? as i64;
        for row in 0..tile_h {
            let row_base = row * tile_w;
            let mut lo: Option<i64> = None;
            let mut hi: i64 = 0;
            for col in 0..tile_w {
                let idx = row_base + col;
                let cnt = *per_pixel.get(idx).ok_or_else(|| {
                    ExrError::Input("Per-pixel count index out of range.".into())
                })?;
                if cnt == 0 {
                    continue;
                }
                let before = *prefix.get(idx).ok_or_else(|| {
                    ExrError::Input("Per-pixel prefix index out of range.".into())
                })?;
                let start = slice.base_offset + before as i64 * slice.sample_stride as i64;
                let far = start + (cnt as i64 - 1) * slice.sample_stride as i64;
                if start < 0 || far < 0 {
                    return Err(ExrError::Arg(
                        "Deep sample address is out of range (negative).".into(),
                    ));
                }
                let span_lo = start.min(far);
                let span_hi = start.max(far) + dst_size;
                lo = Some(lo.map_or(span_lo, |l| l.min(span_lo)));
                hi = hi.max(span_hi);
            }
            out.push(match lo {
                Some(l) => l as usize..hi as usize,
                None => 0..0,
            });
        }
    }
    Ok(out)
}

/// Write one fill sample (Xdr LE) of `type_` from an `f64` value -- mirrors the
/// per-type cast in C++ deep fill (`(unsigned)`, `half`, `float`).  Returns an
/// error for the `NumPixelTypes` sentinel.
fn write_fill_value(dst: &mut [u8], type_: PixelType, fill_value: f64) -> Result<()> {
    match type_ {
        PixelType::Uint => {
            let v = fill_value as u32;
            dst.copy_from_slice(&v.to_le_bytes());
        }
        PixelType::Half => {
            let v = imath_rs::Half::from(fill_value as f32).bits();
            dst.copy_from_slice(&v.to_le_bytes());
        }
        PixelType::Float => {
            let v = fill_value as f32;
            dst.copy_from_slice(&v.to_le_bytes());
        }
        PixelType::NumPixelTypes => {
            return Err(ExrError::Arg("Unknown pixel data type.".into()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests -- write -> read round-trip (the key correctness gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{
        Channel, ChannelList, ChannelListAttribute, Compression, LineOrder,
        TileDescriptionAttribute,
    };
    use crate::attr::tile_description::LevelRoundingMode;
    use crate::deep::frame_buffer::DeepSlice;
    use crate::deep::DeepTiledOutputFile;
    use crate::frame_buffer::Slice;
    use crate::io::{MemIStream, MemOStream};
    use imath_rs::{Half, V2f, V2i};

    /// A deep tiled image description used by the round-trip tests.
    struct DeepImg {
        w: i32,
        h: i32,
        min_x: i32,
        min_y: i32,
        /// per-pixel sample counts, raster order over the w*h data window.
        counts: Vec<u32>,
        /// channel name + on-disk type, file-channel order.
        channels: Vec<(String, PixelType)>,
        /// per channel: the f64 value of each sample (raster order, contiguous).
        values: Vec<Vec<f64>>,
    }

    /// Build a `w`x`h` deep image: channels given, sample value of channel `ci`
    /// at sample `s` of pixel `pix` is `base[ci] + pix*10 + s`.
    fn make_img(
        w: i32,
        h: i32,
        min_x: i32,
        min_y: i32,
        counts: Vec<u32>,
        channels: Vec<(String, PixelType)>,
    ) -> DeepImg {
        let n = (w * h) as usize;
        assert_eq!(counts.len(), n);
        let bases: Vec<f64> = (0..channels.len()).map(|i| 1000.0 * (i as f64 + 1.0)).collect();
        let mut values = vec![Vec::new(); channels.len()];
        for (ci, base) in bases.iter().enumerate() {
            for (pix, &c) in counts.iter().enumerate() {
                for s in 0..c {
                    values[ci].push(base + (pix as f64) * 10.0 + s as f64);
                }
            }
        }
        DeepImg { w, h, min_x, min_y, counts, channels, values }
    }

    /// Header for a deep tiled image with the given tile description + comp.
    fn make_header(img: &DeepImg, comp: Compression, td: TileDescription, lo: LineOrder) -> Header {
        let mut h = Header::with_dimensions(
            img.w,
            img.h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            lo,
            comp,
        )
        .unwrap();
        if img.min_x != 0 || img.min_y != 0 {
            let dw = Box2i {
                min: V2i { x: img.min_x, y: img.min_y },
                max: V2i { x: img.min_x + img.w - 1, y: img.min_y + img.h - 1 },
            };
            *h.data_window_mut().unwrap() = dw;
        }
        h.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
        let mut cl = ChannelList::new();
        for (name, ty) in &img.channels {
            cl.insert(name, Channel::new(*ty, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.set_type(DEEP_TILE).unwrap();
        h
    }

    /// Flat sample-count buffer (u32 LE per pixel) addressed in **absolute**
    /// window coords: `off = y*yStride + x*xStride`, xStride=4, yStride=4*w,
    /// sized to the absolute extent `(min_y + h) * yStride`.
    fn build_count_buffer(img: &DeepImg) -> (Vec<u8>, i32, i32) {
        let x_stride = 4i32;
        let y_stride = 4 * img.w;
        let total = ((img.min_y + img.h) as i64 * y_stride as i64) as usize;
        let mut sc = vec![0u8; total];
        for (i, &c) in img.counts.iter().enumerate() {
            let x = img.min_x + (i as i32 % img.w);
            let y = img.min_y + (i as i32 / img.w);
            let off = (y as i64 * y_stride as i64 + x as i64 * x_stride as i64) as usize;
            sc[off..off + 4].copy_from_slice(&c.to_le_bytes());
        }
        (sc, x_stride, y_stride)
    }

    /// Build the flat per-channel sample heap (all channels concatenated, each
    /// channel contiguous in raster order); returns the heap + each channel's
    /// base offset.
    fn build_sample_heap(img: &DeepImg) -> (Vec<u8>, Vec<i64>) {
        let mut heap = Vec::new();
        let mut bases = Vec::new();
        for (ci, (_name, ty)) in img.channels.iter().enumerate() {
            bases.push(heap.len() as i64);
            for &v in &img.values[ci] {
                match ty {
                    PixelType::Half => heap.extend_from_slice(&Half::from(v as f32).bits().to_le_bytes()),
                    PixelType::Float => heap.extend_from_slice(&(v as f32).to_le_bytes()),
                    PixelType::Uint => heap.extend_from_slice(&(v as u32).to_le_bytes()),
                    PixelType::NumPixelTypes => unreachable!("test never uses the sentinel type"),
                }
            }
        }
        (heap, bases)
    }

    /// Make the DeepFrameBuffer (flat model): sample-count slice + one deep
    /// slice per channel pointing at its base in the heap.
    fn make_frame_buffer(img: &DeepImg, bases: &[i64], x_stride: i32, y_stride: i32) -> DeepFrameBuffer {
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: x_stride as usize,
            y_stride: y_stride as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        for (ci, (name, ty)) in img.channels.iter().enumerate() {
            let ss = pixel_type_size(*ty).unwrap();
            fb.insert(
                name,
                DeepSlice::new(*ty, bases[ci], 8, 8 * img.w as usize, ss, 1, 1, 0.0, false, false)
                    .unwrap(),
            )
            .unwrap();
        }
        fb
    }

    /// Write `img` (whole level 0) to an in-memory deep tiled EXR via `DeepTiledOutputFile`.
    fn write_deep_tiled(img: &DeepImg, comp: Compression, td: TileDescription, lo: LineOrder) -> Vec<u8> {
        let header = make_header(img, comp, td, lo);
        let (sc, xs, ys) = build_count_buffer(img);
        let (heap, bases) = build_sample_heap(img);
        let fb = make_frame_buffer(img, &bases, xs, ys);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            let nx = df.num_x_tiles(0).unwrap();
            let ny = df.num_y_tiles(0).unwrap();
            df.write_tiles(&sc, &heap, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        out.into_data()
    }

    /// One-level tile description.
    fn td_one(x: u32, y: u32) -> TileDescription {
        TileDescription {
            x_size: x,
            y_size: y,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }
    }

    /// Full whole-level-0 round-trip: write, read back two-pass, and
    /// assert (a) per-pixel COUNTS bit-exact and (b) EVERY sample bit-exact.
    fn round_trip(img: &DeepImg, comp: Compression, td: TileDescription, lo: LineOrder) {
        let bytes = write_deep_tiled(img, comp, td, lo);

        let (_sc, xs, ys) = build_count_buffer(img);
        let (_heap, bases) = build_sample_heap(img);
        let fb = make_frame_buffer(img, &bases, xs, ys);

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().type_().unwrap(), DEEP_TILE);
        inf.set_frame_buffer(&fb).unwrap();

        let nx = inf.num_x_tiles(0).unwrap();
        let ny = inf.num_y_tiles(0).unwrap();

        // Pass 1: counts into a fresh zeroed count buffer (absolute extent).
        let (template, x2, y2) = build_count_buffer(img);
        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf)
            .unwrap();

        // Verify per-pixel counts match exactly.
        let n = (img.w * img.h) as usize;
        for i in 0..n {
            let x = img.min_x + (i as i32 % img.w);
            let y = img.min_y + (i as i32 / img.w);
            let off = (y as i64 * y2 as i64 + x as i64 * x2 as i64) as usize;
            let got = u32::from_le_bytes(counts_buf[off..off + 4].try_into().unwrap());
            assert_eq!(got, img.counts[i], "count mismatch at raster {i}");
        }

        // Pass 2: sample data into a fresh zeroed heap sized like the writer's.
        let (expected_heap, _bases) = build_sample_heap(img);
        let mut heap = vec![0u8; expected_heap.len()];
        inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap)
            .unwrap();

        assert_eq!(heap, expected_heap, "deep sample data mismatch ({comp:?}/{lo:?})");
    }

    fn variable_counts_4x4() -> Vec<u32> {
        vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2, 0, 3, 2, 1]
    }

    #[test]
    fn round_trip_one_level_none_half() {
        let img = make_img(4, 4, 0, 0, variable_counts_4x4(), vec![("A".into(), PixelType::Half)]);
        round_trip(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
    }

    /// Read the SAME multi-tile deep tiled file with
    /// `num_threads` in {0, 1, 2, 3, 8, 17} and assert every thread count
    /// produces byte-identical sample data (and matching counts) — the deep
    /// tiled analogue of the flat tiled reader's parallel determinism test.
    /// A `5x4` nominal tile over a `23x19` image gives `5*5 = 25` tiles
    /// (including partial edge tiles), with a non-uniform per-pixel count
    /// pattern exercising the per-pixel `samples_before` addressing across
    /// tile boundaries.
    ///
    /// ONE channel ("A") -- the
    /// "1-channel" half of the required coverage (see
    /// `Probe::PlanDisjointAccepted`); the 2+-channel
    /// counterpart is `parallel_read_zips_matches_across_thread_counts`
    /// below.
    #[test]
    fn parallel_read_is_byte_identical_to_serial_across_thread_counts() {
        let (w, h) = (23i32, 19i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| i % 4).collect();
        let img = make_img(w, h, 0, 0, counts, vec![("A".into(), PixelType::Half)]);
        let td = td_one(5, 4);
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td, LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_heap, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
            let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();
            let nx = inf.num_x_tiles(0).unwrap();
            let ny = inf.num_y_tiles(0).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap.len()];
            inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::DeepTileReadParallel, n);
                assert!(
                    crate::probe::get(crate::probe::Probe::PlanDisjointAccepted) > 0,
                    "num_threads={n}: the parallel decode fast path did not engage \
                     on a 1-channel image"
                );
            }

            assert_eq!(heap, expected_heap, "num_threads={n}: mismatch vs the written sample data");
            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// Slices that disagree on `y_tile_coords` make the parallel read
    /// decline (it shares one per-tile prefix matrix across channels) and
    /// read serially with identical results. One tile row starting at y=0
    /// makes tile-relative and absolute y coincide, so the mixed frame
    /// buffer stays valid for the serial path.
    #[test]
    fn parallel_read_declines_mixed_tile_coords() {
        let (w, h) = (17i32, 6i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = make_img(
            w, h, 0, 0, counts,
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(4, 8), LineOrder::IncreasingY);
        let (template, xs, ys) = build_count_buffer(&img);
        let (_heap, bases) = build_sample_heap(&img);
        let heap_len = build_sample_heap(&img).0.len();

        let mut fb = make_frame_buffer(&img, &bases, xs, ys);
        let (name, ty) = img.channels[0].clone();
        let ss = pixel_type_size(ty).unwrap();
        fb.insert(&name, DeepSlice::new(ty, bases[0], 8, 8 * w as usize, ss, 1, 1, 0.0, false, true).unwrap())
            .unwrap();

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 4] {
            let mut is = MemIStream::new(bytes.clone());
            let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (inf.num_x_tiles(0).unwrap(), inf.num_y_tiles(0).unwrap());
            assert!(nx > 1 && ny == 1, "premise: several tiles in one row");
            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();
            crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
            let mut heap = vec![0u8; heap_len];
            inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();
            crate::probe::assert_declined(crate::probe::Probe::DeepTileReadParallel);
            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from the serial read"),
            }
        }
    }

    /// Tiled counterpart of `deep::scanline_input`'s
    /// `parallel_read_rejects_genuinely_aliased_slices_falls_back_to_serial`
    /// -- see its doc for the full rationale.
    #[test]
    fn parallel_read_rejects_genuinely_aliased_slices_falls_back_to_serial() {
        let (w, h) = (17i32, 15i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = make_img(
            w, h, 0, 0, counts,
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        let td = td_one(4, 4);
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td, LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_heap, real_bases) = build_sample_heap(&img);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap_len = build_sample_heap(&img).0.len();

        // ALIASED frame buffer: both channels point at the SAME base --
        // see the scanline version's identical doc.
        let aliased_bases = vec![real_bases[0], real_bases[0]];
        let fb = make_frame_buffer(&img, &aliased_bases, xs, ys);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
            let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();
            let nx = inf.num_x_tiles(0).unwrap();
            let ny = inf.num_y_tiles(0).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap_len];
            inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();
            crate::probe::assert_declined(crate::probe::Probe::DeepTileReadParallel);
            assert_eq!(
                crate::probe::get(crate::probe::Probe::PlanDisjointAccepted),
                0,
                "num_threads={n}: plan_disjoint must reject a genuinely aliased frame buffer"
            );

            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// Tiled counterpart of `deep::scanline_input`'s
    /// `read_pixels_builds_few_compressors_not_one_per_line` test -- see its
    /// doc for the full rationale. The parallel fast path genuinely engages
    /// too (see `parallel_read_is_byte_identical_to_serial_across_thread_counts`/
    /// `parallel_read_zips_matches_across_thread_counts` above, which
    /// assert engagement directly), this test's own `with_num_threads`
    /// default still exercises the code path this specific counter check
    /// cares about (persistent `self.data_compressor`, not fresh-per-tile).
    #[test]
    fn read_tiles_builds_few_compressors_not_one_per_tile() {
        let (w, h) = (80i32, 80i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = make_img(w, h, 0, 0, counts, vec![("A".into(), PixelType::Half)]);
        let td = td_one(4, 4); // 20x20 = 400 tiles
        let bytes = write_deep_tiled(&img, Compression::Zips, td, LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_heap, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut is = MemIStream::new(bytes.clone());
        // Pin the serial path; the per-thread probe only sees construction on
        // this thread, so a global thread count > 1 would pass vacuously.
        let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(0)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let nx = inf.num_x_tiles(0).unwrap();
        let ny = inf.num_y_tiles(0).unwrap();
        let num_tiles = (nx * ny) as usize;
        assert!(num_tiles >= 300, "test setup: need enough tiles to make a per-tile regression obvious");

        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();

        crate::probe::reset(crate::probe::Probe::NewCompressor);
        let mut heap = vec![0u8; expected_heap.len()];
        inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();
        let delta = crate::probe::get(crate::probe::Probe::NewCompressor);

        assert_eq!(heap, expected_heap, "sanity: the read must still be correct");
        assert!(
            delta < (num_tiles / 4) as u64,
            "read_tiles for {num_tiles} tiles called new_compressor {delta} times -- \
             expected far fewer than one per tile (persistent-field regression?)"
        );
    }

    /// Same matrix over ZIPS (real codec, exercising the per-worker
    /// fresh-compressor requirement for both the count-table and
    /// sample-data compressors) with two channels.
    #[test]
    fn parallel_read_zips_matches_across_thread_counts() {
        let (w, h) = (17i32, 15i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = make_img(
            w, h, 0, 0, counts,
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        let td = td_one(4, 4);
        let bytes = write_deep_tiled(&img, Compression::Zips, td, LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_heap, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
            let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();
            let nx = inf.num_x_tiles(0).unwrap();
            let ny = inf.num_y_tiles(0).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap.len()];
            inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::DeepTileReadParallel, n);
                assert!(
                    crate::probe::get(crate::probe::Probe::PlanDisjointAccepted) > 0,
                    "num_threads={n}: the parallel decode fast path did not engage \
                     (plan_disjoint rejected it -- fell back to serial) on a \
                     2-channel image"
                );
            }

            assert_eq!(heap, expected_heap, "num_threads={n}: mismatch vs the written sample data");
            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    #[test]
    fn round_trip_one_level_zips_half_float() {
        let img = make_img(
            4,
            4,
            0,
            0,
            variable_counts_4x4(),
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        round_trip(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_partial_edge_tiles_none() {
        // 5x3, 2x2 tiles -> 3x2 grid, right column + bottom row are partial.
        let counts = vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2, 0, 3, 1];
        let img = make_img(5, 3, 0, 0, counts, vec![("A".into(), PixelType::Half)]);
        round_trip(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_partial_edge_tiles_zips() {
        let counts = vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2, 0, 3, 1];
        let img = make_img(5, 3, 0, 0, counts, vec![("A".into(), PixelType::Half)]);
        round_trip(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_all_zero_counts() {
        let img = make_img(4, 4, 0, 0, vec![0u32; 16], vec![("A".into(), PixelType::Half)]);
        round_trip(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_non_origin_window() {
        let img = make_img(
            4,
            4,
            -3,
            7,
            variable_counts_4x4(),
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        round_trip(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_decreasing_y() {
        let img = make_img(4, 6, 0, 0, (0..24u32).map(|i| i % 4).collect(),
            vec![("A".into(), PixelType::Half)]);
        round_trip(&img, Compression::Zips, td_one(2, 2), LineOrder::DecreasingY);
    }

    /// Read a single non-(0,0) tile and verify only its pixels' samples land.
    #[test]
    fn read_single_non_origin_tile() {
        let img = make_img(
            4,
            4,
            0,
            0,
            variable_counts_4x4(),
            vec![("A".into(), PixelType::Half)],
        );
        let bytes = write_deep_tiled(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (expected_heap, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        // Counts of ONLY tile (1,1), into the image-shaped buffer (its other
        // pixels stay zero): a one-tile request needs nothing else.
        let (template, _x2, _y2) = build_count_buffer(&img);
        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(1, 1, 1, 1, 0, 0, &mut counts_buf).unwrap();

        // Read ONLY tile (1,1): its pixels are the bottom-right 2x2 of the image.
        let mut heap = vec![0u8; expected_heap.len()];
        inf.read_tile(1, 1, 0, 0, &counts_buf, &mut heap).unwrap();

        // The heap of a one-tile request is that tile's raster: the samples of
        // its pixels (global raster indices 10,11,14,15) back to back, from
        // heap offset 0. Each pixel's byte span in the (single-channel HALF)
        // source heap:
        let w = img.w;
        let mut cum: Vec<u64> = vec![0; (w * img.h) as usize + 1];
        for i in 0..(w * img.h) as usize {
            cum[i + 1] = cum[i] + img.counts[i] as u64;
        }
        let mut expected_tile = Vec::new();
        for &(gx, gy) in &[(2i32, 2i32), (3, 2), (2, 3), (3, 3)] {
            let pix = (gy * w + gx) as usize;
            expected_tile.extend_from_slice(&expected_heap[cum[pix] as usize * 2..cum[pix + 1] as usize * 2]);
        }
        assert!(!expected_tile.is_empty(), "premise: tile (1,1) holds samples");
        assert_eq!(heap[..expected_tile.len()], expected_tile[..], "tile (1,1) samples");
        assert!(
            heap[expected_tile.len()..].iter().all(|&b| b == 0),
            "a one-tile request must not write outside its own raster"
        );
    }

    /// MIPMAP: write all levels, read back level 0 and a higher level.
    #[test]
    fn round_trip_mipmap_levels() {
        let (w, h) = (4, 4);
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let img = make_img(w, h, 0, 0, variable_counts_4x4(), vec![("A".into(), PixelType::Half)]);
        let header = make_header(&img, Compression::Zips, td, LineOrder::IncreasingY);
        let (sc, xs, ys) = build_count_buffer(&img);
        let (heap_src, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            for l in 0..df.num_levels().unwrap() {
                let nx = df.num_x_tiles(l).unwrap();
                let ny = df.num_y_tiles(l).unwrap();
                df.write_tiles_l(&sc, &heap_src, 0, nx - 1, 0, ny - 1, l).unwrap();
            }
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        assert_eq!(inf.num_levels().unwrap(), 3);
        inf.set_frame_buffer(&fb).unwrap();

        // Level 0: full round-trip of the count buffer + heap (the count buffer
        // and heap describe the full 4x4 window; coarser levels address interior
        // sub-rects of that same window, so the same buffers work).
        let (template, x2, y2) = build_count_buffer(&img);
        let mut counts0 = vec![0u8; template.len()];
        let nx0 = inf.num_x_tiles(0).unwrap();
        let ny0 = inf.num_y_tiles(0).unwrap();
        inf.read_pixel_sample_counts_l(0, nx0 - 1, 0, ny0 - 1, 0, &mut counts0).unwrap();
        for i in 0..(w * h) as usize {
            let x = i as i32 % w;
            let y = i as i32 / w;
            let off = (y as i64 * y2 as i64 + x as i64 * x2 as i64) as usize;
            let got = u32::from_le_bytes(counts0[off..off + 4].try_into().unwrap());
            assert_eq!(got, img.counts[i], "L0 count mismatch at {i}");
        }
        let mut heap0 = vec![0u8; heap_src.len()];
        inf.read_tiles_l(0, nx0 - 1, 0, ny0 - 1, 0, &counts0, &mut heap0).unwrap();
        assert_eq!(heap0, heap_src, "L0 sample data mismatch");

        // Higher level (level 1: a single 2x2 tile over the level-1 data window).
        // Read its counts then its samples; assert they match what the writer
        // gathered for that level's data window (an interior rect of the window).
        let l = 1;
        let nx1 = inf.num_x_tiles(l).unwrap();
        let ny1 = inf.num_y_tiles(l).unwrap();
        let mut counts1 = vec![0u8; template.len()];
        inf.read_pixel_sample_counts_l(0, nx1 - 1, 0, ny1 - 1, l, &mut counts1).unwrap();
        let rect = inf.data_window_for_level_l(l).unwrap();
        // The level-1 data window pixels' counts (read back) must equal the
        // source counts at those same absolute pixels (the writer reads the same
        // sample-count buffer for every level).
        for gy in rect.min.y..=rect.max.y {
            for gx in rect.min.x..=rect.max.x {
                let off = (gy as i64 * y2 as i64 + gx as i64 * x2 as i64) as usize;
                let got = u32::from_le_bytes(counts1[off..off + 4].try_into().unwrap());
                let exp = img.counts[(gy * w + gx) as usize];
                assert_eq!(got, exp, "L1 count mismatch at ({gx},{gy})");
            }
        }
        // The heap of a whole-level request is that level's raster, so the
        // level's own counts (`counts1`) are all the read needs: the samples of
        // the level-1 pixels back to back, from heap offset 0.
        let mut heap1 = vec![0u8; heap_src.len()];
        inf.read_tiles_l(0, nx1 - 1, 0, ny1 - 1, l, &counts1, &mut heap1).unwrap();
        // Expected: the level-1 pixels' spans of the source heap (the writer wrote
        // exactly these bytes), in level raster order.
        let mut cum: Vec<u64> = vec![0; (w * h) as usize + 1];
        for i in 0..(w * h) as usize {
            cum[i + 1] = cum[i] + img.counts[i] as u64;
        }
        let mut expected1 = Vec::new();
        for gy in rect.min.y..=rect.max.y {
            for gx in rect.min.x..=rect.max.x {
                let pix = (gy * w + gx) as usize;
                expected1.extend_from_slice(&heap_src[cum[pix] as usize * 2..cum[pix + 1] as usize * 2]);
            }
        }
        assert!(!expected1.is_empty(), "premise: level 1 holds samples");
        assert_eq!(heap1[..expected1.len()], expected1[..], "L1 sample mismatch");
        assert!(heap1[expected1.len()..].iter().all(|&b| b == 0), "L1 read wrote past its raster");
    }

    /// Absent file channel -> filled with its fill value.
    #[test]
    fn absent_channel_is_filled() {
        let img = make_img(
            4,
            4,
            0,
            0,
            variable_counts_4x4(),
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        let bytes = write_deep_tiled(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (heap_src, bases) = build_sample_heap(&img);
        let mut fb = make_frame_buffer(&img, &bases, xs, ys);
        // Fill channel F (FLOAT, fill 9.0) the file does not provide.
        let f_base = heap_src.len() as i64;
        fb.insert(
            "F",
            DeepSlice::new(PixelType::Float, f_base, 8, 8 * img.w as usize, 4, 1, 1, 9.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let nx = inf.num_x_tiles(0).unwrap();
        let ny = inf.num_y_tiles(0).unwrap();
        let (template, _x2, _y2) = build_count_buffer(&img);
        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf).unwrap();

        let total: usize = img.counts.iter().map(|&c| c as usize).sum();
        let mut heap = vec![0u8; heap_src.len() + total * 4];
        inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap).unwrap();

        for s in 0..total {
            let off = f_base as usize + s * 4;
            let v = f32::from_le_bytes(heap[off..off + 4].try_into().unwrap());
            assert_eq!(v, 9.0, "fill sample {s} mismatch");
        }
    }

    #[test]
    fn no_frame_buffer_errors() {
        let img = make_img(4, 4, 0, 0, vec![1u32; 16], vec![("A".into(), PixelType::Half)]);
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        let mut counts = vec![0u8; 16 * 4];
        let err = inf.read_pixel_sample_counts(0, 1, 0, 1, 0, 0, &mut counts).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn invalid_level_errors() {
        let img = make_img(4, 4, 0, 0, vec![1u32; 16], vec![("A".into(), PixelType::Half)]);
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 16 * 4];
        // lx=1 invalid for ONE_LEVEL.
        let err = inf.read_pixel_sample_counts(0, 1, 0, 1, 1, 0, &mut counts).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn rejects_non_deep_tiled_file() {
        // A flat (non-deep) tiled file must be rejected by the deep tiled reader.
        use crate::tiled::TiledOutputFile;
        let td = td_one(2, 2);
        let mut hdr = Header::with_dimensions(
            4, 4, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        hdr.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let pixels = vec![0u8; 4 * 4 * 2];
        let mut fb = crate::frame_buffer::FrameBuffer::new();
        fb.insert("R", Slice {
            type_: PixelType::Half, base_offset: 0, x_stride: 2, y_stride: 8,
            x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
        }).unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = TiledOutputFile::new(&mut out, &hdr).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        match DeepTiledInputFile::new(&mut is) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("deep tiled reader accepted a flat tiled file"),
        }
    }

    // -- Allocation-guard rejection paths (malformed chunk size fields) -----

    /// Byte offset of the (single) tile chunk in a one-tile deep tiled file:
    /// parse the header length, then read the first tile-offset slot.  Chunk
    /// field layout from that offset: dx,dy,lx,ly (4*i32) @ +0..16,
    /// packedCountSize(u64) @ +16, packedDataSize(u64) @ +24,
    /// unpackedDataSize(u64) @ +32.
    fn first_chunk_offset(bytes: &[u8]) -> usize {
        let mut is = MemIStream::new(bytes.to_vec());
        let (_h, _v) = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;
        u64::from_le_bytes(bytes[header_len..header_len + 8].try_into().unwrap()) as usize
    }

    /// One-tile 2x2 single-HALF-channel deep image (a single chunk to patch).
    fn one_tile_img() -> DeepImg {
        make_img(2, 2, 0, 0, vec![1, 1, 1, 1], vec![("A".into(), PixelType::Half)])
    }

    /// A count-table `packedSampleCountTableSize` above the writer's bound
    /// (`maxSampleCountTableSize`) — here `u64::MAX`, also the near-`u64::MAX`
    /// allocation path — must be rejected with `ExrError::Input`, no panic.
    #[test]
    fn rejects_oversize_count_table() {
        let img = one_tile_img();
        let mut bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        bytes[off + 16..off + 24].copy_from_slice(&u64::MAX.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 16];
        match inf.read_pixel_sample_counts(0, 0, 0, 0, 0, 0, &mut counts) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// Pass 1 (counts only) of the one-tile image with its chunk patched:
    /// `unpacked_at` replaces `unpackedDataSize`, `last_count` the tile's last
    /// cumulative count (row 1, x 1). Returns the `read_pixel_sample_counts` result.
    fn read_counts_of_patched(unpacked_at: Option<u64>, last_count: Option<i32>) -> Result<()> {
        let img = one_tile_img();
        let mut bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        if let Some(u) = unpacked_at {
            bytes[off + 32..off + 40].copy_from_slice(&u.to_le_bytes());
        }
        // Chunk leader: dx, dy, lx, ly (16) + three u64 sizes (24), then the raw
        // (NONE) cumulative count table, per tile row: [1, 2], [1, 2].
        if let Some(c) = last_count {
            bytes[off + 52..off + 56].copy_from_slice(&c.to_le_bytes());
        }
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let (template, _x2, _y2) = build_count_buffer(&img);
        let mut counts = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(0, 0, 0, 0, 0, 0, &mut counts)
    }

    /// An `unpackedDataSize` that disagrees with `packedDataSize` under NONE is a
    /// corrupt chunk leader (`OpenEXRCore/chunk.c`): a counts-only read fails too.
    #[test]
    fn rejects_unpacked_size_mismatch() {
        assert!(read_counts_of_patched(None, None).is_ok());
        match read_counts_of_patched(Some(999), None) {
            Err(ExrError::Input(m)) => assert!(m.contains("uncompressed"), "{m}"),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// A monotonic count table claiming more samples than the data holds (5 HALF
    /// samples, unpacked 8 bytes) is C++'s "Corrupt sample count table"
    /// (`OpenEXRCore/decoding.c:270-275`), raised on every count decode --
    /// `readPixelSampleCounts` included.
    #[test]
    fn counts_only_read_rejects_a_table_larger_than_the_data() {
        match read_counts_of_patched(None, Some(3)) {
            Err(ExrError::Input(m)) => assert!(m.contains("implied by the decoded sample counts"), "{m}"),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// A `packedDataSize` larger than `unpackedDataSize` (here `u64::MAX`) must be
    /// rejected before allocating.
    ///
    /// Like the scanline sibling of this test
    /// (`scanline_input.rs::rejects_packed_data_over_unpacked`), rejection
    /// happens at the FIRST chunk access (`read_pixel_sample_counts`) —
    /// `seek_chunk` ports C++'s leader validation
    /// (`OpenEXRCore/chunk.c::read_and_validate_chunk_leader`'s deep-tiled
    /// branch), which both `readPixelSampleCounts` and `readTiles` call via
    /// the same underlying chunk-info reader in real OpenEXR.
    #[test]
    fn rejects_packed_data_over_unpacked() {
        let img = one_tile_img();
        let mut bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        bytes[off + 24..off + 32].copy_from_slice(&u64::MAX.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let (template, _x2, _y2) = build_count_buffer(&img);
        let mut counts = vec![0u8; template.len()];
        match inf.read_pixel_sample_counts(0, 0, 0, 0, 0, 0, &mut counts) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// `raw_tile_data` (the `copy_pixels` raw-chunk
    /// path) must apply the SAME leader bound as the normal decode
    /// path, not allocate straight from an attacker-controlled on-wire size.
    /// Crafts a tile chunk whose `packedDataSize` field claims ~16 EiB
    /// (`u64::MAX`) -- if this test completes at all with an `Err`, the
    /// bound is working; it must never reach `try_filled_vec`/`vec![]` with
    /// that size.
    #[test]
    fn raw_tile_data_rejects_oversize_packed_data_before_allocating() {
        let img = one_tile_img();
        let mut bytes = write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        // Chunk layout: dx,dy,lx,ly (4*i32) @0..16, packedCount(u64) @16,
        // packedData(u64) @24.
        bytes[off + 24..off + 32].copy_from_slice(&u64::MAX.to_le_bytes());

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        match inf.raw_tile_data(0, 0, 0, 0) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error (no giant allocation), got {other:?}"),
        }
    }

    /// A HALF file channel read into a FLOAT frame-buffer slice must
    /// convert each sample, not raw-copy the file's 2 bytes into a 4-byte slot.
    #[test]
    fn converts_half_file_channel_to_float_slice() {
        let img = one_tile_img(); // single HALF channel "A", counts all 1.
        let bytes =
            write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);

        let (sc, xs, ys) = build_count_buffer(&img);
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: xs as usize,
            y_stride: ys as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        fb.insert(
            "A",
            DeepSlice::new(PixelType::Float, 0, 8, 8 * img.w as usize, 4, 1, 1, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts_buf = vec![0u8; sc.len()];
        inf.read_pixel_sample_counts(0, 0, 0, 0, 0, 0, &mut counts_buf).unwrap();
        let total = img.counts.iter().sum::<u32>() as usize;
        let mut heap = vec![0u8; total * 4];
        inf.read_tiles(0, 0, 0, 0, 0, 0, &counts_buf, &mut heap).unwrap();

        // "A"'s values were built as base(1000.0) + pix*10 + s (see `make_img`).
        let (expected_half_heap, _bases) = build_sample_heap(&img);
        assert_eq!(expected_half_heap.len(), total * 2, "sanity: HALF heap is half the FLOAT size");
        for s in 0..total {
            let half_bits = u16::from_le_bytes(
                expected_half_heap[s * 2..s * 2 + 2].try_into().unwrap(),
            );
            let expected = f32::from(Half::from_bits(half_bits));
            let got = f32::from_le_bytes(heap[s * 4..s * 4 + 4].try_into().unwrap());
            assert_eq!(got, expected, "converted sample {s}");
        }
    }

    /// A compressed count table that packs down to zero bytes (every
    /// pixel of the tile has zero samples) is a valid all-zero row.
    #[test]
    fn zero_length_count_table_reads_as_zero_counts() {
        let img = make_img(2, 2, 0, 0, vec![0, 0, 0, 0], vec![("A".into(), PixelType::Half)]);
        let mut bytes =
            write_deep_tiled(&img, Compression::Zips, td_one(2, 2), LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        bytes[off + 16..off + 24].copy_from_slice(&0u64.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let (template, _x2, _y2) = build_count_buffer(&img);
        let mut counts = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(0, 0, 0, 0, 0, 0, &mut counts).unwrap();
        assert!(counts.iter().all(|&b| b == 0), "expected all-zero counts");
    }

    /// The sample-count slice's sampling needs its own check (the
    /// `set_frame_buffer` loop only walks the named deep slices). A count
    /// slice with non-unit sampling must be rejected.
    #[test]
    fn count_slice_rejects_non_unit_sampling() {
        let img = one_tile_img();
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let mut fb = make_frame_buffer(&img, &bases, xs, ys);
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: xs as usize,
            y_stride: ys as usize,
            x_sampling: 2,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();

        let bytes =
            write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        match inf.set_frame_buffer(&fb) {
            Err(ExrError::Arg(_)) => {}
            other => panic!("expected Arg error, got {other:?}"),
        }
    }

    /// `DeepTiledOutputFile::from_part` must reject a header whose
    /// `type` already names a different (non-deeptile) part.
    #[test]
    fn from_part_rejects_type_mismatched_header() {
        let mut hdr = Header::with_dimensions(
            2, 2, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr.insert("tiles", TileDescriptionAttribute::new(td_one(2, 2))).unwrap();
        hdr.set_type(crate::header::SCANLINE_IMAGE).unwrap();

        let mut out = MemOStream::new();
        match crate::deep::DeepTiledOutputFile::from_part(&mut out, &hdr, None, 0) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("expected a type-mismatch error, got Ok"),
        }
    }

    /// `OpenEXRCore/decoding.c:307-318`: a deep tiled part whose
    /// `version` attribute is not 1 must be rejected.
    #[test]
    fn rejects_non_unit_version_attribute() {
        let img = one_tile_img();
        let mut header = make_header(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        header.insert("version", crate::attr::IntAttribute::new(2)).unwrap();

        let mut is = MemIStream::new(Vec::new());
        match DeepTiledInputFile::from_part(&mut is, header, 0, None, vec![8]) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("expected a version-mismatch error, got Ok"),
        }
    }

    /// The reader's own tile-size bound must match C++ 3.4's real
    /// open-time gate (`OpenEXRCore/validation.c:593-595`, `x_size`/`y_size`
    /// each `<= INT_MAX/4`), not an ad-hoc `tileX*tileY*4 > u32::MAX`
    /// product check emulating the LEGACY WRITER's
    /// `maxSampleCountTableSize` guard (`ImfDeepTiledOutputFile.cpp:1210`,
    /// a writer-only concern with no reader counterpart in C++). A per-dimension `x_size` of 0 is invalid either way.
    #[test]
    fn rejects_zero_tile_dimension() {
        let img = one_tile_img();
        let header = make_header(&img, Compression::NoCompression, td_one(0, 4), LineOrder::IncreasingY);
        let mut is = MemIStream::new(Vec::new());
        // Readers route `sanity_check` through `sanity_check_at_open`,
        // which reclassifies a non-"missing attribute" `Arg` to `Input`
        // (`ImfContext.cpp:127-138`, verified) — this is "Invalid tile size in
        // image header.", not a missing attribute, so it becomes `Input`.
        match DeepTiledInputFile::from_part(&mut is, header, 0, None, vec![8]) {
            Err(ExrError::Input(_)) => {}
            Err(other) => panic!("expected Input error, got {other:?}"),
            Ok(_) => panic!("expected a zero-tile-dimension error, got Ok"),
        }
    }

    /// A tile dimension within `INT_MAX/4` is accepted even though its
    /// product with the OTHER dimension times 4 would overflow `u32::MAX` —
    /// proving this is a per-dimension bound, not the old product check
    /// (which would have wrongly rejected this legitimate large-but-valid
    /// tile description).
    #[test]
    fn accepts_tile_dimension_within_int_max_over_4() {
        let img = one_tile_img();
        // 70000 * 70000 * 4 > u32::MAX, but each dimension alone is far
        // under INT_MAX/4 (~536,870,911) -- the old product-based check
        // would have rejected this; the per-dimension check must not.
        let header = make_header(&img, Compression::NoCompression, td_one(70_000, 70_000), LineOrder::IncreasingY);
        let mut is = MemIStream::new(Vec::new());
        DeepTiledInputFile::from_part(&mut is, header, 0, None, vec![8])
            .expect("a within-bound tile dimension pair must be accepted");
    }

    /// Addressing perf regression test, build-mode and load INDEPENDENT: a
    /// wall-clock bound (`elapsed.as_secs_f64() < 2.0`) is inherently
    /// flaky -- it fails under a debug build / a loaded machine even when
    /// the O((W*H)^2) quadratic-addressing bug it guards against is NOT
    /// present.
    ///
    /// Instead this counts real operations via
    /// [`crate::deep::sample_table::sample_count_at_calls`] -- a `#[cfg(test)]`
    /// thread-local counter incremented once per call to `sample_count_at`,
    /// the ONE primitive every deep addressing path (`RowPrefixCache`
    /// included) bottoms out on. With the O(width*height) `RowPrefixCache`
    /// fix in place, a round trip over an NxN image makes O(N^2) calls
    /// total (each row's within-row cumulative array is built at most
    /// once); a 2x linear-dimension increase (4x total pixels) should
    /// therefore make the call count scale ~4x. Before the fix
    /// (`samples_before_pixel` re-summed from `(min_x, min_y)` on every
    /// pixel of every channel of every tile), the SAME comparison would
    /// have shown roughly a 16x+ scaling (verified manually below) -- a
    /// wide, unmissable margin between "linear" and "quadratic" that needs
    /// no wall clock at all.
    ///
    /// Manual verification that this test actually catches the regression
    /// (not committed as a permanent fixture):
    /// temporarily reverting `RowPrefixCache::before_pixel`/`before_line` to
    /// re-walk `[min_x, x)`/`[min_y, y)` from scratch on every call (the old
    /// O((W*H)^2) `samples_before_pixel` free function this module used to
    /// call directly) made this test FAIL with a ratio in the hundreds for
    /// the sizes below, while the current code passes with a ratio close to
    /// 4. The revert was not committed.
    #[test]
    fn addressing_is_linear_not_quadratic() {
        fn sample_count_at_calls_for(w: i32, h: i32) -> u64 {
            // `sample_count_at`'s
            // call counter is `#[cfg(test)]` THREAD-LOCAL (see
            // `sample_table.rs`), which is what gives every concurrently
            // running `cargo test` its own isolated count — a SHARED atomic
            // would be polluted by other tests' deep addressing calls running
            // on other OS threads at the same time. The trade-off: if `round_trip`
            // below ever became a PARALLEL deep read/write (chunks decoded on
            // rayon worker threads), this counter would silently UNDER-count
            // (workers have their own separate thread-locals this test's
            // `crate::probe::get(crate::probe::Probe::SampleCountAt)` never reads back), turning a real
            // O(N^2) regression into a false pass. `round_trip` takes no
            // thread-count parameter and every deep reader/writer constructor
            // still defaults to the process-wide global (asserted here to be
            // 0 — this lib test binary's invariant, see `threading.rs`'s test
            // module doc: "no test in this module may call
            // set_global_thread_count"), so this is a loud, immediate failure
            // instead of a silent miscount if that invariant is ever broken.
            assert_eq!(
                crate::threading::global_thread_count(), 0,
                "the call-count regression test requires SERIAL deep construction \
                 (see this assert's doc comment) -- the global thread count must stay 0 \
                 in this test binary."
            );
            let n = (w * h) as usize;
            // One sample per pixel keeps the test's memory bounded while
            // still exercising every pixel's addressing.
            let counts = vec![1u32; n];
            let channels = vec![
                ("A".into(), PixelType::Float),
                ("B".into(), PixelType::Float),
                ("C".into(), PixelType::Float),
                ("D".into(), PixelType::Float),
            ];
            let img = make_img(w, h, 0, 0, counts, channels);
            let td = td_one(64, 64);

            crate::probe::reset(crate::probe::Probe::SampleCountAt);
            round_trip(&img, Compression::NoCompression, td, LineOrder::IncreasingY);
            crate::probe::get(crate::probe::Probe::SampleCountAt)
        }

        // 128x128 -> 256x256: exactly 4x the total pixel count, small enough
        // to run fast even if the quadratic bug were reintroduced (unlike
        // the old test's 1024x1024, which the module doc says was a real
        // multi-minute+ HANG under that bug, not merely slow).
        let small = sample_count_at_calls_for(128, 128);
        let large = sample_count_at_calls_for(256, 256);

        assert!(small > 0, "the counter must have counted something for the small case");
        let ratio = large as f64 / small as f64;
        assert!(
            ratio < 6.0,
            "sample_count_at call count scaled {ratio:.2}x for a 4x pixel-count increase \
             (small={small} calls, large={large} calls) -- expected near-linear \
             (~4x, threshold 6x); O((W*H)^2) addressing would give ~16x+ (or far worse, \
             since W*H itself grows 4x)"
        );
    }

    /// `frame_buffer()` returns the last frame buffer installed via
    /// `set_frame_buffer`, matching C++ `frameBuffer() const`.
    #[test]
    fn frame_buffer_accessor_returns_installed_buffer() {
        let img = one_tile_img();
        let bytes =
            write_deep_tiled(&img, Compression::NoCompression, td_one(2, 2), LineOrder::IncreasingY);
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        // Empty (never set) until `set_frame_buffer` is called.
        assert!(inf.frame_buffer().iter().next().is_none());
        inf.set_frame_buffer(&fb).unwrap();
        assert_eq!(inf.frame_buffer().iter().count(), fb.iter().count());
        assert!(inf.frame_buffer().find_slice("A").is_some());
    }

    /// `total_tiles()` / `tile_order()` for a MIPMAP file: total tiles
    /// sums every level's `numXTiles * numYTiles`, and `tile_order()` lists
    /// exactly that many tiles (every tile is present since the whole file
    /// was written).
    #[test]
    fn total_tiles_and_tile_order() {
        let img = make_img(4, 4, 0, 0, variable_counts_4x4(), vec![("A".into(), PixelType::Half)]);
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let bytes = write_deep_tiled(&img, Compression::NoCompression, td, LineOrder::IncreasingY);

        let mut is = MemIStream::new(bytes);
        let inf = DeepTiledInputFile::new(&mut is).unwrap();
        let num_levels = inf.num_levels().unwrap();
        let mut expected_total = 0i64;
        for l in 0..num_levels {
            expected_total += inf.num_x_tiles(l).unwrap() as i64 * inf.num_y_tiles(l).unwrap() as i64;
        }
        assert_eq!(inf.total_tiles().unwrap(), expected_total);

        let (dx, dy, lx, ly) = inf.tile_order().unwrap();
        assert_eq!(dx.len(), expected_total as usize);
        assert_eq!(dy.len(), expected_total as usize);
        assert_eq!(lx.len(), expected_total as usize);
        assert_eq!(ly.len(), expected_total as usize);
    }

    /// Any tile rectangle `[dx1..=dx2] x [dy1..=dy2]`, down to one tile, reads
    /// into a heap that is the raster of just those tiles (C++ `readTiles`
    /// takes per-pixel pointers, so pixels outside the request are never
    /// consulted). The expected bytes come straight from the source samples,
    /// independent of the reader's addressing. A non-origin window with edge
    /// tiles clipped on both axes, NONE and ZIPS, serial and parallel decode.
    #[test]
    fn tile_range_reads_use_a_range_local_heap() {
        let (w, h) = (7i32, 6i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 7 + 3) % 4).collect();
        let img = make_img(
            w, h, 0, 2, counts,
            vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)],
        );
        let (tw, th) = (2i32, 4i32); // 4 x 2 tiles; the right and bottom ones are clipped
        let mut cum: Vec<usize> = vec![0; (w * h) as usize + 1];
        for i in 0..(w * h) as usize {
            cum[i + 1] = cum[i] + img.counts[i] as usize;
        }
        let (template, xs, ys) = build_count_buffer(&img);
        for comp in [Compression::NoCompression, Compression::Zips] {
            let bytes = write_deep_tiled(&img, comp, td_one(tw as u32, th as u32), LineOrder::IncreasingY);
            for threads in [0i32, 4] {
                for (dx1, dx2, dy1, dy2) in
                    (0..4).flat_map(|a| (a..4).flat_map(move |b| (0..2).flat_map(move |c| (c..2).map(move |d| (a, b, c, d)))))
                {
                    let (x0, y0) = (dx1 * tw, dy1 * th);
                    let (x1, y1) = (((dx2 + 1) * tw).min(w) - 1, ((dy2 + 1) * th).min(h) - 1);
                    // Source samples of the rectangle's pixels in raster order, per channel.
                    let mut spans = Vec::new();
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let p = (y * w + x) as usize;
                            spans.push(cum[p]..cum[p + 1]);
                        }
                    }
                    let n: usize = spans.iter().map(|r| r.len()).sum();
                    let mut fb = DeepFrameBuffer::new();
                    fb.insert_sample_count_slice(
                        Slice::new(PixelType::Uint, -(img.min_y as i64 * ys as i64 + img.min_x as i64 * xs as i64), xs as usize, ys as usize, 1, 1, 0.0, false, false)
                            .unwrap(),
                    )
                    .unwrap();
                    let mut expected = Vec::new();
                    for (ci, (name, ty)) in img.channels.iter().enumerate() {
                        let size = pixel_type_size(*ty).unwrap();
                        fb.insert(
                            name,
                            DeepSlice::new(*ty, expected.len() as i64, 8, 0, size, 1, 1, 0.0, false, false).unwrap(),
                        )
                        .unwrap();
                        for r in &spans {
                            for &v in &img.values[ci][r.clone()] {
                                match ty {
                                    PixelType::Half => expected.extend_from_slice(&Half::from(v as f32).bits().to_le_bytes()),
                                    _ => expected.extend_from_slice(&(v as f32).to_le_bytes()),
                                }
                            }
                        }
                    }
                    let what = format!("{comp:?} threads={threads} tiles x {dx1}..={dx2} y {dy1}..={dy2}");

                    let mut is = MemIStream::new(bytes.clone());
                    let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(threads)).unwrap();
                    inf.set_frame_buffer(&fb).unwrap();
                    // The count raster is window-sized, addressed by absolute pixel; only the
                    // request's pixels are filled.
                    let mut sc = vec![0u8; template.len()];
                    inf.read_pixel_sample_counts(dx1, dx2, dy1, dy2, 0, 0, &mut sc).unwrap();
                    let mut heap = vec![0u8; expected.len()];
                    inf.read_tiles(dx1, dx2, dy1, dy2, 0, 0, &sc, &mut heap).unwrap();
                    assert_eq!(heap, expected, "{what}: {n} samples");
                }
            }
        }
    }

    /// Compact count buffers and heaps are independent of the level-0 raster.
    #[test]
    fn compact_tile_count_buffers_cover_non_origin_mip_and_rip_ranges() {
        let (w, h) = (7, 6);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 7 + 3) % 4).collect();
        for (min_x, min_y) in [(-9, -12), (13, 21)] {
            let img = make_img(w, h, min_x, min_y, counts.clone(),
                vec![("A".into(), PixelType::Half), ("Z".into(), PixelType::Float)]);
            let mut cum = vec![0usize];
            for &c in &counts { cum.push(cum.last().unwrap() + c as usize); }
            for mode in [LevelMode::OneLevel, LevelMode::MipmapLevels, LevelMode::RipmapLevels] {
                for comp in [Compression::NoCompression, Compression::Zips] {
                    let td = TileDescription { x_size: 2, y_size: 4, mode, rounding_mode: LevelRoundingMode::RoundDown };
                    let header = make_header(&img, comp, td, LineOrder::IncreasingY);
                    let source_counts: Vec<u8> = counts.iter().flat_map(|c| c.to_ne_bytes()).collect();
                    let (source_heap, bases) = build_sample_heap(&img);
                    let mut fb = make_frame_buffer(&img, &bases, 4, 4 * w);
                    fb.insert_sample_count_slice(Slice::new(PixelType::Uint,
                        -(min_y as i64 * 4 * w as i64 + min_x as i64 * 4),
                        4, 4 * w as usize, 1, 1, 0.0, false, false).unwrap()).unwrap();
                    let mut out = MemOStream::new();
                    let mut levels = Vec::new();
                    {
                        let mut writer = DeepTiledOutputFile::new(&mut out, &header).unwrap();
                        writer.set_frame_buffer(&fb).unwrap();
                        for ly in 0..writer.num_y_levels() {
                            for lx in 0..writer.num_x_levels() {
                                if writer.is_valid_level(lx, ly) {
                                    let (nx, ny) = (writer.num_x_tiles(lx).unwrap(), writer.num_y_tiles(ly).unwrap());
                                    writer.write_tiles(&source_counts, &source_heap, 0, nx-1, 0, ny-1, lx, ly).unwrap();
                                    levels.push((lx, ly, nx, ny));
                                }
                            }
                        }
                    }
                    let bytes = out.into_data();
                    for threads in [0, 4] {
                        let mut stream = MemIStream::new(bytes.clone());
                        let mut reader = DeepTiledInputFile::with_num_threads(&mut stream, Some(threads)).unwrap();
                        for &(lx, ly, nx, ny) in &levels {
                            for dx1 in 0..nx {
                                for dx2 in dx1..nx {
                                    for dy1 in 0..ny {
                                        for dy2 in dy1..ny {
                                            let lo = reader.data_window_for_tile(dx1, dy1, lx, ly).unwrap();
                                            let hi = reader.data_window_for_tile(dx2, dy2, lx, ly).unwrap();
                                            let rw = (hi.max.x - lo.min.x + 1) as usize;
                                            let rh = (hi.max.y - lo.min.y + 1) as usize;
                                            let mut expected_counts = Vec::new();
                                            let mut spans = Vec::new();
                                            for y in lo.min.y..=hi.max.y {
                                                for x in lo.min.x..=hi.max.x {
                                                    let p = ((y-min_y)*w + x-min_x) as usize;
                                                    expected_counts.push(counts[p]);
                                                    spans.push(cum[p]..cum[p+1]);
                                                }
                                            }
                                            let mut local_fb = DeepFrameBuffer::new();
                                            local_fb.insert_sample_count_slice(Slice::new(PixelType::Uint,
                                                -(lo.min.y as i64 * (rw*4) as i64 + lo.min.x as i64 * 4),
                                                4, rw*4, 1, 1, 0.0, false, false).unwrap()).unwrap();
                                            let mut expected = Vec::new();
                                            for (ci, (name, ty)) in img.channels.iter().enumerate() {
                                                let size = pixel_type_size(*ty).unwrap();
                                                local_fb.insert(name, DeepSlice::new(*ty, expected.len() as i64,
                                                    8, 0, size, 1, 1, 0.0, false, false).unwrap()).unwrap();
                                                for span in &spans {
                                                    for &v in &img.values[ci][span.clone()] {
                                                        match ty {
                                                            PixelType::Half => expected.extend_from_slice(&Half::from(v as f32).bits().to_ne_bytes()),
                                                            _ => expected.extend_from_slice(&(v as f32).to_ne_bytes()),
                                                        }
                                                    }
                                                }
                                            }
                                            reader.set_frame_buffer(&local_fb).unwrap();
                                            let mut sc = vec![0u8; rw*rh*4];
                                            reader.read_pixel_sample_counts(dx1, dx2, dy1, dy2, lx, ly, &mut sc).unwrap();
                                            let got: Vec<u32> = sc.as_chunks::<4>().0.iter().map(|c| u32::from_ne_bytes(*c)).collect();
                                            assert_eq!(got, expected_counts);
                                            let mut heap = vec![0u8; expected.len()];
                                            crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
                                            reader.read_tiles(dx1, dx2, dy1, dy2, lx, ly, &sc, &mut heap).unwrap();
                                            if threads > 1 && (dx1 < dx2 || dy1 < dy2) {
                                                crate::probe::assert_engaged(crate::probe::Probe::DeepTileReadParallel, threads);
                                            }
                                            assert_eq!(heap, expected, "{mode:?} {comp:?} {threads}: level ({lx},{ly}) tiles ({dx1},{dy1})..({dx2},{dy2})");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn count_and_channel_tile_flags_are_independent_for_a_non_origin_tile() {
        let counts: Vec<u32> = (0..25).map(|i| (i * 7 + 3) % 4).collect();
        let mut cumulative = vec![0usize];
        for &count in &counts {
            cumulative.push(cumulative.last().unwrap() + count as usize);
        }
        for (min_x, min_y) in [(-9, -12), (13, 21)] {
            let img = make_img(
                5,
                5,
                min_x,
                min_y,
                counts.clone(),
                vec![
                    ("A".into(), PixelType::Half),
                    ("Z".into(), PixelType::Float),
                ],
            );
            let td = TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            };
            let header = make_header(&img, Compression::Zips, td, LineOrder::IncreasingY);
            let (source_heap, bases) = build_sample_heap(&img);
            let source_counts: Vec<u8> = counts.iter().flat_map(|c| c.to_ne_bytes()).collect();
            let mut source_fb = make_frame_buffer(&img, &bases, 4, 20);
            source_fb
                .insert_sample_count_slice(
                    Slice::new(
                        PixelType::Uint,
                        -(min_y as i64 * 20 + min_x as i64 * 4),
                        4,
                        20,
                        1,
                        1,
                        0.0,
                        false,
                        false,
                    )
                    .unwrap(),
                )
                .unwrap();
            let mut out = MemOStream::new();
            {
                let mut writer = DeepTiledOutputFile::new(&mut out, &header).unwrap();
                writer.set_frame_buffer(&source_fb).unwrap();
                writer
                    .write_tiles(&source_counts, &source_heap, 0, 2, 0, 2, 0, 0)
                    .unwrap();
            }
            let bytes = out.into_data();
            for count_x in [false, true] {
                for count_y in [false, true] {
                    for channel_x in [false, true] {
                        for channel_y in [false, true] {
                            let mut stream = MemIStream::new(bytes.clone());
                            let mut reader = DeepTiledInputFile::new(&mut stream).unwrap();
                            let rect = reader.data_window_for_tile(1, 1, 0, 0).unwrap();
                            let mut fb = DeepFrameBuffer::new();
                            fb.insert_sample_count_slice(
                                Slice::new(
                                    PixelType::Uint,
                                    -(if count_x { 0 } else { rect.min.x as i64 * 4 })
                                        - (if count_y { 0 } else { rect.min.y as i64 * 8 }),
                                    4,
                                    8,
                                    1,
                                    1,
                                    0.0,
                                    count_x,
                                    count_y,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            let mut expected = Vec::new();
                            for (ci, (name, ty)) in img.channels.iter().enumerate() {
                                let size = pixel_type_size(*ty).unwrap();
                                fb.insert(
                                    name,
                                    DeepSlice::new(
                                        *ty,
                                        expected.len() as i64,
                                        8,
                                        0,
                                        size,
                                        1,
                                        1,
                                        0.0,
                                        channel_x,
                                        channel_y,
                                    )
                                    .unwrap(),
                                )
                                .unwrap();
                                for y in rect.min.y..=rect.max.y {
                                    for x in rect.min.x..=rect.max.x {
                                        let p = ((y - min_y) * 5 + x - min_x) as usize;
                                        for &value in
                                            &img.values[ci][cumulative[p]..cumulative[p + 1]]
                                        {
                                            if *ty == PixelType::Half {
                                                expected.extend_from_slice(
                                                    &Half::from(value as f32).bits().to_ne_bytes(),
                                                );
                                            } else {
                                                expected.extend_from_slice(
                                                    &(value as f32).to_ne_bytes(),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            reader.set_frame_buffer(&fb).unwrap();
                            let mut local_counts = vec![0u8; 16];
                            reader
                                .read_pixel_sample_count(1, 1, 0, 0, &mut local_counts)
                                .unwrap();
                            let mut heap = vec![0u8; expected.len()];
                            reader
                                .read_tiles(1, 1, 1, 1, 0, 0, &local_counts, &mut heap)
                                .unwrap();
                            assert_eq!(
                                heap, expected,
                                "counts({count_x},{count_y}) channel({channel_x},{channel_y})"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn multi_tile_heap_flags_preserve_aliases_fill_and_clipped_edges() {
        for repeated in [true, false] {
            let counts: Vec<u32> = (0..25)
                .map(|p| {
                    let (x, y) = (p % 5, p / 5);
                    if repeated {
                        (x % 2 + 2 * (y % 2)) % 4
                    } else {
                        (p * 7 + 3) % 4
                    }
                })
                .collect();
            let mut cumulative = vec![0usize];
            for &count in &counts {
                cumulative.push(cumulative.last().unwrap() + count as usize);
            }
            for (min_x, min_y) in [(-9, -12), (13, 21)] {
                let img = make_img(
                    5,
                    5,
                    min_x,
                    min_y,
                    counts.clone(),
                    vec![
                        ("A".into(), PixelType::Half),
                        ("Z".into(), PixelType::Float),
                    ],
                );
                let td = TileDescription {
                    x_size: 2,
                    y_size: 2,
                    mode: LevelMode::OneLevel,
                    rounding_mode: LevelRoundingMode::RoundDown,
                };
                let header = make_header(&img, Compression::Zips, td, LineOrder::IncreasingY);
                let (source_heap, bases) = build_sample_heap(&img);
                let source_counts: Vec<u8> = counts.iter().flat_map(|c| c.to_ne_bytes()).collect();
                let mut source_fb = make_frame_buffer(&img, &bases, 4, 20);
                source_fb
                    .insert_sample_count_slice(
                        Slice::new(
                            PixelType::Uint,
                            -(min_y as i64 * 20 + min_x as i64 * 4),
                            4,
                            20,
                            1,
                            1,
                            0.0,
                            false,
                            false,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let mut out = MemOStream::new();
                {
                    let mut writer = DeepTiledOutputFile::new(&mut out, &header).unwrap();
                    writer.set_frame_buffer(&source_fb).unwrap();
                    writer
                        .write_tiles(&source_counts, &source_heap, 0, 2, 0, 2, 0, 0)
                        .unwrap();
                }
                let bytes = out.into_data();
                for count_flags in 0..4 {
                    if !repeated && count_flags != 0 {
                        continue;
                    }
                    for channel_flags in 0..4 {
                        for mixed in [false, true] {
                            for first in [0, 1] {
                                for threads in [0, 4] {
                                    let mut stream = MemIStream::new(bytes.clone());
                                    let mut reader = DeepTiledInputFile::with_num_threads(
                                        &mut stream,
                                        Some(threads),
                                    )
                                    .unwrap();
                                    let lo =
                                        reader.data_window_for_tile(first, first, 0, 0).unwrap();
                                    let hi = reader.data_window_for_tile(2, 2, 0, 0).unwrap();
                                    let cw = if count_flags & 1 != 0 {
                                        2
                                    } else {
                                        hi.max.x - lo.min.x + 1
                                    } as usize;
                                    let ch = if count_flags & 2 != 0 {
                                        2
                                    } else {
                                        hi.max.y - lo.min.y + 1
                                    } as usize;
                                    let mut fb = DeepFrameBuffer::new();
                                    fb.insert_sample_count_slice(
                                        Slice::new(
                                            PixelType::Uint,
                                            -(if count_flags & 1 != 0 {
                                                0
                                            } else {
                                                lo.min.x as i64 * 4
                                            }) - (if count_flags & 2 != 0 {
                                                0
                                            } else {
                                                lo.min.y as i64 * (cw * 4) as i64
                                            }),
                                            4,
                                            cw * 4,
                                            1,
                                            1,
                                            0.0,
                                            count_flags & 1 != 0,
                                            count_flags & 2 != 0,
                                        )
                                        .unwrap(),
                                    )
                                    .unwrap();
                                    let zflags = if mixed {
                                        channel_flags ^ 3
                                    } else {
                                        channel_flags
                                    };
                                    for (name, ty, base, flags, fill) in [
                                        ("A", PixelType::Half, 0, channel_flags, 0.0),
                                        ("Z", PixelType::Float, 512, zflags, 0.0),
                                        ("missing", PixelType::Float, 1024, channel_flags, 7.0),
                                    ] {
                                        fb.insert(
                                            name,
                                            DeepSlice::new(
                                                ty,
                                                base,
                                                8,
                                                0,
                                                pixel_type_size(ty).unwrap(),
                                                1,
                                                1,
                                                fill,
                                                flags & 1 != 0,
                                                flags & 2 != 0,
                                            )
                                            .unwrap(),
                                        )
                                        .unwrap();
                                    }
                                    reader.set_frame_buffer(&fb).unwrap();
                                    let mut local_counts = vec![0u8; cw * ch * 4];
                                    reader
                                        .read_pixel_sample_counts(
                                            first,
                                            2,
                                            first,
                                            2,
                                            0,
                                            0,
                                            &mut local_counts,
                                        )
                                        .unwrap();
                                    let mut expected = vec![0xa5; 1536];
                                    // Independent direct summation, following tile write order and aliases.
                                    for ty in first..=2 {
                                        for tx in first..=2 {
                                            let tile =
                                                reader.data_window_for_tile(tx, ty, 0, 0).unwrap();
                                            for (ci, pty, base, flags) in [
                                                (0, PixelType::Half, 0usize, channel_flags),
                                                (1, PixelType::Float, 512, zflags),
                                                (2, PixelType::Float, 1024, channel_flags),
                                            ] {
                                                let x0 = if flags & 1 != 0 {
                                                    tile.min.x
                                                } else {
                                                    lo.min.x
                                                };
                                                let x1 = if flags & 1 != 0 {
                                                    tile.max.x
                                                } else {
                                                    hi.max.x
                                                };
                                                let y0 = if flags & 2 != 0 {
                                                    tile.min.y
                                                } else {
                                                    lo.min.y
                                                };
                                                for y in tile.min.y..=tile.max.y {
                                                    for x in tile.min.x..=tile.max.x {
                                                        let mut before = 0usize;
                                                        for yy in y0..y {
                                                            for xx in x0..=x1 {
                                                                before += counts[((yy - min_y) * 5
                                                                    + xx
                                                                    - min_x)
                                                                    as usize]
                                                                    as usize;
                                                            }
                                                        }
                                                        for xx in x0..x {
                                                            before += counts[((y - min_y) * 5 + xx
                                                                - min_x)
                                                                as usize]
                                                                as usize;
                                                        }
                                                        let p =
                                                            ((y - min_y) * 5 + x - min_x) as usize;
                                                        let size = pixel_type_size(pty).unwrap();
                                                        for s in 0..counts[p] as usize {
                                                            let value = if ci == 2 {
                                                                7.0
                                                            } else {
                                                                img.values[ci][cumulative[p] + s]
                                                                    as f32
                                                            };
                                                            let at = base + (before + s) * size;
                                                            if pty == PixelType::Half {
                                                                expected[at..at + size]
                                                                    .copy_from_slice(
                                                                        &Half::from(value)
                                                                            .bits()
                                                                            .to_ne_bytes(),
                                                                    );
                                                            } else {
                                                                expected[at..at + size]
                                                                    .copy_from_slice(
                                                                        &value.to_ne_bytes(),
                                                                    );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    let mut heap = vec![0xa5; 1536];
                                    crate::probe::reset(crate::probe::Probe::DeepTileReadParallel);
                                    reader
                                        .read_tiles(
                                            first,
                                            2,
                                            first,
                                            2,
                                            0,
                                            0,
                                            &local_counts,
                                            &mut heap,
                                        )
                                        .unwrap();
                                    assert_eq!(
                                        heap, expected,
                                        "repeat={repeated} counts={count_flags} channel={channel_flags} mixed={mixed} first={first} threads={threads}"
                                    );
                                    if mixed || channel_flags != 0 {
                                        assert_eq!(
                                            crate::probe::get(
                                                crate::probe::Probe::DeepTileReadParallel
                                            ),
                                            0
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
