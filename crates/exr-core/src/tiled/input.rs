//! Port of OpenEXR `ImfTiledInputFile.cpp` — `class TiledInputFile`, the
//! single-part tiled reader.  It is the exact inverse of this crate's
//! [`TiledOutputFile`][crate::tiled::TiledOutputFile] writer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledInputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledInputFile.h>
//!
//! # A note on the port target
//!
//! In OpenEXR v3.3.11 `TiledInputFile` is a thin C++ wrapper over the
//! OpenEXRCore C API (`exr_read_tile_chunk_info`, `exr_decoding_run`, …) — the
//! manual offset-table read, chunk seek and scatter live deep inside the core
//! library.  A literal token-for-token port of the v3.3.11 `.cpp` would have to
//! port the entire C core first, which is out of scope here.
//!
//! The **on-disk byte format** consumed here is fixed across OpenEXR versions
//! and is exactly the format produced by this crate's
//! [`TiledOutputFile`][crate::tiled::TiledOutputFile].  The algorithm ported
//! here therefore mirrors the *legacy* IlmImf `TiledInputFile`
//! (`readTiles` → `readTileData` → `TileBufferTask` scatter), i.e. the inverse
//! of our writer — the canonical description of that byte format:
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.1.0/src/lib/OpenEXR/ImfTiledInputFile.cpp>
//!
//! The v3.3.11 semantics that *are* observable are matched exactly:
//! * The constructor validates magic + version (the `TILED` flag must be set),
//!   reads the header, sanity-checks it as a tiled header (which requires a tile
//!   description), computes the level / tile counts, and reads the chunk-offset
//!   table — reconstructing it from the file when it is zero/corrupt
//!   ([`TileOffsets::read_from`]).
//! * [`TiledInputFile::set_frame_buffer`] performs the x/y-sampling
//!   compatibility check (tiled channels must have sampling `(1,1)`) and records
//!   a *fill list* for channels present in the frame buffer but absent from the
//!   file (`setFrameBuffer`).
//! * [`TiledInputFile::read_tile`] / [`TiledInputFile::read_tiles`] validate the
//!   level/tile, seek to the tile's chunk via the offset table, read the chunk
//!   header (`tileX, tileY, levelX, levelY, dataSize`), **verify those four
//!   coordinates match the request** (the C++ `readTileData` throws on
//!   mismatch — ported here), read `dataSize` bytes, decompress (or take the raw
//!   bytes for `NONE` / the codec "no-shrink → store raw" case), and scatter the
//!   tile's pixels into the user frame buffer over
//!   [`crate::tiled::misc::data_window_for_tile`].
//!
//! # What this module intentionally does NOT cover
//!
//! * **Deep** tiled input — that is [`crate::deep::tiled_input`].
//! * `rawTileData`'s sequence-cursor form (C++ uses it inside `copyPixels`;
//!   this port's `copy_pixels` asks for each tile by coordinates instead); the
//!   exact-coordinate form is [`TiledInputFile::raw_tile_data`].

use std::marker::PhantomData;
use std::ops::DerefMut;

use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
use crate::attr::PixelType;
use crate::compressor::{default_format, new_tile_compressor, Compressor, Format};
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::frame_buffer::FrameBuffer;
use crate::header::{self, Header};
use crate::io::IStream;
use crate::multipart::input::{read_prologue, Storage};
use crate::misc::{copy_into_frame_buffer, pixel_type_size, BufferFormat};
use crate::tiled::misc::{
    calculate_bytes_per_pixel, data_window_for_level, data_window_for_tile, level_size,
    precalculate_tile_info,
};
use crate::tiled::tile_offsets::TileOffsets;

use imath_rs::Box2i;

/// Per-tile `(dx, dy, lx, ly)` coordinate lists, one entry per tile — the
/// return type of [`TiledInputFile::tile_order`].
pub type TileOrder = (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>);

// ---------------------------------------------------------------------------
// TInSliceInfo — port of the anonymous-namespace `struct TInSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel read descriptor — port of `TInSliceInfo`
/// (legacy `ImfTiledInputFile.cpp`).
///
/// Built by [`TiledInputFile::set_frame_buffer`] from the file's channel list
/// and the user frame buffer.  Mirrors the C++ struct: the frame-buffer pixel
/// type (`type_fb`), the file's stored pixel type (`type_file`), strides, the
/// tile-coordinate flags, the fill flag/value, and a `skip` flag for file
/// channels with no destination slice.  The raw `char* base` becomes a signed
/// byte offset (`base_offset`) into the caller-owned output buffer — the same
/// safe-Rust adaptation used throughout this crate (see [`crate::frame_buffer::Slice`]).
#[derive(Clone, Debug)]
struct TInSliceInfo {
    /// Pixel type requested in the frame-buffer slice (the scatter target).
    type_fb: PixelType,
    /// Pixel type stored on disk for this channel (the scatter source).
    ///
    /// Equals `type_fb` for a `fill` slice (no file channel exists).
    type_file: PixelType,
    /// Signed byte offset that replaces the C++ `char* base` (origin baked in).
    base_offset: i64,
    /// Distance in bytes between horizontally adjacent samples (`slice.xStride`).
    x_stride: usize,
    /// Distance in bytes between vertically adjacent rows (`slice.yStride`).
    y_stride: usize,
    /// `slice.xTileCoords` — addresses are relative to the tile's min x.
    x_tile_coords: bool,
    /// `slice.yTileCoords` — addresses are relative to the tile's min y.
    y_tile_coords: bool,
    /// `true` when the file has no channel for this slice — the slice is filled
    /// with [`fill_value`](Self::fill_value) instead of read from the file.
    fill: bool,
    /// Value written into a `fill` slice (`slice.fillValue`).
    fill_value: f64,
    /// `true` when this descriptor exists only to *skip* the file channel's
    /// bytes (the channel is in the file but not in the frame buffer); no
    /// scatter is performed but the read cursor must still advance.
    skip: bool,
}

// ---------------------------------------------------------------------------
// TiledInputFile — port of `class TiledInputFile`
// ---------------------------------------------------------------------------

/// Single-part tiled EXR reader — port of `Imf::TiledInputFile`.
///
/// Construct with [`TiledInputFile::new`] (validates magic + version, reads the
/// header and the tile-offset table), describe the destination pixels with
/// [`TiledInputFile::set_frame_buffer`], then pull tiles with
/// [`TiledInputFile::read_tile`] / [`TiledInputFile::read_tiles`].
///
/// The reader borrows the caller's [`IStream`] for its whole lifetime.  Its
/// structure mirrors the [`TiledOutputFile`][crate::tiled::TiledOutputFile]
/// writer field-for-field so the read path is the exact inverse of the write
/// path.
pub struct TiledInputFile<'a, S = &'a mut dyn IStream> {
    /// The input stream, addressed through `DerefMut<Target = dyn IStream>`
    /// (the C++ `_streamData->is`) -- generic over the stream HOLDER, mirroring
    /// `multipart::MultiPartInputFile<'a, S>` / `scanline::OutputFile<'a,
    /// S>`, so [`TiledInputFile::open`] can OWN a path-opened stream — C++'s
    /// real, public `TiledInputFile(const char*, int)` constructor
    /// (`ImfTiledInputFile.h:42-43`), which this crate could not provide
    /// before.
    stream: S,

    /// Everything else this reader owns: factored into its own,
    /// non-generic, stream-free struct so `crate::multipart::MultiPartInputFile`
    /// can persist it across separate part-reader accessor calls instead of
    /// rebuilding a fresh `TiledInputFile` every time (v3.4.15
    /// `ImfMultiPartInputFile.cpp::getInputPart<T>` caches one reader object
    /// per part -- byte-identical to v3.3.11). See
    /// [`TiledInputFileState`].
    state: TiledInputFileState,

    /// Ties `'a` to the struct even when `S` (e.g. `Box<dyn IStream + 'a>`)
    /// is the only field that mentions it -- same purpose as
    /// `multipart::MultiPartInputFile`'s own `_stream_lt` marker; without
    /// it, a generic `S` with no OTHER field referencing `'a` leaves the
    /// lifetime parameter structurally unused (`E0392`).
    _stream_lt: PhantomData<&'a mut ()>,
}

/// A [`TiledInputFile`] that **owns** its input stream (`Box<dyn IStream>`)
/// -- the read-side analogue of
/// [`scanline::OutputFileOwned`](crate::scanline::OutputFileOwned) /
/// [`multipart::MultiPartInputFileOwned`](crate::multipart::MultiPartInputFileOwned).
/// Build it with [`TiledInputFile::open`]; behaves identically to the
/// borrowed default for the same input.
pub type TiledInputFileOwned<'a> = TiledInputFile<'a, Box<dyn IStream + 'a>>;

/// Everything a [`TiledInputFile`] owns except its stream -- the read-state
/// object the `MultiPartInputFile` part-reader cache persists per
/// tiled part. Has NO lifetime of its own: every
/// method that needs to read takes `stream: &mut dyn IStream` explicitly
/// (mirrors `crate::scanline::input::ScanLineInputFileState`).
pub(crate) struct TiledInputFileState {
    /// The image header (C++ `_data->header`).
    header: Header,

    /// Raw version word read from the file (flags inspected at construction).
    version: i32,

    /// Tile layout copied from the header — C++ `_data->tileDesc`.
    tile_desc: TileDescription,

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
    /// `true` when the on-disk offset table was complete (no reconstruction was
    /// needed) — C++ `_data->fileIsComplete`.  Exposed via [`is_complete`].
    file_is_complete: bool,

    /// Compressor data format (Xdr for every ported codec) — C++ `_data->format`.
    format: Format,

    /// Per-channel read descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][TiledInputFile::set_frame_buffer] is called).
    slices: Vec<TInSliceInfo>,

    /// The current frame buffer — C++ `_data->frameBuffer`.
    frame_buffer: FrameBuffer,

    /// The per-tile decompressor, or `None` for `NO_COMPRESSION` (the raw path)
    /// — mirrors the C++ `TileBuffer::compressor` built by `newTileCompressor`.
    compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this reader drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `TiledInputFile::Data::{multiPartBackwardSupport, partNumber}`.
    ///
    /// The same seam as the scanline reader: when `Some`, each tile chunk's
    /// leading `partNumber` (i32 LE) is read and verified against this part
    /// (C++ `extract_chunk_leader`); when `None` there is no such prefix, so
    /// single-part reads are byte-identical to the pre-multipart reader.
    part_number: Option<i32>,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the scanline reader's identical
    /// field (`scanline/input.rs`).
    num_threads: crate::threading::ThreadCount,

    /// The codec (C++ `_data->_compression`) — kept so the parallel path can
    /// build a fresh per-worker codec instance via [`new_tile_compressor`]
    /// (each worker owns one; codecs are not `Sync`).
    compression: crate::attr::Compression,

    /// `maxBytesPerTileLine` (bytes-per-pixel * nominal tile width) — reused
    /// by the parallel path to size fresh per-worker codec instances.
    max_bytes_per_tile_line: usize,
}

impl<'a> TiledInputFile<'a, &'a mut dyn IStream> {
    /// Open a `TiledInputFile` on `stream` — port of `TiledInputFile`'s
    /// constructor + `Data::initialize`, restricted to the single-part tiled
    /// read path.
    ///
    /// Reads and validates the magic number + version word (the `TILED` flag
    /// must be set), then the header, then the tile-offset table (reconstructing
    /// it from the file when the stored table is zero/corrupt).
    ///
    /// # Errors
    /// * Propagates [`Header::read_from`] failures (bad magic, unknown flags).
    /// * Returns [`ExrError::Arg`] when the file is not a single-part tiled
    ///   image (multipart / deep / scanline — tracked, never mis-read).
    /// * Propagates [`Header::sanity_check`] failures (incl. a missing tile
    ///   description — `sanity_check(true, …)` requires one).
    /// * Returns the [`new_tile_compressor`] error for any compression not yet
    ///   ported (nothing is faked).
    /// * [`ExrError::Arg`] if the tile size exceeds the OpenEXR `INT_MAX` byte
    ///   limit, matching the C++ throw.
    /// * Propagates stream read / position errors.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::with_num_threads(stream, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(stream: &'a mut dyn IStream, num_threads: Option<i32>) -> Result<Self> {
        // ONE prologue reader for the whole crate: magic + version word + the
        // header + the chunk offset table (with reconstruction).  C++ likewise
        // has a single `internal_exr_parse_header` behind every reader class.
        //
        // `auto_add_type = false`, matching
        // C++ `TiledInputFile::header()` (v3.4.15 `ImfTiledInputFile.cpp:218-230`),
        // which returns the context header WITHOUT an invented `type`
        // attribute (mirrors `scanline::InputFile::new`, `input_file.rs:91`).
        // Safe: for a plain single-part file `Storage` is derived purely from
        // the version-word flags, never from the `type` string
        // (`multipart::input::validate_read_part`), so this does not change
        // which storage this constructor accepts.
        let prologue = read_prologue(stream, false)?;
        let version_word = prologue.version;
        let multipart = prologue.multipart;
        let part = prologue
            .parts
            .into_iter()
            .next()
            .ok_or_else(|| ExrError::Input("File contains no parts.".into()))?;

        // C++ `Data::initialize`: the part must be a flat tiled part.
        match part.storage {
            Storage::Tiled => {}
            Storage::DeepScanline | Storage::DeepTiled => {
                return Err(ExrError::Arg(
                    "Deep files are not supported by the tiled reader (use DeepTiledInputFile)."
                        .into(),
                ));
            }
            _ => {
                return Err(ExrError::Arg(
                    "File is not a tiled file (the tiled flag is not set in the version word)."
                        .into(),
                ));
            }
        }

        // C++ `TiledInputFile(const char*)`
        // (v3.4.15 `ImfTiledInputFile.cpp:87-91`) opens a multipart file's
        // part 0 as long as its storage is EXR_STORAGE_TILED — it does not
        // reject multipart files outright. A multipart file's chunks carry a
        // leading `partNumber`; part 0 is C++'s `Data::kDefaultPart`
        // (mirrors `scanline::InputFile::new`, `input_file.rs:98`).
        let part_number = if multipart { Some(0) } else { None };

        // The prologue already computed the
        // correct completeness verdict for this part
        // (`multipart::input::read_prologue`'s `table_complete` after
        // reconstruction, matching C++ `exr_validate_chunk_table` /
        // `OpenEXRCore/part.c:471-497`) — every offset in
        // `[end-of-table, file_size)`. `from_part` cannot see that (it only
        // gets the flat offsets and recomputes a weaker `!= 0` check via
        // `TileOffsets::read_from_vec`/`any_offsets_are_invalid`), so `new()`
        // overrides the field with the trustworthy verdict after construction.
        let complete = part.complete;
        let mut file = Self::from_part_mt(
            stream,
            part.header,
            version_word,
            part_number,
            part.chunk_offsets,
            num_threads,
        )?;
        file.state.file_is_complete = complete;
        Ok(file)
    }

}

// Owned-stream constructor (`S = Box<dyn IStream + 'a>`).
impl<'a> TiledInputFile<'a, Box<dyn IStream + 'a>> {
    /// Open `path`, OWNING the decoded stream — port of C++'s real, public
    /// `TiledInputFile(const char filename[], int numThreads)` constructor
    /// (`ImfTiledInputFile.h:42-43`), previously impossible to provide here
    /// because `TiledInputFile` had no owned-stream variant.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display()))
        })?;
        let stream: Box<dyn IStream + 'a> = Box::new(crate::io::StdIfStream::open(path_str)?);
        Self::new_owned(stream)
    }

    /// Stream form of [`open`](Self::open): open a [`TiledInputFileOwned`]
    /// that owns the given boxed stream directly. Mirrors the borrowed
    /// `new`'s body exactly (prologue -> storage check -> `from_part`, now
    /// fully generic over `S`), just built for the owned stream holder.
    pub fn new_owned(mut stream: Box<dyn IStream + 'a>) -> Result<Self> {
        let prologue = read_prologue(&mut *stream, false)?;
        let version_word = prologue.version;
        let multipart = prologue.multipart;
        let part = prologue
            .parts
            .into_iter()
            .next()
            .ok_or_else(|| ExrError::Input("File contains no parts.".into()))?;

        match part.storage {
            Storage::Tiled => {}
            Storage::DeepScanline | Storage::DeepTiled => {
                return Err(ExrError::Arg(
                    "Deep files are not supported by the tiled reader (use DeepTiledInputFile)."
                        .into(),
                ));
            }
            _ => {
                return Err(ExrError::Arg(
                    "File is not a tiled file (the tiled flag is not set in the version word)."
                        .into(),
                ));
            }
        }

        let part_number = if multipart { Some(0) } else { None };
        let complete = part.complete;
        let mut file =
            Self::from_part(stream, part.header, version_word, part_number, part.chunk_offsets)?;
        file.state.file_is_complete = complete;
        Ok(file)
    }
}

// Shared read path, generic over the stream holder `S`. Covers BOTH the
// borrowed default (`S = &'a mut dyn IStream`) and the owned `Box<dyn
// IStream>` variant — every method below addresses the stream through
// `DerefMut<Target = dyn IStream>`, so the two paths are a single source of
// truth.
impl<'a, S> TiledInputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Open a `TiledInputFile` for **one part of a multipart file** — port of
    /// `TiledInputFile::TiledInputFile(InputPartData*)`.
    ///
    /// The enclosing [`MultiPartInputFile`](crate::multipart::MultiPartInputFile)
    /// has already read the file's magic + version word, every part header, and
    /// every part's chunk offset table.  This constructor takes the
    /// already-parsed `header`, the file `version` word, the part's
    /// `chunk_offsets` (flat, in `[level][dy][dx]` slot order), and the part's
    /// `part_number`, and rebuilds the per-part read state without touching the
    /// stream prologue.  Each tile chunk's leading `partNumber` is verified
    /// against `part_number`. Pure w.r.t. the stream (no I/O at all — just
    /// stores it), so this needs no borrowed/owned split, unlike `new`/`open`.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledInputFile.cpp> (`TiledInputFile(InputPartData*)`) + `OpenEXRCore/chunk.c`.
    pub fn from_part(
        stream: S,
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
        stream: S,
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = TiledInputFileState::from_part_mt(
            header,
            version,
            part_number,
            chunk_offsets,
            num_threads,
        )?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }

    /// Reassemble a `TiledInputFile` view from an already-owned
    /// [`TiledInputFileState`] and a freshly (re)borrowed stream — the
    /// tiled-as-scanlines counterpart of [`from_part`](Self::from_part) for
    /// callers that persist the state SEPARATELY from the stream:
    /// `scanline::input_file::TiledReader` (the tiled-as-scanlines
    /// wrapper `InputFile::as_tiled_input` hands out a `TiledInputFile` view
    /// of) needs to keep its own extra tile-row cache fields alongside a
    /// PERSISTED `TiledInputFileState` across `MultiPartInputFile::input_part`
    /// calls, the exact same ownership-inversion `MultiPartOutputFile`
    /// already uses on the write side. This is a plain, no-op struct literal
    /// (no parsing, no validation, no stream I/O) — the state was already
    /// fully built by an earlier `from_part` call.
    #[allow(dead_code)]
    pub(crate) fn from_borrowed_state(stream: S, state: TiledInputFileState) -> Self {
        Self { stream, state, _stream_lt: PhantomData }
    }

    /// Consume this view, handing back the owned state so a caller that
    /// persists it separately from the stream can store it again — the
    /// inverse of [`from_borrowed_state`](Self::from_borrowed_state).
    #[allow(dead_code)]
    pub(crate) fn into_state(self) -> TiledInputFileState {
        self.state
    }

    // -----------------------------------------------------------------------
    // Accessors — thin delegators to TiledInputFileState
    // -----------------------------------------------------------------------

    /// The image header — port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// Name of the underlying file or stream — port of `const char*
    /// fileName() const` (C++
    /// `ImfTiledInputFile.cpp:212-215`, `_ctxt.fileName()`).
    pub fn file_name(&self) -> &str {
        self.stream.file_name()
    }

    /// The order every tile appears in on disk, sorted by file offset — port
    /// of `void tileOrder(int dx[], int dy[], int lx[], int ly[]) const`
    /// See [`TiledInputFileState::tile_order`].
    pub fn tile_order(&self) -> Result<TileOrder> {
        self.state.tile_order()
    }

    /// The raw file-format version word — port of `int version() const`.
    pub fn version(&self) -> i32 {
        self.state.version()
    }

    /// The current frame buffer — port of `const FrameBuffer& frameBuffer() const`.
    pub fn frame_buffer(&self) -> &FrameBuffer {
        self.state.frame_buffer()
    }

    /// Whether every tile is present in the file — port of `bool isComplete() const`.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Compression type — convenience accessor (C++ reads it off `header()`).
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
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

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode() const`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
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

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(int, int) const`.
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

    /// Set the pixel destination — port of `TiledInputFile::setFrameBuffer`.
    /// See [`TiledInputFileState::set_frame_buffer`].
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Read the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)` into `out` — port of
    /// `TiledInputFile::readTiles(dx1,dx2,dy1,dy2,lx,ly)`. See
    /// [`TiledInputFileState::read_tiles`].
    #[allow(clippy::too_many_arguments)]
    pub fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.state.read_tiles(&mut *self.stream, dx1, dx2, dy1, dy2, lx, ly, out)
    }

    /// Read the rectangle of tiles at MIP level `l` — port of
    /// `TiledInputFile::readTiles(dx1,dx2,dy1,dy2,l)`, forwarding to `(…, l, l)`.
    pub fn read_tiles_l(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.state.read_tiles_l(&mut *self.stream, dx1, dx2, dy1, dy2, l, out)
    }

    /// Read a single tile `(dx, dy, lx, ly)` — port of
    /// `TiledInputFile::readTile(int dx, int dy, int lx, int ly)`, which
    /// forwards to `readTiles(dx, dx, dy, dy, lx, ly)`.
    pub fn read_tile(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, out: &mut [u8]) -> Result<()> {
        self.state.read_tile(&mut *self.stream, dx, dy, lx, ly, out)
    }

    /// Read a single tile `(dx, dy, l)` at MIP level `l` — port of
    /// `TiledInputFile::readTile(int dx, int dy, int l)`, forwarding to
    /// `readTile(dx, dy, l, l)`.
    pub fn read_tile_l(&mut self, dx: i32, dy: i32, l: i32, out: &mut [u8]) -> Result<()> {
        self.state.read_tile_l(&mut *self.stream, dx, dy, l, out)
    }

    /// Read tile `(dx, dy, lx, ly)`'s raw on-disk bytes — packed
    /// (compressed, or already-raw for `NONE`/no-shrink) — without
    /// decompressing or scattering into a frame buffer. Port of `void
    /// rawTileData(int& dx, int& dy, int& lx, int& ly, const char*&
    /// pixelData, int& pixelDataSize)` (`ImfTiledInputFile.cpp:345-396`). See
    /// [`TiledInputFileState::raw_tile_data`] for the in/out-reference
    /// contract this signature preserves.
    pub fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        self.state.raw_tile_data(&mut *self.stream, dx, dy, lx, ly)
    }
}

/// A borrowing view over a tiled input part's PERSISTED read state -- the
/// tiled-part analogue of `crate::scanline::input::ScanLineInputPartReader`.
/// `crate::multipart::MultiPartInputFile`'s part-reader cache returns this
/// instead of rebuilding a fresh [`TiledInputFile`] on every
/// `tiled_input_part` call, the read-side counterpart of
/// [`crate::tiled::TiledOutputPartWriter`] -- v3.4.15's
/// `ImfMultiPartInputFile.cpp::getInputPart<T>` caches one reader object per
/// part (byte-identical to v3.3.11). Exposes the identical
/// public method set as [`TiledInputFile`] by delegating to the same
/// [`TiledInputFileState`] methods, driven by the SAME
/// `stream: &mut dyn IStream` reborrowed from the container each call.
///
/// `scanline::input_file::TiledReaderState` (the tiled-as-scanlines
/// reader) calls `TiledInputFileState`'s own `pub(crate)` methods directly
/// instead of through this view — see that struct's docs. Exercised directly by
/// `tests::tiled_input_part_reader_reuses_persisted_state_across_interleaved_access`
/// below, proving the state/view split resumes interleaved reads correctly.
///
/// `pub` (not `pub(crate)`): `multipart::input::MultiPartInputFile
/// ::tiled_input_part` returns this type directly, so it must be nameable
/// outside the crate the same way `TiledInputFile` itself is.
pub struct TiledInputPartReader<'p> {
    stream: &'p mut dyn IStream,
    state: &'p mut TiledInputFileState,
}

impl<'p> TiledInputPartReader<'p> {
    /// Construct a view over a container-owned stream + persisted state.
    pub(crate) fn new(stream: &'p mut dyn IStream, state: &'p mut TiledInputFileState) -> Self {
        Self { stream, state }
    }

    /// The image header.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The raw file-format version word.
    pub fn version(&self) -> i32 {
        self.state.version()
    }

    /// The current frame buffer.
    pub fn frame_buffer(&self) -> &FrameBuffer {
        self.state.frame_buffer()
    }

    /// Whether every tile is present in the file.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Set the pixel destination.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Read the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level `(lx, ly)`.
    #[allow(clippy::too_many_arguments)]
    pub fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.state.read_tiles(self.stream, dx1, dx2, dy1, dy2, lx, ly, out)
    }

    /// Read a single tile `(dx, dy, lx, ly)`.
    pub fn read_tile(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.state.read_tile(self.stream, dx, dy, lx, ly, out)
    }

    /// Read tile `(dx, dy, lx, ly)`'s raw on-disk bytes.
    pub fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        self.state.raw_tile_data(self.stream, dx, dy, lx, ly)
    }

    /// Tile height in pixels (level 0's tile description `y_size`) — needed
    /// by `scanline::input_file::TiledReader`'s tiled-as-scanlines row math.
    pub fn tile_y_size(&self) -> u32 {
        self.state.tile_y_size()
    }

    /// Number of tiles along X at level `lx`.
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }

    /// Number of tiles along Y at level `ly`.
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.state.num_y_tiles(ly)
    }

    /// The data window covered by tile `(dx, dy)` at level `(lx, ly)`.
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_tile(dx, dy, lx, ly)
    }

    /// Pixel width of level `lx` (level 0 = full width).
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.state.level_width(lx)
    }

    /// This part's `TiledCopySource` support — mirrors `TiledInputFile`'s own
    /// `tile_order`, needed by `multipart::part::TiledInputPart`'s
    /// `TiledCopySource` impl once it wraps this view instead of an owned
    /// `TiledInputFile`.
    pub fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        self.state.tile_order()
    }

    /// This view is always already tiled storage — the `TiledCopySource`
    /// counterpart of `scanline::InputFilePartReader::ensure_tiled` (which
    /// has to check, since it may hold a scanline part instead).
    pub fn ensure_tiled(&mut self) -> Result<()> {
        Ok(())
    }

    /// Number of resolution levels along X — needed by
    /// `tests/corpus.rs`'s `decode_flat_tiled_part`, generic over any tiled
    /// reader.
    pub fn num_x_levels(&self) -> i32 {
        self.state.num_x_levels()
    }

    /// Number of resolution levels along Y.
    pub fn num_y_levels(&self) -> i32 {
        self.state.num_y_levels()
    }

    /// Whether level `(lx, ly)` exists for this file's level mode.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.state.is_valid_level(lx, ly)
    }

    /// The data window covered by level `(lx, ly)`.
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_level(lx, ly)
    }
}

/// Shared surface a level-0 tiled reader needs to expose for
/// `api::Image::from_tiled_input_file` — the tiled counterpart of
/// [`crate::scanline::GenericInputReader`]. Lets that one facade function
/// drive either a standalone [`TiledInputFile`] or a multipart-cached
/// [`TiledInputPartReader`] (the container's `tiled_input_part` returns
/// the latter).
pub trait GenericTiledReader {
    fn header(&self) -> &Header;
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()>;
    fn num_x_tiles(&self, lx: i32) -> Result<i32>;
    fn num_y_tiles(&self, ly: i32) -> Result<i32>;
    #[allow(clippy::too_many_arguments)]
    fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()>;
}

impl<'a> GenericTiledReader for TiledInputFile<'a> {
    fn header(&self) -> &Header {
        TiledInputFile::header(self)
    }
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        TiledInputFile::set_frame_buffer(self, frame_buffer)
    }
    fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        TiledInputFile::num_x_tiles(self, lx)
    }
    fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        TiledInputFile::num_y_tiles(self, ly)
    }
    fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        TiledInputFile::read_tiles(self, dx1, dx2, dy1, dy2, lx, ly, out)
    }
}

impl<'p> GenericTiledReader for TiledInputPartReader<'p> {
    fn header(&self) -> &Header {
        TiledInputPartReader::header(self)
    }
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        TiledInputPartReader::set_frame_buffer(self, frame_buffer)
    }
    fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        TiledInputPartReader::num_x_tiles(self, lx)
    }
    fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        TiledInputPartReader::num_y_tiles(self, ly)
    }
    fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        TiledInputPartReader::read_tiles(self, dx1, dx2, dy1, dy2, lx, ly, out)
    }
}

impl TiledInputFileState {
    /// Build the read state for **one part of a multipart file** (or the
    /// single-part path, via [`TiledInputFile::new`]) — port of
    /// `TiledInputFile::TiledInputFile(InputPartData*)`. Pure (no stream
    /// access): the enclosing prologue reader has already parsed `header`,
    /// `version`, and `chunk_offsets`.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledInputFile.cpp> (`TiledInputFile(InputPartData*)`) + `OpenEXRCore/chunk.c`.
    ///
    /// `pub(crate)` (not private): `multipart::input::MultiPartInputFile` and
    /// `scanline::input_file::TiledReaderState` (the tiled-as-scanlines
    /// reader) both build/cache this state directly, the same reason
    /// `TiledOutputFileState::from_part` is `pub` on the write side.
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default).
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // The part header must describe a (flat) tiled image; re-check here so
        // this path never mis-reads a scanline or deep part as tiled.
        if header.has_type() {
            let t = header.type_()?;
            if t != header::TILED_IMAGE {
                return Err(ExrError::Arg(format!(
                    "Cannot read \"{t}\" part with the tiled reader \
                     (only \"{}\" is supported).",
                    header::TILED_IMAGE
                )));
            }
        }

        // C++: header.sanityCheck(true);  (tiled — requires a tile description).
        header.sanity_check_at_open(true, false)?;

        let tile_desc = header.tile_description()?;
        let data_window = header.data_window()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
            precalculate_tile_info(&tile_desc, min_x, max_x, min_y, max_y)?;

        let max_bytes_per_tile_line =
            calculate_bytes_per_pixel(&header)?.saturating_mul(tile_desc.x_size as usize);
        let tile_buffer_size = max_bytes_per_tile_line.saturating_mul(tile_desc.y_size as usize);
        // C++ 3.4.15 rejects a nominal tile
        // whose x_size/y_size exceeds INT_MAX/4
        // (`OpenEXRCore/validation.c:593-603`); legacy `ImfHeader.cpp:1057-1058`
        // (what this crate ported) only checked `> INT_MAX`, letting a header
        // like `tiles = 600_000_000 x 1` through. Match the 3.4.15 oracle bound.
        if tile_buffer_size > (i32::MAX as usize) / 4 {
            return Err(ExrError::Arg("Invalid tile description size".into()));
        }

        let compression = header.compression()?;
        let compressor = new_tile_compressor(
            compression,
            max_bytes_per_tile_line,
            tile_desc.y_size as usize,
            &header,
        )?;
        let format = default_format(compressor.as_deref());

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

        // Never allocate the nominal `tile_buffer_size` up front — it is
        // driven by the header's `tiles` attribute alone (before any chunk is
        // read) and can be gigabytes for a tiny data window with a huge
        // nominal tile (e.g. `tiles = 32768 x 32767` over a 1x1 window). C++
        // never allocates more than the CLIPPED tile it is about to decode
        // (`OpenEXRCore/chunk.c:1181-1224`); `fetch_tile`
        // allocates each tile's own scratch buffer sized to exactly its
        // (clipped) `dataSize`, so there is no persisted scratch field at all.

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
            format,
            slices: Vec::new(),
            frame_buffer: FrameBuffer::new(),
            compressor,
            part_number,
            num_threads,
            compression,
            max_bytes_per_tile_line,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors — thin wrappers (mirror the tiled writer + scanline reader)
    // -----------------------------------------------------------------------
    //
    // `header`/`version`/`tile_y_size`/`level_width`/`num_x_tiles`/
    // `data_window_for_tile`/`set_frame_buffer` are `pub(crate)` (not just
    // reachable via `TiledInputPartReader`) so `scanline::input_file`'s
    // `TiledReaderState` (the tiled-as-scanlines reader)
    // can hold a `TiledInputFileState` field and call these DIRECTLY:
    // several return `&Self`-lifetime references (`header`, `frame_buffer`)
    // that a transient `TiledInputPartReader` view (whose own `&self`
    // lifetime is just the temporary) cannot hand back to a caller that
    // needs the reference to outlive the call.

    /// The image header — port of `const Header& header() const`.
    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    /// The order every tile appears in on disk, sorted by file offset — port
    /// of `void tileOrder(int dx[], int dy[], int lx[], int ly[]) const`
    /// Returns `(dx, dy, lx, ly)`, one entry per
    /// tile, in ascending file-offset order — the legacy `TileOffsets`
    /// algorithm this crate already ports for `RANDOM_Y`
    /// ([`TileOffsets::get_tile_order`]) applied unconditionally.
    ///
    /// Note: C++ 3.4.15 (`ImfTiledInputFile.cpp:629-651`) special-cases
    /// `INCREASING_Y`/`DECREASING_Y` to fill only `dx[0]`/`dy[0]`/`lx[0]`/
    /// `ly[0]` (the first tile) and return immediately, leaving the rest of
    /// the caller's arrays untouched — a quirk of a `TODO: remove once
    /// TiledOutputFile copy is converted` code path, not a documented
    /// contract. This port always returns the FULL tile list sorted by file
    /// offset (which for `INCREASING_Y`/`DECREASING_Y` is simply line order,
    /// since that is the order those line orders write tiles in), the more
    /// useful and unsurprising behaviour, and the one this crate's own
    /// `TileOffsets::get_tile_order` already implements and tests.
    pub(crate) fn tile_order(&self) -> Result<TileOrder> {
        let n = self.tile_offsets.slot_count();
        let mut dx = vec![0i32; n];
        let mut dy = vec![0i32; n];
        let mut lx = vec![0i32; n];
        let mut ly = vec![0i32; n];
        self.tile_offsets.get_tile_order(&mut dx, &mut dy, &mut lx, &mut ly)?;
        Ok((dx, dy, lx, ly))
    }

    /// The raw file-format version word — port of `int version() const`.
    pub(crate) fn version(&self) -> i32 {
        self.version
    }

    /// The current frame buffer — port of `const FrameBuffer& frameBuffer() const`.
    pub(crate) fn frame_buffer(&self) -> &FrameBuffer {
        &self.frame_buffer
    }

    /// Whether every tile is present in the file — port of `bool isComplete() const`.
    ///
    /// `false` when the on-disk offset table had to be reconstructed because it
    /// was zero/corrupt (an incomplete write), mirroring C++ `_data->fileIsComplete`.
    pub(crate) fn is_complete(&self) -> bool {
        self.file_is_complete
    }

    /// Compression type — convenience accessor (C++ reads it off `header()`).
    pub(crate) fn compression(&self) -> Result<crate::attr::Compression> {
        self.header.compression()
    }

    /// Tile width — port of `unsigned int tileXSize() const`.
    pub(crate) fn tile_x_size(&self) -> u32 {
        self.tile_desc.x_size
    }

    /// Tile height — port of `unsigned int tileYSize() const`.
    pub(crate) fn tile_y_size(&self) -> u32 {
        self.tile_desc.y_size
    }

    /// Level mode — port of `LevelMode levelMode() const`.
    pub(crate) fn level_mode(&self) -> LevelMode {
        self.tile_desc.mode
    }

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode() const`.
    pub(crate) fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.tile_desc.rounding_mode
    }

    /// Number of levels in X — port of `int numXLevels() const`.
    pub(crate) fn num_x_levels(&self) -> i32 {
        self.num_x_levels
    }

    /// Number of levels in Y — port of `int numYLevels() const`.
    pub(crate) fn num_y_levels(&self) -> i32 {
        self.num_y_levels
    }

    /// Number of levels (MIP convenience) — port of `int numLevels() const`.
    ///
    /// Returns [`ExrError::Logic`] for `RIPMAP_LEVELS` (undefined there), exactly
    /// as the C++ throws `LogicExc`.
    pub(crate) fn num_levels(&self) -> Result<i32> {
        if self.level_mode() == LevelMode::RipmapLevels {
            return Err(ExrError::Logic(
                "numLevels() is not defined for files with RIPMAP level mode.".into(),
            ));
        }
        Ok(self.num_x_levels)
    }

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(int, int) const`.
    pub(crate) fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
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
    ///
    /// Range-checks `lx` against the file's
    /// actual level count first. C++ 3.4.15 routes this through
    /// `exr_get_level_sizes` (`OpenEXRCore/part.c:347-383`), which throws
    /// `ArgExc` (`EXR_ERR_ARGUMENT_OUT_OF_RANGE`) for `levelx >=
    /// num_tile_levels_x`; calling `level_size` directly (as before) happily
    /// returned a value (often 1) for any level, in or out of range.
    pub(crate) fn level_width(&self, lx: i32) -> Result<i32> {
        if lx < 0 || lx >= self.num_x_levels {
            return Err(ExrError::Arg("levelWidth(): argument is not in valid range".into()));
        }
        level_size(self.min_x, self.max_x, lx, self.tile_desc.rounding_mode)
    }

    /// Height of level `ly` — port of `int levelHeight(int ly) const`.
    ///
    /// Range-checked like [`Self::level_width`].
    pub(crate) fn level_height(&self, ly: i32) -> Result<i32> {
        if ly < 0 || ly >= self.num_y_levels {
            return Err(ExrError::Arg("levelHeight(): argument is not in valid range".into()));
        }
        level_size(self.min_y, self.max_y, ly, self.tile_desc.rounding_mode)
    }

    /// Number of X tiles in level `lx` — port of `int numXTiles(int lx) const`.
    ///
    /// Returns [`ExrError::Arg`] when `lx` is out of range (C++ `ArgExc`).
    pub(crate) fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.num_x_tiles
            .get(usize::try_from(lx).map_err(|_| {
                ExrError::Arg("numXTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Arg("numXTiles(): argument is not in valid range".into()))
    }

    /// Number of Y tiles in level `ly` — port of `int numYTiles(int ly) const`.
    ///
    /// Returns [`ExrError::Arg`] when `ly` is out of range (C++ `ArgExc`).
    pub(crate) fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.num_y_tiles
            .get(usize::try_from(ly).map_err(|_| {
                ExrError::Arg("numYTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Arg("numYTiles(): argument is not in valid range".into()))
    }

    /// Data window of level `(lx, ly)` — port of
    /// `Box2i dataWindowForLevel(int lx, int ly) const`.
    ///
    /// Range-checks `lx`/`ly` first (see [`Self::level_width`]).
    pub(crate) fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        if lx < 0 || lx >= self.num_x_levels || ly < 0 || ly >= self.num_y_levels {
            return Err(ExrError::Arg(
                "dataWindowForLevel(): argument is not in valid range".into(),
            ));
        }
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
    /// `Box2i dataWindowForLevel(int l) const`, which forwards to `(l, l)`.
    pub(crate) fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.data_window_for_level(l, l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int lx, int ly) const`.
    ///
    /// Returns [`ExrError::Arg`] for an invalid tile (C++ `ArgExc`).
    pub(crate) fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
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
    pub(crate) fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.data_window_for_tile(dx, dy, l, l)
    }

    /// Whether `(dx, dy, lx, ly)` addresses an existing tile — port of
    /// `bool TiledInputFile::isValidTile(int, int, int, int) const`.
    ///
    /// Bounds check against the precomputed per-level tile counts (the same
    /// check the C++ `exr_get_tile_counts` performs).
    fn is_valid_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> bool {
        if lx < 0 || lx >= self.num_x_levels || ly < 0 || ly >= self.num_y_levels {
            return false;
        }
        // For MIPMAP only lx == ly tiles exist.
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
    // setFrameBuffer — ImfTiledInputFile.cpp `setFrameBuffer`
    // -----------------------------------------------------------------------

    /// Set the pixel destination — port of `TiledInputFile::setFrameBuffer`.
    ///
    /// Builds the per-channel `TInSliceInfo` table by walking *both* the file
    /// channel list and the frame buffer:
    ///
    /// * A file channel that also has a frame-buffer slice → a normal scatter
    ///   descriptor (with x/y-sampling compatibility checked, matching C++; a
    ///   tiled file requires sampling `(1,1)`).
    /// * A file channel with **no** frame-buffer slice → a `skip` descriptor:
    ///   its bytes are consumed from the tile chunk but not stored.
    /// * A frame-buffer slice with **no** file channel → a `fill` descriptor:
    ///   the slice is filled with its `fill_value` (the C++ *fill list*).
    ///
    /// The descriptors are built in **file-channel order** so the chunk's
    /// channel-interleaved byte stream is consumed in the order the writer laid
    /// it down.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] when a frame-buffer slice's x/y-sampling is
    /// incompatible with the corresponding header channel (matching C++).
    pub(crate) fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++: compatibility check — every frame-buffer slice that also names a
        // file channel must agree on x/y subsampling.
        for (name, slice) in frame_buffer.iter() {
            if let Some(ch) = channels.find_channel(name) {
                if ch.x_sampling != slice.x_sampling || ch.y_sampling != slice.y_sampling {
                    return Err(ExrError::Arg(format!(
                        "X and/or y subsampling factors of \"{name}\" channel of \
                         input file are not compatible with the frame buffer's \
                         subsampling factors."
                    )));
                }
            } else if slice.x_sampling != 1 || slice.y_sampling != 1 {
                // A FILL slice (no matching
                // file channel) must also be unsampled — C++ 3.4.15
                // `ImfTiledInputFile.cpp` `run_fill` (:971-972) throws
                // `ArgExc("Tiled data should not have subsampling.")` for
                // exactly this case. `scatter_fill_line` ignores sampling and
                // writes `num_pixels` samples at the slice's stride
                // regardless, which either overruns a buffer sized for the
                // subsampled channel or silently overwrites neighbouring data.
                return Err(ExrError::Arg(
                    "Tiled data should not have subsampling.".into(),
                ));
            }
        }

        // Build TInSliceInfo in file-channel order; absent file channels become
        // `skip` slices, frame-buffer slices with no file channel become `fill`
        // slices appended afterwards (the C++ fill_list).
        let mut slices: Vec<TInSliceInfo> = Vec::new();

        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                Some(slice) => {
                    slices.push(TInSliceInfo {
                        type_fb: slice.type_,
                        type_file: ch.type_,
                        base_offset: slice.base_offset,
                        x_stride: slice.x_stride,
                        y_stride: slice.y_stride,
                        x_tile_coords: slice.x_tile_coords,
                        y_tile_coords: slice.y_tile_coords,
                        fill: false,
                        fill_value: slice.fill_value,
                        skip: false,
                    });
                }
                None => {
                    // File channel with no destination: read & discard its bytes.
                    slices.push(TInSliceInfo {
                        type_fb: ch.type_,
                        type_file: ch.type_,
                        base_offset: 0,
                        x_stride: 0,
                        y_stride: 0,
                        x_tile_coords: false,
                        y_tile_coords: false,
                        fill: false,
                        fill_value: 0.0,
                        skip: true,
                    });
                }
            }
        }

        // Fill list: frame-buffer slices that the file does not provide.
        for (name, slice) in frame_buffer.iter() {
            if channels.find_channel(name).is_none() {
                slices.push(TInSliceInfo {
                    type_fb: slice.type_,
                    type_file: slice.type_,
                    base_offset: slice.base_offset,
                    x_stride: slice.x_stride,
                    y_stride: slice.y_stride,
                    x_tile_coords: slice.x_tile_coords,
                    y_tile_coords: slice.y_tile_coords,
                    fill: true,
                    fill_value: slice.fill_value,
                    skip: false,
                });
            }
        }

        self.slices = slices;
        self.frame_buffer = frame_buffer.clone();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // readTiles / readTile — ImfTiledInputFile.cpp `readTiles`/`readTile`
    // -----------------------------------------------------------------------

    /// Read the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)` into `out` — port of
    /// `TiledInputFile::readTiles(dx1,dx2,dy1,dy2,lx,ly)`.
    ///
    /// `out` is the caller-owned destination buffer the frame-buffer slices
    /// address.  The tile coordinate ranges are inclusive; `dx1 > dx2` /
    /// `dy1 > dy2` are swapped (as in C++).  Each tile is located via the
    /// offset table, its chunk read, decompressed and scattered into `out`.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the level is invalid
    ///   (matching the C++ throws).
    /// * [`ExrError::Input`] if a tile is missing, or its chunk header /
    ///   coordinates are corrupt (the C++ `readTileData` mismatch throw).
    /// * Propagates codec / scatter / stream errors.
    ///
    /// `pub(crate)`: `scanline::input_file::TiledReaderState` (the
    /// tiled-as-scanlines reader) calls this directly, threading its own
    /// `stream: &mut dyn IStream` through -- same reasoning as the group of
    /// accessors bumped above.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tiles(
        &mut self,
        stream: &mut dyn IStream,
        mut dx1: i32,
        mut dx2: i32,
        mut dy1: i32,
        mut dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer specified …").
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data destination.".into(),
            ));
        }

        // C++: if (!isValidLevel (lx, ly)) throw ArgExc.
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

        // C++ Data::readTiles: walk the requested rectangle in (ty, tx) order.
        // The scatter is order-independent (each tile carries its own coords,
        // and unlike the writer there is no on-disk emission order to
        // preserve for a READ), so any order reproduces the same pixels —
        // which is what makes the tiles below safe to read in ANY order the
        // parallel path chooses to batch them in.
        let mut coords: Vec<(i32, i32)> = Vec::new();
        for ty in dy1..=dy2 {
            for tx in dx1..=dx2 {
                coords.push((tx, ty));
            }
        }

        // Parallel only when there is more than one tile
        // AND the captured thread count is `> 1` — 1:1 with C++'s
        // `nTiles > 1 && numThreads > 1` (`ImfTiledInputFile.cpp:768`).
        if coords.len() > 1 && self.num_threads.is_parallel()
            && let Some(()) = self.read_tiles_parallel(stream, &coords, lx, ly, out)?
        {
            crate::probe::hit(crate::probe::Probe::TileReadParallel);
            return Ok(());
        }
        // `None` = the planner found overlapping write intervals (an
        // aliased/broadcast frame buffer) — fall back to the serial loop
        // below so the result is identical either way.

        // A local pool gives the serial loop the
        // same allocation-free-after-warmup behavior as scanline's
        // `fetch_chunk_into_line_buffer`, WITHOUT reintroducing the
        // anti-pattern (a persistent `tile_buffer` field sized to the
        // header's nominal tile size, which can be gigabytes for a tiny
        // data window -- see this function's earlier comment). The pool
        // only ever grows to the largest ACTUAL (clipped) tile `dataSize`
        // seen so far, exactly like `fetch_tile`'s own per-call allocation
        // already did, just reused instead of freed+realloc'd each tile.
        let pool = crate::buf::BufPool::new();
        for &(tx, ty) in &coords {
            let raw = self.fetch_tile(stream, tx, ty, lx, ly, &pool)?;
            let decoded = decode_tile(
                self.compressor.as_mut(),
                &raw,
                calculate_bytes_per_pixel(&self.header)?,
            )?;
            let result = scatter_tile(&self.slices, self.format, &raw.tile_range, &decoded, out, 0);
            drop(decoded); // end any borrow of raw.data before giving it back
            pool.give(raw.data);
            result?;
        }

        Ok(())
    }

    /// Fetch one tile's leader-validated, still-possibly-compressed bytes
    /// from `stream` (calling thread only — the sole function in the read
    /// pipeline that touches the stream). Port of the leader-read +
    /// validation half of the legacy per-tile read.
    ///
    /// `data`'s buffer comes from `pool` (a
    /// [`crate::buf::BufPool`]) instead of a fresh `try_filled_vec` per
    /// call — same mechanism as `scanline::input`'s `fetch_chunk`. The
    /// SERIAL caller (this file's `read_pixels` loop) and the PARALLEL
    /// caller (`read_tiles_parallel`) each pass their own pool: the serial
    /// loop's pool lives for the whole `read_pixels` call (one tile in
    /// flight at a time, so the pool converges to size 1 after the first
    /// tile); the parallel path shares one pool between the producer and
    /// worker across `read_tiles_parallel`'s `thread::scope`, exactly like
    /// `scanline::input::read_pixels_parallel`.
    fn fetch_tile(
        &self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        pool: &crate::buf::BufPool,
    ) -> Result<RawTile> {
        if !self.is_valid_tile(dx, dy, lx, ly) {
            return Err(ExrError::Arg(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is outside the image's data window."
            )));
        }

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

        let chunk_offset = self.tile_offsets.get(dx, dy, lx, ly)?;
        if chunk_offset == 0 {
            return Err(ExrError::Input(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is missing from the file."
            )));
        }

        stream.seekg(chunk_offset)?;
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in tile chunk: expected {expected}, found {found}."
                )));
            }
        }
        let r_dx = xdr::read_i32(stream)?;
        let r_dy = xdr::read_i32(stream)?;
        let r_lx = xdr::read_i32(stream)?;
        let r_ly = xdr::read_i32(stream)?;
        let data_size = xdr::read_i32(stream)?;

        if r_dx != dx || r_dy != dy || r_lx != lx || r_ly != ly {
            return Err(ExrError::Input(format!(
                "Unexpected tile coordinates in chunk header: expected \
                 ({dx}, {dy}, {lx}, {ly}), found ({r_dx}, {r_dy}, {r_lx}, {r_ly})."
            )));
        }

        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;
        let bytes_per_pixel = calculate_bytes_per_pixel(&self.header)?;
        let expected_uncompressed = bytes_per_pixel.saturating_mul(tile_w).saturating_mul(tile_h);
        if data_size < 0 || data_size as usize > expected_uncompressed {
            return Err(ExrError::Input(format!(
                "Tile chunk reports an invalid data size of {data_size} bytes \
                 (expected at most {expected_uncompressed} uncompressed bytes \
                 for this tile)."
            )));
        }
        let data_size = data_size as usize;

        let mut data = pool.take(data_size);
        stream.read(&mut data)?;

        Ok(RawTile { tile_range, data })
    }

    /// The parallel read body (mirrors `ScanLineInputFileState::read_pixels_parallel`).
    /// Returns `Ok(Some(()))` when it fully serviced the request, or
    /// `Ok(None)` when [`crate::misc::plan_disjoint`] found the frame
    /// buffer's write intervals unsafely overlapping — the caller then falls
    /// back to the fully serial loop.
    /// Read-ahead pipelined — see `scanline::input::ScanLineInputFileState::read_pixels_parallel`'s
    /// doc comment for the full rationale (measured I/O share at 16k widths,
    /// why a dedicated `std::thread::scope` worker rather than
    /// `rayon::Scope::spawn` directly, and why the decode worker is spawned
    /// ONCE for the whole call and fed windows over a channel rather than
    /// once per window transition — the many-small-chunks overhead that
    /// motivated that change applies identically to many-small-tiles).
    /// Identical shape, tile-flavoured.
    fn read_tiles_parallel(
        &mut self,
        stream: &mut dyn IStream,
        coords: &[(i32, i32)],
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<Option<()>> {
        let mut intervals = Vec::with_capacity(coords.len());
        for &(tx, ty) in coords {
            let tile_range = data_window_for_tile(
                &self.tile_desc,
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                tx,
                ty,
                lx,
                ly,
            )?;
            // One range PER ROW of the tile, not one merged span
            // for the whole tile -- see `tile_write_interval`'s doc for why
            // merging rows falsely "overlaps" with neighbouring tiles.
            intervals.push(tile_write_interval(&self.slices, &tile_range)?);
        }
        let mut dests: Vec<crate::misc::DestSet> = match crate::misc::plan_disjoint(out, &intervals) {
            Some(d) => d,
            None => return Ok(None),
        };

        let max_bytes_per_tile_line = self.max_bytes_per_tile_line;
        let ty_size = self.tile_desc.y_size as usize;
        // Byte-based window sizing — see `scanline::input::ScanLineInputFileState::read_pixels_parallel`'s
        // identical comment and `ThreadCount::window_size_for_bytes`'s doc.
        // `max_bytes_per_tile_line * ty_size` upper-bounds one tile's
        // raw/uncompressed size.
        let bytes_per_chunk = max_bytes_per_tile_line.saturating_mul(ty_size);
        let window = self.num_threads.window_size_for_bytes(bytes_per_chunk).max(1);
        let compression = self.compression;
        let header = &self.header;
        let slices = &self.slices;
        let format = self.format;
        let bytes_per_pixel = calculate_bytes_per_pixel(header)?;
        let count = self.num_threads;

        let coord_windows: Vec<&[(i32, i32)]> = coords.chunks(window).collect();
        let mut dest_windows: Vec<&mut [crate::misc::DestSet]> =
            dests.chunks_mut(window).collect();
        debug_assert_eq!(coord_windows.len(), dest_windows.len());

        if coord_windows.is_empty() {
            return Ok(Some(()));
        }

        // Same single-persistent-worker + channel-handoff design as
        // `ScanLineInputFileState::read_pixels_parallel` (see its doc for the
        // measured 65536-height/1-line-chunk regression this replaced: one
        // OS thread per window transition was 3x SLOWER than serial at that
        // extreme, since decode work per tiny window is nearly free next to
        // thread-spawn overhead).
        type WindowJob<'a, 'b> = (usize, Vec<RawTile>, &'b mut [crate::misc::DestSet<'a>]);
        type WindowDone = (usize, Result<()>);

        // One pool shared (by reference) between the
        // producer and worker, same mechanism as
        // `scanline::input::read_pixels_parallel`.
        let pool = crate::buf::BufPool::new();
        // Same starvation fix as the
        // flat scanline reader -- see `crate::buf::BufPool::prewarm_for_pipeline`'s
        // doc and `scanline::input::read_pixels_parallel`'s identical call
        // for the measured cost this eliminates (the same fetch-time inflation
        // was measured for tiled read).
        pool.prewarm_for_pipeline(crate::buf::READ_WINDOWS_IN_FLIGHT, window, bytes_per_chunk, coords.len());
        let pool = &pool;
        // Same `CodecPool` fix as
        // `scanline::input::read_pixels_parallel` -- see `Compressor`'s
        // trait doc and `crate::compressor::CodecPool`'s own doc.
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;

        std::thread::scope(|scope| -> Result<()> {
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<WindowJob>(1);
            let (done_tx, done_rx) = std::sync::mpsc::channel::<WindowDone>();

            let worker = scope.spawn(move || {
                while let Ok((idx, raws, dest_batch)) = job_rx.recv() {
                    let outcome: Result<()> = (|| {
                        let results: Vec<Result<()>> =
                            crate::threading::run_on_global_pool(count, move || {
                                use rayon::prelude::*;
                                // `into_par_iter()` so each
                                // closure invocation owns its `RawTile` and
                                // can give its buffer back to `pool` once
                                // decode+scatter are done with it -- see
                                // `scanline::input`'s identical fix for why
                                // this must happen AFTER `scatter_tile`, not
                                // right after `decode_tile` (the raw-
                                // passthrough case aliases `raw.data`).
                                raws.into_par_iter()
                                    .zip(dest_batch.par_iter_mut())
                                    .map(|(raw, dest_set)| -> Result<()> {
                                        let mut worker_compressor = codec_pool.checkout(|| {
                                            new_tile_compressor(
                                                compression,
                                                max_bytes_per_tile_line,
                                                ty_size,
                                                header,
                                            )
                                        })?;
                                        let decode_result =
                                            decode_tile(worker_compressor.as_mut(), &raw, bytes_per_pixel);
                                        let result = match decode_result {
                                            Ok(decoded) => {
                                                // One `ChunkDest` PER
                                                // ROW of the tile (see
                                                // `tile_write_interval`'s
                                                // doc) -- scatter row by
                                                // row via the same
                                                // `scatter_tile_row` the
                                                // serial `scatter_tile`
                                                // loops over.
                                                let mut read_offset = 0usize;
                                                let mut r = Ok(());
                                                for (row, y) in
                                                    (raw.tile_range.min.y..=raw.tile_range.max.y).enumerate()
                                                {
                                                    let dest = match dest_set.get_mut(row) {
                                                        Some(d) => d,
                                                        None => {
                                                            r = Err(ExrError::Base(format!(
                                                                "internal error: no destination for tile row {row}"
                                                            )));
                                                            break;
                                                        }
                                                    };
                                                    match scatter_tile_row(
                                                        slices,
                                                        format,
                                                        &raw.tile_range,
                                                        y,
                                                        &decoded,
                                                        read_offset,
                                                        dest.piece,
                                                        dest.base,
                                                    ) {
                                                        Ok(new_ro) => read_offset = new_ro,
                                                        Err(e) => {
                                                            r = Err(e);
                                                            break;
                                                        }
                                                    }
                                                }
                                                drop(decoded);
                                                r
                                            }
                                            Err(e) => Err(e),
                                        };
                                        codec_pool.checkin(worker_compressor);
                                        pool.give(raw.data);
                                        result
                                    })
                                    .collect()
                            })?;
                        for r in results {
                            r?;
                        }
                        Ok(())
                    })();
                    let stop = outcome.is_err();
                    if done_tx.send((idx, outcome)).is_err() || stop {
                        break;
                    }
                }
            });

            let mut io_err: Option<(usize, ExrError)> = None;
            for (i, dest_batch) in dest_windows.iter_mut().enumerate() {
                let raws: Result<Vec<RawTile>> = coord_windows[i]
                    .iter()
                    .map(|&(tx, ty)| self.fetch_tile(stream, tx, ty, lx, ly, pool))
                    .collect();
                let raws = match raws {
                    Ok(r) => r,
                    Err(e) => {
                        io_err = Some((i, e));
                        break;
                    }
                };
                let batch: &mut [crate::misc::DestSet] = std::mem::take(dest_batch);
                if job_tx.send((i, raws, batch)).is_err() {
                    break;
                }
                crate::probe::hit(crate::probe::Probe::ReadAheadWindow);
            }
            drop(job_tx);

            let mut worst: Option<(usize, ExrError)> = None;
            while let Ok((idx, outcome)) = done_rx.recv() {
                if let Err(e) = outcome
                    && worst.as_ref().is_none_or(|(w, _)| idx < *w)
                {
                    worst = Some((idx, e));
                }
            }
            if let Some((idx, e)) = io_err
                && worst.as_ref().is_none_or(|(w, _)| idx < *w)
            {
                worst = Some((idx, e));
            }

            worker
                .join()
                .map_err(|_| ExrError::Base("tiled decode worker thread panicked".into()))?;

            match worst {
                Some((_, e)) => Err(e),
                None => Ok(()),
            }
        })?;


        Ok(Some(()))
    }

    /// Read the rectangle of tiles at MIP level `l` — port of
    /// `TiledInputFile::readTiles(dx1,dx2,dy1,dy2,l)`, forwarding to `(…, l, l)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_tiles_l(
        &mut self,
        stream: &mut dyn IStream,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.read_tiles(stream, dx1, dx2, dy1, dy2, l, l, out)
    }

    /// Read a single tile `(dx, dy, lx, ly)` — port of
    /// `TiledInputFile::readTile(int dx, int dy, int lx, int ly)`, which
    /// forwards to `readTiles(dx, dx, dy, dy, lx, ly)`.
    pub(crate) fn read_tile(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.read_tiles(stream, dx, dx, dy, dy, lx, ly, out)
    }

    /// Read a single tile `(dx, dy, l)` at MIP level `l` — port of
    /// `TiledInputFile::readTile(int dx, int dy, int l)`, forwarding to
    /// `readTile(dx, dy, l, l)`.
    pub(crate) fn read_tile_l(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        l: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.read_tile(stream, dx, dy, l, l, out)
    }

    /// Read tile `(dx, dy, lx, ly)`'s raw on-disk bytes — packed
    /// (compressed, or already-raw for `NONE`/no-shrink) — without
    /// decompressing or scattering into a frame buffer. Port of `void
    /// rawTileData(int& dx, int& dy, int& lx, int& ly, const char*&
    /// pixelData, int& pixelDataSize)` (`ImfTiledInputFile.cpp:345-396`).
    ///
    /// # The in/out coordinate contract
    ///
    /// C++'s signature takes `dx`/`dy`/`lx`/`ly` **by reference**. Its
    /// header comment claims "for single part files, reads the next tile in
    /// the file; for multipart files, reads the tile specified by
    /// dx,dy,lx,ly" — but the actual v3.4.15 body (confirmed byte-identical
    /// to v3.3.11) is `exr_read_tile_chunk_info(_ctxt, partNumber, dx, dy,
    /// lx, ly, &cinfo)` → `validate_and_compute_tile_chunk_off` (OpenEXRCore
    /// `chunk.c:28-175`), which computes the chunk-table slot **purely from
    /// the exact `(tilex, tiley, levelx, levely)` passed in** — there is no
    /// "next tile" / cursor branch anywhere in that call graph, for either
    /// single-part or multipart, in EITHER version. `cinfo->start_x =
    /// tilex;` etc. (`chunk.c:1189-1197`) then simply echoes the input back
    /// into the out-parameters. The header comment is therefore stale
    /// documentation (most likely inherited from a pre-OpenEXRCore
    /// implementation where tiles genuinely were addressed by an internal
    /// read cursor) — confirmed by reading the source, not assumed.
    ///
    /// This port therefore takes `dx`/`dy`/`lx`/`ly` by value (Rust has no
    /// in/out reference parameters) and returns `(dx, dy, lx, ly, data)` —
    /// the ACTUAL coordinates the chunk header confirms (always equal to the
    /// input on success, matching C++'s real behaviour), preserving the
    /// public API's tuple shape exactly rather than silently dropping the
    /// out-parameters because they happen to be redundant today. Any caller
    /// that genuinely needs "iterate every tile in file order" should use
    /// [`Self::tile_order`] to get the coordinate list and call this
    /// once per entry — exactly what [`crate::tiled::TiledOutputFile::copy_pixels`]
    /// already does.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if `(dx, dy, lx, ly)` is outside the image (C++
    ///   `isValidTile` guard).
    /// * [`ExrError::Input`] if the tile is missing, or its chunk header's
    ///   coordinates / declared size don't match (a corrupt offset table or
    ///   file — same checks as [`Self::read_tile`]).
    pub(crate) fn raw_tile_data(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        if !self.is_valid_tile(dx, dy, lx, ly) {
            return Err(ExrError::Arg(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is outside the image's data window."
            )));
        }

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

        let chunk_offset = self.tile_offsets.get(dx, dy, lx, ly)?;
        if chunk_offset == 0 {
            return Err(ExrError::Input(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) is missing from the file."
            )));
        }

        stream.seekg(chunk_offset)?;
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in tile chunk: expected {expected}, found {found}."
                )));
            }
        }
        let r_dx = xdr::read_i32(stream)?;
        let r_dy = xdr::read_i32(stream)?;
        let r_lx = xdr::read_i32(stream)?;
        let r_ly = xdr::read_i32(stream)?;
        let data_size = xdr::read_i32(stream)?;

        if r_dx != dx || r_dy != dy || r_lx != lx || r_ly != ly {
            return Err(ExrError::Input(format!(
                "Unexpected tile coordinates in chunk header: expected \
                 ({dx}, {dy}, {lx}, {ly}), found ({r_dx}, {r_dy}, {r_lx}, {r_ly})."
            )));
        }

        // Bound: the packed size can never exceed this tile's
        // (clipped) uncompressed size — the same check `fetch_tile` applies.
        let tile_w = (tile_range.max.x - tile_range.min.x + 1) as usize;
        let tile_h = (tile_range.max.y - tile_range.min.y + 1) as usize;
        let bytes_per_pixel = calculate_bytes_per_pixel(&self.header)?;
        let expected_uncompressed = bytes_per_pixel
            .saturating_mul(tile_w)
            .saturating_mul(tile_h);
        if data_size < 0 || data_size as usize > expected_uncompressed {
            return Err(ExrError::Input(format!(
                "Tile chunk reports an invalid data size of {data_size} bytes \
                 (expected at most {expected_uncompressed} uncompressed bytes \
                 for this tile)."
            )));
        }

        let mut raw = vec![0u8; data_size as usize];
        stream.read(&mut raw)?;
        // C++: `dx = cinfo.start_x; dy = cinfo.start_y; lx = cinfo.level_x;
        // ly = cinfo.level_y;` — echo the (already chunk-header-verified)
        // actual coordinates back to the caller, matching the reference-out
        // parameters' real contract.
        Ok((r_dx, r_dy, r_lx, r_ly, raw))
    }

}

// ---------------------------------------------------------------------------
// Pure per-tile pipeline functions
// ---------------------------------------------------------------------------
//
// fetch_tile (stream-only, above) / decode_tile / scatter_tile are the
// tiled-reader analogue of the scanline reader's three-function split
// (scanline/input.rs): the serial path in read_tiles composes them in
// the same order the parallel path (read_tiles_parallel) does, so both
// produce byte-identical out.

/// One tile's leader-validated, still-possibly-compressed bytes, fetched by
/// TiledInputFileState::fetch_tile on the calling thread.
struct RawTile {
    /// The tile's pixel rectangle (clipped for an edge tile).
    tile_range: Box2i,
    /// The tile's raw stored bytes (length = the leader's dataSize).
    data: Vec<u8>,
}

/// Decode one RawTile into its uncompressed XDR bytes -- pure function, no
/// stream access, safe to run on any thread (each caller passes its own
/// compressor). Mirrors the C core's decompress_data raw-vs-compressed
/// rule, using the tile-rectangle-aware uncompress_tile.
fn decode_tile<'r>(
    // `&'r mut` (same as `scanline/input.rs::decode_chunk`):
    // unifying `compressor`'s borrow with `raw`'s lets the compressed-tile
    // path return `Cow::Borrowed(decoded)` instead of `decoded.to_vec()` --
    // `Compressor::uncompress_tile(&mut self, ..) -> Result<&[u8]>`'s output
    // borrows from `&mut self` by elision, so tying `compressor` to `'r`
    // makes that output's lifetime `'r` too. See the scanline version's own
    // doc for the NLL argument (each call site's `'r` is instantiated fresh,
    // so the next tile's re-borrow of the same compressor is unaffected).
    compressor: Option<&'r mut Box<dyn Compressor>>,
    raw: &'r RawTile,
    bytes_per_pixel: usize,
) -> Result<std::borrow::Cow<'r, [u8]>> {
    let tile_w = (raw.tile_range.max.x - raw.tile_range.min.x + 1) as usize;
    let tile_h = (raw.tile_range.max.y - raw.tile_range.min.y + 1) as usize;
    let expected_uncompressed = bytes_per_pixel.saturating_mul(tile_w).saturating_mul(tile_h);

    if raw.data.len() == expected_uncompressed {
        // Raw passthrough — borrows `raw.data` instead of cloning it (team-
        // lead follow-up: "as C++ or better"), matching the scanline
        // reader's identical `decode_chunk` fix. `scatter_tile` only ever
        // needs a `&[u8]`.
        return Ok(std::borrow::Cow::Borrowed(&raw.data));
    }
    let Some(c) = compressor else {
        return Err(ExrError::Input(format!(
            "Tile dataSize {} differs from the uncompressed size {expected_uncompressed} \
             but the file has no compressor.",
            raw.data.len()
        )));
    };
    let decoded = c
        .uncompress_tile(&raw.data, raw.tile_range, expected_uncompressed)
        .map_err(crate::error::decode_err)?;
    if decoded.len() != expected_uncompressed {
        return Err(ExrError::Input(format!(
            "Decompressed tile size {} does not match the expected uncompressed \
             size {expected_uncompressed} (corrupt chunk).",
            decoded.len()
        )));
    }
    // Borrow the codec's own output buffer instead of
    // `decoded.to_vec()`.
    Ok(std::borrow::Cow::Borrowed(decoded))
}

/// Scatter one decoded tile's bytes into dest, writing at
/// write_ptr - dest_base instead of write_ptr -- see the scanline
/// reader's identical scatter_chunk doc for why (dest may be the whole
/// frame buffer, dest_base = 0, or one exclusive
/// crate::misc::ChunkDest piece from crate::misc::plan_disjoint).
///
/// Loops `scatter_tile_row` over every row of the tile into the SAME
/// `dest`/`dest_base` -- the shape the serial reader needs (`out`, base
/// 0, one contiguous destination for the whole tile). The parallel reader
/// (`read_tiles_parallel`) instead calls `scatter_tile_row` directly, once
/// per row, against that row's own `crate::misc::ChunkDest` piece (see
/// `tile_write_interval`'s doc for why a tile's rows must be split before
/// `plan_disjoint`, not after) -- ONE row-scatter function, two callers.
fn scatter_tile(
    slices: &[TInSliceInfo],
    format: Format,
    tile_range: &Box2i,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    let mut read_offset = 0usize;
    for y in tile_range.min.y..=tile_range.max.y {
        read_offset = scatter_tile_row(slices, format, tile_range, y, decoded, read_offset, dest, dest_base)?;
    }
    Ok(())
}

/// Scatter ONE row (`y`) of a decoded tile's bytes, starting at `decoded`
/// offset `read_offset`, into `dest`/`dest_base`. Returns the `read_offset`
/// just past this row's bytes (the next row's start), so callers that
/// process a tile row-by-row (the parallel reader) can thread it through
/// exactly like the whole-tile loop in `scatter_tile` does.
#[allow(clippy::too_many_arguments)]
fn scatter_tile_row(
    slices: &[TInSliceInfo],
    format: Format,
    tile_range: &Box2i,
    y: i32,
    decoded: &[u8],
    mut read_offset: usize,
    dest: &mut [u8],
    dest_base: usize,
) -> Result<usize> {
    let buf_format = match format {
        Format::Xdr => BufferFormat::Xdr,
        Format::Native => BufferFormat::Native,
    };
    let num_pixels = (tile_range.max.x - tile_range.min.x + 1) as usize;

    for slice in slices {
        if slice.fill {
            scatter_fill_line(slice, y, tile_range.min.x, tile_range.min.y, num_pixels, dest, dest_base)?;
            continue;
        }

        let file_px_size = pixel_type_size(slice.type_file)?;
        let channel_bytes = file_px_size * num_pixels;

        if slice.skip {
            read_offset += channel_bytes;
            continue;
        }

        let x_offset = if slice.x_tile_coords { tile_range.min.x } else { 0 };
        let y_offset = if slice.y_tile_coords { tile_range.min.y } else { 0 };

        let line_ptr: i64 = slice.base_offset + (y - y_offset) as i64 * slice.y_stride as i64;
        let write_ptr_i: i64 =
            line_ptr + (tile_range.min.x - x_offset) as i64 * slice.x_stride as i64;
        if write_ptr_i < 0 {
            return Err(ExrError::Arg(
                "Frame buffer slice address is out of range (negative).".into(),
            ));
        }
        let write_ptr = local_addr(write_ptr_i as usize, dest_base)?;

        let (new_ro, _new_wp) = copy_into_frame_buffer(
            decoded,
            read_offset,
            dest,
            write_ptr,
            num_pixels,
            slice.x_stride,
            false,
            slice.fill_value,
            buf_format,
            slice.type_fb,
            slice.type_file,
        )?;
        read_offset = new_ro;
    }
    Ok(read_offset)
}

/// abs_addr - base, erroring instead of underflowing when a caller passes
/// an address before its own piece's base -- see the scanline reader's
/// identical helper for the full rationale.
fn local_addr(abs_addr: usize, base: usize) -> Result<usize> {
    abs_addr.checked_sub(base).ok_or_else(|| {
        ExrError::Base(format!(
            "internal error: scatter address {abs_addr} precedes its destination \
             piece's base {base}"
        ))
    })
}

/// Fill one line of a fill slice with its fill_value -- free-function form
/// of the tiled counterpart of TileProcess::run_fill, usable from both the
/// serial scatter (dest_base = 0) and the parallel one (a
/// crate::misc::ChunkDest piece).
#[allow(clippy::too_many_arguments)]
fn scatter_fill_line(
    slice: &TInSliceInfo,
    y: i32,
    tile_min_x: i32,
    tile_min_y: i32,
    num_pixels: usize,
    dest: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    // The offset must be the TILE's min.y (constant across the tile), not
    // the current row y: C++ `run_fill` computes it once per tile and then
    // advances one `yStride` per row.
    let x_offset = if slice.x_tile_coords { tile_min_x } else { 0 };
    let y_offset = if slice.y_tile_coords { tile_min_y } else { 0 };

    let line_ptr: i64 = slice.base_offset + (y - y_offset) as i64 * slice.y_stride as i64;
    let write_ptr_i: i64 = line_ptr + (tile_min_x - x_offset) as i64 * slice.x_stride as i64;
    if write_ptr_i < 0 {
        return Err(ExrError::Arg(
            "Frame buffer fill slice address is out of range (negative).".into(),
        ));
    }
    let write_ptr = local_addr(write_ptr_i as usize, dest_base)?;

    copy_into_frame_buffer(
        &[],
        0,
        dest,
        write_ptr,
        num_pixels,
        slice.x_stride,
        true,
        slice.fill_value,
        BufferFormat::Xdr,
        slice.type_fb,
        slice.type_file,
    )?;
    Ok(())
}

/// Compute, PER ROW of the tile, the bounding byte interval `[lo, hi)`
/// scatter_tile_row would write into out for that row, WITHOUT writing
/// anything -- the input to crate::misc::plan_disjoint. Walks the same
/// address formulas as scatter_tile_row/scatter_fill_line so the two can
/// never disagree.
///
/// A tile's rows are addressed in
/// whole-IMAGE raster order (`y_stride` is the full scanline stride, not
/// the tile's own width), so a multi-row tile's own rows sit FAR apart in
/// the frame buffer -- the gap between them holds every OTHER tile in the
/// same tile-row band. Merging every row into ONE `lo..hi` span (the
/// naive shape) therefore covers those neighbouring tiles' bytes too,
/// so `plan_disjoint` sees two tiles' spans "overlap" even though their
/// real writes never touch, and the parallel path silently, permanently
/// falls back to serial (confirmed by an instrumented count at 64x64 AND
/// 256x256: this is the same cross-row-merging bug class handled for deep
/// tiled reads, `deep_tile_write_interval`, just in
/// the flat reader). One range PER ROW keeps each row's span tight to
/// that row's own tile-width slice of the buffer, so two tiles in the
/// same row band no longer look aliased.
fn tile_write_interval(
    slices: &[TInSliceInfo],
    tile_range: &Box2i,
) -> Result<crate::misc::RangeSet> {
    let num_pixels = (tile_range.max.x - tile_range.min.x + 1).max(0) as usize;
    let tile_h = (tile_range.max.y - tile_range.min.y + 1).max(0) as usize;
    let mut ranges = crate::misc::RangeSet::with_capacity(tile_h);

    if num_pixels == 0 {
        ranges.extend(std::iter::repeat_n(0..0, tile_h));
        return Ok(ranges);
    }

    for y in tile_range.min.y..=tile_range.max.y {
        let mut lo: Option<usize> = None;
        let mut hi: usize = 0;

        for slice in slices {
            if slice.skip {
                continue;
            }
            let fb_size = pixel_type_size(slice.type_fb)?;
            let x_offset = if slice.x_tile_coords { tile_range.min.x } else { 0 };
            let y_offset = if slice.y_tile_coords { tile_range.min.y } else { 0 };

            let line_ptr: i64 = slice.base_offset + (y - y_offset) as i64 * slice.y_stride as i64;
            let write_ptr_i: i64 =
                line_ptr + (tile_range.min.x - x_offset) as i64 * slice.x_stride as i64;
            if write_ptr_i < 0 {
                return Err(ExrError::Arg(
                    "Frame buffer slice address is out of range (negative).".into(),
                ));
            }
            let write_ptr = write_ptr_i as usize;

            let last_pixel_i = write_ptr_i + (num_pixels as i64 - 1) * slice.x_stride as i64;
            if last_pixel_i < 0 {
                return Err(ExrError::Arg(
                    "Frame buffer slice address is out of range (negative).".into(),
                ));
            }
            let span_lo = write_ptr.min(last_pixel_i as usize);
            let span_hi = write_ptr.max(last_pixel_i as usize) + fb_size;

            lo = Some(lo.map_or(span_lo, |l| l.min(span_lo)));
            hi = hi.max(span_hi);
        }

        ranges.push(match lo {
            Some(l) => l..hi,
            None => 0..0,
        });
    }

    Ok(ranges)
}

// ---------------------------------------------------------------------------
// Tests — write→read round-trip vs the tiled writer (the key correctness gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{
        Channel, ChannelList, ChannelListAttribute, Compression, LineOrder,
        TileDescriptionAttribute,
    };
    use crate::frame_buffer::Slice;
    use crate::header::TILED_IMAGE;
    use crate::io::{MemIStream, MemOStream};
    use crate::tiled::TiledOutputFile;
    use imath_rs::{Half, V2f};

    /// Build a tiled header with data window `[min_x, min_x+width-1] x
    /// [min_y, min_y+height-1]`, one HALF channel "R", the given compression,
    /// line order and tile description.
    fn make_header(
        min_x: i32,
        min_y: i32,
        width: i32,
        height: i32,
        line_order: LineOrder,
        comp: Compression,
        td: TileDescription,
    ) -> Header {
        // Header::with_dimensions builds a 0..width-1 / 0..height-1 window; we
        // then override the data window to the requested origin via insert.
        let mut h = Header::with_dimensions(
            width,
            height,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            line_order,
            comp,
        )
        .unwrap();
        // Override data window to the requested (possibly off-origin) box.
        let dw = imath_rs::Box2i::from_min_max(
            imath_rs::V2i { x: min_x, y: min_y },
            imath_rs::V2i {
                x: min_x + width - 1,
                y: min_y + height - 1,
            },
        );
        h.insert("dataWindow", crate::attr::Box2iAttribute::new(dw)).unwrap();
        h.insert("displayWindow", crate::attr::Box2iAttribute::new(dw)).unwrap();
        h.set_type(TILED_IMAGE).unwrap();
        h.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Row-major HALF pixel plane over the data window `[min_x..] x [min_y..]`,
    /// one channel.  Pixel (x, y) holds `half((y - min_y) * width + (x - min_x))`
    /// — a distinct value per pixel.  `base_offset = 0` addresses pixel
    /// (min_x, min_y); strides are window-relative (xTileCoords = false).
    /// Returns `(buffer, frame_buffer)`.
    fn make_pixels(min_x: i32, min_y: i32, width: i32, height: i32) -> (Vec<u8>, FrameBuffer) {
        let row_bytes = width as usize * 2;
        let mut buf = vec![0u8; row_bytes * height as usize];
        for ry in 0..height {
            for rx in 0..width {
                let off = ry as usize * row_bytes + rx as usize * 2;
                let v = Half::from((ry * width + rx) as f32).bits();
                buf[off..off + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        // base_offset addresses (min_x, min_y); the absolute pixel (x, y) lands
        // at base + (y - min_y)*yStride + (x - min_x)*xStride.  Setting
        // base_offset = -(min_y*yStride + min_x*xStride) makes the absolute
        // address arithmetic land at buffer index 0 for (min_x, min_y).
        let x_stride = 2i64;
        let y_stride = row_bytes as i64;
        let base_offset = -(min_y as i64 * y_stride + min_x as i64 * x_stride);
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset,
                x_stride: 2,
                y_stride: row_bytes,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        (buf, fb)
    }

    /// Write all tiles of every level, then read them all back and assert the
    /// frame buffer is pixel-exact equal to what was written.
    fn round_trip_all(
        min_x: i32,
        min_y: i32,
        width: i32,
        height: i32,
        line_order: LineOrder,
        comp: Compression,
        td: TileDescription,
    ) {
        let header = make_header(min_x, min_y, width, height, line_order, comp, td);
        let (pixels, fb) = make_pixels(min_x, min_y, width, height);

        // --- write every tile of every level via the tiled writer ---------
        let mut out = MemOStream::new();
        let levels: Vec<(i32, i32)> = match td.mode {
            LevelMode::OneLevel => vec![(0, 0)],
            LevelMode::MipmapLevels => {
                let mut v = Vec::new();
                // numXLevels == numYLevels for mipmap; lx == ly.
                let (_, _, nxl, _) = precalculate_tile_info(&td, min_x, min_x + width - 1, min_y, min_y + height - 1).unwrap();
                for l in 0..nxl {
                    v.push((l, l));
                }
                v
            }
            LevelMode::RipmapLevels => {
                let mut v = Vec::new();
                let (_, _, nxl, nyl) = precalculate_tile_info(&td, min_x, min_x + width - 1, min_y, min_y + height - 1).unwrap();
                for ly in 0..nyl {
                    for lx in 0..nxl {
                        v.push((lx, ly));
                    }
                }
                v
            }
            LevelMode::Invalid => unreachable!(),
        };
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            for &(lx, ly) in &levels {
                let nx = tof.num_x_tiles(lx).unwrap();
                let ny = tof.num_y_tiles(ly).unwrap();
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, lx, ly).unwrap();
            }
        }
        let bytes = out.into_data();

        // --- read every tile of every level back ---------------------------
        // Each level scatters into its own buffer (the level data windows
        // differ in size); we compare level 0 against the source plane and
        // confirm every other level reads without error and is internally
        // consistent (value(x,y) for that level's own pixels).
        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        assert!(tif.is_complete(), "offset table must be complete");
        assert_eq!(tif.compression().unwrap(), comp);

        for &(lx, ly) in &levels {
            let lvl = tif.data_window_for_level(lx, ly).unwrap();
            let lw = lvl.max.x - lvl.min.x + 1;
            let lh = lvl.max.y - lvl.min.y + 1;
            // Fresh destination plane for this level, base addressing its origin.
            let row_bytes = lw as usize * 2;
            let mut readback = vec![0u8; row_bytes * lh as usize];
            let x_stride = 2i64;
            let y_stride = row_bytes as i64;
            let base_offset = -(lvl.min.y as i64 * y_stride + lvl.min.x as i64 * x_stride);
            let mut rfb = FrameBuffer::new();
            rfb.insert(
                "R",
                Slice {
                    type_: PixelType::Half,
                    base_offset,
                    x_stride: 2,
                    y_stride: row_bytes,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
            tif.set_frame_buffer(&rfb).unwrap();
            let nx = tif.num_x_tiles(lx).unwrap();
            let ny = tif.num_y_tiles(ly).unwrap();
            tif.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut readback).unwrap();

            // For level 0 the readback plane must equal the source plane exactly.
            if lx == 0 && ly == 0 {
                assert_eq!(readback, pixels, "level 0 round-trip mismatch ({comp:?}, {line_order:?})");
            } else {
                // Other levels: the writer's FrameBuffer addresses `pixels` with
                // xTileCoords = false, i.e. by the FULL image's absolute pixel
                // coordinates, and every level shares the same data-window MIN
                // (levels shrink toward the top-left) -- so a level-L tile pixel
                // at LOCAL (x, y) reads the exact same buffer cell as full-res
                // pixel (min_x + x, min_y + y). That cell holds
                // half((y * width + x) as f32) by `make_pixels`'s own formula
                // (row index y, column index x, stride = FULL width, not the
                // level's own narrower width). Assert every level's readback
                // against that independently-recomputed expectation, not just
                // its length -- a length check can't fail on wrong VALUES.
                for y in 0..lh {
                    for x in 0..lw {
                        let expected = Half::from((y * width + x) as f32).bits();
                        let off = y as usize * row_bytes + x as usize * 2;
                        let got = u16::from_ne_bytes([readback[off], readback[off + 1]]);
                        assert_eq!(
                            got, expected,
                            "level ({lx},{ly}) pixel ({x},{y}) mismatch ({comp:?}, {line_order:?})"
                        );
                    }
                }
            }
        }
    }

    /// Read the SAME multi-tile (one-level) file with
    /// `num_threads` in {0, 1, 2, 3, 8, 17} and assert every thread count
    /// produces byte-identical `out` — the tiled-reader analogue of
    /// `scanline::input::tests::parallel_read_is_byte_identical_to_serial_across_thread_counts`.
    /// A `1x1` nominal tile over a `37x29` image gives `37*29 = 1073` tiles,
    /// comfortably exercising multiple parallel windows at every thread count.
    #[test]
    fn parallel_read_tiles_is_byte_identical_to_serial_across_thread_counts() {
        let td = TileDescription {
            x_size: 5,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (37i32, 29i32); // odd sizes, not a multiple of the tile size
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::TileReadParallel);
            let mut tif = TiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            let (mut readback, readback_fb) = make_pixels(0, 0, w, h);
            readback.iter_mut().for_each(|b| *b = 0);
            tif.set_frame_buffer(&readback_fb).unwrap();
            let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
            // With `n > 1` and 8 tile
            // columns, the parallel path must actually ENGAGE, not silently
            // fall back to serial -- see `tile_write_interval`'s doc for the
            // cross-row-merging bug this proves is fixed (pre-fix,
            // `plan_disjoint` rejected every multi-column tiled read, so
            // this assertion would have failed for every `n > 1` here).
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::TileReadParallel, n);
                assert!(
                    crate::probe::get(crate::probe::Probe::PlanDisjointAccepted) > 0,
                    "num_threads={n}: plan_disjoint rejected the tiled parallel path \
                     (fell back to serial) -- expected it to engage with 8 tile columns"
                );
            }

            assert_eq!(readback, pixels, "num_threads={n}: mismatch vs the written pixels");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// Proof of buffer-pool reuse (tiled INPUT, SERIAL path) -- the fully
    /// deterministic counterpart to
    /// `scanline::input::tests::parallel_read_reuses_some_pool_buffers_across_windows`.
    ///
    /// FINDING (worth recording): the first version of this test used the
    /// PROCESS-WIDE `crate::buf::fresh_allocations()` counter (reset before
    /// the read, read after), the same instrumentation the scanline test
    /// uses. In isolation it passed reliably. Under the FULL `cargo test -p
    /// exr-core --release --lib` suite (1000+ tests, `cargo test`'s default
    /// same-process, many-threads-at-once execution model) it failed
    /// intermittently with counts in the hundreds -- swamped by OTHER,
    /// completely unrelated tests' concurrent `BufPool` traffic on the SAME
    /// global counter (`buf.rs`'s own doc already flags this as a
    /// possibility; the actual full-suite noise turned out far larger than
    /// a "small margin", up to 467 in one run, vs. this test's own ceiling
    /// of 16 fresh allocations if reuse were completely broken). A global
    /// counter cannot give a tight, meaningful bound when run alongside
    /// hundreds of other pool-using tests.
    ///
    /// Fixed by testing the [`crate::buf::BufPool`] mechanism directly with
    /// a pool THIS TEST OWNS, calling the same private `fetch_tile` the
    /// real serial loop in `TiledInputFileState::read_tiles` calls, in the
    /// exact same take-then-give sequence. A locally-owned pool has zero
    /// cross-test contamination by construction (nothing outside this
    /// function ever touches it) -- this is a "stable capacity" proof, not a global
    /// counter, and needs no margin at all: every one of the 16 tiles is
    /// the same size, so a working pool returns the exact same allocation
    /// (same capacity) every time after the first.
    #[test]
    fn serial_read_reuses_pool_buffer_across_tiles() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (16i32, 16i32); // 4x4 tiles of 4x4 pixels = 16 tiles
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let tif = TiledInputFile::new(&mut is).unwrap();

        // Mutation check: temporarily changed the loop below to `let pool =
        // crate::buf::BufPool::new();` INSIDE the loop (a fresh pool every
        // tile, defeating reuse) -- the capacity assertion failed on tile 2
        // (a fresh `Vec::new()` has capacity 0, not the first tile's grown
        // capacity). Reverted after confirming red.
        let pool = crate::buf::BufPool::new();
        let mut first_capacity = None;
        for ty in 0..4 {
            for tx in 0..4 {
                let raw = tif.state.fetch_tile(&mut *tif.stream, tx, ty, 0, 0, &pool).unwrap();
                let cap = raw.data.capacity();
                match first_capacity {
                    None => first_capacity = Some(cap),
                    Some(first) => assert_eq!(
                        cap, first,
                        "tile ({tx},{ty}): buffer capacity changed -- expected the exact \
                         same freed allocation to come back from the pool every time (all \
                         16 tiles are the same size, 4x4 Half pixels)"
                    ),
                }
                pool.give(raw.data);
            }
        }
        assert_eq!(
            pool.free_count(),
            1,
            "exactly one buffer must be sitting in the free list after the loop \
             (take-then-give, one at a time, never more than one in flight)"
        );
    }

    /// Same matrix over ZIP-compressed tiles (real codec decode in the
    /// parallel path, per-worker fresh-compressor requirement).
    #[test]
    fn parallel_read_tiles_zip_matches_across_thread_counts() {
        let td = TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (40i32, 33i32);
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::Zip, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::TileReadParallel);
            let mut tif = TiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            let (mut readback, readback_fb) = make_pixels(0, 0, w, h);
            readback.iter_mut().for_each(|b| *b = 0);
            tif.set_frame_buffer(&readback_fb).unwrap();
            let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::TileReadParallel, n);
                assert!(
                    crate::probe::get(crate::probe::Probe::PlanDisjointAccepted) > 0,
                    "num_threads={n}: plan_disjoint rejected the tiled parallel path \
                     (fell back to serial) -- expected it to engage with multiple tile columns"
                );
            }

            assert_eq!(readback, pixels, "num_threads={n}: mismatch vs the written pixels");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// A measurement showed `map_init` constructing ~8192/8192 codecs for
    /// an 8192-tile read, at every thread count. With a real `CodecPool`,
    /// the number of codecs actually CONSTRUCTED must be bounded by
    /// roughly the thread count, not the tile count -- 1024 small tiles at
    /// `num_threads=24` must construct nowhere near 1024 codecs.
    #[test]
    fn parallel_read_tiles_codec_pool_created_is_bounded_by_thread_count_not_tile_count() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (128i32, 128i32); // 32x32 = 1024 tiles
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::Zip, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let n = 24i32;
        crate::probe::reset(crate::probe::Probe::CodecCreated); // the writer above has its own pool
        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
        let (mut readback, readback_fb) = make_pixels(0, 0, w, h);
        readback.iter_mut().for_each(|b| *b = 0);
        tif.set_frame_buffer(&readback_fb).unwrap();
        let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
        tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
        assert_eq!(readback, pixels, "sanity: the read must still be correct");

        let created = crate::probe::get(crate::probe::Probe::CodecCreated);
        assert!(
            created <= (n as u64) * 2,
            "expected CodecPool to construct roughly num_threads={n} codecs for 1024 tiles, \
             not one per tile -- got {created} (map_init's own measured pre-fix behavior on \
             a real file was ~8192/8192, one per tile, regardless of thread count)"
        );
        assert!(created > 0, "sanity: the pool must have constructed at least one codec");
    }

    #[test]
    fn parallel_tiled_pipelines_never_miss_the_prewarmed_pool() {
        // U4/U4b: after `prewarm_for_pipeline` every `BufPool::take` in the
        // parallel tiled writer and reader must hit. With the old shared
        // depth of 2 windows the reader missed exactly one window (its
        // pipeline holds 3: decoding, queued, fetching).
        let td = TileDescription { x_size: 64, y_size: 64, mode: LevelMode::OneLevel, rounding_mode: LevelRoundingMode::RoundDown };
        let (w, h) = (1024i32, 1024i32); // 256 tiles
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::Zip, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);
        let misses = || crate::probe::get(crate::probe::Probe::BufPoolMiss);
        for &n in &[2i32, 8, 24] {
            let mut out = MemOStream::new();
            crate::probe::reset(crate::probe::Probe::BufPoolMiss);
            {
                let mut tof = TiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            assert_eq!(misses(), 0, "num_threads={n}: parallel tiled write missed the prewarmed pool");
            let bytes = out.into_data();

            let mut is = MemIStream::new(bytes);
            let mut tif = TiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            let (mut readback, read_fb) = make_pixels(0, 0, w, h);
            tif.set_frame_buffer(&read_fb).unwrap();
            let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
            crate::probe::reset(crate::probe::Probe::BufPoolMiss);
            crate::probe::reset(crate::probe::Probe::BufPoolHit);
            crate::probe::reset(crate::probe::Probe::ReadAheadWindow);
            tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
            // U6: several windows pipelined, not one covering every tile.
            assert!(crate::probe::get(crate::probe::Probe::ReadAheadWindow) >= 2, "num_threads={n}: read-ahead ran a single window");
            assert_eq!(readback, pixels, "num_threads={n}: mismatch");
            assert_eq!(misses(), 0, "num_threads={n}: parallel tiled read missed the prewarmed pool");
            assert_eq!(crate::probe::get(crate::probe::Probe::BufPoolHit), (nx * ny) as u64, "num_threads={n}: one pooled buffer per tile");
        }
    }

    /// The EXACT tile sizes the thread-scaling bench used to expose the bug (64x64, `tiled read 1.0x`
    /// at 24 threads): a multi-column, multi-row tiled image at both 64x64
    /// and 256x256 tiles must show `plan_disjoint` actually engaging the
    /// parallel path for every `n > 1` -- before the `tile_write_interval`
    /// fix (one merged span per tile instead of one span per row), EVERY
    /// one of these would have fallen back to serial (10/10 in deflate-
    /// fast's own instrumented run).
    #[test]
    fn parallel_read_tiles_engages_at_64_and_256_tile_size() {
        for tile_size in [64i32, 256] {
            let td = TileDescription {
                x_size: tile_size as u32,
                y_size: tile_size as u32,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            };
            // >= 2 tile columns and >= 2 tile rows at both sizes, odd
            // remainder in both axes (not a multiple of the tile size).
            let (w, h) = (tile_size * 3 + 7, tile_size * 2 + 5);
            let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
            let (pixels, fb) = make_pixels(0, 0, w, h);

            let mut out = MemOStream::new();
            {
                let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                assert!(nx >= 2, "tile_size={tile_size}: expected >= 2 tile columns, got {nx}");
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            let bytes = out.into_data();

            let mut reference: Option<Vec<u8>> = None;
            for &n in &[1i32, 2, 8] {
                let mut is = MemIStream::new(bytes.clone());
                crate::probe::reset(crate::probe::Probe::TileReadParallel);
                let mut tif = TiledInputFile::with_num_threads(&mut is, Some(n)).unwrap();
                let (mut readback, readback_fb) = make_pixels(0, 0, w, h);
                readback.iter_mut().for_each(|b| *b = 0);
                tif.set_frame_buffer(&readback_fb).unwrap();
                let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
                crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
                tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
                if n > 1 {
                    crate::probe::assert_engaged(crate::probe::Probe::TileReadParallel, n);
                    assert!(
                        crate::probe::get(crate::probe::Probe::PlanDisjointAccepted) > 0,
                        "tile_size={tile_size}, num_threads={n}: plan_disjoint rejected \
                         the tiled parallel path -- this is exactly the bug deflate-fast's \
                         thread-scaling bench proved (tiled read pinned at 1.0x)"
                    );
                }
                assert_eq!(readback, pixels, "tile_size={tile_size}, num_threads={n}: mismatch");
                match &reference {
                    None => reference = Some(readback),
                    Some(r) => assert_eq!(
                        *r, readback,
                        "tile_size={tile_size}, num_threads={n}: differs from num_threads=1"
                    ),
                }
            }
        }
    }

    /// Negative test: a frame buffer
    /// where DIFFERENT TILES genuinely alias -- `x_tile_coords`/
    /// `y_tile_coords` both `true` with `base_offset = 0` addresses every
    /// tile relative to its OWN origin, so if the caller (mistakenly)
    /// points every tile at the same small shared buffer instead of giving
    /// each tile its own destination, tile (0,0) and tile (1,0) write the
    /// SAME bytes. Two tiles decoded concurrently on different threads
    /// really would race on those bytes, so `plan_disjoint` must reject
    /// this (serial fallback), never accept it for parallel scatter.
    /// (A same-tile, cross-CHANNEL alias -- e.g. two slices at the same
    /// address in one tile -- is NOT a race for this reader: one rayon
    /// task decodes+scatters a whole tile's channels sequentially before
    /// the next tile starts, so it isn't the hazard `plan_disjoint`
    /// exists for here; the deep reader's own negative test covers that
    /// shape instead, where per-slice ranges are kept separate.)
    #[test]
    fn parallel_read_tiles_rejects_genuinely_aliased_tiles_falls_back_to_serial() {
        let td = TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (16i32, 8i32); // 2 tile columns, 1 tile row
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            assert!(nx >= 2, "expected >= 2 tile columns to alias against each other");
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        // Tile-relative addressing, base 0 -- every tile's (0,0) pixel
        // lands at the SAME byte offset in a shared buffer sized for only
        // ONE tile's worth of pixels.
        let aliased = Slice {
            type_: PixelType::Half,
            base_offset: 0,
            x_stride: 2,
            y_stride: 2 * 8, // tile width * pixel size
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: true,
            y_tile_coords: true,
        };
        let mut aliased_fb = FrameBuffer::new();
        aliased_fb.insert("R", aliased).unwrap();
        let mut readback = vec![0u8; 8 * 8 * 2]; // one tile's worth only

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::with_num_threads(&mut is, Some(8)).unwrap();
        tif.set_frame_buffer(&aliased_fb).unwrap();
        let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
        crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
        tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
        assert_eq!(
            crate::probe::get(crate::probe::Probe::PlanDisjointAccepted),
            0,
            "genuinely aliased tiles (shared tile-relative destination) must be rejected \
             by plan_disjoint (serial fallback), not accepted for parallel scatter"
        );
    }

    /// A tiled file whose offset table was never written must still read.
    /// The reconstruction now runs through the SHARED chunk sweep
    /// (`chunk::offsets::reconstruct`), so this also guards that refactor.
    #[test]
    fn zeroed_offset_table_is_reconstructed() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (4i32, 4i32);
        let header = make_header(0, 0, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let mut bytes = out.into_data();

        // The table ends exactly where the first tile chunk begins.
        let (table_pos, count) = {
            let mut is = MemIStream::new(bytes.clone());
            let tif = TiledInputFile::new(&mut is).unwrap();
            let flat: Vec<u64> = tif
                .state.tile_offsets
                .get_offsets()
                .iter()
                .flat_map(|l| l.iter())
                .flat_map(|r| r.iter().copied())
                .collect();
            let first = flat.iter().copied().min().unwrap() as usize;
            (first - flat.len() * 8, flat.len())
        };
        for b in bytes[table_pos..table_pos + count * 8].iter_mut() {
            *b = 0;
        }

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        let mut readback = vec![0u8; pixels.len()];
        tif.set_frame_buffer(&fb).unwrap();
        let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
        tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
        assert_eq!(readback, pixels, "tiled reconstruction round-trip mismatch");
    }

    #[test]
    fn one_level_none_increasing_y_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    #[test]
    fn one_level_none_decreasing_y_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 4, 6, LineOrder::DecreasingY, Compression::NoCompression, td);
    }

    #[test]
    fn mipmap_none_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    /// The `LevelMode::RipmapLevels` branch of `round_trip_all` was never
    /// exercised by any `#[test]` -- a non-square image forces `numXLevels !=
    /// numYLevels`, proving a genuine 2D grid is written/read, not a diagonal
    /// mipmap-shaped one.
    #[test]
    fn ripmap_none_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::RipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 8, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    #[test]
    fn one_level_zip_pixel_exact() {
        // ZIP ignores the tile rectangle → lossless tiled round-trip works now.
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 8, 8, LineOrder::IncreasingY, Compression::Zip, td);
    }

    #[test]
    fn one_level_rle_pixel_exact() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 8, 8, LineOrder::IncreasingY, Compression::Rle, td);
    }

    #[test]
    fn off_origin_data_window_pixel_exact() {
        // Data window NOT at origin (min = (11, 7)) exercises the tile-rectangle
        // math (tileRange.min != 0) and the base_offset addressing.
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(11, 7, 5, 5, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    /// Mutation-testing blind-spot fix:
    /// every OTHER mipmap/ripmap test in this module (see
    /// `mipmap_none_pixel_exact`/`ripmap_none_pixel_exact` below) uses a
    /// data window at the origin, and the only off-origin test
    /// (`off_origin_data_window_pixel_exact` above) is ONE_LEVEL only --
    /// the combination "off-origin data window AND multiple mip levels"
    /// (where `level_size`/`data_window_for_level`'s per-level rounding
    /// math interacts with a non-zero `tileRange.min`) is never exercised
    /// anywhere. Also combines a genuinely partial edge tile (7x5 over
    /// 2x2 tiles: neither dimension divides evenly) with the off-origin
    /// window, matching the same "combine every blind spot in one test"
    /// technique that found real gaps in chunk/ and scanline/.
    #[test]
    fn off_origin_mipmap_partial_edge_tiles_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(11, 7, 7, 5, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    /// Same off-origin + partial-edge-tile combination, but RIPMAP (per-axis
    /// independent level counts) instead of MIPMAP -- `level_size`'s
    /// per-axis (not just per-level) rounding math is a different code path
    /// from mipmap's single shared level index.
    #[test]
    fn off_origin_ripmap_partial_edge_tiles_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::RipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(11, 7, 7, 5, LineOrder::IncreasingY, Compression::NoCompression, td);
    }

    /// Mutation-testing blind-spot fix: `LineOrder::RandomY` is valid for
    /// tiled files (unlike scanline, where only INCREASING_Y/DECREASING_Y
    /// are legal) -- output.rs already writes it in several tests, but
    /// input.rs (this file) has ZERO references to `RandomY` anywhere: no
    /// test confirms a RandomY-ordered tiled file round-trips correctly
    /// through the reader at all.
    #[test]
    fn random_y_round_trip_pixel_exact() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        round_trip_all(0, 0, 7, 5, LineOrder::RandomY, Compression::NoCompression, td);
    }

    #[test]
    fn read_single_non_origin_tile() {
        // Write a 4x4-tile ONE_LEVEL image, then read ONLY tile (1, 1) and
        // assert just that tile's pixels land correctly (rest stays zero).
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (min_x, min_y, w, h) = (0, 0, 4, 4);
        let header = make_header(min_x, min_y, w, h, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(min_x, min_y, w, h);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();

        let row_bytes = w as usize * 2;
        let mut readback = vec![0u8; row_bytes * h as usize];
        let (_p, rfb) = make_pixels(min_x, min_y, w, h);
        tif.set_frame_buffer(&rfb).unwrap();
        // Read only tile (1, 1) → covers pixels x in [2,3], y in [2,3].
        tif.read_tile(1, 1, 0, 0, &mut readback).unwrap();

        for y in 0..h {
            for x in 0..w {
                let off = y as usize * row_bytes + x as usize * 2;
                let got = u16::from_ne_bytes([readback[off], readback[off + 1]]);
                if (2..=3).contains(&x) && (2..=3).contains(&y) {
                    let want = Half::from((y * w + x) as f32).bits();
                    assert_eq!(got, want, "tile (1,1) pixel ({x},{y}) mismatch");
                } else {
                    assert_eq!(got, 0, "pixel ({x},{y}) outside tile (1,1) must be untouched");
                }
            }
        }
    }

    #[test]
    fn corrupt_tile_coords_error_not_panic() {
        // Write a valid file, corrupt the first tile chunk's tileX field, then
        // assert the reader errors (mismatch) rather than panicking.
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let mut bytes = out.into_data();

        // Locate tile (0,0)'s chunk via a throwaway reader, then clobber tileX.
        let chunk0 = {
            let mut is = MemIStream::new(bytes.clone());
            let tif = TiledInputFile::new(&mut is).unwrap();
            tif.state.tile_offsets.get(0, 0, 0, 0).unwrap() as usize
        };
        // tileX is the first i32 of the chunk header; write a bogus value (LE).
        bytes[chunk0..chunk0 + 4].copy_from_slice(&999i32.to_le_bytes());

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        let (_p, rfb) = make_pixels(0, 0, 4, 4);
        tif.set_frame_buffer(&rfb).unwrap();
        let err = tif.read_tile(0, 0, 0, 0, &mut [0u8; 4 * 4 * 2]);
        assert!(err.is_err(), "corrupt tile coords must error, not panic");
    }

    // ---------------------------------------------------------------------
    // compressTile/uncompressTile seam: rectangle-sensitive codec tiles.
    //
    // Before the seam the writer reused `compress(data, min_y)`, which derives
    // the chunk rectangle from the data-window X extent.  When the tile width
    // differs from the image width (always, for real tiled images) the codec
    // read/wrote the wrong rectangle and garbled every off-left-edge tile.
    // These tests drive tile_width < image_width AND a non-square image AND a
    // data window NOT at the origin, so they would FAIL without the seam.
    // ---------------------------------------------------------------------

    /// PIZ is lossless on the stored `u16` bits, so a multi-tile, non-square,
    /// off-origin round-trip must be **pixel-exact**.  Tile width (3) < image
    /// width (7), so the rectangle-sensitive path is exercised on every column
    /// of tiles.  `round_trip_all` asserts `readback == source` for level 0.
    #[test]
    fn one_level_piz_off_origin_non_square_pixel_exact() {
        let td = TileDescription {
            x_size: 3,
            y_size: 5,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // 7x9 image at origin (13, 4): 3 tile columns x 2 tile rows, none of
        // which start at the data-window X min except column 0.
        round_trip_all(13, 4, 7, 9, LineOrder::IncreasingY, Compression::Piz, td);
    }

    /// PIZ pixel-exact over a 3x3-tile grid (tile 34x34 over a 100x100 image),
    /// proving multi-tile correctness at scale.  The stored HALF bit pattern
    /// round-trips exactly regardless of HALF precision (PIZ is lossless).
    #[test]
    fn piz_multi_tile_100x100_pixel_exact() {
        let td = TileDescription {
            x_size: 34,
            y_size: 34,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // ceil(100/34) == 3 → a 3x3 tile grid.
        round_trip_all(0, 0, 100, 100, LineOrder::IncreasingY, Compression::Piz, td);
    }

    /// Smooth low-amplitude HALF plane over `[min_x..] x [min_y..]`.  Values are
    /// a gentle gradient in `[0, 1]` so the lossy codecs (B44/PXR24/DWA) stay
    /// within a tight tolerance.  Returns `(buffer, frame_buffer)`; addressing
    /// matches [`make_pixels`].
    fn make_smooth_pixels(
        min_x: i32,
        min_y: i32,
        width: i32,
        height: i32,
    ) -> (Vec<u8>, FrameBuffer) {
        let row_bytes = width as usize * 2;
        let mut buf = vec![0u8; row_bytes * height as usize];
        for ry in 0..height {
            for rx in 0..width {
                let off = ry as usize * row_bytes + rx as usize * 2;
                // Gradient in [0, 1): smooth so DCT/B44 block error is small.
                let f = (rx as f32 / width as f32 + ry as f32 / height as f32) * 0.5;
                let v = Half::from(f).bits();
                buf[off..off + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let x_stride = 2i64;
        let y_stride = row_bytes as i64;
        let base_offset = -(min_y as i64 * y_stride + min_x as i64 * x_stride);
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset,
                x_stride: 2,
                y_stride: row_bytes,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        (buf, fb)
    }

    /// Write every tile of a single-level smooth image with a lossy `comp`,
    /// read it back, and assert each decoded HALF value is within `tol` (max
    /// absolute error) of the source.  Tile width < image width and the data
    /// window is off-origin, so this fails (garbled tiles, errors way past
    /// `tol`) without the compressTile/uncompressTile seam.
    fn lossy_round_trip(
        min_x: i32,
        min_y: i32,
        width: i32,
        height: i32,
        comp: Compression,
        td: TileDescription,
        tol: f32,
    ) {
        let header = make_header(min_x, min_y, width, height, LineOrder::IncreasingY, comp, td);
        let (pixels, fb) = make_smooth_pixels(min_x, min_y, width, height);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let nx = tof.num_x_tiles(0).unwrap();
            let ny = tof.num_y_tiles(0).unwrap();
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        assert!(tif.is_complete(), "offset table must be complete ({comp:?})");

        let row_bytes = width as usize * 2;
        let mut readback = vec![0u8; row_bytes * height as usize];
        let (_p, rfb) = make_smooth_pixels(min_x, min_y, width, height);
        tif.set_frame_buffer(&rfb).unwrap();
        let nx = tif.num_x_tiles(0).unwrap();
        let ny = tif.num_y_tiles(0).unwrap();
        tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();

        let mut max_err = 0.0f32;
        for ry in 0..height as usize {
            for rx in 0..width as usize {
                let off = ry * row_bytes + rx * 2;
                let src = f32::from(Half::from_bits(u16::from_ne_bytes([
                    pixels[off],
                    pixels[off + 1],
                ])));
                let got = f32::from(Half::from_bits(u16::from_ne_bytes([
                    readback[off],
                    readback[off + 1],
                ])));
                max_err = max_err.max((src - got).abs());
            }
        }
        assert!(
            max_err <= tol,
            "{comp:?} tiled round-trip max abs error {max_err} exceeds tolerance {tol}"
        );
    }

    #[test]
    fn b44_off_origin_non_square_within_tolerance() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // B44 is exact on flat 4x4 blocks but lossy across gradients; the small
        // [0,1] gradient keeps error well under 0.05.
        lossy_round_trip(9, 5, 11, 7, Compression::B44, td, 0.05);
    }

    #[test]
    fn b44a_off_origin_non_square_within_tolerance() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        lossy_round_trip(9, 5, 11, 7, Compression::B44a, td, 0.05);
    }

    #[test]
    fn pxr24_off_origin_non_square_within_tolerance() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // PXR24 on HALF is lossless (it stores HALF verbatim), so the tolerance
        // is tiny; a non-zero bound still guards against rectangle garbling.
        lossy_round_trip(9, 5, 11, 7, Compression::Pxr24, td, 0.01);
    }

    #[test]
    fn dwaa_off_origin_non_square_within_tolerance() {
        let td = TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // DWAA's default lossy DCT on a smooth [0,1] gradient stays small.
        lossy_round_trip(9, 5, 20, 13, Compression::Dwaa, td, 0.1);
    }

    #[test]
    fn dwab_off_origin_non_square_within_tolerance() {
        let td = TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        lossy_round_trip(9, 5, 20, 13, Compression::Dwab, td, 0.1);
    }

    #[test]
    fn dwaa_tile_taller_than_dwa_block() {
        // Tile height 64 > the 32-row DWA block.  Regression for the tiled-DWAA
        // bug where `new_tile_compressor` passed a hardcoded 32-line scanline
        // count instead of the tile height, so only 32 of a 64-row tile decoded
        // ("size 20480 ≠ 40960").  Exercises the FULL TiledOutputFile →
        // TiledInputFile path (the codecs bench's path) at tile > DWA block,
        // including a partial edge tile (70 = 64 + 6).  The other DWAA tiled test
        // uses tile 8 (< 32), which never covered this.
        let td = TileDescription {
            x_size: 64,
            y_size: 64,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        lossy_round_trip(0, 0, 70, 70, Compression::Dwaa, td, 0.1);
    }

    #[test]
    fn no_frame_buffer_errors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        let err = tif.read_tile(0, 0, 0, 0, &mut [0u8; 4 * 4 * 2]);
        assert!(matches!(err, Err(ExrError::Arg(_))));
    }

    /// A fill slice with `yTileCoords = true`
    /// must write every row of the tile, not just row 0.  Rust's
    /// `scatter_fill_line` recomputed `y_offset` as the CURRENT row `y`
    /// instead of the tile's `min.y` (`input.rs::scatter_fill_line`), so
    /// `y - y_offset` collapsed to 0 for every row and every tile row landed
    /// on the buffer's row 0. C++: `ImfTiledInputFile.cpp` `run_fill`
    /// (v3.4.15) computes `yOffset = s.yTileCoords ? 0 : t_absY` once per
    /// tile then advances `ptr += s.yStride` per row, so tile-relative row r
    /// lands at `base + r*yStride`.
    ///
    /// 4x4 tiled file, 2x2 tiles, one HALF channel "R". Frame buffer adds a
    /// FLOAT slice "Z" absent from the file, `x_tile_coords = y_tile_coords =
    /// true`, sized for one 2x2 tile, `fill_value = 7.0`. Reading tile
    /// (dx=1, dy=1, lx=0, ly=0) must fill all four "Z" samples with 7, not
    /// just the first row.
    #[test]
    fn fill_slice_tile_coords_fills_all_rows() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();

        // One shared buffer: "R" (HALF, 2 bytes/px) at the front, "Z" (FLOAT,
        // 4 bytes/px, absent from the file -> fill slice) right after it,
        // both tile-coordinate-relative (base_offset addresses tile (0,0)).
        let r_bytes = 2 * 2 * 2;
        let z_x_stride = 4usize;
        let z_y_stride = 2 * z_x_stride;
        let mut combined = vec![0u8; r_bytes + 2 * 2 * 4];

        let mut rfb = FrameBuffer::new();
        rfb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset: 0,
                x_stride: 2,
                y_stride: 2 * 2,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: true,
                y_tile_coords: true,
            },
        )
        .unwrap();
        rfb.insert(
            "Z",
            Slice {
                type_: PixelType::Float,
                base_offset: r_bytes as i64,
                x_stride: z_x_stride,
                y_stride: z_y_stride,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 7.0,
                x_tile_coords: true,
                y_tile_coords: true,
            },
        )
        .unwrap();
        tif.set_frame_buffer(&rfb).unwrap();
        tif.read_tile(1, 1, 0, 0, &mut combined).unwrap();

        for row in 0..2usize {
            for col in 0..2usize {
                let off = r_bytes + row * z_y_stride + col * z_x_stride;
                let bits = u32::from_ne_bytes(combined[off..off + 4].try_into().unwrap());
                let v = f32::from_bits(bits);
                assert_eq!(v, 7.0, "row {row} col {col} was not filled with 7.0");
            }
        }
    }

    /// A fill slice (no matching file channel)
    /// with subsampling must be rejected by `set_frame_buffer`, matching C++
    /// 3.4.15 `ImfTiledInputFile.cpp` `run_fill` (:971-972,
    /// `ArgExc("Tiled data should not have subsampling.")`). Before the fix,
    /// `scatter_fill_line` ignored sampling entirely and wrote `tile_w`
    /// samples at the slice's stride regardless.
    #[test]
    fn fill_slice_with_subsampling_errors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();

        let mut rfb = FrameBuffer::new();
        rfb.insert(
            "Z",
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4,
                x_sampling: 2,
                y_sampling: 1,
                fill_value: 7.0,
                x_tile_coords: true,
                y_tile_coords: true,
            },
        )
        .unwrap();
        let err = tif.set_frame_buffer(&rfb);
        assert!(matches!(err, Err(ExrError::Arg(_))), "expected Arg error, got {err:?}");
    }

    /// `level_width`/`level_height`/
    /// `data_window_for_level` must reject an out-of-range level instead of
    /// silently returning a value (e.g. 1) computed from the level formula
    /// alone. C++ 3.4.15 routes these through `exr_get_level_sizes`
    /// (`OpenEXRCore/part.c:347-383`), which throws `ArgExc` for `levelx >=
    /// num_tile_levels_x`.
    #[test]
    fn out_of_range_level_errors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let mut out = MemOStream::new();
        {
            let (pixels, fb) = make_pixels(0, 0, 4, 4);
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();
        let mut is = MemIStream::new(bytes);
        let tif = TiledInputFile::new(&mut is).unwrap();

        // ONE_LEVEL: only level 0 is valid.
        assert!(tif.level_width(0).is_ok());
        assert!(matches!(tif.level_width(3), Err(ExrError::Arg(_))));
        assert!(matches!(tif.level_height(3), Err(ExrError::Arg(_))));
        assert!(matches!(tif.level_width(-1), Err(ExrError::Arg(_))));
        assert!(matches!(
            tif.data_window_for_level(3, 0),
            Err(ExrError::Arg(_))
        ));
        assert!(matches!(
            tif.data_window_for_level(0, 3),
            Err(ExrError::Arg(_))
        ));
        assert!(tif.data_window_for_level(0, 0).is_ok());
    }

    /// `file_name`, `tile_order`, and
    /// `raw_tile_data` accessors added to the reader.
    #[test]
    fn tiled_reader_accessors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();

        // `file_name` forwards the stream's own name.
        assert_eq!(tif.file_name(), "(memory)");

        // `tile_order`: 4 tiles, INCREASING_Y => file order is (0,0), (1,0),
        // (0,1), (1,1) — the same order `write_tiles` wrote them in.
        let (dx, dy, lx, ly) = tif.tile_order().unwrap();
        assert_eq!(dx, vec![0, 1, 0, 1]);
        assert_eq!(dy, vec![0, 0, 1, 1]);
        assert_eq!(lx, vec![0, 0, 0, 0]);
        assert_eq!(ly, vec![0, 0, 0, 0]);

        // `raw_tile_data`: NONE compression, so the raw bytes ARE the
        // uncompressed XDR tile data — pixel-exact against the expected plane.
        // Also asserts the in/out coordinate echo.
        let (r_dx, r_dy, r_lx, r_ly, raw) = tif.raw_tile_data(0, 0, 0, 0).unwrap();
        assert_eq!((r_dx, r_dy, r_lx, r_ly), (0, 0, 0, 0));
        assert_eq!(raw, expected_tile_xdr_for_test(0, 0, 1, 1));
        let err = tif.raw_tile_data(5, 5, 0, 0);
        assert!(matches!(err, Err(ExrError::Arg(_))));
    }

    /// Persisted read state: interleaved reads
    /// across two tiled "parts" must reuse the SAME persisted
    /// [`TiledInputFileState`] rather than losing it between accesses — the
    /// object-level property [`TiledInputPartReader`] exists to provide,
    /// which `MultiPartInputFile::tiled_input_part` builds a container
    /// around. Simulated here at
    /// the `tiled/input.rs`-owned level (two independent tiled files stand
    /// in for two container parts), independent of the `MultiPartInputFile`
    /// caching wire-up.
    ///
    /// Sequence: open part A, set its frame buffer, read tile (0,0) — drop
    /// that view, keeping only its `TiledInputFileState` (`into_state`,
    /// simulating what a container would persist). Open part B the same
    /// way. "Interleave" back to A via a fresh [`TiledInputPartReader`] over
    /// the SAME persisted state — and critically, WITHOUT calling
    /// `set_frame_buffer` again — read tile (1,0). This only succeeds if
    /// `slices`/`frame_buffer` survived in the state; a container that
    /// rebuilt a fresh `TiledInputFileState::from_part` for "part A" on this
    /// second access would come back with empty `slices` and
    /// `read_tile` would reject it with "No frame buffer specified", exactly
    /// as the control case below (`fresh_state`) demonstrates.
    #[test]
    fn tiled_input_part_reader_reuses_persisted_state_across_interleaved_access() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header_a = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let header_b = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::NoCompression, td);
        let (pixels_a, fb_a) = make_pixels(0, 0, 4, 4);
        let (pixels_b, fb_b) = make_pixels(0, 0, 4, 4); // same formula; distinguished by tile content below

        let mut out_a = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out_a, &header_a).unwrap();
            tof.set_frame_buffer(&fb_a).unwrap();
            tof.write_tiles(&pixels_a, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let mut out_b = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out_b, &header_b).unwrap();
            tof.set_frame_buffer(&fb_b).unwrap();
            tof.write_tiles(&pixels_b, 0, 1, 0, 1, 0, 0).unwrap();
        }

        let mut is_a = MemIStream::new(out_a.into_data());
        let mut is_b = MemIStream::new(out_b.into_data());

        // The READ side wants a self-contained, per-tile 8-byte buffer (to
        // match `expected_tile_xdr_for_test`'s compact layout) — a DIFFERENT
        // frame buffer than `fb_a`/`fb_b` above (which address pixels by
        // ABSOLUTE window position, `x/y_tile_coords: false`, needed for the
        // single whole-image WRITE above). With absolute addressing and a
        // width-4 image, even tile (0,0) touches byte range [10, 12) (row 1,
        // column 1: `1*y_stride(8) + 1*x_stride(2)`), so an 8-byte buffer is
        // too small for ANY tile except (0,0)'s first row — this is what
        // made the original version of this test panic (buffers sized "for
        // one 2x2 tile" while the frame buffer they were driven through was
        // actually window-relative). `x/y_tile_coords: true` makes the
        // address purely tile-relative, so every tile's own compact 2x2
        // buffer is self-contained regardless of which tile is read.
        fn tile_relative_read_fb() -> FrameBuffer {
            let mut fb = FrameBuffer::new();
            fb.insert(
                "R",
                Slice {
                    type_: PixelType::Half,
                    base_offset: 0,
                    x_stride: 2,
                    y_stride: 4, // 2 pixels/row * 2 bytes
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: true,
                    y_tile_coords: true,
                },
            )
            .unwrap();
            fb
        }

        // Open A: set frame buffer, read tile (0,0), then persist its state.
        let mut buf_a = vec![0u8; 8]; // one 2x2 HALF tile
        let mut state_a = {
            let mut tif = TiledInputFile::new(&mut is_a).unwrap();
            tif.set_frame_buffer(&tile_relative_read_fb()).unwrap();
            tif.read_tile(0, 0, 0, 0, &mut buf_a).unwrap();
            tif.into_state()
        };
        assert_eq!(buf_a, expected_tile_xdr_for_test(0, 0, 1, 1));

        // Open B the same way, independently.
        let mut buf_b = vec![0u8; 8];
        let mut state_b = {
            let mut tif = TiledInputFile::new(&mut is_b).unwrap();
            tif.set_frame_buffer(&tile_relative_read_fb()).unwrap();
            tif.read_tile(0, 0, 0, 0, &mut buf_b).unwrap();
            tif.into_state()
        };
        assert_eq!(buf_b, expected_tile_xdr_for_test(0, 0, 1, 1));

        // "Interleave" back to A via a fresh view over the SAME persisted
        // state — no `set_frame_buffer` call this time.
        {
            let mut view = TiledInputPartReader::new(&mut is_a, &mut state_a);
            let mut buf = vec![0u8; 8];
            view.read_tile(1, 0, 0, 0, &mut buf)
                .expect("persisted state must still carry the frame buffer set earlier");
            assert_eq!(buf, expected_tile_xdr_for_test(2, 0, 3, 1));
        }

        // Then back to B, same pattern, proving no cross-contamination
        // between the two persisted states.
        {
            let mut view = TiledInputPartReader::new(&mut is_b, &mut state_b);
            let mut buf = vec![0u8; 8];
            view.read_tile(1, 0, 0, 0, &mut buf)
                .expect("persisted state must still carry the frame buffer set earlier");
            assert_eq!(buf, expected_tile_xdr_for_test(2, 0, 3, 1));
        }

        // Load-bearing control: a state REBUILT from scratch (the bug
        // this test guards against) has empty `slices` and must reject a
        // read with no frame buffer set, proving the assertions above are
        // actually exercising persisted (not incidentally-still-working)
        // state.
        let mut is_a2 = MemIStream::new({
            let mut out = MemOStream::new();
            {
                let mut tof = TiledOutputFile::new(&mut out, &header_a).unwrap();
                tof.set_frame_buffer(&fb_a).unwrap();
                tof.write_tiles(&pixels_a, 0, 1, 0, 1, 0, 0).unwrap();
            }
            out.into_data()
        });
        let mut fresh_state = TiledInputFile::new(&mut is_a2).unwrap().into_state();
        let mut view = TiledInputPartReader::new(&mut is_a2, &mut fresh_state);
        let mut buf = vec![0u8; 8];
        let err = view.read_tile(1, 0, 0, 0, &mut buf);
        assert!(
            err.is_err(),
            "control case: a freshly-rebuilt state (no set_frame_buffer call) \
             must reject a read — proves the interleaved-access test above is \
             actually exercising PERSISTED state, not some incidental default"
        );
    }

    /// Expected XDR (LE) HALF bytes for the tile rectangle `[min_x..=max_x] x
    /// [min_y..=max_y]` of the `make_pixels`-style value(x,y)=(y*width+x)
    /// plane used by `tiled_reader_accessors` (width = 4, origin (0,0)).
    fn expected_tile_xdr_for_test(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Vec<u8> {
        let width = 4;
        let mut out = Vec::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let bits = Half::from((y * width + x) as f32).bits();
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        out
    }

    /// `TiledInputFile::open` OWNS a
    /// path-opened stream (no caller-side `&mut` to keep alive) -- port of
    /// C++'s real, public `TiledInputFile(const char*, int)` constructor.
    /// Asserts it reads pixel-identical to the borrowed `new` path on the
    /// same bytes.
    #[test]
    fn open_owns_a_path_opened_stream_and_reads_identically_to_new() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: crate::attr::tile_description::LevelRoundingMode::RoundDown,
        };
        let header = make_header(0, 0, 4, 4, LineOrder::IncreasingY, Compression::Zip, td);
        let (pixels, fb) = make_pixels(0, 0, 4, 4);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let nx = tof.num_x_tiles(0).unwrap();
            let ny = tof.num_y_tiles(0).unwrap();
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        let path = std::env::temp_dir()
            .join(format!("exr_core_tiled_open_{}.exr", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let mut owned = TiledInputFile::open(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mut is = MemIStream::new(bytes);
        let mut borrowed = TiledInputFile::new(&mut is).unwrap();

        assert_eq!(owned.header().data_window(), borrowed.header().data_window());
        assert!(owned.is_complete());

        let mut rfb_owned = FrameBuffer::new();
        let mut rfb_borrowed = FrameBuffer::new();
        for (name, _) in header.channels().unwrap().iter() {
            rfb_owned.insert(name, fb.find_slice(name).unwrap().clone()).unwrap();
            rfb_borrowed.insert(name, fb.find_slice(name).unwrap().clone()).unwrap();
        }
        owned.set_frame_buffer(&rfb_owned).unwrap();
        borrowed.set_frame_buffer(&rfb_borrowed).unwrap();

        let mut got_owned = vec![0u8; pixels.len()];
        let mut got_borrowed = vec![0u8; pixels.len()];
        owned.read_tiles(0, 1, 0, 1, 0, 0, &mut got_owned).unwrap();
        borrowed.read_tiles(0, 1, 0, 1, 0, 0, &mut got_borrowed).unwrap();
        assert_eq!(got_owned, got_borrowed, "owned and borrowed reads must match");
        assert_eq!(got_owned, pixels, "and must match the source pixels");
    }
}
