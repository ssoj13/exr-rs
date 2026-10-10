//! Port of OpenEXR `ImfTiledOutputFile.cpp` — `class TiledOutputFile`, the
//! single-part tiled writer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledOutputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledOutputFile.h>
//!
//! # What this module covers (1:1 with C++)
//!
//! * The constructor's *header + magic + version* write (reusing
//!   [`Header::write_to`], whose [`Header::version_field`] folds in the `TILED`
//!   flag automatically when the header carries a tile description), the
//!   `TiledOutputFile::initialize` bookkeeping (validate the tile description,
//!   precompute the per-level tile counts, build the [`TileOffsets`] table,
//!   construct the per-tile compressor), and the chunk-offset-table
//!   *reservation* ([`TileOffsets::write_to`] of the all-zero table).
//! * [`TiledOutputFile::set_frame_buffer`] — the `setFrameBuffer` checks (pixel
//!   type compatibility and the tiled rule that every channel must have
//!   sampling `(1,1)`) plus the per-channel `TOutSliceInfo` slice table
//!   (including the "channel absent → write zeroes" case).
//! * [`TiledOutputFile::write_tile`] / [`TiledOutputFile::write_tiles`] and the
//!   MIPMAP convenience overloads — for each tile in the requested rectangle it
//!   gathers every channel's samples out of the user frame buffer with
//!   [`copy_from_frame_buffer`] (or [`fill_channel_with_zeroes`] for absent
//!   channels) over the tile's pixel rectangle ([`data_window_for_tile`]), runs
//!   the gathered buffer through the per-tile compressor (or writes raw for
//!   `NONE`), writes the chunk (`tileX, tileY, levelX, levelY, dataSize, data`,
//!   exactly the byte layout of `writeTileData`), and back-patches the tile's
//!   offset into the reserved table slot.
//!
//! # Compressed-tile parity scope
//!
//! For `NONE` and the byte-stream codecs (`RLE`/`ZIP`/`ZIPS`, which ignore the
//! chunk rectangle) the written tile chunk is byte-identical to C++.  The
//! rectangle-sensitive codecs (`PIZ`/`B44`/`B44A`/`PXR24`/`DWAA`/`DWAB`) encode
//! against the tile's own pixel rectangle via the **tile compressor seam**
//! [`Compressor::compress_tile(data, tile_range)`][crate::compressor::Compressor::compress_tile]
//! — mirroring C++ `Compressor::compressTile`, which forwards the tile's
//! `Box2i range` into the encode step (ImfCompressor.cpp:118).  This fixes
//! off-origin tiles whose rectangle differs from the data-window X extent
//! (PIZ tiles are pixel-exact; the lossy codecs round-trip within tolerance).
//!
//! # Offset back-patch vs C++ `Drop`
//!
//! C++ rewrites the now-filled `u64` offset table from its destructor (it owns
//! the stream).  This port borrows the caller's [`OStream`] for its whole
//! lifetime, so — exactly like [`crate::scanline::OutputFile`] — it back-patches
//! each tile's offset immediately after the chunk is written (record the chunk
//! position into [`TileOffsets`], then seek to the reserved table slot, write
//! the `u64` LE offset, and seek back).  The file's structure is therefore valid once
//! every tile has been written, without relying on `Drop`; its bytes reach the file when the
//! stream is flushed — [`TiledOutputFile::finish`] checks completeness and flushes, returning
//! the error a destructor could only log.
//!
//! # Tile order on disk
//!
//! Tiles are written to the file in the order the caller invokes `write_tile` /
//! `write_tiles` for `RANDOM_Y`.  For `INCREASING_Y` / `DECREASING_Y`, a tile
//! written out of turn is *buffered*
//! (`tile_map` / `next_tile_to_write`, mirroring C++ `tileMap` /
//! `nextTileToWrite`) until every tile ahead of it in line order has been
//! written, so the on-disk chunk order matches in-order writing regardless
//! of call order — see [`Self::next_tile_coord`] / [`Self::write_single_tile`].
//!
//! # What this module intentionally does NOT cover
//!
//! * The `NATIVE`-format `convertToXdr` fallback: every ported codec reports
//!   [`Format::Xdr`], so the gather always produces Xdr and the codec's own
//!   "no-shrink → store raw" rule keeps the chunk byte-identical to C++.  If a
//!   `NATIVE` codec is ever added, the `convertToXdr` branch must be restored.

use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
use crate::attr::{LineOrder, PixelType};
use crate::compressor::{default_format, new_tile_compressor, Compressor, Format};
use crate::container::{xdr, SIZE_I32, SIZE_I64};
use crate::error::{ExrError, Result};
use crate::frame_buffer::FrameBuffer;
use crate::header::{Header, TILED_IMAGE};
use crate::io::OStream;
use crate::misc::{copy_from_frame_buffer, fill_channel_with_zeroes, BufferFormat};
use crate::tiled::misc::{
    calculate_bytes_per_pixel, data_window_for_level, data_window_for_tile, level_size,
    precalculate_tile_info,
};
use crate::tiled::input::TiledInputFile;
use crate::tiled::tile_offsets::TileOffsets;

use imath_rs::Box2i;
use std::marker::PhantomData;

/// A source [`Self::copy_pixels`] can transcode tiles from — the Rust
/// analogue of C++'s four `TiledOutputFile::copyPixels` overloads
/// (`TiledInputFile&`, `TiledInputPart&`, `InputFile&`, `InputPart&`).
/// [`TiledOutputFile::copy_pixels`] is generic over this trait instead of
/// four separate methods.
///
/// **Redesigned**: the
/// original shape had exactly ONE method, `as_tiled_input(&mut self) ->
/// Result<&mut TiledInputFile<'a>>` — every OTHER call `copy_pixels` makes
/// (`header`, `tile_order`, `raw_tile_data`) was routed through the concrete
/// `TiledInputFile` that method handed back, not through the trait itself.
/// That worked for a source that genuinely OWNS a `TiledInputFile` (the
/// standalone `TiledInputFile`/`InputFile`), but cannot be satisfied by a
/// BORROWING cache view (`TiledInputPartReader`, or the tiled-as-scanlines
/// cache view in `scanline::input_file`) — those never own a
/// `TiledInputFile` value to hand back a `&mut` to.
///
/// The trait now exposes exactly the THREE operations `copy_pixels` actually
/// performs, so any source — owning or borrowing — can implement it without
/// needing to produce a concrete `TiledInputFile` at all. No lifetime
/// parameter: none of these three methods return anything tied to the
/// source's own internal lifetime (unlike the old `as_tiled_input`, which
/// had to name it explicitly).
///
/// `crate::multipart::part::TiledInputPart` and
/// `crate::multipart::part::InputPart` implement this trait too (in
/// `multipart/part.rs`, which already depends on `tiled`) — Rust trait bounds
/// on a generic parameter are not satisfied by `Deref` coercion, unlike a
/// concrete-type call site.
pub trait TiledCopySource {
    /// Confirm this source really is tiled storage, or error — C++'s
    /// `asTiledInput()` check (a source that is ALWAYS tiled, like
    /// `TiledInputFile` itself, simply never errors here). Must be called
    /// once, before the other two methods, on a source whose storage might
    /// not be tiled (`InputFile`/`InputPart` and their cached-view
    /// equivalents); [`copy_pixels`](TiledOutputFileState::copy_pixels)
    /// calls it exactly once, up front.
    fn ensure_tiled(&mut self) -> Result<()>;

    /// This source's header.
    fn header(&self) -> &Header;

    /// The order every tile appears in on disk, sorted by file offset — C++
    /// `rawTileData`'s `tileOrder` companion, used to drive a `RANDOM_Y`
    /// copy in the source's own on-disk order.
    fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder>;

    /// Tile `(dx, dy, lx, ly)`'s raw (still-encoded) on-disk bytes, plus the
    /// chunk-header-verified coordinates — C++ `rawTileData`.
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)>;
}

impl<'a, S> TiledCopySource for TiledInputFile<'a, S>
where
    S: DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    fn ensure_tiled(&mut self) -> Result<()> {
        Ok(())
    }
    fn header(&self) -> &Header {
        TiledInputFile::header(self)
    }
    fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder> {
        TiledInputFile::tile_order(self)
    }
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        TiledInputFile::raw_tile_data(self, dx, dy, lx, ly)
    }
}

impl<'a, S> TiledCopySource for crate::scanline::InputFile<'a, S>
where
    S: DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    fn ensure_tiled(&mut self) -> Result<()> {
        crate::scanline::InputFile::as_tiled_input(self).map(|_| ())
    }
    fn header(&self) -> &Header {
        crate::scanline::InputFile::header(self)
    }
    fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder> {
        crate::scanline::InputFile::as_tiled_input(self)?.tile_order()
    }
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        crate::scanline::InputFile::as_tiled_input(self)?.raw_tile_data(dx, dy, lx, ly)
    }
}
use std::ops::DerefMut;

// ---------------------------------------------------------------------------
// TOutSliceInfo — port of the anonymous-namespace `struct TOutSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel write descriptor — port of `TOutSliceInfo`
/// (ImfTiledOutputFile.cpp lines 69-100).
///
/// Built by [`TiledOutputFile::set_frame_buffer`] from the file's channel list
/// and the user frame buffer.  Mirrors the C++ struct field-for-field, except
/// the raw `const char* base` becomes a signed byte offset (`base_offset`) into
/// the caller-owned pixel buffer — the same safe-Rust adaptation used by
/// [`crate::frame_buffer::Slice`].
#[derive(Clone, Debug)]
struct TOutSliceInfo {
    /// Sample type stored on disk for this channel.
    type_: PixelType,
    /// Signed byte offset that replaces the C++ `char* base` (origin already
    /// baked in; `0`/unused when `zero` is `true`).
    base_offset: i64,
    /// Distance in bytes between horizontally adjacent samples (`slice.xStride`).
    x_stride: usize,
    /// Distance in bytes between vertically adjacent rows (`slice.yStride`).
    y_stride: usize,
    /// `true` when the file channel has no matching frame-buffer slice; the
    /// tile buffer is filled with zeroes for this channel.
    zero: bool,
    /// `slice.xTileCoords` — when set, addresses are relative to the tile's
    /// min x (C++ `xOffset = xTileCoords * tileRange.min.x`).
    x_tile_coords: bool,
    /// `slice.yTileCoords` — when set, addresses are relative to the tile's
    /// min y (C++ `yOffset = yTileCoords * tileRange.min.y`).
    y_tile_coords: bool,
}

// ---------------------------------------------------------------------------
// TiledOutputFile — port of `class TiledOutputFile`
// ---------------------------------------------------------------------------

/// Single-part tiled EXR writer — port of `Imf::TiledOutputFile`.
///
/// Construct with [`TiledOutputFile::new`] (writes the header + reserves the
/// tile-offset table), describe the source pixels with
/// [`TiledOutputFile::set_frame_buffer`], then push tiles with
/// [`TiledOutputFile::write_tile`] / [`TiledOutputFile::write_tiles`].
///
/// The writer back-patches each tile-offset-table entry immediately after
/// writing the corresponding chunk, so the file is valid once all tiles have
/// been written (see the module docs on why `Drop` is not used).
///
/// # Stream ownership (borrowed vs owned)
///
/// The writer is generic over how it holds its output stream, `S`:
/// * the **default** `S = &'a mut dyn OStream` keeps the original
///   borrow-for-lifetime model (C++ `_streamData->os`) — `TiledOutputFile<'a>`
///   is unchanged and fully source-compatible; and
/// * `S = Box<dyn OStream>` (the [`TiledOutputFileOwned`] alias, built with
///   [`new_owned`](TiledOutputFile::new_owned)) lets a long-lived streaming
///   writer **own** its stream, so an enclosing struct can hold both the stream
///   and the writer without being self-referential (see the `vfx-io` streaming
///   output). Both variants share the identical write path and produce
///   byte-identical files.
pub struct TiledOutputFile<'a, S = &'a mut dyn OStream> {
    /// Output stream holder — a borrowed `&'a mut dyn OStream` (default) or an
    /// owned `Box<dyn OStream>`, accessed uniformly through
    /// `DerefMut<Target = dyn OStream>` (the C++ `_streamData->os`).
    stream: S,

    /// Everything else this writer owns — the "ownership inversion"
    /// design (mirroring
    /// `crate::scanline::OutputFileState`): factored into its own,
    /// non-generic, stream-free struct so [`crate::multipart::MultiPartOutputFile`]
    /// can persist it across separate [`crate::multipart::MultiPartOutputFile::tiled_output_part`]
    /// calls instead of rebuilding (and silently resetting `tile_map` /
    /// `next_tile_to_write`) a fresh writer every call. See
    /// [`TiledOutputFileState`].
    state: TiledOutputFileState,

    /// Ties the `'a` lifetime of the default borrowed-stream variant into the
    /// struct.  Zero-sized; for the owned variant (`S = Box<dyn OStream>`) `'a`
    /// is `'static` and this carries no borrow.
    _stream_lt: PhantomData<&'a mut ()>,
}

/// Everything a [`TiledOutputFile`] owns except its stream — the write-state
/// object [`crate::multipart::MultiPartOutputFile`] persists per tiled part
/// Has NO generic stream parameter and NO lifetime of its own: every
/// method that needs to write takes `stream: &mut dyn OStream` explicitly
/// (mirroring `crate::scanline::OutputFileState`), so the state can outlive
/// any one borrow of the container's stream.
pub struct TiledOutputFileState {
    /// The image header (type-fixed to `tiledimage` if it carried a `type`
    /// attribute) — C++ `_data->header`.
    header: Header,

    /// The current frame buffer — C++ `_data->frameBuffer`.
    frame_buffer: FrameBuffer,

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

    /// File position where the header's `preview` attribute's VALUE bytes
    /// begin, or `0` if the header has no preview image — C++
    /// `_data->previewPosition`. Feeds [`Self::update_preview_image`].
    /// Set by [`new_standalone`](Self::new_standalone) (the real value from
    /// `Header::write_to`) or by
    /// [`from_part_with_preview`](Self::from_part_with_preview) (the real
    /// value `MultiPartOutputFile` captured from `Header::write_attributes_to`);
    /// `0` only via the plain
    /// [`from_part`](Self::from_part), used by the public
    /// `TiledOutputFile::from_part` constructor, which does not itself write
    /// any header and so has no real value to supply.
    preview_position: u64,

    /// Compressor data format (Xdr for every ported codec) — C++
    /// `_data->format`.
    format: Format,

    /// Per-channel write descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][TiledOutputFile::set_frame_buffer] is called).
    ///
    /// `set_frame_buffer`
    /// stores only this LAYOUT (channel offsets/strides/types) — no pixel
    /// bytes and no lifetime-tied borrow. The pixel bytes are passed to each
    /// `write_tile`/`write_tiles` call instead (`pixels: &[u8]`), matching
    /// the C++ streaming pattern (one small buffer, refilled and re-passed
    /// per tile) directly instead of forcing the caller's buffer to outlive
    /// the writer. An earlier design stored `Option<&'a [u8]>` here; it was
    /// withdrawn because a stored borrow makes exactly that streaming
    /// pattern (and a fresh per-level MIP buffer each loop iteration)
    /// impossible to express.
    slices: Vec<TOutSliceInfo>,

    /// Byte size of one gathered tile, `maxBytesPerTileLine * tileDesc.ySize`
    /// (the C++ `TileBuffer::buffer` size). The buffer itself comes from the
    /// write call's `BufPool` and is never allocated or zero-filled up front.
    tile_buffer_size: usize,

    /// The per-tile compressor, or `None` for `NO_COMPRESSION` (the raw path) —
    /// mirrors the C++ `TileBuffer::compressor` built by `newTileCompressor`.
    compressor: Option<Box<dyn Compressor>>,

    /// Out-of-order write buffering —
    /// port of C++ `TiledOutputFile::Data::{tileMap, nextTileToWrite}`
    /// (`ImfTiledOutputFile.cpp:281-555`). For `INCREASING_Y`/`DECREASING_Y`,
    /// a tile written out of the file's line order is held here (its final
    /// compressed/raw bytes, keyed by `(dx, dy, lx, ly)`) until every tile
    /// before it in line order has been written, at which point it (and any
    /// further now-ready buffered tiles) are flushed in order. `RANDOM_Y`
    /// never buffers (`next_tile_to_write` stays `None`). A tile still
    /// buffered when the writer is dropped/`finish()`-ed is simply
    /// discarded (never written; its offset-table slot stays 0/missing) —
    /// matching C++'s destructor, which frees `tileMap` without flushing it.
    tile_map: std::collections::HashMap<(i32, i32, i32, i32), Vec<u8>>,

    /// The next tile coordinate, in the file's line order, that is ready to
    /// be written to disk — C++ `Data::nextTileToWrite`. `None` for
    /// `RANDOM_Y` (tiles are always written immediately in call order, C++'s
    /// existing behaviour). Initialized in `build()` to
    /// `(0,0,0,0)` for `INCREASING_Y` or `(0, numYTiles[0]-1, 0,0)` for
    /// `DECREASING_Y` (top row of level 0), advanced by
    /// [`Self::next_tile_coord`] after every tile actually written.
    next_tile_to_write: Option<(i32, i32, i32, i32)>,

    /// `Some(partNumber)` when this writer drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `TiledOutputFile::Data::{multipart, partNumber}`.
    ///
    /// This is the same seam the scanline writer has: when `Some`,
    /// every tile chunk is prefixed with the `partNumber` (i32 LE) exactly as
    /// C++ `writeTileData` does (`if (ofd->multipart) Xdr::write(partNumber)`,
    /// ImfTiledOutputFile.cpp:418-421); when `None`, no prefix is written, so
    /// single-part output is byte-identical to the pre-multipart writer.
    part_number: Option<i32>,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the scanline writer's identical
    /// field (`scanline/output.rs`).
    num_threads: crate::threading::ThreadCount,

    /// The codec (C++ `_data->_compression`) — kept so the parallel path can
    /// build a fresh per-worker codec instance via [`new_tile_compressor`]
    /// (each worker owns one; codecs are not `Sync`).
    compression: crate::attr::Compression,

    /// `maxBytesPerTileLine` — reused by the parallel path to size fresh
    /// per-worker codec instances.
    max_bytes_per_tile_line: usize,
}

/// A [`TiledOutputFile`] that **owns** its output stream (`Box<dyn OStream>`).
///
/// Use this when the writer must outlive the call that created its stream — a
/// long-lived streaming/region writer that owns both the stream and the writer
/// in one struct (impossible with the borrowing default without a
/// self-referential struct).  Build it with
/// [`TiledOutputFile::new_owned`]; the produced file is byte-identical to the
/// borrowed path for the same input.
///
/// Generic over `'a`: `'a` means "how long the pixel buffer(s)
/// passed to [`set_frame_buffer`](TiledOutputFile::set_frame_buffer) — and
/// the boxed stream itself — must live", the same meaning `'a` has for the
/// borrowed-stream default. Pinning it to `'static` would
/// make `set_frame_buffer` unusable with an ordinary local/heap buffer —
/// exactly the buffer a long-lived streaming writer refills every
/// iteration. The stream slot is `Box<dyn OStream + 'a>` rather than the
/// implicit `Box<dyn OStream + 'static>` so its `DerefMut::Target` is
/// literally `dyn OStream + 'a` (an exact match for `build`'s bound, not a
/// subtyping coercion i.e. `Box<dyn OStream + 'static>` can NOT satisfy
/// `S: DerefMut<Target = dyn OStream + 'a>` for a free `'a` — the compiler
/// rejects it as "`'a` must outlive `'static`"; verified by trying it).
/// Ordinary `'static` streams (`StdOfStream`, `MemOStream`) still coerce
/// into a `Box<dyn OStream + 'a>` argument at the `new_owned` call site.
pub type TiledOutputFileOwned<'a> = TiledOutputFile<'a, Box<dyn OStream + 'a>>;

// Borrowed-stream constructors (the default `S = &'a mut dyn OStream`).
impl<'a> TiledOutputFile<'a, &'a mut dyn OStream> {
    /// Open a `TiledOutputFile` on `stream` for `header` — port of
    /// `TiledOutputFile::TiledOutputFile(OStream&, const Header&, int)` and
    /// `TiledOutputFile::initialize`.
    ///
    /// Validates the header (`sanity_check(true, …)` — tiled), computes the
    /// level / tile counts, builds the offset table, writes the magic number,
    /// version word and header, then reserves the tile-offset table (writes
    /// every still-zero offset).
    ///
    /// # Errors
    /// * Propagates [`Header::sanity_check`] failures (incl. a missing tile
    ///   description — `sanity_check(true, …)` requires one).
    /// * Returns the [`new_tile_compressor`] error for any compression not yet
    ///   ported (nothing is faked).
    /// * [`ExrError::Arg`] if the tile size exceeds the OpenEXR `INT_MAX` byte
    ///   limit, matching the C++ throw.
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
        // Single-part: the writer owns the file, so `new_standalone` writes
        // magic + version + header + reserves the tile-offset table, and no
        // per-chunk partNumber prefix (`part_number == None`).
        let state = TiledOutputFileState::new_standalone_mt(&mut *stream, header, num_threads)?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }

    /// Open a `TiledOutputFile` for **one part of a multipart file** — port of
    /// `TiledOutputFile::TiledOutputFile(const OutputPartData*)`.
    ///
    /// The enclosing [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile)
    /// has already written the file's single magic + version word, every part
    /// header, and the (zero-filled) per-part tile-offset tables.  This
    /// constructor therefore does **not** write any of those; it only rebuilds
    /// the per-part write state and remembers `part_number` (the i32 prefix of
    /// every tile chunk) and `tile_offsets_position` (the file position of
    /// *this part's* reserved offset table, back-patched per tile).
    ///
    /// The stream cursor must already be positioned where this part's first
    /// tile chunk should be written (the container leaves it just past the last
    /// reserved offset table).
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledOutputFile.cpp> (`TiledOutputFile(const OutputPartData*)` + `writeTileData`).
    pub fn from_part(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
    ) -> Result<Self> {
        // C++ 3.4.15
        // `TiledOutputFile::TiledOutputFile(const OutputPartData* part)`
        // (`ImfTiledOutputFile.cpp:911-917`) checks
        // `part->header.type() != TILEDIMAGE` and throws
        // `ArgExc("Can't build a TiledOutputFile from a type-mismatched
        // part.")` BEFORE calling `initialize`. `initialize`'s own
        // "fix the type if it happens to be set incorrectly" (`:963`,
        // `hasType() { setType(TILEDIMAGE) }`, ported as `build`'s
        // `hdr.set_type(TILED_IMAGE)` below) is for the OTHER constructor
        // (single-part `new`, which never sees an existing wrong type from
        // a container). Only this multipart-part path skipped the check —
        // the normal `part_typed` container guard means it is reachable
        // only via a direct call, but a direct call must fail the same way
        // C++ does rather than silently retype e.g. a `deeptile` part.
        let part_type = header.type_()?;
        if part_type != TILED_IMAGE {
            return Err(ExrError::Arg(
                "Can't build a TiledOutputFile from a type-mismatched part.".into(),
            ));
        }
        let state =
            TiledOutputFileState::from_part(header, part_number, tile_offsets_position)?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }

    /// Same as [`Self::from_part`], with an explicit `num_threads` override.
    pub fn from_part_mt(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let part_type = header.type_()?;
        if part_type != TILED_IMAGE {
            return Err(ExrError::Arg(
                "Can't build a TiledOutputFile from a type-mismatched part.".into(),
            ));
        }
        let state = TiledOutputFileState::from_part_mt(
            header,
            part_number,
            tile_offsets_position,
            num_threads,
        )?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }
}

// Owned-stream constructor (`S = Box<dyn OStream + 'a>`).  Generic over `'a`
// — see the [`TiledOutputFileOwned`] doc for why this is not pinned to
// `'static`.
impl<'a> TiledOutputFile<'a, Box<dyn OStream + 'a>> {
    /// Open a [`TiledOutputFileOwned`] that **owns** its output stream — the
    /// owned-stream analogue of [`new`](TiledOutputFile::new).
    ///
    /// Writes the header + reserves the tile-offset table exactly as
    /// [`new`](TiledOutputFile::new) does (byte-identical output), but takes the
    /// stream by value so the returned writer can be stored in a struct that
    /// also owns the stream — the long-lived streaming-writer use case that the
    /// borrowing default cannot express in safe Rust.  Pass a boxed concrete
    /// stream, e.g. `Box::new(StdOfStream::create(path)?)`.
    ///
    /// The file is valid as soon as every tile has been written (offsets are
    /// back-patched per tile); there is no separate `finish` step, though
    /// [`finish`](TiledOutputFile::finish) is provided to recover the stream.
    ///
    /// # Errors
    /// Identical to [`new`](TiledOutputFile::new).
    pub fn new_owned(mut stream: Box<dyn OStream + 'a>, header: &Header) -> Result<Self> {
        let state = TiledOutputFileState::new_standalone(&mut *stream, header)?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }
}

// Ownership inversion: the write-state algorithm body lives
// on `TiledOutputFileState`, which owns no stream and no lifetime of its
// own — this is what lets `crate::multipart::MultiPartOutputFile` persist it
// across separate `tiled_output_part(n)` calls (mirrors
// `crate::scanline::OutputFileState`). Every method that must write takes
// `stream: &mut dyn OStream` explicitly instead of reading a `self.stream`
// field.
/// The fields [`TiledOutputFileState::compute`] derives, shared verbatim by
/// every constructor path — a named struct instead of a long tuple so a
/// mutation-testing tool doesn't have to enumerate the product of every
/// element's candidate return values (a 19-element tuple made
/// `cargo mutants --list` blow up). Field names match [`TiledOutputFileState`]'s own fields 1:1 where
/// they overlap, so callers destructure it with a plain `let Self { .. } =`.
struct TiledOutputComputed {
    header: Header,
    tile_desc: TileDescription,
    line_order: LineOrder,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    num_x_levels: i32,
    num_y_levels: i32,
    num_x_tiles: Vec<i32>,
    num_y_tiles: Vec<i32>,
    tile_offsets: TileOffsets,
    tile_buffer_size: usize,
    compressor: Option<Box<dyn Compressor>>,
    format: Format,
    next_tile_to_write: Option<(i32, i32, i32, i32)>,
    num_threads: crate::threading::ThreadCount,
    compression: crate::attr::Compression,
    max_bytes_per_tile_line: usize,
}

/// The read-only level geometry C++'s `Data::nextTileCoord` needs. Shared by
/// the flat and deep tiled writers (one copy of the walk) and borrowed from
/// the writer state's fields, so it can sit next to a mutable [`TileSink`].
#[derive(Clone, Copy)]
pub(crate) struct TileGeom<'a> {
    pub(crate) tile_desc: TileDescription,
    pub(crate) line_order: LineOrder,
    pub(crate) num_x_levels: i32,
    pub(crate) num_y_levels: i32,
    pub(crate) num_x_tiles: &'a [i32],
    pub(crate) num_y_tiles: &'a [i32],
}

impl TileGeom<'_> {
    /// The successor of tile coordinate `a` in the file's line order — port
    /// of C++ `TiledOutputFile::Data::nextTileCoord`
    /// (`ImfTiledOutputFile.cpp:280-330`). Walks `dx` first, then wraps into
    /// `dy` (up for `INCREASING_Y`, down for `DECREASING_Y`), then wraps into
    /// the next level (`(lx+1, ly+1)` for `ONE_LEVEL`/`MIPMAP_LEVELS`; `lx+1`
    /// wrapping into `ly+1` at `lx == numXLevels` for `RIPMAP_LEVELS`).
    ///
    /// The RESULT after the very last tile of the very last level is a
    /// sentinel coordinate that addresses no real tile (mirrors C++, which
    /// does not bounds-check this — no further tile write, buffered-tile
    /// lookup, or `next_tile_coord` call ever consults it once every tile has
    /// been written). Every `dx`/`dy` INPUT here is either the file's own
    /// starting coordinate or a value this function itself already produced
    /// from a real, in-range tile, so the bounds lookups below never fail in
    /// practice; they still return `Result` rather than panicking, matching
    /// this crate's no-panic rule for a public-reachable code path (only
    /// reachable via `write_tile`/`write_tiles`, so a corrupt internal state
    /// surfaces as a clean error instead of an index panic).
    ///
    /// Only valid to call for `INCREASING_Y`/`DECREASING_Y` (`RANDOM_Y` never
    /// tracks a "next" tile).
    pub(crate) fn next(&self, a: (i32, i32, i32, i32)) -> Result<(i32, i32, i32, i32)> {
        let (mut dx, mut dy, mut lx, mut ly) = a;
        let num_x_tiles_at = |lx: i32| -> Result<i32> {
            self.num_x_tiles
                .get(lx as usize)
                .copied()
                .ok_or_else(|| ExrError::Logic("next_tile_coord: level x out of range".into()))
        };
        let num_y_tiles_at = |ly: i32| -> Result<i32> {
            self.num_y_tiles
                .get(ly as usize)
                .copied()
                .ok_or_else(|| ExrError::Logic("next_tile_coord: level y out of range".into()))
        };
        let wrap_level = |lx: &mut i32, ly: &mut i32| -> Result<()> {
            match self.tile_desc.mode {
                LevelMode::OneLevel | LevelMode::MipmapLevels => {
                    *lx += 1;
                    *ly += 1;
                }
                LevelMode::RipmapLevels => {
                    *lx += 1;
                    if *lx >= self.num_x_levels {
                        *lx = 0;
                        *ly += 1;
                    }
                }
                LevelMode::Invalid => {
                    return Err(ExrError::Arg("Invalid tile description".into()));
                }
            }
            Ok(())
        };

        match self.line_order {
            LineOrder::IncreasingY => {
                dx += 1;
                if dx >= num_x_tiles_at(lx)? {
                    dx = 0;
                    dy += 1;
                    if dy >= num_y_tiles_at(ly)? {
                        dy = 0;
                        wrap_level(&mut lx, &mut ly)?;
                    }
                }
            }
            LineOrder::DecreasingY => {
                dx += 1;
                if dx >= num_x_tiles_at(lx)? {
                    dx = 0;
                    dy -= 1;
                    if dy < 0 {
                        wrap_level(&mut lx, &mut ly)?;
                        if ly < self.num_y_levels {
                            dy = num_y_tiles_at(ly)? - 1;
                        }
                    }
                }
            }
            LineOrder::RandomY | LineOrder::Invalid => {
                return Err(ExrError::Logic(
                    "next_tile_coord called for a line order with no tile ordering".into(),
                ));
            }
        }

        Ok((dx, dy, lx, ly))
    }
}

/// The emit-side state of the tiled writer ([`TileSink::feed`] is the
/// `bufferedTileWrite` state machine), borrowed field-by-field from
/// [`TiledOutputFileState::split`] so the encode workers can keep reading the
/// header and layout while tiles are fed.
struct TileSink<'a> {
    geom: TileGeom<'a>,
    tile_offsets: &'a mut TileOffsets,
    tile_offsets_position: u64,
    part_number: Option<i32>,
    tile_map: &'a mut std::collections::HashMap<(i32, i32, i32, i32), Vec<u8>>,
    next_tile_to_write: &'a mut Option<(i32, i32, i32, i32)>,
}

impl TiledOutputFileState {
    /// Pure computation shared by every constructor path: validates the
    /// header, computes level/tile counts, builds the tile-offset table and
    /// per-tile compressor/buffer. Does NOT touch a stream and does NOT set
    /// `tile_offsets_position`/`preview_position` (the two fields that differ
    /// between a standalone writer, which discovers them by writing the
    /// prologue, and a container part, which is handed them already) —
    /// [`Self::new_standalone`] and [`Self::from_part`] each fill those in.
    fn compute(header: &Header, num_threads: Option<i32>) -> Result<TiledOutputComputed> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // C++: header.sanityCheck(true);  (tiled)
        header.sanity_check(true, false)?;

        // --- C++ TiledOutputFile::initialize(header) -----------------------
        let mut hdr = header.clone();
        let line_order = hdr.line_order()?;
        let tile_desc = hdr.tile_description()?;

        // "fix" the type if it happens to be set incorrectly.
        // C++: if (hasType()) setType(TILEDIMAGE);
        if hdr.has_type() {
            hdr.set_type(TILED_IMAGE)?;
        }

        let data_window = hdr.data_window()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        // C++: precalculateTileInfo(tileDesc, minX..maxY, numXTiles, numYTiles,
        //      numXLevels, numYLevels);
        let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
            precalculate_tile_info(&tile_desc, min_x, max_x, min_y, max_y)?;

        // C++: maxBytesPerTileLine = calculateBytesPerPixel(header) * tileDesc.xSize;
        //      tileBufferSize       = maxBytesPerTileLine * tileDesc.ySize;
        let max_bytes_per_tile_line =
            calculate_bytes_per_pixel(&hdr)?.saturating_mul(tile_desc.x_size as usize);
        let tile_buffer_size = max_bytes_per_tile_line.saturating_mul(tile_desc.y_size as usize);

        // C++: if (tileBufferSize > INT_MAX) throw ArgExc("Tile size too large …").
        if tile_buffer_size > i32::MAX as usize {
            return Err(ExrError::Arg("Tile size too large for OpenEXR format".into()));
        }

        // C++: newTileCompressor(compression, maxBytesPerTileLine, tileDesc.ySize, header).
        // For NONE this is Ok(None); any codec not yet ported is an Err we
        // propagate (never faked).  numTileLines == tileDesc.ySize.
        let compression = hdr.compression()?;
        let compressor = new_tile_compressor(
            compression,
            max_bytes_per_tile_line,
            tile_desc.y_size as usize,
            &hdr,
        )?;

        // C++: format = defaultFormat(tileBuffers[0]->compressor);  (Xdr when none).
        let format = default_format(compressor.as_deref());

        // C++: tileOffsets = TileOffsets(mode, numXLevels, numYLevels,
        //      numXTiles, numYTiles);
        let tile_offsets = TileOffsets::new(
            tile_desc.mode,
            num_x_levels,
            num_y_levels,
            &num_x_tiles,
            &num_y_tiles,
        )?;

        // C++ `TiledOutputFile::initialize`
        // (`ImfTiledOutputFile.cpp:993-996`) — "Determine the first tile
        // coordinate that we will be writing if the file is not RANDOM_Y."
        // `RANDOM_Y` never buffers, so it has no next-tile tracking.
        let next_tile_to_write = match line_order {
            LineOrder::IncreasingY => Some((0, 0, 0, 0)),
            LineOrder::DecreasingY => {
                Some((0, num_y_tiles.first().copied().unwrap_or(1).saturating_sub(1), 0, 0))
            }
            LineOrder::RandomY | LineOrder::Invalid => None,
        };

        // Structural detail (debug): tiled writer is up — dimensions, tile size,
        // level mode, level counts, and the selected codec.
        log::debug!(
            "tiled OutputFile: {}x{} tile={}x{} mode={:?} levels={}x{} codec={compression:?}",
            max_x - min_x + 1,
            max_y - min_y + 1,
            tile_desc.x_size,
            tile_desc.y_size,
            tile_desc.mode,
            num_x_levels,
            num_y_levels
        );

        Ok(TiledOutputComputed {
            header: hdr,
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
            tile_buffer_size,
            compressor,
            format,
            next_tile_to_write,
            num_threads,
            compression,
            max_bytes_per_tile_line,
        })
    }

    /// Build the write state for a **standalone** writer — port of
    /// `TiledOutputFile::TiledOutputFile(OStream&, const Header&, int)` +
    /// `initialize`. Writes the file prologue (magic + version + header,
    /// then the reserved tile-offset table) here, since this writer owns the
    /// whole file (C++: `writeMagicNumberAndVersionField`, `header.writeTo`,
    /// `tileOffsets.writeTo`). No per-chunk `partNumber` prefix
    /// (`part_number == None`).
    fn new_standalone(stream: &mut dyn OStream, header: &Header) -> Result<Self> {
        Self::new_standalone_mt(stream, header, None)
    }

    /// Same as [`Self::new_standalone`], with an explicit `num_threads`
    /// override (`None` = capture the current process-wide
    /// default).
    fn new_standalone_mt(
        stream: &mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let computed = Self::compute(header, num_threads)?;

        // `Header::write_to` folds in the TILED flag automatically (the
        // header carries a tile description), so base_flags = 0 is correct.
        let preview_position = computed.header.write_to(stream, 0)?;
        // `TileOffsets::write_to` records the current position then writes
        // every (still-zero) offset; the returned position is the table start.
        let tile_offsets_position = computed.tile_offsets.write_to(stream)?;

        Ok(Self {
            header: computed.header,
            frame_buffer: FrameBuffer::new(),
            tile_desc: computed.tile_desc,
            line_order: computed.line_order,
            min_x: computed.min_x,
            max_x: computed.max_x,
            min_y: computed.min_y,
            max_y: computed.max_y,
            num_x_levels: computed.num_x_levels,
            num_y_levels: computed.num_y_levels,
            num_x_tiles: computed.num_x_tiles,
            num_y_tiles: computed.num_y_tiles,
            tile_offsets: computed.tile_offsets,
            tile_offsets_position,
            preview_position,
            format: computed.format,
            slices: Vec::new(),
            tile_buffer_size: computed.tile_buffer_size,
            compressor: computed.compressor,
            tile_map: std::collections::HashMap::new(),
            next_tile_to_write: computed.next_tile_to_write,
            num_threads: computed.num_threads,
            compression: computed.compression,
            max_bytes_per_tile_line: computed.max_bytes_per_tile_line,
            part_number: None,
        })
    }

    /// Build the write state for **one part of a multipart file** — port of
    /// `TiledOutputFile::TiledOutputFile(const OutputPartData*)`. Pure (no
    /// stream access): the enclosing container has already written the file
    /// prologue and this part's reserved offset table, at
    /// `tile_offsets_position`; this only rebuilds the per-part write state
    /// and remembers `part_number` (the i32 prefix of every tile chunk).
    ///
    /// `pub` (not `pub(crate)`): mirrors
    /// `crate::scanline::OutputFileState::from_part`, called directly by
    /// [`crate::multipart::MultiPartOutputFile::tiled_output_part`] to build
    /// the persisted per-part state.
    pub fn from_part(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
    ) -> Result<Self> {
        // `preview_position` stays 0: the public `TiledOutputFile::from_part`
        // constructor (this function's only OTHER caller besides
        // `MultiPartOutputFile`) does not itself write any header, so it has
        // no real value to supply. `MultiPartOutputFile` calls
        // `from_part_with_preview` instead.
        Self::from_part_with_preview_mt(header, part_number, tile_offsets_position, 0, None)
    }

    /// Same as [`Self::from_part`], with an explicit `num_threads` override.
    pub fn from_part_mt(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::from_part_with_preview_mt(header, part_number, tile_offsets_position, 0, num_threads)
    }

    /// Like [`from_part`](Self::from_part), but additionally records
    /// `preview_position`: C++
    /// `ImfMultiPartOutputFile.cpp:474-477` captures
    /// `parts[i]->previewPosition = headers[i].writeTo(..)` for every part
    /// while writing the header list, so `updatePreviewImage` works on a
    /// multipart part exactly as it does on a standalone file.
    /// [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile) calls
    /// this (not [`from_part`](Self::from_part)) with the value
    /// `Header::write_attributes_to` returned for this part.
    pub fn from_part_with_preview(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        preview_position: u64,
    ) -> Result<Self> {
        Self::from_part_with_preview_mt(header, part_number, tile_offsets_position, preview_position, None)
    }

    /// Same as [`Self::from_part_with_preview`], with an explicit
    /// `num_threads` override.
    pub fn from_part_with_preview_mt(
        header: &Header,
        part_number: Option<i32>,
        tile_offsets_position: u64,
        preview_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let computed = Self::compute(header, num_threads)?;

        Ok(Self {
            header: computed.header,
            frame_buffer: FrameBuffer::new(),
            tile_desc: computed.tile_desc,
            line_order: computed.line_order,
            min_x: computed.min_x,
            max_x: computed.max_x,
            min_y: computed.min_y,
            max_y: computed.max_y,
            num_x_levels: computed.num_x_levels,
            num_y_levels: computed.num_y_levels,
            num_x_tiles: computed.num_x_tiles,
            num_y_tiles: computed.num_y_tiles,
            tile_offsets: computed.tile_offsets,
            tile_offsets_position,
            preview_position,
            format: computed.format,
            slices: Vec::new(),
            tile_buffer_size: computed.tile_buffer_size,
            compressor: computed.compressor,
            tile_map: std::collections::HashMap::new(),
            next_tile_to_write: computed.next_tile_to_write,
            num_threads: computed.num_threads,
            compression: computed.compression,
            max_bytes_per_tile_line: computed.max_bytes_per_tile_line,
            part_number,
        })
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk — port of `void updatePreviewImage(const PreviewRgba
    /// newPixels[])` (C++ `ImfTiledOutputFile.cpp:1824-1873`).
    ///
    /// `new_pixels` must have exactly as many entries as the header's
    /// existing preview image (`width * height`) — C++ trusts the caller's
    /// raw pointer to cover that many pixels (a length mismatch there is
    /// undefined behaviour); this port returns an error instead of reading
    /// past `new_pixels` or leaving part of the preview stale.
    ///
    /// # Errors
    /// * [`ExrError::Logic`] if the file has no preview image (`LogicExc` in
    ///   C++).
    /// * [`ExrError::Arg`] if the header has no `"preview"` attribute of the
    ///   expected type, or `new_pixels.len()` does not match the existing
    ///   preview's pixel count.
    fn update_preview_image(
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

        // `PreviewImage::write_value` ignores its `version` parameter (fixed
        // wire layout regardless of file version — see its own doc); no
        // writer-tracked version word is needed here.
        let saved = stream.tellp()?;
        stream.seekp(self.preview_position)?;
        pia.value.write_value(stream, 0)?;
        stream.seekp(saved)?;
        Ok(())
    }

    /// Overwrite `length` bytes of an already-written tile's chunk, starting
    /// `offset` bytes into it, with the byte `c` — port of `void breakTile(int
    /// dx, int dy, int lx, int ly, int offset, int length, char c)`
    /// (test-only helper for exercising corrupt-chunk error paths,
    /// `ImfTiledOutputFile.cpp:1878-1901`).
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if the tile has not been written yet
    /// (offset-table slot still 0), matching the C++ throw.
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ breakTile's parameter list
    fn break_tile(
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
                "Cannot overwrite tile ({dx}, {dy}, {lx},{ly}). The tile has not \
                 yet been stored in file \"{}\".",
                stream.file_name()
            )));
        }

        stream.seekp(position + offset)?;
        let fill = vec![c; length];
        stream.write(&fill)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors — thin wrappers (ImfTiledOutputFile.cpp lines 1087-1812)
    // -----------------------------------------------------------------------

    /// The image header — port of `const Header& header() const`.
    fn header(&self) -> &Header {
        &self.header
    }

    /// The current frame buffer — port of `const FrameBuffer& frameBuffer() const`.
    fn frame_buffer(&self) -> &FrameBuffer {
        &self.frame_buffer
    }

    /// Tile width — port of `unsigned int tileXSize() const`.
    fn tile_x_size(&self) -> u32 {
        self.tile_desc.x_size
    }

    /// Tile height — port of `unsigned int tileYSize() const`.
    fn tile_y_size(&self) -> u32 {
        self.tile_desc.y_size
    }

    /// Level mode — port of `LevelMode levelMode() const`.
    fn level_mode(&self) -> LevelMode {
        self.tile_desc.mode
    }

    /// Level rounding mode — port of `LevelRoundingMode levelRoundingMode() const`.
    fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.tile_desc.rounding_mode
    }

    /// Compression type — convenience accessor (C++ reads it off `header()`).
    fn compression(&self) -> Result<crate::attr::Compression> {
        self.header.compression()
    }

    /// Number of levels in X — port of `int numXLevels() const`.
    fn num_x_levels(&self) -> i32 {
        self.num_x_levels
    }

    /// Number of levels in Y — port of `int numYLevels() const`.
    fn num_y_levels(&self) -> i32 {
        self.num_y_levels
    }

    /// Number of levels (MIP convenience) — port of `int numLevels() const`.
    ///
    /// Returns [`ExrError::Logic`] for `RIPMAP_LEVELS` (undefined there), exactly
    /// as the C++ throws `LogicExc`.
    fn num_levels(&self) -> Result<i32> {
        if self.level_mode() == LevelMode::RipmapLevels {
            return Err(ExrError::Logic(
                "numLevels() is not defined for RIPMAPs.".into(),
            ));
        }
        Ok(self.num_x_levels)
    }

    /// Whether `(lx, ly)` is a valid level — port of `bool isValidLevel(int, int) const`.
    fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
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
    fn level_width(&self, lx: i32) -> Result<i32> {
        level_size(self.min_x, self.max_x, lx, self.tile_desc.rounding_mode)
    }

    /// Height of level `ly` — port of `int levelHeight(int ly) const`.
    fn level_height(&self, ly: i32) -> Result<i32> {
        level_size(self.min_y, self.max_y, ly, self.tile_desc.rounding_mode)
    }

    /// Number of X tiles in level `lx` — port of `int numXTiles(int lx) const`.
    ///
    /// Returns [`ExrError::Logic`] when `lx` is out of range (C++ `LogicExc`).
    fn num_x_tiles(&self, lx: i32) -> Result<i32> {
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
    fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.num_y_tiles
            .get(usize::try_from(ly).map_err(|_| {
                ExrError::Logic("numYTiles(): argument is not in valid range".into())
            })?)
            .copied()
            .ok_or_else(|| ExrError::Logic("numYTiles(): argument is not in valid range".into()))
    }

    /// Data window of level `(lx, ly)` — port of
    /// `Box2i dataWindowForLevel(int lx, int ly) const`.
    fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        data_window_for_level(&self.tile_desc, self.min_x, self.max_x, self.min_y, self.max_y, lx, ly)
    }

    /// Data window of level `l` (MIP convenience) — port of
    /// `Box2i dataWindowForLevel(int l) const`, which forwards to `(l, l)`.
    fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.data_window_for_level(l, l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` — port of
    /// `Box2i dataWindowForTile(int dx, int dy, int lx, int ly) const`.
    ///
    /// Returns [`ExrError::Arg`] for an invalid tile (C++ `ArgExc`).
    fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
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
    fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.data_window_for_tile(dx, dy, l, l)
    }

    /// Whether `(dx, dy, lx, ly)` addresses an existing tile — port of the
    /// private `bool TiledOutputFile::isValidTile(int, int, int, int) const`.
    ///
    /// This is the writer's own bounds check against the precomputed per-level
    /// tile counts (distinct from [`TileOffsets::is_valid_tile`], which checks
    /// the jagged offset array).
    fn is_valid_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> bool {
        if lx < 0 || lx >= self.num_x_levels || ly < 0 || ly >= self.num_y_levels {
            return false;
        }
        // num_x_tiles[lx] / num_y_tiles[ly] exist for every valid lx/ly.
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
    // setFrameBuffer — ImfTiledOutputFile.cpp lines 1093-1179
    // -----------------------------------------------------------------------

    /// Set the pixel LAYOUT (channel offsets/strides/types) — port of
    /// `TiledOutputFile::setFrameBuffer`.
    ///
    /// This stores only the layout, not
    /// pixel bytes — `write_tile`/`write_tiles` take the bytes per call
    /// (`pixels: &[u8]`), matching C++'s streaming pattern (one small
    /// buffer, refilled and re-passed per tile) instead of forcing the
    /// caller's buffer to outlive the writer. Performs the same
    /// compatibility checks as C++ (pixel type must match the header
    /// channel; **every channel must have sampling `(1,1)`** — subsampling
    /// is not supported in tiled files) and builds the per-channel
    /// `TOutSliceInfo` table, including the "channel present in the header
    /// but absent from the frame buffer → write zeroes" case.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] when a frame-buffer slice's type is
    /// incompatible with the header channel, or when any slice has sampling
    /// other than `(1,1)`.
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++: compatibility checks for every channel that is also in the fb.
        for (name, ch) in channels.iter() {
            if let Some(slice) = frame_buffer.find_slice(name) {
                if ch.type_ != slice.type_ {
                    return Err(ExrError::Arg(format!(
                        "Pixel type of \"{name}\" channel of output file is not \
                         compatible with the frame buffer's pixel type."
                    )));
                }
                if slice.x_sampling != 1 || slice.y_sampling != 1 {
                    return Err(ExrError::Arg(
                        "All channels in a tiled file must have sampling (1,1).".into(),
                    ));
                }
            }
        }

        // C++: build the slice table in header channel order.
        let mut slices: Vec<TOutSliceInfo> = Vec::new();
        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                None => {
                    // Channel absent from the frame buffer → file gets zeroes.
                    slices.push(TOutSliceInfo {
                        type_: ch.type_,
                        base_offset: 0,
                        x_stride: 0,
                        y_stride: 0,
                        zero: true,
                        x_tile_coords: false,
                        y_tile_coords: false,
                    });
                }
                Some(slice) => {
                    slices.push(TOutSliceInfo {
                        type_: slice.type_,
                        base_offset: slice.base_offset,
                        x_stride: slice.x_stride,
                        y_stride: slice.y_stride,
                        zero: false,
                        x_tile_coords: slice.x_tile_coords,
                        y_tile_coords: slice.y_tile_coords,
                    });
                }
            }
        }

        self.slices = slices;
        self.frame_buffer = frame_buffer.clone();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // writeTiles / writeTile — ImfTiledOutputFile.cpp lines 1190-1413
    // -----------------------------------------------------------------------

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)` — port of `TiledOutputFile::writeTiles(dx1,dx2,dy1,dy2,lx,ly)`.
    ///
    /// `pixels` is the caller's current
    /// pixel data for this call — read fresh each call, never stored. The
    /// same small buffer can be refilled and passed again for the next
    /// call, matching C++'s streaming pattern; `set_frame_buffer`'s slices
    /// address it exactly as before (`base_offset`/strides), just resolved
    /// against `pixels` at gather time instead of a stored reference.
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
    /// * Propagates gather (incl. `pixels` bounds violations — never a
    ///   panic) / codec / stream errors.
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ writeTiles' parameter list + `pixels`
    fn write_tiles(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        mut dx1: i32,
        mut dx2: i32,
        mut dy1: i32,
        mut dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer specified …").
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data source.".into(),
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

        // C++: walk dy in the file's line order; dx always low→high. This
        // walk order is also the FILE ORDER the parallel path (below) must
        // feed tiles into `feed_tile` in — gather/encode may run out of
        // order, but on-disk bytes must not depend on thread count.
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
        // AND the captured thread count is `> 1` — 1:1 with C++'s
        // `nTiles > 1 && numThreads > 1` (`ImfTiledOutputFile.cpp:1191`).
        if coords.len() > 1 && self.num_threads.is_parallel() {
            self.write_tiles_parallel(stream, pixels, &coords)?;
            crate::probe::hit(crate::probe::Probe::TileWriteParallel);
            return Ok(());
        }

        // One pool for the whole serial loop, so
        // `write_single_tile`'s `encode_tile` call reuses one buffer across
        // every tile in this call instead of allocating fresh each time.
        let pool = crate::buf::BufPool::new();
        for &(dx, dy, lx, ly) in &coords {
            self.write_single_tile(stream, pixels, dx, dy, lx, ly, &pool)?;
        }

        Ok(())
    }

    /// The parallel write body (tiled variant): gather +
    /// encode the tiles on the shared pool (each on its own pooled buffer and
    /// pooled per-worker codec) while this thread FEEDS finished tiles -
    /// [`feed_tile`], the existing `bufferedTileWrite` state machine - one at a
    /// time, strictly in `coords` order (== the call's own tile-walk order ==
    /// file order), through [`crate::threading::ordered_pipeline`] (C++'s
    /// `TileBufferTask` ring, `ImfTiledOutputFile.cpp:1253-1330`). Bytes on
    /// disk are therefore identical to the serial path at every thread count:
    /// only the CPU-bound gather+encode work is parallelized, never the
    /// stateful emission decision.
    fn write_tiles_parallel(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        coords: &[(i32, i32, i32, i32)],
    ) -> Result<()> {
        // Upfront duplicate-write check for the whole batch (cheap, serial;
        // matches `write_single_tile`'s own check, done before any gather).
        for &(dx, dy, lx, ly) in coords {
            if self.tile_offsets.get(dx, dy, lx, ly)? != 0 {
                return Err(ExrError::Arg(format!(
                    "Attempt to write tile ({dx}, {dy}, {lx}, {ly}) more than once."
                )));
            }
        }

        let max_bytes_per_tile_line = self.max_bytes_per_tile_line;
        let ty_size = self.tile_desc.y_size as usize;
        // Ring size in BYTES - see `ThreadCount::window_size_for_bytes`.
        let window = self
            .num_threads
            .window_size_for_bytes(max_bytes_per_tile_line.saturating_mul(ty_size));
        let compression = self.compression;
        let tile_desc = self.tile_desc;
        let format = self.format;
        let buf_format = match format {
            Format::Xdr => BufferFormat::Xdr,
            Format::Native => BufferFormat::Native,
        };
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        let count = self.num_threads;
        // The workers read the header/layout for the whole call while the
        // emit step mutates the emit state: disjoint field borrows, no clones.
        let (header, slices, mut sink) = self.split();

        // One pool shared by the encode workers and the emit
        // step (which gives each buffer back once `feed_tile` is done with
        // it). At most `window` tiles are outstanding, so `window` buffers
        // cover the ring.
        let pool = crate::buf::BufPool::new();
        pool.prewarm_for_pipeline(crate::buf::WRITE_WINDOWS_IN_FLIGHT, window, max_bytes_per_tile_line.saturating_mul(ty_size), coords.len());
        let pool = &pool;
        // `CodecPool`, not `map_init`: see `Compressor`'s trait doc and
        // `crate::compressor::CodecPool`'s own doc.
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;

        crate::threading::ordered_pipeline(
            count,
            coords.len(),
            crate::threading::Ring::items(window),
            |_| 0,
            |k| -> Result<Vec<u8>> {
                let (dx, dy, lx, ly) = coords[k];
                let mut worker_compressor = codec_pool.lease(|| {
                    new_tile_compressor(compression, max_bytes_per_tile_line, ty_size, header)
                })?;
                let tile_range =
                    data_window_for_tile(&tile_desc, min_x, max_x, min_y, max_y, dx, dy, lx, ly)?;
                // Pooled gather buffer, finished in place -- see
                // `scanline::output`'s identical comment.
                let mut buf = pool.take(max_bytes_per_tile_line.saturating_mul(ty_size));
                let data_size = gather_tile(slices, tile_range, pixels, buf_format, &mut buf)?;
                crate::compressor::finish_chunk(worker_compressor.as_mut(), buf, data_size, |c, raw| {
                    Ok(c.compress_tile(raw, tile_range).map_err(crate::error::encode_err)?.len())
                })
            },
            |k, chunk| -> Result<()> {
                sink.feed(stream, coords[k], &chunk)?;
                // Return the encode buffer once feed_tile is
                // done reading it -- sound even when feed_tile internally
                // buffers a COPY into `self.tile_map` for out-of-turn
                // writing (that copy is a separate allocation, unaffected;
                // `chunk` itself is only ever read here, never stored).
                pool.give(chunk);
                Ok(())
            },
        )
    }

    /// Write the rectangle of tiles at MIP level `l` — port of
    /// `TiledOutputFile::writeTiles(dx1,dxMax,dyMin,dyMax,l)`, which forwards to
    /// `(…, l, l)`.
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ writeTiles' parameter list + `stream`/`pixels`
    fn write_tiles_l(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.write_tiles(stream, pixels, dx1, dx2, dy1, dy2, l, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)` — port of
    /// `TiledOutputFile::writeTile(int dx, int dy, int lx, int ly)`, which
    /// forwards to `writeTiles(dx, dx, dy, dy, lx, ly)`.
    fn write_tile(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.write_tiles(stream, pixels, dx, dx, dy, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l` — port of
    /// `TiledOutputFile::writeTile(int dx, int dy, int l)`, which forwards to
    /// `writeTile(dx, dy, l, l)`.
    fn write_tile_l(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        dx: i32,
        dy: i32,
        l: i32,
    ) -> Result<()> {
        self.write_tile(stream, pixels, dx, dy, l, l)
    }

    /// Quick pixel copy: transcode every tile of `input` straight into this
    /// file, raw (no decompress/recompress) — port of
    /// `TiledOutputFile::copyPixels(TiledInputFile&)` /
    /// `copyPixels(InputFile&)` / the `*Part` overloads
    /// (`ImfTiledOutputFile.cpp:1415-1445` + `:1597-1611`).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if the headers' tile description, data window,
    ///   line order, compression, or channel list differ (C++'s exact
    ///   compatibility checks — each with its own message in C++; folded into
    ///   one [`ExrError::Arg`] here, message says which check failed).
    /// * [`ExrError::Logic`] if this file already has pixel data written
    ///   (C++: "`already contains pixel data`").
    /// * Propagates `input`'s `as_tiled_input`/`tile_order`/`raw_tile_data`
    ///   errors.
    fn copy_pixels<T: TiledCopySource>(
        &mut self,
        stream: &mut dyn OStream,
        input: &mut T,
    ) -> Result<()> {
        input.ensure_tiled()?;

        // --- Header compatibility (ImfTiledOutputFile.cpp:1415-1470) --------
        let hdr = &self.header;
        let in_hdr = input.header();

        let td = hdr.tile_description().map_err(|_| {
            ExrError::Arg("The output file is tiled, but is missing a tile description.".into())
        })?;
        let in_td = in_hdr.tile_description().map_err(|_| {
            ExrError::Arg(
                "Cannot perform a quick pixel copy: the input file is not tiled. \
                 Try using OutputFile::copyPixels() instead."
                    .into(),
            )
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

        // --- Total tile count (ImfTiledOutputFile.cpp:1518-1535) -----------
        let num_all_tiles: i64 = match self.level_mode() {
            LevelMode::OneLevel | LevelMode::MipmapLevels => {
                let mut n: i64 = 0;
                for l in 0..self.num_levels()? {
                    n += self.num_x_tiles(l)? as i64 * self.num_y_tiles(l)? as i64;
                }
                n
            }
            LevelMode::RipmapLevels => {
                let mut n: i64 = 0;
                for ly in 0..self.num_y_levels {
                    for lx in 0..self.num_x_levels {
                        n += self.num_x_tiles(lx)? as i64 * self.num_y_tiles(ly)? as i64;
                    }
                }
                n
            }
            LevelMode::Invalid => {
                return Err(ExrError::Arg("Unknown LevelMode format.".into()));
            }
        };

        let random_y = self.line_order == LineOrder::RandomY;

        // For RANDOM_Y, C++ asks the INPUT file for the on-disk tile order and
        // drives the copy in that order (`in.tileOrder(...)`) — the input's
        // own file layout is what dictates a cheap sequential-read copy.
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
                let (_, _, _, _, chunk) = input.raw_tile_data(dx, dy, lx, ly)?;
                Self::write_tile_chunk_now(
                    stream,
                    &mut self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    coord,
                    &chunk,
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
                let (_, _, _, _, chunk) = input.raw_tile_data(dx, dy, lx, ly)?;
                Self::write_tile_chunk_now(
                    stream,
                    &mut self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    coord,
                    &chunk,
                )?;
                self.next_tile_to_write = Some(self.next_tile_coord(coord)?);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // The gather + compress + write of one tile.
    //
    // Fuses C++ `TileBufferTask::execute` (gather + compress) with
    // `bufferedTileWrite` / `writeTileData` (the chunk write + offset record).
    // Single-threaded. For `INCREASING_Y`/`DECREASING_Y` the on-disk
    // order follows the file's line order regardless of call order (a
    // tile written out of turn is buffered in `self.tile_map` until the
    // tiles before it are written); `RANDOM_Y` still writes immediately at
    // the current stream position in call order, unchanged.
    // -----------------------------------------------------------------------

    /// The successor of tile coordinate `a` in the file's line order (see
    /// [`TileGeom::next`]).
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
    /// (`header`, `slices`) next to the emit state ([`TileSink`]) that
    /// `feed` mutates, so neither needs to be cloned per call.
    fn split(&mut self) -> (&Header, &[TOutSliceInfo], TileSink<'_>) {
        let geom = TileGeom {
            tile_desc: self.tile_desc,
            line_order: self.line_order,
            num_x_levels: self.num_x_levels,
            num_y_levels: self.num_y_levels,
            num_x_tiles: &self.num_x_tiles,
            num_y_tiles: &self.num_y_tiles,
        };
        let sink = TileSink {
            geom,
            tile_offsets: &mut self.tile_offsets,
            tile_offsets_position: self.tile_offsets_position,
            part_number: self.part_number,
            tile_map: &mut self.tile_map,
            next_tile_to_write: &mut self.next_tile_to_write,
        };
        (&self.header, &self.slices, sink)
    }

    /// Write one already-encoded tile chunk to the stream NOW and back-patch
    /// its offset-table slot — port of C++ `writeTileData`
    /// (`ImfTiledOutputFile.cpp:391-437`). Split out of
    /// [`Self::write_single_tile`] so it can be called either
    /// immediately or later, when a buffered tile's turn comes up, without
    /// re-deriving the chunk bytes. Takes the exact fields it touches rather
    /// than `&mut self` so callers can hold `chunk` borrowed from another
    /// field of `self` (e.g. `compressor`) across the call — a
    /// disjoint-field borrow, not a self-referential one.
    fn write_tile_chunk_now(
        stream: &mut dyn OStream,
        tile_offsets: &mut TileOffsets,
        tile_offsets_position: u64,
        part_number: Option<i32>,
        coord: (i32, i32, i32, i32),
        chunk: &[u8],
    ) -> Result<()> {
        let (dx, dy, lx, ly) = coord;
        let chunk_position = stream.tellp()?;
        *tile_offsets.get_mut(dx, dy, lx, ly)? = chunk_position;

        // C++ writeTileData: for a multipart file the tile chunk body is
        // prefixed with the part number (i32 LE) before tileX/tileY/levelX/
        // levelY (ImfTiledOutputFile.cpp:418-421) — the ONLY difference from the
        // single-part layout.  Then tileX, tileY, levelX, levelY, dataSize
        // (5 * i32), then the data bytes.
        if let Some(pn) = part_number {
            xdr::write_i32(&mut *stream, pn)?;
        }
        xdr::write_i32(&mut *stream, dx)?;
        xdr::write_i32(&mut *stream, dy)?;
        xdr::write_i32(&mut *stream, lx)?;
        xdr::write_i32(&mut *stream, ly)?;
        xdr::write_i32(&mut *stream, chunk.len() as i32)?;
        stream.write(chunk)?;

        // --- Back-patch the offset table slot ------------------------------
        // Find the flat slot index in the reserved table and write the u64 LE
        // offset there, then restore the write cursor (C++ does this from its
        // destructor for the whole table; we do it per tile — see module docs).
        let slot_index = tile_offsets.slot_index(dx, dy, lx, ly)?;
        let saved = stream.tellp()?;
        stream.seekp(tile_offsets_position + slot_index as u64 * SIZE_I64 as u64)?;
        xdr::write_u64(&mut *stream, chunk_position)?;
        stream.seekp(saved)?;

        // The chunk header is 5 i32 fields; documented here so the byte layout
        // and the back-patch math (offset table is u64 LE) stay in sync.
        debug_assert_eq!(SIZE_I32, 4);

        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // the buffer-pool param pushed this over the limit; 1:1 shape otherwise
    fn write_single_tile(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        pool: &crate::buf::BufPool,
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

        let buf_format = match self.format {
            Format::Xdr => BufferFormat::Xdr,
            Format::Native => BufferFormat::Native,
        };

        // --- Gather the tile into a pooled buffer ---------------------------
        let mut buf = pool.take(self.tile_buffer_size);
        let data_size = gather_tile(&self.slices, tile_range, pixels, buf_format, &mut buf)?;

        // --- Compress (or raw for NONE) ------------------------------------
        // C++ runs the gathered buffer through compressTile(); the codec's own
        // WRITER decides what is stored: C++ keeps the encoding only when
        // `compSize < _tileBuffer->dataSize` (ImfTiledOutputFile.cpp:783) and
        // stores the raw tile otherwise — at equality too, because readers
        // detect a raw passthrough by `dataSize == uncompressedSize` alone.
        // `stored_chunk` (inside [`encode_tile`]) is that single decision
        // point. Every ported codec reports Format::Xdr, so the NATIVE
        // convertToXdr branch is unreachable here (asserted by `debug_assert`).
        debug_assert_eq!(self.format, Format::Xdr);
        let chunk = crate::compressor::finish_chunk(self.compressor.as_mut(), buf, data_size, |c, raw| {
            Ok(c.compress_tile(raw, tile_range).map_err(crate::error::encode_err)?.len())
        })?;

        let result = self.feed_tile(stream, (dx, dy, lx, ly), &chunk);
        pool.give(chunk);
        result
    }

    /// The `bufferedTileWrite` state machine, extracted as its own step
    /// so the parallel path can gather+encode a whole window
    /// of tiles first and then feed each ALREADY-COMPUTED chunk through this
    /// SAME stateful logic, one at a time, in file order — see
    /// `write_tiles_parallel`'s doc comment for why that keeps on-disk bytes
    /// independent of thread count. Port of C++ `bufferedTileWrite`
    /// (`ImfTiledOutputFile.cpp:441-556`); the serial path's
    /// `write_single_tile` feeds through it too.
    fn feed_tile(
        &mut self,
        stream: &mut dyn OStream,
        key: (i32, i32, i32, i32),
        chunk: &[u8],
    ) -> Result<()> {
        self.split().2.feed(stream, key, chunk)
    }
}

impl TileSink<'_> {
    /// The `bufferedTileWrite` state machine (port of C++
    /// `bufferedTileWrite`, `ImfTiledOutputFile.cpp:441-556`): write `chunk`
    /// now when it is the next tile in the file's line order (then flush any
    /// buffered successors), else hold a copy until its turn. The serial path
    /// and the pipelined emit step both feed through here, which is what
    /// keeps on-disk bytes independent of the thread count.
    fn feed(
        &mut self,
        stream: &mut dyn OStream,
        key: (i32, i32, i32, i32),
        chunk: &[u8],
    ) -> Result<()> {
        let (dx, dy, lx, ly) = key;
        match self.geom.line_order {
            LineOrder::RandomY | LineOrder::Invalid => {
                // Tiles can be written in any order: never buffer.
                TiledOutputFileState::write_tile_chunk_now(
                    stream,
                    &mut *self.tile_offsets,
                    self.tile_offsets_position,
                    self.part_number,
                    key,
                    chunk,
                )?;
            }
            LineOrder::IncreasingY | LineOrder::DecreasingY => {
                if self.tile_map.contains_key(&key) {
                    return Err(ExrError::Arg(format!(
                        "Attempt to write tile ({dx}, {dy}, {lx}, {ly}) more than once."
                    )));
                }

                if *self.next_tile_to_write == Some(key) {
                    // Every tile before this one in line order is already on
                    // disk: write it now, advance the cursor, then flush any
                    // tiles buffered for the coordinates that follow.
                    TiledOutputFileState::write_tile_chunk_now(
                        stream,
                        &mut *self.tile_offsets,
                        self.tile_offsets_position,
                        self.part_number,
                        key,
                        chunk,
                    )?;
                    let mut next = self.geom.next(key)?;
                    while let Some(buffered) = self.tile_map.remove(&next) {
                        TiledOutputFileState::write_tile_chunk_now(
                            stream,
                            &mut *self.tile_offsets,
                            self.tile_offsets_position,
                            self.part_number,
                            next,
                            &buffered,
                        )?;
                        next = self.geom.next(next)?;
                    }
                    *self.next_tile_to_write = Some(next);
                } else {
                    // Out of turn: hold a copy until its turn comes up (C++
                    // `BufferedTile` — also a `memcpy`, so this is not an
                    // extra cost buffering introduces, only the one C++ always
                    // pays for a buffered tile).
                    crate::probe::hit(crate::probe::Probe::TileBuffered);
                    self.tile_map.insert(key, chunk.to_vec());
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure per-tile pipeline functions (writer side)
// ---------------------------------------------------------------------------
//
// `gather_tile`/`encode_tile` are the writer-side analogue of the tiled
// reader's `decode_tile`/`scatter_tile` split (`tiled/input.rs`):
// `write_single_tile` (the serial path AND the parallel path's per-tile
// gather+encode) composes them in the same order, so a tile's chunk bytes
// depend only on `(gathered raw bytes, tile_range, codec params)` — never on
// which thread gathered/encoded it.

/// Gather one tile's channels into `dst` — pure function, no stream access,
/// no `self`. `dst` is a `BufPool` buffer on both the serial and the
/// parallel path. Returns the gathered
/// `data_size` (C++ `writePtr - buffer`).
fn gather_tile(
    slices: &[TOutSliceInfo],
    tile_range: Box2i,
    pixels: &[u8],
    buf_format: BufferFormat,
    dst: &mut [u8],
) -> Result<usize> {
    let num_pixels_per_scan_line = (tile_range.max.x - tile_range.min.x + 1) as usize;
    let mut write_ptr = 0usize;

    for y in tile_range.min.y..=tile_range.max.y {
        for slice in slices {
            if slice.zero {
                write_ptr =
                    fill_channel_with_zeroes(dst, write_ptr, slice.type_, num_pixels_per_scan_line)?;
            } else {
                // C++ address arithmetic (with the safe-Rust base_offset):
                //   xOffset = xTileCoords * tileRange.min.x
                //   yOffset = yTileCoords * tileRange.min.y
                //   readPtr = base + (y - yOffset)*yStride + (minX - xOffset)*xStride
                let x_offset = if slice.x_tile_coords { tile_range.min.x } else { 0 };
                let y_offset = if slice.y_tile_coords { tile_range.min.y } else { 0 };

                let read_ptr_i: i64 = slice.base_offset
                    + (y - y_offset) as i64 * slice.y_stride as i64
                    + (tile_range.min.x - x_offset) as i64 * slice.x_stride as i64;
                if read_ptr_i < 0 {
                    return Err(ExrError::Arg(
                        "Frame buffer slice address is out of range (negative).".into(),
                    ));
                }

                let (new_wo, _new_rp) = copy_from_frame_buffer(
                    dst,
                    write_ptr,
                    pixels,
                    read_ptr_i as usize,
                    num_pixels_per_scan_line,
                    slice.x_stride,
                    buf_format,
                    slice.type_,
                )?;
                write_ptr = new_wo;
            }
        }
    }
    Ok(write_ptr)
}

// Thin wrapper API, generic over the stream holder `S`.  Covers BOTH the
// borrowed default (`S = &'a mut dyn OStream`) and the owned `Box<dyn OStream>`
// variant — every method below addresses the stream through
// `DerefMut<Target = dyn OStream>` and delegates its algorithm to
// [`TiledOutputFileState`], so the two paths (and the multipart
// [`TiledOutputPartWriter`] view) are a single source of truth and emit
// byte-identical files.
impl<'a, S> TiledOutputFile<'a, S>
where
    S: DerefMut<Target = dyn OStream + 'a>,
{
    /// Consume the writer, check that the file is complete, flush the stream and hand it back.
    ///
    /// Every tile offset is back-patched right after its chunk is written (see the module docs), so no
    /// table has to be rewritten here. What `finish` adds is the part dropping the writer cannot do:
    /// * it fails with [`ExrError::Logic`] if any tile of any level was never written (its offset slot is
    ///   still 0) — a half-written tiled file is otherwise indistinguishable from a good one at write
    ///   time;
    /// * it calls [`OStream::flush`], so a failed final write (full disk, vanished network drive) is
    ///   RETURNED. A buffered file stream such as [`StdOfStream`] holds up to 256 KiB, and its `Drop` can
    ///   only log a flush failure.
    ///
    /// For [`TiledOutputFileOwned`] it returns the owned `Box<dyn OStream>`; for the borrowed default the
    /// original `&mut` reference. Dropping the writer without calling `finish` stays valid but gets neither
    /// check.
    pub fn finish(mut self) -> Result<S> {
        let offsets = &self.state.tile_offsets;
        if offsets.any_offsets_are_invalid() {
            let slots = offsets.slot_count();
            let missing = offsets.get_offsets().iter().flatten().flatten().filter(|&&o| o == 0).count();
            return Err(ExrError::Logic(format!(
                "tiled output is incomplete: {missing} of {slots} tiles were never written"
            )));
        }
        OStream::flush(&mut *self.stream)?;
        Ok(self.stream)
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk -- port of `void updatePreviewImage(const PreviewRgba
    /// newPixels[])`.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(&mut *self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written tile's chunk, starting
    /// `offset` bytes into it, with the byte `c` -- port of `void breakTile(...)`
    /// (test-only corrupt-chunk helper).
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ breakTile's parameter list
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
        self.state.break_tile(&mut *self.stream, dx, dy, lx, ly, offset, length, c)
    }

    // -----------------------------------------------------------------------
    // Accessors -- thin delegators to TiledOutputFileState
    // -----------------------------------------------------------------------

    /// The image header -- port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// Name of the underlying file or stream -- port of `const char*
    /// fileName() const`.
    pub fn file_name(&self) -> &str {
        self.stream.file_name()
    }

    /// The current frame buffer -- port of `const FrameBuffer& frameBuffer() const`.
    pub fn frame_buffer(&self) -> &FrameBuffer {
        self.state.frame_buffer()
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

    /// Level rounding mode -- port of `LevelRoundingMode levelRoundingMode() const`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
    }

    /// Compression type -- convenience accessor (C++ reads it off `header()`).
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
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

    /// Whether `(lx, ly)` is a valid level -- port of `bool isValidLevel(int, int) const`.
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

    /// Data window of level `(lx, ly)` -- port of
    /// `Box2i dataWindowForLevel(int lx, int ly) const`.
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_level(lx, ly)
    }

    /// Data window of level `l` (MIP convenience) -- port of
    /// `Box2i dataWindowForLevel(int l) const`.
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.state.data_window_for_level_l(l)
    }

    /// Data window of tile `(dx, dy, lx, ly)` -- port of
    /// `Box2i dataWindowForTile(int dx, int dy, int lx, int ly) const`.
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.state.data_window_for_tile(dx, dy, lx, ly)
    }

    /// Data window of tile `(dx, dy, l)` (MIP convenience) -- port of
    /// `Box2i dataWindowForTile(int dx, int dy, int l) const`.
    pub fn data_window_for_tile_l(&self, dx: i32, dy: i32, l: i32) -> Result<Box2i> {
        self.state.data_window_for_tile_l(dx, dy, l)
    }

    // -----------------------------------------------------------------------
    // setFrameBuffer / writeTiles / writeTile / copyPixels
    // -----------------------------------------------------------------------

    /// Set the pixel LAYOUT (channel offsets/strides/types) -- port of
    /// `TiledOutputFile::setFrameBuffer`. See `TiledOutputFileState::set_frame_buffer`.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level
    /// `(lx, ly)` -- port of `TiledOutputFile::writeTiles(dx1,dx2,dy1,dy2,lx,ly)`.
    /// See `TiledOutputFileState::write_tiles`.
    #[allow(clippy::too_many_arguments)] // 1:1 with C++ writeTiles' parameter list + `pixels`
    pub fn write_tiles(
        &mut self,
        pixels: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tiles(&mut *self.stream, pixels, dx1, dx2, dy1, dy2, lx, ly)
    }

    /// Write the rectangle of tiles at MIP level `l` -- port of
    /// `TiledOutputFile::writeTiles(dx1,dxMax,dyMin,dyMax,l)`.
    pub fn write_tiles_l(
        &mut self,
        pixels: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tiles_l(&mut *self.stream, pixels, dx1, dx2, dy1, dy2, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)` -- port of
    /// `TiledOutputFile::writeTile(int dx, int dy, int lx, int ly)`.
    pub fn write_tile(&mut self, pixels: &[u8], dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
        self.state.write_tile(&mut *self.stream, pixels, dx, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l` -- port of
    /// `TiledOutputFile::writeTile(int dx, int dy, int l)`.
    pub fn write_tile_l(&mut self, pixels: &[u8], dx: i32, dy: i32, l: i32) -> Result<()> {
        self.state.write_tile_l(&mut *self.stream, pixels, dx, dy, l)
    }

    /// Quick pixel copy: transcode every tile of `input` straight into this
    /// file, raw (no decompress/recompress) -- port of the 4 C++
    /// `TiledOutputFile::copyPixels` overloads. See
    /// `TiledOutputFileState::copy_pixels` / [`TiledCopySource`].
    pub fn copy_pixels<T: TiledCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(&mut *self.stream, input)
    }
}

/// A borrowing view over a tiled output part's PERSISTED write state -- the
/// analogue of `crate::scanline::OutputPartWriter` for tiled
/// parts. `crate::multipart::MultiPartOutputFile::tiled_output_part` returns
/// this instead of rebuilding a fresh [`TiledOutputFile`] on every call, so
/// interleaved writes across parts -- write part 0's tiles, write part 1,
/// come back and write more of part 0 -- resume correctly: the SAME
/// [`TiledOutputFileState`] (incl. the `tile_map`/`next_tile_to_write` buffering)
/// is reused, not reset. Exposes the identical public method set as
/// [`TiledOutputFile`] by delegating to the same [`TiledOutputFileState`]
/// methods, driven by the SAME `stream: &mut dyn OStream` reborrowed from
/// the container each call.
pub struct TiledOutputPartWriter<'p> {
    stream: &'p mut dyn OStream,
    state: &'p mut TiledOutputFileState,
}

impl<'p> TiledOutputPartWriter<'p> {
    /// Construct a view over a container-owned stream + persisted state.
    pub fn new(stream: &'p mut dyn OStream, state: &'p mut TiledOutputFileState) -> Self {
        Self { stream, state }
    }

    /// The image header -- port of `const Header& header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// Name of the underlying file or stream.
    pub fn file_name(&self) -> &str {
        self.stream.file_name()
    }

    /// The current frame buffer.
    pub fn frame_buffer(&self) -> &FrameBuffer {
        self.state.frame_buffer()
    }

    /// Tile width.
    pub fn tile_x_size(&self) -> u32 {
        self.state.tile_x_size()
    }

    /// Tile height.
    pub fn tile_y_size(&self) -> u32 {
        self.state.tile_y_size()
    }

    /// Level mode.
    pub fn level_mode(&self) -> LevelMode {
        self.state.level_mode()
    }

    /// Level rounding mode.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.state.level_rounding_mode()
    }

    /// Compression type.
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.state.compression()
    }

    /// Number of levels in X.
    pub fn num_x_levels(&self) -> i32 {
        self.state.num_x_levels()
    }

    /// Number of levels in Y.
    pub fn num_y_levels(&self) -> i32 {
        self.state.num_y_levels()
    }

    /// Number of levels (MIP convenience).
    pub fn num_levels(&self) -> Result<i32> {
        self.state.num_levels()
    }

    /// Whether `(lx, ly)` is a valid level.
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.state.is_valid_level(lx, ly)
    }

    /// Width of level `lx`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.state.level_width(lx)
    }

    /// Height of level `ly`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.state.level_height(ly)
    }

    /// Number of X tiles in level `lx`.
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }

    /// Number of Y tiles in level `ly`.
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

    /// Set the pixel LAYOUT (channel offsets/strides/types).
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write the rectangle of tiles `[dx1..=dx2] x [dy1..=dy2]` at level `(lx, ly)`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_tiles(
        &mut self,
        pixels: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
    ) -> Result<()> {
        self.state.write_tiles(self.stream, pixels, dx1, dx2, dy1, dy2, lx, ly)
    }

    /// Write the rectangle of tiles at MIP level `l`.
    pub fn write_tiles_l(
        &mut self,
        pixels: &[u8],
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        l: i32,
    ) -> Result<()> {
        self.state.write_tiles_l(self.stream, pixels, dx1, dx2, dy1, dy2, l)
    }

    /// Write a single tile `(dx, dy, lx, ly)`.
    pub fn write_tile(&mut self, pixels: &[u8], dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
        self.state.write_tile(self.stream, pixels, dx, dy, lx, ly)
    }

    /// Write a single tile `(dx, dy, l)` at MIP level `l`.
    pub fn write_tile_l(&mut self, pixels: &[u8], dx: i32, dy: i32, l: i32) -> Result<()> {
        self.state.write_tile_l(self.stream, pixels, dx, dy, l)
    }

    /// Quick pixel copy -- see `TiledOutputFile::copy_pixels`.
    pub fn copy_pixels<T: TiledCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk. See [`TiledOutputFileState::update_preview_image`] for the
    /// full doc. Reachable on a
    /// multipart part because `MultiPartOutputFile::tiled_output_part`
    /// builds this part's state with `from_part_with_preview`.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written tile's chunk, starting
    /// `offset` bytes into it, with the byte `c`. See
    /// [`TiledOutputFileState::break_tile`] for the full doc.
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
    use crate::container::xdr;
    use crate::frame_buffer::Slice;
    use crate::io::{MemIStream, MemOStream};
    use imath_rs::{Half, V2f};

    /// Build a tiled header `width x height` with one HALF channel "R",
    /// NO_COMPRESSION, the given line order and tile description.
    fn make_header(
        width: i32,
        height: i32,
        line_order: LineOrder,
        td: TileDescription,
    ) -> Header {
        let mut h = Header::with_dimensions(
            width,
            height,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            line_order,
            Compression::NoCompression,
        )
        .unwrap();
        h.set_type(TILED_IMAGE).unwrap();
        h.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Row-major HALF pixel plane for a `width x height` image; one channel.
    /// value(x, y) = half(y*width + x).  Returns `(buffer, frame_buffer)`.
    fn make_pixels(width: i32, height: i32) -> (Vec<u8>, FrameBuffer) {
        let row_bytes = width as usize * 2;
        let mut buf = vec![0u8; row_bytes * height as usize];
        for y in 0..height {
            for x in 0..width {
                let off = y as usize * row_bytes + x as usize * 2;
                let v = Half::from((y * width + x) as f32).bits();
                buf[off..off + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset: 0,
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

    /// Expected XDR (little-endian) half bytes for the tile rectangle
    /// `[min_x..=max_x] x [min_y..=max_y]` of the value(x,y)=half(y*W+x) image.
    fn expected_tile_xdr(
        width: i32,
        min_x: i32,
        max_x: i32,
        min_y: i32,
        max_y: i32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let bits = Half::from((y * width + x) as f32).bits();
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        out
    }

    /// Parse the written file and assert the global structure: magic, version
    /// (tiled flag set), fully back-patched offset table (no zero slots), and
    /// total size = header + table + Σ chunks.  Returns the chunk-region bytes
    /// and the parsed offset table for finer per-test checks.
    fn parse_and_check_structure(
        header: &Header,
        bytes: &[u8],
        num_offset_slots: usize,
    ) -> (Vec<u64>, usize) {
        // Standalone header write to learn the header prefix length.
        let mut hdr_stream = MemOStream::new();
        header.write_to(&mut hdr_stream, 0).unwrap();
        let header_len = hdr_stream.data().len();

        // 1. Leading header bytes match the standalone header write.
        assert_eq!(&bytes[..header_len], hdr_stream.data());

        // 2. Magic + tiled version flag in the standalone prefix.
        let mut his = MemIStream::new(hdr_stream.data().to_vec());
        let magic = xdr::read_i32(&mut his).unwrap();
        assert_eq!(magic as u32, crate::container::version::MAGIC);
        let version_word = xdr::read_i32(&mut his).unwrap() as u32;
        assert!(
            crate::container::version::is_tiled(version_word),
            "tiled flag must be set in the version word"
        );

        // 3. Offset table: read every u64 LE; none may remain zero.
        let table_start = header_len;
        let chunks_start = table_start + num_offset_slots * 8;
        let mut offsets = Vec::with_capacity(num_offset_slots);
        let mut tis = MemIStream::new(bytes[table_start..chunks_start].to_vec());
        for _ in 0..num_offset_slots {
            let off = xdr::read_u64(&mut tis).unwrap();
            assert_ne!(off, 0, "offset table slot was never back-patched");
            offsets.push(off);
        }
        (offsets, chunks_start)
    }

    /// Write the SAME multi-tile rectangle with `num_threads`
    /// in {0, 1, 2, 3, 8, 17} and assert the produced FILE BYTES are
    /// identical at every thread count — the tiled-writer analogue of
    /// `scanline::output::tests::parallel_write_whole_image_is_byte_identical_across_thread_counts`.
    /// A `5x4` nominal tile over a `37x29` image gives `8*8 = 64` tiles.
    #[test]
    fn parallel_write_tiles_is_byte_identical_across_thread_counts() {
        let td = TileDescription {
            x_size: 5,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (37i32, 29i32);
        let (pixels, fb) = make_pixels(w, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let header = make_header(w, h, LineOrder::IncreasingY, td);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::TileWriteParallel);
                let mut tof = TiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            let bytes = out.into_data();

            // Sanity: round-trips to the original pixels.
            let mut is = MemIStream::new(bytes.clone());
            let mut tif = crate::tiled::TiledInputFile::new(&mut is).unwrap();
            let mut readback = vec![0u8; pixels.len()];
            tif.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
            tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::TileWriteParallel, n);
            assert_eq!(readback, pixels, "num_threads={n}: round-trip mismatch");

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

    /// Same invariant with `DecreasingY` line order (exercises the
    /// `bufferedTileWrite`/`tile_map` out-of-turn buffering path — tiles
    /// gather+encode out of order in the parallel batch, but `feed_tile`
    /// still emits and buffers exactly as it would serially).
    #[test]
    fn parallel_write_tiles_decreasing_y_byte_identical_across_thread_counts() {
        let td = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (33i32, 25i32);
        let (pixels, fb) = make_pixels(w, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let header = make_header(w, h, LineOrder::DecreasingY, td);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::TileWriteParallel);
                let mut tof = TiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            let bytes = out.into_data();

            let mut is = MemIStream::new(bytes.clone());
            let mut tif = crate::tiled::TiledInputFile::new(&mut is).unwrap();
            let mut readback = vec![0u8; pixels.len()];
            tif.set_frame_buffer(&fb).unwrap();
            let (nx, ny) = (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
            tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::TileWriteParallel, n);
            assert_eq!(readback, pixels, "num_threads={n}: round-trip mismatch");

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
    fn one_level_uncompressed_structure() {
        // 4x4 image, 2x2 tiles → ONE_LEVEL, 2x2 grid = 4 tiles.
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(tof.num_x_levels(), 1);
            assert_eq!(tof.num_y_levels(), 1);
            assert_eq!(tof.num_x_tiles(0).unwrap(), 2);
            assert_eq!(tof.num_y_tiles(0).unwrap(), 2);
            tof.set_frame_buffer(&fb).unwrap();
            // Write all 4 tiles in INCREASING_Y order.
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        // 4 tiles → 4 offset slots.
        let (offsets, chunks_start) = parse_and_check_structure(&header, &bytes, 4);

        // Rebuild expected chunks in INCREASING_Y row-major tile order
        // (dy outer, dx inner): tiles (0,0),(1,0),(0,1),(1,1).
        let mut expected_chunks: Vec<u8> = Vec::new();
        let mut expected_offsets_in_order: Vec<u64> = Vec::new();
        for dy in 0..2i32 {
            for dx in 0..2i32 {
                expected_offsets_in_order
                    .push(chunks_start as u64 + expected_chunks.len() as u64);
                let min_x = dx * 2;
                let max_x = min_x + 1;
                let min_y = dy * 2;
                let max_y = min_y + 1;
                let data = expected_tile_xdr(4, min_x, max_x, min_y, max_y);
                expected_chunks.extend_from_slice(&dx.to_le_bytes());
                expected_chunks.extend_from_slice(&dy.to_le_bytes());
                expected_chunks.extend_from_slice(&0i32.to_le_bytes()); // lx
                expected_chunks.extend_from_slice(&0i32.to_le_bytes()); // ly
                expected_chunks.extend_from_slice(&(data.len() as i32).to_le_bytes());
                expected_chunks.extend_from_slice(&data);
            }
        }

        // Chunk bytes match exactly.
        assert_eq!(&bytes[chunks_start..], expected_chunks.as_slice());

        // Offset table (stored in [level][dy][dx] order) matches the write order.
        assert_eq!(offsets, expected_offsets_in_order);

        // Total size = header + table + chunks.
        assert_eq!(bytes.len(), chunks_start + expected_chunks.len());
    }

    #[test]
    fn mipmap_uncompressed_all_levels_structure() {
        // 4x4 image, 2x2 tiles, MIPMAP round-down.
        //   max(w,h)=4 → floorLog2(4)=2 → 3 levels.
        //   level 0: 4x4 → 2x2 tiles (4); level 1: 2x2 → 1x1 (1);
        //   level 2: 1x1 → 1x1 (1).  Total slots = 4+1+1 = 6.
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(tof.num_x_levels(), 3);
            assert_eq!(tof.num_levels().unwrap(), 3);
            // Write every tile of every MIP level (lx == ly).
            for l in 0..3 {
                let nx = tof.num_x_tiles(l).unwrap();
                let ny = tof.num_y_tiles(l).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                tof.write_tiles_l(&pixels, 0, nx - 1, 0, ny - 1, l).unwrap();
            }
        }
        let bytes = out.into_data();

        // 6 offset slots, all back-patched.
        let (offsets, chunks_start) = parse_and_check_structure(&header, &bytes, 6);
        assert_eq!(offsets.len(), 6);

        // Walk every chunk and verify its header coords + dataSize against the
        // level data windows, and that the chunks tile the file exactly.
        let mut pos = chunks_start;
        let mut chunk_count = 0;
        // Expected per-level (lx==ly) tile rectangles, in [level][dy][dx] order.
        let td_ref = header.tile_description().unwrap();
        for l in 0..3i32 {
            let lvl = data_window_for_level(&td_ref, 0, 3, 0, 3, l, l).unwrap();
            let nx = ((lvl.max.x - lvl.min.x) / 2) + 1; // ceil width / 2 with 2-tile
            let ny = ((lvl.max.y - lvl.min.y) / 2) + 1;
            for dy in 0..ny {
                for dx in 0..nx {
                    let mut cis = MemIStream::new(bytes[pos..].to_vec());
                    let rdx = xdr::read_i32(&mut cis).unwrap();
                    let rdy = xdr::read_i32(&mut cis).unwrap();
                    let rlx = xdr::read_i32(&mut cis).unwrap();
                    let rly = xdr::read_i32(&mut cis).unwrap();
                    let dsize = xdr::read_i32(&mut cis).unwrap();
                    assert_eq!((rdx, rdy, rlx, rly), (dx, dy, l, l));

                    // Tile pixel rect → expected uncompressed dataSize.
                    let tile = data_window_for_tile(&td_ref, 0, 3, 0, 3, dx, dy, l, l).unwrap();
                    let w = (tile.max.x - tile.min.x + 1) as usize;
                    let hgt = (tile.max.y - tile.min.y + 1) as usize;
                    assert_eq!(dsize as usize, w * hgt * 2, "tile ({dx},{dy},{l}) size");

                    pos += 5 * 4 + dsize as usize;
                    chunk_count += 1;
                }
            }
        }
        assert_eq!(chunk_count, 6);
        // Chunks tile the file exactly up to its end.
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn writing_a_tile_twice_errors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
        tof.set_frame_buffer(&fb).unwrap();
        tof.write_tile(&pixels, 0, 0, 0, 0).unwrap();
        let err = tof.write_tile(&pixels, 0, 0, 0, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn no_frame_buffer_errors() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let mut out = MemOStream::new();
        let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
        let err = tof.write_tile(&[], 0, 0, 0, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn invalid_tile_coords_error() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);
        let mut out = MemOStream::new();
        let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
        tof.set_frame_buffer(&fb).unwrap();
        // dx=2 is out of range (only 0,1 valid).
        assert!(tof.write_tile(&pixels, 2, 0, 0, 0).is_err());
        // lx=1 invalid for ONE_LEVEL.
        assert!(tof.write_tile(&pixels, 0, 0, 1, 0).is_err());
    }

    /// A frame buffer addressing a fresh read-back buffer for a 4x4 image
    /// (origin (0,0), one HALF "R" plane, row-major).
    fn fb_for_read() -> FrameBuffer {
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset: 0,
                x_stride: 2,
                y_stride: 4 * 2,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        fb
    }

    /// The OWNED-stream path.  A small struct OWNS both its
    /// `StdOfStream` (boxed, via the writer) and the persisted
    /// [`TiledOutputFileOwned`] in ONE struct, and writes tiles across
    /// MULTIPLE `write_tile` calls — proving the lifetime works (impossible
    /// with the borrowing default in safe Rust because the writer would have to
    /// borrow a stream the same struct owns).  The temp file is then read back
    /// via [`TiledInputFile`] and the pixels asserted to match, and the owned
    /// file is shown byte-identical to the borrow path for the same input.
    #[test]
    fn owned_stream_persisted_writer_round_trip_and_byte_parity() {
        use crate::io::{StdIfStream, StdOfStream};

        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        // Unique temp path for the owned-stream output.
        let path = std::env::temp_dir()
            .join(format!("exr_core_owned_tiled_{}.exr", std::process::id()));
        let path_str = path.to_str().unwrap().to_owned();

        // A long-lived holder that OWNS both the stream (inside the writer) and
        // the writer — the exact shape a streaming/region writer needs.
        // `'a` is the stream's own lifetime (`Box<dyn OStream + 'a>`); pixel
        // data carries no lifetime at all (passed per `write_tile` call,
        // never stored).
        struct OwningWriter<'a> {
            writer: TiledOutputFileOwned<'a>,
        }

        let owned_stream: Box<dyn OStream> = Box::new(StdOfStream::create(&path_str).unwrap());
        let writer = TiledOutputFile::new_owned(owned_stream, &header).unwrap();
        let mut holder = OwningWriter { writer };

        // Frame buffer LAYOUT set once, then pixel bytes passed per call (the
        // writer outlives each call — that is the point of owning the stream).
        holder.writer.set_frame_buffer(&fb).unwrap();
        holder.writer.write_tile(&pixels, 0, 0, 0, 0).unwrap();
        holder.writer.write_tile(&pixels, 1, 0, 0, 0).unwrap();
        holder.writer.write_tile(&pixels, 0, 1, 0, 0).unwrap();
        holder.writer.write_tile(&pixels, 1, 1, 0, 0).unwrap();
        // Recover + drop the owned stream so the file is flushed/closed.
        drop(holder.writer.finish().unwrap());

        let owned_bytes = std::fs::read(&path_str).unwrap();
        std::fs::remove_file(&path_str).ok();

        // Borrow path with the SAME input → must be byte-identical.
        let mut mem = MemOStream::new();
        {
            let mut borrow_writer = TiledOutputFile::new(&mut mem, &header).unwrap();
            borrow_writer.set_frame_buffer(&fb).unwrap();
            borrow_writer.write_tile(&pixels, 0, 0, 0, 0).unwrap();
            borrow_writer.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            borrow_writer.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            borrow_writer.write_tile(&pixels, 1, 1, 0, 0).unwrap();
            // A complete file finishes cleanly (completeness check + stream flush).
            borrow_writer.finish().unwrap();
        }
        // `finish` must refuse a half-written file instead of reporting success.
        {
            let mut partial = MemOStream::new();
            let mut w = TiledOutputFile::new(&mut partial, &header).unwrap();
            w.set_frame_buffer(&fb).unwrap();
            w.write_tile(&pixels, 0, 0, 0, 0).unwrap();
            w.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            w.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            let err = w.finish().err().expect("3 of 4 tiles written: finish must fail");
            assert!(err.to_string().contains("1 of 4 tiles"), "unexpected message: {err}");
        }
        let borrow_bytes = mem.into_data();
        assert_eq!(
            owned_bytes, borrow_bytes,
            "owned-stream output must be byte-identical to the borrow path"
        );

        // Read the owned file back via TiledInputFile (over its own StdIfStream,
        // a real file read end-to-end) and assert every pixel matches.
        let path2 = std::env::temp_dir()
            .join(format!("exr_core_owned_tiled_rb_{}.exr", std::process::id()));
        let path2_str = path2.to_str().unwrap().to_owned();
        std::fs::write(&path2_str, &owned_bytes).unwrap();
        let mut is = StdIfStream::open(&path2_str).unwrap();
        let mut tif = crate::tiled::TiledInputFile::new(&mut is).unwrap();
        let mut read_back = vec![0u8; pixels.len()];
        tif.set_frame_buffer(&fb_for_read()).unwrap();
        tif.read_tiles(0, 1, 0, 1, 0, 0, &mut read_back).unwrap();
        drop(tif);
        std::fs::remove_file(&path2_str).ok();
        assert_eq!(read_back, pixels, "owned-path round trip pixel mismatch");
    }

    #[test]
    fn decreasing_y_writes_rows_high_to_low() {
        // 4x6 image, 2x2 tiles → 2 x 3 tile grid.  DECREASING_Y must emit the
        // dy=2 row first, then dy=1, then dy=0.
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 6, LineOrder::DecreasingY, td);
        let (pixels, fb) = make_pixels(4, 6);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 2, 0, 0).unwrap();
        }
        let bytes = out.into_data();
        let (_offsets, chunks_start) = parse_and_check_structure(&header, &bytes, 6);

        // First chunk on disk must be a dy=2 tile.
        let mut cis = MemIStream::new(bytes[chunks_start..].to_vec());
        let _dx = xdr::read_i32(&mut cis).unwrap();
        let dy = xdr::read_i32(&mut cis).unwrap();
        assert_eq!(dy, 2, "DECREASING_Y must write the top tile row (dy=2) first");
    }

    /// `TiledOutputFile::from_part` must reject a
    /// header whose `type` is not `tiledimage` instead of silently retyping
    /// it. C++ 3.4.15 `TiledOutputFile::TiledOutputFile(const
    /// OutputPartData*)` (`ImfTiledOutputFile.cpp:911-917`) throws
    /// `ArgExc("Can't build a TiledOutputFile from a type-mismatched
    /// part.")` for exactly this before `initialize` ever runs.
    #[test]
    fn from_part_rejects_type_mismatched_header() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let mut header = make_header(4, 4, LineOrder::IncreasingY, td);
        header.set_type("deeptile").unwrap();

        let mut out = MemOStream::new();
        match TiledOutputFile::from_part(&mut out, &header, Some(0), 0) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("expected an error, got Ok"),
        }
    }

    /// The C++ streaming-writer
    /// pattern: ONE tile-sized buffer, refilled IN PLACE and passed to
    /// `write_tile` fresh each call (no per-image-sized buffer; no stored
    /// borrow on the writer at all — `set_frame_buffer` only installs the
    /// LAYOUT once). Confirms each tile gets its own current content, not
    /// some earlier call's — this is the streaming case the withdrawn
    /// "writer stores `&'a [u8]`" design made impossible (that design froze
    /// the buffer for the writer's whole life, so it could never be
    /// mutated/reused like this).
    #[test]
    fn write_tile_reuses_one_buffer_refilled_in_place_per_call() {
        use crate::tiled::TiledInputFile;

        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);

        // ONE 2x2-tile-sized buffer (not a 4x4-image-sized one), addressed
        // tile-coordinate-relative so the same small buffer works for every
        // tile — exactly the C++ pattern this finding describes.
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

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();

            // ONE buffer, reused and refilled in place for every tile —
            // impossible under the withdrawn stored-borrow design.
            let mut tile_buf = vec![0u8; 8];
            let values = [11.0f32, 22.0, 33.0, 44.0];
            let coords = [(0, 0), (1, 0), (0, 1), (1, 1)];
            for (&(dx, dy), &v) in coords.iter().zip(values.iter()) {
                let bits = Half::from(v).bits();
                for px in tile_buf.as_chunks_mut::<2>().0 {
                    px.copy_from_slice(&bits.to_ne_bytes());
                }
                tof.write_tile(&tile_buf, dx, dy, 0, 0).unwrap();
            }
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        let mut rfb = FrameBuffer::new();
        let mut readback = vec![0u8; 8]; // one 2x2 tile, HALF (2 bytes/px)
        rfb.insert(
            "R",
            Slice {
                type_: PixelType::Half,
                base_offset: 0,
                x_stride: 2,
                y_stride: 4,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: true,
                y_tile_coords: true,
            },
        )
        .unwrap();
        tif.set_frame_buffer(&rfb).unwrap();

        let coords = [(0, 0), (1, 0), (0, 1), (1, 1)];
        let values = [11.0f32, 22.0, 33.0, 44.0];
        for (&(dx, dy), &expected) in coords.iter().zip(values.iter()) {
            tif.read_tile(dx, dy, 0, 0, &mut readback).unwrap();
            for px in readback.as_chunks::<2>().0 {
                let v = Half::from_bits(u16::from_ne_bytes(*px));
                assert_eq!(
                    f32::from(v),
                    expected,
                    "tile ({dx},{dy}) must have its own in-place-refilled data"
                );
            }
        }
    }

    /// Writing tiles out of the file's
    /// line order must still produce the SAME on-disk chunk order as writing
    /// them in order, for `INCREASING_Y`. Port of C++
    /// `TiledOutputFile::Data::{tileMap, nextTileToWrite}` /
    /// `bufferedTileWrite` (`ImfTiledOutputFile.cpp:441-556`): a tile written
    /// before its turn is buffered until the tiles ahead of it arrive.
    #[test]
    fn increasing_y_out_of_order_writes_match_in_order_byte_for_byte() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        // Reference: tiles written in the file's natural line order —
        // (0,0), (1,0), (0,1), (1,1) — via write_tiles' own walk.
        let mut in_order = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut in_order, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let in_order_bytes = in_order.into_data();

        // Same tiles, called in REVERSE order: (1,1), (0,1), (1,0), (0,0).
        // Every one of the first three calls must be buffered (none is the
        // awaited (0,0,0,0) yet); the last call (0,0) then flushes all four
        // in the correct order in one go.
        let mut reversed = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut reversed, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tile(&pixels, 1, 1, 0, 0).unwrap();
            tof.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            tof.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            tof.write_tile(&pixels, 0, 0, 0, 0).unwrap();
        }
        let reversed_bytes = reversed.into_data();

        assert_eq!(
            in_order_bytes, reversed_bytes,
            "INCREASING_Y must write tiles to disk in line order regardless of call order"
        );
    }

    /// The DECREASING_Y counterpart: on-disk order must match writing
    /// in line order (dy descending) even when the caller writes dy
    /// ascending (the reverse of the file's line order).
    #[test]
    fn decreasing_y_out_of_order_writes_match_in_order_byte_for_byte() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::DecreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut in_order = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut in_order, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let in_order_bytes = in_order.into_data();

        // DECREASING_Y line order is dy=1 row first, then dy=0. Call with
        // dy=0 first (out of turn) so both dy=0 tiles must be buffered
        // until the dy=1 row is written.
        let mut reversed = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut reversed, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tile(&pixels, 0, 0, 0, 0).unwrap();
            tof.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            tof.write_tile(&pixels, 1, 1, 0, 0).unwrap();
            tof.write_tile(&pixels, 0, 1, 0, 0).unwrap();
        }
        let reversed_bytes = reversed.into_data();

        assert_eq!(
            in_order_bytes, reversed_bytes,
            "DECREASING_Y must write tiles to disk in line order regardless of call order"
        );
    }

    /// A tile still buffered when the writer is dropped is simply
    /// discarded (never written), matching C++'s destructor (frees `tileMap`
    /// without flushing it): the file is missing that tile's chunk and its
    /// offset-table slot stays 0.
    #[test]
    fn tile_still_buffered_at_drop_is_discarded() {
        use crate::tiled::TiledInputFile;

        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            // Write everything EXCEPT the awaited first tile (0,0): every
            // one of these stays buffered forever.
            tof.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            tof.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            tof.write_tile(&pixels, 1, 1, 0, 0).unwrap();
            // Dropped here without writing (0,0) — the buffered tiles are
            // discarded along with the writer.
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        assert!(
            !tif.is_complete(),
            "a file with buffered-but-never-flushed tiles must be incomplete"
        );
        // The never-written (0,0) tile has no chunk to read.
        let err = tif.read_tile(0, 0, 0, 0, &mut [0u8; 2 * 2 * 2]);
        assert!(err.is_err(), "tile (0,0) was never written and must not read successfully");
    }

    /// `file_name`, `break_tile`,
    /// `update_preview_image` accessors added to the writer.
    #[test]
    fn tiled_writer_accessors() {
        use crate::attr::{PreviewImage, PreviewImageAttribute, PreviewRgba};
        use crate::tiled::TiledInputFile;

        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let mut header = make_header(4, 4, LineOrder::IncreasingY, td);
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

        let (pixels, fb) = make_pixels(4, 4);
        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(tof.file_name(), "(memory)");

            tof.set_frame_buffer(&fb).unwrap();
            tof.write_tile(&pixels, 0, 0, 0, 0).unwrap();

            // `break_tile`: overwrite 1 byte of the already-written (0,0)
            // tile's chunk, well past the 5*i32 leader, then confirm the
            // read side actually sees the corruption (proves this hits the
            // real chunk bytes, not a no-op).
            tof.break_tile(0, 0, 0, 0, 5 * 4, 1, 0xFF).unwrap();

            // `break_tile` on a never-written tile must error.
            assert!(tof.break_tile(1, 1, 0, 0, 0, 1, 0).is_err());

            // `update_preview_image`: replace both pixels, then finish
            // writing the rest of the tiles.
            tof.update_preview_image(&[
                PreviewRgba::new(9, 9, 9, 9),
                PreviewRgba::new(8, 8, 8, 8),
            ])
            .unwrap();
            // Wrong pixel count must error, not silently truncate/overrun.
            assert!(tof.update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)]).is_err());

            tof.write_tile(&pixels, 1, 0, 0, 0).unwrap();
            tof.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            tof.write_tile(&pixels, 1, 1, 0, 0).unwrap();
        }
        let bytes = out.into_data();

        // The header's preview attribute on disk must reflect the update.
        let mut header_is = MemIStream::new(bytes.clone());
        let (read_back_header, _version_word) = Header::read_checked(&mut header_is).unwrap();
        let pia = read_back_header
            .typed_attribute::<PreviewImage>("preview")
            .unwrap();
        assert_eq!(
            pia.value.pixels(),
            &[PreviewRgba::new(9, 9, 9, 9), PreviewRgba::new(8, 8, 8, 8)]
        );

        // Tile (0,0) is byte-corrupted (NONE compression => the reader takes
        // the raw-passthrough path and returns the corrupted bytes as-is
        // rather than failing decompression), so compare against the
        // corrupted plane explicitly instead of the original pixel values.
        let mut is = MemIStream::new(bytes);
        let mut tif = TiledInputFile::new(&mut is).unwrap();
        let (.., raw) = tif.raw_tile_data(0, 0, 0, 0).unwrap();
        assert_eq!(raw[0], 0xFF, "break_tile must have corrupted the chunk's first payload byte");
    }

    /// `copy_pixels` (`TiledOutputFile::copyPixels`):
    /// write a 4x4 (2x2 tiles) source file, quick-copy it into a fresh
    /// `TiledOutputFile` with a matching header, and confirm the destination's
    /// on-disk chunk bytes are byte-identical to the source's (raw copy, no
    /// decompress/recompress) — for both `INCREASING_Y` (walks
    /// `next_tile_coord`) and `RANDOM_Y` (walks the input's own
    /// `tile_order()`).
    #[test]
    fn copy_pixels_tiled_to_tiled_byte_identical_increasing_and_random_y() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (pixels, fb) = make_pixels(4, 4);

        for line_order in [LineOrder::IncreasingY, LineOrder::RandomY] {
            let mut header = make_header(4, 4, line_order, td);
            // ZIP (not NONE) so a byte-identical result actually proves
            // `copy_pixels` never decompresses+recompresses — with NONE the
            // "raw" and "compressed" bytes are the same thing either way.
            *header.compression_mut().unwrap() = Compression::Zip;

            let mut src_out = MemOStream::new();
            {
                let mut src = TiledOutputFile::new(&mut src_out, &header).unwrap();
                src.set_frame_buffer(&fb).unwrap();
                // Write in an arbitrary call order (RANDOM_Y allows any
                // order; INCREASING_Y's own out-of-order buffering makes this
                // in-order on disk regardless of call order).
                src.write_tile(&pixels, 1, 1, 0, 0).unwrap();
                src.write_tile(&pixels, 0, 0, 0, 0).unwrap();
                src.write_tile(&pixels, 1, 0, 0, 0).unwrap();
                src.write_tile(&pixels, 0, 1, 0, 0).unwrap();
            }
            let src_bytes = src_out.into_data();

            let mut src_is = MemIStream::new(src_bytes.clone());
            let mut src_in = TiledInputFile::new(&mut src_is).unwrap();

            let mut dst_out = MemOStream::new();
            {
                let mut dst = TiledOutputFile::new(&mut dst_out, &header).unwrap();
                dst.copy_pixels(&mut src_in).unwrap();
            }
            let dst_bytes = dst_out.into_data();

            assert_eq!(
                src_bytes, dst_bytes,
                "copy_pixels must reproduce the source file byte-for-byte ({line_order:?})"
            );
        }
    }

    /// `copy_pixels` must reject copying into a file that already has pixel
    /// data (C++ LogicExc "already contains pixel data").
    #[test]
    fn copy_pixels_rejects_when_destination_already_has_pixel_data() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut src_out = MemOStream::new();
        {
            let mut src = TiledOutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = TiledInputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        let mut dst = TiledOutputFile::new(&mut dst_out, &header).unwrap();
        dst.set_frame_buffer(&fb).unwrap();
        dst.write_tile(&pixels, 0, 0, 0, 0).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)));
    }

    /// `copy_pixels` must reject mismatched tile descriptions (C++ ArgExc
    /// "the files have different tile descriptions").
    #[test]
    fn copy_pixels_rejects_mismatched_tile_description() {
        let td_a = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let td_b = TileDescription {
            x_size: 4,
            y_size: 4,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (pixels, fb) = make_pixels(4, 4);

        let src_header = make_header(4, 4, LineOrder::IncreasingY, td_a);
        let mut src_out = MemOStream::new();
        {
            let mut src = TiledOutputFile::new(&mut src_out, &src_header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = TiledInputFile::new(&mut src_is).unwrap();

        let dst_header = make_header(4, 4, LineOrder::IncreasingY, td_b);
        let mut dst_out = MemOStream::new();
        let mut dst = TiledOutputFile::new(&mut dst_out, &dst_header).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    /// `copy_pixels` reached through the `InputFile` (scanline dispatcher)
    /// front door on a tiled part — C++ `TiledOutputFile::copyPixels(InputFile&)`
    /// (`in.asTiledInput()`).
    #[test]
    fn copy_pixels_via_scanline_input_file_as_tiled_input() {
        let td = TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let header = make_header(4, 4, LineOrder::IncreasingY, td);
        let (pixels, fb) = make_pixels(4, 4);

        let mut src_out = MemOStream::new();
        {
            let mut src = TiledOutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_tiles(&pixels, 0, 1, 0, 1, 0, 0).unwrap();
        }
        let src_bytes = src_out.into_data();

        let mut src_is = MemIStream::new(src_bytes.clone());
        let mut generic_in = crate::scanline::InputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        {
            let mut dst = TiledOutputFile::new(&mut dst_out, &header).unwrap();
            dst.copy_pixels(&mut generic_in).unwrap();
        }
        assert_eq!(src_bytes, dst_out.into_data());
    }

    #[test]
    fn owned_and_borrowed_tiled_copy_sources_share_raw_chunks_and_validation() {
        let td = TileDescription {
            x_size: 16,
            y_size: 16,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (pixels, fb) = make_pixels(64, 32);
        for order in [LineOrder::IncreasingY, LineOrder::RandomY] {
            let mut header = make_header(64, 32, order, td);
            *header.compression_mut().unwrap() = Compression::Zip;
            let mut stream = MemOStream::new();
            {
                let mut output = TiledOutputFile::new(&mut stream, &header).unwrap();
                output.set_frame_buffer(&fb).unwrap();
                for dy in (0..2).rev() {
                    for dx in (0..4).rev() {
                        output.write_tile(&pixels, dx, dy, 0, 0).unwrap();
                    }
                }
            }
            let bytes = stream.into_data();
            let mut borrowed_stream = MemIStream::new(bytes.clone());
            let mut borrowed = TiledInputFile::new(&mut borrowed_stream).unwrap();
            assert!(borrowed.raw_tile_data(0, 0, 0, 0).unwrap().4.len() < 16 * 16 * 2);
            let mut owned = crate::tiled::input::TiledInputFileOwned::new_owned(Box::new(
                MemIStream::new(bytes.clone()),
            ))
            .unwrap();
            let mut generic =
                crate::scanline::InputFile::new_owned(Box::new(MemIStream::new(bytes.clone())))
                    .unwrap();
            fn verify<T: TiledCopySource>(source: &mut T, header: &Header, bytes: &[u8]) {
                let mut copied = MemOStream::new();
                {
                    let mut output = TiledOutputFile::new(&mut copied, header).unwrap();
                    output.copy_pixels(source).unwrap();
                }
                assert_eq!(bytes, copied.into_data());
                let mut mismatched = header.clone();
                *mismatched.compression_mut().unwrap() = Compression::NoCompression;
                let mut rejected = MemOStream::new();
                let mut output = TiledOutputFile::new(&mut rejected, &mismatched).unwrap();
                assert!(output.copy_pixels(source).is_err());
            }
            verify(&mut borrowed, &header, &bytes);
            verify(&mut owned, &header, &bytes);
            verify(&mut generic, &header, &bytes);
        }
    }

    /// The tiled ring really overlaps: tiles are fed while later tiles are
    /// still encoding. Mutation: emit only after the window is produced ->
    /// this fails.
    #[test]
    fn parallel_write_feeds_while_later_tiles_encode() {
        let td = TileDescription {
            x_size: 16,
            y_size: 16,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (256i32, 256i32);
        let (pixels, fb) = make_pixels(w, h);
        for &n in &[2i32, 8] {
            assert!(
                crate::probe::saw_overlap(|| {
                    let mut header = make_header(w, h, LineOrder::IncreasingY, td);
                    *header.compression_mut().unwrap() = Compression::Zip;
                    let mut out = MemOStream::new();
                    let mut tof = TiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                    tof.set_frame_buffer(&fb).unwrap();
                    let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                    tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
                }),
                "num_threads={n}: no tile was fed while another was still encoding"
            );
        }
    }

    /// Writer-side pool reuse for the tiled ring: at most one codec per worker
    /// over 64 tiles (8 KB each, so the byte-sized ring window stays under 64 and a
    /// missing `give` shows) and no missed pooled buffer. Mutations: drop the lease's
    /// return / `pool.give` -> this fails.
    #[test]
    fn parallel_write_reuses_codecs_and_buffers() {
        use crate::probe::{get, reset, Probe};
        let td = TileDescription {
            x_size: 64,
            y_size: 64,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let (w, h) = (512i32, 512i32);
        let (pixels, fb) = make_pixels(w, h);
        for &n in &[2i32, 8] {
            for p in [Probe::CodecCreated, Probe::BufPoolMiss, Probe::BufPoolHit] {
                reset(p);
            }
            let mut header = make_header(w, h, LineOrder::IncreasingY, td);
            *header.compression_mut().unwrap() = Compression::Zip;
            let mut out = MemOStream::new();
            {
                let mut tof = TiledOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                tof.set_frame_buffer(&fb).unwrap();
                let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
            }
            assert!(get(Probe::CodecCreated) <= n as u64, "num_threads={n}: {} codecs built for 64 tiles", get(Probe::CodecCreated));
            assert_eq!(get(Probe::BufPoolMiss), 0, "num_threads={n}: a buffer was not recycled");
            assert_eq!(get(Probe::BufPoolHit), 64, "num_threads={n}: every tile takes a pooled buffer");
        }
    }
}
