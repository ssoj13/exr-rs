//! Port of OpenEXR `ImfOutputFile.cpp` — `class OutputFile`, the
//! **single-part, scanline** writer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOutputFile.cpp>
//!
//! # What this ports (1:1 with C++)
//!
//! * The constructor's *header + magic + version* write (reusing
//!   [`Header::write_to`], which folds in
//!   `GenericOutputFile::writeMagicNumberAndVersionField`), the line-offset
//!   table *reservation* (`writeLineOffsets` of an all-zero vector), and the
//!   bookkeeping done by `OutputFile::initialize`.
//! * [`OutputFile::set_frame_buffer`] — the `setFrameBuffer` compatibility
//!   checks plus the per-channel `OutSliceInfo` slice table (including the
//!   "channel absent → write zeroes" case).
//! * [`OutputFile::write_pixels`] — the `writePixels(numScanLines)` loop:
//!   for each scanline it gathers every channel's samples out of the user
//!   frame buffer with [`copy_from_frame_buffer`] (or
//!   [`fill_channel_with_zeroes`] for absent channels), runs a complete block
//!   through the codec [`new_compressor`] built (or the raw NONE path), and
//!   writes the chunk: `y` (i32) + `dataSize` (i32) + data, exactly as
//!   `writePixelData` does for a non-multipart scanline file. Both
//!   `INCREASING_Y` / `DECREASING_Y` line orders are honoured.
//! * The line-offset table back-patch: C++ rewrites the now-filled `u64`
//!   offsets from its destructor.  This port writes each chunk's file offset
//!   into the reserved table slot immediately after the chunk is written
//!   (seek-back, write `u64`, seek-forward), which produces byte-identical file
//!   contents without relying on `Drop` (the borrowed stream cannot be
//!   finalised from `Drop` safely).
//! * [`OutputFile::update_preview_image`] / [`OutputFile::break_scan_line`] —
//!   `updatePreviewImage`/`breakScanLine` (the latter a test-only corrupt-chunk
//!   helper).
//! * [`OutputFile::copy_pixels`] — `copyPixels(InputFile&)` /
//!   `copyPixels(InputPart&)`: a raw (never decompress/recompress) chunk
//!   transcode from a matching-header scanline source, mirroring
//!   [`crate::tiled::output::TiledOutputFile::copy_pixels`]'s design (see
//!   [`ScanLineCopySource`]).

use crate::attr::PixelType;
use crate::compressor::{
    default_format, new_compressor, num_lines_in_buffer, stored_chunk, Compressor, Format,
};
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::frame_buffer::FrameBuffer;
use crate::header::Header;
use crate::io::OStream;
use crate::misc::{
    bytes_per_line_table, copy_from_frame_buffer, fill_channel_with_zeroes,
    line_buffer_max_y, line_buffer_min_y, offset_in_line_buffer_table, try_filled_vec,
    BufferFormat,
};

use imath_rs::{divp, modp};
use std::marker::PhantomData;
use std::ops::DerefMut;

// ---------------------------------------------------------------------------
// ScanLineCopySource — the `copyPixels` source abstraction
// ---------------------------------------------------------------------------

/// Anything [`OutputFile::copy_pixels`] can quick-copy pixels FROM — one impl
/// per C++ `OutputFile::copyPixels` overload's source type (`InputFile&`,
/// `InputPart&`). Mirrors
/// [`crate::tiled::output::TiledCopySource`]'s duck-typed, no-lifetime shape
/// (see that trait's doc comment for why: a borrowing cached-part-reader view
/// cannot hand back a concrete owned reader type, so the trait exposes only
/// the operations `copy_pixels` actually performs).
pub trait ScanLineCopySource {
    /// This source's header.
    fn header(&self) -> &Header;

    /// The chunk covering `first_scan_line`'s raw (still-packed) bytes, plus
    /// the chunk-header-verified `y` — C++ `rawPixelData`.
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)>;
}

impl<'a, S> ScanLineCopySource for super::input::ScanLineInputFile<'a, S>
where
    S: DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    fn header(&self) -> &Header {
        super::input::ScanLineInputFile::header(self)
    }
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        super::input::ScanLineInputFile::raw_pixel_data(self, first_scan_line)
    }
}

impl<'a, S> ScanLineCopySource for super::input_file::InputFile<'a, S>
where
    S: DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    fn header(&self) -> &Header {
        super::input_file::InputFile::header(self)
    }
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        super::input_file::InputFile::raw_pixel_data(self, first_scan_line)
    }
}

// ---------------------------------------------------------------------------
// OutSliceInfo — port of the anonymous-namespace `struct OutSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel write descriptor — port of `OutSliceInfo` (ImfOutputFile.cpp
/// lines 58-89).
///
/// Built by [`OutputFile::set_frame_buffer`] from the file's channel list and
/// the user frame buffer.  Mirrors the C++ struct field-for-field, except that
/// the raw `const char* base` becomes a signed byte offset
/// (`base_offset`) into the caller-owned pixel buffer — the same safe-Rust
/// adaptation used by [`crate::frame_buffer::Slice`].
#[derive(Clone, Debug)]
struct OutSliceInfo {
    /// Sample type stored on disk for this channel.
    type_: PixelType,
    /// Signed byte offset that replaces the C++ `char* base` (origin already
    /// baked in; `0`/unused when `zero` is `true`).
    base_offset: i64,
    /// Distance in bytes between horizontally adjacent samples in the frame
    /// buffer (`slice.xStride`).
    x_stride: usize,
    /// Distance in bytes between vertically adjacent rows (`slice.yStride`).
    y_stride: usize,
    /// Horizontal sub-sampling factor.
    x_sampling: i32,
    /// Vertical sub-sampling factor.
    y_sampling: i32,
    /// `true` when the file channel has no matching frame-buffer slice; the
    /// line buffer is filled with zeroes for this channel.
    zero: bool,
}

// ---------------------------------------------------------------------------
// OutputFile — port of `class OutputFile`
// ---------------------------------------------------------------------------

/// Single-part scanline EXR writer — port of `Imf::OutputFile`.
///
/// Construct with [`OutputFile::new`] (writes the header + reserves the
/// line-offset table), describe the source pixels with
/// [`OutputFile::set_frame_buffer`], then push scanlines with
/// [`OutputFile::write_pixels`].
///
/// Unlike C++ (which owns the stream and finalises the offset table from its
/// destructor), this port back-patches each offset-table entry immediately
/// after writing the corresponding chunk, so the file's structure is valid once all
/// scanlines have been written; its bytes reach the file when the stream is flushed
/// ([`OutputFile::finish`] checks completeness and flushes).
///
/// # Stream ownership (borrowed vs owned)
///
/// The writer is generic over how it holds its output stream, `S`:
/// * the **default** `S = &'a mut dyn OStream` keeps the original
///   borrow-for-lifetime model (C++ `_streamData->os`) — `OutputFile<'a>` is
///   unchanged and fully source-compatible; and
/// * `S = Box<dyn OStream>` (the [`OutputFileOwned`] alias, built with
///   [`new_owned`](OutputFile::new_owned)) lets a long-lived streaming writer
///   **own** its stream, so an enclosing struct can hold both the stream and
///   the writer without being self-referential. Both variants share the
///   identical write path and produce byte-identical files.
pub struct OutputFile<'a, S = &'a mut dyn OStream> {
    /// Output stream holder — a borrowed `&'a mut dyn OStream` (default) or an
    /// owned `Box<dyn OStream>`, accessed uniformly through
    /// `DerefMut<Target = dyn OStream>` (the C++ `_streamData->os`).
    stream: S,

    /// Every field of C++ `OutputFile::Data` EXCEPT the stream itself —
    /// factored out so the same state can be held either OWNED (here) or
    /// BORROWED (see [`OutputPartWriter`], the multipart resumption
    /// view). See [`OutputFileState`]'s own docs.
    state: OutputFileState,

    /// Ties the `'a` lifetime of the default borrowed-stream variant into the
    /// struct.  Zero-sized; for the owned variant (`S = Box<dyn OStream>`) `'a`
    /// is `'static` and this carries no borrow.
    _stream_lt: PhantomData<&'a mut ()>,
}

/// Every field of C++ `OutputFile::Data` except the stream — the writer's
/// **persisted** per-part state.
///
/// Ownership inversion:
/// `MultiPartOutputFile::output_part(n)` used to build a brand-new
/// [`OutputFile`] from the header on EVERY call, silently resetting
/// `current_scan_line` / `missing_scan_lines` / the partial line-buffer block
/// — C++ instead caches ONE `OutputFile` object per part
/// (`ImfMultiPartOutputFile.cpp:313-321`) so writes interleaved across parts
/// resume correctly. Splitting the writer into this pure state struct plus
/// the stream lets `MultiPartOutputFile` persist a `Vec<Option<
/// OutputFileState>>` (one slot per scanline part, built on first access) and
/// hand out a borrowing [`OutputPartWriter`] view — `{ &mut dyn OStream,
/// &mut OutputFileState }` — instead of manufacturing a fresh, amnesiac
/// writer every call.  The standalone [`OutputFile`] is unaffected: it just
/// owns one `OutputFileState` for its whole life, same as before this split.
pub(crate) struct OutputFileState {
    /// The image header (already type-fixed to `scanlineimage` if it carried a
    /// `type` attribute) — C++ `_data->header`.
    header: Header,

    /// Line order copied from the header — C++ `_data->lineOrder`.
    line_order: crate::attr::LineOrder,

    /// Data-window bounds — C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    /// `max_y` mirrors C++ `_data->maxY` — clamps the final (possibly
    /// partial) block's line range in [`write_pixels`](Self::write_pixels).
    max_y: i32,

    /// Next scanline to be written — C++ `_data->currentScanLine`.
    current_scan_line: i32,
    /// Number of scanlines still to write — C++ `_data->missingScanLines`.
    missing_scan_lines: i32,

    /// Compressor data format (Xdr for the NONE path) — C++ `_data->format`.
    format: Format,
    /// Scanlines per line-buffer block (1 for NONE) — C++ `_data->linesInBuffer`.
    lines_in_buffer: i32,

    /// Bytes per scanline, indexed by `y - min_y` — C++ `_data->bytesPerLine`.
    ///
    /// Only consulted at construction (sizing the line buffer / building
    /// [`offset_in_line_buffer`](Self::offset_in_line_buffer)); the write path
    /// derives each chunk's size from the gather cursor instead. Kept for
    /// structural fidelity with the C++ `Data` struct (C++'s own
    /// `convertToXdr` reads it per write, but every codec in this crate
    /// reports `Format::Xdr`, so that branch is unreachable here — see
    /// `debug_assert_eq!(self.format, Format::Xdr)` in `write_pixels`).
    #[allow(dead_code)]
    bytes_per_line: Vec<usize>,
    /// Byte offset of each scanline inside its line buffer, indexed by
    /// `y - min_y` — C++ `_data->offsetInLineBuffer`.
    offset_in_line_buffer: Vec<usize>,

    /// File offsets of each chunk — C++ `_data->lineOffsets`.
    line_offsets: Vec<u64>,
    /// File position where the offset table begins — C++
    /// `_data->lineOffsetsPosition`.
    line_offsets_position: u64,

    /// File position where the header's `preview` attribute's VALUE bytes
    /// begin, or `0` if the header has no preview image — C++
    /// `_data->previewPosition`. Feeds [`Self::update_preview_image`]. Set by
    /// [`new_standalone`](Self::new_standalone) (the real value from
    /// `Header::write_to`) or by
    /// [`from_part_with_preview`](Self::from_part_with_preview) (the real
    /// value `MultiPartOutputFile` captured from `Header::write_attributes_to`,
    /// see `from_part_with_preview_mt`); `0` only via the plain
    /// [`from_part`](Self::from_part), used by the public `OutputFile::from_part`
    /// constructor, which does not itself write any header and so has no real
    /// value to supply. Mirrors
    /// `tiled::output::TiledOutputFileState::preview_position` exactly.
    preview_position: u64,

    /// Per-channel write descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][OutputFile::set_frame_buffer] is called).
    slices: Vec<OutSliceInfo>,

    /// Reusable scratch line buffer (the C++ `LineBuffer::buffer`).  For a
    /// multi-line codec (ZIP, 16 lines) this holds an entire block's worth of
    /// gathered scanlines before it is compressed and written as one chunk.
    line_buffer: Vec<u8>,

    /// Accumulated size of the data written into [`line_buffer`](Self::line_buffer)
    /// for the block currently being gathered — C++
    /// `LineBuffer::endOfLineBufferData - buffer`.  Reset to 0 when a new block
    /// begins; grows as each scanline of the block is gathered.
    block_data_size: usize,

    /// Count of scanlines gathered so far into the current partial block — used
    /// to decide when the block is complete (== the number of lines the block
    /// spans within the data window) and must be flushed as one chunk.  Mirrors
    /// the net effect of C++ `LineBuffer::partiallyFull` across `writePixels`
    /// calls.
    block_lines_gathered: i32,

    /// The scanline compressor, or `None` for `NO_COMPRESSION` (the raw path).
    ///
    /// Mirrors the C++ `LineBuffer::compressor` — created once in the
    /// constructor by [`new_compressor`].  When present, each gathered line
    /// buffer is run through it before the chunk is written; the codec's own
    /// "no-shrink → store raw" rule means the returned slice may be the raw
    /// input, in which case `dataSize` equals the uncompressed size (exactly
    /// what the reader uses to detect the raw passthrough).
    compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this writer drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `OutputFile::Data::{multiPart, partNumber}`.
    ///
    /// This is the single seam that unifies the single-part and multipart chunk
    /// codepaths: when `Some`, every chunk is prefixed with the `partNumber`
    /// (i32 LE) exactly as C++ `writePixelData` does for a multipart file; when
    /// `None`, no prefix is written, so single-part output is byte-identical to
    /// the pre-multipart writer.
    part_number: Option<i32>,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the reader's identical field
    /// (`scanline/input.rs`).
    num_threads: crate::threading::ThreadCount,

    /// The codec (C++ `_data->_compression`) — kept so the parallel path can
    /// build a fresh per-worker codec instance via [`new_compressor`] (each
    /// worker owns one; codecs are not `Sync`).
    compression: crate::attr::Compression,

    /// Bytes of one scanline's widest channel layout — reused by the
    /// parallel path to size fresh per-worker codec instances.
    max_bytes_per_line: usize,
}

/// An [`OutputFile`] that **owns** its output stream (`Box<dyn OStream>`).
///
/// Use this when the scanline writer must outlive the call that created its
/// stream — a long-lived streaming writer that owns both the stream and the
/// writer in one struct (impossible with the borrowing default without a
/// self-referential struct).  Build it with [`OutputFile::new_owned`]; the
/// produced file is byte-identical to the borrowed path for the same input.
/// `'a` here means "how long the boxed stream itself must live" — pixel data
/// carries no lifetime at all (by design: `write_pixels` takes the bytes
/// per call rather than `set_frame_buffer` storing a borrow of them; see that
/// method's docs). The stream slot is `Box<dyn OStream + 'a>` rather than the
/// implicit `Box<dyn OStream + 'static>` so its `DerefMut::Target` is
/// literally `dyn OStream + 'a` (an exact match for `build`'s bound, not a
/// subtyping coercion — `Box<dyn OStream + 'static>` cannot satisfy
/// `S: DerefMut<Target = dyn OStream + 'a>` for a free `'a`; verified by
/// trying it). Ordinary `'static` streams (`StdOfStream`, `MemOStream`) still
/// coerce into a `Box<dyn OStream + 'a>` argument at the `new_owned` call site.
pub type OutputFileOwned<'a> = OutputFile<'a, Box<dyn OStream + 'a>>;

// Borrowed-stream constructors (the default `S = &'a mut dyn OStream`).
impl<'a> OutputFile<'a, &'a mut dyn OStream> {
    /// Open an `OutputFile` on `stream` for `header` — port of
    /// `OutputFile::OutputFile(OStream&, const Header&, int)` restricted to the
    /// single-part, scanline path.
    ///
    /// Validates the header (`sanity_check`), then writes the magic number,
    /// version word and header, and reserves the line-offset table — exactly
    /// the constructor body of C++ `OutputFile`.
    ///
    /// # Errors
    /// * Propagates [`Header::sanity_check`] failures.
    /// * Propagates [`new_compressor`] errors (e.g. an invalid compression).
    /// * Propagates stream write / position errors.
    pub fn new(stream: &'a mut dyn OStream, header: &Header) -> Result<Self> {
        // Single-part: the writer itself emits magic + version + header +
        // offset table, and no per-chunk partNumber prefix (`part_number =
        // None`).  `line_offsets_position` is discovered inside `build`.
        Self::build(stream, header, None, None, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(
        stream: &'a mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::build(stream, header, None, None, num_threads)
    }

    /// Open an `OutputFile` for **one part of a multipart file** — port of
    /// `OutputFile::OutputFile(const OutputPartData*)` for the scanline part.
    ///
    /// The enclosing [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile)
    /// has already written the file's single magic + version word, every part
    /// header, and the (zero-filled) per-part chunk offset tables.  This
    /// constructor therefore does **not** write any of those; it only rebuilds
    /// the per-part write state and remembers:
    ///
    /// * `part_number` — written as the i32 prefix of every chunk
    ///   (C++ `Data::partNumber`, `multiPart == true`), and
    /// * `chunk_offset_table_position` — the file position of *this part's*
    ///   reserved offset table (C++ `OutputPartData::chunkOffsetTablePosition`),
    ///   into which each chunk's start offset is back-patched.
    ///
    /// The stream cursor must already be positioned where this part's first
    /// chunk should be written (the container leaves it just past the last
    /// offset table).
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOutputFile.cpp> (`OutputFile(const OutputPartData*)` + `writePixelData`).
    pub fn from_part(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        chunk_offset_table_position: u64,
    ) -> Result<Self> {
        Self::build(stream, header, part_number, Some(chunk_offset_table_position), None)
    }
}

// Owned-stream constructor (`S = Box<dyn OStream + 'a>`).  Generic over `'a`
// — see the [`OutputFileOwned`] doc for why this is no longer pinned
// to `'static`.
impl<'a> OutputFile<'a, Box<dyn OStream + 'a>> {
    /// Open an [`OutputFileOwned`] that **owns** its output stream — the
    /// owned-stream analogue of [`new`](OutputFile::new).
    ///
    /// Writes the header + reserves the line-offset table exactly as
    /// [`new`](OutputFile::new) does (byte-identical output), but takes the
    /// stream by value so the returned writer can be stored in a struct that
    /// also owns the stream — the long-lived streaming-writer use case that the
    /// borrowing default cannot express in safe Rust.  Pass a boxed concrete
    /// stream, e.g. `Box::new(StdOfStream::create(path)?)`.
    ///
    /// The file is valid as soon as every scanline has been written (offsets
    /// are back-patched per chunk); [`finish`](OutputFile::finish) is provided
    /// to recover the stream.
    ///
    /// # Errors
    /// Identical to [`new`](OutputFile::new).
    pub fn new_owned(stream: Box<dyn OStream + 'a>, header: &Header) -> Result<Self> {
        Self::build(stream, header, None, None, None)
    }
}

// Shared write path, generic over the stream holder `S`.  Covers BOTH the
// borrowed default (`S = &'a mut dyn OStream`) and the owned `Box<dyn OStream>`
// variant — every method below addresses the stream through
// `DerefMut<Target = dyn OStream>`, so the two paths are a single source of
// truth and emit byte-identical files.
impl<'a, S> OutputFile<'a, S>
where
    S: DerefMut<Target = dyn OStream + 'a>,
{
    /// Shared constructor body for [`new`](Self::new) (single-part),
    /// [`from_part`](Self::from_part) (multipart part) and
    /// [`new_owned`](OutputFile::new_owned) (owned single-part).  When
    /// `part_number` is `None` the file prologue (magic + version + header +
    /// reserved offset table) is written here and `line_offsets_position` is
    /// discovered from the stream; when `Some`, the container already wrote the
    /// prologue and `offsets_position` is this part's reserved offset-table
    /// position.
    ///
    /// The actual state computation now lives in
    /// [`OutputFileState::new_standalone`] / [`OutputFileState::from_part`] so
    /// [`OutputPartWriter`] (which BORROWS an `OutputFileState` instead of
    /// owning one) can share the exact same construction logic.
    fn build(
        mut stream: S,
        header: &Header,
        part_number: Option<i32>,
        offsets_position: Option<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = match offsets_position {
            None => OutputFileState::new_standalone_mt(&mut *stream, header, num_threads)?,
            Some(pos) => OutputFileState::from_part_mt(header, part_number, pos, num_threads)?,
        };
        Ok(Self {
            stream,
            state,
            _stream_lt: PhantomData,
        })
    }

    /// Consume the writer, check that the file is complete, flush the stream and hand it back.
    ///
    /// Every chunk offset is back-patched right after its chunk is written, so no table has to be rewritten
    /// here. What `finish` adds is the part dropping the writer cannot do:
    /// * it fails with [`ExrError::Logic`] if scanlines are still missing, because a half-written file is
    ///   otherwise indistinguishable from a good one at write time;
    /// * it calls [`OStream::flush`], so a failed final write (full disk, vanished network drive) is
    ///   RETURNED. A buffered file stream such as `StdOfStream` holds up to 256 KiB and its `Drop` can only
    ///   log a flush failure.
    ///
    /// For [`OutputFileOwned`] it returns the owned `Box<dyn OStream>`; for the borrowed default the original
    /// `&mut` reference. Dropping the writer without calling `finish` stays valid but gets neither check.
    pub fn finish(mut self) -> Result<S> {
        if self.state.missing_scan_lines > 0 {
            return Err(ExrError::Logic(format!(
                "scanline output is incomplete: {} scanlines were never written",
                self.state.missing_scan_lines
            )));
        }
        OStream::flush(&mut *self.stream)?;
        Ok(self.stream)
    }

    /// The image header — port of `const Header& OutputFile::header() const`.
    pub fn header(&self) -> &Header {
        &self.state.header
    }

    /// Next scanline to be written — port of `int OutputFile::currentScanLine()`.
    pub fn current_scan_line(&self) -> i32 {
        self.state.current_scan_line
    }

    /// Set the pixel LAYOUT — port of `OutputFile::setFrameBuffer`.
    ///
    /// Design note: the
    /// writer no longer holds any reference to the caller's pixel bytes at
    /// all. `frame_buffer` supplies only the per-channel addressing (byte
    /// offsets/strides/types) as small, cloned [`OutSliceInfo`] metadata; the
    /// actual pixel BYTES are passed to [`write_pixels`](Self::write_pixels)
    /// **on every call**. This is what makes the core C++ streaming idiom —
    /// refill ONE strip buffer, `writePixels(n)`, repeat — actually work: a
    /// stored `&'a [u8]` borrow (an earlier, withdrawn design) froze the buffer
    /// for the writer's whole life, which is incompatible with reusing one
    /// small buffer across many calls (essential at 16k-65k resolutions) and
    /// broke any writer reused across a loop with a fresh per-iteration
    /// buffer (`tests/parity_exr.rs::ours_write_tiled`'s per-MIPMAP-level
    /// loop, in the tiled sibling of this design).
    ///
    /// Performs the same compatibility checks as C++ (pixel type and
    /// x/y-sampling must match the header channel) and builds the per-channel
    /// `OutSliceInfo` table, including the "channel present in the header but
    /// absent from the frame buffer → write zeroes" case.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] when a frame-buffer slice's type or sampling
    /// is incompatible with the corresponding header channel.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write `num_scan_lines` scanlines — port of `OutputFile::writePixels`.
    /// See [`OutputFileState::write_pixels`] for the full algorithm doc.
    pub fn write_pixels(&mut self, pixels: &[u8], num_scan_lines: i32) -> Result<()> {
        self.state.write_pixels(&mut *self.stream, pixels, num_scan_lines)
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk — port of `void updatePreviewImage(const PreviewRgba
    /// newPixels[])`. See [`OutputFileState::update_preview_image`] for the
    /// full doc.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(&mut *self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written scanline's chunk,
    /// starting `offset` bytes into it, with the byte `c` — port of `void
    /// breakScanLine(int y, int offset, int length, char c)` (test-only
    /// corrupt-chunk helper). See [`OutputFileState::break_scan_line`] for
    /// the full doc.
    pub fn break_scan_line(&mut self, y: i32, offset: u64, length: usize, c: u8) -> Result<()> {
        self.state.break_scan_line(&mut *self.stream, y, offset, length, c)
    }

    /// Quick pixel copy from `input`, raw (no decompress/recompress) —
    /// port of `OutputFile::copyPixels(InputFile&)` /
    /// `copyPixels(InputPart&)`. See [`OutputFileState::copy_pixels`] for the
    /// full doc.
    pub fn copy_pixels<T: ScanLineCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(&mut *self.stream, input)
    }
}

// ---------------------------------------------------------------------------
// OutputFileState — the state-only construction + algorithm, shared by the
// owning `OutputFile` and the borrowing `OutputPartWriter`.
// ---------------------------------------------------------------------------

impl OutputFileState {
    /// Pure state computation shared by `new_standalone_mt` and
    /// `from_part_with_preview_mt` — everything C++ `OutputFile::
    /// initialize(header)` computes that does NOT depend on `line_offsets_position`
    /// (which the two callers fill in differently: written fresh vs. handed in
    /// by the container). Never touches the stream.
    ///
    /// Same as this doc's former `compute`, with an explicit
    /// override (`None` = capture the current process-wide
    /// default). See [`crate::threading::ThreadCount::capture`].
    fn compute_mt(
        header: &Header,
        part_number: Option<i32>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // C++: header.sanityCheck();  (single-part, not tiled)
        // For a multipart part the container already ran sanity_check with the
        // multipart flag; re-running the non-multipart check here is harmless
        // (it is a strict subset) and keeps a single construction path.
        header.sanity_check(false, false)?;

        // C++ OutputFile::initialize(header) -------------------------------
        let mut hdr = header.clone();

        // "fix" the type if it happens to be set incorrectly.
        // C++: if (hasType()) setType(SCANLINEIMAGE);
        if hdr.has_type() {
            hdr.set_type(crate::header::SCANLINE_IMAGE)?;
        }

        let data_window = hdr.data_window()?;
        let line_order = hdr.line_order()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        // C++: currentScanLine = (lineOrder == INCREASING_Y) ? min.y : max.y;
        let current_scan_line = if line_order == crate::attr::LineOrder::IncreasingY {
            min_y
        } else {
            max_y
        };

        let missing_scan_lines = max_y - min_y + 1;

        // C++: bytesPerLineTable(header, bytesPerLine) → maxBytesPerLine.
        let width = max_x - min_x + 1;
        let channels = hdr.channels()?;
        let mut bytes_per_line: Vec<usize> = Vec::new();
        let max_bytes_per_line =
            bytes_per_line_table(min_y, max_y, width, channels, &mut bytes_per_line)?;

        // C++: lineBuffers each get newCompressor(compression, maxBytesPerLine,
        // header).  For NONE this is Ok(None); any other codec is a Phase-5
        // Err which we propagate (never faked).
        let compression = hdr.compression()?;
        let compressor = new_compressor(compression, max_bytes_per_line, &hdr)?;

        // C++: format = defaultFormat(compressor);  (Xdr when no compressor)
        let format = default_format(compressor.as_deref());

        // C++: linesInBuffer = numLinesInBuffer(compressor).  For NONE this is
        // 1 (Compression::NoCompression.num_scanlines() == 1).  When a real
        // compressor exists, mirror C++'s `numLinesInBuffer(compressor)` via
        // the compression type.
        let lines_in_buffer = match compressor.as_deref() {
            Some(c) => c.num_scan_lines(),
            None => num_lines_in_buffer(compression)?,
        };

        // C++: lineBufferSize = maxBytesPerLine * linesInBuffer; buffer resized.
        // Checked (mirrors the reader's guard, `scanline/input.rs`):
        // a multi-line codec's `linesInBuffer` (DWA's 256) can overflow
        // `usize` on a 32-bit target even from an otherwise-plausible
        // `max_bytes_per_line`.
        let line_buffer_size = max_bytes_per_line
            .checked_mul(lines_in_buffer as usize)
            .ok_or_else(|| {
                ExrError::Arg("Line buffer size overflow (scan line size * lines per chunk).".into())
            })?;
        // Fallible allocations, not `vec![..]`
        // -- both are sized from header fields an attacker controls (image
        // height for `line_offsets`, height * codec block size for
        // `line_buffer`); see `try_filled_vec`'s docs for why an infallible
        // allocator call is the wrong tool for an attacker-controlled size.
        let line_buffer = try_filled_vec(line_buffer_size, 0u8)?;

        // C++: lineOffsetSize = (max.y - min.y + linesInBuffer) / linesInBuffer;
        //      lineOffsets.resize(lineOffsetSize);
        let line_offset_size =
            ((max_y - min_y + lines_in_buffer) / lines_in_buffer) as usize;
        let line_offsets = try_filled_vec(line_offset_size, 0u64)?;

        // C++: offsetInLineBufferTable(bytesPerLine, linesInBuffer, offsetInLineBuffer);
        let mut offset_in_line_buffer: Vec<usize> = Vec::new();
        offset_in_line_buffer_table(
            &bytes_per_line,
            lines_in_buffer as usize,
            &mut offset_in_line_buffer,
        );

        // Structural detail (debug): scanline writer is up — dimensions, codec,
        // lines-per-chunk, and the chunk-offset table size.
        log::debug!(
            "scanline OutputFile: {}x{} part={part_number:?} codec={compression:?} \
             linesInBuffer={lines_in_buffer} chunkOffsets={}",
            max_x - min_x + 1,
            max_y - min_y + 1,
            line_offsets.len()
        );

        Ok(Self {
            header: hdr,
            line_order,
            min_x,
            max_x,
            min_y,
            max_y,
            current_scan_line,
            missing_scan_lines,
            format,
            lines_in_buffer,
            bytes_per_line,
            offset_in_line_buffer,
            line_offsets,
            line_offsets_position: 0,
            preview_position: 0,
            slices: Vec::new(),
            line_buffer,
            block_data_size: 0,
            block_lines_gathered: 0,
            compressor,
            part_number,
            num_threads,
            compression,
            max_bytes_per_line,
        })
    }

    /// Standalone single-part construction: writes the magic + version word,
    /// the header, and reserves the (zero-filled) line-offset table on
    /// `stream` — port of the C++ `OutputFile` constructor body, minus the
    /// pure state computation shared with `from_part_with_preview_mt`.
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default).
    fn new_standalone_mt(
        stream: &mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let mut state = Self::compute_mt(header, None, num_threads)?;

        // C++: writeMagicNumberAndVersionField(os, header);
        //      previewPosition = header.writeTo(os);
        // Folded into Header::write_to (single-part → base flags 0).
        state.preview_position = state.header.write_to(stream, 0)?;

        // C++: lineOffsetsPosition = writeLineOffsets(os, lineOffsets);
        // Records the current position then writes every (still-zero) offset;
        // the position it returns is the table start.
        let pos = stream.tellp()?;
        for &off in &state.line_offsets {
            xdr::write_u64(stream, off)?;
        }
        state.line_offsets_position = pos;
        Ok(state)
    }

    /// Multipart-part construction — port of `OutputFile::OutputFile
    /// (const OutputPartData*)`. The container ([`MultiPartOutputFile`](
    /// crate::multipart::MultiPartOutputFile)) already wrote the file
    /// prologue and this part's reserved offset table; this only computes
    /// the pure per-part state and records where that table lives
    /// (`chunk_offset_table_position`, into which each chunk's start offset
    /// is back-patched). Touches no stream at all — this is what lets
    /// `MultiPartOutputFile` build this state ONCE per part and persist it
    /// across `output_part(n)` calls.
    /// Same construction, with an explicit `num_threads` override
    /// (`None` = capture the current process-wide default).
    /// `preview_position` stays 0: the public `OutputFile::from_part`
    /// constructor (this function's only OTHER caller besides
    /// `MultiPartOutputFile`) does not itself write any header, so it has no
    /// real value to supply. `MultiPartOutputFile` calls
    /// `from_part_with_preview[_mt]` instead (see below).
    pub(crate) fn from_part_mt(
        header: &Header,
        part_number: Option<i32>,
        chunk_offset_table_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::from_part_with_preview_mt(
            header,
            part_number,
            chunk_offset_table_position,
            0,
            num_threads,
        )
    }

    /// Records `preview_position`: C++
    /// `ImfMultiPartOutputFile.cpp:474-477` captures
    /// `parts[i]->previewPosition = headers[i].writeTo(..)` for every part
    /// while writing the header list, so `OutputFile::updatePreviewImage`
    /// works on a multipart part exactly as it does on a standalone file.
    /// [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile) calls
    /// this with the value `Header::write_attributes_to` returned for this
    /// part. `num_threads` is the per-file override (`None` =
    /// capture the current process-wide default).
    pub(crate) fn from_part_with_preview_mt(
        header: &Header,
        part_number: Option<i32>,
        chunk_offset_table_position: u64,
        preview_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let mut state = Self::compute_mt(header, part_number, num_threads)?;
        state.line_offsets_position = chunk_offset_table_position;
        state.preview_position = preview_position;
        Ok(state)
    }

    /// Set the pixel LAYOUT — port of `OutputFile::setFrameBuffer`.
    ///
    /// Design note: the
    /// writer no longer holds any reference to the caller's pixel bytes at
    /// all. `frame_buffer` supplies only the per-channel addressing (byte
    /// offsets/strides/types) as small, cloned [`OutSliceInfo`] metadata; the
    /// actual pixel BYTES are passed to [`write_pixels`](Self::write_pixels)
    /// **on every call**. This is what makes the core C++ streaming idiom —
    /// refill ONE strip buffer, `writePixels(n)`, repeat — actually work: a
    /// stored `&'a [u8]` borrow (an earlier, withdrawn design) froze the buffer
    /// for the writer's whole life, which is incompatible with reusing one
    /// small buffer across many calls (essential at 16k-65k resolutions) and
    /// broke any writer reused across a loop with a fresh per-iteration
    /// buffer (`tests/parity_exr.rs::ours_write_tiled`'s per-MIPMAP-level
    /// loop, in the tiled sibling of this design).
    ///
    /// Performs the same compatibility checks as C++ (pixel type and
    /// x/y-sampling must match the header channel) and builds the per-channel
    /// `OutSliceInfo` table, including the "channel present in the header but
    /// absent from the frame buffer → write zeroes" case.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] when a frame-buffer slice's type or sampling
    /// is incompatible with the corresponding header channel.
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
                if ch.x_sampling != slice.x_sampling || ch.y_sampling != slice.y_sampling {
                    return Err(ExrError::Arg(format!(
                        "X and/or y subsampling factors of \"{name}\" channel of \
                         output file are not compatible with the frame buffer's \
                         subsampling factors."
                    )));
                }
            }
        }

        // C++: build the slice table in header channel order.
        let mut slices: Vec<OutSliceInfo> = Vec::new();
        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                None => {
                    // Channel absent from the frame buffer → file gets zeroes.
                    slices.push(OutSliceInfo {
                        type_: ch.type_,
                        base_offset: 0,
                        x_stride: 0,
                        y_stride: 0,
                        x_sampling: ch.x_sampling,
                        y_sampling: ch.y_sampling,
                        zero: true,
                    });
                }
                Some(slice) => {
                    slices.push(OutSliceInfo {
                        type_: slice.type_,
                        base_offset: slice.base_offset,
                        x_stride: slice.x_stride,
                        y_stride: slice.y_stride,
                        x_sampling: slice.x_sampling,
                        y_sampling: slice.y_sampling,
                        zero: false,
                    });
                }
            }
        }

        self.slices = slices;
        Ok(())
    }

    /// Write `num_scan_lines` scanlines — port of `OutputFile::writePixels`.
    ///
    /// Each scanline is gathered from the frame buffer with
    /// [`copy_from_frame_buffer`] / [`fill_channel_with_zeroes`] into its slot
    /// in the line buffer (`offsetInLineBuffer[y - minY]`).  When the line
    /// buffer's **block** is complete — the last line of the block
    /// (`lineBufferMaxY`, clamped to `maxY`) has been gathered — the whole block
    /// is run through the codec (`None` for NONE) and emitted as a single chunk
    /// (`lineBufferMinY` + `dataSize` + data) by the inlined `writePixelData`.
    ///
    /// For NONE / RLE (`linesInBuffer == 1`) a block is one scanline, so a chunk
    /// is written per line.  For ZIP (`linesInBuffer == 16`) up to 16 scanlines
    /// are batched into one chunk; the offset table has one entry per block.
    /// A partial block survives across `write_pixels` calls (mirroring C++
    /// `LineBuffer::partiallyFull`), so the caller may push lines in any
    /// grouping.  Both `INCREASING_Y` and `DECREASING_Y` orders are honoured.
    ///
    /// `pixels` is the caller-owned backing store the frame-buffer slices
    /// (installed by [`set_frame_buffer`](Self::set_frame_buffer)) address for
    /// THIS call only — by design, the writer holds no
    /// reference to it beyond the call, so the same small buffer can be
    /// refilled and re-passed on the next call (the C++ streaming idiom).
    /// `stream` is the SAME split: the state that persists is `self`,
    /// while the stream is threaded through explicitly so a caller
    /// ([`OutputFile`] or [`OutputPartWriter`]) can hold it either owned or
    /// borrowed.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, `pixels` is too short
    ///   for the installed slice layout, or more scanlines are written than
    ///   the data window allows (matching the C++ throws — never a panic).
    /// * Propagates gather / stream / codec errors.
    fn write_pixels(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        num_scan_lines: i32,
    ) -> Result<()> {
        crate::probe::hit(crate::probe::Probe::WritePixels);

        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer specified …");
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data source.".into(),
            ));
        }

        // The buffer encoding for the NONE path is always Xdr (format == Xdr);
        // map the compressor Format onto the misc BufferFormat used by the
        // gather helpers.
        let buf_format = match self.format {
            Format::Xdr => BufferFormat::Xdr,
            Format::Native => BufferFormat::Native,
        };

        let step: i32 = if self.line_order == crate::attr::LineOrder::IncreasingY {
            1
        } else {
            -1
        };

        // C++ `ImfOutputFile.cpp:1085-1107` checks
        // `missingScanLines <= 0` ONCE, at the top, before processing anything
        // -- not once per scanline inside the loop. Two consequences that
        // differ from a naive per-scanline check: (1) calling `write_pixels`
        // for MORE lines than the data window has left (e.g. `write_pixels(25)`
        // with only 20 remaining) writes the 20 that exist and returns `Ok`,
        // it does not error partway through; (2) calling `write_pixels(0)`
        // AFTER the whole window is already written still errors (the check
        // fires unconditionally at entry, even when 0 lines are requested).
        if self.missing_scan_lines <= 0 {
            return Err(ExrError::Arg(
                "Tried to write more scan lines than specified by the data window.".into(),
            ));
        }
        let mut remaining = num_scan_lines.min(self.missing_scan_lines);

        // HEAD: finish any block partially gathered by a PREVIOUS
        // `write_pixels` call, one line at a time, exactly as the fully
        // serial path always has — a partial block can only ever be the
        // FIRST thing a call touches (the "Head" step).
        while remaining > 0 && self.block_lines_gathered > 0 {
            self.write_one_line_serial(stream, pixels, buf_format, step)?;
            remaining -= 1;
        }
        if remaining == 0 {
            return Ok(());
        }

        // Parallel gate: parallel only when the captured thread count is
        // `> 1`. (C++ additionally requires the call to span >1 chunk; the
        // "Body" enumeration below naturally yields 0 or 1 full blocks in
        // that case, which the parallel loop handles as a trivial 1-item
        // batch — correct either way, so no extra gate is needed here.)
        if !self.num_threads.is_parallel() {
            for _ in 0..remaining {
                self.write_one_line_serial(stream, pixels, buf_format, step)?;
            }
            return Ok(());
        }

        // BODY: enumerate every block that starts fresh (we are always at a
        // block boundary here) AND completes entirely within `remaining` —
        // these can be gathered+encoded in any order and emitted back in
        // their original sequence with no observable difference from the
        // serial path. A block that would only be PARTIALLY covered by this
        // call is left for the "Tail" below.
        let mut full_blocks: Vec<i32> = Vec::new();
        let mut left = remaining;
        let mut y = self.current_scan_line;
        while left > 0 {
            let lb_min_y = line_buffer_min_y(y, self.min_y, self.lines_in_buffer);
            let lb_max_y = line_buffer_max_y(y, self.min_y, self.lines_in_buffer).min(self.max_y);
            let block_lines = lb_max_y - lb_min_y + 1;
            if block_lines > left {
                break;
            }
            full_blocks.push(lb_min_y);
            left -= block_lines;
            y += step * block_lines;
        }

        // Ring size in BYTES (see `ThreadCount::window_size_for_bytes`): C++'s
        // `2 * numThreads` line buffers (`ImfOutputFile.cpp:181-193`), grown
        // for tiny chunks so a ring slot is worth a pool dispatch.
        let bytes_per_chunk = self.max_bytes_per_line.saturating_mul(self.lines_in_buffer.max(0) as usize);
        let window = self.num_threads.window_size_for_bytes(bytes_per_chunk);
        let compression = self.compression;
        let max_bytes_per_line = self.max_bytes_per_line;
        let lines_in_buffer = self.lines_in_buffer;
        let header = &self.header;
        let slices = &self.slices;
        let offset_in_line_buffer = &self.offset_in_line_buffer;
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        let count = self.num_threads;

        // Alloc audit C4: one pool shared by the encode workers (which take a
        // buffer to encode into) and the emit step (which gives each buffer
        // back once its bytes are written). At most `window` items are
        // outstanding at once, so `window` buffers cover the whole ring.
        let pool = crate::buf::BufPool::new();
        pool.prewarm_for_pipeline(crate::buf::WRITE_WINDOWS_IN_FLIGHT, window, bytes_per_chunk, full_blocks.len());
        let pool = &pool;
        // `CodecPool`, not `map_init`: see `Compressor`'s trait doc and
        // `crate::compressor::CodecPool`'s own doc.
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;
        let full_blocks_ref = &full_blocks;

        // Gather+encode block `k` on the pool; emit finished blocks here, in
        // `full_blocks` order (== file/call order, the hard byte-identity
        // invariant), while later blocks are still encoding.
        crate::threading::ordered_pipeline(
            count,
            full_blocks.len(),
            crate::threading::Ring::items(window),
            |_| 0,
            |k| -> Result<Vec<u8>> {
                let lb_min_y = full_blocks_ref[k];
                let mut worker_compressor =
                    codec_pool.lease(|| new_compressor(compression, max_bytes_per_line, header))?;
                // Gather straight into a pooled buffer; `finish_chunk`
                // hands that same buffer (or the codec's swapped-out
                // output) back as the chunk, so a block costs no fresh
                // zero-filled allocation and no extra copy.
                let mut buf = pool.take(max_bytes_per_line.saturating_mul(lines_in_buffer as usize));
                let lb_max_y = (lb_min_y + lines_in_buffer - 1).min(max_y);
                let data_size = gather_block(
                    slices,
                    min_x,
                    max_x,
                    lb_min_y,
                    lb_max_y,
                    min_y,
                    offset_in_line_buffer,
                    pixels,
                    buf_format,
                    &mut buf,
                )?;
                crate::compressor::finish_chunk(worker_compressor.as_mut(), buf, data_size, |c, raw| {
                    Ok(c.compress(raw, lb_min_y).map_err(crate::error::encode_err)?.len())
                })
            },
            |k, chunk| -> Result<()> {
                let lb_min_y = full_blocks_ref[k];
                log::trace!(
                    "scanline write chunk: y={lb_min_y} dataSize={} (parallel)",
                    chunk.len()
                );
                Self::write_raw_chunk_now(
                    stream,
                    &mut self.line_offsets,
                    self.line_offsets_position,
                    self.part_number,
                    self.min_y,
                    self.lines_in_buffer,
                    lb_min_y,
                    &chunk,
                )?;
                // Alloc audit C4: back to the pool now that its bytes are
                // written (an empty block just adds a zero-capacity entry).
                pool.give(chunk);

                let lb_max_y = (lb_min_y + self.lines_in_buffer - 1).min(self.max_y);
                let block_lines = lb_max_y - lb_min_y + 1;
                self.current_scan_line += step * block_lines;
                self.missing_scan_lines -= block_lines;
                Ok(())
            },
        )?;
        if !full_blocks.is_empty() {
            crate::probe::hit(crate::probe::Probe::ScanWriteParallel);
        }

        // TAIL: a final block this call only partially covers (or `left ==
        // 0` and there is nothing left) — gathered one line at a time, same
        // as the Head, carrying over to the NEXT `write_pixels` call.
        for _ in 0..left {
            self.write_one_line_serial(stream, pixels, buf_format, step)?;
        }

        Ok(())
    }

    /// Gather and emit exactly ONE scanline, flushing its block if that
    /// scanline completes it — the serial per-line body extracted verbatim
    /// from the original serial `write_pixels` loop (this IS the "safe scalar
    /// source of truth": the parallel body in `write_pixels` above must
    /// produce byte-identical chunks to what a sequence of these calls
    /// would). Used for: the fully-serial path (`num_threads <= 1`), the
    /// "Head" (finishing a block carried over from a previous call) and the
    /// "Tail" (a block this call cannot fully complete).
    fn write_one_line_serial(
        &mut self,
        stream: &mut dyn OStream,
        pixels: &[u8],
        buf_format: BufferFormat,
        step: i32,
    ) -> Result<()> {
        let y = self.current_scan_line;

        // The block this scanline belongs to and its inclusive y-range
        // within the data window (C++ `lineBufferMinY`/`lineBufferMaxY`,
        // the latter clamped to `maxY` for the final, possibly partial,
        // block).
        let lb_min_y = line_buffer_min_y(y, self.min_y, self.lines_in_buffer);
        let lb_max_y = line_buffer_max_y(y, self.min_y, self.lines_in_buffer).min(self.max_y);
        let block_lines = lb_max_y - lb_min_y + 1;

        // --- Gather this scanline into the line buffer (C++ LineBufferTask) ---
        // `offset_in_line_buffer[y - minY]` is this line's byte offset *within
        // its block* (the table resets to 0 at every block boundary).
        let line_index = (y - self.min_y) as usize;
        let base = self
            .offset_in_line_buffer
            .get(line_index)
            .copied()
            .ok_or_else(|| {
                ExrError::Logic(format!(
                    "offsetInLineBuffer index {line_index} out of range"
                ))
            })?;

        let write_ptr =
            gather_line(&self.slices, self.min_x, self.max_x, y, pixels, buf_format, &mut self.line_buffer, base)?;

        // C++: if (endOfLineBufferData < writePtr) endOfLineBufferData = writePtr;
        // For a block the total size is the high-water mark of `write_ptr`
        // over all its lines (lines are laid contiguously, so the last line
        // gathered sets the end — but `max` is robust to any gather order).
        if write_ptr > self.block_data_size {
            self.block_data_size = write_ptr;
        }
        self.block_lines_gathered += 1;

        // C++: currentScanLine += step; missingScanLines -= 1.  Advance
        // before the flush check so the next line's block is computed fresh.
        self.current_scan_line += step;
        self.missing_scan_lines -= 1;

        // Block complete? (all lines in [lb_min_y, lb_max_y] gathered.)
        // C++ flushes when the next scanline steps past the block bounds.
        if self.block_lines_gathered < block_lines {
            return Ok(());
        }

        // --- writePixelData: compress the whole block, record offset, write ---
        let data_size = self.block_data_size;

        // Compress the gathered block if a codec is present, else use the
        // raw bytes (NONE path).  C++ runs the block through
        // `LineBuffer::compressor->compress(buffer, dataSize, minY, …)` and
        // then decides at the WRITER whether to keep the encoding:
        // `if (compSize < dataSize)` (ImfOutputFile.cpp:573) — at equality
        // the raw block is stored, because readers detect a raw passthrough
        // by `dataSize == uncompressedSize` alone.  `stored_chunk` is that
        // single decision point (see its docs for the NATIVE/convertToXdr
        // branch, which is unreachable while every codec reports Xdr).
        //
        // Disjoint-field borrow: `self.compressor` (mut) and
        // `self.line_buffer` (shared) are different fields, so this borrows
        // cleanly.
        debug_assert_eq!(self.format, Format::Xdr);
        let raw: &[u8] = &self.line_buffer[..data_size];
        // C++ `Compressor::compress` short-circuits
        // an empty input BEFORE ever reaching a codec's own `compress()`
        // (`ImfCompressor.cpp:141-147`: `if (inSize == 0) { outPtr =
        // inPtr; return 0; }`) -- an empty block (every channel's
        // `ySampling` skips every line in this block, or an odd line of a
        // 1-line block with `ySampling == 2`) is legal and must stay a
        // zero-byte chunk regardless of codec. Calling a real codec (PIZ,
        // DWA, HTJ2K) on zero bytes is out of its own contract and would
        // either fail or fabricate framing bytes C++ never writes.
        let chunk: &[u8] = if raw.is_empty() {
            raw
        } else {
            match self.compressor.as_mut() {
                Some(c) => stored_chunk(
                    c.compress(raw, lb_min_y).map_err(crate::error::encode_err)?,
                    raw,
                ),
                None => raw,
            }
        };
        // Hot/loop detail (trace, compiled out unless enabled): per-chunk
        // encode — the block's start scanline, raw vs stored byte size.
        log::trace!(
            "scanline write chunk: y={lb_min_y} dataSize={data_size} stored={}",
            chunk.len()
        );

        // C++ writePixelData: leader (+ partNumber prefix for a multipart
        // file — the ONLY difference from the single-part layout) +
        // lineBufferMinY + dataSize + data, then back-patch the reserved
        // offset-table entry — the single low-level primitive
        // `copy_pixels` also uses.
        // Disjoint-field borrow: `chunk` may still borrow `self.line_buffer`
        // (the raw/no-shrink path above) while `&mut self.line_offsets` is
        // passed alongside it — different fields, borrows cleanly (same
        // reasoning `write_raw_chunk_now`'s own doc comment gives).
        Self::write_raw_chunk_now(
            stream,
            &mut self.line_offsets,
            self.line_offsets_position,
            self.part_number,
            self.min_y,
            self.lines_in_buffer,
            lb_min_y,
            chunk,
        )?;

        // Reset for the next block (C++ `partiallyFull = false`).
        self.block_data_size = 0;
        self.block_lines_gathered = 0;
        Ok(())
    }

    /// Quick pixel copy: transcode every remaining scanline chunk of `input`
    /// straight into this file, RAW (no decompress/recompress) — port
    /// of `OutputFile::copyPixels(InputFile&)` / `copyPixels(InputPart&)`
    /// (`ImfOutputFile.cpp:1233-1360`), mirroring
    /// [`crate::tiled::output::TiledOutputFileState::copy_pixels`]'s design.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if the input is tiled (C++: "Try using
    ///   `TiledOutputFile::copyPixels` instead"), or the headers' data window,
    ///   line order, compression, or channel list differ.
    /// * [`ExrError::Logic`] if this file already has pixel data written
    ///   (C++: "already contains pixel data").
    /// * Propagates `input`'s `raw_pixel_data` errors.
    fn copy_pixels<T: ScanLineCopySource>(
        &mut self,
        stream: &mut dyn OStream,
        input: &mut T,
    ) -> Result<()> {
        // --- Header compatibility (ImfOutputFile.cpp:1240-1307) -------------
        let hdr = &self.header;
        let in_hdr = input.header();

        if in_hdr.has_tile_description() {
            return Err(ExrError::Arg(
                "Cannot copy pixels: the input file is tiled, but the output \
                 file is not. Try using TiledOutputFile::copyPixels instead."
                    .into(),
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
        let data_window = hdr.data_window()?;
        if self.missing_scan_lines != data_window.max.y - data_window.min.y + 1 {
            return Err(ExrError::Logic(
                "Quick pixel copy failed: this file already contains pixel data.".into(),
            ));
        }

        // --- Copy every remaining chunk, raw, in this writer's own line-order
        // walk (C++ drives `currentScanLine`/`missingScanLines` exactly the
        // same way `write_pixels` does; a `copyPixels` source's on-disk order
        // already matches, since both files share the same line order,
        // checked above) --------------------------------------------------
        while self.missing_scan_lines > 0 {
            let (_, chunk) = input.raw_pixel_data(self.current_scan_line)?;
            let lb_min_y = line_buffer_min_y(self.current_scan_line, self.min_y, self.lines_in_buffer);

            Self::write_raw_chunk_now(
                stream,
                &mut self.line_offsets,
                self.line_offsets_position,
                self.part_number,
                self.min_y,
                self.lines_in_buffer,
                lb_min_y,
                &chunk,
            )?;

            self.current_scan_line += if self.line_order == crate::attr::LineOrder::IncreasingY {
                self.lines_in_buffer
            } else {
                -self.lines_in_buffer
            };
            self.missing_scan_lines -= self.lines_in_buffer;
        }

        Ok(())
    }

    /// Write one already-encoded (possibly raw) chunk's leader + payload and
    /// back-patch its offset-table slot — the single low-level primitive
    /// [`write_pixels`](Self::write_pixels)'s own tail and
    /// [`copy_pixels`](Self::copy_pixels) both call, mirroring
    /// [`crate::tiled::output::TiledOutputFileState::write_tile_chunk_now`].
    /// Takes the exact fields it touches (not `&mut self`) so a caller can
    /// hold `chunk` borrowed from another field of `self` across the call —
    /// the same disjoint-field-borrow reasoning that primitive documents.
    #[allow(clippy::too_many_arguments)] // takes the exact disjoint fields it touches, see doc comment
    fn write_raw_chunk_now(
        stream: &mut dyn OStream,
        line_offsets: &mut [u64],
        line_offsets_position: u64,
        part_number: Option<i32>,
        min_y: i32,
        lines_in_buffer: i32,
        lb_min_y: i32,
        chunk: &[u8],
    ) -> Result<()> {
        let offset_index = ((lb_min_y - min_y) / lines_in_buffer) as usize;

        let chunk_position = stream.tellp()?;
        *line_offsets.get_mut(offset_index).ok_or_else(|| {
            ExrError::Logic(format!("lineOffsets index {offset_index} out of range"))
        })? = chunk_position;

        if let Some(pn) = part_number {
            xdr::write_i32(stream, pn)?;
        }
        xdr::write_i32(stream, lb_min_y)?;
        xdr::write_i32(stream, chunk.len() as i32)?;
        stream.write(chunk)?;

        let saved = stream.tellp()?;
        stream.seekp(
            line_offsets_position + offset_index as u64 * (crate::container::SIZE_I64 as u64),
        )?;
        xdr::write_u64(stream, chunk_position)?;
        stream.seekp(saved)?;
        Ok(())
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk — port of `void updatePreviewImage(const PreviewRgba
    /// newPixels[])` (`ImfOutputFile.cpp:1363-1411`).
    ///
    /// `new_pixels` must have exactly as many entries as the header's
    /// existing preview image (`width * height`) — C++ trusts the caller's
    /// raw pointer to cover that many pixels (a length mismatch there is
    /// undefined behaviour); this port returns an error instead of reading
    /// past `new_pixels` or leaving part of the preview stale.
    ///
    /// # Errors
    /// * [`ExrError::Logic`] if the file has no preview image (`LogicExc` in
    ///   C++, tested as `previewPosition <= 0`; here `preview_position == 0`
    ///   since it is a file offset and can never legitimately be 0 -- the
    ///   magic number always occupies the first 4 bytes).
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

    /// Overwrite `length` bytes of an already-written scanline's chunk,
    /// starting `offset` bytes into it, with the byte `c` — port of `void
    /// breakScanLine(int y, int offset, int length, char c)`
    /// (`ImfOutputFile.cpp:1417-1441`; test-only helper for exercising
    /// corrupt-chunk error paths).
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if `y` addresses a block whose chunk has not
    /// been written yet (offset-table slot still 0) -- including a `y`
    /// outside the data window, which C++ leaves as an out-of-bounds
    /// `std::vector` index (undefined behaviour); this port errors instead.
    fn break_scan_line(
        &mut self,
        stream: &mut dyn OStream,
        y: i32,
        offset: u64,
        length: usize,
        c: u8,
    ) -> Result<()> {
        let not_stored = || {
            ExrError::Arg(format!(
                "Cannot overwrite scan line {y}. The scan line has not yet \
                 been stored in file \"{}\".",
                stream.file_name()
            ))
        };

        // C++: lineOffsets[(y - minY) / linesInBuffer] -- an unchecked vector
        // index. `y` outside the data window (or a `linesInBuffer` of 0,
        // which cannot occur -- `num_lines_in_buffer` is always >= 1) is
        // rejected here instead of indexing out of bounds.
        let index: usize = ((y - self.min_y) / self.lines_in_buffer)
            .try_into()
            .map_err(|_| not_stored())?;
        let position = *self.line_offsets.get(index).ok_or_else(not_stored)?;
        if position == 0 {
            return Err(not_stored());
        }

        stream.seekp(position + offset)?;
        let fill = vec![c; length];
        stream.write(&fill)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure per-block pipeline functions
// ---------------------------------------------------------------------------
//
// `gather_line`/`gather_block`/`encode_block` are the writer-side half of the
// three-function split: `write_one_line_serial` (the
// serial path AND the parallel path's "Head"/"Tail") and the parallel body
// in `write_pixels` both compose `gather_line`/`encode_block` in the same
// order, so a block's chunk bytes depend only on `(raw gathered bytes,
// lb_min_y, codec params)` — never on which thread gathered/encoded it or
// how many other blocks ran alongside it.

/// Gather ONE scanline's channels into `dst` (a per-block buffer — either
/// the persisted `self.line_buffer` for the serial path, or a fresh
/// worker-owned `Vec` for the parallel path) at `write_ptr`, returning the
/// new high-water mark. Pure function: no stream access, no `self`.
#[allow(clippy::too_many_arguments)]
fn gather_line(
    slices: &[OutSliceInfo],
    min_x: i32,
    max_x: i32,
    y: i32,
    pixels: &[u8],
    buf_format: BufferFormat,
    dst: &mut [u8],
    write_ptr: usize,
) -> Result<usize> {
    let mut write_ptr = write_ptr;
    for slice in slices {
        // C++: if (modp(y, ySampling) != 0) continue;
        if modp(y, slice.y_sampling) != 0 {
            continue;
        }

        let d_min_x = divp(min_x, slice.x_sampling);
        let d_max_x = divp(max_x, slice.x_sampling);
        let num_pixels = (d_max_x - d_min_x + 1) as usize;

        if slice.zero {
            write_ptr = fill_channel_with_zeroes(dst, write_ptr, slice.type_, num_pixels)?;
        } else {
            let line_ptr: i64 =
                slice.base_offset + divp(y, slice.y_sampling) as i64 * slice.y_stride as i64;
            let read_ptr_i: i64 = line_ptr + d_min_x as i64 * slice.x_stride as i64;
            if read_ptr_i < 0 {
                return Err(ExrError::Arg(
                    "Frame buffer slice address is out of range (negative).".into(),
                ));
            }
            let read_ptr = read_ptr_i as usize;

            let (new_wo, _new_rp) = copy_from_frame_buffer(
                dst,
                write_ptr,
                pixels,
                read_ptr,
                num_pixels,
                slice.x_stride,
                buf_format,
                slice.type_,
            )?;
            write_ptr = new_wo;
        }
    }
    Ok(write_ptr)
}

/// Gather every scanline of ONE full block `[lb_min_y, lb_max_y]` into `dst`
/// (a fresh, block-sized buffer whose byte 0 corresponds to the block's own
/// start — `offset_in_line_buffer` already resets to 0 at each block
/// boundary, so it addresses a standalone per-block buffer exactly as it
/// addresses the persisted `self.line_buffer`). Returns the block's total
/// gathered size (the high-water mark over all its lines).
#[allow(clippy::too_many_arguments)]
fn gather_block(
    slices: &[OutSliceInfo],
    min_x: i32,
    max_x: i32,
    lb_min_y: i32,
    lb_max_y: i32,
    file_min_y: i32,
    offset_in_line_buffer: &[usize],
    pixels: &[u8],
    buf_format: BufferFormat,
    dst: &mut [u8],
) -> Result<usize> {
    let mut data_size = 0usize;
    for y in lb_min_y..=lb_max_y {
        let line_index = (y - file_min_y) as usize;
        let base = offset_in_line_buffer.get(line_index).copied().ok_or_else(|| {
            ExrError::Logic(format!("offsetInLineBuffer index {line_index} out of range"))
        })?;
        let write_ptr = gather_line(slices, min_x, max_x, y, pixels, buf_format, dst, base)?;
        data_size = data_size.max(write_ptr);
    }
    Ok(data_size)
}

/// A **borrowing** view over one multipart part's persisted writer state — the
/// "ownership inversion" design: instead of
/// [`MultiPartOutputFile::output_part`](crate::multipart::MultiPartOutputFile::output_part)
/// building a brand-new [`OutputFile`] (and its state) on every call, the
/// container persists one [`OutputFileState`] per scanline part across calls
/// and hands out this thin view — `{ &mut dyn OStream, &mut OutputFileState }`
/// — so interleaved writes across parts resume exactly where they left off
/// (matching C++'s per-part `_outputFiles` cache,
/// `ImfMultiPartOutputFile.cpp:313-321`).
///
/// Exposes the same public surface as [`OutputFile`] (`header`,
/// `current_scan_line`, `set_frame_buffer`, `write_pixels`) so existing
/// callers of `output_part` are unaffected by the return-type change.
pub struct OutputPartWriter<'p> {
    stream: &'p mut dyn OStream,
    state: &'p mut OutputFileState,
}

impl<'p> OutputPartWriter<'p> {
    /// Build a view over an existing (already-constructed, possibly
    /// previously-written-to) part state and the container's stream.
    pub(crate) fn new(stream: &'p mut dyn OStream, state: &'p mut OutputFileState) -> Self {
        Self { stream, state }
    }

    /// The image header — port of `const Header& OutputFile::header() const`.
    pub fn header(&self) -> &Header {
        &self.state.header
    }

    /// Next scanline to be written — port of `int OutputFile::currentScanLine()`.
    pub fn current_scan_line(&self) -> i32 {
        self.state.current_scan_line
    }

    /// Set the pixel LAYOUT — port of `OutputFile::setFrameBuffer`. See
    /// [`OutputFileState::set_frame_buffer`] for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write `num_scan_lines` scanlines — port of `OutputFile::writePixels`.
    /// See [`OutputFileState::write_pixels`] for the full algorithm doc.
    pub fn write_pixels(&mut self, pixels: &[u8], num_scan_lines: i32) -> Result<()> {
        self.state.write_pixels(self.stream, pixels, num_scan_lines)
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk. See [`OutputFileState::update_preview_image`] for the full
    /// doc.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Overwrite `length` bytes of an already-written scanline's chunk,
    /// starting `offset` bytes into it, with the byte `c`. See
    /// [`OutputFileState::break_scan_line`] for the full doc.
    pub fn break_scan_line(&mut self, y: i32, offset: u64, length: usize, c: u8) -> Result<()> {
        self.state.break_scan_line(self.stream, y, offset, length, c)
    }

    /// Quick pixel copy from `input`, raw (no decompress/recompress). See
    /// [`OutputFileState::copy_pixels`] for the full doc.
    pub fn copy_pixels<T: ScanLineCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::frame_buffer::Slice;
    use crate::io::{MemIStream, MemOStream};
    use crate::scanline::InputFile;
    use imath_rs::{Half, V2f};

    /// Build a 4x2 header with channels B, G, R (HALF), the given line order,
    /// and NO_COMPRESSION.
    fn make_header(line_order: LineOrder) -> Header {
        let mut h = Header::with_dimensions(
            4,
            2,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            line_order,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Build a pixel buffer of three HALF planes (B, G, R) for a 4x2 image,
    /// row-major within each plane, plus the matching frame buffer slices.
    ///
    /// Returns `(buffer, frame_buffer)`.  Plane layout (bytes):
    /// `[B(8 halves=16B)][G(16B)][R(16B)]` where each plane is 4 px * 2 rows.
    fn make_pixels() -> (Vec<u8>, FrameBuffer) {
        // Distinct values per channel/pixel so we can verify ordering exactly.
        // value(channel, x, y) = base_channel + y*10 + x
        let plane_pixels = 4 * 2; // 8 halves per plane
        let plane_bytes = plane_pixels * 2;
        let mut buf = vec![0u8; plane_bytes * 3];

        // Helper writing a half (native bytes) at a plane/x/y position.
        let put = |buf: &mut [u8], plane: usize, x: usize, y: usize, v: f32| {
            let off = plane * plane_bytes + (y * 4 + x) * 2;
            let bits = Half::from(v).bits().to_ne_bytes();
            buf[off..off + 2].copy_from_slice(&bits);
        };

        for y in 0..2usize {
            for x in 0..4usize {
                put(&mut buf, 0, x, y, 100.0 + (y * 10 + x) as f32); // B plane
                put(&mut buf, 1, x, y, 200.0 + (y * 10 + x) as f32); // G plane
                put(&mut buf, 2, x, y, 300.0 + (y * 10 + x) as f32); // R plane
            }
        }

        // Frame buffer slices.  base_offset is the byte offset of each plane's
        // origin (origin is (0,0) so no further adjustment).  x_stride=2,
        // y_stride=8 (4 px * 2 bytes).
        let mut fb = FrameBuffer::new();
        let mk = |plane: usize| Slice {
            type_: PixelType::Half,
            base_offset: (plane * plane_bytes) as i64,
            x_stride: 2,
            y_stride: 8,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        fb.insert("B", mk(0)).unwrap();
        fb.insert("G", mk(1)).unwrap();
        fb.insert("R", mk(2)).unwrap();
        (buf, fb)
    }

    /// Expected little-endian (XDR) half bytes for one channel's scanline `y`,
    /// reading from the named plane.  EXR's machine-independent wire format is
    /// little-endian (matches `container::xdr` and C++ OpenEXR).
    fn expected_channel_xdr(plane_base: f32, y: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for x in 0..4usize {
            let v = plane_base + (y * 10 + x) as f32;
            let bits = Half::from(v).bits();
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    /// Round-4 design requirement: `write_pixels` takes the pixel bytes PER
    /// CALL, so ONE small strip buffer can be refilled and re-passed across
    /// many calls — the core streaming pattern this design exists for
    /// (essential at 16k-65k resolutions, where a whole-image buffer is not
    /// an option). Writes a 4x5 FLOAT image one row at a time through a
    /// single 4-float (16-byte) buffer, refilling it with different content
    /// before every `write_pixels(1)` call, and asserts every row's value
    /// made it into the file (not just the last-written content, which is
    /// exactly what the withdrawn stored-borrow design would have produced).
    #[test]
    fn strip_streaming_one_small_buffer_reused_across_many_write_pixels_calls() {
        const SW: i32 = 4;
        const SH: i32 = 5;
        let header = Header::with_dimensions(
            SW,
            SH,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut header = header;
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        header.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let mut fb = FrameBuffer::new();
        fb.insert(
            "Z",
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 4,
                y_stride: 0, // one-row buffer: every row reuses row 0
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            let mut strip = vec![0u8; (SW as usize) * 4];
            for y in 0..SH {
                for x in 0..SW {
                    let v = (y * 100 + x) as f32;
                    strip[(x as usize) * 4..(x as usize) * 4 + 4]
                        .copy_from_slice(&v.to_ne_bytes());
                }
                of.write_pixels(&strip, 1).unwrap();
            }
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        let mut rfb = FrameBuffer::new();
        let mut readback = vec![0u8; (SW as usize) * (SH as usize) * 4];
        rfb.insert(
            "Z",
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 4,
                y_stride: (SW as usize) * 4,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        inf.set_frame_buffer(&rfb).unwrap();
        inf.read_pixels(0, SH - 1, &mut readback).unwrap();

        for y in 0..SH {
            for x in 0..SW {
                let off = (y as usize * SW as usize + x as usize) * 4;
                let got = f32::from_ne_bytes(readback[off..off + 4].try_into().unwrap());
                let expected = (y * 100 + x) as f32;
                assert_eq!(got, expected, "row {y} col {x} must be its OWN strip content");
            }
        }
    }

    /// Build a `w`x`h` header (R,G,B half + A float, matching
    /// `scanline::input::tests::make_header[_comp]`) with the given line
    /// order and compression, for parallel write-determinism tests.
    fn make_header_wh(w: i32, h: i32, line_order: LineOrder, comp: Compression) -> Header {
        let mut hdr = Header::with_dimensions(
            w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, line_order, comp,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr
    }

    /// Interleaved `w`x`h` pixel buffer (`[R half][G half][B half][A float]`
    /// per pixel), one distinct value per channel/pixel. Mirrors
    /// `scanline::input::tests::make_pixels`.
    fn make_pixels_wh(w: usize, h: usize) -> (Vec<u8>, FrameBuffer) {
        let pixel_bytes = 2 + 2 + 2 + 4;
        let row_bytes = pixel_bytes * w;
        let mut buf = vec![0u8; row_bytes * h];
        for y in 0..h {
            for x in 0..w {
                let p = y * row_bytes + x * pixel_bytes;
                let r = Half::from(300.0 + (y * 10 + x) as f32).bits();
                let g = Half::from(200.0 + (y * 10 + x) as f32).bits();
                let b = Half::from(100.0 + (y * 10 + x) as f32).bits();
                let a = 400.0f32 + (y * 10 + x) as f32;
                buf[p..p + 2].copy_from_slice(&r.to_ne_bytes());
                buf[p + 2..p + 4].copy_from_slice(&g.to_ne_bytes());
                buf[p + 4..p + 6].copy_from_slice(&b.to_ne_bytes());
                buf[p + 6..p + 10].copy_from_slice(&a.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        let mk = |ty: PixelType, off: usize| Slice {
            type_: ty, base_offset: off as i64, x_stride: pixel_bytes, y_stride: row_bytes,
            x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
        };
        fb.insert("R", mk(PixelType::Half, 0)).unwrap();
        fb.insert("G", mk(PixelType::Half, 2)).unwrap();
        fb.insert("B", mk(PixelType::Half, 4)).unwrap();
        fb.insert("A", mk(PixelType::Float, 6)).unwrap();
        (buf, fb)
    }

    /// Write the SAME image with `num_threads` in
    /// {0, 1, 2, 3, 8, 17}, in one `write_pixels` call covering the whole
    /// image, and assert the produced FILE BYTES are identical at every
    /// thread count — the hard invariant of the parallel writer.
    /// NO_COMPRESSION gives one chunk per line (64 chunks for 64 rows),
    /// comfortably exercising multiple parallel windows.
    #[test]
    fn parallel_write_whole_image_is_byte_identical_across_thread_counts() {
        let (w, h) = (17i32, 64i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::NoCompression);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::ScanWriteParallel);
                let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                of.write_pixels(&pixels, h).unwrap();
            }
            let bytes = out.into_data();

            // Sanity: the file round-trips to the original pixels.
            let mut is = MemIStream::new(bytes.clone());
            let mut inf = InputFile::new(&mut is).unwrap();
            let (mut readback, read_fb) = make_pixels_wh(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::ScanWriteParallel, n);
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

    /// Same invariant with ZIP (16 lines/block — exercises the "several full
    /// blocks in one call" parallel body path with a real codec, incl. the
    /// per-worker fresh-compressor no-cross-call-state requirement).
    #[test]
    fn parallel_write_zip_multiline_blocks_byte_identical_across_thread_counts() {
        let (w, h) = (23i32, 96i32); // 96/16 = 6 ZIP blocks
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::ScanWriteParallel);
                let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                of.write_pixels(&pixels, h).unwrap();
            }
            let bytes = out.into_data();

            let mut is = MemIStream::new(bytes.clone());
            let mut inf = InputFile::new(&mut is).unwrap();
            let (mut readback, read_fb) = make_pixels_wh(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::ScanWriteParallel, n);
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

    /// Same invariant, but written in irregular strips per call (1 line, 7
    /// lines, a whole block, the rest) rather than the whole image in one
    /// call — exercises the Head/Body/Tail split's partial-block carryover
    /// across MULTIPLE `write_pixels` calls, at every thread count.
    #[test]
    fn parallel_write_irregular_strips_byte_identical_across_thread_counts() {
        let (w, h) = (11i32, 40i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);
        let strips = [1i32, 7, 16, 4, 12]; // sums to 40

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 3, 9] {
            let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::ScanWriteParallel);
                let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                // The frame buffer addresses scanlines by their ABSOLUTE `y`
                // (`base_offset + y*y_stride`), so every call passes the
                // FULL pixel buffer (not a per-call slice) — this test's
                // point is the Head/Body/Tail split across several calls
                // with irregular line counts, not the separate
                // small-strip-buffer pattern already covered by
                // `strip_streaming_one_small_buffer_reused_across_many_write_pixels_calls`.
                for &s in &strips {
                    of.write_pixels(&pixels, s).unwrap();
                }
            }
            let bytes = out.into_data();

            let mut is = MemIStream::new(bytes.clone());
            let mut inf = InputFile::new(&mut is).unwrap();
            let (mut readback, read_fb) = make_pixels_wh(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            crate::probe::assert_engaged(crate::probe::Probe::ScanWriteParallel, n);
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
    fn write_uncompressed_increasing_y_byte_exact() {
        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();

        // --- Independently compute the expected header+offset-table prefix by
        //     writing just the header to a second stream. ---
        let mut hdr_stream = MemOStream::new();
        header.write_to(&mut hdr_stream, 0).unwrap();
        let header_len = hdr_stream.data().len() as u64;

        // Offset table: 2 entries (one per scanline), 8 bytes each.
        let offset_table_start = header_len;
        let chunks_start = offset_table_start + 2 * 8;

        // --- Run the writer. ---
        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            assert_eq!(of.current_scan_line(), 0);
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
            // After writing both lines, current advanced past max.y.
            assert_eq!(of.current_scan_line(), 2);
        }
        let bytes = out.into_data();

        // --- 1. Leading header bytes match the standalone header write. ---
        assert_eq!(&bytes[..header_len as usize], hdr_stream.data());

        // --- 2. Build the expected chunk stream and offset table. ---
        // Chunk layout per line (INCREASING_Y → lines 0 then 1):
        //   y (i32 LE) + dataSize (i32 LE = 24) + [B 8B][G 8B][R 8B]
        let mut expected_chunks: Vec<u8> = Vec::new();
        let mut expected_offsets: Vec<u64> = Vec::new();
        for y in 0..2usize {
            expected_offsets.push(chunks_start + expected_chunks.len() as u64);
            expected_chunks.extend_from_slice(&(y as i32).to_le_bytes());
            expected_chunks.extend_from_slice(&24i32.to_le_bytes());
            expected_chunks.extend_from_slice(&expected_channel_xdr(100.0, y)); // B
            expected_chunks.extend_from_slice(&expected_channel_xdr(200.0, y)); // G
            expected_chunks.extend_from_slice(&expected_channel_xdr(300.0, y)); // R
        }

        // --- 3. Offset table bytes (u64 LE) match the patched table. ---
        let mut expected_table: Vec<u8> = Vec::new();
        for &o in &expected_offsets {
            expected_table.extend_from_slice(&o.to_le_bytes());
        }
        assert_eq!(
            &bytes[offset_table_start as usize..chunks_start as usize],
            expected_table.as_slice(),
            "offset table mismatch"
        );

        // --- 4. Chunk bytes match exactly. ---
        assert_eq!(
            &bytes[chunks_start as usize..],
            expected_chunks.as_slice(),
            "chunk data mismatch"
        );

        // --- 5. Total length is consistent. ---
        assert_eq!(bytes.len() as u64, chunks_start + expected_chunks.len() as u64);
    }

    #[test]
    fn write_uncompressed_decreasing_y_order_and_offsets() {
        let header = make_header(LineOrder::DecreasingY);
        let (pixels, fb) = make_pixels();

        let mut hdr_stream = MemOStream::new();
        header.write_to(&mut hdr_stream, 0).unwrap();
        let header_len = hdr_stream.data().len() as u64;
        let offset_table_start = header_len;
        let chunks_start = offset_table_start + 2 * 8;

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            // DECREASING_Y starts at max.y == 1.
            assert_eq!(of.current_scan_line(), 1);
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
            // After both lines, current advanced below min.y.
            assert_eq!(of.current_scan_line(), -1);
        }
        let bytes = out.into_data();

        // For DECREASING_Y the chunks are written line 1 first, then line 0.
        // The offset table is indexed by (y - minY) regardless of order, so
        // entry[1] = first chunk position, entry[0] = second chunk position.
        let chunk_len = 4 + 4 + 24u64; // y + dataSize + data
        let mut expected_offsets = [0u64; 2];
        expected_offsets[1] = chunks_start; // line 1 written first
        expected_offsets[0] = chunks_start + chunk_len; // line 0 written second

        let mut expected_table: Vec<u8> = Vec::new();
        for o in expected_offsets {
            expected_table.extend_from_slice(&o.to_le_bytes());
        }
        assert_eq!(
            &bytes[offset_table_start as usize..chunks_start as usize],
            expected_table.as_slice(),
            "DECREASING_Y offset table mismatch"
        );

        // First chunk on disk is line 1, second is line 0.
        let mut expected_chunks: Vec<u8> = Vec::new();
        for &y in &[1usize, 0usize] {
            expected_chunks.extend_from_slice(&(y as i32).to_le_bytes());
            expected_chunks.extend_from_slice(&24i32.to_le_bytes());
            expected_chunks.extend_from_slice(&expected_channel_xdr(100.0, y));
            expected_chunks.extend_from_slice(&expected_channel_xdr(200.0, y));
            expected_chunks.extend_from_slice(&expected_channel_xdr(300.0, y));
        }
        assert_eq!(
            &bytes[chunks_start as usize..],
            expected_chunks.as_slice(),
            "DECREASING_Y chunk data mismatch"
        );
    }

    #[test]
    fn missing_channel_is_filled_with_zeroes() {
        // Header has B, G, R but the frame buffer omits G → G must be zeroes.
        let header = make_header(LineOrder::IncreasingY);
        let (pixels, mut fb) = make_pixels();
        // Remove G by rebuilding the frame buffer without it.
        fb = {
            let mut nfb = FrameBuffer::new();
            nfb.insert("B", fb.find_slice("B").unwrap().clone()).unwrap();
            nfb.insert("R", fb.find_slice("R").unwrap().clone()).unwrap();
            nfb
        };

        let mut hdr_stream = MemOStream::new();
        header.write_to(&mut hdr_stream, 0).unwrap();
        let header_len = hdr_stream.data().len() as u64;
        let chunks_start = header_len + 2 * 8;

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
        }
        let bytes = out.into_data();

        let mut expected_chunks: Vec<u8> = Vec::new();
        for y in 0..2usize {
            expected_chunks.extend_from_slice(&(y as i32).to_le_bytes());
            expected_chunks.extend_from_slice(&24i32.to_le_bytes());
            expected_chunks.extend_from_slice(&expected_channel_xdr(100.0, y)); // B
            expected_chunks.extend_from_slice(&[0u8; 8]); // G (zeroes)
            expected_chunks.extend_from_slice(&expected_channel_xdr(300.0, y)); // R
        }
        assert_eq!(&bytes[chunks_start as usize..], expected_chunks.as_slice());
    }

    #[test]
    fn no_frame_buffer_errors() {
        let header = make_header(LineOrder::IncreasingY);
        let mut out = MemOStream::new();
        let mut of = OutputFile::new(&mut out, &header).unwrap();
        let err = of.write_pixels(&[], 1).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn writing_too_many_lines_errors() {
        // C++ clamps an over-request to whatever the
        // data window has left and returns `Ok` -- it does NOT error partway
        // through (`ImfOutputFile.cpp:1085-1107` checks `missingScanLines <=
        // 0` once, at the top, before doing anything). Asking for 3 lines on
        // a 2-line window therefore writes the 2 lines that exist.
        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();
        let mut out = MemOStream::new();
        let mut of = OutputFile::new(&mut out, &header).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, 3).unwrap();
        assert_eq!(of.current_scan_line(), 2, "must have written exactly the 2 available lines");

        // The window is now fully written. C++'s unconditional entry check
        // means even `write_pixels(0)` errors here -- it never even looks at
        // how many lines were requested.
        let err = of.write_pixels(&pixels, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    /// ZIP (16 lines/chunk) write then read back: a 5x40 single-channel image
    /// (40 rows > 16, so it spans 3 chunks: 16 + 16 + 8) must round-trip pixel
    /// exact, proving the multi-line block batching on the write side and the
    /// multi-line scatter on the read side agree.
    #[test]
    fn zip_multiline_block_write_read_round_trip() {
        use crate::scanline::InputFile;
        use crate::io::MemIStream;

        const W: i32 = 5;
        const H: i32 = 40; // 3 ZIP blocks: 16 + 16 + 8

        let mut header = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        header.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        // One HALF channel, row-major. value = half(y*W + x).
        let row_bytes = W as usize * 2;
        let mut pixels = vec![0u8; row_bytes * H as usize];
        for y in 0..H {
            for x in 0..W {
                let off = y as usize * row_bytes + x as usize * 2;
                let v = Half::from((y * W + x) as f32).bits();
                pixels[off..off + 2].copy_from_slice(&v.to_ne_bytes());
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

        let mut mem = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut mem, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, H).unwrap();
        }
        let bytes = mem.into_data();

        // Read back with our reader and compare every pixel.
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().compression().unwrap(), Compression::Zip);
        let mut out = vec![0u8; row_bytes * H as usize];
        inf.set_frame_buffer(&fb).unwrap();
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, pixels, "ZIP multi-line round trip pixel mismatch");
    }

    /// ADD-2: the OWNED-stream path.  A struct OWNS both its `StdOfStream`
    /// (boxed, inside the writer) and the persisted [`OutputFileOwned`] in ONE
    /// struct, writing scanlines across MULTIPLE `write_pixels` calls — proving
    /// the lifetime works (impossible with the borrowing default in safe Rust).
    /// The temp file is read back via [`InputFile`] and the pixels asserted to
    /// match, and the owned file is shown byte-identical to the borrow path.
    #[test]
    fn owned_stream_persisted_writer_round_trip_and_byte_parity() {
        use crate::io::{StdIfStream, StdOfStream};
        use crate::scanline::InputFile;

        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();

        let path = std::env::temp_dir()
            .join(format!("exr_core_owned_scan_{}.exr", std::process::id()));
        let path_str = path.to_str().unwrap().to_owned();

        // Holder owning both the stream (inside the writer) and the writer.
        struct OwningWriter<'a> {
            writer: OutputFileOwned<'a>,
        }

        let owned_stream: Box<dyn OStream> = Box::new(StdOfStream::create(&path_str).unwrap());
        let writer = OutputFile::new_owned(owned_stream, &header).unwrap();
        let mut holder = OwningWriter { writer };

        holder.writer.set_frame_buffer(&fb).unwrap();
        // Two SEPARATE write_pixels calls (1 line each) — the writer outlives
        // each call, which is the whole point of owning the stream.
        holder.writer.write_pixels(&pixels, 1).unwrap();
        holder.writer.write_pixels(&pixels, 1).unwrap();
        drop(holder.writer.finish().unwrap());

        // `finish` must refuse a half-written file instead of reporting success.
        {
            let mut partial = MemOStream::new();
            let mut w = OutputFile::new(&mut partial, &header).unwrap();
            w.set_frame_buffer(&fb).unwrap();
            w.write_pixels(&pixels, 1).unwrap();
            let err = w.finish().err().expect("1 of 2 scanlines written: finish must fail");
            assert!(err.to_string().contains("1 scanlines were never written"), "unexpected message: {err}");
        }

        let owned_bytes = std::fs::read(&path_str).unwrap();
        std::fs::remove_file(&path_str).ok();

        // Borrow path, same input → byte-identical.
        let mut mem = MemOStream::new();
        {
            let mut borrow_writer = OutputFile::new(&mut mem, &header).unwrap();
            borrow_writer.set_frame_buffer(&fb).unwrap();
            borrow_writer.write_pixels(&pixels, 2).unwrap();
        }
        let borrow_bytes = mem.into_data();
        assert_eq!(
            owned_bytes, borrow_bytes,
            "owned-stream output must be byte-identical to the borrow path"
        );

        // Read back via InputFile over a real file and assert pixels match.
        let path2 = std::env::temp_dir()
            .join(format!("exr_core_owned_scan_rb_{}.exr", std::process::id()));
        let path2_str = path2.to_str().unwrap().to_owned();
        std::fs::write(&path2_str, &owned_bytes).unwrap();
        let mut is = StdIfStream::open(&path2_str).unwrap();
        let mut inf = InputFile::new(&mut is).unwrap();
        let mut out = vec![0u8; pixels.len()];
        inf.set_frame_buffer(&fb).unwrap();
        inf.read_pixels(0, 1, &mut out).unwrap();
        drop(inf);
        std::fs::remove_file(&path2_str).ok();
        assert_eq!(out, pixels, "owned-path round trip pixel mismatch");
    }

    /// A block where NO channel has a sample (every
    /// channel's `ySampling` skips every line in the block) must be written
    /// and read back as a zero-byte chunk, matching C++ `Compressor::compress`
    /// short-circuiting `inSize == 0` BEFORE calling the codec's own
    /// `compress()` (`ImfCompressor.cpp:141-147`) -- never actually invoking
    /// the codec on empty input. One HALF channel `R` with `ySampling = 64`
    /// over a 128-row image: blocks covering rows [32,63] and [96,127]
    /// contain no sampled row at all (only y=0 and y=64 are ever sampled),
    /// while the blocks covering [0,31] and [64,95] each contain exactly one.
    /// Exercised over PIZ/ZIP (16/32-line blocks, whose own `compress()`
    /// already happens to tolerate empty input) and DWAA/HTJ2K32 -- the
    /// audit's flagged risk cases (DWA writes a fixed-size header even for
    /// zero channels; HTJ2K's codestream framing is not a no-op on empty
    /// input either) -- covering every block height in the module.
    #[test]
    fn writer_skips_codec_on_a_block_with_no_sampled_line() {
        for comp in [
            Compression::Piz,
            Compression::Zip,
            Compression::Dwaa,
            Compression::Htj2k32,
        ] {
            let width = 8i32;
            let height = 128i32;
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
            let mut cl = ChannelList::new();
            cl.insert("R", Channel::new(PixelType::Half, 1, 64, false))
                .unwrap();
            h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

            // Only y=0 and y=64 are ever sampled (divp(y, 64) selects row 0
            // or row 1), so the caller-side buffer needs exactly 2 rows.
            let row_bytes = width as usize * 2;
            let mut buf = vec![0u8; row_bytes * 2];
            for (row, base) in [(0usize, 1.0f32), (1usize, 100.0f32)] {
                for x in 0..width as usize {
                    let bits = Half::from(base + x as f32).bits().to_ne_bytes();
                    let off = row * row_bytes + x * 2;
                    buf[off..off + 2].copy_from_slice(&bits);
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
                    y_sampling: 64,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();

            let mut out = MemOStream::new();
            {
                let mut of = OutputFile::new(&mut out, &h).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                of.write_pixels(&buf, height).unwrap();
            }
            let bytes = out.into_data();

            let mut is = MemIStream::new(bytes);
            let mut inf = InputFile::new(&mut is).unwrap();
            let mut read_buf = vec![0u8; row_bytes * 2];
            inf.set_frame_buffer(&fb).unwrap();
            inf.read_pixels(0, height - 1, &mut read_buf).unwrap();
            assert_eq!(read_buf, buf, "round trip mismatch for {comp:?}");
        }
    }

    /// Port of `OutputFile::updatePreviewImage` (`ImfOutputFile.cpp:1363-1411`).
    /// Writes a file whose header carries a 2x2 preview image, writes the
    /// scanline pixels, then calls `update_preview_image` with new preview
    /// pixels and asserts (a) the pixel data is untouched (the saved/restored
    /// file position round trip did not clobber it) and (b) re-reading the
    /// header's `preview` attribute from the resulting bytes shows the NEW
    /// pixels, not the original ones.
    #[test]
    fn update_preview_image_rewrites_pixels_in_place_and_preserves_scan_lines() {
        use crate::attr::{PreviewImage, PreviewRgba};

        let mut header = make_header(LineOrder::IncreasingY);
        let original_preview = PreviewImage::new(
            2,
            2,
            Some(vec![
                PreviewRgba::new(1, 2, 3, 4),
                PreviewRgba::new(5, 6, 7, 8),
                PreviewRgba::new(9, 10, 11, 12),
                PreviewRgba::new(13, 14, 15, 16),
            ]),
        )
        .unwrap();
        header.set_preview_image(original_preview).unwrap();

        let (pixels, fb) = make_pixels();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();

            let new_pixels = vec![
                PreviewRgba::new(100, 101, 102, 103),
                PreviewRgba::new(104, 105, 106, 107),
                PreviewRgba::new(108, 109, 110, 111),
                PreviewRgba::new(112, 113, 114, 115),
            ];
            of.update_preview_image(&new_pixels).unwrap();
        }
        let bytes = out.into_data();

        // Scan line pixel data must be exactly as written -- proves the
        // saved/seek/restore dance around the preview rewrite left the file
        // cursor (and therefore every other byte) untouched.
        let mut is = MemIStream::new(bytes.clone());
        let mut inf = InputFile::new(&mut is).unwrap();
        let mut read_back = vec![0u8; pixels.len()];
        inf.set_frame_buffer(&fb).unwrap();
        inf.read_pixels(0, 1, &mut read_back).unwrap();
        drop(inf);
        assert_eq!(read_back, pixels, "scan line data must survive a preview rewrite");

        // The header's own preview attribute (re-parsed from the bytes) must
        // show the NEW pixels.
        let mut is2 = MemIStream::new(bytes);
        let (reread, _version) = Header::read_checked(&mut is2).unwrap();
        assert_eq!(
            reread.preview_image().unwrap().pixels(),
            &[
                PreviewRgba::new(100, 101, 102, 103),
                PreviewRgba::new(104, 105, 106, 107),
                PreviewRgba::new(108, 109, 110, 111),
                PreviewRgba::new(112, 113, 114, 115),
            ]
        );
    }

    /// C++ throws `LogicExc` when the file has no preview image
    /// (`previewPosition <= 0`).
    #[test]
    fn update_preview_image_without_a_preview_is_a_logic_error() {
        use crate::attr::PreviewRgba;

        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();
        let mut out = MemOStream::new();
        let mut of = OutputFile::new(&mut out, &header).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, 2).unwrap();

        let err = of
            .update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)])
            .unwrap_err();
        assert!(
            matches!(err, ExrError::Logic(_)),
            "expected a LogicExc-equivalent, got {err:?}"
        );
    }

    /// Port of `OutputFile::breakScanLine` (`ImfOutputFile.cpp:1417-1441`):
    /// overwrites `length` bytes at `offset` within an already-written
    /// scanline's chunk with byte `c`. Writes a 1-line-per-chunk (NONE) 4x2
    /// image, corrupts the first 2 bytes of row 1's chunk, and asserts (a)
    /// the on-disk bytes at that exact position changed and (b) row 0's
    /// chunk (a different offset-table slot) is untouched.
    #[test]
    fn break_scan_line_overwrites_bytes_of_an_already_written_chunk() {
        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
        }
        let original_bytes = out.into_data();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
            of.break_scan_line(1, 0, 2, 0xAA).unwrap();
        }
        let broken_bytes = out.into_data();

        assert_eq!(
            original_bytes.len(),
            broken_bytes.len(),
            "breakScanLine only overwrites, never resizes"
        );
        let diff_count = original_bytes
            .iter()
            .zip(broken_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff_count, 2, "exactly the 2 broken bytes must differ");
        assert!(
            broken_bytes.windows(2).any(|w| w == [0xAA, 0xAA]),
            "the broken bytes must be present somewhere in the file"
        );
    }

    /// A scan line whose chunk has not been written yet (offset-table slot
    /// still 0) must error, matching C++'s `ArgExc`.
    #[test]
    fn break_scan_line_on_an_unwritten_line_errors() {
        let header = make_header(LineOrder::IncreasingY);
        let mut out = MemOStream::new();
        let mut of = OutputFile::new(&mut out, &header).unwrap();
        let (_, fb) = make_pixels();
        of.set_frame_buffer(&fb).unwrap();
        // No `write_pixels` call at all -- every offset-table slot is still 0.
        let err = of.break_scan_line(0, 0, 1, 0).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)), "expected an ArgExc-equivalent, got {err:?}");
    }

    // -----------------------------------------------------------------
    // OutputFile::copy_pixels (from a scanline InputFile/InputPart)
    // -----------------------------------------------------------------

    /// `copy_pixels` from an `InputFile` must reproduce the source file
    /// byte-for-byte — ZIP (not NONE) so this actually proves the copy is raw
    /// (never decompress+recompress): with NONE the "raw" and "compressed"
    /// bytes are the same thing either way. Mirrors
    /// `tiled::output::tests::copy_pixels_tiled_to_tiled_byte_identical_increasing_and_random_y`.
    #[test]
    fn copy_pixels_from_input_file_byte_identical() {
        for line_order in [LineOrder::IncreasingY, LineOrder::DecreasingY] {
            let mut header = make_header(line_order);
            *header.compression_mut().unwrap() = Compression::Zip;
            let (pixels, fb) = make_pixels();

            let mut src_out = MemOStream::new();
            {
                let mut src = OutputFile::new(&mut src_out, &header).unwrap();
                src.set_frame_buffer(&fb).unwrap();
                src.write_pixels(&pixels, 2).unwrap();
            }
            let src_bytes = src_out.into_data();

            let mut src_is = MemIStream::new(src_bytes.clone());
            let mut src_in = InputFile::new(&mut src_is).unwrap();

            let mut dst_out = MemOStream::new();
            {
                let mut dst = OutputFile::new(&mut dst_out, &header).unwrap();
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
    /// data (C++ `LogicExc` "already contains pixel data").
    #[test]
    fn copy_pixels_rejects_when_destination_already_has_pixel_data() {
        let header = make_header(LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels();

        let mut src_out = MemOStream::new();
        {
            let mut src = OutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_pixels(&pixels, 2).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = InputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        let mut dst = OutputFile::new(&mut dst_out, &header).unwrap();
        dst.set_frame_buffer(&fb).unwrap();
        dst.write_pixels(&pixels, 1).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)), "{err:?}");
    }

    /// `copy_pixels` must reject a tiled source (C++ ArgExc "the input file
    /// is tiled").
    #[test]
    fn copy_pixels_rejects_a_tiled_source() {
        use crate::attr::{LevelMode, TileDescription, TileDescriptionAttribute};
        use crate::tiled::TiledOutputFile;

        let mut tiled_header = make_header(LineOrder::IncreasingY);
        tiled_header
            .insert(
                "tiles",
                TileDescriptionAttribute::new(TileDescription {
                    x_size: 4,
                    y_size: 2,
                    mode: LevelMode::OneLevel,
                    rounding_mode: crate::attr::tile_description::LevelRoundingMode::RoundDown,
                }),
            )
            .unwrap();
        let (pixels, fb) = make_pixels();

        let mut src_out = MemOStream::new();
        {
            let mut src = TiledOutputFile::new(&mut src_out, &tiled_header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_tile(&pixels, 0, 0, 0, 0).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = InputFile::new(&mut src_is).unwrap();

        let header = make_header(LineOrder::IncreasingY);
        let mut dst_out = MemOStream::new();
        let mut dst = OutputFile::new(&mut dst_out, &header).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
    }

    /// `copy_pixels` via a multipart `InputPart` source (C++
    /// `OutputFile::copyPixels(InputPart&)`).
    #[test]
    fn copy_pixels_from_multipart_input_part() {
        use crate::multipart::{MultiPartInputFile, MultiPartOutputFile};

        let mut header = make_header(LineOrder::IncreasingY);
        *header.compression_mut().unwrap() = Compression::Zip;
        header.insert("name", crate::attr::StringAttribute::new("part0".into())).unwrap();
        header.insert("type", crate::attr::StringAttribute::new("scanlineimage".into())).unwrap();
        let (pixels, fb) = make_pixels();

        let mut src_out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut src_out, &[header.clone()]).unwrap();
            let mut p = mp.output_part(0).unwrap();
            p.set_frame_buffer(&fb).unwrap();
            p.write_pixels(&pixels, 2).unwrap();
        }
        let src_bytes = src_out.into_data();

        let mut src_is = MemIStream::new(src_bytes);
        let mut mpi = MultiPartInputFile::new(&mut src_is).unwrap();
        let mut src_in = mpi.input_part_wrapper(0).unwrap();

        let plain_header = make_header(LineOrder::IncreasingY);
        let mut plain_header = plain_header;
        *plain_header.compression_mut().unwrap() = Compression::Zip;
        let mut dst_out = MemOStream::new();
        {
            let mut dst = OutputFile::new(&mut dst_out, &plain_header).unwrap();
            dst.copy_pixels(&mut src_in).unwrap();
        }

        // Verify by reading the copy back and comparing decoded pixels
        // (the multipart source's on-disk bytes carry a partNumber prefix
        // the plain destination does not, so a byte-identical assertion
        // does not apply here -- unlike the single-part InputFile case).
        let mut check_is = MemIStream::new(dst_out.into_data());
        let mut check_in = InputFile::new(&mut check_is).unwrap();
        let mut got = vec![0u8; pixels.len()];
        let (_, check_fb) = make_pixels();
        check_in.set_frame_buffer(&check_fb).unwrap();
        check_in.read_pixels(0, 1, &mut got).unwrap();
        assert_eq!(got, pixels, "copy_pixels from an InputPart must reproduce the source pixels");
    }

    /// The scanline ring really overlaps: with a real codec and 32 blocks,
    /// blocks are emitted while later blocks are still encoding (a
    /// window-at-a-time writer never does). Mutation: emit only after the
    /// window is fully produced -> this fails.
    #[test]
    fn parallel_write_emits_while_later_blocks_encode() {
        let (w, h) = (256i32, 512i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);
        for &n in &[2i32, 8] {
            assert!(
                crate::probe::saw_overlap(|| {
                    let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
                    let mut out = MemOStream::new();
                    let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                    of.set_frame_buffer(&fb).unwrap();
                    of.write_pixels(&pixels, h).unwrap();
                }),
                "num_threads={n}: no block was emitted while another was still encoding"
            );
        }
    }

    /// The reported deadlock: parallel writers invoked from workers of the very
    /// pool they submit to (nested parallelism). Every worker used to block on
    /// its own pipeline's channel, so nothing ran. Watchdog: a regression
    /// fails the test instead of hanging the suite.
    #[test]
    fn parallel_write_from_pool_workers_does_not_deadlock() {
        use rayon::prelude::*;
        let (w, h) = (64i32, 256i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (p2, fb2) = make_pixels_wh(w as usize, h as usize);
        std::thread::spawn(move || {
            let (pixels, fb) = (p2, fb2);
            let count = crate::threading::ThreadCount::capture(Some(2)).unwrap();
            let files = crate::threading::run_on_global_pool(count, || {
                (0..8)
                    .into_par_iter()
                    .map(|_| {
                        let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
                        let mut out = MemOStream::new();
                        let mut of = OutputFile::with_num_threads(&mut out, &header, Some(2)).unwrap();
                        of.set_frame_buffer(&fb).unwrap();
                        of.write_pixels(&pixels, h).unwrap();
                        drop(of);
                        out.into_data()
                    })
                    .collect::<Vec<_>>()
            });
            done_tx.send(files).ok();
        });
        let files = done_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("parallel writers deadlocked when called from pool workers")
            .unwrap();
        // Inline (nested) and pipelined writes produce the same bytes.
        let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
        let mut out = MemOStream::new();
        let mut of = OutputFile::with_num_threads(&mut out, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, h).unwrap();
        drop(of);
        let reference = out.into_data();
        for f in files {
            assert_eq!(f, reference);
        }
    }

    /// Writer-side pool reuse: over 256 blocks the codec pool constructs at
    /// most one codec per worker (a lease that never returned would build one
    /// per block) and every pooled buffer `take` hits (a missing `give` would
    /// starve the prewarmed free list). Mutations: drop the lease's return ->
    /// `created` explodes; drop `pool.give` -> misses.
    #[test]
    fn parallel_write_reuses_codecs_and_buffers() {
        use crate::probe::{get, reset, Probe};
        let (w, h) = (8i32, 4096i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);
        for &n in &[2i32, 8] {
            for p in [Probe::CodecCreated, Probe::BufPoolMiss, Probe::BufPoolHit] {
                reset(p);
            }
            let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
            let mut out = MemOStream::new();
            {
                let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                of.write_pixels(&pixels, h).unwrap();
            }
            assert!(get(Probe::CodecCreated) <= n as u64, "num_threads={n}: {} codecs built for 256 blocks", get(Probe::CodecCreated));
            assert_eq!(get(Probe::BufPoolMiss), 0, "num_threads={n}: a buffer was not recycled");
            assert_eq!(get(Probe::BufPoolHit), 256, "num_threads={n}: every block takes a pooled buffer");
        }
    }

    /// A mid-stream `produce` failure (the pixel buffer ends before the last
    /// blocks) is reported as the error, and the leases still returned every
    /// codec: at most one per worker was ever built.
    #[test]
    fn parallel_write_error_mid_stream_returns_codecs() {
        use crate::probe::{get, reset, Probe};
        let (w, h) = (8i32, 4096i32);
        let (pixels, fb) = make_pixels_wh(w as usize, h as usize);
        let short = &pixels[..pixels.len() / 2];
        reset(Probe::CodecCreated);
        let header = make_header_wh(w, h, LineOrder::IncreasingY, Compression::Zip);
        let mut empty = MemOStream::new();
        drop(OutputFile::with_num_threads(&mut empty, &header, Some(4)).unwrap());
        let prologue = empty.into_data().len();
        let mut out = MemOStream::new();
        let mut of = OutputFile::with_num_threads(&mut out, &header, Some(4)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        let err = of.write_pixels(short, h).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
        drop(of);
        assert!(out.into_data().len() > prologue, "the failure must come after earlier blocks were written");
        assert!(get(Probe::CodecCreated) <= 4, "{} codecs built", get(Probe::CodecCreated));
    }
}
