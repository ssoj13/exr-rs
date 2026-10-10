//! Port of OpenEXR `ImfDeepScanLineOutputFile.cpp` / `.h` —
//! `class DeepScanLineOutputFile`, the single-part **deep scanline** writer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineOutputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineOutputFile.h>
//!
//! # The deep scanline chunk byte layout (confirmed from `writePixelData`)
//!
//! Each chunk written by `writePixelData` (ImfDeepScanLineOutputFile.cpp,
//! ~lines 270–360) has, in order (all multi-byte fields Xdr little-endian):
//!
//! | field                      | type  | bytes | notes |
//! |----------------------------|-------|-------|-------|
//! | `[partNumber]`             | i32   | 4     | only when `multipart` — NOT here (single-part) |
//! | `lineBufferMinY`           | i32   | 4     | y of the first scanline of the chunk |
//! | `packedSampleCountTableSize` | u64 | 8     | size in bytes of the (compressed) count table |
//! | `packedDataSize`           | u64   | 8     | size in bytes of the (compressed) sample data |
//! | `unpackedDataSize`         | u64   | 8     | size in bytes of the raw sample data |
//! | sample-count table         | bytes | `packedSampleCountTableSize` | compressed separately |
//! | sample data                | bytes | `packedDataSize` | compressed by the data compressor |
//!
//! For deep, the only valid codecs are NONE / RLE / ZIPS — all of which have
//! `numScanLines() == 1`, so `linesInBuffer == 1`: every chunk holds exactly
//! one scanline, and the offset table has one entry per scanline.  (16-line ZIP
//! is *not* deep-valid; see [`crate::attr::Compression::is_valid_deep`].)
//!
//! # The cumulative sample-count table (confirmed from `LineBufferTask::execute`)
//!
//! For every scanline `y` in the chunk, C++ walks `x` from `minX` to `maxX`
//! keeping a **running total** `count` that **resets to 0 at the start of each
//! scanline** and, after adding `getSampleCount(x, y)`, writes the running
//! `count` as an Xdr `int` (4 bytes):
//!
//! ```text
//! for y in [minY, maxY]:
//!     count = 0
//!     for x in [minX, maxX]:
//!         count += sampleCount(x, y)
//!         Xdr::write<int>(count)     // cumulative running total, LE
//! ```
//!
//! The table is then compressed by its **own** compressor instance
//! (`sampleCountTableCompressor`, separate from the sample-data compressor); if
//! it does not shrink (or there is no compressor) the raw table is stored and
//! `packedSampleCountTableSize == rawTableSize`.
//!
//! # Sample-data gather (confirmed from `copyFromDeepFrameBuffer`)
//!
//! For every scanline `y`, then every channel slice, then every pixel `x`, the
//! channel's `count = sampleCount(x, y)` samples are emitted in Xdr order.  In
//! C++ the channel `base` is a `char**` (per-pixel pointer into a sample heap):
//! `readPtr = ((const char**)(base + y*yStride + x*xStride))[0]`, then `count`
//! samples are read `sampleStride` bytes apart.  Channels absent from the frame
//! buffer are filled with zeroes for the line's *total* sample count.
//!
//! # Safe-Rust adaptation of the `char**` deep addressing
//!
//! The C++ deep `base` is a pointer-to-pointers (`char**`) that we cannot model
//! with raw pointers (this crate has zero `unsafe` and no FFI).  We
//! keep the *algorithm* 1:1 and replace only the storage model: the caller
//! hands [`DeepScanLineOutputFile::set_frame_buffer`] a **flat per-channel
//! sample heap** (`sample_data: &[u8]`) in which each pixel's sample array is
//! laid out contiguously in `(x, y)` raster order, and a **flat sample-count
//! buffer** (`sample_counts: &[u32]`) addressed by the sample-count slice.
//! A deep slice's `base_offset` is then the byte offset of its channel's heap
//! within `sample_data`; the per-pixel array start is the running sum of
//! `count * sample_stride` over all preceding pixels of that channel.  This is
//! the same descriptor adaptation [`crate::frame_buffer::Slice`] uses for the
//! flat (non-deep) case, extended to the variable-length deep case.  The bytes
//! written to the file are byte-identical to the C++ writer's.
//!
//! # Offset back-patch vs C++ `Drop`
//!
//! C++ rewrites the now-filled line-offset table from its destructor.  Like
//! [`crate::scanline::OutputFile`] and [`crate::tiled::TiledOutputFile`], this
//! port back-patches each chunk's offset into the reserved table slot
//! immediately after the chunk is written (seek-back, write `u64` LE,
//! seek-forward), so the file is valid without relying on `Drop` (the borrowed
//! stream cannot be finalised from `Drop` safely).

use crate::attr::PixelType;
use crate::compressor::{
    default_format, new_compressor, num_lines_in_buffer, stored_chunk, Compressor, Format,
};
use crate::container::{xdr, SIZE_I64};
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::deep::sample_table::{encode_cumulative_row, sample_count_at};
use crate::deep::scanline_input::DeepScanLineInputFile;
use crate::error::{ExrError, Result};
use crate::header::{Header, DEEP_SCANLINE};
use crate::io::OStream;
use crate::misc::pixel_type_size;

use imath_rs::modp;

/// Source of raw (never decompressed) deep scanline chunks for
/// [`DeepScanLineOutputFile::copy_pixels`] / [`DeepScanLineOutputFileState::copy_pixels`]
/// -- one impl per C++ `copyPixels` overload's source type
/// (`DeepScanLineInputFile`, `DeepScanLineInputPart`).
///
/// Duck-typed (no lifetime parameter, no "hand back the owned concrete
/// reader" method, unlike an `as_deep_scanline_input(&mut self) ->
/// Result<&mut DeepScanLineInputFile<'a>>` shape) -- mirrors
/// `crate::tiled::output::TiledCopySource`: a borrowing multipart
/// part-reader view, `DeepScanLineInputPartReader`, never owns a concrete
/// `DeepScanLineInputFile` to hand back, the same constraint that shapes
/// `TiledCopySource`. Implemented
/// directly for [`DeepScanLineInputFile`] here; `multipart::part::
/// DeepScanLineInputPart` implements it in `multipart/part.rs` (trait
/// bounds on a generic parameter are not satisfied via `Deref` coercion the
/// way a concrete-type call site is -- same reasoning `TiledInputPart`'s own
/// forwarding impl documents).
pub trait DeepScanLineCopySource {
    /// This source's header.
    fn header(&self) -> &Header;
    /// Scanline `first_scan_line`'s chunk RAW (still-packed count table +
    /// sample data, no decompression) -- C++ `rawPixelData`.
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>, Vec<u8>, u64)>;
}

impl<'a> DeepScanLineCopySource for DeepScanLineInputFile<'a> {
    fn header(&self) -> &Header {
        DeepScanLineInputFile::header(self)
    }
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>, Vec<u8>, u64)> {
        DeepScanLineInputFile::raw_pixel_data(self, first_scan_line)
    }
}

// ---------------------------------------------------------------------------
// OutSliceInfo — port of the anonymous-namespace `struct OutSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel deep write descriptor — port of `OutSliceInfo`
/// (ImfDeepScanLineOutputFile.cpp lines 58–101).
///
/// Built by [`DeepScanLineOutputFile::set_frame_buffer`] from the file's
/// channel list and the user [`DeepFrameBuffer`].  Mirrors the C++ struct
/// field-for-field, except that the raw `const char* base` (a `char**` for
/// deep) becomes a signed byte offset (`base_offset`) into the caller-owned
/// flat sample heap — see the module-level docs for the safe-Rust adaptation.
#[derive(Clone, Debug)]
struct OutSliceInfo {
    /// Sample type stored on disk for this channel (C++ `type`).
    type_: PixelType,
    /// Signed byte offset of this channel's flat sample heap within the
    /// caller's `sample_data` buffer.  Replaces the C++ `char* base`
    /// (`0`/unused when `zero` is `true`).
    base_offset: i64,
    /// Byte distance between consecutive samples of one pixel — C++
    /// `sampleStride`.
    sample_stride: i32,
    /// Byte distance between horizontally adjacent *pixel pointer slots* — C++
    /// `xStride`.  In the safe-Rust flat model this is unused for addressing
    /// (samples are laid out contiguously) but kept for structural fidelity.
    #[allow(dead_code)]
    x_stride: i32,
    /// Byte distance between vertically adjacent *pixel pointer slots* — C++
    /// `yStride`.  Unused in the flat model (see `x_stride`); kept for fidelity.
    #[allow(dead_code)]
    y_stride: i32,
    /// Horizontal sub-sampling factor (C++ `xSampling`).  Deep channels are
    /// required to have `xSampling == 1` (enforced by `Header::sanity_check`),
    /// so the deep gather walks every x; kept for structural fidelity.
    #[allow(dead_code)]
    x_sampling: i32,
    /// Vertical sub-sampling factor (C++ `ySampling`).  Used by the deep gather
    /// to skip scanlines where `modp(y, ySampling) != 0`.
    y_sampling: i32,
    /// `true` when the file channel has no matching frame-buffer slice; the
    /// data buffer is filled with zeroes for this channel (C++ `zero`).
    zero: bool,
}

// ---------------------------------------------------------------------------
// DeepScanLineOutputFile — port of `class DeepScanLineOutputFile`
// ---------------------------------------------------------------------------

/// Single-part deep scanline EXR writer — port of `Imf::DeepScanLineOutputFile`.
///
/// Construct with [`DeepScanLineOutputFile::new`] (writes the header with the
/// deep flag + reserves the line-offset table), describe the source pixels with
/// [`DeepScanLineOutputFile::set_frame_buffer`], then push scanlines with
/// [`DeepScanLineOutputFile::write_pixels`].
///
/// The writer borrows the caller's [`OStream`] for its whole lifetime and
/// back-patches each offset-table entry immediately after writing the
/// corresponding chunk (see the module docs).
pub struct DeepScanLineOutputFile<'a> {
    /// Borrowed output stream (the C++ `_streamData->os`).
    stream: &'a mut dyn OStream,

    /// Everything else this writer owns (mirroring
    /// `crate::scanline::OutputFileState` / `crate::tiled::output::
    /// TiledOutputFileState`): factored into its own, stream-free struct so
    /// [`crate::multipart::MultiPartOutputFile`] can persist it across
    /// separate [`crate::multipart::MultiPartOutputFile::deep_scanline_output_part`]
    /// calls instead of rebuilding (and silently resetting `current_scan_line`
    /// / `missing_scan_lines`) a fresh writer every call. See
    /// [`DeepScanLineOutputFileState`].
    state: DeepScanLineOutputFileState,
}

/// Every field of C++ `DeepScanLineOutputFile::Data` except the stream — the
/// writer's **persisted** per-part state.
///
/// Building a brand-new [`DeepScanLineOutputFile`] from the header on EVERY
/// `MultiPartOutputFile::deep_scanline_output_part(n)` call would silently
/// reset `current_scan_line`/`missing_scan_lines` — C++ instead
/// caches ONE writer object per part (`ImfMultiPartOutputFile.cpp:313-321`)
/// so writes interleaved across parts resume correctly. Splitting the writer
/// into this pure state struct plus the stream lets `MultiPartOutputFile`
/// persist a `Vec<Option<DeepScanLineOutputFileState>>` (one slot per deep
/// scanline part, built on first access) and hand out a borrowing
/// [`DeepScanLineOutputPartWriter`] view instead. The standalone
/// [`DeepScanLineOutputFile`] just owns one state for its whole life.
pub(crate) struct DeepScanLineOutputFileState {
    /// The image header (type forced to `deepscanline`) — C++ `_data->header`.
    header: Header,

    /// Line order copied from the header — C++ `_data->lineOrder`.
    line_order: crate::attr::LineOrder,

    /// Data-window bounds — C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    /// Max data-window y — C++ `_data->maxY`.  Consumed during construction
    /// (chunk-count / table sizing) and by the `DECREASING_Y` start line; kept
    /// as a field for structural fidelity with the C++ `Data` struct.
    #[allow(dead_code)]
    max_y: i32,

    /// Next scanline to be written — C++ `_data->currentScanLine`.
    current_scan_line: i32,
    /// Number of scanlines still to write — C++ `_data->missingScanLines`.
    missing_scan_lines: i32,

    /// Compressor data format (Xdr for NONE) — C++ `_data->format`.
    ///
    /// The deep-valid codecs (NONE/RLE/ZIPS) all use Xdr, and the gather here
    /// always emits Xdr, so the C++ `convertToXdr`-on-no-shrink path never
    /// fires; the field is kept for structural fidelity with the C++ `Data`.
    #[allow(dead_code)]
    format: Format,
    /// Scanlines per line buffer — C++ `_data->linesInBuffer` (always 1 for the
    /// deep-valid codecs NONE/RLE/ZIPS).
    lines_in_buffer: i32,

    /// File offsets of each chunk — C++ `_data->lineOffsets`.
    line_offsets: Vec<u64>,
    /// File position where the offset table begins — C++
    /// `_data->lineOffsetsPosition`.
    line_offsets_position: u64,

    /// Per-channel write descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][DeepScanLineOutputFile::set_frame_buffer]).
    slices: Vec<OutSliceInfo>,
    /// The caller's frame buffer, kept for [`frame_buffer`][Self::frame_buffer]
    /// — C++ `const DeepFrameBuffer& frameBuffer() const`. Cloned in
    /// `set_frame_buffer`; `DeepFrameBuffer::new()` (empty) until then.
    caller_frame_buffer: DeepFrameBuffer,

    /// Sample-count slice geometry — C++'s `sampleCountSliceBase`/`xStride`/
    /// `yStride`. Pixel data is passed PER WRITE CALL: this file
    /// stores only the LAYOUT; the actual count/sample BYTES are passed fresh
    /// to each [`write_pixels`][Self::write_pixels] call, never stored.
    sample_count_base_offset: i64,
    /// X byte stride for the sample-count buffer — C++ `sampleCountXStride`.
    sample_count_x_stride: i32,
    /// Y byte stride for the sample-count buffer — C++ `sampleCountYStride`.
    sample_count_y_stride: i32,
    /// A growable per-row sample-count prefix cache (reset in
    /// `set_frame_buffer`) so `samples_before_line` computes each row's total
    /// at most once across the whole write, instead of being recomputed from
    /// scratch per channel per line (`gather_scanline` calls it once per
    /// channel per line).
    sample_prefix: crate::deep::sample_table::RowPrefixCache,

    /// The sample-data compressor, or `None` for NONE — C++
    /// `LineBuffer::compressor` (re-created per block in C++; created once here
    /// since `linesInBuffer == 1`).
    compressor: Option<Box<dyn Compressor>>,
    /// The **separate** sample-count-table compressor, or `None` for NONE —
    /// C++ `LineBuffer::sampleCountTableCompressor`.
    count_compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this writer drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `DeepScanLineOutputFile::Data::{multipart, partNumber}`.
    ///
    /// The same unified seam as the flat scanline / tiled writers: when `Some`,
    /// every deep chunk is prefixed with the `partNumber` (i32 LE) exactly as
    /// C++ `writePixelData` does (`if (multiPart) Xdr::write(partNumber)`);
    /// when `None`, no prefix is written, so single-part output is
    /// byte-identical to the pre-multipart writer.
    part_number: Option<i32>,

    /// Byte position of the header's `"preview"` attribute pixels, as
    /// returned by `Header::write_to` — C++ `Data::previewPosition`. Set by
    /// [`new_standalone`](Self::new_standalone) (the real value) or by
    /// [`from_part_with_preview`](Self::from_part_with_preview) (the real
    /// value `MultiPartOutputFile` captured from
    /// `Header::write_attributes_to`); `0`
    /// only via the plain [`from_part`](Self::from_part), used by the public
    /// `DeepScanLineOutputFile::from_part` constructor, which does not
    /// itself write any header and so has no real value to supply.
    preview_position: u64,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the flat scanline writer's
    /// identical field.
    num_threads: crate::threading::ThreadCount,

    /// The codec — kept so the parallel path can build fresh per-worker
    /// codec instances via [`new_compressor`] (each worker owns one; codecs
    /// are not `Sync`).
    compression: crate::attr::Compression,
}

impl<'a> DeepScanLineOutputFile<'a> {
    /// Open a `DeepScanLineOutputFile` on `stream` for `header` — port of
    /// `DeepScanLineOutputFile::DeepScanLineOutputFile(OStream&, const Header&,
    /// int)` followed by `initialize`.
    ///
    /// Validates the header (deep `sanity_check`), forces the type to
    /// `deepscanline`, writes the magic number, version word (with the deep
    /// `NON_IMAGE_FLAG` set automatically by [`Header::version_field`]) and
    /// header, and reserves the line-offset table.
    ///
    /// # Errors
    /// * Propagates [`Header::sanity_check`] failures (including
    ///   "compression not valid for deep data").
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
        // version + header + reserves the offset table, and no per-chunk
        // partNumber prefix (`part_number == None`).
        Self::build(stream, header, None, None, num_threads)
    }

    /// Open a `DeepScanLineOutputFile` for **one part of a multipart file** —
    /// port of `DeepScanLineOutputFile::DeepScanLineOutputFile(const OutputPartData*)`.
    ///
    /// The enclosing [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile)
    /// has already written the file prologue (magic + version + every part
    /// header + the zero-filled per-part offset tables).  This constructor only
    /// rebuilds the per-part write state and remembers `part_number` (the i32
    /// prefix of every deep chunk) and `chunk_offset_table_position` (this
    /// part's reserved offset table, back-patched per chunk).  The stream
    /// cursor must already sit where this part's first chunk belongs.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineOutputFile.cpp> (`DeepScanLineOutputFile(const OutputPartData*)` + `writePixelData`).
    pub fn from_part(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        chunk_offset_table_position: u64,
    ) -> Result<Self> {
        Self::build(stream, header, part_number, Some(chunk_offset_table_position), None)
    }

    /// Same as [`Self::from_part`], with an explicit `num_threads` override.
    pub fn from_part_mt(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        chunk_offset_table_position: u64,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::build(stream, header, part_number, Some(chunk_offset_table_position), num_threads)
    }

    /// Shared constructor body for [`new`](Self::new) (single-part) and
    /// [`from_part`](Self::from_part) (multipart part).
    ///
    /// The actual state computation lives in
    /// [`DeepScanLineOutputFileState::new_standalone`] /
    /// [`DeepScanLineOutputFileState::from_part`] so
    /// [`DeepScanLineOutputPartWriter`] (which BORROWS a
    /// `DeepScanLineOutputFileState` instead of owning one) can share the
    /// exact same construction logic.
    fn build(
        stream: &'a mut dyn OStream,
        header: &Header,
        part_number: Option<i32>,
        offsets_position: Option<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = match offsets_position {
            None => DeepScanLineOutputFileState::new_standalone_mt(stream, header, num_threads)?,
            Some(pos) => DeepScanLineOutputFileState::from_part_mt(header, part_number, pos, num_threads)?,
        };
        Ok(Self { stream, state })
    }

    /// The image header — port of `const Header& DeepScanLineOutputFile::header()`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The currently set frame buffer — port of
    /// `const DeepFrameBuffer& DeepScanLineOutputFile::frameBuffer() const`.
    /// Empty until [`set_frame_buffer`][Self::set_frame_buffer] has
    /// been called.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk — port of `void updatePreviewImage(const PreviewRgba newPixels[])`
    /// (`ImfDeepScanLineOutputFile.cpp:891-918`). See
    /// [`DeepScanLineOutputFileState::update_preview_image`] for the full doc.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Next scanline to be written — port of
    /// `int DeepScanLineOutputFile::currentScanLine() const`.
    pub fn current_scan_line(&self) -> i32 {
        self.state.current_scan_line()
    }

    /// Set the pixel source — port of `DeepScanLineOutputFile::setFrameBuffer`.
    /// See [`DeepScanLineOutputFileState::set_frame_buffer`] for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write `num_scan_lines` scanlines — port of
    /// `DeepScanLineOutputFile::writePixels`. See
    /// [`DeepScanLineOutputFileState::write_pixels`] for the full algorithm doc.
    pub fn write_pixels(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        num_scan_lines: i32,
    ) -> Result<()> {
        self.state.write_pixels(self.stream, sample_counts, sample_data, num_scan_lines)
    }

    /// Quick pixel copy: transcode every scanline chunk of `input` straight
    /// into this file, raw (no decompress/recompress). See
    /// [`DeepScanLineOutputFileState::copy_pixels`] for the full doc.
    pub fn copy_pixels<T: DeepScanLineCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }
}

// ---------------------------------------------------------------------------
// DeepScanLineOutputFileState — the state-only construction + algorithm,
// shared by the owning `DeepScanLineOutputFile` and the borrowing
// `DeepScanLineOutputPartWriter`.
// ---------------------------------------------------------------------------


/// The emit-side state of the deep scanline writer: where each chunk's offset
/// is recorded. Borrowed field-by-field so the pipelined write can emit while
/// the encode workers keep reading the header and layout (no clones).
struct LineSink<'a> {
    line_offsets: &'a mut [u64],
    line_offsets_position: u64,
    part_number: Option<i32>,
    min_y: i32,
    lines_in_buffer: i32,
}

impl LineSink<'_> {
    /// Write one already-encoded chunk and back-patch its offset-table slot
    /// (`writePixelData`, `ImfDeepScanLineOutputFile.cpp`).
    fn write(
        &mut self,
        stream: &mut dyn OStream,
        y: i32,
        packed_count: &[u8],
        packed_data: &[u8],
        unpacked_data_size: u64,
    ) -> Result<()> {
        // --- Record chunk offset, write chunk, back-patch table slot ---
        let offset_index = ((y - self.min_y) / self.lines_in_buffer) as usize;
        let chunk_position = stream.tellp()?;
        *self.line_offsets.get_mut(offset_index).ok_or_else(|| {
            ExrError::Logic(format!("lineOffsets index {offset_index} out of range"))
        })? = chunk_position;

        // writePixelData byte order: for a multipart file the deep chunk is
        // prefixed with the part number (i32 LE) before `y` (C++
        // `if (multiPart) Xdr::write(partNumber)`) — the ONLY difference
        // from the single-part layout.  Then:
        //   y(i32), packedCountSize(u64), packedDataSize(u64),
        //   unpackedDataSize(u64), count table, data.
        if let Some(pn) = self.part_number {
            xdr::write_i32(&mut *stream, pn)?;
        }
        xdr::write_i32(&mut *stream, y)?;
        xdr::write_u64(&mut *stream, packed_count.len() as u64)?;
        xdr::write_u64(&mut *stream, packed_data.len() as u64)?;
        xdr::write_u64(&mut *stream, unpacked_data_size)?;
        stream.write(packed_count)?;
        stream.write(packed_data)?;

        // Back-patch the reserved offset slot (seek-back u64 LE).
        let saved = stream.tellp()?;
        stream.seekp(self.line_offsets_position + offset_index as u64 * SIZE_I64 as u64)?;
        xdr::write_u64(&mut *stream, chunk_position)?;
        stream.seekp(saved)?;
        Ok(())
    }
}

impl DeepScanLineOutputFileState {
    /// Pure state computation shared by
    /// [`new_standalone`](Self::new_standalone) and
    /// [`from_part`](Self::from_part) — everything C++
    /// `DeepScanLineOutputFile::initialize(header)` computes that does NOT
    /// depend on `line_offsets_position`/`preview_position` (which the two
    /// callers fill in differently: written fresh vs. handed in / left 0 by
    /// the container). Never touches the stream.
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default).
    fn compute_mt(
        header: &Header,
        part_number: Option<i32>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // C++ ctor: header.sanityCheck();  (single-part, not tiled)
        header.sanity_check(false, false)?;

        // C++ DeepScanLineOutputFile::initialize(header) -------------------
        let mut hdr = header.clone();
        // `ImfDeepScanLineOutputFile.cpp:869-871`: a part header that
        // already declares a different type cannot be forced into
        // deepscanline — catch the type-mismatched-part misuse before
        // silently overwriting it below.
        if hdr.has_type() && hdr.type_()? != DEEP_SCANLINE {
            return Err(ExrError::Arg(
                "Can't build a DeepScanLineOutputFile from a type-mismatched part.".into(),
            ));
        }
        // C++: _data->header.setType(DEEPSCANLINE);
        hdr.set_type(DEEP_SCANLINE)?;

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

        // C++: Compressor* c = newCompressor(compression, 0, header);
        //      format = defaultFormat(c); linesInBuffer = numLinesInBuffer(c);
        let compression = hdr.compression()?;
        let probe = new_compressor(compression, 0, &hdr)?;
        let format = default_format(probe.as_deref());
        let lines_in_buffer = match probe.as_deref() {
            Some(c) => c.num_scan_lines(),
            None => num_lines_in_buffer(compression)?,
        };
        drop(probe);

        // C++: lineOffsetSize = (maxY - minY + linesInBuffer) / linesInBuffer;
        //      header.setChunkCount(lineOffsetSize); lineOffsets.resize(...).
        let line_offset_size = ((max_y - min_y + lines_in_buffer) / lines_in_buffer) as usize;
        hdr.set_chunk_count(line_offset_size as i32)?;
        let line_offsets = vec![0u64; line_offset_size];

        // C++: maxSampleCountTableSize = min(linesInBuffer, maxY-minY+1) *
        //      (maxX - minX + 1) * sizeof(unsigned int);  used to size each
        //      LineBuffer's sampleCountTableCompressor.
        let lines_per_buffer = lines_in_buffer.min(max_y - min_y + 1) as usize;
        let width = (max_x - min_x + 1) as usize;
        let max_sample_count_table_size = lines_per_buffer * width * core::mem::size_of::<u32>();

        // The two SEPARATE compressors (C++ creates both):
        //  - sampleCountTableCompressor sized for maxSampleCountTableSize,
        //  - the sample-data compressor is created per block in C++ with
        //    maxBytesPerLine; since linesInBuffer == 1 we create it once with a
        //    conservative bound (it is re-fed exact sizes per call anyway).
        let count_compressor = new_compressor(compression, max_sample_count_table_size, &hdr)?;
        let compressor = new_compressor(compression, 0, &hdr)?;

        // Structural detail (debug): deep scanline writer is up.
        log::debug!(
            "deep DeepScanLineOutputFile: {}x{} part={part_number:?} codec={compression:?} \
             chunkOffsets={}",
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
            line_offsets,
            line_offsets_position: 0,
            slices: Vec::new(),
            caller_frame_buffer: DeepFrameBuffer::new(),
            sample_count_base_offset: 0,
            sample_count_x_stride: 0,
            sample_count_y_stride: 0,
            sample_prefix: crate::deep::sample_table::RowPrefixCache::new(min_x, max_x, min_y),
            compressor,
            count_compressor,
            part_number,
            preview_position: 0,
            num_threads,
            compression,
        })
    }

    /// Standalone single-part construction: writes the magic + version word,
    /// the header, and reserves the (zero-filled) line-offset table on
    /// `stream` — port of the C++ `DeepScanLineOutputFile` constructor body,
    /// minus the pure state computation shared with
    /// [`from_part`](Self::from_part).
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default).
    fn new_standalone_mt(
        stream: &mut dyn OStream,
        header: &Header,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let mut state = Self::compute_mt(header, None, num_threads)?;

        // writeMagicNumberAndVersionField + header.writeTo are folded into
        // Header::write_to; version_field sets NON_IMAGE_FLAG from the deep
        // type automatically, so base_flags == 0 is correct.
        let preview_position = state.header.write_to(stream, 0)?;

        // C++: lineOffsetsPosition = writeLineOffsets(os, lineOffsets);
        let pos = stream.tellp()?;
        for &off in &state.line_offsets {
            xdr::write_u64(stream, off)?;
        }
        state.line_offsets_position = pos;
        state.preview_position = preview_position;
        Ok(state)
    }

    /// Multipart-part construction — port of `DeepScanLineOutputFile::
    /// DeepScanLineOutputFile(const OutputPartData*)`. The container
    /// ([`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile))
    /// already wrote the file prologue and this part's reserved offset
    /// table; this only computes the pure per-part state and records where
    /// that table lives (`chunk_offset_table_position`, into which each
    /// chunk's start offset is back-patched). Touches no stream at all —
    /// this is what lets `MultiPartOutputFile` build this state ONCE per
    /// part and persist it across `deep_scanline_output_part(n)` calls.
    /// `preview_position` stays `0` (see the field doc).
    /// `preview_position` stays 0 unless given via
    /// [`from_part_with_preview_mt`](Self::from_part_with_preview_mt) — the
    /// public `DeepScanLineOutputFile::from_part` constructor (this
    /// function's only OTHER caller besides `MultiPartOutputFile`) does not
    /// itself write any header, so it has no real value to supply.
    /// `num_threads` is the per-file override.
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

    /// Like [`from_part`](Self::from_part), but additionally records
    /// `preview_position`: C++
    /// `ImfMultiPartOutputFile.cpp:474-477` captures
    /// `parts[i]->previewPosition = headers[i].writeTo(..)` for every part
    /// while writing the header list, so `update_preview_image` works on a
    /// multipart deep-scanline part exactly as it does on a standalone file.
    /// [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile) calls
    /// this (not [`from_part`](Self::from_part)) with the value
    /// `Header::write_attributes_to` returned for this part.
    /// `num_threads` is the per-file override.
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

    /// The image header — port of `const Header& DeepScanLineOutputFile::header()`.
    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    /// The currently set frame buffer — port of
    /// `const DeepFrameBuffer& DeepScanLineOutputFile::frameBuffer() const`.
    /// Empty until [`set_frame_buffer`][Self::set_frame_buffer] has
    /// been called.
    pub(crate) fn frame_buffer(&self) -> &DeepFrameBuffer {
        &self.caller_frame_buffer
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk — port of `void updatePreviewImage(const PreviewRgba newPixels[])`
    /// (`ImfDeepScanLineOutputFile.cpp:891-918`). Same shape as
    /// `tiled::output::TiledOutputFileState::update_preview_image` /
    /// `deep::tiled_output::DeepTiledOutputFile::update_preview_image`.
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

    /// Next scanline to be written — port of
    /// `int DeepScanLineOutputFile::currentScanLine() const`.
    pub(crate) fn current_scan_line(&self) -> i32 {
        self.current_scan_line
    }

    /// Set the pixel source — port of `DeepScanLineOutputFile::setFrameBuffer`.
    ///
    /// `sample_counts` is the caller-owned flat per-pixel sample-count buffer
    /// addressed by the frame buffer's sample-count slice (byte strides taken
    /// from that slice); it replaces the C++ `int* sampleCountSliceBase`.
    /// `sample_data` is the caller-owned flat per-channel sample heap the deep
    /// slices' `base_offset` index into (the safe-Rust replacement for the C++
    /// `char**`; see the module docs).
    ///
    /// Performs the same compatibility checks as C++ (pixel type and
    /// x/y-sampling must match the header channel; the sample-count slice must
    /// be present) and builds the per-channel `OutSliceInfo` table, including
    /// the "channel present in the header but absent from the frame buffer →
    /// write zeroes" case.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when a slice's type or sampling is incompatible with
    ///   the header channel (matching the C++ `THROW`s).
    /// * [`ExrError::Arg`] when no sample-count slice was set (C++ "Invalid
    ///   base pointer, please set a proper sample count slice.").
    ///
    /// Writer pixel data is passed PER WRITE CALL: this stores only the LAYOUT
    /// (channel slices as offsets/strides/types); it does **not** take or
    /// retain the sample-count / sample-data BYTES. Those are passed to each
    /// [`write_pixels`][Self::write_pixels] call instead — no stored
    /// lifetime, no snapshot, and refilling a streaming buffer between calls
    /// is simply passing the current bytes to the next `write_pixels` call.
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
                if ch.x_sampling != slice.slice.x_sampling
                    || ch.y_sampling != slice.slice.y_sampling
                {
                    return Err(ExrError::Arg(format!(
                        "X and/or y subsampling factors of \"{name}\" channel of \
                         output file are not compatible with the frame buffer's \
                         subsampling factors."
                    )));
                }
            }
        }

        // C++: the sample-count slice base must be set (non-null). In the
        // safe-Rust model a slice with both strides zero is the equivalent of
        // a null base pointer (no buffer bytes are available here to check
        // emptiness against — that check now lives in `write_pixels`).
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

        // C++: build the slice table in header channel order.  The sample-count
        // slice is not in the header, so it is not added here.
        let mut slices: Vec<OutSliceInfo> = Vec::new();
        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                None => slices.push(OutSliceInfo {
                    type_: ch.type_,
                    base_offset: 0,
                    sample_stride: 0,
                    x_stride: 0,
                    y_stride: 0,
                    x_sampling: ch.x_sampling,
                    y_sampling: ch.y_sampling,
                    zero: true,
                }),
                Some(slice) => slices.push(OutSliceInfo {
                    type_: slice.slice.type_,
                    base_offset: slice.slice.base_offset,
                    sample_stride: slice.sample_stride,
                    x_stride: slice.slice.x_stride as i32,
                    y_stride: slice.slice.y_stride as i32,
                    x_sampling: slice.slice.x_sampling,
                    y_sampling: slice.slice.y_sampling,
                    zero: false,
                }),
            }
        }

        // Reset the per-row prefix cache — a fresh `set_frame_buffer`
        // call means a (possibly different) count-slice geometry, so any rows
        // cached under the old geometry must not be reused.
        self.sample_prefix =
            crate::deep::sample_table::RowPrefixCache::new(self.min_x, self.max_x, self.min_y);

        self.slices = slices;
        self.caller_frame_buffer = frame_buffer.clone();
        Ok(())
    }

    /// Validate that `sample_counts` covers every address `write_pixels` will
    /// read this call — a per-call bounds check, because the buffer is passed
    /// fresh to every `write_pixels` call rather than held by
    /// `set_frame_buffer`.
    ///
    /// With non-negative byte strides the extreme
    /// byte offsets are the `(min_x, min_y)` / `(max_x, max_y)` corners;
    /// `i128` keeps the check exact even for adversarial strides.
    fn validate_sample_counts(&self, sample_counts: &[u8]) -> Result<()> {
        if sample_counts.is_empty() {
            return Err(ExrError::Arg(
                "Invalid base pointer, please set a proper sample count slice.".into(),
            ));
        }
        let xs = self.sample_count_x_stride as i128;
        let ys = self.sample_count_y_stride as i128;
        let min_off =
            self.sample_count_base_offset as i128 + self.min_y as i128 * ys + self.min_x as i128 * xs;
        let max_off =
            self.sample_count_base_offset as i128 + self.max_y as i128 * ys + self.max_x as i128 * xs;
        if min_off < 0 || max_off + 4 > sample_counts.len() as i128 {
            return Err(ExrError::Arg(format!(
                "Deep sample-count buffer (len {}) does not cover the data-window \
                 addressing (byte range [{min_off}, {}]).",
                sample_counts.len(),
                max_off + 4
            )));
        }
        Ok(())
    }

    /// Read one pixel's sample count from the flat count buffer — port of the
    /// C++ inline `sampleCount(base, xStride, yStride, x, y)` (ImfMisc.h).
    ///
    /// Thin delegate to the shared [`sample_count_at`] so there is a single
    /// lookup implementation; a negative/out-of-range address is a hard error,
    /// not a silent zero (see [`sample_count_at`]).
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

    /// Number of samples in all scanlines *before* `y` (Σ over y' in
    /// `[minY, y)`, all x) — the channel-relative pixel offset at which line
    /// `y`'s samples begin in the flat per-channel heap.  Multiplying by a
    /// channel's `sample_stride` gives that channel's byte start for line `y`,
    /// keeping the heap globally contiguous per channel (see the module docs).
    ///
    /// O(1) amortized via the growable
    /// [`RowPrefixCache`](crate::deep::sample_table::RowPrefixCache) (each row
    /// summed at most once across the whole write), instead of re-summing
    /// every line from `minY` on every call (which made a whole-image write
    /// O(H²·W·channels) — `gather_scanline` calls this once per channel per
    /// line).
    fn samples_before_line(&mut self, sample_counts: &[u8], y: i32) -> Result<u64> {
        self.sample_prefix.before_line(
            sample_counts,
            self.sample_count_base_offset,
            self.sample_count_x_stride,
            self.sample_count_y_stride,
            y,
        )
    }

    /// Total sample count of scanline `y` (Σ over x in `[minX, maxX]`) — C++
    /// `_ofd->lineSampleCount[y - minY]`.
    fn line_sample_count(&self, sample_counts: &[u8], y: i32) -> Result<u64> {
        let mut total: u64 = 0;
        for x in self.min_x..=self.max_x {
            total += self.sample_count(sample_counts, x, y)? as u64;
        }
        Ok(total)
    }

    /// Write `num_scan_lines` scanlines — port of
    /// `DeepScanLineOutputFile::writePixels`.
    ///
    /// Because the deep-valid codecs all have `linesInBuffer == 1`, each
    /// scanline is its own chunk.  For each scanline the method builds the
    /// cumulative sample-count table (compressed by the *count* compressor),
    /// gathers every channel's variable samples into the data buffer
    /// (compressed by the *data* compressor), writes the chunk in the exact
    /// C++ field order (`y`, `packedCountSize`, `packedDataSize`,
    /// `unpackedDataSize`, count table, data) and back-patches the chunk offset
    /// into the reserved table slot.  Both `INCREASING_Y` and `DECREASING_Y`
    /// orders are honoured.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or if more scanlines are
    ///   written than the data window allows (matching the C++ throws).
    /// * Propagates gather / stream / codec errors.
    pub(crate) fn write_pixels(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        num_scan_lines: i32,
    ) -> Result<()> {
        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer …");
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data source.".into(),
            ));
        }
        // Pixel data is passed PER WRITE CALL: validate the
        // caller's buffer against the layout on every call (the buffer may
        // be a fresh refill each time), never a panic on a short one.
        self.validate_sample_counts(sample_counts)?;

        // Every deep scanline chunk is exactly ONE line
        // (`lines_in_buffer == 1` always for the deep-valid codecs), so —
        // unlike the flat scanline writer — there is no partial-block
        // carryover to worry about: as long as this whole call's lines fit
        // within what remains, they are all "fresh, complete" chunks that
        // can be gathered+encoded in parallel and then emitted strictly in
        // call order. Falls through to the exact original serial loop
        // (which naturally raises the same error at the same point) whenever
        // that precondition doesn't hold, so no existing error behaviour
        // changes.
        if num_scan_lines > 1
            && num_scan_lines <= self.missing_scan_lines
            && self.num_threads.is_parallel()
        {
            self.write_pixels_parallel(stream, sample_counts, sample_data, num_scan_lines)?;
            crate::probe::hit(crate::probe::Probe::DeepScanWriteParallel);
            return Ok(());
        }

        let step: i32 = if self.line_order == crate::attr::LineOrder::IncreasingY {
            1
        } else {
            -1
        };

        for _ in 0..num_scan_lines {
            // C++: if (missingScanLines <= 0) throw ArgExc("Tried to write …");
            if self.missing_scan_lines <= 0 {
                return Err(ExrError::Arg(
                    "Tried to write more scan lines than specified by the data window.".into(),
                ));
            }

            let y = self.current_scan_line;

            // --- Build the cumulative sample-count table for this scanline ---
            // C++ (LineBufferTask::execute): for each y, count resets to 0 and
            // accumulates across x; each running total is written as Xdr int.
            let width = (self.max_x - self.min_x + 1) as usize;
            let mut count_table = Vec::with_capacity(width * SIZE_INT);
            // Collect the row first so `sample_count`'s Result propagates
            // before it is handed to the count-encoder.
            let row: Vec<u32> = (self.min_x..=self.max_x)
                .map(|x| self.sample_count(sample_counts, x, y))
                .collect::<Result<_>>()?;
            encode_cumulative_row(row, &mut count_table)?;
            let raw_table_size = count_table.len();

            // --- Gather the sample data for this scanline ---
            let total_samples = self.line_sample_count(sample_counts, y)?;
            let mut data: Vec<u8> = Vec::new();
            self.gather_scanline(sample_counts, sample_data, y, total_samples, &mut data)?;
            let unpacked_data_size = data.len() as u64;

            // --- Compress the count table (its OWN compressor) ---
            // C++ `ImfDeepScanLineOutputFile.cpp:722-726`: if there is no
            // compressor, or the compressed table is `>= tableDataSize`, store
            // the raw table.  Same rule as the sample data below, expressed once
            // in `stored_chunk`.
            let packed_count: Vec<u8> = match self.count_compressor.as_mut() {
                Some(c) => {
                    let out = c
                        .compress(&count_table, self.min_y)
                        .map_err(crate::error::encode_err)?;
                    stored_chunk(out, &count_table).to_vec()
                }
                None => count_table.clone(),
            };
            debug_assert!(packed_count.len() <= raw_table_size);

            // --- Compress the sample data (the SEPARATE data compressor) ---
            // C++ `ImfDeepScanLineOutputFile.cpp:751`: `compSize < dataSize` →
            // use compressed; else keep raw (and, if NATIVE, convertToXdr — but
            // every ported codec reports Xdr, so the gathered bytes are already
            // Xdr and no conversion is needed).
            debug_assert_eq!(self.format, Format::Xdr);
            let packed_data: Vec<u8> = match self.compressor.as_mut() {
                Some(c) => {
                    let out = c.compress(&data, y).map_err(crate::error::encode_err)?;
                    stored_chunk(out, &data).to_vec()
                }
                None => data.clone(),
            };
            debug_assert!(packed_data.len() as u64 <= unpacked_data_size);

            self.write_chunk_now(stream, y, &packed_count, &packed_data, unpacked_data_size)?;

            // C++: currentScanLine += step; missingScanLines -= 1.
            self.current_scan_line += step;
            self.missing_scan_lines -= 1;
        }

        Ok(())
    }

    /// The parallel write body — every line in `[current_scan_line,
    /// current_scan_line + step*(num_scan_lines-1)]` gathers+encodes on the
    /// global pool (each line is independent: gather only reads
    /// `sample_counts`/`sample_data`, never touches `self` beyond the
    /// already-snapshotted `samples_before`), then EMITS strictly in call
    /// order via [`Self::write_chunk_now`] — on-disk bytes are therefore
    /// identical at every thread count even though gather+encode may finish
    /// out of order. Only reachable when every line in this call is
    /// guaranteed to fit (checked by the caller), so no error-path behaviour
    /// differs from the serial loop.
    fn write_pixels_parallel(
        &mut self,
        stream: &mut dyn OStream,
        sample_counts: &[u8],
        sample_data: &[u8],
        num_scan_lines: i32,
    ) -> Result<()> {
        let step: i32 = if self.line_order == crate::attr::LineOrder::IncreasingY {
            1
        } else {
            -1
        };
        let lines: Vec<i32> =
            (0..num_scan_lines).map(|i| self.current_scan_line + step * i).collect();

        // Serial pre-pass: extend the persisted `RowPrefixCache` in order and
        // snapshot each line's `samples_before` — see the deep scanline
        // READER's identical pre-pass (`read_pixels_parallel`) for why this
        // preserves the cache's amortized-linear guarantee.
        let mut before_line = Vec::with_capacity(lines.len());
        for &y in &lines {
            before_line.push(self.samples_before_line(sample_counts, y)?);
        }

        let (min_x, max_x) = (self.min_x, self.max_x);
        let (base, xs, ys) =
            (self.sample_count_base_offset, self.sample_count_x_stride, self.sample_count_y_stride);
        let compression = self.compression;
        let min_y = self.min_y;
        let max_sample_count_table_size = (max_x - min_x + 1) as usize * SIZE_INT;
        let count = self.num_threads;
        // Each line's footprint is known before anything is encoded: its count
        // table plus its samples, gathered and then encoded (~2x). The ring
        // holds C++'s `2 * numThreads` lines but never more bytes than
        // `Ring::deep` allows (`ImfDeepScanLineOutputFile.cpp:1232-1240,1355-1372`).
        let bytes_per_sample = self.slices.iter().try_fold(0usize, |acc, s| pixel_type_size(s.type_).map(|n| acc + n))?;
        let mut line_bytes = Vec::with_capacity(lines.len());
        for &y in &lines {
            let samples = self.line_sample_count(sample_counts, y)? as usize;
            line_bytes.push(2 * (max_sample_count_table_size + samples.saturating_mul(bytes_per_sample)));
        }

        // Two pools -- count-table
        // codecs and sample-data codecs are constructed with different
        // sizes (`max_sample_count_table_size` vs `0`), so they must not
        // share one free list. See `encode_deep_line`'s doc.
        let count_pool = crate::compressor::CodecPool::new();
        let data_pool = crate::compressor::CodecPool::new();
        type EncodedLine = (i32, Vec<u8>, Vec<u8>, u64);
        let (lines, before_line) = (&lines, &before_line);
        let (count_pool, data_pool) = (&count_pool, &data_pool);
        // The workers read the header/layout for the whole call while the emit
        // step mutates the offset table: disjoint field borrows, no clones.
        let (header, slices) = (&self.header, &self.slices);
        let mut sink = LineSink {
            line_offsets: &mut self.line_offsets,
            line_offsets_position: self.line_offsets_position,
            part_number: self.part_number,
            min_y,
            lines_in_buffer: self.lines_in_buffer,
        };

        crate::threading::ordered_pipeline(
            count,
            lines.len(),
            crate::threading::Ring::deep(count),
            |k| line_bytes[k],
            |k| -> Result<EncodedLine> {
                let (y, samples_before) = (lines[k], before_line[k]);
                let (count_table, data) = gather_deep_line(
                    slices,
                    min_x,
                    max_x,
                    sample_counts,
                    base,
                    xs,
                    ys,
                    sample_data,
                    y,
                    samples_before,
                )?;
                let (packed_count, packed_data) = encode_deep_line(
                    compression,
                    header,
                    max_sample_count_table_size,
                    y,
                    min_y,
                    &count_table,
                    &data,
                    count_pool,
                    data_pool,
                )?;
                Ok((y, packed_count, packed_data, data.len() as u64))
            },
            // Emit STRICTLY in `lines` order (== call/file order, the hard
            // byte-identity invariant) while later lines still encode.
            |_, (y, packed_count, packed_data, unpacked_data_size)| {
                sink.write(stream, y, &packed_count, &packed_data, unpacked_data_size)
            },
        )?;

        self.current_scan_line += step * num_scan_lines;
        self.missing_scan_lines -= num_scan_lines;
        Ok(())
    }

    /// Write one already-gathered-and-compressed deep scanline chunk to the
    /// stream NOW and back-patch its offset-table slot -- factored out
    /// of [`Self::write_pixels`]'s tail so [`Self::copy_pixels`] can reuse the
    /// exact same low-level chunk write for a RAW (never decompressed)
    /// count-table + data pair, mirroring the flat/tiled writers'
    /// `write_tile_chunk_now`-style split.
    fn write_chunk_now(
        &mut self,
        stream: &mut dyn OStream,
        y: i32,
        packed_count: &[u8],
        packed_data: &[u8],
        unpacked_data_size: u64,
    ) -> Result<()> {
        self.sink().write(stream, y, packed_count, packed_data, unpacked_data_size)
    }

    /// The chunk-emit state, borrowed field-by-field (see [`LineSink`]).
    fn sink(&mut self) -> LineSink<'_> {
        LineSink {
            line_offsets: &mut self.line_offsets,
            line_offsets_position: self.line_offsets_position,
            part_number: self.part_number,
            min_y: self.min_y,
            lines_in_buffer: self.lines_in_buffer,
        }
    }

    /// Quick pixel copy: transcode every scanline chunk of `input` straight
    /// into this file, RAW (no decompress/recompress) -- port of the 2
    /// C++ `DeepScanLineOutputFile::copyPixels` overloads
    /// (`ImfDeepScanLineOutputFile.cpp:1416-1571`).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if the headers' data window, line order,
    ///   compression, or channel list differ (C++'s exact compatibility
    ///   checks — each with its own message in C++; folded into one
    ///   [`ExrError::Arg`] here, message says which check failed).
    /// * [`ExrError::Logic`] if this file already has pixel data written
    ///   (C++: "already contains pixel data").
    /// * Propagates `input`'s `raw_pixel_data` errors.
    pub(crate) fn copy_pixels<T: DeepScanLineCopySource>(
        &mut self,
        stream: &mut dyn OStream,
        input: &mut T,
    ) -> Result<()> {
        let hdr = &self.header;
        let in_hdr = input.header();

        if !in_hdr.has_type() || in_hdr.type_()? != DEEP_SCANLINE {
            return Err(ExrError::Arg(
                "Cannot copy pixels: the input needs to be a deep scanline image.".into(),
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

        // C++: missingScanLines != dataWindow height -> LogicExc.
        let dw = hdr.data_window()?;
        if self.missing_scan_lines != dw.max.y - dw.min.y + 1 {
            return Err(ExrError::Logic(
                "Quick pixel copy failed: this file already contains pixel data.".into(),
            ));
        }

        let step: i32 = if self.line_order == crate::attr::LineOrder::IncreasingY {
            1
        } else {
            -1
        };

        while self.missing_scan_lines > 0 {
            let (y, packed_count, packed_data, unpacked) =
                input.raw_pixel_data(self.current_scan_line)?;
            self.write_chunk_now(stream, y, &packed_count, &packed_data, unpacked)?;
            self.current_scan_line += step;
            self.missing_scan_lines -= 1;
        }
        Ok(())
    }

    /// Gather one scanline's samples for every channel into `data` (Xdr LE) —
    /// port of the per-slice loop in `LineBufferTask::execute` /
    /// `copyFromDeepFrameBuffer`.
    ///
    /// Channels skipped by y-subsampling (`modp(y, ySampling) != 0`) contribute
    /// nothing.  Absent channels (`zero`) emit `total_samples` zero-valued
    /// samples of the channel type (C++ `fillChannelWithZeroes` with the line's
    /// total sample count).  Present channels emit, per pixel x, that pixel's
    /// `sampleCount(x, y)` samples read from the flat heap at
    /// `base_offset + pixelStart + i*sampleStride`, where `pixelStart` is the
    /// running byte offset of preceding pixels' samples for that channel.
    fn gather_scanline(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        y: i32,
        total_samples: u64,
        data: &mut Vec<u8>,
    ) -> Result<()> {
        // Clone the (small) slice table so the loop body can call
        // `self.samples_before_line` (which needs `&mut self` for the
        // prefix cache) without holding a borrow of `self.slices` alive.
        let slices = self.slices.clone();
        for slice in &slices {
            // C++: if (modp(y, ySampling) != 0) continue;
            if modp(y, slice.y_sampling) != 0 {
                continue;
            }

            let sample_size = pixel_type_size(slice.type_)?;

            if slice.zero {
                // C++ fillChannelWithZeroes for the line's total sample count.
                let n = total_samples as usize * sample_size;
                data.resize(data.len() + n, 0u8);
                continue;
            }

            // Present channel: walk pixels, reading each pixel's samples from
            // the flat heap.  `pixel_start` advances by count*sampleStride per
            // pixel — the contiguous-heap replacement for the C++ `char**`
            // per-pixel pointer dereference.  Per the module's flat-heap model
            // the per-pixel array start is the running sum of count*sampleStride
            // over *all preceding pixels of that channel* (raster order across
            // the whole image), so this line begins at the cumulative offset of
            // every pixel in the lines before `y` — NOT 0 (which would alias all
            // lines onto the same heap region).
            let mut pixel_start: i64 = self.samples_before_line(sample_counts, y)? as i64
                * slice.sample_stride as i64;
            for x in self.min_x..=self.max_x {
                let cnt = self.sample_count(sample_counts, x, y)?;
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
                    // Samples in the flat heap are already in Xdr LE byte order
                    // (same adaptation as the flat writer); copy verbatim.
                    data.extend_from_slice(src);
                }
                pixel_start += cnt as i64 * slice.sample_stride as i64;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure per-line pipeline functions (deep scanline writer)
// ---------------------------------------------------------------------------

/// Gather one scanline's cumulative sample-count table and sample data —
/// pure function, no `self`, no stream access. Mirrors
/// [`DeepScanLineOutputFileState::gather_scanline`] plus the count-table
/// build the serial `write_pixels` loop does inline, parameterised so the
/// serial and parallel paths are a single source of truth.
#[allow(clippy::too_many_arguments)]
fn gather_deep_line(
    slices: &[OutSliceInfo],
    min_x: i32,
    max_x: i32,
    sample_counts: &[u8],
    sample_count_base_offset: i64,
    sample_count_x_stride: i32,
    sample_count_y_stride: i32,
    sample_data: &[u8],
    y: i32,
    samples_before: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let width = (max_x - min_x + 1) as usize;
    let row: Vec<u32> = (min_x..=max_x)
        .map(|x| {
            sample_count_at(
                sample_counts,
                sample_count_base_offset,
                sample_count_x_stride,
                sample_count_y_stride,
                x,
                y,
            )
        })
        .collect::<Result<_>>()?;
    let mut count_table = Vec::with_capacity(width * SIZE_INT);
    let total_samples: u64 = row.iter().map(|&c| c as u64).sum();
    encode_cumulative_row(row.clone(), &mut count_table)?;

    let mut data: Vec<u8> = Vec::new();
    for slice in slices {
        if modp(y, slice.y_sampling) != 0 {
            continue;
        }
        let sample_size = pixel_type_size(slice.type_)?;
        if slice.zero {
            let n = total_samples as usize * sample_size;
            data.resize(data.len() + n, 0u8);
            continue;
        }
        let mut pixel_start: i64 = samples_before as i64 * slice.sample_stride as i64;
        for (i, _x) in (min_x..=max_x).enumerate() {
            let cnt = row[i];
            for s in 0..cnt as i64 {
                let off = slice.base_offset + pixel_start + s * slice.sample_stride as i64;
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
            pixel_start += cnt as i64 * slice.sample_stride as i64;
        }
    }
    Ok((count_table, data))
}

/// Compress one gathered scanline's count table (with its OWN compressor)
/// and sample data (with the SEPARATE data compressor) — owned-`Vec` form of
/// the `stored_chunk` decisions the serial `write_pixels` loop makes
/// inline, for use from a rayon worker. The
/// count table is always compressed positioned at `min_y` (matching C++ /
/// the serial path exactly — NOT the line's own `y`); the sample data is
/// compressed positioned at the line's own `y`.
/// Codecs come from `count_pool`/`data_pool` (checkout/checkin around each
/// compress call) rather than a per-call `new_compressor`, which would build
/// TWO fresh codecs per line -- see `crate::compressor::CodecPool`'s own doc.
#[allow(clippy::too_many_arguments)]
fn encode_deep_line(
    compression: crate::attr::Compression,
    header: &Header,
    max_sample_count_table_size: usize,
    y: i32,
    min_y: i32,
    count_table: &[u8],
    data: &[u8],
    count_pool: &crate::compressor::CodecPool,
    data_pool: &crate::compressor::CodecPool,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut count_compressor =
        count_pool.lease(|| new_compressor(compression, max_sample_count_table_size, header))?;
    let packed_count: Vec<u8> = match count_compressor.as_mut() {
        Some(c) => {
            let out = c.compress(count_table, min_y).map_err(crate::error::encode_err)?;
            stored_chunk(out, count_table).to_vec()
        }
        None => count_table.to_vec(),
    };
    drop(count_compressor);

    let mut data_compressor = data_pool.lease(|| new_compressor(compression, 0, header))?;
    let packed_data: Vec<u8> = match data_compressor.as_mut() {
        Some(c) => {
            let out = c.compress(data, y).map_err(crate::error::encode_err)?;
            stored_chunk(out, data).to_vec()
        }
        None => data.to_vec(),
    };
    drop(data_compressor);
    Ok((packed_count, packed_data))
}

/// A **borrowing** view over one multipart part's persisted deep scanline
/// writer state — same shape as `crate::scanline::output::
/// OutputPartWriter` / `crate::tiled::output::TiledOutputPartWriter`: instead
/// of [`MultiPartOutputFile::deep_scanline_output_part`](crate::multipart::
/// MultiPartOutputFile::deep_scanline_output_part) building a brand-new
/// [`DeepScanLineOutputFile`] (and its state) on every call, the container
/// persists one [`DeepScanLineOutputFileState`] per deep scanline part across
/// calls and hands out this thin view — `{ &mut dyn OStream, &mut
/// DeepScanLineOutputFileState }` — so interleaved writes across parts resume
/// exactly where they left off.
///
/// Exposes the same public surface as [`DeepScanLineOutputFile`] (`header`,
/// `frame_buffer`, `update_preview_image`, `current_scan_line`,
/// `set_frame_buffer`, `write_pixels`) so callers of
/// `deep_scanline_output_part` use it like a standalone writer.
pub struct DeepScanLineOutputPartWriter<'p> {
    stream: &'p mut dyn OStream,
    state: &'p mut DeepScanLineOutputFileState,
}

impl<'p> DeepScanLineOutputPartWriter<'p> {
    /// Build a view over an existing (already-constructed, possibly
    /// previously-written-to) part state and the container's stream.
    pub(crate) fn new(stream: &'p mut dyn OStream, state: &'p mut DeepScanLineOutputFileState) -> Self {
        Self { stream, state }
    }

    /// The image header — port of `const Header& DeepScanLineOutputFile::header()`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The currently set frame buffer — port of
    /// `const DeepFrameBuffer& DeepScanLineOutputFile::frameBuffer() const`.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// Replace the header's preview-image pixels and rewrite them in place on
    /// disk. See [`DeepScanLineOutputFileState::update_preview_image`].
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.state.update_preview_image(self.stream, new_pixels)
    }

    /// Next scanline to be written — port of
    /// `int DeepScanLineOutputFile::currentScanLine() const`.
    pub fn current_scan_line(&self) -> i32 {
        self.state.current_scan_line()
    }

    /// Set the pixel LAYOUT. See [`DeepScanLineOutputFileState::set_frame_buffer`].
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Write `num_scan_lines` scanlines. See
    /// [`DeepScanLineOutputFileState::write_pixels`] for the full algorithm doc.
    pub fn write_pixels(
        &mut self,
        sample_counts: &[u8],
        sample_data: &[u8],
        num_scan_lines: i32,
    ) -> Result<()> {
        self.state.write_pixels(self.stream, sample_counts, sample_data, num_scan_lines)
    }

    /// Quick pixel copy: transcode every scanline chunk of `input` straight
    /// into this file, raw (no decompress/recompress). See
    /// [`DeepScanLineOutputFileState::copy_pixels`] for the full doc.
    pub fn copy_pixels<T: DeepScanLineCopySource>(&mut self, input: &mut T) -> Result<()> {
        self.state.copy_pixels(self.stream, input)
    }
}

/// Xdr `int` size in bytes — the per-pixel cumulative-count table entry width
/// (C++ `sizeof(int)` written via `Xdr::write<int>`).
const SIZE_INT: usize = 4;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::deep::frame_buffer::DeepSlice;
    use crate::frame_buffer::Slice;
    use crate::io::{IStream, MemIStream, MemOStream};
    use imath_rs::{Half, V2f};

    /// Build a deep header: `width`x`height`, one HALF channel "A", deep type,
    /// the given compression and INCREASING_Y order.
    fn make_deep_header(width: i32, height: i32, comp: Compression) -> Header {
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
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.set_type(DEEP_SCANLINE).unwrap();
        h
    }

    /// Build a 4x3 deep image with per-pixel sample counts `0..=11` (raster
    /// order), one HALF channel whose sample value is `pixel_index*10 + i`.
    ///
    /// Returns `(sample_counts_bytes, sample_data_bytes, frame_buffer,
    /// counts_vec)`.
    fn make_deep_pixels(w: i32, h: i32) -> (Vec<u8>, Vec<u8>, DeepFrameBuffer, Vec<u32>) {
        let n = (w * h) as usize;
        // Per-pixel counts: 0,1,2,3,... in raster order.
        let counts: Vec<u32> = (0..n as u32).collect();

        // Flat sample-count buffer: u32 LE per pixel, x_stride=4, y_stride=4*w.
        let mut sc = Vec::with_capacity(n * 4);
        for &c in &counts {
            sc.extend_from_slice(&c.to_le_bytes());
        }

        // Flat sample heap for channel "A": each pixel contributes `count`
        // HALF samples (2 bytes each), laid out contiguously in raster order.
        // Sample value = pixel_index*10 + i, stored as Xdr (LE) half bits.
        let mut data = Vec::new();
        for (pix, &c) in counts.iter().enumerate() {
            for i in 0..c {
                let v = (pix as f32) * 10.0 + i as f32;
                let bits = Half::from(v).bits();
                data.extend_from_slice(&bits.to_le_bytes());
            }
        }

        // Frame buffer: sample-count slice (UINT) + deep slice "A".
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
                0,                       // base_offset into the flat heap
                8,                       // x_stride (unused in flat model)
                8 * w as usize,          // y_stride (unused in flat model)
                2,                       // sample_stride = sizeof(HALF)
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

    /// Re-parse the written file and assert the deep chunk structure is
    /// self-consistent: deep version flag, fully-patched offset table, each
    /// chunk's `y` + sizes, the decoded cumulative count table, and total size.
    fn verify_deep_file(bytes: &[u8], w: i32, h: i32, comp: Compression, counts: &[u32]) {
        // 1. Re-read header + version word; confirm the deep (non-image) flag.
        let mut is = MemIStream::new(bytes.to_vec());
        let (hdr, version_word) = Header::read_from(&mut is).unwrap();
        assert!(
            crate::container::version::is_non_image(version_word as u32),
            "deep NON_IMAGE_FLAG not set in version word"
        );
        assert_eq!(hdr.type_().unwrap(), DEEP_SCANLINE);
        assert_eq!(hdr.compression().unwrap(), comp);

        // Header byte length = current read cursor (offset table starts here).
        let header_len = is.tellg().unwrap() as usize;
        let chunk_count = h as usize; // linesInBuffer == 1 for deep codecs.
        let table_start = header_len;
        let table_end = table_start + chunk_count * 8;

        // 2. Offset table: read each u64; none may be zero, all must point past
        //    the table and into the file.
        let mut offsets = Vec::with_capacity(chunk_count);
        for k in 0..chunk_count {
            let o = u64::from_le_bytes(
                bytes[table_start + k * 8..table_start + k * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            assert_ne!(o, 0, "offset table slot {k} never back-patched");
            assert!(
                o as usize >= table_end && (o as usize) < bytes.len(),
                "offset {o} out of range"
            );
            offsets.push(o);
        }

        // 3. Walk each chunk; verify y + size fields + decoded count table.
        for (k, &off) in offsets.iter().enumerate() {
            let mut p = off as usize;
            let y = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            assert_eq!(y, k as i32, "chunk {k} has wrong y");

            let packed_count = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            let packed_data = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;
            let _unpacked = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
            p += 8;

            // For NONE the count table is stored raw → decode cumulative ints.
            if comp == Compression::NoCompression {
                let table = &bytes[p..p + packed_count];
                assert_eq!(packed_count, w as usize * 4, "raw count table size");
                let mut expect_run: i32 = 0;
                for x in 0..w as usize {
                    let pix = y as usize * w as usize + x;
                    expect_run += counts[pix] as i32;
                    let got = i32::from_le_bytes(
                        table[x * 4..x * 4 + 4].try_into().unwrap(),
                    );
                    assert_eq!(got, expect_run, "cumulative count mismatch at ({x},{y})");
                }
            }

            p += packed_count + packed_data;
            // The next chunk (or EOF) must start exactly here.
            let expected_next = if k + 1 < offsets.len() {
                offsets[k + 1] as usize
            } else {
                bytes.len()
            };
            assert_eq!(p, expected_next, "chunk {k} size inconsistent");
        }
    }

    /// Write the SAME multi-line deep image with `num_threads`
    /// in {0, 1, 2, 3, 8, 17} and assert the produced FILE BYTES are
    /// identical at every thread count — the deep analogue of the flat
    /// writers' `parallel_write_*_byte_identical_across_thread_counts` tests.
    /// `make_deep_pixels` gives a non-uniform per-pixel count pattern
    /// (0,1,2,3,... raster order, including a zero-count pixel), which
    /// exercises the `samples_before` addressing across chunk boundaries.
    #[test]
    fn parallel_write_is_byte_identical_across_thread_counts() {
        let (w, h) = (9, 20);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let header = make_deep_header(w, h, Compression::NoCompression);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::DeepScanWriteParallel);
                let mut df =
                    DeepScanLineOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                df.write_pixels(&sc, &data, h).unwrap();
            }
            let bytes = out.into_data();
            verify_deep_file(&bytes, w, h, Compression::NoCompression, &counts);

            crate::probe::assert_engaged(crate::probe::Probe::DeepScanWriteParallel, n);
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

    /// Same invariant with ZIPS (a real codec exercising the per-worker
    /// fresh-compressor requirement for both the count-table and
    /// sample-data compressors), and irregular multi-call strips (exercises
    /// the "only parallelize when the whole call fits" gate falling back to
    /// the serial per-line loop for calls that don't fit cleanly).
    #[test]
    fn parallel_write_zips_irregular_strips_byte_identical_across_thread_counts() {
        let (w, h) = (7, 24);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        let strips = [1i32, 5, 10, 3, 5]; // sums to 24, deliberately irregular

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 3, 9] {
            let header = make_deep_header(w, h, Compression::Zips);
            let mut out = MemOStream::new();
            {
                crate::probe::reset(crate::probe::Probe::DeepScanWriteParallel);
                let mut df =
                    DeepScanLineOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                for &s in &strips {
                    df.write_pixels(&sc, &data, s).unwrap();
                }
            }
            let bytes = out.into_data();

            crate::probe::assert_engaged(crate::probe::Probe::DeepScanWriteParallel, n);
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
    fn deep_write_uncompressed_structure() {
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(df.current_scan_line(), 0);
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
            assert_eq!(df.current_scan_line(), h);
        }
        let bytes = out.into_data();
        verify_deep_file(&bytes, w, h, Compression::NoCompression, &counts);
    }

    /// `update_preview_image` (`DeepScanLineOutputFile` has no
    /// `breakScanLine` in C++ — only the tiled writer has `breakTile`; the
    /// deep scanline writer's own missing-API list is just this one method).
    #[test]
    fn update_preview_image_replaces_preview() {
        use crate::attr::{PreviewImage, PreviewImageAttribute, PreviewRgba};

        let (w, h) = (4, 3);
        let mut header = make_deep_header(w, h, Compression::NoCompression);
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

        let bytes = {
            let mut out = MemOStream::new();
            {
                let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                df.write_pixels(&sc, &data, h).unwrap();

                df.update_preview_image(&[
                    PreviewRgba::new(9, 9, 9, 9),
                    PreviewRgba::new(9, 9, 9, 9),
                ])
                .unwrap();
                // Wrong pixel count must error, not silently truncate/overrun.
                assert!(df.update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)]).is_err());
            }
            out.into_data()
        };

        // The header's preview attribute on disk must reflect the update.
        let mut header_is = MemIStream::new(bytes);
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

        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_pixels(&sc, &data, h).unwrap();

        match df.update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)]) {
            Err(ExrError::Logic(_)) => {}
            Err(other) => panic!("expected Logic error, got {other:?}"),
            Ok(_) => panic!("expected an error for a file with no preview image"),
        }
    }

    #[test]
    fn set_frame_buffer_rejects_short_count_buffer() {
        // A non-empty but too-short sample-count buffer must be rejected
        // instead of silently reading out-of-range pixels as zero
        // samples. Pixel data is passed PER WRITE CALL: this
        // bound is now checked in `write_pixels` (the buffer isn't available
        // at `set_frame_buffer` time anymore).
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        let short = &sc[..sc.len() / 2]; // non-empty, but cannot cover the window
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        assert!(df.write_pixels(short, &data, h).is_err());
    }

    #[test]
    fn set_frame_buffer_rejects_oversized_stride() {
        // A sample-count stride above i32::MAX would wrap when narrowed to i32;
        // reject it instead of mis-addressing the buffer.
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, _fb, _counts) = make_deep_pixels(w, h);
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: i32::MAX as usize + 1,
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
            DeepSlice::new(PixelType::Half, 0, 8, 8 * w as usize, 2, 1, 1, 0.0, false, false).unwrap(),
        )
        .unwrap();
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        assert!(df.set_frame_buffer(&fb).is_err());
        let _ = (&sc, &data); // unused now that the check is layout-only
    }

    #[test]
    fn deep_write_zips_structure() {
        // ZIPS is deep-valid (single-line ZIP); the count table + data are each
        // compressed by their own compressor instance.
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::Zips);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
        }
        let bytes = out.into_data();
        verify_deep_file(&bytes, w, h, Compression::Zips, &counts);
    }

    /// `copy_pixels`: transcode a source deep scanline file into a
    /// fresh writer with a matching header, byte-for-byte, without ever
    /// decompressing. ZIPS (not NONE) so a byte-identical result actually
    /// proves `copy_pixels` never decompresses+recompresses.
    #[test]
    fn copy_pixels_byte_identical() {
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::Zips);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut src_out = MemOStream::new();
        {
            let mut src = DeepScanLineOutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_pixels(&sc, &data, h).unwrap();
        }
        let src_bytes = src_out.into_data();

        let mut src_is = MemIStream::new(src_bytes.clone());
        let mut src_in = DeepScanLineInputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        {
            let mut dst = DeepScanLineOutputFile::new(&mut dst_out, &header).unwrap();
            dst.copy_pixels(&mut src_in).unwrap();
        }
        let dst_bytes = dst_out.into_data();

        assert_eq!(
            src_bytes, dst_bytes,
            "copy_pixels must reproduce the source file byte-for-byte"
        );
    }

    /// `copy_pixels` must reject copying into a file that already has
    /// pixel data (C++ LogicExc "already contains pixel data").
    #[test]
    fn copy_pixels_rejects_when_destination_already_has_pixel_data() {
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut src_out = MemOStream::new();
        {
            let mut src = DeepScanLineOutputFile::new(&mut src_out, &header).unwrap();
            src.set_frame_buffer(&fb).unwrap();
            src.write_pixels(&sc, &data, h).unwrap();
        }
        let mut src_is = MemIStream::new(src_out.into_data());
        let mut src_in = DeepScanLineInputFile::new(&mut src_is).unwrap();

        let mut dst_out = MemOStream::new();
        let mut dst = DeepScanLineOutputFile::new(&mut dst_out, &header).unwrap();
        dst.set_frame_buffer(&fb).unwrap();
        dst.write_pixels(&sc, &data, 1).unwrap();

        let err = dst.copy_pixels(&mut src_in).unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)));
    }

    #[test]
    fn deep_write_rle_structure() {
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::Rle);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
        }
        let bytes = out.into_data();
        verify_deep_file(&bytes, w, h, Compression::Rle, &counts);
    }

    #[test]
    fn deep_data_size_matches_total_samples() {
        // unpackedDataSize of each chunk must equal (Σ samples on that line) *
        // sizeof(HALF), proving the gather walked the deep addressing right.
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes.clone());
        let _ = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;

        for y in 0..h as usize {
            let off = u64::from_le_bytes(
                bytes[header_len + y * 8..header_len + y * 8 + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            // unpackedDataSize is the 4th field: y(4) + 3*u64; it sits at off+20.
            let unpacked = u64::from_le_bytes(bytes[off + 20..off + 28].try_into().unwrap());
            let line_samples: u32 = (0..w as usize)
                .map(|x| counts[y * w as usize + x])
                .sum();
            assert_eq!(unpacked, line_samples as u64 * 2, "line {y} data size");
        }
    }

    #[test]
    fn deep_decreasing_y_offset_table() {
        // DECREASING_Y writes line (h-1) first; the offset table is still
        // indexed by (y - minY), so entry[h-1] is the smallest file offset.
        let (w, h) = (4, 3);
        let mut header = make_deep_header(w, h, Compression::NoCompression);
        *header.line_order_mut().unwrap() = LineOrder::DecreasingY;
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            assert_eq!(df.current_scan_line(), h - 1); // starts at max.y
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
            assert_eq!(df.current_scan_line(), -1);
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes.clone());
        let _ = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;

        let read_off = |k: usize| -> u64 {
            u64::from_le_bytes(
                bytes[header_len + k * 8..header_len + k * 8 + 8]
                    .try_into()
                    .unwrap(),
            )
        };
        // Line h-1 written first → smallest offset; line 0 written last.
        assert!(read_off(h as usize - 1) < read_off(0));
        // Verify the chunk at entry[h-1] really carries y == h-1.
        let off = read_off(h as usize - 1) as usize;
        let y = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        assert_eq!(y, h - 1);
    }

    #[test]
    fn no_frame_buffer_errors() {
        let header = make_deep_header(4, 3, Compression::NoCompression);
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        let err = df.write_pixels(&[], &[], 1).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn missing_sample_count_slice_errors() {
        // Pixel data is passed PER WRITE CALL: an empty
        // sample-count buffer is no longer checkable at `set_frame_buffer`
        // (no buffer is passed there anymore) — it is now `write_pixels`'s
        // `validate_sample_counts` that treats an empty buffer as a null
        // base pointer, C++-style.
        let header = make_deep_header(4, 3, Compression::NoCompression);
        let (_sc, data, fb, _counts) = make_deep_pixels(4, 3);
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let err = df.write_pixels(&[], &data, 1).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn writing_too_many_lines_errors() {
        let (w, h) = (4, 3);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let err = df.write_pixels(&sc, &data, h + 1).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn absent_channel_filled_with_zeroes() {
        // Header has A and Z; frame buffer omits Z → Z must be zero-filled for
        // the line's total sample count.
        let (w, h) = (3, 2);
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
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        header.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        header.set_type(DEEP_SCANLINE).unwrap();

        let (sc, data, fb, counts) = make_deep_pixels(w, h); // fb has only "A"

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes.clone());
        let _ = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;

        // Channel order in the header is A (HALF, 2B) then Z (FLOAT, 4B).
        // unpackedDataSize per line = lineSamples*(2 + 4).
        for y in 0..h as usize {
            let off = u64::from_le_bytes(
                bytes[header_len + y * 8..header_len + y * 8 + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let unpacked = u64::from_le_bytes(bytes[off + 20..off + 28].try_into().unwrap());
            let line_samples: u32 = (0..w as usize).map(|x| counts[y * w as usize + x]).sum();
            assert_eq!(unpacked, line_samples as u64 * (2 + 4), "line {y} A+Z size");

            // The Z portion (last lineSamples*4 bytes of the raw data) is zero.
            let data_start = off + 28 + w as usize * 4; // skip raw count table
            let a_bytes = line_samples as usize * 2;
            let z_bytes = line_samples as usize * 4;
            let z_slice = &bytes[data_start + a_bytes..data_start + a_bytes + z_bytes];
            assert!(z_slice.iter().all(|&b| b == 0), "absent Z channel not zero-filled");
        }
    }

    /// Pixel data is passed PER WRITE CALL: `write_pixels`
    /// takes the count/sample bytes directly (no stored buffer at all), so
    /// two calls with two DIFFERENT `sample_data` buffers (simulating a
    /// caller refilling one streaming buffer between calls) must each write
    /// exactly the bytes they were given — this proves the SECOND half of
    /// the image reflects the SECOND buffer's data, not the first (which
    /// would indicate a stale-snapshot bug from some hidden internal cache).
    #[test]
    fn set_frame_buffer_refill_between_writes_uses_latest_data() {
        let (w, h) = (4, 4);
        let header = make_deep_header(w, h, Compression::NoCompression);
        let (sc, data_v1, fb, counts) = make_deep_pixels(w, h);
        // Same counts (required: the flat-heap cumulative addressing depends
        // on counts, not values), different sample VALUES (+1000 per HALF
        // sample) — simulates the caller refilling its heap in place.
        let data_v2: Vec<u8> = data_v1
            .as_chunks::<2>().0.iter()
            .flat_map(|b| {
                let bits = u16::from_le_bytes(*b);
                let v = f32::from(Half::from_bits(bits)) + 1000.0;
                Half::from(v).bits().to_le_bytes()
            })
            .collect();

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            // Pixel data is passed PER WRITE CALL: ONE
            // `set_frame_buffer` (layout only), then each `write_pixels`
            // call gets the CURRENT bytes directly — no refill call needed.
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data_v1, h / 2).unwrap(); // rows 0..h/2, v1 data
            df.write_pixels(&sc, &data_v2, h - h / 2).unwrap(); // rows h/2..h, v2 data
        }
        let bytes = out.into_data();

        // Read back and verify: row 0 (from v1) has the ORIGINAL values; the
        // last row (from v2) has the +1000 values.
        let mut is = MemIStream::new(bytes);
        let mut inf = crate::deep::DeepScanLineInputFile::new(&mut is).unwrap();
        let mut rfb = DeepFrameBuffer::new();
        rfb.insert_sample_count_slice(Slice {
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
        rfb.insert(
            "A",
            DeepSlice::new(PixelType::Half, 0, 8, 8 * w as usize, 2, 1, 1, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();
        inf.set_frame_buffer(&rfb).unwrap();
        let mut counts_buf = vec![0u8; sc.len()];
        inf.read_pixel_sample_counts(0, h - 1, &mut counts_buf).unwrap();
        let total: u32 = counts.iter().sum();
        let mut heap = vec![0u8; total as usize * 2];
        inf.read_pixels(0, h - 1, &counts_buf, &mut heap).unwrap();

        // Row 0's first sample (pixel 0 has count 0 -> pixel 1, the first
        // pixel with a sample) must equal v1's value; the LAST sample
        // (from the last row, count>0) must equal v2's (+1000) value.
        let first_v1 = f32::from(Half::from(1.0f32 * 10.0)); // pixel index 1, sample 0
        let got_first = f32::from(Half::from_bits(u16::from_le_bytes(heap[0..2].try_into().unwrap())));
        assert_eq!(got_first, first_v1, "first sample should still be v1's data");

        let last_off = (total as usize - 1) * 2;
        let last_pix = counts.len() - 1; // last pixel, last sample index = count-1
        let last_sample_idx = counts[last_pix] - 1;
        let expected_last = (last_pix as f32) * 10.0 + last_sample_idx as f32 + 1000.0;
        let got_last = f32::from(Half::from_bits(
            u16::from_le_bytes(heap[last_off..last_off + 2].try_into().unwrap()),
        ));
        assert_eq!(got_last, expected_last, "last sample should be the refilled v2 data");
    }

    /// The deep scanline ring really overlaps: lines are emitted while later
    /// lines are still encoding. Mutation: emit only after the window is
    /// produced -> this fails.
    #[test]
    fn parallel_write_emits_while_later_lines_encode() {
        let (w, h) = (32, 64);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        for &n in &[2i32, 8] {
            assert!(
                crate::probe::saw_overlap(|| {
                    let header = make_deep_header(w, h, Compression::Zips);
                    let mut out = MemOStream::new();
                    let mut df =
                        DeepScanLineOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                    df.set_frame_buffer(&fb).unwrap();
                    df.write_pixels(&sc, &data, h).unwrap();
                }),
                "num_threads={n}: no line was emitted while another was still encoding"
            );
        }
    }

    /// A deep image with large per-line sample data keeps the outstanding
    /// bytes within `Ring::deep`'s cap (here shrunk so the test sees it) - a
    /// ring bounded by item count alone would hold `2 * numThreads` of the
    /// biggest lines. Mutation: ignore the byte cap -> this fails.
    #[test]
    fn parallel_write_bounds_in_flight_bytes() {
        let (w, h) = (16, 32);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        let cap = 40_000usize;
        crate::threading::set_deep_ring_bytes(Some(cap));
        crate::threading::take_peak_bytes();
        let header = make_deep_header(w, h, Compression::Zips);
        let mut out = MemOStream::new();
        let mut df = DeepScanLineOutputFile::with_num_threads(&mut out, &header, Some(4)).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_pixels(&sc, &data, h).unwrap();
        drop(df);
        crate::threading::set_deep_ring_bytes(None);
        let peak = crate::threading::take_peak_bytes();
        assert!(peak > 0 && peak <= cap, "peak outstanding bytes {peak} vs cap {cap}");
    }

    /// Writer-side codec reuse for the deep scanline ring: two pools (count
    /// table + sample data), at most one codec per worker in each, over 64
    /// lines. Mutation: drop the lease's return -> this fails.
    #[test]
    fn parallel_write_reuses_codecs() {
        use crate::probe::{get, reset, Probe};
        let (w, h) = (4, 64);
        let (sc, data, fb, _counts) = make_deep_pixels(w, h);
        for &n in &[2i32, 8] {
            reset(Probe::CodecCreated);
            let header = make_deep_header(w, h, Compression::Zips);
            let mut out = MemOStream::new();
            {
                let mut df = DeepScanLineOutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                df.set_frame_buffer(&fb).unwrap();
                df.write_pixels(&sc, &data, h).unwrap();
            }
            assert!(get(Probe::CodecCreated) <= 2 * n as u64, "num_threads={n}: {} codecs built for 64 lines", get(Probe::CodecCreated));
        }
    }
}
