//! Port of OpenEXR `ImfDeepScanLineInputFile.cpp` / `.h` —
//! `class DeepScanLineInputFile`, the single-part **deep scanline** reader.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineInputFile.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineInputFile.h>
//!
//! # A note on the port target
//!
//! In OpenEXR v3.3.11 `DeepScanLineInputFile` is a thin C++ wrapper over the
//! OpenEXRCore C decode pipeline (`exr_read_scanline_chunk_info`,
//! `exr_decoding_run`, `EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL`, …): the offset
//! table read, chunk seek, separate count-table / sample-data decompress and
//! the cumulative→per-pixel count conversion all live inside the C core.  A
//! literal token-for-token port of the v3.3.11 `.cpp` would have to port the
//! entire C core first (out of scope), so — exactly like the non-deep
//! [`crate::scanline::InputFile`] reader — this module mirrors the **legacy
//! IlmImf `DeepScanLineInputFile`** read algorithm, which is the canonical
//! description of the on-disk byte format and is the precise inverse of this
//! crate's [`DeepScanLineOutputFile`][crate::deep::DeepScanLineOutputFile]
//! writer:
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.1.0/src/lib/OpenEXR/ImfDeepScanLineInputFile.cpp>
//!
//! The v3.3.11 semantics that *are* observable are matched exactly:
//! * the constructor validates magic + version (deep flag set, scanline — i.e.
//!   NOT tiled — flag), reads the header, checks `type == deepscanline`
//!   (`Data::initialize`), and reads the line-offset table;
//! * `setFrameBuffer` runs the deep `prepFillList` checks (x/y-sampling must
//!   match; the sample-count slice base must be set) and records the fill list;
//! * `readPixelSampleCounts(s1,s2)` decompresses *only* each chunk's count
//!   table, converts the cumulative running totals back to per-pixel counts and
//!   writes them into the sample-count slice (C++ `copy_sample_count` consuming
//!   the C core's `EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL` table);
//! * `readPixels(s1,s2)` skips the count table, decompresses the sample data and
//!   scatters each channel's samples into the caller's per-pixel arrays via the
//!   [`crate::deep::DeepSlice`] addressing (`update_pointers` + the decode scatter).
//!
//! # The two-pass deep read (count then data) — see [`crate::deep`]
//!
//! Deep reading is inherently two passes because each pixel's sample count is
//! data-dependent:
//!
//! 1. [`read_pixel_sample_counts`][DeepScanLineInputFile::read_pixel_sample_counts]
//!    fills the per-pixel sample-count buffer.  The caller then sizes the
//!    per-pixel sample arrays and points the [`DeepFrameBuffer`] at them.
//! 2. [`read_pixels`][DeepScanLineInputFile::read_pixels] fills those arrays
//!    with the actual samples.
//!
//! # The cumulative → per-pixel count conversion (the bug-prone spot)
//!
//! The writer ([`DeepScanLineOutputFile`][crate::deep::DeepScanLineOutputFile])
//! stores, per scanline, a **cumulative running total**: for x in `[minX,maxX]`
//! it writes the running sum `Σ_{x'≤x} sampleCount(x', y)` as an Xdr `int`.
//! The last int of a line is the line's *total* sample count.  To recover the
//! per-pixel counts the reader takes the first difference:
//!
//! ```text
//! perPixel[0]     = cumulative[0]
//! perPixel[x]     = cumulative[x] - cumulative[x-1]   (x > 0)
//! ```
//!
//! This is exactly what the C core does when it sets
//! `EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL` before `copy_sample_count` writes
//! the individual counts into the user's sample-count slice.
//!
//! # Safe-Rust adaptation of the `char**` deep addressing
//!
//! The C++ deep `base` is a `char**` (per-pixel pointers into a sample heap),
//! impossible in safe Rust (this crate has zero `unsafe` and no FFI).
//! This reader keeps the *algorithm* 1:1 and uses the **same flat-buffer model**
//! as the writer (see its module docs): the caller provides
//! * a flat **sample-count buffer** (`&mut [u8]`) holding one `u32` per pixel,
//!   addressed by the sample-count slice's byte strides (== C++
//!   `sampleCount(base,xStride,yStride,x,y)`), and
//! * a flat per-channel **sample heap** (`&mut [u8]`) in which each pixel's
//!   sample array is laid out contiguously in `(x, y)` raster order over the
//!   lines of the `read_pixels` call (the first requested line starts the
//!   heap; a whole-window request is the writer's whole-window layout); a deep
//!   slice's `base_offset` is its channel's heap origin and the per-pixel array
//!   start advances by `count * sampleStride` over preceding pixels.
//!
//! The bytes consumed are byte-identical to the writer's, so this reader
//! round-trips the writer pixel-exact (and count-exact).

use crate::attr::name::Name;
use crate::attr::PixelType;

use crate::compressor::{new_compressor, Compressor};
use crate::container::xdr;
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::deep::sample_table::{check_chunk_sizes, count_byte_offset, decode_cumulative_row};
use crate::error::{ExrError, Result};
use crate::header::{Header, DEEP_SCANLINE};
use crate::io::IStream;
use crate::multipart::input::{read_prologue, Storage};
use crate::misc::{pixel_type_size, try_filled_vec};

use imath_rs::modp;
use std::marker::PhantomData;
use std::ops::DerefMut;

/// Xdr `int` size in bytes — the per-pixel cumulative-count table entry width
/// (C++ writes each running total via `Xdr::write<int>`).
const SIZE_INT: usize = 4;

// ---------------------------------------------------------------------------
// InSliceInfo — read-side per-channel deep descriptor
// ---------------------------------------------------------------------------

/// Per-channel deep read descriptor — the read-side counterpart of the
/// writer's `OutSliceInfo` and the inverse of the C++ `update_pointers` /
/// `run_fill` channel handling.
///
/// Built by [`DeepScanLineInputFile::set_frame_buffer`] from the file's channel
/// list and the user [`DeepFrameBuffer`].  Like the non-deep reader's
/// `InSliceInfo`, the raw `char**` per-pixel pointer base becomes a signed byte
/// offset (`base_offset`) into the caller-owned flat sample heap (see the
/// module docs).
#[derive(Clone, Debug)]
struct InSliceInfo {
    /// Channel name (used to match the file channel order on the read scatter).
    name: Name,
    /// Sample type the frame-buffer slice expects (the scatter target type).
    type_: PixelType,
    /// Signed byte offset of this channel's flat sample heap within the
    /// caller's `sample_data` buffer.  Replaces the C++ `char** base`.
    base_offset: i64,
    /// Byte distance between consecutive samples of one pixel — C++
    /// `DeepSlice::sampleStride`.
    sample_stride: i32,
    /// Horizontal sub-sampling factor (C++ `xSampling`).  Deep channels require
    /// `xSampling == 1` (checked against the file channel in `set_frame_buffer`),
    /// so the de-interleave walks every x; kept for structural fidelity with the
    /// C++ `DeepSlice` / `run_fill`'s `xSampling != 1` guard.
    #[allow(dead_code)]
    x_sampling: i32,
    /// Vertical sub-sampling factor (C++ `ySampling`).  Used to skip scanlines
    /// where `modp(y, ySampling) != 0`.
    y_sampling: i32,
    /// `true` when this frame-buffer slice has no matching file channel — the
    /// C++ *fill list* entry.  On read its per-pixel arrays are filled with the
    /// slice's fill value (no bytes are consumed from the chunk data).
    fill: bool,
    /// Fill value written into a `fill` slice's samples (C++ `slice.fillValue`).
    fill_value: f64,
}

// ---------------------------------------------------------------------------
// DeepScanLineInputFile — port of `class DeepScanLineInputFile`
// ---------------------------------------------------------------------------

/// Single-part deep scanline EXR reader — port of `Imf::DeepScanLineInputFile`.
///
/// Construct with [`DeepScanLineInputFile::new`] (validates magic + version,
/// reads the header and the line-offset table), describe the destination with
/// [`DeepScanLineInputFile::set_frame_buffer`], then perform the two-pass read:
/// [`read_pixel_sample_counts`][DeepScanLineInputFile::read_pixel_sample_counts]
/// (fills counts) followed by
/// [`read_pixels`][DeepScanLineInputFile::read_pixels] (fills the samples).
///
/// The reader borrows the caller's [`IStream`] for its whole lifetime.
pub struct DeepScanLineInputFile<'a, S = &'a mut dyn IStream>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Input stream — borrowed (`&'a mut dyn IStream`, the C++ `_streamData->is`)
    /// or owned (`Box<dyn IStream + 'a>`, [`DeepScanLineInputFile::open`]) — the
    /// same owned-stream genericization pattern as [`crate::scanline::ScanLineInputFile`]
    /// / [`crate::tiled::TiledInputFile`] (see their module docs for the rationale).
    stream: S,

    /// Everything else this reader owns (mirrors
    /// `crate::scanline::input::ScanLineInputFileState`): factored into its
    /// own, stream-free struct so [`crate::multipart::MultiPartInputFile`] can
    /// persist it across separate `deep_scanline_input_part` calls instead of
    /// rebuilding a fresh reader every call.
    state: DeepScanLineInputFileState,

    /// Marker only — `S`'s bound (not a field type) is the only thing tying
    /// `'a` to this struct, which leaves `'a` structurally unused (`E0392`)
    /// without this, exactly like `TiledInputFile`'s own marker field.
    _stream_lt: PhantomData<&'a mut ()>,
}

/// A [`DeepScanLineInputFile`] that **owns** its input stream
/// (`Box<dyn IStream>`) — the deep-scanline analogue of
/// [`crate::tiled::TiledInputFileOwned`] / [`crate::scanline::InputFileOwned`].
/// Build it with [`DeepScanLineInputFile::open`].
pub type DeepScanLineInputFileOwned<'a> = DeepScanLineInputFile<'a, Box<dyn IStream + 'a>>;

/// Every field of C++ `DeepScanLineInputFile::Data` except the stream — the
/// reader's **persisted** per-part state, owned by the multipart file between
/// part-view calls.
pub(crate) struct DeepScanLineInputFileState {
    /// The image header — C++ `_data->header`.
    header: Header,

    /// Raw version word read from the file — C++ `_data->version`.
    version: i32,

    /// Data-window bounds — C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,

    /// File offsets of each chunk, indexed by `y - minY` — C++
    /// `_data->lineOffsets`.  Deep-valid codecs all have `linesInBuffer == 1`,
    /// so there is one entry per scanline.
    line_offsets: Vec<u64>,

    /// Per-channel read descriptors — built by
    /// [`set_frame_buffer`][DeepScanLineInputFile::set_frame_buffer] (empty
    /// until then).  C++ `_data->frameBuffer` + `_data->fill_list`.
    slices: Vec<InSliceInfo>,
    /// `true` once a valid frame buffer has been set — C++
    /// `_data->frameBufferValid`.
    frame_buffer_valid: bool,
    /// The caller's frame buffer, kept for [`frame_buffer`][Self::frame_buffer]
    /// — C++ `const DeepFrameBuffer& frameBuffer() const`. Cloned in
    /// `set_frame_buffer`; `DeepFrameBuffer::new()` (empty) until then.
    caller_frame_buffer: DeepFrameBuffer,

    /// Byte offset of the sample-count slice's origin within the caller's count
    /// buffer — C++'s base pointer.  `0` for the whole-window layout; a local
    /// raster sets it negative to map its low corner to `0`.
    sample_count_base_offset: i64,
    /// Byte strides of the sample-count slice — C++ `sampleCountSlice.xStride`
    /// / `.yStride`.  Recorded by `set_frame_buffer`; the read passes scatter
    /// per-pixel counts into the user buffer using these.
    sample_count_x_stride: i32,
    sample_count_y_stride: i32,

    /// The image compression — C++ `header.compression()`.
    compression: crate::attr::Compression,
    /// The **separate** sample-count-table decompressor, or `None` for NONE —
    /// C++ `LineBuffer::sampleCountTableCompressor`.  The count table and the
    /// sample data are compressed independently, so they need separate codec
    /// instances (a single instance reuses one scratch buffer). Persistent
    /// (built once at construction, not per chunk) — see `data_compressor`'s
    /// doc for why this is now safe even though a single scanline's sample
    /// count is unbounded a priori.
    count_compressor: Option<Box<dyn Compressor>>,
    /// The **separate** sample-DATA decompressor for the fully SERIAL read
    /// path (`read_data_line`), or `None` for NONE. Because
    /// `Compressor::uncompress` takes an explicit `expected_len` per call,
    /// this field is built once here (any construction size works;
    /// `expected_len` is authoritative per call) and reused rather than
    /// rebuilt per chunk — the same persistent-codec design as the flat
    /// scanline/tiled readers. The genuinely PARALLEL decode path
    /// (`decode_and_scatter_line`, run from several rayon threads at once)
    /// cannot share this field (or any single `&mut self` compressor) across
    /// concurrent calls, so it checks codecs out of a `CodecPool` instead:
    /// about one per worker, reused across lines.
    data_compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this reader drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `DeepScanLineInputFile::Data::{multiPartBackwardSupport, partNumber}`.
    ///
    /// The same unified seam as the other readers: when `Some`, each deep
    /// chunk's leading `partNumber` (i32 LE) is read and verified against this
    /// part; when `None` there is no such prefix.
    part_number: Option<i32>,

    /// Growable per-row sample-count prefix cache, sourced from the
    /// caller's own `sample_counts` buffer rather than the chunk stream
    /// and rebuilt at the start of every `read_pixels` call, so it
    /// is addressed from that call's first line — see
    /// [`samples_before_line`](Self::samples_before_line).
    sample_prefix: crate::deep::sample_table::RowPrefixCache,

    /// This file's captured thread count — see
    /// [`crate::threading::ThreadCount`] and the flat scanline reader's
    /// identical field. Gates [`Self::read_pixels`]'s parallel path; never
    /// touches [`Self::read_pixel_sample_counts`], mirroring C++ (only
    /// `readPixels` is threaded there, never `readPixelSampleCounts`).
    num_threads: crate::threading::ThreadCount,
}

impl<'a> DeepScanLineInputFile<'a, &'a mut dyn IStream> {
    /// Open a `DeepScanLineInputFile` on `stream` — port of the
    /// `DeepScanLineInputFile` constructor followed by `Data::initialize`.
    ///
    /// Validates the magic number + version word (deep flag set, not tiled, not
    /// multipart), reads the header, checks the part is a deep scanline part
    /// (`type == deepscanline`), runs the deep `sanity_check`, and reads the
    /// line-offset table.  Constructs the two separate decompressors (count
    /// table + sample data).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] for non-deep or tiled files (part 0 of a multipart file is read), or a part
    ///   whose `type` is not `deepscanline` (matching the C++ "File part is not
    ///   a deep scanline part").
    /// * Propagates [`Header::read_from`] / [`Header::sanity_check`] failures.
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

        // C++ `Data::initialize`: storage(partNumber) must be DEEP_SCANLINE.
        if part.storage != Storage::DeepScanline {
            return Err(ExrError::Arg(format!(
                "File part is not a deep scanline part (type must be \"{DEEP_SCANLINE}\")."
            )));
        }

        Self::from_part_mt(stream, part.header, version_word, part_number, part.chunk_offsets, num_threads)
    }
}

/// Owned-stream constructors (`S = Box<dyn IStream + 'a>`) — mirrors
/// [`crate::tiled::TiledInputFile::open`]/`new_owned`.
impl<'a> DeepScanLineInputFile<'a, Box<dyn IStream + 'a>> {
    /// Open `path`, OWNING the decoded stream — the deep-scanline analogue of
    /// [`crate::tiled::TiledInputFile::open`] / [`crate::scanline::InputFile::open`].
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display()))
        })?;
        let stream: Box<dyn IStream + 'a> = Box::new(crate::io::StdIfStream::open(path_str)?);
        Self::new_owned(stream)
    }

    /// Stream form of [`open`](Self::open): mirrors the borrowed `new`'s body
    /// exactly (prologue -> storage check -> `from_part`, now generic over `S`).
    pub fn new_owned(mut stream: Box<dyn IStream + 'a>) -> Result<Self> {
        let prologue = read_prologue(&mut *stream, true)?;
        // C++ opens part 0 of a multipart file like a single-part one (its chunk leaders then
        // carry the part number, which the reader checks).
        let part_number = prologue.multipart.then_some(0);
        let version_word = prologue.version;
        let part = prologue
            .parts
            .into_iter()
            .next()
            .ok_or_else(|| ExrError::Input("File contains no parts.".into()))?;

        if part.storage != Storage::DeepScanline {
            return Err(ExrError::Arg(format!(
                "File part is not a deep scanline part (type must be \"{DEEP_SCANLINE}\")."
            )));
        }

        Self::from_part(stream, part.header, version_word, part_number, part.chunk_offsets)
    }
}

impl<'a, S> DeepScanLineInputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Open a `DeepScanLineInputFile` for **one part of a multipart file** —
    /// port of `DeepScanLineInputFile::DeepScanLineInputFile(InputPartData*)`.
    ///
    /// The enclosing [`MultiPartInputFile`](crate::multipart::MultiPartInputFile)
    /// has already read the file's magic + version word, every part header, and
    /// every part's chunk offset table.  This constructor takes the
    /// already-parsed `header`, the file `version` word, this part's
    /// `line_offsets` table, and the part's `part_number`, and rebuilds the
    /// per-part read state without touching the stream prologue.  Each deep
    /// chunk's leading `partNumber` is verified against `part_number`.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepScanLineInputFile.cpp> (`DeepScanLineInputFile(InputPartData*)`) + `OpenEXRCore/chunk.c`.
    pub fn from_part(
        stream: S,
        header: Header,
        version: i32,
        part_number: Option<i32>,
        line_offsets: Vec<u64>,
    ) -> Result<Self> {
        Self::from_part_mt(stream, header, version, part_number, line_offsets, None)
    }

    /// Same as [`Self::from_part`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::from_part`]).
    pub fn from_part_mt(
        stream: S,
        header: Header,
        version: i32,
        part_number: Option<i32>,
        line_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let state = DeepScanLineInputFileState::from_part_mt(
            header,
            version,
            part_number,
            line_offsets,
            num_threads,
        )?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }

    /// The image header — port of `const Header& DeepScanLineInputFile::header()`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The currently set frame buffer — port of
    /// `const DeepFrameBuffer& DeepScanLineInputFile::frameBuffer() const`.
    /// Empty until [`set_frame_buffer`][Self::set_frame_buffer] has
    /// been called.
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }

    /// Whether the on-disk offset table was complete — port of `bool
    /// DeepScanLineInputFile::isComplete() const`.
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// The raw file-format version word — port of
    /// `int DeepScanLineInputFile::version()`.
    pub fn version(&self) -> i32 {
        self.state.version()
    }

    /// First scanline of the chunk that stores scanline `y` — port of
    /// `DeepScanLineInputFile::firstScanLineInChunk`.  For the deep-valid codecs
    /// (`linesInBuffer == 1`) every chunk is one scanline, so this is `y`.
    pub fn first_scan_line_in_chunk(&self, y: i32) -> i32 {
        self.state.first_scan_line_in_chunk(y)
    }

    /// Last scanline of the chunk that stores scanline `y` — port of
    /// `DeepScanLineInputFile::lastScanLineInChunk` (`== y` for deep codecs).
    pub fn last_scan_line_in_chunk(&self, y: i32) -> i32 {
        self.state.last_scan_line_in_chunk(y)
    }

    /// Set the pixel destination. See
    /// [`DeepScanLineInputFileState::set_frame_buffer`] for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Read the per-pixel sample counts. See
    /// [`DeepScanLineInputFileState::read_pixel_sample_counts`] for the full doc.
    pub fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_counts(&mut *self.stream, scan_line1, scan_line2, sample_counts)
    }

    /// Read deep samples for the inclusive requested scanline range.
    ///
    /// Each channel's sample heap starts at its slice's `base_offset`, with
    /// no sample prefix for rows before this request. Counts follow the
    /// separately configured count slice. To populate a larger heap through
    /// successive requests, update channel base offsets and call
    /// [`Self::set_frame_buffer`] before each read.
    ///
    /// See [`crate::deep::frame_buffer`] for the flat sample-heap addressing
    /// model; count-slice coordinates do not change this request-local domain.
    pub fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixels(&mut *self.stream, scan_line1, scan_line2, sample_counts, sample_data)
    }

    /// Read scanline `first_scan_line`'s chunk RAW (still-packed count table +
    /// sample data, no decompression). See
    /// [`DeepScanLineInputFileState::raw_pixel_data`] for the full doc.
    pub fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>, Vec<u8>, u64)> {
        self.state.raw_pixel_data(&mut *self.stream, first_scan_line)
    }
}

// ---------------------------------------------------------------------------
// DeepScanLineInputFileState — the state-only construction + algorithm,
// shared by the owning `DeepScanLineInputFile` and the borrowing multipart
// part-reader view (`multipart::input`).
// ---------------------------------------------------------------------------

impl DeepScanLineInputFileState {
    /// Multipart-part / standalone construction — port of
    /// `DeepScanLineInputFile::DeepScanLineInputFile(InputPartData*)`. Pure
    /// (no stream access at all): the prologue reader
    /// (`multipart::input::read_prologue`) already parses the header + chunk
    /// offset table before this ever runs (the same holds for
    /// `ScanLineInputFileState::from_part_mt`).
    /// `num_threads` is the per-file override (`None` = capture
    /// the current process-wide default).
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        line_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // The part must be a deep scanline part (the container validated the
        // `type`; re-check so this path never mis-reads another kind).
        if !header.has_type() || header.type_()? != DEEP_SCANLINE {
            return Err(ExrError::Arg(format!(
                "File part is not a deep scanline part (type must be \"{DEEP_SCANLINE}\")."
            )));
        }

        // `OpenEXRCore/decoding.c:307-318`: a deep part's `version`
        // attribute (defaulted to 1 when absent — `multipart/input.rs`) must
        // be exactly 1; any other value is rejected at decode-init time in
        // C++ ("Version %d not supported for deepscanline images"). This is
        // the per-part deep `version` ATTRIBUTE, distinct from the file's
        // magic/version WORD (the `version` parameter above).
        if header.has_version() && header.version()? != 1 {
            return Err(ExrError::Arg(format!(
                "Version {} not supported for deepscanline images.",
                header.version()?
            )));
        }

        // C++: header.sanityCheck(isTiled = false, isMultipart).
        header.sanity_check_at_open(false, part_number.is_some())?;

        let data_window = header.data_window()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        let compression = header.compression()?;
        let lines_in_buffer = {
            let probe = new_compressor(compression, 0, &header)?;
            match probe.as_deref() {
                Some(c) => c.num_scan_lines(),
                None => crate::compressor::num_lines_in_buffer(compression)?,
            }
        };
        let lines_per_buffer = lines_in_buffer.min(max_y - min_y + 1) as usize;
        let width = (max_x - min_x + 1) as usize;
        let max_sample_count_table_size = lines_per_buffer * width * core::mem::size_of::<u32>();
        let count_compressor = new_compressor(compression, max_sample_count_table_size, &header)?;
        // Any construction size works now that
        // `expected_len` is authoritative per `uncompress()` call -- see
        // `data_compressor`'s doc.
        let data_compressor = new_compressor(compression, 0, &header)?;

        Ok(Self {
            header,
            version,
            min_x,
            max_x,
            min_y,
            max_y,
            line_offsets,
            slices: Vec::new(),
            frame_buffer_valid: false,
            caller_frame_buffer: DeepFrameBuffer::new(),
            sample_count_base_offset: 0,
            sample_count_x_stride: 0,
            sample_count_y_stride: 0,
            compression,
            count_compressor,
            data_compressor,
            part_number,
            sample_prefix: crate::deep::sample_table::RowPrefixCache::new(min_x, max_x, min_y),
            num_threads,
        })
    }

    /// The image header — port of `const Header& DeepScanLineInputFile::header()`.
    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    /// The currently set frame buffer — port of
    /// `const DeepFrameBuffer& DeepScanLineInputFile::frameBuffer() const`.
    /// Empty until [`set_frame_buffer`][Self::set_frame_buffer] has
    /// been called.
    pub(crate) fn frame_buffer(&self) -> &DeepFrameBuffer {
        &self.caller_frame_buffer
    }

    /// Whether the on-disk offset table was complete — port of `bool
    /// DeepScanLineInputFile::isComplete() const`
    /// (`ImfDeepScanLineInputFile.cpp:289-293`: `_ctxt.chunkTableValid(...)`).
    /// A missing chunk leaves its offset-table slot at `0` (the crate's own
    /// convention — see `chunk::offsets::read_offset_table`/`reconstruct`),
    /// so "every slot non-zero" is the same condition, computed locally
    /// exactly as `deep::tiled_input::DeepTiledInputFile::is_complete` does
    /// via `TileOffsets::read_from_vec`'s own `complete` out-parameter.
    pub(crate) fn is_complete(&self) -> bool {
        self.line_offsets.iter().all(|&o| o != 0)
    }

    /// The raw file-format version word — port of
    /// `int DeepScanLineInputFile::version()`.
    pub(crate) fn version(&self) -> i32 {
        self.version
    }

    /// First scanline of the chunk that stores scanline `y` — port of
    /// `DeepScanLineInputFile::firstScanLineInChunk`.  For the deep-valid codecs
    /// (`linesInBuffer == 1`) every chunk is one scanline, so this is `y`.
    pub(crate) fn first_scan_line_in_chunk(&self, y: i32) -> i32 {
        y
    }

    /// Last scanline of the chunk that stores scanline `y` — port of
    /// `DeepScanLineInputFile::lastScanLineInChunk` (`== y` for deep codecs).
    pub(crate) fn last_scan_line_in_chunk(&self, y: i32) -> i32 {
        y
    }

    /// Set the pixel destination — port of `DeepScanLineInputFile::setFrameBuffer`
    /// → `Data::prepFillList`.
    ///
    /// Performs the deep `prepFillList` checks (the sample-count slice base must
    /// be set; every frame-buffer slice that also names a file channel must
    /// agree on x/y subsampling) and records the per-channel `InSliceInfo`
    /// table, including the *fill list* (frame-buffer slices the file does not
    /// provide → filled with their fill value).
    ///
    /// `sample_counts_is_set` is the safe-Rust equivalent of "the sample-count
    /// slice base pointer is non-null": the caller passes `true` once it has
    /// registered a real sample-count slice on `frame_buffer`.  (In C++ the
    /// non-null `base` is checked directly; here the count buffer is supplied
    /// per-call to the read methods, so the descriptor records its strides and
    /// validity from the slice geometry.)
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when the sample-count slice has zero strides (treated
    ///   as an unset base — C++ "Invalid base pointer, please set a proper
    ///   sample count slice.").
    /// * [`ExrError::Arg`] when a slice's x/y-sampling is incompatible with the
    ///   header channel (matching the C++ `THROW`).
    pub(crate) fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++ prepFillList: the sample-count slice base must be set.  In the
        // safe-Rust model the count slice is described by its byte strides; a
        // slice with both strides zero is the equivalent of a null base.
        let sc_slice = frame_buffer.get_sample_count_slice();
        if sc_slice.x_stride == 0 && sc_slice.y_stride == 0 {
            return Err(ExrError::Arg(
                "Invalid base pointer, please set a proper sample count slice.".into(),
            ));
        }
        self.sample_count_base_offset = sc_slice.base_offset;
        self.sample_count_x_stride = sc_slice.x_stride as i32;
        self.sample_count_y_stride = sc_slice.y_stride as i32;
        // C++ prepFillList: walk the frame buffer; a slice naming a file
        // channel must agree on subsampling (else it joins the fill list).
        let mut slices: Vec<InSliceInfo> = Vec::new();
        for (name, deep_slice) in frame_buffer.iter() {
            match channels.find_channel(name) {
                None => {
                    // No file channel → fill-list entry (filled with fillValue).
                    // `ImfDeepScanLineInputFile.cpp:928-929`: a
                    // fill-list slice still must have sampling (1,1) — C++
                    // throws `InputExc` "Expect sampling of 1" at run_fill
                    // time; we reject it up front instead of silently
                    // skipping lines via `modp`.
                    if deep_slice.slice.x_sampling != 1 || deep_slice.slice.y_sampling != 1 {
                        return Err(ExrError::Input("Expect sampling of 1".into()));
                    }
                    slices.push(InSliceInfo {
                        name: name.clone(),
                        type_: deep_slice.slice.type_,
                        base_offset: deep_slice.slice.base_offset,
                        sample_stride: deep_slice.sample_stride,
                        x_sampling: deep_slice.slice.x_sampling,
                        y_sampling: deep_slice.slice.y_sampling,
                        fill: true,
                        fill_value: deep_slice.slice.fill_value,
                    });
                }
                Some(ch) => {
                    if ch.x_sampling != deep_slice.slice.x_sampling
                        || ch.y_sampling != deep_slice.slice.y_sampling
                    {
                        return Err(ExrError::Arg(format!(
                            "X and/or y subsampling factors of \"{name}\" channel of \
                             input file are not compatible with the frame buffer's \
                             subsampling factors."
                        )));
                    }
                    slices.push(InSliceInfo {
                        name: name.clone(),
                        type_: deep_slice.slice.type_,
                        base_offset: deep_slice.slice.base_offset,
                        sample_stride: deep_slice.sample_stride,
                        x_sampling: deep_slice.slice.x_sampling,
                        y_sampling: deep_slice.slice.y_sampling,
                        fill: false,
                        fill_value: deep_slice.slice.fill_value,
                    });
                }
            }
        }

        self.slices = slices;
        self.caller_frame_buffer = frame_buffer.clone();
        self.frame_buffer_valid = true;
        Ok(())
    }

    /// Read the per-pixel sample counts for scanlines `[min(s1,s2),
    /// max(s1,s2)]` — port of `DeepScanLineInputFile::readPixelSampleCounts` →
    /// `Data::readData(countsOnly=true)` → `copy_sample_count`.
    ///
    /// For every scanline in range this seeks to the chunk via the offset
    /// table, reads `y` + the three size fields, decompresses *only* the count
    /// table (its own decompressor), converts the cumulative running totals
    /// back to per-pixel counts (first difference, see the module docs) and
    /// writes the per-pixel counts into the caller's flat `sample_counts` buffer
    /// using the sample-count slice's byte strides.
    ///
    /// `sample_counts` is the caller-owned flat per-pixel count buffer
    /// (one `u32` LE per pixel), addressed exactly like the C++
    /// `sampleCount(base, xStride, yStride, x, y)` helper.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the range is outside
    ///   the data window (matching the C++ throws).
    /// * [`ExrError::Input`] on a corrupt chunk header / count table.
    pub(crate) fn read_pixel_sample_counts(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        if !self.frame_buffer_valid {
            return Err(ExrError::Arg(
                "readPixelSampleCounts called with no valid frame buffer".into(),
            ));
        }

        let (min_scan, max_scan) = self.range_check(scan_line1, scan_line2)?;

        for y in min_scan..=max_scan {
            let (per_pixel, _packed_data, _unpacked) = self.read_count_table(stream, y)?;
            // C++ copy_sample_count: write each per-pixel count into the user's
            // sample-count slice at sampleCount(base, xStride, yStride, x, y).
            self.scatter_counts(y, &per_pixel, sample_counts)?;
        }
        Ok(())
    }

    /// Read the deep sample data for scanlines `[min(s1,s2), max(s1,s2)]` into
    /// the caller's flat per-channel sample heap — port of
    /// `DeepScanLineInputFile::readPixels` → `Data::readData(countsOnly=false)`
    /// → `run_decode` / the decode scatter.
    ///
    /// Must be called *after*
    /// [`read_pixel_sample_counts`][DeepScanLineInputFile::read_pixel_sample_counts]
    /// (the caller uses the counts to size the per-pixel arrays inside
    /// `sample_data`).  For every scanline in range this seeks to the chunk,
    /// skips the count table, decompresses the sample data and de-interleaves
    /// each channel's samples into the caller's per-pixel arrays via the
    /// [`crate::deep::DeepSlice`] addressing (`base_offset + pixelStart + i*sampleStride`).
    /// Channels absent from the file (the fill list) are filled with their fill
    /// value.  The chunk's own `y` is verified to equal the requested scanline
    /// (C++ checks this).
    ///
    /// `sample_counts` is the per-pixel count buffer filled by the first pass
    /// (it drives the per-pixel array sizes); `sample_data` is the flat
    /// per-channel sample heap the deep slices' `base_offset` index into.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the range is outside
    ///   the data window.
    /// * [`ExrError::Input`] on a corrupt chunk (bad `y`, wrong decompressed
    ///   size) or a sample address outside `sample_data`.
    pub(crate) fn read_pixels(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        if !self.frame_buffer_valid {
            return Err(ExrError::Arg(
                "readPixels called with no valid frame buffer".into(),
            ));
        }

        let (min_scan, max_scan) = self.range_check(scan_line1, scan_line2)?;

        // The heap is addressed from THIS call's first line (C++ hands `readPixels` per-pixel
        // pointers, so nothing about lines outside the request is ever consulted). Reset cached
        // totals for the new count buffer while retaining row/prefix allocation capacity.
        self.sample_prefix.reset(self.min_x, self.max_x, min_scan);

        // Gate: parallel only when there is more than one line
        // AND the captured thread count is `> 1` — matches C++'s
        // `nchunks > 1 && numThreads > 1` gate on the flat readers (deep
        // chunks are always 1 line each, so "chunks" and "lines" coincide
        // here). Mirrors C++'s own asymmetry: only `readPixels` (this
        // function) is ever threaded, never `readPixelSampleCounts`.
        if (max_scan - min_scan + 1) > 1
            && self.num_threads.is_parallel()
            && self.read_pixels_parallel(stream, min_scan, max_scan, sample_counts, sample_data)?
        {
            crate::probe::hit(crate::probe::Probe::DeepScanReadParallel);
            return Ok(());
        }
        // A `false`/skipped result means either the fast path wasn't
        // eligible, or the planner found overlapping write intervals (an
        // aliased/broadcast deep frame buffer) — fall back to serial.

        for y in min_scan..=max_scan {
            self.read_data_line(stream, y, sample_counts, sample_data)?;
        }
        Ok(())
    }

    /// The parallel read body — same three-function split and read-ahead-free
    /// shape as the flat scanline reader's first parallel version
    /// (decode-parallel-only; unlike the flat readers, there is no read-ahead
    /// pipelining here, since no deep I/O-share measurement has justified
    /// it yet). Returns `Ok(true)` when it fully serviced the request, or
    /// `Ok(false)` when [`crate::misc::plan_disjoint`] found the sample-data
    /// heap's write intervals unsafely overlapping.
    fn read_pixels_parallel(
        &mut self,
        stream: &mut dyn IStream,
        min_scan: i32,
        max_scan: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<bool> {
        let lines: Vec<i32> = (min_scan..=max_scan).collect();

        // Serial pre-pass: extend the (persisted, incrementally-growing)
        // `RowPrefixCache` up to `max_scan` and snapshot each line's
        // `samples_before` into a plain owned `Vec` — the cache itself stays
        // `&mut self`-only (its amortized-linear guarantee depends on
        // sequential, in-order extension), but a plain snapshot is safe to
        // share (read-only) across the parallel decode below. Same total
        // work as calling it once per line serially (no extra O(N) cost),
        // just moved up front.
        let mut before_line = Vec::with_capacity(lines.len());
        for &y in &lines {
            before_line.push(self.samples_before_line(sample_counts, y)?);
        }

        // Fetch every line's chunk (leader + count table decode + packed
        // sample-data bytes) on the calling thread — see `fetch_line`'s doc
        // for why the count table is decoded here rather than deferred.
        // Also verify each line's counts against the caller's buffer
        // while we're still serial and can report a clean per-line error.
        let mut raws = Vec::with_capacity(lines.len());
        for &y in &lines {
            let raw = self.fetch_line(stream, y)?;
            self.verify_counts(y, &raw.per_pixel, sample_counts)?;
            raws.push(raw);
        }

        let mut intervals = Vec::with_capacity(lines.len());
        for (raw, &samples_before) in raws.iter().zip(&before_line) {
            intervals.push(deep_line_write_interval(&self.slices, raw.y, &raw.per_pixel, samples_before)?);
        }
        let dests = match crate::misc::plan_disjoint(sample_data, &intervals) {
            Some(d) => d,
            None => return Ok(false),
        };

        let compression = self.compression;
        let header = &self.header;
        let slices = &self.slices;
        let (min_x, max_x) = (self.min_x, self.max_x);
        let count = self.num_threads;

        // A real checkout/checkin
        // `CodecPool` instead of `map_init` -- see `Compressor`'s trait
        // doc and `crate::compressor::CodecPool`'s own doc for why
        // `map_init` alone was not the "once per job split" reuse it
        // looked like it was. Any construction size works (`expected_len`
        // is authoritative per call).
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;
        let results: Vec<Result<()>> = crate::threading::run_on_global_pool(count, || {
            use rayon::prelude::*;
            raws.par_iter()
                .zip(before_line.par_iter())
                .zip(dests.into_par_iter())
                .map(|((raw, &samples_before), mut dest)| -> Result<()> {
                    let mut worker_compressor =
                        codec_pool.checkout(|| new_compressor(compression, 0, header))?;
                    let result = decode_and_scatter_line(
                        worker_compressor.as_mut(),
                        header,
                        slices,
                        min_x,
                        max_x,
                        raw,
                        samples_before,
                        &mut dest,
                    );
                    codec_pool.checkin(worker_compressor);
                    result
                })
                .collect()
        })?;
        // Deterministic error: the lowest line index that failed, not
        // whichever finished first in time.
        for r in results {
            r?;
        }
        Ok(true)
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    /// Swap `s1 > s2` and validate the range against the data window — the
    /// shared head of C++ `Data::readData`.
    fn range_check(&self, scan_line1: i32, scan_line2: i32) -> Result<(i32, i32)> {
        let (mut min_scan, mut max_scan) = (scan_line1, scan_line2);
        if max_scan < min_scan {
            std::mem::swap(&mut min_scan, &mut max_scan);
        }
        if min_scan < self.min_y || max_scan > self.max_y {
            return Err(ExrError::Arg(format!(
                "Tried to read scan line outside the image file's data window: \
                 {min_scan} - {max_scan} vs datawindow {} - {}",
                self.min_y, self.max_y
            )));
        }
        Ok((min_scan, max_scan))
    }

    /// Seek to scanline `y`'s chunk and read its header — returns
    /// `(packedCountSize, packedDataSize, unpackedDataSize)` with the stream
    /// positioned at the start of the (compressed) count table.  Verifies the
    /// chunk's stored `y` matches the request (C++ checks this).
    fn seek_chunk(&self, stream: &mut dyn IStream, y: i32) -> Result<(u64, u64, u64)> {
        // Deep-valid codecs all have linesInBuffer == 1, so the offset table
        // has one entry per scanline: index = y - minY.
        let offset_index = (y - self.min_y) as usize;
        let chunk_offset = *self.line_offsets.get(offset_index).ok_or_else(|| {
            ExrError::Input(format!("Line offset table index {offset_index} out of range."))
        })?;

        stream.seekg(chunk_offset)?;
        // For a multipart file the deep chunk leader begins with the part
        // number (i32 LE), which must match the part being read (C++
        // `extract_chunk_leader`); single-part files have no such prefix.
        // Then: y(i32), packedCountSize(u64), packedDataSize(u64),
        // unpackedDataSize(u64).
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(&mut *stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in deep scan-line chunk: expected {expected}, \
                     found {found}."
                )));
            }
        }
        let chunk_y = xdr::read_i32(&mut *stream)?;
        if chunk_y != y {
            return Err(ExrError::Input(format!(
                "Deep scanline chunk has y coordinate {chunk_y} but scanline {y} \
                 was requested (corrupt offset table or chunk)."
            )));
        }
        let packed_count = xdr::read_u64(&mut *stream)?;
        let packed_data = xdr::read_u64(&mut *stream)?;
        let unpacked = xdr::read_u64(&mut *stream)?;

        // This is the ONE place all three deep chunk leader sizes are parsed,
        // feeding BOTH the normal decode path (`read_count_table`/
        // `read_data_line`) and the raw `copy_pixels` path (`raw_pixel_data`,
        // which allocates straight from `packed_count`/`packed_data`, so an
        // unchecked size there would be an allocation DoS). Ported C++'s own
        // leader-parse bounds
        // (`OpenEXRCore/chunk.c:1007-1066`, `read_and_validate_chunk_leader`)
        // here instead of only in the decode path, so every consumer of this
        // chunk's sizes -- present and future -- inherits the same guarantee:
        // each size fits `i32` (u64 is always >= 0, so only the upper bound
        // needs checking), and the sample-count table and packed data each
        // fit within the remaining bytes of the file.
        for (name, size) in [
            ("sample table", packed_count),
            ("packed data", packed_data),
            ("unpacked data", unpacked),
        ] {
            if size > i32::MAX as u64 {
                return Err(ExrError::Input(format!(
                    "Deep scanline chunk has an invalid {name} size {size} (corrupt leader)."
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
                    "Deep scanline chunk for scanline {y}: sample table size {packed_count} \
                     + packed data size {packed_data} results in access past the end of the \
                     file (corrupt leader)."
                )));
            }
        }

        Ok((packed_count, packed_data, unpacked))
    }

    /// Read scanline `y`'s chunk RAW -- the packed (still compressed, or
    /// already-raw for NONE/no-shrink) count-table and sample-data bytes,
    /// without decompressing or scattering into a frame buffer -- port
    /// of `void rawPixelData(int firstScanLine, char* pixelData, uint64_t&
    /// pixelDataSize)` (`ImfDeepScanLineInputFile.cpp:324-354`).
    ///
    /// C++'s signature is a two-call C-style sizing protocol (call once with
    /// a null/undersized buffer to learn the size, then again with a real
    /// one); the Rust port returns owned `Vec<u8>`s directly instead, which
    /// is the same data with no equivalent sizing dance needed. Used by
    /// [`DeepScanLineOutputFileState::copy_pixels`] to transcode a chunk
    /// without ever decompressing it (C++'s own `copyPixels` implementation,
    /// `ImfDeepScanLineOutputFile.cpp:1497-1548`, does exactly this: it reads
    /// the raw block, slices the 3 size fields out of its own fixed 28-byte
    /// header layout, and re-writes them via `writePixelData` verbatim).
    ///
    /// # Errors
    /// * [`ExrError::Input`] if the chunk is missing, or its header's stored
    ///   `y` doesn't match the request (corrupt offset table or chunk).
    pub(crate) fn raw_pixel_data(
        &mut self,
        stream: &mut dyn IStream,
        first_scan_line: i32,
    ) -> Result<(i32, Vec<u8>, Vec<u8>, u64)> {
        // `seek_chunk` already bounds `packed_count`/`packed_data` to `i32::MAX`
        // and to the remaining bytes of the file, so
        // these `usize` conversions cannot fail on a 32-bit target's `usize`
        // either -- `i32::MAX` always fits. `try_filled_vec` (fallible
        // `try_reserve`) is still used rather than `vec![0u8; n]`, matching
        // every other attacker-controlled-size allocation in this crate.
        let (packed_count, packed_data, unpacked) = self.seek_chunk(stream, first_scan_line)?;
        let packed_count_usize = usize::try_from(packed_count)
            .map_err(|_| ExrError::Input(format!("Count table size {packed_count} too large.")))?;
        let packed_data_usize = usize::try_from(packed_data)
            .map_err(|_| ExrError::Input(format!("Packed data size {packed_data} too large.")))?;
        let mut count_bytes = try_filled_vec(packed_count_usize, 0u8)?;
        stream.read(&mut count_bytes)?;
        let mut data_bytes = try_filled_vec(packed_data_usize, 0u8)?;
        stream.read(&mut data_bytes)?;
        Ok((first_scan_line, count_bytes, data_bytes, unpacked))
    }

    /// Read + decompress scanline `y`'s count table and convert the cumulative
    /// running totals to per-pixel counts.  Leaves the stream positioned at the
    /// start of the (compressed) sample data and returns `(per_pixel_counts,
    /// packedDataSize, unpackedDataSize)`, both sizes already validated by
    /// [`check_chunk_sizes`] -- the one count-table reader of every path
    /// (sample counts, serial data, the parallel pre-pass).
    fn read_count_table(&mut self, stream: &mut dyn IStream, y: i32) -> Result<(Vec<u32>, usize, usize)> {
        let (packed_count, packed_data, unpacked) = self.seek_chunk(stream, y)?;

        let width = (self.max_x - self.min_x + 1) as usize;
        let raw_table_size = width * SIZE_INT;

        // Reject an implausible count-table size BEFORE allocating a buffer
        // from the untrusted on-wire `packed_count`.  The writer stores the count
        // table either compressed (strictly smaller than the raw table) or raw
        // (exactly `raw_table_size`), so `packed_count` can never exceed
        // `raw_table_size` for a valid file; a malformed near-`u64::MAX` value is
        // a memory-exhaustion DoS and becomes a clean error here.
        if packed_count > raw_table_size as u64 {
            return Err(ExrError::Input(format!(
                "Deep count table size {packed_count} exceeds the maximum \
                 {raw_table_size} the writer could store (corrupt chunk)."
            )));
        }

        // Read the (possibly compressed) count table bytes.
        let packed_count_usize = usize::try_from(packed_count).map_err(|_| {
            ExrError::Input(format!("Count table size {packed_count} too large."))
        })?;
        let mut packed = vec![0u8; packed_count_usize];
        stream.read(&mut packed)?;

        // Decompress the count table with its OWN decompressor, mirroring the
        // writer's "no-shrink (or no compressor) → store raw" rule: if the
        // stored bytes already equal the raw table size, they are the raw table.
        //
        // C++ `OpenEXRCore/chunk.c:1046-1051`: for `EXR_COMPRESSION_NONE`
        // the leader requires `sample_count_table_size == chunkmin` (here
        // `raw_table_size`) EXACTLY — a zero-length table under NONE is a
        // corrupt chunk, not a valid absent one, since NONE stores the raw
        // ints and there is no codec to have "compressed away" an all-zero
        // row. The zero-length exception (below) is compressed-only.
        //
        // C++ `OpenEXRCore/decoding.c:590-609`: for a COMPRESSED count
        // table, an absent (zero-length) one is treated as an all-zero row,
        // not an error — a chunk whose count table compressed to nothing
        // (every pixel has zero samples) is valid and must read back as zero
        // counts rather than failing the decompressed-size check below.
        let cumulative_bytes: Vec<u8> = if packed_count_usize == raw_table_size {
            packed
        } else if self.count_compressor.is_none() {
            return Err(ExrError::Input(format!(
                "Invalid deep sample count size, must be one entry per pixel: \
                 found {packed_count_usize} expected {raw_table_size}."
            )));
        } else if packed_count_usize == 0 {
            vec![0u8; raw_table_size]
        } else if let Some(c) = self.count_compressor.as_mut() {
            let decoded = c
                .uncompress(&packed, y, raw_table_size)
                .map_err(crate::error::decode_err)?;
            if decoded.len() != raw_table_size {
                return Err(ExrError::Input(format!(
                    "Decompressed deep count table size {} does not match the \
                     expected size {raw_table_size} (corrupt chunk).",
                    decoded.len()
                )));
            }
            decoded.to_vec()
        } else {
            return Err(ExrError::Input(format!(
                "Deep count table size {packed_count_usize} differs from the raw \
                 size {raw_table_size} but the file has no compressor."
            )));
        };

        // Convert cumulative running totals → per-pixel counts (first difference,
        // reset to 0 per line) via the shared checked decoder.
        let mut per_pixel = vec![0u32; width];
        decode_cumulative_row(&cumulative_bytes, 0, width, &mut per_pixel)?;

        let (packed_data, unpacked) = check_chunk_sizes(
            &per_pixel, self.header.channels()?, self.compression, packed_data, unpacked, stream,
        )?;
        Ok((per_pixel, packed_data, unpacked))
    }

    /// Write per-pixel counts into the caller's flat sample-count buffer — port
    /// of C++ `copy_sample_count` (the `xStride != sizeof(int)` general path),
    /// addressing `sampleCount(base, xStride, yStride, x, y)`.
    fn scatter_counts(&self, y: i32, per_pixel: &[u32], sample_counts: &mut [u8]) -> Result<()> {
        for (i, x) in (self.min_x..=self.max_x).enumerate() {
            let count = *per_pixel.get(i).ok_or_else(|| {
                ExrError::Input("Per-pixel count index out of range.".into())
            })?;
            let off = count_byte_offset(
                self.sample_count_base_offset,
                self.sample_count_x_stride,
                self.sample_count_y_stride,
                x,
                y,
            )
            .ok_or_else(|| {
                ExrError::Arg(
                    "Sample-count slice address is out of range (negative).".into(),
                )
            })?;
            let dst = sample_counts.get_mut(off..off + SIZE_INT).ok_or_else(|| {
                ExrError::Arg(format!(
                    "Sample-count slice address {off} out of range of the count buffer."
                ))
            })?;
            dst.copy_from_slice(&count.to_le_bytes());
        }
        Ok(())
    }

    /// Fetch scanline `y`'s chunk — leader + count table +
    /// packed sample-data bytes — WITHOUT decompressing the sample data or
    /// scattering. Port of the "fetch" half of what `read_data_line` used to
    /// do in one step (count-table decode included here — see the
    /// module-level note on `read_pixels_parallel` for why: unlike the flat
    /// scanline/tiled readers, the count table's size and values are needed
    /// just to know how many packed DATA bytes to read next and to validate
    /// them, so this "fetch" step necessarily also decodes the (small,
    /// per-line) count table; only the SAMPLE DATA decode — where deep's
    /// real bulk cost lives — is deferred to [`decode_and_scatter_line`],
    /// mirroring C++'s own asymmetry that only `readPixels`, never
    /// `readPixelSampleCounts`, is threaded).
    ///
    /// `&mut self`, using the
    /// persistent `self.count_compressor` — this function has exactly ONE
    /// caller (`read_pixels_parallel`'s serial pre-pass for-loop, itself
    /// `&mut self`), which fetches every line ONE AFTER ANOTHER on the
    /// calling thread BEFORE the genuinely parallel decode step
    /// (`decode_and_scatter_line`, run from several rayon threads with
    /// `CodecPool` codecs) even starts. A prior version built a FRESH count-table
    /// compressor here out of excess caution about that later parallel
    /// step; but nothing this function does ever runs concurrently with
    /// itself or with `decode_and_scatter_line`'s DATA decode (`fetch_line`
    /// only touches the count table, a fully separate codec instance from
    /// `decode_and_scatter_line`'s data compressor), so reusing
    /// `self.count_compressor` here is the same "persistent codec for the
    /// serial path" design as the flat readers, not a correctness risk.
    fn fetch_line(&mut self, stream: &mut dyn IStream, y: i32) -> Result<RawDeepLine> {
        let (per_pixel, packed_data, unpacked_size) = self.read_count_table(stream, y)?;
        let mut packed_data_bytes = vec![0u8; packed_data];
        stream.read(&mut packed_data_bytes)?;
        Ok(RawDeepLine { y, per_pixel, packed_data: packed_data_bytes, unpacked_size })
    }

    /// Read scanline `y`'s sample data and scatter it into the caller's flat
    /// per-channel sample heap — the inverse of the writer's
    /// `gather_scanline` and the port of the deep `readPixels` decode scatter.
    ///
    /// The chunk data is laid out (per the writer): for each channel in *file
    /// channel order* (the header order), then each pixel x, that pixel's
    /// `count(x, y)` samples contiguously.  Fill-list channels (absent from the
    /// file) are not in the chunk data: they are filled with their fill value.
    fn read_data_line(
        &mut self,
        stream: &mut dyn IStream,
        y: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        // Re-read the count table for this line (drives the per-pixel sizes and
        // positions the stream at the start of the sample data).  This mirrors
        // C++ decoding the count table as part of every readPixels chunk.
        let (per_pixel, packed_data_usize, unpacked_usize) = self.read_count_table(stream, y)?;

        // Read the (possibly compressed) sample data bytes.
        let mut packed = vec![0u8; packed_data_usize];
        stream.read(&mut packed)?;

        // Decompress with the SEPARATE, persistent data decompressor
        // (`self.data_compressor`, reused across lines; see its field doc),
        // same raw-store rule: if the stored bytes already equal
        // the unpacked size they are raw.
        let data: Vec<u8> = if packed_data_usize == unpacked_usize {
            packed
        } else if let Some(c) = self.data_compressor.as_mut() {
            let decoded = c.uncompress(&packed, y, unpacked_usize).map_err(crate::error::decode_err)?;
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

        // The line's total sample count (Σ over x) — used to size fill-list
        // channels and to bound the de-interleave (C++ `fillChannelWithZeroes`
        // / the decode scatter use the same total).
        let total_samples: u64 = per_pixel.iter().map(|&c| c as u64).sum();

        // Channel-relative pixel offset at which this line's samples begin in
        // the flat per-channel heap = Σ samples of all lines before `y` (raster
        // order across the whole image).  Mirrors the writer's
        // `samples_before_line`, so the heap stays globally contiguous per
        // channel and the round-trip is byte-exact.
        let samples_before = self.samples_before_line(sample_counts, y)?;

        // Verify the caller's sample-count buffer BEFORE scattering
        // into `sample_data` — a mismatch means the caller did not run
        // `readPixelSampleCounts` first / sized the heap wrong, and checking
        // first avoids a confusing partial write before the error surfaces.
        self.verify_counts(y, &per_pixel, sample_counts)?;

        // De-interleave channel-by-channel in *file channel order* so the read
        // cursor matches the writer's gather order.  Build that order from the
        // header channel list, mapping each present channel to its frame-buffer
        // slice (skip channels absent from the frame buffer — their bytes must
        // still be consumed).
        let channels = self.header.channels()?;
        let mut read_offset = 0usize;
        for (name, ch) in channels.iter() {
            // C++ deep gather skips lines where modp(y, ySampling) != 0; deep
            // x-sampling is always 1.  Look up the matching frame-buffer slice.
            let slice = self
                .slices
                .iter()
                .find(|s| &s.name == name && !s.fill);
            let sample_size = pixel_type_size(ch.type_)?;

            match slice {
                None => {
                    // Channel present in the file but not requested by the frame
                    // buffer: consume its bytes (total_samples of them) without
                    // scattering.  (modp skip only matters for ySampling != 1,
                    // which deep channels never have.)
                    if modp(y, ch.y_sampling) != 0 {
                        continue;
                    }
                    read_offset += total_samples as usize * sample_size;
                }
                Some(slice) => {
                    if modp(y, slice.y_sampling) != 0 {
                        continue;
                    }
                    // Walk pixels, copying each pixel's samples from the chunk
                    // data into the flat heap at base_offset + pixelStart +
                    // i*sampleStride (the inverse of the writer's gather).  The
                    // line's pixels begin at the cumulative offset of all
                    // preceding lines' samples for this channel.
                    // C++ `curchan.user_data_type = fbslice->type` +
                    // `generic_unpack_deep_pointers` (`OpenEXRCore/unpack.c`):
                    // the frame-buffer slice's type need not match the file
                    // channel's type — every HALF/FLOAT/UINT combination is
                    // converted, not raw-copied.  `dst_size` is the
                    // frame-buffer (destination) sample size; `sample_size` is
                    // the file (source) sample size — they may differ.
                    let dst_size = pixel_type_size(slice.type_)?;
                    let mut pixel_start: i64 =
                        samples_before as i64 * slice.sample_stride as i64;
                    for (i, _x) in (self.min_x..=self.max_x).enumerate() {
                        let cnt = *per_pixel.get(i).ok_or_else(|| {
                            ExrError::Input("Per-pixel count index out of range.".into())
                        })? as i64;
                        for s in 0..cnt {
                            let src = read_offset;
                            let src_slice =
                                data.get(src..src + sample_size).ok_or_else(|| {
                                    ExrError::Input(
                                        "Deep sample data underrun during scatter.".into(),
                                    )
                                })?;
                            let dst_off = slice.base_offset
                                + pixel_start
                                + s * slice.sample_stride as i64;
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
                            // Reuse the single checked scalar↔wire converter
                            // (`misc::copy_into_frame_buffer`) for one sample
                            // at a time: `count == 1` in a scratch buffer runs
                            // its loop body exactly once.
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
                        pixel_start += cnt * slice.sample_stride as i64;
                    }
                }
            }
        }

        // Fill list: frame-buffer slices with no file channel.  Their per-pixel
        // arrays are filled with the slice's fill value (C++ `run_fill`),
        // beginning at this line's cumulative channel offset.
        for slice in self.slices.iter().filter(|s| s.fill) {
            if modp(y, slice.y_sampling) != 0 {
                continue;
            }
            fill_line(slice, &per_pixel, samples_before, sample_data)?;
        }

        Ok(())
    }

    /// Number of samples in the scanlines of the current `read_pixels` range
    /// *before* `y` (Σ over y' in `[first line of the call, y)`, all x) — the
    /// channel-relative pixel offset at which line `y`'s samples begin in the
    /// flat per-channel heap. For a whole-window request this is the
    /// writer's `samples_before_line`; a sub-range heap holds only its own
    /// lines (C++ hands `readPixels` per-pixel pointers, so lines outside
    /// the request never matter).
    ///
    /// Sourced from the caller's own `sample_counts` buffer
    /// (already fully populated, independently per chunk, by the mandatory
    /// first `read_pixel_sample_counts` pass) via the growable
    /// [`RowPrefixCache`](crate::deep::sample_table::RowPrefixCache) — never
    /// by re-decoding an earlier chunk's own count-table bytes from the
    /// stream. That old approach (a) made reading line `y` depend on every
    /// earlier chunk being intact (a damaged/missing chunk anywhere
    /// before `y` failed the read), and (b) could leave the stream cursor
    /// positioned at the WRONG chunk: `read_data_line` calls this only after
    /// already seeking to line `y`'s own chunk, so decoding earlier chunks
    /// here would clobber that position before the sample-data read below —
    /// latent unless the cache happened to already be warm. Sourcing from
    /// `sample_counts` touches the stream not at all, for either reason.
    fn samples_before_line(&mut self, sample_counts: &[u8], y: i32) -> Result<u64> {
        self.sample_prefix.before_line(
            sample_counts,
            self.sample_count_base_offset,
            self.sample_count_x_stride,
            self.sample_count_y_stride,
            y,
        )
    }

    /// Verify the caller's stored per-pixel counts match the chunk's counts —
    /// the safe-Rust guard that replaces the C++ assumption that the user
    /// allocated arrays sized exactly by the prior `readPixelSampleCounts`.
    fn verify_counts(&self, y: i32, per_pixel: &[u32], sample_counts: &[u8]) -> Result<()> {
        for (i, x) in (self.min_x..=self.max_x).enumerate() {
            // A negative/out-of-range count address is a hard error, not
            // silently skipped, matching the crate's hard-error policy for
            // bad count addresses (`sample_count_at`'s own rule) — this is a Rust-only guard with
            // no C++ equivalent, so it must be at least as strict as the rest
            // of the crate, not more lenient.
            let off = count_byte_offset(
                self.sample_count_base_offset,
                self.sample_count_x_stride,
                self.sample_count_y_stride,
                x,
                y,
            )
            .ok_or_else(|| {
                ExrError::Arg(format!(
                    "Sample-count buffer address for ({x},{y}) is out of range (negative)."
                ))
            })?;
            let b = sample_counts.get(off..off + SIZE_INT).ok_or_else(|| {
                ExrError::Arg(format!(
                    "Sample-count buffer address {off} for ({x},{y}) is out of range of the \
                     count buffer (len {}).",
                    sample_counts.len()
                ))
            })?;
            let stored = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let expected = *per_pixel.get(i).unwrap_or(&0);
            if stored != expected {
                return Err(ExrError::Arg(format!(
                    "Sample-count mismatch at ({x},{y}): the buffer says {stored} \
                     but the chunk holds {expected}; call readPixelSampleCounts first."
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure per-line pipeline functions (deep scanline reader)
// ---------------------------------------------------------------------------

/// One scanline's fetched-but-not-decoded deep chunk: the count table
/// already decoded (needed to know how many packed data bytes exist and to
/// address the scatter), the still-possibly-compressed sample DATA bytes,
/// and that chunk's declared unpacked size. Built by
/// [`DeepScanLineInputFileState::fetch_line`] on the calling thread; consumed
/// by [`decode_and_scatter_line`] on any thread.
struct RawDeepLine {
    y: i32,
    per_pixel: Vec<u32>,
    packed_data: Vec<u8>,
    unpacked_size: usize,
}

/// Decode one [`RawDeepLine`]'s sample data and scatter it into `dests`
/// -- ONE [`crate::misc::ChunkDest`] PER SLICE, aligned 1:1 (same length,
/// same order) with `slices` (from [`crate::misc::plan_disjoint`], fed
/// [`deep_line_write_interval`]'s per-slice ranges). This is the SAME
/// logic [`DeepScanLineInputFileState::read_data_line`]'s tail runs,
/// parameterised so the serial and parallel paths stay a single source of
/// truth.
///
/// One destination per slice, not a single combined one: deep's
/// `sample_data` heap is per-channel-contiguous, so a single destination
/// cannot represent "this line owns these two disjoint pieces of the
/// heap" (see `deep_line_write_interval`'s doc for the full story). Each
/// slice writes into its OWN `dests[slice_index].piece`, addressed
/// relative to its OWN `dests[slice_index].base`.
///
/// Takes the compressor BY REFERENCE
/// instead of building one from `compression`/`header` internally -- this
/// runs from several rayon threads at once, so each worker checks its OWN
/// compressor out of the call site's `CodecPool` (reused across lines, not
/// built per line), rather than every call
/// constructing a fresh one.
#[allow(clippy::too_many_arguments)]
fn decode_and_scatter_line(
    compressor: Option<&mut Box<dyn Compressor>>,
    header: &Header,
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    raw: &RawDeepLine,
    samples_before: u64,
    dests: &mut [crate::misc::ChunkDest],
) -> Result<()> {
    let y = raw.y;
    let packed_data_usize = raw.packed_data.len();
    let unpacked_usize = raw.unpacked_size;

    let data: Vec<u8> = if packed_data_usize == unpacked_usize {
        raw.packed_data.clone()
    } else if let Some(c) = compressor {
        let decoded = c.uncompress(&raw.packed_data, y, unpacked_usize).map_err(crate::error::decode_err)?;
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

    let total_samples: u64 = raw.per_pixel.iter().map(|&c| c as u64).sum();
    let channels = header.channels()?;
    let mut read_offset = 0usize;
    for (name, ch) in channels.iter() {
        let slice_idx = slices.iter().position(|s| &s.name == name && !s.fill);
        let sample_size = pixel_type_size(ch.type_)?;

        match slice_idx {
            None => {
                if modp(y, ch.y_sampling) != 0 {
                    continue;
                }
                read_offset += total_samples as usize * sample_size;
            }
            Some(idx) => {
                let slice = &slices[idx];
                if modp(y, slice.y_sampling) != 0 {
                    continue;
                }
                let dst_size = pixel_type_size(slice.type_)?;
                let dest = dests.get_mut(idx).ok_or_else(|| {
                    ExrError::Base("internal error: no destination for this slice index".into())
                })?;
                let mut pixel_start: i64 = samples_before as i64 * slice.sample_stride as i64;
                for (i, _x) in (min_x..=max_x).enumerate() {
                    let cnt = *raw.per_pixel.get(i).ok_or_else(|| {
                        ExrError::Input("Per-pixel count index out of range.".into())
                    })? as i64;
                    for s in 0..cnt {
                        let src = read_offset;
                        let src_slice = data.get(src..src + sample_size).ok_or_else(|| {
                            ExrError::Input("Deep sample data underrun during scatter.".into())
                        })?;
                        let dst_off_abs = slice.base_offset + pixel_start + s * slice.sample_stride as i64;
                        if dst_off_abs < 0 {
                            return Err(ExrError::Arg(
                                "Deep sample address is out of range (negative).".into(),
                            ));
                        }
                        let dst_off = local_deep_addr(dst_off_abs as usize, dest.base)?;
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
                    pixel_start += cnt * slice.sample_stride as i64;
                }
            }
        }
    }

    for (idx, slice) in slices.iter().enumerate() {
        if !slice.fill || modp(y, slice.y_sampling) != 0 {
            continue;
        }
        let dest = dests.get_mut(idx).ok_or_else(|| {
            ExrError::Base("internal error: no destination for this slice index".into())
        })?;
        fill_line_into(slice, &raw.per_pixel, samples_before, dest.piece, dest.base)?;
    }

    Ok(())
}

/// `abs_addr - base`, erroring instead of underflowing when a caller passes
/// an address before its own piece's base — see the flat scanline reader's
/// identical helper for the full rationale.
fn local_deep_addr(abs_addr: usize, base: usize) -> Result<usize> {
    abs_addr.checked_sub(base).ok_or_else(|| {
        ExrError::Base(format!(
            "internal error: scatter address {abs_addr} precedes its destination \
             piece's base {base}"
        ))
    })
}

/// Compute the EXACT byte range line `y`'s [`decode_and_scatter_line`] call
/// would write for EACH slice into the sample-data heap, WITHOUT writing
/// anything — the input to [`crate::misc::plan_disjoint`]. Returns one
/// range PER SLICE, aligned 1:1 (same length, same order) with `slices`
/// (an inactive slice for this row -- `y_sampling` skip, or `total == 0`
/// -- gets an empty `0..0` placeholder, matching `plan_disjoint`'s "empty
/// range never overlaps, gets an empty piece" contract).
///
/// Why not ONE range merged across every slice via a `lo`/`hi` union: that
/// is wrong for any multi-channel deep image: `sample_data` is per-channel-contiguous (all
/// of channel A's samples, then all of B's, ...), so a single range
/// spanning from a line's slice in A's region to its slice in B's region
/// necessarily contains every OTHER line's A and B data sitting in
/// between -- guaranteed overlap with every other line, for any image with
/// 2+ active channels. Returning one range PER SLICE (matched to its own
/// [`crate::misc::ChunkDest`] by `plan_disjoint`) avoids this: two
/// different channels' regions never need to be checked against each
/// other for THIS line at all, since `plan_disjoint`'s flattened check
/// naturally allows non-overlapping ranges from different channels to
/// interleave with other lines' same-channel ranges however they like.
fn deep_line_write_interval(
    slices: &[InSliceInfo],
    y: i32,
    per_pixel: &[u32],
    samples_before: u64,
) -> Result<crate::misc::RangeSet> {
    let total: i64 = per_pixel.iter().map(|&c| c as i64).sum();
    let mut out = crate::misc::RangeSet::with_capacity(slices.len());
    for slice in slices {
        if total == 0 || modp(y, slice.y_sampling) != 0 {
            out.push(0..0);
            continue;
        }
        let dst_size = pixel_type_size(slice.type_)?;
        let start = slice.base_offset + samples_before as i64 * slice.sample_stride as i64;
        // EXACT last-sample address, not `start + total*stride` (one full
        // stride PAST the last real sample -- the old, deliberately-
        // conservative bound, itself padded by `dst_size` on top). `total
        // >= 1` here (the `total == 0` case is handled above), so `total -
        // 1` is the last sample's own index; `far` is that sample's own
        // start address, exactly mirroring `start`'s role for the first
        // sample. Works for either stride sign: `min`/`max` below pick
        // whichever of `start`/`far` is the true low/high end regardless of
        // direction.
        let far = start + (total - 1) * slice.sample_stride as i64;
        if start < 0 || far < 0 {
            return Err(ExrError::Arg(
                "Deep sample address is out of range (negative).".into(),
            ));
        }
        let lo = (start.min(far)) as usize;
        let hi = (start.max(far)) as usize + dst_size;
        out.push(lo..hi);
    }
    Ok(out)
}

/// Fill one fill-list channel's per-pixel arrays with its fill value — port of
/// the C++ `run_fill` per-pixel fill loop, in the flat-heap model.
///
/// Writes `count(x, y)` copies of the fill value into each pixel's array at
/// `base_offset + pixelStart + i*sampleStride`, converting the `f64` fill value
/// to the channel type as C++ does (`(unsigned)`, `half(...)`, `float(...)`).
fn fill_line(
    slice: &InSliceInfo,
    per_pixel: &[u32],
    samples_before: u64,
    sample_data: &mut [u8],
) -> Result<()> {
    fill_line_into(slice, per_pixel, samples_before, sample_data, 0)
}

/// [`fill_line`], generalised with `dest_base` — see the flat scanline
/// reader's `scatter_fill_into` for why (`dest` may be the whole heap,
/// `dest_base = 0`, or one exclusive [`crate::misc::ChunkDest`] piece).
fn fill_line_into(
    slice: &InSliceInfo,
    per_pixel: &[u32],
    samples_before: u64,
    sample_data: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    let sample_size = pixel_type_size(slice.type_)?;
    let mut pixel_start: i64 = samples_before as i64 * slice.sample_stride as i64;
    for &cnt in per_pixel.iter() {
        for s in 0..cnt as i64 {
            let dst_off = slice.base_offset + pixel_start + s * slice.sample_stride as i64;
            if dst_off < 0 {
                return Err(ExrError::Arg(
                    "Deep fill address is out of range (negative).".into(),
                ));
            }
            let dst_off = local_deep_addr(dst_off as usize, dest_base)?;
            let dst = sample_data
                .get_mut(dst_off..dst_off + sample_size)
                .ok_or_else(|| {
                    ExrError::Arg(format!(
                        "Deep fill address {dst_off} out of range of the sample data buffer."
                    ))
                })?;
            write_fill_value(dst, slice.type_, slice.fill_value)?;
        }
        pixel_start += cnt as i64 * slice.sample_stride as i64;
    }
    Ok(())
}

/// Write one fill sample (Xdr LE) of `type_` from an `f64` value — mirrors the
/// per-type cast in C++ `run_fill` (`(unsigned)`, `half`, `float`).  Returns an
/// error for the `NumPixelTypes` sentinel (C++ `run_fill` default branch:
/// `throw ArgExc("Unknown pixel data type.")`).
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
// Tests — write → read round-trip (the key correctness gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::deep::frame_buffer::DeepSlice;
    use crate::deep::DeepScanLineOutputFile;
    use crate::frame_buffer::Slice;
    use crate::io::{MemIStream, MemOStream};
    use imath_rs::{Half, V2f};

    /// A deep image description: per-pixel counts (raster order) plus, for each
    /// channel, the sample values laid out contiguously per pixel.
    struct DeepImage {
        w: i32,
        h: i32,
        min_x: i32,
        min_y: i32,
        counts: Vec<u32>,
        /// channel name → (PixelType, flat sample bytes in raster/contiguous order)
        channels: Vec<(String, PixelType)>,
        /// per channel: the f64 value of each sample (raster order, contiguous)
        values: Vec<Vec<f64>>,
    }

    /// Build a header with the given window origin, dimensions, channels, and
    /// compression, deep type + line order.
    fn make_header(
        img: &DeepImage,
        comp: Compression,
        line_order: LineOrder,
    ) -> Header {
        // Header::with_dimensions makes a [0,0]-origin window; for a non-origin
        // window we set the data/display windows explicitly afterwards.
        let mut h = Header::with_dimensions(
            img.w,
            img.h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            line_order,
            comp,
        )
        .unwrap();
        if img.min_x != 0 || img.min_y != 0 {
            let dw = imath_rs::Box2i {
                min: imath_rs::V2i { x: img.min_x, y: img.min_y },
                max: imath_rs::V2i {
                    x: img.min_x + img.w - 1,
                    y: img.min_y + img.h - 1,
                },
            };
            *h.data_window_mut().unwrap() = dw;
        }
        let mut cl = ChannelList::new();
        for (name, ty) in &img.channels {
            cl.insert(name, Channel::new(*ty, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.set_type(DEEP_SCANLINE).unwrap();
        h
    }

    /// Flat sample-count buffer (u32 LE per pixel) addressed in **absolute**
    /// window coordinates: `off = y*yStride + x*xStride`, with `xStride = 4`,
    /// `yStride = 4*w`, and `base_offset = 0` on the slice (neither writer nor
    /// reader applies a base to the sample-count slice, so the buffer is sized
    /// to the absolute extent `(min_y + h) * yStride`).  Returns
    /// `(buffer, xStride, yStride)`.
    fn build_count_buffer(img: &DeepImage) -> (Vec<u8>, i32, i32) {
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
    /// channel contiguous in raster order), returning the heap and each
    /// channel's base offset into it.
    fn build_sample_heap(img: &DeepImage) -> (Vec<u8>, Vec<i64>) {
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

    /// Total samples in the image.
    fn total_samples(img: &DeepImage) -> usize {
        img.counts.iter().map(|&c| c as usize).sum()
    }

    /// Make the writer's DeepFrameBuffer (flat model): sample-count slice +
    /// one deep slice per channel pointing at its base in the heap.
    fn make_frame_buffer(
        img: &DeepImage,
        bases: &[i64],
        x_stride: i32,
        y_stride: i32,
    ) -> DeepFrameBuffer {
        let mut fb = DeepFrameBuffer::new();
        // Sample-count slice: absolute addressing, base_offset = 0 (neither
        // writer nor reader applies a base to the sample-count slice).
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

    /// Write `img` to an in-memory deep EXR via the deep scanline writer.
    fn write_deep(img: &DeepImage, comp: Compression, line_order: LineOrder) -> Vec<u8> {
        let header = make_header(img, comp, line_order);
        let (sc, x_stride, y_stride) = build_count_buffer(img);
        let (heap, bases) = build_sample_heap(img);
        let fb = make_frame_buffer(img, &bases, x_stride, y_stride);

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &heap, img.h).unwrap();
        }
        out.into_data()
    }

    /// A small image: w x h, channels HALF "A" + FLOAT "Z", per-pixel counts
    /// given, sample value = channel_base + pixel_index*10 + sample_index.
    fn sample_image(w: i32, h: i32, min_x: i32, min_y: i32, counts: Vec<u32>) -> DeepImage {
        let channels = vec![("A".to_string(), PixelType::Half), ("Z".to_string(), PixelType::Float)];
        let n = (w * h) as usize;
        assert_eq!(counts.len(), n);
        let mut values = vec![Vec::new(), Vec::new()];
        for (ci, base) in [1000.0f64, 5000.0f64].into_iter().enumerate() {
            for (pix, &c) in counts.iter().enumerate() {
                for s in 0..c {
                    values[ci].push(base + (pix as f64) * 10.0 + s as f64);
                }
            }
        }
        DeepImage { w, h, min_x, min_y, counts, channels, values }
    }

    /// Read back the deep file two-pass and assert counts + every sample match.
    fn round_trip(img: &DeepImage, comp: Compression, line_order: LineOrder) {
        let bytes = write_deep(img, comp, line_order);
        read_and_assert(img, bytes, comp, line_order);
    }

    /// Two-pass read of `bytes` + full comparison against `img`.
    fn read_and_assert(img: &DeepImage, bytes: Vec<u8>, comp: Compression, line_order: LineOrder) {
        let (_sc_written, x_stride, y_stride) = build_count_buffer(img);
        let (_heap_written, bases) = build_sample_heap(img);
        let fb = make_frame_buffer(img, &bases, x_stride, y_stride);

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().type_().unwrap(), DEEP_SCANLINE);
        inf.set_frame_buffer(&fb).unwrap();

        // Pass 1: counts into a fresh zeroed count buffer (absolute extent).
        let (template, x_stride2, y_stride2) = build_count_buffer(img);
        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf)
            .unwrap();

        // Verify per-pixel counts match exactly (absolute addressing).
        let n = (img.w * img.h) as usize;
        for i in 0..n {
            let x = img.min_x + (i as i32 % img.w);
            let y = img.min_y + (i as i32 / img.w);
            let off =
                (y as i64 * y_stride2 as i64 + x as i64 * x_stride2 as i64) as usize;
            let got = u32::from_le_bytes(counts_buf[off..off + 4].try_into().unwrap());
            assert_eq!(got, img.counts[i], "count mismatch at raster {i}");
        }

        // Pass 2: sample data into a fresh zeroed heap sized like the writer's.
        let mut heap = vec![0u8; total_heap_bytes(img)];
        inf.read_pixels(
            img.min_y,
            img.min_y + img.h - 1,
            &counts_buf,
            &mut heap,
        )
        .unwrap();

        // Compare every sample of every channel bit-exact against the source.
        let (expected_heap, _bases) = build_sample_heap(img);
        assert_eq!(heap, expected_heap, "deep sample data mismatch for {comp:?}/{line_order:?}");
    }

    /// Total heap bytes (all channels, every sample).
    fn total_heap_bytes(img: &DeepImage) -> usize {
        // Build the heap once to get its exact length (handles ordering).
        build_sample_heap(img).0.len()
    }

    /// A deep file whose chunk offset table was never written (an
    /// interrupted write) must still read — the table is reconstructed by
    /// scanning the deep chunks (C++ `chunk.c::reconstruct_chunk_table`).
    #[test]
    fn zeroed_offset_table_is_reconstructed() {
        for comp in [Compression::NoCompression, Compression::Zips] {
            let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
            let mut bytes = write_deep(&img, comp, LineOrder::IncreasingY);

            // The table ends exactly where the first chunk begins.
            let (table_pos, count) = {
                let mut is = MemIStream::new(bytes.clone());
                let inf = DeepScanLineInputFile::new(&mut is).unwrap();
                let count = inf.state.line_offsets.len();
                let first = inf.state.line_offsets.iter().copied().min().unwrap() as usize;
                (first - count * 8, count)
            };
            for b in bytes[table_pos..table_pos + count * 8].iter_mut() {
                *b = 0;
            }

            read_and_assert(&img, bytes, comp, LineOrder::IncreasingY);
        }
    }

    /// `is_complete()`: a fully-written file reports complete; a file
    /// whose on-disk offset table was zeroed (and cannot be reconstructed
    /// because the FIRST chunk's own bytes are also gone) reports incomplete.
    #[test]
    fn is_complete_detects_unrecoverable_offset_table() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);

        let mut is = MemIStream::new(bytes.clone());
        let inf = DeepScanLineInputFile::new(&mut is).unwrap();
        assert!(inf.is_complete(), "a fully-written file must report complete");

        // Zero the offset table AND truncate away the first chunk's bytes so
        // the reconstruction sweep has nothing to recover -- leaves at least
        // one offset-table slot at 0.
        let count = inf.state.line_offsets.len();
        let first = inf.state.line_offsets.iter().copied().min().unwrap() as usize;
        let table_pos = first - count * 8;
        let mut truncated = bytes;
        for b in truncated[table_pos..table_pos + count * 8].iter_mut() {
            *b = 0;
        }
        truncated.truncate(first);

        let mut is2 = MemIStream::new(truncated);
        let inf2 = DeepScanLineInputFile::new(&mut is2).unwrap();
        assert!(
            !inf2.is_complete(),
            "a file with an unreconstructable offset table must report incomplete"
        );
    }

    #[test]
    fn round_trip_none_increasing() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::NoCompression, LineOrder::IncreasingY);
    }

    /// Read the SAME multi-line deep scanline file with
    /// `num_threads` in {0, 1, 2, 3, 8, 17} and assert every thread count
    /// produces byte-identical sample data (and matching counts) — the deep
    /// analogue of the flat readers' parallel determinism tests. A
    /// non-uniform per-pixel count pattern (including several zero-count
    /// pixels) exercises the `samples_before`/interval-planner addressing
    /// across chunk boundaries.
    ///
    /// `sample_image` uses TWO channels (A, Z) -- this is the "2+ channel"
    /// half of the required coverage (see `Probe::PlanDisjointAccepted`):
    /// with a single merged write interval per line, this test's parallel
    /// fast path would never engage and would silently exercise only the
    /// serial fallback. The single-channel counterpart is
    /// `parallel_read_single_channel_is_byte_identical_across_thread_counts`.
    #[test]
    fn parallel_read_is_byte_identical_to_serial_across_thread_counts() {
        let (w, h) = (9i32, 40i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| i % 4).collect();
        let img = sample_image(w, h, 0, 0, counts);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);

        let (_sc_written, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap_written, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, x_stride, y_stride);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepScanReadParallel);
            let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf)
                .unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap.len()];
            inf.read_pixels(img.min_y, img.min_y + img.h - 1, &counts_buf, &mut heap)
                .unwrap();
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::DeepScanReadParallel, n);
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

    /// Single-channel counterpart of
    /// `parallel_read_is_byte_identical_to_serial_across_thread_counts` --
    /// see `Probe::PlanDisjointAccepted` for why BOTH the
    /// 1-channel and 2+-channel cases need their own asserted coverage
    /// (the multi-channel bug this whole follow-up fixed would not have
    /// been caught by a single-channel-only test).
    #[test]
    fn parallel_read_single_channel_is_byte_identical_across_thread_counts() {
        let (w, h) = (9i32, 40i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| i % 4).collect();
        let n_total = (w * h) as usize;
        let mut values = Vec::with_capacity(n_total);
        for (pix, &c) in counts.iter().enumerate() {
            for s in 0..c {
                values.push(1000.0 + pix as f64 * 10.0 + s as f64);
            }
        }
        let img = DeepImage {
            w,
            h,
            min_x: 0,
            min_y: 0,
            counts,
            channels: vec![("A".to_string(), PixelType::Half)],
            values: vec![values],
        };
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);

        let (_sc_written, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap_written, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, x_stride, y_stride);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepScanReadParallel);
            let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf)
                .unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap.len()];
            inf.read_pixels(img.min_y, img.min_y + img.h - 1, &counts_buf, &mut heap)
                .unwrap();
            if n > 1 {
                crate::probe::assert_engaged(crate::probe::Probe::DeepScanReadParallel, n);
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

    /// Negative test: a genuinely
    /// ALIASED frame buffer -- two different channels' slices pointing
    /// into OVERLAPPING memory (a real caller mistake, not something the
    /// file format itself could cause) -- must still produce a correct,
    /// deterministic result via the serial fallback, never a data race or
    /// corruption from the parallel path. `plan_disjoint` must reject this
    /// (its ranges genuinely overlap, this is not a bug to fix), so
    /// `Probe::PlanDisjointAccepted` must stay at 0 for every `num_threads`,
    /// including values that WOULD otherwise engage the fast path.
    #[test]
    fn parallel_read_rejects_genuinely_aliased_slices_falls_back_to_serial() {
        let (w, h) = (9i32, 20i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| i % 4).collect();
        let img = sample_image(w, h, 0, 0, counts);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);

        let (_sc_written, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap_written, real_bases) = build_sample_heap(&img);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap_len = build_sample_heap(&img).0.len();

        // ALIASED frame buffer: both channels ("A" and "Z") point at the
        // SAME base offset -- their write ranges genuinely overlap for
        // every line with a non-zero sample count in both channels.
        let aliased_bases = vec![real_bases[0], real_bases[0]];
        let fb = make_frame_buffer(&img, &aliased_bases, x_stride, y_stride);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepScanReadParallel);
            let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf).unwrap();

            crate::probe::reset(crate::probe::Probe::PlanDisjointAccepted);
            let mut heap = vec![0u8; expected_heap_len];
            inf.read_pixels(img.min_y, img.min_y + img.h - 1, &counts_buf, &mut heap).unwrap();
            crate::probe::assert_declined(crate::probe::Probe::DeepScanReadParallel);
            assert_eq!(
                crate::probe::get(crate::probe::Probe::PlanDisjointAccepted),
                0,
                "num_threads={n}: plan_disjoint must reject a genuinely aliased frame buffer \
                 (never engage the parallel fast path on it), not attempt an unsafe parallel scatter"
            );

            // Every thread count must still agree with each other -- the
            // serial fallback is deterministic even when two channels'
            // writes land on the same bytes (last-channel-in-file wins,
            // same as the equivalent fully-serial `read_data_line` loop
            // would produce, since num_threads never changes execution
            // order within the serial path).
            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from num_threads=0 -- the aliased-write outcome must still be deterministic"),
            }
        }
    }

    /// Proves a `NUM_LINES`-line `read_pixels` call does not construct one
    /// data compressor per LINE. This exercises the SERIAL path
    /// (`read_data_line` -> persistent `self.data_compressor`); the parallel
    /// path's codecs come from a `CodecPool`.
    #[test]
    fn read_pixels_builds_few_compressors_not_one_per_line() {
        const NUM_LINES: i32 = 200;
        // Wide enough (300 px, up to 5 samples/pixel) that each line's
        // packed sample-data bytes are large enough for ZIPS to genuinely
        // shrink them -- a narrow/small per-line payload hits zlib's own
        // overhead and takes the "no-shrink -> store raw" path for EVERY
        // line, which would make this test vacuously pass regardless of
        // whether `new_compressor` is called once or once per line, since
        // the data compressor would never be constructed at all.
        let (w, h) = (300i32, NUM_LINES);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = sample_image(w, h, 0, 0, counts);
        let bytes = write_deep(&img, Compression::Zips, LineOrder::IncreasingY);
        let raw_data_bytes: usize = img.counts.iter().map(|&c| c as usize * (2 + 4)).sum();
        assert!(
            bytes.len() < raw_data_bytes,
            "test setup: the written file ({} bytes) must be smaller than the raw \
             sample data ({raw_data_bytes} bytes) -- otherwise ZIPS never actually \
             compresses any line and this test can't distinguish per-line from \
             persistent-field compressor construction",
            bytes.len()
        );

        let (_sc_written, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap_written, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, x_stride, y_stride);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut is = MemIStream::new(bytes.clone());
        // Pin the serial path; the per-thread probe only sees construction on
        // this thread, so a global thread count > 1 would pass vacuously.
        let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(0)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let mut counts_buf = vec![0u8; template.len()];
        inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf).unwrap();

        crate::probe::reset(crate::probe::Probe::NewCompressor);
        let mut heap = vec![0u8; expected_heap.len()];
        inf.read_pixels(img.min_y, img.min_y + img.h - 1, &counts_buf, &mut heap).unwrap();
        let delta = crate::probe::get(crate::probe::Probe::NewCompressor);

        assert_eq!(heap, expected_heap, "sanity: the read must still be correct");
        assert!(
            delta < (NUM_LINES / 4) as u64,
            "read_pixels for {NUM_LINES} lines called new_compressor {delta} times -- \
             expected far fewer than one per line (persistent-field regression?)"
        );
    }

    /// Same matrix over ZIPS (multi-line chunks are NOT a thing for deep —
    /// every deep chunk is always exactly one scanline — but a real codec
    /// still exercises the per-worker fresh-compressor requirement for both
    /// the count-table and sample-data decompressors).
    #[test]
    fn parallel_read_zips_matches_across_thread_counts() {
        let (w, h) = (7i32, 30i32);
        let counts: Vec<u32> = (0..(w * h) as u32).map(|i| (i * 3) % 5).collect();
        let img = sample_image(w, h, 0, 0, counts);
        let bytes = write_deep(&img, Compression::Zips, LineOrder::IncreasingY);

        let (_sc_written, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap_written, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, x_stride, y_stride);
        let (template, ..) = build_count_buffer(&img);
        let expected_heap = build_sample_heap(&img).0;

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::DeepScanReadParallel);
            let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&fb).unwrap();

            let mut counts_buf = vec![0u8; template.len()];
            inf.read_pixel_sample_counts(img.min_y, img.min_y + img.h - 1, &mut counts_buf)
                .unwrap();

            let mut heap = vec![0u8; expected_heap.len()];
            inf.read_pixels(img.min_y, img.min_y + img.h - 1, &counts_buf, &mut heap)
                .unwrap();

            crate::probe::assert_engaged(crate::probe::Probe::DeepScanReadParallel, n);
            assert_eq!(heap, expected_heap, "num_threads={n}: mismatch vs the written sample data");
            match &reference {
                None => reference = Some(heap),
                Some(r) => assert_eq!(*r, heap, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    #[test]
    fn round_trip_zips_increasing() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::Zips, LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_rle_increasing() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::Rle, LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_none_decreasing() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::NoCompression, LineOrder::DecreasingY);
    }

    #[test]
    fn round_trip_zips_decreasing() {
        let img = sample_image(4, 3, 0, 0, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::Zips, LineOrder::DecreasingY);
    }

    #[test]
    fn round_trip_non_origin_window() {
        // Data window not at the origin: min_x = -3, min_y = 7.
        let img = sample_image(4, 3, -3, 7, vec![0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 1, 2]);
        round_trip(&img, Compression::Zips, LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_all_zero_counts() {
        // Every pixel has zero samples: counts all zero, empty sample heap.
        let img = sample_image(3, 2, 0, 0, vec![0, 0, 0, 0, 0, 0]);
        round_trip(&img, Compression::Zips, LineOrder::IncreasingY);
    }

    #[test]
    fn absent_channel_is_filled() {
        // File has A (HALF) and Z (FLOAT); the read frame buffer adds a fill
        // channel F (FLOAT, fillValue 9.0) that the file does not provide.
        let img = sample_image(3, 2, 0, 0, vec![1, 2, 0, 3, 1, 2]);
        let bytes = write_deep(&img, Compression::Zips, LineOrder::IncreasingY);

        let (_sc, x_stride, y_stride) = build_count_buffer(&img);
        let (_heap, bases) = build_sample_heap(&img);
        let mut fb = make_frame_buffer(&img, &bases, x_stride, y_stride);
        // F's samples live in their own heap region after A and Z.
        let f_base = build_sample_heap(&img).0.len() as i64;
        fb.insert(
            "F",
            DeepSlice::new(PixelType::Float, f_base, 8, 8 * img.w as usize, 4, 1, 1, 9.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let n = (img.w * img.h) as usize;
        let mut counts_buf = vec![0u8; n * 4];
        inf.read_pixel_sample_counts(0, img.h - 1, &mut counts_buf).unwrap();

        let total = total_samples(&img);
        // Heap: A+Z region then F region (one FLOAT per sample).
        let mut heap = vec![0u8; build_sample_heap(&img).0.len() + total * 4];
        inf.read_pixels(0, img.h - 1, &counts_buf, &mut heap).unwrap();

        // Every F sample must equal the fill value 9.0.
        for s in 0..total {
            let off = f_base as usize + s * 4;
            let v = f32::from_le_bytes(heap[off..off + 4].try_into().unwrap());
            assert_eq!(v, 9.0, "fill sample {s} mismatch");
        }
    }

    #[test]
    fn no_frame_buffer_errors() {
        let img = sample_image(3, 2, 0, 0, vec![1, 1, 1, 1, 1, 1]);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        let mut counts = vec![0u8; 6 * 4];
        let err = inf.read_pixel_sample_counts(0, 1, &mut counts).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn out_of_range_scanline_errors() {
        let img = sample_image(3, 2, 0, 0, vec![1, 1, 1, 1, 1, 1]);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 6 * 4];
        let err = inf.read_pixel_sample_counts(0, 99, &mut counts).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn rejects_non_deep_file() {
        // A plain scanline (flat) file must be rejected by the deep reader.
        use crate::scanline::OutputFile;
        let mut hdr = Header::with_dimensions(
            3, 2, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let pixels = vec![0u8; 3 * 2 * 2];
        let mut fb = crate::frame_buffer::FrameBuffer::new();
        fb.insert("R", Slice {
            type_: PixelType::Half, base_offset: 0, x_stride: 2, y_stride: 6,
            x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
        }).unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &hdr).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        // `new` returns the reader (which is not Debug) on success, so match
        // explicitly rather than `unwrap_err` (which would require T: Debug).
        match DeepScanLineInputFile::new(&mut is) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("deep reader accepted a flat scanline file"),
        }
    }

    // -- Allocation-guard rejection paths (malformed chunk size fields) -----

    /// Byte offset of the (single) chunk in a 1x1 deep scanline file: parse the
    /// header length, then read the first line-offset slot.  Chunk field layout
    /// from that offset: y(i32) @ +0, packedCountSize(u64) @ +4,
    /// packedDataSize(u64) @ +12, unpackedDataSize(u64) @ +20.
    fn first_chunk_offset(bytes: &[u8]) -> usize {
        let mut is = MemIStream::new(bytes.to_vec());
        let (_h, _v) = Header::read_from(&mut is).unwrap();
        let header_len = is.tellg().unwrap() as usize;
        u64::from_le_bytes(bytes[header_len..header_len + 8].try_into().unwrap()) as usize
    }

    /// A count-table `packedSampleCountTableSize` above the writer's bound (here
    /// `u64::MAX`, also exercising the near-`u64::MAX` allocation path) must be
    /// rejected with `ExrError::Input` and no large allocation / panic.
    #[test]
    fn rejects_oversize_count_table() {
        let img = sample_image(1, 1, 0, 0, vec![2]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        bytes[off + 4..off + 12].copy_from_slice(&u64::MAX.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 4];
        match inf.read_pixel_sample_counts(0, 0, &mut counts) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// Pass 1 (counts only) of a deep chunk whose sizes are corrupt: `unpacked_at`
    /// patches `unpackedDataSize`, `count` patches the 1x1 image's one cumulative
    /// count. Returns the `read_pixel_sample_counts` result.
    fn read_counts_of_patched(unpacked_at: Option<u64>, count: Option<i32>) -> Result<()> {
        let img = sample_image(1, 1, 0, 0, vec![2]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        if let Some(u) = unpacked_at {
            bytes[off + 20..off + 28].copy_from_slice(&u.to_le_bytes());
        }
        // Chunk leader: y (4) + three u64 sizes (24), then the raw (NONE) count table.
        if let Some(c) = count {
            bytes[off + 28..off + 32].copy_from_slice(&c.to_le_bytes());
        }
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 4];
        inf.read_pixel_sample_counts(0, 0, &mut counts)
    }

    /// An `unpackedDataSize` that disagrees with `packedDataSize` under NONE is a
    /// corrupt chunk leader: C++ (`OpenEXRCore/chunk.c`) rejects it before any
    /// decode, so a counts-only read fails too.
    #[test]
    fn rejects_unpacked_size_mismatch() {
        assert!(read_counts_of_patched(None, None).is_ok());
        match read_counts_of_patched(Some(999), None) {
            Err(ExrError::Input(m)) => assert!(m.contains("uncompressed"), "{m}"),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// A monotonic count table that claims more samples than the chunk's data
    /// holds (2 -> 3 samples of 6 bytes, unpacked 12) is C++'s "Corrupt sample
    /// count table" (`OpenEXRCore/decoding.c:270-275`), raised whenever counts are
    /// decoded -- `readPixelSampleCounts` included, so `exrcheck -c` calls such a
    /// file bad. We used to check it only when reading the sample data.
    #[test]
    fn counts_only_read_rejects_a_table_larger_than_the_data() {
        match read_counts_of_patched(None, Some(3)) {
            Err(ExrError::Input(m)) => assert!(m.contains("implied by the decoded sample counts"), "{m}"),
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// A `packedDataSize` larger than `unpackedDataSize` (here `u64::MAX`, also
    /// the near-`u64::MAX` allocation path) must be rejected before allocating.
    ///
    /// The rejection happens at the FIRST chunk access
    /// (`read_pixel_sample_counts`), not only at `read_pixels` — C++'s own
    /// leader validation (`OpenEXRCore/chunk.c::
    /// read_and_validate_chunk_leader`) is ported into `seek_chunk`, the ONE
    /// place all three deep chunk leader sizes are parsed, matching C++:
    /// `readPixelSampleCounts` and `readPixels` both call the SAME
    /// `exr_read_scanline_chunk_info` (`ImfDeepScanLineInputFile.cpp:526/547`
    /// vs `:638`), which performs this exact leader validation
    /// unconditionally (`chunk.c:854-1066`) — a corrupt `packedDataSize`
    /// therefore fails on the first pass in real OpenEXR too, not just the
    /// second.
    #[test]
    fn rejects_packed_data_over_unpacked() {
        let img = sample_image(1, 1, 0, 0, vec![2]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        bytes[off + 12..off + 20].copy_from_slice(&u64::MAX.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let mut counts = vec![0u8; 4];
        match inf.read_pixel_sample_counts(0, 0, &mut counts) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// `raw_pixel_data` (the `copy_pixels` raw-chunk
    /// path) must apply the SAME leader bound as the normal decode
    /// path, not allocate straight from an attacker-controlled on-wire size.
    /// Crafts a chunk whose `packedDataSize` field claims ~16 EiB
    /// (`u64::MAX`) -- if this test completes at all (rather than hanging or
    /// aborting the process on the allocation attempt) with an `Err`, the
    /// bound is working; it must never reach `try_filled_vec`/`vec![]` with
    /// that size.
    #[test]
    fn raw_pixel_data_rejects_oversize_packed_data_before_allocating() {
        let img = sample_image(1, 1, 0, 0, vec![2]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        // Chunk layout: y(i32) @0, packedCount(u64) @4, packedData(u64) @12.
        bytes[off + 12..off + 20].copy_from_slice(&u64::MAX.to_le_bytes());

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        match inf.raw_pixel_data(0) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error (no giant allocation), got {other:?}"),
        }
    }

    /// A HALF file channel read into a FLOAT frame-buffer slice must
    /// convert each sample (`f32::from(Half)`), not raw-copy 2 file bytes
    /// into a 4-byte slot (which would leave the upper bytes holding stale
    /// content instead of the correct value).
    #[test]
    fn converts_half_file_channel_to_float_slice() {
        let img = sample_image(2, 1, 0, 0, vec![1, 2]);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let total = total_samples(&img);

        let (sc, x_stride, y_stride) = build_count_buffer(&img);
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
        // File channel "A" is HALF; read it into a FLOAT slice.
        fb.insert(
            "A",
            DeepSlice::new(PixelType::Float, 0, 8, 8 * img.w as usize, 4, 1, 1, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();
        // Keep "Z" (already FLOAT in the file) in its own heap region so its
        // bytes are consumed rather than skipped.
        fb.insert(
            "Z",
            DeepSlice::new(
                PixelType::Float,
                total as i64 * 4,
                8,
                8 * img.w as usize,
                4,
                1,
                1,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts_buf = vec![0u8; sc.len()];
        inf.read_pixel_sample_counts(0, 0, &mut counts_buf).unwrap();
        let mut heap = vec![0u8; total * 4 * 2];
        inf.read_pixels(0, 0, &counts_buf, &mut heap).unwrap();

        // "A"'s samples were built with base 1000.0 (see `sample_image`).
        let mut expect = Vec::new();
        for (pix, &c) in img.counts.iter().enumerate() {
            for s in 0..c {
                expect.push(1000.0f64 + pix as f64 * 10.0 + s as f64);
            }
        }
        for (i, &e) in expect.iter().enumerate() {
            let off = i * 4;
            let got = f32::from_le_bytes(heap[off..off + 4].try_into().unwrap());
            assert_eq!(got, e as f32, "converted sample {i}");
        }
    }

    /// A crafted chunk whose declared sizes are internally consistent
    /// (matching a huge decoded count table) but exceed `INT_MAX` must be
    /// rejected before the multi-gigabyte allocation is attempted.
    ///
    /// `unpacked`/`packedData` here are set to
    /// `i32::MAX * 6`, over the `i32::MAX` leader bound `seek_chunk`
    /// enforces (see `rejects_packed_data_over_unpacked`'s doc for why C++
    /// validates the leader on EVERY chunk-info read, including
    /// `readPixelSampleCounts`) — pass 1 rejects it, before pass 2
    /// (and its would-be giant allocation) is ever reached.
    #[test]
    fn rejects_internally_consistent_but_oversize_chunk() {
        let img = sample_image(1, 1, 0, 0, vec![0]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        // Chunk layout: y(i32) @0, packedCount(u64) @4, packedData(u64) @12,
        // unpacked(u64) @20, then the raw count table (one Xdr int, width=1).
        let huge_count = i32::MAX; // just under the on-wire cumulative cap
        bytes[off + 24..off + 28].copy_from_slice(&huge_count.to_le_bytes());
        // `unpacked` must equal `total_samples * sum(channel byte sizes)` to
        // pass the consistency check: channels are HALF(2) + FLOAT(4) = 6.
        let expected_unpacked = huge_count as u64 * 6;
        bytes[off + 20..off + 28].copy_from_slice(&expected_unpacked.to_le_bytes());
        bytes[off + 12..off + 20].copy_from_slice(&expected_unpacked.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 4];
        match inf.read_pixel_sample_counts(0, 0, &mut counts) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error (no giant allocation), got {other:?}"),
        }
    }

    /// A compressed count table that packs down to zero bytes (every
    /// pixel has zero samples) is a valid, all-zero row, not a corrupt chunk.
    #[test]
    fn zero_length_count_table_reads_as_zero_counts() {
        let img = sample_image(1, 1, 0, 0, vec![0]);
        let mut bytes = write_deep(&img, Compression::Zips, LineOrder::IncreasingY);
        let off = first_chunk_offset(&bytes);
        // Force packedSampleCountTableSize to 0 (an "absent" count table).
        bytes[off + 4..off + 12].copy_from_slice(&0u64.to_le_bytes());

        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts = vec![0u8; 4];
        inf.read_pixel_sample_counts(0, 0, &mut counts).unwrap();
        assert_eq!(u32::from_le_bytes(counts.try_into().unwrap()), 0);
    }

    /// A fill-list slice (no matching file channel) must still have
    /// sampling (1,1) — C++ `ImfDeepScanLineInputFile.cpp:928-929` throws
    /// `InputExc` "Expect sampling of 1" at fill time; we reject it at
    /// `set_frame_buffer`.
    #[test]
    fn fill_slice_rejects_non_unit_sampling() {
        let img = sample_image(2, 2, 0, 0, vec![1, 1, 1, 1]);
        let bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);
        let (_sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let mut fb = make_frame_buffer(&img, &bases, xs, ys);
        fb.insert(
            "F",
            DeepSlice::new(PixelType::Float, 0, 8, 8 * img.w as usize, 4, 1, 2, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        match inf.set_frame_buffer(&fb) {
            Err(ExrError::Input(_)) => {}
            other => panic!("expected Input error, got {other:?}"),
        }
    }

    /// `DeepScanLineOutputFile::from_part` must reject a header whose
    /// `type` already names a different (non-deepscanline) part.
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
        hdr.set_type(crate::header::SCANLINE_IMAGE).unwrap();

        let mut out = MemOStream::new();
        match crate::deep::DeepScanLineOutputFile::from_part(&mut out, &hdr, None, 0) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("expected a type-mismatch error, got Ok"),
        }
    }

    /// `OpenEXRCore/decoding.c:307-318`: a deep scanline part whose
    /// `version` attribute is not 1 must be rejected.
    #[test]
    fn rejects_non_unit_version_attribute() {
        let img = sample_image(2, 2, 0, 0, vec![1, 1, 1, 1]);
        let mut header = make_header(&img, Compression::NoCompression, LineOrder::IncreasingY);
        header.insert("version", crate::attr::IntAttribute::new(2)).unwrap();

        let mut is = MemIStream::new(Vec::new());
        let stream: &mut dyn IStream = &mut is;
        match DeepScanLineInputFile::from_part(stream, header, 0, None, vec![8]) {
            Err(ExrError::Arg(_)) => {}
            Err(other) => panic!("expected Arg error, got {other:?}"),
            Ok(_) => panic!("expected a version-mismatch error, got Ok"),
        }
    }

    /// Once `read_pixel_sample_counts` has succeeded for a range (per
    /// the documented API contract — it must be called before `read_pixels`),
    /// a later corruption of an EARLIER line's own chunk must not prevent
    /// `read_pixels` from reading a later, uncorrupted line — addressing is
    /// now sourced from the caller's already-populated `sample_counts`
    /// buffer, not by re-decoding the earlier chunk's bytes from the stream.
    #[test]
    fn damaged_earlier_chunk_does_not_block_a_later_line() {
        let img = sample_image(2, 3, 0, 0, vec![1, 2, 0, 1, 3, 1]);
        let mut bytes = write_deep(&img, Compression::NoCompression, LineOrder::IncreasingY);

        let (sc, xs, ys) = build_count_buffer(&img);
        let (_h, bases) = build_sample_heap(&img);
        let fb = make_frame_buffer(&img, &bases, xs, ys);

        // Pass 1 (the whole image) succeeds while the file is still intact —
        // exactly the documented "call read_pixel_sample_counts first" usage.
        let mut counts_buf = vec![0u8; sc.len()];
        {
            let mut is = MemIStream::new(bytes.clone());
            let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
            inf.set_frame_buffer(&fb).unwrap();
            inf.read_pixel_sample_counts(0, 2, &mut counts_buf).unwrap();
        }

        // Now corrupt line 0's own count-table bytes on disk (NoCompression
        // stores the raw cumulative table right after the 28-byte chunk
        // header: y(i32), packedCount/packedData/unpacked(u64 each)).
        // `0xFF` bytes decode as a negative Xdr int, which
        // `decode_cumulative_row` rejects outright.
        let off = first_chunk_offset(&bytes);
        let raw_table_size = img.w as usize * 4;
        for b in bytes[off + 28..off + 28 + raw_table_size].iter_mut() {
            *b = 0xFF;
        }

        // Reading line 1's DATA (a fresh reader on the now-corrupted bytes,
        // reusing the counts pass 1 already established) must still succeed
        // and produce the correct sample data for line 1.
        let mut is = MemIStream::new(bytes);
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut heap = vec![0u8; total_heap_bytes(&img)];
        inf.read_pixels(1, 1, &counts_buf, &mut heap).unwrap();

        // Line 1 covers samples for pixels (0,1) and (1,1) — counts 0 and 1 —
        // verify channel "A"'s one sample matches the source value.
        let expected = 1000.0f64 + 3.0 * 10.0; // pixel index 3 (row 1, col 1), sample 0
        // "A" is channel 0 in the heap; the heap of a one-line request starts
        // at that line, and pixel (0,1) has no samples, so pixel (1,1)'s
        // sample is the first one.
        let a_off = 0;
        let got = f32::from(Half::from_bits(u16::from_le_bytes(
            heap[a_off..a_off + 2].try_into().unwrap(),
        )));
        assert_eq!(got, expected as f32);
    }

    /// Any sub-range `[s1, s2]`, down to a single line, reads into a heap that
    /// holds only its own lines, from a count raster of only its own rows (C++
    /// `readPixels` takes per-pixel pointers, so lines outside the request are
    /// never consulted). The expected bytes come straight from the source
    /// samples, independent of the reader's addressing. A non-origin window
    /// (`min = (0, 7)`), a NONE and a ZIPS file, serial and parallel decode.
    #[test]
    fn sub_range_reads_use_a_range_local_heap() {
        let counts = vec![1, 0, 2, 3, 1, 0, 0, 0, 0, 2, 2, 1, 1, 4, 0];
        for (min_x, min_y) in [(0, 7), (-9, -12), (13, 21)] {
        let img = sample_image(3, 5, min_x, min_y, counts.clone());
        let w = img.w as usize;
        for comp in [Compression::NoCompression, Compression::Zips] {
            let header = make_header(&img, comp, LineOrder::IncreasingY);
            let (source_heap, bases) = build_sample_heap(&img);
            let mut source_fb = make_frame_buffer(&img, &bases, 4, 4 * img.w);
            let source_base = -(img.min_y as i64 * 4 * img.w as i64 + img.min_x as i64 * 4);
            source_fb.insert_sample_count_slice(
                Slice::new(PixelType::Uint, source_base, 4, 4 * img.w as usize, 1, 1, 0.0, false, false).unwrap(),
            ).unwrap();
            let source_counts: Vec<u8> = img.counts.iter().flat_map(|c| c.to_ne_bytes()).collect();
            let mut out = MemOStream::new();
            {
                let mut writer = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
                writer.set_frame_buffer(&source_fb).unwrap();
                writer.write_pixels(&source_counts, &source_heap, img.h).unwrap();
            }
            let bytes = out.into_data();
            for threads in [0, 4] {
                for s1 in img.min_y..img.min_y + img.h {
                    for s2 in s1..img.min_y + img.h {
                        let (r1, r2) = ((s1 - img.min_y) as usize, (s2 - img.min_y + 1) as usize);
                        let rows = &img.counts[r1 * w..r2 * w];
                        let first: u32 = img.counts[..r1 * w].iter().sum();
                        let n = rows.iter().sum::<u32>() as usize;

                        // Count raster of just rows s1..=s2: its low corner maps to offset 0.
                        let y_stride = 4 * img.w;
                        let mut fb = DeepFrameBuffer::new();
                        let base = -(s1 as i64 * y_stride as i64 + img.min_x as i64 * 4);
                        fb.insert_sample_count_slice(
                            Slice::new(PixelType::Uint, base, 4, y_stride as usize, 1, 1, 0.0, false, false).unwrap(),
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
                            for &v in &img.values[ci][first as usize..first as usize + n] {
                                match ty {
                                    PixelType::Half => expected.extend_from_slice(&Half::from(v as f32).bits().to_le_bytes()),
                                    _ => expected.extend_from_slice(&(v as f32).to_le_bytes()),
                                }
                            }
                        }

                        let mut is = MemIStream::new(bytes.clone());
                        let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(threads)).unwrap();
                        inf.set_frame_buffer(&fb).unwrap();
                        let mut sc = vec![0u8; rows.len() * 4];
                        inf.read_pixel_sample_counts(s1, s2, &mut sc).unwrap();
                        let got: Vec<u32> = sc.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                        assert_eq!(got, rows, "{comp:?}: counts of [{s1}, {s2}]");
                        let mut heap = vec![0u8; expected.len()];
                        crate::probe::reset(crate::probe::Probe::DeepScanReadParallel);
                        inf.read_pixels(s1, s2, &sc, &mut heap).unwrap();
                        if threads > 1 && s2 > s1 {
                            crate::probe::assert_engaged(crate::probe::Probe::DeepScanReadParallel, threads);
                        }
                        assert_eq!(heap, expected, "{comp:?} threads={threads}: samples of [{s1}, {s2}]");
                    }
                }
            }
        }
        }
    }
}
