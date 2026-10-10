//! Port of OpenEXR `ImfScanLineInputFile.cpp` — the **scanline** pixel reader.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfScanLineInputFile.cpp>
//!
//! The dispatcher that decides *which* reader a file needs (`ImfInputFile.cpp`)
//! lives next door in [`super::input_file`]; this file is only the scanline
//! engine it delegates to, which is also what a multipart `InputPart` drives.
//!
//! # A note on the port target
//!
//! In OpenEXR v3.3.11 `ScanLineInputFile`/`InputFile` are thin C++ wrappers over
//! the OpenEXRCore C API (`exr_read_scanline_chunk_info`, `exr_decoding_run`,
//! …) — the manual offset-table read, chunk seek and scatter live deep inside
//! the core library.  A literal token-for-token port of the v3.3.11 `.cpp`
//! would therefore have to port the entire C core first.
//!
//! The **on-disk byte format** this reader consumes is fixed and unchanged
//! across OpenEXR versions, and is exactly the format produced by this crate's
//! [`OutputFile`][crate::scanline::OutputFile].  The read algorithm ported here
//! therefore mirrors the *legacy* IlmImf `ScanLineInputFile`
//! (`readPixels` → `readPixelData` → `copyIntoFrameBuffer`) — i.e. the inverse
//! of our writer — which is the canonical description of that byte format:
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.1.0/src/lib/OpenEXR/ImfScanLineInputFile.cpp>
//!
//! The v3.3.11 semantics that *are* observable are matched exactly:
//! * `setFrameBuffer` performs the x/y-sampling compatibility check and records
//!   a *fill list* for channels present in the frame buffer but absent from the
//!   file (`ImfScanLineInputFile.cpp` lines 240-277).
//! * `readPixels(s1, s2)` swaps `s1 > s2`, validates the range against the data
//!   window, then for every scanline seeks to its chunk via the line-offset
//!   table, reads `y` + `dataSize` + data, decompresses and scatters each
//!   channel into the frame buffer honouring strides / sampling, filling absent
//!   channels with their fill value (`Data::readPixels` +
//!   `ScanLineProcess::run_decode` / `run_fill`).

use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::DerefMut;

use crate::attr::PixelType;
use crate::compressor::{new_compressor, Compressor};
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::frame_buffer::FrameBuffer;
use crate::header::{self, Header};
use crate::io::IStream;
use crate::misc::{bytes_per_line_table, copy_into_frame_buffer, try_filled_vec, BufferFormat};

use imath_rs::{divp, modp};

// ---------------------------------------------------------------------------
// InSliceInfo — port of the anonymous-namespace `struct InSliceInfo`
// ---------------------------------------------------------------------------

/// Per-channel read descriptor — port of `InSliceInfo`
/// (`ImfScanLineInputFile.cpp`, legacy layout).
///
/// Built by [`ScanLineInputFile::set_frame_buffer`] from the file's channel list and
/// the user frame buffer.  Mirrors the C++ struct: the frame-buffer pixel type
/// (`type_fb`), the file's stored pixel type (`type_file`), strides / sampling,
/// the fill flag and fill value.  The raw `char* base` becomes a signed byte
/// offset (`base_offset`) into the caller-owned output buffer — the same
/// safe-Rust adaptation used by [`crate::frame_buffer::Slice`] and by the
/// scanline writer's `OutSliceInfo`.
#[derive(Clone, Debug)]
struct InSliceInfo {
    /// Destination buffer this slice writes to (planar read; 0 for the single-buffer API).
    buffer: usize,
    /// `base_offset` relative to THAT buffer; `base_offset` itself is rebased into the virtual
    /// address space of all the planes at every read (`read_pixels_planes`).
    local_base: i64,
    /// Pixel type requested in the frame-buffer slice (the scatter target).
    type_fb: PixelType,
    /// Pixel type stored on disk for this channel (the scatter source).
    ///
    /// Equals `type_fb` for a `fill` slice (no file channel exists, so the C++
    /// code uses the frame-buffer type for both).
    type_file: PixelType,
    /// Signed byte offset that replaces the C++ `char* base` (origin baked in).
    base_offset: i64,
    /// Distance in bytes between horizontally adjacent samples (`slice.xStride`).
    x_stride: usize,
    /// Distance in bytes between vertically adjacent rows (`slice.yStride`).
    y_stride: usize,
    /// Horizontal sub-sampling factor.
    x_sampling: i32,
    /// Vertical sub-sampling factor.
    y_sampling: i32,
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
// InputFile — port of `class InputFile` (dispatcher) + `class ScanLineInputFile`
// ---------------------------------------------------------------------------

/// Single-part scanline EXR reader — port of `Imf::InputFile` dispatching to
/// `Imf::ScanLineInputFile`.
///
/// Construct with [`InputFile::new`](crate::scanline::InputFile::new) (validates magic + version, reads the
/// header and the line-offset table), describe the destination pixels with
/// [`ScanLineInputFile::set_frame_buffer`], then pull scanlines with
/// [`ScanLineInputFile::read_pixels`].
///
/// The reader borrows the caller's [`IStream`] for its whole lifetime.
pub struct ScanLineInputFile<'a, S = &'a mut dyn IStream> {
    /// The input stream, addressed through `DerefMut<Target = dyn IStream>`
    /// (the C++ `_streamData->is`) -- generic over the stream HOLDER, the same split
    /// `multipart::MultiPartInputFile<'a, S>`/`tiled::TiledInputFile<'a, S>`
    /// already have, so `scanline::InputFile` (the front door this type
    /// backs) can own a path-opened stream.
    stream: S,

    /// Every field of C++ `ScanLineInputFile::Data` EXCEPT the stream itself
    /// — factored out so the same state can be held either OWNED (here) or
    /// BORROWED (see `scanline::input_file::PartReaderState::ScanLine`, the
    /// read-side mirror of the writer's
    /// `OutputFileState`/`OutputPartWriter` split). See
    /// [`ScanLineInputFileState`]'s own docs.
    state: ScanLineInputFileState,

    /// Ties `'a` to the struct even when `S` (e.g. `Box<dyn IStream + 'a>`)
    /// is the only field that mentions it -- same purpose as
    /// `MultiPartInputFile`'s own `_stream_lt` marker.
    _stream_lt: PhantomData<&'a mut ()>,
}

struct CachedChunk {
    offset_index: usize,
    min_y: i32,
    max_y: i32,
    raw_size: Option<usize>,
}

/// Every field of C++ `ScanLineInputFile::Data` except the stream — the
/// reader's state, persistable independently of any particular borrow of the
/// stream.
///
/// OpenEXR v3.4.15
/// `ImfMultiPartInputFile.cpp::getInputPart<T>` (byte-identical to v3.3.11,
/// unchanged across that range) still caches one reader object per part
/// (`_data->parts[n].file`, returned via `any_cast` on every subsequent
/// call) — the read-side mirror of the writer's `_outputFiles` cache this
/// crate already mirrors for `MultiPartOutputFile`. Splitting `state` out
/// of `ScanLineInputFile` is the prerequisite for that per-part caching,
/// and it IS wired in: `scanline::input_file::PartReaderState::ScanLine`
/// holds exactly this `ScanLineInputFileState`, and
/// `MultiPartInputFile::input_part` hands out a borrowing
/// `InputFilePartReader` view over it instead of rebuilding a fresh reader
/// per call (worthwhile chiefly for the tiled-as-scanlines path's tile-row
/// cache at 16k-65k resolutions, not for any read-side correctness bug --
/// read calls are already fully self-contained per `read_pixels(y1, y2,
/// out)` call; unlike the writer there is no `currentScanLine`/
/// partial-block state that a rebuild would silently reset). There is no
/// dedicated scanline part-reader wrapper type: the general
/// `PartReaderState` enum (which also covers tiled/deep parts) serves that
/// role -- C++'s own `InputPart` (`ImfInputPart.cpp`) confirms there is no
/// separate per-storage-kind "part reader" class in the reference either:
/// `InputPart` holds a single raw `InputFile*` (`getInputPart<InputFile>`)
/// and forwards every call to it verbatim, exactly like
/// `InputFilePartReader` forwards to `PartReaderState::ScanLine`'s state.
pub(crate) struct ScanLineInputFileState {
    /// The image header (C++ `_data->header`).
    header: Header,

    /// Raw version word read from the file (flags inspected at construction).
    version: i32,

    /// Line order copied from the header — C++ `_data->lineOrder`.
    ///
    /// On the read path the scatter is order-independent (each chunk carries
    /// its own `y`), so this is kept only for structural fidelity with the C++
    /// `Data` struct and for future use by the optimised / multi-line paths.
    #[allow(dead_code)]
    line_order: crate::attr::LineOrder,

    /// Data-window bounds — C++ `_data->{minX,maxX,minY,maxY}`.
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,

    /// File offsets of each chunk, indexed by `(y - min_y) / lines_in_buffer`
    /// — C++ `_data->lineOffsets`.  For the uncompressed path
    /// `lines_in_buffer == 1`, so there is one entry per scanline.
    line_offsets: Vec<u64>,

    /// Scanlines per chunk (1 for NONE) — C++ `_data->linesInBuffer`.
    lines_in_buffer: i32,

    /// Per-channel read descriptors — C++ `_data->slices` (empty until
    /// [`set_frame_buffer`][InputFile::set_frame_buffer] is called).
    slices: Vec<InSliceInfo>,

    /// Reusable scratch buffer holding one chunk's raw (uncompressed) pixel
    /// bytes — the C++ `LineBuffer::buffer` / decode pipeline scratch.
    line_buffer: Vec<u8>,

    /// The last serial chunk survives frame-buffer changes and repeated row reads.
    cached_chunk: Option<CachedChunk>,
    /// Codec output handed off by swap_out, without a decoded-payload copy.
    decoded_buffer: Vec<u8>,

    /// Uncompressed bytes per scan line, indexed by `y - min_y` — C++
    /// `_data->bytesPerLine`.  Used to compute a chunk's expected uncompressed
    /// size so the raw-vs-compressed passthrough can be detected by size, the
    /// way the C core's `decompress_data` does (`packsz == unpacksz` → raw).
    bytes_per_line: Vec<usize>,

    /// The scanline decompressor, or `None` for `NO_COMPRESSION`.
    ///
    /// Mirrors the C++ `LineBuffer::compressor`.  When present and a chunk's
    /// stored `dataSize` is *smaller* than the expected uncompressed size, the
    /// chunk is run through `uncompress`; when it equals the uncompressed size
    /// the bytes are raw and the codec is bypassed (the C core's
    /// `decompress_data` rule).
    compressor: Option<Box<dyn Compressor>>,

    /// `Some(partNumber)` when this reader drives a part of a **multipart**
    /// file, `None` for a standalone single-part file — C++
    /// `ScanLineInputFile::Data::partNumber` (+ the context's `is_multipart`).
    ///
    /// The single seam that unifies the single-part and multipart chunk read
    /// paths: when `Some`, each chunk begins with a `partNumber` (i32 LE) that
    /// is read and verified to equal this part number (C++ `extract_chunk_leader`
    /// rejects a mismatch); when `None`, no prefix is expected, so single-part
    /// reads are byte-identical to the pre-multipart reader.
    part_number: Option<i32>,

    /// This file's captured thread count (C++ `numThreads`,
    /// captured once at construction — see [`crate::threading::ThreadCount`]).
    /// Defaults to the process-wide [`crate::threading::global_thread_count`]
    /// unless [`ScanLineInputFileState::from_part_mt`] was given an explicit
    /// override.
    num_threads: crate::threading::ThreadCount,

    /// The codec (C++ `_data->_compression`) — kept alongside `compressor` so
    /// the parallel path can build a fresh per-worker codec instance
    /// via [`new_compressor`] (each worker owns one; codecs are not `Sync`).
    compression: crate::attr::Compression,

    /// Bytes of one fully-packed chunk's UNCOMPRESSED data at its widest (C++
    /// `maxBytesPerLine * linesInBuffer`... this is `maxBytesPerLine` alone),
    /// reused by the parallel path to size fresh per-worker codec instances.
    max_bytes_per_line: usize,
}

impl<'a, S> ScanLineInputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Build the scanline reader for one already-parsed part — port of
    /// `ScanLineInputFile::ScanLineInputFile(InputPartData*)`.
    ///
    /// This is the **only** constructor: a file's magic + version word, header
    /// list and chunk offset tables are read in exactly one place
    /// (`multipart::input::read_prologue`), by
    /// [`InputFile`](super::InputFile) for a standalone file and by
    /// [`MultiPartInputFile`](crate::multipart::MultiPartInputFile) for a part.
    /// Both hand the result here.
    ///
    /// `part_number` is `None` for a single-part file (chunks carry no
    /// `partNumber` prefix) and `Some(n)` for a part of a multipart file, where
    /// every chunk's leading `partNumber` is verified against `n`
    /// (C++ `OpenEXRCore/chunk.c::extract_chunk_leader`).
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfScanLineInputFile.cpp>
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when the part header's `type` is not
    ///   `scanlineimage`, or when [`Header::sanity_check`] rejects the header.
    /// * Propagates the [`new_compressor`] error for a codec this build cannot
    ///   construct.
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
        let state = ScanLineInputFileState::from_part_mt(
            header,
            version,
            part_number,
            line_offsets,
            num_threads,
        )?;
        Ok(Self { stream, state, _stream_lt: PhantomData })
    }

    /// This part's chunk offset table (unit tests inspect the on-disk layout).
    #[cfg(test)]
    pub(crate) fn line_offsets(&self) -> &[u64] {
        self.state.line_offsets()
    }

    /// Proof-of-reuse test hook: drive `fetch_chunk` with a
    /// caller-owned pool, immediately give the buffer back, and report its
    /// capacity. `pub(crate)` (not module-private) so `scanline::input_file`
    /// (a sibling module) can reach it through `InputFile`'s own matching
    /// test-only wrapper for its `parallel_read_reuses_pool_buffer_across_chunks`
    /// test — see that test's doc for why this is not a global counter.
    #[cfg(test)]
    pub(crate) fn debug_fetch_chunk_capacity(
        &mut self,
        offset_index: usize,
        pool: &crate::buf::BufPool,
    ) -> Result<usize> {
        let raw = self.state.fetch_chunk(&mut *self.stream, offset_index, pool)?;
        let cap = raw.data.capacity();
        pool.give(raw.data);
        Ok(cap)
    }

    /// The image header — port of `const Header& InputFile::header() const`.
    pub fn header(&self) -> &Header {
        self.state.header()
    }

    /// The raw file-format version word — port of `int InputFile::version()`.
    pub fn version(&self) -> i32 {
        self.state.version()
    }

    /// Set the pixel destination — port of `InputFile::setFrameBuffer` /
    /// `ScanLineInputFile::setFrameBuffer`. See
    /// [`ScanLineInputFileState::set_frame_buffer`] for the full doc.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }

    /// Read scanlines `[min(s1,s2), max(s1,s2)]` into `out` — port of
    /// `ScanLineInputFile::readPixels(scanLine1, scanLine2)`. See
    /// [`ScanLineInputFileState::read_pixels`] for the full algorithm doc.
    pub fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixels(&mut *self.stream, scan_line1, scan_line2, out)
    }

    /// [`read_pixels`](Self::read_pixels) into SEVERAL destination buffers: a slice inserted with
    /// `FrameBuffer::insert_in(name, slice, buffer)` lands in `planes[buffer]`, addressed relative to
    /// that buffer. One decode of each chunk serves every plane (the multi-threaded path gives each
    /// worker disjoint rows of every plane); an OpenEXR extension, see the frame-buffer docs.
    pub fn read_pixels_planes(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<()> {
        self.state.read_pixels_planes(&mut *self.stream, scan_line1, scan_line2, planes)
    }

    /// Read the chunk covering `first_scan_line` RAW (still-packed, no
    /// decompression). See
    /// [`ScanLineInputFileState::raw_pixel_data`] for the full doc.
    pub fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        self.state.raw_pixel_data(&mut *self.stream, first_scan_line)
    }
}

impl ScanLineInputFileState {
    /// Pure state computation for [`ScanLineInputFile::from_part_mt`] and
    /// `MultiPartInputFile`'s per-part cache — touches no stream at all,
    /// since the header and line-offset table are already fully parsed by
    /// the time this runs (`multipart::input::read_prologue`). `num_threads`
    /// is the per-file override (`None` = capture the current
    /// process-wide default).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when the part header's `type` is not
    ///   `scanlineimage`, or when [`Header::sanity_check`] rejects the header.
    /// * Propagates the [`new_compressor`] error for a codec this build cannot
    ///   construct.
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        line_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let num_threads = crate::threading::ThreadCount::capture(num_threads)?;
        // The part header must describe a scanline image (the container
        // validated `type`; re-check here so this path never mis-reads a tiled
        // or deep part as scanline).
        if header.has_type() {
            let t = header.type_()?;
            if t != header::SCANLINE_IMAGE {
                return Err(ExrError::Arg(format!(
                    "Cannot read \"{t}\" image data with the scanline reader                      (only \"{}\" is supported).",
                    header::SCANLINE_IMAGE
                )));
            }
        }

        // C++: header.sanityCheck(isTiled = false, isMultipart).
        header.sanity_check_at_open(false, part_number.is_some())?;

        let data_window = header.data_window()?;
        let line_order = header.line_order()?;
        let min_x = data_window.min.x;
        let max_x = data_window.max.x;
        let min_y = data_window.min.y;
        let max_y = data_window.max.y;

        // C++: bytesPerLineTable(header, bytesPerLine) → maxBytesPerLine; used
        // to size the per-chunk scratch buffer (the decode pipeline's unpacked
        // buffer).  Reusing the same helper the writer uses keeps both sides in
        // lock-step.
        let width = max_x - min_x + 1;
        let channels = header.channels()?;
        let mut bytes_per_line: Vec<usize> = Vec::new();
        let max_bytes_per_line =
            bytes_per_line_table(min_y, max_y, width, channels, &mut bytes_per_line)?;

        // C++ `Compressor::Compressor`
        // (`ImfCompressor.cpp:44-48`) throws `OverflowExc` for
        // `maxScanLineSize > INT_MAX` -- guard it here too, BEFORE sizing the
        // per-chunk scratch buffer below. Without this, a crafted header
        // (huge `dataWindow` width, a multi-line codec) requests a
        // `vec![0u8; ..]` allocation of hundreds of gigabytes and aborts the
        // process (`handle_alloc_error`) instead of erroring cleanly.
        if max_bytes_per_line > i32::MAX as usize {
            return Err(ExrError::Arg(format!(
                "Maximum scan line size {max_bytes_per_line} exceeds the OpenEXR \
                 format's INT_MAX byte limit."
            )));
        }

        let compression = header.compression()?;
        let compressor = new_compressor(compression, max_bytes_per_line, &header)?;
        let lines_in_buffer = match compressor.as_deref() {
            Some(c) => c.num_scan_lines(),
            None => crate::compressor::num_lines_in_buffer(compression)?,
        };
        // C++: lineBufferSize = maxBytesPerLine * linesInBuffer. Checked: even
        // with `max_bytes_per_line` under the INT_MAX guard above, multiplying
        // by a multi-line codec's `linesInBuffer` (e.g. DWA's 256) can still
        // overflow `usize` on a 32-bit target or simply request an absurd
        // allocation -- reject instead of trying it.
        let line_buffer_size = max_bytes_per_line
            .checked_mul(lines_in_buffer as usize)
            .ok_or_else(|| {
                ExrError::Arg("Line buffer size overflow (scan line size * lines per chunk).".into())
            })?;
        // Fallible allocation, not `vec![..]` -- see
        // `try_filled_vec`'s docs.
        let line_buffer = try_filled_vec(line_buffer_size, 0u8)?;

        log::debug!(
            "ScanLineInputFile: {}x{} codec={compression:?}              linesInBuffer={lines_in_buffer} chunkOffsets={} part={part_number:?}",
            width,
            max_y - min_y + 1,
            line_offsets.len()
        );

        Ok(Self {
            header,
            version,
            line_order,
            min_x,
            max_x,
            min_y,
            max_y,
            line_offsets,
            lines_in_buffer,
            slices: Vec::new(),
            line_buffer,
            cached_chunk: None,
            decoded_buffer: Vec::new(),
            bytes_per_line,
            compressor,
            part_number,
            num_threads,
            compression,
            max_bytes_per_line,
        })
    }

    /// This part's chunk offset table (unit tests inspect the on-disk layout).
    #[cfg(test)]
    fn line_offsets(&self) -> &[u64] {
        &self.line_offsets
    }

    /// The image header — port of `const Header& InputFile::header() const`.
    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    /// The raw file-format version word — port of `int InputFile::version()`.
    pub(crate) fn version(&self) -> i32 {
        self.version
    }

    /// Set the pixel destination — port of `InputFile::setFrameBuffer` /
    /// `ScanLineInputFile::setFrameBuffer`.
    ///
    /// Builds the per-channel `InSliceInfo` table by walking *both* the file
    /// channel list and the frame buffer:
    ///
    /// * A file channel that also has a frame-buffer slice → a normal scatter
    ///   descriptor (with x/y-sampling compatibility checked, matching C++).
    /// * A file channel with **no** frame-buffer slice → a `skip` descriptor:
    ///   its bytes are consumed from the chunk but not stored.
    /// * A frame-buffer slice with **no** file channel → a `fill` descriptor:
    ///   the slice is filled with its `fill_value` (the C++ *fill list*).
    ///
    /// The scatter slices' `base_offset` fields are byte indices into the
    /// caller's output buffer (the slice supplied here describes where each
    /// channel lands in that buffer).
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] when a frame-buffer slice's x/y-sampling is
    /// incompatible with the corresponding header channel (matching C++).
    pub(crate) fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        let channels = self.header.channels()?;

        // C++: compatibility check — every frame-buffer slice that also names a
        // file channel must agree on x/y subsampling.
        for (name, slice) in frame_buffer.iter() {
            if let Some(ch) = channels.find_channel(name)
                && (ch.x_sampling != slice.x_sampling || ch.y_sampling != slice.y_sampling)
            {
                return Err(ExrError::Arg(format!(
                    "X and/or y subsampling factors of \"{name}\" channel of \
                     input file are not compatible with the frame buffer's \
                     subsampling factors."
                )));
            }
        }

        // C++ builds InSliceInfo in *file channel order* so the on-disk byte
        // stream is consumed in the right order (channels are stored
        // alphabetically in the file).  Channels present in the file but absent
        // from the frame buffer become `skip` slices; frame-buffer slices with
        // no file channel become `fill` slices appended afterwards (the C++
        // `fill_list`).
        let mut slices: Vec<InSliceInfo> = Vec::new();

        for (name, ch) in channels.iter() {
            match frame_buffer.find_slice(name) {
                Some(slice) => {
                    slices.push(InSliceInfo {
                        buffer: frame_buffer.buffer_of(name),
                        local_base: slice.base_offset,
                        type_fb: slice.type_,
                        type_file: ch.type_,
                        base_offset: slice.base_offset,
                        x_stride: slice.x_stride,
                        y_stride: slice.y_stride,
                        x_sampling: ch.x_sampling,
                        y_sampling: ch.y_sampling,
                        fill: false,
                        fill_value: slice.fill_value,
                        skip: false,
                    });
                }
                None => {
                    // File channel with no destination: read & discard its bytes.
                    slices.push(InSliceInfo {
                        buffer: 0,
                        local_base: 0,
                        type_fb: ch.type_,
                        type_file: ch.type_,
                        base_offset: 0,
                        x_stride: 0,
                        y_stride: 0,
                        x_sampling: ch.x_sampling,
                        y_sampling: ch.y_sampling,
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
                slices.push(InSliceInfo {
                    buffer: frame_buffer.buffer_of(name),
                    local_base: slice.base_offset,
                    type_fb: slice.type_,
                    type_file: slice.type_,
                    base_offset: slice.base_offset,
                    x_stride: slice.x_stride,
                    y_stride: slice.y_stride,
                    x_sampling: slice.x_sampling,
                    y_sampling: slice.y_sampling,
                    fill: true,
                    fill_value: slice.fill_value,
                    skip: false,
                });
            }
        }

        self.slices = slices;
        Ok(())
    }

    /// Read scanlines `[min(s1,s2), max(s1,s2)]` into `out` — port of
    /// `ScanLineInputFile::readPixels(scanLine1, scanLine2)` →
    /// `Data::readPixels` → `ScanLineProcess::run_decode` / `run_fill`.
    ///
    /// `out` is the caller-owned destination buffer the frame-buffer slices
    /// address (their `base_offset` fields are byte indices into it).  For
    /// every scanline in the (inclusive) range the chunk is located via the
    /// line-offset table, read, identity-uncompressed (NONE), and scattered
    /// into `out` honouring per-channel strides and sampling; channels with no
    /// file data are filled with their `fill_value`.  Both `INCREASING_Y` and
    /// `DECREASING_Y` line orders are handled — the chunk's own `y` header
    /// drives where its pixels land, so the read order is irrelevant.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the requested range
    ///   lies outside the data window (matching the C++ throws).
    /// * [`ExrError::Input`] if the offset table or a chunk header is corrupt
    ///   (bad `y`, negative / oversized `dataSize`).
    /// * Propagates gather / stream errors.
    pub(crate) fn read_pixels(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        out: &mut [u8],
    ) -> Result<()> {
        self.read_pixels_planes(stream, scan_line1, scan_line2, &mut [out])
    }

    /// [`read_pixels`](Self::read_pixels) into SEVERAL destination buffers (the planar read): the
    /// slice of `FrameBuffer::insert_in(.., buffer)` is written to `planes[buffer]`, with its
    /// `base_offset`/strides relative to that buffer. One decode of each chunk serves every plane;
    /// the multi-threaded path hands each worker disjoint row ranges of every plane.
    ///
    /// # Errors
    /// As [`read_pixels`](Self::read_pixels), plus [`ExrError::Arg`] for a slice whose buffer index
    /// is not in `planes`.
    pub(crate) fn read_pixels_planes(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<()> {
        // Rebase every slice into the virtual address space of the planes (plane `i` starts at the
        // total length of the planes before it); with one plane this is the identity.
        let starts = crate::misc::plane_starts(planes);
        for s in &mut self.slices {
            let start = starts.get(s.buffer).copied().ok_or_else(|| {
                ExrError::Arg(format!(
                    "Frame buffer slice names destination buffer {} but only {} were given.",
                    s.buffer,
                    planes.len()
                ))
            })?;
            s.base_offset = s.local_base + start as i64;
        }
        // C++: if (slices.size() == 0) throw ArgExc("No frame buffer specified …").
        if self.slices.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data destination.".into(),
            ));
        }

        // C++: if (scanLine2 < scanLine1) swap(scanLine1, scanLine2);
        let (mut min_scan, mut max_scan) = (scan_line1, scan_line2);
        if max_scan < min_scan {
            std::mem::swap(&mut min_scan, &mut max_scan);
        }

        // C++: range must lie within [min.y, max.y] of the data window.
        if min_scan < self.min_y || max_scan > self.max_y {
            return Err(ExrError::Arg(format!(
                "Tried to read scan line outside the image file's data window: \
                 {min_scan} - {max_scan} vs datawindow {} - {}",
                self.min_y, self.max_y
            )));
        }

        let first_chunk = ((min_scan - self.min_y) / self.lines_in_buffer) as usize;
        let last_chunk = ((max_scan - self.min_y) / self.lines_in_buffer) as usize;

        // Gate: parallel only when there is more than one chunk
        // AND the captured thread count is `> 1` — 1:1 with C++'s
        // `nchunks > 1 && numThreads > 1` (`ImfScanLineInputFile.cpp:444`).
        if first_chunk < last_chunk && self.num_threads.is_parallel() {
            let chunk_indices: Vec<_> = (first_chunk..=last_chunk).collect();
            if let Some(()) = self.read_pixels_parallel(stream, &chunk_indices, min_scan, max_scan, planes)? {
                crate::probe::hit(crate::probe::Probe::ScanReadParallel);
                return Ok(());
            }
        }
        // `None` = the planner found the frame buffer aliased (fell back
        // to serial); fall through to the serial loop below so the
        // result is identical either way.

        for offset_index in first_chunk..=last_chunk {
            if self.cached_chunk.as_ref().is_none_or(|c| c.offset_index != offset_index) {
                // Invalidate before fetching: an unsuccessful fetch must not leave a key
                // claiming the overwritten scratch still contains the previous raw chunk.
                self.cached_chunk = None;
                let (chunk_min_y, chunk_end_y, data_size) =
                    self.fetch_chunk_into_line_buffer(stream, offset_index)?;
                let decoded = decode_chunk(
                    self.compressor.as_mut(),
                    &self.line_buffer[..data_size],
                    chunk_min_y,
                    chunk_end_y,
                    &self.bytes_per_line,
                    self.min_y,
                )?;
                let decoded_len = decoded.len();
                drop(decoded);
                let raw_size = if decoded_len == data_size {
                    Some(data_size)
                } else {
                    let compressor = self.compressor.as_mut().ok_or_else(|| {
                        ExrError::Logic("Decoded chunk has no compressor.".into())
                    })?;
                    compressor.swap_out(&mut self.decoded_buffer, decoded_len)?;
                    None
                };
                self.cached_chunk = Some(CachedChunk {
                    offset_index,
                    min_y: chunk_min_y,
                    max_y: chunk_end_y,
                    raw_size,
                });
            }
            let cached = self.cached_chunk.as_ref().ok_or_else(|| {
                ExrError::Logic("Decoded chunk cache is empty after reading.".into())
            })?;
            let decoded = match cached.raw_size {
                Some(size) => &self.line_buffer[..size],
                None => &self.decoded_buffer,
            };
            if planes.len() == 1 {
                scatter_chunk(
                    &self.slices,
                    self.min_x,
                    self.max_x,
                    cached.min_y,
                    cached.max_y,
                    min_scan,
                    max_scan,
                    decoded,
                    &mut *planes[0],
                    0,
                )?;
            } else {
                // One pass over the same decoded chunk per destination buffer.
                for (p, plane) in planes.iter_mut().enumerate() {
                    scatter_chunk_in_plane(
                        &self.slices,
                        self.min_x,
                        self.max_x,
                        cached.min_y,
                        cached.max_y,
                        min_scan,
                        max_scan,
                        decoded,
                        plane,
                        starts[p],
                        p,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Seek to chunk `offset_index`, read and validate its leader
    /// (`[partNumber]`, `y`, `dataSize`), and return `(chunk_min_y,
    /// chunk_end_y, data_size)`. This is the half of chunk fetching that
    /// never allocates, shared by [`Self::fetch_chunk`] (parallel path,
    /// owns its bytes) and [`Self::fetch_chunk_into_line_buffer`] (serial
    /// path, reuses `self.line_buffer`). Port of the leader-read + validation
    /// half of the old monolithic `read_pixels` loop body.
    fn read_and_validate_leader(
        &self,
        stream: &mut dyn IStream,
        offset_index: usize,
    ) -> Result<(i32, i32, usize)> {
        let chunk_offset = *self.line_offsets.get(offset_index).ok_or_else(|| {
            ExrError::Input(format!("Line offset table index {offset_index} out of range."))
        })?;

        // Seek to the chunk and read its leader.  For a multipart file the
        // leader begins with the part number (i32 LE), which must match the
        // part being read (C++ `extract_chunk_leader`); for a single-part
        // file there is no such prefix.  Then: y (i32) + dataSize (i32).
        stream.seekg(chunk_offset)?;
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in scan-line chunk: expected {expected}, \
                     found {found}."
                )));
            }
        }
        let chunk_min_y = xdr::read_i32(stream)?;
        let data_size = xdr::read_i32(stream)?;

        // Validate the chunk header against the data window and scratch
        // buffer (corrupt → hard error, never a panic).
        //
        // The chunk must be *exactly* the block this slot addresses.  C++
        // `OpenEXRCore/chunk.c::read_and_validate_chunk_leader` recomputes
        // `(scanline_y - min.y) / lines_per_chunk` from the stored `y` and
        // rejects the chunk when it differs from the requested index; the
        // C++ `ScanLineInputFile` throws the same way.  Without this check a
        // backward offset makes the naive "next chunk" step move backwards —
        // an infinite loop — and a forward offset silently leaves the
        // requested lines zeroed.
        let expected_y = self.min_y as i64 + offset_index as i64 * self.lines_in_buffer as i64;
        if chunk_min_y as i64 != expected_y {
            return Err(ExrError::Input(format!(
                "Scan-line chunk has y coordinate {chunk_min_y} but the chunk \
                 at offset-table index {offset_index} starts at {expected_y} (corrupt \
                 offset table or chunk)."
            )));
        }
        if data_size < 0 || data_size as usize > self.line_buffer.len() {
            return Err(ExrError::Input(format!(
                "Scan-line chunk reports an invalid data size of {data_size} \
                 bytes (buffer capacity {}).",
                self.line_buffer.len()
            )));
        }
        let data_size = data_size as usize;

        // Hot/loop detail (trace, compiled out unless enabled): per-chunk
        // decode — the chunk's start scanline + its stored byte size.
        log::trace!("scanline read chunk: y={chunk_min_y} dataSize={data_size}");

        let chunk_end_y = (chunk_min_y + self.lines_in_buffer - 1).min(self.max_y);
        Ok((chunk_min_y, chunk_end_y, data_size))
    }

    /// Fetch one chunk's leader-validated, still-possibly-compressed bytes
    /// from `stream` (calling thread only — this is the ONLY function in the
    /// read pipeline that touches the stream). Hands out an owned, independently-alive `RawChunk` per
    /// chunk in a window — required for the PARALLEL producer, which must
    /// keep a whole window of chunks alive simultaneously (see
    /// [`Self::fetch_chunk_into_line_buffer`] for the serial path's
    /// allocation-free alternative, which only ever has one chunk in
    /// flight).
    ///
    /// `data`'s buffer comes from `pool` (a
    /// [`crate::buf::BufPool`] shared with the decode worker across the
    /// `std::thread::scope` in [`Self::read_pixels`]/`_mt`) instead of a
    /// fresh `try_filled_vec` per call — the worker gives each `RawChunk`'s
    /// buffer back to the SAME pool once it has scattered that chunk's
    /// decoded pixels (see the `read_pixels`/`_mt` call site), so after the
    /// first window or two the pool is steady-state and this never
    /// allocates.
    fn fetch_chunk(&self, stream: &mut dyn IStream, offset_index: usize, pool: &crate::buf::BufPool) -> Result<RawChunk> {
        let (chunk_min_y, chunk_end_y, data_size) =
            self.read_and_validate_leader(stream, offset_index)?;
        let mut data = pool.take(data_size);
        stream.read(&mut data)?;
        Ok(RawChunk { min_y: chunk_min_y, end_y: chunk_end_y, data })
    }

    /// Fetch one chunk's leader-validated, still-possibly-compressed bytes
    /// into the persistent `self.line_buffer` scratch buffer — the SERIAL
    /// read loop's allocation-free counterpart to [`Self::fetch_chunk`].
    ///
    /// # Why this exists
    /// Calling `fetch_chunk` from the serial loop would take a buffer per
    /// chunk even though `line_buffer` — sized once at construction to fit
    /// the largest possible chunk — is available. Only the serial loop can safely reuse ONE persistent buffer
    /// this way: exactly one chunk is ever "in flight" (fetched, decoded,
    /// scattered, then the next chunk fetched), unlike the parallel
    /// producer, which must keep up to a whole window of chunks alive
    /// simultaneously and therefore needs independently-owned buffers. This
    /// mirrors C++'s single reusable `LineBuffer` for the same (serial)
    /// code path.
    ///
    /// Returns `(chunk_min_y, chunk_end_y, data_size)`; the bytes are at
    /// `self.line_buffer[..data_size]`.
    fn fetch_chunk_into_line_buffer(
        &mut self,
        stream: &mut dyn IStream,
        offset_index: usize,
    ) -> Result<(i32, i32, usize)> {
        let (chunk_min_y, chunk_end_y, data_size) =
            self.read_and_validate_leader(stream, offset_index)?;
        stream.read(&mut self.line_buffer[..data_size])?;
        Ok((chunk_min_y, chunk_end_y, data_size))
    }

    /// The parallel read body (with read-ahead pipelining — see the
    /// note below). Returns `Ok(Some(()))` when it fully
    /// serviced the request, or `Ok(None)` when the destination planner
    /// found the frame buffer's write intervals unsafely overlapping (an
    /// aliased/broadcast layout) and did nothing — the caller then falls
    /// back to the fully serial loop.
    ///
    /// # Read-ahead pipelining
    /// Reading a whole window of `W = num_threads.max_in_flight()` chunks
    /// serially, THEN decoding+scattering it in parallel gives no I/O/decode
    /// overlap. A dedicated measurement (`tests/perf_io_vs_decode.rs`, 16384x8192, real
    /// on-disk file) showed the serial I/O share of the already-parallel
    /// (8-thread) wall time was 12-68% across NONE/RLE/ZIP/PIZ/DWAA —
    /// nowhere close to the "negligible" the original decode-bound-DWA
    /// profiling note assumed, and squarely above the 10% bar for
    /// implementing this. So: while window `k`'s chunks are decoded+scattered
    /// on the global pool, window `k+1`'s chunks are read on the calling
    /// thread AT THE SAME TIME, via one dedicated `std::thread::scope` worker
    /// per window transition.
    ///
    /// **Why a dedicated OS thread instead of `rayon::Scope::spawn`/`in_place_scope`
    /// directly on our own pool:** `IStream`/`OStream` deliberately carry no
    /// `Send` bound (so the bound does not ripple into ~40
    /// `&mut dyn IStream` APIs). Any closure handed to `rayon::ThreadPool::install`
    /// (what `threading::run_on_global_pool` calls) must be `Send`, so a
    /// closure that touches `stream` can never be installed onto our pool.
    /// `std::thread::scope`'s OUTER body has no such bound (only
    /// `Scope::spawn`'s closures do), so `stream` I/O always stays on the
    /// calling thread that owns this call.
    ///
    /// **One persistent decode-worker thread for the WHOLE call, not one per
    /// window transition.** Spawning a fresh `Scope::spawn` thread for
    /// every window transition is fine at the coarse end (large
    /// multiline chunks: few, expensive windows), but at the opposite
    /// extreme — NONE compression (1 scanline/chunk) at a large height —
    /// `tests/perf_io_vs_decode.rs::many_small_windows_thread_spawn_overhead_65536_height_none`
    /// measured 4096 window transitions (65536 height / `max_in_flight()`=16
    /// at 8 threads) for a single `read_pixels` call, and found `num_threads=8`
    /// **3x SLOWER** than `num_threads=0` (2025ms vs 684ms) — OS thread-spawn
    /// overhead (tens of microseconds each, but thousands of them) dwarfing
    /// the near-zero decode+scatter work NONE chunks require. So: spawn the
    /// decode-worker thread ONCE for the entire call; it loops receiving one
    /// window's `(raws, dest_batch)` at a time over a bounded (capacity-1)
    /// `mpsc::sync_channel`, decodes+scatters it (still via
    /// `run_on_global_pool`'s `par_iter` fan-out onto the shared pool — the
    /// window's OWN chunks are still decoded in parallel across threads),
    /// then reports completion over a second channel. The calling thread
    /// (this function's own stack frame, inside the `scope` body) is the
    /// producer: it reads windows off `stream` strictly in order and feeds
    /// them into the job channel; the channel's capacity-1 buffer gives
    /// exactly the one-window-ahead overlap the read-ahead needs (I/O of
    /// window `k+1` runs on the calling thread while decode of window `k`
    /// runs on the persistent worker) without a thread spawn per transition.
    /// Errors are propagated with their window index so the reported error
    /// is always the LOWEST-indexed failing window regardless of which side
    /// (I/O on the producer, decode on the worker) detected it first
    /// (a determinism requirement) — a producer I/O
    /// error for window `i` never overrides an already-reported worker
    /// decode error for an earlier window `j < i`, since the producer only
    /// ever gets that far after successfully sending every window `< i`.
    fn read_pixels_parallel(
        &mut self,
        stream: &mut dyn IStream,
        chunk_indices: &[usize],
        min_scan: i32,
        max_scan: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<Option<()>> {

        // Compute each chunk's bounding write interval up front (cheap: reuses
        // the same address formula the scatter itself uses) and ask the
        // planner whether they are pairwise disjoint. With several destination
        // buffers a chunk has one interval PER PLANE (in plane order).
        let multi = planes.len() > 1;
        let starts_p = crate::misc::plane_starts(planes);
        let mut intervals = Vec::with_capacity(chunk_indices.len());
        for &offset_index in chunk_indices {
            let chunk_min_y = self.min_y + offset_index as i32 * self.lines_in_buffer;
            let chunk_end_y = (chunk_min_y + self.lines_in_buffer - 1).min(self.max_y);
            // `plan_disjoint` takes a SET of ranges per chunk (deep's
            // per-slice destinations need more than one); the flat scanline
            // reader has exactly one contiguous destination per chunk and
            // plane, so a single-plane read builds a single-element `RangeSet`.
            let mut set = crate::misc::RangeSet::new();
            if multi {
                for p in 0..planes.len() {
                    let iv = chunk_write_interval(
                        &self.slices,
                        self.min_x,
                        self.max_x,
                        chunk_min_y,
                        chunk_end_y,
                        min_scan,
                        max_scan,
                        Some(p),
                    )?;
                    // A slice must address ITS OWN plane: an interval that strays into another plane
                    // (or outside all) is a misaddressed slice. Decline the parallel path so the serial
                    // loop, which writes plane `p` only into `planes[p]`, reports it the same way at
                    // every thread count.
                    if !iv.is_empty() && (iv.start < starts_p[p] || iv.end > starts_p[p] + planes[p].len()) {
                        return Ok(None);
                    }
                    set.push(iv);
                }
            } else {
                set.push(chunk_write_interval(
                    &self.slices,
                    self.min_x,
                    self.max_x,
                    chunk_min_y,
                    chunk_end_y,
                    min_scan,
                    max_scan,
                    None,
                )?);
            }
            intervals.push(set);
        }
        // `None` = overlapping / straddling intervals: the caller falls back to the serial loop.
        // Fresh reborrows with ONE lifetime for the planner (the caller's slice-of-slices has two).
        let mut local: Vec<&mut [u8]> = planes.iter_mut().map(|p| &mut **p).collect();
        let mut dests: Vec<crate::misc::DestSet> = match if multi {
            crate::misc::plan_disjoint_planes(&mut local, &intervals)
        } else {
            crate::misc::plan_disjoint(&mut *local[0], &intervals)
        } {
            Some(d) => d,
            None => return Ok(None),
        };

        // Byte-based window sizing (so parallel is never slower than
        // serial) — see `ThreadCount::window_size_for_bytes`'s
        // doc. `max_bytes_per_line * lines_in_buffer` is an upper bound on
        // one chunk's raw/uncompressed size (exactly what the caller already
        // knows without reading the chunk).
        let bytes_per_chunk = self.max_bytes_per_line.saturating_mul(self.lines_in_buffer.max(0) as usize);
        let window = self.num_threads.window_size_for_bytes(bytes_per_chunk).max(1);
        let compression = self.compression;
        let max_bytes_per_line = self.max_bytes_per_line;
        let header = &self.header;
        let bytes_per_line = &self.bytes_per_line;
        let slices = &self.slices;
        let (min_x, max_x) = (self.min_x, self.max_x);
        let file_min_y = self.min_y;
        let count = self.num_threads;

        let index_windows: Vec<&[usize]> = chunk_indices.chunks(window).collect();
        let mut dest_windows: Vec<&mut [crate::misc::DestSet]> =
            dests.chunks_mut(window).collect();
        debug_assert_eq!(index_windows.len(), dest_windows.len());

        if index_windows.is_empty() {
            return Ok(Some(()));
        }

        // One decode-worker job per window: its index (for deterministic
        // error reporting), the window's raw chunks, and its exclusive
        // destination pieces.
        type WindowJob<'a, 'b> = (usize, Vec<RawChunk>, &'b mut [crate::misc::DestSet<'a>]);
        type WindowDone = (usize, Result<()>);

        // One pool, shared (by reference) between the
        // producer (fetches into a pooled buffer) and the worker (gives the
        // buffer back once decode+scatter are done with it), across the
        // whole `thread::scope` below.
        let pool = crate::buf::BufPool::new();
        // Seed the pool with up to TWO windows' worth of right-sized buffers
        // up front, matching this pipeline's own bound on outstanding chunks
        // (the window currently draining on the worker + the next window
        // the producer fetches concurrently, gated by the channel's
        // capacity-1 depth just below). Starting empty would force most of a
        // large parallel read's chunks through a cold `BufPool::take` ->
        // `ensure_len` growth (a real, measured zero-fill cost that GREW
        // with thread count instead of shrinking -- see `BufPool::prewarm`'s
        // own doc for the numbers) instead of the intended steady-state
        // reuse; this pays that cost exactly once, before the parallel loop
        // starts, sized to the actual in-flight bound rather than growing
        // it chunk-by-chunk as the free list starves mid-read.
        pool.prewarm_for_pipeline(crate::buf::READ_WINDOWS_IN_FLIGHT, window, bytes_per_chunk, chunk_indices.len());
        let pool = &pool;

        // A real checkout/checkin `CodecPool`, shared the same way `pool`
        // (the `BufPool`) is -- see `Compressor`'s trait doc for why
        // `map_init` alone does not achieve the reuse it appears to (a
        // construction-count measurement showed essentially one codec built
        // per chunk regardless of thread count).
        let codec_pool = crate::compressor::CodecPool::new();
        let codec_pool = &codec_pool;

        std::thread::scope(|scope| -> Result<()> {
            // Capacity 1: lets the producer (this stack frame) get one window
            // ahead of the persistent worker — exactly the read-ahead overlap
            // this function exists to provide — without unbounded buffering.
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<WindowJob>(1);
            let (done_tx, done_rx) = std::sync::mpsc::channel::<WindowDone>();

            let worker = scope.spawn(move || {
                while let Ok((idx, raws, dest_batch)) = job_rx.recv() {
                    let outcome: Result<()> = (|| {
                        let results: Vec<Result<()>> =
                            crate::threading::run_on_global_pool(count, move || {
                                use rayon::prelude::*;
                                // `.map()` over `CodecPool::checkout`/
                                // `checkin` per item, not `map_init` (see
                                // `Compressor`'s trait doc and
                                // `CodecPool`'s own doc for why
                                // `map_init` alone was not the "once per
                                // job split" reuse it looked like it was).
                                // A construction failure (deterministic
                                // given fixed compression/size/header
                                // inputs) is `Err`, never checked in --
                                // every item would hit the identical
                                // error.
                                // `into_par_iter()` (not
                                // `par_iter()`) so each closure invocation
                                // OWNS its `RawChunk` and can give its
                                // buffer back to `pool` once decode+scatter
                                // are done with it -- the buffer must
                                // outlive `scatter_chunk`, not just
                                // `decode_chunk`, since the raw-passthrough
                                // case (`decode_chunk` returning
                                // `Cow::Borrowed(data)`, i.e. NONE
                                // compression or a codec's own "no-shrink"
                                // case) has `decoded` alias `raw.data`
                                // directly.
                                raws.into_par_iter()
                                    .zip(dest_batch.par_iter_mut())
                                    .map(|(raw, dest)| -> Result<()> {
                                        let mut worker_compressor = codec_pool.checkout(|| {
                                            new_compressor(compression, max_bytes_per_line, header)
                                        })?;
                                        let decode_result = decode_chunk(
                                            worker_compressor.as_mut(),
                                            &raw.data,
                                            raw.min_y,
                                            raw.end_y,
                                            bytes_per_line,
                                            file_min_y,
                                        );
                                        let result = match decode_result {
                                            Ok(decoded) => {
                                                // One destination buffer: the whole-chunk scatter
                                                // (with its packed fast path). Several: one pass per
                                                // plane over the same decoded chunk (`dest[p]` is
                                                // plane `p`'s piece, see the interval planning).
                                                let r = if dest.len() == 1 {
                                                    let d0 = &mut dest[0];
                                                    scatter_chunk(
                                                        slices,
                                                        min_x,
                                                        max_x,
                                                        raw.min_y,
                                                        raw.end_y,
                                                        min_scan,
                                                        max_scan,
                                                        decoded.as_ref(),
                                                        &mut *d0.piece,
                                                        d0.base,
                                                    )
                                                } else {
                                                    dest.iter_mut().enumerate().try_for_each(|(p, d)| {
                                                        scatter_chunk_in_plane(
                                                            slices,
                                                            min_x,
                                                            max_x,
                                                            raw.min_y,
                                                            raw.end_y,
                                                            min_scan,
                                                            max_scan,
                                                            decoded.as_ref(),
                                                            d.piece,
                                                            d.base,
                                                            p,
                                                        )
                                                    })
                                                };
                                                drop(decoded); // end any borrow of raw.data before giving it back
                                                r
                                            }
                                            Err(e) => Err(e),
                                        };
                                        codec_pool.checkin(worker_compressor);
                                        // Return the buffer regardless of
                                        // success/failure -- a decode/
                                        // scatter error on one chunk must
                                        // not shrink the pool for the
                                        // rest of the run.
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

            // Producer: read every window off `stream`, strictly in order,
            // on the calling thread, feeding the worker via the job channel.
            // Stop at the first I/O failure — never queue a later window
            // once an earlier one has failed.
            let mut io_err: Option<(usize, ExrError)> = None;
            for (i, dest_batch) in dest_windows.iter_mut().enumerate() {
                let raws: Result<Vec<RawChunk>> = index_windows[i]
                    .iter()
                    .map(|&idx| self.fetch_chunk(stream, idx, pool))
                    .collect();
                let raws = match raws {
                    Ok(r) => r,
                    Err(e) => {
                        io_err = Some((i, e));
                        break;
                    }
                };
                // Move this window's destination pieces out (leaving an
                // empty slice behind, never reused) so the job can own an
                // exclusive `&mut` to them independent of `dest_windows`.
                let batch: &mut [crate::misc::DestSet] = std::mem::take(dest_batch);
                if job_tx.send((i, raws, batch)).is_err() {
                    // Worker already stopped (its own earlier error) —
                    // nothing more to send.
                    break;
                }
                crate::probe::hit(crate::probe::Probe::ReadAheadWindow);
            }
            drop(job_tx); // tell the worker there is no more work coming

            // Drain every completion the worker reported, keeping only the
            // LOWEST-indexed error — never let a later I/O failure on the
            // producer side override an earlier decode failure the worker
            // already reported, or vice versa (deterministic error reporting).
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
                .map_err(|_| ExrError::Base("scanline decode worker thread panicked".into()))?;

            match worst {
                Some((_, e)) => Err(e),
                None => Ok(()),
            }
        })?;


        Ok(Some(()))
    }

    /// Fill one line of a `fill` slice with its `fill_value` — the read-path
    /// counterpart of `ScanLineProcess::run_fill`. Thin wrapper over the free
    /// function [`scatter_fill_into`] with `dest_base = 0` (this method's
    /// remaining callers all address `out` directly).
    #[allow(dead_code)] // kept for API symmetry / potential future direct callers
    fn scatter_fill(
        &self,
        slice: &InSliceInfo,
        line_y: i32,
        d_min_x: i32,
        d_max_x: i32,
        out: &mut [u8],
    ) -> Result<()> {
        scatter_fill_into(slice, line_y, d_min_x, d_max_x, out, 0)
    }

    /// Read the chunk covering `first_scan_line` RAW (still-packed, never
    /// decompressed) — port of `ScanLineInputFile::rawPixelData`
    /// (`ImfScanLineInputFile.cpp:328-354`; C++'s two-parameter
    /// `char*&`/`int&` sizing dance is replaced by returning an owned
    /// `Vec<u8>` directly, same convention as
    /// [`crate::tiled::input::TiledInputFileState::raw_tile_data`] /
    /// [`crate::deep::scanline_input::DeepScanLineInputFileState::raw_pixel_data`]).
    ///
    /// Like those, this is an **exact-coordinate** read: the only caller,
    /// [`crate::scanline::output::OutputFileState::copy_pixels`], always
    /// supplies its own already-computed `current_scan_line`, never a "next
    /// chunk" cursor.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if `first_scan_line` lies outside the data window.
    /// * [`ExrError::Input`] if the offset table or chunk header is corrupt
    ///   (bad `y`, negative/oversize `dataSize`, or the chunk's leading
    ///   `partNumber` disagrees with this reader's part).
    pub(crate) fn raw_pixel_data(
        &mut self,
        stream: &mut dyn IStream,
        first_scan_line: i32,
    ) -> Result<(i32, Vec<u8>)> {
        if first_scan_line < self.min_y || first_scan_line > self.max_y {
            return Err(ExrError::Arg(format!(
                "Tried to read scan line outside the image file's data window: \
                 {first_scan_line} vs datawindow {} - {}",
                self.min_y, self.max_y
            )));
        }

        // Same chunk-offset-table indexing + leader verification as
        // `read_pixels` above (see its comments for why the stored `y` must
        // be re-checked: a corrupt/backward offset must error, not loop or
        // silently misplace bytes).
        let offset_index = ((first_scan_line - self.min_y) / self.lines_in_buffer) as usize;
        let chunk_offset = *self.line_offsets.get(offset_index).ok_or_else(|| {
            ExrError::Input(format!("Line offset table index {offset_index} out of range."))
        })?;

        stream.seekg(chunk_offset)?;
        if let Some(expected) = self.part_number {
            let found = xdr::read_i32(stream)?;
            if found != expected {
                return Err(ExrError::Input(format!(
                    "Invalid part number in scan-line chunk: expected {expected}, \
                     found {found}."
                )));
            }
        }
        let chunk_min_y = xdr::read_i32(stream)?;
        let data_size = xdr::read_i32(stream)?;

        let expected_y =
            self.min_y as i64 + offset_index as i64 * self.lines_in_buffer as i64;
        if chunk_min_y as i64 != expected_y {
            return Err(ExrError::Input(format!(
                "Scan-line chunk has y coordinate {chunk_min_y} but the chunk \
                 for scan line {first_scan_line} starts at {expected_y} (corrupt \
                 offset table or chunk)."
            )));
        }
        if data_size < 0 {
            return Err(ExrError::Input(format!(
                "Scan-line chunk reports a negative data size of {data_size} bytes."
            )));
        }
        let data_size = data_size as usize;

        // Bound against the remaining stream bytes before allocating — same
        // hardening as the deep/tiled `raw_*_data` readers (a corrupt/huge
        // `dataSize` must not request a process-scale allocation).
        if let Some(file_size) = stream.size() {
            let pos = stream.tellg()?;
            let remaining = file_size.saturating_sub(pos);
            if data_size as u64 > remaining {
                return Err(ExrError::Input(format!(
                    "Scan-line chunk data size {data_size} exceeds the \
                     {remaining} bytes remaining in the file (corrupt chunk)."
                )));
            }
        }

        let mut bytes = try_filled_vec(data_size, 0u8)?;
        stream.read(&mut bytes)?;
        Ok((chunk_min_y, bytes))
    }
}

// ---------------------------------------------------------------------------
// Pure per-chunk pipeline functions
// ---------------------------------------------------------------------------
//
// `fetch_chunk` (stream-only, above) / `decode_chunk` / `scatter_chunk` are
// a three-function split: the serial path in
// `read_pixels` composes them in the same order the parallel path
// (`read_pixels_parallel`) does, so both produce byte-identical `out` — a
// "safe scalar source of truth" argument, the same shape used for SIMD
// codecs elsewhere in this crate.

/// One chunk's leader-validated, still-possibly-compressed bytes, fetched by
/// [`ScanLineInputFileState::fetch_chunk`] on the calling thread.
struct RawChunk {
    /// The chunk's first scanline (its validated leader `y`).
    min_y: i32,
    /// The chunk's last scanline (`min_y + linesInBuffer - 1`, clamped to the
    /// data window's `max_y`).
    end_y: i32,
    /// The chunk's raw stored bytes (length = the leader's `dataSize`).
    data: Vec<u8>,
}

/// Decode one [`RawChunk`] into its uncompressed XDR bytes — pure function,
/// no stream access, safe to run on any thread (each caller passes its own
/// `compressor`, so there is no shared mutable codec state across threads).
///
/// Mirrors the C core's `decompress_data` raw-vs-compressed rule: bytes are
/// raw when `dataSize` already equals the expected uncompressed size
/// (covers both NO_COMPRESSION and any codec's "no-shrink → store raw" chunk
/// — this relies on the writer-side `stored_chunk` invariant: a chunk is
/// stored compressed only when that is strictly smaller than the raw bytes),
/// otherwise the codec's
/// `uncompress` is run and its output length is checked.
fn decode_chunk<'r>(
    // `&mut Box<dyn Compressor>`, not `&mut dyn Compressor`: the latter,
    // nested inside `Option<..>`, hits the default-object-lifetime-bound
    // rule (no default declared for `Option<T>`'s `T`) and defaults the
    // trait object to `dyn Compressor + 'static` — which then can't unify
    // with `self.compressor`'s (or a local `Box`'s) actual, shorter borrow.
    // Taking `&mut Box<..>` sidesteps this: it is a concrete, non-generic
    // target type, so ordinary reference elision applies. Method calls
    // auto-deref through the `Box` to `dyn Compressor` as usual.
    //
    // `&'r mut` (not a fresh, unrelated lifetime): unifying `compressor`'s
    // borrow with `data`'s lets the compressed-chunk path return
    // `Cow::Borrowed(decoded)` instead of `decoded.to_vec()` —
    // `Compressor::uncompress(&mut self, ..) -> Result<&[u8]>`'s output
    // borrows from `&mut self` by elision, so tying `compressor` to `'r`
    // makes that output's lifetime `'r` too, matching `Cow<'r, [u8]>`. Every
    // real caller passes a `self`-owned compressor field alongside
    // `self`-owned chunk bytes, so `'r` is trivially satisfiable; NLL at the
    // call site (not this function's own body) still lets the NEXT loop
    // iteration re-borrow the same compressor once THIS call's `Cow` is
    // done being read (its actual last use, not `'r`'s nominal extent).
    compressor: Option<&'r mut Box<dyn Compressor>>,
    data: &'r [u8],
    chunk_min_y: i32,
    chunk_end_y: i32,
    bytes_per_line: &[usize],
    file_min_y: i32,
) -> Result<Cow<'r, [u8]>> {
    crate::probe::hit(crate::probe::Probe::DecodeChunk);

    let mut expected_uncompressed = 0usize;
    for line_y in chunk_min_y..=chunk_end_y {
        let li = (line_y - file_min_y) as usize;
        expected_uncompressed += bytes_per_line.get(li).copied().ok_or_else(|| {
            ExrError::Input(format!("bytesPerLine index {li} out of range for chunk y {line_y}."))
        })?;
    }

    if data.len() == expected_uncompressed {
        // Raw passthrough (matches NO_COMPRESSION *and* the codec "no-shrink
        // -> store raw" case). BORROWS `data` rather than cloning it, which
        // would pay a full memcpy per chunk for NONE (the single most common
        // case this branch serves) even though `scatter_chunk` only ever
        // needs a `&[u8]`. `Cow::Borrowed` costs nothing when the caller only
        // reads it (every call site does); the compressed path below also
        // borrows — the codec's own persistent `out` buffer, not a
        // fresh copy.
        return Ok(Cow::Borrowed(data));
    }
    let Some(c) = compressor else {
        return Err(ExrError::Input(format!(
            "Chunk dataSize {} differs from the uncompressed size {expected_uncompressed} \
             but the file has no compressor.",
            data.len()
        )));
    };
    let decoded = c.uncompress(data, chunk_min_y, expected_uncompressed).map_err(crate::error::decode_err)?;
    if decoded.len() != expected_uncompressed {
        return Err(ExrError::Input(format!(
            "Decompressed chunk size {} does not match the expected uncompressed \
             size {expected_uncompressed} (corrupt chunk).",
            decoded.len()
        )));
    }
    // Borrow the codec's own output buffer instead of
    // `decoded.to_vec()` — a per-chunk allocation+copy that C never pays
    // (C reads the decompressor's buffer in place).
    Ok(Cow::Borrowed(decoded))
}

/// Scatter one decoded chunk's bytes into `dest`, writing at
/// `write_ptr - dest_base` instead of `write_ptr` — so `dest` may be either
/// the caller's whole frame buffer (`dest_base = 0`, the serial path) or one
/// exclusive [`crate::misc::ChunkDest`] piece handed out by
/// [`crate::misc::plan_disjoint`] (the parallel path). Pure function: no
/// stream access, no shared mutable state.
///
/// `min_scan`/`max_scan` is the caller's originally requested range (a chunk
/// may extend beyond it at either end); only lines inside both the chunk AND
/// the requested range are scattered, matching the original monolithic loop.
#[allow(clippy::too_many_arguments)]
fn scatter_chunk(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    // The packed same-type
    // fast path — see its own doc — only applies to a narrow, common shape
    // (every channel present, unconverted, unsampled, and tiling one
    // fully-packed interleaved pixel). `None` means it declined (falls
    // through to the unchanged generic loop below); `Some(result)` means it
    // fully serviced the chunk. Split into its own function (rather than
    // inlined here) so `scatter_chunk_packed_matches_generic_across_matrix`
    // can drive each path independently with the same inputs.
    if let Some(result) = scatter_chunk_packed(
        slices, min_x, max_x, chunk_min_y, chunk_end_y, min_scan, max_scan, decoded, dest,
        dest_base,
    ) {
        crate::probe::hit(crate::probe::Probe::ScatterChunkPacked);
        return result;
    }
    scatter_chunk_generic(
        slices, min_x, max_x, chunk_min_y, chunk_end_y, min_scan, max_scan, decoded, dest,
        dest_base,
    )
}

/// [`scatter_chunk_generic_plane`] over every slice (the single-buffer read).
#[allow(clippy::too_many_arguments)]
fn scatter_chunk_generic(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    scatter_chunk_generic_plane(
        slices, min_x, max_x, chunk_min_y, chunk_end_y, min_scan, max_scan, decoded, dest,
        dest_base, None,
    )
}

/// Scatter one decoded chunk into destination buffer `plane` ONLY: the slices that belong to
/// other buffers are walked past (their file bytes are consumed) but not written. The planar
/// read calls this once per plane over the same decoded chunk. Never takes the packed fast path
/// (its precondition is one interleaved buffer).
#[allow(clippy::too_many_arguments)]
fn scatter_chunk_in_plane(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
    plane: usize,
) -> Result<()> {
    scatter_chunk_generic_plane(
        slices, min_x, max_x, chunk_min_y, chunk_end_y, min_scan, max_scan, decoded, dest,
        dest_base, Some(plane),
    )
}

/// The general, channel-major scatter loop — handles every layout
/// (subsampling, fill, skip, type conversion, partial in-range chunks), and
/// is [`scatter_chunk_packed`]'s fallback and the source of truth its
/// equivalence test checks against.
#[allow(clippy::too_many_arguments)]
fn scatter_chunk_generic_plane(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
    plane: Option<usize>,
) -> Result<()> {
    let buf_format = BufferFormat::Xdr;
    let first = chunk_min_y.max(min_scan);
    let last = chunk_end_y.min(max_scan);

    let mut read_offset = 0usize;
    for line_y in chunk_min_y..=chunk_end_y {
        for slice in slices {
            if modp(line_y, slice.y_sampling) != 0 {
                continue;
            }

            let d_min_x = divp(min_x, slice.x_sampling);
            let d_max_x = divp(max_x, slice.x_sampling);
            let num_pixels = (d_max_x - d_min_x + 1) as usize;

            let file_px_size = crate::misc::pixel_type_size(slice.type_file)?;
            let channel_bytes = file_px_size * num_pixels;

            // Planar read: a slice that belongs to ANOTHER destination buffer is not this call's
            // business, but its file bytes still occupy the chunk, so the read cursor passes them
            // (a fill slice has no file bytes).
            if plane.is_some_and(|p| slice.buffer != p) {
                if !slice.fill {
                    read_offset += channel_bytes;
                }
                continue;
            }

            let in_range = line_y >= first && line_y <= last;
            if slice.skip || !in_range {
                if !slice.fill {
                    read_offset += channel_bytes;
                }
                if slice.fill && in_range {
                    scatter_fill_into(slice, line_y, d_min_x, d_max_x, dest, dest_base)?;
                }
                continue;
            }

            if slice.fill {
                scatter_fill_into(slice, line_y, d_min_x, d_max_x, dest, dest_base)?;
                continue;
            }

            let line_ptr: i64 =
                slice.base_offset + divp(line_y, slice.y_sampling) as i64 * slice.y_stride as i64;
            let write_ptr_i: i64 = line_ptr + d_min_x as i64 * slice.x_stride as i64;
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
    }
    Ok(())
}

/// Fast path for a fully-packed, same-type (no conversion), unsampled,
/// non-fill/non-skip interleaved chunk: gather every channel's bytes for
/// ONE pixel into a single combined buffer and issue ONE write per pixel,
/// instead of `scatter_chunk`'s channel-major loop (one pass per channel
/// over the whole line). Mirrors C++ `unpack.c`'s
/// `unpack_16bit_4chan_interleave`/`unpack_32bit`'s "gather all channels of
/// a pixel into a single wide store" trick — done safely (bounds-checked
/// once via `chunks_mut`, no `unsafe` pointer casts), and only for the
/// precondition C++'s own `internal_exr_match_decode` selector requires
/// too: every channel shares the same on-disk AND frame-buffer type, no
/// sampling, and the destination slices, taken together, exactly tile one
/// `pixel_size`-byte block with no gaps or overlaps.
///
/// Returns `None` when the precondition doesn't hold — the caller
/// (`scatter_chunk`) then falls back to the unchanged, general channel-major
/// loop, producing identical bytes either way (proved by
/// `scatter_chunk_packed_matches_generic_across_matrix`).
#[allow(clippy::too_many_arguments)]
fn scatter_chunk_packed(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    decoded: &[u8],
    dest: &mut [u8],
    dest_base: usize,
) -> Option<Result<()>> {
    // Only handle a chunk that's ENTIRELY inside the requested range — the
    // generic loop already handles a chunk that straddles `min_scan`/
    // `max_scan` (a sub-range read), which this fast path does not attempt to.
    // Slices of different destination buffers are never one packed pixel.
    if slices.is_empty() || slices.iter().any(|s| s.buffer != slices[0].buffer) || chunk_min_y < min_scan || chunk_end_y > max_scan {
        return None;
    }

    let type0 = slices[0].type_fb;
    let comp_size = crate::misc::pixel_type_size(type0).ok()?;
    let n = slices.len();
    let pixel_size = comp_size.checked_mul(n)?;
    let y_stride = slices[0].y_stride;

    for s in slices {
        if s.fill || s.skip { return None; }
        if s.x_sampling != 1 || s.y_sampling != 1 { return None; }
        if s.type_fb != type0 || s.type_file != type0 { return None; }
        if s.x_stride != pixel_size { return None; }
        if s.y_stride != y_stride { return None; }
    }

    // Every slice's `base_offset` must differ from the group's minimum by
    // an exact multiple of `comp_size`, together tiling `[min, min +
    // pixel_size)` with no gaps or overlaps — the origin adjustment shared
    // by every slice (same data window) cancels out in the differences, so
    // this is exactly the "one contiguous packed pixel" precondition
    // regardless of where the buffer's logical origin sits.
    let base = slices.iter().map(|s| s.base_offset).min()?;
    let mut seen = vec![false; n];
    for s in slices {
        let rel = s.base_offset.checked_sub(base)?;
        if rel < 0 || rel % comp_size as i64 != 0 {
            return None;
        }
        let idx = (rel / comp_size as i64) as usize;
        if idx >= n || seen[idx] {
            return None;
        }
        seen[idx] = true;
    }

    let width = (max_x - min_x + 1) as usize;

    Some((|| -> Result<()> {
        let mut read_offset = 0usize;
        // Hoisted out of the scanline loop below (not a fresh
        // `vec![0u8; ...]` per scanline) -- every byte of `combined`
        // is written by the per-slice/per-pixel `copy_from_slice` loop each
        // iteration (the precondition already checked above guarantees the
        // slices tile `[0, pixel_size)` per pixel with no gaps or overlap,
        // over all `width` pixels), so allocating it once per CALL (one
        // chunk) instead of once per scanline is sound with no correctness
        // change -- same size and same full-overwrite every time.
        let mut combined = vec![0u8; width * pixel_size];
        for line_y in chunk_min_y..=chunk_end_y {
            for slice in slices {
                let off_in_pixel = (slice.base_offset - base) as usize;
                let ch_bytes = decoded.get(read_offset..read_offset + width * comp_size).ok_or_else(|| {
                    ExrError::Input("Decoded chunk buffer too short for the requested channel row.".into())
                })?;
                for (x, src) in ch_bytes.chunks_exact(comp_size).enumerate() {
                    let d = x * pixel_size + off_in_pixel;
                    combined[d..d + comp_size].copy_from_slice(src);
                }
                read_offset += width * comp_size;
            }

            let row_start_i: i64 = base + (line_y as i64) * y_stride as i64
                + min_x as i64 * pixel_size as i64;
            if row_start_i < 0 {
                return Err(ExrError::Arg(
                    "Frame buffer slice address is out of range (negative).".into(),
                ));
            }
            let row_start = local_addr(row_start_i as usize, dest_base)?;
            let row_len = width * pixel_size;
            let dest_len = dest.len();
            let out = dest.get_mut(row_start..row_start + row_len).ok_or_else(|| {
                ExrError::Arg(format!(
                    "Frame buffer slice write range [{row_start}, {}) exceeds the \
                     destination buffer's length ({dest_len}).",
                    row_start + row_len,
                ))
            })?;
            out.copy_from_slice(&combined);
        }
        Ok(())
    })())
}

/// `abs_addr - base`, erroring instead of underflowing when a caller passes
/// an address before its own piece's base — a planner bug, not an
/// attacker-controlled input, but still checked (no-panic rule).
fn local_addr(abs_addr: usize, base: usize) -> Result<usize> {
    abs_addr.checked_sub(base).ok_or_else(|| {
        ExrError::Base(format!(
            "internal error: scatter address {abs_addr} precedes its destination \
             piece's base {base}"
        ))
    })
}

/// Fill one line of a `fill` slice with its `fill_value` — free-function form
/// of the read-path counterpart of `ScanLineProcess::run_fill`, usable from
/// both the serial scatter (`dest_base = 0`) and the parallel one (a
/// [`crate::misc::ChunkDest`] piece).
fn scatter_fill_into(
    slice: &InSliceInfo,
    line_y: i32,
    d_min_x: i32,
    d_max_x: i32,
    dest: &mut [u8],
    dest_base: usize,
) -> Result<()> {
    let line_ptr: i64 =
        slice.base_offset + divp(line_y, slice.y_sampling) as i64 * slice.y_stride as i64;
    let write_ptr_i: i64 = line_ptr + d_min_x as i64 * slice.x_stride as i64;
    if write_ptr_i < 0 {
        return Err(ExrError::Arg(
            "Frame buffer fill slice address is out of range (negative).".into(),
        ));
    }
    let write_ptr = local_addr(write_ptr_i as usize, dest_base)?;
    // Count-driven, not address-driven — a broadcast fill slice
    // (x_stride == 0, every pixel written to the SAME address) still needs
    // exactly `d_max_x - d_min_x + 1` iterations.
    let count = (d_max_x - d_min_x + 1).max(0) as usize;

    copy_into_frame_buffer(
        &[],
        0,
        dest,
        write_ptr,
        count,
        slice.x_stride,
        true,
        slice.fill_value,
        BufferFormat::Xdr,
        slice.type_fb,
        slice.type_file,
    )?;
    Ok(())
}

/// Compute the bounding byte interval `[lo, hi)` a chunk's [`scatter_chunk`]
/// call would write into `out`, WITHOUT writing anything — the input to
/// [`crate::misc::plan_disjoint`]. Walks the exact same address formulas as
/// `scatter_chunk`/`scatter_fill_into` so the two can never disagree.
///
/// A chunk that (for this call's range) writes nothing returns an empty
/// range at its own would-be start, which `plan_disjoint` treats as
/// automatically disjoint from everything.
#[allow(clippy::too_many_arguments)]
fn chunk_write_interval(
    slices: &[InSliceInfo],
    min_x: i32,
    max_x: i32,
    chunk_min_y: i32,
    chunk_end_y: i32,
    min_scan: i32,
    max_scan: i32,
    plane: Option<usize>,
) -> Result<std::ops::Range<usize>> {
    let first = chunk_min_y.max(min_scan);
    let last = chunk_end_y.min(max_scan);
    let mut lo: Option<usize> = None;
    let mut hi: usize = 0;

    for line_y in first..=last {
        for slice in slices {
            if modp(line_y, slice.y_sampling) != 0 {
                continue;
            }
            // Planar read: only the slices of the requested destination buffer.
            if slice.skip || plane.is_some_and(|p| slice.buffer != p) {
                continue;
            }
            let d_min_x = divp(min_x, slice.x_sampling);
            let d_max_x = divp(max_x, slice.x_sampling);
            let num_pixels = (d_max_x - d_min_x + 1).max(0) as usize;
            if num_pixels == 0 {
                continue;
            }

            let fb_size = crate::misc::pixel_type_size(slice.type_fb)?;
            let line_ptr: i64 =
                slice.base_offset + divp(line_y, slice.y_sampling) as i64 * slice.y_stride as i64;
            let write_ptr_i: i64 = line_ptr + d_min_x as i64 * slice.x_stride as i64;
            if write_ptr_i < 0 {
                return Err(ExrError::Arg(
                    "Frame buffer slice address is out of range (negative).".into(),
                ));
            }
            let write_ptr = write_ptr_i as usize;

            // The last pixel written is at `write_ptr + (num_pixels-1)*x_stride`
            // (x_stride may be negative, so the span can extend either way;
            // `x_stride == 0` is a broadcast — every pixel lands on the same
            // `fb_size`-byte span). Compute both ends and take min/max.
            let last_pixel_i =
                write_ptr_i + (num_pixels as i64 - 1) * slice.x_stride as i64;
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
    }

    Ok(match lo {
        Some(l) => l..hi,
        // Nothing written by this chunk for this call: an empty, harmless
        // interval anchored wherever — `plan_disjoint` skips empty ranges.
        None => 0..0,
    })
}

// ---------------------------------------------------------------------------
// Tests — write→read round-trip (the key correctness gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::version;
    use crate::scanline::InputFile;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::frame_buffer::Slice;
    use crate::io::{MemIStream, MemOStream};
    use crate::scanline::OutputFile;
    use imath_rs::{Half, V2f};

    /// A crafted header (huge `dataWindow` width, a
    /// multi-line codec) must error at construction, never request a
    /// hundreds-of-gigabytes allocation (`handle_alloc_error` aborts the
    /// whole process -- not something `Result` can catch, so the only fix is
    /// to reject the size BEFORE calling the allocator).
    #[test]
    fn oversized_scan_line_size_errors_instead_of_aborting() {
        // Width chosen so max_bytes_per_line (one FLOAT channel: width * 4)
        // alone already exceeds i32::MAX, well before multiplying by any
        // multi-line codec's linesInBuffer.
        let width: i32 = 600_000_000; // * 4 bytes/pixel > i32::MAX
        let mut hdr = Header::with_dimensions(
            width,
            1,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let mut is = MemIStream::new(Vec::new());
        let stream: &mut dyn IStream = &mut is;
        let err = match ScanLineInputFile::from_part(stream, hdr, 2, None, vec![0u64; 1]) {
            Ok(_) => panic!("an oversized scan line must error, not succeed (or abort)"),
            Err(e) => e,
        };
        let _ = format!("{err}"); // a real, displayable error -- no abort reached
    }

    /// Build a `w`x`h` header with channels R, G, B (HALF) plus one FLOAT (A),
    /// the given line order, and NO_COMPRESSION.
    fn make_header(w: i32, h: i32, line_order: LineOrder) -> Header {
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            line_order,
            Compression::NoCompression,
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

    /// Interleaved pixel layout: per pixel `[R half][G half][B half][A float]`
    /// = 2+2+2+4 = 10 bytes, row-major.  Returns `(buffer, frame_buffer)`.
    ///
    /// Distinct value per channel/pixel: `value(base, x, y) = base + y*10 + x`.
    fn make_pixels(w: usize, h: usize) -> (Vec<u8>, FrameBuffer) {
        let pixel_bytes = 2 + 2 + 2 + 4; // R,G,B half + A float
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
            type_: ty,
            base_offset: off as i64,
            x_stride: pixel_bytes,
            y_stride: row_bytes,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        fb.insert("R", mk(PixelType::Half, 0)).unwrap();
        fb.insert("G", mk(PixelType::Half, 2)).unwrap();
        fb.insert("B", mk(PixelType::Half, 4)).unwrap();
        fb.insert("A", mk(PixelType::Float, 6)).unwrap();
        (buf, fb)
    }

    /// Write `pixels` to a fresh in-memory EXR via [`OutputFile`].
    fn write_exr(header: &Header, pixels: &[u8], fb: &FrameBuffer, h: i32) -> Vec<u8> {
        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, header).unwrap();
            of.set_frame_buffer(fb).unwrap();
            of.write_pixels(pixels, h).unwrap();
        }
        out.into_data()
    }

    fn round_trip(line_order: LineOrder) {
        let (w, h) = (5i32, 3i32);
        let header = make_header(w, h, line_order);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        // Read back into a fresh buffer with the same layout.
        let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
        readback.iter_mut().for_each(|b| *b = 0);

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&read_fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();

        assert_eq!(readback, pixels, "round-trip mismatch for {line_order:?}");
    }

    #[test]
    fn round_trip_increasing_y_pixel_exact() {
        round_trip(LineOrder::IncreasingY);
    }

    #[test]
    fn round_trip_decreasing_y_pixel_exact() {
        round_trip(LineOrder::DecreasingY);
    }

    /// Read the SAME multi-chunk (NO_COMPRESSION = 1
    /// line/chunk) file with `num_threads` in {0, 1, 2, 3, 8, 17} and assert
    /// every thread count produces byte-identical `out` — the read-side half
    /// of the thread-count determinism matrix. 64 lines gives 64 chunks,
    /// comfortably more than any tested thread count, so the parallel gate
    /// (`nchunks > 1 && numThreads > 1`) and multiple windows are exercised.
    #[test]
    fn parallel_read_is_byte_identical_to_serial_across_thread_counts() {
        let (w, h) = (37i32, 64i32); // odd width on purpose (not a power of two)
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 3, 8, 17] {
            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);

            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();

            crate::probe::assert_engaged(crate::probe::Probe::ScanReadParallel, n);
            assert_eq!(readback, pixels, "num_threads={n}: mismatch vs the written pixels");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// Proof of buffer reuse (scanline INPUT).
    ///
    /// Why not a bound on a PROCESS-WIDE allocation counter: under the full
    /// `cargo test -p exr-core --release --lib` suite (`cargo test`'s default
    /// same-process, many-threads-at-once execution) such a counter is
    /// swamped by OTHER, unrelated tests' concurrent `BufPool` traffic
    /// (counts in the hundreds, 467 in one run), so it cannot give a
    /// meaningful bound; see
    /// `tiled::input::tests::serial_read_reuses_pool_buffer_across_tiles`.
    ///
    /// Instead: test the [`crate::buf::BufPool`] mechanism
    /// directly with a pool THIS TEST OWNS, via
    /// `InputFile::debug_fetch_chunk_capacity` (a `#[cfg(test)]` hook that
    /// forwards to `ScanLineInputFile::debug_fetch_chunk_capacity`, which
    /// calls the same private `fetch_chunk` the real read loops call) in the
    /// same take-then-give sequence the serial loop uses. A locally-owned
    /// pool has zero cross-test contamination by construction. Every chunk
    /// here is the same size (one 4096-pixel HALF line, NONE compression),
    /// so a working pool returns the exact same allocation (same capacity)
    /// every time after the first -- no margin needed.
    #[test]
    fn serial_read_reuses_pool_buffer_across_chunks() {
        let (w, h) = (4096i32, 64i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();

        let pool = crate::buf::BufPool::new();
        let mut first_capacity = None;
        for offset_index in 0..h as usize {
            let cap = inf.debug_fetch_chunk_capacity(offset_index, &pool).unwrap();
            match first_capacity {
                None => first_capacity = Some(cap),
                Some(first) => assert_eq!(
                    cap, first,
                    "chunk {offset_index}: buffer capacity changed -- expected the exact \
                     same freed allocation to come back from the pool every time (every \
                     chunk here is the same size)"
                ),
            }
        }
        assert_eq!(
            pool.free_count(),
            1,
            "exactly one buffer must be sitting in the free list after the loop \
             (take-then-give, one at a time, never more than one in flight)"
        );
    }

    /// The parallel/serial read
    /// dispatch gate (`chunk_indices.len() > 1 && self.num_threads.
    /// is_parallel()`, matching C++'s `nchunks > 1 && numThreads > 1`,
    /// `ImfScanLineInputFile.cpp:444`) produces byte-identical output no
    /// matter which branch runs (`read_pixels_parallel` degrades
    /// gracefully for 0/1 chunks), so no output-correctness assertion can
    /// kill its `&&`/`>` mutants -- this test observes which branch
    /// actually executed via a call counter instead (see
    /// `Probe::ScanReadParallel`). `num_threads = 0` must NEVER
    /// enter the parallel path at all, matching the explicit C++ contract
    /// that `numThreads == 0` means "the calling thread only" -- an
    /// `&&`->`||` mutant would enter it here purely because there are
    /// multiple chunks, regardless of the (zero) thread count.
    #[test]
    fn read_pixels_parallel_dispatch_is_gated_by_both_conditions() {
        let (w, h) = (16i32, 8i32); // 8 chunks (one line each, NONE codec)
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let run = |num_threads: Option<i32>| -> u64 {
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);
            let mut is = MemIStream::new(bytes.clone());
            let mut inf = InputFile::with_num_threads(&mut is, num_threads).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            assert_eq!(readback, pixels, "num_threads={num_threads:?}: must still read correctly");
            crate::probe::get(crate::probe::Probe::ScanReadParallel)
        };

        // 0 threads, many chunks -> must NOT enter the parallel path.
        assert_eq!(run(Some(0)), 0, "num_threads=0 must never dispatch to read_pixels_parallel");

        // 2 threads, many chunks -> must enter the parallel path exactly
        // once (one read_pixels call).
        assert_eq!(run(Some(2)), 1, "num_threads=2 with 8 chunks must dispatch to read_pixels_parallel");

        // 2 threads, exactly ONE chunk (single-scanline read) -> must NOT
        // enter the parallel path, even though num_threads alone would
        // qualify.
        crate::probe::reset(crate::probe::Probe::ScanReadParallel);
        let (mut readback1, read_fb1) = make_pixels(w as usize, 1);
        readback1.iter_mut().for_each(|b| *b = 0);
        let mut is1 = MemIStream::new(bytes.clone());
        let mut inf1 = InputFile::with_num_threads(&mut is1, Some(2)).unwrap();
        inf1.set_frame_buffer(&read_fb1).unwrap();
        inf1.read_pixels(0, 0, &mut readback1).unwrap();
        assert_eq!(
            crate::probe::get(crate::probe::Probe::ScanReadParallel),
            0,
            "num_threads=2 with exactly 1 chunk must NOT dispatch to read_pixels_parallel"
        );
    }

    /// Same matrix over a ZIP file (16 lines/chunk, `singleScan`-style reuse
    /// of the compressor across chunks in a window would corrupt bytes if a
    /// codec carried cross-call state — this is the codec-reuse risk
    /// check for the read side).
    #[test]
    fn parallel_read_zip_multiline_chunks_match_across_thread_counts() {
        let (w, h) = (23i32, 96i32); // 96/16 = 6 ZIP chunks
        let header = make_header_comp(w, h, LineOrder::IncreasingY, Compression::Zip);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);

            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();

            crate::probe::assert_engaged(crate::probe::Probe::ScanReadParallel, n);
            assert_eq!(readback, pixels, "num_threads={n}: mismatch vs the written pixels");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// After `prewarm_for_pipeline` every `BufPool::take` in the
    /// parallel scanline writer and reader must hit (the reader holds 3
    /// windows in flight, the writer 1).
    #[test]
    fn parallel_scanline_pipelines_never_miss_the_prewarmed_pool() {
        let (w, h) = (64i32, 4096i32); // 256 ZIP chunks
        let header = make_header_comp(w, h, LineOrder::IncreasingY, Compression::Zip);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let misses = || crate::probe::get(crate::probe::Probe::BufPoolMiss);
        for &n in &[2i32, 8, 24] {
            let mut out = MemOStream::new();
            crate::probe::reset(crate::probe::Probe::BufPoolMiss);
            {
                let mut of = OutputFile::with_num_threads(&mut out, &header, Some(n)).unwrap();
                of.set_frame_buffer(&fb).unwrap();
                of.write_pixels(&pixels, h).unwrap();
            }
            assert_eq!(misses(), 0, "num_threads={n}: parallel scanline write missed the prewarmed pool");
            let bytes = out.into_data();

            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            let mut is = MemIStream::new(bytes);
            let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            crate::probe::reset(crate::probe::Probe::BufPoolMiss);
            crate::probe::reset(crate::probe::Probe::ReadAheadWindow);
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            assert_eq!(readback, pixels, "num_threads={n}: mismatch");
            assert_eq!(misses(), 0, "num_threads={n}: parallel scanline read missed the prewarmed pool");
            // The read-ahead pipeline must actually pipeline, i.e. hand
            // the worker several windows rather than one that covers the file.
            assert!(crate::probe::get(crate::probe::Probe::ReadAheadWindow) >= 2, "num_threads={n}: read-ahead ran a single window");
        }
    }

    /// `map_init` was measured constructing a FRESH codec for
    /// essentially EVERY chunk (256/256) at every thread count, not "once
    /// per job split" as it appears to. With a real
    /// `CodecPool`, the number of codecs actually CONSTRUCTED for the whole
    /// read must be bounded by roughly the thread count (plus a small
    /// margin for the pool's own warm-up), not by the chunk count -- 256
    /// ZIP chunks at `num_threads=24` must construct nowhere near 256
    /// codecs.
    #[test]
    fn parallel_read_codec_pool_created_is_bounded_by_thread_count_not_chunk_count() {
        let (w, h) = (8i32, 4096i32); // 4096/16 = 256 ZIP chunks, narrow width for speed
        let header = make_header_comp(w, h, LineOrder::IncreasingY, Compression::Zip);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let n = 24i32;
        let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
        readback.iter_mut().for_each(|b| *b = 0);
        crate::probe::reset(crate::probe::Probe::CodecCreated); // the writer above has its own pool
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
        inf.set_frame_buffer(&read_fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();
        assert_eq!(readback, pixels, "sanity: the read must still be correct");

        let created = crate::probe::get(crate::probe::Probe::CodecCreated);
        // A small margin over `n` for the pool's own warm-up races (a few
        // extra checkouts can miss the free list before the first
        // checkins land) -- nowhere close to 256 (one per chunk), which is
        // exactly what the pre-CodecPool `map_init` measurement showed.
        assert!(
            created <= (n as u64) * 2,
            "expected CodecPool to construct roughly num_threads={n} codecs for 256 chunks, \
             not one per chunk -- got {created} (map_init's own measured pre-fix behavior was \
             ~256/256, one per chunk, regardless of thread count)"
        );
        assert!(created > 0, "sanity: the pool must have constructed at least one codec");
    }

    /// An aliased frame buffer (`y_stride = 0`: every scanline broadcasts
    /// into the SAME row) forces [`crate::misc::plan_disjoint`] to report
    /// overlap, so the parallel path must fall back to a fully serial
    /// scatter — and still produce the same bytes as `num_threads = 0`
    /// (the "aliased frame buffer" case).
    #[test]
    fn parallel_read_aliased_frame_buffer_falls_back_and_matches_serial() {
        let (w, h) = (6i32, 12i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        // A single-row aliased frame buffer: y_stride = 0, so every scanline
        // is scattered on top of the same `row_bytes` region. The last
        // scanline in range "wins"; that must be true for every thread
        // count.
        let pixel_bytes = 10usize;
        let row_bytes = pixel_bytes * w as usize;
        let mk = |ty: PixelType, off: usize| Slice {
            type_: ty, base_offset: off as i64, x_stride: pixel_bytes, y_stride: 0,
            x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
        };
        let mut aliased_fb = FrameBuffer::new();
        aliased_fb.insert("R", mk(PixelType::Half, 0)).unwrap();
        aliased_fb.insert("G", mk(PixelType::Half, 2)).unwrap();
        aliased_fb.insert("B", mk(PixelType::Half, 4)).unwrap();
        aliased_fb.insert("A", mk(PixelType::Float, 6)).unwrap();

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 4, 9] {
            let mut readback = vec![0u8; row_bytes];
            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&aliased_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();

            // Must equal the LAST scanline of the written image (row h-1),
            // since every earlier scanline is overwritten in file order.
            let last_row = &pixels[(h as usize - 1) * row_bytes..];
            crate::probe::assert_declined(crate::probe::Probe::ScanReadParallel);
            assert_eq!(readback, last_row, "num_threads={n}: expected the last scanline to win");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    /// Same matrix, but reading in one-line-at-a-time sub-range calls (the
    /// C++ `singleScan` shape) rather than the whole image in one call —
    /// exercises the serial fallback path repeatedly per thread count
    /// (a 1-chunk request is never parallel) alongside the multi-chunk case
    /// above.
    #[test]
    fn parallel_read_sub_ranges_match_across_thread_counts() {
        let (w, h) = (11i32, 20i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 4] {
            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);

            let mut is = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let mut inf = InputFile::with_num_threads(&mut is, Some(n)).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            for y in 0..h {
                inf.read_pixels(y, y, &mut readback).unwrap();
            }

            crate::probe::assert_declined(crate::probe::Probe::ScanReadParallel); // 1-line reads are never parallel
            assert_eq!(readback, pixels, "num_threads={n}: mismatch vs the written pixels");
            match &reference {
                None => reference = Some(readback),
                Some(r) => assert_eq!(*r, readback, "num_threads={n}: differs from num_threads=0"),
            }
        }
    }

    #[test]
    fn header_and_version_recovered() {
        let (w, h) = (4i32, 2i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let mut is = MemIStream::new(bytes);
        let inf = InputFile::new(&mut is).unwrap();
        // Magic + version word must be a plain single-part image (no flags).
        assert_eq!(
            (inf.version() as u32) & version::VERSION_NUMBER_FIELD,
            version::EXR_VERSION
        );
        assert_eq!(inf.header().channels().unwrap().iter().count(), 4);
    }

    #[test]
    fn missing_channel_is_filled() {
        let (w, h) = (4i32, 2i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        // Read-back frame buffer OMITS R and adds a non-file channel Z (fill).
        let pixel_bytes = 10usize;
        let row_bytes = pixel_bytes * w as usize;
        let mut readback = vec![0u8; row_bytes * h as usize];
        readback.iter_mut().for_each(|b| *b = 0);

        let mut read_fb = FrameBuffer::new();
        // G, B, A present; R omitted; Z is a fill channel with fillValue 7.0.
        read_fb.insert("G", Slice {
            type_: PixelType::Half, base_offset: 2, x_stride: pixel_bytes, y_stride: row_bytes,
            x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
        }).unwrap();
        read_fb.insert("Z", Slice {
            type_: PixelType::Float, base_offset: 6, x_stride: pixel_bytes, y_stride: row_bytes,
            x_sampling: 1, y_sampling: 1, fill_value: 7.0, x_tile_coords: false, y_tile_coords: false,
        }).unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&read_fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();

        // G must equal the written G plane; Z must be the fill value everywhere.
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = y * row_bytes + x * pixel_bytes;
                let g_expected = Half::from(200.0 + (y * 10 + x) as f32).bits();
                let g_got = u16::from_ne_bytes([readback[p + 2], readback[p + 3]]);
                assert_eq!(g_got, g_expected, "G mismatch at ({x},{y})");
                let z_got = f32::from_ne_bytes([
                    readback[p + 6], readback[p + 7], readback[p + 8], readback[p + 9],
                ]);
                assert_eq!(z_got, 7.0, "Z fill mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn sub_range_reads_only_requested_lines() {
        let (w, h) = (4i32, 4i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let pixel_bytes = 10usize;
        let row_bytes = pixel_bytes * w as usize;
        let mut readback = vec![0u8; row_bytes * h as usize];

        let (_p, read_fb) = make_pixels(w as usize, h as usize);
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&read_fb).unwrap();
        // Only read the middle two scanlines [1, 2].
        inf.read_pixels(1, 2, &mut readback).unwrap();

        // Lines 1 and 2 must match; lines 0 and 3 must be untouched (zero).
        for x in 0..w as usize {
            for &y in &[1usize, 2usize] {
                let p = y * row_bytes + x * pixel_bytes;
                assert_eq!(&readback[p..p + 10], &pixels[p..p + 10], "line {y} px {x}");
            }
            for &y in &[0usize, 3usize] {
                let p = y * row_bytes + x * pixel_bytes;
                assert!(readback[p..p + 10].iter().all(|&b| b == 0), "line {y} should be zero");
            }
        }
    }

    #[test]
    fn corrupt_data_size_errors_not_panics() {
        let (w, h) = (4i32, 2i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let mut bytes = write_exr(&header, &pixels, &fb, h);

        // Corrupt the first chunk's dataSize to a huge value.  The chunk starts
        // at the offset stored in the (now-filled) first offset-table entry.
        // We locate it by reading the header+table once via a throwaway reader.
        let chunk0_off = {
            let mut is = MemIStream::new(bytes.clone());
            let inf = InputFile::new(&mut is).unwrap();
            inf.chunk_offsets()[0] as usize
        };
        // dataSize is the 4 bytes after the 4-byte y at the chunk start.
        // The on-disk dataSize field is i32 little-endian (matches the reader
        // and container::xdr), so write the huge sentinel as little-endian.
        let ds_pos = chunk0_off + 4;
        bytes[ds_pos..ds_pos + 4].copy_from_slice(&0x7FFF_FFFFi32.to_le_bytes());

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let err = inf.read_pixels(0, h - 1, &mut vec![0u8; pixels.len()]);
        assert!(err.is_err(), "corrupt dataSize must error, not panic");
    }

    // -----------------------------------------------------------------------
    // Offset-table hardening + chunk identity
    // -----------------------------------------------------------------------

    /// `(table_byte_position, entry_count)` of a written file's offset table.
    /// The table ends exactly where the first chunk begins, so the lowest
    /// stored offset marks its end (true for both line orders).
    fn offset_table_span(bytes: &[u8]) -> (usize, usize) {
        let mut is = MemIStream::new(bytes.to_vec());
        let inf = InputFile::new(&mut is).unwrap();
        let count = inf.chunk_offsets().len();
        let first_chunk = inf.chunk_offsets().iter().copied().min().unwrap() as usize;
        (first_chunk - count * 8, count)
    }

    /// Matches C++'s lazy semantics -- C++ `InputFile`/`MultiPartInputFile` constructors
    /// NEVER touch the chunk-offset table at all (`extract_chunk_table` runs
    /// only from `isComplete()`/an actual chunk read), and every one of its
    /// validation failures (including this one -- a `chunkCount` over the
    /// 2^20 cap) is swallowed by `Context::chunkTableValid` into a plain
    /// `isComplete() == false`, never a constructor throw. A header-only
    /// file whose declared chunk count is absurd must therefore OPEN fine;
    /// only `is_complete()` and a real `read_pixels` attempt see the
    /// problem. The DoS protection stays: no allocation is EVER attempted
    /// for the invalid count (see `chunk::offsets::read_offset_table_lenient`).
    #[test]
    fn oversized_chunk_count_defers_to_is_complete_not_a_constructor_error() {
        let header = make_header(4, 1 << 21, LineOrder::IncreasingY);
        let mut out = MemOStream::new();
        header.write_to(&mut out, 0).unwrap();
        let bytes = out.into_data();
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).expect("must open despite the absurd chunk count");
        assert!(!inf.is_complete(), "the table can never be valid for this count");
        let (_, fb) = make_pixels(4, 1);
        let mut out_buf = vec![0u8; 4 * (2 + 2 + 2 + 4)];
        // set_frame_buffer only records the layout and must succeed; the failure
        // surfaces at read time, exactly like C++'s `readPixels` throw.
        inf.set_frame_buffer(&fb)
            .expect("set_frame_buffer must succeed so read_pixels fails for the table, not for a missing frame buffer");
        assert!(
            inf.read_pixels(0, 0, &mut out_buf).is_err(),
            "reading must still fail cleanly -- just not at construction"
        );
    }

    /// Same lazy semantics as the test above -- C++ never rejects a file at open time merely
    /// because its declared chunk-offset table doesn't fit the file; that
    /// failure is deferred to `isComplete() == false` / a real read. The
    /// crafted-file case (a tiny file whose header declares far more chunks
    /// than the file has bytes) must therefore OPEN, report incomplete, and
    /// only fail when actually read -- with NO large allocation ever
    /// attempted for the bogus size (this test completing at all, fast, is
    /// itself evidence of that).
    #[test]
    fn oversized_offset_table_defers_to_is_complete_not_a_constructor_error() {
        let header = make_header(4, 4096, LineOrder::IncreasingY);
        let mut out = MemOStream::new();
        header.write_to(&mut out, 0).unwrap();
        let bytes = out.into_data();
        assert!(bytes.len() < 4096 * 8, "fixture must be smaller than its table");
        let mut is = MemIStream::new(bytes);
        let mut inf =
            InputFile::new(&mut is).expect("must open despite a table larger than the file");
        assert!(!inf.is_complete(), "the table cannot fit in this file");
        let (_, fb) = make_pixels(4, 1);
        let mut out_buf = vec![0u8; 4 * (2 + 2 + 2 + 4)];
        inf.set_frame_buffer(&fb)
            .expect("set_frame_buffer must succeed so read_pixels fails for the table, not for a missing frame buffer");
        assert!(
            inf.read_pixels(0, 0, &mut out_buf).is_err(),
            "reading must still fail cleanly -- just not at construction"
        );
    }

    /// A backward offset used to make `read_pixels` loop forever (the chunk's
    /// `y` was never compared against the block the slot addresses).  It must
    /// now be a clean error — checked under a hard time bound.
    #[test]
    fn backward_chunk_offset_errors_instead_of_looping_forever() {
        let (w, h) = (4i32, 4i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let mut bytes = write_exr(&header, &pixels, &fb, h);

        // Point slot 1 at chunk 0 (a backward offset).
        let (table_pos, count) = offset_table_span(&bytes);
        assert!(count >= 2);
        let chunk0 = bytes[table_pos..table_pos + 8].to_vec();
        bytes[table_pos + 8..table_pos + 16].copy_from_slice(&chunk0);

        let (tx, rx) = std::sync::mpsc::channel();
        let len = pixels.len();
        std::thread::spawn(move || {
            let mut is = MemIStream::new(bytes);
            let mut inf = InputFile::new(&mut is).unwrap();
            let (_p, read_fb) = make_pixels(w as usize, h as usize);
            inf.set_frame_buffer(&read_fb).unwrap();
            let r = inf.read_pixels(0, h - 1, &mut vec![0u8; len]);
            let _ = tx.send(r.is_err());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(20)) {
            Ok(is_err) => assert!(is_err, "a backward chunk offset must error"),
            Err(_) => panic!("read_pixels hung on a backward chunk offset"),
        }
    }

    /// A forward offset used to leave the requested scan line silently zeroed.
    #[test]
    fn forward_chunk_offset_errors_instead_of_zeroing_lines() {
        let (w, h) = (4i32, 4i32);
        let header = make_header(w, h, LineOrder::IncreasingY);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let mut bytes = write_exr(&header, &pixels, &fb, h);

        // Point slot 0 at chunk 1 (a forward offset).
        let (table_pos, _count) = offset_table_span(&bytes);
        let chunk1 = bytes[table_pos + 8..table_pos + 16].to_vec();
        bytes[table_pos..table_pos + 8].copy_from_slice(&chunk1);

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        let (_p, read_fb) = make_pixels(w as usize, h as usize);
        inf.set_frame_buffer(&read_fb).unwrap();
        assert!(
            inf.read_pixels(0, 0, &mut vec![0u8; pixels.len()]).is_err(),
            "a forward chunk offset must error, not silently zero the line"
        );
    }

    /// A file whose offset table was never written (an interrupted write) must
    /// still read — C++ `reconstructLineOffsets`.
    #[test]
    fn zeroed_offset_table_is_reconstructed() {
        for comp in [Compression::NoCompression, Compression::Zip] {
            let (w, h) = (5i32, 20i32);
            let header = make_header_comp(w, h, LineOrder::IncreasingY, comp);
            let (pixels, fb) = make_pixels(w as usize, h as usize);
            let mut bytes = write_exr(&header, &pixels, &fb, h);

            let (table_pos, count) = offset_table_span(&bytes);
            for b in bytes[table_pos..table_pos + count * 8].iter_mut() {
                *b = 0;
            }

            let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
            readback.iter_mut().for_each(|b| *b = 0);
            let mut is = MemIStream::new(bytes);
            let mut inf = InputFile::new(&mut is).unwrap();
            inf.set_frame_buffer(&read_fb).unwrap();
            inf.read_pixels(0, h - 1, &mut readback).unwrap();
            assert_eq!(readback, pixels, "reconstructed read mismatch for {comp:?}");
        }
    }

    /// Build the same header as [`make_header`] but with an explicit
    /// compression type (used to exercise the RLE write→read path).
    fn make_header_comp(w: i32, h: i32, line_order: LineOrder, comp: Compression) -> Header {
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

    /// The planar read (`read_pixels_planes`): ONE decode serves several destination buffers, each
    /// with its own base. Every codec family x line order x thread count (serial, parallel windows,
    /// more threads than chunks) must give the planes the single-buffer read gives, converted at
    /// unpack (HALF file channels into FLOAT planes), for the whole image and for a row sub-range;
    /// a file channel without a destination (G) is skipped, a destination without a file channel is
    /// filled, both in their own planes.
    #[test]
    fn planar_read_matches_the_single_buffer_read() {
        let (w, h) = (23usize, 40usize);
        for comp in [Compression::NoCompression, Compression::Rle, Compression::Zip, Compression::Piz] {
            for order in [LineOrder::IncreasingY, LineOrder::DecreasingY] {
                let header = make_header_comp(w as i32, h as i32, order, comp);
                let (pixels, fb) = make_pixels(w, h);
                let bytes = write_exr(&header, &pixels, &fb, h as i32);
                // Expected float planes straight from the written interleaved pixels.
                let px = 10usize;
                let half_at = |y: usize, x: usize, off: usize| {
                    let p = y * px * w + x * px + off;
                    f32::from(Half::from_bits(u16::from_ne_bytes([pixels[p], pixels[p + 1]])))
                };
                let a_at = |y: usize, x: usize| {
                    let p = y * px * w + x * px + 6;
                    f32::from_ne_bytes([pixels[p], pixels[p + 1], pixels[p + 2], pixels[p + 3]])
                };
                for threads in [0i32, 1, 2, 4, 64] {
                    for (s1, s2) in [(0usize, h - 1), (7, 29)] {
                        let slice = |ty| Slice {
                            type_: ty,
                            base_offset: 0,
                            x_stride: 4,
                            y_stride: w * 4,
                            x_sampling: 1,
                            y_sampling: 1,
                            fill_value: 0.5,
                            x_tile_coords: false,
                            y_tile_coords: false,
                        };
                        let mut fbp = FrameBuffer::new();
                        // plane 0 = B, 1 = R, 2 = A, 3 = X (no file channel: filled with 0.5); G skipped.
                        fbp.insert_in("B", slice(PixelType::Float), 0).unwrap();
                        fbp.insert_in("R", slice(PixelType::Float), 1).unwrap();
                        fbp.insert_in("A", slice(PixelType::Float), 2).unwrap();
                        fbp.insert_in("X", slice(PixelType::Float), 3).unwrap();
                        let mut p0 = vec![0u8; w * h * 4];
                        let mut p1 = vec![0u8; w * h * 4];
                        let mut p2 = vec![0u8; w * h * 4];
                        let mut p3 = vec![0u8; w * h * 4];
crate::probe::reset(crate::probe::Probe::ScanReadParallel);
                        // Borrowed bytes: the decoder copies nothing of the file out of the caller's slice.
                        let mut is = MemIStream::new(&bytes[..]);
                        let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
                        inf.set_frame_buffer(&fbp).unwrap();
                        inf.read_pixels_planes(
                            s1 as i32,
                            s2 as i32,
                            &mut [&mut p0[..], &mut p1[..], &mut p2[..], &mut p3[..]],
                        )
                        .unwrap();
                        // The multi-plane parallel planner really ran (>1 chunk and >1 thread), or the
                        // read stayed serial: the probe counts planner engagements on this thread.
                        let lines = header.compression().unwrap().num_scanlines() as usize;
                        let chunks = s2 / lines - s1 / lines + 1;
                        let want = u64::from(threads > 1 && chunks > 1);
                        assert_eq!(
                            crate::probe::get(crate::probe::Probe::ScanReadParallel),
                            want,
                            "{comp:?} threads={threads} rows {s1}..={s2}"
                        );
                        let f = |p: &[u8], y: usize, x: usize| {
                            let i = (y * w + x) * 4;
                            f32::from_ne_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]])
                        };
                        for y in 0..h {
                            for x in 0..w {
                                if (s1..=s2).contains(&y) {
                                    let ctx = format!("{comp:?} {order:?} threads={threads} rows {s1}..={s2} ({x},{y})");
                                    assert_eq!(f(&p0, y, x), half_at(y, x, 4), "B {ctx}");
                                    assert_eq!(f(&p1, y, x), half_at(y, x, 0), "R {ctx}");
                                    assert_eq!(f(&p2, y, x), a_at(y, x), "A {ctx}");
                                    assert_eq!(f(&p3, y, x), 0.5, "fill {ctx}");
                                } else {
                                    // Rows outside the request stay untouched in every plane.
                                    for p in [&p0, &p1, &p2, &p3] {
                                        assert_eq!(f(p, y, x), 0.0, "untouched row {y} {comp:?} {order:?}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// A slice that addresses the WRONG plane, or rows that overlap, is refused (or at
    /// worst written in place) identically at every thread count - never written into another plane,
    /// never a panic.
    #[test]
    fn planar_read_refuses_misaddressed_and_overlapping_slices_at_every_thread_count() {
        let (w, h) = (16usize, 40usize);
        let header = make_header_comp(w as i32, h as i32, LineOrder::IncreasingY, Compression::NoCompression);
        let (pixels, fb) = make_pixels(w, h);
        let bytes = write_exr(&header, &pixels, &fb, h as i32);
        let slice = |base: i64, y_stride: usize| Slice {
            type_: PixelType::Float,
            base_offset: base,
            x_stride: 4,
            y_stride,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        let plane = (w * h * 4) as i64;
        for threads in [0i32, 1, 4, 8] {
            // G (plane 1) addressed so that its rows fall inside plane 0.
            let mut fbm = FrameBuffer::new();
            fbm.insert_in("R", slice(0, w * 4), 0).unwrap();
            fbm.insert_in("G", slice(-plane, w * 4), 1).unwrap();
            let (mut p0, mut p1) = (vec![0u8; plane as usize], vec![0u8; plane as usize]);
            let mut is = MemIStream::new(&bytes[..]);
            let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
            inf.set_frame_buffer(&fbm).unwrap();
            let r = inf.read_pixels_planes(0, h as i32 - 1, &mut [&mut p0[..], &mut p1[..]]);
            assert!(r.is_err(), "threads={threads}: a misaddressed plane-1 slice must be refused, got Ok");

            // Every row of R at the SAME address (y_stride 0) + a third plane nothing writes to: the
            // overlapping intervals and the empty one that sorts between them must not panic.
            let mut fbo = FrameBuffer::new();
            fbo.insert_in("R", slice(0, 0), 0).unwrap();
            fbo.insert_in("G", slice(0, w * 4), 1).unwrap();
            let (mut a, mut b, mut c) = (vec![0u8; plane as usize], vec![0u8; plane as usize], vec![0u8; 64]);
            let mut is = MemIStream::new(&bytes[..]);
            let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
            inf.set_frame_buffer(&fbo).unwrap();
            // Ok or Err are both fine (the broadcast is legal); a panic is not.
            let _ = inf.read_pixels_planes(0, 33, &mut [&mut a[..], &mut b[..], &mut c[..]]);
        }
    }

    /// A slice that names a destination buffer that was not supplied is an `Arg` error, not a panic.
    #[test]
    fn planar_read_rejects_a_missing_destination_buffer() {
        let (w, h) = (4usize, 2usize);
        let header = make_header_comp(w as i32, h as i32, LineOrder::IncreasingY, Compression::NoCompression);
        let (pixels, fb) = make_pixels(w, h);
        let bytes = write_exr(&header, &pixels, &fb, h as i32);
        let mut fbp = FrameBuffer::new();
        let s = Slice {
            type_: PixelType::Float,
            base_offset: 0,
            x_stride: 4,
            y_stride: w * 4,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        fbp.insert_in("R", s, 3).unwrap();
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fbp).unwrap();
        let mut only = vec![0u8; w * h * 4];
        let err = inf.read_pixels_planes(0, h as i32 - 1, &mut [&mut only[..]]).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
    }

    /// Full RLE write→read round trip through `OutputFile`/`InputFile`.
    ///
    /// The KNOWN image (smooth gradients) compresses well, so the writer takes
    /// the genuine RLE-encoded path (chunk `dataSize` < uncompressed size), and
    /// the reader takes the `uncompress` path — exercising the codec end to end
    /// through both scanline halves.  Round-trip identity proves the predictor +
    /// reorder + RLE byte format inverts exactly.
    fn rle_round_trip(line_order: LineOrder) {
        let (w, h) = (9i32, 4i32);
        let header = make_header_comp(w, h, line_order, Compression::Rle);
        let (pixels, fb) = make_pixels(w as usize, h as usize);
        let bytes = write_exr(&header, &pixels, &fb, h);

        let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
        readback.iter_mut().for_each(|b| *b = 0);

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().compression().unwrap(), Compression::Rle);
        inf.set_frame_buffer(&read_fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();

        assert_eq!(readback, pixels, "RLE round-trip mismatch for {line_order:?}");
    }

    #[test]
    fn rle_round_trip_increasing_y_pixel_exact() {
        rle_round_trip(LineOrder::IncreasingY);
    }

    #[test]
    fn rle_round_trip_decreasing_y_pixel_exact() {
        rle_round_trip(LineOrder::DecreasingY);
    }

    /// RLE with high-entropy pixel data: the codec's "no-shrink → store raw"
    /// rule kicks in (chunk `dataSize` == uncompressed size), so the reader
    /// takes the raw passthrough.  Round-trip identity must still hold.
    #[test]
    fn rle_round_trip_incompressible_raw_store() {
        let (w, h) = (8i32, 3i32);
        let header = make_header_comp(w, h, LineOrder::IncreasingY, Compression::Rle);

        // Fill the pixel buffer with high-entropy bytes so RLE cannot shrink.
        let (mut pixels, fb) = make_pixels(w as usize, h as usize);
        let mut state: u32 = 0xBEEF_0007;
        for b in pixels.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (state >> 16) as u8;
        }
        let bytes = write_exr(&header, &pixels, &fb, h);

        let (mut readback, read_fb) = make_pixels(w as usize, h as usize);
        readback.iter_mut().for_each(|b| *b = 0);

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&read_fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();

        assert_eq!(readback, pixels, "RLE raw-store round-trip mismatch");
    }

    /// Regression: PXR24 is LOSSLESS for HALF, so a HALF-only image
    /// whose height is NOT a multiple of 16 (17 → a full 16-row chunk plus a 1-row
    /// partial trailing chunk) must round-trip bit-exact through the scanline
    /// writer/reader.  Runs both a compressible gradient (exercises the real
    /// compress/uncompress of the 1-row partial chunk) and high-entropy noise
    /// (exercises the codec's "no-shrink → store raw" partial chunk) — the lossless
    /// RGBA codecs pass this exact case, so any divergence is PXR24-specific.
    fn pxr24_half_partial_chunk(fill: impl Fn(usize, usize) -> [u16; 3]) {
        let (w, h) = (7i32, 17i32); // 17 = 16-row chunk + 1-row partial trailing chunk
        let mut hdr = Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Pxr24,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        // 3 interleaved HALF channels → 6 bytes/pixel.
        let pixel_bytes = 6usize;
        let row_bytes = pixel_bytes * w as usize;
        let mut pixels = vec![0u8; row_bytes * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = y * row_bytes + x * pixel_bytes;
                let [r, g, b] = fill(x, y);
                pixels[p..p + 2].copy_from_slice(&r.to_ne_bytes());
                pixels[p + 2..p + 4].copy_from_slice(&g.to_ne_bytes());
                pixels[p + 4..p + 6].copy_from_slice(&b.to_ne_bytes());
            }
        }
        let mk = |off: i64| Slice {
            type_: PixelType::Half,
            base_offset: off,
            x_stride: pixel_bytes,
            y_stride: row_bytes,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        let mut fb = FrameBuffer::new();
        fb.insert("R", mk(0)).unwrap();
        fb.insert("G", mk(2)).unwrap();
        fb.insert("B", mk(4)).unwrap();

        let bytes = write_exr(&hdr, &pixels, &fb, h);

        let mut readback = vec![0u8; pixels.len()];
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        inf.read_pixels(0, h - 1, &mut readback).unwrap();

        assert_eq!(
            readback, pixels,
            "PXR24 HALF 7×17 (16+1 partial trailing chunk) must be lossless"
        );
    }

    #[test]
    fn pxr24_half_partial_chunk_gradient_lossless() {
        pxr24_half_partial_chunk(|x, y| {
            [
                Half::from((y * 7 + x) as f32).bits(),
                Half::from((100 + y) as f32).bits(),
                Half::from((x * 3) as f32).bits(),
            ]
        });
    }

    #[test]
    fn pxr24_half_partial_chunk_noise_lossless() {
        // High-entropy HALF values (a valid finite half per sample) so PXR24 cannot
        // shrink the partial chunk and takes the raw-store path.
        pxr24_half_partial_chunk(|x, y| {
            let mut s = (0x9E37u32).wrapping_mul((y * 7 + x + 1) as u32);
            let mut next = || {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                // Keep the exponent small so the bits are a finite, non-NaN half.
                ((s >> 13) & 0x03FF | ((s >> 4) & 0x3C00)) as u16
            };
            [next(), next(), next()]
        });
    }

    /// An ISOLATED `scatter_chunk` throughput measurement — synthetic, already
    /// XDR-encoded pixel data in memory, NO stream I/O, NO decode/decompress
    /// at all — to settle whether the generic interleaved-per-sample scatter
    /// loop (as opposed to C++ 3.4.15's `unpack.c` specialized fast-path
    /// unpackers like `unpack_16bit_4chan_interleave`) is a meaningful
    /// fraction of the remaining ZIP/PIZ read gap, before deciding whether
    /// porting those fast paths is worth the engineering cost. Reports
    /// GB/s so it can be compared directly against the whole-file read
    /// throughput `tests/perf_ours_vs_cpp.rs` measures.
    ///
    /// Run explicitly (ignored by default):
    /// ```text
    /// cargo test -p exr-core --release --lib scanline::input::tests::scatter_chunk_only_throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn scatter_chunk_only_throughput() {
        use std::time::Instant;

        const WIDTH: usize = 16384;
        const LINES: usize = 8192;
        const PIXEL_BYTES: usize = 2 + 2 + 2 + 4; // R,G,B half + A float (matches perf_ours_vs_cpp.rs)

        // Interleaved RGBA(half,half,half,float) destination — the same
        // layout `perf_ours_vs_cpp.rs`'s `make_pixels` uses, confirmed
        // identical to `oracle.cpp`'s `bench_io`.
        let slices = vec![
            InSliceInfo { buffer: 0, local_base: 0,
                type_fb: PixelType::Half, type_file: PixelType::Half,
                base_offset: 0, x_stride: PIXEL_BYTES, y_stride: PIXEL_BYTES * WIDTH,
                x_sampling: 1, y_sampling: 1, fill: false, fill_value: 0.0, skip: false,
            },
            InSliceInfo { buffer: 0, local_base: 0,
                type_fb: PixelType::Half, type_file: PixelType::Half,
                base_offset: 2, x_stride: PIXEL_BYTES, y_stride: PIXEL_BYTES * WIDTH,
                x_sampling: 1, y_sampling: 1, fill: false, fill_value: 0.0, skip: false,
            },
            InSliceInfo { buffer: 0, local_base: 0,
                type_fb: PixelType::Half, type_file: PixelType::Half,
                base_offset: 4, x_stride: PIXEL_BYTES, y_stride: PIXEL_BYTES * WIDTH,
                x_sampling: 1, y_sampling: 1, fill: false, fill_value: 0.0, skip: false,
            },
            InSliceInfo { buffer: 0, local_base: 0,
                type_fb: PixelType::Float, type_file: PixelType::Float,
                base_offset: 6, x_stride: PIXEL_BYTES, y_stride: PIXEL_BYTES * WIDTH,
                x_sampling: 1, y_sampling: 1, fill: false, fill_value: 0.0, skip: false,
            },
        ];

        // One already-"decoded" (XDR little-endian) line's worth of bytes,
        // reused for every line (scatter_chunk never mutates its `decoded`
        // input) -- isolates scatter's own cost from any allocation.
        let mut one_line = vec![0u8; PIXEL_BYTES * WIDTH];
        for (i, b) in one_line.iter_mut().enumerate() {
            *b = (i * 2654435761u64 as usize) as u8; // cheap non-degenerate fill
        }

        let mut dest = vec![0u8; PIXEL_BYTES * WIDTH * LINES];
        let total_bytes = dest.len();

        let t0 = Instant::now();
        for y in 0..LINES as i32 {
            scatter_chunk(
                &slices,
                0,
                WIDTH as i32 - 1,
                y,
                y,
                0,
                LINES as i32 - 1,
                &one_line,
                &mut dest,
                0,
            )
            .unwrap();
        }
        let elapsed = t0.elapsed();

        let gbps = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!(
            "\nscatter_chunk-only: {WIDTH}x{LINES} interleaved RGBA(half,half,half,float): \
             {:.2}ms total, {:.2} GB/s (dest-buffer bytes/sec)",
            elapsed.as_secs_f64() * 1e3,
            gbps,
        );
        // Sanity: the loop actually did work (not read the same untouched
        // buffer over and over).
        assert!(dest.iter().any(|&b| b != 0), "scatter_chunk wrote nothing");
    }

    // -----------------------------------------------------------------------
    // `scatter_chunk_packed` equivalence harness — the packed fast path must be byte-identical to
    // `scatter_chunk_generic` (the source of truth) for every layout it
    // claims to handle, checked BEFORE it is trusted (it is already wired
    // into `scatter_chunk`, but only ever applies when its own precondition
    // holds — this proves that precondition is sufficient).
    // -----------------------------------------------------------------------

    /// Build `n` fully-packed, same-`ty` slices tiling one `n * size`-byte
    /// pixel — `perm` controls the channel ORDER within the pixel (e.g.
    /// `[1,0]` puts channel 1 first in memory) so the harness also exercises
    /// slices that are NOT already sorted by `base_offset`.
    fn packed_slices(ty: PixelType, n: usize, perm: &[usize], width: usize) -> Vec<InSliceInfo> {
        let size = crate::misc::pixel_type_size(ty).unwrap();
        let pixel_size = size * n;
        let y_stride = pixel_size * width;
        (0..n)
            .map(|logical_ch| {
                let slot = perm[logical_ch]; // this channel's position within the pixel
                InSliceInfo { buffer: 0, local_base: 0,
                    type_fb: ty,
                    type_file: ty,
                    base_offset: (slot * size) as i64,
                    x_stride: pixel_size,
                    y_stride,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill: false,
                    fill_value: 0.0,
                    skip: false,
                }
            })
            .collect()
    }

    /// `n` channels' worth of channel-planar (this crate's decoded-chunk
    /// layout: channel 0's whole row, then channel 1's, ...) synthetic
    /// bytes for `lines` rows of `width` samples each — a cheap deterministic
    /// non-degenerate pattern, not required to be realistic pixel values.
    fn packed_decoded(ty: PixelType, n: usize, width: usize, lines: usize) -> Vec<u8> {
        let size = crate::misc::pixel_type_size(ty).unwrap();
        let mut buf = Vec::with_capacity(lines * n * width * size);
        let mut counter = 0u32;
        for _y in 0..lines {
            for _ch in 0..n {
                for _x in 0..width {
                    counter = counter.wrapping_add(1);
                    match ty {
                        PixelType::Half => buf.extend_from_slice(&(counter as u16).to_le_bytes()),
                        PixelType::Float => buf.extend_from_slice(&(counter as f32).to_le_bytes()),
                        PixelType::Uint => buf.extend_from_slice(&counter.to_le_bytes()),
                        PixelType::NumPixelTypes => unreachable!(),
                    }
                }
            }
        }
        buf
    }

    #[test]
    fn scatter_chunk_packed_matches_generic_across_matrix() {
        for &ty in &[PixelType::Half, PixelType::Float] {
            for n in 1..=4usize {
                for width in [1usize, 2, 7, 16] {
                    for lines in [1usize, 3] {
                        // Two channel orderings: identity, and reversed (not
                        // sorted by base_offset) when there's more than one
                        // channel to permute.
                        let perms: Vec<Vec<usize>> = if n > 1 {
                            vec![(0..n).collect(), (0..n).rev().collect()]
                        } else {
                            vec![vec![0]]
                        };
                        for perm in perms {
                            let slices = packed_slices(ty, n, &perm, width);
                            let decoded = packed_decoded(ty, n, width, lines);
                            let pixel_size = crate::misc::pixel_type_size(ty).unwrap() * n;
                            let dest_len = pixel_size * width * lines;

                            let mut dest_fast = vec![0xAAu8; dest_len];
                            let mut dest_generic = vec![0xAAu8; dest_len];

                            let chunk_min_y = 0;
                            let chunk_end_y = lines as i32 - 1;
                            let min_x = 0;
                            let max_x = width as i32 - 1;

                            scatter_chunk_packed(
                                &slices, min_x, max_x, chunk_min_y, chunk_end_y, chunk_min_y,
                                chunk_end_y, &decoded, &mut dest_fast, 0,
                            )
                            .expect("packed path must apply for this exact-tiling layout")
                            .unwrap();

                            scatter_chunk_generic(
                                &slices, min_x, max_x, chunk_min_y, chunk_end_y, chunk_min_y,
                                chunk_end_y, &decoded, &mut dest_generic, 0,
                            )
                            .unwrap();

                            assert_eq!(
                                dest_fast, dest_generic,
                                "ty={ty:?} n={n} width={width} lines={lines} perm={perm:?}: \
                                 packed path diverged from the generic reference"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The packed path must DECLINE (return `None`) rather than silently
    /// mis-scatter for every precondition violation it documents: mixed
    /// types, subsampling, fill, skip, and a non-fully-packed `x_stride`
    /// (a gap between channels).
    #[test]
    fn scatter_chunk_packed_declines_outside_its_precondition() {
        let width = 4usize;
        let lines = 2usize;
        let decoded = packed_decoded(PixelType::Half, 2, width, lines);

        // Mixed types (HALF + FLOAT) — never qualifies.
        let mut mixed = packed_slices(PixelType::Half, 2, &[0, 1], width);
        mixed[1].type_fb = PixelType::Float;
        mixed[1].type_file = PixelType::Float;
        assert!(scatter_chunk_packed(&mixed, 0, width as i32 - 1, 0, lines as i32 - 1, 0, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());

        // Subsampled.
        let mut subsampled = packed_slices(PixelType::Half, 2, &[0, 1], width);
        subsampled[0].x_sampling = 2;
        assert!(scatter_chunk_packed(&subsampled, 0, width as i32 - 1, 0, lines as i32 - 1, 0, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());

        // Fill slice present.
        let mut filled = packed_slices(PixelType::Half, 2, &[0, 1], width);
        filled[1].fill = true;
        assert!(scatter_chunk_packed(&filled, 0, width as i32 - 1, 0, lines as i32 - 1, 0, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());

        // Skip slice present.
        let mut skipped = packed_slices(PixelType::Half, 2, &[0, 1], width);
        skipped[1].skip = true;
        assert!(scatter_chunk_packed(&skipped, 0, width as i32 - 1, 0, lines as i32 - 1, 0, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());

        // A gap between channels (x_stride bigger than the tight pack).
        let mut gapped = packed_slices(PixelType::Half, 2, &[0, 1], width);
        for s in &mut gapped {
            s.x_stride += 4; // no longer fully packed
        }
        assert!(scatter_chunk_packed(&gapped, 0, width as i32 - 1, 0, lines as i32 - 1, 0, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());

        // A partial-range chunk (chunk extends outside [min_scan, max_scan]).
        let partial = packed_slices(PixelType::Half, 2, &[0, 1], width);
        assert!(scatter_chunk_packed(&partial, 0, width as i32 - 1, 0, lines as i32 - 1, 1, lines as i32 - 1, &decoded, &mut [0u8; 64], 0).is_none());
    }

    /// Repeated row reads unpack one cached chunk into fresh frame buffers.
    #[test]
    fn row_reads_decode_dwab_once_and_survive_parallel_and_layout_changes() {
        use crate::probe::{self, Probe};
        for compression in [Compression::NoCompression, Compression::Zip, Compression::Dwab] {
            let lines = crate::compressor::num_lines_in_buffer(compression).unwrap();
            let (w, h) = (64, lines * 2 + 1);
            let header = make_header_comp(w, h, LineOrder::IncreasingY, compression);
            let (pixels, fb) = make_pixels(w as usize, h as usize);
            let bytes = write_exr(&header, &pixels, &fb, h);
            if compression != Compression::NoCompression {
                assert!(bytes.len() < pixels.len(), "premise: the file contains compressed chunks");
            }
            let mut reference_stream = MemIStream::new(bytes.clone());
            let mut reference_reader = InputFile::with_num_threads(&mut reference_stream, Some(0)).unwrap();
            reference_reader.set_frame_buffer(&fb).unwrap();
            let mut reference = vec![0u8; pixels.len()];
            reference_reader.read_pixels(0, h-1, &mut reference).unwrap();
            let row_bytes = 10 * w as usize;
            for threads in [0, 4] {
                let mut stream = MemIStream::new(bytes.clone());
                let mut reader = InputFile::with_num_threads(&mut stream, Some(threads)).unwrap();
                probe::reset(Probe::DecodeChunk);
                for y in 0..lines {
                    let mut local_fb = FrameBuffer::new();
                    for (name, slice) in fb.iter() {
                        let mut local = slice.clone();
                        local.y_stride = 0;
                        local_fb.insert(name.clone(), local).unwrap();
                    }
                    reader.set_frame_buffer(&local_fb).unwrap();
                    let mut row = vec![0xa5; row_bytes];
                    reader.read_pixels(y, y, &mut row).unwrap();
                    assert_eq!(row, reference[y as usize * row_bytes..(y as usize+1)*row_bytes]);
                }
                assert_eq!(probe::get(Probe::DecodeChunk), 1, "{compression:?}: one decode for {lines} row calls");
                reader.set_frame_buffer(&fb).unwrap();
                let mut whole = vec![0u8; reference.len()];
                probe::reset(Probe::ScanReadParallel);
                reader.read_pixels(0, h-1, &mut whole).unwrap();
                assert_eq!(whole, reference);
                probe::assert_engaged(Probe::ScanReadParallel, threads);
                // A new destination layout can reuse the last serial chunk without stale pixels.
                probe::reset(Probe::DecodeChunk);
                let mut repeated = vec![0u8; reference.len()];
                let y = h-1;
                reader.read_pixels(y, y, &mut repeated).unwrap();
                assert_eq!(&repeated[y as usize * row_bytes..], &reference[y as usize * row_bytes..]);
                assert_eq!(probe::get(Probe::DecodeChunk), if threads > 1 { 1 } else { 0 });
                repeated.fill(0xa5);
                reader.read_pixels(y, y, &mut repeated).unwrap();
                assert_eq!(probe::get(Probe::DecodeChunk), if threads > 1 { 1 } else { 0 });
                assert_eq!(&repeated[y as usize * row_bytes..], &reference[y as usize * row_bytes..]);
            }
        }
    }
}
