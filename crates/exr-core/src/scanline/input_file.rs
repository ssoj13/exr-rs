//! Port of OpenEXR `ImfInputFile.cpp` — the **generic reader front door**.
//!
//! `InputFile` is not a reader; it is the dispatcher C++ calls
//! `InputFile::initialize` (`ImfInputFile.cpp:292-323`): it looks at the part's
//! `exr_storage_t` and constructs the reader that storage needs —
//! `ScanLineInputFile` for a scanline part, `TiledInputFile` for a tiled part
//! (whose tiles it then serves *as scanlines*), `DeepScanLineInputFile` +
//! `CompositeDeepScanLine` for a deep part.
//!
//! Exactly as in C++, the file's magic + version word, header list and chunk
//! offset tables are read **once**, by the multipart prologue reader
//! (`multipart::input::read_prologue`) — C++ `InputFile` literally
//! constructs a `MultiPartInputFile` and takes part 0
//! (`ImfInputFile.cpp:126-136`, with `strictHeaderValidation(false)` and
//! `autoAddType == false`).
//!
//! Port target (pinned tag v3.3.11 / the 3.4 tree for `initialize`):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfInputFile.cpp>

use std::ops::DerefMut;

use imath_rs::Box2i;

use crate::attr::PixelType;
use crate::error::{ExrError, Result, Unsupported};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::io::IStream;
use crate::misc::pixel_type_size;
use crate::multipart::input::{read_prologue, Storage};
use crate::tiled::{TiledInputFile, TiledInputFileState};

use super::input::{ScanLineInputFile, ScanLineInputFileState};

/// The concrete reader `InputFile` dispatched to — C++ `Data::_sFile` /
/// `_tFile` / `_dsFile`.
enum Reader<'a, S = &'a mut dyn IStream>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// `EXR_STORAGE_SCANLINE` → `ScanLineInputFile` (`ImfInputFile.cpp:312-315`).
    ///
    /// Boxed (clippy `large_enum_variant`): `ScanLineInputFile` carries its
    /// own line-offset table + gather scratch state inline, so the enum's
    /// unboxed size was dominated by whichever variant is bigger regardless
    /// of which one is live — every `InputFile` (including the far more
    /// common scanline case) was paying for the larger variant's stack
    /// space. Boxing both variants (`Tiled` already was) keeps `Reader`
    /// itself pointer-sized; this is a representation change only, not a
    /// C++ structural difference (`Data` holds pointers either way).
    ScanLine(Box<ScanLineInputFile<'a, S>>),
    /// `EXR_STORAGE_TILED` → `TiledInputFile` + the tile-row cache
    /// (`ImfInputFile.cpp:306-311`, `Data::bufferedReadPixels`).
    Tiled(Box<TiledReader<'a, S>>),
    /// `EXR_STORAGE_DEEP_SCANLINE` → `DeepScanLineInputFile` flattened through
    /// a `CompositeDeepScanLine` (`ImfInputFile.cpp:300-305`). See
    /// [`DeepReader`] for why the compositor is rebuilt per call rather than
    /// stored.
    Deep(Box<DeepReader<'a, S>>),
}

/// The tiled-as-scanlines path — C++ `Data::_tFile` plus `_cacheFrameBuffer`,
/// `_cachedBuffer`, `_cachedOffset` and `_cachedTileY`.
///
/// Used only by the STANDALONE `InputFile` (`InputFile::new`): a fresh
/// `TiledReader` is built once and lives for the reader's whole life, so it
/// keeps owning a genuine `TiledInputFile<'a>` (with its own embedded
/// stream) — no caching concern here, and this is also what lets
/// [`InputFile::as_tiled_input`] keep handing back a real
/// `&mut TiledInputFile<'a>` (needed by the `TiledCopySource` trait; a
/// borrowed-state view cannot satisfy that signature). The multipart-cache
/// path (`MultiPartInputFile::input_part`) uses the SEPARATE, split
/// [`TiledReaderState`] below instead — the same "own the state, don't
/// change this working standalone struct" choice fix-tiled made for the
/// analogous write-side/read-side splits elsewhere in this crate.
struct TiledReader<'a, S = &'a mut dyn IStream> {
    file: TiledInputFile<'a, S>,
    /// The frame buffer the caller set — C++ `_cacheFrameBuffer`.
    user_fb: FrameBuffer,
    /// One row of tiles' worth of pixels, addressed by [`Self::cache_fb`] —
    /// C++ `_cachedBuffer` (there: one heap block per channel).
    cache_bytes: Vec<u8>,
    /// Slices into [`Self::cache_bytes`], all `yTileCoords = true` so the same
    /// block is reused for every tile row — C++ `_cachedBuffer`'s slices.
    cache_fb: FrameBuffer,
    /// Which tile row `cache_bytes` currently holds — C++ `_cachedTileY`
    /// (`-1` = nothing cached).
    cached_tile_y: i32,
    /// The part's data window (cached; C++ re-reads it from the context).
    data_window: Box2i,
}

/// `EXR_STORAGE_DEEP_SCANLINE` served through `InputFile` — C++
/// `Data::_dsFile` + `Data::_compositor` (`ImfInputFile.cpp:300-305`,
/// `lockedSetFrameBuffer`'s deep branch).
///
/// # Why the compositor is rebuilt per call, not stored
/// C++'s `_compositor` is a `unique_ptr<CompositeDeepScanLine>` that outlives
/// `_dsFile` inside the SAME `Data`, safe there only because `_dsFile` is a
/// raw pointer into heap memory that never moves. `CompositeDeepScanLine<'a>`
/// borrows its sources (`Vec<&'a mut dyn DeepScanLineCompositeSource>`), so
/// storing both the owned `DeepScanLineInputFile` AND a compositor borrowing
/// it in the same struct is self-referential — impossible in safe Rust (no
/// `unsafe`/`Pin`/`ouroboros` in this crate). Every
/// piece of a freshly-built `CompositeDeepScanLine`'s state
/// (`channels`/`buffer_map`/`output_frame_buffer` from `set_frame_buffer`;
/// `zback`/`data_window`/`display_window` from `add_source`, themselves
/// deterministic functions of the source's header) is cheap and exactly
/// reproducible from the owned source + the remembered flat `FrameBuffer`, so
/// a fresh, short-lived compositor is built inside each
/// [`DeepReader::set_frame_buffer`]/[`DeepReader::read_pixels`] call instead
/// — same bytes out, no self-reference.
struct DeepReader<'a, S = &'a mut dyn IStream>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    file: crate::deep::DeepScanLineInputFile<'a, S>,
    /// The caller's flat frame buffer, remembered so [`read_pixels`] can
    /// rebuild an equivalent compositor every call — C++ `_compositor`'s
    /// already-set `_outputFrameBuffer` (there, this is implicit state on the
    /// persisted compositor object itself).
    user_fb: Option<FrameBuffer>,
}

impl<'a, S> DeepReader<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Validate `frame_buffer` against the deep source's channels — port of
    /// `Data::lockedSetFrameBuffer`'s deep branch's `_compositor->setFrameBuffer`
    /// call. Rebuilds a throwaway compositor purely to run the same channel
    /// bookkeeping / validation C++'s persisted one would; only the
    /// validation result and the buffer itself are kept.
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        let mut compositor = crate::deep::CompositeDeepScanLine::new();
        compositor.add_source(&mut self.file)?;
        compositor.set_frame_buffer(frame_buffer)?;
        self.user_fb = Some(frame_buffer.clone());
        Ok(())
    }

    /// Read scanlines `[min(s1,s2), max(s1,s2)]`, flattened, into `out` —
    /// port of `Data::readPixels`'s deep branch (`_compositor->readPixels`).
    /// Rebuilds the same compositor `set_frame_buffer` validated, then reads
    /// through it — see [`DeepReader`]'s doc comment for why this is
    /// byte-identical to C++'s persisted-compositor design.
    fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()> {
        let frame_buffer = self.user_fb.clone().ok_or_else(|| {
            ExrError::Arg("No frame buffer was set.".into())
        })?;
        let mut compositor = crate::deep::CompositeDeepScanLine::new();
        compositor.add_source(&mut self.file)?;
        compositor.set_frame_buffer(&frame_buffer)?;
        compositor.read_pixels(scan_line1, scan_line2, out)
    }
}

/// The tiled-as-scanlines path's PERSISTABLE state — the read-side mirror
/// of the writers' per-part state caching: everything [`TiledReader`] owns except its
/// stream, so [`crate::multipart::MultiPartInputFile`] can persist one of
/// these per tiled-storage part across separate `input_part` calls instead
/// of rebuilding (and silently discarding the tile-row cache of) a fresh one
/// every time — the read-side analogue of `MultiPartOutputFile`'s
/// `tiled_state`/`scanline_state` caching. `tiled` is itself already a
/// stream-free state object ([`TiledInputFileState`], the tiled reader's own
/// split); this struct simply bundles it with the SAME extra cache fields
/// [`TiledReader`] carries, so the whole tiled-as-scanlines emulation —
/// underlying reader state AND row cache — survives across calls together.
///
/// Wired into `MultiPartInputFile::input_part` once
/// the `TiledCopySource` trait redesign (`tiled/output.rs`) unblocked it.
pub(crate) struct TiledReaderState {
    tiled: TiledInputFileState,
    /// The frame buffer the caller set — C++ `_cacheFrameBuffer`.
    user_fb: FrameBuffer,
    /// One row of tiles' worth of pixels — C++ `_cachedBuffer`.
    cache_bytes: Vec<u8>,
    /// Slices into `cache_bytes` — C++ `_cachedBuffer`'s slices.
    cache_fb: FrameBuffer,
    /// Which tile row `cache_bytes` currently holds (`-1` = nothing cached).
    cached_tile_y: i32,
    /// The part's data window.
    data_window: Box2i,
}

impl TiledReaderState {
    /// Pure state construction — mirrors `TiledReader`'s own construction in
    /// [`InputFile::build`], minus the stream (already fully consumed by
    /// `multipart::input::read_prologue` before this ever runs, same as
    /// every other `*State::from_part_mt` in this crate). `num_threads` is
    /// the per-file override (`None` = capture the current
    /// process-wide default).
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let data_window = header.data_window()?;
        let tiled = TiledInputFileState::from_part_mt(
            header,
            version,
            part_number,
            chunk_offsets,
            num_threads,
        )?;
        Ok(Self {
            tiled,
            user_fb: FrameBuffer::new(),
            cache_bytes: Vec::new(),
            cache_fb: FrameBuffer::new(),
            cached_tile_y: -1,
            data_window,
        })
    }

    pub(crate) fn header(&self) -> &Header {
        self.tiled.header()
    }

    pub(crate) fn version(&self) -> i32 {
        self.tiled.version()
    }

    /// This part's on-disk tile order — needed by [`InputFilePartReader`]'s
    /// `TiledCopySource` support (`RANDOM_Y` copy_pixels walks it).
    pub(crate) fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        self.tiled.tile_order()
    }

    /// Read tile `(dx, dy, lx, ly)`'s raw on-disk bytes — needed by
    /// [`InputFilePartReader`]'s `TiledCopySource` support.
    pub(crate) fn raw_tile_data(
        &mut self,
        stream: &mut dyn IStream,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        self.tiled.raw_tile_data(stream, dx, dy, lx, ly)
    }

    /// Body IDENTICAL to [`TiledReader::set_frame_buffer`], operating on
    /// `self.tiled` instead of `self.file` — see that method's docs.
    pub(crate) fn set_frame_buffer(
        &mut self,
        frame_buffer: &FrameBuffer,
        complete: bool,
    ) -> Result<()> {
        if !complete {
            return Err(ExrError::Arg(
                "Unable to use generic API to read with (partially?) corrupt \
                 chunk table."
                    .into(),
            ));
        }

        self.cache_bytes.clear();
        self.cache_fb = FrameBuffer::new();
        self.cached_tile_y = -1;

        let channels = self.tiled.header().channels()?;
        let dw = self.data_window;
        let cached_offset = dw.min.x as i64;
        let width = (dw.max.x as i64 - dw.min.x as i64 + 1).max(0);
        let tile_y = self.tiled.tile_y_size() as i64;
        let level_width = self.tiled.level_width(0)? as i64;

        for (name, slice) in frame_buffer.iter() {
            if channels.find_channel(name).is_none() {
                continue;
            }
            // Design bug fix: C++'s `(s.type == HALF) ? 2 : 4`
            // (`ImfInputFile.cpp:407`) is safe there because a real
            // `Slice::Type` never holds the `NUM_PIXELTYPES` sentinel; our
            // `PixelType` enum can, so the ternary must not silently size an
            // invalid type as 4 bytes -- use the canonical, Result-returning
            // `pixel_type_size` and reject the sentinel instead. (Same fix
            // in `TiledReader::set_frame_buffer` below -- this method's
            // exact duplicate, operating on `self.tiled` instead of
            // `self.file`.)
            let bytes = pixel_type_size(slice.type_)? as i64;
            let tile_bytes = bytes
                .checked_mul(tile_y)
                .and_then(|v| v.checked_mul(width))
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| ExrError::Arg("Tile row cache size overflows.".into()))?;
            let region = self.cache_bytes.len();
            self.cache_bytes.resize(region + tile_bytes, 0);

            self.cache_fb.insert(
                name,
                Slice::new(
                    slice.type_,
                    region as i64 - bytes * cached_offset,
                    bytes as usize,
                    (bytes * level_width) as usize,
                    1,
                    1,
                    slice.fill_value,
                    false,
                    true,
                )?,
            )?;
        }

        self.tiled.set_frame_buffer(&self.cache_fb)?;
        self.user_fb = frame_buffer.clone();
        Ok(())
    }

    /// Body IDENTICAL to [`TiledReader::buffered_read_pixels`], threading
    /// `stream` through every call that needs it instead of reading a
    /// `self.file`-embedded one.
    pub(crate) fn read_pixels(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        out: &mut [u8],
    ) -> Result<()> {
        if self.user_fb.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data destination.".into(),
            ));
        }

        let dw = self.data_window;
        let min_y = scan_line1.min(scan_line2);
        let max_y = scan_line1.max(scan_line2);

        if min_y < dw.min.y || max_y > dw.max.y {
            return Err(ExrError::Arg(
                "Tried to read scan line outside the image file's data window.".into(),
            ));
        }

        let tile_y_size = self.tiled.tile_y_size() as i32;
        if tile_y_size <= 0 {
            return Err(ExrError::Input("Invalid tile y size.".into()));
        }
        let min_dy = (min_y - dw.min.y) / tile_y_size;
        let max_dy = (max_y - dw.min.y) / tile_y_size;
        let last_x_tile = self.tiled.num_x_tiles(0)? - 1;

        if min_y == dw.min.y && max_y == dw.max.y {
            self.tiled.set_frame_buffer(&self.user_fb)?;
            let r = self
                .tiled
                .read_tiles(stream, 0, last_x_tile, min_dy, max_dy, 0, 0, out);
            self.tiled.set_frame_buffer(&self.cache_fb)?;
            self.cached_tile_y = -1;
            return r;
        }

        self.tiled.set_frame_buffer(&self.cache_fb)?;

        for j in min_dy..=max_dy {
            if j != self.cached_tile_y {
                if !self.cache_fb.is_empty() {
                    let mut cache = std::mem::take(&mut self.cache_bytes);
                    let r = self
                        .tiled
                        .read_tiles(stream, 0, last_x_tile, j, j, 0, 0, &mut cache);
                    self.cache_bytes = cache;
                    r?;
                }
                self.cached_tile_y = j;
            }

            let tile_range = self.tiled.data_window_for_tile(0, j, 0, 0)?;
            let min_y_this_row = min_y.max(tile_range.min.y);
            let max_y_this_row = max_y.min(tile_range.max.y);
            let x_start = dw.min.x;
            let width = dw.max.x - dw.min.x + 1;

            for (name, to) in self.user_fb.iter() {
                match self.cache_fb.find_slice(name) {
                    Some(from) => copy_cached_buffer(
                        to,
                        from,
                        &self.cache_bytes,
                        min_y_this_row,
                        max_y_this_row,
                        tile_range.min.y,
                        x_start,
                        width,
                        out,
                    )?,
                    None => fill_buffer(
                        to,
                        min_y_this_row,
                        max_y_this_row,
                        dw.min.y,
                        x_start,
                        width,
                        out,
                    )?,
                }
            }
        }

        Ok(())
    }
}

/// Generic single-part / part-0 EXR reader — port of `Imf::InputFile`.
///
/// Open with [`InputFile::new`], describe the destination pixels with
/// [`set_frame_buffer`](Self::set_frame_buffer), then pull scanlines with
/// [`read_pixels`](Self::read_pixels).  A **tiled** file is served as scanlines
/// (the tiles that intersect the requested rows are read and the rows copied
/// out), and part 0 of a **multipart** file is read like a single-part file —
/// both exactly as C++ `InputFile` does.
pub struct InputFile<'a, S = &'a mut dyn IStream>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    inner: Reader<'a, S>,
    /// C++ `InputFile::isComplete()` → `Context::chunkTableValid(part)`.
    complete: bool,
}

/// An [`InputFile`] that **owns** its input stream (`Box<dyn IStream>`) —
/// the read-side analogue of
/// [`scanline::OutputFileOwned`](crate::scanline::OutputFileOwned) /
/// [`multipart::MultiPartInputFileOwned`](crate::multipart::MultiPartInputFileOwned).
/// Build it with [`InputFile::open`]; behaves identically to the borrowed
/// default for the same input.
pub type InputFileOwned<'a> = InputFile<'a, Box<dyn IStream + 'a>>;

// Borrowed-stream constructor (the default `S = &'a mut dyn IStream`).
impl<'a> InputFile<'a, &'a mut dyn IStream> {
    /// Open `stream` and dispatch on part 0's storage — port of
    /// `InputFile::InputFile(IStream&, int)` + `InputFile::initialize`.
    ///
    /// # Errors
    /// * Propagates every prologue error (bad magic, unsupported version
    ///   number, unknown version flags, corrupt attributes, an unusable chunk
    ///   offset table).
    /// * [`ExrError::Unsupported`] with [`Unsupported::Deep`] for a **deep
    ///   tiled** part (a deep *scanline* part is flattened through
    ///   [`crate::deep::CompositeDeepScanLine`] instead — see
    ///   [`read_pixels`](Self::read_pixels)).
    /// * [`ExrError::Arg`] for a `type` this build does not know
    ///   (C++ "Unable to handle data storage type in file").
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::with_num_threads(stream, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    /// Effective on the **scanline** and **tiled** storage paths; deep parts
    /// still read serially regardless of `num_threads` (a known limitation).
    pub fn with_num_threads(stream: &'a mut dyn IStream, num_threads: Option<i32>) -> Result<Self> {
        // C++ `InputFile` builds a MultiPartInputFile with autoAddType = false
        // and takes part 0 (ImfInputFile.cpp:126-136).
        let prologue = read_prologue(stream, false)?;
        let multipart = prologue.multipart;
        let version = prologue.version;
        let part = prologue.parts.into_iter().next().ok_or_else(|| {
            ExrError::Input("File contains no parts.".into())
        })?;

        // A multipart file's chunks carry a leading `partNumber`; a single-part
        // file's do not.  Part 0 is `Data::kDefaultPart` (ImfInputFile.cpp:43).
        let part_number = if multipart { Some(0) } else { None };

        Self::build(
            stream,
            part.header,
            version,
            part.storage,
            part_number,
            part.complete,
            part.chunk_offsets,
            num_threads,
        )
    }
}

// Owned-stream constructor (`S = Box<dyn IStream + 'a>`): C++'s real,
// public `InputFile(const char fileName[], int numThreads)` constructor
// (`ImfInputFile.h:91`), previously impossible to provide here because
// `InputFile` had no owned-stream variant.
impl<'a> InputFile<'a, Box<dyn IStream + 'a>> {
    /// Open `path`, OWNING the decoded stream, and dispatch on part 0's
    /// storage — the owned-stream analogue of [`new`](InputFile::new).
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display()))
        })?;
        let stream: Box<dyn IStream + 'a> = Box::new(crate::io::StdIfStream::open(path_str)?);
        Self::new_owned(stream)
    }

    /// Stream form of [`open`](Self::open): open an [`InputFileOwned`] that
    /// owns the given boxed stream directly.
    pub fn new_owned(stream: Box<dyn IStream + 'a>) -> Result<Self> {
        let mut stream = stream;
        let prologue = read_prologue(&mut *stream, false)?;
        let multipart = prologue.multipart;
        let version = prologue.version;
        let part = prologue.parts.into_iter().next().ok_or_else(|| {
            ExrError::Input("File contains no parts.".into())
        })?;
        let part_number = if multipart { Some(0) } else { None };
        Self::build(
            stream,
            part.header,
            version,
            part.storage,
            part_number,
            part.complete,
            part.chunk_offsets,
            None,
        )
    }
}

// Shared read path, generic over the stream holder `S`. Covers BOTH the
// borrowed default (`S = &'a mut dyn IStream`) and the owned `Box<dyn
// IStream>` variant — every method below addresses the stream through
// `DerefMut<Target = dyn IStream>`, so the two paths are a single source of
// truth.
impl<'a, S> InputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// The storage dispatch itself — C++ `InputFile::initialize`
    /// (`ImfInputFile.cpp:292-323`).
    #[allow(clippy::too_many_arguments)] // mirrors the C++ ctor's own parameter list plus num_threads
    fn build(
        stream: S,
        header: Header,
        version: i32,
        storage: Storage,
        part_number: Option<i32>,
        complete: bool,
        chunk_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        let inner = match storage {
            Storage::Scanline => Reader::ScanLine(Box::new(ScanLineInputFile::from_part_mt(
                stream,
                header,
                version,
                part_number,
                chunk_offsets,
                num_threads,
            )?)),
            Storage::Tiled => {
                let data_window = header.data_window()?;
                let file = TiledInputFile::from_part_mt(
                    stream,
                    header,
                    version,
                    part_number,
                    chunk_offsets,
                    num_threads,
                )?;
                Reader::Tiled(Box::new(TiledReader {
                    file,
                    user_fb: FrameBuffer::new(),
                    cache_bytes: Vec::new(),
                    cache_fb: FrameBuffer::new(),
                    cached_tile_y: -1,
                    data_window,
                }))
            }
            // C++ `ImfInputFile.cpp:300-305` wraps a deep part in a
            // `CompositeDeepScanLine`, which flattens the deep samples into the
            // caller's flat frame buffer (`deep::CompositeDeepScanLine` —
            // see `DeepReader`'s doc comment for why the compositor itself is
            // rebuilt per call rather than stored alongside the file).
            Storage::DeepScanline => {
                let file = crate::deep::DeepScanLineInputFile::from_part_mt(
                    stream,
                    header,
                    version,
                    part_number,
                    chunk_offsets,
                    num_threads,
                )?;
                Reader::Deep(Box::new(DeepReader { file, user_fb: None }))
            }
            // C++'s generic `InputFile::initialize` (v3.4.15
            // `ImfInputFile.cpp:289-322`) groups `EXR_STORAGE_DEEP_TILED` with
            // `EXR_STORAGE_TILED` and constructs a plain `TiledInputFile` for
            // it — it never adds a compositor source for a deep-tiled part
            // (only the `EXR_STORAGE_DEEP_SCANLINE` branch does). The later
            // `lockedSetFrameBuffer`'s "if deep, use the compositor" branch
            // would therefore only ever see an empty, zero-source compositor
            // for a deep-tiled part — that C++ code path is dead for this
            // storage kind. Faking a working deep-tiled flatten here would
            // silently hand back wrong (all-zero/fill-value) pixels instead
            // of matching C++'s real (non-)behavior, so this stays a hard,
            // precisely-typed error (use `DeepTiledInputFile` directly).
            Storage::DeepTiled => {
                return Err(ExrError::Unsupported(Unsupported::Deep));
            }
            // C++ `initialize`'s final `else`: "Unable to handle data storage
            // type in file '<name>'".
            Storage::Unknown => {
                return Err(ExrError::Arg(
                    "Unable to handle data storage type in file.".into(),
                ));
            }
        };

        Ok(Self { inner, complete })
    }

    /// The part's header — port of `const Header& InputFile::header() const`.
    pub fn header(&self) -> &Header {
        match &self.inner {
            Reader::ScanLine(f) => f.header(),
            Reader::Tiled(t) => t.file.header(),
            Reader::Deep(d) => d.file.header(),
        }
    }

    /// The raw file-format version word — port of `int InputFile::version()`.
    pub fn version(&self) -> i32 {
        match &self.inner {
            Reader::ScanLine(f) => f.version(),
            Reader::Tiled(t) => t.file.version(),
            Reader::Deep(d) => d.file.version(),
        }
    }

    /// Is every chunk of this part present and addressable? — port of
    /// `bool InputFile::isComplete() const`
    /// (`_ctxt.chunkTableValid(part)`, evaluated *after* reconstruction).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Borrow the underlying tiled reader — port of
    /// `TiledInputFile& InputFile::asTiledInput() const` (`ImfInputFile.cpp:286-289`).
    /// C++ dereferences `_data->_tFile` unconditionally (UB if the part isn't
    /// tiled — the caller, `TiledOutputFile::copyPixels(InputFile&)`, is
    /// documented as only valid for a tiled source). This port returns a clean
    /// error instead of UB for a non-tiled part, per this crate's no-panic rule.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if this part's storage is not tiled.
    pub fn as_tiled_input(&mut self) -> Result<&mut TiledInputFile<'a, S>> {
        match &mut self.inner {
            Reader::Tiled(t) => Ok(&mut t.file),
            Reader::ScanLine(_) | Reader::Deep(_) => Err(ExrError::Arg(
                "Cannot call asTiledInput() on a file that is not tiled.".into(),
            )),
        }
    }

    /// This part's chunk offset table (unit tests inspect the on-disk layout).
    #[cfg(test)]
    pub(crate) fn chunk_offsets(&self) -> &[u64] {
        match &self.inner {
            Reader::ScanLine(f) => f.line_offsets(),
            Reader::Tiled(_) | Reader::Deep(_) => &[],
        }
    }

    /// Set the pixel destination — port of `InputFile::setFrameBuffer` /
    /// `Data::lockedSetFrameBuffer` (`ImfInputFile.cpp:334-444`).
    ///
    /// For a scanline part this forwards to `ScanLineInputFile`.  For a tiled
    /// part it additionally (re)builds the one-tile-row cache the scanline
    /// emulation reads through.
    ///
    /// # Errors
    /// Propagates the underlying reader's sampling-compatibility check, and
    /// returns [`ExrError::Arg`] when a tiled part is handed a subsampled slice
    /// (C++ "Tiled data should not have subsampling.").
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        match &mut self.inner {
            Reader::ScanLine(f) => f.set_frame_buffer(frame_buffer),
            Reader::Tiled(t) => t.set_frame_buffer(frame_buffer, self.complete),
            Reader::Deep(d) => d.set_frame_buffer(frame_buffer),
        }
    }

    /// Read scanlines `[min(s1,s2), max(s1,s2)]` into `out` — port of
    /// `InputFile::readPixels` → `Data::readPixels`
    /// (`ImfInputFile.cpp:446-460`).
    ///
    /// A scanline part goes straight to `ScanLineInputFile::readPixels`; a
    /// tiled part goes through `Data::bufferedReadPixels`, which reads each row
    /// of tiles that intersects the range and copies only the requested rows
    /// out (caching the previous tile row so a sequential scan reads each tile
    /// once).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if no frame buffer was set, or the range lies
    ///   outside the data window.
    /// * Propagates chunk / codec / scatter errors.
    pub fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()> {
        match &mut self.inner {
            Reader::ScanLine(f) => f.read_pixels(scan_line1, scan_line2, out),
            Reader::Tiled(t) => t.buffered_read_pixels(scan_line1, scan_line2, out),
            Reader::Deep(d) => d.read_pixels(scan_line1, scan_line2, out),
        }
    }

    /// [`read_pixels`](Self::read_pixels) into SEVERAL destination buffers (the planar read, an
    /// extension beyond OpenEXR): a slice added with `FrameBuffer::insert_in(name, slice, i)` is
    /// written to `planes[i]`, addressed relative to it. One decode of each chunk serves every plane.
    ///
    /// # Errors
    /// [`ExrError::Arg`] for a tiled or deep part (their readers keep the single-buffer path; the
    /// planar read of tiles is not implemented yet), plus everything [`read_pixels`](Self::read_pixels)
    /// returns.
    pub fn read_pixels_planes(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<()> {
        match &mut self.inner {
            Reader::ScanLine(f) => f.read_pixels_planes(scan_line1, scan_line2, planes),
            Reader::Tiled(_) | Reader::Deep(_) => Err(ExrError::Arg(
                "read_pixels_planes supports scanline parts only (tiled / deep: use read_pixels)".into(),
            )),
        }
    }

    /// Alloc audit C3 proof-of-reuse test hook: forwards to
    /// [`ScanLineInputFile::debug_fetch_chunk_capacity`] for a scanline part
    /// (the only kind the reuse-proof test needs). `pub(crate)` since this
    /// crosses from `scanline::input_file` (this module) into the sibling
    /// `scanline::input` module's private `ScanLineInputFileState::fetch_chunk`.
    #[cfg(test)]
    pub(crate) fn debug_fetch_chunk_capacity(
        &mut self,
        offset_index: usize,
        pool: &crate::buf::BufPool,
    ) -> Result<usize> {
        match &mut self.inner {
            Reader::ScanLine(f) => f.debug_fetch_chunk_capacity(offset_index, pool),
            _ => Err(ExrError::Logic(
                "debug_fetch_chunk_capacity: not a scanline-storage file".into(),
            )),
        }
    }

    /// Read the chunk covering `first_scan_line` RAW (still-packed, no
    /// decompression) — port of `InputFile::rawPixelData`
    /// (`ImfInputFile.cpp:239-243`, `_data->_sFile->rawPixelData(..)`).
    ///
    /// C++ dereferences `_sFile` unconditionally, which is only initialised
    /// for `EXR_STORAGE_SCANLINE` (UB for a tiled or deep part — the actual
    /// caller, `OutputFile::copyPixels(InputFile&)`, has already rejected a
    /// tiled source via its own header check by the time this would run, and
    /// C++ has no analogous scanline-`copyPixels` for a deep source at all).
    /// This port returns a clean error instead of UB, per the crate's
    /// no-panic rule.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if this part's storage is not flat scanline.
    pub fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        match &mut self.inner {
            Reader::ScanLine(f) => f.raw_pixel_data(first_scan_line),
            Reader::Tiled(_) | Reader::Deep(_) => Err(ExrError::Arg(
                "Cannot call rawPixelData() on a file that is not a flat scanline file.".into(),
            )),
        }
    }
}

/// The public surface a "read a header + scatter scanlines into a frame
/// buffer" caller needs — lets `rgba::input`'s bulk decoder
/// (`decode_from_input`/`decode_yca`/`decode_grey`/`decode_plain_rgba`)
/// accept EITHER the standalone [`InputFile`] (its own `RgbaInputFile::new`
/// path) or a multipart-cached view (`RgbaInputFile`'s `MultiPartInputFile`
/// path) generically, since both expose the identical three operations but
/// are genuinely different concrete types (one owns
/// its reader state, the other borrows a persisted one).
pub trait GenericInputReader {
    /// The part's header.
    fn header(&self) -> &Header;
    /// Set the pixel destination.
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()>;
    /// Read scanlines `[min(s1,s2), max(s1,s2)]` into `out`.
    fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()>;
    /// Read into several destination buffers (the planar read, see
    /// [`InputFile::read_pixels_planes`]).
    fn read_pixels_planes(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<()>;
}

impl<'a> GenericInputReader for InputFile<'a> {
    fn header(&self) -> &Header {
        InputFile::header(self)
    }
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        InputFile::set_frame_buffer(self, frame_buffer)
    }
    fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()> {
        InputFile::read_pixels(self, scan_line1, scan_line2, out)
    }
    fn read_pixels_planes(&mut self, scan_line1: i32, scan_line2: i32, planes: &mut [&mut [u8]]) -> Result<()> {
        InputFile::read_pixels_planes(self, scan_line1, scan_line2, planes)
    }
}

impl<'p> GenericInputReader for InputFilePartReader<'p> {
    fn header(&self) -> &Header {
        InputFilePartReader::header(self)
    }
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        InputFilePartReader::set_frame_buffer(self, frame_buffer)
    }
    fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()> {
        InputFilePartReader::read_pixels(self, scan_line1, scan_line2, out)
    }
    fn read_pixels_planes(&mut self, scan_line1: i32, scan_line2: i32, planes: &mut [&mut [u8]]) -> Result<()> {
        InputFilePartReader::read_pixels_planes(self, scan_line1, scan_line2, planes)
    }
}

// ---------------------------------------------------------------------------
// The tiled-as-scanlines emulation — `InputFile::Data::bufferedReadPixels`
// ---------------------------------------------------------------------------

impl<'a, S> TiledReader<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// C++ `Data::lockedSetFrameBuffer`, tiled branch
    /// (`ImfInputFile.cpp:337-432`): remember the caller's frame buffer and
    /// rebuild the single-tile-row cache it will be copied out of.
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer, complete: bool) -> Result<()> {
        // C++ `Data::lockedSetFrameBuffer` throws on an
        // incomplete/corrupt chunk table HERE -- whenever the tile-row cache
        // is (re)built, i.e. at `setFrameBuffer` time, BEFORE any read is
        // attempted (`ImfInputFile.cpp:386-394`). This guard used to live only
        // in `buffered_read_pixels`, and only on the PARTIAL-read path (the
        // whole-window fast path skipped it entirely) -- so `set_frame_buffer`
        // succeeded even on an unrecoverable table, and a full-window
        // `read_pixels` would then silently attempt the read instead of
        // erroring. We always rebuild the cache unconditionally (see below),
        // so checking every call here is the conservative, always-correct
        // match for "whenever the cache is rebuilt".
        if !complete {
            return Err(ExrError::Arg(
                "Unable to use generic API to read with (partially?) corrupt \
                 chunk table."
                    .into(),
            ));
        }

        // C++ only rebuilds the cache when the channel *set* or a channel type
        // changed; rebuilding unconditionally is the same pixels, and the only
        // state it discards is the tile row already in the cache.
        self.cache_bytes.clear();
        self.cache_fb = FrameBuffer::new();
        self.cached_tile_y = -1;

        let channels = self.file.header().channels()?;
        let dw = self.data_window;
        // C++ `_cachedOffset = dataWindow.min.x`.
        let cached_offset = dw.min.x as i64;
        let width = (dw.max.x as i64 - dw.min.x as i64 + 1).max(0);
        let tile_y = self.file.tile_y_size() as i64;
        let level_width = self.file.level_width(0)? as i64;

        for (name, slice) in frame_buffer.iter() {
            // C++ omits channels the file does not have; they are filled later.
            if channels.find_channel(name).is_none() {
                continue;
            }
            // C++: `bytes = (s.type == HALF) ? 2 : 4` — the CACHE holds the
            // caller's requested type, so the tiled reader converts once.
            // Design bug fix: see `TiledReaderState::set_frame_buffer`
            // above (this method's exact duplicate) for the rationale.
            let bytes = pixel_type_size(slice.type_)? as i64;
            let tile_bytes = bytes
                .checked_mul(tile_y)
                .and_then(|v| v.checked_mul(width))
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    ExrError::Arg("Tile row cache size overflows.".into())
                })?;
            let region = self.cache_bytes.len();
            self.cache_bytes.resize(region + tile_bytes, 0);

            self.cache_fb.insert(
                name,
                Slice::new(
                    slice.type_,
                    // C++: `base = block - bytes * _cachedOffset`.
                    region as i64 - bytes * cached_offset,
                    bytes as usize,
                    (bytes * level_width) as usize,
                    1,
                    1,
                    slice.fill_value,
                    false,
                    true,
                )?,
            )?;
        }

        self.file.set_frame_buffer(&self.cache_fb)?;
        self.user_fb = frame_buffer.clone();
        Ok(())
    }

    /// C++ `Data::bufferedReadPixels` (`ImfInputFile.cpp:485-563`).
    fn buffered_read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        out: &mut [u8],
    ) -> Result<()> {
        if self.user_fb.is_empty() {
            return Err(ExrError::Arg(
                "No frame buffer specified as pixel data destination.".into(),
            ));
        }

        let dw = self.data_window;
        let min_y = scan_line1.min(scan_line2);
        let max_y = scan_line1.max(scan_line2);

        if min_y < dw.min.y || max_y > dw.max.y {
            return Err(ExrError::Arg(
                "Tried to read scan line outside the image file's data window.".into(),
            ));
        }

        let tile_y_size = self.file.tile_y_size() as i32;
        if tile_y_size <= 0 {
            return Err(ExrError::Input("Invalid tile y size.".into()));
        }
        let min_dy = (min_y - dw.min.y) / tile_y_size;
        let max_dy = (max_y - dw.min.y) / tile_y_size;
        let last_x_tile = self.file.num_x_tiles(0)? - 1;

        // C++: reading the whole data window skips the double copy entirely and
        // scatters straight into the caller's buffer.
        if min_y == dw.min.y && max_y == dw.max.y {
            crate::probe::hit(crate::probe::Probe::TiledWholeWindow);
            self.file.set_frame_buffer(&self.user_fb)?;
            let r = self
                .file
                .read_tiles(0, last_x_tile, min_dy, max_dy, 0, 0, out);
            // Restore the cache frame buffer so a later partial read still
            // decodes into the cache (C++ re-sets it at the top of every call).
            self.file.set_frame_buffer(&self.cache_fb)?;
            self.cached_tile_y = -1;
            return r;
        }

        // The incomplete-chunk-table guard now lives in
        // `set_frame_buffer` (fires before ANY read is attempted, including
        // this whole-window fast path above, matching C++
        // `ImfInputFile.cpp:386-394`) -- `self.user_fb` is only ever non-empty
        // here because `set_frame_buffer` already succeeded, which itself
        // already required `complete`.
        self.file.set_frame_buffer(&self.cache_fb)?;

        for j in min_dy..=max_dy {
            if j != self.cached_tile_y {
                // "if no channels are being read that are present in file,
                // cachedBuffer will be empty" — then there is nothing to read.
                if !self.cache_fb.is_empty() {
                    let mut cache = std::mem::take(&mut self.cache_bytes);
                    let r = self
                        .file
                        .read_tiles(0, last_x_tile, j, j, 0, 0, &mut cache);
                    self.cache_bytes = cache;
                    r?;
                }
                self.cached_tile_y = j;
            }

            let tile_range = self.file.data_window_for_tile(0, j, 0, 0)?;
            let min_y_this_row = min_y.max(tile_range.min.y);
            let max_y_this_row = max_y.min(tile_range.max.y);
            let x_start = dw.min.x;
            let width = dw.max.x - dw.min.x + 1;

            for (name, to) in self.user_fb.iter() {
                match self.cache_fb.find_slice(name) {
                    Some(from) => copy_cached_buffer(
                        to,
                        from,
                        &self.cache_bytes,
                        min_y_this_row,
                        max_y_this_row,
                        tile_range.min.y,
                        x_start,
                        width,
                        out,
                    )?,
                    None => fill_buffer(
                        to,
                        min_y_this_row,
                        max_y_this_row,
                        dw.min.y,
                        x_start,
                        width,
                        out,
                    )?,
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InputFilePartReader — the multipart-cache read-side view, dispatching
// over BOTH storage kinds in one type
// ---------------------------------------------------------------------------

/// Persisted state for ONE cached part reader inside
/// [`crate::multipart::MultiPartInputFile`] — the read-side mirror,
/// covering every storage kind `InputFile`'s own [`Reader`] enum dispatches
/// on, including deep-SCANLINE (multipart mirror of [`Reader::Deep`], added
/// once `InputFile` itself gained deep-scanline flattening).
/// Deep-TILED stays a hard `Unsupported::Deep` error, matching
/// `InputFile::build`'s own `Storage::DeepTiled` arm (see its comment for why
/// that mirrors real, not merely deferred, C++ behaviour).
///
/// Wired into `MultiPartInputFile::input_part`.
pub(crate) enum PartReaderState {
    /// Boxed like the other persisted reader states; chunk caches can grow independently.
    ScanLine(Box<ScanLineInputFileState>),
    /// Boxed: `TiledReaderState` embeds a full `TiledInputFileState` plus the
    /// tile-row cache, comparable in size to the scanline variant's own
    /// `ScanLineInputFileState` -- boxed here (matching `Reader`'s own
    /// boxing) is cheap insurance against `large_enum_variant` as this
    /// crate's usual state objects grow; primarily added to keep the enum's
    /// own footprint from tracking whichever variant happens to be larger.
    Tiled(Box<TiledReaderState>),
    /// Boxed for the same `large_enum_variant` reason as `Tiled` above.
    Deep(Box<DeepScanLineReaderState>),
}

/// A borrowing wrapper pairing a stream with a persisted
/// `DeepScanLineInputFileState`, so it can be added as a
/// [`crate::deep::CompositeDeepScanLine`] source without owning either — the
/// multipart-cache counterpart of [`DeepReader`]'s owned
/// `DeepScanLineInputFile`. Exists only to satisfy the
/// [`crate::deep::DeepScanLineCompositeSource`] trait for the duration of one
/// `add_source` call; see [`DeepScanLineReaderState::set_frame_buffer`] /
/// [`DeepScanLineReaderState::read_pixels`].
struct DeepPartSourceView<'r> {
    stream: &'r mut dyn IStream,
    state: &'r mut crate::deep::scanline_input::DeepScanLineInputFileState,
}

impl crate::deep::DeepScanLineCompositeSource for DeepPartSourceView<'_> {
    fn header(&self) -> &Header {
        self.state.header()
    }
    fn set_frame_buffer(&mut self, frame_buffer: &crate::deep::DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }
    fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_counts(self.stream, scan_line1, scan_line2, sample_counts)
    }
    fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixels(self.stream, scan_line1, scan_line2, sample_counts, sample_data)
    }
}

/// The multipart-cache PERSISTABLE state for a deep-scanline part served
/// through the generic `InputFile`/`InputPart` front door — the cached-state mirror of
/// [`DeepReader`] (see that struct's doc comment for the full "why the
/// compositor is rebuilt per call, not stored" rationale, which applies
/// identically here). Holds the owned, stream-free
/// `DeepScanLineInputFileState` plus the caller's remembered flat
/// `FrameBuffer`; a fresh [`crate::deep::CompositeDeepScanLine`] is built
/// inside every [`set_frame_buffer`](Self::set_frame_buffer) /
/// [`read_pixels`](Self::read_pixels) call via a short-lived
/// [`DeepPartSourceView`] pairing this state with the container's
/// freshly-(re)borrowed stream.
pub(crate) struct DeepScanLineReaderState {
    deep: crate::deep::scanline_input::DeepScanLineInputFileState,
    user_fb: Option<FrameBuffer>,
}

impl DeepScanLineReaderState {
    /// Pure state construction — mirrors [`TiledReaderState::from_part_mt`]:
    /// no stream access at all (the prologue reader already parsed the
    /// header + chunk offset table before this ever runs). `num_threads` is
    /// the per-file override (`None` = capture the current
    /// process-wide default).
    pub(crate) fn from_part_mt(
        header: Header,
        version: i32,
        part_number: Option<i32>,
        line_offsets: Vec<u64>,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Ok(Self {
            deep: crate::deep::scanline_input::DeepScanLineInputFileState::from_part_mt(
                header,
                version,
                part_number,
                line_offsets,
                num_threads,
            )?,
            user_fb: None,
        })
    }

    pub(crate) fn header(&self) -> &Header {
        self.deep.header()
    }

    pub(crate) fn version(&self) -> i32 {
        self.deep.version()
    }

    /// Body identical to [`DeepReader::set_frame_buffer`] except the source
    /// is a [`DeepPartSourceView`] over the persisted state instead of an
    /// owned `DeepScanLineInputFile` — see [`DeepScanLineReaderState`]'s doc.
    pub(crate) fn set_frame_buffer(
        &mut self,
        stream: &mut dyn IStream,
        frame_buffer: &FrameBuffer,
    ) -> Result<()> {
        let mut view = DeepPartSourceView { stream, state: &mut self.deep };
        let mut compositor = crate::deep::CompositeDeepScanLine::new();
        compositor.add_source(&mut view)?;
        compositor.set_frame_buffer(frame_buffer)?;
        self.user_fb = Some(frame_buffer.clone());
        Ok(())
    }

    /// Body identical to [`DeepReader::read_pixels`] except the source is a
    /// [`DeepPartSourceView`] over the persisted state — see
    /// [`DeepScanLineReaderState`]'s doc.
    pub(crate) fn read_pixels(
        &mut self,
        stream: &mut dyn IStream,
        scan_line1: i32,
        scan_line2: i32,
        out: &mut [u8],
    ) -> Result<()> {
        let frame_buffer = self.user_fb.clone().ok_or_else(|| {
            ExrError::Arg("No frame buffer was set.".into())
        })?;
        let mut view = DeepPartSourceView { stream, state: &mut self.deep };
        let mut compositor = crate::deep::CompositeDeepScanLine::new();
        compositor.add_source(&mut view)?;
        compositor.set_frame_buffer(&frame_buffer)?;
        compositor.read_pixels(scan_line1, scan_line2, out)
    }
}

/// A **borrowing** view over one multipart part's persisted reader state —
/// the read-side mirror of `multipart::output`'s writer-side caching,
/// intended to also cover the tiled-as-scanlines path. Would be built by
/// [`crate::multipart::MultiPartInputFile::input_part`], which would persist
/// a [`PartReaderState`] per part (built once, on first access, exactly like
/// the write side's `scanline_state`/`tiled_state`) and hand out one of
/// these instead of rebuilding (and for a tiled part, silently discarding
/// the tile-row cache of) a fresh reader every call.
///
/// Exposes the same public surface a standalone [`InputFile`] does
/// (`header`, `version`, `is_complete`, `set_frame_buffer`, `read_pixels`),
/// which is what `rgba::input`'s [`GenericInputReader`] trait bound is built
/// around, and is now returned directly by
/// `MultiPartInputFile::input_part`.
pub struct InputFilePartReader<'p> {
    stream: &'p mut dyn IStream,
    inner: &'p mut PartReaderState,
    /// C++ `InputFile::isComplete()` — same field `InputFile` itself carries,
    /// computed once by `read_prologue` and handed down; per-part, not
    /// per-call, so it is copied in here rather than looked up through
    /// `inner` (a scanline part's own state has no notion of completeness at
    /// all — only the tiled path's `set_frame_buffer` needs it).
    complete: bool,
}

impl<'p> InputFilePartReader<'p> {
    /// Build a view over an existing (already-constructed, possibly
    /// previously-read-from) part state and the container's stream.
    pub(crate) fn new(
        stream: &'p mut dyn IStream,
        inner: &'p mut PartReaderState,
        complete: bool,
    ) -> Self {
        Self { stream, inner, complete }
    }

    /// The part's header — port of `const Header& InputFile::header() const`.
    pub fn header(&self) -> &Header {
        // `&*self.inner` reborrows the stored `&'p mut PartReaderState`
        // immutably -- `self.inner` alone (used by value) would try to MOVE
        // a `&mut` out of `&self`, which a shared method can never do.
        match &*self.inner {
            PartReaderState::ScanLine(s) => s.header(),
            PartReaderState::Tiled(t) => t.header(),
            PartReaderState::Deep(d) => d.header(),
        }
    }

    /// The raw file-format version word — port of `int InputFile::version()`.
    pub fn version(&self) -> i32 {
        match &*self.inner {
            PartReaderState::ScanLine(s) => s.version(),
            PartReaderState::Tiled(t) => t.version(),
            PartReaderState::Deep(d) => d.version(),
        }
    }

    /// Is every chunk of this part present and addressable? — port of
    /// `bool InputFile::isComplete() const`.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Set the pixel destination — port of `InputFile::setFrameBuffer`.
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        match self.inner {
            PartReaderState::ScanLine(s) => s.set_frame_buffer(frame_buffer),
            PartReaderState::Tiled(t) => t.set_frame_buffer(frame_buffer, self.complete),
            PartReaderState::Deep(d) => d.set_frame_buffer(self.stream, frame_buffer),
        }
    }

    /// Read scanlines `[min(s1,s2), max(s1,s2)]` into `out` — port of
    /// `InputFile::readPixels`.
    pub fn read_pixels(&mut self, scan_line1: i32, scan_line2: i32, out: &mut [u8]) -> Result<()> {
        match self.inner {
            PartReaderState::ScanLine(s) => s.read_pixels(self.stream, scan_line1, scan_line2, out),
            PartReaderState::Tiled(t) => t.read_pixels(self.stream, scan_line1, scan_line2, out),
            PartReaderState::Deep(d) => d.read_pixels(self.stream, scan_line1, scan_line2, out),
        }
    }

    /// [`read_pixels`](Self::read_pixels) into several destination buffers (planar read), for a
    /// scanline part; tiled / deep parts return [`ExrError::Arg`].
    pub fn read_pixels_planes(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        planes: &mut [&mut [u8]],
    ) -> Result<()> {
        match self.inner {
            PartReaderState::ScanLine(s) => {
                s.read_pixels_planes(self.stream, scan_line1, scan_line2, planes)
            }
            PartReaderState::Tiled(_) | PartReaderState::Deep(_) => Err(ExrError::Arg(
                "read_pixels_planes supports scanline parts only (tiled / deep: use read_pixels)".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // `TiledCopySource` support — port of `InputFile::asTiledInput` for the
    // cached multipart view: `multipart::part::InputPart`'s `TiledCopySource`
    // impl now wraps this type (instead of an owned `InputFile`) and forwards
    // through these four methods, exactly the same shape the trait itself
    // uses (`tiled/output.rs`).
    // -----------------------------------------------------------------------

    /// Confirm this part is tiled-storage — port of `InputFile::asTiledInput`
    /// returning `Result<()>` instead of a checked `&mut TiledInputFile`
    /// (a borrowing cache view never owns one to hand back).
    pub fn ensure_tiled(&mut self) -> Result<()> {
        match self.inner {
            PartReaderState::Tiled(_) => Ok(()),
            PartReaderState::ScanLine(_) | PartReaderState::Deep(_) => {
                Err(ExrError::Arg("This part's storage is not tiled.".into()))
            }
        }
    }

    /// This part's on-disk tile order — see [`TiledReaderState::tile_order`].
    ///
    /// # Errors
    /// [`ExrError::Arg`] if this part's storage is not tiled.
    pub fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder> {
        match self.inner {
            PartReaderState::Tiled(t) => t.tile_order(),
            PartReaderState::ScanLine(_) | PartReaderState::Deep(_) => {
                Err(ExrError::Arg("This part's storage is not tiled.".into()))
            }
        }
    }

    /// Read tile `(dx, dy, lx, ly)`'s raw on-disk bytes — see
    /// [`TiledReaderState::raw_tile_data`].
    ///
    /// # Errors
    /// [`ExrError::Arg`] if this part's storage is not tiled.
    pub fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        match self.inner {
            PartReaderState::Tiled(t) => t.raw_tile_data(self.stream, dx, dy, lx, ly),
            PartReaderState::ScanLine(_) | PartReaderState::Deep(_) => {
                Err(ExrError::Arg("This part's storage is not tiled.".into()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // `ScanLineCopySource` support — `multipart::part::InputPart`'s
    // `ScanLineCopySource` impl wraps this type and forwards through this
    // method, exactly the shape the trait itself uses (`scanline/output.rs`).
    // -----------------------------------------------------------------------

    /// The chunk covering `first_scan_line`, raw — see
    /// [`super::input::ScanLineInputFileState::raw_pixel_data`] for the full
    /// doc.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if this part's storage is not a flat scanline part.
    pub fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        match self.inner {
            PartReaderState::ScanLine(s) => s.raw_pixel_data(self.stream, first_scan_line),
            PartReaderState::Tiled(_) | PartReaderState::Deep(_) => Err(ExrError::Arg(
                "Cannot call rawPixelData() on a file that is not a flat scanline file.".into(),
            )),
        }
    }
}

/// C++ `InputFile::Data::copyCachedBuffer` (`ImfInputFile.cpp:568-633`):
/// copy rows `[scanline1, scanline2]` of one channel out of the tile-row cache
/// into the caller's buffer.
#[allow(clippy::too_many_arguments)]
fn copy_cached_buffer(
    to: &Slice,
    from: &Slice,
    cache: &[u8],
    scanline1: i32,
    scanline2: i32,
    y_start: i32,
    x_start: i32,
    width: i32,
    out: &mut [u8],
) -> Result<()> {
    if to.x_sampling != 1 || to.y_sampling != 1 || from.x_sampling != 1 || from.y_sampling != 1 {
        return Err(ExrError::Arg(
            "Tiled data should not have subsampling.".into(),
        ));
    }
    if to.type_ != from.type_ {
        return Err(ExrError::Arg(
            "Invalid type mismatch in slice from setFrameBuffer.".into(),
        ));
    }
    let size = pixel_type_size(from.type_)?;
    if from.x_stride != size {
        return Err(ExrError::Arg(
            "Unhandled type in copying tile cache slice.".into(),
        ));
    }

    for y in scanline1..=scanline2 {
        let mut to_idx = to.base_offset
            + if to.y_tile_coords {
                (y as i64 - y_start as i64) * to.y_stride as i64
            } else {
                y as i64 * to.y_stride as i64
            };
        if !to.x_tile_coords {
            to_idx += x_start as i64 * to.x_stride as i64;
        }
        let mut from_idx = from.base_offset
            + (y as i64 - y_start as i64) * from.y_stride as i64
            + x_start as i64 * size as i64;

        for _ in 0..width {
            let src = usize::try_from(from_idx)
                .ok()
                .filter(|&i| i + size <= cache.len())
                .ok_or_else(|| ExrError::Arg("Tile cache read out of range.".into()))?;
            let dst = usize::try_from(to_idx)
                .ok()
                .filter(|&i| i + size <= out.len())
                .ok_or_else(|| {
                    ExrError::Arg("Frame buffer slice address is out of range.".into())
                })?;
            out[dst..dst + size].copy_from_slice(&cache[src..src + size]);
            to_idx += to.x_stride as i64;
            from_idx += size as i64;
        }
    }
    Ok(())
}

/// C++ `InputFile::Data::fillBuffer` (`ImfInputFile.cpp:638-706`): a channel the
/// file does not have is filled with its `fillValue` over the requested rows.
fn fill_buffer(
    to: &Slice,
    scanline1: i32,
    scanline2: i32,
    y_start: i32,
    x_start: i32,
    width: i32,
    out: &mut [u8],
) -> Result<()> {
    if to.x_sampling != 1 || to.y_sampling != 1 {
        return Err(ExrError::Arg(
            "Tiled data should not have subsampling.".into(),
        ));
    }

    // The native-endian fill pattern for one pixel (the low-level FrameBuffer
    // is native-endian; see the crate's `misc.rs` wire boundary).
    let pattern: Vec<u8> = match to.type_ {
        PixelType::Uint => (to.fill_value as u32).to_ne_bytes().to_vec(),
        PixelType::Half => imath_rs::Half::from(to.fill_value as f32)
            .bits()
            .to_ne_bytes()
            .to_vec(),
        PixelType::Float => (to.fill_value as f32).to_ne_bytes().to_vec(),
        // C++ `fillBuffer`'s `case NUM_PIXELTYPES: break;` — nothing to write.
        PixelType::NumPixelTypes => return Ok(()),
    };
    let size = pattern.len();

    for y in scanline1..=scanline2 {
        let mut to_idx = to.base_offset
            + if to.y_tile_coords {
                (y as i64 - y_start as i64) * to.y_stride as i64
            } else {
                y as i64 * to.y_stride as i64
            };
        if !to.x_tile_coords {
            to_idx += x_start as i64 * to.x_stride as i64;
        }
        for _ in 0..width {
            let dst = usize::try_from(to_idx)
                .ok()
                .filter(|&i| i + size <= out.len())
                .ok_or_else(|| {
                    ExrError::Arg("Frame buffer fill address is out of range.".into())
                })?;
            out[dst..dst + size].copy_from_slice(&pattern);
            to_idx += to.x_stride as i64;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — the `InputFile::initialize` dispatch
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::offsets::testing::{drop_attribute, set_string_attribute, zero_offset_tables};
    use crate::attr::{
        Channel, ChannelList, ChannelListAttribute, Compression, LevelMode, LineOrder,
        TileDescription, TileDescriptionAttribute,
    };
    use crate::header::{SCANLINE_IMAGE, TILED_IMAGE};
    use crate::io::{MemIStream, MemOStream};
    use crate::multipart::{MultiPartInputFile, MultiPartOutputFile};
    use crate::tiled::TiledOutputFile;
    use imath_rs::{Half, V2f};

    const W: i32 = 8;
    const H: i32 = 6;

    /// One HALF channel "R", `W x H` at the origin.
    fn base_header(comp: Compression) -> Header {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            comp,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Pixel (x, y) = half(y * W + x), plus the frame buffer addressing it.
    fn make_plane() -> (Vec<u8>, FrameBuffer) {
        let row = W as usize * 2;
        let mut buf = vec![0u8; row * H as usize];
        for y in 0..H as usize {
            for x in 0..W as usize {
                let v = Half::from((y * W as usize + x) as f32).bits();
                buf[y * row + x * 2..y * row + x * 2 + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap(),
        )
        .unwrap();
        (buf, fb)
    }

    fn expected_plane() -> Vec<u8> {
        make_plane().0
    }

    /// Same shape as [`base_header`]/[`make_plane`] but with a data window
    /// whose origin is NOT `(0, 0)` (`min = (7, 4)`, still `W x H`) --
    /// mutation-testing gap fix: every OTHER header/plane builder in this
    /// module places the data window at the origin, which hides any mutant
    /// in code shaped like `x - dw.min.x` or `y - dw.min.y` (a `min.x ==
    /// 0`/`min.y == 0` file can't distinguish `-` from `+` there). Built via
    /// `Header::with_windows` + `Slice::make` (which computes the frame
    /// buffer's base offset from the window's own origin, same convention
    /// `chunk/mod.rs`'s `write_scanline_shifted_origin` uses).
    fn base_header_shifted(comp: Compression) -> Header {
        let dw = imath_rs::Box2i {
            min: imath_rs::V2i { x: 7, y: 4 },
            max: imath_rs::V2i { x: 7 + W - 1, y: 4 + H - 1 },
        };
        let mut h =
            Header::with_windows(dw, dw, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp)
                .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Shifted-origin counterpart of [`make_plane`]: pixel (x, y) (in image
    /// coordinates, `x` in `7..7+W`, `y` in `4..4+H`) = `half((y - 4) * W +
    /// (x - 7))`, same value pattern as [`make_plane`] but addressed through
    /// `Slice::make`'s `origin` instead of assuming the buffer starts at
    /// `(0, 0)`.
    fn make_plane_shifted() -> (Vec<u8>, FrameBuffer) {
        let row = W as usize * 2;
        let mut buf = vec![0u8; row * H as usize];
        for y in 0..H as usize {
            for x in 0..W as usize {
                let v = Half::from((y * W as usize + x) as f32).bits();
                buf[y * row + x * 2..y * row + x * 2 + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice::make(
                PixelType::Half,
                0,
                imath_rs::V2i { x: 7, y: 4 },
                W as i64,
                2,
                row,
                1,
                1,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();
        (buf, fb)
    }

    /// Mutation-testing gap fix (scanline/ audit, prep for pass 1): a
    /// non-origin data window's write+read round trip must reproduce the
    /// exact pixel values -- exercises every `dw.min.x`/`dw.min.y`-relative
    /// address computation in both `OutputFile::write_pixels` and
    /// `InputFile::read_pixels` at once, with values that would visibly
    /// scramble under a `-`->`+` (or similar) mutant on either side.
    #[test]
    fn shifted_data_window_round_trips_byte_exact() {
        let h = base_header_shifted(Compression::NoCompression);
        let (pixels, fb) = make_plane_shifted();
        let mut out = MemOStream::new();
        {
            let mut of = crate::scanline::OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, H).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        let (_, fb_in) = make_plane_shifted();
        inf.set_frame_buffer(&fb_in).unwrap();
        let mut result = vec![0u8; expected_plane().len()];
        inf.read_pixels(4, 4 + H - 1, &mut result).unwrap();
        assert_eq!(result, make_plane_shifted().0, "pixel values must round-trip exactly");
    }

    /// Mutation-testing gap fix (scanline/ audit prep, static blind-spot
    /// review): every OTHER ZIP-compressed test in this module uses a
    /// height small enough to fit inside ZIP's 16-line block, so the LAST
    /// chunk is always the same size as every other chunk. `PARTIAL_H = 20`
    /// forces a genuinely partial final block (16 lines, then 4) -- the code
    /// path that computes `block_lines`/`lines_in_buffer` bounds for a
    /// short last chunk, both on write (`OutputFile::write_pixels`) and read
    /// (`InputFile::read_pixels`).
    #[test]
    fn zip_partial_last_block_round_trips_byte_exact() {
        const PARTIAL_W: i32 = 4;
        const PARTIAL_H: i32 = 20; // 16 + 4: a genuinely partial last ZIP block.
        let mut h = Header::with_dimensions(
            PARTIAL_W,
            PARTIAL_H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let row = PARTIAL_W as usize * 2;
        let mut pixels = vec![0u8; row * PARTIAL_H as usize];
        for y in 0..PARTIAL_H as usize {
            for x in 0..PARTIAL_W as usize {
                let v = Half::from((y * PARTIAL_W as usize + x) as f32).bits();
                pixels[y * row + x * 2..y * row + x * 2 + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert("R", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
            .unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = crate::scanline::OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, PARTIAL_H).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        let mut fb_in = FrameBuffer::new();
        fb_in
            .insert("R", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
            .unwrap();
        inf.set_frame_buffer(&fb_in).unwrap();
        let mut result = vec![0u8; pixels.len()];
        inf.read_pixels(0, PARTIAL_H - 1, &mut result).unwrap();
        assert_eq!(result, pixels, "pixel values must round-trip exactly across the partial last block");
    }

    /// Mutation-testing gap fix (scanline/ audit, static blind-spot review):
    /// a y-subsampled channel's write+read round trip, combined with a
    /// non-zero data-window origin AND several multi-line ZIP chunks
    /// (including a partial last one) -- the reference technique is
    /// `parity_exr.rs`'s `sub_matrix_*` tests: a dump-layout
    /// frame buffer (each channel's own tightly-packed reduced-grid plane,
    /// channels in alphabetical order, no interleaving), addressed via
    /// `Slice::make` so `x_sampling`/`y_sampling`'s `divp`/`modp` address
    /// math (`scanline::input`'s per-line sampling check) is exercised at a
    /// non-zero origin, which none of this module's other tests do.
    ///
    /// Channel "A" is unsampled (1, 1); channel "S" is subsampled 2x2 --
    /// matching one of `SUB_MATRIX_SAMPLINGS`'s tested cells. Run twice
    /// (`NoCompression`, `Zip`) over an 8x20 image (origin (7, 4)): ZIP's
    /// 16-line block size makes this 2 chunks, the second only 4 lines --
    /// the same partial-last-block shape as the test above, but now with
    /// subsampling layered on top.
    fn sub_val(name: &str, x: i32, y: i32, w: i32) -> f32 {
        match name {
            "A" => 1.0 + (y * w + x) as f32 * 0.1,
            "S" => -30.0 + (y * w + x) as f32 * 0.2,
            _ => unreachable!("sub_val: unexpected channel {name}"),
        }
    }

    fn sub_scanline_round_trip(comp: Compression) {
        const ORIGIN: (i32, i32) = (7, 4);
        const SW: i32 = 8;
        const SH: i32 = 20; // 16 + 4: a partial last ZIP block, same as above.
        const XS: i32 = 1;
        const YS: i32 = 2;

        let dw = imath_rs::Box2i {
            min: imath_rs::V2i { x: ORIGIN.0, y: ORIGIN.1 },
            max: imath_rs::V2i { x: ORIGIN.0 + SW - 1, y: ORIGIN.1 + SH - 1 },
        };
        let mut hdr = Header::with_windows(
            dw,
            dw,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            comp,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("S", Channel::new(PixelType::Half, XS, YS, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl.clone())).unwrap();

        // Dump-layout buffer: "A"'s full SW x SH plane, then "S"'s reduced
        // (SW/XS) x (SH/YS) plane, each tightly packed, alphabetical order
        // (matches `ChannelList::iter`'s own ordering).
        let a_w = SW as usize;
        let a_h = SH as usize;
        let s_w = (SW / XS) as usize;
        let s_h = (SH / YS) as usize;
        let a_bytes = a_w * a_h * 2;
        let s_bytes = s_w * s_h * 2;
        let mut pixels = vec![0u8; a_bytes + s_bytes];
        for y in 0..a_h {
            for x in 0..a_w {
                let bits = Half::from(sub_val("A", x as i32, y as i32, SW)).bits();
                let idx = (y * a_w + x) * 2;
                pixels[idx..idx + 2].copy_from_slice(&bits.to_ne_bytes());
            }
        }
        for y in 0..s_h {
            for x in 0..s_w {
                let bits = Half::from(sub_val("S", x as i32, y as i32, SW)).bits();
                let idx = a_bytes + (y * s_w + x) * 2;
                pixels[idx..idx + 2].copy_from_slice(&bits.to_ne_bytes());
            }
        }

        let mut fb = FrameBuffer::new();
        fb.insert(
            "A",
            Slice::make(
                PixelType::Half,
                0,
                imath_rs::V2i { x: ORIGIN.0, y: ORIGIN.1 },
                SW as i64,
                2,
                a_w * 2,
                1,
                1,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();
        fb.insert(
            "S",
            Slice::make(
                PixelType::Half,
                a_bytes,
                imath_rs::V2i { x: ORIGIN.0, y: ORIGIN.1 },
                SW as i64,
                2,
                s_w * 2,
                XS,
                YS,
                0.0,
                false,
                false,
            )
            .unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = crate::scanline::OutputFile::new(&mut out, &hdr).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, SH).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut result = vec![0u8; pixels.len()];
        inf.read_pixels(ORIGIN.1, ORIGIN.1 + SH - 1, &mut result).unwrap();
        assert_eq!(
            result, pixels,
            "y-subsampled channel round trip must be byte-exact (compression={comp:?})"
        );
    }

    #[test]
    fn subsampled_channel_round_trips_byte_exact_none() {
        sub_scanline_round_trip(Compression::NoCompression);
    }

    #[test]
    fn subsampled_channel_round_trips_byte_exact_zip() {
        sub_scanline_round_trip(Compression::Zip);
    }

    /// A single-part TILED file whose tiles are smaller than the image, so the
    /// scanline emulation has to stitch several tile rows together.
    fn write_tiled(comp: Compression) -> Vec<u8> {
        let mut h = base_header(comp);
        h.set_type(TILED_IMAGE).unwrap();
        h.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 4,
                y_size: 2,
                mode: LevelMode::OneLevel,
                ..TileDescription::default()
            }),
        )
        .unwrap();
        let (pixels, fb) = make_plane();

        let mut out = MemOStream::new();
        {
            let mut tof = TiledOutputFile::new(&mut out, &h).unwrap();
            tof.set_frame_buffer(&fb).unwrap();
            let nx = tof.num_x_tiles(0).unwrap();
            let ny = tof.num_y_tiles(0).unwrap();
            tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        out.into_data()
    }

    /// A 2-part multipart file, both scanline parts carrying the same plane.
    fn write_multipart() -> Vec<u8> {
        let mut h0 = base_header(Compression::Zip);
        h0.set_type(SCANLINE_IMAGE).unwrap();
        h0.set_name("first").unwrap();
        let mut h1 = base_header(Compression::Zip);
        h1.set_type(SCANLINE_IMAGE).unwrap();
        h1.set_name("second").unwrap();
        let (pixels, fb) = make_plane();

        let mut out = MemOStream::new();
        {
            let mut mpo = MultiPartOutputFile::new(&mut out, &[h0, h1]).unwrap();
            for p in 0..2 {
                let mut op = mpo.output_part(p).unwrap();
                op.set_frame_buffer(&fb).unwrap();
                op.write_pixels(&pixels, H).unwrap();
            }
        }
        out.into_data()
    }

    /// A TILED file opens through `InputFile` and is read as scanlines.
    #[test]
    fn input_file_reads_tiled_as_scanlines() {
        let bytes = write_tiled(Compression::Zip);
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert!(inf.is_complete());
        inf.set_frame_buffer(&fb).unwrap();

        // Whole data window — the C++ "skip the double copy" fast path (it
        // must actually be taken; a partial range must not take it).
        let mut out = vec![0u8; expected_plane().len()];
        crate::probe::reset(crate::probe::Probe::TiledWholeWindow);
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(crate::probe::get(crate::probe::Probe::TiledWholeWindow), 1, "whole-window read skipped the fast path");
        assert_eq!(out, expected_plane());
        inf.read_pixels(0, 0, &mut out).unwrap();
        assert_eq!(crate::probe::get(crate::probe::Probe::TiledWholeWindow), 1, "a one-row read must use the tile-row cache");
    }

    /// A partial scanline range through the tile-row cache must touch
    /// ONLY the requested rows (C++ `bufferedReadPixels` + `copyCachedBuffer`).
    #[test]
    fn input_file_tiled_partial_range_touches_only_requested_rows() {
        let bytes = write_tiled(Compression::Zip);
        let (_, fb) = make_plane();
        let expected = expected_plane();
        let row = W as usize * 2;

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        // Rows 1..=4 span three tile rows (tile height 2); rows 0 and 5 must
        // keep the sentinel.
        let mut out = vec![0xABu8; expected.len()];
        inf.read_pixels(1, 4, &mut out).unwrap();
        assert_eq!(&out[..row], &vec![0xABu8; row][..], "row 0 was clobbered");
        assert_eq!(&out[5 * row..], &vec![0xABu8; row][..], "row 5 was clobbered");
        assert_eq!(&out[row..5 * row], &expected[row..5 * row]);
    }

    /// A tiled `InputFile` fills a frame-buffer channel the file lacks
    /// (C++ `Data::fillBuffer`), and only over the requested rows.
    #[test]
    fn input_file_tiled_fills_absent_channel() {
        let bytes = write_tiled(Compression::Zip);
        let row = W as usize * 2;
        let mut fb = FrameBuffer::new();
        fb.insert(
            "Z",
            Slice::new(PixelType::Half, 0, 2, row, 1, 1, 7.0, false, false).unwrap(),
        )
        .unwrap();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();

        let mut out = vec![0u8; row * H as usize];
        inf.read_pixels(2, 3, &mut out).unwrap();
        let seven = Half::from(7.0f32).bits().to_ne_bytes();
        for y in 0..H as usize {
            for x in 0..W as usize {
                let got = &out[y * row + x * 2..y * row + x * 2 + 2];
                if (2..=3).contains(&y) {
                    assert_eq!(got, seven, "({x},{y}) should be filled");
                } else {
                    assert_eq!(got, [0, 0], "({x},{y}) outside the range was written");
                }
            }
        }
    }

    /// Part 0 of a MULTIPART file opens through `InputFile`, chunk
    /// `partNumber` prefixes and all (C++ `InputFile` takes `kDefaultPart`).
    #[test]
    fn input_file_reads_part_zero_of_multipart() {
        let bytes = write_multipart();
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().name().unwrap(), "first");
        let mut out = vec![0u8; expected_plane().len()];
        inf.set_frame_buffer(&fb).unwrap();
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// `MultiPartInputFile::input_part` is the same dispatcher, so
    /// part 1 reads too — and `partComplete` is true for a fully written file.
    #[test]
    fn multipart_input_part_dispatches_and_part_complete() {
        let bytes = write_multipart();
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
        assert_eq!(mpi.parts(), 2);
        assert!(mpi.part_complete(0).unwrap());
        assert!(mpi.part_complete(1).unwrap());
        assert!(mpi.part_complete(2).is_err(), "out-of-range part must error");

        let mut out = vec![0u8; expected_plane().len()];
        let mut part = mpi.input_part(1).unwrap();
        assert_eq!(part.header().name().unwrap(), "second");
        part.set_frame_buffer(&fb).unwrap();
        part.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// S13b: `MultiPartInputFile` accepts a SINGLE-part file (C++ opens it as a
    /// one-part file; we used to reject it).
    #[test]
    fn multipart_input_file_accepts_single_part_file() {
        let bytes = write_tiled(Compression::Zip);
        let mut is = MemIStream::new(bytes);
        let mpi = MultiPartInputFile::new(&mut is).unwrap();
        assert_eq!(mpi.parts(), 1);
        assert_eq!(mpi.part_type(0).unwrap(), TILED_IMAGE);
        assert!(mpi.part_complete(0).unwrap());
    }

    /// `MultiPartInputFile::tiled_input_part(0)` on
    /// a SINGLE-part tiled file (no `partNumber` chunk prefix on disk) must
    /// actually READ every tile correctly, not just open — the tiled sibling
    /// of `multipart_input_part_reads_a_single_part_scanline_file` above.
    #[test]
    fn multipart_tiled_input_part_reads_a_single_part_tiled_file() {
        let bytes = write_tiled(Compression::Zip);
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
        assert_eq!(mpi.parts(), 1);

        let mut tif = mpi.tiled_input_part(0).unwrap();
        tif.set_frame_buffer(&fb).unwrap();
        let mut out = vec![0u8; expected_plane().len()];
        let nx = tif.num_x_tiles(0).unwrap();
        let ny = tif.num_y_tiles(0).unwrap();
        tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// `MultiPartInputFile::input_part(0)` on a SINGLE-part scanline
    /// file (no `partNumber` chunk prefix on disk — the version word carries
    /// no `MULTI_PART_FILE_FLAG`) must actually READ the pixels correctly,
    /// not just open. C++ only reads/verifies the prefix when
    /// `ctxt->is_multipart` (`OpenEXRCore/chunk.c:955-983`,
    /// `rdcnt = is_multipart ? 2 : 1`); before this fix `input_part` always
    /// passed `Some(n)`, so the reader mis-parsed the leading scanline `y` as
    /// a `partNumber` and the real `y` as garbage.
    #[test]
    fn multipart_input_part_reads_a_single_part_scanline_file() {
        let bytes = write_scanline();
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
        assert_eq!(mpi.parts(), 1);

        let mut out = vec![0u8; expected_plane().len()];
        let mut part = mpi.input_part(0).unwrap();
        part.set_frame_buffer(&fb).unwrap();
        part.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// Multipart counterpart: `MultiPartInputFile::input_part` on a
    /// deep-SCANLINE part must flatten it through `CompositeDeepScanLine` the
    /// same way the standalone `InputFile` does (`PartReaderState::Deep`),
    /// with the leading `partNumber` genuinely verified (part 1 of a real
    /// 2-part file, not a single-part shortcut). Compares against the SAME
    /// file read directly via `DeepScanLineInputFile` + a fresh
    /// `CompositeDeepScanLine` (the already oracle-verified path) for
    /// byte-identical output.
    #[test]
    fn multipart_input_part_flattens_a_deep_scanline_part() {
        use crate::attr::{Channel, ChannelList, ChannelListAttribute};
        use crate::header::DEEP_SCANLINE;
        use imath_rs::V2f;

        let w = 2i32;
        let channels = ["Z", "A", "R"];
        let counts = [2u32, 1u32];
        let z: [&[f32]; 2] = [&[0.0, 1.0], &[0.5]];
        let a: [&[f32]; 2] = [&[0.5, 0.5], &[0.25]];
        let r: [&[f32]; 2] = [&[1.0, 2.0], &[4.0]];
        let per_channel = [&z, &a, &r];

        let mut sc = Vec::with_capacity(w as usize * 4);
        for &c in &counts {
            sc.extend_from_slice(&c.to_le_bytes());
        }
        let total: u32 = counts.iter().sum();
        let mut data = vec![0u8; channels.len() * total as usize * 4];
        for (ch_idx, per_x) in per_channel.iter().enumerate() {
            let mut offset = 0usize;
            for (x, samples) in per_x.iter().enumerate() {
                assert_eq!(samples.len(), counts[x] as usize);
                for &v in *samples {
                    let byte_off = (ch_idx * total as usize + offset) * 4;
                    data[byte_off..byte_off + 4].copy_from_slice(&v.to_le_bytes());
                    offset += 1;
                }
            }
        }

        let mut flat_header = base_header(Compression::NoCompression);
        flat_header.set_type(SCANLINE_IMAGE).unwrap();
        flat_header.set_name("flat").unwrap();
        let (flat_pixels, flat_fb) = make_plane();

        // The deep part's own dataWindow is small (2x1 -- unrelated to the
        // flat part's), but MultiPartOutputFile requires the SHARED
        // attributes (displayWindow among them) to agree across every
        // part, so it borrows the flat part's displayWindow rather than
        // building its own via `Header::with_dimensions` (which would set
        // displayWindow == dataWindow and conflict).
        let display_window = flat_header.display_window().unwrap();
        let deep_data_window = imath_rs::Box2i {
            min: imath_rs::V2i { x: 0, y: 0 },
            max: imath_rs::V2i { x: w - 1, y: 0 },
        };
        let mut deep_header = Header::with_windows(
            display_window,
            deep_data_window,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        deep_header.set_name("deep").unwrap();
        deep_header.set_type(DEEP_SCANLINE).unwrap();
        let mut cl = ChannelList::new();
        for name in &channels {
            cl.insert(*name, Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        }
        deep_header.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let mut deep_fb = crate::deep::DeepFrameBuffer::new();
        deep_fb
            .insert_sample_count_slice(Slice {
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
        for (ch_idx, name) in channels.iter().enumerate() {
            deep_fb
                .insert(
                    *name,
                    crate::deep::frame_buffer::DeepSlice::new(
                        PixelType::Float,
                        ch_idx as i64 * total as i64 * 4,
                        8,
                        8 * w as usize,
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
        }

        let mut out = MemOStream::new();
        {
            // Part 0 flat (forces the multipart flag + a real partNumber
            // prefix on every chunk, part 1 the deep-scanline part under test.
            let mut mpo =
                MultiPartOutputFile::new(&mut out, &[flat_header, deep_header]).unwrap();
            let mut fp = mpo.output_part(0).unwrap();
            fp.set_frame_buffer(&flat_fb).unwrap();
            fp.write_pixels(&flat_pixels, H).unwrap();
            let mut dp = mpo.deep_scanline_output_part(1).unwrap();
            dp.set_frame_buffer(&deep_fb).unwrap();
            dp.write_pixels(&sc, &data, 1).unwrap();
        }
        let bytes = out.into_data();

        // Reference: read the deep part directly (with its own standalone
        // reader) via `deep_scanline_input_part` + a fresh
        // `CompositeDeepScanLine` (the already oracle-verified path).
        let (mut expected_out, expected_fb) = {
            let out = vec![0u8; w as usize * 3 * 4];
            let mut fb = FrameBuffer::new();
            for (i, name) in ["Z", "A", "R"].iter().enumerate() {
                fb.insert(
                    *name,
                    Slice {
                        type_: PixelType::Float,
                        base_offset: i as i64 * 4,
                        x_stride: 12,
                        y_stride: 12 * w as usize,
                        x_sampling: 1,
                        y_sampling: 1,
                        fill_value: 0.0,
                        x_tile_coords: false,
                        y_tile_coords: false,
                    },
                )
                .unwrap();
            }
            (out, fb)
        };
        {
            let mut is = MemIStream::new(bytes.clone());
            let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
            let mut deep_part = mpi.deep_scanline_input_part_wrapper(1).unwrap();
            let mut compositor = crate::deep::CompositeDeepScanLine::new();
            compositor.add_source(&mut deep_part).unwrap();
            compositor.set_frame_buffer(&expected_fb).unwrap();
            compositor.read_pixels(0, 0, &mut expected_out).unwrap();
        }

        // Under test: `MultiPartInputFile::input_part` (the generic front
        // door's multipart-cache mirror).
        let mut got_out = vec![0u8; w as usize * 3 * 4];
        let mut is = MemIStream::new(bytes);
        let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
        assert_eq!(mpi.parts(), 2);
        let mut part = mpi.input_part(1).unwrap();
        assert_eq!(part.header().name().unwrap(), "deep");
        part.set_frame_buffer(&expected_fb).unwrap();
        part.read_pixels(0, 0, &mut got_out).unwrap();

        assert_eq!(
            got_out, expected_out,
            "input_part's deep-scanline flatten must match CompositeDeepScanLine directly"
        );
        assert_ne!(&got_out[8..12], &1.0f32.to_le_bytes()[..], "composite must have run");
    }

    /// A deep-TILED part read through `MultiPartInputFile::input_part` stays
    /// `Unsupported::Deep` — mirrors `deep_tiled_part_is_still_unsupported`
    /// for the standalone `InputFile`, see `input_part`'s
    /// `Storage::DeepTiled` arm comment for why.
    #[test]
    fn multipart_input_part_deep_tiled_is_still_unsupported() {
        use crate::deep::DeepTiledOutputFile;
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
        use crate::attr::{Channel, ChannelList, ChannelListAttribute};
        use imath_rs::V2f;

        let mut h = Header::with_dimensions(
            4,
            4,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 4,
                y_size: 4,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();

        let mut fb = crate::deep::DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 16,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        for name in ["Z", "A"] {
            fb.insert(
                name,
                crate::deep::frame_buffer::DeepSlice::new(
                    PixelType::Float, 0, 8, 32, 4, 1, 1, 0.0, false, false,
                )
                .unwrap(),
            )
            .unwrap();
        }

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &h).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            let sc = vec![0u8; 16 * 4];
            let data = vec![0u8; 0];
            df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();
        }

        let mut is = MemIStream::new(out.into_data());
        let mut mpi = MultiPartInputFile::new(&mut is).unwrap();
        match mpi.input_part(0) {
            Err(ExrError::Unsupported(Unsupported::Deep)) => {}
            Err(other) => panic!("expected Unsupported::Deep, got {other:?}"),
            Ok(_) => panic!("expected Unsupported::Deep, got Ok"),
        }
    }

    /// A header missing a required attribute is READ with C++'s default
    /// (`validation.c::validate_req_attr` with `adddefault = !strict`), not
    /// rejected.
    #[test]
    fn missing_required_attribute_is_defaulted_not_fatal() {
        let bytes = write_tiled(Compression::Zip);
        let patched = drop_attribute(&bytes, "screenWindowWidth", 0);
        let mut is = MemIStream::new(patched);
        let inf = InputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().screen_window_width().unwrap(), 1.0);
    }

    /// For a plain single-part image the version FLAGS decide the
    /// storage, not the `type` attribute — a tiled file with no `type` at all
    /// still dispatches to the tiled reader
    /// (`parse_header.c:2791-2798`, `validation.c::validate_part_type`).
    #[test]
    fn storage_comes_from_the_version_flags_not_the_type_attribute() {
        let bytes = write_tiled(Compression::Zip);
        let patched = drop_attribute(&bytes, "type", 0);
        let (_, fb) = make_plane();
        let mut is = MemIStream::new(patched);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut out = vec![0u8; expected_plane().len()];
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }


    /// A single-part SCANLINE file carrying the plane.
    fn write_scanline() -> Vec<u8> {
        let mut h = base_header(Compression::Zip);
        h.set_type(SCANLINE_IMAGE).unwrap();
        let (pixels, fb) = make_plane();
        let mut out = MemOStream::new();
        {
            let mut of = crate::scanline::OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, H).unwrap();
        }
        out.into_data()
    }

    /// A `type` that disagrees with the version flags is SILENTLY
    /// CORRECTED, not rejected — non-strict
    /// `OpenEXRCore/validation.c::validate_part_type` ("the flags win").
    #[test]
    fn type_disagreeing_with_the_flags_is_corrected_not_fatal() {
        let bytes = write_scanline();
        // The version word has no TILED flag, but claim `tiledimage` anyway.
        let patched = set_string_attribute(&bytes, "type", 0, TILED_IMAGE);
        let (_, fb) = make_plane();
        let mut is = MemIStream::new(patched);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert_eq!(inf.header().type_().unwrap(), SCANLINE_IMAGE);
        inf.set_frame_buffer(&fb).unwrap();
        let mut out = vec![0u8; expected_plane().len()];
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// The version NUMBER is checked and must be exactly 2 — fatal
    /// regardless of strictness (`parse_header.c::read_magic_and_flags`
    /// 2686-2697).  It used to go unchecked.
    #[test]
    fn unsupported_version_number_is_rejected() {
        let mut bytes = write_scanline();
        bytes[4] = 3; // version number field (low byte of the version word)
        let mut is = MemIStream::new(bytes);
        let Err(err) = InputFile::new(&mut is) else {
            panic!("version 3 must be rejected");
        };
        assert!(err.to_string().contains("unsupported version"), "{err}");
    }

    /// An unknown flag in the version word is fatal (same function,
    /// 2700-2709).
    #[test]
    fn unknown_version_flags_are_rejected() {
        let mut bytes = write_scanline();
        bytes[7] = 0x40; // a bit outside EXR_VALID_FLAGS
        let mut is = MemIStream::new(bytes);
        assert!(InputFile::new(&mut is).is_err());
    }

    /// An untyped TILED file with a destroyed offset table must be
    /// reconstructed with TILE framing.  The framing comes from the derived
    /// storage, not from `type` — with `autoAddType == false` there is no
    /// `type`, and scanline framing would misparse every chunk leader
    /// (`OpenEXRCore/chunk.c::extract_chunk_leader` switches on
    /// `part->storage_mode`).
    #[test]
    fn untyped_tiled_file_reconstructs_with_tile_framing() {
        let mut bytes = drop_attribute(&write_tiled(Compression::Zip), "type", 0);
        zero_offset_tables(&mut bytes);
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert!(inf.is_complete(), "the table must be fully reconstructed");
        inf.set_frame_buffer(&fb).unwrap();
        let mut out = vec![0u8; expected_plane().len()];
        inf.read_pixels(0, H - 1, &mut out).unwrap();
        assert_eq!(out, expected_plane());
    }

    /// Regression test: a tiled-as-scanlines `InputFile` whose chunk
    /// table has an UNRECOVERABLE hole (not merely zeroed, genuinely
    /// truncated) must fail at `set_frame_buffer` -- matching C++
    /// `Data::lockedSetFrameBuffer` (`ImfInputFile.cpp:386-394`), which
    /// throws whenever the tile-row cache is (re)built, before any read is
    /// attempted. Before the fix, `set_frame_buffer` always returned `Ok`
    /// (the guard lived only inside `buffered_read_pixels`'s PARTIAL-window
    /// branch), so a caller doing the ordinary `set_frame_buffer` then
    /// whole-window `read_pixels` sequence would sail straight into the
    /// fast path and attempt to read past the hole instead of erroring
    /// up front.
    #[test]
    fn set_frame_buffer_errors_on_unrecoverable_chunk_table() {
        // NONE compression + tiles that evenly divide the image -> every
        // chunk is a fixed, predictable size (leader: 5 x i32 = 20 bytes;
        // payload: 4*2 pixels x 2 bytes/HALF = 16 bytes; 36 bytes/chunk),
        // so truncating a known number of trailing bytes reliably destroys
        // the LAST chunk's leader (not just its payload), which the sweep
        // needs to read before it can mark that slot found.
        let mut bytes = write_tiled(Compression::NoCompression);
        zero_offset_tables(&mut bytes);
        // Cut well into the last chunk's 20-byte leader (not just its
        // 16-byte payload) so the reconstruction sweep cannot locate it.
        bytes.truncate(bytes.len() - 30);
        let (_, fb) = make_plane();

        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        assert!(
            !inf.is_complete(),
            "test needs a genuinely unrecoverable table"
        );
        let err = inf
            .set_frame_buffer(&fb)
            .expect_err("set_frame_buffer must reject an incomplete chunk table");
        assert!(err.to_string().contains("corrupt chunk table"), "{err}");
    }

    /// `InputFile::open` OWNS a
    /// path-opened stream (no caller-side `&mut` to keep alive) -- port of
    /// C++'s real, public `InputFile(const char*, int)` constructor. Covers
    /// BOTH storage kinds `InputFile` dispatches on (plain scanline, and
    /// tiled-served-as-scanlines), since those are two different `Reader`
    /// variants with independent generic plumbing.
    #[test]
    fn open_owns_a_path_opened_stream_and_reads_identically_to_new() {
        for (name, bytes) in [
            ("scanline", write_scanline()),
            ("tiled-as-scanlines", write_tiled(Compression::Zip)),
        ] {
            let path = std::env::temp_dir()
                .join(format!("exr_core_input_file_open_{name}_{}.exr", std::process::id()));
            std::fs::write(&path, &bytes).unwrap();

            let mut owned = InputFile::open(&path).unwrap();
            std::fs::remove_file(&path).ok();

            let mut is = MemIStream::new(bytes);
            let mut borrowed = InputFile::new(&mut is).unwrap();

            assert_eq!(
                owned.header().data_window().unwrap(),
                borrowed.header().data_window().unwrap(),
                "{name}"
            );
            assert_eq!(owned.is_complete(), borrowed.is_complete(), "{name}");

            let (_, fb) = make_plane();
            owned.set_frame_buffer(&fb).unwrap();
            borrowed.set_frame_buffer(&fb).unwrap();
            let mut got_owned = vec![0u8; expected_plane().len()];
            let mut got_borrowed = vec![0u8; expected_plane().len()];
            owned.read_pixels(0, H - 1, &mut got_owned).unwrap();
            borrowed.read_pixels(0, H - 1, &mut got_borrowed).unwrap();
            assert_eq!(got_owned, got_borrowed, "{name}: owned and borrowed reads must match");
            assert_eq!(got_owned, expected_plane(), "{name}: and must match the source pixels");
        }
    }

    /// Design bug fix: the tiled-as-scanlines row cache used to size an
    /// invalid `PixelType::NumPixelTypes` slice as 4 bytes (C++'s own `(s.type
    /// == HALF) ? 2 : 4`, safe there because a real C++ `Slice::Type` never
    /// holds that sentinel; our `PixelType` enum can). `FrameBuffer::insert`
    /// already rejects a `NumPixelTypes` slice at construction time, but
    /// `FrameBuffer::get_mut`/`iter_mut` return `&mut Slice` and do NOT
    /// re-validate on every mutation (documented, deliberate — see
    /// `frame_buffer.rs`), so a caller that mutates an already-inserted
    /// slice's `type_` post-insertion can still reach this path. Must error,
    /// not silently size the cache as if the type were FLOAT/UINT.
    #[test]
    fn tiled_as_scanlines_set_frame_buffer_rejects_invalid_pixel_type() {
        let bytes = write_tiled(Compression::NoCompression);
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();

        let (_, mut fb) = make_plane();
        fb.get_mut("R").unwrap().type_ = PixelType::NumPixelTypes;

        let err = inf
            .set_frame_buffer(&fb)
            .expect_err("an invalid pixel type must error, not size as 4 bytes");
        assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
    }

    // -----------------------------------------------------------------
    // InputFile on a deep scanline part flattens via
    // CompositeDeepScanLine (ImfInputFile.cpp:300-305).
    // -----------------------------------------------------------------

    /// Write a tiny (2x1) single-source deep scanline file with `Z`/`A`/`R`
    /// channels: column 0 has 2 front-to-back samples, column 1 has 1.
    fn write_small_deep_scanline() -> Vec<u8> {
        use crate::deep::frame_buffer::DeepSlice;
        use crate::deep::{DeepFrameBuffer, DeepScanLineOutputFile};

        let w = 2i32;
        let channels = ["Z", "A", "R"];
        let counts = [2u32, 1u32];
        // Z, A, R per column, front-to-back order (writer order == read order
        // for a single deep source, no re-sort needed).
        let z: [&[f32]; 2] = [&[0.0, 1.0], &[0.5]];
        let a: [&[f32]; 2] = [&[0.5, 0.5], &[0.25]];
        let r: [&[f32]; 2] = [&[1.0, 2.0], &[4.0]];
        let per_channel = [&z, &a, &r];

        let mut sc = Vec::with_capacity(w as usize * 4);
        for &c in &counts {
            sc.extend_from_slice(&c.to_le_bytes());
        }
        let total: u32 = counts.iter().sum();
        let mut data = vec![0u8; channels.len() * total as usize * 4];
        for (ch_idx, per_x) in per_channel.iter().enumerate() {
            let mut offset = 0usize;
            for (x, samples) in per_x.iter().enumerate() {
                assert_eq!(samples.len(), counts[x] as usize);
                for &v in *samples {
                    let byte_off = (ch_idx * total as usize + offset) * 4;
                    data[byte_off..byte_off + 4].copy_from_slice(&v.to_le_bytes());
                    offset += 1;
                }
            }
        }

        let mut h = Header::with_dimensions(
            w,
            1,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for name in &channels {
            cl.insert(*name, Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

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
        for (ch_idx, name) in channels.iter().enumerate() {
            fb.insert(
                *name,
                DeepSlice::new(
                    PixelType::Float,
                    ch_idx as i64 * total as i64 * 4,
                    8,
                    8 * w as usize,
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
        }

        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &h).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, 1).unwrap();
        }
        out.into_data()
    }

    fn flat_rga_output(w: i32) -> (Vec<u8>, FrameBuffer) {
        let out = vec![0u8; w as usize * 3 * 4];
        let mut fb = FrameBuffer::new();
        for (i, name) in ["Z", "A", "R"].iter().enumerate() {
            fb.insert(
                *name,
                Slice {
                    type_: PixelType::Float,
                    base_offset: i as i64 * 4,
                    x_stride: 12,
                    y_stride: 12 * w as usize,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
        }
        (out, fb)
    }

    /// `InputFile::new` on a deep *scanline* part must flatten it through
    /// `CompositeDeepScanLine`, producing byte-identical output to reading
    /// the same file directly via `DeepScanLineInputFile` +
    /// `CompositeDeepScanLine` (the already oracle-verified path — see
    /// `deep::composite_scanline`'s own `cpp_composite_deep_scanline_bit_exact`
    /// test). C++ 3.4.15 `ImfInputFile.cpp:300-305` routes exactly this way.
    #[test]
    fn deep_scanline_part_is_flattened_through_input_file() {
        let bytes = write_small_deep_scanline();

        // Reference: read directly via DeepScanLineInputFile + a fresh
        // CompositeDeepScanLine (the already-tested path).
        let (mut expected_out, expected_fb) = flat_rga_output(2);
        {
            let mut is = MemIStream::new(bytes.clone());
            let mut deep = crate::deep::DeepScanLineInputFile::new(&mut is).unwrap();
            let mut compositor = crate::deep::CompositeDeepScanLine::new();
            compositor.add_source(&mut deep).unwrap();
            compositor.set_frame_buffer(&expected_fb).unwrap();
            compositor.read_pixels(0, 0, &mut expected_out).unwrap();
        }

        // Under test: the generic InputFile front door.
        let (mut got_out, got_fb) = flat_rga_output(2);
        let mut is = MemIStream::new(bytes);
        let mut inf = InputFile::new(&mut is).unwrap();
        inf.set_frame_buffer(&got_fb).unwrap();
        inf.read_pixels(0, 0, &mut got_out).unwrap();

        assert_eq!(got_out, expected_out, "InputFile-flattened deep read must match CompositeDeepScanLine directly");
        // Sanity: the composite actually did something (front-to-back accumulate
        // over 2 samples at x=0 must differ from either raw R sample).
        assert_ne!(&got_out[8..12], &1.0f32.to_le_bytes()[..]);
    }

    /// A deep *tiled* part is NOT flattened (see `Reader::Deep`'s and
    /// `Storage::DeepTiled`'s doc comments in `build` for why the equivalent
    /// C++ code path is dead for this storage kind) — still a precisely-typed
    /// `Unsupported::Deep`, not a panic or wrong-pixel silent fallback.
    #[test]
    fn deep_tiled_part_is_still_unsupported() {
        use crate::deep::frame_buffer::DeepSlice;
        use crate::deep::{DeepFrameBuffer, DeepTiledOutputFile};

        let mut h = Header::with_dimensions(
            4,
            4,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 4,
                y_size: 4,
                mode: LevelMode::OneLevel,
                rounding_mode: crate::attr::tile_description::LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();

        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 16,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        for name in ["Z", "A"] {
            fb.insert(
                name,
                DeepSlice::new(PixelType::Float, 0, 8, 32, 4, 1, 1, 0.0, false, false).unwrap(),
            )
            .unwrap();
        }

        let mut out = MemOStream::new();
        {
            let mut df = DeepTiledOutputFile::new(&mut out, &h).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            let sc = vec![0u8; 16 * 4];
            let data = vec![0u8; 0];
            df.write_tile(&sc, &data, 0, 0, 0, 0).unwrap();
        }

        let mut is = MemIStream::new(out.into_data());
        match InputFile::new(&mut is) {
            Err(ExrError::Unsupported(Unsupported::Deep)) => {}
            Err(other) => panic!("expected Unsupported::Deep, got {other:?}"),
            Ok(_) => panic!("expected Unsupported::Deep, got Ok"),
        }
    }
}
