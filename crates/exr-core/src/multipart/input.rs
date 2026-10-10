//! Port of OpenEXR `ImfMultiPartInputFile.{h,cpp}` — the multipart container
//! reader.
//!
//! Reads the multipart file prologue written by [`super::MultiPartOutputFile`]
//! (see [`super::output`] for the byte layout):
//!
//! 1. magic + version word; the [`MULTI_PART_FILE_FLAG`][crate::container::MULTI_PART_FILE_FLAG]
//!    bit must be set;
//! 2. every part header in sequence, the header *list* terminated by an extra
//!    empty attribute name.  C++ detects the end of the list by reading the
//!    first byte of the next header: a `\0` there means the list is finished
//!    (the "second terminator"); any other byte is the first byte of the next
//!    header's first attribute name (`OpenEXRCore/parse_header.c`);
//! 3. the per-part chunk offset tables, in part order, each sized by that
//!    part's `chunkCount`;
//! 4. the chunks (read on demand by the part readers; each multipart chunk is
//!    prefixed with its `partNumber`).
//!
//! `read_prologue` is the crate's ONE file-prologue reader: `MultiPartInputFile`
//! is a thin wrapper over it, and so is every single-part reader constructor
//! (`scanline::InputFile`, `TiledInputFile`, both deep readers) — mirroring C++,
//! where one `internal_exr_parse_header` sits behind them all and `InputFile`
//! literally builds a `MultiPartInputFile` and takes part 0.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMultiPartInputFile.cpp>
//! header-list read + partNumber verify: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/parse_header.c>
//! + `OpenEXRCore/chunk.c`.

use std::marker::PhantomData;
use std::ops::DerefMut;

use crate::chunk::offsets;
use crate::container::{version, xdr};
use crate::error::{ExrError, Result, Unsupported};
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::deep::scanline_input::DeepScanLineInputFileState;
use crate::deep::tiled_input::DeepTiledInputFileState;
use crate::header::{self, Header};
use crate::io::IStream;
use crate::scanline::{
    DeepScanLineReaderState, InputFilePartReader, PartReaderState, ScanLineInputFileState,
    TiledReaderState,
};
use crate::tiled::misc::precalculate_tile_info;
use crate::tiled::tile_offsets::TileOffsets;
use crate::tiled::{TiledInputFileState, TiledInputPartReader};

use super::part::{DeepScanLineInputPart, DeepTiledInputPart, InputPart, TiledInputPart};

use super::output::get_chunk_offset_table_size_for_read;

/// Rebuild every part's chunk offset table in one sequential sweep over the
/// chunk region — the multipart form of
/// `OpenEXRCore/chunk.c::reconstruct_chunk_table`.
///
/// Every multipart chunk leads with its `partNumber`, so a single pass can fill
/// all the parts' tables; the shared sweep
/// ([`crate::chunk::offsets::reconstruct`]) parses each chunk leader with that
/// part's framing and stops cleanly at the first fault.
fn reconstruct_all_parts(
    stream: &mut dyn IStream,
    parts: &mut [PartData],
    multipart: bool,
    first_chunk_pos: u64,
) -> Result<()> {
    let framings: Vec<offsets::Framing> = parts
        .iter()
        .map(|p| part_framing(&p.header, p.storage))
        .collect::<Result<_>>()?;

    let mut views: Vec<offsets::PartOffsets<'_>> = parts
        .iter_mut()
        .zip(framings)
        .map(|(p, framing)| offsets::PartOffsets {
            framing,
            offsets: &mut p.chunk_offsets,
        })
        .collect();

    offsets::reconstruct(stream, &mut views, multipart, first_chunk_pos)
}

/// The on-disk chunk framing of one part, derived from its **storage** (C++
/// chunk-leader dispatch in `chunk.c::extract_chunk_leader`, which switches on
/// `part->storage_mode`).
///
/// Storage, not the `type` attribute: with `autoAddType == false` (the
/// `InputFile` path) a single-part tiled file carries no `type` at all, and
/// framing it as a scanline part would make the reconstruction sweep misparse
/// every chunk leader.
fn part_framing(h: &Header, storage: Storage) -> Result<offsets::Framing> {
    let deep = storage.is_deep();
    let dw = h.data_window()?;

    if storage.is_tiled() {
        let td = h.tile_description()?;
        let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
            precalculate_tile_info(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
        // Geometry only — used as the (dx, dy, lx, ly) → slot oracle.
        let geometry = TileOffsets::new(
            td.mode,
            num_x_levels,
            num_y_levels,
            &num_x_tiles,
            &num_y_tiles,
        )?;
        return Ok(if deep {
            offsets::Framing::DeepTile(geometry)
        } else {
            offsets::Framing::Tile(geometry)
        });
    }

    let lines_per_chunk = crate::compressor::num_lines_in_buffer(h.compression()?)?;
    let line_order = h.line_order()?;
    Ok(if deep {
        offsets::Framing::DeepScanline {
            min_y: dw.min.y,
            lines_per_chunk,
            line_order,
        }
    } else {
        offsets::Framing::Scanline {
            min_y: dw.min.y,
            lines_per_chunk,
            line_order,
        }
    })
}

/// How a part's pixel data is stored — port of `exr_storage_t`
/// (`OpenEXRCore/openexr_part.h`), the value `InputFile::initialize`
/// dispatches on.
///
/// For a plain single-part file the storage comes from the version **flags**
/// alone; only a deep or multipart file lets the `type` attribute set it
/// (`OpenEXRCore/parse_header.c:1789-1810, 2791-2798`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `scanlineimage`
    Scanline,
    /// `tiledimage`
    Tiled,
    /// `deepscanline`
    DeepScanline,
    /// `deeptile`
    DeepTiled,
    /// A `type` string this build does not know — kept for forward
    /// compatibility exactly as C++ `EXR_STORAGE_UNKNOWN`.
    Unknown,
}

impl Storage {
    /// The `type` string C++ expects for this storage
    /// (`validation.c::validate_part_type`'s `expectedtype`).
    fn expected_type(self) -> Option<&'static str> {
        match self {
            Storage::Scanline => Some(header::SCANLINE_IMAGE),
            Storage::Tiled => Some(header::TILED_IMAGE),
            Storage::DeepScanline => Some(header::DEEP_SCANLINE),
            Storage::DeepTiled => Some(header::DEEP_TILE),
            Storage::Unknown => None,
        }
    }

    /// Whether this storage is tile-framed (the `isTiled` argument
    /// `sanity_check` takes).
    pub fn is_tiled(self) -> bool {
        matches!(self, Storage::Tiled | Storage::DeepTiled)
    }

    /// Whether this storage carries deep (variable-sample) data.
    pub fn is_deep(self) -> bool {
        matches!(self, Storage::DeepScanline | Storage::DeepTiled)
    }
}

/// One part's parsed state — port of `MultiPartInputFile::Data::Part`
/// (the `InputPartData` payload this reader needs).
pub(crate) struct PartData {
    /// The part header, as read from the file (validated + defaulted).
    pub(crate) header: Header,
    /// This part's pixel storage, derived exactly as C++ derives it.
    pub(crate) storage: Storage,
    /// This part's chunk offset table — C++ `InputPartData::chunkOffsets`.
    pub(crate) chunk_offsets: Vec<u64>,
    /// `false` when this part's chunk offset table still has a hole **after**
    /// reconstruction — C++ `Context::chunkTableValid(part)` /
    /// `MultiPartInputFile::partComplete` (`exr_validate_chunk_table` runs
    /// `extract_chunk_table`, i.e. reconstruction, then re-checks every entry).
    pub(crate) complete: bool,

    /// Persisted reader state for
    /// [`MultiPartInputFile::input_part`] — built lazily, kept across every
    /// later call, mirrors C++'s per-part `_data->parts[n].file` cache.
    /// Covers scanline AND tiled-as-scanlines storage in one slot via
    /// [`PartReaderState`].
    pub(crate) input_reader_state: Option<PartReaderState>,
    /// The same persistence, for [`MultiPartInputFile::tiled_input_part`]
    /// (the direct `TiledInputFile` accessor — a separate slot from
    /// `input_reader_state` above since C++ keys its own single per-part
    /// cache slot by the requested type `T`, so using both accessors on one
    /// part is not really a C++-supported pattern either).
    pub(crate) tiled_reader_state: Option<TiledInputFileState>,
    /// The same persistence, for [`MultiPartInputFile::deep_scanline_input_part`].
    pub(crate) deep_scanline_reader_state: Option<DeepScanLineInputFileState>,
    /// The same persistence, for [`MultiPartInputFile::deep_tiled_input_part`].
    pub(crate) deep_tiled_reader_state: Option<DeepTiledInputFileState>,
}

// ---------------------------------------------------------------------------
// Deep part-reader views — mirror
// `scanline::InputFilePartReader` / `tiled::TiledInputPartReader`'s shape.
// Defined here rather than in `deep/` since `DeepScanLineInputFileState` /
// `DeepTiledInputFileState`'s accessor methods are already `pub(crate)`/`pub`
// (split out for exactly this use), so no cross-file visibility bump was needed —
// only the container-side caching wiring itself was missing.
// ---------------------------------------------------------------------------

/// A borrowing view over one multipart deep-scanline part's persisted reader
/// state, the read-side mirror of `DeepScanLineOutputPartWriter`.
/// Exposes the same public surface as [`DeepScanLineInputFile`] (`header`,
/// `frame_buffer`, `is_complete`, `version`, `set_frame_buffer`,
/// `read_pixel_sample_counts`, `read_pixels`).
pub struct DeepScanLineInputPartReader<'p> {
    stream: &'p mut dyn IStream,
    state: &'p mut DeepScanLineInputFileState,
}

impl<'p> DeepScanLineInputPartReader<'p> {
    fn new(stream: &'p mut dyn IStream, state: &'p mut DeepScanLineInputFileState) -> Self {
        Self { stream, state }
    }

    pub fn header(&self) -> &Header {
        self.state.header()
    }
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
    pub fn version(&self) -> i32 {
        self.state.version()
    }
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }
    pub fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixel_sample_counts(self.stream, scan_line1, scan_line2, sample_counts)
    }
    pub fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.state.read_pixels(self.stream, scan_line1, scan_line2, sample_counts, sample_data)
    }

    /// This part's `DeepScanLineCopySource` support (`copy_pixels`,
    /// a duck-typed shape mirroring
    /// `TiledCopySource` so a borrowing view can implement it).
    /// Scanline `first_scan_line`'s chunk RAW (still-packed count table +
    /// sample data, no decompression).
    pub fn raw_pixel_data(
        &mut self,
        first_scan_line: i32,
    ) -> Result<(i32, Vec<u8>, Vec<u8>, u64)> {
        self.state.raw_pixel_data(self.stream, first_scan_line)
    }
}

/// A borrowing view over one multipart deep-tiled part's persisted reader
/// state, the read-side mirror of `DeepTiledOutputPartWriter`. Exposes
/// the same public surface as [`DeepTiledInputFile`] needed by this crate's
/// own callers (`header`, `frame_buffer`, `is_complete`, `version`,
/// `set_frame_buffer`, `num_x_tiles`/`num_y_tiles`, `read_pixel_sample_counts`,
/// `read_tiles`).
pub struct DeepTiledInputPartReader<'p> {
    stream: &'p mut dyn IStream,
    state: &'p mut DeepTiledInputFileState,
}

impl<'p> DeepTiledInputPartReader<'p> {
    fn new(stream: &'p mut dyn IStream, state: &'p mut DeepTiledInputFileState) -> Self {
        Self { stream, state }
    }

    pub fn header(&self) -> &Header {
        self.state.header()
    }
    pub fn frame_buffer(&self) -> &DeepFrameBuffer {
        self.state.frame_buffer()
    }
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
    pub fn version(&self) -> i32 {
        self.state.version()
    }
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.state.num_x_tiles(lx)
    }
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.state.num_y_tiles(ly)
    }
    /// Number of resolution levels along X — needed by `tests/corpus.rs`'s
    /// `decode_deep_tiled`, generic over any deep-tiled reader.
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
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<imath_rs::Box2i> {
        self.state.data_window_for_level(lx, ly)
    }
    pub fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.state.set_frame_buffer(frame_buffer)
    }
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
        self.state
            .read_pixel_sample_counts(self.stream, dx1, dx2, dy1, dy2, lx, ly, sample_counts)
    }
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
        self.state.read_tiles(
            self.stream,
            dx1,
            dx2,
            dy1,
            dy2,
            lx,
            ly,
            sample_counts,
            sample_data,
        )
    }

    // -----------------------------------------------------------------------
    // `DeepTiledCopySource` support (`copy_pixels`, a
    // duck-typed shape mirroring `TiledCopySource` so a borrowing view can
    // implement it). Added here so
    // `multipart::part::DeepTiledInputPart` (which now wraps this view
    // instead of an owned `DeepTiledInputFile`) can implement it again.
    // -----------------------------------------------------------------------

    /// Total number of tiles across every level.
    pub fn total_tiles(&self) -> Result<i64> {
        self.state.total_tiles()
    }

    /// This part's on-disk tile order.
    pub fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        self.state.tile_order()
    }

    /// Tile `(dx, dy, lx, ly)`'s raw (still-encoded) on-disk bytes, plus the
    /// chunk-header-verified coordinates.
    pub fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<crate::deep::tiled_input::RawTileData> {
        self.state.raw_tile_data(self.stream, dx, dy, lx, ly)
    }
}

/// A whole file's parsed prologue: version word, flags and every part.
///
/// This is the ONE place a file's magic + version + header list + chunk offset
/// tables are read and validated. `MultiPartInputFile::new` is a thin wrapper
/// over it, and `scanline::InputFile::new` uses it to dispatch to the right
/// part reader — mirroring C++, where `InputFile` literally constructs a
/// `MultiPartInputFile` and takes part 0 (`ImfInputFile.cpp:126-136`).
pub(crate) struct Prologue {
    /// The raw file version word.
    pub(crate) version: i32,
    /// `true` when the `MULTI_PART` flag is set.
    pub(crate) multipart: bool,
    /// Every part, in part order.
    pub(crate) parts: Vec<PartData>,
}

/// Read and validate a file's whole prologue — port of
/// `OpenEXRCore/parse_header.c::internal_exr_parse_header` +
/// `validation.c::internal_exr_validate_read_part`, in the **non-strict** mode
/// the C++ `InputFile` / `MultiPartInputFile` classes select
/// (`ContextInitializer().strictHeaderValidation(false)`,
/// `ImfInputFile.cpp:137-144`, `ImfMultiPartInputFile.cpp:80-90`).
///
/// `auto_add_type` mirrors the `MultiPartInputFile` ctor argument
/// (`ImfMultiPartInputFile.h:49`, default `true`): a part with no `type`
/// attribute gets one invented from the version flags.  `InputFile` passes
/// `false` (`ImfInputFile.cpp:131`).
///
/// Accepts **single-part files too** — C++ `MultiPartInputFile` opens them as a
/// one-part file (`_ctxt.partCount()` is simply 1); only the header-list
/// terminator differs on the wire.
pub(crate) fn read_prologue(
    stream: &mut dyn IStream,
    auto_add_type: bool,
) -> Result<Prologue> {
    // --- magic + version word ---------------------------------------------
    // C++ `parse_header.c::read_magic_and_flags` (2645-2711).
    let magic = xdr::read_i32(stream)?;
    if (magic as u32) != version::MAGIC {
        return Err(ExrError::Input("File is not an OpenEXR file.".into()));
    }
    let version_word = xdr::read_i32(stream)?;

    // C++ 2686-2697: the version NUMBER must be exactly 2; anything else is
    // `EXR_ERR_FILE_BAD_HEADER`.  (Previously unchecked here.)
    let ver_num = version::get_version(version_word as u32);
    if ver_num != version::EXR_VERSION {
        return Err(ExrError::Input(format!(
            "File is of an unsupported version: {ver_num} (expected {}).",
            version::EXR_VERSION
        )));
    }

    let flags = version::get_flags(version_word as u32);
    if !version::supports_flags(flags) {
        return Err(ExrError::Input(
            "The file format version number's flag field contains unknown flags.".into(),
        ));
    }

    let tiled_flag = version::is_tiled(flags);
    let multipart = version::is_multi_part(flags);
    let non_image = version::is_non_image(flags);

    // C++ 2781-2790: the single-part TILED flag may not be combined with the
    // deep or multipart flags — "this appears to always be fatal, so do not
    // check strict / not".
    if tiled_flag && (non_image || multipart) {
        return Err(ExrError::Input(format!(
            "Invalid combination of version flags: single part tiled flag found, \
             but also marked as deep ({non_image}) or multipart ({multipart})."
        )));
    }

    // --- read the header(s) -----------------------------------------------
    // C++ `parse_header.c` (~2734): a multipart header list ends when, after a
    // header's own empty-name terminator, the next byte is another `\0` (the
    // "second terminator"). A single-part file has exactly one header and no
    // extra terminator.
    let mut headers: Vec<Header> = Vec::new();
    if multipart {
        loop {
            headers.push(Header::read_attributes_from(stream, version_word)?);

            let pos = stream.tellg()?;
            let mut byte = [0u8; 1];
            stream.read(&mut byte)?;
            if byte[0] == 0 {
                break;
            }
            stream.seekg(pos)?;
        }
    } else {
        headers.push(Header::read_attributes_from(stream, version_word)?);
    }

    // --- validate every part ----------------------------------------------
    let mut validated: Vec<(Header, Storage)> = Vec::with_capacity(headers.len());
    for header in headers {
        validated.push(validate_read_part(
            header,
            tiled_flag,
            multipart,
            non_image,
            auto_add_type,
        )?);
    }

    // Every part after the first must agree with part 0
    // on the SHARED attributes (`displayWindow`, `pixelAspectRatio` bitwise;
    // `timecode`, `chromaticities` by value when the LATER part has one) —
    // C++ `parse_header.c:2855-2887` runs `internal_exr_validate_shared_attrs`
    // unconditionally right after every header is parsed and fails the WHOLE
    // open on the first mismatch (there is no read-side "override", unlike
    // the writer's `overrideSharedAttributes`); `validation.c:725-835` is the
    // exact rule set, already ported as `check_shared_attributes_values` for
    // the writer's own conflict check — reused here in the same direction
    // (`part 0` = `src`, the later part = `dst`) since the rule is symmetric
    // in what it calls "src"/"dst", not in which file said which.
    if let [(first_header, _), rest @ ..] = validated.as_slice() {
        let first_header = first_header.clone();
        for (i, (header, _)) in rest.iter().enumerate() {
            let conflicting =
                crate::multipart::output::check_shared_attributes_values(&first_header, header)?;
            if !conflicting.is_empty() {
                return Err(ExrError::Input(format!(
                    "Part {} ({}) has non-conforming shared attributes: {}.",
                    i + 1,
                    header.name().unwrap_or(""),
                    conflicting.join(", ")
                )));
            }
        }
    }

    // --- read each part's chunk offset table ------------------------------
    // Table positions are computed ANALYTICALLY from each part's header-derived
    // `chunkCount` -- pure arithmetic, no stream access -- mirroring C++
    // `parse_header.c`'s `chunk_table_offset` computation, which runs for
    // every part BEFORE any table is actually read. This is what lets a
    // corrupt / oversized / truncated table in one part never prevent every
    // OTHER part (and the file as a whole) from opening: this reader always
    // knows exactly where the NEXT part's table starts, independent of
    // whatever happened trying to read the current one.
    //
    // `read_offset_table_lenient` never hard-fails on a validation problem
    // (unlike `chunk::RawImage`'s own, deliberately-stricter use of
    // `read_offset_table`) -- C++ defers ALL of `extract_chunk_table`'s
    // failure modes to `Context::chunkTableValid`, which swallows them into
    // a plain `isComplete() == false`; the real error only ever surfaces
    // from an actual chunk read. `InputFile::new`/`MultiPartInputFile::new`
    // must therefore succeed on a header-only, truncated, or
    // bogus-chunk-count file, exactly as C++'s constructors do.
    let mut parts: Vec<PartData> = Vec::with_capacity(validated.len());
    let mut any_incomplete = false;
    let mut table_pos = stream.tellg()?;
    for (header, storage) in validated {
        // Sizing keys off the part's **storage**, not off the `type` string:
        // `InputFile` opens with `autoAddType == false`, so a single-part tiled
        // file legitimately has no `type` attribute and only the version flags
        // say it is tiled (C++ sizes from `part->storage_mode` likewise).
        // The read-side variant never errors on an unknown storage type
        // (0 chunks when no `chunkCount` is present) -- see its doc comment.
        let chunk_count = get_chunk_offset_table_size_for_read(&header, storage)?;

        let this_table_pos = table_pos;
        // Advance analytically for the NEXT part regardless of what this
        // part's table read below actually manages -- a negative or absurd
        // `chunk_count` (rejected inside `read_offset_table_lenient`, not
        // here) must not desynchronize where later parts' tables are.
        table_pos = this_table_pos.saturating_add(
            chunk_count.max(0) as u64 * offsets::OFFSET_BYTES,
        );

        let table =
            offsets::read_offset_table_lenient(stream, chunk_count as i64, this_table_pos)?;
        any_incomplete |= !table.complete;
        log::debug!(
            "part {}: storage={storage:?} chunkCount={chunk_count} complete={}",
            parts.len(),
            table.complete
        );
        parts.push(PartData {
            header,
            storage,
            chunk_offsets: table.offsets,
            complete: table.complete,
            input_reader_state: None,
            tiled_reader_state: None,
            deep_scanline_reader_state: None,
            deep_tiled_reader_state: None,
        });
    }

    // An incomplete table in ANY part → rebuild every part's table in one
    // sequential sweep over the chunk region (each multipart chunk leads with
    // its `partNumber`, so one pass fills all the parts) — C++
    // `OpenEXRCore/chunk.c::reconstruct_chunk_table`.
    if any_incomplete {
        log::debug!("incomplete chunk table(s), reconstructing");
        // `table_pos` now holds the analytically-computed end of the LAST
        // part's table -- i.e. where the chunk region begins -- regardless
        // of whether every part's table was actually readable. Seek there
        // explicitly rather than trusting the stream's current position,
        // which a degraded (empty, un-read) table may have left short of
        // where a real table of that size would have ended.
        let first_chunk_pos = table_pos;
        stream.seekg(first_chunk_pos)?;
        reconstruct_all_parts(stream, &mut parts, multipart, first_chunk_pos)?;

        // `partComplete` is evaluated AFTER reconstruction in C++:
        // `Context::chunkTableValid` → `exr_validate_chunk_table`
        // (`OpenEXRCore/part.c:471`) calls `extract_chunk_table` — which
        // reconstructs — and only then re-checks every entry against
        // `[chunkmin, file_size)`.  A table we rebuilt in full is therefore
        // complete; one still holding a hole is not.
        let stream_size = stream.size();
        for p in &mut parts {
            // A part whose table was invalidated at the LENGTH-validation
            // stage (`read_offset_table_lenient` -- `chunk_count`
            // itself was <= 0, over the 2^20 cap, or the table couldn't
            // possibly fit the file) has `chunk_offsets` deliberately empty:
            // never allocated in the first place, and never a candidate for
            // reconstruction (C++ doesn't attempt it either -- see that
            // function's docs). `table_complete` on an EMPTY slice is
            // vacuously `true` (`[].iter().all(..)`), which would silently
            // un-invalidate exactly the parts this whole mechanism exists to
            // flag; guard against that explicitly rather than changing
            // `table_complete`'s general (and otherwise correct) semantics.
            p.complete = !p.chunk_offsets.is_empty()
                && offsets::table_complete(&p.chunk_offsets, first_chunk_pos, stream_size);
        }
    }

    Ok(Prologue {
        version: version_word,
        multipart,
        parts,
    })
}

/// Validate one part's header and derive its storage — port of
/// `validation.c::internal_exr_validate_read_part` in non-strict mode
/// (`validate_req_attr(f, curpart, adddefault = !strict_header)` +
/// `validate_part_type`), plus the storage derivation in
/// `parse_header.c:1789-1810 / 2791-2798`.
fn validate_read_part(
    mut h: Header,
    tiled_flag: bool,
    multipart: bool,
    non_image: bool,
    auto_add_type: bool,
) -> Result<(Header, Storage)> {
    // Required-attribute defaulting + name/type/version handling is
    // now the single shared `Header::validate_for_read` (also used by
    // `Header::read_checked`, which used to skip this entirely). This
    // function still owns the `Storage` derivation and the `validate_part_type`
    // fix-up below, which need the `Storage` enum (out of `header.rs`'s scope).
    h.validate_for_read(tiled_flag, non_image, multipart, auto_add_type)?;

    // --- storage ----------------------------------------------------------
    // C++ 2791-2798: for a plain single-part image the flags alone decide; the
    // `type` attribute only sets the storage when the file is deep or
    // multipart (2789-2810).
    let storage = if !non_image && !multipart {
        if tiled_flag {
            Storage::Tiled
        } else {
            Storage::Scanline
        }
    } else {
        // Bytes, not `type_()`: any byte string is a legal `type` to the core
        // (`check_populate_type`, `parse_header.c:1789-1819`).
        match h.type_bytes()? {
            b if b == header::SCANLINE_IMAGE.as_bytes() => Storage::Scanline,
            b if b == header::TILED_IMAGE.as_bytes() => Storage::Tiled,
            b if b == header::DEEP_SCANLINE.as_bytes() => Storage::DeepScanline,
            b if b == header::DEEP_TILE.as_bytes() => Storage::DeepTiled,
            // Non-strict: an unknown type string is kept for forward
            // compatibility (parse_header.c:1809-1819).
            _ => Storage::Unknown,
        }
    };

    // --- validate_part_type: the type attribute must match the storage -----
    // Non-strict C++ "silently changed this" (validation.c): the FLAGS win and
    // the `type` attribute is rewritten to the expected string. Strict mode
    // would error; the reader classes are never strict.
    if h.has_type()
        && let Some(expected) = storage.expected_type()
        && h.type_bytes()? != expected.as_bytes()
    {
        log::debug!(
            "part type '{}' disagrees with the file flags; correcting to '{expected}'",
            String::from_utf8_lossy(h.type_bytes()?)
        );
        h.set_type(expected)?;
    }

    // C++ runs the rest of the header checks after the type fix-ups.
    h.sanity_check_at_open(storage.is_tiled(), multipart)?;

    Ok((h, storage))
}

/// Multipart EXR reader — port of `Imf::MultiPartInputFile`.
///
/// Construct with [`MultiPartInputFile::new`]: it reads the magic + version
/// word (validating the multipart flag), every part header, and every part's
/// chunk offset table.  Inspect parts with
/// [`parts`](MultiPartInputFile::parts), [`header`](MultiPartInputFile::header)
/// and [`part_type`](MultiPartInputFile::part_type); read a scanline part with
/// [`input_part`](MultiPartInputFile::input_part).
pub struct MultiPartInputFile<'a, S = &'a mut dyn IStream> {
    /// The input stream, addressed through `DerefMut<Target = dyn IStream>`
    /// (the C++ `_ctxt`'s stream),
    /// generic over the stream HOLDER, mirroring
    /// `scanline::OutputFile`'s `OutputFile<'a, S>`/`OutputFileOwned` split
    /// on the write side, so a reader can OWN a path-opened stream (needed
    /// by [`crate::rgba::RgbaInputFile::open`]) instead of only ever
    /// borrowing one the caller keeps alive.
    stream: S,
    /// Raw file version word — C++ `_ctxt.version()`.
    version: i32,
    /// Per-part state in part order — C++ `Data::parts`.
    parts: Vec<PartData>,
    /// Whether the file's version word carries `MULTI_PART_FILE_FLAG`. A
    /// single-part file opened through this reader has NO
    /// `partNumber` chunk prefix — only a genuinely multipart file does
    /// (`OpenEXRCore/chunk.c:955-983`, `rdcnt = ctxt->is_multipart ? 2 : 1`).
    multipart: bool,
    /// Captured once at construction (C++ `numThreads`),
    /// forwarded to each part's `*_mt` constructor when its reader state is
    /// first built (scanline, tiled and deep parts alike).
    num_threads: Option<i32>,
    /// Ties `'a` to the struct even when `S` (e.g. `Box<dyn IStream + 'a>`)
    /// is the only field that mentions it — same purpose as
    /// `scanline::OutputFile`'s own `_stream_lt` marker; without it, a
    /// generic `S` with no OTHER field referencing `'a` leaves the lifetime
    /// parameter structurally unused (`E0392`).
    _stream_lt: PhantomData<&'a mut ()>,
}

/// A [`MultiPartInputFile`] that **owns** its input stream (`Box<dyn
/// IStream>`) — the read-side analogue of
/// [`scanline::OutputFileOwned`](crate::scanline::OutputFileOwned). Build it
/// with [`MultiPartInputFile::new_owned`]; behaves identically to the
/// borrowed default for the same input.
pub type MultiPartInputFileOwned<'a> = MultiPartInputFile<'a, Box<dyn IStream + 'a>>;

// Borrowed-stream constructor (the default `S = &'a mut dyn IStream`).
impl<'a> MultiPartInputFile<'a, &'a mut dyn IStream> {
    /// Open a `MultiPartInputFile` on `stream` — port of
    /// `MultiPartInputFile::MultiPartInputFile(IStream&, int, bool)`.
    ///
    /// Reads the magic + version word, then every part header until the
    /// header-list terminator, then every part's chunk offset table.
    ///
    /// # Errors
    /// * [`ExrError::Input`] on bad magic, unknown version flags, or a missing
    ///   multipart flag.
    /// * Propagates header / attribute parse errors and stream errors.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::build(stream, None)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(stream: &'a mut dyn IStream, num_threads: Option<i32>) -> Result<Self> {
        Self::build(stream, num_threads)
    }
}

// Owned-stream constructor (`S = Box<dyn IStream + 'a>`).
impl<'a> MultiPartInputFile<'a, Box<dyn IStream + 'a>> {
    /// Open a [`MultiPartInputFileOwned`] that **owns** its input stream —
    /// the owned-stream analogue of [`new`](MultiPartInputFile::new). Pass a
    /// boxed concrete stream, e.g. `Box::new(StdIfStream::open(path)?)`.
    pub fn new_owned(stream: Box<dyn IStream + 'a>) -> Result<Self> {
        Self::build(stream, None)
    }
}

// Shared read path, generic over the stream holder `S`. Covers BOTH the
// borrowed default (`S = &'a mut dyn IStream`) and the owned `Box<dyn
// IStream>` variant — every method below addresses the stream through
// `DerefMut<Target = dyn IStream>`, so the two paths are a single source of
// truth.
impl<'a, S> MultiPartInputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Shared constructor body for [`new`](Self::new) and
    /// [`new_owned`](Self::new_owned).
    fn build(mut stream: S, num_threads: Option<i32>) -> Result<Self> {
        // ONE prologue reader for the whole crate (magic + version + every
        // header + every chunk offset table, with reconstruction) — C++
        // likewise has a single `internal_exr_parse_header` behind both
        // `MultiPartInputFile` and `InputFile`.
        //
        // `auto_add_type = true` mirrors the C++ ctor's default
        // (`ImfMultiPartInputFile.h:49`); `InputFile` passes `false`.
        //
        // A SINGLE-part file is accepted: C++ `MultiPartInputFile` opens it as
        // a one-part file (`_ctxt.partCount() == 1`).  We used to reject it
        // with "file is not marked as multipart".
        let prologue = read_prologue(&mut *stream, true)?;
        Ok(Self {
            stream,
            version: prologue.version,
            parts: prologue.parts,
            multipart: prologue.multipart,
            num_threads,
            _stream_lt: PhantomData,
        })
    }

    /// `true` when part `part_number`'s chunk offset table is complete — port
    /// of `bool MultiPartInputFile::partComplete(int part) const`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `part_number` is out of range.
    pub fn part_complete(&self, part_number: i32) -> Result<bool> {
        self.part(part_number).map(|p| p.complete)
    }

    /// This part's chunk offset table (unit tests inspect the on-disk layout).
    #[cfg(test)]
    pub(crate) fn chunk_offsets(&self, n: i32) -> Result<&[u64]> {
        self.part(n).map(|p| p.chunk_offsets.as_slice())
    }

    /// Number of parts — port of `int MultiPartInputFile::parts() const`.
    pub fn parts(&self) -> i32 {
        self.parts.len() as i32
    }

    /// The raw file version word — port of `int MultiPartInputFile::version()`.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// The header of part `n` — port of
    /// `const Header& MultiPartInputFile::header(int) const`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range.
    pub fn header(&self, n: i32) -> Result<&Header> {
        self.part(n).map(|p| &p.header)
    }

    /// The `type` string of part `n` (empty when the header carries no `type`).
    pub fn part_type(&self, n: i32) -> Result<&str> {
        let h = self.header(n)?;
        if h.has_type() { h.type_() } else { Ok("") }
    }

    /// Open the generic part reader for part `n` — port of
    /// `MultiPartInputFile::getInputPart<InputFile>(int)`.
    ///
    /// Returns an [`InputFilePartReader`] — a **borrowing view** over this
    /// part's PERSISTED reader state (built once,
    /// on first access, and kept across every later call, mirroring
    /// `MultiPartOutputFile::output_part`'s write-side caching exactly).
    /// Drive it with `set_frame_buffer` + `read_pixels` exactly like a
    /// single-part reader; each chunk it reads has its leading `partNumber`
    /// verified against `n`. A **tiled** part is served as scanlines, exactly
    /// as C++ `InputPart` does (via the persisted tiled-as-scanlines
    /// [`TiledReaderState`], including its tile-row cache — no longer
    /// silently discarded and rebuilt every call).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when `n` is out of range.
    /// * Propagates the reader-state construction error (including the
    ///   `Unsupported::Deep` a deep part still yields).
    pub fn input_part(&mut self, n: i32) -> Result<InputFilePartReader<'_>> {
        let idx = self.part_index(n)?;
        let complete = self.parts[idx].complete;
        if self.parts[idx].input_reader_state.is_none() {
            let header = self.parts[idx].header.clone();
            let storage = self.parts[idx].storage;
            let offsets = self.parts[idx].chunk_offsets.clone();
            let version = self.version;
            let part_number = self.multipart.then_some(n);
            let state = match storage {
                Storage::Scanline => {
                    PartReaderState::ScanLine(Box::new(ScanLineInputFileState::from_part_mt(
                        header,
                        version,
                        part_number,
                        offsets,
                        self.num_threads,
                    )?))
                }
                Storage::Tiled => PartReaderState::Tiled(Box::new(TiledReaderState::from_part_mt(
                    header,
                    version,
                    part_number,
                    offsets,
                    self.num_threads,
                )?)),
                // C++ `InputFile::initialize`'s deep branch: mirrors
                // `scanline::input_file::InputFile::build`'s `Reader::Deep`
                // variant — a deep-SCANLINE part is
                // flattened through `CompositeDeepScanLine`, persisted here
                // as a `PartReaderState::Deep` the same way `ScanLine`/
                // `Tiled` are. Deep-TILED stays `Unsupported::Deep`: C++'s
                // own `initialize()` groups it with plain tiled storage and
                // never adds a compositor source for it either (see
                // `InputFile::build`'s `Storage::DeepTiled` arm for the full
                // "this mirrors real, not merely deferred, C++ behaviour"
                // citation).
                Storage::DeepScanline => PartReaderState::Deep(Box::new(
                    DeepScanLineReaderState::from_part_mt(
                        header,
                        version,
                        part_number,
                        offsets,
                        self.num_threads,
                    )?,
                )),
                Storage::DeepTiled => {
                    return Err(ExrError::Unsupported(Unsupported::Deep));
                }
                Storage::Unknown => {
                    return Err(ExrError::Arg(
                        "Unable to handle data storage type in file.".into(),
                    ));
                }
            };
            self.parts[idx].input_reader_state = Some(state);
        }
        let inner = self.parts[idx].input_reader_state.as_mut().ok_or_else(|| {
            ExrError::Logic("input reader state missing right after construction".into())
        })?;
        Ok(InputFilePartReader::new(&mut *self.stream, inner, complete))
    }

    /// Open the **tiled** part reader for part `n` — C++
    /// `MultiPartInputFile::getInputPart<TiledInputFile>(int)`.
    ///
    /// Returns a [`TiledInputPartReader`] — a borrowing view over this part's
    /// PERSISTED tiled reader state (built once,
    /// on first access, and kept across every later call — a fresh
    /// `tiled_input_part(0)` no longer rebuilds the whole read state, matching
    /// `MultiPartOutputFile::tiled_output_part`'s write-side caching).  Each
    /// tile chunk's leading `partNumber` is verified against `n`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `tiledimage` part; propagates the [`TiledInputFileState::from_part`] error.
    pub fn tiled_input_part(&mut self, n: i32) -> Result<TiledInputPartReader<'_>> {
        let idx = self.part_typed_index(n, header::TILED_IMAGE)?;
        if self.parts[idx].tiled_reader_state.is_none() {
            let header = self.parts[idx].header.clone();
            let offsets = self.parts[idx].chunk_offsets.clone();
            let version = self.version;
            let part_number = self.multipart.then_some(n);
            self.parts[idx].tiled_reader_state = Some(TiledInputFileState::from_part_mt(
                header,
                version,
                part_number,
                offsets,
                self.num_threads,
            )?);
        }
        let state = self.parts[idx].tiled_reader_state.as_mut().ok_or_else(|| {
            ExrError::Logic("tiled reader state missing right after construction".into())
        })?;
        Ok(TiledInputPartReader::new(&mut *self.stream, state))
    }

    /// Open the **deep scanline** part reader for part `n` — C++
    /// `MultiPartInputFile::getInputPart<DeepScanLineInputFile>(int)`.
    ///
    /// Returns a [`DeepScanLineInputPartReader`] — a borrowing view over this
    /// part's PERSISTED deep-scanline reader state (a
    /// `DeepScanLineInputFileState`
    /// view): built once, on first access, and kept across every later call.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `deepscanline` part; propagates the
    /// [`DeepScanLineInputFileState::from_part`] error.
    pub fn deep_scanline_input_part(&mut self, n: i32) -> Result<DeepScanLineInputPartReader<'_>> {
        let idx = self.part_typed_index(n, header::DEEP_SCANLINE)?;
        if self.parts[idx].deep_scanline_reader_state.is_none() {
            let header = self.parts[idx].header.clone();
            let offsets = self.parts[idx].chunk_offsets.clone();
            let version = self.version;
            let part_number = self.multipart.then_some(n);
            self.parts[idx].deep_scanline_reader_state = Some(
                DeepScanLineInputFileState::from_part_mt(
                    header,
                    version,
                    part_number,
                    offsets,
                    self.num_threads,
                )?,
            );
        }
        let state = self.parts[idx].deep_scanline_reader_state.as_mut().ok_or_else(|| {
            ExrError::Logic("deep scanline reader state missing right after construction".into())
        })?;
        Ok(DeepScanLineInputPartReader::new(&mut *self.stream, state))
    }

    /// Open the **deep tiled** part reader for part `n` — C++
    /// `MultiPartInputFile::getInputPart<DeepTiledInputFile>(int)`.
    ///
    /// Returns a [`DeepTiledInputPartReader`] — a borrowing view over this
    /// part's PERSISTED deep-tiled reader state (backed by
    /// `DeepTiledInputFileState`): built once, on
    /// first access, and kept across every later call.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `deeptile` part; propagates the [`DeepTiledInputFileState::from_part`] error.
    pub fn deep_tiled_input_part(&mut self, n: i32) -> Result<DeepTiledInputPartReader<'_>> {
        let idx = self.part_typed_index(n, header::DEEP_TILE)?;
        if self.parts[idx].deep_tiled_reader_state.is_none() {
            let header = self.parts[idx].header.clone();
            let offsets = self.parts[idx].chunk_offsets.clone();
            let version = self.version;
            let part_number = self.multipart.then_some(n);
            self.parts[idx].deep_tiled_reader_state = Some(DeepTiledInputFileState::from_part_mt(
                header,
                version,
                part_number,
                offsets,
                self.num_threads,
            )?);
        }
        let state = self.parts[idx].deep_tiled_reader_state.as_mut().ok_or_else(|| {
            ExrError::Logic("deep tiled reader state missing right after construction".into())
        })?;
        Ok(DeepTiledInputPartReader::new(&mut *self.stream, state))
    }

    /// Open the **scanline** part as an [`InputPart`] wrapper (port of
    /// `Imf::InputPart`) — the thin-adapter form of
    /// [`input_part`](Self::input_part).
    pub fn input_part_wrapper(&mut self, n: i32) -> Result<InputPart<'_>> {
        Ok(InputPart::new(self.input_part(n)?))
    }

    /// Open the **tiled** part as a [`TiledInputPart`] wrapper (port of
    /// `Imf::TiledInputPart`).
    pub fn tiled_input_part_wrapper(&mut self, n: i32) -> Result<TiledInputPart<'_>> {
        Ok(TiledInputPart::new(self.tiled_input_part(n)?))
    }

    /// Open the **deep scanline** part as a [`DeepScanLineInputPart`] wrapper
    /// (port of `Imf::DeepScanLineInputPart`).
    pub fn deep_scanline_input_part_wrapper(
        &mut self,
        n: i32,
    ) -> Result<DeepScanLineInputPart<'_>> {
        Ok(DeepScanLineInputPart::new(self.deep_scanline_input_part(n)?))
    }

    /// Open the **deep tiled** part as a [`DeepTiledInputPart`] wrapper (port of
    /// `Imf::DeepTiledInputPart`).
    pub fn deep_tiled_input_part_wrapper(&mut self, n: i32) -> Result<DeepTiledInputPart<'_>> {
        Ok(DeepTiledInputPart::new(self.deep_tiled_input_part(n)?))
    }

    /// Bounds-checked + type-checked part accessor for the typed part readers.
    /// Returns clones of part `n`'s header and chunk offsets after confirming
    /// its header `type` equals `want_type` (the C++ part-type dispatch in
    /// `getInputPart`).
    fn part_typed(&self, n: i32, want_type: &str) -> Result<(Header, Vec<u64>)> {
        let part = self.part(n)?;
        let t = if part.header.has_type() {
            part.header.type_()?
        } else {
            ""
        };
        if t != want_type {
            return Err(ExrError::Arg(format!(
                "input part {n} has type \"{t}\", but a \"{want_type}\" part reader was requested."
            )));
        }
        Ok((part.header.clone(), part.chunk_offsets.clone()))
    }

    /// Bounds-checked part accessor — C++ `getPart(int)`'s range check.
    fn part(&self, n: i32) -> Result<&PartData> {
        if n < 0 || n as usize >= self.parts.len() {
            return Err(ExrError::Arg(format!(
                "MultiPartInputFile::getPart called with invalid part {n} \
                 on file with {} parts",
                self.parts.len()
            )));
        }
        Ok(&self.parts[n as usize])
    }

    /// Bounds-checked part index — same check as [`part`](Self::part), but
    /// returns the index instead of a borrow, so a caller can immediately
    /// follow up with a MUTABLE `&mut self.parts[idx]` access (per-part state caching)
    /// without holding this method's `&self` borrow alive across it.
    fn part_index(&self, n: i32) -> Result<usize> {
        self.part(n)?;
        Ok(n as usize)
    }

    /// Bounds- and type-checked part index — the index-returning sibling of
    /// [`part_typed`](Self::part_typed), for the same reason as
    /// [`part_index`](Self::part_index).
    fn part_typed_index(&self, n: i32, want_type: &str) -> Result<usize> {
        self.part_typed(n, want_type)?;
        Ok(n as usize)
    }
}
