//! The **one** checked chunk-offset-table reader + reconstruction pass, shared
//! by every reader in this crate (scanline, tiled, deep scanline, deep tiled,
//! multipart).
//!
//! Every EXR part stores, right after the header(s), a table of `u64` file
//! offsets — one per chunk — that lets a reader seek straight to any chunk.
//! The table's entry count is *derived from the header*, and the offsets
//! themselves come straight off the wire, so both are attacker-controlled:
//! before this module existed each reader did its own
//! `vec![0u64; count]` + `read_u64` loop with no bound at all, and a 293-byte
//! crafted file drove a 16.4 GB allocation.
//!
//! What is enforced here (all of it ported from C++, cited per check):
//!
//! * a hard cap of 2^20 entries **and** a file-size bound on the table itself —
//!   `OpenEXRCore/chunk.c::extract_chunk_table`
//!   (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.13/src/lib/OpenEXRCore/chunk.c>,
//!   the `part->chunk_count > (1024 * 1024) || chunkbytes + chunkoff > file_size`
//!   test), so nothing is ever allocated from an unvalidated stream value;
//! * per-offset validation against `[end-of-table, file_size)` — same function,
//!   the `if (cchunk < chunkoff || cchunk >= maxoff) complete = 0;` loop.  A
//!   `0` entry is *missing*, never a valid offset, and therefore also marks the
//!   table incomplete (C++ `TileOffsets::anyOffsetsAreInvalid`, and the legacy
//!   `ImfScanLineInputFile.cpp::readLineOffsets` `lineOffsets[i] <= 0` test);
//! * reconstruction of an incomplete table by scanning the chunks —
//!   `OpenEXRCore/chunk.c::reconstruct_chunk_table` +
//!   `read_and_validate_chunk_leader`, which is the same algorithm the legacy
//!   `ImfScanLineInputFile.cpp::reconstructLineOffsets` /
//!   `ImfTileOffsets.cpp::findTiles` implement for their own chunk shape.
//!
//! The reconstruction sweep is written once and parameterised by [`Framing`],
//! so a scanline, deep-scanline, tiled or deep-tiled part — single-part or
//! multipart — all go through the same code.

use crate::attr::LineOrder;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::IStream;
use crate::tiled::tile_offsets::TileOffsets;

/// On-disk size of one offset-table entry (`u64` LE).
pub(crate) const OFFSET_BYTES: u64 = 8;

/// Hard upper bound on the number of chunk-offset-table entries.
///
/// C++ `OpenEXRCore/chunk.c::extract_chunk_table`: *"some of the stream-based
/// objects can't reliably check the file size so the C++ layer also had an
/// arbitrary stop at 2^20 chunk entries which seems safe"*.
pub const MAX_CHUNK_COUNT: i64 = 1024 * 1024;

/// A part's chunk offset table as read from the file.
pub struct OffsetTable {
    /// One file offset per chunk, in slot order (`0` = the chunk was never
    /// written).
    pub offsets: Vec<u64>,
    /// `false` when at least one entry is missing (`0`) or points outside
    /// `[end-of-table, file_size)` — the file is incomplete/corrupt and the
    /// table should be reconstructed (C++ `complete = 0`).
    pub complete: bool,
}

/// Validate a header-derived chunk count before it is used to size an
/// allocation — port of the `extract_chunk_table` guard (`chunk.c`).
///
/// `table_pos` is the file position the table starts at and `stream_size` the
/// stream length when it is known (`IStream::size()`; C++ `ctxt->file_size > 0`).
pub fn check_chunk_count(
    count: i64,
    table_pos: u64,
    stream_size: Option<u64>,
) -> Result<usize> {
    // C++: `if (part->chunk_count <= 0) … "Invalid file with no chunks"`.
    if count <= 0 {
        return Err(ExrError::Input(format!(
            "Invalid file with no chunks (chunk count {count})."
        )));
    }
    if count > MAX_CHUNK_COUNT {
        return Err(ExrError::Input(format!(
            "Chunk offset table of {count} entries exceeds the maximum of \
             {MAX_CHUNK_COUNT} entries."
        )));
    }
    // C++: `chunkbytes + chunkoff > (uint64_t) ctxt->file_size` → hard error.
    let table_bytes = count as u64 * OFFSET_BYTES;
    if let Some(size) = stream_size
        && table_pos.saturating_add(table_bytes) > size
    {
        return Err(ExrError::Input(format!(
            "Chunk offset table size ({table_bytes} bytes at offset {table_pos}) \
             is too big for the file size ({size})."
        )));
    }
    usize::try_from(count)
        .map_err(|_| ExrError::Input("Chunk count out of range".into()))
}

/// Read one part's chunk offset table from the current stream position — the
/// single checked reader every EXR reader in this crate uses.
///
/// `count` is the header-derived chunk count; it is validated by
/// [`check_chunk_count`] *before* anything is allocated.  Each entry is then
/// range-checked against `[end-of-table, file_size)`; a violation (including a
/// `0` entry) only sets [`OffsetTable::complete`] to `false` — exactly as C++
/// does, so that reconstruction, not a hard failure, handles an interrupted
/// write.
pub fn read_offset_table(is: &mut dyn IStream, count: i64) -> Result<OffsetTable> {
    let table_pos = is.tellg()?;
    let stream_size = is.size();
    let n = check_chunk_count(count, table_pos, stream_size)?;

    // C++ `extract_chunk_table`: after the table has been read, `chunkoff`
    // points just past it, and every offset below that (or at/after EOF) marks
    // the table incomplete.
    let min_offset = table_pos + n as u64 * OFFSET_BYTES;
    let max_offset = stream_size.unwrap_or(u64::MAX);

    // Fallible, not `vec![..]` -- `n` is
    // already capped at `MAX_CHUNK_COUNT` (8 MB worst case) so this is
    // defense-in-depth rather than the primary fix, but the same
    // `handle_alloc_error`-aborts-the-process concern applies to any
    // allocation sized from a header-derived count (see
    // `misc::try_filled_vec`'s docs).
    let mut offsets = crate::misc::try_filled_vec(n, 0u64)?;
    for off in offsets.iter_mut() {
        *off = xdr::read_u64(is)?;
    }
    let complete = table_complete(&offsets, min_offset, Some(max_offset));

    Ok(OffsetTable { offsets, complete })
}

/// Lenient offset-table read for `multipart::input::read_prologue`:
/// unlike
/// [`read_offset_table`] above (used by [`crate::chunk::RawImage`], which
/// stays strict on purpose — a deliberately-stricter tool, not a C++ port),
/// a READER CONSTRUCTOR must never hard-fail merely because a chunk-offset
/// table is corrupt, oversized, or truncated. C++ defers every one of
/// `extract_chunk_table`'s validation failures (`chunk_count <= 0`, `>
/// 2^20`, "table too big for file", or an outright truncated read) to
/// `Context::chunkTableValid`, which swallows ALL of them into a plain
/// `isComplete() == false` (`ImfContext.cpp::chunkTableValid`) — the file
/// still opens, and the real, propagated error only ever surfaces from an
/// actual chunk READ (`readPixels`/`readTile(s)`).
///
/// This function reproduces that: it returns `Err` ONLY for a genuine I/O
/// failure on the seek itself (a real disk error) or the (essentially
/// unreachable, since `n` is already capped at `MAX_CHUNK_COUNT` = 8 MB
/// worst case) allocation failure — NEVER for a table-content validation
/// failure. A validation failure yields an EMPTY, `complete: false` table
/// instead: every later read against it fails its own bounds check (a
/// clean `Err`, matching "the real error surfaces at read time"), and
/// `is_complete()` reports `false` without ever having thrown.
///
/// `table_pos` is the ANALYTICALLY-computed position this part's table
/// starts at — the caller must compute every part's table position up
/// front, purely from the header-derived `chunk_count`s (mirrors C++
/// `parse_header.c`'s `chunk_table_offset` computation, which runs before
/// ANY table is actually read), so that a corrupt or undersized table in
/// part `i` never displaces where part `i+1`'s table is looked for.
///
/// **A truncated (short) read is treated the SAME as a size-validation
/// failure — empty table, no reconstruction attempted**:
/// `chunk.c::extract_chunk_table` only enters its
/// per-entry validity check (the loop that can trigger
/// `reconstruct_chunk_table`) in the `else` branch of `if (rv !=
/// EXR_ERR_SUCCESS) { ... UINTPTR_MAX ... } else if (!disable_reconstruct)
/// { ... }` — i.e. ONLY once the FULL `chunkbytes`-sized read
/// (`MUST_READ_ALL`) has already succeeded. A short read takes the `if`
/// branch instead: the table is marked permanently unusable and
/// reconstruction is never attempted for it. Getting this wrong (treating a
/// truncated read like "some entries individually invalid, but the read
/// itself worked") would make our reconstruction sweep SUCCEED in a case
/// C++'s never even tries, and therefore READS in a case C++'s `readPixels`
/// fails — an observable divergence, not just extra work.
pub(crate) fn read_offset_table_lenient(
    is: &mut dyn IStream,
    count: i64,
    table_pos: u64,
) -> Result<OffsetTable> {
    let stream_size = is.size();
    let n = match check_chunk_count(count, table_pos, stream_size) {
        Ok(n) => n,
        // Never blocks opening the file -- this part is simply unusable
        // (empty table) until actually read, exactly like the C++ failure
        // modes `chunkTableValid` swallows into `false`.
        Err(_) => return Ok(OffsetTable { offsets: Vec::new(), complete: false }),
    };

    is.seekg(table_pos)?;

    let min_offset = table_pos + n as u64 * OFFSET_BYTES;
    let max_offset = stream_size.unwrap_or(u64::MAX);

    let mut offsets = crate::misc::try_filled_vec(n, 0u64)?;
    for off in offsets.iter_mut() {
        match xdr::read_u64(is) {
            Ok(v) => *off = v,
            // A short read is `rv != EXR_ERR_SUCCESS` in C++ -- the table
            // becomes permanently unusable (`UINTPTR_MAX`), never handed to
            // `reconstruct_chunk_table`. Discard whatever WAS read (unlike
            // the strict `read_offset_table`, which never reaches this
            // state at all -- its `?` already turned a short read into a
            // hard `Err` for its own, deliberately-stricter callers) and
            // return empty, exactly like a size-validation failure above.
            Err(_) => return Ok(OffsetTable { offsets: Vec::new(), complete: false }),
        }
    }
    let complete = table_complete(&offsets, min_offset, Some(max_offset));

    Ok(OffsetTable { offsets, complete })
}

/// Is every entry of `offsets` a usable chunk offset?
///
/// The validity rule of `OpenEXRCore/chunk.c::extract_chunk_table` /
/// `part.c::exr_validate_chunk_table`: an offset must be `>= chunkmin` (the
/// first byte past the offset table) and `< file_size`.  A `0` entry therefore
/// fails, because a chunk can never start inside the table.
///
/// Used twice: once on the table as read from disk, and again **after**
/// reconstruction — `exr_validate_chunk_table` (which is what
/// `MultiPartInputFile::partComplete` / `InputFile::isComplete` call) runs
/// `extract_chunk_table`, i.e. reconstruction, and only then re-checks every
/// entry.  A file whose table was rebuilt successfully is therefore *complete*.
pub fn table_complete(offsets: &[u64], min_offset: u64, stream_size: Option<u64>) -> bool {
    let max_offset = stream_size.unwrap_or(u64::MAX);
    offsets
        .iter()
        .all(|&v| v >= min_offset && v < max_offset)
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// The on-disk chunk framing of one part: how to parse a chunk leader and how
/// to map it to an offset-table slot.
///
/// Mirrors the four leader shapes in `OpenEXRCore/chunk.c::extract_chunk_leader`
/// (the optional `partNumber` prefix is handled by the sweep itself, since it
/// is a property of the *file*, not of the part type).
pub enum Framing {
    /// Flat scanline: `y (i32)`, `dataSize (i32)`, payload.
    Scanline {
        /// Data-window `min.y` (the scanline the first chunk starts at).
        min_y: i32,
        /// Scanlines per chunk (C++ `lines_per_chunk`).
        lines_per_chunk: i32,
        /// The header's `lineOrder` — reconstruction expects
        /// chunk `ci` (0-based table slot) at file position `ci` for
        /// `IncreasingY`/`RandomY`, or `chunk_count - (ci + 1)` for
        /// `DecreasingY` (`chunk.c::reconstruct_chunk_table`'s
        /// `computed_ci` flip).
        line_order: LineOrder,
    },
    /// Deep scanline: `y (i32)`, `packedCountSize`, `packedDataSize`,
    /// `unpackedDataSize` (3× `u64`), payload.
    DeepScanline {
        /// Data-window `min.y`.
        min_y: i32,
        /// Scanlines per chunk (always 1 for the deep-capable codecs).
        lines_per_chunk: i32,
        /// See [`Framing::Scanline`]'s `line_order`.
        line_order: LineOrder,
    },
    /// Flat tile: `dx, dy, lx, ly (4× i32)`, `dataSize (i32)`, payload.  The
    /// [`TileOffsets`] carries only the level geometry — it is used as the
    /// slot oracle (`is_valid_tile` + `slot_index`).
    Tile(TileOffsets),
    /// Deep tile: `dx, dy, lx, ly (4× i32)`, 3× `u64` sizes, payload.
    DeepTile(TileOffsets),
}

/// One part taking part in a reconstruction sweep: its framing plus the flat
/// offset table (in slot order) to fill in.
pub struct PartOffsets<'a> {
    /// How this part's chunks are framed on disk.
    pub framing: Framing,
    /// The part's offset table, in slot order; reconstructed entries are
    /// written here.
    pub offsets: &'a mut Vec<u64>,
}

/// Rebuild incomplete chunk offset tables by scanning the file's chunk region —
/// port of `OpenEXRCore/chunk.c::reconstruct_chunk_table`.
///
/// C++ reconstructs ONE PART AT A TIME, in part
/// order, each seeded from the end of the PREVIOUS part's own last actual
/// chunk (`:444-462`) — not one interleaved sweep across every part at once.
/// Within a part, for every table SLOT (not file-order chunk) it re-anchors
/// the read position on that slot's OWN stored table entry whenever the
/// entry looks plausible (`>= offset_start && < file_size`, `:472-475`), and
/// does NOT stop the whole part at the first bad chunk — every slot still
/// gets its own attempt (`:470-494`); only a genuinely truncated/corrupt tail
/// leaves the remaining slots unrecovered. Both differences matter for a
/// realistic partial-corruption case: a table with ONE bad entry in the
/// middle used to lose every slot after it; a later part's table used to be
/// reconstructed from `first_chunk_pos` (the file's very first byte of chunk
/// data) instead of from where its own data plausibly starts.
///
/// `multipart` selects whether each chunk carries the leading `partNumber`
/// that must match the part currently being reconstructed (rejected
/// otherwise, `chunk.c:261-269`).
///
/// Errors are **swallowed** at the level C++ swallows them: a bad/truncated
/// chunk fails that ONE slot (recorded as still-missing) without aborting
/// the rest of the part's sweep, and failing to locate the previous part's
/// last chunk (so this part's own seed position cannot be determined) leaves
/// this part — and, since each part depends on the one before, every part
/// after it — unreconstructed rather than erroring the whole file open
/// (matches this module's existing "never abort a whole file over a
/// truncated tail" contract, `read_offset_table`/`read_offset_table_lenient`).
/// The stream is always restored to where it was.
pub fn reconstruct(
    is: &mut dyn IStream,
    parts: &mut [PartOffsets<'_>],
    multipart: bool,
    first_chunk_pos: u64,
) -> Result<()> {
    let saved = is.tellg()?;
    let max_offset = is.size();

    let mut offset_start = first_chunk_pos;
    for part_index in 0..parts.len() {
        reconstruct_one_part(is, parts, multipart, part_index, offset_start, max_offset);

        // Seed the NEXT part from the end of THIS part's own last actual
        // chunk (`chunk.c:452-461`) — not attempted for the last part, since
        // there is no next part to seed. A part whose last-chunk position
        // can't be pinned down (empty/all-zero table, or that chunk's own
        // leader doesn't parse) leaves every later part's `offset_start`
        // unresolved; explicitly stop rather than guessing `first_chunk_pos`
        // again, which would silently reintroduce the old "all parts share
        // one interleaved floor" behaviour this fix removes.
        if part_index + 1 < parts.len() {
            match last_chunk_end(is, parts, multipart, part_index, first_chunk_pos, max_offset) {
                Some(next) => offset_start = next,
                None => break,
            }
        }
    }

    is.clear();
    is.seekg(saved)?;

    Ok(())
}

/// Reconstruct one part's table in place, seeded at `offset_start` — the
/// per-part body of `chunk.c::reconstruct_chunk_table`'s `for (ci ...)` loop
/// (`:470-494`) plus its commit step (`:495-506`).
fn reconstruct_one_part(
    is: &mut dyn IStream,
    parts: &mut [PartOffsets<'_>],
    multipart: bool,
    part_index: usize,
    mut offset_start: u64,
    max_offset: Option<u64>,
) {
    let max = max_offset.unwrap_or(u64::MAX);
    let Some(part) = parts.get(part_index) else { return };
    let chunk_count = part.offsets.len();
    let original: Vec<u64> = part.offsets.clone();
    let decreasing = matches!(
        part.framing,
        Framing::Scanline { line_order: LineOrder::DecreasingY, .. }
            | Framing::DeepScanline { line_order: LineOrder::DecreasingY, .. }
    );
    let is_scanline_like =
        matches!(part.framing, Framing::Scanline { .. } | Framing::DeepScanline { .. });

    let mut found = vec![0u64; chunk_count];
    let mut any_failure = false;

    for ci in 0..chunk_count {
        // Re-anchor on this SLOT's own stored entry if it looks plausible —
        // `chunk.c:472-475`.
        if let Some(&v) = original.get(ci)
            && v >= offset_start
            && v < max
        {
            offset_start = v;
        }
        let chunk_start = offset_start;
        let computed_ci = if decreasing { chunk_count - (ci + 1) } else { ci };

        match parse_chunk_leader(is, parts, multipart, part_index, chunk_start, max_offset) {
            Ok((leader_slot, next)) => {
                offset_start = next;
                // Scanline/deep-scanline chunks must appear at the slot the
                // ITERATION expects (`chunk.c:384`'s `*indexio != chunk`
                // check); tile/deep-tile chunks carry their own address and
                // are accepted wherever they land ("just go with it if the
                // coordinates look sane", `chunk.c:399`).
                if is_scanline_like && leader_slot != computed_ci {
                    any_failure = true;
                }
                if leader_slot < chunk_count && found[leader_slot] == 0 {
                    found[leader_slot] = chunk_start;
                }
            }
            Err(_) => {
                // Structural failure (short read, bad size, foreign part
                // number, EOF): this slot stays unrecovered and `offset_start`
                // does NOT advance — matches C++ (`extract_chunk_leader`
                // returns before writing `*next_offset` on a hard failure),
                // so the NEXT iteration's own re-anchor check above is the
                // only way forward.
                any_failure = true;
            }
        }
    }

    // Commit — `chunk.c:495-506`: a part with ZERO failures gets its table
    // fully replaced by the fresh sweep; a part with any failure only has
    // its POSITIVELY located slots merged in, leaving the rest as they were.
    let Some(part) = parts.get_mut(part_index) else { return };
    if !any_failure {
        *part.offsets = found;
    } else {
        for (slot, &v) in found.iter().enumerate() {
            if v != 0 && slot < part.offsets.len() {
                part.offsets[slot] = v;
            }
        }
    }
}

/// The file offset immediately after `parts[part_index]`'s own LAST actual
/// chunk (by file position, among whatever entries its table now holds) —
/// port of `chunk.c:448-461`'s "find the previous part's own last chunk,
/// then read its size" seed computation for the NEXT part's reconstruction.
/// `None` when no plausible chunk of this part can be pinned down at all
/// (an all-zero/empty table, or that chunk's own leader doesn't parse).
fn last_chunk_end(
    is: &mut dyn IStream,
    parts: &[PartOffsets<'_>],
    multipart: bool,
    part_index: usize,
    floor: u64,
    max_offset: Option<u64>,
) -> Option<u64> {
    let max = max_offset.unwrap_or(u64::MAX);
    let part = parts.get(part_index)?;

    let mut chunk_start = floor;
    for &v in part.offsets.iter() {
        if v > chunk_start && v < max {
            chunk_start = v;
        }
    }
    if chunk_start == floor {
        return None;
    }

    parse_chunk_leader(is, parts, multipart, part_index, chunk_start, max_offset)
        .ok()
        .map(|(_slot, next)| next)
}

/// Read one chunk leader at `pos`, which must belong to `expected_part`
/// (its `partNumber` prefix is checked against it when `multipart`) — port
/// of `chunk.c::extract_chunk_leader` (structural parse: leader fields plus
/// payload size) followed by the per-framing slot computation
/// `read_and_validate_chunk_leader` does for its storage mode, and returns
/// the slot the leader claims plus the file offset immediately after its
/// payload.
///
/// A scanline or deep-scanline leader's claimed slot is computed from `y`
/// alone and is NOT compared against an expected slot in here — that
/// strictness is the caller's job, matching the C++ split between structural
/// parsing and the scanline-specific index check. A tile or deep-tile
/// leader's claimed slot is computed from its coordinates and validated as
/// addressing a real tile.
fn parse_chunk_leader(
    is: &mut dyn IStream,
    parts: &[PartOffsets<'_>],
    multipart: bool,
    expected_part: usize,
    pos: u64,
    max_offset: Option<u64>,
) -> Result<(usize, u64)> {
    if let Some(limit) = max_offset
        && pos >= limit
    {
        return Err(ExrError::Input("Chunk scan reached the end of the file".into()));
    }
    is.seekg(pos)?;

    // C++ `extract_chunk_leader`: the multipart `partNumber` prefix must
    // match the part this leader is being read on behalf of (`chunk.c:261-269`
    // — a foreign part number is rejected, not silently routed).
    if multipart {
        let pn = xdr::read_i32(is)?;
        let pn = usize::try_from(pn)
            .map_err(|_| ExrError::Input(format!("Invalid chunk part number {pn}")))?;
        if pn != expected_part {
            return Err(ExrError::Input(format!(
                "Invalid part number reconstructing chunk table: expected {expected_part}, \
                 found {pn}."
            )));
        }
    }
    let part = parts
        .get(expected_part)
        .ok_or_else(|| ExrError::Input(format!("Chunk names part {expected_part}, which does not exist")))?;

    let (slot, payload) = match &part.framing {
        Framing::Scanline {
            min_y,
            lines_per_chunk,
            ..
        } => {
            let y = xdr::read_i32(is)?;
            let data_size = xdr::read_i32(is)?;
            if data_size < 0 {
                // `ImfTileOffsets.cpp:142` (verified): a chunk-table
                // reconstruction failure is `IoExc`. Not observable from the
                // public API today (this sits inside a catch-all
                // reconstruct); aligned here for fidelity.
                return Err(ExrError::Io("Invalid chunk data size".into()));
            }
            (scanline_slot(y, *min_y, *lines_per_chunk)?, data_size as u64)
        }
        Framing::DeepScanline {
            min_y,
            lines_per_chunk,
            ..
        } => {
            let y = xdr::read_i32(is)?;
            let payload = deep_payload(is)?;
            (scanline_slot(y, *min_y, *lines_per_chunk)?, payload)
        }
        Framing::Tile(offsets) => {
            let (dx, dy, lx, ly) = read_tile_coords(is)?;
            let data_size = xdr::read_i32(is)?;
            if data_size < 0 {
                // `ImfTileOffsets.cpp:159` (verified) — see the
                // scanline arm above.
                return Err(ExrError::Io("Invalid tile data size".into()));
            }
            (tile_slot(offsets, dx, dy, lx, ly)?, data_size as u64)
        }
        Framing::DeepTile(offsets) => {
            let (dx, dy, lx, ly) = read_tile_coords(is)?;
            let payload = deep_payload(is)?;
            (tile_slot(offsets, dx, dy, lx, ly)?, payload)
        }
    };

    if slot >= part.offsets.len() {
        return Err(ExrError::Input("Chunk addresses a slot outside its part".into()));
    }

    let after_leader = is.tellg()?;
    let next = after_leader
        .checked_add(payload)
        .ok_or_else(|| ExrError::Input("Chunk payload size overflows the file".into()))?;
    if let Some(limit) = max_offset
        && next > limit
    {
        return Err(ExrError::Input("Chunk payload runs past the end of the file".into()));
    }

    Ok((slot, next))
}

/// `dx, dy, lx, ly` — the tile chunk leader's coordinates.
fn read_tile_coords(is: &mut dyn IStream) -> Result<(i32, i32, i32, i32)> {
    Ok((
        xdr::read_i32(is)?,
        xdr::read_i32(is)?,
        xdr::read_i32(is)?,
        xdr::read_i32(is)?,
    ))
}

/// Payload byte count of a deep chunk: `packedCountSize + packedDataSize`
/// (the third `u64`, `unpackedDataSize`, is not stored data) — mirrors the
/// bounds check in `ImfTileOffsets.cpp::findTiles`.
fn deep_payload(is: &mut dyn IStream) -> Result<u64> {
    let packed_count = xdr::read_u64(is)?;
    let packed_data = xdr::read_u64(is)?;
    let _unpacked = xdr::read_u64(is)?;
    packed_count
        .checked_add(packed_data)
        .ok_or_else(|| ExrError::Input("Invalid deep chunk size".into()))
}

/// Map a scanline chunk's stored `y` to its offset-table slot — port of the
/// scanline branch of `chunk.c::read_and_validate_chunk_leader`
/// (`chunk = (scanline_y - data_window.min.y) / lines_per_chunk`, rejected when
/// it does not address a real chunk).
///
/// This used to ALSO
/// reject a `y` that is not exactly at a block boundary
/// (`rel % lines_per_chunk != 0`). C++ never checks that: `chunk.c`'s
/// `read_and_validate_chunk_leader` computes the slot via TRUNCATING integer
/// division (C's `/`, truncation-toward-zero -- exactly what Rust's `/` on
/// signed integers already does, so no behavioural port work is needed for
/// the division itself) and rejects the leader ONLY when the computed slot
/// disagrees with the caller's EXPECTED slot, or is out of `[0,
/// chunk_count)`. A misaligned `y` can therefore only arise from a corrupt
/// or malicious leader (every real leader is written at a block boundary by
/// construction); it is still caught, just by the CALLER's slot-match check
/// (`chunk/mod.rs::read_part_chunks` / `scan_one_chunk` below) rather
/// than by this function refusing to even compute a candidate slot. Bounds
/// (`chunk < 0`) fall out of `usize::try_from` failing on a negative
/// division result; `chunk >= chunk_count` is left to the caller too (every
/// caller already bounds-checks the returned slot against a real, capped
/// table -- see this function's two call sites).
pub(crate) fn scanline_slot(y: i32, min_y: i32, lines_per_chunk: i32) -> Result<usize> {
    if lines_per_chunk <= 0 {
        return Err(ExrError::Logic("linesInBuffer must be positive".into()));
    }
    let rel = (y as i64) - (min_y as i64);
    usize::try_from(rel / lines_per_chunk as i64).map_err(|_| {
        ExrError::Input(format!(
            "Invalid chunk scanline {y} (data window min.y {min_y}, \
             {lines_per_chunk} lines per chunk)."
        ))
    })
}

/// Map a tile chunk's coordinates to its flat offset-table slot, rejecting
/// coordinates that address no tile — C++ `findTiles`' `isValidTile` guard.
pub(crate) fn tile_slot(offsets: &TileOffsets, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<usize> {
    if !offsets.is_valid_tile(dx, dy, lx, ly) {
        return Err(ExrError::Input(format!(
            "Invalid tile coordinates ({dx}, {dy}, {lx}, {ly}) in chunk leader."
        )));
    }
    offsets.slot_index(dx, dy, lx, ly)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::MemIStream;

    /// A stream whose `size()` is UNKNOWN (`None`, the `IStream` trait's own
    /// default) but whose actual bytes are finite — the only way to exercise
    /// a genuine short/truncated read past `check_chunk_count`'s "table too
    /// big for file" guard, which is a no-op whenever `stream_size` is
    /// `None` (mirrors C++ `ctxt->file_size > 0` being false for a stream
    /// that cannot report its own length, e.g. a non-seekable pipe).
    struct SizeUnknownStream {
        cursor: std::io::Cursor<Vec<u8>>,
    }

    impl IStream for SizeUnknownStream {
        fn file_name(&self) -> &str {
            "size-unknown-test-stream"
        }
        fn read(&mut self, buf: &mut [u8]) -> Result<bool> {
            use std::io::Read;
            self.cursor
                .read_exact(buf)
                .map_err(crate::error::ExrError::from_io)?;
            Ok(true)
        }
        fn tellg(&mut self) -> Result<u64> {
            Ok(self.cursor.position())
        }
        fn seekg(&mut self, pos: u64) -> Result<()> {
            self.cursor.set_position(pos);
            Ok(())
        }
        // `size()` intentionally left at the trait default (`None`).
    }

    /// Pins the exact C++ behaviour
    /// `chunk.c::extract_chunk_table` has for a table that is VALIDLY sized
    /// (passes `check_chunk_count`) but whose actual read is short/truncated
    /// -- `rv != EXR_ERR_SUCCESS` takes the branch that marks the table
    /// PERMANENTLY unusable (`UINTPTR_MAX`) WITHOUT ever calling
    /// `reconstruct_chunk_table`; only a read that fully succeeds but has
    /// individually-invalid entries reaches reconstruction. Getting this
    /// wrong (feeding the partially-read entries into reconstruction) would
    /// make this crate's reader SUCCEED reading a part C++'s `readPixels`
    /// fails on -- an observable divergence.
    #[test]
    fn read_offset_table_lenient_short_read_yields_empty_table_not_a_partial_one() {
        // Declares 4 entries (32 bytes) but the stream only has 10 bytes
        // total -- `check_chunk_count` cannot catch this because `size()`
        // is unknown (`None`), exactly like a non-seekable C++ stream.
        let mut stream = SizeUnknownStream {
            cursor: std::io::Cursor::new(vec![0u8; 10]),
        };
        let table = read_offset_table_lenient(&mut stream, 4, 0).expect("must not hard-error");
        assert!(
            table.offsets.is_empty(),
            "a short read must yield an EMPTY table (no reconstruction candidate), \
             not a partially-filled one"
        );
        assert!(!table.complete, "a short-read table can never be complete");
    }

    #[test]
    fn chunk_count_cap_rejects_absurd_tables() {
        // 2^20 + 1 entries: rejected even without a known stream size.
        assert!(check_chunk_count(MAX_CHUNK_COUNT + 1, 0, None).is_err());
        // Inside the cap but larger than the file: rejected (the 293-byte
        // crafted-file case — no allocation is attempted).
        assert!(check_chunk_count(2_000_000, 0, Some(293)).is_err());
        assert!(check_chunk_count(1000, 100, Some(293)).is_err());
        // Zero / negative counts are "no chunks", never a valid table.
        assert!(check_chunk_count(0, 0, Some(1024)).is_err());
        assert!(check_chunk_count(-1, 0, Some(1024)).is_err());
        // A table that fits is accepted.
        assert_eq!(check_chunk_count(4, 8, Some(1024)).unwrap(), 4);
    }

    /// Mutation-testing gap fix: the cap
    /// itself, `MAX_CHUNK_COUNT = 1024 * 1024`. A count just above `2048`
    /// (what `1024 + 1024` would give under a `*`→`+` mutant) but well under
    /// the real `2^20` cap must still be accepted.
    #[test]
    fn chunk_count_cap_is_the_real_1048576_not_2048() {
        assert_eq!(check_chunk_count(3000, 0, None).unwrap(), 3000);
    }

    /// Mutation-testing gap fix: `count > MAX_CHUNK_COUNT` (78:14) is a
    /// strict `>` — a count of EXACTLY `MAX_CHUNK_COUNT` must be accepted,
    /// which a `>=` mutant would wrongly reject.
    #[test]
    fn chunk_count_cap_accepts_the_boundary_value_itself() {
        assert_eq!(
            check_chunk_count(MAX_CHUNK_COUNT, 0, None).unwrap(),
            MAX_CHUNK_COUNT as usize
        );
    }

    /// Mutation-testing gap fix: `table_bytes = count as u64 * OFFSET_BYTES`
    /// (85:36) must be a real multiplication, not `count + OFFSET_BYTES`. With
    /// `count = 1000` the true table size is 8000 bytes, which does not fit a
    /// 5000-byte stream; the `+` mutant's "size" of `1008` would wrongly fit.
    #[test]
    fn check_chunk_count_table_bytes_is_count_times_offset_bytes() {
        assert!(
            check_chunk_count(1000, 0, Some(5000)).is_err(),
            "1000 entries * 8 bytes = 8000 bytes must not fit a 5000-byte stream"
        );
    }

    /// Mutation-testing gap fix: `table_pos.saturating_add(table_bytes) > size`
    /// (87:50) is a strict `>` — a table that ends EXACTLY at the stream size
    /// must be accepted, which a `>=` mutant would wrongly reject. 4 entries *
    /// 8 bytes = 32 bytes, starting at table_pos 8, ends exactly at 40.
    #[test]
    fn check_chunk_count_accepts_a_table_that_ends_exactly_at_stream_size() {
        assert_eq!(check_chunk_count(4, 8, Some(40)).unwrap(), 4);
    }

    #[test]
    fn read_offset_table_flags_out_of_range_entries() {
        // Two entries; the table occupies bytes 0..16 of a 64-byte stream.
        let mut data = vec![0u8; 64];
        data[0..8].copy_from_slice(&32u64.to_le_bytes()); // valid
        data[8..16].copy_from_slice(&8u64.to_le_bytes()); // inside the table → invalid
        let mut is = MemIStream::new(data);
        let t = read_offset_table(&mut is, 2).unwrap();
        assert!(!t.complete);
        assert_eq!(t.offsets, vec![32, 8]);

        // Both valid → complete.
        let mut data = vec![0u8; 64];
        data[0..8].copy_from_slice(&16u64.to_le_bytes());
        data[8..16].copy_from_slice(&32u64.to_le_bytes());
        let mut is = MemIStream::new(data);
        assert!(read_offset_table(&mut is, 2).unwrap().complete);
    }

    #[test]
    fn read_offset_table_zero_entry_is_missing_not_valid() {
        let mut data = vec![0u8; 64];
        data[0..8].copy_from_slice(&16u64.to_le_bytes());
        // entry 1 left as 0
        let mut is = MemIStream::new(data);
        let t = read_offset_table(&mut is, 2).unwrap();
        assert!(!t.complete);
        assert_eq!(t.offsets[1], 0);
    }

    /// Mutation-testing gap fix: `min_offset = table_pos + n as u64 *
    /// OFFSET_BYTES` (115:43) must be a real multiplication. With 2 entries
    /// (`n = 2`) the table really spans bytes `0..16`; a `+` mutant would
    /// compute `min_offset = 0 + 2 + 8 = 10`. An entry of `12` lies strictly
    /// between the two thresholds: it is still inside the table's real
    /// on-disk span (must be flagged invalid), but the `+` mutant's `10`
    /// threshold would wrongly accept it.
    #[test]
    fn read_offset_table_min_offset_is_the_table_span_not_table_pos_plus_count() {
        let mut data = vec![0u8; 64];
        data[0..8].copy_from_slice(&12u64.to_le_bytes());
        data[8..16].copy_from_slice(&32u64.to_le_bytes());
        let mut is = MemIStream::new(data);
        let t = read_offset_table(&mut is, 2).unwrap();
        assert!(
            !t.complete,
            "offset 12 lies inside the table's own on-disk span (0..16) and must be flagged invalid"
        );
    }

    /// Same gap as above but for [`read_offset_table_lenient`] (192:43),
    /// which has both a `*`→`+` and a `*`→`/` mutant on its own
    /// `min_offset` computation. `n / OFFSET_BYTES` (`2 / 8 == 0`) and `n +
    /// OFFSET_BYTES` (`2 + 8 == 10`) both wrongly accept the same `12`
    /// entry that the real `n * OFFSET_BYTES == 16` threshold must reject.
    #[test]
    fn read_offset_table_lenient_min_offset_is_the_table_span_not_table_pos_plus_or_div_count() {
        let mut data = vec![0u8; 64];
        data[0..8].copy_from_slice(&12u64.to_le_bytes());
        data[8..16].copy_from_slice(&32u64.to_le_bytes());
        let mut is = MemIStream::new(data);
        let t = read_offset_table_lenient(&mut is, 2, 0).unwrap();
        assert!(
            !t.complete,
            "offset 12 lies inside the table's own on-disk span (0..16) and must be flagged invalid"
        );
    }

    /// Mutation-testing gap fix: `read_tile_coords` reads 4 distinct `i32`
    /// fields in order (dx, dy, lx, ly) -- a single equality assertion
    /// against 4 easily-distinguishable, distinct values (including a
    /// negative one) kills every "replace the whole function with a
    /// constant tuple" mutant in one shot, since any constant substitution
    /// must disagree with at least one of these 4 values.
    #[test]
    fn read_tile_coords_reads_all_four_fields_in_order() {
        let mut data = Vec::new();
        data.extend_from_slice(&1_000_000i32.to_le_bytes());
        data.extend_from_slice(&2_000_000i32.to_le_bytes());
        data.extend_from_slice(&3_000_000i32.to_le_bytes());
        data.extend_from_slice(&(-4_000_000i32).to_le_bytes());
        let mut is = MemIStream::new(data);
        assert_eq!(
            read_tile_coords(&mut is).unwrap(),
            (1_000_000, 2_000_000, 3_000_000, -4_000_000)
        );
    }

    /// Updated from
    /// `scanline_slot_rejects_misaligned_and_negative` -- C++
    /// `read_and_validate_chunk_leader` does NOT reject a misaligned `y` by
    /// itself; it computes a slot via truncating division and leaves the
    /// "does this match what I expected" check to the caller. A
    /// misaligned `y` therefore now returns `Ok` with the TRUNCATED slot
    /// (`5 / 16 == 0`, matching C's `/`), not an `Err` -- misalignment is
    /// still caught, just one level up, by whichever caller compares the
    /// returned slot against the slot it expected (exercised by
    /// `chunk/mod.rs`'s `cross_linked_offset_table_slots_error_not_silently_duplicate`).
    /// Negative results (`y < min_y`) and a non-positive `lines_per_chunk`
    /// are still hard errors -- those can never be a valid slot regardless
    /// of caller expectations.
    #[test]
    fn scanline_slot_truncates_instead_of_requiring_alignment() {
        assert_eq!(scanline_slot(0, 0, 16).unwrap(), 0);
        assert_eq!(scanline_slot(32, 0, 16).unwrap(), 2);
        // Misaligned: truncating division still yields a slot, matching C++.
        assert_eq!(scanline_slot(5, 0, 16).unwrap(), 0);
        assert_eq!(scanline_slot(37, 0, 16).unwrap(), 2);
        // Still hard errors: negative slot, non-positive lines_per_chunk.
        assert!(scanline_slot(-1, 0, 1).is_err());
        assert!(scanline_slot(0, 0, 0).is_err());
    }

    /// Build one scanline chunk's on-disk bytes: `[partNumber?] y dataSize
    /// payload` (all fields LE `i32`), appended to `out`.
    fn push_scanline_chunk(out: &mut Vec<u8>, part_number: Option<i32>, y: i32, payload: &[u8]) {
        if let Some(pn) = part_number {
            out.extend_from_slice(&pn.to_le_bytes());
        }
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(payload);
    }

    /// A 2-part multipart
    /// file where part 0's MIDDLE chunk has been tampered (its `y` field
    /// overwritten with garbage, so its leader fails to parse) while its
    /// FIRST and LAST chunks are intact, and part 1's table is fully zeroed
    /// (needs a genuine from-scratch reconstruction). Both parts' tables are
    /// handed to [`reconstruct`] in one call, exactly as
    /// `multipart::input::reconstruct_all_parts` does.
    ///
    /// This is deliberately NOT recoverable with the OLD single interleaved
    /// sweep this fix replaced: that design shared ONE cursor and ONE
    /// abort-on-first-error flag across every part's chunks combined, so
    /// hitting part 0's corrupted middle chunk terminated the ENTIRE scan —
    /// part 1 (fully independent, and itself perfectly intact on disk) would
    /// never even be attempted, leaving its table all zero. The new
    /// per-part design (a) does not abort part 0's own sweep at its bad
    /// chunk (its LAST chunk is still recovered, via re-anchoring on that
    /// slot's own already-valid table entry to skip past the corrupted
    /// middle chunk), which in turn lets it (d) correctly seed part 1's
    /// reconstruction from the end of part 0's real last chunk, independent
    /// of part 0's internal damage.
    #[test]
    fn bad_middle_chunk_in_one_part_does_not_blind_reconstruction_of_a_later_part() {
        let mut bytes = Vec::new();
        // Part 0: 3 chunks, y = 0, 1 (CORRUPTED), 2 — all 4-byte payloads, so
        // every chunk is a fixed 16 bytes (4 partNumber + 4 y + 4 size + 4
        // payload) and every on-disk position is easy to hand-compute.
        push_scanline_chunk(&mut bytes, Some(0), 0, &[0xAA; 4]); // real chunk 0 @ 0..16
        push_scanline_chunk(&mut bytes, Some(0), 999_999, &[0x00; 4]); // corrupted "chunk 1" @ 16..32
        push_scanline_chunk(&mut bytes, Some(0), 2, &[0xCC; 4]); // real chunk 2 @ 32..48
        // Part 1: 2 intact chunks, y = 0, 1.
        push_scanline_chunk(&mut bytes, Some(1), 0, &[0xDD; 4]); // @ 48..64
        push_scanline_chunk(&mut bytes, Some(1), 1, &[0xEE; 4]); // @ 64..80
        assert_eq!(bytes.len(), 80);

        let mut is = MemIStream::new(bytes);

        // Part 0's table already holds the TRUE on-disk positions (a
        // corrupted-CONTENT chunk does not necessarily mean its own table
        // entry was corrupted too) — slot 1 points straight at the garbage
        // chunk.
        let mut part0_offsets: Vec<u64> = vec![0, 16, 32];
        // Part 1's table is fully zeroed — needs complete reconstruction.
        let mut part1_offsets: Vec<u64> = vec![0, 0];

        let part0_framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        let part1_framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };

        reconstruct(
            &mut is,
            &mut [
                PartOffsets { framing: part0_framing, offsets: &mut part0_offsets },
                PartOffsets { framing: part1_framing, offsets: &mut part1_offsets },
            ],
            true, // multipart
            0,    // first_chunk_pos
        )
        .expect("reconstruct itself only fails on a genuine stream I/O error");

        // Part 0: slot 1 (the corrupted chunk) cannot be recovered — its
        // content is genuinely unreadable — but slots 0 and 2 stay correct;
        // in particular slot 2 being intact is what proves the sweep did
        // NOT abort at slot 1.
        assert_eq!(part0_offsets, vec![0, 16, 32], "part 0's still-valid slots must survive");

        // Part 1: fully reconstructed from scratch, despite part 0's damage —
        // the per-part reconstruction property. The OLD interleaved-sweep design would leave
        // this at [0, 0] (never even attempted).
        assert_eq!(
            part1_offsets,
            vec![48, 64],
            "part 1 must be fully reconstructed independently of part 0's corruption"
        );
    }

    /// Mutation-testing gap fix (batch 3): `reconstruct_one_part`'s re-anchor
    /// guard (381:18, `v < max`) with a `>` mutant. Slot 1's stored entry
    /// (`20`) is a stale-but-in-range value (`v >= offset_start && v < max`),
    /// which the ported algorithm TRUSTS per C++'s own design
    /// (`chunk.c:472-475` does not special-case a plausible-but-wrong entry):
    /// slot 1 genuinely cannot be recovered from this exact input. A `>`
    /// mutant on `v < max` would instead skip the (wrong) re-anchor and, by
    /// accident, land on the correct sequential position.
    #[test]
    fn reconstruct_one_part_trusts_a_plausible_but_stale_re_anchor_entry() {
        let mut bytes = Vec::new();
        push_scanline_chunk(&mut bytes, None, 0, &[0xAA; 4]); // @ 0..12
        push_scanline_chunk(&mut bytes, None, 1, &[0xBB; 4]); // @ 12..24
        push_scanline_chunk(&mut bytes, None, 2, &[0xCC; 4]); // @ 24..36
        assert_eq!(bytes.len(), 36);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![0, 20, 24];
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        reconstruct_one_part(
            &mut is,
            &mut [PartOffsets { framing, offsets: &mut offsets }],
            false,
            0,
            0,
            Some(36),
        );

        assert_eq!(
            offsets,
            vec![0, 20, 24],
            "the re-anchor step trusts any in-range stored entry, even a stale one -- slot 1 is a \
             known-unrecoverable case for this exact input"
        );
    }

    /// Mutation-testing gap fix: the same 381:18 guard's `==`/`<=` mutants.
    /// Slot 1's stored entry (`36`) sits EXACTLY at `max` (one byte past
    /// EOF) -- never a real chunk position -- so it must be rejected, letting
    /// reconstruction fall through to the correct sequential walk. Accepting
    /// it sends `parse_chunk_leader` straight past EOF, which fails, and
    /// (since a failure switches the commit to merge-only-nonzero) leaves
    /// every slot at its original placeholder value.
    #[test]
    fn reconstruct_one_part_re_anchor_rejects_an_entry_exactly_at_max() {
        let mut bytes = Vec::new();
        push_scanline_chunk(&mut bytes, None, 0, &[0xAA; 4]); // @ 0..12
        push_scanline_chunk(&mut bytes, None, 1, &[0xBB; 4]); // @ 12..24
        push_scanline_chunk(&mut bytes, None, 2, &[0xCC; 4]); // @ 24..36
        assert_eq!(bytes.len(), 36);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![999, 36, 999];
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        reconstruct_one_part(
            &mut is,
            &mut [PartOffsets { framing, offsets: &mut offsets }],
            false,
            0,
            0,
            Some(36),
        );

        assert_eq!(
            offsets,
            vec![0, 12, 24],
            "an entry exactly at `max` (one-past-EOF) must be rejected, not accepted as plausible"
        );
    }

    /// Mutation-testing gap fix: `computed_ci = chunk_count - (ci + 1)`
    /// (386:55/386:61) for `DecreasingY` framing. File order here is
    /// high-y-to-low-y (chunk 0 @ pos 0 is `y=2` = table slot 2, chunk 1 @ 12
    /// is `y=1` = slot 1, chunk 2 @ 24 is `y=0` = slot 0). All table entries
    /// are out-of-range placeholders (`999 >= max`) so the re-anchor step
    /// never fires, isolating `computed_ci` as the only thing that can flag
    /// a (false) mismatch and switch the commit path from a full replace to
    /// merge-only-nonzero.
    #[test]
    fn reconstruct_one_part_decreasing_y_computed_slot_is_chunk_count_minus_ci_plus_one() {
        let mut bytes = Vec::new();
        push_scanline_chunk(&mut bytes, None, 2, &[0xAA; 4]); // @ 0..12 -> slot 2
        push_scanline_chunk(&mut bytes, None, 1, &[0xBB; 4]); // @ 12..24 -> slot 1
        push_scanline_chunk(&mut bytes, None, 0, &[0xCC; 4]); // @ 24..36 -> slot 0
        assert_eq!(bytes.len(), 36);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![999, 999, 999];
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::DecreasingY,
        };
        reconstruct_one_part(
            &mut is,
            &mut [PartOffsets { framing, offsets: &mut offsets }],
            false,
            0,
            0,
            Some(36),
        );

        assert_eq!(
            offsets,
            vec![24, 12, 0],
            "chunk_count - (ci + 1) must map file-order chunk ci to its DecreasingY table slot; a \
             wrong computed_ci falsely flags every chunk as misplaced and switches the commit path"
        );
    }

    /// Mutation-testing gap fix: the found-slot write guard's `&&` (399:46).
    /// Two chunks both claim the SAME slot (`y = 0` twice, a genuine
    /// corruption case) at different file positions; the real `&&` keeps
    /// only the FIRST occurrence (`found[slot] == 0` gates every write after
    /// the first), matching C++'s de-dup policy. A `||` mutant makes
    /// `leader_slot < chunk_count` (always true once `parse_chunk_leader`
    /// returns `Ok`) short-circuit the whole condition to `true`, so the
    /// SECOND (later, spurious) occurrence overwrites the first.
    #[test]
    fn reconstruct_one_part_keeps_the_first_occurrence_of_a_duplicate_slot() {
        let mut bytes = vec![0u8; 12]; // 12 bytes of filler before the real data
        push_scanline_chunk(&mut bytes, None, 0, &[0xAA; 4]); // @ 12..24 -> slot 0 (first)
        push_scanline_chunk(&mut bytes, None, 0, &[0xBB; 4]); // @ 24..36 -> slot 0 (duplicate)
        assert_eq!(bytes.len(), 36);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![999, 999];
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        reconstruct_one_part(
            &mut is,
            &mut [PartOffsets { framing, offsets: &mut offsets }],
            false,
            0,
            12,
            Some(36),
        );

        assert_eq!(
            offsets,
            vec![12, 999],
            "the first occurrence (file position 12) must win over the later duplicate (24)"
        );
    }

    /// Mutation-testing gap fix: `last_chunk_end`'s `v > chunk_start && v <
    /// max` (449:28/449:33). A stored offset exactly at `max` (one byte past
    /// EOF, never a real chunk position) must be rejected in favor of the
    /// largest offset strictly below it. Accepting it (a `||` or `<=`
    /// mutant) seeds the next part's reconstruction from an out-of-file
    /// position, which `parse_chunk_leader` immediately rejects.
    #[test]
    fn last_chunk_end_rejects_a_stored_offset_exactly_at_max() {
        let mut bytes = Vec::new();
        push_scanline_chunk(&mut bytes, None, 0, &[0xAA; 4]); // @ 0..12
        push_scanline_chunk(&mut bytes, None, 1, &[0xBB; 4]); // @ 12..24
        push_scanline_chunk(&mut bytes, None, 2, &[0xCC; 4]); // @ 24..36
        assert_eq!(bytes.len(), 36);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![12, 36]; // 36 == max: one-past-EOF, not a real position
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        let parts = [PartOffsets { framing, offsets: &mut offsets }];

        let next = last_chunk_end(&mut is, &parts, false, 0, 0, Some(36));
        assert_eq!(
            next,
            Some(24),
            "the offset exactly at max must be skipped in favor of the largest offset below it \
             (12, whose chunk ends at 24)"
        );
    }

    /// Mutation-testing gap fix: `parse_chunk_leader`'s scanline `data_size <
    /// 0` guard (517:26). A zero-byte payload is a legitimate empty chunk,
    /// not a negative-size corruption -- `<=`/`==` mutants would wrongly
    /// reject it.
    #[test]
    fn parse_chunk_leader_accepts_a_zero_byte_scanline_payload() {
        let mut bytes = Vec::new();
        push_scanline_chunk(&mut bytes, None, 0, &[]); // y=0, dataSize=0, no payload
        let mut is = MemIStream::new(bytes);
        let mut offsets: Vec<u64> = vec![0];
        let framing = Framing::Scanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        let parts = [PartOffsets { framing, offsets: &mut offsets }];
        let result = parse_chunk_leader(&mut is, &parts, false, 0, 0, None);
        assert_eq!(
            result.unwrap(),
            (0, 8),
            "dataSize == 0 is a legitimate empty chunk, not a negative-size error"
        );
    }

    /// Mutation-testing gap fix: `parse_chunk_leader`'s tile `data_size < 0`
    /// guard (538:26), same reasoning as the scanline case above but for the
    /// `Framing::Tile` arm.
    #[test]
    fn parse_chunk_leader_accepts_a_zero_byte_tile_payload() {
        use crate::attr::tile_description::LevelMode;

        let mut bytes = Vec::new();
        // dx, dy, lx, ly (all i32 LE), then dataSize = 0, no payload.
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes()); // dataSize = 0
        let mut is = MemIStream::new(bytes);

        let tile_offsets = TileOffsets::new(LevelMode::OneLevel, 1, 1, &[1], &[1]).unwrap();
        let mut offsets: Vec<u64> = vec![0];
        let framing = Framing::Tile(tile_offsets);
        let parts = [PartOffsets { framing, offsets: &mut offsets }];
        let result = parse_chunk_leader(&mut is, &parts, false, 0, 0, None);
        assert_eq!(
            result.unwrap(),
            (0, 20),
            "dataSize == 0 is a legitimate empty tile chunk, not a negative-size error"
        );
    }

    /// Mutation-testing gap fix: 538:26's `>` mutant (`data_size > 0`) makes
    /// a genuinely NEGATIVE `data_size` slip past the guard entirely (a
    /// negative value is never `> 0`), which the zero-byte test above cannot
    /// catch. A tile leader with `dataSize == -1` must be rejected.
    #[test]
    fn parse_chunk_leader_rejects_a_negative_tile_data_size() {
        use crate::attr::tile_description::LevelMode;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&(-1i32).to_le_bytes()); // dataSize = -1
        let mut is = MemIStream::new(bytes);

        let tile_offsets = TileOffsets::new(LevelMode::OneLevel, 1, 1, &[1], &[1]).unwrap();
        let mut offsets: Vec<u64> = vec![0];
        let framing = Framing::Tile(tile_offsets);
        let parts = [PartOffsets { framing, offsets: &mut offsets }];
        let result = parse_chunk_leader(&mut is, &parts, false, 0, 0, None);
        assert!(
            matches!(result, Err(ExrError::Io(_))),
            "a negative tile dataSize must be rejected as a corrupt chunk, got {result:?}"
        );
    }

    /// Mutation-testing gap fix (chunk/ final rerun, confirmed real gap):
    /// `deep_payload`'s `packed_count + packed_data` (583) drives how far
    /// `parse_chunk_leader` steps past a deep chunk to find the NEXT one --
    /// used by `reconstruct_one_part`/`last_chunk_end` when rebuilding an
    /// incomplete offset table. A stub `Ok(0)`/`Ok(1)` return (ignoring the
    /// two size fields actually on disk) makes reconstruction land inside
    /// the middle of the chunk it just read instead of at the next chunk's
    /// real start, breaking every subsequent slot.
    ///
    /// Builds 2 deep-scanline chunks by hand (`y (i32), packedCountSize,
    /// packedDataSize, unpackedDataSize (3x u64), payload`), each with an
    /// 8-byte payload (`packed_count = 4, packed_data = 4`), and
    /// reconstructs from a fully-zeroed table -- the real payload size (8)
    /// must be added to land chunk 1 at byte 36, not chunk 0's own payload
    /// bytes (0/1 bytes past the leader, i.e. offset 28/29).
    #[test]
    fn deep_payload_size_is_packed_count_plus_packed_data_not_a_stub() {
        fn push_deep_chunk(out: &mut Vec<u8>, y: i32, packed_count: u64, packed_data: u64) {
            out.extend_from_slice(&y.to_le_bytes());
            out.extend_from_slice(&packed_count.to_le_bytes());
            out.extend_from_slice(&packed_data.to_le_bytes());
            out.extend_from_slice(&999u64.to_le_bytes()); // unpackedDataSize, unused here
            let payload_len = (packed_count + packed_data) as usize;
            out.extend(std::iter::repeat_n(0xAAu8, payload_len));
        }

        let mut bytes = Vec::new();
        push_deep_chunk(&mut bytes, 0, 4, 4); // @ 0..36 (28-byte leader + 8-byte payload)
        push_deep_chunk(&mut bytes, 1, 4, 4); // @ 36..72
        assert_eq!(bytes.len(), 72);
        let mut is = MemIStream::new(bytes);

        let mut offsets: Vec<u64> = vec![0, 0]; // fully zeroed: forces reconstruction from scratch
        let framing = Framing::DeepScanline {
            min_y: 0,
            lines_per_chunk: 1,
            line_order: LineOrder::IncreasingY,
        };
        reconstruct_one_part(
            &mut is,
            &mut [PartOffsets { framing, offsets: &mut offsets }],
            false,
            0,
            0,
            Some(72),
        );

        assert_eq!(
            offsets,
            vec![0, 36],
            "chunk 1's real position (36) must be found by adding the ACTUAL packed_count + \
             packed_data (8 bytes), not a stub 0/1"
        );
    }
}

// ---------------------------------------------------------------------------
// Test-only byte surgery on a written EXR image
// ---------------------------------------------------------------------------

/// Helpers that edit a written EXR byte image in place, used by the reader
/// tests across this crate (scanline / multipart) to build the malformed or
/// legacy files C++ still reads.
#[cfg(test)]
pub(crate) mod testing {
    /// Remove the `occurrence`-th (0-based) attribute record named `name` from
    /// an EXR byte image, shifting every chunk offset back by its length.
    ///
    /// The record layout is `name\0 type\0 size(i32 LE) value`.
    pub(crate) fn drop_attribute(bytes: &[u8], name: &str, occurrence: usize) -> Vec<u8> {
        let mut pat = name.as_bytes().to_vec();
        pat.push(0);
        let start = bytes
            .windows(pat.len())
            .enumerate()
            .filter(|(_, w)| *w == pat.as_slice())
            .map(|(i, _)| i)
            .nth(occurrence)
            .expect("attribute not found");
        let mut p = start + pat.len();
        while bytes[p] != 0 {
            p += 1;
        }
        p += 1;
        let size = i32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]) as usize;
        let end = p + 4 + size;
        let mut out = bytes[..start].to_vec();
        out.extend_from_slice(&bytes[end..]);
        shift_offsets(&mut out, (end - start) as u64);
        out
    }

    /// Replace the value of the `occurrence`-th attribute named `name` (a
    /// `string` attribute) with `value`, shifting every chunk offset by the
    /// size delta.  `value` must not be longer than the current value.
    pub(crate) fn set_string_attribute(
        bytes: &[u8],
        name: &str,
        occurrence: usize,
        value: &str,
    ) -> Vec<u8> {
        let mut pat = name.as_bytes().to_vec();
        pat.push(0);
        let start = bytes
            .windows(pat.len())
            .enumerate()
            .filter(|(_, w)| *w == pat.as_slice())
            .map(|(i, _)| i)
            .nth(occurrence)
            .expect("attribute not found");
        let mut p = start + pat.len();
        while bytes[p] != 0 {
            p += 1;
        }
        p += 1; // type-name terminator
        let old_size =
            i32::from_le_bytes(bytes[p..p + 4].try_into().expect("4 bytes")) as usize;
        assert!(value.len() <= old_size, "replacement must not grow the record");

        let mut out = bytes[..p].to_vec();
        out.extend_from_slice(&(value.len() as i32).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(&bytes[p + 4 + old_size..]);
        shift_offsets(&mut out, (old_size - value.len()) as u64);
        out
    }

    /// After a header record is removed, every chunk offset moves back by
    /// `removed` bytes.  The offset tables occupy the bytes from the end of the
    /// header list up to the first chunk — and the first chunk is the one
    /// entry 0 of the first table addresses, which fixes where they end.
    /// Byte offset of the first chunk offset table — i.e. the end of the
    /// header list.
    pub(crate) fn table_start(bytes: &[u8]) -> usize {
        // The MULTI_PART flag decides whether the header list has the extra
        // "second terminator" after the last header.
        let flags = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
        let multipart = flags & 0x1000 != 0;

        let mut p = 8usize; // past magic + version
        loop {
            // Skip this header's attribute records: name  type  size value.
            while bytes[p] != 0 {
                while bytes[p] != 0 {
                    p += 1;
                }
                p += 1; // name terminator
                while bytes[p] != 0 {
                    p += 1;
                }
                p += 1; // type terminator
                let size = i32::from_le_bytes(
                    bytes[p..p + 4].try_into().expect("4 bytes"),
                ) as usize;
                p += 4 + size;
            }
            p += 1; // this header's empty-name terminator
            if !multipart {
                break;
            }
            if bytes[p] == 0 {
                p += 1; // header-list ("second") terminator
                break;
            }
        }
        p
    }

    /// Zero every entry of every chunk offset table, forcing the reader down
    /// the reconstruction path.
    pub(crate) fn zero_offset_tables(bytes: &mut [u8]) {
        let p = table_start(bytes);
        let first = u64::from_le_bytes(bytes[p..p + 8].try_into().expect("8 bytes")) as usize;
        bytes[p..first].fill(0);
    }

    fn shift_offsets(bytes: &mut [u8], removed: u64) {
        let p = table_start(bytes);
        let first_old = u64::from_le_bytes(bytes[p..p + 8].try_into().expect("8 bytes"));
        let table_end = (first_old - removed) as usize;
        let mut q = p;
        while q + 8 <= table_end {
            let v = u64::from_le_bytes(bytes[q..q + 8].try_into().expect("8 bytes"));
            bytes[q..q + 8].copy_from_slice(&(v - removed).to_le_bytes());
            q += 8;
        }
    }
}
