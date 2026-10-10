//! Raw-chunk passthrough — read and write **compressed** EXR pixel chunks
//! verbatim, with no decode / re-encode step.
//!
//! This is the public surface for byte-exact EXR→EXR transcode: a consumer can
//! lift every part's header and every part's raw compressed chunk out of one
//! file and lay them straight back down into another, rebuilding a fresh chunk
//! offset table.  Because the compressed bytes are never touched, lossy codecs
//! (DWAA/DWAB/B44/B44A/PXR24) and HTJ2K survive the copy with zero additional
//! quality loss — exactly what a "copy these pixels through" operation needs.
//!
//! # What this ports
//!
//! OpenEXR exposes the same capability through three C++ entry points; this
//! module is their faithful, unified Rust port:
//!
//! * `InputFile::rawPixelData(int scanLine, …)` — read one raw scanline chunk
//!   (offset-table entry → seek → `[y][dataSize][compressed bytes]`),
//! * `TiledInputFile::rawTileData(…)` — read one raw tile chunk
//!   (`[dx,dy,lx,ly][dataSize][compressed bytes]`),
//! * `OutputFile::copyPixels(InputFile&)` / `TiledOutputFile::copyPixels(
//!   TiledInputFile&)` — write the raw chunks straight through with a fresh
//!   offset table.
//!
//! Port targets (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfInputFile.cpp>
//! (`rawPixelData`),
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTiledInputFile.cpp>
//! (`rawTileData`),
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOutputFile.cpp>
//! (`copyPixels`).  The on-disk chunk framing reproduced here is the one this
//! crate's [`scanline::OutputFile`](crate::scanline::OutputFile) /
//! [`tiled::TiledOutputFile`](crate::tiled::TiledOutputFile) /
//! [`MultiPartOutputFile`](crate::multipart::MultiPartOutputFile) emit, so the
//! writer below is their exact inverse.
//!
//! # Scope
//!
//! **Single-part and multipart, scanline and tiled** (flat images).  Deep parts
//! are intentionally rejected with a tracked [`ExrError`]: their chunk leader is
//! a sample-count / packed-size triple, not the flat `[coord][dataSize][data]`
//! framing this surface models, so mis-reading them is never risked (use the
//! [`deep`](crate::deep) readers/writers for deep data).
//!
//! # Round-trip contract
//!
//! [`RawImage::read`] captures `(version, parts)`, each part being its
//! [`Header`] plus every [`RawChunk`] in ascending file order.  [`RawImage::write`]
//! re-emits magic + version + headers, reserves a fresh chunk offset table per
//! part, then writes every chunk back (re-deriving each chunk's leader from its
//! [`ChunkCoord`] and prefixing the multipart `partNumber` when there is more
//! than one part) and back-patches the offset table.  The compressed
//! [`RawChunk::data`] bytes are emitted verbatim, so a read→write pass preserves
//! every chunk's bytes exactly.

pub mod offsets;

use crate::compressor::num_lines_in_buffer;
use crate::misc::bytes_per_line_table;
use crate::container::{version, xdr, SIZE_I64};
use crate::error::{ExrError, Result};
use crate::header::{self, Header};
use crate::io::{IStream, OStream};
use crate::tiled::misc::{get_tiled_chunk_offset_table_size, precalculate_tile_info};
use crate::tiled::tile_offsets::TileOffsets;

// ---------------------------------------------------------------------------
// ChunkCoord — the coordinate leader that identifies WHERE a chunk belongs
// ---------------------------------------------------------------------------

/// The coordinate leader of a raw chunk — what the offset table indexes and
/// what the chunk's on-disk header carries right after the optional multipart
/// `partNumber`.
///
/// Mirrors the two flat chunk-leader shapes in OpenEXR's `chunk.c`:
/// a scanline chunk carries its block's start `y`; a tile chunk carries its
/// `(dx, dy, lx, ly)`.  The writer re-derives the on-disk leader (and the
/// offset-table slot) from this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkCoord {
    /// Scanline chunk: the starting scanline `y` of the chunk's line block
    /// (`ImfOutputFile.cpp::writePixelData` writes `lineBufferMinY`).
    Scanline {
        /// First scanline of the chunk's block.
        y: i32,
    },
    /// Tile chunk: tile `(dx, dy)` at level `(lx, ly)`
    /// (`ImfTiledOutputFile.cpp::writeTileData`).
    Tile {
        /// Tile column index within the level.
        dx: i32,
        /// Tile row index within the level.
        dy: i32,
        /// Level X index.
        lx: i32,
        /// Level Y index.
        ly: i32,
    },
}

// ---------------------------------------------------------------------------
// RawChunk — one compressed chunk, captured verbatim
// ---------------------------------------------------------------------------

/// One raw EXR pixel chunk: its [`ChunkCoord`] plus the **compressed** payload
/// bytes exactly as stored on disk (the chunk's `dataSize` equals
/// [`data`](Self::data)`.len()`).
///
/// The payload is never decoded, so a [`RawChunk`] round-trips a lossy/HTJ2K
/// chunk byte-for-byte.  The on-disk `dataSize` field and the optional
/// multipart `partNumber` prefix are structural and re-derived by the writer;
/// only the coordinate and the compressed bytes are retained here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawChunk {
    /// Where this chunk's pixels live (the offset-table key + on-disk leader).
    pub coord: ChunkCoord,
    /// The compressed chunk payload, verbatim (`dataSize == data.len()`).
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// RawPart — one part's header + its raw chunks
// ---------------------------------------------------------------------------

/// One part of an EXR file as raw chunks: its [`Header`] plus every
/// [`RawChunk`], in ascending file order.
///
/// A single-part file produces exactly one [`RawPart`]; a multipart file
/// produces one per part, in part order.
///
/// (No `Debug` derive: [`Header`] is intentionally not `Debug` in this crate.)
#[derive(Clone)]
pub struct RawPart {
    /// This part's header (read verbatim; written back unchanged).
    pub header: Header,
    /// This part's raw compressed chunks, in ascending file order.
    pub chunks: Vec<RawChunk>,
}

// ---------------------------------------------------------------------------
// RawImage — a whole EXR file as raw chunks
// ---------------------------------------------------------------------------

/// A whole EXR file captured as raw chunks — the unit of byte-exact passthrough.
///
/// Read one with [`RawImage::read`]; write it back (to any destination) with
/// [`RawImage::write`].  `parts.len() == 1` is a single-part file; more is
/// multipart.  The stored [`version`](Self::version) word is informational —
/// [`write`](Self::write) recomputes the version flags from the parts so that
/// pruning/reordering parts yields a correctly-flagged output.
///
/// (No `Debug` derive: [`RawPart`] holds a non-`Debug` [`Header`].)
#[derive(Clone)]
pub struct RawImage {
    /// The raw file-format version word as read (informational; the writer
    /// recomputes the flags from the parts).
    pub version: i32,
    /// The parts, in part order (one for a single-part file).
    pub parts: Vec<RawPart>,
}

/// Whether a part is scanline-framed or tile-framed (deep is rejected upstream).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartKind {
    Scanline,
    Tile,
}

/// Classify an ALREADY-VALID, caller-constructed header for the WRITE side
/// (`RawImage::write`) — no defaulting/validation here (the header is ours,
/// not attacker input); see [`validate_and_classify_part`] for the read side.
fn part_kind(h: &Header) -> Result<PartKind> {
    if h.has_type() {
        let t = h.type_()?;
        if header::is_deep_data(t) {
            return Err(ExrError::Arg(format!(
                "Raw-chunk passthrough does not support deep part type \"{t}\" \
                 (use the deep readers/writers); chunk framing differs."
            )));
        }
        return Ok(if header::is_tiled_type(t) { PartKind::Tile } else { PartKind::Scanline });
    }
    Ok(if h.has_tile_description() { PartKind::Tile } else { PartKind::Scanline })
}

/// Validate one part's header and classify it as scanline- or tile-framed —
/// the raw-chunk-passthrough analogue of `multipart::input::validate_read_part`
/// (this used to be its own weaker, unvalidated heuristic
/// — no required-attribute defaulting, no fatal name/type checks for a
/// multipart/deep part, and storage read straight off `has_type()`/
/// `has_tile_description()` with no cross-check against the version flags).
///
/// Deep parts are rejected (out of scope for the flat passthrough surface).
fn validate_and_classify_part(
    h: &mut Header,
    tiled_flag: bool,
    non_image: bool,
    multipart: bool,
) -> Result<PartKind> {
    // Required-attribute defaulting + the fatal multipart/deep
    // name/type checks, shared with every other reader constructor
    // (`Header::validate_for_read`, used by `multipart::input::read_prologue`
    // and `Header::read_checked`). `auto_add_type = true` mirrors
    // `MultiPartInputFile::new`'s policy, matching the read side of `RawImage`
    // handling single- and multi-part files uniformly.
    let is_tiled = h.validate_for_read(tiled_flag, non_image, multipart, true)?;

    // Storage: flags alone decide for a plain single-part image; the `type`
    // attribute decides for a deep or multipart part (`parse_header.c:2791-2798`).
    let kind = if !non_image && !multipart {
        if is_tiled { PartKind::Tile } else { PartKind::Scanline }
    } else {
        let t = h.type_()?;
        if header::is_deep_data(t) {
            return Err(ExrError::Arg(format!(
                "Raw-chunk passthrough does not support deep part type \"{t}\" \
                 (use the deep readers/writers); chunk framing differs."
            )));
        }
        if header::is_tiled_type(t) { PartKind::Tile } else { PartKind::Scanline }
    };

    // validate_part_type (non-strict): the flags win; a disagreeing `type` is
    // silently corrected, never an error, matching every other reader.
    let expected = match kind {
        PartKind::Tile => header::TILED_IMAGE,
        PartKind::Scanline => header::SCANLINE_IMAGE,
    };
    if h.has_type() && !non_image && !multipart && h.type_()? != expected {
        h.set_type(expected)?;
    }
    h.sanity_check_at_open(kind == PartKind::Tile, multipart)?;

    Ok(kind)
}

/// Number of chunk-offset-table entries for a part — port of
/// `ImfMisc.cpp::getChunkOffsetTableSize` for the flat (non-deep) part types.
///
/// Tiled parts use the tile-grid sum (`getTiledChunkOffsetTableSize`); scanline
/// parts use `(maxY - minY + linesInBuffer) / linesInBuffer` in int64 math.
fn chunk_count(h: &Header, kind: PartKind) -> Result<usize> {
    match kind {
        PartKind::Tile => {
            let n = get_tiled_chunk_offset_table_size(h)?;
            usize::try_from(n)
                .map_err(|_| ExrError::Input("Negative tiled chunk count".into()))
        }
        PartKind::Scanline => {
            let dw = h.data_window()?;
            let lib = num_lines_in_buffer(h.compression()?)? as i64;
            if lib <= 0 {
                return Err(ExrError::Logic("linesInBuffer must be positive".into()));
            }
            let size = (dw.max.y as i64 - dw.min.y as i64 + lib) / lib;
            usize::try_from(size)
                .map_err(|_| ExrError::Input("Negative scanline chunk count".into()))
        }
    }
}

/// The on-disk chunk framing of a (flat, non-deep) part, for the
/// reconstruction sweep — the passthrough-scope analogue of
/// `multipart::input::part_framing` (deep parts are already rejected by
/// [`part_kind`] upstream, so only the two flat variants are ever produced
/// here).
fn chunk_framing(h: &Header, kind: PartKind) -> Result<offsets::Framing> {
    let dw = h.data_window()?;
    match kind {
        PartKind::Tile => {
            let td = h.tile_description()?;
            let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
                precalculate_tile_info(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
            let geometry = TileOffsets::new(
                td.mode,
                num_x_levels,
                num_y_levels,
                &num_x_tiles,
                &num_y_tiles,
            )?;
            Ok(offsets::Framing::Tile(geometry))
        }
        PartKind::Scanline => Ok(offsets::Framing::Scanline {
            min_y: dw.min.y,
            lines_per_chunk: num_lines_in_buffer(h.compression()?)?,
            line_order: h.line_order()?,
        }),
    }
}

impl RawImage {
    /// Read every part's header and every part's raw compressed chunks from
    /// `stream` — the unified port of `InputFile::rawPixelData` /
    /// `TiledInputFile::rawTileData` over a whole (single-part or multipart)
    /// file.
    ///
    /// Reads magic + version, then the header(s), then each part's chunk offset
    /// table, then seeks to and reads each chunk verbatim (`[coord][dataSize]
    /// [compressed bytes]`, with the multipart `partNumber` prefix verified
    /// against the part index, and each chunk's own coordinate against the
    /// offset-table slot that addressed it).  Chunks are returned in
    /// ascending SLOT order (== ascending file order for a normally-written
    /// file; a `DECREASING_Y` scanline file's slots are still ordered by
    /// ascending `y`, matching `ImfOutputFile.cpp`'s `lineOffsets` indexing).
    ///
    /// # Errors
    /// * [`ExrError::Input`] on bad magic, unknown version flags, a corrupt
    ///   chunk header (negative / oversized `dataSize`), or a multipart
    ///   `partNumber` mismatch.
    /// * [`ExrError::Arg`] for a deep part (out of scope — tracked, never
    ///   mis-read).
    /// * Propagates header-parse and stream errors.
    pub fn read(stream: &mut dyn IStream) -> Result<Self> {
        // --- magic + version word ------------------------------------------
        let magic = xdr::read_i32(stream)?;
        if (magic as u32) != version::MAGIC {
            return Err(ExrError::Input("File is not an OpenEXR file.".into()));
        }
        let version_word = xdr::read_i32(stream)?;
        // The ONE shared version-word validator (supported version
        // number, known flags, and the "single-part TILED can't also be
        // deep/multipart" combo) — this used to be its own weaker check here
        // (no version-number check at all, and no combo check).
        version::validate_version_word(version_word as u32)?;
        let flags = version::get_flags(version_word as u32);
        let tiled_flag = version::is_tiled(flags);
        let non_image = version::is_non_image(flags);
        let multipart = version::is_multi_part(flags);

        // --- read the header(s) --------------------------------------------
        // Single-part: exactly one header.  Multipart: a header list terminated
        // by a second empty attribute name (C++ `parse_header.c`, mirrored by
        // `MultiPartInputFile::new`).
        // The multipart READ path below is currently
        // proven only by the in-crate self-write/self-read loop in
        // `tests/chunk_passthrough.rs`; cross-validation against a real C++
        // `MultiPartOutputFile` golden is pending a multipart golden from
        // the `cargo xtask parity-exr` oracle (the committed `golden_multipart.exr`
        // has a deep part and is intentionally rejected, so it cannot serve here).
        let mut headers: Vec<Header> = Vec::new();
        if multipart {
            loop {
                let h = Header::read_attributes_from(stream, version_word)?;
                headers.push(h);
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

        // --- validate + classify every part --------------------------------
        // Required-attribute defaulting, the fatal multipart/deep
        // name/type checks, the flags-vs-type storage derivation, the
        // non-strict type fix-up, and `sanity_check` — the same validation
        // every other reader constructor runs, via the shared
        // `Header::validate_for_read` (this surface used to skip all of it).
        let kinds: Vec<PartKind> = headers
            .iter_mut()
            .map(|h| validate_and_classify_part(h, tiled_flag, non_image, multipart))
            .collect::<Result<_>>()?;

        // --- read each part's chunk offset table ---------------------------
        // The tables follow the header(s) in part order, each sized by that
        // part's chunkCount.

        // The shared checked reader (`offsets::read_offset_table`) caps the
        // entry count and range-checks every offset before anything is
        // allocated — see `chunk/offsets.rs`.
        let mut offset_tables: Vec<Vec<u64>> = Vec::with_capacity(headers.len());
        let mut any_incomplete = false;
        for (h, &kind) in headers.iter().zip(kinds.iter()) {
            let count = i64::try_from(chunk_count(h, kind)?)
                .map_err(|_| ExrError::Input("Chunk count out of range".into()))?;
            let table = offsets::read_offset_table(stream, count)?;
            any_incomplete |= !table.complete;
            offset_tables.push(table.offsets);
        }

        // An incomplete table used to be silently truncated here (the
        // caller got a `RawImage` missing every chunk whose slot was lost, and
        // `RawImage::write` then wrote a file with holes). C++ instead
        // RECONSTRUCTS the table by scanning the chunk region
        // (`OpenEXRCore/chunk.c::extract_chunk_table` -> `reconstruct_chunk_table`)
        // and only fails if a chunk is still missing after that
        // (`EXR_ERR_INCOMPLETE_CHUNK_TABLE`, surfaced by
        // `ScanLineInputFile::rawPixelDataToBuffer` throwing). This mirrors the
        // exact same pattern `multipart::input::read_prologue` uses.
        if any_incomplete {
            let first_chunk_pos = stream.tellg()?;
            let framings: Vec<offsets::Framing> = headers
                .iter()
                .zip(kinds.iter())
                .map(|(h, &kind)| chunk_framing(h, kind))
                .collect::<Result<_>>()?;
            let mut views: Vec<offsets::PartOffsets<'_>> = offset_tables
                .iter_mut()
                .zip(framings)
                .map(|(offs, framing)| offsets::PartOffsets {
                    framing,
                    offsets: offs,
                })
                .collect();
            offsets::reconstruct(stream, &mut views, multipart, first_chunk_pos)?;

            // `partComplete` is evaluated AFTER reconstruction, same as
            // `multipart::input::read_prologue` (mirrors C++
            // `exr_validate_chunk_table`, `OpenEXRCore/part.c:471`).
            let stream_size = stream.size();
            for offs in &offset_tables {
                if !offsets::table_complete(offs, first_chunk_pos, stream_size) {
                    return Err(ExrError::Input(
                        "Chunk offset table is incomplete and could not be fully \
                         reconstructed (corrupt or truncated file)."
                            .into(),
                    ));
                }
            }
        }

        // Per-slot framing (needed to verify each chunk's leader
        // coordinate maps back to the slot that addressed it), built before
        // `headers` is consumed below.
        let framings: Vec<offsets::Framing> = headers
            .iter()
            .zip(kinds.iter())
            .map(|(h, &k)| chunk_framing(h, k))
            .collect::<Result<_>>()?;

        // --- read each part's chunks ---------------------------------------
        let stream_size = stream.size();
        let mut parts: Vec<RawPart> = Vec::with_capacity(headers.len());
        for (part_index, (((h, &kind), offs), framing)) in headers
            .into_iter()
            .zip(kinds.iter())
            .zip(offset_tables.iter())
            .zip(framings.iter())
            .enumerate()
        {
            // The per-line uncompressed byte count,
            // used to bound each scanline chunk's `dataSize` against the
            // exact unpacked size for the lines it covers (accounting for
            // y-subsampling) -- C++ `chunk.c:1084-1096` rejects `packed >
            // unpacked_size_per_chunk`. Only computed for scanline parts.
            let bytes_per_line: Option<Vec<usize>> = if kind == PartKind::Scanline {
                let dw = h.data_window()?;
                let width = dw.max.x - dw.min.x + 1;
                let mut table = Vec::new();
                bytes_per_line_table(dw.min.y, dw.max.y, width, h.channels()?, &mut table)?;
                Some(table)
            } else {
                None
            };

            // Tile case: the same `unpacked_size_per_chunk` bound, but
            // C++ computes it for tiles as the FULL nominal tile's byte size
            // (`tileDesc.x_size * tileDesc.y_size` per channel, never clipped
            // by an edge tile's smaller actual rect) -- `parse_header.c:2531-2543`.
            let bytes_per_tile: Option<usize> = if kind == PartKind::Tile {
                let td = h.tile_description()?;
                let mut total: usize = 0;
                for (_name, channel) in h.channels()?.iter() {
                    let px = crate::misc::pixel_type_size(channel.type_)?;
                    let tile_pixels = (td.x_size as usize)
                        .checked_mul(td.y_size as usize)
                        .ok_or_else(|| ExrError::Arg("Tile dimensions overflow.".into()))?;
                    let cunpsz = px
                        .checked_mul(tile_pixels)
                        .ok_or_else(|| ExrError::Arg("Tile channel size overflows.".into()))?;
                    total = total
                        .checked_add(cunpsz)
                        .ok_or_else(|| ExrError::Arg("Tile unpacked size overflows.".into()))?;
                }
                Some(total)
            } else {
                None
            };

            let chunks = read_part_chunks(
                stream,
                &PartChunkDescriptor {
                    kind,
                    framing,
                    part_index: part_index as i32,
                    multipart,
                    bytes_per_line: bytes_per_line.as_deref(),
                    bytes_per_tile,
                },
                offs,
                stream_size,
            )?;
            log::debug!(
                "RawImage::read: part {part_index} kind={kind:?} chunks={}",
                chunks.len()
            );
            parts.push(RawPart { header: h, chunks });
        }

        Ok(Self {
            version: version_word,
            parts,
        })
    }

    /// Write every part's header and raw chunks to `stream` with a fresh chunk
    /// offset table — the unified port of `OutputFile::copyPixels` /
    /// `TiledOutputFile::copyPixels`.
    ///
    /// Emits magic + version (flags recomputed from the parts), the header(s)
    /// (plus the header-list terminator when multipart), a reserved (zero-filled)
    /// offset table per part, then every chunk verbatim — re-deriving each
    /// chunk's on-disk leader from its [`ChunkCoord`], prefixing the multipart
    /// `partNumber` when there is more than one part, and back-patching the
    /// offset table immediately after each chunk (so the file is valid without
    /// relying on `Drop`, exactly as this crate's writers do).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when the image has no parts, or a chunk's coordinate
    ///   does not match its part kind (a scanline coord in a tiled part or vice
    ///   versa), or a deep part is present.
    /// * [`ExrError::Input`] when a chunk's payload exceeds the EXR `i32`
    ///   `dataSize` limit.
    /// * Propagates header-write, offset-table and stream errors.
    pub fn write(&self, stream: &mut dyn OStream) -> Result<()> {
        if self.parts.is_empty() {
            return Err(ExrError::Arg(
                "RawImage::write: no parts to write.".into(),
            ));
        }
        let multipart = self.parts.len() != 1;

        let kinds: Vec<PartKind> = self
            .parts
            .iter()
            .map(|p| part_kind(&p.header))
            .collect::<Result<_>>()?;

        // --- prologue: magic + version + header(s) -------------------------
        if multipart {
            write_magic_and_version(stream, &self.parts)?;
            // Every multipart part header must carry the `chunkCount` attribute,
            // exactly as OpenEXR's `MultiPartOutputFile` writes it
            // (`setChunkCount(getChunkOffsetTableSize(header))` per part — see
            // `multipart::output::MultiPartOutputFile::new`'s `is_multi_part`
            // branch). A strict C++ multipart reader requires it; our own reader
            // re-derives the count and so survives without it, which is why a
            // self-write/self-read loop passes regardless — the attribute must be
            // emitted for genuine C++ interop. `chunk_count` here is the same
            // value `getChunkOffsetTableSize` computes for the part's kind
            // (scanline data-window math / tiled grid sum). Single-part output
            // (the `else` branch) intentionally OMITS `chunkCount`, matching both
            // the C++ single-image `MultiPartOutputFile` branch and this crate's
            // single-part writers, so single-part bytes stay byte-identical.
            //
            // Cross-validating this multipart output
            // against a real C++ `MultiPartInputFile` golden is still pending a
            // multipart case in the `cargo xtask parity-exr` oracle; until
            // then the multipart path is proven only by the in-crate self-loop.
            for (p, &kind) in self.parts.iter().zip(kinds.iter()) {
                let mut hdr = p.header.clone();
                let cc = i32::try_from(chunk_count(&hdr, kind)?).map_err(|_| {
                    ExrError::Input("Part chunk count exceeds the i32 range.".into())
                })?;
                hdr.set_chunk_count(cc)?;
                hdr.write_attributes_to(stream)?;
            }
            // The header-list terminator (only written when parts != 1).
            xdr::write_string(stream, "")?;
        } else {
            // Single-part: Header::write_to writes magic + version (flags folded
            // in from the header) + attributes + the header terminator. No
            // `chunkCount` is written (it is derivable on read), keeping
            // single-part output byte-identical to the pre-passthrough writers.
            self.parts[0].header.write_to(stream, 0)?;
        }

        // --- reserve each part's offset table (zeros), in part order -------
        let mut writers: Vec<PartWriter> = Vec::with_capacity(self.parts.len());
        for (p, &kind) in self.parts.iter().zip(kinds.iter()) {
            writers.push(PartWriter::reserve(stream, &p.header, kind)?);
        }

        // --- write each part's chunks, back-patching its table -------------
        for (part_index, ((p, &kind), pw)) in self
            .parts
            .iter()
            .zip(kinds.iter())
            .zip(writers.iter_mut())
            .enumerate()
        {
            let pn = if multipart { Some(part_index as i32) } else { None };
            for chunk in &p.chunks {
                pw.write_chunk(stream, kind, pn, chunk)?;
            }
        }

        Ok(())
    }
}

/// Read one part's chunks by seeking to each (non-zero) offset, in
/// SLOT order — the per-part loop of `copyPixels`, which reads chunks via
/// `rawPixelData` / `rawTileData`.
///
/// This used to sort+dedup the raw offsets and read
/// whatever coordinate the leader claimed, with no check that it matched the
/// SLOT that addressed it — so two offset-table slots cross-linked to the
/// same physical offset silently read the same chunk twice (and the other
/// slot's rightful chunk, if any, was never read). Now every slot's chunk is
/// read independently and its leader coordinate is required to map back to
/// that exact slot via the shared `offsets::{scanline_slot,tile_slot}` oracle
/// — the same one `offsets::reconstruct` uses — matching C++
/// `chunk.c:985-996` (`read_and_validate_chunk_leader`, `EXR_ERR_BAD_CHUNK_LEADER`).
/// Per-part descriptor `read_part_chunks` needs to interpret and validate one
/// part's chunks — not a ported C++ struct (the C++ context threads its
/// equivalent fields through several `part_t`/`context_t` accessors), just a
/// clippy `too_many_arguments` fix: the five fields below always travel
/// together, one per part, for the lifetime of a single `read_part_chunks`
/// call.
struct PartChunkDescriptor<'a> {
    kind: PartKind,
    framing: &'a offsets::Framing,
    part_index: i32,
    multipart: bool,
    /// The per-line uncompressed byte count, scanline parts only.
    bytes_per_line: Option<&'a [usize]>,
    /// Tile case: the FULL nominal tile's uncompressed byte count —
    /// `Σ_channels pixel_size(c) * tileDesc.x_size * tileDesc.y_size` — the
    /// exact bound C++ uses for EVERY tile chunk regardless of whether that
    /// specific tile is edge-clipped smaller (`parse_header.c:2531-2543`
    /// never clips by the tile's actual on-disk rect, only by the nominal
    /// tile dimensions, so an edge tile's real payload is smaller than this
    /// bound but never larger — this is a real bound, not an approximation).
    /// `None` for scanline parts.
    bytes_per_tile: Option<usize>,
}

fn read_part_chunks(
    stream: &mut dyn IStream,
    part: &PartChunkDescriptor<'_>,
    offsets_table: &[u64],
    stream_size: Option<u64>,
) -> Result<Vec<RawChunk>> {
    let &PartChunkDescriptor {
        kind,
        framing,
        part_index,
        multipart,
        bytes_per_line,
        bytes_per_tile,
    } = part;
    let mut chunks = Vec::new();
    for (slot, &off) in offsets_table.iter().enumerate() {
        // A zero offset marks an unwritten slot (incomplete file); the caller
        // already requires the table complete before reaching here.
        if off == 0 {
            continue;
        }
        stream.seekg(off)?;

        // Multipart chunks lead with the part number (C++ `extract_chunk_leader`
        // verifies it matches the requested part).
        if multipart {
            let pn = xdr::read_i32(stream)?;
            if pn != part_index {
                return Err(ExrError::Input(format!(
                    "Invalid part number in chunk: expected {part_index}, found {pn}."
                )));
            }
        }

        let coord = match kind {
            PartKind::Scanline => {
                let y = xdr::read_i32(stream)?;
                let offsets::Framing::Scanline { min_y, lines_per_chunk, .. } = framing else {
                    return Err(ExrError::Logic("scanline part has non-scanline framing".into()));
                };
                let found_slot = offsets::scanline_slot(y, *min_y, *lines_per_chunk)?;
                if found_slot != slot {
                    return Err(ExrError::Input(format!(
                        "Chunk at offset {off} has scanline y={y}, which addresses slot \
                         {found_slot}, not the slot ({slot}) whose offset-table entry pointed \
                         here (corrupt or cross-linked offset table)."
                    )));
                }
                ChunkCoord::Scanline { y }
            }
            PartKind::Tile => {
                let dx = xdr::read_i32(stream)?;
                let dy = xdr::read_i32(stream)?;
                let lx = xdr::read_i32(stream)?;
                let ly = xdr::read_i32(stream)?;
                let offsets::Framing::Tile(geometry) = framing else {
                    return Err(ExrError::Logic("tiled part has non-tile framing".into()));
                };
                let found_slot = offsets::tile_slot(geometry, dx, dy, lx, ly)?;
                if found_slot != slot {
                    return Err(ExrError::Input(format!(
                        "Chunk at offset {off} has tile coords ({dx},{dy},{lx},{ly}), which \
                         address slot {found_slot}, not the slot ({slot}) whose offset-table \
                         entry pointed here (corrupt or cross-linked offset table)."
                    )));
                }
                ChunkCoord::Tile { dx, dy, lx, ly }
            }
        };

        let data_size = xdr::read_i32(stream)?;
        if data_size < 0 {
            return Err(ExrError::Input(format!(
                "Chunk reports a negative data size of {data_size} bytes."
            )));
        }
        let data_size = data_size as usize;
        // Guard against a corrupt size requesting an absurd allocation.
        if let Some(sz) = stream_size
            && data_size as u64 > sz
        {
            return Err(ExrError::Input(format!(
                "Chunk data size {data_size} exceeds the stream size {sz}."
            )));
        }

        // Bound the packed size against the EXACT unpacked size
        // for the lines this scanline chunk covers (accounting for
        // y-subsampling), not just the whole-stream size -- C++
        // `chunk.c:1084-1096` (`packed > part->unpacked_size_per_chunk` is a
        // corrupt-chunk error). A ZIP file with a subsampled channel and an
        // over-declared `dataSize` used to sail through to the codec instead
        // of being rejected at the leader.
        if let (PartKind::Scanline, Some(table)) = (kind, bytes_per_line) {
            let ChunkCoord::Scanline { y } = coord else { unreachable!() };
            let offsets::Framing::Scanline { min_y, lines_per_chunk, .. } = framing else {
                unreachable!("scanline kind always pairs with Scanline framing")
            };
            let idx0 = (y - *min_y) as usize;
            let idx1 = (idx0 + *lines_per_chunk as usize).min(table.len());
            let unpacked_size_per_chunk: usize = table.get(idx0..idx1).map_or(0, |s| s.iter().sum());
            if data_size > unpacked_size_per_chunk {
                return Err(ExrError::Input(format!(
                    "Chunk at y={y} reports a packed size of {data_size} bytes, exceeding \
                     this chunk's unpacked size of {unpacked_size_per_chunk} bytes \
                     (corrupt chunk)."
                )));
            }
        }

        // Tile case: same bound, using the FULL nominal tile's
        // uncompressed byte size (see `bytes_per_tile`'s doc comment for why
        // this is not clipped per edge tile — it matches C++ exactly).
        if let (PartKind::Tile, Some(bound)) = (kind, bytes_per_tile)
            && data_size > bound
        {
            let ChunkCoord::Tile { dx, dy, lx, ly } = coord else { unreachable!() };
            return Err(ExrError::Input(format!(
                "Tile ({dx}, {dy}, {lx}, {ly}) reports a packed size of {data_size} bytes, \
                 exceeding the nominal tile's unpacked size of {bound} bytes (corrupt chunk)."
            )));
        }

        // Fallible, not `vec![..]` -- see
        // `misc::try_filled_vec`'s docs. `data_size` is already bounded by
        // the real stream size above, so this is defense-in-depth for a
        // legitimately huge file rather than an amplification fix.
        let mut data = crate::misc::try_filled_vec(data_size, 0u8)?;
        // `read` returns Ok(false) when the file's last byte was just consumed
        // (the final chunk) — that is success, not an error.
        stream.read(&mut data)?;
        chunks.push(RawChunk { coord, data });
    }

    Ok(chunks)
}

// ---------------------------------------------------------------------------
// PartWriter — per-part reserved offset table + back-patch state
// ---------------------------------------------------------------------------

/// Per-part write state: the reserved offset-table position plus whatever is
/// needed to map a chunk's [`ChunkCoord`] to its table slot.
enum PartWriter {
    /// Scanline part: slot = `(y - min_y) / lines_in_buffer`.
    Scanline {
        table_pos: u64,
        min_y: i32,
        lines_in_buffer: i32,
        /// Number of reserved offset-table slots — the upper bound the chunk's
        /// computed slot is validated against before back-patching (so a
        /// malformed `y` cannot seek past the table). Equals the scanline
        /// `getChunkOffsetTableSize` (`(maxY-minY+linesInBuffer)/linesInBuffer`).
        count: usize,
    },
    /// Tiled part: the [`TileOffsets`] table provides the slot index.
    Tile {
        table_pos: u64,
        offsets: TileOffsets,
    },
}

impl PartWriter {
    /// Reserve (zero-fill) this part's offset table at the current stream
    /// position and capture the state needed to back-patch it — mirrors the
    /// reservation done by this crate's scanline / tiled / multipart writers.
    fn reserve(stream: &mut dyn OStream, h: &Header, kind: PartKind) -> Result<Self> {
        match kind {
            PartKind::Scanline => {
                let dw = h.data_window()?;
                let lines_in_buffer = num_lines_in_buffer(h.compression()?)?;
                let count = chunk_count(h, kind)?;
                let table_pos = stream.tellp()?;
                for _ in 0..count {
                    xdr::write_u64(stream, 0)?;
                }
                Ok(PartWriter::Scanline {
                    table_pos,
                    min_y: dw.min.y,
                    lines_in_buffer,
                    count,
                })
            }
            PartKind::Tile => {
                let td = h.tile_description()?;
                let dw = h.data_window()?;
                let (num_x_tiles, num_y_tiles, num_x_levels, num_y_levels) =
                    precalculate_tile_info(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
                let offsets = TileOffsets::new(
                    td.mode,
                    num_x_levels,
                    num_y_levels,
                    &num_x_tiles,
                    &num_y_tiles,
                )?;
                // `write_to` writes the (zero) table and returns its start pos.
                let table_pos = offsets.write_to(stream)?;
                Ok(PartWriter::Tile { table_pos, offsets })
            }
        }
    }

    /// Write one chunk's on-disk record (`[partNumber?][coord][dataSize][data]`)
    /// and back-patch its offset into the reserved table — the inverse of
    /// [`read_part_chunks`], mirroring `writePixelData` / `writeTileData`.
    fn write_chunk(
        &mut self,
        stream: &mut dyn OStream,
        kind: PartKind,
        part_number: Option<i32>,
        chunk: &RawChunk,
    ) -> Result<()> {
        let data_size = i32::try_from(chunk.data.len()).map_err(|_| {
            ExrError::Input(format!(
                "Chunk payload of {} bytes exceeds the EXR i32 dataSize limit.",
                chunk.data.len()
            ))
        })?;

        let chunk_position = stream.tellp()?;

        match (self, kind, chunk.coord) {
            (
                PartWriter::Scanline {
                    table_pos,
                    min_y,
                    lines_in_buffer,
                    count,
                },
                PartKind::Scanline,
                ChunkCoord::Scanline { y },
            ) => {
                // Map the chunk's `y` to its offset-table slot and bound it to
                // the reserved table BEFORE writing, so a malformed `y` (above
                // the data window, or a hand-built `RawChunk`) cannot seek past
                // the table into the chunk region and corrupt 8 bytes. C++ bounds
                // the slot implicitly: `getScanlineChunkOffsetTableSize` sizes the
                // table and the writer only emits in-window scanlines; here the
                // input is untrusted, so the bound is explicit. (The tiled path is
                // already bounds-checked inside `TileOffsets::slot_index`.)
                let slot = (y - *min_y) / *lines_in_buffer;
                let slot = usize::try_from(slot).map_err(|_| {
                    ExrError::Input(format!("Chunk y {y} is below the data window."))
                })?;
                if slot >= *count {
                    return Err(ExrError::Input(format!(
                        "Chunk y {y} maps to offset-table slot {slot}, but the part \
                         reserved only {count} scanline slots (y is above the data window)."
                    )));
                }

                if let Some(pn) = part_number {
                    xdr::write_i32(stream, pn)?;
                }
                xdr::write_i32(stream, y)?;
                xdr::write_i32(stream, data_size)?;
                stream.write(&chunk.data)?;

                back_patch(stream, *table_pos, slot, chunk_position)?;
            }
            (
                PartWriter::Tile { table_pos, offsets },
                PartKind::Tile,
                ChunkCoord::Tile { dx, dy, lx, ly },
            ) => {
                // Record the offset in the table model first so `slot_index`
                // (and any later completeness query) sees it.
                *offsets.get_mut(dx, dy, lx, ly)? = chunk_position;

                if let Some(pn) = part_number {
                    xdr::write_i32(stream, pn)?;
                }
                xdr::write_i32(stream, dx)?;
                xdr::write_i32(stream, dy)?;
                xdr::write_i32(stream, lx)?;
                xdr::write_i32(stream, ly)?;
                xdr::write_i32(stream, data_size)?;
                stream.write(&chunk.data)?;

                let slot = offsets.slot_index(dx, dy, lx, ly)?;
                back_patch(stream, *table_pos, slot, chunk_position)?;
            }
            _ => {
                return Err(ExrError::Arg(
                    "RawImage::write: chunk coordinate does not match its part kind \
                     (scanline coord in a tiled part or vice versa)."
                        .into(),
                ));
            }
        }

        Ok(())
    }
}

/// Back-patch one `u64` LE offset-table slot, then restore the write cursor —
/// the seek-back/seek-forward used by every borrowed-stream writer in this
/// crate (the C++ writers do this from their destructor instead).
fn back_patch(stream: &mut dyn OStream, table_pos: u64, slot: usize, value: u64) -> Result<()> {
    let saved = stream.tellp()?;
    stream.seekp(table_pos + slot as u64 * SIZE_I64 as u64)?;
    xdr::write_u64(stream, value)?;
    stream.seekp(saved)?;
    Ok(())
}

/// Write the magic number + multipart version word — port of
/// `ImfMultiPartOutputFile.cpp::writeMagicNumberAndVersionField` for the
/// passthrough writer (only the `parts.len() > 1` form is exercised here, since
/// single-part output goes through [`Header::write_to`]).
fn write_magic_and_version(stream: &mut dyn OStream, parts: &[RawPart]) -> Result<()> {
    xdr::write_i32(stream, version::MAGIC as i32)?;

    let mut ver = version::EXR_VERSION;
    if parts.len() == 1 {
        // Faithful to C++; unreached here because single-part uses write_to.
        if parts[0].header.has_type() && parts[0].header.type_()? == header::TILED_IMAGE {
            ver |= version::TILED_FLAG;
        }
    } else {
        ver |= version::MULTI_PART_FILE_FLAG;
    }
    for p in parts {
        if p.header.uses_long_names() {
            ver |= version::LONG_NAMES_FLAG;
        }
        if p.header.has_type() && !header::is_image(p.header.type_()?) {
            ver |= version::NON_IMAGE_FLAG;
        }
    }

    xdr::write_i32(stream, ver as i32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
    use crate::chunk::offsets::testing::{table_start, zero_offset_tables};
    use crate::frame_buffer::{FrameBuffer, Slice};
    use crate::io::{MemIStream, MemOStream};
    use crate::scanline::OutputFile;
    use imath_rs::{Box2i, V2f, Vec2};

    const W: i32 = 4;
    const H: i32 = 6;

    /// A single-part NONE scanline file (one line per chunk, so H rows give H
    /// chunks) carrying a distinguishable byte pattern per chunk.
    fn write_scanline() -> Vec<u8> {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let mut pixels = Vec::with_capacity((W * H) as usize * 4);
        for y in 0..H {
            for x in 0..W {
                pixels.extend_from_slice(&((y * W + x) as f32).to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "Z",
            Slice::new(PixelType::Float, 0, 4, (W as usize) * 4, 1, 1, 0.0, false, false).unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, H).unwrap();
        }
        out.into_data()
    }

    /// A single-part NONE tiled file, one FLOAT channel, `TS`x`TS` tiles over
    /// a `W`x`H` (both multiples of `TS`) data window — used by the
    /// tile-case packed-size test.
    fn write_tiled(tile_size: u32) -> Vec<u8> {
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode};
        use crate::attr::TileDescriptionAttribute;
        use crate::header::TILED_IMAGE;
        use crate::tiled::TiledOutputFile;

        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h.set_type(TILED_IMAGE).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h.insert(
            "tiles",
            TileDescriptionAttribute::new(crate::attr::tile_description::TileDescription {
                x_size: tile_size,
                y_size: tile_size,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();

        let mut pixels = Vec::with_capacity((W * H) as usize * 4);
        for y in 0..H {
            for x in 0..W {
                pixels.extend_from_slice(&((y * W + x) as f32).to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "Z",
            Slice::new(PixelType::Float, 0, 4, (W as usize) * 4, 1, 1, 0.0, false, false).unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut tf = TiledOutputFile::new(&mut out, &h).unwrap();
            tf.set_frame_buffer(&fb).unwrap();
            let nx = tf.num_x_tiles(0).unwrap();
            let ny = tf.num_y_tiles(0).unwrap();
            tf.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
        }
        out.into_data()
    }

    /// Same shape as [`write_scanline`] but with a data window whose origin is
    /// NOT `(0, 0)` (`min = (5, 3)`, `max = (8, 3 + H - 1)`, still `W` wide) --
    /// used to isolate `width = dw.max.x - dw.min.x + 1` (chunk/mod.rs:451)
    /// from a `+` mutant, which a `min.x == 0` file (every other test here)
    /// cannot distinguish (`max.x - 0 + 1 == max.x + 0 + 1`).
    fn write_scanline_shifted_origin() -> Vec<u8> {
        let dw = Box2i {
            min: Vec2 { x: 5, y: 3 },
            max: Vec2 { x: 8, y: 3 + H - 1 },
        };
        let mut h = Header::with_windows(
            dw,
            dw,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let mut pixels = Vec::with_capacity((W * H) as usize * 4);
        for y in 0..H {
            for x in 0..W {
                pixels.extend_from_slice(&((y * W + x) as f32).to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        // `Slice::make` with `origin = (5, 3)` (the data window's own min)
        // computes `base_offset` so the buffer's first byte lands at pixel
        // (5, 3), matching `write_scanline`'s (0, 0) convention.
        fb.insert(
            "Z",
            Slice::make(
                PixelType::Float,
                0,
                Vec2 { x: 5, y: 3 },
                W as i64,
                4,
                (W as usize) * 4,
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
            let mut of = OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, H).unwrap();
        }
        out.into_data()
    }

    /// Mutation-testing gap fix (chunk/ audit, batch 4): `width = dw.max.x -
    /// dw.min.x + 1` (451:38). With a shifted data window (`min.x = 5, max.x
    /// = 8`) the real width is `4`, giving a 16-byte-per-line bound (1 FLOAT
    /// channel); a `+` mutant computes `max.x + min.x + 1 = 14`, a 56-byte
    /// bound. A declared `dataSize` of 32 sits strictly between the two: the
    /// real code must reject it (32 > 16), while the mutant would wrongly
    /// accept it (32 <= 56).
    #[test]
    fn scanline_width_is_max_minus_min_plus_one_not_max_plus_min_plus_one() {
        let mut bytes = write_scanline_shifted_origin();
        let pos = table_start(&bytes);
        let off0 = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        let data_size_pos = off0 + 4;
        let real_data_size = i32::from_le_bytes(
            bytes[data_size_pos..data_size_pos + 4].try_into().unwrap(),
        );
        assert_eq!(real_data_size, 16, "a 4-wide NONE FLOAT line is 16 bytes uncompressed");
        let claimed = 32i32; // > the real 16-byte bound, <= the `+` mutant's 56-byte bound.
        bytes[data_size_pos..data_size_pos + 4].copy_from_slice(&claimed.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, real_data_size as usize));

        let mut is = MemIStream::new(bytes);
        let err = match RawImage::read(&mut is) {
            Ok(_) => panic!(
                "a 32-byte packed size must be rejected against the true 16-byte bound for a \
                 4-wide line, not accepted"
            ),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("unpacked size"), "unexpected error: {msg}");
    }

    /// A scanline file whose chunk offset table was zeroed (e.g. by an
    /// interrupted write) must have every chunk RECOVERED by the reconstruction
    /// sweep, not silently dropped. Before this fix `RawImage::read` returned
    /// `Ok` with the (empty) table's chunks filtered out entirely.
    #[test]
    fn zeroed_offset_table_is_fully_reconstructed() {
        let good = write_scanline();
        let mut is_good = MemIStream::new(good.clone());
        let good_img = RawImage::read(&mut is_good).expect("read pristine file");
        assert_eq!(good_img.parts.len(), 1);
        let expected_chunks = good_img.parts[0].chunks.len();
        assert!(expected_chunks > 1, "test needs more than one chunk");

        let mut zeroed = good;
        zero_offset_tables(&mut zeroed);

        let mut is = MemIStream::new(zeroed);
        let img = RawImage::read(&mut is).expect("reconstruct from a zeroed table");
        assert_eq!(img.parts.len(), 1);
        assert_eq!(
            img.parts[0].chunks.len(),
            expected_chunks,
            "every chunk must be recovered, none silently dropped"
        );
        assert_eq!(img.parts[0].chunks, good_img.parts[0].chunks);
    }

    /// When a chunk is genuinely missing (truncated file, not just an
    /// unwritten offset table), reconstruction cannot recover it, and
    /// `RawImage::read` must error — never return `Ok` with fewer chunks than
    /// the header's `chunkCount` implies.
    #[test]
    fn truncated_file_after_zeroed_table_errors_not_silently_drops() {
        let good = write_scanline();
        let mut zeroed = good;
        zero_offset_tables(&mut zeroed);
        // Cut the file well before the last chunk so reconstruction cannot
        // find every slot.
        zeroed.truncate(zeroed.len() - 4);

        let mut is = MemIStream::new(zeroed);
        let err = match RawImage::read(&mut is) {
            Ok(_) => panic!("a still-incomplete table must error, not return Ok"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("incomplete") || msg.contains("reconstruct"),
            "unexpected error: {msg}"
        );
    }

    /// Mutation-testing gap fix: `read_part_chunks`'s scanline row index
    /// `idx0 = (y - *min_y) as usize` (740:27). Reuses
    /// [`write_scanline_shifted_origin`]'s `min_y = 3`, where a `+` mutant
    /// would compute `idx0 = y + 3` -- for the FIRST row (`y = 3`) that is
    /// `6`, at/past the `bytes_per_line` table's own length (`H = 6`), so
    /// `unpacked_size_per_chunk` collapses to `0` and even the file's own
    /// correctly-declared 16-byte `dataSize` is wrongly rejected. A `-`
    /// mutant would instead correctly reject nothing, and the whole
    /// otherwise-valid file reads cleanly under the real code.
    #[test]
    fn read_part_chunks_scanline_index_is_y_minus_min_y_not_plus() {
        let bytes = write_scanline_shifted_origin();
        let mut is = MemIStream::new(bytes);
        let img = RawImage::read(&mut is).expect("a pristine shifted-origin file must read cleanly");
        assert_eq!(img.parts[0].chunks.len(), H as usize);
    }

    /// Mutation-testing gap fix: the tile packed-size bound's `data_size >
    /// bound` (756:26). A tile whose packed size EXACTLY equals the nominal
    /// bound (the normal, correct case for an uncompressed file) must be
    /// accepted; `==`/`>=` mutants would instead reject exactly this common
    /// case, since the file's own genuine `dataSize` always equals the bound
    /// for `NoCompression`.
    #[test]
    fn read_part_chunks_tile_bound_allows_exact_equality() {
        let bytes = write_tiled(2);
        let mut is = MemIStream::new(bytes);
        let img = RawImage::read(&mut is)
            .expect("a tile whose packed size exactly equals the nominal bound must be accepted");
        assert!(!img.parts[0].chunks.is_empty());
    }

    /// A chunk offset table with two slots cross-linked
    /// to the SAME physical offset must be rejected, not silently read twice
    /// (dropping the other slot's rightful chunk). Matches C++
    /// `chunk.c:985-996` ("BAD_CHUNK_LEADER" -- a chunk's own coordinate must
    /// map back to exactly the slot that addressed it).
    #[test]
    fn cross_linked_offset_table_slots_error_not_silently_duplicate() {
        let mut bytes = write_scanline();
        let pos = table_start(&bytes);
        // Slot 0's offset (first 8 bytes of the table).
        let slot0_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        // Point slot 1 at the same physical chunk as slot 0.
        bytes[pos + 8..pos + 16].copy_from_slice(&slot0_offset.to_le_bytes());

        let mut is = MemIStream::new(bytes);
        let err = match RawImage::read(&mut is) {
            Ok(img) => panic!(
                "a cross-linked offset table must error, not silently duplicate a chunk \
                 (got {} chunks)",
                img.parts[0].chunks.len()
            ),
            Err(e) => e,
        };
        let _ = format!("{err}"); // a real, displayable error
    }

    /// A chunk's declared packed size must not
    /// exceed the EXACT unpacked size for the lines it covers -- C++
    /// `chunk.c:1084-1096`. Patches one chunk's on-disk `dataSize` (the i32
    /// right after its `y`) to claim more bytes than the line actually holds
    /// uncompressed, and pads the file so the inflated read wouldn't hit EOF
    /// first (isolating this check from the separate stream-size guard).
    #[test]
    fn packed_size_exceeding_unpacked_size_per_chunk_errors() {
        let mut bytes = write_scanline();
        // Locate chunk 0 (slot 0's offset), whose leader is `[y i32][dataSize
        // i32][data]`; `y` is 0 for slot 0, so dataSize starts 4 bytes later.
        let pos = table_start(&bytes);
        let off0 = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        let data_size_pos = off0 + 4;
        let real_data_size = i32::from_le_bytes(
            bytes[data_size_pos..data_size_pos + 4].try_into().unwrap(),
        );
        // One NONE-compressed line of a single FLOAT channel, W=4, is 16
        // bytes; claim double that. Pad the file so the (fake) declared
        // length doesn't run past EOF and trip the stream-size guard instead.
        let inflated = real_data_size * 2;
        bytes[data_size_pos..data_size_pos + 4].copy_from_slice(&inflated.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, real_data_size as usize));

        let mut is = MemIStream::new(bytes);
        let err = match RawImage::read(&mut is) {
            Ok(_) => panic!("an over-declared packed size must error, not be accepted"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("unpacked size"), "unexpected error: {msg}");
    }

    /// Tile case: the same `unpacked_size_per_chunk`
    /// bound, but for a TILED part — C++ `parse_header.c:2531-2543` computes
    /// it once, per part, as the FULL nominal tile's uncompressed byte size
    /// (never clipped by an edge tile's smaller actual rect). Patches tile
    /// (0,0)'s on-disk `dataSize` to claim more bytes than even the full
    /// nominal 2x2 FLOAT tile holds uncompressed (16 bytes), padding the file
    /// so the inflated read wouldn't hit EOF first.
    #[test]
    fn tile_packed_size_exceeding_nominal_tile_size_errors() {
        let mut bytes = write_tiled(2);
        // Tile (0,0) is slot 0; its leader is `[dx][dy][lx][ly][dataSize][data]`
        // (5 i32s before the payload), so dataSize starts 16 bytes in.
        let pos = table_start(&bytes);
        let off0 = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        let data_size_pos = off0 + 16;
        let real_data_size = i32::from_le_bytes(
            bytes[data_size_pos..data_size_pos + 4].try_into().unwrap(),
        );
        assert_eq!(real_data_size, 16, "a 2x2 NONE FLOAT tile is 16 bytes uncompressed");
        let inflated = real_data_size * 2;
        bytes[data_size_pos..data_size_pos + 4].copy_from_slice(&inflated.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, real_data_size as usize));

        let mut is = MemIStream::new(bytes);
        let err = match RawImage::read(&mut is) {
            Ok(_) => panic!("an over-declared tile packed size must error, not be accepted"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("unpacked size"), "unexpected error: {msg}");
    }

    /// Mutation-testing gap fix: `validate_and_classify_part`'s
    /// `!non_image && !multipart` gate (only false when EITHER flag is set)
    /// decides whether classification comes from the flags alone or from the
    /// header's own `type` attribute. With `non_image=true`, a `type` of
    /// `DEEP_SCANLINE` must be rejected outright (raw-chunk passthrough does
    /// not support deep data) - the flags-alone branch has no such check, so
    /// an `&&`->`||` mutation (or dropping the `!`s) would incorrectly route
    /// through the flags-alone path and silently accept it as a plain
    /// scanline part instead of erroring.
    #[test]
    fn validate_and_classify_part_rejects_deep_type_when_non_image() {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h.set_type(header::DEEP_SCANLINE).unwrap();
        let err = match validate_and_classify_part(&mut h, false, true, false) {
            Ok(kind) => panic!("a DEEP_SCANLINE type with non_image=true must be rejected, got Ok({kind:?})"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("deep part type"),
            "expected the deep-part-type rejection, got: {err}"
        );
    }

    /// Same as above but for `multipart=true` instead of `non_image=true` -
    /// both flags route to the type-attribute branch, and the two need
    /// separate tests to distinguish `!non_image` from `!multipart` in the
    /// gate (deleting either `!` alone still passes the OTHER flag's test).
    #[test]
    fn validate_and_classify_part_rejects_deep_type_when_multipart() {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h.insert("name", crate::attr::StringAttribute::new("part0".into())).unwrap();
        h.set_type(header::DEEP_SCANLINE).unwrap();
        let err = match validate_and_classify_part(&mut h, false, false, true) {
            Ok(kind) => panic!("a DEEP_SCANLINE type with multipart=true must be rejected, got Ok({kind:?})"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("deep part type"),
            "expected the deep-part-type rejection, got: {err}"
        );
    }

    /// Mutation-testing gap fix: the "silently correct a disagreeing `type`"
    /// guard (`h.has_type() && !non_image && !multipart && h.type_()? !=
    /// expected`) must fire ONLY for a plain single-part image
    /// (`non_image=false, multipart=false`). For a genuinely multipart part,
    /// a `type` that disagrees with the flags-derived `kind` must NOT be
    /// silently rewritten - `sanity_check_at_open` handles multipart type
    /// consistency on its own terms. An `&&`->`||` mutation (or a dropped
    /// `!`) would incorrectly let this guard fire for `multipart=true` too.
    #[test]
    fn validate_and_classify_part_does_not_rewrite_type_when_multipart() {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h.insert("name", crate::attr::StringAttribute::new("part0".into())).unwrap();
        h.set_type(header::SCANLINE_IMAGE).unwrap(); // disagrees with tiled_flag=true below.
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        let kind = validate_and_classify_part(&mut h, true, false, true).unwrap();
        assert_eq!(kind, PartKind::Scanline, "the type attribute (not tiled_flag) decides kind for a multipart part");
        assert_eq!(
            h.type_().unwrap(),
            header::SCANLINE_IMAGE,
            "a multipart part's disagreeing type must NOT be silently rewritten"
        );
    }

    /// Positive counterpart: for a genuinely plain single-part image
    /// (`non_image=false, multipart=false`), a disagreeing `type` attribute
    /// MUST be silently corrected to match the flags (`validate_part_type`,
    /// non-strict - the flags win). Together with the two tests above, this
    /// pins all three flag terms of the `has_type() && !non_image &&
    /// !multipart && type_()!=expected` gate.
    #[test]
    fn validate_and_classify_part_rewrites_disagreeing_type_for_plain_single_part() {
        let mut h = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h.set_type(header::TILED_IMAGE).unwrap(); // disagrees with tiled_flag=false below.
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        let kind = validate_and_classify_part(&mut h, false, false, false).unwrap();
        assert_eq!(kind, PartKind::Scanline, "flags (tiled_flag=false) must win over the disagreeing type");
        assert_eq!(
            h.type_().unwrap(),
            header::SCANLINE_IMAGE,
            "a plain single-part image's disagreeing type MUST be silently corrected"
        );
    }
}
