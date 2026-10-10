//! Port of OpenEXR `ImfMultiPartOutputFile.{h,cpp}` — the multipart container
//! writer.
//!
//! A multipart EXR file stores several independent *parts* (each a scanline,
//! tiled, or deep image with its own header) in one file.  On disk the layout
//! is, in order:
//!
//! 1. the magic number (i32 LE) and a single version word whose
//!    [`MULTI_PART_FILE_FLAG`][crate::container::MULTI_PART_FILE_FLAG] bit is
//!    set (plus `NON_IMAGE_FLAG` / `LONG_NAMES_FLAG` as required) — written
//!    once for the whole file by C++
//!    `GenericOutputFile::writeMagicNumberAndVersionField(os, headers, parts)`;
//! 2. every part header, written sequentially (attributes + a per-header empty
//!    attribute name), followed by **one extra empty attribute name** that
//!    terminates the header *list* (C++ `Data::writeHeadersToFile` writes a
//!    zero-length name "if a multipart file");
//! 3. the per-part chunk offset tables, in part order — each sized by that
//!    part's `chunkCount` (C++ `Data::writeChunkTableOffsets` reserves them as
//!    zeros, to be back-patched as chunks are written);
//! 4. the chunks themselves; **every multipart chunk is prefixed with its
//!    `partNumber` (i32 LE)** before the part-type chunk body (C++
//!    `writePixelData`'s `if (multiPart) Xdr::write(partNumber)`).
//!
//! Each part header carries `name`, `type`, and `chunkCount`; for a scanline
//! part `chunkCount == getScanlineChunkOffsetTableSize(header) ==
//! (dataWindow.max.y - dataWindow.min.y + linesInBuffer) / linesInBuffer`
//! (C++ `ImfMisc.cpp::getChunkOffsetTableSize`).
//!
//! The **scanline** part write path is wired end-to-end via
//! [`OutputFile::from_part`](crate::scanline::OutputFile::from_part).  Tiled /
//! deep part writers use the same `partNumber` chunk seam, which
//! lives in the single-part chunk writers (one `part_number: Option<i32>` flag,
//! not a parallel copy).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMultiPartOutputFile.cpp>
//! magic + version: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfGenericOutputFile.cpp>
//! chunk-count: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMisc.cpp>

use std::collections::HashSet;

use crate::container::{version, xdr};
use crate::error::{ExrError, Result};
use crate::deep::{
    DeepScanLineOutputFileState, DeepScanLineOutputPartWriter, DeepTiledOutputFileState,
    DeepTiledOutputPartWriter,
};
use crate::header::{self, Header};
use crate::io::OStream;
use crate::scanline::{OutputFileState, OutputPartWriter};
use crate::tiled::{TiledOutputFileState, TiledOutputPartWriter};

use super::part::{
    DeepScanLineOutputPart, DeepTiledOutputPart, OutputPart, TiledOutputPart,
};

/// Per-part bookkeeping recorded while the container writes the prologue —
/// port of the parts of `OutputPartData` this writer needs.
struct PartData {
    /// The (type-checked) part header.
    header: Header,
    /// Number of chunk-offset-table entries for this part — C++ `chunkCount`,
    /// equal to `getChunkOffsetTableSize(header)`.
    chunk_count: i32,
    /// File position of this part's reserved (zero-filled) chunk offset table —
    /// C++ `OutputPartData::chunkOffsetTablePosition`.
    chunk_offset_table_position: u64,
    /// Ownership inversion (the file owns per-part state): the persisted
    /// scanline writer STATE for this part, built lazily on the first
    /// [`MultiPartOutputFile::output_part`] call and kept across every
    /// subsequent call — mirrors C++'s `_outputFiles` cache
    /// (`ImfMultiPartOutputFile.cpp:313-321`, one `OutputFile` object per
    /// part, reused for its whole life) instead of rebuilding a fresh,
    /// amnesiac writer (and silently resetting `currentScanLine` /
    /// `missingScanLines` / any partial line-buffer block) on every call.
    /// `None` for a non-scanline part, or before the part's first access.
    scanline_state: Option<OutputFileState>,
    /// The same for **tiled** parts (same design as
    /// `scanline_state` above): the persisted [`TiledOutputFileState`] for
    /// this part, built lazily on the first
    /// [`MultiPartOutputFile::tiled_output_part`] call and kept across every
    /// subsequent call, so `tile_map`/`next_tile_to_write` (the
    /// out-of-order buffering) survive an interleaved-parts write sequence
    /// instead of resetting. `None` for a non-tiled part, or before the
    /// part's first access.
    tiled_state: Option<TiledOutputFileState>,
    /// The same for **deep scanline** parts (same
    /// design as `scanline_state` above): the persisted
    /// [`DeepScanLineOutputFileState`] for this part, built lazily on the
    /// first [`MultiPartOutputFile::deep_scanline_output_part`] call and kept
    /// across every subsequent call. `None` for a non-deep-scanline part, or
    /// before the part's first access.
    deep_scanline_state: Option<DeepScanLineOutputFileState>,
    /// The same for **deep tiled** parts (same
    /// design as `tiled_state` above): the persisted
    /// [`DeepTiledOutputFileState`] for this part, built lazily on the first
    /// [`MultiPartOutputFile::deep_tiled_output_part`] call and kept across
    /// every subsequent call, so the `tile_map`/`next_tile_to_write`
    /// out-of-order tile buffering survives an interleaved-parts write
    /// sequence instead of resetting. `None` for a non-deep-tiled part, or
    /// before the part's first access.
    deep_tiled_state: Option<DeepTiledOutputFileState>,
    /// File position where this part's own `preview` attribute's VALUE bytes
    /// begin, or `0` if this part's header has no preview image — C++
    /// `OutputPartData::previewPosition`, captured from
    /// `headers[i].writeTo(..)`'s return value while writing the header list
    /// (`ImfMultiPartOutputFile.cpp:474-477`). Fed to each writer state's
    /// `from_part_with_preview` constructor
    /// so `update_preview_image` works on a multipart part exactly as it
    /// does on a standalone file.
    preview_position: u64,
}

/// Multipart EXR writer — port of `Imf::MultiPartOutputFile`.
///
/// Construct with [`MultiPartOutputFile::new`]: it validates the headers, sets
/// each part's `chunkCount`, and writes the file prologue (magic + version,
/// all part headers + the header-list terminator, and the reserved per-part
/// offset tables).  Then obtain a scanline part writer with
/// [`output_part`](MultiPartOutputFile::output_part) and drive it like a
/// single-part [`OutputFile`].
///
/// The writer borrows the caller's [`OStream`] for its whole lifetime.  Like
/// the single-part [`OutputFile`], each part back-patches its offset-table
/// entries immediately, so the file is valid once every part's chunks have
/// been written.
pub struct MultiPartOutputFile<'a> {
    /// Borrowed output stream — C++ `Data::os`.
    stream: &'a mut dyn OStream,
    /// Per-part bookkeeping in part order — C++ `Data::parts`.
    parts: Vec<PartData>,
    /// Whether the file is really multipart (`headers.len() > 1`) — C++
    /// `OutputPartData::multipart` (`ImfMultiPartOutputFile.cpp:185-186`,
    /// `:237-238`).  Only then does a chunk carry the `partNumber` (i32 LE)
    /// prefix (`ImfOutputFile.cpp:745,256`); a single-header file must be
    /// byte-identical to the single-part writer.
    is_multi_part: bool,
    /// Captured once at construction (C++ `numThreads`),
    /// forwarded to each part's `*_mt` constructor when its writer state is
    /// first built (scanline, tiled and deep parts alike).
    num_threads: Option<i32>,
}

impl<'a> MultiPartOutputFile<'a> {
    /// Open a `MultiPartOutputFile` on `stream` for `headers` — port of
    /// `MultiPartOutputFile::MultiPartOutputFile(OStream&, const Header*, int,
    /// bool, int)` for the prologue (header + offset-table reservation).
    ///
    /// Mirrors the C++ constructor:
    /// * runs `do_header_sanity_checks` — every part header is sanity-checked
    ///   with the multipart flag (so each requires a non-empty unique `name`
    ///   and a `type`), and `chunkCount` is set from
    ///   `getChunkOffsetTableSize`;
    /// * writes the magic + version word once (multipart flag set);
    /// * writes every part header followed by the header-list terminator;
    /// * reserves (zero-fills) each part's chunk offset table.
    ///
    /// Equivalent to the C++ ctor with `overrideSharedAttributes == false`:
    /// the **strict** `checkSharedAttributesValues` path — a part whose shared
    /// attributes (`displayWindow`, `pixelAspectRatio`, `timecode`,
    /// `chromaticities`) conflict with part 0's is rejected.  Use
    /// [`new_with_override`](Self::new_with_override) for the C++
    /// `overrideSharedAttributes == true` behaviour.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] on an empty header list, a header missing `name`/`type`,
    ///   duplicate part names, conflicting shared attributes, or a failed
    ///   per-part `sanity_check`.
    /// * Propagates stream write / position errors.
    pub fn new(stream: &'a mut dyn OStream, headers: &[Header]) -> Result<Self> {
        Self::new_with_override(stream, headers, false)
    }

    /// Same as [`Self::new`], with an explicit per-file `numThreads`
    /// override (`None` behaves exactly like [`Self::new`]).
    pub fn with_num_threads(
        stream: &'a mut dyn OStream,
        headers: &[Header],
        num_threads: Option<i32>,
    ) -> Result<Self> {
        Self::new_with_override_mt(stream, headers, false, num_threads)
    }

    /// Open a `MultiPartOutputFile` choosing the C++
    /// `overrideSharedAttributes` behaviour — port of
    /// `MultiPartOutputFile::MultiPartOutputFile(…, bool overrideSharedAttributes, …)`
    /// → `Data::do_header_sanity_checks(overrideSharedAttributes)`
    /// (`ImfMultiPartOutputFile.cpp:96-153`).
    ///
    /// `false` → every part 1.. is checked against part 0 with
    /// [`check_shared_attributes_values`] and a conflict is a hard error;
    /// `true` → part 0's shared attributes are copied onto (or erased from)
    /// every other part with [`override_shared_attributes_values`].
    pub fn new_with_override(
        stream: &'a mut dyn OStream,
        headers: &[Header],
        override_shared_attributes: bool,
    ) -> Result<Self> {
        Self::new_with_override_mt(stream, headers, override_shared_attributes, None)
    }

    /// Same as [`Self::new_with_override`], with an explicit `num_threads`
    /// override.
    pub fn new_with_override_mt(
        stream: &'a mut dyn OStream,
        headers: &[Header],
        override_shared_attributes: bool,
        num_threads: Option<i32>,
    ) -> Result<Self> {
        // C++: do_header_sanity_checks(overrideSharedAttributes).
        // size 0 → ArgExc("Empty header list.")
        if headers.is_empty() {
            return Err(ExrError::Arg("Empty header list.".into()));
        }
        let is_multi_part = headers.len() > 1;

        // Build the validated, chunk-counted headers (clones — the container
        // owns its copies, exactly like C++ `_headers`).
        let mut hdrs: Vec<Header> = Vec::with_capacity(headers.len());

        // C++ does part 0 first, then 1.. with the type/uniqueness checks.
        // The per-part work is identical, so it is unified into one loop here;
        // the `is_multi_part` flag carries the multipart-specific requirements
        // into `sanity_check` (name + type) just as C++ passes `isMultiPart`.
        for h in headers {
            let mut hdr = h.clone();

            // C++ `ImfMultiPartOutputFile.cpp:108-113` runs
            // part 0 (and every part) through `sanityCheck(.., isMultiPart)`
            // FIRST -- which is what actually rejects a missing `type` on a
            // multipart part, with the message "Headers in a multipart file
            // should have type attribute." -- and only THEN calls
            // `setChunkCount`. The previous order (a hand-rolled `has_type`
            // check with a different message, then chunk-count, then
            // `sanity_check`) diverged in both error text and order; the
            // `has_type` gate is now `sanity_check`'s own, not duplicated here.
            //
            // C++: setChunkCount(getChunkOffsetTableSize(header)).
            // For a multipart file this is set for every part; for a single
            // "part" that is a non-image type it is also set.  (A single
            // image part keeps no chunkCount — but a 1-element multipart write
            // is unusual; we follow the multipart branch when >1 part and the
            // C++ single-part-non-image branch otherwise.)
            if is_multi_part {
                hdr.sanity_check(hdr.has_tile_description(), true)?;
                let cc = get_chunk_offset_table_size(&hdr, header_is_tiled(&hdr)?)?;
                hdr.set_chunk_count(cc)?;
            } else {
                // Single header: mirror C++'s single-part branch.
                hdr.sanity_check(hdr.has_tile_description(), false)?;
                if hdr.has_type() && !header::is_image(hdr.type_()?) {
                    let cc = get_chunk_offset_table_size(&hdr, header_is_tiled(&hdr)?)?;
                    hdr.set_chunk_count(cc)?;
                }
            }

            hdrs.push(hdr);
        }

        // C++ `do_header_sanity_checks` (ImfMultiPartOutputFile.cpp:120-140):
        // after every part's own sanity check, part 1.. must agree with part 0
        // on the SHARED attributes — either by overriding them from part 0 or
        // by rejecting the conflict.  (Previously undone: two parts with
        // different displayWindow were written silently.)
        if is_multi_part {
            let (first, rest) = hdrs.split_at_mut(1);
            let src = &first[0];
            for dst in rest.iter_mut() {
                if override_shared_attributes {
                    override_shared_attributes_values(src, dst)?;
                } else {
                    let conflicting = check_shared_attributes_values(src, dst)?;
                    if !conflicting.is_empty() {
                        // C++ builds exactly this message from the part name.
                        let mut msg = String::from("Conflicting attributes found for header :: ");
                        msg.push_str(dst.name()?);
                        for a in &conflicting {
                            msg.push_str(&format!(" '{a}' "));
                        }
                        return Err(ExrError::Arg(msg));
                    }
                }
            }
        }

        // C++: headerNameUniquenessCheck (multipart only).
        if is_multi_part {
            let mut names: HashSet<&str> = HashSet::new();
            for h in &hdrs {
                let name = h.name()?;
                if !names.insert(name) {
                    return Err(ExrError::Arg("Each part should have a unique name.".into()));
                }
            }
        }

        // --- write magic + version word (once for the whole file) ----------
        // C++: writeMagicNumberAndVersionField(os, &headers[0], headers.size()).
        write_magic_and_version(stream, &hdrs)?;

        // --- write every part header + the header-list terminator ----------
        // C++ Data::writeHeadersToFile: each header writes its attributes; a
        // TILEDIMAGE part is written with `multiPart == true` (we always pass
        // the attribute-only writer, which is the multipart-correct form).
        // C++ captures
        // `parts[i]->previewPosition = headers[i].writeTo(..)` here
        // (`ImfMultiPartOutputFile.cpp:474-477`) -- `write_attributes_to`'s
        // return value is the equivalent position, previously discarded.
        let mut preview_positions: Vec<u64> = Vec::with_capacity(hdrs.len());
        for h in &hdrs {
            preview_positions.push(h.write_attributes_to(stream)?);
        }
        // C++: if (headers.size() != 1) write a zero-length attribute name to
        // mark the end of all headers.
        if hdrs.len() != 1 {
            xdr::write_string(stream, "")?;
        }

        // --- reserve each part's chunk offset table (zeros) ----------------
        // C++ Data::writeChunkTableOffsets.
        let mut parts: Vec<PartData> = Vec::with_capacity(hdrs.len());
        for (h, preview_position) in hdrs.into_iter().zip(preview_positions) {
            let chunk_count = get_chunk_offset_table_size(&h, header_is_tiled(&h)?)?;
            let pos = stream.tellp()?;
            for _ in 0..chunk_count {
                xdr::write_u64(stream, 0)?;
            }
            parts.push(PartData {
                header: h,
                chunk_count,
                chunk_offset_table_position: pos,
                preview_position,
                scanline_state: None,
                tiled_state: None,
                deep_scanline_state: None,
                deep_tiled_state: None,
            });
        }

        // Structural detail (debug): how many parts the container wrote.
        log::debug!("MultiPartOutputFile: parts={}", parts.len());

        Ok(Self {
            stream,
            parts,
            is_multi_part,
            num_threads,
        })
    }

    /// Number of parts — port of `int MultiPartOutputFile::parts() const`.
    pub fn parts(&self) -> i32 {
        self.parts.len() as i32
    }

    /// The header of part `n` — port of
    /// `const Header& MultiPartOutputFile::header(int) const`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range.
    pub fn header(&self, n: i32) -> Result<&Header> {
        self.part(n).map(|p| &p.header)
    }

    /// Chunk-offset-table size (`chunkCount`) of part `n`.
    pub fn chunk_count(&self, n: i32) -> Result<i32> {
        self.part(n).map(|p| p.chunk_count)
    }

    /// Open the **scanline** part writer for part `n` — C++
    /// `MultiPartOutputFile::getOutputPart<OutputFile>(int)`.
    ///
    /// Returns an [`OutputPartWriter`] view over this part's PERSISTED writer
    /// state (built once, on the first call, and kept across every
    /// later call — mirroring C++'s per-part `_outputFiles` cache,
    /// `ImfMultiPartOutputFile.cpp:313-321`) and this file's stream. Drive it
    /// with `set_frame_buffer` + `write_pixels` exactly like a single-part
    /// writer; every chunk it emits is prefixed with the part number and
    /// back-patched into this part's offset table. Writes interleaved across
    /// parts — write part 0 lines 0-7, write part 1, come back and write part
    /// 0 lines 8-15 — resume correctly, because the SAME state object is
    /// reused: a fresh `output_part(0)` no longer rebuilds (and resets)
    /// `current_scan_line` / `missing_scan_lines` / any partial line-buffer
    /// block.
    ///
    /// The stream cursor must be at the position where the NEXT chunk this
    /// view writes belongs — the caller is responsible for writing each
    /// part's chunks contiguously (matching C++, which offers no seek/resume
    /// across a stream position gap either).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] when `n` is out of range or the part is not a
    ///   scanline image.
    /// * Propagates the [`OutputFileState::from_part`] construction error.
    pub fn output_part(&mut self, n: i32) -> Result<OutputPartWriter<'_>> {
        if n < 0 || n as usize >= self.parts.len() {
            return Err(ExrError::Arg(format!(
                "MultiPartOutputFile::output_part called with invalid part number {n} \
                 on file with {} parts",
                self.parts.len()
            )));
        }
        let idx = n as usize;
        let t = self.parts[idx].header.type_()?;
        if t != header::SCANLINE_IMAGE {
            return Err(ExrError::Arg(format!(
                "output_part: part {n} has type \"{t}\", but only \"{}\" parts are \
                 supported by the scanline part writer.",
                header::SCANLINE_IMAGE
            )));
        }
        if self.parts[idx].scanline_state.is_none() {
            let pos = self.parts[idx].chunk_offset_table_position;
            let preview_position = self.parts[idx].preview_position;
            let part_number = self.is_multi_part.then_some(n);
            self.parts[idx].scanline_state = Some(OutputFileState::from_part_with_preview_mt(
                &self.parts[idx].header,
                part_number,
                pos,
                preview_position,
                self.num_threads,
            )?);
        }
        let state = self.parts[idx]
            .scanline_state
            .as_mut()
            .expect("just ensured Some above");
        Ok(OutputPartWriter::new(self.stream, state))
    }

    /// Open the **tiled** part writer for part `n` — C++
    /// `MultiPartOutputFile::getOutputPart<TiledOutputFile>(int)`.
    ///
    /// Returns a [`TiledOutputPartWriter`] bound to this file's stream and
    /// this part's PERSISTED write state (part number and reserved
    /// tile-offset-table position are baked in; its tile chunks are prefixed
    /// with the part number and back-patched into this part's table).
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `tiledimage` part; propagates the [`TiledOutputFile::from_part`] error.
    ///
    /// Returns a [`TiledOutputPartWriter`] view over this part's PERSISTED
    /// write state (mirrors
    /// [`Self::output_part`]'s scanline design exactly): built once, on the
    /// first call, and kept across every later call, so writes interleaved
    /// across parts resume correctly — a fresh `tiled_output_part(0)` no
    /// longer rebuilds (and resets the `tile_map`/`next_tile_to_write`)
    /// the whole write state.
    pub fn tiled_output_part(&mut self, n: i32) -> Result<TiledOutputPartWriter<'_>> {
        let (header, pos) = self.part_typed(n, header::TILED_IMAGE)?;
        let idx = n as usize;
        if self.parts[idx].tiled_state.is_none() {
            let preview_position = self.parts[idx].preview_position;
            let part_number = self.is_multi_part.then_some(n);
            self.parts[idx].tiled_state = Some(TiledOutputFileState::from_part_with_preview_mt(
                &header,
                part_number,
                pos,
                preview_position,
                self.num_threads,
            )?);
        }
        let state = self.parts[idx]
            .tiled_state
            .as_mut()
            .expect("just ensured Some above");
        Ok(TiledOutputPartWriter::new(self.stream, state))
    }

    /// Open the **deep scanline** part writer for part `n` — C++
    /// `MultiPartOutputFile::getOutputPart<DeepScanLineOutputFile>(int)`.
    ///
    /// Returns a [`DeepScanLineOutputPartWriter`] view over this part's
    /// PERSISTED write state (mirrors
    /// [`Self::output_part`]'s scanline design exactly): built once, on the
    /// first call, and kept across every later call, so writes interleaved
    /// across parts resume correctly.
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `deepscanline` part; propagates the
    /// [`DeepScanLineOutputFileState::from_part`] error.
    pub fn deep_scanline_output_part(&mut self, n: i32) -> Result<DeepScanLineOutputPartWriter<'_>> {
        let (header, pos) = self.part_typed(n, header::DEEP_SCANLINE)?;
        let idx = n as usize;
        if self.parts[idx].deep_scanline_state.is_none() {
            let preview_position = self.parts[idx].preview_position;
            let part_number = self.is_multi_part.then_some(n);
            self.parts[idx].deep_scanline_state =
                Some(DeepScanLineOutputFileState::from_part_with_preview_mt(
                    &header,
                    part_number,
                    pos,
                    preview_position,
                    self.num_threads,
                )?);
        }
        let state = self.parts[idx]
            .deep_scanline_state
            .as_mut()
            .expect("just ensured Some above");
        Ok(DeepScanLineOutputPartWriter::new(self.stream, state))
    }

    /// Open the **deep tiled** part writer for part `n` — C++
    /// `MultiPartOutputFile::getOutputPart<DeepTiledOutputFile>(int)`.
    ///
    /// Returns a [`DeepTiledOutputPartWriter`] view over this part's
    /// PERSISTED write state (mirrors
    /// [`Self::tiled_output_part`]'s design exactly, including the
    /// `tile_map`/`next_tile_to_write` out-of-order tile buffering
    /// surviving across calls).
    ///
    /// # Errors
    /// [`ExrError::Arg`] when `n` is out of range or the part is not a
    /// `deeptile` part; propagates the [`DeepTiledOutputFileState::from_part`] error.
    pub fn deep_tiled_output_part(&mut self, n: i32) -> Result<DeepTiledOutputPartWriter<'_>> {
        let (header, pos) = self.part_typed(n, header::DEEP_TILE)?;
        let idx = n as usize;
        if self.parts[idx].deep_tiled_state.is_none() {
            let preview_position = self.parts[idx].preview_position;
            let part_number = self.is_multi_part.then_some(n);
            self.parts[idx].deep_tiled_state = Some(
                DeepTiledOutputFileState::from_part_with_preview_mt(
                    &header,
                    part_number,
                    pos,
                    preview_position,
                    self.num_threads,
                )?,
            );
        }
        let state = self.parts[idx]
            .deep_tiled_state
            .as_mut()
            .expect("just ensured Some above");
        Ok(DeepTiledOutputPartWriter::new(self.stream, state))
    }

    /// Open the **scanline** part as an [`OutputPart`] wrapper (port of
    /// `Imf::OutputPart`) — the thin-adapter form of
    /// [`output_part`](Self::output_part).
    pub fn output_part_wrapper(&mut self, n: i32) -> Result<OutputPart<'_>> {
        Ok(OutputPart::new(self.output_part(n)?))
    }

    /// Open the **tiled** part as a [`TiledOutputPart`] wrapper (port of
    /// `Imf::TiledOutputPart`).
    pub fn tiled_output_part_wrapper(&mut self, n: i32) -> Result<TiledOutputPart<'_>> {
        Ok(TiledOutputPart::new(self.tiled_output_part(n)?))
    }

    /// Open the **deep scanline** part as a [`DeepScanLineOutputPart`] wrapper
    /// (port of `Imf::DeepScanLineOutputPart`).
    pub fn deep_scanline_output_part_wrapper(
        &mut self,
        n: i32,
    ) -> Result<DeepScanLineOutputPart<'_>> {
        Ok(DeepScanLineOutputPart::new(self.deep_scanline_output_part(n)?))
    }

    /// Open the **deep tiled** part as a [`DeepTiledOutputPart`] wrapper (port
    /// of `Imf::DeepTiledOutputPart`).
    pub fn deep_tiled_output_part_wrapper(&mut self, n: i32) -> Result<DeepTiledOutputPart<'_>> {
        Ok(DeepTiledOutputPart::new(self.deep_tiled_output_part(n)?))
    }

    /// Bounds-checked + type-checked part accessor for the typed part writers.
    /// Returns a clone of part `n`'s header and its reserved offset-table
    /// position after confirming its header `type` equals `want_type` (the C++
    /// part-type dispatch in `getOutputPart`).  Returns by value so the caller
    /// can immediately borrow `self.stream` mutably (the borrow checker treats
    /// a `&self` return as keeping `self` borrowed).
    fn part_typed(&self, n: i32, want_type: &str) -> Result<(Header, u64)> {
        let part = self.part(n)?;
        let t = part.header.type_()?;
        if t != want_type {
            return Err(ExrError::Arg(format!(
                "output part {n} has type \"{t}\", but a \"{want_type}\" part writer was requested."
            )));
        }
        Ok((part.header.clone(), part.chunk_offset_table_position))
    }

    /// Bounds-checked part accessor — C++ `header(int)`'s range check.
    fn part(&self, n: i32) -> Result<&PartData> {
        if n < 0 || n as usize >= self.parts.len() {
            return Err(ExrError::Arg(format!(
                "MultiPartOutputFile::header called with invalid part number {n} \
                 on file with {} parts",
                self.parts.len()
            )));
        }
        Ok(&self.parts[n as usize])
    }
}

/// Write the magic number and the multipart version word — port of
/// `GenericOutputFile::writeMagicNumberAndVersionField(os, const Header*, int)`.
///
/// For `parts == 1` the single-part rule applies (TILED_FLAG if the lone
/// header is a `tiledimage`); for `parts > 1` the `MULTI_PART_FILE_FLAG` is
/// set.  Across all parts, `LONG_NAMES_FLAG` is OR-ed in if any header uses
/// long names, and `NON_IMAGE_FLAG` if any header has a non-image `type`
/// (deep or tiled-typed part).
///
/// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfGenericOutputFile.cpp>.
fn write_magic_and_version(o: &mut dyn OStream, headers: &[Header]) -> Result<()> {
    xdr::write_i32(o, version::MAGIC as i32)?;

    let mut ver = version::EXR_VERSION;

    if headers.len() == 1 {
        // C++ `ImfGenericOutputFile.cpp:69` calls
        // `headers[0].type()` UNCONDITIONALLY here (not guarded by
        // `hasType()`), and `Header::type()` throws `ArgExc` when the `type`
        // attribute is absent (`ImfHeader.cpp:762-765`) — so a single-header
        // `MultiPartOutputFile` with no `type` must be rejected, unlike a
        // plain single-part `OutputFile::new()` (which never requires `type`
        // at all). `type_()` already errors on a missing attribute, so simply
        // calling it unconditionally (instead of gating on `has_type()`)
        // reproduces the throw.
        if headers[0].type_()? == header::TILED_IMAGE {
            ver |= version::TILED_FLAG;
        }
    } else {
        ver |= version::MULTI_PART_FILE_FLAG;
    }

    for h in headers {
        if h.uses_long_names() {
            ver |= version::LONG_NAMES_FLAG;
        }
        // C++: hasType() && isImage(type()) == false  → NON_IMAGE_FLAG.
        if h.has_type() && !header::is_image(h.type_()?) {
            ver |= version::NON_IMAGE_FLAG;
        }
    }

    xdr::write_i32(o, ver as i32)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared attributes — ImfMultiPartOutputFile.cpp:334-400 (override) / 382-448
// (check)
// ---------------------------------------------------------------------------

/// Names of the shared attributes on which `dst` conflicts with `src` — port
/// of `MultiPartOutputFile::Data::checkSharedAttributesValues`
/// (`ImfMultiPartOutputFile.cpp:382-448`).
///
/// C++ returns a `bool conflict` plus an out-vector; the Rust form returns the
/// (possibly empty) vector, which carries both. The rules are exactly C++'s:
/// * `displayWindow` / `pixelAspectRatio` — always present (required
///   attributes), so a plain value comparison;
/// * `timecode` / `chromaticities` — optional: a conflict is reported only when
///   `dst` HAS the attribute and `src` either lacks it or holds a different
///   value.  (`src` having one that `dst` lacks is *not* a conflict — C++ only
///   tests `if (dstTimeCode)`.)
pub fn check_shared_attributes_values(src: &Header, dst: &Header) -> Result<Vec<String>> {
    let mut conflicting: Vec<String> = Vec::new();

    if src.display_window()? != dst.display_window()? {
        conflicting.push("displayWindow".into());
    }
    if src.pixel_aspect_ratio()? != dst.pixel_aspect_ratio()? {
        conflicting.push("pixelAspectRatio".into());
    }

    let src_tc = src.find_typed_attribute::<crate::attr::TimeCode>("timecode");
    let dst_tc = dst.find_typed_attribute::<crate::attr::TimeCode>("timecode");
    if let Some(d) = dst_tc
        && src_tc.map(|s| s.value != d.value).unwrap_or(true)
    {
        conflicting.push("timecode".into());
    }

    let src_ch = src.find_typed_attribute::<crate::attr::Chromaticities>("chromaticities");
    let dst_ch = dst.find_typed_attribute::<crate::attr::Chromaticities>("chromaticities");
    if let Some(d) = dst_ch
        && src_ch.map(|s| s.value != d.value).unwrap_or(true)
    {
        conflicting.push("chromaticities".into());
    }

    Ok(conflicting)
}

/// Copy `src`'s shared attributes onto `dst`, erasing from `dst` the ones
/// `src` does not have — port of
/// `MultiPartOutputFile::Data::overrideSharedAttributesValues`
/// (`ImfMultiPartOutputFile.cpp:334-379`).
///
/// C++ looks each attribute up with
/// `findTypedAttribute<Box2iAttribute|FloatAttribute|TimeCodeAttribute|ChromaticitiesAttribute>`,
/// which returns `None` both when `src` lacks the name AND when `src` holds it
/// under a *different* attribute type. Either way the fallback is the same:
/// erase `name` from `dst`. The previous untyped `src.contains(name)` +
/// `clone_attr()` copied a same-named foreign-typed attribute from `src` onto
/// `dst` verbatim instead of erasing it, and also always overwrote `dst`'s own
/// type even when it disagreed with `src`'s.
pub fn override_shared_attributes_values(src: &Header, dst: &mut Header) -> Result<()> {
    fn copy_or_erase<T: crate::attr::AttrValue + Clone>(
        name: &str,
        src: &Header,
        dst: &mut Header,
    ) -> Result<()> {
        match src.find_typed_attribute::<T>(name) {
            Some(attr) => dst.insert(name, crate::attr::TypedAttribute::new(attr.value.clone())),
            None => {
                if dst.contains(name) {
                    dst.erase(name)?;
                }
                Ok(())
            }
        }
    }

    copy_or_erase::<imath_rs::Box2i>("displayWindow", src, dst)?;
    copy_or_erase::<f32>("pixelAspectRatio", src, dst)?;
    copy_or_erase::<crate::attr::TimeCode>("timecode", src, dst)?;
    copy_or_erase::<crate::attr::Chromaticities>("chromaticities", src, dst)?;
    Ok(())
}

/// Is `h` a tile-framed part, judged by its `type` attribute?
///
/// The writer always sets `type` before sizing, so the attribute is the right
/// oracle there.  The *reader* must instead pass its part's derived storage:
/// with `autoAddType == false` (what `InputFile` uses) a single-part tiled file
/// has no `type` at all and only the version flags say it is tiled.
pub(crate) fn header_is_tiled(h: &Header) -> Result<bool> {
    Ok(h.has_type() && header::is_tiled_type(h.type_()?))
}

/// Chunk-offset-table size for a header — port of
/// `ImfMisc.cpp::getChunkOffsetTableSize(const Header&)`.
///
/// For an unsupported `type` the header's explicit `chunkCount` is used;
/// otherwise `tiled` selects `getTiledChunkOffsetTableSize` (tiled / deep-tiled)
/// or `getScanlineChunkOffsetTableSize` (scanline / deep-scanline).
pub(crate) fn get_chunk_offset_table_size(h: &Header, tiled: bool) -> Result<i32> {
    // Unsupported type → must carry an explicit chunkCount.
    if h.has_type() && !header::is_supported_type(h.type_()?) {
        if h.has_chunk_count() {
            return h.chunk_count();
        }
        return Err(ExrError::Arg(
            "unsupported header type to get chunk offset table size".into(),
        ));
    }

    // Known type: tiled vs scanline.  (`deeptile`/`tiledimage` are tiled and
    // use the tiled-grid sizing; `deepscanline`/`scanlineimage` use the
    // scanline sizing.)  C++ `getChunkOffsetTableSize` →
    // `getTiledChunkOffsetTableSize` for a tiled part (tiled +
    // deep share the same container).
    if tiled {
        return crate::tiled::misc::get_tiled_chunk_offset_table_size(h);
    }

    // getScanlineChunkOffsetTableSize: int64 math, then truncate to i32.
    let dw = h.data_window()?;
    let lines_in_buffer =
        crate::compressor::num_lines_in_buffer(h.compression()?)? as i64;
    let size =
        (dw.max.y as i64 - dw.min.y as i64 + lines_in_buffer) / lines_in_buffer;
    Ok(size as i32)
}

/// Read-side chunk-offset-table sizing — port of `OpenEXRCore/parse_header.c`'s
/// unknown-storage handling (~2477-2491, adopted at 2618-2619).
///
/// Unlike the WRITE-side
/// [`get_chunk_offset_table_size`] (`ImfMisc.cpp::getChunkOffsetTableSize`,
/// which throws `ArgExc` for an unsupported `type` with no `chunkCount`), the
/// *read* path treats a part whose storage this build does not recognise
/// ([`super::input::Storage::Unknown`]) and which carries no `chunkCount`
/// attribute as **0 chunks**, never an error — C++ opens such a part with an
/// empty (but present) part rather than rejecting the whole file. A negative
/// `chunkCount` is clamped to 0, mirroring `chunk_count > 0 ? chunk_count : 0`.
pub(crate) fn get_chunk_offset_table_size_for_read(
    h: &Header,
    storage: super::input::Storage,
) -> Result<i32> {
    if storage == super::input::Storage::Unknown {
        return Ok(if h.has_chunk_count() {
            h.chunk_count()?.max(0)
        } else {
            0
        });
    }
    get_chunk_offset_table_size(h, storage.is_tiled())
}
