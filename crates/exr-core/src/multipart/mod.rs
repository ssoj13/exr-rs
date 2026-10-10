//! Multipart EXR container — port of OpenEXR's `ImfMultiPartOutputFile` /
//! `ImfMultiPartInputFile`.
//!
//! A multipart file packs several independent *parts* (each its own image with
//! its own header) into one EXR.  Compared to a single-part file the
//! differences are exactly three, all derived line-by-line from the C++:
//!
//! * **Version flag.** The version word sets
//!   [`MULTI_PART_FILE_FLAG`][crate::container::MULTI_PART_FILE_FLAG]
//!   (`0x1000`).  It is written once for the whole file (not per header).
//! * **Header list + terminator.** All part headers are written sequentially;
//!   the *list* is closed by one extra empty attribute name (a lone `\0` where
//!   the next header's first attribute name would start).  Each header carries
//!   `name`, `type`, and `chunkCount`.
//! * **Per-part offset tables + partNumber-prefixed chunks.** After the
//!   headers come the per-part chunk offset tables (one per part, in part
//!   order, each sized by that part's `chunkCount`).  Every chunk is then
//!   prefixed with its `partNumber` (i32 LE) before the part-type chunk body.
//!
//! The `partNumber` prefix is the only change to the chunk codepath; it is
//! threaded into the single-part scanline writer/reader as a
//! `part_number: Option<i32>` (see [`crate::scanline::OutputFile::from_part`]
//! and [`crate::scanline::ScanLineInputFile::from_part`]) so there is exactly
//! **one** chunk writer and one chunk reader, not parallel single/multi copies.
//! When `None`, single-part output/input is byte-identical to the pre-multipart
//! code.  On the read side `input::read_prologue` is likewise the one place a
//! file's magic/version/headers/offset tables are parsed, for single-part and
//! multipart alike.
//!
//! ## Coverage
//! * The container + the scanline part write/read path.
//! * The unified `partNumber` chunk seam now also covers the
//!   TILED and DEEP (scanline + tiled) chunk writers/readers (each via the same
//!   `part_number: Option<i32>` `from_part` constructor), plus the thin
//!   per-type [`part`] wrappers (`OutputPart`/`InputPart`,
//!   `TiledOutputPart`/`TiledInputPart`, `DeepScanLine*Part`,
//!   `DeepTiled*Part`) that drive the single-part logic on a chosen
//!   part — reuse, not reimplementation.
//! * The `multiView` (StringVector) attribute + `ImfMultiView`
//!   view/channel helpers.
//! * C++ cross-read parity for mixed-type multipart files.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/tree/v3.3.11/src/lib/OpenEXR>

pub mod input;
pub mod output;
pub mod part;
pub mod part_helper;

pub use input::{
    DeepScanLineInputPartReader, DeepTiledInputPartReader, MultiPartInputFile,
    MultiPartInputFileOwned,
};
pub use output::MultiPartOutputFile;
pub use part::{
    DeepScanLineInputPart, DeepScanLineOutputPart, DeepTiledInputPart, DeepTiledOutputPart,
    InputPart, OutputPart, TiledInputPart, TiledOutputPart,
};
pub use part_helper::{get_channels_in_multi_part_file, split_channels, MultiViewChannelName};

use std::ops::DerefMut;

use crate::error::ExrError;
use crate::header;
use crate::io::IStream;

/// Copy part `ip` of `input` into part `op` of `output` as raw compressed chunks, by the part's
/// storage type (scanline / tiled / deep scanline / deep tiled) — no decode, byte-identical
/// chunks. The output part's header must match the input's (each `copy_pixels` checks it).
///
/// No C++ library counterpart: every OpenEXR tool that copies parts (exrmaketiled
/// `makeTiled.cpp:702-731`, exrmultipart, exrstdattr) writes this dispatch itself; here it is
/// the one place the tools and the Cryptomatte multipart write-back share.
pub fn copy_part<'a, S>(
    input: &mut MultiPartInputFile<'a, S>,
    output: &mut MultiPartOutputFile<'_>,
    ip: i32,
    op: i32,
) -> crate::error::Result<()>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    match input.part_type(ip)? {
        header::SCANLINE_IMAGE => output.output_part_wrapper(op)?.copy_pixels(&mut input.input_part_wrapper(ip)?),
        header::TILED_IMAGE => output.tiled_output_part_wrapper(op)?.copy_pixels(&mut input.tiled_input_part_wrapper(ip)?),
        header::DEEP_SCANLINE => {
            output.deep_scanline_output_part_wrapper(op)?.copy_pixels(&mut input.deep_scanline_input_part_wrapper(ip)?)
        }
        header::DEEP_TILE => {
            output.deep_tiled_output_part_wrapper(op)?.copy_pixels(&mut input.deep_tiled_input_part_wrapper(ip)?)
        }
        other => Err(ExrError::Input(format!("Part {ip} has the unsupported type \"{other}\"."))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
    use crate::container::{version, xdr};
    use crate::error::ExrError;
    use crate::frame_buffer::{FrameBuffer, Slice};
    use crate::header::{Header, SCANLINE_IMAGE};
    use crate::io::{IStream, MemIStream, MemOStream};
    use imath_rs::V2f;

    /// Build a tiny scanline part header (`width` × `height`, single `Z`
    /// float channel) named `name`, with the given compression.
    fn make_header(name: &str, width: i32, height: i32, comp: Compression) -> Header {
        let mut h = Header::with_dimensions(
            width,
            height,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            comp,
        )
        .expect("dims");
        h.set_name(name).expect("name");
        h.set_type(SCANLINE_IMAGE).expect("type");
        let mut channels = ChannelList::new();
        channels
            .insert("Z", Channel::new(PixelType::Float, 1, 1, true))
            .expect("Z channel");
        h.insert("channels", ChannelListAttribute::new(channels))
            .expect("channels");
        h
    }

    /// `height` rows × `width` float pixels, value = part*1000 + y*width + x.
    fn make_pixels(part: i32, width: i32, height: i32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((width * height) as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let v = (part * 1000 + y * width + x) as f32;
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn float_slice(width: i32) -> Slice {
        Slice {
            type_: PixelType::Float,
            base_offset: 0,
            x_stride: 4,
            y_stride: (width as usize) * 4,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        }
    }

    /// Full 2-part scanline round trip: write "left" (NONE) and "right" (ZIP),
    /// read back, assert numParts, per-part header, and pixel-exactness.
    #[test]
    fn two_part_scanline_round_trip() {
        const W: i32 = 4;
        const H: i32 = 5;

        let left = make_pixels(0, W, H);
        let right = make_pixels(1, W, H);

        let mut out = MemOStream::new();
        {
            let h0 = make_header("left", W, H, Compression::NoCompression);
            let h1 = make_header("right", W, H, Compression::Zip);
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
            assert_eq!(mp.parts(), 2);

            // Part 0 ("left"): write all chunks (cursor is at part 0's chunks).
            {
                let mut p = mp.output_part(0).expect("part 0");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 0");
                p.write_pixels(&left, H).expect("write 0");
            }
            // Part 1 ("right"): its chunks follow part 0's in the file.
            {
                let mut p = mp.output_part(1).expect("part 1");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 1");
                p.write_pixels(&right, H).expect("write 1");
            }
        }

        let bytes = out.into_data();

        // Version word must have the MULTI_PART flag and NOT the tiled flag.
        let version_word =
            i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u32;
        let flags = version::get_flags(version_word);
        assert!(version::is_multi_part(flags), "MULTI_PART flag must be set");
        assert!(!version::is_tiled(flags), "TILED flag must not be set");

        // --- read back -----------------------------------------------------
        let mut inp = MemIStream::new(bytes.clone());
        let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        assert_eq!(mp.parts(), 2, "numParts");
        assert_eq!(mp.part_type(0).unwrap(), SCANLINE_IMAGE);
        assert_eq!(mp.part_type(1).unwrap(), SCANLINE_IMAGE);
        assert_eq!(mp.header(0).unwrap().name().unwrap(), "left");
        assert_eq!(mp.header(1).unwrap().name().unwrap(), "right");
        // Dimensions correct (data window 0..W-1, 0..H-1).
        let dw0 = mp.header(0).unwrap().data_window().unwrap();
        assert_eq!((dw0.max.x - dw0.min.x + 1, dw0.max.y - dw0.min.y + 1), (W, H));

        for (part, expected) in [(0i32, &left), (1i32, &right)] {
            let mut p = mp.input_part(part).expect("input part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(W)).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let mut got = vec![0u8; (W * H) as usize * 4];
            p.read_pixels(0, H - 1, &mut got).expect("read");
            assert_eq!(got, *expected, "part {part} pixels must be bit-exact");
        }
    }

    /// Write + read a 2-part multipart file (both parts
    /// multi-chunk: "left" ZIP 16-line blocks, "right" NONE 1-line chunks)
    /// with `num_threads` in {0, 1, 2, 4, 8} on BOTH the writer and the
    /// reader, and assert the produced file bytes AND the read-back pixels
    /// are identical at every thread count — the multipart analogue of the
    /// scanline/tiled parallel-determinism tests, now that `num_threads`
    /// propagates from `MultiPartOutputFile`/`MultiPartInputFile` down to
    /// each part's writer/reader state.
    #[test]
    fn multipart_parallel_write_and_read_are_byte_identical_across_thread_counts() {
        const W: i32 = 23;
        const H: i32 = 64;

        let left = make_pixels(0, W, H);
        let right = make_pixels(1, W, H);

        let mut reference: Option<Vec<u8>> = None;
        for &n in &[0i32, 1, 2, 4, 8] {
            let h0 = make_header("left", W, H, Compression::Zip);
            let h1 = make_header("right", W, H, Compression::NoCompression);

            let mut out = MemOStream::new();
            crate::probe::reset(crate::probe::Probe::ScanWriteParallel);
            {
                let mut mp = MultiPartOutputFile::with_num_threads(&mut out, &[h0, h1], Some(n))
                    .expect("mp out");
                {
                    let mut p = mp.output_part(0).expect("part 0");
                    let mut fb = FrameBuffer::new();
                    fb.insert("Z", float_slice(W)).expect("slice");
                    p.set_frame_buffer(&fb).expect("set fb 0");
                    p.write_pixels(&left, H).expect("write 0");
                }
                {
                    let mut p = mp.output_part(1).expect("part 1");
                    let mut fb = FrameBuffer::new();
                    fb.insert("Z", float_slice(W)).expect("slice");
                    p.set_frame_buffer(&fb).expect("set fb 1");
                    p.write_pixels(&right, H).expect("write 1");
                }
            }
            let bytes = out.into_data();
            // The multipart num_threads must reach every part's writer
            // and reader; both parts are multi-chunk, so each engages.
            let part_writes = crate::probe::get(crate::probe::Probe::ScanWriteParallel);
            assert_eq!(part_writes, if n > 1 { 2 } else { 0 }, "num_threads={n}: parallel part writes");
            crate::probe::reset(crate::probe::Probe::ScanWriteParallel);

            // Read back at the SAME thread count and check pixel-exactness.
            let mut inp = MemIStream::new(bytes.clone());
            crate::probe::reset(crate::probe::Probe::ScanReadParallel);
            let mut mp = MultiPartInputFile::with_num_threads(&mut inp, Some(n)).expect("mp in");
            for (part, expected) in [(0i32, &left), (1i32, &right)] {
                let mut p = mp.input_part(part).expect("input part");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb");
                let mut got = vec![0u8; (W * H) as usize * 4];
                p.read_pixels(0, H - 1, &mut got).expect("read");
                assert_eq!(got, **expected, "num_threads={n}: part {part} pixels must be bit-exact");
            }
            let part_reads = crate::probe::get(crate::probe::Probe::ScanReadParallel);
            assert_eq!(part_reads, if n > 1 { 2 } else { 0 }, "num_threads={n}: parallel part reads");

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

    /// Ownership inversion (per-part writer state persists): writes INTERLEAVED
    /// across two scanline parts, with the first part's writer resumed
    /// MID-BLOCK, must produce a file both we and (conceptually) C++ read back
    /// correctly. `output_part(0)` used to build a brand-new `OutputFile` from
    /// the header every call, silently resetting `current_scan_line` /
    /// `missing_scan_lines` / the partial line-buffer block
    /// (`ImfMultiPartOutputFile.cpp:313-321` caches one `OutputFile` per part
    /// instead) — a second `output_part(0)` call would have restarted part 0
    /// from row 0, corrupting/duplicating the chunks already written for rows
    /// 0..5 instead of resuming at row 5.
    ///
    /// Sequence: part 0 rows 0..5 (Zip = 16 lines/block, so this leaves an
    /// UNFLUSHED partial block) -> drop that view -> part 1 rows 0..20
    /// (fully) -> drop -> part 0 AGAIN, rows 5..20 (completes the first
    /// block at row 15, then a final partial block for rows 16..19). Both
    /// parts must read back bit-exact.
    #[test]
    fn interleaved_writes_across_parts_resume_mid_block() {
        const W: i32 = 4;
        const H: i32 = 20;
        const SPLIT: i32 = 5;

        let left = make_pixels(0, W, H);
        let right = make_pixels(1, W, H);

        // The frame-buffer slice addresses pixels by their ABSOLUTE scanline
        // number (`base_offset + y * y_stride`, per `float_slice`'s
        // `base_offset: 0`) -- exactly the C++ convention (a `base` pointer
        // offset so that indexing by the real `y` lands correctly). So every
        // `write_pixels` call for this part passes the SAME full-size `left`
        // buffer; the writer only ever reads the rows it is currently
        // gathering (`current_scan_line`), whichever call that happens in.
        let mut out = MemOStream::new();
        {
            let h0 = make_header("left", W, H, Compression::Zip);
            let h1 = make_header("right", W, H, Compression::Zip);
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");

            // Part 0, rows 0..SPLIT: leaves block 0 (rows 0..15) partially
            // gathered (5 of 16 lines), unflushed.
            {
                let mut p = mp.output_part(0).expect("part 0 (first slice)");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 0a");
                p.write_pixels(&left, SPLIT).expect("write 0a");
            }
            // Part 1, all rows — its chunks land between part 0's partial
            // block and part 0's resumption.
            {
                let mut p = mp.output_part(1).expect("part 1");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 1");
                p.write_pixels(&right, H).expect("write 1");
            }
            // Part 0 AGAIN, rows SPLIT..H — must resume at row SPLIT (not
            // restart at row 0), completing block 0 then flushing the final
            // partial block.
            {
                let mut p = mp.output_part(0).expect("part 0 (resumed)");
                assert_eq!(
                    p.current_scan_line(),
                    SPLIT,
                    "the SAME persisted state must be handed back, resuming at row {SPLIT}"
                );
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 0b");
                p.write_pixels(&left, H - SPLIT).expect("write 0b");
            }
        }

        let bytes = out.into_data();

        let mut inp = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        assert_eq!(mp.parts(), 2, "numParts");
        for (part, expected) in [(0i32, &left), (1i32, &right)] {
            let mut p = mp.input_part(part).expect("input part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(W)).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let mut got = vec![0u8; (W * H) as usize * 4];
            p.read_pixels(0, H - 1, &mut got).expect("read");
            assert_eq!(
                got, *expected,
                "part {part} pixels must be bit-exact after interleaved, resumed writes"
            );
        }
    }

    /// The same for TILED parts: same interleave
    /// shape as [`interleaved_writes_across_parts_resume_mid_block`] above,
    /// but exercising the out-of-order tile buffering
    /// (`tile_map`/`next_tile_to_write`) instead of a partial scanline
    /// block. Sequence: part 0 writes tile `(1,0)` OUT OF ORDER (buffered,
    /// since `INCREASING_Y`'s first tile is `(0,0)`) -> drop that view ->
    /// part 1 writes all 4 tiles in order -> drop -> part 0 AGAIN writes
    /// tile `(0,0)`, which must flush the buffered `(1,0)` too -> then
    /// re-attempting to write `(1,0)` a THIRD time must error "more than
    /// once" (proves `tile_offsets` itself persisted, not just `tile_map` --
    /// a rebuild-from-header `TiledOutputFileState::from_part` would hand
    /// back a fresh all-zero `TileOffsets`, silently ACCEPTING the duplicate
    /// write instead of rejecting it).
    #[test]
    fn tiled_interleaved_writes_across_parts_flush_buffered_tile_on_resume() {
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
        use crate::attr::TileDescriptionAttribute;
        use crate::header::TILED_IMAGE;

        const W: i32 = 4;
        const H: i32 = 4;
        const TS: u32 = 2; // 2x2 tiles -> a 2x2 tile grid

        fn make_tiled_header(name: &str, comp: Compression) -> Header {
            let mut h = Header::with_dimensions(
                W,
                H,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                comp,
            )
            .expect("dims");
            h.set_name(name).expect("name");
            h.set_type(TILED_IMAGE).expect("type");
            let mut channels = ChannelList::new();
            channels
                .insert("Z", Channel::new(PixelType::Float, 1, 1, false))
                .expect("Z channel");
            h.insert("channels", ChannelListAttribute::new(channels))
                .expect("channels");
            h.insert(
                "tiles",
                TileDescriptionAttribute::new(TileDescription {
                    x_size: TS,
                    y_size: TS,
                    mode: LevelMode::OneLevel,
                    rounding_mode: LevelRoundingMode::RoundDown,
                }),
            )
            .expect("tiles");
            h
        }

        // One TSxTS-tile buffer, tile-coordinate-relative, filled with `v`.
        fn tile_buf(v: f32) -> Vec<u8> {
            let mut buf = Vec::with_capacity((TS * TS) as usize * 4);
            for _ in 0..(TS * TS) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf
        }

        fn tile_slice() -> Slice {
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 4,
                y_stride: (TS as usize) * 4,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: true,
                y_tile_coords: true,
            }
        }

        let mut out = MemOStream::new();
        {
            let h0 = make_tiled_header("left", Compression::Zip);
            let h1 = make_tiled_header("right", Compression::Zip);
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");

            // Part 0: tile (1,0) OUT OF ORDER -- buffered, not written yet.
            {
                let mut p = mp.tiled_output_part(0).expect("part 0 (first tile)");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", tile_slice()).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 0a");
                p.write_tile(&tile_buf(10.0), 1, 0, 0, 0).expect("write (1,0)");
            }
            // Part 1: all 4 tiles, in order -- lands between part 0's
            // buffered tile and part 0's resumption.
            {
                let mut p = mp.tiled_output_part(1).expect("part 1");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", tile_slice()).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 1");
                p.write_tile(&tile_buf(100.0), 0, 0, 0, 0).expect("(0,0)");
                p.write_tile(&tile_buf(101.0), 1, 0, 0, 0).expect("(1,0)");
                p.write_tile(&tile_buf(102.0), 0, 1, 0, 0).expect("(0,1)");
                p.write_tile(&tile_buf(103.0), 1, 1, 0, 0).expect("(1,1)");
            }
            // Part 0 AGAIN: write (0,0) -- must flush the buffered (1,0) too,
            // then continue with (0,1)/(1,1). If the persisted state were
            // lost, `tile_offsets` would come back all-zero and this whole
            // step would look like a normal first write (no error) even
            // though (1,0) is about to be written a second time below.
            {
                let mut p = mp.tiled_output_part(0).expect("part 0 (resumed)");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", tile_slice()).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb 0b");
                p.write_tile(&tile_buf(11.0), 0, 0, 0, 0).expect("(0,0)");
                p.write_tile(&tile_buf(12.0), 0, 1, 0, 0).expect("(0,1)");
                p.write_tile(&tile_buf(13.0), 1, 1, 0, 0).expect("(1,1)");

                // The load-bearing assertion: (1,0) was already flushed by
                // the (0,0) write above (it was the buffered tile waiting
                // for its turn), so writing it again must be rejected --
                // proving the SAME `tile_offsets` (not a fresh all-zero
                // table) survived the interleaved part access.
                let err = p.write_tile(&tile_buf(999.0), 1, 0, 0, 0).unwrap_err();
                assert!(
                    matches!(err, ExrError::Arg(ref m) if m.contains("more than once")),
                    "expected an 'already written' error, got {err:?}"
                );
            }
        }

        let bytes = out.into_data();

        let mut inp = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        assert_eq!(mp.parts(), 2, "numParts");

        // Part 0: (1,0) must read back as 10.0 (its ORIGINAL, only
        // successful write) -- not 999.0 (the rejected duplicate) and not
        // zero (which is what a silently-accepted duplicate write, or a lost
        // buffered tile, would have produced instead).
        {
            let mut p = mp.tiled_input_part(0).expect("input part 0");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", tile_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let mut got = vec![0u8; (TS * TS) as usize * 4];
            p.read_tile(1, 0, 0, 0, &mut got).expect("read (1,0)");
            assert_eq!(got, tile_buf(10.0), "part 0 tile (1,0) must be its original value");
        }

        // Part 1: full 2x2 grid, pixel-exact.
        {
            let mut p = mp.tiled_input_part(1).expect("input part 1");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", tile_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            for (dx, dy, v) in [(0, 0, 100.0), (1, 0, 101.0), (0, 1, 102.0), (1, 1, 103.0)] {
                let mut got = vec![0u8; (TS * TS) as usize * 4];
                p.read_tile(dx, dy, 0, 0, &mut got).expect("read");
                assert_eq!(got, tile_buf(v), "part 1 tile ({dx},{dy})");
            }
        }
    }

    /// A multipart file whose chunk offset tables were never written
    /// must still read — every part's table is rebuilt in one sweep over the
    /// chunk region (C++ `chunk.c::reconstruct_chunk_table`).
    #[test]
    fn zeroed_offset_tables_are_reconstructed() {
        const W: i32 = 4;
        const H: i32 = 5;

        let left = make_pixels(0, W, H);
        let right = make_pixels(1, W, H);

        let mut out = MemOStream::new();
        {
            let h0 = make_header("left", W, H, Compression::NoCompression);
            let h1 = make_header("right", W, H, Compression::NoCompression);
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
            for (part, px) in [(0i32, &left), (1i32, &right)] {
                let mut p = mp.output_part(part).expect("part");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb");
                p.write_pixels(px, H).expect("write");
            }
        }
        let mut bytes = out.into_data();

        // Locate both offset tables: they start right after the header list
        // and hold H entries per part (linesInBuffer == 1 for NONE).
        let table_pos = {
            let mut is = MemIStream::new(bytes.clone());
            let _magic = xdr::read_i32(&mut is).unwrap();
            let version_word = xdr::read_i32(&mut is).unwrap();
            let _h0 = Header::read_attributes_from(&mut is, version_word).unwrap();
            let _h1 = Header::read_attributes_from(&mut is, version_word).unwrap();
            let mut term = [0u8; 1];
            is.read(&mut term).unwrap();
            is.tellg().unwrap() as usize
        };
        let table_bytes = 2 * H as usize * 8;
        for b in bytes[table_pos..table_pos + table_bytes].iter_mut() {
            *b = 0;
        }

        let mut inp = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        for (part, expected) in [(0i32, &left), (1i32, &right)] {
            let mut p = mp.input_part(part).expect("input part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(W)).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let mut got = vec![0u8; (W * H) as usize * 4];
            p.read_pixels(0, H - 1, &mut got).expect("read");
            assert_eq!(got, *expected, "part {part} pixels after reconstruction");
        }
    }

    /// The leading `partNumber` is present and correct in each part's chunk
    /// bytes (and the offset tables are fully back-patched).
    #[test]
    fn chunk_prefix_carries_part_number() {
        const W: i32 = 3;
        const H: i32 = 2;

        let mut out = MemOStream::new();
        let h0 = make_header("a", W, H, Compression::NoCompression);
        let h1 = make_header("b", W, H, Compression::NoCompression);
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp");
            for part in [0i32, 1] {
                let mut p = mp.output_part(part).expect("part");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                let px = make_pixels(part, W, H);
                p.set_frame_buffer(&fb).expect("set fb");
                p.write_pixels(&px, H).expect("write");
            }
        }
        let bytes = out.into_data();

        // Recover each part's first-chunk offset straight from the on-disk
        // offset tables: skip magic + version, read both part headers + the
        // header-list terminator, then read the two offset tables in order.
        let mut is = MemIStream::new(bytes.clone());
        let _magic = xdr::read_i32(&mut is).unwrap();
        let version_word = xdr::read_i32(&mut is).unwrap();
        let _h0 = Header::read_attributes_from(&mut is, version_word).unwrap();
        let _h1 = Header::read_attributes_from(&mut is, version_word).unwrap();
        // header-list terminator (one extra empty name = a single 0 byte).
        let mut term = [0u8; 1];
        is.read(&mut term).unwrap();
        assert_eq!(term[0], 0, "header-list terminator must be an empty name");
        // Offset table 0 then offset table 1 (each H entries for NONE).
        let off0 = xdr::read_u64(&mut is).unwrap();
        for _ in 1..H {
            xdr::read_u64(&mut is).unwrap();
        }
        let off1 = xdr::read_u64(&mut is).unwrap();

        // Each part's first chunk must start with the part number (i32 LE),
        // then the scanline y (0).
        for (part, first_off) in [(0i32, off0), (1i32, off1)] {
            assert_ne!(first_off, 0, "part {part} offset table not back-patched");
            let o = first_off as usize;
            let pn = i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            assert_eq!(pn, part, "chunk leading partNumber for part {part}");
            let y = i32::from_le_bytes([
                bytes[o + 4],
                bytes[o + 5],
                bytes[o + 6],
                bytes[o + 7],
            ]);
            assert_eq!(y, 0, "scanline y after partNumber for part {part}");
        }
    }

    // -----------------------------------------------------------------------
    // Shared-attribute rules (write) + required-attribute rules (read)
    // `ImfMultiPartOutputFile.cpp::do_header_sanity_checks` /
    // `OpenEXRCore/validation.c::validate_req_attr`
    // -----------------------------------------------------------------------

    /// Give `h` a different displayWindow than `make_header` produces.
    fn with_display_window(mut h: Header, max: i32) -> Header {
        let dw = imath_rs::Box2i::from_min_max(
            imath_rs::V2i { x: 0, y: 0 },
            imath_rs::V2i { x: max, y: max },
        );
        h.insert("displayWindow", crate::attr::Box2iAttribute::new(dw))
            .expect("displayWindow");
        h
    }

    /// Write a 2-part file whose parts differ in a shared attribute.
    fn write_conflicting(override_shared: bool) -> Result<Vec<u8>, crate::error::ExrError> {
        const W: i32 = 4;
        const H: i32 = 2;
        let h0 = make_header("left", W, H, Compression::NoCompression);
        let h1 = with_display_window(make_header("right", W, H, Compression::NoCompression), 99);
        let mut out = MemOStream::new();
        {
            let mut mpo = MultiPartOutputFile::new_with_override(
                &mut out,
                &[h0, h1],
                override_shared,
            )?;
            let pixels = make_pixels(0, W, H);
            for p in 0..2 {
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W))?;
                let mut op = mpo.output_part(p)?;
                op.set_frame_buffer(&fb)?;
                op.write_pixels(&pixels, H)?;
            }
        }
        Ok(out.into_data())
    }

    /// With `overrideSharedAttributes == false` (the C++ default) a
    /// part whose `displayWindow` disagrees with part 0 is a hard error, and
    /// the message names the offending part and attribute.
    #[test]
    fn conflicting_shared_attributes_are_rejected() {
        let err = write_conflicting(false).expect_err("conflict must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("Conflicting attributes found for header")
                && msg.contains("right")
                && msg.contains("displayWindow"),
            "unexpected message: {msg}"
        );
    }

    /// With `overrideSharedAttributes == true` part 0's value is
    /// copied onto every other part instead (C++
    /// `overrideSharedAttributesValues`).
    #[test]
    fn override_shared_attributes_copies_part_zero() {
        let bytes = write_conflicting(true).expect("override must succeed");
        let mut is = MemIStream::new(bytes);
        let mpi = MultiPartInputFile::new(&mut is).expect("open");
        assert_eq!(
            mpi.header(0).unwrap().display_window().unwrap(),
            mpi.header(1).unwrap().display_window().unwrap()
        );
    }

    /// On READ, a multipart part with no `name` is fatal
    /// (`validation.c::validate_req_attr`: "'name' attribute for multipart file
    /// not found").
    #[test]
    fn multipart_part_without_name_is_fatal_on_read() {
        let bytes = two_part_bytes();
        // Occurrence 1 = part 1's `name` record (part 0's is occurrence 0).
        let patched = crate::chunk::offsets::testing::drop_attribute(&bytes, "name", 1);
        let mut is = MemIStream::new(patched);
        let Err(err) = MultiPartInputFile::new(&mut is) else {
            panic!("missing name must fail");
        };
        assert!(err.to_string().contains("'name' attribute"), "{err}");
    }

    /// Likewise for `type` — and `autoAddType` does NOT rescue it,
    /// because C++ validates before that loop ever runs.
    #[test]
    fn multipart_part_without_type_is_fatal_on_read() {
        let bytes = two_part_bytes();
        let patched = crate::chunk::offsets::testing::drop_attribute(&bytes, "type", 1);
        let mut is = MemIStream::new(patched);
        let Err(err) = MultiPartInputFile::new(&mut is) else {
            panic!("missing type must fail");
        };
        assert!(err.to_string().contains("'type' attribute"), "{err}");
    }

    /// `partComplete` is `false` when reconstruction cannot recover a
    /// chunk — C++ `exr_validate_chunk_table` re-checks the table *after*
    /// reconstruction, so a hole that survives the sweep is still a hole.
    #[test]
    fn part_complete_is_false_when_a_chunk_cannot_be_recovered() {
        const W: i32 = 4;
        const H: i32 = 3;
        let mut bytes = two_part_bytes_wh(W, H);
        // Drop the file's last chunk, then zero every offset so the reader has
        // to reconstruct: the sweep finds every chunk but the truncated one.
        let last_chunk = {
            let mut is = MemIStream::new(bytes.clone());
            let mpi = MultiPartInputFile::new(&mut is).expect("open");
            let mut all: Vec<u64> = Vec::new();
            for p in 0..mpi.parts() {
                all.extend(mpi.chunk_offsets(p).unwrap().iter().copied());
            }
            *all.iter().max().expect("chunks")
        };
        bytes.truncate(last_chunk as usize);
        let mut is = MemIStream::new(bytes);
        let mpi = MultiPartInputFile::new(&mut is).expect("open truncated");
        assert!(mpi.part_complete(0).unwrap(), "part 0 is intact");
        assert!(
            !mpi.part_complete(1).unwrap(),
            "part 1 lost its last chunk, so its table has a hole"
        );
    }

    /// `MultiPartOutputFile::new` with ONE header must produce a file
    /// byte-identical in *structure* to the single-part writer — no
    /// `partNumber` chunk prefix, and NO `MULTI_PART_FILE_FLAG` in the version
    /// word (C++ `OutputPartData(..., parts > 1)`,
    /// `ImfMultiPartOutputFile.cpp:185-186`/`:237-238`; `ImfOutputFile.cpp:745`
    /// `_data->multiPart = part->multipart`, `:256` prefix only `if
    /// (partdata->multiPart)`). Before the fix every chunk carried a leading
    /// `partNumber` (always 0) even though the version word said single-part,
    /// so `crate::scanline::InputFile::new` mis-parsed that 0 as the scanline
    /// `y` and everything after it as garbage.
    #[test]
    fn single_header_multipart_output_has_no_part_prefix_and_is_readable() {
        const W: i32 = 4;
        const H: i32 = 3;
        let h = make_header("only", W, H, Compression::NoCompression);
        let pixels = make_pixels(0, W, H);

        let mut out = MemOStream::new();
        {
            let mut mpo = MultiPartOutputFile::new(&mut out, &[h]).expect("open");
            assert_eq!(mpo.parts(), 1);
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(W)).expect("Z");
            let mut op = mpo.output_part(0).expect("part 0");
            op.set_frame_buffer(&fb).expect("fb");
            op.write_pixels(&pixels, H).expect("write");
        }
        let bytes = out.into_data();

        // Version word: no MULTI_PART_FILE_FLAG for a single header.
        let version_word = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u32;
        let flags = version::get_flags(version_word);
        assert!(
            !version::is_multi_part(flags),
            "a single-header file must not carry MULTI_PART_FILE_FLAG"
        );

        // Round-trips through the plain single-part reader (which never
        // expects a partNumber prefix) with the right pixels.
        let mut is = MemIStream::new(bytes.clone());
        let mut inf = crate::scanline::InputFile::new(&mut is).expect("InputFile::new");
        let mut in_fb = FrameBuffer::new();
        in_fb.insert("Z", float_slice(W)).expect("Z");
        inf.set_frame_buffer(&in_fb).expect("set fb");
        let mut got = vec![0u8; pixels.len()];
        inf.read_pixels(0, H - 1, &mut got).expect("read");
        assert_eq!(got, pixels);

        // Also round-trips through MultiPartInputFile::input_part(0).
        let mut is2 = MemIStream::new(bytes);
        let mut mpi = MultiPartInputFile::new(&mut is2).expect("mpi");
        assert_eq!(mpi.parts(), 1);
        let mut part = mpi.input_part(0).expect("part 0");
        let mut in_fb2 = FrameBuffer::new();
        in_fb2.insert("Z", float_slice(W)).expect("Z");
        part.set_frame_buffer(&in_fb2).expect("set fb");
        let mut got2 = vec![0u8; pixels.len()];
        part.read_pixels(0, H - 1, &mut got2).expect("read");
        assert_eq!(got2, pixels);
    }

    /// A single-header `MultiPartOutputFile` whose
    /// header carries no `type` must be REJECTED, matching C++
    /// `ImfGenericOutputFile.cpp::writeMagicNumberAndVersionField(os, headers,
    /// parts)`, which calls `headers[0].type()` unconditionally when `parts ==
    /// 1` (`Header::type()` throws `ArgExc` when the attribute is absent).
    /// This is stricter than the plain single-part `scanline::OutputFile::new`,
    /// which never requires `type` at all -- the requirement is specific to
    /// going through the multipart container, even for one part.
    #[test]
    fn single_header_multipart_output_without_type_is_rejected() {
        let mut h = Header::with_dimensions(
            4,
            2,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .expect("dims");
        let mut channels = ChannelList::new();
        channels
            .insert("Z", Channel::new(PixelType::Float, 1, 1, true))
            .expect("Z channel");
        h.insert("channels", ChannelListAttribute::new(channels))
            .expect("channels");
        // Deliberately no `set_type` / `set_name` -- a single-part image
        // header need not carry either for `scanline::OutputFile::new`.
        assert!(!h.has_type());

        let mut out = MemOStream::new();
        let err = match MultiPartOutputFile::new(&mut out, &[h]) {
            Ok(_) => panic!("a typeless single header must be rejected, not accepted"),
            Err(e) => e,
        };
        let _ = format!("{err}"); // just prove it is a real, displayable error
    }

    fn two_part_bytes() -> Vec<u8> {
        two_part_bytes_wh(4, 2)
    }

    fn two_part_bytes_wh(w: i32, h: i32) -> Vec<u8> {
        let h0 = make_header("left", w, h, Compression::NoCompression);
        let h1 = make_header("right", w, h, Compression::NoCompression);
        let mut out = MemOStream::new();
        {
            let mut mpo = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("open");
            for p in 0..2 {
                let pixels = make_pixels(p, w, h);
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(w)).expect("Z");
                let mut op = mpo.output_part(p).expect("part");
                op.set_frame_buffer(&fb).expect("fb");
                op.write_pixels(&pixels, h).expect("write");
            }
        }
        out.into_data()
    }

    /// Overwrite the `occurrence`-th (0-based) `displayWindow` attribute's
    /// 16-byte Box2i VALUE in place (same size, so no offset shifting is
    /// needed) — used to craft a file `MultiPartOutputFile::new` itself would
    /// refuse to write (it enforces the same shared-attribute agreement on
    /// write), so the only way to get disagreeing bytes on disk is to patch
    /// them in after a normal, non-conflicting write.
    fn patch_display_window(bytes: &mut [u8], occurrence: usize, new_box: (i32, i32, i32, i32)) {
        let pat = b"displayWindow\0";
        let start = bytes
            .windows(pat.len())
            .enumerate()
            .filter(|(_, w)| *w == pat)
            .map(|(i, _)| i)
            .nth(occurrence)
            .expect("displayWindow attribute not found");
        let mut p = start + pat.len();
        while bytes[p] != 0 {
            p += 1; // skip the type name ("box2i")
        }
        p += 1;
        let size = i32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        assert_eq!(size, 16, "displayWindow (Box2i) is always a 16-byte value");
        p += 4;
        bytes[p..p + 4].copy_from_slice(&new_box.0.to_le_bytes());
        bytes[p + 4..p + 8].copy_from_slice(&new_box.1.to_le_bytes());
        bytes[p + 8..p + 12].copy_from_slice(&new_box.2.to_le_bytes());
        bytes[p + 12..p + 16].copy_from_slice(&new_box.3.to_le_bytes());
    }

    /// `MultiPartInputFile::new` must reject a file
    /// whose parts disagree on a SHARED attribute — C++
    /// `parse_header.c:2855-2887` runs this check unconditionally right after
    /// parsing every header and fails the whole open on the first mismatch;
    /// there is no read-side override (unlike the writer's
    /// `overrideSharedAttributes`, already covered by
    /// `conflicting_shared_attributes_are_rejected` above for the WRITE
    /// path). Before this fix a multipart file with disagreeing
    /// `displayWindow`s opened fine and each part read its own dataWindow's
    /// pixels — a divergence C++ treats as a hard, un-openable file.
    #[test]
    fn multipart_input_rejects_disagreeing_shared_attributes() {
        let mut bytes = two_part_bytes_wh(4, 4);
        // Both parts currently share the same displayWindow; patch part 1's
        // (occurrence 1 — part 0's is occurrence 0) to a different box.
        patch_display_window(&mut bytes, 1, (0, 0, 7, 7));

        let mut is = MemIStream::new(bytes);
        let err = MultiPartInputFile::new(&mut is)
            .err()
            .expect("disagreeing displayWindow must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("non-conforming shared attributes") && msg.contains("displayWindow"),
            "unexpected error: {msg}"
        );
    }

    /// A part whose `type` this build does not
    /// recognise, and which carries no `chunkCount` attribute, must still
    /// open with 0 chunks -- C++ `OpenEXRCore/parse_header.c:2477-2491`
    /// returns `chunk_count > 0 ? chunk_count : 0` for `EXR_STORAGE_UNKNOWN`,
    /// never an error. The WRITE-side `getChunkOffsetTableSize`
    /// (`ImfMisc.cpp:1980-1987`) DOES throw for this same case, so
    /// `MultiPartOutputFile` cannot write this file directly -- crafted here
    /// by patching a normally-written file's part 1 `type` string and
    /// dropping its `chunkCount` attribute.
    #[test]
    fn unknown_part_type_without_chunk_count_opens_with_zero_chunks() {
        let bytes = two_part_bytes();
        let bytes =
            crate::chunk::offsets::testing::set_string_attribute(&bytes, "type", 1, "futurepart");
        let bytes = crate::chunk::offsets::testing::drop_attribute(&bytes, "chunkCount", 1);

        let mut is = MemIStream::new(bytes);
        let mpi =
            MultiPartInputFile::new(&mut is).expect("unknown-type part must still open, not fail");
        assert_eq!(mpi.parts(), 2);
        assert_eq!(mpi.part_type(1).unwrap(), "futurepart");
        assert_eq!(
            mpi.chunk_offsets(1).unwrap().len(),
            0,
            "no chunkCount -> zero chunks for an unsupported storage type"
        );
    }

    /// C++ rejects a typeless multipart header through
    /// `sanityCheck`'s OWN message ("Headers in a multipart file should have
    /// type attribute.", `ImfHeader.cpp:1019-1023`) -- run BEFORE
    /// `setChunkCount` (`ImfMultiPartOutputFile.cpp:108-113`) -- not a
    /// hand-rolled check with different wording and a different order.
    #[test]
    fn multipart_output_part_zero_without_type_uses_sanity_check_message() {
        let mut h0 = make_header("left", 4, 2, Compression::NoCompression);
        h0.erase("type").expect("erase type");
        let h1 = make_header("right", 4, 2, Compression::NoCompression);
        let mut out = MemOStream::new();
        let err = MultiPartOutputFile::new(&mut out, &[h0, h1])
            .err()
            .expect("typeless part 0 must be rejected");
        assert!(
            err.to_string()
                .contains("Headers in a multipart file should have type attribute."),
            "unexpected error: {err}"
        );
    }

    /// `overrideSharedAttributesValues` looks each
    /// shared attribute up with `findTypedAttribute<...>`, which returns
    /// `None` both when `src` lacks the name AND when `src` holds it under a
    /// *different* attribute type -- either way the fallback is "erase from
    /// dst", never "copy the foreign-typed value across" (which the old
    /// untyped `src.contains(name)` + `clone_attr()` implementation did).
    #[test]
    fn override_shared_attributes_erases_foreign_typed_source_attribute() {
        use crate::attr::{StringAttribute, TimeCode, TypedAttribute};

        let mut src = make_header("left", 4, 2, Compression::NoCompression);
        // src's "timecode" is stored under a FOREIGN attribute type (a plain
        // string), so `findTypedAttribute<TimeCodeAttribute>` on it returns
        // `None` in C++.
        src.insert("timecode", StringAttribute::new("not-a-timecode".into()))
            .expect("insert foreign-typed timecode");

        let mut dst = make_header("right", 4, 2, Compression::NoCompression);
        dst.insert("timecode", TypedAttribute::new(TimeCode::default()))
            .expect("insert real timecode");

        crate::multipart::output::override_shared_attributes_values(&src, &mut dst)
            .expect("override");

        assert!(
            !dst.contains("timecode"),
            "dst's timecode must be erased, not overwritten with src's foreign-typed value"
        );
    }

    /// A multipart SCANLINE part's own preview image
    /// must be updatable through `output_part`, not just a standalone
    /// writer's -- C++ `ImfMultiPartOutputFile.cpp:474-477` captures every
    /// part's real `previewPosition` while writing the header list. Writes a
    /// 2-part file where only part "right" carries a preview, updates it via
    /// `output_part(1)`, and asserts both (a) part 0 (no preview) still
    /// errors and (b) part 1's preview pixels changed on disk while its scan
    /// line pixels did not.
    #[test]
    fn multipart_output_part_preview_image_is_updatable() {
        use crate::attr::{PreviewImage, PreviewRgba};

        const W: i32 = 4;
        const H: i32 = 2;
        let h0 = make_header("left", W, H, Compression::NoCompression);
        let mut h1 = make_header("right", W, H, Compression::NoCompression);
        h1.set_preview_image(
            PreviewImage::new(2, 1, Some(vec![PreviewRgba::new(1, 2, 3, 4); 2])).unwrap(),
        )
        .unwrap();

        let left = make_pixels(0, W, H);
        let right = make_pixels(1, W, H);
        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
            for (part, px) in [(0i32, &left), (1i32, &right)] {
                let mut p = mp.output_part(part).expect("part");
                let mut fb = FrameBuffer::new();
                fb.insert("Z", float_slice(W)).expect("slice");
                p.set_frame_buffer(&fb).expect("set fb");
                p.write_pixels(px, H).expect("write");
            }

            // Part 0 has no preview -- must still error.
            let err = mp
                .output_part(0)
                .unwrap()
                .update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)])
                .unwrap_err();
            assert!(matches!(err, ExrError::Logic(_)), "{err:?}");

            // Part 1 has a preview -- must succeed.
            mp.output_part(1)
                .unwrap()
                .update_preview_image(&[PreviewRgba::new(9, 9, 9, 9), PreviewRgba::new(8, 8, 8, 8)])
                .expect("update part 1's preview");
        }
        let bytes = out.into_data();

        let mut inp = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        assert_eq!(
            mp.header(1).unwrap().preview_image().unwrap().pixels(),
            &[PreviewRgba::new(9, 9, 9, 9), PreviewRgba::new(8, 8, 8, 8)]
        );
        for (part, expected) in [(0i32, &left), (1i32, &right)] {
            let mut p = mp.input_part(part).expect("input part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(W)).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let mut got = vec![0u8; (W * H) as usize * 4];
            p.read_pixels(0, H - 1, &mut got).expect("read");
            assert_eq!(got, *expected, "part {part} pixels must survive the preview update");
        }
    }

    /// Tiled sibling of the scanline test above: a
    /// multipart TILED part's preview image must be updatable through
    /// `tiled_output_part`.
    #[test]
    fn multipart_tiled_output_part_preview_image_is_updatable() {
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
        use crate::attr::{PreviewImage, PreviewRgba, TileDescriptionAttribute};
        use crate::header::TILED_IMAGE;

        const W: i32 = 4;
        const H: i32 = 4;
        const TS: u32 = 2;

        fn make_tiled_header(name: &str) -> Header {
            let mut h = Header::with_dimensions(
                W,
                H,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::NoCompression,
            )
            .expect("dims");
            h.set_name(name).expect("name");
            h.set_type(TILED_IMAGE).expect("type");
            let mut channels = ChannelList::new();
            channels.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).expect("Z");
            h.insert("channels", ChannelListAttribute::new(channels)).expect("channels");
            h.insert(
                "tiles",
                TileDescriptionAttribute::new(TileDescription {
                    x_size: TS,
                    y_size: TS,
                    mode: LevelMode::OneLevel,
                    rounding_mode: LevelRoundingMode::RoundDown,
                }),
            )
            .expect("tiles");
            h
        }

        let mut h0 = make_tiled_header("only");
        h0.set_preview_image(
            PreviewImage::new(1, 1, Some(vec![PreviewRgba::new(1, 2, 3, 4)])).unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0]).expect("mp out");
            let mut p = mp.tiled_output_part(0).expect("part 0");
            let mut fb = FrameBuffer::new();
            fb.insert(
                "Z",
                Slice {
                    type_: PixelType::Float,
                    base_offset: 0,
                    x_stride: 4,
                    y_stride: (TS as usize) * 4,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: true,
                    y_tile_coords: true,
                },
            )
            .expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let tile = vec![0u8; (TS * TS) as usize * 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    p.write_tile(&tile, dx, dy, 0, 0).expect("write tile");
                }
            }
            p.update_preview_image(&[PreviewRgba::new(7, 7, 7, 7)])
                .expect("update preview");
        }
        let bytes = out.into_data();

        let mut inp = MemIStream::new(bytes);
        let mp = MultiPartInputFile::new(&mut inp).expect("mp in");
        assert_eq!(
            mp.header(0).unwrap().preview_image().unwrap().pixels(),
            &[PreviewRgba::new(7, 7, 7, 7)]
        );
    }

    /// `MultiPartInputFile::new_owned` opens a
    /// file from an OWNED stream (no caller-side `&mut` to keep alive) and
    /// reads pixel-identical to the borrowed `new` path -- the read-side
    /// counterpart of `scanline::OutputFileOwned`'s own byte-identity
    /// guarantee.
    #[test]
    fn new_owned_reads_identically_to_the_borrowed_constructor() {
        let bytes = two_part_bytes();

        let mut borrowed_is = MemIStream::new(bytes.clone());
        let mut borrowed = MultiPartInputFile::new(&mut borrowed_is).expect("borrowed open");

        let mut owned = MultiPartInputFile::new_owned(Box::new(MemIStream::new(bytes)))
            .expect("owned open");

        assert_eq!(borrowed.parts(), owned.parts());
        for part in 0..borrowed.parts() {
            assert_eq!(
                borrowed.header(part).unwrap().name(),
                owned.header(part).unwrap().name()
            );
            let mut fb = FrameBuffer::new();
            fb.insert("Z", float_slice(4)).expect("slice");
            let mut got_borrowed = vec![0u8; 4 * 2 * 4];
            let mut got_owned = vec![0u8; 4 * 2 * 4];
            borrowed
                .input_part(part)
                .unwrap()
                .set_frame_buffer(&fb)
                .unwrap();
            borrowed.input_part(part).unwrap().read_pixels(0, 1, &mut got_borrowed).unwrap();
            owned.input_part(part).unwrap().set_frame_buffer(&fb).unwrap();
            owned.input_part(part).unwrap().read_pixels(0, 1, &mut got_owned).unwrap();
            assert_eq!(got_borrowed, got_owned, "part {part} pixels must match");
        }
    }

    /// Interleaved-write coverage for the part-writer
    /// cache: the DEEP SCANLINE sibling of
    /// [`interleaved_writes_across_parts_resume_mid_block`] -- a deep
    /// scanline part's `current_scan_line`/`missing_scan_lines` counters
    /// (`deep::scanline_output::DeepScanLineOutputFileState`) are exactly
    /// the kind of persisted state that a rebuild-from-header
    /// `deep_scanline_output_part` would silently reset, since `write_pixels`
    /// derives the scanline `y` it writes purely from its OWN internal
    /// counter, never from caller-supplied data (verified by reading
    /// `deep/scanline_output.rs::write_pixels`'s body, not assumed).
    ///
    /// Sequence: part 0 row 0 only -> drop that view -> part 1 BOTH rows,
    /// fully -> drop -> part 0 AGAIN, its remaining row 1. If the persisted
    /// state were lost, the second `deep_scanline_output_part(0)` call would
    /// hand back a fresh state with `current_scan_line` reset to row 0, so
    /// this second write would silently re-write row 0's slot (a duplicate,
    /// not an error, since nothing else would object) and leave row 1's
    /// slot in the offset table at 0 forever -- caught below by asserting
    /// BOTH parts are fully complete and pixel-exact after the whole
    /// sequence, not just error-free.
    #[test]
    fn deep_scanline_interleaved_writes_across_parts_resume_mid_block() {
        use crate::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
        use crate::header::DEEP_SCANLINE;
        use imath_rs::Half;

        const W: i32 = 2;
        const H: i32 = 2;

        fn make_deep_header(name: &str) -> Header {
            let mut h = Header::with_dimensions(
                W,
                H,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::NoCompression,
            )
            .expect("dims");
            h.set_name(name).expect("name");
            h.set_type(DEEP_SCANLINE).expect("type");
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).expect("A channel");
            h.insert("channels", ChannelListAttribute::new(cl)).expect("channels");
            h
        }

        // One sample per pixel, value = part*1000 + pixel index.
        fn make_pixels(part: i32) -> (Vec<u8>, Vec<u8>, DeepFrameBuffer) {
            let n = (W * H) as usize;
            let mut sc = Vec::with_capacity(n * 4);
            for _ in 0..n {
                sc.extend_from_slice(&1u32.to_le_bytes());
            }
            let mut data = Vec::with_capacity(n * 2);
            for i in 0..n {
                let bits = Half::from((part * 1000 + i as i32) as f32).bits();
                data.extend_from_slice(&bits.to_le_bytes());
            }
            let mut fb = DeepFrameBuffer::new();
            fb.insert_sample_count_slice(Slice {
                type_: PixelType::Uint,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4 * W as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            })
            .expect("count slice");
            fb.insert(
                "A",
                DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                    .expect("deep slice"),
            )
            .expect("insert A");
            (sc, data, fb)
        }

        let h0 = make_deep_header("left");
        let h1 = make_deep_header("right");
        let (sc0, data0, fb0) = make_pixels(0);
        let (sc1, data1, fb1) = make_pixels(1);

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");

            // Part 0: row 0 only (leaves it mid-write, current_scan_line == 1).
            // Like the flat sibling test above, the SAME full-size buffers
            // are passed every call (the "per write call" convention
            // addresses by absolute row, not a per-call sub-slice); the
            // writer only gathers whichever row it is currently at.
            {
                let mut p = mp.deep_scanline_output_part(0).expect("part 0 (row 0)");
                p.set_frame_buffer(&fb0).expect("set fb 0a");
                p.write_pixels(&sc0, &data0, 1).expect("write part 0 row 0");
                assert_eq!(p.current_scan_line(), 1, "row 0 written, row 1 next");
            }
            // Part 1: both rows, fully -- lands between part 0's two writes.
            {
                let mut p = mp.deep_scanline_output_part(1).expect("part 1");
                p.set_frame_buffer(&fb1).expect("set fb 1");
                p.write_pixels(&sc1, &data1, H).expect("write part 1");
            }
            // Part 0 AGAIN: its remaining row 1. If the state were rebuilt,
            // `current_scan_line()` would read back 0, not 1.
            {
                let mut p = mp.deep_scanline_output_part(0).expect("part 0 (resumed)");
                assert_eq!(
                    p.current_scan_line(),
                    1,
                    "resuming part 0 must not reset its write cursor"
                );
                p.set_frame_buffer(&fb0).expect("set fb 0b");
                p.write_pixels(&sc0, &data0, 1).expect("write part 0 row 1");
            }
        }
        let bytes = out.into_data();

        // Both parts must be COMPLETE (every offset-table slot filled) and
        // pixel-exact -- a silently-duplicated row-0 write would leave row
        // 1's slot at 0 forever, failing `part_complete`.
        let mut is = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut is).expect("mp in");
        for (part, expected_sc, expected_data) in [(0i32, &sc0, &data0), (1i32, &sc1, &data1)] {
            assert!(mp.part_complete(part).unwrap(), "part {part} must be fully written");
            let mut p = mp.deep_scanline_input_part(part).expect("input part");
            let (_, xs, ys) = (0, 4usize, 4 * W as usize);
            let mut rfb = DeepFrameBuffer::new();
            rfb.insert_sample_count_slice(Slice {
                type_: PixelType::Uint,
                base_offset: 0,
                x_stride: xs,
                y_stride: ys,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            })
            .unwrap();
            rfb.insert(
                "A",
                DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                    .unwrap(),
            )
            .unwrap();
            p.set_frame_buffer(&rfb).unwrap();
            let mut got_sc = vec![0u8; expected_sc.len()];
            p.read_pixel_sample_counts(0, H - 1, &mut got_sc).unwrap();
            assert_eq!(got_sc, *expected_sc, "part {part} sample counts");
            let mut got_data = vec![0u8; expected_data.len()];
            p.read_pixels(0, H - 1, &got_sc, &mut got_data).unwrap();
            assert_eq!(got_data, *expected_data, "part {part} sample data");
        }
    }

    /// Interleaved-write coverage for the part-writer
    /// cache: the DEEP TILED sibling of
    /// [`tiled_interleaved_writes_across_parts_flush_buffered_tile_on_resume`]
    /// -- exercises the out-of-order tile buffering
    /// (`deep::tiled_output::DeepTiledOutputFileState::tile_map`/
    /// `next_tile_to_write`) surviving a `deep_tiled_output_part(n)` call
    /// interleaved with another part's writes, the deep-tiled analogue of
    /// the flat-tiled persistence.
    ///
    /// Sequence: part 0 writes tile `(1,0)` OUT OF ORDER (buffered, since
    /// `(0,0)` hasn't been written yet) -> part 1 writes all 4 tiles, fully
    /// -> part 0 AGAIN writes `(0,0)` (must flush the buffered `(1,0)` too)
    /// then `(0,1)`/`(1,1)` to complete. If the persisted state were lost,
    /// the second `deep_tiled_output_part(0)` call would hand back a FRESH
    /// `tile_map`/offset table, silently discarding the already-buffered
    /// `(1,0)` tile and leaving it (and therefore the whole part) forever
    /// incomplete -- caught below by asserting full completion and
    /// pixel-exactness for both parts, not just error-free calls.
    #[test]
    fn deep_tiled_interleaved_writes_across_parts_flush_buffered_tile_on_resume() {
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
        use crate::attr::TileDescriptionAttribute;
        use crate::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
        use crate::header::DEEP_TILE;
        use imath_rs::Half;

        const W: i32 = 4;
        const H: i32 = 4;
        const TS: u32 = 2; // 2x2 tiles -> a 2x2 tile grid

        fn make_deep_tiled_header(name: &str) -> Header {
            let mut h = Header::with_dimensions(
                W,
                H,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                Compression::NoCompression,
            )
            .expect("dims");
            h.set_name(name).expect("name");
            h.set_type(DEEP_TILE).expect("type");
            let mut cl = ChannelList::new();
            cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).expect("A channel");
            h.insert("channels", ChannelListAttribute::new(cl)).expect("channels");
            h.insert(
                "tiles",
                TileDescriptionAttribute::new(TileDescription {
                    x_size: TS,
                    y_size: TS,
                    mode: LevelMode::OneLevel,
                    rounding_mode: LevelRoundingMode::RoundDown,
                }),
            )
            .expect("tiles");
            h
        }

        // One sample per pixel over the WHOLE window, value = part*1000 +
        // pixel index -- like the flat tiled test's `tile_buf`, but the deep
        // writer's `write_tile` addresses the FULL data-window buffer by
        // absolute pixel position (same "per write call" convention as
        // the deep scanline test above), not a tile-local one.
        fn make_pixels(part: i32) -> (Vec<u8>, Vec<u8>, DeepFrameBuffer) {
            let n = (W * H) as usize;
            let mut sc = Vec::with_capacity(n * 4);
            for _ in 0..n {
                sc.extend_from_slice(&1u32.to_le_bytes());
            }
            let mut data = Vec::with_capacity(n * 2);
            for i in 0..n {
                let bits = Half::from((part * 1000 + i as i32) as f32).bits();
                data.extend_from_slice(&bits.to_le_bytes());
            }
            let mut fb = DeepFrameBuffer::new();
            fb.insert_sample_count_slice(Slice {
                type_: PixelType::Uint,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4 * W as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            })
            .expect("count slice");
            fb.insert(
                "A",
                DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                    .expect("deep slice"),
            )
            .expect("insert A");
            (sc, data, fb)
        }

        let h0 = make_deep_tiled_header("left");
        let h1 = make_deep_tiled_header("right");
        let (sc0, data0, fb0) = make_pixels(0);
        let (sc1, data1, fb1) = make_pixels(1);

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");

            // Part 0: tile (1,0) OUT OF ORDER -- buffered, not written yet.
            {
                let mut p = mp.deep_tiled_output_part(0).expect("part 0 (first tile)");
                p.set_frame_buffer(&fb0).expect("set fb 0a");
                p.write_tile(&sc0, &data0, 1, 0, 0, 0).expect("write (1,0)");
            }
            // Part 1: all 4 tiles, in order -- lands between part 0's
            // buffered tile and part 0's resumption.
            {
                let mut p = mp.deep_tiled_output_part(1).expect("part 1");
                p.set_frame_buffer(&fb1).expect("set fb 1");
                p.write_tile(&sc1, &data1, 0, 0, 0, 0).expect("(0,0)");
                p.write_tile(&sc1, &data1, 1, 0, 0, 0).expect("(1,0)");
                p.write_tile(&sc1, &data1, 0, 1, 0, 0).expect("(0,1)");
                p.write_tile(&sc1, &data1, 1, 1, 0, 0).expect("(1,1)");
            }
            // Part 0 AGAIN: write (0,0) -- must flush the buffered (1,0)
            // too, then continue with (0,1)/(1,1). If the persisted state
            // were lost, `tile_offsets` would come back all-zero and this
            // whole part would look freshly started, permanently losing the
            // (1,0) tile's bytes.
            {
                let mut p = mp.deep_tiled_output_part(0).expect("part 0 (resumed)");
                p.set_frame_buffer(&fb0).expect("set fb 0b");
                p.write_tile(&sc0, &data0, 0, 0, 0, 0).expect("(0,0)");
                p.write_tile(&sc0, &data0, 0, 1, 0, 0).expect("(0,1)");
                p.write_tile(&sc0, &data0, 1, 1, 0, 0).expect("(1,1)");
            }
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mut mp = MultiPartInputFile::new(&mut is).expect("mp in");
        for (part, expected_sc, expected_data) in [(0i32, &sc0, &data0), (1i32, &sc1, &data1)] {
            assert!(mp.part_complete(part).unwrap(), "part {part} must be fully written");
            let mut p = mp.deep_tiled_input_part(part).expect("input part");
            let mut rfb = DeepFrameBuffer::new();
            rfb.insert_sample_count_slice(Slice {
                type_: PixelType::Uint,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4 * W as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            })
            .unwrap();
            rfb.insert(
                "A",
                DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                    .unwrap(),
            )
            .unwrap();
            p.set_frame_buffer(&rfb).unwrap();
            let mut got_sc = vec![0u8; expected_sc.len()];
            p.read_pixel_sample_counts(0, 1, 0, 1, 0, 0, &mut got_sc).unwrap();
            assert_eq!(got_sc, *expected_sc, "part {part} sample counts");
            let mut got_data = vec![0u8; expected_data.len()];
            p.read_tiles(0, 1, 0, 1, 0, 0, &got_sc, &mut got_data).unwrap();
            assert_eq!(got_data, *expected_data, "part {part} sample data");
        }
    }

    /// Deep scanline sibling of
    /// `multipart_output_part_preview_image_is_updatable`: a multipart
    /// DEEP SCANLINE part's own preview image must be updatable through
    /// `deep_scanline_output_part`, now that
    /// `DeepScanLineOutputFileState::from_part_with_preview` threads the
    /// container's captured `preview_position` through the same way the
    /// flat scanline/tiled writers already do.
    #[test]
    fn multipart_deep_scanline_output_part_preview_image_is_updatable() {
        use crate::attr::{PreviewImage, PreviewRgba};
        use crate::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
        use crate::header::DEEP_SCANLINE;
        use imath_rs::Half;

        const W: i32 = 2;
        const H: i32 = 2;

        let mut h0 = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h0.set_name("only").unwrap();
        h0.set_type(DEEP_SCANLINE).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h0.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h0.set_preview_image(
            PreviewImage::new(1, 1, Some(vec![PreviewRgba::new(1, 2, 3, 4)])).unwrap(),
        )
        .unwrap();

        let n = (W * H) as usize;
        let mut sc = Vec::with_capacity(n * 4);
        for _ in 0..n {
            sc.extend_from_slice(&1u32.to_le_bytes());
        }
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&Half::from(i as f32).bits().to_le_bytes());
        }
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 4 * W as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        fb.insert(
            "A",
            DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0]).expect("mp out");
            let mut p = mp.deep_scanline_output_part(0).expect("part 0");
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_pixels(&sc, &data, H).expect("write");
            p.update_preview_image(&[PreviewRgba::new(9, 9, 9, 9)])
                .expect("update preview");
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mp = MultiPartInputFile::new(&mut is).expect("mp in");
        assert_eq!(
            mp.header(0).unwrap().preview_image().unwrap().pixels(),
            &[PreviewRgba::new(9, 9, 9, 9)]
        );
    }

    /// DEEP TILED sibling of the preview-image test above.
    #[test]
    fn multipart_deep_tiled_output_part_preview_image_is_updatable() {
        use crate::attr::tile_description::{LevelMode, LevelRoundingMode, TileDescription};
        use crate::attr::{PreviewImage, PreviewRgba, TileDescriptionAttribute};
        use crate::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
        use crate::header::DEEP_TILE;
        use imath_rs::Half;

        const W: i32 = 2;
        const H: i32 = 2;
        const TS: u32 = 2;

        let mut h0 = Header::with_dimensions(
            W,
            H,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        h0.set_name("only").unwrap();
        h0.set_type(DEEP_TILE).unwrap();
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h0.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h0.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: TS,
                y_size: TS,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();
        h0.set_preview_image(
            PreviewImage::new(1, 1, Some(vec![PreviewRgba::new(1, 2, 3, 4)])).unwrap(),
        )
        .unwrap();

        let n = (W * H) as usize;
        let mut sc = Vec::with_capacity(n * 4);
        for _ in 0..n {
            sc.extend_from_slice(&1u32.to_le_bytes());
        }
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&Half::from(i as f32).bits().to_le_bytes());
        }
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 4 * W as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        fb.insert(
            "A",
            DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, 2, 1, 1, 0.0, false, false)
                .unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0]).expect("mp out");
            let mut p = mp.deep_tiled_output_part(0).expect("part 0");
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_tile(&sc, &data, 0, 0, 0, 0).expect("write tile");
            p.update_preview_image(&[PreviewRgba::new(7, 7, 7, 7)])
                .expect("update preview");
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let mp = MultiPartInputFile::new(&mut is).expect("mp in");
        assert_eq!(
            mp.header(0).unwrap().preview_image().unwrap().pixels(),
            &[PreviewRgba::new(7, 7, 7, 7)]
        );
    }
}
