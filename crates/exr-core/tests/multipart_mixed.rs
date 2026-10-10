//! Mixed-type multi-part round-trip through the Part wrappers.
//!
//! Builds ONE multipart EXR holding four parts of different types:
//!
//! | part | type           | codec | wrapper used                 |
//! |------|----------------|-------|------------------------------|
//! | 0    | scanlineimage  | NONE  | `OutputPart` / `InputPart`   |
//! | 1    | tiledimage     | ZIP   | `TiledOutputPart`/`InputPart`|
//! | 2    | deepscanline   | ZIPS  | `DeepScanLine*Part`          |
//! | 3    | deeptile       | ZIPS  | `DeepTiled*Part`             |
//!
//! Each part is written through its `*OutputPart` wrapper and read back through
//! its `*InputPart` wrapper, asserting every part's pixels / counts / samples
//! are bit-exact.  It also confirms each part's first on-disk chunk carries the
//! correct leading `partNumber` (the unified seam threaded through the tiled +
//! deep chunk paths), and that the version word has the MULTI_PART flag.
//!
//! This is the multipart correctness gate: it exercises the partNumber seam in the
//! scanline, tiled, deep-scanline AND deep-tiled chunk writers/readers via the
//! thin Part wrappers, with NONE + ZIP/ZIPS codecs.

use exr_core::attr::tile_description::{LevelMode, LevelRoundingMode};
use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType,
    TileDescription, TileDescriptionAttribute,
};
use exr_core::container::{version, xdr};
use exr_core::deep::frame_buffer::DeepSlice;
use exr_core::deep::{
    DeepFrameBuffer, DeepScanLineInputFile, DeepScanLineOutputFile, DeepTiledInputFile,
    DeepTiledOutputFile,
};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{
    Header, DEEP_SCANLINE, DEEP_TILE, SCANLINE_IMAGE, TILED_IMAGE,
};
use exr_core::io::{IStream, MemIStream, MemOStream};
use exr_core::misc::pixel_type_size;
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile};
use imath_rs::{Half, V2f};

// Common image geometry shared by all four parts so the offset-table maths is
// easy to reason about.
const W: i32 = 4;
const H: i32 = 4;

// ---------------------------------------------------------------------------
// Header builders (one HALF "Z" channel each, named uniquely per the multipart
// uniqueness rule).
// ---------------------------------------------------------------------------

fn base_header(name: &str, comp: Compression, line_order: LineOrder) -> Header {
    let mut h = Header::with_dimensions(
        W,
        H,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        line_order,
        comp,
    )
    .expect("dims");
    h.set_name(name).expect("name");
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 1, 1, false))
        .expect("Z");
    h.insert("channels", ChannelListAttribute::new(cl))
        .expect("channels");
    h
}

fn scanline_header() -> Header {
    let mut h = base_header("flat-scan", Compression::NoCompression, LineOrder::IncreasingY);
    h.set_type(SCANLINE_IMAGE).expect("type");
    h
}

fn tiled_header() -> Header {
    let mut h = base_header("flat-tile", Compression::Zip, LineOrder::IncreasingY);
    h.set_type(TILED_IMAGE).expect("type");
    h.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .expect("tiles");
    h
}

fn deep_scanline_header() -> Header {
    let mut h = base_header("deep-scan", Compression::Zips, LineOrder::IncreasingY);
    h.set_type(DEEP_SCANLINE).expect("type");
    h
}

fn deep_tiled_header() -> Header {
    let mut h = base_header("deep-tile", Compression::Zips, LineOrder::IncreasingY);
    h.set_type(DEEP_TILE).expect("type");
    h.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .expect("tiles");
    h
}

// ---------------------------------------------------------------------------
// Flat pixel helpers (scanline + tiled).  value(x,y) = half(part*100 + y*W + x).
// ---------------------------------------------------------------------------

fn flat_pixels(part: i32) -> Vec<u8> {
    let row_bytes = W as usize * 2;
    let mut buf = vec![0u8; row_bytes * H as usize];
    for y in 0..H {
        for x in 0..W {
            let off = y as usize * row_bytes + x as usize * 2;
            let v = Half::from((part * 100 + y * W + x) as f32).bits();
            buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

fn flat_slice() -> Slice {
    Slice {
        type_: PixelType::Half,
        base_offset: 0,
        x_stride: 2,
        y_stride: W as usize * 2,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    }
}

// ---------------------------------------------------------------------------
// Deep helpers (deep-scanline + deep-tiled).  One HALF "Z" channel; per-pixel
// counts vary; sample value = part*1000 + pixel*10 + s.
// ---------------------------------------------------------------------------

fn deep_counts() -> Vec<u32> {
    // 16 pixels (4x4), variable counts.
    vec![0, 1, 2, 3, 1, 0, 2, 1, 3, 1, 0, 2, 1, 1, 2, 0]
}

/// Flat sample-count buffer (u32 LE per pixel), absolute addressing
/// (xStride=4, yStride=4*W, base 0).
fn build_count_buffer(counts: &[u32]) -> Vec<u8> {
    let y_stride = 4 * W;
    let mut sc = vec![0u8; (H * y_stride) as usize];
    for (i, &c) in counts.iter().enumerate() {
        let off = i * 4;
        sc[off..off + 4].copy_from_slice(&c.to_le_bytes());
    }
    sc
}

/// Flat sample heap (single HALF channel, contiguous in raster order).
fn build_sample_heap(part: i32, counts: &[u32]) -> Vec<u8> {
    let mut heap = Vec::new();
    let base = part as f64 * 1000.0;
    for (pix, &c) in counts.iter().enumerate() {
        for s in 0..c {
            let v = base + pix as f64 * 10.0 + s as f64;
            heap.extend_from_slice(&Half::from(v as f32).bits().to_le_bytes());
        }
    }
    heap
}

/// Deep frame buffer: sample-count slice + one HALF "Z" deep slice at base 0.
fn deep_frame_buffer() -> DeepFrameBuffer {
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
    let ss = pixel_type_size(PixelType::Half).expect("size");
    fb.insert(
        "Z",
        DeepSlice::new(PixelType::Half, 0, 2, 2 * W as usize, ss, 1, 1, 0.0, false, false)
            .expect("Z deep slice"),
    )
    .expect("Z deep slice");
    fb
}

/// The 4-part mixed file (scanline NONE, tiled ZIP, deep scanline ZIPS, deep tiled ZIPS), written
/// through the `*OutputPart` wrappers.
fn mixed_file() -> Vec<u8> {
    let scan_px = flat_pixels(0);
    let tile_px = flat_pixels(1);
    let dscan_counts = deep_counts();
    let dscan_sc = build_count_buffer(&dscan_counts);
    let dscan_heap = build_sample_heap(2, &dscan_counts);
    let dtile_counts = deep_counts();
    let dtile_sc = build_count_buffer(&dtile_counts);
    let dtile_heap = build_sample_heap(3, &dtile_counts);

    // --- write the 4-part file via the *OutputPart wrappers ----------------
    let mut out = MemOStream::new();
    {
        let headers = [
            scanline_header(),
            tiled_header(),
            deep_scanline_header(),
            deep_tiled_header(),
        ];
        let mut mp = MultiPartOutputFile::new(&mut out, &headers).expect("mp out");
        assert_eq!(mp.parts(), 4);

        // Part 0: scanline (NONE) via OutputPart.
        {
            let mut p = mp.output_part_wrapper(0).expect("scan part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", flat_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_pixels(&scan_px, H).expect("write scan");
        }
        // Part 1: tiled (ZIP) via TiledOutputPart.
        {
            let mut p = mp.tiled_output_part_wrapper(1).expect("tile part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", flat_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let nx = p.num_x_tiles(0).expect("nx");
            let ny = p.num_y_tiles(0).expect("ny");
            p.write_tiles(&tile_px, 0, nx - 1, 0, ny - 1, 0, 0).expect("write tiles");
        }
        // Part 2: deep scanline (ZIPS) via DeepScanLineOutputPart.
        {
            let mut p = mp.deep_scanline_output_part_wrapper(2).expect("dscan part");
            let fb = deep_frame_buffer();
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_pixels(&dscan_sc, &dscan_heap, H).expect("write deep scan");
        }
        // Part 3: deep tiled (ZIPS) via DeepTiledOutputPart.
        {
            let mut p = mp.deep_tiled_output_part_wrapper(3).expect("dtile part");
            let fb = deep_frame_buffer();
            p.set_frame_buffer(&fb).expect("set fb");
            let nx = p.num_x_tiles(0).expect("nx");
            let ny = p.num_y_tiles(0).expect("ny");
            p.write_tiles(&dtile_sc, &dtile_heap, 0, nx - 1, 0, ny - 1, 0, 0).expect("write deep tiles");
        }
    }
    out.into_data()
}

#[test]
fn mixed_multipart_round_trip_via_part_wrappers() {
    let scan_px = flat_pixels(0);
    let tile_px = flat_pixels(1);
    let dscan_counts = deep_counts();
    let dscan_sc = build_count_buffer(&dscan_counts);
    let dscan_heap = build_sample_heap(2, &dscan_counts);
    let dtile_counts = deep_counts();
    let dtile_sc = build_count_buffer(&dtile_counts);
    let dtile_heap = build_sample_heap(3, &dtile_counts);
    let bytes = mixed_file();

    // Version word must carry the MULTI_PART flag (and NOT the single-part
    // tiled flag — multipart parts identify their type via the header attr).
    let version_word =
        i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u32;
    let flags = version::get_flags(version_word);
    assert!(version::is_multi_part(flags), "MULTI_PART flag must be set");
    assert!(!version::is_tiled(flags), "single-part TILED flag must NOT be set");

    // --- confirm each part's first chunk starts with its partNumber --------
    // Walk the prologue: magic + version, 4 headers, the header-list
    // terminator, then the 4 per-part offset tables (sized by chunkCount).
    {
        let mut is = MemIStream::new(bytes.clone());
        let _magic = xdr::read_i32(&mut is).unwrap();
        let v = xdr::read_i32(&mut is).unwrap();
        let mut chunk_counts = Vec::new();
        for _ in 0..4 {
            let h = Header::read_attributes_from(&mut is, v).unwrap();
            chunk_counts.push(h.chunk_count().unwrap());
        }
        // header-list terminator (one empty name = a single 0 byte).
        let mut term = [0u8; 1];
        is.read(&mut term).unwrap();
        assert_eq!(term[0], 0, "header-list terminator");

        // First offset of each part's table = first chunk position; assert the
        // chunk there begins with the matching partNumber (i32 LE).
        for (part, &cc) in chunk_counts.iter().enumerate() {
            let first_off = xdr::read_u64(&mut is).unwrap();
            // skip the rest of this part's table.
            for _ in 1..cc {
                xdr::read_u64(&mut is).unwrap();
            }
            assert_ne!(first_off, 0, "part {part} offset table not back-patched");
            let o = first_off as usize;
            let pn = i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            assert_eq!(pn as usize, part, "leading partNumber for part {part}");
        }
    }

    // --- read the 4 parts back via the *InputPart wrappers -----------------
    let mut inp = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut inp).expect("mp in");
    assert_eq!(mp.parts(), 4, "numParts");
    assert_eq!(mp.part_type(0).unwrap(), SCANLINE_IMAGE);
    assert_eq!(mp.part_type(1).unwrap(), TILED_IMAGE);
    assert_eq!(mp.part_type(2).unwrap(), DEEP_SCANLINE);
    assert_eq!(mp.part_type(3).unwrap(), DEEP_TILE);

    // Part 0: scanline.
    {
        let mut p = mp.input_part_wrapper(0).expect("scan in");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        let mut got = vec![0u8; (W * H) as usize * 2];
        p.read_pixels(0, H - 1, &mut got).expect("read scan");
        assert_eq!(got, scan_px, "scanline part pixels bit-exact");
    }
    // Part 1: tiled.
    {
        let mut p = mp.tiled_input_part_wrapper(1).expect("tile in");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        let nx = p.num_x_tiles(0).expect("nx");
        let ny = p.num_y_tiles(0).expect("ny");
        let mut got = vec![0u8; (W * H) as usize * 2];
        p.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut got).expect("read tiles");
        assert_eq!(got, tile_px, "tiled part pixels bit-exact");
    }
    // Part 2: deep scanline (two-pass).
    {
        let mut p = mp.deep_scanline_input_part_wrapper(2).expect("dscan in");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        let mut counts_buf = vec![0u8; dscan_sc.len()];
        p.read_pixel_sample_counts(0, H - 1, &mut counts_buf).expect("counts");
        assert_eq!(counts_buf, dscan_sc, "deep-scanline counts bit-exact");
        let mut heap = vec![0u8; dscan_heap.len()];
        p.read_pixels(0, H - 1, &counts_buf, &mut heap).expect("samples");
        assert_eq!(heap, dscan_heap, "deep-scanline samples bit-exact");
    }
    // Part 3: deep tiled (two-pass).
    {
        let mut p = mp.deep_tiled_input_part_wrapper(3).expect("dtile in");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        let nx = p.num_x_tiles(0).expect("nx");
        let ny = p.num_y_tiles(0).expect("ny");
        let mut counts_buf = vec![0u8; dtile_sc.len()];
        p.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts_buf)
            .expect("counts");
        assert_eq!(counts_buf, dtile_sc, "deep-tiled counts bit-exact");
        let mut heap = vec![0u8; dtile_heap.len()];
        p.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &counts_buf, &mut heap)
            .expect("samples");
        assert_eq!(heap, dtile_heap, "deep-tiled samples bit-exact");
    }
}

/// `MultiPartInputFile::deep_scanline_input_part(0)` reading a genuinely
/// SINGLE-part deep scanline file (written by the plain `DeepScanLineOutputFile`,
/// no `partNumber` chunk prefix) must read every line's counts and samples
/// identically to the dedicated `DeepScanLineInputFile` reading the same bytes
/// — proving the multipart container's single-part path (no `partNumber`
/// prefix expected) is byte-for-byte equivalent to the direct reader.
/// `multipart::copy_part` moves every part kind as raw chunks: copying all four parts of the mixed
/// file into a writer with the same headers reproduces every part's chunks byte for byte (the
/// headers differ only by the `version` attribute the reader adds to flat parts).
#[test]
fn copy_part_copies_every_part_kind_byte_identically() {
    let src = mixed_file();
    let mut is = MemIStream::new(src.clone());
    let mut input = MultiPartInputFile::new(&mut is).expect("mp in");
    let headers: Vec<Header> = (0..input.parts()).map(|p| input.header(p).expect("header").clone()).collect();
    let mut out = MemOStream::new();
    {
        let mut output = MultiPartOutputFile::new(&mut out, &headers).expect("mp out");
        for p in 0..input.parts() {
            exr_core::multipart::copy_part(&mut input, &mut output, p, p).expect("copy_part");
        }
    }
    // The chunk region: from the first chunk (the first offset-table entry, right after the
    // header list) to the end of the file. `RawImage` does not frame deep chunks, so compare the
    // bytes of the whole region.
    let region = |bytes: &[u8]| {
        let mut is = MemIStream::new(bytes.to_vec());
        let _magic = xdr::read_i32(&mut is).unwrap();
        let v = xdr::read_i32(&mut is).unwrap();
        for _ in 0..4 {
            Header::read_attributes_from(&mut is, v).unwrap();
        }
        let mut term = [0u8; 1];
        is.read(&mut term).unwrap();
        let first = xdr::read_u64(&mut is).unwrap() as usize;
        bytes[first..].to_vec()
    };
    let copy = out.into_data();
    let (a, b) = (region(&src), region(&copy));
    assert!(a.len() > 64, "the region holds the four parts' chunks");
    assert_eq!(a, b, "raw part copy must reproduce every part's chunks byte for byte");
}

#[test]
fn multipart_input_file_reads_single_part_deep_scanline() {
    let counts = deep_counts();
    let sc = build_count_buffer(&counts);
    let heap = build_sample_heap(0, &counts);
    let fb = deep_frame_buffer();

    let header = deep_scanline_header();
    let mut out = MemOStream::new();
    {
        let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        df.write_pixels(&sc, &heap, H).unwrap();
    }
    let bytes = out.into_data();

    // Reference: the plain single-part reader.
    let mut is_direct = MemIStream::new(bytes.clone());
    let mut direct = DeepScanLineInputFile::new(&mut is_direct).unwrap();
    direct.set_frame_buffer(&fb).unwrap();
    let mut direct_counts = vec![0u8; sc.len()];
    direct.read_pixel_sample_counts(0, H - 1, &mut direct_counts).unwrap();
    let mut direct_heap = vec![0u8; heap.len()];
    direct.read_pixels(0, H - 1, &direct_counts, &mut direct_heap).unwrap();

    // Under test: the same bytes via MultiPartInputFile's single-part path.
    let mut is_mp = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_mp).unwrap();
    let mut part = mp.deep_scanline_input_part(0).unwrap();
    part.set_frame_buffer(&fb).unwrap();
    let mut mp_counts = vec![0u8; sc.len()];
    part.read_pixel_sample_counts(0, H - 1, &mut mp_counts).unwrap();
    let mut mp_heap = vec![0u8; heap.len()];
    part.read_pixels(0, H - 1, &mp_counts, &mut mp_heap).unwrap();

    assert_eq!(mp_counts, direct_counts, "counts must match the direct reader");
    assert_eq!(mp_heap, direct_heap, "samples must match the direct reader");
    assert_eq!(mp_counts, sc, "counts must match the source data");
    assert_eq!(mp_heap, heap, "samples must match the source data");
}

/// Same as above for deep TILED: `MultiPartInputFile::deep_tiled_input_part(0)`
/// on a genuinely single-part deep tiled file must match the direct
/// `DeepTiledInputFile` reader, tile-for-tile.
#[test]
fn multipart_input_file_reads_single_part_deep_tiled() {
    let counts = deep_counts();
    let sc = build_count_buffer(&counts);
    let heap = build_sample_heap(0, &counts);
    let fb = deep_frame_buffer();

    let header = deep_tiled_header();
    let mut out = MemOStream::new();
    {
        let mut df = DeepTiledOutputFile::new(&mut out, &header).unwrap();
        df.set_frame_buffer(&fb).unwrap();
        let nx = df.num_x_tiles(0).unwrap();
        let ny = df.num_y_tiles(0).unwrap();
        df.write_tiles(&sc, &heap, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
    }
    let bytes = out.into_data();

    let mut is_direct = MemIStream::new(bytes.clone());
    let mut direct = DeepTiledInputFile::new(&mut is_direct).unwrap();
    direct.set_frame_buffer(&fb).unwrap();
    let nx = direct.num_x_tiles(0).unwrap();
    let ny = direct.num_y_tiles(0).unwrap();
    let mut direct_counts = vec![0u8; sc.len()];
    direct
        .read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut direct_counts)
        .unwrap();
    let mut direct_heap = vec![0u8; heap.len()];
    direct
        .read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &direct_counts, &mut direct_heap)
        .unwrap();

    let mut is_mp = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_mp).unwrap();
    let mut part = mp.deep_tiled_input_part(0).unwrap();
    part.set_frame_buffer(&fb).unwrap();
    let mut mp_counts = vec![0u8; sc.len()];
    part.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut mp_counts)
        .unwrap();
    let mut mp_heap = vec![0u8; heap.len()];
    part.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mp_counts, &mut mp_heap)
        .unwrap();

    assert_eq!(mp_counts, direct_counts, "counts must match the direct reader");
    assert_eq!(mp_heap, direct_heap, "samples must match the direct reader");
    assert_eq!(mp_counts, sc, "counts must match the source data");
    assert_eq!(mp_heap, heap, "samples must match the source data");
}

// ---------------------------------------------------------------------------
// Read-side container caching: each of
// `MultiPartInputFile::{input_part, tiled_input_part, deep_scanline_input_part,
// deep_tiled_input_part}` must persist its part's reader state (in particular
// the frame buffer set via `set_frame_buffer`) across SEPARATE calls, exactly
// as the write side's `output_part`/`tiled_output_part`/... family already
// does (`multipart/mod.rs`'s `interleaved_writes_across_parts_resume_mid_block`
// / `tiled_interleaved_writes_across_parts_flush_buffered_tile_on_resume`).
// Each test below: open part 0, set its frame buffer, read HALF its data ->
// drop that view -> open part 1 (a second, independent part of the same
// storage kind) and read it FULLY, proving the two parts' cached state does
// not cross-contaminate -> drop -> open part 0 AGAIN, WITHOUT calling
// `set_frame_buffer` a second time, and read its OTHER half. This only
// succeeds if the frame buffer set on the first access survived in the
// persisted state: a container that rebuilt a fresh reader state per call
// would come back with no frame buffer set and reject the
// read outright (exactly the failure mode
// `tiled::input::tests::tiled_input_part_reader_reuses_persisted_state_across_interleaved_access`'s
// own "fresh_state" control case demonstrates directly).
// ---------------------------------------------------------------------------

/// Read side, SCANLINE storage: `MultiPartInputFile::input_part`.
#[test]
fn input_part_reuses_persisted_frame_buffer_across_interleaved_access() {
    let px_a = flat_pixels(0);
    let px_b = flat_pixels(1);

    let mut out = MemOStream::new();
    {
        let mut h0 = base_header("scan-a", Compression::NoCompression, LineOrder::IncreasingY);
        h0.set_type(SCANLINE_IMAGE).expect("type");
        let mut h1 = base_header("scan-b", Compression::NoCompression, LineOrder::IncreasingY);
        h1.set_type(SCANLINE_IMAGE).expect("type");
        let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
        for (part, px) in [(0i32, &px_a), (1i32, &px_b)] {
            let mut p = mp.output_part(part).expect("part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", flat_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_pixels(px, H).expect("write");
        }
    }
    let bytes = out.into_data();

    let row_bytes = W as usize * 2;
    let mut is_in = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_in).expect("mp in");
    let mut got_a = vec![0u8; row_bytes * H as usize];
    {
        let mut p = mp.input_part(0).expect("part 0 (first half)");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        p.read_pixels(0, 1, &mut got_a).expect("rows 0-1");
    }
    {
        let mut p = mp.input_part(1).expect("part 1 (full)");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        let mut got_b = vec![0u8; row_bytes * H as usize];
        p.read_pixels(0, H - 1, &mut got_b).expect("full read");
        assert_eq!(got_b, px_b, "part 1 must be unaffected by part 0's partial read");
    }
    {
        let mut p = mp.input_part(0).expect("part 0 (resumed, no set_frame_buffer)");
        p.read_pixels(2, H - 1, &mut got_a)
            .expect("persisted frame buffer must still be set from the first access");
    }
    assert_eq!(got_a, px_a, "part 0 must be bit-exact after interleaved, resumed reads");
}

/// Read side, TILED storage: `MultiPartInputFile::tiled_input_part`.
#[test]
fn tiled_input_part_reuses_persisted_frame_buffer_across_interleaved_access() {
    let px_a = flat_pixels(0);
    let px_b = flat_pixels(1);

    let mut out = MemOStream::new();
    {
        let mut h0 = base_header("tile-a", Compression::Zip, LineOrder::IncreasingY);
        h0.set_type(TILED_IMAGE).expect("type");
        h0.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .expect("tiles");
        let mut h1 = base_header("tile-b", Compression::Zip, LineOrder::IncreasingY);
        h1.set_type(TILED_IMAGE).expect("type");
        h1.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .expect("tiles");
        let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
        for (part, px) in [(0i32, &px_a), (1i32, &px_b)] {
            let mut p = mp.tiled_output_part(part).expect("part");
            let mut fb = FrameBuffer::new();
            fb.insert("Z", flat_slice()).expect("slice");
            p.set_frame_buffer(&fb).expect("set fb");
            let nx = p.num_x_tiles(0).expect("nx");
            let ny = p.num_y_tiles(0).expect("ny");
            p.write_tiles(px, 0, nx - 1, 0, ny - 1, 0, 0).expect("write tiles");
        }
    }
    let bytes = out.into_data();

    let row_bytes = W as usize * 2;
    let mut is_in = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_in).expect("mp in");
    let mut got_a = vec![0u8; row_bytes * H as usize];
    {
        let mut p = mp.tiled_input_part(0).expect("part 0 (tile row 0)");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        // Tile row dy=0 covers absolute rows 0..1 -> the buffer's first half.
        p.read_tiles(0, 1, 0, 0, 0, 0, &mut got_a).expect("tile row 0");
    }
    {
        let mut p = mp.tiled_input_part(1).expect("part 1 (full)");
        let mut fb = FrameBuffer::new();
        fb.insert("Z", flat_slice()).expect("slice");
        p.set_frame_buffer(&fb).expect("set fb");
        let mut got_b = vec![0u8; row_bytes * H as usize];
        p.read_tiles(0, 1, 0, 1, 0, 0, &mut got_b).expect("full read");
        assert_eq!(got_b, px_b, "part 1 must be unaffected by part 0's partial read");
    }
    {
        let mut p = mp.tiled_input_part(0).expect("part 0 (resumed, no set_frame_buffer)");
        // Tile row dy=1 covers absolute rows 2..3 -> the buffer's second half.
        p.read_tiles(0, 1, 1, 1, 0, 0, &mut got_a)
            .expect("persisted frame buffer must still be set from the first access");
    }
    assert_eq!(got_a, px_a, "part 0 must be bit-exact after interleaved, resumed reads");
}

/// Read side, DEEP SCANLINE storage: `MultiPartInputFile::deep_scanline_input_part`.
/// Exercises `read_pixel_sample_counts` only (state persistence is the point;
/// full sample decode is already covered by the single-shot tests above).
#[test]
fn deep_scanline_input_part_reuses_persisted_frame_buffer_across_interleaved_access() {
    let counts = deep_counts();
    let sc_a = build_count_buffer(&counts);
    let heap_a = build_sample_heap(0, &counts);
    let sc_b = build_count_buffer(&counts);
    let heap_b = build_sample_heap(1, &counts);

    let mut out = MemOStream::new();
    {
        let mut h0 = base_header("dscan-a", Compression::Zips, LineOrder::IncreasingY);
        h0.set_type(DEEP_SCANLINE).expect("type");
        let mut h1 = base_header("dscan-b", Compression::Zips, LineOrder::IncreasingY);
        h1.set_type(DEEP_SCANLINE).expect("type");
        let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
        for (part, sc, heap) in [(0i32, &sc_a, &heap_a), (1i32, &sc_b, &heap_b)] {
            let mut p = mp.deep_scanline_output_part(part).expect("part");
            let fb = deep_frame_buffer();
            p.set_frame_buffer(&fb).expect("set fb");
            p.write_pixels(sc, heap, H).expect("write");
        }
    }
    let bytes = out.into_data();

    let row_bytes = 4 * W as usize; // one u32 count per pixel
    let mut is_in = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_in).expect("mp in");
    let mut got_a = vec![0u8; row_bytes * H as usize];
    {
        let mut p = mp.deep_scanline_input_part(0).expect("part 0 (rows 0-1)");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        p.read_pixel_sample_counts(0, 1, &mut got_a).expect("rows 0-1");
    }
    {
        let mut p = mp.deep_scanline_input_part(1).expect("part 1 (full)");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        let mut got_b = vec![0u8; row_bytes * H as usize];
        p.read_pixel_sample_counts(0, H - 1, &mut got_b).expect("full read");
        assert_eq!(got_b, sc_b, "part 1 must be unaffected by part 0's partial read");
    }
    {
        let mut p = mp.deep_scanline_input_part(0).expect("part 0 (resumed, no set_frame_buffer)");
        p.read_pixel_sample_counts(2, H - 1, &mut got_a)
            .expect("persisted frame buffer must still be set from the first access");
    }
    assert_eq!(got_a, sc_a, "part 0 counts must be bit-exact after interleaved, resumed reads");
}

/// Read side, DEEP TILED storage: `MultiPartInputFile::deep_tiled_input_part`.
/// Exercises `read_pixel_sample_counts` only, split by tile row like the
/// tiled-flat interleave test above.
#[test]
fn deep_tiled_input_part_reuses_persisted_frame_buffer_across_interleaved_access() {
    let counts = deep_counts();
    let sc_a = build_count_buffer(&counts);
    let heap_a = build_sample_heap(0, &counts);
    let sc_b = build_count_buffer(&counts);
    let heap_b = build_sample_heap(1, &counts);

    let mut out = MemOStream::new();
    {
        let mut h0 = base_header("dtile-a", Compression::Zips, LineOrder::IncreasingY);
        h0.set_type(DEEP_TILE).expect("type");
        h0.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .expect("tiles");
        let mut h1 = base_header("dtile-b", Compression::Zips, LineOrder::IncreasingY);
        h1.set_type(DEEP_TILE).expect("type");
        h1.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .expect("tiles");
        let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
        for (part, sc, heap) in [(0i32, &sc_a, &heap_a), (1i32, &sc_b, &heap_b)] {
            let mut p = mp.deep_tiled_output_part(part).expect("part");
            let fb = deep_frame_buffer();
            p.set_frame_buffer(&fb).expect("set fb");
            let nx = p.num_x_tiles(0).expect("nx");
            let ny = p.num_y_tiles(0).expect("ny");
            p.write_tiles(sc, heap, 0, nx - 1, 0, ny - 1, 0, 0).expect("write");
        }
    }
    let bytes = out.into_data();

    let row_bytes = 4 * W as usize;
    let mut is_in = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is_in).expect("mp in");
    let mut got_a = vec![0u8; row_bytes * H as usize];
    {
        let mut p = mp.deep_tiled_input_part(0).expect("part 0 (tile row 0)");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        p.read_pixel_sample_counts(0, 1, 0, 0, 0, 0, &mut got_a).expect("tile row 0");
    }
    {
        let mut p = mp.deep_tiled_input_part(1).expect("part 1 (full)");
        let fb = deep_frame_buffer();
        p.set_frame_buffer(&fb).expect("set fb");
        let mut got_b = vec![0u8; row_bytes * H as usize];
        p.read_pixel_sample_counts(0, 1, 0, 1, 0, 0, &mut got_b).expect("full read");
        assert_eq!(got_b, sc_b, "part 1 must be unaffected by part 0's partial read");
    }
    {
        let mut p = mp.deep_tiled_input_part(0).expect("part 0 (resumed, no set_frame_buffer)");
        p.read_pixel_sample_counts(0, 1, 1, 1, 0, 0, &mut got_a)
            .expect("persisted frame buffer must still be set from the first access");
    }
    assert_eq!(got_a, sc_a, "part 0 counts must be bit-exact after interleaved, resumed reads");
}
