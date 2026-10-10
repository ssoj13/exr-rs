//! Raw-chunk passthrough round-trip tests (ADD-1).
//!
//! Proves [`RawImage::read`] → [`RawImage::write`] copies a file's compressed
//! chunks **byte-for-byte** (no decode / re-encode), for single-part scanline,
//! single-part tiled, and synthesized multipart files — using the real C++
//! OpenEXR 3.4 golden fixtures committed under `tests/parity/`.
//!
//! The byte-exactness assertion compares the chunks read back from the written
//! file against the chunks read from the source: equal [`RawChunk`]s (coord +
//! verbatim compressed `data`) prove the lossy DWAA / B44 / tiled payloads
//! survived the copy intact.  Re-reading the output also confirms it re-opens
//! and parses.

use std::path::PathBuf;

use exr_core::frame_buffer::Slice;
use exr_core::header::{self, Header};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::InputFile;
use exr_core::tiled::TiledInputFile;
use exr_core::{
    Channel, ChannelList, ChannelListAttribute, Compression, DeepFrameBuffer,
    DeepScanLineOutputFile, DeepSlice, ExrError, LineOrder, PixelType, RawChunk, RawImage,
    RawPart, StringAttribute,
};
use imath_rs::V2f;

/// Read a committed golden EXR from `tests/parity/`.
fn golden(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Read raw chunks from `src`, write them back, and read the output's chunks.
/// Returns `(source_chunks_per_part, output_bytes, output_chunks_per_part)`.
fn round_trip(src: &[u8]) -> (Vec<Vec<RawChunk>>, Vec<u8>, Vec<Vec<RawChunk>>) {
    let mut is = MemIStream::new(src.to_vec());
    let img = RawImage::read(&mut is).expect("read source raw chunks");
    let src_chunks: Vec<Vec<RawChunk>> =
        img.parts.iter().map(|p| p.chunks.clone()).collect();

    let mut os = MemOStream::new();
    img.write(&mut os).expect("write raw chunks");
    let out = os.into_data();

    let mut ois = MemIStream::new(out.clone());
    let out_img = RawImage::read(&mut ois).expect("re-read written raw chunks");
    let out_chunks: Vec<Vec<RawChunk>> =
        out_img.parts.iter().map(|p| p.chunks.clone()).collect();

    (src_chunks, out, out_chunks)
}

/// Assert per-part chunk lists are byte-identical, and at least one chunk
/// carries a non-empty compressed payload (so the test really moved bytes).
fn assert_byte_exact(src: &[Vec<RawChunk>], out: &[Vec<RawChunk>]) {
    assert_eq!(src.len(), out.len(), "part count differs");
    let mut total_bytes = 0usize;
    for (pi, (sc, oc)) in src.iter().zip(out.iter()).enumerate() {
        assert_eq!(sc.len(), oc.len(), "part {pi}: chunk count differs");
        assert!(!sc.is_empty(), "part {pi}: no chunks read");
        for (ci, (s, o)) in sc.iter().zip(oc.iter()).enumerate() {
            assert_eq!(s.coord, o.coord, "part {pi} chunk {ci}: coord differs");
            assert_eq!(
                s.data, o.data,
                "part {pi} chunk {ci}: compressed bytes differ (not byte-exact)"
            );
            total_bytes += s.data.len();
        }
    }
    assert!(total_bytes > 0, "no compressed payload bytes were copied");
}

/// F4 idempotence: prove the passthrough writer is a FIXED POINT over the whole
/// file, not just over chunk payloads. Doing read→write→read→write and asserting
/// the SECOND written file is byte-identical to the FIRST (`out2 == out1`) catches
/// offset-table / header-layout regressions that the chunk-payload compare misses
/// (the first generation may legitimately differ from the original source because
/// the header attribute map is re-serialized, but every generation OURS produces
/// must be stable). Stronger than `assert_byte_exact`, which only checks payloads.
fn assert_idempotent(src: &[u8]) {
    // Generation 1: source → out1.
    let mut is1 = MemIStream::new(src.to_vec());
    let img1 = RawImage::read(&mut is1).expect("read source (gen 1)");
    let mut os1 = MemOStream::new();
    img1.write(&mut os1).expect("write gen 1");
    let out1 = os1.into_data();

    // Generation 2: out1 → out2.
    let mut is2 = MemIStream::new(out1.clone());
    let img2 = RawImage::read(&mut is2).expect("re-read gen-1 output (gen 2)");
    let mut os2 = MemOStream::new();
    img2.write(&mut os2).expect("write gen 2");
    let out2 = os2.into_data();

    assert_eq!(
        out1, out2,
        "second-generation file must be byte-identical to the first \
         (passthrough writer is not a fixed point — whole-file fidelity broken)"
    );
}

#[test]
fn scanline_dwaa_passthrough_byte_exact() {
    // Lossy DWAA scanline file — the passthrough must NOT touch the payload.
    let src = golden("golden_dwaa.exr");
    let (sc, out, oc) = round_trip(&src);
    assert_byte_exact(&sc, &oc);
    // The output must re-open as a scanline file (header + offset table parse).
    let mut is = MemIStream::new(out);
    InputFile::new(&mut is).expect("written DWAA file re-opens as scanline");
    // F4: whole-file fidelity for this single-part lossy file (out2 == out1).
    assert_idempotent(&src);
}

#[test]
fn scanline_b44_passthrough_byte_exact() {
    // A second lossy codec (B44) for good measure.
    let src = golden("golden_b44.exr");
    let (sc, _out, oc) = round_trip(&src);
    assert_byte_exact(&sc, &oc);
    // F4: whole-file fidelity for a second single-part lossy codec (out2 == out1).
    assert_idempotent(&src);
}

#[test]
fn tiled_dwaa_passthrough_byte_exact() {
    // Lossy DWAA *tiled* file — exercises the tile chunk leader + TileOffsets.
    let src = golden("golden_tiled_dwaa_one.exr");
    let (sc, out, oc) = round_trip(&src);
    assert_byte_exact(&sc, &oc);
    let mut is = MemIStream::new(out);
    TiledInputFile::new(&mut is).expect("written tiled DWAA file re-opens as tiled");
    // F4: whole-file fidelity for a single-part TILED file (out2 == out1).
    assert_idempotent(&src);
}

#[test]
fn tiled_none_mipmap_passthrough_byte_exact() {
    // A multi-level (MIPMAP) tiled file: several tiles across several levels,
    // exercising the full TileOffsets slot indexing in both directions.
    let src = golden("golden_tiled_none_mipmap.exr");
    let (sc, out, oc) = round_trip(&src);
    assert_byte_exact(&sc, &oc);
    // The mipmap file has more than one tile chunk.
    assert!(sc[0].len() > 1, "mipmap should have multiple tile chunks");
    let mut is = MemIStream::new(out);
    TiledInputFile::new(&mut is).expect("written mipmap file re-opens as tiled");
    // F4: whole-file fidelity for a multi-level tiled file (out2 == out1).
    assert_idempotent(&src);
}

/// Set the part `type` and a unique `name` so a single-part header can be
/// emitted as one part of a flat multipart file.
fn make_part(mut h: Header, type_name: &str, name: &str, chunks: Vec<RawChunk>) -> RawPart {
    h.set_type(type_name).expect("set type");
    h.insert("name", StringAttribute::new(name.into()))
        .expect("set name");
    RawPart { header: h, chunks }
}

#[test]
fn multipart_scanline_plus_tiled_passthrough_byte_exact() {
    // Build a 2-part flat multipart file from a scanline (DWAA) part and a
    // tiled (DWAA) part, write it through the passthrough writer, then read it
    // back — proving BOTH the multipart write (partNumber prefix + per-part
    // offset tables) and multipart read paths, byte-exact.
    let scan = {
        let mut is = MemIStream::new(golden("golden_dwaa.exr"));
        RawImage::read(&mut is).unwrap()
    };
    let tile = {
        let mut is = MemIStream::new(golden("golden_tiled_dwaa_one.exr"));
        RawImage::read(&mut is).unwrap()
    };

    let scan_part = make_part(
        scan.parts[0].header.clone(),
        header::SCANLINE_IMAGE,
        "scan",
        scan.parts[0].chunks.clone(),
    );
    let tile_part = make_part(
        tile.parts[0].header.clone(),
        header::TILED_IMAGE,
        "tiles",
        tile.parts[0].chunks.clone(),
    );

    let multi = RawImage {
        version: 0,
        parts: vec![scan_part, tile_part],
    };

    let mut os = MemOStream::new();
    multi.write(&mut os).expect("write multipart raw chunks");
    let out = os.into_data();

    // Read back and assert each part's chunks are byte-identical to the source.
    let mut ois = MemIStream::new(out);
    let back = RawImage::read(&mut ois).expect("re-read multipart raw chunks");
    assert_eq!(back.parts.len(), 2, "two parts expected");

    // F6: every multipart part header must carry `chunkCount`, exactly as
    // OpenEXR's MultiPartOutputFile writes it (the source `make_part` headers
    // had none — the passthrough writer must insert it). Its value must equal
    // the part's offset-table size = number of chunks read back.
    for (pi, p) in back.parts.iter().enumerate() {
        assert!(
            p.header.has_chunk_count(),
            "multipart part {pi} header is missing the chunkCount attribute"
        );
        assert_eq!(
            p.header.chunk_count().expect("chunkCount value") as usize,
            p.chunks.len(),
            "part {pi}: chunkCount must equal the number of chunks"
        );
    }

    let src_chunks = vec![
        scan.parts[0].chunks.clone(),
        tile.parts[0].chunks.clone(),
    ];
    let out_chunks: Vec<Vec<RawChunk>> =
        back.parts.iter().map(|p| p.chunks.clone()).collect();
    assert_byte_exact(&src_chunks, &out_chunks);
}

// ---------------------------------------------------------------------------
// F8 — negative paths: an empty image and a deep image must Err, never produce
// a corrupt file or mis-read a deep chunk leader as flat.
// ---------------------------------------------------------------------------

#[test]
fn empty_image_write_errors() {
    // A `RawImage` with no parts must NOT write a (header-less, corrupt) file —
    // it must error up front (`chunk/mod.rs::RawImage::write` parts-empty guard).
    let img = RawImage {
        version: 0,
        parts: vec![],
    };
    let mut os = MemOStream::new();
    let err = img
        .write(&mut os)
        .expect_err("empty RawImage must not produce a file");
    assert!(
        matches!(err, ExrError::Arg(_)),
        "empty image should be rejected with ExrError::Arg, got {err:?}"
    );
}

/// Synthesize a tiny single-part DEEP scanline file in memory via the crate's
/// deep writer (NONE compression, one HALF channel, all-zero per-pixel sample
/// counts → no sample data). Used to prove the raw-chunk passthrough rejects a
/// real deep file rather than mis-parsing its sample-count/size chunk leader as
/// the flat `[coord][dataSize][data]` framing. No on-disk fixture is required.
fn synth_deep_scanline() -> Vec<u8> {
    let (w, h) = (2i32, 2i32);

    let mut hdr = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .expect("deep header");
    let mut cl = ChannelList::new();
    cl.insert("A", Channel::new(PixelType::Half, 1, 1, false))
        .expect("channel");
    hdr.insert("channels", ChannelListAttribute::new(cl))
        .expect("channels attr");
    hdr.set_type(header::DEEP_SCANLINE).expect("deep type");

    // All-zero per-pixel sample counts ⇒ zero samples ⇒ empty sample heap.
    let n = (w * h) as usize;
    let sample_counts = vec![0u8; n * 4]; // u32 LE per pixel
    let sample_data: Vec<u8> = Vec::new();

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
    .expect("sample-count slice");
    fb.insert(
        "A",
        DeepSlice::new(PixelType::Half, 0, 8, 8 * w as usize, 2, 1, 1, 0.0, false, false)
            .expect("deep slice"),
    )
    .expect("deep slice");

    let mut out = MemOStream::new();
    {
        let mut df = DeepScanLineOutputFile::new(&mut out, &hdr).expect("deep writer");
        df.set_frame_buffer(&fb).expect("set deep frame buffer");
        df.write_pixels(&sample_counts, &sample_data, h)
            .expect("write deep pixels");
    }
    out.into_data()
}

#[test]
fn deep_passthrough_read_errors() {
    // A genuine deep file must be rejected by `RawImage::read` (its `part_kind`
    // classifies `deepscanline` as deep and errors), so the deep sample-count
    // chunk leader is never mis-read with the flat framing model.
    let bytes = synth_deep_scanline();
    let mut is = MemIStream::new(bytes);
    match RawImage::read(&mut is) {
        Err(ExrError::Arg(_)) => {}
        Err(other) => panic!("expected ExrError::Arg for a deep file, got {other:?}"),
        Ok(_) => panic!("deep file was unexpectedly accepted by the raw-chunk passthrough"),
    }
}
