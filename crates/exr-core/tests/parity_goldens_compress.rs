//! Engagement check for the C++ parity goldens.
//!
//! `parity_exr.rs` proves our encoder/decoder is byte-identical to C++ 3.4.15 on
//! every committed codec golden. That proof is only about the CODEC if the chunks
//! actually went through it: a chunk that does not shrink is stored raw, and a
//! raw chunk is byte-identical for any codec (an earlier 7x5 golden image had
//! exactly this hole). These tests read each golden as raw chunks and assert that the
//! codec, not the raw-store fallback, produced the stored bytes.
//!
//! The expected raw size of a chunk is derived here, independently of
//! `exr-core`: from the KNOWN image layout (see `parity_exr.rs`) and OpenEXR's
//! fixed lines-per-chunk per compression (OpenEXR `ImfCompression.cpp` /
//! `internal_compress.c` `exr_compression_lines_per_chunk`).

use std::path::PathBuf;

use exr_core::MemIStream;
use exr_core::chunk::{ChunkCoord, RawImage};

/// The KNOWN image (must match `parity_exr.rs` / `oracle.cpp`).
const WIDTH: usize = 67;
const HEIGHT: usize = 43;
/// Bytes per pixel: A (FLOAT 4) + B, G, R (HALF 2 each) + U (UINT 4) + Z (FLOAT 4).
const BYTES_PER_PIXEL: usize = 4 + 2 + 2 + 2 + 4 + 4;
const BYTES_PER_LINE: usize = WIDTH * BYTES_PER_PIXEL;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join(name)
}

/// Read a golden as raw chunks and return, per chunk, `(first_y, stored_len, raw_len)`.
fn chunk_sizes(name: &str, lines_per_chunk: usize) -> Vec<(usize, usize, usize)> {
    let path = golden(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut stream = MemIStream::new(bytes);
    let image = RawImage::read(&mut stream)
        .unwrap_or_else(|e| panic!("read {} as raw chunks: {e}", path.display()));
    assert_eq!(
        image.parts.len(),
        1,
        "{name}: codec goldens are single-part"
    );
    image.parts[0]
        .chunks
        .iter()
        .map(|c| {
            let y = match c.coord {
                ChunkCoord::Scanline { y } => usize::try_from(y).expect("non-negative y"),
                ref other => panic!("{name}: expected scanline chunks, got {other:?}"),
            };
            let lines = lines_per_chunk.min(HEIGHT - y);
            (y, c.data.len(), lines * BYTES_PER_LINE)
        })
        .collect()
}

/// Every chunk of `name` must be strictly smaller than its raw size, i.e. was
/// produced by the codec, never by the raw-store fallback.
fn assert_every_chunk_compressed(name: &str, lines_per_chunk: usize) {
    let sizes = chunk_sizes(name, lines_per_chunk);
    let expected_chunks = HEIGHT.div_ceil(lines_per_chunk);
    assert_eq!(
        sizes.len(),
        expected_chunks,
        "{name}: chunk count for {lines_per_chunk} lines/chunk"
    );
    for &(y, stored, raw) in &sizes {
        assert!(
            stored < raw,
            "{name}: chunk at y={y} is {stored} bytes, not smaller than its raw size {raw}: \
             it was stored raw, so the codec path is not what parity_exr compares"
        );
    }
}

#[test]
fn uncompressed_golden_is_exactly_raw() {
    // Control for the size model above: NO_COMPRESSION chunks must be exactly
    // raw-sized, otherwise BYTES_PER_LINE is wrong and every other assertion here
    // is meaningless.
    for (y, stored, raw) in chunk_sizes("golden.exr", 1) {
        assert_eq!(stored, raw, "golden.exr: chunk at y={y}");
    }
}

#[test]
#[should_panic(expected = "stored raw")]
fn raw_stored_chunks_are_rejected() {
    // The check must bite: a file whose chunks are raw-stored fails it.
    assert_every_chunk_compressed("golden.exr", 1);
}

#[test]
fn zip_goldens_go_through_the_codec() {
    assert_every_chunk_compressed("golden_zip.exr", 16);
    assert_every_chunk_compressed("golden_zips.exr", 1);
}

#[test]
fn rle_golden_goes_through_the_codec() {
    assert_every_chunk_compressed("golden_rle.exr", 1);
}

#[test]
fn piz_golden_goes_through_the_codec() {
    assert_every_chunk_compressed("golden_piz.exr", 32);
}

#[test]
fn pxr24_golden_goes_through_the_codec() {
    assert_every_chunk_compressed("golden_pxr24.exr", 16);
}

#[test]
fn b44_goldens_go_through_the_codec() {
    assert_every_chunk_compressed("golden_b44.exr", 32);
    assert_every_chunk_compressed("golden_b44a.exr", 32);
}

#[test]
fn dwa_goldens_go_through_the_codec() {
    assert_every_chunk_compressed("golden_dwaa.exr", 32);
    assert_every_chunk_compressed("golden_dwab.exr", 256);
}

#[test]
fn htj2k_goldens_go_through_the_codec() {
    assert_every_chunk_compressed("golden_htj2k32.exr", 32);
    assert_every_chunk_compressed("golden_htj2k256.exr", 256);
}
