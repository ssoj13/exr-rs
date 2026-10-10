//! Tiled HTJ2K round-trip regression test for sub-image-width GRID tiles.
//!
//! Regression for the bug where `Htj2kCompressor` sized its chunk buffer from
//! the FULL data-window width (captured at construction) and did not override
//! `compress_tile` / `uncompress_tile`.  The tiled writer feeds tile-sized data
//! plus the correct per-tile `Box2i range`, so whenever a tile is narrower than
//! the image (or a partial edge tile is shorter than a full chunk), the codec's
//! `bytes_per_line * tile_height` did not equal the chunk size and the encode
//! failed.
//!
//! The fix mirrors PIZ / PXR24 / DWA: the codec overrides `compress_tile` /
//! `uncompress_tile` to set `min_x` / `max_x` / `max_y` from the tile rectangle
//! so `image_width()` / `block_height()` equal `internal_ht.cpp`'s
//! `encode->chunk.width` / `encode->chunk.height` (the TILE's dimensions).
//!
//! These tests are self-contained (no C++ oracle): they write a tiled HTJ2K EXR
//! fully in memory with `TiledOutputFile`, read it back with `TiledInputFile`,
//! and assert (a) the decode succeeds, (b) dimensions match, and (c) every
//! pixel comes back **exactly**.
//!
//! HTJ2K is LOSSLESS. C++ `ImfCompression.cpp` registers `htj2k256`/`htj2k32`
//! with `lossy = false`, and `internal_ht.cpp` calls `cod.set_reversible(true)`
//! -- an audit found we classified it lossy and let every HTJ2K test
//! tolerate an absolute error of 1.0, which is large enough to hide a genuinely
//! broken reversible transform.

use exr_core::attr::{
    Box2iAttribute, Channel, ChannelList, ChannelListAttribute, Compression, LevelMode,
    LevelRoundingMode, LineOrder, PixelType, TileDescription, TileDescriptionAttribute,
};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, TILED_IMAGE};
use exr_core::tiled::{TiledInputFile, TiledOutputFile};
use exr_core::{MemIStream, MemOStream};

use imath_rs::{Box2i, Half, V2f, V2i};

/// Channels in alphabetical (file) order: A, B, G, R -- all HALF, 2 bytes each.
const CHANS: [&str; 4] = ["A", "B", "G", "R"];
const PIXEL_BYTES: usize = CHANS.len() * 2;

/// A smooth deterministic source value per channel/pixel.  Smooth data keeps the
/// reversible-wavelet error small and predictable.
fn src_val(ci: usize, x: i32, y: i32) -> f32 {
    ci as f32 * 8.0 + y as f32 * 0.5 + x as f32 * 0.25 + 1.0
}

/// Build a tiled HTJ2K header `w` x `h` (origin (0,0)) with `tile` x `tile` tiles.
fn make_header(w: i32, h: i32, tile: u32, comp: Compression) -> Header {
    let mut hdr = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        comp,
    )
    .unwrap();
    let dw = Box2i::from_min_max(V2i { x: 0, y: 0 }, V2i { x: w - 1, y: h - 1 });
    hdr.insert("dataWindow", Box2iAttribute::new(dw)).unwrap();
    hdr.insert("displayWindow", Box2iAttribute::new(dw)).unwrap();
    hdr.set_type(TILED_IMAGE).unwrap();
    hdr.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: tile,
            y_size: tile,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .unwrap();
    let mut cl = ChannelList::new();
    for name in CHANS {
        cl.insert(name, Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
    }
    hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    hdr
}

/// Interleaved native-endian source buffer + a whole-image FrameBuffer over it.
fn build_image(w: i32, h: i32) -> (Vec<u8>, FrameBuffer) {
    let row_bytes = PIXEL_BYTES * w as usize;
    let mut buf = vec![0u8; row_bytes * h as usize];
    for y in 0..h {
        for x in 0..w {
            let p = y as usize * row_bytes + x as usize * PIXEL_BYTES;
            for (ci, _) in CHANS.iter().enumerate() {
                let bits = Half::from(src_val(ci, x, y)).bits();
                buf[p + ci * 2..p + ci * 2 + 2].copy_from_slice(&bits.to_ne_bytes());
            }
        }
    }
    let mk = |off: usize| Slice {
        type_: PixelType::Half,
        base_offset: off as i64,
        x_stride: PIXEL_BYTES,
        y_stride: row_bytes,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    };
    let mut fb = FrameBuffer::new();
    for (ci, name) in CHANS.iter().enumerate() {
        fb.insert(*name, mk(ci * 2)).unwrap();
    }
    (buf, fb)
}

/// Write a tiled HTJ2K image in memory, read it back, and assert close-match.
fn roundtrip(w: i32, h: i32, tile: u32, comp: Compression) {
    let hdr = make_header(w, h, tile, comp);
    let (src, fb) = build_image(w, h);

    // --- write all tiles (the writer feeds each tile its own Box2i range) ---
    let bytes = {
        let mut out = MemOStream::new();
        let mut of = TiledOutputFile::new(&mut out, &hdr).expect("TiledOutputFile::new");
        of.set_frame_buffer(&fb).expect("set_frame_buffer (write)");
        let nx = of.num_x_tiles(0).unwrap();
        let ny = of.num_y_tiles(0).unwrap();
        of.write_tiles(&src, 0, nx - 1, 0, ny - 1, 0, 0)
            .expect("write_tiles must succeed for sub-image-width tiles");
        out.into_data()
    };

    // --- read every tile back ---
    let mut is = MemIStream::new(bytes);
    let mut tif = TiledInputFile::new(&mut is).expect("TiledInputFile::new");
    assert!(tif.is_complete(), "offset table must be complete");

    let lvl = tif.data_window_for_level(0, 0).unwrap();
    assert_eq!(lvl.min.x, 0);
    assert_eq!(lvl.min.y, 0);
    assert_eq!(lvl.max.x, w - 1, "decoded width must match source");
    assert_eq!(lvl.max.y, h - 1, "decoded height must match source");

    let row_bytes = PIXEL_BYTES * w as usize;
    let mut dst = vec![0u8; row_bytes * h as usize];
    let (_, dfb) = build_image(w, h); // reuse the slice layout (same strides)
    tif.set_frame_buffer(&dfb).expect("set_frame_buffer (read)");
    let nx = tif.num_x_tiles(0).unwrap();
    let ny = tif.num_y_tiles(0).unwrap();
    tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut dst)
        .expect("read_tiles must succeed");

    // --- exact match: the reversible (lossless) HTJ2K path must round-trip
    // every HALF sample bit-for-bit ---
    for y in 0..h {
        for x in 0..w {
            let p = y as usize * row_bytes + x as usize * PIXEL_BYTES;
            for (ci, _) in CHANS.iter().enumerate() {
                let o = p + ci * 2;
                let want = Half::from(src_val(ci, x, y)).bits();
                let got = u16::from_ne_bytes([dst[o], dst[o + 1]]);
                assert_eq!(
                    got, want,
                    "tiled HTJ2K is LOSSLESS but pixel ({x},{y}) ch{ci} came back                      {got:#06x} instead of {want:#06x}"
                );
            }
        }
    }
}

/// The previously broken case: GRID tiles narrower than the image, with partial
/// right (50 % 16 = 2 px) AND bottom (38 % 16 = 6 px) edge tiles -- multiple
/// tiles across and down.  Before the fix the writer's `compress_tile` errored.
#[test]
fn tiled_grid_htj2k256_subwidth_tiles() {
    roundtrip(50, 38, 16, Compression::Htj2k256);
}

/// Same with the 32-line variant, exact-multiple dims (no partial tiles) so the
/// only deviation from full-image width is the per-tile X extent.
#[test]
fn tiled_grid_htj2k32_subwidth_tiles() {
    roundtrip(64, 48, 16, Compression::Htj2k32);
}

/// No-regression: a single full-image-width tile (tile == image) must still
/// round-trip -- this is the path that worked before the fix.
#[test]
fn tiled_full_width_htj2k256_no_regression() {
    roundtrip(40, 24, 64, Compression::Htj2k256);
}
