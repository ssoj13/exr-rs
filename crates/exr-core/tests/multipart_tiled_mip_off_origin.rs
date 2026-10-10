//! Mutation-testing blind-spot coverage:
//! combines every blind spot that has already produced real gaps
//! elsewhere in this audit, INSIDE a multipart file specifically --
//! `crates/exr-core/tests/multipart_mixed.rs`'s tiled part uses
//! `Header::with_dimensions` (origin only) and `LevelMode::OneLevel`, so
//! "off-origin + MIPMAP + partial edge tiles tiled part, inside a
//! multipart container, alongside a different part kind" was untested
//! anywhere in this crate.
//!
//! Part 0: tiled, data window origin (11, 7) (NOT (0, 0)), 7x5 pixels over
//! 2x2 tiles (neither dimension divides evenly -> genuine partial edge
//! tiles), `LevelMode::MipmapLevels`. Every level is round-tripped
//! byte-exact.
//! Part 1: a different kind (plain scanline) at the default origin, to
//! confirm the multipart container's per-part isolation still holds with
//! this much less "boring" part 0.

use exr_core::attr::tile_description::{LevelMode, LevelRoundingMode};
use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType,
    TileDescription, TileDescriptionAttribute,
};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile};
use exr_core::tiled::misc::precalculate_tile_info;
use imath_rs::{Half, V2f};

const MIN_X: i32 = 11;
const MIN_Y: i32 = 7;
const W: i32 = 7;
const H: i32 = 5;

fn tiled_part_header() -> Header {
    let dw = imath_rs::Box2i::from_min_max(
        imath_rs::V2i { x: MIN_X, y: MIN_Y },
        imath_rs::V2i { x: MIN_X + W - 1, y: MIN_Y + H - 1 },
    );
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
    h.set_name("mip-off-origin").unwrap();
    h.set_type(exr_core::header::TILED_IMAGE).unwrap();
    h.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

fn scanline_part_header() -> Header {
    // Multipart requires every part to share the SAME displayWindow -- use
    // the tiled part's, but keep this part's own dataWindow small (4x3 at
    // the origin) to prove per-part data-window independence still holds.
    let display = tiled_part_header().data_window().unwrap();
    let dw = imath_rs::Box2i::from_min_max(
        imath_rs::V2i { x: 0, y: 0 },
        imath_rs::V2i { x: 3, y: 2 },
    );
    let mut h = Header::with_windows(
        display,
        dw,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .unwrap();
    h.set_name("flat-scan").unwrap();
    h.set_type(exr_core::header::SCANLINE_IMAGE).unwrap();
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

/// Row-major HALF pixel plane over `[0..W) x [0..H)`, one channel. Pixel
/// (x, y) holds `half(y * W + x)`. `base_offset` addresses absolute pixel
/// (MIN_X, MIN_Y) at buffer index 0.
fn tiled_plane() -> (Vec<u8>, FrameBuffer) {
    let row_bytes = W as usize * 2;
    let mut buf = vec![0u8; row_bytes * H as usize];
    for ry in 0..H {
        for rx in 0..W {
            let off = ry as usize * row_bytes + rx as usize * 2;
            let v = Half::from((ry * W + rx) as f32).bits();
            buf[off..off + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }
    let x_stride = 2i64;
    let y_stride = row_bytes as i64;
    let base_offset = -(MIN_Y as i64 * y_stride + MIN_X as i64 * x_stride);
    let mut fb = FrameBuffer::new();
    fb.insert(
        "R",
        Slice {
            type_: PixelType::Half,
            base_offset,
            x_stride: 2,
            y_stride: row_bytes,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        },
    )
    .unwrap();
    (buf, fb)
}

fn scanline_plane() -> (Vec<u8>, FrameBuffer) {
    let row_bytes = 4usize * 2;
    let mut buf = vec![0u8; row_bytes * 3];
    for y in 0..3usize {
        for x in 0..4usize {
            let v = Half::from((y * 4 + x) as f32 + 1000.0).bits();
            let off = y * row_bytes + x * 2;
            buf[off..off + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }
    let mut fb = FrameBuffer::new();
    fb.insert("Z", Slice::new(PixelType::Half, 0, 2, row_bytes, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    (buf, fb)
}

#[test]
fn multipart_tiled_off_origin_mipmap_partial_tiles_round_trips_every_level() {
    let tiled_header = tiled_part_header();
    let scan_header = scanline_part_header();
    let (tiled_pixels, tiled_fb) = tiled_plane();
    let (scan_pixels, scan_fb) = scanline_plane();

    let td = tiled_header.tile_description().unwrap();
    let (_, _, num_x_levels, _) =
        precalculate_tile_info(&td, MIN_X, MIN_X + W - 1, MIN_Y, MIN_Y + H - 1).unwrap();
    let levels: Vec<(i32, i32)> = (0..num_x_levels).map(|l| (l, l)).collect();
    assert!(levels.len() > 1, "test setup: MIPMAP over a 7x5/2x2 image must have more than one level");

    let mut out = MemOStream::new();
    {
        let mut mp = MultiPartOutputFile::new(&mut out, &[tiled_header, scan_header]).unwrap();

        {
            let mut p = mp.tiled_output_part_wrapper(0).unwrap();
            p.set_frame_buffer(&tiled_fb).unwrap();
            for &(lx, ly) in &levels {
                let nx = p.num_x_tiles(lx).unwrap();
                let ny = p.num_y_tiles(ly).unwrap();
                p.write_tiles(&tiled_pixels, 0, nx - 1, 0, ny - 1, lx, ly).unwrap();
            }
        }
        {
            let mut p = mp.output_part_wrapper(1).unwrap();
            p.set_frame_buffer(&scan_fb).unwrap();
            p.write_pixels(&scan_pixels, 3).unwrap();
        }
    }
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is).unwrap();

    // --- Part 0: tiled, every mip level byte-exact ------------------------
    {
        let mut p = mp.tiled_input_part(0).unwrap();
        assert!(p.is_complete(), "part 0 offset table must be complete");
        assert_eq!(
            p.header().compression().unwrap(),
            Compression::NoCompression,
            "part 0 compression mismatch"
        );

        for &(lx, ly) in &levels {
            let lvl = p.data_window_for_level(lx, ly).unwrap();
            let lw = lvl.max.x - lvl.min.x + 1;
            let lh = lvl.max.y - lvl.min.y + 1;
            let row_bytes = lw as usize * 2;
            let mut readback = vec![0u8; row_bytes * lh as usize];
            let x_stride = 2i64;
            let y_stride = row_bytes as i64;
            let base_offset = -(lvl.min.y as i64 * y_stride + lvl.min.x as i64 * x_stride);
            let mut rfb = FrameBuffer::new();
            rfb.insert(
                "R",
                Slice {
                    type_: PixelType::Half,
                    base_offset,
                    x_stride: 2,
                    y_stride: row_bytes,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
            p.set_frame_buffer(&rfb).unwrap();
            let nx = p.num_x_tiles(lx).unwrap();
            let ny = p.num_y_tiles(ly).unwrap();
            p.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut readback).unwrap();

            if lx == 0 && ly == 0 {
                assert_eq!(readback, tiled_pixels, "part 0 level 0 round-trip mismatch");
            } else {
                // Other levels: `xTileCoords = false` means every level
                // addresses the SAME full-image buffer as level 0 (levels
                // shrink toward the top-left, sharing data-window MIN), so a
                // level-L tile pixel at LOCAL (x, y) must read the exact same
                // cell as full-res pixel (x, y): half(y * W + x) using the
                // FULL image width -- matching tiled/input.rs's
                // round_trip_all technique (assert exact values, not just
                // "non-zero", since a length/non-zero check can't catch
                // wrong values).
                for ry in 0..lh {
                    for rx in 0..lw {
                        let expected = Half::from((ry * W + rx) as f32).bits();
                        let off = ry as usize * row_bytes + rx as usize * 2;
                        let got = u16::from_ne_bytes([readback[off], readback[off + 1]]);
                        assert_eq!(
                            got, expected,
                            "part 0 level ({lx},{ly}) pixel ({rx},{ry}) mismatch"
                        );
                    }
                }
            }
        }
    }

    // --- Part 1: plain scanline, still byte-exact --------------------------
    {
        let mut p = mp.input_part(1).unwrap();
        let mut readback = vec![0u8; scan_pixels.len()];
        p.set_frame_buffer(&scan_fb).unwrap();
        p.read_pixels(0, 2, &mut readback).unwrap();
        assert_eq!(readback, scan_pixels, "part 1 (scanline) round-trip mismatch");
    }
}
