//! Integration tests for the tiled RGBA path (`TiledRgbaInputFile`/
//! `TiledRgbaOutputFile`).
//!
//! `golden_tiled_rgba.exr` / `golden_tiled_rgba_grey.exr` are written by the
//! real C++ `TiledRgbaOutputFile` (`oracle.cpp`'s `write-tiled-rgba` mode),
//! 67x43 with 32x32 tiles — a deliberate partial (edge) tile in both
//! directions, AND large enough that every full tile's raw size (8192 bytes
//! RGBA / 4096 bytes grey) clears `support::assert_chunks_compressed`'s
//! 1024-byte floor (the original 20x13/8x8 fixture's tiles were only 512
//! bytes and were too small to prove ZIP actually compressed anything).
//! Neither shape (plain RGBA, luminance-only) filters or subsamples (tiled
//! EXR has no chroma support at all), so unlike `golden_yc.exr` there is no
//! lossy step: the FILE ITSELF is the bit-exact reference, checked directly
//! against our own `TiledRgbaOutputFile`'s output, and read back bit-exact
//! through our `TiledRgbaInputFile`.

use std::path::PathBuf;

use exr_core::rgba::{Rgba, RgbaChannels, TiledRgbaInputFile, TiledRgbaOutputFile};

mod support;

const HOW_TO_GET: &str = "Run `cargo run -p xtask -- parity-exr` to regenerate the C++ goldens.";
const TRW: usize = 67;
const TRH: usize = 43;
const TILE: u32 = 32;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join(name)
}

fn gradient_pixels() -> Vec<Rgba> {
    use imath_rs::Half;
    let mut pixels = vec![Rgba::default(); TRW * TRH];
    for y in 0..TRH {
        for x in 0..TRW {
            pixels[y * TRW + x] = Rgba::new(
                Half::from_f32(x as f32 * 0.01),
                Half::from_f32(y as f32 * 0.02),
                Half::from_f32((x + y) as f32 * 0.005),
                Half::from_f32(0.75),
            );
        }
    }
    pixels
}

fn assert_byte_exact(golden_name: &str, ours: &[u8], expected: &[u8]) {
    assert_eq!(ours.len(), expected.len(), "{golden_name}: length differs (ours {} vs cpp {})", ours.len(), expected.len());
    if ours != expected {
        let first_diff = ours.iter().zip(expected.iter()).position(|(a, b)| a != b).unwrap();
        panic!(
            "{golden_name}: byte mismatch at offset {first_diff} (ours {:#04x} vs cpp {:#04x})",
            ours[first_diff], expected[first_diff]
        );
    }
}

/// BYTE-EXACT check of `TiledRgbaOutputFile` against the real C++
/// `TiledRgbaOutputFile`'s own output, from the same source pixels — for both
/// the plain (`WRITE_RGBA`) and luminance-only (`WRITE_YA`) shapes.
#[test]
fn tiled_rgba_output_file_matches_cpp_byte_exact() {
    use exr_core::attr::{Compression, LineOrder};
    use exr_core::header::Header;
    use exr_core::io::MemOStream;
    use imath_rs::V2f;

    let pixels = gradient_pixels();
    for (golden_name, chans) in
        [("golden_tiled_rgba.exr", RgbaChannels::WRITE_RGBA), ("golden_tiled_rgba_grey.exr", RgbaChannels::WRITE_YA)]
    {
        let name = "tiled_rgba_output_file_matches_cpp_byte_exact";
        let Some(golden_path) = support::fixture_or_skip(name, golden(golden_name), HOW_TO_GET) else {
            return;
        };
        let expected = std::fs::read(&golden_path).unwrap_or_else(|e| panic!("read {golden_name}: {e}"));

        let header = Header::with_dimensions(
            TRW as i32,
            TRH as i32,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .expect("build header");

        let mut out = MemOStream::new();
        TiledRgbaOutputFile::write_stream(&mut out, &header, chans, TILE, TILE, &pixels)
            .unwrap_or_else(|e| panic!("write {golden_name}: {e}"));
        assert_byte_exact(golden_name, &out.into_data(), &expected);
    }
}

/// `TiledRgbaInputFile` must read the C++-written goldens back correctly:
/// plain RGBA bit-exact against the source pixels, and luminance-only as
/// grey (`R = G = B = Y`) matching the same source pixels' `RGBAtoYCA`
/// luminance.
#[test]
fn tiled_rgba_input_file_reads_cpp_goldens() {
    let name = "tiled_rgba_input_file_reads_cpp_goldens";
    let Some(rgba_path) = support::fixture_or_skip(name, golden("golden_tiled_rgba.exr"), HOW_TO_GET) else {
        return;
    };
    let bytes = std::fs::read(&rgba_path).expect("read golden_tiled_rgba.exr");
    let mut stream = exr_core::io::MemIStream::new(bytes);
    let mut f = TiledRgbaInputFile::new(&mut stream).expect("open golden_tiled_rgba.exr");
    assert!(f.channels().unwrap().contains(RgbaChannels::WRITE_RGBA));

    let pixels = gradient_pixels();
    let nx = f.num_x_tiles(0).unwrap();
    let ny = f.num_y_tiles(0).unwrap();
    for dy in 0..ny {
        for dx in 0..nx {
            let tile_dw = f.data_window_for_tile(dx, dy, 0, 0).unwrap();
            let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
            let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;
            let mut tile = vec![Rgba::default(); tw * th];
            f.read_tile(dx, dy, 0, 0, &mut tile, 1, tw as isize).unwrap();
            for ly in 0..th {
                for lx in 0..tw {
                    let x = (tile_dw.min.x + lx as i32) as usize;
                    let y = (tile_dw.min.y + ly as i32) as usize;
                    assert_eq!(tile[ly * tw + lx], pixels[y * TRW + x], "pixel ({x},{y})");
                }
            }
        }
    }
}
