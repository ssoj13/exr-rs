//! Integration tests for `exr_core::rgba::{AcesOutputFile, AcesInputFile}`
//! against a real C++ `Imf::AcesOutputFile`/`AcesInputFile` oracle.
//!
//! `golden_aces.exr` is written by C++ `AcesOutputFile` from a known RGBA
//! gradient (`oracle.cpp`'s `write-aces rgba` mode) — a byte-exact check of
//! our `AcesOutputFile` (it stamps ACES chromaticities + `setYCRounding(7,
//! 6)`, exactly like C++).
//!
//! `golden_aces_rec709.exr` is the SAME gradient written through a plain
//! `RgbaOutputFile` with NO explicit chromaticities attribute (`write-aces
//! rec709` mode) — `Header()`'s default is Rec.709, which differs from ACES,
//! so this is the fixture for `AcesInputFile`'s Bradford color conversion.
//! `golden_aces_rec709.exr.dump` is the C++ `AcesInputFile` decode of it
//! (`aces-read` mode, `xtask::DECODE_DUMPS`) — the Bradford matrix has no
//! independent formula to check against (like the other lossy/transform
//! paths in `rgba_yc.rs`), so the C++ decode itself is the reference.

use std::path::PathBuf;

use exr_core::attr::{Compression, LineOrder};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::rgba::{AcesInputFile, AcesOutputFile, Rgba, RgbaChannels};
use imath_rs::{Half, V2f};

mod support;

const HOW_TO_GET: &str = "Run `cargo run -p xtask -- parity-exr` to regenerate the C++ goldens.";
const ACW: usize = 16;
const ACH: usize = 16;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join(name)
}

/// Port of `oracle.cpp`'s `aces_pixel` — the SAME known gradient used for
/// both `write-aces` variants.
fn gradient_pixels() -> Vec<Rgba> {
    let mut pixels = vec![Rgba::default(); ACW * ACH];
    for y in 0..ACH {
        for x in 0..ACW {
            let fx = x as f32 / (ACW - 1) as f32;
            let fy = y as f32 / (ACH - 1) as f32;
            pixels[y * ACW + x] =
                Rgba::new(Half::from_f32(fx), Half::from_f32(fy), Half::from_f32(0.5 - 0.5 * fx), Half::from_f32(1.0));
        }
    }
    pixels
}

fn aces_header() -> Header {
    Header::with_dimensions(ACW as i32, ACH as i32, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::NoCompression)
        .expect("build header")
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

/// BYTE-EXACT check of `AcesOutputFile` against the real C++
/// `AcesOutputFile`'s own output, from the same source pixels.
#[test]
fn aces_output_file_matches_cpp_byte_exact() {
    let name = "aces_output_file_matches_cpp_byte_exact";
    let Some(golden_path) = support::fixture_or_skip(name, golden("golden_aces.exr"), HOW_TO_GET) else {
        return;
    };
    let expected = std::fs::read(&golden_path).unwrap_or_else(|e| panic!("read golden_aces.exr: {e}"));

    let pixels = gradient_pixels();
    let header = aces_header();
    let mut out = MemOStream::new();
    {
        let mut f = AcesOutputFile::new(&mut out, &header, RgbaChannels::WRITE_RGBA).expect("AcesOutputFile::new");
        f.set_frame_buffer(1, ACW as isize).expect("set_frame_buffer");
        f.write_pixels(&pixels, ACH as i32).expect("write_pixels");
    }
    assert_byte_exact("golden_aces.exr", &out.into_data(), &expected);
}

/// Parse `oracle.cpp`'s `aces-read <path> <out>` binary dump: `u32 width, u32
/// height` (LE), then `width*height` pixels row-major, each 4x `u16` LE (r,
/// g, b, a half bits) — the SAME layout `rgba-read` uses (`rgba_yc.rs`'s
/// `parse_rgba_dump`), post color-conversion.
fn parse_aces_dump(bytes: &[u8]) -> (u32, u32, Vec<[u16; 4]>) {
    let w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let mut pixels = Vec::with_capacity((w * h) as usize);
    let mut off = 8usize;
    for _ in 0..(w * h) {
        let mut p = [0u16; 4];
        for slot in &mut p {
            *slot = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            off += 2;
        }
        pixels.push(p);
    }
    assert_eq!(off, bytes.len(), "dump has trailing/missing bytes");
    (w, h, pixels)
}

/// BIT-EXACT check of `AcesInputFile`'s Bradford color-space conversion
/// against a committed C++ `Imf::AcesInputFile` decode dump: the Rec.709
/// golden's chromaticities differ from ACES, so every pixel must be
/// transformed, and the resulting bits must match C++'s float arithmetic
/// exactly (not just approximately).
#[test]
fn aces_input_file_matches_cpp_bit_exact() {
    let name = "aces_input_file_matches_cpp_bit_exact";
    let golden_path = golden("golden_aces_rec709.exr");
    let dump_path = PathBuf::from(format!("{}.dump", golden_path.display()));
    let Some(dump_path) = support::fixture_or_skip(name, dump_path, HOW_TO_GET) else {
        return;
    };
    let Some(golden_path) = support::fixture_or_skip(name, golden_path, HOW_TO_GET) else {
        return;
    };

    let dump_bytes = std::fs::read(&dump_path).unwrap_or_else(|e| panic!("read {}: {e}", dump_path.display()));
    let (dw, dh, expected) = parse_aces_dump(&dump_bytes);

    let bytes = std::fs::read(&golden_path).unwrap_or_else(|e| panic!("read golden_aces_rec709.exr: {e}"));
    let mut stream = MemIStream::new(bytes);
    let mut f = AcesInputFile::new(&mut stream).expect("AcesInputFile::new");
    let dw_box = f.data_window().expect("data_window");
    let (w, h) = ((dw_box.max.x - dw_box.min.x + 1) as usize, (dw_box.max.y - dw_box.min.y + 1) as usize);
    assert_eq!(w as u32, dw, "width mismatch vs C++ dump");
    assert_eq!(h as u32, dh, "height mismatch vs C++ dump");

    let mut base = vec![Rgba::default(); w * h];
    f.read_pixels(dw_box.min.y, dw_box.max.y, &mut base, 1, w as isize).expect("read_pixels");

    let mut mismatches = 0usize;
    for (i, (p, want)) in base.iter().zip(expected.iter()).enumerate() {
        let got = [p.r.bits(), p.g.bits(), p.b.bits(), p.a.bits()];
        if got != *want {
            if mismatches < 5 {
                let x = i % w;
                let y = i / w;
                eprintln!("golden_aces_rec709.exr: mismatch at ({x},{y}): ours={got:04x?} cpp={want:04x?}");
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} pixels differ from the C++ AcesInputFile decode", expected.len());
}
