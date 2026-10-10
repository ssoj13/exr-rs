//! Oracle-verified boundary test: HALF->UINT and FLOAT->UINT
//! conversion at the clamp boundaries (negative, NaN, +/-infinity, overflow),
//! checked against a real C++ OpenEXR 3.4 oracle run (`tests/parity/
//! oracle.cpp`'s new `read-as-uint` mode: every channel read into a UINT
//! slice regardless of its on-disk type, exercising C++'s
//! `OpenEXRCore/unpack.c` -> `internal_coding.h::half_to_uint`/
//! `float_to_uint` conversion paths directly -- the mirror of
//! `half_conversion_boundary.rs`'s ->HALF coverage, completing another pair of
//! cells in the pixel-type conversion matrix `misc::copy_into_frame_buffer`
//! implements on read.
//!
//! Same pattern as `half_conversion_boundary.rs`: the C++ oracle output is a
//! COMMITTED fixture (captured from a real `oracle.exe read-as-uint` run
//! against files written by our own trusted `NO_COMPRESSION` writer), so
//! this test is oracle-free once committed, but re-derives the fixture from
//! a live oracle whenever one is available to catch any future divergence.

use std::path::Path;
use std::process::Command;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};

use imath_rs::V2f;

mod support;

/// Boundary FLOAT values, in file order: negative, negative-near-zero, zero,
/// an ordinary positive value, the largest u32 exactly representable in f32
/// (2^32 - 2^8 = 4294967040), f32::MAX (far past u32::MAX), NaN, +inf, -inf.
const FLOAT_VALS: [f32; 9] = [
    -1.0,
    -0.5,
    0.0,
    1000.0,
    4294967040.0,
    f32::MAX,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
];
/// Committed C++ oracle fixture: `oracle.exe read-as-uint` on a FLOAT channel
/// carrying `FLOAT_VALS`, one UINT hex record per pixel.
const EXPECTED_FLOAT_UINTS: [u32; 9] =
    [0, 0, 0, 1000, 4294967040, u32::MAX, 0, u32::MAX, 0];

/// Boundary HALF values (as f32, converted to half bits below), in file
/// order: negative, negative-near-zero, zero, an ordinary positive value,
/// HALF_MAX (65504 -- the largest finite half), NaN, +inf, -inf.
const HALF_VALS_F32: [f32; 8] = [
    -1.0,
    -0.5,
    0.0,
    1000.0,
    65504.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
];
/// Committed C++ oracle fixture: `oracle.exe read-as-uint` on a HALF channel
/// carrying `HALF_VALS_F32`.
const EXPECTED_HALF_UINTS: [u32; 8] = [0, 0, 0, 1000, 65504, 0, u32::MAX, 0];

/// Write a single-row, single-channel `NO_COMPRESSION` scanline EXR of
/// `type_` carrying `raw_le_values` (already-serialized native-endian bytes,
/// one sample per column) to `path`.
fn write_single_channel(path: &Path, type_: PixelType, values: &[u8], width: i32) {
    let mut h = Header::with_dimensions(
        width,
        1,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(type_, 1, 1, true)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

    let mut fb = FrameBuffer::new();
    let size = exr_core::misc::pixel_type_size(type_).unwrap();
    fb.insert(
        "Z",
        Slice::new(type_, 0, size, (width as usize) * size, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();

    let mut stream = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut stream, &h).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(values, 1).unwrap();
    }
    std::fs::write(path, stream.data()).unwrap();
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn half_bits_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| imath_rs::Half::from(*v).bits().to_ne_bytes())
        .collect()
}

/// Read `path`'s single `Z` channel into a UINT slice with OUR reader and
/// return the values, one per pixel -- the exact codepath under test
/// (`misc::copy_into_frame_buffer`, `(PixelType::Uint, PixelType::{Half,Float})`).
fn ours_read_uint(path: &Path, width: i32) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    let mut fb = FrameBuffer::new();
    fb.insert(
        "Z",
        Slice::new(PixelType::Uint, 0, 4, (width as usize) * 4, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();
    inf.set_frame_buffer(&fb).unwrap();
    let mut out = vec![0u8; (width as usize) * 4];
    inf.read_pixels(0, 0, &mut out).unwrap();
    out.as_chunks::<4>().0.iter()
        .map(|c| u32::from_ne_bytes(*c))
        .collect()
}

/// Run `oracle.exe read-as-uint <path>` and return the UINT values it
/// printed for channel `Z`, in x order.
fn oracle_read_as_uint(oracle: &Path, path: &Path) -> Vec<u32> {
    let output = Command::new(oracle)
        .args(["read-as-uint", path.to_str().unwrap()])
        .output()
        .expect("run oracle read-as-uint");
    assert!(
        output.status.success(),
        "oracle read-as-uint failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut vals = Vec::new();
    for line in stdout.lines() {
        // "PIX Z <x> <y> <hex>"
        let mut it = line.split_whitespace();
        if it.next() != Some("PIX") {
            continue;
        }
        let _chan = it.next();
        let _x = it.next();
        let _y = it.next();
        let hex = it.next().expect("PIX line missing bits field");
        vals.push(u32::from_str_radix(hex, 16).expect("PIX bits must be hex"));
    }
    vals
}

#[test]
fn float_to_uint_boundary_matches_oracle_and_our_reader() {
    let dir = std::env::temp_dir();
    let path = dir.join("exr_core_uint_boundary_float.exr");
    write_single_channel(&path, PixelType::Float, &f32_bytes(&FLOAT_VALS), FLOAT_VALS.len() as i32);

    let ours = ours_read_uint(&path, FLOAT_VALS.len() as i32);
    assert_eq!(
        ours, EXPECTED_FLOAT_UINTS,
        "our FLOAT->UINT conversion diverges from the committed C++ 3.4 fixture"
    );

    if let Some(oracle) = support::oracle_or_skip("float_to_uint_boundary_matches_oracle") {
        let live = oracle_read_as_uint(&oracle, &path);
        assert_eq!(
            live, EXPECTED_FLOAT_UINTS,
            "live oracle result no longer matches the committed fixture -- \
             update EXPECTED_FLOAT_UINTS from a fresh `oracle.exe read-as-uint` run"
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn half_to_uint_boundary_matches_oracle_and_our_reader() {
    let dir = std::env::temp_dir();
    let path = dir.join("exr_core_uint_boundary_half.exr");
    write_single_channel(&path, PixelType::Half, &half_bits_bytes(&HALF_VALS_F32), HALF_VALS_F32.len() as i32);

    let ours = ours_read_uint(&path, HALF_VALS_F32.len() as i32);
    assert_eq!(
        ours, EXPECTED_HALF_UINTS,
        "our HALF->UINT conversion diverges from the committed C++ 3.4 fixture"
    );

    if let Some(oracle) = support::oracle_or_skip("half_to_uint_boundary_matches_oracle") {
        let live = oracle_read_as_uint(&oracle, &path);
        assert_eq!(
            live, EXPECTED_HALF_UINTS,
            "live oracle result no longer matches the committed fixture -- \
             update EXPECTED_HALF_UINTS from a fresh `oracle.exe read-as-uint` run"
        );
    }

    std::fs::remove_file(&path).ok();
}
