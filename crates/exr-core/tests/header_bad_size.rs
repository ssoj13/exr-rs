//! `comp_invalid_unknown.exr` (`testMalformedImages.cpp`): its `displayWindow`
//! declares 32528 bytes with 611 left in the file. C++ 3.4 opens it (core
//! `parse_header.c` drops the attribute WITHOUT skipping, the value's first
//! zero byte ends the header; `exrheader` lists it as an incomplete file with
//! a defaulted 64x64 `displayWindow`) and only the pixel read throws `IoExc`.
//! Our reader must open it the same way and fail at the same later stage.

use exr_core::attr::PixelType;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::StdIfStream;
use exr_core::scanline::InputFile;

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn opens_like_cpp_and_fails_at_the_pixel_read() {
    let path = exr_test_data::openexr_test().join("OpenEXRTest/comp_invalid_unknown.exr");
    let mut s = StdIfStream::open(path.to_str().expect("utf-8 path")).expect("open");
    let mut inf = InputFile::new(&mut s).expect("C++ opens this file, so must we");

    // The oversized displayWindow is dropped and the default window filled in.
    let dw = inf.header().display_window().expect("displayWindow");
    assert_eq!((dw.min.x, dw.min.y, dw.max.x, dw.max.y), (0, 0, 63, 63));
    let dat = inf.header().data_window().expect("dataWindow");
    assert_eq!((dat.min.x, dat.min.y, dat.max.x, dat.max.y), (0, 0, 609, 405));
    assert!(!inf.is_complete(), "exrheader reports this file as incomplete");

    // The pixel read is where `testMalformedImages.cpp` expects the throw.
    let row_bytes = 610 * 2;
    let mut buf = vec![0u8; row_bytes];
    let mut fb = FrameBuffer::new();
    let slice = Slice::new(PixelType::Half, 0, 2, row_bytes, 1, 1, 0.0, false, false).expect("slice");
    fb.insert("o", slice).expect("insert");
    inf.set_frame_buffer(&fb).expect("frame buffer");
    inf.read_pixels(0, 0, &mut buf).expect_err("reading the pixels must fail");
}
