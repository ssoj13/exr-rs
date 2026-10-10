//! Mutation-testing investigation (scanline/ audit, pass 1): cargo-mutants
//! flagged `ScanLineInputPartReader`'s 4 methods (`header`, `version`,
//! `set_frame_buffer`, `read_pixels` -- `crates/exr-core/src/scanline/
//! input.rs:351-383`, `pub(crate)`, `#[allow(dead_code)]`) as missed (6
//! mutants: `header -> Default`, `version -> 0/1/-1`, `set_frame_buffer ->
//! Ok(())`, `read_pixels -> Ok(())`).
//!
//! IMPORTANT FINDING: `ScanLineInputPartReader` is genuinely UNREACHABLE
//! dead code, not merely untested. `grep -rn "ScanLineInputPartReader::new"`
//! across the crate has exactly zero call sites (its deep-scanline cousin,
//! `DeepScanLineInputPartReader::new`, IS called, at
//! `multipart::input::deep_scanline_input_part`, but the flat one never is).
//! `MultiPartInputFile::input_part` -- the obvious way to reach a
//! per-part scanline reader -- goes through `PartReaderState::
//! ScanLine(ScanLineInputFileState)`, calling the STATE's own `header`/
//! `version`/`set_frame_buffer`/`read_pixels` methods DIRECTLY, never
//! constructing a `ScanLineInputPartReader` wrapper around it. Confirmed by
//! hand-mutating each of the 6 targets locally and re-running this test:
//! none of them fail, because none of the 6 mutated lines execute on this
//! (or any other) call path.
//!
//! This test still exercises `InputFilePartReader`'s real forwarding
//! methods end to end through the public API (a legitimate gap this module
//! otherwise had no coverage for), but it does NOT and cannot kill the 6
//! `ScanLineInputPartReader` mutants -- those need either a real call site
//! for `ScanLineInputPartReader` to be added (out of scope here: that's
//! scanline/input.rs) or the dead type deleted
//! entirely, which would remove the mutants along with the code.

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::container::version;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, SCANLINE_IMAGE};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile};
use imath_rs::{Half, V2f};

const W: i32 = 4;
const H: i32 = 3;

fn scanline_header() -> Header {
    let mut h = Header::with_dimensions(
        W,
        H,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .unwrap();
    h.set_name("flat-scan").unwrap();
    h.set_type(SCANLINE_IMAGE).unwrap();
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

/// Pixel (x, y) = half(y * W + x); one HALF channel "Z".
fn make_pixels() -> (Vec<u8>, FrameBuffer) {
    let row = W as usize * 2;
    let mut buf = vec![0u8; row * H as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let v = Half::from((y * W as usize + x) as f32).bits();
            buf[y * row + x * 2..y * row + x * 2 + 2].copy_from_slice(&v.to_ne_bytes());
        }
    }
    let mut fb = FrameBuffer::new();
    fb.insert("Z", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    (buf, fb)
}

#[test]
fn scanline_input_part_reader_header_version_set_frame_buffer_read_pixels() {
    let header = scanline_header();
    // A second part is required to force real multipart output --
    // `MultiPartOutputFile::new` only sets MULTI_PART_FILE_FLAG when
    // `headers.len() > 1` (a single-header "multipart" file degenerates to
    // an ordinary single-part file on disk), and this test wants the
    // genuine multipart path `InputFilePartReader`/`MultiPartInputFile` are
    // actually built for.
    let mut header2 = scanline_header();
    header2.set_name("flat-scan-2").unwrap();
    let (pixels, fb) = make_pixels();

    let mut out = MemOStream::new();
    {
        let mut mp = MultiPartOutputFile::new(&mut out, &[header, header2]).unwrap();
        {
            let mut p = mp.output_part_wrapper(0).unwrap();
            p.set_frame_buffer(&fb).unwrap();
            p.write_pixels(&pixels, H).unwrap();
        }
        {
            let mut p = mp.output_part_wrapper(1).unwrap();
            p.set_frame_buffer(&fb).unwrap();
            p.write_pixels(&pixels, H).unwrap();
        }
    }
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut is).unwrap();
    let mut part = mp.input_part(0).unwrap();

    // `InputFilePartReader::header` (which reaches `ScanLineInputFileState::
    // header` directly, NOT `ScanLineInputPartReader::header` -- see the
    // module doc): a broken forwarder would have neither our data window
    // nor our "Z" channel.
    let dw = part.header().data_window().unwrap();
    assert_eq!((dw.min.x, dw.min.y, dw.max.x, dw.max.y), (0, 0, W - 1, H - 1));
    assert!(part.header().channels().unwrap().iter().any(|(name, _)| name.to_str() == Some("Z")));

    // `InputFilePartReader::version`: the real multipart version word always
    // has MULTI_PART_FILE_FLAG (0x1000) set.
    let v = part.version();
    assert_ne!(v, 0);
    assert_ne!(v, 1);
    assert_ne!(v, -1);
    assert_eq!(
        (v as u32) & version::MULTI_PART_FILE_FLAG,
        version::MULTI_PART_FILE_FLAG,
        "a multipart file's version word must carry MULTI_PART_FILE_FLAG"
    );

    // `InputFilePartReader::set_frame_buffer` + `read_pixels`: a broken
    // forwarder for either would leave `result` all-zero instead of the
    // real pixel values.
    let (_, fb_in) = make_pixels();
    part.set_frame_buffer(&fb_in).unwrap();
    let mut result = vec![0u8; pixels.len()];
    part.read_pixels(0, H - 1, &mut result).unwrap();
    assert_eq!(result, pixels, "read_pixels must return the real pixel data, not a no-op stub");
}
