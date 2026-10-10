//! Mutation-testing gap fix (scanline/ audit pass 1): 3 confirmed survivors
//! in `ScanLineInputFileState`'s `set_frame_buffer`/`read_pixels`
//! (`crates/exr-core/src/scanline/input.rs`, `pub(crate)`), killed through
//! the public `InputFile` API -- no edits to scanline/input.rs
//! needed.
//!
//! Two other survivors on the same list are documented here as EQUIVALENT
//! rather than force-tested, since both are mathematically unreachable:
//!
//! - `input.rs:448:31`, `if max_bytes_per_line > i32::MAX as usize` (`>` ->
//!   `>=`): `max_bytes_per_line` is a sum of terms `width * bytes_per_pixel`
//!   where every `PixelType`'s byte width (`Half`=2, `Float`=4, `Uint`=4) is
//!   EVEN, so the sum is always even. `i32::MAX` (2147483647) is odd. The
//!   two can never be equal, so `>` and `>=` agree on every reachable input
//!   -- the boundary case that would distinguish them is mathematically
//!   impossible to construct.
//! - `input.rs:655:21`, `if max_scan < min_scan` (`<` -> `<=`): this is the
//!   standard "swap if out of order" idiom. `<=` differs from `<` only when
//!   `max_scan == min_scan`, in which case swapping two IDENTICAL values is
//!   a no-op -- no test can observe a difference between performing that
//!   no-op swap and skipping it.

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};
use imath_rs::{Half, V2f};

const W: i32 = 4;
const H: i32 = 6;

fn base_header() -> Header {
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
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

fn make_plane() -> (Vec<u8>, FrameBuffer) {
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

fn write_scanline() -> Vec<u8> {
    let h = base_header();
    let (pixels, fb) = make_plane();
    let mut out = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut out, &h).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, H).unwrap();
    }
    out.into_data()
}

/// Mutation-testing gap fix: `set_frame_buffer`'s subsampling compatibility
/// check (input.rs:546, `||` -> `&&`). Builds a file whose "Z" channel is
/// y-subsampled (`x_sampling = 1, y_sampling = 2`) and sets a frame buffer
/// slice that matches on `x_sampling` but MISMATCHES on `y_sampling` --
/// under the real `||`, disagreeing on EITHER axis is rejected; a `&&`
/// mutant would only reject when BOTH axes disagree, wrongly accepting this
/// single-axis mismatch.
#[test]
fn set_frame_buffer_rejects_a_single_axis_sampling_mismatch() {
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
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 1, 2, false)).unwrap(); // y_sampling = 2
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

    let mut out = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut out, &h).unwrap();
        // Write with a frame buffer slice that actually matches (1, 2), so
        // the file itself is valid.
        let mut fb = FrameBuffer::new();
        let row = W as usize * 2;
        let buf = vec![0u8; row * (H / 2) as usize];
        fb.insert("Z", Slice::new(PixelType::Half, 0, 2, row, 1, 2, 0.0, false, false).unwrap())
            .unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&buf, H).unwrap();
    }
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();

    // A slice for "Z" matching x_sampling (1) but NOT y_sampling (1 instead
    // of the channel's 2) must be rejected.
    let mut fb_mismatch = FrameBuffer::new();
    let row = W as usize * 2;
    fb_mismatch
        .insert("Z", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    let err = match inf.set_frame_buffer(&fb_mismatch) {
        Ok(()) => panic!("a y-sampling mismatch (channel=2, slice=1) must be rejected"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("subsampling"), "unexpected error: {err}");
}

/// Mutation-testing gap fix (found by a mutation run over scanline/): the
/// sibling of the test above, isolating the
/// X-SAMPLING term instead of the y-sampling one (input.rs's
/// `ch.x_sampling != slice.x_sampling || ch.y_sampling != slice.y_sampling`,
/// specifically the `!=`->`==` mutant on the x_sampling comparison). The
/// test above always matches x_sampling and mismatches y_sampling, so a
/// mutation to the x_sampling term alone can't flip the overall `||`
/// result (already true from the y term) -- it went uncaught. This test
/// mismatches x_sampling (channel=2, slice=1) while matching y_sampling
/// (both 1), isolating the x term as the ONLY thing that can trigger
/// rejection.
#[test]
fn set_frame_buffer_rejects_an_x_sampling_mismatch_alone() {
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
    let mut cl = ChannelList::new();
    cl.insert("Z", Channel::new(PixelType::Half, 2, 1, false)).unwrap(); // x_sampling = 2
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

    let mut out = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut out, &h).unwrap();
        // Write with a frame buffer slice that actually matches (2, 1).
        let mut fb = FrameBuffer::new();
        let row = (W / 2) as usize * 2;
        let buf = vec![0u8; row * H as usize];
        fb.insert("Z", Slice::new(PixelType::Half, 0, 2, row, 2, 1, 0.0, false, false).unwrap())
            .unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&buf, H).unwrap();
    }
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();

    // A slice for "Z" matching y_sampling (1) but NOT x_sampling (1 instead
    // of the channel's 2) must be rejected.
    let mut fb_mismatch = FrameBuffer::new();
    let row = W as usize * 2;
    fb_mismatch
        .insert("Z", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    let err = match inf.set_frame_buffer(&fb_mismatch) {
        Ok(()) => panic!("an x-sampling mismatch (channel=2, slice=1) must be rejected"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("subsampling"), "unexpected error: {err}");
}

/// Mutation-testing gap fix: `read_pixels`'s reversed-range swap
/// (input.rs:655, `<` -> `==`). Calling `read_pixels(max, min)` with
/// `scan_line1` greater than `scan_line2` must swap them internally and
/// read the SAME correct data as the properly-ordered call -- a `==`
/// mutant only swaps when the two
/// happen to already be EQUAL, so a genuinely reversed range (5, 0) is
/// never swapped, and the read loop's `while y <= max_scan` (with `y` =
/// the un-swapped, now-larger `min_scan`) never executes, leaving `out`
/// untouched instead of filled with the real pixel data.
#[test]
fn read_pixels_reversed_range_is_swapped_not_left_inverted() {
    let bytes = write_scanline();
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    let (expected, fb) = make_plane();
    inf.set_frame_buffer(&fb).unwrap();

    let mut result = vec![0u8; expected.len()];
    inf.read_pixels(H - 1, 0, &mut result).unwrap(); // reversed: scan_line1 > scan_line2
    assert_eq!(result, expected, "a reversed range must read the same data as the ordered one");
}

/// Mutation-testing gap fix: `read_pixels`'s data-window range check
/// (input.rs:660, `||` -> `&&`). Requesting a range whose upper end lies
/// past the data window's `max_y` (but whose lower end is in-bounds) must
/// be rejected -- under the real `||`, disagreeing on EITHER bound errors;
/// a `&&` mutant would require BOTH bounds to be out of range simultaneously,
/// wrongly accepting this single-sided overrun.
#[test]
fn read_pixels_rejects_a_range_whose_upper_end_overruns_the_data_window() {
    let bytes = write_scanline();
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    let (expected, fb) = make_plane();
    inf.set_frame_buffer(&fb).unwrap();

    let mut result = vec![0u8; expected.len()];
    let err = match inf.read_pixels(0, H + 5, &mut result) {
        Ok(()) => panic!("a range overrunning max_y must be rejected, not silently clamped/accepted"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("data window"), "unexpected error: {err}");
}

/// Investigated for the parallel/serial read dispatch gate
/// (`chunk_indices.len() > 1 && self.num_threads.is_parallel()`,
/// matching C++'s `nchunks > 1 && numThreads > 1` gate,
/// `ImfScanLineInputFile.cpp:444`). This test exercises both the
/// single-chunk (must take the serial path even with `num_threads = 2`)
/// and multi-chunk (must take the parallel path and still be byte-exact)
/// cases -- genuine, useful coverage on its own.
///
/// IMPORTANT: hand-mutating this gate's 3 variants (`&&`->`||`,
/// `>`->`==`, `>`->`<`) and rerunning this exact test shows NONE of them
/// fail. `read_pixels_parallel` is internally robust to being entered
/// with any chunk count (`plan_disjoint`, the window-size computation,
/// and the write-interval planner all degrade gracefully for 0 or 1
/// chunks), so taking the "wrong" branch produces byte-identical,
/// correct output regardless -- the gate's specific `&&`/`>` combination
/// controls ONLY which code path is taken, not whether the result is
/// correct. Killing these 3 mutants would need instrumentation
/// (observing which branch actually ran, e.g. via a call counter) rather
/// than an output-correctness assertion; left documented here rather
/// than forcing a misleading "verified" claim.
#[test]
fn read_pixels_parallel_dispatch_requires_both_more_than_one_chunk_and_parallel_threads() {
    // Single chunk (H = 1): num_threads = 2 must still take the serial
    // path (chunk_indices.len() == 1, so `> 1` is false regardless of
    // is_parallel()).
    let mut h1 = Header::with_dimensions(
        W,
        1,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::NoCompression,
    )
    .unwrap();
    let mut cl1 = ChannelList::new();
    cl1.insert("Z", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h1.insert("channels", ChannelListAttribute::new(cl1)).unwrap();
    let row = W as usize * 2;
    let mut pixels1 = vec![0u8; row];
    for x in 0..W as usize {
        let v = Half::from(x as f32).bits();
        pixels1[x * 2..x * 2 + 2].copy_from_slice(&v.to_ne_bytes());
    }
    let mut fb1 = FrameBuffer::new();
    fb1.insert("Z", Slice::new(PixelType::Half, 0, 2, row, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    let mut out1 = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut out1, &h1).unwrap();
        of.set_frame_buffer(&fb1).unwrap();
        of.write_pixels(&pixels1, 1).unwrap();
    }
    let bytes1 = out1.into_data();

    let mut is1 = MemIStream::new(bytes1);
    let mut inf1 = InputFile::with_num_threads(&mut is1, Some(2)).unwrap();
    inf1.set_frame_buffer(&fb1).unwrap();
    let mut result1 = vec![0u8; pixels1.len()];
    inf1.read_pixels(0, 0, &mut result1).unwrap();
    assert_eq!(result1, pixels1, "single-chunk read with num_threads=2 must still succeed via the serial path");

    // Multi-chunk (H = 8, one line per NONE chunk): num_threads = 2 with
    // MORE than one chunk must genuinely go down the parallel path and
    // still produce byte-exact output.
    let bytes = write_scanline();
    let (expected, fb) = make_plane();
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::with_num_threads(&mut is, Some(2)).unwrap();
    inf.set_frame_buffer(&fb).unwrap();
    let mut result = vec![0u8; expected.len()];
    inf.read_pixels(0, H - 1, &mut result).unwrap();
    assert_eq!(result, expected, "multi-chunk read with num_threads=2 must be byte-exact via the parallel path");
}
