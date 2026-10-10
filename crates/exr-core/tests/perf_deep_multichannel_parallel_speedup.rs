//! Confirm real parallel speedup on a
//! MULTI-CHANNEL deep scanline read now that `deep_line_write_interval`
//! computes one EXACT range per active slice instead of one over-wide
//! merged range across all slices. Before that fix, `plan_disjoint` rejected the
//! parallel fast path for any 2+-channel deep image, so `num_threads`
//! made no difference to `read_pixels`'s wall time on real files with
//! more than one channel -- this measures 1 vs 8 threads to show the
//! parallel path now genuinely engages and helps.
//!
//! A confirmatory measurement of THIS fix's effect, not a
//! formal perf claim; absolute numbers may include some
//! concurrent-load noise, but the direction (N threads faster than 1) is
//! what this test exists to prove.
//!
//! Run explicitly (ignored by default -- real file I/O, several seconds):
//! ```text
//! cargo test -p exr-core --release --test perf_deep_multichannel_parallel_speedup -- --ignored --nocapture
//! ```

use std::time::Instant;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
use exr_core::deep::{DeepScanLineInputFile, DeepScanLineOutputFile};
use exr_core::frame_buffer::Slice;
use exr_core::header::{Header, DEEP_SCANLINE};
use exr_core::io::{StdIfStream, StdOfStream};
use imath_rs::{Half, V2f};

const WIDTH: i32 = 1024;
const HEIGHT: i32 = 2048;

/// 0..=4 cycling (average 2.0) -- moderate density, two channels.
fn count_at(idx: i32) -> u32 {
    (idx % 5) as u32
}

#[test]
#[ignore]
fn deep_scanline_multichannel_parallel_is_faster_than_serial() {
    let mut header = Header::with_dimensions(
        WIDTH, HEIGHT, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Zips,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    header.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    header.set_type(DEEP_SCANLINE).unwrap();

    let n = (WIDTH * HEIGHT) as usize;
    let counts: Vec<u32> = (0..n as i32).map(count_at).collect();
    let mut sc = Vec::with_capacity(n * 4);
    for &c in &counts {
        sc.extend_from_slice(&c.to_le_bytes());
    }

    // Interleaved per-pixel: A's samples then Z's samples (matches the
    // scanline-interleaved writer convention channel-by-channel per line).
    let total_samples: usize = counts.iter().map(|&c| c as usize).sum();
    let mut data = Vec::with_capacity(total_samples * (2 + 4));
    // A channel first for the whole image (writer expects the sample data
    // laid out with the count table driving repeats per pixel per channel,
    // channel-by-channel, matching `DeepSlice` insertion order below via
    // the write path's own per-line gather -- see `make_deep_pixels` in
    // `perf_deep_io_vs_decode.rs` for the established one-channel version
    // this generalizes).
    for (pix, &c) in counts.iter().enumerate() {
        for i in 0..c {
            let v = (pix as f32) * 0.001 + i as f32;
            data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
        }
    }
    for (pix, &c) in counts.iter().enumerate() {
        for i in 0..c {
            let v = (pix as f32) * 0.002 + i as f32;
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    let a_heap_bytes = total_samples * 2;
    let mut fb = DeepFrameBuffer::new();
    fb.insert_sample_count_slice(Slice {
        type_: PixelType::Uint,
        base_offset: 0,
        x_stride: 4,
        y_stride: 4 * WIDTH as usize,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    })
    .unwrap();
    fb.insert("A", DeepSlice::new(PixelType::Half, 0, 8, 8 * WIDTH as usize, 2, 1, 1, 0.0, false, false).unwrap())
        .unwrap();
    fb.insert(
        "Z",
        DeepSlice::new(PixelType::Float, a_heap_bytes as i64, 8, 8 * WIDTH as usize, 4, 1, 1, 0.0, false, false)
            .unwrap(),
    )
    .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("perf_deep_multichannel_speedup.exr");
    let path_str = path.to_str().unwrap().to_string();
    {
        let mut os = StdOfStream::create(&path_str).unwrap();
        let mut of = DeepScanLineOutputFile::with_num_threads(&mut os, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&sc, &data, HEIGHT).unwrap();
    }

    let time_read = |threads: i32| -> std::time::Duration {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(threads)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut counts_buf = vec![0u8; n * 4];
        inf.read_pixel_sample_counts(0, HEIGHT - 1, &mut counts_buf).unwrap();
        let mut heap = vec![0u8; data.len()];
        let t0 = Instant::now();
        inf.read_pixels(0, HEIGHT - 1, &counts_buf, &mut heap).unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(heap, data, "sanity: parallel read must still be byte-correct");
        elapsed
    };

    // Warm up (file cache, allocator) before the timed runs.
    let _ = time_read(1);

    const REPEATS: usize = 5;
    let mut t1 = std::time::Duration::ZERO;
    let mut t8 = std::time::Duration::ZERO;
    for _ in 0..REPEATS {
        t1 += time_read(1);
        t8 += time_read(8);
    }

    let ratio = t1.as_secs_f64() / t8.as_secs_f64();
    eprintln!(
        "deep scanline, 2 channels, {WIDTH}x{HEIGHT}, avg {:.1} samples/px: \
         num_threads=1 {:.2}ms, num_threads=8 {:.2}ms, speedup {ratio:.2}x",
        counts.iter().map(|&c| c as f64).sum::<f64>() / n as f64,
        t1.as_secs_f64() * 1000.0 / REPEATS as f64,
        t8.as_secs_f64() * 1000.0 / REPEATS as f64,
    );

    // A weak bound, deliberately: this is a confirmatory "parallelism now
    // exists" check under uncontrolled machine load, not a strict perf
    // gate -- before the interval fix this ratio was pinned at ~1.0x
    // (fell back to serial regardless of `num_threads`) for ANY 2+-channel
    // image; after it, 8 threads should be meaningfully faster.
    assert!(
        ratio > 1.2,
        "expected num_threads=8 to meaningfully beat num_threads=1 on a \
         2-channel image now that the parallel fast path can engage \
         (ratio={ratio:.2}x) -- if this regresses, plan_disjoint may be \
         rejecting multi-channel deep reads again"
    );
}
