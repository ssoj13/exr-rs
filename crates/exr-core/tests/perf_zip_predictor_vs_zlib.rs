//! ZIP perf breakdown (companion to deflate-rs's deflate-only-vs-libdeflate
//! measurement): an IN-PROCESS timing breakdown of
//! ONE ZIP chunk, splitting `Compressor::compress`/`uncompress`'s total time
//! into "predictor/interleave" (`compressor::predictor::{reorder_and_predict,
//! unpredict_and_reorder}`) vs "everything else" (mostly the zlib
//! compress/decompress call inside `ZipCompressor`, which this crate does not
//! expose directly — its share is derived by subtraction: `total -
//! predictor_only`).
//!
//! Chunk shape: 16 lines x 16384 width x 3 HALF channels (R, G, B) —
//! one ZIP chunk (`numScanLines == 16`).
//!
//! Run explicitly (ignored by default):
//! ```text
//! cargo test -p exr-core --release --test perf_zip_predictor_vs_zlib -- --ignored --nocapture
//! ```
//!
//! Bench discipline: run one benchmark at a time on a quiet machine — enforced
//! by whoever launches this test, not by the test itself. 5 repeats, median +
//! min reported.

use std::time::{Duration, Instant};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::compressor::predictor::{reorder_and_predict, unpredict_and_reorder};
use exr_core::compressor::zip::ZipCompressor;
use exr_core::compressor::Compressor;
use exr_core::header::Header;
use imath_rs::{Half, V2f};

const WIDTH: i32 = 16384;
const NUM_SCAN_LINES: i32 = 16; // ZIP
const REPEATS: usize = 5;

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

/// Cosmetic gradient + a little noise -- realistically compressible HALF
/// data (a flat/zero buffer would over-flatter the predictor's cost and a
/// pure-random buffer would defeat ZIP outright; this matches
/// `perf_codec_isolation.rs`'s `sample()`).
fn sample(c: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let gx = x as f32 / w as f32;
    let gy = y as f32 / h as f32;
    let grad = 0.15 + 0.7 * (gx * 0.6 + gy * 0.4) + 0.1 * (c as f32) * 0.25;
    let n = hash01(
        (x as u32).wrapping_mul(73_856_093)
            ^ (y as u32).wrapping_mul(19_349_663)
            ^ (c as u32).wrapping_mul(83_492_791),
    );
    (grad + 0.06 * (n - 0.5)).clamp(0.0, 8.0)
}

/// One ZIP chunk's already-gathered scanline-interleaved Xdr bytes: per
/// line, channels in alphabetical order (B, G, R -- `ChannelList`'s own
/// ordering), each HALF as 2 little-endian bytes -- the exact layout
/// `ZipCompressor::compress`/`uncompress` consumes/produces.
fn make_chunk() -> Vec<u8> {
    let w = WIDTH as usize;
    let h = NUM_SCAN_LINES as usize;
    let names = ["B", "G", "R"]; // alphabetical, matching ChannelList::iter order
    let mut data = Vec::with_capacity(h * w * names.len() * 2);
    for y in 0..h {
        for (ci, _) in names.iter().enumerate() {
            for x in 0..w {
                let v = Half::from_f32(sample(ci, x, y, w, h * 4096)); // arbitrary large h for a gentle gradient
                data.extend_from_slice(&v.bits().to_le_bytes());
            }
        }
    }
    data
}

fn make_header() -> Header {
    let mut h = Header::with_dimensions(
        WIDTH, NUM_SCAN_LINES, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Zip,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

fn median_min(mut v: Vec<Duration>) -> (Duration, Duration) {
    v.sort();
    let min = v[0];
    let median = v[v.len() / 2];
    (median, min)
}

fn fmt(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}

#[test]
#[ignore]
fn zip_predictor_vs_zlib_share() {
    let header = make_header();
    let data = make_chunk();
    let max_scan_line_size = WIDTH as usize * 3 * 2; // 3 HALF channels, 2 bytes each

    eprintln!(
        "zip_predictor_vs_zlib_share: chunk = {} lines x {} width x 3 half = {} bytes, {REPEATS} repeats",
        NUM_SCAN_LINES, WIDTH, data.len()
    );

    // --- Up-front time estimate (one repeat of everything below) ---
    let est_start = Instant::now();
    {
        let mut scratch = vec![0u8; data.len()];
        reorder_and_predict(&mut scratch, &data);
        let mut c = ZipCompressor::new(&header, max_scan_line_size, NUM_SCAN_LINES).unwrap();
        let compressed = c.compress(&data, 0).unwrap().to_vec();
        let mut out = vec![0u8; data.len()];
        let mut scratch2 = scratch.clone();
        unpredict_and_reorder(&mut out, &mut scratch2);
        let mut c2 = ZipCompressor::new(&header, max_scan_line_size, NUM_SCAN_LINES).unwrap();
        let _ = c2.uncompress(&compressed, 0, 0).unwrap();
    }
    let est_one = est_start.elapsed();
    eprintln!(
        "zip_predictor_vs_zlib_share: one repeat took {}, estimated total ~{}",
        fmt(est_one),
        fmt(est_one * REPEATS as u32)
    );

    // --- Compress side: predictor-only vs total compress() ---
    let mut predict_times = Vec::with_capacity(REPEATS);
    let mut scratch_for_reuse = vec![0u8; data.len()];
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        reorder_and_predict(&mut scratch_for_reuse, &data);
        predict_times.push(t0.elapsed());
    }
    // `scratch_for_reuse` now holds the real reorder+predict output for
    // `data` -- reused below as the exact (lossless) intermediate `uncompress`
    // would itself decode, so `unpredict_and_reorder` can be timed in
    // isolation without needing private access to `uncompress_into`.

    let mut compress_total_times = Vec::with_capacity(REPEATS);
    let mut compressed = Vec::new();
    for _ in 0..REPEATS {
        let mut c = ZipCompressor::new(&header, max_scan_line_size, NUM_SCAN_LINES).unwrap();
        let t0 = Instant::now();
        let out = c.compress(&data, 0).unwrap();
        compress_total_times.push(t0.elapsed());
        compressed = out.to_vec();
    }
    assert!(compressed.len() < data.len(), "test setup: fixture must actually compress, not store raw");

    // --- Uncompress side: total uncompress() vs predictor-only (reusing the
    //     exact intermediate bytes captured above) ---
    let mut uncompress_total_times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let mut c = ZipCompressor::new(&header, max_scan_line_size, NUM_SCAN_LINES).unwrap();
        let t0 = Instant::now();
        let _ = c.uncompress(&compressed, 0, 0).unwrap();
        uncompress_total_times.push(t0.elapsed());
    }

    let mut unpredict_times = Vec::with_capacity(REPEATS);
    let mut out_buf = vec![0u8; data.len()];
    for _ in 0..REPEATS {
        let mut scratch_copy = scratch_for_reuse.clone();
        let t0 = Instant::now();
        unpredict_and_reorder(&mut out_buf, &mut scratch_copy);
        unpredict_times.push(t0.elapsed());
    }
    assert_eq!(out_buf, data, "test setup: predictor round trip must reproduce the chunk");

    let (predict_med, predict_min) = median_min(predict_times.clone());
    let (compress_med, compress_min) = median_min(compress_total_times.clone());
    let (uncompress_med, uncompress_min) = median_min(uncompress_total_times.clone());
    let (unpredict_med, unpredict_min) = median_min(unpredict_times.clone());

    let compress_zlib_share_med = compress_med.saturating_sub(predict_med);
    let uncompress_zlib_share_med = uncompress_med.saturating_sub(unpredict_med);

    eprintln!("--- raw per-repeat (ms) ---");
    eprintln!("predict (reorder_and_predict):   {:?}", predict_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect::<Vec<_>>());
    eprintln!("compress total:                  {:?}", compress_total_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect::<Vec<_>>());
    eprintln!("uncompress total:                {:?}", uncompress_total_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect::<Vec<_>>());
    eprintln!("unpredict (unpredict_and_reorder):{:?}", unpredict_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect::<Vec<_>>());

    eprintln!("--- median (min) ---");
    eprintln!("predictor (compress-side, reorder_and_predict): {} ({})", fmt(predict_med), fmt(predict_min));
    eprintln!("compress() total:                               {} ({})", fmt(compress_med), fmt(compress_min));
    eprintln!(
        "  -> zlib::compress_into share (by subtraction):  {} ({:.1}% of total)",
        fmt(compress_zlib_share_med),
        100.0 * compress_zlib_share_med.as_secs_f64() / compress_med.as_secs_f64()
    );
    eprintln!(
        "  -> predictor share of compress():               {:.1}% of total",
        100.0 * predict_med.as_secs_f64() / compress_med.as_secs_f64()
    );
    eprintln!("uncompress() total:                             {} ({})", fmt(uncompress_med), fmt(uncompress_min));
    eprintln!("unpredictor (uncompress-side, unpredict_and_reorder): {} ({})", fmt(unpredict_med), fmt(unpredict_min));
    eprintln!(
        "  -> zlib::uncompress_into share (by subtraction): {} ({:.1}% of total)",
        fmt(uncompress_zlib_share_med),
        100.0 * uncompress_zlib_share_med.as_secs_f64() / uncompress_med.as_secs_f64()
    );
    eprintln!(
        "  -> predictor share of uncompress():              {:.1}% of total",
        100.0 * unpredict_med.as_secs_f64() / uncompress_med.as_secs_f64()
    );
}
