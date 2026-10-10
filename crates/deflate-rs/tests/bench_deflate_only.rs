//! Isolates "deflate-rs vs C libdeflate" from "exr-core ZIP
//! pre/post-processing" (predictor + interleave) and from any I/O cost, on
//! the SAME bytes.
//!
//! The codec-isolation harness (`crates/exr-core/tests/
//! perf_codec_isolation.rs`) showed ZIP at ~1.8x/2.2x of C++ OpenEXR
//! (which links libdeflate 1.26) for compress/decompress. That harness times
//! the WHOLE ZIP codec (predictor/interleave + deflate), so it cannot say
//! whether the gap is in deflate-rs itself or in exr-core's surrounding
//! code. This test isolates deflate-rs's own
//! `zlib_compress`/`zlib_decompress` against the live C oracle's new `bench`
//! CLI mode (`oracle.c`: `bench zc/zd <level> <reps>`), on the exact same
//! payload bytes: a realistic EXR ZIP chunk (16 scanlines x 16384 pixels x 3
//! half channels, byte-predicted + interleaved exactly as
//! `exr-core::compressor::predictor::reorder_and_predict` does — reimplemented
//! here rather than depending on exr-core, since deflate-rs must not gain a
//! normal dependency on a crate that depends on it).
//!
//! Ignored by default (spawns a subprocess per repeat); run with
//! `cargo test -p deflate-rs --release --test bench_deflate_only -- --ignored --nocapture`.
//! Requires `cargo xtask parity-deflate` to have been run at least once
//! (builds `tests/parity/oracle.exe` with the `bench` mode).
//!
//! Bench discipline: run on a quiet machine; report median+min, ratio, and the machine load/log
//! timestamp alongside the numbers -- never a bare ratio.

mod support;

use deflate_rs::deflate_compress::Compressor as LowCompressor;
use deflate_rs::deflate_decompress::Decompressor as LowDecompressor;
use deflate_rs::zlib_compress::{zlib_compress, zlib_compress_bound};
use deflate_rs::zlib_decompress::zlib_decompress_ex;

const WIDTH: usize = 16384;
const SCANLINES: usize = 16;
const CHANNELS: usize = 3;
/// Levels checked: 4 is OpenEXR's ZIP default (the reference point); 6 is
/// zlib's own overall default, added to confirm the ratio isn't an artifact specific to level 4.
const ZIP_LEVELS: [i32; 2] = [4, 6];
const OUTER_REPEATS: u32 = 5;
/// Inner reps per outer repeat, so each oracle subprocess call does enough
/// work that process-spawn overhead (a few ms on Windows) is negligible next
/// to the measured compress/decompress time.
const INNER_REPS: u32 = 30;

/// Build one packed (pre-predictor) EXR ZIP-chunk-shaped payload: 16
/// scanlines x 16384 pixels x 3 "half-like" 16-bit channels, laid out
/// channel-planar per scanline (R plane, G plane, B plane, matching
/// OpenEXR's per-scanline channel ordering) with a slowly-varying pattern
/// (smooth gradients, not noise) so its redundancy characteristics resemble
/// real rendered image data rather than incompressible random bytes.
fn build_raw_pixels() -> Vec<u8> {
    let mut raw = Vec::with_capacity(WIDTH * SCANLINES * CHANNELS * 2);
    for line in 0..SCANLINES {
        for ch in 0..CHANNELS {
            for x in 0..WIDTH {
                let v = ((x as u32 * 7 + line as u32 * 131 + ch as u32 * 991) % 2048) as u16;
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    raw
}

/// Reimplementation of `exr-core::compressor::predictor::reorder_and_predict`
/// (byte de-interleave + first-difference predict) -- see that module for
/// the full C-reference citation. Kept in sync by inspection; if exr-core's
/// algorithm ever changes, this benchmark's payload would silently stop
/// being representative, but the two are a stable, long-settled part of the
/// OpenEXR ZIP format (not something either side is expected to change).
fn reorder_and_predict(packed: &[u8]) -> Vec<u8> {
    let n = packed.len();
    let mut scratch = vec![0u8; n];
    if n == 0 {
        return scratch;
    }
    let t2_start = n.div_ceil(2);
    let mut i1 = 0usize;
    let mut i2 = t2_start;
    let mut k = 0usize;
    while k < n {
        scratch[i1] = packed[k];
        i1 += 1;
        k += 1;
        if k < n {
            scratch[i2] = packed[k];
            i2 += 1;
            k += 1;
        }
    }
    let mut p = scratch[0] as i32;
    for cell in scratch.iter_mut().skip(1) {
        let cur = *cell as i32;
        let d = cur - p + (128 + 256);
        p = cur;
        *cell = d as u8;
    }
    scratch
}

fn median_min(mut samples: Vec<i64>) -> (i64, i64) {
    samples.sort_unstable();
    (samples[samples.len() / 2], samples[0])
}

/// Time `INNER_REPS` calls of the LOW-LEVEL, buffer-reusing `zlib_compress`
/// on `payload`, returning nanoseconds-per-call for this outer repeat.
///
/// IMPORTANT: this uses `deflate_rs::zlib_compress::zlib_compress` (reuses
/// one caller-owned `out` buffer across the whole loop), NOT the convenience
/// facade `deflate_rs::Compressor::zlib_compress` (which does `vec![0u8;
/// bound]` -- a fresh ~1.5 MiB allocation + zero-fill EVERY call). A first
/// version of this benchmark used the facade and got a wildly inflated
/// decompress ratio (~3.6x) purely from that per-call alloc/memset, which
/// has nothing to do with the DEFLATE algorithm and which the C oracle's
/// `bench` loop (a single reused buffer) never pays. Comparing "ours:
/// allocate + decode" against "C: decode only" is not the same measurement
/// -- fixed to reuse buffers on both sides, matching the oracle's `bench zc`/
/// `bench zd` timing granularity exactly (one `Instant`/`QueryPerformanceCounter`
/// pair per call, buffers allocated once outside the timed loop).
fn time_ours_compress(level: i32, payload: &[u8]) -> Vec<i64> {
    let mut c = LowCompressor::new(level).expect("valid level");
    let bound = zlib_compress_bound(payload.len());
    let mut out = vec![0u8; bound];
    // Warm-up (matches the oracle's one untimed call).
    let _ = zlib_compress(&mut c, payload, &mut out);
    let mut samples = Vec::with_capacity(INNER_REPS as usize);
    for _ in 0..INNER_REPS {
        let t0 = std::time::Instant::now();
        let _ = zlib_compress(&mut c, payload, &mut out);
        samples.push(t0.elapsed().as_nanos() as i64);
    }
    samples
}

fn time_ours_decompress(compressed: &[u8], out_len: usize) -> Vec<i64> {
    let mut d = LowDecompressor::new();
    let mut out = vec![0u8; out_len];
    let _ = zlib_decompress_ex(&mut d, compressed, &mut out);
    let mut samples = Vec::with_capacity(INNER_REPS as usize);
    for _ in 0..INNER_REPS {
        let t0 = std::time::Instant::now();
        let _ = zlib_decompress_ex(&mut d, compressed, &mut out);
        samples.push(t0.elapsed().as_nanos() as i64);
    }
    samples
}

/// Runs the full A/B (compress then decompress) at one level and prints its
/// row of the summary table.
fn bench_one_level(oracle: &std::path::Path, level: i32, payload: &[u8]) {
    // --- Compress: OUTER_REPEATS outer repeats, alternating A/B order, each
    // an oracle subprocess call (INNER_REPS timed calls inside it) vs an
    // equal in-process Rust loop. ---
    let mut ours_c_all = Vec::new();
    let mut cpp_c_all = Vec::new();
    for i in 0..OUTER_REPEATS {
        if i % 2 == 0 {
            ours_c_all.extend(time_ours_compress(level, payload));
            let s = support::oracle_bench(oracle, "zc", level, INNER_REPS, payload);
            cpp_c_all.push(s.median_ns);
        } else {
            let s = support::oracle_bench(oracle, "zc", level, INNER_REPS, payload);
            cpp_c_all.push(s.median_ns);
            ours_c_all.extend(time_ours_compress(level, payload));
        }
    }
    let (ours_c_median, ours_c_min) = median_min(ours_c_all);
    let (cpp_c_median, cpp_c_min) = median_min(cpp_c_all);

    // --- Decompress: same pattern, using OUR compressed bytes as input (both
    // sides decode the identical, already-proven-byte-identical stream). ---
    let mut c = LowCompressor::new(level).expect("valid level");
    let bound = zlib_compress_bound(payload.len());
    let mut compressed_buf = vec![0u8; bound];
    let compressed_len = zlib_compress(&mut c, payload, &mut compressed_buf);
    let compressed = &compressed_buf[..compressed_len];
    let mut ours_u_all = Vec::new();
    let mut cpp_u_all = Vec::new();
    for i in 0..OUTER_REPEATS {
        if i % 2 == 0 {
            ours_u_all.extend(time_ours_decompress(compressed, payload.len()));
            let s = support::oracle_bench(oracle, "zd", level, INNER_REPS, payload);
            cpp_u_all.push(s.median_ns);
        } else {
            let s = support::oracle_bench(oracle, "zd", level, INNER_REPS, payload);
            cpp_u_all.push(s.median_ns);
            ours_u_all.extend(time_ours_decompress(compressed, payload.len()));
        }
    }
    let (ours_u_median, ours_u_min) = median_min(ours_u_all);
    let (cpp_u_median, cpp_u_min) = median_min(cpp_u_all);

    eprintln!(
        "level {level}: compress   ours median={ours_c_median}ns (min={ours_c_min}ns)  cpp median={cpp_c_median}ns (min={cpp_c_min}ns)  ratio={:.3}x",
        ours_c_median as f64 / cpp_c_median as f64
    );
    eprintln!(
        "level {level}: decompress ours median={ours_u_median}ns (min={ours_u_min}ns)  cpp median={cpp_u_median}ns (min={cpp_u_min}ns)  ratio={:.3}x",
        ours_u_median as f64 / cpp_u_median as f64
    );
}

#[test]
#[ignore] // spawns a subprocess per outer repeat; run explicitly, on a quiet machine
fn deflate_only_compress_and_decompress_vs_libdeflate() {
    let Some(oracle) = support::oracle_or_skip("deflate_only_compress_and_decompress_vs_libdeflate")
    else {
        return;
    };

    // BREAKDOWN item 2: which SIMD tier does the zlib path actually run on
    // THIS machine, right now -- verified, not assumed.
    eprintln!("active tiers: {}", deflate_rs::cpu::active_tiers_report());

    let raw = build_raw_pixels();
    let payload = reorder_and_predict(&raw);
    eprintln!(
        "payload: {} scanlines x {WIDTH} x {CHANNELS} half-like channels = {} bytes (post predictor+interleave)",
        SCANLINES,
        payload.len()
    );

    for &level in &ZIP_LEVELS {
        bench_one_level(&oracle, level, &payload);
    }
}
