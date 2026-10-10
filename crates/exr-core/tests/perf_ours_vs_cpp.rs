//! "As C++ or better" wall-time comparison (a numeric claim needs a numeric
//! comparison, not just an internal before/after). Writes and reads the SAME 16384x8192 RGBA half+float
//! image at the same thread count on both sides — ours via
//! `OutputFile::with_num_threads`/`InputFile::with_num_threads`, C++'s via
//! `oracle.cpp`'s `bench-io` mode (`setGlobalThreadCount`) — for every codec in
//! `Compression::ALL`, and logs a per-codec read+write wall-time table to
//! `target/bench/ours_vs_cpp_t<threads>.log`.
//!
//! This is a THROUGHPUT comparison, not a byte-parity one, but the input is
//! identical: `make_pixels` builds the image once and C++ reads that raw
//! buffer, because a codec's time depends on how compressible its input is.
//! Repeats interleave the two sides and alternate which one runs first.
//!
//! Uses `support::oracle_or_skip`, the same opt-out-only skip discipline as
//! every other C++-cross-check in this crate (`EXR_ALLOW_SKIP=1` to skip
//! deliberately; otherwise a missing oracle panics rather than silently
//! passing).
//!
//! Run explicitly on a quiet machine (ignored by default — real file I/O,
//! ~1 min per codec and thread count):
//! ```text
//! cargo test -p exr-core --release --test perf_ours_vs_cpp -- --ignored --nocapture
//! ```

mod support;

use std::process::Command;
use std::time::Instant;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, SCANLINE_IMAGE};
use exr_core::io::{StdIfStream, StdOfStream};
use exr_core::scanline::{InputFile, OutputFile};
use imath_rs::{Half, V2f};

const WIDTH: i32 = 16384;
const HEIGHT: i32 = 8192;

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

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

fn make_header(comp: Compression) -> Header {
    let mut h = Header::with_dimensions(
        WIDTH, HEIGHT, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h.set_type(SCANLINE_IMAGE).unwrap();
    h
}

fn make_pixels() -> (Vec<u8>, FrameBuffer) {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let pixel_bytes = 2 + 2 + 2 + 4;
    let row_bytes = pixel_bytes * w;
    let mut buf = vec![0u8; row_bytes * h];
    for y in 0..h {
        for x in 0..w {
            let p = y * row_bytes + x * pixel_bytes;
            let r = Half::from(sample(0, x, y, w, h)).bits();
            let g = Half::from(sample(1, x, y, w, h)).bits();
            let b = Half::from(sample(2, x, y, w, h)).bits();
            let a = sample(3, x, y, w, h);
            buf[p..p + 2].copy_from_slice(&r.to_ne_bytes());
            buf[p + 2..p + 4].copy_from_slice(&g.to_ne_bytes());
            buf[p + 4..p + 6].copy_from_slice(&b.to_ne_bytes());
            buf[p + 6..p + 10].copy_from_slice(&a.to_ne_bytes());
        }
    }
    let mut fb = FrameBuffer::new();
    let mk = |ty: PixelType, off: usize| Slice {
        type_: ty, base_offset: off as i64, x_stride: pixel_bytes, y_stride: row_bytes,
        x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
    };
    fb.insert("R", mk(PixelType::Half, 0)).unwrap();
    fb.insert("G", mk(PixelType::Half, 2)).unwrap();
    fb.insert("B", mk(PixelType::Half, 4)).unwrap();
    fb.insert("A", mk(PixelType::Float, 6)).unwrap();
    (buf, fb)
}

/// Time our own write+read at `threads` threads, in a fresh temp file.
fn time_ours(path: &str, header: &Header, pixels: &[u8], fb: &FrameBuffer, threads: i32) -> (f64, f64) {
    let t0 = Instant::now();
    {
        let mut os = StdOfStream::create(path).unwrap();
        let mut of = OutputFile::with_num_threads(&mut os, header, Some(threads)).unwrap();
        of.set_frame_buffer(fb).unwrap();
        of.write_pixels(pixels, HEIGHT).unwrap();
    }
    let write_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The readback buffer is allocated once, before any timing, and reused:
    // oracle.cpp's `bench-io` allocates its `readback` vector outside the
    // timed region too, and its warm-up read has already faulted every page
    // in. Allocating a fresh zeroed 1.3 GB buffer inside the timed read
    // charged ~330k page faults to our side only.
    let mut readback = vec![0u8; pixels.len()];

    // Warm-up read (untimed): puts the file in the OS page cache before the
    // TIMED read below, mirroring oracle.cpp's `bench-io`'s own warm-up
    // pass (reduces cross-run noise from a cold vs already-cached first
    // read).
    {
        let mut is = StdIfStream::open(path).unwrap();
        let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
        inf.set_frame_buffer(fb).unwrap();
        inf.read_pixels(0, HEIGHT - 1, &mut readback).unwrap();
    }

    let t1 = Instant::now();
    {
        let mut is = StdIfStream::open(path).unwrap();
        let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
        inf.set_frame_buffer(fb).unwrap();
        inf.read_pixels(0, HEIGHT - 1, &mut readback).unwrap();
    }
    let read_ms = t1.elapsed().as_secs_f64() * 1e3;

    (write_ms, read_ms)
}

/// Run `oracle bench-io <pixels> <path> <comp> <width> <height> <threads>` and
/// parse its `write_ms=... read_ms=...` stdout line. `pixels` is the raw image
/// `make_pixels` built, so C++ compresses exactly what we do.
fn time_cpp(oracle: &std::path::Path, pixels: &str, path: &str, comp_name: &str, threads: i32) -> (f64, f64) {
    let output = Command::new(oracle)
        .args([
            "bench-io",
            pixels,
            path,
            comp_name,
            &WIDTH.to_string(),
            &HEIGHT.to_string(),
            &threads.to_string(),
        ])
        .output()
        .expect("failed to run oracle bench-io");
    assert!(
        output.status.success(),
        "oracle bench-io failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut write_ms = None;
    let mut read_ms = None;
    for tok in stdout.split_whitespace() {
        if let Some(v) = tok.strip_prefix("write_ms=") {
            write_ms = v.parse::<f64>().ok();
        }
        if let Some(v) = tok.strip_prefix("read_ms=") {
            read_ms = v.parse::<f64>().ok();
        }
    }
    (
        write_ms.unwrap_or_else(|| panic!("no write_ms in oracle output: {stdout}")),
        read_ms.unwrap_or_else(|| panic!("no read_ms in oracle output: {stdout}")),
    )
}

/// Median, minimum and spread (`(max - min) / median`, percent) of one series.
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let med = s[s.len() / 2];
    (med, s[0], (s[s.len() - 1] - s[0]) / med * 100.0)
}

/// Repeats per codec/side and takes the MEDIAN — this measurement machine
/// showed 2-3x run-to-run swings for the identical codec/size in
/// back-to-back single-shot runs (background load, page-cache state,
/// antivirus scanning the freshly-written multi-GB temp files), so a single
/// sample per codec is not reliable evidence for "we are N% slower/faster
/// than C++ at codec X" — only the median of several is reported. Bumped
/// from 3 to 5 (to reduce noise further) alongside the
/// warm-up read added to both `time_ours` and `oracle.cpp`'s `bench-io`.
const REPEATS: usize = 5;

/// A short pause between repeats — gives a background
/// burst (antivirus, OS housekeeping) a moment to settle before the next
/// sample, instead of every repeat running back-to-back with no gap.
const PAUSE_BETWEEN_REPEATS: std::time::Duration = std::time::Duration::from_millis(300);

/// Run the comparison at a given thread count on both sides and print the
/// table — `threads=1` isolates the codec/pipeline cost from parallel
/// dispatch overhead (if single-threaded is ALREADY
/// slower, the gap is in decode/encode/scatter, not in how the parallel
/// path is dispatched).
/// Serializes every 16k test in this file; see `run_comparison`.
static BENCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_comparison(threads: i32) {
    // The two #[test]s in this file run concurrently under the default test
    // harness. Running both at once put two 16k benchmarks on the CPU
    // together (every timing contaminated) and had them overwrite each
    // other's same-named temp files mid-read (a spurious "corrupt offset
    // table" failure). Serialize them, and key every file by thread count.
    let _one_at_a_time = BENCH.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(oracle) = support::oracle_or_skip("ours_vs_cpp_16k_wall_time_across_codecs") else {
        return;
    };

    let dir = std::env::temp_dir();
    let (pixels, fb) = make_pixels();
    // The one input both sides compress (C++ reads it in `bench-io`).
    let raw = dir.join(format!("perf_cmp_pixels_t{threads}.raw"));
    std::fs::write(&raw, &pixels).unwrap();
    let raw = raw.to_str().unwrap().to_string();

    // Every run is logged as it happens (per-repeat samples, then the table), so a long run
    // can be followed and its numbers are quoted from the log, never retyped.
    let log_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/bench/ours_vs_cpp_t{threads}.log"));
    std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
    let mut log = std::fs::File::create(&log_path).unwrap();
    let mut say = |line: String| {
        println!("{line}");
        use std::io::Write;
        writeln!(log, "{line}").unwrap();
        log.flush().unwrap();
    };

    say(format!(
        "ours vs C++ OpenEXR (oracle bench-io), {WIDTH}x{HEIGHT} RGBA half+float, identical pixels, \
         num_threads={threads}, {REPEATS} interleaved repeats per codec (the side that goes first \
         alternates), release; ratio = ours/cpp, < 1.0 = we are faster. Expect ~1 min per codec."
    ));
    let mut table = vec![format!(
        "{:>8}  {:>9} {:>9} {:>6} {:>6}   {:>9} {:>9} {:>6} {:>6}   spread% ow/cw/or/cr",
        "codec", "ours_w", "cpp_w", "med", "min", "ours_r", "cpp_r", "med", "min"
    )];

    for comp in Compression::ALL {
        let name = comp.name();
        let header = make_header(comp);
        let ours_path = dir.join(format!("perf_cmp_ours_{name}_t{threads}.exr"));
        let ours_path = ours_path.to_str().unwrap().to_string();
        let cpp_path = dir.join(format!("perf_cmp_cpp_{name}_t{threads}.exr"));
        let cpp_path = cpp_path.to_str().unwrap().to_string();

        let (mut ow, mut or, mut cw, mut cr) = (vec![], vec![], vec![], vec![]);
        for i in 0..REPEATS {
            if i > 0 {
                std::thread::sleep(PAUSE_BETWEEN_REPEATS);
            }
            let ours = || {
                let (w, r) = time_ours(&ours_path, &header, &pixels, &fb, threads);
                std::fs::remove_file(&ours_path).unwrap();
                (w, r)
            };
            let cpp = || {
                let (w, r) = time_cpp(&oracle, &raw, &cpp_path, name, threads);
                std::fs::remove_file(&cpp_path).unwrap();
                (w, r)
            };
            let ((w, r), (cw_i, cr_i)) = if i % 2 == 0 {
                let o = ours();
                (o, cpp())
            } else {
                let c = cpp();
                (ours(), c)
            };
            say(format!(
                "{name} repeat {i}: ours write {w:.1} read {r:.1} | cpp write {cw_i:.1} read {cr_i:.1} ms"
            ));
            ow.push(w);
            or.push(r);
            cw.push(cw_i);
            cr.push(cr_i);
        }

        let (ow, cw, or, cr) = (stats(&ow), stats(&cw), stats(&or), stats(&cr));
        table.push(format!(
            "{:>8}  {:>7.1}ms {:>7.1}ms {:>5.2}x {:>5.2}x   {:>7.1}ms {:>7.1}ms {:>5.2}x {:>5.2}x   {:.0}/{:.0}/{:.0}/{:.0}",
            name.to_uppercase(),
            ow.0, cw.0, ow.0 / cw.0, ow.1 / cw.1,
            or.0, cr.0, or.0 / cr.0, or.1 / cr.1,
            ow.2, cw.2, or.2, cr.2,
        ));
    }
    std::fs::remove_file(&raw).unwrap();
    for line in table {
        say(line);
    }
}

#[test]
#[ignore] // real file I/O + several subprocesses per codec; run explicitly, see module doc
fn ours_vs_cpp_16k_wall_time_across_codecs() {
    run_comparison(8);
}

/// Isolates codec/pipeline cost from parallel dispatch overhead:
/// `setGlobalThreadCount(0)`/`num_threads=0` on both sides runs
/// every chunk on the calling thread, so any remaining "ours slower" gap
/// here is NOT explained by rayon dispatch, channel handoffs, or thread
/// pool overhead — it has to be in the actual gather/decode/encode/scatter
/// work itself.
#[test]
#[ignore]
fn ours_vs_cpp_16k_wall_time_single_threaded() {
    run_comparison(0);
}

/// Profiling target, no oracle: only OUR single-threaded write + read of the
/// 16k image, repeated, for a sampling profiler to attach to. NONE by default
/// (no codec work, so the profile shows the gather/scatter/I/O pipeline);
/// `EXR_PROFILE_CODEC=rle|zip|piz|dwaa` picks another codec. Prints each
/// repeat's wall time.
/// ```text
/// samply record -- <test exe> profile_ours_single_threaded --ignored --nocapture
/// ```
#[test]
#[ignore]
fn profile_ours_single_threaded() {
    let _one_at_a_time = BENCH.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let codec = std::env::var("EXR_PROFILE_CODEC").unwrap_or_else(|_| "none".into());
    let comp = match codec.as_str() {
        "none" => Compression::NoCompression,
        "rle" => Compression::Rle,
        "zip" => Compression::Zip,
        "piz" => Compression::Piz,
        "dwaa" => Compression::Dwaa,
        other => panic!("EXR_PROFILE_CODEC={other}: expected none|rle|zip|piz|dwaa"),
    };
    let header = make_header(comp);
    let (pixels, fb) = make_pixels();
    let path = std::env::temp_dir().join(format!("perf_profile_ours_{codec}.exr"));
    let path = path.to_str().unwrap().to_string();
    for i in 0..REPEATS {
        let (w, r) = time_ours(&path, &header, &pixels, &fb, 0);
        println!("profile {codec} run {i}: write {w:.1}ms read {r:.1}ms");
    }
    std::fs::remove_file(&path).unwrap();
}
