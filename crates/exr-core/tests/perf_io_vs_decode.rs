//! Read-side I/O share measurement: before accepting a "decode-parallel-only,
//! no read-ahead pipelining" simplification (no double-buffered A/B window
//! overlapping the I/O of one window with the decode of the previous),
//! measure how much of the real, whole-file, PARALLEL read wall time is I/O
//! (seek+read off the stream, always serial, on the calling thread) versus
//! decode+scatter (parallelized), at 16k-ish widths, across codecs, on a
//! real on-disk file (not `MemIStream`, which has no real I/O cost to hide).
//!
//! **Honest limitation, stated up front:** there is no portable, unprivileged
//! way to drop the Windows/OS page cache from a test, so "cold" here means
//! "first read right after writing the file" (some of it may still be in the
//! page cache from the write) and "warm" means "read the same file again
//! immediately after". This UNDER-estimates true cold-cache I/O cost (a real
//! cold read from spinning/network storage would be slower), so if the I/O
//! share already looks non-trivial here, it is a LOWER bound — real cold
//! reads would look worse, not better, for the "skip read-ahead" case.
//!
//! Method per codec: write a `WIDTH x HEIGHT` synthetic RGBA (half+float)
//! image to a real temp file (`num_threads = 0`, so the write's own
//! parallelism doesn't confound the read measurement), then time:
//! - `io_only`: loop calling `InputFile::raw_pixel_data` for every chunk —
//!   seek + read the packed bytes, NO decompress, NO scatter. This isolates
//!   exactly the work `fetch_chunk`/`read_pixels_parallel`'s calling thread
//!   does that the parallel workers never touch.
//! - `full_serial` (`num_threads = 0`): the real `read_pixels` — I/O +
//!   decode + scatter, nothing parallel.
//! - `full_parallel` (`num_threads = 8`): the real `read_pixels` — I/O stays
//!   on the calling thread exactly as `io_only` measured; decode+scatter
//!   now overlaps across up to 8 workers.
//!
//! `io_share_at_8_threads = io_only / full_parallel` is the number that
//! actually matters for the read-ahead decision: if I/O this large a
//! fraction of the ALREADY-PARALLEL wall time, overlapping it with the next
//! window's decode (the read-ahead this file's simplification skips) would
//! meaningfully shorten wall time; if I/O is a small sliver even next to a
//! now-much-shorter parallel decode, it is not worth the complexity yet.
//!
//! Run explicitly (ignored by default — writes real temp files, several
//! seconds per codec):
//! ```text
//! cargo test -p exr-core --release --test perf_io_vs_decode -- --ignored --nocapture --test-threads=1
//! ```

use std::time::{Duration, Instant};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, SCANLINE_IMAGE};
use exr_core::io::{StdIfStream, StdOfStream};
use exr_core::scanline::{InputFile, OutputFile};
use imath_rs::{Half, V2f};

const WIDTH: i32 = 16384;
/// A genuine 16k-class image (16384x8192, ~1.3 GB uncompressed per file) —
/// 2048x2048 does not answer
/// "is read-ahead needed at the sizes this repo targets" (16k-65k).
const HEIGHT: i32 = 8192;
const PARALLEL_THREADS: i32 = 8;

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

/// Time `io_only`: seek+read every chunk's raw bytes EXACTLY ONCE, no
/// decode/scatter — matching what the real reader's `fetch_chunk` does
/// (bug found + fixed during this measurement: stepping by 1 line
/// regardless of the codec's block size re-fetched the SAME multi-line
/// chunk `block` times, e.g. 16x too many seeks+reads for ZIP — inflating
/// `io_only` past even the full decode+scatter time and making the
/// io/serial ratio meaningless above 1 line/chunk).
fn time_io_only(path: &str, lines_per_chunk: i32) -> Duration {
    let mut is = StdIfStream::open(path).unwrap();
    let mut inf = InputFile::new(&mut is).unwrap();
    let t0 = Instant::now();
    let mut y = 0i32;
    while y < HEIGHT {
        let (_min_y, _data) = inf.raw_pixel_data(y).unwrap();
        y += lines_per_chunk;
    }
    t0.elapsed()
}

fn time_full_read(path: &str, num_threads: i32, fb: &FrameBuffer, out_len: usize) -> Duration {
    let mut is = StdIfStream::open(path).unwrap();
    let mut inf = InputFile::with_num_threads(&mut is, Some(num_threads)).unwrap();
    inf.set_frame_buffer(fb).unwrap();
    let mut readback = vec![0u8; out_len];
    let t0 = Instant::now();
    inf.read_pixels(0, HEIGHT - 1, &mut readback).unwrap();
    t0.elapsed()
}

fn run_one(name: &str, comp: Compression) {
    let header = make_header(comp);
    let (pixels, fb) = make_pixels();

    let dir = std::env::temp_dir();
    let path = dir.join(format!("perf_io_vs_decode_{name}.exr"));
    let path_str = path.to_str().unwrap().to_string();

    {
        let mut os = StdOfStream::create(&path_str).unwrap();
        let mut of = OutputFile::with_num_threads(&mut os, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, HEIGHT).unwrap();
    }

    let lines_per_chunk = comp.num_scanlines();

    // "Cold" = first read right after writing (see module doc caveat).
    let cold_io = time_io_only(&path_str, lines_per_chunk);
    let cold_serial = time_full_read(&path_str, 0, &fb, pixels.len());
    let cold_parallel = time_full_read(&path_str, PARALLEL_THREADS, &fb, pixels.len());

    // "Warm" = read again immediately (OS page cache should hold the file).
    let warm_io = time_io_only(&path_str, lines_per_chunk);
    let warm_serial = time_full_read(&path_str, 0, &fb, pixels.len());
    let warm_parallel = time_full_read(&path_str, PARALLEL_THREADS, &fb, pixels.len());

    let _ = std::fs::remove_file(&path_str);

    let pct = |num: Duration, den: Duration| 100.0 * num.as_secs_f64() / den.as_secs_f64();
    println!(
        "{name:>10}  cold: io={:>7.2}ms serial={:>7.2}ms par8={:>7.2}ms  io/serial={:>5.1}%  io/par8={:>5.1}%",
        cold_io.as_secs_f64() * 1e3,
        cold_serial.as_secs_f64() * 1e3,
        cold_parallel.as_secs_f64() * 1e3,
        pct(cold_io, cold_serial),
        pct(cold_io, cold_parallel),
    );
    println!(
        "{name:>10}  warm: io={:>7.2}ms serial={:>7.2}ms par8={:>7.2}ms  io/serial={:>5.1}%  io/par8={:>5.1}%",
        warm_io.as_secs_f64() * 1e3,
        warm_serial.as_secs_f64() * 1e3,
        warm_parallel.as_secs_f64() * 1e3,
        pct(warm_io, warm_serial),
        pct(warm_io, warm_parallel),
    );
}

#[test]
#[ignore] // real file I/O + several seconds per codec; run explicitly, see module doc
fn io_share_at_16k_width_across_codecs() {
    println!(
        "\nW3.9 phase B read-side I/O share, {}x{} per codec (par8 = num_threads={}):",
        WIDTH, HEIGHT, PARALLEL_THREADS
    );
    run_one("NONE", Compression::NoCompression);
    run_one("RLE", Compression::Rle);
    run_one("ZIP", Compression::Zip);
    run_one("PIZ", Compression::Piz);
    run_one("DWAA", Compression::Dwaa);
    run_one("HTJ2K256", Compression::Htj2k256);
}

/// The read-ahead pipelining implemented for
/// `read_pixels_parallel` (`scanline/input.rs`) spawns one `std::thread::scope`
/// worker PER WINDOW TRANSITION (`window = max_in_flight() = 2*num_threads`).
/// At the many-small-chunks extreme — NONE compression (1 scanline/chunk) at
/// a large height — that is a LOT of transitions: 65536 height / 16
/// (max_in_flight at num_threads=8) = 4096 OS thread spawns for one
/// `read_pixels` call, each guarding a window whose decode+scatter work is
/// itself nearly free (NONE has no decompression, just a byte copy per
/// line). If OS thread-spawn overhead (microseconds each, but 4096 of them)
/// is a meaningful fraction of the total wall time here, the "single
/// dedicated OS thread per window transition" design does NOT scale down to
/// this end of the workload, and the fix is a single persistent reader
/// thread for the whole `read_pixels` call (a channel/handoff of windows)
/// instead of one thread per transition.
///
/// Width is kept small (256) so the file stays a manageable ~167 MB despite
/// the enormous scanline count — this test is about CHUNK COUNT overhead,
/// not total data volume.
#[test]
#[ignore] // real file I/O; run explicitly
fn many_small_windows_thread_spawn_overhead_65536_height_none() {
    const W: i32 = 256;
    const H: i32 = 65536;

    let mut header = Header::with_dimensions(
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
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    header.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    header.set_type(SCANLINE_IMAGE).unwrap();

    let pixel_bytes = 2 + 2 + 2 + 4;
    let row_bytes = pixel_bytes * W as usize;
    let pixels = vec![0u8; row_bytes * H as usize];
    let mut fb = FrameBuffer::new();
    let mk = |ty: PixelType, off: usize| Slice {
        type_: ty, base_offset: off as i64, x_stride: pixel_bytes, y_stride: row_bytes,
        x_sampling: 1, y_sampling: 1, fill_value: 0.0, x_tile_coords: false, y_tile_coords: false,
    };
    fb.insert("R", mk(PixelType::Half, 0)).unwrap();
    fb.insert("G", mk(PixelType::Half, 2)).unwrap();
    fb.insert("B", mk(PixelType::Half, 4)).unwrap();
    fb.insert("A", mk(PixelType::Float, 6)).unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("perf_many_small_windows_65536_none.exr");
    let path_str = path.to_str().unwrap().to_string();

    {
        let mut os = StdOfStream::create(&path_str).unwrap();
        let mut of = OutputFile::with_num_threads(&mut os, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, H).unwrap();
    }

    let time_read = |num_threads: i32| -> Duration {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = InputFile::with_num_threads(&mut is, Some(num_threads)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut readback = vec![0u8; pixels.len()];
        let t0 = Instant::now();
        inf.read_pixels(0, H - 1, &mut readback).unwrap();
        t0.elapsed()
    };

    // Warm up the page cache with one throwaway read so all three
    // measurements see the same (warm) I/O cost, isolating the parallel
    // dispatch overhead as the variable being measured.
    let _ = time_read(0);
    let serial = time_read(0);
    let par8 = time_read(PARALLEL_THREADS);

    let _ = std::fs::remove_file(&path_str);

    // NOTE: window count is no longer a fixed `2*num_threads` chunks — see
    // `ThreadCount::window_size_for_bytes` (byte-based window sizing, a
    // later follow-up than the "4096 window transitions" figure this test's
    // name/history refers to): a window now grows until it covers at least
    // 256 KiB of raw chunk bytes, so a NONE 1-line-chunk run like this one
    // batches far more than `2*num_threads` chunks per window.
    println!(
        "\nmany small windows, {W}x{H} NONE (1 line/chunk): serial={:.2}ms par8={:.2}ms \
         (byte-based windowing, num_threads={PARALLEL_THREADS}) speedup={:.2}x",
        serial.as_secs_f64() * 1e3,
        par8.as_secs_f64() * 1e3,
        serial.as_secs_f64() / par8.as_secs_f64(),
    );
}
