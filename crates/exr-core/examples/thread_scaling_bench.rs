//! Thread-scaling baseline: real wall-clock
//! scaling of the FLAT (non-deep) scanline and tiled read+write paths at
//! increasing thread counts, for ZIP and PIZ, on one big synthetic image --
//! the input for the user's decision on where to push parallelism next
//! (deep is out of scope here).
//!
//! Methodology, matching this repo's other ad-hoc A/B tools
//! (`ours_only_bench.rs`): a real subprocess-free, in-process measurement,
//! run with no other benchmark on the machine, 2 runs, median+min per cell (median of an even-sized
//! sample here means the LARGER of the two values -- `sorted[len/2]`, same
//! convention this crate's other harnesses use, not a true statistical
//! median). WRITE timing re-builds a fresh writer from scratch at each
//! thread count (that IS what's being timed); READ timing reuses ONE
//! reference-encoded byte buffer per (storage, codec) pair across every
//! thread count, since on-disk bytes are byte-identical regardless of the
//! WRITER's thread count (this crate's own hard invariant) -- re-encoding
//! per read trial would only cost time without changing what's measured.
//!
//! Usage (release only -- this is a real multi-hundred-MB workload):
//! ```text
//! cargo run -p exr-core --release --example thread_scaling_bench
//! ```
//! `BENCH_CODEC_FILTER=zip` / `BENCH_PATH_FILTER=scanline,tiled` /
//! `BENCH_DIR_FILTER=read,write` narrow the sweep (comma-separated,
//! case-insensitive), matching `perf_codec_isolation.rs`'s
//! `BENCH_CODEC_FILTER` convention.

use std::io::Write;
use std::time::Instant;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::tiled::{TiledInputFile, TiledOutputFile};
use exr_core::{Compression, LevelMode, LevelRoundingMode, LineOrder, TileDescription};
use imath_rs::{Half, V2f};

const WIDTH: usize = 8192;
const HEIGHT: usize = 4096;
const TILE_SIZE: u32 = 64;
const THREAD_COUNTS: &[i32] = &[1, 2, 4, 8, 16, 24]; // 24 = this machine's logical core count
const REPEATS: usize = 2;

// ---------------------------------------------------------------------------
// Content generator -- same shape as `benches/parallel.rs`'s, but ALL FOUR
// channels HALF ("half RGBA"), 8 bytes/pixel, not the 2+2+2+4
// RGB-half/A-float layout `benches/parallel.rs` uses.
// ---------------------------------------------------------------------------

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

/// One interleaved (R,G,B,A, all HALF) pixel buffer, plus the matching
/// frame buffer.
fn make_pixels(w: usize, h: usize) -> (Vec<u8>, FrameBuffer) {
    let pixel_bytes = 8; // 4 x half
    let row_bytes = pixel_bytes * w;
    let mut buf = vec![0u8; row_bytes * h];
    for y in 0..h {
        for x in 0..w {
            let p = y * row_bytes + x * pixel_bytes;
            for c in 0..4 {
                let v = Half::from(sample(c, x, y, w, h)).bits();
                buf[p + c * 2..p + c * 2 + 2].copy_from_slice(&v.to_ne_bytes());
            }
        }
    }
    let mut fb = FrameBuffer::new();
    let mk = |off: usize| Slice {
        type_: PixelType::Half,
        base_offset: off as i64,
        x_stride: pixel_bytes,
        y_stride: row_bytes,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    };
    fb.insert("R", mk(0)).unwrap();
    fb.insert("G", mk(2)).unwrap();
    fb.insert("B", mk(4)).unwrap();
    fb.insert("A", mk(6)).unwrap();
    (buf, fb)
}

fn channels() -> ChannelList {
    let mut cl = ChannelList::new();
    for name in ["R", "G", "B", "A"] {
        cl.insert(name, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    }
    cl
}

fn scan_header(w: i32, h: i32, comp: Compression) -> Header {
    let mut h_ =
        Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp)
            .unwrap();
    h_.insert("channels", ChannelListAttribute::new(channels())).unwrap();
    h_
}

fn tiled_header(w: i32, h: i32, comp: Compression) -> Header {
    use exr_core::attr::TileDescriptionAttribute;
    use exr_core::header::TILED_IMAGE;
    let mut h_ = scan_header(w, h, comp);
    h_.set_type(TILED_IMAGE).unwrap();
    let td = TileDescription {
        x_size: TILE_SIZE,
        y_size: TILE_SIZE,
        mode: LevelMode::OneLevel,
        rounding_mode: LevelRoundingMode::RoundDown,
    };
    h_.insert("tiles", TileDescriptionAttribute::new(td)).unwrap();
    h_
}

fn median_min(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = v[0];
    let med = v[v.len() / 2]; // this crate's existing convention: sorted[len/2], the LARGER of 2
    (med, min)
}

struct BenchLog {
    file: std::fs::File,
    start: Instant,
}

impl BenchLog {
    fn open(path: &str) -> Self {
        let dir = std::path::Path::new(path).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("mkdir {dir:?}: {e}"));
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        Self { file, start: Instant::now() }
    }
    fn line(&mut self, msg: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let line = format!("[t+{elapsed:>7.1}s] {msg}");
        println!("{line}");
        if let Err(e) = writeln!(self.file, "{line}").and_then(|()| self.file.flush()) {
            eprintln!("BenchLog: failed to write: {e}");
        }
    }
}

/// Best-effort current total CPU load (%), via PowerShell's `Get-Counter` --
/// same diagnostic-only convention as `perf_codec_isolation.rs`'s
/// `cpu_load_pct` (never fatal if unavailable, e.g. non-Windows).
fn cpu_load_pct() -> Option<f64> {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok()
}

fn env_filter(name: &str) -> Option<Vec<String>> {
    std::env::var(name).ok().map(|v| v.split(',').map(|s| s.trim().to_lowercase()).collect())
}

fn main() {
    let log_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/bench/scaling.log");
    let mut log = BenchLog::open(log_path.to_str().unwrap());

    let codec_filter = env_filter("BENCH_CODEC_FILTER");
    let path_filter = env_filter("BENCH_PATH_FILTER");
    let dir_filter = env_filter("BENCH_DIR_FILTER");

    let all_codecs: &[(&str, Compression)] = &[("zip", Compression::Zip), ("piz", Compression::Piz)];
    let codecs: Vec<(&str, Compression)> = match &codec_filter {
        Some(f) => all_codecs.iter().copied().filter(|(n, _)| f.iter().any(|x| x == n)).collect(),
        None => all_codecs.to_vec(),
    };
    let all_paths: &[&str] = &["scanline", "tiled"];
    let paths: Vec<&str> = match &path_filter {
        Some(f) => all_paths.iter().copied().filter(|p| f.iter().any(|x| x == p)).collect(),
        None => all_paths.to_vec(),
    };
    let all_dirs: &[&str] = &["write", "read"];
    let dirs: Vec<&str> = match &dir_filter {
        Some(f) => all_dirs.iter().copied().filter(|d| f.iter().any(|x| x == d)).collect(),
        None => all_dirs.to_vec(),
    };

    log.line(&format!(
        "thread_scaling_bench starting: {WIDTH}x{HEIGHT} half RGBA ({} MB uncompressed), \
         codecs={codecs:?} paths={paths:?} dirs={dirs:?}, threads={THREAD_COUNTS:?}, \
         REPEATS={REPEATS}",
        (WIDTH * HEIGHT * 8) / (1024 * 1024)
    ));

    let (pixels, fb) = make_pixels(WIDTH, HEIGHT);
    log.line(&format!("pixel buffer generated: {} bytes", pixels.len()));

    let mut summary: Vec<String> = Vec::new();

    for &(codec_name, comp) in &codecs {
        for &path_kind in &paths {
            // Reference-encode ONCE per (path, codec) for the read sweep --
            // on-disk bytes are byte-identical regardless of the writer's
            // thread count (this crate's own hard invariant), so there is
            // no need to re-encode per read trial.
            let reference_bytes = if dirs.contains(&"read") {
                Some(match path_kind {
                    "scanline" => {
                        let header = scan_header(WIDTH as i32, HEIGHT as i32, comp);
                        let mut out = MemOStream::new();
                        let mut of = OutputFile::new(&mut out, &header).unwrap();
                        of.set_frame_buffer(&fb).unwrap();
                        of.write_pixels(&pixels, HEIGHT as i32).unwrap();
                        out.into_data()
                    }
                    "tiled" => {
                        let header = tiled_header(WIDTH as i32, HEIGHT as i32, comp);
                        let mut out = MemOStream::new();
                        let mut tof = TiledOutputFile::new(&mut out, &header).unwrap();
                        tof.set_frame_buffer(&fb).unwrap();
                        let (nx, ny) = (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                        tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
                        out.into_data()
                    }
                    _ => unreachable!(),
                })
            } else {
                None
            };
            if let Some(bytes) = &reference_bytes {
                log.line(&format!(
                    "{path_kind}/{codec_name}: reference-encoded {} bytes ({:.2}x compression)",
                    bytes.len(),
                    pixels.len() as f64 / bytes.len() as f64
                ));
            }

            for &dir in &dirs {
                let mut per_thread: Vec<(i32, f64, f64)> = Vec::new(); // (threads, median_ms, min_ms)
                for &threads in THREAD_COUNTS {
                    let mut samples = Vec::with_capacity(REPEATS);
                    for rep in 0..REPEATS {
                        let elapsed_ms = match (path_kind, dir) {
                            ("scanline", "write") => {
                                let header = scan_header(WIDTH as i32, HEIGHT as i32, comp);
                                let mut out = MemOStream::new();
                                let t0 = Instant::now();
                                let mut of =
                                    OutputFile::with_num_threads(&mut out, &header, Some(threads))
                                        .unwrap();
                                of.set_frame_buffer(&fb).unwrap();
                                of.write_pixels(&pixels, HEIGHT as i32).unwrap();
                                let _ = out.into_data();
                                t0.elapsed().as_secs_f64() * 1e3
                            }
                            ("scanline", "read") => {
                                let bytes = reference_bytes.as_ref().unwrap().clone();
                                let mut readback = vec![0u8; pixels.len()];
                                let mut is = MemIStream::new(bytes);
                                let t0 = Instant::now();
                                let mut inf =
                                    InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
                                inf.set_frame_buffer(&fb).unwrap();
                                inf.read_pixels(0, HEIGHT as i32 - 1, &mut readback).unwrap();
                                t0.elapsed().as_secs_f64() * 1e3
                            }
                            ("tiled", "write") => {
                                let header = tiled_header(WIDTH as i32, HEIGHT as i32, comp);
                                let mut out = MemOStream::new();
                                let t0 = Instant::now();
                                let mut tof = TiledOutputFile::with_num_threads(
                                    &mut out,
                                    &header,
                                    Some(threads),
                                )
                                .unwrap();
                                tof.set_frame_buffer(&fb).unwrap();
                                let (nx, ny) =
                                    (tof.num_x_tiles(0).unwrap(), tof.num_y_tiles(0).unwrap());
                                tof.write_tiles(&pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
                                let _ = out.into_data();
                                t0.elapsed().as_secs_f64() * 1e3
                            }
                            ("tiled", "read") => {
                                let bytes = reference_bytes.as_ref().unwrap().clone();
                                let mut readback = vec![0u8; pixels.len()];
                                let mut is = MemIStream::new(bytes);
                                let t0 = Instant::now();
                                let mut tif =
                                    TiledInputFile::with_num_threads(&mut is, Some(threads))
                                        .unwrap();
                                tif.set_frame_buffer(&fb).unwrap();
                                let (nx, ny) =
                                    (tif.num_x_tiles(0).unwrap(), tif.num_y_tiles(0).unwrap());
                                tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut readback).unwrap();
                                t0.elapsed().as_secs_f64() * 1e3
                            }
                            _ => unreachable!(),
                        };
                        let load = cpu_load_pct().map(|l| format!("{l:.1}%")).unwrap_or_else(|| "n/a".into());
                        log.line(&format!(
                            "{path_kind}/{codec_name}/{dir} threads={threads} rep={}/{REPEATS} \
                             {elapsed_ms:.1}ms cpu_load={load}",
                            rep + 1
                        ));
                        samples.push(elapsed_ms);
                    }
                    let (med, min) = median_min(samples);
                    per_thread.push((threads, med, min));
                }

                let (base_med, base_min) = (per_thread[0].1, per_thread[0].2);
                let mut row = format!("{path_kind:<9} {codec_name:<4} {dir:<5} ");
                for &(threads, med, min) in &per_thread {
                    row.push_str(&format!(
                        " | t{threads}={med:.0}ms({min:.0}) x{:.2}",
                        base_med / med
                    ));
                }
                log.line(&format!("SUMMARY {row}"));
                summary.push(row);
                let _ = base_min; // reported per-cell above; not separately reduced
            }
        }
    }

    log.line("thread_scaling_bench done. Full summary:");
    for row in &summary {
        log.line(row);
    }
}
