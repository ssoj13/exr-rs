//! Deep reader I/O share: measures whether read-ahead pipelining would pay
//! off on the deep readers, rather than skipping it unmeasured. Same method as `tests/perf_io_vs_decode.rs` (flat scanline):
//! write a real on-disk deep file, then time
//! * `io_only` — every chunk's RAW bytes (`raw_pixel_data`/`raw_tile_data`,
//!   no decompress, no scatter) — the work that stays on the calling thread
//!   in the current decode-parallel-only readers;
//! * `full_parallel` (`num_threads = 8`) — the real `read_pixels`/
//!   `read_tiles`, decode+scatter parallelized.
//!
//! `io_share = io_only / full_parallel` decides whether pipelined read-ahead
//! (like the flat scanline/tiled readers already have) is worth adding to
//! the deep readers too.
//!
//! # Why not literally 16384x8192 like the flat measurement
//! A deep image's total data volume scales with `width * height *
//! avg_sample_count`, not just `width * height` — a literal 16k x 8k deep
//! image at even a modest average sample count would be tens of GB and take
//! far too long to write/read in a test. This picks a size (4096x4096, avg
//! ~3.5 samples/pixel, one HALF channel) that is still two orders of
//! magnitude past a toy image (~470 MB of sample data) and large enough that
//! per-chunk fixed overhead (seek+leader-read syscalls, allocation) does not
//! dominate the measurement, while staying practical to run every gate.
//!
//! Run explicitly (ignored by default — real file I/O, tens of seconds):
//! ```text
//! cargo test -p exr-core --release --test perf_deep_io_vs_decode -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType, TileDescription, TileDescriptionAttribute};
use exr_core::attr::{LevelMode, LevelRoundingMode};
use exr_core::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
use exr_core::deep::{DeepScanLineInputFile, DeepScanLineOutputFile, DeepTiledInputFile, DeepTiledOutputFile};
use exr_core::frame_buffer::Slice;
use exr_core::header::{Header, DEEP_SCANLINE, DEEP_TILE};
use exr_core::io::{StdIfStream, StdOfStream};
use imath_rs::{Half, V2f};

const WIDTH: i32 = 4096;
const HEIGHT: i32 = 4096;
const PARALLEL_THREADS: i32 = 8;

/// Per-pixel sample count: 0..=6 cycling (average 3.0), so both zero-sample
/// and multi-sample pixels appear, matching the sample-table module's own
/// test convention.
fn count_at(idx: i32) -> u32 {
    (idx % 7) as u32
}

fn make_deep_scanline_header(comp: Compression) -> Header {
    let mut h = Header::with_dimensions(
        WIDTH, HEIGHT, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h.set_type(DEEP_SCANLINE).unwrap();
    h
}

fn make_deep_tiled_header(comp: Compression) -> Header {
    let mut h = make_deep_scanline_header(comp);
    h.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: 64,
            y_size: 64,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .unwrap();
    h.set_type(DEEP_TILE).unwrap();
    h
}

/// Build the flat sample-count buffer + sample heap for a `WIDTH x HEIGHT`
/// deep image, one HALF channel.
fn make_deep_pixels() -> (Vec<u8>, Vec<u8>, DeepFrameBuffer) {
    let n = (WIDTH * HEIGHT) as usize;
    let counts: Vec<u32> = (0..n as i32).map(count_at).collect();

    let mut sc = Vec::with_capacity(n * 4);
    for &c in &counts {
        sc.extend_from_slice(&c.to_le_bytes());
    }

    let mut data = Vec::new();
    for (pix, &c) in counts.iter().enumerate() {
        for i in 0..c {
            let v = (pix as f32) * 0.001 + i as f32;
            data.extend_from_slice(&Half::from(v).bits().to_le_bytes());
        }
    }

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
    fb.insert(
        "A",
        DeepSlice::new(PixelType::Half, 0, 8, 8 * WIDTH as usize, 2, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();

    (sc, data, fb)
}

fn run_deep_scanline(name: &str, comp: Compression) {
    let header = make_deep_scanline_header(comp);
    let (sc, data, fb) = make_deep_pixels();
    let n = (WIDTH * HEIGHT) as usize;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("perf_deep_scanline_{name}.exr"));
    let path_str = path.to_str().unwrap().to_string();

    {
        let mut os = StdOfStream::create(&path_str).unwrap();
        let mut of = DeepScanLineOutputFile::with_num_threads(&mut os, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&sc, &data, HEIGHT).unwrap();
    }

    let io_only = {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = DeepScanLineInputFile::new(&mut is).unwrap();
        let t0 = Instant::now();
        for y in 0..HEIGHT {
            let _ = inf.raw_pixel_data(y).unwrap();
        }
        t0.elapsed()
    };

    let full_parallel = {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = DeepScanLineInputFile::with_num_threads(&mut is, Some(PARALLEL_THREADS)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut sc_read = vec![0u8; sc.len()];
        let mut data_read = vec![0u8; data.len()];
        let t0 = Instant::now();
        inf.read_pixel_sample_counts(0, HEIGHT - 1, &mut sc_read).unwrap();
        inf.read_pixels(0, HEIGHT - 1, &sc_read, &mut data_read).unwrap();
        t0.elapsed()
    };

    let _ = std::fs::remove_file(&path_str);
    report("deep scanline", name, n, io_only, full_parallel);
}

fn run_deep_tiled(name: &str, comp: Compression) {
    let header = make_deep_tiled_header(comp);
    let (sc, data, fb) = make_deep_pixels();
    let n = (WIDTH * HEIGHT) as usize;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("perf_deep_tiled_{name}.exr"));
    let path_str = path.to_str().unwrap().to_string();

    let (nx, ny);
    {
        let mut os = StdOfStream::create(&path_str).unwrap();
        let mut of = DeepTiledOutputFile::with_num_threads(&mut os, &header, Some(0)).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        nx = of.num_x_tiles(0).unwrap();
        ny = of.num_y_tiles(0).unwrap();
        of.write_tiles(&sc, &data, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
    }

    let io_only = {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = DeepTiledInputFile::new(&mut is).unwrap();
        let t0 = Instant::now();
        for ty in 0..ny {
            for tx in 0..nx {
                let _ = inf.raw_tile_data(tx, ty, 0, 0).unwrap();
            }
        }
        t0.elapsed()
    };

    let full_parallel = {
        let mut is = StdIfStream::open(&path_str).unwrap();
        let mut inf = DeepTiledInputFile::with_num_threads(&mut is, Some(PARALLEL_THREADS)).unwrap();
        inf.set_frame_buffer(&fb).unwrap();
        let mut sc_read = vec![0u8; sc.len()];
        let mut data_read = vec![0u8; data.len()];
        let t0 = Instant::now();
        inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut sc_read).unwrap();
        inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &sc_read, &mut data_read).unwrap();
        t0.elapsed()
    };

    let _ = std::fs::remove_file(&path_str);
    report("deep tiled", name, n, io_only, full_parallel);
}

fn report(kind: &str, name: &str, n_pixels: usize, io_only: Duration, full_parallel: Duration) {
    let share = 100.0 * io_only.as_secs_f64() / full_parallel.as_secs_f64();
    println!(
        "{kind:>13} {name:>5}  ({n_pixels} px): io={:>8.2}ms  par8={:>8.2}ms  io/par8={:>5.1}%",
        io_only.as_secs_f64() * 1e3,
        full_parallel.as_secs_f64() * 1e3,
        share,
    );
}

#[test]
#[ignore] // real file I/O; run explicitly, see module doc
fn deep_io_share_across_codecs() {
    println!(
        "\nW3.9 phase B deep reader I/O share, {WIDTH}x{HEIGHT} (avg ~3 samples/pixel, \
         par8 = num_threads={PARALLEL_THREADS}):"
    );
    run_deep_scanline("NONE", Compression::NoCompression);
    run_deep_scanline("ZIPS", Compression::Zips);
    run_deep_tiled("NONE", Compression::NoCompression);
    run_deep_tiled("ZIPS", Compression::Zips);
}
