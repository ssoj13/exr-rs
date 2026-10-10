//! Ad-hoc A/B tool: time OUR OWN
//! compress/uncompress for one codec, one repeat, no C++ oracle involved --
//! meant to be built at two different commits (two worktrees) and invoked
//! alternately so the same machine noise hits both sides equally, answering
//! "did the allocation/codec-simd work regress our own codec speed"
//! without a noisy machine muddying an ours-vs-C++ comparison.
//!
//! Data generation, header, and chunk layout are copied verbatim from
//! `tests/perf_codec_isolation.rs`'s `sample`/`hash01`/`make_header`/
//! `make_chunk_bytes`/`bench_size`/`time_ours` so the OLD and NEW binaries
//! measure the exact same synthetic input -- the only difference between
//! the two builds is the codec/compressor source itself.
//!
//! Usage: `ours_only_bench <zip|piz|dwaa> [repeats]` -- prints one line per
//! repeat: `repeat=N compress_ms=X.X uncompress_ms=Y.Y`.
//!
//! # Running an OLD-commit-vs-NEW-commit A/B (regression bisect)
//!
//! 1. `git worktree add --detach ../exr-old-bench <old-commit>` (a SEPARATE
//!    checkout, so the old commit's crate builds without disturbing this
//!    one).
//! 2. Build both: `cargo build -p exr-core --release --example
//!    ours_only_bench` in each worktree (the NEW checkout already has this
//!    file; the OLD one needs it copied in first if the commit predates it).
//! 3. Make sure no other benchmark is running on the machine.
//! 4. Alternate `old.exe <codec> 1` / `new.exe <codec> 1` per repeat (never
//!    run all of one side then all of the other -- interleaving means
//!    background machine load hits both sides equally within each repeat).
//!    5 repeats x 2 runs is the project's usual minimum; compute the
//!    new/old ratio's MEDIAN and MIN per direction per run. A ratio > 1.05
//!    that repeats across BOTH runs (not just one) is a real regression,
//!    not noise.
//! 5. `git worktree remove` the old checkout once it's no longer needed (or
//!    leave it for a further bisect step at an even older commit).

use std::time::Instant;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::compressor::new_compressor;
use exr_core::header::Header;
use imath_rs::{Half, V2f};

const WIDTH: i32 = 16384;
const HEIGHT: i32 = 8192;
const TOTAL_LINES: i32 = 1024;

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
    h
}

fn make_chunk_bytes(width: i32, first_y: i32, lines: i32) -> Vec<u8> {
    let (w, h) = (width as usize, HEIGHT as usize);
    let mut buf = Vec::with_capacity(lines as usize * w * 10);
    for y in first_y..first_y + lines {
        let yu = y as usize;
        for x in 0..w {
            buf.extend_from_slice(&sample(3, x, yu, w, h).to_le_bytes());
        }
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(2, x, yu, w, h)).bits().to_le_bytes());
        }
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(1, x, yu, w, h)).bits().to_le_bytes());
        }
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(0, x, yu, w, h)).bits().to_le_bytes());
        }
    }
    buf
}

fn bench_size(comp: Compression) -> (usize, i32) {
    let lines_per_chunk = comp.num_scanlines();
    let n_chunks = (((TOTAL_LINES + lines_per_chunk - 1) / lines_per_chunk).max(1)) as usize;
    (n_chunks, n_chunks as i32 * lines_per_chunk)
}

fn time_ours(comp: Compression) -> (f64, f64) {
    let header = make_header(comp);
    let lines_per_chunk = comp.num_scanlines();
    let max_bytes_per_line = 10 * WIDTH as usize;
    let (n_chunks, _actual_lines) = bench_size(comp);

    let chunks: Vec<Vec<u8>> = (0..n_chunks)
        .map(|i| make_chunk_bytes(WIDTH, i as i32 * lines_per_chunk, lines_per_chunk))
        .collect();

    let t0 = Instant::now();
    let mut packed: Vec<Vec<u8>> = Vec::with_capacity(n_chunks);
    for (i, chunk) in chunks.iter().enumerate() {
        let min_y = i as i32 * lines_per_chunk;
        match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => packed.push(c.compress(chunk, min_y).unwrap().to_vec()),
            None => packed.push(chunk.clone()),
        }
    }
    let compress_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    for (i, p) in packed.iter().enumerate() {
        let min_y = i as i32 * lines_per_chunk;
        match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => {
                let out = c.uncompress(p, min_y, chunks[i].len()).unwrap();
                assert_eq!(out.len(), chunks[i].len());
            }
            None => assert_eq!(p.len(), chunks[i].len()),
        }
    }
    let uncompress_ms = t1.elapsed().as_secs_f64() * 1e3;

    (compress_ms, uncompress_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let codec_name = args.get(1).map(String::as_str).unwrap_or("zip").to_lowercase();
    let repeats: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let comp = match codec_name.as_str() {
        "zip" => Compression::Zip,
        "piz" => Compression::Piz,
        "dwaa" => Compression::Dwaa,
        other => {
            eprintln!("unknown codec {other} (expected zip|piz|dwaa)");
            std::process::exit(2);
        }
    };
    for rep in 1..=repeats {
        let (compress_ms, uncompress_ms) = time_ours(comp);
        println!("repeat={rep} compress_ms={compress_ms:.3} uncompress_ms={uncompress_ms:.3}");
    }
}
