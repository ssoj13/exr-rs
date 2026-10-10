//! Parallel scaling benchmark harness.
//!
//! **Phase A** (`bench_parallel_compress`/`bench_parallel_uncompress`)
//! measures independent per-chunk `Compressor::compress`/`uncompress` calls
//! driven through `threading::run_on_global_pool` — the baseline scaling
//! curve for the entropy-heavy codecs, taken before any reader/writer was
//! wired to the pool.
//!
//! **Phase B** (`bench_whole_file_write`/`bench_whole_file_read`) now
//! exercises the REAL thing: `scanline::OutputFile::with_num_threads` /
//! `InputFile::with_num_threads` writing/reading a whole synthetic image,
//! reporting decode/encode MB/s of uncompressed bytes at each thread count —
//! the acceptance measurement for the parallel paths (scaling to 8 threads,
//! scatter share). Run it on a quiet machine: background load skews the
//! scaling curve.
//!
//! ```text
//! cargo bench -p exr-core --bench parallel -- --warm-up-time 1 --measurement-time 2
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::compressor::new_compressor;
use exr_core::header::{Header, SCANLINE_IMAGE};
use exr_core::threading::{run_on_global_pool, set_global_thread_count, ThreadCount};
use imath_rs::{Half, V2f};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Representative content — same generator shape as `benches/codecs.rs`, kept
// independent (not `include!`d) so this bench stays self-contained and can
// diverge (e.g. bigger images) without touching the codec micro-bench.
// ---------------------------------------------------------------------------

#[inline]
fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

#[inline]
fn sample(c: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let gx = x as f32 / w as f32;
    let gy = y as f32 / h as f32;
    let grad = 0.15 + 0.7 * (gx * 0.6 + gy * 0.4) + 0.1 * (c as f32) * 0.25;
    let n = hash01(
        (x as u32)
            .wrapping_mul(73_856_093)
            ^ (y as u32).wrapping_mul(19_349_663)
            ^ (c as u32).wrapping_mul(83_492_791),
    );
    (grad + 0.06 * (n - 0.5)).clamp(0.0, 8.0)
}

fn bench_channels() -> ChannelList {
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    cl
}

fn scan_header(w: i32, h: i32, comp: Compression) -> Header {
    let mut h_ =
        Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp)
            .unwrap();
    h_.insert("channels", ChannelListAttribute::new(bench_channels())).unwrap();
    h_.set_type(SCANLINE_IMAGE).unwrap();
    h_
}

/// One scanline-interleaved XDR chunk of `rows` lines, `w` wide, in header
/// (A,B,G,R) order — the same byte layout `OutputFile::write_pixels` gathers
/// (see `benches/codecs.rs`'s module docs for the full rationale).
fn build_chunk(w: usize, rows: usize, y0: usize) -> Vec<u8> {
    // Header order is alphabetical: A (float), B, G, R (half).
    let order: [(usize, PixelType); 4] = [
        (3, PixelType::Float),
        (2, PixelType::Half),
        (1, PixelType::Half),
        (0, PixelType::Half),
    ];
    let mut out = Vec::with_capacity(w * rows * 10);
    for y in y0..y0 + rows {
        for &(c, ty) in &order {
            for x in 0..w {
                let v = sample(c, x, y, w, y0 + rows);
                match ty {
                    PixelType::Half => out.extend_from_slice(&Half::from(v).bits().to_le_bytes()),
                    PixelType::Float => out.extend_from_slice(&v.to_le_bytes()),
                    PixelType::Uint | PixelType::NumPixelTypes => unreachable!(),
                }
            }
        }
    }
    out
}

/// A codec under test with its per-chunk block height (`numScanLines`).
struct CodecCase {
    name: &'static str,
    comp: Compression,
    block: usize,
}

/// Entropy-heavy codecs where the acceptance criteria (>=0.8x C++
/// throughput, near-linear scaling to 8 threads) actually apply. NONE is
/// included as the zero-overhead baseline the scatter/gather share (the
/// "scatter share below 15% of wall time" check) is measured against.
const CODECS: &[CodecCase] = &[
    CodecCase { name: "NONE", comp: Compression::NoCompression, block: 1 },
    CodecCase { name: "ZIP", comp: Compression::Zip, block: 16 },
    CodecCase { name: "PIZ", comp: Compression::Piz, block: 32 },
    CodecCase { name: "DWAA", comp: Compression::Dwaa, block: 32 },
];

const THREAD_COUNTS: &[i32] = &[0, 1, 2, 4, 8];

/// Image width and number of independent chunks. `N_CHUNKS` should be well
/// above the largest thread count so every worker stays busy; each chunk is
/// one codec block, matching the real per-chunk unit of work.
const W: usize = 1024;
const N_CHUNKS: usize = 32;

fn bench_parallel_compress(c: &mut Criterion) {
    let mut grp = c.benchmark_group("parallel_compress");
    for case in CODECS {
        let rows = case.block;
        let chunks: Vec<Vec<u8>> = (0..N_CHUNKS).map(|i| build_chunk(W, rows, i * rows)).collect();
        let header = scan_header(W as i32, rows as i32, case.comp);
        let uncompressed_total = (chunks.iter().map(|ch| ch.len()).sum::<usize>()) as u64;

        for &threads in THREAD_COUNTS {
            // Resize the ONE process-wide pool (no per-file pool — see
            // `threading.rs` module docs). Criterion runs one
            // benchmark function at a time, so there is no concurrent bench
            // racing this resize.
            set_global_thread_count(threads).unwrap();
            let count = ThreadCount::capture(None).unwrap();
            let label = format!("{}/threads={}", case.name, threads);
            grp.throughput(Throughput::Bytes(uncompressed_total));
            grp.bench_with_input(BenchmarkId::new("compress", &label), &chunks, |b, chunks| {
                b.iter(|| {
                    run_on_global_pool(count, || {
                        let total: usize = chunks
                            .par_iter()
                            .map(|chunk| {
                                let mut cmp = new_compressor(case.comp, W * 14, &header).unwrap();
                                match cmp.as_mut() {
                                    Some(cc) => cc.compress(black_box(chunk), 0).unwrap().len(),
                                    None => chunk.len(),
                                }
                            })
                            .sum();
                        black_box(total)
                    })
                    .unwrap()
                });
            });
        }
    }
    grp.finish();
}

fn bench_parallel_uncompress(c: &mut Criterion) {
    let mut grp = c.benchmark_group("parallel_uncompress");
    for case in CODECS {
        let rows = case.block;
        let chunks: Vec<Vec<u8>> = (0..N_CHUNKS).map(|i| build_chunk(W, rows, i * rows)).collect();
        let header = scan_header(W as i32, rows as i32, case.comp);
        let uncompressed_total = (chunks.iter().map(|ch| ch.len()).sum::<usize>()) as u64;

        // Pre-compress each chunk once (outside the timed loop).
        let compressed: Vec<Vec<u8>> = chunks
            .iter()
            .map(|chunk| {
                let mut cmp = new_compressor(case.comp, W * 14, &header).unwrap();
                match cmp.as_mut() {
                    Some(cc) => cc.compress(chunk, 0).unwrap().to_vec(),
                    None => chunk.clone(),
                }
            })
            .collect();

        for &threads in THREAD_COUNTS {
            set_global_thread_count(threads).unwrap();
            let count = ThreadCount::capture(None).unwrap();
            let label = format!("{}/threads={}", case.name, threads);
            grp.throughput(Throughput::Bytes(uncompressed_total));
            grp.bench_with_input(
                BenchmarkId::new("uncompress", &label),
                &compressed,
                |b, compressed| {
                    b.iter(|| {
                        run_on_global_pool(count, || {
                            let total: usize = compressed
                                .par_iter()
                                .map(|chunk| {
                                    let mut cmp = new_compressor(case.comp, W * 14, &header).unwrap();
                                    match cmp.as_mut() {
                                        Some(cc) => cc.uncompress(black_box(chunk), 0, W * 14).unwrap().len(),
                                        None => chunk.len(),
                                    }
                                })
                                .sum();
                            black_box(total)
                        })
                        .unwrap()
                    });
                },
            );
        }
    }
    grp.finish();
}

// ---------------------------------------------------------------------------
// Phase B: whole-file scanline write/read throughput at each thread count.
// ---------------------------------------------------------------------------

use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};

/// One interleaved (R,G,B half + A float) pixel buffer for a `w`x`h` image,
/// plus the matching frame buffer — the same layout `api::Image` and most
/// real callers use (as opposed to `benches/codecs.rs`'s raw per-channel
/// chunk bytes, which model the on-disk layout instead of a caller's buffer).
fn whole_image_pixels(w: usize, h: usize) -> (Vec<u8>, FrameBuffer) {
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

/// Image side length for the whole-file benches. 2048 keeps a full
/// {compress+decompress} x {5 codecs} x {5 thread counts} matrix inside a
/// reasonable CI/dev-machine bench run; re-run manually at 4096/8192 for a
/// closer look at large-image scaling (`--bench parallel -- whole_file`).
const WHOLE_W: usize = 2048;
const WHOLE_H: usize = 2048;

fn bench_whole_file_write(c: &mut Criterion) {
    let mut grp = c.benchmark_group("whole_file_write");
    let (pixels, fb) = whole_image_pixels(WHOLE_W, WHOLE_H);
    let uncompressed_total = pixels.len() as u64;

    for case in CODECS {
        let header = scan_header(WHOLE_W as i32, WHOLE_H as i32, case.comp);
        for &threads in THREAD_COUNTS {
            let label = format!("{}/threads={}", case.name, threads);
            grp.throughput(Throughput::Bytes(uncompressed_total));
            grp.bench_with_input(BenchmarkId::new("write", &label), &pixels, |b, pixels| {
                b.iter(|| {
                    let mut out = MemOStream::new();
                    let mut of =
                        OutputFile::with_num_threads(&mut out, &header, Some(threads)).unwrap();
                    of.set_frame_buffer(&fb).unwrap();
                    of.write_pixels(black_box(pixels), WHOLE_H as i32).unwrap();
                    black_box(out.into_data().len())
                });
            });
        }
    }
    grp.finish();
}

fn bench_whole_file_read(c: &mut Criterion) {
    let mut grp = c.benchmark_group("whole_file_read");
    let (pixels, fb) = whole_image_pixels(WHOLE_W, WHOLE_H);
    let uncompressed_total = pixels.len() as u64;

    for case in CODECS {
        let header = scan_header(WHOLE_W as i32, WHOLE_H as i32, case.comp);
        let bytes = {
            let mut out = MemOStream::new();
            let mut of = OutputFile::new(&mut out, &header).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, WHOLE_H as i32).unwrap();
            out.into_data()
        };

        for &threads in THREAD_COUNTS {
            let label = format!("{}/threads={}", case.name, threads);
            grp.throughput(Throughput::Bytes(uncompressed_total));
            grp.bench_with_input(BenchmarkId::new("read", &label), &bytes, |b, bytes| {
                b.iter(|| {
                    let mut is = MemIStream::new(bytes.clone());
                    let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).unwrap();
                    let mut readback = vec![0u8; pixels.len()];
                    inf.set_frame_buffer(&fb).unwrap();
                    inf.read_pixels(0, WHOLE_H as i32 - 1, black_box(&mut readback)).unwrap();
                    black_box(readback.len())
                });
            });
        }
    }
    grp.finish();
}

criterion_group!(
    benches,
    bench_parallel_compress,
    bench_parallel_uncompress,
    bench_whole_file_write,
    bench_whole_file_read
);
criterion_main!(benches);
