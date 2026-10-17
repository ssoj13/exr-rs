//! Temporal decode throughput: GPU-resident (Tier A: INTRA + SKIP tiles resolved per tile) vs the CPU decoders, on
//! `gop=8` clips with a SKIP-dominated and a P-dominated content mix.
//!
//! **Why:** `.bughunt/design_temporal_gpu.md` decided "Tier A, then MEASURE, then decide B/C/D". This prints what the
//! decision needs: the tile-kind mix (SKIP fraction), bytes per codec/gop, CPU playback throughput (stateful
//! `Decoder`: every frame decoded once, the best CPU case; and stateless `decode_frame(k)`: what a viewer loader pays),
//! and the GPU-resident throughput for the clips whose temporal class the GPU path accepts. For P-dominated clips
//! the all-INTRA twin (`gop=1`, same content, same codec) is the UPPER BOUND of what a GPU residual add (Tier B)
//! could reach: decode cost of the INTRA tiles + one memory-bound add pass.
//!
//! Run: `cargo test --release -p exv-gpu --test temporal_bench -- --ignored --nocapture --test-threads=1`
//! (`EXV_BENCH_W`/`EXV_BENCH_H`/`EXV_BENCH_FRAMES` override 4096 x 2048 x 16).

use std::time::Instant;

use exv_codec::Profile;
use exv_codec::frame::{TILE_INTRA, TILE_PRED, TILE_SKIP};
use exv_codec::image::Image;
use exv_format::half::f32_to_half;
use exv_format::{Channel, Dtype, Header};
use exv_gpu::{Chans, Clip, ContentKey, DECODE_BUDGET, DisplayKey, ExvResidentAtlas, GpuContext, Want};
use exv_video::{Decoder, TemporalClass, VideoIndex, decode_frame, frame_view, temporal_class, write_video};

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Deterministic per-pixel hash (static film-grain-like noise).
fn hash(x: u32, y: u32) -> u32 {
    let h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA6B);
    h ^ (h >> 15)
}

/// One RGB F16 frame: a smooth HDR ramp with STATIC noise in the low mantissa bits (every frame shares it, so a
/// temporal residual cancels it) plus `f(x, y)`, the frame-dependent part (a moving block or a slow drift).
fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> Image {
    let planes: Vec<Vec<u8>> = (0..3u32)
        .map(|c| {
            (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let ramp = 2f32.powf((x as f32 / w as f32 + y as f32 / h as f32) * 3.0 - 3.0) * (1.0 - 0.2 * c as f32);
                    let bits = f32_to_half(ramp + f(x, y)) ^ (hash(x, y + c) & 0x7) as u16;
                    bits.to_le_bytes()
                })
                .collect::<Vec<u8>>()
        })
        .collect();
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: ["R", "G", "B"].iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F16 }).collect(),
    };
    Image::new(header, planes).expect("image")
}

/// A static plate with a `bw x bh` block that moves 64 px per frame (changes only the tiles it touches).
fn moving_block(w: u32, h: u32, n: u32, bw: u32, bh: u32) -> Vec<Image> {
    (0..n)
        .map(|k| {
            let (x0, y0) = ((k * 64) % (w - bw).max(1), h / 4);
            frame(w, h, move |x, y| if (x0..x0 + bw).contains(&x) && (y0..y0 + bh).contains(&y) { 0.5 + k as f32 * 0.01 } else { 0.0 })
        })
        .collect()
}

/// Every pixel drifts a little per frame (nothing is SKIP; the residual is small and smooth).
fn drift(w: u32, h: u32, n: u32) -> Vec<Image> {
    (0..n).map(|k| frame(w, h, move |x, _| k as f32 * 0.002 * (x % 7) as f32)).collect()
}

/// `(intra, skip, pred)` tile counts over the non-I frames of a clip.
fn mix(bytes: &[u8], idx: &VideoIndex) -> (usize, usize, usize) {
    let (mut i, mut s, mut p) = (0, 0, 0);
    for di in 0..idx.frame_count() {
        let v = frame_view(bytes, idx, di).expect("view");
        if v.kind == exv_video::FrameKind::Intra {
            continue;
        }
        let rd = exv_codec::frame::FrameReader::open(v.body, idx.header.clone()).expect("reader");
        for t in 0..rd.tile_count() {
            match rd.tile_kind(t) {
                Some(TILE_INTRA) => i += 1,
                Some(TILE_SKIP) => s += 1,
                Some(TILE_PRED) => p += 1,
                _ => {}
            }
        }
    }
    (i, s, p)
}

fn best<F: FnMut() -> f64>(runs: usize, mut f: F) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// CPU: stateful playback decoder (each frame decoded once) and stateless per-frame `decode_frame`.
fn cpu_times(bytes: &[u8], idx: &VideoIndex, n: usize) -> (f64, f64) {
    let seq = best(2, || {
        let t0 = Instant::now();
        let mut d = Decoder::open(bytes).expect("decoder");
        for k in 0..n {
            std::hint::black_box(d.frame(k).expect("frame"));
        }
        t0.elapsed().as_secs_f64()
    });
    let stateless = best(2, || {
        let t0 = Instant::now();
        for k in 0..n {
            std::hint::black_box(decode_frame(bytes, idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("frame"));
        }
        t0.elapsed().as_secs_f64()
    });
    (seq, stateless)
}

/// GPU-resident: request every tile of every frame in playback order, fence after each frame. Returns
/// `(first frame s, all frames s, decodes, reused)`; a fresh atlas per run so the cold first frame is included.
fn gpu_time(ctx: &GpuContext, bytes: &[u8], idx: &VideoIndex, n: usize, nx: u32, ny: u32) -> Option<(f64, f64, u64, u64)> {
    let tiles: Vec<(u32, u32)> = (0..ny).flat_map(|y| (0..nx).map(move |x| (x, y))).collect();
    let mut out = None;
    let mut all = f64::INFINITY;
    for _ in 0..3 {
        let mut atlas = ExvResidentAtlas::new(ctx, 3 << 30, DECODE_BUDGET).expect("atlas");
        let t0 = Instant::now();
        let mut first = 0.0;
        for k in 0..n {
            if let Err(e) = atlas.request(Clip { data: bytes, idx }, k, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)) {
                println!("    GPU: frame {k}: {e}");
                return None;
            }
            ctx.drain().expect("drain");
            if k == 0 {
                first = t0.elapsed().as_secs_f64();
            }
        }
        let t = t0.elapsed().as_secs_f64();
        if t < all {
            all = t;
            out = Some((first, t, atlas.decode_count(), atlas.stats().reused));
        }
    }
    out
}

fn report(ctx: &GpuContext, name: &str, frames: &[Image], prof: &Profile, gop: u32) {
    let n = frames.len();
    let raw = frames[0].header.frame_bytes().expect("frame bytes") as f64;
    let bytes = write_video(frames, prof, gop, 0, 0).expect("encode");
    let idx = VideoIndex::parse(&bytes).expect("idx");
    let class = temporal_class(&bytes, &idx, TemporalClass::Bidir).expect("class");
    let (i, s, p) = mix(&bytes, &idx);
    let tot = (i + s + p).max(1) as f64;
    let (nx, ny) = (idx.header.tiles_x(), idx.header.tiles_y());
    println!(
        "\n{name} gop={gop}: {n} frames {}x{} | {:.1} MiB ({:.1}x vs raw) | class {class:?} | non-I tiles: INTRA {:.0}% SKIP {:.0}% PRED {:.0}%",
        idx.header.width,
        idx.header.height,
        bytes.len() as f64 / 1048576.0,
        raw * n as f64 / bytes.len() as f64,
        100.0 * i as f64 / tot,
        100.0 * s as f64 / tot,
        100.0 * p as f64 / tot
    );
    let (seq, stateless) = cpu_times(&bytes, &idx, n);
    let mb = raw * n as f64 / 1e6;
    println!("    CPU Decoder (each frame once): {:>8.1} ms/frame {:>8.1} MB/s", seq * 1e3 / n as f64, mb / seq);
    println!("    CPU decode_frame(k) per frame: {:>8.1} ms/frame {:>8.1} MB/s", stateless * 1e3 / n as f64, mb / stateless);
    if class <= TemporalClass::SkipOnly {
        match gpu_time(ctx, &bytes, &idx, n, nx, ny) {
            Some((first, t, dec, reused)) => println!(
                "    GPU resident: {:>8.1} ms/frame {:>8.1} MB/s  (cold frame0 {:.1} ms, {dec} tile decodes, {reused} tiles reused)",
                t * 1e3 / n as f64,
                mb / t,
                first * 1e3
            ),
            None => println!("    GPU resident: refused (see above)"),
        }
    } else {
        println!("    GPU resident: class {class:?} is CPU-only (Tier B would add the residual pass)");
    }
}

#[test]
#[ignore = "benchmark; run explicitly (--release --nocapture)"]
fn temporal_throughput() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) - skipping");
            return;
        }
    };
    println!("device {}", ctx.adapter.get_info().name);
    let (w, h, n) = (env_u32("EXV_BENCH_W", 4096), env_u32("EXV_BENCH_H", 2048), env_u32("EXV_BENCH_FRAMES", 16));
    let rt = Profile::exv_rt_dwt(16, 0);
    let htj = Profile::htj2k();

    // SKIP-dominated static plate with a small moving block (~7% of tiles change per frame) and a bigger one (~30%).
    let small = moving_block(w, h, n, 512, 512);
    let big = moving_block(w, h, n, 1536, 1024);
    report(&ctx, "SKIP-dominated  EXV-RT   (small block)", &small, &rt, 8);
    report(&ctx, "SKIP-dominated  EXV-RT   (big block)", &big, &rt, 8);
    report(&ctx, "SKIP-dominated  EXV-RT   gop=1 twin", &small, &rt, 1);
    // P-dominated, lossless HTJ2K default: the gop=8 file vs its all-INTRA twin.
    let d = drift(w, h, n);
    report(&ctx, "P-dominated     HTJ2K     ", &d, &htj, 8);
    report(&ctx, "P-dominated     HTJ2K     gop=1 twin (Tier B upper bound)", &d, &htj, 1);
    report(&ctx, "SKIP-dominated  HTJ2K lossless (small block)", &small, &htj, 8);
}
