//! Seek-storm probe of the async resident atlas (P1-24 d): what a UI thread pays for `plan` while the decode worker is
//! busy and the view keeps changing (scrubbing a huge clip), and how fast the worker reacts to a cancel.
//!
//! Builds a big all-INTRA EXV-RT clip (`EXV_STORM_W` x `EXV_STORM_H`, default 8192 x 4096, 16384 x 8192 with `EXV_BENCH_BIG=1`, RGB F16, tile 256 = 512 (2048)
//! tiles per frame, 2 frames) and prints, on the 'UI thread' (this one):
//!   * the cold `plan` (pass 1 over every tile) that starts the first job,
//!   * `EXV_STORM_N` (default 300) plans of a SLIDING view of the other frame while the job runs (every signature
//!     differs: the worst case, no plan-signature reuse): median / p99 / max per `plan`,
//!   * the same number of plans of ONE repeated signature (the per-redraw case: the signature cache),
//!   * the worker's latency to finish its job, and to stop after `cancel_all` (the dispatch checkpoints).
//!
//! Run (needs ~4 GiB RAM, a GPU): `cargo test --release -p exv-gpu --test seek_storm_bench -- --ignored --nocapture
//! --test-threads=1`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::half::f32_to_half;
use exv_format::{Channel, Dtype, Header};
use exv_gpu::{Bytes, Chans, Clip, ContentKey, DECODE_BUDGET, DisplayKey, Done, ExvResidentAtlas, GpuContext, Want};
use exv_video::{VideoIndex, write_video};
use rayon::prelude::*;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// One all-F16 RGB frame: a smooth ramp with low-bit grain keyed by `seed` (frames differ everywhere, so every tile of
/// every frame is its own INTRA node).
fn frame(w: u32, h: u32, seed: u32) -> Image {
    let plane = |c: u32| -> Vec<u8> {
        let mut p = vec![0u8; w as usize * h as usize * 2];
        p.par_chunks_mut(w as usize * 2).enumerate().for_each(|(y, row)| {
            for (x, px) in row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let v = 2f32.powf((x as f32 / w as f32 + y as f32 / h as f32) * 3.0 - 3.0) * (1.0 - 0.2 * c as f32);
                let g = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA6B) ^ seed.wrapping_mul(0x27D4_EB2F);
                *px = (f32_to_half(v) ^ ((g ^ (g >> 15)) & 0x7) as u16).to_le_bytes();
            }
        });
        p
    };
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 6,
        codec_params: Vec::new(),
        channels: ["R", "G", "B"].iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F16 }).collect(),
    };
    Image::new(header, (0..3).map(plane).collect()).expect("image")
}

/// Every channel of `tiles` at full resolution.
fn want(tiles: &[(u32, u32)]) -> Want<'_> {
    Want { tiles, level: None, chans: Chans::All }
}

/// `(median, p99, max)` of `v` in milliseconds.
fn stats(v: &mut [Duration]) -> (f64, f64, f64) {
    v.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    (ms(v[v.len() / 2]), ms(v[(v.len() * 99 / 100).min(v.len() - 1)]), ms(v[v.len() - 1]))
}

/// Block (bounded) until the worker answers.
fn wait_done(a: &mut ExvResidentAtlas) -> Done {
    let t = Instant::now();
    loop {
        if let Some(d) = a.poll() {
            return d;
        }
        assert!(t.elapsed() < Duration::from_secs(120), "the decode worker never answered");
        std::thread::sleep(Duration::from_micros(200));
    }
}

#[test]
#[ignore = "benchmark; needs a GPU and ~4 GiB RAM (--release --nocapture)"]
fn seek_storm_plan_latency() {
    let Ok(ctx) = GpuContext::new() else {
        eprintln!("no GPU - skipping");
        return;
    };
    let big = std::env::var_os("EXV_BENCH_BIG").is_some();
    let (dw, dh) = if big { (16384, 8192) } else { (8192, 4096) };
    let (w, h, n) = (env_u32("EXV_STORM_W", dw), env_u32("EXV_STORM_H", dh), env_u32("EXV_STORM_N", 300) as usize);
    let start = Instant::now(); // hard budget: every phase below asserts it, so the probe can never hang for tens of minutes
    let t = Instant::now();
    let frames = [frame(w, h, 1), frame(w, h, 2)];
    let bytes = write_video(&frames, &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode");
    drop(frames);
    assert!(start.elapsed() < Duration::from_secs(300), "building + encoding the clip is over the 5 min budget: shrink EXV_STORM_W/H");
    println!("device {} | {w}x{h} RGB F16 EXV-RT x2 frames: built + encoded in {:.1} s, {:.0} MiB", ctx.adapter.get_info().name, t.elapsed().as_secs_f64(), bytes.len() as f64 / 1048576.0);
    let idx = Arc::new(VideoIndex::parse(&bytes).expect("idx"));
    let data: Bytes = Arc::new(bytes);
    let tiles: Vec<(u32, u32)> = (0..idx.header.tiles_y()).flat_map(|y| (0..idx.header.tiles_x()).map(move |x| (x, y))).collect();
    let nt = tiles.len();
    let clip_bytes: &[u8] = (*data).as_ref();
    let clip = Clip { data: clip_bytes, idx: &idx };

    let mut atlas = ExvResidentAtlas::new(&ctx, 3 << 30, DECODE_BUDGET).expect("atlas");

    // Storm A0 (EXV_STORM_IDLE=1): the same sliding plans with NO job running (the plan is dropped, nothing starts): separates
    // the plan's own host cost from preemption by the busy worker's rayon batch.
    if std::env::var_os("EXV_STORM_IDLE").is_some() {
        let win = nt * 3 / 4;
        let mut a0 = Vec::with_capacity(n);
        let mut idle = ExvResidentAtlas::new(&ctx, 3 << 30, DECODE_BUDGET).expect("atlas");
        for i in 0..n {
            let at = (i * 7) % (nt - win);
            let t = Instant::now();
            let r = idle.plan(clip, 1, want(&tiles[at..at + win]), ContentKey(1), DisplayKey(0)).expect("plan");
            a0.push(t.elapsed());
            drop(r);
        }
        let (m, p, x) = stats(&mut a0);
        println!("storm A0 (idle worker, plans dropped, {win} tiles): median {m:.3} ms | p99 {p:.3} ms | max {x:.3} ms");
    }

    // Cold plan: pass 1 over every tile + the job submit.
    let t = Instant::now();
    let plan = atlas.plan(clip, 0, want(&tiles), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
    let cold = t.elapsed();
    let planned = plan.tiles();
    atlas.submit(plan, Arc::clone(&data), Arc::clone(&idx)).expect("submit");
    let t_submit = Instant::now();
    println!("{nt} tiles: cold plan {:.2} ms ({planned} tiles in the first job)", cold.as_secs_f64() * 1e3);

    // Storm A: a sliding view of frame 1 (every plan has a new signature: full pass 1 + chain walk each time).
    let win = nt * 3 / 4;
    let mut a = Vec::with_capacity(n);
    for i in 0..n {
        assert!(start.elapsed() < Duration::from_secs(600), "storm A is over the 10 min budget: lower EXV_STORM_N");
        let at = (i * 7) % (nt - win);
        let t = Instant::now();
        let r = atlas.plan(clip, 1, want(&tiles[at..at + win]), ContentKey(1), DisplayKey(0)).expect("plan");
        a.push(t.elapsed());
        assert!(r.is_none(), "the worker is busy: plan starts nothing");
    }
    // Which plans are slow: the first one opens the frame (container + ancestors), the rest only walk chains.
    let at_max = a.iter().enumerate().max_by_key(|&(_, d)| *d).map(|(i, _)| i).unwrap_or(0);
    println!("storm A: first plan {:.3} ms | 2nd {:.3} ms | max at plan #{at_max}", a[0].as_secs_f64() * 1e3, a.get(1).map_or(0.0, |d| d.as_secs_f64() * 1e3));
    let (m, p, x) = stats(&mut a);
    println!("storm A (new signature each plan, {win} tiles): median {m:.3} ms | p99 {p:.3} ms | max {x:.3} ms  ({n} plans, job still running: {})", atlas.busy());

    // Storm B: ONE signature repeated (every redraw while decoding).
    let mut b = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let r = atlas.plan(clip, 1, want(&tiles), ContentKey(1), DisplayKey(0)).expect("plan");
        b.push(t.elapsed());
        assert!(r.is_none());
    }
    let (m, p, x) = stats(&mut b);
    println!("storm B (same signature, {nt} tiles):          median {m:.3} ms | p99 {p:.3} ms | max {x:.3} ms");

    let done = wait_done(&mut atlas);
    println!("worker: first job ({planned} tiles) ready {:.1} ms after submit (cancelled: {})", t_submit.elapsed().as_secs_f64() * 1e3, done.is_cancelled());
    let landed = atlas.install(done).expect("install");
    println!("install landed {landed} tiles | {}", atlas.stats());

    // Cancel latency: start the next job, cancel it at once, time until the worker answers.
    let plan = atlas.plan(clip, 1, want(&tiles), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
    atlas.submit(plan, Arc::clone(&data), Arc::clone(&idx)).expect("submit");
    let t = Instant::now();
    atlas.cancel_all();
    let done = wait_done(&mut atlas);
    println!("cancel_all -> worker idle in {:.1} ms (cancelled result: {})", t.elapsed().as_secs_f64() * 1e3, done.is_cancelled());
    assert_eq!(atlas.install(done).expect("install"), 0, "nothing lands after cancel_all");
}
