//! WARM kernel throughput of the batched rANS pass (`Pipelines::rans_b`, the stage-1 kernel of LosslessWave / LosslessFast
//! and of EXV-RT blocks > 16) over a synthetic whole-frame descriptor grid, with the host orchestration out of the timing.
//!
//! **Why:** P1-2 (TODO.md) tunes the kernel's workgroup shape and symbol stores; `exv-gpu`'s `cold_bench` hides the
//! kernel behind host time (89-99 %), so a kernel change needs its own isolated, repeatable number. Every pass of a
//! repetition rewrites the same pooled symbol scratch exactly like a chunk of `RansBatch::run` does.
//! Correctness (GPU symbols == the encoder's input) is asserted once, outside the timing, so a broken kernel can never
//! post a fast number; the unpack from the kernel's symbol layout lives in ONE place, [`unpack`].
//!
//! Run ISOLATED (a shared GPU contends): `cargo test --release -p exv-gpu-kernels --test rans_bench -- --ignored --nocapture --test-threads=1`.
//! `EXV_BENCH_UNITS` (default 768 = 4k x 3ch of 256^2 tiles), `EXV_BENCH_LANES` (default 128), `EXV_BENCH_RUNS` (default 9).

use std::time::Instant;

use exv_gpu_kernels::{DESC_WORDS, GpuContext, bind_bufs, encoder, map_read_blocking, new_buf, pass, push_lane, staging, storage_u32};
use gpu_info::wgpu::BufferUsages;
use rans::{Prepared, RansConfig, prepare};

/// Distinct encoded tiles cycled over the grid (distinct streams: no artificial cache reuse of one tile).
const VARIANTS: usize = 16;
/// Symbols per unit: one 256^2 F16 tile of the wave path (2 bytes per sample).
const N: usize = 256 * 256 * 2;
/// Passes recorded per submit (amortises the submit + fence; the pooled symbol scratch is rewritten each time).
const REPS: usize = 8;

fn env(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Wavelet-detail-like bytes: a zero-heavy, geometrically decaying histogram (seeded; deterministic).
fn symbols(seed: u32) -> Vec<u8> {
    let mut x = 0x2545_f491u32 ^ seed.wrapping_mul(0x9e37_79b9);
    (0..N)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // Sum of two geometric draws (trailing zeros of two disjoint bit fields): a smooth zero-heavy histogram, ~3 bits/sym.
            ((x | 1 << 15).trailing_zeros() + ((x >> 16) | 1 << 15).trailing_zeros()) as u8 * 3
        })
        .collect()
}

/// The decoded symbols of units `0..units` out of the kernel's symbol scratch words/bytes (`sym` = raw readback).
/// The ONE place that knows the kernel's symbol layout (u32 per symbol today).
fn unpack(sym: &[u8], first: usize, count: usize) -> Vec<u8> {
    sym.as_chunks::<4>().0.iter().skip(first * N).take(count * N).map(|w| w[0]).collect()
}

#[test]
#[ignore = "benchmark; run explicitly"]
fn rans_b_warm_kernel_throughput() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) - skipping");
            return;
        }
    };
    let (units, lanes, runs) = (env("EXV_BENCH_UNITS", 768), env("EXV_BENCH_LANES", 128) as u8, env("EXV_BENCH_RUNS", 9));
    let syms: Vec<Vec<u8>> = (0..VARIANTS as u32).map(symbols).collect();
    let preps: Vec<Prepared> = syms.iter().map(|s| prepare(s, RansConfig::new(lanes, 12)).expect("encode")).collect();
    let coded: usize = preps.iter().map(|p| p.lane_streams.iter().map(Vec::len).sum::<usize>()).sum::<usize>() / VARIANTS;
    println!(
        "device {} | {units} units x {N} syms, {lanes} lanes, table_log 12 | {:.2} bits/sym",
        ctx.adapter.get_info().name,
        coded as f64 * 8.0 / N as f64
    );

    // Frame-wide buffers exactly as `RansBatch::run` lays them out: one table per distinct model, concatenated streams,
    // absolute lane spans, one `Desc` row per unit.
    let (mut freq, mut cum, mut slot, mut streams, mut lane_off) = (Vec::new(), Vec::new(), Vec::new(), Vec::<u8>::new(), Vec::new());
    let mut table = Vec::new(); // per variant: (freq_base, cum_base, slot_base)
    for p in &preps {
        table.push([freq.len() as u32, cum.len() as u32, slot.len() as u32]);
        freq.extend(p.freq.iter().map(|&f| f as u32));
        cum.extend(p.cum.iter().map(|&c| c as u32));
        slot.extend(p.slot_to_sym.iter().map(|&s| s as u32));
    }
    let mut descs: Vec<u32> = Vec::with_capacity(units * DESC_WORDS);
    for u in 0..units {
        let (p, t) = (&preps[u % VARIANTS], table[u % VARIANTS]);
        let lane_base = (lane_off.len() / 2) as u32;
        for s in &p.lane_streams {
            push_lane(&mut lane_off, streams.len(), streams.len() + s.len()).expect("lane span");
            streams.extend_from_slice(s);
        }
        let row: [u32; DESC_WORDS] = [
            p.cfg.lanes as u32, p.cfg.table_log as u32, N as u32, lane_base, t[0], t[1], t[2], (u * N) as u32,
            2, (N / 2) as u32, 256, 256, 256, 0, 0, 0,
        ];
        descs.extend_from_slice(&row);
    }
    streams.resize(streams.len().next_multiple_of(4), 0);
    let st = BufferUsages::empty();
    let words: Vec<u32> = streams.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let (d, q) = (&ctx.device, &ctx.queue);
    let b_desc = storage_u32(d, q, "d", &descs, st);
    let b_freq = storage_u32(d, q, "f", &freq, st);
    let b_cum = storage_u32(d, q, "c", &cum, st);
    let b_slot = storage_u32(d, q, "s", &slot, st);
    let b_streams = storage_u32(d, q, "st", &words, st);
    let b_off = storage_u32(d, q, "o", &lane_off, st);
    let sym_bytes = (units * N * 4) as u64;
    ctx.check_buf("bench.sym", sym_bytes).expect("sym scratch fits the device");
    let b_sym = new_buf(d, "sym", sym_bytes, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
    let pl = &ctx.pipelines;
    let bind = bind_bufs(d, &pl.rans_b, "b.rans", &[&b_desc, &b_freq, &b_cum, &b_slot, &b_streams, &b_off, &b_sym]);
    let grid = ctx.split_1d("rans units", units as u64).expect("grid");

    // Correctness first (outside the timing): the first VARIANTS-many units must equal the encoder input.
    let check = units.min(VARIANTS);
    let mut enc = encoder(d, "rans.check");
    pass(&ctx, &mut enc, "rans", &pl.rans_b, &bind, grid).expect("pass");
    let rb = staging(d, "rb", (check * N * 4) as u64);
    enc.copy_buffer_to_buffer(&b_sym, 0, &rb, 0, (check * N * 4) as u64);
    q.submit([enc.finish()]);
    let got = unpack(&map_read_blocking(d, &rb).expect("readback"), 0, check);
    // `EXV_BENCH_NOCHECK=1`: timing-only diagnostic of a deliberately incomplete kernel (never for a kept variant).
    for u in 0..if std::env::var_os("EXV_BENCH_NOCHECK").is_some() { 0 } else { check } {
        assert_eq!(got[u * N..(u + 1) * N], syms[u][..], "unit {u}: GPU rANS != encoder input");
    }

    // Warm timing: REPS passes per submit, min of `runs` (other load only ever adds time).
    let mut t = Vec::new();
    for _ in 0..runs {
        let mut enc = encoder(d, "rans.bench");
        for _ in 0..REPS {
            pass(&ctx, &mut enc, "rans", &pl.rans_b, &bind, grid).expect("pass");
        }
        let t0 = Instant::now();
        q.submit([enc.finish()]);
        ctx.drain().expect("drain");
        t.push(t0.elapsed().as_secs_f64() / REPS as f64);
    }
    t.sort_by(f64::total_cmp);
    let out = (units * N) as f64;
    println!(
        "rans_b: min {:.3} ms  med {:.3} ms  -> {:.2} GB/s (min), {:.2} GB/s (med) of decoded bytes",
        t[0] * 1e3,
        t[t.len() / 2] * 1e3,
        out / 1e9 / t[0],
        out / 1e9 / t[t.len() / 2]
    );
}
