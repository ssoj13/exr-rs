//! Atlas plane-texture allocation probe (P1-28, OPT.md Q): where does a cold request's `blit_record` time go?
//! ONE 3-plane request (`W x H` `R32Float`, default 8192 x 4096 = 3 x 134 MB), split into
//!   create = `tex2d` x3 + views on the host (+ `drain`: does the driver allocate lazily?),
//!   first  = the blit pass on those FRESH textures (wgpu's first-use zero-init + the dispatch),
//!   warm   = the same blit again on the now-initialised textures (pure dispatch = what a RECYCLED texture costs).
//! `held` keeps the previous request's textures alive while the next is created (panning before eviction), `freed`
//! drops them first (the free lands in the next submit). Run: `cargo test --release -p exv-gpu --test tex_pool_bench --
//! --ignored --nocapture --test-threads=1` (`EXV_BENCH_W/H/RUNS`).

use std::time::Instant;

use exv_format::Dtype;
use gpu_info::wgpu;
use exv_gpu_kernels::{blit_record, new_buf, tex2d, GpuContext, GpuTile, TileEnc};

fn env(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "GPU bench"]
fn tex_pool_phases() {
    let ctx = GpuContext::new().expect("gpu");
    let (w, h, runs) = (env("EXV_BENCH_W", 8192), env("EXV_BENCH_H", 4096), env("EXV_BENCH_RUNS", 7) as usize);
    let words = (w as usize) * (h as usize);
    let fmt = wgpu::TextureFormat::R32Float;
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let tiles: Vec<GpuTile> = (0..3)
        .map(|_| GpuTile { buf: new_buf(&ctx.device, "bench.src", (words * 4) as u64, usage), off_words: 0, words, w, h, dtype: Dtype::F16, enc: TileEnc::ElemPerWord(2) })
        .collect();
    // Source buffers are first-touched once so their own zero-init is charged to no phase.
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    for t in &tiles {
        enc.clear_buffer(&t.buf, 0, None);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.drain().expect("drain");

    // The blit pass over `texs` (what `blit_record` records beyond the texture creation), submitted and drained.
    let run = |texs: &[wgpu::Texture]| -> f64 {
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let params = exv_gpu_kernels::storage_u32(&ctx.device, &ctx.queue, "bench.params", &[w, h, 0, 1, 2, Dtype::F16.code() as u32], wgpu::BufferUsages::empty());
        let layout = ctx.pipelines.plane_blit_f32.get_bind_group_layout(0);
        for (t, tex) in tiles.iter().zip(texs) {
            let view = tex.create_view(&Default::default());
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bench.bg"),
                layout: &layout,
                entries: &[exv_gpu_kernels::e(0, &t.buf), exv_gpu_kernels::e(1, &params), exv_gpu_kernels::et(2, &view)],
            });
            exv_gpu_kernels::pass(&ctx, &mut enc, "bench.blit", &ctx.pipelines.plane_blit_f32, &bg, [w.div_ceil(exv_gpu_kernels::WG2), h.div_ceil(exv_gpu_kernels::WG2), 1]).expect("pass");
        }
        ctx.queue.submit([enc.finish()]);
        ctx.drain().expect("drain");
        t0.elapsed().as_secs_f64() * 1e3
    };
    // POOLED steady state (the atlas panning with eviction): every request takes its three planes from the pool and
    // gives them back when the tile dies; `blit_record` is exactly the atlas call.
    let pooled = |ctx: &GpuContext| -> f64 {
        let t1 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let out: Vec<_> = tiles.iter().map(|t| blit_record(ctx, &mut enc, t, Dtype::F16, w, h).expect("blit")).collect();
        ctx.queue.submit([enc.finish()]);
        ctx.drain().expect("drain");
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        for (t, _, _) in out {
            ctx.pipelines.recycle_tex(t);
        }
        ms
    };
    let tex_usage =wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
    println!("\n3 x {w}x{h} R32Float ({:.0} MB each), median of {runs}", (words * 4) as f64 / 1e6);
    for held in [false, true] {
        let (mut create, mut drained, mut first, mut warm, mut full) = (vec![], vec![], vec![], vec![], vec![]);
        let mut prev: Vec<wgpu::Texture> = Vec::new();
        for _ in 0..runs {
            if !held {
                prev.clear();
                ctx.drain().expect("drain");
            }
            let t0 = Instant::now();
            let texs: Vec<_> = (0..3).map(|_| tex2d(&ctx.device, "bench.tex", w, h, fmt, tex_usage)).collect();
            create.push(t0.elapsed().as_secs_f64() * 1e3);
            ctx.drain().expect("drain");
            drained.push(t0.elapsed().as_secs_f64() * 1e3);
            first.push(run(&texs));
            warm.push(run(&texs));
            // The full cold request as `blit_record` does it, in ONE timed span (create + clear + dispatch).
            let t1 = Instant::now();
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            let out: Vec<_> = tiles.iter().map(|t| blit_record(&ctx, &mut enc, t, Dtype::F16, w, h).expect("blit")).collect();
            ctx.queue.submit([enc.finish()]);
            ctx.drain().expect("drain");
            full.push(t1.elapsed().as_secs_f64() * 1e3);
            prev = texs;
            prev.extend(out.into_iter().map(|(t, _, _)| t));
        }
        println!(
            "{:>6}: create {:>6.2} | create+drain {:>6.2} | first blit {:>7.2} | warm blit {:>6.2} | blit_record full {:>7.2} (ms)",
            if held { "held" } else { "freed" },
            med(&mut create),
            med(&mut drained),
            med(&mut first),
            med(&mut warm),
            med(&mut full)
        );
    }
    ctx.trim_tex_pool();
    let cold = pooled(&ctx); // pool empty: a fresh allocation, then pooled
    let mut v: Vec<f64> = (0..runs).map(|_| pooled(&ctx)).collect();
    let (n, idle) = ctx.tex_pool_stats();
    println!("pooled: first (cold) {cold:>7.2} ms | steady blit_record full {:>6.2} ms | idle {n} textures {:.0} MB (cap {:.0} MB)", med(&mut v), idle as f64 / 1e6, ctx.tex_pool_cap() as f64 / 1e6);
}
