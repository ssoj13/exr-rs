//! WARM kernel-only timing of the LosslessFast stage 2 (`Pipelines::undelta_b`) and the LosslessWave inverse
//! (`wav_cols_b` / `wav_rows_b`) over a synthetic whole-frame descriptor grid (3 channels of F16, square tiles).
//!
//! **Why:** `cold_bench` hides the kernels behind host time, so a change of the undelta scan or the rows pass needs its
//! own isolated number (OPT.md L). Data is resident and kernel timing is data-independent (zeros are fine); each
//! submit records `REPS` back-to-back passes and the wall time of submit + drain is divided by `REPS`.
//! `EXV_UD_MODE=old` dispatches undelta with the legacy geometry (one thread per (unit, plane): `units*MAX_ESZ/WG`
//! workgroups) so the SAME test binary source times a pre-scan build; default `new` = one workgroup per pair.
//!
//! Run ISOLATED: `cargo test --release -p exv-gpu-kernels --test undelta_bench -- --ignored --nocapture --test-threads=1`.
//! Env: `EXV_UD_SIZES` (default 1024,2048,4096,5120,7936,8192), `EXV_UD_TILE` (default 256), `EXV_UD_RUNS` (default 7).
//! Output: one `UD mode size tile kernel ms` line per kernel (median ms of a whole 3-channel frame).

use std::time::Instant;

use exv_gpu_kernels::{DESC_WORDS, GpuContext, MAX_ESZ, WG, bind_bufs, encoder, new_buf, pass, storage_u32};
use gpu_info::wgpu::BufferUsages;

const CH: u32 = 3;
const ESZ: u32 = 2;
const REPS: usize = 8;

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Median of the per-pass time of `runs` submits of `REPS` passes built by `rec`.
fn med(ctx: &GpuContext, runs: usize, rec: &dyn Fn(&mut gpu_info::wgpu::CommandEncoder)) -> Result<f64, Box<dyn std::error::Error>> {
    let mut v = Vec::new();
    for i in 0..=runs {
        let mut enc = encoder(&ctx.device, "ud.bench");
        for _ in 0..REPS {
            rec(&mut enc);
        }
        let t = Instant::now();
        ctx.queue.submit([enc.finish()]);
        ctx.drain()?;
        if i > 0 {
            v.push(t.elapsed().as_secs_f64() * 1e3 / REPS as f64); // run 0 = warm-up
        }
    }
    v.sort_by(f64::total_cmp);
    Ok(v[v.len() / 2])
}

#[test]
#[ignore = "GPU timing; run isolated"]
fn undelta_rows_kernel_timing() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new()?;
    let old = env("EXV_UD_MODE", "new") == "old";
    let tile: u32 = env("EXV_UD_TILE", "256").parse()?;
    let runs: usize = env("EXV_UD_RUNS", "7").parse()?;
    for s in env("EXV_UD_SIZES", "1024,2048,4096,5120,7936,8192").split(',') {
        let size: u32 = s.parse()?;
        let n = size / tile; // whole tiles only (sizes are multiples of the tile)
        let count = tile * tile;
        // One unit per (channel, tile); `sym_base` = plane-major words, `out_base` = the channel's composite plane.
        let mut descs: Vec<u32> = Vec::new();
        let mut units = 0u32;
        for c in 0..CH {
            for ty in 0..n {
                for tx in 0..n {
                    let mut row = [0u32; DESC_WORDS];
                    row[7] = units * ESZ * count; // sym_base
                    row[8] = ESZ;
                    row[9] = count;
                    row[10] = tile;
                    row[11] = tile;
                    row[12] = size;
                    row[13] = tx * tile;
                    row[14] = ty * tile;
                    row[15] = c * size * size; // out_base (words)
                    descs.extend_from_slice(&row);
                    units += 1;
                }
            }
        }
        let d = storage_u32(&ctx.device, &ctx.queue, "ud.desc", &descs, BufferUsages::empty());
        let syms = new_buf(&ctx.device, "ud.syms", units as u64 * ESZ as u64 * count as u64 * 4, BufferUsages::STORAGE);
        let comp = new_buf(&ctx.device, "ud.comp", CH as u64 * size as u64 * size as u64 * 4, BufferUsages::STORAGE);
        let pl = &ctx.pipelines;
        let bg_un = bind_bufs(&ctx.device, &pl.undelta_b, "ud", &[&d, &syms, &comp]);
        let bg_c = bind_bufs(&ctx.device, &pl.wav_cols_b, "cols", &[&d, &comp]);
        let bg_r = bind_bufs(&ctx.device, &pl.wav_rows_b, "rows", &[&d, &comp]);
        let grid = ctx.split_1d("units", units as u64)?;
        let g_un = if old {
            ctx.split_1d("undelta", (units as u64 * MAX_ESZ as u64).div_ceil(WG as u64))?
        } else {
            ctx.split_1d("undelta", units as u64 * MAX_ESZ as u64)?
        };
        let mode = if old { "old" } else { "new" };
        let t_un = med(&ctx, runs, &|e| pass(&ctx, e, "undelta", &pl.undelta_b, &bg_un, g_un).expect("dispatch"))?;
        let t_c = med(&ctx, runs, &|e| pass(&ctx, e, "cols", &pl.wav_cols_b, &bg_c, grid).expect("dispatch"))?;
        let t_r = med(&ctx, runs, &|e| pass(&ctx, e, "rows", &pl.wav_rows_b, &bg_r, grid).expect("dispatch"))?;
        println!("UD {mode} {size} {tile} undelta {t_un:.3}");
        println!("UD {mode} {size} {tile} cols {t_c:.3}");
        println!("UD {mode} {size} {tile} rows {t_r:.3}");
    }
    Ok(())
}
