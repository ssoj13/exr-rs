//! Host -> VRAM upload FLOOR probe: fresh `write_buffer_with` staging vs a persistent `StagingBelt` (warm pages), both
//! filled by the crate's parallel `fill_parts` exactly like a whole-frame decode (many parts per buffer, several buffers
//! per submit). **Why:** the cold decode is bound by the host fill + first touch of FRESH staging memory (OPT.md H/M);
//! this decides whether a pooled belt in `GpuContext` pays before it is wired anywhere.
//!
//! Run: `cargo test --release -p exv-gpu --test floor_bench -- --ignored --nocapture --test-threads=1`
//! (`EXV_FLOOR_MB=100,400` overrides the frame sizes). Medians of 9; the first (cold) iteration is reported apart.

use std::time::Instant;

use exv_gpu_kernels::{GpuContext, fill_parts, new_buf};
use gpu_info::wgpu;

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One frame = buffers of 60 % / 30 % / 10 % of `n` bytes, each cut into 4 KiB .. 1 MiB-ish parts like units/descs.
fn frame(n: usize) -> Vec<Vec<usize>> {
    [(n * 6 / 10, 48 << 10), (n * 3 / 10, 64), (n / 10, 1 << 10)]
        .iter()
        .map(|&(bytes, part)| {
            let k = (bytes / part).max(1);
            (0..k).map(|i| if i + 1 == k { bytes - part * (k - 1) } else { part }).collect()
        })
        .collect()
}

fn fill(src: &[u8], i: usize, mut d: wgpu::WriteOnly<'_, [u8]>) {
    let l = d.len();
    d.copy_from_slice(&src[i % 64..i % 64 + l]);
}

#[test]
#[ignore = "benchmark; run explicitly"]
fn staging_floor() {
    let Ok(ctx) = GpuContext::new() else { return };
    let (d, q) = (&ctx.device, &ctx.queue);
    let sizes: Vec<usize> = std::env::var("EXV_FLOOR_MB").unwrap_or_else(|_| "100,400".into()).split(',').filter_map(|s| s.parse::<usize>().ok()).collect();
    let src = vec![5u8; 2 << 20];
    for mb in sizes {
        let bufs = frame(mb << 20);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        // A: fresh buffers + write_buffer_with (today's `storage_parts`).
        let (mut a_tot, mut a_host, mut a_cold) = (vec![], vec![], 0.0);
        for it in 0..10 {
            let t = Instant::now();
            let mut keep = vec![];
            for parts in &bufs {
                let size = parts.iter().sum::<usize>() as u64;
                let b = new_buf(d, "a", size, usage);
                let mut v = q.write_buffer_with(&b, 0, wgpu::BufferSize::new(size).unwrap()).unwrap();
                fill_parts(v.slice(..), parts, || (), |_, i, dst| -> Result<(), exv_gpu_kernels::GpuError> {
                    fill(&src, i, dst);
                    Ok(())
                })
                .unwrap();
                drop(v);
                keep.push(b);
            }
            let th = t.elapsed().as_secs_f64();
            q.submit([]);
            ctx.drain().unwrap();
            let tt = t.elapsed().as_secs_f64();
            if it == 0 {
                a_cold = tt;
            } else {
                a_tot.push(tt);
                a_host.push(th);
            }
        }
        // B: persistent belt (chunks recycled across frames), one encoder of copies, one submit.
        let mut belt = wgpu::util::StagingBelt::new(d.clone(), ((mb as u64) << 20) / 2 + (64 << 20));
        let (mut b_tot, mut b_host, mut b_cold) = (vec![], vec![], 0.0);
        for it in 0..10 {
            let t = Instant::now();
            let mut enc = d.create_command_encoder(&Default::default());
            let mut keep = vec![];
            for parts in &bufs {
                let size = parts.iter().sum::<usize>() as u64;
                let b = new_buf(d, "b", size, usage);
                let mut v = belt.write_buffer(&mut enc, &b, 0, wgpu::BufferSize::new(size).unwrap());
                fill_parts(v.slice(..), parts, || (), |_, i, dst| -> Result<(), exv_gpu_kernels::GpuError> {
                    fill(&src, i, dst);
                    Ok(())
                })
                .unwrap();
                drop(v);
                keep.push(b);
            }
            belt.finish_and_recall_on_submit(&enc);
            let th = t.elapsed().as_secs_f64();
            q.submit([enc.finish()]);
            ctx.drain().unwrap();
            let tt = t.elapsed().as_secs_f64();
            if it == 0 {
                b_cold = tt;
            } else {
                b_tot.push(tt);
                b_host.push(th);
            }
        }
        let g = |s: f64| (mb << 20) as f64 / 1e9 / s;
        println!(
            "{mb:>4} MB | fresh write_buffer_with: cold {:.1} ms | median total {:.1} ms ({:.2} GB/s) host {:.1} ms || belt: cold {:.1} ms | median total {:.1} ms ({:.2} GB/s) host {:.1} ms",
            a_cold * 1e3,
            med(a_tot.clone()) * 1e3,
            g(med(a_tot)),
            med(a_host) * 1e3,
            b_cold * 1e3,
            med(b_tot.clone()) * 1e3,
            g(med(b_tot)),
            med(b_host) * 1e3
        );
    }
}
