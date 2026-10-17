//! Resident-atlas cold request cost by channel set: routed `R,G,B` vs every channel of an `R,G,B:F16 + Z:F32 + id:U32`
//! frame (the audit item 5 case: a full miss used to decode + upload all 5 planes although compose shows 3).
//!
//! Prints, per codec route: ms per cold request of every tile (fresh atlas, fenced, min of N), host bytes written into
//! upload staging and resident VRAM bytes (`AtlasStats`). Correctness is pinned by the in-crate gate
//! `atlas_channel_subset_decodes_and_merges_only_the_delta`; this file only measures.
//!
//! Run: `cargo test --release -p exv-gpu --test atlas_chans_bench -- --ignored --nocapture --test-threads=1`
//! (`EXV_BENCH_W` / `EXV_BENCH_H` override 4096 x 2048).

use std::time::Instant;

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::half::f32_to_half;
use exv_format::{Channel, Dtype, Header};
use exv_gpu::{AtlasStats, Chans, Clip, ContentKey, DECODE_BUDGET, DisplayKey, ExvResidentAtlas, GpuContext, Want};
use exv_video::{VideoIndex, write_still};

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Smooth HDR RGB ramp with low-bit grain, a depth ramp and a per-object id (16x16 px blocks).
fn frame(w: u32, h: u32) -> Image {
    let n = (w * h) as usize;
    let px = |i: usize| ((i as u32 % w), (i as u32 / w));
    let grain = |x: u32, y: u32| {
        let v = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA6B);
        (v ^ (v >> 15)) & 0x7
    };
    let mut planes: Vec<Vec<u8>> = (0..3u32)
        .map(|c| {
            (0..n)
                .flat_map(|i| {
                    let (x, y) = px(i);
                    let v = 2f32.powf((x as f32 / w as f32 + y as f32 / h as f32) * 3.0 - 3.0) * (1.0 - 0.2 * c as f32);
                    (f32_to_half(v) ^ grain(x, y + c) as u16).to_le_bytes()
                })
                .collect()
        })
        .collect();
    planes.push((0..n).flat_map(|i| (1.0 + px(i).0 as f32 * 0.01 + px(i).1 as f32 * 0.002).to_le_bytes()).collect());
    planes.push((0..n).flat_map(|i| ((px(i).1 / 16) * (w / 16) + px(i).0 / 16).to_le_bytes()).collect());
    let chans = [("R", Dtype::F16), ("G", Dtype::F16), ("B", Dtype::F16), ("Z", Dtype::F32), ("id", Dtype::U32)];
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: chans.iter().map(|&(name, dtype)| Channel { name: name.into(), dtype }).collect(),
    };
    Image::new(header, planes).expect("image")
}

/// Cold request of every tile with `chans` on a fresh atlas, fenced; min over `runs`. `None` = the route refused.
fn cold(ctx: &GpuContext, bytes: &[u8], idx: &VideoIndex, chans: Chans<'_>, runs: usize) -> Option<(f64, AtlasStats)> {
    let tiles: Vec<(u32, u32)> =
        (0..idx.header.tiles_y()).flat_map(|y| (0..idx.header.tiles_x()).map(move |x| (x, y))).collect();
    let mut best: Option<(f64, AtlasStats)> = None;
    for _ in 0..runs {
        let mut atlas = ExvResidentAtlas::new(ctx, 3 << 30, DECODE_BUDGET).expect("atlas");
        let t0 = Instant::now();
        if let Err(e) = atlas.request(Clip { data: bytes, idx }, 0, Want { tiles: &tiles, level: None, chans }, ContentKey(1), DisplayKey(0)) {
            println!("    refused: {e}");
            return None;
        }
        ctx.drain().expect("drain");
        let t = t0.elapsed().as_secs_f64();
        if best.as_ref().is_none_or(|(b, _)| t < *b) {
            best = Some((t, atlas.stats()));
        }
    }
    best
}

#[test]
#[ignore = "benchmark; run explicitly (--release --nocapture)"]
fn atlas_channel_subset_cost() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) - skipping");
            return;
        }
    };
    let (w, h) = (env_u32("EXV_BENCH_W", 4096), env_u32("EXV_BENCH_H", 2048));
    println!("device {} | {w}x{h} R,G,B:F16 + Z:F32 + id:U32, tile 256", ctx.adapter.get_info().name);
    let img = frame(w, h);
    const MIB: f64 = (1u64 << 20) as f64;
    for (name, prof) in [("htj2k (CPU+upload)", Profile::htj2k()), ("lossless wave", Profile::lossless_wave()), ("exv-rt q0", Profile::exv_rt_dwt(16, 0))] {
        let bytes = write_still(&img, &prof).expect("encode");
        let idx = VideoIndex::parse(&bytes).expect("idx");
        for (label, chans) in [("RGB  ", Chans::Only(&[0, 1, 2])), ("all 5", Chans::All)] {
            if let Some((t, st)) = cold(&ctx, &bytes, &idx, chans, 5) {
                println!(
                    "  {name:<20} {label}: {:>7.1} ms | {:>4} planes | host->staging {:>7.1} MiB | resident {:>7.1} MiB",
                    t * 1e3,
                    st.planes,
                    st.uploaded_bytes as f64 / MIB,
                    st.resident_bytes as f64 / MIB
                );
            }
        }
    }
}
