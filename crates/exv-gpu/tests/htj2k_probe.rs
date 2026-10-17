//! Probe behind the HTJ2K routing rule (`ExvResidentAtlas::htj2k_on_gpu`, docs/OPT.md I.2): per-tile jph-wgpu decode (1 and 4 contexts) vs the CPU `decode_tile` over all tiles in parallel, 4096x2048 RGB F16 (`EXV_PROBE_TILE` = 256/512/1024). Run: `cargo test --release -p exv-gpu --test htj2k_probe -- --ignored --nocapture --test-threads=1`.
use std::time::Instant;

use exv_codec::image::Image;
use exv_codec::{BlobKind, Profile, classify_blob};
use exv_format::half::f32_to_half;
use exv_format::{Channel, Dtype, Header};
use exv_gpu::{DECODE_BUDGET, GpuContext, jph_wgpu_rs};
use exv_video::write_video;

fn frame(w: u32, h: u32, tile: u32) -> Image {
    let planes: Vec<Vec<u8>> = (0..3u32)
        .map(|c| {
            (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let hsh = x.wrapping_mul(0x9E37_79B1) ^ (y + c).wrapping_mul(0x85EB_CA6B);
                    let ramp = 2f32.powf((x as f32 / w as f32 + y as f32 / h as f32) * 3.0 - 3.0) * (1.0 - 0.2 * c as f32);
                    (f32_to_half(ramp) ^ ((hsh ^ (hsh >> 15)) & 7) as u16).to_le_bytes()
                })
                .collect::<Vec<u8>>()
        })
        .collect();
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: tile,
        tile_h: tile,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: ["R", "G", "B"].iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F16 }).collect(),
    };
    Image::new(header, planes).expect("image")
}

#[test]
#[ignore = "benchmark; run explicitly (--release --nocapture)"]
fn htj2k_tile_probe() {
    let ctx = GpuContext::new().expect("gpu");
    let tile: u32 = std::env::var("EXV_PROBE_TILE").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let prof = Profile::htj2k();
    let bytes = write_video(&[frame(4096, 2048, tile)], &prof, 1, 0, 0).expect("enc");
    let (header, body) = exv_video::still_frame_body(&bytes).expect("body");
    let rd = exv_codec::frame::FrameReader::open(body, header).expect("rd");
    let pay: Vec<&[u8]> = (0..rd.tile_count())
        .map(|t| match classify_blob(rd.tile_blob(t).expect("blob")) {
            Some(BlobKind::Htj2k(p)) => p,
            _ => panic!("tag"),
        })
        .collect();
    {
        use rayon::prelude::*;
        for _ in 0..3 {
            let t0 = Instant::now();
            let n: usize = (0..rd.tile_count()).into_par_iter().map(|t| rd.decode_tile(t, &prof, None).expect("cpu").len()).sum();
            println!("tile {tile}: CPU rayon all tiles {:.1} ms ({n} planes)", t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let gpu = jph_wgpu_rs::Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
    for _ in 0..3 {
        let mut rc = jph_wgpu_rs::ResidentCtx::new(&gpu).expect("rc");
        let t0 = Instant::now();
        let keep: Vec<_> = pay.iter().map(|p| rc.decode_to_texture_fresh(&gpu, p, DECODE_BUDGET).expect("dec")).collect();
        let t_sub = t0.elapsed().as_secs_f64();
        ctx.drain().expect("drain");
        println!("serial: submit {:.1} ms, +drain {:.1} ms ({} tiles)", t_sub * 1e3, t0.elapsed().as_secs_f64() * 1e3, keep.len());
    }
    for nt in [4usize] {
        let mut ctxs: Vec<_> = (0..nt).map(|_| jph_wgpu_rs::ResidentCtx::new(&gpu).expect("rc")).collect();
        for _ in 0..3 {
            let t0 = Instant::now();
            let keep: Vec<Vec<_>> = std::thread::scope(|s| {
                let hs: Vec<_> = ctxs
                    .iter_mut()
                    .enumerate()
                    .map(|(k, rc)| {
                        let (gpu, pay) = (&gpu, &pay);
                        s.spawn(move || {
                            pay.iter().skip(k).step_by(nt).map(|p| rc.decode_to_texture_fresh(gpu, p, DECODE_BUDGET).expect("dec")).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("join")).collect()
            });
            let t_sub = t0.elapsed().as_secs_f64();
            ctx.drain().expect("drain");
            println!("{nt} ctx: submit {:.1} ms, +drain {:.1} ms ({} tiles)", t_sub * 1e3, t0.elapsed().as_secs_f64() * 1e3, keep.iter().map(Vec::len).sum::<usize>());
        }
    }
}
