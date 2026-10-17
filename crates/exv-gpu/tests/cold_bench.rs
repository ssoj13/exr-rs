//! Cold-path (whole-frame, per-call) GPU decode throughput on synthetic 4k/8k stills.
//!
//! **Why:** the warm kernel benches (`exvrt_bench`) hide host orchestration, which the perf audit measured as
//! 89-99% of a real frame decode. This one times the PUBLIC whole-still entry (`decode_still_gpu`: parse the
//! container, build every buffer, record, submit, fence) exactly as a player/viewer pays it per frame.
//! Correctness (GPU planes == CPU `read_still`) is checked ONCE outside the timing so a broken decode can
//! never post a fast number. Encoded fixtures are cached under the temp dir (the 8k CPU encode is slow).
//!
//! Run: `cargo test --release -p exv-gpu --test cold_bench -- --ignored --nocapture --test-threads=1`.
//! `EXV_BENCH_SIZES=4096,8192` overrides the sizes; `EXV_BENCH_CODECS=rt,wave,fast` the codecs.

use std::path::PathBuf;
use std::time::Instant;

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::half::f32_to_half;
use exv_format::{Channel, Dtype, Header};

/// Smooth HDR ramp (log-spaced luminance) so the codecs engage realistically; deterministic.
fn ramp(w: u32, h: u32, gain: f32) -> Vec<u8> {
    (0..w as usize * h as usize)
        .flat_map(|i| {
            let fx = (i % w as usize) as f32 / w as f32;
            let fy = (i / w as usize) as f32 / h as f32;
            f32_to_half(gain * 2f32.powf(fx * 4.0 + fy * 4.0 - 4.0)).to_le_bytes()
        })
        .collect()
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key).unwrap_or_else(|_| default.to_string()).split(',').map(|s| s.trim().to_string()).collect()
}

/// Cache path of a fixture. The container + header format versions are part of the name, so a format bump
/// re-encodes instead of failing with "bad exv container" on a stale file (the ONE place the name is built).
fn cache_path(name: &str, w: u32) -> PathBuf {
    let (c, h) = (exv_video::VERSION, exv_format::VERSION);
    std::env::temp_dir().join(format!("exv-gpu-bench-{name}-{w}-c{c}h{h}.exv"))
}

/// Encode (or load from the temp cache) a 3xF16 still of `w`x`h` with `prof`.
fn fixture(name: &str, w: u32, prof: &Profile) -> Vec<u8> {
    let path = cache_path(name, w);
    if let Ok(b) = std::fs::read(&path) {
        return b;
    }
    let header = Header {
        width: w,
        height: w,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: prof.codec_id().0,
        codec_params: prof.params(),
        channels: ["R", "G", "B"].iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F16 }).collect(),
    };
    let planes = vec![ramp(w, w, 1.0), ramp(w, w, 0.7), ramp(w, w, 0.4)];
    let img = Image::new(header, planes).expect("image");
    let bytes = exv_video::write_still(&img, prof).expect("encode");
    std::fs::write(&path, &bytes).expect("cache fixture");
    bytes
}

#[test]
#[ignore = "benchmark; run explicitly"]
fn gpu_cold_throughput() {
    let ctx = match exv_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) - skipping");
            return;
        }
    };
    let lim = ctx.device.limits();
    println!(
        "device {} | max_storage_binding {} MiB | max_buffer {} MiB",
        ctx.adapter.get_info().name,
        lim.max_storage_buffer_binding_size >> 20,
        lim.max_buffer_size >> 20
    );
    let sizes: Vec<u32> = env_list("EXV_BENCH_SIZES", "4096,8192").iter().filter_map(|s| s.parse().ok()).collect();
    let codecs = env_list("EXV_BENCH_CODECS", "rt,wave,fast");
    println!("{:>6} {:>6} {:>10} {:>10} {:>9}  note", "codec", "size", "ms(min)", "host ms", "GB/s");
    for w in sizes {
        for c in &codecs {
            let prof = match c.as_str() {
                "rt" => Profile::exv_rt_dwt(16, 0),
                "wave" => Profile::lossless_wave(),
                "fast" => Profile::lossless_fast(),
                other => panic!("unknown codec {other}"),
            };
            let bytes = fixture(c, w, &prof);
            if std::env::var_os("EXV_BENCH_ONESHOT").is_some() {
                oneshot(&ctx, c, w, &bytes);
                continue;
            }
            // Correctness outside timing; a clean Err (limit/unsupported) is reported, not hidden.
            let first = exv_gpu::decode_still_gpu(&ctx, &bytes, &exv_format::DecodeLimits::default());
            let img = match first {
                Ok(i) => i,
                Err(e) => {
                    println!("{c:>6} {w:>6} {:>10} {:>10} {:>9}  ERR {e}", "-", "-", "-");
                    continue;
                }
            };
            ctx.drain().expect("drain");
            let gpu = exv_gpu::read_back_image(&ctx, &img).expect("readback");
            drop(img);
            let cpu = exv_video::read_still(&bytes).expect("cpu read");
            assert_eq!(gpu, cpu.planes, "{c} {w}: GPU != CPU read_still");
            let (mut tot, mut host) = (Vec::new(), Vec::new());
            for _ in 0..8 {
                let t0 = Instant::now();
                let g = exv_gpu::decode_still_gpu(&ctx, &bytes, &exv_format::DecodeLimits::default()).expect("decode");
                let th = t0.elapsed().as_secs_f64();
                ctx.drain().expect("drain");
                tot.push(t0.elapsed().as_secs_f64());
                host.push(th);
                drop(g);
            }
            tot.sort_by(f64::total_cmp);
            host.sort_by(f64::total_cmp);
            // Min, not median: other processes only ever ADD time (shared GPU/CPU), so the best run is the stable one.
            let (t, h) = (tot[0], host[0]);
            let out = w as f64 * w as f64 * 2.0 * 3.0;
            println!("{c:>6} {w:>6} {:>10.1} {:>10.1} {:>9.2}", t * 1e3, h * 1e3, out / 1e9 / t);
        }
    }
}

/// `EXV_BENCH_ONESHOT=1`: the FIRST decode of a fresh process (no correctness pre-decode, so no warm-up), then 20
/// more frames. Prints first-frame ms and, per 20-frame run, the ms range. Run ONE codec/size per process
/// (`EXV_BENCH_SIZES`/`EXV_BENCH_CODECS`) and read the process's peak working set externally.
fn oneshot(ctx: &exv_gpu::GpuContext, c: &str, w: u32, bytes: &[u8]) {
    let lim = exv_format::DecodeLimits::default();
    let mut ms = Vec::new();
    let mut after_first = (0, 0);
    for i in 0..21 {
        let t0 = Instant::now();
        let g = exv_gpu::decode_still_gpu(ctx, bytes, &lim).expect("decode");
        ctx.drain().expect("drain");
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        drop(g);
        if i == 0 {
            after_first = ctx.staging_stats();
        }
    }
    let (held, fb) = ctx.staging_stats();
    println!("POOL after frame 1: {} MiB held, {} fallbacks | after 21 frames: {} MiB held, {fb} fallbacks", after_first.0 >> 20, after_first.1, held >> 20);
    let rest = &ms[1..];
    println!(
        "ONESHOT {c} {w}: first {:.1} ms | frames 2-21 min {:.1} max {:.1} ms",
        ms[0],
        rest.iter().cloned().fold(f64::MAX, f64::min),
        rest.iter().cloned().fold(0.0, f64::max)
    );
}
