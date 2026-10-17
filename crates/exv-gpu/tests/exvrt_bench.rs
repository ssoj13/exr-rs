//! B-RT.3 **Increment 4** — an HONEST throughput MEASUREMENT of the *existing* EXV-RT GPU decoder.
//!
//! This is a MEASUREMENT-ONLY bench: it does NOT touch the decode kernel or
//! `decode_tile_exvrt_vram` / `decode_exvrt_into` / `decode_blockrt_batched`. It builds realistic F16
//! payloads (smooth HDR ramps → the block codec engages with real adaptive-width substreams, not
//! all-zero), decodes them on the GPU, and reports two rates per case:
//!
//! - **GB/s** over the decoded OUTPUT bytes (`w·h·2` for F16) — the bandwidth the viewer/consumer sees.
//! - **Gsym/s** over the total rANS byte-symbols the decoder must serially expand
//!   (`Σ_blocks block²·width[bi]`) — the codec-invariant work ceiling, independent of pixel count.
//!
//! Methodology mirrors `bc6h_floor_bench.rs`: a `GpuContext`, W warmup iters, then N timed iters where
//! each iter = one decode call fenced by `ctx.drain()` (`device.poll(Wait)`), with the GPU==CPU
//! correctness gate run ONCE OUTSIDE the timed loop so a broken decode can never post a fast bogus rate.
//!
//! # Honesty caveats (printed at runtime too)
//! - **1 invocation per block.** The inverse kernel (`exvrt_inv`) is declared `@workgroup_size(64)` but
//!   only `lid.x == 0` does work — the per-block `sh_coeff[4096]` scratch is *workgroup* storage, so a
//!   block is a single serial invocation and the other 63 lanes idle. A within-block-parallel /
//!   global-scratch variant is the NEXT optimization (Increment 5). This is why the baseline may sit well
//!   below the ~25–30 GB/s OPTIMIZED target; the number here is the honest *starting point*, not the goal.
//! - **Buffer build/upload is INSIDE the timed region.** The public decode API allocates + uploads all
//!   GPU buffers (descriptors, tables, substreams, output) PER CALL, so host-side buffer construction and
//!   uploads are timed. These are END-TO-END *decode-call* rates, NOT isolated kernel throughput.
//!
//! Run: `cargo test -p exv-gpu --test exvrt_bench -- --ignored --nocapture`.

use std::time::Instant;

/// f32 → f16 LE bit pattern (RNE) — the codec's own conversion (`exv_format::half`), so the ramp we feed the
/// encoder round-trips exactly at q=0.
fn f16_le(v: f32) -> [u8; 2] {
    exv_format::half::f32_to_half(v).to_le_bytes()
}

/// Total rANS byte-symbols the GPU decoder must serially expand for ONE EXV-RT plane payload (the
/// tag-less blob that `encode_plane`/`classify_blob` yield): `Σ_blocks block²·width[bi]`. This is the
/// codec-invariant work ceiling driving the Gsym/s rate — every block is `block²` coeffs, each stored as
/// `width[bi]` little-endian byte-symbols (the adaptive per-block width). Blob layout (LE):
/// `block:u8 | q:u8 | table_log:u8 | _pad:u8 | nbx:u16 | nby:u16 | freq[256×u16] | off[(nblk+1)×u32] |
/// width[nblk×u8] | substreams…`. Returns 0 on a short/malformed blob (our own encodes are well-formed).
fn exvrt_symbols(payload: &[u8]) -> u64 {
    if payload.len() < 8 {
        return 0;
    }
    let block = payload[0] as u64;
    let nbx = u16::from_le_bytes([payload[4], payload[5]]) as usize;
    let nby = u16::from_le_bytes([payload[6], payload[7]]) as usize;
    let nblk = nbx * nby;
    // widths sit right after the 8-byte header, the 512-byte freq table, and the (nblk+1)×u32 offset index.
    let woff = 8 + 512 + (nblk + 1) * 4;
    let widths = match payload.get(woff..woff + nblk) {
        Some(w) => w,
        None => return 0,
    };
    let ncoeff = block * block;
    widths.iter().map(|&w| ncoeff * w as u64).sum()
}

/// Max f16 ULP distance between two equal-length f16 LE planes, or `None` if the lengths differ or any
/// SIGN bit flips (a sign flip is a real decode bug, never `exp2` noise). The correctness gate: q=0 must
/// be 0 (pure integer), q>0 must be ≤1 (the single `exp2` approximation in `hdr::decode_value`).
fn max_ulp(gpu: &[u8], cpu: &[u8]) -> Option<i32> {
    if gpu.len() != cpu.len() || !gpu.len().is_multiple_of(2) {
        return None;
    }
    let mut m = 0i32;
    for (g, c) in gpu.as_chunks::<2>().0.iter().zip(cpu.as_chunks::<2>().0.iter()) {
        let gu = u16::from_le_bytes([g[0], g[1]]);
        let cu = u16::from_le_bytes([c[0], c[1]]);
        if (gu & 0x8000) != (cu & 0x8000) {
            return None;
        }
        m = m.max((gu as i32 - cu as i32).abs());
    }
    Some(m)
}

/// Build a smooth positive HDR ramp (log-space, ~2^-4..2^4) as an f16 LE plane — enough structure that
/// the wavelet+adaptive-width block codec produces real, varied substreams rather than a degenerate
/// all-zero payload that would flatter the decode rate.
fn hdr_ramp(w: u32, h: u32, gain: f32, bias: f32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let fx = (i % w) as f32 / w as f32;
            let fy = (i / w) as f32 / h as f32;
            f16_le(gain * 2f32.powf(fx * 4.0 + fy * 4.0 - 4.0) + bias)
        })
        .collect()
}

#[test]
#[ignore = "benchmark; run explicitly"]
fn exvrt_gpu_throughput() {
    let ctx = match exv_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) — skipping exvrt_gpu_throughput");
            return;
        }
    };
    let info = ctx.adapter.get_info();
    let lim = ctx.device.limits();

    let warm = 3u32;
    let iters = 20u32;

    // --- metadata + honesty caveats (mandatory; see module docs) ---
    println!("=== EXV-RT GPU decoder throughput — B-RT.3 Increment 4 (measurement only) ===");
    println!("device : {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    println!(
        "limits : max_wg/dim={}, max_storage_binding={} MiB, max_wg_storage={} B",
        lim.max_compute_workgroups_per_dimension,
        lim.max_storage_buffer_binding_size >> 20,
        lim.max_compute_workgroup_storage_size
    );
    println!("kernel : rANS pass `rans_b` + inverse `exvrt_inv`, @workgroup_size(64)");
    println!("         CAVEAT current kernel = 1 invocation per block (sh_coeff is workgroup storage);");
    println!("         a within-block-parallel or global-scratch variant is the next optimization (Increment 5).");
    println!("timing : {warm} warmup + {iters} timed iters, each = decode call -> device.poll(Wait); MEAN reported.");
    println!("         CAVEAT the public decode API allocates + uploads all GPU buffers PER CALL, so");
    println!("         buffer build + upload overhead IS inside the timed region — these are END-TO-END");
    println!("         decode-call rates, NOT isolated kernel throughput.");
    println!(
        "         GB/s = decoded F16 output bytes (w*h*2) / time; Gsym/s = SUM(block^2 * width[bi]) / time."
    );

    // ----------------------------------------------------------------------------------------------
    // Single F16 plane via `decode_tile_exvrt_vram` (block chosen so nblk <= max_wg/dim on all HW).
    // ----------------------------------------------------------------------------------------------
    println!("\n-- single F16 plane (decode_tile_exvrt_vram) --");
    println!(
        "{:>11} {:>5} {:>3} {:>10} {:>9} {:>9} {:>7}",
        "dims", "block", "q", "ms/iter", "GB/s", "Gsym/s", "maxULP"
    );
    // (w, h, block, q). 4096^2 uses block=32 so nblk=16384 stays under the 65535 dispatch-per-dim cap
    // (block=16 there would be 65536 blocks and hit the gate → CPU fallback, nothing to measure).
    let plane_cases: &[(u32, u32, u32, u32)] = &[
        (2048, 2048, 16, 0),
        (2048, 2048, 16, 4),
        (4096, 4096, 32, 0),
    ];
    for &(w, h, block, q) in plane_cases {
        let src = hdr_ramp(w, h, 1.0, 0.0);
        let payload =
            exv_codec::blockrt_dwt::encode_plane(&src, w, h, block, q, exv_codec::blockrt_dwt::RT_LANES).expect("encode_plane");
        let sym = exvrt_symbols(&payload);

        // Correctness gate OUTSIDE the timed loop: decode once, compare to the CPU oracle. A device-limit
        // gate (a clean Err) is reported as a skip rather than a failure — the current kernel legitimately
        // cannot take some geometries and the CPU is the strict fallback.
        let tile = match exv_gpu::decode_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h) {
            Ok(t) => t,
            Err(e) => {
                println!("{w:>5}x{h:<5} block={block} q={q}: SKIPPED (device gate): {e}");
                continue;
            }
        };
        ctx.drain().expect("drain correctness");
        let gpu = exv_gpu::read_back(&ctx, &tile).expect("readback");
        drop(tile);
        let cpu = exv_codec::blockrt_dwt::decode_plane(&payload, w, h).expect("cpu decode");
        let ulp = max_ulp(&gpu, &cpu)
            .expect("length/sign mismatch — GPU decode is BROKEN, refusing to time it");
        assert!(ulp <= 1, "{w}x{h} q={q}: max ULP {ulp} > 1 — decode incorrect, not timing a broken result");

        // Warmup (pipeline/driver/allocator), untimed.
        for _ in 0..warm {
            let t = exv_gpu::decode_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h)
                .expect("warm decode");
            ctx.drain().expect("drain warm");
            drop(t);
        }
        // Timed: each iter is one decode call fenced by a poll(Wait). Mean over N.
        let mut total = 0f64;
        for _ in 0..iters {
            let t0 = Instant::now();
            let t = exv_gpu::decode_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h)
                .expect("timed decode");
            ctx.drain().expect("drain timed");
            total += t0.elapsed().as_secs_f64();
            drop(t);
        }
        let secs = total / iters as f64;
        let out_bytes = (w as u64) * (h as u64) * 2;
        let gbps = out_bytes as f64 / 1e9 / secs;
        let gsymps = sym as f64 / 1e9 / secs;
        assert!(gbps.is_finite() && gbps > 0.0, "GB/s must be finite/non-zero");
        assert!(gsymps.is_finite() && gsymps > 0.0, "Gsym/s must be finite/non-zero");
        println!(
            "{:>11} {:>5} {:>3} {:>10.3} {:>9.2} {:>9.3} {:>7}",
            format!("{w}x{h}"),
            block,
            q,
            secs * 1e3,
            gbps,
            gsymps,
            ulp
        );
    }

    // ----------------------------------------------------------------------------------------------
    // Whole multi-tile 1-frame `.exv` still, end-to-end, via `decode_still_gpu` (3 F16 channels, q=0 bit-exact).
    // Includes the CPU tile-walk + per-tile buffer builds/uploads — the true end-to-end decode number.
    // ----------------------------------------------------------------------------------------------
    println!("\n-- whole multi-tile .exv (decode_still_gpu, 3x F16 channels) --");
    {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        use exv_format::{BlobKind, Channel, Dtype, Header, classify_blob};

        let (w, h) = (2048u32, 2048u32);
        let (block, q) = (16u8, 0u8); // q=0 → the whole-image path is strictly bit-exact vs CPU read_still.
        let prof = Profile::exv_rt_dwt(block, q);
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: 256,
            tile_h: 256,
            codec_id: prof.codec_id().0,
            codec_params: prof.params(),
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "G".into(), dtype: Dtype::F16 },
                Channel { name: "B".into(), dtype: Dtype::F16 },
            ],
        };
        let planes_in = vec![
            hdr_ramp(w, h, 1.0, 0.0),
            hdr_ramp(w, h, 0.7, 0.1),
            hdr_ramp(w, h, 0.4, 0.2),
        ];
        let img = Image::new(header, planes_in).expect("image");
        let bytes = write_still(&img, &prof).expect("encode .exv");

        // Total decoded output bytes (all channels are F16 → all go through the GPU block decoder) and
        // total rANS symbols summed over every on-disk BlockRt tile blob (the codec-invariant ceiling).
        let (header, body) = exv_video::still_frame_body(&bytes).expect("exv container");
        let reader = exv_codec::frame::FrameReader::open(body, header).expect("reader");
        let nch = reader.header().channels.len();
        let mut sym = 0u64;
        let mut brt_tiles = 0u64;
        for t in 0..reader.tile_count() {
            for ci in 0..nch {
                if let Some(BlobKind::BlockRt(p)) =
                    reader.tile_channel_blob(t, ci).ok().and_then(classify_blob)
                {
                    sym += exvrt_symbols(p);
                    brt_tiles += 1;
                }
            }
        }
        assert!(brt_tiles > 0, "fixture produced no BlockRt tiles — nothing to measure");

        // Correctness OUTSIDE timing: q=0 is strictly bit-exact vs the CPU decode of the SAME blobs.
        let gimg = exv_gpu::decode_still_gpu(&ctx, &bytes, &exv_format::DecodeLimits::default()).expect("gpu still decode");
        ctx.drain().expect("drain correctness");
        let planes = exv_gpu::read_back_image(&ctx, &gimg).expect("readback");
        drop(gimg);
        let cpu = read_still(&bytes).expect("cpu read");
        assert_eq!(
            planes, cpu.planes,
            "whole-image GPU != CPU read_still (q=0 must be bit-exact) — refusing to time a broken decode"
        );

        for _ in 0..warm {
            let g = exv_gpu::decode_still_gpu(&ctx, &bytes, &exv_format::DecodeLimits::default()).expect("warm still");
            ctx.drain().expect("drain warm");
            drop(g);
        }
        let mut total = 0f64;
        for _ in 0..iters {
            let t0 = Instant::now();
            let g = exv_gpu::decode_still_gpu(&ctx, &bytes, &exv_format::DecodeLimits::default()).expect("timed still");
            ctx.drain().expect("drain timed");
            total += t0.elapsed().as_secs_f64();
            drop(g);
        }
        let secs = total / iters as f64;
        let out_bytes = (w as u64) * (h as u64) * 2 * (nch as u64);
        let gbps = out_bytes as f64 / 1e9 / secs;
        let gsymps = sym as f64 / 1e9 / secs;
        assert!(gbps.is_finite() && gbps > 0.0, "GB/s must be finite/non-zero");
        assert!(gsymps.is_finite() && gsymps > 0.0, "Gsym/s must be finite/non-zero");
        println!(
            "{}x{} block={} q={} : {} F16 ch, {} BlockRt tiles | {:.3} ms/iter | {:.2} GB/s | {:.3} Gsym/s",
            w, h, block, q, nch, brt_tiles, secs * 1e3, gbps, gsymps
        );
        println!("(end-to-end: includes CPU tile-walk + per-tile buffer builds/uploads across all tiles)");
    }
}

/// B-RT.3 **Increment 5, STEP 2** — isolate TRUE kernel throughput with a WARM/RESIDENT decode path.
///
/// The public `decode_tile_exvrt_vram` ALLOCATES + UPLOADS all GPU buffers (streams, descriptors, freq
/// table, output) and re-records the command encoder on EVERY call — inside `exvrt_gpu_throughput`'s
/// timed region. The realtime player instead decodes into RESIDENT reused buffers with no per-call alloc.
/// This bench builds each tile ONCE (`prepare_tile_exvrt_vram`) and times ONLY `redecode`
/// (`dispatch+submit`) + `drain`, giving the warm kernel-only rate. It ALSO times the cold full-call rate
/// for the same case and prints the split (`cold_ms`, `warm_ms`, `overhead_ms = cold - warm`) so we know
/// whether the gap to the 40+ GB/s target is per-call buffer overhead (→ resident pooling wins) or the
/// kernel itself (→ STEP 3 pass fusion). Correctness gate (warm output == CPU `decode_plane`, maxULP≤1)
/// runs ONCE OUTSIDE timing. Run: `cargo test -p exv-gpu --test exvrt_bench exvrt_gpu_warm -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark; run explicitly"]
fn exvrt_gpu_warm_kernel_throughput() {
    let ctx = match exv_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) — skipping exvrt_gpu_warm_kernel_throughput");
            return;
        }
    };
    let info = ctx.adapter.get_info();
    let lim = ctx.device.limits();
    let warm = 3u32;
    let iters = 20u32;

    println!("=== EXV-RT GPU WARM kernel-only throughput — B-RT.3 Increment 5 STEP 2 ===");
    println!("device : {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    println!(
        "limits : max_wg/dim={}, max_storage_binding={} MiB, max_wg_storage={} B",
        lim.max_compute_workgroups_per_dimension,
        lim.max_storage_buffer_binding_size >> 20,
        lim.max_compute_workgroup_storage_size
    );
    println!("method : prepare ONCE (buffers built + uploaded, EXCLUDED from timing); correctness ONCE outside timing;");
    println!("         WARM = {warm} warmup + {iters} timed iters of redecode(dispatch+submit) -> drain(poll Wait), MEAN.");
    println!("         COLD = same {warm}+{iters} over the full decode_tile_exvrt_vram (per-call alloc+upload+record).");
    println!("         GB/s = decoded F16 output bytes (w*h*2) / time; Gsym/s = SUM(block^2 * width[bi]) / time.");
    println!("         overhead_ms = cold_ms - warm_ms  (the per-call buffer build/upload cost).");
    println!(
        "\n{:>11} {:>5} {:>3} {:>3} {:>5} | {:>9} {:>9} {:>9} | {:>9} {:>11} | {:>7} verdict",
        "dims", "block", "q", "N", "fast", "warm_ms", "warm GB/s", "warm Gsym", "cold_ms", "overhd_ms", "maxULP"
    );

    // block=16 for BOTH sizes → BOTH hit the thread-dense fast path. 4096²/block16 = 65536 blocks →
    // ceil/64 = 1024 workgroups (well under the dispatch-per-dim cap now that the gate checks the ACTUAL
    // dispatched workgroup count, not nblk). A clean device gate (Err) is reported as SKIPPED, not a fail.
    let cases: &[(u32, u32, u32, u32)] = &[
        (2048, 2048, 16, 0),
        (2048, 2048, 16, 4),
        (4096, 4096, 16, 0),
        (4096, 4096, 16, 4),
        // Larger sizes probe whether warm GB/s CLIMBS with occupancy (more blocks in flight) or is a flat
        // per-block-latency wall. Skipped cleanly if they exceed max_storage_buffer_binding_size.
        (6144, 6144, 16, 0),
        (8192, 8192, 16, 0),
    ];
    for &(w, h, block, q) in cases {
        let src = hdr_ramp(w, h, 1.0, 0.0);
        // Sweep the rANS interleave width per case: N=1 (serial baseline) vs N∈{2,4,8} (N-lane-per-thread
        // ILP). Same pixels for every N (entropy-only) — only the warm kernel time changes. This is the
        // Increment 5 PATH A deliverable: the warm GB/s knee across lane counts.
        for lanes in [1u8, 2, 4, 8] {
        let payload = exv_codec::blockrt_dwt::encode_plane(&src, w, h, block, q, lanes).expect("encode_plane");
        let sym = exvrt_symbols(&payload);
        let out_bytes = (w as u64) * (h as u64) * 2;

        // --- prepare ONCE (buffers built + uploaded here, OUTSIDE all timing) ---
        let tile = match exv_gpu::prepare_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h) {
            Ok(t) => t,
            Err(e) => {
                println!("{w:>5}x{h:<5} block={block} q={q} N={lanes}: SKIPPED (device gate): {e}");
                continue;
            }
        };
        // --- correctness gate ONCE, OUTSIDE timing: warm output == CPU oracle (maxULP <= 1) ---
        tile.redecode(&ctx).expect("redecode");
        ctx.drain().expect("drain correctness");
        let gpu = exv_gpu::read_back(&ctx, &tile.tile()).expect("readback");
        let cpu = exv_codec::blockrt_dwt::decode_plane(&payload, w, h).expect("cpu decode");
        let ulp = max_ulp(&gpu, &cpu)
            .expect("length/sign mismatch — WARM GPU decode is BROKEN, refusing to time it");
        assert!(ulp <= 1, "{w}x{h} q={q}: warm max ULP {ulp} > 1 — decode incorrect, not timing a broken result");

        // --- WARM: time ONLY redecode(dispatch+submit) + drain, reusing the SAME resident buffers ---
        for _ in 0..warm {
            tile.redecode(&ctx).expect("redecode");
            ctx.drain().expect("drain warm");
        }
        let mut warm_total = 0f64;
        for _ in 0..iters {
            let t0 = Instant::now();
            tile.redecode(&ctx).expect("redecode");
            ctx.drain().expect("drain warm timed");
            warm_total += t0.elapsed().as_secs_f64();
        }
        let warm_secs = warm_total / iters as f64;

        // --- COLD: the full public per-call decode (alloc + upload + record + submit) for the same case ---
        for _ in 0..warm {
            let t = exv_gpu::decode_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h)
                .expect("warm cold-decode");
            ctx.drain().expect("drain cold warm");
            drop(t);
        }
        let mut cold_total = 0f64;
        for _ in 0..iters {
            let t0 = Instant::now();
            let t = exv_gpu::decode_tile_exvrt_vram(&ctx, &payload, exv_format::Dtype::F16, w, h)
                .expect("timed cold-decode");
            ctx.drain().expect("drain cold timed");
            cold_total += t0.elapsed().as_secs_f64();
            drop(t);
        }
        let cold_secs = cold_total / iters as f64;

        let warm_gbps = out_bytes as f64 / 1e9 / warm_secs;
        let warm_gsymps = sym as f64 / 1e9 / warm_secs;
        let overhead_ms = (cold_secs - warm_secs) * 1e3;
        // Verdict: if per-call overhead dominates (warm << cold), buffer pooling in the player is the win;
        // if warm ~= cold (overhead negligible), the KERNEL itself is the limiter → STEP 3 pass fusion.
        let overhead_frac = if cold_secs > 0.0 { (cold_secs - warm_secs) / cold_secs } else { 0.0 };
        let verdict = if overhead_frac > 0.5 { "OVERHEAD-bound" } else { "KERNEL-bound" };
        assert!(warm_gbps.is_finite() && warm_gbps > 0.0, "warm GB/s must be finite/non-zero");

        println!(
            "{:>11} {:>5} {:>3} {:>3} {:>5} | {:>9.3} {:>9.2} {:>9.3} | {:>9.3} {:>11.3} | {:>7} {} ({:.0}% overhead)",
            format!("{w}x{h}"),
            block,
            q,
            lanes,
            tile.is_fast(),
            warm_secs * 1e3,
            warm_gbps,
            warm_gsymps,
            cold_secs * 1e3,
            overhead_ms,
            ulp,
            verdict,
            overhead_frac * 100.0
        );
        }
    }
    println!("\nInterpretation: warm GB/s is the kernel-only ceiling of the CURRENT kernel. If it is still");
    println!("far below the 40+ GB/s target, per-call buffer overhead was NOT the dominant cost and the");
    println!("kernel is the limiter → STEP 3 = fuse rANS+inverse into one pass (no global `syms` round-trip).");
}

/// Median of a non-empty list of seconds (the A/B harness reports medians: the GPU is shared and noisy).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Env knob parsed as a comma list (`EXV_RT_BLOCKS=8,16,32`), else `default`.
fn env_list(name: &str, default: &[u32]) -> Vec<u32> {
    match std::env::var(name) {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}

/// P1-4 per-stage split of a WARM whole-frame EXV-RT decode (3 F16 channels as ONE `RtBatch`, the atlas shape):
/// rANS stage alone, inverse stage alone, both, and the plane blit (`blit_record`, 3 planes -> `R32Float`
/// textures) the resident atlas appends. Each figure = median of `EXV_RT_RUNS` (default 9) fenced runs.
/// Knobs: `EXV_RT_W/H` (default 7680x4320), `EXV_RT_BLOCKS` (16), `EXV_RT_QS` (0,4), `EXV_RT_LANES` (1).
/// Run ISOLATED: `cargo test --release -p exv-gpu --test exvrt_bench exvrt_gpu_stage_split -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "benchmark; run explicitly"]
fn exvrt_gpu_stage_split() {
    use exv_gpu_kernels::{RtBatch, RtDest, RtTile, TileEnc};
    use gpu_info::wgpu;
    let ctx = match exv_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) - skipping exvrt_gpu_stage_split");
            return;
        }
    };
    let dim = |n: &str, d: u32| std::env::var(n).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (w, h) = (dim("EXV_RT_W", 7680), dim("EXV_RT_H", 4320));
    let runs = dim("EXV_RT_RUNS", 9) as usize;
    let lanes = dim("EXV_RT_LANES", 1) as u8;
    let nch = 3usize;
    let info = ctx.adapter.get_info();
    println!("=== EXV-RT stage split: {} | {w}x{h} x{nch} F16, lanes {lanes}, median of {runs} ===", info.name);
    println!("{:>5} {:>2} | {:>8} {:>8} {:>8} {:>9} {:>8} | {:>8} {:>9} {:>5}", "block", "q", "rans ms", "inv ms", "both ms", "blit+alloc", "blit pass", "GB/s", "rans Gsym", "ULP");
    let srcs: Vec<Vec<u8>> = (0..nch).map(|c| hdr_ramp(w, h, 1.0 - 0.3 * c as f32, 0.1 * c as f32)).collect();
    for block in env_list("EXV_RT_BLOCKS", &[16]) {
        for q in env_list("EXV_RT_QS", &[0, 4]) {
            let payloads: Vec<Vec<u8>> = srcs
                .iter()
                .map(|s| exv_codec::blockrt_dwt::encode_plane(s, w, h, block, q, lanes).expect("encode_plane"))
                .collect();
            let sym: u64 = payloads.iter().map(|p| exvrt_symbols(p)).sum();
            let words = w as usize * h as usize;
            let outs: Vec<wgpu::Buffer> = (0..nch)
                .map(|_| exv_gpu_kernels::new_buf(&ctx.device, "bench.out", words as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC))
                .collect();
            let mut batch = RtBatch::new();
            for (c, p) in payloads.iter().enumerate() {
                batch.push(RtTile::parse(p, w, h).expect("parse"), RtDest::plane(c, 0, 0, w, h)).expect("push");
            }
            let refs: Vec<&wgpu::Buffer> = outs.iter().collect();
            let res = match ctx.scoped(|| batch.build(&ctx, &refs, &exv_format::bufpool::Cancel::new())) {
                Ok(r) => r,
                Err(e) => {
                    println!("{block:>5} {q:>2} | SKIPPED: {e}");
                    continue;
                }
            };
            // Correctness gate ONCE (channel 0 vs the CPU oracle) before any timing.
            res.dispatch(&ctx).expect("dispatch");
            ctx.drain().expect("drain");
            let tile = exv_gpu_kernels::GpuTile { buf: outs[0].clone(), off_words: 0, words, w, h, dtype: exv_format::Dtype::F16, enc: TileEnc::ElemPerWord(2) };
            let gpu = exv_gpu::read_back(&ctx, &tile).expect("readback");
            let cpu = exv_codec::blockrt_dwt::decode_plane(&payloads[0], w, h).expect("cpu decode");
            let ulp = max_ulp(&gpu, &cpu).expect("GPU decode BROKEN (length/sign), refusing to time it");
            assert!(ulp <= 1, "max ULP {ulp} > 1");
            let time = |f: &dyn Fn()| -> f64 {
                f();
                ctx.drain().expect("drain");
                let mut v: Vec<f64> = (0..runs)
                    .map(|_| {
                        let t0 = Instant::now();
                        f();
                        ctx.drain().expect("drain");
                        t0.elapsed().as_secs_f64()
                    })
                    .collect();
                median(&mut v)
            };
            let t_rans = time(&|| res.dispatch_stages(&ctx, true, false).expect("rans"));
            let t_inv = time(&|| res.dispatch_stages(&ctx, false, true).expect("inv"));
            let t_both = time(&|| res.dispatch_stages(&ctx, true, true).expect("both"));
            let t_blit = time(&|| {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench.blit") });
                let keep: Vec<_> = outs
                    .iter()
                    .map(|b| {
                        let t = exv_gpu_kernels::GpuTile { buf: b.clone(), off_words: 0, words, w, h, dtype: exv_format::Dtype::F16, enc: TileEnc::ElemPerWord(2) };
                        exv_gpu_kernels::blit_record(&ctx, &mut enc, &t, exv_format::Dtype::F16, w, h).expect("blit")
                    })
                    .collect();
                ctx.queue.submit([enc.finish()]);
                ctx.drain().expect("drain blit");
                drop(keep);
            });
            // Pure blit PASS over textures allocated ONCE (what `blit_record` costs beyond fresh-texture allocation + first-use clear).
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
            let texs: Vec<_> = (0..nch).map(|_| exv_gpu_kernels::tex2d(&ctx.device, "bench.tex", w, h, wgpu::TextureFormat::R32Float, usage)).collect();
            let params = exv_gpu_kernels::storage_u32(&ctx.device, &ctx.queue, "bench.params", &[w, h, 0, 1, 2, exv_format::Dtype::F16.code() as u32], wgpu::BufferUsages::empty());
            let views: Vec<_> = texs.iter().map(|t| t.create_view(&Default::default())).collect();
            let layout = ctx.pipelines.plane_blit_f32.get_bind_group_layout(0);
            let bgs: Vec<_> = outs
                .iter()
                .zip(&views)
                .map(|(b, v)| ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("bench.blit.bg"), layout: &layout, entries: &[exv_gpu_kernels::e(0, b), exv_gpu_kernels::e(1, &params), exv_gpu_kernels::et(2, v)] }))
                .collect();
            let t_pass = time(&|| {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench.blitpass") });
                for bg in &bgs {
                    exv_gpu_kernels::pass(&ctx, &mut enc, "bench.blit", &ctx.pipelines.plane_blit_f32, bg, [w.div_ceil(exv_gpu_kernels::WG2), h.div_ceil(exv_gpu_kernels::WG2), 1]).expect("pass");
                }
                ctx.queue.submit([enc.finish()]);
            });
            let gbps = (words * 2 * nch) as f64 / 1e9 / t_both;
            println!(
                "{block:>5} {q:>2} | {:>8.2} {:>8.2} {:>8.2} {:>9.2} {:>8.2} | {:>8.2} {:>9.3} {:>5}",
                t_rans * 1e3,
                t_inv * 1e3,
                t_both * 1e3,
                t_blit * 1e3,
                t_pass * 1e3,
                gbps,
                sym as f64 / 1e9 / t_rans,
                ulp
            );
        }
    }
}
