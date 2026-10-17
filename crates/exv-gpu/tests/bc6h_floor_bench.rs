//! Tier-0 (CODEC_SPEED_PLAN.md) — PROVE the floor/ceiling for an embarrassingly-parallel block
//! codec, with ~zero codec code: can a per-block GPU decode produce a 16k f16 RGBA frame at 40+ GB/s?
//!
//! This measures the DECODE-and-WRITE floor of ANY fixed-rate block codec (BC6H, ASTC, our future
//! EXV-RT) by running a REPRESENTATIVE block decode: one workgroup invocation per 4x4 block reads a
//! fixed 16-byte block (8 bpp, = BC6H's ratio), reconstructs 16 texels by endpoint-lerp (the core
//! cost of every block codec), and writes them to a resident `Rgba16Float` storage texture. It is
//! NOT a full 14-mode BC6H decoder (we measure throughput, not bit-exactness) — the per-pixel cost
//! (read block, lerp two endpoints by a small index, store f16) is the right model, and the run is
//! dominated by the f16 RGBA output write, which is the unavoidable final step of ANY decoder.
//!
//! Numbers reported: GB/s over the f16 RGBA OUTPUT (the 50 GB/s @ 24fps-16k target metric) + the
//! compressed input size (8 bpp). If this clears 40 GB/s, a block codec is bandwidth-feasible and the
//! only remaining question is how much decode work we add on top (Tier 1 HT kernel / Tier 2 entropy).
//! Run: `cargo test -p exv-gpu --test bc6h_floor_bench -- --ignored --nocapture`.

use gpu_info::wgpu;
use std::time::Instant;

/// Representative fixed-rate block decode: 16-byte block (8 bpp) -> 4x4 RGBA16F texels by endpoint
/// lerp. Layout per block (4 u32): w0,w1 = two RGB f16 endpoints (6 f16 across 3 u32 w0,w1,w2.lo),
/// w3 = 16 x 2-bit interpolation indices. One invocation decodes one block (16 texels).
const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> blocks: array<u32>;          // 4 u32 (16 B) per 4x4 block
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;                   // [wblocks, hblocks, _, _]

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= dims.x || by >= dims.y) { return; }
    let base = (by * dims.x + bx) * 4u;
    let w0 = blocks[base];
    let w1 = blocks[base + 1u];
    let w2 = blocks[base + 2u];
    let w3 = blocks[base + 3u];
    // Two RGB endpoints from the packed f16 lanes (representative; real BC6H quantizes these).
    let a = unpack2x16float(w0);
    let b = unpack2x16float(w1);
    let c = unpack2x16float(w2);
    let e0 = vec3<f32>(a.x, a.y, b.x);
    let e1 = vec3<f32>(b.y, c.x, c.y);
    // 16 texels: 2-bit index -> t in {0,1/3,2/3,1}; lerp endpoints; store f16.
    for (var j = 0u; j < 4u; j = j + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let t = f32((w3 >> ((j * 4u + i) * 2u)) & 3u) / 3.0;
            let col = mix(e0, e1, t);
            textureStore(out_tex, vec2<i32>(i32(bx * 4u + i), i32(by * 4u + j)), vec4<f32>(col, 1.0));
        }
    }
}
"#;

#[test]
#[ignore = "requires a GPU"]
fn block_decode_floor_16k() {
    let ctx = match exv_gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}) — skipping");
            return;
        }
    };
    let dev = &ctx.device;
    let q = &ctx.queue;

    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bc6h.floor"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bc6h.floor"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bgl = pipeline.get_bind_group_layout(0);

    let max_dim = dev.limits().max_texture_dimension_2d;
    let max_binding = dev.limits().max_storage_buffer_binding_size as u64;
    eprintln!("device: max_texture_dim_2d={max_dim}, max_storage_binding={} MiB", max_binding >> 20);

    for &edge in &[4096u32, 8192, 16384] {
        if edge > max_dim {
            eprintln!("{edge}x{edge}: exceeds max_texture_dimension_2d {max_dim} — skipped");
            continue;
        }
        let wblocks = edge / 4;
        let hblocks = edge / 4;
        let n_blocks = (wblocks as u64) * (hblocks as u64);
        let in_bytes = n_blocks * 16; // 8 bpp compressed input
        if in_bytes > max_binding {
            eprintln!("{edge}x{edge}: input {} MiB exceeds binding cap {} MiB — skipped", in_bytes >> 20, max_binding >> 20);
            continue;
        }
        let out_bytes = (edge as u64) * (edge as u64) * 8; // RGBA16F output (the target metric)

        // Input block buffer (content irrelevant for throughput; zero-filled).
        let blocks = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blocks"),
            size: in_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Resident output texture (no readback — the realtime path samples it directly).
        let tex = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("out.rgba16f"),
            size: wgpu::Extent3d { width: edge, height: edge, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        // 16-byte uniform: [wblocks, hblocks, 0, 0] as LE bytes (no bytemuck dep in this test).
        let mut dims_bytes = Vec::with_capacity(16);
        for v in [wblocks, hblocks, 0u32, 0u32] {
            dims_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let ubuf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        q.write_buffer(&ubuf, 0, &dims_bytes);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bc6h.bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: blocks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        });

        let gx = wblocks.div_ceil(8);
        let gy = hblocks.div_ceil(8);
        let dispatch = |label: &str| {
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            {
                let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &bind, &[]);
                p.dispatch_workgroups(gx, gy, 1);
            }
            q.submit(Some(enc.finish()));
        };

        // Warm up (pipeline/driver), then time decode+drain (no readback in the timed region).
        for _ in 0..3 {
            dispatch("warm");
        }
        ctx.drain().expect("drain warm");

        let iters = 20u32;
        let t = Instant::now();
        for _ in 0..iters {
            dispatch("timed");
        }
        ctx.drain().expect("drain timed");
        let secs = t.elapsed().as_secs_f64() / iters as f64;

        eprintln!(
            "{edge}x{edge}: block-decode -> RGBA16F  {:.1} GB/s (out)  | {:.2} ms/frame  | in {} MiB (8bpp), out {} MiB  | {:.1} fps",
            out_bytes as f64 / 1e9 / secs,
            secs * 1e3,
            in_bytes >> 20,
            out_bytes >> 20,
            1.0 / secs,
        );
    }
}
