//! GPU vectorscope — zero-copy scatter from the linear source texture.
//!
//! YUV vectorscope rendered via GPU compute (wgpu 29). Depends on `wgpu`
//! directly. Control-point intent is preserved as inline comments.
//! The naga-24 fix for dynamic indexing of `let`-bound arrays in the draw
//! shader (using `var` instead of `let` for graticule tables) is documented
//! in the WGSL below.

use super::pipe::{
    PanelOutput, PanelParams, bins_bg, bins_only_bgl, create_compute_pipeline, draw_bg, draw_bgl,
    scatter_bg, scatter_bgl, write_panel_params,
};
use super::{
    BucketSizer, GpuPanel, PanelCtx, make_bins_buffer, make_params_buffer, sync_egui_texture,
};

/// Eight planes of 65536 bins: `count`, `sum_r`, `sum_g`, `sum_b`, each stored as a
/// `(lo, hi)` u32 pair (see [`SCATTER_WGSL`] for why the accumulators are 64-bit).
const BINS_COUNT: u32 = 65_536 * 8;
const BINS_BYTES: u64 = BINS_COUNT as u64 * 4;

/// `workgroup_size.x` of [`CLEAR_WGSL`]; the clear dispatch is derived from it so the
/// two cannot drift apart when `BINS_COUNT` changes.
const CLEAR_WG: u32 = 256;

const CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 524288u) { return; }
    atomicStore(&bins[gid.x], 0u);
}
"#;

const SCATTER_WGSL: &str = r#"
struct Params { src_w: u32, src_h: u32, out_w: u32, out_h: u32 }
@group(0) @binding(0) var               src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>      params: Params;

// Bins per plane; planes are [count_lo, count_hi, r_lo, r_hi, g_lo, g_hi, b_lo, b_hi].
const S: u32 = 65536u;

// Why 64-bit accumulators: every bin sums one contribution per pixel that lands in it,
// so the worst case is a solid-colour image where *all* pixels hit one bin. At the
// 65536x65536 ceiling that is 2^32 pixels; a u32 colour sum (up to 255 per pixel)
// wraps after ~2^24 such pixels — a 4K flat frame already gets close — and the dot
// then renders an arbitrary wrong colour. Each accumulator is therefore a (lo, hi)
// pair: `atomicAdd` returns the pre-add value, so the carry is detected exactly and
// no CAS loop is needed. The 64-bit total tops out at 2^32 * 255 < 2^40, so `hi`
// itself can never wrap.
fn add64(lo: u32, hi: u32, v: u32) {
    let old = atomicAdd(&bins[lo], v);
    if (old > 0xFFFFFFFFu - v) { atomicAdd(&bins[hi], 1u); }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.src_w || gid.y >= params.src_h) { return; }
    let px = textureLoad(src_tex, vec2i(i32(gid.x), i32(gid.y)), 0);
    let r  = clamp(px.r, 0.0, 1.0);
    let g  = clamp(px.g, 0.0, 1.0);
    let b  = clamp(px.b, 0.0, 1.0);
    let Y  = 0.299 * r + 0.587 * g + 0.114 * b;
    let u  = (b - Y) * 0.565;
    let v  = (r - Y) * 0.713;
    let bx = i32(round(128.0 + u * 204.8));
    let by = i32(round(128.0 - v * 204.8));
    if (bx < 0 || bx >= 256 || by < 0 || by >= 256) { return; }
    let idx = u32(by) * 256u + u32(bx);
    add64(idx,           S + idx,      1u);
    add64(2u * S + idx,  3u * S + idx, u32(r * 255.0));
    add64(4u * S + idx,  5u * S + idx, u32(g * 255.0));
    add64(6u * S + idx,  7u * S + idx, u32(b * 255.0));
}
"#;

const DRAW_WGSL: &str = r#"
struct Params { src_w: u32, src_h: u32, out_w: u32, out_h: u32 }
@group(0) @binding(0) var<storage, read> bins: array<u32>;
@group(0) @binding(1) var<uniform>       params: Params;
@group(0) @binding(2) var                out_tex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265;
const S: u32 = 65536u;

// Read back one (lo, hi) accumulator as f32. The 64-bit total reaches at most 2^40 and
// f32 is exact only to 2^24, but the value is used solely for a ratio and a log, never
// an exact comparison — the relative error is below 1e-7.
fn load64(lo: u32, hi: u32) -> f32 {
    return f32(bins[hi]) * 4294967296.0 + f32(bins[lo]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_w || gid.y >= params.out_h) { return; }
    let side = f32(min(params.out_w, params.out_h));
    let cx   = f32(params.out_w) * 0.5;
    let cy   = f32(params.out_h) * 0.5;
    let R    = side * 0.5 - 2.0;
    let px   = f32(gid.x) + 0.5;
    let py   = f32(gid.y) + 0.5;
    let dx   = px - cx;
    let dy   = py - cy;
    let dist = sqrt(dx * dx + dy * dy);

    var color = vec3f(20.0 / 255.0);
    if (abs(dist - R) < 1.5 || abs(dist - R * 0.5) < 1.0) {
        color = vec3f(60.0 / 255.0);
    }
    if (dist <= R * 0.5 && abs(dist - R * 0.5) >= 1.0) {
        color = vec3f(40.0 / 255.0);
    }

    if (dist <= R) {
        let bx_f = 128.0 + (dx / (R * 1.6)) * 204.8;
        let by_f = 128.0 - (dy / (R * 1.6)) * 204.8;
        let bx   = i32(round(bx_f));
        let by   = i32(round(by_f));
        if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
            let idx = u32(by) * 256u + u32(bx);
            let count = load64(idx, S + idx);
            if (count >= 1.0) {
                let inv = 1.0 / count;
                let dot_rgb = vec3f(
                    load64(2u * S + idx, 3u * S + idx) * inv / 255.0,
                    load64(4u * S + idx, 5u * S + idx) * inv / 255.0,
                    load64(6u * S + idx, 7u * S + idx) * inv / 255.0,
                );
                // The scatter visits every pixel, so normalise by all of them: a
                // bin saturates at 0.5% of the image at any resolution. (This used
                // `pixel_count / max(1, pixel_count / 10000)` ~ 10000, a leftover
                // of a subsampled scatter, which blew the scope out at 4K+.)
                // f32 product: `src_w * src_h` wraps u32 at 65536 x 65536.
                let pixel_count = f32(params.src_w) * f32(params.src_h);
                let log_max = log2(1.0 + pixel_count * 0.005);
                let intensity = clamp(log2(1.0 + count) / log_max, 0.0, 1.0);
                let a = 80.0 / 255.0 * intensity;
                color = mix(color, dot_rgb, a);
            }
        }
    }

    let mr = R * 0.85;
    // naga 24 (wgpu 29) forbids dynamic indexing of a `let`-bound (immutable,
    // non-addressable) array. The graticule loop below indexes these by the
    // runtime counter `i`, so they must be addressable `var`s or the `vs_draw`
    // compute module fails validation and the panel renders blank (the histogram/
    // waveform scopes have no such construct, which is why only this one broke).
    var targets = array<f32, 6>(0.0, 60.0, 120.0, 180.0, 240.0, 300.0);
    var mcolors = array<vec3f, 6>(
        vec3f(1.0, 0.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 1.0, 1.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 1.0),
    );
    for (var i = 0; i < 6; i = i + 1) {
        let ta  = targets[i] * PI / 180.0;
        let mx  = cx + cos(ta) * mr;
        let my  = cy - sin(ta) * mr;
        let ddx = px - mx;
        let ddy = py - my;
        if (ddx * ddx + ddy * ddy < 9.0) {
            color = mcolors[i];
        }
    }

    textureStore(out_tex, vec2i(i32(gid.x), i32(gid.y)), vec4f(color, 1.0));
}
"#;

struct Pipes {
    clear_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    draw_pipeline: wgpu::ComputePipeline,
    scatter_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    bins_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    bins_bg: wgpu::BindGroup,
    output: Option<PanelOutput>,
    draw_bg: Option<wgpu::BindGroup>,
    bucket: BucketSizer,
}

#[derive(Default)]
pub struct GpuVectorscopePanel {
    pipes: Option<Pipes>,
    texture_id: Option<egui::TextureId>,
}

fn init_pipelines(device: &wgpu::Device) -> Pipes {
    let bins_bgl = bins_only_bgl(device, "vs_bins_bgl");
    let scatter_bgl = scatter_bgl(device, "vs_scatter_bgl");
    let draw_bgl = draw_bgl(device, "vs_draw_bgl");

    let bins_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vs_bins_layout"),
        bind_group_layouts: &[Some(&bins_bgl)],
        immediate_size: 0,
    });
    let scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vs_scatter_layout"),
        bind_group_layouts: &[Some(&scatter_bgl)],
        immediate_size: 0,
    });
    let draw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vs_draw_layout"),
        bind_group_layouts: &[Some(&draw_bgl)],
        immediate_size: 0,
    });

    let bins_buf = make_bins_buffer(device, "vs_bins", BINS_BYTES);
    let params_buf = make_params_buffer(device);
    let bins_bg = bins_bg(device, &bins_bgl, "vs_bins_bg", &bins_buf);

    Pipes {
        clear_pipeline: create_compute_pipeline(device, "vs_clear", &bins_layout, CLEAR_WGSL),
        scatter_pipeline: create_compute_pipeline(
            device,
            "vs_scatter",
            &scatter_layout,
            SCATTER_WGSL,
        ),
        draw_pipeline: create_compute_pipeline(device, "vs_draw", &draw_layout, DRAW_WGSL),
        scatter_bgl,
        draw_bgl,
        bins_buf,
        params_buf,
        bins_bg,
        output: None,
        draw_bg: None,
        bucket: BucketSizer::default(),
    }
}

impl GpuVectorscopePanel {
    fn ensure_output(&mut self, device: &wgpu::Device, logical_w: u32, logical_h: u32) {
        let p = self.pipes.get_or_insert_with(|| init_pipelines(device));
        let ((bucket_w, bucket_h), reallocated) = p.bucket.update(logical_w, logical_h);
        if reallocated || p.output.is_none() {
            let output = PanelOutput::alloc(device, "vs_out", bucket_w, bucket_h);
            p.draw_bg = Some(draw_bg(
                device,
                &p.draw_bgl,
                "vs_draw_bg",
                &p.bins_buf,
                &p.params_buf,
                &output.view,
            ));
            p.output = Some(output);
        }
    }
}

impl GpuPanel for GpuVectorscopePanel {
    fn tick(
        &mut self,
        ctx: &mut PanelCtx,
        src: &wgpu::TextureView,
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) {
        // Control-point (historically): the vectorscope tick was suspected of
        // never running with a non-degenerate square output, leaving texture_id
        // None and forcing the empty CPU fallback. We keep the early-out so a
        // degenerate src/out size simply skips (never panics).
        if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
            return;
        }
        self.ensure_output(ctx.device, out_w, out_h);
        let Some(p) = self.pipes.as_mut() else {
            return;
        };
        let Some(out) = p.output.as_ref() else {
            return;
        };
        let Some(draw_bg) = p.draw_bg.as_ref() else {
            return;
        };

        write_panel_params(
            ctx.queue,
            &p.params_buf,
            PanelParams {
                src_w,
                src_h,
                out_w,
                out_h,
            },
        );
        let scatter_bg = scatter_bg(
            ctx.device,
            &p.scatter_bgl,
            "vs_scatter_bg",
            src,
            &p.bins_buf,
            &p.params_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vs_enc"),
            });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vs"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&p.clear_pipeline);
            cp.set_bind_group(0, &p.bins_bg, &[]);
            cp.dispatch_workgroups(BINS_COUNT.div_ceil(CLEAR_WG), 1, 1);

            cp.set_pipeline(&p.scatter_pipeline);
            cp.set_bind_group(0, &scatter_bg, &[]);
            cp.dispatch_workgroups(src_w.div_ceil(16), src_h.div_ceil(16), 1);

            cp.set_pipeline(&p.draw_pipeline);
            cp.set_bind_group(0, draw_bg, &[]);
            cp.dispatch_workgroups(out_w.div_ceil(8), out_h.div_ceil(8), 1);
        }
        ctx.queue.submit(std::iter::once(enc.finish()));
        sync_egui_texture(ctx, &out.view, &mut self.texture_id);
        // Control-point (historically): once `texture_id` is Some, the panel
        // takes the GPU branch instead of an empty fallback. If the panel still
        // looks blank after this, the fault is downstream (feed), not compute.
    }

    fn release(&mut self, renderer: &mut egui_wgpu::Renderer) {
        if let Some(id) = self.texture_id.take() {
            renderer.free_texture(&id);
        }
        // Drop the pipelines + output texture; rebuilt lazily on the next tick.
        self.pipes = None;
    }

    fn texture_id(&self) -> Option<egui::TextureId> {
        self.texture_id
    }

    fn physical_bucket_size(&self) -> (u32, u32) {
        self.pipes
            .as_ref()
            .map(|p| p.bucket.physical())
            .unwrap_or((0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scopes::wgsl_probe::{count, guard_limit, validate, workgroup_x};

    /// Bins per plane, mirroring `const S` in the shaders.
    const PLANE: u32 = 65_536;

    #[test]
    fn clear_covers_every_bin() {
        assert_eq!(BINS_COUNT, PLANE * 8);
        assert_eq!(guard_limit(CLEAR_WGSL), Some(BINS_COUNT));
        assert_eq!(workgroup_x(CLEAR_WGSL), Some(CLEAR_WG));
        assert!(BINS_COUNT.div_ceil(CLEAR_WG) * CLEAR_WG >= BINS_COUNT);
    }

    #[test]
    fn shaders_agree_on_plane_stride() {
        let stride = format!("const S: u32 = {PLANE}u;");
        assert!(SCATTER_WGSL.contains(&stride));
        assert!(DRAW_WGSL.contains(&stride));
        // Highest plane index used is 7, so the buffer must hold 8 planes.
        assert!(SCATTER_WGSL.contains("7u * S + idx"));
        assert_eq!(BINS_COUNT, 8 * PLANE);
    }

    #[test]
    fn colour_sums_accumulate_64_bit() {
        // The wrapping u32 accumulation (`atomicAdd(&bins[65536u + idx], u32(r*255.0))`)
        // must be gone: every accumulator goes through the carry-propagating add64.
        assert!(SCATTER_WGSL.contains("fn add64(lo: u32, hi: u32, v: u32)"));
        assert!(SCATTER_WGSL.contains("if (old > 0xFFFFFFFFu - v)"));
        // Definition + one call per accumulator (count, r, g, b).
        assert_eq!(count(SCATTER_WGSL, "add64("), 5);
        // No bare atomicAdd remains outside add64's own body.
        assert_eq!(count(SCATTER_WGSL, "atomicAdd("), 2);
        assert!(DRAW_WGSL.contains("fn load64(lo: u32, hi: u32) -> f32"));
        assert_eq!(count(DRAW_WGSL, "load64("), 5);
    }

    /// CPU model of the shader's `add64`, used to prove the carry condition.
    fn add64(acc: (u32, u32), v: u32) -> (u32, u32) {
        let (lo, hi) = acc;
        let carry = u32::from(lo > u32::MAX - v);
        (lo.wrapping_add(v), hi + carry)
    }

    #[test]
    fn carry_condition_matches_u64_addition() {
        let mut acc = (0u32, 0u32);
        let mut want = 0u64;
        for v in [0, 1, 255, u32::MAX, u32::MAX, 255, 1, 0, u32::MAX - 1, 2] {
            acc = add64(acc, v);
            want += u64::from(v);
            assert_eq!(u64::from(acc.1) << 32 | u64::from(acc.0), want);
        }
    }

    #[test]
    fn shaders_validate() {
        validate("vectorscope clear", CLEAR_WGSL);
        validate("vectorscope scatter", SCATTER_WGSL);
        validate("vectorscope draw", DRAW_WGSL);
    }

    /// CPU twin of the draw shader's dot intensity.
    fn intensity(count: f32, src_w: u32, src_h: u32) -> f32 {
        let pixel_count = src_w as f32 * src_h as f32;
        ((1.0 + count).log2() / (1.0 + pixel_count * 0.005).log2()).clamp(0.0, 1.0)
    }

    #[test]
    fn intensity_scales_with_the_scattered_pixel_count() {
        assert!(DRAW_WGSL.contains("let pixel_count = f32(params.src_w) * f32(params.src_h);"));
        assert!(DRAW_WGSL.contains("let log_max = log2(1.0 + pixel_count * 0.005);"));
        assert!(!DRAW_WGSL.contains("pixel_count / step"), "subsampled-era normaliser is back");
        // The same fraction of the image reads the same at every resolution ...
        for (w, h) in [(1920u32, 1080u32), (3840, 2160), (8192, 4320), (65_536, 65_536)] {
            let n = w as f32 * h as f32;
            assert!((intensity(n * 0.005, w, h) - 1.0).abs() < 1e-6, "{w}x{h}: 0.5% saturates");
            let tenth = intensity(n * 0.0005, w, h);
            assert!(tenth > 0.7 && tenth < 0.9, "{w}x{h}: 0.05% stays below saturation, got {tenth}");
        }
        // ... where the old ~10000-sample normaliser saturated a 4K bin holding
        // 0.05% of the pixels (4147 of 8.3M): the reported blow-out.
        let old_log_max = (1.0f32 + 10_000.0 * 0.005).log2();
        assert!((1.0f32 + 3840.0 * 2160.0 * 0.0005).log2() / old_log_max > 1.0);
    }

    #[test]
    fn worst_case_image_cannot_wrap_64_bits() {
        // Every pixel of the largest image we design for landing in one bin, each
        // contributing the maximum 255: this is what overflowed a u32 accumulator.
        let pixels = 65_536u64 * 65_536;
        let worst = pixels * 255;
        assert!(worst > u64::from(u32::MAX), "the u32 bug was real");
        assert!(worst < u64::MAX, "the 64-bit accumulator holds it");
        // `hi` counts 2^32-sized carries, so it stays tiny.
        assert!(worst >> 32 < u64::from(u32::MAX));
    }
}
