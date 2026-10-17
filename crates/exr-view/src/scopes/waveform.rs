//! GPU waveform — zero-copy scatter from the linear source texture.
//!
//! Per-column RGB waveform rendered via GPU compute (wgpu 29).
//! Depends on `wgpu` directly (same version as egui_wgpu).

use super::pipe::{
    PanelOutput, PanelParams, bins_bg, bins_only_bgl, create_compute_pipeline, draw_bg, draw_bgl,
    scatter_bg, scatter_bgl, write_panel_params,
};
use super::{
    BucketSizer, GpuPanel, PanelCtx, RenderedScope, make_bins_buffer, make_params_buffer,
    sync_egui_texture,
};

/// 3 planes of 256 columns x 256 rows, plus 3 planes of 256 per-column maxima.
const BINS_COUNT: u32 = 197_376;
const BINS_BYTES: u64 = BINS_COUNT as u64 * 4;

/// `workgroup_size.x` of [`CLEAR_WGSL`]; the clear dispatch is derived from it so the
/// two cannot drift apart when `BINS_COUNT` changes.
const CLEAR_WG: u32 = 256;

const CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 197376u) { return; }
    atomicStore(&bins[gid.x], 0u);
}
"#;

const SCATTER_WGSL: &str = r#"
struct Params { src_w: u32, src_h: u32, out_w: u32, out_h: u32 }
@group(0) @binding(0) var               src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>      params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.src_w || gid.y >= params.src_h) { return; }
    let px = textureLoad(src_tex, vec2i(i32(gid.x), i32(gid.y)), 0);
    let col = min(u32((f32(gid.x) / f32(params.src_w)) * 256.0), 255u);
    let r = clamp(px.r, 0.0, 1.0);
    let g = clamp(px.g, 0.0, 1.0);
    let b = clamp(px.b, 0.0, 1.0);
    let rr = u32(r * 255.0);
    let gr = u32(g * 255.0);
    let br = u32(b * 255.0);
    atomicAdd(&bins[          col * 256u + rr], 1u);
    atomicAdd(&bins[65536u  + col * 256u + gr], 1u);
    atomicAdd(&bins[131072u + col * 256u + br], 1u);
}
"#;

const REDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 256u) { return; }
    let col = gid.x;
    var mr: u32 = 1u;
    var mg: u32 = 1u;
    var mb: u32 = 1u;
    for (var row: u32 = 0u; row < 256u; row = row + 1u) {
        mr = max(mr, atomicLoad(&bins[          col * 256u + row]));
        mg = max(mg, atomicLoad(&bins[65536u  + col * 256u + row]));
        mb = max(mb, atomicLoad(&bins[131072u + col * 256u + row]));
    }
    atomicStore(&bins[196608u + col], mr);
    atomicStore(&bins[196864u + col], mg);
    atomicStore(&bins[197120u + col], mb);
}
"#;

const DRAW_WGSL: &str = r#"
struct Params { src_w: u32, src_h: u32, out_w: u32, out_h: u32 }
@group(0) @binding(0) var<storage, read> bins: array<u32>;
@group(0) @binding(1) var<uniform>      params: Params;
@group(0) @binding(2) var               out_tex: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_w || gid.y >= params.out_h) { return; }
    let col  = min(u32((f32(gid.x) / f32(params.out_w)) * 256.0), 255u);
    let frac = 1.0 - f32(gid.y) / f32(params.out_h);
    let row  = min(u32(frac * 256.0), 255u);
    let rr   = bins[          col * 256u + row];
    let gr   = bins[65536u  + col * 256u + row];
    let br   = bins[131072u + col * 256u + row];
    let mr   = max(bins[196608u + col], 1u);
    let mg   = max(bins[196864u + col], 1u);
    let mb   = max(bins[197120u + col], 1u);
    let r    = f32(rr) / f32(mr);
    let g    = f32(gr) / f32(mg);
    let b    = f32(br) / f32(mb);
    let bg   = vec3f(25.0 / 255.0);
    let thr  = 8.0 / 255.0;
    if (r > thr || g > thr || b > thr) {
        let a   = 120.0 / 255.0;
        let rgb = mix(bg, vec3f(r, g, b), a);
        textureStore(out_tex, vec2i(i32(gid.x), i32(gid.y)), vec4f(rgb, 1.0));
    } else {
        textureStore(out_tex, vec2i(i32(gid.x), i32(gid.y)), vec4f(bg, 1.0));
    }
}
"#;

struct Pipes {
    clear_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
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
pub struct GpuWaveformPanel {
    pipes: Option<Pipes>,
    texture_id: Option<egui::TextureId>,
    rendered: Option<RenderedScope>,
}

fn init_pipelines(device: &wgpu::Device) -> Pipes {
    let bins_bgl = bins_only_bgl(device, "wf_bins_bgl");
    let scatter_bgl = scatter_bgl(device, "wf_scatter_bgl");
    let draw_bgl = draw_bgl(device, "wf_draw_bgl");

    let bins_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wf_bins_layout"),
        bind_group_layouts: &[Some(&bins_bgl)],
        immediate_size: 0,
    });
    let scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wf_scatter_layout"),
        bind_group_layouts: &[Some(&scatter_bgl)],
        immediate_size: 0,
    });
    let draw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wf_draw_layout"),
        bind_group_layouts: &[Some(&draw_bgl)],
        immediate_size: 0,
    });

    let bins_buf = make_bins_buffer(device, "wf_bins", BINS_BYTES);
    let params_buf = make_params_buffer(device);
    let bins_bg = bins_bg(device, &bins_bgl, "wf_bins_bg", &bins_buf);

    Pipes {
        clear_pipeline: create_compute_pipeline(device, "wf_clear", &bins_layout, CLEAR_WGSL),
        scatter_pipeline: create_compute_pipeline(
            device,
            "wf_scatter",
            &scatter_layout,
            SCATTER_WGSL,
        ),
        reduce_pipeline: create_compute_pipeline(device, "wf_reduce", &bins_layout, REDUCE_WGSL),
        draw_pipeline: create_compute_pipeline(device, "wf_draw", &draw_layout, DRAW_WGSL),
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

impl GpuWaveformPanel {
    fn ensure_output(&mut self, device: &wgpu::Device, logical_w: u32, logical_h: u32) -> bool {
        let p = self.pipes.get_or_insert_with(|| init_pipelines(device));
        let ((bucket_w, bucket_h), reallocated) = p.bucket.update(logical_w, logical_h);
        if reallocated || p.output.is_none() {
            let output = PanelOutput::alloc(device, "wf_out", bucket_w, bucket_h);
            p.draw_bg = Some(draw_bg(
                device,
                &p.draw_bgl,
                "wf_draw_bg",
                &p.bins_buf,
                &p.params_buf,
                &output.view,
            ));
            p.output = Some(output);
            return true;
        }
        false
    }
}

impl GpuPanel for GpuWaveformPanel {
    fn tick(
        &mut self,
        ctx: &mut PanelCtx,
        src: &wgpu::TextureView,
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) {
        if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
            return;
        }
        let reallocated = self.ensure_output(ctx.device, out_w, out_h);
        if !reallocated
            && self.texture_id.is_some()
            && ctx
                .source_key
                .is_some_and(|key| self.rendered == Some((key, src_w, src_h, out_w, out_h)))
        {
            return;
        }
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
            "wf_scatter_bg",
            src,
            &p.bins_buf,
            &p.params_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wf_enc"),
            });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wf"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&p.clear_pipeline);
            cp.set_bind_group(0, &p.bins_bg, &[]);
            cp.dispatch_workgroups(BINS_COUNT.div_ceil(CLEAR_WG), 1, 1);

            cp.set_pipeline(&p.scatter_pipeline);
            cp.set_bind_group(0, &scatter_bg, &[]);
            cp.dispatch_workgroups(src_w.div_ceil(16), src_h.div_ceil(16), 1);

            cp.set_pipeline(&p.reduce_pipeline);
            cp.set_bind_group(0, &p.bins_bg, &[]);
            cp.dispatch_workgroups(4, 1, 1);

            cp.set_pipeline(&p.draw_pipeline);
            cp.set_bind_group(0, draw_bg, &[]);
            cp.dispatch_workgroups(out_w.div_ceil(8), out_h.div_ceil(8), 1);
        }
        ctx.queue.submit(std::iter::once(enc.finish()));
        if reallocated || self.texture_id.is_none() {
            sync_egui_texture(ctx, &out.view, &mut self.texture_id);
        }
        self.rendered = ctx.source_key.map(|key| (key, src_w, src_h, out_w, out_h));
    }

    fn release(&mut self, renderer: &mut egui_wgpu::Renderer) {
        if let Some(id) = self.texture_id.take() {
            renderer.free_texture(&id);
        }
        // Drop the pipelines + output texture; rebuilt lazily on the next tick.
        self.pipes = None;
        self.rendered = None;
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

    #[test]
    fn shaders_validate() {
        crate::scopes::wgsl_probe::validate("waveform clear", CLEAR_WGSL);
        crate::scopes::wgsl_probe::validate("waveform scatter", SCATTER_WGSL);
        crate::scopes::wgsl_probe::validate("waveform reduce", REDUCE_WGSL);
        crate::scopes::wgsl_probe::validate("waveform draw", DRAW_WGSL);
    }
    use super::*;
    use crate::scopes::wgsl_probe::{guard_limit, workgroup_x};

    #[test]
    fn clear_covers_every_bin() {
        // 3 x (256 columns x 256 rows) sample planes + 3 x 256 per-column maxima.
        assert_eq!(BINS_COUNT, 3 * 256 * 256 + 3 * 256);
        assert_eq!(guard_limit(CLEAR_WGSL), Some(BINS_COUNT));
        assert_eq!(workgroup_x(CLEAR_WGSL), Some(CLEAR_WG));
        assert!(BINS_COUNT.div_ceil(CLEAR_WG) * CLEAR_WG >= BINS_COUNT);
    }
}
