//! GPU histogram — zero-copy scatter from the linear source texture.
//!
//! Per-channel R/G/B histogram rendered via GPU compute (wgpu 29).
//! Depends on `wgpu` directly (same version as egui_wgpu).

use super::pipe::{
    PanelOutput, PanelParams, bins_bg, bins_only_bgl, create_compute_pipeline, draw_bg, draw_bgl,
    scatter_bg, scatter_bgl, write_panel_params,
};
use super::{
    BucketSizer, GpuPanel, PanelCtx, make_bins_buffer, make_params_buffer, sync_egui_texture,
};

/// Three planes of 256 bins (R, G, B) plus one slot holding the draw normaliser.
///
/// A fourth luma plane used to be scattered into bins 768..1023, but neither REDUCE
/// nor DRAW ever read it — the series was computed and thrown away every frame. It is
/// deleted rather than drawn: a luma overlay is a feature request, not a bug fix, and
/// dropping it removes one atomic per pixel (a quarter of the scatter traffic).
const BINS_COUNT: u32 = 769;
const BINS_BYTES: u64 = BINS_COUNT as u64 * 4;

/// `workgroup_size.x` of [`CLEAR_WGSL`]; the clear dispatch is derived from it so the
/// two cannot drift apart when `BINS_COUNT` changes.
const CLEAR_WG: u32 = 64;

const CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 769u) { return; }
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
    let r = clamp(px.r, 0.0, 1.0);
    let g = clamp(px.g, 0.0, 1.0);
    let b = clamp(px.b, 0.0, 1.0);
    atomicAdd(&bins[       u32(r * 255.0)], 1u);
    atomicAdd(&bins[256u + u32(g * 255.0)], 1u);
    atomicAdd(&bins[512u + u32(b * 255.0)], 1u);
}
"#;

// The normaliser skips the two clipped bins (0 = crushed black, 255 = blown white).
// They are not part of the distribution being inspected: a letterboxed or matted frame
// puts millions of pixels in bin 0, which as the normaliser flattens every real bin to
// a pixel of height and makes the plot look empty. DRAW clamps the two clipped bins to
// full height instead, so they still read as "clipped" without setting the scale.
const REDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u || gid.x >= 255u) { return; }
    let m = max(max(
        atomicLoad(&bins[gid.x]),
        atomicLoad(&bins[256u + gid.x])),
        atomicLoad(&bins[512u + gid.x]));
    atomicMax(&bins[768u], m);
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
    let bin  = min(u32((f32(gid.x) / f32(params.out_w)) * 256.0), 255u);
    let mx   = f32(max(bins[768u], 1u));
    let frac = 1.0 - f32(gid.y) / f32(params.out_h);
    // min(.., 1.0): bins 0 and 255 are excluded from the normaliser (see REDUCE), so
    // their raw height can exceed it — clamp instead of letting them overflow the plot.
    let rh   = min(f32(bins[       bin]) / mx, 1.0);
    let gh   = min(f32(bins[256u + bin]) / mx, 1.0);
    let bh   = min(f32(bins[512u + bin]) / mx, 1.0);
    let bg   = vec3f(30.0 / 255.0);
    let a    = 150.0 / 255.0;
    var c    = bg;
    if (frac < rh) { c = mix(c, vec3f(1.0, 0.0, 0.0),                a); }
    if (frac < gh) { c = mix(c, vec3f(0.0, 1.0, 0.0),                a); }
    if (frac < bh) { c = mix(c, vec3f(80.0/255.0, 120.0/255.0, 1.0), a); }
    textureStore(out_tex, vec2i(i32(gid.x), i32(gid.y)), vec4f(c, 1.0));
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
pub struct GpuHistogramPanel {
    pipes: Option<Pipes>,
    texture_id: Option<egui::TextureId>,
}

fn init_pipelines(device: &wgpu::Device) -> Pipes {
    let bins_bgl = bins_only_bgl(device, "hist_bins_bgl");
    let scatter_bgl = scatter_bgl(device, "hist_scatter_bgl");
    let draw_bgl = draw_bgl(device, "hist_draw_bgl");

    let bins_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hist_bins_layout"),
        bind_group_layouts: &[Some(&bins_bgl)],
        immediate_size: 0,
    });
    let scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hist_scatter_layout"),
        bind_group_layouts: &[Some(&scatter_bgl)],
        immediate_size: 0,
    });
    let draw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hist_draw_layout"),
        bind_group_layouts: &[Some(&draw_bgl)],
        immediate_size: 0,
    });

    let bins_buf = make_bins_buffer(device, "hist_bins", BINS_BYTES);
    let params_buf = make_params_buffer(device);
    let bins_bg = bins_bg(device, &bins_bgl, "hist_bins_bg", &bins_buf);

    Pipes {
        clear_pipeline: create_compute_pipeline(device, "hist_clear", &bins_layout, CLEAR_WGSL),
        scatter_pipeline: create_compute_pipeline(
            device,
            "hist_scatter",
            &scatter_layout,
            SCATTER_WGSL,
        ),
        reduce_pipeline: create_compute_pipeline(device, "hist_reduce", &bins_layout, REDUCE_WGSL),
        draw_pipeline: create_compute_pipeline(device, "hist_draw", &draw_layout, DRAW_WGSL),
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

impl GpuHistogramPanel {
    fn ensure_output(&mut self, device: &wgpu::Device, logical_w: u32, logical_h: u32) {
        let p = self.pipes.get_or_insert_with(|| init_pipelines(device));
        let ((bucket_w, bucket_h), reallocated) = p.bucket.update(logical_w, logical_h);
        if reallocated || p.output.is_none() {
            let output = PanelOutput::alloc(device, "hist_out", bucket_w, bucket_h);
            p.draw_bg = Some(draw_bg(
                device,
                &p.draw_bgl,
                "hist_draw_bg",
                &p.bins_buf,
                &p.params_buf,
                &output.view,
            ));
            p.output = Some(output);
        }
    }
}

impl GpuPanel for GpuHistogramPanel {
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
            "hist_scatter_bg",
            src,
            &p.bins_buf,
            &p.params_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hist_enc"),
            });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hist"),
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
        sync_egui_texture(ctx, &out.view, &mut self.texture_id);
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

    #[test]
    fn shaders_validate() {
        crate::scopes::wgsl_probe::validate("histogram clear", CLEAR_WGSL);
        crate::scopes::wgsl_probe::validate("histogram scatter", SCATTER_WGSL);
        crate::scopes::wgsl_probe::validate("histogram reduce", REDUCE_WGSL);
        crate::scopes::wgsl_probe::validate("histogram draw", DRAW_WGSL);
    }
    use super::*;
    use crate::scopes::wgsl_probe::{count, guard_limit, workgroup_x};

    /// Bins per channel plane.
    const PLANE: u32 = 256;

    #[test]
    fn clear_covers_every_bin() {
        assert_eq!(BINS_COUNT, PLANE * 3 + 1);
        assert_eq!(guard_limit(CLEAR_WGSL), Some(BINS_COUNT));
        assert_eq!(workgroup_x(CLEAR_WGSL), Some(CLEAR_WG));
        assert!(BINS_COUNT.div_ceil(CLEAR_WG) * CLEAR_WG >= BINS_COUNT);
    }

    #[test]
    fn luma_plane_is_gone() {
        // The scatter used to fill bins 768..1023 with a luma series that neither
        // REDUCE nor DRAW read. One atomic per channel plane, three planes, no more.
        assert_eq!(count(SCATTER_WGSL, "atomicAdd("), 3);
        assert!(!SCATTER_WGSL.contains("768u"));
        assert!(!SCATTER_WGSL.contains("0.2126"));
        for wgsl in [CLEAR_WGSL, SCATTER_WGSL, REDUCE_WGSL, DRAW_WGSL] {
            assert!(!wgsl.contains("1024u"), "stale luma-era bin index");
        }
    }

    #[test]
    fn normaliser_slot_is_the_last_bin() {
        let slot = format!("bins[{}u]", BINS_COUNT - 1);
        assert!(REDUCE_WGSL.contains(&format!("atomicMax(&{slot}")));
        assert!(DRAW_WGSL.contains(&format!("max({slot}, 1u)")));
    }

    #[test]
    fn normaliser_skips_the_clipped_bins() {
        // Bins 0 and 255 must not set the scale (a matted frame puts millions of
        // pixels in bin 0), and DRAW must clamp them so they still read as clipped.
        assert!(REDUCE_WGSL.contains("if (gid.x == 0u || gid.x >= 255u) { return; }"));
        assert_eq!(count(DRAW_WGSL, "min(f32(bins["), 3);
        assert_eq!(count(DRAW_WGSL, "/ mx, 1.0)"), 3);
    }
}
