//! Tile renderer: one instanced quad per
//! visible tile, drawn straight into egui's render pass through an
//! `egui_wgpu` paint callback.
//!
//! The fragment shader runs stage 2 over scene-linear pixels with associated alpha. Stage 1
//! (`crate::ocio::Decode`) has already converted the source before layer assembly, effects,
//! and pyramid reduction. Colour is unpremultiplied, then processed by the OCIO display
//! transform or the built-in display gain, exposure x 2^EV, tone operator, gamma dial, and
//! sRGB OETF from [`crate::transfer`], then premultiplied in light. Isolated channels read the
//! same decoded pixels. It writes the canvas
//! (`present::CANVAS_FORMAT`) in the canvas encoding, extended sRGB. What it stores is the canvas
//! contract of `soluwell_viewport::display::backdrop`: encoded ASSOCIATED light, `rgb = oetf(L * a)` of the
//! colour's display light `L`, `a = raw.a` (`transfer::premul`; an associated texel with light
//! and no coverage, a flare, is legal and stays as additive light). That differs from egui's UI
//! (`encode(straight) * alpha`) exactly where the present pass's backdrop composite needs it to:
//! `eotf(rgb) + backdrop * (1 - a)` is OVER in linear light. The blend onto the canvas is egui-wgpu's premultiplied one; dithering and the
//! surface's own encoding are the present pass's (`present.rs`), so this shader does not
//! know the monitor.
//!
//! With an OCIO display transform ([`crate::ocio::Transform`]) the colour image goes
//! through its `ocio_transform` instead of the tone operator: the shader is
//! generated with the transform's WGSL prelude, whose bindings take groups 1 and 2
//! (`GpuOcioResources::build_for`), so the atlas page sits at group 3
//! (`OCIO_HOST_GROUP_START`) in both modes. An encoded transform (SDR output)
//! yields the display's own code values, stored as they are; a linear one (HDR
//! output) yields linear Rec.709 light, 1.0 = SDR white, encoded like the
//! built-in chain. [`TilePipeline::set_colour`] rebuilds the pipeline when the
//! transform changes. The OCIO branch has no exposure of its own: the exposure is the
//! transform's dynamic `linearCC` (in `scene_linear`), whose uniforms `prepare` moves.

use std::ops::Range;

use vfx_ocio::{GpuOcioResources, OCIO_HOST_GROUP_START, OCIO_TEXTURE_GROUP, OCIO_UNIFORM_GROUP};

use crate::ocio::Transform;
use crate::transfer::{self, LUMA_709};

use super::atlas::TileAtlas;
use super::view::TileKey;

/// Everything the tile path keeps on the GPU, stored in egui's
/// `callback_resources` so the paint callback can bind it.
pub struct TileGpu {
    /// The tile atlas.
    pub atlas: TileAtlas<TileKey>,
    /// Pipeline and per-frame buffers.
    pub pipe: TilePipeline,
    pub direct: Option<super::direct::DirectGpu>,
    /// Driver-derived image-display allowance; reserves include actual canvas
    /// allocation and the scope allocation/transition headroom.
    pub vram_budget: u64,
    pub canvas_bytes: u64,
}

impl TileGpu {
    /// The atlas (pages sized by the device limit, count by `vram_budget`) and
    /// the pipeline drawing into the canvas.
    pub fn new(device: &wgpu::Device, vram_budget: u64) -> Self {
        let layout = super::atlas::PageLayout::new(device.limits().max_texture_dimension_2d, vram_budget);
        log::info!(
            "tile atlas: pages of {0}x{0}, up to {1} ({2} MiB)",
            layout.size,
            layout.max_pages,
            (u64::from(layout.size).pow(2) * super::atlas::TEXEL_BYTES * u64::from(layout.max_pages)) >> 20
        );
        Self {
            atlas: TileAtlas::new(layout),
            pipe: TilePipeline::new(device),
            direct: None,
            vram_budget,
            canvas_bytes: 0,
        }
    }
}

/// Per-instance data: the tile's screen rect in NDC of the callback rect and
/// its texel rect in the atlas page (normalised), `[x0, y0, x1, y1]` each.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    /// Destination, NDC (`y` up).
    pub dst: [f32; 4],
    /// Source rect in the page, UV (`v` down).
    pub uv: [f32; 4],
}

/// Display parameters for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Exposure in stops.
    pub exposure_ev: f32,
    /// Display gamma dial (1.0 = identity).
    pub gamma: f32,
    /// `transfer::mode::*`.
    pub channel_mode: u32,
    /// `transfer::tonemap::*`.
    pub tonemap: u32,
    /// Nearest (pixel-exact, magnified) or linear (minified) sampling.
    pub nearest: bool,
    /// The output is HDR: negatives are kept (else clipped, `transfer::canvas`).
    pub hdr: bool,
    /// The gain on the colour image's display light (`ocio::State::frame`).
    pub src: crate::transfer::Src,
}

/// One frame's draw: instances grouped by atlas page.
#[derive(Clone, Debug, Default)]
pub struct DrawList {
    /// All instances, contiguous per page.
    pub instances: Vec<Instance>,
    /// `(page, instance range)`.
    pub batches: Vec<(u32, Range<u32>)>,
}

/// Uniform block (16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    exposure_mult: f32,
    inv_gamma: f32,
    channel_mode: u32,
    nearest: u32,
    page_size: f32,
    tonemap: u32,
    hdr: u32,
    gain: f32,
}

fn colour_wgsl(ocio: Option<&Transform>) -> &'static str {
    // The gamma dial acts on the OCIO output: on the display's code values for an
    // encoded transform (OCIO's own display-gamma place), on linear light for a
    // linear one, which is then encoded like the built-in chain.
    match ocio {
        Some(t) if t.linear() => {
            "let o = ocio_transform(vec4(c, 1.0));
        let l = o.rgb * u.gain;
        d = vec3(srgb_oetf(apply_gamma(l.r)), srgb_oetf(apply_gamma(l.g)), srgb_oetf(apply_gamma(l.b)));"
        }
        Some(_) => {
            "let o = ocio_transform(vec4(c, 1.0));
        d = vec3(apply_gamma(o.r), apply_gamma(o.g), apply_gamma(o.b));"
        }
        None => {
            "let rgb = tone(c * (u.gain * u.exposure_mult), u.tonemap);
        d = vec3(
            srgb_oetf(apply_gamma(rgb.r)),
            srgb_oetf(apply_gamma(rgb.g)),
            srgb_oetf(apply_gamma(rgb.b)),
        );"
        }
    }
}

/// The WGSL of the tile pipeline; with `ocio` the colour image goes through the
/// transform's `ocio_transform`, else through the tone operator and the sRGB OETF.
pub(crate) fn shader_src(ocio: Option<&Transform>) -> String {
    let colour = colour_wgsl(ocio);
    format!(
        r#"{prelude}
struct Uniforms {{
    exposure_mult: f32,
    inv_gamma:     f32,
    channel_mode:  u32,
    nearest:       u32,
    page_size:     f32,
    tonemap:       u32,
    hdr:           u32,
    gain:          f32,
}}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var s_nearest: sampler;
@group(0) @binding(2) var s_linear:  sampler;
@group({page_group}) @binding(0) var page: texture_2d<f32>;

struct VsOut {{
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) uv_rect: vec4<f32>,
}};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) dst: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
) -> VsOut {{
    // Two triangles: (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1).
    var corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    out.pos = vec4(mix(dst.x, dst.z, c.x), mix(dst.y, dst.w, c.y), 0.0, 1.0);
    out.uv = vec2(mix(uv_rect.x, uv_rect.z, c.x), mix(uv_rect.y, uv_rect.w, c.y));
    out.uv_rect = uv_rect;
    return out;
}}

{oetf}
{eotf}
{alpha}
{tone}
fn apply_gamma(v: f32) -> f32 {{
    // Mirrored like `transfer::dial`.
    let g = pow(abs(v), u.inv_gamma);
    return select(g, -g, v < 0.0);
}}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {{
    // Clamp to valid texel centres, independently of rectangular capacity and
    // ragged reduced cells. The same rule keeps adjacent atlas slots out.
    let dimensions = vec2<f32>(textureDimensions(page));
    let first = (floor(in.uv_rect.xy * dimensions) + vec2(0.5)) / dimensions;
    let last = (ceil(in.uv_rect.zw * dimensions) - vec2(0.5)) / dimensions;
    let uv = clamp(in.uv, first, last);
    var raw: vec4<f32>;
    if u.nearest == 1u {{
        raw = textureSampleLevel(page, s_nearest, uv, 0.0);
    }} else {{
        raw = textureSampleLevel(page, s_linear, uv, 0.0);
    }}

    var out_rgb: vec3<f32>;
    var out_a: f32 = raw.a;
    if u.channel_mode == 0u {{
        // Display transforms are not linear: they take the colour, not premultiplied values.
        let c = unpremul(raw.rgb, raw.a);
        var d: vec3<f32>;
        {colour}
        out_rgb = premul(d, raw.a);
    }} else {{
        var val: f32;
        switch u.channel_mode {{
            case 1u: {{ val = raw.r; }}
            case 2u: {{ val = raw.g; }}
            case 3u: {{ val = raw.b; }}
            case 4u: {{ val = raw.a; out_a = 1.0; }}
            default: {{ val = dot(raw.rgb, vec3({lr}, {lg}, {lb})); }}
        }}
        val = srgb_oetf(apply_gamma(val * u.exposure_mult));
        out_rgb = vec3(val);
        if u.channel_mode != 4u {{ out_a = 1.0; }}
    }}
    // `transfer::canvas`: SDR cannot show a negative, and one would subtract in
    // egui's premultiplied blend under a partial alpha.
    if u.hdr == 0u {{
        out_rgb = max(out_rgb, vec3(0.0));
    }}
    return vec4(out_rgb, out_a);
}}
"#,
        prelude = ocio.map_or("", |t| t.prelude.as_str()),
        page_group = OCIO_HOST_GROUP_START,
        colour = colour,
        oetf = transfer::oetf_wgsl(),
        eotf = transfer::eotf_wgsl(),
        alpha = transfer::alpha_wgsl(),
        tone = transfer::tone_wgsl(),
        lr = LUMA_709[0],
        lg = LUMA_709[1],
        lb = LUMA_709[2],
    )
}

/// The WGSL of the scope display-transform pass: shades a linear source texture
/// (`Rgba32Float` or `Rgba16Float`) via `textureLoad` (`filterable: false`) into an
/// `Rgba16Float` scope source texture on the GPU.
pub(crate) fn scope_shader_src(ocio: Option<&Transform>) -> String {
    let colour = colour_wgsl(ocio);
    format!(
        r#"{prelude}
struct Uniforms {{
    exposure_mult: f32,
    inv_gamma:     f32,
    channel_mode:  u32,
    nearest:       u32,
    page_size:     f32,
    tonemap:       u32,
    hdr:           u32,
    gain:          f32,
}}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group({page_group}) @binding(0) var page: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    return vec4(x, y, 0.0, 1.0);
}}

{oetf}
{eotf}
{alpha}
{tone}
fn apply_gamma(v: f32) -> f32 {{
    let g = pow(abs(v), u.inv_gamma);
    return select(g, -g, v < 0.0);
}}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let raw = textureLoad(page, vec2<i32>(pos.xy), 0);
    var out_rgb: vec3<f32>;
    var out_a: f32 = raw.a;
    if u.channel_mode == 0u {{
        let c = unpremul(raw.rgb, raw.a);
        var d: vec3<f32>;
        {colour}
        out_rgb = premul(d, raw.a);
    }} else {{
        var val: f32;
        switch u.channel_mode {{
            case 1u: {{ val = raw.r; }}
            case 2u: {{ val = raw.g; }}
            case 3u: {{ val = raw.b; }}
            case 4u: {{ val = raw.a; out_a = 1.0; }}
            default: {{ val = dot(raw.rgb, vec3({lr}, {lg}, {lb})); }}
        }}
        val = srgb_oetf(apply_gamma(val * u.exposure_mult));
        out_rgb = vec3(val);
        if u.channel_mode != 4u {{ out_a = 1.0; }}
    }}
    if u.hdr == 0u {{
        out_rgb = max(out_rgb, vec3(0.0));
    }}
    return vec4(out_rgb, out_a);
}}
"#,
        prelude = ocio.map_or("", |t| t.prelude.as_str()),
        page_group = OCIO_HOST_GROUP_START,
        colour = colour,
        oetf = transfer::oetf_wgsl(),
        eotf = transfer::eotf_wgsl(),
        alpha = transfer::alpha_wgsl(),
        tone = transfer::tone_wgsl(),
        lr = LUMA_709[0],
        lg = LUMA_709[1],
        lb = LUMA_709[2],
    )
}


/// The render pipeline and its per-frame buffers.
pub struct TilePipeline {
    pipeline: wgpu::RenderPipeline,
    /// Layout of group 0 (uniforms + samplers), kept for pipeline rebuilds.
    globals_bgl: wgpu::BindGroupLayout,
    /// Layout of an atlas page bind group (group `OCIO_HOST_GROUP_START`).
    pub page_bgl: wgpu::BindGroupLayout,
    globals: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    instances: wgpu::Buffer,
    capacity: u64,
    /// The OCIO transform the pipeline was built with; `None` = the built-in display
    /// transform.
    ocio: Option<OcioGpu>,
}

/// The GPU side of one OCIO transform: its groups 1/2 and what moves its dynamic exposure.
struct OcioGpu {
    /// The transform's key (a new key rebuilds the pipeline).
    key: String,
    resources: GpuOcioResources,
    /// The transform's shader code, whose scalar uniforms `update_scalars` re-uploads; its
    /// exposure uniforms ([`Transform::exposure`]) are set here per frame.
    code: vfx_ocio::GpuShaderCode,
    /// The exposure uniforms of `code`.
    exposure: Vec<String>,
    /// The exposure (stops) last uploaded; `None` before the first frame.
    uploaded: Option<f32>,
}

impl OcioGpu {
    /// Move the dynamic exposure to `ev` stops (OCIO's `linearCC`, in `scene_linear`):
    /// set its uniforms and re-upload the scalar block, only when `ev` changed.
    fn set_exposure(&mut self, queue: &wgpu::Queue, ev: f32) {
        if self.exposure.is_empty() || self.uploaded.is_some_and(|u| u.to_bits() == ev.to_bits()) {
            return;
        }
        for name in &self.exposure {
            if !self.code.set_uniform_value(name, vfx_ocio::GpuUniformValue::Float(ev)) {
                log::error!("OCIO: the shader has no uniform {name} for the exposure");
            }
        }
        if self.resources.update_scalars(queue, &self.code) {
            self.uploaded = Some(ev);
        } else {
            log::error!("OCIO: the exposure uniforms could not be uploaded");
        }
    }
}

const INSTANCE_BYTES: u64 = std::mem::size_of::<Instance>() as u64;

/// The tile render pipeline into the canvas: the shader of [`shader_src`] with the
/// optional OCIO `(transform, resources)`, laid out as group 0 = globals, 1/2 = OCIO
/// (absent slots when the transform binds nothing there), 3 = the atlas page.
fn build_pipeline(
    device: &wgpu::Device,
    globals_bgl: &wgpu::BindGroupLayout,
    page_bgl: &wgpu::BindGroupLayout,
    ocio: Option<(&Transform, &GpuOcioResources)>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("exr_view.tiles"),
        source: wgpu::ShaderSource::Wgsl(shader_src(ocio.map(|(t, _)| t)).into()),
    });
    let res = ocio.map(|(_, r)| r);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("exr_view.tiles"),
        bind_group_layouts: &[
            Some(globals_bgl),
            res.and_then(|r| r.bind_group_layout(OCIO_UNIFORM_GROUP)),
            res.and_then(|r| r.bind_group_layout(OCIO_TEXTURE_GROUP)),
            Some(page_bgl),
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("exr_view.tiles"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: INSTANCE_BYTES,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::present::CANVAS_FORMAT,
                // egui-wgpu 0.36's own blend (premultiplied alpha): it keeps the
                // canvas as associated light under any overlap with other draws.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl TilePipeline {
    /// Build the pipeline into the canvas, with the built-in display transform.
    pub fn new(device: &wgpu::Device) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exr_view.tiles.globals"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)),
                entry(2, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let page_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exr_view.tiles.page"),
            entries: &[entry(
                0,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            )],
        });
        let sampler = |filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("exr_view.tiles.sampler"),
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        let s_nearest = sampler(wgpu::FilterMode::Nearest);
        let s_linear = sampler(wgpu::FilterMode::Linear);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exr_view.tiles.uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exr_view.tiles.globals"),
            layout: &globals_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&s_nearest) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&s_linear) },
            ],
        });
        let pipeline = build_pipeline(device, &globals_bgl, &page_bgl, None);
        let capacity = 256;
        let instances = instance_buffer(device, capacity);
        Self { pipeline, globals_bgl, page_bgl, globals, uniforms, instances, capacity, ocio: None }
    }

    /// Draw the colour image through `t` (`None` = the built-in transform),
    /// rebuilding the pipeline and OCIO's bind groups when it is not the one in use.
    /// A transform with LUTs on a device that cannot filter `Rgba32Float` is refused
    /// (`ocio::State` refuses it first; this keeps the pipeline valid regardless).
    pub fn set_colour(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: Option<&Transform>) -> Result<(), String> {
        if t.map(|t| t.key.as_str()) == self.ocio.as_ref().map(|o| o.key.as_str()) {
            return Ok(());
        }
        let ocio = match t {
            Some(t) => {
                if t.has_luts() && !device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
                    return Err("the OCIO transform samples LUT textures, which this GPU cannot filter".to_owned());
                }
                Some(OcioGpu {
                    key: t.key.clone(),
                    resources: GpuOcioResources::build_for(device, queue, &t.code, wgpu::ShaderStages::FRAGMENT),
                    code: t.code.clone(),
                    exposure: t.exposure.clone(),
                    uploaded: None,
                })
            }
            None => None,
        };
        let with = t.zip(ocio.as_ref()).map(|(t, o)| (t, &o.resources));
        self.pipeline = build_pipeline(device, &self.globals_bgl, &self.page_bgl, with);
        self.ocio = ocio;
        Ok(())
    }

    /// Upload this frame's uniforms and instances (paint-callback `prepare`).
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, look: Look, page_size: u32, list: &DrawList) {
        let u = Uniforms {
            exposure_mult: 2.0_f32.powf(look.exposure_ev),
            // The same floor `transfer::display_pixel` applies.
            inv_gamma: 1.0 / look.gamma.max(1e-3),
            channel_mode: look.channel_mode,
            nearest: u32::from(look.nearest),
            page_size: page_size as f32,
            tonemap: look.tonemap,
            hdr: u32::from(look.hdr),
            gain: look.src.gain,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        if let Some(ocio) = &mut self.ocio {
            ocio.set_exposure(queue, look.exposure_ev);
        }
        let need = list.instances.len() as u64;
        if need > self.capacity {
            self.capacity = need.next_power_of_two();
            self.instances = instance_buffer(device, self.capacity);
        }
        if need > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&list.instances));
        }
    }

    /// Record the draws (paint-callback `paint`).
    pub fn paint(&self, pass: &mut wgpu::RenderPass<'static>, pages: &[super::atlas::Page], list: &DrawList) {
        if list.instances.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.globals, &[]);
        if let Some(ocio) = &self.ocio {
            for group in [OCIO_UNIFORM_GROUP, OCIO_TEXTURE_GROUP] {
                if let Some(bg) = ocio.resources.bind_group(group) {
                    pass.set_bind_group(group, bg, &[]);
                }
            }
        }
        pass.set_vertex_buffer(0, self.instances.slice(..));
        for (page, range) in &list.batches {
            let Some(p) = pages.get(*page as usize) else { continue };
            pass.set_bind_group(OCIO_HOST_GROUP_START, &p.bind, &[]);
            pass.draw(0..6, range.clone());
        }
    }
}

fn instance_buffer(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exr_view.tiles.instances"),
        size: capacity * INSTANCE_BYTES,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The paint callback of one frame: owns its draw list and look.
pub struct TileCallback {
    /// What to draw.
    pub list: DrawList,
    /// How to draw it.
    pub look: Look,
    /// Bind the single direct reduced image rather than atlas pages.
    pub direct: bool,
}

impl egui_wgpu::CallbackTrait for TileCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = res.get_mut::<TileGpu>() {
            let size = gpu.atlas.layout.size;
            gpu.pipe.prepare(device, queue, self.look, size, &self.list);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        res: &egui_wgpu::CallbackResources,
    ) {
        if let Some(gpu) = res.get::<TileGpu>() {
            let pages = if self.direct {
                gpu.direct.as_ref().map_or(&[][..], |d| d.pages())
            } else {
                gpu.atlas.pages()
            };
            gpu.pipe.paint(pass, pages, &self.list);
        }
    }
}

/// GPU display-transform pass for scopes in `Display values` mode: shades a linear
/// source texture (`Rgba32Float` or `Rgba16Float`) into an `Rgba16Float` scope source
/// texture without CPU shading or PCIe texture re-uploads.
pub(crate) struct ScopeShaper {
    pipeline: wgpu::RenderPipeline,
    globals_bgl: wgpu::BindGroupLayout,
    src_bgl: wgpu::BindGroupLayout,
    globals: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    ocio: Option<OcioGpu>,
}

fn build_scope_pipeline(
    device: &wgpu::Device,
    globals_bgl: &wgpu::BindGroupLayout,
    src_bgl: &wgpu::BindGroupLayout,
    ocio: Option<(&Transform, &GpuOcioResources)>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("exr_view.scope_shaper"),
        source: wgpu::ShaderSource::Wgsl(scope_shader_src(ocio.map(|(t, _)| t)).into()),
    });
    let res = ocio.map(|(_, r)| r);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("exr_view.scope_shaper"),
        bind_group_layouts: &[
            Some(globals_bgl),
            res.and_then(|r| r.bind_group_layout(OCIO_UNIFORM_GROUP)),
            res.and_then(|r| r.bind_group_layout(OCIO_TEXTURE_GROUP)),
            Some(src_bgl),
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("exr_view.scope_shaper"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::present::CANVAS_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl ScopeShaper {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exr_view.scope_shaper.globals"),
            entries: &[entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            )],
        });
        let src_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exr_view.scope_shaper.src"),
            entries: &[entry(
                0,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            )],
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exr_view.scope_shaper.uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exr_view.scope_shaper.globals"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let pipeline = build_scope_pipeline(device, &globals_bgl, &src_bgl, None);
        Self { pipeline, globals_bgl, src_bgl, globals, uniforms, ocio: None }
    }

    fn set_colour(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: Option<&Transform>) -> Result<(), String> {
        if t.map(|t| t.key.as_str()) == self.ocio.as_ref().map(|o| o.key.as_str()) {
            return Ok(());
        }
        let ocio = match t {
            Some(t) => {
                if t.has_luts() && !device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
                    return Err("the OCIO transform samples LUT textures, which this GPU cannot filter".to_owned());
                }
                Some(OcioGpu {
                    key: t.key.clone(),
                    resources: GpuOcioResources::build_for(device, queue, &t.code, wgpu::ShaderStages::FRAGMENT),
                    code: t.code.clone(),
                    exposure: t.exposure.clone(),
                    uploaded: None,
                })
            }
            None => None,
        };
        let with = t.zip(ocio.as_ref()).map(|(t, o)| (t, &o.resources));
        self.pipeline = build_scope_pipeline(device, &self.globals_bgl, &self.src_bgl, with);
        self.ocio = ocio;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shade(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        w: u32,
        h: u32,
        look: &crate::export::Look,
    ) -> Result<(), String> {
        let t = look.ocio.as_deref().filter(|_| look.mode == transfer::mode::RGBA);
        self.set_colour(device, queue, t)?;
        let u = Uniforms {
            exposure_mult: 2.0_f32.powf(look.exposure),
            inv_gamma: 1.0 / look.gamma.max(1e-3),
            channel_mode: look.mode,
            nearest: 1,
            page_size: 1.0,
            tonemap: look.tonemap,
            hdr: 1,
            gain: look.src.gain,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        if let Some(ocio) = &mut self.ocio {
            ocio.set_exposure(queue, look.exposure);
        }
        let src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exr_view.scope_shaper.src"),
            layout: &self.src_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exr_view.scope_shaper"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exr_view.scope_shaper"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, w, h);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.globals, &[]);
            if let Some(ocio) = &self.ocio {
                for group in [OCIO_UNIFORM_GROUP, OCIO_TEXTURE_GROUP] {
                    if let Some(bg) = ocio.resources.bind_group(group) {
                        pass.set_bind_group(group, bg, &[]);
                    }
                }
            }
            pass.set_bind_group(OCIO_HOST_GROUP_START, &src_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        crate::scopes::wgsl_probe::validate("tiles", &shader_src(None));
        crate::scopes::wgsl_probe::validate("scope_shaper", &scope_shader_src(None));
    }

    /// With an OCIO transform the shader takes its prelude verbatim, calls it in
    /// colour mode instead of the tone operator, and still validates (no name
    /// clash between the prelude and this shader; the page stays at group 3). An
    /// encoded transform stores the display's codes; a linear one encodes light.
    #[test]
    fn shader_with_ocio_validates_and_calls_it() {
        let o = crate::ocio::Ocio::load("ocio://studio-config-latest").unwrap();
        let sel = crate::ocio::Sel::default();
        let space = o.scene();
        for linear in [false, true] {
            let t = o.transform(&o.resolve(&sel, &space, linear).unwrap(), linear).unwrap();
            let src = shader_src(Some(&t));
            crate::scopes::wgsl_probe::validate("tiles + OCIO", &src);
            assert!(src.contains(&t.prelude));
            assert!(src.contains("let o = ocio_transform(vec4(c, 1.0));"));
            assert!(!src.contains("tone(c *"), "the tone operator is not in the OCIO chain");
            let encodes = src.contains("srgb_oetf(apply_gamma(l.r))");
            assert_eq!(encodes, linear, "linear = {linear}");
            assert!(src.contains(&format!("@group({OCIO_HOST_GROUP_START}) @binding(0) var page")));
        }
        assert!(shader_src(None).contains("tone(c *"));
    }

    /// Every display/view of every OCIO built-in config, SDR (encoded) and HDR
    /// (linear), gives a prelude that splices into this shader without a name
    /// clash (its helper names vs `tone`, `apply_gamma`, `srgb_oetf`, ...): each
    /// distinct prelude is validated by naga.
    #[test]
    fn every_builtin_view_splices_into_the_shader() {
        let mut seen = std::collections::HashSet::new();
        for e in vfx_ocio::builtin::embedded::ocio_builtins() {
            let o = crate::ocio::Ocio::load(&e.uri()).unwrap();
            for hdr in [false, true] {
                for d in o.displays(hdr) {
                    // Every view, whatever the viewing rules allow for one colour space.
                    for v in o.views(d, "", hdr) {
                        let names = crate::ocio::Names { input: "scene_linear".into(), display: d.into(), view: v.into(), look: String::new() };
                        let t = o.transform(&names, hdr).unwrap();
                        if seen.insert(t.prelude.clone()) {
                            let what = format!("{} {d} / {v} hdr={hdr}", e.name);
                            crate::scopes::wgsl_probe::validate(&what, &shader_src(Some(&t)));
                        }
                    }
                }
            }
        }
        eprintln!("{} distinct preludes validated", seen.len());
        assert!(seen.len() > 5);
    }

    /// The shader ends with the canvas value itself: no sRGB-target decode, no
    /// dither (both are the present pass's now).
    #[test]
    fn output_is_the_canvas_value() {
        let src = shader_src(None);
        assert!(src.contains("return vec4(out_rgb, out_a);"));
        assert!(src.contains("out_rgb = max(out_rgb, vec3(0.0));"), "the SDR clip of transfer::canvas");
        assert!(!src.contains("linear_from_gamma") && !src.contains("interleaved_gradient_noise"));
    }

    /// The display transform must be the colour pass's, verbatim: same OETF,
    /// same Rec.709 weights, same channel-mode numbering as `transfer::mode`.
    #[test]
    fn shader_carries_the_display_transform() {
        let src = shader_src(None);
        assert!(src.contains(&transfer::oetf_wgsl()), "OETF not spliced in verbatim");
        assert!(src.contains(&transfer::tone_wgsl()), "tone operators not spliced in verbatim");
        assert!(src.contains(transfer::alpha_wgsl()), "the alpha twins not spliced in verbatim");
        assert!(src.contains(&format!("vec3({}, {}, {})", LUMA_709[0], LUMA_709[1], LUMA_709[2])));
        assert_eq!(transfer::mode::RGBA, 0);
        for (mode, arm) in [
            (transfer::mode::R, "case 1u: { val = raw.r; }"),
            (transfer::mode::G, "case 2u: { val = raw.g; }"),
            (transfer::mode::B, "case 3u: { val = raw.b; }"),
            (transfer::mode::A, "case 4u: { val = raw.a; out_a = 1.0; }"),
        ] {
            assert!(src.contains(arm), "mode {mode}: {arm}");
            assert!(arm.contains(&format!("case {mode}u")));
        }
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0, "uniform block stride");
    }
}
