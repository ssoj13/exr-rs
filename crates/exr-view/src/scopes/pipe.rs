//! Shared wgpu helpers for analysis compute panels.
//!
//! Compute-pipeline helpers for the GPU scopes; depends on `wgpu` directly
//! (the same wgpu 29 that egui_wgpu re-exports).

use super::{
    make_output_texture, storage_ro_entry, storage_rw_entry, storage_tex_write_entry,
    texture_2d_entry, uniform_entry,
};

/// Uniform `{ src_w, src_h, out_w, out_h }` — 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PanelParams {
    pub src_w: u32,
    pub src_h: u32,
    pub out_w: u32,
    pub out_h: u32,
}

pub fn write_panel_params(queue: &wgpu::Queue, buf: &wgpu::Buffer, p: PanelParams) {
    queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Resizeable panel output texture (physical bucket size).
pub struct PanelOutput {
    /// Keeps the GPU texture alive while the view is registered with egui.
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl PanelOutput {
    pub fn alloc(device: &wgpu::Device, label: &str, bucket_w: u32, bucket_h: u32) -> Self {
        let (texture, view) = make_output_texture(device, label, bucket_w, bucket_h);
        Self { texture, view }
    }
}

/// Bind group layout: `{ bins rw }` — clear / reduce.
pub fn bins_only_bgl(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[storage_rw_entry(0)],
    })
}

/// `{ src_tex, bins rw, params }` — scatter.
pub fn scatter_bgl(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[texture_2d_entry(0), storage_rw_entry(1), uniform_entry(2)],
    })
}

/// `{ bins ro, params, out_tex }` — draw.
pub fn draw_bgl(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            storage_ro_entry(0),
            uniform_entry(1),
            storage_tex_write_entry(2),
        ],
    })
}

pub fn bins_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    bins: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: bins.as_entire_binding(),
        }],
    })
}

pub fn scatter_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    src: &wgpu::TextureView,
    bins: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

pub fn draw_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    bins: &wgpu::Buffer,
    params: &wgpu::Buffer,
    out: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(out),
            },
        ],
    })
}
