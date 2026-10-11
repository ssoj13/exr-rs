//! Resident display: one original f32 upload, one GPU-selected pyramid level,
//! and the same half-float colour pipeline as the tile atlas.
use super::atlas::Page;
use super::grid::Grid;
use super::render::{DrawList, Instance};
use super::view::{Placement, SourceDesc};
use exr_gpu::UploadLayout;

pub const MAX_DIM: u32 = 8192;

/// Exact integer mean within the checked source domain (`main`), rounded to
/// f32 and then half like the CPU reference; sources outside it (most float
/// renders) use the same integer sum with a per-cell window (`main_wide`).
pub const REDUCE: &str = include_str!("precision.wgsl");

struct Reducer {
    /// Exact fixed-point mean (`main`), for sources inside the checked domain.
    pipeline: wgpu::ComputePipeline,
    /// Per-cell-window integer mean (`main_wide`), for every other source:
    /// falling back to the CPU tile path instead made each float frame of a
    /// sequence wait for rayon tile production.
    pipeline_wide: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}
impl Reducer {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exr_view.direct.reduce"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exr_view.direct.reduce"),
            source: wgpu::ShaderSource::Wgsl(REDUCE.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exr_view.direct.reduce"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_for = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("exr_view.direct.reduce"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = pipeline_for("main");
        let pipeline_wide = pipeline_for("main_wide");
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exr_view.direct.params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            pipeline_wide,
            layout,
            params,
        }
    }
}

struct Original {
    tex: wgpu::Texture,
    cap: (u32, u32),
    bytes: u64,
}
struct Reduced {
    page: Page,
    cap: (u32, u32),
    bytes: u64,
}

/// The original upload and one selected level; no recursive half pyramid.
pub struct DirectGpu {
    reducer: Reducer,
    retired: super::retire::RetiredTextures,
    original: Option<Original>,
    /// The source the original holds (it may run ahead of `desc`: the scopes
    /// can upload a new frame before the viewport reduces it).
    original_desc: Option<SourceDesc>,
    reduced: Option<Reduced>,
    pub desc: Option<SourceDesc>,
    profile_desc: Option<SourceDesc>,
    profile: Option<[i32; 4]>,
    /// Profiles resolved (one per new source).
    #[cfg(test)]
    pub profiles: usize,
    /// Of those, CPU rescans of the pixels (no profile known from the frame).
    #[cfg(test)]
    pub scans: usize,
    extent: (u32, u32),
    level: u32,
    place: Option<Placement>,
    #[cfg(test)]
    pub uploads: usize,
    #[cfg(test)]
    pub dispatches: usize,
}
impl DirectGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            reducer: Reducer::new(device),
            retired: Default::default(),
            original: None,
            original_desc: None,
            reduced: None,
            desc: None,
            profile_desc: None,
            profile: None,
            #[cfg(test)]
            profiles: 0,
            #[cfg(test)]
            scans: 0,
            extent: (0, 0),
            level: 0,
            place: None,
            #[cfg(test)]
            uploads: 0,
            #[cfg(test)]
            dispatches: 0,
        }
    }
    pub fn bytes(&self) -> u64 {
        32 + self.retired.bytes()
            + self.original.as_ref().map_or(0, |t| t.bytes)
            + self.reduced.as_ref().map_or(0, |t| t.bytes)
    }
    pub fn pages(&self) -> &[Page] {
        self.reduced
            .as_ref()
            .map_or(&[], |t| std::slice::from_ref(&t.page))
    }
    /// Only the reduced image is needed to hold a previous frame.
    pub fn retire_original(&mut self, queue: &wgpu::Queue) {
        if let Some(t) = self.original.take() {
            self.retired.retire(queue, t.tex, t.bytes);
        }
        self.original_desc = None;
    }
    pub fn retire_display(&mut self, queue: &wgpu::Queue) {
        self.retire_original(queue);
        if let Some(t) = self.reduced.take() {
            self.retired.retire(queue, t.page.tex, t.bytes);
        }
        self.desc = None;
        self.place = None;
    }
    pub fn reap(&mut self) {
        self.retired.reap();
    }
    pub fn drawable(&self) -> bool {
        self.reduced.is_some()
    }
    /// Whether the last prepared source used the exact fixed-point kernel
    /// (`main`) rather than the per-cell window (`main_wide`).
    pub fn exact(&self) -> bool {
        self.profile.is_some()
    }

    /// Prepare a drawable generation without dropping the old reduced image.
    /// `known` is the source's precision profile when the caller has it (a
    /// resident frame's channel spans); `None` rescans `rgba` once per source.
    /// `allowance` excludes canvas, scopes and the still-held tile atlas.
    /// The peak includes old allocations until their replacements are ready.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page_bgl: &wgpu::BindGroupLayout,
        desc: &SourceDesc,
        rgba: &[f32],
        known: Option<Option<[i32; 4]>>,
        extent: (u32, u32),
        place: Placement,
        allowance: u64,
    ) -> Result<bool, String> {
        let _lap = (self.desc.as_ref() != Some(desc)).then(|| crate::timing::Lap::start("direct prepare"));
        self.retired.reap();
        let limit = device.limits().max_texture_dimension_2d.min(MAX_DIM);
        let input =
            UploadLayout::compute(extent.0, extent.1, 16, limit).map_err(|e| e.to_string())?;
        if rgba.len() != input.cpu_size / 4 {
            return Err("direct source length differs from its checked layout".into());
        }
        if self.profile_desc.as_ref() != Some(desc) {
            self.profile = match known {
                Some(profile) => profile,
                None => {
                    let mut lap = crate::timing::Lap::start("direct profile");
                    lap.bytes(input.cpu_size as u64);
                    #[cfg(test)]
                    {
                        self.scans += 1;
                    }
                    super::precision::profile(rgba)
                }
            };
            self.profile_desc = Some(desc.clone());
            #[cfg(test)]
            {
                self.profiles += 1;
            }
            if self.profile.is_none() {
                log::debug!(
                    "resident display uses per-cell windows: source exceeds the checked fixed-point domain"
                );
            }
        }
        // `None` = outside the exact domain: `main_wide`, which picks `low` per cell.
        let profile = self.profile.unwrap_or([0; 4]);
        let grid = Grid::new(extent.0, extent.1);
        let level = grid.pick_level(place.zoom);
        let size = grid.level_size(level);
        let output = UploadLayout::compute(size.0, size.1, 8, limit).map_err(|e| e.to_string())?;
        let input_fits = self.original.as_ref().is_some_and(|t| fits(t.cap, extent));
        let output_fits = self.reduced.as_ref().is_some_and(|t| fits(t.cap, size));
        let required = self
            .bytes()
            .checked_add(if input_fits { 0 } else { input.byte_size })
            .and_then(|n| n.checked_add(if output_fits { 0 } else { output.byte_size }));
        if required.is_none_or(|bytes| bytes > allowance) {
            return Ok(false);
        }
        if !self.upload(device, queue, desc, rgba, extent, allowance)? {
            return Ok(false);
        }
        let recalculate = self.desc.as_ref() != Some(desc) || self.level != level || !output_fits;

        // Create the replacement while the old reduced image is still intact.
        let next_reduced = if output_fits {
            None
        } else {
            let _alloc = crate::timing::Lap::start("direct alloc");
            let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let tex = texture(
                device,
                &output,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let view = tex.create_view(&Default::default());
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("exr_view.direct.page"),
                layout: page_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            if let Some(error) = pollster::block_on(validation.pop()).or(pollster::block_on(oom.pop())) {
                return Err(format!("direct texture allocation: {error}"));
            }
            Some(Reduced {
                page: Page { tex, bind },
                cap: size,
                bytes: output.byte_size,
            })
        };
        let original = self.original.as_ref().expect("checked direct original");
        let reduced = next_reduced
            .as_ref()
            .or(self.reduced.as_ref())
            .expect("checked direct output");
        if recalculate {
            let _lap = crate::timing::Lap::start("direct reduce");
            let source = original.tex.create_view(&Default::default());
            let destination = reduced.page.tex.create_view(&Default::default());
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("exr_view.direct.reduce"),
                layout: &self.reducer.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&destination),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.reducer.params.as_entire_binding(),
                    },
                ],
            });
            queue.write_buffer(
                &self.reducer.params,
                0,
                bytemuck::cast_slice(&[
                    extent.0,
                    extent.1,
                    1u32 << level,
                    0,
                    profile[0] as u32,
                    profile[1] as u32,
                    profile[2] as u32,
                    profile[3] as u32,
                ]),
            );
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(if self.profile.is_some() { &self.reducer.pipeline } else { &self.reducer.pipeline_wide });
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
            }
            queue.submit([enc.finish()]);
            #[cfg(test)]
            {
                self.dispatches += 1;
            }
        }
        if let Some(t) = next_reduced
            && let Some(old) = self.reduced.replace(t)
        {
            self.retired.retire(queue, old.page.tex, old.bytes);
        }
        self.desc = Some(desc.clone());
        self.extent = extent;
        self.level = level;
        self.place = Some(place);
        Ok(true)
    }

    /// Make the original texture hold `desc`'s level-0 pixels `rgba`; a no-op
    /// when it already does. The display ([`Self::prepare`]) and the scopes
    /// (`ImageBackend::ensure_linear`) share it, so a frame is uploaded once
    /// whichever of them asks first. `Ok(false)` when a new texture would
    /// exceed `allowance`.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &SourceDesc,
        rgba: &[f32],
        extent: (u32, u32),
        allowance: u64,
    ) -> Result<bool, String> {
        let limit = device.limits().max_texture_dimension_2d.min(MAX_DIM);
        let input = UploadLayout::compute(extent.0, extent.1, 16, limit).map_err(|e| e.to_string())?;
        if rgba.len() != input.cpu_size / 4 {
            return Err("direct source length differs from its checked layout".into());
        }
        let input_fits = self.original.as_ref().is_some_and(|t| fits(t.cap, extent));
        if input_fits && self.original_desc.as_ref() == Some(desc) {
            return Ok(true);
        }
        if !input_fits {
            if self.bytes().checked_add(input.byte_size).is_none_or(|bytes| bytes > allowance) {
                return Ok(false);
            }
            let _alloc = crate::timing::Lap::start("direct alloc");
            let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let tex = texture(
                device,
                &input,
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | if cfg!(test) { wgpu::TextureUsages::COPY_SRC } else { wgpu::TextureUsages::empty() },
            );
            if let Some(error) = pollster::block_on(validation.pop()).or(pollster::block_on(oom.pop())) {
                return Err(format!("direct texture allocation: {error}"));
            }
            self.retire_original(queue);
            self.original = Some(Original { tex, cap: extent, bytes: input.byte_size });
        }
        let original = self.original.as_ref().ok_or("direct original is absent")?;
        let mut lap = crate::timing::Lap::start("direct upload");
        lap.bytes(input.cpu_size as u64);
        queue.write_texture(
            original.tex.as_image_copy(),
            bytemuck::cast_slice(rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(input.row_bytes),
                rows_per_image: Some(extent.1),
            },
            wgpu::Extent3d {
                width: extent.0,
                height: extent.1,
                depth_or_array_layers: 1,
            },
        );
        drop(lap);
        #[cfg(test)]
        {
            self.uploads += 1;
        }
        self.original_desc = Some(desc.clone());
        Ok(true)
    }

    /// The original texture's view while it holds `desc`'s pixels (the
    /// scopes' level-0 source).
    pub fn original_view(&self, desc: &SourceDesc) -> Option<wgpu::TextureView> {
        let original = self.original.as_ref().filter(|_| self.original_desc.as_ref() == Some(desc))?;
        Some(original.tex.create_view(&Default::default()))
    }

    #[cfg(test)]
    pub fn original_texture(&self) -> Option<&wgpu::Texture> {
        self.original.as_ref().map(|t| &t.tex)
    }

    /// Preserve level-0 geometry; ragged cells are not stretched to ceil extent.
    pub fn list(&self, current: Placement, held: bool) -> DrawList {
        let (Some(reduced), Some(old)) = (&self.reduced, self.place) else {
            return DrawList::default();
        };
        let place = if held {
            Placement {
                clip: current.clip,
                ..old
            }
        } else {
            current
        };
        let scale = (1u32 << self.level) as f32;
        let uv = [
            0.0,
            0.0,
            self.extent.0 as f32 / scale / reduced.cap.0 as f32,
            self.extent.1 as f32 / scale / reduced.cap.1 as f32,
        ];
        DrawList {
            instances: vec![Instance {
                dst: place.ndc(0, 0, self.extent.0, self.extent.1),
                uv,
            }],
            batches: vec![(0, 0..1)],
        }
    }
}
fn fits(cap: (u32, u32), need: (u32, u32)) -> bool {
    cap.0 >= need.0
        && cap.1 >= need.1
        && u64::from(cap.0) * u64::from(cap.1) <= u64::from(need.0) * u64::from(need.1) * 4
}
fn texture(
    device: &wgpu::Device,
    layout: &UploadLayout,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("exr_view.direct"),
        size: wgpu::Extent3d {
            width: layout.width,
            height: layout.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
#[cfg(test)]
mod tests {
    #[test]
    fn shader_validates() {
        crate::scopes::wgsl_probe::validate("direct reduction", super::REDUCE);
    }
}
