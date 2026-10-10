//! wgpu 30 GPU context: instance / adapter / device / queue + a window surface.
//!
//! No eframe: the viewer owns its winit surface + wgpu device directly, which is
//! what lets it pick the surface's format AND colour space ([`Output`]: SDR 8/10-bit,
//! HDR10, scRGB); eframe / egui-wgpu choose an 8-bit SDR surface themselves.
//! wgpu is mandatory; init failure is a hard error (no CPU path).

use std::sync::Arc;

use anyhow::{Context as _, Result};
use winit::window::Window;

use crate::present::Output;

/// Owns the wgpu quartet plus the per-window surface and its configuration.
///
/// `Arc<Window>` is held so the surface (which borrows the window) stays valid
/// for the surface's lifetime, and so the app can request redraws.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    /// What the surface is configured for.
    pub output: Output,
}

impl GpuContext {
    /// Build the wgpu context for `window`.
    ///
    /// Picks a high-performance adapter compatible with the window surface,
    /// requests a device with the adapter's limits, and configures the surface for
    /// `Output::Sdr8` (every surface has it; [`Self::set_output`] changes it).
    /// Returns a hard error on any failure; there is no software fallback.
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let mut inst_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        inst_desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(inst_desc);

        // The surface borrows the window; `Arc<Window>` upgrades the borrow to
        // `'static` so `Surface<'static>` can live in this struct.
        let surface = instance
            .create_surface(window.clone())
            .context("wgpu: create_surface failed")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false, // wgpu 30
        }))
        .context("wgpu: no suitable GPU adapter found")?;

        // User-meaningful milestone (info): the GPU adapter wgpu selected.
        let info = adapter.get_info();
        log::info!(
            "GPU adapter: {} ({:?}) backend={:?} device_type={:?}",
            info.name,
            info.vendor,
            info.backend,
            info.device_type
        );
        // Full capability report (limits / features / formats) via `wgpu-info`,
        // so the log shows exactly what the GPU supports (HDR/f16 framebuffer,
        // storage formats, workgroup sizes) instead of guessing.
        log::debug!(
            "GPU capabilities:\n{}",
            wgpu_info::adapter_summary(&adapter)
        );
        // Native per-backend VRAM budget (DXGI / Vulkan heaps / Metal). `None` on
        // backends that can't report it — the cache then falls back to its RAM
        // heuristic. Logged so the real number is visible live for verification.
        match wgpu_info::vram_budget_bytes(&adapter) {
            Some(b) => log::info!("GPU VRAM budget: {} MiB ({} bytes)", b >> 20, b),
            None => log::info!(
                "GPU VRAM budget: unavailable for backend {:?}",
                info.backend
            ),
        }

        // Build the device through `wgpu-info` so we use the adapter to its
        // fullest instead of `Limits::default()` (which caps max texture size at
        // 8192 and would reject large EXRs / 4K+ frames). `request_max_device`
        // requests `adapter.limits()` (the real maximums) plus the subset of the
        // features we ask for that the GPU actually supports.
        //
        // One optional feature: `FLOAT32_FILTERABLE`, granted when the adapter has
        // it (wgpu-info intersects the request with the adapter). Only an OCIO
        // transform with LUTs needs it (`Rgba32Float` LUT textures sampled with a
        // filtering sampler); every such transform checks `device.features()` and
        // is refused with a message without it (`ocio::State`,
        // `TilePipeline::set_colour`). Nothing else may assume it (the old
        // colour pass requested it and then used it unconditionally, a validation
        // panic on adapters without it); the tiles are `Rgba16Float`, filterable
        // everywhere, and the scopes read with `textureLoad`.
        let (device, queue) =
            wgpu_info::request_max_device_blocking(&adapter, wgpu::Features::FLOAT32_FILTERABLE)
                .context("wgpu: request_device (max limits via wgpu-info) failed")?;
        log::info!(
            "FLOAT32_FILTERABLE: {} (OCIO LUT transforms need it)",
            device.features().contains(wgpu::Features::FLOAT32_FILTERABLE)
        );
        log::info!(
            "device: adapter max limits; max_texture_dimension_2d={}",
            device.limits().max_texture_dimension_2d
        );

        // wgpu's DEFAULT uncaptured-error handler is `panic!`. Anything that slips
        // past our own validation (a driver quirk, an out-of-range copy) would kill
        // the viewer mid-frame. Log it loudly instead and keep the window alive so
        // the user can save/close; every error we can predict is already turned
        // into a typed failure before the call that would raise it.
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            log::error!("wgpu uncaptured error (frame may be wrong): {e}");
        }));

        let caps = surface.get_capabilities(&adapter);
        let (format, color_space) = Output::Sdr8
            .surface(&caps)
            .context("wgpu: the surface offers no 8-bit sRGB format")?;
        log::info!("surface formats: {:?}", caps.format_capabilities);
        let alpha_mode = caps
            .alpha_modes
            .first()
            .copied()
            .context("wgpu: the surface offers no alpha modes")?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            color_space,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Lifecycle milestone (info): the surface format + size we settled on.
        log::info!("surface configured: format={format:?} {color_space:?} {width}x{height} present_mode=Fifo");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            config,
            output: Output::Sdr8,
        })
    }

    /// The outputs this surface can be configured for, menu order.
    pub fn outputs(&self) -> Vec<Output> {
        let caps = self.surface.get_capabilities(&self.adapter);
        Output::ALL.into_iter().filter(|o| o.surface(&caps).is_some()).collect()
    }

    /// Reconfigure the surface for `output`. An output the surface does not offer
    /// is an error and leaves the surface as it was.
    pub fn set_output(&mut self, output: Output) -> Result<(), String> {
        let caps = self.surface.get_capabilities(&self.adapter);
        let (format, color_space) = output
            .surface(&caps)
            .ok_or_else(|| format!("{} is not available on this display / GPU", output.label()))?;
        self.config.format = format;
        self.config.color_space = color_space;
        self.surface.configure(&self.device, &self.config);
        self.output = output;
        log::info!("output: {} ({format:?}, {color_space:?})", output.label());
        Ok(())
    }

    /// The display's HDR figures (peak / SDR-white nits, headroom), re-queried from
    /// the OS: call after the window moved or the display changed.
    pub fn hdr_info(&self) -> wgpu::DisplayHdrInfo {
        self.surface.display_hdr_info(&self.adapter)
    }

    /// The surface format (`Output::surface` picked it).
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Reconfigure the surface after a window resize. No-op on zero area.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        // Structural detail (debug): surface reconfigured to the new window size.
        log::debug!("gpu surface resize {width}x{height}");
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }
}
