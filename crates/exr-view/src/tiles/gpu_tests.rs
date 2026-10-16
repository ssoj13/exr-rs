//! GPU end-to-end checks (need an adapter, so `#[ignore]`d like the repo's other
//! GPU tests: `cargo test -p exr-view -- --ignored`).
//!
//! The tile path: a multi-tile image through the real path (`view::frame`: worker
//! tiles, RAM cache, atlas upload, plan, draw list; `TilePipeline`) into the canvas
//! format, read back and compared with the CPU twin (`transfer::display_pixel`, or
//! `ocio::Transform::display` with OCIO) of the same level's `f16` texels. The
//! present pass: every encoding of `present.rs` against its twin `present::encode`.

use std::sync::Arc;

use imath_rs::Half;

use super::atlas::{PageLayout, TileAtlas};
use super::grid::Grid;
use super::reduce::level_from_rgba;
use super::render::{Look, TileGpu, TilePipeline};
use super::source::FrameSource;
use super::view::{self, Placement, SourceDesc, TileState};
use crate::transfer::{self, mode};

/// A device with `FLOAT32_FILTERABLE` when the adapter has it, as `gpu_ctx` asks.
pub(crate) fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("a GPU adapter (this test is #[ignore]d for machines without one)");
    let desc = wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
        ..Default::default()
    };
    pollster::block_on(adapter.request_device(&desc)).expect("device")
}

/// Deterministic image: R in [-0.3, 1.2) (negatives: the SDR clip vs the HDR
/// keep of `transfer::canvas` differ by ~0.5 there), G/B in [0, 1.5), alpha in
/// [0.2, 1.0) (varying, so the alpha-isolate mode and the alpha output are really
/// checked).
fn image(w: u32, h: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 7 + y * 3) % 150) as f32 / 100.0 - 0.3);
            v.push(((x * 5 + y * 11) % 150) as f32 / 100.0);
            v.push(((x * 13 + y * 2) % 150) as f32 / 100.0);
            v.push(0.2 + ((x + 2 * y) % 80) as f32 / 100.0);
        }
    }
    v
}

/// A worker gate: no timing assumption can make the next frame accidentally
/// arrive before the assertion. Drop releases workers even after an assertion.
struct WorkerGate(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
fn test_look() -> Look {
    Look { exposure_ev: 0.0, gamma: 1.0, channel_mode: mode::RGBA, tonemap: 0, nearest: true, hdr: false, src: transfer::Src::LINEAR }
}
impl WorkerGate {
    fn new() -> Self { Self(Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()))) }
    fn release(&self) {
        *self.0.0.lock().unwrap() = true;
        self.0.1.notify_all();
    }
}
impl Drop for WorkerGate {
    fn drop(&mut self) { self.release(); }
}
struct GatedSource {
    source: FrameSource,
    gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}
impl super::source::TileSource for GatedSource {
    fn size(&self) -> (u32, u32) { (self.source.w, self.source.h) }
    fn unit(&self, id: super::grid::TileId) -> super::source::Unit { self.source.unit(id) }
    fn produce(
        &self,
        grid: &Grid,
        unit: super::source::Unit,
        context: &super::source::ProductionContext,
    ) -> super::source::Production {
        let mut ready = self.gate.0.lock().unwrap();
        while !*ready {
            ready = self.gate.1.wait(ready).unwrap();
        }
        self.source.produce(grid, unit, context)
    }
    fn scope_level(&self, _: &Grid, _: u64, _: u32) -> Option<(u32, Arc<Vec<f32>>)> { None }
}

/// Draw immediately; unlike `render`, this never waits for a tile worker.
fn draw_now(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut TileGpu, state: &mut TileState, place: Placement) -> Vec<f32> {
    let list = view::frame(state, &mut gpu.atlas, device, queue, &gpu.pipe.page_bgl, place, 0);
    let tex = target(device, crate::present::CANVAS_FORMAT, 16, 8);
    let target_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    gpu.pipe.prepare(device, queue, test_look(), gpu.atlas.layout.size, &list);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })], ..Default::default()
        }).forget_lifetime();
        gpu.pipe.paint(&mut pass, gpu.atlas.pages(), &list);
    }
    queue.submit([enc.finish()]);
    halves(&read_texture(device, queue, &tex, 16, 8, 8))
}

#[test]
#[ignore = "requires a GPU adapter"]
fn frame_transition_holds_pixels_until_the_next_generation_is_drawable() {
    let (device, queue) = device();
    // One slot: holding the old frame must not require a second atlas slot.
    let mut gpu = TileGpu {
        atlas: TileAtlas::new(PageLayout {
            size: 512,
            side: 1,
            max_pages: 1,
        }),
        pipe: TilePipeline::new(&device),
        direct: None,
        vram_budget: 64 << 20,
        canvas_bytes: 0,
    };
    let mut state = TileState::new(1);
    let desc = |frame| SourceDesc { frame, view: Default::default(), pixels_gen: frame, stage: frame };
    let source = |w, h, value| FrameSource { rgba: Arc::new([value, value, value, 1.0].repeat((w * h) as usize)), w, h };
    state.set_source(desc(1), Arc::new(source(8, 4, 0.25)));
    render(&device, &queue, &mut gpu, &mut state, 1.0, test_look(), 16, 8);
    let place = Placement { origin: egui::pos2(0.0, 0.0), zoom: 1.0, clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(16.0, 8.0)) };
    let old = draw_now(&device, &queue, &mut gpu, &mut state, place);
    assert!(old[0] > 0.5 && old[3] == 1.0, "old image really rendered");
    let gate = WorkerGate::new();
    for frame in [2, 3] {
        state.set_source(desc(frame), Arc::new(GatedSource { source: source(10, 6, 0.75), gate: gate.0.clone() }));
        assert_eq!(draw_now(&device, &queue, &mut gpu, &mut state, place), old, "pending frame {frame} must retain every old pixel");
    }
    gate.release();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.pending() != 0 {
        state.drain();
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    let next = draw_now(&device, &queue, &mut gpu, &mut state, place);
    assert!(next[0] > 0.8 && next[3] == 1.0, "next image rendered, not the held pixels");
    assert!(next[(5 * 16 + 9) * 4] > 0.8, "new extent is fully covered");
    assert_eq!(next[(6 * 16 + 10) * 4 + 3], 0.0, "outside new extent stays empty");
}

/// Read `texture` (`w x h`, `texel` bytes each) back, rows packed.
pub(crate) fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    w: u32,
    h: u32,
    texel: u32,
) -> Vec<u8> {
    read_texture_at(device, queue, texture, w, h, texel, wgpu::Origin3d::ZERO)
}

/// Read a texel region from the slot actually bound by the presentation.
pub(crate) fn read_texture_at(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    w: u32,
    h: u32,
    texel: u32,
    origin: wgpu::Origin3d,
) -> Vec<u8> {
    let row = (w * texel).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-test.readback"),
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut source = texture.as_image_copy();
    source.origin = origin;
    enc.copy_texture_to_buffer(
        source,
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    let data = buf.slice(..).get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((w * h * texel) as usize);
    for y in 0..h {
        let r = (y * row) as usize;
        out.extend_from_slice(&data[r..r + (w * texel) as usize]);
    }
    out
}

/// A render target of `format` that can be read back.
pub(crate) fn target(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu-test.target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// `f16` RGBA texels read back as `f32`.
pub(crate) fn halves(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|b| Half::from_bits(u16::from_le_bytes(*b)).to_f32()).collect()
}

/// The draw list of `state` at `zoom` (image origin at the top-left of a `w x h`
/// canvas), once the workers have finished, so the frame draws the level it picked, not
/// a fallback.
fn settled_list(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut TileGpu,
    state: &mut TileState,
    zoom: f32,
    w: u32,
    h: u32,
) -> super::render::DrawList {
    let place = Placement {
        origin: egui::pos2(0.0, 0.0),
        zoom,
        clip: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(w as f32, h as f32)),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let TileGpu { atlas, pipe, .. } = &mut *gpu;
        view::frame(state, atlas, device, queue, &pipe.page_bgl, place, usize::MAX);
        if state.pending() == 0 {
            break view::frame(state, atlas, device, queue, &pipe.page_bgl, place, usize::MAX);
        }
        assert!(std::time::Instant::now() < deadline, "tiles never landed");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Render `state`'s content at `zoom` (image origin at the target's top-left)
/// into a `w x h` canvas-format target and read the RGBA values back.
#[allow(clippy::too_many_arguments)]
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &mut TileGpu,
    state: &mut TileState,
    zoom: f32,
    look: Look,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let list = settled_list(device, queue, gpu, state, zoom, w, h);
    let texture = target(device, crate::present::CANVAS_FORMAT, w, h);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    gpu.pipe.prepare(device, queue, look, gpu.atlas.layout.size, &list);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tiles-test.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        gpu.pipe.paint(&mut pass, gpu.atlas.pages(), &list);
    }
    queue.submit([enc.finish()]);
    halves(&read_texture(device, queue, &texture, w, h, 8))
}

/// Compare a render with the built-in CPU twin over the level-`k` `f16` texels.
fn check(got: &[f32], level: &[f32], w: u32, h: u32, look: Look, what: &str) {
    let twin = |t: [f32; 4]| {
        transfer::canvas(transfer::display_pixel(t, &look.src, look.exposure_ev, look.gamma, look.channel_mode, look.tonemap), look.hdr)
    };
    check_with(got, level, w, h, what, twin);
}

/// Compare a render with `twin` (linear texel -> canvas value) over the `f16`
/// texels: within one 8-bit code (what an SDR surface shows) plus the canvas's
/// `f16` rounding (relative 2^-10) for values above white. GPU and CPU `pow` /
/// OCIO math differ by ~1e-4, which the curve's steep toe amplifies near zero.
fn check_with(got: &[f32], level: &[f32], w: u32, h: u32, what: &str, twin: impl Fn([f32; 4]) -> [f32; 4]) {
    let mut worst = 0f32;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let texel = [0, 1, 2, 3].map(|c| f32::from(Half::from(level[i + c])));
            let want = twin(texel);
            for c in 0..4 {
                let d = (got[i + c] - want[c]).abs();
                worst = worst.max(d);
                assert!(
                    d <= 1.0 / 255.0 + want[c].abs() / 1024.0,
                    "{what}: pixel {x},{y} channel {c}: got {} want {}",
                    got[i + c],
                    want[c]
                );
            }
        }
    }
    eprintln!("{what}: worst difference {worst:e}");
}

/// [`check`] in linear light for the absolute curves (PQ, HLG): their toe subtracts two numbers
/// near `c1` (PQ) before a power of 6.3, so the GPU's `pow` (~1e-6 relative) leaves ~1e-3 of the
/// light near black, which the canvas's sRGB slope (12.92) and a gamut matrix that cancels
/// components magnify past one 8-bit code. Within 3e-3 of the light, relative above 1 (the
/// `f16` canvas alone rounds by 2^-10, ~2.4e-3 once decoded to light).
fn check_light(got: &[f32], level: &[f32], w: u32, h: u32, look: Look, what: &str) {
    let mut worst = 0f32;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let texel = [0, 1, 2, 3].map(|c| f32::from(Half::from(level[i + c])));
            let want = transfer::canvas(transfer::display_pixel(texel, &look.src, look.exposure_ev, look.gamma, look.channel_mode, look.tonemap), look.hdr);
            for c in 0..3 {
                let (g, w) = (transfer::eotf(got[i + c]), transfer::eotf(want[c]));
                let d = (g - w).abs();
                worst = worst.max(d);
                assert!(d <= 3e-3 * w.abs().max(1.0), "{what}: pixel {x},{y} channel {c}: light {g} want {w}");
            }
            assert!((got[i + 3] - want[3]).abs() <= 1.0 / 255.0, "{what}: alpha at {x},{y}");
        }
    }
    eprintln!("{what}: worst light difference {worst:e}");
}

#[test]
#[ignore = "needs a GPU adapter"]
fn tiles_render_the_display_transform_of_every_texel() {
    let (device, queue) = device();
    let pipe = TilePipeline::new(&device);
    // A small page (1024) keeps the test's VRAM tiny; 700x300 still spans
    // several tiles with partial edges at level 0.
    let mut gpu = TileGpu {
        atlas: TileAtlas::new(PageLayout::new(1024, 64 << 20)),
        pipe,
        direct: None,
        vram_budget: 64 << 20,
        canvas_bytes: 0,
    };
    let (w, h) = (700u32, 300u32);
    let rgba = image(w, h);
    let mut state = TileState::new(usize::MAX);
    state.set_source(
        SourceDesc { frame: 1, view: crate::image::Selection::default(), pixels_gen: 0, stage: 0 },
        Arc::new(FrameSource { rgba: Arc::new(rgba.clone()), w, h }),
    );

    use crate::transfer::tonemap;
    for (ev, gamma, channel_mode, tonemap, hdr) in [
        (0.0, 1.0, mode::RGBA, tonemap::NONE, false),
        (0.0, 1.0, mode::RGBA, tonemap::NONE, true),
        (1.0, 1.2, mode::RGBA, tonemap::NONE, false),
        (1.0, 1.2, mode::RGBA, tonemap::NONE, true),
        (-0.5, 1.0, mode::R, tonemap::NONE, false),
        (-0.5, 1.0, mode::R, tonemap::NONE, true),
        (0.0, 1.0, mode::LUM, tonemap::NONE, false),
        (0.0, 1.0, mode::A, tonemap::NONE, false),
        (2.0, 1.0, mode::RGBA, tonemap::REINHARD, false),
        (2.0, 1.0, mode::RGBA, tonemap::AGX, false),
        (-3.0, 1.1, mode::RGBA, tonemap::AGX, false),
    ] {
        let look = Look { exposure_ev: ev, gamma, channel_mode, tonemap, nearest: true, hdr, src: transfer::Src::LINEAR };
        let got = render(&device, &queue, &mut gpu, &mut state, 1.0, look, w, h);
        let what = format!("level 0 ev={ev} gamma={gamma} mode={channel_mode} tone={tonemap} hdr={hdr}");
        check(&got, &rgba, w, h, look, &what);
    }
    // A frame read through its own tag (`transfer::decode`) and alpha kind: the built-in decode
    // of sRGB (straight and associated alpha), PQ and HLG (absolute, scaled to a 203-nit SDR
    // white), Display P3, Cineon; an isolated channel of an encoded frame is shown as its code.
    // The curves past sRGB read straight texels: their codes stay in 0..1, the domain they are
    // defined on (an associated code over a 0.2 alpha is 5, past PQ's 1.0 and Cineon's f16 range).
    let codes: Vec<f32> = rgba.iter().map(|v| v.rem_euclid(1.0)).collect();
    for (case, (space, straight, gain, channel_mode, hdr, what)) in [
        ("srgb_rec709_scene", true, 1.0, mode::RGBA, false, "sRGB straight"),
        ("srgb_rec709_scene", false, 1.0, mode::RGBA, false, "sRGB associated"),
        ("srgb_rec709_scene", true, 1.0, mode::R, false, "sRGB isolate"),
        ("pq_rec2020_display", true, 100.0 / 203.0, mode::RGBA, true, "PQ"),
        ("hlg_rec2020_display", true, 100.0 / 203.0, mode::RGBA, true, "HLG"),
        ("srgb_p3d65_display", true, 1.0, mode::RGBA, true, "Display P3"),
        ("KodakLog", true, 1.0, mode::RGBA, false, "Cineon"),
    ].into_iter().enumerate() {
        let mut frame = crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
        frame.tag = exr_import::ColorTag { space: space.to_owned(), origin: exr_import::TagOrigin::File };
        frame.straight = straight;
        let mut light = codes.clone();
        if channel_mode == mode::RGBA {
            crate::ocio::Decoder::builtin_only().decode(&frame).unwrap().apply(&mut light);
        }
        state.set_source(
            SourceDesc { frame: case as u64 + 2, view: crate::image::Selection::default(), pixels_gen: 0, stage: 0 },
            Arc::new(FrameSource { rgba: Arc::new(light.clone()), w, h }),
        );
        let src = transfer::Src { gain };
        let look = Look { exposure_ev: 0.5, gamma: 1.1, channel_mode, tonemap: tonemap::NONE, nearest: true, hdr, src };
        let got = render(&device, &queue, &mut gpu, &mut state, 1.0, look, w, h);
        let what = format!("decode {what} mode={channel_mode} hdr={hdr}");
        if space.contains("pq_") || space.contains("hlg_") {
            check_light(&got, &light, w, h, look, &what);
        } else {
            check(&got, &light, w, h, look, &what);
        }
    }
    state.set_source(
        SourceDesc { frame: 1, view: crate::image::Selection::default(), pixels_gen: 0, stage: 0 },
        Arc::new(FrameSource { rgba: Arc::new(rgba.clone()), w, h }),
    );

    // Zoom 0.5: level 1, one level texel per target pixel.
    let grid = Grid::new(w, h);
    let (lw, lh) = grid.level_size(1);
    let level1 = level_from_rgba(&rgba, &grid, 1).unwrap();
    let look = Look { exposure_ev: 0.0, gamma: 1.0, channel_mode: mode::RGBA, tonemap: 0, nearest: false, hdr: false, src: transfer::Src::LINEAR };
    let got = render(&device, &queue, &mut gpu, &mut state, 0.5, look, lw, lh);
    check(&got, &level1, lw, lh, look, "level 1 at zoom 0.5");
}

/// HDR ramp: each channel sweeps 2^-8 .. 2^4 (0.004 .. 16) at a different rate,
/// so highlights far above 1.0 and deep shadows both go through the transform.
fn hdr_image(w: u32, h: u32) -> Vec<f32> {
    let ramp = |t: f32| 2f32.powf(-8.0 + 12.0 * t.fract());
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            v.push(ramp(fx));
            v.push(ramp(fx * 0.7 + fy));
            v.push(ramp(fy * 1.3 + fx * 0.2));
            v.push(0.3 + 0.7 * fy);
        }
    }
    v
}

/// A 3D LUT that mixes channels and bends them (not separable, not identity), so a
/// wrong binding, axis order or sampler shows.
fn lut_transform() -> crate::ocio::Transform {
    let mut lut = vfx_ocio::Lut3DTransform::identity(17);
    for e in &mut lut.data {
        let [r, g, b] = *e;
        *e = [(0.7 * r + 0.3 * g).sqrt(), g * g, 1.0 - 0.8 * b];
    }
    let proc = vfx_ocio::Processor::from_transform(&vfx_ocio::Transform::Lut3D(lut), vfx_ocio::TransformDirection::Forward)
        .expect("LUT processor");
    crate::ocio::Transform::new(proc, "test LUT".to_owned(), false, false).expect("LUT transform")
}

/// The OCIO chain on the GPU equals its CPU twin on every texel of a multi-tile
/// HDR image: the ACES 2.0 SDR view (fixed function, group 1 only) with exposure
/// and gamma, an isolate mode (which bypasses OCIO), the HDR 1000-nit view in its
/// linear (HDR output) form, absolute, on an HDR output (negatives kept) and on an
/// SDR one (negatives clipped), and a 3D LUT transform (group 2: an `Rgba32Float`
/// LUT with a filtering sampler).
#[test]
#[ignore = "needs a GPU adapter"]
fn tiles_render_the_ocio_transform_of_every_texel() {
    let (device, queue) = device();
    let pipe = TilePipeline::new(&device);
    let mut gpu = TileGpu {
        atlas: TileAtlas::new(PageLayout::new(1024, 64 << 20)),
        pipe,
        direct: None,
        vram_budget: 64 << 20,
        canvas_bytes: 0,
    };
    let (w, h) = (700u32, 300u32);
    let rgba = hdr_image(w, h);
    let mut state = TileState::new(usize::MAX);
    state.set_source(
        SourceDesc { frame: 1, view: crate::image::Selection::default(), pixels_gen: 0, stage: 0 },
        Arc::new(FrameSource { rgba: Arc::new(rgba.clone()), w, h }),
    );

    let o = crate::ocio::Ocio::load("ocio://studio-config-latest").unwrap();
    let scene = o.scene();
    let aces = o.transform(&o.resolve(&crate::ocio::Sel::default(), &scene, false).unwrap(), false).unwrap();
    assert!(!aces.has_luts(), "the ACES 2.0 view is fixed function; the LUT case below covers group 2");
    // The HDR (linear) form of a 1000-nit view: canvas values well above 1.0.
    let hdr_sel = crate::ocio::Sel {
        display: "Rec.2100-PQ - Display".into(),
        scene_view: "ACES 2.0 - HDR 1000 nits (P3 D65)".into(),
        ..Default::default()
    };
    let hdr = o.transform(&o.resolve(&hdr_sel, &scene, true).unwrap(), true).unwrap();
    assert!(hdr.linear());
    let lut = lut_transform();
    assert!(lut.has_luts());
    assert!(
        device.features().contains(wgpu::Features::FLOAT32_FILTERABLE),
        "this adapter cannot filter Rgba32Float: the LUT case cannot run here"
    );

    // SDR white of the HDR cases: the absolute 1000-nit view scales by 100/203.
    let white = crate::present::Target { hdr: true, white: 203.0, peak: 1000.0 };
    let gain = |t: &crate::ocio::Transform| crate::ocio::display_gain(t.absolute().then_some(100.0), &white);
    assert!((gain(&hdr) - 100.0 / 203.0).abs() < 1e-6 && gain(&aces) == 1.0);
    for (t, ev, gamma, channel_mode, out_hdr) in [
        (&aces, 0.0, 1.0, mode::RGBA, false),
        (&aces, 1.5, 1.3, mode::RGBA, false),
        (&aces, -2.0, 0.8, mode::RGBA, false),
        (&aces, 0.0, 1.0, mode::G, false),
        (&hdr, 0.0, 1.0, mode::RGBA, true),
        (&hdr, 1.0, 1.2, mode::RGBA, true),
        (&hdr, 0.0, 1.0, mode::RGBA, false),
        (&lut, 0.0, 1.0, mode::RGBA, false),
        (&lut, -3.0, 1.2, mode::RGBA, false),
    ] {
        gpu.pipe.set_colour(&device, &queue, Some(t)).unwrap();
        let src = transfer::Src { gain: gain(t) };
        let look = Look {
            exposure_ev: ev,
            gamma,
            channel_mode,
            tonemap: transfer::tonemap::AGX,
            nearest: true,
            hdr: out_hdr,
            src,
        };
        let got = render(&device, &queue, &mut gpu, &mut state, 1.0, look, w, h);
        let what = format!("{} ev={ev} gamma={gamma} mode={channel_mode} hdr={out_hdr}", t.key);
        check_with(&got, &rgba, w, h, &what, |texel| {
            if channel_mode != mode::RGBA {
                // Isolated channels are data: the built-in path, tone operator ignored.
                let d = transfer::display_pixel(texel, &src, ev, gamma, channel_mode, transfer::tonemap::NONE);
                return transfer::canvas(d, out_hdr);
            }
            let mut px = [texel];
            t.display(&mut px, &src, ev, gamma);
            transfer::canvas(px[0], out_hdr)
        });
    }

    // Back to the built-in transform: the pipeline really switches back.
    gpu.pipe.set_colour(&device, &queue, None).unwrap();
    let look = Look {
        exposure_ev: 0.0,
        gamma: 1.0,
        channel_mode: mode::RGBA,
        tonemap: transfer::tonemap::AGX,
        nearest: true,
        hdr: false,
        src: transfer::Src::LINEAR,
    };
    let got = render(&device, &queue, &mut gpu, &mut state, 1.0, look, w, h);
    check(&got, &rgba, w, h, look, "built-in again");
}

/// Every encoding of the present pass against its CPU twin (`present::encode`),
/// dither off, over canvas values from below black to far above white: 8-bit
/// unorm, an `*Srgb` format (decoded, re-encoded by the hardware), 10-bit, PQ, HLG
/// (10-bit and float, a 600-nit peak so the system gamma is not 1.2) and scRGB. A
/// uniform canvas per value (a clear), read back from a small target.
#[test]
#[ignore = "needs a GPU adapter"]
fn present_encodes_like_its_twin() {
    use crate::present::{self, Encoding, Output, PresentPass};
    use wgpu::TextureFormat as F;
    let (device, queue) = device();
    let (w, h) = (4u32, 4u32);
    // (output, format, its encoding, one code of the format for the tolerance).
    let cases: [(Output, F, Encoding, f32); 7] = [
        (Output::Sdr8, F::Rgba8Unorm, Encoding::Unorm, 1.0 / 255.0),
        (Output::Sdr8, F::Rgba8UnormSrgb, Encoding::SrgbFormat, 1.0 / 255.0),
        (Output::Sdr10, F::Rgb10a2Unorm, Encoding::Unorm, 1.0 / 1023.0),
        (Output::Hdr10, F::Rgb10a2Unorm, Encoding::Pq, 1.0 / 1023.0),
        (Output::Hlg, F::Rgb10a2Unorm, Encoding::Hlg, 1.0 / 1023.0),
        (Output::Hlg, F::Rgba16Float, Encoding::Hlg, 0.0),
        (Output::Scrgb, F::Rgba16Float, Encoding::Scrgb, 0.0),
    ];
    let values = [-0.3f32, 0.0, 0.02, 0.18, 0.4613, 0.73, 1.0, 1.6, 3.5, f32::NAN];
    let levels = present::Target { hdr: true, white: 240.0, peak: 600.0 };
    for (output, format, encoding, code) in cases {
        assert_eq!(output.encoding(format), encoding);
        let mut pass = PresentPass::new(&device, format, output);
        pass.dither = false;
        let frame = target(&device, format, w, h);
        let frame_view = frame.create_view(&wgpu::TextureViewDescriptor::default());
        for v in values {
            // The canvas is f16: the twin gets the value the canvas holds.
            let canvas = [v, v * 0.5, v * 0.25].map(|c| Half::from(c).to_f32());
            let canvas_view = pass.canvas(&device, w, h).expect("the test canvas").clone();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present-test.canvas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &canvas_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(canvas[0]),
                            g: f64::from(canvas[1]),
                            b: f64::from(canvas[2]),
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.draw(&queue, &mut enc, &frame_view, levels);
            queue.submit([enc.finish()]);
            let texel = if format == F::Rgba16Float { 8 } else { 4 };
            let bytes = read_texture(&device, &queue, &frame, w, h, texel);
            let got: [f32; 3] = match format {
                F::Rgba16Float => {
                    let f = halves(&bytes[..8]);
                    [f[0], f[1], f[2]]
                }
                F::Rgb10a2Unorm => {
                    let p = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    [p & 1023, (p >> 10) & 1023, (p >> 20) & 1023].map(|c| c as f32 / 1023.0)
                }
                _ => [bytes[0], bytes[1], bytes[2]].map(|c| f32::from(c) / 255.0),
            };
            // What the surface holds: an `*Srgb` format stores the encode of the
            // pass's (decoded) output, so compare the sRGB codes.
            let want = present::encode(canvas, encoding, levels);
            let want = if encoding == Encoding::SrgbFormat { want.map(crate::transfer::oetf) } else { want };
            for c in 0..3 {
                let tol = code.max(1e-3 + want[c].abs() / 1024.0) + 1e-4;
                assert!(
                    (got[c] - want[c]).abs() <= tol,
                    "{output:?} {format:?} canvas {canvas:?} channel {c}: got {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an actual display GPU"]
fn single_slot_atlas_accepts_overview_when_optional_fine_uploads_are_refused() {
    let (device, queue) = device();
    let mut gpu = TileGpu::new(&device, 2 << 20);
    assert_eq!(gpu.atlas.layout.per_page(), 1);
    assert_eq!(gpu.atlas.layout.max_pages, 1);
    let source = Arc::new(FrameSource { rgba: Arc::new(vec![1.0; 1024 * 512 * 4]), w: 1024, h: 512 });
    let mut state = TileState::new(32 << 20);
    let desc = SourceDesc { frame: 41, stage: 0, view: Default::default(), pixels_gen: 0 };
    state.set_source(desc.clone(), source);
    let grid = Grid::new(1024, 512);
    let top = super::grid::TileId { level: grid.levels() - 1, tx: 0, ty: 0 };
    state.request([top, super::grid::TileId { level: 0, tx: 0, ty: 0 },
        super::grid::TileId { level: 0, tx: 1, ty: 0 }]);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.pending() != 0 {
        state.drain();
        assert!(std::time::Instant::now() < deadline, "resident fixture never completed");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let place = Placement { origin: egui::Pos2::ZERO, zoom: 1.0,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 512.0)) };
    for _ in 0..8 {
        let list = view::frame(&mut state, &mut gpu.atlas, &device, &queue,
            &gpu.pipe.page_bgl, place, usize::MAX);
        assert_eq!(state.accepted_description(), Some(&desc));
        assert_eq!(list.instances.len(), 2, "both fine rectangles must use the complete overview");
        assert_eq!(gpu.atlas.pages().len(), 1, "optional fine uploads cannot grow this atlas");
    }
}

/// The reported bug end to end: a texel with light and no coverage (a lens flare:
/// `rgb > 0`, `a == 0`) goes through the real tile pass onto a cleared canvas and the
/// real present pass over each backdrop, and keeps its light. The canvas the tile
/// shader writes is encoded associated light (the contract of `egui_display::backdrop`),
/// so the result in scRGB (linear, white 80 nits) is `light + backdrop * (1 - a)` for
/// every texel: emissive, partial alpha and opaque.
#[test]
#[ignore = "needs a GPU adapter"]
fn emissive_texel_survives_the_backdrop_present() {
    use egui_display::{Backdrop, BackdropKind, BackdropView, Output, PresentPass, Target};
    let (device, queue) = device();
    let mut gpu = TileGpu {
        atlas: TileAtlas::new(PageLayout::new(1024, 64 << 20)),
        pipe: TilePipeline::new(&device),
        direct: None,
        vram_budget: 64 << 20,
        canvas_bytes: 0,
    };
    let (w, h) = (8u32, 4u32);
    // Associated (premultiplied) EXR light per column: flare, tiny alpha, half alpha, opaque.
    let texels: [[f32; 4]; 4] =
        [[2.0, 0.5, 0.05, 0.0], [3.0, 1.0, 0.2, 1e-4], [0.2, 0.1, 0.05, 0.5], [0.3, 0.6, 0.9, 1.0]];
    let rgba: Vec<f32> = (0..w * h).flat_map(|i| texels[(i % 4) as usize]).collect();
    let mut state = TileState::new(usize::MAX);
    state.set_source(
        SourceDesc { frame: 1, view: crate::image::Selection::default(), pixels_gen: 0, stage: 0 },
        Arc::new(FrameSource { rgba: Arc::new(rgba), w, h }),
    );
    let list = settled_list(&device, &queue, &mut gpu, &mut state, 1.0, w, h);
    let look = Look { hdr: true, ..test_look() };
    gpu.pipe.prepare(&device, &queue, look, gpu.atlas.layout.size, &list);

    let mut pass = PresentPass::new(&device, wgpu::TextureFormat::Rgba16Float, Output::Scrgb);
    pass.dither = false;
    let canvas = pass.canvas(&device, w, h).expect("the test canvas").clone();
    let frame = target(&device, wgpu::TextureFormat::Rgba16Float, w, h);
    let levels = Target { hdr: true, white: 80.0, peak: 1000.0 };
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
    for kind in [BackdropKind::Black, BackdropKind::Gray50] {
        let backdrop = Backdrop { kind, ..Backdrop::default() };
        let uniform = backdrop.uniform(&BackdropView::new(egui::Pos2::ZERO, 1.0), rect);
        pass.backdrops = vec![uniform];
        let bd = uniform.color_at([0.5, 0.5]).map(crate::transfer::eotf);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            // What `show_backdrop` does to the canvas, then the tiles on top of it.
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("emissive-test.canvas"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &canvas,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            gpu.pipe.paint(&mut rp, gpu.atlas.pages(), &list);
        }
        pass.draw(&queue, &mut enc, &frame.create_view(&wgpu::TextureViewDescriptor::default()), levels);
        queue.submit([enc.finish()]);
        let out = halves(&read_texture(&device, &queue, &frame, w, h, 8));
        for y in 0..h {
            for x in 0..w {
                let t = texels[(x % 4) as usize];
                let i = ((y * w + x) * 4) as usize;
                for c in 0..3 {
                    let want = t[c] + bd[c] * (1.0 - t[3]);
                    let tol = 4e-3 * (1.0 + want);
                    assert!(
                        (out[i + c] - want).abs() <= tol,
                        "{kind:?} texel {x},{y} ch{c}: got {} want {want} (texel {t:?})",
                        out[i + c]
                    );
                }
            }
        }
    }
}
