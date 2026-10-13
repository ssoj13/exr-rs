//! Real GPU checks for direct engagement, reduction precision and shared drawing.
#[test]
#[ignore = "needs a GPU adapter"]
fn precision_half_domain_cancellation_and_rounding_edges() {
    let (d, q) = device();
    let pipe = super::render::TilePipeline::new(&d);
    let mut direct = DirectGpu::new(&d);
    let patterns: &[[f32; 4]] = &[
        [65504.0, f32::from_bits(0x35800000), -65504.0, 0.0],
        [-65504.0, -f32::from_bits(0x35800000), 65504.0, 0.0],
        [0.0, -0.0, 0.0, -0.0],
        [0.000000059604645, 0.00000011920929, 0.0, 0.0],
        [
            1.0,
            f32::from_bits(0x3f80_2000),
            1.0,
            f32::from_bits(0x3f80_2000),
        ],
        [f32::from_bits(0x3f801000); 4],
        [f32::from_bits(0x3f801001); 4],
        [f32::from_bits(0x3f800fff); 4],
        [f32::from_bits(0x3f803000); 4],
        [f32::from_bits(0x33000000); 4],
        [f32::from_bits(0x33000001); 4],
        [-f32::from_bits(0x33000001); 4],
    ];
    for (case, pattern) in patterns.iter().enumerate() {
        let mut rgba = vec![0.0; 2048 * 4];
        for (x, v) in pattern.iter().enumerate() {
            rgba[x * 4..x * 4 + 4].fill(*v);
        }
        let grid = Grid::new(2048, 1);
        let zoom = 0.25;
        // Height 1 still reduces four clipped horizontal samples per cell.
        let desc = SourceDesc {
            frame: case as u64 + 1,
            view: Default::default(),
            pixels_gen: 0,
            stage: 0,
        };
        assert!(
            direct
                .prepare(
                    &d,
                    &q,
                    &pipe.page_bgl,
                    &desc,
                    super::direct::Texels::F32(&rgba),
                    None,
                    (2048, 1),
                    place(512, 1, zoom),
                    64 << 20
                )
                .unwrap()
        );
        let k = grid.pick_level(zoom);
        assert_eq!(k, 2, "four clipped horizontal samples per cell");
        let reference = level_from_rgba(&rgba, &grid, k).unwrap();
        let (w, h) = grid.level_size(k);
        let bytes = read_texture(&d, &q, &direct.pages()[0].tex, w, h, 8);
        for (i, (b, want)) in bytes
            .as_chunks::<2>().0.iter()
            .zip(reference.iter().copied())
            .enumerate()
        {
            let actual = Half::from_bits(u16::from_le_bytes(*b));
            let want = Half::from(want);
            if want.to_f32().is_nan() {
                assert!(actual.to_f32().is_nan(), "NaN component{i}");
            } else if want.to_f32().is_infinite() {
                assert_eq!(actual.bits(), want.bits(), "infinity component{i}");
            } else {
                assert!(
                    actual.bits() == want.bits()
                        || (actual.bits() & 0x7fff == 0 && want.bits() & 0x7fff == 0),
                    "component{i} actual{} want{}",
                    actual.to_f32(),
                    want.to_f32()
                );
            }
        }
        assert_eq!(
            u16::from_le_bytes([bytes[0], bytes[1]]),
            Half::from(reference[0]).bits(),
            "case{case} exact staged rounding"
        );
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn exceptional_float_sources_reduce_on_the_gpu_like_the_f64_reference() {
    let (d, q) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&d, 64 << 20));
    let mut state = DisplayTarget::new(8 << 20, usize::MAX);
    let cases = [
        [1e6, 0.01, -1e6, 0.0],
        [-1e6, -0.01, 1e6, 0.0],
        [f32::MAX, f32::MAX, -f32::MAX, -f32::MAX],
        [f32::MAX, 1.0, -f32::MAX, 0.0],
        [f32::INFINITY, 1.0, 0.0, 0.0],
        [f32::NEG_INFINITY, 1.0, 0.0, 0.0],
        [f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0, 0.0],
        [f32::from_bits(1), 0.0, 0.0, 0.0],
        [65520.0; 4],
    ];
    for (case, values) in cases.iter().enumerate() {
        let mut frame = frame_with(&["R", "G", "B", "A"]);
        frame.width = 2048;
        frame.height = 4;
        frame.id = case as u64 + 100;
        for (name, channel) in &mut frame.channels {
            let v = (0..8192)
                .map(|i| match name.as_str() {
                    "R" => values[i % 4],
                    "G" => 0.25,
                    "B" => 0.0,
                    _ => 1.0,
                })
                .collect::<Vec<_>>();
            *channel = exr_core::api::ChannelData::Float(v);
        }
        let mut image = ImageBackend::from_frame(Arc::new(frame));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let cb = loop {
            let cb = image
                .present(
                    &d,
                    &q,
                    &mut renderer,
                    &mut state,
                    None,
                    place(512, 1, 0.25),
                    look(),
                    None,
                )
                .unwrap();
            if state.can_advance(case as u64 + 100) && state.tiles_pending() == 0 {
                break cb;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "fallback must become drawable"
            );
            std::thread::yield_now();
        };
        // Outside the exact fixed-point domain the resident path still engages,
        // through the per-cell-window integer kernel (`main_wide`), not CPU tiles.
        assert!(cb.direct, "case{case}: the direct path must engage");
        let gpu = renderer.callback_resources.get::<TileGpu>().unwrap();
        let direct = gpu.direct.as_ref().unwrap();
        assert_eq!(direct.dispatches, case + 1, "case{case}: one GPU reduction per frame");
        assert_eq!(direct.uploads, case + 1, "case{case}: one upload per frame");
        assert_eq!(state.scheduled_units(), 0, "case{case}: no CPU tile work");
        let bytes = read_texture(&d, &q, &direct.pages()[0].tex, 512, 1, 8);
        let actual = Half::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        // The f64 CPU reference: the 4x4 cell at level 2 holds each value 4 times.
        let mut sum = 0f64;
        for _ in 0..4 {
            for &v in values {
                sum += f64::from(v);
            }
        }
        let want = Half::from((sum / 16.0) as f32);
        eprintln!("case{case} got={:#06x} want={:#06x}", actual.bits(), want.bits());
        if want.to_f32().is_nan() {
            assert!(actual.to_f32().is_nan(), "case{case}");
        } else {
            assert_eq!(actual.bits(), want.bits(), "case{case}");
        }
        if want.to_f32().is_finite() {
            let shown = draw_callback(&d, &q, &mut renderer, &cb, 512, 1);
            let expected = crate::transfer::canvas(
                crate::transfer::display_pixel(
                    [want.to_f32(), 0.25, 0.0, 1.0],
                    &crate::transfer::Src::LINEAR,
                    0.0,
                    1.0,
                    crate::transfer::mode::RGBA,
                    0,
                ),
                false,
            );
            for c in 0..4 {
                assert!(
                    (shown[c] - expected[c]).abs() <= 1.0 / 255.0,
                    "case{case} c{c}"
                );
            }
        }
    }
}

/// Render-like float data (signed, exponents 2^-40..2^20, so the frame is far
/// outside the exact domain) reduced on the GPU at every level the direct path
/// can pick: each half must equal the f64 CPU reference bit for bit.
#[test]
#[ignore = "needs a GPU adapter"]
fn wide_kernel_matches_the_f64_reference_on_render_like_floats() {
    let (d, q) = device();
    let pipe = super::render::TilePipeline::new(&d);
    let mut direct = DirectGpu::new(&d);
    // A small deterministic LCG: reproducible without a dev-dependency.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    // Odd extents: every level has clipped edge cells (n < 4^k); 4133 wide keeps
    // five levels, so the direct path's coarsest 16 x 16 cell is reached.
    let (w, h) = (4133u32, 261u32);
    let rgba: Vec<f32> = (0..w as usize * h as usize * 4)
        .map(|_| {
            let r = next();
            let exponent = (r % 61) as i32 - 40;
            let mantissa = 1.0 + f64::from(next() & 0xffff) / 65536.0;
            let sign = if r & (1 << 20) != 0 { -1.0 } else { 1.0 };
            (sign * mantissa * 2f64.powi(exponent)) as f32
        })
        .collect();
    assert!(super::precision::profile(&rgba).is_none(), "the data must need the wide kernel");
    let grid = Grid::new(w, h);
    for (case, zoom) in [0.5f32, 0.25, 0.125, 0.0625].into_iter().enumerate() {
        let k = grid.pick_level(zoom);
        assert_eq!(k, case as u32 + 1);
        let desc = SourceDesc { frame: case as u64 + 1, view: Default::default(), pixels_gen: 0, stage: 0 };
        let (lw, lh) = grid.level_size(k);
        assert!(direct.prepare(&d, &q, &pipe.page_bgl, &desc, super::direct::Texels::F32(&rgba), None, (w, h), place(lw, lh, zoom), 256 << 20).unwrap());
        let reference = level_from_rgba(&rgba, &grid, k).unwrap();
        let bytes = read_texture(&d, &q, &direct.pages()[0].tex, lw, lh, 8);
        let mut mismatches = 0;
        for (i, (b, want)) in bytes.as_chunks::<2>().0.iter().zip(reference.iter().copied()).enumerate() {
            let actual = u16::from_le_bytes(*b);
            let want = Half::from(want).bits();
            // +0 and -0 are the same display value.
            if actual != want && !(actual & 0x7fff == 0 && want & 0x7fff == 0) {
                if mismatches < 8 {
                    eprintln!("level{k} component{i}: got {actual:#06x} want {want:#06x}");
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "level{k}: GPU halves differ from the f64 reference");
    }
}

struct Stage {
    key: u64,
    calls: usize,
    output: Vec<f32>,
}
impl crate::image::ImageStage for Stage {
    fn key(&self) -> u64 {
        self.key
    }
    fn process<'a>(&'a mut self, pixels: &'a [f32], _: u32, _: u32) -> &'a [f32] {
        self.calls += 1;
        self.output.clear();
        self.output.extend_from_slice(pixels);
        let gain = if self.key == 1 { 0.5 } else { 0.25 };
        for px in self.output.as_chunks_mut::<4>().0 {
            for v in &mut px[..3] {
                *v *= gain;
            }
        }
        &self.output
    }
}
/// An idle stage (key 0) is not run: the shown pixels are the materialized
/// ones, not a processed copy (the viewer always passes its effect host).
#[test]
fn idle_stage_shares_the_materialized_pixels() {
    let mut target = DisplayTarget::new(8 << 20, usize::MAX);
    let mut image = ImageBackend::from_frame(Arc::new(frame_with(&["R", "G", "B"])));
    let mut stage = Stage { key: 0, calls: 0, output: Vec::new() };
    let (_, _, _, export) = image.export_level(&mut target, 8192, Some(&mut stage)).unwrap();
    assert_eq!(&export[..], &[0.0, 1.0, 2.0, 1.0], "the unprocessed pixels");
    assert_eq!(stage.calls, 0, "an idle stage is never asked to process");
    stage.key = 1;
    let (_, _, _, export) = image.export_level(&mut target, 8192, Some(&mut stage)).unwrap();
    assert_eq!(&export[..], &[0.0, 0.5, 1.0, 1.0]);
    assert_eq!(stage.calls, 1);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn export_scope_view_layer_coverage_and_effect_share_one_source() {
    let (d, q) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&d, 64 << 20));
    let mut target = DisplayTarget::new(8 << 20, usize::MAX);
    let mut frame = frame_with(&["R", "G", "B"]);
    frame.id = 40;
    let mut image = ImageBackend::from_frame(Arc::new(frame));
    let mut stage = Stage {
        key: 1,
        calls: 0,
        output: Vec::new(),
    };
    let (_, _, _, export) = image
        .export_level(&mut target, 8192, Some(&mut stage))
        .unwrap();
    assert_eq!(&export[..], &[0.0, 0.5, 1.0, 1.0]);
    image
        .ensure_linear(&d, &q, &mut renderer, &mut target, Some(&mut stage), None)
        .unwrap();
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut target,
            Some(&mut stage),
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(cb.direct);
    assert_eq!(stage.calls, 1);
    image.set_coverage_overlay(Some(vec![0.25]), Some(("crypto".into(), "object".into())));
    let (_, _, _, export) = image
        .export_level(&mut target, 8192, Some(&mut stage))
        .unwrap();
    assert_eq!(&export[..], &[0.125, 0.125, 0.125, 1.0]);
    assert!(
        image
            .present(
                &d,
                &q,
                &mut renderer,
                &mut target,
                Some(&mut stage),
                place(1, 1, 1.0),
                look(),
                None
            )
            .unwrap()
            .direct
    );
    image
        .ensure_linear(&d, &q, &mut renderer, &mut target, Some(&mut stage), None)
        .unwrap();
    assert_eq!(stage.calls, 2);
    image.set_coverage_overlay(None, None);
    image.set_view(&crate::image::Selection::channel("", "G"));
    let (_, _, _, export) = image
        .export_level(&mut target, 8192, Some(&mut stage))
        .unwrap();
    assert_eq!(&export[..], &[0.5, 0.5, 0.5, 1.0]);
    assert!(
        image
            .present(
                &d,
                &q,
                &mut renderer,
                &mut target,
                Some(&mut stage),
                place(1, 1, 1.0),
                look(),
                None
            )
            .unwrap()
            .direct
    );
    assert_eq!(
        image.pixel_at(0, 0),
        Some([1.0, 1.0, 1.0, 1.0]),
        "readout retains full-precision materialized input"
    );
    stage.key = 2;
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut target,
            Some(&mut stage),
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(cb.direct);
    let (_, _, _, export) = image
        .export_level(&mut target, 8192, Some(&mut stage))
        .unwrap();
    assert_eq!(&export[..], &[0.25, 0.25, 0.25, 1.0]);
    assert_eq!(stage.calls, 4);
    assert_eq!(target.scheduled_units(), 0);
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .direct
            .as_ref()
            .unwrap()
            .profiles,
        4
    );
}

struct Gate {
    flag: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}
impl Gate {
    fn new() -> Self {
        Self {
            flag: Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())),
        }
    }
    fn release(&self) {
        *self.flag.0.lock().unwrap() = true;
        self.flag.1.notify_all();
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        self.release();
    }
}
struct Delayed {
    inner: super::source::FrameSource,
    gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}
impl super::source::TileSource for Delayed {
    fn size(&self) -> (u32, u32) {
        (self.inner.w, self.inner.h)
    }
    fn unit(&self, id: super::grid::TileId) -> super::source::Unit {
        self.inner.unit(id)
    }
    fn produce(
        &self,
        g: &Grid,
        u: super::source::Unit,
        context: &super::source::ProductionContext,
    ) -> super::source::Production {
        let mut ready = self.gate.0.lock().unwrap();
        while !*ready {
            ready = self.gate.1.wait(ready).unwrap();
        }
        self.inner.produce(g, u, context)
    }
    fn scope_level(&self, _: &Grid, _: u64, _: u32) -> Option<(u32, Arc<Vec<f32>>)> {
        None
    }
}
fn draw_callback(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    r: &mut egui_wgpu::Renderer,
    callback: &super::render::TileCallback,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let gpu = r.callback_resources.get_mut::<TileGpu>().unwrap();
    gpu.pipe
        .prepare(d, q, callback.look, gpu.atlas.layout.size, &callback.list);
    let texture = target(d, crate::present::CANVAS_FORMAT, w, h);
    let view = texture.create_view(&Default::default());
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        let pages = if callback.direct {
            gpu.direct.as_ref().unwrap().pages()
        } else {
            gpu.atlas.pages()
        };
        gpu.pipe.paint(&mut pass, pages, &callback.list);
    }
    q.submit([enc.finish()]);
    halves(&read_texture(d, q, &texture, w, h, 8))
}
#[test]
#[ignore = "needs a GPU adapter"]
fn direct_to_delayed_tiles_holds_pixels_and_budget_refusal_does_not_stall() {
    let (d, q) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&d, 64 << 20));
    let mut state = DisplayTarget::new(8 << 20, usize::MAX);
    let mut f = frame_with(&["R", "G", "B"]);
    f.id = 1;
    let mut image = ImageBackend::from_frame(Arc::new(f));
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut state,
            None,
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    let old = draw_callback(&d, &q, &mut renderer, &cb, 1, 1);
    let gate = Gate::new();
    let mut next = frame_with(&["R", "G", "B"]);
    next.id = 2;
    next.file_types = Some(vec![exr_core::attr::PixelType::Float; 3]);
    image.set_frame(Arc::new(next));
    image.prepare_cpu().unwrap();
    state.install_test_source(
        &image,
        Arc::new(Delayed {
            inner: super::source::FrameSource {
                rgba: Arc::new(vec![0.75, 0.25, 0.0, 1.0]),
                w: 1,
                h: 1,
            },
            gate: gate.flag.clone(),
        }),
    );
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut state,
            None,
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(cb.direct);
    assert!(!state.can_advance(2));
    assert_eq!(
        draw_callback(&d, &q, &mut renderer, &cb, 1, 1),
        old,
        "pending overview retains previous pixels"
    );
    renderer
        .callback_resources
        .get_mut::<TileGpu>()
        .unwrap()
        .vram_budget = 56;
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut state,
            None,
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(cb.direct);
    assert!(state.can_advance(2), "budget refusal cannot trap playback");
    assert_eq!(draw_callback(&d, &q, &mut renderer, &cb, 1, 1), old);
    renderer
        .callback_resources
        .get_mut::<TileGpu>()
        .unwrap()
        .vram_budget = 64 << 20;
    gate.release();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let cb = loop {
        let cb = image
            .present(
                &d,
                &q,
                &mut renderer,
                &mut state,
                None,
                place(1, 1, 1.0),
                look(),
                None,
            )
            .unwrap();
        if !cb.direct {
            break cb;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "released overview must promote"
        );
        std::thread::yield_now();
    };
    assert!(state.can_advance(2));
    let shown = draw_callback(&d, &q, &mut renderer, &cb, 1, 1);
    let expected = crate::transfer::canvas(
        crate::transfer::display_pixel(
            [0.75, 0.25, 0.0, 1.0],
            &crate::transfer::Src::LINEAR,
            0.0,
            1.0,
            crate::transfer::mode::RGBA,
            0,
        ),
        false,
    );
    for c in 0..4 {
        assert!((shown[c] - expected[c]).abs() <= 1.0 / 255.0);
    }
    let mut next = frame_with(&["R", "G", "B"]);
    next.id = 3;
    image.set_frame(Arc::new(next));
    let cb = image
        .present(
            &d,
            &q,
            &mut renderer,
            &mut state,
            None,
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(
        cb.direct,
        "tile to resident handoff prepares direct texture"
    );
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .atlas
            .pages()
            .len(),
        0
    );
}

use super::{
    direct::DirectGpu,
    gpu_tests::{device, halves, read_texture, target},
    grid::Grid,
    reduce::level_from_rgba,
    render::{Look, TileGpu},
    view::{Placement, SourceDesc},
};
use crate::image::{DisplayTarget, ImageBackend, tests_support::frame_with};
use imath_rs::Half;
use std::sync::Arc;
fn place(w: u32, h: u32, zoom: f32) -> Placement {
    Placement {
        origin: egui::Pos2::ZERO,
        zoom,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32)),
    }
}
fn look() -> Look {
    Look {
        exposure_ev: 0.0,
        gamma: 1.0,
        channel_mode: crate::transfer::mode::RGBA,
        tonemap: 0,
        nearest: true,
        hdr: false,
        src: crate::transfer::Src::LINEAR,
    }
}
#[test]
#[ignore = "needs a GPU adapter"]
fn resident_present_bypasses_workers_and_reuses_upload() {
    let (d, q) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&d, 64 << 20));
    let mut target = DisplayTarget::new(8 << 20, usize::MAX);
    let mut first = frame_with(&["R", "G", "B"]);
    first.id = 1;
    let mut img = ImageBackend::from_frame(Arc::new(first));
    let callback = img
        .present(
            &d,
            &q,
            &mut renderer,
            &mut target,
            None,
            place(1, 1, 1.0),
            look(),
            None,
        )
        .unwrap();
    assert!(callback.direct);
    assert_eq!(target.scheduled_units(), 0);
    assert_eq!(target.tiles_pending(), 0);
    assert_eq!(target.tiles_ram().0, 0);
    assert!(target.can_advance(1));
    let gpu = renderer.callback_resources.get::<TileGpu>().unwrap();
    assert_eq!(gpu.atlas.bytes().0, 0);
    assert_eq!(gpu.direct.as_ref().unwrap().uploads, 1);
    assert_eq!(
        (gpu.direct.as_ref().unwrap().profiles, gpu.direct.as_ref().unwrap().scans),
        (1, 0),
        "a resident frame's profile comes from its channel spans, not a pixel rescan"
    );
    img.present(
        &d,
        &q,
        &mut renderer,
        &mut target,
        None,
        place(1, 1, 1.0),
        look(),
        None,
    )
    .unwrap();
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .direct
            .as_ref()
            .unwrap()
            .uploads,
        1
    );
    img.present(
        &d,
        &q,
        &mut renderer,
        &mut target,
        None,
        place(1, 1, 0.25),
        look(),
        None,
    )
    .unwrap();
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .direct
            .as_ref()
            .unwrap()
            .profiles,
        1,
        "zoom does not rescan the source"
    );
    let (level, w, h, rgba) = img
        .export_level(&mut target, d.limits().max_texture_dimension_2d, None)
        .unwrap();
    assert_eq!((level, w, h), (0, 1, 1));
    assert_eq!(&rgba[..], &[0.0, 1.0, 2.0, 1.0]);
    img.ensure_linear(&d, &q, &mut renderer, &mut target, None, None).unwrap();
    let mut next = frame_with(&["R", "G", "B"]);
    next.id = 2;
    img.set_frame(Arc::new(next));
    assert!(
        img.present(
            &d,
            &q,
            &mut renderer,
            &mut target,
            None,
            place(1, 1, 1.0),
            look(),
            None
        )
        .unwrap()
        .direct
    );
    assert_eq!(target.scheduled_units(), 0);
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .direct
            .as_ref()
            .unwrap()
            .uploads,
        2
    );
    assert_eq!(
        renderer
            .callback_resources
            .get::<TileGpu>()
            .unwrap()
            .direct
            .as_ref()
            .unwrap()
            .profiles,
        2
    );
}
/// The scopes read the resident display's original texture: one upload per
/// frame whichever of viewport and scopes asks first, no scope texture, and
/// the scopes see exactly the materialized level 0.
#[test]
#[ignore = "needs a GPU adapter"]
fn scopes_read_the_resident_original_without_a_second_upload() {
    use crate::image::tests_support::frame_of;
    use exr_core::api::ChannelData;
    let (d, q) = device();
    let mut renderer = egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer.callback_resources.insert(TileGpu::new(&d, 64 << 20));
    let mut target = DisplayTarget::new(8 << 20, usize::MAX);
    let (w, h) = (3u32, 2u32);
    let frame = |id: u64, base: f32| {
        let plane = |k: f32| ChannelData::Float((0..6).map(|i| base + k + i as f32 / 8.0).collect());
        let mut f = frame_of(w, h, vec![("B".into(), plane(0.25)), ("G".into(), plane(0.5)), ("R".into(), plane(0.75))]);
        f.id = id;
        Arc::new(f)
    };
    let direct = |r: &egui_wgpu::Renderer| {
        let gpu = r.callback_resources.get::<TileGpu>().unwrap();
        let direct = gpu.direct.as_ref().unwrap();
        (direct.uploads, direct.original_texture().cloned())
    };
    let expect = |base: f32| -> Vec<f32> {
        (0..6).flat_map(|i| {
            let v = |k: f32| base + k + i as f32 / 8.0;
            [v(0.75), v(0.5), v(0.25), 1.0]
        }).collect()
    };
    let mut img = ImageBackend::from_frame(frame(1, 0.0));
    assert!(img.present(&d, &q, &mut renderer, &mut target, None, place(w, h, 1.0), look(), None).unwrap().direct);
    let (_, sw, sh) = img.ensure_linear(&d, &q, &mut renderer, &mut target, None, None).unwrap();
    assert_eq!((sw, sh), (w, h));
    let (uploads, tex) = direct(&renderer);
    assert_eq!(uploads, 1, "the scopes reuse the display's upload");
    assert!(target.scope_texture().is_none(), "no separate scope texture");
    let read = |tex: &wgpu::Texture| bytemuck::pod_collect_to_vec::<u8, f32>(&read_texture(&d, &q, tex, w, h, 16));
    assert_eq!(read(&tex.unwrap()), expect(0.0));

    // A sequence step whose scopes run before the viewport: they upload, the
    // viewport then only reduces.
    img.set_frame(frame(2, 4.0));
    img.ensure_linear(&d, &q, &mut renderer, &mut target, None, None).unwrap();
    assert_eq!(direct(&renderer).0, 2);
    assert!(img.present(&d, &q, &mut renderer, &mut target, None, place(w, h, 1.0), look(), None).unwrap().direct);
    let (uploads, tex) = direct(&renderer);
    assert_eq!(uploads, 2, "the viewport reuses the scopes' upload");
    assert!(target.scope_texture().is_none());
    assert_eq!(read(&tex.unwrap()), expect(4.0));
}

/// An all-half frame reaches the resident display as `Rgba16Float` straight
/// from its planes (no f32 materialize), and the GPU sees exactly what the
/// f32 path gives it: the same original values and bit-identical reduced
/// levels, at several zooms.
#[test]
#[ignore = "needs a GPU adapter"]
fn half_planes_upload_matches_the_f32_path() {
    use crate::image::tests_support::frame_of;
    use exr_core::api::ChannelData;
    let (d, q) = device();
    let (w, h) = (67u32, 43u32);
    let n = (w * h) as usize;
    // Leading specials (NaN payloads, +/-Inf, subnormals, -0) so an upload that
    // canonicalizes or flushes them fails, not only one that reorders slots.
    const SPECIALS: [u16; 8] = [0x7e00, 0x7c01, 0xfe55, 0x7c00, 0xfc00, 0x0001, 0x83ff, 0x8000];
    let plane = |mul: u16, seed: u16| {
        ChannelData::Half(
            (0..n as u16)
                .map(|i| match SPECIALS.get(usize::from(i)) {
                    Some(&bits) => Half::from_bits(bits),
                    None => Half::from_bits((i.wrapping_mul(mul) % 0x7800) ^ seed),
                })
                .collect(),
        )
    };
    let mut frame = frame_of(w, h, vec![("B".into(), plane(37, 0)), ("G".into(), plane(101, 0x8000)), ("R".into(), plane(13, 0))]);
    frame.id = 9;
    let frame = Arc::new(frame);
    let rgba = crate::image::materialize_layer(&frame, &Default::default()).unwrap();
    for zoom in [1.0, 0.5, 0.125] {
        let mut renderer = egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
        renderer.callback_resources.insert(TileGpu::new(&d, 64 << 20));
        let mut target = DisplayTarget::new(8 << 20, usize::MAX);
        let mut img = ImageBackend::from_frame(Arc::clone(&frame));
        crate::timing::take();
        assert!(img.present(&d, &q, &mut renderer, &mut target, None, place(w, h, zoom), look(), None).unwrap().direct);
        let laps: Vec<_> = crate::timing::take().into_iter().map(|(what, _, _)| what).collect();
        assert!(img.shows_planes() && img.materialized_len() == 0, "shown from the planes");
        assert!(!laps.iter().any(|l| l.starts_with("materialize")), "no f32 materialize: {laps:?}");
        assert!(laps.contains(&"direct texels f16"), "{laps:?}");
        let gpu = renderer.callback_resources.get::<TileGpu>().unwrap();
        let direct = gpu.direct.as_ref().unwrap();
        assert_eq!(direct.original_format(), Some(wgpu::TextureFormat::Rgba16Float));
        assert_eq!((direct.uploads, direct.scans), (1, 0));
        let original = halves(&read_texture(&d, &q, direct.original_texture().unwrap(), w, h, 8));
        assert_eq!(
            original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rgba.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the half original holds the materialized values"
        );
        let grid = Grid::new(w, h);
        let level = grid.pick_level(zoom);
        let (lw, lh) = grid.level_size(level);
        let planes_reduced = read_texture(&d, &q, &direct.pages()[0].tex, lw, lh, 8);

        // The f32 path over the same source description and placement.
        let mut reference = DirectGpu::new(&d);
        let desc = direct.desc.clone().unwrap();
        assert!(
            reference
                .prepare(&d, &q, &gpu.pipe.page_bgl, &desc, super::direct::Texels::F32(&rgba), None, (w, h), place(w, h, zoom), 64 << 20)
                .unwrap()
        );
        assert_eq!(reference.exact(), direct.exact());
        assert_eq!(read_texture(&d, &q, &reference.pages()[0].tex, lw, lh, 8), planes_reduced, "zoom {zoom}");
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn direct_reduction_matches_clipped_f64_half_reference() {
    let (d, q) = device();
    let mut gpu = TileGpu::new(&d, 128 << 20);
    let mut direct = DirectGpu::new(&d);
    for (serial, (w, h, zoom)) in [
        (1, 1, 1.0),
        (65, 33, 1.0),
        (513, 1, 0.5),
        (1, 513, 0.5),
        (1001, 3, 0.25),
        (8192, 16, 0.0625),
        (8177, 19, 0.0625),
        (8178, 19, 0.0625),
        (8179, 21, 0.0625),
        (1100, 600, 0.25),
        (1001, 300, 0.25),
    ]
    .into_iter()
    .enumerate()
    {
        let serial = serial as u64 + 1;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[
                    ((x * 7 + y * 3) % 150) as f32 / 100.0 - 0.3,
                    if (x + y) % 2 == 0 { 0.0 } else { 1.0 },
                    0.3333,
                    1.0,
                ]);
            }
        }
        let desc = SourceDesc {
            frame: serial,
            view: Default::default(),
            pixels_gen: 0,
            stage: 0,
        };
        assert!(
            direct
                .prepare(
                    &d,
                    &q,
                    &gpu.pipe.page_bgl,
                    &desc,
                    super::direct::Texels::F32(&rgba),
                    None,
                    (w, h),
                    place(w, h, zoom),
                    128 << 20
                )
                .unwrap()
        );
        let grid = Grid::new(w, h);
        let k = grid.pick_level(zoom);
        let (rw, rh) = grid.level_size(k);
        let reference = level_from_rgba(&rgba, &grid, k).unwrap();
        let bytes = read_texture(&d, &q, &direct.pages()[0].tex, rw, rh, 8);
        for (i, (actual, want)) in bytes
            .as_chunks::<2>().0.iter()
            .zip(reference.iter().copied())
            .enumerate()
        {
            let actual = u16::from_le_bytes([actual[0], actual[1]]);
            let want = Half::from(want).bits();
            assert!(
                actual == want || (actual & 0x7fff == 0 && want & 0x7fff == 0),
                "{w}x{h} level{k} component{i}: half {actual:04x} vs {want:04x}"
            );
        }
        // The shared display pipeline consumes the same direct texture.
        for nearest in [true, false] {
            let mut current_look = look();
            current_look.nearest = nearest;
            let sampling_place = Placement {
                origin: egui::pos2(0.125, 0.125),
                ..place(rw, rh, zoom)
            };
            let list = direct.list(sampling_place, false);
            let tex = target(&d, crate::present::CANVAS_FORMAT, rw, rh);
            let view = tex.create_view(&Default::default());
            gpu.pipe.prepare(&d, &q, current_look, 512, &list);
            let mut enc = d.create_command_encoder(&Default::default());
            {
                let mut pass = enc
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    })
                    .forget_lifetime();
                gpu.pipe.paint(&mut pass, direct.pages(), &list);
            }
            q.submit([enc.finish()]);
            let shown = halves(&read_texture(&d, &q, &tex, rw, rh, 8));
            for y in 0..rh {
                for x in 0..rw {
                    let i = ((y * rw + x) * 4) as usize;
                    let inside = (x as f32 + 0.5 - 0.125) < w as f32 * zoom
                        && (y as f32 + 0.5 - 0.125) < h as f32 * zoom;
                    let expected = if inside {
                        let sx = (x as f32 + 0.5 - 0.125) / zoom / (1u32 << k) as f32;
                        let sy = (y as f32 + 0.5 - 0.125) / zoom / (1u32 << k) as f32;
                        let rgba = sample_reference(&reference, rw, rh, sx, sy, nearest);
                        crate::transfer::canvas(
                            crate::transfer::display_pixel(
                                rgba,
                                &crate::transfer::Src::LINEAR,
                                0.0,
                                1.0,
                                crate::transfer::mode::RGBA,
                                0,
                            ),
                            false,
                        )
                    } else {
                        [0.0; 4]
                    };
                    for c in 0..4 {
                        assert!(
                            (shown[i + c] - expected[c]).abs() <= 1.0 / 255.0,
                            "display {w}x{h} pixel{x},{y} c{c}: {} expected{}",
                            shown[i + c],
                            expected[c]
                        );
                    }
                }
            }
        }
    }
}
fn sample_reference(pixels: &[f32], w: u32, h: u32, x: f32, y: f32, nearest: bool) -> [f32; 4] {
    let x = x.clamp(0.5, w as f32 - 0.5);
    let y = y.clamp(0.5, h as f32 - 0.5);
    let at = |x: u32, y: u32, c: usize| Half::from(pixels[((y * w + x) * 4) as usize + c]).to_f32();
    if nearest {
        return [0, 1, 2, 3].map(|c| at(x.floor() as u32, y.floor() as u32, c));
    }
    let tx = x - 0.5;
    let ty = y - 0.5;
    let x0 = tx.floor() as u32;
    let y0 = ty.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = tx - x0 as f32;
    let fy = ty - y0 as f32;
    [0, 1, 2, 3].map(|c| {
        let a = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
        let b = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
        a * (1.0 - fy) + b * fy
    })
}

/// The n=1 shader branch must preserve staged ties-even f32 -> Half conversion,
/// both at level zero and in the one-sample corner of a clipped reduced block.
#[test]
#[ignore = "needs a GPU adapter"]
fn single_sample_half_rounding_is_exact_at_level_zero_and_clipped_edges() {
    let (d, q) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&d, 8 << 20));
    let mut state = DisplayTarget::new(8 << 20, usize::MAX);
    let inputs = [
        0x0000_0000, // both signs of zero become +0 in the ordered f64 sum
        0x3300_0000,
        0x32ff_ffff,
        0x3300_0001, // zero/minimum-subnormal midpoint
        0x3380_0000, // exact minimum Half subnormal
        0x33c0_0000,
        0x33bf_ffff,
        0x33c0_0001, // odd subnormal tie and neighbours
        0x387f_e000,
        0x387f_dfff,
        0x387f_e001, // subnormal/normal midpoint
        0x3880_0000, // exact minimum Half normal
        0x3f80_1000,
        0x3f80_0fff,
        0x3f80_1001, // even normal tie and neighbours
        0x3f80_3000,
        0x3f80_2fff,
        0x3f80_3001, // odd normal tie and neighbours
        0x477f_e000, // exact maximum finite Half
    ];
    let mut frame_id = 10_000u64;
    for input in inputs {
        let value = f32::from_bits(input);
        let components = [value, -value, 0.0, -0.0];
        assert!(
            super::precision::profile(&components).is_some(),
            "eligible input {input:08x}"
        );
        for (w, h, zoom, expected_level) in [(1u32, 1u32, 1.0, 0), (3, 1, 1.0, 0), (3, 513, 0.5, 1)]
        {
            let grid = Grid::new(w, h);
            let level = grid.pick_level(zoom);
            assert_eq!(level, expected_level);
            let scale = 1u32 << level;
            let (rw, rh) = grid.level_size(level);
            assert_eq!(
                (w - (rw - 1) * scale) * (h - (rh - 1) * scale),
                1,
                "premise: last output cell has exactly one source sample"
            );

            let mut frame = frame_with(&["R", "G", "B", "A"]);
            frame.width = w;
            frame.height = h;
            frame.id = frame_id;
            frame_id += 1;
            for (name, channel) in &mut frame.channels {
                let component = match name.as_str() {
                    "R" => 0,
                    "G" => 1,
                    "B" => 2,
                    "A" => 3,
                    _ => unreachable!(),
                };
                *channel = exr_core::api::ChannelData::Float(vec![
                    components[component];
                    (w * h) as usize
                ]);
            }
            let mut image = ImageBackend::from_frame(Arc::new(frame));
            let before = renderer
                .callback_resources
                .get::<TileGpu>()
                .unwrap()
                .direct
                .as_ref()
                .map_or(0, |direct| direct.dispatches);
            let cb = image
                .present(
                    &d,
                    &q,
                    &mut renderer,
                    &mut state,
                    None,
                    place(rw, rh, zoom),
                    look(),
                    None,
                )
                .unwrap();
            assert!(
                cb.direct,
                "{w}x{h} input {input:08x}: direct source must engage"
            );
            assert_eq!(state.scheduled_units(), 0, "no CPU tile jobs");
            let gpu = renderer.callback_resources.get::<TileGpu>().unwrap();
            let direct = gpu.direct.as_ref().unwrap();
            assert_eq!(direct.dispatches, before + 1, "real GPU reduction engaged");
            let texture = &direct.pages()[0].tex;
            assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
            let actual = read_texture(&d, &q, texture, rw, rh, 8);
            for y in 0..rh {
                for x in 0..rw {
                    let n = (w - x * scale).min(scale) * (h - y * scale).min(scale);
                    for (c, value) in components.into_iter().enumerate() {
                        // Independent CPU reference follows the ordered f64 accumulation,
                        // f32 mean, then Half. It deliberately canonicalizes input -0.
                        let mut sum = 0.0f64;
                        for _ in 0..n {
                            sum += f64::from(value);
                        }
                        let expected = Half::from((sum / f64::from(n)) as f32).bits();
                        let offset = ((y * rw + x) as usize * 4 + c) * 2;
                        let got = u16::from_le_bytes([actual[offset], actual[offset + 1]]);
                        assert_eq!(
                            got, expected,
                            "{w}x{h} level{level} input {input:08x} pixel({x},{y}) n={n} component{c}"
                        );
                    }
                }
            }
        }
    }
}

/// Encoded half planes need a fresh precision profile after stage 1: the curves can enlarge
/// their range and introduce fractional values the stored half profile cannot represent.
#[test]
#[ignore = "needs a GPU adapter"]
fn decoded_colour_reduction_profiles_linear_pixels() {
    let (d, q) = device();
    let (w, h) = (1025u32, 3u32);
    let grid = Grid::new(w, h);
    let values = [Half::from(0.11), Half::from(0.3333), Half::from(0.61), Half::from(0.95)];
    for ocio_on in [false, true] {
        for space in ["srgb_rec709_display", "pq_rec2020_display"] {
            let plane = || exr_core::api::ChannelData::Half((0..w * h).map(|i| values[(i % w % 4) as usize]).collect());
            let mut frame = crate::image::tests_support::frame_of(w, h,
                ["R", "G", "B"].into_iter().map(|name| (name.into(), plane())).collect());
            frame.tag = exr_import::ColorTag { space: space.into(), origin: exr_import::TagOrigin::File };
            let frame = Arc::new(frame);
            let mut colour = crate::ocio::State::new(crate::ocio::Sel {
                on: ocio_on, config: "ocio://studio-config-latest".into(), ..Default::default()
            });
            let mut image = ImageBackend::from_frame(frame.clone());
            let mut target = DisplayTarget::new(8 << 20, usize::MAX);
            let mut renderer = egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
            renderer.callback_resources.insert(TileGpu::new(&d, 64 << 20));
            image.colour(&mut colour, &crate::present::Target::default()).unwrap();
            let mut linear: Vec<f32> = (0..w * h).flat_map(|i| {
                let v = values[(i % w % 4) as usize].to_f32();
                [v, v, v, 1.0]
            }).collect();
            colour.decode().unwrap().apply(&mut linear);
            let callback = image.present(&d, &q, &mut renderer, &mut target, None,
                place(257, 1, 0.25), look(), None).unwrap();
            assert!(callback.direct, "{space} OCIO={ocio_on}: resident GPU path");
            let gpu = renderer.callback_resources.get::<TileGpu>().unwrap();
            let direct = gpu.direct.as_ref().unwrap();
            assert_eq!((direct.profiles, direct.scans), (1, 1),
                "{space} OCIO={ocio_on}: stored-plane profile cannot describe decoded light");
            let level = grid.pick_level(0.25);
            assert_eq!(level, 2);
            let reference = level_from_rgba(&linear, &grid, level).unwrap();
            let (rw, rh) = grid.level_size(level);
            let bytes = read_texture(&d, &q, &direct.pages()[0].tex, rw, rh, 8);
            for (i, (got, want)) in bytes.as_chunks::<2>().0.iter().zip(reference).enumerate() {
                assert_eq!(u16::from_le_bytes(*got), Half::from(want).bits(),
                    "{space} OCIO={ocio_on}: reduced component {i}");
            }
        }
    }
}
