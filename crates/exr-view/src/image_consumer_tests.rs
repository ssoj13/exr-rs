//! Consumer integration over an actual accepted native-effect mosaic.
//! Right fine deliberately fails: overscan requests it even with a clipped viewport.
//! Native Overlay V2 draw and pen callbacks use the accepted placement and PAR.
use super::*;
use crate::tiles::gpu_tests::{device, halves, read_texture, target as texture_target};
use crate::tiles::{
    grid::TileId,
    processed::ChargedPixelStorage,
    render::{Look as DisplayLook, TileCallback, TileGpu},
    view::Placement,
};
use ofx::{Effect as NativeEffect, ImageEffectContext, Plugin, PropertySet, sys::*};
use ofx_runtime::ofx_host::OfxRectI;
use ofx_runtime::PluginKey;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

const FINE: f32 = 0.25001;
const COARSE: f32 = 0.75003;
static SERIAL: Mutex<()> = Mutex::new(());
static OLD_FINE: AtomicUsize = AtomicUsize::new(0);
static OLD_COARSE: AtomicUsize = AtomicUsize::new(0);
static RIGHT_FAILED: AtomicUsize = AtomicUsize::new(0);
static NEXT: AtomicUsize = AtomicUsize::new(0);
static OVERLAY_DRAW: Mutex<Vec<(f64, [f64; 2])>> = Mutex::new(Vec::new());
type OverlayPenFact = (f64, [f64; 2], [f64; 2]);
static OVERLAY_PEN: Mutex<Vec<OverlayPenFact>> = Mutex::new(Vec::new());

struct ConsumerOverlay;
impl Plugin for ConsumerOverlay {
    const ID: &'static str = "com.exr-view.acceptance.consumer-overlay";
    const VERSION: (u32, u32) = (1, 0);
    const KIND: ofx::PluginKind = ofx::PluginKind::Interact;
    fn new() -> Self {
        Self
    }
    fn render(&self, _: &NativeEffect, _: PropertySet) -> ofx::Result<()> {
        Err(ofx::Status::REPLY_DEFAULT)
    }
    // The SDK routes these actions to the interact-specific hooks, whose
    // defaults are ReplyDefault (ofx/src/plugin.rs). A real overlay explicitly
    // accepts descriptor/instance lifetime, as the host's fake_overlay does.
    fn describe_interact(&self, _: &ofx::Interact) -> ofx::Result<()> {
        Ok(())
    }
    fn create_instance_interact(&self, _: &ofx::Interact) -> ofx::Result<()> {
        Ok(())
    }
    fn destroy_instance_interact(&self, _: &ofx::Interact) -> ofx::Result<()> {
        Ok(())
    }
    fn interact_draw(&self, interact: &ofx::Interact, args: PropertySet) -> ofx::Result<()> {
        let time = args.get_double(kOfxPropTime)?;
        let scale = [
            args.get_double_at(kOfxInteractPropPixelScale, 0)?,
            args.get_double_at(kOfxInteractPropPixelScale, 1)?,
        ];
        OVERLAY_DRAW
            .lock()
            .map_err(|_| ofx::Status::FAILED)?
            .push((time, scale));
        let draw = interact.draw_context(&args)?;
        draw.set_colour(OfxRGBAColourF {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        })?;
        draw.draw(
            kOfxDrawPrimitiveLines,
            &[
                OfxPointD { x: 384.75, y: 3.5 },
                OfxPointD { x: 384.75, y: 5.5 },
            ],
        )
    }
    fn interact_pen_motion(&self, _: &ofx::Interact, args: PropertySet) -> ofx::Result<()> {
        let time = args.get_double(kOfxPropTime)?;
        let scale = [
            args.get_double_at(kOfxInteractPropPixelScale, 0)?,
            args.get_double_at(kOfxInteractPropPixelScale, 1)?,
        ];
        let pen = [
            args.get_double_at(kOfxInteractPropPenPosition, 0)?,
            args.get_double_at(kOfxInteractPropPenPosition, 1)?,
        ];
        OVERLAY_PEN
            .lock()
            .map_err(|_| ofx::Status::FAILED)?
            .push((time, scale, pen));
        // Observing motion does not edit parameters or request a new render.
        Err(ofx::Status::REPLY_DEFAULT)
    }
}

struct ConsumerEffect;
impl Plugin for ConsumerEffect {
    const ID: &'static str = "com.exr-view.acceptance.consumer-mosaic";
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self {
        Self
    }
    fn describe(&self, e: &NativeEffect) -> ofx::Result<()> {
        e.set_label("Accepted consumer mosaic")?;
        e.set_overlay_interact_id(ConsumerOverlay::ID, ofx::OverlayVersion::V2)?;
        e.set_supported_contexts(&[kOfxImageEffectContextGenerator])?;
        e.set_supported_pixel_depths(&[kOfxBitDepthFloat])?;
        e.set_render_thread_safety(kOfxImageEffectRenderInstanceSafe)?;
        e.properties().set_cpu_render_supported(true)?;
        e.properties()
            .set_int(kOfxImageEffectPropSupportsTiles, 1)?;
        e.properties()
            .set_int(kOfxImageEffectPropSupportsMultiResolution, 1)
    }
    fn describe_in_context(&self, e: &NativeEffect, _: ImageEffectContext) -> ofx::Result<()> {
        let output = e.clip_define(kOfxImageEffectOutputClipName)?;
        output.set_string_at(
            kOfxImageEffectPropSupportedComponents,
            0,
            "OfxImageComponentRGBA",
        )
    }
    fn render(&self, e: &NativeEffect, args: PropertySet) -> ofx::Result<()> {
        let time = args.get_double(kOfxPropTime)?;
        let scale = args.get_double_at(kOfxImageEffectPropRenderScale, 0)?;
        let window = args.get_rect_i(kOfxImageEffectPropRenderWindow)?;
        if time < 1.0 && scale == 1.0 && window.x1 >= 512 {
            RIGHT_FAILED.fetch_add(1, Ordering::AcqRel);
            return Err(ofx::Status::FAILED);
        }
        let value = if time >= 1.0 {
            NEXT.fetch_add(1, Ordering::AcqRel);
            0.5
        } else if scale == 1.0 {
            OLD_FINE.fetch_add(1, Ordering::AcqRel);
            FINE
        } else {
            OLD_COARSE.fetch_add(1, Ordering::AcqRel);
            COARSE
        };
        let output = e
            .clip(kOfxImageEffectOutputClipName)?
            .get_image(time, None)?;
        let plane = output.plane()?;
        // PixelPlane validates coordinates and signed row layout; Float RGBA
        // was the sole advertised native format and this is exclusive Output.
        for y in window.y1..window.y2 {
            for x in window.x1..window.x2 {
                let ptr = plane.ptr(x, y)?.cast::<f32>();
                for (c, sample) in [value, value, value, 1.0].into_iter().enumerate() {
                    // SAFETY: live checked Float RGBA Output pixel, one callback writer.
                    unsafe {
                        ptr.add(c).write_unaligned(sample);
                    }
                }
            }
        }
        Ok(())
    }
}
/// The fixture effect's plug-in ID, for tests elsewhere.
pub(crate) const EFFECT_ID: &str = ConsumerEffect::ID;
/// This module's linked fixtures; called once by `OfxEffects::test_scan` before any test
/// opens the process's plug-in registry.
pub(crate) fn register_linked() {
    ofx::linked::register::<ConsumerOverlay>().unwrap();
    ofx::linked::register::<ConsumerEffect>().unwrap();
}
fn stage() -> crate::ofx::OfxEffects {
    let mut stage = crate::ofx::OfxEffects::with_scan(crate::ofx::OfxEffects::test_scan());
    stage.choose_for_test(PluginKey::new(ConsumerEffect::ID, 1));
    assert!(stage.active());
    stage.set_time(0.25);
    stage
}
fn look() -> DisplayLook {
    DisplayLook {
        exposure_ev: 0.0,
        gamma: 1.0,
        channel_mode: crate::transfer::mode::RGBA,
        tonemap: 0,
        nearest: true,
        hdr: false,
        ocio_gain: 1.0,
    }
}
fn place(zoom: f32, width: f32) -> Placement {
    Placement {
        origin: egui::Pos2::ZERO,
        zoom,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 7.0)),
    }
}
fn present(
    image: &mut ImageBackend,
    display: &mut DisplayTarget,
    stage: &mut crate::ofx::OfxEffects,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut egui_wgpu::Renderer,
    placement: Placement,
) -> TileCallback {
    image
        .present(
            device,
            queue,
            renderer,
            display,
            Some(stage),
            placement,
            look(),
            None,
        )
        .unwrap()
}
fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut egui_wgpu::Renderer,
    callback: &TileCallback,
) -> Vec<f32> {
    assert!(
        !callback.direct,
        "actual worker path must use TileGpu atlas"
    );
    let gpu = renderer.callback_resources.get_mut::<TileGpu>().unwrap();
    gpu.pipe.prepare(
        device,
        queue,
        callback.look,
        gpu.atlas.layout.size,
        &callback.list,
    );
    let texture = texture_target(device, crate::present::CANVAS_FORMAT, 770, 7);
    let view = texture.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder
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
        gpu.pipe.paint(&mut pass, gpu.atlas.pages(), &callback.list);
    }
    queue.submit([encoder.finish()]);
    halves(&read_texture(device, queue, &texture, 770, 7, 8))
}
fn overlay_probe(
    stage: &mut crate::ofx::OfxEffects,
    ctx: &egui::Context,
    accepted: crate::tiles::view::AcceptedEffect,
    rect: egui::Rect,
    expected_time: f64,
    expected_scale: [f64; 2],
    expected_line: [egui::Pos2; 2],
) {
    stage.accepted(Some(accepted));
    // Move outside first so repeated probes exercise actual PenMotion rather
    // than relying on whether the pump's previous center sample changed.
    let outside = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1100.0, 100.0),
        )),
        events: vec![egui::Event::PointerMoved(egui::pos2(-1.0, -1.0))],
        ..Default::default()
    };
    let mut warm_output = ctx.run_ui(outside, |ui| {
        stage.tick(ui.ctx());
        let response = ui.interact(
            rect,
            egui::Id::new("consumer-native-overlay"),
            egui::Sense::click_and_drag(),
        );
        assert!(!stage.overlay(ui.painter(), &response, rect, accepted.size));
    });
    // Shape-only probe intentionally consumes no egui font/texture pixels.
    // epaint 0.36 TexturesDelta::Drop requires explicit intentional clearing.
    warm_output.textures_delta.clear();
    OVERLAY_DRAW.lock().unwrap().clear();
    OVERLAY_PEN.lock().unwrap().clear();
    let pointer = rect.center();
    // Keep a fixed old screen rectangle during Pending: deriving it from the
    // requested PAR2 would cancel the very incorrect overlay mapping we test.
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1100.0, 100.0),
        )),
        events: vec![egui::Event::PointerMoved(pointer)],
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| {
        stage.tick(ui.ctx());
        let response = ui.interact(
            rect,
            egui::Id::new("consumer-native-overlay"),
            egui::Sense::click_and_drag(),
        );
        assert!(
            stage.takes_pointer(),
            "the real attached Overlay V2 takes viewport input"
        );
        assert!(
            !stage.overlay(ui.painter(), &response, rect, accepted.size),
            "metadata-only fixture motion never creates an effect edit"
        );
    });
    output.textures_delta.clear();
    let draws = OVERLAY_DRAW.lock().unwrap();
    assert!(!draws.is_empty(), "actual native Draw callback ran");
    for &(time, scale) in draws.iter() {
        assert_eq!(time, expected_time);
        for axis in 0..2 {
            assert!((scale[axis] - expected_scale[axis]).abs() < 1e-9);
        }
    }
    let line_matches = output.shapes.iter().any(|shape| {
        if let egui::Shape::Path(path) = &shape.shape {
            path.points.len() == 2
                && path
                    .points
                    .iter()
                    .zip(expected_line)
                    .all(|(got, expected)| got.distance(expected) < 0.001)
        } else {
            false
        }
    });
    assert!(
        line_matches,
        "actual DrawSuite emitted expected screen PathShape"
    );
    let pens = OVERLAY_PEN.lock().unwrap();
    assert!(!pens.is_empty(), "actual native PenMotion callback ran");
    for &(time, scale, position) in pens.iter() {
        assert_eq!(time, expected_time);
        for axis in 0..2 {
            assert!((scale[axis] - expected_scale[axis]).abs() < 1e-9);
        }
        let expected = [if expected_time < 1.0 { 384.75 } else { 513.0 }, 3.5];
        for axis in 0..2 {
            assert!((position[axis] - expected[axis]).abs() < 1e-6);
        }
    }
}

fn wait_for(mut poll: impl FnMut() -> bool, message: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while !poll() {
        assert!(std::time::Instant::now() < deadline, "{message}");
        std::thread::yield_now();
    }
}
struct DelayedWriter {
    release: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
}
impl DelayedWriter {
    fn finish(mut self) {
        self.release.take().unwrap().send(()).unwrap();
        self.thread.take().unwrap().join().unwrap().unwrap();
    }
}
impl Drop for DelayedWriter {
    fn drop(&mut self) {
        // Assertion unwind releases the checkpoint and joins the writer, so
        // neither RowSession nor Raw can leave admission pinned indefinitely.
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
struct PngPath(std::path::PathBuf);
impl Drop for PngPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn verify_png(path: &std::path::Path) {
    // Independent f64 ST 2084 goldens: 25.001/75.003 nits at white=100
    // encode to 24677/31424. At 257x4 the mean is (256*fine+coarse)/257;
    // cLLI stores nits*10_000. Expectations never call the production kernel.
    let file = std::fs::read(path).unwrap();
    let mut at = 8;
    let mut clli = None;
    let mut cicp = None;
    while at + 12 <= file.len() {
        let n = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
        let kind = &file[at + 4..at + 8];
        assert!(
            kind != b"sRGB" && kind != b"gAMA",
            "PQ cICP forbids conflicting SDR tags"
        );
        let data = &file[at + 8..at + 8 + n];
        if kind == b"cLLI" {
            clli = Some([
                u32::from_be_bytes(data[..4].try_into().unwrap()),
                u32::from_be_bytes(data[4..8].try_into().unwrap()),
            ]);
        }
        if kind == b"cICP" {
            cicp = Some(data.to_vec());
        }
        at += n + 12;
    }
    assert_eq!(cicp.unwrap(), [9, 16, 0, 1]);
    let [max, mean] = clli.unwrap();
    assert!(max.abs_diff(750030) <= 3, "independent cLLI peak: {max}");
    assert!(
        mean.abs_diff(251956) <= 3,
        "independent cLLI mosaic mean: {mean}"
    );
    let decoder = png::Decoder::new(std::io::Cursor::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut bytes).unwrap();
    assert_eq!((info.width, info.height), (257, 4));
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    assert_eq!(info.color_type, png::ColorType::Rgb);
    for y in 0..4usize {
        for x in 0..257usize {
            for c in 0..3usize {
                let at = (y * 257 + x) * 6 + c * 2;
                let code = u16::from_be_bytes([bytes[at], bytes[at + 1]]);
                let expected = if x == 256 { 31424 } else { 24677 };
                assert!(
                    code.abs_diff(expected) <= 1,
                    "independent PQ code ({x},{y},{c}): {code}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an actual display GPU; includes native effect and delayed PNG export"]
fn accepted_native_mosaic_drives_display_scopes_readout_and_frozen_two_pass_export() {
    let _serial = SERIAL.lock().unwrap();
    for counter in [&OLD_FINE, &OLD_COARSE, &RIGHT_FAILED, &NEXT] {
        counter.store(0, Ordering::Release);
    }
    let (device, queue) = device();
    let mut renderer =
        egui_wgpu::Renderer::new(&device, crate::present::CANVAS_FORMAT, Default::default());
    renderer
        .callback_resources
        .insert(TileGpu::new(&device, 64 << 20));
    let mut display = DisplayTarget::new(1 << 20, usize::MAX);
    let budget = display.tiles.region_budget();
    let mut frame = tests_support::frame_with(&["R", "G", "B", "A"]);
    frame.id = 61;
    frame.width = 513;
    frame.height = 7;
    frame.pixel_aspect = 1.5;
    frame.data_window =
        imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5), imath_rs::V2i::new(499, 11));
    frame.display_window = frame.data_window;
    for (_, channel) in &mut frame.channels {
        *channel = exr_core::api::ChannelData::Float(vec![0.1; 513 * 7]);
    }
    let mut image = ImageBackend::from_frame(Arc::new(frame));
    let mut stage = stage();
    let clipped = place(1.0, 768.0);
    wait_for(
        || {
            let _ = present(
                &mut image,
                &mut display,
                &mut stage,
                &device,
                &queue,
                &mut renderer,
                clipped,
            );
            display.accepted_pixels().is_some()
                && display.tiles_pending() == 0
                && OLD_FINE.load(Ordering::Acquire) > 0
                && OLD_COARSE.load(Ordering::Acquire) > 0
                && RIGHT_FAILED.load(Ordering::Acquire) > 0
        },
        "actual old mixed native presentation never accepted",
    );
    assert_eq!(OLD_FINE.load(Ordering::Acquire), 1);
    assert_eq!(OLD_COARSE.load(Ordering::Acquire), 1);
    assert_eq!(RIGHT_FAILED.load(Ordering::Acquire), 1);
    assert!(
        display
            .tiles
            .cached(TileId {
                level: 0,
                tx: 1,
                ty: 0
            })
            .is_none(),
        "deliberately failed right fine cannot enter accepted coverage"
    );
    let (fine_run, coarse_run) = {
        let pixels = display.accepted_pixels().unwrap();
        assert_eq!(pixels.lineage().pixel_aspect, 1.5);
        assert_eq!(pixels.lineage().data_window.min.x, -13);
        assert_eq!(pixels.lineage().time, 0.25);
        let fine = pixels.canvas_sample(511, 6).unwrap();
        let coarse = pixels.canvas_sample(512, 6).unwrap();
        assert_eq!(
            fine.pixel.map(f32::to_bits),
            [FINE, FINE, FINE, 1.0].map(f32::to_bits)
        );
        assert_eq!(
            coarse.pixel.map(f32::to_bits),
            [COARSE, COARSE, COARSE, 1.0].map(f32::to_bits)
        );
        assert_eq!((fine.level, coarse.level), (0, 1));
        assert_ne!(fine.run, coarse.run);
        (fine.run, coarse.run)
    };
    // This is the viewport/app accepted CanvasSample readout SSOT; original
    // ImageBackend::readout_at would read the unrelated 0.1 primary channels.
    let overlay_metadata = display.accepted_effect().unwrap();
    assert_eq!(
        (
            overlay_metadata.time,
            overlay_metadata.aspect,
            overlay_metadata.size
        ),
        (0.25, 1.5, (513, 7))
    );
    let callback = present(
        &mut image,
        &mut display,
        &mut stage,
        &device,
        &queue,
        &mut renderer,
        place(1.0, 770.0),
    );
    let drawn = draw(&device, &queue, &mut renderer, &callback);
    for (x, v) in [(100usize, FINE), (768usize, COARSE)] {
        let half = imath_rs::Half::from_f32(v).to_f32();
        let expected = crate::transfer::display_pixel(
            [half, half, half, 1.0],
            0.0,
            1.0,
            crate::transfer::mode::RGBA,
            0,
        );
        for c in 0..4 {
            assert!(
                (drawn[x * 4 + c] - expected[c]).abs() < 0.002,
                "real atlas display {x} channel {c}"
            );
        }
    }
    let (_, w, h) = image
        .ensure_linear(&device, &queue, &mut renderer, &mut display, Some(&mut stage))
        .unwrap();
    assert_eq!((w, h), (513, 7));
    let scope = display.scope.as_ref().unwrap();
    let bytes = read_texture(&device, &queue, &scope.tex, w, h, 16);
    let old_scope_key = scope.key;
    for (i, bytes) in bytes.as_chunks::<16>().0.iter().enumerate() {
        let value = if i % 513 == 512 { COARSE } else { FINE };
        for c in 0..4 {
            let actual = f32::from_ne_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap());
            let expected = if c == 3 { 1.0 } else { value };
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "linear scopes keep original f32 mosaic"
            );
        }
    }
    // TileState::reselect_presentation (tiles/view.rs) publishes a new accepted
    // revision when viewport picks change. Freeze identity after the final old
    // viewport and scope preparation, immediately beside export snapshot creation.
    let old_revision = display.accepted_pixels().unwrap().revision();
    let overlay_metadata = display.accepted_effect().unwrap();
    assert_eq!(overlay_metadata.run, old_revision);
    assert_eq!(overlay_metadata.time, 0.25);
    assert_eq!(old_scope_key.unwrap().1, old_revision);
    let old_owner_bytes = budget.used();
    let raw = crate::export::grab(&mut image, &mut display, 257, Some(&mut stage))
        .unwrap()
        .unwrap();
    assert_eq!((raw.level, raw.w, raw.h), (1, 257, 4));
    let export_scratch_bytes = budget.used() - old_owner_bytes;
    assert!(
        export_scratch_bytes > 0,
        "Raw owns admitted physical row scratch"
    );
    assert_eq!(
        display.tiles.accepted_retirement_credit(),
        0,
        "Raw pins the actual accepted owners"
    );
    let filler_bytes = budget.retained_limit() - old_owner_bytes - 16;
    assert_eq!(filler_bytes % 16, 0);
    let filler_attempt = budget.begin_attempt().unwrap();
    let filler = ChargedPixelStorage::allocate(
        OfxRectI {
            x1: 0,
            y1: -1,
            x2: (filler_bytes / 16) as i32,
            y2: 0,
        },
        &filler_attempt,
    )
    .unwrap();
    filler_attempt.promote().unwrap();
    drop(filler_attempt);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = PngPath(std::env::temp_dir().join(format!(
        "accepted-consumers-{}-{unique}.png",
        std::process::id()
    )));
    let (entered, ready) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let output_path = path.0.clone();
    let writer = DelayedWriter {
        release: Some(release),
        thread: Some(std::thread::spawn(move || {
            crate::export::set_statistics_checkpoint(Box::new(move || {
                let _ = entered.send(());
                let _ = blocked.recv();
            }));
            let look = crate::export::Look {
                white: 100.0,
                ..Default::default()
            };
            crate::export::write_hdr_png(&output_path, &raw, &look, crate::export::HdrPng::Pq)
        })),
    };
    ready
        .recv_timeout(std::time::Duration::from_secs(15))
        .unwrap();
    let overlay_context = egui::Context::default();
    let old_overlay_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(769.5, 7.0));
    let old_overlay_line = [egui::pos2(384.75, 3.5), egui::pos2(384.75, 1.5)];
    overlay_probe(
        &mut stage,
        &overlay_context,
        overlay_metadata,
        old_overlay_rect,
        0.25,
        [1.0, 1.0],
        old_overlay_line,
    );
    let before_candidate = budget.used();
    assert_eq!(
        before_candidate,
        old_owner_bytes + export_scratch_bytes + filler_bytes
    );
    // PixelRows uses AcceptedPixels::budget().physical(): RegionBudget::reserve
    // charges physical used bytes without retained attempt membership. Filler
    // leaves 16 retained bytes; export scratch consumes the separate physical
    // capacity, with ample headroom for the real native candidate preparation.
    assert!(before_candidate + 3 * 257 * 4 * 16 < budget.limit());
    stage.set_time(1.25);
    let mut next = tests_support::frame_with(&["R", "G", "B", "A"]);
    next.id = 62;
    next.width = 513;
    next.height = 7;
    next.pixel_aspect = 2.0;
    next.data_window =
        imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5), imath_rs::V2i::new(499, 11));
    next.display_window = next.data_window;
    for (_, channel) in &mut next.channels {
        *channel = exr_core::api::ChannelData::Float(vec![0.2; 513 * 7]);
    }
    image.set_frame(Arc::new(next));
    let next_place = place(0.5, 513.0);
    wait_for(
        || {
            let _ = present(
                &mut image,
                &mut display,
                &mut stage,
                &device,
                &queue,
                &mut renderer,
                next_place,
            );
            NEXT.load(Ordering::Acquire) == 1 && display.tiles_pending() == 0
        },
        "native replacement did not complete under retained filler pressure",
    );
    assert_eq!(
        budget.used() - before_candidate,
        257 * 4 * 16,
        "the completed replacement owns one actual Float RGBA allocation"
    );
    let frozen_count = NEXT.load(Ordering::Acquire);
    for _ in 0..8 {
        let callback = present(
            &mut image,
            &mut display,
            &mut stage,
            &device,
            &queue,
            &mut renderer,
            next_place,
        );
        assert!(!callback.list.instances.is_empty());
        let pixels = display.accepted_pixels().unwrap();
        assert_eq!(pixels.revision(), old_revision);
        assert_eq!(pixels.canvas_sample(511, 6).unwrap().run, fine_run);
        assert_eq!(pixels.canvas_sample(512, 6).unwrap().run, coarse_run);
        assert_eq!(
            display.accepted_effect().unwrap().time,
            overlay_metadata.time,
            "accepted overlay metadata holds old time until drawable promotion"
        );
        assert_eq!(pixels.retirement_credit(), 0);
        assert_eq!(
            NEXT.load(Ordering::Acquire),
            frozen_count,
            "completed native candidate cannot rerender"
        );
    }
    let held_overlay = display.accepted_effect().unwrap();
    assert_eq!((held_overlay.time, held_overlay.aspect), (0.25, 1.5));
    overlay_probe(
        &mut stage,
        &overlay_context,
        held_overlay,
        old_overlay_rect,
        0.25,
        [1.0, 1.0],
        old_overlay_line,
    );
    let (_, w, h) = image
        .ensure_linear(&device, &queue, &mut renderer, &mut display, Some(&mut stage))
        .unwrap();
    assert_eq!((w, h), (513, 7));
    let held_scope = display.scope.as_ref().unwrap();
    assert_eq!(
        held_scope.key, old_scope_key,
        "pending replacement keeps old accepted scope revision"
    );
    assert_eq!(
        read_texture(&device, &queue, &held_scope.tex, w, h, 16),
        bytes,
        "actual scope GPU pixels remain old during completed replacement pressure"
    );
    writer.finish(); // Both PNG passes finish from the same old Raw before its owners retire.
    verify_png(&path.0);
    assert_eq!(
        display.tiles.accepted_retirement_credit(),
        old_owner_bytes,
        "after Raw drops exactly the old physical allocations become retirement credit"
    );
    wait_for(
        || {
            let _ = present(
                &mut image,
                &mut display,
                &mut stage,
                &device,
                &queue,
                &mut renderer,
                next_place,
            );
            display
                .accepted_effect()
                .is_some_and(|accepted| accepted.time == 1.25)
        },
        "Raw release did not permit metadata-only completed-owner promotion",
    );
    assert_eq!(NEXT.load(Ordering::Acquire), frozen_count);
    let next_overlay = display.accepted_effect().unwrap();
    assert_eq!((next_overlay.time, next_overlay.aspect), (1.25, 2.0));
    assert_eq!(display.accepted_pixels().unwrap().lineage().frame_id, 62);
    overlay_probe(
        &mut stage,
        &overlay_context,
        next_overlay,
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(513.0, 3.5)),
        1.25,
        [2.0, 2.0],
        [egui::pos2(192.375, 1.75), egui::pos2(192.375, 0.75)],
    );
    assert_ne!(display.accepted_pixels().unwrap().revision(), old_revision);
    drop(display);
    assert_eq!(
        budget.used(),
        filler_bytes,
        "only independent retained filler survives"
    );
    drop(filler);
    assert_eq!(budget.used(), 0);
}
