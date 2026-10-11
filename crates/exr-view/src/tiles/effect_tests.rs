//! Actual native effects exercise render-once production and failed-owner retirement.
//! Completed output promotion reuses its owner; lazy unannounced input Pending is
//! separately retryable under the runtime input contract, without a completed output.
#[test]
#[ignore = "requires an actual display GPU; run only after the integration test stop is lifted"]
fn native_completed_replacement_waits_for_external_snapshot_without_rerendering() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release);
    FACTS.lock().unwrap().clear();
    let (device, queue) = super::gpu_tests::device();
    let mut gpu = super::render::TileGpu::new(&device, 16 << 20);
    // Measured native preparation needs 812624 bytes for this 270864-byte
    // output. Create aggregate retained pressure with another real physical
    // owner; the returned output itself is not a retained guarded copy.
    let output_bounds = OfxRectI { x1: 0, y1: -33, x2: 513, y2: 0 };
    let payload = super::processed::ChargedPixelStorage::required_bytes(output_bounds).unwrap();
    let cap = payload * 8 + 64;
    let mut state = TileState::new(cap);
    let budget = state.region_budget();
    let filler_bytes = budget.retained_limit() - payload - 16;
    assert_eq!(filler_bytes % 16, 0);
    let filler_attempt = budget.begin_attempt().unwrap();
    let filler = super::processed::ChargedPixelStorage::allocate(
        OfxRectI { x1: 0, y1: -1, x2: (filler_bytes / 16) as i32, y2: 0 },
        &filler_attempt).unwrap();
    filler_attempt.promote().unwrap();
    drop(filler_attempt);
    assert_eq!(budget.used(), filler_bytes, "actual additional retained pixels carry their own lease");
    let described = runtime.describe(&PluginKey::new(Whole::<0>::ID, 1), HostContext::Generator).unwrap();
    let effect = runtime.effect(0, &described, EffectSetup::default());
    let mut frame = frame_with(&["R", "G", "B", "A"]);
    frame.width = 513; frame.height = 33; frame.id = 17; frame.pixel_aspect = 1.5;
    frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5),
        imath_rs::V2i::new(499, 37));
    frame.display_window = frame.data_window;
    for (_, channel) in &mut frame.channels {
        *channel = exr_core::api::ChannelData::Float(vec![0.0; 513 * 33]);
    }
    let frame = Arc::new(frame);
    let source = Arc::new(EffectSource::new(effect,
        Arc::new(Inputs::Static { frame: frame.clone(), index: 0 }), frame,
        Selection::default(), 0.25, Some(24.0), Some([0.0, 0.0]), budget.clone()).unwrap());
    let old_place = super::view::Placement { origin: egui::pos2(0.0, 0.0), zoom: 1.0,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 64.0)) };
    let next_place = super::view::Placement { origin: egui::pos2(9.0, 7.0), zoom: 0.5, ..old_place };
    let draw = |state: &mut TileState, gpu: &mut super::render::TileGpu, place| {
        super::view::frame(state, &mut gpu.atlas, &device, &queue,
            &gpu.pipe.page_bgl, place, usize::MAX)
    };
    state.set_source(SourceDesc { frame: 17, view: Selection::default(),
        pixels_gen: 0, stage: 1 }, source.clone());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.accepted_pixels().is_none() {
        let _ = draw(&mut state, &mut gpu, old_place);
        assert!(std::time::Instant::now() < deadline, "first native overview never accepted");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let exported = state.accepted_pixels().unwrap();
    let old_revision = exported.revision();
    let old_sample = exported.canvas_sample(0, 0).unwrap();
    let old_charge = budget.used();
    assert_eq!(NORMAL.load(Ordering::Acquire), 1);
    assert_eq!(old_sample.level, 0, "the finest intrinsic-fit whole output is retained");
    assert_eq!(old_charge - filler_bytes, payload, "measure actual returned output survival");
    assert_eq!(old_charge, budget.retained_limit() - 16,
        "aggregate retained physical owners leave less than one output of headroom");
    assert_eq!(exported.retirement_credit(), 0, "external snapshot pins the old accepted owner");
    assert_eq!(state.accepted_retirement_credit(), 0);
    assert_eq!(exported.lineage().pixel_aspect, 1.5);
    assert_eq!(exported.lineage().time.to_bits(), 0.25f64.to_bits());
    assert_eq!(state.accepted_effect().unwrap().place.origin, old_place.origin);

    // Keep the same real Effect instance and the old exported owner alive.
    state.set_source(SourceDesc { frame: 17, view: Selection::default(),
        pixels_gen: 0, stage: 2 }, source.clone());
    state.request([TileId { level: 1, tx: 0, ty: 0 }]);
    finish(&mut state);
    assert_eq!(NORMAL.load(Ordering::Acquire), 2);
    let candidate_charge = budget.used() - old_charge;
    assert_eq!(candidate_charge, payload, "completed candidate retains one actual output");
    assert!(candidate_charge <= budget.retained_limit());
    assert!(old_charge + candidate_charge > budget.retained_limit());
    for _ in 0..8 {
        let list = draw(&mut state, &mut gpu, next_place);
        assert!(!list.instances.is_empty(), "pressure preserves the old accepted display");
        let shown = state.accepted_pixels().unwrap();
        assert_eq!(shown.revision(), old_revision);
        assert_eq!(shown.canvas_sample(64, 32).unwrap().run, old_sample.run);
        assert_eq!(state.accepted_effect().unwrap().place.origin, old_place.origin);
        assert_eq!(shown.lineage().frame_id, exported.lineage().frame_id);
        assert!(budget.used() <= cap);
        assert_eq!(NORMAL.load(Ordering::Acquire), 2, "completed W must never re-enter Render");
    }
    drop(exported);
    assert_eq!(state.accepted_retirement_credit(), payload,
        "the sole accepted snapshot credits exactly its uniquely held physical output");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let _ = draw(&mut state, &mut gpu, next_place);
        if state.accepted_pixels().is_some_and(|shown| shown.description().stage == 2) { break; }
        assert!(std::time::Instant::now() < deadline, "metadata-only policy context pinned old R");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let accepted = state.accepted_pixels().unwrap();
    let sample = accepted.canvas_sample(0, 0).unwrap();
    assert_ne!(sample.run, old_sample.run);
    assert_ne!(accepted.revision(), old_revision);
    assert_eq!(state.accepted_effect().unwrap().place.origin, next_place.origin);
    assert_eq!(NORMAL.load(Ordering::Acquire), 2);
    let before = sample.run;
    drop(accepted);
    let coarse = TileId { level: 1, tx: 0, ty: 0 };
    state.request([coarse]);
    finish(&mut state);
    let upload = state.cached(coarse).unwrap();
    assert_eq!((upload.w, upload.h), (257, 17), "coarser upload samples the original whole owner");
    drop(upload);
    assert_eq!(state.accepted_pixels().unwrap().canvas_sample(512, 32).unwrap().run, before);
    assert_eq!(NORMAL.load(Ordering::Acquire), 2);
    drop(state);
    drop(source);
    assert_eq!(budget.used(), filler_bytes, "the independent physical owner is still charged");
    drop(filler);
    assert_eq!(budget.used(), 0, "last physical owner releases R and W despite a live Runtime");
    drop(runtime);
}


#[test]
fn native_source_rejects_a_different_primary_identity_before_render_or_admission() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release);
    let described = runtime.describe(&PluginKey::new(Whole::<0>::ID, 1), HostContext::Generator).unwrap();
    let effect = runtime.effect(0, &described, EffectSetup::default());
    let mut primary = frame_with(&["R", "G", "B", "A"]);
    primary.id = 17;
    let mut other = frame_with(&["R", "G", "B", "A"]);
    other.id = 18;
    let budget = RegionBudget::new(4096);
    let result = EffectSource::new(effect,
        Arc::new(Inputs::Static { frame: Arc::new(other), index: 0 }),
        Arc::new(primary), Selection::default(), 0.25, Some(24.0),
        Some([0.0, 0.0]), budget.clone());
    assert!(matches!(result, Err(RegionError::Cancelled)));
    assert_eq!(NORMAL.load(Ordering::Acquire), 0);
    assert_eq!(budget.used(), 0);
    assert!(budget.begin_attempt().is_ok(), "rejected identity never claimed W");
}


#[test]
#[ignore = "requires an actual display GPU; run only after the integration test stop is lifted"]
fn native_mismatched_presentation_lineage_is_rejected_before_old_atlas_retirement() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release);
    FACTS.lock().unwrap().clear();
    let (device, queue) = super::gpu_tests::device();
    let mut gpu = super::render::TileGpu::new(&device, 16 << 20);
    let mut state = TileState::new(1 << 20);
    let budget = state.region_budget();
    let source = native_source(&runtime, 0, budget.clone());
    let old_desc = SourceDesc { frame: 17, view: Selection::default(),
        pixels_gen: 0, stage: 1 };
    let place = super::view::Placement { origin: egui::pos2(0.0, 0.0), zoom: 1.0,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 64.0)) };
    let draw = |state: &mut TileState, gpu: &mut super::render::TileGpu| {
        super::view::frame(state, &mut gpu.atlas, &device, &queue,
            &gpu.pipe.page_bgl, place, usize::MAX)
    };
    state.set_source(old_desc.clone(), source.clone());
    let old_generation = state.generation();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.accepted_pixels().is_none() {
        let _ = draw(&mut state, &mut gpu);
        assert!(std::time::Instant::now() < deadline, "valid native overview never accepted");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let accepted = state.accepted_pixels().unwrap();
    let revision = accepted.revision();
    let sample = accepted.canvas_sample(0, 0).unwrap();
    let overview = TileId { level: state.grid().unwrap().levels() - 1, tx: 0, ty: 0 };
    let old_key = super::view::TileKey { source: old_generation, run: sample.run, id: overview };
    assert!(gpu.atlas.resident(&old_key).is_some());
    assert_eq!(NORMAL.load(Ordering::Acquire), 1);

    // Inputs and the real EffectSource provenance remain valid. Only the
    // caller's description lies about which primary frame owns this output.
    state.set_source(SourceDesc { frame: 18, stage: 2, ..old_desc.clone() }, source.clone());
    state.request([overview]);
    finish(&mut state);
    assert_eq!(NORMAL.load(Ordering::Acquire), 2);
    for _ in 0..3 {
        let list = draw(&mut state, &mut gpu);
        assert!(!list.instances.is_empty(), "invalid candidate must hold the old display");
        assert_eq!(state.accepted_description(), Some(&old_desc));
        let shown = state.accepted_pixels().unwrap();
        assert_eq!(shown.revision(), revision);
        let shown_sample = shown.canvas_sample(0, 0).unwrap();
        assert_eq!(shown_sample.run, sample.run);
        assert_eq!(shown_sample.pixel.map(f32::to_bits), sample.pixel.map(f32::to_bits));
        assert!(gpu.atlas.resident(&old_key).is_some(),
            "preflight must refuse before destructive atlas retirement");
        assert_eq!(NORMAL.load(Ordering::Acquire), 2, "completed invalid output is not rerendered");
        assert!(budget.used() <= 1 << 20);
    }
    drop(accepted);
    drop(state);
    drop(source);
    assert_eq!(budget.used(), 0, "both valid and refused physical owners retire on last drop");
}

#[test]
fn native_large_output_preparation_and_surviving_owner_probe() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release);
    FACTS.lock().unwrap().clear();
    let described = runtime.describe(&PluginKey::new(Whole::<0>::ID, 1), HostContext::Generator).unwrap();
    let effect = runtime.effect(0, &described, EffectSetup::default());
    let mut frame = frame_with(&["R", "G", "B", "A"]);
    frame.width = 513; frame.height = 33; frame.id = 17; frame.pixel_aspect = 1.5;
    frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5),
        imath_rs::V2i::new(499, 37));
    frame.display_window = frame.data_window;
    for (_, channel) in &mut frame.channels {
        *channel = exr_core::api::ChannelData::Float(vec![0.0; 513 * 33]);
    }
    let frame = Arc::new(frame);
    let make = |budget| EffectSource::new(effect.clone(),
        Arc::new(Inputs::Static { frame: frame.clone(), index: 0 }), frame.clone(),
        Selection::default(), 0.25, Some(24.0), Some([0.0, 0.0]), budget).unwrap();
    let bounds = OfxRectI { x1: 0, y1: -33, x2: 513, y2: 0 };
    let payload = super::processed::ChargedPixelStorage::required_bytes(bounds).unwrap();
    let original = RegionBudget::new(payload * 4 + 64);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    // A partial Output must be rejected for this non-tile plug-in.
    let fine = make(original.clone());
    let refused = pool.install(|| fine.render_window(&fine.grid().unwrap(),
        TileId { level: 0, tx: 0, ty: 0 }, &ProductionContext::default()));
    eprintln!("large original-cap result {:?}, count {}, used {}",
        refused.as_ref().err(), NORMAL.load(Ordering::Acquire), original.used());
    // A deliberately partial Output is invalid for this non-tile plug-in. The
    // host rejects its declaration before pixel preparation or native Render.
    assert!(matches!(refused, Err(RegionError::Failed(_))),
        "partial non-tile Output must be rejected before Render");
    assert_eq!(NORMAL.load(Ordering::Acquire), 0, "invalid Output never enters Render");
    assert_eq!(original.used(), 0);
    // The valid whole-level request exceeds preparation capacity at level 0.
    // Actual pre-entry refusal must choose the next scale without extra Render.
    let work = pool.install(|| make(original.clone()).render_level(&ProductionContext::default())).unwrap();
    assert_eq!(work.level, 1, "choose the finest level whose actual preparation fits");
    assert_eq!(work.dimensions(), (257, 17), "ragged signed lattice rounds outward");
    assert_eq!(NORMAL.load(Ordering::Acquire), 1, "only the accepted proxy enters Render");
    assert_eq!(FACTS.lock().unwrap().as_slice(), &[(0.25, [0.5; 2], [0.0; 2], 1.5)]);
    assert_eq!(work.bytes(), original.used(), "failed finer owners have all retired");
    assert_eq!(work.budget().attempt_members(), Some(2));
    assert_eq!(work.retirement_credit(), work.bytes());
    drop(work);
    assert_eq!(original.used(), 0, "fallback keeps no previous scale allocation");
    let ample = RegionBudget::new(payload * 8 + 64);
    let work = pool.install(|| make(ample.clone()).render_level(&ProductionContext::default())).unwrap();
    eprintln!("large ample-cap survivor bytes {}, members {:?}, credit {}, level {}",
        work.bytes(), work.budget().attempt_members(), work.retirement_credit(), work.level);
    assert_eq!(NORMAL.load(Ordering::Acquire), 2);
    assert_eq!(work.level, 0);
    assert_eq!(work.bytes(), ample.used());
    assert_eq!(work.budget().attempt_members(), Some(2));
    assert_eq!(work.retirement_credit(), work.bytes());
    drop(work);
    assert_eq!(ample.used(), 0, "live Runtime/Effect retain no failed or returned pixel owner");
}

use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}};
use ofx::{Effect as NativeEffect, ImageEffectContext, Plugin, PropertySet};
use ofx::sys::*;
use ofx_runtime::{EffectSetup, PluginKey, Runtime};
use ofx_runtime::ofx_host::HostContext;
use super::{effect::EffectSource, effect_frames::Inputs, grid::TileId,
    region::{RegionBudget, RegionError}, source::{ProductionContext, TileSource, Unit},
    view::{SourceDesc, TileState}};
use crate::image::{Selection, tests_support::frame_with};

#[path = "effect_temporal_tests.rs"]
mod temporal;
#[path = "effect_identity_tests.rs"]
mod identity;

static SERIAL: Mutex<()> = Mutex::new(());
static NORMAL: AtomicUsize = AtomicUsize::new(0);
static BUDGET: AtomicUsize = AtomicUsize::new(0);
static CANCEL: AtomicUsize = AtomicUsize::new(0);
static CANCEL_FLAG: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);
// Native render time, scale, project offset, and negotiated pixel aspect.
type NativeRenderFacts = (f64, [f64; 2], [f64; 2], f64);
static FACTS: Mutex<Vec<NativeRenderFacts>> = Mutex::new(Vec::new());
struct Whole<const KIND: usize>;
impl<const KIND: usize> Plugin for Whole<KIND> {
    const ID: &'static str = match KIND {
        0 => "com.exr-view.module3.whole",
        1 => "com.exr-view.module3.render-budget",
        _ => "com.exr-view.module3.cancel",
    };
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self { Self }
    fn describe(&self, e: &NativeEffect) -> ofx::Result<()> {
        e.set_label("Module3 render counter")?;
        e.set_supported_contexts(&[kOfxImageEffectContextGenerator])?;
        e.set_supported_pixel_depths(&[kOfxBitDepthFloat])?;
        e.set_render_thread_safety(kOfxImageEffectRenderInstanceSafe)?;
        e.properties().set_cpu_render_supported(true)?;
        e.properties().set_int(kOfxImageEffectPropSupportsTiles, 0)?;
        e.properties().set_int(kOfxImageEffectPropSupportsMultiResolution, 1)
    }
    fn describe_in_context(&self, e: &NativeEffect, _: ImageEffectContext) -> ofx::Result<()> {
        let output = e.clip_define(kOfxImageEffectOutputClipName)?;
        output.set_string_at(kOfxImageEffectPropSupportedComponents, 0, "OfxImageComponentRGBA")?;
        output.set_int(kOfxImageEffectPropSupportsTiles, 0)
    }
    fn render(&self, e: &NativeEffect, a: PropertySet) -> ofx::Result<()> {
        match KIND {
            0 => { NORMAL.fetch_add(1, Ordering::AcqRel); },
            1 => { BUDGET.fetch_add(1, Ordering::AcqRel); },
            _ => { CANCEL.fetch_add(1, Ordering::AcqRel); },
        }
        let time = a.get_double(kOfxPropTime)?;
        let scale = [a.get_double_at(kOfxImageEffectPropRenderScale, 0)?,
                     a.get_double_at(kOfxImageEffectPropRenderScale, 1)?];
        let offset = [e.properties().get_double_at(kOfxImageEffectPropProjectOffset, 0)?,
                      e.properties().get_double_at(kOfxImageEffectPropProjectOffset, 1)?];
        let output = e.clip(kOfxImageEffectOutputClipName)?.get_image(time, None)?;
        FACTS.lock().unwrap().push((time, scale, offset, output.pixel_aspect_ratio()?));
        if KIND == 1 { let _must_refuse = e.image_memory_alloc(1 << 20)?; }
        if KIND == 2
            && let Some(flag) = CANCEL_FLAG.lock().unwrap().as_ref()
        {
            flag.store(true, Ordering::Release);
        }
        Ok(())
    }
}
/// This module's linked fixtures (incl. `temporal`'s); called once by
/// `OfxEffects::test_scan` before any test opens the process's plug-in registry.
pub(crate) fn register_linked() {
    ofx::linked::register::<Whole<0>>().unwrap();
    ofx::linked::register::<Whole<1>>().unwrap();
    ofx::linked::register::<Whole<2>>().unwrap();
    ofx::linked::register::<EscapedPreparation<0>>().unwrap();
    ofx::linked::register::<EscapedPreparation<64>>().unwrap();
    ofx::linked::register::<IdentityPreparation>().unwrap();
    temporal::register_linked();
    identity::register_linked();
}
fn runtime() -> Arc<Runtime<u32>> {
    Runtime::new(crate::ofx::OfxEffects::host_info(), crate::ofx::OfxEffects::test_scan())
}
fn native_source(runtime: &Arc<Runtime<u32>>, kind: usize, budget: RegionBudget) -> Arc<EffectSource> {
    let id = match kind {
        0 => Whole::<0>::ID, 1 => Whole::<1>::ID, _ => Whole::<2>::ID,
    };
    let described = runtime.describe(&PluginKey::new(id, 1), HostContext::Generator).unwrap();
    assert!(!described.tiling.tiles, "the actual native declaration is nonTile");
    let effect = runtime.effect(kind as u32, &described, EffectSetup::default());
    let mut frame = frame_with(&["R", "G", "B", "A"]);
    frame.width = 65; frame.height = 33; frame.id = 17; frame.pixel_aspect = 1.5;
    frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5),
        imath_rs::V2i::new(51, 37));
    frame.display_window = frame.data_window;
    for (_, channel) in &mut frame.channels {
        *channel = exr_core::api::ChannelData::Float(vec![0.0; 65 * 33]);
    }
    let frame = Arc::new(frame);
    let inputs = Arc::new(Inputs::Static { frame: frame.clone(), index: 0 });
    Arc::new(EffectSource::new(effect, inputs, frame, Selection::default(),
        0.25, Some(24.0), Some([0.0, 0.0]), budget).unwrap())
}
fn finish(state: &mut TileState) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.pending() != 0 {
        state.drain();
        assert!(std::time::Instant::now() < deadline, "native worker never completed");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}
#[test]
fn native_non_tile_renders_once_at_the_finest_admissible_proxy_and_reuses_that_owner() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release); FACTS.lock().unwrap().clear();
    let mut state = TileState::new(6144);
    let budget = state.region_budget();
    let source = native_source(&runtime, 0, budget.clone());
    let mut grid = source.grid().unwrap();
    grid.include_level(3).unwrap();
    let top = TileId { level: grid.levels() - 1, tx: 0, ty: 0 };
    let (w, h) = grid.level_size(2);
    let output = super::processed::ChargedPixelStorage::required_bytes(
        OfxRectI { x1: 0, y1: -(h as i32), x2: w as i32, y2: 0 }).unwrap();
    let upload = super::processed::ProcessedWork::upload_bytes(&grid, top, 2).unwrap();
    assert!(output + upload > budget.candidate_limit(),
        "the next finer proxy must fail the real output+first-upload preflight");
    assert_eq!(source.unit(TileId { level: 0, tx: 0, ty: 0 }), Unit::Level);
    assert_eq!(source.unit(TileId { level: 3, tx: 0, ty: 0 }), Unit::Level);
    state.set_source(SourceDesc { frame: 17, view: Selection::default(),
        pixels_gen: 0, stage: 1 }, source);
    state.request((0..4).map(|level| TileId { level, tx: 0, ty: 0 }));
    finish(&mut state);
    assert_eq!(NORMAL.load(Ordering::Acquire), 1);
    assert_eq!(state.scheduled_units, 1);
    assert_eq!(state.grid().unwrap().levels(), 4, "level three is the first output+upload intrinsic fit");
    let upload = state.cached(TileId { level: 3, tx: 0, ty: 0 }).unwrap();
    assert_eq!((upload.w, upload.h), (9, 5));
    drop(upload);
    state.request((0..4).map(|level| TileId { level, tx: 0, ty: 0 }));
    finish(&mut state);
    assert_eq!(NORMAL.load(Ordering::Acquire), 1, "zoom requests cannot re-render the stateful effect");
    assert_eq!(FACTS.lock().unwrap().as_slice(), &[(0.25, [0.125; 2], [0.0; 2], 1.5)]);
    eprintln!("native proxy3 physical bytes {}", budget.used());
    drop(state);
    assert_eq!(budget.used(), 0);
}
#[test]
fn render_time_budget_is_terminal_once_and_failed_output_releases_workspace_for_retry() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    BUDGET.store(0, Ordering::Release); NORMAL.store(0, Ordering::Release);
    let budget = RegionBudget::new(6144);
    let retained = budget.begin_attempt().unwrap();
    let old_pixels = retained.reserve(64).unwrap();
    retained.promote().unwrap();
    let source = native_source(&runtime, 1, budget.clone());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let result = pool.install(|| source.render_level(&ProductionContext::default()));
    assert!(matches!(result, Err(RegionError::Budget { needed: 1_048_576, limit: 6144 })),
        "Render-time fixture returned {:?}; actual Render count {}",
        result.as_ref().err(), BUDGET.load(Ordering::Acquire));
    assert_eq!(BUDGET.load(Ordering::Acquire), 1, "Render-time Budget must never invoke a coarser Render");
    assert_eq!(budget.used(), 64, "production has no last-delivered failed-output oracle retention");
    let retry = native_source(&runtime, 0, budget.clone());
    let work = pool.install(|| retry.render_level(&ProductionContext::default())).unwrap();
    assert_eq!(NORMAL.load(Ordering::Acquire), 1);
    assert_eq!(work.level, 3);
    eprintln!("native proxy3 survivor bytes {}, members {:?}, retirement credit {}",
        work.bytes(), work.budget().attempt_members(), work.retirement_credit());
    work.budget().promote().unwrap();
    assert_eq!(budget.used(), 64 + work.bytes(), "negotiated storage and its guards retain their actual physical charge");
    drop(work); drop(old_pixels); drop(retained);
    assert_eq!(budget.used(), 0);
}
#[test]
fn cancellation_after_actual_native_render_retires_only_the_failed_candidate() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime(); CANCEL.store(0, Ordering::Release);
    // Reach actual Render before cancellation: runtime preparation also admits
    // negotiated copies, which must not trigger the conservative dispatch Budget rule.
    let budget = RegionBudget::new(1 << 20);
    let retained = budget.begin_attempt().unwrap();
    let old_pixels = retained.reserve(64).unwrap();
    retained.promote().unwrap();
    let source = native_source(&runtime, 2, budget.clone());
    let context = ProductionContext::default();
    *CANCEL_FLAG.lock().unwrap() = Some(context.cancelled.clone());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let result = pool.install(|| source.render_level(&context));
    *CANCEL_FLAG.lock().unwrap() = None;
    assert!(matches!(result, Err(RegionError::Cancelled)),
        "cancelled native render returned {:?}; render count {}; cancel flag {}",
        result.as_ref().err(), CANCEL.load(Ordering::Acquire), context.is_cancelled());
    assert_eq!(CANCEL.load(Ordering::Acquire), 1);
    assert_eq!(budget.used(), 64);
    let work = pool.install(|| source.render_level(&ProductionContext::default())).unwrap();
    assert_eq!(CANCEL.load(Ordering::Acquire), 2, "same instance is retryable without hidden restart");
    drop(work); drop(old_pixels); drop(retained);
    assert_eq!(budget.used(), 0);
}


/// This fixture retains host memory across a failed preparation, until instance destruction.
struct RetainedPreparationMemory(ofx::ImageMemory);
// SAFETY: the fixture declares InstanceSafe and uses the handle only in serialized
// Create/Destroy callbacks. No locked pointer or borrowed slice crosses a callback.
unsafe impl Send for RetainedPreparationMemory {}
static ESCAPED_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ESCAPED_DESTROYS: AtomicUsize = AtomicUsize::new(0);
static ESCAPED_RENDERS: AtomicUsize = AtomicUsize::new(0);
struct EscapedPreparation<const BYTES: usize>;
impl<const BYTES: usize> Plugin for EscapedPreparation<BYTES> {
    const ID: &'static str = if BYTES == 0 {
        "com.exr-view.module3.preparation-retains-zero-request"
    } else { "com.exr-view.module3.preparation-retains-memory" };
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self { Self }
    fn describe(&self, effect: &NativeEffect) -> ofx::Result<()> {
        Whole::<0>.describe(effect)
    }
    fn describe_in_context(&self, effect: &NativeEffect, context: ImageEffectContext) -> ofx::Result<()> {
        Whole::<0>.describe_in_context(effect, context)
    }
    fn create_instance(&self, effect: &NativeEffect) -> ofx::Result<()> {
        let memory = effect.image_memory_alloc(BYTES)?;
        {
            let locked = memory.lock()?;
            // SAFETY: the host allocates at least one byte for a zero-size OFX request.
            unsafe { locked.as_ptr().cast::<u8>().write(0xa5); }
        }
        effect.set_instance_data(RetainedPreparationMemory(memory))?;
        ESCAPED_ALLOCATIONS.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
    fn destroy_instance(&self, effect: &NativeEffect) -> ofx::Result<()> {
        if let Some(memory) = effect.take_instance_data::<RetainedPreparationMemory>()? {
            {
                let locked = memory.0.lock()?;
                // SAFETY: serialized callback; this is the live byte written at creation.
                assert_eq!(unsafe { locked.as_ptr().cast::<u8>().read() }, 0xa5);
            }
            memory.0.close()?;
            ESCAPED_DESTROYS.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }
    fn render(&self, _: &NativeEffect, _: PropertySet) -> ofx::Result<()> {
        ESCAPED_RENDERS.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[test]
fn escaped_preparation_storage_keeps_original_budget_failure_and_retires_on_destroy() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    for (id, bytes) in [(EscapedPreparation::<0>::ID, 1), (EscapedPreparation::<64>::ID, 64)] {
        ESCAPED_ALLOCATIONS.store(0, Ordering::Release);
        ESCAPED_DESTROYS.store(0, Ordering::Release);
        ESCAPED_RENDERS.store(0, Ordering::Release);
        let described = runtime.describe(&PluginKey::new(id, 1), HostContext::Generator).unwrap();
        let effect = runtime.effect(bytes as u32, &described, EffectSetup::default());
        let mut frame = frame_with(&["R", "G", "B", "A"]);
        frame.width = 513; frame.height = 33; frame.id = 17;
        frame.data_window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(-13, 5),
            imath_rs::V2i::new(499, 37));
        frame.display_window = frame.data_window;
        for (_, channel) in &mut frame.channels {
            *channel = exr_core::api::ChannelData::Float(vec![0.0; 513 * 33]);
        }
        let frame = Arc::new(frame);
        let budget = RegionBudget::new(513 * 33 * 16 * 4 + 64);
        let source = EffectSource::new(effect,
            Arc::new(Inputs::Static { frame: frame.clone(), index: 0 }), frame,
            Selection::default(), 0.0, None, None, budget.clone()).unwrap();
        let result = pool.install(|| source.render_level(&ProductionContext::default()));
        assert!(matches!(result, Err(RegionError::Budget { needed, limit })
            if needed > limit && limit == budget.candidate_limit()),
            "escaped owners keep the original intrinsic failure: {:?}", result.as_ref().err());
        assert_eq!(ESCAPED_RENDERS.load(Ordering::Acquire), 0);
        assert_eq!(ESCAPED_ALLOCATIONS.load(Ordering::Acquire), 1);
        assert_eq!(ESCAPED_DESTROYS.load(Ordering::Acquire), 0);
        assert_eq!(budget.used(), bytes, "a zero OFX size still has one physical byte");
        assert!(matches!(budget.begin_attempt(), Err(RegionError::Pending)),
            "the escaped owner still claims this candidate; do not debit or fabricate retirement");
        drop(source);
        assert_eq!(budget.used(), bytes, "the live Runtime still owns this persistent Effect");
        runtime.retain(|key, _| *key != bytes as u32);
        assert_eq!(ESCAPED_DESTROYS.load(Ordering::Acquire), 1);
        assert_eq!(budget.used(), 0, "Destroy releases the last actual allocation");
        drop(budget.begin_attempt().unwrap());
    }
}

#[test]
fn zero_byte_permit_pins_candidate_and_retired_factory_cannot_charge_next_scale() {
    let budget = RegionBudget::new(1024);
    let attempt = budget.begin_attempt().unwrap();
    let factory = attempt.admission_view().unwrap();
    let empty_owner = factory.reserve(0).unwrap();
    assert_eq!(attempt.attempt_live(), Some(0));
    assert_eq!(attempt.attempt_members(), Some(2), "zero bytes do not imply owner retirement");
    drop(attempt);
    assert!(matches!(budget.begin_attempt(), Err(RegionError::Pending)));
    assert_eq!(budget.used(), 0);
    drop(empty_owner);
    let next = budget.begin_attempt().unwrap();
    assert!(matches!(factory.reserve(1), Err(RegionError::Cancelled)),
        "an expired scale factory cannot attribute late storage to the next attempt");
    assert_eq!(next.attempt_live(), Some(0));
    drop(next);
    assert_eq!(budget.used(), 0);
}

#[test]
fn transient_physical_pressure_keeps_whole_level_scale_until_capacity_returns() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    NORMAL.store(0, Ordering::Release);
    FACTS.lock().unwrap().clear();
    let budget = RegionBudget::new(1 << 20);
    let source = native_source(&runtime, 0, budget.clone());
    let occupied = budget.physical().reserve(budget.limit() - 64).unwrap();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let context = ProductionContext::default();
    context.cancelled.store(true, Ordering::Release);
    assert!(matches!(pool.install(|| source.render_level(&context)), Err(RegionError::Cancelled)),
        "cancellation outranks a currently occupied candidate or physical cap");
    context.cancelled.store(false, Ordering::Release);
    assert!(matches!(pool.install(|| source.render_level(&context)), Err(RegionError::Pending)),
        "transient pressure must not turn into a coarse fit or memoized intrinsic failure");
    assert_eq!(NORMAL.load(Ordering::Acquire), 0);
    assert!(FACTS.lock().unwrap().is_empty());
    drop(occupied);
    let work = pool.install(|| source.render_level(&context)).unwrap();
    assert_eq!(work.level, 0, "retry the original finest level after pressure is released");
    assert_eq!(NORMAL.load(Ordering::Acquire), 1);
    assert_eq!(FACTS.lock().unwrap()[0].1, [1.0; 2]);
    drop(work);
    assert_eq!(budget.used(), 0);
}


static IDENTITY_QUERIES: AtomicUsize = AtomicUsize::new(0);
static IDENTITY_BEGINS: AtomicUsize = AtomicUsize::new(0);
static IDENTITY_RENDERS: AtomicUsize = AtomicUsize::new(0);
struct IdentityPreparation;
impl Plugin for IdentityPreparation {
    const ID: &'static str = "com.exr-view.module3.identity";
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self { Self }
    fn describe(&self, effect: &NativeEffect) -> ofx::Result<()> {
        Whole::<0>.describe(effect)?;
        effect.set_supported_contexts(&[kOfxImageEffectContextFilter])
    }
    fn describe_in_context(&self, effect: &NativeEffect, _: ImageEffectContext) -> ofx::Result<()> {
        for name in [kOfxImageEffectSimpleSourceClipName, kOfxImageEffectOutputClipName] {
            let clip = effect.clip_define(name)?;
            clip.set_string_at(kOfxImageEffectPropSupportedComponents, 0, "OfxImageComponentRGBA")?;
            clip.set_int(kOfxImageEffectPropSupportsTiles, 0)?;
        }
        Ok(())
    }
    fn is_identity(&self, _: &NativeEffect, input: PropertySet, output: PropertySet) -> ofx::Result<()> {
        IDENTITY_QUERIES.fetch_add(1, Ordering::AcqRel);
        output.set_string(kOfxPropName, "Source")?;
        output.set_double(kOfxPropTime, input.get_double(kOfxPropTime)?)
    }
    fn begin_sequence_render(&self, _: &NativeEffect, _: PropertySet) -> ofx::Result<()> {
        IDENTITY_BEGINS.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
    fn render(&self, _: &NativeEffect, _: PropertySet) -> ofx::Result<()> {
        IDENTITY_RENDERS.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[test]
fn completed_identity_keeps_same_owner_through_later_upload_budget_failure() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    IDENTITY_QUERIES.store(0, Ordering::Release);
    IDENTITY_BEGINS.store(0, Ordering::Release);
    IDENTITY_RENDERS.store(0, Ordering::Release);
    let described = runtime.describe(&PluginKey::new(IdentityPreparation::ID, 1), HostContext::Filter).unwrap();
    let effect = runtime.effect(99003, &described, EffectSetup::default());
    let frame = Arc::new(frame_with(&["R", "G", "B", "A"]));
    let budget = RegionBudget::new(1 << 20);
    let source = EffectSource::new(effect,
        Arc::new(Inputs::Static { frame: frame.clone(), index: 0 }), frame,
        Selection::default(), 0.0, None, None, budget.clone()).unwrap();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let work = pool.install(|| source.render_level(&ProductionContext::default())).unwrap();
    assert_eq!(work.level, 0);
    assert_eq!(work.pixel(0, 0).unwrap().map(f32::to_bits), [0.0f32, 1.0, 2.0, 3.0].map(f32::to_bits));
    assert!(IDENTITY_QUERIES.load(Ordering::Acquire) > 0);
    assert_eq!(IDENTITY_BEGINS.load(Ordering::Acquire), 0);
    assert_eq!(IDENTITY_RENDERS.load(Ordering::Acquire), 0);
    let queries = IDENTITY_QUERIES.load(Ordering::Acquire);
    let run = work.run;
    let spare = work.budget().candidate_limit() - work.budget().attempt_live().unwrap();
    let temporary_owner = work.budget().reserve(spare).unwrap();
    let grid = source.grid().unwrap();
    let id = TileId { level: 0, tx: 0, ty: 0 };
    assert!(matches!(work.upload(&grid, id), Err(RegionError::Budget { .. })));
    assert_eq!(work.run, run);
    assert_eq!(work.level, 0);
    drop(temporary_owner);
    let upload = work.upload(&grid, id).unwrap();
    assert_eq!(IDENTITY_QUERIES.load(Ordering::Acquire), queries, "reuse completed identity output without another host request");
    assert_eq!(IDENTITY_BEGINS.load(Ordering::Acquire), 0);
    assert_eq!(IDENTITY_RENDERS.load(Ordering::Acquire), 0);
    drop(upload);
    drop(work);
    assert_eq!(budget.used(), 0);
}
