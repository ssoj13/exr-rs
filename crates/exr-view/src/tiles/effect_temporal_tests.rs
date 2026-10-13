//! Native temporal requests against the production demand lease and rewrite seam.
use super::super::effect_frames::{Fetch, FrameOwner, MAX_INPUT_FRAMES};
use super::*;
use std::sync::{atomic::AtomicU64, mpsc};
use std::time::{Duration, Instant};

type TimingFacts = (f64, f64, f64, [f64; 2], f64);
type FetchCheckpoint = (mpsc::Sender<()>, mpsc::Receiver<()>);
static REQUEST_TIME: AtomicU64 = AtomicU64::new(0);
static TIMING: Mutex<Vec<TimingFacts>> = Mutex::new(Vec::new());
static AFTER_FETCH: Mutex<Option<FetchCheckpoint>> = Mutex::new(None);

struct TemporalProbe;
impl Plugin for TemporalProbe {
    const ID: &'static str = "com.exr-view.module3.temporal-probe";
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self {
        Self
    }
    fn describe(&self, effect: &NativeEffect) -> ofx::Result<()> {
        effect.set_supported_contexts(&[kOfxImageEffectContextFilter])?;
        effect.set_supported_pixel_depths(&[kOfxBitDepthFloat])?;
        effect.set_render_thread_safety(kOfxImageEffectRenderInstanceSafe)?;
        effect.properties().set_cpu_render_supported(true)?;
        effect
            .properties()
            .set_int(kOfxImageEffectPropSupportsTiles, 0)?;
        effect
            .properties()
            .set_int(kOfxImageEffectPropSupportsMultiResolution, 1)
    }
    fn describe_in_context(&self, effect: &NativeEffect, _: ImageEffectContext) -> ofx::Result<()> {
        for name in [
            kOfxImageEffectSimpleSourceClipName,
            kOfxImageEffectOutputClipName,
        ] {
            let clip = effect.clip_define(name)?;
            clip.set_string_at(
                kOfxImageEffectPropSupportedComponents,
                0,
                "OfxImageComponentRGBA",
            )?;
            clip.set_int(kOfxImageEffectPropSupportsTiles, 0)?;
        }
        Ok(())
    }
    fn render(&self, effect: &NativeEffect, args: PropertySet) -> ofx::Result<()> {
        let time = args.get_double(kOfxPropTime)?;
        let source = effect.clip(kOfxImageEffectSimpleSourceClipName)?;
        let properties = source.properties();
        let fetch_time = f64::from_bits(REQUEST_TIME.load(Ordering::Acquire));
        TIMING.lock().unwrap().push((
            time,
            effect
                .properties()
                .get_double(kOfxImageEffectPropFrameRate)?,
            properties.get_double(kOfxImageEffectPropFrameRate)?,
            [
                properties.get_double_at(kOfxImageEffectPropFrameRange, 0)?,
                properties.get_double_at(kOfxImageEffectPropFrameRange, 1)?,
            ],
            fetch_time,
        ));
        let image = source.fetch_image(fetch_time, None)?;
        // The borrowed image remains inside this callback while the driver rewrites
        // its exact input index. No SDK pointer or frame owner crosses the channel.
        if let Some((entered, release)) = AFTER_FETCH.lock().unwrap().take() {
            entered.send(()).map_err(|_| ofx::Status::FAILED)?;
            release
                .recv_timeout(Duration::from_secs(10))
                .map_err(|_| ofx::Status::FAILED)?;
        }
        let sample = if let Some(image) = image.as_ref() {
            let plane = image.plane()?;
            let bounds = plane.bounds();
            // SAFETY: fixture requests Float RGBA, checked in-bounds CPU pixel;
            // read_unaligned also honors a host whose row layout is not aligned.
            unsafe {
                plane
                    .ptr(bounds.x1, bounds.y1)?
                    .cast::<[f32; 4]>()
                    .read_unaligned()
            }
        } else {
            [0.0; 4]
        };
        let output = effect
            .clip(kOfxImageEffectOutputClipName)?
            .get_image(time, None)?;
        let plane = output.plane()?;
        let bounds = plane.bounds();
        // SAFETY: checked Float RGBA output pixel, owned by this native Render.
        unsafe {
            plane
                .ptr(bounds.x1, bounds.y1)?
                .cast::<[f32; 4]>()
                .write_unaligned(sample);
        }
        Ok(())
    }
}

// Mirror the event loop's deferred retry wake; loader errors remain Pending until
// scancache's bounded retry policy exhausts, so a completion wake alone is insufficient.
fn wait_cache(
    wake: &mpsc::Receiver<()>,
    cache: &scancache::Scancache<crate::image::DecodedFrame>,
    deadline: Instant,
) {
    let remaining = deadline.saturating_duration_since(Instant::now());
    assert!(!remaining.is_zero(), "cache fixture deadline expired");
    let delay = cache
        .next_wake()
        .map_or(remaining, |deferred| deferred.min(remaining));
    match wake.recv_timeout(delay) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            assert!(Instant::now() < deadline, "cache fixture deadline expired");
            cache.poll();
        }
        Err(error) => panic!("cache wake closed: {error}"),
    }
}

fn ready(
    inputs: &Inputs,
    index: i32,
    wake: &mpsc::Receiver<()>,
    cache: &scancache::Scancache<crate::image::DecodedFrame>,
) -> FrameOwner {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        // Consume the coalesced wake as the UI does before checking state. Without
        // this acknowledgement a later worker completion has no new rising edge.
        cache.take_dirty();
        match inputs.fetch(index) {
            Fetch::Ready(frame) => return frame,
            Fetch::Pending => wait_cache(wake, cache, deadline),
            _ => panic!("exact known input did not reach Ready"),
        }
    }
}

/// This module's linked fixture; registered through `super::register_linked`.
pub(super) fn register_linked() {
    ofx::linked::register::<TemporalProbe>().unwrap();
}

#[test]
fn native_temporal_requests_preserve_project_timing_and_reject_rewritten_inputs() {
    let _serial = SERIAL.lock().unwrap();
    let runtime = runtime();
    let described = runtime
        .describe(&PluginKey::new(TemporalProbe::ID, 1), HostContext::Filter)
        .unwrap();
    let effect = runtime.effect(710, &described, EffectSetup::default());
    let (wake_tx, wake) = mpsc::channel();
    let (loader_entered, loader_started) = mpsc::channel();
    let (release_loader, loader_release) = mpsc::channel();
    let loader_release = Arc::new(Mutex::new(loader_release));
    let generation = Arc::new(AtomicUsize::new(41));
    let load_generation = generation.clone();
    let cache = scancache::Scancache::new(
        vec![-2, 0, 2],
        4096,
        1,
        Arc::new(crate::image::DecodedFrame::byte_size),
        Arc::new(move |index, _| {
            if index == 2 {
                return Err("temporal fixture decode failed".into());
            }
            if index == 0 {
                loader_entered.send(()).unwrap();
                loader_release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .map_err(|_| "temporal loader barrier closed".to_string())?;
            }
            let mut frame = frame_with(&["R", "G", "B", "A"]);
            frame.id = if index == -2 {
                load_generation.load(Ordering::Acquire) as u64
            } else {
                100
            };
            let value = if index == -2 {
                if frame.id == 41 { 1.0001 } else { 2.0001 }
            } else {
                3.0001
            };
            for (channel, (_, pixels)) in frame.channels.iter_mut().enumerate() {
                *pixels =
                    exr_core::api::ChannelData::Float(vec![if channel == 3 { 1.0 } else { value }]);
            }
            Ok(frame)
        }),
        Arc::new(move || {
            let _ = wake_tx.send(());
        }),
    );
    let holding = scancache::HoldingBudget::new(4096);
    let inputs = Arc::new(Inputs::Demand {
        lease: cache
            .open_demand(holding.clone(), MAX_INPUT_FRAMES)
            .unwrap(),
        primary_index: -2,
        primary_id: 41,
    });
    let primary = ready(&inputs, -2, &wake, &cache);
    let budget = RegionBudget::new(1 << 20);
    let make = |inputs: Arc<Inputs>, primary: Arc<crate::image::DecodedFrame>| {
        EffectSource::new(
            effect.clone(),
            inputs,
            primary,
            Selection::default(), std::sync::Arc::new(crate::ocio::Decoder::builtin_only()),
            -1.75,
            Some(23.976),
            Some([-2.0, 2.0]),
            budget.clone(),
        )
        .unwrap()
    };
    let source = make(inputs.clone(), cache.get(-2).unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    TIMING.lock().unwrap().clear();

    REQUEST_TIME.store((-1.75f64).to_bits(), Ordering::Release);
    let work = pool
        .install(|| source.render_level(&ProductionContext::default()))
        .unwrap();
    assert_eq!(
        work.pixel(0, 0).unwrap().map(f32::to_bits),
        [1.0001f32, 1.0001, 1.0001, 1.0].map(f32::to_bits)
    );
    assert_eq!(work.time, -1.75);
    drop(work);

    REQUEST_TIME.store((-0.1f64).to_bits(), Ordering::Release);
    let work = pool
        .install(|| source.render_level(&ProductionContext::default()))
        .unwrap();
    assert_eq!(
        work.pixel(0, 0).unwrap(),
        [0.0; 4],
        "floor(-0.1) is absent -1; a nearest present frame is forbidden"
    );
    drop(work);

    REQUEST_TIME.store(0.25f64.to_bits(), Ordering::Release);
    let pending = pool.install(|| source.render_level(&ProductionContext::default()));
    assert!(
        matches!(pending, Err(RegionError::Pending)),
        "known loading input must stay Pending: {:?}",
        pending.as_ref().err()
    );
    loader_started
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    release_loader.send(()).unwrap();
    drop(ready(&inputs, 0, &wake, &cache));
    let work = pool
        .install(|| source.render_level(&ProductionContext::default()))
        .unwrap();
    assert_eq!(
        work.pixel(0, 0).unwrap().map(f32::to_bits),
        [3.0001f32, 3.0001, 3.0001, 1.0].map(f32::to_bits)
    );
    drop(work);

    // Finish the known failed loader deterministically before asking for it natively.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        cache.take_dirty();
        match inputs.fetch(2) {
            Fetch::Failed(error) => {
                assert!(error.contains("temporal fixture decode failed"));
                break;
            }
            Fetch::Pending => wait_cache(&wake, &cache, deadline),
            _ => panic!("known failed input became missing or substituted"),
        }
    }
    REQUEST_TIME.store(2.25f64.to_bits(), Ordering::Release);
    let failed = pool.install(|| source.render_level(&ProductionContext::default()));
    assert!(matches!(&failed, Err(RegionError::Failed(error))
        if error.contains("temporal fixture decode failed")));
    assert_eq!(
        budget.used(),
        0,
        "failed/pending attempts retain no hidden pixel storage"
    );

    REQUEST_TIME.store((-1.75f64).to_bits(), Ordering::Release);
    let (fetched_tx, fetched) = mpsc::channel();
    let (release_native, native_release) = mpsc::channel();
    *AFTER_FETCH.lock().unwrap() = Some((fetched_tx, native_release));
    std::thread::scope(|scope| {
        let render =
            scope.spawn(|| pool.install(|| source.render_level(&ProductionContext::default())));
        fetched.recv_timeout(Duration::from_secs(10)).unwrap();
        generation.store(42, Ordering::Release);
        cache.invalidate(-2);
        release_native.send(()).unwrap();
        let stale = render.join().unwrap();
        assert!(
            matches!(stale, Err(RegionError::Cancelled)),
            "successful native callback cannot publish a rewritten input: {:?}",
            stale.as_ref().err()
        );
    });
    assert_eq!(budget.used(), 0);
    drop(source);
    drop(primary);
    drop(inputs);

    let new_inputs = Arc::new(Inputs::Demand {
        lease: cache
            .open_demand(holding.clone(), MAX_INPUT_FRAMES)
            .unwrap(),
        primary_index: -2,
        primary_id: 42,
    });
    let new_primary = ready(&new_inputs, -2, &wake, &cache);
    assert_eq!(new_primary.frame().id, 42);
    let new_source = make(new_inputs.clone(), cache.get(-2).unwrap());
    let work = pool
        .install(|| new_source.render_level(&ProductionContext::default()))
        .unwrap();
    assert_eq!(
        work.pixel(0, 0).unwrap().map(f32::to_bits),
        [2.0001f32, 2.0001, 2.0001, 1.0].map(f32::to_bits)
    );
    let facts = TIMING.lock().unwrap();
    assert_eq!(
        facts.iter().map(|fact| fact.4).collect::<Vec<_>>(),
        [-1.75, -0.1, 0.25, 0.25, 2.25, -1.75, -1.75],
        "every case reaches the actual native temporal fetch, exactly once"
    );
    assert!(
        facts
            .iter()
            .all(|&(time, project_fps, clip_fps, range, _)| time == -1.75
                && project_fps == 23.976
                && clip_fps == 23.976
                && range == [-2.0, 2.0]),
        "fractional native time and project/clip timing must survive every request: {facts:?}"
    );
    drop(facts);
    drop(work);
    drop(new_source);
    drop(new_primary);
    drop(new_inputs);
    assert_eq!(budget.used(), 0);
    assert_eq!(
        holding.used_bytes(),
        0,
        "rewrite releases every demand owner"
    );
}
