//! The plug-in input image identity (`kOfxImagePropUniqueIdentifier`): stable for an unchanged input
//! however often it is rebuilt, different for anything that changes the pixels.
use super::*;
use crate::image::DecodedFrame;
use crate::tiles::source::ProductionContext;
use exr_core::api::ChannelData;
use imath_rs::{Box2i, V2i};

static IDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A filter that records the unique identifier of the Source image it was handed.
struct IdProbe;
impl Plugin for IdProbe {
    const ID: &'static str = "com.exr-view.identity.probe";
    const VERSION: (u32, u32) = (1, 0);
    fn new() -> Self {
        Self
    }
    fn describe(&self, effect: &NativeEffect) -> ofx::Result<()> {
        effect.set_supported_contexts(&[kOfxImageEffectContextFilter])?;
        effect.set_supported_pixel_depths(&[kOfxBitDepthFloat])?;
        effect.set_render_thread_safety(kOfxImageEffectRenderInstanceSafe)?;
        effect.properties().set_cpu_render_supported(true)?;
        effect.properties().set_int(kOfxImageEffectPropSupportsTiles, 1)?;
        effect.properties().set_int(kOfxImageEffectPropSupportsMultiResolution, 1)
    }
    fn describe_in_context(&self, effect: &NativeEffect, _: ImageEffectContext) -> ofx::Result<()> {
        for name in [kOfxImageEffectSimpleSourceClipName, kOfxImageEffectOutputClipName] {
            let clip = effect.clip_define(name)?;
            clip.set_string_at(kOfxImageEffectPropSupportedComponents, 0, "OfxImageComponentRGBA")?;
            clip.set_int(kOfxImageEffectPropSupportsTiles, 1)?;
        }
        Ok(())
    }
    fn render(&self, effect: &NativeEffect, args: PropertySet) -> ofx::Result<()> {
        let time = args.get_double(kOfxPropTime)?;
        let source = effect.clip(kOfxImageEffectSimpleSourceClipName)?.fetch_image(time, None)?;
        if let Some(image) = source {
            IDS.lock().unwrap().push(image.properties().get_string(kOfxImagePropUniqueIdentifier)?);
        }
        Ok(())
    }
}

pub(super) fn register_linked() {
    ofx::linked::register::<IdProbe>().unwrap();
}

/// A `size` x `size` EXR with channels R, G, B, A (every sample `value`).
fn write(path: &std::path::Path, size: i32, value: f32) {
    let window = Box2i::from_min_max(V2i::new(0, 0), V2i::new(size - 1, size - 1));
    let mut image = exr_core::api::Image::new(window);
    for name in ["R", "G", "B", "A"] {
        image = image.with_channel(name, ChannelData::Float(vec![value; (size * size) as usize]));
    }
    image.write(path, exr_core::Compression::NoCompression).unwrap();
}

/// The Source identifier the probe sees when `frame` is rendered at pyramid `level` through `view`.
/// `effect` is one instance throughout: the identifier names the instance too.
fn id_of(effect: &Arc<ofx_runtime::Effect<u32>>, frame: DecodedFrame, view: Selection, level: u32,
    decoder: Arc<crate::ocio::Decoder>) -> String {
    let effect = effect.clone();
    let frame = Arc::new(frame);
    let inputs = Arc::new(Inputs::Static { frame: frame.clone(), index: 0 });
    let source = EffectSource::new(effect, inputs, frame, view, decoder, 0.0, Some(24.0), Some([0.0, 0.0]),
        crate::tiles::region::RegionBudget::new(64 << 20)).unwrap();
    let grid = source.grid().unwrap();
    IDS.lock().unwrap().clear();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| source.render_window(&grid, TileId { level, tx: 0, ty: 0 }, &ProductionContext::default())).unwrap();
    let ids = IDS.lock().unwrap().clone();
    assert_eq!(ids.len(), 1, "the probe saw its Source once: {ids:?}");
    ids[0].clone()
}

/// RED: a frame rebuilt without a stamp gets a new identifier every render (the plug-in's caches never
/// hit); a stamp that ignores the layer, level or file version makes distinct pictures alias.
#[test]
fn input_identity_is_stable_for_an_unchanged_input_and_changes_with_the_pixels() {
    let _serial = SERIAL.lock().unwrap();
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("identity-{}-{unique}.exr", std::process::id()));
    write(&path, 1100, 0.25);
    let load = || DecodedFrame::load(&path).unwrap();
    let runtime = runtime();
    let described = runtime.describe(&PluginKey::new(IdProbe::ID, 1), HostContext::Filter).unwrap();
    let effect = runtime.effect(5, &described, EffectSetup::default());
    let builtin = Arc::new(crate::ocio::Decoder::builtin_only());
    let base = id_of(&effect, load(), Selection::default(), 0, builtin.clone());
    let ocio = crate::ocio::State::new(crate::ocio::Sel { on: true, ..crate::ocio::Sel::default() });
    let converted = id_of(&effect, load(), Selection::default(), 0, ocio.decoder());
    assert_ne!(base, converted, "a changed decode must invalidate plug-in input caches");
    let id_of = |frame, view, level| id_of(&effect, frame, view, level, builtin.clone());

    let base = id_of(load(), Selection::default(), 0);
    assert_eq!(base, id_of(load(), Selection::default(), 0), "a rebuilt unchanged input keeps its identity");
    assert_ne!(base, id_of(load(), Selection::channel("", "G"), 0), "another channel is other pixels");
    assert_ne!(base, id_of(load(), Selection::default(), 1), "another mip level is other pixels");

    write(&path, 1200, 0.5);
    assert_ne!(base, id_of(load(), Selection::default(), 0), "a rewritten file is other pixels");

    // No file behind the frame: unversionable, so new content every render.
    let synthetic = || crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
    assert_ne!(id_of(synthetic(), Selection::default(), 0), id_of(synthetic(), Selection::default(), 0));
    std::fs::remove_file(&path).unwrap();
}
