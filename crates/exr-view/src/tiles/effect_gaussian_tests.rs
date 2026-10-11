//! Actual Gaussian proxy evidence; the reference runs the same plug-in at the same scale.
use super::*;
use crate::image::tests_support::frame_with;
use crate::tiles::region::{RegionPlan, RegionPolicy, SamplingPhase};
use exr_core::api::ChannelData;
use imath_rs::V2i;
use ofx_runtime::ofx_host::{ClipName, HostContext, OfxRectD, ParameterOverride, ParameterValue};
use ofx_runtime::{Creation, EffectSetup, ParamStore, PluginKey, Runtime};

struct GaussianParameters;
impl ParamStore for GaussianParameters {
    fn owns(&self, name: &str) -> bool {
        matches!(name, "sigma" | "backend")
    }
    fn creation(&self, _: f64) -> Creation {
        Creation {
            overrides: vec![
                ParameterOverride {
                    name: "sigma".into(),
                    value: ParameterValue::Doubles(vec![2.0, 2.0]),
                },
                ParameterOverride {
                    name: "backend".into(),
                    value: ParameterValue::Integer(1),
                },
            ],
            ..Default::default()
        }
    }
}
fn source(effect: Arc<Effect<u32>>, primary: Arc<DecodedFrame>) -> EffectSource {
    EffectSource::new(
        effect,
        Arc::new(Inputs::Static {
            frame: primary.clone(),
            index: 0,
        }),
        primary,
        Selection::default(),
        0.75,
        Some(24.0),
        Some([0.0, 0.0]),
        RegionBudget::new(32 << 20),
    )
    .unwrap()
}
fn same_scale_whole(source: &EffectSource, level: u32) -> Arc<ProcessedWork> {
    let attempt = source.budget.begin_attempt().unwrap();
    let policy: Arc<dyn PixelAdmission> = Arc::new(HostPixels(attempt.admission_view().unwrap()));
    let work = source
        .render_bounds(
            level,
            None,
            &ProductionContext::default(),
            &attempt,
            &policy,
            &mut false,
        )
        .unwrap();
    attempt.clear_workspace();
    work
}
fn coordinates(r: OfxRectD) -> [f64; 4] {
    [r.x1, r.y1, r.x2, r.y2]
}
fn integers(r: OfxRectI) -> [i32; 4] {
    [r.x1, r.y1, r.x2, r.y2]
}

/// Independent geometry oracle only; pixel values come from the actual Gaussian.
fn verify_trace(source: &EffectSource, level: u32, output: OfxRectI) {
    let scale = 2.0_f64.powi(-(level as i32));
    let rx = (6.0 * scale / 1.5).ceil();
    let ry = (6.0 * scale).ceil();
    let mx = rx * 1.5 / scale;
    let my = ry / scale;
    if level == 2 {
        assert_eq!(
            my, 8.0,
            "rounded pixel radius exceeds a fixed six-unit halo"
        );
    }
    // Filter's default Output RoD is Source's canonical RoD. The host clips
    // the requested Output ROI to that RoD before asking Gaussian for its halo
    // (HostCore::set_output_rod); the remaining canvas is transparent.
    let expected = OfxRectD {
        x1: (f64::from(output.x1) * 1.5 / scale).max(4.5) - mx,
        y1: (f64::from(output.y1) / scale).max(3.0) - my,
        x2: (f64::from(output.x2) * 1.5 / scale).min(3075.0) + mx,
        y2: (f64::from(output.y2) / scale).min(34.0) + my,
    };
    let plans = source.trace.planned.lock().unwrap();
    assert_eq!(plans.len(), 1);
    let planned = plans[0]
        .iter()
        .find(|(clip, _)| *clip == ClipName::source())
        .unwrap()
        .1;
    assert_eq!(
        coordinates(planned),
        coordinates(expected),
        "actual GetRegionsOfInterest answer"
    );
    drop(plans);
    let full = OfxRectI {
        x1: (3.0 * scale).floor() as i32,
        y1: (3.0 * scale).floor() as i32,
        x2: (2050.0 * scale).ceil() as i32,
        y2: (34.0 * scale).ceil() as i32,
    };
    let clipped = OfxRectI {
        x1: ((expected.x1 / 1.5 * scale).floor() as i32).max(full.x1),
        y1: ((expected.y1 * scale).floor() as i32).max(full.y1),
        x2: ((expected.x2 / 1.5 * scale).ceil() as i32).min(full.x2),
        y2: ((expected.y2 * scale).ceil() as i32).min(full.y2),
    };
    let rect = super::super::grid::PixRect {
        x: (clipped.x1 - full.x1) as u32,
        y: (clipped.y1 - full.y1) as u32,
        w: (clipped.x2 - clipped.x1) as u32,
        h: (clipped.y2 - clipped.y1) as u32,
    };
    let plan = RegionPlan::on_lattice(
        (2047, 31),
        level,
        rect,
        RegionPolicy::OriginalArea,
        SamplingPhase::BOTTOM_LEFT,
        (3, 3),
    )
    .unwrap();
    let delivered = source.trace.delivered.lock().unwrap();
    assert!(
        !delivered.is_empty(),
        "actual FrameInput callbacks must deliver original providers"
    );
    for input in delivered.iter() {
        assert_eq!(input.time, 0.75);
        assert_eq!(input.scale, [scale, scale]);
        assert_eq!(
            coordinates(input.roi.unwrap()),
            coordinates(planned),
            "planned ROI reaches provider"
        );
        assert_eq!(integers(input.full), integers(full));
        assert_eq!(integers(input.bounds), integers(clipped));
        assert_eq!(
            input.plan, plan,
            "provider keeps checked clipping, signed origin and phase"
        );
        assert_eq!(input.plan.lattice_origin(), (3, 3));
        assert_eq!(input.plan.phase(), SamplingPhase::BOTTOM_LEFT);
        assert_eq!(input.par, 1.5);
        assert_eq!(input.row, (clipped.x2 - clipped.x1) * 16);
        assert_eq!(input.bytes, plan.output_bytes());
    }
}

#[test]
fn actual_gaussian_proxy_levels_preserve_roi_halos_odd_anchor_and_stitched_pixels() {
    let runtime = Runtime::new(
        crate::ofx::OfxEffects::host_info(),
        crate::ofx::OfxEffects::bundle_scan(),
    );
    let described = runtime
        .describe(
            &PluginKey::new("net.ofxrs.gaussian_blur", 1),
            HostContext::Filter,
        )
        .unwrap();
    assert!(described.tiling.tiles);
    let effect = runtime.effect(
        41,
        &described,
        EffectSetup {
            store: Some(Arc::new(GaussianParameters)),
            ..Default::default()
        },
    );
    let mut primary = frame_with(&["R", "G", "B", "A"]);
    primary.width = 2047;
    primary.height = 31;
    primary.pixel_aspect = 1.5;
    primary.data_window = Box2i::from_min_max(V2i::new(-10, 8), V2i::new(2036, 38));
    primary.display_window = Box2i::from_min_max(V2i::new(-13, 5), V2i::new(2039, 41));
    for (c, (_, channel)) in primary.channels.iter_mut().enumerate() {
        *channel = ChannelData::Float(
            (0..primary.width * primary.height)
                .map(|i| {
                    if c == 3 {
                        1.0
                    } else {
                        ((i * 13 + c as u32 * 7) % 97) as f32 / 97.0
                    }
                })
                .collect(),
        );
    }
    let primary = Arc::new(primary);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for level in [1, 2] {
        let whole_source = source(effect.clone(), primary.clone());
        let whole = pool.install(|| same_scale_whole(&whole_source, level));
        assert_eq!(whole.level, level);
        verify_trace(&whole_source, level, whole.bounds);
        let grid = whole_source.grid().unwrap();
        let (columns, rows) = grid.tiles(level);
        assert!(
            columns >= 2,
            "both proxy levels cross a real 512-pixel boundary"
        );
        assert_eq!(rows, 1);
        for tx in 0..columns {
            // Fresh source makes each metadata trace local to this one actual tile render.
            let tiled_source = source(effect.clone(), primary.clone());
            let id = TileId { level, tx, ty: 0 };
            let rect = grid.rect(id).unwrap();
            let work = pool
                .install(|| tiled_source.render_window(&grid, id, &ProductionContext::default()))
                .unwrap();
            verify_trace(&tiled_source, level, work.bounds);
            for y in 0..rect.h {
                for x in 0..rect.w {
                    let got = work.pixel(x, y).unwrap();
                    let expected = whole.pixel(rect.x + x, rect.y + y).unwrap();
                    for c in 0..4 {
                        assert!(
                            (got[c] - expected[c]).abs() <= 1e-6,
                            "level {level} seam ({},{}) channel {c}: {} vs {}",
                            rect.x + x,
                            rect.y + y,
                            got[c],
                            expected[c]
                        );
                    }
                }
            }
            drop(work);
            assert_eq!(
                tiled_source.budget.used(),
                0,
                "metadata trace retains no pixels or permits"
            );
        }
        drop(whole);
        assert_eq!(whole_source.budget.used(), 0);
    }
}
