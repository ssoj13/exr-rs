//! Frame-switch cost of a resident sequence, through the viewer's own path
//! (`ImageBackend::set_frame` + `present` on the direct display, then the
//! scopes' `ensure_linear`), attributed to stages by the `scrub-timing` laps.
//!
//! `EXR_VIEW_SCRUB_SEQ=<dir> cargo test -p exr-view --release --lib scrub_bench -- --ignored --nocapture`
//! decodes the first `EXR_VIEW_SCRUB_N` (default 20) images of `<dir>`, keeps them
//! resident like the frame cache, and switches through them for several rounds.
use super::{render::TileGpu, view::Placement};
use crate::image::{DecodedFrame, DisplayTarget, ImageBackend};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

const ROUNDS: usize = 6;

/// The viewer always hands its effect host to the display as the stage; with
/// no effect chosen it is this: key 0, pixels passed through.
struct Idle;
impl crate::image::ImageStage for Idle {
    fn key(&self) -> u64 {
        0
    }
}

/// The viewer's device: high-performance adapter, its own maximum limits.
fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a GPU adapter (this bench is #[ignore]d for machines without one)");
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    let desc = wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
        required_limits: adapter.limits(),
        ..Default::default()
    };
    pollster::block_on(adapter.request_device(&desc)).expect("device")
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore = "needs a GPU adapter and EXR_VIEW_SCRUB_SEQ"]
fn resident_frame_switch_cost() {
    let Some(dir) = std::env::var_os("EXR_VIEW_SCRUB_SEQ") else {
        println!("skipped: set EXR_VIEW_SCRUB_SEQ to a directory of sequence frames");
        return;
    };
    let n: usize = std::env::var("EXR_VIEW_SCRUB_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("readable sequence directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    paths.truncate(n);
    assert!(paths.len() >= 2, "need at least two frames in {dir:?}");

    let t = Instant::now();
    let frames: Vec<Arc<DecodedFrame>> = paths
        .iter()
        .map(|p| Arc::new(DecodedFrame::open(p, u64::MAX).expect("decodable frame")))
        .collect();
    // Load-side work (on the frame cache's workers in the viewer).
    let mut load: Vec<f64> = crate::timing::take().into_iter().map(|(_, ms, _)| ms).collect();
    println!("load-side frame spans: median {:.2} ms over {} frames", median(&mut load), load.len());
    let first = &frames[0];
    let types: Vec<_> = first
        .channels
        .iter()
        .map(|(name, c)| {
            let ty = match c {
                exr_core::api::ChannelData::Half(_) => "half",
                exr_core::api::ChannelData::Float(_) => "float",
                exr_core::api::ChannelData::Uint(_) => "uint",
            };
            format!("{name}:{ty}")
        })
        .collect();
    println!(
        "decoded {} frames {}x{} [{}] in {:.0} ms ({:.1} MB resident each)",
        frames.len(),
        first.width,
        first.height,
        types.join(" "),
        ms(t),
        first.byte_size() as f64 / 1e6
    );

    let (d, q) = device();
    let mut renderer = egui_wgpu::Renderer::new(&d, crate::present::CANVAS_FORMAT, Default::default());
    renderer.callback_resources.insert(TileGpu::new(&d, 8 << 30));
    let mut target = DisplayTarget::new(8 << 20, usize::MAX);
    // A fit-to-window zoom on a ~1200 px tall viewport, as when scrubbing.
    let (w, h) = (first.width, first.height);
    let zoom = 1200.0 / h as f32;
    let place = Placement {
        origin: egui::Pos2::ZERO,
        zoom,
        clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 * zoom, 1200.0)),
    };
    let look = super::render::Look {
        exposure_ev: 0.0,
        gamma: 1.0,
        channel_mode: crate::transfer::mode::RGBA,
        tonemap: 0,
        nearest: false,
        hdr: false,
        ocio_gain: 1.0,
    };

    let mut img = ImageBackend::from_frame(Arc::clone(first));
    // stage name -> (ms samples, bytes per switch)
    let mut stages: BTreeMap<&'static str, (Vec<f64>, u64)> = BTreeMap::new();
    let mut record = |what: &'static str, ms: f64, bytes: u64, warm: bool| {
        if warm {
            let entry = stages.entry(what).or_default();
            entry.0.push(ms);
            entry.1 = bytes;
        }
    };
    for round in 0..ROUNDS {
        // Round 0 allocates the retained buffers and textures: reported apart.
        let warm = round > 0;
        for frame in &frames {
            crate::timing::take();
            let switch = Instant::now();

            let t = Instant::now();
            img.set_frame(Arc::clone(frame));
            record("1 set_frame", ms(t), 0, warm);

            let t = Instant::now();
            let callback = img
                .present(&d, &q, &mut renderer, &mut target, Some(&mut Idle), place, look, None)
                .expect("present");
            assert!(callback.direct, "a resident frame must take the direct path");
            record("2 present (cpu)", ms(t), 0, warm);

            // What the viewer's next queue submit + the GPU then owe: the staged
            // upload's copy and the reduce dispatch.
            let t = Instant::now();
            q.submit([]);
            d.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
            record("3 gpu tail after present", ms(t), 0, warm);
            let present_total = ms(switch);

            let t = Instant::now();
            img.ensure_linear(&d, &q, &mut target, Some(&mut Idle)).expect("scope source");
            q.submit([]);
            d.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
            record("4 scopes ensure_linear + gpu", ms(t), 0, warm);

            record("switch total, no scopes", present_total, 0, warm);
            record("switch total, scopes open", ms(switch), 0, warm);
            for (what, ms, bytes) in crate::timing::take() {
                let name = match what {
                    "materialize" => "2a materialize",
                    "direct profile" => "2b direct profile",
                    "direct alloc" => "2c direct alloc",
                    "direct upload" => "2d direct upload (write_texture)",
                    "direct reduce" => "2e direct reduce (encode+submit)",
                    "present" => "2 present (lap)",
                    "scope upload" => "4a scope upload (write_texture)",
                    other => other,
                };
                record(name, ms, bytes, warm);
                if !warm {
                    println!("  cold {name}: {ms:.2} ms");
                }
            }
        }
    }
    // Split the GPU tail: the reduce alone (a level change re-reduces the
    // resident original without uploading), then bare uploads of the same frame
    // as f32 and as half texels into retained textures.
    let wait = || {
        q.submit([]);
        d.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    };
    for (i, zoom) in [zoom * 2.0, zoom].into_iter().cycle().take(2 * ROUNDS).enumerate() {
        let t = Instant::now();
        img.present(&d, &q, &mut renderer, &mut target, Some(&mut Idle), Placement { zoom, ..place }, look, None)
            .expect("present");
        let cpu = ms(t);
        wait();
        crate::timing::take();
        record("5 reduce only: present cpu", cpu, 0, i > 1);
        record("5 reduce only: present + gpu", ms(t), 0, i > 1);
    }
    for (format, texel, name_cpu, name_all) in [
        (wgpu::TextureFormat::Rgba32Float, 16u32, "6 bare upload f32: cpu", "6 bare upload f32: + gpu"),
        (wgpu::TextureFormat::Rgba16Float, 8, "7 bare upload f16: cpu", "7 bare upload f16: + gpu"),
    ] {
        let tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("scrub-bench.upload"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes = vec![0x3cu8; (w * h * texel) as usize];
        for i in 0..2 * ROUNDS {
            let t = Instant::now();
            q.write_texture(
                tex.as_image_copy(),
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * texel), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            let cpu = ms(t);
            wait();
            record(name_cpu, cpu, bytes.len() as u64, i > 1);
            record(name_all, ms(t), bytes.len() as u64, i > 1);
        }
        // The same copy through one retained MAP_WRITE staging buffer, filled
        // in place each time (what a reused staging path would cost).
        let row = (w * texel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let bytes = vec![0x3cu8; (row * h) as usize];
        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scrub-bench.staging"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let name = if texel == 16 { "8 retained staging f32: + gpu" } else { "9 retained staging f16: + gpu" };
        for i in 0..2 * ROUNDS {
            let t = Instant::now();
            staging.slice(..).map_async(wgpu::MapMode::Write, |r| r.expect("map"));
            wait();
            staging.slice(..).get_mapped_range_mut().expect("mapped").copy_from_slice(&bytes);
            staging.unmap();
            let mut enc = d.create_command_encoder(&Default::default());
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
                },
                tex.as_image_copy(),
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            q.submit([enc.finish()]);
            wait();
            record(name, ms(t), bytes.len() as u64, i > 1);
        }
    }

    let gpu = renderer.callback_resources.get::<TileGpu>().expect("tile gpu");
    let direct = gpu.direct.as_ref().expect("direct display");
    println!("exact kernel: {}", direct.exact());
    println!("{:<40} {:>9} {:>9} {:>9} {:>6}", "stage", "median ms", "min ms", "MB", "n");
    for (name, (mut samples, bytes)) in stages {
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let n = samples.len();
        println!("{name:<40} {:>9.2} {min:>9.2} {:>9.1} {n:>6}", median(&mut samples), bytes as f64 / 1e6);
    }
}
