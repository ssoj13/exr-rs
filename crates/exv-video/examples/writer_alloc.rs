//! Writer-side motion-compensation churn (TODO P1-19): bytes allocated and wall time per encoded P / B frame of
//! `ClipWriter` / `write_clip` with the MV search on and content that really pans (so the MC reference is a shifted
//! COPY, not a borrow). Run: `cargo run --release -p exv-video --example writer_alloc` (`W=1920 H=1080 N=12`).
//! Plain `lossless_fast` keeps the codec's own churn small and constant, so the delta between two builds is the MC.

use std::io::Cursor;
use std::time::Instant;

use exv_format::bufpool::track::{TrackingAlloc, total_allocated};
use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header};
use exv_video::{ClipWriter, WriteOpts, write_clip};

#[global_allocator]
static GLOBAL: TrackingAlloc<std::alloc::System> = TrackingAlloc::new(std::alloc::System);

fn env(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// A frame whose content pans 2 px per frame (a smooth F16 ramp + a U32 id plane).
fn frame(w: u32, h: u32, t: u32) -> Image {
    let (wu, n) = (w as usize, (w * h) as usize);
    let f: Vec<u8> = (0..n).flat_map(|i| (((i % wu) as u32 + 2 * t) as u16 & 0x3fff).to_le_bytes()).collect();
    let u: Vec<u8> = (0..n).flat_map(|i| (((i % wu) as u32 + 2 * t) / 64).to_le_bytes()).collect();
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![
            Channel { name: "R".into(), dtype: Dtype::F16 },
            Channel { name: "G".into(), dtype: Dtype::F16 },
            Channel { name: "id".into(), dtype: Dtype::U32 },
        ],
    };
    Image::new(header, vec![f.clone(), f, u]).expect("img")
}

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let (w, h, n) = (env("W", 1920), env("H", 1080), env("N", 12) as usize);
    let frames: Vec<Image> = (0..n as u32).map(|t| frame(w, h, t)).collect();
    let raw: usize = frames[0].planes.iter().map(Vec::len).sum();
    let prof = Profile::lossless_fast();
    for (name, b) in [("P (push_win 4)", 0u32), ("B2 (write_clip)", 2)] {
        let opts = WriteOpts { gop: 8, search: 4, b_frames: b, ..WriteOpts::default() };
        let (mut ms, mut ax) = (Vec::new(), Vec::new());
        for run in 0..7 {
            let t0 = Instant::now();
            let a0 = total_allocated();
            if b == 0 {
                let mut cw = ClipWriter::new(Cursor::new(Vec::new()), &frames[0].header, &prof, &opts).expect("writer");
                for c in frames.chunks(4) {
                    cw.push_win(c.to_vec(), &|_| {}).expect("push");
                }
                cw.finish().expect("finish");
            } else {
                write_clip(Cursor::new(Vec::new()), &frames, &prof, &opts).expect("clip");
            }
            if run > 0 {
                ms.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
                // the output Vec and the frame clones are in both builds; the delta between builds is the MC copies
                ax.push((total_allocated() - a0) as f64 / (n * raw) as f64);
            }
        }
        println!("{name:<16} {w}x{h}: {:.1} ms/frame, {:.2}x raw allocated per frame (median of 6)", med(ms), med(ax));
    }
}
