//! The pooled, channel-selective frame decode (`decode_frame` / `decode_with_refs` over the clip's `PlanePool`):
//! - every display frame of every gop / B / MV configuration decodes BIT-IDENTICAL to its lossless source (the old
//!   allocating path was bit-exact to the source too, so this is the "same output" gate), whole-chain and one-step;
//! - a channel subset decodes exactly the source's subset planes and leaves the others `None` (never zero planes),
//!   also through P/B references;
//! - a second frame of a P chain allocates no new plane bytes (`PlaneStats`: hits, no misses);
//! - `frame_refs` + `decode_with_refs` need their references (typed error otherwise).

mod common;

use std::io::Cursor;

use common::clip;
use exv_codec::Profile;
use exv_video::{Chans, Frame, VideoIndex, WriteOpts, decode_frame, decode_with_refs, frame_refs, write_clip};

/// Encode `clip(n, 16)` losslessly with `gop` / `b` / MV `search` and check every display frame + subset.
fn check(n: u32, gop: u32, b: u32, search: i32) {
    let frames = clip(n, 16);
    let opts = WriteOpts { gop, b_frames: b, search, ..WriteOpts::default() };
    let (out, _) = write_clip(Cursor::new(Vec::new()), &frames, &Profile::lossless_fast(), &opts).expect("encode");
    let bytes = out.into_inner();
    let idx = VideoIndex::parse(&bytes).expect("index");
    let n = n as usize;
    let cfg = format!("gop {gop} b {b} search {search}");
    let all: Vec<Frame> = (0..n).map(|k| decode_frame(&bytes, &idx, k, Chans::All).expect("chain decode")).collect();
    let sub: &[usize] = &[2, 0];
    for k in 0..n {
        for ci in 0..3 {
            assert_eq!(all[k].planes.get(ci), Some(frames[k].planes[ci].as_slice()), "{cfg}: frame {k} ch {ci} != source");
        }
        let refs = frame_refs(&bytes, &idx, k).expect("refs");
        let one = decode_with_refs(&bytes, &idx, k, Chans::All, |r| all.get(r).map(|f| &f.planes)).expect("one-frame decode");
        let one = one.into_image().expect("all channels");
        assert_eq!(one.planes, frames[k].planes, "{cfg}: one-step frame {k} (refs {refs:?})");
        // Channel subset: chain decode and one-step decode (refs decoded with the same subset).
        let part = decode_frame(&bytes, &idx, k, Chans::Only(sub)).expect("subset chain decode");
        assert_eq!(part.planes.get(1), None, "{cfg}: unselected channel must be None");
        for &ci in sub {
            assert_eq!(part.planes.get(ci), Some(frames[k].planes[ci].as_slice()), "{cfg}: subset frame {k} ch {ci}");
        }
        assert!(part.into_image().is_err(), "a subset frame is not a whole Image");
        // A frame that needs references cannot be decoded without them: typed error, never a guess.
        if !refs.is_empty() {
            assert!(decode_with_refs(&bytes, &idx, k, Chans::All, |_| None).is_err(), "frame {k} decoded without its references");
        }
    }
    let subs: Vec<Frame> = (0..n).map(|k| decode_frame(&bytes, &idx, k, Chans::Only(sub)).expect("subset")).collect();
    for (k, src) in frames.iter().enumerate() {
        let one = decode_with_refs(&bytes, &idx, k, Chans::Only(sub), |r| subs.get(r).map(|f| &f.planes)).expect("subset one-step");
        for &ci in sub {
            assert_eq!(one.planes.get(ci), Some(src.planes[ci].as_slice()), "{cfg}: subset one-step frame {k} ch {ci}");
        }
    }
    assert!(frame_refs(&bytes, &idx, n).is_err(), "out-of-range frame");
}

#[test]
fn p_chain_refs_match_source() {
    check(7, 4, 0, 0);
    check(5, 8, 0, 0);
}

#[test]
fn b_frames_have_two_anchors_and_match_source() {
    check(9, 8, 2, 0);
}

/// The property sweep: gop x B x global-MV search (the clip pans, so `search > 0` codes real MVs).
#[test]
fn gop_b_mv_sweep_matches_source() {
    for gop in [1, 3, 8] {
        for b in [0, 2] {
            for search in [0, 4] {
                if b > 0 && gop < 3 {
                    continue; // B needs room for anchors
                }
                check(10, gop, b, search);
            }
        }
    }
}

/// A P chain decodes into recycled planes: after the first frames warm the pool, decoding further frames takes every
/// plane from it (no miss = no new allocation, zero-fill or page fault), and a dropped frame shelves its planes.
#[test]
fn pool_reuses_planes_across_frames() {
    let frames = clip(6, 16);
    let opts = WriteOpts { gop: 8, ..WriteOpts::default() };
    let (out, _) = write_clip(Cursor::new(Vec::new()), &frames, &Profile::lossless_fast(), &opts).expect("encode");
    let bytes = out.into_inner();
    let idx = VideoIndex::parse(&bytes).expect("index");
    drop(decode_frame(&bytes, &idx, 1, Chans::All).expect("warm")); // I + P: 2 frames resident at the peak
    let warm = idx.pool().stats();
    assert!(warm.idle > 0, "a dropped frame returns its planes to the pool");
    for k in 2..6 {
        drop(decode_frame(&bytes, &idx, k, Chans::All).expect("decode"));
    }
    let after = idx.pool().stats();
    assert_eq!(after.misses, warm.misses, "warm pool: no new plane allocation ({warm:?} -> {after:?})");
    assert!(after.hits > warm.hits);
}

/// MV != 0 (shifted references) and B-frames (averaged references) take their motion-compensated planes from the clip
/// pool too, and the whole-clip readers (`read_windows`, `transcode`) recycle their decode planes: after one warm pass
/// a second pass over the same index allocates no plane (`PlaneStats.misses` unchanged), output still bit-exact.
#[test]
fn mc_refs_windows_and_transcode_reuse_pool_planes() {
    use exv_video::{read_windows, transcode};
    let frames = clip(8, 16);
    let opts = WriteOpts { gop: 4, b_frames: 1, search: 4, ..WriteOpts::default() };
    let (out, _) = write_clip(Cursor::new(Vec::new()), &frames, &Profile::lossless_fast(), &opts).expect("encode");
    let bytes = out.into_inner();
    let idx = VideoIndex::parse(&bytes).expect("index");
    let pass = || {
        for (k, want) in frames.iter().enumerate() {
            let f = decode_frame(&bytes, &idx, k, Chans::All).expect("decode");
            assert_eq!(f.planes.get(0), Some(want.planes[0].as_slice()), "frame {k}");
        }
    };
    pass();
    let warm = idx.pool().stats();
    pass();
    let after = idx.pool().stats();
    assert_eq!(after.misses, warm.misses, "MC / B references reuse pooled planes ({warm:?} -> {after:?})");
    // One GOP per window (budget below two GOPs): windows after the first decode into the planes the previous gave back.
    let one_gop = 4 * 48 * 32 * (2 + 4 + 2) as u64;
    let read = || read_windows(&bytes, &idx, one_gop).flat_map(|w| w.expect("window")).collect::<Vec<_>>();
    assert_eq!(read().iter().map(|i| &i.planes).collect::<Vec<_>>(), frames.iter().map(|i| &i.planes).collect::<Vec<_>>());
    let warm = idx.pool().stats();
    drop(read());
    let reprof = Profile::lossless_wave();
    let ip = WriteOpts { gop: 4, ..WriteOpts::default() }; // the push writers refuse B-frames
    let (t, _) = transcode(Cursor::new(Vec::new()), &bytes, &idx, &reprof, &ip, one_gop, &|i| i).expect("transcode");
    let after = idx.pool().stats();
    assert_eq!(after.misses, warm.misses, "read_windows / transcode decode into recycled planes ({warm:?} -> {after:?})");
    let back = VideoIndex::parse(t.get_ref()).expect("transcoded index");
    assert_eq!(decode_frame(t.get_ref(), &back, 7, Chans::All).expect("decode").planes.get(1), Some(frames[7].planes[1].as_slice()));
}

#[test]
fn all_intra_has_no_refs() {
    let frames = clip(3, 16);
    let (out, _) = write_clip(Cursor::new(Vec::new()), &frames, &Profile::lossless_fast(), &WriteOpts::default()).expect("encode");
    let bytes = out.into_inner();
    let idx = VideoIndex::parse(&bytes).expect("index");
    for k in 0..3 {
        assert_eq!(frame_refs(&bytes, &idx, k).expect("refs"), Vec::<usize>::new());
    }
}

/// Per-frame CPU decode cost of a viewer-shaped clip (R,G,B f16 + Z f32 + id u32 noise, 256^2 tiles, HTJ2K, all-intra),
/// interleaved A/B, median of 7 per mode: `fresh` = a pool that keeps nothing (every frame allocates, zero-fills and
/// page-faults its planes: the old path), `pooled` = the clip pool warm, `rgb` = pooled + `Chans::Only(R,G,B)`.
/// `EXV_BENCH_SIZES=4096x2048,8192x4096`. Run: `cargo test --release -p exv-video --test refs -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark; run explicitly (--release --nocapture)"]
fn frame_decode_bench() {
    use exv_codec::image::Image;
    use exv_format::{Channel, Dtype, Header};
    use std::time::Instant;
    let sizes = std::env::var("EXV_BENCH_SIZES").unwrap_or_else(|_| "4096x2048".into());
    for size in sizes.split(',') {
        let (w, h) = size.split_once('x').map(|(a, b)| (a.parse::<u32>().expect("w"), b.parse::<u32>().expect("h"))).expect("WxH");
        let n = (w * h) as usize;
        let noise = |i: usize, s: u32| (i as u32).wrapping_mul(0x9E37_79B1).rotate_left(s) ^ (i as u32 >> 3);
        let half = |s: u32| (0..n).flat_map(|i| ((noise(i, s) & 0x3bff) as u16).to_le_bytes()).collect::<Vec<u8>>();
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: 256,
            tile_h: 256,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: [("R", Dtype::F16), ("G", Dtype::F16), ("B", Dtype::F16), ("Z", Dtype::F32), ("id", Dtype::U32)]
                .iter()
                .map(|&(n, d)| Channel { name: n.into(), dtype: d })
                .collect(),
        };
        let z: Vec<u8> = (0..n).flat_map(|i| ((noise(i, 7) & 0xffff) as f32).to_le_bytes()).collect();
        let id: Vec<u8> = (0..n).flat_map(|i| (noise(i, 11) & 0xff).to_le_bytes()).collect();
        let img = Image::new(header, vec![half(1), half(2), half(3), z, id]).expect("image");
        let frames = vec![img; 3];
        let opts = WriteOpts { gop: 1, ..WriteOpts::default() };
        let (out, _) = write_clip(Cursor::new(Vec::new()), &frames, &Profile::htj2k(), &opts).expect("encode");
        let bytes = out.into_inner();
        let cold = VideoIndex::parse(&bytes).expect("index").with_pool(exv_video::PlanePool::new(exv_tile::Budget::new(0)));
        let warm = VideoIndex::parse(&bytes).expect("index");
        let rgb: &[usize] = &[0, 1, 2];
        let modes: [(&str, &VideoIndex, Chans<'_>); 3] = [("fresh", &cold, Chans::All), ("pooled", &warm, Chans::All), ("rgb", &warm, Chans::Only(rgb))];
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
        for it in 0..8 {
            for (m, (_, idx, chans)) in modes.iter().enumerate() {
                let t = Instant::now();
                drop(decode_frame(&bytes, idx, it % 3, *chans).expect("decode"));
                if it > 0 {
                    ms[m].push(t.elapsed().as_secs_f64() * 1e3); // iteration 0 warms the pool + the codec caches
                }
            }
        }
        let line: Vec<String> = modes
            .iter()
            .zip(&mut ms)
            .map(|((name, ..), v)| {
                v.sort_by(f64::total_cmp);
                format!("{name} {:.1} ms", v[v.len() / 2])
            })
            .collect();
        println!("{w}x{h} R,G,B f16 + Z f32 + id u32, HTJ2K 256^2: {} | pool {:?}", line.join(" | "), warm.pool().stats());
    }
}
