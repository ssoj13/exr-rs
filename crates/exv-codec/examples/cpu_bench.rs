//! CPU hot-path micro-bench (docs/OPT.md "CPU numbers"). Run:
//! `RAYON_NUM_THREADS=1 cargo run --release -p exv-codec --example cpu_bench [filter]`
//!
//! Why an example and not criterion: no heavy deps, bit-identical inputs between before/after runs, and the
//! MEDIAN of N repeats (other processes make single timings useless). Every row prints median ms and MB/s over
//! the RAW bytes the stage handles. `filter` = substring of the row name.

use std::time::Instant;

use exv_codec::{Profile, TileBuf, decode_tile, decode_tile_into, encode_tile, reorder, temporal, wavelet};
use exv_format::Dtype;

/// `(median, min)` wall time (ms) of `reps` runs of `f` (after one warm-up). The min is the least noisy figure
/// on a machine shared with compilers; the median is the honest typical one.
fn med(reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[t.len() / 2], t[0])
}

/// xorshift32 — deterministic noise so the inputs never change between runs.
fn rng(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

/// The `exv gen` synthetic tile (R,G,B f16 ramps + Z f32 + id u32), optionally with `noise` LSBs added to
/// the f16/Z channels so entropy coders face real work.
fn tile(w: u32, h: u32, noise: u32) -> (Vec<Vec<u8>>, Vec<Dtype>) {
    let n = (w * h) as usize;
    let mut s = 0x1234_5678u32;
    let mut f16 = |seed: u32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let x = i as u32 % w;
                let y = i as u32 / w;
                let nz = if noise > 0 { rng(&mut s) % noise } else { 0 };
                let v = (((x * 3 + y * 5 + seed * 97) & 0x3ff) + nz) as u16;
                v.to_le_bytes()
            })
            .collect()
    };
    let (r, g, b) = (f16(0), f16(1), f16(2));
    let z: Vec<u8> = (0..n)
        .flat_map(|i| ((i as u32 % w) as f32 * 0.1 + (rng(&mut s) % (noise + 1)) as f32 * 1e-3).to_le_bytes())
        .collect();
    let id: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
    (vec![r, g, b, z, id], vec![Dtype::F16, Dtype::F16, Dtype::F16, Dtype::F32, Dtype::U32])
}

/// Whole-frame decode rows (`frame.*`, all rayon threads): a synthetic `S`x`S` frame (`EXV_FRAME_SIZES`, default
/// `4096`; e.g. `4096,8192`) with 256^2 tiles, coded once per profile. `fresh` = [`FrameReader::decode_all`] (new
/// planes every frame: allocation + first touch); `alloc.*` isolates the plane allocation itself.
fn frame_rows(want: impl Fn(&str) -> bool, row: impl Fn(&str, (f64, f64), usize)) {
    use exv_codec::frame::{Chans, FrameReader, Planes, write_frame_tiles};
    use exv_format::{Channel, DecodeLimits, Header};
    let sizes = std::env::var("EXV_FRAME_SIZES").unwrap_or_else(|_| "4096".into());
    for s in sizes.split(',').filter_map(|v| v.trim().parse::<u32>().ok()) {
        if !["alloc.zeroed", "frame.fresh.htj2k", "frame.fresh.fast"].iter().any(|n| want(&format!("{n}.{s}"))) {
            continue;
        }
        let names = ["R", "G", "B", "Z", "id"];
        let (planes, dts) = tile(s, s, 64);
        let raw: usize = planes.iter().map(Vec::len).sum();
        let lim = DecodeLimits::new(64 << 30);
        let name = format!("alloc.zeroed.{s}");
        if want(&name) {
            row(&name, med(7, || drop(std::hint::black_box(exv_format::limits::alloc_zeroed(raw).expect("alloc")))), raw);
        }
        for (pn, p) in [("htj2k", Profile::htj2k()), ("fast", Profile::lossless_fast())] {
            let hd = Header {
                width: s,
                height: s,
                frame_count: 1,
                tile_w: 256,
                tile_h: 256,
                codec_id: p.codec_id().0,
                codec_params: p.params(),
                channels: names.iter().zip(&dts).map(|(n, &dtype)| Channel { name: (*n).into(), dtype }).collect(),
            };
            let name = format!("frame.fresh.{pn}.{s}");
            if !want(&name) {
                continue;
            }
            let body = write_frame_tiles(&planes, None, &hd, &p).expect("encode frame");
            let rd = FrameReader::open(&body, hd.clone()).expect("open").with_limits(lim);
            assert!(rd.decode_all(&p, None).expect("decode") == planes, "{pn} round-trip");
            row(&name, med(7, || drop(std::hint::black_box(rd.decode_all(&p, None).expect("decode")))), raw);
            // `reuse` = decode_all_into the SAME planes every frame (playback steady state); `rgb` = only R,G,B.
            let mut out = Planes::default();
            row(&format!("frame.reuse.{pn}.{s}"), med(7, || rd.decode_all_into(&p, None, Chans::All, &mut out).expect("decode")), raw);
            let rgb_raw = planes[..3].iter().map(Vec::len).sum();
            let mut out = Planes::default();
            row(
                &format!("frame.rgb.{pn}.{s}"),
                med(7, || rd.decode_all_into(&p, None, Chans::Only(&[0, 1, 2]), &mut out).expect("decode")),
                rgb_raw,
            );
        }
    }
}

fn main() {
    let filt = std::env::args().nth(1).unwrap_or_default();
    let want = |n: &str| filt.is_empty() || n.contains(&filt);
    let row = |n: &str, (ms, mn): (f64, f64), bytes: usize| {
        println!(
            "{n:<34} med {ms:>8.3} ms {:>8.1} MB/s | min {mn:>8.3} ms {:>8.1} MB/s",
            bytes as f64 / 1e6 / (ms / 1e3),
            bytes as f64 / 1e6 / (mn / 1e3)
        )
    };

    // ---- whole-tile codec paths: 256x256x5ch tile (raw 1.0 MB), 16 tiles per sample ----
    for (label, noise) in [("synth", 0u32), ("noisy", 64)] {
        let (planes, dts) = tile(256, 256, noise);
        let chans: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
        let raw: usize = planes.iter().map(|p| p.len()).sum();
        let profiles: [(&str, Profile); 6] = [
            ("htj2k", Profile::htj2k()),
            ("fast", Profile::lossless_fast()),
            ("wave", Profile::lossless_wave()),
            ("lossy-wave4", Profile::lossy_wave(4)),
            ("rt", Profile::exv_rt_dwt(8, 0)),
            ("lossy-htj2k16", Profile::htj2k_lossy(16)),
        ];
        for (pn, p) in &profiles {
            let name = format!("tile.enc.{pn}.{label}");
            let blob = encode_tile(&chans, &dts, 256, 256, p).expect("encode");
            if want(&name) {
                let ms = med(21, || {
                    for _ in 0..16 {
                        std::hint::black_box(encode_tile(&chans, &dts, 256, 256, p).expect("encode"));
                    }
                });
                row(&name, (ms.0 / 16.0, ms.1 / 16.0), raw);
            }
            // `dec` = the whole-frame path (decode into a recycled tile buffer); `decv` = the allocating front.
            let name = format!("tile.dec.{pn}.{label}");
            if want(&name) {
                let mut buf = TileBuf::default();
                let ms = med(21, || {
                    for _ in 0..16 {
                        decode_tile_into(&blob, &dts, 256, 256, p, &mut buf).expect("decode");
                        std::hint::black_box(&buf);
                    }
                });
                row(&name, (ms.0 / 16.0, ms.1 / 16.0), raw);
            }
            let name = format!("tile.decv.{pn}.{label}");
            if want(&name) {
                let ms = med(21, || {
                    for _ in 0..16 {
                        std::hint::black_box(decode_tile(&blob, &dts, 256, 256, p).expect("decode"));
                    }
                });
                row(&name, (ms.0 / 16.0, ms.1 / 16.0), raw);
            }
        }
    }

    frame_rows(want, row);

    // ---- stage micro-benches on a 2 MiB plane ----
    let (planes, _) = tile(1024, 1024, 64);
    let f16 = &planes[0]; // 2 MiB
    let f32p = &planes[3]; // 4 MiB
    for (nm, data, st) in [("f16", f16, 2usize), ("f32", f32p, 4)] {
        let name = format!("reorder.split.{nm}");
        if want(&name) {
            row(&name, med(31, || drop(std::hint::black_box(reorder::split_delta(data, st)))), data.len());
        }
        let sd = reorder::split_delta(data, st);
        let name = format!("reorder.join.{nm}");
        if want(&name) {
            row(&name, med(31, || drop(std::hint::black_box(reorder::join_undelta(&sd, st)))), data.len());
        }
        let dt = if st == 2 { Dtype::F16 } else { Dtype::F32 };
        let prev: Vec<u8> = data.iter().map(|b| b.wrapping_add(3)).collect();
        let name = format!("temporal.delta.{nm}");
        if want(&name) {
            row(
                &name,
                med(31, || drop(std::hint::black_box(temporal::delta(data, &prev, dt).expect("delta")))),
                data.len(),
            );
        }
        let res = temporal::delta(data, &prev, dt).expect("delta");
        let name = format!("temporal.undelta.{nm}");
        if want(&name) {
            row(
                &name,
                med(31, || drop(std::hint::black_box(temporal::undelta(&res, &prev, dt).expect("undelta")))),
                data.len(),
            );
        }
        let name = format!("wavelet.fwd.{nm}");
        if want(&name) {
            row(
                &name,
                med(31, || drop(std::hint::black_box(wavelet::forward_2d(data, dt, 1024, 1024).expect("fwd")))),
                data.len(),
            );
        }
        let co = wavelet::forward_2d(data, dt, 1024, 1024).expect("fwd");
        let name = format!("wavelet.inv.{nm}");
        if want(&name) {
            row(
                &name,
                med(31, || drop(std::hint::black_box(wavelet::inverse_2d(&co, dt, 1024, 1024).expect("inv")))),
                data.len(),
            );
        }
    }

    // ---- rANS: skewed byte stream (decorrelated f16 high/low planes), 2 MiB ----
    let sd = reorder::split_delta(f16, 2);
    let cfg = rans::RansConfig::default();
    let name = "rans.enc.2MiB";
    if want(name) {
        row(name, med(31, || drop(std::hint::black_box(rans::encode(&sd, cfg).expect("enc")))), sd.len());
    }
    let enc = rans::encode(&sd, cfg).expect("enc");
    let name = "rans.dec.2MiB";
    if want(name) {
        row(name, med(31, || drop(std::hint::black_box(rans::decode_capped(&enc, sd.len()).expect("dec")))), sd.len());
    }
    assert_eq!(rans::decode_capped(&enc, sd.len()).as_deref(), Some(sd.as_slice()));
}
