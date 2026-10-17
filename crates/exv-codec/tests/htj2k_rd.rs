//! Rate-distortion spike: is the HTJ2K irreversible 9/7 profile (`Profile::htj2k_ldr`) worth a CLI flag?
//!
//! `lossy_wave(q)` and `htj2k_lossy(q)` (hdr codes) are lossy for FLOAT channels only: integer channels stay bit-exact
//! under both (the project rule for ids/masks), so on U8/U16 input they are just the lossless rows and the 9/7 is the only
//! lossy integer codec. The fair rate-distortion comparison therefore feeds the SAME image as normalised F32 to
//! `lossy_wave(q)` and `htj2k_lossy(q)` and compares their bytes at equal PSNR against `htj2k_ldr(q)` on the integers.
//! Decision rule recorded in `docs/FORMAT.md` section 7: expose `--lossy-htj2k-ldr` only when the 9/7 is >= 25 % smaller
//! than lossy-wave at equal PSNR.
//!
//! Run: `cargo test -p exv-codec --release --test htj2k_rd -- --ignored --nocapture`
//!
//! Content (3 channels, one 512x512 tile): `grad` = smooth gradient + mild noise, `photo` = fractal value noise with
//! hard edges + sensor-like noise, `ui` = flat blocks + thin lines (the worst case for any wavelet codec).
//! PSNR is measured on the 0..peak integer scale over all channels (peak 255 for U8, 65535 for U16).

#![cfg(feature = "htj2k-backend")]

use exv_codec::{Profile, decode_tile, encode_tile};
use exv_format::Dtype;

const W: u32 = 512;
const H: u32 = 512;

/// xorshift32: deterministic noise without a dependency.
fn rng(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

/// Value noise at integer lattice `(x, y)` of octave seed `o` in `[0, 1)`.
fn lattice(x: i32, y: i32, o: u32) -> f32 {
    let mut s = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA6B) ^ o.wrapping_mul(0xC2B2_AE35) ^ 0x1234_5678;
    (rng(&mut s) >> 8) as f32 / (1u32 << 24) as f32
}

/// Smoothly interpolated fractal value noise in `[0, 1)` (5 octaves).
fn fbm(x: f32, y: f32) -> f32 {
    let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, 1.0 / 64.0);
    for o in 0..5 {
        let (fx, fy) = (x * f, y * f);
        let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
        let (tx, ty) = (fx - ix as f32, fy - iy as f32);
        let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let a = lattice(ix, iy, o) * (1.0 - sx) + lattice(ix + 1, iy, o) * sx;
        let b = lattice(ix, iy + 1, o) * (1.0 - sx) + lattice(ix + 1, iy + 1, o) * sx;
        sum += (a * (1.0 - sy) + b * sy) * amp;
        norm += amp;
        amp *= 0.5;
        f *= 2.0;
    }
    sum / norm
}

/// One synthetic 3-channel image as normalised `[0, 1]` f32 planes.
fn content(kind: &str) -> Vec<Vec<f32>> {
    let mut seed = 0xC0FF_EE11u32;
    (0..3u32)
        .map(|c| {
            (0..W * H)
                .map(|i| {
                    let (x, y) = ((i % W) as f32, (i / W) as f32);
                    let noise = |s: &mut u32, a: f32| (rng(s) >> 8) as f32 / (1u32 << 24) as f32 * a - a / 2.0;
                    let v = match kind {
                        "grad" => (x / W as f32 * 0.6 + y / H as f32 * 0.4) * (0.8 + 0.1 * c as f32) + noise(&mut seed, 0.02),
                        "photo" => {
                            // soft scene + a few hard-edged "objects" + sensor noise (stronger in the shadows)
                            let base = fbm(x + 40.0 * c as f32, y);
                            let edge = if ((x as i32 / 96) + (y as i32 / 80)) % 3 == 0 { 0.15 } else { 0.0 };
                            base + edge + noise(&mut seed, 0.03 * (1.2 - base))
                        }
                        _ => {
                            let block = ((x as i32 / 64) * 7 + (y as i32 / 48) * 3 + c as i32) % 5;
                            let line = if (x as i32 % 32) < 2 || (y as i32 % 40) < 1 { 0.9 } else { 0.0 };
                            0.12 + block as f32 * 0.17 + line * 0.1
                        }
                    };
                    v.clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect()
}

fn to_int(planes: &[Vec<f32>], dt: Dtype) -> Vec<Vec<u8>> {
    let max = if dt == Dtype::U8 { 255.0 } else { 65535.0 };
    planes
        .iter()
        .map(|p| {
            p.iter()
                .flat_map(|&v| {
                    let q = (v * max).round() as u32;
                    if dt == Dtype::U8 { vec![q as u8] } else { (q as u16).to_le_bytes().to_vec() }
                })
                .collect()
        })
        .collect()
}

/// Samples of an integer plane as f64 on the `0..peak` scale.
fn samples(p: &[u8], dt: Dtype) -> Vec<f64> {
    if dt == Dtype::U8 { p.iter().map(|&b| f64::from(b)).collect() } else { p.as_chunks::<2>().0.iter().map(|c| f64::from(u16::from_le_bytes([c[0], c[1]]))).collect() }
}

fn psnr(a: &[Vec<f64>], b: &[Vec<f64>], peak: f64) -> f64 {
    let (mut se, mut n) = (0.0, 0usize);
    for (pa, pb) in a.iter().zip(b) {
        for (x, y) in pa.iter().zip(pb) {
            se += (x - y) * (x - y);
            n += 1;
        }
    }
    let mse = se / n as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (peak * peak / mse).log10() }
}

/// `(bytes, psnr)` of one profile on the integer planes.
fn run_int(planes: &[Vec<u8>], dt: Dtype, prof: &Profile) -> (usize, f64) {
    let refs: Vec<&[u8]> = planes.iter().map(Vec::as_slice).collect();
    let dts = vec![dt; planes.len()];
    let blob = encode_tile(&refs, &dts, W, H, prof).expect("encode");
    let back = decode_tile(&blob, &dts, W, H, prof).expect("decode");
    let peak = if dt == Dtype::U8 { 255.0 } else { 65535.0 };
    let a: Vec<Vec<f64>> = planes.iter().map(|p| samples(p, dt)).collect();
    let b: Vec<Vec<f64>> = back.iter().map(|p| samples(p, dt)).collect();
    (blob.len(), psnr(&a, &b, peak))
}

/// The same image as normalised F32 under the hdr-codes profile; PSNR on the 0..peak scale.
fn run_hdr(planes: &[Vec<f32>], peak: f64, prof: &Profile) -> (usize, f64) {
    let bytes: Vec<Vec<u8>> = planes.iter().map(|p| p.iter().flat_map(|v| v.to_le_bytes()).collect()).collect();
    let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let dts = vec![Dtype::F32; bytes.len()];
    let blob = encode_tile(&refs, &dts, W, H, prof).expect("encode");
    let back = decode_tile(&blob, &dts, W, H, prof).expect("decode");
    let f = |p: &[u8]| p.as_chunks::<4>().0.iter().map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])) * peak).collect::<Vec<_>>();
    let a: Vec<Vec<f64>> = bytes.iter().map(|p| f(p)).collect();
    let b: Vec<Vec<f64>> = back.iter().map(|p| f(p)).collect();
    (blob.len(), psnr(&a, &b, peak))
}

/// Bytes at `target` dB by log-linear interpolation along a `(bytes, psnr)` curve (any order); `None` outside its span.
fn bytes_at(curve: &[(usize, f64)], target: f64) -> Option<f64> {
    let mut c: Vec<_> = curve.iter().filter(|p| p.1 < 98.0).copied().collect();
    c.sort_by(|a, b| a.1.total_cmp(&b.1));
    c.windows(2).find(|w| w[0].1 <= target && target <= w[1].1 && w[1].1 > w[0].1).map(|w| {
        let t = (target - w[0].1) / (w[1].1 - w[0].1);
        ((w[0].0 as f64).ln() * (1.0 - t) + (w[1].0 as f64).ln() * t).exp()
    })
}

#[test]
#[ignore = "measurement run (prints the RD table); see the module docs"]
fn htj2k_rd() {
    let targets = [30.0, 35.0, 40.0, 45.0, 50.0];
    let mut wins = Vec::new();
    for dt in [Dtype::U8, Dtype::U16] {
        let peak = if dt == Dtype::U8 { 255.0 } else { 65535.0 };
        let sh = 0;
        for kind in ["grad", "photo", "ui"] {
            let f = content(kind);
            let planes = to_int(&f, dt);
            let raw = planes.iter().map(Vec::len).sum::<usize>();
            let lossless = run_int(&planes, dt, &Profile::htj2k());
            println!("\n== {kind} {dt:?} {W}x{H}x3 raw {raw} B | htj2k lossless {} B ({:.2}x)", lossless.0, raw as f64 / lossless.0 as f64);
            let wave_int = run_int(&planes, dt, &Profile::lossy_wave(4));
            println!("  lossy-wave on {dt:?} input (integers stay exact): {} B / {:.1} dB", wave_int.0, wave_int.1);
            let wave: Vec<_> = (1..=(12 + sh)).map(|q| (q, run_hdr(&f, peak, &Profile::lossy_wave(q as u8)))).collect();
            let ldr: Vec<_> = (4..=(if dt == Dtype::U8 { 24 } else { 40 })).step_by(2).map(|q| (q, run_int(&planes, dt, &Profile::htj2k_ldr(q as u8)))).collect();
            let hdr: Vec<_> = (1..=16).step_by(3).map(|q| (q, run_hdr(&f, peak, &Profile::htj2k_lossy(q as u8)))).collect();
            for (name, rows) in [("lossy-wave q (F32 in)", &wave), ("htj2k-ldr q", &ldr), ("hdr-codes q (F32 in)", &hdr)] {
                let line: Vec<String> = rows.iter().map(|(q, (b, p))| format!("q{q}:{b}B/{p:.1}dB")).collect();
                println!("  {name:<24}{}", line.join("  "));
            }
            let curve = |rows: &Vec<(i32, (usize, f64))>| rows.iter().map(|r| r.1).collect::<Vec<_>>();
            let (cw, cl, ch) = (curve(&wave), curve(&ldr), curve(&hdr));
            println!("  equal-PSNR bytes     {:>8} {:>10} {:>10} {:>10}   ldr vs wave", "dB", "lossy-wave", "htj2k-ldr", "hdr-codes");
            for t in targets {
                let (w, l, hd) = (bytes_at(&cw, t), bytes_at(&cl, t), bytes_at(&ch, t));
                let fmt = |v: Option<f64>| v.map_or("-".to_string(), |b| format!("{b:.0}"));
                let delta = w.zip(l).map(|(w, l)| (1.0 - l / w) * 100.0);
                if let (Some(d), "photo" | "grad") = (delta, kind) {
                    wins.push(d);
                }
                println!("                       {t:>8.0} {:>10} {:>10} {:>10}   {}", fmt(w), fmt(l), fmt(hd), delta.map_or("-".to_string(), |d| format!("{d:+.0}% smaller")));
            }
        }
    }
    wins.sort_by(|a, b| a.total_cmp(b));
    if let Some(med) = wins.get(wins.len() / 2) {
        println!("\nmedian saving of htj2k-ldr vs lossy-wave over grad+photo @ equal PSNR: {med:+.1}% (flag threshold: 25%)");
    }
}
