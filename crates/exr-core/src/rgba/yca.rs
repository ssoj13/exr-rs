//! Conversion between RGBA and YCA (luminance / subsampled chroma / alpha).
//!
//! 1:1 port of `ImfRgbaYca.cpp` (the `RgbaYca` namespace). All arithmetic is in
//! `f32` exactly as the C++ (where `half` promotes to `float`); results are stored
//! back to `half`. The 27-pixel windowed-sinc filter coefficients are transcribed
//! verbatim; each filter is summed in the SAME left-to-right order as the C++ (a
//! `0.0 + t0` accumulator start is exact, so a coefficient-array loop is
//! bit-identical to the hand-written C++ sum — float add is order-sensitive).
//!
//! `Y = wr*R + wg*G + wb*B`; chroma `RY = (R-Y)/Y`, `BY = (B-Y)/Y`. Subsampling
//! rates are hardwired to 2. `yw` is the RGB→luminance weights from
//! [`crate::attr::Chromaticities::compute_yw`].

// The filters are 1:1 ports of index-arithmetic loops: the loop variable derives
// several indices (parity, the padded `i = N2 + j`, `yca_in[i + off]`, and the
// output), so a range loop is the faithful (and clearest) form here.
#![allow(clippy::needless_range_loop)]

use imath_rs::Half;

use crate::rgba::Rgba;

/// Filter width (pixels) and its half — `ImfRgbaYca.h` `N`, `N2`.
pub const N: usize = 27;
pub const N2: usize = N / 2;

/// Largest finite `half` (`HALF_MAX`); the RGBA→YCA chroma test uses it.
const HALF_MAX: f32 = 65504.0;

// Horizontal/vertical decimation (low-pass) taps — `(offset, coeff)` in the exact
// order the C++ sums them. Even output pixels/lines are filtered.
const DECIMATE: [(i32, f32); 15] = [
    (-13, 0.001064),
    (-11, -0.003771),
    (-9, 0.009801),
    (-7, -0.021586),
    (-5, 0.043978),
    (-3, -0.093067),
    (-1, 0.313659),
    (0, 0.499846),
    (1, 0.313659),
    (3, -0.093067),
    (5, 0.043978),
    (7, -0.021586),
    (9, 0.009801),
    (11, -0.003771),
    (13, 0.001064),
];

// Reconstruction taps — `(offset, coeff)`. No centre tap (the missing chroma is
// interpolated from its neighbours).
const RECONSTRUCT: [(i32, f32); 14] = [
    (-13, 0.002128),
    (-11, -0.007540),
    (-9, 0.019597),
    (-7, -0.043159),
    (-5, 0.087929),
    (-3, -0.186077),
    (-1, 0.627123),
    (1, 0.627123),
    (3, -0.186077),
    (5, 0.087929),
    (7, -0.043159),
    (9, 0.019597),
    (11, -0.007540),
    (13, 0.002128),
];

/// `RGBAtoYCA` — convert `n` RGBA pixels to YCA. `g`=Y, `r`=RY, `b`=BY chroma.
pub fn rgba_to_yca(yw: &[f32; 3], a_is_valid: bool, rgba_in: &[Rgba], yca_out: &mut [Rgba]) {
    for (out, &src) in yca_out.iter_mut().zip(rgba_in.iter()) {
        // Conversion works only for finite, non-negative R, G, B.
        let mut r = src.r.to_f32();
        let mut g = src.g.to_f32();
        let mut b = src.b.to_f32();
        if !r.is_finite() || r < 0.0 {
            r = 0.0;
        }
        if !g.is_finite() || g < 0.0 {
            g = 0.0;
        }
        if !b.is_finite() || b < 0.0 {
            b = 0.0;
        }

        if r == g && g == b {
            // Special case R==G==B: luminance = G, chroma = 0 (lossless grey).
            out.r = Half::from_f32(0.0);
            out.g = Half::from_f32(g);
            out.b = Half::from_f32(0.0);
        } else {
            let y = r * yw[0] + g * yw[1] + b * yw[2];
            out.g = Half::from_f32(y);
            // C++ (`ImfRgbaYca.cpp:74`) reads `Y` back from the just-rounded
            // half `out.g`, not the f32 `y` — both the threshold test and the
            // division use the HALF-ROUNDED luminance.
            let y_rounded = out.g.to_f32();
            out.r = Half::from_f32(if (r - y_rounded).abs() < HALF_MAX * y_rounded {
                (r - y_rounded) / y_rounded
            } else {
                0.0
            });
            out.b = Half::from_f32(if (b - y_rounded).abs() < HALF_MAX * y_rounded {
                (b - y_rounded) / y_rounded
            } else {
                0.0
            });
        }
        out.a = if a_is_valid { src.a } else { Half::from_f32(1.0) };
    }
}

/// `decimateChromaHoriz` — horizontal low-pass + 2x subsampling. `yca_in` must hold
/// `n + N - 1` pixels (N2 edge-replicated copies on each side); `yca_out` holds `n`.
pub fn decimate_chroma_horiz(yca_in: &[Rgba], yca_out: &mut [Rgba]) {
    let n = yca_out.len();
    for j in 0..n {
        let i = (N2 + j) as i32;
        if j & 1 == 0 {
            // Seed the accumulator with the FIRST tap (not `0.0`) so an
            // all-negative-zero sum yields `-0.0`, matching the C++ single
            // expression (`0.0 + (-0.0) = +0.0` would flip the sign).
            let (off0, c0) = DECIMATE[0];
            let p0 = yca_in[(i + off0) as usize];
            let mut sr = p0.r.to_f32() * c0;
            let mut sb = p0.b.to_f32() * c0;
            for &(off, c) in &DECIMATE[1..] {
                let p = yca_in[(i + off) as usize];
                sr += p.r.to_f32() * c;
                sb += p.b.to_f32() * c;
            }
            yca_out[j].r = Half::from_f32(sr);
            yca_out[j].b = Half::from_f32(sb);
        }
        yca_out[j].g = yca_in[i as usize].g;
        yca_out[j].a = yca_in[i as usize].a;
    }
}

/// `decimateChromaVert` — vertical low-pass: `N` input lines → one output line.
pub fn decimate_chroma_vert(yca_in: &[&[Rgba]], yca_out: &mut [Rgba]) {
    let n = yca_out.len();
    for i in 0..n {
        if i & 1 == 0 {
            // See `decimate_chroma_horiz` — seed with the first tap.
            let (off0, c0) = DECIMATE[0];
            let row0 = (13 + off0) as usize;
            let mut sr = yca_in[row0][i].r.to_f32() * c0;
            let mut sb = yca_in[row0][i].b.to_f32() * c0;
            for (row, c) in DECIMATE[1..].iter().map(|&(off, c)| ((13 + off) as usize, c)) {
                sr += yca_in[row][i].r.to_f32() * c;
                sb += yca_in[row][i].b.to_f32() * c;
            }
            yca_out[i].r = Half::from_f32(sr);
            yca_out[i].b = Half::from_f32(sb);
        }
        yca_out[i].g = yca_in[13][i].g;
        yca_out[i].a = yca_in[13][i].a;
    }
}

/// `roundYCA` — round luminance to `round_y` and chroma to `round_c` significand bits.
pub fn round_yca(round_y: u32, round_c: u32, yca_in: &[Rgba], yca_out: &mut [Rgba]) {
    for (i, (out, src)) in yca_out.iter_mut().zip(yca_in.iter()).enumerate() {
        out.g = src.g.round(round_y);
        out.a = src.a;
        if i & 1 == 0 {
            out.r = src.r.round(round_c);
            out.b = src.b.round(round_c);
        }
    }
}

/// `reconstructChromaHoriz` — fill missing chroma for the odd pixels of a line that
/// has valid chroma only every other pixel. `yca_in` holds `n + N - 1` pixels.
pub fn reconstruct_chroma_horiz(yca_in: &[Rgba], yca_out: &mut [Rgba]) {
    let n = yca_out.len();
    for j in 0..n {
        let i = (N2 + j) as i32;
        if j & 1 != 0 {
            // See `decimate_chroma_horiz` — seed with the first tap.
            let (off0, c0) = RECONSTRUCT[0];
            let p0 = yca_in[(i + off0) as usize];
            let mut sr = p0.r.to_f32() * c0;
            let mut sb = p0.b.to_f32() * c0;
            for &(off, c) in &RECONSTRUCT[1..] {
                let p = yca_in[(i + off) as usize];
                sr += p.r.to_f32() * c;
                sb += p.b.to_f32() * c;
            }
            yca_out[j].r = Half::from_f32(sr);
            yca_out[j].b = Half::from_f32(sb);
        } else {
            yca_out[j].r = yca_in[i as usize].r;
            yca_out[j].b = yca_in[i as usize].b;
        }
        yca_out[j].g = yca_in[i as usize].g;
        yca_out[j].a = yca_in[i as usize].a;
    }
}

/// `reconstructChromaVert` — reconstruct chroma for a line that has only luminance,
/// from the surrounding `N` lines.
pub fn reconstruct_chroma_vert(yca_in: &[&[Rgba]], yca_out: &mut [Rgba]) {
    let n = yca_out.len();
    for i in 0..n {
        // See `decimate_chroma_horiz` — seed with the first tap.
        let (off0, c0) = RECONSTRUCT[0];
        let row0 = (13 + off0) as usize;
        let mut sr = yca_in[row0][i].r.to_f32() * c0;
        let mut sb = yca_in[row0][i].b.to_f32() * c0;
        for (row, c) in RECONSTRUCT[1..].iter().map(|&(off, c)| ((13 + off) as usize, c)) {
            sr += yca_in[row][i].r.to_f32() * c;
            sb += yca_in[row][i].b.to_f32() * c;
        }
        yca_out[i].r = Half::from_f32(sr);
        yca_out[i].b = Half::from_f32(sb);
        yca_out[i].g = yca_in[13][i].g;
        yca_out[i].a = yca_in[13][i].a;
    }
}

/// `YCAtoRGBA` — inverse of [`rgba_to_yca`].
pub fn yca_to_rgba(yw: &[f32; 3], yca_in: &[Rgba], rgba_out: &mut [Rgba]) {
    for (out, &src) in rgba_out.iter_mut().zip(yca_in.iter()) {
        let in_r = src.r.to_f32();
        let in_b = src.b.to_f32();
        if in_r == 0.0 && in_b == 0.0 {
            // Special case: zero chroma → R=G=B=Y (lossless grey).
            out.r = src.g;
            out.g = src.g;
            out.b = src.g;
            out.a = src.a;
        } else {
            let y = src.g.to_f32();
            let r = (in_r + 1.0) * y;
            let b = (in_b + 1.0) * y;
            let g = (y - r * yw[0] - b * yw[2]) / yw[1];
            out.r = Half::from_f32(r);
            out.g = Half::from_f32(g);
            out.b = Half::from_f32(b);
            out.a = src.a;
        }
    }
}

/// `std::max(a, b)` == `(a < b) ? b : a` — NaN-positional (returns `a` whenever
/// the comparison is false, including when either operand is NaN), unlike
/// `f32::max`'s IEEE `maxNum` (which returns the non-NaN operand). The
/// saturation/desaturation math ports C++ `std::max`/`std::min` literally so
/// any future NaN input propagates identically.
fn cmax(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

/// `std::min(a, b)` == `(b < a) ? b : a`. See [`cmax`].
fn cmin(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

/// Saturation of an RGB(A) pixel — `1 - min/max` (0 when `max <= 0`).
fn saturation(p: &Rgba) -> f32 {
    let (r, g, b) = (p.r.to_f32(), p.g.to_f32(), p.b.to_f32());
    let rgb_max = cmax(r, cmax(g, b));
    let rgb_min = cmin(r, cmin(g, b));
    if rgb_max > 0.0 {
        1.0 - rgb_min / rgb_max
    } else {
        0.0
    }
}

/// Desaturate `p` by factor `f` while preserving its luminance.
fn desaturate(p: &Rgba, f: f32, yw: &[f32; 3], out: &mut Rgba) {
    let (r, g, b) = (p.r.to_f32(), p.g.to_f32(), p.b.to_f32());
    let rgb_max = cmax(r, cmax(g, b));

    let mut or = cmax(rgb_max - (rgb_max - r) * f, 0.0);
    let mut og = cmax(rgb_max - (rgb_max - g) * f, 0.0);
    let mut ob = cmax(rgb_max - (rgb_max - b) * f, 0.0);

    // C++ reads back the already-half-rounded outputs for the luminance scaling.
    or = Half::from_f32(or).to_f32();
    og = Half::from_f32(og).to_f32();
    ob = Half::from_f32(ob).to_f32();

    let yin = r * yw[0] + g * yw[1] + b * yw[2];
    let yout = or * yw[0] + og * yw[1] + ob * yw[2];
    if yout > 0.0 {
        let s = yin / yout;
        or *= s;
        og *= s;
        ob *= s;
    }
    out.r = Half::from_f32(or);
    out.g = Half::from_f32(og);
    out.b = Half::from_f32(ob);
    out.a = p.a;
}

/// `fixSaturation` — desaturate super-saturated pixels (RGB ringing near edges)
/// using three adjacent input lines `rgba_in[0..3]`; writes the middle line.
pub fn fix_saturation(yw: &[f32; 3], rgba_in: &[&[Rgba]; 3], rgba_out: &mut [Rgba]) {
    let n = rgba_out.len();
    // No-panic rule: a zero-width line, or any input row shorter than
    // `n`, would otherwise index out of bounds below. Neither is reachable
    // through `RgbaInputFile` (width is always >= 1 and every row is sized
    // `n`), but this is a library function, so the invariant is checked
    // rather than assumed — C++'s equivalent is unchecked (UB on misuse).
    if n == 0 || rgba_in.iter().any(|row| row.len() < n) {
        return;
    }
    let mut neighbor_a2 = saturation(&rgba_in[0][0]);
    let mut neighbor_a1 = neighbor_a2;
    let mut neighbor_b2 = saturation(&rgba_in[2][0]);
    let mut neighbor_b1 = neighbor_b2;

    for i in 0..n {
        let neighbor_a0 = neighbor_a1;
        neighbor_a1 = neighbor_a2;
        let neighbor_b0 = neighbor_b1;
        neighbor_b1 = neighbor_b2;
        if i < n - 1 {
            neighbor_a2 = saturation(&rgba_in[0][i + 1]);
            neighbor_b2 = saturation(&rgba_in[2][i + 1]);
        }

        let s_mean =
            (0.25f32 * (neighbor_a0 + neighbor_a2 + neighbor_b0 + neighbor_b2)).min(1.0);

        let p = rgba_in[1][i];
        let s = saturation(&p);
        if s > s_mean {
            let s_max = (1.0 - (1.0 - s_mean) * 0.25).min(1.0);
            if s > s_max {
                desaturate(&p, s_max / s, yw, &mut rgba_out[i]);
                continue;
            }
        }
        rgba_out[i] = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::Chromaticities;

    /// A constant grey image round-trips through the full YCA pipeline EXACTLY:
    /// R==G==B → chroma 0 (special case) → reconstruct keeps 0 → YCAtoRGBA grey.
    #[test]
    fn grey_round_trips_exactly() {
        let yw = Chromaticities::default().compute_yw().unwrap();
        let n = 8;
        let grey = Rgba::from_f32(0.5, 0.5, 0.5, 1.0);
        let rgba: Vec<Rgba> = vec![grey; n];

        let mut yca = vec![Rgba::default(); n];
        rgba_to_yca(&yw, true, &rgba, &mut yca);
        // Chroma must be exactly zero for grey.
        assert!(yca.iter().all(|p| p.r.to_f32() == 0.0 && p.b.to_f32() == 0.0));

        let mut back = vec![Rgba::default(); n];
        yca_to_rgba(&yw, &yca, &mut back);
        for p in &back {
            assert_eq!(p.r.to_f32(), 0.5);
            assert_eq!(p.g.to_f32(), 0.5);
            assert_eq!(p.b.to_f32(), 0.5);
        }
    }

    /// Reconstruction filter coefficients sum to ~1.0 (unity DC gain), so a
    /// constant-chroma line reconstructs to the same constant on odd pixels.
    #[test]
    fn reconstruct_horiz_unity_dc() {
        let n = 6;
        let c = Rgba::from_f32(0.25, 1.0, -0.10, 1.0); // r=RY, g=Y, b=BY
        let padded = vec![c; n + N - 1];
        let mut out = vec![Rgba::default(); n];
        reconstruct_chroma_horiz(&padded, &mut out);
        for p in &out {
            assert!((p.r.to_f32() - 0.25).abs() < 1e-2, "RY={}", p.r.to_f32());
            assert!((p.b.to_f32() - (-0.10)).abs() < 1e-2, "BY={}", p.b.to_f32());
        }
    }

    /// `rgba_to_yca` must divide by the HALF-ROUNDED luminance, not the
    /// f32 one — regression using a concrete example.
    #[test]
    // The literals are exact half values from the repro (dyadic, exactly
    // representable in f32); the extra digits are the point, not noise.
    #[allow(clippy::excessive_precision)]
    fn rgba_to_yca_uses_half_rounded_luminance() {
        let yw = Chromaticities::default().compute_yw().unwrap();
        let rgba = [Rgba::from_f32(2.046875, 3.802734375, 0.57666015625, 1.0)];
        let mut yca = [Rgba::default(); 1];
        rgba_to_yca(&yw, true, &rgba, &mut yca);

        let r = rgba[0].r.to_f32();
        let g = rgba[0].g.to_f32();
        let b = rgba[0].b.to_f32();
        let y = r * yw[0] + g * yw[1] + b * yw[2];
        let y_rounded = Half::from_f32(y).to_f32();

        let expected_ry = Half::from_f32(if (r - y_rounded).abs() < HALF_MAX * y_rounded {
            (r - y_rounded) / y_rounded
        } else {
            0.0
        });
        let expected_by = Half::from_f32(if (b - y_rounded).abs() < HALF_MAX * y_rounded {
            (b - y_rounded) / y_rounded
        } else {
            0.0
        });
        assert_eq!(yca[0].r, expected_ry);
        assert_eq!(yca[0].b, expected_by);

        // The bug's symptom: dividing by the UNROUNDED `y` gives a different
        // bit pattern for this input.
        let wrong_ry = Half::from_f32((r - y) / y);
        assert_ne!(yca[0].r, wrong_ry, "regression: must not divide by the unrounded Y");
    }

    /// Seeding the accumulator with `0.0` instead of the first tap flips
    /// the sign of an all-negative-zero sum (`0.0 + (-0.0) = +0.0`).
    #[test]
    fn decimate_chroma_horiz_seeds_accumulator_with_first_tap() {
        // n = 1 (only j = 0, the even branch), so the input window is exactly
        // the N taps centred at N2.
        let mut input = vec![Rgba::default(); N];
        for &(off, c) in &DECIMATE {
            let idx = (N2 as i32 + off) as usize;
            // Choose the sign of the (zero) input so `input * c` is `-0.0` for
            // EVERY tap, regardless of the tap's own sign.
            input[idx].r = if c > 0.0 { Half::from_bits(0x8000) } else { Half::from_bits(0x0000) };
        }
        let mut out = vec![Rgba::default(); 1];
        decimate_chroma_horiz(&input, &mut out);
        assert_eq!(out[0].r.to_f32(), 0.0);
        assert_ne!(
            out[0].r.bits() & 0x8000,
            0,
            "sum of all-negative-zero products must stay -0.0, not flip to +0.0"
        );
    }

    /// `cmax`/`cmin` must match C++ `std::max`/`std::min`'s NaN-positional
    /// semantics (`(a < b) ? b : a`), not `f32::max`'s IEEE `maxNum` (which
    /// always prefers the non-NaN operand).
    #[test]
    fn cmax_cmin_are_nan_positional_like_std_max_min() {
        // a < b is false when a is NaN, so std::max(NaN, 1.0) returns `a` = NaN.
        assert!(cmax(f32::NAN, 1.0).is_nan());
        assert!(cmin(f32::NAN, 1.0).is_nan());
        // a < b is false when b is NaN too, so std::max(1.0, NaN) returns `a` = 1.0.
        assert_eq!(cmax(1.0, f32::NAN), 1.0);
        assert_eq!(cmin(1.0, f32::NAN), 1.0);
    }

    /// A zero-width (or otherwise malformed) call must not panic.
    #[test]
    fn fix_saturation_does_not_panic_on_empty_input() {
        let yw = Chromaticities::default().compute_yw().unwrap();
        let empty: [Rgba; 0] = [];
        let rows: [&[Rgba]; 3] = [&empty, &empty, &empty];
        let mut out: [Rgba; 0] = [];
        fix_saturation(&yw, &rows, &mut out); // must return, not panic

        // A mismatched (too-short) row must also not panic.
        let one = [Rgba::default()];
        let rows2: [&[Rgba]; 3] = [&one, &empty, &one];
        let mut out2 = vec![Rgba::default(); 1];
        fix_saturation(&yw, &rows2, &mut out2);
    }
}
