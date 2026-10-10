//! The display transfer function: **one** source of truth for the tile shader,
//! the present pass and their CPU twins.
//!
//! ## The display chain
//!
//! Everything on screen (egui's UI and the image tiles) is drawn into the
//! **canvas**, an `Rgba16Float` texture (`present::CANVAS_FORMAT`), in one
//! encoding: **extended sRGB**, the sRGB curve of this module with 1.0 = SDR
//! reference white, continued above 1.0 (HDR highlights) and mirrored below 0.0
//! (colours outside the Rec.709 gamut). egui writes it because a non-sRGB target
//! gets egui's `fs_main_gamma_framebuffer`, which stores its sRGB-authored colours
//! unchanged; the tile shader writes it through [`oetf`] (`srgb_oetf`).
//!
//! The present pass (`present.rs`) then turns the canvas into what the chosen
//! window surface wants (8/10-bit sRGB, HDR10 PQ or scRGB), decoding with [`eotf`].
//! The tile shader does not know the surface; the present pass does not know the
//! image. An SDR surface clips the canvas to 0..1, so for SDR the chain is the one
//! it always was: scene-linear 0.18 reaches the screen as code 118.
//!
//! ## Why the WGSL is generated
//!
//! [`oetf_wgsl`] and [`eotf_wgsl`] are built with `format!` from the very
//! constants [`oetf`] / [`eotf`] use, so the shaders and the CPU twins cannot
//! drift. [`display_pixel`] is the CPU twin of the whole per-pixel tile shader
//! body, so the chain is verified numerically without a GPU.

// Shared canvas encoding and gamut conversion; image operators remain viewer-owned.
pub use egui_display::transfer::{oetf, eotf, oetf_wgsl};
#[cfg(test)]
use egui_display::transfer::{eotf_wgsl, SRGB_CUT, SRGB_SLOPE, SRGB_SCALE, SRGB_OFFSET, SRGB_EXP};
pub(crate) use egui_display::transfer::{SRGB_TO_REC2020, REC2020_TO_SRGB};

/// Format an `f32` as a WGSL float literal that round-trips exactly (`{:?}`
/// prints the shortest representation that parses back to the same `f32`, and
/// always includes a `.` or an exponent so WGSL reads it as `f32`, not `i32`).
pub(crate) fn fl(v: f32) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

// ── CPU twin of the colour pass ───────────────────────────────────────────────

/// Channel-display mode, as the tile shader's `channel_mode` uniform encodes it
/// (the single definition; the shader's `switch` arms are checked against it).
/// Plain `u32` so the twin takes exactly what the shader takes.
pub mod mode {
    pub const RGBA: u32 = 0;
    pub const R: u32 = 1;
    pub const G: u32 = 2;
    pub const B: u32 = 3;
    pub const A: u32 = 4;
    pub const LUM: u32 = 5;
}

/// Rec.709 luminance weights — shared by this twin and the WGSL below.
pub const LUMA_709: [f32; 3] = [0.2126, 0.7152, 0.0722];

// ── Tone mapping (the colour image only; isolate modes show raw values) ────────────

/// Tone operator, as the tile shader's `tonemap` uniform encodes it.
pub mod tonemap {
    /// Scene-linear straight to the display (clipped above 1).
    pub const NONE: u32 = 0;
    /// `c / (1 + c)` per channel.
    pub const REINHARD: u32 = 1;
    /// Troy Sobotka's AgX, the three.js / Filament form (below).
    pub const AGX: u32 = 2;
    /// Every operator with its UI name, in cycle order.
    pub const ALL: [(u32, &str); 3] = [(NONE, "None"), (REINHARD, "Reinhard"), (AGX, "AgX")];

    /// The UI name of `op` (an unknown code reads as `None`, which is what it renders as).
    pub fn name(op: u32) -> &'static str {
        ALL.iter().find(|(c, _)| *c == op).map_or("None", |(_, n)| n)
    }

    /// `op` when it is an operator, else `NONE` (a hand-edited config could say anything).
    pub fn known(op: u32) -> u32 {
        if ALL.iter().any(|(c, _)| *c == op) { op } else { NONE }
    }

    /// The operator after `op` in [`ALL`], wrapping.
    pub fn next(op: u32) -> u32 {
        let i = ALL.iter().position(|(c, _)| *c == op).map_or(0, |i| i + 1);
        ALL[i % ALL.len()].0
    }
}

/// A 3x3 matrix by COLUMNS, like WGSL's `mat3x3<f32>(c0, c1, c2)` constructor.
type Mat3 = [[f32; 3]; 3];

/// AgX inset (Blender's config.ocio, Rec.2020), by columns.
const AGX_INSET: Mat3 = [
    [0.856_627_15, 0.137_318_97, 0.111_898_21],
    [0.095_121_24, 0.761_242, 0.076_799_42],
    [0.048_251_606, 0.101_439_04, 0.811_302_37],
];
/// AgX outset (EaryChow's AgXBaseRec2020, inverted), by columns.
const AGX_OUTSET: Mat3 = [
    [1.127_100_6, -0.141_329_76, -0.141_329_76],
    [-0.110_606_64, 1.157_823_7, -0.110_606_64],
    [-0.016_493_939, -0.016_493_939, 1.251_936_4],
];
/// The log2 window AgX encodes, in stops.
const AGX_MIN_EV: f32 = -12.473_93;
const AGX_MAX_EV: f32 = 4.026_069;
/// The sigmoid fit, highest power first (x^6 .. x^0; mean error^2 3.67e-6).
const AGX_POLY: [f32; 7] = [15.5, -40.14, 31.96, -6.868, 0.4298, 0.1191, -0.00232];

fn mul(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[0][r] * v[0] + m[1][r] * v[1] + m[2][r] * v[2])
}

/// AgX on linear sRGB, to linear sRGB in `[0, 1]`: the minimal AgX of Benjamin
/// Wrensch (iolite-engine.com/blog_posts/minimal_agx_implementation) in the
/// three.js / Filament form of dmnsgn/glsl-tone-map `agx.wgsl` (MIT): Rec.2020
/// working space, inset, log2 window, the sigmoid fit, outset, the 2.2 display
/// EOTF, back to Rec.709 and a clamp. The sRGB OETF after it re-encodes.
fn agx(rgb: [f32; 3]) -> [f32; 3] {
    let c = mul(&AGX_INSET, mul(&SRGB_TO_REC2020, rgb));
    let c = c.map(|v| {
        let e = v.max(1e-10).log2().clamp(AGX_MIN_EV, AGX_MAX_EV);
        let x = ((e - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)).clamp(0.0, 1.0);
        AGX_POLY.iter().fold(0.0, |acc, k| acc * x + k)
    });
    let c = mul(&AGX_OUTSET, c).map(|v| v.max(0.0).powf(2.2));
    mul(&REC2020_TO_SRGB, c).map(|v| v.clamp(0.0, 1.0))
}

/// The tone operator `op` ([`tonemap`]) on exposed linear RGB.
pub fn tone(rgb: [f32; 3], op: u32) -> [f32; 3] {
    match op {
        tonemap::REINHARD => rgb.map(|v| {
            let v = v.max(0.0);
            v / (1.0 + v)
        }),
        tonemap::AGX => agx(rgb),
        _ => rgb,
    }
}

/// The WGSL twin of [`tone`], generated from the same constants. Declares
/// `fn tone(rgb: vec3<f32>, op: u32) -> vec3<f32>`.
pub fn tone_wgsl() -> String {
    let mat = |m: &Mat3| {
        let cols: Vec<String> = m.iter().map(|c| format!("vec3<f32>({}, {}, {})", fl(c[0]), fl(c[1]), fl(c[2]))).collect();
        format!("mat3x3<f32>({})", cols.join(", "))
    };
    let p = AGX_POLY.map(fl);
    format!(
        "\
fn agx(rgb: vec3<f32>) -> vec3<f32> {{
    var c = {inset} * ({to2020} * rgb);
    let e = clamp(log2(max(c, vec3<f32>(1e-10))), vec3<f32>({lo}), vec3<f32>({hi}));
    let x = clamp((e - vec3<f32>({lo})) / ({hi} - ({lo})), vec3<f32>(0.0), vec3<f32>(1.0));
    c = (((((({p0} * x + {p1}) * x + {p2}) * x + {p3}) * x + {p4}) * x + {p5}) * x + {p6});
    c = pow(max({outset} * c, vec3<f32>(0.0)), vec3<f32>(2.2));
    return clamp({to709} * c, vec3<f32>(0.0), vec3<f32>(1.0));
}}

fn tone(rgb: vec3<f32>, op: u32) -> vec3<f32> {{
    if op == {reinhard}u {{
        let v = max(rgb, vec3<f32>(0.0));
        return v / (vec3<f32>(1.0) + v);
    }}
    if op == {agx_op}u {{
        return agx(rgb);
    }}
    return rgb;
}}
",
        inset = mat(&AGX_INSET),
        to2020 = mat(&SRGB_TO_REC2020),
        outset = mat(&AGX_OUTSET),
        to709 = mat(&REC2020_TO_SRGB),
        lo = fl(AGX_MIN_EV),
        hi = fl(AGX_MAX_EV),
        p0 = p[0],
        p1 = p[1],
        p2 = p[2],
        p3 = p[3],
        p4 = p[4],
        p5 = p[5],
        p6 = p[6],
        reinhard = tonemap::REINHARD,
        agx_op = tonemap::AGX,
    )
}

/// The gamma dial, `sign(v) * pow(|v|, 1/gamma)` (mirrored like [`oetf`], so an
/// out-of-gamut negative survives to an HDR surface): the CPU twin of the tile
/// shader's `apply_gamma`. The gamma is floored like the shader's uniform
/// (`TilePipeline::prepare`), so a zero/negative dial cannot produce NaNs.
pub fn dial(v: f32, gamma: f32) -> f32 {
    v.abs().powf(1.0 / gamma.max(1e-3)).copysign(v)
}

/// CPU twin of the colour-pass shader body: one linear RGBA sample in,
/// one display-encoded RGBA sample out.
///
/// `exposure_ev` is in stops, `gamma` is the user dial (1.0 = identity, applied
/// as `pow(c, 1/gamma)` BEFORE the sRGB encode), `channel_mode` is one of
/// `mode`, `op` one of `tonemap` (after the exposure, colour mode only: an
/// isolated channel is data and shows raw). This exists so the display chain can be asserted numerically in
/// tests; the shader is generated from the same constants, and
/// `tiles::render::tests::shader_carries_the_display_transform` guards the
/// structural match.
pub fn display_pixel(raw: [f32; 4], exposure_ev: f32, gamma: f32, channel_mode: u32, op: u32) -> [f32; 4] {
    let dial = |v: f32| dial(v, gamma);
    let mult = 2.0_f32.powf(exposure_ev);

    if channel_mode == mode::RGBA {
        let [r, g, b] = tone([raw[0] * mult, raw[1] * mult, raw[2] * mult], op);
        [oetf(dial(r)), oetf(dial(g)), oetf(dial(b)), raw[3]]
    } else {
        let (val, out_a) = match channel_mode {
            mode::R => (raw[0], 1.0),
            mode::G => (raw[1], 1.0),
            mode::B => (raw[2], 1.0),
            // Alpha isolation shows alpha as opaque grey, like every other
            // single-channel mode (the shader's `out_a = 1.0`).
            mode::A => (raw[3], 1.0),
            _ => (
                raw[0] * LUMA_709[0] + raw[1] * LUMA_709[1] + raw[2] * LUMA_709[2],
                1.0,
            ),
        };
        let v = oetf(dial(val * mult));
        [v, v, v, out_a]
    }
}

/// The canvas value the tile shader stores for display value `d` on an output:
/// negatives are clipped to 0 on SDR (a negative would subtract in egui's
/// premultiplied blend under a partial alpha; SDR cannot show it anyway), kept on
/// HDR (colours outside Rec.709). The CPU twin of the tile shader's last step
/// (test-only: the export clips in `code8`).
#[cfg(test)]
pub fn canvas(d: [f32; 4], hdr: bool) -> [f32; 4] {
    if hdr { d } else { [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0), d[3]] }
}

/// The 8-bit code an SDR surface stores for a canvas value (the present pass
/// clips to 0..1): what a screengrab / colour picker reads.
pub fn code8(display_encoded: f32) -> u8 {
    (display_encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived matrix is the BT.2407 / BT.2087 table to its four digits (rows are
    /// the Rec.2020 channels; the stored matrix is by columns), its rows sum to exactly
    /// 1 (white stays white, unlike the rounded table's 0.9999), and the two directions
    /// are inverses.
    #[test]
    fn gamut_matrices_are_derived_from_the_primaries() {
        let table = [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]];
        for r in 0..3 {
            let sum: f32 = (0..3).map(|c| SRGB_TO_REC2020[c][r]).sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {r} sums to {sum}");
            for c in 0..3 {
                assert!((SRGB_TO_REC2020[c][r] - table[r][c]).abs() < 5e-5, "[{r}][{c}] {}", SRGB_TO_REC2020[c][r]);
            }
        }
        for v in [[1.0, 0.0, 0.0], [0.2, 0.7, 0.05], [1.0, 1.0, 1.0]] {
            let back = mul(&REC2020_TO_SRGB, mul(&SRGB_TO_REC2020, v));
            for k in 0..3 {
                assert!((back[k] - v[k]).abs() < 1e-6, "{v:?} -> {back:?}");
            }
        }
    }

    /// The number the whole W5.1 fix is about: scene-linear 18 % grey must reach
    /// an sRGB-8 surface as code 118 (0.4613), not code 46 (which is what the
    /// old "write linear, let egui encode" chain produced).
    #[test]
    fn middle_grey_lands_on_code_118() {
        let out = display_pixel([0.18, 0.18, 0.18, 1.0], 0.0, 1.0, mode::RGBA, tonemap::NONE);
        assert!(
            (out[0] - 0.4613).abs() < 5e-4,
            "linear 0.18 → sRGB {}; want ≈0.4613",
            out[0]
        );
        assert_eq!(code8(out[0]), 118);
        // The old chain stored the linear value itself → code 46. Guard against
        // a regression back to it.
        assert_ne!(code8(0.18), 118);
        assert_eq!(code8(0.18), 46);
    }

    /// Black and white are exact fixed points of the chain.
    #[test]
    fn black_and_white_are_exact() {
        let black = display_pixel([0.0, 0.0, 0.0, 1.0], 0.0, 1.0, mode::RGBA, tonemap::NONE);
        assert_eq!(black[0], 0.0);
        assert_eq!(code8(black[0]), 0);

        let white = display_pixel([1.0, 1.0, 1.0, 1.0], 0.0, 1.0, mode::RGBA, tonemap::NONE);
        assert!((white[0] - 1.0).abs() < 1e-6, "linear 1.0 → {}", white[0]);
        assert_eq!(code8(white[0]), 255);
    }

    /// The OETF is continuous across the linear/power joint and monotonic.
    #[test]
    fn oetf_is_continuous_and_monotonic() {
        let below = oetf(SRGB_CUT - 1e-7);
        let above = oetf(SRGB_CUT + 1e-7);
        assert!(
            (below - above).abs() < 1e-4,
            "joint discontinuity {below} vs {above}"
        );

        let mut prev = -1.0;
        for i in 0..=1000 {
            let v = oetf(i as f32 / 1000.0);
            assert!(v >= prev, "not monotonic at {i}");
            prev = v;
        }
    }

    /// The extended curve: mirrored below zero (a negative is kept for an HDR
    /// surface, black on an SDR one), continued above 1.0, and inverted by `eotf`.
    #[test]
    fn extended_curve_mirrors_and_inverts() {
        assert_eq!(oetf(-1.0), -oetf(1.0));
        assert!((oetf(-1.0) + 1.0).abs() < 1e-6);
        assert_eq!(oetf(-0.18), -oetf(0.18));
        let out = display_pixel([-0.5, -0.5, -0.5, 1.0], 0.0, 1.0, mode::RGBA, tonemap::NONE);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(out[0], -oetf(0.5));
        assert_eq!(code8(out[0]), 0);
        assert!(oetf(4.0) > 1.0, "above white stays above white");
        for v in [-8.0f32, -0.5, -0.002, 0.0, 0.002, 0.003_130_8, 0.18, 1.0, 16.0] {
            let back = eotf(oetf(v));
            assert!((back - v).abs() <= 1e-6 + 1e-5 * v.abs(), "{v} -> {back}");
        }
        assert_eq!(dial(-4.0, 2.0), -2.0);
    }

    /// Exposure is a linear gain applied BEFORE the encode: +1 EV on 0.09 must
    /// land exactly where 0.18 does.
    #[test]
    fn exposure_is_a_pre_encode_linear_gain() {
        let a = display_pixel([0.09, 0.09, 0.09, 1.0], 1.0, 1.0, mode::RGBA, tonemap::NONE);
        let b = display_pixel([0.18, 0.18, 0.18, 1.0], 0.0, 1.0, mode::RGBA, tonemap::NONE);
        assert_eq!(code8(a[0]), code8(b[0]));
        assert_eq!(code8(a[0]), 118);
    }

    /// Channel isolate shows the picked component as grey; `A` keeps alpha,
    /// every other mode forces it opaque.
    #[test]
    fn channel_isolate_greys_and_alpha_rules() {
        let raw = [1.0, 0.18, 0.0, 0.25];
        let g = display_pixel(raw, 0.0, 1.0, mode::G, tonemap::NONE);
        assert_eq!(code8(g[0]), 118);
        assert_eq!(g[0], g[1]);
        assert_eq!(g[1], g[2]);
        assert_eq!(g[3], 1.0, "non-alpha isolate is opaque");

        let a = display_pixel(raw, 0.0, 1.0, mode::A, tonemap::NONE);
        assert_eq!(a[0], oetf(0.25), "alpha isolate shows alpha as grey");
        assert_eq!(a[3], 1.0, "and, like the display shader, opaque");

        // Rec.709 luma of pure red.
        let l = display_pixel([1.0, 0.0, 0.0, 1.0], 0.0, 1.0, mode::LUM, tonemap::NONE);
        assert_eq!(code8(l[0]), code8(oetf(LUMA_709[0])));
    }

    /// A gamma dial of 1.0 is the identity — the dial must not change the
    /// default look (the bug the old chain was "fixed" for).
    #[test]
    fn gamma_one_is_identity() {
        for v in [0.0_f32, 0.01, 0.18, 0.5, 1.0, 4.0] {
            let out = display_pixel([v; 4], 0.0, 1.0, mode::RGBA, tonemap::NONE);
            assert!((out[0] - oetf(v)).abs() < 1e-6, "gamma 1.0 changed {v}");
        }
        // And a zero gamma cannot produce NaN.
        let out = display_pixel([0.18; 4], 0.0, 0.0, mode::RGBA, tonemap::NONE);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// The generated WGSL carries the same literals the Rust constants hold —
    /// this is what makes "one source of truth" more than a comment.
    #[test]
    fn generated_wgsl_carries_the_constants() {
        let src = oetf_wgsl();
        assert!(src.contains("fn srgb_oetf"));
        for c in [SRGB_CUT, SRGB_SLOPE, SRGB_SCALE, SRGB_OFFSET, SRGB_EXP] {
            assert!(
                src.contains(&fl(c)),
                "WGSL is missing the literal for {c}:\n{src}"
            );
        }
        let inv = eotf_wgsl();
        for c in [SRGB_CUT * SRGB_SLOPE, SRGB_SLOPE, SRGB_SCALE, SRGB_OFFSET, 1.0 / SRGB_EXP] {
            assert!(inv.contains(&fl(c)), "EOTF WGSL is missing {c}:\n{inv}");
        }
        // `fl` must never emit a bare integer literal (WGSL would type it i32).
        assert_eq!(fl(12.92), "12.92");
        assert_eq!(fl(1.0), "1.0");
    }

    /// T walks None -> Reinhard -> AgX -> None; a code from nowhere (a hand-edited config)
    /// is `NONE`, named as what it renders.
    #[test]
    fn tone_operator_cycle_and_codes() {
        use tonemap::{AGX, NONE, REINHARD, known, name, next};
        assert_eq!([next(NONE), next(REINHARD), next(AGX)], [REINHARD, AGX, NONE]);
        assert_eq!([known(AGX), known(7)], [AGX, NONE]);
        assert_eq!([name(REINHARD), name(7)], ["Reinhard", "None"]);
    }

    /// The operators on the colour image: none is the identity, Reinhard is
    /// `c / (1 + c)`, AgX keeps a neutral grey neutral and is monotone and bounded;
    /// isolate modes are not tone-mapped.
    #[test]
    fn tone_operators() {
        assert_eq!(tone([0.18, 2.0, -1.0], tonemap::NONE), [0.18, 2.0, -1.0]);
        assert_eq!(tone([1.0, 3.0, -1.0], tonemap::REINHARD), [0.5, 0.75, 0.0]);
        let grey = tone([0.18; 3], tonemap::AGX);
        assert!((grey[0] - grey[1]).abs() < 2e-3 && (grey[1] - grey[2]).abs() < 2e-3, "{grey:?}");
        let mut last = -1.0;
        for i in 0..40 {
            let v = tone([2f32.powf(i as f32 * 0.5 - 12.0); 3], tonemap::AGX)[1];
            assert!((0.0..=1.0).contains(&v) && v >= last, "stop {i}: {v} after {last}");
            last = v;
        }
        assert!(last > 0.9, "highlights reach white: {last}");
        let raw = [4.0, 0.5, 0.25, 1.0];
        assert_eq!(display_pixel(raw, 0.0, 1.0, mode::R, tonemap::AGX), display_pixel(raw, 0.0, 1.0, mode::R, tonemap::NONE));
        assert_ne!(display_pixel(raw, 0.0, 1.0, mode::RGBA, tonemap::AGX), display_pixel(raw, 0.0, 1.0, mode::RGBA, tonemap::NONE));
    }
}
