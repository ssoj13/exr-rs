//! C++ Imath golden-parity test for `imath-rs`.
//!
//! Reads the committed `tests/parity/golden.txt` (produced by the C++ Imath
//! oracle via `cargo xtask parity`), recomputes the SAME operations with
//! `imath-rs`, and asserts agreement:
//!
//! * **BIT-EXACT** (`to_bits()` / raw integer equality) for everything whose
//!   result is exactly representable from the inputs through correctly-rounded
//!   IEEE operations: half<->float conversions, matrix multiply / determinant /
//!   inverse / transpose / multVec / multDir, dot / cross, length / length2 /
//!   normalize (`sqrt` is IEEE correctly-rounded, hence bit-exact), HSV<->RGB
//!   (no transcendentals), packed colors, and the Rand32/Rand48 sequences.
//! * **TIGHT ULP** (a few ULP / ~1e-9 rel, documented per group) only where the
//!   value depends on a transcendental libm call (`sin/cos/tan/asin/acos/atan2`):
//!   quat slerp / setRotation / axis-angle matrix, Euler matrix & extraction,
//!   frustum fov. These can differ between MSVC's libm and Rust's libm in the
//!   last bit(s); a real algorithmic divergence would blow far past the tolerance.
//!
//! ## How values are parsed (exactly)
//! The oracle prints every float as `KEY=<hexfloat> bits=<hex>` where `<hex>` is
//! the raw IEEE bit pattern (8 nibbles for f32, 16 for f64). We parse the EXACT
//! value from `bits=` via `f32::from_bits` / `f64::from_bits` — no decimal or
//! hex-float text parsing, so there is zero ingestion rounding. Integer keys
//! (`%04x`/`%08x`/`%d`) are parsed directly.
//!
//! ## Imath 3.2.2 vs ported 3.2.0
//! `imath-rs` was ported from Imath v3.2.0; vcpkg ships 3.2.2 (a patch release —
//! the algorithms are identical). If a value ever mismatches, the failure message
//! is intended to make triage easy: bit-exact groups failing => a real port bug;
//! a tight-ULP group failing by 1–2 ULP => libm rounding (MSVC vs Rust), not a bug.
//!
//! If `golden.txt` is missing the test FAILS. A parity gate that reports success
//! because it did not run is worse than no gate. Set `EXR_ALLOW_SKIP=1` to opt
//! out explicitly (e.g. on a machine without the C++ toolchain).

use std::collections::HashMap;
use std::path::PathBuf;

use imath_rs::color_algo::{
    hsv2rgb, packed2rgb, packed2rgb_c4_int, packed2rgb_int, rgb2hsv, rgb2packed,
};
use imath_rs::euler::{Euler, InputLayout, Order};
use imath_rs::matrix_algo::{
    extract_shrt_m44_euler, extract_shrt_m44_vec, jacobi_eigen_solver_m33_default,
    jacobi_eigen_solver_m44_default, procrustes_rotation_and_translation,
};
use imath_rs::quat::{Quatd, slerp};
use imath_rs::vec::Vec3;
use imath_rs::{
    Box, Color4, Frustumd, Frustumf, Half, Line3f, M33f, M44d, M44f, Plane3f, Rand32, Rand48,
    Sphere3f, V2f, V3d, V3f, V4d, V4f, box_transform,
};

// ===========================================================================
// Golden-file ingestion
// ===========================================================================

/// One parsed golden record: raw value plus, for floats, the exact IEEE bits.
struct Golden {
    map: HashMap<String, String>,
    /// Number of `KEY=...` records parsed (for the final count print).
    count: usize,
}

impl Golden {
    /// Get the raw value string for `key`, panicking with a clear message if the
    /// key is absent (means the oracle and test drifted out of sync).
    fn raw(&self, key: &str) -> &str {
        self.map
            .get(key)
            .unwrap_or_else(|| panic!("golden.txt missing key `{key}` (oracle/test out of sync)"))
    }

    /// Exact f32 from the `bits=<8 hex>` tail the oracle prints for f32 keys.
    fn f32(&self, key: &str) -> f32 {
        let v = self.raw(key);
        let bits = bits_tail(v)
            .unwrap_or_else(|| panic!("key `{key}` has no `bits=` tail: `{v}`"));
        f32::from_bits(bits as u32)
    }

    /// Exact f64 from the `bits=<16 hex>` tail the oracle prints for f64 keys.
    fn f64(&self, key: &str) -> f64 {
        let v = self.raw(key);
        let bits =
            bits_tail(v).unwrap_or_else(|| panic!("key `{key}` has no `bits=` tail: `{v}`"));
        f64::from_bits(bits)
    }

    /// Parse a plain hex integer value (`%04x`/`%08x`).
    fn hex(&self, key: &str) -> u64 {
        let v = self.raw(key);
        u64::from_str_radix(v.trim(), 16)
            .unwrap_or_else(|_| panic!("key `{key}` value `{v}` is not hex"))
    }

    /// Parse a plain decimal integer value (`%d`/`%lld`).
    fn int(&self, key: &str) -> i64 {
        let v = self.raw(key);
        v.trim()
            .parse()
            .unwrap_or_else(|_| panic!("key `{key}` value `{v}` is not an integer"))
    }
}

/// Extract the `bits=<hex>` token from a value line and parse it as u64.
fn bits_tail(value: &str) -> Option<u64> {
    for tok in value.split_whitespace() {
        if let Some(hex) = tok.strip_prefix("bits=") {
            return u64::from_str_radix(hex, 16).ok();
        }
    }
    None
}

/// Locate and parse `tests/parity/golden.txt`. Returns `None` if absent.
fn load_golden() -> Option<Golden> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("golden.txt");
    let text = std::fs::read_to_string(&path).ok()?;
    let mut map = HashMap::new();
    let mut count = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            map.insert(key.trim().to_string(), val.trim().to_string());
            count += 1;
        }
    }
    Some(Golden { map, count })
}

// ===========================================================================
// Assertion helpers
// ===========================================================================

/// Running tally so the final report can state the bit-exact / tolerance split.
#[derive(Default)]
struct Tally {
    bit_exact: usize,
    ulp: usize,
}

/// Bit-exact f32 comparison (`to_bits`). Use for every op the policy marks exact.
fn eq_f32(t: &mut Tally, key: &str, got: f32, want: f32) {
    t.bit_exact += 1;
    assert!(
        got.to_bits() == want.to_bits(),
        "BIT-EXACT MISMATCH `{key}`:\n  C++  = {want:?} (bits {:08x})\n  Rust = {got:?} (bits {:08x})\n  => bit-exact group failing means a REAL imath-rs port bug (do NOT loosen).",
        want.to_bits(),
        got.to_bits()
    );
}

/// Bit-exact f64 comparison (`to_bits`).
fn eq_f64(t: &mut Tally, key: &str, got: f64, want: f64) {
    t.bit_exact += 1;
    assert!(
        got.to_bits() == want.to_bits(),
        "BIT-EXACT MISMATCH `{key}`:\n  C++  = {want:?} (bits {:016x})\n  Rust = {got:?} (bits {:016x})\n  => bit-exact group failing means a REAL imath-rs port bug (do NOT loosen).",
        want.to_bits(),
        got.to_bits()
    );
}

/// Tight-ULP f64 comparison for transcendental-dependent results.
///
/// Tolerance: max(`abs_eps`, `rel * |want|`). Default callers use abs 1e-12 and
/// rel 1e-9 (~a handful of ULP near unit magnitude) — far tighter than any real
/// algorithmic error, loose enough only for MSVC-vs-Rust libm last-bit drift.
fn near_f64(t: &mut Tally, key: &str, got: f64, want: f64, abs_eps: f64, rel: f64) {
    t.ulp += 1;
    let tol = abs_eps.max(rel * want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "TIGHT-ULP MISMATCH `{key}`:\n  C++  = {want:?}\n  Rust = {got:?}\n  |diff| = {diff:e} > tol {tol:e}\n  => if diff is ~1-2 ULP it is MSVC-vs-Rust libm rounding; a large diff is a port bug.",
    );
}

/// Tight-ULP f32 comparison.
fn near_f32(t: &mut Tally, key: &str, got: f32, want: f32, abs_eps: f32, rel: f32) {
    t.ulp += 1;
    let tol = abs_eps.max(rel * want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "TIGHT-ULP MISMATCH `{key}`:\n  C++  = {want:?}\n  Rust = {got:?}\n  |diff| = {diff:e} > tol {tol:e}",
    );
}

// ===========================================================================
// The single integration test
// ===========================================================================

#[test]
fn parity_golden() {
    let Some(g) = load_golden() else {
        if std::env::var_os("EXR_ALLOW_SKIP").is_some_and(|v| v == "1") {
            eprintln!(
                "SKIP parity_golden (EXR_ALLOW_SKIP=1): tests/parity/golden.txt not found."
            );
            return;
        }
        panic!(
            "parity_golden: tests/parity/golden.txt not found -- the C++ parity gate did NOT run. \
             Run `cargo xtask parity` (Windows + vcpkg Imath) to generate it, or set \
             EXR_ALLOW_SKIP=1 to skip it deliberately."
        );
    };
    let mut t = Tally::default();

    check_half(&g, &mut t);
    check_vec(&g, &mut t);
    check_matrix(&g, &mut t);
    check_matrix_algo(&g, &mut t);
    check_quat(&g, &mut t);
    check_euler(&g, &mut t);
    check_color(&g, &mut t);
    check_color_int(&g, &mut t);
    check_geometry(&g, &mut t);
    check_frustum_depth(&g, &mut t);
    check_random(&g, &mut t);

    eprintln!(
        "parity_golden: {} golden records, {} bit-exact + {} tight-ULP checks passed.",
        g.count, t.bit_exact, t.ulp
    );
}

// ---------------------------------------------------------------------------
// half  (BIT-EXACT)
// ---------------------------------------------------------------------------

fn check_half(g: &Golden, t: &mut Tally) {
    // u16 -> f32: iterate over every key the oracle emitted for this group.
    for key in g.map.keys() {
        // The two aggregate (exhaustive) keys are handled below, not per-pattern.
        if key.ends_with(".fnv1a64") {
            continue;
        }
        if let Some(hex) = key.strip_prefix("half.u16_to_f32.") {
            let bits = u16::from_str_radix(hex, 16).unwrap();
            let got = Half::from_bits(bits).to_f32();
            eq_f32(t, key, got, g.f32(key));
        } else if let Some(hex) = key.strip_prefix("half.f32_to_u16.") {
            let fbits = u32::from_str_radix(hex, 16).unwrap();
            let got = Half::from_f32(f32::from_bits(fbits)).bits();
            let want = g.hex(key) as u16;
            t.bit_exact += 1;
            assert_eq!(got, want, "half f32->u16 bits mismatch `{key}`");
        } else if let Some(hex) = key.strip_prefix("half.flags.") {
            let bits = u16::from_str_radix(hex, 16).unwrap();
            let h = Half::from_bits(bits);
            let mut flags = 0u32;
            flags |= h.is_finite() as u32;
            flags |= (h.is_normalized() as u32) << 1;
            flags |= (h.is_denormalized() as u32) << 2;
            flags |= (h.is_zero() as u32) << 3;
            flags |= (h.is_nan() as u32) << 4;
            flags |= (h.is_infinity() as u32) << 5;
            flags |= (h.is_negative() as u32) << 6;
            let want = g.hex(key) as u32;
            t.bit_exact += 1;
            assert_eq!(flags, want, "half classification flags mismatch `{key}`");
        }
    }

    // EXHAUSTIVE half <-> float. The oracle folds all 65536 conversions into one
    // FNV-1a-64 hash; we recompute it with the SAME byte order here. A single
    // differing conversion (NaN payload, denormal, rounding) changes the hash.
    {
        let mut h = FNV_OFFSET;
        for i in 0..=u16::MAX {
            let fb = Half::from_bits(i).to_f32().to_bits();
            for b in fb.to_le_bytes() {
                h = fnv1a64(h, b);
            }
        }
        t.bit_exact += 1;
        let want = g.hex("half.u16_to_f32.all.fnv1a64");
        assert_eq!(
            h, want,
            "EXHAUSTIVE half->float mismatch: FNV-1a-64 over all 65536 conversions \
             is {h:016x}, C++ Imath gives {want:016x}. At least one of the 65536 \
             half bit patterns converts differently."
        );
    }
    {
        let mut h = FNV_OFFSET;
        for i in 0..=u16::MAX {
            let back = Half::from_f32(Half::from_bits(i).to_f32()).bits();
            for b in back.to_le_bytes() {
                h = fnv1a64(h, b);
            }
        }
        t.bit_exact += 1;
        let want = g.hex("half.f32_to_u16.roundtrip.fnv1a64");
        assert_eq!(
            h, want,
            "EXHAUSTIVE float->half round-trip mismatch: got {h:016x}, C++ {want:016x}."
        );
    }
}

/// FNV-1a-64 offset basis (same constant the oracle seeds with).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// One FNV-1a-64 step, byte for byte identical to the oracle's inner loop.
fn fnv1a64(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

// ---------------------------------------------------------------------------
// Vec3 / Vec4  (BIT-EXACT — sqrt is correctly rounded)
// ---------------------------------------------------------------------------

fn check_vec(g: &Golden, t: &mut Tally) {
    {
        let a = V3f::new(1.5, -2.25, 3.75);
        let b = V3f::new(-0.5, 4.0, 2.0);
        eq_f32(t, "v3f.dot", a.dot(&b), g.f32("v3f.dot"));
        let c = a.cross(&b);
        eq_f32(t, "v3f.cross.x", c.x, g.f32("v3f.cross.x"));
        eq_f32(t, "v3f.cross.y", c.y, g.f32("v3f.cross.y"));
        eq_f32(t, "v3f.cross.z", c.z, g.f32("v3f.cross.z"));
        eq_f32(t, "v3f.length2", a.length2(), g.f32("v3f.length2"));
        eq_f32(t, "v3f.length", a.length(), g.f32("v3f.length"));
        let n = a.normalized();
        eq_f32(t, "v3f.norm.x", n.x, g.f32("v3f.norm.x"));
        eq_f32(t, "v3f.norm.y", n.y, g.f32("v3f.norm.y"));
        eq_f32(t, "v3f.norm.z", n.z, g.f32("v3f.norm.z"));
    }
    {
        let a = V3d::new(1.5, -2.25, 3.75);
        let b = V3d::new(-0.5, 4.0, 2.0);
        eq_f64(t, "v3d.dot", a.dot(&b), g.f64("v3d.dot"));
        let c = a.cross(&b);
        eq_f64(t, "v3d.cross.x", c.x, g.f64("v3d.cross.x"));
        eq_f64(t, "v3d.cross.y", c.y, g.f64("v3d.cross.y"));
        eq_f64(t, "v3d.cross.z", c.z, g.f64("v3d.cross.z"));
        eq_f64(t, "v3d.length2", a.length2(), g.f64("v3d.length2"));
        eq_f64(t, "v3d.length", a.length(), g.f64("v3d.length"));
        let n = a.normalized();
        eq_f64(t, "v3d.norm.x", n.x, g.f64("v3d.norm.x"));
        eq_f64(t, "v3d.norm.y", n.y, g.f64("v3d.norm.y"));
        eq_f64(t, "v3d.norm.z", n.z, g.f64("v3d.norm.z"));
    }
    {
        let a = V4f::new(1.0, 2.0, -3.0, 0.5);
        let b = V4f::new(2.0, -1.0, 0.25, 4.0);
        eq_f32(t, "v4f.dot", a.dot(&b), g.f32("v4f.dot"));
        eq_f32(t, "v4f.length2", a.length2(), g.f32("v4f.length2"));
        eq_f32(t, "v4f.length", a.length(), g.f32("v4f.length"));
        let n = a.normalized();
        eq_f32(t, "v4f.norm.x", n.x, g.f32("v4f.norm.x"));
        eq_f32(t, "v4f.norm.y", n.y, g.f32("v4f.norm.y"));
        eq_f32(t, "v4f.norm.z", n.z, g.f32("v4f.norm.z"));
        eq_f32(t, "v4f.norm.w", n.w, g.f32("v4f.norm.w"));
    }
    {
        let a = V4d::new(1.0, 2.0, -3.0, 0.5);
        let b = V4d::new(2.0, -1.0, 0.25, 4.0);
        eq_f64(t, "v4d.dot", a.dot(&b), g.f64("v4d.dot"));
        eq_f64(t, "v4d.length2", a.length2(), g.f64("v4d.length2"));
        eq_f64(t, "v4d.length", a.length(), g.f64("v4d.length"));
        let n = a.normalized();
        eq_f64(t, "v4d.norm.x", n.x, g.f64("v4d.norm.x"));
        eq_f64(t, "v4d.norm.y", n.y, g.f64("v4d.norm.y"));
        eq_f64(t, "v4d.norm.z", n.z, g.f64("v4d.norm.z"));
        eq_f64(t, "v4d.norm.w", n.w, g.f64("v4d.norm.w"));
    }
}

// ---------------------------------------------------------------------------
// Matrix33 / Matrix44  (BIT-EXACT)
// ---------------------------------------------------------------------------

fn check_matrix(g: &Golden, t: &mut Tally) {
    let a3 = M33f::new(2.0, -1.0, 0.5, 0.25, 3.0, -2.0, 1.0, 0.5, 4.0);
    let b3 = M33f::new(1.0, 2.0, -0.5, -1.0, 0.25, 3.0, 0.5, -2.0, 1.0);
    let p3 = a3.mul_matrix(&b3);
    for i in 0..3 {
        for j in 0..3 {
            let k = format!("m33f.mul.{i}{j}");
            eq_f32(t, &k, p3[i][j], g.f32(&k));
        }
    }
    eq_f32(t, "m33f.det", a3.determinant(), g.f32("m33f.det"));
    let inv3 = a3.inverse();
    for i in 0..3 {
        for j in 0..3 {
            let k = format!("m33f.inv.{i}{j}");
            eq_f32(t, &k, inv3[i][j], g.f32(&k));
        }
    }
    let tr3 = a3.transposed();
    for i in 0..3 {
        for j in 0..3 {
            let k = format!("m33f.transpose.{i}{j}");
            eq_f32(t, &k, tr3[i][j], g.f32(&k));
        }
    }
    {
        let src = V2f::new(1.5, -2.0);
        let mut dst = V2f::new(0.0, 0.0);
        a3.mult_vec_matrix(&src, &mut dst);
        eq_f32(t, "m33f.multVec.x", dst.x, g.f32("m33f.multVec.x"));
        eq_f32(t, "m33f.multVec.y", dst.y, g.f32("m33f.multVec.y"));
        a3.mult_dir_matrix(&src, &mut dst);
        eq_f32(t, "m33f.multDir.x", dst.x, g.f32("m33f.multDir.x"));
        eq_f32(t, "m33f.multDir.y", dst.y, g.f32("m33f.multDir.y"));
    }

    let a4 = M44d::new(
        2.0, -1.0, 0.5, 0.25, 0.25, 3.0, -2.0, 1.0, 1.0, 0.5, 4.0, -0.5, -0.5, 0.25, 1.0, 2.0,
    );
    let b4 = M44d::new(
        1.0, 2.0, -0.5, 0.0, -1.0, 0.25, 3.0, 0.5, 0.5, -2.0, 1.0, -1.0, 0.25, 0.5, -0.25, 1.0,
    );
    let p4 = M44d::multiply(&a4, &b4);
    for i in 0..4 {
        for j in 0..4 {
            let k = format!("m44d.mul.{i}{j}");
            eq_f64(t, &k, p4[i][j], g.f64(&k));
        }
    }
    eq_f64(t, "m44d.det", a4.determinant(), g.f64("m44d.det"));
    let inv4 = a4.inverse();
    for i in 0..4 {
        for j in 0..4 {
            let k = format!("m44d.inv.{i}{j}");
            eq_f64(t, &k, inv4[i][j], g.f64(&k));
        }
    }
    let tr4 = a4.transposed();
    for i in 0..4 {
        for j in 0..4 {
            let k = format!("m44d.transpose.{i}{j}");
            eq_f64(t, &k, tr4[i][j], g.f64(&k));
        }
    }
    {
        let src = V3d::new(1.5, -2.0, 3.0);
        let mut dst = V3d::new(0.0, 0.0, 0.0);
        a4.mult_vec_matrix(&src, &mut dst);
        eq_f64(t, "m44d.multVec.x", dst.x, g.f64("m44d.multVec.x"));
        eq_f64(t, "m44d.multVec.y", dst.y, g.f64("m44d.multVec.y"));
        eq_f64(t, "m44d.multVec.z", dst.z, g.f64("m44d.multVec.z"));
        a4.mult_dir_matrix(&src, &mut dst);
        eq_f64(t, "m44d.multDir.x", dst.x, g.f64("m44d.multDir.x"));
        eq_f64(t, "m44d.multDir.y", dst.y, g.f64("m44d.multDir.y"));
        eq_f64(t, "m44d.multDir.z", dst.z, g.f64("m44d.multDir.z"));
    }
}

// ---------------------------------------------------------------------------
// matrix_algo: extractSHRT (ULP — trig), jacobi (ULP), procrustes (ULP)
// ---------------------------------------------------------------------------

fn check_matrix_algo(g: &Golden, t: &mut Tally) {
    check_shrt_failure(g, t);

    // extractSHRT — input matrix built from the SAME literal values as the oracle.
    {
        let m = M44d::new(
            0.96 * 2.0, 0.10, -0.20, 0.0, -0.15, 0.97 * 1.5, 0.05, 0.0, 0.20, -0.05, 0.98 * 0.5,
            0.0, 5.0, -3.0, 2.0, 1.0,
        );
        let mut s = V3d::new(0.0, 0.0, 0.0);
        let mut h = V3d::new(0.0, 0.0, 0.0);
        let mut r = V3d::new(0.0, 0.0, 0.0);
        let mut tr = V3d::new(0.0, 0.0, 0.0);
        let ok = extract_shrt_m44_vec(&m, &mut s, &mut h, &mut r, &mut tr, false).unwrap();
        t.bit_exact += 1;
        assert_eq!(ok as i64, g.int("shrt.ok"), "shrt.ok mismatch");
        // Scale/shear/translation are algebraic (bit-exact in practice), but the
        // rotation uses atan2 -> compare the whole SHRT with tight ULP for safety.
        near_f64(t, "shrt.s.x", s.x, g.f64("shrt.s.x"), 1e-12, 1e-9);
        near_f64(t, "shrt.s.y", s.y, g.f64("shrt.s.y"), 1e-12, 1e-9);
        near_f64(t, "shrt.s.z", s.z, g.f64("shrt.s.z"), 1e-12, 1e-9);
        near_f64(t, "shrt.h.x", h.x, g.f64("shrt.h.x"), 1e-12, 1e-9);
        near_f64(t, "shrt.h.y", h.y, g.f64("shrt.h.y"), 1e-12, 1e-9);
        near_f64(t, "shrt.h.z", h.z, g.f64("shrt.h.z"), 1e-12, 1e-9);
        near_f64(t, "shrt.r.x", r.x, g.f64("shrt.r.x"), 1e-12, 1e-9);
        near_f64(t, "shrt.r.y", r.y, g.f64("shrt.r.y"), 1e-12, 1e-9);
        near_f64(t, "shrt.r.z", r.z, g.f64("shrt.r.z"), 1e-12, 1e-9);
        near_f64(t, "shrt.t.x", tr.x, g.f64("shrt.t.x"), 1e-12, 1e-9);
        near_f64(t, "shrt.t.y", tr.y, g.f64("shrt.t.y"), 1e-12, 1e-9);
        near_f64(t, "shrt.t.z", tr.z, g.f64("shrt.t.z"), 1e-12, 1e-9);
    }
    // jacobiEigenSolver M33 eigenvalues. Jacobi sweeps use atan-style rotations
    // and an order-dependent reduction, so compare eigenvalues with tight ULP.
    {
        let mut a = imath_rs::M33d::new(4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0);
        let mut s = V3d::new(0.0, 0.0, 0.0);
        let mut v = imath_rs::M33d::default();
        jacobi_eigen_solver_m33_default(&mut a, &mut s, &mut v);
        near_f64(t, "jacobi33.s.x", s.x, g.f64("jacobi33.s.x"), 1e-10, 1e-9);
        near_f64(t, "jacobi33.s.y", s.y, g.f64("jacobi33.s.y"), 1e-10, 1e-9);
        near_f64(t, "jacobi33.s.z", s.z, g.f64("jacobi33.s.z"), 1e-10, 1e-9);
    }
    {
        let mut a = M44d::new(
            4.0, 1.0, 2.0, 0.5, 1.0, 5.0, 3.0, 1.0, 2.0, 3.0, 6.0, 1.5, 0.5, 1.0, 1.5, 7.0,
        );
        let mut s = V4d::new(0.0, 0.0, 0.0, 0.0);
        let mut v = M44d::default();
        jacobi_eigen_solver_m44_default(&mut a, &mut s, &mut v);
        near_f64(t, "jacobi44.s.x", s.x, g.f64("jacobi44.s.x"), 1e-10, 1e-9);
        near_f64(t, "jacobi44.s.y", s.y, g.f64("jacobi44.s.y"), 1e-10, 1e-9);
        near_f64(t, "jacobi44.s.z", s.z, g.f64("jacobi44.s.z"), 1e-10, 1e-9);
        near_f64(t, "jacobi44.s.w", s.w, g.f64("jacobi44.s.w"), 1e-10, 1e-9);
    }
    // procrustes (rigid). Uses an SVD with Jacobi rotations -> tight ULP.
    {
        let from = [
            V3d::new(0.0, 0.0, 0.0),
            V3d::new(1.0, 0.0, 0.0),
            V3d::new(0.0, 1.0, 0.0),
            V3d::new(0.0, 0.0, 1.0),
        ];
        // sin(45 deg) = 1/sqrt(2); the C++ oracle used the same decimal literal,
        // which rounds to this exact f64 (== FRAC_1_SQRT_2), keeping inputs bit-identical.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let to = [
            V3d::new(1.0, 2.0, 3.0),
            V3d::new(1.0, 2.0 + s, 3.0 + s),
            V3d::new(2.0, 2.0, 3.0),
            V3d::new(1.0, 2.0 + s, 3.0 - s),
        ];
        let res = procrustes_rotation_and_translation(&from, &to, 4, false);
        for i in 0..4 {
            for j in 0..4 {
                let k = format!("procrustes.{i}{j}");
                near_f64(t, &k, res[i][j], g.f64(&k), 1e-9, 1e-7);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quat (ULP — trig)
// ---------------------------------------------------------------------------

fn check_quat(g: &Golden, t: &mut Tally) {
    {
        let mut q = Quatd::identity();
        let mut axis = V3d::new(0.0, 0.0, 1.0);
        axis.normalize();
        // 45 deg = pi/4; identical f64 bits to the oracle's decimal literal.
        q.set_axis_angle(&axis, std::f64::consts::FRAC_PI_4);
        let m = q.to_matrix44();
        for i in 0..4 {
            for j in 0..4 {
                let k = format!("quat.axisangle.m{i}{j}");
                near_f64(t, &k, m[i][j], g.f64(&k), 1e-12, 1e-9);
            }
        }
    }
    {
        let mut q1 = Quatd::identity();
        q1.set_axis_angle(&V3d::new(0.0, 0.0, 1.0), 0.0);
        let mut q2 = Quatd::identity();
        // 90 deg = pi/2; identical f64 bits to the oracle's decimal literal.
        q2.set_axis_angle(&V3d::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        for (i, &tt) in [0.25, 0.5, 0.75].iter().enumerate() {
            let q = slerp(&q1, &q2, tt);
            near_f64(t, &format!("quat.slerp.{i}.r"), q.r, g.f64(&format!("quat.slerp.{i}.r")), 1e-12, 1e-9);
            near_f64(t, &format!("quat.slerp.{i}.x"), q.v.x, g.f64(&format!("quat.slerp.{i}.x")), 1e-12, 1e-9);
            near_f64(t, &format!("quat.slerp.{i}.y"), q.v.y, g.f64(&format!("quat.slerp.{i}.y")), 1e-12, 1e-9);
            near_f64(t, &format!("quat.slerp.{i}.z"), q.v.z, g.f64(&format!("quat.slerp.{i}.z")), 1e-12, 1e-9);
        }
    }
    {
        let mut q = Quatd::identity();
        q.set_rotation(&V3d::new(1.0, 0.0, 0.0), &V3d::new(0.0, 1.0, 0.0));
        near_f64(t, "quat.setrot.r", q.r, g.f64("quat.setrot.r"), 1e-12, 1e-9);
        near_f64(t, "quat.setrot.x", q.v.x, g.f64("quat.setrot.x"), 1e-12, 1e-9);
        near_f64(t, "quat.setrot.y", q.v.y, g.f64("quat.setrot.y"), 1e-12, 1e-9);
        near_f64(t, "quat.setrot.z", q.v.z, g.f64("quat.setrot.z"), 1e-12, 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Euler (ULP — trig in toMatrix44 / extract)
// ---------------------------------------------------------------------------

fn check_euler(g: &Golden, t: &mut Tally) {
    let cases = [("XYZ", Order::XYZ), ("ZYX", Order::ZYX), ("XZX", Order::XZX)];
    let ang = [0.3_f64, -0.5, 0.8];
    for (name, order) in cases {
        let e = Euler::from_angles(ang[0], ang[1], ang[2], order, InputLayout::IJKLayout);
        let m = e.to_matrix44();
        for i in 0..4 {
            for j in 0..4 {
                let k = format!("euler.{name}.m{i}{j}");
                near_f64(t, &k, m[i][j], g.f64(&k), 1e-12, 1e-9);
            }
        }
        let mut ex = Euler::from_order(order);
        ex.extract44(&m);
        near_f64(t, &format!("euler.{name}.extract.x"), ex.vec.x, g.f64(&format!("euler.{name}.extract.x")), 1e-9, 1e-8);
        near_f64(t, &format!("euler.{name}.extract.y"), ex.vec.y, g.f64(&format!("euler.{name}.extract.y")), 1e-9, 1e-8);
        near_f64(t, &format!("euler.{name}.extract.z"), ex.vec.z, g.f64(&format!("euler.{name}.extract.z")), 1e-9, 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Color (BIT-EXACT — no transcendentals)
// ---------------------------------------------------------------------------

fn check_color(g: &Golden, t: &mut Tally) {
    {
        let hsv = V3f::new(0.3333333, 0.5, 0.75);
        let rgb = hsv2rgb(hsv);
        eq_f32(t, "color.hsv2rgb.r", rgb.x, g.f32("color.hsv2rgb.r"));
        eq_f32(t, "color.hsv2rgb.g", rgb.y, g.f32("color.hsv2rgb.g"));
        eq_f32(t, "color.hsv2rgb.b", rgb.z, g.f32("color.hsv2rgb.b"));
        let hsv2 = rgb2hsv(rgb);
        eq_f32(t, "color.rgb2hsv.h", hsv2.x, g.f32("color.rgb2hsv.h"));
        eq_f32(t, "color.rgb2hsv.s", hsv2.y, g.f32("color.rgb2hsv.s"));
        eq_f32(t, "color.rgb2hsv.v", hsv2.z, g.f32("color.rgb2hsv.v"));
    }
    {
        let rgb = V3f::new(0.25, 0.5, 0.75);
        let p = rgb2packed(rgb);
        t.bit_exact += 1;
        assert_eq!(p, g.hex("color.rgb2packed") as u32, "rgb2packed mismatch");
        let mut back = V3f::new(0.0, 0.0, 0.0);
        packed2rgb(p, &mut back);
        // packed2rgb float branch now computes `T f = T(1)/T(255)` and
        // `byte * f` in T (= f32 here), exactly matching C++. BIT-EXACT.
        eq_f32(t, "color.packed2rgb.r", back.x, g.f32("color.packed2rgb.r"));
        eq_f32(t, "color.packed2rgb.g", back.y, g.f32("color.packed2rgb.g"));
        eq_f32(t, "color.packed2rgb.b", back.z, g.f32("color.packed2rgb.b"));
    }
    {
        // Color4 -> packed: rebuild via a Vec3 + alpha path is not equivalent;
        // use the Color4 overload to match the oracle exactly.
        let c = imath_rs::Color4f::new(0.1, 0.4, 0.7, 1.0);
        let p = imath_rs::color_algo::rgb2packed_c4(c);
        t.bit_exact += 1;
        assert_eq!(p, g.hex("color.rgb2packed.c4") as u32, "rgb2packed.c4 mismatch");
    }
}

// ---------------------------------------------------------------------------
// Geometry: box (exact), line (exact), plane (exact), sphere (exact, sqrt),
//           frustum projection (exact), frustum fov (ULP — atan)
// ---------------------------------------------------------------------------

fn check_geometry(g: &Golden, t: &mut Tally) {
    {
        let b = Box::from_min_max(V3f::new(-1.0, -2.0, -0.5), V3f::new(2.0, 1.0, 3.0));
        let m = M44f::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, -1.0, 2.0, 1.0,
        );
        let r = box_transform(&b, &m);
        eq_f32(t, "box.transform.min.x", r.min.x, g.f32("box.transform.min.x"));
        eq_f32(t, "box.transform.min.y", r.min.y, g.f32("box.transform.min.y"));
        eq_f32(t, "box.transform.min.z", r.min.z, g.f32("box.transform.min.z"));
        eq_f32(t, "box.transform.max.x", r.max.x, g.f32("box.transform.max.x"));
        eq_f32(t, "box.transform.max.y", r.max.y, g.f32("box.transform.max.y"));
        eq_f32(t, "box.transform.max.z", r.max.z, g.f32("box.transform.max.z"));
    }
    {
        let l = Line3f::from_points(V3f::new(0.0, 0.0, 0.0), V3f::new(1.0, 1.0, 0.0));
        let cp = l.closest_point_to_point(V3f::new(1.0, 0.0, 0.0));
        eq_f32(t, "line.closest.x", cp.x, g.f32("line.closest.x"));
        eq_f32(t, "line.closest.y", cp.y, g.f32("line.closest.y"));
        eq_f32(t, "line.closest.z", cp.z, g.f32("line.closest.z"));
    }
    {
        let pl = Plane3f::from_normal_distance(V3f::new(0.0, 0.0, 1.0), 2.0);
        let l = Line3f::from_points(V3f::new(0.0, 0.0, 0.0), V3f::new(0.5, 0.25, 1.0));
        let mut ip = V3f::new(0.0, 0.0, 0.0);
        let hit = pl.intersect(&l, &mut ip);
        t.bit_exact += 1;
        assert_eq!(hit as i64, g.int("plane.intersect.hit"), "plane hit mismatch");
        eq_f32(t, "plane.intersect.x", ip.x, g.f32("plane.intersect.x"));
        eq_f32(t, "plane.intersect.y", ip.y, g.f32("plane.intersect.y"));
        eq_f32(t, "plane.intersect.z", ip.z, g.f32("plane.intersect.z"));
    }
    {
        let sp = Sphere3f::from_center_radius(V3f::new(0.0, 0.0, 5.0), 2.0);
        let l = Line3f::from_points(V3f::new(0.0, 0.0, 0.0), V3f::new(0.0, 0.0, 1.0));
        let mut ip = V3f::new(0.0, 0.0, 0.0);
        let hit = sp.intersect(&l, &mut ip);
        t.bit_exact += 1;
        assert_eq!(hit as i64, g.int("sphere.intersect.hit"), "sphere hit mismatch");
        eq_f32(t, "sphere.intersect.x", ip.x, g.f32("sphere.intersect.x"));
        eq_f32(t, "sphere.intersect.y", ip.y, g.f32("sphere.intersect.y"));
        eq_f32(t, "sphere.intersect.z", ip.z, g.f32("sphere.intersect.z"));
    }
    {
        let f = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, false);
        let pm = f.projection_matrix();
        for i in 0..4 {
            for j in 0..4 {
                let k = format!("frustum.proj.{i}{j}");
                eq_f32(t, &k, pm[i][j], g.f32(&k));
            }
        }
        let mut planes = [Plane3f::new(); 6];
        f.planes_transformed(&mut planes, &M44f::default());
        for (p, plane) in planes.iter().enumerate() {
            eq_f32(t, &format!("frustum.plane{p}.n.x"), plane.normal.x, g.f32(&format!("frustum.plane{p}.n.x")));
            eq_f32(t, &format!("frustum.plane{p}.n.y"), plane.normal.y, g.f32(&format!("frustum.plane{p}.n.y")));
            eq_f32(t, &format!("frustum.plane{p}.n.z"), plane.normal.z, g.f32(&format!("frustum.plane{p}.n.z")));
            eq_f32(t, &format!("frustum.plane{p}.d"), plane.distance, g.f32(&format!("frustum.plane{p}.d")));
        }
        // fovx/fovy use atan -> tight ULP.
        near_f32(t, "frustum.fovx", f.fovx(), g.f32("frustum.fovx"), 1e-6, 1e-6);
        near_f32(t, "frustum.fovy", f.fovy(), g.f32("frustum.fovy"), 1e-6, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Random (BIT-EXACT — pure integer LCGs + deterministic float assembly)
// ---------------------------------------------------------------------------

fn check_random(g: &Golden, t: &mut Tally) {
    {
        let mut r = Rand32::new(12345);
        for i in 0..8 {
            let k = format!("rand32.nexti.{i}");
            t.bit_exact += 1;
            assert_eq!(r.nexti(), g.hex(&k) as u32, "rand32 nexti mismatch `{k}`");
        }
        let mut rf = Rand32::new(12345);
        for i in 0..8 {
            let k = format!("rand32.nextf.{i}");
            eq_f32(t, &k, rf.nextf(), g.f32(&k));
        }
    }
    {
        let mut r = Rand48::new(12345);
        for i in 0..8 {
            let k = format!("rand48.nexti.{i}");
            t.bit_exact += 1;
            assert_eq!(r.nexti(), g.int(&k), "rand48 nexti mismatch `{k}`");
        }
        let mut rf = Rand48::new(12345);
        for i in 0..8 {
            let k = format!("rand48.nextf.{i}");
            eq_f64(t, &k, rf.nextf(), g.f64(&k));
        }
    }
}

// ---------------------------------------------------------------------------
// packed2rgb integer branch (BIT-EXACT -- pure integer arithmetic)
// ---------------------------------------------------------------------------

fn check_color_int(g: &Golden, t: &mut Tally) {
    for packed in [0x0000_0000u32, 0xFFFF_FFFF, 0x1234_5678, 0x0080_FF7F] {
        let mut vu8 = Vec3::new(0u8, 0, 0);
        packed2rgb_int(packed, &mut vu8);
        let mut vi16 = Vec3::new(0i16, 0, 0);
        packed2rgb_int(packed, &mut vi16);
        let mut vi32 = Vec3::new(0i32, 0, 0);
        packed2rgb_int(packed, &mut vi32);
        let mut vi64 = Vec3::new(0i64, 0, 0);
        packed2rgb_int(packed, &mut vi64);
        let mut ci64 = Color4::new(0i64, 0, 0, 0);
        packed2rgb_c4_int(packed, &mut ci64);

        let got: [(&str, i64); 16] = [
            ("u8.x", vu8.x as i64), ("u8.y", vu8.y as i64), ("u8.z", vu8.z as i64),
            ("i16.x", vi16.x as i64), ("i16.y", vi16.y as i64), ("i16.z", vi16.z as i64),
            ("i32.x", vi32.x as i64), ("i32.y", vi32.y as i64), ("i32.z", vi32.z as i64),
            ("i64.x", vi64.x), ("i64.y", vi64.y), ("i64.z", vi64.z),
            ("c4i64.r", ci64.r), ("c4i64.g", ci64.g), ("c4i64.b", ci64.b), ("c4i64.a", ci64.a),
        ];
        for (suffix, value) in got {
            let (ty, comp) = suffix.split_once('.').expect("suffix has a component");
            let key = format!("color.packed2rgb.{ty}.{packed:08x}.{comp}");
            t.bit_exact += 1;
            assert_eq!(
                value,
                g.int(&key),
                "packed2rgb integer branch mismatch `{key}`: the C++ divisor \
                 `numeric_limits<T>::max() / 0xFF` is computed in T, not in u32"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Frustum ZToDepth / DepthToZ (BIT-EXACT -- only +-*/ and one float->int cast)
// ---------------------------------------------------------------------------

fn check_frustum_depth(g: &Golden, t: &mut Tally) {
    let fp = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, false);
    let fo = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, true);
    let dp = Frustumd::from_params(1.0, 1000.0, -1.0, 1.0, 1.0, -1.0, false);

    let ranges: [(&str, i64, i64); 4] = [
        ("u16", 0, 65535),
        ("u24", 0, 16_777_215),
        ("i31", 0, 2_000_000_000),
        ("sym", -1000, 1000),
    ];
    let frac = [0.0f64, 0.25, 0.5, 0.75, 1.0];

    for (name, zmin, zmax) in ranges {
        for (i, f) in frac.iter().enumerate() {
            let zval = zmin + (f * (zmax - zmin) as f64) as i64;
            let k = format!("frustum.ztodepth.persp.{name}.{i}");
            eq_f32(t, &k, fp.z_to_depth(zval, zmin, zmax), g.f32(&k));
            let k = format!("frustum.ztodepth.ortho.{name}.{i}");
            eq_f32(t, &k, fo.z_to_depth(zval, zmin, zmax), g.f32(&k));
            let k = format!("frustum.ztodepth.d.{name}.{i}");
            eq_f64(t, &k, dp.z_to_depth(zval, zmin, zmax), g.f64(&k));
        }
    }

    let fdepths = [-1.0f32, -2.0, -10.0, -50.0, -100.0];
    let ddepths = [-1.0f64, -2.0, -10.0, -50.0, -1000.0];
    for (name, zmin, zmax) in ranges {
        for i in 0..5 {
            let k = format!("frustum.depthtoz.persp.{name}.{i}");
            t.bit_exact += 1;
            assert_eq!(fp.depth_to_z(fdepths[i], zmin, zmax), g.int(&k), "`{k}`");
            let k = format!("frustum.depthtoz.ortho.{name}.{i}");
            t.bit_exact += 1;
            assert_eq!(fo.depth_to_z(fdepths[i], zmin, zmax), g.int(&k), "`{k}`");
            let k = format!("frustum.depthtoz.d.{name}.{i}");
            t.bit_exact += 1;
            assert_eq!(dp.depth_to_z(ddepths[i], zmin, zmax), g.int(&k), "`{k}`");
        }
    }
}

// ---------------------------------------------------------------------------
// extractSHRT(Euler) failure path: C++ leaves the caller's angles untouched
// ---------------------------------------------------------------------------

fn check_shrt_failure(g: &Golden, t: &mut Tally) {
    // Row 0 is all zero -> extractAndRemoveScalingAndShear fails at
    // checkForZeroScaleInRow (ImathMatrixAlgo.h:587) and extractSHRT returns
    // false without writing `r` (:840). The oracle asserts that C++ really does
    // take the failure path, so a future change to the input can not silently
    // turn this into a success-path test.
    assert_eq!(
        g.int("shrt.fail.ok"),
        0,
        "oracle input is no longer singular -- the extractSHRT FAILURE path is untested"
    );

    let sing = M44d::new(
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let mut s = V3d::new(0.0, 0.0, 0.0);
    let mut h = V3d::new(0.0, 0.0, 0.0);
    let mut tr = V3d::new(0.0, 0.0, 0.0);
    let mut r = Euler::from_vec_order_layout(
        V3d::new(0.25, -0.5, 0.75),
        Order::XYZ,
        InputLayout::IJKLayout,
    );
    let ok = extract_shrt_m44_euler(&sing, &mut s, &mut h, &mut r, &mut tr, false)
        .expect("exc=false never returns Err");
    t.bit_exact += 1;
    assert!(!ok, "extractSHRT should report failure on a singular matrix");
    eq_f64(t, "shrt.fail.r.x", r.vec.x, g.f64("shrt.fail.r.x"));
    eq_f64(t, "shrt.fail.r.y", r.vec.y, g.f64("shrt.fail.r.y"));
    eq_f64(t, "shrt.fail.r.z", r.vec.z, g.f64("shrt.fail.r.z"));
}
