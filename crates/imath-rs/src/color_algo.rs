//! Port of Imath ImathColorAlgo.h + ImathColorAlgo.cpp.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathColorAlgo.h>
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathColorAlgo.cpp>
//!
//! ## Functions ported
//! - [`hsv2rgb_d`] (Vec3<f64> overload) — non-templated helper from the `.cpp`
//! - [`hsv2rgb_d_c4`] (Color4<f64> overload) — non-templated helper from the `.cpp`
//! - [`rgb2hsv_d`] (Vec3<f64> overload) — non-templated helper from the `.cpp`
//! - [`rgb2hsv_d_c4`] (Color4<f64> overload) — non-templated helper from the `.cpp`
//! - [`hsv2rgb`] (Vec3<T> template overload)
//! - [`hsv2rgb_c4`] (Color4<T> template overload)
//! - [`rgb2hsv`] (Vec3<T> template overload)
//! - [`rgb2hsv_c4`] (Color4<T> template overload)
//! - [`rgb2packed`] (Vec3<T> overload)
//! - [`rgb2packed_c4`] (Color4<T> overload)
//! - [`packed2rgb`] (Vec3<T> out-param overload)
//! - [`packed2rgb_c4`] (Color4<T> out-param overload)
//!
//! ## C++ function naming vs Rust
//! C++ overloads `hsv2rgb` on `Vec3`/`Color4` and `rgb2packed`/`packed2rgb` on the
//! same two types — resolved at call-site by type. Rust cannot overload, so the
//! `Color4` variants are named with a `_c4` suffix (e.g. `hsv2rgb_c4`). The
//! non-templated `hsv2rgb_d`/`rgb2hsv_d` helpers are two C++ overloads; the
//! `Color4<double>` variants are `hsv2rgb_d_c4` / `rgb2hsv_d_c4`.
//!
//! ## `is_integer` dispatch
//! C++ uses `std::numeric_limits<T>::is_integer` to branch between two code paths.
//! Rust encodes this as the [`IsInteger`] marker trait: implemented for `u8` /
//! `i16` / `i32` / `i64`; NOT implemented for `f32` / `f64` / `Half`. The
//! templates dispatch via two trait impls (`ColorHsvFloat` / `ColorHsvInt`).

use crate::color::{Color4, PackedColor};
use crate::vec::Vec3;

// ============================================================================
// Non-templated double helpers (port of ImathColorAlgo.cpp)
// ============================================================================

/// Convert HSV → RGB for `Vec3<f64>` (non-templated helper; port of C++
/// `Vec3<double> hsv2rgb_d(const Vec3<double>& hsv)`).
///
/// Algorithm: standard 6-sector HSV wheel. Matches the C++ switch exactly.
#[inline]
pub fn hsv2rgb_d(hsv: Vec3<f64>) -> Vec3<f64> {
    let mut hue = hsv.x;
    let sat = hsv.y;
    let val = hsv.z;

    let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);

    if hue == 1.0 {
        hue = 0.0;
    } else {
        hue *= 6.0;
    }

    let i = hue.floor() as i32;
    let f = hue - i as f64;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t = val * (1.0 - sat * (1.0 - f));

    match i {
        0 => { x = val; y = t;   z = p;   }
        1 => { x = q;   y = val; z = p;   }
        2 => { x = p;   y = val; z = t;   }
        3 => { x = p;   y = q;   z = val; }
        4 => { x = t;   y = p;   z = val; }
        5 => { x = val; y = p;   z = q;   }
        _ => {}
    }

    Vec3::new(x, y, z)
}

/// Convert HSV → RGB for `Color4<f64>` (non-templated helper; port of C++
/// `Color4<double> hsv2rgb_d(const Color4<double>& hsv)`).
///
/// Alpha channel is passed through unchanged, matching the C++ `Color4<double>(r, g, b, hsv.a)`.
#[inline]
pub fn hsv2rgb_d_c4(hsv: Color4<f64>) -> Color4<f64> {
    let mut hue = hsv.r;
    let sat = hsv.g;
    let val = hsv.b;

    let (mut r, mut g, mut b) = (0.0_f64, 0.0_f64, 0.0_f64);

    if hue == 1.0 {
        hue = 0.0;
    } else {
        hue *= 6.0;
    }

    let i = hue.floor() as i32;
    let f = hue - i as f64;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t = val * (1.0 - sat * (1.0 - f));

    match i {
        0 => { r = val; g = t;   b = p;   }
        1 => { r = q;   g = val; b = p;   }
        2 => { r = p;   g = val; b = t;   }
        3 => { r = p;   g = q;   b = val; }
        4 => { r = t;   g = p;   b = val; }
        5 => { r = val; g = p;   b = q;   }
        _ => {}
    }

    Color4::new(r, g, b, hsv.a)
}

/// Convert RGB → HSV for `Vec3<f64>` (non-templated helper; port of C++
/// `Vec3<double> rgb2hsv_d(const Vec3<double>& c)`).
#[inline]
pub fn rgb2hsv_d(c: Vec3<f64>) -> Vec3<f64> {
    let x = c.x;
    let y = c.y;
    let z = c.z;

    let max = if x > y { if x > z { x } else { z } } else { if y > z { y } else { z } };
    let min = if x < y { if x < z { x } else { z } } else { if y < z { y } else { z } };
    let range = max - min;
    let val = max;
    let mut sat = 0.0_f64;
    let mut hue = 0.0_f64;

    if max != 0.0 {
        sat = range / max;
    }

    if sat != 0.0 {
        let h = if x == max {
            (y - z) / range
        } else if y == max {
            2.0 + (z - x) / range
        } else {
            4.0 + (x - y) / range
        };

        hue = h / 6.0;
        if hue < 0.0 {
            hue += 1.0;
        }
    }

    Vec3::new(hue, sat, val)
}

/// Convert RGB → HSV for `Color4<f64>` (non-templated helper; port of C++
/// `Color4<double> rgb2hsv_d(const Color4<double>& c)`).
///
/// Alpha channel is passed through unchanged.
#[inline]
pub fn rgb2hsv_d_c4(c: Color4<f64>) -> Color4<f64> {
    let r = c.r;
    let g = c.g;
    let b = c.b;

    let max = if r > g { if r > b { r } else { b } } else { if g > b { g } else { b } };
    let min = if r < g { if r < b { r } else { b } } else { if g < b { g } else { b } };
    let range = max - min;
    let val = max;
    let mut sat = 0.0_f64;
    let mut hue = 0.0_f64;

    if max != 0.0 {
        sat = range / max;
    }

    if sat != 0.0 {
        let h = if r == max {
            (g - b) / range
        } else if g == max {
            2.0 + (b - r) / range
        } else {
            4.0 + (r - g) / range
        };

        hue = h / 6.0;
        if hue < 0.0 {
            hue += 1.0;
        }
    }

    Color4::new(hue, sat, val, c.a)
}

// ============================================================================
// IsInteger marker trait — encodes `std::numeric_limits<T>::is_integer`
// ============================================================================

/// Marker trait for integer element types, encoding C++
/// `std::numeric_limits<T>::is_integer`. Implemented for `u8`, `i16`, `i32`,
/// `i64`; NOT implemented for `f32`, `f64`, `Half`. Used to select the integer
/// vs floating-point branch in the template HSV/packed conversions.
pub trait IsInteger {}
impl IsInteger for u8  {}
impl IsInteger for i16 {}
impl IsInteger for i32 {}
impl IsInteger for i64 {}

// ============================================================================
// Templated hsv2rgb (Vec3<T>) — port of the template in ImathColorAlgo.h
// ============================================================================

/// Convert HSV → RGB for `Vec3<T>` (float branch — `T` not integer).
///
/// C++ template body: promotes to `Vec3<double>`, calls `hsv2rgb_d`, casts back.
/// This is the float/half code path (`std::numeric_limits<T>::is_integer == false`).
#[inline]
pub fn hsv2rgb<T>(hsv: Vec3<T>) -> Vec3<T>
where
    T: Copy + ColorFromF64,
{
    let v = Vec3::new(hsv.x.to_f64(), hsv.y.to_f64(), hsv.z.to_f64());
    let c = hsv2rgb_d(v);
    Vec3::new(T::from_f64(c.x), T::from_f64(c.y), T::from_f64(c.z))
}

/// Convert HSV → RGB for `Vec3<T>` where `T` is an integer type.
///
/// C++ integer branch: normalise by `numeric_limits<T>::max()` into `[0,1]`,
/// call `hsv2rgb_d`, scale back by `max()`.
#[inline]
pub fn hsv2rgb_int<T>(hsv: Vec3<T>) -> Vec3<T>
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx = T::int_max_as_f64();
    let v = Vec3::new(
        hsv.x.to_f64() / mx,
        hsv.y.to_f64() / mx,
        hsv.z.to_f64() / mx,
    );
    let c = hsv2rgb_d(v);
    Vec3::new(
        T::from_f64(c.x * mx),
        T::from_f64(c.y * mx),
        T::from_f64(c.z * mx),
    )
}

// ============================================================================
// Templated hsv2rgb (Color4<T>)
// ============================================================================

/// Convert HSV → RGB for `Color4<T>` (float branch).
///
/// C++ uses `float(numeric_limits<T>::max())` — note: `float`, not `double`, for
/// the Color4 integer branch in the header. Matched exactly in [`hsv2rgb_c4_int`].
#[inline]
pub fn hsv2rgb_c4<T>(hsv: Color4<T>) -> Color4<T>
where
    T: Copy + ColorFromF64,
{
    let v = Color4::new(hsv.r.to_f64(), hsv.g.to_f64(), hsv.b.to_f64(), hsv.a.to_f64());
    let c = hsv2rgb_d_c4(v);
    Color4::new(T::from_f64(c.r), T::from_f64(c.g), T::from_f64(c.b), T::from_f64(c.a))
}

/// Convert HSV → RGB for `Color4<T>` where `T` is an integer type.
///
/// C++ integer branch uses `float(numeric_limits<T>::max())` (note: `float` cast,
/// not `double` — the C++ header has `hsv.r / float(...)` for Color4). We match
/// this by using `f32` as the intermediate normalisation factor.
#[inline]
pub fn hsv2rgb_c4_int<T>(hsv: Color4<T>) -> Color4<T>
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx_f = T::int_max_as_f32();
    let mx_d = mx_f as f64;
    let v = Color4::new(
        hsv.r.to_f64() / mx_d,
        hsv.g.to_f64() / mx_d,
        hsv.b.to_f64() / mx_d,
        hsv.a.to_f64() / mx_d,
    );
    let c = hsv2rgb_d_c4(v);
    Color4::new(
        T::from_f64(c.r * mx_d),
        T::from_f64(c.g * mx_d),
        T::from_f64(c.b * mx_d),
        T::from_f64(c.a * mx_d),
    )
}

// ============================================================================
// Templated rgb2hsv (Vec3<T>)
// ============================================================================

/// Convert RGB → HSV for `Vec3<T>` (float branch).
#[inline]
pub fn rgb2hsv<T>(rgb: Vec3<T>) -> Vec3<T>
where
    T: Copy + ColorFromF64,
{
    let v = Vec3::new(rgb.x.to_f64(), rgb.y.to_f64(), rgb.z.to_f64());
    let c = rgb2hsv_d(v);
    Vec3::new(T::from_f64(c.x), T::from_f64(c.y), T::from_f64(c.z))
}

/// Convert RGB → HSV for `Vec3<T>` where `T` is an integer type.
#[inline]
pub fn rgb2hsv_int<T>(rgb: Vec3<T>) -> Vec3<T>
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx = T::int_max_as_f64();
    let v = Vec3::new(
        rgb.x.to_f64() / mx,
        rgb.y.to_f64() / mx,
        rgb.z.to_f64() / mx,
    );
    let c = rgb2hsv_d(v);
    Vec3::new(
        T::from_f64(c.x * mx),
        T::from_f64(c.y * mx),
        T::from_f64(c.z * mx),
    )
}

// ============================================================================
// Templated rgb2hsv (Color4<T>)
// ============================================================================

/// Convert RGB → HSV for `Color4<T>` (float branch).
#[inline]
pub fn rgb2hsv_c4<T>(rgb: Color4<T>) -> Color4<T>
where
    T: Copy + ColorFromF64,
{
    let v = Color4::new(rgb.r.to_f64(), rgb.g.to_f64(), rgb.b.to_f64(), rgb.a.to_f64());
    let c = rgb2hsv_d_c4(v);
    Color4::new(T::from_f64(c.r), T::from_f64(c.g), T::from_f64(c.b), T::from_f64(c.a))
}

/// Convert RGB → HSV for `Color4<T>` where `T` is an integer type.
#[inline]
pub fn rgb2hsv_c4_int<T>(rgb: Color4<T>) -> Color4<T>
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx_f = T::int_max_as_f32();
    let v = Color4::new(
        rgb.r.to_f64() / mx_f as f64,
        rgb.g.to_f64() / mx_f as f64,
        rgb.b.to_f64() / mx_f as f64,
        rgb.a.to_f64() / mx_f as f64,
    );
    let c = rgb2hsv_d_c4(v);
    Color4::new(
        T::from_f64(c.r * mx_f as f64),
        T::from_f64(c.g * mx_f as f64),
        T::from_f64(c.b * mx_f as f64),
        T::from_f64(c.a * mx_f as f64),
    )
}

// ============================================================================
// rgb2packed (Vec3<T>) — port of the template in ImathColorAlgo.h
// ============================================================================

/// Convert `Vec3<T>` RGB → `PackedColor` (float/half branch).
///
/// C++ float branch:
/// ```cpp
/// (PackedColor)(c.x * 255) | (((PackedColor)(c.y * 255)) << 8)
/// | (((PackedColor)(c.z * 255)) << 16) | 0xFF000000
/// ```
/// `c.x * 255` is computed in T (C++: `c.x` is T, `255` is int → promotes to T).
/// Alpha is set to 0xFF (fully opaque) in the Vec3 overload.
#[inline]
pub fn rgb2packed<T>(c: Vec3<T>) -> PackedColor
where
    T: Copy + ColorFromF64,
{
    c.x.times_255_to_packed()
        | (c.y.times_255_to_packed() << 8)
        | (c.z.times_255_to_packed() << 16)
        | 0xFF00_0000
}

/// Convert `Vec3<T>` RGB → `PackedColor` where `T` is an integer type.
///
/// C++ integer branch: normalise by `float(numeric_limits<T>::max())`, then call
/// the float overload `rgb2packed(V3f(...))`.
#[inline]
pub fn rgb2packed_int<T>(c: Vec3<T>) -> PackedColor
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx = T::int_max_as_f32();
    let v = Vec3::<f32>::new(
        c.x.to_f64() as f32 / mx,
        c.y.to_f64() as f32 / mx,
        c.z.to_f64() as f32 / mx,
    );
    rgb2packed(v)
}

// ============================================================================
// rgb2packed (Color4<T>)
// ============================================================================

/// Convert `Color4<T>` RGBA → `PackedColor` (float/half branch).
///
/// C++ float branch:
/// ```cpp
/// (PackedColor)(c.r * 255) | ... | (((PackedColor)(c.a * 255)) << 24)
/// ```
/// Each `c.r * 255` etc. is computed in T precision, matching C++.
#[inline]
pub fn rgb2packed_c4<T>(c: Color4<T>) -> PackedColor
where
    T: Copy + ColorFromF64,
{
    c.r.times_255_to_packed()
        | (c.g.times_255_to_packed() << 8)
        | (c.b.times_255_to_packed() << 16)
        | (c.a.times_255_to_packed() << 24)
}

/// Convert `Color4<T>` RGBA → `PackedColor` where `T` is an integer type.
#[inline]
pub fn rgb2packed_c4_int<T>(c: Color4<T>) -> PackedColor
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    let mx = T::int_max_as_f32();
    let v = Color4::<f32>::new(
        c.r.to_f64() as f32 / mx,
        c.g.to_f64() as f32 / mx,
        c.b.to_f64() as f32 / mx,
        c.a.to_f64() as f32 / mx,
    );
    rgb2packed_c4(v)
}

// ============================================================================
// packed2rgb (Vec3<T> out-param)
// ============================================================================

/// Unpack `PackedColor` into `Vec3<T>` RGB (float/half branch).
///
/// C++ float branch: `T f = T(1) / T(255); out.x = (packed & 0xFF) * f; …`
/// Division `T(1)/T(255)` and multiply are performed in T, matching C++ exactly.
#[inline]
pub fn packed2rgb<T>(packed: PackedColor, out: &mut Vec3<T>)
where
    T: Copy + ColorFromF64,
{
    out.x = T::byte_to_channel(packed & 0xFF);
    out.y = T::byte_to_channel((packed & 0xFF00) >> 8);
    out.z = T::byte_to_channel((packed & 0x00FF_0000) >> 16);
}

/// Unpack `PackedColor` into `Vec3<T>` RGB where `T` is an integer type.
///
/// C++ integer branch (`ImathColorAlgo.h:219-224`):
/// `T f = numeric_limits<T>::max() / (PackedColor)0xFF; out.x = (packed & 0xFF) * f; …`
///
/// `f` is computed in `T`'s own (signed, up to 64-bit) width — see
/// [`IntColorMax::byte_times_max_over_255`]. Computing it in `u32`, as this
/// port used to, made `T = i64` short by a factor of ~2^31.
#[inline]
pub fn packed2rgb_int<T>(packed: PackedColor, out: &mut Vec3<T>)
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    out.x = T::byte_times_max_over_255(packed & 0xFF);
    out.y = T::byte_times_max_over_255((packed & 0xFF00) >> 8);
    out.z = T::byte_times_max_over_255((packed & 0x00FF_0000) >> 16);
}

// ============================================================================
// packed2rgb (Color4<T> out-param)
// ============================================================================

/// Unpack `PackedColor` into `Color4<T>` RGBA (float/half branch).
///
/// C++ float branch: `T f = T(1) / T(255); out.r = (packed & 0xFF) * f; …`
/// Division `T(1)/T(255)` and multiply are performed in T, matching C++ exactly.
#[inline]
pub fn packed2rgb_c4<T>(packed: PackedColor, out: &mut Color4<T>)
where
    T: Copy + ColorFromF64,
{
    out.r = T::byte_to_channel(packed & 0xFF);
    out.g = T::byte_to_channel((packed & 0xFF00) >> 8);
    out.b = T::byte_to_channel((packed & 0x00FF_0000) >> 16);
    out.a = T::byte_to_channel((packed & 0xFF00_0000) >> 24);
}

/// Unpack `PackedColor` into `Color4<T>` RGBA where `T` is an integer type.
///
/// C++ integer branch: `ImathColorAlgo.h:243-249`, same `f` as [`packed2rgb_int`].
#[inline]
pub fn packed2rgb_c4_int<T>(packed: PackedColor, out: &mut Color4<T>)
where
    T: Copy + IsInteger + IntColorMax + ColorFromF64,
{
    out.r = T::byte_times_max_over_255(packed & 0xFF);
    out.g = T::byte_times_max_over_255((packed & 0xFF00) >> 8);
    out.b = T::byte_times_max_over_255((packed & 0x00FF_0000) >> 16);
    out.a = T::byte_times_max_over_255((packed & 0xFF00_0000) >> 24);
}

// ============================================================================
// Helper traits: ColorFromF64, IntColorMax
// ============================================================================

/// Convert an `f64` to `T`, mirroring the C++ `(T)(value)` cast used in the
/// template bodies (`Vec3<T>((T)c.x, ...)`, `(T)(c.x * max)`, etc.).
pub trait ColorFromF64: Sized {
    fn from_f64(v: f64) -> Self;
    /// Convert self to f64 for normalisation (used by integer branches).
    fn to_f64(self) -> f64;

    /// C++ `(PackedColor)(c.x * 255)` — multiply by 255 in T, truncate to u32.
    /// Used by `rgb2packed` / `rgb2packed_c4` float branches to match C++ T-precision.
    fn times_255_to_packed(self) -> PackedColor;

    /// C++ `T f = T(1) / T(255); (packed_byte) * f` — decode one 0..=255 byte
    /// channel to T, with the division performed in T precision.
    /// Used by `packed2rgb` / `packed2rgb_c4` float branches.
    fn byte_to_channel(byte: PackedColor) -> Self;
}

impl ColorFromF64 for f32 {
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    /// `(self * 255f32) as u32` — multiplication in f32, matching C++.
    #[inline] fn times_255_to_packed(self) -> PackedColor { (self * 255_f32) as PackedColor }
    /// `(byte as f32) * (1f32 / 255f32)` — divide and multiply in f32, matching C++.
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self {
        let f: f32 = 1.0_f32 / 255.0_f32;
        (byte as f32) * f
    }
}
impl ColorFromF64 for f64 {
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    /// `(self * 255f64) as u32` — multiplication in f64.
    #[inline] fn times_255_to_packed(self) -> PackedColor { (self * 255_f64) as PackedColor }
    /// `(byte as f64) * (1f64 / 255f64)` — divide and multiply in f64.
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self {
        let f: f64 = 1.0_f64 / 255.0_f64;
        (byte as f64) * f
    }
}
impl ColorFromF64 for u8 {
    #[inline] fn from_f64(v: f64) -> Self { v as u8 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    // Integer types use the _int branch; these methods are unreachable for u8
    // in the float-branch functions, but the trait requires an impl.
    #[inline] fn times_255_to_packed(self) -> PackedColor { self as PackedColor }
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self { byte as u8 }
}
impl ColorFromF64 for i16 {
    #[inline] fn from_f64(v: f64) -> Self { v as i16 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn times_255_to_packed(self) -> PackedColor { self as PackedColor }
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self { byte as i16 }
}
impl ColorFromF64 for i32 {
    #[inline] fn from_f64(v: f64) -> Self { v as i32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn times_255_to_packed(self) -> PackedColor { self as PackedColor }
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self { byte as i32 }
}
impl ColorFromF64 for i64 {
    #[inline] fn from_f64(v: f64) -> Self { v as i64 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn times_255_to_packed(self) -> PackedColor { self as PackedColor }
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self { byte as i64 }
}
impl ColorFromF64 for crate::half::Half {
    #[inline] fn from_f64(v: f64) -> Self { crate::half::Half::new(v as f32) }
    #[inline] fn to_f64(self) -> f64 { self.to_f32() as f64 }
    /// Half arithmetic goes through f32 (Imath half implements * as f32). Matches
    /// C++ `(PackedColor)(c.x * 255)` where c.x is a Half: Half * int promotes to float.
    #[inline] fn times_255_to_packed(self) -> PackedColor { (self.to_f32() * 255_f32) as PackedColor }
    /// C++ `T f = T(1)/T(255)` with T=half: division in f32 (half arithmetic through f32).
    #[inline] fn byte_to_channel(byte: PackedColor) -> Self {
        let f: f32 = 1.0_f32 / 255.0_f32;
        crate::half::Half::new((byte as f32) * f)
    }
}

/// `numeric_limits<T>::max()` as the various numeric types needed by the integer
/// branches (`f64`, `f32`, `u32`).
///
/// The C++ integer branch uses `double(numeric_limits<T>::max())` for Vec3 and
/// `float(numeric_limits<T>::max())` for Color4; the packed branches use
/// `numeric_limits<T>::max() / (PackedColor)0xFF` (integer divide).
pub trait IntColorMax: Copy {
    fn int_max_as_f64() -> f64;
    fn int_max_as_f32() -> f32;

    /// C++ `T f = numeric_limits<T>::max() / (PackedColor) 0xFF;` followed by
    /// `byte * f` (`ImathColorAlgo.h:219-224` / `:243-249`).
    ///
    /// The usual arithmetic conversions promote the `unsigned int` divisor to
    /// `T`'s (wider or equal) integer type, so both the divide and the multiply
    /// happen in `T`. Every `T` we implement fits in `i64`, and the largest
    /// product — `255 * (i64::MAX / 255) == 9223372036854775680` — still fits,
    /// so `i64` is a faithful common evaluation type.
    fn byte_times_max_over_255(byte: PackedColor) -> Self;
}

/// Shared body of [`IntColorMax::byte_times_max_over_255`]: `(max / 255) * byte`
/// evaluated in `i64`, the widest `T` any implementation uses.
#[inline]
fn byte_times(max: i64, byte: PackedColor) -> i64 {
    (max / 0xFF) * i64::from(byte)
}

impl IntColorMax for u8 {
    #[inline] fn int_max_as_f64() -> f64 { u8::MAX as f64 }
    #[inline] fn int_max_as_f32() -> f32 { u8::MAX as f32 }
    #[inline] fn byte_times_max_over_255(byte: PackedColor) -> Self {
        byte_times(u8::MAX as i64, byte) as u8
    }
}
impl IntColorMax for i16 {
    #[inline] fn int_max_as_f64() -> f64 { i16::MAX as f64 }
    #[inline] fn int_max_as_f32() -> f32 { i16::MAX as f32 }
    #[inline] fn byte_times_max_over_255(byte: PackedColor) -> Self {
        byte_times(i16::MAX as i64, byte) as i16
    }
}
impl IntColorMax for i32 {
    #[inline] fn int_max_as_f64() -> f64 { i32::MAX as f64 }
    #[inline] fn int_max_as_f32() -> f32 { i32::MAX as f32 }
    #[inline] fn byte_times_max_over_255(byte: PackedColor) -> Self {
        byte_times(i32::MAX as i64, byte) as i32
    }
}
impl IntColorMax for i64 {
    #[inline] fn int_max_as_f64() -> f64 { i64::MAX as f64 }
    #[inline] fn int_max_as_f32() -> f32 { i64::MAX as f32 }
    #[inline] fn byte_times_max_over_255(byte: PackedColor) -> Self {
        byte_times(i64::MAX, byte)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::C4f;
    use crate::half::Half;
    use crate::vec::{V3f, Vec3};

    // --- hsv2rgb_d / rgb2hsv_d roundtrip ---
    #[test]
    fn hsv2rgb_rgb2hsv_roundtrip_f64() {
        // Known colour: H=0.0 (red), S=1.0, V=1.0 → RGB(1,0,0)
        let hsv = Vec3::new(0.0_f64, 1.0_f64, 1.0_f64);
        let rgb = hsv2rgb_d(hsv);
        assert!((rgb.x - 1.0).abs() < 1e-10, "R should be 1, got {}", rgb.x);
        assert!((rgb.y - 0.0).abs() < 1e-10, "G should be 0, got {}", rgb.y);
        assert!((rgb.z - 0.0).abs() < 1e-10, "B should be 0, got {}", rgb.z);

        // Roundtrip: RGB -> HSV -> RGB
        let rgb_in = Vec3::new(0.2_f64, 0.5_f64, 0.8_f64);
        let hsv_out = rgb2hsv_d(rgb_in);
        let rgb_out = hsv2rgb_d(hsv_out);
        assert!((rgb_in.x - rgb_out.x).abs() < 1e-10, "x roundtrip fail");
        assert!((rgb_in.y - rgb_out.y).abs() < 1e-10, "y roundtrip fail");
        assert!((rgb_in.z - rgb_out.z).abs() < 1e-10, "z roundtrip fail");
    }

    // --- hsv2rgb / rgb2hsv for float (Vec3<f32>) ---
    #[test]
    fn hsv2rgb_rgb2hsv_roundtrip_f32() {
        let rgb_in = V3f::new(0.3_f32, 0.6_f32, 0.9_f32);
        let hsv = rgb2hsv(rgb_in);
        let rgb_out = hsv2rgb(hsv);
        assert!((rgb_in.x - rgb_out.x).abs() < 1e-5, "x roundtrip fail");
        assert!((rgb_in.y - rgb_out.y).abs() < 1e-5, "y roundtrip fail");
        assert!((rgb_in.z - rgb_out.z).abs() < 1e-5, "z roundtrip fail");
    }

    // --- hsv2rgb / rgb2hsv Color4 roundtrip ---
    #[test]
    fn hsv2rgb_rgb2hsv_color4_roundtrip() {
        let rgb_in = C4f::new(0.2_f32, 0.4_f32, 0.6_f32, 1.0_f32);
        let hsv = rgb2hsv_c4(rgb_in);
        let rgb_out = hsv2rgb_c4(hsv);
        assert!((rgb_in.r - rgb_out.r).abs() < 1e-5);
        assert!((rgb_in.g - rgb_out.g).abs() < 1e-5);
        assert!((rgb_in.b - rgb_out.b).abs() < 1e-5);
        // alpha passes through
        assert!((rgb_in.a - rgb_out.a).abs() < 1e-5);
    }

    // --- rgb2packed / packed2rgb Vec3<f32> roundtrip ---
    #[test]
    fn rgb2packed_packed2rgb_roundtrip_f32() {
        let c = V3f::new(1.0_f32, 0.5_f32, 0.0_f32);
        let packed = rgb2packed(c);
        let mut out = V3f::default();
        packed2rgb(packed, &mut out);
        // lossy (quantised to 8-bit); allow 1/255 tolerance
        assert!((c.x - out.x).abs() < 2.0 / 255.0, "x fail: {} vs {}", c.x, out.x);
        assert!((c.y - out.y).abs() < 2.0 / 255.0, "y fail: {} vs {}", c.y, out.y);
        assert!((c.z - out.z).abs() < 2.0 / 255.0, "z fail: {} vs {}", c.z, out.z);
        // packed alpha should be 0xFF (opaque) for Vec3 overload
        assert_eq!((packed >> 24) & 0xFF, 0xFF, "alpha should be 0xFF");
    }

    // --- rgb2packed / packed2rgb Color4<f32> roundtrip ---
    #[test]
    fn rgb2packed_packed2rgb_color4_roundtrip() {
        let c = C4f::new(1.0_f32, 0.5_f32, 0.25_f32, 0.75_f32);
        let packed = rgb2packed_c4(c);
        let mut out = C4f::new(0.0, 0.0, 0.0, 0.0);
        packed2rgb_c4(packed, &mut out);
        assert!((c.r - out.r).abs() < 2.0 / 255.0, "r fail");
        assert!((c.g - out.g).abs() < 2.0 / 255.0, "g fail");
        assert!((c.b - out.b).abs() < 2.0 / 255.0, "b fail");
        assert!((c.a - out.a).abs() < 2.0 / 255.0, "a fail");
    }

    // --- C3c (u8) hsv roundtrip via int path ---
    #[test]
    fn hsv_rgb_c3c_int_roundtrip() {
        // u8: 0..=255 maps to 0..=1 in double space
        let rgb_in = Vec3::<u8>::new(200, 100, 50);
        let hsv = rgb2hsv_int(rgb_in);
        let rgb_out = hsv2rgb_int(hsv);
        // 8-bit quantisation: allow ±2 per channel
        assert!((rgb_in.x as i32 - rgb_out.x as i32).abs() <= 2, "x fail: {} vs {}", rgb_in.x, rgb_out.x);
        assert!((rgb_in.y as i32 - rgb_out.y as i32).abs() <= 2, "y fail: {} vs {}", rgb_in.y, rgb_out.y);
        assert!((rgb_in.z as i32 - rgb_out.z as i32).abs() <= 2, "z fail: {} vs {}", rgb_in.z, rgb_out.z);
    }

    // --- C3h (Half) smoke test ---
    #[test]
    fn hsv_rgb_c3h_half_roundtrip() {
        let rgb_in = Vec3::new(Half::new(0.3), Half::new(0.6), Half::new(0.9));
        let hsv = rgb2hsv(rgb_in);
        let rgb_out = hsv2rgb(hsv);
        // Half has ~3 decimal digits; allow 0.002 tolerance
        assert!((rgb_in.x.to_f32() - rgb_out.x.to_f32()).abs() < 0.005, "x fail");
        assert!((rgb_in.y.to_f32() - rgb_out.y.to_f32()).abs() < 0.005, "y fail");
        assert!((rgb_in.z.to_f32() - rgb_out.z.to_f32()).abs() < 0.005, "z fail");
    }

    // --- rgb2packed known values ---
    #[test]
    fn rgb2packed_known_values() {
        // Pure red (1,0,0) → packed = 0xFF_00_00_FF (R=255, G=0, B=0, A=255)
        let red = V3f::new(1.0_f32, 0.0, 0.0);
        let p = rgb2packed(red);
        assert_eq!(p & 0xFF, 255, "R byte should be 255");
        assert_eq!((p >> 8) & 0xFF, 0, "G byte should be 0");
        assert_eq!((p >> 16) & 0xFF, 0, "B byte should be 0");
        assert_eq!((p >> 24) & 0xFF, 0xFF, "A byte should be 0xFF");
    }

    // --- packed2rgb known values ---
    #[test]
    fn packed2rgb_known_values() {
        // packed = 0x00_FF_00_80: R=0x80=128, G=0xFF=255, B=0x00=0, A=0x00
        let packed: PackedColor = 0x0000_FF80;
        let mut out = V3f::default();
        packed2rgb(packed, &mut out);
        assert!((out.x - 128.0 / 255.0).abs() < 0.005, "x={}", out.x);
        assert!((out.y - 255.0 / 255.0).abs() < 0.005, "y={}", out.y);
        assert!((out.z - 0.0).abs() < 0.005, "z={}", out.z);
    }

    // --- hsv2rgb_d sector coverage: all 6 switch cases ---
    #[test]
    fn hsv2rgb_d_six_sectors() {
        // H=0/6, S=1, V=1 → red (case 0): R=1,G=0~t,B=0
        let c0 = hsv2rgb_d(Vec3::new(0.0_f64, 1.0, 1.0));
        assert!((c0.x - 1.0).abs() < 1e-9);
        // H=1/6 → case 1 boundary — just check no panic and valid range
        let c1 = hsv2rgb_d(Vec3::new(1.0_f64 / 6.0, 1.0, 1.0));
        assert!(c1.x >= 0.0 && c1.x <= 1.0);
        // H=2/6 → case 2
        let c2 = hsv2rgb_d(Vec3::new(2.0_f64 / 6.0, 1.0, 1.0));
        assert!(c2.y >= 0.0 && c2.y <= 1.0);
        // H=3/6 → case 3
        let c3 = hsv2rgb_d(Vec3::new(3.0_f64 / 6.0, 1.0, 1.0));
        assert!(c3.z >= 0.0 && c3.z <= 1.0);
        // H=4/6 → case 4
        let c4 = hsv2rgb_d(Vec3::new(4.0_f64 / 6.0, 1.0, 1.0));
        assert!(c4.x >= 0.0 && c4.x <= 1.0);
        // H=5/6 → case 5
        let c5 = hsv2rgb_d(Vec3::new(5.0_f64 / 6.0, 1.0, 1.0));
        assert!(c5.x >= 0.0 && c5.x <= 1.0);
        // hue==1.0 maps to hue=0 (case 0)
        let c_wrap = hsv2rgb_d(Vec3::new(1.0_f64, 1.0, 1.0));
        assert!((c_wrap.x - c0.x).abs() < 1e-9);
    }

    // --- rgb2hsv_d known value: pure blue ---
    #[test]
    fn rgb2hsv_d_pure_blue() {
        // RGB(0,0,1) → HSV(240°/360° = 2/3, 1, 1)
        let hsv = rgb2hsv_d(Vec3::new(0.0_f64, 0.0, 1.0));
        assert!((hsv.x - 2.0 / 3.0).abs() < 1e-10, "H={}", hsv.x);
        assert!((hsv.y - 1.0).abs() < 1e-10, "S={}", hsv.y);
        assert!((hsv.z - 1.0).abs() < 1e-10, "V={}", hsv.z);
    }

    // --- u8 packed2rgb integer branch ---
    #[test]
    fn packed2rgb_int_u8() {
        // pack a u8 colour and unpack it
        let c_in = Vec3::<u8>::new(128, 64, 32);
        let packed = rgb2packed_int(c_in);
        let mut out = Vec3::<u8>::new(0, 0, 0);
        packed2rgb_int(packed, &mut out);
        // tolerance ±1 per channel due to integer quantisation
        assert!((c_in.x as i32 - out.x as i32).abs() <= 1, "x: {} vs {}", c_in.x, out.x);
        assert!((c_in.y as i32 - out.y as i32).abs() <= 1, "y: {} vs {}", c_in.y, out.y);
        assert!((c_in.z as i32 - out.z as i32).abs() <= 1, "z: {} vs {}", c_in.z, out.z);
    }

    // --- packed2rgb integer branch: the C++ divisor is computed in T ---

    /// C++ `T f = numeric_limits<T>::max() / (PackedColor) 0xFF` with
    /// `T = int64_t` gives 36170086419038336, so a full byte maps to
    /// 9223372036854775680 — the value the real Imath oracle prints
    /// (`color.packed2rgb.i64.ffffffff.x` in tests/parity/golden.txt).
    /// Computing `f` in `u32` (what this port used to do) caps it at 16843009,
    /// i.e. short by a factor of ~2^31.
    #[test]
    fn packed2rgb_i64_uses_the_full_i64_range() {
        // Independent expectation, spelled out the way ImathColorAlgo.h does.
        let f: i64 = i64::MAX / 0xFF;
        assert_eq!(f, 36_170_086_419_038_336);

        // x <- packed & 0xFF, y <- byte 1, z <- byte 2.
        let mut v = Vec3::new(0i64, 0, 0);
        packed2rgb_int(0x00FF_7FFFu32, &mut v);
        assert_eq!(v.x, 0xFF * f);
        assert_eq!(v.x, 9_223_372_036_854_775_680);
        assert_eq!(v.y, 0x7F * f);
        assert_eq!(v.z, 0xFF * f);

        let mut c = crate::color::Color4::new(0i64, 0, 0, 0);
        packed2rgb_c4_int(0x8000_0001u32, &mut c);
        assert_eq!(c.r, f);
        assert_eq!(c.a, 0x80 * f);
    }

    /// The narrower integer types keep the values C++ produces (and, unlike the
    /// i64 case, the old u32 arithmetic already agreed — this pins that down).
    #[test]
    fn packed2rgb_narrow_ints_match_the_cpp_divisor() {
        let mut v8 = Vec3::new(0u8, 0, 0);
        packed2rgb_int(0x0000_00FFu32, &mut v8);
        assert_eq!(v8.x, 255); // 255/255 == 1

        let mut v16 = Vec3::new(0i16, 0, 0);
        packed2rgb_int(0x0000_00FFu32, &mut v16);
        assert_eq!(v16.x, 0xFF * (i16::MAX / 0xFF)); // 255 * 128 == 32640

        let mut v32 = Vec3::new(0i32, 0, 0);
        packed2rgb_int(0x0000_00FFu32, &mut v32);
        assert_eq!(v32.x, 0xFF * (i32::MAX / 0xFF)); // 2147483520
    }
}
