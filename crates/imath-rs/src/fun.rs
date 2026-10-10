//! Port of Imath ImathFun.h + ImathFun.cpp.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathFun.h>
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathFun.cpp>
//!
//! Miscellaneous utility functions. Every function from both files is ported 1:1:
//! `abs`, `sign`, `lerp`, `ulerp`, `lerpfactor`, `clamp`, `cmp`, `cmpt`, `iszero`,
//! `equal`, the integer-returning `floor`/`ceil`/`trunc`, the signed/positive
//! integer division+remainder pairs `divs`/`mods`/`divp`/`modp`, the float
//! successor/predecessor pair `succf`/`predf`/`succd`/`predd` (bodies from the
//! `.cpp`), and the bit-trick finiteness tests `finitef`/`finited`.

use crate::traits::Scalar;

/// Port of `abs(T a)`: `(a > T(0)) ? a : -a`.
///
/// Branch-based absolute value, valid for any ordered, negatable [`Scalar`]
/// (floats and signed ints alike), matching the C++ template exactly.
#[inline]
pub fn abs<T: Scalar>(a: T) -> T {
    if a > T::ZERO {
        a
    } else {
        -a
    }
}

/// Port of `sign(T a)`: `+1`, `-1`, or `0` for positive, negative, or zero `a`.
#[inline]
pub fn sign<T: Scalar>(a: T) -> i32 {
    if a > T::ZERO {
        1
    } else if a < T::ZERO {
        -1
    } else {
        0
    }
}

/// Port of `lerp(T a, T b, Q t)`: `a * (1 - t) + b * t`.
///
/// Linear interpolation. The interpolation parameter type `Q` is independent of
/// the value type `T` (as in the C++ template); the result is of type `T`.
#[inline]
pub fn lerp<T, Q>(a: T, b: T, t: Q) -> T
where
    Q: Scalar,
    T: core::ops::Mul<Q, Output = T> + core::ops::Add<Output = T>,
{
    a * (Q::ONE - t) + b * t
}

/// Port of `ulerp(T a, T b, Q t)`:
/// `(a > b) ? (a - (a - b) * t) : (a + (b - a) * t)`.
///
/// "Unsigned"/monotonic lerp that subtracts the larger from the smaller to reduce
/// rounding error; result is of type `T`.
#[inline]
pub fn ulerp<T, Q>(a: T, b: T, t: Q) -> T
where
    Q: Scalar,
    T: Copy
        + PartialOrd
        + core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>
        + core::ops::Mul<Q, Output = T>,
{
    if a > b {
        a - (a - b) * t
    } else {
        a + (b - a) * t
    }
}

/// Port of `lerpfactor(T m, T a, T b)`.
///
/// Returns `t` such that `m == lerp(a, b, t)`, i.e. how far `m` lies between `a`
/// and `b`. Returns `0` when `a == b`. The guard reproduces the C++ condition
/// `abs(d) > T(1) || abs(n) < max * abs(d)`, which avoids overflow when dividing.
#[inline]
pub fn lerpfactor<T>(m: T, a: T, b: T) -> T
where
    T: Scalar + Bounded,
{
    let d = b - a;
    let n = m - a;

    if abs(d) > T::ONE || abs(n) < T::MAX_VALUE * abs(d) {
        n / d
    } else {
        T::ZERO
    }
}

/// Supplies `std::numeric_limits<T>::max()` for the float types used by
/// [`lerpfactor`]. Public only because it appears in that bound.
pub trait Bounded {
    /// `std::numeric_limits<T>::max()`.
    const MAX_VALUE: Self;
}

impl Bounded for f32 {
    const MAX_VALUE: Self = f32::MAX;
}

impl Bounded for f64 {
    const MAX_VALUE: Self = f64::MAX;
}

/// Port of `clamp(T a, T l, T h)`: `(a < l) ? l : ((a > h) ? h : a)`.
#[inline]
pub fn clamp<T: Scalar>(a: T, l: T, h: T) -> T {
    if a < l {
        l
    } else if a > h {
        h
    } else {
        a
    }
}

/// Port of `cmp(T a, T b)`: `sign(a - b)` — three-way comparison (`-1`/`0`/`+1`).
#[inline]
pub fn cmp<T: Scalar>(a: T, b: T) -> i32 {
    sign(a - b)
}

/// Port of `cmpt(T a, T b, T t)`: like [`cmp`] but treats values within tolerance
/// `t` as equal — `(abs(a - b) <= t) ? 0 : cmp(a, b)`.
#[inline]
pub fn cmpt<T: Scalar>(a: T, b: T, t: T) -> i32 {
    if abs(a - b) <= t {
        0
    } else {
        cmp(a, b)
    }
}

/// Port of `iszero(T a, T t)`: `abs(a) <= t`.
///
/// C++ returns `int` (1/0); Rust returns the equivalent `bool`.
#[inline]
pub fn iszero<T: Scalar>(a: T, t: T) -> bool {
    abs(a) <= t
}

/// Port of `equal(T1 a, T2 b, T3 t)`: `abs(a - b) <= t`.
///
/// The C++ template takes three independent types `T1,T2,T3`. In Imath's actual
/// usage these collapse to one scalar type; we port the common monomorphic form
/// over a single [`Scalar`] `T`, which covers every real call site while keeping
/// the semantics (`abs(a - b) <= t`) identical.
#[inline]
pub fn equal<T: Scalar>(a: T, b: T, t: T) -> bool {
    abs(a - b) <= t
}

/// Port of the integer-returning `floor(T x)`:
/// `(x >= 0) ? int(x) : -(int(-x) + (-x > int(-x)))`.
///
/// Returns the largest integer `<= x` as an `i32`, matching the exact branch/cast
/// behavior of the C++ template (truncation toward zero plus a correction for the
/// negative, non-integral case).
#[inline]
pub fn floor<T: FloorCeil>(x: T) -> i32 {
    x.imath_floor()
}

/// Port of the integer-returning `ceil(T x)`: `-floor(-x)`.
#[inline]
pub fn ceil<T: FloorCeil>(x: T) -> i32 {
    -floor(x.neg_for_ceil())
}

/// Port of the integer-returning `trunc(T x)`: `(x >= 0) ? int(x) : -int(-x)`.
#[inline]
pub fn trunc<T: FloorCeil>(x: T) -> i32 {
    x.imath_trunc()
}

/// Backs the integer-returning [`floor`]/[`ceil`]/[`trunc`] for the float types.
///
/// These reproduce the precise C++ cast logic (`int(x)` truncates toward zero,
/// with the `floor` correction term `(-x > int(-x))`). Public only because it
/// appears in the function bounds above.
pub trait FloorCeil: Copy {
    /// 1:1 with the C++ integer `floor(T x)`.
    fn imath_floor(self) -> i32;
    /// 1:1 with the C++ integer `trunc(T x)`.
    fn imath_trunc(self) -> i32;
    /// Returns `-self`, used to express `ceil(x) = -floor(-x)`.
    fn neg_for_ceil(self) -> Self;
}

macro_rules! impl_floorceil {
    ($t:ty) => {
        impl FloorCeil for $t {
            #[inline]
            fn imath_floor(self) -> i32 {
                // (x >= 0) ? int(x) : -(int(-x) + (-x > int(-x)))
                if self >= 0 as $t {
                    self as i32
                } else {
                    let nx = -self;
                    let inx = nx as i32; // int(-x), truncated toward zero
                    -(inx + (nx > inx as $t) as i32)
                }
            }
            #[inline]
            fn imath_trunc(self) -> i32 {
                // (x >= 0) ? int(x) : -int(-x)
                if self >= 0 as $t {
                    self as i32
                } else {
                    -((-self) as i32)
                }
            }
            #[inline]
            fn neg_for_ceil(self) -> Self {
                -self
            }
        }
    };
}

impl_floorceil!(f32);
impl_floorceil!(f64);

//
// Integer division and remainder where the remainder of x/y has the same sign as x.
//

/// Port of `divs(int x, int y)`: signed division whose remainder takes the sign of
/// `x` — `divs(x,y) == (abs(x)/abs(y)) * (sign(x)*sign(y))`.
#[inline]
pub fn divs(x: i32, y: i32) -> i32 {
    if x >= 0 {
        if y >= 0 {
            x / y
        } else {
            -(x / -y)
        }
    } else if y >= 0 {
        -(-x / y)
    } else {
        -x / -y
    }
}

/// Port of `mods(int x, int y)`: remainder matching [`divs`] (same sign as `x`).
#[inline]
pub fn mods(x: i32, y: i32) -> i32 {
    if x >= 0 {
        if y >= 0 {
            x % y
        } else {
            x % -y
        }
    } else if y >= 0 {
        -(-x % y)
    } else {
        -(-x % -y)
    }
}

//
// Integer division and remainder where the remainder of x/y is always positive.
//

/// Port of `divp(int x, int y)`: floored division —
/// `divp(x,y) == floor(double(x)/double(y))`, so the remainder is always positive.
#[inline]
pub fn divp(x: i32, y: i32) -> i32 {
    if x >= 0 {
        if y >= 0 {
            x / y
        } else {
            -(x / -y)
        }
    } else if y >= 0 {
        -((y - 1 - x) / y)
    } else {
        (-y - 1 - x) / -y
    }
}

/// Port of `modp(int x, int y)`: `x - y * divp(x, y)` — always-positive remainder.
#[inline]
pub fn modp(x: i32, y: i32) -> i32 {
    x - y * divp(x, y)
}

//
// Successor / predecessor for floating-point numbers (bodies from ImathFun.cpp).
//

/// Port of `succf(float f)` (ImathFun.cpp): the next representable `f32` toward
/// `+inf`, or `f` unchanged if `f` is NaN/infinite.
#[inline]
pub fn succf(f: f32) -> f32 {
    if f.is_finite() {
        next_after_f32(f, f32::INFINITY)
    } else {
        f
    }
}

/// Port of `predf(float f)` (ImathFun.cpp): the next representable `f32` toward
/// `-inf`, or `f` unchanged if `f` is NaN/infinite.
#[inline]
pub fn predf(f: f32) -> f32 {
    if f.is_finite() {
        next_after_f32(f, f32::NEG_INFINITY)
    } else {
        f
    }
}

/// Port of `succd(double d)` (ImathFun.cpp): the next representable `f64` toward
/// `+inf`, or `d` unchanged if `d` is NaN/infinite.
#[inline]
pub fn succd(d: f64) -> f64 {
    if d.is_finite() {
        next_after_f64(d, f64::INFINITY)
    } else {
        d
    }
}

/// Port of `predd(double d)` (ImathFun.cpp): the next representable `f64` toward
/// `-inf`, or `d` unchanged if `d` is NaN/infinite.
#[inline]
pub fn predd(d: f64) -> f64 {
    if d.is_finite() {
        next_after_f64(d, f64::NEG_INFINITY)
    } else {
        d
    }
}

/// `std::nextafter` for `f32`. Rust's std lacks a stable `next_after`, so we
/// reproduce the IEEE-754 bit-stepping that the C library performs: walk the
/// integer encoding of `from` one ULP toward `to`.
#[inline]
fn next_after_f32(from: f32, to: f32) -> f32 {
    if from == to {
        return to;
    }
    if from == 0.0 {
        // Smallest subnormal with the sign of `to`.
        return f32::from_bits(1).copysign(to);
    }
    let bits = from.to_bits();
    // Moving away from zero increases the magnitude bits; toward zero decreases.
    let next = if (to > from) == (from > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

/// `std::nextafter` for `f64`. See [`next_after_f32`] for the rationale.
#[inline]
fn next_after_f64(from: f64, to: f64) -> f64 {
    if from == to {
        return to;
    }
    if from == 0.0 {
        return f64::from_bits(1).copysign(to);
    }
    let bits = from.to_bits();
    let next = if (to > from) == (from > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

/// Port of `finitef(float f)`: true if `f` is neither NaN nor infinite, via the
/// exponent bit-pattern test `(bits & 0x7f800000) != 0x7f800000`.
#[inline]
pub fn finitef(f: f32) -> bool {
    (f.to_bits() & 0x7f80_0000) != 0x7f80_0000
}

/// Port of `finited(double d)`: true if `d` is neither NaN nor infinite, via the
/// exponent bit-pattern test `(bits & 0x7ff0000000000000) != 0x7ff0000000000000`.
#[inline]
pub fn finited(d: f64) -> bool {
    (d.to_bits() & 0x7ff0_0000_0000_0000) != 0x7ff0_0000_0000_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_sign() {
        assert_eq!(abs(-5.0_f64), 5.0);
        assert_eq!(abs(5.0_f64), 5.0);
        assert_eq!(abs(-7_i32), 7);
        assert_eq!(sign(-3.0_f64), -1);
        assert_eq!(sign(3.0_f64), 1);
        assert_eq!(sign(0.0_f64), 0);
    }

    #[test]
    fn lerp_ulerp() {
        assert!((lerp(0.0_f64, 10.0, 0.25) - 2.5).abs() < 1e-12);
        assert!((ulerp(10.0_f64, 0.0, 0.25) - 7.5).abs() < 1e-12);
        assert!((ulerp(0.0_f64, 10.0, 0.25) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn lerpfactor_roundtrip() {
        let a = 2.0_f64;
        let b = 8.0_f64;
        let m = 5.0_f64;
        let t = lerpfactor(m, a, b);
        assert!((t - 0.5).abs() < 1e-12);
        // a == b -> 0
        assert_eq!(lerpfactor(5.0_f64, 3.0, 3.0), 0.0);
    }

    #[test]
    fn clamp_cmp() {
        assert_eq!(clamp(5, 0, 3), 3);
        assert_eq!(clamp(-1, 0, 3), 0);
        assert_eq!(clamp(2, 0, 3), 2);
        assert_eq!(cmp(3.0_f64, 5.0), -1);
        assert_eq!(cmp(5.0_f64, 5.0), 0);
        assert_eq!(cmpt(5.0_f64, 5.05, 0.1), 0);
        assert!(iszero(0.05_f64, 0.1));
        assert!(equal(1.0_f64, 1.05, 0.1));
    }

    #[test]
    fn int_floor_ceil_trunc() {
        assert_eq!(floor(2.7_f64), 2);
        assert_eq!(floor(-2.3_f64), -3);
        assert_eq!(floor(-2.0_f64), -2);
        assert_eq!(ceil(2.1_f64), 3);
        assert_eq!(ceil(-2.7_f64), -2);
        assert_eq!(trunc(2.7_f64), 2);
        assert_eq!(trunc(-2.7_f64), -2);
    }

    #[test]
    fn signed_div_mod() {
        // divs/mods: remainder has sign of x.
        assert_eq!(divs(-7, 2), -3);
        assert_eq!(mods(-7, 2), -1);
        assert_eq!(divs(7, -2), -3);
        assert_eq!(mods(7, -2), 1);
        // Verify the divs/mods invariant: x == y*divs + mods.
        for &(x, y) in &[(-7, 2), (7, -2), (-7, -2), (7, 2)] {
            assert_eq!(x, y * divs(x, y) + mods(x, y));
        }
    }

    #[test]
    fn positive_div_mod() {
        // divp/modp: remainder always positive.
        assert_eq!(divp(-7, 2), -4);
        assert_eq!(modp(-7, 2), 1);
        assert_eq!(divp(7, 2), 3);
        assert_eq!(modp(7, 2), 1);
        for &(x, y) in &[(-7, 2), (7, 2), (-1, 5), (13, 4)] {
            let m = modp(x, y);
            assert!(m >= 0);
            assert_eq!(x, y * divp(x, y) + m);
        }
    }

    #[test]
    fn succ_pred() {
        assert!(succf(1.0_f32) > 1.0);
        assert!(predf(1.0_f32) < 1.0);
        assert_eq!(succf(f32::INFINITY), f32::INFINITY);
        assert!(succf(f32::NAN).is_nan());
        assert!(succd(1.0_f64) > 1.0);
        assert!(predd(1.0_f64) < 1.0);
        // One ULP up then down returns to the original value.
        assert_eq!(predf(succf(2.0_f32)), 2.0);
        assert_eq!(predd(succd(2.0_f64)), 2.0);
    }

    #[test]
    fn finiteness() {
        assert!(finitef(1.0));
        assert!(!finitef(f32::INFINITY));
        assert!(!finitef(f32::NAN));
        assert!(finited(1.0));
        assert!(!finited(f64::INFINITY));
        assert!(!finited(f64::NAN));
    }
}
