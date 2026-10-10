//! Port of Imath ImathMath.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMath.h>
//!
//! `ImathMath.h` defines the (deprecated) `template <class T> struct Math` whose
//! static methods dispatch to the correct `std::` math function for `T` (float vs
//! double). It also defines three free helpers: `sinx_over_x`, `equalWithAbsError`,
//! and `equalWithRelError`.
//!
//! In Rust we model `Math<T>` as the [`Math`] trait, implemented for `f32` and
//! `f64`. The function set is *exactly* the one in the header — no more, no less —
//! each forwarding to the corresponding Rust `std` (`libm`-backed) method, which is
//! the analog of the C++ `std::` call. The free helpers become generic free
//! functions over [`Math`].

/// 1:1 port of Imath's `template <class T> struct Math`.
///
/// Each method mirrors the same-named static in `ImathMath.h` and forwards to the
/// matching Rust standard-library method (the equivalent of C++ `std::`). Imath
/// marks every method `IMATH_DEPRECATED("use std::math functions")`; that note is
/// preserved here as documentation only — Rust users should likewise prefer the
/// inherent `f32`/`f64` methods directly.
pub trait Math: Copy {
    /// `std::acos`.
    fn acos(self) -> Self;
    /// `std::asin`.
    fn asin(self) -> Self;
    /// `std::atan`.
    fn atan(self) -> Self;
    /// `std::atan2(x, y)` — note Imath's argument order is `atan2(x, y)`,
    /// forwarded verbatim as `x.atan2(y)`.
    fn atan2(self, y: Self) -> Self;
    /// `std::cos`.
    fn cos(self) -> Self;
    /// `std::sin`.
    fn sin(self) -> Self;
    /// `std::tan`.
    fn tan(self) -> Self;
    /// `std::cosh`.
    fn cosh(self) -> Self;
    /// `std::sinh`.
    fn sinh(self) -> Self;
    /// `std::tanh`.
    fn tanh(self) -> Self;
    /// `std::exp`.
    fn exp(self) -> Self;
    /// `std::log`.
    fn log(self) -> Self;
    /// `std::log10`.
    fn log10(self) -> Self;
    /// `std::modf` — splits `self` into integral and fractional parts. Returns the
    /// fractional part (Imath's return value) and writes the integral part to
    /// `iptr`, matching `Math<T>::modf(T x, T* iptr)`.
    fn modf(self, iptr: &mut Self) -> Self;
    /// `std::pow(x, y)`.
    fn pow(self, y: Self) -> Self;
    /// `std::sqrt`.
    fn sqrt(self) -> Self;
    /// `std::ceil`.
    fn ceil(self) -> Self;
    /// `std::fabs`.
    fn fabs(self) -> Self;
    /// `std::floor`.
    fn floor(self) -> Self;
    /// `std::fmod(x, y)`.
    fn fmod(self, y: Self) -> Self;
    /// `std::hypot(x, y)`.
    fn hypot(self, y: Self) -> Self;
}

/// Generates the `Math` impl for a floating-point type. Both `f32` and `f64`
/// expose the identical inherent-method set, so the bodies are identical.
macro_rules! impl_math {
    ($t:ty) => {
        impl Math for $t {
            #[inline]
            fn acos(self) -> Self {
                self.acos()
            }
            #[inline]
            fn asin(self) -> Self {
                self.asin()
            }
            #[inline]
            fn atan(self) -> Self {
                self.atan()
            }
            #[inline]
            fn atan2(self, y: Self) -> Self {
                self.atan2(y)
            }
            #[inline]
            fn cos(self) -> Self {
                self.cos()
            }
            #[inline]
            fn sin(self) -> Self {
                self.sin()
            }
            #[inline]
            fn tan(self) -> Self {
                self.tan()
            }
            #[inline]
            fn cosh(self) -> Self {
                self.cosh()
            }
            #[inline]
            fn sinh(self) -> Self {
                self.sinh()
            }
            #[inline]
            fn tanh(self) -> Self {
                self.tanh()
            }
            #[inline]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn log(self) -> Self {
                self.ln()
            }
            #[inline]
            fn log10(self) -> Self {
                self.log10()
            }
            #[inline]
            fn modf(self, iptr: &mut Self) -> Self {
                // C++ std::modf: integral part -> *iptr, fractional part returned.
                // The integral part truncates toward zero and carries the sign.
                let ival = self.trunc();
                *iptr = ival;
                self - ival
            }
            #[inline]
            fn pow(self, y: Self) -> Self {
                self.powf(y)
            }
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn ceil(self) -> Self {
                self.ceil()
            }
            #[inline]
            fn fabs(self) -> Self {
                self.abs()
            }
            #[inline]
            fn floor(self) -> Self {
                self.floor()
            }
            #[inline]
            fn fmod(self, y: Self) -> Self {
                // C++ std::fmod returns a result with the sign of x; Rust's `%`
                // (Rem) on floats has identical semantics (truncated remainder).
                self % y
            }
            #[inline]
            fn hypot(self, y: Self) -> Self {
                self.hypot(y)
            }
        }
    };
}

impl_math!(f32);
impl_math!(f64);

/// Port of Imath's `sinx_over_x` — Don Hatch's accurate `sin(x)/x`.
///
/// Returns `1` for `x == 0` (and for any `x` so small that `x*x` underflows below
/// the type's epsilon), avoiding the `0/0` and catastrophic-cancellation problems
/// of the naive form. 1:1 with the C++ template.
#[inline]
pub fn sinx_over_x<T>(x: T) -> T
where
    T: Math + Copy + PartialOrd + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + Epsilon,
{
    if x * x < T::EPSILON {
        T::ONE_VALUE
    } else {
        x.sin() / x
    }
}

/// Helper supplying the per-type constants `sinx_over_x` needs (`epsilon` and the
/// literal `T(1)`), mirroring `std::numeric_limits<T>::epsilon()` and `T(1)` from
/// the C++ body. Kept module-private in spirit (public only because it appears in
/// the `sinx_over_x` bound) and implemented only for the float types Imath uses.
pub trait Epsilon {
    /// `std::numeric_limits<T>::epsilon()`.
    const EPSILON: Self;
    /// The literal `T(1)`.
    const ONE_VALUE: Self;
}

impl Epsilon for f32 {
    const EPSILON: Self = f32::EPSILON;
    const ONE_VALUE: Self = 1.0;
}

impl Epsilon for f64 {
    const EPSILON: Self = f64::EPSILON;
    const ONE_VALUE: Self = 1.0;
}

/// Port of Imath's `equalWithAbsError`.
///
/// Returns `true` if `x1` and `x2` differ by no more than an absolute error `e`:
/// `abs(x1 - x2) <= e`. Faithfully reproduces the branch-based absolute value of
/// the C++ template (no reliance on `fabs`, so it works for any ordered subtractive
/// type).
#[inline]
pub fn equal_with_abs_error<T>(x1: T, x2: T, e: T) -> bool
where
    T: Copy + PartialOrd + core::ops::Sub<Output = T>,
{
    (if x1 > x2 { x1 - x2 } else { x2 - x1 }) <= e
}

/// Port of Imath's `equalWithRelError`.
///
/// Returns `true` if `x1` and `x2` differ by no more than a relative error `e`:
/// `abs(x1 - x2) <= e * abs(x1)`. Reproduces the C++ branch logic exactly,
/// including the `(x1 > 0) ? x1 : -x1` magnitude of the reference value.
#[inline]
pub fn equal_with_rel_error<T>(x1: T, x2: T, e: T) -> bool
where
    T: Copy
        + PartialOrd
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Neg<Output = T>
        + crate::traits::Scalar,
{
    let diff = if x1 > x2 { x1 - x2 } else { x2 - x1 };
    let mag = if x1 > T::ZERO { x1 } else { -x1 };
    diff <= e * mag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_trait_f32() {
        assert!((Math::sqrt(4.0_f32) - 2.0).abs() < 1e-6);
        // atan2(1, 1) == pi/4 (argument order matches Imath: atan2(x, y)).
        assert!((Math::atan2(1.0_f32, 1.0_f32) - core::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!((Math::log(core::f32::consts::E) - 1.0).abs() < 1e-6);
        assert!((Math::pow(2.0_f32, 10.0) - 1024.0).abs() < 1e-3);
        assert!((Math::fabs(-3.5_f32) - 3.5).abs() < 1e-6);
        assert_eq!(Math::floor(2.7_f32), 2.0);
        assert_eq!(Math::ceil(2.1_f32), 3.0);
        assert!((Math::fmod(5.5_f32, 2.0) - 1.5).abs() < 1e-6);
        assert!((Math::hypot(3.0_f32, 4.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn math_trait_f64() {
        assert!((Math::sqrt(9.0_f64) - 3.0).abs() < 1e-12);
        assert!((Math::atan2(1.0_f64, 1.0_f64) - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((Math::sin(0.0_f64)).abs() < 1e-12);
        assert!((Math::cos(0.0_f64) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn modf_splits_parts() {
        let mut ip = 0.0_f64;
        let frac = Math::modf(3.75_f64, &mut ip);
        assert_eq!(ip, 3.0);
        assert!((frac - 0.75).abs() < 1e-12);
        // Negative: integral part truncates toward zero, fraction keeps sign.
        let frac = Math::modf(-3.75_f64, &mut ip);
        assert_eq!(ip, -3.0);
        assert!((frac + 0.75).abs() < 1e-12);
    }

    #[test]
    fn sinx_over_x_at_zero_is_one() {
        assert_eq!(sinx_over_x(0.0_f32), 1.0);
        assert_eq!(sinx_over_x(0.0_f64), 1.0);
        // For a normal x, equals sin(x)/x.
        let x = 0.5_f64;
        assert!((sinx_over_x(x) - x.sin() / x).abs() < 1e-12);
    }

    #[test]
    fn equal_with_errors() {
        assert!(equal_with_abs_error(1.0_f64, 1.05, 0.1));
        assert!(!equal_with_abs_error(1.0_f64, 1.2, 0.1));
        assert!(equal_with_rel_error(100.0_f64, 105.0, 0.1));
        assert!(!equal_with_rel_error(100.0_f64, 130.0, 0.1));
    }
}
