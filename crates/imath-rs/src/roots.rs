//! Port of Imath ImathRoots.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathRoots.h>
//!
//! Functions to solve linear, quadratic, and cubic equations:
//! [`solve_linear`], [`solve_quadratic`], [`solve_normalized_cubic`], and
//! [`solve_cubic`] — all ported 1:1.
//!
//! `solveNormalizedCubic` uses `std::complex<T>` for intermediate results even
//! though only real roots are returned. The project forbids external crates, so
//! this module carries a tiny private [`Complex`] type implementing exactly the
//! operations Cardano's formula needs (`sqrt`, `pow` by a real exponent, complex
//! `* / + -`, and `.real()`), preserving the C++ algorithm bit-for-bit in
//! structure.

use crate::math::Math;
use crate::traits::Scalar;

/// Float scalar capabilities the root solvers need, beyond the base [`Scalar`]
/// arithmetic: the transcendental/utility functions from [`Math`] used by
/// Cardano's formula, plus small integer-literal construction (`T(2)`, `T(3)`,
/// `T(27)`, `T(4)`), `copysign`, and the `sqrt(3)` constant. Implemented for the
/// float types Imath instantiates these templates with (`f32`, `f64`).
pub trait RootScalar: Scalar + Math {
    /// Build `T` from a small integer literal (e.g. the `2`, `3`, `4`, `27` that
    /// appear in the cubic/quadratic formulas).
    fn from_i32(n: i32) -> Self;
    /// `std::copysign(self, sign)` — magnitude of `self` with the sign of `sign`.
    fn copysign(self, sign: Self) -> Self;
    /// The `sqrt(3)` constant used by the cubic solver, with the exact digit
    /// string from the C++ source ("enough digits for long double").
    fn sqrt3() -> Self;
}

impl RootScalar for f32 {
    #[inline]
    fn from_i32(n: i32) -> Self {
        n as f32
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }
    #[inline]
    #[allow(clippy::excessive_precision)] // keep Imath's exact digit string; excess digits round away for f32
    fn sqrt3() -> Self {
        1.732_050_807_568_877_293_527_446_341_505_87_f32
    }
}

impl RootScalar for f64 {
    #[inline]
    fn from_i32(n: i32) -> Self {
        n as f64
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    #[inline]
    #[allow(clippy::excessive_precision)] // keep Imath's exact digit string ("enough digits for long double")
    fn sqrt3() -> Self {
        1.732_050_807_568_877_293_527_446_341_505_87_f64
    }
}

/// Port of `solveLinear(T a, T b, T& x)`.
///
/// Solves `a*x + b == 0`. Returns `1` if there is a unique solution (written to
/// `x`), `0` if there is no solution, and `-1` if every real number is a solution.
pub fn solve_linear<T: RootScalar>(a: T, b: T, x: &mut T) -> i32 {
    if a != T::ZERO {
        *x = -b / a;
        1
    } else if b != T::ZERO {
        0
    } else {
        -1
    }
}

/// Port of `solveQuadratic(T a, T b, T c, T x[2])`.
///
/// Solves `a*x^2 + b*x + c == 0`. Returns the number of real solutions (`2`, `1`,
/// or `0`), or `-1` if every real number is a solution. Uses the numerically
/// stable form `q = -(b + sign(b)*sqrt(D))/2` to avoid cancellation, matching the
/// C++ source exactly.
pub fn solve_quadratic<T: RootScalar>(a: T, b: T, c: T, x: &mut [T; 2]) -> i32 {
    if a == T::ZERO {
        // Falls back to the linear solver on x[0], exactly as C++ does.
        let mut x0 = x[0];
        let r = solve_linear(b, c, &mut x0);
        x[0] = x0;
        r
    } else {
        let d = b * b - T::from_i32(4) * a * c;

        if d > T::ZERO {
            let s = d.sqrt();
            let q = -(b + (if b > T::ZERO { T::ONE } else { -T::ONE }) * s) / T::from_i32(2);
            x[0] = q / a;
            x[1] = c / q;
            2
        } else if d == T::ZERO {
            x[0] = -b / (T::from_i32(2) * a);
            1
        } else {
            0
        }
    }
}

/// Port of `solveNormalizedCubic(T r, T s, T t, T x[3])`.
///
/// Solves `x^3 + r*x^2 + s*x + t == 0` via Cardano's formula. Returns the number
/// of real solutions, `0` for none, or `-1` if every real number is a solution.
/// Only real roots are written; intermediate values may be complex.
pub fn solve_normalized_cubic<T: RootScalar>(r: T, s: T, t: T, x: &mut [T; 3]) -> i32 {
    let p = (T::from_i32(3) * s - r * r) / T::from_i32(3);
    let q = T::from_i32(2) * r * r * r / T::from_i32(27) - r * s / T::from_i32(3) + t;
    let p3 = p / T::from_i32(3);
    let q2 = q / T::from_i32(2);
    let d = p3 * p3 * p3 + q2 * q2;

    if d == T::ZERO && p3 == T::ZERO {
        x[0] = -r / T::from_i32(3);
        x[1] = -r / T::from_i32(3);
        x[2] = -r / T::from_i32(3);
        return 1;
    }

    if d > T::ZERO {
        // real_root(a, x) = sign(a) * pow(sign(a)*a, 1/x)
        let real_root = |a: T, xp: T| -> T {
            let sign = T::ONE.copysign(a);
            sign * (sign * a).pow(T::ONE / xp)
        };

        let u = real_root(-q / T::from_i32(2) + d.sqrt(), T::from_i32(3));
        let v = -p / (T::from_i32(3) * u);

        x[0] = u + v - r / T::from_i32(3);
        return 1;
    }

    // D < 0 (or D == 0 with p3 != 0): use complex intermediates, matching C++:
    //   u = pow(-q/2 + sqrt(complex(D)), 1/3)
    //   v = -p / (3 * u)
    let base = Complex::new(-q / T::from_i32(2), T::ZERO) + Complex::new(d, T::ZERO).sqrt();
    let u = base.powf(T::ONE / T::from_i32(3));
    let v = Complex::new(-p, T::ZERO) / (u * T::from_i32(3));

    // sqrt(3) literal, enough digits for long double, matching the C++ constant.
    let sqrt3 = T::sqrt3();

    let y0 = u + v;
    let i_sqrt3 = Complex::new(T::ZERO, sqrt3);
    let half = T::ONE / T::from_i32(2);
    let y1 = -(u + v) * half + (u - v) * half * i_sqrt3;
    let y2 = -(u + v) * half - (u - v) * half * i_sqrt3;

    if d == T::ZERO {
        x[0] = y0.re - r / T::from_i32(3);
        x[1] = y1.re - r / T::from_i32(3);
        2
    } else {
        x[0] = y0.re - r / T::from_i32(3);
        x[1] = y1.re - r / T::from_i32(3);
        x[2] = y2.re - r / T::from_i32(3);
        3
    }
}

/// Port of `solveCubic(T a, T b, T c, T d, T x[3])`.
///
/// Solves `a*x^3 + b*x^2 + c*x + d == 0`. Falls back to [`solve_quadratic`] when
/// `a == 0`; otherwise normalizes and calls [`solve_normalized_cubic`].
pub fn solve_cubic<T: RootScalar>(a: T, b: T, c: T, d: T, x: &mut [T; 3]) -> i32 {
    if a == T::ZERO {
        // C++ passes the same x[] (length 3) to solveQuadratic (which uses x[0..2]).
        let mut xq = [x[0], x[1]];
        let r = solve_quadratic(b, c, d, &mut xq);
        x[0] = xq[0];
        x[1] = xq[1];
        r
    } else {
        solve_normalized_cubic(b / a, c / a, d / a, x)
    }
}

/// Minimal complex number used only by [`solve_normalized_cubic`], replacing
/// `std::complex<T>` without an external crate. Implements exactly the operations
/// Cardano's formula uses; all arithmetic follows the standard complex rules.
#[derive(Clone, Copy)]
struct Complex<T: RootScalar> {
    /// Real component.
    re: T,
    /// Imaginary component.
    im: T,
}

impl<T: RootScalar> Complex<T> {
    #[inline]
    fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Principal complex square root, matching `std::sqrt(std::complex<T>)`.
    ///
    /// Computed via the magnitude/half-angle form so it is correct for negative
    /// real inputs (the case the cubic solver relies on when `D < 0`).
    #[inline]
    fn sqrt(self) -> Self {
        if self.im == T::ZERO {
            if self.re >= T::ZERO {
                Complex::new(self.re.sqrt(), T::ZERO)
            } else {
                // sqrt of a negative real is purely imaginary.
                Complex::new(T::ZERO, (-self.re).sqrt())
            }
        } else {
            let r = self.modulus();
            let re = ((r + self.re) / T::from_i32(2)).sqrt();
            let mut im = ((r - self.re) / T::from_i32(2)).sqrt();
            if self.im < T::ZERO {
                im = -im;
            }
            Complex::new(re, im)
        }
    }

    /// Complex power with a real exponent, matching `std::pow(complex, T)`.
    ///
    /// Uses the polar form `r^p * (cos(p*theta) + i*sin(p*theta))`, equivalent to
    /// `exp(p * log(z))` for the principal branch.
    #[inline]
    fn powf(self, p: T) -> Self {
        let r = self.modulus();
        if r == T::ZERO {
            return Complex::new(T::ZERO, T::ZERO);
        }
        let theta = self.arg();
        let rp = r.pow(p);
        let pt = p * theta;
        Complex::new(rp * pt.cos(), rp * pt.sin())
    }

    /// `|z|` — the complex modulus.
    #[inline]
    fn modulus(self) -> T {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// `arg(z)` — the principal argument.
    #[inline]
    fn arg(self) -> T {
        self.im.atan2(self.re)
    }
}

impl<T: RootScalar> core::ops::Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<T: RootScalar> core::ops::Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<T: RootScalar> core::ops::Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: RootScalar> core::ops::Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl<T: RootScalar> core::ops::Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, s: T) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
}

impl<T: RootScalar> core::ops::Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let denom = o.re * o.re + o.im * o.im;
        Complex::new(
            (self.re * o.re + self.im * o.im) / denom,
            (self.im * o.re - self.re * o.im) / denom,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear() {
        let mut x = 0.0_f64;
        // 2x + 4 = 0 -> x = -2
        assert_eq!(solve_linear(2.0, 4.0, &mut x), 1);
        assert!((x + 2.0).abs() < 1e-12);
        // 0x + 4 = 0 -> no solution
        assert_eq!(solve_linear(0.0, 4.0, &mut x), 0);
        // 0x + 0 = 0 -> all reals
        assert_eq!(solve_linear(0.0, 0.0, &mut x), -1);
    }

    #[test]
    fn quadratic_two_roots() {
        // x^2 - 5x + 6 = 0 -> roots 2 and 3
        let mut x = [0.0_f64; 2];
        let n = solve_quadratic(1.0, -5.0, 6.0, &mut x);
        assert_eq!(n, 2);
        let mut roots = [x[0], x[1]];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] - 2.0).abs() < 1e-10);
        assert!((roots[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn quadratic_double_and_none() {
        // x^2 - 2x + 1 = 0 -> double root 1
        let mut x = [0.0_f64; 2];
        assert_eq!(solve_quadratic(1.0, -2.0, 1.0, &mut x), 1);
        assert!((x[0] - 1.0).abs() < 1e-10);
        // x^2 + 1 = 0 -> no real solution
        assert_eq!(solve_quadratic(1.0, 0.0, 1.0, &mut x), 0);
    }

    #[test]
    fn cubic_three_real_roots() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let mut x = [0.0_f64; 3];
        let n = solve_cubic(1.0, -6.0, 11.0, -6.0, &mut x);
        assert_eq!(n, 3);
        let mut roots = x;
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] - 1.0).abs() < 1e-8, "{roots:?}");
        assert!((roots[1] - 2.0).abs() < 1e-8, "{roots:?}");
        assert!((roots[2] - 3.0).abs() < 1e-8, "{roots:?}");
    }

    #[test]
    fn cubic_one_real_root() {
        // x^3 + x + 1 = 0 -> one real root ~ -0.6823278
        let mut x = [0.0_f64; 3];
        let n = solve_cubic(1.0, 0.0, 1.0, 1.0, &mut x);
        assert_eq!(n, 1);
        // Verify it is actually a root.
        let r = x[0];
        let val = r * r * r + r + 1.0;
        assert!(val.abs() < 1e-9, "root {r} residual {val}");
    }

    #[test]
    fn cubic_falls_back_to_quadratic() {
        // a == 0 -> quadratic x^2 - 5x + 6
        let mut x = [0.0_f64; 3];
        let n = solve_cubic(0.0, 1.0, -5.0, 6.0, &mut x);
        assert_eq!(n, 2);
    }
}
