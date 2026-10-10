//! Port of Imath ImathQuat.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathQuat.h>
//!
//! The [`Quat`] quaternion type plus its free functions (`slerp`,
//! `slerpShortestArc`, `squad`, `spline`, `intermediate`, `angle4D`, the
//! `Matrix33`/scalar/`Vec3` operators, and stream output). 1:1 with the C++
//! header; only float/double element types are instantiated (matching the
//! `Quatf`/`Quatd` typedefs — there is no integer or half quaternion in Imath).
//!
//! ## Trait bounds vs. the C++ template
//! The C++ `Quat<T>` is templated on a single `T` and relies on `std::` math
//! (`sqrt`/`acos`/`atan2`/…), `std::numeric_limits<T>::max()`, and the literal
//! `T(0.25)`. Rust spells those implicit requirements out as the [`QuatFloat`]
//! bound (analogous to the existing `VecFloat`/`Bounded` helpers). Everything the
//! generic algorithm needs — and nothing more — lives there, so `Quat<f32>` and
//! `Quat<f64>` are the only instantiations, exactly as in Imath.
//!
//! ## Foreign-interop skip
//! `ImathQuat.h` has no `IMATH_FOREIGN_*_INTEROP` SFINAE of its own (that lives in
//! `ImathVec.h`/`ImathMatrix.h`); there is nothing quaternion-specific to skip.
//!
//! ## `throw` mapping
//! `ImathQuat.h` never throws — none of its functions has an `_exc`/`Result`
//! counterpart, so none is added here.

use core::fmt;

use crate::math::{equal_with_abs_error, equal_with_rel_error, sinx_over_x};
use crate::matrix::{Matrix33, Matrix44};
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat};

/// Floating-point capabilities the quaternion algorithms require of their element
/// type `T`, beyond what [`VecFloat`] already provides.
///
/// `VecFloat` (a super-bound here) already supplies `Scalar` + `Math`, the literal
/// `T(2)` (`VecFloat::TWO`), `std::abs` (`VecFloat::vabs`), and gates the
/// `Vec3::length`/`normalized` family. `crate::math::Epsilon` is required because
/// `slerp` calls `sinx_over_x`, which is bounded on `Epsilon`.
///
/// This trait adds only the few extra per-type constants `log`/`exp`/`intermediate`
/// need that have no `VecFloat` analog: `std::numeric_limits<T>::max()`, the
/// literal `T(1)` magnitude reference, `T(-0.25)`, and `std::min`. Implemented only
/// for `f32`/`f64`, the only types Imath instantiates `Quat` with.
pub trait QuatFloat: VecFloat + crate::math::Epsilon {
    /// `std::numeric_limits<T>::max()`.
    const MAX: Self;
    /// The literal `T(1)`, used as a magnitude reference in `log`/`exp`.
    const ONE_LIT: Self;
    /// The literal `T(-0.25)` used by `intermediate`.
    const NEG_QUARTER: Self;
    /// `std::min(self, other)` (used by `log`'s `std::min(r, T(1))`).
    fn vmin(self, other: Self) -> Self;
}

impl QuatFloat for f32 {
    const MAX: Self = f32::MAX;
    const ONE_LIT: Self = 1.0;
    const NEG_QUARTER: Self = -0.25;
    #[inline]
    fn vmin(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
}

impl QuatFloat for f64 {
    const MAX: Self = f64::MAX;
    const ONE_LIT: Self = 1.0;
    const NEG_QUARTER: Self = -0.25;
    #[inline]
    fn vmin(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
}

/// Quaternion (port of C++ `template <class T> class Quat`).
///
/// `r` is the real (scalar) part; `v` is the imaginary vector `(i, j, k)`.
/// `#[repr(C)]` keeps the layout identical to the C++ class (`T r; Vec3<T> v;`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Quat<T> {
    /// The real part (C++ `T r`).
    pub r: T,
    /// The imaginary vector (C++ `Vec3<T> v`).
    pub v: Vec3<T>,
}

// ----------------------------------------------------------------------------
// Constructors, identity, assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Quat<T> {
    /// Port of the default constructor `Quat()` — the identity quaternion
    /// `r = 1, v = (0, 0, 0)`.
    #[inline]
    fn default() -> Self {
        Quat {
            r: T::ONE,
            v: Vec3::new(T::ZERO, T::ZERO, T::ZERO),
        }
    }
}

impl<T: Copy> Quat<T> {
    /// Port of `Quat(T s, T i, T j, T k)` — real part `s`, imaginary `(i, j, k)`.
    #[inline]
    pub const fn new(s: T, i: T, j: T, k: T) -> Self {
        Quat {
            r: s,
            v: Vec3::new(i, j, k),
        }
    }

    /// Port of `Quat(T s, Vec3<T> d)` — real part `s`, imaginary vector `d`.
    #[inline]
    pub const fn from_scalar_vec(s: T, d: Vec3<T>) -> Self {
        Quat { r: s, v: d }
    }

    /// Port of `template <class S> Quat(const Quat<S>& q)` — cross-base-type
    /// element conversion (C++ `r(q.r), v(q.v)`).
    #[inline]
    pub fn from_quat<S: Copy>(q: Quat<S>) -> Self
    where
        T: crate::vec::VecCast<S>,
    {
        Quat {
            r: T::from_elem(q.r),
            v: Vec3::from_vec3(q.v),
        }
    }
}

impl<T: Scalar> Quat<T> {
    /// Port of `static Quat<T> identity()` — `Quat()` (the identity).
    #[inline]
    pub fn identity() -> Self {
        Quat::default()
    }
}

// ----------------------------------------------------------------------------
// Element access (port of `operator[]`)
// ----------------------------------------------------------------------------

impl<T> core::ops::Index<usize> for Quat<T> {
    type Output = T;
    /// Port of `T operator[](int index) const`: `q[0]` is the real part,
    /// `(q[1], q[2], q[3])` is the imaginary part.
    #[inline]
    fn index(&self, index: usize) -> &T {
        // C++: `index ? v[index - 1] : r`.
        if index == 0 {
            &self.r
        } else {
            &self.v[index - 1]
        }
    }
}

impl<T> core::ops::IndexMut<usize> for Quat<T> {
    /// Port of `T& operator[](int index)`.
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index == 0 {
            &mut self.r
        } else {
            &mut self.v[index - 1]
        }
    }
}

// ----------------------------------------------------------------------------
// Equality (port of `operator==` / `operator!=`, and the helpers)
// ----------------------------------------------------------------------------

impl<T: Scalar> PartialEq for Quat<T> {
    /// Port of `operator==`: `r == q.r && v == q.v`.
    #[inline]
    fn eq(&self, q: &Quat<T>) -> bool {
        // Vec3 has no derived PartialEq; compare componentwise as C++ `v == q.v` does.
        self.r == q.r && self.v.x == q.v.x && self.v.y == q.v.y && self.v.z == q.v.z
    }
}

impl<T: Scalar> Quat<T> {
    /// Port of `equalWithAbsError`-style comparison built on Imath's
    /// `equalWithAbsError`. (Not declared as a named method in the header but the
    /// 4D componentwise absolute-error compare is provided for parity with the
    /// Vec/Matrix API; uses the same `equalWithAbsError` primitive.)
    #[inline]
    pub fn equal_with_abs_error(&self, q: &Quat<T>, e: T) -> bool {
        equal_with_abs_error(self.r, q.r, e)
            && equal_with_abs_error(self.v.x, q.v.x, e)
            && equal_with_abs_error(self.v.y, q.v.y, e)
            && equal_with_abs_error(self.v.z, q.v.z, e)
    }

    /// Componentwise relative-error compare, mirroring [`Self::equal_with_abs_error`]
    /// but using Imath's `equalWithRelError` primitive.
    #[inline]
    pub fn equal_with_rel_error(&self, q: &Quat<T>, e: T) -> bool {
        equal_with_rel_error(self.r, q.r, e)
            && equal_with_rel_error(self.v.x, q.v.x, e)
            && equal_with_rel_error(self.v.y, q.v.y, e)
            && equal_with_rel_error(self.v.z, q.v.z, e)
    }
}

// ----------------------------------------------------------------------------
// Basic algebra (compound-assignment operators)
// ----------------------------------------------------------------------------

impl<T: Scalar> core::ops::MulAssign for Quat<T> {
    /// Port of `operator*=(const Quat& q)` — quaternion multiplication.
    #[inline]
    fn mul_assign(&mut self, q: Quat<T>) {
        // T rtmp = r * q.r - (v ^ q.v);
        // v      = r * q.v + v * q.r + v % q.v;
        // r      = rtmp;
        let rtmp = self.r * q.r - (self.v ^ q.v);
        self.v = q.v * self.r + self.v * q.r + (self.v % q.v);
        self.r = rtmp;
    }
}

impl<T: Scalar> core::ops::MulAssign<T> for Quat<T> {
    /// Port of `operator*=(T t)` — scalar multiplication of both parts.
    #[inline]
    fn mul_assign(&mut self, t: T) {
        self.r = self.r * t;
        self.v = self.v * t;
    }
}

impl<T: Scalar> core::ops::DivAssign for Quat<T> {
    /// Port of `operator/=(const Quat& q)` — `*this = *this * q.inverse()`.
    ///
    /// The `*` inside this `DivAssign` impl is intentional and faithful: C++
    /// defines quaternion `operator/=` as multiplication by the inverse.
    #[inline]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn div_assign(&mut self, q: Quat<T>) {
        *self *= q.inverse();
    }
}

impl<T: Scalar> core::ops::DivAssign<T> for Quat<T> {
    /// Port of `operator/=(T t)` — scalar division of both parts.
    #[inline]
    fn div_assign(&mut self, t: T) {
        self.r = self.r / t;
        self.v = self.v / t;
    }
}

impl<T: Scalar> core::ops::AddAssign for Quat<T> {
    /// Port of `operator+=(const Quat& q)`.
    #[inline]
    fn add_assign(&mut self, q: Quat<T>) {
        self.r = self.r + q.r;
        self.v = self.v + q.v;
    }
}

impl<T: Scalar> core::ops::SubAssign for Quat<T> {
    /// Port of `operator-=(const Quat& q)`.
    #[inline]
    fn sub_assign(&mut self, q: Quat<T>) {
        self.r = self.r - q.r;
        self.v = self.v - q.v;
    }
}

// ----------------------------------------------------------------------------
// Query and utility methods
// ----------------------------------------------------------------------------

impl<T: Scalar> Quat<T> {
    /// Port of `euclideanInnerProduct(const Quat& q)`:
    /// `r * q.r + v.x*q.v.x + v.y*q.v.y + v.z*q.v.z`.
    #[inline]
    pub fn euclidean_inner_product(&self, q: &Quat<T>) -> T {
        self.r * q.r + self.v.x * q.v.x + self.v.y * q.v.y + self.v.z * q.v.z
    }

    /// Port of `inverse()`: `1/Q == Q* / (Q* Q)` where `Q* Q == Q ^ Q` (4D dot).
    #[inline]
    pub fn inverse(&self) -> Quat<T> {
        // T qdot = *this ^ *this;  return Quat(r / qdot, -v / qdot);
        let qdot = quat_dot(self, self);
        Quat::from_scalar_vec(self.r / qdot, -self.v / qdot)
    }

    /// Port of `invert()`: in-place `this = 1 / this`.
    #[inline]
    pub fn invert(&mut self) -> &mut Self {
        let qdot = quat_dot(self, self);
        self.r = self.r / qdot;
        self.v = -self.v / qdot;
        self
    }

    /// Port of `rotateVector(const Vec3& original)`: `p' = q p q*`, assuming a unit
    /// quaternion.
    #[inline]
    pub fn rotate_vector(&self, original: &Vec3<T>) -> Vec3<T> {
        // Quat vec (0, original); Quat inv (*this); inv.v *= -1;
        // Quat result = *this * vec * inv; return result.v;
        let vec = Quat::from_scalar_vec(T::ZERO, *original);
        let mut inv = *self;
        inv.v *= -T::ONE; // C++: inv.v *= -1 (unit multiplicative inverse)
        let result = (*self * vec) * inv;
        result.v
    }
}

impl<T: QuatFloat> Quat<T> {
    /// Port of `length()`: the R4 length `sqrt(r*r + (v ^ v))`.
    #[inline]
    pub fn length(&self) -> T {
        // `self.v ^ self.v` is the genuine Vec3 self-dot (R3 squared length), not a
        // typo — faithful to C++ `sqrt(r*r + (v ^ v))`.
        #[allow(clippy::eq_op)]
        let v_dot_v = self.v ^ self.v;
        (self.r * self.r + v_dot_v).sqrt()
    }

    /// Port of `normalize()` — in place. A null quaternion becomes the identity.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();
        if l != T::ZERO {
            self.r = self.r / l;
            self.v /= l;
        } else {
            self.r = T::ONE;
            self.v = Vec3::from_scalar(T::ZERO);
        }
        self
    }

    /// Port of `normalized()` — copy; null quaternion -> identity.
    #[inline]
    pub fn normalized(&self) -> Quat<T> {
        let l = self.length();
        if l != T::ZERO {
            return Quat::from_scalar_vec(self.r / l, self.v / l);
        }
        Quat::default()
    }

    /// Port of `angle()`: `2 * atan2(v.length(), r)`.
    #[inline]
    pub fn angle(&self) -> T {
        T::TWO * self.v.length().atan2(self.r)
    }

    /// Port of `axis()`: `v.normalized()`.
    #[inline]
    pub fn axis(&self) -> Vec3<T> {
        self.v.normalized()
    }

    /// Port of `setAxisAngle(const Vec3& axis, T radians)` — rotation of `radians`
    /// about `axis`. Returns `&mut self` (C++ returns `*this`).
    #[inline]
    pub fn set_axis_angle(&mut self, axis: &Vec3<T>, radians: T) -> &mut Self {
        // r = cos(radians/2);  v = axis.normalized() * sin(radians/2);
        let half = radians / T::TWO;
        self.r = half.cos();
        self.v = axis.normalized() * half.sin();
        self
    }

    /// Port of `log()` — the logarithm of a unit quaternion.
    #[inline]
    pub fn log(&self) -> Quat<T> {
        // theta = acos(min(r, 1.0));
        let theta = self.r.vmin(T::ONE_LIT).acos();

        if theta == T::ZERO {
            return Quat::from_scalar_vec(T::ZERO, self.v);
        }

        let sintheta = theta.sin();

        // if (|sintheta| < 1 && |theta| >= max * |sintheta|) k = 1; else k = theta/sintheta;
        let k = if sintheta.vabs() < T::ONE_LIT && theta.vabs() >= T::MAX * sintheta.vabs() {
            T::ONE_LIT
        } else {
            theta / sintheta
        };

        Quat::new(T::ZERO, self.v.x * k, self.v.y * k, self.v.z * k)
    }

    /// Port of `exp()` — the exponent of a pure (zero-scalar) quaternion.
    #[inline]
    pub fn exp(&self) -> Quat<T> {
        // theta = v.length(); sintheta = sin(theta);
        let theta = self.v.length();
        let sintheta = theta.sin();

        // if (|theta| < 1 && |sintheta| >= max * |theta|) k = 1; else k = sintheta/theta;
        let k = if theta.vabs() < T::ONE_LIT && sintheta.vabs() >= T::MAX * theta.vabs() {
            T::ONE_LIT
        } else {
            sintheta / theta
        };

        let costheta = theta.cos();

        Quat::new(costheta, self.v.x * k, self.v.y * k, self.v.z * k)
    }

    /// Port of `setRotation(const Vec3& fromDirection, const Vec3& toDirection)` —
    /// builds the rotation mapping `from` onto `to`. Returns `&mut self`.
    ///
    /// Faithfully reproduces the C++ split-angle strategy: for angles `<= pi/2`,
    /// call [`set_rotation_internal`](Quat::set_rotation_internal) directly; for
    /// larger angles, rotate in two halves through the bisector `h0`; and for the
    /// exactly-opposite case (`h0` degenerate), rotate by pi about an arbitrary
    /// axis orthogonal to `f0`.
    #[inline]
    pub fn set_rotation(&mut self, from: &Vec3<T>, to: &Vec3<T>) -> &mut Self {
        let f0 = from.normalized();
        let t0 = to.normalized();

        if (f0 ^ t0) >= T::ZERO {
            // The rotation angle is <= pi/2.
            Self::set_rotation_internal(&f0, &t0, self);
        } else {
            // The angle is > pi/2: split it via the bisector h0.
            let h0 = (f0 + t0).normalized();

            // `h0 ^ h0` is h0's squared length; nonzero means f0/t0 are not exactly
            // opposite. Faithful to C++ `(h0 ^ h0) != 0` — not a typo.
            #[allow(clippy::eq_op)]
            let h0_sq = h0 ^ h0;
            if h0_sq != T::ZERO {
                Self::set_rotation_internal(&f0, &h0, self);

                let mut q = Quat::default();
                Self::set_rotation_internal(&h0, &t0, &mut q);

                *self *= q;
            } else {
                // f0 and t0 point in exactly opposite directions: rotate by pi
                // about an arbitrary axis orthogonal to f0.
                self.r = T::ZERO;

                let f02 = f0 * f0;

                if f02.x <= f02.y && f02.x <= f02.z {
                    self.v = (f0 % Vec3::new(T::ONE, T::ZERO, T::ZERO)).normalized();
                } else if f02.y <= f02.z {
                    self.v = (f0 % Vec3::new(T::ZERO, T::ONE, T::ZERO)).normalized();
                } else {
                    self.v = (f0 % Vec3::new(T::ZERO, T::ZERO, T::ONE)).normalized();
                }
            }
        }

        self
    }

    /// Port of the private `setRotationInternal(const Vec3& f0, const Vec3& t0,
    /// Quat& q)`. Assumes `f0`/`t0` are normalized and the angle between them is
    /// not much greater than pi/2. Equivalent to `setAxisAngle(n, 2*phi)` with `n`
    /// orthogonal to `f0`/`t0`.
    #[inline]
    fn set_rotation_internal(f0: &Vec3<T>, t0: &Vec3<T>, q: &mut Quat<T>) {
        // h0 = (f0 + t0).normalized();
        let h0 = (*f0 + *t0).normalized();

        // q.r = f0 ^ h0;  q.v = f0 % h0;
        q.r = *f0 ^ h0;
        q.v = *f0 % h0;
    }

    /// Port of `toMatrix33()` — the 3x3 rotation matrix for this quaternion.
    #[inline]
    pub fn to_matrix33(&self) -> Matrix33<T> {
        let v = self.v;
        let r = self.r;
        let two = T::TWO;
        let one = T::ONE;
        Matrix33::new(
            one - two * (v.y * v.y + v.z * v.z),
            two * (v.x * v.y + v.z * r),
            two * (v.z * v.x - v.y * r),
            two * (v.x * v.y - v.z * r),
            one - two * (v.z * v.z + v.x * v.x),
            two * (v.y * v.z + v.x * r),
            two * (v.z * v.x + v.y * r),
            two * (v.y * v.z - v.x * r),
            one - two * (v.y * v.y + v.x * v.x),
        )
    }

    /// Port of `toMatrix44()` — the 4x4 rotation matrix for this quaternion.
    #[inline]
    pub fn to_matrix44(&self) -> Matrix44<T> {
        let v = self.v;
        let r = self.r;
        let two = T::TWO;
        let one = T::ONE;
        let zero = T::ZERO;
        Matrix44::new(
            one - two * (v.y * v.y + v.z * v.z),
            two * (v.x * v.y + v.z * r),
            two * (v.z * v.x - v.y * r),
            zero,
            two * (v.x * v.y - v.z * r),
            one - two * (v.z * v.z + v.x * v.x),
            two * (v.y * v.z + v.x * r),
            zero,
            two * (v.z * v.x + v.y * r),
            two * (v.y * v.z - v.x * r),
            one - two * (v.y * v.y + v.x * v.x),
            zero,
            zero,
            zero,
            zero,
            one,
        )
    }
}

impl<T: Scalar> Quat<T> {
    /// Port of `typedef T BaseType` — the element type, exposed as an associated
    /// constant-style helper. (Rust has no nested-typedef analog usable in generic
    /// bounds; this method documents the relationship.)
    #[inline]
    pub fn base_type_value(self) -> T {
        self.r
    }
}

// ----------------------------------------------------------------------------
// Free 4D dot product (port of `operator^(const Quat&, const Quat&)`)
// ----------------------------------------------------------------------------

/// Port of `T operator^(const Quat<T>& q1, const Quat<T>& q2)` — the 4D dot
/// product `q1.r*q2.r + (q1.v ^ q2.v)`.
#[inline]
pub fn quat_dot<T: Scalar>(q1: &Quat<T>, q2: &Quat<T>) -> T {
    q1.r * q2.r + (q1.v ^ q2.v)
}

// `operator^` as a Rust `BitXor` for ergonomic `q1 ^ q2` (4D dot, returns scalar),
// mirroring how Vec3 exposes its dot product through `BitXor`.
impl<T: Scalar> core::ops::BitXor for Quat<T> {
    type Output = T;
    /// Port of `operator^`: the 4D dot product. See [`quat_dot`].
    #[inline]
    fn bitxor(self, q: Quat<T>) -> T {
        quat_dot(&self, &q)
    }
}

// ----------------------------------------------------------------------------
// Binary operators (free functions in C++)
// ----------------------------------------------------------------------------

impl<T: Scalar> core::ops::Mul for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator*(const Quat& q1, const Quat& q2)` — quaternion product.
    #[inline]
    fn mul(self, q2: Quat<T>) -> Quat<T> {
        // Quat(q1.r*q2.r - (q1.v ^ q2.v), q1.r*q2.v + q1.v*q2.r + q1.v % q2.v)
        Quat::from_scalar_vec(
            self.r * q2.r - (self.v ^ q2.v),
            q2.v * self.r + self.v * q2.r + (self.v % q2.v),
        )
    }
}

impl<T: Scalar> core::ops::Div for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator/(const Quat& q1, const Quat& q2)` — `q1 * q2.inverse()`.
    ///
    /// The `*` inside this `Div` impl is intentional: C++ defines quaternion
    /// `operator/` as multiplication by the inverse.
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, q2: Quat<T>) -> Quat<T> {
        self * q2.inverse()
    }
}

impl<T: Scalar> core::ops::Div<T> for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator/(const Quat& q, T t)` — `Quat(q.r/t, q.v/t)`.
    #[inline]
    fn div(self, t: T) -> Quat<T> {
        Quat::from_scalar_vec(self.r / t, self.v / t)
    }
}

impl<T: Scalar> core::ops::Mul<T> for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator*(const Quat& q, T t)` — `Quat(q.r*t, q.v*t)`.
    #[inline]
    fn mul(self, t: T) -> Quat<T> {
        Quat::from_scalar_vec(self.r * t, self.v * t)
    }
}

// `operator*(T t, const Quat& q)` — the scalar-on-the-left overload. Rust cannot
// write a blanket `impl Mul<Quat<T>> for T`, so it is provided as a free function
// `scalar_mul`, plus concrete `f32`/`f64` impls for natural `t * q` syntax.
/// Port of `operator*(T t, const Quat<T>& q)` — `Quat(q.r*t, q.v*t)`.
#[inline]
pub fn scalar_mul<T: Scalar>(t: T, q: Quat<T>) -> Quat<T> {
    Quat::from_scalar_vec(q.r * t, q.v * t)
}

impl core::ops::Mul<Quat<f32>> for f32 {
    type Output = Quat<f32>;
    /// Port of `operator*(T t, const Quat& q)` for `T = float`.
    #[inline]
    fn mul(self, q: Quat<f32>) -> Quat<f32> {
        scalar_mul(self, q)
    }
}

impl core::ops::Mul<Quat<f64>> for f64 {
    type Output = Quat<f64>;
    /// Port of `operator*(T t, const Quat& q)` for `T = double`.
    #[inline]
    fn mul(self, q: Quat<f64>) -> Quat<f64> {
        scalar_mul(self, q)
    }
}

impl<T: Scalar> core::ops::Add for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator+(const Quat& q1, const Quat& q2)`.
    #[inline]
    fn add(self, q2: Quat<T>) -> Quat<T> {
        Quat::from_scalar_vec(self.r + q2.r, self.v + q2.v)
    }
}

impl<T: Scalar> core::ops::Sub for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator-(const Quat& q1, const Quat& q2)`.
    #[inline]
    fn sub(self, q2: Quat<T>) -> Quat<T> {
        Quat::from_scalar_vec(self.r - q2.r, self.v - q2.v)
    }
}

impl<T: Scalar> core::ops::Not for Quat<T> {
    type Output = Quat<T>;
    /// Port of `operator~(const Quat& q)` — the conjugate `Quat(q.r, -q.v)`.
    ///
    /// C++ spells the conjugate as the bitwise-not operator `~q`; Rust's closest
    /// unary operator is `!` (`Not`), so the conjugate is exposed as `!q` and as
    /// the named [`Quat::conjugate`].
    #[inline]
    fn not(self) -> Quat<T> {
        Quat::from_scalar_vec(self.r, -self.v)
    }
}

impl<T: Scalar> Quat<T> {
    /// Named form of the conjugate (`operator~`), `Quat(r, -v)`.
    #[inline]
    pub fn conjugate(&self) -> Quat<T> {
        Quat::from_scalar_vec(self.r, -self.v)
    }
}

impl<T: Scalar> core::ops::Neg for Quat<T> {
    type Output = Quat<T>;
    /// Port of unary `operator-(const Quat& q)` — `Quat(-q.r, -q.v)`.
    #[inline]
    fn neg(self) -> Quat<T> {
        Quat::from_scalar_vec(-self.r, -self.v)
    }
}

// ----------------------------------------------------------------------------
// Matrix33 * Quat and Quat * Matrix33 (free functions in C++)
// ----------------------------------------------------------------------------

/// Port of `operator*(const Matrix33<T>& M, const Quat<T>& q)` — `M * q.toMatrix33()`.
#[inline]
pub fn matrix33_mul_quat<T: QuatFloat>(m: &Matrix33<T>, q: &Quat<T>) -> Matrix33<T> {
    m.mul_matrix(&q.to_matrix33())
}

/// Port of `operator*(const Quat<T>& q, const Matrix33<T>& M)` — `q.toMatrix33() * M`.
#[inline]
pub fn quat_mul_matrix33<T: QuatFloat>(q: &Quat<T>, m: &Matrix33<T>) -> Matrix33<T> {
    q.to_matrix33().mul_matrix(m)
}

// ----------------------------------------------------------------------------
// Vec3 * Quat (free function / operator in C++)
// ----------------------------------------------------------------------------

/// Port of `operator*(const Vec3<T>& v, const Quat<T>& q)` — rotate `v` by `q`.
///
/// `Vec3 a = q.v % v; Vec3 b = q.v % a; return v + 2 * (q.r * a + b);`
#[inline]
pub fn vec3_mul_quat<T: QuatFloat>(v: &Vec3<T>, q: &Quat<T>) -> Vec3<T> {
    let a = q.v % *v;
    let b = q.v % a;
    *v + (a * q.r + b) * T::TWO
}

// `Vec3 * Quat` via the `Mul` trait, mirroring the C++ `operator*`.
impl<T: QuatFloat> core::ops::Mul<Quat<T>> for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator*(const Vec3& v, const Quat& q)`. See [`vec3_mul_quat`].
    #[inline]
    fn mul(self, q: Quat<T>) -> Vec3<T> {
        vec3_mul_quat(&self, &q)
    }
}

// ----------------------------------------------------------------------------
// Interpolation and spline free functions
// ----------------------------------------------------------------------------

/// Port of `angle4D(const Quat<T>& q1, const Quat<T>& q2)` — the angle between two
/// quaternions interpreted as 4D vectors.
#[inline]
pub fn angle4_d<T: QuatFloat>(q1: &Quat<T>, q2: &Quat<T>) -> T {
    // Quat d = q1 - q2; lengthD = sqrt(d ^ d);
    // Quat s = q1 + q2; lengthS = sqrt(s ^ s);
    // return 2 * atan2(lengthD, lengthS);
    let d = *q1 - *q2;
    let length_d = quat_dot(&d, &d).sqrt();

    let s = *q1 + *q2;
    let length_s = quat_dot(&s, &s).sqrt();

    T::TWO * length_d.atan2(length_s)
}

/// Port of `slerp(const Quat<T>& q1, const Quat<T>& q2, T t)` — spherical linear
/// interpolation (Don Hatch's accurate `sinx_over_x` formulation). Does *not*
/// flip to the shortest arc; assumes `q1`/`q2` are normalized and `q1 != -q2`.
#[inline]
pub fn slerp<T: QuatFloat>(q1: &Quat<T>, q2: &Quat<T>, t: T) -> Quat<T> {
    let a = angle4_d(q1, q2);
    let s = T::ONE - t;

    // sinx_over_x(s*a)/sinx_over_x(a) * s * q1 + sinx_over_x(t*a)/sinx_over_x(a) * t * q2
    let c1 = sinx_over_x(s * a) / sinx_over_x(a) * s;
    let c2 = sinx_over_x(t * a) / sinx_over_x(a) * t;

    let q = scalar_mul(c1, *q1) + scalar_mul(c2, *q2);

    q.normalized()
}

/// Port of `slerpShortestArc(const Quat<T>& q1, const Quat<T>& q2, T t)` —
/// interpolates along the shorter of the two arcs to `q2` or `-q2`.
#[inline]
pub fn slerp_shortest_arc<T: QuatFloat>(q1: &Quat<T>, q2: &Quat<T>, t: T) -> Quat<T> {
    if quat_dot(q1, q2) >= T::ZERO {
        slerp(q1, q2, t)
    } else {
        slerp(q1, &(-*q2), t)
    }
}

/// Port of `intermediate(const Quat<T>& q0, const Quat<T>& q1, const Quat<T>& q2)`
/// — the inner quadrangle point guaranteeing tangent continuity for [`spline`].
#[inline]
pub fn intermediate<T: QuatFloat>(q0: &Quat<T>, q1: &Quat<T>, q2: &Quat<T>) -> Quat<T> {
    // q1inv = q1.inverse(); c1 = q1inv*q2; c2 = q1inv*q0;
    // c3 = (-0.25) * (c2.log() + c1.log()); qa = q1 * c3.exp(); qa.normalize();
    let q1inv = q1.inverse();
    let c1 = q1inv * *q2;
    let c2 = q1inv * *q0;
    let c3 = scalar_mul(T::NEG_QUARTER, c2.log() + c1.log());
    let mut qa = *q1 * c3.exp();
    qa.normalize();
    qa
}

/// Port of `squad(const Quat<T>& q1, const Quat<T>& qa, const Quat<T>& qb,
/// const Quat<T>& q2, T t)` — spherical quadrangle interpolation.
#[inline]
pub fn squad<T: QuatFloat>(
    q1: &Quat<T>,
    qa: &Quat<T>,
    qb: &Quat<T>,
    q2: &Quat<T>,
    t: T,
) -> Quat<T> {
    let r1 = slerp(q1, q2, t);
    let r2 = slerp(qa, qb, t);
    slerp(&r1, &r2, T::TWO * t * (T::ONE - t))
}

/// Port of `spline(const Quat<T>& q0, q1, q2, q3, T t)` — spherical cubic spline
/// interpolation between `q1` and `q2`, using neighbours `q0`/`q3` for tangents.
#[inline]
pub fn spline<T: QuatFloat>(
    q0: &Quat<T>,
    q1: &Quat<T>,
    q2: &Quat<T>,
    q3: &Quat<T>,
    t: T,
) -> Quat<T> {
    let qa = intermediate(q0, q1, q2);
    let qb = intermediate(q1, q2, q3);
    squad(q1, &qa, &qb, q2, t)
}

// ----------------------------------------------------------------------------
// Stream output (port of `operator<<`)
// ----------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Quat<T> {
    /// Port of `operator<<` — `"(r x y z)"`.
    #[inline]
    fn fmt(&self, o: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(o, "({} {} {} {})", self.r, self.v.x, self.v.y, self.v.z)
    }
}

// ----------------------------------------------------------------------------
// Typedefs (port of `typedef Quat<float> Quatf; typedef Quat<double> Quatd;`)
// ----------------------------------------------------------------------------

/// Quaternion of type `float` (port of `typedef Quat<float> Quatf`).
pub type Quatf = Quat<f32>;

/// Quaternion of type `double` (port of `typedef Quat<double> Quatd`).
pub type Quatd = Quat<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx_q(a: &Quatd, b: &Quatd, eps: f64) -> bool {
        a.equal_with_abs_error(b, eps)
    }

    #[test]
    fn identity_and_default() {
        let q = Quatd::identity();
        assert_eq!(q.r, 1.0);
        assert_eq!(q.v.x, 0.0);
        assert_eq!(q.v.y, 0.0);
        assert_eq!(q.v.z, 0.0);
        assert!(approx_q(&q, &Quatd::default(), EPS));
    }

    #[test]
    fn index_access() {
        let q = Quatd::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q[0], 1.0);
        assert_eq!(q[1], 2.0);
        assert_eq!(q[2], 3.0);
        assert_eq!(q[3], 4.0);
    }

    #[test]
    fn axis_angle_roundtrip() {
        let axis = Vec3::new(0.0_f64, 0.0, 1.0);
        let radians = 0.7;
        let mut q = Quatd::default();
        q.set_axis_angle(&axis, radians);

        assert!((q.angle() - radians).abs() < 1e-12);
        let a = q.axis();
        assert!((a.x - axis.x).abs() < 1e-12);
        assert!((a.y - axis.y).abs() < 1e-12);
        assert!((a.z - axis.z).abs() < 1e-12);
    }

    #[test]
    fn quat_composition_matches_matrix() {
        // Two rotations about different axes. The quaternion product `q1 * q2`
        // must produce, via toMatrix33, exactly the product of the individual
        // rotation matrices. Imath uses the row-vector convention `v' = v * M`, so
        // applying the `q2` effect first and the `q1` effect second is
        // `v * M_q2 * M_q1`; hence `(q1*q2).toMatrix33() == M_q2 * M_q1`.
        let mut q1 = Quatd::default();
        q1.set_axis_angle(&Vec3::new(1.0, 0.0, 0.0), 0.5);
        let mut q2 = Quatd::default();
        q2.set_axis_angle(&Vec3::new(0.0, 1.0, 0.0), 0.9);

        let qp = q1 * q2;
        let m_from_qp = qp.to_matrix33();
        let m_product = q2.to_matrix33().mul_matrix(&q1.to_matrix33());

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_from_qp.x[i][j] - m_product.x[i][j]).abs() < 1e-10,
                    "({i},{j}): {} vs {}",
                    m_from_qp.x[i][j],
                    m_product.x[i][j]
                );
            }
        }

        // Convention-independent cross-check: rotating a vector by the composed
        // quaternion equals rotating it by the matrix from that same quaternion.
        let p = Vec3::new(0.3, -0.7, 0.2);
        let via_quat = vec3_mul_quat(&p, &qp);
        let via_mat = {
            // v * M (row-vector), matching the Vec3*Quat convention.
            let m = qp.to_matrix33();
            Vec3::new(
                p.x * m.x[0][0] + p.y * m.x[1][0] + p.z * m.x[2][0],
                p.x * m.x[0][1] + p.y * m.x[1][1] + p.z * m.x[2][1],
                p.x * m.x[0][2] + p.y * m.x[1][2] + p.z * m.x[2][2],
            )
        };
        assert!((via_quat.x - via_mat.x).abs() < 1e-10);
        assert!((via_quat.y - via_mat.y).abs() < 1e-10);
        assert!((via_quat.z - via_mat.z).abs() < 1e-10);
    }

    #[test]
    fn rotate_vector_matches_matrix() {
        let mut q = Quatd::default();
        q.set_axis_angle(&Vec3::new(0.0, 0.0, 1.0), core::f64::consts::FRAC_PI_2);

        let p = Vec3::new(1.0, 0.0, 0.0);

        // rotateVector: q p q*  (90deg about z maps (1,0,0) -> (0,1,0)).
        let r = q.rotate_vector(&p);
        assert!((r.x - 0.0).abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
        assert!((r.z - 0.0).abs() < 1e-12);

        // operator* (v * q) must agree with toMatrix33-based transform.
        let r2 = vec3_mul_quat(&p, &q);
        assert!((r2.x - r.x).abs() < 1e-12);
        assert!((r2.y - r.y).abs() < 1e-12);
        assert!((r2.z - r.z).abs() < 1e-12);
    }

    #[test]
    fn slerp_endpoints_and_midpoint() {
        let mut q0 = Quatd::default();
        q0.set_axis_angle(&Vec3::new(0.0, 0.0, 1.0), 0.0);
        let mut q1 = Quatd::default();
        q1.set_axis_angle(&Vec3::new(0.0, 0.0, 1.0), 1.2);

        let s0 = slerp(&q0, &q1, 0.0);
        let s1 = slerp(&q0, &q1, 1.0);
        assert!(approx_q(&s0, &q0.normalized(), 1e-10));
        assert!(approx_q(&s1, &q1.normalized(), 1e-10));

        let sm = slerp(&q0, &q1, 0.5);
        assert!((sm.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_inverse() {
        let q = Quatd::new(1.0, 2.0, 3.0, 4.0);
        let qn = q.normalized();
        assert!((qn.length() - 1.0).abs() < 1e-12);

        let prod = q * q.inverse();
        assert!(approx_q(&prod, &Quatd::identity(), 1e-10));
    }

    #[test]
    fn exp_log_roundtrip() {
        // Small rotation -> unit quaternion; log then exp returns the original.
        let mut q = Quatd::default();
        q.set_axis_angle(&Vec3::new(0.3, 0.5, 0.8), 0.4);
        let q = q.normalized();

        let back = q.log().exp();
        assert!(approx_q(&back, &q, 1e-10));
    }

    #[test]
    fn set_rotation_maps_from_to() {
        // Both an acute case and an obtuse (>pi/2) case.
        for (from, to) in [
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.2, 0.0)),
        ] {
            let mut q = Quatd::default();
            q.set_rotation(&from, &to);

            let rotated = q.rotate_vector(&from.normalized());
            let target = to.normalized();
            assert!((rotated.x - target.x).abs() < 1e-9);
            assert!((rotated.y - target.y).abs() < 1e-9);
            assert!((rotated.z - target.z).abs() < 1e-9);
        }
    }

    #[test]
    fn set_rotation_opposite() {
        // Exactly-opposite directions: rotation by pi maps from -> to.
        let from = Vec3::new(1.0, 0.0, 0.0);
        let to = Vec3::new(-1.0, 0.0, 0.0);
        let mut q = Quatd::default();
        q.set_rotation(&from, &to);
        let rotated = q.rotate_vector(&from);
        assert!((rotated.x - to.x).abs() < 1e-9);
        assert!((rotated.y - to.y).abs() < 1e-9);
        assert!((rotated.z - to.z).abs() < 1e-9);
    }

    #[test]
    fn conjugate_and_display() {
        let q = Quatf::new(1.0, 2.0, 3.0, 4.0);
        let c = q.conjugate();
        assert_eq!(c.r, 1.0);
        assert_eq!(c.v.x, -2.0);
        assert_eq!(format!("{}", q), "(1 2 3 4)");
        // operator~ (Not) agrees with conjugate.
        let n = !q;
        assert_eq!(n.v.x, c.v.x);
    }
}
