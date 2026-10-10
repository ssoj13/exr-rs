//! Port of Imath ImathVecAlgo.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathVecAlgo.h>
//!
//! Algorithms applied to or in conjunction with points ([`Vec2`], [`Vec3`],
//! [`Vec4`]): [`project`], [`orthogonal`], [`reflect`], and [`closest_vertex`].
//!
//! # Generic-over-`Vec` mapping
//! In C++ these are `template <class Vec>` free functions that use
//! `Vec::BaseType`, `operator^` (dot), `operator*`/`operator-`, `.normalized()`,
//! and `.length2()`. The float-only ones carry
//! `IMATH_ENABLE_IF(!is_integral<Vec::BaseType>)`. Rust expresses the same set of
//! requirements as the [`VecAlgo`] trait, implemented for `Vec{2,3,4}<f32>`,
//! `Vec{2,3,4}<f64>`, and `Vec{2,3,4}<Half>` — i.e. exactly the non-integral
//! element types, reproducing the C++ `enable_if` restriction. `closestVertex` is
//! not `enable_if`-guarded in C++, but it calls `.length2()` (which only exists
//! for floating-point vectors), so it is likewise bound on [`VecAlgo`] here.

use crate::traits::Scalar;
use crate::vec::{Vec2, Vec3, Vec4};

/// The operations the ImathVecAlgo functions need from a vector type, plus its
/// element type [`VecAlgo::Base`] (the Rust analogue of C++ `Vec::BaseType`).
///
/// Implemented only for floating-point vectors, matching the C++
/// `IMATH_ENABLE_IF(!is_integral<Vec::BaseType>)` constraint.
pub trait VecAlgo: Copy {
    /// The element type, mirroring C++ `Vec::BaseType`.
    type Base: Scalar;

    /// Dot product (C++ `operator^`).
    fn vdot(self, other: Self) -> Self::Base;
    /// Vector subtraction (C++ `operator-`).
    fn vsub(self, other: Self) -> Self;
    /// Scalar multiplication (C++ `vec * scalar`).
    fn vscale(self, s: Self::Base) -> Self;
    /// Normalized copy (C++ `.normalized()`).
    fn vnormalized(self) -> Self;
    /// Squared length (C++ `.length2()`).
    fn vlength2(self) -> Self::Base;
    /// Build `Base` from a small integer literal (the `2` in `reflect`).
    fn base_from_i32(n: i32) -> Self::Base;
}

/// Implements [`VecAlgo`] for one concrete `Vec{2,3,4}<$t>` type.
macro_rules! impl_vec_algo {
    ($vec:ident, $t:ty) => {
        impl VecAlgo for $vec<$t> {
            type Base = $t;
            #[inline]
            fn vdot(self, other: Self) -> $t {
                self.dot(&other)
            }
            #[inline]
            fn vsub(self, other: Self) -> Self {
                self - other
            }
            #[inline]
            fn vscale(self, s: $t) -> Self {
                self * s
            }
            #[inline]
            fn vnormalized(self) -> Self {
                self.normalized()
            }
            #[inline]
            fn vlength2(self) -> $t {
                self.length2()
            }
            #[inline]
            fn base_from_i32(n: i32) -> $t {
                // For Half this routes through the f32 `as` cast via VecCast-style
                // conversion; for f32/f64 it is the direct `as` cast. Either way it
                // reproduces the C++ `typename Vec::BaseType (2)` literal build.
                <$t as crate::vec::VecCast<i32>>::from_elem(n)
            }
        }
    };
}

// Floating-point element types only (the non-integral C++ instantiations).
impl_vec_algo!(Vec2, f32);
impl_vec_algo!(Vec3, f32);
impl_vec_algo!(Vec4, f32);
impl_vec_algo!(Vec2, f64);
impl_vec_algo!(Vec3, f64);
impl_vec_algo!(Vec4, f64);
impl_vec_algo!(Vec2, crate::half::Half);
impl_vec_algo!(Vec3, crate::half::Half);
impl_vec_algo!(Vec4, crate::half::Half);

/// Port of `project(const Vec& s, const Vec& t)`.
///
/// Find the projection of vector `t` onto vector `s`. Only defined for
/// floating-point element types (enforced by the [`VecAlgo`] bound).
#[inline]
pub fn project<V: VecAlgo>(s: V, t: V) -> V {
    let s_normalized = s.vnormalized();
    // sNormalized * (sNormalized ^ t)
    s_normalized.vscale(s_normalized.vdot(t))
}

/// Port of `orthogonal(const Vec& s, const Vec& t)`.
///
/// Find a vector perpendicular to `s` in the plane of `s` and `t`:
/// `t - project(s, t)`.
#[inline]
pub fn orthogonal<V: VecAlgo>(s: V, t: V) -> V {
    t.vsub(project(s, t))
}

/// Port of `reflect(const Vec& s, const Vec& t)`.
///
/// Find the direction of a ray `s` after reflection off a plane with normal `t`:
/// `s - BaseType(2) * (s - project(t, s))`.
#[inline]
pub fn reflect<V: VecAlgo>(s: V, t: V) -> V {
    let two = V::base_from_i32(2);
    // s - 2 * (s - project(t, s))
    s.vsub(s.vsub(project(t, s)).vscale(two))
}

/// Port of `closestVertex(const Vec& v0, const Vec& v1, const Vec& v2, const Vec&
/// p)`.
///
/// Find the vertex of triangle `(v0, v1, v2)` closest to point `p`, comparing
/// squared distances exactly as the C++ does (keeping the first vertex on ties).
#[inline]
pub fn closest_vertex<V: VecAlgo>(v0: V, v1: V, v2: V, p: V) -> V {
    let mut nearest = v0;
    let mut neardot = v0.vsub(p).vlength2();
    let mut tmp = v1.vsub(p).vlength2();

    if tmp < neardot {
        neardot = tmp;
        nearest = v1;
    }

    tmp = v2.vsub(p).vlength2();

    if tmp < neardot {
        // `neardot` is not read again; assignment omitted to match the value while
        // avoiding an unused-assignment warning. (C++ writes it for symmetry.)
        nearest = v2;
    }

    nearest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::Half;
    use crate::vec::{V3h, Vec2, Vec3};

    #[test]
    fn project_onto_axis() {
        // Projecting t onto the x axis keeps only the x component.
        let s = Vec3::new(2.0_f32, 0.0, 0.0);
        let t = Vec3::new(3.0_f32, 4.0, 5.0);
        let p = project(s, t);
        assert!((p.x - 3.0).abs() < 1e-5);
        assert!(p.y.abs() < 1e-5 && p.z.abs() < 1e-5);
    }

    #[test]
    fn orthogonal_is_perpendicular() {
        let s = Vec3::new(1.0_f32, 0.0, 0.0);
        let t = Vec3::new(3.0_f32, 4.0, 0.0);
        let o = orthogonal(s, t);
        // Result is perpendicular to s (dot == 0) and equals the y component.
        assert!(o.dot(&s).abs() < 1e-5);
        assert!((o.y - 4.0).abs() < 1e-5);
    }

    #[test]
    fn reflect_off_plane() {
        // C++ `reflect(s, t) = s - 2*(s - project(t, s)) = 2*project(t,s) - s`.
        // For s=(1,-1,0), normal t=(0,1,0): project(t,s) = (0,-1,0), so the result
        // is 2*(0,-1,0) - (1,-1,0) = (-1,-1,0).
        let s = Vec3::new(1.0_f32, -1.0, 0.0);
        let n = Vec3::new(0.0_f32, 1.0, 0.0);
        let r = reflect(s, n);
        assert!((r.x - (-1.0)).abs() < 1e-5);
        assert!((r.y - (-1.0)).abs() < 1e-5);
        assert!(r.z.abs() < 1e-5);
    }

    #[test]
    fn reflect_vec2() {
        // Same formula as the Vec3 case: 2*project(t,s) - s = (-1,-1).
        let s = Vec2::new(1.0_f64, -1.0);
        let n = Vec2::new(0.0_f64, 1.0);
        let r = reflect(s, n);
        assert!((r.x - (-1.0)).abs() < 1e-12);
        assert!((r.y - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn closest_vertex_picks_nearest() {
        let v0 = Vec3::new(0.0_f32, 0.0, 0.0);
        let v1 = Vec3::new(10.0_f32, 0.0, 0.0);
        let v2 = Vec3::new(0.0_f32, 10.0, 0.0);
        let p = Vec3::new(9.0_f32, 1.0, 0.0);
        assert_eq!(closest_vertex(v0, v1, v2, p), v1);
        let p2 = Vec3::new(0.5_f32, 0.5, 0.0);
        assert_eq!(closest_vertex(v0, v1, v2, p2), v0);
    }

    #[test]
    fn algo_on_half_vectors() {
        // reflect works for V3h (float-promotion path). Result = 2*project(t,s) - s.
        let s = V3h::new(Half::new(1.0), Half::new(-1.0), Half::new(0.0));
        let n = V3h::new(Half::new(0.0), Half::new(1.0), Half::new(0.0));
        let r = reflect(s, n);
        assert!((r.x.to_f32() - (-1.0)).abs() < 0.1);
        assert!((r.y.to_f32() - (-1.0)).abs() < 0.1);
    }
}
