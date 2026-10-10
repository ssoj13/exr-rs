//! Port of Imath ImathSphere.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathSphere.h>
//!
//! A 3D sphere defined by a centre point and a radius.

use core::fmt;

use crate::box_::Box;
use crate::line::Line3;
use crate::math::Math;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};

/// A 3D sphere (port of C++ `template <class T> class Sphere3`).
///
/// Public fields `center` and `radius` mirror the C++ members.
/// Default is centre at the origin with radius 0.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Sphere3<T> {
    /// Centre of the sphere (C++ `Vec3<T> center`).
    pub center: Vec3<T>,
    /// Radius of the sphere (C++ `T radius`).
    pub radius: T,
}

impl<T: Scalar + VecFloat + VecLimits + Math> Sphere3<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of `Sphere3()` — center (0,0,0), radius 0.
    #[inline]
    pub fn new() -> Self {
        Sphere3 {
            center: Vec3::from_scalar(T::ZERO),
            radius: T::ZERO,
        }
    }

    /// Port of `Sphere3(const Vec3<T>& c, T r)` — given center and radius.
    #[inline]
    pub fn from_center_radius(center: Vec3<T>, radius: T) -> Self {
        Sphere3 { center, radius }
    }

    // -----------------------------------------------------------------------
    // Manipulation
    // -----------------------------------------------------------------------

    /// Port of `circumscribe(const Box<Vec3<T>>& box)`.
    ///
    /// Sets the sphere to the smallest sphere enclosing `box`:
    /// `center = (box.min + box.max) / 2`, `radius = length(box.max - center)`.
    #[inline]
    pub fn circumscribe(&mut self, b: &Box<Vec3<T>>) {
        // C++: center = T(0.5) * (box.min + box.max);
        //      radius = (box.max - center).length();
        let half = T::ONE / (T::ONE + T::ONE);
        self.center = (b.min + b.max) * half;
        self.radius = (b.max - self.center).length();
    }

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /// Port of `intersectT(const Line3<T>& line, T& t)`.
    ///
    /// Finds the smallest `t >= 0` such that `line(t)` lies on the sphere.
    /// Returns `false` if no such `t` exists (the line misses the sphere or
    /// only the `t < 0` portion intersects).
    #[inline]
    pub fn intersect_t(&self, line: &Line3<T>, t: &mut T) -> bool {
        // C++ formula — quadratic in t (A = 1 because dir is normalised):
        //   v = line.pos - center
        //   B = 2 * dot(line.dir, v)
        //   C = dot(v, v) - radius²
        //   discr = B² - 4C
        let v = line.pos - self.center;
        let two = T::ONE + T::ONE;
        let four = two + two;
        let b = two * (line.dir ^ v);
        let c = v.dot(&v) - self.radius * self.radius; // |v|² — self-dot intentional (C++: v ^ v)

        let discr = b * b - four * c;

        if discr < T::ZERO {
            // No intersection
            return false;
        }

        let sq_root = T::sqrt(discr);
        // Try t0 = (-B - sqrt(discr)) / 2  (smaller root — near intersection)
        let t0 = (-b - sq_root) * (T::ONE / two);
        *t = t0;

        if *t < T::ZERO {
            // Near intersection is behind; try t1 = (-B + sqrt(discr)) / 2
            let t1 = (-b + sq_root) * (T::ONE / two);
            *t = t1;
        }

        if *t < T::ZERO {
            // Both roots behind the ray origin
            return false;
        }

        true
    }

    /// Port of `intersect(const Line3<T>& line, Vec3<T>& intersection)`.
    ///
    /// Finds the intersection point closest to `line.pos` with `t >= 0`.
    /// Returns `false` if the sphere is missed or entirely behind the ray.
    #[inline]
    pub fn intersect(&self, line: &Line3<T>, intersection: &mut Vec3<T>) -> bool {
        let mut t = T::ZERO;
        if self.intersect_t(line, &mut t) {
            *intersection = line.point_at(t);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> Default for Sphere3<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Sphere3<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.center == other.center && self.radius == other.radius
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Sphere3<T> {
    /// Formats as "(center, radius)".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.center, self.radius)
    }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathSphere.h)
// ---------------------------------------------------------------------------

/// Sphere of type `f32` (C++ `Sphere3f`).
pub type Sphere3f = Sphere3<f32>;
/// Sphere of type `f64` (C++ `Sphere3d`).
pub type Sphere3d = Sphere3<f64>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_::Box;
    use crate::line::Line3;
    use crate::vec::Vec3;

    /// circumscribe a unit cube [-1..1]³ — radius should be sqrt(3).
    #[test]
    fn sphere3f_circumscribe_unit_cube() {
        let mut s = Sphere3f::new();
        let b = Box::<Vec3<f32>> {
            min: Vec3::new(-1.0f32, -1.0, -1.0),
            max: Vec3::new(1.0f32, 1.0, 1.0),
        };
        s.circumscribe(&b);
        assert!((s.center.x).abs() < 1e-6);
        assert!((s.center.y).abs() < 1e-6);
        assert!((s.center.z).abs() < 1e-6);
        let expected_radius = (3.0f32).sqrt();
        assert!((s.radius - expected_radius).abs() < 1e-5, "radius={}", s.radius);
    }

    /// Ray along +X through origin hits a unit sphere centred at (5, 0, 0).
    /// Expected near-intersection at x = 4.
    #[test]
    fn sphere3f_intersect_hit() {
        let s = Sphere3f::from_center_radius(Vec3::new(5.0f32, 0.0, 0.0), 1.0);
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0f32, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(s.intersect(&line, &mut pt));
        assert!((pt.x - 4.0f32).abs() < 1e-5, "pt.x={}", pt.x);
        assert!((pt.y).abs() < 1e-5);
        assert!((pt.z).abs() < 1e-5);
    }

    /// Ray that completely misses the sphere.
    #[test]
    fn sphere3f_intersect_miss() {
        let s = Sphere3f::from_center_radius(Vec3::new(5.0f32, 0.0, 0.0), 1.0);
        // Ray along +X but offset far in Y — misses the sphere
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0f32, 10.0, 0.0),
            Vec3::new(1.0, 10.0, 0.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(!s.intersect(&line, &mut pt));
    }

    /// intersectT returns the correct parametric t for the near intersection.
    #[test]
    fn sphere3f_intersect_t_value() {
        let s = Sphere3f::from_center_radius(Vec3::new(5.0f32, 0.0, 0.0), 1.0);
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0f32, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let mut t = 0.0f32;
        assert!(s.intersect_t(&line, &mut t));
        // Near intersection at x=4, pos=(0,0,0), dir=(1,0,0) → t=4
        assert!((t - 4.0f32).abs() < 1e-5, "t={}", t);
    }

    /// Ray starting inside the sphere: near root is negative, far root positive.
    #[test]
    fn sphere3f_intersect_t_from_inside() {
        let s = Sphere3f::from_center_radius(Vec3::new(0.0f32, 0.0, 0.0), 2.0);
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0f32, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let mut t = 0.0f32;
        // Should return true with t = 2 (exit point)
        assert!(s.intersect_t(&line, &mut t));
        assert!(t > 0.0f32, "t should be positive, got {}", t);
        assert!((t - 2.0f32).abs() < 1e-5, "t={}", t);
    }

    /// Display format smoke test.
    #[test]
    fn sphere3f_display() {
        let s = Sphere3f::from_center_radius(Vec3::new(1.0f32, 2.0, 3.0), 4.0);
        let disp = format!("{}", s);
        assert!(disp.contains("4"), "display: {disp}");
    }

    /// PartialEq
    #[test]
    fn sphere3f_eq() {
        let a = Sphere3f::from_center_radius(Vec3::new(1.0f32, 0.0, 0.0), 2.0);
        let b = Sphere3f::from_center_radius(Vec3::new(1.0f32, 0.0, 0.0), 2.0);
        assert_eq!(a, b);
    }
}
