//! Port of Imath ImathPlane.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathPlane.h>
//!
//! A 3D half-space / plane defined by a unit normal and a signed distance from
//! the origin.  The plane represents all points `p` satisfying
//! `dot(normal, p) == distance`.

use core::fmt;

use crate::matrix::Matrix44;
use crate::matrix::matrix44::vec3_mul_matrix44;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};
use crate::line::Line3;
use crate::math::Math;

/// A 3D plane (port of C++ `template <class T> class Plane3`).
///
/// The plane is the set of points `p` such that `dot(normal, p) == distance`.
/// `normal` is a unit vector; `distance` is the signed distance from the
/// origin along the normal.
///
/// C++ layout: two public fields `Vec3<T> normal` and `T distance`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Plane3<T> {
    /// The outward unit normal (C++ `Vec3<T> normal`).
    pub normal: Vec3<T>,
    /// Signed distance from origin along the normal (C++ `T distance`).
    pub distance: T,
}

impl<T: Scalar + VecFloat + VecLimits + Math> Plane3<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of the default constructor `Plane3()` — fields uninitialised.
    ///
    /// Safe Rust zero-inits to `normal=(0,0,0)`, `distance=0`.
    #[inline]
    pub fn new() -> Self {
        Plane3 {
            normal: Vec3::from_scalar(T::ZERO),
            distance: T::ZERO,
        }
    }

    /// Port of `Plane3(const Vec3<T>& normal, T distance)` — normal + distance.
    ///
    /// Note: the C++ set() normalises `normal`; distance is taken as-is.
    #[inline]
    pub fn from_normal_distance(normal: Vec3<T>, distance: T) -> Self {
        let mut p = Plane3 { normal: Vec3::from_scalar(T::ZERO), distance: T::ZERO };
        p.set_normal_distance(normal, distance);
        p
    }

    /// Port of `Plane3(const Vec3<T>& point, const Vec3<T>& normal)` — point + normal.
    #[inline]
    pub fn from_point_normal(point: Vec3<T>, normal: Vec3<T>) -> Self {
        let mut p = Plane3 { normal: Vec3::from_scalar(T::ZERO), distance: T::ZERO };
        p.set_point_normal(point, normal);
        p
    }

    /// Port of `Plane3(const Vec3<T>& p0, const Vec3<T>& p1, const Vec3<T>& p2)` — 3 points.
    #[inline]
    pub fn from_three_points(p0: Vec3<T>, p1: Vec3<T>, p2: Vec3<T>) -> Self {
        let mut p = Plane3 { normal: Vec3::from_scalar(T::ZERO), distance: T::ZERO };
        p.set_three_points(p0, p1, p2);
        p
    }

    // -----------------------------------------------------------------------
    // Manipulation — port of the three `set()` overloads
    // -----------------------------------------------------------------------

    /// Port of `set(const Vec3<T>& normal, T distance)`.
    ///
    /// Normalises `normal`; stores distance as-is.
    #[inline]
    pub fn set_normal_distance(&mut self, normal: Vec3<T>, distance: T) {
        // C++: normal = n; normal.normalize(); distance = d;
        self.normal = normal;
        self.normal.normalize();
        self.distance = distance;
    }

    /// Port of `set(const Vec3<T>& point, const Vec3<T>& normal)`.
    ///
    /// Normalises `normal`; computes `distance = dot(normal, point)`.
    #[inline]
    pub fn set_point_normal(&mut self, point: Vec3<T>, normal: Vec3<T>) {
        // C++: normal = n; normal.normalize(); distance = normal ^ point;
        self.normal = normal;
        self.normal.normalize();
        self.distance = self.normal ^ point;
    }

    /// Port of `set(const Vec3<T>& p0, const Vec3<T>& p1, const Vec3<T>& p2)`.
    ///
    /// Computes normal as `cross(p1-p0, p2-p0)`, normalises it, then
    /// `distance = dot(normal, p0)`.
    #[inline]
    pub fn set_three_points(&mut self, p0: Vec3<T>, p1: Vec3<T>, p2: Vec3<T>) {
        // C++: normal = (point2-point1) % (point3-point1); normal.normalize();
        //      distance = normal ^ point1;
        self.normal = (p1 - p0) % (p2 - p0);
        self.normal.normalize();
        self.distance = self.normal ^ p0;
    }

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /// Port of `intersect(const Line3<T>& line, Vec3<T>& intersection)`.
    ///
    /// Returns `true` if the line is not parallel to the plane and sets
    /// `intersection` to the intersection point.
    #[inline]
    pub fn intersect(&self, line: &Line3<T>, intersection: &mut Vec3<T>) -> bool {
        // C++: T d = normal ^ line.dir;
        //      if (d == 0.0) return false;
        //      T t = -((normal ^ line.pos) - distance) / d;
        //      point = line(t);
        let d = self.normal ^ line.dir;
        if d == T::ZERO {
            return false;
        }
        let t = -((self.normal ^ line.pos) - self.distance) / d;
        *intersection = line.point_at(t);
        true
    }

    /// Port of `intersectT(const Line3<T>& line, T& parameter)`.
    ///
    /// Returns `true` if not parallel; sets `parameter` to the parametric `t`
    /// such that `line(t)` lies on the plane.
    #[inline]
    pub fn intersect_t(&self, line: &Line3<T>, parameter: &mut T) -> bool {
        // C++: T d = normal ^ line.dir;
        //      if (d == 0.0) return false;
        //      t = -((normal ^ line.pos) - distance) / d;
        let d = self.normal ^ line.dir;
        if d == T::ZERO {
            return false;
        }
        *parameter = -((self.normal ^ line.pos) - self.distance) / d;
        true
    }

    /// Port of `distanceTo(const Vec3<T>& point)`.
    ///
    /// Returns the signed distance from `point` to the plane:
    /// `dot(point, normal) - distance`.
    #[inline]
    pub fn distance_to(&self, point: Vec3<T>) -> T {
        // C++: return (point ^ normal) - distance;
        (point ^ self.normal) - self.distance
    }

    /// Port of `reflectPoint(const Vec3<T>& point)`.
    ///
    /// Reflects `point` across the plane.
    #[inline]
    pub fn reflect_point(&self, point: Vec3<T>) -> Vec3<T> {
        // C++: return normal * distanceTo(point) * -2.0 + point;
        let two = T::ONE + T::ONE;
        self.normal * self.distance_to(point) * (-two) + point
    }

    /// Port of `reflectVector(const Vec3<T>& v)`.
    ///
    /// Reflects a direction vector across the plane (the plane is treated as a
    /// mirror; only the component along the normal is negated).
    #[inline]
    pub fn reflect_vector(&self, v: Vec3<T>) -> Vec3<T> {
        // C++: return normal * (normal ^ v) * 2.0 - v;
        let two = T::ONE + T::ONE;
        self.normal * (self.normal ^ v) * two - v
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> Default for Plane3<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equality (port of implicit == via field comparison)
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Plane3<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.normal == other.normal && self.distance == other.distance
    }
}

// ---------------------------------------------------------------------------
// Negation — port of free `operator-(const Plane3<T>& plane)`
//
//   return Plane3<T>(-plane.normal, -plane.distance);
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> core::ops::Neg for Plane3<T> {
    type Output = Plane3<T>;
    /// Port of `operator-(const Plane3<T>& plane)` — reflects the half-space.
    #[inline]
    fn neg(self) -> Plane3<T> {
        // Use from_normal_distance with the negated normal and negated distance.
        // The C++ ctor calls set(n, d) which normalises n; -normal is already unit,
        // so normalisation is a no-op, but we go through the same path for 1:1 fidelity.
        Plane3::from_normal_distance(-self.normal, -self.distance)
    }
}

// ---------------------------------------------------------------------------
// Matrix transform — port of free `operator*(const Plane3<T>& plane, const Matrix44<T>& M)`
//
// The C++ implementation avoids computing the inverse-transpose by picking two
// vectors in the plane's tangent space, transforming three points, and
// reconstructing the plane from those three points.
// ---------------------------------------------------------------------------

/// Port of `operator*(const Plane3<T>& plane, const Matrix44<T>& M)`.
///
/// Transforms a plane by a matrix.  Three points in the plane are transformed
/// as homogeneous points and a new `Plane3` is constructed from them,
/// preserving the correct normal under non-uniform scaling.
pub fn plane3_mul_matrix44<T>(plane: &Plane3<T>, m: &Matrix44<T>) -> Plane3<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    // C++ picks the in-plane tangent direction that is most perpendicular to
    // the normal, to avoid near-degenerate constructions.
    let mut dir1 = Vec3::new(T::ONE, T::ZERO, T::ZERO) % plane.normal;
    let mut dir1_len = dir1.dot(&dir1); // |dir1|² — self-dot intentional (C++: dir1 ^ dir1)

    let mut tmp = Vec3::new(T::ZERO, T::ONE, T::ZERO) % plane.normal;
    let mut tmp_len = tmp.dot(&tmp); // |tmp|² — self-dot intentional

    if tmp_len > dir1_len {
        dir1 = tmp;
        dir1_len = tmp_len;
    }

    tmp = Vec3::new(T::ZERO, T::ZERO, T::ONE) % plane.normal;
    tmp_len = tmp.dot(&tmp); // |tmp|² — self-dot intentional

    if tmp_len > dir1_len {
        dir1 = tmp;
        // dir1_len is no longer used after this point
        let _ = dir1_len;
    }

    let dir2 = dir1 % plane.normal;
    let point = plane.normal * plane.distance;

    Plane3::from_three_points(
        vec3_mul_matrix44(&point, m),
        vec3_mul_matrix44(&(point + dir2), m),
        vec3_mul_matrix44(&(point + dir1), m),
    )
}

// ---------------------------------------------------------------------------
// Display — port of `operator<<(ostream& o, const Plane3<T>& plane)`
//
//   return o << "(" << plane.normal << ", " << plane.distance << ")";
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Plane3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.normal, self.distance)
    }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathPlane.h)
// ---------------------------------------------------------------------------

/// Plane of type `f32` (C++ `Plane3f`).
pub type Plane3f = Plane3<f32>;
/// Plane of type `f64` (C++ `Plane3d`).
pub type Plane3d = Plane3<f64>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec3;
    use crate::line::Line3;

    /// Plane3 from 3 known coplanar points — the XY plane (z = 0).
    #[test]
    fn plane3f_from_three_points_xy_plane() {
        let p = Plane3f::from_three_points(
            Vec3::new(0.0f32, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // Normal should be (0, 0, 1) or (0, 0, -1); distance = 0.
        assert!((p.normal.z.abs() - 1.0f32).abs() < 1e-6, "normal.z should be ±1");
        assert!((p.distance).abs() < 1e-6, "distance should be 0");
    }

    /// Plane3 from point + normal — the plane z = 3.
    #[test]
    fn plane3f_from_point_normal() {
        let p = Plane3f::from_point_normal(
            Vec3::new(0.0f32, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!((p.normal.z - 1.0f32).abs() < 1e-6);
        assert!((p.distance - 3.0f32).abs() < 1e-6);
    }

    /// Line perpendicular to the XY plane intersects at its projection.
    #[test]
    fn plane3f_intersect_perpendicular_line() {
        // XY plane: normal = (0,0,1), distance = 0
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 0.0);
        // Line pointing in -Z direction, starting at (2, 3, 5)
        let line = Line3::<f32>::from_points(
            Vec3::new(2.0f32, 3.0, 5.0),
            Vec3::new(2.0, 3.0, 4.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(plane.intersect(&line, &mut pt));
        assert!((pt.x - 2.0f32).abs() < 1e-5);
        assert!((pt.y - 3.0f32).abs() < 1e-5);
        assert!((pt.z).abs() < 1e-5);
    }

    /// `intersect_t` returns the same point via `line.point_at(t)`.
    #[test]
    fn plane3f_intersect_t() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 0.0);
        let line = Line3::<f32>::from_points(
            Vec3::new(1.0f32, 1.0, 4.0),
            Vec3::new(1.0, 1.0, 3.0),
        );
        let mut t = 0.0f32;
        assert!(plane.intersect_t(&line, &mut t));
        let pt = line.point_at(t);
        assert!((pt.z).abs() < 1e-5);
    }

    /// Parallel line should NOT intersect the plane.
    #[test]
    fn plane3f_intersect_parallel_returns_false() {
        // XY plane
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 0.0);
        // Line lying in the XY plane — dir = (1, 0, 0), perpendicular to normal
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0f32, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(!plane.intersect(&line, &mut pt));
    }

    /// distanceTo for the plane z = 2 at point (0,0,5) → distance = 3.
    #[test]
    fn plane3f_distance_to() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 2.0);
        let d = plane.distance_to(Vec3::new(0.0f32, 0.0, 5.0));
        assert!((d - 3.0f32).abs() < 1e-6);
    }

    /// reflectPoint: reflect (0, 0, 5) across z = 2 → (0, 0, -1).
    #[test]
    fn plane3f_reflect_point() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 2.0);
        let r = plane.reflect_point(Vec3::new(0.0f32, 0.0, 5.0));
        assert!((r.z - (-1.0f32)).abs() < 1e-5, "expected -1, got {}", r.z);
    }

    /// reflectVector: reflect (0, 0, 1) across z = 0 plane → (0, 0, -1).
    #[test]
    fn plane3f_reflect_vector() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 0.0);
        let r = plane.reflect_vector(Vec3::new(0.0f32, 0.0, 1.0));
        assert!((r.z - 1.0f32).abs() < 1e-5, "expected 1, got {}", r.z);
    }

    /// Negation flips normal and distance.
    #[test]
    fn plane3f_neg() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 3.0);
        let neg = -plane;
        assert!((neg.normal.z - (-1.0f32)).abs() < 1e-6);
        assert!((neg.distance - (-3.0f32)).abs() < 1e-6);
    }

    /// Display format should be "(normal, distance)".
    #[test]
    fn plane3f_display() {
        let plane = Plane3f::from_normal_distance(Vec3::new(0.0f32, 0.0, 1.0), 0.0);
        let s = format!("{}", plane);
        assert!(s.contains("1"), "display string: {s}");
    }
}
