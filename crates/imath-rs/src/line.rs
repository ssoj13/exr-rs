//! Port of Imath ImathLine.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathLine.h>
//!
//! A 3D line defined by a base point and a normalised direction vector.

use core::fmt;

use crate::math::Math;
use crate::matrix::matrix44::vec3_mul_matrix44;
use crate::matrix::Matrix44;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};

/// A 3D line (port of C++ `template <class T> class Line3`).
///
/// `pos` is an arbitrary point on the line; `dir` is the **normalised**
/// direction vector.  The two-point constructor and `set` normalise `dir`
/// automatically.  The default constructor leaves both fields uninitialised,
/// matching the C++ `constexpr Line3() IMATH_NOEXCEPT {}`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Line3<T> {
    /// A point on the line (C++ `Vec3<T> pos`).
    pub pos: Vec3<T>,
    /// The normalised direction of the line (C++ `Vec3<T> dir`).
    pub dir: Vec3<T>,
}

impl<T: Scalar + VecFloat + VecLimits + Math> Line3<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of the default constructor `Line3()` — both fields uninitialised.
    ///
    /// Safe Rust zero-inits to `(0,0,0)` for both `pos` and `dir` (C++ leaves
    /// them indeterminate; callers must call `set` or the two-point constructor
    /// before use).
    #[inline]
    pub fn new() -> Self {
        Line3 {
            pos: Vec3::from_scalar(T::ZERO),
            dir: Vec3::from_scalar(T::ZERO),
        }
    }

    /// Port of `Line3(const Vec3<T>& point1, const Vec3<T>& point2)` — two-point
    /// constructor.  `dir` is normalised `(point2 - point1)`.
    #[inline]
    pub fn from_points(point1: Vec3<T>, point2: Vec3<T>) -> Self {
        let mut l = Line3 { pos: Vec3::from_scalar(T::ZERO), dir: Vec3::from_scalar(T::ZERO) };
        l.set(point1, point2);
        l
    }

    // -----------------------------------------------------------------------
    // Manipulation
    // -----------------------------------------------------------------------

    /// Port of `set(const Vec3<T>& point1, const Vec3<T>& point2)` — redefine
    /// the line through two points; `dir = normalise(point2 - point1)`.
    #[inline]
    pub fn set(&mut self, point1: Vec3<T>, point2: Vec3<T>) {
        self.pos = point1;
        self.dir = point2 - point1;
        self.dir.normalize();
    }

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /// Port of `operator()(T parameter) const` — point on the line at
    /// parameter `t`: `pos + dir * t`.
    #[inline]
    pub fn point_at(&self, t: T) -> Vec3<T> {
        Vec3::new(
            self.pos.x + self.dir.x * t,
            self.pos.y + self.dir.y * t,
            self.pos.z + self.dir.z * t,
        )
    }

    /// Port of `closestPointTo(const Vec3<T>& point)` — the point on this line
    /// nearest to the given point.
    ///
    /// C++ formula: `((point - pos) ^ dir) * dir + pos`
    /// where `^` is the dot product.
    #[inline]
    pub fn closest_point_to_point(&self, point: Vec3<T>) -> Vec3<T> {
        // t = dot(point - pos, dir); result = pos + dir * t
        let diff = point - self.pos;
        let t = diff ^ self.dir; // dot product via operator^
        Vec3::new(
            self.pos.x + self.dir.x * t,
            self.pos.y + self.dir.y * t,
            self.pos.z + self.dir.z * t,
        )
    }

    /// Port of `distanceTo(const Vec3<T>& point) const` — distance from line to
    /// point.
    #[inline]
    pub fn distance_to_point(&self, point: Vec3<T>) -> T {
        let closest = self.closest_point_to_point(point);
        (closest - point).length()
    }

    /// Port of `closestPointTo(const Line3<T>& line) const` — the point on this
    /// line closest to another line.
    ///
    /// Assumes both lines are normalised.  Returns `self.pos` if the lines are
    /// (nearly) parallel.
    #[inline]
    pub fn closest_point_to_line(&self, line: &Line3<T>) -> Vec3<T> {
        // Port of the C++ implementation verbatim:
        //   Vec3<T> posLpos = pos - line.pos;
        //   T c   = dir ^ posLpos;
        //   T a   = line.dir ^ dir;
        //   T f   = line.dir ^ posLpos;
        //   T num = c - a * f;
        //   T denom = a * a - 1;
        //   if |denom| < 1 && |num| >= |denom| * MAX  → return pos (parallel)
        //   return pos + dir * (num / denom)
        let pos_lpos = self.pos - line.pos;
        let c = self.dir ^ pos_lpos;
        let a = line.dir ^ self.dir;
        let f = line.dir ^ pos_lpos;
        let num = c - a * f;

        let denom = a * a - T::ONE;

        let abs_denom = if denom >= T::ZERO { denom } else { T::ZERO - denom };

        if abs_denom < T::ONE {
            let abs_num = if num >= T::ZERO { num } else { T::ZERO - num };
            if abs_num >= abs_denom * T::MAX {
                return self.pos;
            }
        }

        self.point_at(num / denom)
    }

    /// Port of `distanceTo(const Line3<T>& line) const` — distance between two
    /// lines.
    ///
    /// C++ formula: `d = (dir % line.dir) ^ (line.pos - pos)` (absolute value).
    #[inline]
    pub fn distance_to_line(&self, line: &Line3<T>) -> T {
        // `%` is cross product, `^` is dot product on Vec3
        let cross = self.dir % line.dir; // operator%
        let d = cross ^ (line.pos - self.pos); // operator^
        if d >= T::ZERO { d } else { T::ZERO - d }
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> Default for Line3<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equality (port of `operator==`)
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Line3<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.dir == other.dir
    }
}

// ---------------------------------------------------------------------------
// Display (port of `operator<<`: "(pos dir)")
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Line3<T> {
    /// Port of `operator<<(ostream& o, const Line3<T>& line)` → `"(pos, dir)"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.pos, self.dir)
    }
}

// ---------------------------------------------------------------------------
// Matrix transform — port of free `operator*(const Line3<S>& line, const Matrix44<T>& M)`
//
//   return Line3<S>(line.pos * M, (line.pos + line.dir) * M);
//
// Both `line.pos` and `line.pos + line.dir` are point transforms (homogeneous
// divide), which is what `vec3_mul_matrix44` provides.
// ---------------------------------------------------------------------------

/// Port of `operator*(const Line3<S>& line, const Matrix44<T>& M)`.
///
/// Transforms the line by the matrix: both `pos` and `pos + dir` are
/// transformed as homogeneous points, then a new `Line3` is constructed from
/// those two transformed points (which normalises `dir`).
pub fn line3_mul_matrix44<T>(line: &Line3<T>, m: &Matrix44<T>) -> Line3<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    let p0 = vec3_mul_matrix44(&line.pos, m);
    let p1 = vec3_mul_matrix44(&(line.pos + line.dir), m);
    Line3::from_points(p0, p1)
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathLine.h)
// ---------------------------------------------------------------------------

/// Line of type `f32` (C++ `Line3f`).
pub type Line3f = Line3<f32>;
/// Line of type `f64` (C++ `Line3d`).
pub type Line3d = Line3<f64>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec3;

    #[test]
    fn line3f_from_points_normalises_dir() {
        let l = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(0.0, 0.0, 3.0));
        // dir should be normalised
        let len = (l.dir.x * l.dir.x + l.dir.y * l.dir.y + l.dir.z * l.dir.z).sqrt();
        assert!((len - 1.0f32).abs() < 1e-6);
        assert_eq!(l.dir, Vec3::new(0.0f32, 0.0, 1.0));
    }

    #[test]
    fn line3f_point_at() {
        let l = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let p = l.point_at(3.0f32);
        assert!((p.x - 3.0f32).abs() < 1e-6);
        assert!((p.y).abs() < 1e-6);
        assert!((p.z).abs() < 1e-6);
    }

    #[test]
    fn line3f_closest_point_to_point() {
        // Line along X axis through origin
        let l = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Point off the line
        let pt = Vec3::new(3.0f32, 5.0, 0.0);
        let closest = l.closest_point_to_point(pt);
        // Closest point should be at (3, 0, 0)
        assert!((closest.x - 3.0f32).abs() < 1e-5);
        assert!((closest.y).abs() < 1e-5);
    }

    #[test]
    fn line3f_distance_to_point() {
        // Line along X axis
        let l = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Point at (0, 3, 4) — distance from X axis = sqrt(9+16) = 5
        let dist = l.distance_to_point(Vec3::new(0.0f32, 3.0, 4.0));
        assert!((dist - 5.0f32).abs() < 1e-5);
    }

    #[test]
    fn line3f_closest_point_to_line() {
        // Two parallel lines along X axis at y=0 and y=1
        let l1 = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line3f::from_points(Vec3::new(0.0f32, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        // Parallel lines: should return l1.pos unchanged
        let cp = l1.closest_point_to_line(&l2);
        assert!((cp.x - l1.pos.x).abs() < 1e-4);

        // Two perpendicular lines: X axis and Y axis through (0,0,1)
        let lx = Line3f::from_points(Vec3::new(0.0f32, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0));
        let ly = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let cp2 = lx.closest_point_to_line(&ly);
        // Closest point on lx to ly should be at x=0
        assert!((cp2.x).abs() < 1e-4);
    }

    #[test]
    fn line3f_transform_by_translation_matrix() {
        use crate::matrix::Matrix44;
        use crate::vec::Vec3;

        // Line along X axis at origin
        let l = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Translate by (0, 5, 0)
        let mut m = Matrix44::<f32>::default();
        m.translate(&Vec3::new(0.0f32, 5.0, 0.0));
        let l2 = line3_mul_matrix44(&l, &m);
        // pos should move to (0, 5, 0); dir should remain (1, 0, 0)
        assert!((l2.pos.y - 5.0f32).abs() < 1e-5);
        assert!((l2.dir.x - 1.0f32).abs() < 1e-5);
        assert!((l2.dir.y).abs() < 1e-5);
    }
}
