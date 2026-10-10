//! Port of Imath ImathLineAlgo.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathLineAlgo.h>
//!
//! Free functions operating on `Line3<T>`: closest-point pairs between two
//! lines, line–triangle intersection (with barycentric coordinates), closest
//! triangle vertex, and rotating a point around a line.

use crate::math::Math;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};
use crate::line::Line3;

/// Port of `closestPoints(const Line3<T>& line1, const Line3<T>& line2,
///                        Vec3<T>& point1, Vec3<T>& point2)`.
///
/// Finds the pair of points — one on each line — at minimum mutual distance.
/// Returns `false` if the lines are parallel or nearly parallel (would require
/// dividing by zero or near-zero).
///
/// Assumes `line1.dir` and `line2.dir` are normalised.
pub fn closest_points<T>(
    line1: &Line3<T>,
    line2: &Line3<T>,
    point1: &mut Vec3<T>,
    point2: &mut Vec3<T>,
) -> bool
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    // C++ verbatim port:
    //   Vec3<T> w = line1.pos - line2.pos;
    //   T d1w  = line1.dir ^ w;
    //   T d2w  = line2.dir ^ w;
    //   T d1d2 = line1.dir ^ line2.dir;
    //   T n1   = d1d2 * d2w - d1w;
    //   T n2   = d2w - d1d2 * d1w;
    //   T d    = 1 - d1d2 * d1d2;
    //   T absD = abs(d);
    //   if (absD > 1 || (|n1| < MAX*absD && |n2| < MAX*absD)) { ... return true }
    //   else { return false }
    let w = line1.pos - line2.pos;
    let d1w = line1.dir ^ w;
    let d2w = line2.dir ^ w;
    let d1d2 = line1.dir ^ line2.dir;
    let n1 = d1d2 * d2w - d1w;
    let n2 = d2w - d1d2 * d1w;
    let d = T::ONE - d1d2 * d1d2;
    let abs_d = if d >= T::ZERO { d } else { T::ZERO - d };

    let abs_n1 = if n1 >= T::ZERO { n1 } else { T::ZERO - n1 };
    let abs_n2 = if n2 >= T::ZERO { n2 } else { T::ZERO - n2 };

    // C++ `abs(d) > 1` — uses the raw value `1` (not `T::ONE * something`),
    // because `d = 1 - cos²θ` which is naturally in [0, 1] for normalised dirs.
    if abs_d > T::ONE
        || (abs_n1 < T::MAX * abs_d && abs_n2 < T::MAX * abs_d)
    {
        *point1 = line1.point_at(n1 / d);
        *point2 = line2.point_at(n2 / d);
        true
    } else {
        false
    }
}

/// Port of `intersect(const Line3<T>& line, const Vec3<T>& v0, const Vec3<T>& v1,
///                    const Vec3<T>& v2, Vec3<T>& pt, Vec3<T>& barycentric, bool& front)`.
///
/// Tests the intersection of a ray (line) with a triangle `(v0, v1, v2)`.
///
/// Returns `false` if:
/// - The triangle has zero area.
/// - The line and the triangle's plane are nearly parallel.
/// - The intersection point is outside the triangle.
///
/// On `true` sets:
/// - `pt` — the 3D intersection point.
/// - `barycentric` — barycentric coordinates such that
///   `pt == v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z`.
/// - `front` — `true` if the line hits the front face (dot(dir, tri_normal) < 0).
pub fn intersect<T>(
    line: &Line3<T>,
    v0: Vec3<T>,
    v1: Vec3<T>,
    v2: Vec3<T>,
    pt: &mut Vec3<T>,
    barycentric: &mut Vec3<T>,
    front: &mut bool,
) -> bool
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    let edge0 = v1 - v0;
    let edge1 = v2 - v1;
    let mut normal = edge1 % edge0;

    let l = normal.length();
    if l != T::ZERO {
        normal = Vec3::new(normal.x / l, normal.y / l, normal.z / l);
    } else {
        return false; // zero-area triangle
    }

    // d  = signed distance of line.pos from the triangle's plane
    // nd = dot(normal, line.dir)
    // intersection: line.pos + (d/nd) * line.dir
    let d = normal ^ (v0 - line.pos);
    let nd = normal ^ line.dir;

    let abs_nd = if nd >= T::ZERO { nd } else { T::ZERO - nd };
    let abs_d = if d >= T::ZERO { d } else { T::ZERO - d };

    if abs_nd > T::ONE || abs_d < T::MAX * abs_nd {
        *pt = line.point_at(d / nd);
    } else {
        return false; // line and plane nearly parallel
    }

    // Compute barycentric.z — projection onto the plane perpendicular to edge0
    {
        let en = edge0.normalized();
        let a = *pt - v0;
        let b = v2 - v0;
        let c = a - en * (en ^ a);
        let dv = b - en * (en ^ b);
        let e = c ^ dv;
        let f = dv.dot(&dv); // |dv|² — self-dot intentional (C++: d ^ d)

        if e >= T::ZERO && e <= f {
            barycentric.z = e / f;
        } else {
            return false; // outside
        }
    }

    // Compute barycentric.x — projection onto the plane perpendicular to edge1
    {
        let en = edge1.normalized();
        let a = *pt - v1;
        let b = v0 - v1;
        let c = a - en * (en ^ a);
        let dv = b - en * (en ^ b);
        let e = c ^ dv;
        let f = dv.dot(&dv); // |dv|² — self-dot intentional (C++: d ^ d)

        if e >= T::ZERO && e <= f {
            barycentric.x = e / f;
        } else {
            return false; // outside
        }
    }

    barycentric.y = T::ONE - barycentric.x - barycentric.z;

    if barycentric.y < T::ZERO {
        return false; // outside
    }

    *front = (line.dir ^ normal) < T::ZERO;
    true
}

/// Port of `closestVertex(const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2,
///                        const Line3<T>& l)`.
///
/// Returns whichever of `v0`, `v1`, `v2` is closest (in 3D space) to the line `l`.
pub fn closest_vertex<T>(v0: Vec3<T>, v1: Vec3<T>, v2: Vec3<T>, l: &Line3<T>) -> Vec3<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    // For each vertex, compute its distance to the line via closest_point_to_point.
    let mut nearest = v0;
    let cp0 = l.closest_point_to_point(v0);
    let diff0 = v0 - cp0;
    let mut near_dot = diff0.x * diff0.x + diff0.y * diff0.y + diff0.z * diff0.z;

    let cp1 = l.closest_point_to_point(v1);
    let diff1 = v1 - cp1;
    let tmp = diff1.x * diff1.x + diff1.y * diff1.y + diff1.z * diff1.z;
    if tmp < near_dot {
        near_dot = tmp;
        nearest = v1;
    }

    let cp2 = l.closest_point_to_point(v2);
    let diff2 = v2 - cp2;
    let tmp2 = diff2.x * diff2.x + diff2.y * diff2.y + diff2.z * diff2.z;
    if tmp2 < near_dot {
        nearest = v2;
    }

    nearest
}

/// Port of `rotatePoint(const Vec3<T> p, Line3<T> l, T angle)`.
///
/// Rotates point `p` by `angle` (radians) around the axis defined by line `l`.
///
/// The rotation is in the plane through `p` perpendicular to `l.dir`.
pub fn rotate_point<T>(p: Vec3<T>, l: Line3<T>, angle: T) -> Vec3<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    // Form a local coordinate frame with x toward p from its projection q,
    // and y = normalise(x × l.dir).  The rotated point is:
    //   q + x * radius * cos(angle) + y * radius * sin(angle)
    let q = l.closest_point_to_point(p);
    let x_vec = p - q;
    let radius = x_vec.length();

    let x_norm = x_vec.normalized();
    // y = normalise(x × dir) — same as C++ `(x % l.dir).normalize()`
    let y_norm = (x_norm % l.dir).normalized();

    let cos_angle = T::cos(angle);
    let sin_angle = T::sin(angle);

    q + x_norm * radius * cos_angle + y_norm * radius * sin_angle
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line::Line3;
    use crate::vec::Vec3;

    /// Two skew lines: line1 along X at y=0,z=1; line2 along Y at x=0,z=-1.
    /// Closest points should both lie on the segment connecting their midpoints
    /// at x=0,y=0 (projected).
    #[test]
    fn closest_points_skew_lines() {
        // line1: along +X at y=0, z=1
        let l1 = Line3::<f64>::from_points(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        );
        // line2: along +Y at x=0, z=-1
        let l2 = Line3::<f64>::from_points(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, -1.0),
        );
        let mut p1 = Vec3::new(0.0, 0.0, 0.0);
        let mut p2 = Vec3::new(0.0, 0.0, 0.0);
        assert!(closest_points(&l1, &l2, &mut p1, &mut p2));
        // Closest on l1 should be at (0, 0, 1), on l2 at (0, 0, -1)
        assert!((p1.x).abs() < 1e-10, "p1.x={}", p1.x);
        assert!((p1.y).abs() < 1e-10, "p1.y={}", p1.y);
        assert!((p1.z - 1.0).abs() < 1e-10, "p1.z={}", p1.z);
        assert!((p2.x).abs() < 1e-10, "p2.x={}", p2.x);
        assert!((p2.y).abs() < 1e-10, "p2.y={}", p2.y);
        assert!((p2.z - (-1.0)).abs() < 1e-10, "p2.z={}", p2.z);
    }

    /// Parallel lines should return false.
    #[test]
    fn closest_points_parallel_returns_false() {
        let l1 = Line3::<f64>::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l2 = Line3::<f64>::from_points(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        let mut p1 = Vec3::new(0.0, 0.0, 0.0);
        let mut p2 = Vec3::new(0.0, 0.0, 0.0);
        // Parallel dirs → d = 1 - 1 = 0 → should return false
        assert!(!closest_points(&l1, &l2, &mut p1, &mut p2));
    }

    /// Line–triangle intersection: a ray straight down the Z axis hits a triangle
    /// lying in the Z=0 plane.
    #[test]
    fn intersect_ray_hits_triangle() {
        // Triangle in Z=0 plane
        let v0 = Vec3::new(0.0f32, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 2.0, 0.0);
        // Ray from (0.5, 0.5, 5) pointing in -Z direction
        let line = Line3::<f32>::from_points(
            Vec3::new(0.5, 0.5, 5.0),
            Vec3::new(0.5, 0.5, 4.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        let mut bary = Vec3::new(0.0f32, 0.0, 0.0);
        let mut front = false;
        assert!(intersect(&line, v0, v1, v2, &mut pt, &mut bary, &mut front));
        assert!((pt.x - 0.5f32).abs() < 1e-5, "pt.x={}", pt.x);
        assert!((pt.y - 0.5f32).abs() < 1e-5, "pt.y={}", pt.y);
        assert!((pt.z).abs() < 1e-5, "pt.z={}", pt.z);
        // Barycentric coords must sum to 1
        let sum = bary.x + bary.y + bary.z;
        assert!((sum - 1.0f32).abs() < 1e-5, "bary sum={}", sum);
    }

    /// Ray parallel to triangle plane returns false.
    #[test]
    fn intersect_ray_parallel_to_triangle() {
        let v0 = Vec3::new(0.0f32, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 2.0, 0.0);
        // Ray lying in z=1 plane, parallel to triangle
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        );
        let mut pt = Vec3::new(0.0f32, 0.0, 0.0);
        let mut bary = Vec3::new(0.0f32, 0.0, 0.0);
        let mut front = false;
        assert!(!intersect(&line, v0, v1, v2, &mut pt, &mut bary, &mut front));
    }

    /// closestVertex returns the triangle vertex nearest to the line.
    #[test]
    fn closest_vertex_test() {
        let v0 = Vec3::new(0.0f32, 0.0, 0.0);
        let v1 = Vec3::new(10.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 10.0, 0.0);
        // Line along X axis at y=0, z=0 — v0 and v1 are ON the line, v2 is off.
        // Closest to (0, 10, 0) is... v2 itself (distance = 10 from the X axis).
        // Closest to v0 = 0, closest to v1 = 0. They're equal so expect v0 (first).
        let line = Line3::<f32>::from_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let cv = closest_vertex(v0, v1, v2, &line);
        // v0 and v1 both have distance 0 (they lie on the line), v0 should win (first).
        assert_eq!(cv, v0);
    }

    /// rotatePoint — rotating a point 90° around the Z axis through the origin.
    ///
    /// C++ computes `y = (x % l.dir).normalize()` i.e. `x × dir`.
    /// With `x = (1,0,0)` and `dir = (0,0,1)`:
    ///   `x × dir = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)`
    /// So rotating (1,0,0) by π/2 yields (0,-1,0).
    #[test]
    fn rotate_point_90_degrees() {
        // Line = Z axis through origin
        let axis = Line3::<f64>::from_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let p = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotate_point(p, axis, core::f64::consts::FRAC_PI_2);
        assert!((rotated.x).abs() < 1e-10, "x={}", rotated.x);
        // C++ y = (x % dir).normalize() = (1,0,0)×(0,0,1) = (0,-1,0), so sin gives -1
        assert!((rotated.y - (-1.0)).abs() < 1e-10, "y={}", rotated.y);
        assert!((rotated.z).abs() < 1e-10, "z={}", rotated.z);
    }
}
