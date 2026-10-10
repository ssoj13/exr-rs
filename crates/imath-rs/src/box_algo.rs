//! Port of Imath ImathBoxAlgo.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathBoxAlgo.h>
//!
//! Axis-aligned bounding box utility functions: clip, closestPointInBox,
//! closestPointOnBox, transform (returning + into-result overloads),
//! affineTransform (returning + into-result overloads),
//! findEntryAndExitPoints, intersects (bool+point + bool-only overloads).

use crate::box_::{Box, BoxVector};
use crate::fun::{abs, clamp};
use crate::line::Line3;
use crate::matrix::Matrix44;
use crate::matrix::matrix44::vec3_mul_matrix44;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};

// ---------------------------------------------------------------------------
// clip
// ---------------------------------------------------------------------------

/// Port of `clip(const T& p, const Box<T>& box)`.
///
/// Clips all components of `p` to lie within `[box.min, box.max]`.
/// Returns the closest point to `p` that is inside (or on the surface of) the box.
/// Works generically for both 2D and 3D boxes.
#[inline]
pub fn clip<V>(p: V, b: &Box<V>) -> V
where
    V: BoxVector,
{
    let mut q = p;
    for i in 0..V::dimensions() {
        let pi = p.get(i);
        if pi < b.min.get(i) {
            q.set(i, b.min.get(i));
        } else if pi > b.max.get(i) {
            q.set(i, b.max.get(i));
        } else {
            q.set(i, pi);
        }
    }
    q
}

// ---------------------------------------------------------------------------
// closestPointInBox
// ---------------------------------------------------------------------------

/// Port of `closestPointInBox(const T& p, const Box<T>& box)`.
///
/// Returns the point in or on `box` that is closest to `p`.
/// Delegates to `clip`.
#[inline]
pub fn closest_point_in_box<V>(p: V, b: &Box<V>) -> V
where
    V: BoxVector,
{
    clip(p, b)
}

// ---------------------------------------------------------------------------
// closestPointOnBox
// ---------------------------------------------------------------------------

/// Port of `closestPointOnBox(const Vec3<T>& p, const Box<Vec3<T>>& box)`.
///
/// Returns the point on the **surface** of `box` closest to `p`.
/// If `box` is empty, returns `p` unchanged.
/// If `p` is strictly inside the box, the result is projected onto the
/// nearest face using the same tie-breaking as C++ (d.x < d.y && d.x < d.z,
/// else d.y < d.z).
#[inline]
pub fn closest_point_on_box<T>(p: Vec3<T>, b: &Box<Vec3<T>>) -> Vec3<T>
where
    T: Scalar + VecLimits,
{
    if b.is_empty() {
        return p;
    }

    let mut q = closest_point_in_box(p, b);

    if q == p {
        // p is strictly inside — find the nearest face in each axis
        let d1 = Vec3::new(p.x - b.min.x, p.y - b.min.y, p.z - b.min.z);
        let d2 = Vec3::new(b.max.x - p.x, b.max.y - p.y, b.max.z - p.z);

        let dx = if d1.x < d2.x { d1.x } else { d2.x };
        let dy = if d1.y < d2.y { d1.y } else { d2.y };
        let dz = if d1.z < d2.z { d1.z } else { d2.z };

        if dx < dy && dx < dz {
            q.x = if d1.x < d2.x { b.min.x } else { b.max.x };
        } else if dy < dz {
            q.y = if d1.y < d2.y { b.min.y } else { b.max.y };
        } else {
            q.z = if d1.z < d2.z { b.min.z } else { b.max.z };
        }
    }

    q
}

// ---------------------------------------------------------------------------
// transform (returning) — Box<Vec3<S>> transform(box, M44)
// ---------------------------------------------------------------------------

/// Port of `Box<Vec3<S>> transform(const Box<Vec3<S>>& box, const Matrix44<T>& m)`.
///
/// Transforms a 3D AABB by `m` and returns the tightest enclosing AABB of the
/// result.  Empty and infinite boxes are returned unchanged.
///
/// If `m` has its last column equal to `(0 0 0 1)` (i.e. it is affine), the
/// fast Graphics Gems trick (James Arvo, 1990) is used.  Otherwise all eight
/// corners are transformed explicitly.
#[inline]
pub fn transform<T>(b: &Box<Vec3<T>>, m: &Matrix44<T>) -> Box<Vec3<T>>
where
    T: Scalar + VecLimits,
{
    if b.is_empty() || b.is_infinite() {
        return *b;
    }

    // Affine fast path: last column is (0 0 0 1).
    if m.x[0][3] == T::ZERO
        && m.x[1][3] == T::ZERO
        && m.x[2][3] == T::ZERO
        && m.x[3][3] == T::ONE
    {
        let mut nb = Box::<Vec3<T>>::new();
        for i in 0..3 {
            nb.min.set(i, m.x[3][i]);
            nb.max.set(i, m.x[3][i]);

            for j in 0..3 {
                let a = m.x[j][i] * b.min.get(j);
                let b_val = m.x[j][i] * b.max.get(j);
                if a < b_val {
                    nb.min.set(i, nb.min.get(i) + a);
                    nb.max.set(i, nb.max.get(i) + b_val);
                } else {
                    nb.min.set(i, nb.min.get(i) + b_val);
                    nb.max.set(i, nb.max.get(i) + a);
                }
            }
        }
        return nb;
    }

    // Projection / non-affine: transform all 8 corners and find enclosing AABB.
    let corners = eight_corners(b);
    let mut nb = Box::<Vec3<T>>::new();
    for c in &corners {
        nb.extend_by_point(vec3_mul_matrix44(c, m));
    }
    nb
}

// ---------------------------------------------------------------------------
// transform (into-result) — void transform(box, M44, result)
// ---------------------------------------------------------------------------

/// Port of `void transform(const Box<Vec3<S>>& box, const Matrix44<T>& m, Box<Vec3<S>>& result)`.
///
/// Same algorithm as the returning variant but writes into `result`.
/// Empty or infinite boxes cause an early return without touching `result`.
#[inline]
pub fn transform_into<T>(b: &Box<Vec3<T>>, m: &Matrix44<T>, result: &mut Box<Vec3<T>>)
where
    T: Scalar + VecLimits,
{
    if b.is_empty() || b.is_infinite() {
        return;
    }

    if m.x[0][3] == T::ZERO
        && m.x[1][3] == T::ZERO
        && m.x[2][3] == T::ZERO
        && m.x[3][3] == T::ONE
    {
        for i in 0..3 {
            result.min.set(i, m.x[3][i]);
            result.max.set(i, m.x[3][i]);

            for j in 0..3 {
                let a = m.x[j][i] * b.min.get(j);
                let b_val = m.x[j][i] * b.max.get(j);
                if a < b_val {
                    result.min.set(i, result.min.get(i) + a);
                    result.max.set(i, result.max.get(i) + b_val);
                } else {
                    result.min.set(i, result.min.get(i) + b_val);
                    result.max.set(i, result.max.get(i) + a);
                }
            }
        }
        return;
    }

    let corners = eight_corners(b);
    for c in &corners {
        result.extend_by_point(vec3_mul_matrix44(c, m));
    }
}

// ---------------------------------------------------------------------------
// affineTransform (returning) — Box<Vec3<S>> affineTransform(box, M44)
// ---------------------------------------------------------------------------

/// Port of `Box<Vec3<S>> affineTransform(const Box<Vec3<S>>& box, const Matrix44<T>& m)`.
///
/// Like `transform`, but asserts the matrix is affine (skips the last-column
/// check and always uses the Graphics Gems fast path).  Empty and infinite
/// boxes are returned unchanged.
#[inline]
pub fn affine_transform<T>(b: &Box<Vec3<T>>, m: &Matrix44<T>) -> Box<Vec3<T>>
where
    T: Scalar + VecLimits,
{
    if b.is_empty() || b.is_infinite() {
        return *b;
    }

    let mut nb = Box::<Vec3<T>>::new();
    for i in 0..3 {
        nb.min.set(i, m.x[3][i]);
        nb.max.set(i, m.x[3][i]);

        for j in 0..3 {
            let a = m.x[j][i] * b.min.get(j);
            let b_val = m.x[j][i] * b.max.get(j);
            if a < b_val {
                nb.min.set(i, nb.min.get(i) + a);
                nb.max.set(i, nb.max.get(i) + b_val);
            } else {
                nb.min.set(i, nb.min.get(i) + b_val);
                nb.max.set(i, nb.max.get(i) + a);
            }
        }
    }
    nb
}

// ---------------------------------------------------------------------------
// affineTransform (into-result) — void affineTransform(box, M44, result)
// ---------------------------------------------------------------------------

/// Port of `void affineTransform(const Box<Vec3<S>>& box, const Matrix44<T>& m, Box<Vec3<S>>& result)`.
///
/// Like the returning variant but writes into `result`.
/// An empty `box` resets `result` to empty; an infinite `box` makes `result` infinite.
#[inline]
pub fn affine_transform_into<T>(b: &Box<Vec3<T>>, m: &Matrix44<T>, result: &mut Box<Vec3<T>>)
where
    T: Scalar + VecLimits,
{
    if b.is_empty() {
        result.make_empty();
        return;
    }
    if b.is_infinite() {
        result.make_infinite();
        return;
    }

    for i in 0..3 {
        result.min.set(i, m.x[3][i]);
        result.max.set(i, m.x[3][i]);

        for j in 0..3 {
            let a = m.x[j][i] * b.min.get(j);
            let b_val = m.x[j][i] * b.max.get(j);
            if a < b_val {
                result.min.set(i, result.min.get(i) + a);
                result.max.set(i, result.max.get(i) + b_val);
            } else {
                result.min.set(i, result.min.get(i) + b_val);
                result.max.set(i, result.max.get(i) + a);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// findEntryAndExitPoints
// ---------------------------------------------------------------------------

/// Port of `bool findEntryAndExitPoints(const Line3<T>& r, const Box<Vec3<T>>& b,
///   Vec3<T>& entry, Vec3<T>& exit)`.
///
/// Computes the points where ray `r` enters and exits box `b`.
/// Returns `true` if the ray intersects the box (including the case where its
/// origin is inside the box); `false` if `b` is empty or the ray misses.
/// `entry` and `exit` are undefined when the function returns `false`.
///
/// The algorithm is the slab method: the most distant "front-facing" slab
/// intersection must be closer than the nearest "back-facing" one.
#[inline]
pub fn find_entry_and_exit_points<T>(
    r: &Line3<T>,
    b: &Box<Vec3<T>>,
    entry: &mut Vec3<T>,
    exit: &mut Vec3<T>,
) -> bool
where
    T: Scalar + VecLimits + VecFloat,
{
    if b.is_empty() {
        return false;
    }

    let tmax: T = T::MAX;

    let mut t_front_max: T = -tmax;
    let mut t_back_min: T = tmax;

    // --- X slabs ---
    if r.dir.x >= T::ZERO {
        let d1 = b.max.x - r.pos.x;
        let d2 = b.min.x - r.pos.x;

        if r.dir.x > T::ONE
            || (abs(d1) < tmax * r.dir.x && abs(d2) < tmax * r.dir.x)
        {
            let t1 = d1 / r.dir.x;
            let t2 = d2 / r.dir.x;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = b.max.x;
                exit.y = clamp(r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
                exit.z = clamp(r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = b.min.x;
                entry.y = clamp(r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
                entry.z = clamp(r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
            }
        } else if r.pos.x < b.min.x || r.pos.x > b.max.x {
            return false;
        }
    } else {
        // r.dir.x < 0
        let d1 = b.min.x - r.pos.x;
        let d2 = b.max.x - r.pos.x;

        if r.dir.x < -T::ONE
            || (abs(d1) < -tmax * r.dir.x && abs(d2) < -tmax * r.dir.x)
        {
            let t1 = d1 / r.dir.x;
            let t2 = d2 / r.dir.x;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = b.min.x;
                exit.y = clamp(r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
                exit.z = clamp(r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = b.max.x;
                entry.y = clamp(r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
                entry.z = clamp(r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
            }
        } else if r.pos.x < b.min.x || r.pos.x > b.max.x {
            return false;
        }
    }

    // --- Y slabs ---
    if r.dir.y >= T::ZERO {
        let d1 = b.max.y - r.pos.y;
        let d2 = b.min.y - r.pos.y;

        if r.dir.y > T::ONE
            || (abs(d1) < tmax * r.dir.y && abs(d2) < tmax * r.dir.y)
        {
            let t1 = d1 / r.dir.y;
            let t2 = d2 / r.dir.y;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = clamp(r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
                exit.y = b.max.y;
                exit.z = clamp(r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = clamp(r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
                entry.y = b.min.y;
                entry.z = clamp(r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
            }
        } else if r.pos.y < b.min.y || r.pos.y > b.max.y {
            return false;
        }
    } else {
        // r.dir.y < 0
        let d1 = b.min.y - r.pos.y;
        let d2 = b.max.y - r.pos.y;

        if r.dir.y < -T::ONE
            || (abs(d1) < -tmax * r.dir.y && abs(d2) < -tmax * r.dir.y)
        {
            let t1 = d1 / r.dir.y;
            let t2 = d2 / r.dir.y;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = clamp(r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
                exit.y = b.min.y;
                exit.z = clamp(r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = clamp(r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
                entry.y = b.max.y;
                entry.z = clamp(r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
            }
        } else if r.pos.y < b.min.y || r.pos.y > b.max.y {
            return false;
        }
    }

    // --- Z slabs ---
    if r.dir.z >= T::ZERO {
        let d1 = b.max.z - r.pos.z;
        let d2 = b.min.z - r.pos.z;

        if r.dir.z > T::ONE
            || (abs(d1) < tmax * r.dir.z && abs(d2) < tmax * r.dir.z)
        {
            let t1 = d1 / r.dir.z;
            let t2 = d2 / r.dir.z;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = clamp(r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
                exit.y = clamp(r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
                exit.z = b.max.z;
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = clamp(r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
                entry.y = clamp(r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
                entry.z = b.min.z;
            }
        } else if r.pos.z < b.min.z || r.pos.z > b.max.z {
            return false;
        }
    } else {
        // r.dir.z < 0
        let d1 = b.min.z - r.pos.z;
        let d2 = b.max.z - r.pos.z;

        if r.dir.z < -T::ONE
            || (abs(d1) < -tmax * r.dir.z && abs(d2) < -tmax * r.dir.z)
        {
            let t1 = d1 / r.dir.z;
            let t2 = d2 / r.dir.z;

            if t_back_min > t1 {
                t_back_min = t1;
                exit.x = clamp(r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
                exit.y = clamp(r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
                exit.z = b.min.z;
            }

            if t_front_max < t2 {
                t_front_max = t2;
                entry.x = clamp(r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
                entry.y = clamp(r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
                entry.z = b.max.z;
            }
        } else if r.pos.z < b.min.z || r.pos.z > b.max.z {
            return false;
        }
    }

    t_front_max <= t_back_min
}

// ---------------------------------------------------------------------------
// intersects — bool + intersection-point overload
// ---------------------------------------------------------------------------

/// Port of `bool intersects(const Box<Vec3<T>>& b, const Line3<T>& r, Vec3<T>& ip)`.
///
/// Tests whether ray `r` intersects box `b`.  Returns `true` and sets `ip` to:
/// - `r.pos` if the ray starts inside the box, or
/// - the entry point on a box face if the ray starts outside and hits the box.
///
/// Returns `false` if the ray starts outside and either misses the box entirely
/// or would hit it only behind the ray origin (tFrontMax < 0 || > tBackMin).
#[inline]
pub fn intersects_with_point<T>(b: &Box<Vec3<T>>, r: &Line3<T>, ip: &mut Vec3<T>) -> bool
where
    T: Scalar + VecLimits + VecFloat,
{
    if b.is_empty() {
        return false;
    }

    // Ray origin is inside: intersection point is the origin itself.
    if b.intersects_point(r.pos) {
        *ip = r.pos;
        return true;
    }

    let tmax: T = T::MAX;
    // tFrontMax starts at -1 (not -TMAX) — matches C++ exactly.
    let neg_one: T = -(T::ONE);
    let mut t_front_max: T = neg_one;
    let mut t_back_min: T = tmax;

    // --- X slabs ---
    if r.dir.x > T::ZERO {
        if r.pos.x > b.max.x {
            return false;
        }

        let d = b.max.x - r.pos.x;
        if r.dir.x > T::ONE || d < tmax * r.dir.x {
            let t = d / r.dir.x;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.x <= b.min.x {
            let d2 = b.min.x - r.pos.x;
            let t = if r.dir.x > T::ONE || d2 < tmax * r.dir.x {
                d2 / r.dir.x
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = b.min.x;
                ip.y = clamp(r.pos.y + t * r.dir.y, b.min.y, b.max.y);
                ip.z = clamp(r.pos.z + t * r.dir.z, b.min.z, b.max.z);
            }
        }
    } else if r.dir.x < T::ZERO {
        if r.pos.x < b.min.x {
            return false;
        }

        let d = b.min.x - r.pos.x;
        if r.dir.x < -T::ONE || d > tmax * r.dir.x {
            let t = d / r.dir.x;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.x >= b.max.x {
            let d2 = b.max.x - r.pos.x;
            let t = if r.dir.x < -T::ONE || d2 > tmax * r.dir.x {
                d2 / r.dir.x
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = b.max.x;
                ip.y = clamp(r.pos.y + t * r.dir.y, b.min.y, b.max.y);
                ip.z = clamp(r.pos.z + t * r.dir.z, b.min.z, b.max.z);
            }
        }
    } else {
        // r.dir.x == 0
        if r.pos.x < b.min.x || r.pos.x > b.max.x {
            return false;
        }
    }

    // --- Y slabs ---
    if r.dir.y > T::ZERO {
        if r.pos.y > b.max.y {
            return false;
        }

        let d = b.max.y - r.pos.y;
        if r.dir.y > T::ONE || d < tmax * r.dir.y {
            let t = d / r.dir.y;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.y <= b.min.y {
            let d2 = b.min.y - r.pos.y;
            let t = if r.dir.y > T::ONE || d2 < tmax * r.dir.y {
                d2 / r.dir.y
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = clamp(r.pos.x + t * r.dir.x, b.min.x, b.max.x);
                ip.y = b.min.y;
                ip.z = clamp(r.pos.z + t * r.dir.z, b.min.z, b.max.z);
            }
        }
    } else if r.dir.y < T::ZERO {
        if r.pos.y < b.min.y {
            return false;
        }

        let d = b.min.y - r.pos.y;
        if r.dir.y < -T::ONE || d > tmax * r.dir.y {
            let t = d / r.dir.y;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.y >= b.max.y {
            let d2 = b.max.y - r.pos.y;
            let t = if r.dir.y < -T::ONE || d2 > tmax * r.dir.y {
                d2 / r.dir.y
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = clamp(r.pos.x + t * r.dir.x, b.min.x, b.max.x);
                ip.y = b.max.y;
                ip.z = clamp(r.pos.z + t * r.dir.z, b.min.z, b.max.z);
            }
        }
    } else {
        // r.dir.y == 0
        if r.pos.y < b.min.y || r.pos.y > b.max.y {
            return false;
        }
    }

    // --- Z slabs ---
    if r.dir.z > T::ZERO {
        if r.pos.z > b.max.z {
            return false;
        }

        let d = b.max.z - r.pos.z;
        if r.dir.z > T::ONE || d < tmax * r.dir.z {
            let t = d / r.dir.z;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.z <= b.min.z {
            let d2 = b.min.z - r.pos.z;
            let t = if r.dir.z > T::ONE || d2 < tmax * r.dir.z {
                d2 / r.dir.z
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = clamp(r.pos.x + t * r.dir.x, b.min.x, b.max.x);
                ip.y = clamp(r.pos.y + t * r.dir.y, b.min.y, b.max.y);
                ip.z = b.min.z;
            }
        }
    } else if r.dir.z < T::ZERO {
        if r.pos.z < b.min.z {
            return false;
        }

        let d = b.min.z - r.pos.z;
        if r.dir.z < -T::ONE || d > tmax * r.dir.z {
            let t = d / r.dir.z;
            if t_back_min > t {
                t_back_min = t;
            }
        }

        if r.pos.z >= b.max.z {
            let d2 = b.max.z - r.pos.z;
            let t = if r.dir.z < -T::ONE || d2 > tmax * r.dir.z {
                d2 / r.dir.z
            } else {
                tmax
            };

            if t_front_max < t {
                t_front_max = t;
                ip.x = clamp(r.pos.x + t * r.dir.x, b.min.x, b.max.x);
                ip.y = clamp(r.pos.y + t * r.dir.y, b.min.y, b.max.y);
                ip.z = b.max.z;
            }
        }
    } else {
        // r.dir.z == 0
        if r.pos.z < b.min.z || r.pos.z > b.max.z {
            return false;
        }
    }

    t_front_max <= t_back_min
}

// ---------------------------------------------------------------------------
// intersects — bool-only overload
// ---------------------------------------------------------------------------

/// Port of `bool intersects(const Box<Vec3<T>>& box, const Line3<T>& ray)`.
///
/// Convenience wrapper: returns whether the ray intersects the box (point ignored).
#[inline]
pub fn intersects<T>(b: &Box<Vec3<T>>, r: &Line3<T>) -> bool
where
    T: Scalar + VecLimits + VecFloat,
{
    let mut ignored = Vec3 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
    };
    intersects_with_point(b, r, &mut ignored)
}

// ---------------------------------------------------------------------------
// Internal helper: build the 8 corners of a Box<Vec3<T>>
// ---------------------------------------------------------------------------

/// Builds the 8 corners of the AABB, matching the C++ `points[8]` layout in
/// `transform` and `affineTransform` (non-affine fallback path).
///
/// C++ layout (indices by axis):
///   [0..3] → min.x,  [4..7] → max.x
///   [0,1,4,5] → min.y,  [2,3,6,7] → max.y
///   [0,2,4,6] → min.z,  [1,3,5,7] → max.z
#[inline]
fn eight_corners<T: Scalar + VecLimits>(b: &Box<Vec3<T>>) -> [Vec3<T>; 8] {
    let (mn, mx) = (b.min, b.max);
    [
        Vec3::new(mn.x, mn.y, mn.z), // 0
        Vec3::new(mn.x, mn.y, mx.z), // 1
        Vec3::new(mn.x, mx.y, mn.z), // 2
        Vec3::new(mn.x, mx.y, mx.z), // 3
        Vec3::new(mx.x, mn.y, mn.z), // 4
        Vec3::new(mx.x, mn.y, mx.z), // 5
        Vec3::new(mx.x, mx.y, mn.z), // 6
        Vec3::new(mx.x, mx.y, mx.z), // 7
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_::Box3f;
    use crate::line::Line3f;
    use crate::matrix::Matrix44;
    use crate::vec::{Vec3, V3f};

    // --- clip / closestPointInBox ---

    #[test]
    fn clip_inside_is_identity() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let p = Vec3::new(1.0f32, 1.0, 1.0);
        assert_eq!(clip(p, &b), p);
        assert_eq!(closest_point_in_box(p, &b), p);
    }

    #[test]
    fn clip_outside_clamps_to_surface() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        // point beyond max in x and y, below min in z
        let p = Vec3::new(5.0f32, 3.0, -1.0);
        let q = clip(p, &b);
        assert_eq!(q, Vec3::new(2.0f32, 2.0, 0.0));
    }

    // --- closestPointOnBox ---

    #[test]
    fn closest_point_on_box_outside() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        // point outside only in x
        let p = Vec3::new(5.0f32, 1.0, 1.0);
        let q = closest_point_on_box(p, &b);
        // clamped in x, unchanged in y/z — which is already a face point
        assert!((q.x - 2.0f32).abs() < 1e-6);
        assert!((q.y - 1.0f32).abs() < 1e-6);
        assert!((q.z - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn closest_point_on_box_inside_projects_to_nearest_face() {
        // unit cube [0,1]^3; inside point closer to min.x face
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        // p=(0.1, 0.5, 0.5) → d1.x=0.1, d2.x=0.9 → dx=0.1 (minimum)
        let p = Vec3::new(0.1f32, 0.5, 0.5);
        let q = closest_point_on_box(p, &b);
        // should be projected to x=0 face
        assert!((q.x - 0.0f32).abs() < 1e-6);
        assert!((q.y - 0.5f32).abs() < 1e-6);
        assert!((q.z - 0.5f32).abs() < 1e-6);
    }

    #[test]
    fn closest_point_on_box_empty_returns_p() {
        let b = Box3f::new(); // empty
        let p = Vec3::new(1.0f32, 2.0, 3.0);
        assert_eq!(closest_point_on_box(p, &b), p);
    }

    // --- transform ---

    #[test]
    fn transform_identity_is_unchanged() {
        let b = Box3f::from_min_max(Vec3::new(1.0f32, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let m = Matrix44::<f32>::default(); // identity
        let tb = transform(&b, &m);
        // identity transform should return same box
        assert!((tb.min.x - 1.0).abs() < 1e-5);
        assert!((tb.max.z - 6.0).abs() < 1e-5);
    }

    #[test]
    fn transform_translation() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let mut m = Matrix44::<f32>::default();
        m.translate(&Vec3::new(3.0f32, 0.0, 0.0));
        let tb = transform(&b, &m);
        assert!((tb.min.x - 3.0).abs() < 1e-5);
        assert!((tb.max.x - 4.0).abs() < 1e-5);
    }

    #[test]
    fn transform_rotation_expands_aabb() {
        // A unit cube rotated 45° around Z should have a larger AABB
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let mut m = Matrix44::<f32>::default();
        let angle = core::f32::consts::FRAC_PI_4; // 45°
        // set_euler_angles rotates around Z when only z is set
        m.set_euler_angles(&Vec3::new(0.0f32, 0.0, angle));
        let tb = transform(&b, &m);
        // The AABB of a square with half-extent 1 rotated by `angle` has new
        // half-extent |cos(angle)| + |sin(angle)| (each corner's rotated x/y
        // is +-cos +-sin, and the AABB takes the max magnitude over all 4
        // sign combinations). At 45 degrees cos == sin == sqrt(2)/2, so the
        // half-extent is exactly sqrt(2) -- this is the independent
        // closed-form expectation the old `expected` variable computed
        // WRONG (and then never checked, `let _ = expected;`).
        let half_extent = angle.cos().abs() + angle.sin().abs();
        assert!(
            (tb.max.x - half_extent).abs() < 1e-5,
            "max.x={} should be sqrt(2)={half_extent} after a 45-degree rotation",
            tb.max.x
        );
        assert!(
            (tb.min.x + half_extent).abs() < 1e-5,
            "min.x={} should be -sqrt(2)=-{half_extent} (symmetric AABB)",
            tb.min.x
        );
        // Rotation is about Z only, so the Z extent is unchanged.
        assert!((tb.max.z - 1.0).abs() < 1e-5, "max.z={} should stay 1 (Z-only rotation)", tb.max.z);
        assert!((tb.min.z + 1.0).abs() < 1e-5, "min.z={} should stay -1 (Z-only rotation)", tb.min.z);
    }

    #[test]
    fn transform_into_matches_returning() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 3.0, 4.0));
        let mut m = Matrix44::<f32>::default();
        m.translate(&Vec3::new(1.0f32, 2.0, 3.0));
        let r1 = transform(&b, &m);
        let mut r2 = Box3f::new();
        transform_into(&b, &m, &mut r2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn affine_transform_agrees_with_transform_for_affine_matrix() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0));
        let mut m = Matrix44::<f32>::default();
        m.translate(&Vec3::new(10.0f32, 0.0, 0.0));
        let t = transform(&b, &m);
        let a = affine_transform(&b, &m);
        assert_eq!(t, a);
    }

    #[test]
    fn affine_transform_into_matches_returning() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let m = Matrix44::<f32>::default();
        let r1 = affine_transform(&b, &m);
        let mut r2 = Box3f::new();
        affine_transform_into(&b, &m, &mut r2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn transform_empty_box_unchanged() {
        let b: Box3f = Box3f::new(); // empty
        let m = Matrix44::<f32>::default();
        let tb = transform(&b, &m);
        assert!(tb.is_empty());
    }

    #[test]
    fn affine_transform_into_empty_makes_result_empty() {
        let b: Box3f = Box3f::new();
        let m = Matrix44::<f32>::default();
        let mut result = Box3f::from_point(Vec3::new(1.0f32, 1.0, 1.0));
        affine_transform_into(&b, &m, &mut result);
        assert!(result.is_empty());
    }

    // --- findEntryAndExitPoints ---

    #[test]
    fn find_entry_exit_ray_along_x_axis() {
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Ray from (-5, 0, 0) pointing in +X
        let ray = Line3f::from_points(Vec3::new(-5.0f32, 0.0, 0.0), Vec3::new(-4.0, 0.0, 0.0));
        let mut entry = Vec3::new(0.0f32, 0.0, 0.0);
        let mut exit = Vec3::new(0.0f32, 0.0, 0.0);
        let hit = find_entry_and_exit_points(&ray, &b, &mut entry, &mut exit);
        assert!(hit);
        assert!((entry.x - (-1.0f32)).abs() < 1e-5);
        assert!((exit.x - 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn find_entry_exit_ray_misses() {
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Ray pointing in +X but above the box in Y
        let ray = Line3f::from_points(Vec3::new(-5.0f32, 5.0, 0.0), Vec3::new(-4.0, 5.0, 0.0));
        let mut entry = Vec3::new(0.0f32, 0.0, 0.0);
        let mut exit = Vec3::new(0.0f32, 0.0, 0.0);
        let hit = find_entry_and_exit_points(&ray, &b, &mut entry, &mut exit);
        assert!(!hit);
    }

    // --- intersects ---

    #[test]
    fn intersects_ray_hits() {
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Line3f::from_points(Vec3::new(-5.0f32, 0.0, 0.0), Vec3::new(-4.0, 0.0, 0.0));
        let mut ip: V3f = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(intersects_with_point(&b, &ray, &mut ip));
        // intersection on the -x face
        assert!((ip.x - (-1.0f32)).abs() < 1e-5);
    }

    #[test]
    fn intersects_ray_inside_returns_origin() {
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Line3f::from_points(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let mut ip: V3f = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(intersects_with_point(&b, &ray, &mut ip));
        // origin is inside, so ip == ray.pos
        assert!((ip.x).abs() < 1e-6);
        assert!((ip.y).abs() < 1e-6);
        assert!((ip.z).abs() < 1e-6);
    }

    #[test]
    fn intersects_ray_misses() {
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Line3f::from_points(Vec3::new(-5.0f32, 5.0, 0.0), Vec3::new(-4.0, 5.0, 0.0));
        let mut ip: V3f = Vec3::new(0.0f32, 0.0, 0.0);
        assert!(!intersects_with_point(&b, &ray, &mut ip));
        assert!(!intersects(&b, &ray));
    }

    #[test]
    fn intersects_bool_only_matches_with_point() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let ray_hit = Line3f::from_points(Vec3::new(-5.0f32, 1.0, 1.0), Vec3::new(-4.0, 1.0, 1.0));
        let ray_miss = Line3f::from_points(Vec3::new(-5.0f32, 5.0, 5.0), Vec3::new(-4.0, 5.0, 5.0));
        assert!(intersects(&b, &ray_hit));
        assert!(!intersects(&b, &ray_miss));
    }
}
