//! Port of Imath ImathFrame.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathFrame.h>
//!
//! Functions for computing Gram-Schmidt parallel-transport reference frames
//! along a curve.  A typical usage (from the C++ docs):
//!
//! ```text
//! m[0] = first_frame(p[0], p[1], p[2]);
//! for i in 1..n-1 {
//!     m[i] = next_frame(&m[i-1], p[i-1], p[i], &mut t[i-1], &mut t[i]);
//! }
//! m[n-1] = last_frame(&m[n-2], p[n-2], p[n-1]);
//! ```
//!
//! See *Graphics Gems I* for the underlying algorithm.
//!
//! # Exception mapping
//! The C++ `firstFrame` calls `t.normalizeExc()` which throws
//! `std::domain_error` when `pi == pj`.  We return `Result<Matrix44<T>, &'static str>`
//! instead (the `_exc` convention used across this crate).

use crate::math::Math;
use crate::matrix::Matrix44;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};

/// Port of `firstFrame(pi, pj, pk)` — compute the first reference frame along
/// a curve defined by three points.
///
/// Returns `Err` (C++ `throw std::domain_error`) when `pi == pj`
/// (i.e. `(pj - pi)` has zero length and cannot be normalised).
///
/// If the vectors `<pi,pj>` and `<pi,pk>` are collinear, an arbitrary twist
/// is chosen (matching C++ behaviour).
pub fn first_frame<T>(pi: Vec3<T>, pj: Vec3<T>, pk: Vec3<T>) -> Result<Matrix44<T>, &'static str>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    // t = normalise(pj - pi); throws if pi == pj
    let mut t = pj - pi;
    t.normalize_exc()?; // returns Err if zero-length

    // n = normalise(t × (pk - pi)); if collinear, pick an arbitrary axis
    let mut n = t.cross(&(pk - pi));
    n.normalize();
    if n.length() == T::ZERO {
        // Collinear: pick a basis vector not parallel to t
        let mut v = Vec3::from_scalar(T::ZERO);
        // Find the component of t with smallest absolute value and set it to 1
        let abs_x = if t.x >= T::ZERO { t.x } else { T::ZERO - t.x };
        let abs_y = if t.y >= T::ZERO { t.y } else { T::ZERO - t.y };
        let abs_z = if t.z >= T::ZERO { t.z } else { T::ZERO - t.z };
        // i = index of smallest absolute component
        let i = if abs_x < abs_y {
            if abs_z < abs_x { 2 } else { 0 }
        } else {
            if abs_z < abs_y { 2 } else { 1 }
        };
        v[i] = T::ONE;
        n = t.cross(&v);
        n.normalize();
    }

    // b = t × n
    let b = t.cross(&n);

    // Build the row-major Matrix44 (matching C++ M[0]=t, M[1]=n, M[2]=b, M[3]=pi)
    Ok(Matrix44::from_array([
        [t.x,  t.y,  t.z,  T::ZERO],
        [n.x,  n.y,  n.z,  T::ZERO],
        [b.x,  b.y,  b.z,  T::ZERO],
        [pi.x, pi.y, pi.z, T::ONE ],
    ]))
}

/// Port of `nextFrame(Mi, pi, pj, ti, tj)` — parallel-transport the frame
/// from `pi` to `pj` using the rotation that maps tangent `ti` to `tj`.
///
/// `ti` and `tj` are passed as `&mut` because the C++ function normalises them
/// in place (matching the C++ `Vec3<T>& ti, Vec3<T>& tj` signature).
pub fn next_frame<T>(
    mi: &Matrix44<T>,
    pi: Vec3<T>,
    pj: Vec3<T>,
    ti: &mut Vec3<T>,
    tj: &mut Vec3<T>,
) -> Matrix44<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    let mut a = Vec3::from_scalar(T::ZERO); // rotation axis
    let mut r = T::ZERO;                    // rotation angle

    if ti.length() != T::ZERO && tj.length() != T::ZERO {
        ti.normalize();
        tj.normalize();
        let mut dot = ti.dot(tj);

        // Clamp to [-1, 1] — "this is *really* necessary" (C++ comment)
        if dot > T::ONE  { dot = T::ONE; }
        else if dot < (T::ZERO - T::ONE) { dot = T::ZERO - T::ONE; }

        r = dot.acos();
        a = ti.cross(tj);
    }

    if a.length() != T::ZERO && r != T::ZERO {
        // R = rotation by r about a; Tj = translate(+pj); Ti = translate(-pi)
        let mut rot = Matrix44::<T>::default();
        rot.set_axis_angle(&a, r);

        let mut tj_mat = Matrix44::<T>::default();
        tj_mat.translate(&pj);

        let mut ti_mat = Matrix44::<T>::default();
        let neg_pi = Vec3::new(T::ZERO - pi.x, T::ZERO - pi.y, T::ZERO - pi.z);
        ti_mat.translate(&neg_pi);

        // Mi * Ti * R * Tj
        let mi_ti  = Matrix44::multiply(mi, &ti_mat);
        let mi_ti_r = Matrix44::multiply(&mi_ti, &rot);
        Matrix44::multiply(&mi_ti_r, &tj_mat)
    } else {
        // Pure translation: Mi * Tr  where Tr = translate(pj - pi)
        let mut tr = Matrix44::<T>::default();
        tr.translate(&(pj - pi));
        Matrix44::multiply(mi, &tr)
    }
}

/// Port of `lastFrame(Mi, pi, pj)` — translate the previous frame to the last
/// point.
pub fn last_frame<T>(mi: &Matrix44<T>, pi: Vec3<T>, pj: Vec3<T>) -> Matrix44<T>
where
    T: Scalar + VecFloat + VecLimits + Math,
{
    let mut tr = Matrix44::<T>::default();
    tr.translate(&(pj - pi));
    Matrix44::multiply(mi, &tr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec3;
    use crate::matrix::Matrix44;

    /// Check that a 3x3 sub-block of a Matrix44 is orthonormal (columns are
    /// unit length and mutually perpendicular).
    fn is_orthonormal_f32(m: &Matrix44<f32>, tol: f32) -> bool {
        // Rows 0, 1, 2 are the tangent / normal / binormal vectors
        let t = Vec3::new(m.x[0][0], m.x[0][1], m.x[0][2]);
        let n = Vec3::new(m.x[1][0], m.x[1][1], m.x[1][2]);
        let b = Vec3::new(m.x[2][0], m.x[2][1], m.x[2][2]);

        let len_t = (t.x*t.x + t.y*t.y + t.z*t.z).sqrt();
        let len_n = (n.x*n.x + n.y*n.y + n.z*n.z).sqrt();
        let len_b = (b.x*b.x + b.y*b.y + b.z*b.z).sqrt();

        let dot_tn = t.x*n.x + t.y*n.y + t.z*n.z;
        let dot_tb = t.x*b.x + t.y*b.y + t.z*b.z;
        let dot_nb = n.x*b.x + n.y*b.y + n.z*b.z;

        (len_t - 1.0).abs() < tol
            && (len_n - 1.0).abs() < tol
            && (len_b - 1.0).abs() < tol
            && dot_tn.abs() < tol
            && dot_tb.abs() < tol
            && dot_nb.abs() < tol
    }

    #[test]
    fn first_frame_orthonormal() {
        let p0 = Vec3::new(0.0f32, 0.0, 0.0);
        let p1 = Vec3::new(1.0f32, 0.0, 0.0);
        let p2 = Vec3::new(1.0f32, 1.0, 0.0);
        let m = first_frame(p0, p1, p2).expect("first_frame should succeed");
        assert!(is_orthonormal_f32(&m, 1e-5));
        // Translation part should be pi = (0,0,0)
        assert!((m.x[3][0]).abs() < 1e-6);
        assert!((m.x[3][1]).abs() < 1e-6);
        assert!((m.x[3][2]).abs() < 1e-6);
    }

    #[test]
    fn first_frame_error_on_equal_points() {
        let p = Vec3::new(1.0f32, 2.0, 3.0);
        let result = first_frame(p, p, Vec3::new(0.0, 0.0, 0.0));
        assert!(result.is_err());
    }

    #[test]
    fn next_frame_orthonormal() {
        let p0 = Vec3::new(0.0f32, 0.0, 0.0);
        let p1 = Vec3::new(1.0f32, 0.0, 0.0);
        let p2 = Vec3::new(2.0f32, 1.0, 0.0);
        let p3 = Vec3::new(3.0f32, 0.0, 0.0);

        let m0 = first_frame(p0, p1, p2).expect("first_frame");
        let mut t0 = p1 - p0;
        let mut t1 = p2 - p1;
        let m1 = next_frame(&m0, p1, p2, &mut t0, &mut t1);

        assert!(is_orthonormal_f32(&m1, 1e-4));

        let mut t2 = p3 - p2;
        let m2 = next_frame(&m1, p2, p3, &mut t1, &mut t2);
        assert!(is_orthonormal_f32(&m2, 1e-4));
    }

    #[test]
    fn last_frame_translates_correctly() {
        let p0 = Vec3::new(0.0f32, 0.0, 0.0);
        let p1 = Vec3::new(1.0f32, 0.0, 0.0);
        let p2 = Vec3::new(2.0f32, 0.0, 0.0);
        let p3 = Vec3::new(5.0f32, 0.0, 0.0);

        let m0 = first_frame(p0, p1, p2).expect("first_frame");
        let m_last = last_frame(&m0, p1, p3);
        // Translation row should be at p3 offset from original
        // The last row's x should have moved by (p3-p1).x = 4 from the original 0
        assert!((m_last.x[3][0] - (m0.x[3][0] + 4.0)).abs() < 1e-4);
    }

    #[test]
    fn first_frame_collinear_fallback() {
        // Collinear three points — should still produce an orthonormal frame
        let p0 = Vec3::new(0.0f32, 0.0, 0.0);
        let p1 = Vec3::new(1.0f32, 0.0, 0.0);
        let p2 = Vec3::new(2.0f32, 0.0, 0.0); // collinear
        let m = first_frame(p0, p1, p2).expect("first_frame collinear");
        assert!(is_orthonormal_f32(&m, 1e-4));
    }
}
