//! Port of Imath ImathMatrixAlgo.h / .cpp.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrixAlgo.h>
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrixAlgo.cpp>
//!
//! Free functions operating on `Matrix22`/`Matrix33`/`Matrix44`: scale/shear/
//! rotation/translation extraction (`extractSHRT` family), Euler extraction,
//! `rotationMatrix*`, `alignZAxisWithTargetDir`, `computeLocalFrame`, `addOffset`,
//! `computeRSMatrix`, `outerProduct`, the Procrustes solver, and the two-sided
//! Jacobi SVD / symmetric Jacobi eigen-solver.
//!
//! # Fidelity notes
//! - Every function mirrors the C++ algorithm exactly: same sweeps, the same
//!   convergence thresholds (`maxIter = 20`, `absTol = tol * maxOffDiag`), the same
//!   sign/flip conventions, and the same epsilons.
//! - The C++ exposes a runtime `bool exc` on the singular-matrix functions: when
//!   `exc` is true it throws `std::domain_error`; when false it returns a
//!   "failed" sentinel. Rust has no exceptions, so these functions return
//!   [`Result`]: `Err(&'static str)` is the C++ `throw` path (taken only when
//!   `exc == true`), and `Ok(false)` / `Ok(mat)` is the non-throwing degenerate
//!   path (taken when `exc == false`). The boolean `exc` parameter is preserved
//!   verbatim so callers reproduce C++ control flow 1:1. Functions whose C++ form
//!   always throws on degeneracy (`computeRSMatrix`) return `Result` with no `exc`
//!   flag.
//! - `procrustesRotationAndTranslation`, `jacobiSVD`, `jacobiEigenSolver`,
//!   `maxEigenVector`, `minEigenVector` are explicitly instantiated for `float`/
//!   `double` only in the C++ `.cpp`; they are ported as generics bound to the
//!   float element traits (`VecFloat` etc.). Procrustes always accumulates in
//!   `double` and returns `M44d`, exactly like the C++.

use crate::euler::{Euler, EulerFloat, InputLayout, Order};
use crate::fun::abs;
use crate::math::Math;
use crate::matrix::{Matrix33, Matrix44};
use crate::platform::M_PI;
use crate::quat::{Quat, QuatFloat};
use crate::traits::Scalar;
use crate::vec::{Vec2, Vec3, Vec4, VecCast, VecFloat, VecLimits};

// ============================================================================
// Identity matrices (port of the EXPORT_CONST module constants in the .cpp).
// ============================================================================

use crate::matrix::{M22d, M22f, M33d, M33f, M44d, M44f};

/// Port of `M22f identity22f`.
pub const IDENTITY22F: M22f = identity22f();
/// Port of `M22d identity22d`.
pub const IDENTITY22D: M22d = identity22d();
/// Port of `M33f identity33f`.
pub const IDENTITY33F: M33f = M33f::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
/// Port of `M33d identity33d`.
pub const IDENTITY33D: M33d = M33d::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
/// Port of `M44f identity44f`.
pub const IDENTITY44F: M44f = M44f::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
);
/// Port of `M44d identity44d`.
pub const IDENTITY44D: M44d = M44d::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
);

// `Matrix22` has no `const fn new` taking four scalars exposed under that exact
// name here, so build the 2x2 identities via `from_array` in a const helper.
use crate::matrix::Matrix22;

const fn identity22f() -> M22f {
    Matrix22 {
        x: [[1.0, 0.0], [0.0, 1.0]],
    }
}
const fn identity22d() -> M22d {
    Matrix22 {
        x: [[1.0, 0.0], [0.0, 1.0]],
    }
}

// ============================================================================
// Implementation for 4x4 Matrix
// ============================================================================

/// Port of `bool extractScaling (const Matrix44<T>& mat, Vec3<T>& scl, bool exc)`.
///
/// Returns `Ok(true)` with `scl` filled on success, `Ok(false)` on the
/// non-throwing degenerate path (`exc == false`), or `Err` on the throwing path.
pub fn extract_scaling_m44<T>(
    mat: &Matrix44<T>,
    scl: &mut Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m44(&mut m, scl, &mut shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `Matrix44<T> sansScaling (const Matrix44<T>& mat, bool exc)`.
pub fn sans_scaling_m44<T>(mat: &Matrix44<T>, exc: bool) -> Result<Matrix44<T>, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    let mut scl = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut rot = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut tran = Vec3::new(T::ZERO, T::ZERO, T::ZERO);

    if !extract_shrt_m44_vec(mat, &mut scl, &mut shr, &mut rot, &mut tran, exc)? {
        return Ok(*mat);
    }

    let mut m = Matrix44::default();
    m.translate(&tran);
    m.rotate(&rot);
    m.shear_vec(&shr);

    Ok(m)
}

/// Port of `bool removeScaling (Matrix44<T>& mat, bool exc)` — operates in place.
pub fn remove_scaling_m44<T>(mat: &mut Matrix44<T>, exc: bool) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    let mut scl = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut rot = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut tran = Vec3::new(T::ZERO, T::ZERO, T::ZERO);

    if !extract_shrt_m44_vec(mat, &mut scl, &mut shr, &mut rot, &mut tran, exc)? {
        return Ok(false);
    }

    mat.make_identity();
    mat.translate(&tran);
    mat.rotate(&rot);
    mat.shear_vec(&shr);

    Ok(true)
}

/// Port of `bool extractScalingAndShear (const Matrix44<T>& mat, Vec3<T>& scl,
/// Vec3<T>& shr, bool exc)`.
pub fn extract_scaling_and_shear_m44<T>(
    mat: &Matrix44<T>,
    scl: &mut Vec3<T>,
    shr: &mut Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m44(&mut m, scl, shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `Matrix44<T> sansScalingAndShear (const Matrix44<T>& mat, bool exc)`.
pub fn sans_scaling_and_shear_m44<T>(
    mat: &Matrix44<T>,
    exc: bool,
) -> Result<Matrix44<T>, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut scl = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m44(&mut m, &mut scl, &mut shr, exc)? {
        return Ok(*mat);
    }

    Ok(m)
}

/// Port of `void sansScalingAndShear (Matrix44<T>& result, const Matrix44<T>& mat,
/// bool exc)` — the in-place-into-`result` overload.
pub fn sans_scaling_and_shear_into_m44<T>(
    result: &mut Matrix44<T>,
    mat: &Matrix44<T>,
    exc: bool,
) -> Result<(), &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut scl = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);

    if !extract_and_remove_scaling_and_shear_m44(result, &mut scl, &mut shr, exc)? {
        *result = *mat;
    }

    Ok(())
}

/// Port of `bool removeScalingAndShear (Matrix44<T>& mat, bool exc)`.
pub fn remove_scaling_and_shear_m44<T>(
    mat: &mut Matrix44<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut scl = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut shr = Vec3::new(T::ZERO, T::ZERO, T::ZERO);

    if !extract_and_remove_scaling_and_shear_m44(mat, &mut scl, &mut shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `bool extractAndRemoveScalingAndShear (Matrix44<T>& mat, Vec3<T>& scl,
/// Vec3<T>& shr, bool exc)`.
///
/// Follows Spencer W. Thomas's "Decomposing a Matrix into Simple Transformations"
/// (Graphics Gems II, p. 320) — transcribed line-for-line from the C++.
pub fn extract_and_remove_scaling_and_shear_m44<T>(
    mat: &mut Matrix44<T>,
    scl: &mut Vec3<T>,
    shr: &mut Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut row = [
        Vec3::new(mat[0][0], mat[0][1], mat[0][2]),
        Vec3::new(mat[1][0], mat[1][1], mat[1][2]),
        Vec3::new(mat[2][0], mat[2][1], mat[2][2]),
    ];

    let mut max_val = T::ZERO;
    for r in &row {
        for j in 0..3 {
            if abs(r[j]) > max_val {
                max_val = abs(r[j]);
            }
        }
    }

    // Normalize the 3x3 to improve numerical stability; the scaling factors are
    // multiplied back by `max_val` at the end (shear/rotation are unaffected).
    if max_val != T::ZERO {
        for r in &mut row {
            if !check_for_zero_scale_in_row_v3(max_val, r, exc)? {
                return Ok(false);
            }
            *r = div_v3(*r, max_val);
        }
    }

    // Compute X scale factor.
    scl.x = row[0].length();
    if !check_for_zero_scale_in_row_v3(scl.x, &row[0], exc)? {
        return Ok(false);
    }

    // Normalize first row.
    row[0] = div_v3(row[0], scl.x);

    // Compute XY shear factor and make 2nd row orthogonal to 1st.
    shr[0] = row[0].dot(&row[1]);
    row[1] = sub_v3(row[1], mul_v3(row[0], shr[0]));

    // Now, compute Y scale.
    scl.y = row[1].length();
    if !check_for_zero_scale_in_row_v3(scl.y, &row[1], exc)? {
        return Ok(false);
    }

    // Normalize 2nd row and correct the XY shear factor for Y scaling.
    row[1] = div_v3(row[1], scl.y);
    shr[0] = shr[0] / scl.y;

    // Compute XZ and YZ shears, orthogonalize 3rd row.
    shr[1] = row[0].dot(&row[2]);
    row[2] = sub_v3(row[2], mul_v3(row[0], shr[1]));
    shr[2] = row[1].dot(&row[2]);
    row[2] = sub_v3(row[2], mul_v3(row[1], shr[2]));

    // Next, get Z scale.
    scl.z = row[2].length();
    if !check_for_zero_scale_in_row_v3(scl.z, &row[2], exc)? {
        return Ok(false);
    }

    // Normalize 3rd row and correct the XZ and YZ shear factors for Z scaling.
    row[2] = div_v3(row[2], scl.z);
    shr[1] = shr[1] / scl.z;
    shr[2] = shr[2] / scl.z;

    // Check for a coordinate system flip. If the determinant is < 0, negate the
    // matrix and the scaling factors.
    if row[0].dot(&row[1].cross(&row[2])) < T::ZERO {
        for i in 0..3 {
            scl[i] = scl[i] * (-T::ONE);
            row[i] = mul_v3(row[i], -T::ONE);
        }
    }

    // Copy over the orthonormal rows into the returned matrix (now a rotation).
    for i in 0..3 {
        mat[i][0] = row[i][0];
        mat[i][1] = row[i][1];
        mat[i][2] = row[i][2];
    }

    // Correct the scaling factors for the normalization step.
    scl.x = scl.x * max_val;
    scl.y = scl.y * max_val;
    scl.z = scl.z * max_val;

    Ok(true)
}

/// Port of `void extractEulerXYZ (const Matrix44<T>& mat, Vec3<T>& rot)`.
pub fn extract_euler_xyz<T>(mat: &Matrix44<T>, rot: &mut Vec3<T>)
where
    T: VecFloat + Math,
{
    // Normalize the local x, y and z axes to remove scaling.
    let i = Vec3::new(mat[0][0], mat[0][1], mat[0][2]).normalized();
    let j = Vec3::new(mat[1][0], mat[1][1], mat[1][2]).normalized();
    let k = Vec3::new(mat[2][0], mat[2][1], mat[2][2]).normalized();

    let m = Matrix44::new(
        i[0],
        i[1],
        i[2],
        T::ZERO,
        j[0],
        j[1],
        j[2],
        T::ZERO,
        k[0],
        k[1],
        k[2],
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ONE,
    );

    // Extract the first angle, rot.x.
    rot.x = m[1][2].atan2(m[2][2]);

    // Remove the rot.x rotation from M so the remaining rotation N is around two
    // axes (gimbal lock cannot occur).
    let mut n = Matrix44::default();
    n.rotate(&Vec3::new(-rot.x, T::ZERO, T::ZERO));
    let n = n.mul_matrix(&m);

    // Extract rot.y and rot.z from N.
    let cy = (n[0][0] * n[0][0] + n[0][1] * n[0][1]).sqrt();
    rot.y = (-n[0][2]).atan2(cy);
    rot.z = (-n[1][0]).atan2(n[1][1]);
}

/// Port of `void extractEulerZYX (const Matrix44<T>& mat, Vec3<T>& rot)`.
pub fn extract_euler_zyx<T>(mat: &Matrix44<T>, rot: &mut Vec3<T>)
where
    T: VecFloat + Math,
{
    // Normalize the local x, y and z axes to remove scaling.
    let i = Vec3::new(mat[0][0], mat[0][1], mat[0][2]).normalized();
    let j = Vec3::new(mat[1][0], mat[1][1], mat[1][2]).normalized();
    let k = Vec3::new(mat[2][0], mat[2][1], mat[2][2]).normalized();

    let m = Matrix44::new(
        i[0],
        i[1],
        i[2],
        T::ZERO,
        j[0],
        j[1],
        j[2],
        T::ZERO,
        k[0],
        k[1],
        k[2],
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ONE,
    );

    // Extract the first angle, rot.x.
    rot.x = -m[1][0].atan2(m[0][0]);

    // Remove the x rotation from M.
    let mut n = Matrix44::default();
    n.rotate(&Vec3::new(T::ZERO, T::ZERO, -rot.x));
    let n = n.mul_matrix(&m);

    // Extract rot.y and rot.z from N.
    let cy = (n[2][2] * n[2][2] + n[2][1] * n[2][1]).sqrt();
    rot.y = -(-n[2][0]).atan2(cy);
    rot.z = -(-n[1][2]).atan2(n[1][1]);
}

/// Port of `Quat<T> extractQuat (const Matrix44<T>& mat)`.
pub fn extract_quat<T>(mat: &Matrix44<T>) -> Quat<T>
where
    T: VecFloat + Math,
{
    let mut quat = Quat::default();
    let mut q = [T::ZERO; 4];

    let nxt = [1usize, 2, 0];
    let tr = mat[0][0] + mat[1][1] + mat[2][2];

    // check the diagonal
    if tr > T::ZERO {
        let mut s = (tr + T::ONE).sqrt();
        quat.r = s / (T::ONE + T::ONE);
        s = (T::ONE / (T::ONE + T::ONE)) / s;

        quat.v.x = (mat[1][2] - mat[2][1]) * s;
        quat.v.y = (mat[2][0] - mat[0][2]) * s;
        quat.v.z = (mat[0][1] - mat[1][0]) * s;
    } else {
        // diagonal is negative
        let mut i = 0usize;
        if mat[1][1] > mat[0][0] {
            i = 1;
        }
        if mat[2][2] > mat[i][i] {
            i = 2;
        }

        let j = nxt[i];
        let k = nxt[j];
        let mut s = ((mat[i][i] - (mat[j][j] + mat[k][k])) + T::ONE).sqrt();

        q[i] = s * (T::ONE / (T::ONE + T::ONE));
        if s != T::ZERO {
            s = (T::ONE / (T::ONE + T::ONE)) / s;
        }

        q[3] = (mat[j][k] - mat[k][j]) * s;
        q[j] = (mat[i][j] + mat[j][i]) * s;
        q[k] = (mat[i][k] + mat[k][i]) * s;

        quat.v.x = q[0];
        quat.v.y = q[1];
        quat.v.z = q[2];
        quat.r = q[3];
    }

    quat
}

/// Port of the full `bool extractSHRT (mat, s, h, r, t, exc, rOrder)` (4x4).
#[allow(clippy::too_many_arguments)]
pub fn extract_shrt_m44_order<T>(
    mat: &Matrix44<T>,
    s: &mut Vec3<T>,
    h: &mut Vec3<T>,
    r: &mut Vec3<T>,
    t: &mut Vec3<T>,
    exc: bool,
    r_order: Order,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    let mut rot = *mat;
    if !extract_and_remove_scaling_and_shear_m44(&mut rot, s, h, exc)? {
        return Ok(false);
    }

    extract_euler_xyz(&rot, r);

    t.x = mat[3][0];
    t.y = mat[3][1];
    t.z = mat[3][2];

    if r_order != Order::XYZ {
        let e_xyz = Euler::from_vec_order_layout(*r, Order::XYZ, InputLayout::IJKLayout);
        let e = Euler::from_euler_order(&e_xyz, r_order);
        *r = e.to_xyz_vector();
    }

    Ok(true)
}

/// Port of `bool extractSHRT (mat, s, h, r, t, exc)` — XYZ-order convenience (4x4,
/// rotation as a `Vec3` of XYZ euler angles).
pub fn extract_shrt_m44_vec<T>(
    mat: &Matrix44<T>,
    s: &mut Vec3<T>,
    h: &mut Vec3<T>,
    r: &mut Vec3<T>,
    t: &mut Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    extract_shrt_m44_order(mat, s, h, r, t, exc, Order::XYZ)
}

/// Port of `bool extractSHRT (mat, s, h, Euler<T>& r, t, exc)` — the rotation comes
/// out in the `Euler`'s own order.
pub fn extract_shrt_m44_euler<T>(
    mat: &Matrix44<T>,
    s: &mut Vec3<T>,
    h: &mut Vec3<T>,
    r: &mut Euler<T>,
    t: &mut Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    let order = r.order();
    // C++ (`ImathMatrixAlgo.h:870-881`) passes the Euler (a Vec3 subclass)
    // straight through extractSHRT, so the scratch vector starts as the caller's
    // current angles and is only written when the extraction succeeds — on the
    // `return false` path (singular matrix, `exc == false`) C++ leaves `r`
    // untouched. Seeding from `r.vec` and assigning only on success reproduces
    // both halves of that aliasing exactly.
    let mut rvec = r.vec;
    let ok = extract_shrt_m44_order(mat, s, h, &mut rvec, t, exc, order)?;
    if ok {
        r.vec = rvec;
    }
    Ok(ok)
}

/// Port of `bool checkForZeroScaleInRow (const T& scl, const Vec3<T>& row, bool exc)`.
///
/// Returns `Ok(true)` if the scale can be safely removed, `Ok(false)` on the
/// non-throwing overflow path, or `Err` on the throwing path.
pub fn check_for_zero_scale_in_row_v3<T>(
    scl: T,
    row: &Vec3<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: Scalar + VecLimits,
{
    for i in 0..3 {
        if abs(scl) < T::ONE && abs(row[i]) >= T::MAX * abs(scl) {
            if exc {
                return Err("Cannot remove zero scaling from matrix.");
            } else {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Port of `Matrix44<T> outerProduct (const Vec4<T>& a, const Vec4<T>& b)`.
///
/// Note: this faithfully reproduces the C++ source, which has a long-standing
/// typo — three of the fourth-column entries use `a.x * b.w` instead of the
/// expected `a.y/a.z * b.w`. Replicated verbatim for 1:1 parity.
pub fn outer_product_v4<T>(a: &Vec4<T>, b: &Vec4<T>) -> Matrix44<T>
where
    T: Scalar,
{
    Matrix44::new(
        a.x * b.x,
        a.x * b.y,
        a.x * b.z,
        a.x * b.w,
        a.y * b.x,
        a.y * b.y,
        a.y * b.z,
        a.x * b.w,
        a.z * b.x,
        a.z * b.y,
        a.z * b.z,
        a.x * b.w,
        a.w * b.x,
        a.w * b.y,
        a.w * b.z,
        a.w * b.w,
    )
}

/// Port of `Matrix44<T> rotationMatrix (const Vec3<T>& from, const Vec3<T>& to)`.
pub fn rotation_matrix<T>(from: &Vec3<T>, to: &Vec3<T>) -> Matrix44<T>
where
    T: QuatFloat,
{
    let mut q = Quat::default();
    q.set_rotation(from, to);
    q.to_matrix44()
}

/// Port of `Matrix44<T> rotationMatrixWithUpDir (fromDir, toDir, upDir)`.
pub fn rotation_matrix_with_up_dir<T>(
    from_dir: &Vec3<T>,
    to_dir: &Vec3<T>,
    up_dir: &Vec3<T>,
) -> Matrix44<T>
where
    T: VecFloat,
{
    // The from direction must be non-zero; we allow zero to/up dirs.
    if from_dir.length() == T::ZERO {
        return Matrix44::default();
    }

    let mut z_axis2_from_dir = Matrix44::default();
    align_z_axis_with_target_dir(&mut z_axis2_from_dir, *from_dir, Vec3::new(T::ZERO, T::ONE, T::ZERO));

    let from_dir2_z_axis = z_axis2_from_dir.transposed();

    let mut z_axis2_to_dir = Matrix44::default();
    align_z_axis_with_target_dir(&mut z_axis2_to_dir, *to_dir, *up_dir);

    from_dir2_z_axis.mul_matrix(&z_axis2_to_dir)
}

/// Port of `void alignZAxisWithTargetDir (Matrix44<T>& result, Vec3<T> targetDir,
/// Vec3<T> upDir)`.
pub fn align_z_axis_with_target_dir<T>(
    result: &mut Matrix44<T>,
    mut target_dir: Vec3<T>,
    mut up_dir: Vec3<T>,
) where
    T: VecFloat,
{
    // Ensure that the target direction is non-zero.
    if target_dir.length() == T::ZERO {
        target_dir = Vec3::new(T::ZERO, T::ZERO, T::ONE);
    }

    // Ensure that the up direction is non-zero.
    if up_dir.length() == T::ZERO {
        up_dir = Vec3::new(T::ZERO, T::ONE, T::ZERO);
    }

    // Check for degeneracies: if upDir and targetDir are parallel or opposite,
    // compute a new, arbitrary up direction.
    if up_dir.cross(&target_dir).length() == T::ZERO {
        up_dir = target_dir.cross(&Vec3::new(T::ONE, T::ZERO, T::ZERO));
        if up_dir.length() == T::ZERO {
            up_dir = target_dir.cross(&Vec3::new(T::ZERO, T::ZERO, T::ONE));
        }
    }

    // Compute the x-, y-, and z-axis vectors of the new coordinate system.
    let target_perp_dir = up_dir.cross(&target_dir);
    let target_up_dir = target_dir.cross(&target_perp_dir);

    // Rotate the x-axis into targetPerpDir (row 0), y into targetUpDir (row 1),
    // z into targetDir (row 2).
    let row = [
        target_perp_dir.normalized(),
        target_up_dir.normalized(),
        target_dir.normalized(),
    ];

    result.x[0][0] = row[0][0];
    result.x[0][1] = row[0][1];
    result.x[0][2] = row[0][2];
    result.x[0][3] = T::ZERO;

    result.x[1][0] = row[1][0];
    result.x[1][1] = row[1][1];
    result.x[1][2] = row[1][2];
    result.x[1][3] = T::ZERO;

    result.x[2][0] = row[2][0];
    result.x[2][1] = row[2][1];
    result.x[2][2] = row[2][2];
    result.x[2][3] = T::ZERO;

    result.x[3][0] = T::ZERO;
    result.x[3][1] = T::ZERO;
    result.x[3][2] = T::ZERO;
    result.x[3][3] = T::ONE;
}

/// Port of `Matrix44<T> computeLocalFrame (const Vec3<T>& p, const Vec3<T>& xDir,
/// const Vec3<T>& normal)`.
pub fn compute_local_frame<T>(p: &Vec3<T>, x_dir: &Vec3<T>, normal: &Vec3<T>) -> Matrix44<T>
where
    T: VecFloat,
{
    // C++ uses the `%` (cross) operator on Vec3 and the in-place `normalize()`.
    let mut x_dir_copy = *x_dir;
    let x = *x_dir_copy.normalize();
    let y = *(*normal % x).normalize();
    let z = *(x % y).normalize();

    let mut l = Matrix44::default();
    l[0][0] = x[0];
    l[0][1] = x[1];
    l[0][2] = x[2];
    l[0][3] = T::ZERO;

    l[1][0] = y[0];
    l[1][1] = y[1];
    l[1][2] = y[2];
    l[1][3] = T::ZERO;

    l[2][0] = z[0];
    l[2][1] = z[1];
    l[2][2] = z[2];
    l[2][3] = T::ZERO;

    l[3][0] = p[0];
    l[3][1] = p[1];
    l[3][2] = p[2];
    l[3][3] = T::ONE;

    l
}

/// Port of `Matrix44<T> addOffset (inMat, tOffset, rOffset, sOffset, ref)`.
///
/// `rOffset` is given in degrees and converted to radians via `T(M_PI/180.0)`.
pub fn add_offset<T>(
    in_mat: &Matrix44<T>,
    t_offset: &Vec3<T>,
    r_offset: &Vec3<T>,
    s_offset: &Vec3<T>,
    reference: &Matrix44<T>,
) -> Matrix44<T>
where
    T: VecFloat + Math + VecCast<f64>,
{
    let mut o = Matrix44::default();

    let deg2rad = T::from_elem(M_PI / 180.0);
    let r_off = Vec3::new(r_offset.x * deg2rad, r_offset.y * deg2rad, r_offset.z * deg2rad);
    o.rotate(&r_off);

    o[3][0] = t_offset[0];
    o[3][1] = t_offset[1];
    o[3][2] = t_offset[2];

    let mut s = Matrix44::default();
    s.scale(s_offset);

    // X = S * O * inMat * ref
    s.mul_matrix(&o).mul_matrix(in_mat).mul_matrix(reference)
}

/// Port of `Matrix44<T> computeRSMatrix (keepRotateA, keepScaleA, A, B)`.
///
/// The C++ throws `std::domain_error` on a degenerate `A` or `B`; here that is the
/// `Err` arm (the `exc` is implicitly `true` in the original `extractSHRT` calls).
pub fn compute_rs_matrix<T>(
    keep_rotate_a: bool,
    keep_scale_a: bool,
    a: &Matrix44<T>,
    b: &Matrix44<T>,
) -> Result<Matrix44<T>, &'static str>
where
    T: VecFloat + VecLimits + EulerFloat,
{
    let mut as_ = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut ah = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut ar = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut at = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    if !extract_shrt_m44_vec(a, &mut as_, &mut ah, &mut ar, &mut at, true)? {
        return Err("degenerate A matrix in computeRSMatrix");
    }

    let mut bs = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut bh = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut br = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut bt = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    if !extract_shrt_m44_vec(b, &mut bs, &mut bh, &mut br, &mut bt, true)? {
        return Err("degenerate B matrix in computeRSMatrix");
    }

    if !keep_rotate_a {
        ar = br;
    }
    if !keep_scale_a {
        as_ = bs;
    }

    let mut mat = Matrix44::default();
    mat.make_identity();
    mat.translate(&at);
    mat.rotate(&ar);
    mat.scale(&as_);

    Ok(mat)
}

// ============================================================================
// Implementation for 3x3 Matrix
// ============================================================================

/// Port of `bool extractScaling (const Matrix33<T>& mat, Vec2<T>& scl, bool exc)`.
pub fn extract_scaling_m33<T>(
    mat: &Matrix33<T>,
    scl: &mut Vec2<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut shr = T::ZERO;
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m33(&mut m, scl, &mut shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `Matrix33<T> sansScaling (const Matrix33<T>& mat, bool exc)`.
pub fn sans_scaling_m33<T>(mat: &Matrix33<T>, exc: bool) -> Result<Matrix33<T>, &'static str>
where
    T: VecFloat + VecLimits + Math,
{
    let mut scl = Vec2::new(T::ZERO, T::ZERO);
    let mut shr = T::ZERO;
    let mut rot = T::ZERO;
    let mut tran = Vec2::new(T::ZERO, T::ZERO);

    if !extract_shrt_m33(mat, &mut scl, &mut shr, &mut rot, &mut tran, exc)? {
        return Ok(*mat);
    }

    let mut m = Matrix33::default();
    m.translate(&tran);
    m.rotate(rot);
    m.shear(&Vec2::new(shr, T::ZERO));

    Ok(m)
}

/// Port of `bool removeScaling (Matrix33<T>& mat, bool exc)`.
pub fn remove_scaling_m33<T>(mat: &mut Matrix33<T>, exc: bool) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + Math,
{
    let mut scl = Vec2::new(T::ZERO, T::ZERO);
    let mut shr = T::ZERO;
    let mut rot = T::ZERO;
    let mut tran = Vec2::new(T::ZERO, T::ZERO);

    if !extract_shrt_m33(mat, &mut scl, &mut shr, &mut rot, &mut tran, exc)? {
        return Ok(false);
    }

    mat.make_identity();
    mat.translate(&tran);
    mat.rotate(rot);
    mat.shear(&Vec2::new(shr, T::ZERO));

    Ok(true)
}

/// Port of `bool extractScalingAndShear (const Matrix33<T>& mat, Vec2<T>& scl,
/// T& shr, bool exc)`.
pub fn extract_scaling_and_shear_m33<T>(
    mat: &Matrix33<T>,
    scl: &mut Vec2<T>,
    shr: &mut T,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m33(&mut m, scl, shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `Matrix33<T> sansScalingAndShear (const Matrix33<T>& mat, bool exc)`.
pub fn sans_scaling_and_shear_m33<T>(
    mat: &Matrix33<T>,
    exc: bool,
) -> Result<Matrix33<T>, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut scl = Vec2::new(T::ZERO, T::ZERO);
    let mut shr = T::ZERO;
    let mut m = *mat;

    if !extract_and_remove_scaling_and_shear_m33(&mut m, &mut scl, &mut shr, exc)? {
        return Ok(*mat);
    }

    Ok(m)
}

/// Port of `bool removeScalingAndShear (Matrix33<T>& mat, bool exc)`.
pub fn remove_scaling_and_shear_m33<T>(
    mat: &mut Matrix33<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut scl = Vec2::new(T::ZERO, T::ZERO);
    let mut shr = T::ZERO;

    if !extract_and_remove_scaling_and_shear_m33(mat, &mut scl, &mut shr, exc)? {
        return Ok(false);
    }

    Ok(true)
}

/// Port of `bool extractAndRemoveScalingAndShear (Matrix33<T>& mat, Vec2<T>& scl,
/// T& shr, bool exc)`.
pub fn extract_and_remove_scaling_and_shear_m33<T>(
    mat: &mut Matrix33<T>,
    scl: &mut Vec2<T>,
    shr: &mut T,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits,
{
    let mut row = [
        Vec2::new(mat[0][0], mat[0][1]),
        Vec2::new(mat[1][0], mat[1][1]),
    ];

    let mut max_val = T::ZERO;
    for i in 0..2 {
        for j in 0..2 {
            if abs(mat[i][j]) > max_val {
                max_val = abs(mat[i][j]);
            }
        }
    }

    // Normalize the 2x2 for numerical stability.
    if max_val != T::ZERO {
        for r in &mut row {
            if !check_for_zero_scale_in_row_v2(max_val, r, exc)? {
                return Ok(false);
            }
            *r = div_v2(*r, max_val);
        }
    }

    // Compute X scale factor.
    scl.x = row[0].length();
    if !check_for_zero_scale_in_row_v2(scl.x, &row[0], exc)? {
        return Ok(false);
    }

    // Normalize first row.
    row[0] = div_v2(row[0], scl.x);

    // Compute XY shear factor and make 2nd row orthogonal to 1st.
    *shr = row[0].dot(&row[1]);
    row[1] = sub_v2(row[1], mul_v2(row[0], *shr));

    // Now, compute Y scale.
    scl.y = row[1].length();
    if !check_for_zero_scale_in_row_v2(scl.y, &row[1], exc)? {
        return Ok(false);
    }

    // Normalize 2nd row and correct the XY shear factor for Y scaling.
    row[1] = div_v2(row[1], scl.y);
    *shr = *shr / scl.y;

    // Check for a coordinate system flip; if the determinant is -1, flip the
    // rotation matrix and adjust scale(Y) and shear(XY).
    if row[0].x * row[1].y - row[0].y * row[1].x < T::ZERO {
        row[1].x = row[1].x * (-T::ONE);
        row[1].y = row[1].y * (-T::ONE);
        scl.y = scl.y * (-T::ONE);
        *shr = *shr * (-T::ONE);
    }

    // Copy over the orthonormal rows into the returned matrix.
    for i in 0..2 {
        mat[i][0] = row[i].x;
        mat[i][1] = row[i].y;
    }

    scl.x = scl.x * max_val;
    scl.y = scl.y * max_val;

    Ok(true)
}

/// Port of `void extractEuler (const Matrix22<T>& mat, T& rot)`.
pub fn extract_euler_m22<T>(mat: &Matrix22<T>, rot: &mut T)
where
    T: VecFloat + Math,
{
    // Normalize the local x and y axes to remove scaling.
    let i = normalize_v2(Vec2::new(mat[0][0], mat[0][1]));
    let j = normalize_v2(Vec2::new(mat[1][0], mat[1][1]));

    // Extract the angle, rot.
    *rot = -j[0].atan2(i[0]);
}

/// Port of `void extractEuler (const Matrix33<T>& mat, T& rot)`.
pub fn extract_euler_m33<T>(mat: &Matrix33<T>, rot: &mut T)
where
    T: VecFloat + Math,
{
    // Normalize the local x and y axes to remove scaling.
    let i = normalize_v2(Vec2::new(mat[0][0], mat[0][1]));
    let j = normalize_v2(Vec2::new(mat[1][0], mat[1][1]));

    // Extract the angle, rot.
    *rot = -j[0].atan2(i[0]);
}

/// Port of `bool extractSHRT (const Matrix33<T>& mat, Vec2<T>& s, T& h, T& r,
/// Vec2<T>& t, bool exc)`.
pub fn extract_shrt_m33<T>(
    mat: &Matrix33<T>,
    s: &mut Vec2<T>,
    h: &mut T,
    r: &mut T,
    t: &mut Vec2<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: VecFloat + VecLimits + Math,
{
    let mut rot = *mat;
    if !extract_and_remove_scaling_and_shear_m33(&mut rot, s, h, exc)? {
        return Ok(false);
    }

    extract_euler_m33(&rot, r);

    t.x = mat[2][0];
    t.y = mat[2][1];

    Ok(true)
}

/// Port of `bool checkForZeroScaleInRow (const T& scl, const Vec2<T>& row, bool exc)`.
pub fn check_for_zero_scale_in_row_v2<T>(
    scl: T,
    row: &Vec2<T>,
    exc: bool,
) -> Result<bool, &'static str>
where
    T: Scalar + VecLimits,
{
    for i in 0..2 {
        if abs(scl) < T::ONE && abs(row[i]) >= T::MAX * abs(scl) {
            if exc {
                return Err("Cannot remove zero scaling from matrix.");
            } else {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Port of `Matrix33<T> outerProduct (const Vec3<T>& a, const Vec3<T>& b)`.
pub fn outer_product_v3<T>(a: &Vec3<T>, b: &Vec3<T>) -> Matrix33<T>
where
    T: Scalar,
{
    Matrix33::new(
        a.x * b.x,
        a.x * b.y,
        a.x * b.z,
        a.y * b.x,
        a.y * b.y,
        a.y * b.z,
        a.z * b.x,
        a.z * b.y,
        a.z * b.z,
    )
}

// ============================================================================
// Procrustes (port of the .cpp anonymous-namespace KahanSum + the two
// procrustesRotationAndTranslation overloads).
// ============================================================================

/// Kahan compensated summation (port of the anonymous-namespace `class KahanSum`).
struct KahanSum {
    total: f64,
    correction: f64,
}

impl KahanSum {
    #[inline]
    fn new() -> Self {
        KahanSum {
            total: 0.0,
            correction: 0.0,
        }
    }

    #[inline]
    fn add(&mut self, val: f64) {
        let y = val - self.correction;
        let t = self.total + y;
        self.correction = (t - self.total) - y;
        self.total = t;
    }

    #[inline]
    fn get(&self) -> f64 {
        self.total
    }
}

/// Port of `M44d procrustesRotationAndTranslation (A, B, weights, numPoints,
/// doScale)` — the weighted form.
///
/// Accumulation is always in `double` (matching the C++); the result is `M44d`.
/// `weights == None` is the C++ `weights == 0` (unweighted) branch.
pub fn procrustes_rotation_and_translation_weighted<T>(
    a: &[Vec3<T>],
    b: &[Vec3<T>],
    weights: Option<&[T]>,
    num_points: usize,
    do_scale: bool,
) -> M44d
where
    T: VecFloat,
    f64: VecCast<T>,
{
    if num_points == 0 {
        return M44d::default();
    }

    // Always do the accumulation in double precision.
    let mut a_center = v3d_zero();
    let mut b_center = v3d_zero();
    let mut weights_sum: f64 = 0.0;

    match weights {
        None => {
            for i in 0..num_points {
                a_center = add_v3(a_center, to_v3d(&a[i]));
                b_center = add_v3(b_center, to_v3d(&b[i]));
            }
            weights_sum = num_points as f64;
        }
        Some(w) => {
            for i in 0..num_points {
                let wi = f64::from_elem(w[i]);
                weights_sum += wi;
                a_center = add_v3(a_center, mul_v3(to_v3d(&a[i]), wi));
                b_center = add_v3(b_center, mul_v3(to_v3d(&b[i]), wi));
            }
        }
    }

    if weights_sum == 0.0 {
        return M44d::default();
    }

    a_center = div_v3(a_center, weights_sum);
    b_center = div_v3(b_center, weights_sum);

    // Let C = B A^T  (3x3). Compute the SVD C = U D V^T, throw away D, and return
    // Q = U V^T. (Golub/Van Loan, p.601.)
    let mut c = M33d::from_scalar(0.0);
    match weights {
        None => {
            // C++ indexes `A[i]`/`B[i]` together; mirror that index-based loop.
            #[allow(clippy::needless_range_loop)]
            for i in 0..num_points {
                let op = outer_product_v3(
                    &sub_v3(to_v3d(&b[i]), b_center),
                    &sub_v3(to_v3d(&a[i]), a_center),
                );
                c.add_assign_matrix(&op);
            }
        }
        Some(w) => {
            for i in 0..num_points {
                let wi = f64::from_elem(w[i]);
                let op = outer_product_v3(
                    &mul_v3(sub_v3(to_v3d(&b[i]), b_center), wi),
                    &sub_v3(to_v3d(&a[i]), a_center),
                );
                c.add_assign_matrix(&op);
            }
        }
    }

    let mut u = M33d::default();
    let mut v = M33d::default();
    let mut s = v3d_zero();
    jacobi_svd_m33(&c, &mut u, &mut s, &mut v, f64::EPSILON, true);

    // We want Q.transposed() since we multiply vectors on the right (v' = v*A^T).
    let qt = v.mul_matrix(&u.transposed());

    let mut scale = 1.0f64;
    if do_scale && num_points > 1 {
        // s = tr(Q^T*A^T*B) / tr(A^T*A)
        let mut trace_ata = KahanSum::new();
        match weights {
            None => {
                for ai in a.iter().take(num_points) {
                    trace_ata.add(sub_v3(to_v3d(ai), a_center).length2());
                }
            }
            Some(w) => {
                for i in 0..num_points {
                    trace_ata
                        .add(f64::from_elem(w[i]) * sub_v3(to_v3d(&a[i]), a_center).length2());
                }
            }
        }

        let mut trace_batq = KahanSum::new();
        for i in 0..3 {
            for j in 0..3 {
                trace_batq.add(qt[j][i] * c[i][j]);
            }
        }

        scale = trace_batq.get() / trace_ata.get();
    }

    // translate = Bcenter - s * Acenter * Qt
    let translate = sub_v3(b_center, mul_v3(vec3_mul_m33d(&a_center, &qt), scale));

    M44d::new(
        scale * qt.x[0][0],
        scale * qt.x[0][1],
        scale * qt.x[0][2],
        0.0,
        scale * qt.x[1][0],
        scale * qt.x[1][1],
        scale * qt.x[1][2],
        0.0,
        scale * qt.x[2][0],
        scale * qt.x[2][1],
        scale * qt.x[2][2],
        0.0,
        translate.x,
        translate.y,
        translate.z,
        1.0,
    )
}

/// Port of `M44d procrustesRotationAndTranslation (A, B, numPoints, doScale)` —
/// the unweighted form (forwards to the weighted form with `weights == 0`).
pub fn procrustes_rotation_and_translation<T>(
    a: &[Vec3<T>],
    b: &[Vec3<T>],
    num_points: usize,
    do_scale: bool,
) -> M44d
where
    T: VecFloat,
    f64: VecCast<T>,
{
    procrustes_rotation_and_translation_weighted(a, b, None, num_points, do_scale)
}

// ============================================================================
// Jacobi SVD (two-sided) — port of the .cpp anonymous-namespace machinery and
// the public jacobiSVD entry points.
// ============================================================================

/// Port of the `template <typename T, int j, int k> jacobiRotateRight(Matrix33)` —
/// `J * A` applied from the right, touching only columns `j`, `k`.
#[inline]
fn jacobi_rotate_right_m33<T: Scalar>(a: &mut Matrix33<T>, j: usize, k: usize, c: T, s: T) {
    for i in 0..3 {
        let tau1 = a[i][j];
        let tau2 = a[i][k];
        a[i][j] = c * tau1 - s * tau2;
        a[i][k] = s * tau1 + c * tau2;
    }
}

/// Port of the `template <typename T> jacobiRotateRight(Matrix44, j, k, c, s)`.
#[inline]
fn jacobi_rotate_right_m44<T: Scalar>(a: &mut Matrix44<T>, j: usize, k: usize, c: T, s: T) {
    for i in 0..4 {
        let tau1 = a[i][j];
        let tau2 = a[i][k];
        a[i][j] = c * tau1 - s * tau2;
        a[i][k] = s * tau1 + c * tau2;
    }
}

/// Port of `twoSidedJacobiRotation<T, j, k, l>(Matrix33 A, U, V, tol)` — the 'USVD'
/// 2x2 SVD step (Brent/Luk/Van Loan), symmetrize then diagonalize.
#[allow(clippy::too_many_arguments)]
fn two_sided_jacobi_rotation_m33<T>(
    a: &mut Matrix33<T>,
    u: &mut Matrix33<T>,
    v: &mut Matrix33<T>,
    j: usize,
    k: usize,
    l: usize,
    tol: T,
) -> bool
where
    T: VecFloat + Math,
{
    let w = a[j][j];
    let x = a[j][k];
    let y = a[k][j];
    let z = a[k][k];

    let mut changed = false;

    // Symmetrize the 2x2 matrix.
    let mut mu_1 = w + z;
    let mut mu_2 = x - y;

    let c;
    let s;
    if mu_2.fabs() <= tol * mu_1.fabs() {
        // Already symmetric to tolerance. (`<=` is important to bypass the rho
        // computation when mu_1 = mu_2 = 0.)
        c = T::ONE;
        s = T::ZERO;

        let p = w;
        let r = z;
        mu_1 = r - p;
        mu_2 = x + y;
    } else {
        let rho = mu_1 / mu_2;
        let mut s_local = T::ONE / (T::ONE + rho * rho).sqrt();
        if rho < T::ZERO {
            s_local = -s_local;
        }
        s = s_local;
        c = s_local * rho;

        mu_1 = s * (x + y) + c * (z - w); // = r - p
        mu_2 = (T::ONE + T::ONE) * (c * x - s * z); // = 2*q

        changed = true;
    }

    // Diagonalize.
    let c_2;
    let s_2;
    if mu_2.fabs() <= tol * mu_1.fabs() {
        c_2 = T::ONE;
        s_2 = T::ZERO;
    } else {
        let rho_2 = mu_1 / mu_2;
        let mut t_2 = T::ONE / (rho_2.fabs() + (T::ONE + rho_2 * rho_2).sqrt());
        if rho_2 < T::ZERO {
            t_2 = -t_2;
        }
        c_2 = T::ONE / (T::ONE + t_2 * t_2).sqrt();
        s_2 = c_2 * t_2;

        changed = true;
    }

    let c_1 = c_2 * c - s_2 * s;
    let s_1 = s_2 * c + c_2 * s;

    if !changed {
        // Off-diagonals already small enough; zero them and short-circuit.
        a[k][j] = T::ZERO;
        a[j][k] = T::ZERO;
        return false;
    }

    let d_1 = c_1 * (w * c_2 - x * s_2) - s_1 * (y * c_2 - z * s_2);
    let d_2 = s_1 * (w * s_2 + x * c_2) + c_1 * (y * s_2 + z * c_2);

    a[j][j] = d_1;
    a[k][k] = d_2;
    a[k][j] = T::ZERO;
    a[j][k] = T::ZERO;

    // Rotate the entries that weren't involved in the 2x2 SVD: left by R1 (rows).
    {
        let tau1 = a[j][l];
        let tau2 = a[k][l];
        a[j][l] = c_1 * tau1 - s_1 * tau2;
        a[k][l] = s_1 * tau1 + c_1 * tau2;
    }
    // Right by R2 (columns).
    {
        let tau1 = a[l][j];
        let tau2 = a[l][k];
        a[l][j] = c_2 * tau1 - s_2 * tau2;
        a[l][k] = s_2 * tau1 + c_2 * tau2;
    }

    // Apply rotations to U and V.
    jacobi_rotate_right_m33(u, j, k, c_1, s_1);
    jacobi_rotate_right_m33(v, j, k, c_2, s_2);

    true
}

/// Port of `twoSidedJacobiRotation(Matrix44 A, j, k, U, V, tol)`.
#[allow(clippy::too_many_arguments)]
fn two_sided_jacobi_rotation_m44<T>(
    a: &mut Matrix44<T>,
    j: usize,
    k: usize,
    u: &mut Matrix44<T>,
    v: &mut Matrix44<T>,
    tol: T,
) -> bool
where
    T: VecFloat + Math,
{
    let w = a[j][j];
    let x = a[j][k];
    let y = a[k][j];
    let z = a[k][k];

    let mut changed = false;

    let mut mu_1 = w + z;
    let mut mu_2 = x - y;

    let c;
    let s;
    if mu_2.fabs() <= tol * mu_1.fabs() {
        c = T::ONE;
        s = T::ZERO;

        let p = w;
        let r = z;
        mu_1 = r - p;
        mu_2 = x + y;
    } else {
        let rho = mu_1 / mu_2;
        let mut s_local = T::ONE / (T::ONE + rho * rho).sqrt();
        if rho < T::ZERO {
            s_local = -s_local;
        }
        s = s_local;
        c = s_local * rho;

        mu_1 = s * (x + y) + c * (z - w);
        mu_2 = (T::ONE + T::ONE) * (c * x - s * z);

        changed = true;
    }

    let c_2;
    let s_2;
    if mu_2.fabs() <= tol * mu_1.fabs() {
        c_2 = T::ONE;
        s_2 = T::ZERO;
    } else {
        let rho_2 = mu_1 / mu_2;
        let mut t_2 = T::ONE / (rho_2.fabs() + (T::ONE + rho_2 * rho_2).sqrt());
        if rho_2 < T::ZERO {
            t_2 = -t_2;
        }
        c_2 = T::ONE / (T::ONE + t_2 * t_2).sqrt();
        s_2 = c_2 * t_2;

        changed = true;
    }

    let c_1 = c_2 * c - s_2 * s;
    let s_1 = s_2 * c + c_2 * s;

    if !changed {
        a[k][j] = T::ZERO;
        a[j][k] = T::ZERO;
        return false;
    }

    let d_1 = c_1 * (w * c_2 - x * s_2) - s_1 * (y * c_2 - z * s_2);
    let d_2 = s_1 * (w * s_2 + x * c_2) + c_1 * (y * s_2 + z * c_2);

    a[j][j] = d_1;
    a[k][k] = d_2;
    a[k][j] = T::ZERO;
    a[j][k] = T::ZERO;

    // Rotate rows not involved in the 2x2 SVD (left by R1).
    for l in 0..4 {
        if l == j || l == k {
            continue;
        }
        let tau1 = a[j][l];
        let tau2 = a[k][l];
        a[j][l] = c_1 * tau1 - s_1 * tau2;
        a[k][l] = s_1 * tau1 + c_1 * tau2;
    }

    // Rotate columns not involved (right by R2).
    for l in 0..4 {
        if l == j || l == k {
            continue;
        }
        let tau1 = a[l][j];
        let tau2 = a[l][k];
        a[l][j] = c_2 * tau1 - s_2 * tau2;
        a[l][k] = s_2 * tau1 + c_2 * tau2;
    }

    jacobi_rotate_right_m44(u, j, k, c_1, s_1);
    jacobi_rotate_right_m44(v, j, k, c_2, s_2);

    true
}

/// Port of `void swapColumns (Matrix33<T>& A, int j, int k)`.
#[inline]
fn swap_columns_m33<T: Scalar>(a: &mut Matrix33<T>, j: usize, k: usize) {
    for i in 0..3 {
        a[i].swap(j, k);
    }
}

/// Port of `T maxOffDiag (const Matrix33<T>& A)`.
fn max_off_diag_m33<T>(a: &Matrix33<T>) -> T
where
    T: VecFloat + Math,
{
    let mut result = T::ZERO;
    result = vmax(result, a[0][1].fabs());
    result = vmax(result, a[0][2].fabs());
    result = vmax(result, a[1][0].fabs());
    result = vmax(result, a[1][2].fabs());
    result = vmax(result, a[2][0].fabs());
    result = vmax(result, a[2][1].fabs());
    result
}

/// Port of `T maxOffDiag (const Matrix44<T>& A)`.
fn max_off_diag_m44<T>(a: &Matrix44<T>) -> T
where
    T: VecFloat + Math,
{
    let mut result = T::ZERO;
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                result = vmax(result, a[i][j].fabs());
            }
        }
    }
    result
}

/// Port of `void twoSidedJacobiSVD (Matrix33 A, U, S, V, tol, forcePositiveDeterminant)`.
fn two_sided_jacobi_svd_m33<T>(
    mut a: Matrix33<T>,
    u: &mut Matrix33<T>,
    s: &mut Vec3<T>,
    v: &mut Matrix33<T>,
    tol: T,
    force_positive_determinant: bool,
) where
    T: VecFloat + Math,
{
    u.make_identity();
    v.make_identity();

    let max_iter = 20; // Prevents infinite loops in unlucky cases.
    let abs_tol = tol * max_off_diag_m33(&a); // Tolerance in terms of max off-diag.
    if abs_tol != T::ZERO {
        let mut num_iter = 0;
        loop {
            num_iter += 1;
            let mut changed = two_sided_jacobi_rotation_m33(&mut a, u, v, 0, 1, 2, tol);
            changed = two_sided_jacobi_rotation_m33(&mut a, u, v, 0, 2, 1, tol) || changed;
            changed = two_sided_jacobi_rotation_m33(&mut a, u, v, 1, 2, 0, tol) || changed;
            if !changed {
                break;
            }
            if !(max_off_diag_m33(&a) > abs_tol && num_iter < max_iter) {
                break;
            }
        }
    }

    // Whatever's left on the diagonal are the singular values.
    s.x = a[0][0];
    s.y = a[1][1];
    s.z = a[2][2];

    // Flip any negative singular values (contract: all positive).
    for i in 0..3 {
        if s[i] < T::ZERO {
            s[i] = -s[i];
            for j in 0..3 {
                u[j][i] = -u[j][i];
            }
        }
    }

    // Order the singular values largest-to-smallest via two bubble-sort passes.
    for i in 0..2 {
        for j in 0..(2 - i) {
            if s[j] < s[j + 1] {
                let tmp = s[j];
                s[j] = s[j + 1];
                s[j + 1] = tmp;
                swap_columns_m33(u, j, j + 1);
                swap_columns_m33(v, j, j + 1);
            }
        }
    }

    if force_positive_determinant {
        // Guarantee positive determinant via the L = diag(1,1,-1) trick.
        if u.determinant() < T::ZERO {
            for i in 0..3 {
                u[i][2] = -u[i][2];
            }
            s.z = -s.z;
        }
        if v.determinant() < T::ZERO {
            for i in 0..3 {
                v[i][2] = -v[i][2];
            }
            s.z = -s.z;
        }
    }
}

/// Port of `void twoSidedJacobiSVD (Matrix44 A, U, S, V, tol, forcePositiveDeterminant)`.
fn two_sided_jacobi_svd_m44<T>(
    mut a: Matrix44<T>,
    u: &mut Matrix44<T>,
    s: &mut Vec4<T>,
    v: &mut Matrix44<T>,
    tol: T,
    force_positive_determinant: bool,
) where
    T: VecFloat + Math,
{
    u.make_identity();
    v.make_identity();

    let max_iter = 20;
    let abs_tol = tol * max_off_diag_m44(&a);
    if abs_tol != T::ZERO {
        let mut num_iter = 0;
        loop {
            num_iter += 1;
            let mut changed = two_sided_jacobi_rotation_m44(&mut a, 0, 1, u, v, tol);
            changed = two_sided_jacobi_rotation_m44(&mut a, 0, 2, u, v, tol) || changed;
            changed = two_sided_jacobi_rotation_m44(&mut a, 0, 3, u, v, tol) || changed;
            changed = two_sided_jacobi_rotation_m44(&mut a, 1, 2, u, v, tol) || changed;
            changed = two_sided_jacobi_rotation_m44(&mut a, 1, 3, u, v, tol) || changed;
            changed = two_sided_jacobi_rotation_m44(&mut a, 2, 3, u, v, tol) || changed;
            if !changed {
                break;
            }
            if !(max_off_diag_m44(&a) > abs_tol && num_iter < max_iter) {
                break;
            }
        }
    }

    s.x = a[0][0];
    s.y = a[1][1];
    s.z = a[2][2];
    s.w = a[3][3];

    for i in 0..4 {
        if s[i] < T::ZERO {
            s[i] = -s[i];
            for j in 0..4 {
                u[j][i] = -u[j][i];
            }
        }
    }

    // Order singular values largest-to-smallest using insertion sort (by abs).
    for i in 1..4 {
        let u_col = Vec4::new(u[0][i], u[1][i], u[2][i], u[3][i]);
        let v_col = Vec4::new(v[0][i], v[1][i], v[2][i], v[3][i]);
        let s_val = s[i];

        let mut j = i as isize - 1;
        while j >= 0 && s[j as usize].fabs() < s_val.fabs() {
            let ju = j as usize;
            for k in 0..4 {
                u[k][ju + 1] = u[k][ju];
            }
            for k in 0..4 {
                v[k][ju + 1] = v[k][ju];
            }
            s[ju + 1] = s[ju];
            j -= 1;
        }

        let dst = (j + 1) as usize;
        for k in 0..4 {
            u[k][dst] = u_col[k];
        }
        for k in 0..4 {
            v[k][dst] = v_col[k];
        }
        s[dst] = s_val;
    }

    if force_positive_determinant {
        if u.determinant() < T::ZERO {
            for i in 0..4 {
                u[i][3] = -u[i][3];
            }
            s.w = -s.w;
        }
        if v.determinant() < T::ZERO {
            for i in 0..4 {
                v[i][3] = -v[i][3];
            }
            s.w = -s.w;
        }
    }
}

/// Port of `void jacobiSVD (const Matrix33<T>& A, U, S, V, tol, forcePositiveDeterminant)`.
pub fn jacobi_svd_m33<T>(
    a: &Matrix33<T>,
    u: &mut Matrix33<T>,
    s: &mut Vec3<T>,
    v: &mut Matrix33<T>,
    tol: T,
    force_positive_determinant: bool,
) where
    T: VecFloat + Math,
{
    two_sided_jacobi_svd_m33(*a, u, s, v, tol, force_positive_determinant);
}

/// Port of `void jacobiSVD (const Matrix44<T>& A, U, S, V, tol, forcePositiveDeterminant)`.
pub fn jacobi_svd_m44<T>(
    a: &Matrix44<T>,
    u: &mut Matrix44<T>,
    s: &mut Vec4<T>,
    v: &mut Matrix44<T>,
    tol: T,
    force_positive_determinant: bool,
) where
    T: VecFloat + Math,
{
    two_sided_jacobi_svd_m44(*a, u, s, v, tol, force_positive_determinant);
}

// ============================================================================
// Jacobi eigen-solver (symmetric) — port of the second anonymous namespace and
// the public jacobiEigenSolver / maxEigenVector / minEigenVector entry points.
// ============================================================================

/// Port of `jacobiRotateRight<j, k, TM>(TM& A, s, tau)` for Matrix33 (the variant
/// used by the eigen-solver, which updates with `tau`).
#[inline]
fn jacobi_rotate_right_tau_m33<T: Scalar>(a: &mut Matrix33<T>, j: usize, k: usize, s: T, tau: T) {
    for i in 0..3 {
        let nu1 = a[i][j];
        let nu2 = a[i][k];
        a[i][j] = nu1 - s * (nu2 + tau * nu1);
        a[i][k] = nu2 + s * (nu1 - tau * nu2);
    }
}

/// Port of `jacobiRotateRight<j, k, TM>(TM& A, s, tau)` for Matrix44.
#[inline]
fn jacobi_rotate_right_tau_m44<T: Scalar>(a: &mut Matrix44<T>, j: usize, k: usize, s: T, tau: T) {
    for i in 0..4 {
        let nu1 = a[i][j];
        let nu2 = a[i][k];
        a[i][j] = nu1 - s * (nu2 + tau * nu1);
        a[i][k] = nu2 + s * (nu1 - tau * nu2);
    }
}

/// Port of `jacobiRotation<j, k, l>(Matrix33 A, V, Z, tol)`.
///
/// `A` is assumed symmetric; only upper-triangular entries are updated.
#[allow(clippy::too_many_arguments)]
fn jacobi_rotation_m33<T>(
    a: &mut Matrix33<T>,
    v: &mut Matrix33<T>,
    z: &mut Vec3<T>,
    j: usize,
    k: usize,
    l: usize,
    tol: T,
) -> bool
where
    T: VecFloat + Math,
{
    let x = a[j][j];
    let y = a[j][k];
    let zz = a[k][k];

    let mu1 = zz - x;
    let mu2 = (T::ONE + T::ONE) * y;

    if mu2.fabs() <= tol * mu1.fabs() {
        a[j][k] = T::ZERO;
        return false;
    }

    let rho = mu1 / mu2;
    let t = (if rho < T::ZERO { -T::ONE } else { T::ONE }) / (rho.fabs() + (T::ONE + rho * rho).sqrt());
    let c = T::ONE / (T::ONE + t * t).sqrt();
    let s = t * c;
    let tau = s / (T::ONE + c);
    let h = t * y;

    // Update diagonal elements.
    z[j] = z[j] - h;
    z[k] = z[k] + h;
    a[j][j] = a[j][j] - h;
    a[k][k] = a[k][k] + h;

    a[j][k] = T::ZERO;

    // Only update upper-triangular off-diagonals of A (it is symmetric).
    let (o1r, o1c) = if l < j { (l, j) } else { (j, l) };
    let (o2r, o2c) = if l < k { (l, k) } else { (k, l) };
    let nu1 = a[o1r][o1c];
    let nu2 = a[o2r][o2c];
    a[o1r][o1c] = nu1 - s * (nu2 + tau * nu1);
    a[o2r][o2c] = nu2 + s * (nu1 - tau * nu2);

    jacobi_rotate_right_tau_m33(v, j, k, s, tau);

    true
}

/// Port of `jacobiRotation<j, k, l1, l2>(Matrix44 A, V, Z, tol)`.
#[allow(clippy::too_many_arguments)]
fn jacobi_rotation_m44<T>(
    a: &mut Matrix44<T>,
    v: &mut Matrix44<T>,
    z: &mut Vec4<T>,
    j: usize,
    k: usize,
    l1: usize,
    l2: usize,
    tol: T,
) -> bool
where
    T: VecFloat + Math,
{
    let x = a[j][j];
    let y = a[j][k];
    let zz = a[k][k];

    let mu1 = zz - x;
    let mu2 = (T::ONE + T::ONE) * y;

    // Note: the C++ returns `true` here (unlike the 3x3 variant which returns
    // false) — replicated verbatim.
    if mu2.fabs() <= tol * mu1.fabs() {
        a[j][k] = T::ZERO;
        return true;
    }

    let rho = mu1 / mu2;
    let t = (if rho < T::ZERO { -T::ONE } else { T::ONE }) / (rho.fabs() + (T::ONE + rho * rho).sqrt());
    let c = T::ONE / (T::ONE + t * t).sqrt();
    let s = c * t;
    let tau = s / (T::ONE + c);
    let h = t * y;

    z[j] = z[j] - h;
    z[k] = z[k] + h;
    a[j][j] = a[j][j] - h;
    a[k][k] = a[k][k] + h;
    a[j][k] = T::ZERO;

    {
        let (o1r, o1c) = if l1 < j { (l1, j) } else { (j, l1) };
        let (o2r, o2c) = if l1 < k { (l1, k) } else { (k, l1) };
        let nu1 = a[o1r][o1c];
        let nu2 = a[o2r][o2c];
        a[o1r][o1c] = nu1 - s * (nu2 + tau * nu1);
        a[o2r][o2c] = nu2 + s * (nu1 - tau * nu2);
    }
    {
        let (o1r, o1c) = if l2 < j { (l2, j) } else { (j, l2) };
        let (o2r, o2c) = if l2 < k { (l2, k) } else { (k, l2) };
        let nu1 = a[o1r][o1c];
        let nu2 = a[o2r][o2c];
        a[o1r][o1c] = nu1 - s * (nu2 + tau * nu1);
        a[o2r][o2c] = nu2 + s * (nu1 - tau * nu2);
    }

    jacobi_rotate_right_tau_m44(v, j, k, s, tau);

    true
}

/// Port of `T maxOffDiagSymm (const TM& A)` for Matrix33.
fn max_off_diag_symm_m33<T>(a: &Matrix33<T>) -> T
where
    T: VecFloat + Math,
{
    let mut result = T::ZERO;
    for i in 0..3 {
        for j in (i + 1)..3 {
            result = vmax(result, a[i][j].fabs());
        }
    }
    result
}

/// Port of `T maxOffDiagSymm (const TM& A)` for Matrix44.
fn max_off_diag_symm_m44<T>(a: &Matrix44<T>) -> T
where
    T: VecFloat + Math,
{
    let mut result = T::ZERO;
    for i in 0..4 {
        for j in (i + 1)..4 {
            result = vmax(result, a[i][j].fabs());
        }
    }
    result
}

/// Port of `void jacobiEigenSolver (Matrix33<T>& A, Vec3<T>& S, Matrix33<T>& V,
/// const T tol)`.
pub fn jacobi_eigen_solver_m33<T>(a: &mut Matrix33<T>, s: &mut Vec3<T>, v: &mut Matrix33<T>, tol: T)
where
    T: VecFloat + Math,
{
    v.make_identity();
    for i in 0..3 {
        s[i] = a[i][i];
    }

    let max_iter = 20;
    let abs_tol = tol * max_off_diag_symm_m33(a);
    if abs_tol != T::ZERO {
        let mut num_iter = 0;
        loop {
            // Z accumulates small diagonal changes (h) over one sweep to avoid
            // cancellation when h is tiny relative to the diagonal entry.
            let mut z = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            num_iter += 1;
            let mut changed = jacobi_rotation_m33(a, v, &mut z, 0, 1, 2, tol);
            changed = jacobi_rotation_m33(a, v, &mut z, 0, 2, 1, tol) || changed;
            changed = jacobi_rotation_m33(a, v, &mut z, 1, 2, 0, tol) || changed;
            // Add accumulated changes (Z) to the eigenvalues (S); update A's
            // diagonal for better accuracy too.
            for i in 0..3 {
                s[i] = s[i] + z[i];
                a[i][i] = s[i];
            }
            if !changed {
                break;
            }
            if !(max_off_diag_symm_m33(a) > abs_tol && num_iter < max_iter) {
                break;
            }
        }
    }
}

/// Port of the inline `jacobiEigenSolver (Matrix33, S, V)` — default tolerance.
pub fn jacobi_eigen_solver_m33_default<T>(a: &mut Matrix33<T>, s: &mut Vec3<T>, v: &mut Matrix33<T>)
where
    T: VecFloat + Math + VecLimits,
{
    jacobi_eigen_solver_m33(a, s, v, T::EPSILON);
}

/// Port of `void jacobiEigenSolver (Matrix44<T>& A, Vec4<T>& S, Matrix44<T>& V,
/// const T tol)`.
pub fn jacobi_eigen_solver_m44<T>(a: &mut Matrix44<T>, s: &mut Vec4<T>, v: &mut Matrix44<T>, tol: T)
where
    T: VecFloat + Math,
{
    v.make_identity();
    for i in 0..4 {
        s[i] = a[i][i];
    }

    let max_iter = 20;
    let abs_tol = tol * max_off_diag_symm_m44(a);
    if abs_tol != T::ZERO {
        let mut num_iter = 0;
        loop {
            num_iter += 1;
            let mut z = Vec4::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
            let mut changed = jacobi_rotation_m44(a, v, &mut z, 0, 1, 2, 3, tol);
            changed = jacobi_rotation_m44(a, v, &mut z, 0, 2, 1, 3, tol) || changed;
            changed = jacobi_rotation_m44(a, v, &mut z, 0, 3, 1, 2, tol) || changed;
            changed = jacobi_rotation_m44(a, v, &mut z, 1, 2, 0, 3, tol) || changed;
            changed = jacobi_rotation_m44(a, v, &mut z, 1, 3, 0, 2, tol) || changed;
            changed = jacobi_rotation_m44(a, v, &mut z, 2, 3, 0, 1, tol) || changed;
            for i in 0..4 {
                s[i] = s[i] + z[i];
                a[i][i] = s[i];
            }
            if !changed {
                break;
            }
            if !(max_off_diag_symm_m44(a) > abs_tol && num_iter < max_iter) {
                break;
            }
        }
    }
}

/// Port of the inline `jacobiEigenSolver (Matrix44, S, V)` — default tolerance.
pub fn jacobi_eigen_solver_m44_default<T>(a: &mut Matrix44<T>, s: &mut Vec4<T>, v: &mut Matrix44<T>)
where
    T: VecFloat + Math + VecLimits,
{
    jacobi_eigen_solver_m44(a, s, v, T::EPSILON);
}

/// Port of `void maxEigenVector (Matrix33<T>& A, Vec3<T>& V)`.
pub fn max_eigen_vector_m33<T>(a: &mut Matrix33<T>, out: &mut Vec3<T>)
where
    T: VecFloat + Math + VecLimits,
{
    let mut s = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut mv = Matrix33::default();
    jacobi_eigen_solver_m33_default(a, &mut s, &mut mv);

    let mut max_idx = 0usize;
    for i in 1..3 {
        if s[i].fabs() > s[max_idx].fabs() {
            max_idx = i;
        }
    }
    for i in 0..3 {
        out[i] = mv[i][max_idx];
    }
}

/// Port of `void maxEigenVector (Matrix44<T>& A, Vec4<T>& V)`.
pub fn max_eigen_vector_m44<T>(a: &mut Matrix44<T>, out: &mut Vec4<T>)
where
    T: VecFloat + Math + VecLimits,
{
    let mut s = Vec4::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    let mut mv = Matrix44::default();
    jacobi_eigen_solver_m44_default(a, &mut s, &mut mv);

    let mut max_idx = 0usize;
    for i in 1..4 {
        if s[i].fabs() > s[max_idx].fabs() {
            max_idx = i;
        }
    }
    for i in 0..4 {
        out[i] = mv[i][max_idx];
    }
}

/// Port of `void minEigenVector (Matrix33<T>& A, Vec3<T>& V)`.
pub fn min_eigen_vector_m33<T>(a: &mut Matrix33<T>, out: &mut Vec3<T>)
where
    T: VecFloat + Math + VecLimits,
{
    let mut s = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
    let mut mv = Matrix33::default();
    jacobi_eigen_solver_m33_default(a, &mut s, &mut mv);

    let mut min_idx = 0usize;
    for i in 1..3 {
        if s[i].fabs() < s[min_idx].fabs() {
            min_idx = i;
        }
    }
    for i in 0..3 {
        out[i] = mv[i][min_idx];
    }
}

/// Port of `void minEigenVector (Matrix44<T>& A, Vec4<T>& V)`.
pub fn min_eigen_vector_m44<T>(a: &mut Matrix44<T>, out: &mut Vec4<T>)
where
    T: VecFloat + Math + VecLimits,
{
    let mut s = Vec4::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    let mut mv = Matrix44::default();
    jacobi_eigen_solver_m44_default(a, &mut s, &mut mv);

    let mut min_idx = 0usize;
    for i in 1..4 {
        if s[i].fabs() < s[min_idx].fabs() {
            min_idx = i;
        }
    }
    for i in 0..4 {
        out[i] = mv[i][min_idx];
    }
}

// ============================================================================
// Small Vec2/Vec3 helpers (the foundation `Vec` types expose `dot`/`length`/
// `cross` but the C++ also uses `operator+/-/* /` componentwise; these wrappers
// keep the port readable without reshaping the foundation crate).
// ============================================================================

#[inline]
fn div_v3<T: Scalar>(v: Vec3<T>, s: T) -> Vec3<T> {
    Vec3::new(v.x / s, v.y / s, v.z / s)
}
#[inline]
fn mul_v3<T: Scalar>(v: Vec3<T>, s: T) -> Vec3<T> {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}
#[inline]
fn sub_v3<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}
#[inline]
fn add_v3<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

#[inline]
fn div_v2<T: Scalar>(v: Vec2<T>, s: T) -> Vec2<T> {
    Vec2::new(v.x / s, v.y / s)
}
#[inline]
fn mul_v2<T: Scalar>(v: Vec2<T>, s: T) -> Vec2<T> {
    Vec2::new(v.x * s, v.y * s)
}
#[inline]
fn sub_v2<T: Scalar>(a: Vec2<T>, b: Vec2<T>) -> Vec2<T> {
    Vec2::new(a.x - b.x, a.y - b.y)
}
#[inline]
fn normalize_v2<T: VecFloat>(mut v: Vec2<T>) -> Vec2<T> {
    v.normalize();
    v
}

#[inline]
fn vmax<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { b } else { a }
}

#[inline]
fn v3d_zero() -> V3d {
    V3d::new(0.0, 0.0, 0.0)
}

/// Convert a `Vec3<T>` to `V3d` (the C++ `(V3d) A[i]` cast).
#[inline]
fn to_v3d<T>(v: &Vec3<T>) -> V3d
where
    f64: VecCast<T>,
    T: Copy,
{
    V3d::new(f64::from_elem(v.x), f64::from_elem(v.y), f64::from_elem(v.z))
}

use crate::vec::V3d;

/// `v * M` for a 3-vector against a 3x3 matrix (the C++ `Vec3 * Matrix33` operator
/// used by procrustes for `Acenter * Qt`).
#[inline]
fn vec3_mul_m33d(v: &V3d, m: &M33d) -> V3d {
    V3d::new(
        v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0],
        v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1],
        v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2],
    )
}

#[cfg(test)]
mod tests;
