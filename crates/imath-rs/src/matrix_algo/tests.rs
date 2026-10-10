//! Unit tests for `matrix_algo` — verify the ported algorithms against KNOWN
//! values (decomposition round-trips, Euler extraction, rotation/align, the
//! Procrustes rigid-fit, and the symmetric Jacobi eigen-solver).

use super::*;

const PI: f64 = core::f64::consts::PI;

/// Build M = S * R * T (Imath composes via successive in-place ops) with no shear.
fn build_srt(s: Vec3<f64>, rot: Vec3<f64>, t: Vec3<f64>) -> Matrix44<f64> {
    // Match the inverse of sansScaling's reconstruction order: translate, rotate,
    // scale — which yields a matrix whose extractSHRT recovers (s, 0, rot, t).
    let mut m = Matrix44::<f64>::default();
    m.translate(&t);
    m.rotate(&rot);
    m.scale(&s);
    m
}

#[test]
fn extract_shrt_recovers_components() {
    let s = Vec3::new(2.0, 3.0, 0.5);
    let rot = Vec3::new(0.3, -0.4, 0.7);
    let t = Vec3::new(5.0, -2.0, 11.0);
    let m = build_srt(s, rot, t);

    let mut os = Vec3::new(0.0, 0.0, 0.0);
    let mut oh = Vec3::new(0.0, 0.0, 0.0);
    let mut orr = Vec3::new(0.0, 0.0, 0.0);
    let mut ot = Vec3::new(0.0, 0.0, 0.0);
    let ok = extract_shrt_m44_vec(&m, &mut os, &mut oh, &mut orr, &mut ot, false).unwrap();
    assert!(ok);

    assert!((os.x - s.x).abs() < 1e-9, "sx {} vs {}", os.x, s.x);
    assert!((os.y - s.y).abs() < 1e-9);
    assert!((os.z - s.z).abs() < 1e-9);

    // Shear must be ~0.
    assert!(oh.x.abs() < 1e-9 && oh.y.abs() < 1e-9 && oh.z.abs() < 1e-9);

    // Rotation recovered (XYZ euler).
    assert!((orr.x - rot.x).abs() < 1e-9, "rx {} vs {}", orr.x, rot.x);
    assert!((orr.y - rot.y).abs() < 1e-9);
    assert!((orr.z - rot.z).abs() < 1e-9);

    // Translation recovered.
    assert!((ot.x - t.x).abs() < 1e-9);
    assert!((ot.y - t.y).abs() < 1e-9);
    assert!((ot.z - t.z).abs() < 1e-9);
}

#[test]
fn extract_scaling_recovers_scale() {
    let s = Vec3::new(4.0, 0.25, 1.5);
    let m = build_srt(s, Vec3::new(0.2, 0.1, -0.3), Vec3::new(0.0, 0.0, 0.0));
    let mut scl = Vec3::new(0.0, 0.0, 0.0);
    assert!(extract_scaling_m44(&m, &mut scl, false).unwrap());
    assert!((scl.x - s.x).abs() < 1e-9);
    assert!((scl.y - s.y).abs() < 1e-9);
    assert!((scl.z - s.z).abs() < 1e-9);
}

#[test]
fn extract_and_remove_leaves_pure_rotation() {
    let s = Vec3::new(2.0, 3.0, 0.5);
    let rot = Vec3::new(0.3, -0.4, 0.7);
    let mut m = build_srt(s, rot, Vec3::new(0.0, 0.0, 0.0));
    let mut scl = Vec3::new(0.0, 0.0, 0.0);
    let mut shr = Vec3::new(0.0, 0.0, 0.0);
    assert!(extract_and_remove_scaling_and_shear_m44(&mut m, &mut scl, &mut shr, false).unwrap());

    // The upper-left 3x3 of m is now a rotation matrix: rows orthonormal, det +1.
    let r0 = Vec3::new(m[0][0], m[0][1], m[0][2]);
    let r1 = Vec3::new(m[1][0], m[1][1], m[1][2]);
    let r2 = Vec3::new(m[2][0], m[2][1], m[2][2]);
    assert!((r0.length() - 1.0).abs() < 1e-9);
    assert!((r1.length() - 1.0).abs() < 1e-9);
    assert!((r2.length() - 1.0).abs() < 1e-9);
    assert!(r0.dot(&r1).abs() < 1e-9);
    assert!(r0.dot(&r2).abs() < 1e-9);
    let det = r0.dot(&r1.cross(&r2));
    assert!((det - 1.0).abs() < 1e-9);
}

#[test]
fn extract_euler_xyz_known_rotation() {
    // Pure rotation of 90 deg about Z.
    let mut m = Matrix44::<f64>::default();
    m.rotate(&Vec3::new(0.0, 0.0, PI / 2.0));
    let mut rot = Vec3::new(0.0, 0.0, 0.0);
    extract_euler_xyz(&m, &mut rot);
    assert!(rot.x.abs() < 1e-9);
    assert!(rot.y.abs() < 1e-9);
    assert!((rot.z - PI / 2.0).abs() < 1e-9, "rz = {}", rot.z);
}

#[test]
fn rotation_matrix_maps_from_to() {
    let from = Vec3::new(1.0_f64, 0.0, 0.0);
    let to = Vec3::new(0.0_f64, 1.0, 0.0);
    let m = rotation_matrix(&from, &to);
    // Imath multiplies row-vectors on the left: v' = v * M.
    let mut dst = Vec3::new(0.0, 0.0, 0.0);
    m.mult_dir_matrix(&from, &mut dst);
    assert!((dst.x - to.x).abs() < 1e-9, "{:?}", dst);
    assert!((dst.y - to.y).abs() < 1e-9);
    assert!((dst.z - to.z).abs() < 1e-9);
}

#[test]
fn align_z_axis_points_z_at_target() {
    let target = Vec3::new(1.0, 1.0, 0.0);
    let up = Vec3::new(0.0, 0.0, 1.0);
    let mut m = Matrix44::<f64>::default();
    align_z_axis_with_target_dir(&mut m, target, up);
    // Row 2 is the z-axis = normalized target.
    let z = Vec3::new(m[2][0], m[2][1], m[2][2]);
    let tn = target.normalized();
    assert!((z.x - tn.x).abs() < 1e-9);
    assert!((z.y - tn.y).abs() < 1e-9);
    assert!((z.z - tn.z).abs() < 1e-9);
}

#[test]
fn procrustes_recovers_rigid_transform() {
    // A known rotation (about Z by 30 deg) + translation applied to a point set.
    let angle = PI / 6.0;
    let (cs, sn) = (angle.cos(), angle.sin());
    let t = Vec3::new(10.0, -5.0, 3.0);

    let from = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(2.0, 3.0, -1.0),
        Vec3::new(-4.0, 1.0, 2.0),
    ];
    // Apply rotation about Z then translate, in column-vector convention:
    // p' = R*p + t. R (column-vector) about Z:
    //   x' = cs*x - sn*y; y' = sn*x + cs*y; z' = z
    let apply = |p: Vec3<f64>| {
        Vec3::new(cs * p.x - sn * p.y + t.x, sn * p.x + cs * p.y + t.y, p.z + t.z)
    };
    let to: Vec<Vec3<f64>> = from.iter().map(|p| apply(*p)).collect();

    let m = procrustes_rotation_and_translation(&from, &to, from.len(), false);

    // The returned M44d maps from -> to under Imath row-vector convention
    // (v' = v * M). Verify each point.
    for (f, expected) in from.iter().zip(to.iter()) {
        let mut dst = Vec3::new(0.0, 0.0, 0.0);
        m.mult_vec_matrix(f, &mut dst);
        assert!((dst.x - expected.x).abs() < 1e-6, "x {} vs {}", dst.x, expected.x);
        assert!((dst.y - expected.y).abs() < 1e-6, "y {} vs {}", dst.y, expected.y);
        assert!((dst.z - expected.z).abs() < 1e-6, "z {} vs {}", dst.z, expected.z);
    }
}

#[test]
fn jacobi_eigen_solver_known_symmetric() {
    // Diagonal matrix: eigenvalues are the diagonal, eigenvectors the axes.
    let mut a = Matrix33::new(3.0_f64, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0);
    let mut s = Vec3::new(0.0, 0.0, 0.0);
    let mut v = Matrix33::default();
    jacobi_eigen_solver_m33(&mut a, &mut s, &mut v, f64::EPSILON);

    // For a diagonal input, no sweeps run (off-diagonals already zero), so S is
    // the diagonal directly.
    assert!((s.x - 3.0).abs() < 1e-9);
    assert!((s.y - 5.0).abs() < 1e-9);
    assert!((s.z - 7.0).abs() < 1e-9);

    // Non-trivial symmetric matrix: A = V * diag(S) * V^T must hold.
    let mut b = Matrix33::new(2.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0);
    let orig = b;
    let mut s2 = Vec3::new(0.0, 0.0, 0.0);
    let mut v2 = Matrix33::default();
    jacobi_eigen_solver_m33(&mut b, &mut s2, &mut v2, f64::EPSILON);

    // Reconstruct V * diag(S) * V^T and compare with original.
    let mut diag = Matrix33::<f64>::from_scalar(0.0);
    diag[0][0] = s2.x;
    diag[1][1] = s2.y;
    diag[2][2] = s2.z;
    let recon = v2.mul_matrix(&diag).mul_matrix(&v2.transposed());
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (recon[i][j] - orig[i][j]).abs() < 1e-9,
                "recon[{}][{}] = {} vs {}",
                i,
                j,
                recon[i][j],
                orig[i][j]
            );
        }
    }

    // Known eigenvalues of [[2,1,0],[1,2,1],[0,1,2]] are 2, 2±sqrt(2).
    let mut evals = [s2.x, s2.y, s2.z];
    evals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let expected = [2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()];
    for (got, exp) in evals.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-9, "eig {} vs {}", got, exp);
    }
}

#[test]
fn jacobi_svd_m33_reconstructs() {
    let a = Matrix33::new(1.0_f64, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0);
    let mut u = Matrix33::default();
    let mut s = Vec3::new(0.0, 0.0, 0.0);
    let mut v = Matrix33::default();
    jacobi_svd_m33(&a, &mut u, &mut s, &mut v, f64::EPSILON, false);

    // A = U * diag(S) * V^T.
    let mut diag = Matrix33::<f64>::from_scalar(0.0);
    diag[0][0] = s.x;
    diag[1][1] = s.y;
    diag[2][2] = s.z;
    let recon = u.mul_matrix(&diag).mul_matrix(&v.transposed());
    for i in 0..3 {
        for j in 0..3 {
            assert!((recon[i][j] - a[i][j]).abs() < 1e-9, "{} {} {} {}", i, j, recon[i][j], a[i][j]);
        }
    }
    // Singular values sorted descending and positive.
    assert!(s.x >= s.y && s.y >= s.z);
    assert!(s.z >= 0.0);
}

#[test]
fn extract_euler_m33_known() {
    let mut m = Matrix33::<f64>::default();
    m.rotate(0.5);
    let mut rot = 0.0;
    extract_euler_m33(&m, &mut rot);
    assert!((rot - 0.5).abs() < 1e-9, "rot = {}", rot);
}

#[test]
fn check_zero_scale_exc_vs_nonexc() {
    // Healthy scale: ok.
    let row = Vec3::new(1.0_f64, 0.0, 0.0);
    assert_eq!(check_for_zero_scale_in_row_v3(1.0, &row, true), Ok(true));
    // Degenerate: scl tiny, row entry huge -> overflow path.
    let bad = Vec3::new(f64::MAX, 0.0, 0.0);
    assert!(check_for_zero_scale_in_row_v3(1e-300, &bad, true).is_err());
    assert_eq!(check_for_zero_scale_in_row_v3(1e-300, &bad, false), Ok(false));
}

#[test]
fn outer_product_v3_known() {
    let a = Vec3::new(1.0_f64, 2.0, 3.0);
    let b = Vec3::new(4.0_f64, 5.0, 6.0);
    let m = outer_product_v3(&a, &b);
    assert_eq!(m[0][0], 4.0);
    assert_eq!(m[0][1], 5.0);
    assert_eq!(m[2][2], 18.0);
    assert_eq!(m[1][0], 8.0);
}

#[test]
fn compute_local_frame_orthonormal() {
    let p = Vec3::new(1.0_f64, 2.0, 3.0);
    let x_dir = Vec3::new(1.0, 0.0, 0.0);
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let l = compute_local_frame(&p, &x_dir, &normal);
    // Position row.
    assert_eq!(l[3][0], 1.0);
    assert_eq!(l[3][1], 2.0);
    assert_eq!(l[3][2], 3.0);
    // Rows orthonormal.
    let r0 = Vec3::new(l[0][0], l[0][1], l[0][2]);
    let r1 = Vec3::new(l[1][0], l[1][1], l[1][2]);
    let r2 = Vec3::new(l[2][0], l[2][1], l[2][2]);
    assert!((r0.length() - 1.0).abs() < 1e-12);
    assert!((r1.length() - 1.0).abs() < 1e-12);
    assert!((r2.length() - 1.0).abs() < 1e-12);
    assert!(r0.dot(&r1).abs() < 1e-12);
}

/// `extractSHRT(mat, s, h, Euler<T>& r, t, exc)` forwards the caller's Euler as
/// the rotation out-param (`ImathMatrixAlgo.h:870-881`). On the `return false`
/// path — a singular upper 3x3, `exc == false` — C++ returns before
/// `extractEulerXYZ` ever writes it, so the caller's angles survive.
/// This port used to zero them through a scratch vector.
///
/// The bit-exact cross-check against real C++ Imath lives in the parity golden
/// (`shrt.fail.ok` / `shrt.fail.r.*`).
#[test]
fn extract_shrt_euler_leaves_angles_untouched_on_failure() {
    use crate::euler::{Euler, InputLayout, Order};

    // Row 0 is all zero => scl.x == 0 => checkForZeroScaleInRow fails.
    let singular = Matrix44::<f64>::new(
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let angles = Vec3::new(0.25_f64, -0.5, 0.75);
    let mut r = Euler::from_vec_order_layout(angles, Order::XYZ, InputLayout::IJKLayout);
    let mut s = Vec3::new(0.0_f64, 0.0, 0.0);
    let mut h = Vec3::new(0.0_f64, 0.0, 0.0);
    let mut t = Vec3::new(0.0_f64, 0.0, 0.0);

    let ok = extract_shrt_m44_euler(&singular, &mut s, &mut h, &mut r, &mut t, false)
        .expect("exc = false never returns Err");
    assert!(!ok, "a singular matrix must report failure");
    assert_eq!(r.vec.x, angles.x);
    assert_eq!(r.vec.y, angles.y);
    assert_eq!(r.vec.z, angles.z);
    assert_eq!(r.order(), Order::XYZ, "the Euler's order must be preserved too");
}
