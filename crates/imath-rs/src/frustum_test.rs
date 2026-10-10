//! Port of Imath ImathFrustumTest.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathFrustumTest.h>
//!
//! Accelerated frustum visibility tests against pre-cached plane equations.
//! Designed for the common case of many tests against the same frustum.

use crate::box_::Box;
use crate::frustum::Frustum;
use crate::math::Math;
use crate::matrix::Matrix44;
use crate::plane::Plane3;
use crate::sphere::Sphere3;
use crate::traits::Scalar;
use crate::vec::{Vec3, VecFloat, VecLimits};

/// Accelerated frustum visibility tester (port of C++ `template <class T> class FrustumTest`).
///
/// Pre-computes and caches the six world-space plane equations in "transposed" form
/// (all X components grouped, all Y components grouped, etc.) to enable three
/// dot products at once via vector multiply-add.
///
/// # Usage
/// ```text
/// let ft = FrustumTest::new(my_frustum, camera_world_matrix);
/// // ... whenever camera moves:
/// ft.set_frustum(my_frustum, camera_world_matrix);
/// // per-object:
/// ft.is_visible_point(point);
/// ft.is_visible_sphere(sphere);
/// ft.is_visible_box(bbox);
/// ft.completely_contains_sphere(sphere);
/// ft.completely_contains_box(bbox);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FrustumTest<T> {
    // The six plane normals stored in "transpose" form: all X's together, etc.
    // planeNormX[0] = (planes[0].nx, planes[1].nx, planes[2].nx)
    // planeNormX[1] = (planes[3].nx, planes[4].nx, planes[5].nx)
    /// X components of the six plane normals, packed into two Vec3s.
    pub plane_norm_x: [Vec3<T>; 2],
    /// Y components of the six plane normals, packed into two Vec3s.
    pub plane_norm_y: [Vec3<T>; 2],
    /// Z components of the six plane normals, packed into two Vec3s.
    pub plane_norm_z: [Vec3<T>; 2],

    /// Distance offsets from the six plane equations, packed into two Vec3s.
    pub plane_offset_vec: [Vec3<T>; 2],

    /// Absolute values of X components (for bbox tests).
    pub plane_norm_abs_x: [Vec3<T>; 2],
    /// Absolute values of Y components (for bbox tests).
    pub plane_norm_abs_y: [Vec3<T>; 2],
    /// Absolute values of Z components (for bbox tests).
    pub plane_norm_abs_z: [Vec3<T>; 2],

    /// Stored frustum (for debugging / re-use).
    pub curr_frustum: Frustum<T>,
    /// Stored camera matrix (for debugging / re-use).
    pub camera_matrix: Matrix44<T>,
}

impl<T: Scalar + VecFloat + VecLimits + Math> FrustumTest<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of `FrustumTest()` — initializes with a default frustum and identity camera matrix.
    pub fn new() -> Self {
        let frust = Frustum::<T>::new();
        let mut cam = Matrix44::<T>::default(); // identity
        cam.make_identity();
        let mut ft = Self::zeroed();
        ft.set_frustum(&frust, &cam);
        ft
    }

    /// Port of `FrustumTest(const Frustum<T>&, const Matrix44<T>&)`.
    pub fn from_frustum(frustum: &Frustum<T>, camera_mat: &Matrix44<T>) -> Self {
        let mut ft = Self::zeroed();
        ft.set_frustum(frustum, camera_mat);
        ft
    }

    // -----------------------------------------------------------------------
    // Set
    // -----------------------------------------------------------------------

    /// Port of `setFrustum(const Frustum<T>&, const Matrix44<T>&)`.
    ///
    /// Recomputes all cached plane data from the frustum and camera matrix.
    /// This should be called once per frame (or whenever the camera moves).
    pub fn set_frustum(&mut self, frustum: &Frustum<T>, camera_mat: &Matrix44<T>) {
        let mut frustum_planes: [Plane3<T>; 6] = [Plane3::new(); 6];
        frustum.planes_transformed(&mut frustum_planes, camera_mat);

        // Transpose the plane equations into two groups of three.
        // C++: for (int i = 0; i < 2; ++i) { int index = i*3; ... }
        for i in 0..2usize {
            let idx = i * 3;

            self.plane_norm_x[i] = Vec3::new(
                frustum_planes[idx].normal.x,
                frustum_planes[idx + 1].normal.x,
                frustum_planes[idx + 2].normal.x,
            );
            self.plane_norm_y[i] = Vec3::new(
                frustum_planes[idx].normal.y,
                frustum_planes[idx + 1].normal.y,
                frustum_planes[idx + 2].normal.y,
            );
            self.plane_norm_z[i] = Vec3::new(
                frustum_planes[idx].normal.z,
                frustum_planes[idx + 1].normal.z,
                frustum_planes[idx + 2].normal.z,
            );

            // Absolute values for bbox tests
            self.plane_norm_abs_x[i] = Vec3::new(
                abs_t(self.plane_norm_x[i].x),
                abs_t(self.plane_norm_x[i].y),
                abs_t(self.plane_norm_x[i].z),
            );
            self.plane_norm_abs_y[i] = Vec3::new(
                abs_t(self.plane_norm_y[i].x),
                abs_t(self.plane_norm_y[i].y),
                abs_t(self.plane_norm_y[i].z),
            );
            self.plane_norm_abs_z[i] = Vec3::new(
                abs_t(self.plane_norm_z[i].x),
                abs_t(self.plane_norm_z[i].y),
                abs_t(self.plane_norm_z[i].z),
            );

            self.plane_offset_vec[i] = Vec3::new(
                frustum_planes[idx].distance,
                frustum_planes[idx + 1].distance,
                frustum_planes[idx + 2].distance,
            );
        }

        self.curr_frustum = *frustum;
        self.camera_matrix = *camera_mat;
    }

    // -----------------------------------------------------------------------
    // Visibility queries
    // -----------------------------------------------------------------------

    /// Port of `isVisible(const Vec3<T>& vec)`.
    ///
    /// Returns true if the point is inside the frustum (exact test).
    pub fn is_visible_point(&self, vec: Vec3<T>) -> bool {
        // Vertical dot-product on first three planes
        let d0 = self.plane_norm_x[0] * vec.x
            + self.plane_norm_y[0] * vec.y
            + self.plane_norm_z[0] * vec.z
            - self.plane_offset_vec[0];
        if d0.x >= T::ZERO || d0.y >= T::ZERO || d0.z >= T::ZERO {
            return false;
        }

        // Vertical dot-product on second three planes
        let d1 = self.plane_norm_x[1] * vec.x
            + self.plane_norm_y[1] * vec.y
            + self.plane_norm_z[1] * vec.z
            - self.plane_offset_vec[1];
        if d1.x >= T::ZERO || d1.y >= T::ZERO || d1.z >= T::ZERO {
            return false;
        }

        true
    }

    /// Port of `isVisible(const Sphere3<T>& sphere)`.
    ///
    /// Returns true if any part of the sphere is inside the frustum.
    /// May return false-positives but never false-negatives.
    pub fn is_visible_sphere(&self, sphere: &Sphere3<T>) -> bool {
        let center = sphere.center;
        let rv = Vec3::new(sphere.radius, sphere.radius, sphere.radius);

        // C++: d0 = normX[0]*cx + normY[0]*cy + normZ[0]*cz - radiusVec - offsetVec[0]
        let d0 = self.plane_norm_x[0] * center.x
            + self.plane_norm_y[0] * center.y
            + self.plane_norm_z[0] * center.z
            - rv
            - self.plane_offset_vec[0];
        if d0.x >= T::ZERO || d0.y >= T::ZERO || d0.z >= T::ZERO {
            return false;
        }

        let d1 = self.plane_norm_x[1] * center.x
            + self.plane_norm_y[1] * center.y
            + self.plane_norm_z[1] * center.z
            - rv
            - self.plane_offset_vec[1];
        if d1.x >= T::ZERO || d1.y >= T::ZERO || d1.z >= T::ZERO {
            return false;
        }

        true
    }

    /// Port of `isVisible(const Box<Vec3<T>>& box)`.
    ///
    /// Returns true if any part of the axis-aligned bounding box is inside the frustum.
    /// May return false-positives but never false-negatives.
    pub fn is_visible_box(&self, b: &Box<Vec3<T>>) -> bool {
        if b.is_empty() {
            return false;
        }

        let two = T::ONE + T::ONE;
        let center = (b.min + b.max) * (T::ONE / two);
        let extent = b.max - center;

        // C++: d0 = normX*cx + normY*cy + normZ*cz - absNormX*ex - absNormY*ey - absNormZ*ez - offsetVec
        let d0 = self.plane_norm_x[0] * center.x
            + self.plane_norm_y[0] * center.y
            + self.plane_norm_z[0] * center.z
            - self.plane_norm_abs_x[0] * extent.x
            - self.plane_norm_abs_y[0] * extent.y
            - self.plane_norm_abs_z[0] * extent.z
            - self.plane_offset_vec[0];
        if d0.x >= T::ZERO || d0.y >= T::ZERO || d0.z >= T::ZERO {
            return false;
        }

        let d1 = self.plane_norm_x[1] * center.x
            + self.plane_norm_y[1] * center.y
            + self.plane_norm_z[1] * center.z
            - self.plane_norm_abs_x[1] * extent.x
            - self.plane_norm_abs_y[1] * extent.y
            - self.plane_norm_abs_z[1] * extent.z
            - self.plane_offset_vec[1];
        if d1.x >= T::ZERO || d1.y >= T::ZERO || d1.z >= T::ZERO {
            return false;
        }

        true
    }

    /// Port of `completelyContains(const Sphere3<T>& sphere)`.
    ///
    /// Returns true if the entire sphere is inside the frustum.
    /// May return false-negatives but never false-positives.
    pub fn completely_contains_sphere(&self, sphere: &Sphere3<T>) -> bool {
        let center = sphere.center;
        let rv = Vec3::new(sphere.radius, sphere.radius, sphere.radius);

        // C++: d0 = normX*cx + normY*cy + normZ*cz + radiusVec - offsetVec  (note: + radius)
        let d0 = self.plane_norm_x[0] * center.x
            + self.plane_norm_y[0] * center.y
            + self.plane_norm_z[0] * center.z
            + rv
            - self.plane_offset_vec[0];
        if d0.x >= T::ZERO || d0.y >= T::ZERO || d0.z >= T::ZERO {
            return false;
        }

        let d1 = self.plane_norm_x[1] * center.x
            + self.plane_norm_y[1] * center.y
            + self.plane_norm_z[1] * center.z
            + rv
            - self.plane_offset_vec[1];
        if d1.x >= T::ZERO || d1.y >= T::ZERO || d1.z >= T::ZERO {
            return false;
        }

        true
    }

    /// Port of `completelyContains(const Box<Vec3<T>>& box)`.
    ///
    /// Returns true if the entire axis-aligned bounding box is inside the frustum.
    /// May return false-negatives but never false-positives.
    pub fn completely_contains_box(&self, b: &Box<Vec3<T>>) -> bool {
        if b.is_empty() {
            return false;
        }

        let two = T::ONE + T::ONE;
        let center = (b.min + b.max) * (T::ONE / two);
        let extent = b.max - center;

        // C++: d0 = normX*cx + normY*cy + normZ*cz + absNormX*ex + absNormY*ey + absNormZ*ez - offsetVec
        let d0 = self.plane_norm_x[0] * center.x
            + self.plane_norm_y[0] * center.y
            + self.plane_norm_z[0] * center.z
            + self.plane_norm_abs_x[0] * extent.x
            + self.plane_norm_abs_y[0] * extent.y
            + self.plane_norm_abs_z[0] * extent.z
            - self.plane_offset_vec[0];
        if d0.x >= T::ZERO || d0.y >= T::ZERO || d0.z >= T::ZERO {
            return false;
        }

        let d1 = self.plane_norm_x[1] * center.x
            + self.plane_norm_y[1] * center.y
            + self.plane_norm_z[1] * center.z
            + self.plane_norm_abs_x[1] * extent.x
            + self.plane_norm_abs_y[1] * extent.y
            + self.plane_norm_abs_z[1] * extent.z
            - self.plane_offset_vec[1];
        if d1.x >= T::ZERO || d1.y >= T::ZERO || d1.z >= T::ZERO {
            return false;
        }

        true
    }

    // -----------------------------------------------------------------------
    // Accessors (for debugging)
    // -----------------------------------------------------------------------

    /// Port of `cameraMat()` — return the camera matrix (for debugging).
    #[inline]
    pub fn camera_mat(&self) -> Matrix44<T> {
        self.camera_matrix
    }

    /// Port of `currentFrustum()` — return the stored frustum (for debugging).
    #[inline]
    pub fn current_frustum(&self) -> Frustum<T> {
        self.curr_frustum
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Construct a zeroed instance (all plane vectors = zero, frustum = default, matrix = identity).
    fn zeroed() -> Self {
        let z = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
        FrustumTest {
            plane_norm_x: [z, z],
            plane_norm_y: [z, z],
            plane_norm_z: [z, z],
            plane_offset_vec: [z, z],
            plane_norm_abs_x: [z, z],
            plane_norm_abs_y: [z, z],
            plane_norm_abs_z: [z, z],
            curr_frustum: Frustum::new(),
            camera_matrix: Matrix44::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> Default for FrustumTest<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal scalar abs helper (no std::abs — uses T arithmetic)
// ---------------------------------------------------------------------------

#[inline]
fn abs_t<T: Scalar>(v: T) -> T {
    if v >= T::ZERO { v } else { -v }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathFrustumTest.h)
// ---------------------------------------------------------------------------

/// FrustumTest of type `f32` (C++ `FrustumTestf`).
pub type FrustumTestf = FrustumTest<f32>;
/// FrustumTest of type `f64` (C++ `FrustumTestd`).
pub type FrustumTestd = FrustumTest<f64>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_::Box;
    use crate::frustum::Frustum;
    use crate::sphere::Sphere3;
    use crate::vec::Vec3;

    /// Build a standard perspective frustum (identity camera) and verify:
    /// - A point clearly inside the frustum is visible.
    /// - A point behind the far plane is not visible.
    #[test]
    fn frustum_test_point_inside_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Point at (0, 0, -10): well within the frustum
        let inside = Vec3::new(0.0f32, 0.0, -10.0);
        assert!(ft.is_visible_point(inside), "inside point should be visible");
    }

    #[test]
    fn frustum_test_point_behind_far_not_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Point at z = -200: behind the far plane
        let behind_far = Vec3::new(0.0f32, 0.0, -200.0);
        assert!(
            !ft.is_visible_point(behind_far),
            "point behind far should not be visible"
        );
    }

    #[test]
    fn frustum_test_point_in_front_of_near_not_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Point at z = -0.5: in front of (closer than) the near plane
        let before_near = Vec3::new(0.0f32, 0.0, -0.5);
        assert!(
            !ft.is_visible_point(before_near),
            "point in front of near should not be visible"
        );
    }

    /// Sphere straddling the near plane: partially inside → isVisible = true.
    #[test]
    fn frustum_test_sphere_straddling_near_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Sphere centered at z = -1 (on the near plane) with radius 1: straddles near plane
        let sphere = Sphere3::from_center_radius(Vec3::new(0.0f32, 0.0, -1.0), 1.0);
        assert!(
            ft.is_visible_sphere(&sphere),
            "sphere straddling near plane should be visible"
        );
    }

    /// Sphere entirely outside (far side).
    #[test]
    fn frustum_test_sphere_outside_far_not_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Sphere centered far behind the far plane, entirely outside
        let sphere = Sphere3::from_center_radius(Vec3::new(0.0f32, 0.0, -300.0), 1.0);
        assert!(
            !ft.is_visible_sphere(&sphere),
            "sphere behind far should not be visible"
        );
    }

    /// Box fully inside the frustum.
    #[test]
    fn frustum_test_box_inside_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -10.0, 10.0, 10.0, -10.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Tiny box well inside the frustum
        let b = Box::<Vec3<f32>> {
            min: Vec3::new(-0.5f32, -0.5, -5.0),
            max: Vec3::new(0.5f32, 0.5, -2.0),
        };
        assert!(ft.is_visible_box(&b), "box inside frustum should be visible");
    }

    /// Box fully outside (behind far).
    #[test]
    fn frustum_test_box_outside_far_not_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        let b = Box::<Vec3<f32>> {
            min: Vec3::new(-1.0f32, -1.0, -500.0),
            max: Vec3::new(1.0f32, 1.0, -200.0),
        };
        assert!(
            !ft.is_visible_box(&b),
            "box behind far plane should not be visible"
        );
    }

    /// completelyContains: sphere clearly inside should return true.
    #[test]
    fn frustum_test_completely_contains_sphere_inside() {
        // Wide frustum so the sphere fits entirely
        let f = Frustum::<f32>::from_params(1.0, 100.0, -50.0, 50.0, 50.0, -50.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        let sphere = Sphere3::from_center_radius(Vec3::new(0.0f32, 0.0, -10.0), 0.1);
        assert!(
            ft.completely_contains_sphere(&sphere),
            "tiny sphere in wide frustum should be completely contained"
        );
    }

    /// completelyContains: large sphere straddling a plane should return false.
    #[test]
    fn frustum_test_completely_contains_sphere_straddling_false() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // Sphere straddling the far plane
        let sphere = Sphere3::from_center_radius(Vec3::new(0.0f32, 0.0, -99.0), 5.0);
        assert!(
            !ft.completely_contains_sphere(&sphere),
            "sphere straddling far plane should not be completely contained"
        );
    }

    /// completelyContains: small box clearly inside a very wide frustum.
    #[test]
    fn frustum_test_completely_contains_box_inside() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -50.0, 50.0, 50.0, -50.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        let b = Box::<Vec3<f32>> {
            min: Vec3::new(-0.1f32, -0.1, -5.0),
            max: Vec3::new(0.1f32, 0.1, -2.0),
        };
        assert!(
            ft.completely_contains_box(&b),
            "tiny box in wide frustum should be completely contained"
        );
    }

    /// Empty box should not be visible.
    #[test]
    fn frustum_test_empty_box_not_visible() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let ft = FrustumTest::from_frustum(&f, &cam);

        // An empty box (min > max): construct one manually via make_empty()
        let mut b = Box::<Vec3<f32>> {
            min: Vec3::new(0.0f32, 0.0, 0.0),
            max: Vec3::new(0.0f32, 0.0, 0.0),
        };
        b.make_empty();
        assert!(!ft.is_visible_box(&b), "empty box should not be visible");
        assert!(
            !ft.completely_contains_box(&b),
            "empty box should not be completely contained"
        );
    }

    /// setFrustum updates the cached planes: after moving camera, visibility changes.
    #[test]
    fn frustum_test_set_frustum_updates_planes() {
        let f = Frustum::<f32>::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let cam = Matrix44::<f32>::default();
        let mut ft = FrustumTest::from_frustum(&f, &cam);

        // Point visible with identity camera
        let pt = Vec3::new(0.0f32, 0.0, -10.0);
        assert!(ft.is_visible_point(pt));

        // Translate camera so the frustum no longer covers pt
        // Move camera to (0, 0, -500) — translate world by (0, 0, +500) in camera space
        let mut cam2 = Matrix44::<f32>::default();
        cam2.translate(&Vec3::new(0.0f32, 0.0, 500.0));
        ft.set_frustum(&f, &cam2);

        // Now pt=(0,0,-10) in world space is well in front of near plane from camera's POV
        // (camera is at z=-500, near=1, so pt is at z=490 relative to camera → in front of near)
        assert!(
            !ft.is_visible_point(pt),
            "point should not be visible after camera move"
        );
    }

    /// Default constructor smoke test.
    #[test]
    fn frustum_test_default() {
        let ft = FrustumTestf::new();
        // Just verify it doesn't panic and the camera matrix is identity
        let m = ft.camera_mat();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                assert!((m.x[i][j] - expected).abs() < 1e-6, "m[{i}][{j}]={}", m.x[i][j]);
            }
        }
    }
}
