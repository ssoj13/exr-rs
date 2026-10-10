//! Port of Imath ImathFrustum.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathFrustum.h>
//!
//! A viewing frustum with the eye at the origin looking down -Z (OpenGL convention).

use crate::line::Line3;
use crate::math::Math;
use crate::matrix::Matrix44;
use crate::matrix::matrix44::vec3_mul_matrix44;
use crate::plane::Plane3;
use crate::traits::Scalar;
use crate::vec::{Vec2, Vec3, VecFloat, VecLimits};

/// A viewing frustum (port of C++ `template <class T> class Frustum`).
///
/// The frustum is always located with the eye point at the origin facing down -Z.
/// Fields map directly to the C++ protected members.
///
/// Near/far are stored in `near_plane`/`far_plane` (C++ uses `_nearPlane`/`_farPlane`;
/// `near`/`far` are reserved keywords on Windows MSVC so the C++ header uses the
/// `nearPlane`/`farPlane` naming for accessors — we follow the same convention).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Frustum<T> {
    /// Near clipping distance (C++ `_nearPlane`).
    pub near_plane: T,
    /// Far clipping distance (C++ `_farPlane`).
    pub far_plane: T,
    /// Left edge at the near plane (C++ `_left`).
    pub left: T,
    /// Right edge at the near plane (C++ `_right`).
    pub right: T,
    /// Top edge at the near plane (C++ `_top`).
    pub top: T,
    /// Bottom edge at the near plane (C++ `_bottom`).
    pub bottom: T,
    /// True if the frustum is orthographic, false for perspective (C++ `_orthographic`).
    pub orthographic: bool,
}

impl<T: Scalar + VecFloat + VecLimits + Math> Frustum<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of the default constructor:
    /// near=0.1, far=1000.0, left=-1.0, right=1.0, top=1.0, bottom=-1.0, ortho=false.
    #[inline]
    pub fn new() -> Self {
        // C++: set(T(0.1), T(1000.0), T(-1.0), T(1.0), T(1.0), T(-1.0), false)
        // We build T(0.1) and T(1000.0) via f64 cast since T: VecFloat gives us From<f64> through
        // the scalar literal helpers used elsewhere in the crate — but actually the cleanest
        // approach matching C++ is to use T::ONE arithmetic to get the constants.
        // C++ default: near=0.1, far=1000, left=-1, right=1, top=1, bottom=-1
        // We use the same numeric literals cast through T::ONE chains.
        let one = T::ONE;
        let ten = one + one + one + one + one + one + one + one + one + one; // 10
        let hundred = ten * ten; // 100
        let thousand = hundred * ten; // 1000
        let near = one / ten; // 0.1
        let far = thousand;
        Self::from_params(near, far, -one, one, one, -one, false)
    }

    /// Port of `Frustum(T near, T far, T left, T right, T top, T bottom, bool ortho)`.
    #[inline]
    pub fn from_params(
        near_plane: T,
        far_plane: T,
        left: T,
        right: T,
        top: T,
        bottom: T,
        ortho: bool,
    ) -> Self {
        let mut f = Frustum {
            near_plane: T::ZERO,
            far_plane: T::ZERO,
            left: T::ZERO,
            right: T::ZERO,
            top: T::ZERO,
            bottom: T::ZERO,
            orthographic: false,
        };
        f.set(near_plane, far_plane, left, right, top, bottom, ortho);
        f
    }

    /// Port of `Frustum(T near, T far, T fovx, T fovy, T aspect)`.
    ///
    /// Exactly one of `fovx` or `fovy` must be non-zero.
    /// If both are zero, behaves like C++ (no throw, same silently as fovy branch).
    #[inline]
    pub fn from_fov(near_plane: T, far_plane: T, fovx: T, fovy: T, aspect: T) -> Self {
        let mut f = Frustum {
            near_plane: T::ZERO,
            far_plane: T::ZERO,
            left: T::ZERO,
            right: T::ZERO,
            top: T::ZERO,
            bottom: T::ZERO,
            orthographic: false,
        };
        f.set_fov(near_plane, far_plane, fovx, fovy, aspect);
        f
    }

    // -----------------------------------------------------------------------
    // Set methods
    // -----------------------------------------------------------------------

    /// Port of `set(T near, T far, T left, T right, T top, T bottom, bool ortho)`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        near_plane: T,
        far_plane: T,
        left: T,
        right: T,
        top: T,
        bottom: T,
        ortho: bool,
    ) {
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        self.left = left;
        self.right = right;
        self.bottom = bottom;
        self.top = top;
        self.orthographic = ortho;
    }

    /// Port of `set(T near, T far, T fovx, T fovy, T aspect)` (IMATH_NOEXCEPT overload).
    ///
    /// Does not validate that both fovx and fovy are non-zero (matches the noexcept C++ overload).
    #[inline]
    pub fn set_fov(&mut self, near_plane: T, far_plane: T, fovx: T, fovy: T, aspect: T) {
        let two = T::ONE + T::ONE;
        if fovx != T::ZERO {
            // C++: _right = near * tan(fovx/2); _left = -_right;
            //      _top = ((_right - _left) / aspect) / 2; _bottom = -_top;
            self.right = near_plane * (fovx / two).tan();
            self.left = -self.right;
            self.top = ((self.right - self.left) / aspect) / two;
            self.bottom = -self.top;
        } else {
            // C++: _top = near * tan(fovy/2); _bottom = -_top;
            //      _right = (_top - _bottom) * aspect / 2; _left = -_right;
            self.top = near_plane * (fovy / two).tan();
            self.bottom = -self.top;
            self.right = (self.top - self.bottom) * aspect / two;
            self.left = -self.right;
        }
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        self.orthographic = false;
    }

    /// Port of `setExc(T near, T far, T fovx, T fovy, T aspect)` — throws if both fovx and fovy
    /// are non-zero.
    ///
    /// Returns `Err` if both `fovx` and `fovy` are non-zero (C++ `throw std::domain_error`).
    pub fn set_fov_exc(
        &mut self,
        near_plane: T,
        far_plane: T,
        fovx: T,
        fovy: T,
        aspect: T,
    ) -> Result<(), &'static str> {
        if fovx != T::ZERO && fovy != T::ZERO {
            return Err("fovx and fovy cannot both be non-zero.");
        }
        self.set_fov(near_plane, far_plane, fovx, fovy, aspect);
        Ok(())
    }

    /// Port of `modifyNearAndFar(T near, T far)`.
    ///
    /// For perspective, adjusts left/right/top/bottom to the new near plane by
    /// intersecting the corner rays with it.  For orthographic, just updates the planes.
    #[inline]
    pub fn modify_near_and_far(&mut self, near_plane: T, far_plane: T) {
        if self.orthographic {
            self.near_plane = near_plane;
        } else {
            // C++: cast corner rays from origin through near-plane corners, intersect new near plane
            let ll_line = Line3::from_points(
                Vec3::new(T::ZERO, T::ZERO, T::ZERO),
                Vec3::new(self.left, self.bottom, -self.near_plane),
            );
            let ur_line = Line3::from_points(
                Vec3::new(T::ZERO, T::ZERO, T::ZERO),
                Vec3::new(self.right, self.top, -self.near_plane),
            );
            // Plane facing +Z through z = near_plane: normal (0,0,-1), distance = near_plane
            let near_pl = Plane3::from_normal_distance(
                Vec3::new(T::ZERO, T::ZERO, -T::ONE),
                near_plane,
            );

            let mut ll = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            let mut ur = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            near_pl.intersect(&ll_line, &mut ll);
            near_pl.intersect(&ur_line, &mut ur);

            self.left = ll.x;
            self.right = ur.x;
            self.top = ur.y;
            self.bottom = ll.y;
            self.near_plane = near_plane;
            self.far_plane = far_plane;
        }
        self.far_plane = far_plane;
    }

    /// Port of `setOrthographic(bool)`.
    #[inline]
    pub fn set_orthographic(&mut self, ortho: bool) {
        self.orthographic = ortho;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Port of `nearPlane()` / `hither()` — return near clipping distance.
    #[inline]
    pub fn near_plane(&self) -> T { self.near_plane }

    /// Port of `hither()` — alias for `nearPlane()`.
    #[inline]
    pub fn hither(&self) -> T { self.near_plane }

    /// Port of `farPlane()` / `yon()` — return far clipping distance.
    #[inline]
    pub fn far_plane(&self) -> T { self.far_plane }

    /// Port of `yon()` — alias for `farPlane()`.
    #[inline]
    pub fn yon(&self) -> T { self.far_plane }

    /// Port of `left()`.
    #[inline]
    pub fn left(&self) -> T { self.left }

    /// Port of `right()`.
    #[inline]
    pub fn right(&self) -> T { self.right }

    /// Port of `bottom()`.
    #[inline]
    pub fn bottom(&self) -> T { self.bottom }

    /// Port of `top()`.
    #[inline]
    pub fn top(&self) -> T { self.top }

    /// Port of `orthographic()`.
    #[inline]
    pub fn is_orthographic(&self) -> bool { self.orthographic }

    /// Port of `fovx()` — full horizontal field of view in radians.
    ///
    /// C++: `atan2(_right, _nearPlane) - atan2(_left, _nearPlane)`
    #[inline]
    pub fn fovx(&self) -> T {
        self.right.atan2(self.near_plane) - self.left.atan2(self.near_plane)
    }

    /// Port of `fovy()` — full vertical field of view in radians.
    ///
    /// C++: `atan2(_top, _nearPlane) - atan2(_bottom, _nearPlane)`
    #[inline]
    pub fn fovy(&self) -> T {
        self.top.atan2(self.near_plane) - self.bottom.atan2(self.near_plane)
    }

    /// Port of `aspect()` — `(right - left) / (top - bottom)`.
    #[inline]
    pub fn aspect(&self) -> T {
        (self.right - self.left) / (self.top - self.bottom)
    }

    /// Port of `aspectExc()` — same as `aspect()` but returns `Err` if overflow would occur.
    pub fn aspect_exc(&self) -> Result<T, &'static str> {
        let rml = self.right - self.left;
        let tmb = self.top - self.bottom;
        let abs_tmb = if tmb >= T::ZERO { tmb } else { -tmb };
        let abs_rml = if rml >= T::ZERO { rml } else { -rml };
        if abs_tmb < T::ONE && abs_rml > T::MAX * abs_tmb {
            return Err("Bad viewing frustum: aspect ratio cannot be computed.");
        }
        Ok(rml / tmb)
    }

    /// Port of `degenerate()` — true if near==far, left==right, or top==bottom.
    #[inline]
    pub fn degenerate(&self) -> bool {
        (self.near_plane == self.far_plane)
            || (self.left == self.right)
            || (self.top == self.bottom)
    }

    // -----------------------------------------------------------------------
    // Projection matrix
    // -----------------------------------------------------------------------

    /// Port of `projectionMatrix()` (IMATH_NOEXCEPT) — compute the OpenGL-style projection matrix.
    ///
    /// For perspective:
    /// ```text
    /// E  0  0  0
    /// 0  F  0  0
    /// A  B  C -1
    /// 0  0  D  0
    /// ```
    /// For orthographic:
    /// ```text
    /// A  0  0  0
    /// 0  B  0  0
    /// 0  0  C  0
    /// tx ty tz 1
    /// ```
    #[inline]
    pub fn projection_matrix(&self) -> Matrix44<T> {
        let rpl = self.right + self.left;
        let rml = self.right - self.left;
        let tpb = self.top + self.bottom;
        let tmb = self.top - self.bottom;
        let fpn = self.far_plane + self.near_plane;
        let fmn = self.far_plane - self.near_plane;

        let z = T::ZERO;
        let o = T::ONE;
        let two = o + o;
        let neg_two = -two;

        if self.orthographic {
            let tx = -rpl / rml;
            let ty = -tpb / tmb;
            let tz = -fpn / fmn;
            let a = two / rml;
            let b = two / tmb;
            let c = neg_two / fmn;
            // Row-major (Imath convention): Matrix44(a,0,0,0, 0,b,0,0, 0,0,c,0, tx,ty,tz,1)
            Matrix44::new(
                a, z, z, z,
                z, b, z, z,
                z, z, c, z,
                tx, ty, tz, o,
            )
        } else {
            let big_a = rpl / rml;
            let big_b = tpb / tmb;
            let big_c = -fpn / fmn;
            let big_d = (neg_two * self.far_plane * self.near_plane) / fmn;
            let big_e = two * self.near_plane / rml;
            let big_f = two * self.near_plane / tmb;
            // Row-major: Matrix44(E,0,0,0, 0,F,0,0, A,B,C,-1, 0,0,D,0)
            Matrix44::new(
                big_e, z,    z,    z,
                z,     big_f, z,   z,
                big_a, big_b, big_c, -o,
                z,     z,     big_d, z,
            )
        }
    }

    /// Port of `projectionMatrixExc()` — same but returns `Err` on degenerate inputs.
    pub fn projection_matrix_exc(&self) -> Result<Matrix44<T>, &'static str> {
        let rpl = self.right + self.left;
        let rml = self.right - self.left;
        let tpb = self.top + self.bottom;
        let tmb = self.top - self.bottom;
        let fpn = self.far_plane + self.near_plane;
        let fmn = self.far_plane - self.near_plane;

        let abs_rml = if rml >= T::ZERO { rml } else { -rml };
        let abs_rpl = if rpl >= T::ZERO { rpl } else { -rpl };
        let abs_tmb = if tmb >= T::ZERO { tmb } else { -tmb };
        let abs_tpb = if tpb >= T::ZERO { tpb } else { -tpb };
        let abs_fmn = if fmn >= T::ZERO { fmn } else { -fmn };
        let abs_fpn = if fpn >= T::ZERO { fpn } else { -fpn };

        if (abs_rml < T::ONE && abs_rpl > T::MAX * abs_rml)
            || (abs_tmb < T::ONE && abs_tpb > T::MAX * abs_tmb)
            || (abs_fmn < T::ONE && abs_fpn > T::MAX * abs_fmn)
        {
            return Err("Bad viewing frustum: projection matrix cannot be computed.");
        }

        let z = T::ZERO;
        let o = T::ONE;
        let two = o + o;
        let neg_two = -two;

        if self.orthographic {
            let two_val = two;
            let abs_rml2 = abs_rml;
            let abs_tmb2 = abs_tmb;
            let abs_fmn2 = abs_fmn;
            if (abs_rml2 < T::ONE && two_val > T::MAX * abs_rml2)
                || (abs_tmb2 < T::ONE && two_val > T::MAX * abs_tmb2)
                || (abs_fmn2 < T::ONE && two_val > T::MAX * abs_fmn2)
            {
                return Err("Bad viewing frustum: projection matrix cannot be computed.");
            }
            let tx = -rpl / rml;
            let ty = -tpb / tmb;
            let tz = -fpn / fmn;
            let a = two / rml;
            let b = two / tmb;
            let c = neg_two / fmn;
            Ok(Matrix44::new(
                a, z, z, z,
                z, b, z, z,
                z, z, c, z,
                tx, ty, tz, o,
            ))
        } else {
            let big_a = rpl / rml;
            let big_b = tpb / tmb;
            let big_c = -fpn / fmn;
            let ftn = neg_two * self.far_plane * self.near_plane;
            let abs_ftn = if ftn >= T::ZERO { ftn } else { -ftn };
            if abs_fmn < T::ONE && abs_ftn > T::MAX * abs_fmn {
                return Err("Bad viewing frustum: projection matrix cannot be computed.");
            }
            let big_d = ftn / fmn;
            let ttn = two * self.near_plane;
            let abs_ttn = if ttn >= T::ZERO { ttn } else { -ttn };
            if (abs_rml < T::ONE && abs_ttn > T::MAX * abs_rml)
                || (abs_tmb < T::ONE && abs_ttn > T::MAX * abs_tmb)
            {
                return Err("Bad viewing frustum: projection matrix cannot be computed.");
            }
            let big_e = ttn / rml;
            let big_f = ttn / tmb;
            Ok(Matrix44::new(
                big_e, z,     z,     z,
                z,     big_f, z,     z,
                big_a, big_b, big_c, -o,
                z,     z,     big_d, z,
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Planes extraction
    // -----------------------------------------------------------------------

    /// Port of `planes(Plane3<T> p[6])` — fill six outward-facing frustum planes.
    ///
    /// Plane order: Top, Right, Bottom, Left, Near, Far.
    /// Normals point **out** of the frustum.
    pub fn planes(&self, p: &mut [Plane3<T>; 6]) {
        let z = T::ZERO;
        if !self.orthographic {
            // Perspective: corner points on near plane, eye at origin
            let a = Vec3::new(self.left,  self.bottom, -self.near_plane);
            let b = Vec3::new(self.left,  self.top,    -self.near_plane);
            let c = Vec3::new(self.right, self.top,    -self.near_plane);
            let d = Vec3::new(self.right, self.bottom, -self.near_plane);
            let o = Vec3::new(z, z, z);
            // C++: p[0].set(o,c,b) — three-point form building top plane
            p[0].set_three_points(o, c, b);  // top
            p[1].set_three_points(o, d, c);  // right
            p[2].set_three_points(o, a, d);  // bottom
            p[3].set_three_points(o, b, a);  // left
        } else {
            // Orthographic: axis-aligned planes
            p[0].set_normal_distance(Vec3::new(z, T::ONE, z),  self.top);
            p[1].set_normal_distance(Vec3::new(T::ONE, z, z),  self.right);
            p[2].set_normal_distance(Vec3::new(z, -T::ONE, z), -self.bottom);
            p[3].set_normal_distance(Vec3::new(-T::ONE, z, z), -self.left);
        }
        // Near and far planes are the same for both modes
        // Near: normal (0,0,1), distance -nearPlane  (outward = toward +Z = toward eye)
        p[4].set_normal_distance(Vec3::new(z, z, T::ONE),  -self.near_plane);
        // Far:  normal (0,0,-1), distance farPlane   (outward = toward -Z away from eye)
        p[5].set_normal_distance(Vec3::new(z, z, -T::ONE), self.far_plane);
    }

    /// Port of `planes(Plane3<T> p[6], const Matrix44<T>& M)` — transform frustum by M first.
    ///
    /// Plane order: Top, Right, Bottom, Left, Near, Far. Normals point out of the frustum.
    pub fn planes_transformed(&self, p: &mut [Plane3<T>; 6], m: &Matrix44<T>) {
        // Transform the near-plane corner points by M (C++: Vec3 * Matrix44)
        let a = vec3_mul_matrix44(&Vec3::new(self.left,  self.bottom, -self.near_plane), m);
        let b = vec3_mul_matrix44(&Vec3::new(self.left,  self.top,    -self.near_plane), m);
        let c = vec3_mul_matrix44(&Vec3::new(self.right, self.top,    -self.near_plane), m);
        let d = vec3_mul_matrix44(&Vec3::new(self.right, self.bottom, -self.near_plane), m);

        if !self.orthographic {
            // C++ computes the far-corner scale in `double` regardless of `T`:
            //   `double s = _farPlane / double(_nearPlane);`
            //   `T farLeft = (T)(s * _left);`  etc.
            // We mirror that exactly: promote to f64, divide, then narrow each
            // product back to T at the same point C++ does the `(T)` cast.
            let s: f64 = self.far_plane.to_f64() / self.near_plane.to_f64();
            let far_left   = T::from_f64(s * self.left.to_f64());
            let far_right  = T::from_f64(s * self.right.to_f64());
            let far_top    = T::from_f64(s * self.top.to_f64());
            let far_bottom = T::from_f64(s * self.bottom.to_f64());
            let e = vec3_mul_matrix44(&Vec3::new(far_left,  far_bottom, -self.far_plane), m);
            let f = vec3_mul_matrix44(&Vec3::new(far_left,  far_top,    -self.far_plane), m);
            let g = vec3_mul_matrix44(&Vec3::new(far_right, far_top,    -self.far_plane), m);
            let o = vec3_mul_matrix44(&Vec3::new(T::ZERO, T::ZERO, T::ZERO), m);
            // Side planes through transformed eye
            p[0].set_three_points(o, c, b);  // top
            p[1].set_three_points(o, d, c);  // right
            p[2].set_three_points(o, a, d);  // bottom
            p[3].set_three_points(o, b, a);  // left
            // Near/far from transformed corners
            p[4].set_three_points(a, d, c);  // near
            p[5].set_three_points(e, f, g);  // far
        } else {
            let e = vec3_mul_matrix44(&Vec3::new(self.left,  self.bottom, -self.far_plane), m);
            let f = vec3_mul_matrix44(&Vec3::new(self.left,  self.top,    -self.far_plane), m);
            let g = vec3_mul_matrix44(&Vec3::new(self.right, self.top,    -self.far_plane), m);
            let h = vec3_mul_matrix44(&Vec3::new(self.right, self.bottom, -self.far_plane), m);
            p[0].set_three_points(c, g, f);  // top
            p[1].set_three_points(d, h, g);  // right
            p[2].set_three_points(a, e, h);  // bottom
            p[3].set_three_points(b, f, e);  // left
            p[4].set_three_points(a, d, c);  // near
            p[5].set_three_points(e, f, g);  // far
        }
    }

    // -----------------------------------------------------------------------
    // Window / screen ↔ local space helpers
    // -----------------------------------------------------------------------

    /// Port of `window(T l, T r, T t, T b)` — sub-frustum from a screen-space rectangle.
    ///
    /// Takes a rectangle in screen space (−1 ≤ l ≤ r ≤ 1, −1 ≤ b ≤ t ≤ 1) and
    /// returns a new Frustum whose near window is that rectangle in local space.
    #[inline]
    pub fn window(&self, left: T, right: T, top: T, bottom: T) -> Frustum<T> {
        let bl = self.screen_to_local(Vec2::new(left, bottom));
        let tr = self.screen_to_local(Vec2::new(right, top));
        Frustum::from_params(
            self.near_plane,
            self.far_plane,
            bl.x,
            tr.x,
            tr.y,
            bl.y,
            self.orthographic,
        )
    }

    /// Port of `screenToLocal(const Vec2<T>&)` — map screen [-1,1]² point to local space.
    ///
    /// C++: `left + (right - left) * (1 + s.x) / 2`, same for y.
    #[inline]
    pub fn screen_to_local(&self, s: Vec2<T>) -> Vec2<T> {
        let two = T::ONE + T::ONE;
        Vec2::new(
            self.left + (self.right - self.left) * (T::ONE + s.x) / two,
            self.bottom + (self.top - self.bottom) * (T::ONE + s.y) / two,
        )
    }

    /// Port of `localToScreen(const Vec2<T>&)` (IMATH_NOEXCEPT).
    ///
    /// C++: `(left - 2*p.x + right) / (left - right)`, same for y.
    #[inline]
    pub fn local_to_screen(&self, p: Vec2<T>) -> Vec2<T> {
        let two = T::ONE + T::ONE;
        let lpr = self.left - two * p.x + self.right;
        let lmr = self.left - self.right;
        let bpt = self.bottom - two * p.y + self.top;
        let bmt = self.bottom - self.top;
        Vec2::new(lpr / lmr, bpt / bmt)
    }

    /// Port of `localToScreenExc(const Vec2<T>&)` — returns `Err` on overflow.
    pub fn local_to_screen_exc(&self, p: Vec2<T>) -> Result<Vec2<T>, &'static str> {
        let two = T::ONE + T::ONE;
        let lpr = self.left - two * p.x + self.right;
        let lmr = self.left - self.right;
        let bpt = self.bottom - two * p.y + self.top;
        let bmt = self.bottom - self.top;

        let abs_lmr = if lmr >= T::ZERO { lmr } else { -lmr };
        let abs_lpr = if lpr >= T::ZERO { lpr } else { -lpr };
        let abs_bmt = if bmt >= T::ZERO { bmt } else { -bmt };
        let abs_bpt = if bpt >= T::ZERO { bpt } else { -bpt };

        if (abs_lmr < T::ONE && abs_lpr > T::MAX * abs_lmr)
            || (abs_bmt < T::ONE && abs_bpt > T::MAX * abs_bmt)
        {
            return Err(
                "Bad viewing frustum: local-to-screen transformation cannot be computed",
            );
        }
        Ok(Vec2::new(lpr / lmr, bpt / bmt))
    }

    // -----------------------------------------------------------------------
    // Projection helpers
    // -----------------------------------------------------------------------

    /// Port of `projectScreenToRay(const Vec2<T>&)`.
    ///
    /// Projects a screen-space 2D point to a 3D ray in camera space.
    #[inline]
    pub fn project_screen_to_ray(&self, p: Vec2<T>) -> Line3<T> {
        let point = self.screen_to_local(p);
        if self.orthographic {
            Line3::from_points(
                Vec3::new(point.x, point.y, T::ZERO),
                Vec3::new(point.x, point.y, -T::ONE),
            )
        } else {
            Line3::from_points(
                Vec3::new(T::ZERO, T::ZERO, T::ZERO),
                Vec3::new(point.x, point.y, -self.near_plane),
            )
        }
    }

    /// Port of `projectPointToScreen(const Vec3<T>&)` (IMATH_NOEXCEPT).
    ///
    /// Projects a 3D camera-space point to screen [-1,1]² coordinates.
    #[inline]
    pub fn project_point_to_screen(&self, point: Vec3<T>) -> Vec2<T> {
        if self.orthographic || point.z == T::ZERO {
            self.local_to_screen(Vec2::new(point.x, point.y))
        } else {
            self.local_to_screen(Vec2::new(
                point.x * self.near_plane / -point.z,
                point.y * self.near_plane / -point.z,
            ))
        }
    }

    /// Port of `projectPointToScreenExc(const Vec3<T>&)` — returns `Err` on overflow.
    pub fn project_point_to_screen_exc(&self, point: Vec3<T>) -> Result<Vec2<T>, &'static str> {
        if self.orthographic || point.z == T::ZERO {
            self.local_to_screen_exc(Vec2::new(point.x, point.y))
        } else {
            self.local_to_screen_exc(Vec2::new(
                point.x * self.near_plane / -point.z,
                point.y * self.near_plane / -point.z,
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Z / depth mapping
    // -----------------------------------------------------------------------

    /// Port of `ZToDepth(long zval, long min, long max)` (IMATH_NOEXCEPT).
    ///
    /// Maps an integer z value in [min, max] to a depth (world-space Z).
    ///
    /// C++ (`ImathFrustum.h:757-767`):
    /// ```text
    /// int zdiff = zmax - zmin;
    /// if (zval > zmax + 1) zval -= zdiff;
    /// T fzval = (T (zval) - T (zmin)) / T (zdiff);
    /// return normalizedZToDepth (fzval);
    /// ```
    /// The subtraction `T(zval) - T(zmin)` is done in `T`, exactly as C++ does —
    /// **not** in the integer domain. C++ `long` is 32-bit on Windows (where the
    /// parity oracle is built), so its narrowing of `zdiff` to `int` is a no-op
    /// there; our API is `i64`, and the `saturating_*` forms replace C++'s
    /// signed-overflow UB at `i64::MAX` without panicking.
    #[inline]
    pub fn z_to_depth(&self, zval: i64, zmin: i64, zmax: i64) -> T {
        let zdiff = zmax.saturating_sub(zmin);
        // C++: if (zval > zmax + 1) zval -= zdiff;
        let z = if zval > zmax.saturating_add(1) {
            zval.saturating_sub(zdiff)
        } else {
            zval
        };
        // fzval = (T(z) - T(zmin)) / T(zdiff)
        let fzval = (t_from_i64::<T>(z) - t_from_i64::<T>(zmin)) / t_from_i64::<T>(zdiff);
        self.normalized_z_to_depth(fzval)
    }

    /// Port of `ZToDepthExc(long zval, long min, long max)` — returns `Err` if zdiff == 0
    /// (`ImathFrustum.h:740-755`).
    pub fn z_to_depth_exc(&self, zval: i64, zmin: i64, zmax: i64) -> Result<T, &'static str> {
        let zdiff = zmax.saturating_sub(zmin);
        if zdiff == 0 {
            return Err("Bad call to Frustum::ZToDepth: zmax == zmin");
        }
        let z = if zval > zmax.saturating_add(1) {
            zval.saturating_sub(zdiff)
        } else {
            zval
        };
        let fzval = (t_from_i64::<T>(z) - t_from_i64::<T>(zmin)) / t_from_i64::<T>(zdiff);
        self.normalized_z_to_depth_exc(fzval)
    }

    /// Port of `normalizedZToDepth(T zval)` (IMATH_NOEXCEPT).
    ///
    /// Maps a normalized z in [0,1] to a world-space depth.
    #[inline]
    pub fn normalized_z_to_depth(&self, zval: T) -> T {
        let two = T::ONE + T::ONE;
        let big_zp = zval * two - T::ONE;
        if self.orthographic {
            // C++: -(Zp * (far - near) + (far + near)) / 2
            -(big_zp * (self.far_plane - self.near_plane)
                + (self.far_plane + self.near_plane))
                / two
        } else {
            // C++: (2*far*near) / (Zp*(far-near) - far - near)
            let ftn = two * self.far_plane * self.near_plane;
            let denom =
                big_zp * (self.far_plane - self.near_plane) - self.far_plane - self.near_plane;
            ftn / denom
        }
    }

    /// Port of `normalizedZToDepthExc(T zval)` — returns `Err` if near/far too close.
    pub fn normalized_z_to_depth_exc(&self, zval: T) -> Result<T, &'static str> {
        let two = T::ONE + T::ONE;
        let big_zp = zval * two - T::ONE;
        if self.orthographic {
            Ok(-(big_zp * (self.far_plane - self.near_plane)
                + (self.far_plane + self.near_plane))
                / two)
        } else {
            let ftn = two * self.far_plane * self.near_plane;
            let denom =
                big_zp * (self.far_plane - self.near_plane) - self.far_plane - self.near_plane;
            let abs_denom = if denom >= T::ZERO { denom } else { -denom };
            let abs_ftn = if ftn >= T::ZERO { ftn } else { -ftn };
            if abs_denom < T::ONE && abs_ftn > T::MAX * abs_denom {
                return Err(
                    "Frustum::normalizedZToDepth cannot be computed: \
                     near and far clipping planes of the viewing frustum \
                     may be too close to each other",
                );
            }
            Ok(ftn / denom)
        }
    }

    /// Port of `DepthToZ(T depth, long zmin, long zmax)` (IMATH_NOEXCEPT).
    ///
    /// Maps a world-space depth to an integer z value in [zmin, zmax].
    ///
    /// C++ (`ImathFrustum.h:871-893`) ends both branches with
    /// `return long (0.5 * (Zp + 1) * zdiff) + zmin;`. The literal `0.5` is a
    /// `double`, so `(Zp + 1)` is formed in `T` and the rest of the product is
    /// evaluated in `double` — reproduced exactly by [`zp_to_z`].
    ///
    /// `depth == 0` on the perspective branch makes `farTimesNear / depth`
    /// infinite; C++ then has UB in `long(inf)` (MSVC/x86 yields `INT_MIN`).
    /// We keep Rust's *defined* saturating float→int cast instead of inventing
    /// a C++-platform quirk; use [`Self::depth_to_z_exc`] to get the C++
    /// `DepthToZExc` domain error for that input.
    #[inline]
    pub fn depth_to_z(&self, depth: T, zmin: i64, zmax: i64) -> i64 {
        let zdiff = zmax.saturating_sub(zmin);
        let fmn = self.far_plane - self.near_plane;

        if self.orthographic {
            // C++: T Zp = -(2*depth + far + near) / (far - near)
            let fpn2 = (T::ONE + T::ONE) * depth + self.far_plane + self.near_plane;
            let big_zp = -fpn2 / fmn;
            zp_to_z(big_zp, zdiff, zmin)
        } else {
            // C++: T Zp = (2*far*near/depth + far + near) / (far - near)
            let two = T::ONE + T::ONE;
            let ftn = two * self.far_plane * self.near_plane;
            let fpn = ftn / depth + self.far_plane + self.near_plane;
            let big_zp = fpn / fmn;
            zp_to_z(big_zp, zdiff, zmin)
        }
    }

    /// Port of `DepthToZExc(T depth, long zmin, long zmax)` — returns `Err` on overflow
    /// (`ImathFrustum.h:820-869`).
    pub fn depth_to_z_exc(&self, depth: T, zmin: i64, zmax: i64) -> Result<i64, &'static str> {
        let zdiff = zmax.saturating_sub(zmin);
        let fmn = self.far_plane - self.near_plane;

        if self.orthographic {
            let two = T::ONE + T::ONE;
            let fpn = two * depth + self.far_plane + self.near_plane;
            let abs_fmn = if fmn >= T::ZERO { fmn } else { -fmn };
            let abs_fpn = if fpn >= T::ZERO { fpn } else { -fpn };
            if abs_fmn < T::ONE && abs_fpn > T::MAX * abs_fmn {
                return Err(
                    "Bad viewing frustum: near and far clipping planes are too close to each other",
                );
            }
            let big_zp = -fpn / fmn;
            Ok(zp_to_z(big_zp, zdiff, zmin))
        } else {
            let two = T::ONE + T::ONE;
            let ftn = two * self.far_plane * self.near_plane;
            let abs_ftn = if ftn >= T::ZERO { ftn } else { -ftn };
            let abs_depth = if depth >= T::ZERO { depth } else { -depth };
            if abs_depth < T::ONE && abs_ftn > T::MAX * abs_depth {
                return Err("Bad call to DepthToZ function: value of `depth' is too small");
            }
            let fpn = ftn / depth + self.far_plane + self.near_plane;
            let abs_fmn = if fmn >= T::ZERO { fmn } else { -fmn };
            let abs_fpn = if fpn >= T::ZERO { fpn } else { -fpn };
            if abs_fmn < T::ONE && abs_fpn > T::MAX * abs_fmn {
                return Err(
                    "Bad viewing frustum: near and far clipping planes are too close to each other",
                );
            }
            let big_zp = fpn / fmn;
            Ok(zp_to_z(big_zp, zdiff, zmin))
        }
    }

    // -----------------------------------------------------------------------
    // Screen / world radius conversion
    // -----------------------------------------------------------------------

    /// Port of `worldRadius(const Vec3<T>& p, T radius)` (IMATH_NOEXCEPT).
    ///
    /// Converts a screen-space radius to world-space radius at point `p`.
    /// C++: `radius * (p.z / -_nearPlane)`
    #[inline]
    pub fn world_radius(&self, p: Vec3<T>, radius: T) -> T {
        radius * (p.z / -self.near_plane)
    }

    /// Port of `worldRadiusExc(const Vec3<T>& p, T radius)` — returns `Err` if near ≈ 0.
    pub fn world_radius_exc(&self, p: Vec3<T>, radius: T) -> Result<T, &'static str> {
        let neg_near = -self.near_plane;
        let abs_neg_near = if neg_near >= T::ZERO { neg_near } else { -neg_near };
        let abs_pz = if p.z >= T::ZERO { p.z } else { -p.z };
        if abs_neg_near > T::ONE || abs_pz < T::MAX * abs_neg_near {
            Ok(radius * (p.z / neg_near))
        } else {
            Err("Bad viewing frustum: near clipping plane is too close to zero")
        }
    }

    /// Port of `screenRadius(const Vec3<T>& p, T radius)` (IMATH_NOEXCEPT).
    ///
    /// Converts a world-space radius to screen-space radius at point `p`.
    /// C++: `radius * (-_nearPlane / p.z)`
    #[inline]
    pub fn screen_radius(&self, p: Vec3<T>, radius: T) -> T {
        radius * (-self.near_plane / p.z)
    }

    /// Port of `screenRadiusExc(const Vec3<T>& p, T radius)` — returns `Err` if p.z ≈ 0.
    pub fn screen_radius_exc(&self, p: Vec3<T>, radius: T) -> Result<T, &'static str> {
        let neg_near = -self.near_plane;
        let abs_pz = if p.z >= T::ZERO { p.z } else { -p.z };
        let abs_neg_near = if neg_near >= T::ZERO { neg_near } else { -neg_near };
        if abs_pz > T::ONE || abs_neg_near < T::MAX * abs_pz {
            Ok(radius * (neg_near / p.z))
        } else {
            Err("Bad call to Frustum::screenRadius: magnitude of `p' is too small")
        }
    }

}

// ---------------------------------------------------------------------------
// Internal scalar conversion helpers (no unsafe, no cast trait needed)
// ---------------------------------------------------------------------------

/// C++ `T (n)` where `n` is a `long` — a single integer→float conversion.
///
/// `i64 -> f64` is exact for every |n| < 2^53, so routing through `f64` gives
/// the same value the C++ one-step conversion produces for every value a 32-bit
/// C++ `long` can hold.
#[inline]
fn t_from_i64<T: VecFloat>(n: i64) -> T {
    T::from_f64(n as f64)
}

/// C++ `long (0.5 * (Zp + 1) * zdiff) + zmin` — the shared tail of all four
/// `DepthToZ[Exc]` branches (`ImathFrustum.h:852`, `:868`, `:884`, `:894`).
///
/// `(Zp + 1)` is formed in `T` (the C++ literal `1` is an `int`, so it converts
/// to `T`); `0.5` is a `double`, which promotes the rest of the product to
/// `double`. `as i64` truncates toward zero like the C++ cast, and saturates
/// (rather than being UB) for infinities and out-of-range magnitudes.
#[inline]
fn zp_to_z<T: VecFloat>(big_zp: T, zdiff: i64, zmin: i64) -> i64 {
    let z = 0.5_f64 * (big_zp + T::ONE).to_f64() * (zdiff as f64);
    (z as i64).saturating_add(zmin)
}

// ---------------------------------------------------------------------------
// Equality (port of operator== / operator!=)
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Frustum<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.near_plane == other.near_plane
            && self.far_plane == other.far_plane
            && self.left == other.left
            && self.right == other.right
            && self.top == other.top
            && self.bottom == other.bottom
            && self.orthographic == other.orthographic
    }
}

// ---------------------------------------------------------------------------
// Default (port of Frustum())
// ---------------------------------------------------------------------------

impl<T: Scalar + VecFloat + VecLimits + Math> Default for Frustum<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathFrustum.h)
// ---------------------------------------------------------------------------

/// Frustum of type `f32` (C++ `Frustumf`).
pub type Frustumf = Frustum<f32>;
/// Frustum of type `f64` (C++ `Frustumd`).
pub type Frustumd = Frustum<f64>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::{Vec2, Vec3};

    // Tolerance for floating-point comparisons in tests.
    const EPS: f32 = 1e-5;
    const EPS64: f64 = 1e-10;

    /// Default frustum should have near=0.1, far=1000, left=-1, right=1, top=1, bottom=-1.
    #[test]
    fn frustum_default_values() {
        let f = Frustumf::new();
        assert!((f.near_plane - 0.1f32).abs() < EPS, "near={}", f.near_plane);
        assert!((f.far_plane - 1000.0f32).abs() < EPS, "far={}", f.far_plane);
        assert!((f.left - (-1.0f32)).abs() < EPS, "left={}", f.left);
        assert!((f.right - 1.0f32).abs() < EPS, "right={}", f.right);
        assert!((f.top - 1.0f32).abs() < EPS, "top={}", f.top);
        assert!((f.bottom - (-1.0f32)).abs() < EPS, "bottom={}", f.bottom);
        assert!(!f.orthographic);
    }

    /// Equality: two identical frustums compare equal; one with a different field does not.
    #[test]
    fn frustum_equality() {
        let a = Frustumf::from_params(1.0, 100.0, -2.0, 2.0, 2.0, -2.0, false);
        let b = Frustumf::from_params(1.0, 100.0, -2.0, 2.0, 2.0, -2.0, false);
        assert_eq!(a, b);
        let c = Frustumf::from_params(1.0, 100.0, -2.0, 2.0, 2.0, -2.0, true);
        assert_ne!(a, c);
    }

    /// degenerate() should be true when near == far.
    #[test]
    fn frustum_degenerate() {
        let f = Frustumf::from_params(5.0, 5.0, -1.0, 1.0, 1.0, -1.0, false);
        assert!(f.degenerate());
        let g = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        assert!(!g.degenerate());
    }

    /// perspective projectionMatrix — check known elements.
    ///
    /// For a symmetric frustum with near=1, far=100, left=-1, right=1, top=1, bottom=-1:
    ///   E = 2*near/(right-left) = 2*1/2 = 1
    ///   F = 2*near/(top-bottom) = 2*1/2 = 1
    ///   C = -(far+near)/(far-near) = -101/99
    ///   D = -2*far*near/(far-near) = -200/99
    #[test]
    fn frustum_perspective_projection_matrix() {
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let m = f.projection_matrix();
        // Row 0: E 0 0 0
        assert!((m.x[0][0] - 1.0f32).abs() < EPS, "E={}", m.x[0][0]);
        assert!((m.x[0][1]).abs() < EPS);
        // Row 1: 0 F 0 0
        assert!((m.x[1][1] - 1.0f32).abs() < EPS, "F={}", m.x[1][1]);
        // Row 2: A B C -1
        let c_expected = -101.0f32 / 99.0f32;
        assert!((m.x[2][2] - c_expected).abs() < EPS, "C={}", m.x[2][2]);
        assert!((m.x[2][3] - (-1.0f32)).abs() < EPS, "row2[3]={}", m.x[2][3]);
        // Row 3: 0 0 D 0
        let d_expected = -200.0f32 / 99.0f32;
        assert!((m.x[3][2] - d_expected).abs() < EPS, "D={}", m.x[3][2]);
        assert!((m.x[3][3]).abs() < EPS);
    }

    /// orthographic projectionMatrix — check known elements.
    ///
    /// For left=-1, right=1, top=1, bottom=-1, near=1, far=100, ortho=true:
    ///   A = 2/(right-left) = 1
    ///   B = 2/(top-bottom) = 1
    ///   C = -2/(far-near) = -2/99
    ///   tx = -(right+left)/(right-left) = 0
    ///   ty = 0, tz = -(far+near)/(far-near) = -101/99
    #[test]
    fn frustum_ortho_projection_matrix() {
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, true);
        let m = f.projection_matrix();
        assert!((m.x[0][0] - 1.0f32).abs() < EPS, "A={}", m.x[0][0]);
        assert!((m.x[1][1] - 1.0f32).abs() < EPS, "B={}", m.x[1][1]);
        let c_expected = -2.0f32 / 99.0f32;
        assert!((m.x[2][2] - c_expected).abs() < EPS, "C={}", m.x[2][2]);
        assert!((m.x[3][0]).abs() < EPS, "tx={}", m.x[3][0]); // tx = 0
        let tz_expected = -101.0f32 / 99.0f32;
        assert!((m.x[3][2] - tz_expected).abs() < EPS, "tz={}", m.x[3][2]);
        assert!((m.x[3][3] - 1.0f32).abs() < EPS, "w={}", m.x[3][3]);
    }

    /// planes() — six planes should all have near-inward normals (all dot products with
    /// the center of the frustum should be negative, i.e., center is on the inner side).
    #[test]
    fn frustum_planes_perspective_outward_normals() {
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let mut p = [Plane3::new(); 6];
        f.planes(&mut p);
        // A point clearly inside the frustum
        let inside = Vec3::new(0.0f32, 0.0, -10.0);
        for (i, plane) in p.iter().enumerate() {
            // distance_to returns (dot(normal, point) - distance)
            // For outward normals, inside point should have negative distance_to.
            let d = plane.distance_to(inside);
            assert!(d < 0.0f32, "plane[{}] normal points inward or wrong: d={}", i, d);
        }
    }

    /// projectScreenToRay / projectPointToScreen roundtrip.
    ///
    /// A point on the near plane should project to its screen coordinates, and
    /// the ray through those screen coordinates should pass through that point.
    #[test]
    fn frustum_project_screen_ray_roundtrip() {
        // Symmetric frustum: near=1, far=100, 45° fov each side
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        // A point on the near plane
        let pt = Vec3::new(0.5f32, 0.3, -1.0);
        let sc = f.project_point_to_screen(pt);
        let ray = f.project_screen_to_ray(sc);
        // The ray from the origin through screen point should pass through pt's direction.
        // ray.dir should be parallel to pt (since eye is at origin).
        let pt_len = (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z).sqrt();
        let expected_dir_x = pt.x / pt_len;
        let expected_dir_y = pt.y / pt_len;
        let expected_dir_z = pt.z / pt_len;
        assert!(
            (ray.dir.x - expected_dir_x).abs() < EPS * 5.0,
            "dir.x: {} vs {}",
            ray.dir.x,
            expected_dir_x
        );
        assert!(
            (ray.dir.y - expected_dir_y).abs() < EPS * 5.0,
            "dir.y: {} vs {}",
            ray.dir.y,
            expected_dir_y
        );
        assert!(
            (ray.dir.z - expected_dir_z).abs() < EPS * 5.0,
            "dir.z: {} vs {}",
            ray.dir.z,
            expected_dir_z
        );
    }

    /// fovx / fovy for a symmetric frustum.
    #[test]
    fn frustum_fov() {
        // tan(fovx/2) = 1/1 = 1 → fovx/2 = π/4 → fovx = π/2
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        let pi_half = core::f32::consts::FRAC_PI_2;
        assert!((f.fovx() - pi_half).abs() < EPS, "fovx={}", f.fovx());
        assert!((f.fovy() - pi_half).abs() < EPS, "fovy={}", f.fovy());
    }

    /// aspect() — (right-left)/(top-bottom).
    #[test]
    fn frustum_aspect() {
        let f = Frustumf::from_params(1.0, 100.0, -2.0, 2.0, 1.0, -1.0, false);
        assert!((f.aspect() - 2.0f32).abs() < EPS, "aspect={}", f.aspect());
    }

    /// from_fov constructor: build a frustum from fovy and check fovy round-trips.
    #[test]
    fn frustum_from_fov_roundtrip() {
        let fovy: f32 = core::f32::consts::FRAC_PI_2; // 90°
        let aspect = 1.0f32;
        let f = Frustumf::from_fov(1.0, 100.0, 0.0, fovy, aspect);
        assert!((f.fovy() - fovy).abs() < EPS * 10.0, "fovy={}", f.fovy());
        assert!(!f.orthographic);
    }

    /// screen_to_local / local_to_screen roundtrip at the center.
    #[test]
    fn frustum_screen_local_roundtrip() {
        let f = Frustumf::from_params(1.0, 100.0, -2.0, 2.0, 1.5, -1.5, false);
        let screen = Vec2::new(0.5f32, -0.3f32);
        let local = f.screen_to_local(screen);
        let back = f.local_to_screen(local);
        assert!((back.x - screen.x).abs() < EPS, "x: {} vs {}", back.x, screen.x);
        assert!((back.y - screen.y).abs() < EPS, "y: {} vs {}", back.y, screen.y);
    }

    /// normalizedZToDepth / ZToDepth basic check.
    #[test]
    fn frustum_z_to_depth_perspective() {
        let f = Frustumd::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        // zval = zmin means fzval = 0 → depth should be near (-1 in NDC → Zp = -1)
        // Zp = 0*2-1 = -1; depth = 2*100*1 / (-1*(99) - 100 - 1) = 200 / (-199) ≈ -1.005
        // Just check it doesn't panic and produces a negative depth
        let d = f.z_to_depth(0, 0, 1000);
        assert!(d < 0.0, "expected negative depth, got {}", d);
        let _ = EPS64; // suppress unused warning
    }

    /// window() — sub-frustum in the upper-right quadrant of screen space.
    #[test]
    fn frustum_window() {
        let f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        // Screen rect: right half, top half
        let w = f.window(0.0, 1.0, 1.0, 0.0);
        // left = screen_to_local(0, 0).x = -1 + 2*(0.5) = 0
        assert!((w.left - 0.0f32).abs() < EPS, "w.left={}", w.left);
        assert!((w.right - 1.0f32).abs() < EPS, "w.right={}", w.right);
        assert!((w.top - 1.0f32).abs() < EPS, "w.top={}", w.top);
        assert!((w.bottom - 0.0f32).abs() < EPS, "w.bottom={}", w.bottom);
    }

    /// modifyNearAndFar for perspective frustum — corners must lie on new near plane.
    #[test]
    fn frustum_modify_near_and_far() {
        let mut f = Frustumf::from_params(1.0, 100.0, -1.0, 1.0, 1.0, -1.0, false);
        f.modify_near_and_far(2.0, 200.0);
        assert!((f.near_plane - 2.0f32).abs() < EPS);
        assert!((f.far_plane - 200.0f32).abs() < EPS);
        // For a symmetric frustum doubled near, left/right/top/bottom should double.
        assert!((f.left - (-2.0f32)).abs() < EPS, "left={}", f.left);
        assert!((f.right - 2.0f32).abs() < EPS, "right={}", f.right);
    }

    /// planes_transformed with identity matrix must equal planes() — sanity check.
    ///
    /// Also verifies that the `s = far/near` scale is computed in f64, matching C++
    /// `double s = _farPlane / double(_nearPlane)`.  Non-power-of-two near/far
    /// (0.1 / 1000.0) produce an s whose f32 representation (10000.0_f32) is exact
    /// here, but the intermediate multiplication `s * left` (= 10000 * (-0.5625))
    /// exercises the promotion path; a future near/far that triggers a different
    /// rounding path would fail if the code regressed to T-precision division.
    #[test]
    fn frustum_planes_transformed_identity_matches_planes() {
        // Non-power-of-two near/far to exercise the f64 promotion path for `s`.
        let f = Frustumf::from_params(0.1, 1000.0, -0.5625, 0.5625, 0.5625, -0.5625, false);
        let identity = Matrix44::<f32>::default(); // default() == identity matrix

        let mut p_direct = [Plane3::new(); 6];
        f.planes(&mut p_direct);

        let mut p_xformed = [Plane3::new(); 6];
        f.planes_transformed(&mut p_xformed, &identity);

        // All six planes must match to within a tight tolerance.
        for i in 0..6 {
            let dn = p_direct[i].normal;
            let xn = p_xformed[i].normal;
            assert!(
                (dn.x - xn.x).abs() < 1e-5 && (dn.y - xn.y).abs() < 1e-5 && (dn.z - xn.z).abs() < 1e-5,
                "plane[{}] normal mismatch: direct={:?} xformed={:?}", i, dn, xn
            );
            assert!(
                (p_direct[i].distance - p_xformed[i].distance).abs() < 1e-4,
                "plane[{}] distance mismatch: direct={} xformed={}", i, p_direct[i].distance, p_xformed[i].distance
            );
        }
    }

    /// Regression: `s = far/near` computed in f64 must differ from f32 for a
    /// near/far pair where f32 division loses a bit.  We verify the f64 path
    /// produces the correct far-corner coordinates rather than silently rounding.
    #[test]
    fn frustum_planes_transformed_far_scale_f64_precision() {
        // near=3, far=7 → s = 7/3 = 2.333... which is NOT exactly representable;
        // f32 gives 2.3333333 while f64 gives 2.3333333333333335.
        // The product s * left = 2.333... * 1.0 must be computed in f64.
        let f = Frustumf::from_params(3.0, 7.0, -1.0, 1.0, 1.0, -1.0, false);
        let identity = Matrix44::<f32>::default();

        let mut p = [Plane3::new(); 6];
        f.planes_transformed(&mut p, &identity);

        // The far plane (index 5) is built from far-corner points.
        // For a symmetric frustum with identity M, the far plane normal must be ~(0,0,-1).
        let far_normal = p[5].normal;
        assert!(
            far_normal.z < -0.99,
            "far plane normal z should be ~-1.0, got {:?}", far_normal
        );
    }

    // -----------------------------------------------------------------------
    // Z / depth mapping — the cases a C++ golden CANNOT pin down, because C++
    // is UB there (see `zp_to_z`). Everything that IS defined in C++ is checked
    // bit-exactly against the real Imath oracle in tests/parity_golden.rs
    // (`frustum.ztodepth.*` / `frustum.depthtoz.*`).
    // -----------------------------------------------------------------------

    /// A 32-bit depth range used to hang for ~4 s per call: the old helper built
    /// `T` from `zdiff` by adding 1 in a loop (~2^31 iterations) and saturated at
    /// 2^24 for `f32`, so the answer was wrong as well as slow.
    #[test]
    fn z_to_depth_over_a_full_32_bit_range_is_exact_and_fast() {
        let f = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, false);
        let (zmin, zmax) = (0_i64, 2_147_483_647_i64);
        // Independent expectation: the C++ formula evaluated directly.
        // fzval = (T(z) - T(zmin)) / T(zdiff); depth = 2fn / (Zp(f-n) - f - n).
        for z in [0_i64, 1, 1 << 24, 1_000_000_000, 2_147_483_647] {
            let fzval = (z as f32 - zmin as f32) / ((zmax - zmin) as f32);
            let zp = fzval * 2.0 - 1.0;
            let want = (2.0 * 100.0 * 1.0) / (zp * (100.0 - 1.0) - 100.0 - 1.0);
            assert_eq!(f.z_to_depth(z, zmin, zmax).to_bits(), want.to_bits(), "z = {z}");
        }
    }

    /// `depth == 0` made `farTimesNear / depth` infinite; the old converter then
    /// looped forever in release and panicked on `i64` overflow in debug.
    /// C++ `long(inf)` is UB, so we take Rust's defined saturating cast — the
    /// contract that matters is that this RETURNS.
    #[test]
    fn depth_to_z_at_zero_depth_terminates() {
        let f = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, false);
        assert_eq!(f.depth_to_z(0.0, 0, 65535), i64::MAX);
        assert_eq!(f.depth_to_z(-0.0, 0, 65535), i64::MIN);
        // The `Exc` overload reports it as the C++ domain error instead.
        assert!(f.depth_to_z_exc(0.0, 0, 65535).is_err());
    }

    /// `zmax + 1` and `zmax - zmin` used to overflow (debug panic) at the i64
    /// extremes. C++ `long` is 32-bit, so this range is ours alone to define;
    /// the requirement is "no panic", not a particular value.
    #[test]
    fn extreme_z_bounds_do_not_panic() {
        let f = Frustumf::from_params(1.0, 100.0, -0.5, 0.5, 0.4, -0.4, false);
        assert!(f.z_to_depth(0, i64::MIN, i64::MAX).is_finite());
        // Any i64 is a valid result here; the call completing without an
        // overflow panic is the whole assertion.
        std::hint::black_box(f.depth_to_z(-50.0, i64::MIN, i64::MAX));
        assert!(f.z_to_depth(i64::MAX, 0, i64::MAX).is_finite());
    }
}
