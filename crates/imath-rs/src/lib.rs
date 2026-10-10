//! `imath-rs` — pure-Rust, dependency-free 1:1 port of Imath.
//!
//! Foundation crate for `exr-core`. Port target (pinned at Imath `v3.2.0`):
//! <https://github.com/AcademySoftwareFoundation/Imath/tree/v3.2.0/src/Imath>.
//!
//! # Module ⟷ header mapping (foundation)
//! Each module is a 1:1 port of one C++ header:
//! - [`platform`] ← `ImathPlatform.h`
//! - [`traits`]   ← `ImathTypeTraits.h`
//! - [`math`]     ← `ImathMath.h`
//! - [`fun`]      ← `ImathFun.h` + `ImathFun.cpp`
//! - [`roots`]    ← `ImathRoots.h`
//!
//! # Module ⟷ header mapping (half)
//! - [`half`] ← `half.h` + `half.cpp` (the `Half` type), with submodules
//!   `half::limits` ← `halfLimits.h`, `half::function` ← `halfFunction.h`, and
//!   `half::to_float` ← `toFloat.h` / `toFloat.cpp` (the half↔f32 implementation).
//!
//! # Module ⟷ header mapping (vectors)
//! - [`vec`]      ← `ImathVec.h` (the `Vec2`/`Vec3`/`Vec4` types + typedefs), split
//!   into `vec::vec2`, `vec::vec3`, `vec::vec4`.
//! - [`vec_algo`] ← `ImathVecAlgo.h` (`project`, `orthogonal`, `reflect`,
//!   `closest_vertex`).
//!
//! # Module ⟷ header mapping (matrices)
//! - [`matrix`] ← `ImathMatrix.h` (the `Matrix22`/`Matrix33`/`Matrix44` types +
//!   typedefs), split into `matrix::matrix22`, `matrix::matrix33`,
//!   `matrix::matrix44`.
//!
//! The remaining modules (`box_`, `color`, `random`, `frustum`, …) are
//! declared below with the C++ headers they port.
//!
//! # Headers with no portable logic (mapped to Rust language features, not skipped)
//! These C++ headers carry only namespace/export/compatibility machinery; there is
//! nothing to port into a module, so they are documented here instead:
//! - `ImathNamespace.h` — defines the `Imath`/`IMATH_INTERNAL_NAMESPACE` macros and
//!   the `IMATH_INTERNAL_NAMESPACE_HEADER_ENTER/EXIT` wrappers used to version the
//!   C++ namespace. In Rust, **this crate is the namespace**; the macros have no
//!   equivalent and need none.
//! - `ImathExport.h` — Windows/GCC DLL visibility macros (`IMATH_EXPORT`,
//!   `IMATH_HOSTDEVICE`, `IMATH_DEPRECATED`, …). Symbol visibility in Rust is
//!   governed by `pub`; there is no `__declspec(dllexport)` analog to port.
//! - `ImathForward.h` — forward declarations of the Imath class templates so
//!   headers can reference each other before full definition. Rust resolves names
//!   at the module/item level regardless of order, so forward decls are N/A.
//! - `ImathInt64.h` — defines the deprecated `Int64`/`SInt64` typedefs (unsigned /
//!   signed 64-bit). Rust has fixed-width `u64`/`i64` built in; use those directly.
//!   No module is created.
//!
//! Two further headers (`ImathGL.h`, `ImathGLU.h`) are justified exclusions (pure
//! OpenGL FFI wrappers, forbidden by the project no-FFI rule).

// imath-rs is pure math: no SIMD, no FFI, no raw pointers. The "zero unsafe"
// invariant is enforced by the compiler here rather than by review.
#![forbid(unsafe_code)]

// random — Rand32/Rand48/sphere
pub mod random;

// box/interval/line/frame
// Note: `box` is a Rust keyword; the module is named `box_`.
pub mod box_;
pub mod frame;
pub mod interval;
pub mod line;

// plane/line_algo/sphere
pub mod line_algo;
pub mod plane;
pub mod sphere;

// box_algo
pub mod box_algo;

// frustum / frustum_test
pub mod frustum;
pub mod frustum_test;

pub mod color;
pub mod color_algo;
pub mod euler;
pub mod fun;
pub mod half;
pub mod math;
pub mod matrix;
pub mod matrix_algo;
pub mod platform;
pub mod quat;
pub mod roots;
pub mod shear;
pub mod traits;
pub mod vec;
pub mod vec_algo;

// Re-exports of the key public items, mirroring how Imath exposes them at
// namespace scope.

pub use math::{Math, equal_with_abs_error, equal_with_rel_error, sinx_over_x};
pub use platform::{M_PI, M_PI_2};
pub use traits::{ColorScalar, Scalar};

pub use fun::{
    abs, ceil, clamp, cmp, cmpt, divp, divs, equal, finited, finitef, floor, iszero, lerp,
    lerpfactor, modp, mods, predd, predf, sign, succd, succf, trunc, ulerp,
};

pub use roots::{solve_cubic, solve_linear, solve_normalized_cubic, solve_quadratic};

// `vec` module: Imath's `Vec2`/`Vec3`/`Vec4` (port of ImathVec.h). Re-export the
// three types, the `InfException` marker, and every Imath typedef (V2f, V3i, …).
pub use vec::{
    InfException, V2d, V2f, V2h, V2i, V2i64, V2s, V3d, V3f, V3h, V3i, V3i64, V3s, V4d, V4f, V4h, V4i,
    V4i64, V4s, Vec2, Vec3, Vec4,
};

// `vec_algo` module: the ImathVecAlgo.h free functions.
pub use vec_algo::{closest_vertex, orthogonal, project, reflect};

// `half` module: Imath's `half` class, ported as [`half::Half`] (capital H). Fully
// self-contained pure Rust — NO external `half` crate. Re-export the type and limits.
pub use half::{Half, function::HalfFunction, limits::HalfLimits};

// `shear` module: Imath's `Shear6<T>` and the four ImathShear.h typedefs.
// `Shear3f`/`Shear3d` are Vec3 aliases (matching C++ `typedef Vec3<float> Shear3f`).
pub use shear::{Shear3d, Shear3f, Shear6, Shear6d, Shear6f};

// `matrix` module: Imath's `Matrix22`/`Matrix33`/`Matrix44` (port of ImathMatrix.h).
// Re-export the three types and every Imath typedef (M22f, M33d, M44f, …).
pub use matrix::{
    M22d, M22f, M33d, M33f, M44d, M44f, Matrix22, Matrix33, Matrix44,
};

// `quat` module: Imath's `Quat<T>` (port of ImathQuat.h). Re-export the type, the
// `Quatf`/`Quatd` typedefs, the `QuatFloat` element bound, and every free function
// (interpolation, the 4D dot product, and the Matrix33/scalar/Vec3 operators) as
// Imath exposes them at namespace scope.
pub use quat::{
    Quat, QuatFloat, Quatd, Quatf, angle4_d, intermediate, matrix33_mul_quat, quat_dot,
    quat_mul_matrix33, scalar_mul, slerp, slerp_shortest_arc, spline, squad, vec3_mul_quat,
};

// `euler` module: Imath's `Euler<T>` (port of ImathEuler.h). Re-export the type,
// the `Order`/`Axis`/`InputLayout` enums, the `EulerFloat` element bound, and the
// `Eulerf`/`Eulerd` typedefs as Imath exposes them at namespace scope.
pub use euler::{Axis, Euler, EulerFloat, Eulerd, Eulerf, InputLayout, Order};

// `color` module: Imath's `Color3<T>` / `Color4<T>` (port of ImathColor.h). Re-export
// the two types, all 12 typedefs (Color3f/c/h, C3f/c/h, Color4f/c/h, C4f/c/h), and
// the `PackedColor` typedef.
pub use color::{
    C3c, C3f, C3h, C4c, C4f, C4h, Color3, Color3c, Color3f, Color3h, Color4, Color4c, Color4f,
    Color4h, PackedColor,
};

// `color_algo` module: the ImathColorAlgo.h/.cpp free functions. Re-export the
// non-templated double helpers and every templated conversion function.
pub use color_algo::{
    hsv2rgb, hsv2rgb_c4, hsv2rgb_c4_int, hsv2rgb_d, hsv2rgb_d_c4, hsv2rgb_int, packed2rgb,
    packed2rgb_c4, packed2rgb_c4_int, packed2rgb_int, rgb2hsv, rgb2hsv_c4, rgb2hsv_c4_int,
    rgb2hsv_d, rgb2hsv_d_c4, rgb2hsv_int, rgb2packed, rgb2packed_c4, rgb2packed_c4_int,
    rgb2packed_int,
};

// `matrix_algo` module: the ImathMatrixAlgo.h/.cpp free functions (extract SHRT /
// scaling / shear, Euler extraction, rotation/align/local-frame builders,
// outer products, Procrustes, and the Jacobi SVD / eigen solvers). Re-exported as
// Imath exposes them at namespace scope. The C++ `exc`-bool overloads map to a
// single `exc: bool` parameter returning `Result` (see the module docs).
pub use matrix_algo::{
    IDENTITY22D, IDENTITY22F, IDENTITY33D, IDENTITY33F, IDENTITY44D, IDENTITY44F, add_offset,
    align_z_axis_with_target_dir, check_for_zero_scale_in_row_v2, check_for_zero_scale_in_row_v3,
    compute_local_frame, compute_rs_matrix, extract_and_remove_scaling_and_shear_m33,
    extract_and_remove_scaling_and_shear_m44, extract_euler_m22, extract_euler_m33,
    extract_euler_xyz, extract_euler_zyx, extract_quat, extract_scaling_and_shear_m33,
    extract_scaling_and_shear_m44, extract_scaling_m33, extract_scaling_m44, extract_shrt_m33,
    extract_shrt_m44_euler, extract_shrt_m44_order, extract_shrt_m44_vec,
    jacobi_eigen_solver_m33, jacobi_eigen_solver_m33_default, jacobi_eigen_solver_m44,
    jacobi_eigen_solver_m44_default, jacobi_svd_m33, jacobi_svd_m44, max_eigen_vector_m33,
    max_eigen_vector_m44, min_eigen_vector_m33, min_eigen_vector_m44, outer_product_v3,
    outer_product_v4, procrustes_rotation_and_translation,
    procrustes_rotation_and_translation_weighted, remove_scaling_and_shear_m33,
    remove_scaling_and_shear_m44, remove_scaling_m33, remove_scaling_m44, rotation_matrix,
    rotation_matrix_with_up_dir, sans_scaling_and_shear_into_m44, sans_scaling_and_shear_m33,
    sans_scaling_and_shear_m44, sans_scaling_m33, sans_scaling_m44,
};

// `box_` module: Imath's `Box<V>` (port of ImathBox.h). Re-export the generic type,
// the `BoxVector` trait, the `line3_mul_matrix44` free function, and all 12 typedefs.
// Module is named `box_` because `box` is a Rust keyword.
pub use box_::{
    Box, BoxVector,
    Box2s, Box2i, Box2i64, Box2h, Box2f, Box2d,
    Box3s, Box3i, Box3i64, Box3h, Box3f, Box3d,
};

// `interval` module: Imath's `Interval<T>` (port of ImathInterval.h).
pub use interval::{Interval, Intervalf, Intervald, Intervals, Intervali};

// `line` module: Imath's `Line3<T>` (port of ImathLine.h).
pub use line::{Line3, Line3f, Line3d, line3_mul_matrix44};

// `frame` module: the three parallel-transport frame functions (port of ImathFrame.h).
pub use frame::{first_frame, next_frame, last_frame};

// `plane` module: Imath's `Plane3<T>` (port of ImathPlane.h). Re-export the type,
// the `Plane3f`/`Plane3d` typedefs, and the free matrix-transform function.
pub use plane::{Plane3, Plane3f, Plane3d, plane3_mul_matrix44};

// `line_algo` module: the ImathLineAlgo.h free functions.
pub use line_algo::{closest_points, intersect as line_intersect_triangle, closest_vertex as line_closest_vertex, rotate_point};

// `sphere` module: Imath's `Sphere3<T>` (port of ImathSphere.h). Re-export the type
// and the `Sphere3f`/`Sphere3d` typedefs.
pub use sphere::{Sphere3, Sphere3f, Sphere3d};

// `frustum` module: Imath's `Frustum<T>` (port of ImathFrustum.h). Re-export the type,
// the `Frustumf`/`Frustumd` typedefs.
pub use frustum::{Frustum, Frustumf, Frustumd};

// `frustum_test` module: Imath's `FrustumTest<T>` (port of ImathFrustumTest.h). Re-export
// the type and the `FrustumTestf`/`FrustumTestd` typedefs.
pub use frustum_test::{FrustumTest, FrustumTestf, FrustumTestd};

// `box_algo` module: the ImathBoxAlgo.h free functions. Re-export as Imath exposes
// them at namespace scope. C++ overloads mapped to distinct Rust names:
//   transform(box, m)           → `box_algo::transform`
//   transform(box, m, result)   → `box_algo::transform_into`
//   affineTransform(box, m)     → `box_algo::affine_transform`
//   affineTransform(box, m, r)  → `box_algo::affine_transform_into`
//   intersects(box, ray, ip)    → `box_algo::intersects_with_point`
//   intersects(box, ray)        → `box_algo::intersects`  (bool-only)
pub use box_algo::{
    affine_transform, affine_transform_into, clip, closest_point_in_box, closest_point_on_box,
    find_entry_and_exit_points, intersects as box_intersects,
    intersects_with_point as box_intersects_with_point, transform as box_transform,
    transform_into as box_transform_into,
};

// `random` module: Imath's `Rand32`/`Rand48` PRNGs and sphere-distribution helpers
// (port of ImathRandom.h / ImathRandom.cpp). Re-export the two PRNG types, the
// helper trait, standalone erand48/nrand48/drand48/lrand48/srand48 functions,
// and the free sphere/Gaussian functions as Imath exposes them at namespace scope.
pub use random::{
    Rand32, Rand48,
    SphereVec, SphereRand,
    erand48, nrand48, drand48, lrand48, srand48,
    solid_sphere_rand, hollow_sphere_rand, gauss_rand, gauss_sphere_rand,
};
