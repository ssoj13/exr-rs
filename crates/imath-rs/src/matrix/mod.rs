//! Port of Imath ImathMatrix.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrix.h>
//!
//! 2x2, 3x3 and 4x4 transformation matrix templates: [`Matrix22`], [`Matrix33`],
//! [`Matrix44`].
//!
//! The C++ header is one ~5200-line file; here it is split for focus into
//! [`matrix22`], [`matrix33`], [`matrix44`] (one type each), with this module
//! providing the `pub use` of the types, the free vector-times-matrix operator
//! functions, and all Imath typedefs (`M22f`, `M22d`, … `M44d`).
//!
//! # Shared design decisions (apply to all three types)
//! - **Storage.** Each matrix stores its elements as a real row-major array field
//!   `x: [[T; N]; N]`, byte-for-byte matching the C++ `T x[N][N]`. `#[repr(C)]`
//!   guarantees that layout for later byte-level EXR I/O.
//! - **`operator[]` (row access).** C++ returns a row pointer so that `m[i][j]`
//!   works. The Rust port implements `Index<usize>`/`IndexMut<usize>` returning the
//!   row `&[T; N]`, so `m[i][j]` is the same two-step indexing with **zero
//!   `unsafe`** — no pointer arithmetic, OOB panics like a checked access.
//! - **Default = identity.** The C++ default constructor sets the identity; the
//!   Rust `Default` impl matches that exactly (it is NOT a zero matrix).
//! - **Singular inverse: throwing vs non-throwing.** Imath ships two flavours of
//!   `inverse`/`invert`/`gjInverse`/`gjInvert`: one taking `bool singExc` that
//!   `throw`s `std::invalid_argument` on a singular matrix, and a default
//!   non-throwing one that returns the identity. The port mirrors both: the
//!   non-throwing form keeps the C++ name (`inverse`, returns identity on
//!   singular), and the throwing form gets an `_exc` suffix returning
//!   `Result<_, &'static str>`. The pivoting, row swaps and epsilon handling are
//!   reproduced 1:1.
//! - **Method-name overloads.** Where C++ overloads one name on argument type
//!   (e.g. `setScale(T)` vs `setScale(const Vec2&)`, or `setShear(Vec3)` vs
//!   `setShear(Shear6)`), Rust cannot overload, so the scalar/secondary forms get a
//!   disambiguating suffix (`set_scale_scalar`, `set_shear_vec`, …) while the
//!   "primary" vector form keeps the base name.
//!
//! # Same-type instantiation of the `template <class S>` members
//! Several members are templated on a second type `S` (the vector element type),
//! e.g. `multVecMatrix(const Vec2<S>&, Vec2<S>&)`. OpenEXR/IlmImf only ever
//! instantiates these with `S == T`, so the port fixes `S = T`. This matches every
//! real call site and keeps the trait bounds clean; the cross-element-type vector
//! transform is not exercised anywhere in the EXR codebase.
//!
//! # `IMATH_FOREIGN_VECTOR_INTEROP` — deliberately not ported
//! ImathMatrix.h guards a block (the `has_double_subscript`-constrained templated
//! constructors and `operator=`) behind `#if IMATH_FOREIGN_VECTOR_INTEROP`. As with
//! `ImathVec.h`/`ImathTypeTraits.h` (Waves 1a/2), this is pure C++ SFINAE interop
//! for silently adopting a *foreign* application's matrix layout; Rust has no
//! `decltype`/`sizeof` structural reflection, the exr-rs port defines its own
//! matrix types, and OpenEXR/IlmImf has no consumer of it. It is the ONLY thing
//! skipped from ImathMatrix.h; everything else is ported 1:1.
//!
//! # No half matrix typedefs
//! Imath v3.2.0 defines matrix typedefs only for `float`/`double` (`M22f`/`M22d`/…
//! `M44f`/`M44d`) — there is **no** `M44h` or any half-typed matrix typedef, so no
//! `Half` impl additions are required here.

pub mod matrix22;
pub mod matrix33;
pub mod matrix44;

pub use matrix22::Matrix22;
pub use matrix33::Matrix33;
pub use matrix44::Matrix44;

// Free vector-times-matrix operator functions (port of the trailing
// `operator*`/`operator*=` set). Re-exported here so callers can reach them
// without naming the per-type submodule.
pub use matrix22::{vec2_mul_assign_matrix22, vec2_mul_matrix22};
pub use matrix33::{
    vec2_mul_assign_matrix33, vec2_mul_matrix33, vec3_mul_assign_matrix33, vec3_mul_matrix33,
};
pub use matrix44::{
    vec3_mul_assign_matrix44, vec3_mul_matrix44, vec4_mul_assign_matrix44, vec4_mul_matrix44,
};

//-------------------------
// Typedefs for convenience (1:1 with ImathMatrix.h)
//-------------------------

/// 2x2 matrix of `f32` (C++ `M22f`, `Matrix22<float>`).
pub type M22f = Matrix22<f32>;
/// 2x2 matrix of `f64` (C++ `M22d`, `Matrix22<double>`).
pub type M22d = Matrix22<f64>;
/// 3x3 matrix of `f32` (C++ `M33f`, `Matrix33<float>`).
pub type M33f = Matrix33<f32>;
/// 3x3 matrix of `f64` (C++ `M33d`, `Matrix33<double>`).
pub type M33d = Matrix33<f64>;
/// 4x4 matrix of `f32` (C++ `M44f`, `Matrix44<float>`).
pub type M44f = Matrix44<f32>;
/// 4x4 matrix of `f64` (C++ `M44d`, `Matrix44<double>`).
pub type M44d = Matrix44<f64>;
