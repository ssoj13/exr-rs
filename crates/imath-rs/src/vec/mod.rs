//! Port of Imath ImathVec.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathVec.h>
//!
//! 2D, 3D and 4D point/vector class templates: [`Vec2`], [`Vec3`], [`Vec4`].
//!
//! The C++ header is one ~2700-line file; here it is split for focus into
//! [`vec2`], [`vec3`], [`vec4`] (one type each), with this module providing the
//! shared element-type traits, the `InfException` marker, and the `pub use` of the
//! types and all Imath typedefs.
//!
//! # Element-type bounds (how the C++ `template <class T>` maps to Rust)
//! C++ silently assumes the element type `T` supports `+ - * /`, comparison, and
//! the literals `T(0)`/`T(1)`; that contract is the crate's [`Scalar`] trait. Three
//! further capabilities the vector code needs cannot live on `Scalar` (which must
//! also admit integer element types), so they are expressed as dedicated traits
//! here:
//! - [`VecLimits`] — `std::numeric_limits<T>` access for the static
//!   `baseTypeLowest/Max/Smallest/Epsilon` accessors. Implemented for **every**
//!   element type (`i16`, `i32`, `i64`, `f32`, `f64`, [`Half`]).
//! - [`VecFloat`] — the floating-point-only machinery for `length`/`normalize`
//!   (`sqrt`, `abs`, the `T(2)` literal, and `std::numeric_limits<T>::min()`).
//!   Implemented **only** for `f32`, `f64`, [`Half`]. This is the Rust analogue of
//!   the C++ `= delete` of `length`/`normalize` for the integer specializations
//!   `Vec{2,3,4}<{short,int,int64_t}>`: those `length`/`normalize` methods simply
//!   do not exist for integer vectors because `T` does not implement [`VecFloat`].
//!
//! # `IMATH_FOREIGN_VECTOR_INTEROP` — deliberately not ported
//! ImathVec.h guards a block (the `has_xy`/`has_subscript`-constrained templated
//! constructors and `operator=`) behind `#if IMATH_FOREIGN_VECTOR_INTEROP`. Its
//! sole purpose is C++ SFINAE interop: at compile time sniff whether some *foreign*
//! application struct "looks like" a vector (right members or subscript, right
//! arity) and silently convert to/from it. Rust has no `decltype`/`sizeof`-based
//! structural reflection, the exr-rs port defines its own vector types and never
//! needs to adopt a third-party layout, and OpenEXR/IlmImf has no consumer of this
//! feature. This is the SAME decision already recorded for the `ImathTypeTraits.h`
//! detection traits (see `crate::traits`). It is the ONLY thing skipped
//! from ImathVec.h; everything else is ported 1:1.

pub mod vec2;
pub mod vec3;
pub mod vec4;

use crate::half::Half;

/// Enum for the `Vec4` to `Vec3` conversion constructor (port of C++
/// `enum InfException { INF_EXCEPTION }`).
///
/// Passed to [`Vec3::from_vec4_exc`] to select the exception-throwing
/// (here: `Result`-returning) flavour of the `Vec4`->`Vec3` perspective divide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfException {
    /// The single enumerator, matching C++ `INF_EXCEPTION`.
    InfException,
}

/// Element-type conversion, the Rust analogue of the C++ `static_cast`/functional
/// cast `T(v.x)` used by the cross-base-type vector constructors.
///
/// Port target: the `template <class S> Vec2(const Vec2<S>& v)` (and Vec3/Vec4)
/// constructors, whose bodies are `x (T (v.x)), …`. C++ allows a (possibly lossy)
/// numeric cast between any two arithmetic element types; Rust has no single
/// built-in trait covering every such pair (`as` is not a trait, `From` is missing
/// for the lossy directions, and [`Half`] needs explicit handling), so we spell the
/// contract out as this trait and implement it for every ordered pair of element
/// types Imath's typedefs use. `T: VecCast<S>` reads "a `Vec<S>` can be converted
/// to a `Vec<T>`".
pub trait VecCast<S>: Sized {
    /// Convert one element, matching C++ `T(s)` for `T = Self`, source type `S`.
    fn from_elem(s: S) -> Self;
}

/// Implements [`VecCast`] for a `(dst, src)` numeric pair using `as` (which has the
/// same truncation/rounding semantics as the C++ `static_cast` Imath relies on).
macro_rules! impl_veccast_as {
    ($dst:ty, $src:ty) => {
        impl VecCast<$src> for $dst {
            #[inline]
            fn from_elem(s: $src) -> Self {
                s as $dst
            }
        }
    };
}

// Numeric element types: every (dst, src) pair, including the identity pair (used
// by the copy/convert constructor when `S == T`).
macro_rules! impl_veccast_numeric_row {
    ($dst:ty) => {
        impl_veccast_as!($dst, i16);
        impl_veccast_as!($dst, i32);
        impl_veccast_as!($dst, i64);
        impl_veccast_as!($dst, f32);
        impl_veccast_as!($dst, f64);
    };
}
impl_veccast_numeric_row!(i16);
impl_veccast_numeric_row!(i32);
impl_veccast_numeric_row!(i64);
impl_veccast_numeric_row!(f32);
impl_veccast_numeric_row!(f64);

// `Half` as destination: convert source -> f32 -> Half (C++ `half(static_cast<...>)`
// goes through the implicit float promotion). For numeric sources cast to f32 first.
impl VecCast<i16> for Half {
    #[inline]
    fn from_elem(s: i16) -> Self {
        Half::new(s as f32)
    }
}
impl VecCast<i32> for Half {
    #[inline]
    fn from_elem(s: i32) -> Self {
        Half::new(s as f32)
    }
}
impl VecCast<i64> for Half {
    #[inline]
    fn from_elem(s: i64) -> Self {
        Half::new(s as f32)
    }
}
impl VecCast<f32> for Half {
    #[inline]
    fn from_elem(s: f32) -> Self {
        Half::new(s)
    }
}
impl VecCast<f64> for Half {
    #[inline]
    fn from_elem(s: f64) -> Self {
        Half::new(s as f32)
    }
}
impl VecCast<Half> for Half {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s
    }
}

// `Half` as source: convert Half -> f32 -> numeric dst (C++ `static_cast<T>(half)`
// promotes the half to float, then casts to the destination).
impl VecCast<Half> for i16 {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s.to_f32() as i16
    }
}
impl VecCast<Half> for i32 {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s.to_f32() as i32
    }
}
impl VecCast<Half> for i64 {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s.to_f32() as i64
    }
}
impl VecCast<Half> for f32 {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s.to_f32()
    }
}
impl VecCast<Half> for f64 {
    #[inline]
    fn from_elem(s: Half) -> Self {
        s.to_f32() as f64
    }
}

/// `std::numeric_limits<T>` access for the static `baseType*` accessors on the
/// vector types.
///
/// Port target: the four `IMATH_HOSTDEVICE constexpr static T baseType*()` members
/// of `Vec2`/`Vec3`/`Vec4`, each forwarding to a `std::numeric_limits<T>` query.
/// Implemented for every element type Imath's typedefs use so that, e.g.,
/// `V3i::base_type_max()` and `V3f::base_type_max()` both exist.
pub trait VecLimits: Copy {
    /// `std::numeric_limits<T>::lowest()` — largest-magnitude negative value.
    const LOWEST: Self;
    /// `std::numeric_limits<T>::max()` — largest positive value.
    const MAX: Self;
    /// `std::numeric_limits<T>::min()` — smallest positive value (for floats, the
    /// smallest positive *normalized* value).
    const SMALLEST: Self;
    /// `std::numeric_limits<T>::epsilon()` — smallest `e` with `1 + e != 1`.
    /// For integer types C++ `numeric_limits<T>::epsilon()` is `0`.
    const EPSILON: Self;
}

impl VecLimits for i16 {
    const LOWEST: Self = i16::MIN;
    const MAX: Self = i16::MAX;
    const SMALLEST: Self = i16::MIN; // integer numeric_limits::min() == lowest
    const EPSILON: Self = 0;
}

impl VecLimits for i32 {
    const LOWEST: Self = i32::MIN;
    const MAX: Self = i32::MAX;
    const SMALLEST: Self = i32::MIN;
    const EPSILON: Self = 0;
}

impl VecLimits for i64 {
    const LOWEST: Self = i64::MIN;
    const MAX: Self = i64::MAX;
    const SMALLEST: Self = i64::MIN;
    const EPSILON: Self = 0;
}

impl VecLimits for f32 {
    const LOWEST: Self = f32::MIN; // std::numeric_limits<float>::lowest() == -FLT_MAX
    const MAX: Self = f32::MAX;
    const SMALLEST: Self = f32::MIN_POSITIVE; // numeric_limits::min() (normalized)
    const EPSILON: Self = f32::EPSILON;
}

impl VecLimits for f64 {
    const LOWEST: Self = f64::MIN;
    const MAX: Self = f64::MAX;
    const SMALLEST: Self = f64::MIN_POSITIVE;
    const EPSILON: Self = f64::EPSILON;
}

impl VecLimits for Half {
    // `std::numeric_limits<half>` (from halfLimits.h): lowest = -HALF_MAX,
    // max = HALF_MAX, min = HALF_NRM_MIN (smallest normalized), epsilon = HALF_EPSILON.
    const LOWEST: Self = Half::from_bits(0xfbff); // -65504.0 (== -HALF_MAX)
    const MAX: Self = Half::from_bits(0x7bff); //  65504.0 (==  HALF_MAX)
    const SMALLEST: Self = Half::from_bits(0x0400); //  HALF_NRM_MIN (6.1035e-05)
    const EPSILON: Self = Half::from_bits(0x1400); //  HALF_EPSILON (0.00097656)
}

/// Floating-point-only capabilities required by the `length`/`normalize` family.
///
/// Port target: the bodies of `length`/`length2`'s `lengthTiny` helper and the
/// `normalize*` methods, which call `std::sqrt`, `std::abs`, and compare against
/// `T(2) * std::numeric_limits<T>::min()`. C++ deletes these methods for the
/// integer specializations; in Rust we instead simply do **not** implement
/// `VecFloat` for `i16`/`i32`/`i64`, so the methods bounded on `VecFloat` are
/// absent for integer vectors — the same observable result.
pub trait VecFloat: crate::traits::Scalar + crate::math::Math {
    /// The literal `T(2)`, used by the `length` tiny-magnitude guard.
    const TWO: Self;
    /// `std::numeric_limits<T>::min()` — smallest positive normalized value, used
    /// in the `length` tiny-magnitude guard `length2 < T(2) * min()`.
    const MIN_POSITIVE: Self;
    /// `std::abs(self)` — magnitude, used by `lengthTiny`.
    fn vabs(self) -> Self;
    /// Widen `self` to `f64`, matching the C++ `double(x)` promotions in the
    /// Frustum template (e.g. `double(_nearPlane)`, `double(_farPlane)`).
    fn to_f64(self) -> f64;
    /// Narrow an `f64` back to `T`, matching the C++ `(T)(...)` casts that
    /// store a double-precision intermediate into a `T`-typed variable
    /// (e.g. `T farLeft = (T)(s * _left)`).
    fn from_f64(v: f64) -> Self;
}

impl VecFloat for f32 {
    const TWO: Self = 2.0;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    #[inline]
    fn vabs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn to_f64(self) -> f64 { self as f64 }
    #[inline]
    fn from_f64(v: f64) -> Self { v as f32 }
}

impl VecFloat for f64 {
    const TWO: Self = 2.0;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    #[inline]
    fn vabs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn to_f64(self) -> f64 { self }
    #[inline]
    fn from_f64(v: f64) -> Self { v }
}

impl VecFloat for Half {
    /// `Half(2.0)` — bit pattern `0x4000`.
    const TWO: Self = Half::from_bits(0x4000);
    /// `std::numeric_limits<half>::min()` == `HALF_NRM_MIN`.
    const MIN_POSITIVE: Self = Half::from_bits(0x0400);
    #[inline]
    fn vabs(self) -> Self {
        // Magnitude: clear the sign bit. Matches std::abs(half) via float promotion
        // for all finite values (and for inf/NaN, mirrors clearing the sign).
        Half::from_bits(self.bits() & 0x7fff)
    }
    #[inline]
    fn to_f64(self) -> f64 { self.to_f32() as f64 }
    #[inline]
    fn from_f64(v: f64) -> Self { Half::new(v as f32) }
}

pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;

//-------------------------
// Typedefs for convenience (1:1 with ImathVec.h)
//-------------------------

/// `Vec2` of [`Half`] (C++ `V2h`).
pub type V2h = Vec2<Half>;
/// `Vec2` of `i16` (C++ `V2s`, `short`).
pub type V2s = Vec2<i16>;
/// `Vec2` of `i32` (C++ `V2i`, `int`).
pub type V2i = Vec2<i32>;
/// `Vec2` of `i64` (C++ `V2i64`, `int64_t`).
pub type V2i64 = Vec2<i64>;
/// `Vec2` of `f32` (C++ `V2f`, `float`).
pub type V2f = Vec2<f32>;
/// `Vec2` of `f64` (C++ `V2d`, `double`).
pub type V2d = Vec2<f64>;

/// `Vec3` of [`Half`] (C++ `V3h`).
pub type V3h = Vec3<Half>;
/// `Vec3` of `i16` (C++ `V3s`, `short`).
pub type V3s = Vec3<i16>;
/// `Vec3` of `i32` (C++ `V3i`, `int`).
pub type V3i = Vec3<i32>;
/// `Vec3` of `i64` (C++ `V3i64`, `int64_t`).
pub type V3i64 = Vec3<i64>;
/// `Vec3` of `f32` (C++ `V3f`, `float`).
pub type V3f = Vec3<f32>;
/// `Vec3` of `f64` (C++ `V3d`, `double`).
pub type V3d = Vec3<f64>;

/// `Vec4` of [`Half`] (C++ `V4h`).
pub type V4h = Vec4<Half>;
/// `Vec4` of `i16` (C++ `V4s`, `short`).
pub type V4s = Vec4<i16>;
/// `Vec4` of `i32` (C++ `V4i`, `int`).
pub type V4i = Vec4<i32>;
/// `Vec4` of `i64` (C++ `V4i64`, `int64_t`).
pub type V4i64 = Vec4<i64>;
/// `Vec4` of `f32` (C++ `V4f`, `float`).
pub type V4f = Vec4<f32>;
/// `Vec4` of `f64` (C++ `V4d`, `double`).
pub type V4d = Vec4<f64>;
