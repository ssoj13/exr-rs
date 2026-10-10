//! Port of Imath ImathShear.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathShear.h>
//!
//! Defines [`Shear6`] — a 6-element shear transformation coefficient set — and the
//! four typedefs the header provides: [`Shear3f`], [`Shear3d`], [`Shear6f`],
//! [`Shear6d`].
//!
//! ## Circular-include check
//! `ImathShear.h` includes `ImathMath.h` and `ImathVec.h` only — it does NOT include
//! `ImathMatrix.h`, so there is no circular dependency.
//!
//! ## No `IMATH_FOREIGN_*_INTEROP` sections
//! This header contains no `IMATH_FOREIGN_VECTOR_INTEROP`-guarded SFINAE blocks.
//! There is nothing to skip.
//!
//! ## Half shear typedef
//! `ImathShear.h` defines no half-typed shear typedef (`Shear6h` does not exist in
//! Imath v3.2.0). No `Half` impl additions are required.
//!
//! ## `operator[]` implementation note
//! The C++ `operator[]` uses `reinterpret_cast<T*>(this)[i]` to index into the
//! struct. This is UB-free in C++ only because `Shear6` is a standard-layout type
//! and all six `T` fields are contiguous. In Rust there is no safe equivalent of
//! pointer-arithmetic struct traversal; the port uses `match` on the index instead,
//! which is safe, zero-cost (compiler folds it to direct field access), and OOB-panics
//! at runtime just as a debug-mode `assert` would in C++.
//!
//! ## `getValue() -> T*` implementation note
//! The C++ `getValue()` / const overload returns a raw pointer `(T*) &xy`. The Rust
//! port follows the Vec3 precedent: the mutable reference variant is replaced by an
//! `as_array_mut()` that returns `[&mut T; 6]` (safe field references), and the const
//! variant becomes `to_array()` returning a by-value `[T; 6]`. No unsafe is used.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math::{equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{Vec3, VecLimits};

// ============================================================================
// Shear6<T>
// ============================================================================

/// 6-element shear transformation coefficient set (port of C++ `template <class T>
/// class Shear6`).
///
/// The six coefficients correspond to the six off-diagonal elements of the 3×3
/// upper-left block of the 4×4 shear matrix:
///
/// ```text
/// | X' |   |  1  YX  ZX  0 |   | X |
/// | Y' |   | XY   1  ZY  0 |   | Y |
/// | Z' | = | XZ  YZ   1  0 | * | Z |
/// | 1  |   |  0   0   0  1 |   | 1 |
/// ```
///
/// Fields are in declaration order: `xy, xz, yz, yx, zx, zy` — matching the C++ struct
/// layout exactly. `#[repr(C)]` guarantees the same in-memory order as the C++ class.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Shear6<T> {
    /// C++ field `xy`: contribution of Y to X'.
    pub xy: T,
    /// C++ field `xz`: contribution of Z to X'.
    pub xz: T,
    /// C++ field `yz`: contribution of Z to Y'.
    pub yz: T,
    /// C++ field `yx`: contribution of X to Y' (the "extra" shear, index 3).
    pub yx: T,
    /// C++ field `zx`: contribution of X to Z' (the "extra" shear, index 4).
    pub zx: T,
    /// C++ field `zy`: contribution of Y to Z' (the "extra" shear, index 5).
    pub zy: T,
}

// ============================================================================
// Element access — port of `operator[]` (mutable and const)
// ============================================================================

impl<T> Index<usize> for Shear6<T> {
    type Output = T;

    /// Port of `const T& operator[](int i) const`.
    ///
    /// Indices: 0=xy, 1=xz, 2=yz, 3=yx, 4=zx, 5=zy.
    /// Panics on out-of-range index (mirrors C++ UB / debug assert).
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.xy,
            1 => &self.xz,
            2 => &self.yz,
            3 => &self.yx,
            4 => &self.zx,
            5 => &self.zy,
            _ => panic!("Shear6 index out of range: {i}"),
        }
    }
}

impl<T> IndexMut<usize> for Shear6<T> {
    /// Port of `T& operator[](int i)`.
    ///
    /// Panics on out-of-range index.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.xy,
            1 => &mut self.xz,
            2 => &mut self.yz,
            3 => &mut self.yx,
            4 => &mut self.zx,
            5 => &mut self.zy,
            _ => panic!("Shear6 index out of range: {i}"),
        }
    }
}

// ============================================================================
// Constructors and Assignment
// ============================================================================

impl<T: Scalar> Default for Shear6<T> {
    /// Port of `Shear6()` — initialize all six components to zero.
    #[inline]
    fn default() -> Self {
        Shear6 {
            xy: T::ZERO,
            xz: T::ZERO,
            yz: T::ZERO,
            yx: T::ZERO,
            zx: T::ZERO,
            zy: T::ZERO,
        }
    }
}

impl<T: Scalar> Shear6<T> {
    /// Port of `Shear6(T XY, T XZ, T YZ)` — set xy/xz/yz; yx/zx/zy = 0.
    #[inline]
    pub fn from_xyz(xy: T, xz: T, yz: T) -> Self {
        Shear6 { xy, xz, yz, yx: T::ZERO, zx: T::ZERO, zy: T::ZERO }
    }

    /// Port of `Shear6(const Vec3<T>& v)` — xy=v.x, xz=v.y, yz=v.z; yx/zx/zy = 0.
    #[inline]
    pub fn from_vec3(v: Vec3<T>) -> Self {
        Shear6 { xy: v.x, xz: v.y, yz: v.z, yx: T::ZERO, zx: T::ZERO, zy: T::ZERO }
    }

    /// Port of `Shear6(T XY, T XZ, T YZ, T YX, T ZX, T ZY)` — all six components.
    #[inline]
    pub fn new(xy: T, xz: T, yz: T, yx: T, zx: T, zy: T) -> Self {
        Shear6 { xy, xz, yz, yx, zx, zy }
    }
}

impl<T: Copy> Shear6<T> {
    /// Port of `template <class S> Shear6(const Vec3<S>& v)` — cross-type from Vec3.
    ///
    /// Maps: xy = T(v.x), xz = T(v.y), yz = T(v.z); yx/zx/zy = T(0).
    #[inline]
    pub fn from_vec3_cast<S: Copy>(v: Vec3<S>) -> Self
    where
        T: ShearCast<S>,
    {
        Shear6 {
            xy: T::shear_cast(v.x),
            xz: T::shear_cast(v.y),
            yz: T::shear_cast(v.z),
            yx: T::shear_zero(),
            zx: T::shear_zero(),
            zy: T::shear_zero(),
        }
    }

    /// Port of `template <class S> Shear6(const Shear6<S>& h)` — cross-type copy.
    ///
    /// Each field: `T(h.field)` — a possibly-lossy numeric cast.
    #[inline]
    pub fn from_shear6<S: Copy>(h: Shear6<S>) -> Self
    where
        T: ShearCast<S>,
    {
        Shear6 {
            xy: T::shear_cast(h.xy),
            xz: T::shear_cast(h.xz),
            yz: T::shear_cast(h.yz),
            yx: T::shear_cast(h.yx),
            zx: T::shear_cast(h.zx),
            zy: T::shear_cast(h.zy),
        }
    }
}

// ============================================================================
// `setValue` / `getValue` (port of the Sb-compatibility methods)
// ============================================================================

impl<T: Copy> Shear6<T> {
    /// Port of `template <class S> void setValue(S XY, S XZ, S YZ, S YX, S ZX, S ZY)`.
    ///
    /// Assigns all six components from possibly-different-type `S` values.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, xy: S, xz: S, yz: S, yx: S, zx: S, zy: S)
    where
        T: ShearCast<S>,
    {
        self.xy = T::shear_cast(xy);
        self.xz = T::shear_cast(xz);
        self.yz = T::shear_cast(yz);
        self.yx = T::shear_cast(yx);
        self.zx = T::shear_cast(zx);
        self.zy = T::shear_cast(zy);
    }

    /// Port of `template <class S> void setValue(const Shear6<S>& h)`.
    ///
    /// Assigns all six components from a `Shear6<S>`, converting via `T(s)`.
    #[inline]
    pub fn set_value_from<S: Copy>(&mut self, h: Shear6<S>)
    where
        T: ShearCast<S>,
    {
        self.xy = T::shear_cast(h.xy);
        self.xz = T::shear_cast(h.xz);
        self.yz = T::shear_cast(h.yz);
        self.yx = T::shear_cast(h.yx);
        self.zx = T::shear_cast(h.zx);
        self.zy = T::shear_cast(h.zy);
    }

    /// Port of `template <class S> void getValue(S& XY, ...) const`.
    ///
    /// Writes each field into the provided mutable references, converting via `S(field)`.
    #[inline]
    pub fn get_value<S>(&self, xy: &mut S, xz: &mut S, yz: &mut S, yx: &mut S, zx: &mut S, zy: &mut S)
    where
        S: ShearCast<T>,
    {
        *xy = S::shear_cast(self.xy);
        *xz = S::shear_cast(self.xz);
        *yz = S::shear_cast(self.yz);
        *yx = S::shear_cast(self.yx);
        *zx = S::shear_cast(self.zx);
        *zy = S::shear_cast(self.zy);
    }

    /// Port of `template <class S> void getValue(Shear6<S>& h) const`.
    ///
    /// Copies all six fields into `h`, converting via `S(field)`.
    #[inline]
    pub fn get_value_into<S>(&self, h: &mut Shear6<S>)
    where
        S: Copy + ShearCast<T>,
    {
        h.xy = S::shear_cast(self.xy);
        h.xz = S::shear_cast(self.xz);
        h.yz = S::shear_cast(self.yz);
        h.yx = S::shear_cast(self.yx);
        h.zx = S::shear_cast(self.zx);
        h.zy = S::shear_cast(self.zy);
    }

    /// Port of `T* getValue()` (mutable raw-pointer overload).
    ///
    /// C++ returns a raw pointer to the first field; this safe Rust equivalent
    /// returns the six mutable field references as an array, preserving
    /// in-order access without unsafe pointer arithmetic.
    #[inline]
    pub fn as_array_mut(&mut self) -> [&mut T; 6] {
        [&mut self.xy, &mut self.xz, &mut self.yz, &mut self.yx, &mut self.zx, &mut self.zy]
    }

    /// Port of `const T* getValue() const`.
    ///
    /// Returns all six fields as a by-value `[T; 6]` (T: Copy). Replaces the
    /// C++ `(const T*) &xy` pattern; no unsafe required.
    #[inline]
    pub fn to_array(&self) -> [T; 6] {
        [self.xy, self.xz, self.yz, self.yx, self.zx, self.zy]
    }
}

// ============================================================================
// Equality — port of `operator==` and `operator!=`
// ============================================================================

impl<T: PartialEq> PartialEq for Shear6<T> {
    /// Port of `template <class S> bool operator==(const Shear6<S>& h) const`.
    ///
    /// Component-wise equality across all six fields.
    #[inline]
    fn eq(&self, h: &Shear6<T>) -> bool {
        self.xy == h.xy
            && self.xz == h.xz
            && self.yz == h.yz
            && self.yx == h.yx
            && self.zx == h.zx
            && self.zy == h.zy
    }
}

// ============================================================================
// Approximate comparison
// ============================================================================

impl<T: Scalar> Shear6<T> {
    /// Port of `bool equalWithAbsError(const Shear6<T>& h, T e) const`.
    ///
    /// Returns true if all components differ by no more than the absolute error `e`:
    /// `abs(this[i] - h[i]) <= e` for all i in 0..6.
    #[inline]
    pub fn equal_with_abs_error(&self, h: &Shear6<T>, e: T) -> bool {
        equal_with_abs_error(self.xy, h.xy, e)
            && equal_with_abs_error(self.xz, h.xz, e)
            && equal_with_abs_error(self.yz, h.yz, e)
            && equal_with_abs_error(self.yx, h.yx, e)
            && equal_with_abs_error(self.zx, h.zx, e)
            && equal_with_abs_error(self.zy, h.zy, e)
    }

    /// Port of `bool equalWithRelError(const Shear6<T>& h, T e) const`.
    ///
    /// Returns true if all components differ by no more than the relative error `e`:
    /// `abs(this[i] - h[i]) <= e * abs(this[i])` for all i in 0..6.
    #[inline]
    pub fn equal_with_rel_error(&self, h: &Shear6<T>, e: T) -> bool {
        equal_with_rel_error(self.xy, h.xy, e)
            && equal_with_rel_error(self.xz, h.xz, e)
            && equal_with_rel_error(self.yz, h.yz, e)
            && equal_with_rel_error(self.yx, h.yx, e)
            && equal_with_rel_error(self.zx, h.zx, e)
            && equal_with_rel_error(self.zy, h.zy, e)
    }
}

// ============================================================================
// Arithmetic — component-wise shear+shear, shear+scalar
// ============================================================================

impl<T: Scalar> AddAssign for Shear6<T> {
    /// Port of `const Shear6& operator+=(const Shear6& h)`.
    #[inline]
    fn add_assign(&mut self, h: Shear6<T>) {
        self.xy = self.xy + h.xy;
        self.xz = self.xz + h.xz;
        self.yz = self.yz + h.yz;
        self.yx = self.yx + h.yx;
        self.zx = self.zx + h.zx;
        self.zy = self.zy + h.zy;
    }
}

impl<T: Scalar> Add for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator+(const Shear6& h) const`.
    #[inline]
    fn add(self, h: Shear6<T>) -> Shear6<T> {
        Shear6::new(
            self.xy + h.xy,
            self.xz + h.xz,
            self.yz + h.yz,
            self.yx + h.yx,
            self.zx + h.zx,
            self.zy + h.zy,
        )
    }
}

impl<T: Scalar> SubAssign for Shear6<T> {
    /// Port of `const Shear6& operator-=(const Shear6& h)`.
    #[inline]
    fn sub_assign(&mut self, h: Shear6<T>) {
        self.xy = self.xy - h.xy;
        self.xz = self.xz - h.xz;
        self.yz = self.yz - h.yz;
        self.yx = self.yx - h.yx;
        self.zx = self.zx - h.zx;
        self.zy = self.zy - h.zy;
    }
}

impl<T: Scalar> Sub for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator-(const Shear6& h) const`.
    #[inline]
    fn sub(self, h: Shear6<T>) -> Shear6<T> {
        Shear6::new(
            self.xy - h.xy,
            self.xz - h.xz,
            self.yz - h.yz,
            self.yx - h.yx,
            self.zx - h.zx,
            self.zy - h.zy,
        )
    }
}

impl<T: Scalar> Neg for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator-() const` — component-wise negation.
    #[inline]
    fn neg(self) -> Shear6<T> {
        Shear6::new(-self.xy, -self.xz, -self.yz, -self.yx, -self.zx, -self.zy)
    }
}

impl<T: Scalar> Shear6<T> {
    /// Port of `const Shear6& negate()` — in-place component-wise negation.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.xy = -self.xy;
        self.xz = -self.xz;
        self.yz = -self.yz;
        self.yx = -self.yx;
        self.zx = -self.zx;
        self.zy = -self.zy;
        self
    }
}

impl<T: Scalar> MulAssign for Shear6<T> {
    /// Port of `const Shear6& operator*=(const Shear6& h)` — component-wise multiply.
    #[inline]
    fn mul_assign(&mut self, h: Shear6<T>) {
        self.xy = self.xy * h.xy;
        self.xz = self.xz * h.xz;
        self.yz = self.yz * h.yz;
        self.yx = self.yx * h.yx;
        self.zx = self.zx * h.zx;
        self.zy = self.zy * h.zy;
    }
}

impl<T: Scalar> MulAssign<T> for Shear6<T> {
    /// Port of `const Shear6& operator*=(T a)` — scalar multiply.
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.xy = self.xy * a;
        self.xz = self.xz * a;
        self.yz = self.yz * a;
        self.yx = self.yx * a;
        self.zx = self.zx * a;
        self.zy = self.zy * a;
    }
}

impl<T: Scalar> Mul for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator*(const Shear6& h) const` — component-wise multiply.
    #[inline]
    fn mul(self, h: Shear6<T>) -> Shear6<T> {
        Shear6::new(
            self.xy * h.xy,
            self.xz * h.xz,
            self.yz * h.yz,
            self.yx * h.yx,
            self.zx * h.zx,
            self.zy * h.zy,
        )
    }
}

impl<T: Scalar> Mul<T> for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator*(T a) const` — scalar multiply.
    #[inline]
    fn mul(self, a: T) -> Shear6<T> {
        Shear6::new(
            self.xy * a,
            self.xz * a,
            self.yz * a,
            self.yx * a,
            self.zx * a,
            self.zy * a,
        )
    }
}

impl<T: Scalar> DivAssign for Shear6<T> {
    /// Port of `const Shear6& operator/=(const Shear6& h)` — component-wise divide.
    #[inline]
    fn div_assign(&mut self, h: Shear6<T>) {
        self.xy = self.xy / h.xy;
        self.xz = self.xz / h.xz;
        self.yz = self.yz / h.yz;
        self.yx = self.yx / h.yx;
        self.zx = self.zx / h.zx;
        self.zy = self.zy / h.zy;
    }
}

impl<T: Scalar> DivAssign<T> for Shear6<T> {
    /// Port of `const Shear6& operator/=(T a)` — scalar divide.
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.xy = self.xy / a;
        self.xz = self.xz / a;
        self.yz = self.yz / a;
        self.yx = self.yx / a;
        self.zx = self.zx / a;
        self.zy = self.zy / a;
    }
}

impl<T: Scalar> Div for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator/(const Shear6& h) const` — component-wise divide.
    #[inline]
    fn div(self, h: Shear6<T>) -> Shear6<T> {
        Shear6::new(
            self.xy / h.xy,
            self.xz / h.xz,
            self.yz / h.yz,
            self.yx / h.yx,
            self.zx / h.zx,
            self.zy / h.zy,
        )
    }
}

impl<T: Scalar> Div<T> for Shear6<T> {
    type Output = Shear6<T>;
    /// Port of `Shear6 operator/(T a) const` — scalar divide.
    #[inline]
    fn div(self, a: T) -> Shear6<T> {
        Shear6::new(
            self.xy / a,
            self.xz / a,
            self.yz / a,
            self.yx / a,
            self.zx / a,
            self.zy / a,
        )
    }
}

// ============================================================================
// Numeric limits — port of `baseTypeLowest/Max/Smallest/Epsilon` and `dimensions`
// ============================================================================

impl<T: VecLimits> Shear6<T> {
    /// Port of `static T baseTypeLowest()` — `std::numeric_limits<T>::lowest()`.
    #[inline]
    pub fn base_type_lowest() -> T {
        T::LOWEST
    }

    /// Port of `static T baseTypeMax()` — `std::numeric_limits<T>::max()`.
    #[inline]
    pub fn base_type_max() -> T {
        T::MAX
    }

    /// Port of `static T baseTypeSmallest()` — `std::numeric_limits<T>::min()`.
    #[inline]
    pub fn base_type_smallest() -> T {
        T::SMALLEST
    }

    /// Port of `static T baseTypeEpsilon()` — `std::numeric_limits<T>::epsilon()`.
    #[inline]
    pub fn base_type_epsilon() -> T {
        T::EPSILON
    }
}

impl<T> Shear6<T> {
    /// Port of `static unsigned int dimensions()` — always 6.
    #[inline]
    pub const fn dimensions() -> usize {
        6
    }
}

// ============================================================================
// Display — port of `operator<<(ostream&, const Shear6<T>&)`
// ============================================================================

impl<T: fmt::Display> fmt::Display for Shear6<T> {
    /// Port of `operator<<`: formats as `(xy xz yz yx zx zy)`.
    ///
    /// Note: the C++ implementation has a missing space before `h.yx`
    /// (it writes `h.yz << h.yx` without a separator). This port reproduces
    /// that behavior faithfully for 1:1 parity.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} {} {}{} {} {})",
            self.xy, self.xz, self.yz, self.yx, self.zx, self.zy
        )
    }
}

// ============================================================================
// Reverse scalar multiplication — port of `operator*(S a, const Shear6<T>& h)`
// ============================================================================

/// Port of `template <class S, class T> Shear6<T> operator*(S a, const Shear6<T>& h)`.
///
/// Reverse-order scalar multiplication: `a * h` where `a: S` and `h: Shear6<T>`.
/// The C++ template converts `a` to `T` implicitly via `T(a * h.xy)` — here the
/// scalar `a` must be the same type `T` (the Imath typedefs only use homogeneous
/// types), matching the same-type instantiation that OpenEXR/IlmImf uses.
///
/// Implemented via a macro for the concrete types, mirroring the Vec3 approach.
macro_rules! impl_scalar_mul_shear6 {
    ($t:ty) => {
        impl Mul<Shear6<$t>> for $t {
            type Output = Shear6<$t>;
            /// Port of `operator*(S a, const Shear6<T>& h)` — `a * h`.
            #[inline]
            fn mul(self, h: Shear6<$t>) -> Shear6<$t> {
                Shear6::new(
                    self * h.xy,
                    self * h.xz,
                    self * h.yz,
                    self * h.yx,
                    self * h.zx,
                    self * h.zy,
                )
            }
        }
    };
}

// Only instantiate for types that implement `Scalar` (f32, f64, i32).
// i16 and i64 are not `Scalar`-bounded in this crate (no `Shear6<i16>` typedef exists),
// so they are excluded — matching the Vec3 pattern where scalar-mul is macro-generated
// per concrete type but only for those that actually satisfy the trait bounds.
impl_scalar_mul_shear6!(f32);
impl_scalar_mul_shear6!(f64);
impl_scalar_mul_shear6!(i32);

// ============================================================================
// ShearCast — the Rust analogue of the C++ `T(s)` functional cast used by the
// cross-type constructors and setValue/getValue.
// ============================================================================

/// Element-type conversion for shear cross-type constructors and set/getValue.
///
/// Mirrors [`crate::vec::VecCast`]: `T: ShearCast<S>` reads "a `Shear6<S>` element
/// can be converted to `T`". Implemented for every ordered pair of the concrete
/// element types Imath's shear typedefs use (`f32`, `f64`, `i16`, `i32`, `i64`).
///
/// The `shear_zero()` associated function provides the `T(0)` literal needed by
/// the three-component constructors that zero-initialize yx/zx/zy.
pub trait ShearCast<S>: Sized {
    /// Convert one element, matching C++ `T(s)` for `T = Self`, source `S`.
    fn shear_cast(s: S) -> Self;
    /// The zero literal for `T`, used to zero-init yx/zx/zy in 3-component ctors.
    fn shear_zero() -> Self;
}

macro_rules! impl_shearcast_as {
    ($dst:ty, $src:ty) => {
        impl ShearCast<$src> for $dst {
            #[inline]
            fn shear_cast(s: $src) -> Self {
                s as $dst
            }
            #[inline]
            fn shear_zero() -> Self {
                0 as $dst
            }
        }
    };
}

macro_rules! impl_shearcast_row {
    ($dst:ty) => {
        impl_shearcast_as!($dst, i16);
        impl_shearcast_as!($dst, i32);
        impl_shearcast_as!($dst, i64);
        impl_shearcast_as!($dst, f32);
        impl_shearcast_as!($dst, f64);
    };
}

impl_shearcast_row!(i16);
impl_shearcast_row!(i32);
impl_shearcast_row!(i64);
impl_shearcast_row!(f32);
impl_shearcast_row!(f64);

// ============================================================================
// Typedefs — port of ImathShear.h typedef section
// ============================================================================

/// 3D shear of type `f32` (C++ `typedef Vec3<float> Shear3f`).
///
/// Imath represents the "simple" 3-value shear as a Vec3.
pub type Shear3f = Vec3<f32>;

/// 3D shear of type `f64` (C++ `typedef Vec3<double> Shear3d`).
///
/// Imath represents the "simple" 3-value shear as a Vec3.
pub type Shear3d = Vec3<f64>;

/// Shear6 of type `f32` (C++ `typedef Shear6<float> Shear6f`).
pub type Shear6f = Shear6<f32>;

/// Shear6 of type `f64` (C++ `typedef Shear6<double> Shear6d`).
pub type Shear6d = Shear6<f64>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec3;

    // -- Construction --

    #[test]
    fn default_is_zero() {
        let s = Shear6f::default();
        assert_eq!(s.xy, 0.0);
        assert_eq!(s.xz, 0.0);
        assert_eq!(s.yz, 0.0);
        assert_eq!(s.yx, 0.0);
        assert_eq!(s.zx, 0.0);
        assert_eq!(s.zy, 0.0);
    }

    #[test]
    fn from_xyz_zeros_yx_zx_zy() {
        let s = Shear6f::from_xyz(1.0, 2.0, 3.0);
        assert_eq!((s.xy, s.xz, s.yz), (1.0, 2.0, 3.0));
        assert_eq!((s.yx, s.zx, s.zy), (0.0, 0.0, 0.0));
    }

    #[test]
    fn new_all_six() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!([s.xy, s.xz, s.yz, s.yx, s.zx, s.zy], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_vec3_constructor() {
        let v = Vec3::new(1.0_f32, 2.0, 3.0);
        let s = Shear6f::from_vec3(v);
        assert_eq!((s.xy, s.xz, s.yz), (1.0, 2.0, 3.0));
        assert_eq!((s.yx, s.zx, s.zy), (0.0, 0.0, 0.0));
    }

    #[test]
    fn from_vec3_cast_cross_type() {
        // Build from a Vec3<f64> into Shear6<f32>
        let v = Vec3::new(1.0_f64, 2.0, 3.0);
        let s: Shear6<f32> = Shear6::from_vec3_cast(v);
        assert_eq!((s.xy, s.xz, s.yz), (1.0_f32, 2.0, 3.0));
        assert_eq!((s.yx, s.zx, s.zy), (0.0, 0.0, 0.0));
    }

    #[test]
    fn from_shear6_cross_type() {
        let a = Shear6d::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let b: Shear6f = Shear6::from_shear6(a);
        assert_eq!([b.xy, b.xz, b.yz, b.yx, b.zx, b.zy], [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -- Indexing --

    #[test]
    fn index_all_fields() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!((s[0], s[1], s[2], s[3], s[4], s[5]), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    #[should_panic(expected = "Shear6 index out of range")]
    fn index_oob_panics() {
        let s = Shear6f::default();
        let _ = s[6];
    }

    #[test]
    fn index_mut() {
        let mut s = Shear6f::default();
        s[3] = 7.0;
        assert_eq!(s.yx, 7.0);
    }

    // -- setValue / getValue --

    #[test]
    fn set_value_and_to_array() {
        let mut s = Shear6f::default();
        s.set_value(1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(s.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn get_value_scalar() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let (mut a, mut b, mut c, mut d, mut e, mut g) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
        s.get_value(&mut a, &mut b, &mut c, &mut d, &mut e, &mut g);
        assert_eq!([a, b, c, d, e, g], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn get_value_into_shear() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let mut h = Shear6d::default();
        s.get_value_into(&mut h);
        assert_eq!([h.xy, h.xz, h.yz, h.yx, h.zx, h.zy], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -- Arithmetic --

    #[test]
    fn add_sub() {
        let a = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let b = Shear6f::new(6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let sum = a + b;
        assert_eq!(sum.to_array(), [7.0; 6]);
        let diff = a - b;
        assert_eq!(diff.to_array(), [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn mul_div_shear() {
        let a = Shear6f::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        let b = Shear6f::new(3.0, 3.0, 3.0, 3.0, 3.0, 3.0);
        let m = a * b;
        assert_eq!(m.to_array(), [6.0; 6]);
        let d = b / a;
        assert_eq!(d.to_array(), [1.5; 6]);
    }

    #[test]
    fn mul_div_scalar() {
        let s = Shear6f::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        let m = s * 2.0_f32;
        assert_eq!(m.to_array(), [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]);
        let d = s / 2.0_f32;
        assert_eq!(d.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn neg_and_negate() {
        let a = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let n = -a;
        assert_eq!(n.to_array(), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
        let mut b = a;
        b.negate();
        assert_eq!(b.to_array(), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
    }

    #[test]
    fn reverse_scalar_mul() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let r = 2.0_f32 * s;
        assert_eq!(r.to_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn assign_ops() {
        let mut a = Shear6f::new(6.0, 6.0, 6.0, 6.0, 6.0, 6.0);
        a += Shear6f::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.to_array(), [7.0; 6]);
        a -= Shear6f::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        assert_eq!(a.to_array(), [5.0; 6]);
        a *= Shear6f::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        assert_eq!(a.to_array(), [10.0; 6]);
        a /= Shear6f::new(5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
        assert_eq!(a.to_array(), [2.0; 6]);
        a *= 3.0_f32;
        assert_eq!(a.to_array(), [6.0; 6]);
        a /= 2.0_f32;
        assert_eq!(a.to_array(), [3.0; 6]);
    }

    // -- equalWithAbsError / equalWithRelError --

    #[test]
    fn equal_with_abs_error() {
        let a = Shear6f::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let b = Shear6f::new(1.05, 1.05, 1.05, 1.05, 1.05, 1.05);
        assert!(a.equal_with_abs_error(&b, 0.1));
        assert!(!a.equal_with_abs_error(&b, 0.01));
    }

    #[test]
    fn equal_with_rel_error_test() {
        let a = Shear6f::new(100.0, 100.0, 100.0, 100.0, 100.0, 100.0);
        let b = Shear6f::new(105.0, 105.0, 105.0, 105.0, 105.0, 105.0);
        assert!(a.equal_with_rel_error(&b, 0.1));
        assert!(!a.equal_with_rel_error(&b, 0.01));
    }

    // -- numeric limits and dimensions --

    #[test]
    fn numeric_limits() {
        assert_eq!(Shear6f::base_type_lowest(), f32::MIN);
        assert_eq!(Shear6f::base_type_max(), f32::MAX);
        assert_eq!(Shear6f::base_type_smallest(), f32::MIN_POSITIVE);
        assert_eq!(Shear6f::base_type_epsilon(), f32::EPSILON);
    }

    #[test]
    fn dimensions_is_6() {
        assert_eq!(Shear6::<f32>::dimensions(), 6);
        assert_eq!(Shear6::<f64>::dimensions(), 6);
    }

    // -- typedef smoke tests --

    #[test]
    fn typedef_shear3() {
        // Shear3f and Shear3d are Vec3 aliases — confirm they exist and work.
        let _s3f: Shear3f = Vec3::new(1.0_f32, 2.0, 3.0);
        let _s3d: Shear3d = Vec3::new(1.0_f64, 2.0, 3.0);
    }

    #[test]
    fn typedef_shear6fd() {
        let _sf = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let _sd = Shear6d::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    // -- Display --

    #[test]
    fn display() {
        let s = Shear6f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let formatted = format!("{s}");
        // Reproduce C++ formatting: "(xy xz yz{yx} zx zy)" — missing space before yx.
        assert_eq!(formatted, "(1 2 34 5 6)");
    }
}
