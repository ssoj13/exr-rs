//! Port of Imath ImathColor.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathColor.h>
//!
//! ## Types
//! - [`Color3`] — 3-channel colour (inherits `Vec3<T>` via `Deref`/`DerefMut`).
//! - [`Color4`] — 4-channel colour with fields `r, g, b, a`.
//!
//! ## Inheritance: `Color3<T> : public Vec3<T>`
//! C++ `Color3` publicly inherits from `Vec3<T>`, so the three field names are
//! `x`, `y`, `z` (not `r`, `g`, `b`). We replicate this via composition: `Color3`
//! wraps a `pub vec: Vec3<T>` and `Deref`/`DerefMut` to it, exactly as `Euler` does.
//! `#[repr(C)]` keeps layout identical to the C++ base subobject.
//!
//! ## Trait bound: `ColorScalar` instead of `Scalar`
//! C++ `Color3<T>` / `Color4<T>` are instantiated with `float`, `half`, and
//! `unsigned char`. The full [`Scalar`] trait requires `Neg + Rem` which `u8`
//! cannot satisfy under Rust's orphan rules. [`ColorScalar`] is the weaker bound:
//! it omits `Neg` and `Rem`. Methods that require negation carry an additional
//! `+ Neg<Output = T>` bound so they are simply absent for `u8`, matching C++
//! behaviour (negating `Color3<unsigned char>` is valid but unusual, and
//! `Color3c` users in practice never negate).
//!
//! ## `PackedColor`
//! Typedef `PackedColor = u32` (C++ `typedef unsigned int PackedColor`).

use core::fmt;
use core::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::half::Half;
use crate::traits::ColorScalar;
use crate::vec::Vec3;

// ============================================================================
// Color3<T>
// ============================================================================

/// 3-channel colour class that inherits from `Vec3<T>` (port of C++ `Color3<T>`).
///
/// Fields are accessed as `.x`, `.y`, `.z` through the `Deref` to [`Vec3<T>`],
/// matching the C++ note: *"because Color3 inherits from Vec3, its member fields
/// are called `x`, `y`, and `z`"*.
///
/// `#[repr(C)]` ensures the layout is three contiguous `T`, identical to the
/// C++ `Color3<T>` base-class subobject.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Color3<T> {
    /// The underlying Vec3 — provides inherited Vec3 operations via Deref.
    pub vec: Vec3<T>,
}

// --- Deref: Color3 "is a" Vec3 (C++ public inheritance) ---

impl<T> Deref for Color3<T> {
    type Target = Vec3<T>;
    #[inline]
    fn deref(&self) -> &Vec3<T> {
        &self.vec
    }
}

impl<T> DerefMut for Color3<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec3<T> {
        &mut self.vec
    }
}

// --- Constructors ---

impl<T: ColorScalar> Default for Color3<T> {
    /// Port of `Color3()` — no-init default; safe Rust zero-initialises.
    #[inline]
    fn default() -> Self {
        Color3 { vec: Vec3 { x: T::ZERO, y: T::ZERO, z: T::ZERO } }
    }
}

impl<T: Copy> Color3<T> {
    /// Port of `explicit Color3(T a)` — initialize to `(a, a, a)`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Color3 { vec: Vec3::new(a, a, a) }
    }

    /// Port of `Color3(T a, T b, T c)` — initialize to `(a, b, c)`.
    #[inline]
    pub const fn new(a: T, b: T, c: T) -> Self {
        Color3 { vec: Vec3::new(a, b, c) }
    }

    /// Port of `Color3(const Color3& c)` — copy constructor.
    #[inline]
    pub const fn from_color3(c: Color3<T>) -> Self {
        Color3 { vec: c.vec }
    }

    /// Port of `template <class S> Color3(const Vec3<S>& v)` — construct from `Vec3`.
    #[inline]
    pub fn from_vec3(v: Vec3<T>) -> Self {
        Color3 { vec: v }
    }
}

// --- Equality ---

impl<T: PartialEq> PartialEq for Color3<T> {
    /// Port of `operator==`: delegates to `Vec3::operator==`.
    #[inline]
    fn eq(&self, other: &Color3<T>) -> bool {
        self.vec == other.vec
    }
}

// --- Arithmetic: Color3 OP Color3 ---

impl<T: ColorScalar> Add for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator+(const Color3&)`.
    #[inline]
    fn add(self, c: Color3<T>) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x + c.x, self.y + c.y, self.z + c.z) }
    }
}

impl<T: ColorScalar> AddAssign for Color3<T> {
    /// Port of `operator+=(const Color3&)`.
    #[inline]
    fn add_assign(&mut self, c: Color3<T>) {
        self.vec.x = self.vec.x + c.vec.x;
        self.vec.y = self.vec.y + c.vec.y;
        self.vec.z = self.vec.z + c.vec.z;
    }
}

impl<T: ColorScalar> Sub for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator-(const Color3&)`.
    #[inline]
    fn sub(self, c: Color3<T>) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x - c.x, self.y - c.y, self.z - c.z) }
    }
}

impl<T: ColorScalar> SubAssign for Color3<T> {
    /// Port of `operator-=(const Color3&)`.
    #[inline]
    fn sub_assign(&mut self, c: Color3<T>) {
        self.vec.x = self.vec.x - c.vec.x;
        self.vec.y = self.vec.y - c.vec.y;
        self.vec.z = self.vec.z - c.vec.z;
    }
}

/// Port of unary `operator-()` — only available when `T: Neg`.
///
/// `Color3<u8>` does not implement this (unsigned chars don't negate), matching
/// the C++ behaviour where negating a `Color3<unsigned char>` would wrap.
impl<T: ColorScalar + Neg<Output = T>> Neg for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator-() const`.
    #[inline]
    fn neg(self) -> Color3<T> {
        Color3 { vec: Vec3::new(-self.x, -self.y, -self.z) }
    }
}

impl<T: ColorScalar> Mul for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator*(const Color3&)`.
    #[inline]
    fn mul(self, c: Color3<T>) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x * c.x, self.y * c.y, self.z * c.z) }
    }
}

impl<T: ColorScalar> Mul<T> for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator*(T a)`.
    #[inline]
    fn mul(self, a: T) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x * a, self.y * a, self.z * a) }
    }
}

impl<T: ColorScalar> MulAssign for Color3<T> {
    /// Port of `operator*=(const Color3&)`.
    #[inline]
    fn mul_assign(&mut self, c: Color3<T>) {
        self.vec.x = self.vec.x * c.vec.x;
        self.vec.y = self.vec.y * c.vec.y;
        self.vec.z = self.vec.z * c.vec.z;
    }
}

impl<T: ColorScalar> MulAssign<T> for Color3<T> {
    /// Port of `operator*=(T a)`.
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.vec.x = self.vec.x * a;
        self.vec.y = self.vec.y * a;
        self.vec.z = self.vec.z * a;
    }
}

impl<T: ColorScalar> Div for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator/(const Color3&)`.
    #[inline]
    fn div(self, c: Color3<T>) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x / c.x, self.y / c.y, self.z / c.z) }
    }
}

impl<T: ColorScalar> Div<T> for Color3<T> {
    type Output = Color3<T>;
    /// Port of `operator/(T a)`.
    #[inline]
    fn div(self, a: T) -> Color3<T> {
        Color3 { vec: Vec3::new(self.x / a, self.y / a, self.z / a) }
    }
}

impl<T: ColorScalar> DivAssign for Color3<T> {
    /// Port of `operator/=(const Color3&)`.
    #[inline]
    fn div_assign(&mut self, c: Color3<T>) {
        self.vec.x = self.vec.x / c.vec.x;
        self.vec.y = self.vec.y / c.vec.y;
        self.vec.z = self.vec.z / c.vec.z;
    }
}

impl<T: ColorScalar> DivAssign<T> for Color3<T> {
    /// Port of `operator/=(T a)`.
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.vec.x = self.vec.x / a;
        self.vec.y = self.vec.y / a;
        self.vec.z = self.vec.z / a;
    }
}

impl<T: ColorScalar + Neg<Output = T>> Color3<T> {
    /// Port of `negate()` — component-wise multiply by -1 in place.
    ///
    /// Only available when `T: Neg` (i.e. not for `Color3<u8>`).
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.vec.x = -self.vec.x;
        self.vec.y = -self.vec.y;
        self.vec.z = -self.vec.z;
        self
    }
}

impl<T> Color3<T> {
    /// Port of `dimensions()` — always 3 for Color3.
    #[inline]
    pub const fn dimensions() -> usize {
        3
    }
}

// --- equalWithAbsError / equalWithRelError ---
//
// These are inherited from Vec3 and available via Deref on any `Color3<T: Scalar>`.

// --- Display / Debug ---

impl<T: fmt::Display> fmt::Display for Color3<T> {
    /// Port of the inherited `Vec3` stream `operator<<`: `(x y z)`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.vec.x, self.vec.y, self.vec.z)
    }
}

impl<T: fmt::Debug> fmt::Debug for Color3<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color3({:?}, {:?}, {:?})", self.vec.x, self.vec.y, self.vec.z)
    }
}

// ============================================================================
// Color4<T>
// ============================================================================

/// 4-channel colour class (port of C++ `Color4<T>`).
///
/// Fields `r`, `g`, `b`, `a` are public, as in C++. `#[repr(C)]` guarantees
/// the layout is four contiguous `T`, matching the C++ struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Color4<T> {
    /// Red channel.
    pub r: T,
    /// Green channel.
    pub g: T,
    /// Blue channel.
    pub b: T,
    /// Alpha channel.
    pub a: T,
}

// --- Constructors ---

impl<T: ColorScalar> Default for Color4<T> {
    /// Port of `Color4()` — no-init default; safe Rust zero-initialises.
    #[inline]
    fn default() -> Self {
        Color4 { r: T::ZERO, g: T::ZERO, b: T::ZERO, a: T::ZERO }
    }
}

impl<T: Copy> Color4<T> {
    /// Port of `explicit Color4(T a)` — initialize to `(a, a, a, a)`.
    #[inline]
    pub fn from_scalar(x: T) -> Self {
        Color4 { r: x, g: x, b: x, a: x }
    }

    /// Port of `Color4(T a, T b, T c, T d)` — initialize to `(a, b, c, d)`.
    #[inline]
    pub const fn new(r: T, g: T, b: T, a: T) -> Self {
        Color4 { r, g, b, a }
    }

    /// Port of `Color4(const Color4& v)` — copy constructor.
    #[inline]
    pub const fn from_color4(v: Color4<T>) -> Self {
        Color4 { r: v.r, g: v.g, b: v.b, a: v.a }
    }

    /// Port of `template <class S> Color4(const Color4<S>& v)` — cross-element conversion.
    #[inline]
    pub fn from_color4_cast<S: Copy>(v: Color4<S>) -> Self
    where
        T: ColorCast<S>,
    {
        Color4 {
            r: T::color_from(v.r),
            g: T::color_from(v.g),
            b: T::color_from(v.b),
            a: T::color_from(v.a),
        }
    }
}

// --- Index operator ---

impl<T> core::ops::Index<usize> for Color4<T> {
    type Output = T;
    /// Port of `T& operator[](int i)`.
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            3 => &self.a,
            _ => panic!("Color4 index out of range: {i}"),
        }
    }
}

impl<T> core::ops::IndexMut<usize> for Color4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            3 => &mut self.a,
            _ => panic!("Color4 index out of range: {i}"),
        }
    }
}

// --- Equality ---

impl<T: PartialEq> PartialEq for Color4<T> {
    /// Port of `operator==`: `r == v.r && g == v.g && b == v.b && a == v.a`.
    #[inline]
    fn eq(&self, v: &Color4<T>) -> bool {
        self.r == v.r && self.g == v.g && self.b == v.b && self.a == v.a
    }
}

// --- setValue / getValue (Sb compatibility) ---

impl<T: Copy> Color4<T> {
    /// Port of `template <class S> void setValue(S a, S b, S c, S d)`.
    #[inline]
    pub fn set_value_rgba<S: Copy>(&mut self, a: S, b: S, c: S, d: S)
    where
        T: ColorCast<S>,
    {
        self.r = T::color_from(a);
        self.g = T::color_from(b);
        self.b = T::color_from(c);
        self.a = T::color_from(d);
    }

    /// Port of `template <class S> void setValue(const Color4<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: Color4<S>)
    where
        T: ColorCast<S>,
    {
        self.r = T::color_from(v.r);
        self.g = T::color_from(v.g);
        self.b = T::color_from(v.b);
        self.a = T::color_from(v.a);
    }

    /// Port of `template <class S> void getValue(S& a, S& b, S& c, S& d) const`.
    #[inline]
    pub fn get_value_rgba<S>(&self, a: &mut S, b: &mut S, c: &mut S, d: &mut S)
    where
        S: ColorCast<T>,
    {
        *a = S::color_from(self.r);
        *b = S::color_from(self.g);
        *c = S::color_from(self.b);
        *d = S::color_from(self.a);
    }

    /// Port of `template <class S> void getValue(Color4<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Color4<S>)
    where
        S: Copy + ColorCast<T>,
    {
        v.r = S::color_from(self.r);
        v.g = S::color_from(self.g);
        v.b = S::color_from(self.b);
        v.a = S::color_from(self.a);
    }

    /// Port of `T* getValue() const` — copies fields into a by-value array.
    ///
    /// The C++ returns a raw pointer via `reinterpret_cast`. Rust avoids raw
    /// pointers; the safe analogue copies the fields into `[T; 4]`.
    #[inline]
    pub fn to_array(&self) -> [T; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

// --- Numeric limits and dimensions ---

impl<T> Color4<T> {
    /// Port of `dimensions()` — always 4 for Color4.
    #[inline]
    pub const fn dimensions() -> usize {
        4
    }
}

impl<T: crate::vec::VecLimits> Color4<T> {
    /// Port of `baseTypeLowest()`.
    #[inline]
    pub fn base_type_lowest() -> T { T::LOWEST }
    /// Port of `baseTypeMax()`.
    #[inline]
    pub fn base_type_max() -> T { T::MAX }
    /// Port of `baseTypeSmallest()`.
    #[inline]
    pub fn base_type_smallest() -> T { T::SMALLEST }
    /// Port of `baseTypeEpsilon()`.
    #[inline]
    pub fn base_type_epsilon() -> T { T::EPSILON }
}

// --- Arithmetic: Color4 OP Color4 ---

impl<T: ColorScalar> Add for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn add(self, v: Color4<T>) -> Color4<T> {
        Color4 { r: self.r + v.r, g: self.g + v.g, b: self.b + v.b, a: self.a + v.a }
    }
}

impl<T: ColorScalar> AddAssign for Color4<T> {
    #[inline]
    fn add_assign(&mut self, v: Color4<T>) {
        self.r = self.r + v.r;
        self.g = self.g + v.g;
        self.b = self.b + v.b;
        self.a = self.a + v.a;
    }
}

impl<T: ColorScalar> Sub for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn sub(self, v: Color4<T>) -> Color4<T> {
        Color4 { r: self.r - v.r, g: self.g - v.g, b: self.b - v.b, a: self.a - v.a }
    }
}

impl<T: ColorScalar> SubAssign for Color4<T> {
    #[inline]
    fn sub_assign(&mut self, v: Color4<T>) {
        self.r = self.r - v.r;
        self.g = self.g - v.g;
        self.b = self.b - v.b;
        self.a = self.a - v.a;
    }
}

/// Port of unary `operator-()` for Color4 — only when `T: Neg`.
impl<T: ColorScalar + Neg<Output = T>> Neg for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn neg(self) -> Color4<T> {
        Color4 { r: -self.r, g: -self.g, b: -self.b, a: -self.a }
    }
}

impl<T: ColorScalar> Mul for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn mul(self, v: Color4<T>) -> Color4<T> {
        Color4 { r: self.r * v.r, g: self.g * v.g, b: self.b * v.b, a: self.a * v.a }
    }
}

impl<T: ColorScalar> Mul<T> for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn mul(self, a: T) -> Color4<T> {
        Color4 { r: self.r * a, g: self.g * a, b: self.b * a, a: self.a * a }
    }
}

impl<T: ColorScalar> MulAssign for Color4<T> {
    #[inline]
    fn mul_assign(&mut self, v: Color4<T>) {
        self.r = self.r * v.r;
        self.g = self.g * v.g;
        self.b = self.b * v.b;
        self.a = self.a * v.a;
    }
}

impl<T: ColorScalar> MulAssign<T> for Color4<T> {
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.r = self.r * a;
        self.g = self.g * a;
        self.b = self.b * a;
        self.a = self.a * a;
    }
}

impl<T: ColorScalar> Div for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn div(self, v: Color4<T>) -> Color4<T> {
        Color4 { r: self.r / v.r, g: self.g / v.g, b: self.b / v.b, a: self.a / v.a }
    }
}

impl<T: ColorScalar> Div<T> for Color4<T> {
    type Output = Color4<T>;
    #[inline]
    fn div(self, a: T) -> Color4<T> {
        Color4 { r: self.r / a, g: self.g / a, b: self.b / a, a: self.a / a }
    }
}

impl<T: ColorScalar> DivAssign for Color4<T> {
    #[inline]
    fn div_assign(&mut self, v: Color4<T>) {
        self.r = self.r / v.r;
        self.g = self.g / v.g;
        self.b = self.b / v.b;
        self.a = self.a / v.a;
    }
}

impl<T: ColorScalar> DivAssign<T> for Color4<T> {
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.r = self.r / a;
        self.g = self.g / a;
        self.b = self.b / a;
        self.a = self.a / a;
    }
}

impl<T: ColorScalar + Neg<Output = T>> Color4<T> {
    /// Port of `negate()` — component-wise multiply by -1 in place.
    ///
    /// Only available when `T: Neg` (i.e. not for `Color4<u8>`).
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.r = -self.r;
        self.g = -self.g;
        self.b = -self.b;
        self.a = -self.a;
        self
    }
}

// --- Reverse multiplication: `S * Color4<T>` ---

macro_rules! impl_scalar_mul_color4 {
    ($t:ty) => {
        impl Mul<Color4<$t>> for $t {
            type Output = Color4<$t>;
            /// Port of `operator*(S a, const Color4<T>& v)`.
            #[inline]
            fn mul(self, v: Color4<$t>) -> Color4<$t> {
                Color4 { r: self * v.r, g: self * v.g, b: self * v.b, a: self * v.a }
            }
        }
    };
}
impl_scalar_mul_color4!(u8);
impl_scalar_mul_color4!(f32);
impl_scalar_mul_color4!(f64);
impl_scalar_mul_color4!(Half);

// --- Display / Debug ---

impl<T: fmt::Display> fmt::Display for Color4<T> {
    /// Port of `operator<<(ostream&, const Color4<T>&)`: `(r g b a)`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {} {})", self.r, self.g, self.b, self.a)
    }
}

impl<T: fmt::Debug> fmt::Debug for Color4<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color4({:?}, {:?}, {:?}, {:?})", self.r, self.g, self.b, self.a)
    }
}

// ============================================================================
// ColorCast — element-type conversion for Color4 cross-base-type constructors
// ============================================================================

/// Element-type conversion for `Color4` cross-base-type constructors.
///
/// Mirrors [`crate::vec::VecCast`]: `T: ColorCast<S>` means a `Color4<S>` can be
/// converted to `Color4<T>` (C++ `T(v.r)` cast). Implemented for every ordered
/// pair used by the `Color4*` typedefs (`f32`, `f64`, `u8`, `Half`).
pub trait ColorCast<S>: Sized {
    fn color_from(s: S) -> Self;
}

macro_rules! impl_colorcast_as {
    ($dst:ty, $src:ty) => {
        impl ColorCast<$src> for $dst {
            #[inline]
            fn color_from(s: $src) -> Self { s as $dst }
        }
    };
}

impl_colorcast_as!(f32, f32);
impl_colorcast_as!(f32, f64);
impl_colorcast_as!(f32, u8);
impl_colorcast_as!(f64, f32);
impl_colorcast_as!(f64, f64);
impl_colorcast_as!(f64, u8);
impl_colorcast_as!(u8,  f32);
impl_colorcast_as!(u8,  f64);
impl_colorcast_as!(u8,  u8);

impl ColorCast<f32>  for Half { #[inline] fn color_from(s: f32)  -> Self { Half::new(s) } }
impl ColorCast<f64>  for Half { #[inline] fn color_from(s: f64)  -> Self { Half::new(s as f32) } }
impl ColorCast<u8>   for Half { #[inline] fn color_from(s: u8)   -> Self { Half::new(s as f32) } }
impl ColorCast<Half> for Half { #[inline] fn color_from(s: Half) -> Self { s } }
impl ColorCast<Half> for f32  { #[inline] fn color_from(s: Half) -> Self { s.to_f32() } }
impl ColorCast<Half> for f64  { #[inline] fn color_from(s: Half) -> Self { s.to_f32() as f64 } }
impl ColorCast<Half> for u8   { #[inline] fn color_from(s: Half) -> Self { s.to_f32() as u8 } }

// ============================================================================
// VecLimits for u8 — required by Color3c/Color4c baseType* accessors
// ============================================================================

impl crate::vec::VecLimits for u8 {
    const LOWEST: Self  = u8::MIN;
    const MAX: Self     = u8::MAX;
    // C++ numeric_limits<unsigned char>::min() == 0 for unsigned types
    const SMALLEST: Self = 0;
    const EPSILON: Self  = 0;
}

// ============================================================================
// Typedefs (1:1 with ImathColor.h)
// ============================================================================

/// 3 `f32` channels (C++ `Color3f`).
pub type Color3f = Color3<f32>;
/// 3 [`Half`] channels (C++ `Color3h`).
pub type Color3h = Color3<Half>;
/// 3 `u8` channels (C++ `Color3c`, `unsigned char`).
pub type Color3c = Color3<u8>;

/// 3 [`Half`] channels — short alias (C++ `C3h`).
pub type C3h = Color3<Half>;
/// 3 `f32` channels — short alias (C++ `C3f`).
pub type C3f = Color3<f32>;
/// 3 `u8` channels — short alias (C++ `C3c`).
pub type C3c = Color3<u8>;

/// 4 `f32` channels (C++ `Color4f`).
pub type Color4f = Color4<f32>;
/// 4 [`Half`] channels (C++ `Color4h`).
pub type Color4h = Color4<Half>;
/// 4 `u8` channels (C++ `Color4c`, `unsigned char`).
pub type Color4c = Color4<u8>;

/// 4 `f32` channels — short alias (C++ `C4f`).
pub type C4f = Color4<f32>;
/// 4 [`Half`] channels — short alias (C++ `C4h`).
pub type C4h = Color4<Half>;
/// 4 `u8` channels — short alias (C++ `C4c`).
pub type C4c = Color4<u8>;

/// Packed 32-bit RGBA (C++ `typedef unsigned int PackedColor`).
pub type PackedColor = u32;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::Half;

    #[test]
    fn color3_construct_deref() {
        let c = Color3f::new(1.0, 2.0, 3.0);
        assert_eq!((c.x, c.y, c.z), (1.0_f32, 2.0, 3.0));
        assert_eq!(c[0], 1.0_f32);
        assert_eq!(c[2], 3.0_f32);
        let s = Color3f::from_scalar(5.0);
        assert_eq!((s.x, s.y, s.z), (5.0_f32, 5.0, 5.0));
        assert_eq!(Color3f::dimensions(), 3);
    }

    #[test]
    fn color3_arithmetic() {
        let a = Color3f::new(1.0, 2.0, 3.0);
        let b = Color3f::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert_eq!((sum.x, sum.y, sum.z), (5.0_f32, 7.0, 9.0));
        let diff = b - a;
        assert_eq!((diff.x, diff.y, diff.z), (3.0_f32, 3.0, 3.0));
        let scaled = a * 2.0_f32;
        assert_eq!((scaled.x, scaled.y, scaled.z), (2.0_f32, 4.0, 6.0));
        let neg = -a;
        assert_eq!((neg.x, neg.y, neg.z), (-1.0_f32, -2.0, -3.0));
        let mut m = a;
        m += b;
        assert_eq!((m.x, m.y, m.z), (5.0_f32, 7.0, 9.0));
    }

    #[test]
    fn color3c_u8_smoke() {
        let c = Color3c::new(10, 20, 30);
        assert_eq!((c.x, c.y, c.z), (10_u8, 20, 30));
        let d = Color3c::new(1, 2, 3);
        let sum = c + d;
        assert_eq!((sum.x, sum.y, sum.z), (11_u8, 22, 33));
        assert_eq!(Color3c::dimensions(), 3);
    }

    #[test]
    fn color3h_half_smoke() {
        let c = Color3h::new(Half::new(0.5), Half::new(1.0), Half::new(2.0));
        let d = Color3h::new(Half::new(0.5), Half::new(0.0), Half::new(0.0));
        let sum = c + d;
        assert!((sum.x.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn color4_construct_index() {
        let c = Color4f::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!((c.r, c.g, c.b, c.a), (1.0_f32, 2.0, 3.0, 4.0));
        assert_eq!(c[0], 1.0_f32);
        assert_eq!(c[3], 4.0_f32);
        let s = Color4f::from_scalar(7.0);
        assert_eq!((s.r, s.g, s.b, s.a), (7.0_f32, 7.0, 7.0, 7.0));
        assert_eq!(Color4f::dimensions(), 4);
    }

    #[test]
    fn color4_arithmetic() {
        let a = Color4f::new(1.0, 2.0, 3.0, 4.0);
        let b = Color4f::new(5.0, 6.0, 7.0, 8.0);
        let sum = a + b;
        assert_eq!((sum.r, sum.g, sum.b, sum.a), (6.0_f32, 8.0, 10.0, 12.0));
        let neg = -a;
        assert_eq!((neg.r, neg.g, neg.b, neg.a), (-1.0_f32, -2.0, -3.0, -4.0));
        let scaled = a * 3.0_f32;
        assert_eq!((scaled.r, scaled.g, scaled.b, scaled.a), (3.0_f32, 6.0, 9.0, 12.0));
        let rev = 2.0_f32 * a;
        assert_eq!((rev.r, rev.g, rev.b, rev.a), (2.0_f32, 4.0, 6.0, 8.0));
    }

    #[test]
    fn color4c_u8_smoke() {
        let c = Color4c::new(10, 20, 30, 255);
        assert_eq!((c.r, c.g, c.b, c.a), (10_u8, 20, 30, 255));
        assert_eq!(Color4c::dimensions(), 4);
    }

    #[test]
    fn color4h_half_smoke() {
        let c = Color4h::new(
            Half::new(1.0), Half::new(0.5), Half::new(0.25), Half::new(1.0),
        );
        assert!((c.r.to_f32() - 1.0).abs() < 0.01);
        assert!((c.g.to_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn color3_display() {
        let c = Color3f::new(1.0, 2.0, 3.0);
        let s = format!("{c}");
        assert!(s.starts_with('(') && s.ends_with(')'));
    }

    #[test]
    fn color4_display() {
        let c = Color4f::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{c}");
        assert!(s.starts_with('(') && s.ends_with(')'));
    }

    #[test]
    fn color3_eq() {
        let a = Color3f::new(1.0, 2.0, 3.0);
        let b = Color3f::new(1.0, 2.0, 3.0);
        let c = Color3f::new(1.0, 2.0, 4.0);
        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn color4_eq() {
        let a = Color4f::new(1.0, 2.0, 3.0, 4.0);
        let b = Color4f::new(1.0, 2.0, 3.0, 4.0);
        let c = Color4f::new(1.0, 2.0, 3.0, 5.0);
        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn color4_to_array() {
        let c = Color4f::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(c.to_array(), [1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn color3_negate() {
        let mut c = Color3f::new(1.0, -2.0, 3.0);
        c.negate();
        assert_eq!((c.x, c.y, c.z), (-1.0_f32, 2.0, -3.0));
    }

    #[test]
    fn color4_negate() {
        let mut c = Color4f::new(1.0, -2.0, 3.0, -4.0);
        c.negate();
        assert_eq!((c.r, c.g, c.b, c.a), (-1.0_f32, 2.0, -3.0, 4.0));
    }

    #[test]
    fn color4_set_get_value() {
        let mut c = Color4f::new(0.0, 0.0, 0.0, 0.0);
        c.set_value_rgba(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32);
        assert_eq!((c.r, c.g, c.b, c.a), (1.0_f32, 2.0, 3.0, 4.0));
        let (mut a, mut b, mut cc, mut d) = (0.0_f32, 0.0, 0.0, 0.0);
        c.get_value_rgba(&mut a, &mut b, &mut cc, &mut d);
        assert_eq!((a, b, cc, d), (1.0_f32, 2.0, 3.0, 4.0));
    }
}
