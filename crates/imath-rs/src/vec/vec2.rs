//! Port of Imath ImathVec.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathVec.h>
//!
//! The [`Vec2`] 2-element vector. See [`crate::vec`] for the element-type bounds
//! and the foreign-vector-interop skip note.

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, Sub, SubAssign,
};

use crate::math::{equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{VecCast, VecFloat, VecLimits};

/// 2-element vector (port of C++ `template <class T> class Vec2`).
///
/// Fields are public `x, y`, matching the C++ direct-access members. `#[repr(C)]`
/// guarantees the in-memory layout matches the C++ class (two contiguous `T`), as
/// required for later byte-level EXR I/O and for the `Index` element access.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec2<T> {
    /// The x coordinate (C++ `T x`).
    pub x: T,
    /// The y coordinate (C++ `T y`).
    pub y: T,
}

// ----------------------------------------------------------------------------
// Element access by index (port of `operator[]`, const + non-const)
// ----------------------------------------------------------------------------
//
// C++ returns `reinterpret_cast<T*>(this)[i]`. With `#[repr(C)]` the fields are
// laid out as a contiguous `[T; 2]`, so indexing maps `0 -> x`, `1 -> y`. We expose
// it through `Index`/`IndexMut`; out-of-range indices panic (Rust's checked-access
// analogue of the C++ undefined behaviour on OOB).

impl<T> Index<usize> for Vec2<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of range: {i}"),
        }
    }
}

impl<T> IndexMut<usize> for Vec2<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of range: {i}"),
        }
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Vec2<T> {
    /// Port of the default constructor `Vec2()`.
    ///
    /// C++ leaves the elements *uninitialized*; Rust cannot expose uninitialized
    /// memory safely, so the safe analogue zero-initializes (`(0, 0)`). This is the
    /// only behavioural divergence and it is strictly safer.
    #[inline]
    fn default() -> Self {
        Vec2 {
            x: T::ZERO,
            y: T::ZERO,
        }
    }
}

impl<T: Copy> Vec2<T> {
    /// Port of `Vec2(T a, T b)` — initialize to the given elements `(a, b)`.
    #[inline]
    pub const fn new(a: T, b: T) -> Self {
        Vec2 { x: a, y: b }
    }

    /// Port of `explicit Vec2(T a)` — initialize to a scalar `(a, a)`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Vec2 { x: a, y: a }
    }
}

impl<T: Copy> Vec2<T> {
    /// Port of the cross-base-type constructor `template <class S> Vec2(const
    /// Vec2<S>& v)` — convert each element with the C++ `T(v.x)` cast.
    #[inline]
    pub fn from_vec2<S: Copy>(v: Vec2<S>) -> Self
    where
        T: VecCast<S>,
    {
        Vec2 {
            x: T::from_elem(v.x),
            y: T::from_elem(v.y),
        }
    }
}

// ----------------------------------------------------------------------------
// Compatibility with Sb (setValue / getValue)
// ----------------------------------------------------------------------------

impl<T: Copy> Vec2<T> {
    /// Port of `template <class S> void setValue(S a, S b)`.
    #[inline]
    pub fn set_value_xy<S: Copy>(&mut self, a: S, b: S)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(a);
        self.y = T::from_elem(b);
    }

    /// Port of `template <class S> void setValue(const Vec2<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: Vec2<S>)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(v.x);
        self.y = T::from_elem(v.y);
    }

    /// Port of `template <class S> void getValue(S& a, S& b) const`.
    #[inline]
    pub fn get_value_xy<S>(&self, a: &mut S, b: &mut S)
    where
        S: VecCast<T>,
    {
        *a = S::from_elem(self.x);
        *b = S::from_elem(self.y);
    }

    /// Port of `template <class S> void getValue(Vec2<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Vec2<S>)
    where
        S: Copy + VecCast<T>,
    {
        v.x = S::from_elem(self.x);
        v.y = S::from_elem(self.y);
    }

    /// Port of `const T* getValue() const` — copies fields into a by-value array.
    ///
    /// Returns `[x, y]` as a safe owned array (T: Copy). Replaces the C++
    /// `reinterpret_cast<const T*>(this)` pattern; no `unsafe` required because
    /// we copy the fields directly rather than reinterpreting the struct pointer.
    #[inline]
    pub fn to_array(&self) -> [T; 2] {
        [self.x, self.y]
    }
}

// ----------------------------------------------------------------------------
// Equality (port of operator== / operator!=)
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Vec2<T> {
    /// Port of `operator==`: `x == v.x && y == v.y`.
    #[inline]
    fn eq(&self, v: &Vec2<T>) -> bool {
        self.x == v.x && self.y == v.y
    }
    // `operator!=` is the negation, provided automatically by `PartialEq`.
}

// ----------------------------------------------------------------------------
// Approximate comparison, dot/cross, arithmetic
// ----------------------------------------------------------------------------

impl<T: Scalar> Vec2<T> {
    /// Port of `equalWithAbsError(const Vec2<T>& v, T e)`.
    #[inline]
    pub fn equal_with_abs_error(&self, v: &Vec2<T>, e: T) -> bool {
        equal_with_abs_error(self.x, v.x, e) && equal_with_abs_error(self.y, v.y, e)
    }

    /// Port of `equalWithRelError(const Vec2<T>& v, T e)`.
    #[inline]
    pub fn equal_with_rel_error(&self, v: &Vec2<T>, e: T) -> bool {
        equal_with_rel_error(self.x, v.x, e) && equal_with_rel_error(self.y, v.y, e)
    }

    /// Port of `dot(const Vec2& v)`: `x * v.x + y * v.y`.
    #[inline]
    pub fn dot(&self, v: &Vec2<T>) -> T {
        self.x * v.x + self.y * v.y
    }

    /// Port of `cross(const Vec2& v)`: the z component of the 3D cross product,
    /// `x * v.y - y * v.x`.
    #[inline]
    pub fn cross(&self, v: &Vec2<T>) -> T {
        self.x * v.y - self.y * v.x
    }

    /// Port of `negate()` — component-wise multiplication by -1, in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self
    }
}

// `dot` via `operator^` (port of `T operator^`).
impl<T: Scalar> core::ops::BitXor for Vec2<T> {
    type Output = T;
    /// Port of `operator^`: alias for [`Vec2::dot`].
    #[inline]
    fn bitxor(self, v: Vec2<T>) -> T {
        self.dot(&v)
    }
}

// `cross` via `operator%` (port of `T operator%` — Vec2 cross returns a scalar).
impl<T: Scalar> Rem for Vec2<T> {
    type Output = T;
    /// Port of `operator%`: the scalar 2D cross product, alias for [`Vec2::cross`].
    #[inline]
    fn rem(self, v: Vec2<T>) -> T {
        self.cross(&v)
    }
}

// --- Vector +/-/*/÷ vector, and compound assignment ---

impl<T: Scalar> Add for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator+`: component-wise addition.
    #[inline]
    fn add(self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.x + v.x, self.y + v.y)
    }
}

impl<T: Scalar> AddAssign for Vec2<T> {
    /// Port of `operator+=`.
    #[inline]
    fn add_assign(&mut self, v: Vec2<T>) {
        self.x = self.x + v.x;
        self.y = self.y + v.y;
    }
}

impl<T: Scalar> Sub for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator-`: component-wise subtraction.
    #[inline]
    fn sub(self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.x - v.x, self.y - v.y)
    }
}

impl<T: Scalar> SubAssign for Vec2<T> {
    /// Port of `operator-=`.
    #[inline]
    fn sub_assign(&mut self, v: Vec2<T>) {
        self.x = self.x - v.x;
        self.y = self.y - v.y;
    }
}

impl<T: Scalar> Neg for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of unary `operator-`: component-wise multiplication by -1.
    #[inline]
    fn neg(self) -> Vec2<T> {
        Vec2::new(-self.x, -self.y)
    }
}

impl<T: Scalar> Mul for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator*(const Vec2&)`: component-wise multiplication.
    #[inline]
    fn mul(self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.x * v.x, self.y * v.y)
    }
}

impl<T: Scalar> Mul<T> for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator*(T a)`: component-wise scalar multiplication.
    #[inline]
    fn mul(self, a: T) -> Vec2<T> {
        Vec2::new(self.x * a, self.y * a)
    }
}

impl<T: Scalar> MulAssign for Vec2<T> {
    /// Port of `operator*=(const Vec2&)`.
    #[inline]
    fn mul_assign(&mut self, v: Vec2<T>) {
        self.x = self.x * v.x;
        self.y = self.y * v.y;
    }
}

impl<T: Scalar> MulAssign<T> for Vec2<T> {
    /// Port of `operator*=(T a)`.
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.x = self.x * a;
        self.y = self.y * a;
    }
}

impl<T: Scalar> Div for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator/(const Vec2&)`: component-wise division.
    #[inline]
    fn div(self, v: Vec2<T>) -> Vec2<T> {
        Vec2::new(self.x / v.x, self.y / v.y)
    }
}

impl<T: Scalar> Div<T> for Vec2<T> {
    type Output = Vec2<T>;
    /// Port of `operator/(T a)`: component-wise scalar division.
    #[inline]
    fn div(self, a: T) -> Vec2<T> {
        Vec2::new(self.x / a, self.y / a)
    }
}

impl<T: Scalar> DivAssign for Vec2<T> {
    /// Port of `operator/=(const Vec2&)`.
    #[inline]
    fn div_assign(&mut self, v: Vec2<T>) {
        self.x = self.x / v.x;
        self.y = self.y / v.y;
    }
}

impl<T: Scalar> DivAssign<T> for Vec2<T> {
    /// Port of `operator/=(T a)`.
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.x = self.x / a;
        self.y = self.y / a;
    }
}

// ----------------------------------------------------------------------------
// Numeric limits, dimensions, BaseType (port of the static accessors)
// ----------------------------------------------------------------------------

impl<T: VecLimits> Vec2<T> {
    /// Port of `baseTypeLowest()` — `std::numeric_limits<T>::lowest()`.
    #[inline]
    pub fn base_type_lowest() -> T {
        T::LOWEST
    }
    /// Port of `baseTypeMax()` — `std::numeric_limits<T>::max()`.
    #[inline]
    pub fn base_type_max() -> T {
        T::MAX
    }
    /// Port of `baseTypeSmallest()` — `std::numeric_limits<T>::min()`.
    #[inline]
    pub fn base_type_smallest() -> T {
        T::SMALLEST
    }
    /// Port of `baseTypeEpsilon()` — `std::numeric_limits<T>::epsilon()`.
    #[inline]
    pub fn base_type_epsilon() -> T {
        T::EPSILON
    }
}

impl<T> Vec2<T> {
    /// Port of `dimensions()` — the number of dimensions, i.e. 2.
    #[inline]
    pub const fn dimensions() -> usize {
        2
    }
}

// ----------------------------------------------------------------------------
// Query and Manipulation: length / normalize family (floating-point only)
// ----------------------------------------------------------------------------
//
// These are bounded on `VecFloat`, so they exist ONLY for `Vec2<f32>`,
// `Vec2<f64>`, `Vec2<Half>` — the Rust analogue of the C++ `= delete` of these
// methods for `Vec2<short>`, `Vec2<int>`, `Vec2<int64_t>`.

impl<T: VecFloat> Vec2<T> {
    /// Port of `length2()` — the squared Euclidean norm (the self dot product).
    #[inline]
    pub fn length2(&self) -> T {
        self.dot(self)
    }

    /// Port of the private `lengthTiny()` helper: robust norm for tiny vectors,
    /// scaling by the max component to avoid underflow before the `sqrt`.
    #[inline]
    fn length_tiny(&self) -> T {
        let mut abs_x = self.x.vabs();
        let mut abs_y = self.y.vabs();

        let mut max = abs_x;
        if max < abs_y {
            max = abs_y;
        }

        if max == T::ZERO {
            return T::ZERO;
        }

        // Do not replace the divisions by max with multiplications by 1/max:
        // computing 1/max can overflow, the divisions below cannot.
        abs_x = abs_x / max;
        abs_y = abs_y / max;

        max * (abs_x * abs_x + abs_y * abs_y).sqrt()
    }

    /// Port of `length()` — the Euclidean norm, with the tiny-magnitude guard.
    #[inline]
    pub fn length(&self) -> T {
        let length2 = self.dot(self);

        if length2 < T::TWO * T::MIN_POSITIVE {
            return self.length_tiny();
        }

        length2.sqrt()
    }

    /// Port of `normalize()` — normalize in place; a null vector is left unchanged.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();

        if l != T::ZERO {
            self.x = self.x / l;
            self.y = self.y / l;
        }

        self
    }

    /// Port of `normalizeExc()` — normalize in place; error if `length() == 0`.
    ///
    /// C++ throws `std::domain_error`; the Rust analogue returns `Err`.
    #[inline]
    pub fn normalize_exc(&mut self) -> Result<&mut Self, &'static str> {
        let l = self.length();

        if l == T::ZERO {
            return Err("Cannot normalize null vector.");
        }

        self.x = self.x / l;
        self.y = self.y / l;
        Ok(self)
    }

    /// Port of `normalizeNonNull()` — normalize in place without the zero check.
    #[inline]
    pub fn normalize_non_null(&mut self) -> &mut Self {
        let l = self.length();
        self.x = self.x / l;
        self.y = self.y / l;
        self
    }

    /// Port of `normalized()` — return a normalized copy; null vector -> `(0, 0)`.
    #[inline]
    pub fn normalized(&self) -> Vec2<T> {
        let l = self.length();
        if l == T::ZERO {
            return Vec2::from_scalar(T::ZERO);
        }
        Vec2::new(self.x / l, self.y / l)
    }

    /// Port of `normalizedExc()` — normalized copy; error if `length() == 0`.
    #[inline]
    pub fn normalized_exc(&self) -> Result<Vec2<T>, &'static str> {
        let l = self.length();
        if l == T::ZERO {
            return Err("Cannot normalize null vector.");
        }
        Ok(Vec2::new(self.x / l, self.y / l))
    }

    /// Port of `normalizedNonNull()` — normalized copy without the zero check.
    #[inline]
    pub fn normalized_non_null(&self) -> Vec2<T> {
        let l = self.length();
        Vec2::new(self.x / l, self.y / l)
    }
}

// ----------------------------------------------------------------------------
// Reverse scalar multiplication (port of `operator*(T a, const Vec2<T>& v)`)
// ----------------------------------------------------------------------------
//
// C++ provides a free `operator*` so `scalar * vec` works. Rust cannot write a
// blanket `impl<T: Scalar> Mul<Vec2<T>> for T` (orphan rule on the foreign scalar
// types), so we implement it per concrete element type via a macro.

macro_rules! impl_scalar_mul_vec2 {
    ($t:ty) => {
        impl Mul<Vec2<$t>> for $t {
            type Output = Vec2<$t>;
            /// Port of `operator*(T a, const Vec2<T>& v)`.
            #[inline]
            fn mul(self, v: Vec2<$t>) -> Vec2<$t> {
                Vec2::new(self * v.x, self * v.y)
            }
        }
    };
}
impl_scalar_mul_vec2!(i16);
impl_scalar_mul_vec2!(i32);
impl_scalar_mul_vec2!(i64);
impl_scalar_mul_vec2!(f32);
impl_scalar_mul_vec2!(f64);
impl_scalar_mul_vec2!(crate::half::Half);

// ----------------------------------------------------------------------------
// Stream output (port of `operator<<`) -> Display, as "(x y)"
// ----------------------------------------------------------------------------

impl<T: core::fmt::Display> core::fmt::Display for Vec2<T> {
    /// Port of `operator<<(ostream&, const Vec2<T>&)`: formats as `(x y)`.
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({} {})", self.x, self.y)
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Vec2<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec2({:?}, {:?})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::Half;
    use crate::vec::V2h;

    #[test]
    fn construct_and_index() {
        let v = Vec2::new(3.0_f32, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(v[0], 3.0);
        assert_eq!(v[1], 4.0);
        let s = Vec2::from_scalar(2.0_f32);
        assert_eq!((s.x, s.y), (2.0, 2.0));
        let d: Vec2<f32> = Vec2::default();
        assert_eq!((d.x, d.y), (0.0, 0.0));
        let mut m = v;
        m[0] = 10.0;
        assert_eq!(m.x, 10.0);
        assert_eq!(Vec2::<f32>::dimensions(), 2);
    }

    #[test]
    fn cross_type_convert() {
        let vd = Vec2::new(1.5_f64, 2.5);
        let vf: Vec2<f32> = Vec2::from_vec2(vd);
        assert_eq!((vf.x, vf.y), (1.5_f32, 2.5));
        let vi: Vec2<i32> = Vec2::from_vec2(Vec2::new(3.9_f32, -2.1));
        assert_eq!((vi.x, vi.y), (3, -2)); // truncation like C++ static_cast
    }

    #[test]
    fn arithmetic_ops() {
        let a = Vec2::new(1.0_f32, 2.0);
        let b = Vec2::new(3.0_f32, 5.0);
        assert_eq!(a + b, Vec2::new(4.0, 7.0));
        assert_eq!(b - a, Vec2::new(2.0, 3.0));
        assert_eq!(a * b, Vec2::new(3.0, 10.0));
        assert_eq!(b / a, Vec2::new(3.0, 2.5));
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(2.0_f32 * a, Vec2::new(2.0, 4.0)); // reverse scalar mul
        assert_eq!(a / 2.0, Vec2::new(0.5, 1.0));
        assert_eq!(-a, Vec2::new(-1.0, -2.0));
        let mut c = a;
        c += b;
        assert_eq!(c, Vec2::new(4.0, 7.0));
        c -= b;
        assert_eq!(c, a);
        c *= 3.0;
        assert_eq!(c, Vec2::new(3.0, 6.0));
        c /= 3.0;
        assert_eq!(c, a);
        let mut n = a;
        n.negate();
        assert_eq!(n, Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn dot_cross_length() {
        let a = Vec2::new(3.0_f32, 4.0);
        let b = Vec2::new(1.0_f32, 2.0);
        assert_eq!(a.dot(&b), 11.0);
        assert_eq!(a ^ b, 11.0); // operator^
        // cross-as-scalar: x*v.y - y*v.x
        assert_eq!(a.cross(&b), 3.0 * 2.0 - 4.0 * 1.0);
        assert_eq!(a % b, 2.0); // operator%
        assert_eq!(a.length2(), 25.0);
        assert_eq!(a.length(), 5.0);
        let n = a.normalized();
        assert!((n.length() - 1.0).abs() < 1e-6);
        let mut m = a;
        m.normalize();
        assert!((m.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equal_with_error_and_limits() {
        let a = Vec2::new(1.0_f32, 2.0);
        let b = Vec2::new(1.05_f32, 2.05);
        assert!(a.equal_with_abs_error(&b, 0.1));
        assert!(!a.equal_with_abs_error(&b, 0.01));
        assert!(a.equal_with_rel_error(&b, 0.1));
        assert_eq!(Vec2::<f32>::base_type_max(), f32::MAX);
        assert_eq!(Vec2::<i32>::base_type_lowest(), i32::MIN);
    }

    #[test]
    fn get_set_value_and_slice() {
        let mut v = Vec2::new(1.0_f32, 2.0);
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        v.get_value_xy(&mut a, &mut b);
        assert_eq!((a, b), (1.0, 2.0));
        v.set_value_xy(7.0_f64, 8.0);
        assert_eq!(v, Vec2::new(7.0, 8.0));
        // to_array() is the safe replacement for the former unsafe as_slice().
        assert_eq!(v.to_array(), [7.0, 8.0]);
        // Mutation via IndexMut (safe) replaces the former unsafe as_mut_slice().
        v[1] = 9.0;
        assert_eq!(v.y, 9.0);
    }

    #[test]
    fn half_vector_smoke() {
        let a = V2h::new(Half::new(1.0), Half::new(2.0));
        let b = V2h::new(Half::new(3.0), Half::new(4.0));
        let c = a + b;
        assert_eq!(c.x.to_f32(), 4.0);
        assert_eq!(c.y.to_f32(), 6.0);
        // Convert V2h -> V2f and back.
        let f: Vec2<f32> = Vec2::from_vec2(c);
        assert_eq!(f, Vec2::new(4.0_f32, 6.0));
        let back: V2h = Vec2::from_vec2(f);
        assert_eq!(back.x.to_f32(), 4.0);
        // length via float promotion.
        let d = V2h::new(Half::new(3.0), Half::new(4.0));
        assert!((d.length().to_f32() - 5.0).abs() < 0.1);
    }
}
