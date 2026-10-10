//! Port of Imath ImathVec.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathVec.h>
//!
//! The [`Vec4`] 4-element vector. See [`crate::vec`] for the element-type bounds
//! and the foreign-vector-interop skip note.

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::math::{equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{Vec3, VecCast, VecFloat, VecLimits};

/// 4-element vector (port of C++ `template <class T> class Vec4`).
///
/// Public `x, y, z, w` fields mirror the C++ direct-access members; `#[repr(C)]`
/// guarantees the layout matches the C++ class (four contiguous `T`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec4<T> {
    /// The x coordinate (C++ `T x`).
    pub x: T,
    /// The y coordinate (C++ `T y`).
    pub y: T,
    /// The z coordinate (C++ `T z`).
    pub z: T,
    /// The w coordinate (C++ `T w`).
    pub w: T,
}

// ----------------------------------------------------------------------------
// Element access by index (port of `operator[]`)
// ----------------------------------------------------------------------------

impl<T> Index<usize> for Vec4<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of range: {i}"),
        }
    }
}

impl<T> IndexMut<usize> for Vec4<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of range: {i}"),
        }
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Vec4<T> {
    /// Port of the default constructor `Vec4()`.
    ///
    /// C++ leaves the elements uninitialized; the safe Rust analogue zero-inits
    /// to `(0, 0, 0, 0)`.
    #[inline]
    fn default() -> Self {
        Vec4 {
            x: T::ZERO,
            y: T::ZERO,
            z: T::ZERO,
            w: T::ZERO,
        }
    }
}

impl<T: Copy> Vec4<T> {
    /// Port of `Vec4(T a, T b, T c, T d)` — initialize to `(a, b, c, d)`.
    #[inline]
    pub const fn new(a: T, b: T, c: T, d: T) -> Self {
        Vec4 {
            x: a,
            y: b,
            z: c,
            w: d,
        }
    }

    /// Port of `explicit Vec4(T a)` — initialize to a scalar `(a, a, a, a)`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Vec4 {
            x: a,
            y: a,
            z: a,
            w: a,
        }
    }

    /// Port of `template <class S> Vec4(const Vec4<S>& v)` — cross-base-type
    /// element conversion via the C++ `T(v.x)` cast.
    #[inline]
    pub fn from_vec4<S: Copy>(v: Vec4<S>) -> Self
    where
        T: VecCast<S>,
    {
        Vec4 {
            x: T::from_elem(v.x),
            y: T::from_elem(v.y),
            z: T::from_elem(v.z),
            w: T::from_elem(v.w),
        }
    }

    /// Port of `template <class S> explicit Vec4(const Vec3<S>& v)` — Vec3 to Vec4
    /// conversion, setting `w` to 1.
    #[inline]
    pub fn from_vec3<S: Copy>(v: Vec3<S>) -> Self
    where
        T: VecCast<S> + Scalar,
    {
        Vec4 {
            x: T::from_elem(v.x),
            y: T::from_elem(v.y),
            z: T::from_elem(v.z),
            w: T::ONE,
        }
    }
}

// ----------------------------------------------------------------------------
// Compatibility with Sb (setValue / getValue)
// ----------------------------------------------------------------------------

impl<T: Copy> Vec4<T> {
    /// Port of `template <class S> void setValue(S a, S b, S c, S d)`.
    #[inline]
    pub fn set_value_xyzw<S: Copy>(&mut self, a: S, b: S, c: S, d: S)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(a);
        self.y = T::from_elem(b);
        self.z = T::from_elem(c);
        self.w = T::from_elem(d);
    }

    /// Port of `template <class S> void setValue(const Vec4<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: Vec4<S>)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(v.x);
        self.y = T::from_elem(v.y);
        self.z = T::from_elem(v.z);
        self.w = T::from_elem(v.w);
    }

    /// Port of `template <class S> void getValue(S& a, S& b, S& c, S& d) const`.
    #[inline]
    pub fn get_value_xyzw<S>(&self, a: &mut S, b: &mut S, c: &mut S, d: &mut S)
    where
        S: VecCast<T>,
    {
        *a = S::from_elem(self.x);
        *b = S::from_elem(self.y);
        *c = S::from_elem(self.z);
        *d = S::from_elem(self.w);
    }

    /// Port of `template <class S> void getValue(Vec4<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Vec4<S>)
    where
        S: Copy + VecCast<T>,
    {
        v.x = S::from_elem(self.x);
        v.y = S::from_elem(self.y);
        v.z = S::from_elem(self.z);
        v.w = S::from_elem(self.w);
    }

    /// Port of `const T* getValue() const` — copies fields into a by-value array.
    ///
    /// Returns `[x, y, z, w]` as a safe owned array (T: Copy). Replaces the C++
    /// `reinterpret_cast<const T*>(this)` pattern; no `unsafe` required.
    #[inline]
    pub fn to_array(&self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

// ----------------------------------------------------------------------------
// Equality
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Vec4<T> {
    /// Port of `operator==`: `x == v.x && y == v.y && z == v.z && w == v.w`.
    #[inline]
    fn eq(&self, v: &Vec4<T>) -> bool {
        self.x == v.x && self.y == v.y && self.z == v.z && self.w == v.w
    }
}

// ----------------------------------------------------------------------------
// Approximate comparison, dot, arithmetic (Vec4 has NO cross product)
// ----------------------------------------------------------------------------

impl<T: Scalar> Vec4<T> {
    /// Port of `equalWithAbsError(const Vec4<T>& v, T e)`.
    #[inline]
    pub fn equal_with_abs_error(&self, v: &Vec4<T>, e: T) -> bool {
        equal_with_abs_error(self.x, v.x, e)
            && equal_with_abs_error(self.y, v.y, e)
            && equal_with_abs_error(self.z, v.z, e)
            && equal_with_abs_error(self.w, v.w, e)
    }

    /// Port of `equalWithRelError(const Vec4<T>& v, T e)`.
    #[inline]
    pub fn equal_with_rel_error(&self, v: &Vec4<T>, e: T) -> bool {
        equal_with_rel_error(self.x, v.x, e)
            && equal_with_rel_error(self.y, v.y, e)
            && equal_with_rel_error(self.z, v.z, e)
            && equal_with_rel_error(self.w, v.w, e)
    }

    /// Port of `dot(const Vec4& v)`: `x*v.x + y*v.y + z*v.z + w*v.w`.
    #[inline]
    pub fn dot(&self, v: &Vec4<T>) -> T {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
    }

    /// Port of `negate()` — component-wise multiplication by -1, in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self.w = -self.w;
        self
    }
}

// `dot` via `operator^`.
impl<T: Scalar> core::ops::BitXor for Vec4<T> {
    type Output = T;
    /// Port of `operator^`: alias for [`Vec4::dot`].
    #[inline]
    fn bitxor(self, v: Vec4<T>) -> T {
        self.dot(&v)
    }
}

// --- Vector +/-/*/÷ vector and scalar, with compound assignment ---

impl<T: Scalar> Add for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator+`.
    #[inline]
    fn add(self, v: Vec4<T>) -> Vec4<T> {
        Vec4::new(self.x + v.x, self.y + v.y, self.z + v.z, self.w + v.w)
    }
}

impl<T: Scalar> AddAssign for Vec4<T> {
    /// Port of `operator+=`.
    #[inline]
    fn add_assign(&mut self, v: Vec4<T>) {
        self.x = self.x + v.x;
        self.y = self.y + v.y;
        self.z = self.z + v.z;
        self.w = self.w + v.w;
    }
}

impl<T: Scalar> Sub for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator-`.
    #[inline]
    fn sub(self, v: Vec4<T>) -> Vec4<T> {
        Vec4::new(self.x - v.x, self.y - v.y, self.z - v.z, self.w - v.w)
    }
}

impl<T: Scalar> SubAssign for Vec4<T> {
    /// Port of `operator-=`.
    #[inline]
    fn sub_assign(&mut self, v: Vec4<T>) {
        self.x = self.x - v.x;
        self.y = self.y - v.y;
        self.z = self.z - v.z;
        self.w = self.w - v.w;
    }
}

impl<T: Scalar> Neg for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of unary `operator-`.
    #[inline]
    fn neg(self) -> Vec4<T> {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl<T: Scalar> Mul for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator*(const Vec4&)`.
    #[inline]
    fn mul(self, v: Vec4<T>) -> Vec4<T> {
        Vec4::new(self.x * v.x, self.y * v.y, self.z * v.z, self.w * v.w)
    }
}

impl<T: Scalar> Mul<T> for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator*(T a)`.
    #[inline]
    fn mul(self, a: T) -> Vec4<T> {
        Vec4::new(self.x * a, self.y * a, self.z * a, self.w * a)
    }
}

impl<T: Scalar> MulAssign for Vec4<T> {
    /// Port of `operator*=(const Vec4&)`.
    #[inline]
    fn mul_assign(&mut self, v: Vec4<T>) {
        self.x = self.x * v.x;
        self.y = self.y * v.y;
        self.z = self.z * v.z;
        self.w = self.w * v.w;
    }
}

impl<T: Scalar> MulAssign<T> for Vec4<T> {
    /// Port of `operator*=(T a)`.
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.x = self.x * a;
        self.y = self.y * a;
        self.z = self.z * a;
        self.w = self.w * a;
    }
}

impl<T: Scalar> Div for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator/(const Vec4&)`.
    #[inline]
    fn div(self, v: Vec4<T>) -> Vec4<T> {
        Vec4::new(self.x / v.x, self.y / v.y, self.z / v.z, self.w / v.w)
    }
}

impl<T: Scalar> Div<T> for Vec4<T> {
    type Output = Vec4<T>;
    /// Port of `operator/(T a)`.
    #[inline]
    fn div(self, a: T) -> Vec4<T> {
        Vec4::new(self.x / a, self.y / a, self.z / a, self.w / a)
    }
}

impl<T: Scalar> DivAssign for Vec4<T> {
    /// Port of `operator/=(const Vec4&)`.
    #[inline]
    fn div_assign(&mut self, v: Vec4<T>) {
        self.x = self.x / v.x;
        self.y = self.y / v.y;
        self.z = self.z / v.z;
        self.w = self.w / v.w;
    }
}

impl<T: Scalar> DivAssign<T> for Vec4<T> {
    /// Port of `operator/=(T a)`.
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.x = self.x / a;
        self.y = self.y / a;
        self.z = self.z / a;
        self.w = self.w / a;
    }
}

// ----------------------------------------------------------------------------
// Numeric limits, dimensions
// ----------------------------------------------------------------------------

impl<T: VecLimits> Vec4<T> {
    /// Port of `baseTypeLowest()`.
    #[inline]
    pub fn base_type_lowest() -> T {
        T::LOWEST
    }
    /// Port of `baseTypeMax()`.
    #[inline]
    pub fn base_type_max() -> T {
        T::MAX
    }
    /// Port of `baseTypeSmallest()`.
    #[inline]
    pub fn base_type_smallest() -> T {
        T::SMALLEST
    }
    /// Port of `baseTypeEpsilon()`.
    #[inline]
    pub fn base_type_epsilon() -> T {
        T::EPSILON
    }
}

impl<T> Vec4<T> {
    /// Port of `dimensions()` — the number of dimensions, i.e. 4.
    #[inline]
    pub const fn dimensions() -> usize {
        4
    }
}

// ----------------------------------------------------------------------------
// Query and Manipulation: length / normalize family (floating-point only)
// ----------------------------------------------------------------------------

impl<T: VecFloat> Vec4<T> {
    /// Port of `length2()` — squared Euclidean norm.
    #[inline]
    pub fn length2(&self) -> T {
        self.dot(self)
    }

    /// Port of the private `lengthTiny()` helper.
    #[inline]
    fn length_tiny(&self) -> T {
        let mut abs_x = self.x.vabs();
        let mut abs_y = self.y.vabs();
        let mut abs_z = self.z.vabs();
        let mut abs_w = self.w.vabs();

        let mut max = abs_x;
        if max < abs_y {
            max = abs_y;
        }
        if max < abs_z {
            max = abs_z;
        }
        if max < abs_w {
            max = abs_w;
        }

        if max == T::ZERO {
            return T::ZERO;
        }

        abs_x = abs_x / max;
        abs_y = abs_y / max;
        abs_z = abs_z / max;
        abs_w = abs_w / max;

        max * (abs_x * abs_x + abs_y * abs_y + abs_z * abs_z + abs_w * abs_w).sqrt()
    }

    /// Port of `length()`.
    #[inline]
    pub fn length(&self) -> T {
        let length2 = self.dot(self);

        if length2 < T::TWO * T::MIN_POSITIVE {
            return self.length_tiny();
        }

        length2.sqrt()
    }

    /// Port of `normalize()` — in place; null vector left unchanged.
    #[inline]
    pub fn normalize(&mut self) -> &mut Self {
        let l = self.length();
        if l != T::ZERO {
            self.x = self.x / l;
            self.y = self.y / l;
            self.z = self.z / l;
            self.w = self.w / l;
        }
        self
    }

    /// Port of `normalizeExc()` — error if `length() == 0`.
    #[inline]
    pub fn normalize_exc(&mut self) -> Result<&mut Self, &'static str> {
        let l = self.length();
        if l == T::ZERO {
            return Err("Cannot normalize null vector.");
        }
        self.x = self.x / l;
        self.y = self.y / l;
        self.z = self.z / l;
        self.w = self.w / l;
        Ok(self)
    }

    /// Port of `normalizeNonNull()`.
    #[inline]
    pub fn normalize_non_null(&mut self) -> &mut Self {
        let l = self.length();
        self.x = self.x / l;
        self.y = self.y / l;
        self.z = self.z / l;
        self.w = self.w / l;
        self
    }

    /// Port of `normalized()` — copy; null vector -> `(0, 0, 0, 0)`.
    #[inline]
    pub fn normalized(&self) -> Vec4<T> {
        let l = self.length();
        if l == T::ZERO {
            return Vec4::from_scalar(T::ZERO);
        }
        Vec4::new(self.x / l, self.y / l, self.z / l, self.w / l)
    }

    /// Port of `normalizedExc()`.
    #[inline]
    pub fn normalized_exc(&self) -> Result<Vec4<T>, &'static str> {
        let l = self.length();
        if l == T::ZERO {
            return Err("Cannot normalize null vector.");
        }
        Ok(Vec4::new(self.x / l, self.y / l, self.z / l, self.w / l))
    }

    /// Port of `normalizedNonNull()`.
    #[inline]
    pub fn normalized_non_null(&self) -> Vec4<T> {
        let l = self.length();
        Vec4::new(self.x / l, self.y / l, self.z / l, self.w / l)
    }
}

// ----------------------------------------------------------------------------
// Reverse scalar multiplication (port of `operator*(T a, const Vec4<T>& v)`)
// ----------------------------------------------------------------------------

macro_rules! impl_scalar_mul_vec4 {
    ($t:ty) => {
        impl Mul<Vec4<$t>> for $t {
            type Output = Vec4<$t>;
            /// Port of `operator*(T a, const Vec4<T>& v)`.
            #[inline]
            fn mul(self, v: Vec4<$t>) -> Vec4<$t> {
                Vec4::new(self * v.x, self * v.y, self * v.z, self * v.w)
            }
        }
    };
}
impl_scalar_mul_vec4!(i16);
impl_scalar_mul_vec4!(i32);
impl_scalar_mul_vec4!(i64);
impl_scalar_mul_vec4!(f32);
impl_scalar_mul_vec4!(f64);
impl_scalar_mul_vec4!(crate::half::Half);

// ----------------------------------------------------------------------------
// Stream output -> Display, as "(x y z w)"
// ----------------------------------------------------------------------------

impl<T: core::fmt::Display> core::fmt::Display for Vec4<T> {
    /// Port of `operator<<(ostream&, const Vec4<T>&)`: formats as `(x y z w)`.
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({} {} {} {})", self.x, self.y, self.z, self.w)
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Vec4<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec4({:?}, {:?}, {:?}, {:?})", self.x, self.y, self.z, self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::Half;
    use crate::vec::V4h;

    #[test]
    fn construct_index_convert() {
        let v = Vec4::new(1.0_f32, 2.0, 3.0, 4.0);
        assert_eq!((v[0], v[1], v[2], v[3]), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(Vec4::from_scalar(2.0_f32), Vec4::new(2.0, 2.0, 2.0, 2.0));
        let d: Vec4<f32> = Vec4::default();
        assert_eq!(d, Vec4::new(0.0, 0.0, 0.0, 0.0));
        let vi: Vec4<i32> = Vec4::from_vec4(Vec4::new(1.9_f64, 2.9, 3.9, 4.9));
        assert_eq!(vi, Vec4::new(1, 2, 3, 4));
        assert_eq!(Vec4::<f32>::dimensions(), 4);
    }

    #[test]
    fn vec3_to_vec4_sets_w_one() {
        let v3 = Vec3::new(5.0_f32, 6.0, 7.0);
        let v4: Vec4<f32> = Vec4::from_vec3(v3);
        assert_eq!(v4, Vec4::new(5.0, 6.0, 7.0, 1.0));
    }

    #[test]
    fn arithmetic_dot_length() {
        let a = Vec4::new(1.0_f32, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0_f32, 6.0, 7.0, 8.0);
        assert_eq!(a.dot(&b), 5.0 + 12.0 + 21.0 + 32.0);
        assert_eq!(a ^ b, a.dot(&b));
        assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
        assert_eq!(b - a, Vec4::new(4.0, 4.0, 4.0, 4.0));
        assert_eq!(a * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(2.0_f32 * a, Vec4::new(2.0, 4.0, 6.0, 8.0));
        // (1,2,2,4)? choose vector with integer length: (2,2,2,... ) -> use known.
        let c = Vec4::new(1.0_f32, 2.0, 2.0, 4.0);
        assert_eq!(c.length2(), 1.0 + 4.0 + 4.0 + 16.0);
        assert_eq!(c.length(), 5.0); // sqrt(25)
        assert!((c.normalized().length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equal_with_error() {
        let a = Vec4::new(1.0_f32, 2.0, 3.0, 4.0);
        let b = Vec4::new(1.01_f32, 2.01, 2.99, 4.0);
        assert!(a.equal_with_abs_error(&b, 0.05));
        assert!(!a.equal_with_abs_error(&b, 0.001));
        assert!(a.equal_with_rel_error(&b, 0.05));
    }

    #[test]
    fn half_vector_smoke() {
        let a = V4h::new(
            Half::new(1.0),
            Half::new(2.0),
            Half::new(3.0),
            Half::new(4.0),
        );
        let b = V4h::new(
            Half::new(1.0),
            Half::new(1.0),
            Half::new(1.0),
            Half::new(1.0),
        );
        let c = a + b;
        assert_eq!(c.x.to_f32(), 2.0);
        assert_eq!(c.w.to_f32(), 5.0);
        // V4h -> V4f and back.
        let f: Vec4<f32> = Vec4::from_vec4(a);
        assert_eq!(f, Vec4::new(1.0, 2.0, 3.0, 4.0));
        let back: V4h = Vec4::from_vec4(f);
        assert_eq!(back.z.to_f32(), 3.0);
    }
}
