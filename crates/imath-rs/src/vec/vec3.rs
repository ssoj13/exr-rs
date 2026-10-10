//! Port of Imath ImathVec.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathVec.h>
//!
//! The [`Vec3`] 3-element vector. See [`crate::vec`] for the element-type bounds
//! and the foreign-vector-interop skip note.

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};

use crate::math::{equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{InfException, Vec4, VecCast, VecFloat, VecLimits};

/// 3-element vector (port of C++ `template <class T> class Vec3`).
///
/// Public `x, y, z` fields mirror the C++ direct-access members; `#[repr(C)]`
/// guarantees the layout matches the C++ class (three contiguous `T`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec3<T> {
    /// The x coordinate (C++ `T x`).
    pub x: T,
    /// The y coordinate (C++ `T y`).
    pub y: T,
    /// The z coordinate (C++ `T z`).
    pub z: T,
}

// ----------------------------------------------------------------------------
// Element access by index (port of `operator[]`)
// ----------------------------------------------------------------------------

impl<T> Index<usize> for Vec3<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of range: {i}"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of range: {i}"),
        }
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Vec3<T> {
    /// Port of the default constructor `Vec3()`.
    ///
    /// C++ leaves the elements uninitialized; the safe Rust analogue zero-inits
    /// to `(0, 0, 0)`.
    #[inline]
    fn default() -> Self {
        Vec3 {
            x: T::ZERO,
            y: T::ZERO,
            z: T::ZERO,
        }
    }
}

impl<T: Copy> Vec3<T> {
    /// Port of `Vec3(T a, T b, T c)` — initialize to `(a, b, c)`.
    #[inline]
    pub const fn new(a: T, b: T, c: T) -> Self {
        Vec3 { x: a, y: b, z: c }
    }

    /// Port of `explicit Vec3(T a)` — initialize to a scalar `(a, a, a)`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Vec3 { x: a, y: a, z: a }
    }

    /// Port of `template <class S> Vec3(const Vec3<S>& v)` — cross-base-type
    /// element conversion via the C++ `T(v.x)` cast.
    #[inline]
    pub fn from_vec3<S: Copy>(v: Vec3<S>) -> Self
    where
        T: VecCast<S>,
    {
        Vec3 {
            x: T::from_elem(v.x),
            y: T::from_elem(v.y),
            z: T::from_elem(v.z),
        }
    }
}

impl<T: Scalar> Vec3<T> {
    /// Port of `template <class S> explicit Vec3(const Vec4<S>& v)` — the Vec4 to
    /// Vec3 perspective divide: `x, y, z` each divided by `w`, even if `w` is 0
    /// (the result then depends on the platform's floating-point exception model;
    /// here it follows Rust's IEEE division, yielding inf/NaN).
    #[inline]
    pub fn from_vec4<S: Scalar>(v: Vec4<S>) -> Self
    where
        T: VecCast<S>,
    {
        // C++: x (T (v.x / v.w)) — the division happens in the *source* type S, then
        // the quotient is cast to T. Reproduce that ordering exactly.
        Vec3 {
            x: T::from_elem(v.x / v.w),
            y: T::from_elem(v.y / v.w),
            z: T::from_elem(v.z / v.w),
        }
    }

    /// Port of `template <class S> explicit Vec3(const Vec4<S>& v, InfException)`
    /// — perspective divide that errors if `w` is zero or the divide would
    /// overflow. C++ throws `std::domain_error("Cannot normalize point at
    /// infinity.")`; the Rust analogue returns `Err`.
    ///
    /// The `_marker: InfException` argument preserves the C++ overload selector.
    #[inline]
    pub fn from_vec4_exc<S: Copy>(v: Vec4<S>, _marker: InfException) -> Result<Self, &'static str>
    where
        T: VecCast<S> + VecLimits,
    {
        let vx = T::from_elem(v.x);
        let vy = T::from_elem(v.y);
        let vz = T::from_elem(v.z);
        let vw = T::from_elem(v.w);

        let abs_w = if vw >= T::ZERO { vw } else { -vw };

        if abs_w < T::ONE {
            let m = T::MAX * abs_w;

            if vx <= -m || vx >= m || vy <= -m || vy >= m || vz <= -m || vz >= m {
                return Err("Cannot normalize point at infinity.");
            }
        }

        Ok(Vec3 {
            x: vx / vw,
            y: vy / vw,
            z: vz / vw,
        })
    }
}

// ----------------------------------------------------------------------------
// Compatibility with Sb (setValue / getValue)
// ----------------------------------------------------------------------------

impl<T: Copy> Vec3<T> {
    /// Port of `template <class S> void setValue(S a, S b, S c)`.
    #[inline]
    pub fn set_value_xyz<S: Copy>(&mut self, a: S, b: S, c: S)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(a);
        self.y = T::from_elem(b);
        self.z = T::from_elem(c);
    }

    /// Port of `template <class S> void setValue(const Vec3<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: Vec3<S>)
    where
        T: VecCast<S>,
    {
        self.x = T::from_elem(v.x);
        self.y = T::from_elem(v.y);
        self.z = T::from_elem(v.z);
    }

    /// Port of `template <class S> void getValue(S& a, S& b, S& c) const`.
    #[inline]
    pub fn get_value_xyz<S>(&self, a: &mut S, b: &mut S, c: &mut S)
    where
        S: VecCast<T>,
    {
        *a = S::from_elem(self.x);
        *b = S::from_elem(self.y);
        *c = S::from_elem(self.z);
    }

    /// Port of `template <class S> void getValue(Vec3<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Vec3<S>)
    where
        S: Copy + VecCast<T>,
    {
        v.x = S::from_elem(self.x);
        v.y = S::from_elem(self.y);
        v.z = S::from_elem(self.z);
    }

    /// Port of `const T* getValue() const` — copies fields into a by-value array.
    ///
    /// Returns `[x, y, z]` as a safe owned array (T: Copy). Replaces the C++
    /// `reinterpret_cast<const T*>(this)` pattern; no `unsafe` required.
    #[inline]
    pub fn to_array(&self) -> [T; 3] {
        [self.x, self.y, self.z]
    }
}

// ----------------------------------------------------------------------------
// Equality
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Vec3<T> {
    /// Port of `operator==`: `x == v.x && y == v.y && z == v.z`.
    #[inline]
    fn eq(&self, v: &Vec3<T>) -> bool {
        self.x == v.x && self.y == v.y && self.z == v.z
    }
}

// ----------------------------------------------------------------------------
// Approximate comparison, dot/cross, arithmetic
// ----------------------------------------------------------------------------

impl<T: Scalar> Vec3<T> {
    /// Port of `equalWithAbsError(const Vec3<T>& v, T e)`.
    #[inline]
    pub fn equal_with_abs_error(&self, v: &Vec3<T>, e: T) -> bool {
        equal_with_abs_error(self.x, v.x, e)
            && equal_with_abs_error(self.y, v.y, e)
            && equal_with_abs_error(self.z, v.z, e)
    }

    /// Port of `equalWithRelError(const Vec3<T>& v, T e)`.
    #[inline]
    pub fn equal_with_rel_error(&self, v: &Vec3<T>, e: T) -> bool {
        equal_with_rel_error(self.x, v.x, e)
            && equal_with_rel_error(self.y, v.y, e)
            && equal_with_rel_error(self.z, v.z, e)
    }

    /// Port of `dot(const Vec3& v)`: `x * v.x + y * v.y + z * v.z`.
    #[inline]
    pub fn dot(&self, v: &Vec3<T>) -> T {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Port of `cross(const Vec3& v)`: the right-handed 3D cross product.
    #[inline]
    pub fn cross(&self, v: &Vec3<T>) -> Vec3<T> {
        Vec3::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }

    /// Port of `negate()` — component-wise multiplication by -1, in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }
}

// `dot` via `operator^`.
impl<T: Scalar> core::ops::BitXor for Vec3<T> {
    type Output = T;
    /// Port of `operator^`: alias for [`Vec3::dot`].
    #[inline]
    fn bitxor(self, v: Vec3<T>) -> T {
        self.dot(&v)
    }
}

// `cross` via `operator%` and `operator%=` (Vec3 cross returns a Vec3).
impl<T: Scalar> Rem for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator%`: the right-handed cross product, alias for [`Vec3::cross`].
    #[inline]
    fn rem(self, v: Vec3<T>) -> Vec3<T> {
        self.cross(&v)
    }
}

impl<T: Scalar> RemAssign for Vec3<T> {
    /// Port of `operator%=`: in-place right-handed cross product.
    #[inline]
    fn rem_assign(&mut self, v: Vec3<T>) {
        let a = self.y * v.z - self.z * v.y;
        let b = self.z * v.x - self.x * v.z;
        let c = self.x * v.y - self.y * v.x;
        self.x = a;
        self.y = b;
        self.z = c;
    }
}

// --- Vector +/-/*/÷ vector and scalar, with compound assignment ---

impl<T: Scalar> Add for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator+`.
    #[inline]
    fn add(self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl<T: Scalar> AddAssign for Vec3<T> {
    /// Port of `operator+=`.
    #[inline]
    fn add_assign(&mut self, v: Vec3<T>) {
        self.x = self.x + v.x;
        self.y = self.y + v.y;
        self.z = self.z + v.z;
    }
}

impl<T: Scalar> Sub for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator-`.
    #[inline]
    fn sub(self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl<T: Scalar> SubAssign for Vec3<T> {
    /// Port of `operator-=`.
    #[inline]
    fn sub_assign(&mut self, v: Vec3<T>) {
        self.x = self.x - v.x;
        self.y = self.y - v.y;
        self.z = self.z - v.z;
    }
}

impl<T: Scalar> Neg for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of unary `operator-`.
    #[inline]
    fn neg(self) -> Vec3<T> {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl<T: Scalar> Mul for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator*(const Vec3&)`.
    #[inline]
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(self.x * v.x, self.y * v.y, self.z * v.z)
    }
}

impl<T: Scalar> Mul<T> for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator*(T a)`.
    #[inline]
    fn mul(self, a: T) -> Vec3<T> {
        Vec3::new(self.x * a, self.y * a, self.z * a)
    }
}

impl<T: Scalar> MulAssign for Vec3<T> {
    /// Port of `operator*=(const Vec3&)`.
    #[inline]
    fn mul_assign(&mut self, v: Vec3<T>) {
        self.x = self.x * v.x;
        self.y = self.y * v.y;
        self.z = self.z * v.z;
    }
}

impl<T: Scalar> MulAssign<T> for Vec3<T> {
    /// Port of `operator*=(T a)`.
    #[inline]
    fn mul_assign(&mut self, a: T) {
        self.x = self.x * a;
        self.y = self.y * a;
        self.z = self.z * a;
    }
}

impl<T: Scalar> Div for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator/(const Vec3&)`.
    #[inline]
    fn div(self, v: Vec3<T>) -> Vec3<T> {
        Vec3::new(self.x / v.x, self.y / v.y, self.z / v.z)
    }
}

impl<T: Scalar> Div<T> for Vec3<T> {
    type Output = Vec3<T>;
    /// Port of `operator/(T a)`.
    #[inline]
    fn div(self, a: T) -> Vec3<T> {
        Vec3::new(self.x / a, self.y / a, self.z / a)
    }
}

impl<T: Scalar> DivAssign for Vec3<T> {
    /// Port of `operator/=(const Vec3&)`.
    #[inline]
    fn div_assign(&mut self, v: Vec3<T>) {
        self.x = self.x / v.x;
        self.y = self.y / v.y;
        self.z = self.z / v.z;
    }
}

impl<T: Scalar> DivAssign<T> for Vec3<T> {
    /// Port of `operator/=(T a)`.
    #[inline]
    fn div_assign(&mut self, a: T) {
        self.x = self.x / a;
        self.y = self.y / a;
        self.z = self.z / a;
    }
}

// ----------------------------------------------------------------------------
// Numeric limits, dimensions
// ----------------------------------------------------------------------------

impl<T: VecLimits> Vec3<T> {
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

impl<T> Vec3<T> {
    /// Port of `dimensions()` — the number of dimensions, i.e. 3.
    #[inline]
    pub const fn dimensions() -> usize {
        3
    }
}

// ----------------------------------------------------------------------------
// Query and Manipulation: length / normalize family (floating-point only)
// ----------------------------------------------------------------------------

impl<T: VecFloat> Vec3<T> {
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

        let mut max = abs_x;
        if max < abs_y {
            max = abs_y;
        }
        if max < abs_z {
            max = abs_z;
        }

        if max == T::ZERO {
            return T::ZERO;
        }

        abs_x = abs_x / max;
        abs_y = abs_y / max;
        abs_z = abs_z / max;

        max * (abs_x * abs_x + abs_y * abs_y + abs_z * abs_z).sqrt()
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
        Ok(self)
    }

    /// Port of `normalizeNonNull()`.
    #[inline]
    pub fn normalize_non_null(&mut self) -> &mut Self {
        let l = self.length();
        self.x = self.x / l;
        self.y = self.y / l;
        self.z = self.z / l;
        self
    }

    /// Port of `normalized()` — copy; null vector -> `(0, 0, 0)`.
    #[inline]
    pub fn normalized(&self) -> Vec3<T> {
        let l = self.length();
        if l == T::ZERO {
            return Vec3::from_scalar(T::ZERO);
        }
        Vec3::new(self.x / l, self.y / l, self.z / l)
    }

    /// Port of `normalizedExc()`.
    #[inline]
    pub fn normalized_exc(&self) -> Result<Vec3<T>, &'static str> {
        let l = self.length();
        if l == T::ZERO {
            return Err("Cannot normalize null vector.");
        }
        Ok(Vec3::new(self.x / l, self.y / l, self.z / l))
    }

    /// Port of `normalizedNonNull()`.
    #[inline]
    pub fn normalized_non_null(&self) -> Vec3<T> {
        let l = self.length();
        Vec3::new(self.x / l, self.y / l, self.z / l)
    }
}

// ----------------------------------------------------------------------------
// Reverse scalar multiplication (port of `operator*(T a, const Vec3<T>& v)`)
// ----------------------------------------------------------------------------

macro_rules! impl_scalar_mul_vec3 {
    ($t:ty) => {
        impl Mul<Vec3<$t>> for $t {
            type Output = Vec3<$t>;
            /// Port of `operator*(T a, const Vec3<T>& v)`.
            #[inline]
            fn mul(self, v: Vec3<$t>) -> Vec3<$t> {
                Vec3::new(self * v.x, self * v.y, self * v.z)
            }
        }
    };
}
impl_scalar_mul_vec3!(i16);
impl_scalar_mul_vec3!(i32);
impl_scalar_mul_vec3!(i64);
impl_scalar_mul_vec3!(f32);
impl_scalar_mul_vec3!(f64);
impl_scalar_mul_vec3!(crate::half::Half);

// ----------------------------------------------------------------------------
// Stream output -> Display, as "(x y z)"
// ----------------------------------------------------------------------------

impl<T: core::fmt::Display> core::fmt::Display for Vec3<T> {
    /// Port of `operator<<(ostream&, const Vec3<T>&)`: formats as `(x y z)`.
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({} {} {})", self.x, self.y, self.z)
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Vec3<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec3({:?}, {:?}, {:?})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::half::Half;
    use crate::vec::V3h;

    #[test]
    fn construct_index_convert() {
        let v = Vec3::new(1.0_f32, 2.0, 3.0);
        assert_eq!((v[0], v[1], v[2]), (1.0, 2.0, 3.0));
        assert_eq!(Vec3::from_scalar(5.0_f32), Vec3::new(5.0, 5.0, 5.0));
        let d: Vec3<f32> = Vec3::default();
        assert_eq!(d, Vec3::new(0.0, 0.0, 0.0));
        let vi: Vec3<i32> = Vec3::from_vec3(Vec3::new(1.9_f64, 2.9, 3.9));
        assert_eq!(vi, Vec3::new(1, 2, 3));
        assert_eq!(Vec3::<f32>::dimensions(), 3);
    }

    #[test]
    fn arithmetic_and_cross() {
        let a = Vec3::new(1.0_f32, 0.0, 0.0);
        let b = Vec3::new(0.0_f32, 1.0, 0.0);
        // right-handed cross: x cross y = z
        assert_eq!(a.cross(&b), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(a % b, Vec3::new(0.0, 0.0, 1.0)); // operator%
        let mut c = a;
        c %= b;
        assert_eq!(c, Vec3::new(0.0, 0.0, 1.0)); // operator%=
        let p = Vec3::new(2.0_f32, 3.0, 4.0);
        let q = Vec3::new(5.0_f32, 6.0, 7.0);
        assert_eq!(p.dot(&q), 2.0 * 5.0 + 3.0 * 6.0 + 4.0 * 7.0);
        assert_eq!(p ^ q, p.dot(&q));
        assert_eq!(p + q, Vec3::new(7.0, 9.0, 11.0));
        assert_eq!(3.0_f32 * p, Vec3::new(6.0, 9.0, 12.0));
    }

    #[test]
    fn length_known_value() {
        // V3f(2,3,6).length() == 7
        let v = Vec3::new(2.0_f32, 3.0, 6.0);
        assert_eq!(v.length2(), 49.0);
        assert_eq!(v.length(), 7.0);
        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-6);
        // null vector normalized stays zero.
        let z: Vec3<f32> = Vec3::default();
        assert_eq!(z.normalized(), Vec3::new(0.0, 0.0, 0.0));
        assert!(z.normalized_exc().is_err());
    }

    #[test]
    fn vec4_to_vec3_perspective_divide() {
        let v4 = Vec4::new(2.0_f32, 4.0, 6.0, 2.0);
        let v3: Vec3<f32> = Vec3::from_vec4(v4);
        assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
        // exception variant: finite w -> Ok.
        let ok = Vec3::<f32>::from_vec4_exc(v4, InfException::InfException);
        assert_eq!(ok.unwrap(), Vec3::new(1.0, 2.0, 3.0));
        // tiny w with huge components -> Err (point at infinity).
        let inf = Vec4::new(1.0e30_f32, 0.0, 0.0, 1.0e-30);
        assert!(Vec3::<f32>::from_vec4_exc(inf, InfException::InfException).is_err());
    }

    #[test]
    fn equal_with_error() {
        let a = Vec3::new(1.0_f32, 2.0, 3.0);
        let b = Vec3::new(1.01_f32, 2.01, 2.99);
        assert!(a.equal_with_abs_error(&b, 0.05));
        assert!(!a.equal_with_abs_error(&b, 0.001));
        assert!(a.equal_with_rel_error(&b, 0.05));
    }

    #[test]
    fn half_vector_smoke() {
        let a = V3h::new(Half::new(1.0), Half::new(2.0), Half::new(2.0));
        let b = V3h::new(Half::new(1.0), Half::new(0.0), Half::new(0.0));
        let c = a + b;
        assert_eq!(c.x.to_f32(), 2.0);
        assert_eq!(c.y.to_f32(), 2.0);
        // V3h -> V3f and back.
        let f: Vec3<f32> = Vec3::from_vec3(a);
        assert_eq!(f, Vec3::new(1.0, 2.0, 2.0));
        let back: V3h = Vec3::from_vec3(f);
        assert_eq!(back.x.to_f32(), 1.0);
        // (1,2,2) length == 3 via float promotion.
        assert!((a.length().to_f32() - 3.0).abs() < 0.1);
        // cross product on halves.
        let x = V3h::new(Half::new(1.0), Half::new(0.0), Half::new(0.0));
        let y = V3h::new(Half::new(0.0), Half::new(1.0), Half::new(0.0));
        let cr = x.cross(&y);
        assert_eq!(cr.z.to_f32(), 1.0);
    }
}
