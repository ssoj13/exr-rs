//! Port of Imath ImathMatrix.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrix.h>
//!
//! The [`Matrix22`] 2x2 transformation matrix. See [`crate::matrix`] for the shared
//! design notes (row-major `x: [[T; 2]; 2]` storage, identity-default, the
//! exception-vs-`Result` convention for singular inverse, and the foreign-interop
//! skip).

use core::fmt;
use core::ops::{Index, IndexMut};

use crate::math::{Math, equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{Vec2, VecCast, VecLimits};

/// 2x2 transformation matrix (port of C++ `template <class T> class Matrix22`).
///
/// Elements are stored row-major in `x: [[T; 2]; 2]`, byte-for-byte matching the
/// C++ `T x[2][2]`. `#[repr(C)]` guarantees that layout for later EXR I/O. Row
/// access (`m[i]`) is the safe analogue of the C++ `operator[]` that returns a row
/// pointer; `m[i][j]` then indexes the element.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix22<T> {
    /// Matrix elements, row-major (`x[row][col]`), mirroring C++ `T x[2][2]`.
    pub x: [[T; 2]; 2],
}

// ----------------------------------------------------------------------------
// Row access — port of `operator[]` (returns the row, so `m[i][j]` works)
// ----------------------------------------------------------------------------

impl<T> Index<usize> for Matrix22<T> {
    type Output = [T; 2];
    /// Port of `T* operator[](int i)` — returns row `i` (panics if out of range).
    #[inline]
    fn index(&self, i: usize) -> &[T; 2] {
        &self.x[i]
    }
}

impl<T> IndexMut<usize> for Matrix22<T> {
    /// Port of `const T* operator[](int i) const` (mutable form) — returns row `i`.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut [T; 2] {
        &mut self.x[i]
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Matrix22<T> {
    /// Port of the default constructor `Matrix22()` — the identity matrix.
    #[inline]
    fn default() -> Self {
        Matrix22 {
            x: [[T::ONE, T::ZERO], [T::ZERO, T::ONE]],
        }
    }
}

impl<T: Copy> Matrix22<T> {
    /// Port of `Matrix22(T a)` — every element set to the scalar `a`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Matrix22 { x: [[a, a], [a, a]] }
    }

    /// Port of `Matrix22(const T a[2][2])` — construct from a 2x2 array.
    #[inline]
    pub const fn from_array(a: [[T; 2]; 2]) -> Self {
        Matrix22 {
            x: [[a[0][0], a[0][1]], [a[1][0], a[1][1]]],
        }
    }

    /// Port of `Matrix22(T a, T b, T c, T d)` — element-wise:
    ///
    /// ```text
    /// a b
    /// c d
    /// ```
    #[inline]
    pub const fn new(a: T, b: T, c: T, d: T) -> Self {
        Matrix22 { x: [[a, b], [c, d]] }
    }

    /// Port of `template <class S> explicit Matrix22(const Matrix22<S>& v)` —
    /// cross-base-type element conversion via the C++ `T(v.x[i][j])` cast.
    #[inline]
    pub fn from_matrix22<S: Copy>(v: Matrix22<S>) -> Self
    where
        T: VecCast<S>,
    {
        Matrix22 {
            x: [
                [T::from_elem(v.x[0][0]), T::from_elem(v.x[0][1])],
                [T::from_elem(v.x[1][0]), T::from_elem(v.x[1][1])],
            ],
        }
    }
}

// ----------------------------------------------------------------------------
// getValue / setValue (Sb compatibility) — safe, no raw pointers
// ----------------------------------------------------------------------------

impl<T: Copy> Matrix22<T> {
    /// Port of `const T* getValue() const` — returns the elements as a flat
    /// row-major `[T; 4]`. Replaces the C++ `reinterpret_cast<T*>(this)`.
    #[inline]
    pub fn to_array(&self) -> [T; 4] {
        [self.x[0][0], self.x[0][1], self.x[1][0], self.x[1][1]]
    }

    /// Port of `template <class S> void getValue(Matrix22<S>& v) const` — copy the
    /// elements into `v`, converting via `S(...)`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Matrix22<S>)
    where
        S: Copy + VecCast<T>,
    {
        v.x[0][0] = S::from_elem(self.x[0][0]);
        v.x[0][1] = S::from_elem(self.x[0][1]);
        v.x[1][0] = S::from_elem(self.x[1][0]);
        v.x[1][1] = S::from_elem(self.x[1][1]);
    }

    /// Port of `template <class S> Matrix22& setValue(const Matrix22<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: &Matrix22<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        self.x[0][0] = T::from_elem(v.x[0][0]);
        self.x[0][1] = T::from_elem(v.x[0][1]);
        self.x[1][0] = T::from_elem(v.x[1][0]);
        self.x[1][1] = T::from_elem(v.x[1][1]);
        self
    }

    /// Port of `template <class S> Matrix22& setTheMatrix(const Matrix22<S>& v)`.
    #[inline]
    pub fn set_the_matrix<S: Copy>(&mut self, v: &Matrix22<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        self.set_value(v)
    }
}

// ----------------------------------------------------------------------------
// Manipulation: makeIdentity
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix22<T> {
    /// Port of `void makeIdentity()` — set this matrix to the identity.
    #[inline]
    pub fn make_identity(&mut self) {
        self.x = [[T::ONE, T::ZERO], [T::ZERO, T::ONE]];
    }
}

// ----------------------------------------------------------------------------
// Equality — port of operator== / operator!=
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Matrix22<T> {
    /// Port of `bool operator==(const Matrix22& v) const`.
    #[inline]
    fn eq(&self, v: &Matrix22<T>) -> bool {
        self.x[0][0] == v.x[0][0]
            && self.x[0][1] == v.x[0][1]
            && self.x[1][0] == v.x[1][0]
            && self.x[1][1] == v.x[1][1]
    }
}

impl<T: Scalar> Matrix22<T> {
    /// Port of `bool equalWithAbsError(const Matrix22<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_abs_error(&self, m: &Matrix22<T>, e: T) -> bool {
        for i in 0..2 {
            for j in 0..2 {
                if !equal_with_abs_error(self.x[i][j], m.x[i][j], e) {
                    return false;
                }
            }
        }
        true
    }

    /// Port of `bool equalWithRelError(const Matrix22<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_rel_error(&self, m: &Matrix22<T>, e: T) -> bool {
        for i in 0..2 {
            for j in 0..2 {
                if !equal_with_rel_error(self.x[i][j], m.x[i][j], e) {
                    return false;
                }
            }
        }
        true
    }
}

// ----------------------------------------------------------------------------
// Arithmetic — component-wise +, -, *, /, unary -, and in-place forms
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix22<T> {
    /// Port of `const Matrix22& operator+=(const Matrix22& v)`.
    #[inline]
    pub fn add_assign_matrix(&mut self, v: &Matrix22<T>) -> &mut Self {
        self.x[0][0] = self.x[0][0] + v.x[0][0];
        self.x[0][1] = self.x[0][1] + v.x[0][1];
        self.x[1][0] = self.x[1][0] + v.x[1][0];
        self.x[1][1] = self.x[1][1] + v.x[1][1];
        self
    }

    /// Port of `const Matrix22& operator+=(T a)`.
    #[inline]
    pub fn add_assign_scalar(&mut self, a: T) -> &mut Self {
        self.x[0][0] = self.x[0][0] + a;
        self.x[0][1] = self.x[0][1] + a;
        self.x[1][0] = self.x[1][0] + a;
        self.x[1][1] = self.x[1][1] + a;
        self
    }

    /// Port of `Matrix22 operator+(const Matrix22& v) const`.
    #[inline]
    pub fn add_matrix(&self, v: &Matrix22<T>) -> Matrix22<T> {
        Matrix22::new(
            self.x[0][0] + v.x[0][0],
            self.x[0][1] + v.x[0][1],
            self.x[1][0] + v.x[1][0],
            self.x[1][1] + v.x[1][1],
        )
    }

    /// Port of `const Matrix22& operator-=(const Matrix22& v)`.
    #[inline]
    pub fn sub_assign_matrix(&mut self, v: &Matrix22<T>) -> &mut Self {
        self.x[0][0] = self.x[0][0] - v.x[0][0];
        self.x[0][1] = self.x[0][1] - v.x[0][1];
        self.x[1][0] = self.x[1][0] - v.x[1][0];
        self.x[1][1] = self.x[1][1] - v.x[1][1];
        self
    }

    /// Port of `const Matrix22& operator-=(T a)`.
    #[inline]
    pub fn sub_assign_scalar(&mut self, a: T) -> &mut Self {
        self.x[0][0] = self.x[0][0] - a;
        self.x[0][1] = self.x[0][1] - a;
        self.x[1][0] = self.x[1][0] - a;
        self.x[1][1] = self.x[1][1] - a;
        self
    }

    /// Port of `Matrix22 operator-(const Matrix22& v) const`.
    #[inline]
    pub fn sub_matrix(&self, v: &Matrix22<T>) -> Matrix22<T> {
        Matrix22::new(
            self.x[0][0] - v.x[0][0],
            self.x[0][1] - v.x[0][1],
            self.x[1][0] - v.x[1][0],
            self.x[1][1] - v.x[1][1],
        )
    }

    /// Port of `Matrix22 operator-() const` — component-wise negation.
    #[inline]
    pub fn negated(&self) -> Matrix22<T> {
        Matrix22::new(-self.x[0][0], -self.x[0][1], -self.x[1][0], -self.x[1][1])
    }

    /// Port of `const Matrix22& negate()` — negate in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        self.x[0][0] = -self.x[0][0];
        self.x[0][1] = -self.x[0][1];
        self.x[1][0] = -self.x[1][0];
        self.x[1][1] = -self.x[1][1];
        self
    }

    /// Port of `const Matrix22& operator*=(T a)` — scalar multiply in place.
    #[inline]
    pub fn mul_assign_scalar(&mut self, a: T) -> &mut Self {
        self.x[0][0] = self.x[0][0] * a;
        self.x[0][1] = self.x[0][1] * a;
        self.x[1][0] = self.x[1][0] * a;
        self.x[1][1] = self.x[1][1] * a;
        self
    }

    /// Port of `Matrix22 operator*(T a) const` — scalar multiply.
    #[inline]
    pub fn mul_scalar(&self, a: T) -> Matrix22<T> {
        Matrix22::new(
            self.x[0][0] * a,
            self.x[0][1] * a,
            self.x[1][0] * a,
            self.x[1][1] * a,
        )
    }

    /// Port of `const Matrix22& operator/=(T a)` — scalar divide in place.
    #[inline]
    pub fn div_assign_scalar(&mut self, a: T) -> &mut Self {
        self.x[0][0] = self.x[0][0] / a;
        self.x[0][1] = self.x[0][1] / a;
        self.x[1][0] = self.x[1][0] / a;
        self.x[1][1] = self.x[1][1] / a;
        self
    }

    /// Port of `Matrix22 operator/(T a) const` — scalar divide.
    #[inline]
    pub fn div_scalar(&self, a: T) -> Matrix22<T> {
        Matrix22::new(
            self.x[0][0] / a,
            self.x[0][1] / a,
            self.x[1][0] / a,
            self.x[1][1] / a,
        )
    }

    /// Port of `const Matrix22& operator*=(const Matrix22& v)` —
    /// matrix-matrix multiply in place (`this = this * v`).
    #[inline]
    pub fn mul_assign_matrix(&mut self, v: &Matrix22<T>) -> &mut Self {
        *self = self.mul_matrix(v);
        self
    }

    /// Port of `Matrix22 operator*(const Matrix22& v) const` — matrix product.
    #[inline]
    pub fn mul_matrix(&self, v: &Matrix22<T>) -> Matrix22<T> {
        let mut tmp = Matrix22::from_scalar(T::ZERO);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    tmp.x[i][j] = tmp.x[i][j] + self.x[i][k] * v.x[k][j];
                }
            }
        }
        tmp
    }
}

// ----------------------------------------------------------------------------
// Vector * matrix multiplication helpers
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix22<T> {
    /// Port of `template <class S> void multDirMatrix(const Vec2<S>& src,
    /// Vec2<S>& dst) const`.
    ///
    /// Computes `src * M` (vector treated as a row). Here `S == T` matching the
    /// homogeneous instantiation OpenEXR uses.
    #[inline]
    pub fn mult_dir_matrix(&self, src: &Vec2<T>, dst: &mut Vec2<T>) {
        let a = src.x * self.x[0][0] + src.y * self.x[1][0];
        let b = src.x * self.x[0][1] + src.y * self.x[1][1];
        dst.x = a;
        dst.y = b;
    }
}

// ----------------------------------------------------------------------------
// Transpose
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix22<T> {
    /// Port of `const Matrix22& transpose()` — transpose in place.
    #[inline]
    pub fn transpose(&mut self) -> &mut Self {
        let tmp = Matrix22::new(self.x[0][0], self.x[1][0], self.x[0][1], self.x[1][1]);
        *self = tmp;
        self
    }

    /// Port of `Matrix22 transposed() const` — return the transpose.
    #[inline]
    pub fn transposed(&self) -> Matrix22<T> {
        Matrix22::new(self.x[0][0], self.x[1][0], self.x[0][1], self.x[1][1])
    }
}

// ----------------------------------------------------------------------------
// Inverse / determinant / trace
// ----------------------------------------------------------------------------

impl<T: Scalar + VecLimits> Matrix22<T> {
    /// Port of the throwing `Matrix22<T> inverse(bool singExc) const` with
    /// `singExc == true`. Returns `Err` for a singular matrix instead of throwing
    /// `std::invalid_argument`.
    pub fn inverse_exc(&self) -> Result<Matrix22<T>, &'static str> {
        let mut s = Matrix22::new(self.x[1][1], -self.x[0][1], -self.x[1][0], self.x[0][0]);
        let r = self.x[0][0] * self.x[1][1] - self.x[1][0] * self.x[0][1];

        if abs_t(r) >= T::ONE {
            for i in 0..2 {
                for j in 0..2 {
                    s.x[i][j] = s.x[i][j] / r;
                }
            }
        } else {
            let mr = abs_t(r) / T::SMALLEST;
            for i in 0..2 {
                for j in 0..2 {
                    if mr > abs_t(s.x[i][j]) {
                        s.x[i][j] = s.x[i][j] / r;
                    } else {
                        return Err("Cannot invert singular matrix.");
                    }
                }
            }
        }
        Ok(s)
    }

    /// Port of the non-throwing `Matrix22<T> inverse() const`. Returns the identity
    /// matrix for a singular matrix (matching the C++ `return Matrix22()`).
    pub fn inverse(&self) -> Matrix22<T> {
        let mut s = Matrix22::new(self.x[1][1], -self.x[0][1], -self.x[1][0], self.x[0][0]);
        let r = self.x[0][0] * self.x[1][1] - self.x[1][0] * self.x[0][1];

        if abs_t(r) >= T::ONE {
            for i in 0..2 {
                for j in 0..2 {
                    s.x[i][j] = s.x[i][j] / r;
                }
            }
        } else {
            let mr = abs_t(r) / T::SMALLEST;
            for i in 0..2 {
                for j in 0..2 {
                    if mr > abs_t(s.x[i][j]) {
                        s.x[i][j] = s.x[i][j] / r;
                    } else {
                        return Matrix22::default();
                    }
                }
            }
        }
        s
    }

    /// Port of the throwing `const Matrix22& invert(bool singExc)` (`singExc==true`)
    /// — invert in place, error if singular.
    pub fn invert_exc(&mut self) -> Result<&mut Self, &'static str> {
        *self = self.inverse_exc()?;
        Ok(self)
    }

    /// Port of the non-throwing `const Matrix22& invert()` — invert in place,
    /// leaving the identity if the matrix is singular.
    pub fn invert(&mut self) -> &mut Self {
        *self = self.inverse();
        self
    }
}

impl<T: Scalar> Matrix22<T> {
    /// Port of `T determinant() const`.
    #[inline]
    pub fn determinant(&self) -> T {
        self.x[0][0] * self.x[1][1] - self.x[1][0] * self.x[0][1]
    }

    /// Port of `T trace() const`.
    #[inline]
    pub fn trace(&self) -> T {
        self.x[0][0] + self.x[1][1]
    }
}

// ----------------------------------------------------------------------------
// Rotation / scale (transcendental: needs `Math` for cos/sin)
// ----------------------------------------------------------------------------

impl<T: Scalar + Math> Matrix22<T> {
    /// Port of `template <class S> const Matrix22& setRotation(S r)` — set this to
    /// a rotation by `r` radians. Here `S == T` (the float instantiation).
    #[inline]
    pub fn set_rotation(&mut self, r: T) -> &mut Self {
        let cos_r = r.cos();
        let sin_r = r.sin();

        self.x[0][0] = cos_r;
        self.x[0][1] = sin_r;
        self.x[1][0] = -sin_r;
        self.x[1][1] = cos_r;
        self
    }

    /// Port of `template <class S> const Matrix22& rotate(S r)` — compose `this`
    /// with a rotation by `r` (`this *= setRotation(r)`).
    #[inline]
    pub fn rotate(&mut self, r: T) -> &mut Self {
        let mut rot = Matrix22::<T>::default();
        rot.set_rotation(r);
        *self = self.mul_matrix(&rot);
        self
    }
}

impl<T: Scalar> Matrix22<T> {
    /// Port of `const Matrix22& setScale(T s)` — set this to a uniform scale.
    #[inline]
    pub fn set_scale_scalar(&mut self, s: T) -> &mut Self {
        self.x[0][0] = s;
        self.x[0][1] = T::ZERO;
        self.x[1][0] = T::ZERO;
        self.x[1][1] = s;
        self
    }

    /// Port of `template <class S> const Matrix22& setScale(const Vec2<S>& s)` —
    /// set this to a non-uniform scale (here `S == T`).
    #[inline]
    pub fn set_scale(&mut self, s: &Vec2<T>) -> &mut Self {
        self.x[0][0] = s.x;
        self.x[0][1] = T::ZERO;
        self.x[1][0] = T::ZERO;
        self.x[1][1] = s.y;
        self
    }

    /// Port of `template <class S> const Matrix22& scale(const Vec2<S>& s)` —
    /// scale this matrix by `s`.
    #[inline]
    pub fn scale(&mut self, s: &Vec2<T>) -> &mut Self {
        self.x[0][0] = self.x[0][0] * s.x;
        self.x[0][1] = self.x[0][1] * s.x;
        self.x[1][0] = self.x[1][0] * s.y;
        self.x[1][1] = self.x[1][1] * s.y;
        self
    }
}

// ----------------------------------------------------------------------------
// Numeric limits / dimensions
// ----------------------------------------------------------------------------

impl<T: VecLimits> Matrix22<T> {
    /// Port of `static T baseTypeLowest()`.
    #[inline]
    pub fn base_type_lowest() -> T {
        T::LOWEST
    }
    /// Port of `static T baseTypeMax()`.
    #[inline]
    pub fn base_type_max() -> T {
        T::MAX
    }
    /// Port of `static T baseTypeSmallest()`.
    #[inline]
    pub fn base_type_smallest() -> T {
        T::SMALLEST
    }
    /// Port of `static T baseTypeEpsilon()`.
    #[inline]
    pub fn base_type_epsilon() -> T {
        T::EPSILON
    }
}

impl<T> Matrix22<T> {
    /// Port of `static unsigned int dimensions()` — always 2.
    #[inline]
    pub const fn dimensions() -> usize {
        2
    }
}

// ----------------------------------------------------------------------------
// Display — port of operator<<
// ----------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Matrix22<T> {
    /// Port of `operator<<` — `(m00 m01\n m10 m11)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} {}\n {} {})\n",
            self.x[0][0], self.x[0][1], self.x[1][0], self.x[1][1]
        )
    }
}

// ----------------------------------------------------------------------------
// Small helper: branch-based absolute value (no `fabs`, works for any Scalar).
// Mirrors `IMATH_INTERNAL_NAMESPACE::abs` as used by the C++ inverse code.
// ----------------------------------------------------------------------------

#[inline]
fn abs_t<T: Scalar>(a: T) -> T {
    if a >= T::ZERO { a } else { -a }
}

// ----------------------------------------------------------------------------
// Vector-times-matrix free operators (port of the trailing operator*/*= set)
// ----------------------------------------------------------------------------

/// Port of `const Vec2<S>& operator*=(Vec2<S>& v, const Matrix22<T>& m)` —
/// row-vector times matrix, in place (`v = v * m`). Here `S == T`.
#[inline]
pub fn vec2_mul_assign_matrix22<T: Scalar>(v: &mut Vec2<T>, m: &Matrix22<T>) {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1];
    v.x = x;
    v.y = y;
}

/// Port of `Vec2<S> operator*(const Vec2<S>& v, const Matrix22<T>& m)` —
/// row-vector times matrix (`v * m`). Here `S == T`.
#[inline]
pub fn vec2_mul_matrix22<T: Scalar>(v: &Vec2<T>, m: &Matrix22<T>) -> Vec2<T> {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1];
    Vec2::new(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        let m = Matrix22::<f32>::default();
        assert_eq!(m.x, [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn per_element_ctor_and_index() {
        let m = Matrix22::new(1.0_f64, 2.0, 3.0, 4.0);
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 2.0);
        assert_eq!(m[1][0], 3.0);
        assert_eq!(m[1][1], 4.0);
    }

    #[test]
    fn matrix_product() {
        let a = Matrix22::new(1.0_f64, 2.0, 3.0, 4.0);
        let b = Matrix22::new(5.0, 6.0, 7.0, 8.0);
        // [1 2;3 4]*[5 6;7 8] = [19 22; 43 50]
        let c = a.mul_matrix(&b);
        assert_eq!(c.x, [[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn transpose_roundtrip() {
        let m = Matrix22::new(1.0_f64, 2.0, 3.0, 4.0);
        assert_eq!(m.transposed().x, [[1.0, 3.0], [2.0, 4.0]]);
    }

    #[test]
    fn determinant_known() {
        let m = Matrix22::new(1.0_f64, 2.0, 3.0, 4.0);
        assert_eq!(m.determinant(), -2.0);
    }

    #[test]
    fn inverse_roundtrip() {
        let m = Matrix22::new(4.0_f64, 7.0, 2.0, 6.0);
        let inv = m.inverse();
        let prod = m.mul_matrix(&inv);
        assert!(prod.equal_with_abs_error(&Matrix22::default(), 1e-12));
    }

    #[test]
    fn inverse_singular_returns_identity() {
        let m = Matrix22::from_scalar(0.0_f64);
        assert_eq!(m.inverse(), Matrix22::default());
        assert!(m.inverse_exc().is_err());
    }

    #[test]
    fn set_rotation_known() {
        let mut m = Matrix22::<f64>::default();
        m.set_rotation(core::f64::consts::FRAC_PI_2);
        // rotate (1,0) -> should map to (0,1) under v*m
        let v = Vec2::new(1.0, 0.0);
        let r = vec2_mul_matrix22(&v, &m);
        assert!(r.x.abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trace_known() {
        let m = Matrix22::new(1.0_f64, 2.0, 3.0, 4.0);
        assert_eq!(m.trace(), 5.0);
    }

    #[test]
    fn dimensions_is_2() {
        assert_eq!(Matrix22::<f32>::dimensions(), 2);
    }

    #[test]
    fn cross_type_convert() {
        let m = Matrix22::new(1.0_f64, 2.5, 3.0, 4.0);
        let mf: Matrix22<f32> = Matrix22::from_matrix22(m);
        assert_eq!(mf.x, [[1.0_f32, 2.5], [3.0, 4.0]]);
    }
}
