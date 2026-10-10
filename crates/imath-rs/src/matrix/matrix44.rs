//! Port of Imath ImathMatrix.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrix.h>
//!
//! The [`Matrix44`] 4x4 transformation matrix. See [`crate::matrix`] for the shared
//! design notes (row-major `x: [[T; 4]; 4]` storage, identity-default, the
//! exception-vs-`Result` convention for singular inverse, and the foreign-interop
//! skip).

use core::fmt;
use core::ops::{Index, IndexMut};

use crate::math::{Math, equal_with_abs_error, equal_with_rel_error};
use crate::matrix::Matrix33;
use crate::shear::Shear6;
use crate::traits::Scalar;
use crate::vec::{Vec3, Vec4, VecCast, VecFloat, VecLimits};

/// 4x4 transformation matrix (port of C++ `template <class T> class Matrix44`).
///
/// Elements are stored row-major in `x: [[T; 4]; 4]`, byte-for-byte matching the
/// C++ `T x[4][4]`. `#[repr(C)]` guarantees that layout. Row access (`m[i]`) is the
/// safe analogue of the C++ `operator[]` row pointer; `m[i][j]` indexes an element.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix44<T> {
    /// Matrix elements, row-major (`x[row][col]`), mirroring C++ `T x[4][4]`.
    pub x: [[T; 4]; 4],
}

// ----------------------------------------------------------------------------
// Row access — port of `operator[]`
// ----------------------------------------------------------------------------

impl<T> Index<usize> for Matrix44<T> {
    type Output = [T; 4];
    /// Port of `T* operator[](int i)` — returns row `i`.
    #[inline]
    fn index(&self, i: usize) -> &[T; 4] {
        &self.x[i]
    }
}

impl<T> IndexMut<usize> for Matrix44<T> {
    /// Port of `operator[]` (mutable) — returns row `i`.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut [T; 4] {
        &mut self.x[i]
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Matrix44<T> {
    /// Port of the default constructor `Matrix44()` — the identity matrix.
    #[inline]
    fn default() -> Self {
        let z = T::ZERO;
        let o = T::ONE;
        Matrix44 {
            x: [
                [o, z, z, z],
                [z, o, z, z],
                [z, z, o, z],
                [z, z, z, o],
            ],
        }
    }
}

impl<T: Copy> Matrix44<T> {
    /// Port of `Matrix44(T a)` — every element set to the scalar `a`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Matrix44 {
            x: [[a, a, a, a], [a, a, a, a], [a, a, a, a], [a, a, a, a]],
        }
    }

    /// Port of `Matrix44(const T a[4][4])` — construct from a 4x4 array.
    #[inline]
    pub const fn from_array(a: [[T; 4]; 4]) -> Self {
        Matrix44 { x: a }
    }

    /// Port of the 16-scalar constructor:
    ///
    /// ```text
    /// a b c d
    /// e f g h
    /// i j k l
    /// m n o p
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        a: T, b: T, c: T, d: T, e: T, f: T, g: T, h: T, i: T, j: T, k: T, l: T, m: T, n: T, o: T,
        p: T,
    ) -> Self {
        Matrix44 {
            x: [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]],
        }
    }

    /// Port of `template <class S> explicit Matrix44(const Matrix44<S>& v)` —
    /// cross-base-type element conversion via the C++ `T(v.x[i][j])` cast.
    #[inline]
    pub fn from_matrix44<S: Copy>(v: Matrix44<S>) -> Self
    where
        T: VecCast<S>,
    {
        let mut out = Matrix44::from_scalar(T::from_elem(v.x[0][0]));
        for i in 0..4 {
            for j in 0..4 {
                out.x[i][j] = T::from_elem(v.x[i][j]);
            }
        }
        out
    }
}

impl<T: Scalar> Matrix44<T> {
    /// Port of `Matrix44(Matrix33<T> r, Vec3<T> t)` — a 3x3 rotation block plus a
    /// translation row:
    ///
    /// ```text
    /// r r r 0
    /// r r r 0
    /// r r r 0
    /// t t t 1
    /// ```
    #[inline]
    pub fn from_rot_trans(r: Matrix33<T>, t: Vec3<T>) -> Self {
        Matrix44 {
            x: [
                [r.x[0][0], r.x[0][1], r.x[0][2], T::ZERO],
                [r.x[1][0], r.x[1][1], r.x[1][2], T::ZERO],
                [r.x[2][0], r.x[2][1], r.x[2][2], T::ZERO],
                [t.x, t.y, t.z, T::ONE],
            ],
        }
    }
}

// ----------------------------------------------------------------------------
// getValue / setValue (Sb compatibility) — safe, no raw pointers
// ----------------------------------------------------------------------------

impl<T: Copy> Matrix44<T> {
    /// Port of `const T* getValue() const` — flat row-major `[T; 16]`.
    #[inline]
    pub fn to_array(&self) -> [T; 16] {
        let mut out = [self.x[0][0]; 16];
        let mut idx = 0;
        for i in 0..4 {
            for j in 0..4 {
                out[idx] = self.x[i][j];
                idx += 1;
            }
        }
        out
    }

    /// Port of `template <class S> void getValue(Matrix44<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Matrix44<S>)
    where
        S: Copy + VecCast<T>,
    {
        for i in 0..4 {
            for j in 0..4 {
                v.x[i][j] = S::from_elem(self.x[i][j]);
            }
        }
    }

    /// Port of `template <class S> Matrix44& setValue(const Matrix44<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: &Matrix44<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = T::from_elem(v.x[i][j]);
            }
        }
        self
    }

    /// Port of `template <class S> Matrix44& setTheMatrix(const Matrix44<S>& v)`.
    #[inline]
    pub fn set_the_matrix<S: Copy>(&mut self, v: &Matrix44<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        self.set_value(v)
    }
}

// ----------------------------------------------------------------------------
// makeIdentity
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `void makeIdentity()`.
    #[inline]
    pub fn make_identity(&mut self) {
        *self = Matrix44::default();
    }
}

// ----------------------------------------------------------------------------
// Equality
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Matrix44<T> {
    /// Port of `bool operator==(const Matrix44& v) const`.
    #[inline]
    fn eq(&self, v: &Matrix44<T>) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if self.x[i][j] != v.x[i][j] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Scalar> Matrix44<T> {
    /// Port of `bool equalWithAbsError(const Matrix44<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_abs_error(&self, m: &Matrix44<T>, e: T) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if !equal_with_abs_error(self.x[i][j], m.x[i][j], e) {
                    return false;
                }
            }
        }
        true
    }

    /// Port of `bool equalWithRelError(const Matrix44<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_rel_error(&self, m: &Matrix44<T>, e: T) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if !equal_with_rel_error(self.x[i][j], m.x[i][j], e) {
                    return false;
                }
            }
        }
        true
    }
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `const Matrix44& operator+=(const Matrix44& v)`.
    #[inline]
    pub fn add_assign_matrix(&mut self, v: &Matrix44<T>) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] + v.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix44& operator+=(T a)`.
    #[inline]
    pub fn add_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] + a;
            }
        }
        self
    }

    /// Port of `Matrix44 operator+(const Matrix44& v) const`.
    #[inline]
    pub fn add_matrix(&self, v: &Matrix44<T>) -> Matrix44<T> {
        let mut out = *self;
        out.add_assign_matrix(v);
        out
    }

    /// Port of `const Matrix44& operator-=(const Matrix44& v)`.
    #[inline]
    pub fn sub_assign_matrix(&mut self, v: &Matrix44<T>) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] - v.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix44& operator-=(T a)`.
    #[inline]
    pub fn sub_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] - a;
            }
        }
        self
    }

    /// Port of `Matrix44 operator-(const Matrix44& v) const`.
    #[inline]
    pub fn sub_matrix(&self, v: &Matrix44<T>) -> Matrix44<T> {
        let mut out = *self;
        out.sub_assign_matrix(v);
        out
    }

    /// Port of `Matrix44 operator-() const` — component-wise negation.
    #[inline]
    pub fn negated(&self) -> Matrix44<T> {
        let mut out = *self;
        out.negate();
        out
    }

    /// Port of `const Matrix44& negate()` — negate in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = -self.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix44& operator*=(T a)` — scalar multiply in place.
    #[inline]
    pub fn mul_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] * a;
            }
        }
        self
    }

    /// Port of `Matrix44 operator*(T a) const` — scalar multiply.
    #[inline]
    pub fn mul_scalar(&self, a: T) -> Matrix44<T> {
        let mut out = *self;
        out.mul_assign_scalar(a);
        out
    }

    /// Port of `const Matrix44& operator/=(T a)` — scalar divide in place.
    #[inline]
    pub fn div_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..4 {
            for j in 0..4 {
                self.x[i][j] = self.x[i][j] / a;
            }
        }
        self
    }

    /// Port of `Matrix44 operator/(T a) const` — scalar divide.
    #[inline]
    pub fn div_scalar(&self, a: T) -> Matrix44<T> {
        let mut out = *self;
        out.div_assign_scalar(a);
        out
    }

    /// Port of `static Matrix44 multiply(const Matrix44& a, const Matrix44& b)` —
    /// the unrolled matrix product (1:1 with the C++ that caches each `a` row).
    #[inline]
    pub fn multiply(a: &Matrix44<T>, b: &Matrix44<T>) -> Matrix44<T> {
        let a00 = a.x[0][0];
        let a01 = a.x[0][1];
        let a02 = a.x[0][2];
        let a03 = a.x[0][3];
        let c00 = a00 * b.x[0][0] + a01 * b.x[1][0] + a02 * b.x[2][0] + a03 * b.x[3][0];
        let c01 = a00 * b.x[0][1] + a01 * b.x[1][1] + a02 * b.x[2][1] + a03 * b.x[3][1];
        let c02 = a00 * b.x[0][2] + a01 * b.x[1][2] + a02 * b.x[2][2] + a03 * b.x[3][2];
        let c03 = a00 * b.x[0][3] + a01 * b.x[1][3] + a02 * b.x[2][3] + a03 * b.x[3][3];

        let a10 = a.x[1][0];
        let a11 = a.x[1][1];
        let a12 = a.x[1][2];
        let a13 = a.x[1][3];
        let c10 = a10 * b.x[0][0] + a11 * b.x[1][0] + a12 * b.x[2][0] + a13 * b.x[3][0];
        let c11 = a10 * b.x[0][1] + a11 * b.x[1][1] + a12 * b.x[2][1] + a13 * b.x[3][1];
        let c12 = a10 * b.x[0][2] + a11 * b.x[1][2] + a12 * b.x[2][2] + a13 * b.x[3][2];
        let c13 = a10 * b.x[0][3] + a11 * b.x[1][3] + a12 * b.x[2][3] + a13 * b.x[3][3];

        let a20 = a.x[2][0];
        let a21 = a.x[2][1];
        let a22 = a.x[2][2];
        let a23 = a.x[2][3];
        let c20 = a20 * b.x[0][0] + a21 * b.x[1][0] + a22 * b.x[2][0] + a23 * b.x[3][0];
        let c21 = a20 * b.x[0][1] + a21 * b.x[1][1] + a22 * b.x[2][1] + a23 * b.x[3][1];
        let c22 = a20 * b.x[0][2] + a21 * b.x[1][2] + a22 * b.x[2][2] + a23 * b.x[3][2];
        let c23 = a20 * b.x[0][3] + a21 * b.x[1][3] + a22 * b.x[2][3] + a23 * b.x[3][3];

        let a30 = a.x[3][0];
        let a31 = a.x[3][1];
        let a32 = a.x[3][2];
        let a33 = a.x[3][3];
        let c30 = a30 * b.x[0][0] + a31 * b.x[1][0] + a32 * b.x[2][0] + a33 * b.x[3][0];
        let c31 = a30 * b.x[0][1] + a31 * b.x[1][1] + a32 * b.x[2][1] + a33 * b.x[3][1];
        let c32 = a30 * b.x[0][2] + a31 * b.x[1][2] + a32 * b.x[2][2] + a33 * b.x[3][2];
        let c33 = a30 * b.x[0][3] + a31 * b.x[1][3] + a32 * b.x[2][3] + a33 * b.x[3][3];

        Matrix44::new(
            c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
        )
    }

    /// Port of the static `void multiply(const Matrix44& a, const Matrix44& b,
    /// Matrix44& c)` — writes the product into `c`.
    #[inline]
    pub fn multiply_into(a: &Matrix44<T>, b: &Matrix44<T>, c: &mut Matrix44<T>) {
        *c = Matrix44::multiply(a, b);
    }

    /// Port of `Matrix44 operator*(const Matrix44& v) const`.
    #[inline]
    pub fn mul_matrix(&self, v: &Matrix44<T>) -> Matrix44<T> {
        Matrix44::multiply(self, v)
    }

    /// Port of `const Matrix44& operator*=(const Matrix44& v)`.
    #[inline]
    pub fn mul_assign_matrix(&mut self, v: &Matrix44<T>) -> &mut Self {
        *self = Matrix44::multiply(self, v);
        self
    }
}

// ----------------------------------------------------------------------------
// Vector * matrix multiplication helpers
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `template <class S> void multVecMatrix(const Vec3<S>& src,
    /// Vec3<S>& dst) const` — homogeneous transform with the divide by `w`.
    #[inline]
    pub fn mult_vec_matrix(&self, src: &Vec3<T>, dst: &mut Vec3<T>) {
        let a = src.x * self.x[0][0] + src.y * self.x[1][0] + src.z * self.x[2][0] + self.x[3][0];
        let b = src.x * self.x[0][1] + src.y * self.x[1][1] + src.z * self.x[2][1] + self.x[3][1];
        let c = src.x * self.x[0][2] + src.y * self.x[1][2] + src.z * self.x[2][2] + self.x[3][2];
        let w = src.x * self.x[0][3] + src.y * self.x[1][3] + src.z * self.x[2][3] + self.x[3][3];
        dst.x = a / w;
        dst.y = b / w;
        dst.z = c / w;
    }

    /// Port of `template <class S> void multDirMatrix(const Vec3<S>& src,
    /// Vec3<S>& dst) const` — upper-left 3x3 only.
    #[inline]
    pub fn mult_dir_matrix(&self, src: &Vec3<T>, dst: &mut Vec3<T>) {
        let a = src.x * self.x[0][0] + src.y * self.x[1][0] + src.z * self.x[2][0];
        let b = src.x * self.x[0][1] + src.y * self.x[1][1] + src.z * self.x[2][1];
        let c = src.x * self.x[0][2] + src.y * self.x[1][2] + src.z * self.x[2][2];
        dst.x = a;
        dst.y = b;
        dst.z = c;
    }
}

// ----------------------------------------------------------------------------
// Transpose
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `const Matrix44& transpose()`.
    #[inline]
    pub fn transpose(&mut self) -> &mut Self {
        *self = self.transposed();
        self
    }

    /// Port of `Matrix44 transposed() const`.
    #[inline]
    pub fn transposed(&self) -> Matrix44<T> {
        Matrix44::new(
            self.x[0][0],
            self.x[1][0],
            self.x[2][0],
            self.x[3][0],
            self.x[0][1],
            self.x[1][1],
            self.x[2][1],
            self.x[3][1],
            self.x[0][2],
            self.x[1][2],
            self.x[2][2],
            self.x[3][2],
            self.x[0][3],
            self.x[1][3],
            self.x[2][3],
            self.x[3][3],
        )
    }
}

// ----------------------------------------------------------------------------
// Gauss-Jordan inverse
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Shared Gauss-Jordan elimination with partial pivoting (1:1 with C++
    /// `gjInverse`). `Err(())` on singular.
    fn gj_inverse_impl(&self) -> Result<Matrix44<T>, ()> {
        let mut s = Matrix44::<T>::default();
        let mut t = *self;

        // Forward elimination
        for i in 0..3 {
            let mut pivot = i;
            let mut pivotsize = t.x[i][i];
            if pivotsize < T::ZERO {
                pivotsize = -pivotsize;
            }

            for j in (i + 1)..4 {
                let mut tmp = t.x[j][i];
                if tmp < T::ZERO {
                    tmp = -tmp;
                }
                if tmp > pivotsize {
                    pivot = j;
                    pivotsize = tmp;
                }
            }

            if pivotsize == T::ZERO {
                return Err(());
            }

            if pivot != i {
                for j in 0..4 {
                    let tmp = t.x[i][j];
                    t.x[i][j] = t.x[pivot][j];
                    t.x[pivot][j] = tmp;

                    let tmp = s.x[i][j];
                    s.x[i][j] = s.x[pivot][j];
                    s.x[pivot][j] = tmp;
                }
            }

            for j in (i + 1)..4 {
                let f = t.x[j][i] / t.x[i][i];
                for k in 0..4 {
                    t.x[j][k] = t.x[j][k] - f * t.x[i][k];
                    s.x[j][k] = s.x[j][k] - f * s.x[i][k];
                }
            }
        }

        // Backward substitution
        for i in (0..4).rev() {
            let f = t.x[i][i];
            if f == T::ZERO {
                return Err(());
            }

            for j in 0..4 {
                t.x[i][j] = t.x[i][j] / f;
                s.x[i][j] = s.x[i][j] / f;
            }

            for j in 0..i {
                let f = t.x[j][i];
                for k in 0..4 {
                    t.x[j][k] = t.x[j][k] - f * t.x[i][k];
                    s.x[j][k] = s.x[j][k] - f * s.x[i][k];
                }
            }
        }

        Ok(s)
    }

    /// Port of the throwing `Matrix44<T> gjInverse(bool singExc) const`.
    #[inline]
    pub fn gj_inverse_exc(&self) -> Result<Matrix44<T>, &'static str> {
        self.gj_inverse_impl()
            .map_err(|_| "Cannot invert singular matrix.")
    }

    /// Port of the non-throwing `Matrix44<T> gjInverse() const` — identity on
    /// singular.
    #[inline]
    pub fn gj_inverse(&self) -> Matrix44<T> {
        self.gj_inverse_impl().unwrap_or_default()
    }

    /// Port of the throwing `const Matrix44& gjInvert(bool singExc)`.
    #[inline]
    pub fn gj_invert_exc(&mut self) -> Result<&mut Self, &'static str> {
        *self = self.gj_inverse_exc()?;
        Ok(self)
    }

    /// Port of the non-throwing `const Matrix44& gjInvert()`.
    #[inline]
    pub fn gj_invert(&mut self) -> &mut Self {
        *self = self.gj_inverse();
        self
    }
}

// ----------------------------------------------------------------------------
// Determinant-based inverse (with affine fast path / gjInverse fallback)
// ----------------------------------------------------------------------------

impl<T: Scalar + VecLimits> Matrix44<T> {
    /// Shared implementation of `inverse`. `sing_exc` selects `Err` vs identity on
    /// singular. Mirrors the C++: if the last column isn't `[0,0,0,1]`, defer to
    /// `gjInverse`; otherwise use the cofactor formula for the upper-left 3x3 plus
    /// the translation correction.
    fn inverse_impl(&self, sing_exc: bool) -> Result<Matrix44<T>, &'static str> {
        if self.x[0][3] != T::ZERO
            || self.x[1][3] != T::ZERO
            || self.x[2][3] != T::ZERO
            || self.x[3][3] != T::ONE
        {
            return if sing_exc {
                self.gj_inverse_exc()
            } else {
                Ok(self.gj_inverse())
            };
        }

        let mut s = Matrix44::new(
            self.x[1][1] * self.x[2][2] - self.x[2][1] * self.x[1][2],
            self.x[2][1] * self.x[0][2] - self.x[0][1] * self.x[2][2],
            self.x[0][1] * self.x[1][2] - self.x[1][1] * self.x[0][2],
            T::ZERO,
            self.x[2][0] * self.x[1][2] - self.x[1][0] * self.x[2][2],
            self.x[0][0] * self.x[2][2] - self.x[2][0] * self.x[0][2],
            self.x[1][0] * self.x[0][2] - self.x[0][0] * self.x[1][2],
            T::ZERO,
            self.x[1][0] * self.x[2][1] - self.x[2][0] * self.x[1][1],
            self.x[2][0] * self.x[0][1] - self.x[0][0] * self.x[2][1],
            self.x[0][0] * self.x[1][1] - self.x[1][0] * self.x[0][1],
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ONE,
        );

        let r = self.x[0][0] * s.x[0][0] + self.x[0][1] * s.x[1][0] + self.x[0][2] * s.x[2][0];

        if abs_t(r) >= T::ONE {
            for i in 0..3 {
                for j in 0..3 {
                    s.x[i][j] = s.x[i][j] / r;
                }
            }
        } else {
            let mr = abs_t(r) / T::SMALLEST;
            for i in 0..3 {
                for j in 0..3 {
                    if mr > abs_t(s.x[i][j]) {
                        s.x[i][j] = s.x[i][j] / r;
                    } else if sing_exc {
                        return Err("Cannot invert singular matrix.");
                    } else {
                        return Ok(Matrix44::default());
                    }
                }
            }
        }

        s.x[3][0] = -self.x[3][0] * s.x[0][0] - self.x[3][1] * s.x[1][0] - self.x[3][2] * s.x[2][0];
        s.x[3][1] = -self.x[3][0] * s.x[0][1] - self.x[3][1] * s.x[1][1] - self.x[3][2] * s.x[2][1];
        s.x[3][2] = -self.x[3][0] * s.x[0][2] - self.x[3][1] * s.x[1][2] - self.x[3][2] * s.x[2][2];

        Ok(s)
    }

    /// Port of the throwing `Matrix44<T> inverse(bool singExc) const`.
    #[inline]
    pub fn inverse_exc(&self) -> Result<Matrix44<T>, &'static str> {
        self.inverse_impl(true)
    }

    /// Port of the non-throwing `Matrix44<T> inverse() const` — identity on
    /// singular.
    #[inline]
    pub fn inverse(&self) -> Matrix44<T> {
        self.inverse_impl(false).unwrap_or_default()
    }

    /// Port of the throwing `const Matrix44& invert(bool singExc)`.
    #[inline]
    pub fn invert_exc(&mut self) -> Result<&mut Self, &'static str> {
        *self = self.inverse_exc()?;
        Ok(self)
    }

    /// Port of the non-throwing `const Matrix44& invert()`.
    #[inline]
    pub fn invert(&mut self) -> &mut Self {
        *self = self.inverse();
        self
    }
}

// ----------------------------------------------------------------------------
// minorOf / fastMinor / determinant / trace
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `T fastMinor(r0, r1, r2, c0, c1, c2) const` — the 3x3 minor on the
    /// given rows/cols.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn fast_minor(
        &self,
        r0: usize,
        r1: usize,
        r2: usize,
        c0: usize,
        c1: usize,
        c2: usize,
    ) -> T {
        self.x[r0][c0] * (self.x[r1][c1] * self.x[r2][c2] - self.x[r1][c2] * self.x[r2][c1])
            + self.x[r0][c1] * (self.x[r1][c2] * self.x[r2][c0] - self.x[r1][c0] * self.x[r2][c2])
            + self.x[r0][c2] * (self.x[r1][c0] * self.x[r2][c1] - self.x[r1][c1] * self.x[r2][c0])
    }

    /// Port of `T minorOf(const int r, const int c) const` — builds the 3x3
    /// submatrix excluding row `r`, col `c` and returns its determinant.
    #[inline]
    pub fn minor_of(&self, r: usize, c: usize) -> T {
        let r0 = if r < 1 { 1 } else { 0 };
        let r1 = 1 + if r < 2 { 1 } else { 0 };
        let r2 = 2 + if r < 3 { 1 } else { 0 };
        let c0 = if c < 1 { 1 } else { 0 };
        let c1 = 1 + if c < 2 { 1 } else { 0 };
        let c2 = 2 + if c < 3 { 1 } else { 0 };

        let working = Matrix33::new(
            self.x[r0][c0],
            self.x[r1][c0],
            self.x[r2][c0],
            self.x[r0][c1],
            self.x[r1][c1],
            self.x[r2][c1],
            self.x[r0][c2],
            self.x[r1][c2],
            self.x[r2][c2],
        );
        working.determinant()
    }

    /// Port of `T determinant() const` — cofactor expansion along the last column,
    /// skipping zero terms exactly as the C++ does.
    #[inline]
    pub fn determinant(&self) -> T {
        let mut sum = T::ZERO;
        if self.x[0][3] != T::ZERO {
            sum = sum - self.x[0][3] * self.fast_minor(1, 2, 3, 0, 1, 2);
        }
        if self.x[1][3] != T::ZERO {
            sum = sum + self.x[1][3] * self.fast_minor(0, 2, 3, 0, 1, 2);
        }
        if self.x[2][3] != T::ZERO {
            sum = sum - self.x[2][3] * self.fast_minor(0, 1, 3, 0, 1, 2);
        }
        if self.x[3][3] != T::ZERO {
            sum = sum + self.x[3][3] * self.fast_minor(0, 1, 2, 0, 1, 2);
        }
        sum
    }

    /// Port of `T trace() const`.
    #[inline]
    pub fn trace(&self) -> T {
        self.x[0][0] + self.x[1][1] + self.x[2][2] + self.x[3][3]
    }
}

// ----------------------------------------------------------------------------
// Euler angles / axis-angle / rotate (transcendental)
// ----------------------------------------------------------------------------

impl<T: Scalar + Math> Matrix44<T> {
    /// Port of `template <class S> const Matrix44& setEulerAngles(const Vec3<S>& r)`
    /// — XYZ Euler rotation. Composition order is exactly the C++ one.
    #[inline]
    pub fn set_euler_angles(&mut self, r: &Vec3<T>) -> &mut Self {
        let cos_rz = r.z.cos();
        let cos_ry = r.y.cos();
        let cos_rx = r.x.cos();
        let sin_rz = r.z.sin();
        let sin_ry = r.y.sin();
        let sin_rx = r.x.sin();

        self.x[0][0] = cos_rz * cos_ry;
        self.x[0][1] = sin_rz * cos_ry;
        self.x[0][2] = -sin_ry;
        self.x[0][3] = T::ZERO;

        self.x[1][0] = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
        self.x[1][1] = cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
        self.x[1][2] = cos_ry * sin_rx;
        self.x[1][3] = T::ZERO;

        self.x[2][0] = sin_rz * sin_rx + cos_rz * sin_ry * cos_rx;
        self.x[2][1] = -cos_rz * sin_rx + sin_rz * sin_ry * cos_rx;
        self.x[2][2] = cos_ry * cos_rx;
        self.x[2][3] = T::ZERO;

        self.x[3][0] = T::ZERO;
        self.x[3][1] = T::ZERO;
        self.x[3][2] = T::ZERO;
        self.x[3][3] = T::ONE;
        self
    }

    /// Port of `template <class S> const Matrix44& rotate(const Vec3<S>& r)` —
    /// compose this with an XYZ Euler rotation. Mirrors the C++ exactly, including
    /// its `m2x = -sin_rz * -sin_rx + ...` sign pattern.
    #[inline]
    pub fn rotate(&mut self, r: &Vec3<T>) -> &mut Self {
        let cos_rz = r.z.cos();
        let cos_ry = r.y.cos();
        let cos_rx = r.x.cos();
        let sin_rz = r.z.sin();
        let sin_ry = r.y.sin();
        let sin_rx = r.x.sin();

        let m00 = cos_rz * cos_ry;
        let m01 = sin_rz * cos_ry;
        let m02 = -sin_ry;
        let m10 = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
        let m11 = cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
        let m12 = cos_ry * sin_rx;
        let m20 = -sin_rz * -sin_rx + cos_rz * sin_ry * cos_rx;
        let m21 = cos_rz * -sin_rx + sin_rz * sin_ry * cos_rx;
        let m22 = cos_ry * cos_rx;

        let p = *self;

        self.x[0][0] = p.x[0][0] * m00 + p.x[1][0] * m01 + p.x[2][0] * m02;
        self.x[0][1] = p.x[0][1] * m00 + p.x[1][1] * m01 + p.x[2][1] * m02;
        self.x[0][2] = p.x[0][2] * m00 + p.x[1][2] * m01 + p.x[2][2] * m02;
        self.x[0][3] = p.x[0][3] * m00 + p.x[1][3] * m01 + p.x[2][3] * m02;

        self.x[1][0] = p.x[0][0] * m10 + p.x[1][0] * m11 + p.x[2][0] * m12;
        self.x[1][1] = p.x[0][1] * m10 + p.x[1][1] * m11 + p.x[2][1] * m12;
        self.x[1][2] = p.x[0][2] * m10 + p.x[1][2] * m11 + p.x[2][2] * m12;
        self.x[1][3] = p.x[0][3] * m10 + p.x[1][3] * m11 + p.x[2][3] * m12;

        self.x[2][0] = p.x[0][0] * m20 + p.x[1][0] * m21 + p.x[2][0] * m22;
        self.x[2][1] = p.x[0][1] * m20 + p.x[1][1] * m21 + p.x[2][1] * m22;
        self.x[2][2] = p.x[0][2] * m20 + p.x[1][2] * m21 + p.x[2][2] * m22;
        self.x[2][3] = p.x[0][3] * m20 + p.x[1][3] * m21 + p.x[2][3] * m22;
        self
    }
}

impl<T: Scalar + Math + VecFloat> Matrix44<T> {
    /// Port of `template <class S> const Matrix44& setAxisAngle(const Vec3<S>& ax,
    /// S ang)` — rotation by `angle` radians about `axis`. Needs `VecFloat` for
    /// `axis.normalized()`.
    #[inline]
    pub fn set_axis_angle(&mut self, axis: &Vec3<T>, angle: T) -> &mut Self {
        let unit = axis.normalized();
        let sine = angle.sin();
        let cosine = angle.cos();
        let one = T::ONE;

        self.x[0][0] = unit.x * unit.x * (one - cosine) + cosine;
        self.x[0][1] = unit.x * unit.y * (one - cosine) + unit.z * sine;
        self.x[0][2] = unit.x * unit.z * (one - cosine) - unit.y * sine;
        self.x[0][3] = T::ZERO;

        self.x[1][0] = unit.x * unit.y * (one - cosine) - unit.z * sine;
        self.x[1][1] = unit.y * unit.y * (one - cosine) + cosine;
        self.x[1][2] = unit.y * unit.z * (one - cosine) + unit.x * sine;
        self.x[1][3] = T::ZERO;

        self.x[2][0] = unit.x * unit.z * (one - cosine) + unit.y * sine;
        self.x[2][1] = unit.y * unit.z * (one - cosine) - unit.x * sine;
        self.x[2][2] = unit.z * unit.z * (one - cosine) + cosine;
        self.x[2][3] = T::ZERO;

        self.x[3][0] = T::ZERO;
        self.x[3][1] = T::ZERO;
        self.x[3][2] = T::ZERO;
        self.x[3][3] = T::ONE;
        self
    }
}

// ----------------------------------------------------------------------------
// Scale / translation / shear
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix44<T> {
    /// Port of `const Matrix44& setScale(T s)` — uniform 3D scale.
    #[inline]
    pub fn set_scale_scalar(&mut self, s: T) -> &mut Self {
        let z = T::ZERO;
        self.x = [
            [s, z, z, z],
            [z, s, z, z],
            [z, z, s, z],
            [z, z, z, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix44& setScale(const Vec3<S>& s)`.
    #[inline]
    pub fn set_scale(&mut self, s: &Vec3<T>) -> &mut Self {
        let z = T::ZERO;
        self.x = [
            [s.x, z, z, z],
            [z, s.y, z, z],
            [z, z, s.z, z],
            [z, z, z, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix44& scale(const Vec3<S>& s)`.
    #[inline]
    pub fn scale(&mut self, s: &Vec3<T>) -> &mut Self {
        for j in 0..4 {
            self.x[0][j] = self.x[0][j] * s.x;
        }
        for j in 0..4 {
            self.x[1][j] = self.x[1][j] * s.y;
        }
        for j in 0..4 {
            self.x[2][j] = self.x[2][j] * s.z;
        }
        self
    }

    /// Port of `template <class S> const Matrix44& setTranslation(const Vec3<S>& t)`.
    #[inline]
    pub fn set_translation(&mut self, t: &Vec3<T>) -> &mut Self {
        let z = T::ZERO;
        let o = T::ONE;
        self.x = [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, o, z],
            [t.x, t.y, t.z, o],
        ];
        self
    }

    /// Port of `Vec3<T> translation() const`.
    #[inline]
    pub fn translation(&self) -> Vec3<T> {
        Vec3::new(self.x[3][0], self.x[3][1], self.x[3][2])
    }

    /// Port of `template <class S> const Matrix44& translate(const Vec3<S>& t)`.
    #[inline]
    pub fn translate(&mut self, t: &Vec3<T>) -> &mut Self {
        for j in 0..4 {
            self.x[3][j] =
                self.x[3][j] + t.x * self.x[0][j] + t.y * self.x[1][j] + t.z * self.x[2][j];
        }
        self
    }

    /// Port of `template <class S> const Matrix44& setShear(const Vec3<S>& h)` —
    /// the 3-factor shear (h.x: x per y, h.y: x per z, h.z: y per z).
    #[inline]
    pub fn set_shear_vec(&mut self, h: &Vec3<T>) -> &mut Self {
        let z = T::ZERO;
        let o = T::ONE;
        self.x = [
            [o, z, z, z],
            [h.x, o, z, z],
            [h.y, h.z, o, z],
            [z, z, z, o],
        ];
        self
    }

    /// Port of `template <class S> const Matrix44& setShear(const Shear6<S>& h)` —
    /// the 6-factor shear.
    #[inline]
    pub fn set_shear(&mut self, h: &Shear6<T>) -> &mut Self {
        let z = T::ZERO;
        let o = T::ONE;
        self.x = [
            [o, h.yx, h.zx, z],
            [h.xy, o, h.zy, z],
            [h.xz, h.yz, o, z],
            [z, z, z, o],
        ];
        self
    }

    /// Port of `template <class S> const Matrix44& shear(const Vec3<S>& h)` — the
    /// 3-factor composing shear (`shear * this`), no temp copy needed.
    #[inline]
    pub fn shear_vec(&mut self, h: &Vec3<T>) -> &mut Self {
        for i in 0..4 {
            self.x[2][i] = self.x[2][i] + h.y * self.x[0][i] + h.z * self.x[1][i];
            self.x[1][i] = self.x[1][i] + h.x * self.x[0][i];
        }
        self
    }

    /// Port of `template <class S> const Matrix44& shear(const Shear6<S>& h)` — the
    /// 6-factor composing shear (`shear * this`).
    #[inline]
    pub fn shear(&mut self, h: &Shear6<T>) -> &mut Self {
        let p = *self;
        for i in 0..4 {
            self.x[0][i] = p.x[0][i] + h.yx * p.x[1][i] + h.zx * p.x[2][i];
            self.x[1][i] = h.xy * p.x[0][i] + p.x[1][i] + h.zy * p.x[2][i];
            self.x[2][i] = h.xz * p.x[0][i] + h.yz * p.x[1][i] + p.x[2][i];
        }
        self
    }
}

// ----------------------------------------------------------------------------
// Numeric limits / dimensions
// ----------------------------------------------------------------------------

impl<T: VecLimits> Matrix44<T> {
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

impl<T> Matrix44<T> {
    /// Port of `static unsigned int dimensions()` — always 4.
    #[inline]
    pub const fn dimensions() -> usize {
        4
    }
}

// ----------------------------------------------------------------------------
// Display
// ----------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Matrix44<T> {
    /// Port of `operator<<` — four space-separated rows in parentheses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} {} {} {}\n {} {} {} {}\n {} {} {} {}\n {} {} {} {})\n",
            self.x[0][0],
            self.x[0][1],
            self.x[0][2],
            self.x[0][3],
            self.x[1][0],
            self.x[1][1],
            self.x[1][2],
            self.x[1][3],
            self.x[2][0],
            self.x[2][1],
            self.x[2][2],
            self.x[2][3],
            self.x[3][0],
            self.x[3][1],
            self.x[3][2],
            self.x[3][3],
        )
    }
}

// ----------------------------------------------------------------------------
// Branch-based abs (mirrors IMATH_INTERNAL_NAMESPACE::abs used by inverse)
// ----------------------------------------------------------------------------

#[inline]
fn abs_t<T: Scalar>(a: T) -> T {
    if a >= T::ZERO { a } else { -a }
}

// ----------------------------------------------------------------------------
// Vector-times-matrix free operators
// ----------------------------------------------------------------------------

/// Port of `operator*=(Vec3<S>& v, const Matrix44<T>& m)` — homogeneous transform
/// (divide by `w`), in place.
#[inline]
pub fn vec3_mul_assign_matrix44<T: Scalar>(v: &mut Vec3<T>, m: &Matrix44<T>) {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + m.x[3][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + m.x[3][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + m.x[3][2];
    let w = v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + m.x[3][3];
    v.x = x / w;
    v.y = y / w;
    v.z = z / w;
}

/// Port of `operator*(const Vec3<S>& v, const Matrix44<T>& m)`.
#[inline]
pub fn vec3_mul_matrix44<T: Scalar>(v: &Vec3<T>, m: &Matrix44<T>) -> Vec3<T> {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + m.x[3][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + m.x[3][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + m.x[3][2];
    let w = v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + m.x[3][3];
    Vec3::new(x / w, y / w, z / w)
}

/// Port of `operator*=(Vec4<S>& v, const Matrix44<T>& m)` — full 4-vector times
/// matrix, in place (no divide).
#[inline]
pub fn vec4_mul_assign_matrix44<T: Scalar>(v: &mut Vec4<T>, m: &Matrix44<T>) {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + v.w * m.x[3][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + v.w * m.x[3][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + v.w * m.x[3][2];
    let w = v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + v.w * m.x[3][3];
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
}

/// Port of `operator*(const Vec4<S>& v, const Matrix44<T>& m)`.
#[inline]
pub fn vec4_mul_matrix44<T: Scalar>(v: &Vec4<T>, m: &Matrix44<T>) -> Vec4<T> {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + v.w * m.x[3][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + v.w * m.x[3][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + v.w * m.x[3][2];
    let w = v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + v.w * m.x[3][3];
    Vec4::new(x, y, z, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        let m = Matrix44::<f64>::default();
        let mut id = Matrix44::from_scalar(0.0);
        for i in 0..4 {
            id.x[i][i] = 1.0;
        }
        assert_eq!(m, id);
    }

    #[test]
    fn matrix_product_identity() {
        let a = Matrix44::new(
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        );
        assert_eq!(a.mul_matrix(&Matrix44::default()), a);
    }

    #[test]
    fn mult_vec_matrix_translation() {
        let mut m = Matrix44::<f64>::default();
        m.set_translation(&Vec3::new(10.0, 20.0, 30.0));
        let mut dst = Vec3::new(0.0, 0.0, 0.0);
        m.mult_vec_matrix(&Vec3::new(1.0, 2.0, 3.0), &mut dst);
        assert!((dst.x - 11.0).abs() < 1e-12);
        assert!((dst.y - 22.0).abs() < 1e-12);
        assert!((dst.z - 33.0).abs() < 1e-12);
    }

    #[test]
    fn transpose_known() {
        let mut m = Matrix44::<f64>::default();
        m.x[0][1] = 5.0;
        m.x[3][0] = 7.0;
        let t = m.transposed();
        assert_eq!(t.x[1][0], 5.0);
        assert_eq!(t.x[0][3], 7.0);
    }

    #[test]
    fn determinant_identity() {
        assert_eq!(Matrix44::<f64>::default().determinant(), 1.0);
    }

    #[test]
    fn inverse_roundtrip_affine() {
        let mut m = Matrix44::<f64>::default();
        m.set_translation(&Vec3::new(3.0, -2.0, 5.0));
        m.x[0][0] = 2.0;
        m.x[1][1] = 0.5;
        m.x[2][2] = -3.0;
        let inv = m.inverse();
        let prod = m.mul_matrix(&inv);
        assert!(prod.equal_with_abs_error(&Matrix44::default(), 1e-12));
    }

    #[test]
    fn inverse_roundtrip_general() {
        // A general (non-affine) matrix forces the gjInverse path.
        let m = Matrix44::new(
            2.0_f64, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0,
        );
        let inv = m.inverse();
        let prod = m.mul_matrix(&inv);
        assert!(prod.equal_with_abs_error(&Matrix44::default(), 1e-10));
    }

    #[test]
    fn gj_inverse_agrees_with_inverse_affine() {
        let mut m = Matrix44::<f64>::default();
        m.set_translation(&Vec3::new(3.0, -2.0, 5.0));
        m.x[0][0] = 2.0;
        m.x[1][1] = 0.5;
        m.x[2][2] = -3.0;
        let a = m.inverse();
        let b = m.gj_inverse();
        assert!(a.equal_with_abs_error(&b, 1e-12));
    }

    #[test]
    fn inverse_singular() {
        let m = Matrix44::from_scalar(0.0_f64);
        assert_eq!(m.inverse(), Matrix44::default());
        assert!(m.inverse_exc().is_err());
        assert!(m.gj_inverse_exc().is_err());
    }

    #[test]
    fn set_euler_angles_known() {
        // 90 deg about Z: setEulerAngles((0,0,pi/2)).
        let mut m = Matrix44::<f64>::default();
        m.set_euler_angles(&Vec3::new(0.0, 0.0, core::f64::consts::FRAC_PI_2));
        // v*m of (1,0,0) -> (0,1,0)
        let r = vec3_mul_matrix44(&Vec3::new(1.0, 0.0, 0.0), &m);
        assert!(r.x.abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
        assert!(r.z.abs() < 1e-12);
    }

    #[test]
    fn set_axis_angle_z_matches_euler_z() {
        let angle = 0.7_f64;
        let mut a = Matrix44::<f64>::default();
        a.set_axis_angle(&Vec3::new(0.0, 0.0, 1.0), angle);
        let mut e = Matrix44::<f64>::default();
        e.set_euler_angles(&Vec3::new(0.0, 0.0, angle));
        assert!(a.equal_with_abs_error(&e, 1e-12));
    }

    #[test]
    fn from_rot_trans() {
        let mut r = Matrix33::<f64>::default();
        r.x[0][0] = 2.0;
        let m = Matrix44::from_rot_trans(r, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.x[0][0], 2.0);
        assert_eq!(m.x[3][0], 1.0);
        assert_eq!(m.x[3][3], 1.0);
        assert_eq!(m.x[0][3], 0.0);
    }

    #[test]
    fn shear6_roundtrip() {
        let mut m = Matrix44::<f64>::default();
        let h = Shear6::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        m.set_shear(&h);
        assert_eq!(m.x[0][1], h.yx);
        assert_eq!(m.x[1][0], h.xy);
        assert_eq!(m.x[2][0], h.xz);
    }

    #[test]
    fn dimensions_is_4() {
        assert_eq!(Matrix44::<f32>::dimensions(), 4);
    }

    #[test]
    fn cross_type_convert() {
        let m = Matrix44::from_scalar(2.5_f64);
        let mf: Matrix44<f32> = Matrix44::from_matrix44(m);
        assert_eq!(mf.x[0][0], 2.5_f32);
    }
}
