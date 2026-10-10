//! Port of Imath ImathMatrix.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathMatrix.h>
//!
//! The [`Matrix33`] 3x3 transformation matrix. See [`crate::matrix`] for the shared
//! design notes (row-major `x: [[T; 3]; 3]` storage, identity-default, the
//! exception-vs-`Result` convention for singular inverse, and the foreign-interop
//! skip).

use core::fmt;
use core::ops::{Index, IndexMut};

use crate::math::{Math, equal_with_abs_error, equal_with_rel_error};
use crate::traits::Scalar;
use crate::vec::{Vec2, Vec3, VecCast, VecLimits};

/// 3x3 transformation matrix (port of C++ `template <class T> class Matrix33`).
///
/// Elements are stored row-major in `x: [[T; 3]; 3]`, byte-for-byte matching the
/// C++ `T x[3][3]`. `#[repr(C)]` guarantees that layout. Row access (`m[i]`) is the
/// safe analogue of the C++ `operator[]` row pointer; `m[i][j]` indexes an element.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix33<T> {
    /// Matrix elements, row-major (`x[row][col]`), mirroring C++ `T x[3][3]`.
    pub x: [[T; 3]; 3],
}

// ----------------------------------------------------------------------------
// Row access — port of `operator[]`
// ----------------------------------------------------------------------------

impl<T> Index<usize> for Matrix33<T> {
    type Output = [T; 3];
    /// Port of `T* operator[](int i)` — returns row `i`.
    #[inline]
    fn index(&self, i: usize) -> &[T; 3] {
        &self.x[i]
    }
}

impl<T> IndexMut<usize> for Matrix33<T> {
    /// Port of `operator[]` (mutable) — returns row `i`.
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut [T; 3] {
        &mut self.x[i]
    }
}

// ----------------------------------------------------------------------------
// Constructors and assignment
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Matrix33<T> {
    /// Port of the default constructor `Matrix33()` — the identity matrix.
    #[inline]
    fn default() -> Self {
        Matrix33 {
            x: [
                [T::ONE, T::ZERO, T::ZERO],
                [T::ZERO, T::ONE, T::ZERO],
                [T::ZERO, T::ZERO, T::ONE],
            ],
        }
    }
}

impl<T: Copy> Matrix33<T> {
    /// Port of `Matrix33(T a)` — every element set to the scalar `a`.
    #[inline]
    pub const fn from_scalar(a: T) -> Self {
        Matrix33 {
            x: [[a, a, a], [a, a, a], [a, a, a]],
        }
    }

    /// Port of `Matrix33(const T a[3][3])` — construct from a 3x3 array.
    #[inline]
    pub const fn from_array(a: [[T; 3]; 3]) -> Self {
        Matrix33 { x: a }
    }

    /// Port of `Matrix33(T a, T b, T c, T d, T e, T f, T g, T h, T i)`:
    ///
    /// ```text
    /// a b c
    /// d e f
    /// g h i
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(a: T, b: T, c: T, d: T, e: T, f: T, g: T, h: T, i: T) -> Self {
        Matrix33 {
            x: [[a, b, c], [d, e, f], [g, h, i]],
        }
    }

    /// Port of `template <class S> explicit Matrix33(const Matrix33<S>& v)` —
    /// cross-base-type element conversion via the C++ `T(v.x[i][j])` cast.
    #[inline]
    pub fn from_matrix33<S: Copy>(v: Matrix33<S>) -> Self
    where
        T: VecCast<S>,
    {
        let mut out = Matrix33::from_scalar(T::from_elem(v.x[0][0]));
        for i in 0..3 {
            for j in 0..3 {
                out.x[i][j] = T::from_elem(v.x[i][j]);
            }
        }
        out
    }
}

// ----------------------------------------------------------------------------
// getValue / setValue (Sb compatibility) — safe, no raw pointers
// ----------------------------------------------------------------------------

impl<T: Copy> Matrix33<T> {
    /// Port of `const T* getValue() const` — flat row-major `[T; 9]`.
    #[inline]
    pub fn to_array(&self) -> [T; 9] {
        [
            self.x[0][0],
            self.x[0][1],
            self.x[0][2],
            self.x[1][0],
            self.x[1][1],
            self.x[1][2],
            self.x[2][0],
            self.x[2][1],
            self.x[2][2],
        ]
    }

    /// Port of `template <class S> void getValue(Matrix33<S>& v) const`.
    #[inline]
    pub fn get_value<S>(&self, v: &mut Matrix33<S>)
    where
        S: Copy + VecCast<T>,
    {
        for i in 0..3 {
            for j in 0..3 {
                v.x[i][j] = S::from_elem(self.x[i][j]);
            }
        }
    }

    /// Port of `template <class S> Matrix33& setValue(const Matrix33<S>& v)`.
    #[inline]
    pub fn set_value<S: Copy>(&mut self, v: &Matrix33<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = T::from_elem(v.x[i][j]);
            }
        }
        self
    }

    /// Port of `template <class S> Matrix33& setTheMatrix(const Matrix33<S>& v)`.
    #[inline]
    pub fn set_the_matrix<S: Copy>(&mut self, v: &Matrix33<S>) -> &mut Self
    where
        T: VecCast<S>,
    {
        self.set_value(v)
    }
}

// ----------------------------------------------------------------------------
// makeIdentity
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix33<T> {
    /// Port of `void makeIdentity()`.
    #[inline]
    pub fn make_identity(&mut self) {
        *self = Matrix33::default();
    }
}

// ----------------------------------------------------------------------------
// Equality
// ----------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Matrix33<T> {
    /// Port of `bool operator==(const Matrix33& v) const`.
    #[inline]
    fn eq(&self, v: &Matrix33<T>) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if self.x[i][j] != v.x[i][j] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Scalar> Matrix33<T> {
    /// Port of `bool equalWithAbsError(const Matrix33<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_abs_error(&self, m: &Matrix33<T>, e: T) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if !equal_with_abs_error(self.x[i][j], m.x[i][j], e) {
                    return false;
                }
            }
        }
        true
    }

    /// Port of `bool equalWithRelError(const Matrix33<T>& m, T e) const`.
    #[inline]
    pub fn equal_with_rel_error(&self, m: &Matrix33<T>, e: T) -> bool {
        for i in 0..3 {
            for j in 0..3 {
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

impl<T: Scalar> Matrix33<T> {
    /// Port of `const Matrix33& operator+=(const Matrix33& v)`.
    #[inline]
    pub fn add_assign_matrix(&mut self, v: &Matrix33<T>) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] + v.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix33& operator+=(T a)`.
    #[inline]
    pub fn add_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] + a;
            }
        }
        self
    }

    /// Port of `Matrix33 operator+(const Matrix33& v) const`.
    #[inline]
    pub fn add_matrix(&self, v: &Matrix33<T>) -> Matrix33<T> {
        let mut out = *self;
        out.add_assign_matrix(v);
        out
    }

    /// Port of `const Matrix33& operator-=(const Matrix33& v)`.
    #[inline]
    pub fn sub_assign_matrix(&mut self, v: &Matrix33<T>) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] - v.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix33& operator-=(T a)`.
    #[inline]
    pub fn sub_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] - a;
            }
        }
        self
    }

    /// Port of `Matrix33 operator-(const Matrix33& v) const`.
    #[inline]
    pub fn sub_matrix(&self, v: &Matrix33<T>) -> Matrix33<T> {
        let mut out = *self;
        out.sub_assign_matrix(v);
        out
    }

    /// Port of `Matrix33 operator-() const` — component-wise negation.
    #[inline]
    pub fn negated(&self) -> Matrix33<T> {
        let mut out = *self;
        out.negate();
        out
    }

    /// Port of `const Matrix33& negate()` — negate in place.
    #[inline]
    pub fn negate(&mut self) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = -self.x[i][j];
            }
        }
        self
    }

    /// Port of `const Matrix33& operator*=(T a)` — scalar multiply in place.
    #[inline]
    pub fn mul_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] * a;
            }
        }
        self
    }

    /// Port of `Matrix33 operator*(T a) const` — scalar multiply.
    #[inline]
    pub fn mul_scalar(&self, a: T) -> Matrix33<T> {
        let mut out = *self;
        out.mul_assign_scalar(a);
        out
    }

    /// Port of `const Matrix33& operator/=(T a)` — scalar divide in place.
    #[inline]
    pub fn div_assign_scalar(&mut self, a: T) -> &mut Self {
        for i in 0..3 {
            for j in 0..3 {
                self.x[i][j] = self.x[i][j] / a;
            }
        }
        self
    }

    /// Port of `Matrix33 operator/(T a) const` — scalar divide.
    #[inline]
    pub fn div_scalar(&self, a: T) -> Matrix33<T> {
        let mut out = *self;
        out.div_assign_scalar(a);
        out
    }

    /// Port of `Matrix33 operator*(const Matrix33& v) const` — matrix product.
    ///
    /// Fully unrolled, exactly as the C++ (which unrolls for autovectorization).
    #[inline]
    pub fn mul_matrix(&self, v: &Matrix33<T>) -> Matrix33<T> {
        let x = &self.x;
        Matrix33::new(
            x[0][0] * v.x[0][0] + x[0][1] * v.x[1][0] + x[0][2] * v.x[2][0],
            x[0][0] * v.x[0][1] + x[0][1] * v.x[1][1] + x[0][2] * v.x[2][1],
            x[0][0] * v.x[0][2] + x[0][1] * v.x[1][2] + x[0][2] * v.x[2][2],
            x[1][0] * v.x[0][0] + x[1][1] * v.x[1][0] + x[1][2] * v.x[2][0],
            x[1][0] * v.x[0][1] + x[1][1] * v.x[1][1] + x[1][2] * v.x[2][1],
            x[1][0] * v.x[0][2] + x[1][1] * v.x[1][2] + x[1][2] * v.x[2][2],
            x[2][0] * v.x[0][0] + x[2][1] * v.x[1][0] + x[2][2] * v.x[2][0],
            x[2][0] * v.x[0][1] + x[2][1] * v.x[1][1] + x[2][2] * v.x[2][1],
            x[2][0] * v.x[0][2] + x[2][1] * v.x[1][2] + x[2][2] * v.x[2][2],
        )
    }

    /// Port of `const Matrix33& operator*=(const Matrix33& v)` — in place.
    #[inline]
    pub fn mul_assign_matrix(&mut self, v: &Matrix33<T>) -> &mut Self {
        *self = self.mul_matrix(v);
        self
    }
}

// ----------------------------------------------------------------------------
// Vector * matrix multiplication helpers
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix33<T> {
    /// Port of `template <class S> void multVecMatrix(const Vec2<S>& src,
    /// Vec2<S>& dst) const` — homogeneous transform with the divide by `w`.
    #[inline]
    pub fn mult_vec_matrix(&self, src: &Vec2<T>, dst: &mut Vec2<T>) {
        let a = src.x * self.x[0][0] + src.y * self.x[1][0] + self.x[2][0];
        let b = src.x * self.x[0][1] + src.y * self.x[1][1] + self.x[2][1];
        let w = src.x * self.x[0][2] + src.y * self.x[1][2] + self.x[2][2];
        dst.x = a / w;
        dst.y = b / w;
    }

    /// Port of `template <class S> void multDirMatrix(const Vec2<S>& src,
    /// Vec2<S>& dst) const` — upper-left 2x2 only (no translation, no divide).
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

impl<T: Scalar> Matrix33<T> {
    /// Port of `const Matrix33& transpose()` — transpose in place.
    #[inline]
    pub fn transpose(&mut self) -> &mut Self {
        *self = self.transposed();
        self
    }

    /// Port of `Matrix33 transposed() const`.
    #[inline]
    pub fn transposed(&self) -> Matrix33<T> {
        Matrix33::new(
            self.x[0][0],
            self.x[1][0],
            self.x[2][0],
            self.x[0][1],
            self.x[1][1],
            self.x[2][1],
            self.x[0][2],
            self.x[1][2],
            self.x[2][2],
        )
    }
}

// ----------------------------------------------------------------------------
// Gauss-Jordan inverse (gjInverse / gjInvert)
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix33<T> {
    /// Shared Gauss-Jordan elimination with partial pivoting. Returns `Err` on a
    /// singular matrix; the public wrappers map that to either the identity matrix
    /// (non-throwing C++ form) or a `Result` (the `singExc` form). The algorithm —
    /// pivot selection, row swaps, forward elimination, backward substitution — is
    /// 1:1 with the C++ `gjInverse`.
    fn gj_inverse_impl(&self) -> Result<Matrix33<T>, ()> {
        let mut s = Matrix33::<T>::default();
        let mut t = *self;

        // Forward elimination
        for i in 0..2 {
            let mut pivot = i;
            let mut pivotsize = t.x[i][i];
            if pivotsize < T::ZERO {
                pivotsize = -pivotsize;
            }

            for j in (i + 1)..3 {
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
                for j in 0..3 {
                    let tmp = t.x[i][j];
                    t.x[i][j] = t.x[pivot][j];
                    t.x[pivot][j] = tmp;

                    let tmp = s.x[i][j];
                    s.x[i][j] = s.x[pivot][j];
                    s.x[pivot][j] = tmp;
                }
            }

            for j in (i + 1)..3 {
                let f = t.x[j][i] / t.x[i][i];
                for k in 0..3 {
                    t.x[j][k] = t.x[j][k] - f * t.x[i][k];
                    s.x[j][k] = s.x[j][k] - f * s.x[i][k];
                }
            }
        }

        // Backward substitution
        for i in (0..3).rev() {
            let f = t.x[i][i];
            if f == T::ZERO {
                return Err(());
            }

            for j in 0..3 {
                t.x[i][j] = t.x[i][j] / f;
                s.x[i][j] = s.x[i][j] / f;
            }

            for j in 0..i {
                let f = t.x[j][i];
                for k in 0..3 {
                    t.x[j][k] = t.x[j][k] - f * t.x[i][k];
                    s.x[j][k] = s.x[j][k] - f * s.x[i][k];
                }
            }
        }

        Ok(s)
    }

    /// Port of the throwing `Matrix33<T> gjInverse(bool singExc) const`
    /// (`singExc == true`). Returns `Err` for a singular matrix.
    #[inline]
    pub fn gj_inverse_exc(&self) -> Result<Matrix33<T>, &'static str> {
        self.gj_inverse_impl()
            .map_err(|_| "Cannot invert singular matrix.")
    }

    /// Port of the non-throwing `Matrix33<T> gjInverse() const` — identity on
    /// singular.
    #[inline]
    pub fn gj_inverse(&self) -> Matrix33<T> {
        self.gj_inverse_impl().unwrap_or_default()
    }

    /// Port of the throwing `const Matrix33& gjInvert(bool singExc)`.
    #[inline]
    pub fn gj_invert_exc(&mut self) -> Result<&mut Self, &'static str> {
        *self = self.gj_inverse_exc()?;
        Ok(self)
    }

    /// Port of the non-throwing `const Matrix33& gjInvert()`.
    #[inline]
    pub fn gj_invert(&mut self) -> &mut Self {
        *self = self.gj_inverse();
        self
    }
}

// ----------------------------------------------------------------------------
// Adjoint/determinant inverse (inverse / invert) — the fast path
// ----------------------------------------------------------------------------

impl<T: Scalar + VecLimits> Matrix33<T> {
    /// Shared implementation of the determinant-based `inverse`. The `sing_exc`
    /// flag selects whether a singular matrix returns `Err` (true) or the identity
    /// (false). Mirrors the two C++ overloads, including the affine fast path
    /// (`x[*][2] == identity column`) and the cofactor-then-scale-by-`r` logic.
    fn inverse_impl(&self, sing_exc: bool) -> Result<Matrix33<T>, &'static str> {
        // The affine-vs-general test mirrors the C++ exactly.
        if self.x[0][2] != T::ZERO || self.x[1][2] != T::ZERO || self.x[2][2] != T::ONE {
            let mut s = Matrix33::new(
                self.x[1][1] * self.x[2][2] - self.x[2][1] * self.x[1][2],
                self.x[2][1] * self.x[0][2] - self.x[0][1] * self.x[2][2],
                self.x[0][1] * self.x[1][2] - self.x[1][1] * self.x[0][2],
                self.x[2][0] * self.x[1][2] - self.x[1][0] * self.x[2][2],
                self.x[0][0] * self.x[2][2] - self.x[2][0] * self.x[0][2],
                self.x[1][0] * self.x[0][2] - self.x[0][0] * self.x[1][2],
                self.x[1][0] * self.x[2][1] - self.x[2][0] * self.x[1][1],
                self.x[2][0] * self.x[0][1] - self.x[0][0] * self.x[2][1],
                self.x[0][0] * self.x[1][1] - self.x[1][0] * self.x[0][1],
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
                            return Ok(Matrix33::default());
                        }
                    }
                }
            }

            Ok(s)
        } else {
            // Affine (last column is [0,0,1]) fast path.
            let mut s = Matrix33::new(
                self.x[1][1],
                -self.x[0][1],
                T::ZERO,
                -self.x[1][0],
                self.x[0][0],
                T::ZERO,
                T::ZERO,
                T::ZERO,
                T::ONE,
            );

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
                        } else if sing_exc {
                            return Err("Cannot invert singular matrix.");
                        } else {
                            return Ok(Matrix33::default());
                        }
                    }
                }
            }

            s.x[2][0] = -self.x[2][0] * s.x[0][0] - self.x[2][1] * s.x[1][0];
            s.x[2][1] = -self.x[2][0] * s.x[0][1] - self.x[2][1] * s.x[1][1];

            Ok(s)
        }
    }

    /// Port of the throwing `Matrix33<T> inverse(bool singExc) const`
    /// (`singExc == true`).
    #[inline]
    pub fn inverse_exc(&self) -> Result<Matrix33<T>, &'static str> {
        self.inverse_impl(true)
    }

    /// Port of the non-throwing `Matrix33<T> inverse() const` — identity on
    /// singular.
    #[inline]
    pub fn inverse(&self) -> Matrix33<T> {
        // `inverse_impl(false)` never returns Err.
        self.inverse_impl(false).unwrap_or_default()
    }

    /// Port of the throwing `const Matrix33& invert(bool singExc)`.
    #[inline]
    pub fn invert_exc(&mut self) -> Result<&mut Self, &'static str> {
        *self = self.inverse_exc()?;
        Ok(self)
    }

    /// Port of the non-throwing `const Matrix33& invert()`.
    #[inline]
    pub fn invert(&mut self) -> &mut Self {
        *self = self.inverse();
        self
    }
}

// ----------------------------------------------------------------------------
// minorOf / fastMinor / determinant / trace
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix33<T> {
    /// Port of `T minorOf(const int r, const int c) const`.
    #[inline]
    pub fn minor_of(&self, r: usize, c: usize) -> T {
        let r0 = if r < 1 { 1 } else { 0 };
        let r1 = 1 + if r < 2 { 1 } else { 0 };
        let c0 = if c < 1 { 1 } else { 0 };
        let c1 = 1 + if c < 2 { 1 } else { 0 };
        self.x[r0][c0] * self.x[r1][c1] - self.x[r1][c0] * self.x[r0][c1]
    }

    /// Port of `T fastMinor(const int r0, const int r1, const int c0, const int c1) const`.
    #[inline]
    pub fn fast_minor(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> T {
        self.x[r0][c0] * self.x[r1][c1] - self.x[r0][c1] * self.x[r1][c0]
    }

    /// Port of `T determinant() const`.
    #[inline]
    pub fn determinant(&self) -> T {
        self.x[0][0] * (self.x[1][1] * self.x[2][2] - self.x[1][2] * self.x[2][1])
            + self.x[0][1] * (self.x[1][2] * self.x[2][0] - self.x[1][0] * self.x[2][2])
            + self.x[0][2] * (self.x[1][0] * self.x[2][1] - self.x[1][1] * self.x[2][0])
    }

    /// Port of `T trace() const`.
    #[inline]
    pub fn trace(&self) -> T {
        self.x[0][0] + self.x[1][1] + self.x[2][2]
    }
}

// ----------------------------------------------------------------------------
// Rotation (transcendental)
// ----------------------------------------------------------------------------

impl<T: Scalar + Math> Matrix33<T> {
    /// Port of `template <class S> const Matrix33& setRotation(S r)` — rotation
    /// about (0,0,1) by `r` radians.
    #[inline]
    pub fn set_rotation(&mut self, r: T) -> &mut Self {
        let cos_r = r.cos();
        let sin_r = r.sin();

        self.x[0][0] = cos_r;
        self.x[0][1] = sin_r;
        self.x[0][2] = T::ZERO;
        self.x[1][0] = -sin_r;
        self.x[1][1] = cos_r;
        self.x[1][2] = T::ZERO;
        self.x[2][0] = T::ZERO;
        self.x[2][1] = T::ZERO;
        self.x[2][2] = T::ONE;
        self
    }

    /// Port of `template <class S> const Matrix33& rotate(S r)`.
    #[inline]
    pub fn rotate(&mut self, r: T) -> &mut Self {
        let mut rot = Matrix33::<T>::default();
        rot.set_rotation(r);
        *self = self.mul_matrix(&rot);
        self
    }
}

// ----------------------------------------------------------------------------
// Scale / translation / shear
// ----------------------------------------------------------------------------

impl<T: Scalar> Matrix33<T> {
    /// Port of `const Matrix33& setScale(T s)` — uniform scale (homogeneous 2D).
    #[inline]
    pub fn set_scale_scalar(&mut self, s: T) -> &mut Self {
        self.x = [
            [s, T::ZERO, T::ZERO],
            [T::ZERO, s, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix33& setScale(const Vec2<S>& s)`.
    #[inline]
    pub fn set_scale(&mut self, s: &Vec2<T>) -> &mut Self {
        self.x = [
            [s.x, T::ZERO, T::ZERO],
            [T::ZERO, s.y, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix33& scale(const Vec2<S>& s)`.
    #[inline]
    pub fn scale(&mut self, s: &Vec2<T>) -> &mut Self {
        self.x[0][0] = self.x[0][0] * s.x;
        self.x[0][1] = self.x[0][1] * s.x;
        self.x[0][2] = self.x[0][2] * s.x;
        self.x[1][0] = self.x[1][0] * s.y;
        self.x[1][1] = self.x[1][1] * s.y;
        self.x[1][2] = self.x[1][2] * s.y;
        self
    }

    /// Port of `template <class S> const Matrix33& setTranslation(const Vec2<S>& t)`.
    #[inline]
    pub fn set_translation(&mut self, t: &Vec2<T>) -> &mut Self {
        self.x = [
            [T::ONE, T::ZERO, T::ZERO],
            [T::ZERO, T::ONE, T::ZERO],
            [t.x, t.y, T::ONE],
        ];
        self
    }

    /// Port of `Vec2<T> translation() const`.
    #[inline]
    pub fn translation(&self) -> Vec2<T> {
        Vec2::new(self.x[2][0], self.x[2][1])
    }

    /// Port of `template <class S> const Matrix33& translate(const Vec2<S>& t)`.
    #[inline]
    pub fn translate(&mut self, t: &Vec2<T>) -> &mut Self {
        self.x[2][0] = self.x[2][0] + t.x * self.x[0][0] + t.y * self.x[1][0];
        self.x[2][1] = self.x[2][1] + t.x * self.x[0][1] + t.y * self.x[1][1];
        self.x[2][2] = self.x[2][2] + t.x * self.x[0][2] + t.y * self.x[1][2];
        self
    }

    /// Port of `template <class S> const Matrix33& setShear(const S& xy)` — the
    /// scalar overload: shear x for each y coord by factor `xy`.
    #[inline]
    pub fn set_shear_scalar(&mut self, xy: T) -> &mut Self {
        self.x = [
            [T::ONE, T::ZERO, T::ZERO],
            [xy, T::ONE, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix33& setShear(const Vec2<S>& h)` —
    /// shear x by `h.y` per y and y by `h.x` per x.
    #[inline]
    pub fn set_shear(&mut self, h: &Vec2<T>) -> &mut Self {
        self.x = [
            [T::ONE, h.y, T::ZERO],
            [h.x, T::ONE, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE],
        ];
        self
    }

    /// Port of `template <class S> const Matrix33& shear(const S& xy)` — the scalar
    /// overload (no temp copy needed, matching the C++ note).
    #[inline]
    pub fn shear_scalar(&mut self, xy: T) -> &mut Self {
        self.x[1][0] = self.x[1][0] + xy * self.x[0][0];
        self.x[1][1] = self.x[1][1] + xy * self.x[0][1];
        self.x[1][2] = self.x[1][2] + xy * self.x[0][2];
        self
    }

    /// Port of `template <class S> const Matrix33& shear(const Vec2<S>& h)`.
    #[inline]
    pub fn shear(&mut self, h: &Vec2<T>) -> &mut Self {
        let p = *self;
        self.x[0][0] = p.x[0][0] + h.y * p.x[1][0];
        self.x[0][1] = p.x[0][1] + h.y * p.x[1][1];
        self.x[0][2] = p.x[0][2] + h.y * p.x[1][2];
        self.x[1][0] = p.x[1][0] + h.x * p.x[0][0];
        self.x[1][1] = p.x[1][1] + h.x * p.x[0][1];
        self.x[1][2] = p.x[1][2] + h.x * p.x[0][2];
        self
    }
}

// ----------------------------------------------------------------------------
// Numeric limits / dimensions
// ----------------------------------------------------------------------------

impl<T: VecLimits> Matrix33<T> {
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

impl<T> Matrix33<T> {
    /// Port of `static unsigned int dimensions()` — always 3.
    #[inline]
    pub const fn dimensions() -> usize {
        3
    }
}

// ----------------------------------------------------------------------------
// Display
// ----------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Matrix33<T> {
    /// Port of `operator<<` — three space-separated rows in parentheses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} {} {}\n {} {} {}\n {} {} {})\n",
            self.x[0][0],
            self.x[0][1],
            self.x[0][2],
            self.x[1][0],
            self.x[1][1],
            self.x[1][2],
            self.x[2][0],
            self.x[2][1],
            self.x[2][2],
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

/// Port of `operator*=(Vec2<S>& v, const Matrix33<T>& m)` — homogeneous 2D
/// row-vector times matrix, in place, with divide by `w`.
#[inline]
pub fn vec2_mul_assign_matrix33<T: Scalar>(v: &mut Vec2<T>, m: &Matrix33<T>) {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + m.x[2][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + m.x[2][1];
    let w = v.x * m.x[0][2] + v.y * m.x[1][2] + m.x[2][2];
    v.x = x / w;
    v.y = y / w;
}

/// Port of `operator*(const Vec2<S>& v, const Matrix33<T>& m)`.
#[inline]
pub fn vec2_mul_matrix33<T: Scalar>(v: &Vec2<T>, m: &Matrix33<T>) -> Vec2<T> {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + m.x[2][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + m.x[2][1];
    let w = v.x * m.x[0][2] + v.y * m.x[1][2] + m.x[2][2];
    Vec2::new(x / w, y / w)
}

/// Port of `operator*=(Vec3<S>& v, const Matrix33<T>& m)` — full 3-vector times
/// matrix, in place.
#[inline]
pub fn vec3_mul_assign_matrix33<T: Scalar>(v: &mut Vec3<T>, m: &Matrix33<T>) {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2];
    v.x = x;
    v.y = y;
    v.z = z;
}

/// Port of `operator*(const Vec3<S>& v, const Matrix33<T>& m)`.
#[inline]
pub fn vec3_mul_matrix33<T: Scalar>(v: &Vec3<T>, m: &Matrix33<T>) -> Vec3<T> {
    let x = v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0];
    let y = v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1];
    let z = v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2];
    Vec3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        let m = Matrix33::<f64>::default();
        assert_eq!(m, Matrix33::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn matrix_product_known() {
        let a = Matrix33::new(1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0);
        let id = Matrix33::default();
        assert_eq!(a.mul_matrix(&id), a);
    }

    #[test]
    fn determinant_known() {
        let m = Matrix33::new(6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0);
        assert_eq!(m.determinant(), -306.0);
    }

    #[test]
    fn inverse_roundtrip_general() {
        let m = Matrix33::new(6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0);
        let inv = m.inverse();
        let prod = m.mul_matrix(&inv);
        assert!(prod.equal_with_abs_error(&Matrix33::default(), 1e-10));
    }

    #[test]
    fn inverse_roundtrip_affine() {
        // Affine matrix (last column [0,0,1]): exercises the fast path.
        let mut m = Matrix33::<f64>::default();
        m.set_translation(&Vec2::new(3.0, -2.0));
        m.x[0][0] = 2.0;
        m.x[1][1] = 0.5;
        let inv = m.inverse();
        let prod = m.mul_matrix(&inv);
        assert!(prod.equal_with_abs_error(&Matrix33::default(), 1e-12));
    }

    #[test]
    fn gj_inverse_agrees_with_inverse() {
        let m = Matrix33::new(6.0_f64, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0);
        let a = m.inverse();
        let b = m.gj_inverse();
        assert!(a.equal_with_abs_error(&b, 1e-10));
    }

    #[test]
    fn inverse_singular() {
        let m = Matrix33::from_scalar(0.0_f64);
        assert_eq!(m.inverse(), Matrix33::default());
        assert!(m.inverse_exc().is_err());
        assert_eq!(m.gj_inverse(), Matrix33::default());
        assert!(m.gj_inverse_exc().is_err());
    }

    #[test]
    fn mult_vec_matrix_translation() {
        // Homogeneous 2D translation via Matrix33.
        let mut m = Matrix33::<f64>::default();
        m.set_translation(&Vec2::new(5.0, 7.0));
        let mut dst = Vec2::new(0.0, 0.0);
        m.mult_vec_matrix(&Vec2::new(1.0, 2.0), &mut dst);
        assert!((dst.x - 6.0).abs() < 1e-12);
        assert!((dst.y - 9.0).abs() < 1e-12);
    }

    #[test]
    fn set_rotation_known() {
        let mut m = Matrix33::<f64>::default();
        m.set_rotation(core::f64::consts::FRAC_PI_2);
        let r = vec2_mul_matrix33(&Vec2::new(1.0, 0.0), &m);
        assert!(r.x.abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn transpose_known() {
        let m = Matrix33::new(1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        assert_eq!(
            m.transposed(),
            Matrix33::new(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0)
        );
    }

    #[test]
    fn minor_and_trace() {
        let m = Matrix33::new(1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        // minorOf(0,0) = det of [[5,6],[8,9]] = 45-48 = -3
        assert_eq!(m.minor_of(0, 0), -3.0);
        assert_eq!(m.fast_minor(1, 2, 1, 2), 5.0 * 9.0 - 6.0 * 8.0);
        assert_eq!(m.trace(), 15.0);
    }

    #[test]
    fn dimensions_is_3() {
        assert_eq!(Matrix33::<f32>::dimensions(), 3);
    }
}
