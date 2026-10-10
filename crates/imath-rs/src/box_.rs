//! Port of Imath ImathBox.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathBox.h>
//!
//! Axis-aligned bounding box generic over a vector type `V`.
//!
//! # Module name
//! `box` is a reserved keyword in Rust, so this module is named `box_`.
//! The public type and typedefs are re-exported from `crate` as `Box2f` etc.
//! Note that `imath_rs::Box` shadows `std::boxed::Box` — users who need both
//! in scope should `use imath_rs::box_::Box as BBox;` or similar.
//!
//! # C++ specialisations
//! The C++ header has a generic `Box<V>` plus explicit partial specialisations for
//! `Box<Vec2<T>>` and `Box<Vec3<T>>`.  All three encode the same algorithm; the
//! specialisations merely inline the per-component comparisons.  In Rust the single
//! generic `impl` block already inlines to identical code, so one implementation
//! serves all instantiations.

use core::fmt;

use crate::vec::{Vec2, Vec3, VecLimits};
use crate::traits::Scalar;

// ---------------------------------------------------------------------------
// Trait bound for a Box vector type
// ---------------------------------------------------------------------------

/// Element-type bound for [`Box`]'s vector parameter `V`.
///
/// `V` must expose the `base_type_lowest()` / `base_type_max()` static
/// helpers (from [`VecLimits`] via the vector `impl`) and scalar arithmetic /
/// comparison so the `makeEmpty` / `makeInfinite` initialisers work.
///
/// This trait is sealed (no external implementations needed) — it is
/// implemented for every concrete `Vec2<T>` and `Vec3<T>` where `T: Scalar +
/// VecLimits`.
pub trait BoxVector: Copy + PartialEq {
    /// The scalar element type (C++ `BaseType`).
    type BaseType: Scalar + VecLimits;

    /// Number of dimensions (C++ `V::dimensions()`).
    fn dimensions() -> usize;

    /// Per-element access (C++ `operator[]`).
    fn get(&self, i: usize) -> Self::BaseType;
    /// Per-element mutable access.
    fn set(&mut self, i: usize, v: Self::BaseType);

    /// Construct from a scalar repeated in every component (C++ `V(scalar)`).
    fn from_scalar(s: Self::BaseType) -> Self;

    /// Addition (C++ `operator+`).
    fn vadd(self, other: Self) -> Self;
    /// Subtraction (C++ `operator-`).
    fn vsub(self, other: Self) -> Self;
    /// Division by scalar 2 (C++ `operator/(T(2))`).
    fn vdiv2(self) -> Self;
}

// ---------------------------------------------------------------------------
// BoxVector impls for Vec2<T> and Vec3<T>
// ---------------------------------------------------------------------------

impl<T: Scalar + VecLimits> BoxVector for Vec2<T> {
    type BaseType = T;

    #[inline] fn dimensions() -> usize { 2 }

    #[inline] fn get(&self, i: usize) -> T { self[i] }
    #[inline] fn set(&mut self, i: usize, v: T) { self[i] = v; }

    #[inline] fn from_scalar(s: T) -> Self { Vec2::from_scalar(s) }

    #[inline] fn vadd(self, o: Self) -> Self { self + o }
    #[inline] fn vsub(self, o: Self) -> Self { self - o }
    #[inline]
    fn vdiv2(self) -> Self {
        // T(2) — construct from ONE + ONE
        let two = T::ONE + T::ONE;
        Vec2::new(self.x / two, self.y / two)
    }
}

impl<T: Scalar + VecLimits> BoxVector for Vec3<T> {
    type BaseType = T;

    #[inline] fn dimensions() -> usize { 3 }

    #[inline] fn get(&self, i: usize) -> T { self[i] }
    #[inline] fn set(&mut self, i: usize, v: T) { self[i] = v; }

    #[inline] fn from_scalar(s: T) -> Self { Vec3::from_scalar(s) }

    #[inline] fn vadd(self, o: Self) -> Self { self + o }
    #[inline] fn vsub(self, o: Self) -> Self { self - o }
    #[inline]
    fn vdiv2(self) -> Self {
        let two = T::ONE + T::ONE;
        Vec3::new(self.x / two, self.y / two, self.z / two)
    }
}

// ---------------------------------------------------------------------------
// Box<V>
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box (port of C++ `template <class V> class Box`).
///
/// `V` is typically `Vec2<T>` or `Vec3<T>`.  The `min` and `max` members are
/// public, matching C++.
///
/// An empty box has `min = V::baseTypeMax()` and `max = V::baseTypeLowest()`
/// (any component of max < min).  An infinite box has `min =
/// V::baseTypeLowest()` and `max = V::baseTypeMax()`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Box<V: BoxVector> {
    /// The minimum corner of the box.
    pub min: V,
    /// The maximum corner of the box.
    pub max: V,
}

impl<V: BoxVector> Box<V> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of `Box()` — construct an empty bounding box.
    ///
    /// Sets `min` to `V::baseTypeMax()` and `max` to `V::baseTypeLowest()`
    /// so the box is empty.
    #[inline]
    pub fn new() -> Self {
        let mut b = Box { min: V::from_scalar(V::BaseType::ZERO), max: V::from_scalar(V::BaseType::ZERO) };
        b.make_empty();
        b
    }

    /// Port of `Box(const V& point)` — single-point box.
    #[inline]
    pub fn from_point(point: V) -> Self {
        Box { min: point, max: point }
    }

    /// Port of `Box(const V& minV, const V& maxV)` — box from min/max corners.
    #[inline]
    pub fn from_min_max(min: V, max: V) -> Self {
        Box { min, max }
    }

    // -----------------------------------------------------------------------
    // Manipulation
    // -----------------------------------------------------------------------

    /// Port of `makeEmpty()` — set min to `baseTypeMax`, max to `baseTypeLowest`.
    #[inline]
    pub fn make_empty(&mut self) {
        self.min = V::from_scalar(V::BaseType::MAX);
        self.max = V::from_scalar(V::BaseType::LOWEST);
    }

    /// Port of `makeInfinite()` — set min to `baseTypeLowest`, max to `baseTypeMax`.
    #[inline]
    pub fn make_infinite(&mut self) {
        self.min = V::from_scalar(V::BaseType::LOWEST);
        self.max = V::from_scalar(V::BaseType::MAX);
    }

    /// Port of `extendBy(const V& point)` — expand box to include a point.
    #[inline]
    pub fn extend_by_point(&mut self, point: V) {
        for i in 0..V::dimensions() {
            let p = point.get(i);
            if p < self.min.get(i) { self.min.set(i, p); }
            if p > self.max.get(i) { self.max.set(i, p); }
        }
    }

    /// Port of `extendBy(const Box<V>& box)` — expand box to include another box.
    #[inline]
    pub fn extend_by_box(&mut self, other: &Box<V>) {
        for i in 0..V::dimensions() {
            if other.min.get(i) < self.min.get(i) { self.min.set(i, other.min.get(i)); }
            if other.max.get(i) > self.max.get(i) { self.max.set(i, other.max.get(i)); }
        }
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Port of `size()` — `(max - min)`, or zero vector if empty.
    #[inline]
    pub fn size(&self) -> V {
        if self.is_empty() {
            return V::from_scalar(V::BaseType::ZERO);
        }
        self.max.vsub(self.min)
    }

    /// Port of `center()` — `(max + min) / 2`.
    #[inline]
    pub fn center(&self) -> V {
        self.max.vadd(self.min).vdiv2()
    }

    /// Port of `intersects(const V& point)` — true if point is inside the box.
    #[inline]
    pub fn intersects_point(&self, point: V) -> bool {
        for i in 0..V::dimensions() {
            let p = point.get(i);
            if p < self.min.get(i) || p > self.max.get(i) { return false; }
        }
        true
    }

    /// Port of `intersects(const Box<V>& box)` — true if the boxes overlap.
    #[inline]
    pub fn intersects_box(&self, other: &Box<V>) -> bool {
        for i in 0..V::dimensions() {
            if other.max.get(i) < self.min.get(i) || other.min.get(i) > self.max.get(i) {
                return false;
            }
        }
        true
    }

    /// Port of `majorAxis()` — index of the dimension with the greatest extent.
    #[inline]
    pub fn major_axis(&self) -> usize {
        let s = self.size();
        let mut major = 0usize;
        for i in 1..V::dimensions() {
            if s.get(i) > s.get(major) { major = i; }
        }
        major
    }

    /// Port of `isEmpty()` — true if any component of max < min.
    #[inline]
    pub fn is_empty(&self) -> bool {
        for i in 0..V::dimensions() {
            if self.max.get(i) < self.min.get(i) { return true; }
        }
        false
    }

    /// Port of `hasVolume()` — true if every component of max > min.
    #[inline]
    pub fn has_volume(&self) -> bool {
        for i in 0..V::dimensions() {
            if self.max.get(i) <= self.min.get(i) { return false; }
        }
        true
    }

    /// Port of `isInfinite()` — true if min == baseTypeLowest && max == baseTypeMax.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        for i in 0..V::dimensions() {
            if self.min.get(i) != V::BaseType::LOWEST || self.max.get(i) != V::BaseType::MAX {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<V: BoxVector> Default for Box<V> {
    /// Port of `Box()` — empty box.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equality operators (port of `operator==` / `operator!=`)
// ---------------------------------------------------------------------------

impl<V: BoxVector + PartialEq> PartialEq for Box<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}

impl<V: BoxVector + PartialEq> Eq for Box<V> where V: Eq {}

// ---------------------------------------------------------------------------
// Display (matches C++ stream output `(min max)` implied by Imath convention)
// ---------------------------------------------------------------------------

impl<V: BoxVector + fmt::Display> fmt::Display for Box<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.min, self.max)
    }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathBox.h)
// ---------------------------------------------------------------------------

use crate::vec::{V2s, V2i, V2i64, V2h, V2f, V2d, V3s, V3i, V3i64, V3h, V3f, V3d};

/// 2D box of base type `i16` (C++ `Box2s`).
pub type Box2s   = Box<V2s>;
/// 2D box of base type `i32` (C++ `Box2i`).
pub type Box2i   = Box<V2i>;
/// 2D box of base type `i64` (C++ `Box2i64`).
pub type Box2i64 = Box<V2i64>;
/// 2D box of base type [`Half`] (C++ `Box2h`).
pub type Box2h   = Box<V2h>;
/// 2D box of base type `f32` (C++ `Box2f`).
pub type Box2f   = Box<V2f>;
/// 2D box of base type `f64` (C++ `Box2d`).
pub type Box2d   = Box<V2d>;
/// 3D box of base type `i16` (C++ `Box3s`).
pub type Box3s   = Box<V3s>;
/// 3D box of base type `i32` (C++ `Box3i`).
pub type Box3i   = Box<V3i>;
/// 3D box of base type `i64` (C++ `Box3i64`).
pub type Box3i64 = Box<V3i64>;
/// 3D box of base type [`Half`] (C++ `Box3h`).
pub type Box3h   = Box<V3h>;
/// 3D box of base type `f32` (C++ `Box3f`).
pub type Box3f   = Box<V3f>;
/// 3D box of base type `f64` (C++ `Box3d`).
pub type Box3d   = Box<V3d>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::{Vec2, Vec3};

    #[test]
    fn box3f_empty_default() {
        let b = Box3f::new();
        assert!(b.is_empty());
        assert!(!b.has_volume());
        assert!(!b.is_infinite());
    }

    #[test]
    fn box3f_extend_by_point() {
        let mut b = Box3f::new();
        b.extend_by_point(Vec3::new(1.0f32, 2.0, 3.0));
        b.extend_by_point(Vec3::new(-1.0f32, 0.0, 5.0));
        assert!(!b.is_empty());
        assert_eq!(b.min, Vec3::new(-1.0f32, 0.0, 3.0));
        assert_eq!(b.max, Vec3::new(1.0f32, 2.0, 5.0));
        assert_eq!(b.size(), Vec3::new(2.0f32, 2.0, 2.0));
        assert_eq!(b.center(), Vec3::new(0.0f32, 1.0, 4.0));
    }

    #[test]
    fn box3f_extend_by_box() {
        let mut a = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Box3f::from_min_max(Vec3::new(-1.0f32, -1.0, -1.0), Vec3::new(0.5, 0.5, 0.5));
        a.extend_by_box(&b);
        assert_eq!(a.min, Vec3::new(-1.0f32, -1.0, -1.0));
        assert_eq!(a.max, Vec3::new(1.0f32, 1.0, 1.0));
    }

    #[test]
    fn box3f_intersects() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        // point inside
        assert!(b.intersects_point(Vec3::new(1.0f32, 1.0, 1.0)));
        // point outside
        assert!(!b.intersects_point(Vec3::new(3.0f32, 1.0, 1.0)));
        // overlapping box
        let c = Box3f::from_min_max(Vec3::new(1.0f32, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(b.intersects_box(&c));
        // non-overlapping box
        let d = Box3f::from_min_max(Vec3::new(3.0f32, 3.0, 3.0), Vec3::new(4.0, 4.0, 4.0));
        assert!(!b.intersects_box(&d));
    }

    #[test]
    fn box3f_major_axis() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(b.major_axis(), 2); // z is largest (3)
    }

    #[test]
    fn box3f_has_volume() {
        let b = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(b.has_volume());
        // degenerate (flat) box
        let flat = Box3f::from_min_max(Vec3::new(0.0f32, 0.0, 0.0), Vec3::new(1.0, 0.0, 1.0));
        assert!(!flat.has_volume());
    }

    #[test]
    fn box3f_make_infinite() {
        let mut b = Box3f::new();
        b.make_infinite();
        assert!(b.is_infinite());
        assert!(!b.is_empty());
    }

    #[test]
    fn box2i_basic() {
        let mut b = Box2i::new();
        assert!(b.is_empty());
        b.extend_by_point(Vec2::new(3i32, 4));
        b.extend_by_point(Vec2::new(-1i32, 7));
        assert_eq!(b.size(), Vec2::new(4i32, 3));
        assert!(b.intersects_point(Vec2::new(0i32, 5)));
        assert!(!b.intersects_point(Vec2::new(5i32, 0)));
        assert_eq!(b.major_axis(), 0); // x-extent is 4, y-extent is 3 → x (0)
    }

    #[test]
    fn box_equality() {
        let a = Box3f::from_point(Vec3::new(1.0f32, 2.0, 3.0));
        let b = Box3f::from_point(Vec3::new(1.0f32, 2.0, 3.0));
        assert_eq!(a, b);
        let c = Box3f::from_point(Vec3::new(0.0f32, 0.0, 0.0));
        assert_ne!(a, c);
    }
}
