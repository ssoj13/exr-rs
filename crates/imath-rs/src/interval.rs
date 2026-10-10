//! Port of Imath ImathInterval.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathInterval.h>
//!
//! A scalar 1-D interval: essentially `Box<T>` where `T` is a plain scalar
//! rather than a vector.  The C++ comment says "basically a Box<T> that allows
//! T to be a scalar".

use core::fmt;

use crate::traits::Scalar;
use crate::vec::VecLimits;

/// 1-D interval (port of C++ `template <class T> class Interval`).
///
/// An interval is empty if `max < min`.  An infinite interval has
/// `min == T::LOWEST` and `max == T::MAX`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Interval<T> {
    /// The minimum of the interval.
    pub min: T,
    /// The maximum of the interval.
    pub max: T,
}

impl<T: Scalar + VecLimits> Interval<T> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Port of `Interval()` — empty interval: `min = MAX`, `max = LOWEST`.
    #[inline]
    pub fn new() -> Self {
        let mut i = Interval { min: T::ZERO, max: T::ZERO };
        i.make_empty();
        i
    }

    /// Port of `Interval(const T& point)` — single-point interval.
    #[inline]
    pub fn from_point(point: T) -> Self {
        Interval { min: point, max: point }
    }

    /// Port of `Interval(const T& minT, const T& maxT)` — explicit min/max.
    #[inline]
    pub fn from_min_max(min: T, max: T) -> Self {
        Interval { min, max }
    }

    // -----------------------------------------------------------------------
    // Manipulation
    // -----------------------------------------------------------------------

    /// Port of `makeEmpty()` — set `min = MAX`, `max = LOWEST`.
    #[inline]
    pub fn make_empty(&mut self) {
        self.min = T::MAX;
        self.max = T::LOWEST;
    }

    /// Port of `makeInfinite()` — set `min = LOWEST`, `max = MAX`.
    #[inline]
    pub fn make_infinite(&mut self) {
        self.min = T::LOWEST;
        self.max = T::MAX;
    }

    /// Port of `extendBy(const T& point)` — expand to include the given scalar.
    #[inline]
    pub fn extend_by_point(&mut self, point: T) {
        if point < self.min { self.min = point; }
        if point > self.max { self.max = point; }
    }

    /// Port of `extendBy(const Interval<T>& interval)` — expand to include another interval.
    #[inline]
    pub fn extend_by_interval(&mut self, other: &Interval<T>) {
        if other.min < self.min { self.min = other.min; }
        if other.max > self.max { self.max = other.max; }
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Port of `size()` — `(max - min)`, or `T(0)` if empty.
    #[inline]
    pub fn size(&self) -> T {
        if self.is_empty() { return T::ZERO; }
        self.max - self.min
    }

    /// Port of `center()` — `(max + min) / 2`.
    #[inline]
    pub fn center(&self) -> T {
        let two = T::ONE + T::ONE;
        (self.max + self.min) / two
    }

    /// Port of `intersects(const T& point)` — true if `min <= point <= max`.
    #[inline]
    pub fn intersects_point(&self, point: T) -> bool {
        point >= self.min && point <= self.max
    }

    /// Port of `intersects(const Interval<T>& interval)` — true if intervals overlap.
    #[inline]
    pub fn intersects_interval(&self, other: &Interval<T>) -> bool {
        other.max >= self.min && other.min <= self.max
    }

    /// Port of `isEmpty()` — true if `max < min`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.max < self.min
    }

    /// Port of `hasVolume()` — true if `max > min`.
    #[inline]
    pub fn has_volume(&self) -> bool {
        self.max > self.min
    }

    /// Port of `isInfinite()` — true if `min == LOWEST && max == MAX`.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.min == T::LOWEST && self.max == T::MAX
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl<T: Scalar + VecLimits> Default for Interval<T> {
    /// Port of `Interval()` — empty interval.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Equality operators (port of `operator==` / `operator!=`)
// ---------------------------------------------------------------------------

impl<T: PartialEq> PartialEq for Interval<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}

impl<T: Eq> Eq for Interval<T> {}

// ---------------------------------------------------------------------------
// Display (port of `operator<<`: "(min max)")
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Interval<T> {
    /// Port of `operator<<(ostream& s, const Interval<T>& v)` → `"(min max)"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.min, self.max)
    }
}

// ---------------------------------------------------------------------------
// Typedefs (1:1 with ImathInterval.h)
// ---------------------------------------------------------------------------

/// Interval of `f32` (C++ `Intervalf`).
pub type Intervalf = Interval<f32>;
/// Interval of `f64` (C++ `Intervald`).
pub type Intervald = Interval<f64>;
/// Interval of `i16` (C++ `Intervals`).
pub type Intervals = Interval<i16>;
/// Interval of `i32` (C++ `Intervali`).
pub type Intervali = Interval<i32>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervalf_empty_default() {
        let i = Intervalf::new();
        assert!(i.is_empty());
        assert!(!i.has_volume());
        assert!(!i.is_infinite());
    }

    #[test]
    fn intervalf_extend_by_point() {
        let mut i = Intervalf::new();
        i.extend_by_point(3.0f32);
        i.extend_by_point(-1.0f32);
        i.extend_by_point(5.0f32);
        assert!(!i.is_empty());
        assert_eq!(i.min, -1.0f32);
        assert_eq!(i.max, 5.0f32);
        assert_eq!(i.size(), 6.0f32);
        assert_eq!(i.center(), 2.0f32);
    }

    #[test]
    fn intervalf_extend_by_interval() {
        let mut a = Intervalf::from_min_max(0.0, 1.0);
        let b = Intervalf::from_min_max(-1.0, 0.5);
        a.extend_by_interval(&b);
        assert_eq!(a.min, -1.0f32);
        assert_eq!(a.max, 1.0f32);
    }

    #[test]
    fn intervalf_intersects() {
        let i = Intervalf::from_min_max(0.0f32, 2.0);
        assert!(i.intersects_point(1.0));
        assert!(!i.intersects_point(3.0));
        // overlapping
        let j = Intervalf::from_min_max(1.0f32, 3.0);
        assert!(i.intersects_interval(&j));
        // non-overlapping
        let k = Intervalf::from_min_max(3.0f32, 5.0);
        assert!(!i.intersects_interval(&k));
    }

    #[test]
    fn intervalf_has_volume() {
        let i = Intervalf::from_min_max(0.0f32, 1.0);
        assert!(i.has_volume());
        let point = Intervalf::from_point(1.0f32);
        assert!(!point.has_volume());
    }

    #[test]
    fn intervalf_make_infinite() {
        let mut i = Intervalf::new();
        i.make_infinite();
        assert!(i.is_infinite());
        assert!(!i.is_empty());
    }

    #[test]
    fn intervali_basic() {
        let mut i = Intervali::new();
        i.extend_by_point(10i32);
        i.extend_by_point(-5i32);
        assert_eq!(i.size(), 15i32);
        assert!(i.intersects_point(0i32));
        assert!(!i.intersects_point(11i32));
    }

    #[test]
    fn interval_equality() {
        let a = Intervalf::from_min_max(1.0f32, 2.0);
        let b = Intervalf::from_min_max(1.0f32, 2.0);
        assert_eq!(a, b);
        let c = Intervalf::from_min_max(0.0f32, 2.0);
        assert_ne!(a, c);
    }

    #[test]
    fn interval_display() {
        let i = Intervalf::from_min_max(1.0f32, 2.0);
        assert_eq!(format!("{}", i), "(1 2)");
    }
}
