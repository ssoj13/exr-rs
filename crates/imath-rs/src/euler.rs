//! Port of Imath ImathEuler.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathEuler.h>
//!
//! The [`Euler`] euler-angle orientation type, a 1:1 port of Ken Shoemake's
//! GemsIV code as exposed by Imath. `Euler<T>` represents an orientation as three
//! angles plus a [`Order`] that selects one of the 24 possible euler-angle
//! conventions (axis sequence, parity, repeated-axis, and static-vs-relative
//! frame). Units are radians.
//!
//! ## Inheritance (`Euler<T> : public Vec3<T>`)
//! In C++ `Euler<T>` publicly derives from `Vec3<T>`: it *is* a `Vec3` of the
//! three angles (`.x`/`.y`/`.z` are the i/j/k angles in the order's own
//! convention, not literal x/y/z), and the algorithms freely use the inherited
//! `Vec3` interface (field access, `operator[]`, `operator*=`, `operator==`).
//!
//! We replicate this faithfully by *composing* a `Vec3<T>` field named `vec` and
//! implementing [`Deref`]/[`DerefMut`] to it. That makes every inherited `Vec3`
//! operation available on an `Euler` (e.g. `e.x`, `e[i]`, `*e *= t`, `e.dot(..)`)
//! exactly as the C++ base-class access does, without re-deriving the vector
//! machinery. `#[repr(C)]` keeps the layout C++-compatible: the `Vec3<T>` base
//! comes first (the three angles), followed by the order state, mirroring the
//! C++ class where the `Vec3<T>` base subobject precedes the `_frameStatic`,
//! `_initialRepeated`, `_parityEven`, and `_initialAxis` members.
//!
//! ## Trait bounds vs. the C++ template
//! `Euler<T>` is instantiated only for `float`/`double` (the `Eulerf`/`Eulerd`
//! typedefs) and uses `std::` transcendentals (`atan2`/`sqrt`/`sin`/`cos`/`fmod`)
//! plus the `M_PI` literal. The [`EulerFloat`] bound spells out exactly those
//! implicit requirements (it extends [`QuatFloat`], which already provides the
//! `Scalar` + `Math` + `Vec3::length` machinery, and adds the per-type `PI`
//! constant and an `f64`→`T` narrowing cast for the `M_PI` casts). Implemented
//! only for `f32`/`f64`, the only types Imath instantiates `Euler` with.
//!
//! ## `angleMod`'s `float` return (faithful quirk)
//! The C++ declares `static float angleMod(T angle)` — it computes in `T` but
//! **narrows the result to `float`** regardless of `T`. This is reproduced 1:1:
//! [`Euler::angle_mod`] returns `f32`. For `Eulerd` this narrows, matching C++.
//!
//! ## No `throw`, no foreign interop
//! `ImathEuler.h` never throws and has no `IMATH_FOREIGN_*_INTEROP` SFINAE of its
//! own, so there is no `_exc`/`Result` counterpart and nothing to skip.

use core::fmt;
use core::ops::{Deref, DerefMut};

use crate::matrix::{Matrix33, Matrix44};
use crate::quat::{Quat, QuatFloat};
use crate::traits::Scalar;
use crate::vec::Vec3;

/// Floating-point capabilities the euler algorithms require of their element type
/// `T`, beyond what [`QuatFloat`] already provides.
///
/// `QuatFloat` (a super-bound here) already supplies `Scalar` + `Math` and the
/// `Vec3` float machinery. This trait adds only the few extra per-type pieces the
/// euler code needs that have no `QuatFloat` analog:
/// - [`PI`](EulerFloat::PI) / [`TWO_PI`](EulerFloat::TWO_PI): the `static_cast<T>(M_PI)`
///   and `2 * pi` literals used by `angleMod`/`nearestRotation`.
/// - [`NEG_ONE`](EulerFloat::NEG_ONE): the `-1` scalar used by `*this *= -1` and
///   `angles *= -1.0`.
/// - [`HALF`](EulerFloat::HALF): the `0.5` literal used by `toQuat`.
/// - [`from_f64`](EulerFloat::from_f64): the `static_cast<T>(M_PI)` narrowing for
///   the `nearestRotation` `M_PI + ...` arithmetic.
///
/// Implemented only for `f32`/`f64`, the only types Imath instantiates `Euler`
/// with.
pub trait EulerFloat: QuatFloat {
    /// `static_cast<T>(M_PI)`.
    const PI: Self;
    /// `2 * pi` (i.e. `static_cast<T>(2 * M_PI)`).
    const TWO_PI: Self;
    /// The literal `T(-1)`.
    const NEG_ONE: Self;
    /// The literal `T(0.5)` (used by `toQuat`).
    const HALF: Self;
    /// Narrow an `f64` (e.g. the `M_PI` literal) to `T`, matching the C++
    /// `static_cast<T>(...)`.
    fn from_f64(v: f64) -> Self;
    /// Narrow `self` to `f32`, mirroring the C++ implicit `T` → `float`
    /// conversion at `angleMod`'s `return` (faithful quirk: `angleMod` returns
    /// `float` regardless of `T`).
    fn to_f32(self) -> f32;
    /// Widen an `f32` back to `T` so an `angleMod` result can re-enter `T`
    /// arithmetic (e.g. `targetXyzRot.x + angleMod(d.x)`).
    fn from_f32(v: f32) -> Self;
}

impl EulerFloat for f32 {
    const PI: Self = core::f64::consts::PI as f32;
    const TWO_PI: Self = (2.0 * core::f64::consts::PI) as f32;
    const NEG_ONE: Self = -1.0;
    const HALF: Self = 0.5;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl EulerFloat for f64 {
    const PI: Self = core::f64::consts::PI;
    const TWO_PI: Self = 2.0 * core::f64::consts::PI;
    const NEG_ONE: Self = -1.0;
    const HALF: Self = 0.5;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
}

/// All 24 possible euler-angle orderings (port of `Euler<T>::Order`).
///
/// The discriminants are the **exact** C++ bit encodings. Each value packs four
/// 1-bit flags into a 16-bit word laid out as nibbles `A B C D` (see the legend
/// below), and the order/decoding accessors rely on this encoding verbatim:
///
/// ```text
///   ||||
///   VVVV
///   ABCD
/// Legend (each letter is a nibble; the meaningful bit is the low bit of the nibble):
///   A -> Initial Axis (0x0000==x, 0x1000==y, 0x2000==z)
///   B -> Parity Even  (0x0100==true)
///   C -> Initial Repeated (0x0010==true)
///   D -> Frame Static (0x0001==true)
/// ```
///
/// `repr(i32)` keeps the discriminants usable as plain integers for the bit math
/// in `setOrder`/`order`/`legal`, matching the C++ `enum` (whose underlying type
/// is `int`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
#[rustfmt::skip]
#[allow(clippy::upper_case_acronyms)] // axis-sequence names (XYZ, …) are the canonical Imath identifiers.
pub enum Order {
    // "usual" orderings
    XYZ = 0x0101,
    XZY = 0x0001,
    YZX = 0x1101,
    YXZ = 0x1001,
    ZXY = 0x2101,
    ZYX = 0x2001,

    // first axis repeated
    XZX = 0x0011,
    XYX = 0x0111,
    YXY = 0x1011,
    YZY = 0x1111,
    ZYZ = 0x2011,
    ZXZ = 0x2111,

    // relative orderings -- not common
    XYZr = 0x2000,
    XZYr = 0x2100,
    YZXr = 0x1000,
    YXZr = 0x1100,
    ZXYr = 0x0000,
    ZYXr = 0x0100,

    // relative first axis repeated
    XZXr = 0x2110,
    XYXr = 0x2010,
    YXYr = 0x1110,
    YZYr = 0x1010,
    ZYZr = 0x0110,
    ZXZr = 0x0010,
}

impl Order {
    /// Bitwise OR of every legal order, port of the C++ `Euler::Legal` enumerator.
    ///
    /// `legal()` uses this mask to reject any bit set outside the legal set.
    pub const LEGAL: i32 = Order::XYZ as i32
        | Order::XZY as i32
        | Order::YZX as i32
        | Order::YXZ as i32
        | Order::ZXY as i32
        | Order::ZYX as i32
        | Order::XZX as i32
        | Order::XYX as i32
        | Order::YXY as i32
        | Order::YZY as i32
        | Order::ZYZ as i32
        | Order::ZXZ as i32
        | Order::XYZr as i32
        | Order::XZYr as i32
        | Order::YZXr as i32
        | Order::YXZr as i32
        | Order::ZXYr as i32
        | Order::ZYXr as i32
        | Order::XZXr as i32
        | Order::XYXr as i32
        | Order::YXYr as i32
        | Order::YZYr as i32
        | Order::ZYZr as i32
        | Order::ZXZr as i32;

    /// Port of the C++ `Min` enumerator (`0x0000`).
    pub const MIN: i32 = 0x0000;
    /// Port of the C++ `Max` enumerator (`0x2111`).
    pub const MAX: i32 = 0x2111;
    /// Port of the C++ `Default` enumerator (`XYZ`).
    pub const DEFAULT: Order = Order::XYZ;
}

/// The three rotation axes (port of `Euler<T>::Axis`).
///
/// The discriminants are the C++ values `X = 0, Y = 1, Z = 2`, used directly as
/// indices into the angle vector and as the `_initialAxis` state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Axis {
    /// The x axis (`0`).
    X = 0,
    /// The y axis (`1`).
    Y = 1,
    /// The z axis (`2`).
    Z = 2,
}

/// Input-vector interpretation for the vector/scalar constructors (port of
/// `Euler<T>::InputLayout`).
///
/// - [`XYZLayout`](InputLayout::XYZLayout): the input `(x, y, z)` are literal x/y/z
///   angles and are reshuffled into ijk order via `setXYZVector`.
/// - [`IJKLayout`](InputLayout::IJKLayout): the input is already in the order's own
///   ijk slot order and is stored verbatim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum InputLayout {
    /// Input interpreted as literal x/y/z angles (reshuffled to ijk).
    XYZLayout,
    /// Input interpreted as already-ijk-ordered angles (stored verbatim).
    IJKLayout,
}

/// Euler-angle orientation (port of C++ `template <class T> class Euler : public Vec3<T>`).
///
/// `vec` holds the three angles (the inherited `Vec3<T>` base); the four order
/// fields hold the decoded [`Order`] state. Access the angles through the
/// [`Deref`]/[`DerefMut`] to `Vec3<T>` (`e.x`, `e.y`, `e.z`, `e[i]`, `*e *= t`,
/// …), exactly as C++ uses the inherited base-class interface.
///
/// `#[repr(C)]` matches the C++ layout: the `Vec3<T>` base first, then the order
/// state.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Euler<T> {
    /// The three euler angles (the inherited C++ `Vec3<T>` base subobject). In
    /// the order's own convention these are the i/j/k angles, **not** literal
    /// x/y/z (see the module docs).
    pub vec: Vec3<T>,
    /// C++ `bool _frameStatic : 1` — static (true) vs. relative (false) frame.
    frame_static: bool,
    /// C++ `bool _initialRepeated : 1` — initial axis repeated as the last.
    initial_repeated: bool,
    /// C++ `bool _parityEven : 1` — parity of the axis permutation.
    parity_even: bool,
    /// C++ `Axis _initialAxis : 2` — first axis of rotation.
    initial_axis: Axis,
}

// ----------------------------------------------------------------------------
// Vec3 "inheritance": Deref/DerefMut expose the angle vector's full interface,
// mirroring `Euler<T> : public Vec3<T>`.
// ----------------------------------------------------------------------------

impl<T> Deref for Euler<T> {
    type Target = Vec3<T>;
    #[inline]
    fn deref(&self) -> &Vec3<T> {
        &self.vec
    }
}

impl<T> DerefMut for Euler<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec3<T> {
        &mut self.vec
    }
}

// ----------------------------------------------------------------------------
// Constructors
//
// All default to ZYX non-relative (ala Softimage 3D/Maya) where there is no
// argument to specify it. (Note: the *enum* Default is XYZ; the constructor
// default arguments use `Default = XYZ`, matching the header's signatures.)
// ----------------------------------------------------------------------------

impl<T: Scalar> Default for Euler<T> {
    /// Port of the default constructor `Euler()` — no rotation, with the default
    /// decoded state `(_frameStatic = true, _initialRepeated = false,
    /// _parityEven = true, _initialAxis = X)`. Angles initialized to `(0, 0, 0)`.
    #[inline]
    fn default() -> Self {
        Euler {
            vec: Vec3::new(T::ZERO, T::ZERO, T::ZERO),
            frame_static: true,
            initial_repeated: false,
            parity_even: true,
            initial_axis: Axis::X,
        }
    }
}

impl<T: EulerFloat> Euler<T> {
    /// Port of `Euler(Order p)` — zero angles, with the given order.
    #[inline]
    pub fn from_order(p: Order) -> Self {
        let mut e = Euler {
            vec: Vec3::new(T::ZERO, T::ZERO, T::ZERO),
            frame_static: true,
            initial_repeated: false,
            parity_even: true,
            initial_axis: Axis::X,
        };
        e.set_order(p);
        e
    }

    /// Port of `Euler(const Vec3<T>& v, Order o = Default, InputLayout l = IJKLayout)`.
    ///
    /// Sets the order, then stores the angle vector either reshuffled from x/y/z
    /// layout (`XYZLayout`) or verbatim in ijk layout (`IJKLayout`).
    #[inline]
    pub fn from_vec_order_layout(v: Vec3<T>, o: Order, l: InputLayout) -> Self {
        let mut e = Euler::default();
        e.set_order(o);
        if l == InputLayout::XYZLayout {
            e.set_xyz_vector(&v);
        } else {
            e.vec.x = v.x;
            e.vec.y = v.y;
            e.vec.z = v.z;
        }
        e
    }

    /// Port of `Euler(T i, T j, T k, Order o = Default, InputLayout l = IJKLayout)`
    /// — construct from three explicit angles, an order, and a layout.
    #[inline]
    pub fn from_angles(xi: T, yi: T, zi: T, o: Order, l: InputLayout) -> Self {
        let mut e = Euler::default();
        e.set_order(o);
        if l == InputLayout::XYZLayout {
            e.set_xyz_vector(&Vec3::new(xi, yi, zi));
        } else {
            e.vec.x = xi;
            e.vec.y = yi;
            e.vec.z = zi;
        }
        e
    }

    /// Port of `Euler(const Euler<T>& euler, Order newp)` — copy `euler` but
    /// re-express it under a new order. Implemented exactly as C++: set the new
    /// order, build the source rotation matrix, and re-extract the angles.
    #[inline]
    pub fn from_euler_order(euler: &Euler<T>, newp: Order) -> Self {
        let mut e = Euler::default();
        e.set_order(newp);
        let m = euler.to_matrix33();
        e.extract33(&m);
        e
    }

    /// Port of `Euler(const Matrix33<T>&, Order o = Default)`.
    #[inline]
    pub fn from_matrix33(m: &Matrix33<T>, o: Order) -> Self {
        let mut e = Euler::default();
        e.set_order(o);
        e.extract33(m);
        e
    }

    /// Port of `Euler(const Matrix44<T>&, Order o = Default)`.
    #[inline]
    pub fn from_matrix44(m: &Matrix44<T>, o: Order) -> Self {
        let mut e = Euler::default();
        e.set_order(o);
        e.extract44(m);
        e
    }
}

// ----------------------------------------------------------------------------
// Query
// ----------------------------------------------------------------------------

impl<T> Euler<T> {
    /// Port of `static bool legal(Order)` — whether the value is a legal order
    /// (no bit set outside [`Order::LEGAL`]).
    #[inline]
    pub fn legal(order: Order) -> bool {
        (order as i32 & !Order::LEGAL) == 0
    }

    /// Port of `bool frameStatic() const`.
    #[inline]
    pub fn frame_static(&self) -> bool {
        self.frame_static
    }

    /// Port of `bool initialRepeated() const`.
    #[inline]
    pub fn initial_repeated(&self) -> bool {
        self.initial_repeated
    }

    /// Port of `bool parityEven() const`.
    #[inline]
    pub fn parity_even(&self) -> bool {
        self.parity_even
    }

    /// Port of `Axis initialAxis() const`.
    #[inline]
    pub fn initial_axis(&self) -> Axis {
        self.initial_axis
    }

    /// Port of `Order order() const` — re-encode the decoded state back to its
    /// [`Order`] discriminant. Reproduces the C++ bit assembly exactly:
    /// initial-axis nibble, then OR-in parity / repeated, then `+1` for static.
    #[inline]
    pub fn order(&self) -> Order {
        // `bits` is the C++ `foo` local: the order word assembled bit by bit.
        let mut bits: i32 = if self.initial_axis == Axis::Z {
            0x2000
        } else if self.initial_axis == Axis::Y {
            0x1000
        } else {
            0
        };

        if self.parity_even {
            bits |= 0x0100;
        }
        if self.initial_repeated {
            bits |= 0x0010;
        }
        if self.frame_static {
            bits += 1;
        }

        // `bits` is, by construction, always one of the 24 legal encodings.
        order_from_i32(bits)
    }

    /// Port of `void angleOrder(int& i, int& j, int& k) const` — unpack the i/j/k
    /// axis indices for the current order from the initial axis and parity.
    #[inline]
    pub fn angle_order(&self) -> (i32, i32, i32) {
        let i = self.initial_axis as i32;
        let j = if self.parity_even {
            (i + 1) % 3
        } else if i > 0 {
            i - 1
        } else {
            2
        };
        let k = if self.parity_even {
            if i > 0 { i - 1 } else { 2 }
        } else {
            (i + 1) % 3
        };
        (i, j, k)
    }

    /// Port of `void angleMapping(int& i, int& j, int& k) const` — determine the
    /// mapping from xyz to ijk (reshuffle xyz to match the order).
    #[inline]
    pub fn angle_mapping(&self) -> (i32, i32, i32) {
        let mut m = [0i32; 3];
        let a = self.initial_axis as usize;
        m[a] = 0;
        m[(a + 1) % 3] = if self.parity_even { 1 } else { 2 };
        m[(a + 2) % 3] = if self.parity_even { 2 } else { 1 };
        (m[0], m[1], m[2])
    }
}

// ----------------------------------------------------------------------------
// Set value
// ----------------------------------------------------------------------------

impl<T: Copy> Euler<T> {
    /// Port of `void setXYZVector(const Vec3<T>&)` — set the first angle to the
    /// xyz-mapped slot, etc.; reshuffles `(v.x, v.y, v.z)` into ijk order.
    #[inline]
    pub fn set_xyz_vector(&mut self, v: &Vec3<T>) {
        let (i, j, k) = self.angle_mapping();
        self.vec[i as usize] = v.x;
        self.vec[j as usize] = v.y;
        self.vec[k as usize] = v.z;
    }

    /// Port of `void set(Axis initial, bool relative, bool parityEven, bool firstRepeats)`
    /// — set the decoded order state directly. Note `_frameStatic = !relative`.
    #[inline]
    pub fn set(&mut self, axis: Axis, relative: bool, parity_even: bool, first_repeats: bool) {
        self.initial_axis = axis;
        self.frame_static = !relative;
        self.parity_even = parity_even;
        self.initial_repeated = first_repeats;
    }

    /// Port of `void setOrder(Order)` — decode an [`Order`] into the four state
    /// fields. Mirrors the C++ bit tests exactly (including the inverted
    /// static-flag and the `!!` boolean coercions).
    #[inline]
    pub fn set_order(&mut self, p: Order) {
        let p = p as i32;
        self.set(
            if p & 0x2000 != 0 {
                Axis::Z
            } else if p & 0x1000 != 0 {
                Axis::Y
            } else {
                Axis::X
            }, // initial axis
            (p & 0x1) == 0,    // static?  ->  relative = !(p & 0x1)
            (p & 0x100) != 0,  // permutation even?
            (p & 0x10) != 0,   // initial repeats?
        );
    }
}

// ----------------------------------------------------------------------------
// Assignments and conversions: extract / toMatrix / toQuat / toXYZVector
// ----------------------------------------------------------------------------

impl<T: EulerFloat> Euler<T> {
    /// Port of `Vec3<T> toXYZVector() const` — reorder the angles so x comes
    /// first, then y, z (the repeated angle, if any, ends in z).
    #[inline]
    pub fn to_xyz_vector(&self) -> Vec3<T> {
        let (i, j, k) = self.angle_mapping();
        Vec3::new(
            self.vec[i as usize],
            self.vec[j as usize],
            self.vec[k as usize],
        )
    }

    /// Port of `void extract(const Quat<T>&)` — extract from a quaternion by way
    /// of its 3x3 rotation matrix.
    #[inline]
    pub fn extract_quat(&mut self, q: &Quat<T>) {
        let m = q.to_matrix33();
        self.extract33(&m);
    }

    /// Port of `void extract(const Matrix33<T>&)` — extract euler angles from an
    /// affine 3x3 rotation matrix. 1:1 with the C++ algorithm, including the
    /// gimbal-lock-avoiding "remove first rotation, then solve the remaining 2D
    /// problem" trick and the final parity/frame fix-ups.
    pub fn extract33(&mut self, m: &Matrix33<T>) {
        let (i, j, k) = self.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        if self.initial_repeated {
            // Extract the first angle, x.
            self.vec.x = m[j][i].atan2(m[k][i]);

            // Remove the x rotation from M so the remaining rotation N is only
            // around two axes and gimbal lock cannot occur.
            let mut r = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            r[i] = if self.parity_even { -self.vec.x } else { self.vec.x };

            let mut n = Matrix44::default();
            n.rotate(&r);
            let n = n.mul_matrix(&Matrix44::new(
                m[0][0], m[0][1], m[0][2], T::ZERO,
                m[1][0], m[1][1], m[1][2], T::ZERO,
                m[2][0], m[2][1], m[2][2], T::ZERO,
                T::ZERO, T::ZERO, T::ZERO, T::ONE,
            ));

            // Extract the other two angles, y and z, from N.
            let sy = (n[j][i] * n[j][i] + n[k][i] * n[k][i]).sqrt();
            self.vec.y = sy.atan2(n[i][i]);
            self.vec.z = n[j][k].atan2(n[j][j]);
        } else {
            // Extract the first angle, x.
            self.vec.x = m[j][k].atan2(m[k][k]);

            // Remove the x rotation from M (see above).
            let mut r = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            r[i] = if self.parity_even { -self.vec.x } else { self.vec.x };

            let mut n = Matrix44::default();
            n.rotate(&r);
            let n = n.mul_matrix(&Matrix44::new(
                m[0][0], m[0][1], m[0][2], T::ZERO,
                m[1][0], m[1][1], m[1][2], T::ZERO,
                m[2][0], m[2][1], m[2][2], T::ZERO,
                T::ZERO, T::ZERO, T::ZERO, T::ONE,
            ));

            // Extract the other two angles, y and z, from N.
            let cy = (n[i][i] * n[i][i] + n[i][j] * n[i][j]).sqrt();
            self.vec.y = (-n[i][k]).atan2(cy);
            self.vec.z = (-n[j][i]).atan2(n[j][j]);
        }

        if !self.parity_even {
            self.vec *= T::NEG_ONE;
        }

        if !self.frame_static {
            // C++ `T t = x; x = z; z = t;` — a plain swap of the first/last angle.
            core::mem::swap(&mut self.vec.x, &mut self.vec.z);
        }
    }

    /// Port of `void extract(const Matrix44<T>&)` — same algorithm as
    /// [`extract33`](Self::extract33) but operating directly on the 4x4 matrix
    /// (no copy into a fresh `Matrix44` is needed; `N = N * M` uses `M` itself).
    pub fn extract44(&mut self, m: &Matrix44<T>) {
        let (i, j, k) = self.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        if self.initial_repeated {
            // Extract the first angle, x.
            self.vec.x = m[j][i].atan2(m[k][i]);

            let mut r = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            r[i] = if self.parity_even { -self.vec.x } else { self.vec.x };

            let mut n = Matrix44::default();
            n.rotate(&r);
            let n = n.mul_matrix(m);

            let sy = (n[j][i] * n[j][i] + n[k][i] * n[k][i]).sqrt();
            self.vec.y = sy.atan2(n[i][i]);
            self.vec.z = n[j][k].atan2(n[j][j]);
        } else {
            // Extract the first angle, x.
            self.vec.x = m[j][k].atan2(m[k][k]);

            let mut r = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
            r[i] = if self.parity_even { -self.vec.x } else { self.vec.x };

            let mut n = Matrix44::default();
            n.rotate(&r);
            let n = n.mul_matrix(m);

            let cy = (n[i][i] * n[i][i] + n[i][j] * n[i][j]).sqrt();
            self.vec.y = (-n[i][k]).atan2(cy);
            self.vec.z = (-n[j][i]).atan2(n[j][j]);
        }

        if !self.parity_even {
            self.vec *= T::NEG_ONE;
        }

        if !self.frame_static {
            // C++ `T t = x; x = z; z = t;` — a plain swap of the first/last angle.
            core::mem::swap(&mut self.vec.x, &mut self.vec.z);
        }
    }

    /// Port of `Matrix33<T> toMatrix33() const` — build the 3x3 rotation matrix
    /// for the current order and angles. 1:1 with the C++ trigonometric assembly
    /// (separate repeated-axis and non-repeated cases).
    pub fn to_matrix33(&self) -> Matrix33<T> {
        let (i, j, k) = self.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        let mut angles = if self.frame_static {
            self.vec
        } else {
            Vec3::new(self.vec.z, self.vec.y, self.vec.x)
        };

        if !self.parity_even {
            angles *= T::NEG_ONE;
        }

        let ci = angles.x.cos();
        let cj = angles.y.cos();
        let ch = angles.z.cos();
        let si = angles.x.sin();
        let sj = angles.y.sin();
        let sh = angles.z.sin();

        let cc = ci * ch;
        let cs = ci * sh;
        let sc = si * ch;
        let ss = si * sh;

        let mut m = Matrix33::default();

        if self.initial_repeated {
            m[i][i] = cj;
            m[j][i] = sj * si;
            m[k][i] = sj * ci;
            m[i][j] = sj * sh;
            m[j][j] = -cj * ss + cc;
            m[k][j] = -cj * cs - sc;
            m[i][k] = -sj * ch;
            m[j][k] = cj * sc + cs;
            m[k][k] = cj * cc - ss;
        } else {
            m[i][i] = cj * ch;
            m[j][i] = sj * sc - cs;
            m[k][i] = sj * cc + ss;
            m[i][j] = cj * sh;
            m[j][j] = sj * ss + cc;
            m[k][j] = sj * cs - sc;
            m[i][k] = -sj;
            m[j][k] = cj * si;
            m[k][k] = cj * ci;
        }

        m
    }

    /// Port of `Matrix44<T> toMatrix44() const` — same trigonometric assembly as
    /// [`to_matrix33`](Self::to_matrix33), written into the upper-left 3x3 block
    /// of an identity 4x4 matrix.
    pub fn to_matrix44(&self) -> Matrix44<T> {
        let (i, j, k) = self.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        let mut angles = if self.frame_static {
            self.vec
        } else {
            Vec3::new(self.vec.z, self.vec.y, self.vec.x)
        };

        if !self.parity_even {
            angles *= T::NEG_ONE;
        }

        let ci = angles.x.cos();
        let cj = angles.y.cos();
        let ch = angles.z.cos();
        let si = angles.x.sin();
        let sj = angles.y.sin();
        let sh = angles.z.sin();

        let cc = ci * ch;
        let cs = ci * sh;
        let sc = si * ch;
        let ss = si * sh;

        let mut m = Matrix44::default();

        if self.initial_repeated {
            m[i][i] = cj;
            m[j][i] = sj * si;
            m[k][i] = sj * ci;
            m[i][j] = sj * sh;
            m[j][j] = -cj * ss + cc;
            m[k][j] = -cj * cs - sc;
            m[i][k] = -sj * ch;
            m[j][k] = cj * sc + cs;
            m[k][k] = cj * cc - ss;
        } else {
            m[i][i] = cj * ch;
            m[j][i] = sj * sc - cs;
            m[k][i] = sj * cc + ss;
            m[i][j] = cj * sh;
            m[j][j] = sj * ss + cc;
            m[k][j] = sj * cs - sc;
            m[i][k] = -sj;
            m[j][k] = cj * si;
            m[k][k] = cj * ci;
        }

        m
    }

    /// Port of `Quat<T> toQuat() const` — convert to a quaternion. 1:1 with the
    /// C++ half-angle assembly (separate repeated-axis and non-repeated cases),
    /// including the `_parityEven`-driven sign of the middle angle and the
    /// `parity` factor on the `j` component.
    pub fn to_quat(&self) -> Quat<T> {
        let (i, j, k) = self.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        let mut angles = if self.frame_static {
            self.vec
        } else {
            Vec3::new(self.vec.z, self.vec.y, self.vec.x)
        };

        if !self.parity_even {
            angles.y = -angles.y;
        }

        let ti = angles.x * T::HALF;
        let tj = angles.y * T::HALF;
        let th = angles.z * T::HALF;
        let ci = ti.cos();
        let cj = tj.cos();
        let ch = th.cos();
        let si = ti.sin();
        let sj = tj.sin();
        let sh = th.sin();
        let cc = ci * ch;
        let cs = ci * sh;
        let sc = si * ch;
        let ss = si * sh;

        let parity = if self.parity_even { T::ONE } else { T::NEG_ONE };

        let mut q = Quat::default();
        let mut a = Vec3::new(T::ZERO, T::ZERO, T::ZERO);

        if self.initial_repeated {
            a[i] = cj * (cs + sc);
            a[j] = sj * (cc + ss) * parity;
            a[k] = sj * (cs - sc);
            q.r = cj * (cc - ss);
        } else {
            a[i] = cj * sc - sj * cs;
            a[j] = (cj * ss + sj * cc) * parity;
            a[k] = cj * cs - sj * sc;
            q.r = cj * cc + sj * ss;
        }

        q.v = a;
        q
    }
}

// ----------------------------------------------------------------------------
// Assignment
// ----------------------------------------------------------------------------

impl<T: Copy> Euler<T> {
    /// Port of `const Euler<T>& operator=(const Vec3<T>&)` — assign just the angle
    /// vector, leaving the order state unchanged.
    #[inline]
    pub fn assign_vec(&mut self, v: &Vec3<T>) -> &mut Self {
        self.vec.x = v.x;
        self.vec.y = v.y;
        self.vec.z = v.z;
        self
    }
}

// Port of `const Euler<T>& operator=(const Euler<T>&)`: in Rust the derived
// `Clone`/`Copy` (and ordinary `=`) already copies both the angle vector and the
// full order state field-by-field, which is exactly what the C++ `operator=` does.

// ----------------------------------------------------------------------------
// Comparison (port of `operator==` / `operator!=`)
//
// The C++ comparison is `Vec3<T>::operator==(other) && order() == other.order()`.
// Imath's `operator==` is heterogeneous (`template <class S>`), comparing a
// `Euler<T>` against a `Euler<S>`; we model the homogeneous (same-`T`) case via
// `PartialEq`, which is the only instantiation the OpenEXR port needs.
// ----------------------------------------------------------------------------

impl<T: EulerFloat> PartialEq for Euler<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vec == other.vec && self.order() == other.order()
    }
}

// ----------------------------------------------------------------------------
// Utility methods for continuous rotations
// ----------------------------------------------------------------------------

impl<T: EulerFloat> Euler<T> {
    /// Port of `static float angleMod(T angle)` — convert an angle to its
    /// equivalent in `[-PI, PI]`.
    ///
    /// Faithful quirk: the C++ signature returns `float` regardless of `T`, so
    /// this returns `f32` even for `Euler<f64>` (the computation is done in `T`
    /// and narrowed on return, matching C++).
    #[inline]
    pub fn angle_mod(angle: T) -> f32 {
        let pi = T::PI;
        let mut angle = angle.fmod(T::TWO_PI);

        if angle < -pi {
            angle = angle + T::TWO_PI;
        }
        if angle > pi {
            angle = angle - T::TWO_PI;
        }

        // C++ narrows the `T` result to `float` on return.
        T::to_f32(angle)
    }

    /// Port of `static void simpleXYZRotation(Vec3<T>& xyzRot, const Vec3<T>& targetXyzRot)`
    /// — adjust `xyz_rot` so its components differ from `target_xyz_rot` by no
    /// more than +/-PI.
    #[inline]
    pub fn simple_xyz_rotation(xyz_rot: &mut Vec3<T>, target_xyz_rot: &Vec3<T>) {
        let d = *xyz_rot - *target_xyz_rot;
        xyz_rot.x = target_xyz_rot.x + T::from_f32(Self::angle_mod(d.x));
        xyz_rot.y = target_xyz_rot.y + T::from_f32(Self::angle_mod(d.y));
        xyz_rot.z = target_xyz_rot.z + T::from_f32(Self::angle_mod(d.z));
    }

    /// Port of `static void nearestRotation(Vec3<T>& xyzRot, const Vec3<T>& targetXyzRot, Order order = XYZ)`
    /// — adjust `xyz_rot` so its components differ from `target_xyz_rot` by as
    /// little as possible. (`xyz` here really means ijk, hence the `order`.)
    pub fn nearest_rotation(xyz_rot: &mut Vec3<T>, target_xyz_rot: &Vec3<T>, order: Order) {
        let e = Euler::<T>::from_angles(T::ZERO, T::ZERO, T::ZERO, order, InputLayout::IJKLayout);
        let (i, j, k) = e.angle_order();
        let (i, j, k) = (i as usize, j as usize, k as usize);

        Self::simple_xyz_rotation(xyz_rot, target_xyz_rot);

        let pi = T::PI;
        let mut other_xyz_rot = Vec3::new(T::ZERO, T::ZERO, T::ZERO);
        other_xyz_rot[i] = pi + xyz_rot[i];
        other_xyz_rot[j] = pi - xyz_rot[j];
        other_xyz_rot[k] = pi + xyz_rot[k];

        Self::simple_xyz_rotation(&mut other_xyz_rot, target_xyz_rot);

        let d = *xyz_rot - *target_xyz_rot;
        let od = other_xyz_rot - *target_xyz_rot;
        let d_mag = d.dot(&d);
        let od_mag = od.dot(&od);

        if od_mag < d_mag {
            *xyz_rot = other_xyz_rot;
        }
    }

    /// Port of `void makeNear(const Euler<T>& target)` — adjust `self` so its
    /// components differ from `target` by as little as possible. Re-expresses the
    /// target under `self`'s order first if they differ. (As the C++ notes, this
    /// is dubious for repeated-axis / relative orderings — a known TODO upstream.)
    pub fn make_near(&mut self, target: &Euler<T>) {
        let mut xyz_rot = self.to_xyz_vector();
        let target_xyz = if self.order() != target.order() {
            let target_same_order = Euler::from_euler_order(target, self.order());
            target_same_order.to_xyz_vector()
        } else {
            target.to_xyz_vector()
        };

        let order = self.order();
        Self::nearest_rotation(&mut xyz_rot, &target_xyz, order);

        self.set_xyz_vector(&xyz_rot);
    }
}

/// Stream output, port of `operator<<(ostream&, const Euler<T>&)` — formatted as
/// `(x y z IJKr)` where `IJK` are the order's axis letters and a trailing `r`
/// marks a relative (non-static) frame.
impl<T: EulerFloat + fmt::Display> fmt::Display for Euler<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = *b"XYZ";

        let r = if self.frame_static() { "" } else { "r" };
        let (i, j, mut k) = self.angle_order();

        if self.initial_repeated() {
            k = i;
        }

        write!(
            f,
            "({} {} {} {}{}{}{})",
            self.vec.x,
            self.vec.y,
            self.vec.z,
            a[i as usize] as char,
            a[j as usize] as char,
            a[k as usize] as char,
            r
        )
    }
}

/// Decode an `i32` known to be one of the 24 legal [`Order`] encodings back to the
/// enum. Used by [`Euler::order`], whose bit assembly is guaranteed to land on a
/// legal value.
#[inline]
#[rustfmt::skip]
fn order_from_i32(v: i32) -> Order {
    match v {
        0x0101 => Order::XYZ,
        0x0001 => Order::XZY,
        0x1101 => Order::YZX,
        0x1001 => Order::YXZ,
        0x2101 => Order::ZXY,
        0x2001 => Order::ZYX,
        0x0011 => Order::XZX,
        0x0111 => Order::XYX,
        0x1011 => Order::YXY,
        0x1111 => Order::YZY,
        0x2011 => Order::ZYZ,
        0x2111 => Order::ZXZ,
        0x2000 => Order::XYZr,
        0x2100 => Order::XZYr,
        0x1000 => Order::YZXr,
        0x1100 => Order::YXZr,
        0x0000 => Order::ZXYr,
        0x0100 => Order::ZYXr,
        0x2110 => Order::XZXr,
        0x2010 => Order::XYXr,
        0x1110 => Order::YXYr,
        0x1010 => Order::YZYr,
        0x0110 => Order::ZYZr,
        0x0010 => Order::ZXZr,
        // `order()` only ever produces a legal encoding; any other value is a
        // logic error in this module.
        other => unreachable!("order_from_i32: illegal order encoding {other:#06x}"),
    }
}

/// Euler of type `f32` — port of `typedef Euler<float> Eulerf`.
pub type Eulerf = Euler<f32>;
/// Euler of type `f64` — port of `typedef Euler<double> Eulerd`.
pub type Eulerd = Euler<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    /// Multiply a 4x4 rotation matrix (row-vector convention, as Imath uses) by a
    /// point: `v * M`, i.e. row-vector times matrix, ignoring translation.
    fn xform(m: &Matrix44<f64>, v: Vec3<f64>) -> Vec3<f64> {
        Vec3::new(
            v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0],
            v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1],
            v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2],
        )
    }

    fn vclose(a: Vec3<f64>, b: Vec3<f64>, eps: f64) -> bool {
        (a.x - b.x).abs() < eps && (a.y - b.y).abs() < eps && (a.z - b.z).abs() < eps
    }

    /// The default decoded state matches the C++ default constructor.
    #[test]
    fn default_state() {
        let e = Eulerd::default();
        assert!(e.frame_static());
        assert!(!e.initial_repeated());
        assert!(e.parity_even());
        assert_eq!(e.initial_axis(), Axis::X);
        assert_eq!(e.order(), Order::XYZ);
        assert_eq!(e.vec, Vec3::new(0.0, 0.0, 0.0));
    }

    /// `setOrder` decoding and `order()` re-encoding round-trip for every one of
    /// the 24 orders, and `legal()` accepts them all.
    #[test]
    fn order_roundtrip_all_24() {
        let orders = [
            Order::XYZ, Order::XZY, Order::YZX, Order::YXZ, Order::ZXY, Order::ZYX,
            Order::XZX, Order::XYX, Order::YXY, Order::YZY, Order::ZYZ, Order::ZXZ,
            Order::XYZr, Order::XZYr, Order::YZXr, Order::YXZr, Order::ZXYr, Order::ZYXr,
            Order::XZXr, Order::XYXr, Order::YXYr, Order::YZYr, Order::ZYZr, Order::ZXZr,
        ];
        for o in orders {
            assert!(Eulerd::legal(o), "{o:?} should be legal");
            let e = Eulerd::from_order(o);
            assert_eq!(e.order(), o, "order() must re-encode {o:?}");
        }
    }

    /// `angleOrder` decoding matches the documented axis sequences.
    #[test]
    fn angle_order_decoding() {
        // XYZ: initial axis X(0), parity even -> i=0, j=1, k=2.
        let e = Eulerd::from_order(Order::XYZ);
        assert_eq!(e.angle_order(), (0, 1, 2));
        // ZYX (0x2001): initial axis Z(2), parity ODD (0x2001 & 0x100 == 0).
        // parity odd -> j = (i>0? i-1 : 2) = 1, k = (i+1)%3 = 0.
        let e = Eulerd::from_order(Order::ZYX);
        assert!(!e.parity_even());
        assert_eq!(e.angle_order(), (2, 1, 0));
        // XZY (0x0001): initial axis X(0), parity ODD.
        // parity odd -> j = (i>0? i-1:2) = 2, k = (i+1)%3 = 1.
        let e = Eulerd::from_order(Order::XZY);
        assert_eq!(e.angle_order(), (0, 2, 1));
    }

    /// toMatrix44 -> extract round-trips the angles for a non-repeating,
    /// non-relative order (XYZ), well away from gimbal lock.
    #[test]
    fn matrix_extract_roundtrip_xyz() {
        let src = Eulerd::from_angles(0.3, -0.7, 1.1, Order::XYZ, InputLayout::IJKLayout);
        let m = src.to_matrix44();
        let mut back = Eulerd::from_order(Order::XYZ);
        back.extract44(&m);
        assert!(
            vclose(src.vec, back.vec, 1e-9),
            "xyz roundtrip: {:?} vs {:?}",
            src.vec,
            back.vec
        );
    }

    /// Same round-trip for ZYX (different initial axis) confirms the order is
    /// honored end to end.
    #[test]
    fn matrix_extract_roundtrip_zyx() {
        let src = Eulerd::from_angles(0.5, 0.2, -0.9, Order::ZYX, InputLayout::IJKLayout);
        let m = src.to_matrix44();
        let mut back = Eulerd::from_order(Order::ZYX);
        back.extract44(&m);
        assert!(vclose(src.vec, back.vec, 1e-9));
    }

    /// Round-trip through a repeated-axis order (XYX) and a relative order (XYZr)
    /// to exercise both special branches of extract/toMatrix.
    #[test]
    fn matrix_extract_roundtrip_special_orders() {
        for o in [Order::XYX, Order::XYZr] {
            let src = Eulerd::from_angles(0.4, 0.6, -0.3, o, InputLayout::IJKLayout);
            let m = src.to_matrix44();
            // Rebuild from the matrix and compare the *resulting matrices* (angle
            // vectors can differ by an equivalent representation, but the
            // orientation must match).
            let mut back = Eulerd::from_order(o);
            back.extract44(&m);
            let m2 = back.to_matrix44();
            for r in 0..4 {
                for c in 0..4 {
                    assert!(
                        (m[r][c] - m2[r][c]).abs() < 1e-9,
                        "{o:?} matrix mismatch at [{r}][{c}]"
                    );
                }
            }
        }
    }

    /// toQuat and toMatrix33 must rotate a vector identically (consistency).
    #[test]
    fn quat_matrix_consistency() {
        let e = Eulerd::from_angles(0.3, -0.7, 1.1, Order::XYZ, InputLayout::IJKLayout);
        let m = e.to_matrix44();
        let q = e.to_quat();
        let qm = q.to_matrix44();
        let v = Vec3::new(0.37, -0.21, 0.88);
        let by_m = xform(&m, v);
        let by_q = xform(&qm, v);
        assert!(
            vclose(by_m, by_q, 1e-9),
            "quat vs matrix: {by_m:?} vs {by_q:?}"
        );
    }

    /// Different orders with the same ijk angles must produce different matrices.
    #[test]
    fn distinct_orders_distinct_matrices() {
        let a = Eulerd::from_angles(0.3, 0.7, 1.1, Order::XYZ, InputLayout::IJKLayout);
        let b = Eulerd::from_angles(0.3, 0.7, 1.1, Order::ZYX, InputLayout::IJKLayout);
        let ma = a.to_matrix44();
        let mb = b.to_matrix44();
        let mut differ = false;
        for r in 0..4 {
            for c in 0..4 {
                if (ma[r][c] - mb[r][c]).abs() > 1e-6 {
                    differ = true;
                }
            }
        }
        assert!(differ, "XYZ and ZYX of the same angles must differ");
    }

    /// `toXYZVector`/`setXYZVector` round-trip and respect the order's mapping.
    #[test]
    fn xyz_vector_roundtrip() {
        // For YXZ, the ijk slots are reshuffled; setting an XYZ vector then
        // reading it back must return the same xyz vector.
        let mut e = Eulerd::from_order(Order::YXZ);
        let xyz = Vec3::new(0.11, 0.22, 0.33);
        e.set_xyz_vector(&xyz);
        assert!(vclose(e.to_xyz_vector(), xyz, EPS));
        // And the XYZLayout constructor must equal setXYZVector.
        let e2 = Eulerd::from_vec_order_layout(xyz, Order::YXZ, InputLayout::XYZLayout);
        assert_eq!(e.vec, e2.vec);
    }

    /// `angleMod` maps angles into [-PI, PI] and returns f32 (faithful quirk).
    #[test]
    fn angle_mod_range() {
        let pi = core::f64::consts::PI;
        // 3*pi -> pi (then -pi side). 3*pi mod 2*pi = pi, which is not > pi, so stays pi.
        let r = Eulerd::angle_mod(3.0 * pi);
        assert!((r as f64).abs() <= pi + 1e-5);
        // 5*pi/2 -> pi/2.
        let r = Eulerd::angle_mod(2.5 * pi) as f64;
        assert!((r - pi / 2.0).abs() < 1e-5);
        // The return type is f32 even for Eulerd.
        let _: f32 = Eulerd::angle_mod(1.0);
    }

    /// `makeNear` keeps the same orientation while bringing the angles near a
    /// target (verified by matrix equivalence, since makeNear must not change the
    /// orientation it represents).
    ///
    /// Tolerance note: `angleMod` returns `f32` even for `Eulerd` (a faithful C++
    /// quirk), so a makeNear on `f64` is only accurate to ~f32 epsilon. The
    /// threshold reflects that genuine precision limit of the 1:1 port.
    #[test]
    fn make_near_preserves_orientation() {
        let mut e = Eulerd::from_angles(0.3, -0.7, 1.1, Order::XYZ, InputLayout::IJKLayout);
        let before = e.to_matrix44();
        let target = Eulerd::from_angles(6.5, -7.0, 7.4, Order::XYZ, InputLayout::IJKLayout);
        e.make_near(&target);
        let after = e.to_matrix44();
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (before[r][c] - after[r][c]).abs() < 1e-5,
                    "makeNear changed orientation at [{r}][{c}]: {} vs {}",
                    before[r][c],
                    after[r][c]
                );
            }
        }
        // And the angles moved close to the target (mod the equivalence).
        let target_xyz = target.to_xyz_vector();
        let got = e.to_xyz_vector();
        // Each component must be within PI of the target after makeNear.
        let pi = core::f64::consts::PI;
        assert!((got.x - target_xyz.x).abs() <= pi + 1e-4);
        assert!((got.y - target_xyz.y).abs() <= pi + 1e-4);
        assert!((got.z - target_xyz.z).abs() <= pi + 1e-4);
    }

    /// Equality compares both angles and order.
    #[test]
    fn equality() {
        let a = Eulerd::from_angles(0.1, 0.2, 0.3, Order::XYZ, InputLayout::IJKLayout);
        let b = Eulerd::from_angles(0.1, 0.2, 0.3, Order::XYZ, InputLayout::IJKLayout);
        let c = Eulerd::from_angles(0.1, 0.2, 0.3, Order::ZYX, InputLayout::IJKLayout);
        assert!(a == b);
        assert!(a != c);
    }

    /// Display formatting matches the C++ `(x y z IJKr)` layout.
    #[test]
    fn display_format() {
        let e = Eulerf::from_angles(1.0, 2.0, 3.0, Order::XYZ, InputLayout::IJKLayout);
        let s = format!("{e}");
        assert!(s.ends_with("XYZ)"), "got: {s}");
        assert!(s.starts_with('('));
        // A relative order ends with the 'r' marker before the closing paren.
        let er = Eulerf::from_order(Order::XYZr);
        let sr = format!("{er}");
        assert!(sr.ends_with("r)"), "got: {sr}");
    }

    /// Float instantiation also builds and round-trips (covers `Eulerf`).
    #[test]
    fn eulerf_roundtrip() {
        let src = Eulerf::from_angles(0.3, -0.7, 1.1, Order::XYZ, InputLayout::IJKLayout);
        let m = src.to_matrix44();
        let mut back = Eulerf::from_order(Order::XYZ);
        back.extract44(&m);
        assert!((src.vec.x - back.vec.x).abs() < 1e-4);
        assert!((src.vec.y - back.vec.y).abs() < 1e-4);
        assert!((src.vec.z - back.vec.z).abs() < 1e-4);
    }
}
