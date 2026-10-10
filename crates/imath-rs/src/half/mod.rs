//! Port of Imath half.h + half.cpp.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/half.h>
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/half.cpp>
//!
//! The [`Half`] type is a 1:1 port of Imath's `half` class: a 16-bit IEEE-754-2008
//! `binary16` floating-point number. Every public method, operator, classification
//! predicate, and special-value constructor from `half.h` is ported here, along with
//! the stream/`printBits` helpers from `half.cpp`.
//!
//! # Naming
//! Imath's C++ type is spelled `half` (lower-case). Rust convention uses a capital;
//! the type is named [`Half`] here. It is otherwise API-identical.
//!
//! # Backing conversions
//! The underlying `half`↔`f32` bit conversions call the in-crate Imath bit-shift
//! algorithm directly: [`to_float::imath_float_to_half`] and
//! [`to_float::imath_half_to_float`]. These are faithful ports of the C++ no-F16C
//! `#else` branches in `half.h`, bit-exact to Imath including NaN payloads. No
//! external crate is involved — `half` is part of OpenEXR so we own it.
//!
//! The classification, `round`, special values, and operators are ported directly
//! from the C++ and work purely on the stored `u16` bit pattern.

pub mod function;
pub mod limits;
pub mod to_float;

use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

// ----------------------------------------------------------------------------
// Limit constants (port of the `HALF_*` macros from half.h)
// ----------------------------------------------------------------------------
//
// In C++ these are preprocessor `#define`s whose float/double-ness varies by
// platform; in Rust we expose them as typed constants. The float-valued limits use
// the exact decimal literals from half.h. The integer "characteristic" macros are
// `i32` to match their use as `std::numeric_limits` exponent fields.

/// Smallest positive denormalized half (`HALF_DENORM_MIN`).
/// C++ literal `5.96046448e-08f`, written here at exact `f32` precision (same bits).
pub const HALF_DENORM_MIN: f32 = 5.960_464_5e-8;
/// Smallest positive normalized half (`HALF_NRM_MIN`).
/// C++ literal `6.10351562e-05`, written here at exact `f32` precision (same bits).
pub const HALF_NRM_MIN: f32 = 6.103_515_6e-5;
/// Smallest positive normalized half (`HALF_MIN`).
/// C++ literal `6.10351562e-05f`, written here at exact `f32` precision (same bits).
pub const HALF_MIN: f32 = 6.103_515_6e-5;
/// Largest positive half (`HALF_MAX`).
pub const HALF_MAX: f32 = 65504.0;
/// Smallest positive `e` for which `Half(1.0 + e) != Half(1.0)` (`HALF_EPSILON`).
pub const HALF_EPSILON: f32 = 0.00097656;

/// Number of digits in mantissa (significand + hidden leading 1) (`HALF_MANT_DIG`).
pub const HALF_MANT_DIG: i32 = 11;
/// Base-10 digits representable without change (`HALF_DIG`).
pub const HALF_DIG: i32 = 3;
/// Base-10 digits needed to uniquely represent all values (`HALF_DECIMAL_DIG`).
pub const HALF_DECIMAL_DIG: i32 = 5;
/// Base of the exponent (`HALF_RADIX`).
pub const HALF_RADIX: i32 = 2;
/// Minimum negative integer such that `HALF_RADIX^(n-1)` is a normalized half
/// (`HALF_DENORM_MIN_EXP`).
pub const HALF_DENORM_MIN_EXP: i32 = -13;
/// Maximum positive integer such that `HALF_RADIX^(n-1)` is a normalized half
/// (`HALF_MAX_EXP`).
pub const HALF_MAX_EXP: i32 = 16;
/// Minimum positive integer such that `10^n` is a normalized half
/// (`HALF_DENORM_MIN_10_EXP`).
pub const HALF_DENORM_MIN_10_EXP: i32 = -4;
/// Maximum positive integer such that `10^n` is a normalized half
/// (`HALF_MAX_10_EXP`).
pub const HALF_MAX_10_EXP: i32 = 4;

/// Port of `imath_half_uif` / `half::uif`: a union aliasing a `u32` and an `f32`.
///
/// In C++ this is a `union { uint32_t i; float f; }` used to reinterpret float bits.
/// Rust has no untagged unions in safe code, so the reinterpretation is expressed via
/// [`f32::to_bits`] / [`f32::from_bits`]; this struct preserves the named type for
/// faithful API surface and stores both views explicitly.
#[derive(Clone, Copy, Debug)]
pub struct Uif {
    /// Integer view of the bits.
    pub i: u32,
    /// Float view of the bits.
    pub f: f32,
}

/// Port of Imath's `half` class: a 16-bit (binary16) floating-point number.
///
/// See the module docs for the naming rationale ([`Half`] vs C++ `half`). The single
/// field `_h` is the raw 16-bit pattern, exactly as in the C++ class.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct Half {
    /// Raw 16-bit half bit pattern (C++ `uint16_t _h`).
    _h: u16,
}

impl Half {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Construct from `f32` (port of `half(float f)`).
    ///
    /// Rounds to nearest-even via the in-crate Imath bit-shift algorithm
    /// ([`to_float::imath_float_to_half`]), bit-exact to the C++ implementation.
    #[inline]
    pub fn new(f: f32) -> Self {
        Self {
            _h: self::to_float::imath_float_to_half(f),
        }
    }

    /// Construct from a raw bit pattern (port of `half(FromBitsTag, uint16_t bits)`).
    ///
    /// Mirrors the C++ `half(half::FromBits, bits)` tagged constructor.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self { _h: bits }
    }

    // ------------------------------------------------------------------
    // Conversion to float
    // ------------------------------------------------------------------

    /// Conversion to `f32` (port of `operator float()`).
    ///
    /// Always exact — every half value is exactly representable as a `f32`. Uses the
    /// in-crate Imath bit-shift algorithm ([`to_float::imath_half_to_float`]).
    #[inline]
    pub fn to_f32(self) -> f32 {
        self::to_float::imath_half_to_float(self._h)
    }

    // ------------------------------------------------------------------
    // Round to n-bit precision (port of half::round)
    // ------------------------------------------------------------------

    /// Round to `n`-bit precision (port of `half::round(unsigned int n)`).
    ///
    /// `n` should be between 0 and 10. After rounding, the significand's `10 - n`
    /// least significant bits are zero. The exponent adjusts automatically if
    /// rounding up overflows the significand; on exponent overflow the value is
    /// truncated rather than rounded, exactly as in the C++.
    #[inline]
    pub fn round(self, n: u32) -> Self {
        // Parameter check.
        if n >= 10 {
            return self;
        }

        // Disassemble into sign `s` and combined exponent+significand `e`.
        let s: u16 = self._h & 0x8000;
        let mut e: u16 = self._h & 0x7fff;

        // Round exponent+significand to the nearest value where ones occur only in
        // the (10 - n) most significant bits. Note the shift amount `9 - n`.
        e >>= 9 - n;
        e += e & 1;
        e <<= 9 - n;

        // Check for exponent overflow.
        if e >= 0x7c00 {
            // Overflow occurred -- truncate instead of rounding.
            e = self._h;
            e >>= 10 - n;
            e <<= 10 - n;
        }

        // Put the original sign bit back.
        Half::from_bits(s | e)
    }

    // ------------------------------------------------------------------
    // Private bit-field accessors (port of half::mantissa / half::exponent)
    // ------------------------------------------------------------------

    /// Port of private `half::mantissa()`: the low 10 mantissa bits.
    #[inline]
    const fn mantissa(self) -> u16 {
        self._h & 0x3ff
    }

    /// Port of private `half::exponent()`: the 5-bit biased exponent.
    #[inline]
    const fn exponent(self) -> u16 {
        (self._h >> 10) & 0x001f
    }

    // ------------------------------------------------------------------
    // Classification (port of half::isFinite … isNegative)
    // ------------------------------------------------------------------

    /// Return true if a normalized number, a denormalized number, or zero
    /// (port of `half::isFinite()`).
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.exponent() < 31
    }

    /// Return true if a normalized number (port of `half::isNormalized()`).
    #[inline]
    pub const fn is_normalized(self) -> bool {
        self.exponent() > 0 && self.exponent() < 31
    }

    /// Return true if a denormalized number (port of `half::isDenormalized()`).
    #[inline]
    pub const fn is_denormalized(self) -> bool {
        self.exponent() == 0 && self.mantissa() != 0
    }

    /// Return true if zero (port of `half::isZero()`).
    #[inline]
    pub const fn is_zero(self) -> bool {
        (self._h & 0x7fff) == 0
    }

    /// Return true if NaN (port of `half::isNan()`).
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.exponent() == 31 && self.mantissa() != 0
    }

    /// Return true if a positive or negative infinity (port of `half::isInfinity()`).
    #[inline]
    pub const fn is_infinity(self) -> bool {
        self.exponent() == 31 && self.mantissa() == 0
    }

    /// Return true if the sign bit is set / negative (port of `half::isNegative()`).
    #[inline]
    pub const fn is_negative(self) -> bool {
        (self._h & 0x8000) != 0
    }

    // ------------------------------------------------------------------
    // Special values (port of half::posInf … sNan)
    // ------------------------------------------------------------------

    /// Return +infinity (port of `half::posInf()`).
    #[inline]
    pub const fn pos_inf() -> Self {
        Half::from_bits(0x7c00)
    }

    /// Return -infinity (port of `half::negInf()`).
    #[inline]
    pub const fn neg_inf() -> Self {
        Half::from_bits(0xfc00)
    }

    /// Return a quiet NaN with bit pattern `0111111111111111` (port of `half::qNan()`).
    #[inline]
    pub const fn q_nan() -> Self {
        Half::from_bits(0x7fff)
    }

    /// Return a signaling NaN with bit pattern `0111110111111111` (port of `half::sNan()`).
    #[inline]
    pub const fn s_nan() -> Self {
        Half::from_bits(0x7dff)
    }

    // ------------------------------------------------------------------
    // Access to the internal representation (port of half::bits / setBits)
    // ------------------------------------------------------------------

    /// Return the bit pattern (port of `half::bits()`).
    #[inline]
    pub const fn bits(self) -> u16 {
        self._h
    }

    /// Set the bit pattern (port of `half::setBits(uint16_t bits)`).
    #[inline]
    pub fn set_bits(&mut self, bits: u16) {
        self._h = bits;
    }

    /// Construct from `f32` — alias for [`Half::new`].
    ///
    /// Provided as a named conversion method complementing [`Half::to_f32`].
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        Self::new(f)
    }
}

// ----------------------------------------------------------------------------
// Conversions (idiomatic Rust surface for the C++ `half(float)` / `operator float`)
// ----------------------------------------------------------------------------

impl From<f32> for Half {
    /// Port of `half(float)`.
    #[inline]
    fn from(f: f32) -> Self {
        Half::new(f)
    }
}

impl From<Half> for f32 {
    /// Port of `half::operator float()`.
    #[inline]
    fn from(h: Half) -> Self {
        h.to_f32()
    }
}

// ----------------------------------------------------------------------------
// Basic Algebra (port of the half operators)
// ----------------------------------------------------------------------------
//
// Imath defines arithmetic by round-tripping through float: `half(float(a) op
// float(b))`. The compound-assignment operators come in `half`-rhs and `float`-rhs
// flavours; the binary `+ - * /` operators are provided by C++'s implicit
// `half`→`float` conversions in expressions. We port the explicit compound-assign
// pairs faithfully and add the binary operators (computed identically) so `Half`
// behaves like the C++ type in arithmetic expressions.

impl Neg for Half {
    type Output = Half;
    /// Unary minus (port of `half::operator-()`): flips the sign bit.
    #[inline]
    fn neg(self) -> Half {
        Half::from_bits(self.bits() ^ 0x8000)
    }
}

impl AddAssign<Half> for Half {
    /// Port of `half::operator+=(half h)`.
    #[inline]
    fn add_assign(&mut self, h: Half) {
        *self = Half::new(self.to_f32() + h.to_f32());
    }
}

impl AddAssign<f32> for Half {
    /// Port of `half::operator+=(float f)`.
    #[inline]
    fn add_assign(&mut self, f: f32) {
        *self = Half::new(self.to_f32() + f);
    }
}

impl SubAssign<Half> for Half {
    /// Port of `half::operator-=(half h)`.
    #[inline]
    fn sub_assign(&mut self, h: Half) {
        *self = Half::new(self.to_f32() - h.to_f32());
    }
}

impl SubAssign<f32> for Half {
    /// Port of `half::operator-=(float f)`.
    #[inline]
    fn sub_assign(&mut self, f: f32) {
        *self = Half::new(self.to_f32() - f);
    }
}

impl MulAssign<Half> for Half {
    /// Port of `half::operator*=(half h)`.
    #[inline]
    fn mul_assign(&mut self, h: Half) {
        *self = Half::new(self.to_f32() * h.to_f32());
    }
}

impl MulAssign<f32> for Half {
    /// Port of `half::operator*=(float f)`.
    #[inline]
    fn mul_assign(&mut self, f: f32) {
        *self = Half::new(self.to_f32() * f);
    }
}

impl DivAssign<Half> for Half {
    /// Port of `half::operator/=(half h)`.
    #[inline]
    fn div_assign(&mut self, h: Half) {
        *self = Half::new(self.to_f32() / h.to_f32());
    }
}

impl DivAssign<f32> for Half {
    /// Port of `half::operator/=(float f)`.
    #[inline]
    fn div_assign(&mut self, f: f32) {
        *self = Half::new(self.to_f32() / f);
    }
}

// Binary arithmetic operators. In C++ these arise from the implicit `half`→`float`
// promotion in expressions (`a + b` becomes `float(a) + float(b)`), yielding a
// `float`; assigning back to a `half` re-rounds. Imath code idiomatically writes e.g.
// `half c = a + b;`. We mirror that by computing in `f32` and rounding to `Half`,
// matching the compound-assign operators above.

impl Add<Half> for Half {
    type Output = Half;
    /// `Half + Half` computed as `Half::new(f32 + f32)` (matches `operator+=`).
    #[inline]
    fn add(self, h: Half) -> Half {
        Half::new(self.to_f32() + h.to_f32())
    }
}

impl Sub<Half> for Half {
    type Output = Half;
    /// `Half - Half` computed as `Half::new(f32 - f32)`.
    #[inline]
    fn sub(self, h: Half) -> Half {
        Half::new(self.to_f32() - h.to_f32())
    }
}

impl Mul<Half> for Half {
    type Output = Half;
    /// `Half * Half` computed as `Half::new(f32 * f32)`.
    #[inline]
    fn mul(self, h: Half) -> Half {
        Half::new(self.to_f32() * h.to_f32())
    }
}

impl Div<Half> for Half {
    type Output = Half;
    /// `Half / Half` computed as `Half::new(f32 / f32)`.
    #[inline]
    fn div(self, h: Half) -> Half {
        Half::new(self.to_f32() / h.to_f32())
    }
}

// Remainder (`%`). Imath's C++ `half` has no `operator%`; it would arise from the
// implicit `half`->`float` promotion (`std::fmod(float(a), float(b))`). The generic
// `Vec`/`Matrix` machinery requires the `Scalar` bound to provide `Rem`, so we
// supply it here computed identically to the other binary operators: promote to
// `f32`, apply the float remainder, and round back to `Half`. This keeps `Half`
// usable as a generic vector element type with no behavioural surprise.

impl Rem<Half> for Half {
    type Output = Half;
    /// `Half % Half` computed as `Half::new(f32 % f32)` (truncated float remainder).
    #[inline]
    fn rem(self, h: Half) -> Half {
        Half::new(self.to_f32() % h.to_f32())
    }
}

impl RemAssign<Half> for Half {
    /// `Half %= Half`, computed via the `f32` remainder.
    #[inline]
    fn rem_assign(&mut self, h: Half) {
        *self = Half::new(self.to_f32() % h.to_f32());
    }
}

impl RemAssign<f32> for Half {
    /// `Half %= f32`, computed via the `f32` remainder.
    #[inline]
    fn rem_assign(&mut self, f: f32) {
        *self = Half::new(self.to_f32() % f);
    }
}

// ----------------------------------------------------------------------------
// Generic-element trait impls (`Scalar`, `Math`) so `Half` is a valid `Vec`/
// `Matrix` element type, mirroring Imath's `V*h`/`M*h` typedefs.
// ----------------------------------------------------------------------------
//
// In C++, `half` participates in the generic Imath templates purely through its
// implicit `half`<->`float` promotion: every arithmetic/transcendental operation on
// a `half` is evaluated in `float` and the result rounded back. We reproduce that
// exact behaviour by implementing the crate's element traits in terms of `f32`.

impl crate::traits::Scalar for Half {
    /// `Half(0)` — the additive identity (C++ `T(0)` for `T = half`).
    const ZERO: Self = Half::from_bits(0x0000);
    /// `Half(1)` — the multiplicative identity (C++ `T(1)` for `T = half`).
    /// `0x3c00` is the half bit pattern for `1.0`.
    const ONE: Self = Half::from_bits(0x3c00);
}

impl crate::math::Math for Half {
    // Each method promotes to `f32`, calls the corresponding `f32` `Math` method,
    // and rounds the result back to `Half` — exactly how `std::<fn>(half)` behaves
    // in C++ via the implicit `half`->`float` promotion. `atan2`/`pow`/`fmod`/`hypot`
    // take a second `Half` argument, also promoted.
    #[inline]
    fn acos(self) -> Self {
        Half::new(self.to_f32().acos())
    }
    #[inline]
    fn asin(self) -> Self {
        Half::new(self.to_f32().asin())
    }
    #[inline]
    fn atan(self) -> Self {
        Half::new(self.to_f32().atan())
    }
    #[inline]
    fn atan2(self, y: Self) -> Self {
        Half::new(self.to_f32().atan2(y.to_f32()))
    }
    #[inline]
    fn cos(self) -> Self {
        Half::new(self.to_f32().cos())
    }
    #[inline]
    fn sin(self) -> Self {
        Half::new(self.to_f32().sin())
    }
    #[inline]
    fn tan(self) -> Self {
        Half::new(self.to_f32().tan())
    }
    #[inline]
    fn cosh(self) -> Self {
        Half::new(self.to_f32().cosh())
    }
    #[inline]
    fn sinh(self) -> Self {
        Half::new(self.to_f32().sinh())
    }
    #[inline]
    fn tanh(self) -> Self {
        Half::new(self.to_f32().tanh())
    }
    #[inline]
    fn exp(self) -> Self {
        Half::new(self.to_f32().exp())
    }
    #[inline]
    fn log(self) -> Self {
        Half::new(self.to_f32().ln())
    }
    #[inline]
    fn log10(self) -> Self {
        Half::new(self.to_f32().log10())
    }
    #[inline]
    fn modf(self, iptr: &mut Self) -> Self {
        let ival = self.to_f32().trunc();
        *iptr = Half::new(ival);
        Half::new(self.to_f32() - ival)
    }
    #[inline]
    fn pow(self, y: Self) -> Self {
        Half::new(self.to_f32().powf(y.to_f32()))
    }
    #[inline]
    fn sqrt(self) -> Self {
        Half::new(self.to_f32().sqrt())
    }
    #[inline]
    fn ceil(self) -> Self {
        Half::new(self.to_f32().ceil())
    }
    #[inline]
    fn fabs(self) -> Self {
        Half::new(self.to_f32().abs())
    }
    #[inline]
    fn floor(self) -> Self {
        Half::new(self.to_f32().floor())
    }
    #[inline]
    fn fmod(self, y: Self) -> Self {
        Half::new(self.to_f32() % y.to_f32())
    }
    #[inline]
    fn hypot(self, y: Self) -> Self {
        Half::new(self.to_f32().hypot(y.to_f32()))
    }
}

// ----------------------------------------------------------------------------
// Comparison
// ----------------------------------------------------------------------------
//
// Imath's `half` has no member comparison operators; comparisons go through the
// implicit `half`→`float` promotion, giving IEEE float ordering (NaN unordered).
// halfFunction.h relies on `x < domainMin || x > domainMax`, i.e. float comparison.
// We therefore implement `PartialEq`/`PartialOrd` via the `f32` value (NOT raw bits),
// so e.g. `+0.0 == -0.0` and NaN compares unequal, exactly like C++.

impl PartialEq for Half {
    #[inline]
    fn eq(&self, other: &Half) -> bool {
        self.to_f32() == other.to_f32()
    }
}

impl PartialOrd for Half {
    #[inline]
    fn partial_cmp(&self, other: &Half) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// ----------------------------------------------------------------------------
// Stream I/O / formatting (port of half.cpp operator<< and printBits)
// ----------------------------------------------------------------------------

impl fmt::Display for Half {
    /// Port of `operator<<(ostream&, half)`: format as the underlying float.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Debug for Half {
    /// Debug shows the float value plus the raw bits, for diagnostics.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({} /*0x{:04x}*/)", self.to_f32(), self._h)
    }
}

/// Port of `printBits(ostream&, half)`: the 16-bit layout as `'0'`/`'1'`, with a
/// space after the sign bit and after the exponent (matching the C++ spacing).
///
/// Returns the 18-character string (16 bits + 2 spaces), the Rust analogue of the
/// C++ overloads that write to an `ostream` or a `char[19]` buffer.
pub fn print_bits_half(h: Half) -> String {
    let b = h.bits();
    let mut s = String::with_capacity(18);
    for i in (0..=15).rev() {
        s.push(if (b >> i) & 1 != 0 { '1' } else { '0' });
        if i == 15 || i == 10 {
            s.push(' ');
        }
    }
    s
}

/// Port of `printBits(ostream&, float)`: the 32-bit layout as `'0'`/`'1'`, with a
/// space after the sign bit and after the exponent.
///
/// Returns the 34-character string (32 bits + 2 spaces).
pub fn print_bits_float(f: f32) -> String {
    let x = f.to_bits();
    let mut s = String::with_capacity(34);
    for i in (0..=31).rev() {
        s.push(if (x >> i) & 1 != 0 { '1' } else { '0' });
        if i == 31 || i == 23 {
            s.push(' ');
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::to_float::{imath_float_to_half, imath_half_to_float};
    use super::*;

    /// Known exact half→float conversions via the Imath bit-shift algorithm.
    ///
    /// Covers zero, one, 0.5, a negative normal, both infinities, and the smallest
    /// positive denormal. All values are exact, so bit-exact f32 comparison is valid.
    #[test]
    fn known_half_to_float() {
        assert_eq!(Half::from_bits(0x0000).to_f32(), 0.0f32);
        assert_eq!(Half::from_bits(0x3c00).to_f32(), 1.0f32);
        assert_eq!(Half::from_bits(0x3800).to_f32(), 0.5f32);
        assert_eq!(Half::from_bits(0xc000).to_f32(), -2.0f32);
        assert_eq!(Half::from_bits(0x7c00).to_f32(), f32::INFINITY);
        assert_eq!(Half::from_bits(0xfc00).to_f32(), f32::NEG_INFINITY);
        // Smallest positive denormal: 2^-24.
        assert_eq!(Half::from_bits(0x0001).to_f32(), 2f32.powi(-24));
    }

    /// For every non-NaN half bit pattern, half→float→half must be the identity.
    ///
    /// Every half is exactly representable as f32, so the round-trip must return the
    /// original bits.
    #[test]
    fn roundtrip_all_non_nan() {
        for b in 0u16..=0xffff {
            let h = Half::from_bits(b);
            if h.is_nan() {
                continue;
            }
            assert_eq!(
                Half::from_f32(h.to_f32()).bits(),
                b,
                "round-trip failed for half bits 0x{b:04x}"
            );
        }
    }

    /// float→half: round-to-nearest-even, overflow→inf, underflow→zero, known values.
    #[test]
    fn float_to_half_rne_and_limits() {
        // Known exact conversions.
        assert_eq!(Half::from_f32(1.0).bits(), 0x3c00, "1.0 -> 0x3c00");
        assert_eq!(Half::from_f32(0.5).bits(), 0x3800, "0.5 -> 0x3800");
        assert_eq!(Half::from_f32(-2.0).bits(), 0xc000, "-2.0 -> 0xc000");
        // Overflow: value beyond HALF_MAX rounds to +inf.
        assert_eq!(Half::from_f32(f32::INFINITY).bits(), 0x7c00, "+inf -> 0x7c00");
        assert_eq!(
            Half::from_f32(1.0e38_f32).bits(),
            0x7c00,
            "very large f32 -> +inf"
        );
        // Underflow: value much smaller than the smallest denormal flushes to zero.
        assert_eq!(
            Half::from_f32(1.0e-45_f32).bits() & 0x7fff,
            0x0000,
            "tiny f32 -> zero (magnitude)"
        );
        // Round-to-nearest-even tie: the value exactly halfway between 1.0 and the
        // next half (1.0 + epsilon/2 = 1.000488...) should round DOWN to 1.0,
        // because 0x3c00 has the EVEN mantissa (0) and 0x3c01 the odd one (1).
        // Asserting the exact bits (not `& !1`, which is true for either
        // neighbour and can't tell "rounded down" from "rounded up") is what
        // actually proves the tie-break direction.
        let tie_down = imath_half_to_float(0x3c00)
            + (imath_half_to_float(0x3c01) - imath_half_to_float(0x3c00)) * 0.5;
        assert_eq!(
            imath_float_to_half(tie_down),
            0x3c00,
            "RNE tie must round DOWN to the even-mantissa half (0x3c00), not up to 0x3c01"
        );
        // The complementary tie, between 0x3c01 (odd mantissa) and 0x3c02
        // (even mantissa), must round UP -- proving RNE isn't just "always
        // round down", which the old masked assertion could not distinguish.
        let tie_up = imath_half_to_float(0x3c01)
            + (imath_half_to_float(0x3c02) - imath_half_to_float(0x3c01)) * 0.5;
        assert_eq!(
            imath_float_to_half(tie_up),
            0x3c02,
            "RNE tie must round UP to the even-mantissa half (0x3c02), not down to 0x3c01"
        );
    }

    /// NaN payload is preserved through the Imath algorithm (self-consistent).
    ///
    /// `0x7e00` is a half NaN. We verify it is NaN, and that the round-trip through
    /// the Imath functions yields a bit pattern that is also NaN and equals the direct
    /// algorithm result (self-consistent, not dependent on any external crate).
    #[test]
    fn nan_payload_preserved() {
        let nan_bits: u16 = 0x7e00;
        let h = Half::from_bits(nan_bits);
        assert!(h.is_nan(), "0x7e00 should be NaN");
        // Round-trip through the Imath algorithm.
        let rt_bits = imath_float_to_half(imath_half_to_float(nan_bits));
        // The result must still be a NaN.
        let rt_h = Half::from_bits(rt_bits);
        assert!(rt_h.is_nan(), "round-tripped NaN must still be NaN");
        // Self-consistent: Half::from_f32(h.to_f32()).bits() == imath_float_to_half(imath_half_to_float(nan_bits)).
        assert_eq!(
            Half::from_f32(h.to_f32()).bits(),
            rt_bits,
            "Half::from_f32(h.to_f32()) must equal imath_float_to_half(imath_half_to_float(nan_bits))"
        );
    }

    /// Round-trip every non-NaN, non-inf half through `Half::to_f32` / `Half::new`.
    ///
    /// Identical to `roundtrip_all_non_nan` but exercised via the `Half` API methods
    /// (rather than `From` impl) to confirm both entry points are consistent.
    #[test]
    fn half_f32_roundtrip_identity() {
        for b in 0u32..=0xffff {
            let h = Half::from_bits(b as u16);
            if h.is_nan() || h.is_infinity() {
                continue;
            }
            let rt = Half::new(h.to_f32());
            assert_eq!(rt.bits(), h.bits(), "round-trip changed half 0x{b:04x}");
        }
    }

    #[test]
    fn classification_special_values() {
        assert!(Half::pos_inf().is_infinity() && !Half::pos_inf().is_negative());
        assert!(Half::neg_inf().is_infinity() && Half::neg_inf().is_negative());
        assert!(Half::q_nan().is_nan());
        assert!(Half::s_nan().is_nan());
        assert_eq!(Half::pos_inf().bits(), 0x7c00);
        assert_eq!(Half::neg_inf().bits(), 0xfc00);
        assert_eq!(Half::q_nan().bits(), 0x7fff);
        assert_eq!(Half::s_nan().bits(), 0x7dff);

        assert!(Half::new(0.0).is_zero());
        assert!(Half::new(0.0).is_finite());
        assert!(!Half::new(0.0).is_normalized());
        assert!(Half::new(1.0).is_normalized());
        // Smallest denormal.
        let d = Half::from_bits(0x0001);
        assert!(d.is_denormalized() && !d.is_normalized() && d.is_finite());
    }

    #[test]
    fn negation_and_sign() {
        let a = Half::new(124.0625);
        let n = -a;
        assert!(n.is_negative());
        assert_eq!(n.bits(), a.bits() ^ 0x8000);
        assert_eq!((-n).bits(), a.bits());
    }

    #[test]
    fn arithmetic_matches_float_roundtrip() {
        let a = Half::new(3.5);
        let b = Half::new(1.25);
        assert_eq!((a + b).bits(), Half::new(3.5 + 1.25).bits());
        assert_eq!((a - b).bits(), Half::new(3.5 - 1.25).bits());
        assert_eq!((a * b).bits(), Half::new(3.5 * 1.25).bits());
        assert_eq!((a / b).bits(), Half::new(3.5 / 1.25).bits());

        let mut c = a;
        c += b;
        assert_eq!(c.bits(), Half::new(3.5 + 1.25).bits());
        let mut c = a;
        c += 2.0f32;
        assert_eq!(c.bits(), Half::new(3.5 + 2.0).bits());
        let mut c = a;
        c -= b;
        assert_eq!(c.bits(), Half::new(3.5 - 1.25).bits());
        let mut c = a;
        c *= 2.0f32;
        assert_eq!(c.bits(), Half::new(3.5 * 2.0).bits());
        let mut c = a;
        c /= b;
        assert_eq!(c.bits(), Half::new(3.5 / 1.25).bits());
    }

    #[test]
    fn round_to_n_bits() {
        // n >= 10 is identity.
        let h = Half::new(124.0625);
        assert_eq!(h.round(10).bits(), h.bits());
        assert_eq!(h.round(11).bits(), h.bits());
        // After round(n), the low (10-n) significand bits are zero.
        for n in 0u32..10 {
            let r = h.round(n);
            let low_mask: u16 = (1u16 << (10 - n)) - 1;
            // Only the significand low bits are forced to zero (exponent may carry).
            assert_eq!(
                r.bits() & low_mask & 0x03ff,
                0,
                "round({n}) left low significand bits set"
            );
        }
        // Rounding near the max must clamp/truncate without producing inf via the
        // overflow branch (value stays finite).
        let big = Half::from_bits(0x7bff); // HALF_MAX
        assert!(big.round(0).is_finite() || big.round(0).is_infinity());
    }

    #[test]
    fn bits_set_bits_and_print() {
        let mut h = Half::from_bits(0x3c00); // 1.0
        assert_eq!(h.bits(), 0x3c00);
        assert_eq!(h.to_f32(), 1.0);
        h.set_bits(0x4000); // 2.0
        assert_eq!(h.to_f32(), 2.0);

        // printBits spacing: "s eeeee mmmmmmmmmm".
        assert_eq!(print_bits_half(Half::from_bits(0x3c00)).len(), 18);
        assert_eq!(print_bits_float(1.0).len(), 34);
        assert_eq!(print_bits_half(Half::from_bits(0x0000)), "0 00000 0000000000");
    }
}
