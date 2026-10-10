//! Port of Imath halfLimits.h (and the `std::numeric_limits<half>` specialization
//! that now lives in half.h).
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/halfLimits.h>
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/half.h>
//!
//! `halfLimits.h` itself is now a deprecated shim that merely re-includes `half.h`
//! (the actual `std::numeric_limits<half>` specialization was moved *into* `half.h`,
//! "because [the limits] should be regarded as inseparable from the half class").
//! Rust has no `std::numeric_limits`; the faithful port is to expose every member of
//! that specialization as associated constants / functions on [`HalfLimits`], a
//! zero-sized stand-in for `numeric_limits<half>`. The float-round-style and
//! denorm-style enums are ported as small Rust enums matching `<limits>` values.

use super::Half;
use super::{
    HALF_DECIMAL_DIG, HALF_DENORM_MIN_10_EXP, HALF_DENORM_MIN_EXP, HALF_DIG, HALF_MANT_DIG,
    HALF_MAX_10_EXP, HALF_MAX_EXP, HALF_RADIX,
};

/// Port of `std::float_round_style`: rounding behaviour of a floating type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatRoundStyle {
    /// `round_indeterminate` (-1).
    RoundIndeterminate,
    /// `round_toward_zero` (0).
    RoundTowardZero,
    /// `round_to_nearest` (1).
    RoundToNearest,
    /// `round_toward_infinity` (2).
    RoundTowardInfinity,
    /// `round_toward_neg_infinity` (3).
    RoundTowardNegInfinity,
}

/// Port of `std::float_denorm_style`: subnormal support of a floating type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatDenormStyle {
    /// `denorm_indeterminate` (-1).
    DenormIndeterminate,
    /// `denorm_absent` (0).
    DenormAbsent,
    /// `denorm_present` (1).
    DenormPresent,
}

/// Port of `std::numeric_limits<IMATH_INTERNAL_NAMESPACE::half>`.
///
/// A zero-sized type carrying every member of the C++ specialization as associated
/// constants (the `static constexpr` fields) and associated functions (the
/// `static constexpr half ...()` accessors, which cannot be `const` values in Rust
/// because [`Half`] is constructed at call sites). All bit patterns are taken
/// verbatim from the C++ `FromBits` literals.
pub struct HalfLimits;

impl HalfLimits {
    /// `is_specialized = true`.
    pub const IS_SPECIALIZED: bool = true;

    /// `min()` — smallest positive normalized half (`HALF_MIN`, bits `0x0400`).
    #[inline]
    pub const fn min() -> Half {
        Half::from_bits(0x0400)
    }

    /// `max()` — largest finite half (`HALF_MAX`, bits `0x7bff`).
    #[inline]
    pub const fn max() -> Half {
        Half::from_bits(0x7bff)
    }

    /// `lowest()` — most negative finite half (`-HALF_MAX`, bits `0xfbff`).
    #[inline]
    pub const fn lowest() -> Half {
        Half::from_bits(0xfbff)
    }

    /// `digits = HALF_MANT_DIG`.
    pub const DIGITS: i32 = HALF_MANT_DIG;
    /// `digits10 = HALF_DIG`.
    pub const DIGITS10: i32 = HALF_DIG;
    /// `max_digits10 = HALF_DECIMAL_DIG`.
    pub const MAX_DIGITS10: i32 = HALF_DECIMAL_DIG;
    /// `is_signed = true`.
    pub const IS_SIGNED: bool = true;
    /// `is_integer = false`.
    pub const IS_INTEGER: bool = false;
    /// `is_exact = false`.
    pub const IS_EXACT: bool = false;
    /// `radix = HALF_RADIX`.
    pub const RADIX: i32 = HALF_RADIX;

    /// `epsilon()` — `HALF_EPSILON` (bits `0x1400`).
    #[inline]
    pub const fn epsilon() -> Half {
        Half::from_bits(0x1400)
    }

    /// `round_error()` — `0.5` (bits `0x3800`).
    #[inline]
    pub const fn round_error() -> Half {
        Half::from_bits(0x3800)
    }

    /// `min_exponent = HALF_DENORM_MIN_EXP`.
    pub const MIN_EXPONENT: i32 = HALF_DENORM_MIN_EXP;
    /// `min_exponent10 = HALF_DENORM_MIN_10_EXP`.
    pub const MIN_EXPONENT10: i32 = HALF_DENORM_MIN_10_EXP;
    /// `max_exponent = HALF_MAX_EXP`.
    pub const MAX_EXPONENT: i32 = HALF_MAX_EXP;
    /// `max_exponent10 = HALF_MAX_10_EXP`.
    pub const MAX_EXPONENT10: i32 = HALF_MAX_10_EXP;

    /// `has_infinity = true`.
    pub const HAS_INFINITY: bool = true;
    /// `has_quiet_NaN = true`.
    pub const HAS_QUIET_NAN: bool = true;
    /// `has_signaling_NaN = true`.
    pub const HAS_SIGNALING_NAN: bool = true;
    /// `has_denorm = denorm_present`.
    pub const HAS_DENORM: FloatDenormStyle = FloatDenormStyle::DenormPresent;
    /// `has_denorm_loss = false`.
    pub const HAS_DENORM_LOSS: bool = false;

    /// `infinity()` — `+inf` (bits `0x7c00`, == `half::posInf()`).
    #[inline]
    pub const fn infinity() -> Half {
        Half::from_bits(0x7c00)
    }

    /// `quiet_NaN()` — qNaN (bits `0x7fff`, == `half::qNan()`).
    #[inline]
    pub const fn quiet_nan() -> Half {
        Half::from_bits(0x7fff)
    }

    /// `signaling_NaN()` — sNaN (bits `0x7dff`, == `half::sNan()`).
    #[inline]
    pub const fn signaling_nan() -> Half {
        Half::from_bits(0x7dff)
    }

    /// `denorm_min()` — smallest positive denormal (bits `0x0001`).
    #[inline]
    pub const fn denorm_min() -> Half {
        Half::from_bits(0x0001)
    }

    /// `is_iec559 = false`.
    pub const IS_IEC559: bool = false;
    /// `is_bounded = false`.
    pub const IS_BOUNDED: bool = false;
    /// `is_modulo = false`.
    pub const IS_MODULO: bool = false;

    /// `traps = true`.
    pub const TRAPS: bool = true;
    /// `tinyness_before = false`.
    pub const TINYNESS_BEFORE: bool = false;
    /// `round_style = round_to_nearest`.
    pub const ROUND_STYLE: FloatRoundStyle = FloatRoundStyle::RoundToNearest;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_bit_patterns_and_flags() {
        assert_eq!(HalfLimits::min().bits(), 0x0400);
        assert_eq!(HalfLimits::max().bits(), 0x7bff);
        assert_eq!(HalfLimits::lowest().bits(), 0xfbff);
        assert_eq!(HalfLimits::epsilon().bits(), 0x1400);
        assert_eq!(HalfLimits::round_error().bits(), 0x3800);
        assert_eq!(HalfLimits::infinity().bits(), 0x7c00);
        assert_eq!(HalfLimits::quiet_nan().bits(), 0x7fff);
        assert_eq!(HalfLimits::signaling_nan().bits(), 0x7dff);
        assert_eq!(HalfLimits::denorm_min().bits(), 0x0001);

        // Bind the `bool` consts to locals so the assertions are not constant
        // expressions (avoids `clippy::assertions_on_constants`) while still
        // verifying each member of the `numeric_limits<half>` specialization.
        let (is_specialized, is_signed, is_integer) =
            (HalfLimits::IS_SPECIALIZED, HalfLimits::IS_SIGNED, HalfLimits::IS_INTEGER);
        assert!(is_specialized);
        assert!(is_signed);
        assert!(!is_integer);
        assert_eq!(HalfLimits::DIGITS, 11);
        assert_eq!(HalfLimits::DIGITS10, 3);
        assert_eq!(HalfLimits::MAX_DIGITS10, 5);
        assert_eq!(HalfLimits::RADIX, 2);
        assert_eq!(HalfLimits::MIN_EXPONENT, -13);
        assert_eq!(HalfLimits::MAX_EXPONENT, 16);
        assert_eq!(HalfLimits::MIN_EXPONENT10, -4);
        assert_eq!(HalfLimits::MAX_EXPONENT10, 4);
        assert_eq!(HalfLimits::HAS_DENORM, FloatDenormStyle::DenormPresent);
        assert_eq!(HalfLimits::ROUND_STYLE, FloatRoundStyle::RoundToNearest);
        let (has_inf, has_qnan, has_snan, is_iec559, traps) = (
            HalfLimits::HAS_INFINITY,
            HalfLimits::HAS_QUIET_NAN,
            HalfLimits::HAS_SIGNALING_NAN,
            HalfLimits::IS_IEC559,
            HalfLimits::TRAPS,
        );
        assert!(has_inf);
        assert!(has_qnan);
        assert!(has_snan);
        assert!(!is_iec559);
        assert!(traps);

        // min() / max() / lowest() must be the documented values.
        assert_eq!(HalfLimits::min().to_f32(), super::super::HALF_MIN);
        assert_eq!(HalfLimits::max().to_f32(), super::super::HALF_MAX);
    }
}
