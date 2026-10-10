//! Port of OpenEXR's `ImfCheckedArithmetic.h` — integer arithmetic that
//! reports overflow / underflow / division-by-zero as a `Result` instead of
//! wrapping (release) or panicking (debug).
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfCheckedArithmetic.h>
//!
//! The C++ header is a template over any unsigned integer type. Rust has no
//! direct equivalent without pulling in a numeric-trait crate (against the
//! "port everything ourselves, no external crates for format-intrinsic code"
//! rule for a five-function header), so [`checked_arithmetic_for`] is
//! instantiated by hand for the concrete types this port's call sites need —
//! `usize` (`ui_mult`/`ui_div`/`ui_add`/`ui_sub`) and `u32`
//! (`ui_mult_u32`/`ui_div_u32`/`ui_add_u32`/`ui_sub_u32`, needed by
//! [`crate::attr::preview_image::PreviewImage::new`], whose C++ counterpart
//! multiplies `unsigned int` dimensions). Adding another instantiation (e.g.
//! `u64`) is one macro invocation away.
//!
//! **Error mapping** (`IexMathExc.h`, verified): `uiMult`/`uiAdd`/
//! `checkArraySize` throw `OverflowExc` -> [`ExrError::Overflow`]; `uiDiv`
//! throws `DivzeroExc` -> [`ExrError::DivZero`]; `uiSub` throws `UnderflowExc`
//! -> [`ExrError::Underflow`]. All three are `MathExc` subclasses
//! (`IexClass::Math`), themselves `BaseExc` children — see
//! [`crate::error::IexClass`].

use crate::error::{ExrError, Result};

/// Instantiate `ui_mult`/`ui_div`/`ui_add`/`ui_sub` for one concrete unsigned
/// integer type, with the given function names. Each mirrors the identically
/// named C++ template function bit-for-bit (same overflow/underflow tests,
/// same order of operations), differing only in which `ExrError` variant a
/// failure produces.
macro_rules! checked_arithmetic_for {
    ($T:ty, $mult:ident, $div:ident, $add:ident, $sub:ident) => {
        /// Unsigned multiplication — port of `uiMult<T>`.
        ///
        /// # Errors
        /// [`ExrError::Overflow`] ("Integer multiplication overflow.") if
        /// `a * b` would overflow this instantiation's integer type.
        pub fn $mult(a: $T, b: $T) -> Result<$T> {
            if a > 0 && b > <$T>::MAX / a {
                return Err(ExrError::Overflow(
                    "Integer multiplication overflow.".into(),
                ));
            }
            Ok(a * b)
        }

        /// Unsigned division — port of `uiDiv<$T>`.
        ///
        /// # Errors
        /// [`ExrError::DivZero`] ("Integer division by zero.") if `b == 0`.
        pub fn $div(a: $T, b: $T) -> Result<$T> {
            if b == 0 {
                return Err(ExrError::DivZero("Integer division by zero.".into()));
            }
            Ok(a / b)
        }

        /// Unsigned addition — port of `uiAdd<$T>`.
        ///
        /// # Errors
        /// [`ExrError::Overflow`] ("Integer addition overflow.") if `a + b`
        /// would overflow.
        pub fn $add(a: $T, b: $T) -> Result<$T> {
            if a > <$T>::MAX - b {
                return Err(ExrError::Overflow("Integer addition overflow.".into()));
            }
            Ok(a + b)
        }

        /// Unsigned subtraction — port of `uiSub<$T>`.
        ///
        /// # Errors
        /// [`ExrError::Underflow`] ("Integer subtraction underflow.") if
        /// `a < b`.
        pub fn $sub(a: $T, b: $T) -> Result<$T> {
            if a < b {
                return Err(ExrError::Underflow(
                    "Integer subtraction underflow.".into(),
                ));
            }
            Ok(a - b)
        }
    };
}

checked_arithmetic_for!(usize, ui_mult, ui_div, ui_add, ui_sub);
checked_arithmetic_for!(u32, ui_mult_u32, ui_div_u32, ui_add_u32, ui_sub_u32);

/// Port of `checkArraySize<T>(n, s)`: verifies that `n` elements of `s` bytes
/// each can be sized without overflowing `usize` (the C++ counterpart guards
/// a `new T[n]`-style allocation of `size_t(n) * s` bytes), and returns `n`
/// unchanged — matching C++, which returns the (`size_t`-cast) element
/// count, not the byte count.
///
/// `s == 0` (a zero-sized element type) trivially never overflows and is
/// **not** the division-by-zero case `ExrError::DivZero` describes — that
/// variant is reserved for [`ui_div`]'s own C++ counterpart (`uiDiv`), a
/// different function `checkArraySize` never calls. Guarding it here (C++
/// has no such guard — `size_t` division by a `sizeof` expression is never
/// zero in practice) keeps this a hard error rather than a Rust integer-divide
/// panic, honouring the crate's no-panics rule for any future zero-sized
/// caller.
///
/// # Errors
/// [`ExrError::Overflow`] ("Integer multiplication overflow.") if `n * s`
/// would overflow `usize`.
pub fn check_array_size(n: usize, s: usize) -> Result<usize> {
    if s != 0 && n > usize::MAX / s {
        return Err(ExrError::Overflow(
            "Integer multiplication overflow.".into(),
        ));
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_mult_overflow_is_overflow_class() {
        assert!(matches!(
            ui_mult(usize::MAX, 2),
            Err(ExrError::Overflow(_))
        ));
        assert_eq!(ui_mult(3, 4).unwrap(), 12);
        // C++ `uiMult` short-circuits the `a > 0` check on `a == 0`.
        assert_eq!(ui_mult(0, usize::MAX).unwrap(), 0);
    }

    #[test]
    fn ui_div_by_zero_is_divzero_class() {
        assert!(matches!(ui_div(10, 0), Err(ExrError::DivZero(_))));
        assert_eq!(ui_div(10, 3).unwrap(), 3);
    }

    #[test]
    fn ui_add_overflow_is_overflow_class() {
        assert!(matches!(
            ui_add(usize::MAX, 1),
            Err(ExrError::Overflow(_))
        ));
        assert_eq!(ui_add(2, 3).unwrap(), 5);
    }

    #[test]
    fn ui_sub_underflow_is_underflow_class() {
        assert!(matches!(ui_sub(1, 2), Err(ExrError::Underflow(_))));
        assert_eq!(ui_sub(5, 3).unwrap(), 2);
    }

    #[test]
    fn u32_instantiation_matches_usize_semantics() {
        assert!(matches!(
            ui_mult_u32(u32::MAX, 2),
            Err(ExrError::Overflow(_))
        ));
        assert_eq!(ui_mult_u32(3, 4).unwrap(), 12);
        assert!(matches!(ui_div_u32(1, 0), Err(ExrError::DivZero(_))));
        assert!(matches!(
            ui_add_u32(u32::MAX, 1),
            Err(ExrError::Overflow(_))
        ));
        assert!(matches!(ui_sub_u32(0, 1), Err(ExrError::Underflow(_))));
    }

    #[test]
    fn check_array_size_overflow_is_overflow_class() {
        assert!(matches!(
            check_array_size(usize::MAX, 2),
            Err(ExrError::Overflow(_))
        ));
        assert_eq!(check_array_size(10, 4).unwrap(), 10);
        // Not a DivZero case — see the function doc.
        assert_eq!(check_array_size(10, 0).unwrap(), 10);
    }

    #[test]
    fn every_math_error_is_a_base_and_math_error() {
        use crate::error::IexClass;
        for e in [
            ExrError::Overflow("x".into()),
            ExrError::DivZero("x".into()),
            ExrError::Underflow("x".into()),
        ] {
            assert!(e.is_a(IexClass::Math));
            assert!(e.is_a(IexClass::Base));
        }
    }
}
