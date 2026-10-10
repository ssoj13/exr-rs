//! Port of Imath ImathPlatform.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathPlatform.h>
//!
//! `ImathPlatform.h` exists to paper over platform/compiler gaps in C++ (functions
//! and constants "which aren't provided by the system libraries, compilers, or
//! includes on certain platforms"). Most of its content is pure C preprocessor
//! compatibility machinery that has no Rust equivalent and needs none:
//!
//! - `IMATH_CPLUSPLUS_VERSION` — detects the C++ standard in use (`__cplusplus`).
//!   Rust has a fixed edition/toolchain; not applicable.
//! - `IMATH_CONSTEXPR14` — conditionally expands to `constexpr` only on C++14+.
//!   In Rust, `const fn` availability is a property of the toolchain, not a macro;
//!   not applicable.
//! - `IMATH_RESTRICT` — maps to the compiler's `__restrict` pointer-aliasing hint.
//!   Rust expresses non-aliasing through `&mut` and the borrow checker; not applicable.
//!
//! The only portable, value-bearing content is the math constants `M_PI` and
//! `M_PI_2`, defined here only when the platform's `<math.h>` lacks them. Rust's
//! [`core::f64::consts`] provides these, but Imath uses the literal digit strings
//! below; we reproduce the exact same literals so any numeric reference is 1:1.

/// Pi, matching Imath's `M_PI` literal in `ImathPlatform.h`.
///
/// Imath only `#define`s this when `<math.h>` does not already provide `M_PI`.
/// Imath's literal `3.14159265358979323846` rounds to exactly the same `f64` as
/// Rust's [`core::f64::consts::PI`] (verified in the test below), so we expose the
/// std constant directly for parity (and to satisfy clippy's `approx_constant`).
pub const M_PI: f64 = core::f64::consts::PI;

/// Pi/2, matching Imath's `M_PI_2` literal in `ImathPlatform.h`.
///
/// Imath only `#define`s this when `<math.h>` does not already provide `M_PI_2`.
/// Imath's literal `1.57079632679489661923` rounds to exactly the same `f64` as
/// Rust's [`core::f64::consts::FRAC_PI_2`] (verified below); we expose the std
/// constant directly for parity.
pub const M_PI_2: f64 = core::f64::consts::FRAC_PI_2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_constants_match_imath_literals() {
        // Confirm the std constants we re-export equal Imath's exact digit strings
        // when parsed to f64 (so substituting the std constant is a true 1:1).
        assert_eq!(M_PI, "3.14159265358979323846".parse::<f64>().unwrap());
        assert_eq!(M_PI_2, "1.57079632679489661923".parse::<f64>().unwrap());
    }
}
