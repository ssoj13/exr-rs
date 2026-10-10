//! Port of Imath ImathTypeTraits.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/ImathTypeTraits.h>
//!
//! ## What `ImathTypeTraits.h` contains
//! The C++ header defines `Imath::enable_if_t` plus a family of SFINAE detection
//! traits — `has_xy<T,Base>`, `has_xyz`, `has_xyzw`, `has_subscript<T,Base,N>`,
//! and `has_double_subscript<T,Base,R,C>`. Their entire job is **foreign-vector
//! interop**: at compile time, sniff whether some *external* application type
//! "looks like" an Imath vector/matrix (right members, right size) so that
//! `V3f`/`M44f` can implicitly convert to and from it. The whole block is wrapped
//! in `#if IMATH_FOREIGN_VECTOR_INTEROP` and `enable_if_t` is just C++11/14 glue.
//!
//! ## Why there is no 1:1 Rust port of those traits
//! - `enable_if_t` and `IMATH_ENABLE_IF` are SFINAE plumbing; Rust expresses the
//!   same intent with ordinary trait bounds and `where` clauses, so there is no
//!   standalone symbol to port.
//! - `has_xy`/`has_subscript`/… detect *third-party* C++ struct layouts via
//!   `sizeof` and `decltype`. Rust has no equivalent reflection, and the exr-rs
//!   port defines its own `Vec`/`Matrix` types — it never needs to silently
//!   adopt a foreign application's vector type. This is the `IMATH_FOREIGN_VECTOR_INTEROP`
//!   feature, which we do not enable; porting it would be speculative API with no
//!   consumer in OpenEXR/IlmImf.
//!
//! ## What this module DOES provide
//! The header that matters for the generic `Vec<T>`/`Matrix<T>` machinery is
//! `ImathMath.h`: every Imath template silently assumes its element type `T` is a
//! copyable arithmetic scalar that supports `+ - * /`, comparison, and the
//! literals `T(0)`/`T(1)`. C++ leaves this assumption implicit; Rust must spell it
//! out as a trait bound. [`Scalar`] is exactly that minimal bound — nothing more,
//! derived from what Imath's templates actually use.

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Minimal numeric-scalar bound shared by every generic Imath type (`Vec<T>`,
/// `Matrix<T>`, …).
///
/// This captures, explicitly, the contract that C++ Imath templates assume
/// implicitly of their element type `T`:
/// - `Copy` — Imath passes scalars by value and relies on trivial copy.
/// - the four arithmetic operators (`+ - * /`) plus `Rem` (`%`, used by the
///   integer `mods`/`modp` helpers) and unary `Neg` (`-a`, used pervasively, e.g.
///   `abs`/`sign`).
/// - `PartialOrd` + `PartialEq` — Imath compares scalars (`a > T(0)`, `a < l`).
/// - [`ZERO`](Scalar::ZERO) / [`ONE`](Scalar::ONE) — the `T(0)` / `T(1)` literals
///   that appear throughout Imath (e.g. in `abs`, `lerp`, `sign`).
///
/// It is intentionally tiny. Floating-point-only transcendental functions live in
/// the separate [`Math`](crate::math::Math) trait (1:1 with `ImathMath.h`), so that
/// integer scalars (`i32`, used by `divs`/`divp`/`mods`/`modp`) can implement
/// `Scalar` without pretending to have `sqrt`/`sin`/etc.
pub trait Scalar:
    Copy
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity, equivalent to C++ `T(0)`.
    const ZERO: Self;
    /// The multiplicative identity, equivalent to C++ `T(1)`.
    const ONE: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Scalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

/// Minimal scalar bound for `Color3<T>` and `Color4<T>` element types.
///
/// C++ `Color3<T>` / `Color4<T>` are instantiated with `float`, `half`, and
/// `unsigned char` (`u8`). The broader [`Scalar`] trait requires `Neg + Rem` which
/// `u8` cannot satisfy under Rust's orphan rules (neither `Neg` nor `u8` is local).
/// `ColorScalar` is therefore the weaker bound used by color types: `Copy`,
/// `PartialOrd`, `PartialEq`, the four arithmetic operators, and the `ZERO`/`ONE`
/// identities. Negation (`Neg`) is added as an additional bound only on the methods
/// that actually call it (`operator-`, `negate`).
pub trait ColorScalar:
    Copy
    + PartialOrd
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// The additive identity, equivalent to C++ `T(0)`.
    const ZERO: Self;
    /// The multiplicative identity, equivalent to C++ `T(1)`.
    const ONE: Self;
}

impl ColorScalar for f32  { const ZERO: Self = 0.0; const ONE: Self = 1.0; }
impl ColorScalar for f64  { const ZERO: Self = 0.0; const ONE: Self = 1.0; }
impl ColorScalar for i32  { const ZERO: Self = 0;   const ONE: Self = 1;   }

/// `u8` `ColorScalar` — for `Color3c` / `Color4c` (`unsigned char` in C++).
///
/// `u8` satisfies all arithmetic operators that `Color3`/`Color4` use for its
/// unsigned channel data. Negation/negate methods on `Color3<u8>`/`Color4<u8>`
/// are excluded at the type level (those methods carry an extra `Neg` bound).
impl ColorScalar for u8   { const ZERO: Self = 0;   const ONE: Self = 1;   }

impl ColorScalar for crate::half::Half {
    const ZERO: Self = crate::half::Half::from_bits(0x0000); // 0.0
    const ONE:  Self = crate::half::Half::from_bits(0x3c00); // 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generic helper that exercises the `Scalar` bound the way Imath templates do.
    fn doubled<T: Scalar>(a: T) -> T {
        a * (T::ONE + T::ONE)
    }

    #[test]
    fn scalar_for_floats_and_ints() {
        assert_eq!(doubled(3.0_f32), 6.0);
        assert_eq!(doubled(3.0_f64), 6.0);
        assert_eq!(doubled(3_i32), 6);
        assert_eq!(<f32 as Scalar>::ZERO, 0.0);
        assert_eq!(<f64 as Scalar>::ONE, 1.0);
        assert_eq!(<i32 as Scalar>::ONE, 1);
    }
}
