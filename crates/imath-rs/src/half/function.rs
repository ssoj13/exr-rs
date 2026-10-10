//! Port of Imath halfFunction.h.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/halfFunction.h>
//!
//! [`HalfFunction`] is a 1:1 port of the C++ `halfFunction<T>` template: a class for
//! fast evaluation of `half -> T` functions via a precomputed 65,536-entry lookup
//! table, one slot per possible `half` bit pattern.
//!
//! The constructor evaluates the supplied function `f` for every finite half value in
//! the inclusive interval `[domain_min, domain_max]` and stores the results. For
//! finite halfs outside that interval it stores `default_value`; for +inf, -inf and
//! NaN it stores `pos_inf_value`, `neg_inf_value` and `nan_value` respectively. After
//! construction, evaluation is a single table lookup keyed by the half's bits.
//!
//! The C++ has two storage variants gated on `IMATH_HAVE_LARGE_STACK` (a stack array
//! vs a heap `new[]`); the observable behaviour is identical. We use a heap [`Vec`],
//! which is the faithful analogue of the (default) non-large-stack `new T[1 << 16]`
//! path, and avoids placing a 64 K-entry array on the Rust stack.

use super::Half;
use super::HALF_MAX;

/// Number of table entries: one per possible `half` bit pattern (`1 << 16`).
const TABLE_SIZE: usize = 1 << 16;

/// Port of `halfFunction<T>`: a tabulated `half -> T` function.
///
/// `T` must be [`Copy`] so entries can be returned by value from the lookup, matching
/// the C++ where `operator()` returns `T` by value from `_lut[x.bits()]`.
pub struct HalfFunction<T: Copy> {
    /// The lookup table, indexed by a half's 16-bit pattern (port of `_lut`).
    lut: Vec<T>,
}

impl<T: Copy> HalfFunction<T> {
    /// Port of the `halfFunction` constructor.
    ///
    /// Builds the lookup table by evaluating `f` over `[domain_min, domain_max]`.
    /// The defaults mirror the C++ signature: `domain_min = -HALF_MAX`,
    /// `domain_max = HALF_MAX`, and `default/pos_inf/neg_inf/nan` values supplied by
    /// the caller (in C++ they default to `T(0)`; Rust has no per-type default `0`,
    /// so the caller passes them explicitly — see [`HalfFunction::with_defaults`]).
    ///
    /// Classification order matches the C++ exactly: NaN first, then infinity (sign
    /// chooses pos/neg), then out-of-domain (`x < domain_min || x > domain_max`,
    /// i.e. float comparison), otherwise `f(x)`.
    pub fn new<F>(
        f: F,
        domain_min: Half,
        domain_max: Half,
        default_value: T,
        pos_inf_value: T,
        neg_inf_value: T,
        nan_value: T,
    ) -> Self
    where
        F: Fn(Half) -> T,
    {
        let mut lut: Vec<T> = Vec::with_capacity(TABLE_SIZE);
        for i in 0..TABLE_SIZE {
            // C++: `half x; x.setBits(i);`
            let x = Half::from_bits(i as u16);

            let v = if x.is_nan() {
                nan_value
            } else if x.is_infinity() {
                if x.is_negative() {
                    neg_inf_value
                } else {
                    pos_inf_value
                }
            } else if x < domain_min || x > domain_max {
                default_value
            } else {
                f(x)
            };
            lut.push(v);
        }
        Self { lut }
    }

    /// Convenience matching the C++ default arguments: `domain = [-HALF_MAX, HALF_MAX]`
    /// and a single `fill` value used for the default/pos-inf/neg-inf/nan slots
    /// (the C++ defaults those four to `T(0)`; here the caller names the fill).
    #[inline]
    pub fn with_defaults<F>(f: F, fill: T) -> Self
    where
        F: Fn(Half) -> T,
    {
        Self::new(
            f,
            Half::new(-HALF_MAX),
            Half::new(HALF_MAX),
            fill,
            fill,
            fill,
            fill,
        )
    }

    /// Port of `T operator()(half x) const`: evaluate via a single table lookup.
    #[inline]
    pub fn eval(&self, x: Half) -> T {
        self.lut[x.bits() as usize]
    }
}

// Allow `f(x)` call syntax to mirror the C++ `operator()`. `Fn`/`FnOnce` traits are
// unstable to implement on stable Rust, so [`HalfFunction::eval`] is the call entry
// point; this `Index`-free design keeps the lookup a plain method.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabulates_and_handles_special_values() {
        // h(x) = 2*x as a half, on the default full domain.
        let f = HalfFunction::new(
            |x: Half| Half::new(x.to_f32() * 2.0),
            Half::new(-HALF_MAX),
            Half::new(HALF_MAX),
            Half::new(0.0),   // default (out of domain)
            Half::new(111.0), // +inf
            Half::new(-111.0),// -inf
            Half::new(222.0), // nan
        );

        assert_eq!(f.eval(Half::new(3.5)).to_f32(), 7.0);
        assert_eq!(f.eval(Half::pos_inf()).to_f32(), 111.0);
        assert_eq!(f.eval(Half::neg_inf()).to_f32(), -111.0);
        assert_eq!(f.eval(Half::q_nan()).to_f32(), 222.0);
        assert_eq!(f.eval(Half::s_nan()).to_f32(), 222.0);
    }

    #[test]
    fn out_of_domain_uses_default() {
        // Restrict domain to [0, HALF_MAX]; negative finite -> default.
        let f = HalfFunction::new(
            |x: Half| x,
            Half::new(0.0),
            Half::new(HALF_MAX),
            Half::q_nan(),    // default for out-of-domain (e.g. sqrt of negative)
            Half::pos_inf(),  // +inf
            Half::q_nan(),    // -inf
            Half::q_nan(),    // nan
        );
        // In-domain passes through.
        assert_eq!(f.eval(Half::new(2.0)).to_f32(), 2.0);
        // Negative finite is out of domain -> default (qNan).
        assert!(f.eval(Half::new(-2.0)).is_nan());
        // +inf maps to pos_inf_value.
        assert!(f.eval(Half::pos_inf()).is_infinity());
    }

    #[test]
    fn with_defaults_fills_specials() {
        let zero = Half::new(0.0);
        let f = HalfFunction::with_defaults(|x: Half| x, zero);
        assert_eq!(f.eval(Half::new(5.0)).to_f32(), 5.0);
        // Specials all filled with the single `fill` value (0.0 here).
        assert!(f.eval(Half::q_nan()).is_zero());
        assert!(f.eval(Half::pos_inf()).is_zero());
        assert!(f.eval(Half::neg_inf()).is_zero());
    }
}
