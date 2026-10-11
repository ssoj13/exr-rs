//! Checked fixed-point domain for the resident GPU area mean.
//! With <=256 source samples and <=45 occupied binary positions, every
//! partial signed sum fits 53 magnitude bits and is exactly representable
//! in the CPU reference's f64 arithmetic.
//!
//! The profile of an RGBA buffer folds each slot's values into a [`Span`] of
//! occupied binary positions. A slot's span is its channel's, so a frame's
//! channels are profiled once at load ([`crate::image::DecodedFrame`] keeps them) and a
//! displayed layer's profile is assembled from its channels' spans without
//! rescanning the materialized pixels on every sequence step.

/// The binary positions occupied by the nonzero values of one stream: the
/// lowest set mantissa bit and the highest, as exponents. `None` (where a span
/// is optional) = a value outside the checked domain (subnormal, infinite,
/// NaN or beyond the half range), which sends the whole image to `main_wide`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    low: i32,
    high: i32,
}

impl Span {
    /// No nonzero value yet.
    pub const EMPTY: Self = Self { low: i32::MAX, high: i32::MIN };

    /// `self` with `v` folded in; `None` when `v` is outside the domain.
    #[inline]
    pub fn add(self, v: f32) -> Option<Self> {
        let bits = v.to_bits() & 0x7fffffff;
        if bits == 0 {
            return Some(self);
        }
        let exponent = (bits >> 23) & 255;
        if exponent == 0 || exponent == 255 || v.abs() > 65504.0 {
            return None;
        }
        let mantissa = (bits & 0x7fffff) | 0x800000;
        let base = exponent as i32 - 150;
        Some(Self {
            low: self.low.min(base + mantissa.trailing_zeros() as i32),
            high: self.high.max(base + (31 - mantissa.leading_zeros()) as i32),
        })
    }

    /// The span of `values`; `None` when any is outside the domain.
    pub fn of(values: impl IntoIterator<Item = f32>) -> Option<Self> {
        values.into_iter().try_fold(Self::EMPTY, Self::add)
    }

    /// The span of half-float bit patterns, from the distinct patterns only:
    /// min/max do not depend on repetition, and a 16-bit set is 8 KiB, so a
    /// half plane costs one bit store per sample instead of a float decode.
    pub fn of_halves(bits: impl IntoIterator<Item = u16>) -> Option<Self> {
        let mut seen = vec![0u64; 1 << 10];
        for b in bits {
            seen[usize::from(b >> 6)] |= 1 << (b & 63);
        }
        let mut span = Self::EMPTY;
        for (word, &set) in seen.iter().enumerate() {
            let mut set = set;
            while set != 0 {
                let b = (word << 6) as u16 | set.trailing_zeros() as u16;
                set &= set - 1;
                span = span.add(imath_rs::Half::from_bits(b).to_f32())?;
            }
        }
        Some(span)
    }
}

/// The kernel parameters for an image whose four slots have `spans`: each
/// slot's low position, or `None` (the per-cell `main_wide` kernel) when a
/// slot left the domain or occupies more than 45 positions.
pub fn finish(spans: [Option<Span>; 4]) -> Option<[i32; 4]> {
    let mut low = [0; 4];
    for (c, span) in spans.into_iter().enumerate() {
        let span = span?;
        if span.low == i32::MAX {
            continue;
        }
        if span.high - span.low + 1 > 45 {
            return None;
        }
        low[c] = span.low;
    }
    Some(low)
}

/// The profile of an interleaved RGBA buffer (the reference the per-channel
/// spans must reproduce).
pub fn profile(rgba: &[f32]) -> Option<[i32; 4]> {
    let mut spans = [Span::EMPTY; 4];
    for px in rgba.chunks(4) {
        for (span, &v) in spans.iter_mut().zip(px) {
            *span = span.add(v)?;
        }
    }
    finish(spans.map(Some))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_finite_half_value_shares_one_checked_domain() {
        let rgba = (0..=u16::MAX)
            .map(imath_rs::Half::from_bits)
            .map(|h| h.to_f32())
            .filter(|v| v.is_finite())
            .flat_map(|v| [v; 4])
            .collect::<Vec<_>>();
        assert_eq!(profile(&rgba), Some([-24; 4]));
    }
    #[test]
    fn half_domain_and_cancellation_fit_float_extremes_do_not() {
        assert!(profile(&[65504.0, f32::from_bits(0x33800000), -65504.0, 0.0].repeat(4)).is_some());
        let rgba = [65504.0, f32::from_bits(0x35800000), -65504.0, 0.0]
            .into_iter()
            .flat_map(|v| [v; 4])
            .collect::<Vec<_>>();
        assert!(profile(&rgba).is_some());
        let wide = [1e6, 0.01, -1e6, 0.0]
            .into_iter()
            .flat_map(|v| [v; 4])
            .collect::<Vec<_>>();
        assert_eq!(profile(&wide), None);
        for value in [
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
        ] {
            assert_eq!(profile(&[value; 4]), None);
        }
    }
    /// The distinct-pattern half span equals the per-value span, for every
    /// pattern alone (incl. infinities and NaNs) and for mixed sets.
    #[test]
    fn half_pattern_span_matches_the_value_span() {
        for b in 0..=u16::MAX {
            let v = imath_rs::Half::from_bits(b).to_f32();
            assert_eq!(Span::of_halves([b]), Span::of([v]), "pattern {b:#06x}");
        }
        let mixed = [0x3c00u16, 0x0001, 0x7bff, 0x8000, 0xc400, 0x3555];
        let values = mixed.map(|b| imath_rs::Half::from_bits(b).to_f32());
        assert_eq!(Span::of_halves(mixed), Span::of(values));
        assert_eq!(Span::of_halves([0x3c00, 0x7c00]), None, "an infinity leaves the domain");
    }
}
