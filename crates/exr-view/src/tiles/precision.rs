//! Checked fixed-point domain for the resident GPU area mean.
//! With <=256 source samples and <=45 occupied binary positions, every
//! partial signed sum fits 53 magnitude bits and is exactly representable
//! in the CPU reference's f64 arithmetic.
pub fn profile(rgba: &[f32]) -> Option<[i32; 4]> {
    let mut low = [i32::MAX; 4];
    let mut high = [i32::MIN; 4];
    for (i, &v) in rgba.iter().enumerate() {
        let bits = v.to_bits() & 0x7fffffff;
        if bits == 0 {
            continue;
        }
        let exponent = (bits >> 23) & 255;
        if exponent == 0 || exponent == 255 || v.abs() > 65504.0 {
            return None;
        }
        let mantissa = (bits & 0x7fffff) | 0x800000;
        let base = exponent as i32 - 150;
        let c = i % 4;
        low[c] = low[c].min(base + mantissa.trailing_zeros() as i32);
        high[c] = high[c].max(base + (31 - mantissa.leading_zeros()) as i32);
    }
    for c in 0..4 {
        if low[c] == i32::MAX {
            low[c] = 0;
        } else if high[c] - low[c] + 1 > 45 {
            return None;
        }
    }
    Some(low)
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
}
