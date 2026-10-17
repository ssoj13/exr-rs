//! f16 / bf16 <-> f32: the ONE shared scalar implementation (SSOT for codec, gpu, capi, cli, view).
//!
//! Lives in `exv-format` because [`crate::Dtype`] (the thing being converted) is here and this crate is
//! zero-dep and below every consumer. We do not use the `half` crate / std `f16`: the NaN rule below is
//! part of the CPU<->GPU determinism contract (the WGSL kernels mirror it) and must not drift.
//!
//! # NaN / non-finite policy
//! - Widening ([`half_to_f32`], [`bf16_to_f32`]) is exact: sign and NaN payload are preserved bit-for-bit.
//! - Narrowing ([`f32_to_half`], [`f32_to_bf16`]) is round-to-nearest-even; overflow -> +-inf; NaN -> the
//!   canonical quiet NaN of the same sign (payload dropped), so a NaN never rounds into an infinity.

/// IEEE half (binary16) bits -> `f32`, exact. Handles zero/denormal/normal/inf/NaN (payload kept).
pub fn half_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            // f16 denorm value = 2^-24 * mant(normalized). exp bias chain: f32 bias 127, f16 bias 15,
            // +1 for the leading bit consumed by the normalize loop, +1 for the f16 min-exponent
            // offset (smallest denorm 0x0001 == 2^-24, not 2^-25).
            sign | (((127 - 15 + 2 + e) as u32) << 23) | (m << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// `f32` -> IEEE half (binary16) bits, round-to-nearest-even. Overflow -> +-inf, subnormals handled,
/// NaN -> canonical quiet NaN of the same sign.
pub fn f32_to_half(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let mant = x & 0x7f_ffff;
    if exp >= 0x1f {
        return if (x & 0x7fff_ffff) > 0x7f80_0000 {
            sign | 0x7e00 // NaN
        } else {
            sign | 0x7c00 // inf / overflow
        };
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half_mant = (mant >> shift) as u16;
        let rem = mant & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round = (rem > halfway) || (rem == halfway && (half_mant & 1) == 1);
        return sign | (half_mant + u16::from(round));
    }
    let half = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    let round = (rem > 0x1000) || (rem == 0x1000 && (half & 1) == 1);
    half + u16::from(round)
}

/// bfloat16 bits -> `f32`, exact (NaN payload kept).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `f32` -> bfloat16 bits, round-to-nearest-even (overflow -> +-inf; NaN -> quiet NaN, same sign).
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return ((bits >> 16) as u16) | 0x7fc0;
    }
    let round = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact round trips (incl. f16 subnormals: smallest denorm 0x0001 must decode to 2^-24) and the NaN policy.
    #[test]
    fn half_conversions() {
        for &v in &[0.0f32, 1.0, 2.0, 0.5, 100.0, -3.0, 2f32.powi(-24), 2f32.powi(-20), 2f32.powi(-15)] {
            assert_eq!(half_to_f32(f32_to_half(v)), v, "f16 round-trip {v}");
        }
        for &v in &[1.0f32, 2.0, -4.0, 0.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v, "bf16 round-trip {v}");
        }
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        // Every f16 bit pattern survives widen->narrow (NaNs canonicalize to quiet, payload dropped).
        for h in 0..=u16::MAX {
            let back = f32_to_half(half_to_f32(h));
            if (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 {
                assert_eq!(back, (h & 0x8000) | 0x7e00, "NaN {h:#06x}");
            } else {
                assert_eq!(back, h, "f16 {h:#06x}");
            }
        }
    }

    /// The NaN policy, pinned: widen keeps sign+payload; narrow -> canonical quiet NaN, same sign.
    #[test]
    fn nan_policy() {
        assert_eq!(half_to_f32(0xfd55).to_bits(), 0xffaa_a000);
        assert_eq!(bf16_to_f32(0x7f81).to_bits(), 0x7f81_0000);
        assert_eq!(f32_to_half(f32::from_bits(0xffc0_1234)), 0xfe00);
        assert_eq!(f32_to_half(f32::from_bits(0x7f80_0001)), 0x7e00);
        assert_eq!(f32_to_half(70000.0), 0x7c00);
        assert_eq!(f32_to_half(-70000.0), 0xfc00);
        // A NaN with all payload in the low bits must not round into inf in bf16.
        assert_eq!(f32_to_bf16(f32::from_bits(0x7f80_0001)), 0x7fc0);
        assert_eq!(f32_to_bf16(f32::from_bits(0xff80_0001)), 0xffc0);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
    }
}
