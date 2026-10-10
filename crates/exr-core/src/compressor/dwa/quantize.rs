//! DWA bit-minimising quantisation — full scalar port of `algoQuantize` and
//! the quant-table builders from `OpenEXRCore/internal_dwa_encoder.h`.
//!
//! For each DCT coefficient DWA does NOT round to a fixed step; instead it
//! searches for the half-float value, within a per-coefficient error tolerance,
//! that has the **fewest set bits** (sparser mantissas compress better under
//! the downstream Huffman/zlib).  `algoQuantize` is the exact candidate-search
//! C++ uses; it dispatches on how the tolerance's exponent relates to the
//! source's, testing up to four masked/round-up alternates per case and keeping
//! the best (fewest bits, tie-broken by smallest in-tolerance delta).
//!
//! The candidate tests are *direction-aware* (the C++ `TEST_QUANT_ALTERNATE_LARGE`
//! / `_SMALL` macros): LARGE expects `alt >= src` and measures `alt - src`,
//! SMALL expects `alt <= src` and measures `src - alt`.  Using `abs()` would
//! wrongly admit out-of-window candidates, so the directions are kept distinct.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_encoder.h>

use imath_rs::Half;

use super::dct::BLOCK_ELEMENTS;
use super::zigzag::INV_ZIGZAG;

/// JPEG luminance quantisation table (relative sensitivities, min = 10).
pub const QUANT_TABLE_Y: [i32; BLOCK_ELEMENTS] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// JPEG chrominance quantisation table (min = 17).
pub const QUANT_TABLE_CBCR: [i32; BLOCK_ELEMENTS] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const QUANT_TABLE_Y_MIN: f32 = 10.0;
const QUANT_TABLE_CBCR_MIN: f32 = 17.0;

/// Count set bits — C `countSetBits`.
#[inline]
fn count_bits(x: u32) -> u32 {
    x.count_ones()
}

/// Count leading zeros — C `countLeadingZeros`.
#[inline]
fn count_leading_zeros(x: u32) -> u32 {
    x.leading_zeros()
}

/// Half bits → `f32` — C `half_to_float`.
#[inline]
fn half_to_float(h: u32) -> f32 {
    Half::from_bits(h as u16).to_f32()
}

// ---------------------------------------------------------------------------
// Direction-aware "best candidate so far" tracker.
//
// Models the C++ `TEST_QUANT_ALTERNATE_LARGE` / `_SMALL` macros: keep the
// candidate with the fewest bits set, tie-broken by the smallest in-direction
// delta, but only ever beating the initial `err_tol` sentinel.
// ---------------------------------------------------------------------------

/// Tracks the best (fewest-bits, smallest-delta) in-tolerance candidate.
struct BestCandidate {
    val: u32,
    bits: u32,
    delta: f32,
}

impl BestCandidate {
    #[inline]
    fn new(abs_src: u32, err_tol: f32) -> Self {
        Self {
            val: abs_src,
            bits: count_bits(abs_src),
            delta: err_tol, // sentinel: a candidate must be strictly within tol
        }
    }

    /// `TEST_QUANT_ALTERNATE_LARGE`: `alt >= src`, `delta = alt_f - src_f`.
    #[inline]
    fn test_large(&mut self, alt: u32, src_float: f32, err_tol: f32) {
        let bits = count_bits(alt);
        if bits < self.bits {
            let delta = half_to_float(alt) - src_float;
            if delta < err_tol {
                self.bits = bits;
                self.delta = delta;
                self.val = alt;
            }
        } else if bits == self.bits {
            let delta = half_to_float(alt) - src_float;
            if delta < self.delta {
                self.val = alt;
                self.delta = delta;
            }
        }
    }

    /// `TEST_QUANT_ALTERNATE_SMALL`: `alt <= src`, `delta = src_f - alt_f`.
    #[inline]
    fn test_small(&mut self, alt: u32, src_float: f32, err_tol: f32) {
        let bits = count_bits(alt);
        if bits < self.bits {
            let delta = src_float - half_to_float(alt);
            if delta < err_tol {
                self.bits = bits;
                self.delta = delta;
                self.val = alt;
            }
        } else if bits == self.bits {
            let delta = src_float - half_to_float(alt);
            if delta < self.delta {
                self.val = alt;
                self.delta = delta;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// handleQuantizeLargerSig — srcMaskedVal > tolSig.
// ---------------------------------------------------------------------------
fn handle_quantize_larger_sig(
    abs_src: u32,
    npow2: u32,
    mask: u32,
    err_tol: f32,
    src_float: f32,
) -> u32 {
    let mask2 = mask ^ (npow2 | (npow2 >> 1));

    let alt0 = abs_src & mask2;
    let alt1 = (abs_src + npow2) & mask;

    let bits0 = count_bits(alt0);
    let bits1 = count_bits(alt1);

    if bits1 < bits0 {
        let delta = half_to_float(alt1) - src_float;
        if delta < err_tol {
            return alt1;
        }
        let delta = src_float - half_to_float(alt0);
        if delta < err_tol {
            return alt0;
        }
    } else if bits1 == bits0 {
        let delta0 = src_float - half_to_float(alt0);
        let delta1 = half_to_float(alt1) - src_float;
        if delta0 < err_tol {
            return if delta1 < delta0 { alt1 } else { alt0 };
        }
        if delta1 < err_tol {
            return alt1;
        }
    } else {
        let delta = src_float - half_to_float(alt0);
        if delta < err_tol {
            return alt0;
        }
        let src_bits = count_bits(abs_src);
        if bits1 < src_bits {
            let delta = half_to_float(alt1) - src_float;
            if delta < err_tol {
                return alt1;
            }
        }
    }
    abs_src
}

// ---------------------------------------------------------------------------
// handleQuantizeSmallerSig — srcMaskedVal < tolSig.
// ---------------------------------------------------------------------------
fn handle_quantize_smaller_sig(
    abs_src: u32,
    npow2: u32,
    mask: u32,
    err_tol: f32,
    src_float: f32,
) -> u32 {
    let alt0 = abs_src & mask;
    let alt1 = (abs_src + npow2) & mask;

    let bits0 = count_bits(alt0);
    let bits1 = count_bits(alt1);

    if bits1 < bits0 {
        let delta = half_to_float(alt1) - src_float;
        if delta < err_tol {
            return alt1;
        }
        let delta = src_float - half_to_float(alt0);
        if delta < err_tol {
            return alt0;
        }
    } else if bits1 == bits0 {
        let delta0 = src_float - half_to_float(alt0);
        let delta1 = half_to_float(alt1) - src_float;
        if delta0 < err_tol {
            return if delta1 < delta0 { alt1 } else { alt0 };
        }
        if delta1 < err_tol {
            return alt1;
        }
    } else {
        let delta = src_float - half_to_float(alt0);
        if delta < err_tol {
            return alt0;
        }
        let src_bits = count_bits(abs_src);
        if bits1 < src_bits {
            let delta = half_to_float(alt1) - src_float;
            if delta < err_tol {
                return alt1;
            }
        }
    }
    abs_src
}

// ---------------------------------------------------------------------------
// handleQuantizeEqualSig — srcMaskedVal == tolSig.
// ---------------------------------------------------------------------------
fn handle_quantize_equal_sig(
    abs_src: u32,
    npow2: u32,
    mask: u32,
    err_tol: f32,
    src_float: f32,
) -> u32 {
    let alt0_base = abs_src & mask;
    let alt1 = (abs_src + npow2) & mask;

    // Fast path: mask truncation is usually within tolerance.
    let delta0 = src_float - half_to_float(alt0_base);
    let (alt0, delta0) = if delta0 >= err_tol {
        let mask2 = mask ^ (npow2 | (npow2 >> 1));
        let a = abs_src & mask2;
        let d = src_float - half_to_float(a);
        if d >= err_tol {
            let delta1 = half_to_float(alt1) - src_float;
            if delta1 < err_tol {
                let bits1 = count_bits(alt1);
                let src_bits = count_bits(abs_src);
                if bits1 < src_bits {
                    return alt1;
                }
            }
            return abs_src;
        }
        (a, d)
    } else {
        (alt0_base, delta0)
    };

    let bits0 = count_bits(alt0);
    let bits1 = count_bits(alt1);

    if bits1 < bits0 {
        let delta1 = half_to_float(alt1) - src_float;
        if delta1 < err_tol {
            return alt1;
        }
    } else if bits1 == bits0 {
        let delta1 = half_to_float(alt1) - src_float;
        if delta1 < delta0 {
            return alt1;
        }
    }
    alt0
}

// ---------------------------------------------------------------------------
// handleQuantizeDefault — expDiff > 1 or denormal source.
// ---------------------------------------------------------------------------
fn handle_quantize_default(abs_src: u32, tol_sig: u32, err_tol: f32, src_float: f32) -> u32 {
    let ts_shift = 32 - count_leading_zeros(tol_sig);
    let npow2 = 1u32 << ts_shift;
    let lower_mask = npow2 - 1;
    let mask = !lower_mask;
    let src_masked_val = abs_src & lower_mask;

    if src_masked_val > tol_sig {
        handle_quantize_larger_sig(abs_src, npow2, mask, err_tol, src_float)
    } else if src_masked_val < tol_sig {
        handle_quantize_smaller_sig(abs_src, npow2, mask, err_tol, src_float)
    } else {
        handle_quantize_equal_sig(abs_src, npow2, mask, err_tol, src_float)
    }
}

// ---------------------------------------------------------------------------
// handleQuantizeDenormTol — tol_exp == 0 and expDiff > 1.
// ---------------------------------------------------------------------------
fn handle_quantize_denorm_tol(abs_src: u32, tol_sig: u32, err_tol: f32, src_float: f32) -> u32 {
    let ts_shift = 32 - count_leading_zeros(tol_sig);
    let npow2 = 1u32 << ts_shift;
    let lower_mask = npow2 - 1;
    let mask = !lower_mask;
    let mask2 = mask ^ npow2;

    let mut best = BestCandidate::new(abs_src, err_tol);
    best.test_small(abs_src & mask2, src_float, err_tol);
    best.test_small(abs_src & mask, src_float, err_tol);
    best.test_large((abs_src + npow2) & mask, src_float, err_tol);
    best.test_large((abs_src + (npow2 << 1)) & mask, src_float, err_tol);
    best.val
}

// ---------------------------------------------------------------------------
// handleQuantizeGeneric — tol_exp == 0 and (expDiff == 0 || 1).
// ---------------------------------------------------------------------------
fn handle_quantize_generic(abs_src: u32, tol_sig: u32, err_tol: f32, src_float: f32) -> u32 {
    let ts_shift = 32 - count_leading_zeros(tol_sig);
    let npow2 = 1u32 << ts_shift;
    let lower_mask = npow2 - 1;
    let mask = !lower_mask;
    let mask2 = mask ^ npow2;
    let src_masked_val = abs_src & lower_mask;
    let extra_bit = tol_sig > src_masked_val;

    let mask3 =
        mask2 ^ (((npow2 << 1) * (extra_bit as u32)) | ((npow2 >> 1) * (!extra_bit as u32)));

    let mut best = BestCandidate::new(abs_src, err_tol);

    if extra_bit {
        best.test_small(abs_src & mask3, src_float, err_tol);
        best.test_small(abs_src & mask2, src_float, err_tol);
        best.test_small(abs_src & mask, src_float, err_tol);
    } else if (abs_src & npow2) != 0 {
        best.test_small(abs_src & mask2, src_float, err_tol);
        best.test_small(abs_src & mask3, src_float, err_tol);
        best.test_small(abs_src & mask, src_float, err_tol);
    } else {
        best.test_small(abs_src & mask2, src_float, err_tol);
        best.test_small(abs_src & mask, src_float, err_tol);
        best.test_small(abs_src & mask3, src_float, err_tol);
    }
    best.test_large((abs_src + npow2) & mask, src_float, err_tol);

    best.val
}

// ---------------------------------------------------------------------------
// handleQuantizeEqualExp — expDiff == 0 (npow2 = 0x0800).
// ---------------------------------------------------------------------------
fn handle_quantize_equal_exp(abs_src: u32, tol_sig: u32, err_tol: f32, src_float: f32) -> u32 {
    const NPOW2: u32 = 0x0800;
    const LOWER_MASK: u32 = NPOW2 - 1;
    const MASK: u32 = !LOWER_MASK;
    const MASK2: u32 = MASK ^ NPOW2;

    let src_masked_val = abs_src & LOWER_MASK;
    let extra_bit = tol_sig > src_masked_val;

    let mask3 =
        MASK2 ^ (((NPOW2 << 1) * (extra_bit as u32)) | ((NPOW2 >> 1) * (!extra_bit as u32)));

    let mut best = BestCandidate::new(abs_src, err_tol);

    if src_masked_val == abs_src {
        best.test_small(abs_src & mask3, src_float, err_tol);
    } else {
        let alt0 = abs_src & MASK2;
        let alt1 = abs_src & MASK;
        let (alt0, alt1) = if alt0 == alt1 {
            (abs_src & mask3, alt1)
        } else {
            (alt0, alt1)
        };
        best.test_small(alt0, src_float, err_tol);
        best.test_small(alt1, src_float, err_tol);
    }
    best.test_large((abs_src + NPOW2) & MASK, src_float, err_tol);

    best.val
}

// ---------------------------------------------------------------------------
// handleQuantizeCloseExp — expDiff == 1 (npow2 = 0x0400).
// ---------------------------------------------------------------------------
fn handle_quantize_close_exp(abs_src: u32, tol_sig: u32, err_tol: f32, src_float: f32) -> u32 {
    const NPOW2: u32 = 0x0400;
    const LOWER_MASK: u32 = NPOW2 - 1;
    const MASK: u32 = !LOWER_MASK;
    const MASK2: u32 = MASK ^ NPOW2;

    let src_masked_val = abs_src & LOWER_MASK;
    let extra_bit = tol_sig > src_masked_val;

    let mask3 =
        MASK2 ^ (((NPOW2 << 1) * (extra_bit as u32)) | ((NPOW2 >> 1) * (!extra_bit as u32)));

    let alternates: [u32; 3] = if (abs_src & NPOW2) == 0 {
        if extra_bit {
            [abs_src & mask3, abs_src & MASK, (abs_src + NPOW2) & MASK]
        } else {
            [abs_src & MASK, abs_src & mask3, (abs_src + NPOW2) & MASK]
        }
    } else if extra_bit {
        let a0 = abs_src & mask3;
        let a1_candidate = abs_src & MASK2;
        let alt1_delta = src_float - half_to_float(a1_candidate);
        let a1 = if alt1_delta >= err_tol {
            abs_src & MASK
        } else {
            a1_candidate
        };
        [a0, a1, (abs_src + NPOW2) & MASK]
    } else {
        let a0_candidate = abs_src & MASK2;
        let alt0_delta = src_float - half_to_float(a0_candidate);
        let a0 = if alt0_delta >= err_tol {
            abs_src & MASK
        } else {
            a0_candidate
        };
        let a1 = abs_src & mask3;
        [a0, a1, (abs_src + NPOW2) & MASK]
    };

    let mut best = BestCandidate::new(abs_src, err_tol);
    best.test_small(alternates[0], src_float, err_tol);
    best.test_small(alternates[1], src_float, err_tol);
    best.test_large(alternates[2], src_float, err_tol);

    best.val
}

// ---------------------------------------------------------------------------
// algoQuantize — full port.
//
// Returns the half bit pattern (sign restored) with the fewest set bits that is
// within `err_tol` of `src_float`.  `h_err_tol` is the half bit pattern of the
// (always positive) tolerance; `err_tol` its f32 value.
// ---------------------------------------------------------------------------
fn algo_quantize(src: u16, h_err_tol: u16, err_tol: f32, src_float: f32) -> u16 {
    let src = src as u32;
    let h_err_tol = h_err_tol as u32;

    let sign = src & 0x8000;
    let abs_src = src & 0x7FFF;
    let src_float = src_float.abs();

    let src_exp_biased = src & 0x7C00;
    let tol_exp_biased = h_err_tol & 0x7C00;

    // NaN / Inf pass through unchanged.
    if src_exp_biased == 0x7C00 {
        return (sign | abs_src) as u16;
    }

    // Within tolerance of zero → zero.
    if src_float < err_tol {
        return 0;
    }

    // expDiff: tolerance exponent steps below source.  wrapping_sub so that a
    // tol-above-src underflow shifts tolSig to 0 (→ src returned unchanged).
    let exp_diff = src_exp_biased.wrapping_sub(tol_exp_biased) >> 10;

    // Denormal tolerance path.
    if tol_exp_biased == 0 {
        if exp_diff == 0 || exp_diff == 1 {
            let tol_sig = h_err_tol & 0x3FF;
            if tol_sig == 0 {
                return (sign | abs_src) as u16;
            }
            return (sign | handle_quantize_generic(abs_src, tol_sig, err_tol, src_float)) as u16;
        }

        let tol_sig_raw = h_err_tol & 0x3FF;
        if tol_sig_raw == 0 {
            return (sign | abs_src) as u16;
        }
        let tol_sig = if exp_diff >= 32 {
            0
        } else {
            tol_sig_raw >> exp_diff
        };
        let tol_sig = if tol_sig == 0 { 1 } else { tol_sig };

        return (sign | handle_quantize_denorm_tol(abs_src, tol_sig, err_tol, src_float)) as u16;
    }

    // Normal tolerance: shift the significand (with implicit leading 1) into the
    // source's exponent frame.
    let tol_sig = if exp_diff >= 32 {
        0
    } else {
        ((h_err_tol & 0x3FF) | (1 << 10)) >> exp_diff
    };

    if tol_sig == 0 {
        return (sign | abs_src) as u16;
    }

    if exp_diff > 1 || src_exp_biased == 0 {
        return (sign | handle_quantize_default(abs_src, tol_sig, err_tol, src_float)) as u16;
    }

    if exp_diff == 0 {
        return (sign | handle_quantize_equal_exp(abs_src, tol_sig, err_tol, src_float)) as u16;
    }
    (sign | handle_quantize_close_exp(abs_src, tol_sig, err_tol, src_float)) as u16
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Quantise a natural-order `f32` DCT block and place the results in zigzag
/// order — port of `quantizeCoeffAndZigXDR` (without the XDR flip, since our
/// pipeline is native).
///
/// `tolerances` / `half_tols` are the per-coefficient f32 / half tolerances
/// from [`build_quant_table_y`] / [`build_quant_table_cbcr`].
pub fn quantize_and_zigzag(
    dct_vals: &[f32; BLOCK_ELEMENTS],
    tolerances: &[f32; BLOCK_ELEMENTS],
    half_tols: &[u16; BLOCK_ELEMENTS],
    output: &mut [u16; BLOCK_ELEMENTS],
) {
    for i in 0..BLOCK_ELEMENTS {
        // C++ rounds the f32 to half first, then re-reads it as f32 (so the
        // search compares against the actually-stored value).
        let src = Half::from_f32(dct_vals[i]).bits();
        let src_float = Half::from_bits(src).to_f32();
        let quantized = algo_quantize(src, half_tols[i], tolerances[i], src_float);
        output[INV_ZIGZAG[i]] = quantized;
    }
}

/// Build the luminance (Y) tolerance tables for a given base error.
pub fn build_quant_table_y(base_error: f32) -> ([f32; BLOCK_ELEMENTS], [u16; BLOCK_ELEMENTS]) {
    let mut tols = [0.0f32; BLOCK_ELEMENTS];
    let mut htols = [0u16; BLOCK_ELEMENTS];
    for i in 0..BLOCK_ELEMENTS {
        tols[i] = base_error * (QUANT_TABLE_Y[i] as f32) / QUANT_TABLE_Y_MIN;
        htols[i] = Half::from_f32(tols[i]).bits();
    }
    (tols, htols)
}

/// Build the chrominance (CbCr) tolerance tables for a given base error.
pub fn build_quant_table_cbcr(base_error: f32) -> ([f32; BLOCK_ELEMENTS], [u16; BLOCK_ELEMENTS]) {
    let mut tols = [0.0f32; BLOCK_ELEMENTS];
    let mut htols = [0u16; BLOCK_ELEMENTS];
    for i in 0..BLOCK_ELEMENTS {
        tols[i] = base_error * (QUANT_TABLE_CBCR[i] as f32) / QUANT_TABLE_CBCR_MIN;
        htols[i] = Half::from_f32(tols[i]).bits();
    }
    (tols, htols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_bits_basic() {
        assert_eq!(count_bits(0), 0);
        assert_eq!(count_bits(0b1010), 2);
        assert_eq!(count_bits(0xffff), 16);
    }

    #[test]
    fn tiny_value_quantizes_to_zero() {
        let small = Half::from_f32(0.001).bits();
        let result = algo_quantize(small, Half::from_f32(0.01).bits(), 0.01, 0.001);
        assert_eq!(result, 0);
    }

    #[test]
    fn nan_inf_passthrough() {
        let inf = Half::from_bits(0x7c00).bits();
        assert_eq!(algo_quantize(inf, 0x3C00, 1.0, f32::INFINITY), inf);
        let neg_inf = 0xfc00u16;
        assert_eq!(algo_quantize(neg_inf, 0x3C00, 1.0, f32::NEG_INFINITY), neg_inf);
    }

    #[test]
    fn sign_preserved() {
        let neg = Half::from_f32(-1.5).bits();
        let result = algo_quantize(neg, Half::from_f32(0.1).bits(), 0.1, -1.5);
        assert_eq!(result & 0x8000, 0x8000);
    }

    #[test]
    fn bit_reduction_within_tolerance() {
        let one = Half::from_f32(1.0).bits();
        let tol = Half::from_f32(0.5).bits();
        let result = algo_quantize(one, tol, 0.5, 1.0);
        assert!(count_bits(result as u32) <= count_bits(one as u32));
    }

    #[test]
    fn table_y_dc_tolerance() {
        let (tols, _h) = build_quant_table_y(0.045);
        assert!((tols[0] - 0.045 * 1.6).abs() < 0.001); // 16/10
    }
}
