//! Port of Imath toFloat.h / toFloat.cpp.
//! <https://github.com/AcademySoftwareFoundation/Imath/blob/v3.2.0/src/Imath/toFloat.h>
//!
//! `toFloat.h` is an automatically-generated 65,536-entry lookup table holding the
//! `float` bit pattern for every possible `half` bit pattern; `half.cpp` exposes it
//! as `imath_half_to_float_table`. Rather than embed a 1 MB generated table, we
//! reproduce the *exact algorithm* that produced it: the table is, entry-for-entry,
//! the result of the bit-shift `imath_half_to_float` path defined in `half.h`
//! (`#else` branch, when no F16C / no lookup table). This module ports that path
//! verbatim, plus the bit-shift `imath_float_to_half` path. Together they are the
//! authoritative `half`↔`f32` implementation for the whole crate — [`super::Half`]
//! calls them directly.
//!
//! Both functions are pure `u16`/`f32` bit manipulations with no dependencies.

/// Port of the bit-shift `imath_half_to_float` (`half.h`, no-table `#else` branch).
///
/// Returns the IEEE-754 `f32` whose bit pattern Imath stores for the given `half`
/// bit pattern `h`. This is the generator of every entry in `toFloat.h`, so it is
/// the authoritative half→float oracle. Conversions from half to float are always
/// exact (every half is representable as a float), so there is no rounding here.
///
/// The algorithm (faithful to the C++):
/// 1. `hexpmant` = the half's exponent+mantissa, left-justified into a float's
///    exponent+mantissa field (`(h << 17) >> 4`).
/// 2. Sign bit is `h`'s top bit moved to bit 31.
/// 3. If `hexpmant >= 0x0080_0000` the half is normal or inf/nan: copy the bits in,
///    then either add the exponent-bias difference (`0x3800_0000`) for a normal
///    number, or force all float exponent bits set (`0x7f80_0000`) for inf/nan.
/// 4. Otherwise, if non-zero, it is a denormal half: normalize it via a
///    count-leading-zeros shift and fix up the exponent.
/// 5. Zero falls through with only the sign bit set.
pub fn imath_half_to_float(h: u16) -> f32 {
    // `hexpmant`: half's 5-bit exponent + 10-bit mantissa, shifted up so they sit in
    // the float's exponent (bits 30..23) + mantissa (bits 22..0) positions. The C++
    // does `((uint32_t)h << 17) >> 4` on the full 16-bit value; the sign bit is
    // discarded by the `<< 17` (it shifts off the top) and re-added separately.
    let hexpmant: u32 = ((h as u32) << 17) >> 4;
    // Sign bit: half's bit 15 → float's bit 31.
    let mut vi: u32 = ((h >> 15) as u32) << 31;

    if hexpmant >= 0x0080_0000 {
        vi |= hexpmant;
        // Normal number: add the bias difference (127 - 15, suitably shifted).
        // Inf/nan: set all exponent bits.
        if hexpmant < 0x0f80_0000 {
            vi += 0x3800_0000;
        } else {
            vi |= 0x7f80_0000;
        }
    } else if hexpmant != 0 {
        // Denormalized half. Count-leading-zeros of `hexpmant` (it is non-zero here,
        // so `leading_zeros` is well defined and matches `__builtin_clz`).
        let lc: u32 = hexpmant.leading_zeros() - 8;
        vi |= 0x3880_0000;
        vi |= hexpmant << lc;
        vi = vi.wrapping_sub(lc << 23);
    }

    f32::from_bits(vi)
}

/// Port of the bit-shift `imath_float_to_half` (`half.h`, no-F16C `#else` branch).
///
/// Returns the `half` bit pattern nearest to `f`, using round-to-nearest-even (the
/// only mode the original OpenEXR library supported) with overflow→infinity and
/// underflow→zero / denormal handling. This is the authoritative float→half oracle.
///
/// The algorithm (faithful to the C++, FP-exception hooks omitted — the project
/// builds without `IMATH_HALF_ENABLE_FP_EXCEPTIONS`):
/// - `ui` is `|f|`'s bits, `ret` carries the sign bit pre-positioned at bit 15.
/// - `ui >= 0x3880_0000`: exponent large enough for a normal half (or inf/nan/over).
///   inf/nan preserve nan-ness; values above `0x477f_efff` round to infinity;
///   otherwise rebias and round-to-nearest-even.
/// - `ui < 0x3300_0001`: flushes to zero.
/// - Otherwise: produce a denormalized half with round-to-nearest-even.
pub fn imath_float_to_half(f: f32) -> u16 {
    let vi: u32 = f.to_bits();

    let mut ui: u32 = vi & !0x8000_0000;
    let mut ret: u16 = ((vi >> 16) & 0x8000) as u16;

    // Exponent large enough to result in a normal number; round and return.
    if ui >= 0x3880_0000 {
        // inf or nan
        if ui >= 0x7f80_0000 {
            ret |= 0x7c00;
            if ui == 0x7f80_0000 {
                return ret;
            }
            let m: u32 = (ui & 0x7f_ffff) >> 13;
            // Ensure at least one mantissa bit survives the shift to preserve nan-ness.
            return ret | (m as u16) | ((m == 0) as u16);
        }

        // Too large: round to infinity.
        if ui > 0x477f_efff {
            return ret | 0x7c00;
        }

        ui -= 0x3800_0000;
        ui = (ui + 0x0000_0fff + ((ui >> 13) & 1)) >> 13;
        return ret | (ui as u16);
    }

    // Zero or flush to zero.
    if ui < 0x3300_0001 {
        return ret;
    }

    // Produce a denormalized half.
    let e: u32 = ui >> 23;
    let shift: u32 = 0x7e - e;
    let m: u32 = 0x80_0000 | (ui & 0x7f_ffff);
    let r: u32 = m << (32 - shift);
    ret |= (m >> shift) as u16;
    if r > 0x8000_0000 || (r == 0x8000_0000 && (ret & 0x1) != 0) {
        ret += 1;
    }
    ret
}
