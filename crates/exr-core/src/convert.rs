//! Port of OpenEXR `ImfConvert.{h,cpp}` — the single source of truth for
//! scalar pixel-sample conversion between HALF, FLOAT and UINT, extended with
//! the `OpenEXRCore/internal_coding.h` primitives the live 3.4 unpack path
//! actually uses.
//!
//! Every reader/writer/facade/downstream-crate conversion of one pixel sample
//! from one [`crate::attr::PixelType`] to another goes through this module
//! instead of a hand-rolled copy — see `misc.rs`'s
//! `copy_into_frame_buffer` for the highest-traffic caller.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfConvert.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfConvert.cpp>
//!
//! # Two different HALF overflow rules — do not conflate them
//!
//! * **UINT -> HALF** (`uint_to_half`) overflows to **+inf as soon as `u >
//!   65504`** (HALF_MAX) — a plain threshold, no rounding involved
//!   (`ImfConvert.cpp:97-102`, `OpenEXRCore/internal_coding.h:275-280`
//!   `uint_to_half`). `65505` is already `+inf`, not the largest finite half.
//! * **FLOAT -> HALF** (`float_to_half`) is the LIVE 3.4 unpack path
//!   (`OpenEXRCore/unpack.c::UNPACK_FLOAT_TO_HALF_SAMPLES` ->
//!   `internal_coding.h::float_to_half` -> Imath's `imath_float_to_half`,
//!   round-to-nearest-even), which overflows to +/-inf only at `|f| >=
//!   65520` (`|bits| > 0x477fefff`); `65504 <= |f| < 65520` rounds to the
//!   largest finite half. The legacy `ImfConvert::floatToHalf` (dead code in
//!   3.4, clamps at HALF_MAX like `uint_to_half`) is NOT the live path and is
//!   deliberately not ported. `Half::from(f32)` (== `imath_rs`'s
//!   `imath_float_to_half`) already implements the 65520 threshold, so
//!   `float_to_half` needs no extra clamping.
//!
//! Both boundaries are oracle-verified: `tests/half_conversion_boundary.rs`
//! (->HALF) and `tests/uint_conversion_boundary.rs` (->UINT).

use imath_rs::half::HALF_MAX;
use imath_rs::Half;

/// `halfToUint` — saturate a HALF value to `[0, u32::MAX]`.
///
/// NaN or negative -> `0`; `> u32::MAX` (including +inf) -> `u32::MAX`.
/// Port of `ImfConvert.cpp:75-84` / `internal_coding.h::half_to_uint`.
pub fn half_to_uint(h: Half) -> u32 {
    float_to_uint(f32::from(h))
}

/// `floatToUint` — saturate a FLOAT value to `[0, u32::MAX]`.
///
/// NaN or negative -> `0`; `> u32::MAX` (including +inf) -> `u32::MAX`.
/// Port of `ImfConvert.cpp:86-94` / `internal_coding.h::float_to_uint`.
pub fn float_to_uint(f: f32) -> u32 {
    if f.is_nan() || f < 0.0 {
        0
    } else if f > u32::MAX as f32 {
        u32::MAX
    } else {
        f as u32
    }
}

/// `uintToHalf` — `u` above `HALF_MAX` (65504) becomes `+inf`, never the
/// largest finite half. A plain `Half::from(u as f32)` alone would instead
/// round `65505..=65519` down to `65504`, which diverges from C++ for those
/// values. Port of `ImfConvert.cpp:97-102` / `internal_coding.h:275-280`.
pub fn uint_to_half(u: u32) -> Half {
    if u as f32 > HALF_MAX {
        Half::pos_inf()
    } else {
        Half::from(u as f32)
    }
}

/// `uintToFloat` — a raw `(float)ui` cast, no saturation.
/// Port of `ImfMisc.cpp:426-436,602,1027,1373` / `internal_coding.h:282-286`.
pub fn uint_to_float(u: u32) -> f32 {
    u as f32
}

/// FLOAT -> HALF on the LIVE 3.4 unpack path (`unpack.c` ->
/// `internal_coding.h::float_to_half` -> Imath round-to-nearest-even):
/// overflows to +/-inf only at `|f| >= 65520`, not at `HALF_MAX` (65504).
/// See the module doc for why this is NOT the legacy `ImfConvert::floatToHalf`.
pub fn float_to_half(f: f32) -> Half {
    Half::from(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_to_uint_boundaries() {
        assert_eq!(half_to_uint(Half::from(-1.0)), 0);
        assert_eq!(half_to_uint(Half::from(f32::NAN)), 0);
        assert_eq!(half_to_uint(Half::pos_inf()), u32::MAX);
        assert_eq!(half_to_uint(Half::from(1000.0)), 1000);
    }

    #[test]
    fn float_to_uint_boundaries() {
        assert_eq!(float_to_uint(-1.0), 0);
        assert_eq!(float_to_uint(f32::NAN), 0);
        assert_eq!(float_to_uint(f32::INFINITY), u32::MAX);
        assert_eq!(float_to_uint(f32::NEG_INFINITY), 0);
        assert_eq!(float_to_uint(4294967040.0), 4294967040);
        assert_eq!(float_to_uint(f32::MAX), u32::MAX);
        assert_eq!(float_to_uint(1000.0), 1000);
    }

    #[test]
    fn uint_to_half_boundaries() {
        assert_eq!(uint_to_half(65504).bits(), 0x7bff);
        assert_eq!(uint_to_half(65505).bits(), 0x7c00);
        assert_eq!(uint_to_half(70000).bits(), 0x7c00);
        assert_eq!(uint_to_half(0).bits(), 0x0000);
        assert_eq!(uint_to_half(1000).bits(), Half::from(1000.0_f32).bits());
    }

    #[test]
    fn uint_to_float_is_raw_cast() {
        assert_eq!(uint_to_float(1000), 1000.0);
        assert_eq!(uint_to_float(u32::MAX), u32::MAX as f32);
    }

    #[test]
    fn float_to_half_boundaries() {
        // 65504 <= |f| < 65520 rounds to the largest finite half, NOT inf.
        assert_eq!(float_to_half(65504.0).bits(), 0x7bff);
        assert_eq!(float_to_half(65505.0).bits(), 0x7bff);
        assert_eq!(float_to_half(65519.0).bits(), 0x7bff);
        assert_eq!(float_to_half(65520.0).bits(), 0x7c00);
        assert_eq!(float_to_half(70000.0).bits(), 0x7c00);
        assert_eq!(float_to_half(-65505.0).bits(), 0xfbff);
    }
}
