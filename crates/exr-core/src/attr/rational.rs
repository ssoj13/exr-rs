//! Port of OpenEXR `ImfRational.h` and `ImfRationalAttribute.*`.
//!
//! Defines [`Rational`] (n: i32, d: u32) and its 8-byte wire serialisation.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfRational.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfRationalAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::Result;
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// Rational — mirrors `class Rational` in ImfRational.h
// ---------------------------------------------------------------------------

/// A rational number n/d.
///
/// **Semantics (from C++ header):**
/// - `d > 0`: value is `n/d`
/// - `n > 0, d == 0`: positive infinity
/// - `n < 0, d == 0`: negative infinity
/// - `n == 0, d == 0`: NaN
///
/// **On-disk layout (8 bytes):** `n` as `i32` LE + `d` as `u32` LE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    /// Numerator (signed).
    pub n: i32,
    /// Denominator (unsigned).
    pub d: u32,
}

impl Rational {
    /// Construct — mirrors `Rational(int n, int d)`.
    pub fn new(n: i32, d: u32) -> Self {
        Self { n, d }
    }

    /// Convert to `f64` — mirrors `operator double()`.
    pub fn as_f64(self) -> f64 {
        self.n as f64 / self.d as f64
    }

    /// Construct the best rational approximation of a `double` — a 1:1 port
    /// of `Rational::Rational(double x)` (continued-fraction expansion
    /// contributed by Greg Ward).
    ///
    /// `x == NaN` yields `n = 0, d = 0`; `|x| >= 2^31 - 0.5` yields the
    /// signed infinity encoding (`n = ±1, d = 0`), matching the C++ exactly.
    ///
    /// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfRational.cpp>
    pub fn from_f64(x: f64) -> Self {
        let (sign, x) = if x.is_nan() {
            return Self { n: 0, d: 0 };
        } else if x >= 0.0 {
            (1i32, x)
        } else {
            (-1i32, -x)
        };

        if x >= (1u32 << 31) as f64 - 0.5 {
            return Self { n: sign, d: 0 };
        }

        let e = (if x < 1.0 { 1.0 } else { x }) / (1u32 << 30) as f64;
        let d = rational_denom(x, e) as u32;
        let n = sign * (x * d as f64 + 0.5).floor() as i32;
        Self { n, d }
    }
}

// ---------------------------------------------------------------------------
// `from_f64` helpers — the anonymous-namespace `frac`/`square`/`denom` in
// `ImfRational.cpp`, ported 1:1 (including the mutual recursion in `denom`).
// ---------------------------------------------------------------------------

fn rational_frac(x: f64, e: f64) -> f64 {
    x - (x + e).floor()
}

fn rational_square(x: f64) -> f64 {
    x * x
}

fn rational_denom(x: f64, e: f64) -> f64 {
    if e > rational_frac(x, e) {
        1.0
    } else {
        let r = rational_frac(1.0 / x, e);
        if e > r {
            (1.0 / x + e).floor()
        } else {
            rational_denom(rational_frac(1.0 / r, e), e / rational_square(x * r))
                + (1.0 / x + e).floor() * rational_denom(rational_frac(1.0 / x, e), e / rational_square(x))
        }
    }
}

impl Default for Rational {
    /// Mirrors `Rational() : n(0), d(1) {}`.
    fn default() -> Self {
        Self { n: 0, d: 1 }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (i32 + u32 = 8 bytes on disk)
// ---------------------------------------------------------------------------

impl AttrValue for Rational {
    fn static_type_name() -> &'static str {
        // ImfRationalAttribute.cpp: return "rational";
        "rational"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: Xdr::write(os, _value.n); Xdr::write(os, _value.d);
        xdr::write_i32(o, self.n)?;
        xdr::write_u32(o, self.d)
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // C++: Xdr::read(is, _value.n); Xdr::read(is, _value.d);
        let n = xdr::read_i32(i)?;
        let d = xdr::read_u32(i)?;
        Ok(Self { n, d })
    }
}

/// `typedef TypedAttribute<Rational> RationalAttribute;`
pub type RationalAttribute = crate::attr::attribute::TypedAttribute<Rational>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(r: Rational) -> Vec<u8> {
        let mut o = MemOStream::new();
        r.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(Rational::static_type_name(), "rational");
    }

    /// `Rational::from_f64` (the ported `Rational(double)` ctor)
    /// finds a small exact fraction for common values.
    #[test]
    fn from_f64_exact_fractions() {
        assert_eq!(Rational::from_f64(0.0), Rational { n: 0, d: 1 });
        assert_eq!(Rational::from_f64(24.0).as_f64(), 24.0);
        let r = Rational::from_f64(24000.0 / 1001.0);
        assert_eq!(r.n, 24000);
        assert_eq!(r.d, 1001);
        let r = Rational::from_f64(-2.5);
        assert!((r.as_f64() - -2.5).abs() < 1e-9);
    }

    /// NaN and near-`i32::MAX` magnitudes hit the special-cased
    /// branches (NaN encoding, signed infinity) rather than the continued
    /// fraction.
    #[test]
    fn from_f64_special_values() {
        let nan = Rational::from_f64(f64::NAN);
        assert_eq!(nan, Rational { n: 0, d: 0 });
        let inf = Rational::from_f64(1e30);
        assert_eq!(inf, Rational { n: 1, d: 0 });
        let neg_inf = Rational::from_f64(-1e30);
        assert_eq!(neg_inf, Rational { n: -1, d: 0 });
    }

    #[test]
    fn size_is_8_bytes() {
        assert_eq!(write_bytes(Rational::default()).len(), 8);
    }

    #[test]
    fn exact_bytes() {
        // n=1 i32 LE = 01 00 00 00
        // d=1 u32 LE = 01 00 00 00
        assert_eq!(
            write_bytes(Rational::new(1, 1)),
            vec![0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        );

        // n=24 i32 LE = 18 00 00 00
        // d=1  u32 LE = 01 00 00 00
        assert_eq!(
            write_bytes(Rational::new(24, 1)),
            vec![0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        );

        // n=-1 i32 LE = FF FF FF FF
        // d=0  u32 LE = 00 00 00 00  (negative infinity)
        assert_eq!(
            write_bytes(Rational::new(-1, 0)),
            vec![0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn round_trip() {
        for (n, d) in [(0, 1), (24, 1), (30000, 1001), (-1, 0), (0, 0)] {
            let r = Rational::new(n, d);
            let bytes = write_bytes(r);
            let size = bytes.len() as i32;
            let mut i = MemIStream::new(bytes);
            let back = Rational::read_value(&mut i, size, 0).unwrap();
            assert_eq!(back, r);
        }
    }
}
