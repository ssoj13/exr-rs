//! Port of OpenEXR `ImfChromaticities.h/.cpp` and `ImfChromaticitiesAttribute.*`.
//!
//! Defines [`Chromaticities`] (8 × f32 = 32 bytes on disk) with the Rec. ITU-R
//! BT.709-3 defaults, and the wire serialisation.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChromaticities.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChromaticities.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChromaticitiesAttribute.cpp>

use imath_rs::M44f;

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// Chromaticities — mirrors `struct Chromaticities` in ImfChromaticities.h
// ---------------------------------------------------------------------------

/// CIE (x, y) chromaticities for the red, green, blue and white primaries.
///
/// **On-disk layout (32 bytes):** red.x, red.y, green.x, green.y, blue.x,
/// blue.y, white.x, white.y — all `f32` LE.
///
/// The default constructor produces the Rec. ITU-R BT.709-3 primaries,
/// matching the C++ `Chromaticities()` default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticities {
    /// CIE x, y of the red primary.
    pub red: [f32; 2],
    /// CIE x, y of the green primary.
    pub green: [f32; 2],
    /// CIE x, y of the blue primary.
    pub blue: [f32; 2],
    /// CIE x, y of the white point.
    pub white: [f32; 2],
}

impl Default for Chromaticities {
    /// Rec. ITU-R BT.709-3 primaries — matches C++ `Chromaticities()` default.
    fn default() -> Self {
        Self {
            red:   [0.6400, 0.3300],
            green: [0.3000, 0.6000],
            blue:  [0.1500, 0.0600],
            white: [0.3127, 0.3290],
        }
    }
}

// ---------------------------------------------------------------------------
// Colour-science conversions — port of ImfChromaticities.cpp RGBtoXYZ + the
// RgbaYca::computeYw weights (used by the luminance-chroma read path).
// ---------------------------------------------------------------------------

impl Chromaticities {
    /// Port of `Imf::RGBtoXYZ` (ImfChromaticities.cpp): the RGB→CIE-XYZ matrix for
    /// these primaries, scaled so RGB `(1,1,1)` maps to a white of luminance `y`.
    /// Used to derive the luminance weights. Returns [`ExrError::Arg`] for
    /// degenerate primaries — mirrors the C++ `std::invalid_argument` (no panic).
    pub fn rgb_to_xyz(&self, y: f32) -> Result<M44f> {
        let [rx, ry] = self.red;
        let [gx, gy] = self.green;
        let [bx, by] = self.blue;
        let [wx, wy] = self.white;

        // X and Z of RGB (1,1,1) = "white"; guard the division that rounds to zero.
        if wy.abs() <= 1.0 && (wx * y).abs() >= wy.abs() * f32::MAX {
            return Err(ExrError::Arg(
                "Bad chromaticities: white.y cannot be zero".into(),
            ));
        }
        let big_x = wx * y / wy;
        let big_z = (1.0 - wx - wy) * y / wy;

        // Scale-factor numerators and common denominator.
        let d = rx * (by - gy) + bx * (gy - ry) + gx * (ry - by);
        let sr_n = big_x * (by - gy) - gx * (y * (by - 1.0) + by * (big_x + big_z))
            + bx * (y * (gy - 1.0) + gy * (big_x + big_z));
        let sg_n = big_x * (ry - by) + rx * (y * (by - 1.0) + by * (big_x + big_z))
            - bx * (y * (ry - 1.0) + ry * (big_x + big_z));
        let sb_n = big_x * (gy - ry) - rx * (y * (gy - 1.0) + gy * (big_x + big_z))
            + gx * (y * (ry - 1.0) + ry * (big_x + big_z));

        if d.abs() < 1.0
            && (sr_n.abs() >= d.abs() * f32::MAX
                || sg_n.abs() >= d.abs() * f32::MAX
                || sb_n.abs() >= d.abs() * f32::MAX)
        {
            return Err(ExrError::Arg(
                "Bad chromaticities: RGBtoXYZ matrix is degenerate".into(),
            ));
        }
        let (sr, sg, sb) = (sr_n / d, sg_n / d, sb_n / d);

        // Assemble: rows 0–2 carry the colour transform; row 3 stays identity.
        let mut m = M44f::default();
        m[0][0] = sr * rx;
        m[0][1] = sr * ry;
        m[0][2] = sr * (1.0 - rx - ry);
        m[1][0] = sg * gx;
        m[1][1] = sg * gy;
        m[1][2] = sg * (1.0 - gx - gy);
        m[2][0] = sb * bx;
        m[2][1] = sb * by;
        m[2][2] = sb * (1.0 - bx - by);
        Ok(m)
    }

    /// Port of `Imf::XYZtoRGB` (ImfChromaticities.cpp): the CIE-XYZ→RGB matrix
    /// for these primaries — mathematically the inverse of
    /// [`rgb_to_xyz`][Self::rgb_to_xyz]. Returns [`ExrError::Arg`] for
    /// degenerate primaries, mirroring the C++ throw from the underlying
    /// matrix inversion.
    pub fn xyz_to_rgb(&self, y: f32) -> Result<M44f> {
        let forward = self.rgb_to_xyz(y)?;
        forward
            .inverse_exc()
            .map_err(|e| ExrError::Arg(format!("Bad chromaticities: {e}")))
    }

    /// Port of `RgbaYca::computeYw`: the RGB→luminance weights `[wr, wg, wb]` for
    /// these primaries (the Y column of `rgb_to_xyz(1)`, normalised to sum 1).
    pub fn compute_yw(&self) -> Result<[f32; 3]> {
        let m = self.rgb_to_xyz(1.0)?;
        let sum = m[0][1] + m[1][1] + m[2][1];
        Ok([m[0][1] / sum, m[1][1] / sum, m[2][1] / sum])
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (8 × f32 = 32 bytes on disk)
// ---------------------------------------------------------------------------

impl AttrValue for Chromaticities {
    fn static_type_name() -> &'static str {
        // ImfChromaticitiesAttribute.cpp: return "chromaticities";
        "chromaticities"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: Xdr::write(os, red.x); Xdr::write(os, red.y);
        //      Xdr::write(os, green.x); Xdr::write(os, green.y);
        //      Xdr::write(os, blue.x); Xdr::write(os, blue.y);
        //      Xdr::write(os, white.x); Xdr::write(os, white.y);
        xdr::write_f32(o, self.red[0])?;
        xdr::write_f32(o, self.red[1])?;
        xdr::write_f32(o, self.green[0])?;
        xdr::write_f32(o, self.green[1])?;
        xdr::write_f32(o, self.blue[0])?;
        xdr::write_f32(o, self.blue[1])?;
        xdr::write_f32(o, self.white[0])?;
        xdr::write_f32(o, self.white[1])
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        Ok(Self {
            red:   [xdr::read_f32(i)?, xdr::read_f32(i)?],
            green: [xdr::read_f32(i)?, xdr::read_f32(i)?],
            blue:  [xdr::read_f32(i)?, xdr::read_f32(i)?],
            white: [xdr::read_f32(i)?, xdr::read_f32(i)?],
        })
    }
}

/// `typedef TypedAttribute<Chromaticities> ChromaticitiesAttribute;`
pub type ChromaticitiesAttribute = crate::attr::attribute::TypedAttribute<Chromaticities>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(c: &Chromaticities) -> Vec<u8> {
        let mut o = MemOStream::new();
        c.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(Chromaticities::static_type_name(), "chromaticities");
    }

    #[test]
    fn size_is_32_bytes() {
        assert_eq!(write_bytes(&Chromaticities::default()).len(), 32);
    }

    #[test]
    fn round_trip() {
        let c = Chromaticities::default();
        let bytes = write_bytes(&c);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = Chromaticities::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, c);
    }

    /// `xyz_to_rgb` must be the inverse of `rgb_to_xyz`.
    #[test]
    fn xyz_to_rgb_inverts_rgb_to_xyz() {
        let c = Chromaticities::default();
        let fwd = c.rgb_to_xyz(1.0).unwrap();
        let inv = c.xyz_to_rgb(1.0).unwrap();
        let round_trip = fwd.mul_matrix(&inv);
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (round_trip[row][col] - expected).abs() < 1e-4,
                    "[{row}][{col}] = {}",
                    round_trip[row][col]
                );
            }
        }
    }

    #[test]
    fn yw_matches_rec709() {
        // The Rec.709 (default) primaries yield the standard luminance weights
        // 0.2126 / 0.7152 / 0.0722 (BT.709). This validates rgb_to_xyz + compute_yw.
        let yw = Chromaticities::default().compute_yw().unwrap();
        assert!((yw[0] - 0.2126).abs() < 1e-3, "wr={}", yw[0]);
        assert!((yw[1] - 0.7152).abs() < 1e-3, "wg={}", yw[1]);
        assert!((yw[2] - 0.0722).abs() < 1e-3, "wb={}", yw[2]);
        assert!((yw[0] + yw[1] + yw[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn exact_bytes_white_point() {
        // white.x = 0.3127f32, white.y = 0.3290f32
        // We know them as LE f32; just check round-trip at the f32 level.
        let c = Chromaticities::default();
        let bytes = write_bytes(&c);
        // bytes[24..28] = white.x = 0.3127f32
        let wx = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let wy = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        assert!((wx - 0.3127).abs() < 1e-6);
        assert!((wy - 0.3290).abs() < 1e-6);
    }
}
