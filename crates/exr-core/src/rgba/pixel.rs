//! `Rgba` pixel — port of `struct Rgba` in `ImfRgba.h`.
//!
//! Four `half` components. In the luminance-chroma (YCA) path the same struct
//! carries either RGBA or YCA data (`g` = luminance Y, `r` = RY chroma, `b` = BY
//! chroma, `a` = alpha), exactly as the C++ `Rgba` is reused by `ImfRgbaYca`.

use imath_rs::Half;

/// An RGBA (or YCA) pixel of four `half` values. Mirrors C++ `Imf::Rgba`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
    pub r: Half,
    pub g: Half,
    pub b: Half,
    pub a: Half,
}

impl Rgba {
    /// Construct from four `half` values.
    pub fn new(r: Half, g: Half, b: Half, a: Half) -> Self {
        Self { r, g, b, a }
    }

    /// Construct from four `f32` values (each rounded to `half`).
    pub fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: Half::from_f32(r),
            g: Half::from_f32(g),
            b: Half::from_f32(b),
            a: Half::from_f32(a),
        }
    }
}

impl Default for Rgba {
    /// All channels zero (`half` 0.0).
    fn default() -> Self {
        let z = Half::from_f32(0.0);
        Self { r: z, g: z, b: z, a: z }
    }
}

/// Channels present in an RGBA file — port of `enum RgbaChannels` (`ImfRgba.h`).
///
/// A plain newtype-over-bitmask rather than the `bitflags` crate: the crate's
/// convention is no external dependencies for format-intrinsic types, and the
/// set of operations needed here (`|`, [`contains`](Self::contains),
/// [`intersects`](Self::intersects)) is small.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbaChannels(u32);

impl RgbaChannels {
    pub const NONE: Self = Self(0x00);
    pub const WRITE_R: Self = Self(0x01);
    pub const WRITE_G: Self = Self(0x02);
    pub const WRITE_B: Self = Self(0x04);
    pub const WRITE_A: Self = Self(0x08);
    /// Luminance, for black-and-white images, or in combination with chroma.
    pub const WRITE_Y: Self = Self(0x10);
    /// Chroma (two subsampled channels, RY and BY), scanline files only.
    pub const WRITE_C: Self = Self(0x20);
    pub const WRITE_RGB: Self = Self(0x07);
    pub const WRITE_RGBA: Self = Self(0x0f);
    pub const WRITE_YC: Self = Self(0x30);
    pub const WRITE_YA: Self = Self(0x18);
    pub const WRITE_YCA: Self = Self(0x38);

    /// `true` when every bit set in `other` is also set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// `true` when `self` and `other` share at least one set bit.
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for RgbaChannels {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for RgbaChannels {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
