//! Port of OpenEXR `ImfTileDescription.h` and `ImfTileDescriptionAttribute.*`.
//!
//! Defines [`LevelMode`], [`LevelRoundingMode`], and [`TileDescription`] with
//! the 9-byte on-disk layout: `u32 xSize + u32 ySize + u8 (mode | roundingMode << 4)`.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTileDescription.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTileDescriptionAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// LevelMode — mirrors `enum LevelMode` in ImfTileDescription.h
// ---------------------------------------------------------------------------

/// MIP/RIP level mode for tiled images.
///
/// Stored in the low nibble of the mode byte on disk.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelMode {
    /// A single resolution level.
    OneLevel = 0,
    /// A complete MIP-map pyramid.
    MipmapLevels = 1,
    /// A complete RIP-map pyramid.
    RipmapLevels = 2,
    /// Sentinel — holds an invalid encoded value.
    ///
    /// The C++ comment: "3 is a special valid enum value that indicates bad values".
    Invalid = 3,
}

impl LevelMode {
    /// Construct from the raw nibble value (bits 3:0 of the mode byte).
    ///
    /// Values > 3 are clamped to 3 (`Invalid`), matching the C++ bounds check:
    /// `if (levelMode > 3) levelMode = 3;`
    pub fn from_nibble(v: u8) -> Self {
        match v & 0x0f {
            0 => Self::OneLevel,
            1 => Self::MipmapLevels,
            2 => Self::RipmapLevels,
            _ => Self::Invalid,
        }
    }
}

// ---------------------------------------------------------------------------
// LevelRoundingMode — mirrors `enum LevelRoundingMode` in ImfTileDescription.h
// ---------------------------------------------------------------------------

/// Rounding direction when computing MIP/RIP level sizes.
///
/// Stored in the high nibble of the mode byte on disk.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelRoundingMode {
    /// Round level sizes down.
    RoundDown = 0,
    /// Round level sizes up.
    RoundUp = 1,
    /// Sentinel — holds an invalid encoded value.
    ///
    /// The C++ comment: "2 is a special enum value for 'bad enum'".
    Invalid = 2,
}

impl LevelRoundingMode {
    /// Construct from the raw nibble value (bits 7:4 of the mode byte).
    ///
    /// Values > 2 are clamped to 2 (`Invalid`), matching the C++ bounds check:
    /// `if (levelRoundingMode > 2) levelRoundingMode = 2;`
    pub fn from_nibble(v: u8) -> Self {
        match v & 0x0f {
            0 => Self::RoundDown,
            1 => Self::RoundUp,
            _ => Self::Invalid,
        }
    }
}

// ---------------------------------------------------------------------------
// TileDescription — mirrors `class TileDescription` in ImfTileDescription.h
// ---------------------------------------------------------------------------

/// Tile size and level mode for tiled OpenEXR images.
///
/// **On-disk layout (9 bytes total):**
/// 1. `xSize` — `u32` LE (4 bytes)
/// 2. `ySize` — `u32` LE (4 bytes)
/// 3. mode byte — `u8`: bits 3:0 = `LevelMode` discriminant,
///    bits 7:4 = `LevelRoundingMode` discriminant.
///
/// C++ default: `xSize=32, ySize=32, mode=ONE_LEVEL, roundingMode=ROUND_DOWN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileDescription {
    /// Width of a tile in pixels.
    pub x_size: u32,
    /// Height of a tile in pixels.
    pub y_size: u32,
    /// Level mode (ONE/MIPMAP/RIPMAP).
    pub mode: LevelMode,
    /// Rounding mode for level-size computation.
    pub rounding_mode: LevelRoundingMode,
}

impl Default for TileDescription {
    /// C++ default: `TileDescription(32, 32, ONE_LEVEL, ROUND_DOWN)`.
    fn default() -> Self {
        Self {
            x_size: 32,
            y_size: 32,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (9 bytes: u32 + u32 + u8)
// ---------------------------------------------------------------------------

impl AttrValue for TileDescription {
    fn static_type_name() -> &'static str {
        // ImfTileDescriptionAttribute.cpp: return "tiledesc";
        "tiledesc"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++:
        //   Xdr::write<StreamIO>(os, _value.xSize);
        //   Xdr::write<StreamIO>(os, _value.ySize);
        //   unsigned char tmp = _value.mode | (_value.roundingMode << 4);
        //   Xdr::write<StreamIO>(os, tmp);
        xdr::write_u32(o, self.x_size)?;
        xdr::write_u32(o, self.y_size)?;
        let mode_byte = (self.mode as u8) | ((self.rounding_mode as u8) << 4);
        xdr::write_u8_val(o, mode_byte)
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // 3.3.11 `TileDescriptionAttribute::readValueFrom` clamped an
        // out-of-range nibble to the `Invalid` sentinel:
        //   unsigned char levelMode = tmp & 0x0f; if (levelMode > 3) levelMode = 3;
        //   unsigned char levelRoundingMode = (tmp >> 4) & 0x0f; if (> 2) = 2;
        //
        // 3.4 core `check_populate_tiles`
        // (`OpenEXRCore/parse_header.c:1586-1603`) instead REJECTS the whole
        // attribute outright when `lev >= EXR_TILE_LAST_TYPE(3)` or
        // `rnd >= EXR_TILE_ROUND_LAST_TYPE(2)` — non-strict then drops it —
        // rather than silently storing an `Invalid` mode/rounding that a
        // written-back file would round-trip unchanged.
        // <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L1586-L1603>
        let x_size = xdr::read_u32(i)?;
        let y_size = xdr::read_u32(i)?;
        let tmp = xdr::read_u8_val(i)?;
        let level_mode = tmp & 0x0f;
        let level_rounding_mode = (tmp >> 4) & 0x0f;
        if level_mode >= 3 {
            return Err(ExrError::Input(format!(
                "Invalid level mode ({level_mode}) in tile description header"
            )));
        }
        if level_rounding_mode >= 2 {
            return Err(ExrError::Input(format!(
                "Invalid rounding mode ({level_rounding_mode}) in tile description header"
            )));
        }
        Ok(Self {
            x_size,
            y_size,
            mode: LevelMode::from_nibble(level_mode),
            rounding_mode: LevelRoundingMode::from_nibble(level_rounding_mode),
        })
    }
}

/// `typedef TypedAttribute<TileDescription> TileDescriptionAttribute;`
pub type TileDescriptionAttribute = crate::attr::attribute::TypedAttribute<TileDescription>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(td: &TileDescription) -> Vec<u8> {
        let mut o = MemOStream::new();
        td.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(TileDescription::static_type_name(), "tiledesc");
    }

    #[test]
    fn exact_bytes_default() {
        // Default: xSize=32, ySize=32, ONE_LEVEL, ROUND_DOWN
        // mode byte = 0 | (0 << 4) = 0x00
        let td = TileDescription::default();
        let bytes = write_bytes(&td);
        assert_eq!(bytes.len(), 9);
        // xSize=32 LE: 0x20 0x00 0x00 0x00
        assert_eq!(&bytes[0..4], &[0x20, 0x00, 0x00, 0x00]);
        // ySize=32 LE: 0x20 0x00 0x00 0x00
        assert_eq!(&bytes[4..8], &[0x20, 0x00, 0x00, 0x00]);
        // mode byte: ONE_LEVEL(0) | (ROUND_DOWN(0) << 4) = 0x00
        assert_eq!(bytes[8], 0x00);
    }

    #[test]
    fn exact_bytes_mipmap_round_up() {
        // MIPMAP_LEVELS=1, ROUND_UP=1 → mode byte = 1 | (1 << 4) = 0x11
        let td = TileDescription {
            x_size: 64,
            y_size: 64,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundUp,
        };
        let bytes = write_bytes(&td);
        assert_eq!(bytes[8], 0x11);
    }

    #[test]
    fn exact_bytes_ripmap_round_down() {
        // RIPMAP_LEVELS=2, ROUND_DOWN=0 → mode byte = 2 | (0 << 4) = 0x02
        let td = TileDescription {
            x_size: 16,
            y_size: 16,
            mode: LevelMode::RipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        let bytes = write_bytes(&td);
        assert_eq!(bytes[8], 0x02);
    }

    #[test]
    fn round_trip() {
        let td = TileDescription {
            x_size: 128,
            y_size: 256,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundUp,
        };
        let bytes = write_bytes(&td);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = TileDescription::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, td);
    }

    /// 3.4 core `check_populate_tiles` REJECTS an out-of-range mode or
    /// rounding nibble outright (the non-strict header reader then drops the
    /// whole `tiles` attribute); 3.3.11 used to silently clamp it to
    /// `Invalid` instead, which this port matched until this fix.
    #[test]
    fn read_value_rejects_invalid_mode_nibble() {
        let raw_bytes: Vec<u8> = vec![
            0x01, 0x00, 0x00, 0x00, // xSize=1
            0x01, 0x00, 0x00, 0x00, // ySize=1
            0x07,                   // levelMode=7 (>= 3, invalid), roundingMode=0
        ];
        let mut i = MemIStream::new(raw_bytes);
        assert!(TileDescription::read_value(&mut i, 9, 0).is_err());
    }

    /// Same for an out-of-range rounding-mode nibble.
    #[test]
    fn read_value_rejects_invalid_rounding_nibble() {
        let raw_bytes: Vec<u8> = vec![
            0x01, 0x00, 0x00, 0x00, // xSize=1
            0x01, 0x00, 0x00, 0x00, // ySize=1
            0x50,                   // levelMode=0, roundingMode=5 (>= 2, invalid)
        ];
        let mut i = MemIStream::new(raw_bytes);
        assert!(TileDescription::read_value(&mut i, 9, 0).is_err());
    }
}
