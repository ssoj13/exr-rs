//! Port of OpenEXR `ImfLineOrder.h` and `ImfLineOrderAttribute.*`.
//!
//! Defines the [`LineOrder`] enum and its 1-byte wire serialisation.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfLineOrder.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfLineOrderAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// LineOrder enum — mirrors `enum LineOrder` in ImfLineOrder.h
// ---------------------------------------------------------------------------

/// Scan-line storage order, or tile write order.
///
/// Discriminants match the C++ enum exactly.  `NumLineOrders` (3) is the
/// sentinel for invalid values, mirroring the C++ handling in
/// `ImfLineOrderAttribute.cpp`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineOrder {
    /// First scan line has the lowest y coordinate.
    IncreasingY = 0,
    /// First scan line has the highest y coordinate.
    DecreasingY = 1,
    /// Tiles written in random order (tiled files only).
    RandomY = 2,
    /// Sentinel for a caller-built invalid value (`Header::sanity_check` rejects
    /// it); reading a file never produces it, see [`AttrValue::read_value`].
    Invalid = 3,
}

impl LineOrder {
    /// Construct from the raw `u8` wire value.
    ///
    /// Unknown values yield `LineOrder::Invalid`, mirroring
    /// `ImfLineOrderAttribute::readValueFrom`.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::IncreasingY,
            1 => Self::DecreasingY,
            2 => Self::RandomY,
            _ => Self::Invalid,
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (1 byte on disk)
// ---------------------------------------------------------------------------

impl AttrValue for LineOrder {
    fn static_type_name() -> &'static str {
        // ImfLineOrderAttribute.cpp: return "lineOrder";
        "lineOrder"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: unsigned char tmp = _value; Xdr::write<StreamIO>(os, tmp);
        xdr::write_u8_val(o, *self as u8)
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // C++ 3.4 core `extract_attr_uint8(.., EXR_LINEORDER_LAST_TYPE=3)`
        // (`parse_header.c:423-456`, reached for the required `lineOrder` at
        // `:1237-1245` and for any other attribute of this type at `:2162-2167`):
        // `value >= 3` fails the attribute with `EXR_ERR_INVALID_ATTR`, the byte
        // already consumed, so the non-strict header loop drops it and the
        // required-attribute pass defaults `lineOrder` to INCREASING_Y. The
        // legacy `readValueFrom`'s NUM_LINEORDERS sentinel is never stored on read.
        let raw = xdr::read_u8_val(i)?;
        match Self::from_u8(raw) {
            Self::Invalid => Err(ExrError::Input(format!(
                "lineOrder value {raw} is out of range (expected 0, 1 or 2)"
            ))),
            lo => Ok(lo),
        }
    }
}

/// `using LineOrderAttribute = TypedAttribute<LineOrder>;`
pub type LineOrderAttribute = crate::attr::attribute::TypedAttribute<LineOrder>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(lo: LineOrder) -> Vec<u8> {
        let mut o = MemOStream::new();
        lo.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(LineOrder::static_type_name(), "lineOrder");
    }

    #[test]
    fn exact_bytes() {
        assert_eq!(write_bytes(LineOrder::IncreasingY), vec![0]);
        assert_eq!(write_bytes(LineOrder::DecreasingY), vec![1]);
        assert_eq!(write_bytes(LineOrder::RandomY), vec![2]);
    }

    #[test]
    fn round_trip() {
        for raw in 0u8..3 {
            let lo = LineOrder::from_u8(raw);
            let bytes = write_bytes(lo);
            let mut i = MemIStream::new(bytes);
            let back = LineOrder::read_value(&mut i, 1, 0).unwrap();
            assert_eq!(back, lo);
        }
    }

    #[test]
    fn read_value_rejects_out_of_range() {
        for raw in [3u8, 99, 128, 255] {
            let mut i = MemIStream::new(vec![raw]);
            assert!(LineOrder::read_value(&mut i, 1, 0).is_err(), "{raw}");
        }
    }
}
