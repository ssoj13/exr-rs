//! Port of OpenEXR `ImfDeepImageState.h` and `ImfDeepImageStateAttribute.*`.
//!
//! Defines the [`DeepImageState`] enum and its 1-byte wire serialisation.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepImageState.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepImageStateAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::Result;
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// DeepImageState enum — mirrors `enum DeepImageState` in ImfDeepImageState.h
// ---------------------------------------------------------------------------

/// Describes how orderly the pixel samples in a deep image are.
///
/// Discriminants match the C++ enum exactly.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepImageState {
    /// Neither sorted nor non-overlapping.
    Messy = 0,
    /// All pixels are sorted by depth.
    Sorted = 1,
    /// No two samples in any pixel overlap in depth.
    NonOverlapping = 2,
    /// Sorted AND non-overlapping.
    Tidy = 3,
}

impl DeepImageState {
    /// Construct from the raw `u8` wire value.
    ///
    /// `v` is assumed already validated to be `< 4` (see
    /// [`read_value`][AttrValue::read_value], which now rejects out-of-range
    /// values before calling this). The `_ => Messy` arm only exists so
    /// the function stays total; it is unreachable via the attribute reader.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Messy,
            1 => Self::Sorted,
            2 => Self::NonOverlapping,
            3 => Self::Tidy,
            _ => Self::Messy,
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (1 byte on disk)
// ---------------------------------------------------------------------------

impl AttrValue for DeepImageState {
    fn static_type_name() -> &'static str {
        // ImfDeepImageStateAttribute.cpp: return "deepImageState";
        "deepImageState"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: unsigned char tmp = _value; Xdr::write<StreamIO>(os, tmp);
        xdr::write_u8_val(o, *self as u8)
    }

    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        // C++ 3.4 core `extract_attr_uint8(.., EXR_DIS_LAST_TYPE=4)`
        // (`OpenEXRCore/parse_header.c:432-453`): `attrsz != 1` ->
        // EXR_ERR_ATTR_SIZE_MISMATCH, `value >= 4` -> EXR_ERR_INVALID_ATTR.
        // Both fail the attribute (the header reader's non-strict loop then
        // drops it) instead of silently remapping the byte.
        // <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L445>
        if size != 1 {
            return Err(crate::error::ExrError::Input(format!(
                "invalid size {size} for deepImageState attribute (expected 1)"
            )));
        }
        let raw = xdr::read_u8_val(i)?;
        if raw >= 4 {
            return Err(crate::error::ExrError::Input(format!(
                "deepImageState value {raw} is out of range (expected 0..=3)"
            )));
        }
        Ok(Self::from_u8(raw))
    }
}

/// `typedef TypedAttribute<DeepImageState> DeepImageStateAttribute;`
pub type DeepImageStateAttribute = crate::attr::attribute::TypedAttribute<DeepImageState>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(s: DeepImageState) -> Vec<u8> {
        let mut o = MemOStream::new();
        s.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(DeepImageState::static_type_name(), "deepImageState");
    }

    #[test]
    fn exact_bytes() {
        assert_eq!(write_bytes(DeepImageState::Messy), vec![0]);
        assert_eq!(write_bytes(DeepImageState::Sorted), vec![1]);
        assert_eq!(write_bytes(DeepImageState::NonOverlapping), vec![2]);
        assert_eq!(write_bytes(DeepImageState::Tidy), vec![3]);
    }

    #[test]
    fn round_trip() {
        for raw in 0u8..4 {
            let s = DeepImageState::from_u8(raw);
            let bytes = write_bytes(s);
            let mut i = MemIStream::new(bytes);
            let back = DeepImageState::read_value(&mut i, 1, 0).unwrap();
            assert_eq!(back, s);
        }
    }

    /// An out-of-range raw value must be rejected, not silently
    /// remapped to `Messy`.
    #[test]
    fn read_value_rejects_out_of_range() {
        let mut i = MemIStream::new(vec![7u8]);
        assert!(DeepImageState::read_value(&mut i, 1, 0).is_err());
    }

    /// A declared size other than 1 must be rejected.
    #[test]
    fn read_value_rejects_wrong_size() {
        let mut i = MemIStream::new(vec![0u8, 0u8]);
        assert!(DeepImageState::read_value(&mut i, 2, 0).is_err());
    }
}
