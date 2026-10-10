//! Port of OpenEXR `ImfBytesAttribute.h/.cpp` — the `"bytes"` attribute
//! (new in OpenEXR 3.4).
//!
//! Unlike [`OpaqueAttribute`][crate::attr::OpaqueAttribute], which is the
//! reader's catch-all for a type name it does not recognise, `bytes` is a
//! *known* registered type: arbitrary binary data with an optional MIME-style
//! type-hint string, explicitly intended for storing opaque payloads (as
//! opposed to `OpaqueAttribute`'s "I don't understand this type" fallback).
//!
//! Port target (pinned tag v3.4.15 — new in 3.4, not present in 3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfBytesAttribute.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfBytesAttribute.cpp>

use crate::attr::attribute::{AttrValue, TypedAttribute};
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

/// `Xdr::size<uint32_t>()` — the 4-byte type-hint length header.
const HINT_LENGTH_FIELD: usize = 4;

/// On-disk form of a `bytes` attribute — port of `class BytesAttribute`'s data
/// members (`typeHint`, `_data`).
///
/// **On-disk layout:** `hintLength` (`u32` LE) · `hintLength` raw bytes (the
/// type-hint string, byte-opaque — not required to be UTF-8, matching C++
/// `std::string` semantics) · the remaining `size - 4 - hintLength` bytes as
/// the payload.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Bytes {
    /// Optional MIME-style hint describing the payload's format (e.g.
    /// `"text/plain"`), stored byte-opaque like C++ `std::string typeHint`.
    pub type_hint: Vec<u8>,
    /// The raw, un-interpreted payload bytes (`BytesAttribute::_data`).
    pub data: Vec<u8>,
}

impl Bytes {
    /// Construct from a type-hint string and raw payload bytes — mirrors
    /// `BytesAttribute(size_t size, const void* data, const std::string&
    /// typeHint)`.
    pub fn new(type_hint: &str, data: Vec<u8>) -> Self {
        Self {
            type_hint: type_hint.as_bytes().to_vec(),
            data,
        }
    }
}

impl AttrValue for Bytes {
    fn static_type_name() -> &'static str {
        // ImfBytesAttribute.cpp: BytesAttribute::staticTypeName() -> "bytes"
        "bytes"
    }

    /// Mirrors `BytesAttribute::writeValueTo`: `hintLength` (`u32`) + hint
    /// bytes (only when non-empty) + the raw payload.
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        let hint_len = u32::try_from(self.type_hint.len()).map_err(|_| {
            ExrError::Overflow(format!(
                "bytes attribute type-hint length {} exceeds the u32 wire limit",
                self.type_hint.len()
            ))
        })?;
        xdr::write_u32(o, hint_len)?;
        if !self.type_hint.is_empty() {
            xdr::write_chars(o, &self.type_hint)?;
        }
        xdr::write_chars(o, &self.data)
    }

    /// Mirrors `BytesAttribute::readValueFrom`: reads `hintLength`, rejects a
    /// declared `size` too small to hold `4 + hintLength` bytes (C++
    /// `InputExc`; 3.4 core additionally rejects a negative hint length,
    /// which cannot arise here since `hintLength` is unsigned on the wire),
    /// then reads the hint and the remaining bytes as the payload.
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        if size < 0 {
            return Err(ExrError::Input(
                "Negative size field reading bytes attribute".into(),
            ));
        }
        let hint_length = xdr::read_u32(i)? as usize;
        if (size as i64) < (HINT_LENGTH_FIELD as i64 + hint_length as i64) {
            return Err(ExrError::Input(format!(
                "Invalid bytes attribute type string length: {hint_length}"
            )));
        }
        let type_hint = xdr::read_bounded_bytes(i, hint_length)?;
        let data_size = size as usize - HINT_LENGTH_FIELD - hint_length;
        let data = xdr::read_bounded_bytes(i, data_size)?;
        Ok(Self { type_hint, data })
    }
}

/// `typedef TypedAttribute<Bytes> BytesAttribute;`
pub type BytesAttribute = TypedAttribute<Bytes>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(b: &Bytes) -> Vec<u8> {
        let mut o = MemOStream::new();
        b.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(Bytes::static_type_name(), "bytes");
    }

    #[test]
    fn round_trip_with_hint() {
        let b = Bytes::new("text/plain", b"hello world".to_vec());
        let bytes = write_bytes(&b);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = Bytes::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn round_trip_without_hint() {
        let b = Bytes { type_hint: Vec::new(), data: vec![1, 2, 3, 4] };
        let bytes = write_bytes(&b);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = Bytes::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn exact_bytes_layout() {
        let b = Bytes::new("ab", vec![9, 9]);
        let bytes = write_bytes(&b);
        // hintLength(4) + "ab"(2) + data(2) = 8
        assert_eq!(bytes, vec![2, 0, 0, 0, b'a', b'b', 9, 9]);
    }

    /// A declared size too small for `4 + hintLength` is rejected.
    #[test]
    fn read_value_rejects_size_too_small_for_hint() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&10u32.to_le_bytes()); // claims a 10-byte hint
        let mut i = MemIStream::new(raw);
        // declared size (5) is smaller than 4 + hintLength(10).
        assert!(Bytes::read_value(&mut i, 5, 0).is_err());
    }

    /// A round-trip through the `TypedAttribute<Bytes>` alias, exercised via
    /// the registry, since `BytesAttribute` is now a registered type.
    #[test]
    fn registered_via_header() {
        use crate::header::Header;
        let mut h = Header::new();
        h.channels_mut().unwrap();
        h.insert("myBytes", BytesAttribute::new(Bytes::new("application/json", b"{}".to_vec())))
            .unwrap();
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let mut i = MemIStream::new(o.into_data());
        let (back, _) = Header::read_from(&mut i).unwrap();
        let attr = back.get("myBytes").unwrap();
        assert_eq!(attr.type_name(), b"bytes");
        let typed = attr
            .as_any()
            .downcast_ref::<BytesAttribute>()
            .expect("registry must construct a typed BytesAttribute, not OpaqueAttribute");
        assert_eq!(typed.value.data, b"{}".to_vec());
    }
}
