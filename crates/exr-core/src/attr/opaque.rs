//! Port of OpenEXR `ImfOpaqueAttribute.h/.cpp`.
//!
//! When a file is read, attributes whose type-name is **not** recognised by the
//! reader are stored as [`OpaqueAttribute`] objects: the raw value bytes plus
//! the type-name string are kept verbatim so the attribute can be copied and
//! written back out byte-for-byte without the reader understanding its meaning.
//! This is the catch-all the registry returns for any unknown type-name.
//!
//! Unlike the other typed attributes, an opaque attribute carries its type-name
//! at run time (it is data, not a compile-time constant), so it cannot be a
//! [`TypedAttribute<T>`][crate::attr::attribute::TypedAttribute]; it implements
//! the object-safe [`Attribute`] trait directly, exactly as the C++
//! `OpaqueAttribute` derives straight from `class Attribute`.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOpaqueAttribute.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfOpaqueAttribute.cpp>

use std::any::Any;

use crate::attr::attribute::Attribute;
use crate::attr::name::Name;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

/// Holds an unrecognised attribute's raw bytes and type-name — port of
/// `class OpaqueAttribute`.
///
/// The value bytes are opaque to this program (`writeValueTo` writes them back
/// unchanged; `readValueFrom` reads exactly `size` bytes), which is what makes
/// unknown attributes round-trip losslessly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueAttribute {
    /// The wire type-name string (`OpaqueAttribute::_typeName`).
    ///
    /// [`Name`] (raw bytes), not `String` — C-08c: the type-name is discovered
    /// at read time from the wire, and C++ never checks its encoding, so an
    /// unregistered type whose on-disk name is not valid UTF-8 still
    /// round-trips byte-exact.
    type_name: Name,
    /// The raw, un-interpreted value bytes (`OpaqueAttribute::_data`).
    ///
    /// `_dataSize` in C++ is just `data.len()` here.
    data: Vec<u8>,
}

impl OpaqueAttribute {
    /// Construct an empty opaque attribute for `type_name`.
    ///
    /// Mirrors `OpaqueAttribute(const char typeName[])`.
    pub fn new(type_name: impl Into<Name>) -> Self {
        Self {
            type_name: type_name.into(),
            data: Vec::new(),
        }
    }

    /// Construct from a type-name and raw value bytes.
    ///
    /// Mirrors `OpaqueAttribute(const char typeName[], long dataSize, const void* data)`.
    pub fn with_data(type_name: impl Into<Name>, data: Vec<u8>) -> Self {
        Self {
            type_name: type_name.into(),
            data,
        }
    }

    /// Number of raw value bytes — mirrors `OpaqueAttribute::dataSize()`.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// The raw value bytes — mirrors `OpaqueAttribute::data()`.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Read a fully-formed opaque attribute of `type_name` from a stream.
    ///
    /// Mirrors the C++ pattern of constructing an `OpaqueAttribute(typeName)`
    /// then calling `readValueFrom(is, size, version)`: reads exactly `size`
    /// raw bytes (`Xdr::read<StreamIO>(is, _data, size)`).  A negative `size`
    /// is rejected rather than silently treated as zero.
    pub fn read_from(
        type_name: impl Into<Name>,
        i: &mut dyn IStream,
        size: i32,
        _version: i32,
    ) -> Result<Self> {
        if size < 0 {
            return Err(ExrError::Input(
                "Negative size field reading opaque attribute".into(),
            ));
        }
        // Bounds-check the declared size against the stream before allocating.
        let data = xdr::read_bounded_bytes(i, size as usize)?;
        Ok(Self {
            type_name: type_name.into(),
            data,
        })
    }
}

impl Attribute for OpaqueAttribute {
    /// Mirrors `OpaqueAttribute::typeName()`.
    ///
    /// Borrows the runtime type-name bytes stored in `self.type_name`. The
    /// trait returns `&[u8]` (lifetime tied to `&self`) so no leak is needed —
    /// typed attributes return `T::static_type_name().as_bytes()`.
    fn type_name(&self) -> &[u8] {
        self.type_name.as_bytes()
    }

    /// Mirrors `OpaqueAttribute::writeValueTo` — writes the stored raw bytes.
    fn write_value_to(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // Xdr::write<StreamIO>(os, _data, _dataSize);  — verbatim bytes.
        xdr::write_chars(o, &self.data)
    }

    fn clone_attr(&self) -> Box<dyn Attribute> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    #[test]
    fn round_trip_lossless() {
        // An unknown type's raw bytes must survive read -> write unchanged.
        let raw = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
        let size = raw.len() as i32;
        let mut i = MemIStream::new(raw.clone());

        let attr = OpaqueAttribute::read_from("mytype", &mut i, size, 0).unwrap();
        assert_eq!(attr.type_name(), b"mytype");
        assert_eq!(attr.data_size(), raw.len());
        assert_eq!(attr.data(), raw.as_slice());

        let mut o = MemOStream::new();
        attr.write_value_to(&mut o, 0).unwrap();
        assert_eq!(o.into_data(), raw);
    }

    #[test]
    fn empty_payload() {
        let mut i = MemIStream::new(vec![]);
        let attr = OpaqueAttribute::read_from("empty", &mut i, 0, 0).unwrap();
        assert_eq!(attr.data_size(), 0);

        let mut o = MemOStream::new();
        attr.write_value_to(&mut o, 0).unwrap();
        assert!(o.into_data().is_empty());
    }

    // A tiny buffer declaring a ~2 GiB opaque payload must error
    // (bounded by the stream's remaining bytes) rather than allocate/abort.
    #[test]
    fn huge_size_rejected_without_alloc() {
        let mut i = MemIStream::new(vec![0u8; 4]);
        let err = OpaqueAttribute::read_from("mytype", &mut i, i32::MAX, 0)
            .expect_err("huge declared size must be rejected");
        assert!(matches!(err, ExrError::Input(_)));
    }

    #[test]
    fn clone_via_trait() {
        let attr = OpaqueAttribute::with_data("t", vec![1, 2, 3]);
        let boxed: Box<dyn Attribute> = Box::new(attr);
        let cloned = boxed.clone_attr();
        let concrete = cloned
            .as_any()
            .downcast_ref::<OpaqueAttribute>()
            .expect("downcast");
        assert_eq!(concrete.data(), &[1, 2, 3]);
    }
}
