//! [`ImageAttributes`] — the header metadata an [`Image`](super::Image) carries
//! that is **not** derivable from its own fields.
//!
//! # Why this exists
//!
//! `Image` models windows, compression, line order and a flat channel map. A
//! real EXR header holds much more: `pixelAspectRatio`, `screenWindowCenter`,
//! `screenWindowWidth`, `chromaticities`, `comments`, `owner`, timecodes,
//! Cryptomatte keys, and any number of vendor attributes the reader preserved
//! as [`OpaqueAttribute`](crate::attr::OpaqueAttribute). Before this type the
//! facade dropped all of it on read, so `Image::read` → `Image::write` (and
//! therefore `exr-rs convert`) silently stripped every piece of metadata a file
//! carried.
//!
//! # One representation
//!
//! This is the **single** carrier for those attributes. [`Image::write`] and
//! [`Image::write_with_attrs`] both apply it; there is no separate "preserve
//! metadata" path for the CLI or the Python binding to opt into.
//!
//! # What is *not* carried ([`ImageAttributes::DERIVED`])
//!
//! Two kinds of attribute are deliberately excluded:
//!
//! * **Derived** from `Image`'s own fields — `dataWindow`, `displayWindow`,
//!   `compression`, `lineOrder`, `channels`. Carrying a stale copy could
//!   contradict the fields the caller edited.
//! * **Structural**, i.e. describing a file shape the facade does not write —
//!   `type`, `tiles`, `chunkCount`, `version`, `name`. `Image` always writes a
//!   single-part **scanline** file; a carried `tiles` attribute would make
//!   [`Header::version_field`] set the TILED flag on a file that has no tiles,
//!   and a carried `chunkCount` / `name` would describe a multipart part that
//!   no longer exists.
//!
//! Everything else survives byte-for-byte, including unknown types (the reader
//! keeps those as opaque blobs, and this carrier never inspects a value).

use std::collections::BTreeMap;
use std::collections::btree_map::Iter;

use crate::attr::{Attribute, Name};
use crate::container::version;
use crate::error::{ExrError, Result};
use crate::header::Header;
use crate::io::MemOStream;

/// Header attributes carried alongside an [`Image`](super::Image).
///
/// Ordered by name, exactly like [`Header`]'s own map (C++ `std::map<Name,
/// Attribute*>`), so applying these to a header reproduces the source file's
/// attribute order on the wire.
///
/// Keyed by [`Name`] (raw bytes), not `String` — a non-UTF-8
/// attribute name (however unlikely on a real *attribute*, as opposed to
/// *channel*, identifier) is carried byte-exact through `Image::read` →
/// `Image::write` instead of being silently dropped, matching this crate's
/// own no-silent-fallback rule rather than a C++ citation
/// — `ImageAttributes` itself has no C++ counterpart to cite.
///
/// See the module docs for what is and is not carried.
#[derive(Default)]
pub struct ImageAttributes {
    map: BTreeMap<Name, Box<dyn Attribute>>,
}

impl ImageAttributes {
    /// Attribute names an [`Image`](super::Image) never carries: those derived
    /// from its own fields, plus those describing a file shape the facade does
    /// not write. See the module docs for the reasoning behind each entry.
    ///
    /// This is the single list used by both [`Self::from_header`] (what to
    /// skip) and [`Self::apply_to`] (what to refuse), so the two directions can
    /// not disagree.
    pub const DERIVED: &'static [&'static str] = &[
        // Derived from Image's own fields.
        "channels",
        "compression",
        "dataWindow",
        "displayWindow",
        "lineOrder",
        // Structural: describe a shape the facade does not write.
        "chunkCount",
        "name",
        "tiles",
        "type",
        "version",
    ];

    /// An empty attribute set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `name` is one of the [`Self::DERIVED`] names.
    ///
    /// Generic over `K: AsRef<[u8]>` so both a `&str` literal (every
    /// existing call site) and a raw [`Name`]/`&[u8]` work; every entry in
    /// [`Self::DERIVED`] is ASCII, so a non-UTF-8 `name` simply can never
    /// match one and is correctly "not derived".
    pub fn is_derived<K: AsRef<[u8]> + ?Sized>(name: &K) -> bool {
        let bytes = name.as_ref();
        Self::DERIVED.iter().any(|d| d.as_bytes() == bytes)
    }

    /// Capture every carryable attribute of `header`.
    ///
    /// Attributes are deep-copied (`Attribute::clone_attr`, the port of C++
    /// `Attribute::copy()`), so the result is independent of `header`.
    pub fn from_header(header: &Header) -> Self {
        let mut map = BTreeMap::new();
        for (name, attr) in header.iter() {
            if Self::is_derived(name) {
                continue;
            }
            map.insert(name.clone(), attr.clone_attr());
        }
        Self { map }
    }

    /// Insert every carried attribute into `header`.
    ///
    /// Call this **after** the header's derived attributes are set: a carried
    /// value replaces a same-named, same-typed one already present (that is how
    /// `pixelAspectRatio` / `screenWindow*` from the source file override the
    /// [`Header::new`] defaults).
    ///
    /// # Errors
    /// Propagates [`Header::insert_boxed`], which rejects a value whose type
    /// name differs from an existing attribute of the same name.
    pub fn apply_to(&self, header: &mut Header) -> Result<()> {
        for (name, attr) in &self.map {
            header.insert_boxed(name, attr.clone_attr())?;
        }
        Ok(())
    }

    /// Number of carried attributes.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether nothing is carried.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Whether `name` is carried.
    pub fn contains<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> bool {
        self.map.contains_key(Name::normalize(name.as_ref()))
    }

    /// Borrow a carried attribute.
    pub fn get<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<&dyn Attribute> {
        self.map.get(Name::normalize(name.as_ref())).map(|b| b.as_ref())
    }

    /// Add or replace an attribute.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] for an empty name (matching [`Header::insert_boxed`])
    ///   or for one of the [`Self::DERIVED`] names — those come from `Image`'s
    ///   own fields and setting them here would be silently ignored on write.
    pub fn insert(&mut self, name: impl Into<Name>, attribute: Box<dyn Attribute>) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Image attribute name cannot be an empty string.".into(),
            ));
        }
        if Self::is_derived(&name) {
            return Err(ExrError::Arg(format!(
                "Attribute \"{name}\" is derived from the Image's own fields (or \
                 describes a file shape the facade does not write) and cannot be \
                 carried; set the corresponding Image field instead."
            )));
        }
        self.map.insert(name, attribute);
        Ok(())
    }

    /// Remove an attribute, returning whether it was present.
    pub fn remove<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> bool {
        self.map.remove(Name::normalize(name.as_ref())).is_some()
    }

    /// Iterate carried attributes in name order.
    pub fn iter(&self) -> Iter<'_, Name, Box<dyn Attribute>> {
        self.map.iter()
    }

    /// Serialised identity of one attribute: its wire type name plus the exact
    /// bytes it writes. Two attributes are equal iff both match — which is the
    /// only value-level comparison [`Attribute`] supports (it is object-safe and
    /// deliberately has no `PartialEq`), and it is exactly the notion of
    /// equality that matters for a round trip: the file bytes.
    ///
    /// A value whose serialisation fails compares unequal to everything except
    /// another failure of the same attribute type — it can never be written to a
    /// file either.
    fn wire_identity(attr: &dyn Attribute) -> (String, Option<Vec<u8>>) {
        let mut tmp = MemOStream::new();
        let bytes = attr
            .write_value_to(&mut tmp, version::EXR_VERSION as i32)
            .ok()
            .map(|()| tmp.into_data());
        (String::from_utf8_lossy(attr.type_name()).into_owned(), bytes)
    }
}

impl Clone for ImageAttributes {
    /// Deep clone, mirroring [`Header`]'s own `Clone` (C++ copy-constructs each
    /// attribute via `Attribute::copy()`).
    fn clone(&self) -> Self {
        Self {
            map: self
                .map
                .iter()
                .map(|(name, attr)| (name.clone(), attr.clone_attr()))
                .collect(),
        }
    }
}

impl std::fmt::Debug for ImageAttributes {
    /// `Attribute` is object-safe and carries no `Debug`, so this prints each
    /// attribute's name and wire type — enough to see what an `Image` carries
    /// without the trait growing a formatting method.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.map.iter().map(|(n, a)| (n, a.type_name())))
            .finish()
    }
}

impl PartialEq for ImageAttributes {
    /// Compares by wire identity (see [`Self::wire_identity`]): same names, same
    /// types, same serialised bytes.
    fn eq(&self, other: &Self) -> bool {
        self.map.len() == other.map.len()
            && self.map.iter().zip(other.map.iter()).all(|(a, b)| {
                a.0 == b.0
                    && Self::wire_identity(a.1.as_ref()) == Self::wire_identity(b.1.as_ref())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{ExrString, StringAttribute, TypedAttribute};

    fn header_with(name: &str, value: &str) -> Header {
        let mut h = Header::new();
        h.insert(name, StringAttribute::new(ExrString::from(value)))
            .expect("insert");
        h
    }

    #[test]
    fn from_header_skips_derived_and_keeps_the_rest() {
        let attrs = ImageAttributes::from_header(&header_with("owner", "me"));
        for derived in ImageAttributes::DERIVED {
            assert!(!attrs.contains(derived), "{derived} must not be carried");
        }
        assert!(attrs.contains("owner"));
        // Header::new()'s non-derived defaults ride along, so the facade stops
        // dropping pixelAspectRatio / screenWindow*.
        assert!(attrs.contains("pixelAspectRatio"));
        assert!(attrs.contains("screenWindowCenter"));
        assert!(attrs.contains("screenWindowWidth"));
    }

    #[test]
    fn apply_to_round_trips_a_custom_attribute() {
        let attrs = ImageAttributes::from_header(&header_with("owner", "me"));
        let mut target = Header::new();
        attrs.apply_to(&mut target).expect("apply");
        let got = target
            .typed_attribute::<ExrString>("owner")
            .expect("owner present");
        assert_eq!(got.value.as_str(), Some("me"));
    }

    #[test]
    fn insert_rejects_derived_and_empty_names() {
        let mut attrs = ImageAttributes::new();
        let boxed = || -> Box<dyn Attribute> {
            Box::new(StringAttribute::new(ExrString::from("x")))
        };
        assert!(attrs.insert("", boxed()).is_err());
        assert!(attrs.insert("tiles", boxed()).is_err());
        assert!(attrs.insert("dataWindow", boxed()).is_err());
        assert!(attrs.insert("owner", boxed()).is_ok());
    }

    #[test]
    fn eq_compares_wire_bytes_not_identity() {
        let a = ImageAttributes::from_header(&header_with("owner", "me"));
        let b = ImageAttributes::from_header(&header_with("owner", "me"));
        let c = ImageAttributes::from_header(&header_with("owner", "you"));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, a.clone());
    }

    #[test]
    fn eq_distinguishes_types_with_identical_bytes() {
        // An i32 `1` and a float whose bits happen to differ is easy; the real
        // risk is two types that serialise to the same 4 bytes. `int` 1065353216
        // and `float` 1.0 do exactly that, so the type name must be part of the
        // comparison.
        let mut ha = Header::new();
        ha.insert("x", TypedAttribute::<i32>::new(1_065_353_216))
            .expect("insert int");
        let mut hb = Header::new();
        hb.insert("x", TypedAttribute::<f32>::new(1.0))
            .expect("insert float");
        assert_ne!(
            ImageAttributes::from_header(&ha),
            ImageAttributes::from_header(&hb)
        );
    }
}
