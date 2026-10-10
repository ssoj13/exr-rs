//! Port of OpenEXR `ImfAttribute.h/.cpp` and the scalar/math typed attributes.
//!
//! See the module-level docs in [`crate::attr`] for how the C++ `Attribute` /
//! `TypedAttribute<T>` design maps onto the [`Attribute`] object-safe trait, the
//! [`AttrValue`] per-type trait, and the generic [`TypedAttribute<T>`].
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfAttribute.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfAttribute.cpp>

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;

use imath_rs::{
    Box2f, Box2i, M33d, M33f, M44d, M44f, V2d, V2f, V2i, V3d, V3f, V3i, Vec2, Vec3,
};

// Support types registered in the factory below.
use crate::attr::bytes_attr::Bytes;
use crate::attr::channel_list::ChannelList;
use crate::attr::chromaticities::Chromaticities;
use crate::attr::compression::Compression;
use crate::attr::deep_image_state::DeepImageState;
use crate::attr::envmap::Envmap;
use crate::attr::key_code::KeyCode;
use crate::attr::id_manifest::CompressedIDManifest;
use crate::attr::line_order::LineOrder;
use crate::attr::opaque::OpaqueAttribute;
use crate::attr::preview_image::PreviewImage;
use crate::attr::rational::Rational;
use crate::attr::tile_description::TileDescription;
use crate::attr::time_code::TimeCode;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ===========================================================================
// AttrValue — per-value-type trait (the content of every C++ specialisation)
// ===========================================================================

/// Trait implemented once per attribute value type `T`.
///
/// This is the Rust analogue of the C++ per-type specialisations of
/// `TypedAttribute<T>::staticTypeName()`, `writeValueTo()` and
/// `readValueFrom()`.  Implementing it for a type makes that type usable as the
/// payload of [`TypedAttribute<T>`].
///
/// # Byte-exactness
/// [`write_value`][AttrValue::write_value] / [`read_value`][AttrValue::read_value]
/// must produce / consume exactly the same bytes as the corresponding C++
/// `writeValueTo` / `readValueFrom`, using the shared [`xdr`] layer.
pub trait AttrValue: Clone + Send + Sync + 'static {
    /// The wire type-name string, e.g. `"int"`, `"v2f"`, `"m44f"`.
    ///
    /// Mirrors `TypedAttribute<T>::staticTypeName()`.
    fn static_type_name() -> &'static str;

    /// Serialise the value to `o` — mirrors `writeValueTo(os, version)`.
    ///
    /// `version` is the file version word; the scalar/math attributes ignore it
    /// (exactly as the C++ specialisations do), but it is threaded through to
    /// match the C++ signature.
    fn write_value(&self, o: &mut dyn OStream, version: i32) -> Result<()>;

    /// Deserialise a value from `i` — mirrors `readValueFrom(is, size, version)`.
    ///
    /// `size` is the attribute's declared byte length from the header (used by
    /// the variable-length attributes: string, stringvector, floatvector).
    fn read_value(i: &mut dyn IStream, size: i32, version: i32) -> Result<Self>;
}

// ===========================================================================
// Attribute — object-safe trait stored in the header map (C++ abstract base)
// ===========================================================================

/// Object-safe attribute trait — port of the abstract C++ `class Attribute`.
///
/// The EXR header stores `Box<dyn Attribute>` values.  Downcasting to a
/// concrete [`TypedAttribute<T>`] replaces the C++ `dynamic_cast` /
/// `TypedAttribute<T>::cast` via [`Any`].
///
/// `Send + Sync`: attribute payloads are plain owned data (numbers, strings,
/// vectors, imath value types) with no interior mutability or shared handles, so
/// `Box<dyn Attribute>` — and therefore [`crate::header::Header`], the chunk
/// `RawPart`, and the `Image` facade — are safely shareable across threads. This
/// lets downstream consumers hold them in thread-safe contexts (e.g. PyO3
/// pyclasses) without an `unsendable` opt-out.
pub trait Attribute: Any + Send + Sync {
    /// This attribute's wire type name — mirrors `Attribute::typeName()`.
    ///
    /// Returns a borrow tied to `&self` so that `OpaqueAttribute` can return
    /// its runtime [`crate::attr::Name`] field directly without leaking
    /// memory.  Typed attributes return `T::static_type_name().as_bytes()`
    /// (always ASCII).
    ///
    /// C-08c: raw bytes, not `&str` — C++ `Attribute::typeName()` returns a
    /// `const char*`/`std::string` with no encoding check anywhere
    /// (`parse_header.c`'s `read_text`), so an unregistered type whose name
    /// on disk is not valid UTF-8 still round-trips byte-exact through
    /// [`OpaqueAttribute`] instead of being rejected or mangled.
    fn type_name(&self) -> &[u8];

    /// Serialise the value — mirrors `Attribute::writeValueTo(os, version)`.
    fn write_value_to(&self, o: &mut dyn OStream, version: i32) -> Result<()>;

    /// Make a boxed clone — mirrors `Attribute::copy()`.
    fn clone_attr(&self) -> Box<dyn Attribute>;

    /// Upcast to [`Any`] for shared downcasting (replaces `dynamic_cast`).
    fn as_any(&self) -> &dyn Any;

    /// Upcast to [`Any`] for mutable downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ===========================================================================
// TypedAttribute<T> — the single generic wrapper (C++ class template)
// ===========================================================================

/// Generic attribute wrapper — port of the C++ `template<class T> TypedAttribute<T>`.
///
/// Holds one value of type `T: AttrValue`.  All typed attributes (`IntAttribute`,
/// `V2fAttribute`, `M44fAttribute`, …) are aliases for `TypedAttribute<T>` with a
/// concrete `T`, exactly as in C++.
#[derive(Clone, Debug)]
pub struct TypedAttribute<T: AttrValue> {
    /// The attribute's value (`TypedAttribute<T>::_value` in C++).
    pub value: T,
}

impl<T: AttrValue> TypedAttribute<T> {
    /// Construct from a value — mirrors `TypedAttribute<T>(const T&)`.
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Read a value from the stream into a fresh `TypedAttribute<T>`.
    ///
    /// Mirrors the C++ pattern of `newAttribute()` (default-construct) followed
    /// by `readValueFrom(is, size, version)`, collapsed into one step so that no
    /// `Default` bound is required on the imath value types.
    pub fn read_from(i: &mut dyn IStream, size: i32, version: i32) -> Result<Self> {
        Ok(Self {
            value: T::read_value(i, size, version)?,
        })
    }

    /// Static type name — mirrors `TypedAttribute<T>::staticTypeName()`.
    pub fn static_type_name() -> &'static str {
        T::static_type_name()
    }
}

impl<T: AttrValue> Attribute for TypedAttribute<T> {
    fn type_name(&self) -> &[u8] {
        T::static_type_name().as_bytes()
    }

    fn write_value_to(&self, o: &mut dyn OStream, version: i32) -> Result<()> {
        self.value.write_value(o, version)
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

// ===========================================================================
// AttrValue implementations — one per C++ value type (skip nothing)
// ===========================================================================

// ---------------------------------------------------------------------------
// IntAttribute / FloatAttribute / DoubleAttribute
//
// These use the *generic* `TypedAttribute<T>::writeValueTo`/`readValueFrom`
// (a single `Xdr::write`/`Xdr::read` of the scalar) — ImfIntAttribute.h etc.
// only `typedef TypedAttribute<int> IntAttribute;` and let the generic template
// body do the I/O.
// ---------------------------------------------------------------------------

impl AttrValue for i32 {
    fn static_type_name() -> &'static str {
        "int"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_i32(o, *self)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        xdr::read_i32(i)
    }
}

impl AttrValue for f32 {
    fn static_type_name() -> &'static str {
        "float"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f32(o, *self)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        xdr::read_f32(i)
    }
}

impl AttrValue for f64 {
    fn static_type_name() -> &'static str {
        "double"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f64(o, *self)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        xdr::read_f64(i)
    }
}

// ---------------------------------------------------------------------------
// ExrString — byte-opaque string value type
//
// C++ `std::string` is a byte container with no encoding guarantee.  EXR
// attribute strings are therefore byte-opaque: the C++ reader stores the raw
// bytes without any UTF-8 validation.  We mirror that with `Vec<u8>` wrapped
// in a newtype so callers can still ask "is this valid UTF-8?" via `as_str()`.
//
// Port reference: ImfStringAttribute.h / ImfStringAttribute.cpp (v3.3.11)
// ---------------------------------------------------------------------------

/// Byte-opaque EXR string value — mirrors C++ `std::string` attribute payload.
///
/// EXR attribute strings carry arbitrary bytes (no encoding guarantee).  This
/// type stores them as raw `Vec<u8>` so non-UTF-8 files round-trip faithfully.
/// Use [`as_str`][ExrString::as_str] to get a `&str` when the content is
/// known to be valid UTF-8, or [`to_string_lossy`][ExrString::to_string_lossy]
/// for display.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExrString(Vec<u8>);

impl ExrString {
    /// Construct from raw bytes — no encoding check performed.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// The raw byte content.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// `Some(&str)` iff the bytes are valid UTF-8; `None` otherwise.
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }

    /// Lossless display: valid UTF-8 passes through; invalid bytes are
    /// replaced with U+FFFD (same contract as `String::from_utf8_lossy`).
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.0)
    }

    /// Number of bytes in the string value.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True iff the byte sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&str> for ExrString {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }
}

impl From<String> for ExrString {
    fn from(s: String) -> Self {
        Self(s.into_bytes())
    }
}

impl From<Vec<u8>> for ExrString {
    fn from(b: Vec<u8>) -> Self {
        Self(b)
    }
}

// ---------------------------------------------------------------------------
// StringAttribute — ImfStringAttribute.cpp
//
// C++ writes the raw characters with NO length prefix and NO NUL terminator:
//   for (i = 0; i < size; i++) Xdr::write<StreamIO>(os, _value[i]);
// On read the byte count comes from the attribute `size` field:
//   _value.resize(size); for (i = 0; i < size; i++) Xdr::read(is, _value[i]);
// So this is *not* the NUL-terminated `Xdr::write_string` form — it is raw,
// size-delimited bytes.
// ---------------------------------------------------------------------------

impl AttrValue for ExrString {
    fn static_type_name() -> &'static str {
        "string"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // Raw bytes, no terminator — the surrounding attribute size delimits it.
        xdr::write_chars(o, &self.0)
    }
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        if size < 0 {
            return Err(ExrError::Input(
                "Negative size field reading string attribute".into(),
            ));
        }
        // Bounds-check the declared size against the stream before allocating.
        let buf = xdr::read_bounded_bytes(i, size as usize)?;
        // Store raw bytes without UTF-8 validation — mirrors C++ std::string.
        Ok(ExrString(buf))
    }
}

// ---------------------------------------------------------------------------
// StringVectorAttribute — ImfStringVectorAttribute.cpp
//
// Each element is length-prefixed: write `int strSize` then `strSize` raw chars.
// Read loops until `read == size`, validating `0 <= strSize <= size - read`.
// ---------------------------------------------------------------------------

impl AttrValue for Vec<ExrString> {
    fn static_type_name() -> &'static str {
        "stringvector"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for s in self {
            xdr::write_wire_len(o, s.len())?;
            xdr::write_chars(o, s.as_bytes())?;
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        let mut out: Vec<ExrString> = Vec::new();
        let mut read: i32 = 0;
        while read < size {
            let str_size = xdr::read_i32(i)?;
            // C++: read += Xdr::size<int>(); (== 4)
            read = read
                .checked_add(xdr::SIZE_I32 as i32)
                .ok_or_else(|| ExrError::Input("size overflow reading stringvector".into()))?;

            // Match the C++ bounds check exactly.
            if str_size < 0 || str_size > size - read {
                return Err(ExrError::Input(
                    "Invalid size field reading stringvector attribute".into(),
                ));
            }

            // Bounds-check each element size against the stream before allocating
            // (the C++ `strSize <= size - read` check above still applies too).
            let buf = xdr::read_bounded_bytes(i, str_size as usize)?;

            read = read
                .checked_add(str_size)
                .ok_or_else(|| ExrError::Input("size overflow reading stringvector".into()))?;

            // Store raw bytes without UTF-8 validation — mirrors C++ std::string.
            out.push(ExrString(buf));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FloatVectorAttribute — ImfFloatVectorAttribute.cpp
//
// Writes each float back-to-back (no count). On read, element count is derived
// from the attribute size: n = size / Xdr::size<float>().
// ---------------------------------------------------------------------------

impl AttrValue for Vec<f32> {
    fn static_type_name() -> &'static str {
        "floatvector"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for v in self {
            xdr::write_f32(o, *v)?;
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        if size < 0 {
            return Err(ExrError::Input(
                "Negative size field reading floatvector attribute".into(),
            ));
        }
        // C++ 3.4 core `check_bad_attrsz` rejects a floatvector whose
        // declared size is not a multiple of 4 (the whole attribute is then
        // dropped non-strictly), rather than silently reading `size/4` floats
        // and leaving the remainder for the header reader to reinterpret.
        if !(size as usize).is_multiple_of(xdr::SIZE_F32) {
            return Err(ExrError::Input(format!(
                "floatvector attribute size {size} is not a multiple of {}",
                xdr::SIZE_F32
            )));
        }
        let n = (size as usize) / xdr::SIZE_F32;
        // Fallible reservation: an attacker-declared `size` (bounded to the stream
        // remainder by `read_attributes_from` for size-reporting streams, but not
        // for a custom unknown-size `IStream`) must not abort the process on an
        // eager `with_capacity`.
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| {
            ExrError::Input(format!(
                "Failed to allocate {n} floats for a floatvector attribute."
            ))
        })?;
        for _ in 0..n {
            out.push(xdr::read_f32(i)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BoxAttribute — ImfBoxAttribute.cpp
//
// Box2i / Box2f write min.x, min.y, max.x, max.y in order.
// ---------------------------------------------------------------------------

impl AttrValue for Box2i {
    fn static_type_name() -> &'static str {
        "box2i"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_i32(o, self.min.x)?;
        xdr::write_i32(o, self.min.y)?;
        xdr::write_i32(o, self.max.x)?;
        xdr::write_i32(o, self.max.y)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let min_x = xdr::read_i32(i)?;
        let min_y = xdr::read_i32(i)?;
        let max_x = xdr::read_i32(i)?;
        let max_y = xdr::read_i32(i)?;
        Ok(Box2i {
            min: Vec2 { x: min_x, y: min_y },
            max: Vec2 { x: max_x, y: max_y },
        })
    }
}

impl AttrValue for Box2f {
    fn static_type_name() -> &'static str {
        "box2f"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f32(o, self.min.x)?;
        xdr::write_f32(o, self.min.y)?;
        xdr::write_f32(o, self.max.x)?;
        xdr::write_f32(o, self.max.y)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let min_x = xdr::read_f32(i)?;
        let min_y = xdr::read_f32(i)?;
        let max_x = xdr::read_f32(i)?;
        let max_y = xdr::read_f32(i)?;
        Ok(Box2f {
            min: Vec2 { x: min_x, y: min_y },
            max: Vec2 { x: max_x, y: max_y },
        })
    }
}

// ---------------------------------------------------------------------------
// VecAttribute — ImfVecAttribute.cpp
//
// V2* write x, y; V3* write x, y, z.
// ---------------------------------------------------------------------------

impl AttrValue for V2i {
    fn static_type_name() -> &'static str {
        "v2i"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_i32(o, self.x)?;
        xdr::write_i32(o, self.y)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_i32(i)?;
        let y = xdr::read_i32(i)?;
        Ok(Vec2 { x, y })
    }
}

impl AttrValue for V2f {
    fn static_type_name() -> &'static str {
        "v2f"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f32(o, self.x)?;
        xdr::write_f32(o, self.y)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_f32(i)?;
        let y = xdr::read_f32(i)?;
        Ok(Vec2 { x, y })
    }
}

impl AttrValue for V2d {
    fn static_type_name() -> &'static str {
        "v2d"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f64(o, self.x)?;
        xdr::write_f64(o, self.y)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_f64(i)?;
        let y = xdr::read_f64(i)?;
        Ok(Vec2 { x, y })
    }
}

impl AttrValue for V3i {
    fn static_type_name() -> &'static str {
        "v3i"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_i32(o, self.x)?;
        xdr::write_i32(o, self.y)?;
        xdr::write_i32(o, self.z)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_i32(i)?;
        let y = xdr::read_i32(i)?;
        let z = xdr::read_i32(i)?;
        Ok(Vec3 { x, y, z })
    }
}

impl AttrValue for V3f {
    fn static_type_name() -> &'static str {
        "v3f"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f32(o, self.x)?;
        xdr::write_f32(o, self.y)?;
        xdr::write_f32(o, self.z)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_f32(i)?;
        let y = xdr::read_f32(i)?;
        let z = xdr::read_f32(i)?;
        Ok(Vec3 { x, y, z })
    }
}

impl AttrValue for V3d {
    fn static_type_name() -> &'static str {
        "v3d"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_f64(o, self.x)?;
        xdr::write_f64(o, self.y)?;
        xdr::write_f64(o, self.z)
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let x = xdr::read_f64(i)?;
        let y = xdr::read_f64(i)?;
        let z = xdr::read_f64(i)?;
        Ok(Vec3 { x, y, z })
    }
}

// ---------------------------------------------------------------------------
// MatrixAttribute — ImfMatrixAttribute.cpp
//
// Row-major: write _value[row][col] for all rows then cols (M33 = 9, M44 = 16).
// imath-rs stores the elements as `Matrix33::x: [[T;3];3]` / `Matrix44::x: [[T;4];4]`
// in the same row-major layout, so `self.x[r][c]` maps directly to `_value[r][c]`.
// ---------------------------------------------------------------------------

impl AttrValue for M33f {
    fn static_type_name() -> &'static str {
        "m33f"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for row in &self.x {
            for v in row {
                xdr::write_f32(o, *v)?;
            }
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let mut m = [[0.0f32; 3]; 3];
        for row in &mut m {
            for v in row {
                *v = xdr::read_f32(i)?;
            }
        }
        Ok(M33f { x: m })
    }
}

impl AttrValue for M33d {
    fn static_type_name() -> &'static str {
        "m33d"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for row in &self.x {
            for v in row {
                xdr::write_f64(o, *v)?;
            }
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let mut m = [[0.0f64; 3]; 3];
        for row in &mut m {
            for v in row {
                *v = xdr::read_f64(i)?;
            }
        }
        Ok(M33d { x: m })
    }
}

impl AttrValue for M44f {
    fn static_type_name() -> &'static str {
        "m44f"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for row in &self.x {
            for v in row {
                xdr::write_f32(o, *v)?;
            }
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let mut m = [[0.0f32; 4]; 4];
        for row in &mut m {
            for v in row {
                *v = xdr::read_f32(i)?;
            }
        }
        Ok(M44f { x: m })
    }
}

impl AttrValue for M44d {
    fn static_type_name() -> &'static str {
        "m44d"
    }
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for row in &self.x {
            for v in row {
                xdr::write_f64(o, *v)?;
            }
        }
        Ok(())
    }
    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        let mut m = [[0.0f64; 4]; 4];
        for row in &mut m {
            for v in row {
                *v = xdr::read_f64(i)?;
            }
        }
        Ok(M44d { x: m })
    }
}

// ===========================================================================
// Type aliases — mirror the C++ `typedef TypedAttribute<T> XxxAttribute;`
// ===========================================================================

/// `typedef TypedAttribute<int> IntAttribute;`
pub type IntAttribute = TypedAttribute<i32>;
/// `typedef TypedAttribute<float> FloatAttribute;`
pub type FloatAttribute = TypedAttribute<f32>;
/// `typedef TypedAttribute<double> DoubleAttribute;`
pub type DoubleAttribute = TypedAttribute<f64>;
/// `typedef TypedAttribute<std::string> StringAttribute;`
pub type StringAttribute = TypedAttribute<ExrString>;
/// `typedef TypedAttribute<StringVector> StringVectorAttribute;`
pub type StringVectorAttribute = TypedAttribute<Vec<ExrString>>;
/// `typedef TypedAttribute<FloatVector> FloatVectorAttribute;`
pub type FloatVectorAttribute = TypedAttribute<Vec<f32>>;
/// `using Box2iAttribute = TypedAttribute<Imath::Box2i>;`
pub type Box2iAttribute = TypedAttribute<Box2i>;
/// `using Box2fAttribute = TypedAttribute<Imath::Box2f>;`
pub type Box2fAttribute = TypedAttribute<Box2f>;
/// `typedef TypedAttribute<Imath::V2i> V2iAttribute;`
pub type V2iAttribute = TypedAttribute<V2i>;
/// `typedef TypedAttribute<Imath::V2f> V2fAttribute;`
pub type V2fAttribute = TypedAttribute<V2f>;
/// `typedef TypedAttribute<Imath::V2d> V2dAttribute;`
pub type V2dAttribute = TypedAttribute<V2d>;
/// `typedef TypedAttribute<Imath::V3i> V3iAttribute;`
pub type V3iAttribute = TypedAttribute<V3i>;
/// `typedef TypedAttribute<Imath::V3f> V3fAttribute;`
pub type V3fAttribute = TypedAttribute<V3f>;
/// `typedef TypedAttribute<Imath::V3d> V3dAttribute;`
pub type V3dAttribute = TypedAttribute<V3d>;
/// `typedef TypedAttribute<Imath::M33f> M33fAttribute;`
pub type M33fAttribute = TypedAttribute<M33f>;
/// `typedef TypedAttribute<Imath::M33d> M33dAttribute;`
pub type M33dAttribute = TypedAttribute<M33d>;
/// `typedef TypedAttribute<Imath::M44f> M44fAttribute;`
pub type M44fAttribute = TypedAttribute<M44f>;
/// `typedef TypedAttribute<Imath::M44d> M44dAttribute;`
pub type M44dAttribute = TypedAttribute<M44d>;

// ===========================================================================
// Registry — port of Attribute::newAttribute / registerAttributeType
// ===========================================================================

/// A factory that reads a fully-formed attribute of one registered type.
///
/// This collapses the C++ two-step of `newAttribute()` (default-construct) plus
/// `readValueFrom(is, size, version)` into a single reader, mirroring the only
/// way the factory is actually used by the header reader.  The C++ map value is
/// `Attribute* (*)()`; ours additionally performs the read so that no `Default`
/// bound is needed on the imath value types.
pub type AttributeReader = fn(&mut dyn IStream, i32, i32) -> Result<Box<dyn Attribute>>;

/// Generic reader registered for each `T: AttrValue`.
///
/// Equivalent to `TypedAttribute<T>::makeNewAttribute()` followed by a
/// `readValueFrom` on the resulting object.
fn read_typed<T: AttrValue>(
    i: &mut dyn IStream,
    size: i32,
    version: i32,
) -> Result<Box<dyn Attribute>> {
    Ok(Box::new(TypedAttribute::<T>::read_from(i, size, version)?))
}

/// The global type-name → reader map (port of the C++ `typeMap()` singleton).
///
/// Built once on first use with every built-in attribute type registered,
/// mirroring the `registerAttributeType()` calls IlmImf performs at startup.
///
/// A `RwLock` (not the immutable map this used to be) because this crate ports
/// `Attribute::registerAttributeType`/`unRegisterAttributeType`: C++ guards
/// its map with a mutex specifically so a host application can add its own
/// attribute types at runtime (`ImfAttribute.cpp:63-119`), and this port now
/// supports the same via [`register_attribute_type`]/[`unregister_attribute_type`].
fn type_map() -> &'static std::sync::RwLock<HashMap<String, AttributeReader>> {
    static MAP: OnceLock<std::sync::RwLock<HashMap<String, AttributeReader>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: HashMap<String, AttributeReader> = HashMap::new();
        // One entry per value type, keyed by its exact C++ staticTypeName().
        m.insert(<i32 as AttrValue>::static_type_name().to_string(), read_typed::<i32>);
        m.insert(<f32 as AttrValue>::static_type_name().to_string(), read_typed::<f32>);
        m.insert(<f64 as AttrValue>::static_type_name().to_string(), read_typed::<f64>);
        m.insert(<ExrString as AttrValue>::static_type_name().to_string(), read_typed::<ExrString>);
        m.insert(
            <Vec<ExrString> as AttrValue>::static_type_name().to_string(),
            read_typed::<Vec<ExrString>>,
        );
        m.insert(
            <Vec<f32> as AttrValue>::static_type_name().to_string(),
            read_typed::<Vec<f32>>,
        );
        m.insert(<Box2i as AttrValue>::static_type_name().to_string(), read_typed::<Box2i>);
        m.insert(<Box2f as AttrValue>::static_type_name().to_string(), read_typed::<Box2f>);
        m.insert(<V2i as AttrValue>::static_type_name().to_string(), read_typed::<V2i>);
        m.insert(<V2f as AttrValue>::static_type_name().to_string(), read_typed::<V2f>);
        m.insert(<V2d as AttrValue>::static_type_name().to_string(), read_typed::<V2d>);
        m.insert(<V3i as AttrValue>::static_type_name().to_string(), read_typed::<V3i>);
        m.insert(<V3f as AttrValue>::static_type_name().to_string(), read_typed::<V3f>);
        m.insert(<V3d as AttrValue>::static_type_name().to_string(), read_typed::<V3d>);
        m.insert(<M33f as AttrValue>::static_type_name().to_string(), read_typed::<M33f>);
        m.insert(<M33d as AttrValue>::static_type_name().to_string(), read_typed::<M33d>);
        m.insert(<M44f as AttrValue>::static_type_name().to_string(), read_typed::<M44f>);
        m.insert(<M44d as AttrValue>::static_type_name().to_string(), read_typed::<M44d>);
        // Support types
        m.insert(<Compression as AttrValue>::static_type_name().to_string(),    read_typed::<Compression>);
        m.insert(<LineOrder as AttrValue>::static_type_name().to_string(),       read_typed::<LineOrder>);
        m.insert(<Envmap as AttrValue>::static_type_name().to_string(),          read_typed::<Envmap>);
        m.insert(<DeepImageState as AttrValue>::static_type_name().to_string(),  read_typed::<DeepImageState>);
        m.insert(<TileDescription as AttrValue>::static_type_name().to_string(), read_typed::<TileDescription>);
        m.insert(<Chromaticities as AttrValue>::static_type_name().to_string(),  read_typed::<Chromaticities>);
        m.insert(<KeyCode as AttrValue>::static_type_name().to_string(),         read_typed::<KeyCode>);
        m.insert(<TimeCode as AttrValue>::static_type_name().to_string(),        read_typed::<TimeCode>);
        m.insert(<Rational as AttrValue>::static_type_name().to_string(),        read_typed::<Rational>);
        // Channel list, preview, opaque and ID-manifest types
        m.insert(<ChannelList as AttrValue>::static_type_name().to_string(),     read_typed::<ChannelList>);
        m.insert(<PreviewImage as AttrValue>::static_type_name().to_string(),    read_typed::<PreviewImage>);
        m.insert(<CompressedIDManifest as AttrValue>::static_type_name().to_string(), read_typed::<CompressedIDManifest>);
        // OpenEXR 3.4 addition.
        m.insert(<Bytes as AttrValue>::static_type_name().to_string(), read_typed::<Bytes>);
        std::sync::RwLock::new(m)
    })
}

/// Whether a given attribute type name is registered — port of `Attribute::knownType`.
pub fn known_type(type_name: &str) -> bool {
    read_type_map().contains_key(type_name)
}

/// Register a new attribute type at runtime — port of
/// `Attribute::registerAttributeType`.
///
/// Lets a host application extend the registry with its own attribute types,
/// exactly as C++ does (`ImfAttribute.cpp:72-90`). `reader` collapses C++'s
/// `newAttribute()` (default-construct) + `readValueFrom(is, size, version)`
/// into one factory, matching every built-in entry's shape — see
/// [`AttributeReader`].
///
/// Returns [`ExrError::Arg`] if `type_name` is already registered (built-in
/// or previously registered), matching C++'s `THROW(ArgExc, …)`.
pub fn register_attribute_type(type_name: &str, reader: AttributeReader) -> Result<()> {
    let mut map = write_type_map();
    if map.contains_key(type_name) {
        return Err(ExrError::Arg(format!(
            "Cannot register image file attribute type \"{type_name}\". \
             The type has already been registered."
        )));
    }
    map.insert(type_name.to_string(), reader);
    Ok(())
}

/// Remove a previously-[`register_attribute_type`]d entry — port of
/// `Attribute::unRegisterAttributeType`. A no-op if `type_name` is not
/// registered (matching C++'s `tMap.erase`, which is likewise a no-op on a
/// missing key). Removing a BUILT-IN type is possible (as in C++) — a
/// subsequent read of that type falls back to [`OpaqueAttribute`], same as
/// any other unknown type.
pub fn unregister_attribute_type(type_name: &str) {
    write_type_map().remove(type_name);
}

/// Create (read) an attribute of the named type from a stream.
///
/// Port of the EXR header reader's behaviour (`ImfHeader.cpp readFrom`): if the
/// type is registered (`Attribute::knownType`), it is read via the registered
/// factory + `readValueFrom`; otherwise the value is preserved as an
/// [`OpaqueAttribute`] so unknown attributes round-trip losslessly.  This is
/// why this never errors on an unknown type — unlike the lower-level
/// `Attribute::newAttribute`, which throws `ArgExc`; the header reader catches
/// that and substitutes an opaque attribute, and we collapse the two paths.
///
/// C-08c: `type_name` is raw bytes — every registered type name is ASCII
/// (registration itself takes `&str`), so a non-UTF-8 type name can never
/// match the registry and always falls through to [`OpaqueAttribute`],
/// exactly as C++'s `strcmp`-based `typeMap` lookup would (no entry's key
/// contains a byte a non-UTF-8 name could accidentally collide with).
pub fn new_attribute(
    type_name: &[u8],
    i: &mut dyn IStream,
    size: i32,
    version: i32,
) -> Result<Box<dyn Attribute>> {
    // Look up and copy the `fn` pointer out, then drop the read lock before
    // calling it — `reader` may itself be a caller-registered closure-backed
    // `fn` that (however unusually) touches the registry, and holding the
    // lock across the call would deadlock that case for no benefit here.
    let reader = std::str::from_utf8(type_name)
        .ok()
        .and_then(|s| read_type_map().get(s).copied());
    match reader {
        Some(reader) => reader(i, size, version),
        None => Ok(Box::new(OpaqueAttribute::read_from(
            type_name, i, size, version,
        )?)),
    }
}

/// Read-lock the type registry, recovering from a poisoned lock (a panic
/// while holding the lock elsewhere) rather than propagating the poison —
/// the map itself is always left in a consistent state by every writer here
/// (a single `insert`/`remove` call), so a stale panic on another thread is
/// never a reason to fail every future header read.
fn read_type_map() -> std::sync::RwLockReadGuard<'static, HashMap<String, AttributeReader>> {
    type_map()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write-lock counterpart of [`read_type_map`].
fn write_type_map() -> std::sync::RwLockWriteGuard<'static, HashMap<String, AttributeReader>> {
    type_map()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    /// A custom type can be registered at runtime, is then read via
    /// the registered factory (not `OpaqueAttribute`), and a second
    /// registration of the same name is rejected. `unregister_attribute_type`
    /// then reverts it to the unknown/opaque path.
    #[test]
    fn register_and_unregister_custom_attribute_type() {
        const TYPE_NAME: &str = "exrRsTestCustomType_registerAndUnregister";
        assert!(!known_type(TYPE_NAME));

        register_attribute_type(TYPE_NAME, read_typed::<i32>).unwrap();
        assert!(known_type(TYPE_NAME));

        // Registering the same name again must fail — matches C++'s "already
        // registered" `ArgExc`.
        assert!(register_attribute_type(TYPE_NAME, read_typed::<i32>).is_err());

        // A value of the newly-registered type is read via its factory, not
        // preserved as `OpaqueAttribute`.
        let mut o = MemOStream::new();
        42i32.write_value(&mut o, 0).unwrap();
        let bytes = o.into_data();
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let attr = new_attribute(TYPE_NAME.as_bytes(), &mut i, size, 0).unwrap();
        assert!(attr.as_any().downcast_ref::<TypedAttribute<i32>>().is_some());

        unregister_attribute_type(TYPE_NAME);
        assert!(!known_type(TYPE_NAME));

        // After unregistering, the same bytes are read as an opaque attribute.
        let mut i2 = MemIStream::new(vec![42, 0, 0, 0]);
        let attr2 = new_attribute(TYPE_NAME.as_bytes(), &mut i2, 4, 0).unwrap();
        assert_eq!(attr2.type_name(), TYPE_NAME.as_bytes());
        assert!(attr2.as_any().downcast_ref::<OpaqueAttribute>().is_some());
    }

    /// `unregister_attribute_type` on a name that was never
    /// registered is a harmless no-op, matching C++'s `tMap.erase`.
    #[test]
    fn unregister_unknown_type_is_a_no_op() {
        unregister_attribute_type("exrRsTestCustomType_neverRegistered");
    }

    /// Serialise a value through its [`AttrValue::write_value`] and return the bytes.
    fn write_bytes<T: AttrValue>(v: &T) -> Vec<u8> {
        let mut o = MemOStream::new();
        v.write_value(&mut o, 0).expect("write");
        o.into_data()
    }

    /// Round-trip a value through write/read and assert it survives.
    fn round_trip<T: AttrValue + PartialEq + std::fmt::Debug>(v: T) {
        let bytes = write_bytes(&v);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = T::read_value(&mut i, size, 0).expect("read");
        assert_eq!(v, back);
    }

    // -- exact-byte checks ------------------------------------------------

    #[test]
    fn int_exact_bytes() {
        assert_eq!(write_bytes(&1i32), vec![0x01, 0x00, 0x00, 0x00]);
        assert_eq!(<i32 as AttrValue>::static_type_name(), "int");
    }

    #[test]
    fn float_exact_bytes() {
        // 1.0f32 -> 0x3F800000 LE
        assert_eq!(write_bytes(&1.0f32), vec![0x00, 0x00, 0x80, 0x3F]);
        assert_eq!(<f32 as AttrValue>::static_type_name(), "float");
    }

    #[test]
    fn double_exact_bytes() {
        // 1.0f64 -> 0x3FF0000000000000 LE
        assert_eq!(
            write_bytes(&1.0f64),
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F]
        );
        assert_eq!(<f64 as AttrValue>::static_type_name(), "double");
    }

    #[test]
    fn string_exact_bytes_no_terminator() {
        // Raw bytes, no NUL terminator.
        assert_eq!(write_bytes(&ExrString::from("abc")), b"abc".to_vec());
        assert_eq!(<ExrString as AttrValue>::static_type_name(), "string");
    }

    #[test]
    fn v2i_exact_bytes() {
        let v = V2i { x: 3, y: 7 };
        assert_eq!(
            write_bytes(&v),
            vec![0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00]
        );
        assert_eq!(<V2i as AttrValue>::static_type_name(), "v2i");
    }

    #[test]
    fn box2i_exact_bytes() {
        let b = Box2i {
            min: Vec2 { x: 1, y: 2 },
            max: Vec2 { x: 3, y: 4 },
        };
        assert_eq!(
            write_bytes(&b),
            vec![
                0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
                0x00, 0x00,
            ]
        );
        assert_eq!(<Box2i as AttrValue>::static_type_name(), "box2i");
    }

    #[test]
    fn stringvector_exact_bytes() {
        // ["ab", "c"] -> [02 00 00 00 'a' 'b'][01 00 00 00 'c']
        let v = vec![ExrString::from("ab"), ExrString::from("c")];
        assert_eq!(
            write_bytes(&v),
            vec![
                0x02, 0x00, 0x00, 0x00, b'a', b'b', 0x01, 0x00, 0x00, 0x00, b'c',
            ]
        );
        assert_eq!(<Vec<ExrString> as AttrValue>::static_type_name(), "stringvector");
    }

    #[test]
    fn floatvector_exact_bytes() {
        // [1.0, 0.0] -> 00 00 80 3F 00 00 00 00
        let v = vec![1.0f32, 0.0f32];
        assert_eq!(
            write_bytes(&v),
            vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(<Vec<f32> as AttrValue>::static_type_name(), "floatvector");
    }

    #[test]
    fn matrix_type_names() {
        assert_eq!(<M33f as AttrValue>::static_type_name(), "m33f");
        assert_eq!(<M33d as AttrValue>::static_type_name(), "m33d");
        assert_eq!(<M44f as AttrValue>::static_type_name(), "m44f");
        assert_eq!(<M44d as AttrValue>::static_type_name(), "m44d");
    }

    #[test]
    fn vec_type_names() {
        assert_eq!(<V2f as AttrValue>::static_type_name(), "v2f");
        assert_eq!(<V2d as AttrValue>::static_type_name(), "v2d");
        assert_eq!(<V3i as AttrValue>::static_type_name(), "v3i");
        assert_eq!(<V3f as AttrValue>::static_type_name(), "v3f");
        assert_eq!(<V3d as AttrValue>::static_type_name(), "v3d");
        assert_eq!(<Box2f as AttrValue>::static_type_name(), "box2f");
    }

    // -- round trips for every value type --------------------------------

    #[test]
    fn round_trips_scalars() {
        round_trip(-123_456i32);
        round_trip(core::f32::consts::PI);
        round_trip(core::f64::consts::E);
    }

    #[test]
    fn round_trips_strings() {
        round_trip(ExrString::from("hello world"));
        round_trip(ExrString::default());
        round_trip(vec![
            ExrString::from("one"),
            ExrString::from("two"),
            ExrString::default(),
        ]);
        round_trip(Vec::<ExrString>::new());
    }

    /// Non-UTF-8 bytes must survive a StringAttribute write→read cycle byte-identically.
    /// This is the bug being fixed: previously `read_value` would return an error for
    /// bytes such as `[0xFF, 0xFE, 0x00, 0x80]` which are valid EXR but not valid UTF-8.
    #[test]
    fn string_non_utf8_round_trips() {
        let raw: Vec<u8> = vec![0xFF, 0xFE, 0x00, 0x80];
        let s = ExrString::from_bytes(raw.clone());

        // Byte-exact serialisation check.
        let written = write_bytes(&s);
        assert_eq!(written, raw, "written bytes must equal raw input");

        // Full round-trip through read_value.
        let size = written.len() as i32;
        let mut stream = crate::io::std_io::MemIStream::new(written);
        let back = ExrString::read_value(&mut stream, size, 0).expect("read must succeed");
        assert_eq!(back.as_bytes(), raw.as_slice(), "round-trip bytes must match");

        // as_str() must return None for non-UTF-8.
        assert!(back.as_str().is_none(), "as_str() must be None for non-UTF-8");
        // as_str() must return Some for valid UTF-8.
        assert_eq!(ExrString::from("hello").as_str(), Some("hello"));
    }

    /// Non-UTF-8 elements in a stringvector must also survive a round-trip.
    #[test]
    fn stringvector_non_utf8_round_trips() {
        let elem_bytes: Vec<u8> = vec![0xC0, 0x80, 0xFF];
        let v = vec![
            ExrString::from("valid"),
            ExrString::from_bytes(elem_bytes.clone()),
        ];
        let written = write_bytes(&v);
        let size = written.len() as i32;
        let mut stream = crate::io::std_io::MemIStream::new(written);
        let back = Vec::<ExrString>::read_value(&mut stream, size, 0)
            .expect("stringvector read must succeed for non-UTF-8 elements");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].as_str(), Some("valid"));
        assert_eq!(back[1].as_bytes(), elem_bytes.as_slice());
        assert!(back[1].as_str().is_none());
    }

    #[test]
    fn round_trips_floatvector() {
        round_trip(vec![1.0f32, -2.5, 3.25, 0.0]);
        round_trip(Vec::<f32>::new());
    }

    /// A floatvector whose declared size is not a multiple of 4 is
    /// rejected outright, matching C++ 3.4 core `check_bad_attrsz`.
    #[test]
    fn floatvector_rejects_non_multiple_of_4_size() {
        let mut stream = crate::io::std_io::MemIStream::new(vec![0u8; 6]);
        assert!(Vec::<f32>::read_value(&mut stream, 6, 0).is_err());
    }

    #[test]
    fn round_trips_box() {
        let bi = Box2i {
            min: Vec2 { x: -1, y: -2 },
            max: Vec2 { x: 9, y: 8 },
        };
        round_trip(bi);
        let bf = Box2f {
            min: Vec2 { x: -1.5, y: -2.5 },
            max: Vec2 { x: 9.5, y: 8.5 },
        };
        round_trip(bf);
    }

    #[test]
    fn round_trips_vec() {
        round_trip(V2i { x: 3, y: 7 });
        round_trip(V2f { x: 1.5, y: -2.5 });
        round_trip(V2d { x: 1.25, y: -2.75 });
        round_trip(V3i { x: 1, y: 2, z: 3 });
        round_trip(V3f { x: 1.0, y: 2.0, z: 3.0 });
        round_trip(V3d {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        });
    }

    #[test]
    fn round_trips_matrix() {
        round_trip(M33f {
            x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        });
        round_trip(M33d {
            x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        });
        round_trip(M44f {
            x: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        });
        round_trip(M44d {
            x: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        });
    }

    // -- Attribute trait object / downcast -------------------------------

    #[test]
    fn typed_attribute_trait_object() {
        let attr = IntAttribute::new(42);
        let boxed: Box<dyn Attribute> = Box::new(attr);
        assert_eq!(boxed.type_name(), b"int");

        // Serialise via the object-safe trait.
        let mut o = MemOStream::new();
        boxed.write_value_to(&mut o, 0).unwrap();
        assert_eq!(o.into_data(), vec![0x2A, 0x00, 0x00, 0x00]);

        // Clone via the object-safe trait, then downcast back to the concrete type.
        let cloned = boxed.clone_attr();
        let concrete = cloned
            .as_any()
            .downcast_ref::<IntAttribute>()
            .expect("downcast");
        assert_eq!(concrete.value, 42);
    }

    // -- registry --------------------------------------------------------

    #[test]
    fn registry_known_types() {
        assert!(known_type("int"));
        assert!(known_type("v2f"));
        assert!(known_type("m44f"));
        assert!(known_type("stringvector"));
        assert!(!known_type("nope"));
    }

    #[test]
    fn registry_reads_int() {
        let bytes = vec![0x07, 0x00, 0x00, 0x00];
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let attr = new_attribute(b"int", &mut i, size, 0).unwrap();
        assert_eq!(attr.type_name(), b"int");
        let v = attr.as_any().downcast_ref::<IntAttribute>().unwrap();
        assert_eq!(v.value, 7);
    }

    #[test]
    fn registry_reads_v2f() {
        // V2f(1.0, 2.0)
        let mut o = MemOStream::new();
        V2f { x: 1.0, y: 2.0 }.write_value(&mut o, 0).unwrap();
        let bytes = o.into_data();
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let attr = new_attribute(b"v2f", &mut i, size, 0).unwrap();
        assert_eq!(attr.type_name(), b"v2f");
        let v = attr.as_any().downcast_ref::<V2fAttribute>().unwrap();
        assert_eq!(v.value.x, 1.0);
        assert_eq!(v.value.y, 2.0);
    }

    #[test]
    fn registry_reads_m44f() {
        let m = M44f {
            x: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let mut o = MemOStream::new();
        m.write_value(&mut o, 0).unwrap();
        let bytes = o.into_data();
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let attr = new_attribute(b"m44f", &mut i, size, 0).unwrap();
        assert_eq!(attr.type_name(), b"m44f");
        let v = attr.as_any().downcast_ref::<M44fAttribute>().unwrap();
        assert_eq!(v.value.x, m.x);
    }

    #[test]
    fn registry_unknown_type_returns_opaque() {
        // Unknown types are preserved as OpaqueAttribute (lossless round-trip),
        // matching the EXR header reader rather than the throwing low-level
        // Attribute::newAttribute.
        use crate::attr::opaque::OpaqueAttribute;
        let raw = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let size = raw.len() as i32;
        let mut i = MemIStream::new(raw.clone());
        let attr = new_attribute(b"bogus", &mut i, size, 0).unwrap();
        assert_eq!(attr.type_name(), b"bogus");
        let opaque = attr
            .as_any()
            .downcast_ref::<OpaqueAttribute>()
            .expect("opaque");
        assert_eq!(opaque.data(), raw.as_slice());
    }

    // A tiny buffer declaring a ~2 GiB string attribute must error
    // (bounded by the stream's remaining bytes) rather than allocate/abort.
    #[test]
    fn string_huge_size_rejected_without_alloc() {
        let mut i = MemIStream::new(vec![0u8; 4]);
        let err = ExrString::read_value(&mut i, i32::MAX, 0)
            .expect_err("huge declared size must be rejected");
        assert!(matches!(err, ExrError::Input(_)));
    }

    // Same guard via the header registry path for an unknown (opaque) type.
    #[test]
    fn opaque_huge_size_rejected_without_alloc() {
        let mut i = MemIStream::new(vec![0u8; 4]);
        // `Box<dyn Attribute>` is not Debug, so match on the result directly.
        let r = new_attribute(b"bogus", &mut i, i32::MAX, 0);
        assert!(matches!(r, Err(ExrError::Input(_))));
    }

    // A well-formed small string still round-trips unchanged.
    #[test]
    fn string_small_still_round_trips() {
        let raw = b"hello".to_vec();
        let mut i = MemIStream::new(raw.clone());
        let s = ExrString::read_value(&mut i, raw.len() as i32, 0).expect("read");
        assert_eq!(s.0, raw);
    }
}
