//! Port of OpenEXR `ImfHeader.h` / `ImfHeader.cpp` — the image file header.
//!
//! A [`Header`] is an ordered map from attribute name to a boxed
//! [`Attribute`].  C++ uses `std::map<Name, Attribute*>`, which keeps the
//! attributes sorted by name and iterates in that sorted order; we mirror that
//! with a [`BTreeMap<String, Box<dyn Attribute>>`] so that the on-disk
//! attribute order — and therefore the entire serialised header — is
//! byte-identical to IlmImf.
//!
//! The default constructor seeds the eight *required* attributes
//! (`displayWindow`, `dataWindow`, `pixelAspectRatio`, `screenWindowCenter`,
//! `screenWindowWidth`, `lineOrder`, `compression`, `channels`) exactly as the
//! C++ `initialize()` helper does.  [`write_to`][Header::write_to] /
//! [`read_from`][Header::read_from] reproduce `Header::writeTo` /
//! `Header::readFrom` plus the version-word computation that C++ performs in
//! `GenericOutputFile::writeMagicNumberAndVersionField`.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfHeader.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfHeader.h>
//! version word: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfGenericOutputFile.cpp>
//! `usesLongNames`: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMisc.cpp>

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::btree_map::{Iter, IterMut};

use imath_rs::{Box2i, V2f, Vec2};

use crate::attr::{
    AttrValue, Attribute, Box2iAttribute, ChannelList, ChannelListAttribute, Compression,
    CompressionAttribute, ExrString, FloatAttribute, IntAttribute, LineOrder, LineOrderAttribute,
    Name, PreviewImage, PreviewImageAttribute, StringAttribute, TileDescription, TileDescriptionAttribute,
    TypedAttribute, V2fAttribute, new_attribute,
};
use crate::container::version;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};
use crate::io::std_io::MemOStream;

/// Maximum attribute / type / channel name length supported when *not* using
/// the long-names flag.  Mirrors the threshold in C++ `usesLongNames` — a name
/// of 32 or more bytes forces [`LONG_NAMES_FLAG`][version::LONG_NAMES_FLAG].
///
/// (C++ `Name::MAX_LENGTH` is 255; this constant is the *short-name* limit, the
/// `>= 32` test, not the buffer size.)
const SHORT_NAME_LIMIT: usize = 31;

/// `Name::MAX_LENGTH` — the largest name C++ will read before erroring
/// (`Name::SIZE - 1`, i.e. 255).  Used as the read-side cap for attribute and
/// type names regardless of the long-names flag, matching `Header::readFrom`
/// which always passes `Name::MAX_LENGTH` to the string reader.
const NAME_MAX_LENGTH: usize = 255;

/// `INT_MAX / 2` — the data/display window coordinate bound used by
/// `sanityCheck` to keep `max - min + 1` and `max + min` from overflowing.
const INT_MAX_DIV_2: i32 = i32::MAX / 2;

// The three `sanity_check` messages that
// `sanity_check_at_open` keeps as `ExrError::Arg` (a required attribute
// missing, not defaultable) instead of reclassifying to `ExrError::Input`.
const MISSING_NAME_MSG: &str = "Headers in a multipart file should have name attribute.";
const MISSING_TYPE_MSG: &str = "Headers in a multipart file should have type attribute.";
const MISSING_TILE_DESC_MSG: &str = "Tiled image has no tile description attribute.";

/// Process-global max-image/max-tile size state — see
/// [`Header::set_max_image_size`] / [`Header::set_max_tile_size`]. `0` means
/// unlimited, matching the C++ default (`exr_set_default_maximum_image_size`
/// et al.).
static MAX_IMAGE_WIDTH: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static MAX_IMAGE_HEIGHT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static MAX_TILE_WIDTH: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static MAX_TILE_HEIGHT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Process-global default ZIP compression level — mirrors C++
/// `exr_{set,get}_default_zip_compression_level` (`OpenEXRCore/base.c`).
/// `-1` (the C++ default) means "let the ZIP codec pick its own default"
/// (`EXR_DEFAULT_ZLIB_COMPRESS_LEVEL`, 4 — see `compressor::zip`). Stored as
/// the raw `i32` (negative values round-trip through `AtomicI32` directly).
static DEFAULT_ZIP_LEVEL: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Process-global default DWA compression quality — mirrors C++
/// `exr_{set,get}_default_dwa_compression_quality`. C++ default: `45.0`.
/// Stored as the `f32` bit pattern since `std` has no `AtomicF32`.
static DEFAULT_DWA_LEVEL_BITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(45.0f32.to_bits());

/// The image file header — port of the C++ `class Header`.
///
/// Stores attributes in a [`BTreeMap`] keyed by name so iteration is in sorted
/// name order, exactly matching the C++ `std::map<Name, Attribute*>`.
pub struct Header {
    /// The attribute map (`Header::_map`).
    map: BTreeMap<Name, Box<dyn Attribute>>,
    /// Per-header ZIP compression-level override — `None` falls back
    /// to the process-global default ([`Header::default_zip_compression_level`]).
    /// Mirrors the C++ `CompressionRecord` side-table's `zip_level`, but
    /// stored directly on the header instead of keyed by pointer identity.
    zip_compression_level: Option<i32>,
    /// Per-header DWA compression-level override — `None` falls back
    /// to the process-global default ([`Header::default_dwa_compression_level`]).
    /// Set automatically when a `dwaCompressionLevel` float attribute is
    /// inserted (`Header::insert_boxed`), mirroring C++ `Header::insert`'s
    /// special-cased hook (`ImfHeader.cpp:440-446`).
    dwa_compression_level: Option<f32>,
    /// `Header::_readsNothing` — true when the last [`Header::read_attributes_from`]
    /// call for this header parsed a header with zero attributes (an empty
    /// header list, distinct from an I/O error). Defaults `false`, matching
    /// every C++ constructor.
    reads_nothing: bool,
}

impl Header {
    // -----------------------------------------------------------------------
    // Constructors — mirror the three C++ Header(...) overloads + initialize()
    // -----------------------------------------------------------------------

    /// Full constructor — port of the C++
    /// `Header(displayWindow, dataWindow, pixelAspectRatio,
    /// screenWindowCenter, screenWindowWidth, lineOrder, compression)`.
    ///
    /// Inserts the eight required attributes in the same order as the C++
    /// `initialize()` helper.  Returns [`ExrError::Arg`] for an invalid pixel
    /// aspect ratio (non-normal or negative), mirroring the C++ `THROW(ArgExc…)`.
    pub fn with_windows(
        display_window: Box2i,
        data_window: Box2i,
        pixel_aspect_ratio: f32,
        screen_window_center: V2f,
        screen_window_width: f32,
        line_order: LineOrder,
        compression: Compression,
    ) -> Result<Self> {
        let mut h = Header {
            map: BTreeMap::new(),
            zip_compression_level: None,
            dwa_compression_level: None,
            reads_nothing: false,
        };
        h.initialize(
            display_window,
            data_window,
            pixel_aspect_ratio,
            screen_window_center,
            screen_window_width,
            line_order,
            compression,
        )?;
        Ok(h)
    }

    /// Construct from a width/height — port of the C++ `Header(int width,
    /// int height, …)` overload (display and data windows both
    /// `[(0,0), (width-1, height-1)]`).
    ///
    /// Returns [`ExrError::Arg`] if `width < 1 || height < 1`
    /// (`sanityCheckDisplayWindow`).
    pub fn with_dimensions(
        width: i32,
        height: i32,
        pixel_aspect_ratio: f32,
        screen_window_center: V2f,
        screen_window_width: f32,
        line_order: LineOrder,
        compression: Compression,
    ) -> Result<Self> {
        sanity_check_display_window(width, height)?;
        let dw = Box2i {
            min: Vec2 { x: 0, y: 0 },
            max: Vec2 {
                x: width - 1,
                y: height - 1,
            },
        };
        Self::with_windows(
            dw,
            dw,
            pixel_aspect_ratio,
            screen_window_center,
            screen_window_width,
            line_order,
            compression,
        )
    }

    /// Default-valued header — the C++ default constructor
    /// `Header(width=64, height=64, pixelAspectRatio=1,
    /// screenWindowCenter=(0,0), screenWindowWidth=1,
    /// lineOrder=INCREASING_Y, compression=ZIP_COMPRESSION)`.
    pub fn new() -> Self {
        // 64x64 with the documented defaults can never fail the
        // pixel-aspect-ratio or display-window checks, but we still propagate
        // through the fallible path and collapse the impossible error to keep a
        // panic-free library surface.
        Self::with_dimensions(
            64,
            64,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap_or_else(|_| Header {
            map: BTreeMap::new(),
            zip_compression_level: None,
            dwa_compression_level: None,
            reads_nothing: false,
        })
    }

    /// Port of the C++ free function `initialize(header, …)` — inserts the
    /// eight required attributes in their canonical order.
    //
    // Faithful 1:1 port of the C++ free function `initialize(header, …)`, which
    // takes the seven window/order/compression parameters; as an inherent
    // method it also carries `&mut self`, so the arg count exceeds clippy's
    // default threshold. The signature mirrors C++ exactly and is not split.
    #[allow(clippy::too_many_arguments)]
    fn initialize(
        &mut self,
        display_window: Box2i,
        data_window: Box2i,
        pixel_aspect_ratio: f32,
        screen_window_center: V2f,
        screen_window_width: f32,
        line_order: LineOrder,
        compression: Compression,
    ) -> Result<()> {
        self.insert("displayWindow", Box2iAttribute::new(display_window))?;
        self.insert("dataWindow", Box2iAttribute::new(data_window))?;
        // C++: if (!std::isnormal(par) || par < 0) THROW(ArgExc, …)
        if !pixel_aspect_ratio.is_normal() || pixel_aspect_ratio < 0.0 {
            return Err(ExrError::Arg("Invalid pixel aspect ratio".into()));
        }
        self.insert("pixelAspectRatio", FloatAttribute::new(pixel_aspect_ratio))?;
        self.insert(
            "screenWindowCenter",
            V2fAttribute::new(screen_window_center),
        )?;
        self.insert("screenWindowWidth", FloatAttribute::new(screen_window_width))?;
        self.insert("lineOrder", LineOrderAttribute::new(line_order))?;
        self.insert("compression", CompressionAttribute::new(compression))?;
        self.insert("channels", ChannelListAttribute::new(ChannelList::new()))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Attribute map manipulation — Header::insert / erase / operator[] / find
    // -----------------------------------------------------------------------

    /// Insert (or replace) an attribute — port of `Header::insert`.
    ///
    /// If no attribute with `name` exists, a clone is stored.  If one exists
    /// with the *same* type name, its value is replaced.  If one exists with a
    /// *different* type name, [`ExrError::Type`] is returned (the C++
    /// `TypeExc`).  An empty `name` is rejected with [`ExrError::Arg`].
    pub fn insert<A: Attribute>(&mut self, name: impl Into<Name>, attribute: A) -> Result<()> {
        self.insert_boxed(name, Box::new(attribute))
    }

    /// Insert an already-boxed attribute — the object-safe form of [`Header::insert`].
    ///
    /// This is the path used by the header reader, which constructs
    /// `Box<dyn Attribute>` values via the registry.
    ///
    /// `name` accepts anything convertible to a byte-string [`Name`]
    /// (attribute names are raw bytes in C++, no encoding assumed); the exact
    /// C++ `strncpy` truncation (first NUL, then 255 bytes) is applied by
    /// [`Name::new`] instead of the old UTF-8-boundary-respecting
    /// `xdr::truncate_name`.
    pub fn insert_boxed(&mut self, name: impl Into<Name>, attribute: Box<dyn Attribute>) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Image attribute name cannot be an empty string.".into(),
            ));
        }

        // `Header::insert`'s special-cased hook (`ImfHeader.cpp:440-446`)
        // — inserting a `dwaCompressionLevel` FLOAT attribute also updates
        // the compression-level record, so `dwa_level()` picks
        // it up even before any codec reads the attribute directly.
        if name == "dwaCompressionLevel"
            && attribute.type_name() == b"float"
            && let Some(typed) = attribute.as_any().downcast_ref::<FloatAttribute>()
        {
            self.dwa_compression_level = Some(typed.value);
        }

        match self.map.get_mut(&name) {
            Some(existing) => {
                if existing.type_name() != attribute.type_name() {
                    return Err(ExrError::Type(format!(
                        "Cannot assign a value of type \"{}\" to image attribute \
                         \"{}\" of type \"{}\".",
                        String::from_utf8_lossy(attribute.type_name()),
                        name,
                        String::from_utf8_lossy(existing.type_name())
                    )));
                }
                *existing = attribute;
            }
            None => {
                self.map.insert(name, attribute);
            }
        }
        Ok(())
    }

    /// [`Header::insert_boxed`] on the **read** path.
    ///
    /// A duplicate attribute name with a different type on read is a header
    /// open failure — core `INVALID_ATTR` (`ImfContext.cpp:136`, verified) —
    /// so this re-wraps [`ExrError::Type`] as [`ExrError::Input`]. A caller
    /// building a `Header` programmatically (not reading a file) keeps
    /// `insert`/`insert_boxed`'s own `Type` error.
    fn insert_boxed_on_read(&mut self, name: impl Into<Name>, attribute: Box<dyn Attribute>) -> Result<()> {
        self.insert_boxed(name, attribute).map_err(|e| match e {
            ExrError::Type(m) => ExrError::Input(format!("Unable to open image file. {m}")),
            other => other,
        })
    }

    /// Remove an attribute — port of `Header::erase`.
    ///
    /// A no-op if the name is absent.  An empty `name` is rejected with
    /// [`ExrError::Arg`] (matching the C++ `THROW(ArgExc, …)`).
    pub fn erase(&mut self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Image attribute name cannot be an empty string.".into(),
            ));
        }
        self.map.remove(Name::normalize(name.as_bytes()));
        Ok(())
    }

    /// Borrow an attribute by name — port of `Header::operator[] const`.
    ///
    /// Returns [`ExrError::Arg`] if the attribute is absent.
    pub fn get(&self, name: &str) -> Result<&dyn Attribute> {
        match self.map.get(Name::normalize(name.as_bytes())) {
            Some(a) => Ok(a.as_ref()),
            None => Err(ExrError::Arg(format!(
                "Cannot find image attribute \"{name}\"."
            ))),
        }
    }

    /// Mutably borrow an attribute by name — port of `Header::operator[]`.
    pub fn get_mut(&mut self, name: &str) -> Result<&mut (dyn Attribute + 'static)> {
        match self.map.get_mut(Name::normalize(name.as_bytes())) {
            Some(a) => Ok(a.as_mut()),
            None => Err(ExrError::Arg(format!(
                "Cannot find image attribute \"{name}\"."
            ))),
        }
    }

    /// Whether an attribute with the given name exists (any type).
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(Name::normalize(name.as_bytes()))
    }

    /// Find a typed attribute — port of `Header::findTypedAttribute<T>`.
    ///
    /// Returns `Some(&TypedAttribute<T>)` if an attribute named `name` exists
    /// and has value type `T`, else `None` (the C++ returns a null pointer in
    /// both the missing and wrong-type cases).
    pub fn find_typed_attribute<T: AttrValue>(&self, name: &str) -> Option<&TypedAttribute<T>> {
        self.map
            .get(Name::normalize(name.as_bytes()))
            .and_then(|a| a.as_any().downcast_ref::<TypedAttribute<T>>())
    }

    /// Mutable variant of [`find_typed_attribute`][Header::find_typed_attribute].
    pub fn find_typed_attribute_mut<T: AttrValue>(
        &mut self,
        name: &str,
    ) -> Option<&mut TypedAttribute<T>> {
        self.map
            .get_mut(Name::normalize(name.as_bytes()))
            .and_then(|a| a.as_any_mut().downcast_mut::<TypedAttribute<T>>())
    }

    /// Borrow a typed attribute — port of `Header::typedAttribute<T> const`.
    ///
    /// Returns [`ExrError::Arg`] if absent (C++ `ArgExc`) or [`ExrError::Type`]
    /// if present with the wrong type (C++ `TypeExc`).
    pub fn typed_attribute<T: AttrValue>(&self, name: &str) -> Result<&TypedAttribute<T>> {
        let attr = self.get(name)?;
        attr.as_any()
            .downcast_ref::<TypedAttribute<T>>()
            .ok_or_else(|| ExrError::Type("Unexpected attribute type.".into()))
    }

    /// Mutable variant of [`typed_attribute`][Header::typed_attribute].
    pub fn typed_attribute_mut<T: AttrValue>(
        &mut self,
        name: &str,
    ) -> Result<&mut TypedAttribute<T>> {
        // Resolve presence first so the error matches C++ ArgExc vs TypeExc.
        if !self.map.contains_key(Name::normalize(name.as_bytes())) {
            return Err(ExrError::Arg(format!(
                "Cannot find image attribute \"{name}\"."
            )));
        }
        self.find_typed_attribute_mut::<T>(name)
            .ok_or_else(|| ExrError::Type("Unexpected attribute type.".into()))
    }

    /// Iterate over `(name, &attribute)` in sorted name order — `Header::begin/end`.
    pub fn iter(&self) -> Iter<'_, Name, Box<dyn Attribute>> {
        self.map.iter()
    }

    /// Mutable iteration over `(name, &mut attribute)` in sorted name order.
    pub fn iter_mut(&mut self) -> IterMut<'_, Name, Box<dyn Attribute>> {
        self.map.iter_mut()
    }

    // -----------------------------------------------------------------------
    // Required-attribute typed accessors — Header::displayWindow() etc.
    //
    // These mirror the C++ accessors which `static_cast` the known attribute
    // and return a reference to its value.  Because the required attributes are
    // always present (seeded by the constructor), they cannot legitimately be
    // missing; we still return `Result` so a header doctored via `erase` cannot
    // panic the library.
    // -----------------------------------------------------------------------

    /// `const Box2i& displayWindow() const`.
    pub fn display_window(&self) -> Result<Box2i> {
        Ok(self.typed_attribute::<Box2i>("displayWindow")?.value)
    }

    /// `Box2i& displayWindow()` — mutable access to the value.
    pub fn display_window_mut(&mut self) -> Result<&mut Box2i> {
        Ok(&mut self.typed_attribute_mut::<Box2i>("displayWindow")?.value)
    }

    /// `const Box2i& dataWindow() const`.
    pub fn data_window(&self) -> Result<Box2i> {
        Ok(self.typed_attribute::<Box2i>("dataWindow")?.value)
    }

    /// `Box2i& dataWindow()`.
    pub fn data_window_mut(&mut self) -> Result<&mut Box2i> {
        Ok(&mut self.typed_attribute_mut::<Box2i>("dataWindow")?.value)
    }

    /// `const float& pixelAspectRatio() const`.
    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        Ok(self.typed_attribute::<f32>("pixelAspectRatio")?.value)
    }

    /// `float& pixelAspectRatio()`.
    pub fn pixel_aspect_ratio_mut(&mut self) -> Result<&mut f32> {
        Ok(&mut self.typed_attribute_mut::<f32>("pixelAspectRatio")?.value)
    }

    /// `const V2f& screenWindowCenter() const`.
    pub fn screen_window_center(&self) -> Result<V2f> {
        Ok(self.typed_attribute::<V2f>("screenWindowCenter")?.value)
    }

    /// `V2f& screenWindowCenter()`.
    pub fn screen_window_center_mut(&mut self) -> Result<&mut V2f> {
        Ok(&mut self.typed_attribute_mut::<V2f>("screenWindowCenter")?.value)
    }

    /// `const float& screenWindowWidth() const`.
    pub fn screen_window_width(&self) -> Result<f32> {
        Ok(self.typed_attribute::<f32>("screenWindowWidth")?.value)
    }

    /// `float& screenWindowWidth()`.
    pub fn screen_window_width_mut(&mut self) -> Result<&mut f32> {
        Ok(&mut self.typed_attribute_mut::<f32>("screenWindowWidth")?.value)
    }

    /// `const ChannelList& channels() const`.
    pub fn channels(&self) -> Result<&ChannelList> {
        Ok(&self.typed_attribute::<ChannelList>("channels")?.value)
    }

    /// `ChannelList& channels()`.
    pub fn channels_mut(&mut self) -> Result<&mut ChannelList> {
        Ok(&mut self.typed_attribute_mut::<ChannelList>("channels")?.value)
    }

    /// `const LineOrder& lineOrder() const`.
    pub fn line_order(&self) -> Result<LineOrder> {
        Ok(self.typed_attribute::<LineOrder>("lineOrder")?.value)
    }

    /// `LineOrder& lineOrder()`.
    pub fn line_order_mut(&mut self) -> Result<&mut LineOrder> {
        Ok(&mut self.typed_attribute_mut::<LineOrder>("lineOrder")?.value)
    }

    /// `const Compression& compression() const`.
    pub fn compression(&self) -> Result<Compression> {
        Ok(self.typed_attribute::<Compression>("compression")?.value)
    }

    /// `Compression& compression()`.
    pub fn compression_mut(&mut self) -> Result<&mut Compression> {
        Ok(&mut self.typed_attribute_mut::<Compression>("compression")?.value)
    }

    // -----------------------------------------------------------------------
    // Multipart attributes — name / type / version
    // -----------------------------------------------------------------------

    /// `void setName(const string&)` — insert the `name` string attribute.
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        self.insert("name", StringAttribute::new(ExrString::from(name)))
    }

    /// `bool hasName() const`.
    pub fn has_name(&self) -> bool {
        self.find_typed_attribute::<ExrString>("name").is_some()
    }

    /// `const string& name() const`.
    pub fn name(&self) -> Result<&str> {
        self.typed_attribute::<ExrString>("name")?.value.as_str()
            .ok_or_else(|| ExrError::Input("Non-UTF-8 bytes in `name` attribute".into()))
    }

    /// `void setType(const string&)` — insert the `type` string attribute.
    ///
    /// Returns [`ExrError::Arg`] for an unsupported type, and (matching the C++
    /// TODO) sets `version` to 1 for deep data when no version is present.
    pub fn set_type(&mut self, type_name: &str) -> Result<()> {
        if !is_supported_type(type_name) {
            return Err(ExrError::Arg(format!(
                "{type_name} is not a supported image type. The following are \
                 supported: {SCANLINE_IMAGE}, {TILED_IMAGE}, {DEEP_SCANLINE} or {DEEP_TILE}."
            )));
        }
        self.insert("type", StringAttribute::new(ExrString::from(type_name)))?;
        if is_deep_data(type_name) && !self.has_version() {
            self.set_version(1)?;
        }
        Ok(())
    }

    /// `bool hasType() const`.
    pub fn has_type(&self) -> bool {
        self.find_typed_attribute::<ExrString>("type").is_some()
    }

    /// `type` as the raw bytes it has on disk. C++ keeps it a `std::string` with no
    /// encoding check, and OpenEXRCore accepts any bytes (`check_populate_type`,
    /// `parse_header.c:1720-1822`); the open-time validation must therefore compare
    /// bytes, not go through the UTF-8-checked [`type_`][Header::type_].
    pub fn type_bytes(&self) -> Result<&[u8]> {
        Ok(self.typed_attribute::<ExrString>("type")?.value.as_bytes())
    }

    /// `const string& type() const`.
    pub fn type_(&self) -> Result<&str> {
        self.typed_attribute::<ExrString>("type")?.value.as_str()
            .ok_or_else(|| ExrError::Input("Non-UTF-8 bytes in `type` attribute".into()))
    }

    /// `void setVersion(const int)` — only version 1 is accepted.
    pub fn set_version(&mut self, version: i32) -> Result<()> {
        if version != 1 {
            return Err(ExrError::Arg("We can only process version 1".into()));
        }
        self.insert("version", IntAttribute::new(version))
    }

    /// `bool hasVersion() const`.
    pub fn has_version(&self) -> bool {
        self.find_typed_attribute::<i32>("version").is_some()
    }

    /// `const int& version() const`.
    pub fn version(&self) -> Result<i32> {
        Ok(self.typed_attribute::<i32>("version")?.value)
    }

    /// `void setChunkCount(int)`.
    pub fn set_chunk_count(&mut self, chunks: i32) -> Result<()> {
        self.insert("chunkCount", IntAttribute::new(chunks))
    }

    /// `bool hasChunkCount() const`.
    pub fn has_chunk_count(&self) -> bool {
        self.find_typed_attribute::<i32>("chunkCount").is_some()
    }

    /// `const int& chunkCount() const`.
    pub fn chunk_count(&self) -> Result<i32> {
        Ok(self.typed_attribute::<i32>("chunkCount")?.value)
    }

    /// `void setView(const string&)`.
    pub fn set_view(&mut self, view: &str) -> Result<()> {
        self.insert("view", StringAttribute::new(ExrString::from(view)))
    }

    /// `bool hasView() const`.
    pub fn has_view(&self) -> bool {
        self.find_typed_attribute::<ExrString>("view").is_some()
    }

    /// `const string& view() const`.
    pub fn view(&self) -> Result<&str> {
        self.typed_attribute::<ExrString>("view")?.value.as_str()
            .ok_or_else(|| ExrError::Input("Non-UTF-8 bytes in `view` attribute".into()))
    }

    // -----------------------------------------------------------------------
    // Tile description — Header::setTileDescription / tileDescription / has…
    // -----------------------------------------------------------------------

    /// `void setTileDescription(const TileDescription&)` — insert `tiles`.
    pub fn set_tile_description(&mut self, td: TileDescription) -> Result<()> {
        self.insert("tiles", TileDescriptionAttribute::new(td))
    }

    /// `bool hasTileDescription() const`.
    pub fn has_tile_description(&self) -> bool {
        self.find_typed_attribute::<TileDescription>("tiles").is_some()
    }

    /// `const TileDescription& tileDescription() const`.
    pub fn tile_description(&self) -> Result<TileDescription> {
        Ok(self.typed_attribute::<TileDescription>("tiles")?.value)
    }

    /// `TileDescription& tileDescription()`.
    pub fn tile_description_mut(&mut self) -> Result<&mut TileDescription> {
        Ok(&mut self.typed_attribute_mut::<TileDescription>("tiles")?.value)
    }

    // -----------------------------------------------------------------------
    // setPreviewImage / previewImage / hasPreviewImage.
    // -----------------------------------------------------------------------

    /// `void Header::setPreviewImage(const PreviewImage&)` — `insert("preview", …)`.
    pub fn set_preview_image(&mut self, preview: PreviewImage) -> Result<()> {
        self.insert("preview", PreviewImageAttribute::new(preview))
    }

    /// `PreviewImage& Header::previewImage()`.
    pub fn preview_image(&self) -> Result<&PreviewImage> {
        Ok(&self.typed_attribute::<PreviewImage>("preview")?.value)
    }

    /// Mutable counterpart of [`Header::preview_image`].
    pub fn preview_image_mut(&mut self) -> Result<&mut PreviewImage> {
        Ok(&mut self.typed_attribute_mut::<PreviewImage>("preview")?.value)
    }

    /// `bool Header::hasPreviewImage() const`.
    pub fn has_preview_image(&self) -> bool {
        self.find_typed_attribute::<PreviewImage>("preview").is_some()
    }

    /// `bool Header::readsNothing()` — true iff the last
    /// [`Header::read_attributes_from`] parsed a header with zero attributes
    /// (an immediately-empty header list). Always `false` for a
    /// programmatically constructed header.
    pub fn reads_nothing(&self) -> bool {
        self.reads_nothing
    }

    // -----------------------------------------------------------------------
    // setMaxImageSize / setMaxTileSize — port of the process-global
    // limits `Header::{set,get}Max{Image,Tile}Size` read/write via
    // `exr_{set,get}_default_maximum_{image,tile}_size`
    // (`OpenEXRCore/context.c`). A value of 0 for either dimension means
    // "unlimited" for that dimension, matching the C++ default and the
    // `maxWidth > 0` / `maxHeight > 0` guards in `sanityCheck` below.
    // -----------------------------------------------------------------------

    /// Set the process-wide maximum image (data window) width/height —
    /// mirrors `Header::setMaxImageSize`. `0` means unlimited.
    pub fn set_max_image_size(max_width: i32, max_height: i32) {
        MAX_IMAGE_WIDTH.store(max_width, std::sync::atomic::Ordering::Relaxed);
        MAX_IMAGE_HEIGHT.store(max_height, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the process-wide maximum image (data window) width/height —
    /// mirrors `Header::getMaxImageSize`.
    pub fn max_image_size() -> (i32, i32) {
        (
            MAX_IMAGE_WIDTH.load(std::sync::atomic::Ordering::Relaxed),
            MAX_IMAGE_HEIGHT.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Set the process-wide maximum tile width/height — mirrors
    /// `Header::setMaxTileSize`. `0` means unlimited.
    pub fn set_max_tile_size(max_width: i32, max_height: i32) {
        MAX_TILE_WIDTH.store(max_width, std::sync::atomic::Ordering::Relaxed);
        MAX_TILE_HEIGHT.store(max_height, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the process-wide maximum tile width/height — mirrors
    /// `Header::getMaxTileSize`.
    pub fn max_tile_size() -> (i32, i32) {
        (
            MAX_TILE_WIDTH.load(std::sync::atomic::Ordering::Relaxed),
            MAX_TILE_HEIGHT.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    // -----------------------------------------------------------------------
    // zipCompressionLevel / dwaCompressionLevel — port of the C++
    // `CompressionRecord` side-table (`ImfHeader.cpp:68-183`) that backs
    // `Header::{zip,dwa}CompressionLevel()` and the process-global
    // `setDefaultZipCompressionLevel`/`setDefaultDwaCompressionLevel` free
    // functions. C++ keys this by the `Header`'s own pointer identity in a
    // global map; here it is a field on `Header` directly (`None` = "not
    // overridden on this header, fall back to the process default"), which
    // is simpler and behaviourally equivalent (a moved/cloned `Header` keeps
    // its own record either way).
    // -----------------------------------------------------------------------

    /// Set the process-wide default ZIP compression level — mirrors the free
    /// function `Imf::setDefaultZipCompressionLevel`. Clamped to `-1..=9`
    /// (`-1` = "let the codec pick its own default"), matching C++
    /// `exr_set_default_zip_compression_level`.
    pub fn set_default_zip_compression_level(level: i32) {
        let clamped = level.clamp(-1, 9);
        DEFAULT_ZIP_LEVEL.store(clamped, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the process-wide default ZIP compression level.
    pub fn default_zip_compression_level() -> i32 {
        DEFAULT_ZIP_LEVEL.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the process-wide default DWA compression quality — mirrors the
    /// free function `Imf::setDefaultDwaCompressionLevel`. Clamped to
    /// `0.0..=100.0`, matching C++ `exr_set_default_dwa_compression_quality`.
    pub fn set_default_dwa_compression_level(level: f32) {
        let clamped = level.clamp(0.0, 100.0);
        DEFAULT_DWA_LEVEL_BITS.store(clamped.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the process-wide default DWA compression quality.
    pub fn default_dwa_compression_level() -> f32 {
        f32::from_bits(DEFAULT_DWA_LEVEL_BITS.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// This header's effective ZIP compression level — mirrors
    /// `Header::zipCompressionLevel() const`: the per-header override if one
    /// was set via [`Header::set_zip_level`], else the process-global
    /// default.
    ///
    /// Named `zip_level` (not `zip_compression_level`) because
    /// `"zipCompressionLevel"` is not itself a standard attribute — there is
    /// no naming clash to avoid the way there is for DWA, but the pair is
    /// named consistently for symmetry.
    ///
    /// `compressor::zip`'s writer factory
    /// reads this (through [`Header::codec_zip_level`]) instead of a hardcoded
    /// default level, so a caller that sets a per-header or
    /// process-global level actually affects the deflate stream C++ would
    /// produce for the same header.
    pub fn zip_level(&self) -> i32 {
        self.zip_compression_level.unwrap_or_else(Self::default_zip_compression_level)
    }

    /// Set this header's ZIP compression level override — mirrors the
    /// mutable `Header::zipCompressionLevel()` reference accessor (`= value`),
    /// which stores the value as is. An out-of-range level is resolved where
    /// the codec reads it: [`Header::codec_zip_level`].
    pub fn set_zip_level(&mut self, level: i32) {
        self.zip_compression_level = Some(level);
    }

    /// The ZIP level a codec actually uses for this header. C++
    /// `Compressor::Compressor` (`ImfCompressor.cpp:55`) hands the header's
    /// level to `exr_set_zip_compression_level`, which rejects anything outside
    /// `-1..=9` (`OpenEXRCore/part.c:568-577`); the return value is ignored, so
    /// the part keeps its context default — the process-global default level.
    pub fn codec_zip_level(&self) -> i32 {
        let level = self.zip_level();
        if (-1..=9).contains(&level) { level } else { Self::default_zip_compression_level() }
    }

    /// This header's effective DWA compression level — mirrors
    /// `Header::dwaCompressionLevel() const`: the per-header override
    /// (either set directly via [`Header::set_dwa_level`], or populated
    /// automatically when a `dwaCompressionLevel` float attribute is
    /// inserted — see [`Header::insert_boxed`]) if present, else the
    /// process-global default.
    ///
    /// Named `dwa_level` (not `dwa_compression_level`) because that name is
    /// already taken by the *standard attribute* accessor generated in
    /// `standard_attributes.rs` (`Result<&f32>`, the raw `dwaCompressionLevel`
    /// attribute value, no process-global fallback) — the two are genuinely
    /// different things and `compressor::dwa::DwaCompressor::new` already
    /// depends on the attribute accessor's exact name and `Result` signature.
    ///
    /// The attribute accessor
    /// (`header.dwa_compression_level()`) errors when the attribute is
    /// absent; this method never does — it falls back to a level set via
    /// [`Header::set_dwa_level`] with no attribute present (the C++
    /// `Header::dwaCompressionLevel()` API), and finally to the
    /// process-global default.
    pub fn dwa_level(&self) -> f32 {
        self.dwa_compression_level.unwrap_or_else(Self::default_dwa_compression_level)
    }

    /// Set this header's DWA compression level override — mirrors the
    /// mutable `Header::dwaCompressionLevel()` reference accessor, which stores
    /// the value as is (the `dwaCompressionLevel` attribute path does the same,
    /// see [`Header::insert_boxed`]). An out-of-range level is resolved where
    /// the codec reads it: [`Header::codec_dwa_level`].
    pub fn set_dwa_level(&mut self, level: f32) {
        self.dwa_compression_level = Some(level);
    }

    /// The DWA level a codec actually uses for this header. C++ passes the
    /// header's level to `exr_set_dwa_compression_level`, which accepts
    /// `0 ..= 65504 * 100000` as is and rejects anything else, NaN included
    /// (`OpenEXRCore/part.c:615-624`); the return value is ignored
    /// (`ImfCompressor.cpp:56`), so the part keeps the process-global default.
    pub fn codec_dwa_level(&self) -> f32 {
        let level = self.dwa_level();
        // The bound in f32 arithmetic, exactly as C++ computes it.
        let max = 65504.0_f32 * 100000.0_f32;
        if (0.0..=max).contains(&level) { level } else { Self::default_dwa_compression_level() }
    }

    /// Reset both compression-level overrides on this header to "use the
    /// process-global default" — mirrors `Header::resetDefaultCompressionLevels`.
    pub fn reset_default_compression_levels(&mut self) {
        self.zip_compression_level = None;
        self.dwa_compression_level = None;
    }

    // -----------------------------------------------------------------------
    // sanityCheck — port of Header::sanityCheck.
    // -----------------------------------------------------------------------

    /// Examine the header for consistency — port of `Header::sanityCheck`.
    ///
    /// Returns [`ExrError::Arg`] on the first problem found, mirroring the C++
    /// `ArgExc` throws. The max-image-size / max-tile-size limits are
    /// the process-global values set via [`Header::set_max_image_size`] /
    /// [`Header::set_max_tile_size`]; both default to `(0, 0)` ("unlimited"),
    /// matching the C++ default, so this is a no-op unless a caller opts in.
    ///
    /// `is_tiled` should be `true` for tiled / multi-resolution headers and
    /// `is_multipart_file` for headers belonging to a multipart file.
    ///
    /// **Callers:** writers call this directly and keep [`ExrError::Arg`]
    /// (`ImfOutputFile.cpp:626, 688` and the tiled/deep writers call
    /// `sanityCheck` themselves and let `ArgExc` propagate). **Readers**
    /// route through [`Header::sanity_check_at_open`] instead — see its doc
    /// for the read path.
    pub fn sanity_check(&self, is_tiled: bool, is_multipart_file: bool) -> Result<()> {
        // displayWindow validity.
        let dw = self.display_window()?;
        if dw.min.x > dw.max.x
            || dw.min.y > dw.max.y
            || dw.min.x <= -INT_MAX_DIV_2
            || dw.min.y <= -INT_MAX_DIV_2
            || dw.max.x >= INT_MAX_DIV_2
            || dw.max.y >= INT_MAX_DIV_2
        {
            return Err(ExrError::Arg(
                "Invalid display window in image header.".into(),
            ));
        }

        // dataWindow validity.
        let data = self.data_window()?;
        if data.min.x > data.max.x
            || data.min.y > data.max.y
            || data.min.x <= -INT_MAX_DIV_2
            || data.min.y <= -INT_MAX_DIV_2
            || data.max.x >= INT_MAX_DIV_2
            || data.max.y >= INT_MAX_DIV_2
        {
            return Err(ExrError::Arg("Invalid data window in image header.".into()));
        }

        // Process-global max-image-size limits — `ImfHeader.cpp:924-964`.
        // `0` (the default) means unlimited, matching C++.
        let (max_image_width, max_image_height) = Self::max_image_size();
        let w = data.max.x - data.min.x + 1;
        if max_image_width > 0 && max_image_width < w {
            return Err(ExrError::Arg(format!(
                "The width of the data window exceeds the maximum width of \
                 {max_image_width} pixels."
            )));
        }
        let h = data.max.y - data.min.y + 1;
        if max_image_height > 0 && max_image_height < h {
            return Err(ExrError::Arg(format!(
                "The height of the data window exceeds the maximum height of \
                 {max_image_height} pixels."
            )));
        }
        // chunkCount must not exceed the maximum image area (only reachable
        // for unknown types or damaged files — a regular image type would
        // already have failed the width/height checks above).
        if max_image_width > 0
            && max_image_height > 0
            && self.has_chunk_count()
            && let Ok(chunk_count) = self.chunk_count()
        {
            let max_area = (max_image_width as u64) * (max_image_height as u64);
            if (chunk_count as u64) > max_area {
                return Err(ExrError::Arg(format!(
                    "chunkCount exceeds maximum area of {max_area} pixels."
                )));
            }
        }

        // pixelAspectRatio range.
        let par = self.pixel_aspect_ratio()?;
        const MIN_PIXEL_ASPECT_RATIO: f32 = 1e-6;
        const MAX_PIXEL_ASPECT_RATIO: f32 = 1e+6;
        // C++: `!isnormal(par) || par < MIN || par > MAX`; the range test is
        // expressed with `RangeInclusive::contains` for clarity (identical
        // semantics — `is_normal` already excludes NaN).
        if !par.is_normal() || !(MIN_PIXEL_ASPECT_RATIO..=MAX_PIXEL_ASPECT_RATIO).contains(&par) {
            return Err(ExrError::Arg(
                "Invalid pixel aspect ratio in image header.".into(),
            ));
        }

        // screenWindowWidth must be non-negative.
        let sww = self.screen_window_width()?;
        if sww < 0.0 {
            return Err(ExrError::Arg(
                "Invalid screen window width in image header.".into(),
            ));
        }

        // Multipart files require name + type attributes.
        if is_multipart_file {
            if !self.has_name() {
                return Err(ExrError::Arg(MISSING_NAME_MSG.into()));
            }
            if !self.has_type() {
                return Err(ExrError::Arg(MISSING_TYPE_MSG.into()));
            }
        }

        // An absent `type` attr is empty. A non-UTF-8 one (`type_()`'s only error once
        // `has_type()`) can name no supported type, so it skips the rest like any other
        // unknown one.
        let part_type: &str = match self.has_type().then(|| self.type_()) {
            None => "",
            Some(Ok(t)) => t,
            Some(Err(_)) => return Ok(()),
        };

        // Unsupported types skip the remaining checks (they may not hold).
        if !part_type.is_empty() && !is_supported_type(part_type) {
            return Ok(());
        }

        let is_deep = is_deep_data(part_type);

        // Line order + tile description validity.
        let line_order = self.line_order()?;
        if is_tiled {
            if !self.has_tile_description() {
                return Err(ExrError::Arg(MISSING_TILE_DESC_MSG.into()));
            }
            let td = self.tile_description()?;
            // C++ compares against INT_MAX; xSize/ySize are u32 here, so the
            // "> INT_MAX" branch can only trip for genuinely oversized values.
            if td.x_size == 0 || td.y_size == 0 || td.x_size > i32::MAX as u32 || td.y_size > i32::MAX as u32 {
                return Err(ExrError::Arg("Invalid tile size in image header.".into()));
            }
            // Process-global max-tile-size limits — `ImfHeader.cpp:1055-1084`.
            let (max_tile_width, max_tile_height) = Self::max_tile_size();
            if max_tile_width > 0 && (max_tile_width as u32) < td.x_size {
                return Err(ExrError::Arg(format!(
                    "The width of a tile exceeds the maximum width of {max_tile_width} pixels."
                )));
            }
            if max_tile_height > 0 && (max_tile_height as u32) < td.y_size {
                return Err(ExrError::Arg(format!(
                    "The height of a tile exceeds the maximum height of {max_tile_height} pixels."
                )));
            }
            if !matches!(
                td.mode,
                crate::attr::LevelMode::OneLevel
                    | crate::attr::LevelMode::MipmapLevels
                    | crate::attr::LevelMode::RipmapLevels
            ) {
                return Err(ExrError::Arg("Invalid level mode in image header.".into()));
            }
            if !matches!(
                td.rounding_mode,
                crate::attr::LevelRoundingMode::RoundUp | crate::attr::LevelRoundingMode::RoundDown
            ) {
                return Err(ExrError::Arg(
                    "Invalid level rounding mode in image header.".into(),
                ));
            }
            if !matches!(
                line_order,
                LineOrder::IncreasingY | LineOrder::DecreasingY | LineOrder::RandomY
            ) {
                return Err(ExrError::Arg("Invalid line order in image header.".into()));
            }
            // C++ `sanityCheck` calls `getTiledChunkOffsetTableSize`
            // (`ImfHeader.cpp` -> `ImfTiledMisc.cpp:309-360`), which throws a
            // `LogicExc` when the tile count overflows `i32`. Readers already
            // get this for free via `read_prologue`'s own offset-table sizing,
            // but header-only consumers of `sanity_check` (`read_checked`, the
            // writer-side callers) did not, so a 1x1-tile header spanning a
            // huge data window passed here even though no reader could open it.
            crate::tiled::misc::get_tiled_chunk_offset_table_size(self)?;
        } else if !matches!(line_order, LineOrder::IncreasingY | LineOrder::DecreasingY) {
            return Err(ExrError::Arg("Invalid line order in image header.".into()));
        }

        // Compression must be a known value (and valid for deep when deep).
        let compression = self.compression()?;
        if !compression.is_valid() {
            return Err(ExrError::Arg(
                "Unknown compression type in image header.".into(),
            ));
        }
        if is_deep && !compression.is_valid_deep() {
            return Err(ExrError::Arg(
                "Compression type in header not valid for deep data".into(),
            ));
        }

        // Channel list checks.
        let channels = self.channels()?;
        if channels.is_empty() {
            return Err(ExrError::Arg(
                "Missing or empty channel list in header".into(),
            ));
        }

        if is_tiled || is_deep {
            for (name, ch) in channels.iter() {
                if !is_valid_pixel_type(ch.type_) {
                    return Err(ExrError::Arg(format!(
                        "Pixel type of \"{name}\" image channel is invalid."
                    )));
                }
                if ch.x_sampling != 1 {
                    return Err(ExrError::Arg(format!(
                        "The x subsampling factor for the \"{name}\" channel is not 1."
                    )));
                }
                if ch.y_sampling != 1 {
                    return Err(ExrError::Arg(format!(
                        "The y subsampling factor for the \"{name}\" channel is not 1."
                    )));
                }
            }
        } else {
            for (name, ch) in channels.iter() {
                if !is_valid_pixel_type(ch.type_) {
                    return Err(ExrError::Arg(format!(
                        "Pixel type of \"{name}\" image channel is invalid."
                    )));
                }
                if ch.x_sampling < 1 {
                    return Err(ExrError::Arg(format!(
                        "The x subsampling factor for the \"{name}\" channel is invalid."
                    )));
                }
                if ch.y_sampling < 1 {
                    return Err(ExrError::Arg(format!(
                        "The y subsampling factor for the \"{name}\" channel is invalid."
                    )));
                }
                if data.min.x % ch.x_sampling != 0 {
                    return Err(ExrError::Arg(format!(
                        "The minimum x coordinate of the image's data window is not a \
                         multiple of the x subsampling factor of the \"{name}\" channel."
                    )));
                }
                if data.min.y % ch.y_sampling != 0 {
                    return Err(ExrError::Arg(format!(
                        "The minimum y coordinate of the image's data window is not a \
                         multiple of the y subsampling factor of the \"{name}\" channel."
                    )));
                }
                if (data.max.x - data.min.x + 1) % ch.x_sampling != 0 {
                    return Err(ExrError::Arg(format!(
                        "Number of pixels per row in the image's data window is not a \
                         multiple of the x subsampling factor of the \"{name}\" channel."
                    )));
                }
                if (data.max.y - data.min.y + 1) % ch.y_sampling != 0 {
                    return Err(ExrError::Arg(format!(
                        "Number of pixels per column in the image's data window is not a \
                         multiple of the y subsampling factor of the \"{name}\" channel."
                    )));
                }
            }
        }

        Ok(())
    }

    /// The read-open wrapper around [`Header::sanity_check`].
    ///
    /// C++ readers never call `Header::sanityCheck` themselves — that method
    /// is a **writer**-only path (`ImfOutputFile.cpp:626, 688` and the
    /// tiled/deep writers). A reader's header validation instead runs through
    /// OpenEXRCore's `validation.c`, whose failures — `ImfContext.cpp:127-138`
    /// (verified) — become `InputExc` "Unable to open image file … {reason}",
    /// **except** a missing required attribute the core cannot default-fill,
    /// which stays `ArgExc`.
    ///
    /// This port reuses `sanity_check`'s checks on the read path too (rather
    /// than duplicating them against `validation.c`), so this wrapper
    /// reclassifies its `Arg` result the same way: the three "is a required
    /// attribute missing" messages stay [`ExrError::Arg`]; every other
    /// failure becomes [`ExrError::Input`]. Matching on these three
    /// crate-owned constants (not on arbitrary text) is the deliberate
    /// classification key — the alternative, giving `sanity_check` two
    /// return shapes, would burden the writer call sites that need none of
    /// this.
    pub fn sanity_check_at_open(&self, is_tiled: bool, is_multipart_file: bool) -> Result<()> {
        match self.sanity_check(is_tiled, is_multipart_file) {
            Err(ExrError::Arg(m))
                if m == MISSING_NAME_MSG || m == MISSING_TYPE_MSG || m == MISSING_TILE_DESC_MSG =>
            {
                Err(ExrError::Arg(m))
            }
            Err(ExrError::Arg(m)) => Err(ExrError::Input(format!("Unable to open image file. {m}"))),
            other => other,
        }
    }

    // -----------------------------------------------------------------------
    // Long-name detection — port of ImfMisc.cpp usesLongNames
    // -----------------------------------------------------------------------

    /// Whether the header contains any attribute name, attribute type name, or
    /// channel name of 32 or more bytes — port of `usesLongNames(header)`.
    ///
    /// When true, the writer must set [`LONG_NAMES_FLAG`][version::LONG_NAMES_FLAG]
    /// in the file version word.
    pub fn uses_long_names(&self) -> bool {
        for (name, attr) in self.map.iter() {
            if name.len() > SHORT_NAME_LIMIT || attr.type_name().len() > SHORT_NAME_LIMIT {
                return true;
            }
        }
        // Channel names.
        if let Ok(channels) = self.channels() {
            for (name, _) in channels.iter() {
                if name.len() > SHORT_NAME_LIMIT {
                    return true;
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Serialisation — Header::writeTo / readFrom (+ version-word computation)
    // -----------------------------------------------------------------------

    /// Compute the version word for this header — port of the body of
    /// `GenericOutputFile::writeMagicNumberAndVersionField`.
    ///
    /// `base_flags` lets the caller inject the flags it owns (e.g.
    /// [`MULTI_PART_FILE_FLAG`][version::MULTI_PART_FILE_FLAG] for a multipart
    /// file).  On top of that this folds in:
    /// * [`NON_IMAGE_FLAG`][version::NON_IMAGE_FLAG] when the header declares a
    ///   deep `type`, **or** [`TILED_FLAG`][version::TILED_FLAG] when it carries
    ///   a tile description (mutually exclusive, exactly as C++);
    /// * [`LONG_NAMES_FLAG`][version::LONG_NAMES_FLAG] when
    ///   [`uses_long_names`][Header::uses_long_names] is true.
    pub fn version_field(&self, base_flags: u32) -> u32 {
        let mut version = version::EXR_VERSION | base_flags;

        // Deep vs. tiled (mutually exclusive in the single-part path).
        // C++: `header.hasType() && isDeepData(header.type())`.
        let is_deep = if self.has_type() {
            self.type_().map(is_deep_data).unwrap_or(false)
        } else {
            false
        };
        if is_deep {
            version |= version::NON_IMAGE_FLAG;
        } else if self.has_tile_description() {
            version |= version::TILED_FLAG;
        }

        if self.uses_long_names() {
            version |= version::LONG_NAMES_FLAG;
        }
        version
    }

    /// Write magic + version word + every attribute — combines C++
    /// `GenericOutputFile::writeMagicNumberAndVersionField` and
    /// `Header::writeTo`.
    ///
    /// Layout, byte-for-byte:
    /// 1. `MAGIC` (i32 LE)
    /// 2. version word (i32 LE) — see [`version_field`][Header::version_field]
    /// 3. for each attribute in sorted name order:
    ///    name (`\0`-terminated) · type name (`\0`-terminated) ·
    ///    value length (i32 LE) · value bytes
    /// 4. an empty name (`\0`) terminating the header
    ///
    /// `base_flags` is forwarded to [`version_field`][Header::version_field].
    /// Returns the byte offset of any `preview` attribute's value (0 if none),
    /// matching the C++ `writeTo` return used by `updatePreviewImage`.
    pub fn write_to(&self, o: &mut dyn OStream, base_flags: u32) -> Result<u64> {
        // 1. magic + 2. version word.
        xdr::write_i32(o, version::MAGIC as i32)?;
        let version_word = self.version_field(base_flags) as i32;
        xdr::write_i32(o, version_word)?;

        // 3. + 4. the attribute list and its terminating empty name.
        self.write_attributes_to(o)
    }

    /// Write only the attribute list + the header-terminating empty name —
    /// port of the body of C++ `Header::writeTo` *without* the magic number /
    /// version word (those are written once by
    /// `GenericOutputFile::writeMagicNumberAndVersionField`).
    ///
    /// This is the unification seam shared by the single-part writers (via
    /// [`write_to`][Header::write_to], which prepends magic + version) and the
    /// multipart writer (which writes magic + version once for the whole file,
    /// then calls this for every part header).
    ///
    /// Layout, byte-for-byte:
    /// * for each attribute in sorted name order: name (`\0`) · type name
    ///   (`\0`) · value length (i32 LE) · value bytes
    /// * an empty name (`\0`) terminating this header
    ///
    /// Returns the byte offset of any `preview` attribute's value (0 if none),
    /// matching the C++ `writeTo` return used by `updatePreviewImage`.
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfHeader.cpp> (`Header::writeTo`).
    pub fn write_attributes_to(&self, o: &mut dyn OStream) -> Result<u64> {
        // The version number passed to each attribute's writer (C++ uses the
        // bare EXR_VERSION here, not the flagged word).
        let attr_version = version::EXR_VERSION as i32;

        let mut preview_position: u64 = 0;

        // attributes, in sorted name order (BTreeMap iteration order).
        for (name, attr) in self.map.iter() {
            // name + type name, each NUL-terminated. Both are raw
            // byte strings (no encoding assumed) — a registered type's name is
            // always ASCII, but an OpaqueAttribute's may not be.
            xdr::write_name(o, name.as_bytes())?;
            xdr::write_name(o, attr.type_name())?;

            // Serialise the value to a temporary buffer to learn its length,
            // exactly as C++ does with a StdOSStream.
            let mut tmp = MemOStream::new();
            attr.write_value_to(&mut tmp, attr_version)?;
            let bytes = tmp.into_data();

            // value length (i32, checked) then the bytes.
            xdr::write_wire_len(o, bytes.len())?;
            // C++ records the stream position of the attribute named "preview"
            // (a PreviewImageAttribute) for OutputFile::updatePreviewImage.
            if name == "preview" && attr.type_name() == b"preview" {
                preview_position = o.tellp()?;
            }
            xdr::write_chars(o, &bytes)?;
        }

        // empty name to mark the end of the header.
        xdr::write_string(o, "")?;

        Ok(preview_position)
    }

    /// Read and **validate** a single-part file's header — the entry point any
    /// consumer that is going to *use* the header should call.
    ///
    /// [`read_from`][Header::read_from] is the 1:1 port of C++
    /// `Header::readFrom`, which deliberately does not validate: in C++ the
    /// check happens when a file is *opened*, not when a header is parsed —
    /// `parse_header.c:2813` calls `internal_exr_validate_read_part`
    /// (`validation.c:697`), the C-core equivalent of `Header::sanityCheck`, for
    /// every part. Every reader in this crate mirrors that by calling
    /// [`sanity_check`][Header::sanity_check] in its constructor; a consumer
    /// that parses a header *without* opening a reader (the `exr-rs info` CLI,
    /// the Python header inspector) has no such constructor, and without this
    /// entry point it would go on to compute with attacker-controlled values —
    /// a crafted `dataWindow` of `i32::MIN..i32::MAX` made `info` print a
    /// wrapped `[0 x 4]` extent in release and panic in debug.
    ///
    /// `is_tiled` for the check comes from the file's own version flags, as in
    /// C++ `InputFile::initialize`.
    ///
    /// # Errors
    /// * Everything [`read_from`][Header::read_from] reports (bad magic,
    ///   unknown version flags, malformed attributes).
    /// * [`ExrError::Arg`] from [`sanity_check`][Header::sanity_check] for an
    ///   invalid window / aspect ratio / line order / compression / channel list.
    /// * [`ExrError::Arg`] if the file is **multipart**: only the first part's
    ///   header follows the version word, so returning it would misdescribe the
    ///   file. Open those with [`MultiPartInputFile`](crate::MultiPartInputFile),
    ///   which reads and checks every part.
    pub fn read_checked(i: &mut dyn IStream) -> Result<(Header, i32)> {
        let (mut header, version_word) = Self::read_from(i)?;
        let flags = version::get_flags(version_word as u32);
        if version::is_multi_part(flags) {
            return Err(ExrError::Arg(
                "This is a multipart file; only its first part header follows the \
                 version word. Use MultiPartInputFile to read every part."
                    .into(),
            ));
        }
        // Apply the same non-strict required-attribute defaults and
        // name/type/version handling the reader constructors get via
        // `multipart::input::read_prologue` (-> `validate_read_part`), so a
        // header-only consumer sees exactly what `InputFile` would.
        // `auto_add_type = true` matches the C++ `MultiPartInputFile` ctor's
        // default (`multipart/input.rs:513`), which is what a single-part
        // `InputFile` construction goes through.
        let is_tiled = header.validate_for_read(
            version::is_tiled(flags),
            version::is_non_image(flags),
            false,
            true,
        )?;
        header.sanity_check_at_open(is_tiled, false)?;
        Ok((header, version_word))
    }

    /// Apply C++ 3.4 core's non-strict required-attribute defaulting and
    /// name/type/version handling — the shared body of `validate_req_attr`
    /// (`adddefault=1`) plus the legacy `autoAddType` — shared by
    /// [`read_checked`][Header::read_checked] and
    /// `multipart::input::validate_read_part`, so the two cannot drift.
    ///
    /// * Fills the 7 required attributes C++ readers default rather than
    ///   reject when absent (`compression`, `dataWindow`, `displayWindow`,
    ///   `lineOrder`, `pixelAspectRatio`, `screenWindowCenter`,
    ///   `screenWindowWidth`).
    /// * For a multipart or deep (`non_image`) part: `name` (multipart only)
    ///   and `type` remain FATAL when absent (`ExrError::Input`); a deep part
    ///   missing `version` defaults to 1.
    /// * For an ordinary single-part image missing `type`: auto-adds it from
    ///   `tiled_flag` when `auto_add_type` is set (the legacy
    ///   `MultiPartInputFile` "silently invent a type" behaviour).
    ///
    /// Returns the effective `is_tiled` flag for
    /// [`sanity_check`][Header::sanity_check]: the file's version-word flag
    /// for a plain single-part image, or derived from the (possibly just
    /// defaulted) `type` attribute for a deep/multipart part — mirroring
    /// C++ 3.4 core (`parse_header.c:2789-2810`) where only a deep or
    /// multipart file lets `type` decide storage.
    ///
    /// This does **not** fix a `type` that disagrees with the storage the
    /// flags/type otherwise imply (`multipart::input::Storage`'s
    /// `validate_part_type` non-strict correction) — that needs the
    /// `Storage` enum, which stays in `multipart::input`.
    pub fn validate_for_read(
        &mut self,
        tiled_flag: bool,
        non_image: bool,
        multipart: bool,
        auto_add_type: bool,
    ) -> Result<bool> {
        if !self.contains("compression") {
            self.insert("compression", CompressionAttribute::new(Compression::Zip))?;
        }
        // C++ default window is {{0,0},{63,63}} for BOTH windows.
        let default_window = Box2i {
            min: Vec2 { x: 0, y: 0 },
            max: Vec2 { x: 63, y: 63 },
        };
        if !self.contains("dataWindow") {
            self.insert("dataWindow", Box2iAttribute::new(default_window))?;
        }
        if !self.contains("displayWindow") {
            self.insert("displayWindow", Box2iAttribute::new(default_window))?;
        }
        if !self.contains("lineOrder") {
            self.insert("lineOrder", LineOrderAttribute::new(LineOrder::IncreasingY))?;
        }
        if !self.contains("pixelAspectRatio") {
            self.insert("pixelAspectRatio", FloatAttribute::new(1.0))?;
        }
        if !self.contains("screenWindowCenter") {
            self.insert("screenWindowCenter", V2fAttribute::new(V2f { x: 0.0, y: 0.0 }))?;
        }
        if !self.contains("screenWindowWidth") {
            self.insert("screenWindowWidth", FloatAttribute::new(1.0))?;
        }

        if multipart || non_image {
            if multipart && !self.has_name() {
                return Err(ExrError::Input(
                    "'name' attribute for multipart file not found".into(),
                ));
            }
            if !self.has_type() {
                return Err(ExrError::Input(
                    "'type' attribute for v2+ file not found".into(),
                ));
            }
            if non_image && !self.has_version() {
                self.set_version(1)?;
            }
        } else if auto_add_type && !self.has_type() {
            self.set_type(if tiled_flag { TILED_IMAGE } else { SCANLINE_IMAGE })?;
        }

        // Effective `is_tiled` for `sanity_check`: flags alone for a plain
        // single-part image; the (now possibly defaulted) `type` attribute
        // for a deep or multipart part.
        let is_tiled = if !non_image && !multipart {
            tiled_flag
        } else {
            matches!(self.type_(), Ok(TILED_IMAGE) | Ok(DEEP_TILE))
        };
        Ok(is_tiled)
    }

    /// Read magic + version word + every attribute into a fresh header —
    /// combines magic/version validation with `Header::readFrom`.
    ///
    /// **This does not validate the header's contents.** It is the 1:1 port of
    /// C++ `Header::readFrom`, which does not either; the readers call
    /// [`sanity_check`][Header::sanity_check] themselves right after. If you are
    /// parsing a header to *use* its values rather than to hand it to a reader,
    /// call [`read_checked`][Header::read_checked] instead.
    ///
    /// Returns the reconstructed [`Header`] and the raw version word (so the
    /// caller can inspect the tiled / deep / multipart / long-names flags).
    ///
    /// Unknown attribute types are preserved as
    /// [`OpaqueAttribute`][crate::attr::OpaqueAttribute] via the registry, so
    /// every header round-trips losslessly.  Attribute and type names are
    /// capped at `Name::MAX_LENGTH` (255) bytes, matching C++ `readFrom`.
    pub fn read_from(i: &mut dyn IStream) -> Result<(Header, i32)> {
        // Magic number.
        let magic = xdr::read_i32(i)?;
        if (magic as u32) != version::MAGIC {
            return Err(ExrError::Input(
                "File is not an OpenEXR file.".into(),
            ));
        }

        // Version word: version number, known flags, and flag-combination
        // validity — matches C++ 3.4 core `parse_header.c` right after
        // the magic number, so a version-3 file or a TILED+NON_IMAGE /
        // TILED+MULTI_PART word is rejected here rather than only by the
        // reader constructors' own `read_prologue`.
        let version_word = xdr::read_i32(i)?;
        version::validate_version_word(version_word as u32)?;

        let header = Header::read_attributes_from(i, version_word)?;

        Ok((header, version_word))
    }

    /// Read one header's attribute list (terminated by an empty attribute
    /// name) into a fresh [`Header`] — port of the body of C++
    /// `Header::readFrom` *without* the magic / version validation (those are
    /// read once by `GenericInputFile::readMagicNumberAndVersionField`).
    ///
    /// This is the unification seam shared by the single-part reader (via
    /// [`read_from`][Header::read_from], which first reads magic + version) and
    /// the multipart reader (which reads magic + version once, then reads each
    /// part header with this and detects the end of the header list by an
    /// immediately-empty name).
    ///
    /// `version_word` is the raw file version word, threaded to attribute
    /// readers that depend on it (e.g. long-name handling).
    ///
    /// C++ cite: <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfHeader.cpp> (`Header::readFrom`).
    pub fn read_attributes_from(i: &mut dyn IStream, version_word: i32) -> Result<Header> {
        let mut header = Header {
            map: BTreeMap::new(),
            zip_compression_level: None,
            dwa_compression_level: None,
            reads_nothing: false,
        };

        let mut name_buf: Vec<u8> = Vec::new();
        let mut type_buf: Vec<u8> = Vec::new();
        // `Header::_readsNothing`: true iff the header list
        // was empty (the very first name was already the terminator) —
        // counts every attribute NAME encountered, even one later dropped by
        // the non-strict recovery below, matching C++'s `attrCount++` placement
        // right after the name is read (`ImfHeader.cpp:1370-1385`).
        let mut attr_count: usize = 0;

        loop {
            // Attribute name; empty name terminates the header.
            //
            // An over-long name/type is a C++ `read_text` error raised after
            // `max_len + 1` bytes; the non-strict loop drops it and parses on from
            // there, so `read_name` reports it instead of failing the whole header.
            if !xdr::read_name(i, &mut name_buf, NAME_MAX_LENGTH)? {
                log::warn!("dropping header attribute: name longer than {NAME_MAX_LENGTH} bytes");
                continue;
            }
            if name_buf.is_empty() {
                break;
            }
            attr_count += 1;
            // An attribute name is a raw byte string in C++ (no
            // encoding check anywhere in `parse_header.c`'s `read_text`);
            // accept non-UTF-8 instead of aborting the whole header read.
            let name = Name::new(name_buf.as_slice());

            // Type name: raw bytes, matching C++ (an unregistered
            // type's on-disk name is never encoding-checked; it round-trips
            // byte-exact through `OpaqueAttribute` instead of being rejected).
            if !xdr::read_name(i, &mut type_buf, NAME_MAX_LENGTH)? {
                log::warn!("dropping header attribute '{name}': type name longer than {NAME_MAX_LENGTH} bytes");
                continue;
            }
            let type_name = Name::new(type_buf.as_slice());

            // C++ core `pull_attr` rejects an empty type name BEFORE reading
            // the size field ("Invalid empty string encountered parsing attribute
            // type", `parse_header.c:2072`), and the non-strict loop resumes right
            // after the type's NUL -- the size bytes become the next attribute's
            // name (oracle-verified).
            if type_name.is_empty() {
                log::warn!("dropping header attribute '{name}': empty type name");
                continue;
            }

            // Declared value size.
            let size = xdr::read_i32(i)?;
            let attr_start = i.tellg()?;
            let remaining = i.size().map(|total| total.saturating_sub(attr_start));
            let required = name.to_str().is_some_and(|s| REQUIRED_ATTR_TYPES.contains_key(s));

            // A repeated non-required name: `exr_attr_list_add*` refuses a `string`
            // (it always asks for extra storage) or a different type before the value
            // is touched (`attributes.c:576-620,776-795,900-915`) -- no bytes
            // consumed. The same type is overwritten, and if that overwrite then
            // fails C++ removes the earlier value too (`forget` below).
            let prev = (!required).then(|| header.map.get(&name).map(|a| a.type_name() == type_name.as_bytes())).flatten();
            if prev == Some(false) || (prev == Some(true) && type_name == "string") {
                log::warn!("dropping header attribute '{name}' (type '{type_name}'): repeated name");
                continue;
            }
            let forget = |header: &mut Header| {
                if prev == Some(true) {
                    header.map.remove(&name);
                }
            };

            // C++ core rejects a wrong declared size BEFORE consuming a value
            // byte and without skipping it (`extract_attr_*` / `check_bad_attrsz`
            // return straight away); the non-strict loop then simply carries on
            // parsing at the cursor, i.e. inside the value bytes. Reproduce
            // that exactly: `comp_invalid_unknown.exr` (a `displayWindow`
            // declaring 32528 bytes) opens in C++ because the value's first
            // byte is the header terminator.
            if bad_size_resync(&name, &type_name, size, remaining) {
                log::warn!(
                    "dropping header attribute '{name}' (type '{type_name}'): \
                     invalid declared size {size}"
                );
                forget(&mut header);
                continue;
            }

            if size < 0 {
                return Err(ExrError::Input(
                    "Invalid size field in header attribute".into(),
                ));
            }

            // `preview` / `stringvector` / `bytes` fail part-way through the value:
            // C++ stops wherever its parser stopped, again without skipping.
            if !required
                && let Some(used) = value_shape_fail(i, &type_name, size)?
            {
                log::warn!(
                    "dropping header attribute '{name}' (type '{type_name}'): malformed value"
                );
                i.seekg(attr_start + used)?;
                forget(&mut header);
                continue;
            }

            // Guard for the paths where C++ skips the declared size (channels,
            // wrong-typed required attributes, ...) and so runs into EOF, which
            // is fatal there too: a truncated file can otherwise declare a
            // ~2 GiB attribute; per-reader bounded allocation is the second line.
            if let Some(remaining) = remaining
                && (size as u64) > remaining
            {
                return Err(ExrError::Input(format!(
                    "header attribute declares {size} bytes but only {remaining} remain"
                )));
            }

            // --- non-strict required-attribute type dispatch --------------
            // C++ 3.4 core `check_req_attr` (`parse_header.c:1966`) routes a
            // fixed set of by-name attributes to a specific
            // `check_populate_*` validator that enforces BOTH the exact wire
            // type name and (for the fixed-size ones) the exact declared
            // size; a mismatch drops the attribute (`pull_attr` failure ->
            // non-strict `sequential_skip(attrsz)`) instead of storing a
            // value of the wrong type/width under a required name (e.g. a
            // `pixelAspectRatio` typed `double`, or a `float` with a padded
            // declared size of 8).
            if let Some(&(expected_type, expected_size)) =
                name.to_str().and_then(|s| REQUIRED_ATTR_TYPES.get(s))
                && (type_name != expected_type
                    || expected_size.is_some_and(|n| size as usize != n))
            {
                log::warn!(
                    "dropping required header attribute '{name}': expected type \
                     '{expected_type}'{}, found type '{type_name}' size {size}",
                    expected_size
                        .map(|n| format!(" size {n}"))
                        .unwrap_or_default()
                );
                i.seekg(attr_start + size as u64)?;
                continue;
            }

            // --- `chlist`: C++ `extract_attr_chlist` appends to a list that already
            // exists under this name (`channels` by name, any other name because
            // `exr_attr_list_add_by_type` hands back the earlier same-typed entry), so
            // channels added before a failure stay. The parser also leaves the stream
            // where C++ does, which is why it is not routed through `new_attribute`. A
            // failing non-required chlist is removed again, `channels` itself is kept.
            if type_name == "chlist" {
                let existing = header
                    .map
                    .get_mut(&name)
                    .and_then(|a| a.as_any_mut().downcast_mut::<ChannelListAttribute>());
                let rejected = if let Some(cl) = existing {
                    cl.value.read_into(i, size)?
                } else {
                    let mut list = ChannelList::new();
                    let rejected = list.read_into(i, size)?;
                    if rejected.is_none() {
                        header.insert_boxed_on_read(&name, Box::new(ChannelListAttribute::new(list)))?;
                    }
                    rejected
                };
                if let Some(e) = rejected {
                    log::warn!("dropping header attribute '{name}' (type 'chlist'): {e}");
                    if name != "channels" {
                        forget(&mut header);
                    }
                }
                continue;
            }

            // --- duplicate name/type/version/chunkCount: first one wins
            // — C++ `parse_header.c` skips the second occurrence of
            // these four rather than overwriting (everything else keeps
            // last-wins, matching `Header::insert`'s replace semantics) ----
            if matches!(name.to_str(), Some("name" | "type" | "version" | "chunkCount"))
                && header.map.contains_key(&name)
            {
                log::warn!("dropping duplicate required attribute '{name}' (first occurrence wins)");
                i.seekg(attr_start + size as u64)?;
                continue;
            }

            // Construct + read the attribute via the registry (unknown →
            // opaque). `version_word` is threaded through for attributes that
            // care about it.
            match new_attribute(type_name.as_bytes(), i, size, version_word) {
                Ok(attr) => {
                    // Re-synchronise on the *declared* size, not
                    // how far the type's own reader advanced the stream
                    // (C++ `extract_attr_*` is always bounded by `attrsz`,
                    // `sequential_skip`s any remainder).
                    //
                    // A FIXED-width type (int, float, box2i, tiledesc,
                    // chromaticities, …) has exactly one correct byte count;
                    // C++'s `extract_attr_32bit`/`_64bit`/`check_populate_*`
                    // reject any other declared size outright
                    // (`EXR_ERR_ATTR_SIZE_MISMATCH`), so a mismatch here drops
                    // the whole attribute rather than keeping a value read
                    // from a stream position that does not match what was
                    // declared (a size-12/size-5 `tiledesc`, and the
                    // same gap for every other fixed-size type).
                    //
                    // A VARIABLE-length type (`string`, `chlist`,
                    // `floatvector`, `stringvector`, `preview`, an unknown/
                    // opaque type, …) may legitimately consume fewer bytes
                    // than declared (trailing padding, e.g. a padded
                    // `channels`); the value is kept and the padding is
                    // skipped. Consuming *more* than declared means the
                    // reader ran past the boundary (e.g. an unterminated
                    // name) — treat that the same as a hard read failure.
                    let consumed = i.tellg()?.saturating_sub(attr_start);
                    let declared = size as u64;
                    if let Some(&fixed_size) = FIXED_ATTR_SIZES.get(type_name.to_str().unwrap_or("")) {
                        if consumed != declared || declared != fixed_size as u64 {
                            log::warn!(
                                "dropping header attribute '{name}' (type '{type_name}'): \
                                 fixed-size type expects {fixed_size} bytes, attribute declared {declared}"
                            );
                            forget(&mut header);
                            i.seekg(attr_start + declared)?;
                            continue;
                        }
                        header.insert_boxed_on_read(&name, attr)?;
                    } else if consumed > declared {
                        log::warn!(
                            "dropping header attribute '{name}' (type '{type_name}'): \
                             consumed {consumed} bytes past its declared size {declared}"
                        );
                        forget(&mut header);
                        i.seekg(attr_start + declared)?;
                    } else {
                        if consumed < declared {
                            i.seekg(attr_start + declared)?;
                        }
                        header.insert_boxed_on_read(&name, attr)?;
                    }
                }
                Err(e) => {
                    // Non-strict: drop the malformed attribute and
                    // resynchronise to the declared size (C++
                    // `parse_header.c:2848-2852` `pull_attr` failure ->
                    // `if (!strict) { skip; continue; }`). This is what
                    // recovers, e.g., an out-of-range `envmap`/
                    // `deepImageState` byte or a duplicate/
                    // invalid `channels` entry without losing the
                    // rest of the header.
                    log::warn!("dropping header attribute '{name}' (type '{type_name}'): {e}");
                    // `idmanifest` is opaque to the core parser but the IlmImf layer then
                    // validates it and throws, which fails the whole open (oracle-verified).
                    if type_name == "idmanifest" {
                        return Err(e);
                    }
                    forget(&mut header);
                    i.seekg(attr_start + size as u64)?;
                }
            }
        }

        header.reads_nothing = attr_count == 0;
        Ok(header)
    }
}

/// Required-attribute name -> (expected wire type name, expected declared
/// size in bytes, or `None` for a variable-length attribute) — the dispatch
/// table C++ 3.4 core `check_req_attr` (`OpenEXRCore/parse_header.c:1966`)
/// uses to route each by-name-required attribute to its own
/// `check_populate_*` validator.
///
/// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L1966>
static REQUIRED_ATTR_TYPES: std::sync::LazyLock<HashMap<&'static str, (&'static str, Option<usize>)>> =
    std::sync::LazyLock::new(|| {
        HashMap::from([
            ("channels", ("chlist", None)),
            ("compression", ("compression", Some(1))),
            ("dataWindow", ("box2i", Some(16))),
            ("displayWindow", ("box2i", Some(16))),
            ("lineOrder", ("lineOrder", Some(1))),
            ("pixelAspectRatio", ("float", Some(4))),
            ("screenWindowCenter", ("v2f", Some(8))),
            ("screenWindowWidth", ("float", Some(4))),
            ("tiles", ("tiledesc", Some(9))),
            ("name", ("string", None)),
            ("type", ("string", None)),
            ("version", ("int", Some(4))),
            ("chunkCount", ("int", Some(4))),
        ])
    });

/// Required attributes whose `check_populate_*` validator `sequential_skip`s
/// the declared size when it is wrong (`parse_header.c` 1314, 1397, 1485,
/// 1568, 1855, 1928). The other fixed-size required ones (`compression`,
/// `dataWindow`, `displayWindow`, `lineOrder`) go through `extract_attr_*`,
/// which returns without consuming anything.
const SKIPS_BAD_SIZE: [&str; 6] = [
    "pixelAspectRatio",
    "screenWindowCenter",
    "screenWindowWidth",
    "tiles",
    "version",
    "chunkCount",
];

/// True when C++ 3.4 core drops an attribute over its declared `size` having
/// consumed nothing and skipped nothing, so the header parse resumes right
/// after the size field (`remaining` = bytes left in the stream, if known).
///
/// * a fixed-size type (see [`FIXED_ATTR_SIZES`]) declared with any other size
///   (`extract_attr_32bit`/`_64bit`/`_uint8`, `extract_attr_tiledesc`:
///   `parse_header.c` 429, 467, 504, 712), except the [`SKIPS_BAD_SIZE`]
///   required names;
/// * a `check_bad_attrsz` type (`string`, `stringvector`, `floatvector`,
///   opaque/unknown types, and the required `name`/`type`) declared negative
///   or larger than what remains, or a `floatvector` not a multiple of 4
///   (`parse_header.c` 265-311). `chlist` is not one of them: it skips.
///
/// Types with their own preview rule and a wrong-typed required attribute
/// (which C++ skips) return `false`. The empty type name is handled by the
/// caller's own branch.
///
/// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L265-L311>
fn bad_size_resync(name: &Name, type_name: &Name, size: i32, remaining: Option<u64>) -> bool {
    let bad_var = || size < 0 || remaining.is_some_and(|r| size as u64 > r);
    if type_name.is_empty() {
        return false;
    }
    // A non-UTF-8 type name is an unregistered one: no fixed size, no required name
    // expects it, so it falls to the generic `check_bad_attrsz` rule below.
    let tn = type_name.to_str().unwrap_or("");
    let nm = name.to_str().unwrap_or("");
    if matches!(nm, "name" | "type") {
        return bad_var();
    }
    if let Some(&(req_type, _)) = REQUIRED_ATTR_TYPES.get(nm)
        && tn != req_type
    {
        return false;
    }
    if let Some(&fixed) = FIXED_ATTR_SIZES.get(tn) {
        return size != fixed as i32 && !SKIPS_BAD_SIZE.contains(&nm);
    }
    match tn {
        "chlist" | "preview" => false,
        "floatvector" => bad_var() || size % 4 != 0,
        _ => bad_var(),
    }
}

/// For the value types whose C++ parser can fail PART-WAY through a well-sized value, returns
/// how many value bytes C++ has consumed when it gives up (`None` = the value is well formed;
/// the stream is left at the value's start either way). The non-strict header loop resumes
/// parsing exactly there, without skipping the rest (`parse_header.c`):
///
/// * `preview`: `< 8` bytes fails at once; otherwise the `w`,`h` words (8 bytes) are consumed
///   and the declared size must equal `8 + 4*w*h` (32-bit wrapping product), non-zero and
///   below the file size (`extract_attr_preview`, 855-925);
/// * `stringvector`: each length word is consumed, then it must be `0..=bytes left`
///   (`extract_attr_string_vector`, 600-700);
/// * `bytes`: `< 4` fails at once; the hint-length word is consumed and must fit the value
///   (`extract_attr_bytes`, 759-815).
///
/// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L855-L925>
fn value_shape_fail(i: &mut dyn IStream, type_name: &Name, size: i32) -> Result<Option<u64>> {
    let start = i.tellg()?;
    let size = i64::from(size);
    let used = match type_name.to_str().unwrap_or("") {
        "preview" if size < 8 => Some(0),
        "preview" => {
            let w = xdr::read_u32(i)?;
            let h = xdr::read_u32(i)?;
            let bytes = u64::from(4u32.wrapping_mul(w).wrapping_mul(h));
            let too_big = i.size().is_some_and(|f| f > 0 && bytes >= f);
            (size as u64 != 8 + bytes || bytes == 0 || too_big).then_some(8)
        }
        "stringvector" => {
            let mut pulled = 0i64;
            let mut bad = None;
            while pulled < size {
                let n = i64::from(xdr::read_i32(i)?);
                pulled += 4;
                if n < 0 || n > size - pulled {
                    bad = Some(pulled as u64);
                    break;
                }
                i.seekg(start + (pulled + n) as u64)?;
                pulled += n;
            }
            bad
        }
        "bytes" if size < 4 => Some(0),
        "bytes" => {
            let hint = xdr::read_u32(i)? as i32;
            (hint < 0 || i64::from(hint) > size - 4).then_some(4)
        }
        _ => None,
    };
    i.seekg(start)?;
    Ok(used)
}

/// Wire type name -> exact on-disk byte width, for every *fixed-size*
/// registered attribute type. Absent from this table means
/// "variable-length" (string, stringvector, floatvector, chlist, preview, an
/// id manifest, or an unknown/opaque type) — those may legitimately consume
/// fewer bytes than declared (trailing padding). A fixed-size type's declared
/// size and consumed bytes must both equal this width, mirroring C++ 3.4 core
/// `extract_attr_32bit`/`extract_attr_64bit`/`check_populate_*`, which reject
/// any other declared size outright rather than reading past or short of it.
static FIXED_ATTR_SIZES: std::sync::LazyLock<HashMap<&'static str, usize>> =
    std::sync::LazyLock::new(|| {
        HashMap::from([
            ("int", 4),
            ("float", 4),
            ("double", 8),
            ("box2i", 16),
            ("box2f", 16),
            ("v2i", 8),
            ("v2f", 8),
            ("v2d", 16),
            ("v3i", 12),
            ("v3f", 12),
            ("v3d", 24),
            ("m33f", 36),
            ("m33d", 72),
            ("m44f", 64),
            ("m44d", 128),
            ("compression", 1),
            ("lineOrder", 1),
            ("envmap", 1),
            ("deepImageState", 1),
            ("tiledesc", 9),
            ("chromaticities", 32),
            ("keycode", 28),
            ("timecode", 8),
            ("rational", 8),
        ])
    });

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Header {
    /// Deep clone — mirrors the C++ copy constructor, which `insert`s a `copy()`
    /// of every attribute.
    fn clone(&self) -> Self {
        let mut map = BTreeMap::new();
        for (name, attr) in self.map.iter() {
            map.insert(name.clone(), attr.clone_attr());
        }
        Header {
            map,
            zip_compression_level: self.zip_compression_level,
            dwa_compression_level: self.dwa_compression_level,
            reads_nothing: self.reads_nothing,
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers — ImfPartType.cpp predicates + sanityCheckDisplayWindow
// ---------------------------------------------------------------------------

/// `static const std::string SCANLINEIMAGE = "scanlineimage";`
pub const SCANLINE_IMAGE: &str = "scanlineimage";
/// `static const std::string TILEDIMAGE = "tiledimage";`
pub const TILED_IMAGE: &str = "tiledimage";
/// `static const std::string DEEPSCANLINE = "deepscanline";`
pub const DEEP_SCANLINE: &str = "deepscanline";
/// `static const std::string DEEPTILE = "deeptile";`
pub const DEEP_TILE: &str = "deeptile";

/// `bool isImage(const string&)` — scanline or tiled (non-deep) image.
pub fn is_image(name: &str) -> bool {
    name == SCANLINE_IMAGE || name == TILED_IMAGE
}

/// `bool isTiled(const string&)` — tiled or deep-tiled.
pub fn is_tiled_type(name: &str) -> bool {
    name == TILED_IMAGE || name == DEEP_TILE
}

/// `bool isDeepData(const string&)` — deep tile or deep scanline.
pub fn is_deep_data(name: &str) -> bool {
    name == DEEP_TILE || name == DEEP_SCANLINE
}

/// `bool isSupportedType(const string&)` — any of the four known part types.
pub fn is_supported_type(name: &str) -> bool {
    name == SCANLINE_IMAGE
        || name == TILED_IMAGE
        || name == DEEP_SCANLINE
        || name == DEEP_TILE
}

/// Port of the anonymous-namespace `sanityCheckDisplayWindow(width, height)`.
fn sanity_check_display_window(width: i32, height: i32) -> Result<()> {
    if width < 1 || height < 1 {
        return Err(ExrError::Arg("Invalid display window in image header.".into()));
    }
    Ok(())
}

// `string_from_name` (the UTF-8-checked name reader) was removed —
// both the attribute name and the attribute type name are read as raw
// `Name` bytes now (see `read_attributes_from`), matching C++'s
// encoding-agnostic `read_text`. The `checkIsNullTerminated` length guard is
// still enforced upstream by `xdr::read_string`'s `max_len` cap.

/// Whether a channel pixel type is one of the three predefined values
/// (`UINT`, `HALF`, `FLOAT`) — the check `sanityCheck` performs per channel.
fn is_valid_pixel_type(t: crate::attr::PixelType) -> bool {
    matches!(
        t,
        crate::attr::PixelType::Uint | crate::attr::PixelType::Half | crate::attr::PixelType::Float
    )
}

// ---------------------------------------------------------------------------
// Tests
//
// These exercise the header port against the C++ wire format: default-attribute
// seeding, byte-exact magic + version word, attribute round-trip through
// MemOStream -> MemIStream, sanityCheck accept/reject, and the long-names
// version-flag computation.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, OpaqueAttribute, PixelType};
    use crate::io::std_io::MemIStream;

    /// Compression levels are stored as given and resolved at the codec seam the way
    /// OpenEXRCore's setters do (`part.c:568-577` zip `-1..=9`, `:615-624` dwa
    /// `0..=65504*100000`, NaN rejected); a rejected level means the process
    /// default (`ImfCompressor.cpp:55-56` ignores the error).
    #[test]
    fn codec_levels_follow_the_core_range_rules() {
        let (zd, dd) = (Header::default_zip_compression_level(), Header::default_dwa_compression_level());
        let zip = |l: i32| {
            let mut h = Header::new();
            h.set_zip_level(l);
            (h.zip_level(), h.codec_zip_level())
        };
        assert_eq!(zip(-1), (-1, -1));
        assert_eq!(zip(0), (0, 0));
        assert_eq!(zip(9), (9, 9));
        assert_eq!(zip(10), (10, zd));
        assert_eq!(zip(-2), (-2, zd));
        let dwa = |l: f32| {
            let mut h = Header::new();
            h.set_dwa_level(l);
            (h.dwa_level(), h.codec_dwa_level())
        };
        let max = 65504.0_f32 * 100000.0_f32;
        assert_eq!(dwa(0.0), (0.0, 0.0));
        assert_eq!(dwa(200.0), (200.0, 200.0));
        assert_eq!(dwa(2000.0), (2000.0, 2000.0));
        assert_eq!(dwa(max), (max, max));
        assert_eq!(dwa(-0.5).1, dd);
        assert_eq!(dwa(max * 2.0).1, dd);
        assert!(dwa(f32::NAN).0.is_nan());
        assert_eq!(dwa(f32::NAN).1, dd);
        assert_eq!(dwa(f32::INFINITY).1, dd);
    }

    /// The four magic bytes on disk, little-endian: `76 2F 31 01`.
    const MAGIC_LE: [u8; 4] = [0x76, 0x2F, 0x31, 0x01];

    /// Helper: a Box2i from min/max coordinate pairs.
    fn box2i(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Box2i {
        Box2i {
            min: Vec2 { x: min_x, y: min_y },
            max: Vec2 { x: max_x, y: max_y },
        }
    }

    /// An attribute name with an interior NUL is truncated at insert
    /// time, matching C++ `Name::operator=`, instead of writing an
    /// unreadable header.
    #[test]
    fn insert_truncates_name_at_interior_nul() {
        let mut h = Header::new();
        h.insert("a\0b", IntAttribute::new(1)).unwrap();
        assert!(h.get("a").is_ok());
        // A lookup key is normalized identically to an insert key
        // (C++ builds a temporary `Name` for `map::find(const char*)` too),
        // so "a\0b" is an alias for "a", not a miss.
        assert!(h.get("a\0b").is_ok());
    }

    /// An attribute name over 255 bytes is truncated to `Name::MAX_LENGTH`.
    #[test]
    fn insert_truncates_overlong_name() {
        let mut h = Header::new();
        let long = "x".repeat(300);
        h.insert(&long, IntAttribute::new(1)).unwrap();
        let truncated = "x".repeat(NAME_MAX_LENGTH);
        assert!(h.get(&truncated).is_ok());
    }

    /// A non-UTF-8 attribute name (C++ `Name`/`parse_header.c` never
    /// checks encoding) round-trips byte-exact through the real wire
    /// encode/decode (`write_attributes_to` / `read_attributes_from`), not
    /// just through the in-memory map. Oracle cross-read coverage (a real
    /// C++-written file with such a name) is tracked as follow-up work.
    #[test]
    fn non_utf8_attribute_name_round_trips_through_the_wire() {
        let raw_name: &[u8] = &[0x5a, 0xe9]; // "Z\xE9" (invalid UTF-8)
        let mut h = Header::new();
        h.insert_boxed(Name::new(raw_name), Box::new(IntAttribute::new(42)))
            .unwrap();

        let mut o = MemOStream::new();
        h.write_attributes_to(&mut o).unwrap();
        let bytes = o.into_data();

        let mut i = MemIStream::new(bytes);
        let back = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        let (_, attr) = back
            .iter()
            .find(|(k, _)| k.as_bytes() == raw_name)
            .expect("non-UTF-8 attribute name must survive the wire round trip");
        let attr: &IntAttribute = attr
            .as_any()
            .downcast_ref()
            .expect("attribute must still be an IntAttribute");
        assert_eq!(attr.value, 42);
    }

    /// `setPreviewImage`/`previewImage`/`hasPreviewImage`
    /// round-trip through the "preview" attribute.
    #[test]
    fn preview_image_accessors_round_trip() {
        let mut h = Header::new();
        assert!(!h.has_preview_image());
        h.set_preview_image(PreviewImage::new(2, 1, None).unwrap()).unwrap();
        assert!(h.has_preview_image());
        assert_eq!(h.preview_image().unwrap().pixels().len(), 2);
        h.preview_image_mut().unwrap();
    }

    /// `readsNothing` is false by default and true only
    /// after parsing an immediately-empty header list.
    #[test]
    fn reads_nothing_reflects_the_parsed_attribute_count() {
        let h = Header::new();
        assert!(!h.reads_nothing());

        let mut o = MemOStream::new();
        write_header_terminator(&mut o); // an immediately-empty header list
        let mut i = MemIStream::new(o.into_data());
        let empty = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(empty.reads_nothing());

        let mut o2 = MemOStream::new();
        write_raw_attr(&mut o2, "customInt", "int", 4, &7i32.to_le_bytes());
        write_header_terminator(&mut o2);
        let mut i2 = MemIStream::new(o2.into_data());
        let nonempty = Header::read_attributes_from(&mut i2, version::EXR_VERSION as i32).unwrap();
        assert!(!nonempty.reads_nothing());
    }

    #[test]
    fn default_header_seeds_required_attributes() {
        let h = Header::new();
        assert_eq!(h.display_window().unwrap(), box2i(0, 0, 63, 63));
        assert_eq!(h.data_window().unwrap(), box2i(0, 0, 63, 63));
        assert_eq!(h.pixel_aspect_ratio().unwrap(), 1.0);
        assert_eq!(h.screen_window_center().unwrap(), V2f { x: 0.0, y: 0.0 });
        assert_eq!(h.screen_window_width().unwrap(), 1.0);
        assert_eq!(h.line_order().unwrap(), LineOrder::IncreasingY);
        assert_eq!(h.compression().unwrap(), Compression::Zip);
        assert!(h.channels().unwrap().is_empty());
        for n in [
            "displayWindow", "dataWindow", "pixelAspectRatio", "screenWindowCenter",
            "screenWindowWidth", "lineOrder", "compression", "channels",
        ] {
            assert!(h.contains(n), "missing required attribute {n}");
        }
    }

    #[test]
    fn invalid_pixel_aspect_ratio_rejected() {
        assert!(Header::with_dimensions(
            64, 64, 0.0, V2f { x: 0.0, y: 0.0 }, 1.0,
            LineOrder::IncreasingY, Compression::Zip,
        )
        .is_err());
    }

    #[test]
    fn zero_dimensions_rejected() {
        assert!(Header::with_dimensions(
            0, 64, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0,
            LineOrder::IncreasingY, Compression::Zip,
        )
        .is_err());
    }

    #[test]
    fn insert_rejects_empty_name() {
        let mut h = Header::new();
        assert!(h.insert("", IntAttribute::new(1)).is_err());
    }

    #[test]
    fn insert_type_mismatch_rejected() {
        let mut h = Header::new();
        h.insert("custom", IntAttribute::new(1)).unwrap();
        assert!(h.insert("custom", FloatAttribute::new(1.0)).is_err());
        h.insert("custom", IntAttribute::new(42)).unwrap();
        assert_eq!(h.typed_attribute::<i32>("custom").unwrap().value, 42);
    }

    #[test]
    fn write_to_emits_exact_magic_and_version_word() {
        let h = Header::new();
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let bytes = o.into_data();
        assert_eq!(&bytes[0..4], &MAGIC_LE);
        let version_word = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version_word as u32, version::EXR_VERSION);
        assert_eq!(version::get_version(version_word as u32), version::EXR_VERSION);
        assert_eq!(version::get_flags(version_word as u32), 0);
    }

    #[test]
    fn write_to_sets_tiled_flag_with_tile_description() {
        let mut h = Header::new();
        h.set_tile_description(TileDescription::default()).unwrap();
        let v = h.version_field(0);
        assert!(version::is_tiled(v));
        assert!(!version::is_non_image(v));
    }

    #[test]
    fn round_trip_preserves_all_values() {
        let mut h = Header::with_dimensions(
            128, 96, 2.0, V2f { x: 0.5, y: -0.25 }, 3.0,
            LineOrder::DecreasingY, Compression::Piz,
        )
        .unwrap();
        {
            let ch = h.channels_mut().unwrap();
            ch.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            ch.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
            ch.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        }
        h.insert("customInt", IntAttribute::new(-7)).unwrap();
        h.insert("customFloat", FloatAttribute::new(1.5)).unwrap();
        h.insert("customStr", StringAttribute::new(ExrString::from("hello"))).unwrap();

        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let mut i = MemIStream::new(o.into_data());
        let (h2, version_word) = Header::read_from(&mut i).unwrap();

        assert_eq!(version_word as u32, version::EXR_VERSION);
        assert_eq!(h2.display_window().unwrap(), box2i(0, 0, 127, 95));
        assert_eq!(h2.data_window().unwrap(), box2i(0, 0, 127, 95));
        assert_eq!(h2.pixel_aspect_ratio().unwrap(), 2.0);
        assert_eq!(h2.screen_window_center().unwrap(), V2f { x: 0.5, y: -0.25 });
        assert_eq!(h2.screen_window_width().unwrap(), 3.0);
        assert_eq!(h2.line_order().unwrap(), LineOrder::DecreasingY);
        assert_eq!(h2.compression().unwrap(), Compression::Piz);
        let ch2 = h2.channels().unwrap();
        assert_eq!(ch2.find_channel("R").unwrap().type_, PixelType::Half);
        assert_eq!(ch2.find_channel("Z").unwrap().type_, PixelType::Float);
        assert_eq!(h2.typed_attribute::<i32>("customInt").unwrap().value, -7);
        assert_eq!(h2.typed_attribute::<f32>("customFloat").unwrap().value, 1.5);
        assert_eq!(
            h2.typed_attribute::<ExrString>("customStr").unwrap().value,
            ExrString::from("hello")
        );
    }

    #[test]
    fn round_trip_is_byte_stable() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("owner", StringAttribute::new(ExrString::from("acme"))).unwrap();

        let mut o1 = MemOStream::new();
        h.write_to(&mut o1, 0).unwrap();
        let first = o1.into_data();
        let mut i = MemIStream::new(first.clone());
        let (h2, _) = Header::read_from(&mut i).unwrap();
        let mut o2 = MemOStream::new();
        h2.write_to(&mut o2, 0).unwrap();
        assert_eq!(first, o2.into_data());
    }

    #[test]
    fn unknown_attribute_type_preserved_as_opaque() {
        let mut h = Header::new();
        h.insert_boxed(
            "weird",
            Box::new(OpaqueAttribute::with_data("madeUpType", vec![1u8, 2, 3, 4, 5])),
        )
        .unwrap();
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let mut i = MemIStream::new(o.into_data());
        let (h2, _) = Header::read_from(&mut i).unwrap();
        assert_eq!(h2.get("weird").unwrap().type_name(), b"madeUpType");
    }

    #[test]
    fn read_checked_rejects_a_window_read_from_accepts() {
        // `read_from` is the 1:1 `Header::readFrom` port and validates
        // nothing; `read_checked` is the entry point for a consumer that parses
        // a header without opening a reader (the `exr-rs info` CLI). A crafted
        // dataWindow used to reach `info`'s arithmetic unchecked.
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        *h.data_window_mut().unwrap() = box2i(0, 0, INT_MAX_DIV_2, 10);
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let bytes = o.into_data();

        let mut i = MemIStream::new(bytes.clone());
        assert!(Header::read_from(&mut i).is_ok(), "read_from stays 1:1 (no check)");

        let mut i = MemIStream::new(bytes);
        // `read_checked` routes `sanity_check` through
        // `sanity_check_at_open`, which reclassifies a non-"missing
        // attribute" `Arg` to `Input` (`ImfContext.cpp:127-138`, verified) —
        // "Invalid data window in image header." is not a missing attribute.
        assert!(matches!(
            Header::read_checked(&mut i),
            Err(ExrError::Input(_))
        ));
    }

    #[test]
    fn read_checked_rejects_a_multipart_file() {
        // Only the FIRST part header follows the version word, so returning it
        // would misdescribe the file — MultiPartInputFile reads every part.
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.set_name("part0").unwrap();
        h.set_type(SCANLINE_IMAGE).unwrap();
        let mut o = MemOStream::new();
        h.write_to(&mut o, version::MULTI_PART_FILE_FLAG).unwrap();
        let mut i = MemIStream::new(o.into_data());
        assert!(matches!(
            Header::read_checked(&mut i),
            Err(ExrError::Arg(_))
        ));
    }

    /// `read_checked` must default a missing required attribute the
    /// way the reader constructors do (via `validate_for_read`), not error.
    #[test]
    fn read_checked_defaults_missing_screen_window_center() {
        let mut o = MemOStream::new();
        // No `screenWindowCenter` at all — declared size includes only
        // `channels`, `type` is omitted (single-part, auto-added).
        write_raw_attr(&mut o, "displayWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(&mut o, "dataWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(&mut o, "pixelAspectRatio", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(&mut o, "screenWindowWidth", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(&mut o, "lineOrder", "lineOrder", 1, &[0u8]);
        write_raw_attr(&mut o, "compression", "compression", 1, &[0u8]);
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut buf = MemOStream::new();
        cl.write_value(&mut buf, 0).unwrap();
        let bytes = buf.into_data();
        write_raw_attr(&mut o, "channels", "chlist", bytes.len() as i32, &bytes);
        write_header_terminator(&mut o);
        let attrs = o.into_data();

        let mut file = MemOStream::new();
        xdr::write_i32(&mut file, version::MAGIC as i32).unwrap();
        xdr::write_i32(&mut file, version::EXR_VERSION as i32).unwrap();
        file.write(&attrs).unwrap();

        let mut i = MemIStream::new(file.into_data());
        let (header, _) = Header::read_checked(&mut i).unwrap();
        assert_eq!(header.screen_window_center().unwrap(), V2f { x: 0.0, y: 0.0 });
        assert_eq!(header.type_().unwrap(), SCANLINE_IMAGE);
    }

    #[test]
    fn read_checked_accepts_a_valid_header() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let mut i = MemIStream::new(o.into_data());
        let (back, _) = Header::read_checked(&mut i).unwrap();
        assert_eq!(back.data_window().unwrap(), h.data_window().unwrap());
    }

    /// Write one raw header attribute record: name, type name, an explicit
    /// declared size, then `value` verbatim (which need not match `size` —
    /// tests exercise the mismatch on purpose).
    fn write_raw_attr(o: &mut MemOStream, name: &str, type_name: &str, size: i32, value: &[u8]) {
        xdr::write_string(o, name).unwrap();
        xdr::write_string(o, type_name).unwrap();
        xdr::write_i32(o, size).unwrap();
        o.write(value).unwrap();
    }

    /// Build a minimal but otherwise-valid header byte stream (the 8 required
    /// attributes with sane values) so a test can splice in one deliberately
    /// malformed attribute and see it dropped rather than derailing the rest.
    fn write_minimal_required_attrs(o: &mut MemOStream) {
        write_raw_attr(o, "displayWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(o, "dataWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(o, "pixelAspectRatio", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(o, "screenWindowCenter", "v2f", 8, &[0u8; 8]);
        write_raw_attr(o, "screenWindowWidth", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(o, "lineOrder", "lineOrder", 1, &[0u8]);
        write_raw_attr(o, "compression", "compression", 1, &[0u8]);
    }

    /// Terminate a raw header attribute list (the empty-name marker).
    fn write_header_terminator(o: &mut MemOStream) {
        xdr::write_string(o, "").unwrap();
    }

    /// A required attribute stored under the wrong wire type (a
    /// `double` `pixelAspectRatio` instead of `float`) is dropped, not
    /// inserted with the wrong type, so the header still parses and the rest
    /// of the attribute list is not desynchronised.
    #[test]
    fn read_attributes_drops_required_attr_of_wrong_type() {
        let mut o = MemOStream::new();
        write_raw_attr(&mut o, "displayWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(&mut o, "dataWindow", "box2i", 16, &[0u8; 16]);
        // pixelAspectRatio typed `double` (8 bytes) instead of `float`.
        write_raw_attr(&mut o, "pixelAspectRatio", "double", 8, &1.0f64.to_le_bytes());
        write_raw_attr(&mut o, "screenWindowCenter", "v2f", 8, &[0u8; 8]);
        write_raw_attr(&mut o, "screenWindowWidth", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(&mut o, "lineOrder", "lineOrder", 1, &[0u8]);
        write_raw_attr(&mut o, "compression", "compression", 1, &[0u8]);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(!header.contains("pixelAspectRatio"), "the mistyped attribute must be dropped");
    }

    /// A required attribute whose declared size is padded past its
    /// fixed width (a `float` `pixelAspectRatio` declared as 8 bytes) is
    /// dropped and the stream re-synchronises on the *declared* size, not the
    /// 4 bytes the `float` reader actually consumed.
    #[test]
    fn read_attributes_drops_required_attr_with_padded_size() {
        let mut o = MemOStream::new();
        write_raw_attr(&mut o, "displayWindow", "box2i", 16, &[0u8; 16]);
        write_raw_attr(&mut o, "dataWindow", "box2i", 16, &[0u8; 16]);
        // pixelAspectRatio: correct type, but declared size 8 (4 real bytes +
        // 4 padding) instead of 4.
        let mut par_bytes = 1.0f32.to_le_bytes().to_vec();
        par_bytes.extend_from_slice(&[0u8; 4]);
        write_raw_attr(&mut o, "pixelAspectRatio", "float", 8, &par_bytes);
        write_raw_attr(&mut o, "screenWindowCenter", "v2f", 8, &[0u8; 8]);
        write_raw_attr(&mut o, "screenWindowWidth", "float", 4, &1.0f32.to_le_bytes());
        write_raw_attr(&mut o, "lineOrder", "lineOrder", 1, &[0u8]);
        write_raw_attr(&mut o, "compression", "compression", 1, &[0u8]);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(!header.contains("pixelAspectRatio"));
        // The rest of the header must still parse correctly — the stream
        // resynchronised on the declared 8-byte size, not the 4 bytes read.
        assert!(header.contains("screenWindowCenter"));
        assert!(header.contains("compression"));
    }

    /// A duplicate `channels` attribute accumulates into the existing
    /// list instead of replacing it.
    #[test]
    fn read_attributes_merges_duplicate_channels() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        // First `channels`: just "R".
        let mut cl1 = ChannelList::new();
        cl1.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut buf1 = MemOStream::new();
        cl1.write_value(&mut buf1, 0).unwrap();
        let bytes1 = buf1.into_data();
        write_raw_attr(&mut o, "channels", "chlist", bytes1.len() as i32, &bytes1);
        // Second `channels`: just "G".
        let mut cl2 = ChannelList::new();
        cl2.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut buf2 = MemOStream::new();
        cl2.write_value(&mut buf2, 0).unwrap();
        let bytes2 = buf2.into_data();
        write_raw_attr(&mut o, "channels", "chlist", bytes2.len() as i32, &bytes2);
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        let channels = header.channels().unwrap();
        assert!(channels.find_channel("R").is_some());
        assert!(channels.find_channel("G").is_some());
    }

    /// The second of a duplicate `name` / `type` / `version` /
    /// `chunkCount` attribute is dropped; the first occurrence wins.
    #[test]
    fn read_attributes_first_wins_for_duplicate_name() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        write_raw_attr(&mut o, "name", "string", 1, b"a");
        write_raw_attr(&mut o, "name", "string", 1, b"b");
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert_eq!(header.name().unwrap(), "a");
    }

    /// A padded `channels` attribute (the terminator followed by
    /// trailing zero bytes still inside the declared size) must not desync
    /// the rest of the header: the padding is skipped via the declared size,
    /// not misread as the next attribute's name.
    #[test]
    fn read_attributes_channels_padding_does_not_desync_header() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        let mut cl = ChannelList::new();
        cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut buf = MemOStream::new();
        cl.write_value(&mut buf, 0).unwrap();
        let mut bytes = buf.into_data();
        let real_len = bytes.len() as i32;
        bytes.extend_from_slice(&[0u8; 4]); // declared size includes 4 pad bytes
        write_raw_attr(&mut o, "channels", "chlist", real_len + 4, &bytes);
        write_raw_attr(&mut o, "customInt", "int", 4, &7i32.to_le_bytes());
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(header.channels().unwrap().find_channel("R").is_some());
        assert!(header.contains("customInt"), "trailing attribute must still parse");
    }

    /// A `tiledesc` attribute whose declared size is not the fixed 9
    /// bytes is dropped (and the header still parses past it), matching C++
    /// 3.4 core `parse_header.c:712-744` (`attrsz != 9` -> error).
    #[test]
    fn read_attributes_drops_tiledesc_with_wrong_size() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        // 9 real bytes + 3 bytes that would otherwise desync into the next
        // attribute's name.
        let mut tiles_bytes = vec![32u8, 0, 0, 0, 32, 0, 0, 0, 0u8];
        tiles_bytes.extend_from_slice(&[0u8; 3]);
        write_raw_attr(&mut o, "tiles", "tiledesc", 12, &tiles_bytes);
        write_raw_attr(&mut o, "customInt", "int", 4, &7i32.to_le_bytes());
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(!header.contains("tiles"));
        assert!(header.contains("customInt"), "trailing attribute must still parse");
    }

    /// A malformed *non-required* attribute (a `chromaticities` with a
    /// wrong declared size) is dropped rather than aborting the whole header
    /// parse, matching C++ 3.4's non-strict `pull_attr` loop. C++ drops it
    /// WITHOUT skipping the declared size (`extract_attr_32bit` returns
    /// before consuming), so parsing resumes inside the value: here its first
    /// zero byte is read as the header terminator and `customInt` is never
    /// seen (oracle: `exrheader` on the same bytes prints no `customInt`).
    #[test]
    fn read_attributes_drops_malformed_optional_attribute() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        write_raw_attr(&mut o, "chromaticities", "chromaticities", 16, &[0u8; 16]);
        write_raw_attr(&mut o, "customInt", "int", 4, &7i32.to_le_bytes());
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(!header.contains("chromaticities"));
        assert!(!header.contains("customInt"), "the value's zero byte ends the header, as in C++");
    }

    /// Parse a raw attribute list (a valid `dataWindow` followed by what `build` writes) the
    /// way [`Header::read_attributes_from`] does.
    fn parse_after_required(build: impl FnOnce(&mut MemOStream)) -> Result<Header> {
        let mut o = MemOStream::new();
        write_raw_attr(&mut o, "dataWindow", "box2i", 16, &[0u8; 16]);
        build(&mut o);
        let mut i = MemIStream::new(o.into_data());
        Header::read_attributes_from(&mut i, version::EXR_VERSION as i32)
    }

    /// A wrong declared size drops the attribute and resumes at the cursor
    /// right after the size field, so the value bytes are parsed as the next
    /// attribute (oracle-verified: `exrheader` lists `XXXXXXXXXXXXXXXXcustomInt`).
    #[test]
    fn bad_fixed_size_resumes_inside_the_value() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "chromaticities", "chromaticities", 16, &[b'X'; 16]);
            write_raw_attr(o, "customInt", "int", 4, &7i32.to_le_bytes());
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("chromaticities"));
        assert!(h.contains("XXXXXXXXXXXXXXXXcustomInt"), "value bytes must be re-parsed as a name");
    }

    /// A negative declared size is the same drop-and-resume, not a hard error
    /// (oracle: `foo int -1` followed by `customInt` yields `customInt`).
    #[test]
    fn negative_size_resumes_after_the_size_field() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "foo", "int", -1, &[]);
            write_raw_attr(o, "customInt", "int", 4, &7i32.to_le_bytes());
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("foo"));
        assert!(h.contains("customInt"));
    }

    /// The `comp_invalid_unknown.exr` condition: a required `displayWindow`
    /// declaring far more bytes than the file has left. C++ reports a size
    /// mismatch without skipping and the value's first zero byte ends the
    /// header, so the file opens; the too-big-for-the-stream guard must not
    /// pre-empt that.
    #[test]
    fn oversized_fixed_attr_does_not_pre_empt_the_size_mismatch() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "displayWindow", "box2i", 32528, &[0u8; 4]);
        })
        .unwrap();
        assert!(h.contains("dataWindow"));
        assert!(!h.contains("displayWindow"));
    }

    /// `check_bad_attrsz` (string/opaque/...) rejects an oversized value
    /// without consuming it, and the header resumes in the value bytes.
    #[test]
    fn oversized_variable_attr_is_dropped_without_skipping() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "foo", "string", 1000, &[0u8; 8]);
        })
        .unwrap();
        assert!(!h.contains("foo"));
        assert!(h.contains("dataWindow"));
    }

    /// Where C++ DOES skip the declared size (`chlist`, the `float`-typed
    /// required attributes), an oversized value runs into EOF, which is fatal
    /// there and here.
    #[test]
    fn oversized_skipped_attr_is_fatal() {
        assert!(parse_after_required(|o| write_raw_attr(o, "channels", "chlist", 1000, &[0u8; 8])).is_err());
        assert!(
            parse_after_required(|o| write_raw_attr(o, "pixelAspectRatio", "float", 1000, &[0u8; 8])).is_err()
        );
    }

    /// The required `float` attributes keep C++'s skip-on-bad-size rule: the
    /// padded value is skipped and the next attribute still parses.
    #[test]
    fn padded_skip_listed_attr_is_skipped() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "pixelAspectRatio", "float", 8, &[0u8; 8]);
            write_raw_attr(o, "customInt", "int", 4, &7i32.to_le_bytes());
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("pixelAspectRatio"));
        assert!(h.contains("customInt"));
    }


    /// The raw bytes of `customInt` (an `int` attribute worth 7), used as the marker that shows
    /// where header parsing resumed after a dropped attribute.
    fn custom_int(o: &mut MemOStream) {
        write_raw_attr(o, "customInt", "int", 4, &7i32.to_le_bytes());
    }

    /// An empty type name is rejected BEFORE the size field is read, and parsing resumes
    /// right after the type's NUL: the size bytes and value become the next attribute (oracle:
    /// `foo`, `""`, then `XYZ int 9` lists `XYZ` and `customInt`).
    #[test]
    fn empty_type_name_resumes_before_the_size_field() {
        let h = parse_after_required(|o| {
            xdr::write_string(o, "foo").unwrap();
            xdr::write_string(o, "").unwrap();
            write_raw_attr(o, "XYZ", "int", 4, &9i32.to_le_bytes());
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("foo"));
        assert!(h.contains("XYZ") && h.contains("customInt"));
    }

    /// A name/type longer than 255 bytes is dropped after `max + 1` bytes and parsing goes on
    /// with the rest of the bytes (oracle: 300 `N` yields an `int` named with the last 44).
    #[test]
    fn overlong_name_resumes_after_max_plus_one_bytes() {
        let h = parse_after_required(|o| {
            o.write(&[b'N'; 300]).unwrap();
            o.write(&[0]).unwrap();
            xdr::write_string(o, "int").unwrap();
            xdr::write_i32(o, 4).unwrap();
            o.write(&1i32.to_le_bytes()).unwrap();
            write_header_terminator(o);
        })
        .unwrap();
        assert!(h.contains(&"N".repeat(44)));
    }

    /// `preview`: `w`,`h` are consumed before the size is checked against `8 + 4*w*h`; a size
    /// under 8 consumes nothing (oracle: `abcdefghijklcustomInt` / `abcdcustomInt`).
    #[test]
    fn preview_size_mismatch_resumes_after_the_dimensions() {
        let wh = [1u8, 0, 0, 0, 1, 0, 0, 0];
        let h = parse_after_required(|o| {
            write_raw_attr(o, "pv", "preview", 20, &[&wh[..], b"abcdefghijkl"].concat());
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("pv") && h.contains("abcdefghijklcustomInt"));

        let h = parse_after_required(|o| {
            write_raw_attr(o, "pv", "preview", 4, b"abcd");
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("pv") && h.contains("abcdcustomInt"));

        let h = parse_after_required(|o| {
            write_raw_attr(o, "pv", "preview", 12, &[&wh[..], b"abcd"].concat());
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(h.contains("pv") && h.contains("customInt"));
    }

    /// `stringvector` / `bytes` with a length word that does not fit fail right after that word
    /// (oracle: both resume at `abcd`, giving `abcdcustomInt`).
    #[test]
    fn bad_inner_length_resumes_after_the_length_word() {
        for ty in ["stringvector", "bytes"] {
            let h = parse_after_required(|o| {
                write_raw_attr(o, "v", ty, 8, &[&100u32.to_le_bytes()[..], b"abcd"].concat());
                custom_int(o);
                write_header_terminator(o);
            })
            .unwrap();
            assert!(!h.contains("v") && h.contains("abcdcustomInt"), "{ty}");
        }
    }

    /// Repeated names (oracle-verified): a second `string` is refused without consuming its
    /// value; a second attribute of another type likewise; the same fixed type with a bad size
    /// fails and takes the earlier value with it.
    #[test]
    fn repeated_attribute_names_follow_attr_list_add() {
        let h = parse_after_required(|o| {
            write_raw_attr(o, "foo", "string", 2, b"hi");
            write_raw_attr(o, "foo", "string", 2, b"yo");
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert_eq!(h.get("foo").unwrap().type_name(), b"string");
        assert!(h.contains("yocustomInt") && !h.contains("customInt"));

        let h = parse_after_required(|o| {
            write_raw_attr(o, "foo", "int", 4, &1i32.to_le_bytes());
            write_raw_attr(o, "foo", "float", 4, &[0u8; 4]);
            custom_int(o);
        })
        .unwrap();
        assert_eq!(h.get("foo").unwrap().type_name(), b"int");
        assert!(!h.contains("customInt"), "the float's zero value ended the header");

        let h = parse_after_required(|o| {
            write_raw_attr(o, "foo", "int", 4, &1i32.to_le_bytes());
            write_raw_attr(o, "foo", "int", 2, &[0u8; 2]);
            custom_int(o);
        })
        .unwrap();
        assert!(!h.contains("foo") && !h.contains("customInt"));
    }

    /// `idmanifest` is opaque to the core parser, but the IlmImf layer throws on a value under
    /// 8 bytes, which fails the whole open (oracle: exrheader refuses the file).
    #[test]
    fn short_idmanifest_fails_the_header() {
        assert!(parse_after_required(|o| write_raw_attr(o, "idm", "idmanifest", 4, b"abcd")).is_err());
    }

    /// One `chlist` entry as it sits on the wire: name, pixel type, pLinear + 3 pad bytes and
    /// both samplings (1); `terminate` appends the empty name that ends the list.
    fn chan(name: &str, ty: i32, terminate: bool) -> Vec<u8> {
        let mut o = MemOStream::new();
        xdr::write_string(&mut o, name).unwrap();
        xdr::write_i32(&mut o, ty).unwrap();
        o.write(&[0; 4]).unwrap();
        xdr::write_i32(&mut o, 1).unwrap();
        xdr::write_i32(&mut o, 1).unwrap();
        if terminate {
            xdr::write_string(&mut o, "").unwrap();
        }
        o.into_data()
    }

    fn chlist(o: &mut MemOStream, name: &str, bytes: &[u8]) {
        write_raw_attr(o, name, "chlist", bytes.len() as i32, bytes);
    }

    /// A repeated `channels` whose list fails part-way KEEPS the channels it had already added
    /// (`extract_attr_chlist` appends in place; oracle: `G` and `R` listed, no `B`).
    #[test]
    fn repeated_channels_keep_what_was_added_before_a_bad_channel() {
        let h = parse_after_required(|o| {
            chlist(o, "channels", &chan("R", 1, true));
            chlist(o, "channels", &[chan("G", 1, false), chan("B", 7, true)].concat());
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        let cl = h.channels().unwrap();
        assert!(cl.find_channel("R").is_some() && cl.find_channel("G").is_some());
        assert!(cl.find_channel("B").is_none());
        assert!(h.contains("customInt"), "parsing resumed at the declared end");
    }

    /// A channel name that runs past the declared size makes C++'s trailing skip a ~4 GiB one
    /// that hits EOF: fatal (oracle: exrheader cannot open the file).
    #[test]
    fn chlist_name_past_the_declared_size_is_fatal() {
        let r = parse_after_required(|o| {
            write_raw_attr(o, "channels", "chlist", 3, &[b"ABCDEFGH\0".as_slice(), &[0; 40]].concat());
        });
        assert!(r.is_err());
    }

    /// A declared size that ends after a channel's name but inside its 16 field bytes stops the
    /// list (`extract_attr_chlist`: `maxlen < 16` -> "Out of data parsing") and skips only the
    /// rest of the budget, so the next attribute is read intact rather than eaten as fields.
    #[test]
    fn chlist_cut_inside_a_channels_fields_skips_only_its_budget() {
        let h = parse_after_required(|o| {
            chlist(o, "foo", &chan("G", 1, true));
            chlist(o, "foo", &chan("B", 1, false)[..10]);
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("foo"), "the failing repeat is removed");
        assert!(h.contains("customInt"), "parsing resumed at the declared end");
    }

    /// The channel loop is bounded by the declared size, not by the terminator: a list whose
    /// size stops right after the last channel is complete (oracle: `G` and `customInt` seen).
    #[test]
    fn chlist_needs_no_terminator_inside_its_declared_size() {
        let h = parse_after_required(|o| {
            chlist(o, "channels", &chan("G", 1, false));
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(h.channels().unwrap().find_channel("G").is_some());
        assert!(h.contains("customInt"));
    }

    /// A repeated non-required `chlist` that fails removes the attribute (C++
    /// `exr_attr_list_remove`), unlike `channels` (oracle: `foo` gone, `customInt` kept).
    #[test]
    fn failing_repeated_optional_chlist_is_removed() {
        let h = parse_after_required(|o| {
            chlist(o, "foo", &chan("G", 1, true));
            chlist(o, "foo", &[chan("G", 1, false), chan("B", 7, true)].concat());
            custom_int(o);
            write_header_terminator(o);
        })
        .unwrap();
        assert!(!h.contains("foo") && h.contains("customInt"));
    }

    /// An out-of-range `envmap` value is dropped instead of being
    /// remapped to `LatLong`, so the header does not misclassify an ordinary
    /// image as an environment map.
    #[test]
    fn read_attributes_drops_out_of_range_envmap() {
        let mut o = MemOStream::new();
        write_minimal_required_attrs(&mut o);
        write_raw_attr(&mut o, "channels", "chlist", 1, &[0u8]);
        write_raw_attr(&mut o, "envmap", "envmap", 1, &[2u8]);
        write_header_terminator(&mut o);

        let mut i = MemIStream::new(o.into_data());
        let header = Header::read_attributes_from(&mut i, version::EXR_VERSION as i32).unwrap();
        assert!(!header.contains("envmap"));
    }

    /// A version word other than 2 must be rejected by `read_from`
    /// (and therefore by `read_checked`), matching C++ 3.4 core
    /// `parse_header.c` (`if (ctxt->version != 2)`).
    #[test]
    fn read_checked_rejects_version_3() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut o = MemOStream::new();
        // magic + version word with version number 3 (base_flags all clear).
        xdr::write_i32(&mut o, version::MAGIC as i32).unwrap();
        xdr::write_i32(&mut o, 3).unwrap();
        h.write_attributes_to(&mut o).unwrap();
        let mut i = MemIStream::new(o.into_data());
        assert!(Header::read_from(&mut i).is_err());
    }

    /// TILED combined with NON_IMAGE (or MULTI_PART) is an invalid
    /// flag combination and must be rejected up front, not silently accepted
    /// by `read_checked` / a header-only passthrough.
    #[test]
    fn read_checked_rejects_tiled_plus_deep_flags() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let mut o = MemOStream::new();
        xdr::write_i32(&mut o, version::MAGIC as i32).unwrap();
        xdr::write_i32(
            &mut o,
            (version::EXR_VERSION | version::TILED_FLAG | version::NON_IMAGE_FLAG) as i32,
        )
        .unwrap();
        h.write_attributes_to(&mut o).unwrap();
        let mut i = MemIStream::new(o.into_data());
        assert!(Header::read_from(&mut i).is_err());
    }

    #[test]
    fn read_from_rejects_bad_magic() {
        let mut i = MemIStream::new(vec![0u8, 0, 0, 0, 2, 0, 0, 0]);
        assert!(Header::read_from(&mut i).is_err());
    }

    /// `sanity_check` must call `getTiledChunkOffsetTableSize` and
    /// reject a tile count that overflows `i32`, matching C++
    /// `ImfHeader.cpp` (`sanityCheck` -> `ImfTiledMisc.cpp:309-360`).

    // The process-global default zip/dwa-compression-level tests
    // (`zip_and_dwa_compression_level_override_and_default`,
    // `insert_dwa_compression_level_attribute_updates_record`) moved to
    // `tests/global_compression_defaults.rs`, and the process-global
    // max-image/max-tile-size tests (`max_image_and_tile_size_round_trip`,
    // `sanity_check_enforces_max_image_width`,
    // `sanity_check_enforces_max_tile_width`) moved to
    // `tests/global_max_sizes.rs` -- both are SEPARATE processes from this
    // `--lib` unit-test binary, so mutating those globals there can no
    // longer race the many OTHER lib tests that read the effective
    // compression level / size limit without taking any lock (a lock only
    // serializes mutators against each other, not against readers you don't
    // control -- see each file's module doc for the full derivation).
    // `header::test_support` and the local `MAX_SIZE_TEST_LOCK`/
    // `MaxSizeResetGuard` were deleted along with them since nothing else
    // used them.

    #[test]
    fn sanity_check_rejects_tile_count_overflow() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        *h.data_window_mut().unwrap() = box2i(0, 0, 65535, 65535);
        h.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 1,
                y_size: 1,
                ..Default::default()
            }),
        )
        .unwrap();
        assert!(h.sanity_check(true, false).is_err());
    }

    #[test]
    fn sanity_check_accepts_valid_scanline_header() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.sanity_check(false, false).unwrap();
    }

    #[test]
    fn sanity_check_rejects_empty_channel_list() {
        let h = Header::new();
        assert!(h.sanity_check(false, false).is_err());
    }

    #[test]
    fn sanity_check_rejects_bad_data_window() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        *h.data_window_mut().unwrap() = box2i(10, 10, 0, 0);
        assert!(h.sanity_check(false, false).is_err());
    }

    #[test]
    fn sanity_check_rejects_oversized_window() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        *h.data_window_mut().unwrap() = box2i(0, 0, INT_MAX_DIV_2, 10);
        assert!(h.sanity_check(false, false).is_err());
    }

    #[test]
    fn sanity_check_tiled_requires_tile_description() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        assert!(h.sanity_check(true, false).is_err());
    }

    /// Tiled files reject ANY x-subsampled channel outright, matching C++
    /// `ImfHeader.cpp:1163-1171` (v3.4.15) -- ported 1:1 at `sanity_check`
    /// above (`if ch.x_sampling != 1` for `is_tiled || is_deep`). No real C++
    /// writer can produce a tiled file with a sub-sampled channel to cross-
    /// check against, so this pins the rejection directly instead of an
    /// oracle round trip.
    #[test]
    fn sanity_check_rejects_tiled_x_subsampled_channel() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
        h.insert("tiles", TileDescriptionAttribute::new(TileDescription::default())).unwrap();
        let err = h.sanity_check(true, false).expect_err("tiled + x_sampling=2 must be rejected");
        assert!(
            err.to_string().contains("x subsampling"),
            "expected an x-subsampling error, got: {err}"
        );
    }

    /// Same as above for y-subsampling (`ImfHeader.cpp:1173-1181`, `if
    /// ch.y_sampling != 1`).
    #[test]
    fn sanity_check_rejects_tiled_y_subsampled_channel() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 2, false)).unwrap();
        h.insert("tiles", TileDescriptionAttribute::new(TileDescription::default())).unwrap();
        let err = h.sanity_check(true, false).expect_err("tiled + y_sampling=2 must be rejected");
        assert!(
            err.to_string().contains("y subsampling"),
            "expected a y-subsampling error, got: {err}"
        );
    }

    /// Odd dimensions: a SCANLINE image whose data-window height
    /// is not a multiple of a channel's `y_sampling` is rejected at
    /// `sanity_check` too -- `ImfHeader.cpp:1244-1254` (v3.4.15, "Number of
    /// pixels per column ... is not a multiple of the y subsampling factor"),
    /// mirrored by the READ-path validator `OpenEXRCore/validation.c:461-475`
    /// (the same modulus check, so a crafted file cannot bypass this by
    /// skipping the writer-only C++ `sanityCheck`). This means HTJ2K's (and
    /// every other codec's) internal geometry guard rejecting an odd
    /// `image_height`/`y_sampling` combination it is ever handed
    /// (`htj2k_rejects_component_geometry_mismatch_odd_height_isolates_first_term`)
    /// is defense-in-depth only: no real OR crafted-but-openable EXR file
    /// (C++'s or ours) can ever reach a codec with this combination, since
    /// both C++ and this crate refuse the file at open time first. Same
    /// resolution as a data-window `min.x` that is not a multiple of
    /// `x_sampling`, which `sanity_check` also rejects at open time.
    #[test]
    fn sanity_check_rejects_scanline_height_not_multiple_of_y_sampling() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("S", Channel::new(PixelType::Half, 1, 2, false)).unwrap();
        // Default data window height (from `Header::new()`) is 1 pixel tall,
        // which trivially satisfies `1 % 2 != 0` too -- force an odd height
        // (15) explicitly so the failure is unambiguously the height/sampling
        // check, not some other default-window degeneracy.
        *h.data_window_mut().unwrap() = box2i(0, 0, 31, 14);
        let err = h.sanity_check(false, false).expect_err("height=15 with y_sampling=2 must be rejected");
        assert!(
            err.to_string().contains("multiple of the y subsampling factor"),
            "expected a y-subsampling-divisibility error, got: {err}"
        );
    }

    /// Same as above for width/x_sampling (`ImfHeader.cpp:1256-1266`).
    #[test]
    fn sanity_check_rejects_scanline_width_not_multiple_of_x_sampling() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("S", Channel::new(PixelType::Half, 2, 1, false)).unwrap();
        *h.data_window_mut().unwrap() = box2i(0, 0, 14, 31);
        let err = h.sanity_check(false, false).expect_err("width=15 with x_sampling=2 must be rejected");
        assert!(
            err.to_string().contains("multiple of the x subsampling factor"),
            "expected an x-subsampling-divisibility error, got: {err}"
        );
    }

    #[test]
    fn sanity_check_multipart_requires_name_and_type() {
        let mut h = Header::new();
        h.channels_mut().unwrap()
            .insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        assert!(h.sanity_check(false, true).is_err());
        h.set_name("part0").unwrap();
        h.set_type(SCANLINE_IMAGE).unwrap();
        h.sanity_check(false, true).unwrap();
    }

    #[test]
    fn long_attribute_name_sets_long_names_flag() {
        let mut h = Header::new();
        let long = "a".repeat(32);
        assert_eq!(long.len(), 32);
        h.insert(&long, IntAttribute::new(1)).unwrap();
        assert!(h.uses_long_names());
        assert!(version::get_flags(h.version_field(0)) & version::LONG_NAMES_FLAG != 0);
        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let bytes = o.into_data();
        let vw = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u32;
        assert!(vw & version::LONG_NAMES_FLAG != 0);
    }

    #[test]
    fn short_names_do_not_set_long_names_flag() {
        let mut h = Header::new();
        let name = "a".repeat(31);
        h.insert(&name, IntAttribute::new(1)).unwrap();
        assert!(!h.uses_long_names());
        assert_eq!(version::get_flags(h.version_field(0)) & version::LONG_NAMES_FLAG, 0);
    }

    #[test]
    fn long_channel_name_sets_long_names_flag() {
        let mut h = Header::new();
        let long = "c".repeat(40);
        h.channels_mut().unwrap()
            .insert(&long, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        assert!(h.uses_long_names());
    }

    #[test]
    fn set_type_deep_sets_version_and_non_image_flag() {
        let mut h = Header::new();
        h.set_type(DEEP_SCANLINE).unwrap();
        assert!(h.has_version());
        assert_eq!(h.version().unwrap(), 1);
        assert!(version::is_non_image(h.version_field(0)));
    }

    #[test]
    fn set_type_rejects_unsupported() {
        let mut h = Header::new();
        assert!(h.set_type("bogusType").is_err());
    }

    #[test]
    fn set_type_error_message_is_readable() {
        // Regression guard: the diagnostic must keep its word spacing (the
        // format string previously glued `{type_name}` to "is" and "type." to
        // "The").
        let mut h = Header::new();
        let ExrError::Arg(msg) = h.set_type("bogusType").unwrap_err() else {
            panic!("expected ExrError::Arg");
        };
        assert!(msg.contains("bogusType is not a supported image type."), "{msg}");
        assert!(msg.contains("type. The following are supported"), "{msg}");
    }
}
