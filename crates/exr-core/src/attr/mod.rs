//! Port of OpenEXR's attribute layer — `ImfAttribute.h/.cpp` plus every typed
//! attribute header (`ImfIntAttribute.h`, `ImfFloatAttribute.h`,
//! `ImfDoubleAttribute.h`, `ImfStringAttribute.h/.cpp`,
//! `ImfStringVectorAttribute.h/.cpp`, `ImfFloatVectorAttribute.h/.cpp`,
//! `ImfBoxAttribute.h/.cpp`, `ImfVecAttribute.h/.cpp`,
//! `ImfMatrixAttribute.h/.cpp`), and the support types
//! (`ImfCompression.*`, `ImfLineOrder.*`, `ImfEnvmap.*`, `ImfDeepImageState.*`,
//! `ImfTileDescription.*`, `ImfChromaticities.*`, `ImfKeyCode.*`,
//! `ImfTimeCode.*`, `ImfRational.*`).
//!
//! An EXR header is a map from attribute name to a typed attribute value.  Each
//! attribute serialises its value through the [`crate::container::xdr`] layer so
//! that the on-disk bytes are identical to IlmImf.
//!
//! # Structure (how this mirrors the C++)
//! C++ uses `class Attribute` (an abstract base with `typeName`/`copy`/
//! `writeValueTo`/`readValueFrom`) and the class template
//! `template<class T> TypedAttribute<T>`.  Per value type `T` the library
//! specialises `staticTypeName()` and (where needed) `writeValueTo` /
//! `readValueFrom`.
//!
//! This port keeps that exact split but expresses it idiomatically without
//! hand-writing fifteen near-identical structs:
//!
//! * [`AttrValue`] is the per-value-type trait.  It carries the static type
//!   name and the byte-exact value (de)serialisation — i.e. the *content* of
//!   every C++ `staticTypeName`/`writeValueTo`/`readValueFrom` specialisation.
//! * [`TypedAttribute<T>`] is the single generic wrapper, the analogue of the
//!   C++ class template.
//! * [`Attribute`] is the object-safe trait stored in the header map (the
//!   analogue of the abstract C++ base class), with `Any`-based downcasting in
//!   place of C++ `dynamic_cast`.
//! * The `registry` mirrors `Attribute::newAttribute` / `registerAttributeType`:
//!   it maps a type-name string to a factory that reads a fully-formed
//!   `Box<dyn Attribute>` from a stream.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfAttribute.h>

pub mod attribute;

// Byte-exact name type (attribute/channel/part identity), port of ImfName.h.
pub mod name;

// Support-type modules
pub mod chromaticities;
pub mod compression;
pub mod deep_image_state;
pub mod envmap;
pub mod key_code;
pub mod line_order;
pub mod rational;
pub mod tile_description;
pub mod time_code;

// Channels, preview image, opaque catch-all, ID manifest.
pub mod channel_list;
pub mod id_manifest;
pub mod opaque;
pub mod preview_image;

// OpenEXR 3.4 addition: the `"bytes"` attribute.
pub mod bytes_attr;

// `framesPerSecond` convenience functions.
pub mod frames_per_second;

// Channel grouping into Nuke-style layers (shared by exr-gpu upload + exr-view UI).
pub mod layers;

// ---------------------------------------------------------------------------
// Re-exports from the base attribute module
// ---------------------------------------------------------------------------

pub use attribute::{
    AttrValue, Attribute, Box2fAttribute, Box2iAttribute, DoubleAttribute, ExrString,
    FloatAttribute, FloatVectorAttribute, IntAttribute, M33dAttribute, M33fAttribute,
    M44dAttribute, M44fAttribute, StringAttribute, StringVectorAttribute, TypedAttribute,
    V2dAttribute, V2fAttribute, V2iAttribute, V3dAttribute, V3fAttribute, V3iAttribute,
    AttributeReader, known_type, new_attribute, register_attribute_type, unregister_attribute_type,
};

// ---------------------------------------------------------------------------
// Re-exports from the support types
// ---------------------------------------------------------------------------

pub use chromaticities::{Chromaticities, ChromaticitiesAttribute};
pub use compression::{Compression, CompressionAttribute};
pub use deep_image_state::{DeepImageState, DeepImageStateAttribute};
pub use envmap::{CubeMapFace, Envmap, EnvmapAttribute};
pub use key_code::{KeyCode, KeyCodeAttribute};
pub use line_order::{LineOrder, LineOrderAttribute};
pub use rational::{Rational, RationalAttribute};
pub use tile_description::{
    LevelMode, LevelRoundingMode, TileDescription, TileDescriptionAttribute,
};
pub use time_code::{Packing as TimeCodePacking, TimeCode, TimeCodeAttribute};

// ---------------------------------------------------------------------------
// Re-exports from the channel / preview / opaque / ID-manifest types
// ---------------------------------------------------------------------------

pub use channel_list::{
    Channel, ChannelList, ChannelListAttribute, PixelType,
};
pub use bytes_attr::{Bytes, BytesAttribute};
pub use frames_per_second::{
    fps_23_976, fps_24, fps_25, fps_29_97, fps_30, fps_47_952, fps_48, fps_50, fps_59_94, fps_60,
    guess_exact_fps, guess_exact_fps_rational,
};
pub use id_manifest::{
    ChannelGroupManifest, CompressedIDManifest, CUSTOMHASH, IDManifest, IDManifestAttribute,
    ID_SCHEME, ID2_SCHEME, IdLifetime, MURMURHASH3_32, MURMURHASH3_64, NOTHASHED, UNKNOWN,
};
pub use layers::{LayerGroup, component_priority, group_layers, layer_prefix, channel_suffix};
pub use name::Name;
pub use opaque::OpaqueAttribute;
pub use preview_image::{PreviewImage, PreviewImageAttribute, PreviewRgba};
