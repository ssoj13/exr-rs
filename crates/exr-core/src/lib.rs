//! `exr-core` — a pure-Rust, dependency-light 1:1 port of OpenEXR (the C++
//! `IlmImf` library): the EXR container, every compression codec, tiling /
//! mip-rip levels, deep data, and multipart / multiview — read and write.
//!
//! The container/streaming layer is ported fresh from the C++; the codecs
//! converge to the C++ 1:1. This crate itself writes NO `unsafe`: SIMD hot
//! paths behind a byte-identical safe scalar fallback live in `deflate-rs`
//! (this workspace) and in `codec-simd` (the `compressor::predictor`
//! decode-side transform's dependency, a safe API; exr-core itself writes no
//! `unsafe`). Output is verified
//! byte-for-byte (and, for lossy codecs, decode-exact) against real C++
//! OpenEXR 3.4 by the parity harness, and read against the official
//! `openexr-images` corpus.
//!
//! Port target (pinned tag v3.3.11, with the HTJ2K codec back-ported from 3.4):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/>
//!
//! # Two layers: the `api` facade vs. the low-level modules
//!
//! `exr-core` exposes **two** levels, and most callers want the first:
//!
//! * The ergonomic **[`api`] facade** — [`Image`] plus [`read`] / [`Image::write`].
//!   This is the "load / save an image" common case: one in-memory [`Image`]
//!   (data/display windows + named, typed channel buffers) read from or written
//!   to a single-part **scanline or tiled** (non-deep) file. It is the only
//!   place in the crate where idiomatic, non-1:1 Rust lives; it *drives* the
//!   verified readers/writers rather than reimplementing them.
//! * The **low-level modules**, which mirror IlmImf file-for-file, for anything
//!   the facade deliberately does not cover:
//!   * [`scanline`] / [`tiled`] — per-scanline / per-tile streaming, mip/rip
//!     levels, subsampled (chroma) channels via an explicit [`frame_buffer`];
//!   * [`deep`] — deep scanline / deep tiled images;
//!   * [`multipart`] / [`multi_view`] — multipart and multiview files;
//!   * [`header`], [`attr`], [`compressor`], [`container`], [`io`], [`misc`] —
//!     the supporting header / attribute / codec / Xdr / stream machinery.
//!
//! When the facade meets a deep, multipart, or subsampled file it returns a
//! clear [`ExrError`] pointing at the matching low-level type, never a
//! half-decoded result.
//!
//! # Getting started
//!
//! Build an [`Image`] in memory, write it, and read it straight back — this
//! example runs as a doctest over an in-memory stream (no file on disk):
//!
//! ```
//! use exr_core::{ChannelData, Image};
//! use exr_core::attr::Compression;
//! use exr_core::io::{MemIStream, MemOStream};
//! use imath_rs::{Box2i, Half, V2i};
//!
//! // A 4x3 image with one HALF "R" channel, raster order over the data window.
//! let data_window = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 3, y: 2 } };
//! let r: Vec<Half> = (0..4 * 3).map(|i| Half::from(i as f32)).collect();
//! let img = Image::new(data_window).with_channel("R", ChannelData::Half(r));
//!
//! // Write it as a single-part scanline file (here into a memory buffer).
//! let mut out = MemOStream::new();
//! img.write_to(&mut out, Compression::Zip)?;
//!
//! // Read it back and confirm the round-trip.
//! let mut input = MemIStream::new(out.into_data());
//! let back = Image::read_from(&mut input)?;
//! assert_eq!(back.width(), 4);
//! assert_eq!(back.height(), 3);
//! assert!(back.channel("R").is_some());
//! # Ok::<(), exr_core::ExrError>(())
//! ```
//!
//! To read an EXR from disk, use [`read`] (or [`Image::read`]); to inspect any
//! file kind without decoding pixels, see the `exr-rs info` CLI.
//!
//! # Naming note — `container` vs `core`
//! The obvious name `core` for the Xdr/version module would shadow Rust's
//! built-in `::core` crate and cause confusion in paths.  It is therefore named
//! [`container`], which matches its purpose (the EXR *container* layer) and is
//! unambiguous.  Downstream code imports it as `exr_core::container`.

pub mod api;
pub mod attr;
pub mod buf;
pub mod checked_arithmetic;
pub mod chunk;
pub mod compressor;
pub mod container;
pub mod convert;
pub mod deep;
pub mod error;
pub mod frame_buffer;
pub mod header;
pub mod io;
pub mod misc;
pub mod multi_view;
pub mod multipart;
pub(crate) mod probe;
pub mod rgba;
pub mod scanline;
pub mod sse;
pub mod standard_attributes;
pub mod threading;
pub mod tiled;

// ---------------------------------------------------------------------------
// Crate-level re-exports
// ---------------------------------------------------------------------------

pub use error::{decode_err, encode_err, ExrError, IexClass, Result};

// Raw-chunk passthrough (ADD-1): byte-exact EXR→EXR transcode without decoding.
// Port of `InputFile::rawPixelData` / `TiledInputFile::rawTileData` +
// `OutputFile::copyPixels` / `TiledOutputFile::copyPixels`.
pub use chunk::{ChunkCoord, RawChunk, RawImage, RawPart};

// Idiomatic image facade — the ergonomic common-case API over
// the verified low-level core. Deep / multipart use the low-level types above.
pub use api::{ChannelData, Image, Unsupported, classify_unsupported, read, write_rgba_file};

// Deep data model + deep I/O
pub use deep::{
    DeepFrameBuffer, DeepScanLineInputFile, DeepScanLineOutputFile, DeepSlice,
    DeepTiledInputFile, DeepTiledOutputFile,
};

// Multipart container
pub use multipart::{MultiPartInputFile, MultiPartOutputFile};

pub use rgba::{Rgba, RgbaImage, RgbaInputFile, RgbaOutputFile};

// Multi-view channel/view helpers (ImfMultiView)
pub use multi_view::{
    are_counterparts, channel_in_all_views, channel_in_other_view, channels_in_no_view,
    channels_in_view, default_view_name, insert_view_name, remove_view_name, view_from_channel_name,
};

// Attribute base + scalar/math typed attributes
pub use attr::{
    AttrValue, Attribute, Box2fAttribute, Box2iAttribute, DoubleAttribute, FloatAttribute,
    FloatVectorAttribute, IntAttribute, M33dAttribute, M33fAttribute, M44dAttribute, M44fAttribute,
    StringAttribute, StringVectorAttribute, TypedAttribute, V2dAttribute, V2fAttribute,
    V2iAttribute, V3dAttribute, V3fAttribute, V3iAttribute, AttributeReader, known_type,
    new_attribute, register_attribute_type, unregister_attribute_type,
};

// Support types
pub use attr::{
    Chromaticities, ChromaticitiesAttribute,
    Compression, CompressionAttribute,
    CubeMapFace, Envmap, EnvmapAttribute,
    DeepImageState, DeepImageStateAttribute,
    KeyCode, KeyCodeAttribute,
    LevelMode, LevelRoundingMode, TileDescription, TileDescriptionAttribute,
    LineOrder, LineOrderAttribute,
    Rational, RationalAttribute,
    TimeCode, TimeCodeAttribute, TimeCodePacking,
};

// Byte-exact name type (attribute/channel/part identity).
pub use attr::Name;

// Channel list, preview, opaque and ID-manifest types
pub use attr::{
    Channel, ChannelList, ChannelListAttribute, PixelType,
    ChannelGroupManifest, CompressedIDManifest, CUSTOMHASH, IDManifest, IDManifestAttribute,
    ID_SCHEME, ID2_SCHEME, IdLifetime, MURMURHASH3_32, MURMURHASH3_64, NOTHASHED, UNKNOWN,
    OpaqueAttribute,
    PreviewImage, PreviewImageAttribute, PreviewRgba,
};

// OpenEXR 3.4 additions
pub use attr::{
    Bytes, BytesAttribute,
    fps_23_976, fps_24, fps_25, fps_29_97, fps_30, fps_47_952, fps_48, fps_50, fps_59_94, fps_60,
    guess_exact_fps, guess_exact_fps_rational,
};

// Stream traits
pub use io::{IStream, MemIStream, MemOStream, OStream, StdIfStream, StdOfStream};

// Version / magic constants and predicates
pub use container::{
    ALL_FLAGS, EXR_VERSION, LIBRARY_VERSION, LONG_NAMES_FLAG, MAGIC, MULTI_PART_FILE_FLAG,
    NON_IMAGE_FLAG, TILED_FLAG, VERSION_FLAGS_FIELD, VERSION_NUMBER_FIELD,
    get_flags, get_library_version, get_version, is_imf_magic, is_multi_part, is_non_image,
    is_single_part_general, is_tiled, make_not_tiled, make_tiled, supports_flags,
};

// isOpenExrFile family
pub use container::{
    ExrFileProbe, is_deep_open_exr_file, is_deep_open_exr_file_stream, is_multi_part_open_exr_file,
    is_multi_part_open_exr_file_stream, is_open_exr_file, is_open_exr_file_stream,
    is_tiled_open_exr_file, is_tiled_open_exr_file_stream,
};

// Xdr write/read functions and size constants
pub use container::{
    SIZE_BOOL, SIZE_F32, SIZE_F64, SIZE_HALF, SIZE_I16, SIZE_I32, SIZE_I64, SIZE_I8,
    pad, read_bool, read_char, read_chars, read_f32, read_f64, read_half, read_i16, read_i32,
    read_i64, read_i8, read_string, read_u16, read_u32, read_u64, read_u8_val, skip, write_bool,
    write_char, write_chars, write_f32, write_f64, write_half, write_i16, write_i32, write_i64,
    write_i8, write_string, write_u16, write_u32, write_u64, write_u8_val,
};
