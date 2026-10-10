//! Cryptomatte helpers for exr-rs — decode and encode ID mattes.
//!
//! Design: `docs/history/specs/2026-08-10-cryptomatte-design.md`.
//! Hashing uses [`murmur3::x86_32`]; EXR I/O uses [`exr_core`].

#![deny(missing_docs)]

pub mod channels;
pub mod encode;
pub mod error;
pub mod file;
pub mod id;
pub mod manifest;
pub mod matte;
pub mod metadata;
pub mod rank;

pub use channels::{
    rank_index, ranked_channel_of, ranked_layers_for, Component, RankedLayer, Spelling,
    MAX_RANKED_LAYERS,
};
pub use encode::{
    carried_header_attrs, from_named_mattes, inject_named_matte, unpack_contributions,
    EncodeOptions, EncodedCryptomatte, HeaderAttrs, InjectArgs, ManifestPlacement, NamedMatte,
    DEFAULT_RANKED_LAYERS,
};
pub use error::{Error, Result};
pub use file::{discover_path, CryptomatteFile};
pub use id::{
    float_id_to_hex, hash32_to_float_id, hex_to_float_id, layer_metadata_id, name_to_float_id,
    name_to_hash32,
};
pub use manifest::{
    load_manifest, resolve_sidecar, validate_sidecar, validate_sidecar_for_write, Manifest,
};
pub use matte::{
    channel_as_f32, extract_coverage, extract_coverage_with, id_channel_as_f32, require_channel,
    Matte,
};
pub use metadata::{
    discover_layers, metadata_key, LayerInfo, CONVERSION_UINT32_TO_FLOAT32, HASH_MURMUR3_32,
    METADATA_PREFIXES,
};
pub use rank::{
    merge_and_rank, merge_contribution, pack_channels, pack_image, replace_ranked_channels,
    Contribution, MergeOp, PackedChannels, RankedPlanes, Windows,
};
