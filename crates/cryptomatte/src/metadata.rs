//! Discover Cryptomatte layers from OpenEXR header string attributes.

use std::collections::BTreeMap;

use exr_core::attr::ExrString;
use exr_core::header::Header;

use crate::channels::{rank_index, ranked_layers_for, RankedLayer};
use crate::error::{Error, Result};

/// Legal metadata key prefixes (Nuke `CRYPTO_METADATA_LEGAL_PREFIX`).
pub const METADATA_PREFIXES: &[&str] = &["cryptomatte/", "exr/cryptomatte/"];

/// Spec hash method name (`MurmurHash3_x86_32` implementation).
///
/// Spec 1.2.0 p.3: "hash: the type of hash used. "MurmurHash3_32" is the only
/// one currently supported."
pub const HASH_MURMUR3_32: &str = "MurmurHash3_32";

/// Spec conversion method name — the only legal `conversion` value to write.
///
/// Spec 1.2.0 p.3: "conversion: The conversion method used to convert the hash
/// value into pixel color data. Only one conversion method,
/// "uint32_to_float32", is supported."
pub const CONVERSION_UINT32_TO_FLOAT32: &str = "uint32_to_float32";

/// Header key for one Cryptomatte metadata field — SSOT for the key format.
///
/// `cryptomatte/<metadata_id>/<field>` (or the legacy `exr/cryptomatte/` prefix).
#[inline]
pub fn metadata_key(md_prefix: &str, metadata_id: &str, field: &str) -> String {
    format!("{md_prefix}{metadata_id}/{field}")
}

/// One Cryptomatte type discovered in a file header.
#[derive(Clone, Debug)]
pub struct LayerInfo {
    /// Seven-hex metadata id (e.g. `ae93ba3`).
    pub metadata_id: String,
    /// Cryptomatte type / channel name prefix (e.g. `uCryptoObject`).
    pub name: String,
    /// Hash method string from header (expected [`HASH_MURMUR3_32`]).
    pub hash: String,
    /// Conversion string — spec value [`CONVERSION_UINT32_TO_FLOAT32`]; the
    /// non-conformant `float` spelling is also accepted on read.
    pub conversion: String,
    /// Which legal prefix was used for this layer's keys.
    pub md_prefix: String,
    /// Optional sidecar relative path (`manif_file`).
    pub manif_file: Option<String>,
    /// Optional inline JSON manifest string (may be large; lazy-load consumers).
    pub inline_manifest: Option<String>,
    /// Ranked numbered layers derived from channel names.
    pub ranked: Vec<RankedLayer>,
    /// Part of a multipart file that holds [`Self::ranked`]'s channels (`0`
    /// for a single-part file).
    pub part: usize,
}

impl LayerInfo {
    /// Full header key for a field (`name`, `manifest`, …).
    pub fn metadata_key(&self, field: &str) -> String {
        metadata_key(&self.md_prefix, &self.metadata_id, field)
    }

    /// Key prefix owned by this layer (`cryptomatte/<id>/`) — every header key
    /// starting with it belongs to this Cryptomatte type.
    ///
    /// W4.6 (design_r5.md §4.4): expressed via the canonical [`metadata_key`]
    /// with an empty `field` (`"{prefix}{id}/"`) instead of a second
    /// hand-written format string for the same shape.
    pub fn key_prefix(&self) -> String {
        metadata_key(&self.md_prefix, &self.metadata_id, "")
    }

    /// Validate hash + conversion for decode v1.
    pub fn validate_methods(&self) -> Result<()> {
        if !self.hash.is_empty() && !is_supported_hash(&self.hash) {
            return Err(Error::UnsupportedHash(self.hash.clone()));
        }
        if !self.conversion.is_empty() && !is_supported_conversion(&self.conversion) {
            return Err(Error::UnsupportedConversion(self.conversion.clone()));
        }
        Ok(())
    }
}

/// True for Cryptomatte MurmurHash3_32 (case-insensitive).
pub fn is_supported_hash(s: &str) -> bool {
    s.eq_ignore_ascii_case(HASH_MURMUR3_32)
}

/// True for the spec conversion [`CONVERSION_UINT32_TO_FLOAT32`].
///
/// The bare `float` spelling is also accepted on read: it is what older exr-rs
/// builds wrote, and the reference decoders (Nuke `cryptomatte_utilities.py`,
/// the Fusion Lua module) never look at the value at all, so tolerating it
/// cannot make us stricter than the reference. Encoders must always write the
/// conformant value.
pub fn is_supported_conversion(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    s == CONVERSION_UINT32_TO_FLOAT32 || s == "float"
}

fn header_string(header: &Header, key: &str) -> Option<String> {
    header
        .find_typed_attribute::<ExrString>(key)
        .and_then(|a| a.value.as_str().map(str::to_owned))
}

/// Discover Cryptomatte layers from a header and a list of channel names.
///
/// Channel names come from the same file (header channel list or decoded
/// [`exr_core::Image`]) so ranking stays consistent with matte extract.
pub fn discover_layers(header: &Header, channel_names: &[&str]) -> Result<Vec<LayerInfo>> {
    // metadata_id → (prefix, field → value)
    let mut raw: BTreeMap<String, (String, BTreeMap<String, String>)> = BTreeMap::new();

    for (key, _) in header.iter() {
        // C-08 follow-up: Cryptomatte metadata keys are always ASCII in
        // practice; a non-UTF-8 attribute name simply cannot match one of
        // the `METADATA_PREFIXES` and is skipped.
        let Some(key) = key.to_str() else {
            continue;
        };
        let Some((prefix, rest)) = METADATA_PREFIXES
            .iter()
            .find_map(|p| key.strip_prefix(p).map(|r| (*p, r)))
        else {
            continue;
        };
        let Some((metadata_id, field)) = rest.split_once('/') else {
            continue;
        };
        if metadata_id.is_empty() || field.is_empty() {
            continue;
        }
        let Some(value) = header_string(header, key) else {
            log::debug!("cryptomatte: skip non-string attr {key}");
            continue;
        };
        // Keys under both prefixes for one id merge field by field, as in Nuke
        // (`cryptomatte_utilities.py`: one dict per metadata id, whichever
        // prefix a key came with). The layer keeps the first prefix seen, so a
        // re-encode writes its keys back under that prefix.
        raw.entry(metadata_id.to_string())
            .or_insert_with(|| (prefix.to_string(), BTreeMap::new()))
            .1
            .insert(field.to_string(), value);
    }

    let mut layers = Vec::with_capacity(raw.len());
    for (metadata_id, (md_prefix, fields)) in raw {
        let name = fields.get("name").cloned().unwrap_or_default();
        let hash = fields.get("hash").cloned().unwrap_or_default();
        let conversion = fields.get("conversion").cloned().unwrap_or_default();
        let manif_file = fields
            .get("manif_file")
            .cloned()
            .filter(|s| !s.is_empty());
        let inline_manifest = fields
            .get("manifest")
            .cloned()
            .filter(|s| !s.is_empty());

        let ranked = if name.is_empty() {
            Vec::new()
        } else {
            ranked_layers_for(&name, channel_names.iter().copied())
        };
        // Spec 1.2.0 p.3 lists `hash` and `conversion` as required; the
        // reference decoder never reads them, so a missing one is reported,
        // not fatal.
        for (field, value) in [("hash", &hash), ("conversion", &conversion)] {
            if value.is_empty() {
                log::warn!(
                    "cryptomatte: layer `{name}` ({metadata_id}) has no `{field}` metadata \
                     (required by spec 1.2.0); assuming the spec method"
                );
            }
        }
        warn_on_rank_gaps(&name, &ranked);

        let layer = LayerInfo {
            metadata_id,
            name,
            hash,
            conversion,
            md_prefix,
            manif_file,
            inline_manifest,
            ranked,
            part: 0,
        };
        // One layer with an unknown hash/conversion must not hide the rest of
        // the file: skip it and keep going (Nuke does not validate these at
        // all, so failing the whole file is stricter than the reference).
        if let Err(e) = layer.validate_methods() {
            log::warn!(
                "cryptomatte: skipping layer `{}` ({}): {e}",
                layer.name,
                layer.metadata_id
            );
            continue;
        }
        layers.push(layer);
    }

    layers.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(layers)
}

/// Warn when a type's ranked layers are not exactly `00, 01, ..`: a gap means
/// channels are missing from the file. Coverage still sums over the ranks
/// that are present, so this is reported, not fatal.
fn warn_on_rank_gaps(name: &str, ranked: &[RankedLayer]) {
    let indices: Vec<Option<usize>> = ranked.iter().map(|l| rank_index(name, &l.base)).collect();
    if indices.iter().enumerate().any(|(i, idx)| *idx != Some(i)) {
        let bases: Vec<&str> = ranked.iter().map(|l| l.base.as_str()).collect();
        log::warn!("cryptomatte: layer `{name}` ranks are not contiguous from 00: {bases:?}");
    }
}

/// Channel names from a header's channel list attribute.
pub fn channel_names_from_header(header: &Header) -> Result<Vec<String>> {
    let list = header.channels()?;
    Ok(list.iter().map(|(name, _)| name.to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr_core::StringAttribute;

    fn header_with(entries: &[(&str, &str)]) -> Header {
        let mut h = Header::new();
        for (k, v) in entries {
            h.insert(*k, StringAttribute::new(ExrString::from(*v)))
                .expect("insert");
        }
        h
    }

    #[test]
    fn bad_layer_does_not_hide_good_layers() {
        let header = header_with(&[
            ("cryptomatte/aaaaaaa/name", "CryptoAsset"),
            ("cryptomatte/aaaaaaa/hash", "sha1"), // unsupported -> skipped
            ("cryptomatte/aaaaaaa/conversion", CONVERSION_UINT32_TO_FLOAT32),
            ("cryptomatte/bbbbbbb/name", "CryptoObject"),
            ("cryptomatte/bbbbbbb/hash", HASH_MURMUR3_32),
            ("cryptomatte/bbbbbbb/conversion", CONVERSION_UINT32_TO_FLOAT32),
        ]);
        let layers = discover_layers(&header, &["CryptoObject00.R", "CryptoObject00.G"])
            .expect("discover");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "CryptoObject");
        assert_eq!(layers[0].ranked.len(), 1);
    }

    #[test]
    fn legacy_float_conversion_still_reads() {
        let header = header_with(&[
            ("cryptomatte/bbbbbbb/name", "CryptoObject"),
            ("cryptomatte/bbbbbbb/hash", HASH_MURMUR3_32),
            ("cryptomatte/bbbbbbb/conversion", "float"),
        ]);
        let layers = discover_layers(&header, &[]).expect("discover");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].metadata_key("conversion"), "cryptomatte/bbbbbbb/conversion");
        assert_eq!(layers[0].key_prefix(), "cryptomatte/bbbbbbb/");
    }
}

