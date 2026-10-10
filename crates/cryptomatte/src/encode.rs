//! Cryptomatte encode: named coverages -> ranked EXR image + header metadata.
//!
//! Builds Cryptomatte 1.2.0 file payloads (channels + `cryptomatte/<id7>/...`
//! string attrs + JSON manifest). Renderer sample-filter kernels are out of
//! scope; supply already-aggregated coverages.

use std::fs;
use std::path::Path;

use exr_core::attr::{ExrString, Name};
use exr_core::header::Header;
use exr_core::io::OStream;
use exr_core::{Attribute, Compression, Image, StringAttribute};

use crate::channels::RankedLayer;
use crate::error::{Error, Result};
use crate::file::Host;
use crate::id::{float_id_to_hex, layer_metadata_id, name_to_float_id};
use crate::manifest::{validate_sidecar_for_write, Manifest};
use crate::metadata::{
    metadata_key, CONVERSION_UINT32_TO_FLOAT32, HASH_MURMUR3_32, METADATA_PREFIXES,
};
use crate::rank::{
    merge_contribution, pack_image, replace_ranked_channels, Contribution, MergeOp, RankedPlanes,
    Windows,
};

/// Default number of numbered EXR layers (spec suggested default: 3 => 6 ranks).
pub const DEFAULT_RANKED_LAYERS: usize = 3;

/// Where to place the JSON manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestPlacement {
    /// Inline `cryptomatte/<id>/manifest` header string attribute.
    Inline,
    /// Sidecar file relative to the EXR; writes `manif_file`. The path must
    /// pass [`validate_sidecar_for_write`] (spec 1.2.0 p.3: relative, forward
    /// slashes, no `./` or `..`, `.json`). Only [`EncodedCryptomatte::write`]
    /// can place it; a stream write refuses this placement.
    Sidecar {
        /// Path relative to the EXR file's directory.
        relative_path: String,
    },
}

/// Encode options for one Cryptomatte type.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Channel / type name prefix (e.g. `CryptoObject`).
    pub crypto_name: String,
    /// Number of `NameNN` RGBA layers (each holds 2 ID slots).
    pub ranked_layers: usize,
    /// Metadata key prefix (`cryptomatte/` by default).
    pub md_prefix: String,
    /// Manifest placement.
    pub manifest: ManifestPlacement,
}

impl EncodeOptions {
    /// Defaults: 3 ranked layers, modern `cryptomatte/` prefix, inline manifest.
    pub fn new(crypto_name: impl Into<String>) -> Self {
        Self {
            crypto_name: crypto_name.into(),
            ranked_layers: DEFAULT_RANKED_LAYERS,
            md_prefix: METADATA_PREFIXES[0].to_string(),
            manifest: ManifestPlacement::Inline,
        }
    }
}

/// One named coverage plane (`width * height` samples, row-major).
#[derive(Clone, Debug)]
pub struct NamedMatte {
    /// Object / material / asset name (UTF-8; hashed for ID).
    pub name: String,
    /// Coverage samples (same layout as [`crate::matte::Matte::coverage`]).
    pub coverage: Vec<f32>,
}

impl NamedMatte {
    /// Borrowing constructor that copies the coverage plane.
    pub fn new(name: impl Into<String>, coverage: &[f32]) -> Self {
        Self {
            name: name.into(),
            coverage: coverage.to_vec(),
        }
    }
}

/// Header string attributes for Cryptomatte metadata keys.
///
/// C-08 (closed): [`Name`] keys, not `String` — this list also carries
/// attributes copied verbatim from a SOURCE header
/// ([`carried_header_attrs`]), which may have a non-UTF-8 name (the
/// underlying `exr_core::Header` never checks encoding); byte-exact carry-
/// through instead of a lossy/skip fallback.
pub type HeaderAttrs = Vec<(Name, Box<dyn Attribute>)>;

/// Encoded Cryptomatte image ready to write.
///
/// `Debug` is implemented by hand: the carried header attributes are trait
/// objects without `Debug`, so it reports their keys.
pub struct EncodedCryptomatte {
    /// Packed ranked float channels.
    pub image: Image,
    /// Cryptomatte type name.
    pub crypto_name: String,
    /// Seven-hex metadata id.
    pub metadata_id: String,
    /// Metadata key prefix used for attrs.
    pub md_prefix: String,
    /// Manifest name -> float ID.
    pub manifest: Manifest,
    /// Ranked layer channel map.
    pub ranked: Vec<RankedLayer>,
    /// Header attributes to insert on write.
    pub attrs: HeaderAttrs,
    /// Sidecar relative path when using [`ManifestPlacement::Sidecar`].
    pub sidecar_rel: Option<String>,
    /// Sidecar JSON body (if any), written beside the EXR on [`Self::write`].
    pub sidecar_json: Option<String>,
    /// Set by [`crate::CryptomatteFile::inject`] for a multipart source: the file
    /// and part this payload replaces, so a write splices it into a copy of the
    /// source's other parts instead of writing a single-part file.
    pub(crate) host: Option<Host>,
}

impl std::fmt::Debug for EncodedCryptomatte {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedCryptomatte")
            .field("image", &self.image)
            .field("crypto_name", &self.crypto_name)
            .field("metadata_id", &self.metadata_id)
            .field("md_prefix", &self.md_prefix)
            .field("manifest", &self.manifest)
            .field("ranked", &self.ranked)
            .field("attrs", &self.attrs.iter().map(|(k, _)| k).collect::<Vec<_>>())
            .field("sidecar_rel", &self.sidecar_rel)
            .finish_non_exhaustive()
    }
}

fn string_attr(value: &str) -> Box<dyn Attribute> {
    Box::new(StringAttribute::new(ExrString::from(value)))
}

fn build_attrs(
    md_prefix: &str,
    metadata_id: &str,
    crypto_name: &str,
    placement: &ManifestPlacement,
    manif: &Manifest,
) -> Result<(HeaderAttrs, Option<String>)> {
    let key = |field: &str| -> Name { metadata_key(md_prefix, metadata_id, field).into() };
    // Spec 1.2.0 p.3: "conversion: The conversion method used to convert the
    // hash value into pixel color data. Only one conversion method,
    // "uint32_to_float32", is supported." (We used to write "float", which no
    // reference encoder produces.)
    let mut attrs: HeaderAttrs = vec![
        (key("name"), string_attr(crypto_name)),
        (key("hash"), string_attr(HASH_MURMUR3_32)),
        (
            key("conversion"),
            string_attr(CONVERSION_UINT32_TO_FLOAT32),
        ),
    ];
    let mut sidecar_json = None;
    match placement {
        ManifestPlacement::Inline => {
            let json = manif.to_json_str()?;
            attrs.push((key("manifest"), string_attr(&json)));
        }
        ManifestPlacement::Sidecar { relative_path } => {
            validate_sidecar_for_write(relative_path)?;
            attrs.push((key("manif_file"), string_attr(relative_path)));
            sidecar_json = Some(manif.to_json_str()?);
        }
    }
    Ok((attrs, sidecar_json))
}

fn manifest_from_names(names: &[String]) -> Manifest {
    Manifest::from_by_name(names.iter().map(|n| (n.clone(), name_to_float_id(n))).collect())
}

/// Encode named coverage mattes into a Cryptomatte image + metadata.
///
/// Every matte covers `windows.data` row-major; the image keeps both windows.
pub fn from_named_mattes(
    windows: Windows,
    mattes: &[NamedMatte],
    opts: &EncodeOptions,
) -> Result<EncodedCryptomatte> {
    let (width, height) = windows.size()?;
    let n = width
        .checked_mul(height)
        .ok_or_else(|| Error::Metadata("image pixel count overflow".into()))?;
    if mattes.is_empty() {
        return Err(Error::Metadata("encode requires at least one named matte".into()));
    }
    for m in mattes {
        if m.name.is_empty() {
            return Err(Error::Metadata("named matte has empty name".into()));
        }
        if m.coverage.len() != n {
            return Err(Error::Metadata(format!(
                "matte `{}` has {} samples, want {n}",
                m.name,
                m.coverage.len()
            )));
        }
    }

    let ids: Vec<f32> = mattes.iter().map(|m| name_to_float_id(&m.name)).collect();
    let names: Vec<String> = mattes.iter().map(|m| m.name.clone()).collect();

    let (image, ranked) = pack_image(
        windows,
        &opts.crypto_name,
        opts.ranked_layers,
        |i| {
            let mut contribs = Vec::with_capacity(mattes.len());
            for (m, &id) in mattes.iter().zip(ids.iter()) {
                let c = m.coverage[i];
                if c.is_finite() && c > 0.0 {
                    contribs.push(Contribution::new(id, c));
                }
            }
            contribs
        },
    )?;

    let metadata_id = layer_metadata_id(&opts.crypto_name);
    let manifest = manifest_from_names(&names);
    let (attrs, sidecar_json) = build_attrs(
        &opts.md_prefix,
        &metadata_id,
        &opts.crypto_name,
        &opts.manifest,
        &manifest,
    )?;
    let sidecar_rel = match &opts.manifest {
        ManifestPlacement::Sidecar { relative_path } => Some(relative_path.clone()),
        ManifestPlacement::Inline => None,
    };

    log::info!(
        "cryptomatte encode: name={} layers={} names={} size={}x{}",
        opts.crypto_name,
        opts.ranked_layers,
        names.len(),
        width,
        height
    );

    Ok(EncodedCryptomatte {
        image,
        crypto_name: opts.crypto_name.clone(),
        metadata_id,
        md_prefix: opts.md_prefix.clone(),
        manifest,
        ranked,
        attrs,
        sidecar_rel,
        sidecar_json,
        host: None,
    })
}

/// Unpack ranked Cryptomatte channels at linear pixel `i` into contributions.
///
/// One-shot convenience: it converts the ranked planes on every call, so a loop
/// over pixels must build a [`RankedPlanes`] once and index that instead.
pub fn unpack_contributions(
    image: &Image,
    ranked: &[RankedLayer],
    pixel_index: usize,
    pixel_count: usize,
) -> Result<Vec<Contribution>> {
    RankedPlanes::new(image, ranked, pixel_count)?.contributions_at(pixel_index)
}

/// Header attributes of `src` that must be carried over when an [`Image`] is
/// written back — everything except the ones the writer rebuilds from the
/// image itself and the keys of the Cryptomatte type being re-encoded.
///
/// The windows / compression / line order travel on the [`Image`]; `tiles`,
/// `chunkCount` and the multipart keys would describe a file layout the
/// scanline writer does not produce, so they are dropped rather than lied about.
pub fn carried_header_attrs(src: &Header, drop_key_prefix: Option<&str>) -> HeaderAttrs {
    const REBUILT: &[&str] = &[
        "channels",
        "dataWindow",
        "displayWindow",
        "compression",
        "lineOrder",
        "type",
        "tiles",
        "chunkCount",
        "version",
        "name",
        "view",
    ];
    // C-08 (closed): byte-exact throughout — REBUILT/drop_key_prefix are
    // ASCII, so comparing raw bytes correctly leaves a non-UTF-8 attribute
    // name untouched by either filter (it can never match one), and the
    // carried key/value are copied byte-exact rather than skipped.
    src.iter()
        .filter(|(key, _)| !REBUILT.iter().any(|r| r.as_bytes() == key.as_bytes()))
        .filter(|(key, _)| {
            !drop_key_prefix.is_some_and(|p| key.as_bytes().starts_with(p.as_bytes()))
        })
        .map(|(key, attr)| (key.clone(), attr.clone_attr()))
        .collect()
}

/// Inputs for [`inject_named_matte`].
pub struct InjectArgs<'a> {
    /// Existing Cryptomatte image (ranked channels).
    pub image: &'a Image,
    /// Cryptomatte type name (must match `opts.crypto_name`).
    pub crypto_name: &'a str,
    /// Ranked layers already present on `image`.
    pub ranked: &'a [RankedLayer],
    /// Manifest of names already in the layer.
    pub existing_manifest: &'a Manifest,
    /// New object / material name.
    pub name: &'a str,
    /// New coverage plane (`width * height`).
    pub coverage: &'a [f32],
    /// Over or under composite.
    pub op: MergeOp,
    /// Encode options (ranked layer count, manifest placement, …).
    pub opts: &'a EncodeOptions,
}

/// Inject a named matte into an existing Cryptomatte (Encryptomatte-domain).
///
/// Unpacks all ranks, applies [`MergeOp`], then fully re-ranks into
/// `ranked_layers` slots (more accurate than Nuke cascading 2-slot merge).
pub fn inject_named_matte(args: InjectArgs<'_>) -> Result<EncodedCryptomatte> {
    let InjectArgs {
        image,
        crypto_name,
        ranked,
        existing_manifest,
        name,
        coverage,
        op,
        opts,
    } = args;
    let width = usize::try_from(image.width())
        .map_err(|_| Error::Metadata(format!("invalid image width {}", image.width())))?;
    let height = usize::try_from(image.height())
        .map_err(|_| Error::Metadata(format!("invalid image height {}", image.height())))?;
    let n = width
        .checked_mul(height)
        .ok_or_else(|| Error::Metadata("image pixel count overflow".into()))?;
    if coverage.len() != n {
        return Err(Error::Metadata(format!(
            "inject coverage has {} samples, want {n}",
            coverage.len()
        )));
    }
    if name.is_empty() {
        return Err(Error::Metadata("inject name is empty".into()));
    }
    if crypto_name != opts.crypto_name {
        return Err(Error::Metadata(format!(
            "crypto_name mismatch: image `{crypto_name}` vs opts `{}`",
            opts.crypto_name
        )));
    }

    let new_id = name_to_float_id(name);
    let ranked_layers = if opts.ranked_layers > 0 {
        opts.ranked_layers
    } else if ranked.is_empty() {
        DEFAULT_RANKED_LAYERS
    } else {
        ranked.len()
    };

    // Convert the ranked planes once, not once per pixel (the old per-pixel
    // `unpack_contributions` call made a HALF inject O(n²)).
    let planes = RankedPlanes::new(image, ranked, n)?;
    let mut per_pixel: Vec<Vec<Contribution>> = Vec::with_capacity(n);
    for (i, &cov) in coverage.iter().enumerate() {
        let existing = planes.contributions_at(i)?;
        per_pixel.push(merge_contribution(&existing, new_id, cov, op));
    }

    // Re-encode this type inside the source image: every other channel, the
    // windows, the compression and the line order survive.
    let (out_image, out_ranked) =
        replace_ranked_channels(image, crypto_name, ranked_layers, |i| {
            std::mem::take(&mut per_pixel[i])
        })?;

    // Keep the file's own manifest entries (their IDs may not be plain hashes)
    // and add the injected name.
    let mut manifest = existing_manifest.clone();
    manifest.insert(name, new_id);

    let metadata_id = layer_metadata_id(crypto_name);
    let (attrs, sidecar_json) = build_attrs(
        &opts.md_prefix,
        &metadata_id,
        crypto_name,
        &opts.manifest,
        &manifest,
    )?;
    let sidecar_rel = match &opts.manifest {
        ManifestPlacement::Sidecar { relative_path } => Some(relative_path.clone()),
        ManifestPlacement::Inline => None,
    };

    Ok(EncodedCryptomatte {
        image: out_image,
        crypto_name: crypto_name.to_string(),
        metadata_id,
        md_prefix: opts.md_prefix.clone(),
        manifest,
        ranked: out_ranked,
        attrs,
        sidecar_rel,
        sidecar_json,
        host: None,
    })
}

impl EncodedCryptomatte {
    /// Carry `src`'s header attributes into this payload so a write-back keeps
    /// the source file's metadata (comments, chromaticities, custom attrs …).
    ///
    /// `drop_key_prefix` is the re-encoded type's key prefix
    /// ([`crate::LayerInfo::key_prefix`]); its attributes come from this
    /// payload instead. Carried attributes are placed first, so this payload's
    /// own Cryptomatte attributes always win.
    pub fn carry_header_attrs(&mut self, src: &Header, drop_key_prefix: Option<&str>) {
        let mut carried = carried_header_attrs(src, drop_key_prefix);
        carried.append(&mut self.attrs);
        self.attrs = carried;
    }

    /// Compression to write with: the caller's choice, else the image's own
    /// (the source file's, for inject).
    fn write_compression(&self, compression: Option<Compression>) -> Compression {
        compression.unwrap_or(self.image.compression)
    }

    /// Header attributes to overlay on the image when it is written.
    fn extra_attrs(&self) -> HeaderAttrs {
        self.attrs.iter().map(|(k, a)| (k.clone(), a.clone_attr())).collect()
    }

    /// Write EXR (and sidecar JSON if configured) to `path`.
    ///
    /// `compression` of `None` keeps the image's compression — what inject
    /// wants, so a re-encoded file is not silently recompressed. For a payload
    /// injected into a multipart file this writes the whole multipart file: the
    /// other parts are raw-copied from the source and `compression` applies to
    /// the injected part only. `path` must not be the source file.
    ///
    /// That source is re-opened **by path** at write time (a change made to it since
    /// [`crate::CryptomatteFile::inject`] is what gets copied). The file is written
    /// beside `path` and renamed into place, so a failing write leaves no output and
    /// an existing `path` untouched.
    pub fn write(&self, path: impl AsRef<Path>, compression: Option<Compression>) -> Result<()> {
        let path = path.as_ref();
        let compression = self.write_compression(compression);
        match &self.host {
            None => self.image.write_with_attrs(path, compression, self.extra_attrs())?,
            Some(host) => self.write_spliced_file(host, path, compression)?,
        }
        if let (Some(rel), Some(json)) = (&self.sidecar_rel, &self.sidecar_json) {
            let side = path.parent().unwrap_or_else(|| Path::new(".")).join(rel);
            if let Some(parent) = side.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&side, json)?;
            log::info!("cryptomatte sidecar: {}", side.display());
        }
        Ok(())
    }

    /// Write EXR bytes to a stream — stream form of [`EncodedCryptomatte::write`].
    ///
    /// Inline manifests only: a [`ManifestPlacement::Sidecar`] payload is
    /// refused, because its `manif_file` key would point at a JSON file this
    /// call has no directory to write, leaving a dangling reference in the EXR.
    /// For a multipart source the stream must not be that source file.
    pub fn write_to(
        &self,
        stream: &mut dyn OStream,
        compression: Option<Compression>,
    ) -> Result<()> {
        if let Some(rel) = &self.sidecar_rel {
            return Err(Error::Manifest(format!(
                "a stream write cannot place the sidecar manifest `{rel}`; \
                 use `write(path)` or an inline manifest"
            )));
        }
        let compression = self.write_compression(compression);
        match &self.host {
            None => self.image.write_to_with_attrs(stream, compression, self.extra_attrs())?,
            Some(host) => self.write_spliced(host, stream, compression)?,
        }
        Ok(())
    }

    /// Hex id for `name` from the built manifest (or freshly hashed).
    pub fn hex_for_name(&self, name: &str) -> String {
        self.manifest
            .by_name
            .get(name)
            .map(|id| float_id_to_hex(*id))
            .unwrap_or_else(|| float_id_to_hex(name_to_float_id(name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matte::extract_coverage;
    use exr_core::io::{MemIStream, MemOStream};

    #[test]
    fn encode_roundtrip_extract() {
        let w = 4usize;
        let h = 2usize;
        let n = w * h;
        let mut bunny = vec![0.0f32; n];
        let mut cube = vec![0.0f32; n];
        bunny[0] = 1.0;
        bunny[1] = 0.5;
        cube[1] = 0.5;
        cube[2] = 1.0;

        let mattes = [
            NamedMatte::new("bunny", &bunny),
            NamedMatte::new("cube", &cube),
        ];
        let opts = EncodeOptions::new("CryptoObject");
        let enc = from_named_mattes(Windows::origin(w, h).unwrap(), &mattes, &opts).unwrap();

        let mut out = MemOStream::new();
        enc.write_to(&mut out, Some(Compression::Zip)).unwrap();
        let bytes = out.into_data();
        let mut input = MemIStream::new(bytes);
        let image = Image::read_from(&mut input).unwrap();

        let ranked = crate::channels::ranked_layers_for(
            "CryptoObject",
            image.channels().map(|(n, _)| n),
        );
        assert!(!ranked.is_empty());

        let id_b = name_to_float_id("bunny");
        let id_c = name_to_float_id("cube");
        let mb = extract_coverage(&image, &ranked, &[id_b]).unwrap();
        let mc = extract_coverage(&image, &ranked, &[id_c]).unwrap();
        assert_eq!(mb.coverage[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(mb.coverage[1].to_bits(), 0.5f32.to_bits());
        assert_eq!(mc.coverage[1].to_bits(), 0.5f32.to_bits());
        assert_eq!(mc.coverage[2].to_bits(), 1.0f32.to_bits());

        let hex = enc.manifest.by_name.get("bunny").map(|i| float_id_to_hex(*i));
        assert_eq!(hex.as_deref(), Some("13851a76"));
    }

    #[test]
    fn inject_over_reduces_background() {
        let w = 1usize;
        let h = 1usize;
        let bg = NamedMatte::new("cube", &[1.0]);
        let opts = EncodeOptions {
            ranked_layers: 1,
            ..EncodeOptions::new("CryptoObject")
        };
        let base = from_named_mattes(Windows::origin(w, h).unwrap(), &[bg], &opts).unwrap();
        let inj = inject_named_matte(InjectArgs {
            image: &base.image,
            crypto_name: "CryptoObject",
            ranked: &base.ranked,
            existing_manifest: &base.manifest,
            name: "bunny",
            coverage: &[0.25],
            op: MergeOp::Over,
            opts: &opts,
        })
        .unwrap();
        let id_b = name_to_float_id("bunny");
        let id_c = name_to_float_id("cube");
        let mb = extract_coverage(&inj.image, &inj.ranked, &[id_b]).unwrap();
        let mc = extract_coverage(&inj.image, &inj.ranked, &[id_c]).unwrap();
        assert!((mb.coverage[0] - 0.25).abs() < 1e-6);
        assert!((mc.coverage[0] - 0.75).abs() < 1e-6);
    }

    /// C4: an offset data window and its display window survive encode.
    #[test]
    fn encode_keeps_both_windows() {
        let data = imath_rs::Box2i {
            min: imath_rs::V2i { x: 10, y: -3 },
            max: imath_rs::V2i { x: 13, y: -2 },
        };
        let display = imath_rs::Box2i {
            min: imath_rs::V2i { x: 0, y: -5 },
            max: imath_rs::V2i { x: 19, y: 4 },
        };
        let windows = Windows { data, display };
        let bunny = NamedMatte::new("bunny", &[1.0; 8]);
        let enc = from_named_mattes(windows, &[bunny], &EncodeOptions::new("CryptoObject")).unwrap();
        let mut out = MemOStream::new();
        enc.write_to(&mut out, Some(Compression::Zip)).unwrap();
        let image = Image::read_from(&mut MemIStream::new(out.into_data())).unwrap();
        assert_eq!(image.data_window, data);
        assert_eq!(image.display_window, display);
    }

    /// C12: a stream write must not emit a `manif_file` it cannot back.
    #[test]
    fn stream_write_refuses_a_sidecar_manifest() {
        let opts = EncodeOptions {
            manifest: ManifestPlacement::Sidecar { relative_path: "m.json".into() },
            ..EncodeOptions::new("CryptoObject")
        };
        let bunny = NamedMatte::new("bunny", &[1.0]);
        let enc = from_named_mattes(Windows::origin(1, 1).unwrap(), &[bunny], &opts).unwrap();
        let err = enc.write_to(&mut MemOStream::new(), None).unwrap_err();
        assert!(matches!(err, Error::Manifest(_)), "{err:?}");
    }

    /// C11: the encoder only writes spec-conformant sidecar paths.
    #[test]
    fn encode_rejects_a_nonconformant_sidecar_path() {
        let bunny = NamedMatte::new("bunny", &[1.0]);
        for rel in ["../m.json", "./m.json", "/abs/m.json", "m.txt", "dir\\m.json"] {
            let opts = EncodeOptions {
                manifest: ManifestPlacement::Sidecar { relative_path: rel.into() },
                ..EncodeOptions::new("CryptoObject")
            };
            let err = from_named_mattes(Windows::origin(1, 1).unwrap(), std::slice::from_ref(&bunny), &opts)
                .unwrap_err();
            assert!(matches!(err, Error::Manifest(_)), "{rel:?}: {err:?}");
        }
    }

    /// C14: more ranked layers than two-digit names allow is an error.
    #[test]
    fn encode_rejects_more_than_100_ranked_layers() {
        let opts = EncodeOptions { ranked_layers: 101, ..EncodeOptions::new("CryptoObject") };
        let bunny = NamedMatte::new("bunny", &[1.0]);
        assert!(from_named_mattes(Windows::origin(1, 1).unwrap(), std::slice::from_ref(&bunny), &opts).is_err());
        let opts = EncodeOptions { ranked_layers: 100, ..opts };
        assert!(from_named_mattes(Windows::origin(1, 1).unwrap(), &[bunny], &opts).is_ok());
    }
}
