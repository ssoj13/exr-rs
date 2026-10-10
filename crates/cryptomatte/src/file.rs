//! Single-open Cryptomatte file facade: header discover + [`Image`] decode.

use std::path::{Path, PathBuf};

use exr_core::attr::{LevelMode, Name};
use exr_core::header::{self, Header};
use exr_core::io::{MemIStream, MemOStream, OStream, StdIfStream};
use exr_core::multipart::copy_part;
use exr_core::{Compression, IStream, Image, MultiPartInputFile, MultiPartOutputFile};

use crate::channels::ranked_layers_for;
use crate::encode::{inject_named_matte, EncodeOptions, EncodedCryptomatte, InjectArgs};
use crate::error::{Error, Result};
use crate::manifest::{load_manifest, Manifest};
use crate::matte::{extract_coverage, Matte};
use crate::metadata::{channel_names_from_header, discover_layers, LayerInfo};
use crate::rank::MergeOp;

/// One part of an open Cryptomatte EXR: its header, decoded pixels and the
/// Cryptomatte types whose channels it holds.
///
/// A single-part file is part `0`. For a multipart file the part is chosen by
/// [`CryptomatteFile::open_for_layer`] (or given to
/// [`CryptomatteFile::open_part`]); a type's metadata may sit in another
/// part's header, see [`discover_path`].
#[derive(Clone)]
pub struct CryptomatteFile {
    /// Source path.
    pub path: PathBuf,
    /// Index of the decoded part.
    pub part: usize,
    /// Whether the source is a multipart file.
    pub multipart: bool,
    /// Header of the decoded part (all attributes, including non-Cryptomatte metadata).
    pub header: Header,
    /// Cryptomatte types whose ranked channels are in this part (sorted by name).
    pub layers: Vec<LayerInfo>,
    /// Decoded part (all channels).
    pub image: Image,
}

// `Header` has no `Debug` (exr-core), so summarise it rather than drop the
// derive on everything that holds a `CryptomatteFile`.
impl std::fmt::Debug for CryptomatteFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CryptomatteFile")
            .field("path", &self.path)
            .field("part", &self.part)
            .field("multipart", &self.multipart)
            .field("header_attrs", &self.header.iter().count())
            .field("layers", &self.layers)
            .field("image", &self.image)
            .finish()
    }
}

impl CryptomatteFile {
    /// Open the part holding the first Cryptomatte type (by name), or part 0
    /// when the file has none. Same as `open_for_layer(path, None)`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_for_layer(path, None)
    }

    /// Open the part holding Cryptomatte type `name`, or with `None` the part
    /// holding the first type by name (part 0 when the file has none).
    ///
    /// # Errors
    /// [`Error::Metadata`] when `name` is given and no part holds that type.
    pub fn open_for_layer(path: impl AsRef<Path>, name: Option<&str>) -> Result<Self> {
        let path = path.as_ref();
        let layers = discover_path(path)?;
        let part = match name {
            Some(n) => layers
                .iter()
                .find(|l| l.name == n)
                .map(|l| l.part)
                .ok_or_else(|| Error::Metadata(format!("Cryptomatte layer `{n}` not found")))?,
            None => layers.first().map_or(0, |l| l.part),
        };
        Self::open_parts(path, part, layers)
    }

    /// Open part `part` (`0` for a single-part file).
    ///
    /// # Errors
    /// Out-of-range `part`; [`exr_core::ExrError::Unsupported`] for a deep part.
    pub fn open_part(path: impl AsRef<Path>, part: usize) -> Result<Self> {
        let path = path.as_ref();
        let layers = discover_path(path)?;
        Self::open_parts(path, part, layers)
    }

    /// Decode `part` and keep the discovered types that live in it.
    fn open_parts(path: &Path, part: usize, layers: Vec<LayerInfo>) -> Result<Self> {
        let path_str = utf8(path)?;
        log::info!("cryptomatte open: {path_str} part {part}");
        let mut stream = StdIfStream::open(path_str)?;
        let mut file = MultiPartInputFile::new(&mut stream)?;
        // More than one part is what inject must not flatten; a one-part file
        // flagged multipart loses nothing when written back single-part.
        let multipart = file.parts() > 1;
        let n = i32::try_from(part)
            .map_err(|_| Error::Metadata(format!("part index {part} out of range")))?;
        let header = file.header(n)?.clone();
        let image = match header.type_()? {
            header::TILED_IMAGE => {
                let mut image = Image::from_tiled_input_file(&mut file.tiled_input_part(n)?)?;
                // The reader loads level 0 only; a ONE_LEVEL tiling is all there is, so keep it
                // and inject writes the part back tiled. MIP/RIP is refused at inject.
                let td = header.tile_description()?;
                if td.mode == LevelMode::OneLevel {
                    image.set_tiling(Some(td));
                }
                image
            }
            header::SCANLINE_IMAGE => Image::from_input_file(&mut file.input_part(n)?)?,
            _ => return Err(exr_core::ExrError::Unsupported(exr_core::Unsupported::Deep).into()),
        };
        let layers: Vec<LayerInfo> = layers.into_iter().filter(|l| l.part == part).collect();
        log::info!(
            "cryptomatte open: {path_str} part {part} layers={} size={}x{}",
            layers.len(),
            image.width(),
            image.height()
        );
        Ok(Self { path: path.to_path_buf(), part, multipart, header, layers, image })
    }

    /// Type `name`, or the first type (by name) with `None` -- the one layer
    /// choice every front end (CLI, Python) makes.
    ///
    /// # Errors
    /// [`Error::Metadata`] when the part has no Cryptomatte type, or not `name`.
    pub fn select_layer(&self, name: Option<&str>) -> Result<&LayerInfo> {
        match name {
            Some(n) => self
                .layer_by_name(n)
                .ok_or_else(|| Error::Metadata(format!("Cryptomatte layer `{n}` not found"))),
            None => self
                .layers
                .first()
                .ok_or_else(|| Error::Metadata("no Cryptomatte layers in file".into())),
        }
    }

    /// Find a layer by Cryptomatte type name (`uCryptoObject`, …).
    pub fn layer_by_name(&self, name: &str) -> Option<&LayerInfo> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Load manifest for a layer (sidecar preferred over inline).
    pub fn load_layer_manifest(&self, layer: &LayerInfo) -> Result<Manifest> {
        load_manifest(
            &self.path,
            layer.manif_file.as_deref(),
            layer.inline_manifest.as_deref(),
        )
    }

    /// Manifest for a layer, or an empty one if it is missing / unreadable.
    ///
    /// A manifest is optional (spec 1.2.0 p.3) and the reference decoder only
    /// prints a message when it cannot read one — names still resolve by
    /// hashing (Nuke `cryptomatte_utilities.py`: `name_to_ID` is
    /// `mm3hash_float(name)`), so a broken sidecar must not block extraction.
    pub fn layer_manifest_or_empty(&self, layer: &LayerInfo) -> Manifest {
        match self.load_layer_manifest(layer) {
            Ok(m) => m,
            Err(e) => {
                log::warn!(
                    "cryptomatte: layer `{}` manifest unavailable ({e}); resolving names by hash",
                    layer.name
                );
                Manifest::default()
            }
        }
    }

    /// Extract coverage for object/material names (manifest, else hashed name).
    pub fn extract_names(&self, layer: &LayerInfo, names: &[&str]) -> Result<Matte> {
        let manif = self.layer_manifest_or_empty(layer);
        let mut ids: Vec<f32> = names.iter().map(|&n| manif.id_for_name(n)).collect();
        ids.sort_by_key(|f| f.to_bits());
        ids.dedup_by_key(|f| f.to_bits());
        extract_coverage(&self.image, &layer.ranked, &ids)
    }

    /// Extract coverage for explicit float IDs.
    pub fn extract_ids(&self, layer: &LayerInfo, ids: &[f32]) -> Result<Matte> {
        extract_coverage(&self.image, &layer.ranked, ids)
    }

    /// Inject a named matte into one of this file's Cryptomatte types.
    ///
    /// The single read-all/write-all path: the payload it returns carries every
    /// other channel, both windows, the compression, a ONE_LEVEL tiling and this
    /// part's header attributes (including other Cryptomatte types), with only
    /// `layer`'s ranked channels and metadata re-encoded. Write it with
    /// [`EncodedCryptomatte::write`] (pass `None` to keep the compression).
    ///
    /// For a multipart source the write keeps every other part as raw chunks
    /// (byte-identical, never decoded) and drops `layer`'s metadata keys from
    /// their headers, so no stale manifest outlives the re-encode. The source is
    /// re-opened by path at write time, and the output appears atomically (see
    /// [`EncodedCryptomatte::write`]). The re-encoded channels keep the suffix
    /// spelling (`.red` / `.R` / `.r`) the source used for this type.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a part tiled with MIP/RIP levels.
    pub fn inject(
        &self,
        layer: &LayerInfo,
        name: &str,
        coverage: &[f32],
        op: MergeOp,
        opts: &EncodeOptions,
    ) -> Result<EncodedCryptomatte> {
        // Level 0 is all the reader loads, so a MIP/RIP pyramid would silently lose its
        // other levels on write-back: refuse it.
        if self.header.has_tile_description() {
            let mode = self.header.tile_description()?.mode;
            if mode != LevelMode::OneLevel {
                return Err(Error::Unsupported(format!(
                    "inject into part {} of {}: it is tiled with {mode:?} levels; \
                     only a ONE_LEVEL tiling can be written back without losing levels",
                    self.part,
                    self.path.display()
                )));
            }
        }
        let manifest = self.layer_manifest_or_empty(layer);
        // Re-encoding a layer in place keeps that layer's own metadata prefix
        // (a file written with the legacy `exr/cryptomatte/` keys stays legacy).
        let opts = &EncodeOptions {
            md_prefix: layer.md_prefix.clone(),
            ..opts.clone()
        };
        let mut enc = inject_named_matte(InjectArgs {
            image: &self.image,
            crypto_name: &layer.name,
            ranked: &layer.ranked,
            existing_manifest: &manifest,
            name,
            coverage,
            op,
            opts,
        })?;
        enc.carry_header_attrs(&self.header, Some(&layer.key_prefix()));
        // A tiled source part is written back tiled (the facade writes scanlines otherwise).
        enc.image.set_tiling(self.image.tiling());
        if self.multipart {
            enc.host = Some(Host {
                path: self.path.clone(),
                part: self.part,
                strip_prefix: layer.key_prefix(),
            });
        }
        Ok(enc)
    }
}

/// Discover Cryptomatte types across every part without decoding pixels.
///
/// Each [`LayerInfo::part`] names the part holding the type's ranked
/// channels. A type is listed once: when the same metadata appears in several
/// part headers, the part with its channels wins; when a part's header
/// describes a type whose channels sit in another part (no channels in its
/// own), the type is placed in the part that has them.
pub fn discover_path(path: impl AsRef<Path>) -> Result<Vec<LayerInfo>> {
    let mut stream = StdIfStream::open(utf8(path.as_ref())?)?;
    let file = MultiPartInputFile::new(&mut stream)?;
    let parts = usize::try_from(file.parts())
        .map_err(|_| Error::Metadata("negative part count".into()))?;

    let mut channels: Vec<Vec<String>> = Vec::with_capacity(parts);
    let mut found: Vec<LayerInfo> = Vec::new();
    for p in 0..parts {
        let header = file.header(p as i32)?;
        let names = channel_names_from_header(header)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        for mut layer in discover_layers(header, &refs)? {
            layer.part = p;
            found.push(layer);
        }
        channels.push(names);
    }

    // Metadata without channels in its own part: move it to the part that has them.
    for layer in found.iter_mut().filter(|l| l.ranked.is_empty() && !l.name.is_empty()) {
        if let Some((q, ranked)) = channels.iter().enumerate().find_map(|(q, names)| {
            let r = ranked_layers_for(&layer.name, names.iter().map(String::as_str));
            (!r.is_empty()).then_some((q, r))
        }) {
            layer.part = q;
            layer.ranked = ranked;
        }
    }

    // One entry per type: prefer the entry with channels, then the lowest part.
    found.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then(a.ranked.is_empty().cmp(&b.ranked.is_empty()))
            .then(a.part.cmp(&b.part))
    });
    found.dedup_by(|b, a| a.name == b.name);
    Ok(found)
}

pub(crate) fn utf8(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::Metadata(format!("path is not valid UTF-8: {}", path.display())))
}

/// Where an injected payload goes back to in a multipart source: the file it
/// was read from and the part it replaces. Set by [`CryptomatteFile::inject`].
#[derive(Clone, Debug)]
pub(crate) struct Host {
    /// Source file, re-opened at write time to raw-copy the other parts.
    pub path: PathBuf,
    /// Index of the part the payload replaces.
    pub part: usize,
    /// Metadata key prefix of the re-encoded type as the source wrote it; its
    /// attributes are dropped from the other parts' headers.
    pub strip_prefix: String,
}

impl Host {
    /// Reject an output that is the source: it is read while the output is written.
    pub fn refuse_output(&self, out: &Path) -> Result<()> {
        // An output that does not exist yet cannot be the source.
        if let (Ok(a), Ok(b)) = (std::fs::canonicalize(out), std::fs::canonicalize(&self.path))
            && a == b
        {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("output {} is the multipart source; write to another file", out.display()),
            )));
        }
        Ok(())
    }
}

/// Drop every attribute under `prefix` from `h`.
fn strip_keys(h: &mut Header, prefix: &str) -> Result<()> {
    let keys: Vec<Name> = h
        .iter()
        .filter(|(k, _)| k.as_bytes().starts_with(prefix.as_bytes()))
        .map(|(k, _)| k.clone())
        .collect();
    for k in keys {
        let name = k
            .to_str()
            .ok_or_else(|| Error::Metadata(format!("non-UTF-8 attribute name under `{prefix}`")))?;
        h.erase(name)?;
    }
    Ok(())
}

impl EncodedCryptomatte {
    /// Header of the injected part inside the multipart file: the payload's own
    /// header plus the identity attributes `carry_header_attrs` leaves to the
    /// container (`name`, `view`). The part `type` is already there: the multipart
    /// reader adds it to the payload's header (tiled or scanline, as written).
    fn spliced_header<'a, S>(&self, payload: &MultiPartInputFile<'a, S>, source: &Header) -> Result<Header>
    where
        S: std::ops::DerefMut<Target = dyn IStream + 'a>,
    {
        let mut h = payload.header(0)?.clone();
        for key in ["name", "view"] {
            if let Some((k, a)) = source.iter().find(|(k, _)| k.as_bytes() == key.as_bytes()) {
                h.insert_boxed(k.clone(), a.clone_attr())?;
            }
        }
        Ok(h)
    }

    /// [`Self::write_spliced`] into `path`, atomically: the bytes go to a sibling file that
    /// is renamed over `path` only once complete, and removed on any failure, so neither a
    /// missing or corrupt source nor a refused header leaves a truncated output behind.
    pub(crate) fn write_spliced_file(&self, host: &Host, path: &Path, compression: Compression) -> Result<()> {
        // Before anything is created: the output would truncate the source we read.
        host.refuse_output(path)?;
        // All or nothing, final flush checked: the workspace's one file writer.
        exr_core::io::write_file(path, |stream| self.write_spliced(host, stream, compression))
    }

    /// Write the multipart file: the injected part from this payload, every
    /// other part raw-copied from the host's source file (`copy_part`).
    ///
    /// The payload is first written as a small single-part file in memory (the one
    /// facade write path), whose chunks are then copied into its part like any
    /// other, so no chunk layout is reimplemented here.
    pub(crate) fn write_spliced(&self, host: &Host, out: &mut dyn OStream, compression: Compression) -> Result<()> {
        let mut mem = MemOStream::new();
        let extra = self.attrs.iter().map(|(k, a)| (k.clone(), a.clone_attr())).collect();
        self.image.write_to_with_attrs(&mut mem, compression, extra)?;
        let mut mem_in = MemIStream::new(mem.into_data());
        let mut payload = MultiPartInputFile::new(&mut mem_in)?;

        let mut src_stream = StdIfStream::open(utf8(&host.path)?)?;
        let mut src = MultiPartInputFile::new(&mut src_stream)?;
        let parts = src.parts();
        let touched = i32::try_from(host.part)
            .ok()
            .filter(|&t| t < parts)
            .ok_or_else(|| Error::Metadata(format!("part {} is not in {}", host.part, host.path.display())))?;

        let mut headers = Vec::new();
        for p in 0..parts {
            let source = src.header(p)?;
            headers.push(if p == touched {
                self.spliced_header(&payload, source)?
            } else {
                let mut h = source.clone();
                strip_keys(&mut h, &host.strip_prefix)?;
                h
            });
        }

        log::info!("cryptomatte splice: part {touched} of {parts} re-encoded, the rest raw-copied");
        let mut dst = MultiPartOutputFile::new(out, &headers)?;
        for p in 0..parts {
            if p == touched {
                copy_part(&mut payload, &mut dst, 0, p)?;
            } else {
                copy_part(&mut src, &mut dst, p, p)?;
            }
        }
        Ok(())
    }
}
