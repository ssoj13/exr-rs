//! Non-EXR stills into an [`exr_core::api::Image`]: scene-linear, associated alpha.
//!
//! One loader for the viewer and the CLI. EXR itself never comes through here (it keeps the
//! exr-core paths: tiles, levels, deep, multipart); [`vfx_io`] decodes everything else, and this
//! crate turns what it returns into what the rest of the workspace assumes of a frame.
//!
//! # Colour (OIIO's rule, `oiio:ColorSpace` first)
//!
//! * The reader's colour-space tag wins (`srgb_rec709_scene`, `lin_rec709_scene`,
//!   `g<NN>_rec709_scene`, `Rec709`, `KodakLog`): PNG/JPEG/BMP/GIF/WebP tag sRGB (PNG honours its
//!   `sRGB`/`gAMA` chunks), Radiance HDR is linear, PNM is `Rec709` (OIIO `pnminput.cpp:404`,
//!   PFM included), DPX maps its transfer byte (linear, log, ITU-709, user gamma).
//! * No tag: integer data is sRGB, float data is linear (the formats that never tag: TIFF without
//!   an EXIF block, SGI, IFF, ICO, Softimage, DPX printing-density/video codes, RLA without gamma).
//! * `KodakLog` (Cineon, DPX log) is a bare name in OIIO: the conversion belongs to the OCIO
//!   config. Here it is the standard Kodak printing-density curve (white 685, black 95, density
//!   step 0.002, gamma 0.6 of 1023 codes), see [`Tf::Cineon`].
//! * Any other tag goes through vfx-io's colour-space table (Color Interop Forum IDs and OIIO's
//!   aliases, e.g. `g24_rec709_display`, `lin_ap1_scene`, `pq_rec2020_display`): the ID names a
//!   transfer curve and primaries; primaries other than Rec.709 are rotated into linear Rec.709
//!   (Bradford when the white points differ), negatives kept.
//! * A tag the table does not know, or that is not RGB (`CMYK`, XYZ), is a typed error.
//!
//! # Alpha
//!
//! The viewer's canvas contract is encoded ASSOCIATED light. PNG, TGA, WebP and unassociated TIFF
//! are straight on disk; vfx-io would premultiply them in the encoded space (OIIO's default),
//! which is wrong once the colour is linearised, so they are read with `unassociated_alpha` and
//! multiplied here, in linear light. Every other format is taken as vfx-io returns it (OIIO's
//! model: associated, even where the file does not say so): its colour is divided by alpha,
//! linearised, and multiplied back.

use std::path::{Path, PathBuf};

use exr_core::api::{ChannelData, Image};
use imath_rs::{Box2i, Half, V2i};
use vfx_io::{FormatReader, ImageData, PixelData, colorspace};

/// A decoded still: the pixels plus what the Metadata panel shows of the source.
pub struct Loaded {
    /// Linear, associated-alpha channels (`R G B [A]`, `Y [A]`, extra data channels as named).
    pub image: Image,
    /// Source facts in display order: format, colour space, alpha handling, reader attributes.
    pub meta: Vec<(String, String)>,
}

/// Why a still did not load.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// An EXR was given: it has its own reader.
    #[error("{0}: an OpenEXR file, not a foreign format")]
    Exr(PathBuf),
    /// The decoder refused the file (unreadable, malformed, format not built in).
    #[error("{path}: {source}")]
    Decode { path: String, source: vfx_io::IoError },
    /// The reader tagged a colour space this loader has no curve for.
    #[error("unsupported colour space `{0}`")]
    ColorSpace(String),
    /// The pixels do not fit the facade (no channels, size mismatch).
    #[error("{0}")]
    Pixels(String),
}

// -- the format registry (the one list of what the viewer can open) ------------------------------

/// One openable format family: what the Open dialog filters by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Family {
    /// Stable key (the persisted filter): `"EXR"` or a vfx-io format name.
    pub key: String,
    /// Dialog label, `PNG (*.png)`.
    pub label: String,
    /// Lower-case extensions, no dot.
    pub exts: Vec<String>,
}

/// EXR extensions: `.sxr` is a stereo EXR some tools write.
const EXR_EXTS: [&str; 2] = ["exr", "sxr"];

/// Every family: EXR first, then the registered non-EXR formats by name (the registry is a hash
/// map; the order must not depend on it).
pub fn families() -> Vec<Family> {
    let reg = vfx_io::FormatRegistry::global();
    let mut names: Vec<&str> = reg.format_names().collect();
    names.sort_unstable();
    let mut out = vec![family("EXR", EXR_EXTS.iter().map(|e| (*e).to_owned()).collect())];
    for name in names {
        let Some(info) = reg.get(name) else { continue };
        let exts: Vec<String> = info
            .extensions
            .iter()
            .map(|e| e.to_ascii_lowercase())
            .filter(|e| !EXR_EXTS.contains(&e.as_str()) && e != "tx")
            .collect();
        if !exts.is_empty() {
            out.push(family(name, exts));
        }
    }
    out
}

fn family(key: &str, exts: Vec<String>) -> Family {
    let glob: Vec<String> = exts.iter().map(|e| format!("*.{e}")).collect();
    Family { key: key.to_owned(), label: format!("{key} ({})", glob.join(", ")), exts }
}

fn ext_of(path: &Path) -> Option<String> {
    path.extension()?.to_str().map(str::to_ascii_lowercase)
}

/// Whether `path` is an EXR by extension.
pub fn is_exr(path: &Path) -> bool {
    ext_of(path).is_some_and(|e| EXR_EXTS.contains(&e.as_str()))
}

/// Whether the viewer can open `path`, by extension: EXR or a registered foreign format.
pub fn is_image(path: &Path) -> bool {
    is_exr(path) || is_foreign(path)
}

/// Whether `path` is a foreign (non-EXR) format [`load`] reads, by extension.
pub fn is_foreign(path: &Path) -> bool {
    ext_of(path).is_some_and(|e| e != "tx" && vfx_io::FormatRegistry::global().supports_extension(&e))
}

// -- colour ---------------------------------------------------------------------------------------

/// The decode curve of a source colour space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tf {
    Linear,
    Srgb,
    /// BT.709 camera OETF inverse (OIIO `Rec709`).
    Rec709,
    /// Pure power law (`g22_rec709_scene`, DPX user gamma).
    Gamma(f32),
    /// Kodak Cineon printing-density log. `lin = (10^((code - 685) * 0.002 / 0.6) - k) / (1 - k)`
    /// with `k = 10^((95 - 685) * 0.002 / 0.6)`, `code = v * 1023`: 685 maps to 1, 95 to 0.
    Cineon,
    /// SMPTE ST 2084, 100 cd/m2 = 1.0.
    Pq,
    /// BT.2100 HLG inverse OETF (scene light, no display OOTF).
    Hlg,
}

impl Tf {
    /// Encoded value `v` to scene-linear.
    pub fn lin(self, v: f32) -> f32 {
        match self {
            Tf::Linear => v,
            Tf::Srgb => vfx_transfer::srgb::eotf(v),
            Tf::Rec709 => vfx_transfer::rec709::eotf(v),
            Tf::Gamma(g) => vfx_transfer::gamma::gamma_eotf(v, g),
            Tf::Pq => vfx_transfer::pq::eotf_normalized(v),
            Tf::Hlg => vfx_transfer::hlg::eotf(v),
            Tf::Cineon => {
                let k = 10f32.powf((95.0 - 685.0) * 0.002 / 0.6);
                (10f32.powf((v * 1023.0 - 685.0) * 0.002 / 0.6) - k) / (1.0 - k)
            }
        }
    }
}

/// A source colour space reduced to what the loader applies: a decode curve and, for primaries
/// other than Rec.709, the matrix into linear Rec.709.
#[derive(Clone, Debug, PartialEq)]
pub struct Space {
    pub tf: Tf,
    /// Row-major linear-RGB matrix into Rec.709 primaries; `None` when the source already is.
    pub to709: Option<[[f32; 3]; 3]>,
    /// The tag as shown in the Metadata panel.
    pub name: String,
}

/// Transfer token of a Color Interop Forum ID (`<transfer>_<primaries>_<scene|display>`).
///
/// A `*_display` ID is linearised by inverting the display encoding (its EOTF), like OIIO's
/// colorconvert to `lin_rec709_scene` does for the encoding itself. The ACES output transform
/// (tone map, gamut compression) a studio config puts between display and scene is NOT inverted:
/// a still tagged this way is a plain encoded picture, not an ACES rendering.
fn tf_of(tok: &str) -> Option<Tf> {
    Some(match tok {
        "lin" => Tf::Linear,
        "srgb" | "srgbe" => Tf::Srgb,
        "g18" => Tf::Gamma(1.8),
        "g22" => Tf::Gamma(2.2),
        "g24" => Tf::Gamma(2.4),
        "g26" => Tf::Gamma(2.6),
        "pq" => Tf::Pq,
        "hlg" => Tf::Hlg,
        _ => return None,
    })
}

/// Primaries token of a Color Interop Forum ID. XYZ and `data`/`unknown` have no RGB primaries.
fn primaries_of(tok: &str) -> Option<&'static vfx_primaries::Primaries> {
    use vfx_primaries as p;
    Some(match tok {
        "rec709" => &p::REC709,
        "p3d65" => &p::DISPLAY_P3,
        "rec2020" => &p::REC2020,
        "ap0" => &p::ACES_AP0,
        "ap1" => &p::ACES_AP1,
        "adobergb" => &p::ADOBE_RGB,
        _ => return None,
    })
}

/// Curve and primaries of a tag, resolved through vfx-io's colour-space table (the one owner of
/// OIIO's names and aliases); `None` when the table does not know the name or it is not RGB.
fn resolve(name: &str, gamma: Option<f32>) -> Option<(Tf, &'static vfx_primaries::Primaries)> {
    let is = |canon: &str| colorspace::equivalent(name, canon);
    let rec709 = &vfx_primaries::REC709;
    // The names OIIO's readers set that carry no interop ID, or a gamma only `oiio:Gamma` knows.
    if is(colorspace::SRGB_REC709_SCENE) {
        return Some((Tf::Srgb, rec709));
    } else if is(colorspace::LIN_REC709_SCENE) {
        return Some((Tf::Linear, rec709));
    } else if is(colorspace::REC709) {
        return Some((Tf::Rec709, rec709));
    } else if is(colorspace::KODAK_LOG) {
        return Some((Tf::Cineon, rec709));
    } else if let Some(g) = colorspace::rec709_gamma_of(name, gamma) {
        return Some((Tf::Gamma(g), rec709));
    }
    // Everything else by its interop ID, which spells both halves out.
    let id = colorspace::interop_id(name)?;
    let mut parts = id.split('_');
    Some((tf_of(parts.next()?)?, primaries_of(parts.next()?)?))
}

/// The [`Space`] of `img` (`unset` rule included).
pub fn space_of(img: &ImageData) -> Result<Space, ImportError> {
    let integer = matches!(img.data, PixelData::U8(_) | PixelData::U16(_));
    let Some(name) = img.metadata.colorspace() else {
        let (tf, name) = if integer { (Tf::Srgb, "sRGB (untagged integer)") } else { (Tf::Linear, "linear (untagged float)") };
        return Ok(Space { tf, to709: None, name: name.into() });
    };
    let (tf, prim) = resolve(name, img.metadata.gamma()).ok_or_else(|| ImportError::ColorSpace(name.to_owned()))?;
    // Bradford-adapted when the white points differ (AP0/AP1 are D60); out-of-gamut results stay
    // negative, the scene-linear canvas keeps them.
    let to709 = (*prim != vfx_primaries::REC709).then(|| vfx_primaries::rgb_to_rgb_matrix(prim, &vfx_primaries::REC709).m);
    Ok(Space { tf, to709, name: name.to_owned() })
}

/// Rotate the linear `R G B` planes by `m` (associated alpha commutes with a linear map).
fn rotate(planes: &mut [Vec<f32>], names: &[String], m: &[[f32; 3]; 3]) {
    let at = |n: &str| names.iter().position(|s| s == n);
    let (Some(r), Some(g), Some(b)) = (at("R"), at("G"), at("B")) else { return };
    // Three disjoint mutable planes out of the slice, in R G B order.
    let mut rgb: [Option<&mut Vec<f32>>; 3] = [None, None, None];
    for (i, plane) in planes.iter_mut().enumerate() {
        if let Some(k) = [r, g, b].iter().position(|&c| c == i) {
            rgb[k] = Some(plane);
        }
    }
    let [Some(pr), Some(pg), Some(pb)] = rgb else { return };
    for ((r, g), b) in pr.iter_mut().zip(pg.iter_mut()).zip(pb.iter_mut()) {
        let v = [*r, *g, *b];
        let out = m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
        (*r, *g, *b) = (out[0], out[1], out[2]);
    }
}

/// Decode table for 8/16-bit sources: one curve evaluation per code, not per sample.
fn table(data: &PixelData, tf: Tf) -> Option<Vec<f32>> {
    let max = match data {
        PixelData::U8(_) => 255usize,
        PixelData::U16(_) => 65535,
        _ => return None,
    };
    Some((0..=max).map(|i| tf.lin(i as f32 / max as f32)).collect())
}

fn is_straight(img: &ImageData) -> bool {
    img.metadata.attrs.get("oiio:UnassociatedAlpha").and_then(|v| v.as_u32()).is_some_and(|v| v != 0)
}

/// The decoded planes: linearised colour, associated alpha, data channels untouched.
fn planes(img: &ImageData, tf: Tf, names: &[String], alpha: Option<usize>) -> Vec<Vec<f32>> {
    let c = img.channels as usize;
    let n = img.pixel_count();
    let straight = is_straight(img);
    let lut = table(&img.data, tf);
    let colour: Vec<bool> = names.iter().map(|s| matches!(s.as_str(), "R" | "G" | "B" | "Y")).collect();
    // The table serves a code as stored; a division by alpha leaves the codes.
    let lin = |i: usize, div: f32| match (&lut, &img.data) {
        (Some(t), PixelData::U8(d)) if div == 1.0 => t[d[i] as usize],
        (Some(t), PixelData::U16(d)) if div == 1.0 => t[d[i] as usize],
        _ => tf.lin(img.sample_f32(i) / div),
    };
    let mut out = vec![Vec::with_capacity(n); c];
    for p in 0..n {
        let base = p * c;
        let a = alpha.map(|k| img.sample_f32(base + k));
        for (ch, plane) in out.iter_mut().enumerate() {
            let i = base + ch;
            plane.push(if !colour[ch] {
                img.sample_f32(i)
            } else {
                match a {
                    None => lin(i, 1.0),
                    Some(a) if straight => lin(i, 1.0) * a,
                    Some(a) if a > 0.0 => lin(i, a) * a,
                    Some(_) => lin(i, 1.0),
                }
            });
        }
    }
    out
}

// -- loading --------------------------------------------------------------------------------------

/// Decode `path` with vfx-io. PNG, TIFF, TGA and WebP are asked for straight alpha (see the
/// module docs); the rest go through the registry, format picked by magic bytes like vfx-io does.
fn decode(path: &Path) -> Result<ImageData, ImportError> {
    use vfx_io::{Format, png, targa, tiff, webp};
    let bad = |source| ImportError::Decode { path: path.display().to_string(), source };
    match Format::detect(path).map_err(bad)? {
        Format::Exr | Format::Tx => Err(ImportError::Exr(path.to_path_buf())),
        Format::Png => png::PngReader::with_options(png::PngReaderOptions { unassociated_alpha: true }).read(path).map_err(bad),
        Format::Tiff => tiff::TiffReader::with_options(tiff::TiffReaderOptions { unassociated_alpha: true, ..Default::default() })
            .read(path)
            .map_err(bad),
        Format::Targa => targa::TgaReader::with_options(targa::TgaReaderOptions { unassociated_alpha: true }).read(path).map_err(bad),
        Format::WebP => webp::WebpReader::with_options(webp::WebpReaderOptions { unassociated_alpha: true }).read(path).map_err(bad),
        _ => vfx_io::read(path).map_err(bad),
    }
}

/// Load a foreign still: first frame/page of multi-image files (GIF, ICO, TIFF, WebP).
pub fn load(path: &Path) -> Result<Loaded, ImportError> {
    let img = decode(path)?;
    let Space { tf, to709, name: space } = space_of(&img)?;
    let spec = img.to_spec();
    let names = spec.channel_names.clone();
    let alpha = usize::try_from(spec.alpha_channel).ok();
    let (w, h) = (img.width as i32, img.height as i32);
    if names.len() != img.channels as usize || w == 0 || h == 0 {
        return Err(ImportError::Pixels(format!("{}: {}x{} with {} channel name(s) for {} channel(s)", path.display(), w, h, names.len(), img.channels)));
    }
    let mut planes = planes(&img, tf, &names, alpha);
    if let Some(m) = &to709 {
        rotate(&mut planes, &names, m);
    }
    // 8-bit sources keep half the memory: half's 11 bits beat a linear 8-bit step except in the
    // deepest shadows, where the step itself is far below visibility.
    let half = matches!(img.data, PixelData::U8(_));
    let mut out = Image::new(Box2i::from_min_max(V2i::new(0, 0), V2i::new(w - 1, h - 1)));
    for (name, plane) in names.iter().zip(planes) {
        let data = if half { ChannelData::Half(plane.into_iter().map(Half::from).collect()) } else { ChannelData::Float(plane) };
        out.insert_channel(name.clone(), data).map_err(|e| ImportError::Pixels(e.to_string()))?;
    }

    let straight = is_straight(&img);
    let mut meta = vec![
        ("Source format".to_owned(), vfx_io::Format::detect(path).map(|f| format!("{f:?}")).unwrap_or_default()),
        ("Source colour space".to_owned(), space),
        ("Source depth".to_owned(), format!("{:?}", img.format)),
    ];
    if alpha.is_some() {
        let how = if straight { "straight, premultiplied in linear light" } else { "associated as stored" };
        meta.push(("Source alpha".to_owned(), how.to_owned()));
    }
    for (k, v) in img.metadata.attrs.iter_flat() {
        let v = v.to_string();
        if v.len() <= 200 {
            meta.push((k.to_string(), v));
        }
    }
    Ok(Loaded { image: out, meta })
}

/// A header-level summary for `exr-rs info` (size, channels, format) without decoding the pixels.
pub fn probe(path: &Path) -> Result<Vec<(String, String)>, ImportError> {
    let bad = |source| ImportError::Decode { path: path.display().to_string(), source };
    let format = vfx_io::Format::detect(path).map_err(bad)?;
    if matches!(format, vfx_io::Format::Exr | vfx_io::Format::Tx) {
        return Err(ImportError::Exr(path.to_path_buf()));
    }
    let (w, h, c) = vfx_io::probe_image_info(path).map_err(bad)?;
    Ok(vec![
        ("format".into(), format!("{format:?}")),
        ("size".into(), format!("{w}x{h}")),
        ("channels".into(), c.to_string()),
    ])
}

#[cfg(test)]
mod tests;
