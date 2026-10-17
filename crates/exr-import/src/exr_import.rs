//! Non-EXR stills into an [`exr_core::api::Image`], and the colour tag of every frame.
//!
//! One loader for the viewer and the CLI. EXR itself never comes through [`load`] (it keeps the
//! exr-core paths: tiles, levels, deep, multipart); [`vfx_io`] decodes everything else. This crate
//! is also the one owner of the tag rules: [`load`] tags a foreign still, [`exr_tag`] an EXR
//! header, and [`builtin`] says how the viewer decodes a tag when no OCIO config is in use.
//!
//! # Pixels as read, plus a tag (`COLOR_PLAN.md`)
//!
//! A frame keeps its values as the file holds them (integers normalised to 0..1, floats as they
//! are) and a [`ColorTag`]: its colour space and where that came from. Nothing is linearised or
//! rotated into a working space here: a viewer converts once, at display, from the frame's own
//! space (one OCIO processor per frame), so a display-referred picture (a PQ HDR10 PNG, an sRGB
//! screenshot) is not rendered a second time by a scene view.
//!
//! # Foreign tags (OIIO's rule, `oiio:ColorSpace` first)
//!
//! * The reader's colour-space tag wins: PNG (`sRGB` / `gAMA` / `cICP` chunks), JPEG, BMP, GIF,
//!   WebP, Radiance HDR, PNM, DPX (transfer byte), Cineon. It is stored by its Color Interop ID
//!   when vfx-io's table knows one (`sRGB` -> `srgb_rec709_scene`), else as the reader named it
//!   (`Rec709`, `KodakLog`, `g20_rec709_scene`); the OCIO config resolves it later (names and
//!   aliases), and a name it does not know is a typed error there, never a substitute.
//! * No tag (TIFF without an EXIF block, SGI, IFF, ICO, Softimage, ...): [`TagOrigin::Untagged`],
//!   left to the config's file rules; the explicit fallback when they say nothing more than their
//!   default (or there is no config) is `srgb_rec709_display` for integer data and
//!   `lin_rec709_scene` for float data.
//!
//! # Alpha
//!
//! Whether the file's alpha is straight or associated is kept ([`Loaded::straight`]), not baked:
//! PNG, TGA, WebP and unassociated TIFF are read straight (vfx-io would premultiply them in the
//! encoded space, OIIO's default); every other format is taken as vfx-io returns it (associated,
//! OIIO's model). The viewer's shader handles both (unpremultiply, convert, premultiply).

use std::path::{Path, PathBuf};

use exr_core::header::Header;
use exr_core::api::{ChannelData, Image};
use imath_rs::{Box2i, Half, V2i};
use vfx_io::{FormatReader, ImageData, PixelData, colorspace};

/// A decoded still: the pixels as read, their colour tag, and what the Metadata panel shows.
pub struct Loaded {
    /// The channels as read (`R G B [A]`, `Y [A]`, extra data channels as named): integers
    /// normalised to 0..1, floats as stored. Never linearised, never premultiplied here.
    pub image: Image,
    /// The colour space of the colour channels and where it came from.
    pub tag: ColorTag,
    /// The alpha channel is straight (unassociated); `false` = associated, or no alpha.
    pub straight: bool,
    /// Source facts in display order: format, depth, reader attributes.
    pub meta: Vec<(String, String)>,
}

/// Why a still did not load, or a header has no usable colour facts.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// An EXR was given: it has its own reader.
    #[error("{0}: an OpenEXR file, not a foreign format")]
    Exr(PathBuf),
    /// The decoder refused the file (unreadable, malformed, format not built in).
    #[error("{path}: {source}")]
    Decode { path: String, source: vfx_io::IoError },
    /// The tag names no curve and primaries the built-in decode knows (`CMYK`, XYZ, an OCIO
    /// name such as `ACEScct`): only an OCIO config can show it.
    #[error("no built-in decode for colour space `{0}`")]
    ColorSpace(String),
    /// A colour attribute of an EXR header is malformed (`whiteLuminance` not finite and > 0).
    #[error("{0}")]
    Attr(String),
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

// -- colour tags ----------------------------------------------------------------------------------

/// The colour space of a frame's colour channels and where it came from. The space is a name an
/// OCIO config resolves (a colour space name, alias, role or Color Interop ID); the viewer maps it
/// to the config's colour space once per frame, at display.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorTag {
    /// The colour space; for [`TagOrigin::Untagged`] the explicit fallback (see there).
    pub space: String,
    /// Where `space` came from.
    pub origin: TagOrigin,
}

/// Where a [`ColorTag`] came from, in the order the tag rules try them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TagOrigin {
    /// The file names it: `oiio:ColorSpace` of a foreign still (PNG chunks, DPX transfer, ...) or
    /// the EXR `colorInteropID` attribute (OpenEXR `exrstdattr.rst`).
    File,
    /// EXR `chromaticities`: matched to a `lin_<primaries>_scene` interop ID (`to709` = `None`),
    /// or primaries no ID names, taken as they are: `space` is `lin_rec709_scene` and `to709`
    /// carries the pixels there (Bradford to D65). Never a guess, never dropped.
    Chromaticities { to709: Option<[[f32; 3]; 3]> },
    /// EXR `whiteLuminance` without `colorInteropID`: by OpenEXR's definition the nits of RGB
    /// (1, 1, 1), so with the primaries the pixels are colorimetric CIE XYZ, display light whose
    /// white is `nits`. `space` is the config's `cie_xyz_d65_interchange` role (OCIO's display
    /// reference) and `to_xyz` takes the pixels there (primaries, Bradford to D65) with RGB 1 at
    /// the reference's 1.0, the white of a relative (SDR) display; an absolute (HDR) output shows
    /// that 1.0 at `nits` (`Builtin::nits`, the viewer's display gain). So a picture saved as
    /// display light at its SDR white reopens as it was on both kinds of display.
    Luminance { to_xyz: [[f32; 3]; 3], nits: f32 },
    /// The file says nothing. The config's file rules decide; when the path matches only their
    /// default rule (or there is no config) `space` is used: `srgb_rec709_display` for integer
    /// stills, `lin_rec709_scene` for float stills, `scene_linear` for EXR.
    Untagged,
}

impl ColorTag {
    /// The tag of an EXR without colour attributes (and of frames built in tests): untagged,
    /// `scene_linear` after the file rules.
    pub fn scene_linear() -> Self {
        Self { space: colorspace::SCENE_LINEAR.to_owned(), origin: TagOrigin::Untagged }
    }
}

impl TagOrigin {
    /// How the Metadata panel and the status bar name it.
    pub fn label(&self) -> &'static str {
        match self {
            Self::File => "the file's tag",
            Self::Chromaticities { .. } => "EXR chromaticities",
            Self::Luminance { .. } => "EXR whiteLuminance (absolute light)",
            Self::Untagged => "untagged",
        }
    }
}

/// The interop-ID tokens of the primaries an EXR `chromaticities` attribute is matched against.
const EXR_PRIMARIES: [(&str, &vfx_primaries::Primaries); 5] = [
    ("ap0", &vfx_primaries::ACES_AP0),
    ("ap1", &vfx_primaries::ACES_AP1),
    ("rec709", &vfx_primaries::REC709),
    ("rec2020", &vfx_primaries::REC2020),
    ("p3d65", &vfx_primaries::DISPLAY_P3),
];

/// How close (CIE xy) a stored chromaticity must be to a known one: a file rounds its floats, but
/// distinct standard primaries are far further apart than this.
const XY_TOLERANCE: f32 = 1e-3;

/// The OCIO role of OCIO's display reference, linear CIE XYZ D65 with 1.0 = 100 nits.
pub const XYZ_DISPLAY_ROLE: &str = "cie_xyz_d65_interchange";

fn primaries_of_exr(c: &exr_core::Chromaticities) -> vfx_primaries::Primaries {
    vfx_primaries::Primaries {
        r: (c.red[0], c.red[1]),
        g: (c.green[0], c.green[1]),
        b: (c.blue[0], c.blue[1]),
        w: (c.white[0], c.white[1]),
        name: "EXR chromaticities",
    }
}

/// The tag of one EXR part (`exrinfo.rst`: colour attributes are per part): `colorInteropID`
/// first; else `whiteLuminance` (display light, [`TagOrigin::Luminance`]); else `chromaticities`
/// (matched to AP0 / AP1 / Rec.709 / Rec.2020 / P3-D65, or other primaries by their own matrix);
/// else untagged (`scene_linear` after the file rules).
pub fn exr_tag(header: &Header) -> Result<ColorTag, ImportError> {
    let attr = |e: exr_core::ExrError| ImportError::Attr(e.to_string());
    if header.has_color_interop_id() {
        let id = String::from_utf8_lossy(header.color_interop_id().map_err(attr)?.as_bytes()).trim().to_owned();
        if !id.is_empty() {
            return Ok(ColorTag { space: id, origin: TagOrigin::File });
        }
    }
    let chroma = if header.has_chromaticities() { Some(*header.chromaticities().map_err(attr)?) } else { None };
    if header.has_white_luminance() {
        let nits = *header.white_luminance().map_err(attr)?;
        if !(nits.is_finite() && nits > 0.0) {
            return Err(ImportError::Attr(format!("whiteLuminance {nits} must be finite and positive")));
        }
        // No chromaticities means Rec.709, as OpenEXR's `Chromaticities()` default.
        let prim = primaries_of_exr(&chroma.unwrap_or_default());
        if vfx_primaries::try_rgb_to_xyz_matrix(&prim).is_none() {
            return Err(ImportError::Attr(format!("chromaticities {:?} are degenerate", chroma.unwrap_or_default())));
        }
        let rec709 = &vfx_primaries::REC709;
        let to_xyz = vfx_primaries::rgb_to_xyz_matrix(rec709) * vfx_primaries::rgb_to_rgb_matrix(&prim, rec709);
        return Ok(ColorTag { space: XYZ_DISPLAY_ROLE.to_owned(), origin: TagOrigin::Luminance { to_xyz: to_xyz.m, nits } });
    }
    if let Some(c) = chroma {
        let near = |a: (f32, f32), b: [f32; 2]| (a.0 - b[0]).abs() <= XY_TOLERANCE && (a.1 - b[1]).abs() <= XY_TOLERANCE;
        let hit = EXR_PRIMARIES
            .iter()
            .find(|(_, p)| near(p.r, c.red) && near(p.g, c.green) && near(p.b, c.blue) && near(p.w, c.white));
        if let Some((tok, _)) = hit {
            return Ok(ColorTag { space: format!("lin_{tok}_scene"), origin: TagOrigin::Chromaticities { to709: None } });
        }
        // Primaries no interop ID names: their own matrix into linear Rec.709.
        let prim = primaries_of_exr(&c);
        if vfx_primaries::try_rgb_to_xyz_matrix(&prim).is_none() {
            return Err(ImportError::Attr(format!("chromaticities {c:?} are degenerate")));
        }
        let to709 = vfx_primaries::rgb_to_rgb_matrix(&prim, &vfx_primaries::REC709).m;
        return Ok(ColorTag { space: colorspace::LIN_REC709_SCENE.to_owned(), origin: TagOrigin::Chromaticities { to709: Some(to709) } });
    }
    Ok(ColorTag::scene_linear())
}

/// The tag of a decoded foreign still (see the module docs).
fn foreign_tag(img: &ImageData) -> ColorTag {
    match img.metadata.colorspace() {
        Some(name) => {
            let space = colorspace::interop_id(name).filter(|id| id != "unknown").unwrap_or_else(|| name.to_owned());
            ColorTag { space, origin: TagOrigin::File }
        }
        None => {
            let integer = matches!(img.data, PixelData::U8(_) | PixelData::U16(_));
            let space = if integer { "srgb_rec709_display" } else { colorspace::LIN_REC709_SCENE };
            ColorTag { space: space.to_owned(), origin: TagOrigin::Untagged }
        }
    }
}

// -- the built-in decode (no OCIO config) ---------------------------------------------------------

/// A decode curve the viewer's built-in path applies to a frame's stored values (its tile shader
/// and CPU twin evaluate it; this crate only names it). The light it yields is relative (1.0 =
/// the display's white) for SDR curves and absolute for [`Tf::Pq`] / [`Tf::Hlg`]: OCIO's display
/// reference, 1.0 = 100 nits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tf {
    Linear,
    /// The piecewise sRGB EOTF.
    Srgb,
    /// BT.709 camera OETF inverse (OIIO `Rec709`).
    Rec709,
    /// Pure power law (`g22_rec709_display`, `g24_rec709_display` = BT.1886 with a zero black,
    /// DPX user gamma).
    Gamma(f32),
    /// Kodak Cineon printing-density log: 685 of 1023 codes maps to 1, 95 to 0 (density step
    /// 0.002, gamma 0.6). `KodakLog` is a bare name in OIIO; this is the standard curve.
    Cineon,
    /// SMPTE ST 2084, in units of 100 nits.
    Pq,
    /// BT.2100 HLG on its 1000-nit reference display (inverse OETF, then the OOTF with system
    /// gamma 1.2 on the BT.2020 luminance), in units of 100 nits: OCIO's `Rec.2100-HLG - Display`.
    Hlg,
}

/// How the built-in path decodes a tag: the curve, then a linear-RGB matrix into Rec.709 (the
/// canvas primaries; identity for Rec.709 sources), and, for absolute light, the nits of its 1.0
/// (shown at those nits on an HDR output) rather than relative to SDR white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Builtin {
    pub tf: Tf,
    /// Row-major linear-RGB matrix into Rec.709 primaries (Bradford when the white points differ).
    pub to709: [[f32; 3]; 3],
    /// The nits of light 1.0 when the light is absolute (PQ / HLG: 100, the decode's unit; an EXR
    /// with `whiteLuminance`: that); `None` = relative, 1.0 is SDR white.
    pub nits: Option<f32>,
}

/// The identity matrix.
pub const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl Builtin {
    /// Linear Rec.709, relative: the built-in path's own reference.
    pub const LINEAR: Builtin = Builtin { tf: Tf::Linear, to709: IDENTITY, nits: None };
}

/// Transfer token of a Color Interop Forum ID (`<transfer>_<primaries>_<scene|display>`).
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

/// Curve and primaries of a space name, through vfx-io's colour-space table (the one owner of
/// OIIO's names and aliases); `None` when the table does not know the name or it is not RGB.
fn curve_of(name: &str) -> Option<(Tf, &'static vfx_primaries::Primaries)> {
    let is = |canon: &str| colorspace::equivalent(name, canon);
    let rec709 = &vfx_primaries::REC709;
    // Without a config the built-in reference is linear Rec.709, so is the untagged EXR's role.
    if name.eq_ignore_ascii_case(colorspace::SCENE_LINEAR) || is(colorspace::LIN_REC709_SCENE) {
        return Some((Tf::Linear, rec709));
    } else if is(colorspace::SRGB_REC709_SCENE) {
        return Some((Tf::Srgb, rec709));
    } else if is(colorspace::REC709) {
        return Some((Tf::Rec709, rec709));
    } else if is(colorspace::KODAK_LOG) {
        return Some((Tf::Cineon, rec709));
    } else if let Some(g) = colorspace::rec709_gamma_of(name, None) {
        return Some((Tf::Gamma(g), rec709));
    }
    // Everything else by its interop ID, which spells both halves out.
    let id = colorspace::interop_id(name)?;
    let mut parts = id.split('_');
    Some((tf_of(parts.next()?)?, primaries_of(parts.next()?)?))
}

/// The built-in decode of `tag`: what the viewer shows without an OCIO config. A tag with no
/// curve here (an OCIO-only name, XYZ, CMYK) is a typed error, never a guess.
pub fn builtin(tag: &ColorTag) -> Result<Builtin, ImportError> {
    if let TagOrigin::Chromaticities { to709: Some(to709) } = tag.origin {
        return Ok(Builtin { tf: Tf::Linear, to709, nits: None });
    }
    if let TagOrigin::Luminance { to_xyz, nits } = tag.origin {
        let m = vfx_primaries::xyz_to_rgb_matrix(&vfx_primaries::REC709) * vfx_primaries::Mat3 { m: to_xyz };
        return Ok(Builtin { tf: Tf::Linear, to709: m.m, nits: Some(nits) });
    }
    let (tf, prim) = curve_of(&tag.space).ok_or_else(|| ImportError::ColorSpace(tag.space.clone()))?;
    // Out-of-gamut results stay negative: the canvas keeps them on an HDR output.
    let to709 = if *prim == vfx_primaries::REC709 { IDENTITY } else { vfx_primaries::rgb_to_rgb_matrix(prim, &vfx_primaries::REC709).m };
    Ok(Builtin { tf, to709, nits: matches!(tf, Tf::Pq | Tf::Hlg).then_some(100.0) })
}

fn is_straight(img: &ImageData) -> bool {
    img.metadata.attrs.get("oiio:UnassociatedAlpha").and_then(|v| v.as_u32()).is_some_and(|v| v != 0)
}

/// The decoded planes as read, one per channel: integers normalised to 0..1, floats as stored.
fn planes(img: &ImageData) -> Vec<Vec<f32>> {
    let c = img.channels as usize;
    let n = img.pixel_count();
    let mut out: Vec<Vec<f32>> = (0..c).map(|_| Vec::with_capacity(n)).collect();
    if c == 0 || n == 0 {
        return out;
    }
    match &img.data {
        PixelData::U8(buf) => {
            let scale = 1.0 / 255.0;
            for px in buf.chunks_exact(c).take(n) {
                for (plane, &v) in out.iter_mut().zip(px) {
                    plane.push(f32::from(v) * scale);
                }
            }
        }
        PixelData::U16(buf) => {
            let scale = 1.0 / 65535.0;
            for px in buf.chunks_exact(c).take(n) {
                for (plane, &v) in out.iter_mut().zip(px) {
                    plane.push(f32::from(v) * scale);
                }
            }
        }
        PixelData::U32(buf) => {
            let scale = 1.0 / 4294967295.0f64;
            for px in buf.chunks_exact(c).take(n) {
                for (plane, &v) in out.iter_mut().zip(px) {
                    plane.push((f64::from(v) * scale) as f32);
                }
            }
        }
        PixelData::F32(buf) => {
            for px in buf.chunks_exact(c).take(n) {
                for (plane, &v) in out.iter_mut().zip(px) {
                    plane.push(v);
                }
            }
        }
        _ => {
            for p in 0..n {
                for (ch, plane) in out.iter_mut().enumerate() {
                    plane.push(img.sample_f32(p * c + ch));
                }
            }
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
    let tag = foreign_tag(&img);
    let spec = img.to_spec();
    let names = spec.channel_names.clone();
    let alpha = spec.alpha_channel >= 0;
    let (w, h) = (img.width as i32, img.height as i32);
    if names.len() != img.channels as usize || w == 0 || h == 0 {
        return Err(ImportError::Pixels(format!("{}: {}x{} with {} channel name(s) for {} channel(s)", path.display(), w, h, names.len(), img.channels)));
    }
    // 8-bit sources keep half the memory: half's 11 bits hold every code k/255 to well under half
    // a code step.
    let half = matches!(img.data, PixelData::U8(_));
    let mut out = Image::new(Box2i::from_min_max(V2i::new(0, 0), V2i::new(w - 1, h - 1)));
    for (name, plane) in names.iter().zip(planes(&img)) {
        let data = if half { ChannelData::Half(plane.into_iter().map(Half::from).collect()) } else { ChannelData::Float(plane) };
        out.insert_channel(name.clone(), data).map_err(|e| ImportError::Pixels(e.to_string()))?;
    }

    let straight = alpha && is_straight(&img);
    // The tag and the alpha kind are the frame's own fields ([`Loaded::tag`], [`Loaded::straight`]);
    // the viewer shows them with the resolved colour space, so they are not repeated here.
    let mut meta = vec![
        ("Source format".to_owned(), vfx_io::Format::detect(path).map(|f| format!("{f:?}")).unwrap_or_default()),
        ("Source depth".to_owned(), format!("{:?}", img.format)),
    ];
    for (k, v) in img.metadata.attrs.iter_flat() {
        let v = v.to_string();
        if v.len() <= 200 {
            meta.push((k.to_string(), v));
        }
    }
    Ok(Loaded { image: out, tag, straight, meta })
}

impl Loaded {
    /// The still as an EXR image, as `oiiotool in.png -o out.exr` writes it: the values as read
    /// (no colour conversion), alpha associated (straight colour multiplied by alpha in the stored
    /// encoding, OpenEXR's alpha model), and the tag as the standard `colorInteropID` attribute
    /// when the file named it (an untagged still stays untagged, so a reader's file rules decide).
    pub fn to_exr(self) -> Result<Image, ImportError> {
        let Loaded { image, tag, straight, .. } = self;
        let mut out = Image::new(image.data_window);
        let alpha = image.channel("A").map(ChannelData::to_f32);
        let pixels = |e: exr_core::ExrError| ImportError::Pixels(e.to_string());
        for (name, data) in image.into_channels().map_err(pixels)? {
            let data = match (&alpha, straight && matches!(name.as_str(), "R" | "G" | "B" | "Y")) {
                (Some(a), true) => {
                    let v: Vec<f32> = data.to_f32().iter().zip(a).map(|(c, a)| c * a).collect();
                    match data {
                        ChannelData::Half(_) => ChannelData::Half(v.into_iter().map(Half::from).collect()),
                        _ => ChannelData::Float(v),
                    }
                }
                _ => data,
            };
            out.insert_channel(name, data).map_err(pixels)?;
        }
        if tag.origin == TagOrigin::File {
            let id = exr_core::attr::ExrString::from(tag.space.as_str());
            out.attributes_mut()
                .insert("colorInteropID", Box::new(exr_core::attr::TypedAttribute::new(id)))
                .map_err(pixels)?;
        }
        Ok(out)
    }
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
