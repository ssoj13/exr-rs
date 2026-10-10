//! P7 legacy byte-channel API (`OpenEXR.InputFile` / `OutputFile` / `Header`) — the
//! deprecated, pre-numpy interface that reads/writes channels as raw `bytes` and
//! uses the companion `Imath` types in its header dict.
//!
//! Scope (this slice): `Header(w, h)`, `OutputFile.writePixels` (single full-image
//! call), `InputFile.channel`/`channels` with pixel-type conversion and scanline
//! slicing. The full Imath-typed CUSTOM-attribute round-trip and chunked
//! `writePixels` are the remaining legacy work.

use exr_core::ChannelData;
use exr_core::api::Image;
use exr_core::attr::{
    Attribute, Box2fAttribute, Box2iAttribute, Chromaticities, ChromaticitiesAttribute,
    Compression, CompressionAttribute, KeyCode as CoreKeyCode, KeyCodeAttribute, LineOrder,
    LineOrderAttribute, PixelType, Rational, RationalAttribute, TileDescription as CoreTileDescription,
    TileDescriptionAttribute, TimeCode as CoreTimeCode, TimeCodeAttribute, V2fAttribute, V2iAttribute,
};
use exr_core::header::Header;
use imath_rs::{Box2f, Box2i, Half, V2f, V2i};
use pyo3::exceptions::{PyIOError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyString};
use pyo3::wrap_pyfunction;

/// Header dict keys that are handled specially (not re-inserted as generic
/// attributes): the ones `Header(w,h)` sets, plus `channels`/`compression`,
/// which the caller's data already drives.
const STANDARD_KEYS: &[&str] =
    &["dataWindow", "displayWindow", "channels", "compression", "pixelAspectRatio", "lineOrder"];

/// Read one `(x, y)` pair off an `Imath.point`-derived object (`V2i`/`V2f`,
/// or a `Box2i`/`Box2f`'s `.min`/`.max`).
fn xy_i(v: &Bound<'_, PyAny>) -> PyResult<(i32, i32)> {
    Ok((v.getattr("x")?.extract()?, v.getattr("y")?.extract()?))
}
fn xy_f(v: &Bound<'_, PyAny>) -> PyResult<(f32, f32)> {
    Ok((v.getattr("x")?.extract()?, v.getattr("y")?.extract()?))
}

/// Convert one legacy `Imath.py` header value into an `exr_core` attribute, for
/// `OutputFile.close()`'s custom-metadata round trip (mirrors `PyOpenEXR_old.cpp`'s
/// `insertAttribute` for the types the legacy `Imath` module actually has: boxes,
/// 2-vectors, chromaticities, line order, compression, rational, timecode, keycode,
/// tile description — plus plain str/int/float, which the legacy header dict also
/// allows). Returns `None` for a type with no legacy attribute mapping (e.g. a
/// `Channel`/`PixelType` value, which is not a standalone header attribute).
fn imath_to_attr(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<Option<Box<dyn Attribute>>> {
    let imath = py.import("Imath")?;
    macro_rules! isa {
        ($name:literal) => {
            v.is_instance(&imath.getattr($name)?)?
        };
    }
    if isa!("Box2i") {
        let (minx, miny) = xy_i(&v.getattr("min")?)?;
        let (maxx, maxy) = xy_i(&v.getattr("max")?)?;
        return Ok(Some(Box::new(Box2iAttribute::new(Box2i {
            min: V2i::new(minx, miny),
            max: V2i::new(maxx, maxy),
        }))));
    }
    if isa!("Box2f") {
        let (minx, miny) = xy_f(&v.getattr("min")?)?;
        let (maxx, maxy) = xy_f(&v.getattr("max")?)?;
        return Ok(Some(Box::new(Box2fAttribute::new(Box2f {
            min: V2f::new(minx, miny),
            max: V2f::new(maxx, maxy),
        }))));
    }
    if isa!("V2i") {
        let (x, y) = xy_i(v)?;
        return Ok(Some(Box::new(V2iAttribute::new(V2i::new(x, y)))));
    }
    if isa!("V2f") {
        let (x, y) = xy_f(v)?;
        return Ok(Some(Box::new(V2fAttribute::new(V2f::new(x, y)))));
    }
    if isa!("Chromaticities") {
        let (rx, ry) = xy_f(&v.getattr("red")?)?;
        let (gx, gy) = xy_f(&v.getattr("green")?)?;
        let (bx, by) = xy_f(&v.getattr("blue")?)?;
        let (wx, wy) = xy_f(&v.getattr("white")?)?;
        return Ok(Some(Box::new(ChromaticitiesAttribute::new(Chromaticities {
            red: [rx, ry],
            green: [gx, gy],
            blue: [bx, by],
            white: [wx, wy],
        }))));
    }
    if isa!("LineOrder") {
        let code: i32 = v.getattr("v")?.extract()?;
        return Ok(Some(Box::new(LineOrderAttribute::new(line_order_from_code(code)))));
    }
    if isa!("Compression") {
        let code: i32 = v.getattr("v")?.extract()?;
        return Ok(Some(Box::new(CompressionAttribute::new(comp_from_code(code)?))));
    }
    if isa!("Rational") {
        let n: i32 = v.getattr("n")?.extract()?;
        let d: u32 = v.getattr("d")?.extract()?;
        return Ok(Some(Box::new(RationalAttribute::new(Rational { n, d }))));
    }
    if isa!("TimeCode") {
        let g = |name: &str| -> PyResult<i32> { v.getattr(name)?.extract() };
        let gb = |name: &str| -> PyResult<bool> { v.getattr(name)?.extract() };
        let tc = CoreTimeCode::from_fields(
            g("hours")?, g("minutes")?, g("seconds")?, g("frame")?,
            gb("dropFrame")?, gb("colorFrame")?, gb("fieldPhase")?,
            gb("bgf0")?, gb("bgf1")?, gb("bgf2")?,
            g("binaryGroup1")?, g("binaryGroup2")?, g("binaryGroup3")?, g("binaryGroup4")?,
            g("binaryGroup5")?, g("binaryGroup6")?, g("binaryGroup7")?, g("binaryGroup8")?,
        )
        .map_err(verr)?;
        return Ok(Some(Box::new(TimeCodeAttribute::new(tc))));
    }
    if isa!("KeyCode") {
        let g = |name: &str| -> PyResult<i32> { v.getattr(name)?.extract() };
        let kc = CoreKeyCode::new(
            g("filmMfcCode")?, g("filmType")?, g("prefix")?, g("count")?,
            g("perfOffset")?, g("perfsPerFrame")?, g("perfsPerCount")?,
        )
        .map_err(verr)?;
        return Ok(Some(Box::new(KeyCodeAttribute::new(kc))));
    }
    if isa!("TileDescription") {
        let x_size: u32 = v.getattr("xSize")?.extract()?;
        let y_size: u32 = v.getattr("ySize")?.extract()?;
        let mode_code: i32 = v.getattr("mode")?.getattr("v")?.extract()?;
        let rounding_code: i32 = v.getattr("roundingMode")?.getattr("v")?.extract()?;
        return Ok(Some(Box::new(TileDescriptionAttribute::new(CoreTileDescription {
            x_size,
            y_size,
            mode: level_mode_from_code(mode_code),
            rounding_mode: rounding_mode_from_code(rounding_code),
        }))));
    }
    if v.cast::<PyString>().is_ok() {
        return Ok(Some(Box::new(exr_core::attr::StringAttribute::new(
            exr_core::attr::ExrString::from(v.extract::<String>()?.as_str()),
        ))));
    }
    if v.cast::<pyo3::types::PyBool>().is_ok() {
        return Ok(None); // bool is an int subclass; EXR has no bool attribute.
    }
    if v.cast::<PyInt>().is_ok() {
        return Ok(Some(Box::new(exr_core::attr::IntAttribute::new(v.extract::<i32>()?))));
    }
    if v.cast::<PyFloat>().is_ok() {
        return Ok(Some(Box::new(exr_core::attr::FloatAttribute::new(v.extract::<f64>()? as f32))));
    }
    Ok(None)
}

fn line_order_from_code(code: i32) -> LineOrder {
    match code {
        0 => LineOrder::IncreasingY,
        1 => LineOrder::DecreasingY,
        2 => LineOrder::RandomY,
        _ => LineOrder::Invalid,
    }
}
fn level_mode_from_code(code: i32) -> exr_core::attr::LevelMode {
    use exr_core::attr::LevelMode as L;
    match code {
        0 => L::OneLevel,
        1 => L::MipmapLevels,
        2 => L::RipmapLevels,
        _ => L::Invalid,
    }
}
fn rounding_mode_from_code(code: i32) -> exr_core::attr::LevelRoundingMode {
    use exr_core::attr::LevelRoundingMode as R;
    match code {
        0 => R::RoundDown,
        1 => R::RoundUp,
        _ => R::RoundDown,
    }
}

/// Collect every header-dict entry that is not one of `STANDARD_KEYS` as an
/// extra `exr_core` attribute, for `Image::write_with_attrs`. Values with no
/// legacy mapping (per [`imath_to_attr`]) are skipped rather than erroring, so
/// an unrecognised entry does not block writing the rest of the header.
fn legacy_extra_attrs(py: Python<'_>, header: &Bound<'_, PyDict>) -> PyResult<Vec<(exr_core::attr::Name, Box<dyn Attribute>)>> {
    let mut out = Vec::new();
    for (k, v) in header.iter() {
        let Ok(name) = k.extract::<String>() else { continue };
        if STANDARD_KEYS.contains(&name.as_str()) {
            continue;
        }
        if let Some(attr) = imath_to_attr(py, &v)? {
            out.push((name.into(), attr));
        }
    }
    Ok(out)
}

/// Build the corresponding legacy `Imath.py` object for one carried (non-standard)
/// header attribute, the read-side inverse of [`imath_to_attr`]. `None` for a type
/// this legacy round trip does not map (opaque/unknown types stay out of the dict,
/// same as the reference `PyOpenEXR_old.cpp`'s `attrTypeIsSupported` gate).
fn attr_to_imath(py: Python<'_>, attr: &dyn Attribute, header: &Header, name: &str) -> PyResult<Option<Py<PyAny>>> {
    let imath = py.import("Imath")?;
    macro_rules! get {
        ($t:ty) => {
            header.find_typed_attribute::<$t>(name).map(|a| &a.value)
        };
    }
    let v2i = |x: i32, y: i32| imath.getattr("V2i").and_then(|c| c.call1((x, y)));
    let v2f = |x: f32, y: f32| imath.getattr("V2f").and_then(|c| c.call1((x, y)));
    // C-08c: `type_name()` is raw bytes now; every arm below is ASCII, so a
    // non-UTF-8 (necessarily-unregistered) type name simply can't match any
    // of them and falls through to `_ => None`, same as any other unknown type.
    let obj = match std::str::from_utf8(attr.type_name()).unwrap_or("") {
        "box2i" => match get!(Box2i) {
            Some(b) => Some(
                imath
                    .getattr("Box2i")?
                    .call1((v2i(b.min.x, b.min.y)?, v2i(b.max.x, b.max.y)?))?
                    .unbind(),
            ),
            None => None,
        },
        "box2f" => match get!(Box2f) {
            Some(b) => Some(
                imath
                    .getattr("Box2f")?
                    .call1((v2f(b.min.x, b.min.y)?, v2f(b.max.x, b.max.y)?))?
                    .unbind(),
            ),
            None => None,
        },
        "v2i" => get!(V2i).map(|p| v2i(p.x, p.y)).transpose()?.map(|o| o.unbind()),
        "v2f" => get!(V2f).map(|p| v2f(p.x, p.y)).transpose()?.map(|o| o.unbind()),
        "chromaticities" => match get!(Chromaticities) {
            Some(c) => Some(
                imath
                    .getattr("Chromaticities")?
                    .call1((
                        v2f(c.red[0], c.red[1])?,
                        v2f(c.green[0], c.green[1])?,
                        v2f(c.blue[0], c.blue[1])?,
                        v2f(c.white[0], c.white[1])?,
                    ))?
                    .unbind(),
            ),
            None => None,
        },
        "lineOrder" => match get!(LineOrder) {
            Some(l) => Some(imath.getattr("LineOrder")?.call1((*l as i32,))?.unbind()),
            None => None,
        },
        // An unmappable code (HTJ2K) is omitted rather than raising, so reading the
        // header of a file exr-core can actually decode doesn't fail outright —
        // same reasoning as header.rs's "compression" arm.
        "compression" => match get!(Compression).and_then(|c| comp_to_code(*c).ok()) {
            Some(code) => Some(imath.getattr("Compression")?.call1((code,))?.unbind()),
            None => None,
        },
        "rational" => match get!(Rational) {
            Some(r) => Some(imath.getattr("Rational")?.call1((r.n, r.d))?.unbind()),
            None => None,
        },
        "timecode" => match get!(CoreTimeCode) {
            Some(tc) => Some(
                imath
                    .getattr("TimeCode")?
                    .call1((
                        tc.hours(), tc.minutes(), tc.seconds(), tc.frame(),
                        tc.drop_frame(), tc.color_frame(), tc.field_phase(),
                        tc.bgf0(), tc.bgf1(), tc.bgf2(),
                    ))?
                    .unbind(),
            ),
            None => None,
        },
        "keycode" => match get!(CoreKeyCode) {
            Some(kc) => Some(
                imath
                    .getattr("KeyCode")?
                    .call1((
                        kc.film_mfc_code(), kc.film_type(), kc.prefix(), kc.count(),
                        kc.perf_offset(), kc.perfs_per_frame(), kc.perfs_per_count(),
                    ))?
                    .unbind(),
            ),
            None => None,
        },
        "tiledesc" => match get!(CoreTileDescription) {
            Some(t) => {
                let mode = imath.getattr("LevelMode")?.call1((t.mode as i32,))?;
                let rounding = imath.getattr("LevelRoundingMode")?.call1((t.rounding_mode as i32,))?;
                Some(
                    imath
                        .getattr("TileDescription")?
                        .call1((t.x_size, t.y_size, mode, rounding))?
                        .unbind(),
                )
            }
            None => None,
        },
        "string" => match header.find_typed_attribute::<exr_core::attr::ExrString>(name) {
            Some(a) => Some(a.value.as_str().unwrap_or("").into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "int" => match header.find_typed_attribute::<i32>(name) {
            Some(a) => Some(a.value.into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "float" => match header.find_typed_attribute::<f32>(name) {
            Some(a) => Some(a.value.into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        _ => None, // no legacy Imath mapping for this attribute type
    };
    Ok(obj)
}

/// A code with no legacy Imath.Compression member
/// (HTJ2K256/HTJ2K32/Invalid) raises instead of silently reporting ZIP (3).
fn comp_to_code(c: Compression) -> PyResult<i32> {
    match c {
        Compression::NoCompression => Ok(0),
        Compression::Rle => Ok(1),
        Compression::Zips => Ok(2),
        Compression::Zip => Ok(3),
        Compression::Piz => Ok(4),
        Compression::Pxr24 => Ok(5),
        Compression::B44 => Ok(6),
        Compression::B44a => Ok(7),
        Compression::Dwaa => Ok(8),
        Compression::Dwab => Ok(9),
        Compression::Htj2k256 => Ok(10),
        Compression::Htj2k32 => Ok(11),
        _ => Err(PyValueError::new_err(
            "compression code has no legacy Imath.Compression member (on-disk garbage)",
        )),
    }
}

fn ioerr(e: impl std::fmt::Display) -> PyErr {
    PyIOError::new_err(e.to_string())
}
fn verr(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Coerce a Python value (int, our `PixelType`, or `Imath.PixelType`) to a pixel
/// type code via `int()`.
///
/// `PyOpenEXR_old.cpp:436` raises `TypeError` for "Invalid PixelType object" (and
/// `:459`/`:614` "Unknown type") — never `ValueError` — for this whole family of
/// "not a valid `Imath.PixelType`" failures.
fn ptype_from_py(v: &Bound<'_, PyAny>) -> PyResult<PixelType> {
    let code: i32 = v
        .py()
        .import("builtins")?
        .getattr("int")?
        .call1((v,))?
        .extract()?;
    Ok(match code {
        0 => PixelType::Uint,
        1 => PixelType::Half,
        2 => PixelType::Float,
        _ => return Err(PyTypeError::new_err(format!("invalid pixel type {code}"))),
    })
}

// ── sample access + (de)serialisation ───────────────────────────────────────────

/// Serialise samples `[start, end)` of `cd` in `target` type as little-endian
/// (XDR) bytes.
///
/// Routes entirely through the canonical `exr_core` conversion
/// surface instead of hand-rolled per-sample arithmetic — `ChannelData::
/// convert` (which is exact: e.g. UINT->UINT is a `.clone()`, never an
/// intermediate `f32` round-trip that would lose precision above 2^24) for
/// the type conversion, then `ChannelData::to_xdr_bytes` for the wire bytes,
/// and slice out just the requested sample range. `target` is always a real
/// `PixelType` here (never the `NumPixelTypes` sentinel: it comes from
/// `ptype_from_py`, which itself errors on an unrecognised code, or from
/// `cd.pixel_type()`), so `sz == 0` / a `convert` error are both unreachable
/// defensive paths, not a real code path this function's callers exercise.
fn cd_to_bytes(cd: &ChannelData, target: PixelType, start: usize, end: usize) -> Vec<u8> {
    let sz = exr_core::misc::pixel_type_size(target).unwrap_or(0);
    let Ok(converted) = cd.convert(target) else {
        return Vec::new();
    };
    let bytes = converted.to_xdr_bytes();
    let lo = start.saturating_mul(sz).min(bytes.len());
    let hi = end.min(converted.len()).saturating_mul(sz).min(bytes.len());
    bytes.get(lo..hi).map(<[u8]>::to_vec).unwrap_or_default()
}

/// Parse little-endian `bytes` of `count` samples of `ptype` into a `ChannelData`.
fn bytes_to_cd(bytes: &[u8], ptype: PixelType, count: usize) -> PyResult<ChannelData> {
    let need = count * exr_core::misc::pixel_type_size(ptype).unwrap_or(0);
    if bytes.len() != need {
        // Matches `outwrite`'s buffer-size check (`PyOpenEXR_old.cpp:1198-1208`,
        // "Data for channel '%s' should have size %zu but got %zu") -> `TypeError`.
        return Err(PyTypeError::new_err(format!(
            "channel data is {} bytes but {count} {ptype:?} samples need {need}",
            bytes.len()
        )));
    }
    Ok(match ptype {
        PixelType::Float => ChannelData::Float(
            bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        ),
        PixelType::Uint => ChannelData::Uint(
            bytes.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect(),
        ),
        PixelType::Half => ChannelData::Half(
            bytes.as_chunks::<2>().0.iter().map(|&c| Half::from_bits(u16::from_le_bytes(c))).collect(),
        ),
        // Matches "Unknown type" (`PyOpenEXR_old.cpp:459`/`:614`) -> `TypeError`.
        PixelType::NumPixelTypes => return Err(PyTypeError::new_err("invalid pixel type")),
    })
}

// ── Header(width, height) ───────────────────────────────────────────────────────

/// `Header(width, height, channels="R,G,B")` — a legacy header dict with
/// Imath-typed defaults. `channels` is a comma-separated name list, each
/// inserted as a FLOAT `Imath.Channel` (`PyOpenEXR_old.cpp:1549-1564`); an
/// empty name in the list is skipped rather than inserted as `""`.
#[pyfunction]
#[pyo3(name = "Header")]
#[pyo3(signature = (width, height, channels="R,G,B"))]
fn header(width: i32, height: i32, channels: &str, py: Python<'_>) -> PyResult<Py<PyDict>> {
    if width < 1 || height < 1 {
        return Err(verr(format!(
            "Header dimensions must be positive (got {width} x {height})"
        )));
    }
    let imath = py.import("Imath")?;
    let v2i = |x: i32, y: i32| imath.getattr("V2i").and_then(|c| c.call1((x, y)));
    let dw = imath
        .getattr("Box2i")?
        .call1((v2i(0, 0)?, v2i(width - 1, height - 1)?))?;
    let chans = PyDict::new(py);
    let float_pt = imath.getattr("PixelType")?.call1((2,))?; // FLOAT
    for name in channels.split(',') {
        if name.is_empty() {
            continue;
        }
        chans.set_item(name, imath.getattr("Channel")?.call1((&float_pt,))?)?;
    }
    let d = PyDict::new(py);
    d.set_item("dataWindow", &dw)?;
    d.set_item("displayWindow", &dw)?;
    d.set_item("channels", chans)?;
    d.set_item("compression", imath.getattr("Compression")?.call1((3,))?)?; // ZIP
    d.set_item("pixelAspectRatio", 1.0f32)?;
    d.set_item("lineOrder", imath.getattr("LineOrder")?.call1((0,))?)?;
    Ok(d.unbind())
}

/// `(width, height)` from a legacy header's `dataWindow` (an `Imath.Box2i`).
fn header_dims(header: &Bound<'_, PyDict>) -> PyResult<(i32, i32)> {
    // The reference never encounters a header missing `dataWindow` here (its
    // `OutputFile` ctor always fills one via `Header(64,64)`/`Header(w,h)`) --
    // there's no 1:1 exception-class precedent, so this stays consistent with
    // every OTHER structural-header-shape failure in this module (never
    // `KeyError`): `TypeError`.
    let dw = header
        .get_item("dataWindow")?
        .ok_or_else(|| PyTypeError::new_err("header has no 'dataWindow'"))?;
    let minx: i32 = dw.getattr("min")?.getattr("x")?.extract()?;
    let miny: i32 = dw.getattr("min")?.getattr("y")?.extract()?;
    let maxx: i32 = dw.getattr("max")?.getattr("x")?.extract()?;
    let maxy: i32 = dw.getattr("max")?.getattr("y")?.extract()?;
    // Checked extent: reject an inverted or i32-overflowing data window instead of
    // panicking / wrapping on the i32 `max - min + 1`.
    let w = maxx as i64 - minx as i64 + 1;
    let h = maxy as i64 - miny as i64 + 1;
    if w <= 0 || h <= 0 {
        // Matches "Invalid image dimensions" (`PyOpenEXR_old.cpp:464`/`:627`) ->
        // `ValueError`.
        return Err(verr(format!(
            "header dataWindow has non-positive extent ({w} x {h})"
        )));
    }
    // Matches "Integer overflow computing allocation size" (`:474`/`:637`) ->
    // `OverflowError`.
    let w = i32::try_from(w)
        .map_err(|_| PyOverflowError::new_err("header dataWindow width exceeds the i32 range"))?;
    let h = i32::try_from(h)
        .map_err(|_| PyOverflowError::new_err("header dataWindow height exceeds the i32 range"))?;
    Ok((w, h))
}

// ── OutputFile ──────────────────────────────────────────────────────────────────

/// Legacy `OutputFile(path, header)`: `writePixels({name: bytes})` then `close()`.
#[pyclass(module = "OpenEXR")]
pub struct OutputFile {
    path: String,
    header: Py<PyDict>,
    /// Accumulated little-endian sample bytes per channel (supports chunked
    /// `writePixels` across calls); flushed to the file by `close`.
    buffers: std::collections::BTreeMap<String, Vec<u8>>,
    done: bool,
}

#[pymethods]
impl OutputFile {
    #[new]
    fn new(path: &str, header: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Validate the destination is writable now (the reference raises here).
        std::fs::File::create(path).map_err(ioerr)?;
        Ok(OutputFile {
            path: path.to_string(),
            header: header.clone().unbind(),
            buffers: std::collections::BTreeMap::new(),
            done: false,
        })
    }

    /// Append each channel's little-endian sample bytes. Called once for a full
    /// image, or repeatedly with consecutive scanline blocks (chunked write); the
    /// `num_scan_lines` count is implied by the byte length. Flushed by `close`.
    #[pyo3(signature = (pixels, _scan_lines=None))]
    fn writePixels(&mut self, pixels: &Bound<'_, PyDict>, _scan_lines: Option<i32>) -> PyResult<()> {
        for (name, value) in pixels.iter() {
            let name: String = name.extract()?;
            let bytes = value.cast::<PyBytes>().map_err(|e| PyTypeError::new_err(e.to_string()))?;
            self.buffers.entry(name).or_default().extend_from_slice(bytes.as_bytes());
        }
        Ok(())
    }

    /// Assemble the accumulated channels into the file (idempotent; mirrors the
    /// reference writing on destruction).
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        let header = self.header.bind(py);
        let (w, h) = header_dims(header)?;
        // Same "no 1:1 reference precedent, stay consistent with TypeError" reasoning
        // as `header_dims`'s missing-`dataWindow` case.
        let chans = header
            .get_item("channels")?
            .ok_or_else(|| PyTypeError::new_err("header has no 'channels'"))?;
        let chans: Bound<'_, PyDict> = chans.cast_into().map_err(|e| PyTypeError::new_err(e.to_string()))?;
        // Matches "Integer overflow computing allocation size" -> `OverflowError`.
        let count = (w as usize)
            .checked_mul(h as usize)
            .ok_or_else(|| PyOverflowError::new_err("image pixel count overflows usize"))?;

        let mut image = Image::new(Box2i { min: V2i::new(0, 0), max: V2i::new(w - 1, h - 1) });
        for (name, bytes) in &self.buffers {
            let ch = chans
                .get_item(name)?
                .ok_or_else(|| PyTypeError::new_err(format!("header channels has no '{name}'")))?;
            let ptype = ptype_from_py(&ch.getattr("type")?)?;
            let cd = bytes_to_cd(bytes, ptype, count)?;
            // Not a size/argument mismatch (already checked by `bytes_to_cd`) --
            // an exr-core failure assembling the image, closest to the
            // reference's `setFrameBuffer`/`writePixels` `catch(std::exception&)`
            // -> `IOError`.
            image.insert_channel(name, cd).map_err(ioerr)?;
        }
        let comp = match header.get_item("compression")? {
            Some(c) => {
                let code = py.import("builtins")?.getattr("int")?.call1((c,))?.extract()?;
                comp_from_code(code)?
            }
            None => Compression::Zip,
        };
        // Honour a carried `lineOrder`, same as the reference OutputFile ctor.
        if let Some(lo) = header.get_item("lineOrder")?
            && let Ok(code) = py.import("builtins")?.getattr("int")?.call1((&lo,))?.extract::<i32>()
        {
            image.line_order = line_order_from_code(code);
        }
        // Custom header metadata (Box2i/V2f/Chromaticities/TimeCode/KeyCode/Rational/
        // TileDescription/str/int/float) round-trips through the extra-attribute
        // overlay instead of being silently dropped.
        let extra = legacy_extra_attrs(py, header)?;
        // Matches `outwrite`'s `try { setFrameBuffer; writePixels; } catch
        // (std::exception&) { IOError }` (`PyOpenEXR_old.cpp:1223-1233`).
        image.write_with_attrs(&self.path, comp, extra).map_err(ioerr)?;
        Ok(())
    }
}

/// An out-of-range code — including 3 itself, which this
/// `match` used to miss entirely and fall through to the `_ => Zip` default for the
/// right answer by accident — now raises instead of silently becoming ZIP.
fn comp_from_code(code: i32) -> PyResult<Compression> {
    match code {
        0 => Ok(Compression::NoCompression),
        1 => Ok(Compression::Rle),
        2 => Ok(Compression::Zips),
        3 => Ok(Compression::Zip),
        4 => Ok(Compression::Piz),
        5 => Ok(Compression::Pxr24),
        6 => Ok(Compression::B44),
        7 => Ok(Compression::B44a),
        8 => Ok(Compression::Dwaa),
        9 => Ok(Compression::Dwab),
        10 => Ok(Compression::Htj2k256),
        11 => Ok(Compression::Htj2k32),
        _ => Err(PyValueError::new_err(format!("unknown compression code {code}"))),
    }
}

// ── InputFile ───────────────────────────────────────────────────────────────────

/// Legacy `InputFile(path)`: `header()`, `channel(name, pixel_type, scanLine1,
/// scanLine2)` -> bytes, `channels(names)` -> list of bytes.
#[pyclass(module = "OpenEXR")]
pub struct InputFile {
    image: Image,
    width: i32,
    min_y: i32,
}

#[pymethods]
impl InputFile {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let image = Image::read(path).map_err(ioerr)?;
        let width = image.width() as i32;
        let min_y = image.data_window.min.y;
        Ok(InputFile { image, width, min_y })
    }

    /// A legacy header dict: `dataWindow`/`displayWindow` as `Imath.Box2i`,
    /// `channels` as `{name: Imath.Channel}`, plus `compression`.
    fn header(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let imath = py.import("Imath")?;
        let v2i = |x: i32, y: i32| imath.getattr("V2i").and_then(|c| c.call1((x, y)));
        let dw = &self.image.data_window;
        let box2i = imath
            .getattr("Box2i")?
            .call1((v2i(dw.min.x, dw.min.y)?, v2i(dw.max.x, dw.max.y)?))?;
        let disp = &self.image.display_window;
        let disp_box = imath
            .getattr("Box2i")?
            .call1((v2i(disp.min.x, disp.min.y)?, v2i(disp.max.x, disp.max.y)?))?;
        let chans = PyDict::new(py);
        for (name, data) in self.image.channels() {
            let pt = imath.getattr("PixelType")?.call1((data.pixel_type() as i32,))?;
            chans.set_item(name, imath.getattr("Channel")?.call1((pt,))?)?;
        }
        let d = PyDict::new(py);
        d.set_item("dataWindow", box2i)?;
        d.set_item("displayWindow", disp_box)?;
        d.set_item("channels", chans)?;
        // `compression` is always present in the legacy header dict, so an
        // unmappable code (HTJ2K, no legacy Imath.Compression member) can't be
        // omitted the way the generic-attribute path does — fall back to the raw
        // int code rather than fail `header()` outright for a file exr-core can
        // otherwise read fine.
        let comp_value = match comp_to_code(self.image.compression) {
            Ok(code) => imath.getattr("Compression")?.call1((code,))?,
            Err(_) => (self.image.compression as i32).into_pyobject(py)?.into_any(),
        };
        d.set_item("compression", comp_value)?;
        d.set_item(
            "lineOrder",
            imath.getattr("LineOrder")?.call1((self.image.line_order as i32,))?,
        )?;
        // Carried custom metadata: everything
        // `OutputFile.close()` wrote via `legacy_extra_attrs` comes back here,
        // typed with the legacy `Imath` classes.
        let mut extra_header = Header::new();
        self.image.attributes().apply_to(&mut extra_header).map_err(verr)?;
        for (name, attr) in self.image.attributes().iter() {
            // `name` is a raw-byte `Name` now; the legacy dict is
            // Python-`str`-keyed, so a non-UTF-8 name is skipped here rather
            // than raising (this legacy binding already drops opaque/unknown
            // types the same way — see `attr_to_imath`'s doc).
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if let Some(v) = attr_to_imath(py, attr.as_ref(), &extra_header, name_str)? {
                d.set_item(name_str, v)?;
            }
        }
        Ok(d.unbind())
    }

    /// One channel's pixels as `bytes` in `pixel_type` (default: native), optionally
    /// limited to scan lines `scanLine1..=scanLine2`.
    #[pyo3(signature = (name, pixel_type=None, scanLine1=None, scanLine2=None))]
    fn channel(
        &self,
        name: &str,
        pixel_type: Option<&Bound<'_, PyAny>>,
        scanLine1: Option<i32>,
        scanLine2: Option<i32>,
        py: Python<'_>,
    ) -> PyResult<Py<PyBytes>> {
        // Matches "There is no channel '%s' in the image" (`PyOpenEXR_old.cpp:427`)
        // -> `TypeError`, not `KeyError`.
        let cd = self
            .image
            .channel(name)
            .ok_or_else(|| PyTypeError::new_err(format!("no channel '{name}'")))?;
        let target = match pixel_type {
            Some(v) => ptype_from_py(v)?,
            None => cd.pixel_type(),
        };
        let w = self.width as usize;
        let (start, end) = match (scanLine1, scanLine2) {
            (Some(s1), Some(s2)) => {
                // Validate the inclusive scanline range against the data window
                // BEFORE any arithmetic (mirrors PyOpenEXR_old.cpp:406-421): an
                // inverted (s1 > s2) or out-of-window range would otherwise
                // underflow `end - start` in `cd_to_bytes` (panic) or over-read.
                let h = cd.len().checked_div(w).unwrap_or(0);
                let max_y = self.min_y + h as i32 - 1;
                // Matches "scanLine1 must be <= scanLine2" /
                // "scanLine1/2 cannot be outside dataWindow" (`:407-419`) ->
                // `TypeError`, not `ValueError`.
                if s1 > s2 {
                    return Err(PyTypeError::new_err(format!(
                        "scanLine1 ({s1}) must be <= scanLine2 ({s2})"
                    )));
                }
                if s1 < self.min_y || s2 > max_y {
                    return Err(PyTypeError::new_err(format!(
                        "scan line range [{s1}, {s2}] is outside the data window \
                         [{}, {max_y}]",
                        self.min_y
                    )));
                }
                let r1 = (s1 - self.min_y) as usize;
                let r2 = (s2 - self.min_y) as usize;
                (r1 * w, (r2 + 1) * w)
            }
            _ => (0, cd.len()),
        };
        let bytes = cd_to_bytes(cd, target, start, end.min(cd.len()));
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    /// Several channels' bytes, as a list in `names` order (native type).
    fn channels(&self, names: Vec<String>, py: Python<'_>) -> PyResult<Vec<Py<PyBytes>>> {
        names
            .iter()
            .map(|n| self.channel(n, None, None, None, py))
            .collect()
    }

    fn close(&self) {}
    fn isComplete(&self) -> bool {
        true
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(header, m)?)?;
    m.add_class::<OutputFile>()?;
    m.add_class::<InputFile>()?;
    Ok(())
}
