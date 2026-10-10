//! P4 (header read) — dump a full `exr_core` `Header` into a Python dict, mapping
//! each attribute type the way the reference does: scalars as Python int/float/str,
//! Imath vectors/matrices as numpy arrays, boxes as tuples, enums as the module's
//! enum members, and `channels` as a `{name: Channel-descriptor}` dict.
//!
//! Read via `MultiPartInputFile`, which serves single- AND multi-part files
//! uniformly. Rare attribute types (keycode/timecode/preview) are not yet mapped
//! here — they round-trip opaquely; see CLAUDE.md.

use exr_core::attr::{
    Attribute, Box2fAttribute, Box2iAttribute, Chromaticities, Compression, Envmap, ExrString,
    FloatAttribute, IntAttribute, KeyCode, KeyCodeAttribute, LineOrder, PixelType, PreviewImage,
    PreviewImageAttribute, Rational, StringAttribute, TileDescription, TileDescriptionAttribute,
    TimeCode, TimeCodeAttribute, V2fAttribute, V2iAttribute, V3fAttribute, V3iAttribute,
};
use exr_core::header::Header;
use imath_rs::{Box2f, Box2i, M33f, M44f, V2d, V2f, V2i, V3d, V3f, V3i};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::enums;
use crate::types;

/// Build the full header dict for `header`.
///
/// C-08: an attribute NAME is a raw byte string in the underlying `Header`
/// (no encoding assumed — C++ `Name` never checks). The real C++ PyOpenEXR
/// binding converts it with `py::str(const char*)`, which is `pybind11`'s
/// strict UTF-8 `PyUnicode_FromString` — a non-UTF-8 name raises
/// `UnicodeDecodeError` there, so this mirrors that instead of silently
/// mangling or dropping it.
pub fn to_dict(py: Python<'_>, header: &Header) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for (name, attr) in header.iter() {
        let name_str = name_to_py_str(name)?;
        // C-08c: `type_name()` is raw bytes now; `attr_to_py`'s match arms are
        // all ASCII, so a non-UTF-8 (necessarily-unregistered) type name
        // simply can't match any of them — same "falls through" reasoning as
        // `legacy.rs::attr_to_imath`.
        let type_name = std::str::from_utf8(attr.type_name()).unwrap_or("");
        if let Some(v) = attr_to_py(py, header, name_str, type_name, attr.as_ref())? {
            d.set_item(name_str, v)?;
        }
    }
    Ok(d.unbind())
}

/// Convert an [`exr_core::attr::Name`] to `&str`, raising the same exception
/// class real C++ PyOpenEXR raises for a non-UTF-8 name (`py::str` is a
/// strict UTF-8 decode; see [`to_dict`]'s doc).
pub(crate) fn name_to_py_str(name: &exr_core::attr::Name) -> PyResult<&str> {
    bytes_to_py_str(name.as_bytes())
}

/// Byte-string version of [`name_to_py_str`] — same strict-UTF-8 `py::str`
/// behavior, for a raw `&[u8]` that isn't a [`exr_core::attr::Name`] (e.g. an
/// [`exr_core::attr::OpaqueAttribute`]'s `type_name()`, C-08c).
pub(crate) fn bytes_to_py_str(bytes: &[u8]) -> PyResult<&str> {
    std::str::from_utf8(bytes).map_err(|_| {
        pyo3::exceptions::PyUnicodeDecodeError::new_err((
            "utf-8",
            bytes.to_vec(),
            0usize,
            bytes.len(),
            "invalid utf-8",
        ))
    })
}

/// Convert one attribute (looked up by name + type) to a Python object, or `None`
/// to omit an unmapped type.
fn attr_to_py(
    py: Python<'_>,
    h: &Header,
    name: &str,
    type_name: &str,
    attr: &dyn Attribute,
) -> PyResult<Option<Py<PyAny>>> {
    macro_rules! get {
        ($t:ty) => {
            h.find_typed_attribute::<$t>(name).map(|a| &a.value)
        };
    }
    let obj: Option<Py<PyAny>> = match type_name {
        "int" => match get!(i32) {
            Some(v) => Some(v.into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "float" => match get!(f32) {
            Some(v) => Some(v.into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        // The reference returns a 1-element numpy `float64` array, not a plain
        // Python `float` (`PyOpenEXR.cpp:1587-1588`, `py::array_t<double>(1,
        // &v->value())`).
        "double" => get!(f64).map(|v| arr1_d(py, &[*v])),
        "string" => match get!(ExrString) {
            Some(v) => Some(v.as_str().unwrap_or("").into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "stringvector" => match get!(Vec<ExrString>) {
            Some(v) => {
                let items: Vec<String> =
                    v.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect();
                Some(PyList::new(py, items)?.into_any().unbind())
            }
            None => None,
        },
        // The reference returns a Python `list` of `float`, not a numpy array
        // (`PyOpenEXR.cpp:1730-1735`) -- fixes `test_write_float_vector_ndarray`'s
        // `assertEqual(header["floatvector32"], [0.0, 1.0, 2.0, 3.0])`, which is
        // ambiguous (`ValueError`) against a numpy array.
        "floatvector" => match get!(Vec<f32>) {
            Some(v) => Some(PyList::new(py, v.iter().map(|&f| f as f64))?.into_any().unbind()),
            None => None,
        },
        "v2i" => get!(V2i).map(|v| arr1_i(py, &[v.x, v.y])),
        "v3i" => get!(V3i).map(|v| arr1_i(py, &[v.x, v.y, v.z])),
        "v2f" => get!(V2f).map(|v| arr1_f(py, &[v.x, v.y])),
        "v3f" => get!(V3f).map(|v| arr1_f(py, &[v.x, v.y, v.z])),
        "v2d" => get!(V2d).map(|v| arr1_d(py, &[v.x, v.y])),
        "v3d" => get!(V3d).map(|v| arr1_d(py, &[v.x, v.y, v.z])),
        "box2i" => match get!(Box2i) {
            Some(b) => Some(box_i(py, b.min.x, b.min.y, b.max.x, b.max.y)?),
            None => None,
        },
        "box2f" => match get!(Box2f) {
            Some(b) => Some(box_f(py, b.min.x, b.min.y, b.max.x, b.max.y)?),
            None => None,
        },
        "m33f" => match get!(M33f) {
            Some(m) => Some(mat(py, &flatten33(&m.x), 3)?),
            None => None,
        },
        "m44f" => match get!(M44f) {
            Some(m) => Some(mat(py, &flatten44(&m.x), 4)?),
            None => None,
        },
        "chromaticities" => get!(Chromaticities).map(|c| chroma(py, c)),
        // A code `comp_to_py` can't map (on-disk garbage — HTJ2K256/HTJ2K32 have
        // real enum members now that the target is v3.4.15) is OMITTED, like any
        // other unmapped attribute below, rather than raising and failing the whole
        // header read (r3_surfaces.md item 8).
        "compression" => match get!(Compression).and_then(|c| comp_member(*c).ok()) {
            Some(c) => Some(c.into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "lineOrder" => match get!(LineOrder) {
            Some(l) => Some((*l as i32).into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "envmap" => match get!(Envmap) {
            Some(e) => Some((*e as i32).into_pyobject(py)?.into_any().unbind()),
            None => None,
        },
        "rational" => match get!(Rational) {
            Some(r) => Some(Py::new(py, types::Rational { n: r.n, d: r.d })?.into_any()),
            None => None,
        },
        "tiledesc" => match get!(TileDescription) {
            Some(t) => Some(tiledesc(py, t)?),
            None => None,
        },
        "chlist" => Some(channels_dict(py, h)?),
        "keycode" => match get!(KeyCode) {
            Some(kc) => Some(Py::new(py, types::KeyCode::from_core(kc.clone()))?.into_any()),
            None => None,
        },
        "timecode" => match get!(TimeCode) {
            Some(tc) => Some(Py::new(py, types::TimeCode::from_core(tc.clone()))?.into_any()),
            None => None,
        },
        "preview" => match get!(PreviewImage).and_then(|pv| types::PreviewImage::from_core(py, pv).ok()) {
            Some(p) => Some(Py::new(py, p)?.into_any()),
            None => None,
        },
        // OpenEXR 3.4's "bytes" attribute (C-11) — target is v3.4.15, not v3.3.11
        // (team-lead correction 2026-09-22): `OpenEXR.Bytes` exists in the
        // reference (`PyOpenEXR.cpp:2694`), so this is a real gap, not out-of-scope.
        "bytes" => match get!(exr_core::attr::Bytes) {
            Some(b) => Some(Py::new(py, types::Bytes::from_core(b))?.into_any()),
            None => None,
        },
        // Any type-name exr-core's registry didn't recognise is stored internally
        // as an `OpaqueAttribute` (exr-core's own catch-all, not a wire type) —
        // surface it the same way the reference does (`OpenEXR.OpaqueAttribute`,
        // `PyOpenEXR.cpp:2728-2754`) instead of silently dropping it.
        _ => match attr.as_any().downcast_ref::<exr_core::attr::OpaqueAttribute>() {
            Some(op) => Some(Py::new(py, types::OpaqueAttribute::from_core(op)?)?.into_any()),
            None => None,
        },
    };
    Ok(obj)
}

fn arr1_i(py: Python<'_>, v: &[i32]) -> Py<PyAny> {
    Array1::from(v.to_vec()).into_pyarray(py).into_any().unbind()
}
fn arr1_f(py: Python<'_>, v: &[f32]) -> Py<PyAny> {
    Array1::from(v.to_vec()).into_pyarray(py).into_any().unbind()
}
fn arr1_d(py: Python<'_>, v: &[f64]) -> Py<PyAny> {
    Array1::from(v.to_vec()).into_pyarray(py).into_any().unbind()
}
fn mat(py: Python<'_>, flat: &[f32], n: usize) -> PyResult<Py<PyAny>> {
    let arr = Array2::from_shape_vec((n, n), flat.to_vec())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}
fn flatten33(m: &[[f32; 3]; 3]) -> Vec<f32> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}
fn flatten44(m: &[[f32; 4]; 4]) -> Vec<f32> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}
fn box_i(py: Python<'_>, a: i32, b: i32, c: i32, d: i32) -> PyResult<Py<PyAny>> {
    let mn = PyTuple::new(py, [a, b])?;
    let mx = PyTuple::new(py, [c, d])?;
    Ok(PyTuple::new(py, [mn, mx])?.into_any().unbind())
}
fn box_f(py: Python<'_>, a: f32, b: f32, c: f32, d: f32) -> PyResult<Py<PyAny>> {
    let mn = PyTuple::new(py, [a, b])?;
    let mx = PyTuple::new(py, [c, d])?;
    Ok(PyTuple::new(py, [mn, mx])?.into_any().unbind())
}
fn chroma(py: Python<'_>, c: &Chromaticities) -> Py<PyAny> {
    let v = [
        c.red[0], c.red[1], c.green[0], c.green[1], c.blue[0], c.blue[1], c.white[0], c.white[1],
    ];
    Array1::from(v.to_vec()).into_pyarray(py).into_any().unbind()
}
fn tiledesc(py: Python<'_>, t: &TileDescription) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        types::TileDescription {
            xSize: t.x_size,
            ySize: t.y_size,
            mode: level_mode(t.mode),
            roundingMode: rounding_mode(t.rounding_mode),
        },
    )?
    .into_any())
}

/// The `channels` attribute as a `{name: Channel-descriptor}` dict (each value a
/// `Channel` with name + pixel type; pixels are filled by the reader, not here).
fn channels_dict(py: Python<'_>, h: &Header) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    if let Ok(chans) = h.channels() {
        for (name, ch) in chans.iter() {
            let name = name_to_py_str(name)?;
            let pt = pixeltype_member(ch.type_);
            let chan = crate::file::channel_descriptor(py, name, pt)?;
            d.set_item(name, chan)?;
        }
    }
    Ok(d.into_any().unbind())
}

fn comp_member(c: Compression) -> PyResult<enums::Compression> {
    crate::file::comp_to_py(c)
}
fn pixeltype_member(p: PixelType) -> enums::PixelType {
    match p {
        PixelType::Uint => enums::PixelType::UINT,
        PixelType::Half => enums::PixelType::HALF,
        PixelType::Float => enums::PixelType::FLOAT,
        PixelType::NumPixelTypes => enums::PixelType::NUM_PIXELTYPES,
    }
}
fn level_mode(m: exr_core::attr::LevelMode) -> enums::LevelMode {
    use exr_core::attr::LevelMode as L;
    match m {
        L::OneLevel => enums::LevelMode::ONE_LEVEL,
        L::MipmapLevels => enums::LevelMode::MIPMAP_LEVELS,
        L::RipmapLevels => enums::LevelMode::RIPMAP_LEVELS,
        _ => enums::LevelMode::NUM_LEVEL_MODES,
    }
}
fn rounding_mode(m: exr_core::attr::LevelRoundingMode) -> enums::LevelRoundingMode {
    use exr_core::attr::LevelRoundingMode as R;
    match m {
        R::RoundDown => enums::LevelRoundingMode::ROUND_DOWN,
        R::RoundUp => enums::LevelRoundingMode::ROUND_UP,
        _ => enums::LevelRoundingMode::NUM_ROUNDING_MODES,
    }
}
// Write-side inverses of `level_mode`/`rounding_mode` above, for
// `py_to_attr`'s `TileDescription` conversion. `NUM_LEVEL_MODES`/
// `NUM_ROUNDING_MODES` (the sentinel a caller should never actually
// construct) fall back to the C++ default (`ONE_LEVEL`/`ROUND_DOWN`) rather
// than erroring — matching how `level_mode`/`rounding_mode` themselves treat
// an unrecognised value as a sentinel, not a hard failure.
fn level_mode_from_py(m: enums::LevelMode) -> exr_core::attr::LevelMode {
    match m {
        enums::LevelMode::ONE_LEVEL => exr_core::attr::LevelMode::OneLevel,
        enums::LevelMode::MIPMAP_LEVELS => exr_core::attr::LevelMode::MipmapLevels,
        enums::LevelMode::RIPMAP_LEVELS => exr_core::attr::LevelMode::RipmapLevels,
        enums::LevelMode::NUM_LEVEL_MODES => exr_core::attr::LevelMode::OneLevel,
    }
}
fn rounding_mode_from_py(m: enums::LevelRoundingMode) -> exr_core::attr::LevelRoundingMode {
    match m {
        enums::LevelRoundingMode::ROUND_DOWN => exr_core::attr::LevelRoundingMode::RoundDown,
        enums::LevelRoundingMode::ROUND_UP => exr_core::attr::LevelRoundingMode::RoundUp,
        enums::LevelRoundingMode::NUM_ROUNDING_MODES => exr_core::attr::LevelRoundingMode::RoundDown,
    }
}

// ── Write side: Python header dict -> extra exr_core attributes ─────────────────

/// Standard keys the `Image` already carries (set from the dict elsewhere) or
/// that describe a file shape the facade derives/doesn't write; they must NOT be
/// re-inserted as generic typed attributes via `py_to_attrs`'s generic path.
///
/// This deliberately does NOT include `"tiles"` even though it's one of
/// `exr_core::api::attrs::ImageAttributes::DERIVED`'s entries: the single-part
/// write path (`File::write`) relies on `"tiles"` flowing through
/// `py_to_attrs` -> `extra` -> `Image::write_with_attrs`, which applies `extra`
/// directly to the header (not through the stricter `ImageAttributes::insert`
/// the multipart path uses), and that's how a `TileDescription` in the header
/// dict actually makes the file tiled. Adding it here once regressed
/// `test_write_tiled`/`test_write_tiles`. The multipart write loop filters it
/// out itself, right before the `ImageAttributes::insert` call that WOULD
/// reject it as derived — see the call site in `file.rs`.
const STANDARD_KEYS: &[&str] = &[
    "type",
    "compression",
    "dataWindow",
    "displayWindow",
    "lineOrder",
    "channels",
    "chunkCount",
    "name",
    "version",
];

/// Convert a Python header dict into extra `(name, attribute)` pairs for writing
/// (skipping the standard keys). Values that don't map to a known attribute type
/// are omitted rather than erroring — so a header round-trips its common metadata
/// (strings, ints, floats, V2f/V2i/V3f/V3i) without rejecting exotic entries.
pub fn py_to_attrs(
    header: &Bound<'_, PyDict>,
) -> PyResult<Vec<(exr_core::attr::Name, Box<dyn Attribute>)>> {
    let mut out = Vec::new();
    for (k, v) in header.iter() {
        let Ok(name) = k.extract::<String>() else {
            continue;
        };
        if STANDARD_KEYS.contains(&name.as_str()) {
            continue;
        }
        if let Some(attr) = py_to_attr(&v)? {
            out.push((name.into(), attr));
        }
    }
    Ok(out)
}

/// Whether a scalar (Python `float`/numpy `float32`/`float64`, but NOT a Python
/// `int` or a numpy integer scalar) should coerce a 2-tuple into a V2f rather than
/// a V2i. Distinguishing this at the element level — instead of only accepting a
/// numpy array — is what lets a `screenWindowCenter`-style fixed-V2f key reject a
/// plain-int tuple: `Header::insert_boxed` already refuses a type mismatch against
/// `Header::new`'s V2f default, so mapping an int 2-tuple to V2i here is what makes
/// that rejection happen naturally (test_v2f_int_rejection), while a numpy
/// `float32`/`float64` scalar (test_v2f_numpy_scalars) still reads as float-like.
fn is_float_like(v: &Bound<'_, PyAny>) -> PyResult<bool> {
    use pyo3::types::{PyBool, PyFloat, PyInt};
    if v.cast::<PyBool>().is_ok() || v.cast::<PyInt>().is_ok() {
        return Ok(false);
    }
    if v.cast::<PyFloat>().is_ok() {
        return Ok(true);
    }
    // A numpy scalar: go by its dtype name ("float32"/"float64" vs "int32"/…).
    let type_name = v.get_type().name()?.to_string();
    Ok(type_name.starts_with("float"))
}

fn py_to_attr(v: &Bound<'_, PyAny>) -> PyResult<Option<Box<dyn Attribute>>> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};
    // numpy vectors first (a float32/int32 array: 2/3 elements are the fixed-size
    // V2f/V3f/V2i/V3i vectors; any other length is a floatvector/[i32 has none]).
    if let Ok(a) = v.cast::<PyArray1<f32>>() {
        let s: Vec<f32> = a.readonly().as_array().iter().copied().collect();
        return Ok(Some(match s.len() {
            2 => Box::new(V2fAttribute::new(V2f::new(s[0], s[1]))),
            3 => Box::new(V3fAttribute::new(V3f::new(s[0], s[1], s[2]))),
            _ => Box::new(exr_core::attr::FloatVectorAttribute::new(s)),
        }));
    }
    if let Ok(a) = v.cast::<PyArray1<i32>>() {
        let s: Vec<i32> = a.readonly().as_array().iter().copied().collect();
        return Ok(match s.len() {
            2 => Some(Box::new(V2iAttribute::new(V2i::new(s[0], s[1])))),
            3 => Some(Box::new(V3iAttribute::new(V3i::new(s[0], s[1], s[2])))),
            _ => None,
        });
    }
    // Helper pyclasses carrying verified exr_core attribute values.
    if let Ok(kc) = v.cast::<crate::types::KeyCode>() {
        return Ok(Some(Box::new(KeyCodeAttribute::new(kc.borrow().core()))));
    }
    if let Ok(tc) = v.cast::<crate::types::TimeCode>() {
        return Ok(Some(Box::new(TimeCodeAttribute::new(tc.borrow().core()))));
    }
    if let Ok(pv) = v.cast::<crate::types::PreviewImage>() {
        let core = pv.borrow().to_core(v.py())?;
        return Ok(Some(Box::new(PreviewImageAttribute::new(core))));
    }
    if let Ok(b) = v.cast::<crate::types::Bytes>() {
        return Ok(Some(Box::new(exr_core::attr::BytesAttribute::new(b.borrow().to_core()))));
    }
    if let Ok(op) = v.cast::<crate::types::OpaqueAttribute>() {
        return Ok(Some(Box::new(op.borrow().to_core())));
    }
    // py-parity2 exr-core need (a) follow-up: `header["tiles"] =
    // TileDescription(...)` (the modern numpy API's own pyclass, distinct
    // from legacy.rs's Imath-style `TileDescription` with a nested `.mode.v`)
    // was previously unrecognised here and silently DROPPED -- so even once
    // `exr_core::api::Image::write_to_with_attrs` learned to dispatch on a
    // `tiles` attribute, nothing ever reached it through this API surface.
    // Mirrors `legacy.rs`'s existing `isa!("TileDescription")` conversion for
    // the OLD Imath-style object; discriminant values match 1:1
    // (`enums::LevelMode`/`LevelRoundingMode` were defined to mirror
    // `exr_core::attr::LevelMode`/`LevelRoundingMode` exactly).
    if let Ok(td) = v.cast::<crate::types::TileDescription>() {
        let t = td.borrow();
        return Ok(Some(Box::new(TileDescriptionAttribute::new(TileDescription {
            x_size: t.xSize,
            y_size: t.ySize,
            mode: level_mode_from_py(t.mode),
            rounding_mode: rounding_mode_from_py(t.roundingMode),
        }))));
    }
    if v.cast::<PyString>().is_ok() {
        let s: String = v.extract()?;
        return Ok(Some(Box::new(StringAttribute::new(ExrString::from(s.as_str())))));
    }
    // A 2-tuple of 2-element sub-sequences is a box (`PyOpenEXR.cpp:2206-2220`,
    // "Recognize 2-tuples of 2-vectors as boxes" — any custom attribute name, not
    // just the standard `dataWindow`/`displayWindow` keys): both int-like -> Box2i,
    // both float-like -> Box2f (`test_write_float`'s `header["box2i"] = ((0,1),
    // (2,3))`).
    if let Ok(t) = v.cast::<PyTuple>()
        && t.len() == 2 && t.get_item(0)?.len().unwrap_or(0) == 2 && t.get_item(1)?.len().unwrap_or(0) == 2
    {
        if let Ok(b) = crate::file::py_to_box(v) {
            return Ok(Some(Box::new(Box2iAttribute::new(b))));
        }
        if let Ok(b) = crate::file::py_to_boxf(v) {
            return Ok(Some(Box::new(Box2fAttribute::new(b))));
        }
    }
    // A 2-tuple of plain scalars is unambiguously meant to be a V2f/V2i vector
    // (`screenWindowCenter`'s usual shape in the reference, and the common case
    // for arbitrary custom metadata): both elements float-like -> V2f, both
    // int-like -> V2i. A tuple that isn't consistently one or the other
    // (`test_tuple`: `(0, "x")`, `(1, 2.3)`) used to fall through and be silently
    // DROPPED — now raises, since a 2-tuple of scalars has no other legitimate
    // meaning here.
    if let Ok(t) = v.cast::<PyTuple>()
        && t.len() == 2
    {
        let (a, b) = (t.get_item(0)?, t.get_item(1)?);
        if let (Ok(fa), Ok(fb)) = (is_float_like(&a), is_float_like(&b)) {
            if fa && fb {
                if let (Ok(x), Ok(y)) = (a.extract::<f32>(), b.extract::<f32>()) {
                    return Ok(Some(Box::new(V2fAttribute::new(V2f::new(x, y)))));
                }
            } else if !fa && !fb
                && let (Ok(x), Ok(y)) = (a.extract::<i32>(), b.extract::<i32>())
            {
                return Ok(Some(Box::new(V2iAttribute::new(V2i::new(x, y)))));
            }
        }
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "a 2-tuple header value must be either all int or all float (V2i or V2f)",
        ));
    }
    // A list of strings -> stringvector.
    if let Ok(items) = v.cast::<pyo3::types::PyList>()
        && !items.is_empty() && items.iter().all(|it| it.cast::<PyString>().is_ok())
    {
        let strs: Vec<ExrString> = items
            .iter()
            .filter_map(|it| it.extract::<String>().ok())
            .map(|s| ExrString::from(s.as_str()))
            .collect();
        return Ok(Some(Box::new(exr_core::attr::StringVectorAttribute::new(strs))));
    }
    // `bool` is an `int` subclass and EXR has no bool attribute — skip it.
    if v.cast::<PyBool>().is_ok() {
        return Ok(None);
    }
    if v.cast::<PyInt>().is_ok() {
        return Ok(Some(Box::new(IntAttribute::new(v.extract::<i32>()?))));
    }
    if v.cast::<PyFloat>().is_ok() {
        return Ok(Some(Box::new(FloatAttribute::new(v.extract::<f64>()? as f32))));
    }
    Ok(None)
}
