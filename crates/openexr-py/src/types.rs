//! P6 helper value classes (`TileDescription`, `Rational`, `KeyCode`, `TimeCode`,
//! `PreviewRgba`, `PreviewImage`) + the module functions `isOpenExrFile` /
//! `isTiledOpenExrFile`.
//!
//! The bit-packed types (`KeyCode`, `TimeCode`) WRAP the already 1:1-ported
//! `exr_core::attr` types — the verified packing logic is reused, never re-derived.
//! The plain holders mirror the reference fields exactly.

use std::io::Read;

use exr_core::attr::{
    Attribute, Bytes as CoreBytes, KeyCode as CoreKeyCode, TimeCode as CoreTimeCode, TimeCodePacking,
};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;

use crate::enums::{LevelMode, LevelRoundingMode};

fn verr(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ── TileDescription ─────────────────────────────────────────────────────────────

/// Tile description for tiled images: tile size + level/rounding mode.
#[pyclass(module = "OpenEXR", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct TileDescription {
    #[pyo3(get, set)]
    pub xSize: u32,
    #[pyo3(get, set)]
    pub ySize: u32,
    #[pyo3(get, set)]
    pub mode: LevelMode,
    #[pyo3(get, set)]
    pub roundingMode: LevelRoundingMode,
}

#[pymethods]
impl TileDescription {
    #[new]
    #[pyo3(signature = (xSize=32, ySize=32, mode=LevelMode::ONE_LEVEL, roundingMode=LevelRoundingMode::ROUND_DOWN))]
    fn new(xSize: u32, ySize: u32, mode: LevelMode, roundingMode: LevelRoundingMode) -> Self {
        TileDescription { xSize, ySize, mode, roundingMode }
    }
    fn __repr__(&self) -> String {
        format!("TileDescription({}, {}, {:?}, {:?})", self.xSize, self.ySize, self.mode, self.roundingMode)
    }
}

// ── Rational ────────────────────────────────────────────────────────────────────

/// A number expressed as a ratio `n/d`.
#[pyclass(module = "OpenEXR", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct Rational {
    #[pyo3(get, set)]
    pub n: i32,
    #[pyo3(get, set)]
    pub d: u32,
}

#[pymethods]
impl Rational {
    #[new]
    #[pyo3(signature = (n=0, d=1))]
    fn new(n: i32, d: u32) -> Self {
        Rational { n, d }
    }
    fn __repr__(&self) -> String {
        format!("Rational({}, {})", self.n, self.d)
    }
}

// ── KeyCode ─────────────────────────────────────────────────────────────────────

/// Motion-picture film characteristics (wraps the verified `exr_core` `KeyCode`,
/// whose setters validate each field).
#[pyclass(module = "OpenEXR", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct KeyCode {
    inner: CoreKeyCode,
}

#[pymethods]
impl KeyCode {
    #[new]
    #[pyo3(signature = (filmMfcCode=0, filmType=0, prefix=0, count=0, perfOffset=0, perfsPerFrame=4, perfsPerCount=64))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        filmMfcCode: i32,
        filmType: i32,
        prefix: i32,
        count: i32,
        perfOffset: i32,
        perfsPerFrame: i32,
        perfsPerCount: i32,
    ) -> PyResult<Self> {
        let inner = CoreKeyCode::new(filmMfcCode, filmType, prefix, count, perfOffset, perfsPerFrame, perfsPerCount)
            .map_err(verr)?;
        Ok(KeyCode { inner })
    }

    #[getter] fn filmMfcCode(&self) -> i32 { self.inner.film_mfc_code() }
    #[setter] fn set_filmMfcCode(&mut self, v: i32) -> PyResult<()> { self.inner.set_film_mfc_code(v).map_err(verr) }
    #[getter] fn filmType(&self) -> i32 { self.inner.film_type() }
    #[setter] fn set_filmType(&mut self, v: i32) -> PyResult<()> { self.inner.set_film_type(v).map_err(verr) }
    #[getter] fn prefix(&self) -> i32 { self.inner.prefix() }
    #[setter] fn set_prefix(&mut self, v: i32) -> PyResult<()> { self.inner.set_prefix(v).map_err(verr) }
    #[getter] fn count(&self) -> i32 { self.inner.count() }
    #[setter] fn set_count(&mut self, v: i32) -> PyResult<()> { self.inner.set_count(v).map_err(verr) }
    #[getter] fn perfOffset(&self) -> i32 { self.inner.perf_offset() }
    #[setter] fn set_perfOffset(&mut self, v: i32) -> PyResult<()> { self.inner.set_perf_offset(v).map_err(verr) }
    #[getter] fn perfsPerFrame(&self) -> i32 { self.inner.perfs_per_frame() }
    #[setter] fn set_perfsPerFrame(&mut self, v: i32) -> PyResult<()> { self.inner.set_perfs_per_frame(v).map_err(verr) }
    #[getter] fn perfsPerCount(&self) -> i32 { self.inner.perfs_per_count() }
    #[setter] fn set_perfsPerCount(&mut self, v: i32) -> PyResult<()> { self.inner.set_perfs_per_count(v).map_err(verr) }

    fn __repr__(&self) -> String {
        format!("KeyCode(prefix={}, count={})", self.inner.prefix(), self.inner.count())
    }
}

// ── TimeCode ────────────────────────────────────────────────────────────────────

/// SMPTE time + control code (wraps `exr_core` `TimeCode`; the BCD packing is the
/// verified 1:1 port). `timeAndFlags` uses TV60 packing, like the reference.
#[pyclass(module = "OpenEXR", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct TimeCode {
    inner: CoreTimeCode,
}

#[pymethods]
impl TimeCode {
    /// The reference's C++ signature takes plain `bool` parameters, which pybind11
    /// converts implicitly from any Python object via the truthiness protocol (so
    /// `TimeCode(..., dropFrame=5, ...)` — an `int`, not a `bool` — is accepted,
    /// per the official `test_write_float`). PyO3's `bool` extraction is strict
    /// (no implicit int->bool), so these six flags take `&Bound<PyAny>` and go
    /// through `is_truthy()` instead of a plain `bool` parameter type.
    #[new]
    #[pyo3(signature = (hours=0, minutes=0, seconds=0, frame=0, dropFrame=None, colorFrame=None,
        fieldPhase=None, bgf0=None, bgf1=None, bgf2=None,
        binaryGroup1=0, binaryGroup2=0, binaryGroup3=0, binaryGroup4=0,
        binaryGroup5=0, binaryGroup6=0, binaryGroup7=0, binaryGroup8=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        hours: i32, minutes: i32, seconds: i32, frame: i32,
        dropFrame: Option<&Bound<'_, PyAny>>, colorFrame: Option<&Bound<'_, PyAny>>,
        fieldPhase: Option<&Bound<'_, PyAny>>, bgf0: Option<&Bound<'_, PyAny>>,
        bgf1: Option<&Bound<'_, PyAny>>, bgf2: Option<&Bound<'_, PyAny>>,
        binaryGroup1: i32, binaryGroup2: i32, binaryGroup3: i32, binaryGroup4: i32,
        binaryGroup5: i32, binaryGroup6: i32, binaryGroup7: i32, binaryGroup8: i32,
    ) -> PyResult<Self> {
        let truthy = |v: Option<&Bound<'_, PyAny>>| -> PyResult<bool> {
            match v {
                Some(v) => v.is_truthy(),
                None => Ok(false),
            }
        };
        let inner = CoreTimeCode::from_fields(
            hours, minutes, seconds, frame,
            truthy(dropFrame)?, truthy(colorFrame)?, truthy(fieldPhase)?,
            truthy(bgf0)?, truthy(bgf1)?, truthy(bgf2)?,
            binaryGroup1, binaryGroup2, binaryGroup3, binaryGroup4, binaryGroup5, binaryGroup6,
            binaryGroup7, binaryGroup8,
        ).map_err(verr)?;
        Ok(TimeCode { inner })
    }

    #[getter] fn hours(&self) -> i32 { self.inner.hours() }
    #[setter] fn set_hours(&mut self, v: i32) -> PyResult<()> { self.inner.set_hours(v).map_err(verr) }
    #[getter] fn minutes(&self) -> i32 { self.inner.minutes() }
    #[setter] fn set_minutes(&mut self, v: i32) -> PyResult<()> { self.inner.set_minutes(v).map_err(verr) }
    #[getter] fn seconds(&self) -> i32 { self.inner.seconds() }
    #[setter] fn set_seconds(&mut self, v: i32) -> PyResult<()> { self.inner.set_seconds(v).map_err(verr) }
    #[getter] fn frame(&self) -> i32 { self.inner.frame() }
    #[setter] fn set_frame(&mut self, v: i32) -> PyResult<()> { self.inner.set_frame(v).map_err(verr) }
    #[getter] fn dropFrame(&self) -> bool { self.inner.drop_frame() }
    #[setter] fn set_dropFrame(&mut self, v: bool) { self.inner.set_drop_frame(v) }
    #[getter] fn colorFrame(&self) -> bool { self.inner.color_frame() }
    #[setter] fn set_colorFrame(&mut self, v: bool) { self.inner.set_color_frame(v) }
    #[getter] fn fieldPhase(&self) -> bool { self.inner.field_phase() }
    #[setter] fn set_fieldPhase(&mut self, v: bool) { self.inner.set_field_phase(v) }
    #[getter] fn bgf0(&self) -> bool { self.inner.bgf0() }
    #[setter] fn set_bgf0(&mut self, v: bool) { self.inner.set_bgf0(v) }
    #[getter] fn bgf1(&self) -> bool { self.inner.bgf1() }
    #[setter] fn set_bgf1(&mut self, v: bool) { self.inner.set_bgf1(v) }
    #[getter] fn bgf2(&self) -> bool { self.inner.bgf2() }
    #[setter] fn set_bgf2(&mut self, v: bool) { self.inner.set_bgf2(v) }
    #[getter] fn userData(&self) -> u32 { self.inner.user_data() }
    #[setter] fn set_userData(&mut self, v: u32) { self.inner.set_user_data(v) }

    fn timeAndFlags(&self) -> u32 {
        self.inner.time_and_flags(TimeCodePacking::Tv60)
    }
    fn setTimeAndFlags(&mut self, value: u32) {
        self.inner.set_time_and_flags(value, TimeCodePacking::Tv60)
    }

    fn __repr__(&self) -> String {
        format!(
            "TimeCode({:02}:{:02}:{:02}:{:02})",
            self.inner.hours(), self.inner.minutes(), self.inner.seconds(), self.inner.frame()
        )
    }
}

// ── PreviewRgba / PreviewImage ──────────────────────────────────────────────────

/// One 8-bit RGBA pixel of the preview thumbnail.
#[pyclass(module = "OpenEXR", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PreviewRgba {
    #[pyo3(get, set)]
    pub r: u8,
    #[pyo3(get, set)]
    pub g: u8,
    #[pyo3(get, set)]
    pub b: u8,
    #[pyo3(get, set)]
    pub a: u8,
}

#[pymethods]
impl PreviewRgba {
    #[new]
    #[pyo3(signature = (r=0, g=0, b=0, a=255))]
    fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        PreviewRgba { r, g, b, a }
    }
    fn __repr__(&self) -> String {
        format!("PreviewRgba({}, {}, {}, {})", self.r, self.g, self.b, self.a)
    }
}

/// Thumbnail version of the image. `pixels` is an `(H, W, 4)` uint8 numpy array
/// (R,G,B,A). (The reference uses a structured `PreviewRgba` array; the flat uint8
/// form is equivalent data and easier to consume.)
#[pyclass(module = "OpenEXR")]
pub struct PreviewImage {
    #[pyo3(get, set)]
    pub pixels: Py<PyAny>,
}

#[pymethods]
impl PreviewImage {
    /// Three reference overloads (`PyOpenEXR.cpp:2824-2830`): `PreviewImage()`
    /// (empty), `PreviewImage(width, height)` (zeroed), and
    /// `PreviewImage(array)` — a 2-D numpy STRUCTURED array with fields named
    /// `r`/`g`/`b`/`a` (any integer dtype width — pybind11's `array_t<PreviewRgba>`
    /// implicitly truncate-casts each field to `uint8`, exactly like
    /// `test_preview_image`'s `u4`-field array; `.astype("uint8")` per field
    /// reproduces that truncation). `.pixels` is ALWAYS normalised to a `(H, W)`
    /// structured array with `uint8` `r`/`g`/`b`/`a` fields — matching the
    /// reference's own `array_t<PreviewRgba>` shape/dtype exactly (an
    /// earlier version used a flat `(H,W,4)` uint8 array,
    /// which round-trips internally but fails official `preview_pixels_equal`
    /// shape/dtype comparisons against the real reference layout).
    #[new]
    #[pyo3(signature = (arg0=None, height=None))]
    fn new(arg0: Option<&Bound<'_, PyAny>>, height: Option<usize>, py: Python<'_>) -> PyResult<Self> {
        let Some(arg0) = arg0 else {
            return Ok(PreviewImage { pixels: make_structured(py, 0, 0)?.unbind() });
        };
        if let Ok(width) = arg0.extract::<usize>() {
            let height = height.ok_or_else(|| {
                PyValueError::new_err("PreviewImage(width, height): both arguments are required")
            })?;
            return Ok(PreviewImage { pixels: make_structured(py, height, width)?.unbind() });
        }
        Self::from_structured(py, arg0)
    }

    /// `.def(py::self == py::self)` (`PyOpenEXR.cpp:2829`) — compares `.pixels`
    /// elementwise via numpy (structured-array equality), not Python identity.
    fn __eq__(&self, other: &PreviewImage, py: Python<'_>) -> PyResult<bool> {
        let np = py.import("numpy")?;
        let eq = np.call_method1("array_equal", (self.pixels.bind(py), other.pixels.bind(py)))?;
        eq.extract()
    }

    fn width(&self, py: Python<'_>) -> PyResult<usize> {
        let shape: Vec<usize> = self.pixels.bind(py).getattr("shape")?.extract()?;
        shape.get(1).copied().ok_or_else(|| PyValueError::new_err("PreviewImage.pixels must be a 2-D array"))
    }
    fn height(&self, py: Python<'_>) -> PyResult<usize> {
        let shape: Vec<usize> = self.pixels.bind(py).getattr("shape")?.extract()?;
        shape.first().copied().ok_or_else(|| PyValueError::new_err("PreviewImage.pixels must be a 2-D array"))
    }
}

/// A `(h, w)` numpy structured array, uninitialised-then-zeroed, with `uint8`
/// `r`/`g`/`b`/`a` fields — the canonical `PreviewImage.pixels` layout (see
/// [`PreviewImage::new`]). numpy accepts a list of `(name, format)` tuples
/// directly as a `dtype=` argument, no explicit `np.dtype(...)` call needed.
fn make_structured<'py>(py: Python<'py>, h: usize, w: usize) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let dtype = PyList::new(
        py,
        [
            PyTuple::new(py, ["r", "u1"])?,
            PyTuple::new(py, ["g", "u1"])?,
            PyTuple::new(py, ["b", "u1"])?,
            PyTuple::new(py, ["a", "u1"])?,
        ],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    np.call_method("zeros", (PyTuple::new(py, [h, w])?,), Some(&kwargs))
}

// Plain (non-Python) constructors so the header reader can build these from the
// verified `exr_core` attribute values.
impl KeyCode {
    pub(crate) fn from_core(inner: CoreKeyCode) -> Self {
        KeyCode { inner }
    }
    pub(crate) fn core(&self) -> CoreKeyCode {
        self.inner.clone()
    }
}
impl TimeCode {
    pub(crate) fn from_core(inner: CoreTimeCode) -> Self {
        TimeCode { inner }
    }
    pub(crate) fn core(&self) -> CoreTimeCode {
        self.inner.clone()
    }
}
impl PreviewImage {
    /// Read one `uint8`-castable field of a 2-D structured array as an owned
    /// `Array2<u8>` (`.astype("uint8")` reproduces pybind11's implicit
    /// truncating `array_t<PreviewRgba>` conversion — `test_preview_image`
    /// deliberately uses `u4` fields to exercise exactly this truncation).
    fn u8_field(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<numpy::ndarray::Array2<u8>> {
        let field = arr.get_item(name)?;
        let casted = field.call_method1("astype", ("uint8",))?;
        let a = casted
            .cast::<PyArray2<u8>>()
            .map_err(|_| PyValueError::new_err("PreviewImage(array): each field must be a 2-D array"))?;
        Ok(a.readonly().as_array().to_owned())
    }

    /// `PreviewImage(array)`: a 2-D numpy structured array with `r`/`g`/`b`/`a`
    /// fields (any integer dtype). Normalises into the canonical `(H, W)`
    /// structured `uint8` array (`make_structured`) rather than keeping the
    /// caller's own field dtype, so `.pixels`'s dtype is always exactly the
    /// reference's `PreviewRgba` layout regardless of what was passed in.
    fn from_structured(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (r, g, b, a) = (Self::u8_field(arr, "r")?, Self::u8_field(arr, "g")?, Self::u8_field(arr, "b")?, Self::u8_field(arr, "a")?);
        let (h, w) = r.dim();
        let out = make_structured(py, h, w)?;
        for (name, field) in [("r", &r), ("g", &g), ("b", &b), ("a", &a)] {
            out.set_item(name, field.clone().into_pyarray(py))?;
        }
        Ok(PreviewImage { pixels: out.unbind() })
    }

    /// Build from an `exr_core` preview (`PreviewRgba` row-major) as a `(H, W)`
    /// structured `uint8` r/g/b/a numpy array — matching the reference's
    /// `array_t<PreviewRgba>` shape/dtype exactly (not a flat `(H,W,4)` array).
    pub(crate) fn from_core(py: Python<'_>, pv: &exr_core::attr::PreviewImage) -> PyResult<Self> {
        let (w, h) = (pv.width() as usize, pv.height() as usize);
        let (mut rs, mut gs, mut bs, mut as_) =
            (Vec::with_capacity(w * h), Vec::with_capacity(w * h), Vec::with_capacity(w * h), Vec::with_capacity(w * h));
        for p in pv.pixels() {
            rs.push(p.r);
            gs.push(p.g);
            bs.push(p.b);
            as_.push(p.a);
        }
        let out = make_structured(py, h, w)?;
        for (name, data) in [("r", rs), ("g", gs), ("b", bs), ("a", as_)] {
            let field = Array2::<u8>::from_shape_vec((h, w), data).map_err(verr)?;
            out.set_item(name, field.into_pyarray(py))?;
        }
        Ok(PreviewImage { pixels: out.unbind() })
    }

    /// The inverse of [`from_core`]: read the `(H, W)` structured `uint8` array
    /// into an `exr_core` preview for writing.
    pub(crate) fn to_core(&self, py: Python<'_>) -> PyResult<exr_core::attr::PreviewImage> {
        let arr = self.pixels.bind(py);
        let (r, g, b, a) = (Self::u8_field(arr, "r")?, Self::u8_field(arr, "g")?, Self::u8_field(arr, "b")?, Self::u8_field(arr, "a")?);
        let (h, w) = r.dim();
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                px.push(exr_core::attr::PreviewRgba::new(r[[y, x]], g[[y, x]], b[[y, x]], a[[y, x]]));
            }
        }
        exr_core::attr::PreviewImage::new(w as u32, h as u32, Some(px)).map_err(verr)
    }
}

// ── Module functions ────────────────────────────────────────────────────────────

/// Read the first 8 bytes of `path`: `(magic_ok, version_word)`.
fn read_magic(path: &str) -> Option<(bool, u32)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).ok()?;
    let magic = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Some((magic as u32 == exr_core::container::version::MAGIC, version))
}

/// True if `path` is an OpenEXR file (correct magic number).
#[pyfunction]
fn isOpenExrFile(path: &str) -> bool {
    matches!(read_magic(path), Some((true, _)))
}

/// True if `path` is a tiled OpenEXR file.
#[pyfunction]
fn isTiledOpenExrFile(path: &str) -> bool {
    match read_magic(path) {
        Some((true, version)) => {
            let flags = exr_core::container::version::get_flags(version);
            exr_core::container::version::is_tiled(flags)
        }
        _ => false,
    }
}

// ── Bytes ────────────────────────────────────────────────────────────────────────

/// Arbitrary binary data with an optional MIME-style type hint (OpenEXR 3.4's
/// `"bytes"` attribute — a *known*, registered type, unlike `OpaqueAttribute`'s
/// "unrecognised type" catch-all). Wraps `exr_core::attr::Bytes` 1:1.
///
/// Reference: `openexr/src/wrappers/python/PyOpenEXR.cpp` (v3.4.15,
/// `py::class_<BytesAttribute>(m, "Bytes")`) — `Bytes(data: bytes, type_hint: str =
/// "")`, `.data` (rw bytes), `.type_hint` (rw str), `==`, `repr`.
#[pyclass(module = "OpenEXR", skip_from_py_object)]
#[derive(Clone)]
pub struct Bytes {
    data: Vec<u8>,
    #[pyo3(get, set)]
    pub type_hint: String,
}

#[pymethods]
impl Bytes {
    #[new]
    #[pyo3(signature = (data, type_hint=String::new()))]
    fn new(data: &Bound<'_, PyBytes>, type_hint: String) -> Self {
        Bytes { data: data.as_bytes().to_vec(), type_hint }
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    #[setter]
    fn set_data(&mut self, value: &Bound<'_, PyBytes>) {
        self.data = value.as_bytes().to_vec();
    }

    fn __eq__(&self, other: &Bytes) -> bool {
        self.data == other.data && self.type_hint == other.type_hint
    }

    fn __repr__(&self) -> String {
        format!(
            "<Bytes data=b'...' ({} bytes), type_hint='{}'>",
            self.data.len(),
            self.type_hint
        )
    }
}

impl Bytes {
    pub(crate) fn from_core(inner: &CoreBytes) -> Self {
        Bytes {
            data: inner.data.clone(),
            // The reference's `type_hint` is a C++ `std::string` (pybind11 exposes
            // it as a Python `str`); our core storage is byte-opaque (matching C++
            // semantics for arbitrary bytes), so decode lossily rather than raise —
            // a non-UTF-8 hint is exotic and this is metadata, not the payload.
            type_hint: String::from_utf8_lossy(&inner.type_hint).into_owned(),
        }
    }

    pub(crate) fn to_core(&self) -> CoreBytes {
        CoreBytes::new(&self.type_hint, self.data.clone())
    }
}

// ── OpaqueAttribute ────────────────────────────────────────────────────────────

/// The raw bytes + type-name of an unrecognised header attribute — this is
/// exr-core's own catch-all for a type-name it doesn't know how to parse
/// (`exr_core::attr::OpaqueAttribute`), surfaced to Python 1:1.
///
/// Reference: `PyOpenEXR.cpp:2728-2754` — `OpaqueAttribute(data: bytes, typeName:
/// str = "")`, `.data` (READ-ONLY bytes, unlike `Bytes.data`), `.typeName`
/// (read-only str), `==`, `repr`.
#[pyclass(module = "OpenEXR", skip_from_py_object)]
#[derive(Clone)]
pub struct OpaqueAttribute {
    data: Vec<u8>,
    type_name: String,
}

#[pymethods]
impl OpaqueAttribute {
    #[new]
    #[pyo3(signature = (data, typeName=String::new()))]
    fn new(data: &Bound<'_, PyBytes>, typeName: String) -> Self {
        OpaqueAttribute { data: data.as_bytes().to_vec(), type_name: typeName }
    }

    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    #[getter]
    #[allow(non_snake_case)]
    fn typeName(&self) -> String {
        self.type_name.clone()
    }

    fn __eq__(&self, other: &OpaqueAttribute) -> bool {
        self.data == other.data && self.type_name == other.type_name
    }

    fn __repr__(&self) -> String {
        format!("<OpaqueAttribute type={}, data=b'...' ({} bytes)>", self.type_name, self.data.len())
    }
}

impl OpaqueAttribute {
    /// C-08c: `inner.type_name()` is raw bytes now (an unregistered type's
    /// on-disk name is never encoding-checked); `OpaqueAttribute.type_name`
    /// stays a Python `str` (this pyclass exposes it as one), so — matching
    /// `name_to_py_str`'s citation of real C++ PyOpenEXR's `py::str`
    /// (`PyUnicode_FromStringAndSize`, strict UTF-8) — a non-UTF-8 type name
    /// raises `UnicodeDecodeError` here instead of being mangled.
    pub(crate) fn from_core(inner: &exr_core::attr::OpaqueAttribute) -> PyResult<Self> {
        Ok(OpaqueAttribute {
            data: inner.data().to_vec(),
            type_name: crate::header::bytes_to_py_str(inner.type_name())?.to_string(),
        })
    }

    pub(crate) fn to_core(&self) -> exr_core::attr::OpaqueAttribute {
        exr_core::attr::OpaqueAttribute::with_data(&self.type_name, self.data.clone())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TileDescription>()?;
    m.add_class::<Rational>()?;
    m.add_class::<KeyCode>()?;
    m.add_class::<TimeCode>()?;
    m.add_class::<PreviewRgba>()?;
    m.add_class::<PreviewImage>()?;
    m.add_class::<Bytes>()?;
    m.add_class::<OpaqueAttribute>()?;
    m.add_function(wrap_pyfunction!(isOpenExrFile, m)?)?;
    m.add_function(wrap_pyfunction!(isTiledOpenExrFile, m)?)?;
    Ok(())
}
