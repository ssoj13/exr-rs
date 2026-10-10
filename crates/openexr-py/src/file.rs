//! P2/P3 — the numpy-based `File` + `Channel` classes (read / write).
//!
//! Mirrors the reference (`openexr/src/wrappers/python/PyOpenEXR.cpp`):
//! - `File(path, separate_channels=False, header_only=False)` reads a single-part
//!   scanline/tiled EXR; `channels()` returns `{name: Channel}` (R/G/B[/A] grouped
//!   into an interleaved `RGB`/`RGBA` array unless `separate_channels=True`),
//!   `header()` returns the header dict.
//! - `File(header: dict, channels: dict)` builds an image; `write(path)` saves it.
//!   Channel dict values may be raw numpy arrays (2-D `(H,W)`, or 3-D `(H,W,3|4)`
//!   for `RGB`/`RGBA`) or `Channel` objects.
//!
//! Pixel dtypes ↔ EXR pixel types: `float16`↔HALF, `float32`↔FLOAT, `uint32`↔UINT.
//!
//! Scope (this slice): single-part, full-resolution, flat channels, via the
//! `exr_core::api::Image` facade. Multipart / deep / tiled-write / the full header
//! attribute set are the next roadmap phases (see CLAUDE.md). Errors map to Python
//! exceptions — never a panic across the boundary.

use exr_core::ChannelData;
use exr_core::api::Image;
use exr_core::attr::{Compression, LineOrder, PixelType};
use imath_rs::{Box2f, Box2i, Half, V2f, V2i};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::enums;

/// Map an `exr_core::ExrError` (any variant — Arg/Input/Io/Logic/Overflow/Base,
/// all 1:1 ports of `Iex` exception classes, r3_surfaces.md item C-18 note in
/// design_r5.md §3.5) to a Python exception. The reference never registers a
/// pybind11 exception translator for `Iex`, so every one of these falls through
/// to pybind11's DEFAULT translation — `RuntimeError`, not `ValueError`
/// (`PyOpenEXR.cpp` only ever throws `std::invalid_argument` — which pybind11
/// DOES map to `ValueError` — directly from the BINDING's own argument
/// validation, via explicit `PyTypeError`/`PyValueError::new_err(...)` calls
/// elsewhere in this module, never for an error surfaced from exr-core itself).
fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ── numpy <-> ChannelData ──────────────────────────────────────────────────────

/// Build a 2-D `(H, W)` numpy array (float16/float32/uint32) from a channel buffer.
fn plane_to_py(py: Python<'_>, data: &ChannelData, h: usize, w: usize) -> PyResult<Py<PyAny>> {
    let obj = match data {
        ChannelData::Half(v) => {
            // imath Half -> numpy float16 by COPYING the 16-bit pattern (bit-exact,
            // incl. NaN payloads, and symmetric with the write path). The `half`
            // crate is only the numpy storage type, not a maths dependency.
            let f: Vec<half::f16> = v.iter().map(|x| half::f16::from_bits(x.bits())).collect();
            Array2::from_shape_vec((h, w), f).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
        ChannelData::Float(v) => {
            Array2::from_shape_vec((h, w), v.clone()).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
        ChannelData::Uint(v) => {
            Array2::from_shape_vec((h, w), v.clone()).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
    };
    Ok(obj)
}

/// Interleave several same-typed planes into a 3-D `(H, W, C)` numpy array (the
/// `RGB`/`RGBA` grouping). All `planes` must share a pixel type and length `h*w`.
fn interleave_to_py(py: Python<'_>, planes: &[&ChannelData], h: usize, w: usize) -> PyResult<Py<PyAny>> {
    let c = planes.len();
    let n = h * w;
    let obj = match planes[0] {
        ChannelData::Half(_) => {
            let mut out = vec![half::f16::from_bits(0); n * c];
            for (ci, p) in planes.iter().enumerate() {
                let ChannelData::Half(v) = p else { return Err(mixed_type()) };
                for i in 0..n {
                    out[i * c + ci] = half::f16::from_bits(v[i].bits());
                }
            }
            Array3::from_shape_vec((h, w, c), out).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
        ChannelData::Float(_) => {
            let mut out = vec![0.0f32; n * c];
            for (ci, p) in planes.iter().enumerate() {
                let ChannelData::Float(v) = p else { return Err(mixed_type()) };
                for i in 0..n {
                    out[i * c + ci] = v[i];
                }
            }
            Array3::from_shape_vec((h, w, c), out).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
        ChannelData::Uint(_) => {
            let mut out = vec![0u32; n * c];
            for (ci, p) in planes.iter().enumerate() {
                let ChannelData::Uint(v) = p else { return Err(mixed_type()) };
                for i in 0..n {
                    out[i * c + ci] = v[i];
                }
            }
            Array3::from_shape_vec((h, w, c), out).map_err(err)?.into_pyarray(py).into_any().unbind()
        }
    };
    Ok(obj)
}

fn mixed_type() -> PyErr {
    PyValueError::new_err(
        "channels grouped into RGB/RGBA must share a pixel type; pass \
         separate_channels=True to read them individually",
    )
}

/// Read a 2-D numpy array (float16/float32/uint32) into a `ChannelData` + its
/// `(height, width)`. Iterates the logical (row-major) order, so any layout works.
fn py_to_plane(obj: &Bound<'_, PyAny>) -> PyResult<(ChannelData, usize, usize)> {
    if let Ok(a) = obj.cast::<PyArray2<f32>>() {
        let s = a.shape();
        let v: Vec<f32> = a.readonly().as_array().iter().copied().collect();
        return Ok((ChannelData::Float(v), s[0], s[1]));
    }
    if let Ok(a) = obj.cast::<PyArray2<half::f16>>() {
        let s = a.shape();
        let v: Vec<Half> = a.readonly().as_array().iter().map(|x| Half::from_bits(x.to_bits())).collect();
        return Ok((ChannelData::Half(v), s[0], s[1]));
    }
    if let Ok(a) = obj.cast::<PyArray2<u32>>() {
        let s = a.shape();
        let v: Vec<u32> = a.readonly().as_array().iter().copied().collect();
        return Ok((ChannelData::Uint(v), s[0], s[1]));
    }
    Err(PyTypeError::new_err(
        "channel array must be a 2-D numpy array of dtype float16, float32, or uint32",
    ))
}

/// Split a 3-D `(H, W, C)` numpy array into `C` per-plane `ChannelData` buffers
/// (the inverse of [`interleave_to_py`]), returning them with `(height, width)`.
fn py_to_planes(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<ChannelData>, usize, usize)> {
    if let Ok(a) = obj.cast::<PyArray3<f32>>() {
        let s = a.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        let view = a.readonly();
        let arr = view.as_array();
        let planes = (0..c)
            .map(|ci| ChannelData::Float(arr.index_axis(numpy::ndarray::Axis(2), ci).iter().copied().collect()))
            .collect();
        return Ok((planes, h, w));
    }
    if let Ok(a) = obj.cast::<PyArray3<half::f16>>() {
        let s = a.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        let view = a.readonly();
        let arr = view.as_array();
        let planes = (0..c)
            .map(|ci| {
                ChannelData::Half(
                    arr.index_axis(numpy::ndarray::Axis(2), ci).iter().map(|x| Half::from_bits(x.to_bits())).collect(),
                )
            })
            .collect();
        return Ok((planes, h, w));
    }
    if let Ok(a) = obj.cast::<PyArray3<u32>>() {
        let s = a.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        let view = a.readonly();
        let arr = view.as_array();
        let planes = (0..c)
            .map(|ci| ChannelData::Uint(arr.index_axis(numpy::ndarray::Axis(2), ci).iter().copied().collect()))
            .collect();
        return Ok((planes, h, w));
    }
    Err(PyTypeError::new_err(
        "interleaved channel array must be a 3-D numpy array of dtype float16, float32, or uint32",
    ))
}

/// The pixel type of a numpy array of any rank (float16/float32/uint32); used to
/// wrap a raw channel-dict array into a `Channel` object where a 3-D interleaved
/// `RGB`/`RGBA` array is just as valid as a 2-D one (unlike [`array_ptype`], which
/// is 2-D-only for a `Channel(...)` constructor call).
/// `PyChannel::pixelType`'s object-dtype branch (`PyOpenEXR.cpp:2571-2586`): scan
/// a 2-D `dtype=object` array for the first non-`None` element and report ITS
/// pixel type; `NUM_PIXELTYPES` if every element is `None` (matches the
/// reference's fall-through, not an error).
fn deep_array_pixel_type(pixels: &Bound<'_, PyAny>) -> PyResult<enums::PixelType> {
    let shape: Vec<usize> = pixels.getattr("shape")?.extract()?;
    let (h, w) = (*shape.first().unwrap_or(&0), *shape.get(1).unwrap_or(&0));
    for y in 0..h {
        for x in 0..w {
            let elem = pixels.call_method1("__getitem__", ((y, x),))?;
            if elem.is_none() {
                continue;
            }
            if let Ok(pt) = array_pixel_type_any(&elem) {
                return Ok(pixeltype_to_py(pt));
            }
        }
    }
    Ok(enums::PixelType::NUM_PIXELTYPES)
}

/// `PyChannel::validatePixelArray` (`PyOpenEXR.cpp:2422-2461`), ported 1:1 —
/// called from every reference `PyChannel` constructor that takes an array, so
/// this is invoked both from `Channel::new` and from `normalize_channels`'s
/// "wrap any array into a Channel" path (`PyPart::PyPart`'s equivalent,
/// `:2389-2394`). The reference throws `std::invalid_argument` for every case
/// here, which pybind11 maps to `ValueError` — NOT `TypeError` (team-lead
/// correction 2026-09-23).
///
/// Returns the element pixel type for a plain numeric array, or
/// `PixelType::NumPixelTypes` for a (validated) `dtype=object` deep array — the
/// real per-pixel type for THOSE is computed lazily by `Channel::type_()`
/// (`deep_array_pixel_type`), never cached here, matching
/// `PyChannel::pixelType`.
fn validate_pixel_array(pixels: &Bound<'_, PyAny>) -> PyResult<PixelType> {
    let ndim: usize = pixels.getattr("ndim")?.extract()?;
    if !(2..=3).contains(&ndim) {
        return Err(PyValueError::new_err(
            "invalid pixel array: must be 2D or 3D numpy array",
        ));
    }
    let kind: String = pixels.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "O" {
        let shape: Vec<usize> = pixels.getattr("shape")?.extract()?;
        let (h, w) = (*shape.first().unwrap_or(&0), *shape.get(1).unwrap_or(&0));
        for y in 0..h {
            for x in 0..w {
                let elem = pixels.call_method1("__getitem__", ((y, x),))?;
                if elem.is_none() || array_pixel_type_any(&elem).is_ok() {
                    continue;
                }
                let msg = if elem.cast::<numpy::PyUntypedArray>().is_ok() {
                    let dtype: String = elem.getattr("dtype")?.str()?.extract()?;
                    format!("invalid deep pixel array: entry at {y},{x} is array of unsupported type '{dtype}'")
                } else {
                    let cls: String = elem.get_type().name()?.to_string();
                    format!("invalid deep pixel array: entry at {y},{x} is of unsupported type '{cls}'")
                };
                return Err(PyValueError::new_err(msg));
            }
        }
        return Ok(PixelType::NumPixelTypes);
    }
    array_pixel_type_any(pixels).map_err(|_| {
        let cls = pixels.get_type().name().map(|n| n.to_string()).unwrap_or_default();
        PyValueError::new_err(format!("invalid pixel array: unsupported type {cls}"))
    })
}

fn array_pixel_type_any(arr: &Bound<'_, PyAny>) -> PyResult<PixelType> {
    if arr.cast::<numpy::PyArrayDyn<f32>>().is_ok() {
        Ok(PixelType::Float)
    } else if arr.cast::<numpy::PyArrayDyn<half::f16>>().is_ok() {
        Ok(PixelType::Half)
    } else if arr.cast::<numpy::PyArrayDyn<u32>>().is_ok() {
        Ok(PixelType::Uint)
    } else {
        Err(PyTypeError::new_err(
            "Channel pixels must be a numpy array of dtype float16, float32, or uint32",
        ))
    }
}

/// Normalise a channels dict the way the reference does when a `File`/`Part` is
/// constructed for writing (`PyOpenEXR.cpp`'s `channels_to_pychannels`, r3_surfaces
/// §3.4 item 3): every raw numpy array becomes a `Channel(name, array)`, and every
/// `Channel` whose `name` is still empty (`Channel(array, 1, 1)`, no name argument)
/// gets the dict key as its name. A `Channel` that already has a name keeps it.
/// Every header dict key must be a string (attribute name) — the reference raises
/// at construction time (`PyPart::PyPart`, r3_surfaces.md item 3 /
/// `test_exceptions.py`'s `File({1:1}, {})`), not later at write time when a
/// non-string key would otherwise just be silently skipped by `py_to_attrs`.
fn validate_header_keys(header: &Bound<'_, PyDict>) -> PyResult<()> {
    for (k, _) in header.iter() {
        if k.cast::<PyString>().is_err() {
            return Err(PyTypeError::new_err("header key must be string (attribute name)"));
        }
    }
    Ok(())
}

fn normalize_channels(py: Python<'_>, channels: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in channels.iter() {
        let name: String = key.extract().map_err(|_| {
            PyTypeError::new_err("channel dict keys must be strings")
        })?;
        if let Ok(ch) = value.cast::<Channel>() {
            if ch.borrow().name.is_empty() {
                ch.borrow_mut().name = name.clone();
            }
            out.set_item(&name, &value)?;
        } else if value.cast::<numpy::PyUntypedArray>().is_ok() {
            // Any numpy array — numeric (float16/float32/uint32) OR `dtype=object`
            // (deep write: per-pixel sample arrays / `None`) — becomes a named
            // `Channel`, matching the reference exactly: `PyPart::PyPart` wraps
            // ANY `py::array` value the same way (`PyOpenEXR.cpp:2389-2394`,
            // `channels[name] = PyChannel(name, a)`), with no dtype special-case.
            // `Channel::type_()` recomputes the real pixel type lazily for a
            // `dtype=object` array (scans for the first non-`None` sample), so
            // the placeholder `NumPixelTypes` `validate_pixel_array` returns for
            // one is never actually read.
            let ptype = validate_pixel_array(&value)?;
            out.set_item(&name, make_channel(py, &name, value.clone().unbind(), ptype, 0)?)?;
        } else {
            return Err(PyTypeError::new_err(format!(
                "channel \"{name}\" must be a numpy array or Channel"
            )));
        }
    }
    Ok(out.unbind())
}

// ── Channel ────────────────────────────────────────────────────────────────────

/// One image channel: a name + its pixel array, mirroring the reference `Channel`.
#[pyclass(module = "OpenEXR")]
pub struct Channel {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub pixels: Py<PyAny>,
    #[pyo3(get, set)]
    pub xSampling: i32,
    #[pyo3(get, set)]
    pub ySampling: i32,
    #[pyo3(get, set)]
    pub pLinear: bool,
    #[pyo3(get)]
    pub channel_index: i32,
    /// The EXR pixel type (also reported by `type()`).
    ptype: PixelType,
}

#[pymethods]
impl Channel {
    /// The reference's overloaded `Channel(...)`: an optional leading `name` (str),
    /// an optional 2-D `pixels` numpy array (float16/float32/uint32), then either
    /// nothing or `(xSampling, ySampling[, pLinear])`. Anything else raises — e.g.
    /// `Channel(1)`, `Channel("C", 2)` (a lone sampling arg), or a 1-D / bad-dtype
    /// pixels array.
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<'_, PyTuple>, py: Python<'_>) -> PyResult<Self> {
        let items: Vec<Bound<'_, PyAny>> = args.iter().collect();
        let mut i = 0usize;

        let mut name = String::new();
        if i < items.len() && items[i].cast::<PyString>().is_ok() {
            name = items[i].extract()?;
            i += 1;
        }

        let mut pixels = py.None();
        let mut ptype = PixelType::Half;
        if i < items.len() && items[i].cast::<PyUntypedArray>().is_ok() {
            // A 2-D `(H,W)` array (a single channel), a 3-D `(H,W,3|4)` array (an
            // interleaved RGB/RGBA channel — r3_surfaces.md item 2, the reference's
            // `Channel(rgb_array)` with no name is exactly how `test_rgba.py`'s
            // `do_rgb`/`do_rgba`/`do_rgba_prefix` construct every channel), or a
            // `dtype=object` deep array — `validate_pixel_array` is
            // `PyChannel::validatePixelArray` ported 1:1 (rank 2-or-3; for
            // `dtype=object` every element must be `None` or a uint32/half/float
            // array), so it alone decides what's acceptable here, not a
            // last-dim-3-or-4 special case this used to hard-code.
            ptype = validate_pixel_array(&items[i])?;
            pixels = items[i].clone().unbind();
            i += 1;
        }

        let (xSampling, ySampling, pLinear) = match items.len() - i {
            0 => (1, 1, false),
            2 => (items[i].extract()?, items[i + 1].extract()?, false),
            3 => (items[i].extract()?, items[i + 1].extract()?, items[i + 2].extract()?),
            _ => {
                return Err(PyTypeError::new_err(
                    "Channel(...): expected optional name, optional 2-D pixels array, \
                     then (xSampling, ySampling[, pLinear])",
                ));
            }
        };
        Ok(Channel {
            name,
            pixels,
            xSampling,
            ySampling,
            pLinear,
            channel_index: 0,
            ptype,
        })
    }

    /// The channel's EXR [`enums::PixelType`] (UINT / HALF / FLOAT). Exposed under
    /// the reference's name `type` (a Rust keyword, hence the pyo3 rename).
    ///
    /// For a `dtype=object` (deep) `pixels` array, the reference computes this
    /// LAZILY by scanning for the first non-`None` per-pixel sample array and
    /// reporting ITS dtype (`PyChannel::pixelType`, `PyOpenEXR.cpp:2557-2589`) —
    /// `NUM_PIXELTYPES` if every pixel is `None`. A plain numeric array still
    /// reports the fixed `ptype` recorded at construction, as before.
    #[pyo3(name = "type")]
    fn type_(&self, py: Python<'_>) -> PyResult<enums::PixelType> {
        let pixels = self.pixels.bind(py);
        if pixels.is_none() {
            return Ok(pixeltype_to_py(self.ptype));
        }
        if let Ok(dtype) = pixels.getattr("dtype")
            && dtype.getattr("kind").ok().and_then(|k| k.extract::<String>().ok()).as_deref() == Some("O")
        {
            return deep_array_pixel_type(pixels);
        }
        Ok(pixeltype_to_py(self.ptype))
    }

    fn __repr__(&self) -> String {
        format!("Channel(\"{}\", {:?})", self.name, self.ptype)
    }
}

// The reference method is literally `type` (a Python keyword in Rust), so expose it
// under that name explicitly.
#[doc(hidden)]
fn pixeltype_to_py(p: PixelType) -> enums::PixelType {
    match p {
        PixelType::Uint => enums::PixelType::UINT,
        PixelType::Half => enums::PixelType::HALF,
        PixelType::Float => enums::PixelType::FLOAT,
        // The `NumPixelTypes` sentinel never tags a real channel; map it for
        // exhaustiveness only.
        PixelType::NumPixelTypes => enums::PixelType::NUM_PIXELTYPES,
    }
}

// ── Part ──────────────────────────────────────────────────────

/// One image part: name / storage type / dimensions + its header and channels.
#[pyclass(module = "OpenEXR")]
pub struct Part {
    name_: Option<String>,
    width_: i32,
    height_: i32,
    #[pyo3(get, set)]
    header: Py<PyDict>,
    #[pyo3(get, set)]
    channels: Py<PyDict>,
    #[pyo3(get)]
    part_index: i32,
}

#[pymethods]
impl Part {
    /// `Part(header={}, channels={}, name="")` — construct a part for writing
    /// (all arguments optional: `Part()` is an empty part).
    #[new]
    #[pyo3(signature = (header=None, channels=None, name=""))]
    fn new(
        header: Option<&Bound<'_, PyDict>>,
        channels: Option<&Bound<'_, PyDict>>,
        name: &str,
        py: Python<'_>,
    ) -> PyResult<Self> {
        if let Some(h) = header {
            validate_header_keys(h)?;
        }
        let header = header.map(|h| h.clone().unbind()).unwrap_or_else(|| PyDict::new(py).unbind());
        let channels = match channels {
            Some(c) => normalize_channels(py, c)?,
            None => PyDict::new(py).unbind(),
        };
        Ok(Part {
            name_: if name.is_empty() { None } else { Some(name.to_string()) },
            width_: 0,
            height_: 0,
            header,
            channels,
            part_index: 0,
        })
    }
    /// `PyPart::name()` (`PyOpenEXR.cpp:2515-2520`): `""` when the part has no
    /// `name` attribute, never `None` (`test_write_tiled`'s
    /// `assert o.parts[0].name() == ""`).
    fn name(&self) -> String {
        self.name_.clone().unwrap_or_default()
    }
    /// Storage kind — from `header["type"]` (default `scanlineimage`).
    #[pyo3(name = "type")]
    fn type_(&self, py: Python<'_>) -> enums::Storage {
        self.header
            .bind(py)
            .get_item("type")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<enums::Storage>().ok())
            .unwrap_or(enums::Storage::scanlineimage)
    }
    /// Compression — from `header["compression"]` (default `ZIP_COMPRESSION`).
    fn compression(&self, py: Python<'_>) -> PyResult<enums::Compression> {
        comp_to_py(header_compression(self.header.bind(py))?)
    }
    fn width(&self) -> i32 {
        self.width_
    }
    fn height(&self) -> i32 {
        self.height_
    }
    fn __repr__(&self) -> String {
        format!("Part(name={:?}, {}x{})", self.name_, self.width_, self.height_)
    }
}

// ── File ───────────────────────────────────────────────────────────────────────

/// The reference `File`: read an EXR, or hold a header+channels to write.
#[pyclass(module = "OpenEXR")]
pub struct File {
    #[pyo3(get, set)]
    pub filename: String,
    /// Header dict (read: populated from the file; write: as given).
    header: Py<PyDict>,
    /// Grouped channels `{name: Channel}` (read) or the to-write dict (write).
    channels: Py<PyDict>,
    /// The image parts (single-part files have exactly one). Read-write, matching
    /// the reference (`PyOpenEXR.cpp:2698`): reassigning it changes what `write()`
    /// writes.
    #[pyo3(get, set)]
    parts: Py<PyList>,
}

#[pymethods]
impl File {
    /// The reference (`PyOpenEXR.cpp`) registers FOUR distinct pybind11 constructor
    /// overloads and lets pybind11 pick one by argument TYPE, not by a shared
    /// positional slot:
    ///   - `File()` — empty.
    ///   - `File(filename: str, separate_channels: bool=False, header_only: bool=False)` — read.
    ///   - `File(header: dict, channels: dict)` — construct for writing.
    ///   - `File(parts: list)` — multi-part from a list of `Part`.
    /// A single PyO3 `#[pyo3(signature = ...)]` binds fixed *names* to fixed
    /// *positions*, so position 2 can't be "channels" for one overload and
    /// "separate_channels" for another — `File(path, False)` used to hard-TypeError
    /// ("argument 'channels': 'bool' object is not an instance of 'dict'",
    /// `test_tiled_deep`) because position 2 was hard-wired to the dict-typed
    /// `channels` parameter. Dispatch on `*args`/`**kwargs` by hand instead, exactly
    /// mirroring pybind11's per-overload argument types.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>, py: Python<'_>) -> PyResult<Self> {
        let kw = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            match kwargs {
                Some(d) => d.get_item(name),
                None => Ok(None),
            }
        };
        // File(parts=[Part, ...]) — the keyword-argument spelling of File([Part, ...]).
        if let Some(v) = kw("parts")? {
            let list = v.cast::<PyList>().map_err(|_| PyTypeError::new_err("parts must be a list"))?;
            return File::from_parts(py, list);
        }
        let pos: Vec<Bound<'_, PyAny>> = args.iter().collect();
        let arg0 = match pos.first().cloned() {
            Some(v) => Some(v),
            None => kw("header")?,
        };
        // File() — an empty file (can be assigned to later).
        let Some(arg0) = arg0 else {
            return Ok(File::empty(py));
        };
        // File([Part, ...]) — a multi-part file from a list of parts.
        if let Ok(list) = arg0.cast::<PyList>() {
            return File::from_parts(py, list);
        }
        // Construct-for-write form: File(header_dict, channels_dict).
        if let Ok(header) = arg0.cast::<PyDict>() {
            let chans_any = match pos.get(1).cloned() {
                Some(v) => Some(v),
                None => kw("channels")?,
            };
            let chans_any = chans_any.ok_or_else(|| {
                PyTypeError::new_err("File(header, channels): the channels dict is required")
            })?;
            let chans = chans_any
                .cast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("File(header, channels): channels must be a dict"))?;
            validate_header_keys(header)?;
            let chans = normalize_channels(py, chans)?;
            let part = Py::new(
                py,
                Part {
                    name_: None,
                    width_: 0,
                    height_: 0,
                    header: header.clone().unbind(),
                    channels: chans.clone_ref(py),
                    part_index: 0,
                },
            )?;
            return Ok(File {
                filename: String::new(),
                header: header.clone().unbind(),
                channels: chans,
                parts: PyList::new(py, [part])?.unbind(),
            });
        }
        // Read form: File(path, separate_channels=False, header_only=False) —
        // `separate_channels`/`header_only` may be given positionally (2nd/3rd arg)
        // OR as keywords; a positional bool never binds to `channels` in this form.
        let path: String = arg0.extract().map_err(|_| {
            PyTypeError::new_err("File(...): first argument must be a path (str) or a header dict")
        })?;
        let bool_arg = |idx: usize, name: &str| -> PyResult<bool> {
            match pos.get(idx).cloned() {
                Some(v) => v.extract(),
                None => match kw(name)? {
                    Some(v) => v.extract(),
                    None => Ok(false),
                },
            }
        };
        let separate_channels = bool_arg(1, "separate_channels")?;
        let header_only = bool_arg(2, "header_only")?;
        read_path(py, path, separate_channels, header_only)
    }

    /// The header dictionary of part `part` (default 0). Raises on an empty file or
    /// an out-of-range index.
    #[pyo3(signature = (part=0))]
    fn header(&self, part: i32, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.part_dict(py, part, true)
    }

    /// The channels dictionary `{name: Channel}` of part `part` (default 0).
    #[pyo3(signature = (part=0))]
    fn channels(&self, part: i32, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.part_dict(py, part, false)
    }

    /// Write the held header + channels to `path` (single-part scanline).
    fn write(&self, path: &str, py: Python<'_>) -> PyResult<()> {
        // Deep write (header type deepscanline/deeptile).
        if is_deep_storage(self.header.bind(py))? {
            return crate::deep::write_deep(py, path, self.header.bind(py), self.channels.bind(py));
        }
        let parts = self.parts.bind(py);
        // Multi-part write: one scanline part per `Part`.
        if parts.len() > 1 {
            let mut out: Vec<(Image, Compression, String)> = Vec::with_capacity(parts.len());
            for (i, item) in parts.iter().enumerate() {
                let p = item.cast::<Part>().map_err(|e| {
                    pyo3::exceptions::PyTypeError::new_err(e.to_string())
                })?;
                let b = p.borrow();
                // `File(path, header_only=True)` loads no pixels at all (that is
                // the point of the flag), but the reference still lets a caller
                // edit ONLY the header and write the file back with its original
                // pixels untouched (`test_header_only`'s "Test modifying header
                // when reading just the header"). If this part's channels dict is
                // still empty and we know the file we were read from, lazily
                // re-decode just that part's pixels now, at write time, instead
                // of requiring the caller to have loaded them.
                let channels_dict = if b.channels.bind(py).is_empty() && !self.filename.is_empty() {
                    reload_part_channels(py, &self.filename, b.part_index)?
                } else {
                    b.channels.clone_ref(py)
                };
                let mut image = assemble_image(py, b.header.bind(py), channels_dict.bind(py))?;
                let compression = header_compression(b.header.bind(py))?;
                let name = b.name_.clone().unwrap_or_else(|| format!("Part{i}"));
                // Carry each part's own custom header metadata through — this
                // used to be silently dropped entirely (r3_surfaces.md item 6 /
                // `test_header_only`'s `header["foo"]`/`header["n"]`), the same
                // gap the single-part path below already closed via
                // `write_with_attrs`. `ImageAttributes::insert` itself rejects
                // any DERIVED name (`type`/`tiles`/`name`/…), so this can't be
                // (ab)used to override the scanline type `write_multipart_scanline`
                // requires.
                for (attr_name, attr) in crate::header::py_to_attrs(b.header.bind(py))? {
                    // "tiles" is NOT in `STANDARD_KEYS` (the single-part path
                    // needs it to flow through — see that constant's doc), but
                    // `write_multipart_scanline` always forces a scanline part
                    // (r3_surfaces.md item 5/6: multipart tiled write isn't
                    // supported), so `ImageAttributes::insert` would reject it
                    // as derived here. Drop it silently rather than fail the
                    // whole multipart write over a part that can't be tiled
                    // anyway.
                    if attr_name == "tiles" {
                        continue;
                    }
                    image.attributes_mut().insert(&attr_name, attr).map_err(err)?;
                }
                out.push((image, compression, name));
            }
            exr_core::api::write_multipart_scanline(path, out).map_err(err)?;
            return Ok(());
        }
        // Single-part write, carrying arbitrary header metadata (comments, screen
        // window, customs, …) through, not just the windows/compression the Image
        // models.
        let image = assemble_image(py, self.header.bind(py), self.channels.bind(py))?;
        let compression = header_compression(self.header.bind(py))?;
        let extra = crate::header::py_to_attrs(self.header.bind(py))?;
        image.write_with_attrs(path, compression, extra).map_err(err)?;
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> bool {
        false
    }
}

/// Read a file path into a `File`, dispatching single-part vs multi-part on flags.
fn read_path(py: Python<'_>, path: String, separate: bool, header_only: bool) -> PyResult<File> {
    let flags = peek_flags(&path)?;
    if exr_core::container::version::is_non_image(flags) {
        // Deep file.
        let (header, channels, w, h) = crate::deep::read_deep(py, &path, separate)?;
        let part = mk_part(py, header.clone_ref(py), channels.clone_ref(py), w, h, None, 0)?;
        return Ok(File { filename: path, header, channels, parts: PyList::new(py, [part])?.unbind() });
    }
    if exr_core::container::version::is_multi_part(flags) {
        read_multipart(py, path, separate, header_only)
    } else {
        read_single(py, path, separate, header_only)
    }
}

/// True if the header's `type` is a deep storage kind (deepscanline / deeptile).
fn is_deep_storage(header: &Bound<'_, PyDict>) -> PyResult<bool> {
    match header.get_item("type")? {
        Some(v) => {
            let code = pyany_to_i32(&v).unwrap_or(0);
            Ok(code == 2 || code == 3) // deepscanline=2, deeptile=3
        }
        None => Ok(false),
    }
}

/// Peek the version flags from the first 8 bytes of `path`.
pub(crate) fn peek_flags(path: &str) -> PyResult<u32> {
    let mut f = std::fs::File::open(path).map_err(err)?;
    let mut buf = [0u8; 8];
    std::io::Read::read_exact(&mut f, &mut buf).map_err(err)?;
    Ok(exr_core::container::version::get_flags(u32::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7],
    ])))
}

/// Single-part read: pixels via the `Image` facade, full header via the low-level
/// reader; one `Part`.
fn read_single(py: Python<'_>, path: String, separate: bool, header_only: bool) -> PyResult<File> {
    // C-08: the header dict (attribute + channel names via `py::str`) is built
    // BEFORE any pixel decode, matching real C++ PyOpenEXR (`PyOpenEXR.cpp`'s
    // `PyFile` constructor fills `P.header` from `header.begin()/end()` first,
    // UNCAUGHT, then only afterwards attempts pixel reads inside a
    // try/catch that swallows per-part decode failures — verified by reading
    // `PyOpenEXR.cpp` v3.4.15 directly, `:178-231`). A non-UTF-8 attribute
    // name must therefore raise `UnicodeDecodeError` (from `read_full_header`)
    // even when the pixel data itself would otherwise decode fine; the old
    // order let a hard pixel-facade error (e.g. a non-UTF-8 CHANNEL name,
    // `ExrError::Input` from `api::Image::from_header`) mask that with a
    // generic `RuntimeError` instead. `read_multipart` below already builds
    // its per-part header dict first for the same reason (see its own
    // `PyOpenEXR.cpp:230-238` citation) -- this brings `read_single` in line
    // with it instead of leaving the two paths order pixel vs. header work
    // differently.
    let header = read_full_header(py, &path)?;
    // py-parity2 exr-core need (b): `header_only=True` must not decode any
    // pixels -- `Image::read_header_only` parses the header + chunk offset
    // table only (no `read_pixels`/`read_tiles`), unlike the full
    // `Image::read` this used to call unconditionally regardless of the flag.
    //
    // A single-part file's pixel decode is inside the SAME uncaught-header/
    // caught-pixels region real C++ uses uniformly for every part (`PyFile`'s
    // constructor loops `_inputFile->parts()` generically, single-part or
    // not; `:186-231`): a decode failure (corrupt chunk, truncated file, …)
    // is caught (`catch (const std::exception&) { py::print("Warning: ...");
    // }`, `:227-229`) and the part is dropped, not raised. `read_multipart`
    // below already does this (`.ok()` + `warnings.warn`, citing the same
    // lines); this brings `read_single` in line with it instead of hard-
    // failing the whole `File()` call on what C++ treats as "no readable
    // parts", matching PyOpenEXR's drop-in contract (a `warnings.warn` is
    // used instead of C++'s `printf` to stdout — visible and filterable,
    // strictly better, not a behavioural gap).
    if header_only {
        let name = header.bind(py).get_item("name")?.and_then(|v| v.extract::<String>().ok());
        let channels = PyDict::new(py).unbind();
        let part = mk_part(py, header.clone_ref(py), channels.clone_ref(py), 0, 0, name, 0)?;
        return Ok(File { filename: path, header, channels, parts: PyList::new(py, [part])?.unbind() });
    }
    // Only the PIXEL DECODE itself (`Image::read`'s chunk-table walk, the
    // analogue of C++'s `readPixels`/`readTiles`) is caught-and-skipped.
    // `build_channels`'s own coalesce validation (duplicate/mixed-type "R"/
    // "G"/"B" keys, …) is the analogue of C++'s `validateCoalescedChannels`,
    // called BEFORE the try/catch in `PyOpenEXR.cpp` (`:196-197`, versus the
    // try block at `:206-231`) and therefore raises uncaught, exactly like a
    // malformed channel layout must still raise `ValueError` here (proven by
    // the official `test_mixed_type_rgb_coalesce_flat_rejected` / `test_
    // literal_rgb_key_collision_flat_rejected`, which regressed to a silent
    // skip when this was first wrapped in the same `.ok()` as the decode).
    let image = match Image::read(&path) {
        Ok(image) => image,
        Err(_) => {
            let name = header.bind(py).get_item("name")?.and_then(|v| v.extract::<String>().ok());
            let label = name.unwrap_or_else(|| "the image".to_string());
            py.import("warnings")?.call_method1(
                "warn",
                (format!("{path}: {label} has incomplete or unreadable pixel data; skipping"),),
            )?;
            // Matches `read_multipart`'s contract when every part fails to
            // decode: the top-level `File.header`/`.channels` fall back to
            // empty dicts (no part loaded to source them from), `.parts` empty.
            return Ok(File {
                filename: path,
                header: PyDict::new(py).unbind(),
                channels: PyDict::new(py).unbind(),
                parts: PyList::empty(py).unbind(),
            });
        }
    };
    let channels = build_channels(py, &image, separate)?;
    let name = header.bind(py).get_item("name")?.and_then(|v| v.extract::<String>().ok());
    let part = mk_part(py, header.clone_ref(py), channels.clone_ref(py), image.width() as i32, image.height() as i32, name, 0)?;
    Ok(File { filename: path, header, channels, parts: PyList::new(py, [part])?.unbind() })
}

/// Multi-part read: each part via `MultiPartInputFile` — header + (scanline/tiled)
/// pixels through the facade's `from_input_file` / `from_tiled_input_file`. Deep
/// parts contribute their header but no pixels yet.
/// Re-decode part `part_index`'s pixels from `path` on demand (not-`separate`,
/// grouped RGB/RGBA channels) — the write-time fallback for a `Part` whose
/// channels dict is still empty because it was loaded with `header_only=True`.
/// See the call site in `File::write`'s multipart branch.
fn reload_part_channels(py: Python<'_>, path: &str, part_index: i32) -> PyResult<Py<PyDict>> {
    let mut stream = exr_core::io::StdIfStream::open(path).map_err(err)?;
    let mut mp = exr_core::MultiPartInputFile::new(&mut stream).map_err(err)?;
    let ptype: String = mp.part_type(part_index).map_err(err)?.to_string();
    match ptype.as_str() {
        "scanlineimage" => {
            let mut inf = mp.input_part(part_index).map_err(err)?;
            let img = Image::from_input_file(&mut inf).map_err(err)?;
            build_channels(py, &img, false)
        }
        "tiledimage" => {
            let mut inf = mp.tiled_input_part(part_index).map_err(err)?;
            let img = Image::from_tiled_input_file(&mut inf).map_err(err)?;
            build_channels(py, &img, false)
        }
        _ => Err(err(format!(
            "cannot write back part {part_index}: header_only lazy-reload isn't \
             implemented for storage type \"{ptype}\""
        ))),
    }
}

fn read_multipart(py: Python<'_>, path: String, separate: bool, header_only: bool) -> PyResult<File> {
    let mut stream = exr_core::io::StdIfStream::open(&path).map_err(err)?;
    let mut mp = exr_core::MultiPartInputFile::new(&mut stream).map_err(err)?;
    let nparts = mp.parts();
    let parts = PyList::empty(py);
    let mut first: Option<(Py<PyDict>, Py<PyDict>)> = None;
    for n in 0..nparts {
        let header_dict = {
            let h = mp.header(n).map_err(err)?;
            crate::header::to_dict(py, h)?
        };
        let ptype: String = mp.part_type(n).map_err(err)?.to_string();
        header_dict.bind(py).set_item("type", storage_from_type(&ptype))?;
        // An interrupted write can list more part headers than have valid pixel
        // data. `header_only` always returns every part (headers cost nothing to
        // read). Otherwise, a part whose reader reports `!is_complete()` — or
        // whose pixel decode fails outright — is SKIPPED entirely, not raised:
        // "loading should return only parts whose pixels read successfully, not
        // fail the entire file open" (`test_unfinished_multipart_load`, matching
        // the reference's `PyOpenEXR.cpp:230-238` per-part warn-and-skip).
        // Only the PIXEL DECODE itself (`Image::from_input_file`/
        // `from_tiled_input_file`, the analogue of C++'s `readPixels`/
        // `readTiles`) is caught-and-skipped below. `build_channels`'s own
        // coalesce validation (duplicate/mixed-type "R"/"G"/"B" keys, …) is
        // the analogue of C++'s `validateCoalescedChannels`, called BEFORE
        // the try/catch in `PyOpenEXR.cpp` (`:196-197`, versus the try block
        // at `:206-231`), and therefore propagates with `?` — a malformed
        // channel layout in one part must still raise `ValueError`, not be
        // silently skipped (see `read_single`'s identical split and the two
        // official tests that caught the same bug there when this was first
        // wrapped in one `.ok()`).
        let decoded: Option<(Py<PyDict>, i32, i32)> = if header_only {
            Some((PyDict::new(py).unbind(), 0, 0))
        } else {
            match ptype.as_str() {
                "scanlineimage" => match mp.input_part(n) {
                    Ok(mut inf) if inf.is_complete() => match Image::from_input_file(&mut inf) {
                        Ok(img) => {
                            let channels = build_channels(py, &img, separate)?;
                            Some((channels, img.width() as i32, img.height() as i32))
                        }
                        Err(_) => None,
                    },
                    _ => None,
                },
                "tiledimage" => match mp.tiled_input_part(n) {
                    Ok(mut inf) if inf.is_complete() => match Image::from_tiled_input_file(&mut inf) {
                        Ok(img) => {
                            let channels = build_channels(py, &img, separate)?;
                            Some((channels, img.width() as i32, img.height() as i32))
                        }
                        Err(_) => None,
                    },
                    _ => None,
                },
                _ => Some((PyDict::new(py).unbind(), 0, 0)), // deep: header only for now
            }
        };
        let Some((channels, w, h)) = decoded else {
            // Never silently discard: the reference prints a warning for a
            // skipped part (`PyOpenEXR.cpp:230-238`); do the same instead of
            // dropping it with no trace.
            let name = header_dict.bind(py).get_item("name")?.and_then(|v| v.extract::<String>().ok());
            let label = name.unwrap_or_else(|| format!("part {n}"));
            py.import("warnings")?.call_method1(
                "warn",
                (format!("{path}: {label} has incomplete or unreadable pixel data; skipping"),),
            )?;
            continue; // incomplete/failed part: drop it from the part list
        };
        let name = header_dict.bind(py).get_item("name")?.and_then(|v| v.extract::<String>().ok());
        let part = mk_part(py, header_dict.clone_ref(py), channels.clone_ref(py), w, h, name, n)?;
        if first.is_none() {
            first = Some((header_dict.clone_ref(py), channels.clone_ref(py)));
        }
        parts.append(part)?;
    }
    let (header, channels) =
        first.unwrap_or_else(|| (PyDict::new(py).unbind(), PyDict::new(py).unbind()));
    Ok(File { filename: path, header, channels, parts: parts.unbind() })
}

fn storage_from_type(s: &str) -> enums::Storage {
    match s {
        "tiledimage" => enums::Storage::tiledimage,
        "deepscanline" => enums::Storage::deepscanline,
        "deeptile" => enums::Storage::deeptile,
        _ => enums::Storage::scanlineimage,
    }
}

/// Build a `Part` object.
fn mk_part(
    py: Python<'_>,
    header: Py<PyDict>,
    channels: Py<PyDict>,
    width: i32,
    height: i32,
    name: Option<String>,
    index: i32,
) -> PyResult<Py<Part>> {
    Py::new(
        py,
        Part {
            name_: name,
            width_: width,
            height_: height,
            header,
            channels,
            part_index: index,
        },
    )
}

/// Plain (non-Python) helpers for `File`.
impl File {
    /// An empty file: no name, no parts; `header()`/`channels()` then raise.
    fn empty(py: Python<'_>) -> Self {
        File {
            filename: String::new(),
            header: PyDict::new(py).unbind(),
            channels: PyDict::new(py).unbind(),
            parts: PyList::empty(py).unbind(),
        }
    }

    /// `File([Part, ...])` — validate every element is a `Part`.
    fn from_parts(py: Python<'_>, list: &Bound<'_, PyList>) -> PyResult<Self> {
        let parts = PyList::empty(py);
        for item in list.iter() {
            let p = item.cast::<Part>().map_err(|_| {
                PyTypeError::new_err("File(parts): every element must be a Part")
            })?;
            parts.append(p)?;
        }
        let (header, channels) = if parts.is_empty() {
            (PyDict::new(py).unbind(), PyDict::new(py).unbind())
        } else {
            let p0 = parts.get_item(0)?;
            let p0 = p0.cast::<Part>().map_err(|e| PyTypeError::new_err(e.to_string()))?;
            let b = p0.borrow();
            (b.header.clone_ref(py), b.channels.clone_ref(py))
        };
        Ok(File {
            filename: String::new(),
            header,
            channels,
            parts: parts.unbind(),
        })
    }

    /// The header or channels dict of part `part`, bounds-checked.
    fn part_dict(&self, py: Python<'_>, part: i32, header: bool) -> PyResult<Py<PyDict>> {
        let parts = self.parts.bind(py);
        let n = parts.len() as i32;
        if n == 0 {
            return Err(PyValueError::new_err("File has no parts"));
        }
        if part < 0 || part >= n {
            return Err(PyValueError::new_err(format!(
                "invalid part index {part} (file has {n} part(s))"
            )));
        }
        let item = parts.get_item(part as usize)?;
        let p = item.cast::<Part>().map_err(|e| PyTypeError::new_err(e.to_string()))?;
        let b = p.borrow();
        Ok(if header { b.header.clone_ref(py) } else { b.channels.clone_ref(py) })
    }
}

/// Read the full header dict from `path` (all attributes). Dispatches on the file's
/// version flags to the right low-level reader (single-part scanline/tiled, or
/// multipart) — `MultiPartInputFile` rejects non-multipart files, so we cannot use
/// it uniformly.
fn read_full_header(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    use exr_core::container::version;
    // Peek the version flags (first 8 bytes) to choose the reader.
    let flags = {
        let mut f = std::fs::File::open(path).map_err(err)?;
        let mut buf = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut buf).map_err(err)?;
        version::get_flags(u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]))
    };
    let mut stream = exr_core::io::StdIfStream::open(path).map_err(err)?;
    let d = if version::is_multi_part(flags) {
        let mp = exr_core::MultiPartInputFile::new(&mut stream).map_err(err)?;
        crate::header::to_dict(py, mp.header(0).map_err(err)?)?
    } else if version::is_tiled(flags) {
        let inf = exr_core::tiled::TiledInputFile::new(&mut stream).map_err(err)?;
        crate::header::to_dict(py, inf.header())?
    } else {
        let inf = exr_core::scanline::InputFile::new(&mut stream).map_err(err)?;
        crate::header::to_dict(py, inf.header())?
    };
    // Normalise `type` to the Storage enum (single-part scanline files omit the
    // string attribute; multipart files carry it as a string — we override either).
    let storage = if version::is_non_image(flags) {
        if version::is_tiled(flags) { enums::Storage::deeptile } else { enums::Storage::deepscanline }
    } else if version::is_tiled(flags) {
        enums::Storage::tiledimage
    } else {
        enums::Storage::scanlineimage
    };
    d.bind(py).set_item("type", storage)?;
    Ok(d)
}

/// A `Channel` carrying name + pixels + pixel type (used by the deep reader).
pub(crate) fn channel_with_pixels(
    py: Python<'_>,
    name: &str,
    pixels: Py<PyAny>,
    ptype: PixelType,
) -> PyResult<Py<Channel>> {
    make_channel(py, name, pixels, ptype, 0)
}

/// A `Channel` descriptor (name + pixel type, no pixels) — for a header's `chlist`.
pub(crate) fn channel_descriptor(
    py: Python<'_>,
    name: &str,
    ptype: enums::PixelType,
) -> PyResult<Py<Channel>> {
    let core_pt = match ptype {
        enums::PixelType::UINT => PixelType::Uint,
        enums::PixelType::FLOAT => PixelType::Float,
        _ => PixelType::Half,
    };
    make_channel(py, name, py.None(), core_pt, 0)
}

/// Group/return channels: `R/G/B[/A]` interleaved into `RGB`/`RGBA` unless
/// `separate` is set; every other channel stays its own `(H, W)` array.
/// The prefix `channelNameToRGBA` groups a channel under, e.g. `"left.R"` -> `Some("left")`,
/// `"R"` -> `Some("")`, `"Zdepth"` -> `None` (doesn't end in R/G/B/A, or the character
/// before the final one isn't `.`/start-of-string — e.g. "AR" is not grouped: popping
/// the final 'R' leaves "A", which is neither empty nor "."-terminated).
///
/// Mirrors the reference 1:1 (`PyOpenEXR.cpp` `PyPart::channelNameToRGBA`,
/// r3_surfaces.md item 1 — the old version only grouped bare top-level `R/G/B/A`,
/// missing prefixed groups like `left.R/G/B`).
/// The channel name for component `comp` under `prefix` (the inverse of
/// [`rgba_prefix`]): `("", 'R')` -> `"R"`, `("left", 'R')` -> `"left.R"` — `prefix`
/// itself never carries the trailing dot, so it must be reinserted for a non-empty
/// prefix.
pub(crate) fn rgba_full_name(prefix: &str, comp: char) -> String {
    if prefix.is_empty() {
        comp.to_string()
    } else {
        format!("{prefix}.{comp}")
    }
}

pub(crate) fn rgba_prefix(name: &str) -> Option<&str> {
    let mut chars = name.chars();
    let last = chars.next_back()?;
    if !matches!(last, 'R' | 'G' | 'B' | 'A') {
        return None;
    }
    let stripped = chars.as_str(); // name with the final R/G/B/A removed
    if stripped.is_empty() {
        Some("")
    } else if let Some(prefix) = stripped.strip_suffix('.') {
        Some(prefix)
    } else {
        None
    }
}

/// Group/return channels: every maximal `R/G/B[/A]` group (top-level `RGB`/`RGBA`,
/// or a prefixed group like `left.R/G/B` -> group name `"left"`) interleaved into
/// one array, unless `separate` is set; every other channel stays its own `(H, W)`
/// array.
fn build_channels(py: Python<'_>, image: &Image, separate: bool) -> PyResult<Py<PyDict>> {
    let (h, w) = (image.height() as usize, image.width() as usize);
    let out = PyDict::new(py);

    if separate {
        return separate_channels_dict(py, image, h, w);
    }

    // Every prefix (including "" for top-level) that has an R, a G, and a B.
    let has = |prefix: &str, comp: char| image.channel(&rgba_full_name(prefix, comp)).is_some();
    let mut prefixes: Vec<&str> = image.channels().filter_map(|(name, _)| rgba_prefix(name)).collect();
    prefixes.sort_unstable();
    prefixes.dedup();
    let mut grouped_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for prefix in prefixes {
        if !(has(prefix, 'R') && has(prefix, 'G') && has(prefix, 'B')) {
            continue;
        }
        let alpha = has(prefix, 'A');
        let comps: &[char] = if alpha { &['R', 'G', 'B', 'A'] } else { &['R', 'G', 'B'] };
        let names: Vec<String> = comps.iter().map(|&c| rgba_full_name(prefix, c)).collect();
        let Some(planes) = names
            .iter()
            .map(|n| image.channel(n))
            .collect::<Option<Vec<&ChannelData>>>()
        else {
            continue; // shouldn't happen (just checked), but never panic across the FFI boundary
        };
        // Group name: the prefix itself if non-empty (`left.R/G/B` -> `"left"`),
        // else the reference's `"RGB"`/`"RGBA"` for the top-level group.
        let group = if prefix.is_empty() {
            if alpha { "RGBA".to_string() } else { "RGB".to_string() }
        } else {
            prefix.to_string()
        };
        // A literal channel already using the synthesized group name (a channel
        // dict with both e.g. "left" AND "left.R"/"left.G"/"left.B") collides with
        // the grouping key — raise instead of silently shadowing it
        // (test_literal_rgb_key_collision_flat_rejected).
        if image.channel(&group).is_some() {
            return Err(PyValueError::new_err(format!(
                "channel \"{group}\" collides with the grouped R/G/B[/A] channels; pass \
                 separate_channels=True to read them individually"
            )));
        }
        let pixels = interleave_to_py(py, &planes, h, w)?;
        out.set_item(group.as_str(), make_channel(py, group.as_str(), pixels, planes[0].pixel_type(), 0)?)?;
        grouped_names.extend(names);
    }
    // Every channel not absorbed into a group stays separate.
    for (i, (name, data)) in image.channels().enumerate() {
        if grouped_names.contains(name) {
            continue;
        }
        let pixels = plane_to_py(py, data, h, w)?;
        out.set_item(name, make_channel(py, name, pixels, data.pixel_type(), i as i32)?)?;
    }
    Ok(out.unbind())
}

/// Every channel as its own `(H, W)` array (the `separate_channels=True` path and
/// the fallback when no `R/G/B` group is present).
fn separate_channels_dict(py: Python<'_>, image: &Image, h: usize, w: usize) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    for (i, (name, data)) in image.channels().enumerate() {
        let pixels = plane_to_py(py, data, h, w)?;
        out.set_item(name, make_channel(py, name, pixels, data.pixel_type(), i as i32)?)?;
    }
    Ok(out.unbind())
}

/// Construct a `Channel` object carrying a name, its numpy pixels, and pixel type.
fn make_channel(
    py: Python<'_>,
    name: &str,
    pixels: Py<PyAny>,
    ptype: PixelType,
    index: i32,
) -> PyResult<Py<Channel>> {
    Py::new(
        py,
        Channel {
            name: name.to_string(),
            pixels,
            xSampling: 1,
            ySampling: 1,
            pLinear: false,
            channel_index: index,
            ptype,
        },
    )
}

/// Assemble an `exr_core::Image` from a channels dict (numpy arrays or `Channel`s)
/// for writing. `RGB`/`RGBA` 3-D arrays are split into R/G/B[/A]; 2-D arrays become
/// the named channel.
fn assemble_image(py: Python<'_>, header: &Bound<'_, PyDict>, channels: &Bound<'_, PyDict>) -> PyResult<Image> {
    // Collect (channel_name, ChannelData) + the common dimensions.
    let mut planes: Vec<(String, ChannelData)> = Vec::new();
    let mut dims: Option<(usize, usize)> = None;

    for (key, value) in channels.iter() {
        let name: String = key.extract()?;
        // Unwrap a Channel object to its `.pixels`, else use the array directly.
        let array: Bound<'_, PyAny> = match value.cast::<Channel>() {
            Ok(ch) => ch.borrow().pixels.clone_ref(py).into_bound(py),
            Err(_) => value.clone(),
        };
        let ndim = array.cast::<numpy::PyUntypedArray>().map_err(|_| {
            PyTypeError::new_err(format!("channel \"{name}\" must be a numpy array or Channel"))
        })?;
        let nd = ndim.ndim();
        if nd == 3 {
            let (pls, h, w) = py_to_planes(&array)?;
            let sub = split_names(&name, pls.len());
            set_dims(&mut dims, h, w, &name)?;
            for (suffix, data) in sub.into_iter().zip(pls) {
                planes.push((suffix, data));
            }
        } else {
            let (data, h, w) = py_to_plane(&array)?;
            set_dims(&mut dims, h, w, &name)?;
            planes.push((name, data));
        }
    }

    let (h, w) = dims.ok_or_else(|| PyValueError::new_err("File.write: no channels to write"))?;
    // Honour an explicit `header["dataWindow"]` (validated against the pixel dims),
    // else default to an origin window covering the arrays.
    let data_window = match header.get_item("dataWindow")? {
        Some(dw) => {
            let b = py_to_box(&dw)?;
            let bw = (b.max.x - b.min.x + 1) as usize;
            let bh = (b.max.y - b.min.y + 1) as usize;
            if bw != w || bh != h {
                return Err(PyValueError::new_err(format!(
                    "header dataWindow is {bw}x{bh} but the channels are {w}x{h}"
                )));
            }
            b
        }
        None => Box2i {
            min: V2i::new(0, 0),
            max: V2i::new(w as i32 - 1, h as i32 - 1),
        },
    };
    let mut image = Image::new(data_window);
    // Honour an explicit displayWindow if the header carries one.
    if let Some(dw) = header.get_item("displayWindow")?
        && let Ok(b) = py_to_box(&dw)
    {
        image.display_window = b;
    }
    // Honour an explicit `header["lineOrder"]` (r3_surfaces.md item 9: it used to
    // be dropped as a "standard key" without ever being copied onto the `Image`,
    // so every write silently reset it to INCREASING_Y).
    image.line_order = header_line_order(header)?;
    for (name, data) in planes {
        image.insert_channel(name, data).map_err(err)?;
    }
    Ok(image)
}

/// Sub-channel names for an interleaved `(H, W, C)` array, matching the reference
/// (PyOpenEXR.cpp): the prefix is empty for `RGB`/`RGBA`, otherwise `"name."`; the
/// components are R, G, B, A in order, then numbered for any channels beyond 4. So
/// `RGB`→R,G,B; `RGBA`→R,G,B,A; `X` (3)→X.R,X.G,X.B; `X` (4)→X.R,X.G,X.B,X.A.
fn split_names(name: &str, count: usize) -> Vec<String> {
    let prefix = match name {
        "RGB" | "RGBA" => String::new(),
        other => format!("{other}."),
    };
    const COMPS: [&str; 4] = ["R", "G", "B", "A"];
    (0..count)
        .map(|i| match COMPS.get(i) {
            Some(c) => format!("{prefix}{c}"),
            None => format!("{prefix}{i}"),
        })
        .collect()
}

fn set_dims(dims: &mut Option<(usize, usize)>, h: usize, w: usize, name: &str) -> PyResult<()> {
    match dims {
        None => {
            *dims = Some((h, w));
            Ok(())
        }
        Some((dh, dw)) if *dh == h && *dw == w => Ok(()),
        Some((dh, dw)) => Err(PyValueError::new_err(format!(
            "channel \"{name}\" is {h}x{w} but a previous channel is {dh}x{dw}"
        ))),
    }
}

/// Coerce a Python value to an `i32` code — a plain int OR an `OpenEXR.Compression`
/// (etc.) enum member (via `int()`, which the `eq_int` enums support).
fn pyany_to_i32(v: &Bound<'_, PyAny>) -> Option<i32> {
    if let Ok(i) = v.extract::<i32>() {
        return Some(i);
    }
    v.py()
        .import("builtins")
        .ok()?
        .getattr("int")
        .ok()?
        .call1((v,))
        .ok()?
        .extract::<i32>()
        .ok()
}

/// The write line order from `header["lineOrder"]` (default INCREASING_Y),
/// accepting an `OpenEXR.LineOrder` member or a plain integer code. An unknown code
/// raises rather than silently defaulting (same reasoning as `header_compression`).
fn header_line_order(header: &Bound<'_, PyDict>) -> PyResult<LineOrder> {
    let Some(v) = header.get_item("lineOrder")? else {
        return Ok(LineOrder::IncreasingY);
    };
    let code = pyany_to_i32(&v)
        .ok_or_else(|| PyValueError::new_err("header[\"lineOrder\"] must be an int or a LineOrder member"))?;
    match code {
        0 => Ok(LineOrder::IncreasingY),
        1 => Ok(LineOrder::DecreasingY),
        2 => Ok(LineOrder::RandomY),
        _ => Err(PyValueError::new_err(format!("unknown lineOrder code {code}"))),
    }
}

/// The write compression from `header["compression"]` (default ZIP), accepting an
/// `OpenEXR.Compression` member or a plain integer code.
///
/// r3_surfaces.md item 8: an out-of-range/garbage code used to silently become ZIP.
/// That is a fallback where a hard error is the logically correct behaviour — a
/// caller who wrote a bogus compression code should see that, not have it silently
/// reinterpreted as ZIP. Only an ABSENT key defaults to ZIP, matching the
/// reference's default. HTJ2K256/HTJ2K32 (10/11) are real, valid codes (target is
/// v3.4.15) — only genuinely unknown codes (12+) raise.
fn header_compression(header: &Bound<'_, PyDict>) -> PyResult<Compression> {
    let Some(v) = header.get_item("compression")? else {
        return Ok(Compression::Zip);
    };
    let code = pyany_to_i32(&v)
        .ok_or_else(|| PyValueError::new_err("header[\"compression\"] must be an int or a Compression member"))?;
    code_to_compression(code)
}

/// The single source of truth for a raw `int` compression code -> `exr_core`
/// `Compression` (shared by `header_compression` here and `deep_compression` in
/// `deep.rs`, design_r5.md W4.6 dedup). Only a genuinely unknown code (12+, or a
/// negative value) raises.
pub(crate) fn code_to_compression(code: i32) -> PyResult<Compression> {
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
        // HTJ2K (v3.4 addition; target is v3.4.15, corrected 2026-09-22).
        10 => Ok(Compression::Htj2k256),
        11 => Ok(Compression::Htj2k32),
        _ => Err(PyValueError::new_err(format!("unknown compression code {code}"))),
    }
}

/// `exr_core` compression → the Python `Compression` enum member. Raises only for
/// `Compression::Invalid` (r3_surfaces.md item 8's original complaint was HTJ2K
/// being misreported as ZIP — since the real target is v3.4.15, HTJ2K256/HTJ2K32
/// now have real enum members, so only `Invalid` — an on-disk garbage byte — has no
/// faithful member to report).
pub(crate) fn comp_to_py(c: Compression) -> PyResult<enums::Compression> {
    use enums::Compression as P;
    match c as i32 {
        0 => Ok(P::NO_COMPRESSION),
        1 => Ok(P::RLE_COMPRESSION),
        2 => Ok(P::ZIPS_COMPRESSION),
        3 => Ok(P::ZIP_COMPRESSION),
        4 => Ok(P::PIZ_COMPRESSION),
        5 => Ok(P::PXR24_COMPRESSION),
        6 => Ok(P::B44_COMPRESSION),
        7 => Ok(P::B44A_COMPRESSION),
        8 => Ok(P::DWAA_COMPRESSION),
        9 => Ok(P::DWAB_COMPRESSION),
        10 => Ok(P::HTJ2K256_COMPRESSION),
        11 => Ok(P::HTJ2K32_COMPRESSION),
        code => Err(PyValueError::new_err(format!(
            "compression code {code} has no OpenEXR Compression member (on-disk garbage)"
        ))),
    }
}

/// `(x, y)` from any 2-element indexable object — a Python tuple, or a numpy
/// array/scalar pair (`test_box2i_numpy_scalars`: `dataWindow` given as
/// `(np.array((0,0)), np.array((15,15)))` or numpy `int32`/`int64` scalars).
/// `PyAnyMethods::get_item` works uniformly across tuples and numpy arrays, unlike
/// a typed `(i32, i32)` extraction, which requires an actual `PyTuple`.
pub(crate) fn xy_i32(v: &Bound<'_, PyAny>) -> PyResult<(i32, i32)> {
    Ok((v.get_item(0)?.extract()?, v.get_item(1)?.extract()?))
}

/// `((min.x,min.y),(max.x,max.y))` → `Box2i` (min/max each any 2-element
/// indexable object — see [`xy_i32`]).
pub(crate) fn py_to_box(obj: &Bound<'_, PyAny>) -> PyResult<Box2i> {
    let mn = obj.get_item(0)?;
    let mx = obj.get_item(1)?;
    let (minx, miny) = xy_i32(&mn)?;
    let (maxx, maxy) = xy_i32(&mx)?;
    Ok(Box2i {
        min: V2i::new(minx, miny),
        max: V2i::new(maxx, maxy),
    })
}

/// `(x, y)` from any 2-element indexable object, float version of [`xy_i32`].
pub(crate) fn xy_f32(v: &Bound<'_, PyAny>) -> PyResult<(f32, f32)> {
    Ok((v.get_item(0)?.extract()?, v.get_item(1)?.extract()?))
}

/// `((min.x,min.y),(max.x,max.y))` → `Box2f`, float version of [`py_to_box`].
pub(crate) fn py_to_boxf(obj: &Bound<'_, PyAny>) -> PyResult<Box2f> {
    let mn = obj.get_item(0)?;
    let mx = obj.get_item(1)?;
    let (minx, miny) = xy_f32(&mn)?;
    let (maxx, maxy) = xy_f32(&mx)?;
    Ok(Box2f {
        min: V2f::new(minx, miny),
        max: V2f::new(maxx, maxy),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Channel>()?;
    m.add_class::<Part>()?;
    m.add_class::<File>()?;
    Ok(())
}
