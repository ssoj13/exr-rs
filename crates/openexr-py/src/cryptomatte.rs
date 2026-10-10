//! `OpenEXR.cryptomatte` — exr-rs extension submodule (not ASWF OpenEXR).
//!
//! Thin PyO3 facade over the Rust [`cryptomatte`] crate. Canonical import:
//! `import OpenEXR.cryptomatte` or `from OpenEXR import cryptomatte`.

use cryptomatte::{
    self as crypto, CryptomatteFile, EncodeOptions, ManifestPlacement, MergeOp,
    NamedMatte,
};
use exr_core::Compression;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

/// `OpenEXR.cryptomatte` is an exr-rs extension (not part of the ASWF reference),
/// but it wraps the same underlying `exr_core`/`cryptomatte` crate errors this
/// module's other surfaces do — keep the same Iex-style convention (`file.rs::err`)
/// for consistency: an underlying library error is a `RuntimeError`, not a
/// `ValueError`; a `PyValueError::new_err(...)` written directly at a call site
/// (this module's own argument validation) is unaffected.
fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// MurmurHash3_x86_32 → Cryptomatte float ID.
#[pyfunction]
fn name_to_float_id(name: &str) -> f32 {
    crypto::name_to_float_id(name)
}

/// Float ID → 8 lowercase hex digits (manifest form).
#[pyfunction]
fn float_id_to_hex(id: f32) -> String {
    crypto::float_id_to_hex(id)
}

/// Parse 8-digit hex → float ID.
#[pyfunction]
fn hex_to_float_id(hex: &str) -> PyResult<f32> {
    crypto::hex_to_float_id(hex)
        .ok_or_else(|| PyValueError::new_err(format!("invalid Cryptomatte hex id: {hex}")))
}

fn layer_to_dict<'py>(py: Python<'py>, layer: &crypto::LayerInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("metadata_id", &layer.metadata_id)?;
    d.set_item("name", &layer.name)?;
    d.set_item("hash", &layer.hash)?;
    d.set_item("conversion", &layer.conversion)?;
    d.set_item("md_prefix", &layer.md_prefix)?;
    d.set_item("manif_file", layer.manif_file.as_deref())?;
    d.set_item("has_inline_manifest", layer.inline_manifest.is_some())?;
    let ranked: Vec<String> = layer.ranked.iter().map(|r| r.base.clone()).collect();
    d.set_item("ranked_layers", ranked)?;
    d.set_item("part", layer.part)?;
    Ok(d)
}

/// List Cryptomatte layers in an EXR (header discover, no pixel decode).
#[pyfunction]
fn layers(py: Python<'_>, path: &str) -> PyResult<Py<PyList>> {
    let found = crypto::discover_path(path).map_err(err)?;
    let out = PyList::empty(py);
    for layer in &found {
        out.append(layer_to_dict(py, layer)?)?;
    }
    Ok(out.unbind())
}

/// Extract a coverage matte as a float32 numpy array shaped `(H, W)`.
#[pyfunction]
#[pyo3(signature = (path, names=None, ids=None, layer=None))]
fn extract<'py>(
    py: Python<'py>,
    path: &str,
    names: Option<Vec<String>>,
    ids: Option<Vec<f32>>,
    layer: Option<String>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let file = CryptomatteFile::open_for_layer(path, layer.as_deref()).map_err(err)?;
    let layer_info = file.select_layer(layer.as_deref()).map_err(err)?;

    let mut selected = Vec::new();
    if let Some(ns) = names {
        // W4.6 (design_r5.md §4.4): route through the canonical
        // `Manifest::id_for_name` (manifest hit, else hash fallback — the
        // hash IS the id definition, spec 1.2.0 p.3) instead of a hand-rolled
        // `by_name` lookup. A broken/missing manifest must not block
        // extraction (names still resolve by hash); surface it as a Python
        // warning instead of silently swallowing it or raising.
        let manif = match file.load_layer_manifest(layer_info) {
            Ok(m) => m,
            Err(e) => {
                py.import("warnings")?.call_method1(
                    "warn",
                    (format!(
                        "cryptomatte: layer `{}` manifest unavailable ({e}); \
                         resolving names by hash",
                        layer_info.name
                    ),),
                )?;
                cryptomatte::Manifest::default()
            }
        };
        for n in &ns {
            selected.push(manif.id_for_name(n));
        }
    }
    if let Some(raw_ids) = ids {
        selected.extend(raw_ids);
    }
    if selected.is_empty() {
        return Err(PyValueError::new_err(
            "extract requires names= and/or ids=",
        ));
    }
    selected.sort_by_key(|f| f.to_bits());
    selected.dedup_by_key(|f| f.to_bits());

    let matte = file.extract_ids(layer_info, &selected).map_err(err)?;
    let arr = Array2::from_shape_vec((matte.height, matte.width), matte.coverage)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

fn dict_to_named_mattes(mattes: &Bound<'_, PyDict>) -> PyResult<(usize, usize, Vec<NamedMatte>)> {
    if mattes.is_empty() {
        return Err(PyValueError::new_err("requires at least one named matte"));
    }
    let mut named: Vec<NamedMatte> = Vec::with_capacity(mattes.len());
    let mut width = 0usize;
    let mut height = 0usize;
    for (key, value) in mattes.iter() {
        let name: String = key.extract()?;
        let arr = value.cast::<PyArray2<f32>>().map_err(|_| {
            PyValueError::new_err(format!(
                "matte `{name}` must be a 2-D numpy float32 array shaped (H, W)"
            ))
        })?;
        let shape = arr.shape();
        let h = shape[0];
        let w = shape[1];
        if named.is_empty() {
            height = h;
            width = w;
        } else if h != height || w != width {
            return Err(PyValueError::new_err(format!(
                "matte `{name}` shape ({h},{w}) does not match ({height},{width})"
            )));
        }
        let coverage: Vec<f32> = arr.readonly().as_array().iter().copied().collect();
        named.push(NamedMatte { name, coverage });
    }
    Ok((width, height, named))
}

/// Encode named coverage mattes to a Cryptomatte EXR.
///
/// `mattes`: name -> HxW float32. `ranked_layers` defaults to
/// `DEFAULT_RANKED_LAYERS` (3, the spec's suggested 6 ranks). Optional
/// `sidecar` is a relative `.json` path (forward slashes) written as
/// `manif_file` next to the EXR.
#[pyfunction]
#[pyo3(signature = (path, crypto_name, mattes, ranked_layers=None, sidecar=None))]
fn encode(
    path: &str,
    crypto_name: &str,
    mattes: Bound<'_, PyDict>,
    ranked_layers: Option<usize>,
    sidecar: Option<String>,
) -> PyResult<()> {
    let (width, height, named) = dict_to_named_mattes(&mattes)?;
    let mut opts = EncodeOptions::new(crypto_name);
    opts.ranked_layers = ranked_layers.unwrap_or(crypto::DEFAULT_RANKED_LAYERS);
    if let Some(rel) = sidecar {
        opts.manifest = ManifestPlacement::Sidecar {
            relative_path: rel,
        };
    }
    // A numpy array carries no placement: origin data window = display window.
    let windows = crypto::Windows::origin(width, height).map_err(err)?;
    let enc = crypto::from_named_mattes(windows, &named, &opts).map_err(err)?;
    // A freshly encoded Cryptomatte has no source file to inherit from, so the
    // codec is an explicit choice: ZIP, the Cryptomatte-spec-friendly default.
    enc.write(path, Some(Compression::Zip)).map_err(err)?;
    Ok(())
}

/// Inject a named matte into an existing Cryptomatte EXR (over/under), write result.
///
/// `coverage` is HxW float32. `operation` is `"over"` or `"under"`.
// The 8 parameters are exactly the Python-facing keyword arguments of
// `cryptomatte.inject(...)` (`#[pyo3(signature = ...)]` below) -- bundling
// them into a struct would not reduce the real arity, only hide it.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (src_path, dst_path, name, coverage, operation="over", layer=None, ranked_layers=None, sidecar=None))]
fn inject(
    src_path: &str,
    dst_path: &str,
    name: &str,
    coverage: Bound<'_, PyArray2<f32>>,
    operation: &str,
    layer: Option<String>,
    ranked_layers: Option<usize>,
    sidecar: Option<String>,
) -> PyResult<()> {
    let op = match operation.to_ascii_lowercase().as_str() {
        "over" => MergeOp::Over,
        "under" => MergeOp::Under,
        other => {
            return Err(PyValueError::new_err(format!(
                "operation must be 'over' or 'under', got `{other}`"
            )));
        }
    };
    let shape = coverage.shape();
    let (h, w) = (shape[0], shape[1]);
    let plane: Vec<f32> = coverage.readonly().as_array().iter().copied().collect();

    let file = CryptomatteFile::open_for_layer(src_path, layer.as_deref()).map_err(err)?;
    let layer_info = file.select_layer(layer.as_deref()).map_err(err)?;
    // The (H, W) shape itself must match, not just the element count: a
    // transposed array has the right count and scrambles every row.
    let (img_h, img_w) = (file.image.height(), file.image.width());
    if i64::try_from(h).ok() != Some(img_h) || i64::try_from(w).ok() != Some(img_w) {
        return Err(PyValueError::new_err(format!(
            "coverage shape ({h}, {w}) does not match the image ({img_h}, {img_w})"
        )));
    }
    let mut opts = EncodeOptions::new(&layer_info.name);
    opts.ranked_layers = ranked_layers.unwrap_or(if layer_info.ranked.is_empty() {
        crypto::DEFAULT_RANKED_LAYERS
    } else {
        layer_info.ranked.len()
    });
    if let Some(rel) = sidecar {
        opts.manifest = ManifestPlacement::Sidecar {
            relative_path: rel,
        };
    }

    // W4.6 (design_r5.md §4.4): route through the canonical
    // `CryptomatteFile::inject` instead of hand-assembling `InjectArgs` — it
    // resolves the manifest via `layer_manifest_or_empty` (a broken/missing
    // sidecar warns and falls back, matching `extract`'s policy, instead of
    // raising), keeps the layer's own metadata-key prefix
    // (`md_prefix`, so a legacy `exr/cryptomatte/`-keyed file stays legacy),
    // and carries over every other header attribute via `carry_header_attrs`.
    let inj = file.inject(layer_info, name, &plane, op, &opts).map_err(err)?;
    // `None` keeps the SOURCE file's compression: inject rewrites one layer of an
    // existing file and must not silently recompress the rest of it.
    inj.write(dst_path, None).map_err(err)?;
    Ok(())
}

/// Register the `cryptomatte` submodule on `OpenEXR`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "cryptomatte")?;
    m.setattr(
        "__doc__",
        "Cryptomatte decode/encode helpers (exr-rs extension; not part of ASWF OpenEXR).",
    )?;
    m.add_function(wrap_pyfunction!(name_to_float_id, &m)?)?;
    m.add_function(wrap_pyfunction!(float_id_to_hex, &m)?)?;
    m.add_function(wrap_pyfunction!(hex_to_float_id, &m)?)?;
    m.add_function(wrap_pyfunction!(layers, &m)?)?;
    m.add_function(wrap_pyfunction!(extract, &m)?)?;
    m.add_function(wrap_pyfunction!(encode, &m)?)?;
    m.add_function(wrap_pyfunction!(inject, &m)?)?;

    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("OpenEXR.cryptomatte", &m)?;
    Ok(())
}
