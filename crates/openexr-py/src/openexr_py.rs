//! The `OpenEXR` Python extension module (PyO3) — a drop-in replacement for the
//! official OpenEXR Python module, backed by our pure-Rust core (`exr-core` +
//! `imath-rs`) instead of the C++ library.
//!
//! ## Status
//! - **P1 (done):** the enum TYPES (`Compression`, `PixelType`, `Storage`,
//!   `LineOrder`, `LevelMode`, `LevelRoundingMode`, `Envmap`) + their members
//!   exported as module-level names (`OpenEXR.ZIP_COMPRESSION` IS
//!   `Compression.ZIP_COMPRESSION`, and `== 3`) — the reference's `export_values()`.
//! - **P2/P3 (done):** the numpy `File` + `Channel` classes — read (`channels()`
//!   numpy arrays + `header()`, `separate_channels`, RGB/RGBA grouping) and write
//!   (round-trip) of single-part scanline images. See the `file` module
//!   for the covered surface and the deferred attributes.
//!
//! ## Parity contract
//! Every public name / method / parameter / constant value must match the reference
//! at `openexr/src/wrappers/python/` (PyOpenEXR.cpp = new numpy API,
//! PyOpenEXR_old.cpp = legacy API). The reference `tests/` are the acceptance gate.

// The crate / module / fn are deliberately named `OpenEXR` (not snake_case), and the
// enum variants keep their OpenEXR names (`ZIP_COMPRESSION`, `scanlineimage`, …) —
// both ARE the Python API surface (drop-in parity), so Rust's casing lints are
// suppressed crate-wide here rather than renamed.
#![allow(non_snake_case, non_camel_case_types)]

use pyo3::prelude::*;

mod cryptomatte;
mod deep;
mod enums;
mod file;
mod header;
mod legacy;
mod types;

/// The OpenEXR API version this module replicates (drop-in tools read
/// `OpenEXR.__version__`). Bump in lockstep with the parity target.
const OPENEXR_API_VERSION: &str = "3.4.0";

/// The `OpenEXR` module entry point. The `#[pymodule]` fn name IS the import name,
/// so it must stay `OpenEXR`.
#[pymodule]
pub fn OpenEXR(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", OPENEXR_API_VERSION)?;
    m.add("OPENEXR_VERSION", OPENEXR_API_VERSION)?;
    enums::register(m)?;
    types::register(m)?;
    file::register(m)?;
    legacy::register(m)?;
    // exr-rs extension (not ASWF OpenEXR parity)
    cryptomatte::register(m)?;
    Ok(())
}
