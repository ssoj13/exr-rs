//! The OpenEXR enum TYPES + their module-level member exports.
//!
//! Each enum mirrors an OpenEXR/Imf enum 1:1 (names + integer codes). `register`
//! adds the class AND exports every member as a module-level name — the reference's
//! `py::enum_<…>().export_values()`: `OpenEXR.ZIP_COMPRESSION` IS
//! `Compression.ZIP_COMPRESSION`, and via `eq_int` it compares equal to its integer
//! code (`== 3`), so header dictionaries `{"compression": OpenEXR.ZIP_COMPRESSION}`
//! round-trip exactly as the C++ binding's do.

use pyo3::prelude::*;

/// Define a pyo3 simple enum + a `register` fn that adds the class and exports each
/// member into the module namespace (export_values semantics).
macro_rules! exr_enum {
    ($name:ident, $doc:literal, $reg:ident, { $($variant:ident = $val:expr),+ $(,)? }) => {
        #[doc = $doc]
        #[pyclass(eq, eq_int, module = "OpenEXR", from_py_object)]
        #[derive(Clone, Copy, PartialEq, Debug)]
        // Variant names are the exact Python-facing constant names (`OpenEXR.UINT`,
        // `.HALF`, `.FLOAT`, …), matching real C++ PyOpenEXR 1:1 — not renameable
        // without breaking the drop-in API, regardless of Rust naming convention.
        #[allow(clippy::upper_case_acronyms)]
        pub enum $name { $($variant = $val),+ }

        fn $reg(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add_class::<$name>()?;
            // Export each member by aliasing the CLASS attribute object (not a fresh
            // conversion), so `OpenEXR.ZIP_COMPRESSION is Compression.ZIP_COMPRESSION`
            // — identity-equal, exactly like the reference's `export_values()`.
            let cls = m.getattr(stringify!($name))?;
            $( m.add(stringify!($variant), cls.getattr(stringify!($variant))?)?; )+
            Ok(())
        }
    };
}

exr_enum!(Compression, "Compression method", register_compression, {
    NO_COMPRESSION = 0, RLE_COMPRESSION = 1, ZIPS_COMPRESSION = 2, ZIP_COMPRESSION = 3,
    PIZ_COMPRESSION = 4, PXR24_COMPRESSION = 5, B44_COMPRESSION = 6, B44A_COMPRESSION = 7,
    DWAA_COMPRESSION = 8, DWAB_COMPRESSION = 9,
    // v3.4 additions (target is v3.4.15, corrected 2026-09-22 -- team-lead) --
    // matches `openexr/.../PyOpenEXR.cpp:2671-2673` exactly, including the
    // NUM_COMPRESSION_METHODS shift from 10 to 12 that makes room for them.
    HTJ2K256_COMPRESSION = 10, HTJ2K32_COMPRESSION = 11, NUM_COMPRESSION_METHODS = 12,
});
exr_enum!(PixelType, "Data type for pixel arrays", register_pixeltype, {
    UINT = 0, HALF = 1, FLOAT = 2, NUM_PIXELTYPES = 3,
});
exr_enum!(Storage, "Image storage format", register_storage, {
    scanlineimage = 0, tiledimage = 1, deepscanline = 2, deeptile = 3, NUM_STORAGE_TYPES = 4,
});
exr_enum!(LineOrder, "Line order for scanline images", register_lineorder, {
    INCREASING_Y = 0, DECREASING_Y = 1, RANDOM_Y = 2, NUM_LINE_ORDERS = 3,
});
exr_enum!(LevelMode, "Level mode for tiled images", register_levelmode, {
    ONE_LEVEL = 0, MIPMAP_LEVELS = 1, RIPMAP_LEVELS = 2, NUM_LEVEL_MODES = 3,
});
exr_enum!(LevelRoundingMode, "Rounding mode for tiled images", register_levelrounding, {
    ROUND_DOWN = 0, ROUND_UP = 1, NUM_ROUNDING_MODES = 2,
});
exr_enum!(Envmap, "Environment map type", register_envmap, {
    ENVMAP_LATLONG = 0, ENVMAP_CUBE = 1, NUM_ENVMAPTYPES = 2,
});

/// Register every enum type + export its members into the module namespace.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_compression(m)?;
    register_pixeltype(m)?;
    register_storage(m)?;
    register_lineorder(m)?;
    register_levelmode(m)?;
    register_levelrounding(m)?;
    register_envmap(m)?;
    Ok(())
}
