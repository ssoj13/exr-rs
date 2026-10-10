//! Deep image read/write (`deepscanline` + `deeptile`) via exr-core's verified
//! deep API, marshalled to/from numpy `object` arrays (each pixel a 1-D numpy
//! sample array, or `None`).
//!
//! Model: a flat per-channel sample heap + a per-pixel UINT count buffer (the
//! exr-core deep frame-buffer contract). The count buffer is a LOCAL `w*h` raster
//! and the count slice's `base_offset` is shifted by
//! `-(min_y*yStride + min_x*xStride)`, so any data window — including a negative-
//! or large-positive origin — round-trips exactly without an origin-sized
//! allocation. Every channel at a pixel must share the same sample count
//! (validated before write). Default read groups `R/G/B[/A]` into an
//! `(nsamples, 3|4)` array per pixel; `separate_channels` keeps one `object`
//! array per channel.

use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, PixelType, TileDescription,
    TileDescriptionAttribute,
};
use exr_core::deep::{
    DeepFrameBuffer, DeepScanLineInputFile, DeepScanLineOutputFile, DeepSlice, DeepTiledInputFile,
    DeepTiledOutputFile,
};
use exr_core::frame_buffer::Slice;
use exr_core::header::{DEEP_SCANLINE, DEEP_TILE, Header};
use exr_core::io::StdIfStream;

use crate::enums;
use imath_rs::{Half, V2f};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

/// Same reasoning as `file.rs::err` — an `exr_core::ExrError` is an `Iex`
/// exception with no pybind11 translator registered in the reference, so it
/// falls through to `RuntimeError`, not `ValueError`.
fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// `w*h` as a validated `usize`. `w`/`h` come from either
/// an untrusted numpy `object` array shape (write) or an untrusted file's
/// `dataWindow` (read) — plain `(w * h) as usize` computes the product in native
/// `i32` first, which overflows (panic in debug, silently wraps in release) before
/// the cast ever runs. Widen to `i64`, reject non-positive dimensions, and cap the
/// pixel count at 2^32 (already far beyond any real deep image) instead of
/// allocating from a wrapped/garbage size.
fn checked_npix(w: i32, h: i32) -> PyResult<usize> {
    if w <= 0 || h <= 0 {
        return Err(PyValueError::new_err(format!("deep image dimensions must be positive, got {w}x{h}")));
    }
    let n = (w as i64) * (h as i64);
    if n > (1i64 << 32) {
        return Err(PyValueError::new_err(format!("deep image {w}x{h} ({n} pixels) is too large")));
    }
    Ok(n as usize)
}

/// `w*h*4` bytes (the per-pixel `u32` sample-count buffer), same overflow guard as
/// [`checked_npix`].
fn checked_count_bytes(w: i32, h: i32) -> PyResult<usize> {
    checked_npix(w, h)?
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("deep count buffer size overflow"))
}

/// Routes through the canonical exr-core helper (`misc::pixel_type_size`)
/// instead of a hand-rolled copy; a deep channel can only genuinely be
/// HALF/FLOAT/UINT, so the invalid `NumPixelTypes` sentinel (a corrupted
/// on-disk channel type) raises instead of pretending to have a byte size.
fn ptype_size(t: PixelType) -> PyResult<usize> {
    exr_core::misc::pixel_type_size(t).map_err(|_| err(format!("invalid deep channel pixel type: {t:?}")))
}

/// numpy dtype string of a 1-D sample array -> EXR pixel type.
fn array_ptype(a: &Bound<'_, PyAny>) -> PyResult<PixelType> {
    if a.cast::<PyArray1<f32>>().is_ok() {
        Ok(PixelType::Float)
    } else if a.cast::<PyArray1<half::f16>>().is_ok() {
        Ok(PixelType::Half)
    } else if a.cast::<PyArray1<u32>>().is_ok() {
        Ok(PixelType::Uint)
    } else {
        Err(PyTypeError::new_err(
            "deep sample arrays must be 1-D numpy float16/float32/uint32",
        ))
    }
}

/// Append a 1-D sample array's values to `heap` in `ptype`'s little-endian form.
fn append_samples(heap: &mut Vec<u8>, a: &Bound<'_, PyAny>, ptype: PixelType) -> PyResult<usize> {
    let n = match ptype {
        PixelType::Float => {
            let arr = a.cast::<PyArray1<f32>>().map_err(|e| err(e.to_string()))?;
            let v: Vec<f32> = arr.readonly().as_array().iter().copied().collect();
            for x in &v {
                heap.extend_from_slice(&x.to_le_bytes());
            }
            v.len()
        }
        PixelType::Half => {
            let arr = a.cast::<PyArray1<half::f16>>().map_err(|e| err(e.to_string()))?;
            let v: Vec<half::f16> = arr.readonly().as_array().iter().copied().collect();
            for x in &v {
                heap.extend_from_slice(&Half::from_bits(x.to_bits()).bits().to_le_bytes());
            }
            v.len()
        }
        PixelType::Uint => {
            let arr = a.cast::<PyArray1<u32>>().map_err(|e| err(e.to_string()))?;
            let v: Vec<u32> = arr.readonly().as_array().iter().copied().collect();
            for x in &v {
                heap.extend_from_slice(&x.to_le_bytes());
            }
            v.len()
        }
        PixelType::NumPixelTypes => 0,
    };
    Ok(n)
}

/// Read `count` samples of `ptype` from `heap[off..]` into a 1-D numpy array.
fn samples_to_py(
    py: Python<'_>,
    heap: &[u8],
    off: usize,
    count: usize,
    ptype: PixelType,
) -> PyResult<Py<PyAny>> {
    let sz = ptype_size(ptype)?;
    let bytes = heap
        .get(off..off + count * sz)
        .ok_or_else(|| err("deep heap out of range"))?;
    Ok(match ptype {
        PixelType::Float => {
            let v: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            v.into_pyarray(py).into_any().unbind()
        }
        PixelType::Half => {
            let v: Vec<half::f16> = bytes.as_chunks::<2>().0.iter().map(|&c| half::f16::from_bits(u16::from_le_bytes(c))).collect();
            v.into_pyarray(py).into_any().unbind()
        }
        _ => {
            let v: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect();
            v.into_pyarray(py).into_any().unbind()
        }
    })
}

// ── Write ───────────────────────────────────────────────────────────────────────

/// Write a deepscanline file from `channels` (each value an `(H, W)` numpy `object`
/// array of 1-D sample arrays or `None`). The data window is the origin.
pub fn write_deep(
    py: Python<'_>,
    path: &str,
    header: &Bound<'_, PyDict>,
    channels: &Bound<'_, PyDict>,
) -> PyResult<()> {
    // Order channels by name (header / on-disk order); gather (name, object array).
    let mut chans: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    for (k, v) in channels.iter() {
        // Unwrap a `Channel` object to its raw `.pixels` object array — a deep read
        // (`finish_deep`) now returns `Channel`s the same way a flat read does, so
        // `File(deep_path).write(other_path)` (round-tripping what `channels()` just
        // returned) must accept a `Channel` here too, not only a bare ndarray
        // (`test_deep_rgba`'s `infile.write(...)`).
        let arr = match v.cast::<crate::file::Channel>() {
            Ok(ch) => ch.borrow().pixels.clone_ref(v.py()).into_bound(v.py()),
            Err(_) => v,
        };
        let name: String = k.extract()?;
        // A GROUPED "RGB"/"RGBA"/prefix array (each pixel an `(nsamples, 3|4)`
        // array, e.g. what `finish_deep` hands back from a read, or a write
        // constructed directly with a stacked array) is split into its R/G/B[/A]
        // component channels here, the deep-write inverse of `build_grouped`
        // (`write_deep` used to only accept
        // an already-flat, ungrouped `(H,W)` `dtype=object` array per channel and
        // failed with a confusing `obj_shape`/indexing error on a grouped one).
        if let Some(n_comp) = grouped_component_count(py, &arr)? {
            let prefix = if name == "RGB" || name == "RGBA" { String::new() } else { name.clone() };
            let comps: &[char] = if n_comp == 4 { &['R', 'G', 'B', 'A'] } else { &['R', 'G', 'B'] };
            let split = split_grouped(py, &arr, n_comp)?;
            for (&c, plane) in comps.iter().zip(split) {
                chans.push((crate::file::rgba_full_name(&prefix, c), plane.into_bound(py)));
            }
            continue;
        }
        chans.push((name, arr));
    }
    chans.sort_by(|a, b| a.0.cmp(&b.0));
    if chans.is_empty() {
        return Err(PyValueError::new_err("deep write: no channels"));
    }
    // Dimensions from the first channel's object array shape.
    let (h, w) = obj_shape(&chans[0].1)?;
    let npix = checked_npix(w, h)?;

    // Per-pixel sample counts + per-channel pixel type.  OpenEXR requires every
    // channel at a pixel to have the SAME number of samples: channel 0 establishes
    // the per-pixel count and every other channel must match it (a `None` element
    // counts as 0 samples).  A mismatch would make the count table describe one
    // channel while the heaps hold a different layout, corrupting the file — so
    // reject it here with a coordinate-specific error before anything is written
    // (mirrors the reference validation in PyOpenEXR.cpp:1026-1033).
    let mut counts = vec![0u32; npix];
    let mut ptypes = Vec::with_capacity(chans.len());
    for (ci, (name, arr)) in chans.iter().enumerate() {
        let mut ptype = PixelType::Float;
        let mut seen = false;
        for (pix, count) in counts.iter_mut().enumerate() {
            let elem = obj_get(py, arr, pix / w as usize, pix % w as usize)?;
            let n = if elem.is_none() {
                0
            } else {
                if !seen {
                    ptype = array_ptype(&elem)?;
                    seen = true;
                }
                elem.len()? as u32
            };
            if ci == 0 {
                *count = n;
            } else if *count != n {
                let (col, row) = (pix as i32 % w, pix as i32 / w);
                return Err(PyValueError::new_err(format!(
                    "deep write: channel \"{name}\" has {n} samples at pixel \
                     ({col}, {row}) but the pixel's sample count is {} — all deep \
                     channels must share the same per-pixel sample count",
                    *count
                )));
            }
        }
        ptypes.push(ptype);
    }

    // Local `w*h` count buffer (row-major window order) + per-channel sample heap.
    // make_fb shifts the count slice's base_offset to map absolute window coords
    // onto this local raster, so the buffer is `w*h` regardless of the (possibly
    // non-origin) data window — no absolute-origin allocation or negative address.
    let (min_x, min_y) = dw_min(header, py)?;
    let mut count_buf = vec![0u8; checked_count_bytes(w, h)?];
    for (pix, &c) in counts.iter().enumerate() {
        let o = pix * 4;
        count_buf[o..o + 4].copy_from_slice(&c.to_le_bytes());
    }
    let mut heap = Vec::new();
    let mut bases = Vec::with_capacity(chans.len());
    for (ci, (_n, arr)) in chans.iter().enumerate() {
        bases.push(heap.len() as i64);
        for pix in 0..npix {
            let elem = obj_get(py, arr, pix / w as usize, pix % w as usize)?;
            if !elem.is_none() {
                append_samples(&mut heap, &elem, ptypes[ci])?;
            }
        }
    }

    // Header (deep-valid compression: NONE/RLE/ZIPS; default ZIPS).
    let comp = deep_compression(header, py)?;
    let names: Vec<String> = chans.iter().map(|(n, _)| n.clone()).collect();
    let tiled = is_deeptile(header, py)?;
    let mut hd = Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, exr_core::attr::LineOrder::IncreasingY, comp)
        .map_err(err)?;
    if min_x != 0 || min_y != 0 {
        *hd.data_window_mut().map_err(err)? = imath_rs::Box2i {
            min: imath_rs::V2i::new(min_x, min_y),
            max: imath_rs::V2i::new(min_x + w - 1, min_y + h - 1),
        };
    }
    let mut cl = ChannelList::new();
    for (name, ty) in names.iter().zip(&ptypes) {
        cl.insert(name, Channel::new(*ty, 1, 1, false)).map_err(err)?;
    }
    hd.insert("channels", ChannelListAttribute::new(cl)).map_err(err)?;

    let fb = make_fb(&names, &ptypes, &bases, min_x, min_y, w)?;
    if tiled {
        // Deep tiled: one level, the whole image in `num_x_tiles * num_y_tiles`
        // tiles of the header's tile description.
        hd.insert("tiles", TileDescriptionAttribute::new(tile_desc_from_header(header, py)?)).map_err(err)?;
        hd.set_type(DEEP_TILE).map_err(err)?;
    } else {
        hd.set_type(DEEP_SCANLINE).map_err(err)?;
    }
    // All or nothing, final flush checked: the workspace's one file writer.
    exr_core::io::write_file(path, |stream| -> exr_core::Result<()> {
        if tiled {
            let mut out = DeepTiledOutputFile::new(stream, &hd)?;
            out.set_frame_buffer(&fb)?;
            let nx = out.num_x_tiles(0)?;
            let ny = out.num_y_tiles(0)?;
            out.write_tiles(&count_buf, &heap, 0, nx - 1, 0, ny - 1, 0, 0)
        } else {
            let mut out = DeepScanLineOutputFile::new(stream, &hd)?;
            out.set_frame_buffer(&fb)?;
            out.write_pixels(&count_buf, &heap, h)
        }
    })
    .map_err(err)
}

/// True if the header's `type` is `deeptile` (storage code 3).
fn is_deeptile(header: &Bound<'_, PyDict>, py: Python<'_>) -> PyResult<bool> {
    match header.get_item("type")? {
        Some(v) => {
            let code = py.import("builtins")?.getattr("int")?.call1((v,))?.extract::<i32>().unwrap_or(2);
            Ok(code == 3)
        }
        None => Ok(false),
    }
}

/// The tile description from `header["tiles"]` (default 32×32 ONE_LEVEL).
fn tile_desc_from_header(header: &Bound<'_, PyDict>, _py: Python<'_>) -> PyResult<TileDescription> {
    let mut td = TileDescription::default();
    if let Some(t) = header.get_item("tiles")?
        && let Ok(d) = t.cast::<crate::types::TileDescription>()
    {
        let b = d.borrow();
        td.x_size = b.xSize;
        td.y_size = b.ySize;
    }
    Ok(td)
}

/// An unknown/invalid code used to silently become ZIPS. Raise
/// instead, like the flat path's `comp_to_py`/`header_compression` (`file.rs`) —
/// only an ABSENT key defaults (to ZIPS, the deep default, since plain ZIP isn't
/// deep-valid), and only a genuinely `is_valid_deep()` code is otherwise accepted.
fn deep_compression(header: &Bound<'_, PyDict>, py: Python<'_>) -> PyResult<Compression> {
    let Some(v) = header.get_item("compression")? else {
        return Ok(Compression::Zips);
    };
    let code = py.import("builtins")?.getattr("int")?.call1((v,))?.extract::<i32>()?;
    let comp = crate::file::code_to_compression(code)?;
    if comp.is_valid_deep() {
        Ok(comp)
    } else {
        Err(PyValueError::new_err(format!(
            "compression code {code} ({comp:?}) is not valid for a deep image \
             (only NO_COMPRESSION, RLE_COMPRESSION, ZIPS_COMPRESSION are)"
        )))
    }
}

/// Build a `DeepFrameBuffer` for a LOCAL `w*h` count/sample raster: a UINT count
/// slice + one `DeepSlice` per channel pointing at its heap base.
///
/// The exr-core deep reader/writer address the count slice with ABSOLUTE window
/// coords `base_offset + y*yStride + x*xStride`.  We size the count buffer to the
/// local `w*h` (never the absolute origin) and shift the count slice's
/// `base_offset` by `-(min_y*yStride + min_x*xStride)` so the window's top-left
/// pixel maps to local offset 0 — the C++ base-pointer model.  This makes a
/// negative- or large-positive-origin data window representable without a huge
/// (or negative-addressed) count buffer.  The deep slices' x/y strides stay
/// cosmetic (addressing is count-driven via base_offset + pixelStart).
fn make_fb(
    names: &[String],
    ptypes: &[PixelType],
    bases: &[i64],
    min_x: i32,
    min_y: i32,
    w: i32,
) -> PyResult<DeepFrameBuffer> {
    let y_stride = 4i64 * w as i64;
    let count_base = -((min_y as i64) * y_stride + (min_x as i64) * 4);
    let mut fb = DeepFrameBuffer::new();
    fb.insert_sample_count_slice(Slice {
        type_: PixelType::Uint,
        base_offset: count_base,
        x_stride: 4,
        y_stride: y_stride as usize,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    })
    .map_err(err)?;
    for ((name, ty), base) in names.iter().zip(ptypes).zip(bases) {
        let ss = ptype_size(*ty)?;
        let slice = DeepSlice::new(*ty, *base, 8, 8 * w as usize, ss, 1, 1, 0.0, false, false)
            .map_err(err)?;
        fb.insert(name, slice).map_err(err)?;
    }
    Ok(fb)
}

/// Window-raster per-pixel counts from the local `w*h` count buffer (one Xdr LE
/// `u32` per pixel in row-major window order).
fn read_counts(buf: &[u8], w: i32, h: i32) -> PyResult<Vec<u32>> {
    let n = checked_npix(w, h)?;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        let chunk = buf
            .get(o..o + 4)
            .ok_or_else(|| err("deep count buffer shorter than w*h"))?;
        v.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(v)
}

/// `(min_x, min_y)` of `header["dataWindow"]` (a `((minx,miny),(maxx,maxy))` tuple);
/// `(0, 0)` if absent.
fn dw_min(header: &Bound<'_, PyDict>, _py: Python<'_>) -> PyResult<(i32, i32)> {
    if let Some(dw) = header.get_item("dataWindow")?
        && let Ok(mn) = dw.get_item(0)
    {
        let minx: i32 = mn.get_item(0)?.extract()?;
        let miny: i32 = mn.get_item(1)?.extract()?;
        return Ok((minx, miny));
    }
    Ok((0, 0))
}

// ── Read ────────────────────────────────────────────────────────────────────────

/// Read a deep file (deepscanline or deeptile) into `(header_dict, channels_dict)`.
/// Default groups `R/G/B[/A]` into `(nsamples, 3|4)` arrays; `separate` keeps
/// per-channel arrays.
///
/// Dispatches on the file's own header `type` attribute, NOT a
/// try-the-tiled-reader-then-fall-back-on-any-error probe: that
/// used to swallow a real deeptile parse error and re-attempt the file as
/// deepscanline, producing a confusing secondary error instead of the real one.
///
/// This does NOT use the container version word's `TILED_FLAG` the way
/// `file::read_full_header` does for non-deep files — `TILED_FLAG` combined with
/// `NON_IMAGE_FLAG` is an INVALID version word (`version::validate_version_word`
/// rejects it, matching the C++ format spec), so single-part deep files, tiled or
/// not, never set it. Deepscanline vs deeptile is only distinguishable by the
/// header's own `type` string attribute, so peek that directly with
/// `Header::read_from` (a generic, storage-kind-agnostic header parse) instead.
pub fn read_deep(py: Python<'_>, path: &str, separate: bool) -> PyResult<(Py<PyDict>, Py<PyDict>, i32, i32)> {
    let mut stream = StdIfStream::open(path).map_err(err)?;
    let (peeked, _version_word) = Header::read_from(&mut stream).map_err(err)?;
    if peeked.type_().map_err(err)? == DEEP_TILE {
        read_deep_tiled(py, path, separate)
    } else {
        read_deep_scanline(py, path, separate)
    }
}

fn read_deep_scanline(py: Python<'_>, path: &str, separate: bool) -> PyResult<(Py<PyDict>, Py<PyDict>, i32, i32)> {
    let mut stream = StdIfStream::open(path).map_err(err)?;
    let mut inf = DeepScanLineInputFile::new(&mut stream).map_err(err)?;
    let dw = inf.header().data_window().map_err(err)?;
    let (min_x, min_y) = (dw.min.x, dw.min.y);
    let (w, h) = (dw.max.x - min_x + 1, dw.max.y - min_y + 1);
    let (names, ptypes) = deep_channels(inf.header())?;
    let comp = inf.header().compression().map_err(err)?;

    let fb0 = make_fb(&names, &ptypes, &vec![0i64; names.len()], min_x, min_y, w)?;
    inf.set_frame_buffer(&fb0).map_err(err)?;
    let mut count_buf = vec![0u8; checked_count_bytes(w, h)?];
    inf.read_pixel_sample_counts(dw.min.y, dw.max.y, &mut count_buf).map_err(err)?;
    let counts = read_counts(&count_buf, w, h)?;
    let total: usize = counts.iter().map(|&c| c as usize).sum();

    let (bases, heap_len) = heap_bases(&ptypes, total)?;
    let fb = make_fb(&names, &ptypes, &bases, min_x, min_y, w)?;
    inf.set_frame_buffer(&fb).map_err(err)?;
    let mut heap = vec![0u8; heap_len];
    inf.read_pixels(dw.min.y, dw.max.y, &count_buf, &mut heap).map_err(err)?;

    finish_deep(py, &heap, &bases, &ptypes, &names, &counts, separate, w, h, comp, dw, enums::Storage::deepscanline)
}

fn read_deep_tiled(py: Python<'_>, path: &str, separate: bool) -> PyResult<(Py<PyDict>, Py<PyDict>, i32, i32)> {
    let mut stream = StdIfStream::open(path).map_err(err)?;
    let mut inf = DeepTiledInputFile::new(&mut stream).map_err(err)?;
    let dw = inf.header().data_window().map_err(err)?;
    let (min_x, min_y) = (dw.min.x, dw.min.y);
    let (w, h) = (dw.max.x - min_x + 1, dw.max.y - min_y + 1);
    let (names, ptypes) = deep_channels(inf.header())?;
    let comp = inf.header().compression().map_err(err)?;
    let nx = inf.num_x_tiles(0).map_err(err)?;
    let ny = inf.num_y_tiles(0).map_err(err)?;

    let fb0 = make_fb(&names, &ptypes, &vec![0i64; names.len()], min_x, min_y, w)?;
    inf.set_frame_buffer(&fb0).map_err(err)?;
    let mut count_buf = vec![0u8; checked_count_bytes(w, h)?];
    inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut count_buf).map_err(err)?;
    let counts = read_counts(&count_buf, w, h)?;
    let total: usize = counts.iter().map(|&c| c as usize).sum();

    let (bases, heap_len) = heap_bases(&ptypes, total)?;
    let fb = make_fb(&names, &ptypes, &bases, min_x, min_y, w)?;
    inf.set_frame_buffer(&fb).map_err(err)?;
    let mut heap = vec![0u8; heap_len];
    inf.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &count_buf, &mut heap).map_err(err)?;

    finish_deep(py, &heap, &bases, &ptypes, &names, &counts, separate, w, h, comp, dw, enums::Storage::deeptile)
}

/// Channel names + pixel types of a deep header (on-disk order).
fn deep_channels(header: &Header) -> PyResult<(Vec<String>, Vec<PixelType>)> {
    let mut names = Vec::new();
    let mut ptypes = Vec::new();
    for (name, ch) in header.channels().map_err(err)?.iter() {
        names.push(name.to_string());
        ptypes.push(ch.type_);
    }
    Ok((names, ptypes))
}

/// Per-channel heap base offsets + total heap size for `total` samples/channel.
fn heap_bases(ptypes: &[PixelType], total: usize) -> PyResult<(Vec<i64>, usize)> {
    let mut bases = Vec::with_capacity(ptypes.len());
    let mut len = 0i64;
    for ty in ptypes {
        bases.push(len);
        len += (total * ptype_size(*ty)?) as i64;
    }
    Ok((bases, len as usize))
}

/// De-interleave the flat heap into the channels dict (RGB grouping or separate)
/// + build the deep header dict. Shared by the scanline and tiled readers.
#[allow(clippy::too_many_arguments)]
fn finish_deep(
    py: Python<'_>,
    heap: &[u8],
    bases: &[i64],
    ptypes: &[PixelType],
    names: &[String],
    counts: &[u32],
    separate: bool,
    w: i32,
    h: i32,
    comp: Compression,
    dw: imath_rs::Box2i,
    storage: crate::enums::Storage,
) -> PyResult<(Py<PyDict>, Py<PyDict>, i32, i32)> {
    let npix = checked_npix(w, h)?;
    let mut starts = vec![0usize; npix];
    let mut cum = 0usize;
    for (pix, &c) in counts.iter().enumerate() {
        starts[pix] = cum;
        cum += c as usize;
    }
    let out = PyDict::new(py);
    let has = |prefix: &str, comp: char| names.iter().any(|n| *n == crate::file::rgba_full_name(prefix, comp));
    // Every maximal R/G/B[/A] group, same rule as `file.rs::build_channels`
    // (this used to only group bare top-level R/G/B/A, not a
    // prefixed group like "left.R/G/B").
    let mut prefixes: Vec<&str> = names.iter().filter_map(|n| crate::file::rgba_prefix(n)).collect();
    prefixes.sort_unstable();
    prefixes.dedup();
    let mut grouped: std::collections::HashSet<&str> = std::collections::HashSet::new();
    if !separate {
        for prefix in prefixes {
            if !(has(prefix, 'R') && has(prefix, 'G') && has(prefix, 'B')) {
                continue;
            }
            let alpha = has(prefix, 'A');
            let comps: &[char] = if alpha { &['R', 'G', 'B', 'A'] } else { &['R', 'G', 'B'] };
            let comp_names: Vec<String> = comps.iter().map(|&c| crate::file::rgba_full_name(prefix, c)).collect();
            let cidx: Vec<usize> = comp_names
                .iter()
                .filter_map(|cn| names.iter().position(|n| n == cn))
                .collect();
            if cidx.len() != comp_names.len() {
                continue; // shouldn't happen (just checked), never panic across the FFI boundary
            }
            let group = if prefix.is_empty() {
                if alpha { "RGBA".to_string() } else { "RGB".to_string() }
            } else {
                prefix.to_string()
            };
            // Same two guards as `file.rs::build_channels`: a literal channel
            // already named like the synthesized group key collides with it
            // (test_literal_rgb_key_collision_deep_rejected), and mismatched
            // per-channel pixel types can't be coalesced into one array
            // (test_mixed_type_rgb_coalesce_deep_rejected) — both raise with
            // "separate_channels" in the message, matching the flat-read errors.
            if names.iter().any(|n| n == &group) {
                return Err(PyValueError::new_err(format!(
                    "channel \"{group}\" collides with the grouped deep R/G/B[/A] \
                     channels; pass separate_channels=True to read them individually"
                )));
            }
            if cidx.iter().any(|&ci| ptypes[ci] != ptypes[cidx[0]]) {
                return Err(PyValueError::new_err(
                    "deep channels grouped into RGB/RGBA must share a pixel type; pass \
                     separate_channels=True to read them individually",
                ));
            }
            let arr = build_grouped(py, heap, bases, ptypes, counts, &starts, &cidx, w, h)?;
            out.set_item(group.as_str(), make_deep_channel(py, group.as_str(), arr, ptypes[cidx[0]])?)?;
            for &ci in &cidx {
                grouped.insert(names[ci].as_str());
            }
        }
    }
    for (ci, name) in names.iter().enumerate() {
        if grouped.contains(name.as_str()) {
            continue;
        }
        let arr = build_separate(py, heap, bases[ci], ptypes[ci], counts, &starts, w, h)?;
        out.set_item(name, make_deep_channel(py, name, arr, ptypes[ci])?)?;
    }
    let header = build_deep_header(py, storage, comp, dw)?;
    Ok((header, out.unbind(), w, h))
}

/// One channel's deep pixels: an `(H, W)` numpy `object` array of 1-D sample arrays
/// (or `None` for zero-sample pixels).
// One channel's heap/geometry, all needed at the one call site -- a bundling
// struct would only rename the same 8 fields, not reduce the real arity.
#[allow(clippy::too_many_arguments)]
fn build_separate(
    py: Python<'_>,
    heap: &[u8],
    base: i64,
    ptype: PixelType,
    counts: &[u32],
    starts: &[usize],
    w: i32,
    h: i32,
) -> PyResult<Py<PyAny>> {
    let arr = empty_object_array(py, h, w)?;
    let sz = ptype_size(ptype)?;
    for pix in 0..checked_npix(w, h)? {
        let c = counts[pix] as usize;
        let (y, x) = (pix / w as usize, pix % w as usize);
        if c == 0 {
            continue; // leave as None (np.empty(object) is None-filled)
        }
        let off = base as usize + starts[pix] * sz;
        let samples = samples_to_py(py, heap, off, c, ptype)?;
        obj_set(py, &arr, y, x, samples)?;
    }
    Ok(arr.unbind())
}

/// Grouped deep pixels: an `(H, W)` object array where each pixel is an
/// `(nsamples, n_comp)` numpy array (or `None`).
#[allow(clippy::too_many_arguments)]
fn build_grouped(
    py: Python<'_>,
    heap: &[u8],
    bases: &[i64],
    ptypes: &[PixelType],
    counts: &[u32],
    starts: &[usize],
    cidx: &[usize],
    w: i32,
    h: i32,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let arr = empty_object_array(py, h, w)?;
    for pix in 0..checked_npix(w, h)? {
        let c = counts[pix] as usize;
        let (y, x) = (pix / w as usize, pix % w as usize);
        if c == 0 {
            continue;
        }
        // Build per-component 1-D arrays, then column-stack into (c, n_comp).
        let cols: Vec<Py<PyAny>> = cidx
            .iter()
            .map(|&ci| {
                let ty = ptypes[ci];
                let off = bases[ci] as usize + starts[pix] * ptype_size(ty)?;
                samples_to_py(py, heap, off, c, ty)
            })
            .collect::<PyResult<_>>()?;
        let stacked = np.call_method1("column_stack", (PyTuple::new(py, cols)?,))?;
        obj_set(py, &arr, y, x, stacked.unbind())?;
    }
    Ok(arr.unbind())
}

fn build_deep_header(py: Python<'_>, storage: crate::enums::Storage, comp: Compression, dw: imath_rs::Box2i) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("type", storage)?;
    // Deep files can only be NONE/RLE/ZIPS in practice (deep_compression on write
    // restricts to those), so comp_to_py cannot hit its HTJ2K/Invalid error path
    // here — but propagate it rather than assume, in case a deep file was produced
    // by some other writer with an unusual compression code.
    d.set_item("compression", crate::file::comp_to_py(comp)?)?;
    let mn = PyTuple::new(py, [dw.min.x, dw.min.y])?;
    let mx = PyTuple::new(py, [dw.max.x, dw.max.y])?;
    d.set_item("dataWindow", PyTuple::new(py, [mn, mx])?)?;
    Ok(d.unbind())
}

fn make_deep_channel(py: Python<'_>, name: &str, pixels: Py<PyAny>, ptype: PixelType) -> PyResult<Py<crate::file::Channel>> {
    crate::file::channel_with_pixels(py, name, pixels, ptype)
}

// ── numpy object-array helpers ───────────────────────────────────────────────────

fn empty_object_array<'py>(py: Python<'py>, h: i32, w: i32) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "object")?;
    np.call_method("empty", (PyTuple::new(py, [h, w])?,), Some(&kwargs))
}

fn obj_shape(arr: &Bound<'_, PyAny>) -> PyResult<(i32, i32)> {
    let shape: Vec<i32> = arr.getattr("shape")?.extract()?;
    if shape.len() != 2 {
        return Err(PyTypeError::new_err("deep channel must be a 2-D (H,W) object array"));
    }
    Ok((shape[0], shape[1]))
}

fn obj_get<'py>(py: Python<'py>, arr: &Bound<'py, PyAny>, y: usize, x: usize) -> PyResult<Bound<'py, PyAny>> {
    arr.call_method1("__getitem__", (PyTuple::new(py, [y, x])?,))
}

fn obj_set(py: Python<'_>, arr: &Bound<'_, PyAny>, y: usize, x: usize, value: Py<PyAny>) -> PyResult<()> {
    arr.call_method1("__setitem__", (PyTuple::new(py, [y, x])?, value))?;
    Ok(())
}

/// The number of components (3 or 4) if `arr`'s first non-`None` pixel is a 2-D
/// `(nsamples, n_comp)` array (a GROUPED RGB/RGBA deep channel, `finish_deep`'s
/// read-side output for the "RGB"/"RGBA"/prefix group), or `None` if every element
/// is 1-D or `None` (an ordinary single-component deep channel).
fn grouped_component_count(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let (h, w) = obj_shape(arr)?;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let elem = obj_get(py, arr, y, x)?;
            if elem.is_none() {
                continue;
            }
            let ndim: usize = elem.getattr("ndim")?.extract()?;
            if ndim != 2 {
                return Ok(None);
            }
            let n_comp: usize = elem.getattr("shape")?.get_item(1)?.extract()?;
            return if matches!(n_comp, 3 | 4) { Ok(Some(n_comp)) } else { Ok(None) };
        }
    }
    Ok(None) // every pixel is None -- nothing to split, treat as an ordinary channel
}

/// Split a GROUPED `(H, W)` object array (each pixel an `(nsamples, n_comp)` numpy
/// array, or `None`) into `n_comp` per-component `(H, W)` object arrays of
/// `(nsamples,)` 1-D arrays — the write-side inverse of `build_grouped`.
/// `elem[:, ci]` is built as `elem[(slice(None), ci)]` since
/// PyO3 has no numpy-fancy-indexing sugar.
fn split_grouped(py: Python<'_>, arr: &Bound<'_, PyAny>, n_comp: usize) -> PyResult<Vec<Py<PyAny>>> {
    let (h, w) = obj_shape(arr)?;
    let outs: Vec<Bound<'_, PyAny>> =
        (0..n_comp).map(|_| empty_object_array(py, h, w)).collect::<PyResult<_>>()?;
    let full = PySlice::full(py);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let elem = obj_get(py, arr, y, x)?;
            if elem.is_none() {
                continue;
            }
            for (ci, out) in outs.iter().enumerate() {
                let key = PyTuple::new(py, [full.clone().into_any(), ci.into_pyobject(py)?.into_any()])?;
                let col = elem.get_item(key)?;
                // A view isn't safe to keep across further numpy ops on `elem`;
                // materialise a fresh contiguous 1-D array (matches how every
                // other deep sample array in this module is built).
                let col = col.call_method0("copy")?;
                obj_set(py, out, y, x, col.unbind())?;
            }
        }
    }
    Ok(outs.into_iter().map(|o| o.unbind()).collect())
}
