//! Idiomatic `Image` read/write facade over the verified low-level core.
//!
//! This is the **one** place in `exr-core` where an idiomatic, non-1:1 Rust
//! shape is allowed; the rest of the crate mirrors the C++ structure.  It is deliberately
//! *minimal*: it covers the common "load / save a single-part image" case
//! cleanly and **drives** the verified scanline / tiled readers and writers —
//! it never reimplements decoding or encoding.  Anything outside the common
//! case (deep data, multipart) returns a clear
//! [`ExrError`] that points at the low-level API, rather than being
//! half-supported.
//!
//! # When to use this vs. the low-level API
//!
//! | Task                                   | Use                                   |
//! |----------------------------------------|---------------------------------------|
//! | Read/write a single-part RGBA image    | [`Image::read`] / [`Image::write`]    |
//! | Deep data                              | [`crate::DeepScanLineInputFile`] etc. |
//! | Multipart / multiview                  | [`crate::MultiPartInputFile`] etc.    |
//! | Per-tile / per-scanline streaming      | [`crate::scanline`] / [`crate::tiled`]|
//! | Tiled / MIP / RIP / subsampled files   | [`Image::read_all`] / [`Image::read_level`], [`Image::with_tiling`] |
//!
//! # Pixel layout
//!
//! Each channel is stored as one flat buffer in raster order over the data
//! window: `width * height` samples, row-major, `index = (y - min.y) * width +
//! (x - min.x)`.  A subsampled channel ([`Image::insert_sub`], [`Image::read_all`])
//! holds its own smaller sample grid ([`Image::samples`]); [`Image::read`] never
//! returns one (it refuses with [`Unsupported::Subsampled`]), so code that indexes
//! `y * width + x` is never handed a mis-sized buffer.
//!
//! # Single source of truth
//!
//! [`Image::write`] builds a [`Header`] + [`FrameBuffer`] and runs them through
//! [`scanline::OutputFile`] (or [`TiledOutputFile`], every level, when a tiling is
//! set), producing byte-for-byte the same file the low-level path would (it *is*
//! that path, with an ergonomic buffer-assembly layer).  [`Image::read`] likewise drives [`scanline::InputFile`] /
//! [`crate::tiled::TiledInputFile`].

use std::collections::BTreeMap;
use std::path::Path;

use imath_rs::Box2i;

use crate::attr::{
    Attribute, Compression, ExrString, LineOrder, Name, PixelType, StringAttribute,
    TileDescription,
};
use crate::container::version;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::header::{self, Header};
use crate::io::{IStream, OStream, StdIfStream};
use crate::scanline;
use crate::tiled::misc as level_math;
use crate::tiled::{GenericTiledReader, TiledInputFile, TiledOutputFile};

use imath_rs::Half;

// The structured escape-hatch kind lives in `crate::error` (next to the
// `ExrError::Unsupported` variant that carries it), because it is now part of
// the error type rather than a facade-private classification. Re-exported here
// so `api::Unsupported` and the crate-root `exr_core::Unsupported` still resolve.
pub use crate::error::Unsupported;

pub mod attrs;
pub mod describe;
pub mod planar;
pub mod planes;
mod store;
pub use attrs::ImageAttributes;
use store::{Chan, Planes, ReadPlan, check_level, has_level, level_grid, level_window, sample_dims};

/// Classify a facade escape-hatch rejection.
///
/// Returns `Some(kind)` when `e` is the [`ExrError::Unsupported`] that [`read`]
/// / `ReadPlan::from_header` produce for a file shape the [`Image`] facade does
/// not cover, else `None` (any genuine error). This is now a trivial match on
/// the structured variant — the previous fragile `String::contains` matching is
/// gone. Kept as a convenience adapter (and for API compatibility) so consumers
/// that only want the `Option<Unsupported>` shape need not pattern-match the
/// whole [`ExrError`] enum themselves.
pub fn classify_unsupported(e: &ExrError) -> Option<Unsupported> {
    match e {
        ExrError::Unsupported(kind) => Some(*kind),
        _ => None,
    }
}

/// Read a single-part scanline or tiled (non-deep) EXR file into an [`Image`].
///
/// Free-function convenience wrapper over [`Image::read`], re-exported at the
/// crate root as `exr_core::read`.
pub fn read<P: AsRef<Path>>(path: P) -> Result<Image> {
    Image::read(path)
}

/// Convenience: write a 4-channel **RGBA `Float`** single-part scanline EXR by
/// sampling `colors(x, y)` per pixel (raster order over `[0,width) x
/// [0,height)`), ZIP-compressed. Re-exported at the crate root as
/// `exr_core::write_rgba_file`.
///
/// This mirrors the old `exrs`-fork `write_rgba_file` helper (same
/// `Fn(usize, usize) -> (f32, f32, f32, f32)` shape) so "just save the pixels"
/// call sites (screenshots, frame dumps, maketx fixtures) port unchanged. It
/// drives [`Image::write`] — no separate encode path.
pub fn write_rgba_file<P: AsRef<Path>>(
    path: P,
    width: usize,
    height: usize,
    colors: impl Fn(usize, usize) -> (f32, f32, f32, f32),
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(ExrError::Arg(
            "write_rgba_file: width and height must be non-zero".into(),
        ));
    }
    // Validate dimensions BEFORE allocating/filling, so an oversized request
    // fails cheaply instead of allocating gigabytes first.
    //
    // The data-window max is an i32 pixel coordinate; `width`/`height` are
    // `usize`. A plain `as i32` would silently truncate (wrong `dataWindow`) for
    // dimensions past the i32 range — reject those. `width >= 1`/`height >= 1`
    // hold (checked above), so `width - 1` cannot underflow.
    let max_x = i32::try_from(width - 1).map_err(|_| {
        ExrError::Arg(format!(
            "write_rgba_file: width {width} exceeds the i32 pixel-coordinate range"
        ))
    })?;
    let max_y = i32::try_from(height - 1).map_err(|_| {
        ExrError::Arg(format!(
            "write_rgba_file: height {height} exceeds the i32 pixel-coordinate range"
        ))
    })?;
    // Element count must not overflow `usize` (matters on 32-bit targets; on
    // 64-bit the i32 caps above already bound it). The buffers are `Vec<f32>`,
    // so a pathological-but-in-range `n` can still OOM — that is an inherent
    // cost of the requested size, not an arithmetic bug.
    let n = width.checked_mul(height).ok_or_else(|| {
        ExrError::Overflow(format!(
            "write_rgba_file: width*height overflows usize (width={width}, height={height})"
        ))
    })?;
    let (mut r, mut g) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut b, mut a) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for y in 0..height {
        for x in 0..width {
            let (cr, cg, cb, ca) = colors(x, y);
            r.push(cr);
            g.push(cg);
            b.push(cb);
            a.push(ca);
        }
    }
    let data_window = Box2i {
        min: imath_rs::V2i { x: 0, y: 0 },
        max: imath_rs::V2i { x: max_x, y: max_y },
    };
    Image::new(data_window)
        .with_channel("R", ChannelData::Float(r))
        .with_channel("G", ChannelData::Float(g))
        .with_channel("B", ChannelData::Float(b))
        .with_channel("A", ChannelData::Float(a))
        .write(path, Compression::Zip)
}

/// Write several images as the parts of a single multi-part **scanline** EXR.
///
/// Each tuple is `(image, compression, part_name)`; the name becomes the part's
/// required `name` attribute. Drives the verified
/// [`crate::multipart::MultiPartOutputFile`] plus per-part scanline writers,
/// reusing the same header/framebuffer assembly as [`Image::write`]. Used by the
/// Python binding's `File([Part, …]).write()`.
pub fn write_multipart_scanline<P: AsRef<Path>>(
    path: P,
    parts: Vec<(Image, Compression, String)>,
) -> Result<()> {
    if parts.is_empty() {
        return Err(ExrError::Arg("multipart write needs at least one part.".into()));
    }
    let mut headers = Vec::with_capacity(parts.len());
    for (img, comp, name) in &parts {
        if img.tiling.is_some() || !img.levels.is_empty() {
            return Err(ExrError::Arg(
                "write_multipart_scanline writes scanline parts; this Image is tiled.".into(),
            ));
        }
        let mut header = img.build_header(*comp)?;
        header.insert("name", StringAttribute::new(ExrString::from(name.as_str())))?;
        // Unlike a single-part file, EVERY multipart header must carry a type
        // (C++ `ImfMultiPartOutputFile.cpp:117` throws without one, and
        // `Header::sanity_check(_, is_multipart = true)` requires name + type).
        header.set_type(header::SCANLINE_IMAGE)?;
        headers.push(header);
    }
    let path = path.as_ref();
    let path_str = path
        .to_str()
        .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
    crate::io::write_file(path_str, |stream| {
        let mut mp = crate::multipart::MultiPartOutputFile::new(stream, &headers)?;
        for (i, (img, _, _)) in parts.iter().enumerate() {
            let (buffer, fb) = img.base.write_buf(img.data_window)?;
            let mut out = mp.output_part(i as i32)?;
            out.set_frame_buffer(&fb)?;
            out.write_pixels(&buffer, img.height() as i32)?;
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// ChannelData — one typed, flat pixel buffer per channel
// ---------------------------------------------------------------------------

/// One channel's pixels as a flat, typed buffer in raster order over the data
/// window (`width * height` samples, row-major).
///
/// The variant records the channel's [`PixelType`]; the facade preserves the
/// type the file declared (or the type the builder was given).
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelData {
    /// 16-bit float samples (`HALF`).
    Half(Vec<Half>),
    /// 32-bit float samples (`FLOAT`).
    Float(Vec<f32>),
    /// Unsigned 32-bit integer samples (`UINT`).
    Uint(Vec<u32>),
}

impl ChannelData {
    /// The EXR [`PixelType`] this buffer carries.
    pub fn pixel_type(&self) -> PixelType {
        match self {
            ChannelData::Half(_) => PixelType::Half,
            ChannelData::Float(_) => PixelType::Float,
            ChannelData::Uint(_) => PixelType::Uint,
        }
    }

    /// Number of samples in the buffer.
    pub fn len(&self) -> usize {
        match self {
            ChannelData::Half(v) => v.len(),
            ChannelData::Float(v) => v.len(),
            ChannelData::Uint(v) => v.len(),
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size in bytes of one sample of this channel's type.
    ///
    /// `ChannelData` only ever holds `Half`/`Float`/`Uint` samples, so this
    /// never actually hits the `NumPixelTypes` sentinel `crate::misc::
    /// pixel_type_size` (the canonical size function) rejects.
    fn sample_size(&self) -> Result<usize> {
        crate::misc::pixel_type_size(self.pixel_type())
    }

    // -------------------------------------------------------------------
    // Channel <-> f32 / bytes conversion, single source of truth
    // for every downstream crate (cryptomatte, exr-gpu, exr-view, openexr-py)
    // instead of each hand-rolling its own copy.
    // -------------------------------------------------------------------

    /// Every sample widened to `f32`. UINT is a **raw cast** (`crate::convert
    /// ::uint_to_float`), matching C++'s `uintToFloat` — NOT normalized; a
    /// caller that wants a `[0,1]` display value (exr-view) applies that
    /// itself, since raw-vs-normalized is a display policy, not a type
    /// conversion.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            ChannelData::Half(v) => v.iter().map(|&h| h.to_f32()).collect(),
            ChannelData::Float(v) => v.clone(),
            ChannelData::Uint(v) => v.iter().map(|&u| crate::convert::uint_to_float(u)).collect(),
        }
    }

    /// One sample widened to `f32` (raw UINT cast, see [`Self::to_f32`]), or
    /// `None` if `i` is out of range — a bounds-checked alternative to
    /// indexing so a caller with a stale/mismatched index gets an explicit
    /// "missing" signal instead of a panic or a value from the wrong sample.
    pub fn sample_f32(&self, i: usize) -> Option<f32> {
        match self {
            ChannelData::Half(v) => v.get(i).map(|&h| h.to_f32()),
            ChannelData::Float(v) => v.get(i).copied(),
            ChannelData::Uint(v) => v.get(i).map(|&u| crate::convert::uint_to_float(u)),
        }
    }

    /// Convert every sample to `to`'s [`PixelType`], through the canonical
    /// [`crate::convert`] boundary rules (UINT<->HALF overflow, live-path
    /// FLOAT->HALF rounding — see that module's docs).
    ///
    /// # Errors
    /// [`ExrError::Arg`] if `to` is the `NumPixelTypes` sentinel.
    pub fn convert(&self, to: PixelType) -> Result<ChannelData> {
        use crate::convert;
        Ok(match (self, to) {
            (ChannelData::Uint(v), PixelType::Uint) => ChannelData::Uint(v.clone()),
            (ChannelData::Half(v), PixelType::Uint) => {
                ChannelData::Uint(v.iter().map(|&h| convert::half_to_uint(h)).collect())
            }
            (ChannelData::Float(v), PixelType::Uint) => {
                ChannelData::Uint(v.iter().map(|&f| convert::float_to_uint(f)).collect())
            }
            (ChannelData::Uint(v), PixelType::Half) => {
                ChannelData::Half(v.iter().map(|&u| convert::uint_to_half(u)).collect())
            }
            (ChannelData::Half(v), PixelType::Half) => ChannelData::Half(v.clone()),
            (ChannelData::Float(v), PixelType::Half) => {
                ChannelData::Half(v.iter().map(|&f| convert::float_to_half(f)).collect())
            }
            (ChannelData::Uint(v), PixelType::Float) => {
                ChannelData::Float(v.iter().map(|&u| convert::uint_to_float(u)).collect())
            }
            (ChannelData::Half(v), PixelType::Float) => {
                ChannelData::Float(v.iter().map(|&h| h.to_f32()).collect())
            }
            (ChannelData::Float(v), PixelType::Float) => ChannelData::Float(v.clone()),
            (_, PixelType::NumPixelTypes) => {
                return Err(ExrError::Arg("Unknown pixel data type.".into()));
            }
        })
    }

    /// Encode to **native-endian** bytes (the low-level `FrameBuffer`'s wire
    /// convention — see the "Plane codecs" comment below this `impl` block).
    pub fn to_ne_bytes(&self) -> Vec<u8> {
        match self {
            ChannelData::Half(v) => v.iter().flat_map(|h| h.bits().to_ne_bytes()).collect(),
            ChannelData::Float(v) => v.iter().flat_map(|f| f.to_ne_bytes()).collect(),
            ChannelData::Uint(v) => v.iter().flat_map(|u| u.to_ne_bytes()).collect(),
        }
    }

    /// Decode `pixels` samples of `type_` from native-endian `bytes`.
    ///
    /// # Errors
    /// [`ExrError::Logic`] if `bytes` is too short; [`ExrError::Arg`] for the
    /// `NumPixelTypes` sentinel.
    pub fn from_ne_bytes(type_: PixelType, bytes: &[u8], pixels: usize) -> Result<ChannelData> {
        decode_plane_native(type_, bytes, pixels)
    }

    /// Encode to little-endian XDR wire bytes (EXR's on-disk/wire format is
    /// LE). Half/Uint/Float samples are `2`/`4`/`4`
    /// bytes each, the raw bit pattern (not a value re-encoding).
    pub fn to_xdr_bytes(&self) -> Vec<u8> {
        match self {
            ChannelData::Half(v) => v.iter().flat_map(|h| h.bits().to_le_bytes()).collect(),
            ChannelData::Float(v) => v.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect(),
            ChannelData::Uint(v) => v.iter().flat_map(|u| u.to_le_bytes()).collect(),
        }
    }

    /// Decode `count` samples of `type_` from little-endian XDR wire `bytes`.
    ///
    /// # Errors
    /// [`ExrError::Logic`] if `bytes.len()` does not exactly match
    /// `count * pixel_type_size(type_)`; [`ExrError::Arg`] for the
    /// `NumPixelTypes` sentinel.
    pub fn from_xdr_bytes(type_: PixelType, bytes: &[u8], count: usize) -> Result<ChannelData> {
        let sz = crate::misc::pixel_type_size(type_)?;
        let need = count
            .checked_mul(sz)
            .ok_or_else(|| ExrError::Arg("XDR channel byte count overflows.".into()))?;
        if bytes.len() != need {
            return Err(ExrError::Logic(format!(
                "XDR channel data is {} bytes but {count} samples of {sz} bytes need {need}.",
                bytes.len()
            )));
        }
        Ok(match type_ {
            PixelType::Half => ChannelData::Half(
                bytes
                    .as_chunks::<2>().0.iter()
                    .map(|c| Half::from_bits(u16::from_le_bytes(*c)))
                    .collect(),
            ),
            PixelType::Float => ChannelData::Float(
                bytes
                    .as_chunks::<4>().0.iter()
                    .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
                    .collect(),
            ),
            PixelType::Uint => ChannelData::Uint(
                bytes
                    .as_chunks::<4>().0.iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect(),
            ),
            PixelType::NumPixelTypes => {
                return Err(ExrError::Arg("Unknown pixel data type.".into()));
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Image — an in-memory single-part image
// ---------------------------------------------------------------------------

/// A simple in-memory single-part image: the data/display windows, the
/// compression and line order, an ordered map of channel name →
/// [`ChannelData`], and — for a tiled image — its tile description and any MIP/RIP
/// levels beyond level 0.
///
/// Channels are kept in a [`BTreeMap`], so iteration yields them in the same
/// alphabetical order the EXR header uses — round-tripping the on-disk channel
/// order exactly.
///
/// # Levels
/// Level `(0, 0)` is the image itself (its channels are [`Image::channel`]).
/// The other levels of a MIPMAP/RIPMAP tiling are separate plane sets
/// ([`Image::insert_level`]); a level's window is never stored, it is always
/// [`Image::level_window`] (derived from `data_window` and the tiling), so a level
/// cannot disagree with its geometry. `Image::read` returns level 0 only with no
/// tiling ([`Image::read_all`] keeps the file as stored).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Pixel data window (inclusive min/max), as in the header.
    pub data_window: Box2i,
    /// Display window (inclusive min/max), as in the header.
    pub display_window: Box2i,
    /// Compression of the source file, or the default used when writing.
    pub compression: Compression,
    /// Scanline order of the source file (informational; the writer uses the
    /// header default unless changed).
    pub line_order: LineOrder,
    /// `Some`: written as a tiled file with this tile size / level mode / rounding.
    tiling: Option<TileDescription>,
    /// Level 0: channel name → samples + sampling, alphabetical (header) order.
    base: Planes,
    /// Every other level `(lx, ly)` of a MIPMAP/RIPMAP tiling; empty otherwise.
    levels: BTreeMap<(i32, i32), Planes>,
    /// Header metadata not derivable from the fields above (chromaticities,
    /// comments, `pixelAspectRatio`, custom / opaque attributes, …).  Captured
    /// on read and written back by [`Image::write`] — see [`ImageAttributes`].
    attributes: ImageAttributes,
}

/// A read-only view of one level of an [`Image`]: its derived window and planes.
#[derive(Clone, Copy, Debug)]
pub struct LevelView<'a> {
    window: Box2i,
    planes: &'a Planes,
}

impl LevelView<'_> {
    /// This level's data window (`min` is the image's `min`).
    pub fn window(&self) -> Box2i {
        self.window
    }

    /// One channel's samples over [`Self::window`] (its sample grid, see
    /// [`Self::samples`]).
    pub fn channel(&self, name: &str) -> Option<&ChannelData> {
        self.planes.chans.get(name).map(|c| &c.data)
    }

    /// Channels in header (alphabetical) order.
    pub fn channels(&self) -> impl Iterator<Item = (&str, &ChannelData)> {
        self.planes.chans.iter().map(|(k, c)| (k.as_str(), &c.data))
    }

    /// `(xSampling, ySampling)` of a channel.
    pub fn sampling(&self, name: &str) -> Option<(i32, i32)> {
        self.planes.chans.get(name).map(|c| (c.xs, c.ys))
    }

    /// `(columns, rows)` of a channel's sample grid over this level's window; `None`
    /// for an unknown channel.
    pub fn samples(&self, name: &str) -> Option<(usize, usize)> {
        let c = self.planes.chans.get(name)?;
        sample_dims(self.window, c.xs, c.ys).ok()
    }
}

impl Image {
    /// Create an empty image with the given data window.
    ///
    /// The display window defaults to the data window; compression defaults to
    /// [`Compression::Zip`] and line order to [`LineOrder::IncreasingY`] (the
    /// EXR header defaults).  Add channels with [`Image::with_channel`] /
    /// [`Image::insert_channel`].
    pub fn new(data_window: Box2i) -> Self {
        Self {
            data_window,
            display_window: data_window,
            compression: Compression::Zip,
            line_order: LineOrder::IncreasingY,
            tiling: None,
            base: Planes::default(),
            levels: BTreeMap::new(),
            attributes: ImageAttributes::new(),
        }
    }

    /// The header metadata this image carries beyond its own fields.
    ///
    /// Populated by [`Image::read`] from the source header and written back by
    /// [`Image::write`], so a facade round trip (and therefore `exr-rs
    /// convert`) preserves comments, chromaticities, custom attributes and
    /// anything the reader kept as an opaque blob.
    pub fn attributes(&self) -> &ImageAttributes {
        &self.attributes
    }

    /// Mutable access to the carried header metadata — add, replace or drop
    /// attributes before writing.
    pub fn attributes_mut(&mut self) -> &mut ImageAttributes {
        &mut self.attributes
    }

    /// Image width in pixels (data-window extent).
    ///
    /// The subtraction is widened to `i64` before the `+ 1` so an extreme
    /// user-built window cannot overflow/panic; the raw extent is returned
    /// as-is (it may be `<= 0` for an inverted window, matching C++).
    pub fn width(&self) -> i64 {
        self.data_window.max.x as i64 - self.data_window.min.x as i64 + 1
    }

    /// Image height in pixels (data-window extent).
    ///
    /// Widened like [`Image::width`]; see it for the overflow/inverted-window
    /// contract.
    pub fn height(&self) -> i64 {
        self.data_window.max.y as i64 - self.data_window.min.y as i64 + 1
    }

    /// Builder: add a full-resolution channel and return `self` (chainable).
    ///
    /// Use [`Image::insert_channel`] for the fallible, non-chaining form when
    /// you want to detect a size mismatch up front.
    #[must_use]
    pub fn with_channel(mut self, name: impl Into<String>, data: ChannelData) -> Self {
        self.base.chans.insert(name.into(), Chan::full(data));
        self
    }

    /// Insert (or replace) a full-resolution channel, checking its buffer length
    /// matches the data window (`width * height`).
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if `data.len() != width * height`.
    pub fn insert_channel(&mut self, name: impl Into<String>, data: ChannelData) -> Result<()> {
        self.insert_sub(name, data, 1, 1)
    }

    /// Insert (or replace) a channel stored at sampling `(xs, ys)` — a chroma channel
    /// has one sample per `xs` x `ys` block. `data` holds its sample grid,
    /// [`Image::samples`] columns by rows. Subsampling needs a scanline file (tiled
    /// files cannot carry it; the write then fails), and the EXR header requires the
    /// data window's min and size to be multiples of the sampling.
    ///
    /// # Errors
    /// [`ExrError::Arg`] for a sampling below 1 or a `data` length that is not the
    /// sample grid.
    pub fn insert_sub(&mut self, name: impl Into<String>, data: ChannelData, xs: i32, ys: i32) -> Result<()> {
        let name = name.into();
        let (w, h) = sample_dims(self.data_window, xs, ys)?;
        if data.len() != w * h {
            return Err(ExrError::Arg(format!(
                "Channel \"{name}\" has {} samples but the data window needs {} ({w}x{h} at \
                 sampling {xs}x{ys}).",
                data.len(),
                w * h,
            )));
        }
        self.base.chans.insert(name, Chan { data, xs, ys });
        Ok(())
    }

    /// Look up a channel of level 0 by name. A subsampled channel's buffer is its
    /// sample grid ([`Image::samples`]), not `width * height`.
    pub fn channel(&self, name: &str) -> Option<&ChannelData> {
        self.base.chans.get(name).map(|c| &c.data)
    }

    /// Iterate the channels of level 0 in header (alphabetical) order.
    pub fn channels(&self) -> impl Iterator<Item = (&str, &ChannelData)> {
        self.base.chans.iter().map(|(k, c)| (k.as_str(), &c.data))
    }

    /// `(xSampling, ySampling)` of a channel of level 0 (`(1, 1)` = full resolution).
    pub fn sampling(&self, name: &str) -> Option<(i32, i32)> {
        self.base.chans.get(name).map(|c| (c.xs, c.ys))
    }

    /// `(columns, rows)` of a level-0 channel's sample grid; `None` for an unknown
    /// channel or an invalid data window.
    pub fn samples(&self, name: &str) -> Option<(usize, usize)> {
        let c = self.base.chans.get(name)?;
        sample_dims(self.data_window, c.xs, c.ys).ok()
    }

    /// Consume the image and take ownership of its level-0 channels, in header
    /// (alphabetical) order.
    ///
    /// For a caller that owns a freshly-read, about-to-be-dropped
    /// `Image` and wants its channel data without cloning every pixel
    /// buffer (the pattern `img.channels().map(|(n, c)| (n.to_owned(),
    /// c.clone()))` was cloning the whole image just to move it into a
    /// different owned container) -- `into_channels` moves the
    /// `BTreeMap`'s entries out directly.
    ///
    /// # Errors
    /// [`ExrError::Arg`] naming the channel if one is subsampled: the plain
    /// `(String, ChannelData)` pairs cannot carry its sampling, and dropping it
    /// silently would hand back a buffer that is not `width * height`.
    pub fn into_channels(self) -> Result<impl Iterator<Item = (String, ChannelData)>> {
        if let Some((name, c)) = self.base.chans.iter().find(|(_, c)| (c.xs, c.ys) != (1, 1)) {
            return Err(ExrError::Arg(format!(
                "Channel \"{name}\" is subsampled ({}x{}); into_channels would drop its sampling.",
                c.xs, c.ys
            )));
        }
        Ok(self.base.chans.into_iter().map(|(k, c)| (k, c.data)))
    }

    // -----------------------------------------------------------------------
    // Tiling and levels
    // -----------------------------------------------------------------------

    /// The tile description this image is written with, if tiled.
    pub fn tiling(&self) -> Option<TileDescription> {
        self.tiling
    }

    /// Builder form of [`Image::set_tiling`].
    #[must_use]
    pub fn with_tiling(mut self, td: TileDescription) -> Self {
        self.set_tiling(Some(td));
        self
    }

    /// Choose tiled (`Some`) or scanline (`None`) output. Changing the tiling changes
    /// the level geometry, so any levels inserted so far are dropped.
    pub fn set_tiling(&mut self, td: Option<TileDescription>) {
        self.tiling = td;
        self.levels.clear();
    }

    /// `(x levels, y levels)` of the current tiling (`(1, 1)` when untiled or
    /// ONE_LEVEL).
    ///
    /// # Errors
    /// [`ExrError::Arg`] for an unusable tiling or data window.
    pub fn num_levels(&self) -> Result<(i32, i32)> {
        let Some(td) = &self.tiling else { return Ok((1, 1)) };
        let dw = self.data_window;
        Ok((
            level_math::num_x_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?,
            level_math::num_y_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?,
        ))
    }

    /// Data window of level `(lx, ly)`, derived from the data window and the tiling
    /// (C++ `dataWindowForLevel`).
    ///
    /// # Errors
    /// [`ExrError::Arg`] if the level does not exist under the current tiling.
    pub fn level_window(&self, lx: i32, ly: i32) -> Result<Box2i> {
        let exists = match &self.tiling {
            Some(td) => has_level(td, self.data_window, lx, ly)?,
            None => (lx, ly) == (0, 0),
        };
        if !exists {
            return Err(ExrError::Arg(format!("Level ({lx}, {ly}) does not exist in this image's tiling.")));
        }
        match &self.tiling {
            Some(td) => level_window(td, self.data_window, lx, ly),
            None => Ok(self.data_window),
        }
    }

    /// A view of level `(lx, ly)`: [`LevelView`] of level 0 is this image's own
    /// channels. `None` if the level does not exist or was not inserted.
    pub fn level(&self, lx: i32, ly: i32) -> Option<LevelView<'_>> {
        let window = self.level_window(lx, ly).ok()?;
        let planes = if (lx, ly) == (0, 0) { &self.base } else { self.levels.get(&(lx, ly))? };
        Some(LevelView { window, planes })
    }

    /// Insert level `(lx, ly)` (not `(0, 0)`, which is the image itself) of a
    /// MIPMAP/RIPMAP tiling from `channels`, full-resolution planes over
    /// [`Image::level_window`]. Replaces a level inserted before.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if the tiling has no such level (set it first with
    /// [`Image::set_tiling`]), or the channels differ from level 0's in name or type,
    /// or a plane is not the level's `width * height`.
    pub fn insert_level<N: Into<String>>(
        &mut self,
        lx: i32,
        ly: i32,
        channels: impl IntoIterator<Item = (N, ChannelData)>,
    ) -> Result<()> {
        if (lx, ly) == (0, 0) {
            return Err(ExrError::Arg("Level (0, 0) is the image itself; use insert_channel.".into()));
        }
        let td = self.tiling.ok_or_else(|| ExrError::Arg("insert_level needs a tiling; set_tiling first.".into()))?;
        let planes = Planes { chans: channels.into_iter().map(|(n, d)| (n.into(), Chan::full(d))).collect() };
        check_level(&td, self.data_window, &self.base, lx, ly, &planes)?;
        self.levels.insert((lx, ly), planes);
        Ok(())
    }

    /// The planes of `(lx, ly)` for writing: level 0 or an inserted level, else a hard
    /// error naming the missing level.
    fn planes_at(&self, lx: i32, ly: i32) -> Result<&Planes> {
        if (lx, ly) == (0, 0) {
            return Ok(&self.base);
        }
        self.levels
            .get(&(lx, ly))
            .ok_or_else(|| ExrError::Arg(format!("Level ({lx}, {ly}) of the tiling has no pixel data; insert_level it.")))
    }

    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

    /// Read a single-part scanline or tiled (non-deep) EXR file into an
    /// [`Image`]: level 0 as one flat raster, tiling dropped.
    ///
    /// # Errors
    /// * Propagates any I/O or decode error from the low-level reader.
    /// * Returns [`ExrError::Unsupported`] for deep / multipart / subsampled
    ///   files (see [`Unsupported`]); its [`Display`](std::fmt::Display) points
    ///   the caller at [`crate::DeepScanLineInputFile`] /
    ///   [`crate::MultiPartInputFile`] / [`Image::read_all`], and
    ///   consumers route on the variant (via [`classify_unsupported`]) rather
    ///   than the message. The facade never half-supports these shapes.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Image> {
        let path = path.as_ref();
        let path_str = path.to_str().ok_or_else(|| {
            ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display()))
        })?;
        // User-meaningful milestone: opening a file via the facade (info). The
        // dims/channels/compression follow once the header is parsed.
        log::info!("reading image: {path_str}");
        let mut stream = StdIfStream::open(path_str)?;
        let image = Image::read_from(&mut stream)?;
        log::info!(
            "read image: {path_str} {}x{} channels={} compression={:?}",
            image.width(),
            image.height(),
            image.base.chans.len(),
            image.compression
        );
        Ok(image)
    }

    /// Read a single-part scanline or tiled (non-deep) image from an arbitrary
    /// input stream — the stream form of [`Image::read`].
    ///
    /// The stream is rewound to the start (`seekg(0)`) before dispatch, so the
    /// chosen low-level reader sees the full file from its magic number.
    pub fn read_from(stream: &mut dyn IStream) -> Result<Image> {
        match peek_kind(stream)? {
            Kind::Tiled => {
                log::debug!("facade: detected single-part tiled image");
                Self::from_tiled_input_file(&mut TiledInputFile::new(stream)?)
            }
            Kind::Scanline => {
                log::debug!("facade: detected single-part scanline image");
                Self::from_input_file(&mut scanline::InputFile::new(stream)?)
            }
        }
    }

    /// Read the file AS STORED: a tiled file keeps its tile description and loads
    /// every MIP/RIP level; channels keep their native sampling (a subsampled chroma
    /// channel is its own smaller grid, see [`Image::samples`]). `read_all` then
    /// [`Image::write`] reproduces the storage shape (and, for lossless codecs, the
    /// bytes) — `read` flattens it to level 0 and, for subsampled files, refuses.
    ///
    /// Deep and multipart files are still [`ExrError::Unsupported`].
    pub fn read_all<P: AsRef<Path>>(path: P) -> Result<Image> {
        Image::read_all_from(&mut open_path(path.as_ref())?)
    }

    /// Stream form of [`Image::read_all`].
    pub fn read_all_from(stream: &mut dyn IStream) -> Result<Image> {
        match peek_kind(stream)? {
            Kind::Scanline => Self::load_scanline(&mut scanline::InputFile::new(stream)?, true),
            Kind::Tiled => {
                let mut input = TiledInputFile::new(stream)?;
                let header = input.header().clone();
                let td = header.tile_description()?;
                let dw = header.data_window()?;
                let mut image = None;
                for (lx, ly) in level_grid(&td, dw)? {
                    let planes = Self::load_level(&mut input, &header, lx, ly)?;
                    match image.as_mut() {
                        None => image = Some(Self::from_header(&header, dw, planes)?),
                        Some(img) => {
                            let img: &mut Image = img;
                            img.levels.insert((lx, ly), planes);
                        }
                    }
                }
                let mut image = image.ok_or_else(|| ExrError::Logic("tiled file with no levels".into()))?;
                image.tiling = Some(td);
                Ok(image)
            }
        }
    }

    /// Read ONE level `(lx, ly)` as a plain (untiled) image whose data window is that
    /// level's window. A scanline file has only `(0, 0)`. Channels keep native
    /// sampling like [`Image::read_all`]; the display window and attributes are the
    /// file's.
    pub fn read_level<P: AsRef<Path>>(path: P, lx: i32, ly: i32) -> Result<Image> {
        Image::read_level_from(&mut open_path(path.as_ref())?, lx, ly)
    }

    /// Stream form of [`Image::read_level`].
    pub fn read_level_from(stream: &mut dyn IStream, lx: i32, ly: i32) -> Result<Image> {
        match peek_kind(stream)? {
            Kind::Scanline if (lx, ly) == (0, 0) => Self::load_scanline(&mut scanline::InputFile::new(stream)?, true),
            Kind::Scanline => Err(ExrError::Arg(format!("Level ({lx}, {ly}) does not exist in a scanline file."))),
            Kind::Tiled => {
                let mut input = TiledInputFile::new(stream)?;
                let header = input.header().clone();
                let td = header.tile_description()?;
                let dw = header.data_window()?;
                if !level_grid(&td, dw)?.contains(&(lx, ly)) {
                    return Err(ExrError::Arg(format!("Level ({lx}, {ly}) does not exist in this file's tiling.")));
                }
                let planes = Self::load_level(&mut input, &header, lx, ly)?;
                Self::from_header(&header, level_window(&td, dw, lx, ly)?, planes)
            }
        }
    }

    /// Read a file's header and dimensions WITHOUT decoding any pixel data —
    /// a true "header only" entry point (py-parity2 exr-core need (b)):
    /// `openexr-py`'s `header_only=True` used to call the full [`Image::read`]
    /// and simply discard the pixel result, so every chunk was decoded
    /// anyway. This opens the same low-level reader [`Image::read_from`]
    /// would (so the header + chunk offset table are parsed exactly the same
    /// way, including the same escape-hatch/validation errors), but never
    /// calls `read_pixels`/`read_tiles` — the returned [`Image`]'s channel
    /// data is correctly TYPED and SIZED (so [`Image::width`]/
    /// [`Image::height`]/[`Image::channels`]/[`Image::attributes`] are all
    /// genuine), but every sample is `0`, never real pixel content.
    ///
    /// # Errors
    /// Same as [`Image::read_from`] (bad magic, the deep/multipart escape
    /// hatch, a subsampled channel, header validation) — this is the same
    /// header parse, just stopped one step earlier.
    pub fn read_header_only<P: AsRef<Path>>(path: P) -> Result<Image> {
        Image::read_header_only_from(&mut open_path(path.as_ref())?)
    }

    /// Stream form of [`Image::read_header_only`].
    pub fn read_header_only_from(stream: &mut dyn IStream) -> Result<Image> {
        // Opening the low-level reader is what parses the header AND the
        // chunk offset table (with reconstruction if needed) — exactly the
        // "header(s) + offsets only, no chunk decode" the ask calls for.
        let header = match peek_kind(stream)? {
            Kind::Tiled => {
                log::debug!("facade: header-only read, single-part tiled image");
                TiledInputFile::new(stream)?.header().clone()
            }
            Kind::Scanline => {
                log::debug!("facade: header-only read, single-part scanline image");
                scanline::InputFile::new(stream)?.header().clone()
            }
        };

        let dw = header.data_window()?;
        let plan = ReadPlan::new(&header, dw, false)?;
        let zeros = vec![0u8; plan.total];
        let planes = plan.into_planes(&zeros)?;
        Self::from_header(&header, dw, planes)
    }

    /// Read an already-opened single-part **scanline** reader into an [`Image`].
    ///
    /// Factored out of [`Image::read`] so callers that already hold an
    /// [`scanline::InputFile`] — in particular a multipart PART obtained via
    /// `MultiPartInputFile::input_part` — can turn it into the flat `Image` model
    /// without re-opening the stream. Single point of responsibility for
    /// "scanline reader → Image". Subsampled channels are rejected
    /// ([`Image::read_all`] is the native-sampling reader).
    pub fn from_input_file<R: scanline::GenericInputReader>(input: &mut R) -> Result<Image> {
        Self::load_scanline(input, false)
    }

    /// Read an already-opened single-part **tiled** reader (level 0) into an
    /// [`Image`] — the tiled counterpart of [`Image::from_input_file`], usable on
    /// a multipart tiled part from `MultiPartInputFile::tiled_input_part`.
    pub fn from_tiled_input_file<R: GenericTiledReader>(input: &mut R) -> Result<Image> {
        let header = input.header().clone();
        let planes = Self::load_level(input, &header, 0, 0)?;
        Self::from_header(&header, header.data_window()?, planes)
    }

    /// Scanline reader → Image; `native` keeps subsampled channels at their sampling.
    fn load_scanline<R: scanline::GenericInputReader>(input: &mut R, native: bool) -> Result<Image> {
        let header = input.header().clone();
        let dw = header.data_window()?;
        let plan = ReadPlan::new(&header, dw, native)?;
        let mut buffer = vec![0u8; plan.total];
        input.set_frame_buffer(&plan.frame_buffer()?)?;
        input.read_pixels(dw.min.y, dw.max.y, &mut buffer)?;
        Self::from_header(&header, dw, plan.into_planes(&buffer)?)
    }

    /// The planes of tiled level `(lx, ly)` (every tile of it).
    fn load_level<R: GenericTiledReader>(input: &mut R, header: &Header, lx: i32, ly: i32) -> Result<Planes> {
        let window = level_window(&header.tile_description()?, header.data_window()?, lx, ly)?;
        let plan = ReadPlan::new(header, window, false)?;
        let mut buffer = vec![0u8; plan.total];
        input.set_frame_buffer(&plan.frame_buffer()?)?;
        let (nx, ny) = (input.num_x_tiles(lx)?, input.num_y_tiles(ly)?);
        input.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut buffer)?;
        plan.into_planes(&buffer)
    }

    /// An untiled `Image` over `window` holding `planes`, with the header's
    /// compression, line order, display window and carried attributes (so a
    /// read -> write round trip, and `exr-rs convert`, preserves them).
    fn from_header(header: &Header, window: Box2i, planes: Planes) -> Result<Image> {
        Ok(Image {
            data_window: window,
            display_window: header.display_window()?,
            compression: header.compression()?,
            line_order: header.line_order()?,
            tiling: None,
            base: planes,
            levels: BTreeMap::new(),
            attributes: ImageAttributes::from_header(header),
        })
    }

    // -----------------------------------------------------------------------
    // Writing
    // -----------------------------------------------------------------------

    /// Write this image as a single-part file with the chosen compression:
    /// **scanline**, or **tiled** (every level) when a tiling is set
    /// ([`Image::set_tiling`]).
    ///
    /// This is the common-case writer; it reuses the verified
    /// [`scanline::OutputFile`] / [`TiledOutputFile`] so the bytes are identical to
    /// the low-level path.
    ///
    /// # Errors
    /// * Returns [`ExrError::Arg`] if the image has no channels, or a MIPMAP/RIPMAP
    ///   tiling is missing a level's pixels.
    /// * Propagates header / writer errors (e.g. a subsampled channel in a tiled file).
    pub fn write<P: AsRef<Path>>(&self, path: P, compression: Compression) -> Result<()> {
        self.write_with_attrs(path, compression, Vec::new())
    }

    /// Write this image to an arbitrary output stream — the stream form of
    /// [`Image::write`].
    pub fn write_to(&self, stream: &mut dyn OStream, compression: Compression) -> Result<()> {
        self.write_to_with_attrs(stream, compression, Vec::new())
    }

    /// Like [`Image::write`], but inserts `extra` header attributes **on top of**
    /// the ones the image already carries ([`Image::attributes`]).
    ///
    /// `extra` wins on a name clash, so a caller that manages a full header of
    /// its own (the Python binding, `cryptomatte`) keeps overriding behaviour;
    /// everything it does not mention still round-trips. This is not a second
    /// metadata path — it is [`Image::write`] plus a caller-supplied overlay.
    pub fn write_with_attrs<P: AsRef<Path>>(
        &self,
        path: P,
        compression: Compression,
        extra: Vec<(Name, Box<dyn Attribute>)>,
    ) -> Result<()> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
        crate::io::write_file(path_str, |stream| self.write_to_with_attrs(stream, compression, extra))
    }

    /// Stream form of [`Image::write_with_attrs`] — and the **one** write path in the
    /// facade: [`Image::write_to`] (and therefore [`Image::write`] and `exr-rs
    /// convert`) is this function with an empty overlay, so metadata handling can not
    /// differ between them.
    pub fn write_to_with_attrs(
        &self,
        stream: &mut dyn OStream,
        compression: Compression,
        extra: Vec<(Name, Box<dyn Attribute>)>,
    ) -> Result<()> {
        if self.base.chans.is_empty() {
            return Err(ExrError::Arg("Cannot write an Image with no channels.".into()));
        }

        // Header: windows + compression + line order + channels (+ tiling), then the
        // carried attributes, then the caller's overlay (which wins).
        let mut header = self.build_header(compression)?;
        for (name, attr) in extra {
            header.insert_boxed(name, attr)?;
        }

        // py-parity2 exr-core need (a): a caller-supplied overlay carrying a
        // `tiles` attribute (e.g. the Python binding's `header["tiles"] =
        // TileDescription(...)`, `header["type"] = OpenEXR.tiledimage`) means
        // C++ would open this header with `TiledOutputFile`, not `OutputFile`.
        // `has_tile_description()` is the same signal `Header::write_to` uses to
        // decide the TILED version flag, so this dispatch and the bytes it produces
        // agree by construction.
        if header.has_tile_description() {
            return self.write_tiled_to(stream, header);
        }
        if !self.levels.is_empty() {
            return Err(ExrError::Arg("Image has MIP/RIP levels but no tiling to write them with.".into()));
        }

        // User-meaningful milestone: writing an image (info).
        log::info!(
            "writing scanline image: {}x{} channels={} compression={compression:?}",
            self.width(),
            self.height(),
            self.base.chans.len()
        );

        // `write_buf` re-checks every plane against the window: `with_channel` is
        // deliberately unchecked, `insert_channel`/`insert_sub` are the checked path.
        let (buffer, fb) = self.base.write_buf(self.data_window)?;
        let mut output = scanline::OutputFile::new(stream, &header)?;
        output.set_frame_buffer(&fb)?;
        output.write_pixels(&buffer, self.height() as i32)?;
        Ok(())
    }

    /// Write this image as a single-part **tiled** file: every level of `header`'s
    /// tile description, in file order, each as ONE `write_tiles` over the whole level
    /// with the level's own frame buffer — the same gather `Planes::write_buf` the
    /// scanline path uses. ONE_LEVEL writes level 0 only.
    ///
    /// Level order changes memory, not bytes: the tiled writer emits tiles in file order
    /// whatever order they arrive in, buffering early ones, so levels are fed in file order
    /// to keep that buffer empty.
    ///
    /// Every level is validated BEFORE the first byte is written, so a missing level
    /// never leaves a file that claims a pyramid it does not hold.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] for a missing/extra/mis-sized level or a level mode the
    ///   inserted levels do not match.
    /// * Propagates header / writer errors (subsampled channels fail the tiled header
    ///   sanity check, like C++).
    fn write_tiled_to(&self, stream: &mut dyn OStream, header: Header) -> Result<()> {
        let td = header.tile_description()?;
        let grid = level_grid(&td, self.data_window)?;
        if let Some(&(lx, ly)) = self.levels.keys().find(|k| !grid.contains(k)) {
            return Err(ExrError::Arg(format!(
                "Image has level ({lx}, {ly}) which the {:?} tiling does not have.",
                td.mode
            )));
        }
        // The overlay's tiling may differ from `self.tiling`, so validate against `td`.
        for &(lx, ly) in grid.iter().filter(|&&l| l != (0, 0)) {
            check_level(&td, self.data_window, &self.base, lx, ly, self.planes_at(lx, ly)?)?;
        }

        log::info!(
            "writing tiled image: {}x{} tile={}x{} mode={:?} levels={} channels={} compression={:?}",
            self.width(),
            self.height(),
            td.x_size,
            td.y_size,
            td.mode,
            grid.len(),
            self.base.chans.len(),
            header.compression()?
        );

        let mut output = TiledOutputFile::new(stream, &header)?;
        for (lx, ly) in grid {
            let (buffer, fb) = self.planes_at(lx, ly)?.write_buf(level_window(&td, self.data_window, lx, ly)?)?;
            output.set_frame_buffer(&fb)?;
            let (nx, ny) = (output.num_x_tiles(lx)?, output.num_y_tiles(ly)?);
            output.write_tiles(&buffer, 0, nx - 1, 0, ny - 1, lx, ly)?;
        }
        Ok(())
    }

    /// Build the EXR [`Header`] for this image (windows, compression, line
    /// order, channels in name order with their sampling, the tiling if any), then
    /// apply the carried [`ImageAttributes`].
    ///
    /// The carried attributes are applied **last** so the source file's
    /// `pixelAspectRatio` / `screenWindow*` replace [`Header::new`]'s defaults;
    /// they can never clash with the derived ones because
    /// [`ImageAttributes::DERIVED`] is excluded on both capture and insert.
    ///
    /// **No `type` attribute.** A single-part file does not need one, and C++
    /// only *fixes* a type the caller's header already carried
    /// (`ImfOutputFile.cpp:775` `if (_data->header.hasType ()) setType
    /// (SCANLINEIMAGE)`; our low-level writer mirrors that at
    /// `scanline/output.rs:360`). The facade builds its header from
    /// [`Header::new`], which has no type — so writing one unconditionally made
    /// every facade file carry a `type:string:13` attribute real C++ would not
    /// write. The multipart writer adds it explicitly, where C++ *requires* it
    /// (`ImfMultiPartOutputFile.cpp:117`).
    fn build_header(&self, compression: Compression) -> Result<Header> {
        let mut header = Header::new();
        *header.data_window_mut()? = self.data_window;
        *header.display_window_mut()? = self.display_window;
        *header.compression_mut()? = compression;
        // RANDOM_Y is legal only for tiled files (see `scanline_line_order`).
        *header.line_order_mut()? = if self.tiling.is_some() { self.line_order } else { self.scanline_line_order() };
        *header.channels_mut()? = self.base.channel_list()?;
        if let Some(td) = self.tiling {
            header.set_tile_description(td)?;
        }
        self.attributes.apply_to(&mut header)?;
        Ok(header)
    }

    /// The line order to write, given that the facade always emits a
    /// single-part **scanline** file.
    ///
    /// `RANDOM_Y` is legal only for a tiled image: C++ `Header::sanityCheck`
    /// accepts `INCREASING_Y | DECREASING_Y | RANDOM_Y` in the tiled branch and
    /// only `INCREASING_Y | DECREASING_Y` otherwise
    /// (`ImfHeader.cpp:893` `sanityCheck`; our 1:1 port is
    /// `header.rs::sanity_check`). Reading a tiled `RANDOM_Y` file through this
    /// facade already drops the tiling — the pixels come back as one flat
    /// raster and go out as scanlines — so the tile-only line order has no
    /// meaning in the file being written, and carrying it verbatim produced a
    /// header the writer (correctly) rejected with "Invalid line order in image
    /// header".
    ///
    /// It is normalised to `INCREASING_Y`, which is the order the scanline
    /// writer actually emits. The sanity check itself is untouched: an
    /// `Image` whose `line_order` is genuinely invalid still fails there.
    fn scanline_line_order(&self) -> LineOrder {
        match self.line_order {
            LineOrder::RandomY => {
                // Metadata loss the caller may care about; surfaced once per
                // write rather than silently.
                log::warn!(
                    "api::Image: lineOrder RANDOM_Y is tiled-only; writing a \
                     scanline file as INCREASING_Y"
                );
                LineOrder::IncreasingY
            }
            other => other,
        }
    }
}

/// Storage kind of a single-part, non-deep file (the only kinds the facade reads).
enum Kind {
    Scanline,
    Tiled,
}

/// Classify a file from its magic + version word and rewind, so the chosen
/// low-level reader sees it from byte 0. Deep and multipart files are the
/// structured [`ExrError::Unsupported`] escape hatches.
fn peek_kind(stream: &mut dyn IStream) -> Result<Kind> {
    stream.seekg(0)?;
    let magic = xdr::read_i32(stream)?;
    if magic as u32 != version::MAGIC {
        return Err(ExrError::Input("File is not an OpenEXR file (bad magic number).".into()));
    }
    let flags = version::get_flags(xdr::read_i32(stream)? as u32);
    stream.seekg(0)?;
    if version::is_multi_part(flags) {
        return Err(ExrError::Unsupported(Unsupported::Multipart));
    }
    if version::is_non_image(flags) {
        return Err(ExrError::Unsupported(Unsupported::Deep));
    }
    Ok(if version::is_tiled(flags) { Kind::Tiled } else { Kind::Scanline })
}

/// Open `path` for reading (UTF-8 paths only, like every facade entry point).
fn open_path(path: &Path) -> Result<StdIfStream> {
    let path_str = path
        .to_str()
        .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
    StdIfStream::open(path_str)
}

// ---------------------------------------------------------------------------
// Plane codecs — the low-level FrameBuffer is **native-endian** (the gather /
// scatter in `misc.rs` converts native ↔ XDR at the wire boundary, storing
// `to_ne_bytes` into the frame buffer and reading `from_ne_bytes` out of it).
// The facade therefore lays its planes out in native byte order to match what
// `OutputFile::set_frame_buffer` consumes and `InputFile::read_pixels`
// produces — the single source of truth for endianness.
// ---------------------------------------------------------------------------

/// Decode a channel plane from the **native-endian** bytes the low-level
/// readers scatter into the frame buffer.
fn decode_plane_native(type_: PixelType, plane: &[u8], pixels: usize) -> Result<ChannelData> {
    // Exact-size `as_chunks::<N>().0.iter().map(..).collect()` instead
    // of a per-sample `Vec::with_capacity` + bounds-checked `.get()` + push
    // loop — the bounds check moves up front (once, on the whole plane)
    // instead of once per sample, matching the sibling `from_xdr_bytes`'s own
    // shape. `collect()` on an `ExactSizeIterator` (slice iterators + `map`
    // both preserve the exact length) allocates the destination `Vec` ONCE,
    // at its final size, with no reallocation as it fills — the same
    // guarantee `Vec::with_capacity(pixels)` gave, just without a manual
    // per-element bounds check to get there.
    // `pixel_type_size` already errors on `NumPixelTypes`, so by the time
    // `sz` is in hand `type_` is provably one of the three real variants.
    let sz = crate::misc::pixel_type_size(type_)?;
    let need = pixels
        .checked_mul(sz)
        .ok_or_else(|| ExrError::Arg("plane byte count overflows.".into()))?;
    let plane = plane.get(..need).ok_or_else(|| ExrError::Logic("plane truncated.".into()))?;
    Ok(match type_ {
        PixelType::Half => ChannelData::Half(
            plane.as_chunks::<2>().0.iter().map(|b| Half::from_bits(u16::from_ne_bytes(*b))).collect(),
        ),
        PixelType::Float => ChannelData::Float(
            plane.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes(*b)).collect(),
        ),
        PixelType::Uint => ChannelData::Uint(
            plane.as_chunks::<4>().0.iter().map(|b| u32::from_ne_bytes(*b)).collect(),
        ),
        // Unreachable in practice (see above), but no `unreachable!()` in
        // library code: fail the same way `pixel_type_size` already would.
        PixelType::NumPixelTypes => return Err(ExrError::Arg("Invalid pixel type in channel.".into())),
    })
}

#[cfg(test)]
mod tests;
