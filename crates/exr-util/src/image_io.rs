//! Port of `ImfImageIO.{h,cpp}`: load or save a whole flat or deep image of any storage kind, plus the
//! plumbing [`crate::flat_image_io`] and [`crate::deep_image_io`] share (the C++ duplicates it in
//! both files).
//!
//! `loadImage` returns the `Image*` of whichever dynamic type the file needs; here that is
//! [`AnyImage`]. Where the C++ fills a caller's `Header&`, the loaders return the header.

use std::path::Path;

use exr_core::attr::{Channel, Compression, LevelMode, Name, PixelType, TileDescription};
use exr_core::container::is_open_exr_file_stream;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{is_tiled_type, Header};
use exr_core::io::StdIfStream;
use exr_core::multipart::MultiPartInputFile;
use exr_core::{ExrError, Result};
use imath_rs::Box2i;

use crate::deep_image::DeepImage;
use crate::deep_image_io::{load_deep_scanline_image, load_deep_tiled_image, save_deep_image};
use crate::flat_image::FlatImage;
use crate::flat_image_io::{load_flat_scanline_image, load_flat_tiled_image, save_flat_image};
use crate::image::Image;
use crate::image_data_window::{data_window_for_file, DataWindowSource};
use crate::image_level::ImageLevel;

/// Bytes the strip buffers of the file readers and writers aim for: images are moved between the
/// file and the typed channels a few megabytes at a time, never as one whole-image byte copy.
const STRIP_BYTES: usize = 16 << 20;

/// A loaded image of either kind (the C++ `Image*` with its dynamic type).
pub enum AnyImage {
    /// A `FlatImage`.
    Flat(FlatImage),
    /// A `DeepImage`.
    Deep(DeepImage),
}

/// `saveImage`: save `img` in an OpenEXR file. The file is tiled if the image has more than one
/// level or `hdr` has a tile description, deep if the image is deep. With `hdr` the channel list is
/// replaced by the image's and the level mode and rounding mode of the tile description by the
/// image's; without it the file gets a default header whose display window is the data window.
pub fn save_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &AnyImage, dws: DataWindowSource) -> Result<()> {
    match img {
        AnyImage::Flat(f) => save_flat_image(path, hdr, f, dws),
        AnyImage::Deep(d) => save_deep_image(path, hdr, d, dws),
    }
}

/// `loadImage`: load a single-part flat or deep, scanline or tiled OpenEXR file; also returns the
/// header of the file.
///
/// # Errors
/// `Arg` if the file is not an OpenEXR file or has several parts.
pub fn load_image(path: impl AsRef<Path>) -> Result<(AnyImage, Header)> {
    let path = path.as_ref();
    let name = path.display();
    let mut stream = StdIfStream::open(path_str(path)?)?;
    let probe = is_open_exr_file_stream(&mut stream);
    if !probe.is_exr {
        return Err(ExrError::Arg(format!("Cannot load image file {name}.  The file is not an OpenEXR file.")));
    }
    if probe.is_multi_part {
        return Err(ExrError::Arg(format!("Cannot load image file {name}.  Multi-part file loading is not supported.")));
    }
    let tiled = is_tiled_file(path)?;
    if probe.is_deep {
        let mut img = DeepImage::empty()?;
        let hdr = if tiled { load_deep_tiled_image(path, &mut img)? } else { load_deep_scanline_image(path, &mut img)? };
        Ok((AnyImage::Deep(img), hdr))
    } else {
        let mut img = FlatImage::empty()?;
        let hdr = if tiled { load_flat_tiled_image(path, &mut img)? } else { load_flat_scanline_image(path, &mut img)? };
        Ok((AnyImage::Flat(img), hdr))
    }
}

/// Whether the first part is a tiled one. The C++ notes that the version word's tiled flag is
/// unreliable (`ImfImageIO.cpp:76`) and looks at the `type` attribute of the first header instead.
pub(crate) fn is_tiled_file(path: &Path) -> Result<bool> {
    let mpi = MultiPartInputFile::new_owned(Box::new(StdIfStream::open(path_str(path)?)?))?;
    if mpi.parts() == 0 {
        return Ok(false);
    }
    let h = mpi.header(0)?;
    Ok(h.has_type() && is_tiled_type(h.type_()?))
}

/// Check the probe flags `loadFlatImage` / `loadDeepImage` test before dispatching.
pub(crate) fn probe_kind(path: &Path, want_deep: bool) -> Result<bool> {
    let name = path.display();
    let mut stream = StdIfStream::open(path_str(path)?)?;
    let probe = is_open_exr_file_stream(&mut stream);
    if !probe.is_exr {
        return Err(ExrError::Arg(format!("Cannot load image file {name}.  The file is not an OpenEXR file.")));
    }
    if probe.is_multi_part {
        return Err(ExrError::Arg(format!("Cannot load image file {name}.  Multi-part file loading is not supported.")));
    }
    match (probe.is_deep, want_deep) {
        (true, false) => Err(ExrError::Arg(format!("Cannot load deep image file {name} as a flat image."))),
        (false, true) => Err(ExrError::Arg(format!("Cannot load flat image file {name} as a deep image."))),
        _ => is_tiled_file(path),
    }
}

/// The path as the `&str` the exr-core file streams take.
pub(crate) fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| ExrError::Arg(format!("The path {} is not valid UTF-8.", p.display())))
}

/// Whether `save*Image` without an explicit choice writes a tiled file.
pub(crate) fn save_tiled<L: ImageLevel>(hdr: Option<&Header>, img: &Image<L>) -> bool {
    img.level_mode() != LevelMode::OneLevel || hdr.is_some_and(Header::has_tile_description)
}

/// The header of the file that saves `img`: the attributes of `hdr` (or of a default header with
/// the image's data window as display window) except `dataWindow`, `tiles` and `channels`; for a
/// tiled file the tile size of `hdr` (64 x 64 by default) with the image's level mode; the data
/// window `dws` selects; `ZIPS` compression for deep files (the C++ note at
/// `ImfDeepImageIO.cpp:155`: other deep codecs save but do not load); and the image's channels.
pub(crate) fn file_header<L: ImageLevel>(
    hdr: Option<&Header>,
    img: &Image<L>,
    dws: DataWindowSource,
    tiled: bool,
    deep: bool,
) -> Result<Header> {
    let default;
    let src = match hdr {
        Some(h) => h,
        None => {
            let mut h = Header::new();
            *h.display_window_mut()? = *img.data_window();
            default = h;
            &default
        }
    };
    let mut new = Header::new();
    for (name, attr) in src.iter() {
        if *name == "dataWindow" || *name == "tiles" || *name == "channels" {
            continue;
        }
        new.insert_boxed(name.clone(), attr.clone_attr())?;
    }
    if tiled {
        let (x_size, y_size) = match src.has_tile_description() {
            true => {
                let td = src.tile_description()?;
                (td.x_size, td.y_size)
            }
            false => (64, 64),
        };
        new.set_tile_description(TileDescription {
            x_size,
            y_size,
            mode: img.level_mode(),
            rounding_mode: img.level_rounding_mode(),
        })?;
    }
    *new.data_window_mut()? = data_window_for_file(src, img, dws)?;
    if deep {
        *new.compression_mut()? = Compression::Zips;
    }
    for (name, ch) in img.channels() {
        new.channels_mut()?.insert(name.clone(), *ch)?;
    }
    Ok(new)
}

/// What the loaders hand back: a copy of the file header. The C++ inserts into the caller's (by
/// default empty) header; scanline loads leave out the `tiles` attribute (`ImfFlatImageIO.cpp:185`).
pub(crate) fn copy_header(file: &Header, skip_tiles: bool) -> Result<Header> {
    let mut h = Header::new();
    for (name, attr) in file.iter() {
        if !(skip_tiles && *name == "tiles") {
            h.insert_boxed(name.clone(), attr.clone_attr())?;
        }
    }
    Ok(h)
}

/// The levels of an image in file order: `ONE_LEVEL` `(0,0)`; `MIPMAP` `(l,l)`; `RIPMAP` row by row.
pub(crate) fn level_order(mode: LevelMode, nx: i32, ny: i32) -> Vec<(i32, i32)> {
    match mode {
        LevelMode::MipmapLevels => (0..nx).map(|l| (l, l)).collect(),
        LevelMode::RipmapLevels => (0..ny).flat_map(|y| (0..nx).map(move |x| (x, y))).collect(),
        _ => vec![(0, 0)],
    }
}

/// The window of level `(lx, ly)` of the file that saves `img`: the file's own data window `fw`
/// for a ONE_LEVEL image (which `DataWindowSource::Header` may have cropped), else the image's.
pub(crate) fn level_window<L: ImageLevel>(img: &Image<L>, fw: &Box2i, lx: i32, ly: i32) -> Result<Box2i> {
    match img.level_mode() {
        LevelMode::OneLevel => Ok(*fw),
        _ => img.data_window_for_level_at(lx, ly).copied(),
    }
}

/// `ceil (a / b)` for `b > 0`.
fn div_ceil(a: i64, b: i64) -> i64 {
    a.div_euclid(b) + i64::from(a.rem_euclid(b) != 0)
}

/// How the samples of one channel sit in a strip buffer: `rows` rows of `cols` samples starting at
/// byte `off`, for the window pixels that lie on the channel's sampling grid. Part of the strip
/// streaming API (see [`Strip`]).
pub struct Plane {
    /// Sample type.
    pub type_: PixelType,
    /// Sampling rates of the channel.
    pub xs: i32,
    /// See `xs`.
    pub ys: i32,
    /// Bytes per sample.
    pub size: usize,
    /// Samples per row in the buffer, and rows.
    pub cols: usize,
    /// See `cols`.
    pub rows: usize,
    /// Byte offset of the plane in the buffer.
    pub off: usize,
    /// Size of the plane in bytes (`cols * rows * size`).
    pub len: usize,
    /// Grid coordinates (`x / xSampling`, `y / ySampling`) of the first sample of the plane.
    pub gx: i64,
    /// See `gx`.
    pub gy: i64,
}

/// The layout of one window of a level in a strip buffer: a [`Plane`] per channel, in channel order.
/// Strips are how pixels move between files and typed channels a few megabytes at a time (16 MiB,
/// never a whole-image byte copy); the window is a contiguous range of rows of the level.
///
/// Used by [`crate::flat_image_io`] and by any tool that streams an EXR through exr-core's
/// readers / writers: pick `rows` with [`strip_rows`] and the chunk alignment with [`scanline_unit`],
/// walk the rows with [`strip_ranges`], build a `Strip` per range, hand
/// [`Strip::frame_buffer`] to the reader or writer and the byte buffer (`bytes` long, plane `p` at
/// `p.off`) as the pixel buffer; samples are native-endian.
pub struct Strip {
    /// One plane per channel, in the order the channels were given.
    pub planes: Vec<(Name, Plane)>,
    /// Size of the whole buffer in bytes.
    pub bytes: usize,
}

impl Strip {
    /// Layout of the window `[x0, x1] x [y0, y1]` for `channels` (empty if `x1 < x0` or `y1 < y0`); the
    /// window origin must lie on every channel's sampling grid for the frame buffer to be exact.
    pub fn new<'a>(channels: impl Iterator<Item = (&'a Name, Channel)>, (x0, x1): (i32, i32), (y0, y1): (i32, i32)) -> Result<Self> {
        let too_big = || ExrError::Arg("An image strip is too large to buffer.".into());
        let mut planes = Vec::new();
        let mut bytes = 0usize;
        for (name, ch) in channels {
            let (xs, ys) = (ch.x_sampling.max(1) as i64, ch.y_sampling.max(1) as i64);
            let (gx, gy) = (div_ceil(x0 as i64, xs), div_ceil(y0 as i64, ys));
            let count = |hi: i32, g: i64, s: i64| usize::try_from((hi as i64).div_euclid(s) - g + 1).unwrap_or(0);
            let (cols, rows) = (count(x1, gx, xs), count(y1, gy, ys));
            let size = match ch.type_ {
                PixelType::Half => 2,
                PixelType::Float | PixelType::Uint => 4,
                PixelType::NumPixelTypes => return Err(ExrError::Arg("Unknown pixel type.".into())),
            };
            let len = cols.checked_mul(rows).and_then(|n| n.checked_mul(size)).ok_or_else(too_big)?;
            planes.push((name.clone(), Plane { type_: ch.type_, xs: ch.x_sampling, ys: ch.y_sampling, size, cols, rows, off: bytes, len, gx, gy }));
            bytes = bytes.checked_add(len).ok_or_else(too_big)?;
        }
        Ok(Self { planes, bytes })
    }

    /// The frame buffer that addresses this layout by absolute pixel coordinates; slice names are the
    /// channel names.
    pub fn frame_buffer(&self) -> Result<FrameBuffer> {
        self.frame_buffer_with(Name::clone)
    }

    /// Like [`Strip::frame_buffer`], but the slice of channel `n` is named `key (n)`: the direction is
    /// image channel name -> file channel name (for example `right.R` -> `R` when a tool reads one view
    /// of a multi-view file into an image whose channels carry the view prefix). The planes keep the
    /// image channel names, so [`crate::flat_image_io::pack`] / [`crate::flat_image_io::unpack`] work
    /// unchanged.
    pub fn frame_buffer_with(&self, key: impl Fn(&Name) -> Name) -> Result<FrameBuffer> {
        let mut fb = FrameBuffer::new();
        for (name, p) in &self.planes {
            let y_stride = p.cols * p.size;
            let base = p.off as i64 - p.gx * p.size as i64 - p.gy * y_stride as i64;
            fb.insert(key(name), Slice::new(p.type_, base, p.size, y_stride, p.xs, p.ys, 0.0, false, false)?)?;
        }
        Ok(fb)
    }
}

/// Rows per strip for a window of `width` pixels: a multiple of `unit` (the alignment of the file's
/// chunks) that keeps the buffer near 16 MiB.
pub fn strip_rows<'a>(channels: impl Iterator<Item = (&'a Name, Channel)>, width: i64, unit: i64) -> i64 {
    let per_row: i64 = channels.map(|(_, c)| width / c.x_sampling.max(1) as i64 * if c.type_ == PixelType::Half { 2 } else { 4 }).sum();
    unit * (STRIP_BYTES as i64 / (unit * per_row.max(1))).max(1)
}

/// Alignment for strips of a scanline file: a whole number of chunks and of every channel's y
/// sampling, so no strip starts inside a chunk or off a channel's sampling grid.
pub fn scanline_unit<'a>(channels: impl Iterator<Item = (&'a Name, Channel)>, comp: Compression) -> i64 {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    channels.fold(comp.num_scanlines().max(1) as i64, |u, (_, c)| {
        let s = c.y_sampling.max(1) as i64;
        u / gcd(u, s) * s
    })
}

/// Row ranges `[y0, y1]` of `[min_y, max_y]` in strips of `rows`, in the order a file of the given
/// line order is written (or read).
pub fn strip_ranges(min_y: i32, max_y: i32, rows: i64, unit: i64, decreasing: bool) -> Vec<(i32, i32)> {
    let (lo, hi) = (min_y as i64, max_y as i64);
    let mut out = Vec::new();
    if decreasing {
        let mut top = hi;
        while top >= lo {
            // Start on a chunk boundary so every strip but the first is chunk-aligned.
            let block = lo + (top - lo) / unit * unit;
            let y0 = (block - (rows - unit)).max(lo);
            out.push((y0 as i32, top as i32));
            top = y0 - 1;
        }
    } else {
        let mut y = lo;
        while y <= hi {
            out.push((y as i32, (y + rows - 1).min(hi) as i32));
            y += rows;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_come_in_file_order() {
        assert_eq!(level_order(LevelMode::OneLevel, 1, 1), [(0, 0)]);
        assert_eq!(level_order(LevelMode::MipmapLevels, 3, 3), [(0, 0), (1, 1), (2, 2)]);
        // RIPMAP: one row of x levels after the other, the order the C++ savers write (and the tiled
        // writers then need no buffering for).
        assert_eq!(level_order(LevelMode::RipmapLevels, 2, 3), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn strips_cover_the_rows_once_in_writing_order() {
        assert_eq!(strip_ranges(-3, 6, 4, 2, false), [(-3, 0), (1, 4), (5, 6)]);
        // Bottom-up, every strip but the first starting on a chunk boundary (multiples of 2 from min y).
        let v = strip_ranges(-3, 6, 4, 2, true);
        assert_eq!(v, [(3, 6), (-1, 2), (-3, -2)]);
        for (a, b, dec) in [(0, 99, false), (0, 99, true), (5, 5, true)] {
            let mut rows: Vec<i32> = strip_ranges(a, b, 12, 4, dec).iter().flat_map(|&(y0, y1)| y0..=y1).collect();
            rows.sort_unstable();
            assert_eq!(rows, (a..=b).collect::<Vec<_>>());
        }
    }

    #[test]
    fn strip_layout_addresses_by_absolute_coordinates() {
        let half = Channel::new(PixelType::Half, 1, 1, false);
        let sub = Channel::new(PixelType::Float, 2, 2, false);
        let (a, b) = (Name::from("A"), Name::from("B"));
        // Window x -4..=3, y 2..=5: A has 8 x 4 samples, B (2x2 subsampled) 4 x 2.
        let strip = Strip::new([(&a, half), (&b, sub)].into_iter(), (-4, 3), (2, 5)).unwrap();
        let (pa, pb) = (&strip.planes[0].1, &strip.planes[1].1);
        assert_eq!((pa.cols, pa.rows, pa.off, pa.len), (8, 4, 0, 64));
        assert_eq!((pb.cols, pb.rows, pb.off, pb.len), (4, 2, 64, 32));
        assert_eq!(strip.bytes, 96);
        let fb = strip.frame_buffer().unwrap();
        // Pixel (-4, 2) of A is the first byte of its plane, (3, 5) the last sample.
        let s = fb.get("A").unwrap();
        let at = |x: i64, y: i64| s.base_offset + x * s.x_stride as i64 + y * s.y_stride as i64;
        assert_eq!((at(-4, 2), at(3, 5)), (0, 62));
        let s = fb.get("B").unwrap();
        let at = |x: i64, y: i64| s.base_offset + (x / 2) * s.x_stride as i64 + (y / 2) * s.y_stride as i64;
        assert_eq!((at(-4, 2), at(2, 4)), (64, 64 + 4 * 4 + 12));
    }

    #[test]
    fn strip_windows_off_the_sampling_grid_start_at_the_first_sample() {
        let sub = Channel::new(PixelType::Half, 3, 3, false);
        let a = Name::from("A");
        // x 1..=10 has samples at 3, 6, 9; y 1..=4 at 3.
        let strip = Strip::new([(&a, sub)].into_iter(), (1, 10), (1, 4)).unwrap();
        let p = &strip.planes[0].1;
        assert_eq!((p.cols, p.rows, p.gx, p.gy), (3, 1, 1, 1));
    }
}
