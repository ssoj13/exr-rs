//! Port of `ImfDeepImageIO.{h,cpp}`: load and save deep images, scanline or tiled, every level.
//!
//! The exr-core deep readers and writers take a per-pixel count raster (little-endian `u32`) and one
//! flat sample heap in which every channel owns a region holding its samples pixel after pixel, with
//! no gaps. The image keeps lists with room to grow (see [`crate::sample_count_channel`]), so the
//! level is converted to and from that layout, a level at a time: counts first (they size the
//! heap), then samples. Like the flat I/O the conversion is the only copy; `save*` and `load*`
//! follow the conventions described in [`crate::flat_image_io`].

use std::path::Path;

use exr_core::attr::{LevelMode, LevelRoundingMode, Name, PixelType};
use exr_core::deep::{
    DeepFrameBuffer, DeepScanLineInputFile, DeepScanLineOutputFile, DeepSlice, DeepTiledInputFile,
    DeepTiledOutputFile,
};
use exr_core::frame_buffer::Slice;
use exr_core::header::Header;
use exr_core::io::{write_file, StdIfStream};
use exr_core::{ExrError, Result};
use imath_rs::Box2i;

use crate::deep_image::DeepImage;
use crate::deep_image_level::DeepImageLevel;
use crate::image_channel::{zeroed, ImageChannel};
use crate::image_data_window::DataWindowSource;
use crate::image_io::{copy_header, file_header, level_order, level_window, path_str, probe_kind, save_tiled};
use crate::image_level::ImageLevel;

/// The per-pixel sample counts of a window, as the exr-core readers and writers address them: a
/// raster of little-endian `u32`, one row per window row, whose slice maps the window's corner to
/// offset 0.
///
/// Rows are as wide as the file's data window (level 0), also for the smaller levels of a
/// multi-resolution file: the tiled deep readers and writers order the samples of a level by
/// summing whole data-window rows of counts (`RowPrefixCache` in exr-core), so a level's counts
/// must sit in the top-left of a level-0-wide raster whose remaining entries are zero.
struct Counts {
    bytes: Vec<u8>,
    slice: Slice,
}

impl Counts {
    /// A zeroed raster for `win`, rows `row_w` counts wide (the width of the file's data window).
    fn new(win: &Box2i, row_w: i64) -> Result<Self> {
        let h = win.max.y as i64 - win.min.y as i64 + 1;
        let too_big = || ExrError::Arg("The sample count raster of a deep image is too large.".into());
        let y_stride = usize::try_from(row_w).ok().and_then(|w| w.checked_mul(4)).ok_or_else(too_big)?;
        let n = usize::try_from(h).ok().and_then(|h| h.checked_mul(y_stride)).ok_or_else(too_big)?;
        let base = -(win.min.y as i64 * y_stride as i64 + win.min.x as i64 * 4);
        Ok(Self {
            bytes: zeroed(n)?,
            slice: Slice::new(PixelType::Uint, base, 4, y_stride, 1, 1, 0.0, false, false)?,
        })
    }

    /// The counts of the part `rect = (col0, row0, cols, rows)` of a level that the file window
    /// `win` covers, and their sum.
    fn of_level(lv: &DeepImageLevel, rect: (usize, usize, usize, usize), win: &Box2i, row_w: i64) -> Result<(Self, u64)> {
        let sc = lv.sample_counts();
        let (col0, row0, cols, rows) = rect;
        let w = sc.pixels_per_row() as usize;
        let mut counts = Self::new(win, row_w)?;
        let mut total = 0u64;
        let src = sc.num_samples();
        let row_bytes = usize::try_from(row_w).unwrap_or(0) * 4;
        for (r, dst) in counts.bytes.chunks_mut(row_bytes.max(1)).take(rows).enumerate() {
            let a = (row0 + r) * w + col0;
            let row = src
                .get(a..a + cols)
                .ok_or_else(|| ExrError::Logic("the file window lies outside the image level".into()))?;
            for (d, &n) in dst.as_chunks_mut::<4>().0.iter_mut().zip(row) {
                *d = n.to_le_bytes();
                total += n as u64;
            }
        }
        Ok((counts, total))
    }

    /// Store the raster into a level's count array, `cols` counts per row (the
    /// `SampleCountChannel::Edit` body of loading); the raster rows may be wider than the level.
    fn copy_into(&self, dst: &mut [u32], cols: usize) {
        let row_bytes = self.slice.y_stride.max(4);
        for (dst_row, src_row) in dst.chunks_mut(cols.max(1)).zip(self.bytes.chunks(row_bytes)) {
            for (d, b) in dst_row.iter_mut().zip(src_row.as_chunks::<4>().0) {
                *d = u32::from_le_bytes(*b);
            }
        }
    }
}

/// The flat sample heap: every channel's samples, region after region.
struct Heap {
    bytes: Vec<u8>,
    /// Channel, type, and the byte range of its region.
    regions: Vec<(Name, PixelType, std::ops::Range<usize>)>,
}

impl Heap {
    /// A zeroed heap for `total` samples of each of the channels. With `total == 0` no memory is
    /// allocated and every region is empty.
    fn new(lv: &DeepImageLevel, total: u64) -> Result<Self> {
        let too_big = || ExrError::Arg("The sample heap of a deep image is too large.".into());
        let mut regions = Vec::new();
        let mut off = 0usize;
        for (name, ch) in lv.iter() {
            let size = if ch.pixel_type() == PixelType::Half { 2 } else { 4 };
            let len = usize::try_from(total).ok().and_then(|t| t.checked_mul(size)).ok_or_else(too_big)?;
            let end = off.checked_add(len).ok_or_else(too_big)?;
            regions.push((name.clone(), ch.pixel_type(), off..end));
            off = end;
        }
        Ok(Self { bytes: zeroed(off)?, regions })
    }

    /// The frame buffer of counts and samples: channel `k`'s samples start at its region.
    fn frame_buffer(&self, counts: &Slice) -> Result<DeepFrameBuffer> {
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(counts.clone())?;
        for (name, type_, range) in &self.regions {
            let size = if *type_ == PixelType::Half { 2 } else { 4 };
            let base = i64::try_from(range.start).map_err(|_| ExrError::Arg("sample heap too large".into()))?;
            fb.insert(name.clone(), DeepSlice::new(*type_, base, 8, 0, size, 1, 1, 0.0, false, false)?)?;
        }
        Ok(fb)
    }

    /// Gather the samples of the pixels in `rect` of the level into the heap.
    fn gather(&mut self, lv: &DeepImageLevel, rect: (usize, usize, usize, usize)) -> Result<()> {
        for ((_, ch), (_, _, range)) in lv.iter().zip(&self.regions) {
            let out = self.bytes.get_mut(range.clone()).unwrap_or_default();
            ch.store_lists(lv.sample_counts(), rect, out)?;
        }
        Ok(())
    }

    /// Distribute the heap into the level's sample lists.
    fn scatter(&self, lv: &mut DeepImageLevel) -> Result<()> {
        let (channels, sc) = lv.channels_and_counts();
        for ((_, ch), (_, _, range)) in channels.iter_mut().zip(&self.regions) {
            ch.load_lists(sc, self.bytes.get(range.clone()).unwrap_or_default())?;
        }
        Ok(())
    }
}

/// Width of a window in pixels.
fn width(w: &Box2i) -> i64 {
    w.max.x as i64 - w.min.x as i64 + 1
}

/// The part of a level window `lw` that the file window `fw` covers, as `(col0, row0, cols, rows)`.
fn rect_of(lw: &Box2i, fw: &Box2i) -> (usize, usize, usize, usize) {
    let d = |a: i32, b: i32| usize::try_from(a as i64 - b as i64).unwrap_or(0);
    (d(fw.min.x, lw.min.x), d(fw.min.y, lw.min.y), d(fw.max.x, fw.min.x - 1), d(fw.max.y, fw.min.y - 1))
}

/// `saveDeepImage`: a tiled file if the image has several levels or `hdr` has a tile description,
/// else a scanline file. The compression is always ZIPS (see `image_io::file_header`).
pub fn save_deep_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &DeepImage, dws: DataWindowSource) -> Result<()> {
    if save_tiled(hdr, img) {
        save_deep_tiled_image(path, hdr, img, dws)
    } else {
        save_deep_scanline_image(path, hdr, img, dws)
    }
}

/// `loadDeepImage`: load a single-part deep file into `img`, tiled or not; returns the file's header.
///
/// # Errors
/// `Arg` if the file is not an OpenEXR file, is multipart or flat.
pub fn load_deep_image(path: impl AsRef<Path>, img: &mut DeepImage) -> Result<Header> {
    let path = path.as_ref();
    if probe_kind(path, true)? {
        load_deep_tiled_image(path, img)
    } else {
        load_deep_scanline_image(path, img)
    }
}

/// `saveDeepScanLineImage`: save level 0 of `img` in a deep scanline file.
pub fn save_deep_scanline_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &DeepImage, dws: DataWindowSource) -> Result<()> {
    let new_hdr = file_header(hdr, img, dws, false, true)?;
    let fw = new_hdr.data_window()?;
    let lv = img.level(0)?;
    let rect = rect_of(lv.data_window(), &fw);
    let (counts, total) = Counts::of_level(lv, rect, &fw, width(&fw))?;
    let mut heap = Heap::new(lv, total)?;
    heap.gather(lv, rect)?;
    write_file(path, |stream| {
        let mut out = DeepScanLineOutputFile::new(stream, &new_hdr)?;
        out.set_frame_buffer(&heap.frame_buffer(&counts.slice)?)?;
        out.write_pixels(&counts.bytes, &heap.bytes, rect.3 as i32)
    })
}

/// `loadDeepScanLineImage`: load a deep scanline file as a single-level image; returns the file's
/// header without its `tiles` attribute.
pub fn load_deep_scanline_image(path: impl AsRef<Path>, img: &mut DeepImage) -> Result<Header> {
    let mut inp = DeepScanLineInputFile::open(path.as_ref())?;
    let file_hdr = inp.header().clone();
    img.clear_channels();
    for (name, ch) in file_hdr.channels()?.iter() {
        img.insert_channel(name.clone(), ch)?;
    }
    img.resize_with(&file_hdr.data_window()?, LevelMode::OneLevel, LevelRoundingMode::RoundDown)?;
    let lv = img.level_mut(0)?;
    let win = *lv.data_window();
    let mut counts = Counts::new(&win, width(&win))?;
    inp.set_frame_buffer(&Heap::new(lv, 0)?.frame_buffer(&counts.slice)?)?;
    inp.read_pixel_sample_counts(win.min.y, win.max.y, &mut counts.bytes)?;
    let cols = (win.max.x as i64 - win.min.x as i64 + 1) as usize;
    lv.edit_counts(|c| counts.copy_into(c, cols))?;
    let mut heap = Heap::new(lv, lv.sample_counts().total_samples() as u64)?;
    inp.set_frame_buffer(&heap.frame_buffer(&counts.slice)?)?;
    inp.read_pixels(win.min.y, win.max.y, &counts.bytes, &mut heap.bytes)?;
    heap.scatter(lv)?;
    copy_header(&file_hdr, true)
}

/// Tiles per row and column of a level window.
fn tile_counts(lw: &Box2i, x_size: u32, y_size: u32) -> (i32, i32) {
    let n = |lo: i32, hi: i32, t: u32| ((hi as i64 - lo as i64 + t as i64) / t.max(1) as i64) as i32;
    (n(lw.min.x, lw.max.x, x_size), n(lw.min.y, lw.max.y, y_size))
}

/// `saveDeepTiledImage`: save all levels of `img` in a deep tiled file (64 x 64 tiles unless `hdr`
/// has a tile description).
pub fn save_deep_tiled_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &DeepImage, dws: DataWindowSource) -> Result<()> {
    let new_hdr = file_header(hdr, img, dws, true, true)?;
    let fw = new_hdr.data_window()?;
    let td = new_hdr.tile_description()?;
    write_file(path, |stream| {
        let mut out = DeepTiledOutputFile::new(stream, &new_hdr)?;
        for (lx, ly) in level_order(img.level_mode(), img.num_x_levels(), img.num_y_levels()) {
            let lv = img.level_at(lx, ly)?;
            let lw = level_window(img, &fw, lx, ly)?;
            let rect = rect_of(lv.data_window(), &lw);
            let (counts, total) = Counts::of_level(lv, rect, &lw, width(&fw))?;
            let mut heap = Heap::new(lv, total)?;
            heap.gather(lv, rect)?;
            let (ntx, nty) = tile_counts(&lw, td.x_size, td.y_size);
            out.set_frame_buffer(&heap.frame_buffer(&counts.slice)?)?;
            out.write_tiles(&counts.bytes, &heap.bytes, 0, ntx - 1, 0, nty - 1, lx, ly)?;
        }
        Ok(())
    })
}

/// `loadDeepTiledImage`: load all levels of a deep tiled file; returns the file's header.
pub fn load_deep_tiled_image(path: impl AsRef<Path>, img: &mut DeepImage) -> Result<Header> {
    let mut stream = StdIfStream::open(path_str(path.as_ref())?)?;
    let mut inp = DeepTiledInputFile::new(&mut stream)?;
    let file_hdr = inp.header().clone();
    let td = file_hdr.tile_description()?;
    img.clear_channels();
    for (name, ch) in file_hdr.channels()?.iter() {
        img.insert_channel(name.clone(), ch)?;
    }
    img.resize_with(&file_hdr.data_window()?, td.mode, td.rounding_mode)?;
    let row_w = width(img.data_window());
    for (lx, ly) in level_order(img.level_mode(), img.num_x_levels(), img.num_y_levels()) {
        let lv = img.level_at_mut(lx, ly)?;
        let win = *lv.data_window();
        let (ntx, nty) = (inp.num_x_tiles(lx)?, inp.num_y_tiles(ly)?);
        let mut counts = Counts::new(&win, row_w)?;
        inp.set_frame_buffer(&Heap::new(lv, 0)?.frame_buffer(&counts.slice)?)?;
        inp.read_pixel_sample_counts(0, ntx - 1, 0, nty - 1, lx, ly, &mut counts.bytes)?;
        let cols = (win.max.x as i64 - win.min.x as i64 + 1) as usize;
        lv.edit_counts(|c| counts.copy_into(c, cols))?;
        let mut heap = Heap::new(lv, lv.sample_counts().total_samples() as u64)?;
        inp.set_frame_buffer(&heap.frame_buffer(&counts.slice)?)?;
        inp.read_tiles(0, ntx - 1, 0, nty - 1, lx, ly, &counts.bytes, &mut heap.bytes)?;
        heap.scatter(lv)?;
    }
    copy_header(&file_hdr, false)
}
