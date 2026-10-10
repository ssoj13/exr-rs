//! Port of `ImfFlatImageIO.{h,cpp}`: load and save flat images, scanline or tiled, every level.
//!
//! The C++ hands the readers and writers `Slice`s that point into the channels' storage. Slices
//! here are descriptors over a caller byte buffer, so the pixels travel through a strip buffer of a
//! few megabytes (see [`crate::image_io`]): a window of rows is read into it and converted into the
//! typed channels, or gathered from them and written. Memory stays at one copy of the image plus
//! one strip, and no whole-image byte buffer exists.
//!
//! `save*` take the header the C++ overloads take as `Option<&Header>` (`None` = the overload
//! without a header); `load*` return the file's header instead of filling a `Header&`.

use std::path::Path;

use exr_core::attr::{Channel, LevelMode, LevelRoundingMode, LineOrder, Name};
use exr_core::buf::ensure_len;
use exr_core::header::Header;
use exr_core::io::{write_file, StdIfStream};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::tiled::{TiledInputFile, TiledOutputFile};
use exr_core::Result;
use imath_rs::Box2i;

use crate::flat_image::FlatImage;
use crate::flat_image_level::FlatImageLevel;
use crate::image_channel::ImageChannel;
use crate::image_data_window::DataWindowSource;
use crate::image_io::{
    copy_header, file_header, level_order, level_window, path_str, probe_kind, save_tiled, scanline_unit, strip_ranges,
    strip_rows, Strip,
};
use crate::image_level::ImageLevel;

/// The channels of a level as the strip layouts want them.
fn chans(lv: &FlatImageLevel) -> impl Iterator<Item = (&Name, Channel)> {
    lv.iter().map(|(n, c)| (n, c.channel()))
}

/// Copy a strip buffer (filled by an exr-core reader through [`Strip::frame_buffer`]) into the image:
/// every plane into the rectangle of its channel that starts at
/// the plane's first sample.
pub fn unpack(lv: &mut FlatImageLevel, strip: &Strip, buf: &[u8]) -> Result<()> {
    let dw = *lv.data_window();
    for (name, p) in &strip.planes {
        let (col0, row0) = (p.gx - dw.min.x as i64 / p.xs as i64, p.gy - dw.min.y as i64 / p.ys as i64);
        let bytes = buf.get(p.off..p.off + p.len).unwrap_or_default();
        lv.channel_mut(name)?.load_rect(col0 as usize, row0 as usize, p.cols, p.rows, bytes)?;
    }
    Ok(())
}

/// Fill a strip buffer from the image (the inverse of [`unpack`]), ready for an exr-core writer that
/// was given [`Strip::frame_buffer`]. The planes' channels must exist in `lv`.
pub fn pack(lv: &FlatImageLevel, strip: &Strip, buf: &mut [u8]) -> Result<()> {
    let dw = lv.data_window();
    for (name, p) in &strip.planes {
        let (col0, row0) = (p.gx - dw.min.x as i64 / p.xs as i64, p.gy - dw.min.y as i64 / p.ys as i64);
        let out = buf.get_mut(p.off..p.off + p.len).unwrap_or_default();
        lv.channel(name)?.store_rect(col0 as usize, row0 as usize, p.cols, p.rows, out)?;
    }
    Ok(())
}

/// `saveFlatImage`: a tiled file if the image has several levels or `hdr` has a tile description,
/// else a scanline file.
pub fn save_flat_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &FlatImage, dws: DataWindowSource) -> Result<()> {
    if save_tiled(hdr, img) {
        save_flat_tiled_image(path, hdr, img, dws)
    } else {
        save_flat_scanline_image(path, hdr, img, dws)
    }
}

/// `loadFlatImage`: load a single-part flat file into `img`, tiled or not; returns the file's header.
///
/// # Errors
/// `Arg` if the file is not an OpenEXR file, is multipart or deep.
pub fn load_flat_image(path: impl AsRef<Path>, img: &mut FlatImage) -> Result<Header> {
    let path = path.as_ref();
    if probe_kind(path, false)? {
        load_flat_tiled_image(path, img)
    } else {
        load_flat_scanline_image(path, img)
    }
}

/// `saveFlatScanLineImage`: save level 0 of `img` in a scanline file.
pub fn save_flat_scanline_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &FlatImage, dws: DataWindowSource) -> Result<()> {
    let new_hdr = file_header(hdr, img, dws, false, false)?;
    let fw = new_hdr.data_window()?;
    let lv = img.level(0)?;
    let unit = scanline_unit(chans(lv), new_hdr.compression()?);
    let rows = strip_rows(chans(lv), fw.max.x as i64 - fw.min.x as i64 + 1, unit);
    let decreasing = new_hdr.line_order()? == LineOrder::DecreasingY;
    write_file(path, |stream| {
        let mut out = OutputFile::new(stream, &new_hdr)?;
        let mut buf = Vec::new();
        for (y0, y1) in strip_ranges(fw.min.y, fw.max.y, rows, unit, decreasing) {
            let strip = Strip::new(chans(lv), (fw.min.x, fw.max.x), (y0, y1))?;
            ensure_len(&mut buf, strip.bytes);
            pack(lv, &strip, &mut buf)?;
            out.set_frame_buffer(&strip.frame_buffer()?)?;
            out.write_pixels(&buf, y1 - y0 + 1)?;
        }
        Ok(())
    })
}

/// `loadFlatScanLineImage`: load a scanline file (a tiled one is read through the scanline
/// interface) as a single-level image; returns the file's header without its `tiles` attribute.
pub fn load_flat_scanline_image(path: impl AsRef<Path>, img: &mut FlatImage) -> Result<Header> {
    let mut inp = InputFile::open(path.as_ref())?;
    let file_hdr = inp.header().clone();
    img.clear_channels();
    for (name, ch) in file_hdr.channels()?.iter() {
        img.insert_channel(name.clone(), ch)?;
    }
    img.resize_with(&file_hdr.data_window()?, LevelMode::OneLevel, LevelRoundingMode::RoundDown)?;
    let lv = img.level_mut(0)?;
    let dw = *lv.data_window();
    let unit = scanline_unit(chans(lv), file_hdr.compression()?);
    let rows = strip_rows(chans(lv), dw.max.x as i64 - dw.min.x as i64 + 1, unit);
    let mut buf = Vec::new();
    for (y0, y1) in strip_ranges(dw.min.y, dw.max.y, rows, unit, false) {
        let strip = Strip::new(chans(lv), (dw.min.x, dw.max.x), (y0, y1))?;
        ensure_len(&mut buf, strip.bytes);
        inp.set_frame_buffer(&strip.frame_buffer()?)?;
        inp.read_pixels(y0, y1, &mut buf)?;
        unpack(lv, &strip, &buf)?;
    }
    copy_header(&file_hdr, true)
}

/// Tile-row groups `[t1, t2]` of a level with `ny` tile rows of `tile_h` pixels, in file order,
/// sized for the strip buffer.
fn tile_groups(ny: i32, per_group: i64, decreasing: bool) -> Vec<(i32, i32)> {
    let g = per_group.clamp(1, ny.max(1) as i64) as i32;
    let mut v: Vec<_> = (0..ny).step_by(g as usize).map(|t| (t, (t + g - 1).min(ny - 1))).collect();
    if decreasing {
        v.reverse();
    }
    v
}

/// Pixel rows of tile rows `t1..=t2` of a level with window `lw` and tiles `tile_h` high.
fn group_rows(lw: &Box2i, tile_h: i64, t1: i32, t2: i32) -> (i32, i32) {
    let y0 = lw.min.y as i64 + t1 as i64 * tile_h;
    let y1 = (lw.min.y as i64 + (t2 as i64 + 1) * tile_h - 1).min(lw.max.y as i64);
    (y0 as i32, y1 as i32)
}

/// `saveFlatTiledImage`: save all levels of `img` in a tiled file (64 x 64 tiles unless `hdr` has a
/// tile description).
pub fn save_flat_tiled_image(path: impl AsRef<Path>, hdr: Option<&Header>, img: &FlatImage, dws: DataWindowSource) -> Result<()> {
    let new_hdr = file_header(hdr, img, dws, true, false)?;
    let fw = new_hdr.data_window()?;
    let td = new_hdr.tile_description()?;
    let decreasing = new_hdr.line_order()? == LineOrder::DecreasingY;
    write_file(path, |stream| {
        let mut out = TiledOutputFile::new(stream, &new_hdr)?;
        let mut buf = Vec::new();
        for (lx, ly) in level_order(img.level_mode(), img.num_x_levels(), img.num_y_levels()) {
            let lv = img.level_at(lx, ly)?;
            let lw = level_window(img, &fw, lx, ly)?;
            let (ntx, nty) = (out.num_x_tiles(lx)?, out.num_y_tiles(ly)?);
            let tile_h = td.y_size as i64;
            let fit_rows = strip_rows(chans(lv), lw.max.x as i64 - lw.min.x as i64 + 1, 1);
            for (t1, t2) in tile_groups(nty, fit_rows / tile_h, decreasing) {
                let rows = group_rows(&lw, tile_h, t1, t2);
                let strip = Strip::new(chans(lv), (lw.min.x, lw.max.x), rows)?;
                ensure_len(&mut buf, strip.bytes);
                pack(lv, &strip, &mut buf)?;
                out.set_frame_buffer(&strip.frame_buffer()?)?;
                out.write_tiles(&buf, 0, ntx - 1, t1, t2, lx, ly)?;
            }
        }
        Ok(())
    })
}

/// `loadFlatTiledImage`: load all levels of a tiled file; returns the file's header.
pub fn load_flat_tiled_image(path: impl AsRef<Path>, img: &mut FlatImage) -> Result<Header> {
    let mut stream = StdIfStream::open(path_str(path.as_ref())?)?;
    let mut inp = TiledInputFile::new(&mut stream)?;
    let file_hdr = inp.header().clone();
    let td = file_hdr.tile_description()?;
    img.clear_channels();
    for (name, ch) in file_hdr.channels()?.iter() {
        img.insert_channel(name.clone(), ch)?;
    }
    img.resize_with(&file_hdr.data_window()?, td.mode, td.rounding_mode)?;
    let mut buf = Vec::new();
    for (lx, ly) in level_order(img.level_mode(), img.num_x_levels(), img.num_y_levels()) {
        let lv = img.level_at_mut(lx, ly)?;
        let lw = *lv.data_window();
        let (ntx, nty) = (inp.num_x_tiles(lx)?, inp.num_y_tiles(ly)?);
        let tile_h = td.y_size as i64;
        let fit_rows = strip_rows(chans(lv), lw.max.x as i64 - lw.min.x as i64 + 1, 1);
        for (t1, t2) in tile_groups(nty, fit_rows / tile_h, false) {
            let strip = Strip::new(chans(lv), (lw.min.x, lw.max.x), group_rows(&lw, tile_h, t1, t2))?;
            ensure_len(&mut buf, strip.bytes);
            inp.set_frame_buffer(&strip.frame_buffer()?)?;
            inp.read_tiles(0, ntx - 1, t1, t2, lx, ly, &mut buf)?;
            unpack(lv, &strip, &buf)?;
        }
    }
    copy_header(&file_hdr, false)
}
