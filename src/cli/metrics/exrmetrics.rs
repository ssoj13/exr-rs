//! `exrmetrics.cpp` / `exrmetrics.h`: copy an image through memory part by part, timing every read,
//! write and re-read, and measuring sizes.
//!
//! A part is read whole into buffers laid out like the output wants it (every pixel converted to
//! the requested type by the reader), written from those buffers to a memory stream (or a file),
//! and read back from what was written. Scanline, tiled, deep scanline and deep tiled parts each
//! have their own buffers and calls, as in the C++.
//!
//! Deliberate differences from `exrmetrics.cpp`, each a defect there:
//! * a single-part tiled file without a `type` attribute (nearly every ordinary tiled image) fails
//!   in the C++ with `Cannot find image attribute "type"` (`TiledInputPart` does not see the type
//!   `MultiPartInputFile` adds); here it is measured like any other;
//! * deep tiled sample buffers are sized by the INPUT channel types (`exrmetrics.cpp:568-570`) while
//!   the frame buffer addresses them with the OUTPUT types, so `--pixelmode` on a half deep tiled
//!   image writes past the buffers; here one layout serves both;
//! * a part of a type that is none of the four image types is an error (the C++ skips it and reports
//!   zeros for it).
//!
//! * `tile count` of a mipmapped part adds the tiles of every PAIR of levels, the off-diagonal ones a
//!   mipmap does not have included (`exrmetrics.cpp:244`, the addition sits outside the `if`);
//! * a deep tiled part records no compression, part type or tile count (`initAndReadDeepTiled`
//!   forgets them), so the report says `mixed` / `mixed` / `0`;
//! * `--pixelmode mixed` takes the channel suffix after the last lower case `r` (`:1079`), which
//!   makes `layer.R` a non-RGBA channel, against its own comment ("'R' in 'layer.R'").

use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;
use std::time::Instant;

use exr_core::attr::{Compression, LevelMode, Name, PixelType, TileDescription};
use exr_core::deep::{DeepFrameBuffer, DeepSlice};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{self, Header};
use exr_core::io::{write_file as atomic_write, MemIStream, MemOStream, OStream, StdIfStream};
use exr_core::multipart::{DeepScanLineInputPart, DeepTiledInputPart, MultiPartInputFileOwned, MultiPartOutputFile, TiledInputPart};
use exr_core::tiled::misc::data_window_for_level;
use exr_core::ExrError;
use imath_rs::Box2i;

use crate::cli::tool_io::CountOut;
use crate::cli::CliError;

type Res<T> = Result<T, CliError>;

/// How pixel data types are changed on the way through (`--pixelmode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelMode {
    Original,
    AllHalf,
    AllFloat,
    /// Half for R, G, B and A, float for the rest.
    MixedHalfFloat,
}

impl PixelMode {
    /// `modeName`.
    pub fn name(self) -> &'static str {
        match self {
            PixelMode::AllFloat => "float",
            PixelMode::AllHalf => "half",
            PixelMode::MixedHalfFloat => "mixed",
            PixelMode::Original => "original",
        }
    }
}

/// What a part is made of (`partSizeData`).
#[derive(Clone, Debug, Default)]
pub struct PartSizeData {
    /// Bytes the pixel data needs on its own.
    pub raw_size: u64,
    /// Pixels, every level of a tiled image included.
    pub pixel_count: u64,
    pub channel_count: u64,
    pub tile_count: u64,
    pub is_deep: bool,
    pub is_tiled: bool,
    /// `None`: no compression recorded (`NUM_COMPRESSION_METHODS` in the C++).
    pub compression: Option<Compression>,
    pub part_type: String,
}

/// Times and sizes of one part (`partStats`); one time per pass, one read time in all.
#[derive(Clone, Debug, Default)]
pub struct PartStats {
    pub count_read: Vec<f64>,
    pub read: Vec<f64>,
    pub write: Vec<f64>,
    pub count_reread: Vec<f64>,
    pub reread: Vec<f64>,
    pub size_on_disk: u64,
    pub size_data: PartSizeData,
}

/// Everything measured for one file, compression and pixel mode (`fileMetrics`).
#[derive(Debug, Default)]
pub struct FileMetrics {
    pub stats: Vec<PartStats>,
    pub total: PartStats,
    pub input_size: u64,
    pub output_size: u64,
}

// ----- buffers -----------------------------------------------------------------------------

/// Bytes of one sample (`pixelTypeSize`).
fn size_of(t: PixelType) -> usize {
    if t == PixelType::Half { 2 } else { 4 }
}

fn overflow() -> CliError {
    CliError::Exr(ExrError::Overflow("Integer multiplication overflow.".into()))
}

fn fail(msg: impl Into<String>) -> CliError {
    CliError::Other(msg.into())
}

/// `vectorSize`: the bytes of `count` samples of `size` bytes, without overflow.
fn vector_size(count: u64, size: usize) -> Res<usize> {
    usize::try_from(count).ok().and_then(|c| c.checked_mul(size)).ok_or_else(overflow)
}

/// A zeroed buffer; the allocation failing is an error, not an abort.
fn zeroed(n: usize) -> Res<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| fail(format!("cannot allocate {n} bytes")))?;
    v.resize(n, 0);
    Ok(v)
}

/// Width and height of a window in pixels.
fn dims(w: &Box2i) -> (u64, u64) {
    let d = |lo: i32, hi: i32| u64::try_from(i64::from(hi) + 1 - i64::from(lo)).unwrap_or(0);
    (d(w.min.x, w.max.x), d(w.min.y, w.max.y))
}

/// The pixels of a part and the frame buffers that address them.
enum Buf {
    /// One buffer holding every channel's plane.
    Scan { fb: FrameBuffer, pixels: Vec<u8> },
    /// One buffer and frame buffer per level.
    Tiled { levels: Vec<(FrameBuffer, Vec<u8>)> },
    Deep(DeepBuf),
}

/// A deep part: the per-pixel sample counts (a raster of little-endian `u32`) and one heap in which
/// every channel owns a region holding its samples pixel after pixel, the layout exr-core's deep
/// readers and writers address (the C++ points each pixel at its samples instead).
struct DeepBuf {
    counts: Vec<u8>,
    count_slice: Slice,
    heap: Vec<u8>,
    /// Channel, type and the byte range of its region of `heap`.
    regions: Vec<(Name, PixelType, Range<usize>)>,
}

impl DeepBuf {
    /// A zeroed count raster for the window `dw`; the heap is sized when the counts are known.
    fn new(dw: &Box2i, outh: &Header) -> Res<Self> {
        let (w, h) = dims(dw);
        let y_stride = usize::try_from(w).ok().and_then(|w| w.checked_mul(4)).ok_or_else(overflow)?;
        let n = usize::try_from(h).ok().and_then(|h| h.checked_mul(y_stride)).ok_or_else(overflow)?;
        let base = -(i64::from(dw.min.y) * y_stride as i64 + i64::from(dw.min.x) * 4);
        let regions = outh.channels()?.iter().map(|(name, ch)| (name.clone(), ch.type_, 0..0)).collect();
        Ok(DeepBuf {
            counts: zeroed(n)?,
            count_slice: Slice::new(PixelType::Uint, base, 4, y_stride, 1, 1, 0.0, false, false)?,
            heap: Vec::new(),
            regions,
        })
    }

    /// The samples of the whole window (the sum of the counts).
    fn total(&self) -> u64 {
        self.counts.as_chunks::<4>().0.iter().map(|c| u64::from(u32::from_le_bytes(*c))).sum()
    }

    /// Size the heap for `total` samples of every channel.
    fn alloc(&mut self, total: u64) -> Res<()> {
        let mut at = 0usize;
        for (_, t, range) in &mut self.regions {
            let len = vector_size(total, size_of(*t))?;
            *range = at..at.checked_add(len).ok_or_else(overflow)?;
            at = range.end;
        }
        self.heap = zeroed(at)?;
        Ok(())
    }

    fn fb(&self) -> Res<DeepFrameBuffer> {
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(self.count_slice.clone())?;
        for (name, t, range) in &self.regions {
            let base = i64::try_from(range.start).map_err(|_| overflow())?;
            fb.insert(name.clone(), DeepSlice::new(*t, base, 8, 0, size_of(*t), 1, 1, 0.0, false, false)?)?;
        }
        Ok(fb)
    }
}

/// Seconds since `start`.
fn since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

/// The type of the part as the four storage kinds tell it; anything else cannot be measured.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Scan,
    Tiled,
    DeepScan,
    DeepTiled,
}

fn kind(ty: &str, part: i32) -> Res<Kind> {
    match ty {
        header::SCANLINE_IMAGE => Ok(Kind::Scan),
        header::TILED_IMAGE => Ok(Kind::Tiled),
        header::DEEP_SCANLINE => Ok(Kind::DeepScan),
        header::DEEP_TILE => Ok(Kind::DeepTiled),
        other => Err(CliError::Exr(ExrError::Input(format!("Part {part} has the unsupported type \"{other}\".")))),
    }
}

// ----- scanline ----------------------------------------------------------------------------

/// `initScanLine`: a plane per channel of the OUTPUT header (so reading converts), full resolution.
fn init_scan_line(inh: &Header, outh: &Header) -> Res<(Buf, PartSizeData)> {
    let dw = inh.data_window()?;
    let (width, height) = dims(&dw);
    let num_pixels = width.checked_mul(height).ok_or_else(overflow)?;
    let row = usize::try_from(width).map_err(|_| overflow())?;

    let (mut raw_size, mut at, mut planes) = (0u64, 0usize, Vec::new());
    for (name, ch) in outh.channels()?.iter() {
        let size = size_of(ch.type_);
        raw_size += (width / ch.x_sampling as u64) * (height / ch.y_sampling as u64) * size as u64;
        planes.push((name.clone(), *ch, at, size));
        at = at.checked_add(vector_size(num_pixels, size)?).ok_or_else(overflow)?;
    }
    let pixels = zeroed(at)?;
    let mut fb = FrameBuffer::new();
    for (name, ch, start, size) in planes {
        let y_stride = row.checked_mul(size).ok_or_else(overflow)?;
        fb.insert(name, Slice::make_from_window(ch.type_, start, dw, size, y_stride, ch.x_sampling, ch.y_sampling, 0.0, false, false)?)?;
    }
    let data = PartSizeData {
        raw_size,
        pixel_count: num_pixels,
        channel_count: inh.channels()?.len() as u64,
        part_type: inh.type_()?.to_string(),
        compression: Some(inh.compression()?),
        ..PartSizeData::default()
    };
    Ok((Buf::Scan { fb, pixels }, data))
}

// ----- tiled -------------------------------------------------------------------------------

/// `initTiled`: a buffer per level. `tileCount` adds the tiles of the levels there are (the C++ adds
/// those of every pair of levels, off-diagonal ones a mipmap does not have included).
fn init_tiled(inh: &Header, outh: &Header, part: &TiledInputPart<'_>) -> Res<(Buf, PartSizeData)> {
    let td = inh.tile_description()?;
    let dw = inh.data_window()?;
    let (mut levels, mut pixel_size, mut total_pixels, mut tile_count) = (Vec::new(), 0u64, 0u64, 0u64);
    for xl in 0..part.num_x_levels() {
        for yl in 0..part.num_y_levels() {
            if td.mode == LevelMode::RipmapLevels || xl == yl {
                let lw = data_window_for_level(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y, xl, yl)?;
                let (w, h) = dims(&lw);
                let num_pixels = w.checked_mul(h).ok_or_else(overflow)?;
                let row = usize::try_from(w).map_err(|_| overflow())?;
                // Samples between the start of a plane and the origin of the coordinates.
                let origin = (w as i64).checked_mul(i64::from(lw.min.y)).and_then(|o| o.checked_add(i64::from(lw.min.x))).ok_or_else(overflow)?;

                let (mut at, mut planes) = (0usize, Vec::new());
                pixel_size = 0;
                for (name, ch) in outh.channels()?.iter() {
                    let size = size_of(ch.type_);
                    planes.push((name.clone(), ch.type_, at, size));
                    at = at.checked_add(vector_size(num_pixels, size)?).ok_or_else(overflow)?;
                    pixel_size += size as u64;
                }
                let pixels = zeroed(at)?;
                let mut fb = FrameBuffer::new();
                for (name, t, start, size) in planes {
                    let base = i64::try_from(start).ok().zip(origin.checked_mul(size as i64)).and_then(|(s, o)| s.checked_sub(o)).ok_or_else(overflow)?;
                    fb.insert(name, Slice::new(t, base, size, row.checked_mul(size).ok_or_else(overflow)?, 1, 1, 0.0, false, false)?)?;
                }
                total_pixels += num_pixels;
                tile_count += part.num_x_tiles(xl)? as u64 * part.num_y_tiles(yl)? as u64;
                levels.push((fb, pixels));
            }
        }
    }
    let data = PartSizeData {
        raw_size: pixel_size * total_pixels,
        pixel_count: total_pixels,
        tile_count,
        is_tiled: true,
        channel_count: inh.channels()?.len() as u64,
        part_type: inh.type_()?.to_string(),
        compression: Some(inh.compression()?),
        ..PartSizeData::default()
    };
    Ok((Buf::Tiled { levels }, data))
}

/// Whether the level pair `(xl, yl)` exists in a file with this tiling.
fn has_level(td: &TileDescription, xl: i32, yl: i32) -> bool {
    td.mode == LevelMode::RipmapLevels || xl == yl
}

// ----- deep --------------------------------------------------------------------------------

/// `initAndReadDeepScanLine` / `initAndReadDeepTiled`: with `reuse` (the sample total of an earlier
/// read) only the buffers are made, for the re-read; without it the counts and then the samples are
/// read, each timed on its own.
fn init_deep(inh: &Header, outh: &Header, tiles: Option<u64>, reuse: Option<u64>, mut io: impl DeepIo, count_perf: &mut Vec<f64>, sample_perf: &mut Vec<f64>) -> Res<(Buf, PartSizeData)> {
    let dw = inh.data_window()?;
    let (width, height) = dims(&dw);
    let num_pixels = width.checked_mul(height).ok_or_else(overflow)?;
    let mut buf = DeepBuf::new(&dw, outh)?;
    let bytes_per_sample: u64 = buf.regions.iter().map(|(_, t, _)| size_of(*t) as u64).sum();

    let total = match reuse {
        Some(total) => total,
        None => {
            io.set(&buf.fb()?)?;
            let start = Instant::now();
            io.read_counts(&dw, &mut buf.counts)?;
            count_perf.push(since(start));
            buf.total()
        }
    };
    buf.alloc(total)?;
    if reuse.is_none() {
        io.set(&buf.fb()?)?;
        let start = Instant::now();
        io.read_samples(&dw, &buf.counts, &mut buf.heap)?;
        sample_perf.push(since(start));
    }
    // The C++ forgets to record the compression, part type and tile count of a deep tiled part.
    let data = PartSizeData {
        raw_size: total * bytes_per_sample + num_pixels * 4,
        pixel_count: num_pixels,
        is_deep: true,
        is_tiled: tiles.is_some(),
        tile_count: tiles.unwrap_or(0),
        channel_count: inh.channels()?.len() as u64,
        part_type: inh.type_()?.to_string(),
        compression: Some(inh.compression()?),
    };
    Ok((Buf::Deep(buf), data))
}

/// What the deep scanline and deep tiled readers have in common, for [`init_deep`].
trait DeepIo {
    fn set(&mut self, fb: &DeepFrameBuffer) -> Res<()>;
    fn read_counts(&mut self, dw: &Box2i, counts: &mut [u8]) -> Res<()>;
    fn read_samples(&mut self, dw: &Box2i, counts: &[u8], heap: &mut [u8]) -> Res<()>;
}

impl DeepIo for &mut DeepScanLineInputPart<'_> {
    fn set(&mut self, fb: &DeepFrameBuffer) -> Res<()> {
        Ok(self.set_frame_buffer(fb)?)
    }
    fn read_counts(&mut self, dw: &Box2i, counts: &mut [u8]) -> Res<()> {
        Ok(self.read_pixel_sample_counts(dw.min.y, dw.max.y, counts)?)
    }
    fn read_samples(&mut self, dw: &Box2i, counts: &[u8], heap: &mut [u8]) -> Res<()> {
        Ok(self.read_pixels(dw.min.y, dw.max.y, counts, heap)?)
    }
}

impl DeepIo for &mut DeepTiledInputPart<'_> {
    fn set(&mut self, fb: &DeepFrameBuffer) -> Res<()> {
        Ok(self.set_frame_buffer(fb)?)
    }
    fn read_counts(&mut self, _: &Box2i, counts: &mut [u8]) -> Res<()> {
        let (nx, ny) = (self.num_x_tiles(0)?, self.num_y_tiles(0)?);
        Ok(self.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, counts)?)
    }
    fn read_samples(&mut self, _: &Box2i, counts: &[u8], heap: &mut [u8]) -> Res<()> {
        let (nx, ny) = (self.num_x_tiles(0)?, self.num_y_tiles(0)?);
        Ok(self.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, counts, heap)?)
    }
}

// ----- the whole file ----------------------------------------------------------------------

/// A part's buffers for reading the input (and writing the output from them) and, when the output
/// is read back, a second set for that.
struct PartBufs {
    read: Buf,
    reread: Option<Buf>,
}

/// `initAndReadFile`: allocate every part's buffers and read the input into them (timed).
fn init_and_read(input: &mut MultiPartInputFileOwned<'_>, out_headers: &[Header], part: i32, parts: &mut Vec<PartBufs>, metrics: &mut FileMetrics, reread: bool) -> Res<()> {
    for (p, outh) in out_headers.iter().enumerate() {
        let rp = if part == -1 { p as i32 } else { part };
        let inh = input.header(rp)?.clone();
        let stats = &mut metrics.stats[p];
        match kind(inh.type_()?, rp)? {
            Kind::Scan => {
                let (mut read, data) = init_scan_line(&inh, outh)?;
                stats.size_data = data;
                let again = if reread { Some(init_scan_line(&inh, outh)?.0) } else { None };
                if let Buf::Scan { fb, pixels } = &mut read {
                    let dw = inh.data_window()?;
                    let mut inpart = input.input_part_wrapper(rp)?;
                    let start = Instant::now();
                    inpart.set_frame_buffer(fb)?;
                    inpart.read_pixels(dw.min.y, dw.max.y, pixels)?;
                    stats.read.push(since(start));
                }
                parts.push(PartBufs { read, reread: again });
            }
            Kind::Tiled => {
                let mut inpart = input.tiled_input_part_wrapper(rp)?;
                let (mut read, data) = init_tiled(&inh, outh, &inpart)?;
                stats.size_data = data;
                let again = if reread { Some(init_tiled(&inh, outh, &inpart)?.0) } else { None };
                if let Buf::Tiled { levels } = &mut read {
                    read_tiled(&mut inpart, levels, &mut stats.read)?;
                }
                parts.push(PartBufs { read, reread: again });
            }
            Kind::DeepScan => {
                let mut inpart = input.deep_scanline_input_part_wrapper(rp)?;
                let (read, data) = init_deep(&inh, outh, None, None, &mut inpart, &mut stats.count_read, &mut stats.read)?;
                let again = match (reread, &read) {
                    (true, Buf::Deep(b)) => Some(init_deep(&inh, outh, None, Some(b.total()), &mut inpart, &mut stats.count_read, &mut stats.read)?.0),
                    _ => None,
                };
                stats.size_data = data;
                parts.push(PartBufs { read, reread: again });
            }
            Kind::DeepTiled => {
                let td = inh.tile_description()?;
                match td.mode {
                    LevelMode::MipmapLevels => return Err(fail("exrmetrics does not support mipmapped deep tiled parts")),
                    LevelMode::RipmapLevels => return Err(fail("exrmetrics does not support ripmapped deep tiled parts")),
                    _ => {}
                }
                let mut inpart = input.deep_tiled_input_part_wrapper(rp)?;
                let tiles = inpart.num_x_tiles(0)? as u64 * inpart.num_y_tiles(0)? as u64;
                let (read, data) = init_deep(&inh, outh, Some(tiles), None, &mut inpart, &mut stats.count_read, &mut stats.read)?;
                let again = match (reread, &read) {
                    (true, Buf::Deep(b)) => Some(init_deep(&inh, outh, Some(tiles), Some(b.total()), &mut inpart, &mut stats.count_read, &mut stats.read)?.0),
                    _ => None,
                };
                stats.size_data = data;
                parts.push(PartBufs { read, reread: again });
            }
        }
    }
    Ok(())
}

/// `readTiled`: every level, the frame buffer set inside the timing.
fn read_tiled(part: &mut TiledInputPart<'_>, levels: &mut [(FrameBuffer, Vec<u8>)], perf: &mut Vec<f64>) -> Res<()> {
    let td = part.header().tile_description()?;
    let start = Instant::now();
    let mut idx = 0;
    for xl in 0..part.num_x_levels() {
        for yl in 0..part.num_y_levels() {
            if has_level(&td, xl, yl) {
                let (fb, pixels) = levels.get_mut(idx).ok_or_else(|| fail("a level without a buffer"))?;
                part.set_frame_buffer(fb)?;
                let (nx, ny) = (part.num_x_tiles(xl)?, part.num_y_tiles(yl)?);
                part.read_tiles(0, nx - 1, 0, ny - 1, xl, yl, pixels)?;
                idx += 1;
            }
        }
    }
    perf.push(since(start));
    Ok(())
}

/// `writeFile`: every part from its read buffers. `perf` says whether to time the writes; `tap`, the
/// stream position after each part (so `sizeOnDisk` is what that part added).
fn write_file(out: &mut MultiPartOutputFile<'_>, parts: &[PartBufs], metrics: &mut FileMetrics, log: bool, tap: Option<&Tap>) -> Res<()> {
    let mut last = tap.map_or(0, |t| t.get());
    for (p, part) in parts.iter().enumerate() {
        let n = p as i32;
        let dw = out.header(n)?.data_window()?;
        let perf = if log { Some(&mut metrics.stats[p].write) } else { None };
        match &part.read {
            Buf::Scan { fb, pixels } => {
                let mut o = out.output_part_wrapper(n)?;
                o.set_frame_buffer(fb)?;
                let start = Instant::now();
                o.write_pixels(pixels, rows(&dw)?)?;
                if let Some(perf) = perf {
                    perf.push(since(start));
                }
            }
            Buf::Tiled { levels } => {
                let mut o = out.tiled_output_part_wrapper(n)?;
                let td = o.header().tile_description()?;
                let start = Instant::now();
                let mut idx = 0;
                for xl in 0..o.num_x_levels() {
                    for yl in 0..o.num_y_levels() {
                        if has_level(&td, xl, yl) {
                            let (fb, pixels) = levels.get(idx).ok_or_else(|| fail("a level without a buffer"))?;
                            o.set_frame_buffer(fb)?;
                            let (nx, ny) = (o.num_x_tiles(xl)?, o.num_y_tiles(yl)?);
                            o.write_tiles(pixels, 0, nx - 1, 0, ny - 1, xl, yl)?;
                            idx += 1;
                        }
                    }
                }
                if let Some(perf) = perf {
                    perf.push(since(start));
                }
            }
            Buf::Deep(b) if out.header(n)?.type_()? == header::DEEP_SCANLINE => {
                let mut o = out.deep_scanline_output_part_wrapper(n)?;
                o.set_frame_buffer(&b.fb()?)?;
                let start = Instant::now();
                o.write_pixels(&b.counts, &b.heap, rows(&dw)?)?;
                if let Some(perf) = perf {
                    perf.push(since(start));
                }
            }
            Buf::Deep(b) => {
                let mut o = out.deep_tiled_output_part_wrapper(n)?;
                o.set_frame_buffer(&b.fb()?)?;
                let (nx, ny) = (o.num_x_tiles(0)?, o.num_y_tiles(0)?);
                let start = Instant::now();
                o.write_tiles(&b.counts, &b.heap, 0, nx - 1, 0, ny - 1, 0, 0)?;
                if let Some(perf) = perf {
                    perf.push(since(start));
                }
            }
        }
        if let Some(t) = tap {
            let now = t.get();
            metrics.stats[p].size_on_disk = now.saturating_sub(last);
            last = now;
        }
    }
    Ok(())
}

/// Scanlines of a data window, for `writePixels`.
fn rows(dw: &Box2i) -> Res<i32> {
    i32::try_from(dims(dw).1).map_err(|_| overflow())
}

/// `rereadFile`: every part of what was written, into the second set of buffers.
fn reread_file(input: &mut MultiPartInputFileOwned<'_>, parts: &mut [PartBufs], metrics: &mut FileMetrics) -> Res<()> {
    for (p, part) in parts.iter_mut().enumerate() {
        let n = p as i32;
        let stats = &mut metrics.stats[p];
        let dw = input.header(n)?.data_window()?;
        match part.reread.as_mut() {
            Some(Buf::Scan { fb, pixels }) => {
                let mut i = input.input_part_wrapper(n)?;
                let start = Instant::now();
                i.set_frame_buffer(fb)?;
                i.read_pixels(dw.min.y, dw.max.y, pixels)?;
                stats.reread.push(since(start));
            }
            Some(Buf::Tiled { levels }) => {
                let mut i = input.tiled_input_part_wrapper(n)?;
                read_tiled(&mut i, levels, &mut stats.reread)?;
            }
            Some(Buf::Deep(b)) => {
                if input.header(n)?.type_()? == header::DEEP_SCANLINE {
                    let mut i = input.deep_scanline_input_part_wrapper(n)?;
                    i.set_frame_buffer(&b.fb()?)?;
                    let start = Instant::now();
                    i.read_pixel_sample_counts(dw.min.y, dw.max.y, &mut b.counts)?;
                    stats.count_reread.push(since(start));
                    let start = Instant::now();
                    i.read_pixels(dw.min.y, dw.max.y, &b.counts, &mut b.heap)?;
                    stats.reread.push(since(start));
                } else {
                    let mut i = input.deep_tiled_input_part_wrapper(n)?;
                    i.set_frame_buffer(&b.fb()?)?;
                    let (nx, ny) = (i.num_x_tiles(0)?, i.num_y_tiles(0)?);
                    let start = Instant::now();
                    i.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut b.counts)?;
                    stats.count_reread.push(since(start));
                    let start = Instant::now();
                    i.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &b.counts, &mut b.heap)?;
                    stats.reread.push(since(start));
                }
            }
            None => {}
        }
    }
    Ok(())
}

// ----- streams -----------------------------------------------------------------------------

/// The position of a stream somebody else is writing to.
type Tap = Rc<Cell<u64>>;

/// `MemOStream` of the C++: the output file in memory, whose position the writer of the parts can
/// also read while the file is being written (the stream itself belongs to the `MultiPartOutputFile`).
struct MemOut {
    mem: MemOStream,
    at: Tap,
}

impl OStream for MemOut {
    fn file_name(&self) -> &str {
        "<memory>"
    }
    fn write(&mut self, buf: &[u8]) -> exr_core::Result<()> {
        self.mem.write(buf)?;
        self.at.set(self.mem.tellp()?);
        Ok(())
    }
    fn tellp(&mut self) -> exr_core::Result<u64> {
        Ok(self.at.get())
    }
    fn seekp(&mut self, pos: u64) -> exr_core::Result<()> {
        self.mem.seekp(pos)?;
        self.at.set(pos);
        Ok(())
    }
}

/// `accumulate`: add `input` to `output` entry by entry, growing `output` to fit.
fn accumulate(output: &mut Vec<f64>, input: &[f64]) {
    if output.len() < input.len() {
        output.resize(input.len(), 0.0);
    }
    for (o, i) in output.iter_mut().zip(input) {
        *o += i;
    }
}

/// `exrmetrics`: read `in_file` (one part or all) into memory as `mode` and `compression`, write it
/// `passes` times (to `out_file`, else to memory) and read the result back, measuring as it goes.
#[allow(clippy::too_many_arguments)]
pub fn exrmetrics(
    in_file: &str,
    out_file: Option<&str>,
    part: i32,
    compression: Option<Compression>,
    level: f32,
    passes: i32,
    write: bool,
    reread: bool,
    mode: PixelMode,
    verbose: bool,
) -> Res<FileMetrics> {
    if verbose {
        err!("read {in_file} as {}... ", mode.name());
    }
    let mut input = MultiPartInputFileOwned::new_owned(Box::new(StdIfStream::open(in_file)?))?;
    if part != -1 && part >= input.parts() {
        return Err(fail(format!("{in_file} only contains {} parts. Cannot copy part {part}", input.parts())));
    }
    let mut metrics = FileMetrics::default();

    // Every part, or the chosen one.
    let mut out_headers: Vec<Header> = if part == -1 {
        (0..input.parts()).map(|p| input.header(p).cloned()).collect::<exr_core::Result<_>>()?
    } else {
        vec![input.header(part)?.clone()]
    };

    let level_applies = !level.is_infinite() && level >= -1.0;
    let mut level_set = false;
    for h in &mut out_headers {
        if let Some(c) = compression {
            *h.compression_mut()? = c;
        }
        if level_applies {
            match h.compression()? {
                Compression::Dwaa | Compression::Dwab => {
                    h.set_dwa_level(level);
                    level_set = true;
                }
                Compression::Zip | Compression::Zips => {
                    h.set_zip_level(level as i32);
                    level_set = true;
                }
                _ => {}
            }
        }
        if mode != PixelMode::Original {
            let names: Vec<Name> = h.channels()?.iter().map(|(n, _)| n.clone()).collect();
            for name in names {
                // The channel's own name after the last `.` (`R` of `layer.R`); the C++ looks after
                // the last lower case `r` (`strrchr (name, 'r')`, `exrmetrics.cpp:1079`).
                let bytes = name.as_bytes();
                let suffix = bytes.iter().rposition(|&b| b == b'.').map_or(bytes, |i| &bytes[i + 1..]);
                let rgba = matches!(suffix, b"R" | b"G" | b"B" | b"A");
                let ty = match mode {
                    PixelMode::AllHalf => PixelType::Half,
                    PixelMode::MixedHalfFloat if rgba => PixelType::Half,
                    _ => PixelType::Float,
                };
                h.channels_mut()?.get_mut(&name)?.type_ = ty;
            }
        }
    }
    if level_applies && !level_set {
        return Err(fail("-l option only works for DWAA/DWAB,ZIP/ZIPS or ZSTD compression"));
    }

    metrics.stats = vec![PartStats::default(); out_headers.len()];
    let mut parts = Vec::with_capacity(out_headers.len());
    init_and_read(&mut input, &out_headers, part, &mut parts, &mut metrics, reread)?;

    if write {
        // Without an output file, the size is worked out first by writing into a stream that keeps
        // nothing, then the passes write to memory.
        let mut file_size = 0;
        if out_file.is_none() {
            let mut tmp = CountOut(0);
            {
                let mut out = MultiPartOutputFile::new(&mut tmp, &out_headers)?;
                write_file(&mut out, &parts, &mut metrics, false, None)?;
            }
            file_size = tmp.tellp()?;
        }
        if verbose {
            err!(" write ");
            if let Some(c) = compression {
                err!("compression {}", c.name());
            }
            err!("... ");
        }
        for i in 0..passes {
            if verbose && passes > 1 {
                err!("{i} ");
            }
            let written = match out_file {
                Some(path) => {
                    // All or nothing: a refused header set or a failed write leaves the target as it was.
                    atomic_write(path, |stream| {
                        let mut out = MultiPartOutputFile::new(stream, &out_headers)?;
                        write_file(&mut out, &parts, &mut metrics, true, None)
                    })?;
                    None
                }
                None => {
                    let at: Tap = Rc::default();
                    let mut stream = MemOut { mem: MemOStream::new(), at: at.clone() };
                    {
                        let mut out = MultiPartOutputFile::new(&mut stream, &out_headers)?;
                        write_file(&mut out, &parts, &mut metrics, true, Some(&at))?;
                    }
                    Some(stream.mem.into_data())
                }
            };
            if reread {
                let source: Box<dyn exr_core::io::IStream> = match (out_file, written) {
                    (Some(path), _) => Box::new(StdIfStream::open(path)?),
                    (None, Some(data)) => Box::new(MemIStream::new(data)),
                    (None, None) => return Err(fail("the output was not kept")),
                };
                let mut back = MultiPartInputFileOwned::new_owned(source)?;
                reread_file(&mut back, &mut parts, &mut metrics)?;
            }
        }

        metrics.input_size = std::fs::metadata(in_file).map_err(|e| fail(format!("cannot stat {in_file}: {e}")))?.len();
        metrics.output_size = match out_file {
            Some(path) => std::fs::metadata(path).map_err(|e| fail(format!("cannot stat {path}: {e}")))?.len(),
            None => file_size,
        };
    }

    // Sum across the parts, noticing mixed compressions and part types.
    let mut total = metrics.stats[0].clone();
    for s in &metrics.stats[1..] {
        accumulate(&mut total.read, &s.read);
        accumulate(&mut total.count_read, &s.count_read);
        accumulate(&mut total.write, &s.write);
        accumulate(&mut total.reread, &s.reread);
        accumulate(&mut total.count_reread, &s.count_reread);
        let (t, d) = (&mut total.size_data, &s.size_data);
        t.pixel_count += d.pixel_count;
        t.channel_count += d.channel_count;
        t.raw_size += d.raw_size;
        t.tile_count += d.tile_count;
        t.is_deep |= d.is_deep;
        t.is_tiled |= d.is_tiled;
        if d.compression != t.compression {
            t.compression = None;
        }
        if d.part_type != t.part_type {
            t.part_type.clear();
        }
    }
    metrics.total = total;
    if verbose {
        errln!();
    }
    Ok(metrics)
}
