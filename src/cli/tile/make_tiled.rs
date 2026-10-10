//! `makeTiled.cpp`: produce a tiled version of an OpenEXR image (`openexr/src/bin/exrmaketiled`,
//! v3.4.15).
//!
//! The image is read into one [`FlatImageLevel`] (the C++ `Image`), then the file is written level
//! by level: each lower level is made from the one before by a 2x box-like four-tap reduction
//! (`reduce_x` / `reduce_y`), so at most three levels are in memory at once. The filter is computed
//! in `f64` with the C++ operation order: output files are byte-identical to `exrmaketiled`'s.
//!
//! Deliberate differences from the C++ (each a defect there):
//! * an error while writing the tiled part is an error (exit 1). `makeTiled.cpp:697-700` prints it
//!   and carries on, leaving a half-written file and exit code 0;
//! * a part whose type is none of the four image types is an error; the C++ writes its header
//!   without pixels (`makeTiled.cpp:702-731` has no `else`);
//! * the input file is opened once (main.cpp opens it a second time for the conversion).

use std::collections::BTreeSet;

use exr_core::attr::{Channel, Compression, LevelMode, LevelRoundingMode, LineOrder, Name, PixelType, TileDescription};
use exr_core::buf::ensure_len;
use exr_core::header::{self, Header};
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile, TiledOutputPart};
// MSVC SSE operand order for NaN payloads (`exrmaketiled` is oracle-verified byte-identical on
// files with NaN samples); the helper picks the NaN by bits, so the optimizer cannot change it.
use exr_core::sse::sd;
use exr_core::tiled::misc::get_tiled_chunk_offset_table_size;
use exr_core::{ExrError, Result};
use exr_util::flat_image_channel::FlatChannel;
use exr_util::flat_image_level::FlatImageLevel;
use exr_util::image_channel::{ImageChannel, Pixel};
use exr_util::image_level::ImageLevel;
use imath_rs::{Box2i, Half, V2i};

use exr_util::flat_image_io::{pack, unpack};
use exr_util::image_io::strip_ranges;

use crate::cli::tool_io::{rows_of, strip_of};

/// How the image is extended beyond its data window when a filter reaches past it (`-e`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extrapolation {
    Black,
    Clamp,
    Periodic,
    Mirror,
}

impl Extrapolation {
    /// `extToString`: the word the `wrapmodes` attribute carries.
    fn name(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::Clamp => "clamp",
            Self::Periodic => "periodic",
            Self::Mirror => "mirror",
        }
    }
}

/// Everything `makeTiled` takes: the files, the part and the output format.
pub struct Job {
    pub input: String,
    pub output: String,
    pub part: i32,
    pub mode: LevelMode,
    pub rounding: LevelRoundingMode,
    pub compression: Compression,
    /// Tile width and height in pixels, both positive.
    pub tile: (u32, u32),
    /// Names of the channels that are resampled without low-pass filtering (`-f`).
    pub no_filter: BTreeSet<Vec<u8>>,
    pub ext: (Extrapolation, Extrapolation),
    pub verbose: bool,
}

/// A channel sample the filters average: read as `f64` and stored back through `f64`, exactly the
/// C++ `double (channel (x, y))` and `T (double)`.
trait Sample: Pixel {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl Sample for Half {
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    // The C++ `half (float)` constructor after the implicit double -> float.
    fn from_f64(v: f64) -> Self {
        Half::from_f32(v as f32)
    }
}

impl Sample for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl Sample for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    // A convex combination of u32 values stays in range, so the saturating cast never saturates.
    fn from_f64(v: f64) -> Self {
        v as u32
    }
}

/// A channel's samples as a `w x h` raster. Reading outside it gives 0 (the `BLACK` rule); the other
/// extrapolation modes map the coordinate into the raster first, so they never read outside.
struct Plane<'a, T> {
    px: &'a [T],
    w: i32,
    h: i32,
}

impl<'a, T: Sample> Plane<'a, T> {
    /// The raster of `ch`; its storage must be exactly `w x h`, both at least 1.
    fn new(ch: &'a FlatChannel<T>) -> Result<Self> {
        let (w, h) = (ch.pixels_per_row(), ch.pixels_per_column());
        if w < 1 || h < 1 || ch.pixels().len() != w as usize * h as usize {
            return Err(ExrError::Logic("a channel raster does not match its window".into()));
        }
        Ok(Self { px: ch.pixels(), w, h })
    }

    fn at(&self, x: i32, y: i32) -> f64 {
        if x < 0 || y < 0 || x >= self.w || y >= self.h {
            return 0.0;
        }
        self.px.get(y as usize * self.w as usize + x as usize).map_or(0.0, |v| v.to_f64())
    }
}

/// `mirror`: reflect `x` into `[0, w)`, flipping on every other repeat.
fn mirror(x: i32, w: i32) -> i32 {
    let (d, m) = (x.div_euclid(w), x.rem_euclid(w));
    if d & 1 != 0 { w - 1 - m } else { m }
}

/// The index a sample at `i` is read from on an axis of length `n` (`BLACK` keeps `i`: the
/// [`Plane`] reads 0 outside).
fn wrap(ext: Extrapolation, i: i32, n: i32) -> i32 {
    match ext {
        Extrapolation::Black => i,
        // Imath `clamp`: `a < l ? l : (a > h ? h : a)`.
        Extrapolation::Clamp => {
            if i < 0 {
                0
            } else if i > n - 1 {
                n - 1
            } else {
                i
            }
        }
        Extrapolation::Periodic => i.rem_euclid(n),
        Extrapolation::Mirror => mirror(i, n),
    }
}

/// The weighted sum of the two samples `a` (at `floor (t)`) and `b` (the next one) of a position
/// `t`, whose upper neighbour is `upper`: `s * a + (1 - s) * b` with `s = upper - t`.
fn blend(t: f64, upper: i32, a: f64, b: f64) -> f64 {
    let s = upper as f64 - t;
    // `s * a + t * b` as MSVC emits it: the second product is the destination operand.
    sd::add((1.0 - s) * b, s * a)
}


/// `sampleX`: the channel at `(x, y)`, `x` fractional.
fn sample_x<T: Sample>(p: &Plane<'_, T>, x: f64, y: i32, ext: Extrapolation) -> f64 {
    let xs = x.floor() as i32;
    let xt = xs + 1;
    let (vs, vt) = (p.at(wrap(ext, xs, p.w), y), p.at(wrap(ext, xt, p.w), y));
    blend(x, xt, vs, vt)
}

/// `sampleY`: the channel at `(x, y)`, `y` fractional.
fn sample_y<T: Sample>(p: &Plane<'_, T>, x: i32, y: f64, ext: Extrapolation) -> f64 {
    let ys = y.floor() as i32;
    let yt = ys + 1;
    let (vs, vt) = (p.at(x, wrap(ext, ys, p.h)), p.at(x, wrap(ext, yt, p.h)));
    blend(y, yt, vs, vt)
}

/// The four taps summed left to right (`((t0 + t1) + t2) + t3`).
fn sum4(t: [f64; 4]) -> f64 {
    sd::add(sd::add(sd::add(t[0], t[1]), t[2]), t[3])
}

/// `filterX`: the horizontal four-tap filter centred on pixel `(x + 0.5, y)`.
fn filter_x<T: Sample>(p: &Plane<'_, T>, x: f64, y: i32, ext: Extrapolation) -> T {
    let t = [0.125 * sample_x(p, x - 1.0, y, ext), 0.375 * sample_x(p, x, y, ext), 0.375 * sample_x(p, x + 1.0, y, ext), 0.125 * sample_x(p, x + 2.0, y, ext)];
    T::from_f64(sum4(t))
}

/// `filterY`: the vertical four-tap filter centred on pixel `(x, y + 0.5)`.
fn filter_y<T: Sample>(p: &Plane<'_, T>, x: i32, y: f64, ext: Extrapolation) -> T {
    let t = [0.125 * sample_y(p, x, y - 1.0, ext), 0.375 * sample_y(p, x, y, ext), 0.375 * sample_y(p, x, y + 1.0, ext), 0.125 * sample_y(p, x, y + 2.0, ext)];
    T::from_f64(sum4(t))
}

/// The raster of `dst` as rows.
fn rows_mut<T: Sample>(dst: &mut FlatChannel<T>) -> Result<std::slice::ChunksMut<'_, T>> {
    let w = dst.pixels_per_row();
    if w < 1 {
        return Err(ExrError::Logic("an empty destination raster".into()));
    }
    Ok(dst.pixels_mut().chunks_mut(w as usize))
}

/// `reduceX (channel0, channel1, ...)`: shrink `src` horizontally by 2 into `dst` (whose height is
/// `src`'s).
fn reduce_x_chan<T: Sample>(src: &FlatChannel<T>, dst: &mut FlatChannel<T>, filter: bool, ext: Extrapolation, odd: bool) -> Result<()> {
    let p = Plane::new(src)?;
    let (w0, w1) = (p.w, dst.pixels_per_row());
    if dst.pixels_per_column() != p.h {
        return Err(ExrError::Logic("reduceX: source and destination heights differ".into()));
    }
    let rows = rows_mut(dst)?;
    // Low-pass filter and resample. For pixels (0, y) and (w1 - 1, y) of the result the filter in
    // `src` is centred on (0.5, y) and (w0 - 1.5, y).
    let f = if w1 > 1 { (w0 - 2) as f64 / (w1 - 1) as f64 } else { 1.0 };
    // Without filtering every other pixel is skipped; the skipped one alternates between the
    // rightmost (even passes) and the leftmost (odd passes) so the channel does not slide right when
    // it is resampled repeatedly.
    let offset = if odd { (w0 - 1) - 2 * (w1 - 1) } else { 0 };
    for (y, row) in rows.enumerate() {
        for (x, out) in row.iter_mut().enumerate() {
            *out = if filter {
                filter_x(&p, x as f64 * f, y as i32, ext)
            } else {
                let sx = 2 * x as i32 + offset;
                let at = usize::try_from(sx).ok().map(|sx| y * w0 as usize + sx);
                *at.and_then(|i| p.px.get(i)).ok_or_else(|| ExrError::Logic("reduceX reads outside its source".into()))?
            };
        }
    }
    Ok(())
}

/// `reduceY (channel0, channel1, ...)`: shrink `src` vertically by 2 into `dst` (whose width is
/// `src`'s).
fn reduce_y_chan<T: Sample>(src: &FlatChannel<T>, dst: &mut FlatChannel<T>, filter: bool, ext: Extrapolation, odd: bool) -> Result<()> {
    let p = Plane::new(src)?;
    let (h0, h1) = (p.h, dst.pixels_per_column());
    if dst.pixels_per_row() != p.w {
        return Err(ExrError::Logic("reduceY: source and destination widths differ".into()));
    }
    let rows = rows_mut(dst)?;
    let w1 = p.w as usize;
    // As in `reduce_x_chan`, with the centres at (x, 0.5) and (x, h0 - 1.5); the skipped row is
    // the top one on even passes and the bottom one on odd passes.
    let f = if h1 > 1 { (h0 - 2) as f64 / (h1 - 1) as f64 } else { 1.0 };
    let offset = if odd { (h0 - 1) - 2 * (h1 - 1) } else { 0 };
    for (y, row) in rows.enumerate() {
        if filter {
            for (x, out) in row.iter_mut().enumerate() {
                *out = filter_y(&p, x as i32, y as f64 * f, ext);
            }
        } else {
            let sy = usize::try_from(2 * y as i32 + offset).map_err(|_| ExrError::Logic("reduceY reads above its source".into()))?;
            let src_row = p.px.get(sy * w1..(sy + 1) * w1).ok_or_else(|| ExrError::Logic("reduceY reads below its source".into()))?;
            row.copy_from_slice(src_row);
        }
    }
    Ok(())
}

/// Which axis a reduction runs along.
#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

/// `reduceX` / `reduceY (channels, doNotFilter, ext, odd, image0, image1)`: shrink every channel of
/// `src` by a factor of 2 along `axis` into `dst`.
fn reduce(axis: Axis, chans: &[(Name, Channel)], no_filter: &BTreeSet<Vec<u8>>, ext: Extrapolation, odd: bool, src: &FlatImageLevel, dst: &mut FlatImageLevel) -> Result<()> {
    macro_rules! typed {
        ($t:ty, $name:expr, $filter:expr) => {{
            let (s, d) = (src.typed_channel::<$t>($name)?, dst.typed_channel_mut::<$t>($name)?);
            match axis {
                Axis::X => reduce_x_chan(s, d, $filter, ext, odd),
                Axis::Y => reduce_y_chan(s, d, $filter, ext, odd),
            }
        }};
    }
    for (name, ch) in chans {
        let filter = !no_filter.contains(name.as_bytes());
        match ch.type_ {
            PixelType::Half => typed!(Half, name, filter)?,
            PixelType::Float => typed!(f32, name, filter)?,
            PixelType::Uint => typed!(u32, name, filter)?,
            PixelType::NumPixelTypes => {}
        }
    }
    Ok(())
}

/// `storeLevel`: write the pixels of `level` as level `(lx, ly)` of the output part, one group of
/// tile rows at a time through a strip buffer.
fn store_level(out: &mut TiledOutputPart<'_>, lx: i32, ly: i32, level: &FlatImageLevel, buf: &mut Vec<u8>) -> Result<()> {
    let lw = *level.data_window();
    let (ntx, nty) = (out.num_x_tiles(lx)?, out.num_y_tiles(ly)?);
    let tile_h = out.tile_y_size() as i64;
    let rows = rows_of(level, lw.max.x as i64 - lw.min.x as i64 + 1, tile_h);
    let per_group = (rows / tile_h).max(1) as usize;
    for t1 in (0..nty).step_by(per_group) {
        let t2 = (t1 as i64 + per_group as i64 - 1).min(nty as i64 - 1) as i32;
        let y0 = lw.min.y as i64 + t1 as i64 * tile_h;
        let y1 = (lw.min.y as i64 + (t2 as i64 + 1) * tile_h - 1).min(lw.max.y as i64);
        let strip = strip_of(level, (lw.min.x, lw.max.x), (y0 as i32, y1 as i32))?;
        ensure_len(buf, strip.bytes);
        pack(level, &strip, buf)?;
        out.set_frame_buffer(&strip.frame_buffer()?)?;
        out.write_tiles(buf, 0, ntx - 1, t1, t2, lx, ly)?;
    }
    Ok(())
}

/// Read part `p` into `image` (strips of rows aligned to the file's chunks).
fn read_part(input: &mut MultiPartInputFile<'_>, p: i32, hdr: &Header, image: &mut FlatImageLevel) -> Result<()> {
    let dw = hdr.data_window()?;
    let unit = if hdr.has_tile_description() {
        hdr.tile_description()?.y_size as i64
    } else {
        hdr.compression()?.num_scanlines() as i64
    };
    let rows = rows_of(image, dw.max.x as i64 - dw.min.x as i64 + 1, unit);
    let mut part = input.input_part_wrapper(p)?;
    let mut buf = Vec::new();
    for (y0, y1) in strip_ranges(dw.min.y, dw.max.y, rows, unit, false) {
        let strip = strip_of(image, (dw.min.x, dw.max.x), (y0, y1))?;
        ensure_len(&mut buf, strip.bytes);
        part.set_frame_buffer(&strip.frame_buffer()?)?;
        part.read_pixels(y0, y1, &mut buf)?;
        unpack(image, &strip, &buf)?;
    }
    Ok(())
}

/// `makeTiled`: load part `job.part` of `input`, save every part to `job.output`, the loaded one
/// tiled as asked.
pub fn make_tiled(input: &mut MultiPartInputFile<'_>, job: &Job) -> Result<()> {
    let parts = input.parts();
    let mut image0 = FlatImageLevel::new(0, 0, &Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 0, y: 0 } })?;
    let mut image1 = FlatImageLevel::new(0, 0, &Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 0, y: 0 } })?;
    let mut image2 = FlatImageLevel::new(0, 0, &Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 0, y: 0 } })?;
    let mut chans: Vec<(Name, Channel)> = Vec::new();
    let mut headers = Vec::with_capacity(parts as usize);

    // Load the input image.
    for p in 0..parts {
        if job.verbose {
            outln!("reading file {}", job.input);
        }
        if p != job.part {
            headers.push(input.header(p)?.clone());
            continue;
        }
        let mut hdr = input.header(p)?.clone();
        if hdr.has_envmap() && job.mode != LevelMode::OneLevel {
            // Low-pass filtering and subsampling of environment maps is not implemented here.
            return Err(ExrError::Base(
                "This program cannot generate multiresolution environment maps.  Use exrenvmap instead.".into(),
            ));
        }
        image0.resize(&hdr.data_window()?)?;
        for (name, ch) in hdr.channels()?.iter() {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::Input("Sub-sampled image channels are not supported in tiled files.".into()));
            }
            for img in [&mut image0, &mut image1, &mut image2] {
                img.insert_channel(name, ch)?;
            }
            chans.push((name.clone(), *ch));
        }
        read_part(input, p, &hdr, &mut image0)?;

        // The output header: the input's, made tiled.
        hdr.set_tile_description(TileDescription {
            x_size: job.tile.0,
            y_size: job.tile.1,
            mode: job.mode,
            rounding_mode: job.rounding,
        })?;
        *hdr.compression_mut()? = job.compression;
        *hdr.line_order_mut()? = LineOrder::IncreasingY;
        if job.mode != LevelMode::OneLevel {
            hdr.add_wrapmodes(format!("{},{}", job.ext.0.name(), job.ext.1.name()).into())?;
        }
        // Type, tile description and chunk count for the multipart container.
        hdr.set_type(header::TILED_IMAGE)?;
        let count = get_tiled_chunk_offset_table_size(&hdr)?;
        hdr.set_chunk_count(count)?;
        headers.push(hdr);
    }

    // Store the highest-resolution level in the output file, then the lower ones.
    exr_core::io::write_file(&job.output, |stream| -> exr_core::Result<_> {
        let mut output = MultiPartOutputFile::new(stream, &headers)?;
        let mut buf = Vec::new();
        for p in 0..parts {
            if p != job.part {
                exr_core::multipart::copy_part(input, &mut output, p, p)?;
                continue;
            }
            let mut out = output.tiled_output_part_wrapper(p)?;
            if job.verbose {
                outln!("writing file {}\nlevel (0, 0)", job.output);
            }
            store_level(&mut out, 0, 0, &image0, &mut buf)?;

            // If necessary, generate the lower-resolution mipmap or ripmap levels.
            if job.mode == LevelMode::MipmapLevels {
                for l in 1..out.num_levels()? {
                    image1.resize(&out.data_window_for_level(l, l - 1)?)?;
                    reduce(Axis::X, &chans, &job.no_filter, job.ext.0, l & 1 != 0, &image0, &mut image1)?;
                    image0.resize(&out.data_window_for_level(l, l)?)?;
                    reduce(Axis::Y, &chans, &job.no_filter, job.ext.1, l & 1 != 0, &image1, &mut image0)?;
                    if job.verbose {
                        outln!("level ({l}, {l})");
                    }
                    store_level(&mut out, l, l, &image0, &mut buf)?;
                }
            }
            if job.mode == LevelMode::RipmapLevels {
                // The C++ swaps pointers to its three images; swapping the levels themselves is the
                // same (a level is a map of buffers, moved without copying them).
                let (i0, i1, i2) = (&mut image0, &mut image1, &mut image2);
                for ly in 0..out.num_y_levels() {
                    if ly < out.num_y_levels() - 1 {
                        i2.resize(&out.data_window_for_level(0, ly + 1)?)?;
                        reduce(Axis::Y, &chans, &job.no_filter, job.ext.1, ly & 1 != 0, i0, i2)?;
                    }
                    for lx in 0..out.num_x_levels() {
                        if lx != 0 || ly != 0 {
                            if job.verbose {
                                outln!("level ({lx}, {ly})");
                            }
                            store_level(&mut out, lx, ly, i0, &mut buf)?;
                        }
                        if lx < out.num_x_levels() - 1 {
                            i1.resize(&out.data_window_for_level(lx + 1, ly)?)?;
                            reduce(Axis::X, &chans, &job.no_filter, job.ext.0, lx & 1 != 0, i0, i1)?;
                            std::mem::swap(i0, i1);
                        }
                    }
                    std::mem::swap(i2, i0);
                }
            }
        }
        Ok(())
    })?;
    if job.verbose {
        outln!("done.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w x h` level with the one channel `Z` holding `vals` (row-major).
    fn level<T: Pixel>(w: i32, h: i32, vals: &[T]) -> FlatImageLevel {
        let window = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } };
        let mut l = FlatImageLevel::new(0, 0, &window).unwrap();
        l.insert_channel(&Name::from("Z"), &Channel::new(T::TYPE, 1, 1, false)).unwrap();
        l.typed_channel_mut::<T>("Z").unwrap().pixels_mut().copy_from_slice(vals);
        l
    }

    /// Reduce `vals` (`w x h`) along `axis` into `w1 x h1` and return the result.
    fn run<T: Sample>(axis: Axis, (w, h): (i32, i32), (w1, h1): (i32, i32), vals: &[T], filter: bool, ext: Extrapolation, odd: bool) -> Vec<T> {
        let src = level(w, h, vals);
        let mut dst = level::<T>(w1, h1, &vec![T::default(); (w1 * h1) as usize]);
        let no_filter: BTreeSet<Vec<u8>> = if filter { BTreeSet::new() } else { BTreeSet::from([b"Z".to_vec()]) };
        let chans = [(Name::from("Z"), Channel::new(T::TYPE, 1, 1, false))];
        reduce(axis, &chans, &no_filter, ext, odd, &src, &mut dst).unwrap();
        dst.typed_channel::<T>("Z").unwrap().pixels().to_vec()
    }

    const ROW: [f32; 4] = [0.0, 4.0, 8.0, 12.0];

    /// Hand-computed four-tap results for `[0, 4, 8, 12]` -> 2 pixels: the filter is centred on
    /// 0 and 2 (f = (4 - 2) / (2 - 1)), weights 1/8 3/8 3/8 1/8 over the taps at -1..=2 and 1..=4.
    #[test]
    fn the_filter_extrapolates_as_asked() {
        use Extrapolation::*;
        // clamp: (0, 0, 4, 8) and (4, 8, 12, 12).
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &ROW, true, Clamp, false), [2.5, 9.5]);
        // periodic: tap -1 is 12, tap 4 is 0.
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &ROW, true, Periodic, false), [4.0, 8.0]);
        // black: taps outside read 0.
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &ROW, true, Black, false), [2.5, 8.0]);
        // mirror: tap -1 is 0, tap 4 is 12.
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &ROW, true, Mirror, false), [2.5, 9.5]);
    }

    /// The vertical pass is the horizontal one turned on its side.
    #[test]
    fn the_vertical_filter_matches_the_horizontal() {
        for (ext, want) in [(Extrapolation::Clamp, [2.5, 9.5]), (Extrapolation::Periodic, [4.0, 8.0]), (Extrapolation::Black, [2.5, 8.0])] {
            assert_eq!(run(Axis::Y, (1, 4), (1, 2), &ROW, true, ext, false), want, "{ext:?}");
        }
        // Columns are independent: two columns of the same values.
        let two: Vec<f32> = ROW.iter().flat_map(|v| [*v, *v + 1.0]).collect();
        assert_eq!(run(Axis::Y, (2, 4), (2, 2), &two, true, Extrapolation::Clamp, false), [2.5, 3.5, 9.5, 10.5]);
    }

    /// Without the filter every other pixel is taken; the skipped one alternates with the pass.
    #[test]
    fn unfiltered_resampling_alternates_the_skipped_pixel() {
        let v = [10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &v, false, Extrapolation::Clamp, false), [10.0, 30.0]);
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &v, false, Extrapolation::Clamp, true), [20.0, 40.0]);
        assert_eq!(run(Axis::Y, (1, 4), (1, 2), &v, false, Extrapolation::Clamp, false), [10.0, 30.0]);
        assert_eq!(run(Axis::Y, (1, 4), (1, 2), &v, false, Extrapolation::Clamp, true), [20.0, 40.0]);
        // Odd extent (ROUND_DOWN 5 -> 2 skips the last pixel on even passes, ROUND_UP 5 -> 3 keeps all).
        let five = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(run(Axis::X, (5, 1), (2, 1), &five, false, Extrapolation::Clamp, false), [0.0, 2.0]);
        // offset (5 - 1) - 2 * (2 - 1) = 2: the two leftmost pixels are skipped.
        assert_eq!(run(Axis::X, (5, 1), (2, 1), &five, false, Extrapolation::Clamp, true), [2.0, 4.0]);
        assert_eq!(run(Axis::X, (5, 1), (3, 1), &five, false, Extrapolation::Clamp, true), [0.0, 2.0, 4.0]);
    }

    /// `T (double)`: halves round to nearest, integers truncate.
    #[test]
    fn samples_convert_like_the_cpp_constructors() {
        let u = [0u32, 4, 8, 12];
        assert_eq!(run(Axis::X, (4, 1), (2, 1), &u, true, Extrapolation::Clamp, false), [2, 9], "2.5 and 9.5 truncate");
        let h = ROW.map(Half::from_f32);
        let got = run(Axis::X, (4, 1), (2, 1), &h, true, Extrapolation::Clamp, false);
        assert_eq!(got.iter().map(|x| x.to_f32()).collect::<Vec<_>>(), [2.5, 9.5]);
        // Taps 1, 1, 1 + u, 1 + 3u (u = 2^-10, one half step above 1) weigh 1 + 0.75 u: rounding to
        // nearest gives 1 + u, truncation would give 1.
        let (a, b, c) = (1.0f32, 1.0 + 1.0 / 1024.0, 1.0 + 3.0 / 1024.0);
        let near = run(Axis::X, (3, 1), (1, 1), &[a, b, c].map(Half::from_f32), true, Extrapolation::Clamp, false);
        assert_eq!(near[0].to_f32(), b);
    }

    /// The mirror repeats the edge pixel: 3 2 1 0 | 0 1 2 3 | 3 2 1 0.
    #[test]
    fn mirror_reflects_with_the_edge_repeated() {
        let seq: Vec<i32> = (-8..12).map(|x| mirror(x, 4)).collect();
        assert_eq!(seq, [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]);
        assert_eq!((-3..9).map(|x| wrap(Extrapolation::Periodic, x, 3)).collect::<Vec<_>>(), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!((-2..6).map(|x| wrap(Extrapolation::Clamp, x, 3)).collect::<Vec<_>>(), [0, 0, 0, 1, 2, 2, 2, 2]);
        assert_eq!(wrap(Extrapolation::Black, -7, 3), -7, "black leaves the index, the plane reads 0");
    }

    /// Fractional positions interpolate between their two neighbours.
    #[test]
    fn sampling_between_pixels_interpolates() {
        let l = level(4, 1, &ROW);
        let p = Plane::new(l.typed_channel::<f32>("Z").unwrap()).unwrap();
        assert_eq!(sample_x(&p, 0.5, 0, Extrapolation::Clamp), 2.0);
        assert_eq!(sample_x(&p, 2.25, 0, Extrapolation::Clamp), 0.75 * 8.0 + 0.25 * 12.0);
        assert_eq!(sample_x(&p, -0.5, 0, Extrapolation::Clamp), 0.0);
        assert_eq!(sample_x(&p, -0.5, 0, Extrapolation::Periodic), 6.0, "half of 12 and half of 0");
        assert_eq!(sample_x(&p, -0.5, 0, Extrapolation::Black), 0.0);
        assert_eq!(sample_x(&p, 3.5, 0, Extrapolation::Black), 6.0, "half of 12 and half of black");
    }

    /// Which NaN survives a sum is the x86 rule the C++ binary has: the first tap's in the four-tap
    /// sum, the second sample's in the two-sample blend (byte-identical on the NaN test images).
    #[test]
    fn nan_payloads_follow_the_oracle_operand_order() {
        let (n1, n2) = (f64::from_bits(0x7ff8_0000_0000_0001), f64::from_bits(0x7ff8_0000_0000_0002));
        assert_eq!(sum4([1.0, n1, n2, 0.0]).to_bits(), n1.to_bits());
        assert_eq!(sum4([n2, n1, 0.0, 0.0]).to_bits(), n2.to_bits());
        assert_eq!(blend(0.5, 1, n1, n2).to_bits(), n2.to_bits());
        assert_eq!(blend(0.5, 1, n1, 3.0).to_bits(), n1.to_bits());
        assert_eq!(blend(0.5, 1, 3.0, 5.0), 4.0);
    }

    /// Mismatched rasters are errors, not out-of-range reads.
    #[test]
    fn reductions_check_their_geometry() {
        let src = level(4, 2, &[0.0f32; 8]);
        let mut wrong_height = level::<f32>(2, 1, &[0.0; 2]);
        let chans = [(Name::from("Z"), Channel::new(PixelType::Float, 1, 1, false))];
        let none = BTreeSet::new();
        assert!(reduce(Axis::X, &chans, &none, Extrapolation::Clamp, false, &src, &mut wrong_height).is_err());
        let mut wrong_width = level::<f32>(3, 1, &[0.0; 3]);
        assert!(reduce(Axis::Y, &chans, &none, Extrapolation::Clamp, false, &src, &mut wrong_width).is_err());
    }
}
