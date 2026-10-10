//! Planar read into CALLER-OWNED buffers: decode one part of an in-memory EXR straight into one
//! destination plane per channel, with the part's data window placed inside a frame the caller
//! chose. The allocation-free sibling of [`super::Image::read`]: no per-channel `Vec`, no interleave,
//! no copy of the file (the stream borrows `bytes`).
//!
//! Beyond OpenEXR (which addresses one `char*` per slice): [`PlaneDst`] is one buffer per channel
//! (an `AVFrame` plane, an `ImageBuf` channel plane), [`Frame`] is the frame those planes cover.
//!
//! # What lands where
//! A pixel at image coordinate `(x, y)` is written to plane byte
//! `(y - frame.origin.y) * y_stride + (x - frame.origin.x) * x_stride`. The data window is placed at
//! its OWN coordinates, so a frame equal to the display window (origin = display min) shows the
//! image as the file means it; a frame at origin `(0, 0)` reproduces `exr.c`, which sizes the frame
//! from the display window but places the data window at its absolute coordinates. The part of the
//! frame the data window does not cover is ZERO (a channel the file does not have is `fill`
//! everywhere), and the part of the data window outside the frame is cropped.
//!
//! # Cost
//! Scanline part whose data window lies horizontally inside the frame: one decode, written in place
//! (single- and multi-threaded, disjoint rows per plane), HALF/UINT converted to the plane's type at
//! unpack. Tiled part, or a data window sticking out sideways: the data window rows are decoded into
//! ONE tight temporary holding all planes, then copied (clipped) into the destinations - one extra
//! copy, documented here so nobody mistakes it for the fast path.

use imath_rs::V2i;

use crate::attr::PixelType;
use crate::error::{ExrError, Result, Unsupported};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::io::MemIStream;
use crate::misc::pixel_type_size;
use crate::multipart::MultiPartInputFile;

/// One destination plane: where the samples of file channel `channel` go.
pub struct PlaneDst<'a> {
    /// Full channel name as in the header (`"R"`, `"diffuse.G"`).
    pub channel: &'a str,
    /// The caller's buffer for this plane.
    pub data: &'a mut [u8],
    /// Bytes between horizontally adjacent samples.
    pub x_stride: usize,
    /// Bytes between rows.
    pub y_stride: usize,
    /// Sample type in `data`: [`PixelType::Half`] or [`PixelType::Float`] (native endian). The
    /// file's HALF / FLOAT / UINT samples are converted to it at unpack.
    pub type_: PixelType,
    /// Value of a channel the file does NOT have (everywhere in the frame).
    pub fill: f64,
}

/// The frame the planes cover: `width x height` pixels whose top-left is image coordinate `origin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Image coordinate of the frame's top-left pixel.
    pub origin: V2i,
    /// Width in pixels (> 0).
    pub width: usize,
    /// Height in pixels (> 0).
    pub height: usize,
}

/// Decode part `part` of the EXR in `bytes` into `dst` (see the module docs for placement, margins
/// and cost); returns the part's header (the file is opened here: use [`read_planes_part`] when it is already open). `num_threads` as [`MultiPartInputFile::with_num_threads`]
/// (`None` = the process default).
///
/// # Errors
/// [`ExrError::Arg`] for an empty frame, a plane that is too small for the frame, a plane type that
/// is not HALF / FLOAT, or a bad part index; [`ExrError::Unsupported`] for a deep or subsampled part;
/// everything the file parse / decode can return (corrupt data is an error, never a panic).
pub fn read_planes(
    bytes: &[u8],
    part: i32,
    frame: Frame,
    dst: &mut [PlaneDst<'_>],
    num_threads: Option<i32>,
) -> Result<Header> {
    let mut is = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::with_num_threads(&mut is, num_threads)?;
    read_planes_part(&mut mp, part, frame, dst)?;
    Ok(mp.header(part)?.clone())
}

/// [`read_planes`] over an ALREADY OPEN file: the caller opened the [`MultiPartInputFile`] once (one
/// parse of the headers and offset tables, e.g. to describe the image before decoding it) and decodes
/// part `part` with the very same parse. The part's header stays available as `mp.header(part)`.
///
/// # Errors
/// As [`read_planes`].
pub fn read_planes_part(
    mp: &mut MultiPartInputFile<'_>,
    part: i32,
    frame: Frame,
    dst: &mut [PlaneDst<'_>],
) -> Result<()> {
    if frame.width == 0 || frame.height == 0 {
        return Err(ExrError::Arg("read_planes: the frame is empty.".into()));
    }
    for (i, d) in dst.iter().enumerate() {
        check_plane(d, &frame)?;
        // One channel, one plane: a second destination for the same channel would be silently left
        // unwritten (the frame buffer keeps one slice per name).
        if dst[..i].iter().any(|e| e.channel == d.channel) {
            return Err(ExrError::Arg(format!(
                "read_planes: channel \"{}\" has more than one destination plane.",
                d.channel
            )));
        }
    }
    // The facts the decode needs, read off the parsed header (the borrow ends before the reader opens).
    let (present, dw, tiled) = {
        let header = mp.header(part)?;
        if header.has_type() && header.type_()?.starts_with("deep") {
            return Err(ExrError::Unsupported(Unsupported::Deep));
        }
        let channels = header.channels()?;
        for d in dst.iter() {
            if let Some(c) = channels.find_channel(d.channel)
                && (c.x_sampling != 1 || c.y_sampling != 1)
            {
                return Err(ExrError::Unsupported(Unsupported::Subsampled));
            }
        }
        let present: Vec<bool> = dst.iter().map(|d| channels.find_channel(d.channel).is_some()).collect();
        (present, header.data_window()?, header.has_tile_description())
    };

    // Frame rows / columns the data window covers.
    let (ox, oy) = (i64::from(frame.origin.x), i64::from(frame.origin.y));
    let (w, h) = (frame.width as i64, frame.height as i64);
    let (r0, r1) = (i64::from(dw.min.y).max(oy), i64::from(dw.max.y).min(oy + h - 1));
    let (c0, c1) = (i64::from(dw.min.x).max(ox), i64::from(dw.max.x).min(ox + w - 1));
    let covered = r0 <= r1 && c0 <= c1;

    // Margins first (they never overlap what the decode writes): zero, or `fill` for absent channels.
    for (d, &has) in dst.iter_mut().zip(&present) {
        let v = if has { 0.0 } else { d.fill };
        let covered_rows = covered.then(|| (r0 - oy)..=(r1 - oy));
        for y in 0..h {
            let in_rows = covered_rows.as_ref().is_some_and(|r| r.contains(&y));
            // Whole row outside the covered rows, else only the columns left / right of the data window.
            let spans: [(i64, i64); 2] = if in_rows { [(0, c0 - ox), (c1 - ox + 1, w)] } else { [(0, w), (0, 0)] };
            for (from, to) in spans {
                fill_span(d, y as usize, from as usize, to as usize, v)?;
            }
        }
    }
    if !covered {
        return Ok(());
    }

    let (y0, y1) = (r0 as i32, r1 as i32);
    let direct = !tiled && i64::from(dw.min.x) >= ox && i64::from(dw.max.x) < ox + w;
    let mut rd = mp.input_part(part)?;
    if direct {
        // One decode, straight into the callers' planes.
        let mut fb = FrameBuffer::new();
        for (i, d) in dst.iter().enumerate() {
            let base = -(ox * d.x_stride as i64 + oy * d.y_stride as i64);
            fb.insert_in(d.channel, slice(d.type_, base, d.x_stride, d.y_stride, d.fill), i)?;
        }
        rd.set_frame_buffer(&fb)?;
        let mut planes: Vec<&mut [u8]> = dst.iter_mut().map(|d| &mut *d.data).collect();
        rd.read_pixels_planes(y0, y1, &mut planes)?;
        return Ok(());
    }

    // Gather path: all planes tight at the data-window width in one temporary, then a clipped copy.
    let dww = (i64::from(dw.max.x) - i64::from(dw.min.x) + 1) as usize;
    let rows = (r1 - r0 + 1) as usize;
    let sizes: Vec<usize> = dst.iter().map(|d| pixel_type_size(d.type_)).collect::<Result<_>>()?;
    let mut starts = Vec::with_capacity(dst.len());
    let mut total = 0usize;
    for &s in &sizes {
        starts.push(total);
        total = s
            .checked_mul(dww)
            .and_then(|b| b.checked_mul(rows))
            .and_then(|b| total.checked_add(b))
            .ok_or_else(|| ExrError::Arg("read_planes: data window too large.".into()))?;
    }
    let mut temp: Vec<u8> = Vec::new();
    temp.try_reserve_exact(total)
        .map_err(|_| ExrError::Arg("read_planes: cannot allocate the gather buffer.".into()))?;
    temp.resize(total, 0);
    let mut fb = FrameBuffer::new();
    for (i, d) in dst.iter().enumerate() {
        let (s, ys) = (sizes[i], dww * sizes[i]);
        let base = starts[i] as i64 - (i64::from(dw.min.x) * s as i64 + r0 * ys as i64);
        fb.insert(d.channel, slice(d.type_, base, s, ys, d.fill))?;
    }
    rd.set_frame_buffer(&fb)?;
    rd.read_pixels(y0, y1, &mut temp)?;
    for (i, d) in dst.iter_mut().enumerate() {
        let s = sizes[i];
        for y in r0..=r1 {
            for x in c0..=c1 {
                let from = starts[i] + (((y - r0) as usize * dww) + (x - i64::from(dw.min.x)) as usize) * s;
                let to = (y - oy) as usize * d.y_stride + (x - ox) as usize * d.x_stride;
                d.data[to..to + s].copy_from_slice(&temp[from..from + s]);
            }
        }
    }
    Ok(())
}

fn slice(type_: PixelType, base: i64, x_stride: usize, y_stride: usize, fill: f64) -> Slice {
    Slice {
        type_,
        base_offset: base,
        x_stride,
        y_stride,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: fill,
        x_tile_coords: false,
        y_tile_coords: false,
    }
}

/// Reject a plane the frame cannot be addressed in (before anything is decoded or written).
fn check_plane(d: &PlaneDst<'_>, f: &Frame) -> Result<()> {
    if !matches!(d.type_, PixelType::Half | PixelType::Float) {
        return Err(ExrError::Arg(format!("read_planes: plane \"{}\" must be HALF or FLOAT.", d.channel)));
    }
    let s = pixel_type_size(d.type_)?;
    let need = (f.height - 1)
        .checked_mul(d.y_stride)
        .and_then(|a| (f.width - 1).checked_mul(d.x_stride).and_then(|b| a.checked_add(b)))
        .and_then(|a| a.checked_add(s));
    if need.is_none_or(|n| n > d.data.len()) {
        return Err(ExrError::Arg(format!(
            "read_planes: plane \"{}\" ({} bytes) is too small for a {}x{} frame (strides {}, {}).",
            d.channel,
            d.data.len(),
            f.width,
            f.height,
            d.x_stride,
            d.y_stride
        )));
    }
    Ok(())
}

/// Write `value` (as the plane's sample type) into samples `[from, to)` of frame row `y`.
fn fill_span(d: &mut PlaneDst<'_>, y: usize, from: usize, to: usize, value: f64) -> Result<()> {
    if from >= to {
        return Ok(());
    }
    let s = pixel_type_size(d.type_)?;
    let bytes: [u8; 4] = match d.type_ {
        PixelType::Half => {
            let b = imath_rs::Half::from(value as f32).bits().to_ne_bytes();
            [b[0], b[1], 0, 0]
        }
        _ => (value as f32).to_ne_bytes(),
    };
    let row = y * d.y_stride;
    if d.x_stride == s {
        // Packed row: one pass; a zero fill is the common case.
        let span = &mut d.data[row + from * s..row + to * s];
        if bytes[..s].iter().all(|&b| b == 0) {
            span.fill(0);
        } else {
            span.chunks_exact_mut(s).for_each(|c| c.copy_from_slice(&bytes[..s]));
        }
    } else {
        for x in from..to {
            let at = row + x * d.x_stride;
            d.data[at..at + s].copy_from_slice(&bytes[..s]);
        }
    }
    Ok(())
}
