//! Every channel of a flat part at its NATIVE sampling: the one reader of a
//! subsampled file as stored (C++ `Slice(type, base, xStride, yStride, xSampling,
//! ySampling)`), with no reconstruction.
//!
//! [`super::Image`] rejects subsampled files ([`super::Unsupported::Subsampled`]);
//! this is where they are read instead. exr-gpu uploads these planes as they are,
//! and exr-view shows them (a chroma image reconstructed by
//! [`crate::rgba::RgbaInputFile`], other channels scaled up to the data window).

use super::ChannelData;
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::misc::pixel_type_size;
use crate::scanline::GenericInputReader;

/// One channel as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    /// Full channel name as in the header (e.g. `"diffuse.R"`).
    pub name: String,
    /// `(x_sampling, y_sampling)`; `(1, 1)` = full resolution.
    pub sampling: (i32, i32),
    /// Samples per row: the data-window width at `(1, 1)`, else
    /// `num_samples(x_sampling, min.x, max.x)`.
    pub width: u32,
    /// Sample rows.
    pub height: u32,
    /// `width * height` samples, raster order.
    pub data: ChannelData,
}

impl Plane {
    /// The sample covering data-window pixel `(x, y)` (0-based within the window
    /// whose first pixel is `min`): sample `divp(x, xs) - divp(min.x, xs)` of its
    /// row, like the reader's own addressing. `None` outside the plane.
    pub fn at(&self, min: (i32, i32), x: u32, y: u32) -> Option<usize> {
        let (xs, ys) = self.sampling;
        let col = (i64::from(min.0) + i64::from(x)).div_euclid(i64::from(xs)) - i64::from(min.0).div_euclid(i64::from(xs));
        let row = (i64::from(min.1) + i64::from(y)).div_euclid(i64::from(ys)) - i64::from(min.1).div_euclid(i64::from(ys));
        let (col, row) = (usize::try_from(col).ok()?, usize::try_from(row).ok()?);
        (col < self.width as usize && row < self.height as usize).then_some(row * self.width as usize + col)
    }
}

/// Every channel of `reader`'s part at its native sampling, in header order.
///
/// # Errors
/// A channel with a sampling below 1, an extent that does not fit, or any read
/// error of the part.
pub fn planes<R: GenericInputReader>(reader: &mut R) -> Result<Vec<Plane>> {
    struct Layout {
        name: String,
        type_: crate::attr::PixelType,
        offset: usize,
        len: usize,
        width: u32,
        height: u32,
        sampling: (i32, i32),
    }
    let header = reader.header();
    let dw = header.data_window()?;
    let (min, max) = (dw.min, dw.max);

    let mut fb = FrameBuffer::new();
    let mut layout = Vec::new();
    let mut total = 0usize;
    for (name, ch) in header.channels()?.iter() {
        let (xs, ys) = (ch.x_sampling, ch.y_sampling);
        if xs < 1 || ys < 1 {
            return Err(ExrError::Arg(format!("channel \"{name}\" has sampling {xs}x{ys}")));
        }
        let (nx, ny) = super::store::sample_dims(dw, xs, ys)?;
        let (Ok(width), Ok(height)) = (u32::try_from(nx), u32::try_from(ny)) else {
            return Err(ExrError::Arg(format!("channel \"{name}\": {nx}x{ny} samples")));
        };
        let size = pixel_type_size(ch.type_)?;
        let row = width as usize * size;
        let len = row * height as usize;
        // Sample `(x, y)` lives at `base + divp(x, xs)*size + divp(y, ys)*row`
        // (C++'s frame-buffer addressing); put the window's first sample at this
        // plane's offset. Imath `divp` is floored division: `div_euclid` for xs > 0.
        let base = total as i64 - i64::from(min.x.div_euclid(xs)) * size as i64 - i64::from(min.y.div_euclid(ys)) * row as i64;
        fb.insert(
            name,
            Slice {
                type_: ch.type_,
                base_offset: base,
                x_stride: size,
                y_stride: row,
                x_sampling: xs,
                y_sampling: ys,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )?;
        layout.push(Layout { name: name.to_string(), type_: ch.type_, offset: total, len, width, height, sampling: (xs, ys) });
        total += len;
    }

    let mut buf = vec![0u8; total];
    reader.set_frame_buffer(&fb)?;
    reader.read_pixels(min.y, max.y, &mut buf)?;
    layout
        .into_iter()
        .map(|l| {
            let bytes = buf.get(l.offset..l.offset + l.len).ok_or_else(|| ExrError::Logic("plane outside the read buffer".into()))?;
            let data = ChannelData::from_ne_bytes(l.type_, bytes, l.width as usize * l.height as usize)?;
            Ok(Plane { name: l.name, sampling: l.sampling, width: l.width, height: l.height, data })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(sampling: (i32, i32), width: u32, height: u32) -> Plane {
        let data = ChannelData::Float(vec![0.0; width as usize * height as usize]);
        Plane { name: "S".into(), sampling, width, height, data }
    }

    /// The covering sample follows floored division of ABSOLUTE coordinates, so a
    /// window starting at an odd or negative coordinate splits its first 2x2 block.
    #[test]
    fn at_follows_absolute_floored_division() {
        // Window x in [-3, 2]: samples divp(-3,2)=-2 .. divp(2,2)=1 -> 4 columns.
        let p = plane((2, 1), 4, 1);
        let cols: Vec<Option<usize>> = (0..6).map(|x| p.at((-3, 0), x, 0)).collect();
        assert_eq!(cols, [Some(0), Some(1), Some(1), Some(2), Some(2), Some(3)]);
        // x = 6 (absolute 3) still falls in sample column 3; x = 8 (absolute 5) is past it.
        assert_eq!(p.at((-3, 0), 6, 0), Some(3));
        assert_eq!(p.at((-3, 0), 8, 0), None, "past the plane");
        let q = plane((1, 2), 1, 3);
        assert_eq!(q.at((0, 1), 0, 0), Some(0));
        assert_eq!(q.at((0, 1), 0, 1), Some(1), "y = 2 starts the next row pair");
    }
}
