//! Port of `ImfFlatImageChannel.{h,cpp}`: a channel that stores at most one sample per pixel.
//!
//! `TypedFlatImageChannel<T>` is [`FlatChannel<T>`]; the C++ base class `FlatImageChannel` (which
//! code holds when the sample type is not known statically) is the enum [`FlatImageChannel`].
//!
//! C++ also has `slice()`, a frame-buffer slice that points into the channel's storage. Slices here
//! are pointer-free descriptors over caller byte buffers, so the file I/O code
//! ([`crate::flat_image_io`]) describes strip buffers instead and moves samples with
//! [`FlatImageChannel::load_rect`] / [`FlatImageChannel::store_rect`].

use exr_core::attr::{Channel, PixelType};
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, Half};

use crate::image_channel::{zeroed, ChannelGeom, ImageChannel, Pixel};

/// `TypedFlatImageChannel<T>`: `pixelsPerRow x pixelsPerColumn` samples of type `T`, row-major, for
/// the pixels of the level's data window that lie on the channel's sampling grid.
#[derive(Clone, Debug)]
pub struct FlatChannel<T: Pixel> {
    geom: ChannelGeom,
    pixels: Vec<T>,
}

impl<T: Pixel> FlatChannel<T> {
    /// A channel for a level with data window `dw`; all samples are zero.
    pub fn new(ch: &Channel, dw: &Box2i) -> Result<Self> {
        let mut c = Self { geom: ChannelGeom::new(ch.x_sampling, ch.y_sampling, ch.p_linear), pixels: Vec::new() };
        c.resize(dw)?;
        Ok(c)
    }

    /// `TypedFlatImageChannel::resize`: new data window, reallocated zeroed storage.
    pub(crate) fn resize(&mut self, dw: &Box2i) -> Result<()> {
        self.pixels = Vec::new();
        self.geom.resize(dw)?;
        self.pixels = zeroed(self.geom.num_pixels())?;
        Ok(())
    }

    /// `operator() / at`: the sample of pixel `(x, y)`.
    ///
    /// # Errors
    /// `Arg` if `(x, y)` is outside the data window or off the channel's sampling grid.
    pub fn at(&self, x: i32, y: i32) -> Result<&T> {
        self.geom.bounds_check(x, y)?;
        let i = self.geom.idx(x, y)?;
        self.pixels.get(i).ok_or_else(|| ExrError::Logic(format!("pixel ({x}, {y}) is outside the channel storage")))
    }

    /// Mutable form of [`FlatChannel::at`].
    pub fn at_mut(&mut self, x: i32, y: i32) -> Result<&mut T> {
        self.geom.bounds_check(x, y)?;
        let i = self.geom.idx(x, y)?;
        self.pixels
            .get_mut(i)
            .ok_or_else(|| ExrError::Logic(format!("pixel ({x}, {y}) is outside the channel storage")))
    }

    /// `row (r)`: the samples of row `r` (0-based, in the channel's own grid), if it exists.
    pub fn row(&self, r: i32) -> Option<&[T]> {
        let w = self.geom.pixels_per_row() as usize;
        let a = usize::try_from(r).ok()?.checked_mul(w)?;
        self.pixels.get(a..a.checked_add(w)?)
    }

    /// Mutable form of [`FlatChannel::row`].
    pub fn row_mut(&mut self, r: i32) -> Option<&mut [T]> {
        let w = self.geom.pixels_per_row() as usize;
        let a = usize::try_from(r).ok()?.checked_mul(w)?;
        self.pixels.get_mut(a..a.checked_add(w)?)
    }

    /// All samples, row-major.
    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }

    /// All samples, row-major, mutable.
    pub fn pixels_mut(&mut self) -> &mut [T] {
        &mut self.pixels
    }

    /// Whether the rectangle lies inside the channel (checked before anything is copied).
    fn rect_fits(&self, col0: usize, row0: usize, cols: usize, rows: usize) -> bool {
        let (w, h) = (self.geom.pixels_per_row() as usize, self.geom.pixels_per_column() as usize);
        col0.checked_add(cols).is_some_and(|c| c <= w) && row0.checked_add(rows).is_some_and(|r| r <= h)
    }

    /// Fill the rectangle of `cols x rows` samples whose top-left sample is `(col0, row0)` (in the
    /// channel's own grid) from native-endian `bytes`, rows packed back to back. The decode half of
    /// the file readers' strip copy; the rectangle must lie inside the channel.
    pub fn load_rect(&mut self, col0: usize, row0: usize, cols: usize, rows: usize, bytes: &[u8]) -> Result<()> {
        let w = self.geom.pixels_per_row() as usize;
        if bytes.len() < cols * rows * T::SIZE || !self.rect_fits(col0, row0, cols, rows) {
            return Err(rect_error(col0, row0, cols, rows));
        }
        let src_rows = bytes.chunks(cols.max(1) * T::SIZE).take(rows);
        for (r, src) in src_rows.enumerate() {
            let a = (row0 + r) * w + col0;
            let dst = self.pixels.get_mut(a..a + cols).ok_or_else(|| rect_error(col0, row0, cols, rows))?;
            T::read_ne(src, dst);
        }
        Ok(())
    }

    /// Write the rectangle of `cols x rows` samples whose top-left sample is `(col0, row0)` as
    /// native-endian bytes, rows packed back to back, into `out`; the encode half of the file
    /// writers' strip copy.
    pub fn store_rect(&self, col0: usize, row0: usize, cols: usize, rows: usize, out: &mut [u8]) -> Result<()> {
        let w = self.geom.pixels_per_row() as usize;
        if out.len() < cols * rows * T::SIZE || !self.rect_fits(col0, row0, cols, rows) {
            return Err(rect_error(col0, row0, cols, rows));
        }
        let dst_rows = out.chunks_mut(cols.max(1) * T::SIZE).take(rows);
        for (r, dst) in dst_rows.enumerate() {
            let a = (row0 + r) * w + col0;
            let src = self.pixels.get(a..a + cols).ok_or_else(|| rect_error(col0, row0, cols, rows))?;
            T::write_ne(src, dst);
        }
        Ok(())
    }
}

fn rect_error(col0: usize, row0: usize, cols: usize, rows: usize) -> ExrError {
    ExrError::Logic(format!(
        "the {cols}x{rows} sample rectangle at ({col0}, {row0}) does not fit the channel or the byte buffer"
    ))
}

impl<T: Pixel> ImageChannel for FlatChannel<T> {
    fn geom(&self) -> &ChannelGeom {
        &self.geom
    }

    fn pixel_type(&self) -> PixelType {
        T::TYPE
    }
}

/// `FlatImageChannel`: a flat channel of any of the three sample types.
#[derive(Clone, Debug)]
pub enum FlatImageChannel {
    /// `FlatHalfChannel`.
    Half(FlatChannel<Half>),
    /// `FlatFloatChannel`.
    Float(FlatChannel<f32>),
    /// `FlatUIntChannel`.
    Uint(FlatChannel<u32>),
}

/// Apply an expression to the typed channel inside a [`FlatImageChannel`].
macro_rules! each {
    ($ch:expr, $c:ident => $e:expr) => {
        match $ch {
            FlatImageChannel::Half($c) => $e,
            FlatImageChannel::Float($c) => $e,
            FlatImageChannel::Uint($c) => $e,
        }
    };
}

impl FlatImageChannel {
    /// `FlatImageLevel::insertChannel`'s `switch (type)`: a zeroed channel for window `dw`.
    ///
    /// # Errors
    /// `Arg` for an unknown pixel type, invalid sampling, or a window that does not fit the sampling.
    pub fn new(ch: &Channel, dw: &Box2i) -> Result<Self> {
        match ch.type_ {
            PixelType::Half => FlatChannel::new(ch, dw).map(Self::Half),
            PixelType::Float => FlatChannel::new(ch, dw).map(Self::Float),
            PixelType::Uint => FlatChannel::new(ch, dw).map(Self::Uint),
            PixelType::NumPixelTypes => Err(ExrError::Arg("Unknown pixel type.".into())),
        }
    }

    pub(crate) fn resize(&mut self, dw: &Box2i) -> Result<()> {
        each!(self, c => c.resize(dw))
    }

    pub(crate) fn shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        each!(self, c => c.geom.shift(dx, dy))
    }

    /// See [`FlatChannel::load_rect`].
    pub fn load_rect(&mut self, col0: usize, row0: usize, cols: usize, rows: usize, bytes: &[u8]) -> Result<()> {
        each!(self, c => c.load_rect(col0, row0, cols, rows, bytes))
    }

    /// See [`FlatChannel::store_rect`].
    pub fn store_rect(&self, col0: usize, row0: usize, cols: usize, rows: usize, out: &mut [u8]) -> Result<()> {
        each!(self, c => c.store_rect(col0, row0, cols, rows, out))
    }
}

impl ImageChannel for FlatImageChannel {
    fn geom(&self) -> &ChannelGeom {
        each!(self, c => &c.geom)
    }

    fn pixel_type(&self) -> PixelType {
        each!(self, c => c.pixel_type())
    }
}
