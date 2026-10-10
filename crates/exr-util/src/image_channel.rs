//! Port of `ImfImageChannel.{h,cpp}` (`openexr/src/lib/OpenEXRUtil`, v3.4.15): what every image
//! channel has in common (sampling rates, per-channel pixel grid, bounds checks), plus the [`Pixel`]
//! trait that stands in for the C++ `half` / `float` / `unsigned int` template argument.
//!
//! Differences from the C++ class hierarchy, all forced by ownership:
//! - a channel has no `level()` back-reference; it keeps its own copy of the level's data window
//!   ([`ChannelGeom`]) and the owning level refreshes it on resize / shift;
//! - `ImageChannel` is a trait, implemented by the closed set of channel enums of this crate.

use exr_core::attr::{Channel, PixelType};
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, Half, V2i};

use crate::deep_image_channel::{DeepChannel, DeepImageChannel};
use crate::flat_image_channel::{FlatChannel, FlatImageChannel};

/// The empty data window `(0,0)-(-1,-1)` every C++ image / level starts with.
pub(crate) fn empty_dw() -> Box2i {
    Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: -1, y: -1 } }
}

/// Move a data window by `(dx, dy)`; an error (and no change) if a coordinate would overflow.
pub(crate) fn shift_box(dw: &mut Box2i, dx: i32, dy: i32) -> Result<()> {
    let over = || ExrError::Arg(format!("Cannot shift the data window by ({dx}, {dy}): the coordinates overflow."));
    let moved = Box2i {
        min: V2i { x: dw.min.x.checked_add(dx).ok_or_else(over)?, y: dw.min.y.checked_add(dy).ok_or_else(over)? },
        max: V2i { x: dw.max.x.checked_add(dx).ok_or_else(over)?, y: dw.max.y.checked_add(dy).ok_or_else(over)? },
    };
    *dw = moved;
    Ok(())
}

/// `vec![0; n]` that reports an allocation failure as an error, like the `std::bad_alloc` the C++
/// `new T[n]` throws, instead of aborting the process (`n` comes from file headers).
pub(crate) fn zeroed<T: Copy + Default>(n: usize) -> Result<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|e| ExrError::Arg(format!("Cannot allocate {n} image samples ({e}).")))?;
    v.resize(n, T::default());
    Ok(v)
}

/// A sample type a channel can hold: `half`, `float` or `unsigned int`.
pub trait Pixel: Copy + Default + PartialEq + std::fmt::Debug + Send + Sync + 'static {
    /// The file-format type of the samples.
    const TYPE: PixelType;
    /// Bytes per sample in the native-endian byte buffers the file readers and writers use.
    const SIZE: usize;

    /// Convert as many whole samples of native-endian bytes as fit into `dst`.
    fn read_ne(src: &[u8], dst: &mut [Self]);
    /// Convert samples into native-endian bytes, as many as fit into `dst`.
    fn write_ne(src: &[Self], dst: &mut [u8]);

    /// This type's flat channel inside `ch`, if `ch` holds such samples.
    fn flat(ch: &FlatImageChannel) -> Option<&FlatChannel<Self>>;
    /// Mutable form of [`Pixel::flat`].
    fn flat_mut(ch: &mut FlatImageChannel) -> Option<&mut FlatChannel<Self>>;
    /// This type's deep channel inside `ch`, if `ch` holds such samples.
    fn deep(ch: &DeepImageChannel) -> Option<&DeepChannel<Self>>;
    /// Mutable form of [`Pixel::deep`].
    fn deep_mut(ch: &mut DeepImageChannel) -> Option<&mut DeepChannel<Self>>;
}

macro_rules! impl_pixel {
    ($t:ty, $ty:expr, $n:literal, $variant:ident, |$b:ident| $from:expr, |$v:ident| $to:expr) => {
        impl Pixel for $t {
            const TYPE: PixelType = $ty;
            const SIZE: usize = $n;

            fn read_ne(src: &[u8], dst: &mut [Self]) {
                let (chunks, _) = src.as_chunks::<$n>();
                for (d, $b) in dst.iter_mut().zip(chunks) {
                    *d = $from;
                }
            }

            fn write_ne(src: &[Self], dst: &mut [u8]) {
                let (chunks, _) = dst.as_chunks_mut::<$n>();
                for (&$v, c) in src.iter().zip(chunks) {
                    *c = $to;
                }
            }

            fn flat(ch: &FlatImageChannel) -> Option<&FlatChannel<Self>> {
                match ch {
                    FlatImageChannel::$variant(c) => Some(c),
                    _ => None,
                }
            }

            fn flat_mut(ch: &mut FlatImageChannel) -> Option<&mut FlatChannel<Self>> {
                match ch {
                    FlatImageChannel::$variant(c) => Some(c),
                    _ => None,
                }
            }

            fn deep(ch: &DeepImageChannel) -> Option<&DeepChannel<Self>> {
                match ch {
                    DeepImageChannel::$variant(c) => Some(c),
                    _ => None,
                }
            }

            fn deep_mut(ch: &mut DeepImageChannel) -> Option<&mut DeepChannel<Self>> {
                match ch {
                    DeepImageChannel::$variant(c) => Some(c),
                    _ => None,
                }
            }
        }
    };
}

impl_pixel!(Half, PixelType::Half, 2, Half, |b| Half::from_bits(u16::from_ne_bytes(*b)), |v| v.bits().to_ne_bytes());
impl_pixel!(f32, PixelType::Float, 4, Float, |b| f32::from_ne_bytes(*b), |v| v.to_ne_bytes());
impl_pixel!(u32, PixelType::Uint, 4, Uint, |b| u32::from_ne_bytes(*b), |v| v.to_ne_bytes());

/// The state `ImageChannel` holds in C++: sampling rates, the `pLinear` flag, and the pixel grid
/// derived from the level's data window.
#[derive(Clone, Debug)]
pub struct ChannelGeom {
    x_sampling: i32,
    y_sampling: i32,
    p_linear: bool,
    pixels_per_row: i32,
    pixels_per_column: i32,
    num_pixels: usize,
    /// Copy of the owning level's data window (C++ reads it through `level()`).
    dw: Box2i,
}

impl ChannelGeom {
    /// `ImageChannel::ImageChannel`: an unallocated geometry; call [`ChannelGeom::resize`].
    pub(crate) fn new(x_sampling: i32, y_sampling: i32, p_linear: bool) -> Self {
        Self {
            x_sampling,
            y_sampling,
            p_linear,
            pixels_per_row: 0,
            pixels_per_column: 0,
            num_pixels: 0,
            dw: empty_dw(),
        }
    }

    /// `ImageChannel::resize`: adopt the level's data window `dw` and recompute the pixel grid.
    ///
    /// # Errors
    /// `Arg` for sampling below 1, a window origin or size that is not a multiple of the sampling
    /// rates, or a grid too large to index.
    pub(crate) fn resize(&mut self, dw: &Box2i) -> Result<()> {
        let (xs, ys) = (self.x_sampling as i64, self.y_sampling as i64);
        if xs < 1 || ys < 1 {
            return Err(ExrError::Arg("The x and y sampling rates for an image channel must be at least 1.".into()));
        }
        if dw.min.x as i64 % xs != 0 || dw.min.y as i64 % ys != 0 {
            return Err(ExrError::Arg(
                "The minimum x and y coordinates of the data window of an image level must be multiples of the x \
                 and y subsampling factors of all channels in the image."
                    .into(),
            ));
        }
        let width = dw.max.x as i64 - dw.min.x as i64 + 1;
        let height = dw.max.y as i64 - dw.min.y as i64 + 1;
        if width % xs != 0 || height % ys != 0 {
            return Err(ExrError::Arg(
                "The width and height of the data window of an image level must be multiples of the x and y \
                 subsampling factors of all channels in the image."
                    .into(),
            ));
        }
        let too_large = || ExrError::Arg("Image channel dimensions too large.".into());
        let per_row = i32::try_from(width / xs).map_err(|_| too_large())?;
        let per_col = i32::try_from(height / ys).map_err(|_| too_large())?;
        let num = (per_row.max(0) as u64).checked_mul(per_col.max(0) as u64).ok_or_else(too_large)?;
        self.num_pixels = usize::try_from(num).map_err(|_| too_large())?;
        self.pixels_per_row = per_row.max(0);
        self.pixels_per_column = per_col.max(0);
        self.dw = *dw;
        Ok(())
    }

    /// `resetBasePointer` after `ImageLevel::shiftPixels`: the window moved, the pixels did not.
    pub(crate) fn shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        shift_box(&mut self.dw, dx, dy)
    }

    /// `ImageChannel::boundsCheck`: `(x, y)` must lie in the data window and on the sampling grid.
    pub fn bounds_check(&self, x: i32, y: i32) -> Result<()> {
        let dw = &self.dw;
        if x < dw.min.x || x > dw.max.x || y < dw.min.y || y > dw.max.y {
            return Err(ExrError::Arg(format!(
                "Attempt to access a pixel at location ({x}, {y}) in an image whose data window is ({}, {}) - ({}, {}).",
                dw.min.x, dw.min.y, dw.max.x, dw.max.y
            )));
        }
        if x % self.x_sampling != 0 || y % self.y_sampling != 0 {
            return Err(ExrError::Arg(format!(
                "Attempt to access a pixel at location ({x}, {y}) in a channel whose x and y sampling rates are {} and \
                 {}.  The pixel coordinates are not divisible by the sampling rates.",
                self.x_sampling, self.y_sampling
            )));
        }
        Ok(())
    }

    /// Index of pixel `(x, y)` in row-major storage of the channel's own grid; the coordinates must
    /// have passed [`ChannelGeom::bounds_check`].
    pub(crate) fn idx(&self, x: i32, y: i32) -> Result<usize> {
        let (xs, ys) = (self.x_sampling as i64, self.y_sampling as i64);
        let col = x as i64 / xs - self.dw.min.x as i64 / xs;
        let row = y as i64 / ys - self.dw.min.y as i64 / ys;
        usize::try_from(row * self.pixels_per_row as i64 + col)
            .map_err(|_| ExrError::Logic(format!("pixel ({x}, {y}) maps to a negative index")))
    }

    /// The data window of the level this channel belongs to.
    pub fn data_window(&self) -> &Box2i {
        &self.dw
    }

    /// Samples per row of the channel's own grid.
    pub fn pixels_per_row(&self) -> i32 {
        self.pixels_per_row
    }

    /// Rows of the channel's own grid.
    pub fn pixels_per_column(&self) -> i32 {
        self.pixels_per_column
    }

    /// Samples (flat) or pixels (deep) the channel stores.
    pub fn num_pixels(&self) -> usize {
        self.num_pixels
    }
}

/// `class ImageChannel`: the read-only interface shared by flat, deep and sample-count channels.
pub trait ImageChannel {
    /// The shared geometry.
    fn geom(&self) -> &ChannelGeom;
    /// The type of the samples.
    fn pixel_type(&self) -> PixelType;

    /// `ImageChannel::channel`: the file-format description of this channel.
    fn channel(&self) -> Channel {
        Channel::new(self.pixel_type(), self.x_sampling(), self.y_sampling(), self.p_linear())
    }
    /// Horizontal sampling rate.
    fn x_sampling(&self) -> i32 {
        self.geom().x_sampling
    }
    /// Vertical sampling rate.
    fn y_sampling(&self) -> i32 {
        self.geom().y_sampling
    }
    /// The "perceptually approximately linear" flag.
    fn p_linear(&self) -> bool {
        self.geom().p_linear
    }
    /// Samples per row of the channel's own grid.
    fn pixels_per_row(&self) -> i32 {
        self.geom().pixels_per_row
    }
    /// Rows of the channel's own grid.
    fn pixels_per_column(&self) -> i32 {
        self.geom().pixels_per_column
    }
    /// Samples (flat) or pixels (deep) the channel stores.
    fn num_pixels(&self) -> usize {
        self.geom().num_pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dw(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
        Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
    }

    #[test]
    fn resize_rejects_bad_sampling_and_misaligned_windows() {
        let mut g = ChannelGeom::new(2, 1, false);
        assert!(g.resize(&dw(1, 0, 4, 3)).is_err(), "origin not a multiple of xSampling");
        assert!(g.resize(&dw(0, 0, 2, 3)).is_err(), "width 3 not a multiple of xSampling 2");
        g.resize(&dw(-2, 0, 5, 3)).unwrap();
        assert_eq!((g.pixels_per_row, g.pixels_per_column, g.num_pixels), (4, 4, 16));
        assert!(ChannelGeom::new(0, 1, false).resize(&dw(0, 0, 1, 1)).is_err());
    }

    #[test]
    fn bounds_check_window_and_grid() {
        let mut g = ChannelGeom::new(2, 2, false);
        g.resize(&dw(0, 0, 3, 3)).unwrap();
        assert!(g.bounds_check(2, 2).is_ok());
        assert!(g.bounds_check(1, 2).is_err(), "off the sampling grid");
        assert!(g.bounds_check(4, 0).is_err(), "outside the window");
        assert_eq!(g.idx(2, 2).unwrap(), 3);
    }

    #[test]
    fn pixel_bytes_round_trip() {
        let src = [Half::from_f32(1.5), Half::from_f32(-2.0)];
        let mut bytes = [0u8; 4];
        Half::write_ne(&src, &mut bytes);
        let mut back = [Half::default(); 2];
        Half::read_ne(&bytes, &mut back);
        assert_eq!(back[0].bits(), src[0].bits());
        assert_eq!(back[1].bits(), src[1].bits());
        let mut f = [0f32; 1];
        f32::read_ne(&3.25f32.to_ne_bytes(), &mut f);
        assert_eq!(f[0], 3.25);
    }
}
