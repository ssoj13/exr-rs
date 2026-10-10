//! Port of `ImfDeepImageChannel.{h,cpp}`: a channel that stores a list of samples per pixel.
//!
//! `TypedDeepImageChannel<T>` is [`DeepChannel<T>`] and the base class `DeepImageChannel` is the
//! enum [`DeepImageChannel`]. A channel owns only its sample buffer; the sample counts and list
//! positions live in the level's [`SampleCountChannel`], so reading a pixel's samples goes through
//! a [`DeepView`] that pairs the two (`level.typed_channel::<T>(name)`).

use std::collections::BTreeMap;

use exr_core::attr::{Channel, Name, PixelType};
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, Half};

use crate::image_channel::{zeroed, ChannelGeom, ImageChannel, Pixel};
use crate::sample_count_channel::{Layout, SampleCountChannel, SampleStore};

/// A staged sample buffer of any sample type (see [`SampleStore::stage`]).
pub(crate) enum DeepBuf {
    Half(Vec<Half>),
    Float(Vec<f32>),
    Uint(Vec<u32>),
}

/// The room a list needs: `base..base + n` of `buf`, or a logic error if the layout is inconsistent.
fn span<T>(buf: &[T], base: usize, n: usize) -> Result<&[T]> {
    buf.get(base..base.saturating_add(n))
        .ok_or_else(|| ExrError::Logic(format!("sample list {base}+{n} is outside a buffer of {}", buf.len())))
}

fn span_mut<T>(buf: &mut [T], base: usize, n: usize) -> Result<&mut [T]> {
    let len = buf.len();
    buf.get_mut(base..base.saturating_add(n))
        .ok_or_else(|| ExrError::Logic(format!("sample list {base}+{n} is outside a buffer of {len}")))
}

/// `TypedDeepImageChannel<T>`: the samples of every pixel, all lists in one buffer.
#[derive(Clone, Debug)]
pub struct DeepChannel<T: Pixel> {
    geom: ChannelGeom,
    samples: Vec<T>,
}

impl<T: Pixel> DeepChannel<T> {
    /// A channel for a level with data window `dw` whose lists need `buffer_size` entries.
    pub(crate) fn new(ch: &Channel, dw: &Box2i, buffer_size: usize) -> Result<Self> {
        let mut c = Self { geom: ChannelGeom::new(1, 1, ch.p_linear), samples: Vec::new() };
        c.resize(dw, buffer_size)?;
        Ok(c)
    }

    /// `resize`: new data window and a zeroed sample buffer of `buffer_size` entries.
    pub(crate) fn resize(&mut self, dw: &Box2i, buffer_size: usize) -> Result<()> {
        self.samples = Vec::new();
        self.geom.resize(dw)?;
        self.samples = zeroed(buffer_size)?;
        Ok(())
    }

    /// The whole sample buffer: lists at the positions [`SampleCountChannel`] reports.
    pub fn samples(&self) -> &[T] {
        &self.samples
    }

    /// The whole sample buffer, mutable.
    pub fn samples_mut(&mut self) -> &mut [T] {
        &mut self.samples
    }

    fn stage_buf(&self, old: Option<Layout<'_>>, new: Layout<'_>) -> Result<Vec<T>> {
        let mut buf: Vec<T> = zeroed(new.size)?;
        if let Some(old) = old {
            for ((&on, &op), (&nn, &np)) in old.num.iter().zip(old.pos).zip(new.num.iter().zip(new.pos)) {
                let k = on.min(nn) as usize;
                span_mut(&mut buf, np, k)?.copy_from_slice(span(&self.samples, op, k)?);
            }
        }
        Ok(buf)
    }

    /// Decode the samples of all pixels from native-endian `bytes`, the lists packed back to back in
    /// pixel order (the flat heap layout of the file readers), into their places in the buffer.
    pub fn load_lists(&mut self, sc: &SampleCountChannel, bytes: &[u8]) -> Result<()> {
        let mut src = 0usize;
        for (&n, &pos) in sc.num_samples().iter().zip(sc.sample_list_positions()) {
            let n = n as usize;
            let end = src.checked_add(n * T::SIZE).ok_or_else(|| ExrError::Logic("sample heap overflow".into()))?;
            let b = bytes.get(src..end).ok_or_else(|| ExrError::Logic("sample heap is too short".into()))?;
            T::read_ne(b, span_mut(&mut self.samples, pos, n)?);
            src = end;
        }
        Ok(())
    }

    /// Encode the samples of the pixels in the rectangle of `cols x rows` pixels whose top-left
    /// pixel is `(col0, row0)` as native-endian bytes, the lists packed back to back in raster
    /// order, into `out` (the inverse of [`DeepChannel::load_lists`] for a whole level).
    pub fn store_lists(
        &self,
        sc: &SampleCountChannel,
        (col0, row0, cols, rows): (usize, usize, usize, usize),
        out: &mut [u8],
    ) -> Result<()> {
        let w = self.geom.pixels_per_row() as usize;
        let mut dst = 0usize;
        for r in row0..row0 + rows {
            for c in col0..col0 + cols {
                let (pos, n) = list_of(sc, r * w + c)?;
                let end = dst.checked_add(n * T::SIZE).ok_or_else(|| ExrError::Logic("sample heap overflow".into()))?;
                let o = out.get_mut(dst..end).ok_or_else(|| ExrError::Logic("sample heap is too short".into()))?;
                T::write_ne(span(&self.samples, pos, n)?, o);
                dst = end;
            }
        }
        Ok(())
    }
}

impl<T: Pixel> ImageChannel for DeepChannel<T> {
    fn geom(&self) -> &ChannelGeom {
        &self.geom
    }

    fn pixel_type(&self) -> PixelType {
        T::TYPE
    }
}

/// `DeepImageChannel`: a deep channel of any of the three sample types.
#[derive(Clone, Debug)]
pub enum DeepImageChannel {
    /// `DeepHalfChannel`.
    Half(DeepChannel<Half>),
    /// `DeepFloatChannel`.
    Float(DeepChannel<f32>),
    /// `DeepUIntChannel`.
    Uint(DeepChannel<u32>),
}

macro_rules! each {
    ($ch:expr, $c:ident => $e:expr) => {
        match $ch {
            DeepImageChannel::Half($c) => $e,
            DeepImageChannel::Float($c) => $e,
            DeepImageChannel::Uint($c) => $e,
        }
    };
}

impl DeepImageChannel {
    /// `DeepImageLevel::insertChannel`'s `switch (type)`.
    pub(crate) fn new(ch: &Channel, dw: &Box2i, buffer_size: usize) -> Result<Self> {
        match ch.type_ {
            PixelType::Half => DeepChannel::new(ch, dw, buffer_size).map(Self::Half),
            PixelType::Float => DeepChannel::new(ch, dw, buffer_size).map(Self::Float),
            PixelType::Uint => DeepChannel::new(ch, dw, buffer_size).map(Self::Uint),
            PixelType::NumPixelTypes => Err(ExrError::Arg("Unknown pixel type.".into())),
        }
    }

    pub(crate) fn resize(&mut self, dw: &Box2i, buffer_size: usize) -> Result<()> {
        each!(self, c => c.resize(dw, buffer_size))
    }

    pub(crate) fn shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        each!(self, c => c.geom.shift(dx, dy))
    }

    /// Replace the sample buffer by a staged one of the same sample type.
    fn install(&mut self, buf: DeepBuf) -> Result<()> {
        match (self, buf) {
            (Self::Half(c), DeepBuf::Half(b)) => c.samples = b,
            (Self::Float(c), DeepBuf::Float(b)) => c.samples = b,
            (Self::Uint(c), DeepBuf::Uint(b)) => c.samples = b,
            _ => return Err(ExrError::Logic("a staged sample buffer has the wrong sample type".into())),
        }
        Ok(())
    }

    /// See [`DeepChannel::load_lists`].
    pub fn load_lists(&mut self, sc: &SampleCountChannel, bytes: &[u8]) -> Result<()> {
        each!(self, c => c.load_lists(sc, bytes))
    }

    /// See [`DeepChannel::store_lists`].
    pub fn store_lists(&self, sc: &SampleCountChannel, rect: (usize, usize, usize, usize), out: &mut [u8]) -> Result<()> {
        each!(self, c => c.store_lists(sc, rect, out))
    }
}

impl ImageChannel for DeepImageChannel {
    fn geom(&self) -> &ChannelGeom {
        each!(self, c => &c.geom)
    }

    fn pixel_type(&self) -> PixelType {
        each!(self, c => c.pixel_type())
    }
}

impl SampleStore for BTreeMap<Name, DeepImageChannel> {
    fn set_zero(&mut self, pos: usize, old_n: u32, new_n: u32) -> Result<()> {
        for ch in self.values_mut() {
            each!(ch, c => span_mut(&mut c.samples, pos + old_n as usize, (new_n - old_n) as usize)?.fill(Default::default()));
        }
        Ok(())
    }

    fn move_list(&mut self, old_pos: usize, old_n: u32, new_n: u32, new_pos: usize) -> Result<()> {
        let k = old_n.min(new_n) as usize;
        for ch in self.values_mut() {
            each!(ch, c => {
                // Validate both ranges first: `copy_within` panics on a bad one.
                span(&c.samples, old_pos, k)?;
                span(&c.samples, new_pos, new_n as usize)?;
                c.samples.copy_within(old_pos..old_pos + k, new_pos);
                span_mut(&mut c.samples, new_pos + k, (new_n as usize) - k)?.fill(Default::default());
            });
        }
        Ok(())
    }

    fn stage(&self, old: Option<Layout<'_>>, new: Layout<'_>) -> Result<Vec<DeepBuf>> {
        self.values()
            .map(|ch| {
                Ok(match ch {
                    DeepImageChannel::Half(c) => DeepBuf::Half(c.stage_buf(old, new)?),
                    DeepImageChannel::Float(c) => DeepBuf::Float(c.stage_buf(old, new)?),
                    DeepImageChannel::Uint(c) => DeepBuf::Uint(c.stage_buf(old, new)?),
                })
            })
            .collect()
    }

    fn commit(&mut self, bufs: Vec<DeepBuf>) -> Result<()> {
        // `stage` made the buffers from this very map, in this order; anything else is a bug, caught
        // before a single channel is touched.
        let matches = |ch: &DeepImageChannel, buf: &DeepBuf| {
            matches!(
                (ch, buf),
                (DeepImageChannel::Half(_), DeepBuf::Half(_))
                    | (DeepImageChannel::Float(_), DeepBuf::Float(_))
                    | (DeepImageChannel::Uint(_), DeepBuf::Uint(_))
            )
        };
        if bufs.len() != self.len() || !self.values().zip(&bufs).all(|(c, b)| matches(c, b)) {
            return Err(ExrError::Logic("staged sample buffers do not match the channels of the level".into()));
        }
        for (ch, buf) in self.values_mut().zip(bufs) {
            ch.install(buf)?;
        }
        Ok(())
    }
}

/// A deep channel paired with its level's sample counts: `level.typed_channel::<T>(name)`.
pub struct DeepView<'a, T: Pixel> {
    pub(crate) ch: &'a DeepChannel<T>,
    pub(crate) sc: &'a SampleCountChannel,
}

impl<'a, T: Pixel> DeepView<'a, T> {
    /// `at (x, y)`: the samples of pixel `(x, y)`.
    ///
    /// # Errors
    /// `Arg` if `(x, y)` is outside the data window.
    pub fn at(&self, x: i32, y: i32) -> Result<&'a [T]> {
        self.ch.geom.bounds_check(x, y)?;
        let i = self.ch.geom.idx(x, y)?;
        list_of(self.sc, i).and_then(|(pos, n)| span(&self.ch.samples, pos, n))
    }

    /// The channel itself.
    pub fn channel(&self) -> &'a DeepChannel<T> {
        self.ch
    }
}

/// Mutable form of [`DeepView`]: `level.typed_channel_mut::<T>(name)`.
pub struct DeepViewMut<'a, T: Pixel> {
    pub(crate) ch: &'a mut DeepChannel<T>,
    pub(crate) sc: &'a SampleCountChannel,
}

impl<T: Pixel> DeepViewMut<'_, T> {
    /// `at (x, y)`: the samples of pixel `(x, y)`, writable (their number is the sample count).
    pub fn at_mut(&mut self, x: i32, y: i32) -> Result<&mut [T]> {
        self.ch.geom.bounds_check(x, y)?;
        let i = self.ch.geom.idx(x, y)?;
        let (pos, n) = list_of(self.sc, i)?;
        span_mut(&mut self.ch.samples, pos, n)
    }
}

/// Start and length of pixel `i`'s sample list.
fn list_of(sc: &SampleCountChannel, i: usize) -> Result<(usize, usize)> {
    match (sc.sample_list_positions().get(i), sc.num_samples().get(i)) {
        (Some(&p), Some(&n)) => Ok((p, n as usize)),
        _ => Err(ExrError::Logic(format!("pixel {i} has no sample list"))),
    }
}
