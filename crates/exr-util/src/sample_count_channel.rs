//! Port of `ImfSampleCountChannel.{h,cpp}`: the per-pixel sample counts of a deep image level, and
//! the bookkeeping of where each pixel's sample list lives in the level's channels.
//!
//! Every deep channel of a level keeps all its samples in one buffer of `sample_buffer_size`
//! entries. Pixel `i`'s list starts at `sample_list_positions[i]`, holds `num_samples[i]` samples
//! and has room for `sample_list_sizes[i]` (the count rounded up to a power of two), so a list can
//! usually grow in place; growing past its room moves the list to fresh space at the end, and a
//! full buffer is rebuilt with 50% slack. This is the C++ layout, with two differences forced by
//! ownership:
//! - C++ keeps a `T*` per pixel and channel; the pointer always equals `buffer + position`, so the
//!   positions here are the only state;
//! - the channel cannot reach the other channels of its level (C++ goes through `deepLevel()`), so
//!   every mutating operation takes a `SampleStore`, implemented by the level's channel map, and
//!   users call the same-named methods on the level.
//!
//! Buffer rebuilds stage every channel's new buffer first and install them only when all
//! allocations succeeded, so a failed allocation leaves the level unchanged (C++ empties the whole
//! image in that case).

use exr_core::attr::PixelType;
use exr_core::{ExrError, Result};
use imath_rs::Box2i;

use crate::deep_image_channel::DeepBuf;
use crate::image_channel::{zeroed, ChannelGeom, ImageChannel};

/// `roundListSizeUp`: the room reserved for a list of `n` samples.
fn round_list_size_up(n: u32) -> Result<u32> {
    if n == 0 {
        return Ok(0);
    }
    if n > 0x8000_0000 {
        return Err(ExrError::Arg(format!("Sample count {n} is too large (cannot round up to next power of two)")));
    }
    Ok(n.next_power_of_two())
}

/// `roundBufferSizeUp`.
fn round_buffer_size_up(n: usize) -> usize {
    n + n / 2
}

/// Where the sample lists of a level sit: the state of [`SampleCountChannel`] that the channels need.
#[derive(Clone, Copy)]
pub(crate) struct Layout<'a> {
    /// Samples per pixel.
    pub num: &'a [u32],
    /// First sample of each pixel's list.
    pub pos: &'a [usize],
    /// Entries in a channel's sample buffer.
    pub size: usize,
}

/// The deep channels of a level, as seen by [`SampleCountChannel`] (`DeepImageLevel::setSamplesToZero`
/// and friends in C++).
pub(crate) trait SampleStore {
    /// Zero the samples `old_n..new_n` of pixel list `pos`, which grew inside its reserved room.
    fn set_zero(&mut self, pos: usize, old_n: u32, new_n: u32) -> Result<()>;
    /// Move a list from `old_pos` to `new_pos` (past every used entry), keeping `min (old_n, new_n)`
    /// samples and zeroing the rest.
    fn move_list(&mut self, old_pos: usize, old_n: u32, new_n: u32, new_pos: usize) -> Result<()>;
    /// Allocate every channel's buffer for `new`, carrying the samples over from `old` when given
    /// (zeroed lists otherwise). Nothing is changed; pass the result to [`SampleStore::commit`].
    fn stage(&self, old: Option<Layout<'_>>, new: Layout<'_>) -> Result<Vec<DeepBuf>>;
    /// Install the buffers staged by [`SampleStore::stage`].
    fn commit(&mut self, bufs: Vec<DeepBuf>) -> Result<()>;
}

/// `SampleCountChannel`: a `UINT` channel with one count per pixel, plus the list bookkeeping.
#[derive(Clone, Debug)]
pub struct SampleCountChannel {
    geom: ChannelGeom,
    num: Vec<u32>,
    sizes: Vec<u32>,
    pos: Vec<usize>,
    total_num: usize,
    total_occupied: usize,
    buffer_size: usize,
}

impl ImageChannel for SampleCountChannel {
    fn geom(&self) -> &ChannelGeom {
        &self.geom
    }

    fn pixel_type(&self) -> PixelType {
        PixelType::Uint
    }
}

impl SampleCountChannel {
    /// An unallocated channel; call [`SampleCountChannel::resize`].
    pub(crate) fn new() -> Self {
        Self {
            geom: ChannelGeom::new(1, 1, false),
            num: Vec::new(),
            sizes: Vec::new(),
            pos: Vec::new(),
            total_num: 0,
            total_occupied: 0,
            buffer_size: 0,
        }
    }

    /// `SampleCountChannel::resize`: new data window, all counts zero.
    pub(crate) fn resize(&mut self, dw: &Box2i) -> Result<()> {
        *self = Self::new();
        self.geom.resize(dw)?;
        let n = self.geom.num_pixels();
        self.num = zeroed(n)?;
        self.sizes = zeroed(n)?;
        self.pos = zeroed(n)?;
        Ok(())
    }

    pub(crate) fn shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        self.geom.shift(dx, dy)
    }

    /// `operator() / at`: the sample count of pixel `(x, y)`.
    pub fn at(&self, x: i32, y: i32) -> Result<u32> {
        self.geom.bounds_check(x, y)?;
        let i = self.geom.idx(x, y)?;
        self.num.get(i).copied().ok_or_else(|| ExrError::Logic(format!("pixel ({x}, {y}) is outside the counts")))
    }

    /// `row (r)`: the sample counts of row `r`.
    pub fn row(&self, r: i32) -> Option<&[u32]> {
        let w = self.geom.pixels_per_row() as usize;
        let a = usize::try_from(r).ok()?.checked_mul(w)?;
        self.num.get(a..a.checked_add(w)?)
    }

    /// `numSamples`: every pixel's sample count, row-major.
    pub fn num_samples(&self) -> &[u32] {
        &self.num
    }

    /// `sampleListSizes`: the room reserved for each pixel's list.
    pub fn sample_list_sizes(&self) -> &[u32] {
        &self.sizes
    }

    /// `sampleListPositions`: the first sample of each pixel's list in a channel's sample buffer.
    pub fn sample_list_positions(&self) -> &[usize] {
        &self.pos
    }

    /// `sampleBufferSize`: entries in each channel's sample buffer.
    pub fn sample_buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// The sum of all sample counts.
    pub fn total_samples(&self) -> usize {
        self.total_num
    }

    /// Room, positions and totals of lists sized for `num` (what `endEdit` and the rebuild branch of
    /// `set` compute): `(sizes, positions, occupied, total)`.
    fn layout_for(num: &[u32]) -> Result<(Vec<u32>, Vec<usize>, usize, usize)> {
        let overflow = || ExrError::Arg("Total sample count of a deep image level is too large.".into());
        let mut sizes: Vec<u32> = zeroed(num.len())?;
        let mut pos: Vec<usize> = zeroed(num.len())?;
        let (mut occupied, mut total) = (0usize, 0usize);
        for (i, &n) in num.iter().enumerate() {
            let size = round_list_size_up(n)?;
            sizes[i] = size;
            pos[i] = occupied;
            total = total.checked_add(n as usize).ok_or_else(overflow)?;
            occupied = occupied.checked_add(size as usize).ok_or_else(overflow)?;
        }
        Ok((sizes, pos, occupied, total))
    }

    /// Recompute the layout for `new_num` (or the current counts), stage the channels' buffers and
    /// install everything. `carry` keeps the existing samples (the rebuild branch of `set`);
    /// otherwise the lists start zeroed (`endEdit`, `clear`).
    fn rebuild(&mut self, store: &mut dyn SampleStore, new_num: Option<Vec<u32>>, carry: bool) -> Result<()> {
        let num = new_num.as_deref().unwrap_or(&self.num);
        let (sizes, pos, occupied, total) = Self::layout_for(num)?;
        let size = round_buffer_size_up(occupied);
        let old = carry.then_some(Layout { num: &self.num, pos: &self.pos, size: self.buffer_size });
        let bufs = store.stage(old, Layout { num, pos: &pos, size })?;
        store.commit(bufs)?; // last fallible step: the bookkeeping below cannot fail
        if let Some(n) = new_num {
            self.num = n;
        }
        self.sizes = sizes;
        self.pos = pos;
        self.total_num = total;
        self.total_occupied = occupied;
        self.buffer_size = size;
        Ok(())
    }

    /// `set (x, y, n)`: change the sample count of one pixel, keeping its samples (new ones are zero).
    pub(crate) fn set(&mut self, x: i32, y: i32, n: u32, store: &mut dyn SampleStore) -> Result<()> {
        self.geom.bounds_check(x, y)?;
        let i = self.geom.idx(x, y)?;
        let cur = *self.num.get(i).ok_or_else(|| ExrError::Logic("pixel outside the counts".into()))?;
        if n <= cur {
            self.total_num -= (cur - n) as usize;
            self.num[i] = n;
            return Ok(());
        }
        if n <= self.sizes[i] {
            store.set_zero(self.pos[i], cur, n)?;
            self.total_num += (n - cur) as usize;
            self.num[i] = n;
            return Ok(());
        }
        let new_size = round_list_size_up(n)? as usize;
        if self.total_occupied + new_size <= self.buffer_size {
            store.move_list(self.pos[i], cur, n, self.total_occupied)?;
            self.pos[i] = self.total_occupied;
            // The C++ code forgets to record the new room (`ImfSampleCountChannel.cpp:113-119`), so
            // the next `set` on this pixel moves the list again although it would fit.
            self.sizes[i] = new_size as u32;
            self.total_occupied += new_size;
            self.total_num += (n - cur) as usize;
            self.num[i] = n;
            return Ok(());
        }
        let mut new_num = self.num.clone();
        new_num[i] = n;
        self.rebuild(store, Some(new_num), true)
    }

    /// `set (r, counts)`: set the counts of a whole row of the level.
    pub(crate) fn set_row(&mut self, r: i32, counts: &[u32], store: &mut dyn SampleStore) -> Result<()> {
        if r < 0 || r >= self.geom.pixels_per_column() {
            return Err(ExrError::Arg(format!(
                "Attempt to set sample counts for row {r} in an image channel with {} rows.",
                self.geom.pixels_per_column()
            )));
        }
        let (mut x, y) = (self.geom.data_window().min.x, r + self.geom.data_window().min.y);
        let w = self.geom.pixels_per_row() as usize;
        if counts.len() < w {
            return Err(ExrError::Arg(format!("Attempt to set the sample counts of a row of {w} pixels from {} counts.", counts.len())));
        }
        for &n in &counts[..w] {
            self.set(x, y, n, store)?;
            x += 1;
        }
        Ok(())
    }

    /// Give up on a failed rebuild: every count zero, empty lists. C++ empties the whole image
    /// instead (`ImfSampleCountChannel.cpp:136`); an empty-list stage allocates nothing, so it cannot
    /// fail in practice.
    fn reset(&mut self, store: &mut dyn SampleStore) {
        self.num.fill(0);
        self.sizes.fill(0);
        self.pos.fill(0);
        self.total_num = 0;
        self.total_occupied = 0;
        self.buffer_size = 0;
        let empty = Layout { num: &self.num, pos: &self.pos, size: 0 };
        if let Err(e) = store.stage(None, empty).and_then(|bufs| store.commit(bufs)) {
            log::error!("cannot reset the sample lists of a deep level: {e}");
        }
    }

    /// `clear`: every count back to zero.
    pub(crate) fn clear(&mut self, store: &mut dyn SampleStore) -> Result<()> {
        self.num.fill(0);
        let r = self.rebuild(store, None, false);
        if r.is_err() {
            self.reset(store);
        }
        r
    }

    /// `beginEdit` / `Edit` / `endEdit`: let `f` change the counts freely; afterwards the lists are
    /// re-laid out for the new counts and all samples are zero.
    pub(crate) fn edit<R>(&mut self, store: &mut dyn SampleStore, f: impl FnOnce(&mut [u32]) -> R) -> Result<R> {
        let r = f(&mut self.num);
        if let Err(e) = self.rebuild(store, None, false) {
            self.reset(store);
            return Err(e);
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_room_is_a_power_of_two() {
        assert_eq!(round_list_size_up(0).unwrap(), 0);
        assert_eq!(round_list_size_up(1).unwrap(), 1);
        assert_eq!(round_list_size_up(5).unwrap(), 8);
        assert_eq!(round_list_size_up(0x8000_0000).unwrap(), 0x8000_0000);
        assert!(round_list_size_up(0x8000_0001).is_err());
        assert_eq!(round_buffer_size_up(10), 15);
    }

    #[test]
    fn layout_places_lists_back_to_back() {
        let (sizes, pos, occupied, total) = SampleCountChannel::layout_for(&[3, 0, 1, 5]).unwrap();
        assert_eq!(sizes, [4, 0, 1, 8]);
        assert_eq!(pos, [0, 4, 4, 5]);
        assert_eq!((occupied, total), (13, 9));
    }
}
