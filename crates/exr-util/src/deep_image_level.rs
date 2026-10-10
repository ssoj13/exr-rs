//! Port of `ImfDeepImageLevel.{h,cpp}`: one resolution level of a deep image: the level's
//! [`SampleCountChannel`] plus a named set of [`DeepImageChannel`]s.
//!
//! Changing sample counts touches both the counts and every channel's buffer, so the C++
//! `level.sampleCounts().set (...)` spelling becomes [`DeepImageLevel::set_count`] and friends here
//! (the channel cannot reach its siblings; see [`crate::sample_count_channel`]).

use std::collections::BTreeMap;

use exr_core::attr::{Channel, Name};
use exr_core::{ExrError, Result};
use imath_rs::Box2i;

use crate::deep_image_channel::{DeepImageChannel, DeepView, DeepViewMut};
use crate::image_channel::Pixel;
use crate::image_channel_renaming::{rename_in_map, RenamingMap};
use crate::image_level::{bad_channel_name, bad_channel_name_or_type, channel_exists, key, ImageLevel, LevelBase};
use crate::sample_count_channel::SampleCountChannel;

/// `DeepImageLevel`.
pub struct DeepImageLevel {
    base: LevelBase,
    channels: BTreeMap<Name, DeepImageChannel>,
    sc: SampleCountChannel,
}

impl DeepImageLevel {
    /// `sampleCounts`: the per-pixel sample counts (read only; change them with the methods below).
    pub fn sample_counts(&self) -> &SampleCountChannel {
        &self.sc
    }

    /// `sampleCounts().set (x, y, n)`: set one pixel's sample count. Its samples are kept (new ones
    /// are zero) and the other pixels' samples are untouched.
    pub fn set_count(&mut self, x: i32, y: i32, n: u32) -> Result<()> {
        self.sc.set(x, y, n, &mut self.channels)
    }

    /// `sampleCounts().set (r, counts)`: set the counts of row `r` (0-based) of the level.
    pub fn set_counts_row(&mut self, r: i32, counts: &[u32]) -> Result<()> {
        self.sc.set_row(r, counts, &mut self.channels)
    }

    /// `sampleCounts().clear ()`: every count zero.
    pub fn clear_counts(&mut self) -> Result<()> {
        self.sc.clear(&mut self.channels)
    }

    /// `SampleCountChannel::Edit`: run `f` on the raw count array (row-major); afterwards the sample
    /// lists are laid out for the new counts and every sample is zero.
    pub fn edit_counts<R>(&mut self, f: impl FnOnce(&mut [u32]) -> R) -> Result<R> {
        self.sc.edit(&mut self.channels, f)
    }

    /// `findChannel`.
    pub fn find_channel(&self, name: impl AsRef<[u8]>) -> Option<&DeepImageChannel> {
        self.channels.get(&key(name))
    }

    /// `channel`: like [`DeepImageLevel::find_channel`] but a missing channel is an `Arg` error.
    pub fn channel(&self, name: impl AsRef<[u8]>) -> Result<&DeepImageChannel> {
        let k = key(name);
        self.channels.get(&k).ok_or_else(|| bad_channel_name(&k))
    }

    /// `typedChannel<T>`: the channel together with the level's counts.
    ///
    /// # Errors
    /// `Arg` if there is no channel of that name holding samples of type `T`.
    pub fn typed_channel<T: Pixel>(&self, name: impl AsRef<[u8]>) -> Result<DeepView<'_, T>> {
        let k = key(name);
        let ch = self.channels.get(&k).and_then(T::deep).ok_or_else(|| bad_channel_name_or_type(&k))?;
        Ok(DeepView { ch, sc: &self.sc })
    }

    /// `typedChannel<T>`, with writable samples.
    pub fn typed_channel_mut<T: Pixel>(&mut self, name: impl AsRef<[u8]>) -> Result<DeepViewMut<'_, T>> {
        let k = key(name);
        let ch = self.channels.get_mut(&k).and_then(T::deep_mut).ok_or_else(|| bad_channel_name_or_type(&k))?;
        Ok(DeepViewMut { ch, sc: &self.sc })
    }

    /// `begin () .. end ()`: the channels in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&Name, &DeepImageChannel)> {
        self.channels.iter()
    }

    /// Mutable channels and the counts they are laid out by, for the file readers.
    pub(crate) fn channels_and_counts(&mut self) -> (&mut BTreeMap<Name, DeepImageChannel>, &SampleCountChannel) {
        (&mut self.channels, &self.sc)
    }
}

impl ImageLevel for DeepImageLevel {
    fn new(lx: i32, ly: i32, dw: &Box2i) -> Result<Self> {
        let mut l = Self { base: LevelBase::new(lx, ly), channels: BTreeMap::new(), sc: SampleCountChannel::new() };
        l.resize(dw)?;
        Ok(l)
    }

    fn base(&self) -> &LevelBase {
        &self.base
    }

    fn resize(&mut self, dw: &Box2i) -> Result<()> {
        self.base.resize(dw)?;
        self.sc.resize(dw)?;
        let size = self.sc.sample_buffer_size();
        self.channels.values_mut().try_for_each(|c| c.resize(dw, size))
    }

    fn shift_pixels(&mut self, dx: i32, dy: i32) -> Result<()> {
        self.base.shift(dx, dy)?;
        self.sc.shift(dx, dy)?;
        self.channels.values_mut().try_for_each(|c| c.shift(dx, dy))
    }

    fn insert_channel(&mut self, name: &Name, ch: &Channel) -> Result<()> {
        // `ImfDeepImageLevel.cpp:174` tests `xSampling != 1 && ySampling != 1`, which lets a channel
        // that is subsampled in one direction through with its sampling silently ignored; the
        // message it throws says both must be 1, so both are required here.
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::Arg(format!(
                "Cannot create deep image channel {name} with x sampling rate {} and and y sampling rate {}. X and y \
                 sampling rates for deep channels must be 1.",
                ch.x_sampling, ch.y_sampling
            )));
        }
        if self.channels.contains_key(name) {
            return Err(channel_exists(name));
        }
        let c = DeepImageChannel::new(ch, self.data_window(), self.sc.sample_buffer_size())?;
        self.channels.insert(name.clone(), c);
        Ok(())
    }

    fn erase_channel(&mut self, name: &Name) {
        self.channels.remove(name);
    }

    fn clear_channels(&mut self) {
        self.channels.clear();
    }

    fn rename_channel(&mut self, old: &Name, new: &Name) -> Result<()> {
        if self.channels.contains_key(new) {
            return Err(channel_exists(new));
        }
        let c = self.channels.remove(old).ok_or_else(|| bad_channel_name(old))?;
        self.channels.insert(new.clone(), c);
        Ok(())
    }

    fn rename_channels(&mut self, map: &RenamingMap) {
        rename_in_map(map, &mut self.channels);
    }
}
