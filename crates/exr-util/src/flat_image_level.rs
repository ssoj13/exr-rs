//! Port of `ImfFlatImageLevel.{h,cpp}`: one resolution level of a flat image, a named set of
//! [`FlatImageChannel`]s over one data window.

use std::collections::BTreeMap;

use exr_core::attr::{Channel, Name};
use exr_core::Result;
use imath_rs::Box2i;

use crate::flat_image_channel::{FlatChannel, FlatImageChannel};
use crate::image_channel::Pixel;
use crate::image_channel_renaming::{rename_in_map, RenamingMap};
use crate::image_level::{bad_channel_name, bad_channel_name_or_type, channel_exists, key, ImageLevel, LevelBase};

/// `FlatImageLevel`. Channels are kept in name order, as the C++ `std::map` does.
pub struct FlatImageLevel {
    base: LevelBase,
    channels: BTreeMap<Name, FlatImageChannel>,
}

impl FlatImageLevel {
    /// `findChannel`.
    pub fn find_channel(&self, name: impl AsRef<[u8]>) -> Option<&FlatImageChannel> {
        self.channels.get(&key(name))
    }

    /// `findChannel`, mutable.
    pub fn find_channel_mut(&mut self, name: impl AsRef<[u8]>) -> Option<&mut FlatImageChannel> {
        self.channels.get_mut(&key(name))
    }

    /// `channel`: like [`FlatImageLevel::find_channel`] but a missing channel is an `Arg` error.
    pub fn channel(&self, name: impl AsRef<[u8]>) -> Result<&FlatImageChannel> {
        let k = key(name);
        self.channels.get(&k).ok_or_else(|| bad_channel_name(&k))
    }

    /// `channel`, mutable.
    pub fn channel_mut(&mut self, name: impl AsRef<[u8]>) -> Result<&mut FlatImageChannel> {
        let k = key(name);
        self.channels.get_mut(&k).ok_or_else(|| bad_channel_name(&k))
    }

    /// `findTypedChannel<T>`: the channel if it exists and holds samples of type `T`.
    pub fn find_typed_channel<T: Pixel>(&self, name: impl AsRef<[u8]>) -> Option<&FlatChannel<T>> {
        self.channels.get(&key(name)).and_then(T::flat)
    }

    /// `typedChannel<T>`: as [`FlatImageLevel::find_typed_channel`], an `Arg` error if absent.
    pub fn typed_channel<T: Pixel>(&self, name: impl AsRef<[u8]>) -> Result<&FlatChannel<T>> {
        let k = key(name);
        self.channels.get(&k).and_then(T::flat).ok_or_else(|| bad_channel_name_or_type(&k))
    }

    /// `typedChannel<T>`, mutable.
    pub fn typed_channel_mut<T: Pixel>(&mut self, name: impl AsRef<[u8]>) -> Result<&mut FlatChannel<T>> {
        let k = key(name);
        self.channels.get_mut(&k).and_then(T::flat_mut).ok_or_else(|| bad_channel_name_or_type(&k))
    }

    /// `begin () .. end ()`: the channels in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&Name, &FlatImageChannel)> {
        self.channels.iter()
    }

    /// Mutable form of [`FlatImageLevel::iter`].
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Name, &mut FlatImageChannel)> {
        self.channels.iter_mut()
    }
}

impl ImageLevel for FlatImageLevel {
    fn new(lx: i32, ly: i32, dw: &Box2i) -> Result<Self> {
        let mut l = Self { base: LevelBase::new(lx, ly), channels: BTreeMap::new() };
        l.resize(dw)?;
        Ok(l)
    }

    fn base(&self) -> &LevelBase {
        &self.base
    }

    fn resize(&mut self, dw: &Box2i) -> Result<()> {
        self.base.resize(dw)?;
        self.channels.values_mut().try_for_each(|c| c.resize(dw))
    }

    fn shift_pixels(&mut self, dx: i32, dy: i32) -> Result<()> {
        self.base.shift(dx, dy)?;
        self.channels.values_mut().try_for_each(|c| c.shift(dx, dy))
    }

    fn insert_channel(&mut self, name: &Name, ch: &Channel) -> Result<()> {
        if self.channels.contains_key(name) {
            return Err(channel_exists(name));
        }
        let c = FlatImageChannel::new(ch, self.data_window())?;
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
