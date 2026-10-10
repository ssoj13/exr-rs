//! Port of `ImfImage.{h,cpp}`: an in-memory OpenEXR image with one or more resolution levels and an
//! arbitrary set of channels.
//!
//! C++ has an abstract `Image` with the concrete `FlatImage` and `DeepImage` derived from it (they
//! only supply `newLevel` and covariant `level()` accessors). Here [`Image`] is generic over the
//! level type, and `FlatImage` / `DeepImage` are aliases: one implementation of the level
//! arithmetic, channel bookkeeping and renaming for both.
//!
//! Level numbering, sizes and counts follow `ImfImage.cpp` exactly (`levelSize`, `roundLog2`,
//! `computeNumXLevels`); a MIPMAP image stores only the levels where x and y level numbers agree.

use std::collections::{BTreeMap, BTreeSet};

use exr_core::attr::{Channel, LevelMode, LevelRoundingMode, Name, PixelType, TileDescription};
use exr_core::tiled::misc::{num_x_levels, num_y_levels};
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, V2i};

use crate::image_channel::{empty_dw, shift_box};
use crate::image_channel_renaming::{rename_in_map, RenamingMap};
use crate::image_level::{key, ImageLevel};

/// `INT_MAX / 2`: the bound `validateDataWindow` puts on data window coordinates.
const HALF_INT_MAX: i64 = (i32::MAX / 2) as i64;

/// Extent of `[min, max]`, 0 for an empty range (`pixelExtent`).
fn extent(min: i64, max: i64) -> i64 {
    if max < min { 0 } else { max - min + 1 }
}

/// `validateDataWindow`.
fn validate_dw(dw: &Box2i) -> Result<()> {
    let bad = || ExrError::Arg("Invalid data window for image resize.".into());
    let (x0, y0, x1, y1) = (dw.min.x as i64, dw.min.y as i64, dw.max.x as i64, dw.max.y as i64);
    if x0 > x1 + 1 || y0 > y1 + 1 {
        return Err(bad());
    }
    if x0 <= -HALF_INT_MAX || y0 <= -HALF_INT_MAX || x1 >= HALF_INT_MAX || y1 >= HALF_INT_MAX {
        return Err(bad());
    }
    Ok(())
}

/// `levelSize` of `ImfImage.cpp` (`:61`): exr-core's [`level_size`](exr_core::tiled::misc::level_size) (the
/// `ImfTiledMisc.cpp` function, at least 1) except that an empty range `[min, max]` has size 0, so that the
/// data window `(0, 0) - (-1, -1)` of an empty image stays empty on every level.
fn level_size(min: i32, max: i32, l: i32, rounding: LevelRoundingMode) -> Result<i32> {
    if extent(min as i64, max as i64) == 0 {
        return Ok(0);
    }
    exr_core::tiled::misc::level_size(min, max, l, rounding)
        .map_err(|_| ExrError::Arg("Invalid level size for image resize.".into()))
}

/// `computeNumXLevels` (`y == false`) / `computeNumYLevels` (`y == true`) of `ImfImage.cpp` (`:140`,
/// `:169`): exr-core's `num_x_levels` / `num_y_levels`, except that their `roundLog2` of an empty extent
/// is 0 while `ImfImage.cpp`'s `roundLog2` (`:134`) answers 1 for `x < 1`, which gives an empty MIPMAP or
/// RIPMAP image two levels (the second one empty).
fn num_levels(dw: &Box2i, mode: LevelMode, rounding: LevelRoundingMode, y: bool) -> Result<i32> {
    let (w, h) = (extent(dw.min.x as i64, dw.max.x as i64), extent(dw.min.y as i64, dw.max.y as i64));
    let deciding = match mode {
        LevelMode::OneLevel => return Ok(1),
        LevelMode::MipmapLevels => w.max(h),
        LevelMode::RipmapLevels if y => h,
        LevelMode::RipmapLevels => w,
        LevelMode::Invalid => return Err(ExrError::Arg("Invalid level mode for image resize.".into())),
    };
    if deciding < 1 {
        return Ok(2);
    }
    let td = TileDescription { mode, rounding_mode: rounding, ..Default::default() };
    let count = if y { num_y_levels } else { num_x_levels };
    count(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)
}

/// An image: levels of channels. See the module docs; use [`FlatImage`](crate::flat_image::FlatImage)
/// or [`DeepImage`](crate::deep_image::DeepImage).
pub struct Image<L: ImageLevel> {
    dw: Box2i,
    mode: LevelMode,
    rounding: LevelRoundingMode,
    /// The channel set shared by every level, by name (`ChannelMap` of `ImfImage.h`).
    channels: BTreeMap<Name, Channel>,
    nx: i32,
    ny: i32,
    /// Row-major `ny * nx` table; `None` for the levels a MIPMAP image does not have.
    levels: Vec<Option<L>>,
}

impl<L: ImageLevel> Image<L> {
    /// `FlatImage (dataWindow, levelMode, levelRoundingMode)`: an image with no channels.
    pub fn new(dw: &Box2i, mode: LevelMode, rounding: LevelRoundingMode) -> Result<Self> {
        let mut img = Self::empty()?;
        img.resize_with(dw, mode, rounding)?;
        Ok(img)
    }

    /// `FlatImage ()`: the empty data window `(0,0)-(-1,-1)`, one level.
    pub fn empty() -> Result<Self> {
        let mut img = Self {
            dw: empty_dw(),
            mode: LevelMode::OneLevel,
            rounding: LevelRoundingMode::RoundDown,
            channels: BTreeMap::new(),
            nx: 0,
            ny: 0,
            levels: Vec::new(),
        };
        img.resize_with(&empty_dw(), LevelMode::OneLevel, LevelRoundingMode::RoundDown)?;
        Ok(img)
    }

    /// `levelMode`.
    pub fn level_mode(&self) -> LevelMode {
        self.mode
    }

    /// `levelRoundingMode`.
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.rounding
    }

    /// `numLevels`: levels of a ONE_LEVEL or MIPMAP image.
    ///
    /// # Errors
    /// `Logic` for a RIPMAP image, which needs [`Image::num_x_levels`] / [`Image::num_y_levels`].
    pub fn num_levels(&self) -> Result<i32> {
        match self.mode {
            LevelMode::OneLevel | LevelMode::MipmapLevels => Ok(self.nx),
            _ => Err(ExrError::Logic("Number of levels query for image must specify x or y direction.".into())),
        }
    }

    /// `numXLevels`.
    pub fn num_x_levels(&self) -> i32 {
        self.nx
    }

    /// `numYLevels`.
    pub fn num_y_levels(&self) -> i32 {
        self.ny
    }

    /// `dataWindow`: the window of level (0, 0).
    pub fn data_window(&self) -> &Box2i {
        &self.dw
    }

    /// `dataWindowForLevel (l)`.
    pub fn data_window_for_level(&self, l: i32) -> Result<&Box2i> {
        self.data_window_for_level_at(l, l)
    }

    /// `dataWindowForLevel (lx, ly)`.
    pub fn data_window_for_level_at(&self, lx: i32, ly: i32) -> Result<&Box2i> {
        self.level_at(lx, ly)
            .map(ImageLevel::data_window)
            .map_err(|_| ExrError::Arg(format!("Cannot get data window for invalid image level ({lx}, {ly}).")))
    }

    /// `levelWidth`: width of a level with x level number `lx`.
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        if lx < 0 || lx >= self.nx {
            return Err(ExrError::Arg(format!("Cannot get level width for invalid image level number {lx}.")));
        }
        level_size(self.dw.min.x, self.dw.max.x, lx, self.rounding)
    }

    /// `levelHeight`: height of a level with y level number `ly`.
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        if ly < 0 || ly >= self.ny {
            return Err(ExrError::Arg(format!("Cannot get level height for invalid image level number {ly}.")));
        }
        level_size(self.dw.min.y, self.dw.max.y, ly, self.rounding)
    }

    /// `resize (dataWindow)`: keep the level mode and rounding mode.
    pub fn resize(&mut self, dw: &Box2i) -> Result<()> {
        self.resize_with(dw, self.mode, self.rounding)
    }

    /// `resize (dataWindow, levelMode, levelRoundingMode)`: new levels, same channel set, pixels lost.
    /// On failure the image is left with an empty data window and no levels.
    pub fn resize_with(&mut self, dw: &Box2i, mode: LevelMode, rounding: LevelRoundingMode) -> Result<()> {
        self.clear_levels();
        let r = self.build_levels(dw, mode, rounding);
        if r.is_err() {
            self.clear_levels();
        }
        r
    }

    fn build_levels(&mut self, dw: &Box2i, mode: LevelMode, rounding: LevelRoundingMode) -> Result<()> {
        validate_dw(dw)?;
        let nx = num_levels(dw, mode, rounding, false)?;
        let ny = num_levels(dw, mode, rounding, true)?;
        let mut levels = Vec::new();
        levels
            .try_reserve_exact(nx as usize * ny as usize)
            .map_err(|e| ExrError::Arg(format!("Cannot allocate image levels ({e}).")))?;
        for y in 0..ny {
            for x in 0..nx {
                if mode == LevelMode::MipmapLevels && x != y {
                    levels.push(None);
                    continue;
                }
                let w = level_size(dw.min.x, dw.max.x, x, rounding)?;
                let h = level_size(dw.min.y, dw.max.y, y, rounding)?;
                let ldw = Box2i { min: dw.min, max: V2i { x: dw.min.x + w - 1, y: dw.min.y + h - 1 } };
                let mut lv = L::new(x, y, &ldw)?;
                for (name, ch) in &self.channels {
                    lv.insert_channel(name, ch)?;
                }
                levels.push(Some(lv));
            }
        }
        self.levels = levels;
        self.nx = nx;
        self.ny = ny;
        self.dw = *dw;
        self.mode = mode;
        self.rounding = rounding;
        Ok(())
    }

    /// `clearLevels`: drop every level and reset the data window.
    fn clear_levels(&mut self) {
        self.dw = empty_dw();
        self.levels.clear();
        self.nx = 0;
        self.ny = 0;
    }

    /// `shiftPixels`: move the data window and the pixels by `(dx, dy)`.
    ///
    /// # Errors
    /// `Arg` if a shift distance is not a multiple of the sampling rate of every channel.
    pub fn shift_pixels(&mut self, dx: i32, dy: i32) -> Result<()> {
        for (name, ch) in &self.channels {
            if dx % ch.x_sampling != 0 {
                return Err(ExrError::Arg(format!(
                    "Cannot shift image horizontally by {dx} pixels.  The shift distance must be a multiple of the x \
                     sampling rate of all channels, but the x sampling rate channel {name} is {}.",
                    ch.x_sampling
                )));
            }
            if dy % ch.y_sampling != 0 {
                return Err(ExrError::Arg(format!(
                    "Cannot shift image vertically by {dy} pixels.  The shift distance must be a multiple of the y \
                     sampling rate of all channels, but the y sampling rate channel {name} is {}.",
                    ch.y_sampling
                )));
            }
        }
        // Every level's window lies inside the image's, so if this one fits all of them do.
        shift_box(&mut self.dw, dx, dy)?;
        self.levels.iter_mut().flatten().try_for_each(|l| l.shift_pixels(dx, dy))
    }

    /// `insertChannel (name, channel)`: add a channel to every level. A channel of the same name is
    /// replaced, as the C++ documentation promises (`ImfImage.h:225-233`); the C++ code instead
    /// fails inside the level (`throwChannelExists`) and then erases the old channel, so the
    /// documented behaviour is what we implement.
    pub fn insert_channel(&mut self, name: impl Into<Name>, ch: &Channel) -> Result<()> {
        if ch.x_sampling < 1 || ch.y_sampling < 1 {
            return Err(ExrError::Arg("The x and y sampling rates for an image channel must be at least 1.".into()));
        }
        let name = name.into();
        self.erase(&name);
        self.channels.insert(name.clone(), *ch);
        let r = self.levels.iter_mut().flatten().try_for_each(|l| l.insert_channel(&name, ch));
        if r.is_err() {
            self.erase(&name);
        }
        r
    }

    /// `insertChannel (name, type)`: a channel with sampling 1, 1 and `pLinear` off.
    pub fn insert_type(&mut self, name: impl Into<Name>, type_: PixelType) -> Result<()> {
        self.insert_channel(name, &Channel::new(type_, 1, 1, false))
    }

    /// `eraseChannel`.
    pub fn erase_channel(&mut self, name: impl AsRef<[u8]>) {
        self.erase(&key(name));
    }

    fn erase(&mut self, name: &Name) {
        self.levels.iter_mut().flatten().for_each(|l| l.erase_channel(name));
        self.channels.remove(name);
    }

    /// `clearChannels`.
    pub fn clear_channels(&mut self) {
        self.levels.iter_mut().flatten().for_each(ImageLevel::clear_channels);
        self.channels.clear();
    }

    /// `renameChannel`.
    ///
    /// # Errors
    /// `Arg` if `old` does not exist or `new` does; a failing level leaves both channels erased.
    pub fn rename_channel(&mut self, old: impl AsRef<[u8]>, new: impl AsRef<[u8]>) -> Result<()> {
        let (old, new) = (key(old), key(new));
        if old == new {
            return Ok(());
        }
        let Some(info) = self.channels.get(&old).copied() else {
            return Err(ExrError::Arg(format!(
                "Cannot rename image channel {old} to {new}.  The image does not have a channel called {old}."
            )));
        };
        if self.channels.contains_key(&new) {
            return Err(ExrError::Arg(format!(
                "Cannot rename image channel {old} to {new}.  The image already has a channel called {new}."
            )));
        }
        let r = self.levels.iter_mut().flatten().try_for_each(|l| l.rename_channel(&old, &new));
        match r {
            Ok(()) => {
                self.channels.remove(&old);
                self.channels.insert(new, info);
                Ok(())
            }
            Err(e) => {
                self.erase(&old);
                self.erase(&new);
                Err(e)
            }
        }
    }

    /// `renameChannels`: apply `map` to all channel names at once.
    ///
    /// # Errors
    /// `Arg` if two channels would get the same name; nothing is renamed then.
    pub fn rename_channels(&mut self, map: &RenamingMap) -> Result<()> {
        let mut new_names = BTreeSet::new();
        for name in self.channels.keys() {
            let new = map.get(name).unwrap_or(name);
            if !new_names.insert(new) {
                return Err(ExrError::Arg(format!(
                    "Cannot rename image channels.  More than one channel would be named \"{new}\"."
                )));
            }
        }
        rename_in_map(map, &mut self.channels);
        self.levels.iter_mut().flatten().for_each(|l| l.rename_channels(map));
        Ok(())
    }

    /// The channel set: name and description, in name order.
    pub fn channels(&self) -> impl Iterator<Item = (&Name, &Channel)> {
        self.channels.iter()
    }

    fn slot(&self, lx: i32, ly: i32) -> Option<usize> {
        (lx >= 0 && lx < self.nx && ly >= 0 && ly < self.ny).then(|| ly as usize * self.nx as usize + lx as usize)
    }

    fn bad_level(lx: i32, ly: i32) -> ExrError {
        ExrError::Arg(format!("Cannot access image level with invalid level number ({lx}, {ly})."))
    }

    /// `level (l)`.
    pub fn level(&self, l: i32) -> Result<&L> {
        self.level_at(l, l)
    }

    /// `level (l)`, mutable.
    pub fn level_mut(&mut self, l: i32) -> Result<&mut L> {
        self.level_at_mut(l, l)
    }

    /// `level (lx, ly)`.
    pub fn level_at(&self, lx: i32, ly: i32) -> Result<&L> {
        self.slot(lx, ly)
            .and_then(|i| self.levels.get(i))
            .and_then(Option::as_ref)
            .ok_or_else(|| Self::bad_level(lx, ly))
    }

    /// `level (lx, ly)`, mutable.
    pub fn level_at_mut(&mut self, lx: i32, ly: i32) -> Result<&mut L> {
        let i = self.slot(lx, ly);
        i.and_then(|i| self.levels.get_mut(i))
            .and_then(Option::as_mut)
            .ok_or_else(|| Self::bad_level(lx, ly))
    }
}
