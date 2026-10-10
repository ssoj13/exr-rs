//! Port of `ImfImageLevel.{h,cpp}`: the state every image level shares and the interface the
//! generic [`Image`](crate::image::Image) needs from a level.
//!
//! C++ uses a virtual base class with a back-reference to the owning image; here the image owns its
//! levels and drives them through [`ImageLevel`], and a level knows only its own level numbers and
//! data window.

use exr_core::attr::{Channel, Name};
use exr_core::{ExrError, Result};
use imath_rs::Box2i;

use crate::image_channel::{empty_dw, shift_box};
use crate::image_channel_renaming::RenamingMap;

/// What `ImageLevel` stores in C++: the level numbers and the level's data window.
#[derive(Clone, Debug)]
pub struct LevelBase {
    x_level: i32,
    y_level: i32,
    dw: Box2i,
}

impl LevelBase {
    /// `ImageLevel::ImageLevel`: an empty data window; call [`LevelBase::resize`].
    pub(crate) fn new(x_level: i32, y_level: i32) -> Self {
        Self { x_level, y_level, dw: empty_dw() }
    }

    /// `ImageLevel::resize`: set the data window (an empty window, `min == max + 1`, is valid).
    pub(crate) fn resize(&mut self, dw: &Box2i) -> Result<()> {
        if dw.min.x as i64 > dw.max.x as i64 + 1 || dw.min.y as i64 > dw.max.y as i64 + 1 {
            return Err(ExrError::Arg(format!(
                "Cannot reset data window for image level to ({}, {}) - ({}, {}). The new data window is invalid.",
                dw.min.x, dw.min.y, dw.max.x, dw.max.y
            )));
        }
        self.dw = *dw;
        Ok(())
    }

    /// `ImageLevel::shiftPixels`: move the window.
    pub(crate) fn shift(&mut self, dx: i32, dy: i32) -> Result<()> {
        shift_box(&mut self.dw, dx, dy)
    }
}

/// The key a channel name is stored under: `Name`'s C++ `strncpy` cut (first NUL, 255 bytes).
pub(crate) fn key(name: impl AsRef<[u8]>) -> Name {
    Name::new(name)
}

/// `throwChannelExists`.
pub(crate) fn channel_exists(name: &Name) -> ExrError {
    ExrError::Arg(format!(
        "Cannot insert a new image channel with name \"{name}\" into an image level. A channel with the same name \
         exists already."
    ))
}

/// `throwBadChannelName`.
pub(crate) fn bad_channel_name(name: &Name) -> ExrError {
    ExrError::Arg(format!("Attempt to access non-existent image channel \"{name}\"."))
}

/// `throwBadChannelNameOrType`.
pub(crate) fn bad_channel_name_or_type(name: &Name) -> ExrError {
    ExrError::Arg(format!("Image channel \"{name}\" does not exist or is not of the expected type."))
}

/// The interface an [`Image`](crate::image::Image) drives its levels through: the protected virtual
/// functions of C++ `ImageLevel` plus its constructor (`Image::newLevel`).
pub trait ImageLevel: Sized {
    /// Create the level `(lx, ly)` with data window `dw` and no channels (`newLevel`).
    fn new(lx: i32, ly: i32, dw: &Box2i) -> Result<Self>;
    /// The level numbers and data window.
    fn base(&self) -> &LevelBase;
    /// Give the level a new data window and reallocate every channel; pixels are lost.
    fn resize(&mut self, dw: &Box2i) -> Result<()>;
    /// Move the data window by `(dx, dy)` together with the pixels.
    fn shift_pixels(&mut self, dx: i32, dy: i32) -> Result<()>;
    /// Add a channel; an existing channel of that name is an error.
    fn insert_channel(&mut self, name: &Name, ch: &Channel) -> Result<()>;
    /// Remove a channel if present.
    fn erase_channel(&mut self, name: &Name);
    /// Remove every channel.
    fn clear_channels(&mut self);
    /// Rename a channel that exists to a name that does not (the image checks both first).
    fn rename_channel(&mut self, old: &Name, new: &Name) -> Result<()>;
    /// Rename several channels at once.
    fn rename_channels(&mut self, map: &RenamingMap);

    /// `xLevelNumber`.
    fn x_level_number(&self) -> i32 {
        self.base().x_level
    }
    /// `yLevelNumber`.
    fn y_level_number(&self) -> i32 {
        self.base().y_level
    }
    /// The range of x and y for which the level stores pixels.
    fn data_window(&self) -> &Box2i {
        &self.base().dw
    }
}
