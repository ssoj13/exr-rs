//! Port of `ImfImageDataWindow.{h,cpp}`: which data window a saved image file gets.

use exr_core::header::Header;
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, V2i};

use crate::image::Image;
use crate::image_level::ImageLevel;

/// `DataWindowSource`: where `save*Image` takes the file's data window from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DataWindowSource {
    /// `USE_IMAGE_DATA_WINDOW`: the data window of the image.
    #[default]
    Image,
    /// `USE_HEADER_DATA_WINDOW`: the intersection of the header's and the image's data windows
    /// (crops the image; ONE_LEVEL images only).
    Header,
}

/// `dataWindowForFile`: the data window of the file that saves `img` with header `hdr`.
///
/// # Errors
/// `Arg` for [`DataWindowSource::Header`] on a multi-resolution image.
pub fn data_window_for_file<L: ImageLevel>(hdr: &Header, img: &Image<L>, dws: DataWindowSource) -> Result<Box2i> {
    match dws {
        DataWindowSource::Image => Ok(*img.data_window()),
        DataWindowSource::Header => {
            if img.level_mode() != exr_core::attr::LevelMode::OneLevel {
                return Err(ExrError::Arg("Cannot crop multi-resolution images.".into()));
            }
            let (h, i) = (hdr.data_window()?, img.data_window());
            Ok(Box2i {
                min: V2i { x: h.min.x.max(i.min.x), y: h.min.y.max(i.min.y) },
                max: V2i { x: h.max.x.min(i.max.x), y: h.max.y.min(i.max.y) },
            })
        }
    }
}
