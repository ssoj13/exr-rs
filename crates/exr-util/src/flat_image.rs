//! Port of `ImfFlatImage.{h,cpp}`: an image whose channels hold at most one sample per pixel.
//!
//! The C++ `FlatImage` only adds `newLevel` and covariant `level()` accessors to `Image`, which the
//! generic [`Image`] already provides for any [`ImageLevel`](crate::image_level::ImageLevel).

use crate::flat_image_level::FlatImageLevel;
use crate::image::Image;

/// `FlatImage`: `Image::new (dw, mode, rounding)` / `Image::empty ()` construct it.
pub type FlatImage = Image<FlatImageLevel>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_channel::ImageChannel;
    use crate::image_level::ImageLevel;
    use exr_core::attr::{Channel, LevelMode, LevelRoundingMode, Name, PixelType};
    use imath_rs::{Box2i, Half, V2i};

    fn bx(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
        Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
    }

    const MIP: LevelMode = LevelMode::MipmapLevels;
    const RIP: LevelMode = LevelMode::RipmapLevels;
    const DOWN: LevelRoundingMode = LevelRoundingMode::RoundDown;
    const UP: LevelRoundingMode = LevelRoundingMode::RoundUp;

    #[test]
    fn empty_image_matches_the_cpp_default() {
        let img = FlatImage::empty().unwrap();
        assert_eq!(*img.data_window(), bx(0, 0, -1, -1));
        assert_eq!((img.num_x_levels(), img.num_y_levels()), (1, 1));
        assert_eq!(*img.level(0).unwrap().data_window(), bx(0, 0, -1, -1));
    }

    #[test]
    fn level_counts_and_sizes_follow_imfimage() {
        // 100 x 37: floor(log2 100) + 1 = 7 mip levels; RIP has 7 x 6 levels (floor(log2 37) = 5).
        let mip = FlatImage::new(&bx(0, 0, 99, 36), MIP, DOWN).unwrap();
        assert_eq!(mip.num_levels().unwrap(), 7);
        assert!(mip.level_at(1, 2).is_err(), "a MIPMAP image has only diagonal levels");
        let sizes: Vec<_> = (0..7).map(|l| (mip.level_width(l).unwrap(), mip.level_height(l).unwrap())).collect();
        assert_eq!(sizes, [(100, 37), (50, 18), (25, 9), (12, 4), (6, 2), (3, 1), (1, 1)]);
        let rip = FlatImage::new(&bx(0, 0, 99, 36), RIP, DOWN).unwrap();
        assert_eq!((rip.num_x_levels(), rip.num_y_levels()), (7, 6));
        assert!(rip.num_levels().is_err());
        // round up: ceil(log2 100) + 1 = 8 levels, sizes ceil-divided
        let up = FlatImage::new(&bx(0, 0, 99, 36), MIP, UP).unwrap();
        assert_eq!(up.num_levels().unwrap(), 8);
        assert_eq!((up.level_width(1).unwrap(), up.level_height(1).unwrap()), (50, 19));
        assert_eq!(*up.data_window_for_level(2).unwrap(), bx(0, 0, 24, 9));
    }

    #[test]
    fn an_empty_window_keeps_imfimages_level_counts_and_stays_empty() {
        // ImfImage.cpp:134: roundLog2 (x < 1) is 1, so an empty MIPMAP / RIPMAP image has 2 levels, both empty.
        let empty = bx(0, 0, -1, -1);
        let mip = FlatImage::new(&empty, MIP, DOWN).unwrap();
        assert_eq!(mip.num_levels().unwrap(), 2);
        assert_eq!(*mip.level(1).unwrap().data_window(), empty);
        let rip = FlatImage::new(&empty, RIP, UP).unwrap();
        assert_eq!((rip.num_x_levels(), rip.num_y_levels()), (2, 2));
        assert_eq!((rip.level_width(1).unwrap(), rip.level_height(1).unwrap()), (0, 0));
        // a 1 x 1 image still has one level
        assert_eq!(FlatImage::new(&bx(0, 0, 0, 0), MIP, DOWN).unwrap().num_levels().unwrap(), 1);
    }

    #[test]
    fn invalid_windows_are_rejected() {
        assert!(FlatImage::new(&bx(5, 0, 2, 0), LevelMode::OneLevel, DOWN).is_err());
        assert!(FlatImage::new(&bx(0, 0, i32::MAX / 2, 1), LevelMode::OneLevel, DOWN).is_err());
    }

    #[test]
    fn pixels_survive_insert_and_shift() {
        let mut img = FlatImage::new(&bx(-2, -2, 1, 1), LevelMode::OneLevel, DOWN).unwrap();
        img.insert_type("R", PixelType::Half).unwrap();
        img.insert_type("Z", PixelType::Float).unwrap();
        {
            let lv = img.level_mut(0).unwrap();
            *lv.typed_channel_mut::<Half>("R").unwrap().at_mut(-1, 0).unwrap() = Half::from_f32(0.5);
            *lv.typed_channel_mut::<f32>("Z").unwrap().at_mut(1, 1).unwrap() = 9.0;
            assert!(lv.typed_channel::<u32>("R").is_err(), "wrong sample type");
            assert!(lv.typed_channel::<f32>("nope").is_err());
            assert!(lv.typed_channel::<Half>("R").unwrap().at(2, 0).is_err(), "outside the window");
        }
        img.shift_pixels(10, 20).unwrap();
        let lv = img.level(0).unwrap();
        assert_eq!(*lv.data_window(), bx(8, 18, 11, 21));
        assert_eq!(lv.typed_channel::<Half>("R").unwrap().at(9, 20).unwrap().to_f32(), 0.5);
        assert_eq!(*lv.typed_channel::<f32>("Z").unwrap().at(11, 21).unwrap(), 9.0);
    }

    #[test]
    fn shift_needs_multiples_of_the_sampling() {
        let mut img = FlatImage::new(&bx(0, 0, 3, 3), LevelMode::OneLevel, DOWN).unwrap();
        img.insert_channel("BY", &Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        assert!(img.shift_pixels(1, 0).is_err());
        assert!(img.shift_pixels(0, 3).is_err());
        img.shift_pixels(2, 4).unwrap();
        let ch = img.level(0).unwrap().typed_channel::<Half>("BY").unwrap();
        assert_eq!((ch.row(1).unwrap().len(), ch.at(2, 4).is_ok(), ch.at(3, 4).is_err()), (2, true, true));
    }

    #[test]
    fn subsampled_channel_needs_an_aligned_window() {
        let mut img = FlatImage::new(&bx(0, 0, 4, 3), LevelMode::OneLevel, DOWN).unwrap();
        assert!(img.insert_channel("BY", &Channel::new(PixelType::Half, 2, 1, false)).is_err(), "width 5");
        assert_eq!(img.channels().count(), 0, "a failed insert leaves no channel behind");
        assert!(img.insert_channel("X", &Channel::new(PixelType::Half, 0, 1, false)).is_err());
    }

    #[test]
    fn insert_replaces_and_resize_keeps_channels() {
        let mut img = FlatImage::new(&bx(0, 0, 3, 3), MIP, DOWN).unwrap();
        img.insert_type("R", PixelType::Half).unwrap();
        img.insert_type("R", PixelType::Uint).unwrap();
        assert_eq!(img.level(2).unwrap().find_channel("R").map(|c| c.pixel_type()), Some(PixelType::Uint));
        img.resize(&bx(0, 0, 7, 7)).unwrap();
        assert_eq!(img.num_levels().unwrap(), 4);
        for l in 0..4 {
            assert!(img.level(l).unwrap().find_channel("R").is_some());
        }
    }

    #[test]
    fn rect_copies_respect_the_origin_and_the_byte_buffer() {
        let mut img = FlatImage::new(&bx(0, 0, 3, 2), LevelMode::OneLevel, DOWN).unwrap();
        img.insert_type("U", PixelType::Uint).unwrap();
        let ch = img.level_mut(0).unwrap().channel_mut("U").unwrap();
        let bytes: Vec<u8> = (0u32..6).flat_map(|v| v.to_ne_bytes()).collect(); // a 3 x 2 rectangle
        ch.load_rect(1, 2, 3, 2, &bytes).unwrap_err(); // row 3 does not exist in a 3-row image
        ch.load_rect(1, 0, 3, 2, &bytes).unwrap();
        let u = img.level(0).unwrap().typed_channel::<u32>("U").unwrap();
        assert_eq!(u.pixels(), [0, 0, 1, 2, 0, 3, 4, 5, 0, 0, 0, 0]);
        let mut out = vec![0u8; 24];
        img.level(0).unwrap().channel("U").unwrap().store_rect(1, 0, 3, 2, &mut out).unwrap();
        assert_eq!(out, bytes);
        assert!(img.level(0).unwrap().channel("U").unwrap().store_rect(0, 0, 3, 2, &mut out[..20]).is_err(), "short buffer");
    }

    #[test]
    fn shifting_past_the_coordinate_range_is_an_error_and_changes_nothing() {
        let mut img = FlatImage::new(&bx(0, 0, 3, 3), MIP, DOWN).unwrap();
        img.insert_type("R", PixelType::Half).unwrap();
        assert!(img.shift_pixels(i32::MAX - 1, 0).is_err());
        assert!(img.shift_pixels(0, i32::MAX - 2).is_err());
        assert_eq!(*img.data_window(), bx(0, 0, 3, 3));
        assert_eq!(*img.level(1).unwrap().data_window(), bx(0, 0, 1, 1));
    }

    #[test]
    fn renaming() {
        let mut img = FlatImage::new(&bx(0, 0, 1, 1), MIP, DOWN).unwrap();
        for n in ["a", "b", "c"] {
            img.insert_type(n, PixelType::Float).unwrap();
        }
        assert!(img.rename_channel("x", "y").is_err());
        assert!(img.rename_channel("a", "b").is_err());
        img.rename_channel("a", "A").unwrap();
        let map = [("b", "c"), ("c", "b")].map(|(a, b)| (Name::from(a), Name::from(b))).into();
        img.rename_channels(&map).unwrap();
        let clash = [("b", "A")].map(|(a, b)| (Name::from(a), Name::from(b))).into();
        assert!(img.rename_channels(&clash).is_err());
        let names: Vec<_> = img.channels().map(|(n, _)| n.to_string()).collect();
        assert_eq!(names, ["A", "b", "c"]);
        assert!(img.level(1).unwrap().find_channel("A").is_some());
        img.erase_channel("b");
        img.clear_channels();
        assert_eq!(img.channels().count(), 0);
        assert_eq!(img.level(0).unwrap().iter().count(), 0);
    }
}
