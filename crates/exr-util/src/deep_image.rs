//! Port of `ImfDeepImage.{h,cpp}`: an image whose channels hold a list of samples per pixel.
//!
//! As for [`FlatImage`](crate::flat_image::FlatImage), the C++ class only supplies `newLevel` and
//! covariant `level()` accessors, which the generic [`Image`] already has.

use crate::deep_image_level::DeepImageLevel;
use crate::image::Image;

/// `DeepImage`: `Image::new (dw, mode, rounding)` / `Image::empty ()` construct it.
pub type DeepImage = Image<DeepImageLevel>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_channel::ImageChannel;
    use exr_core::attr::{Channel, LevelMode, LevelRoundingMode, PixelType};
    use imath_rs::{Box2i, Half, V2i};

    fn bx(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
        Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
    }

    fn img(mode: LevelMode) -> DeepImage {
        let mut i = DeepImage::new(&bx(-1, 0, 2, 1), mode, LevelRoundingMode::RoundDown).unwrap();
        i.insert_type("R", PixelType::Half).unwrap();
        i.insert_type("Z", PixelType::Float).unwrap();
        i
    }

    /// Sample `k` of pixel `(x, y)` in channel Z: a value that identifies the pixel and the index.
    fn tag(x: i32, y: i32, k: usize) -> f32 {
        (100 * (y + 1) + 10 * (x + 2)) as f32 + k as f32
    }

    #[test]
    fn set_count_keeps_samples_through_every_growth_path() {
        let mut im = img(LevelMode::OneLevel);
        let lv = im.level_mut(0).unwrap();
        // Grow each pixel step by step; every step exercises in-list growth, moves and rebuilds.
        let mut counts = [[0usize; 4]; 2];
        for round in 1..=9usize {
            for y in 0..2 {
                for x in -1..3 {
                    let n = if (x + y + round as i32) % 3 == 0 { counts[y as usize][(x + 1) as usize] } else { round };
                    let n = n.max(counts[y as usize][(x + 1) as usize]);
                    lv.set_count(x, y, n as u32).unwrap();
                    counts[y as usize][(x + 1) as usize] = n;
                    let mut z = lv.typed_channel_mut::<f32>("Z").unwrap();
                    for (k, s) in z.at_mut(x, y).unwrap().iter_mut().enumerate() {
                        if *s == 0.0 {
                            *s = tag(x, y, k);
                        }
                    }
                }
            }
        }
        let z = lv.typed_channel::<f32>("Z").unwrap();
        for y in 0..2 {
            for x in -1..3 {
                let got = z.at(x, y).unwrap();
                assert_eq!(got.len(), counts[y as usize][(x + 1) as usize]);
                for (k, &s) in got.iter().enumerate() {
                    assert_eq!(s, tag(x, y, k), "pixel ({x},{y}) sample {k}");
                }
            }
        }
        let total: usize = counts.iter().flatten().sum();
        let sc = lv.sample_counts();
        assert_eq!(sc.total_samples(), total);
        assert!(sc.sample_buffer_size() >= total);
        // Every list has room for its samples and the lists do not overlap (the C++ code forgets to
        // record the room of a moved list, see `SampleCountChannel::set`).
        let mut spans: Vec<_> = (0..sc.num_samples().len())
            .map(|i| (sc.sample_list_positions()[i], sc.sample_list_sizes()[i] as usize, sc.num_samples()[i] as usize))
            .collect();
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].2 <= w[0].1, "room holds the samples");
            assert!(w[0].0 + w[0].1 <= w[1].0 || w[0].1 == 0, "lists overlap: {w:?}");
        }
    }

    #[test]
    fn shrinking_then_growing_zeroes_new_samples() {
        let mut im = img(LevelMode::OneLevel);
        let lv = im.level_mut(0).unwrap();
        lv.set_count(0, 0, 3).unwrap();
        lv.typed_channel_mut::<Half>("R").unwrap().at_mut(0, 0).unwrap().fill(Half::from_f32(1.0));
        lv.set_count(0, 0, 1).unwrap();
        lv.set_count(0, 0, 3).unwrap();
        let r = lv.typed_channel::<Half>("R").unwrap();
        let got: Vec<f32> = r.at(0, 0).unwrap().iter().map(|h| h.to_f32()).collect();
        assert_eq!(got, [1.0, 0.0, 0.0], "the two regrown samples are zero, the kept one survives");
    }

    #[test]
    fn edit_counts_relays_out_and_zeroes() {
        let mut im = img(LevelMode::OneLevel);
        let lv = im.level_mut(0).unwrap();
        lv.edit_counts(|c| c.iter_mut().enumerate().for_each(|(i, n)| *n = i as u32)).unwrap();
        let sc = lv.sample_counts();
        assert_eq!(sc.num_samples(), [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(sc.total_samples(), 28);
        assert_eq!(sc.at(2, 1).unwrap(), 7);
        assert_eq!(lv.typed_channel::<f32>("Z").unwrap().at(2, 1).unwrap().len(), 7);
        lv.clear_counts().unwrap();
        assert_eq!(lv.sample_counts().total_samples(), 0);
        assert_eq!(lv.typed_channel::<f32>("Z").unwrap().at(2, 1).unwrap().len(), 0);
    }

    #[test]
    fn rows_bounds_and_types() {
        let mut im = img(LevelMode::OneLevel);
        let lv = im.level_mut(0).unwrap();
        lv.set_counts_row(1, &[1, 2, 3, 4]).unwrap();
        assert_eq!(lv.sample_counts().row(1).unwrap(), [1, 2, 3, 4]);
        assert!(lv.set_counts_row(2, &[0; 4]).is_err());
        assert!(lv.set_counts_row(0, &[1, 2, 3]).is_err(), "a short row of counts is not silently truncated");
        assert_eq!(lv.sample_counts().row(0).unwrap(), [0, 0, 0, 0]);
        assert!(lv.set_count(3, 0, 1).is_err(), "outside the window");
        assert!(lv.typed_channel::<u32>("Z").is_err(), "wrong sample type");
        assert!(lv.set_count(0, 0, 0x8000_0001).is_err(), "list room would overflow");
    }

    #[test]
    fn a_new_channel_fits_the_existing_layout() {
        let mut im = img(LevelMode::OneLevel);
        im.level_mut(0).unwrap().set_count(1, 1, 5).unwrap();
        im.insert_type("A", PixelType::Uint).unwrap();
        assert_eq!(im.level(0).unwrap().typed_channel::<u32>("A").unwrap().at(1, 1).unwrap(), [0; 5]);
    }

    #[test]
    fn deep_channels_must_be_unsampled() {
        let mut im = img(LevelMode::MipmapLevels);
        assert!(im.insert_channel("S", &Channel::new(PixelType::Half, 2, 1, false)).is_err());
        assert!(im.insert_channel("S", &Channel::new(PixelType::Half, 1, 2, false)).is_err());
        assert_eq!(im.level(1).unwrap().iter().count(), 2);
    }

    #[test]
    fn levels_shift_and_rename() {
        let mut im = img(LevelMode::MipmapLevels);
        im.level_mut(1).unwrap().set_count(-1, 0, 2).unwrap();
        im.shift_pixels(4, 6).unwrap();
        assert_eq!(im.level(1).unwrap().sample_counts().at(3, 6).unwrap(), 2);
        im.rename_channel("Z", "depth").unwrap();
        let ch = im.level(1).unwrap().channel("depth").unwrap();
        assert_eq!(ch.pixel_type(), PixelType::Float);
        assert!(im.level(1).unwrap().channel("Z").is_err());
    }
}
