//! Port of OpenEXR's OpenEXRUtil library (`openexr/src/lib/OpenEXRUtil`, v3.4.15): in-memory
//! flat and deep images with all their levels, whole-file image I/O, and `CheckFile`.
//!
//! One Rust file per C++ file (`ImfFoo.cpp` -> `foo.rs`).
//!
//! An [`Image`](image::Image) is a set of levels, a level a set of channels, a channel an array of
//! samples; [`FlatImage`](flat_image::FlatImage) has one sample per pixel,
//! [`DeepImage`](deep_image::DeepImage) a list of them. Files of any kind (scanline or tiled, every
//! MIP / RIP level, flat or deep) are loaded and saved with one call:
//!
//! ```
//! use exr_core::attr::{LevelMode, LevelRoundingMode, PixelType};
//! use exr_util::flat_image::FlatImage;
//! use exr_util::flat_image_io::{load_flat_image, save_flat_image};
//! use exr_util::image_data_window::DataWindowSource;
//! use imath_rs::{Box2i, Half, V2i};
//!
//! let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 15, y: 7 } };
//! let mut img = FlatImage::new(&dw, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown)?;
//! img.insert_type("R", PixelType::Half)?;
//! *img.level_mut(2)?.typed_channel_mut::<Half>("R")?.at_mut(1, 1)? = Half::from_f32(0.5);
//!
//! let path = std::env::temp_dir().join(format!("exr-util-doc-{}.exr", std::process::id()));
//! save_flat_image(&path, None, &img, DataWindowSource::Image)?; // tiled: the image has several levels
//! let mut back = FlatImage::empty()?;
//! load_flat_image(&path, &mut back)?;
//! assert_eq!(back.level(2)?.typed_channel::<Half>("R")?.at(1, 1)?.to_f32(), 0.5);
//! assert!(std::fs::remove_file(&path).is_ok());
//! # Ok::<(), exr_core::ExrError>(())
//! ```

pub mod check_file;
pub mod deep_image;
pub mod deep_image_channel;
pub mod deep_image_io;
pub mod deep_image_level;
pub mod flat_image;
pub mod flat_image_channel;
pub mod flat_image_io;
pub mod flat_image_level;
pub mod image;
pub mod image_channel;
pub mod image_channel_renaming;
pub mod image_data_window;
pub mod image_io;
pub mod image_level;
pub mod sample_count_channel;
