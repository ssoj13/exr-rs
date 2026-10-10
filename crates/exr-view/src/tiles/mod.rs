//! Tiled display pyramid (`.bughunt/design_tiling.md`): the grid geometry, the
//! one reduction rule for every level, and the byte-budget LRU the RAM tile
//! cache and the GPU atlas share.

pub mod atlas;
pub mod cache;
pub mod direct;
#[cfg(feature = "ofx")]
pub(crate) mod effect;
#[cfg(feature = "ofx")]
pub(crate) mod effect_frames;
#[cfg(feature = "ofx")]
mod effect_geometry;
#[cfg(feature = "ofx")]
pub(crate) use effect_geometry::checked_extent;
#[cfg(feature = "ofx")]
mod effect_input;
#[cfg(all(feature = "ofx", test))]
pub(crate) mod effect_tests;
#[cfg(feature = "ofx")]
pub(crate) mod processed;
pub mod file;
pub mod grid;
pub mod precision;
pub mod reduce;
pub mod region;
pub mod render;
pub mod retire;
pub mod source;
pub mod view;

#[cfg(test)]
mod direct_tests;
#[cfg(test)]
pub(crate) mod gpu_tests;
