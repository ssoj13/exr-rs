//! RGBA convenience interface — port of OpenEXR's `ImfRgba.h`, `ImfRgbaYca.*` and
//! (read side) `ImfRgbaFile.*`.
//!
//! Purpose: read **luminance-chroma (subsampled)** EXR files as full-resolution
//! RGBA. The on-disk channels are `Y` (luminance, full res) plus `RY`/`BY` (chroma,
//! subsampled 2x2); reconstructing RGBA requires the fixed 27-tap windowed-sinc
//! filters in [`yca`]. This is the piece the `Image` facade rejects (it only
//! handles full-resolution channels), so `exr-gpu`'s subsampled path lives here.

pub mod aces;
pub mod input;
pub mod lut;
pub mod output;
pub mod pixel;
pub mod tiled_input;
pub mod tiled_output;
pub mod yca;

pub use aces::{aces_chromaticities, AcesInputFile, AcesOutputFile};
pub use input::{RgbaImage, RgbaInputFile};
pub use lut::{round12_log, HalfLut, RgbaLut, RoundNBit};
pub use output::RgbaOutputFile;
pub use pixel::{Rgba, RgbaChannels};
pub use tiled_input::TiledRgbaInputFile;
pub use tiled_output::TiledRgbaOutputFile;
