//! The layer → `wgpu::TextureFormat` contract — the ONE decision the whole crate
//! hangs on (all three design reviews agreed this is what matters).
//!
//! Rules (deliberate, written down before any upload, no run-time guessing):
//! - A layer's channel count picks the component arity: 1 → `R`, 2 → `Rg`,
//!   3 or 4 → `Rgba`. wgpu has NO 3-component format, so 3-channel layers are
//!   PADDED to 4 (the pad component carries 1.0/opaque for colour layers, 0.0 for
//!   vector/unknown layers — see `upload`).
//! - The sample type picks 16- vs 32-bit, NEVER lossy: `Half` → `*16Float`,
//!   `Float` → `*32Float`, `Uint` → `*32Uint`.
//! - More than 4 channels, 0 channels, or mixed pixel types within a layer → a
//!   hard `FormatMismatch` (project rule: hard error over silent fallback).
//!   Mixed-type is checked in `upload` (it needs the per-channel types) — arity
//!   and type here.
//!
//! Everything is delivered LINEAR scene-referred — no tone-map, no sRGB. Colour
//! management is the consumer's job (e.g. the viewer's colour pass).

use exr_core::attr::PixelType;
use wgpu::TextureFormat;

use crate::error::{GpuError, Result};

/// The GPU format chosen for a layer plus how it was packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerFormat {
    /// The `wgpu` texture format the layer uploads as.
    pub format: TextureFormat,
    /// Packed component count in the texture: 1, 2, or 4. A 3-channel layer
    /// reports 4 (the 4th is the pad).
    pub components: u8,
    /// Bytes per component (2 for `Half`, 4 for `Float`/`Uint`).
    pub bytes_per_component: u8,
    /// `true` if a pad component was added (only for 3-channel layers → RGBA).
    pub padded: bool,
}

impl LayerFormat {
    /// Bytes one packed texel occupies (`components * bytes_per_component`).
    pub fn texel_size(&self) -> u32 {
        self.components as u32 * self.bytes_per_component as u32
    }
}

/// A validated byte-layout for uploading one layer to a 2-D texture — the SINGLE
/// source of truth for every size the pack / upload / readback / estimate paths
/// need (row stride, GPU allocation, CPU buffer length).
///
/// Every product is range-checked here, BEFORE any wgpu resource is created, so a
/// valid-header image whose width/height exceeds the device's
/// `max_texture_dimension_2d`, or whose byte size overflows the arithmetic,
/// returns a clean [`GpuError::TooLarge`] instead of tripping a raw wgpu
/// validation panic downstream. Both constructors are pure (no `Device`) so the
/// size/limit math is unit-testable with a synthetic maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadLayout {
    /// Texture width in texels (guaranteed non-zero and `<= max_dim`).
    pub width: u32,
    /// Texture height in texels (guaranteed non-zero and `<= max_dim`).
    pub height: u32,
    /// Bytes one packed texel occupies (`components * bytes_per_component`).
    pub texel_size: u32,
    /// Tightly-packed row stride in bytes (`width * texel_size`), checked `u32`.
    pub row_bytes: u32,
    /// GPU texture / CPU pack allocation in bytes (`row_bytes * height`), checked
    /// `u64` — the value `vram_estimate` reports and the upload/readback size.
    pub byte_size: u64,
    /// `byte_size` validated to fit `usize`, for CPU `Vec` sizing on any target.
    pub cpu_size: usize,
}

impl UploadLayout {
    /// Validate + build the layout for `width x height` texels of `texel_size`
    /// bytes, enforcing the device's `max_texture_dimension_2d`. Use this at any
    /// upload seam where the limit is known
    /// (`device.limits().max_texture_dimension_2d`).
    ///
    /// `texel_size` is passed explicitly (rather than a [`LayerFormat`]) so the
    /// ONE checked sizing path also serves consumers that pick their own texture
    /// format — `exr-view`'s display target is the other caller, and it must not
    /// grow a second copy of this arithmetic.
    pub fn compute(width: u32, height: u32, texel_size: u32, max_dim: u32) -> Result<Self> {
        if width > max_dim || height > max_dim {
            return Err(GpuError::TooLarge(format!(
                "image {width}x{height} exceeds the device max texture dimension ({max_dim})"
            )));
        }
        Self::for_texture(width, height, texel_size)
    }

    /// Validate + build the layout from an explicit texel size, WITHOUT a device
    /// limit. Used for readback (dimensions already validated at upload) and for
    /// the device-free `vram_estimate`. Rejects a zero dimension and any byte
    /// product that overflows.
    pub fn for_texture(width: u32, height: u32, texel_size: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(GpuError::TooLarge(format!(
                "image has a zero dimension ({width}x{height})"
            )));
        }
        let row_bytes = width.checked_mul(texel_size).ok_or_else(|| {
            GpuError::TooLarge(format!(
                "row byte size overflows u32 ({width} texels x {texel_size} B/texel)"
            ))
        })?;
        let byte_size = u64::from(row_bytes).checked_mul(u64::from(height)).ok_or_else(|| {
            GpuError::TooLarge(format!(
                "image byte size overflows u64 ({row_bytes} B/row x {height} rows)"
            ))
        })?;
        let cpu_size = usize::try_from(byte_size).map_err(|_| {
            GpuError::TooLarge(format!(
                "image byte size {byte_size} exceeds this platform's addressable memory"
            ))
        })?;
        Ok(Self { width, height, texel_size, row_bytes, byte_size, cpu_size })
    }
}

/// Map (channel count, uniform pixel type) → [`LayerFormat`].
///
/// `n_channels` is the layer's real channel count (before padding). Caller must
/// have already verified all channels share `pixel_type` (see `upload`).
pub fn select_format(n_channels: usize, pixel_type: PixelType) -> Result<LayerFormat> {
    use PixelType::{Float, Half, Uint};
    use TextureFormat::{
        R16Float, R32Float, R32Uint, Rg16Float, Rg32Float, Rg32Uint, Rgba16Float, Rgba32Float,
        Rgba32Uint,
    };

    let bpc: u8 = match pixel_type {
        Half => 2,
        Float | Uint => 4,
        // `NumPixelTypes` is an internal sentinel that exr-core rejects in its
        // header sanity check, so it can never reach us; refuse it loudly anyway.
        _ => {
            return Err(GpuError::FormatMismatch(
                "channel has an invalid pixel-type sentinel".into(),
            ));
        }
    };

    // (arity, type) → (format, packed_components, padded)
    let (format, components, padded) = match (n_channels, pixel_type) {
        (1, Half) => (R16Float, 1, false),
        (1, Float) => (R32Float, 1, false),
        (1, Uint) => (R32Uint, 1, false),

        (2, Half) => (Rg16Float, 2, false),
        (2, Float) => (Rg32Float, 2, false),
        (2, Uint) => (Rg32Uint, 2, false),

        // 3 channels pad to RGBA (no Rgb* texture format exists in wgpu).
        (3, Half) => (Rgba16Float, 4, true),
        (3, Float) => (Rgba32Float, 4, true),
        (3, Uint) => (Rgba32Uint, 4, true),

        (4, Half) => (Rgba16Float, 4, false),
        (4, Float) => (Rgba32Float, 4, false),
        (4, Uint) => (Rgba32Uint, 4, false),

        (0, _) => {
            return Err(GpuError::FormatMismatch("layer has no channels".into()));
        }
        (n, _) => {
            return Err(GpuError::FormatMismatch(format!(
                "layer has {n} channels; wgpu textures hold at most 4 — split the layer, \
                 or use the planar storage-buffer path (exr-gpu Phase D)"
            )));
        }
    };

    Ok(LayerFormat {
        format,
        components,
        bytes_per_component: bpc,
        padded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_and_type_to_format() {
        assert_eq!(select_format(1, PixelType::Half).unwrap().format, TextureFormat::R16Float);
        assert_eq!(select_format(2, PixelType::Float).unwrap().format, TextureFormat::Rg32Float);
        assert_eq!(select_format(1, PixelType::Uint).unwrap().format, TextureFormat::R32Uint);

        // 3-channel pads to RGBA and reports padded=true, components=4.
        let f3 = select_format(3, PixelType::Half).unwrap();
        assert_eq!(f3.format, TextureFormat::Rgba16Float);
        assert_eq!(f3.components, 4);
        assert!(f3.padded);

        let f4 = select_format(4, PixelType::Float).unwrap();
        assert_eq!(f4.format, TextureFormat::Rgba32Float);
        assert!(!f4.padded);
        assert_eq!(f4.bytes_per_component, 4);
    }

    #[test]
    fn over_four_and_zero_are_hard_errors() {
        assert!(matches!(select_format(5, PixelType::Half), Err(GpuError::FormatMismatch(_))));
        assert!(matches!(select_format(0, PixelType::Half), Err(GpuError::FormatMismatch(_))));
    }

    #[test]
    fn upload_layout_accepts_ordinary_layers() {
        // 3-channel Half padded to RGBA16 (8 B/texel).
        let fmt = select_format(3, PixelType::Half).unwrap();
        let l = UploadLayout::compute(1920, 1080, fmt.texel_size(), 8192).unwrap();
        assert_eq!(l.texel_size, 8);
        assert_eq!(l.row_bytes, 1920 * 8);
        assert_eq!(l.byte_size, 1920 * 1080 * 8);
        assert_eq!(l.cpu_size, 1920 * 1080 * 8);

        // 4-channel Float (Rgba32Float, 16 B/texel).
        let fmt = select_format(4, PixelType::Float).unwrap();
        let l = UploadLayout::compute(64, 64, fmt.texel_size(), 8192).unwrap();
        assert_eq!(l.texel_size, 16);
        assert_eq!(l.byte_size, 64 * 64 * 16);
    }

    #[test]
    fn upload_layout_rejects_zero_dimension() {
        let fmt = select_format(3, PixelType::Half).unwrap();
        assert!(matches!(UploadLayout::compute(0, 16, fmt.texel_size(), 8192), Err(GpuError::TooLarge(_))));
        assert!(matches!(UploadLayout::compute(16, 0, fmt.texel_size(), 8192), Err(GpuError::TooLarge(_))));
        assert!(matches!(UploadLayout::for_texture(0, 16, 8), Err(GpuError::TooLarge(_))));
    }

    #[test]
    fn upload_layout_rejects_over_max_dimension() {
        let fmt = select_format(4, PixelType::Half).unwrap();
        // Both just over a synthetic 4096 device limit.
        assert!(matches!(UploadLayout::compute(4097, 100, fmt.texel_size(), 4096), Err(GpuError::TooLarge(_))));
        assert!(matches!(UploadLayout::compute(100, 4097, fmt.texel_size(), 4096), Err(GpuError::TooLarge(_))));
        // Exactly at the limit is accepted.
        assert!(UploadLayout::compute(4096, 4096, fmt.texel_size(), 4096).is_ok());
    }

    #[test]
    fn upload_layout_rejects_overflowing_byte_products() {
        // width * texel_size overflows u32 (u32::MAX texels * 8 B).
        assert!(matches!(
            UploadLayout::for_texture(u32::MAX, 2, 8),
            Err(GpuError::TooLarge(_))
        ));
    }
}
