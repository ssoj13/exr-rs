//! `GpuError` — the crate's error domain.
//!
//! Wraps `exr_core::ExrError` (decode failures) plus the GPU-bridge-specific
//! failures (a layer with no matching texture format, an unsupported file shape,
//! a wgpu resource error). A distinct type isolates `exr-gpu` from `exr-core`
//! error-variant churn and gives callers one `Result` to match on.

use std::fmt;

/// Errors returned by `exr-gpu`.
#[derive(Debug)]
pub enum GpuError {
    /// An EXR I/O or decode failure bubbled up from `exr-core`.
    Exr(exr_core::ExrError),

    /// The file shape is not (yet) handled by this crate — deep, multipart, or
    /// subsampled via the single-part path. The message names the low-level
    /// `exr-core` API to use instead (e.g. `MultiPartInputFile`). We surface a
    /// hard error rather than half-loading (project rule: no silent fallback).
    Unsupported(String),

    /// A layer's (channel-count × pixel-type) has no matching `wgpu`
    /// `TextureFormat`: > 4 channels, zero channels, or mixed pixel types within
    /// one layer. No lossy coercion is ever performed — the caller must split or
    /// re-type the layer.
    FormatMismatch(String),

    /// A wgpu resource / API error (device, texture, queue).
    Gpu(String),

    /// The image is too large to upload: a zero dimension, a width/height over
    /// the device's `max_texture_dimension_2d`, or a byte size that overflows the
    /// row / allocation arithmetic. Raised BEFORE any wgpu resource is created so
    /// an out-of-range header returns a clean error instead of a wgpu validation
    /// panic (see `UploadLayout`).
    TooLarge(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Exr(e) => write!(f, "EXR decode error: {e}"),
            GpuError::Unsupported(m) => write!(f, "unsupported EXR for exr-gpu: {m}"),
            GpuError::FormatMismatch(m) => write!(f, "no GPU texture format for layer: {m}"),
            GpuError::Gpu(m) => write!(f, "GPU error: {m}"),
            GpuError::TooLarge(m) => write!(f, "image too large for GPU upload: {m}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::Exr(e) => Some(e),
            _ => None,
        }
    }
}

impl From<exr_core::ExrError> for GpuError {
    /// Route the facade's escape-hatch errors. `api::read` returns
    /// `ExrError::Unsupported(kind)` for deep / multipart / subsampled; map
    /// those to `GpuError::Unsupported` so callers can distinguish "wrong file
    /// shape, use the low-level API" from a genuine decode failure.
    fn from(e: exr_core::ExrError) -> Self {
        // Match the structured variant directly — no diagnostic-text scraping.
        // The kind's `Display` carries the human-readable "use <API>" hint;
        // callers that need the *kind* re-classify with
        // `exr_core::classify_unsupported` (see `GpuLoader::open`).
        match e {
            exr_core::ExrError::Unsupported(kind) => GpuError::Unsupported(kind.to_string()),
            other => GpuError::Exr(other),
        }
    }
}

impl From<wgpu::MapRangeError> for GpuError {
    /// wgpu 30 made `BufferSlice::get_mapped_range` fallible; a failed buffer map
    /// is a wgpu resource error, so it routes to `GpuError::Gpu`.
    fn from(e: wgpu::MapRangeError) -> Self {
        GpuError::Gpu(format!("buffer map-range failed: {e}"))
    }
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, GpuError>;
