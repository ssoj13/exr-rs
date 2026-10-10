//! Cryptomatte errors.

use thiserror::Error;

/// Errors from Cryptomatte metadata / manifest / matte extraction.
#[derive(Debug, Error)]
pub enum Error {
    /// Propagated [`exr_core::ExrError`].
    #[error("exr I/O: {0}")]
    Exr(#[from] exr_core::ExrError),

    /// Malformed or incomplete Cryptomatte header metadata.
    #[error("invalid Cryptomatte metadata: {0}")]
    Metadata(String),

    /// Hash method other than MurmurHash3_32.
    #[error("unsupported hash `{0}` (only MurmurHash3_32)")]
    UnsupportedHash(String),

    /// Conversion method other than `uint32_to_float32` (or the legacy `float`).
    #[error("unsupported conversion `{0}` (only uint32_to_float32)")]
    UnsupportedConversion(String),

    /// An operation this crate does not perform on this kind of file.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Manifest parse / resolve failure.
    #[error("manifest: {0}")]
    Manifest(String),

    /// Filesystem I/O (sidecar manifest, etc.).
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// JSON deserialize error.
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
