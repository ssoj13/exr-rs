//! [`DecodeError`] — the one typed error of every EXV decode/parse path.
//!
//! # Why
//! Decode used to return `Option`, so "file is corrupt", "this would need 400 GB" and "caller asked for
//! frame 99 of 3" were indistinguishable `None`s. A viewer / CLI / C-ABI needs the cause (a budget
//! failure is retryable with a larger [`crate::limits::DecodeLimits`], corruption is not).
//!
//! # Where used
//! `exv-format` limits, `exv-codec` (`decode_channel`/`decode_tile`/`frame`), `exv-video` (`VideoIndex::parse`,
//! `read_video`, `decode_frame`, `Decoder`). The encode side keeps its own `exv_codec::CodecError`.

use std::fmt;

use crate::crc::Mismatch;
use crate::types::CodecId;

/// Why a header's `(codec_id, codec_params)` cannot be opened — the typed answer of the codec registry
/// (`exv_codec::registry::open`). Lives here (not in exv-codec) so it travels inside [`DecodeError::Codec`]
/// through every open/decode path (cli verify, capi, gpu, viewer) with the offending id intact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// `id` is in the spec range but defines no codec (e.g. the reserved 5).
    Unknown(CodecId),
    /// `id` is in the vendor range and no backend with that id is linked into this build.
    Vendor(CodecId),
    /// `id` is `RAW`: a per-blob tag only, no file-level codec/profile.
    NoProfile(CodecId),
    /// Defined spec codec whose backend is not compiled in; `feature` names the cargo feature.
    NotBuilt { id: CodecId, feature: &'static str },
    /// Backend found, but `codec_params` violate its contract; `why` is a static reason.
    BadParams { id: CodecId, why: &'static str },
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "codec {id} is not defined by the format"),
            Self::Vendor(id) => write!(f, "codec {id} is a vendor codec not linked into this build"),
            Self::NoProfile(id) => write!(f, "codec {id} is a per-blob tag, not a file codec"),
            Self::NotBuilt { id, feature } => write!(f, "codec {id} needs the `{feature}` cargo feature (not built)"),
            Self::BadParams { id, why } => write!(f, "codec {id}: bad codec_params: {why}"),
        }
    }
}

impl std::error::Error for OpenError {}

/// Why a decode/parse failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The stream is malformed, truncated or self-inconsistent; the text names the structure that failed.
    Corrupt(&'static str),
    /// Well-formed but not decodable by this build (unsupported dtype/feature/blob tag).
    Unsupported(&'static str),
    /// The header's codec cannot be opened (unknown / vendor / not built / bad params) — see [`OpenError`].
    Codec(OpenError),
    /// The caller's request is invalid for this stream (frame/tile out of range, missing reference).
    Request(&'static str),
    /// Decoding would materialize `need` bytes, over the caller's budget `max` ([`crate::limits::DecodeLimits`]).
    Limit { need: u64, max: u64 },
    /// The allocator refused a `bytes`-sized buffer (hostile or genuinely too large for this machine).
    Alloc(u64),
    /// A tile blob does not match the CRC-32C its frame index records (bit rot, truncated copy, tampering): the bytes
    /// would decode to wrong pixels or fail somewhere deeper. Only raised under [`crate::limits::Verify::On`].
    Checksum(Mismatch),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Corrupt(m) => write!(f, "corrupt stream: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Codec(e) => write!(f, "unsupported: {e}"),
            Self::Request(m) => write!(f, "invalid request: {m}"),
            Self::Limit { need, max } => {
                write!(f, "decode needs {need} bytes, over the {max}-byte budget (raise DecodeLimits::max_alloc)")
            }
            Self::Alloc(n) => write!(f, "allocation of {n} bytes failed"),
            Self::Checksum(m) => {
                write!(f, "checksum mismatch in tile {}", m.tile)?;
                if let Some(c) = m.chan {
                    write!(f, " channel {c}")?;
                }
                write!(f, ": index records crc32c {:#010x}, blob has {:#010x}", m.expected, m.got)
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<Mismatch> for DecodeError {
    fn from(m: Mismatch) -> Self {
        Self::Checksum(m)
    }
}

impl From<OpenError> for DecodeError {
    fn from(e: OpenError) -> Self {
        Self::Codec(e)
    }
}

/// A refused fallible allocation (`bufpool::alloc_zeroed`, `AlignedBuf::zeroed`, `PlanePool::take`) is
/// [`DecodeError::Alloc`]: `?` works in every decode path.
impl From<crate::bufpool::AllocError> for DecodeError {
    fn from(e: crate::bufpool::AllocError) -> Self {
        Self::Alloc(e.bytes as u64)
    }
}

/// Lets `io::Result` code (the CLI, the C-ABI bridge) propagate a decode failure with `?`.
impl From<DecodeError> for std::io::Error {
    fn from(e: DecodeError) -> Self {
        let kind = match e {
            DecodeError::Corrupt(_) | DecodeError::Unsupported(_) | DecodeError::Codec(_) | DecodeError::Checksum(_) => {
                std::io::ErrorKind::InvalidData
            }
            DecodeError::Request(_) => std::io::ErrorKind::InvalidInput,
            DecodeError::Limit { .. } | DecodeError::Alloc(_) => std::io::ErrorKind::OutOfMemory,
        };
        std::io::Error::new(kind, e)
    }
}
