//! [`VideoError`] — the one typed error of every WRITE path (`ClipWriter`, `write_clip*`, `write_video*`,
//! `transcode`) and of `read_windows`.
//!
//! Why: the writers used to return `Option`, so a disk-full, a bad frame loader, an unrepresentable header
//! and a codec failure were all the same `None` (the CLI even kept a `Mutex<first_err>` to recover a cause).
//! The cause now survives to the CLI message and the C ABI `last_error`. Reads keep
//! [`exv_format::DecodeError`]; [`VideoError::Decode`] wraps it where a read feeds a write.

use std::fmt;
use std::io;

use exv_format::bufpool::AllocError;
use exv_codec::CodecError;
use exv_format::{DecodeError, WriteError};

/// Why a clip write (or a window read feeding one) failed.
#[derive(Debug)]
pub enum VideoError {
    /// The sink failed (disk full, closed pipe, ...). Carries the OS error.
    Io(io::Error),
    /// The clip header cannot be represented in the format (too many channels, long name, bad geometry).
    Header(WriteError),
    /// The tile codec refused a frame (unsupported dtype for the profile, plane/geometry mismatch, backend error).
    Codec(CodecError),
    /// A source frame failed to decode (`.exv` -> `.exv` transcode).
    Decode {
        /// DISPLAY index of the frame when known, else the decode index.
        frame: usize,
        /// The decode error.
        err: DecodeError,
    },
    /// The allocator refused a pooled motion-compensation plane (never an abort).
    Alloc(AllocError),
    /// Frame planes are inconsistent with the header (wrong count/length) or the request is malformed.
    Shape(&'static str),
    /// Frame `frame` differs from the clip's first frame (dimensions, channels or plane sizes).
    Mismatch {
        /// Index of the offending frame.
        frame: usize,
    },
    /// The requested combination is not available on this path (e.g. B-frames on a streaming push).
    Unsupported(&'static str),
    /// A clip needs at least one frame.
    Empty,
    /// More frames than the format's `u32` frame counter can hold.
    TooManyFrames,
    /// The caller's frame loader failed (cause preserved).
    Source {
        /// Index of the frame that could not be loaded.
        frame: usize,
        /// The loader's message.
        msg: String,
    },
    /// An earlier call failed; the writer's output is incomplete and no further call is accepted.
    Poisoned,
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "write failed: {e}"),
            Self::Header(e) => write!(f, "header not representable: {e}"),
            Self::Codec(e) => write!(f, "encode failed: {e}"),
            Self::Decode { frame, err } => write!(f, "frame {frame}: decode failed: {err}"),
            Self::Alloc(e) => write!(f, "motion compensation: {e}"),
            Self::Shape(m) => write!(f, "inconsistent frame: {m}"),
            Self::Mismatch { frame } => write!(f, "frame {frame} differs from frame 0 (dimensions/channels/planes)"),
            Self::Unsupported(m) => write!(f, "unsupported here: {m}"),
            Self::Empty => f.write_str("a clip needs at least one frame"),
            Self::TooManyFrames => f.write_str("too many frames for the u32 frame counter"),
            Self::Source { frame, msg } => write!(f, "frame {frame}: {msg}"),
            Self::Poisoned => f.write_str("writer is unusable after an earlier error"),
        }
    }
}

impl std::error::Error for VideoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Header(e) => Some(e),
            Self::Codec(e) => Some(e),
            Self::Decode { err, .. } => Some(err),
            Self::Alloc(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for VideoError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<WriteError> for VideoError {
    fn from(e: WriteError) -> Self {
        Self::Header(e)
    }
}

impl From<CodecError> for VideoError {
    fn from(e: CodecError) -> Self {
        Self::Codec(e)
    }
}

impl From<AllocError> for VideoError {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}
