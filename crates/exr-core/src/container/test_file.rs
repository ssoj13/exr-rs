//! Port of OpenEXR `ImfTestFile.h/.cpp` — quick "is this an OpenEXR file, and
//! is it tiled/deep/multipart" probes.
//!
//! C++ never throws from these: any I/O failure (missing file, truncated
//! read, …) is caught and reported as `false`. This port mirrors that by
//! reading through the fallible [`IStream`] API and mapping any `Err` the
//! same way, rather than propagating it.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfTestFile.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfTestFile.cpp>

use crate::container::{version, xdr};
use crate::io::std_io::StdIfStream;
use crate::io::stream::IStream;

/// The four flags [`is_open_exr_file_stream`] reports (C++'s four
/// `bool&` out-parameters, collapsed into one return value since Rust has no
/// reference out-parameters and no partially-modified-on-exception state to
/// preserve — C++'s "on exception, `tiled` is set false, everything else
/// keeps its caller-supplied value" is not a contract worth reproducing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExrFileProbe {
    /// The magic number matched (`magic == MAGIC`) — "is this an OpenEXR file at all".
    pub is_exr: bool,
    /// The version word's `TILED_FLAG` bit.
    pub is_tiled: bool,
    /// The version word's `NON_IMAGE_FLAG` bit (i.e. contains deep data).
    pub is_deep: bool,
    /// The version word's `MULTI_PART_FILE_FLAG` bit.
    pub is_multi_part: bool,
}

/// `bool isOpenExrFile(IStream&, bool&, bool&, bool&)` — read the magic
/// number and version word from `stream` and report what they say.
///
/// Note this only reads two `i32`s (8 bytes) and does not validate the
/// version number or flag combination the way [`version::validate_version_word`]
/// does — it is a cheap probe, exactly as lightweight as C++'s.
pub fn is_open_exr_file_stream(stream: &mut dyn IStream) -> ExrFileProbe {
    let probe = (|| -> crate::error::Result<ExrFileProbe> {
        let magic = xdr::read_i32(stream)?;
        let version_word = xdr::read_i32(stream)? as u32;
        let flags = version::get_flags(version_word);
        Ok(ExrFileProbe {
            is_exr: (magic as u32) == version::MAGIC,
            is_tiled: version::is_tiled(flags),
            is_deep: version::is_non_image(flags),
            is_multi_part: version::is_multi_part(flags),
        })
    })();
    // C++ `catch (...) { tiled = false; return false; }` — any read failure
    // is "not an OpenEXR file", every flag false.
    probe.unwrap_or_default()
}

/// `bool isOpenExrFile(const char[], bool&, bool&, bool&)` — open `path` and
/// probe it. A missing/unreadable file is "not an OpenEXR file" (`is_exr =
/// false`), not an error, matching C++'s `catch (...)`.
pub fn is_open_exr_file(path: &str) -> ExrFileProbe {
    match StdIfStream::open(path) {
        Ok(mut stream) => is_open_exr_file_stream(&mut stream),
        Err(_) => ExrFileProbe::default(),
    }
}

/// `bool isTiledOpenExrFile(const char[])`.
pub fn is_tiled_open_exr_file(path: &str) -> bool {
    let probe = is_open_exr_file(path);
    probe.is_exr && probe.is_tiled
}

/// `bool isDeepOpenExrFile(const char[])`.
pub fn is_deep_open_exr_file(path: &str) -> bool {
    let probe = is_open_exr_file(path);
    probe.is_exr && probe.is_deep
}

/// `bool isMultiPartOpenExrFile(const char[])`.
pub fn is_multi_part_open_exr_file(path: &str) -> bool {
    let probe = is_open_exr_file(path);
    probe.is_exr && probe.is_multi_part
}

/// `bool isTiledOpenExrFile(IStream&)`.
pub fn is_tiled_open_exr_file_stream(stream: &mut dyn IStream) -> bool {
    let probe = is_open_exr_file_stream(stream);
    probe.is_exr && probe.is_tiled
}

/// `bool isDeepOpenExrFile(IStream&)`.
pub fn is_deep_open_exr_file_stream(stream: &mut dyn IStream) -> bool {
    let probe = is_open_exr_file_stream(stream);
    probe.is_exr && probe.is_deep
}

/// `bool isMultiPartOpenExrFile(IStream&)`.
pub fn is_multi_part_open_exr_file_stream(stream: &mut dyn IStream) -> bool {
    let probe = is_open_exr_file_stream(stream);
    probe.is_exr && probe.is_multi_part
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::MemIStream;

    fn version_word_bytes(magic: u32, version_word: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic.to_le_bytes());
        bytes.extend_from_slice(&version_word.to_le_bytes());
        bytes
    }

    #[test]
    fn detects_plain_scanline_file() {
        let mut s = MemIStream::new(version_word_bytes(version::MAGIC, version::EXR_VERSION));
        let probe = is_open_exr_file_stream(&mut s);
        assert_eq!(
            probe,
            ExrFileProbe { is_exr: true, is_tiled: false, is_deep: false, is_multi_part: false }
        );
    }

    #[test]
    fn detects_every_flag() {
        let word = version::EXR_VERSION | version::TILED_FLAG;
        let mut s = MemIStream::new(version_word_bytes(version::MAGIC, word));
        assert!(is_tiled_open_exr_file_stream(&mut s));

        let word = version::EXR_VERSION | version::NON_IMAGE_FLAG;
        let mut s = MemIStream::new(version_word_bytes(version::MAGIC, word));
        assert!(is_deep_open_exr_file_stream(&mut s));

        let word = version::EXR_VERSION | version::MULTI_PART_FILE_FLAG;
        let mut s = MemIStream::new(version_word_bytes(version::MAGIC, word));
        assert!(is_multi_part_open_exr_file_stream(&mut s));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut s = MemIStream::new(version_word_bytes(0xDEAD_BEEF, version::EXR_VERSION));
        let probe = is_open_exr_file_stream(&mut s);
        assert!(!probe.is_exr);
    }

    /// C++ `catch (...)` on any I/O failure (e.g. a file too short to even
    /// hold magic + version) reports "not an OpenEXR file", not an error.
    #[test]
    fn truncated_stream_reports_not_exr_rather_than_erroring() {
        let mut s = MemIStream::new(vec![0x76, 0x2F]); // 2 bytes, nowhere near 8
        let probe = is_open_exr_file_stream(&mut s);
        assert!(!probe.is_exr);
    }

    #[test]
    fn missing_file_reports_not_exr() {
        let probe = is_open_exr_file("/nonexistent/path/that/should/not/exist.exr");
        assert!(!probe.is_exr);
        assert!(!is_tiled_open_exr_file("/nonexistent/path/that/should/not/exist.exr"));
    }
}
