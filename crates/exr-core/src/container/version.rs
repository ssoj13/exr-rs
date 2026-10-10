//! Port of OpenEXR `ImfVersion.h` / `ImfVersion.cpp` — magic number and version flags.
//!
//! Every OpenEXR file starts with a 4-byte magic number followed by a 4-byte
//! version word.  The version word is split into an 8-bit version number and a
//! 24-bit flags field.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfVersion.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfVersion.cpp>

// ---------------------------------------------------------------------------
// Magic number
// ---------------------------------------------------------------------------

/// The 4-byte magic number stored at the start of every OpenEXR file.
///
/// C++: `static const int MAGIC = 20000630;`  
/// `20000630` decimal = `0x01312F76`.  
/// Stored on disk in **little-endian** order: bytes `0x76 0x2F 0x31 0x01`.
pub const MAGIC: u32 = 20_000_630;

// ---------------------------------------------------------------------------
// Version word fields
// ---------------------------------------------------------------------------

/// Mask for the 8-bit version number field (low byte).
///
/// C++: `static const int VERSION_NUMBER_FIELD = 0x000000ff`.
pub const VERSION_NUMBER_FIELD: u32 = 0x0000_00FF;

/// Mask for the 24-bit flags field (high three bytes).
///
/// C++: `static const int VERSION_FLAGS_FIELD = 0xffffff00`.
pub const VERSION_FLAGS_FIELD: u32 = 0xFFFF_FF00;

// ---------------------------------------------------------------------------
// Version number
// ---------------------------------------------------------------------------

/// The current EXR version number stored in the low byte of the version word.
///
/// C++: `static const int EXR_VERSION = 2`.
pub const EXR_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Flag bits (live in the upper 24 bits of the version word)
// ---------------------------------------------------------------------------

/// File is tiled (as opposed to scanline-only).
///
/// C++: `static const int TILED_FLAG = 0x00000200`.
pub const TILED_FLAG: u32 = 0x0000_0200;

/// File contains long attribute or channel names (> 31 characters).
///
/// C++: `static const int LONG_NAMES_FLAG = 0x00000400`.
pub const LONG_NAMES_FLAG: u32 = 0x0000_0400;

/// File has at least one non-image (deep) part.
///
/// C++: `static const int NON_IMAGE_FLAG = 0x00000800`.
pub const NON_IMAGE_FLAG: u32 = 0x0000_0800;

/// File has multiple parts.
///
/// C++: `static const int MULTI_PART_FILE_FLAG = 0x00001000`.
pub const MULTI_PART_FILE_FLAG: u32 = 0x0000_1000;

/// Bitwise OR of all known flag bits.
///
/// C++: `static const int ALL_FLAGS = TILED_FLAG | LONG_NAMES_FLAG | NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG`.
pub const ALL_FLAGS: u32 = TILED_FLAG | LONG_NAMES_FLAG | NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG;

// ---------------------------------------------------------------------------
// Predicate functions — port of the inline functions in ImfVersion.h
// ---------------------------------------------------------------------------

/// `bool isTiled(int version)` — true if the tiled flag is set.
#[inline]
pub fn is_tiled(version: u32) -> bool {
    version & TILED_FLAG != 0
}

/// `bool isMultiPart(int version)` — true if the multi-part flag is set.
#[inline]
pub fn is_multi_part(version: u32) -> bool {
    version & MULTI_PART_FILE_FLAG != 0
}

/// `bool isNonImage(int version)` — true if the non-image (deep) flag is set.
#[inline]
pub fn is_non_image(version: u32) -> bool {
    version & NON_IMAGE_FLAG != 0
}

/// `bool isSinglePartGeneral` is implicit: a file is "general" (single-part
/// scanline/tiled) when it has neither the deep nor the multi-part flag.
///
/// Not a C++ function but used in several readers; included for completeness.
#[inline]
pub fn is_single_part_general(version: u32) -> bool {
    !is_multi_part(version) && !is_non_image(version)
}

/// `bool supportsFlags(int flags)` — true if `flags` has no unknown bits set.
///
/// C++: `return !(flags & ~ALL_FLAGS)`.
#[inline]
pub fn supports_flags(flags: u32) -> bool {
    flags & !ALL_FLAGS == 0
}

/// `int getVersion(int version)` — extract the 8-bit version number.
///
/// C++: `return version & VERSION_NUMBER_FIELD`.
#[inline]
pub fn get_version(version: u32) -> u32 {
    version & VERSION_NUMBER_FIELD
}

/// `int getFlags(int version)` — extract the 24-bit flags word.
///
/// C++: `return version & VERSION_FLAGS_FIELD`.
#[inline]
pub fn get_flags(version: u32) -> u32 {
    version & VERSION_FLAGS_FIELD
}

// ---------------------------------------------------------------------------
// Mutators — port of the inline functions in ImfVersion.h
// ---------------------------------------------------------------------------

/// `int makeTiled(int version)` — set the tiled flag.
#[inline]
pub fn make_tiled(version: u32) -> u32 {
    version | TILED_FLAG
}

/// `int makeNotTiled(int version)` — clear the tiled flag.
#[inline]
pub fn make_not_tiled(version: u32) -> u32 {
    version & !TILED_FLAG
}

/// Port of `Imf::getLibraryVersion` (`ImfMisc.h/.cpp`) — the OpenEXR release
/// this port's on-disk format and behaviour targets.
///
/// C++ returns its own build's `OPENEXR_VERSION_STRING`; since this is a
/// from-scratch Rust port rather than a build of the C++ library, there is no
/// analogous compiled-in constant — this reports the version this crate's
/// container/attribute/codec layer is written against (the behavioural target is C++
/// 3.4.15), which is exactly what
/// a caller checking `getLibraryVersion()` for compatibility purposes wants.
///
/// [`Imf::WidenFilename`](https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfMisc.cpp)
/// (UTF-8 -> `wstring` for the Windows wide-character file APIs) has no
/// port: every `IStream`/`OStream` implementation here takes a plain Rust
/// `&str` path and opens it through `std::fs::File`, which already handles
/// Unicode paths correctly on every platform (including Windows, via `std`'s
/// own UTF-8 <-> UTF-16 conversion) — there is no FFI boundary that needs a
/// manually widened `wchar_t*`, so the function would have no callers.
pub const LIBRARY_VERSION: &str = "3.4.15";

/// Function form of [`LIBRARY_VERSION`] — mirrors the C++ call syntax
/// `getLibraryVersion()`.
pub fn get_library_version() -> &'static str {
    LIBRARY_VERSION
}

// ---------------------------------------------------------------------------
// Full version-word validation — port of the checks OpenEXRCore's
// `parse_header.c` runs right after the magic number, in every C++ 3.4
// reader's non-strict path (`InputFile`/`MultiPartInputFile` construction).
// ---------------------------------------------------------------------------

/// Validate a raw version word the way the C++ readers do at open time — the
/// version number and the flag combination, on top of
/// [`supports_flags`] (unknown flag bits).
///
/// C++ 3.4 core:
/// * `if (ctxt->version != 2)` -> `EXR_ERR_FILE_BAD_HEADER`
///   (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L2687>).
/// * `TILED_FLAG` together with `NON_IMAGE_FLAG` or `MULTI_PART_FILE_FLAG` ->
///   "Invalid combination of version flags"
///   (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L2783>).
///
/// Every header-only entry point that does not otherwise route through a
/// reader's `read_prologue` (`Header::read_checked`, the raw-chunk
/// passthrough) must call this alongside [`supports_flags`] — the
/// reader constructors already reject these inputs via
/// `multipart::input::read_prologue`.
pub fn validate_version_word(version_word: u32) -> crate::error::Result<()> {
    if get_version(version_word) != EXR_VERSION {
        return Err(crate::error::ExrError::Input(format!(
            "Unsupported file format version {} (expected {EXR_VERSION})",
            get_version(version_word)
        )));
    }
    let flags = get_flags(version_word);
    if !supports_flags(flags) {
        return Err(crate::error::ExrError::Input(
            "The file format version number's flag field contains unknown flags.".into(),
        ));
    }
    if is_tiled(flags) && (is_non_image(flags) || is_multi_part(flags)) {
        return Err(crate::error::ExrError::Input(
            "Invalid combination of version flags.".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Magic check — port of `bool isImfMagic(const char bytes[4])`
// ---------------------------------------------------------------------------

/// Check whether a 4-byte prefix matches the EXR magic number.
///
/// C++ (`ImfVersion.cpp`):
/// ```text
/// bytes[0] == (MAGIC >> 0)  & 0xff &&
/// bytes[1] == (MAGIC >> 8)  & 0xff &&
/// bytes[2] == (MAGIC >> 16) & 0xff &&
/// bytes[3] == (MAGIC >> 24) & 0xff
/// ```
/// i.e. the magic is stored in **little-endian** order in the file.
#[inline]
pub fn is_imf_magic(bytes: &[u8; 4]) -> bool {
    bytes[0] == (MAGIC & 0xFF) as u8
        && bytes[1] == ((MAGIC >> 8) & 0xFF) as u8
        && bytes[2] == ((MAGIC >> 16) & 0xFF) as u8
        && bytes[3] == ((MAGIC >> 24) & 0xFF) as u8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_value() {
        // MAGIC = 20000630 = 0x01312F76
        assert_eq!(MAGIC, 0x0131_2F76);
    }

    #[test]
    fn magic_on_disk_bytes() {
        // LE on disk: 76 2F 31 01
        assert_eq!((MAGIC & 0xFF) as u8, 0x76);
        assert_eq!(((MAGIC >> 8) & 0xFF) as u8, 0x2F);
        assert_eq!(((MAGIC >> 16) & 0xFF) as u8, 0x31);
        assert_eq!(((MAGIC >> 24) & 0xFF) as u8, 0x01);
    }

    #[test]
    fn is_imf_magic_true() {
        assert!(is_imf_magic(&[0x76, 0x2F, 0x31, 0x01]));
    }

    #[test]
    fn is_imf_magic_false() {
        assert!(!is_imf_magic(&[0x01, 0x2F, 0x31, 0x76]));
        assert!(!is_imf_magic(&[0x00, 0x00, 0x00, 0x00]));
    }

    #[test]
    fn exr_version_number() {
        assert_eq!(EXR_VERSION, 2);
    }

    #[test]
    fn get_version_and_flags() {
        let word = EXR_VERSION | TILED_FLAG;
        assert_eq!(get_version(word), EXR_VERSION);
        assert_eq!(get_flags(word), TILED_FLAG);
    }

    #[test]
    fn flag_predicates() {
        let plain = EXR_VERSION;
        let tiled = EXR_VERSION | TILED_FLAG;
        let multi = EXR_VERSION | MULTI_PART_FILE_FLAG;
        let deep = EXR_VERSION | NON_IMAGE_FLAG;

        assert!(!is_tiled(plain));
        assert!(is_tiled(tiled));
        assert!(!is_multi_part(plain));
        assert!(is_multi_part(multi));
        assert!(!is_non_image(plain));
        assert!(is_non_image(deep));

        assert!(is_single_part_general(plain));
        assert!(is_single_part_general(tiled));
        assert!(!is_single_part_general(multi));
        assert!(!is_single_part_general(deep));
    }

    #[test]
    fn supports_flags_known() {
        assert!(supports_flags(ALL_FLAGS));
        assert!(supports_flags(TILED_FLAG));
        assert!(supports_flags(0)); // no flags is fine
    }

    #[test]
    fn supports_flags_unknown() {
        // A flag bit not in ALL_FLAGS should be rejected.
        assert!(!supports_flags(0x0002_0000));
    }

    #[test]
    fn make_tiled_roundtrip() {
        let v = EXR_VERSION;
        let t = make_tiled(v);
        assert!(is_tiled(t));
        let nt = make_not_tiled(t);
        assert!(!is_tiled(nt));
        assert_eq!(get_version(nt), EXR_VERSION);
    }

    /// A version number other than 2 must be rejected.
    #[test]
    fn validate_version_word_rejects_bad_version() {
        assert!(validate_version_word(3).is_err());
        assert!(validate_version_word(EXR_VERSION).is_ok());
    }

    /// TILED combined with NON_IMAGE or MULTI_PART is an invalid
    /// combination of flags.
    #[test]
    fn validate_version_word_rejects_bad_flag_combo() {
        assert!(validate_version_word(EXR_VERSION | TILED_FLAG | NON_IMAGE_FLAG).is_err());
        assert!(validate_version_word(EXR_VERSION | TILED_FLAG | MULTI_PART_FILE_FLAG).is_err());
        assert!(validate_version_word(EXR_VERSION | TILED_FLAG).is_ok());
        assert!(validate_version_word(EXR_VERSION | NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG).is_ok());
    }

    #[test]
    fn library_version_reports_the_oracle_target() {
        assert_eq!(get_library_version(), "3.4.15");
        assert_eq!(get_library_version(), LIBRARY_VERSION);
    }

    #[test]
    fn all_flags_mask() {
        assert_eq!(
            ALL_FLAGS,
            TILED_FLAG | LONG_NAMES_FLAG | NON_IMAGE_FLAG | MULTI_PART_FILE_FLAG
        );
    }
}
