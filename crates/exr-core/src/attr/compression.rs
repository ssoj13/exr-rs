//! Port of OpenEXR `ImfCompression.h/.cpp` and `ImfCompressionAttribute.*`.
//!
//! Defines the [`Compression`] enum (12 codecs), helper predicates/queries that
//! mirror the C++ free functions, and the [`AttrValue`] impl that serialises the
//! codec as a single `u8` on disk.
//!
//! # 3.4 extension — HTJ2K (deliberate, documented)
//! The crate's base port target is OpenEXR **v3.3.11**, whose `enum Compression`
//! ends at `DWAB = 9` with `NUM_COMPRESSION_METHODS = 10`.  OpenEXR **3.4** adds
//! two High-Throughput JPEG 2000 codecs *after* DWAB, with the exact
//! discriminants `HTJ2K256_COMPRESSION = 10` (256 scanlines/chunk) and
//! `HTJ2K32_COMPRESSION = 11` (32 scanlines/chunk) — see vcpkg's OpenEXR 3.4
//! `ImfCompression.h`.  We extend our enum with these two values (the exact 3.4
//! discriminants) so exr-core can read/write HTJ2K files that interoperate with
//! a real OpenEXR 3.4.  The sentinel [`Compression::Invalid`] therefore moves to
//! `12` (= `NUM_COMPRESSION_METHODS` in 3.4).  On-disk serialisation stays a
//! single byte (values 0..=11), unchanged in framing.
//!
//! Port targets:
//! * v3.3.11 base — <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompression.h>
//! * v3.3.11 base — <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompression.cpp>
//! * v3.3.11 base — <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompressionAttribute.cpp>
//! * 3.4 HTJ2K   — <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.0/src/lib/OpenEXR/ImfCompression.h>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// Compression enum — mirrors `enum Compression` in ImfCompression.h
// ---------------------------------------------------------------------------

/// All available compression methods.
///
/// Discriminants match the C++ enum: 0–9 from v3.3.11, plus the two 3.4 HTJ2K
/// codecs at 10/11 (see the module-level "3.4 extension" note).
/// `NUM_COMPRESSION_METHODS` (now 12, per 3.4) is used as a sentinel for
/// "invalid" on read, matching the C++ behaviour in `ImfCompressionAttribute.cpp`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    /// No compression (codec id 0).
    NoCompression = 0,
    /// Run-length encoding (codec id 1).
    Rle = 1,
    /// zlib, one scan line at a time (codec id 2).
    Zips = 2,
    /// zlib, 16 scan lines per block (codec id 3).
    Zip = 3,
    /// PIZ wavelet, 32 scan lines per block (codec id 4).
    Piz = 4,
    /// Lossy 24-bit float, 16 scan lines per block (codec id 5).
    Pxr24 = 5,
    /// Lossy 4×4 pixel block, fixed rate (codec id 6).
    B44 = 6,
    /// Lossy 4×4 pixel block, flat-fields compressed more (codec id 7).
    B44a = 7,
    /// Lossy DCT, 32 scan lines per block (codec id 8).
    Dwaa = 8,
    /// Lossy DCT, 256 scan lines per block (codec id 9).
    Dwab = 9,
    /// High-Throughput JPEG 2000 (HTJ2K), **256** scan lines per block
    /// (codec id 10).  **OpenEXR 3.4 extension** — `HTJ2K256_COMPRESSION`.
    /// Lossy (uses the JPEG 2000 reversible path, but quantisation + the RGB
    /// reversible colour transform make round-tripping non-bit-exact in general).
    Htj2k256 = 10,
    /// High-Throughput JPEG 2000 (HTJ2K), **32** scan lines per block
    /// (codec id 11).  **OpenEXR 3.4 extension** — `HTJ2K32_COMPRESSION`.
    Htj2k32 = 11,
    /// Sentinel — written when an unknown byte is encountered on read.
    ///
    /// Mirrors `NUM_COMPRESSION_METHODS` (12 in OpenEXR 3.4) used in
    /// `ImfCompressionAttribute.cpp` to hold invalid values so that
    /// `Header::sanityCheck` can catch them.
    Invalid = 12,
}

impl Compression {
    /// Construct from the raw `u8` wire value.
    ///
    /// Unknown values yield `Compression::Invalid`, mirroring the C++ handling
    /// in `ImfCompressionAttribute::readValueFrom`.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::NoCompression,
            1 => Self::Rle,
            2 => Self::Zips,
            3 => Self::Zip,
            4 => Self::Piz,
            5 => Self::Pxr24,
            6 => Self::B44,
            7 => Self::B44a,
            8 => Self::Dwaa,
            9 => Self::Dwab,
            10 => Self::Htj2k256,
            11 => Self::Htj2k32,
            _ => Self::Invalid,
        }
    }

    /// Returns `true` if `self` is a recognised, non-sentinel value.
    ///
    /// Mirrors `bool isValidCompression(int id)` in `ImfCompression.cpp`.
    /// With the 3.4 HTJ2K extension, valid ids are `0..=11`
    /// (`NUM_COMPRESSION_METHODS == 12`).
    pub fn is_valid(self) -> bool {
        (self as u8) < 12
    }

    /// Returns `true` if this codec does *not* preserve data integrity.
    ///
    /// Mirrors `bool isLossyCompression(Compression id)` — the `lossy` column of
    /// `IdToCompressionDesc` in `ImfCompression.cpp`.  The HTJ2K codecs are
    /// **lossless** there (`"htj2k256"`/`"htj2k32"` carry `lossy = false`): the
    /// JPEG 2000 path is reversible, and our round-trips are bit-exact against
    /// the C++ 3.4.15 encoder/decoder in both directions.
    pub fn is_lossy(self) -> bool {
        matches!(
            self,
            Self::Pxr24 | Self::B44 | Self::B44a | Self::Dwaa | Self::Dwab
        )
    }

    /// Returns `true` if this codec can compress deep data.
    ///
    /// Mirrors `bool isValidDeepCompression(Compression id)`.
    pub fn is_valid_deep(self) -> bool {
        matches!(self, Self::NoCompression | Self::Rle | Self::Zips)
    }

    /// Number of scan lines expected per chunk for this codec.
    ///
    /// Returns `-1` for invalid/sentinel values.
    /// Mirrors `int getCompressionNumScanlines(Compression id)`.
    pub fn num_scanlines(self) -> i32 {
        match self {
            Self::NoCompression | Self::Rle | Self::Zips => 1,
            Self::Zip | Self::Pxr24 => 16,
            Self::Piz | Self::B44 | Self::B44a | Self::Dwaa | Self::Htj2k32 => 32,
            Self::Dwab | Self::Htj2k256 => 256,
            Self::Invalid => -1,
        }
    }

    /// Short lowercase codec name, mirrors `getCompressionNameFromId`.
    pub fn name(self) -> &'static str {
        match self {
            Self::NoCompression => "none",
            Self::Rle => "rle",
            Self::Zips => "zips",
            Self::Zip => "zip",
            Self::Piz => "piz",
            Self::Pxr24 => "pxr24",
            Self::B44 => "b44",
            Self::B44a => "b44a",
            Self::Dwaa => "dwaa",
            Self::Dwab => "dwab",
            // OpenEXR 3.4 `IdToDesc` names.
            Self::Htj2k256 => "htj2k256",
            Self::Htj2k32 => "htj2k32",
            Self::Invalid => "INVALID COMPRESSION ID",
        }
    }

    /// The one-line human-readable description from C++'s `IdToDesc[].desc`
    /// (`ImfCompression.cpp:113-186`) — the second half of
    /// `getCompressionDescriptionFromId`'s `"name: desc"` string.
    fn short_description(self) -> &'static str {
        match self {
            Self::NoCompression => "no compression.",
            Self::Rle => "run-length encoding.",
            Self::Zips => "zlib compression, one scan line at a time.",
            Self::Zip => "zlib compression, in blocks of 16 scan lines.",
            Self::Piz => "piz-based wavelet compression, in blocks of 32 scan lines.",
            Self::Pxr24 => "lossy 24-bit float compression, in blocks of 16 scan lines.",
            Self::B44 => "lossy 4-by-4 pixel block compression, fixed compression rate.",
            Self::B44a => {
                "lossy 4-by-4 pixel block compression, flat fields are compressed more."
            }
            Self::Dwaa => {
                "lossy DCT based compression, in blocks of 32 scanlines. More efficient \
                 for partial buffer access."
            }
            Self::Dwab => {
                "lossy DCT based compression, in blocks of 256 scanlines. More efficient \
                 space wise and faster to decode full frames than DWAA_COMPRESSION."
            }
            Self::Htj2k256 => "High-Throughput JPEG 2000 (256 lines)",
            Self::Htj2k32 => "High-Throughput JPEG 2000 (32 lines)",
            Self::Invalid => "",
        }
    }

    /// Port of `getCompressionDescriptionFromId` — `"<name>: <description>"`,
    /// or the C++ `UNKNOWN_COMPRESSION_ID_MSG` for an out-of-range id (only
    /// reachable here via [`Compression::Invalid`], since every other value
    /// is a real enum discriminant).
    pub fn description(self) -> String {
        if !self.is_valid() {
            return "unrecognized compression method".to_string();
        }
        format!("{}: {}", self.name(), self.short_description())
    }

    /// Port of `getCompressionNamesString` — every real codec's short name
    /// (in enum order, [`Compression::ALL`]) joined by `separator`.
    pub fn names_string(separator: &str) -> String {
        Self::ALL
            .iter()
            .map(|c| c.name())
            .collect::<Vec<_>>()
            .join(separator)
    }

    /// Every real codec (all variants except the `Invalid` sentinel), in
    /// discriminant order. Single source of truth for "which codecs exist" so
    /// callers (e.g. CLI help text) iterate `ALL` + [`Self::name`] instead of
    /// hand-maintaining a list that can drift out of sync.
    pub const ALL: [Compression; 12] = [
        Self::NoCompression,
        Self::Rle,
        Self::Zips,
        Self::Zip,
        Self::Piz,
        Self::Pxr24,
        Self::B44,
        Self::B44a,
        Self::Dwaa,
        Self::Dwab,
        Self::Htj2k256,
        Self::Htj2k32,
    ];

    /// Construct from a name string, case-insensitively.
    ///
    /// Mirrors `getCompressionIdFromName`
    /// (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfCompression.cpp#L234-L238>),
    /// which lowercases the name before looking it up in `CompressionNameToId`
    /// — `"ZIP"` and `"Piz"` must resolve the same as `"zip"` / `"piz"`.
    /// Returns `None` if the name is not known.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "no" | "none" => Some(Self::NoCompression),
            "rle" => Some(Self::Rle),
            "zips" => Some(Self::Zips),
            "zip" => Some(Self::Zip),
            "piz" => Some(Self::Piz),
            "pxr24" => Some(Self::Pxr24),
            "b44" => Some(Self::B44),
            "b44a" => Some(Self::B44a),
            "dwaa" => Some(Self::Dwaa),
            "dwab" => Some(Self::Dwab),
            // OpenEXR 3.4 `CompressionNameToId` entries.
            "htj2k256" => Some(Self::Htj2k256),
            "htj2k32" => Some(Self::Htj2k32),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (1 byte on disk)
// ---------------------------------------------------------------------------

impl AttrValue for Compression {
    fn static_type_name() -> &'static str {
        // ImfCompressionAttribute.cpp: return "compression";
        "compression"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: unsigned char tmp = _value; Xdr::write<StreamIO>(os, tmp);
        xdr::write_u8_val(o, *self as u8)
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // C++ 3.4 core `extract_attr_uint8(.., EXR_COMPRESSION_LAST_TYPE=12)`
        // (`parse_header.c:423-456`, reached for the required `compression` at
        // `:1030-1038` and for any other attribute of this type at `:2142-2147`):
        // `value >= 12` fails the attribute with `EXR_ERR_INVALID_ATTR`, the byte
        // already consumed, so the non-strict header loop drops it and the
        // required-attribute pass defaults `compression` to ZIP. The legacy
        // `readValueFrom`'s NUM_COMPRESSION_METHODS sentinel is never stored on read.
        let raw = xdr::read_u8_val(i)?;
        match Self::from_u8(raw) {
            Self::Invalid => Err(ExrError::Input(format!(
                "compression value {raw} is out of range (expected 0..=11)"
            ))),
            c => Ok(c),
        }
    }
}

/// `typedef TypedAttribute<Compression> CompressionAttribute;`
pub type CompressionAttribute = crate::attr::attribute::TypedAttribute<Compression>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(c: Compression) -> Vec<u8> {
        let mut o = MemOStream::new();
        c.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(Compression::static_type_name(), "compression");
    }

    #[test]
    fn exact_byte_each_variant() {
        // Each codec serialises to its discriminant byte.
        assert_eq!(write_bytes(Compression::NoCompression), vec![0]);
        assert_eq!(write_bytes(Compression::Rle), vec![1]);
        assert_eq!(write_bytes(Compression::Zips), vec![2]);
        assert_eq!(write_bytes(Compression::Zip), vec![3]);
        assert_eq!(write_bytes(Compression::Piz), vec![4]);
        assert_eq!(write_bytes(Compression::Pxr24), vec![5]);
        assert_eq!(write_bytes(Compression::B44), vec![6]);
        assert_eq!(write_bytes(Compression::B44a), vec![7]);
        assert_eq!(write_bytes(Compression::Dwaa), vec![8]);
        assert_eq!(write_bytes(Compression::Dwab), vec![9]);
        // 3.4 HTJ2K extension discriminants.
        assert_eq!(write_bytes(Compression::Htj2k256), vec![10]);
        assert_eq!(write_bytes(Compression::Htj2k32), vec![11]);
    }

    #[test]
    fn round_trip_all_valid() {
        // 0..=11 are all valid (3.4 extends the range with HTJ2K 10/11).
        for raw in 0u8..12 {
            let c = Compression::from_u8(raw);
            assert!(c.is_valid());
            assert_eq!(write_bytes(c), vec![raw]);
            let bytes = vec![raw];
            let mut i = MemIStream::new(bytes);
            let back = Compression::read_value(&mut i, 1, 0).unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn invalid_byte_is_rejected_on_read() {
        // 12 is NUM_COMPRESSION_METHODS in 3.4 (the first invalid value): the core drops the
        // attribute instead of storing the sentinel.
        for raw in [12u8, 13, 200, 255] {
            let mut i = MemIStream::new(vec![raw]);
            assert!(Compression::read_value(&mut i, 1, 0).is_err(), "{raw}");
            assert!(!Compression::from_u8(raw).is_valid());
        }
    }

    #[test]
    fn helpers() {
        assert!(!Compression::NoCompression.is_lossy());
        assert!(Compression::Pxr24.is_lossy());
        assert!(Compression::B44.is_lossy());
        assert!(Compression::Dwab.is_lossy());
        // HTJ2K is LOSSLESS: `ImfCompression.cpp` gives both htj2k descriptors
        // `lossy = false` (verified against vcpkg OpenEXR 3.4.15).
        assert!(!Compression::Htj2k256.is_lossy());
        assert!(!Compression::Htj2k32.is_lossy());
        assert!(Compression::NoCompression.is_valid_deep());
        assert!(Compression::Rle.is_valid_deep());
        assert!(!Compression::Zip.is_valid_deep());
        // HTJ2K is not a valid deep compression.
        assert!(!Compression::Htj2k256.is_valid_deep());
        assert!(!Compression::Htj2k32.is_valid_deep());
        assert_eq!(Compression::Zip.num_scanlines(), 16);
        assert_eq!(Compression::Dwab.num_scanlines(), 256);
        assert_eq!(Compression::Htj2k256.num_scanlines(), 256);
        assert_eq!(Compression::Htj2k32.num_scanlines(), 32);
        assert_eq!(Compression::Invalid.num_scanlines(), -1);
    }

    #[test]
    fn from_name() {
        assert_eq!(Compression::from_name("none"), Some(Compression::NoCompression));
        assert_eq!(Compression::from_name("no"), Some(Compression::NoCompression));
        assert_eq!(Compression::from_name("zip"), Some(Compression::Zip));
        assert_eq!(Compression::from_name("dwab"), Some(Compression::Dwab));
        assert_eq!(Compression::from_name("htj2k256"), Some(Compression::Htj2k256));
        assert_eq!(Compression::from_name("htj2k32"), Some(Compression::Htj2k32));
        assert_eq!(Compression::Htj2k256.name(), "htj2k256");
        assert_eq!(Compression::Htj2k32.name(), "htj2k32");
        assert_eq!(Compression::from_name("unknown"), None);
    }

    /// `from_name` must be case-insensitive, matching C++ 3.4.15
    /// `getCompressionIdFromName` (lowercases the name before lookup).
    #[test]
    fn from_name_case_insensitive() {
        assert_eq!(Compression::from_name("ZIP"), Some(Compression::Zip));
        assert_eq!(Compression::from_name("Piz"), Some(Compression::Piz));
        assert_eq!(Compression::from_name("NONE"), Some(Compression::NoCompression));
        assert_eq!(Compression::from_name("Htj2k256"), Some(Compression::Htj2k256));
    }

    /// `description()` mirrors `getCompressionDescriptionFromId`'s
    /// `"<name>: <desc>"` format, and reports the unknown-id message for `Invalid`.
    #[test]
    fn description_format() {
        assert_eq!(Compression::Zip.description(), "zip: zlib compression, in blocks of 16 scan lines.");
        assert_eq!(Compression::Invalid.description(), "unrecognized compression method");
    }

    /// `names_string` joins every real codec's name in enum
    /// order, matching `getCompressionNamesString`.
    #[test]
    fn names_string_joins_all_codecs() {
        let s = Compression::names_string(", ");
        assert_eq!(s, "none, rle, zips, zip, piz, pxr24, b44, b44a, dwaa, dwab, htj2k256, htj2k32");
    }
}
