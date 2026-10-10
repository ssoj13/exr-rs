//! Port of OpenEXR `ImfIDManifestAttribute.h/.cpp` + `ImfIDManifest.h/.cpp` —
//! the `"idmanifest"` attribute, its on-disk [`CompressedIDManifest`]
//! payload, and the full structured [`IDManifest`] model.
//!
//! # Module layout
//! * `mod.rs` (this file) — the attribute wire format
//!   (`IDManifestAttribute::writeValueTo`/`readValueFrom`, byte-exact) and
//!   [`CompressedIDManifest`]'s zlib bridge to/from a raw byte payload.
//! * [`manifest`] — [`IDManifest`] / [`ChannelGroupManifest`] / [`IdLifetime`]:
//!   the structured table itself (accessors, `insert`/`find`/`erase`,
//!   `merge`, `PartialEq`) — ported from `ImfIDManifest.h`'s class
//!   declarations.
//! * `codec` — [`IDManifest`]'s own binary (de)serialisation
//!   (`IDManifest::init`/`serialize`), the prefix-compressed string table +
//!   RLE index remapping scheme.
//! * `varint` — the shared binary primitives `codec` builds on (variable
//!   length integers, Pascal strings, length-prefixed string lists).
//! * `hash` — `IDManifest::MurmurHash32`/`MurmurHash64` (via the `murmur3`
//!   crate) + the `catString` join helper.
//!
//! # Attribute wire format (byte-exact, ported 1:1)
//!   * first 8 bytes: `uncompressedDataSize` (`u64`, the size the payload would
//!     have once zlib-decompressed),
//!   * remaining `size - 8` bytes: the raw, **already zlib-compressed** manifest
//!     bytes, stored verbatim.
//!
//! This is exactly what the C++ `CompressedIDManifest` struct holds
//! (`_uncompressedDataSize`, `_compressedDataSize`, `_data`).
//! [`IDManifest::to_compressed`]/[`IDManifest::from_compressed`] bridge
//! between the structured model and this wire form (serialize/parse +
//! compress/decompress), matching the C++ implicit-cast constructors
//! `CompressedIDManifest(const IDManifest&)` / `IDManifest(const
//! CompressedIDManifest&)`.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifestAttribute.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifestAttribute.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp>

mod codec;
mod hash;
pub mod manifest;
mod varint;

pub use manifest::{
    ChannelGroupManifest, IDManifest, IdLifetime, CUSTOMHASH, ID2_SCHEME, ID_SCHEME,
    MURMURHASH3_32, MURMURHASH3_64, NOTHASHED, UNKNOWN,
};

use crate::attr::attribute::{AttrValue, TypedAttribute};
use crate::compressor::zlib::{
    compress_into, new_deflate_compressor, uncompress_into, LowDecompressor, ShortOutputPolicy, DEFAULT_ZIP_LEVEL,
};
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

/// `Xdr::size<uint64_t>()` — the 8-byte header that precedes the compressed
/// data in the `"idmanifest"` wire format.
const UNCOMPRESSED_SIZE_FIELD: usize = 8;

/// On-disk form of an ID manifest — port of `struct CompressedIDManifest`.
///
/// Holds the zlib-compressed manifest bytes (`data`) plus the size the manifest
/// would occupy once decompressed (`uncompressed_data_size`).  The compressed
/// byte length is simply `data.len()` (C++'s `_compressedDataSize`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct CompressedIDManifest {
    /// Size in bytes of the manifest after zlib decompression
    /// (`CompressedIDManifest::_uncompressedDataSize`).
    pub uncompressed_data_size: u64,
    /// The raw, already zlib-compressed manifest bytes
    /// (`CompressedIDManifest::_data`, with length `_compressedDataSize`).
    pub data: Vec<u8>,
}

impl CompressedIDManifest {
    /// Construct from the decompressed size and the raw compressed bytes.
    pub fn new(uncompressed_data_size: u64, data: Vec<u8>) -> Self {
        Self {
            uncompressed_data_size,
            data,
        }
    }

    /// Length of the compressed byte payload — mirrors `_compressedDataSize`.
    pub fn compressed_data_size(&self) -> usize {
        self.data.len()
    }

    /// Decompress this manifest's payload — port of the bounds-checked half
    /// of `IDManifest::IDManifest(const CompressedIDManifest&)`
    /// (`ImfIDManifest.cpp:586-638`).
    ///
    /// This reproduces the 3.4 hardening C++ added around zlib decompression —
    /// * `uncompressed_data_size == 0` is rejected outright (C++ throws
    ///   rather than decompressing into a zero-length buffer),
    /// * the declared uncompressed size is bounded by the COMPRESSED size
    ///   actually present, scaled by zlib's own worst-case expansion ratio
    ///   (`MAX_EXPANSION = 1032`), so a tiny file cannot force an enormous
    ///   allocation before any content check runs,
    /// * the actual decompressed length must equal the declared size exactly
    ///   (matching C++'s post-hoc `outSize != uncompressedDataSize` check),
    /// * routed through the same SSOT
    ///   [`crate::compressor::zlib::uncompress_into`] every other codec uses,
    ///   which also rejects trailing bytes after the zlib stream —
    ///   `deflate_rs::Decompressor::zlib_decompress_exact` (the old call
    ///   here) only checked the OUTPUT length, not this.
    ///
    /// Returns the raw decompressed bytes. Use
    /// [`IDManifest::from_compressed`] for the full pipeline through to a
    /// structured manifest ([`codec::parse`] on top of this).
    ///
    /// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp#L586-L638>
    pub fn decompress(&self) -> Result<Vec<u8>> {
        // zlib's documented worst-case expansion ratio.
        const MAX_EXPANSION: u64 = 1032;

        let compressed_len = self.data.len() as u64;

        if self.uncompressed_data_size == 0
            || compressed_len > u64::MAX / MAX_EXPANSION
            || self.uncompressed_data_size > compressed_len.saturating_mul(MAX_EXPANSION)
        {
            return Err(ExrError::Input(
                "IDManifest has an implausible uncompressed data size".into(),
            ));
        }

        let expected_size = usize::try_from(self.uncompressed_data_size).map_err(|_| {
            ExrError::Input(
                "IDManifest declares an uncompressed data size too large for this platform"
                    .into(),
            )
        })?;

        let mut out = Vec::new();
        // One-shot, once-per-manifest-attribute operation (not a per-chunk
        // hot path), so a throwaway `Decompressor` is fine here -- no chunk
        // loop for a persistent one to amortize over (see
        // `compressor::zlib`'s module doc).
        let mut decompressor = LowDecompressor::new();
        uncompress_into(&mut decompressor, &self.data, &mut out, expected_size, "IDManifest", ShortOutputPolicy::RequireExact)?;
        if out.len() != expected_size {
            return Err(ExrError::Input(format!(
                "IDManifest zlib stream produced {} bytes, expected {expected_size}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Compress a raw (already-serialized) manifest payload into a
    /// [`CompressedIDManifest`] — the write-side counterpart of
    /// [`decompress`][Self::decompress], mirroring
    /// `CompressedIDManifest::CompressedIDManifest(const IDManifest&)`
    /// (`ImfIDManifest.cpp:1137-1172`) from the point where C++ has already
    /// serialized the structured model into a flat byte buffer. Use
    /// [`IDManifest::to_compressed`] for the full pipeline from a structured
    /// manifest ([`codec::serialize`] feeding into this).
    pub fn compress(uncompressed: &[u8]) -> Result<Self> {
        // DEFAULT_ZIP_LEVEL is now the ONE constant for this
        // fallback (`OpenEXRCore/compression.c:112-114`'s
        // `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL`) -- this used to be its own local
        // `const EXR_DEFAULT_ZLIB_COMPRESS_LEVEL: i32 = 4;` duplicate.
        let mut compressed = Vec::new();
        // Same one-shot reasoning as `decompress`'s throwaway `Decompressor`.
        let mut comp = new_deflate_compressor(DEFAULT_ZIP_LEVEL)?;
        compress_into(&mut comp, uncompressed, &mut compressed)?;
        Ok(Self {
            uncompressed_data_size: uncompressed.len() as u64,
            data: compressed,
        })
    }
}

impl IDManifest {
    /// Full decode pipeline: [`CompressedIDManifest::decompress`] then
    /// [`codec::parse`] — mirrors the implicit-cast constructor
    /// `IDManifest::IDManifest(const CompressedIDManifest&)`
    /// (`ImfIDManifest.cpp:586-641`, which chains the same two steps:
    /// bounds-checked zlib decompress then `init`).
    pub fn from_compressed(compressed: &CompressedIDManifest) -> Result<Self> {
        let bytes = compressed.decompress()?;
        codec::parse(&bytes)
    }

    /// Full encode pipeline: [`codec::serialize`] then
    /// [`CompressedIDManifest::compress`] — mirrors the implicit-cast
    /// constructor `CompressedIDManifest::CompressedIDManifest(const
    /// IDManifest&)` (`ImfIDManifest.cpp:1137-1172`, which chains the same
    /// two steps: `serialize` then zlib-compress the result).
    pub fn to_compressed(&self) -> Result<CompressedIDManifest> {
        let bytes = codec::serialize(self)?;
        CompressedIDManifest::compress(&bytes)
    }
}

impl AttrValue for CompressedIDManifest {
    fn static_type_name() -> &'static str {
        // ImfIDManifestAttribute.cpp: return "idmanifest";
        "idmanifest"
    }

    /// Mirrors `IDManifestAttribute::writeValueTo`.
    ///
    /// Writes `uncompressedDataSize` (`u64`) followed by the raw compressed
    /// bytes (`Xdr::write<StreamIO>(os, _data, _compressedDataSize)`).
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_u64(o, self.uncompressed_data_size)?;
        xdr::write_chars(o, &self.data)
    }

    /// Mirrors `IDManifestAttribute::readValueFrom`.
    ///
    /// Rejects `size < 8` (`InputExc`), reads the 8-byte `uncompressedDataSize`,
    /// then reads the remaining `size - 8` bytes as the compressed payload.
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        if (size as i64) < UNCOMPRESSED_SIZE_FIELD as i64 {
            return Err(ExrError::Input(
                "Invalid size field reading idmanifest attribute".into(),
            ));
        }

        // first eight bytes: data size once data is uncompressed
        let uncompressed_data_size = xdr::read_u64(i)?;

        // Bounded read (shared helper): rejects a `size` larger than the stream
        // remainder before allocating, so a tiny malformed file cannot request a
        // huge idmanifest buffer.
        let compressed_len = (size as usize) - UNCOMPRESSED_SIZE_FIELD;
        let data = if compressed_len > 0 {
            xdr::read_bounded_bytes(i, compressed_len)?
        } else {
            Vec::new()
        };

        Ok(Self {
            uncompressed_data_size,
            data,
        })
    }
}

/// `typedef TypedAttribute<CompressedIDManifest> IDManifestAttribute;`
pub type IDManifestAttribute = TypedAttribute<CompressedIDManifest>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(m: &CompressedIDManifest) -> Vec<u8> {
        let mut o = MemOStream::new();
        m.write_value(&mut o, 0).expect("write");
        o.into_data()
    }

    #[test]
    fn type_name_is_idmanifest() {
        assert_eq!(CompressedIDManifest::static_type_name(), "idmanifest");
    }

    #[test]
    fn exact_bytes() {
        // uncompressedDataSize = 0x0102 (258), compressed payload = [AA, BB, CC]
        let m = CompressedIDManifest::new(258, vec![0xAA, 0xBB, 0xCC]);
        let bytes = write_bytes(&m);

        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // uncompressedDataSize = 258 (u64 LE)
            0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // raw compressed bytes
            0xAA, 0xBB, 0xCC,
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn round_trip_raw_framed() {
        let m = CompressedIDManifest::new(4096, vec![1, 2, 3, 4, 5, 6, 7]);
        let bytes = write_bytes(&m);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = CompressedIDManifest::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.compressed_data_size(), 7);
    }

    #[test]
    fn size_too_small_rejected() {
        let mut i = MemIStream::new(vec![0u8; 4]);
        // size < 8 must error.
        assert!(CompressedIDManifest::read_value(&mut i, 4, 0).is_err());
    }

    #[test]
    fn empty_compressed_payload() {
        // size == 8 means an empty compressed payload (valid).
        let m = CompressedIDManifest::new(0, vec![]);
        let bytes = write_bytes(&m);
        assert_eq!(bytes.len(), 8);
        let mut i = MemIStream::new(bytes);
        let back = CompressedIDManifest::read_value(&mut i, 8, 0).unwrap();
        assert_eq!(back, m);
    }

    /// `compress`/`decompress` round-trip an arbitrary payload.
    #[test]
    fn compress_decompress_round_trip() {
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(20);
        let compressed = CompressedIDManifest::compress(&payload).unwrap();
        assert!(compressed.data.len() < payload.len(), "repetitive data should compress");
        let back = compressed.decompress().unwrap();
        assert_eq!(back, payload);
    }

    /// The wire attribute round-trips a real compressed manifest end
    /// to end: `compress` -> `write_value` -> `read_value` -> `decompress`.
    #[test]
    fn wire_round_trip_through_decompress() {
        let payload = b"id manifest payload bytes".to_vec();
        let m = CompressedIDManifest::compress(&payload).unwrap();
        let bytes = write_bytes(&m);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = CompressedIDManifest::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back.decompress().unwrap(), payload);
    }

    /// `uncompressed_data_size == 0` is rejected outright (matches
    /// C++ 3.4's hardened `IDManifest::IDManifest(const
    /// CompressedIDManifest&)`, which would otherwise decompress into a
    /// zero-length buffer).
    #[test]
    fn decompress_rejects_zero_uncompressed_size() {
        let m = CompressedIDManifest::new(0, vec![1, 2, 3]);
        assert!(m.decompress().is_err());
    }

    /// A declared uncompressed size wildly out of proportion to the
    /// compressed size (beyond zlib's ~1032:1 worst-case ratio) is rejected
    /// before any allocation, rather than trusting an attacker-controlled
    /// 64-bit size taken directly from the file.
    #[test]
    fn decompress_rejects_implausible_expansion_ratio() {
        // 3 bytes of "compressed" data claiming to expand to 4 GiB.
        let m = CompressedIDManifest::new(4 * 1024 * 1024 * 1024, vec![1, 2, 3]);
        assert!(m.decompress().is_err());
    }

    /// A declared size that does not match the stream's actual
    /// decompressed length is rejected (`zlib_decompress_exact`'s
    /// short-output check), matching C++'s post-hoc `outSize != declared` test.
    #[test]
    fn decompress_rejects_size_mismatch() {
        let payload = b"short payload".to_vec();
        let mut m = CompressedIDManifest::compress(&payload).unwrap();
        // Lie about the uncompressed size (still small enough to pass the
        // expansion-ratio gate, so the exact-size check is what must fire).
        m.uncompressed_data_size = payload.len() as u64 + 1;
        assert!(m.decompress().is_err());
    }

    /// The full structured pipeline: build an `IDManifest`,
    /// `to_compressed()` it, write/read it through the wire attribute, then
    /// `from_compressed()` it back, and it must equal the original.
    #[test]
    fn structured_manifest_round_trips_through_the_wire_attribute() {
        let mut manifest = IDManifest::new();
        {
            let g = manifest.add_channel("crypto_object");
            g.set_component("name").unwrap();
            g.set_hash_scheme(MURMURHASH3_32);
            g.set_encoding_scheme(ID_SCHEME);
            g.set_lifetime(IdLifetime::Stable);
            g.insert_hashed_single("/world/hero").unwrap();
            g.insert_hashed_single("/world/prop_sword").unwrap();
        }

        let compressed = manifest.to_compressed().unwrap();
        let bytes = write_bytes(&compressed);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let read_back = CompressedIDManifest::read_value(&mut i, size, 0).unwrap();

        let restored = IDManifest::from_compressed(&read_back).unwrap();
        assert_eq!(restored, manifest);
    }
}
