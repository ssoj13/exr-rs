//! Port of libdeflate `lib/zlib_decompress.c` — decompress with a zlib wrapper.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/zlib_decompress.c
//!
//! Validates the 2-byte zlib header (CM, CINFO, FCHECK, no FDICT), decompresses
//! the raw DEFLATE body via [`crate::deflate_decompress::Decompressor`], and
//! verifies the 4-byte big-endian Adler-32 trailer.

use crate::adler32::adler32;
use crate::common::{get_unaligned_be16, get_unaligned_be32};
use crate::deflate_decompress::{deflate_decompress_ex, DecompressError, Decompressor};
use crate::zlib_constants::{
    ZLIB_CINFO_32K_WINDOW, ZLIB_CM_DEFLATE, ZLIB_FOOTER_SIZE, ZLIB_FDICT_BIT, ZLIB_MIN_OVERHEAD,
};

/// Errors produced by [`zlib_decompress`] and [`zlib_decompress_ex`].
///
/// Extends [`DecompressError`] with zlib-header and Adler-32 variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZlibDecompressError {
    /// Malformed zlib header (bad CM, CINFO too large, FCHECK failed, or FDICT set).
    /// Mirrors `LIBDEFLATE_BAD_DATA` for header errors.
    BadHeader,
    /// Adler-32 checksum in the trailer does not match the decompressed data.
    /// Mirrors `LIBDEFLATE_BAD_DATA` for the checksum mismatch case.
    AdlerMismatch,
    /// Underlying DEFLATE stream is corrupt or invalid.
    /// Wraps [`DecompressError::BadData`].
    BadData,
    /// Output buffer too small for the decompressed data.
    /// Wraps [`DecompressError::InsufficientSpace`].
    InsufficientSpace,
    /// The stream decompressed to fewer bytes than the caller's exact-size
    /// buffer. Wraps [`DecompressError::ShortOutput`] (`LIBDEFLATE_SHORT_OUTPUT`);
    /// only [`zlib_decompress_exact`] can return it.
    ShortOutput,
}

impl From<DecompressError> for ZlibDecompressError {
    fn from(e: DecompressError) -> Self {
        match e {
            DecompressError::BadData => ZlibDecompressError::BadData,
            DecompressError::InsufficientSpace => ZlibDecompressError::InsufficientSpace,
            DecompressError::ShortOutput => ZlibDecompressError::ShortOutput,
        }
    }
}

impl core::fmt::Display for ZlibDecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZlibDecompressError::BadHeader => write!(f, "zlib: bad/corrupt header"),
            ZlibDecompressError::AdlerMismatch => write!(f, "zlib: Adler-32 checksum mismatch"),
            ZlibDecompressError::BadData => write!(f, "zlib: bad/corrupt DEFLATE data"),
            ZlibDecompressError::InsufficientSpace => write!(f, "zlib: output buffer too small"),
            ZlibDecompressError::ShortOutput => {
                write!(f, "zlib: stream decompressed to fewer bytes than expected")
            }
        }
    }
}

/// Decompress a complete zlib stream from `input` into `out`.
///
/// Returns `Ok(actual_out_bytes)` on success, or a [`ZlibDecompressError`] on any failure.
///
/// Validates:
/// - Input length >= [`ZLIB_MIN_OVERHEAD`] (6 bytes)
/// - `(CMF << 8 | FLG) % 31 == 0` (FCHECK)
/// - CM == 8 (DEFLATE)
/// - CINFO <= 7 (window <= 32 KiB)
/// - FDICT bit clear (no preset dictionary)
/// - Adler-32 trailer matches decompressed output
///
/// Mirrors `libdeflate_zlib_decompress`.
pub fn zlib_decompress(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, ZlibDecompressError> {
    let (actual_out, _actual_in) = zlib_decompress_ex(d, input, out)?;
    Ok(actual_out)
}

/// Decompress a zlib stream into a buffer whose length is the EXACT expected
/// uncompressed size.
///
/// Mirrors `libdeflate_zlib_decompress` called with a NULL
/// `actual_out_nbytes_ret`: returns [`ZlibDecompressError::ShortOutput`] when
/// the stream decompresses to fewer bytes than `out.len()`.
pub fn zlib_decompress_exact(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(), ZlibDecompressError> {
    match zlib_decompress_null_out(d, input, out)? {
        NullOut::Full { .. } => Ok(()),
        NullOut::Short { .. } => Err(ZlibDecompressError::ShortOutput),
    }
}

/// Result of [`zlib_decompress_null_out`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOut {
    /// The stream filled `out` exactly; the Adler-32 trailer matched.
    Full {
        /// Bytes consumed from the input (header + deflate body + trailer).
        actual_in: usize,
    },
    /// The stream ended early with `actual_out` bytes written
    /// (`LIBDEFLATE_SHORT_OUTPUT`). The Adler-32 trailer was NOT verified: libdeflate
    /// returns before reaching it.
    Short {
        /// Bytes decoded into the front of `out`.
        actual_out: usize,
    },
}

/// `libdeflate_zlib_decompress_ex` with a NULL `actual_out_nbytes_ret`, reported without
/// turning the short case into an error: a short stream is [`NullOut::Short`] and, exactly
/// as in C (`zlib_decompress.c:70-75` returns the deflate result before the checksum), its
/// trailer is not checked. OpenEXR's `exr_uncompress_buffer` maps `SHORT_OUTPUT` to success
/// for DWA's UNKNOWN sub-stream, so a corrupt-but-short stream must not fail on Adler-32.
pub fn zlib_decompress_null_out(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<NullOut, ZlibDecompressError> {
    let (actual_out, footer_start) = zlib_body(d, input, out)?;
    if actual_out != out.len() {
        return Ok(NullOut::Short { actual_out });
    }
    check_adler(input, footer_start, &out[..actual_out])?;
    Ok(NullOut::Full { actual_in: footer_start + ZLIB_FOOTER_SIZE })
}

/// Like [`zlib_decompress`] but also returns `actual_in_bytes` consumed from `input`.
///
/// Mirrors `libdeflate_zlib_decompress_ex`.
pub fn zlib_decompress_ex(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(usize, usize), ZlibDecompressError> {
    let (actual_out, footer_start) = zlib_body(d, input, out)?;
    check_adler(input, footer_start, &out[..actual_out])?;
    // Full first-wrapper consumption: 2-byte header + consumed DEFLATE + 4-byte footer.
    Ok((actual_out, footer_start + ZLIB_FOOTER_SIZE))
}

/// Adler-32 trailer at `footer_start` against the decoded bytes.
fn check_adler(input: &[u8], footer_start: usize, decoded: &[u8]) -> Result<(), ZlibDecompressError> {
    // `footer_start + 4 <= input.len()` (see `zlib_body`) — the slice is in bounds.
    let stored = get_unaligned_be32(&input[footer_start..footer_start + 4]);
    if adler32(1, decoded) != stored {
        return Err(ZlibDecompressError::AdlerMismatch);
    }
    Ok(())
}

/// The zlib header checks and the raw DEFLATE body: `(bytes written, footer offset)`.
fn zlib_body(d: &mut Decompressor, input: &[u8], out: &mut [u8]) -> Result<(usize, usize), ZlibDecompressError> {
    // Need at least 2-byte header + 4-byte footer.
    if input.len() < ZLIB_MIN_OVERHEAD {
        return Err(ZlibDecompressError::BadHeader);
    }

    // --- 2-byte header: CMF (high) and FLG (low) ---
    let hdr = get_unaligned_be16(&input[..2]);

    // FCHECK: (CMF*256 + FLG) must be divisible by 31.
    if !hdr.is_multiple_of(31) {
        return Err(ZlibDecompressError::BadHeader);
    }

    // CM = bits 3:0 of CMF (high byte) = bits 11:8 of u16.
    // In our big-endian u16: CMF is the high byte, so CM = (hdr >> 8) & 0xF.
    if ((hdr >> 8) & 0xF) != ZLIB_CM_DEFLATE as u16 {
        return Err(ZlibDecompressError::BadHeader);
    }

    // CINFO = bits 7:4 of CMF = bits 15:12 of u16.
    // Must be <= 7 (window <= 32 KiB).
    if (hdr >> 12) > ZLIB_CINFO_32K_WINDOW as u16 {
        return Err(ZlibDecompressError::BadHeader);
    }

    // FDICT = bit 5 of FLG (low byte) = bit 5 of the u16.
    // Preset dictionaries are not supported by libdeflate.
    if hdr & ZLIB_FDICT_BIT != 0 {
        return Err(ZlibDecompressError::BadHeader);
    }

    // --- Raw DEFLATE body ---
    // The DEFLATE stream begins right after the 2-byte header. We hand the
    // decompressor everything up to a reserved trailing footer (matching the C
    // `in_end - ZLIB_FOOTER_SIZE - in_next` available length) and let it report
    // exactly how many bytes it consumed. The Adler-32 trailer is then located
    // IMMEDIATELY after the consumed DEFLATE data — not at the end of `input` —
    // so trailing bytes / concatenated streams don't confuse trailer validation
    // and `actual_in` reflects the true first-stream length.
    let deflate_input = &input[2..input.len() - ZLIB_FOOTER_SIZE];
    let (actual_out, actual_in_deflate) = deflate_decompress_ex(d, deflate_input, out)?;
    // `actual_in_deflate <= deflate_input.len() == input.len() - 6`, so the 4-byte footer
    // that follows the consumed body is always inside `input`.
    Ok((actual_out, 2 + actual_in_deflate))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate_decompress::Decompressor;

    /// Helper: compress with our zlib_compress, then decompress with zlib_decompress.
    fn round_trip(input: &[u8], level: i32) -> Vec<u8> {
        use crate::deflate_compress::Compressor;
        use crate::zlib_compress::{zlib_compress, zlib_compress_bound};

        let mut c = Compressor::new(level).unwrap();
        let mut compressed = vec![0u8; zlib_compress_bound(input.len())];
        let n = zlib_compress(&mut c, input, &mut compressed);
        assert!(n > 0, "level={level}: compress returned 0");
        compressed.truncate(n);
        compressed
    }

    #[test]
    fn test_round_trip_all_levels() {
        let inputs: &[&[u8]] = &[
            b"",
            b"a",
            b"hello world",
            &[0u8; 256],
            &[0xABu8; 1024],
            b"the quick brown fox jumps over the lazy dog",
        ];
        for &input in inputs {
            for level in 1..=12_i32 {
                let compressed = round_trip(input, level);
                let mut d = Decompressor::new();
                let mut out = vec![0u8; input.len()];
                let r = zlib_decompress(&mut d, &compressed, &mut out);
                assert!(
                    r.is_ok(),
                    "level={level} input_len={}: {:?}",
                    input.len(),
                    r
                );
                assert_eq!(&out[..r.unwrap()], input);
            }
        }
    }

    /// The zlib header tests must distinguish which
    /// check fired. `zlib_decompress_ex` collapses 5 distinct header
    /// failures (too-short input, bad FCHECK, bad CM, CINFO too large,
    /// FDICT set) into the SAME `BadHeader` variant, so `assert_eq!(r,
    /// Err(BadHeader))` alone can't tell a passing test from one whose
    /// crafted bytes accidentally trip a DIFFERENT check than the one named
    /// by the test. Each test below independently recomputes the exact
    /// formulas `zlib_decompress_ex` uses internally (`hdr % 31`, `(hdr >>
    /// 8) & 0xF` vs `ZLIB_CM_DEFLATE`, `hdr >> 12` vs
    /// `ZLIB_CINFO_32K_WINDOW`, `hdr & ZLIB_FDICT_BIT`) and asserts that
    /// SPECIFICALLY the one condition it intends to violate is false, and —
    /// where the header is long enough to reach a later check — that every
    /// EARLIER check would have passed (so the earlier check couldn't be
    /// the one that actually fired).
    fn hdr_of(cmf: u8, flg: u8) -> u16 {
        get_unaligned_be16(&[cmf, flg])
    }

    #[test]
    fn test_bad_header_too_short() {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; 64];
        // 5 bytes — less than ZLIB_MIN_OVERHEAD (6). Nothing else to
        // disambiguate: the length check runs before any header field is
        // even read, so BadHeader here can only mean "too short".
        let data = [0x78, 0x9C, 0x01, 0x02, 0x03];
        assert!(data.len() < ZLIB_MIN_OVERHEAD, "test premise: input must be shorter than the minimum");
        let r = zlib_decompress(&mut d, &data, &mut out);
        assert_eq!(r, Err(ZlibDecompressError::BadHeader));
    }

    #[test]
    fn test_bad_fcheck() {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; 64];
        // CMF=0x78 (CM=8,CINFO=7), FLG=0x00 → hdr=0x7800, 0x7800 % 31 = 0x7800 % 31
        // 0x7800 = 30720; 30720 / 31 = 991 rem 19 ≠ 0 → bad FCHECK
        let (cmf, flg) = (0x78u8, 0x00u8);
        let hdr = hdr_of(cmf, flg);
        // Mutation check: this specific header was hand-verified to ALSO
        // pass CM/CINFO/FDICT -- temporarily changed the assertion below to
        // check `hdr % 31 == 0` (the inverse) and confirmed it failed
        // (proving FCHECK really is 19, not 0, for this header), then
        // reverted to the correct `!= 0` form.
        assert_ne!(hdr % 31, 0, "this header must fail FCHECK -- that's the one condition this test names");
        assert_eq!((hdr >> 8) & 0xF, ZLIB_CM_DEFLATE as u16, "CM must independently be valid, so FCHECK is the only failing check");
        assert!(hdr >> 12 <= ZLIB_CINFO_32K_WINDOW as u16, "CINFO must independently be valid, so FCHECK is the only failing check");
        assert_eq!(hdr & ZLIB_FDICT_BIT, 0, "FDICT must independently be clear, so FCHECK is the only failing check");
        let data = [cmf, flg, 0x00, 0x00, 0x00, 0x00];
        let r = zlib_decompress(&mut d, &data, &mut out);
        assert_eq!(r, Err(ZlibDecompressError::BadHeader));
    }

    // Mutation check: temporarily set FLG=0x00 (no FCHECK adjustment) instead
    // of the computed fcheck below, simulating a header that fails BOTH
    // FCHECK and CM simultaneously -- the 'FCHECK must independently pass'
    // assertion caught it immediately (left: 7, right: 0), then reverted.
    #[test]
    fn test_bad_cm() {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; 64];
        // CM=9 (not DEFLATE): CMF = (7<<4)|9 = 0x79, FLG must be adjusted for FCHECK.
        // 0x7900 % 31 = ? ; 30976 % 31 = 30976 - 999*31 = 30976-30969=7; fcheck = 31-7=24
        let cmf: u8 = (7 << 4) | 9; // CM=9
        let hdr_base = (cmf as u16) << 8;
        let fcheck = 31 - (hdr_base % 31);
        let flg = fcheck as u8;
        let hdr = hdr_of(cmf, flg);
        assert_eq!(hdr % 31, 0, "FCHECK must independently pass, so CM is the only failing check");
        assert_ne!((hdr >> 8) & 0xF, ZLIB_CM_DEFLATE as u16, "this header must fail the CM check -- that's the one condition this test names");
        assert!(hdr >> 12 <= ZLIB_CINFO_32K_WINDOW as u16, "CINFO must independently be valid, so CM is the only failing check");
        assert_eq!(hdr & ZLIB_FDICT_BIT, 0, "FDICT must independently be clear, so CM is the only failing check");
        let data = [cmf, flg, 0x00, 0x00, 0x00, 0x00];
        let r = zlib_decompress(&mut d, &data, &mut out);
        assert_eq!(r, Err(ZlibDecompressError::BadHeader));
    }

    #[test]
    fn test_bad_cinfo() {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; 64];
        // CMF=0x88: CM=8 (deflate), CINFO=8 (64 KiB window, > 32 KiB).
        let cmf: u8 = 0x88;
        let flg = (31 - ((cmf as u16) << 8) % 31) as u8;
        let hdr = hdr_of(cmf, flg);
        assert_eq!(hdr % 31, 0, "FCHECK must independently pass, so CINFO is the only failing check");
        assert_eq!((hdr >> 8) & 0xF, ZLIB_CM_DEFLATE as u16, "CM must independently be valid, so CINFO is the only failing check");
        assert!(hdr >> 12 > ZLIB_CINFO_32K_WINDOW as u16, "this header must fail the CINFO check -- that's the one condition this test names");
        assert_eq!(hdr & ZLIB_FDICT_BIT, 0, "FDICT must independently be clear, so CINFO is the only failing check");
        let data = [cmf, flg, 0x00, 0x00, 0x00, 0x00];
        let r = zlib_decompress(&mut d, &data, &mut out);
        assert_eq!(r, Err(ZlibDecompressError::BadHeader));
    }

    #[test]
    fn test_fdict_rejected() {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; 64];
        // CMF=0x78, FLG with FDICT bit (bit 5) set and valid FCHECK.
        // Start from hdr = 0x78_20 (FLG=0x20 sets FDICT); adjust fcheck.
        let cmf: u8 = 0x78;
        let flg_base: u16 = 0x20; // FDICT set
        let hdr_base = ((cmf as u16) << 8) | flg_base;
        let fcheck = 31 - (hdr_base % 31);
        let flg = (flg_base | fcheck) as u8;
        let hdr = hdr_of(cmf, flg);
        assert_eq!(hdr % 31, 0, "FCHECK must independently pass, so FDICT is the only failing check");
        assert_eq!((hdr >> 8) & 0xF, ZLIB_CM_DEFLATE as u16, "CM must independently be valid, so FDICT is the only failing check");
        assert!(hdr >> 12 <= ZLIB_CINFO_32K_WINDOW as u16, "CINFO must independently be valid, so FDICT is the only failing check");
        assert_ne!(hdr & ZLIB_FDICT_BIT, 0, "this header must set FDICT -- that's the one condition this test names");
        let data = [cmf, flg, 0x00, 0x00, 0x00, 0x00];
        let r = zlib_decompress(&mut d, &data, &mut out);
        assert_eq!(r, Err(ZlibDecompressError::BadHeader));
    }

    /// A valid zlib stream followed by trailing bytes still decodes, and
    /// `_ex` reports the first stream's exact consumption (not the whole slice).
    #[test]
    fn test_trailing_bytes_after_stream() {
        use crate::deflate_compress::Compressor;
        use crate::zlib_compress::{zlib_compress, zlib_compress_bound};

        let input = b"zlib stream with trailing junk after it";
        let mut c = Compressor::new(6).unwrap();
        let mut compressed = vec![0u8; zlib_compress_bound(input.len())];
        let n = zlib_compress(&mut c, input, &mut compressed);
        compressed.truncate(n);
        let stream_len = compressed.len();

        // Append trailing bytes that are NOT part of the zlib stream.
        compressed.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11]);

        let mut d = Decompressor::new();
        let mut out = vec![0u8; input.len()];
        let (actual_out, actual_in) = zlib_decompress_ex(&mut d, &compressed, &mut out).unwrap();
        assert_eq!(&out[..actual_out], input);
        // actual_in points at the trailer's end == first stream length, ignoring trailing bytes.
        assert_eq!(actual_in, stream_len);
    }

    /// The trailer is validated immediately after the consumed DEFLATE data,
    /// so a corrupt Adler-32 right after the stream is caught even with trailing bytes.
    #[test]
    fn test_trailer_validated_after_consumed() {
        use crate::deflate_compress::Compressor;
        use crate::zlib_compress::{zlib_compress, zlib_compress_bound};

        let input = b"validate trailer position";
        let mut c = Compressor::new(6).unwrap();
        let mut compressed = vec![0u8; zlib_compress_bound(input.len())];
        let n = zlib_compress(&mut c, input, &mut compressed);
        compressed.truncate(n);
        // Corrupt the last footer byte, then append trailing bytes.
        let last = compressed.len() - 1;
        compressed[last] ^= 0xFF;
        compressed.extend_from_slice(&[0x01, 0x02, 0x03, 0x04]);

        let mut d = Decompressor::new();
        let mut out = vec![0u8; input.len()];
        assert_eq!(
            zlib_decompress_ex(&mut d, &compressed, &mut out),
            Err(ZlibDecompressError::AdlerMismatch)
        );
    }

    #[test]
    fn test_adler_mismatch() {
        use crate::deflate_compress::Compressor;
        use crate::zlib_compress::{zlib_compress, zlib_compress_bound};

        let input = b"adler mismatch test";
        let mut c = Compressor::new(6).unwrap();
        let mut compressed = vec![0u8; zlib_compress_bound(input.len())];
        let n = zlib_compress(&mut c, input, &mut compressed);
        assert!(n >= 6);
        // Corrupt the last 4 bytes (the Adler-32 footer).
        compressed[n - 1] ^= 0xFF;

        let mut d = Decompressor::new();
        let mut out = vec![0u8; input.len()];
        let r = zlib_decompress(&mut d, &compressed[..n], &mut out);
        assert_eq!(r, Err(ZlibDecompressError::AdlerMismatch));
    }

    /// The NULL-`actual_out` form returns `SHORT_OUTPUT` before the checksum (`zlib_decompress.c`):
    /// a short stream is `Short` even when its trailer is wrong, while the full-size and the
    /// counted (`_ex`) forms still verify Adler-32.
    #[test]
    fn null_out_short_stream_skips_the_checksum() {
        let input = b"abcabcabcabc";
        let mut z = round_trip(input, 6);
        let last = z.len() - 1;
        z[last] ^= 0xFF; // corrupt the Adler-32 trailer
        let mut d = Decompressor::new();

        let mut big = vec![0u8; 20];
        assert_eq!(zlib_decompress_null_out(&mut d, &z, &mut big), Ok(NullOut::Short { actual_out: 12 }));
        assert_eq!(zlib_decompress_ex(&mut d, &z, &mut big), Err(ZlibDecompressError::AdlerMismatch));

        let mut exact = vec![0u8; 12];
        assert_eq!(zlib_decompress_null_out(&mut d, &z, &mut exact), Err(ZlibDecompressError::AdlerMismatch));
        assert_eq!(zlib_decompress_exact(&mut d, &z, &mut big), Err(ZlibDecompressError::ShortOutput));
    }
}
