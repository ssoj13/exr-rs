//! Port of libdeflate `lib/gzip_decompress.c` — decompress with a gzip wrapper.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/gzip_decompress.c
//!
//! Validates the gzip header (magic bytes, CM, reserved FLG bits), skips any
//! optional fields signalled by FLG (FEXTRA / FNAME / FCOMMENT / FHCRC),
//! decompresses the raw DEFLATE body, then verifies the 8-byte footer
//! (CRC-32 + ISIZE). Both the `_ex` variant (returns actual input consumed)
//! and the simple variant are ported.

use crate::common::{get_unaligned_le16, get_unaligned_le32};
use crate::crc32::crc32;
use crate::deflate_decompress::{deflate_decompress_ex, DecompressError, Decompressor};
use crate::gzip_constants::{
    GZIP_CM_DEFLATE, GZIP_FCOMMENT, GZIP_FEXTRA, GZIP_FHCRC, GZIP_FNAME,
    GZIP_FOOTER_SIZE, GZIP_FRESERVED, GZIP_ID1, GZIP_ID2, GZIP_MIN_OVERHEAD,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the gzip decompressor.
///
/// Mirrors the `enum libdeflate_result` error values relevant to gzip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GzipDecompressError {
    /// The gzip stream is malformed (bad magic, CM, reserved FLG bits, CRC32 or ISIZE mismatch).
    /// Mirrors `LIBDEFLATE_BAD_DATA`.
    BadData,
    /// The output buffer is too small for the decompressed data.
    /// Mirrors `LIBDEFLATE_INSUFFICIENT_SPACE`.
    InsufficientSpace,
    /// The stream decompressed to fewer bytes than the caller's exact-size
    /// buffer. Mirrors `LIBDEFLATE_SHORT_OUTPUT`; only
    /// [`gzip_decompress_exact`] can return it.
    ShortOutput,
}

impl From<DecompressError> for GzipDecompressError {
    fn from(e: DecompressError) -> Self {
        match e {
            DecompressError::BadData => GzipDecompressError::BadData,
            DecompressError::InsufficientSpace => GzipDecompressError::InsufficientSpace,
            DecompressError::ShortOutput => GzipDecompressError::ShortOutput,
        }
    }
}

impl core::fmt::Display for GzipDecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GzipDecompressError::BadData => write!(f, "gzip: bad/corrupt gzip data"),
            GzipDecompressError::InsufficientSpace => write!(f, "gzip: output buffer too small"),
            GzipDecompressError::ShortOutput => {
                write!(f, "gzip: stream decompressed to fewer bytes than expected")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal decompressor state (shared between calls)
// ---------------------------------------------------------------------------

/// Returns the thread-local DEFLATE decompressor.
/// We create one per call to keep the public API allocation-free (mirrors
/// libdeflate where the caller manages the Decompressor object lifetime).
fn new_decompressor() -> Decompressor {
    Decompressor::new()
}

// ---------------------------------------------------------------------------
// gzip_decompress_ex  (mirrors libdeflate_gzip_decompress_ex)
// ---------------------------------------------------------------------------

/// Decompress a complete gzip member from `input` into `out`.
///
/// Returns `Ok((actual_out, actual_in))` on success:
/// - `actual_out` bytes of decompressed data were written to `out`.
/// - `actual_in` bytes of `input` were consumed (= first member only).
///
/// Mirrors `libdeflate_gzip_decompress_ex`.
pub fn gzip_decompress_ex(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(usize, usize), GzipDecompressError> {
    let in_len = input.len();

    if in_len < GZIP_MIN_OVERHEAD {
        return Err(GzipDecompressError::BadData);
    }

    let mut pos = 0usize; // current read position in `input`

    // --- Fixed 10-byte header ---
    if input[pos] != GZIP_ID1 { return Err(GzipDecompressError::BadData); } pos += 1;
    if input[pos] != GZIP_ID2 { return Err(GzipDecompressError::BadData); } pos += 1;
    if input[pos] != GZIP_CM_DEFLATE { return Err(GzipDecompressError::BadData); } pos += 1;
    let flg = input[pos]; pos += 1;
    pos += 4; // MTIME (4 bytes, ignored)
    pos += 1; // XFL (ignored)
    pos += 1; // OS  (ignored)
    // pos == 10 here

    // Reserved FLG bits must be zero (mirrors C `if (flg & GZIP_FRESERVED) return BAD_DATA`).
    if flg & GZIP_FRESERVED != 0 {
        return Err(GzipDecompressError::BadData);
    }

    // --- Optional FEXTRA field ---
    if flg & GZIP_FEXTRA != 0 {
        if pos + 2 > in_len { return Err(GzipDecompressError::BadData); }
        let xlen = get_unaligned_le16(&input[pos..]) as usize;
        pos += 2;
        // Need at least xlen bytes + the 8-byte footer.
        if in_len.saturating_sub(pos) < xlen + GZIP_FOOTER_SIZE {
            return Err(GzipDecompressError::BadData);
        }
        pos += xlen;
    }

    // --- Optional FNAME field (NUL-terminated) ---
    if flg & GZIP_FNAME != 0 {
        loop {
            if pos >= in_len { return Err(GzipDecompressError::BadData); }
            let b = input[pos]; pos += 1;
            if b == 0 { break; }
        }
        if in_len.saturating_sub(pos) < GZIP_FOOTER_SIZE {
            return Err(GzipDecompressError::BadData);
        }
    }

    // --- Optional FCOMMENT field (NUL-terminated) ---
    if flg & GZIP_FCOMMENT != 0 {
        loop {
            if pos >= in_len { return Err(GzipDecompressError::BadData); }
            let b = input[pos]; pos += 1;
            if b == 0 { break; }
        }
        if in_len.saturating_sub(pos) < GZIP_FOOTER_SIZE {
            return Err(GzipDecompressError::BadData);
        }
    }

    // --- Optional FHCRC header CRC16 (2 bytes, not verified — same as libdeflate) ---
    if flg & GZIP_FHCRC != 0 {
        pos += 2;
        if in_len.saturating_sub(pos) < GZIP_FOOTER_SIZE {
            return Err(GzipDecompressError::BadData);
        }
    }

    // --- Raw DEFLATE body: everything between pos and the 8-byte footer ---
    let deflate_start = pos;
    if in_len.saturating_sub(deflate_start) < GZIP_FOOTER_SIZE {
        return Err(GzipDecompressError::BadData);
    }
    let deflate_end = in_len - GZIP_FOOTER_SIZE;
    if deflate_end < deflate_start {
        return Err(GzipDecompressError::BadData);
    }
    let deflate_input = &input[deflate_start..deflate_end];

    let (actual_out, actual_in_deflate) = deflate_decompress_ex(d, deflate_input, out)?;

    // Advance pos to IMMEDIATELY after the bytes the DEFLATE decoder actually
    // consumed — this is where THIS member's 8-byte footer begins (not the end
    // of `input`), so concatenated members can be iterated via the returned
    // `actual_in`. Mirrors the C `in_next += actual_in_nbytes`.
    // `actual_in_deflate <= deflate_input.len() == in_len - 8 - deflate_start`,
    // so `pos + 8 <= in_len` — the footer reads below stay in bounds.
    pos = deflate_start + actual_in_deflate;

    // --- 8-byte footer ---
    let stored_crc  = get_unaligned_le32(&input[pos..pos + 4]); pos += 4;
    let stored_isize = get_unaligned_le32(&input[pos..pos + 4]); pos += 4;

    if crc32(0, &out[..actual_out]) != stored_crc {
        return Err(GzipDecompressError::BadData);
    }
    if actual_out as u32 != stored_isize {
        return Err(GzipDecompressError::BadData);
    }

    Ok((actual_out, pos))
}

/// Decompress the first gzip member from `input` into `out`.
///
/// Returns `Ok(actual_out_bytes)` on success.  If multiple gzip members are
/// concatenated, only the first is decompressed (use `gzip_decompress_ex` to
/// learn how many input bytes were consumed).
///
/// Mirrors `libdeflate_gzip_decompress`.
pub fn gzip_decompress(
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, GzipDecompressError> {
    let mut d = new_decompressor();
    let (actual_out, _) = gzip_decompress_ex(&mut d, input, out)?;
    Ok(actual_out)
}

/// Decompress a gzip stream into a buffer whose length is the EXACT expected
/// uncompressed size.
///
/// Mirrors `libdeflate_gzip_decompress` called with a NULL
/// `actual_out_nbytes_ret`: returns [`GzipDecompressError::ShortOutput`] when
/// the stream decompresses to fewer bytes than `out.len()`.
pub fn gzip_decompress_exact(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(), GzipDecompressError> {
    let (actual_out, _actual_in) = gzip_decompress_ex(d, input, out)?;
    if actual_out != out.len() {
        return Err(GzipDecompressError::ShortOutput);
    }
    Ok(())
}

/// Like [`gzip_decompress`] but uses a caller-owned [`Decompressor`].
/// Matches the explicit-decompressor form used by the `Decompressor` facade.
pub fn gzip_decompress_with(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, GzipDecompressError> {
    let (actual_out, _) = gzip_decompress_ex(d, input, out)?;
    Ok(actual_out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate_compress::Compressor;
    use crate::gzip_compress::{gzip_compress, gzip_compress_bound};

    fn compress_at(level: i32, input: &[u8]) -> Vec<u8> {
        let mut c = Compressor::new(level).unwrap();
        let mut out = vec![0u8; gzip_compress_bound(input.len())];
        let n = gzip_compress(&mut c, input, &mut out);
        assert!(n > 0, "level={level} compress returned 0");
        out.truncate(n);
        out
    }

    #[test]
    fn test_round_trip_all_levels() {
        let inputs: &[&[u8]] = &[
            b"",
            b"a",
            b"hello, gzip!",
            &[0u8; 256],
            &[0xABu8; 1024],
            b"the quick brown fox jumps over the lazy dog",
        ];
        for &input in inputs {
            for level in 1..=12_i32 {
                let compressed = compress_at(level, input);
                let mut out = vec![0u8; input.len()];
                let r = gzip_decompress(&compressed, &mut out);
                assert!(r.is_ok(), "level={level} input_len={}: {:?}", input.len(), r);
                assert_eq!(&out[..r.unwrap()], input);
            }
        }
    }

    #[test]
    fn test_bad_magic() {
        let mut data = compress_at(6, b"test");
        data[0] ^= 0xFF; // corrupt ID1
        let mut out = vec![0u8; 16];
        assert_eq!(gzip_decompress(&data, &mut out), Err(GzipDecompressError::BadData));
    }

    #[test]
    fn test_bad_cm() {
        let mut data = compress_at(6, b"test");
        data[2] = 9; // wrong CM
        let mut out = vec![0u8; 16];
        assert_eq!(gzip_decompress(&data, &mut out), Err(GzipDecompressError::BadData));
    }

    #[test]
    fn test_reserved_flg_bits_rejected() {
        let mut data = compress_at(6, b"test");
        data[3] |= 0x80; // set a reserved FLG bit
        let mut out = vec![0u8; 16];
        assert_eq!(gzip_decompress(&data, &mut out), Err(GzipDecompressError::BadData));
    }

    #[test]
    fn test_corrupted_crc32() {
        let mut data = compress_at(6, b"hello world");
        let n = data.len();
        data[n - 8] ^= 0xFF; // corrupt CRC32
        let mut out = vec![0u8; 64];
        assert_eq!(gzip_decompress(&data, &mut out), Err(GzipDecompressError::BadData));
    }

    #[test]
    fn test_corrupted_isize() {
        let mut data = compress_at(6, b"hello world");
        let n = data.len();
        data[n - 4] ^= 0xFF; // corrupt ISIZE
        let mut out = vec![0u8; 64];
        assert_eq!(gzip_decompress(&data, &mut out), Err(GzipDecompressError::BadData));
    }

    #[test]
    fn test_too_short_returns_bad_data() {
        let mut out = vec![0u8; 64];
        assert_eq!(
            gzip_decompress(&[0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF], &mut out),
            Err(GzipDecompressError::BadData)
        );
    }

    #[test]
    fn test_decompress_ex_returns_input_consumed() {
        let input = b"exact input consumed";
        let compressed = compress_at(6, input);
        let mut d = Decompressor::new();
        let mut out = vec![0u8; input.len()];
        let r = gzip_decompress_ex(&mut d, &compressed, &mut out);
        assert!(r.is_ok());
        let (actual_out, actual_in) = r.unwrap();
        assert_eq!(actual_out, input.len());
        assert_eq!(actual_in, compressed.len()); // single member: all bytes consumed
    }

    /// A valid gzip member followed by trailing bytes still decodes, and
    /// `_ex` reports the first member's exact consumption (not the whole slice).
    #[test]
    fn test_trailing_bytes_after_member() {
        let input = b"gzip member then trailing junk";
        let member = compress_at(6, input);
        let member_len = member.len();

        let mut buf = member.clone();
        buf.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00]);

        let mut d = Decompressor::new();
        let mut out = vec![0u8; input.len()];
        let (actual_out, actual_in) = gzip_decompress_ex(&mut d, &buf, &mut out).unwrap();
        assert_eq!(&out[..actual_out], input);
        assert_eq!(actual_in, member_len); // trailing bytes ignored
    }

    /// Two concatenated gzip members are iterated by advancing over the
    /// returned `actual_in` offset of each member.
    #[test]
    fn test_two_concatenated_members() {
        let a = b"first gzip member payload";
        let b = b"second gzip member payload, different length";
        let mut stream = compress_at(6, a);
        stream.extend_from_slice(&compress_at(9, b));

        // Member 1.
        let mut d = Decompressor::new();
        let mut out1 = vec![0u8; a.len()];
        let (out1_n, in1) = gzip_decompress_ex(&mut d, &stream, &mut out1).unwrap();
        assert_eq!(&out1[..out1_n], a);

        // Member 2 begins exactly at `in1`.
        let mut out2 = vec![0u8; b.len()];
        let (out2_n, in2) = gzip_decompress_ex(&mut d, &stream[in1..], &mut out2).unwrap();
        assert_eq!(&out2[..out2_n], b);

        // The two members together account for the whole stream.
        assert_eq!(in1 + in2, stream.len());
    }
}
