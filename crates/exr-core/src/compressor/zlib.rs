//! Single SSOT for every codec's zlib compress/decompress call — port of
//! `exr_compress_buffer` / `exr_uncompress_buffer`
//! (`OpenEXRCore/compression.c`, v3.4.15
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/compression.c>).
//!
//! ZIP, PXR24, DWA (its AC-deflate/DC/RLE sub-streams), and `IDManifest` all
//! deflate/inflate raw byte buffers through this ONE pair of functions
//! instead of each having its own
//! `deflate_rs::Compressor::new(level)`/`.zlib_compress(..)` call site and its
//! own re-derivation of the `level < 0` fallback and the corrupt-chunk error
//! text. One error kind ([`ExrError::Input`]) and one message format per
//! function, tagged with the caller's `codec` name.
//!
//! # Persistent codec state
//! A deflate-only-vs-libdeflate breakdown found the FACADE this
//! module used to call (`deflate_rs::Compressor::zlib_compress`) allocating a
//! fresh, zero-filled `~1.5 MiB` `Vec` per call regardless of the caller's own
//! persistent output buffer, and this module's own `uncompress_into` doing an
//! unconditional `out.clear(); out.resize(cap, 0)` — a full zero-fill of the
//! decode buffer EVERY call, even though the decoder overwrites almost all of
//! it and C never zeroes at all. Both are fixed here:
//! - [`compress_into`]/[`uncompress_into`] now take the caller's OWN
//!   persistent `deflate_rs::deflate_compress::Compressor` /
//!   `deflate_rs::deflate_decompress::Decompressor` (the crate's LOW-LEVEL,
//!   buffer-reusing API — [`crate::compressor::zip::ZipCompressor`]/
//!   [`crate::compressor::pxr24::Pxr24Compressor`]/
//!   [`crate::compressor::dwa::DwaCompressor`] each own one now, constructed
//!   once via [`new_deflate_compressor`]/`LowDecompressor::new()`, not
//!   per-chunk) instead of constructing one internally every call.
//! - `uncompress_into`'s destination buffer is grown (with zero-fill for the
//!   NEW bytes only) or `truncate`d to `cap` — never unconditionally
//!   zero-filled — since the decoder only ever reads back bytes it already
//!   wrote (LZ77 back-references never reach un-written output), stale bytes
//!   left over from a PRIOR call are never observed mid-decode. The final
//!   `out.truncate(actual_out)` still runs regardless, so
//!   [`ShortOutputPolicy::TolerateShort`]'s deterministic-zero-tail guarantee
//!   is unaffected: it never came
//!   from THIS buffer holding zeroes past `actual_out` (which is truncated
//!   away either way) — it comes from `DwaCompressor::decompress_dwa`'s own
//!   `output` buffer being zero-resized before its assembly loop reads the
//!   (possibly short) `unknown_data` bounded by `.min(unknown_data.len())`.
//!
//! `IDManifest::compress`/`decompress` are one-shot, once-per-file-attribute
//! operations (not a per-chunk hot path), so they construct a throwaway
//! `Compressor`/`Decompressor` locally at each call site instead of carrying
//! a persistent field — there is no chunk loop for a persistent instance to
//! amortize over there.

use crate::error::{ExrError, Result};

pub(crate) use deflate_rs::deflate_compress::Compressor as LowCompressor;
pub(crate) use deflate_rs::deflate_decompress::Decompressor as LowDecompressor;

/// Default ZIP/ZIPS/PXR24/DWA libdeflate compression level used by OpenEXR.
///
/// Derived from `compression.c: EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4` (used when
/// the context's `default_zip_level` is the unset sentinel `-1`). Using this
/// level with [`deflate_rs`] reproduces the exact zlib bytes C++ writes.
pub const DEFAULT_ZIP_LEVEL: i32 = 4;

/// Resolve `level < 0` ("let the codec pick") to [`DEFAULT_ZIP_LEVEL`] and
/// construct the persistent low-level compressor for it — the ONE place that
/// performs this resolution (was `compress_into`'s job before this became a
/// construction-time step instead of a per-call one).
///
/// `exr_compress_buffer` (`compression.c:111-117`) re-reads the process-global
/// default and, if that is *also* still unset, falls back to
/// `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4`. Callers that already resolve a
/// per-header level (`Header::zip_level()`/`dwa_level()`) only hit this
/// fallback on "still unset anywhere", the same case C++ falls back to.
pub(crate) fn new_deflate_compressor(level: i32) -> Result<LowCompressor> {
    let level = if level < 0 { DEFAULT_ZIP_LEVEL } else { level };
    LowCompressor::new(level).map_err(|_| {
        ExrError::Arg(format!("deflate-rs rejected zlib compression level {level} (internal error)"))
    })
}

/// Deflate `src` into a caller-owned, reusable buffer using the caller's OWN
/// persistent `comp` (constructed once via [`new_deflate_compressor`], not
/// per call — see the module doc).
///
/// `out` is grown (zero-filling only the newly added bytes) to the worst-case
/// bound if it is not already at least that large, then truncated to the
/// real compressed length — no fresh `Vec` allocation on the common path
/// where the caller's buffer is already big enough.
pub(crate) fn compress_into(comp: &mut LowCompressor, src: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let bound = deflate_rs::zlib_compress::zlib_compress_bound(src.len());
    if out.len() < bound {
        out.resize(bound, 0);
    }
    let written = deflate_rs::zlib_compress::zlib_compress(comp, src, out);
    if written == 0 {
        // `zlib_compress` only returns 0 when `out.len() <= ZLIB_MIN_OVERHEAD`
        // (mirrors `libdeflate_zlib_compress`'s "buffer too small" contract) --
        // unreachable given `out.len() >= bound` above, but surfaced as a
        // clear error rather than silently truncating `out` to empty.
        return Err(ExrError::Logic(
            "deflate-rs zlib_compress returned 0 despite a correctly-sized buffer".into(),
        ));
    }
    out.truncate(written);
    Ok(())
}

/// Which of libdeflate's two `actual_out_nbytes_ret` contracts a caller wants
/// — 1:1 port of the NULL vs non-NULL choice each C call site makes before
/// calling `libdeflate_zlib_decompress_ex` (`decompress_template.h`'s tail:
/// `if (actual_out_nbytes_ret) { *actual_out_nbytes_ret = ...; } else if
/// (out_next != out_end) { return LIBDEFLATE_SHORT_OUTPUT; }`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ShortOutputPolicy {
    /// Non-NULL `actual_out_nbytes_ret` in C — used by ZIP, PXR24, DWA's
    /// AC-deflate/DC/RLE sub-streams, and `IDManifest`. `LIBDEFLATE_SUCCESS`
    /// is the only possible non-error outcome (`SHORT_OUTPUT` only exists in
    /// the NULL branch), so the consume-all-input check ALWAYS
    /// applies, regardless of how many bytes were actually decoded.
    RequireExact,
    /// NULL `actual_out_nbytes_ret` in C — used ONLY by DWA's UNKNOWN
    /// sub-stream (`internal_dwa_compressor.h:922-930`). A decode that
    /// produces FEWER than `cap` bytes hits `LIBDEFLATE_SHORT_OUTPUT`, which
    /// `exr_uncompress_buffer` (`compression.c:144-208`) converts to
    /// `EXR_ERR_SUCCESS` UNCONDITIONALLY — no consume-all check, no
    /// exact-size requirement (our undecoded tail is a deterministic
    /// zero — see the module doc for where that guarantee actually lives).
    TolerateShort,
}

/// Inflate `src` into a caller-owned, reusable buffer using the caller's OWN
/// persistent `inner` (constructed once via `LowDecompressor::new()`, not per
/// call — see the module doc) — a 1:1 port of `exr_uncompress_buffer`
/// (`compression.c:144-208`), including its NULL vs non-NULL
/// `actual_out_nbytes_ret` split (see [`ShortOutputPolicy`]).
///
/// `out` is grown (zero-filling only the newly added bytes) to `cap` if it is
/// not already at least that large, else truncated to `cap` WITHOUT touching
/// existing bytes — no unconditional zero-fill. This is sound because
/// `zlib_decompress_ex`'s underlying LZ77 decode only ever reads back bytes
/// it has already written earlier in the SAME call (a back-reference can
/// never point past what has been decoded so far), so stale bytes left over
/// from a prior, unrelated call in the tail of a reused buffer are never
/// observed mid-decode — and the final `out.truncate(actual_out)` discards
/// that tail (real or stale) either way.
///
/// # The consume-all-input check
/// Under [`ShortOutputPolicy::RequireExact`] this always applies (see the
/// variant's doc — it is exactly what C does for ZIP/PXR24/AC/DC/RLE/
/// IDManifest, not an approximation). Under [`ShortOutputPolicy::TolerateShort`]
/// it applies only when the decode fills `cap` exactly (the ordinary SUCCESS
/// path); a genuinely short decode skips it entirely, matching C's
/// unconditional `SHORT_OUTPUT -> EXR_ERR_SUCCESS` conversion.
///
/// The old code called the *counted* form
/// (`Decompressor::zlib_decompress`), which silently discarded
/// `actual_in_bytes`, so trailing bytes after a valid zlib stream were
/// accepted instead of rejected as `CORRUPT_CHUNK`. DWA's own duplicate
/// `zlib_decompress` (since removed) had the same gap for its AC/DC/RLE
/// sub-streams — closed by routing them through here.
///
/// A caller that needs EXACTLY `cap` bytes under `RequireExact` (not just "at
/// most") checks `out.len()` itself after this returns — matching each C
/// call site's own follow-up `destLen != uncompBytes` check
/// (`internal_dwa_compressor.h:978`/`:1017`/`:1060`), which is REAL and
/// necessary: a non-NULL `actual_out` tolerates any actual size `<= cap`, it
/// does not require an exact fill.
pub(crate) fn uncompress_into(
    inner: &mut LowDecompressor,
    src: &[u8],
    out: &mut Vec<u8>,
    cap: usize,
    codec: &str,
    policy: ShortOutputPolicy,
) -> Result<()> {
    super::ensure_len(out, cap);
    let bad = |e| ExrError::Input(format!("{codec} zlib stream failed to decompress: {e:?}"));
    let (actual_out, actual_in) = match policy {
        ShortOutputPolicy::RequireExact => {
            deflate_rs::zlib_decompress::zlib_decompress_ex(inner, src, out).map_err(bad)?
        }
        // NULL `actual_out`: a short stream is success and libdeflate never reaches its
        // Adler-32 trailer, so a corrupt-but-short stream must not fail on the checksum.
        ShortOutputPolicy::TolerateShort => {
            match deflate_rs::zlib_decompress::zlib_decompress_null_out(inner, src, out).map_err(bad)? {
                deflate_rs::zlib_decompress::NullOut::Short { actual_out } => {
                    out.truncate(actual_out);
                    return Ok(());
                }
                deflate_rs::zlib_decompress::NullOut::Full { actual_in } => (cap, actual_in),
            }
        }
    };
    if actual_in != src.len() {
        return Err(ExrError::Input(format!(
            "{codec} zlib stream: {} trailing byte(s) after the deflate stream (corrupt chunk)",
            src.len() - actual_in
        )));
    }
    out.truncate(actual_out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `abcabc...` as a zlib stream at level 6.
    fn stream() -> Vec<u8> {
        let mut c = LowCompressor::new(6).expect("level");
        let src = b"abcabcabcabc";
        let mut z = vec![0u8; deflate_rs::zlib_compress::zlib_compress_bound(src.len())];
        let n = deflate_rs::zlib_compress::zlib_compress(&mut c, src, &mut z);
        z.truncate(n);
        z
    }

    /// DWA's UNKNOWN sub-stream (`TolerateShort`, C's NULL `actual_out`): a short stream is
    /// accepted WITHOUT checking Adler-32 (`libdeflate` returns `SHORT_OUTPUT` first), so even
    /// a wrecked trailer passes; the exact-size policy still rejects it. Found by the
    /// pixel-class parity test: a flipped `golden_dwab.exr` decoded fine in C++ only.
    #[test]
    fn tolerate_short_ignores_the_checksum_of_a_short_stream() {
        let mut z = stream();
        let last = z.len() - 1;
        z[last] ^= 0xFF;
        let mut d = LowDecompressor::new();
        let mut out = Vec::new();
        uncompress_into(&mut d, &z, &mut out, 20, "t", ShortOutputPolicy::TolerateShort).expect("short is tolerated");
        assert_eq!(out, b"abcabcabcabc");
        assert!(uncompress_into(&mut d, &z, &mut out, 20, "t", ShortOutputPolicy::RequireExact).is_err());
        // A stream that fills the buffer exactly still gets its checksum verified.
        assert!(uncompress_into(&mut d, &z, &mut out, 12, "t", ShortOutputPolicy::TolerateShort).is_err());
    }
}
