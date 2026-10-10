//! Shared **byte predictor + interleave** used by both the RLE and ZIP/ZIPS
//! scanline codecs.
//!
//! # Why this module exists (DEDUP / single-responsibility)
//!
//! OpenEXR applies the *exact same* two-step byte pre-processing before both
//! RLE and ZIP compression:
//!
//! 1. **reorder** — de-interleave the bytes into two halves so even-index bytes
//!    land in the first half and odd-index bytes in the second half (this
//!    groups the high/low bytes of 16-bit samples, the typical EXR layout,
//!    improving run lengths / deflate matches);
//! 2. **predict** — replace every byte (after the first) by its signed delta
//!    from the previous byte, biased by `+(128+256)` and truncated to a byte
//!    (the standard OpenEXR first-difference predictor).
//!
//! In the C++ / OpenEXRCore source this is literally the same algorithm copied
//! into two places:
//! * `internal_rle.c` — `reorder_and_predict` / `unpredict_and_reorder`;
//! * `internal_zip.c` — `internal_zip_deconstruct_bytes` (reorder+predict) and
//!   `internal_zip_reconstruct_bytes` → `reconstruct` (unpredict) + `interleave`
//!   (reorder inverse).
//!
//! The two implementations are byte-for-byte identical (the scalar `reconstruct`
//! in `internal_zip.c` — `t[-1] + t[0] - 128` — and the scalar `interleave`
//! match `unpredict_and_reorder` exactly; the SSE/NEON variants in
//! `internal_zip.c` are SIMD optimisations that produce identical output).  To
//! avoid duplicating it, the shared scalar implementation lives here and is
//! used by both `compressor::rle` and `compressor::zip`.
//!
//! # SIMD (decode side only — via `codec-simd`)
//! Profiling ZIP found the DECODE-side transform
//! (`unpredict_and_reorder`) taking **~53% of total `ZipCompressor::uncompress`
//! time** — MORE than the zlib inflate itself — because it was pure scalar,
//! one byte at a time, while C's `reconstruct`/`interleave`
//! (`internal_zip.c:49-244`) are SIMD-vectorized (SSE4.1/NEON prefix-sum for
//! `reconstruct`, SSE2/NEON unpack for `interleave`; `IMF_HAVE_SSE2`/
//! `IMF_HAVE_SSE4_1` are unconditionally defined on MSVC x86/x64 builds
//! regardless of `/arch`, so the vcpkg oracle genuinely takes these paths).
//! The ENCODE-side transform (`reorder_and_predict`) has NO SIMD counterpart
//! in C at all (`internal_zip_deconstruct_bytes`, scalar-only) and measured
//! at only ~3% of total `compress()` time, so it is intentionally left
//! scalar here too — no gap to close, matching C exactly.
//!
//! `unpredict_and_reorder` calls [`codec_simd::prefix_sum_bias_u8_inplace`] +
//! [`codec_simd::interleave_halves_u8`] — 1:1 ports of C's SSE4.1/SSE2 (+
//! NEON) `reconstruct`/`interleave`. This crate does **not** write its own
//! `core::arch`/`unsafe` SIMD (an earlier version briefly did; moved to
//! `codec-simd` per the project's "unsafe stays in deflate-rs / codec-simd,
//! never in exr-core" invariant). `codec-simd`'s own
//! `tests/cross_tier.rs` is the differential-vs-scalar test for these two
//! primitives; this module only needs to prove the ROUND TRIP is correct
//! (below), since correctness-per-tier is `codec-simd`'s contract. That a
//! SIMD body is actually selected on a capable CPU is pinned by
//! `tests/codec_simd_tier.rs`.
//!
//! Port targets (pinned tag v3.3.11, SIMD cited against v3.4.15's
//! `internal_zip.c` — see the SIMD section above):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_rle.c>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_zip.c>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_zip.c>

// ---------------------------------------------------------------------------
// reorder + predict — port of `reorder_and_predict` (internal_rle.c lines 67-91)
//                      == `internal_zip_deconstruct_bytes` (internal_zip.c)
// ---------------------------------------------------------------------------

/// De-interleave then first-difference-predict `packed` into `scratch`.
///
/// `scratch` must be exactly `packed.len()` bytes (the caller sizes it so).
///
/// C reference (`internal_rle.c` / `internal_zip_deconstruct_bytes`):
/// ```c
/// uint8_t* t1 = scratch;
/// uint8_t* t2 = t1 + (count + 1) / 2;
/// const uint8_t* raw = source;
/// const uint8_t* const stop = raw + count;
/// while (raw < stop) { *(t1++) = *(raw++); if (raw < stop) *(t2++) = *(raw++); }
/// t1 = scratch; t2 = t1 + count; t1++;
/// p = t1[-1];
/// while (t1 < t2) { d = t1[0] - p + (128+256); p = t1[0]; t1[0] = (uint8_t)d; ++t1; }
/// ```
///
/// Scalar only — see the module's SIMD section for why (C has no SIMD
/// variant for this direction either).
pub fn reorder_and_predict(scratch: &mut [u8], packed: &[u8]) {
    let n = packed.len();
    if n == 0 {
        return;
    }

    // --- reorder: even bytes → first half, odd bytes → second half ---
    // t2 starts at ceil(n/2) — the split the C code uses (`(count + 1) / 2`).
    let t2_start = n.div_ceil(2);
    let mut i1 = 0usize; // write cursor into the first half
    let mut i2 = t2_start; // write cursor into the second half
    let mut k = 0usize; // read cursor into `packed`
    while k < n {
        scratch[i1] = packed[k];
        i1 += 1;
        k += 1;
        if k < n {
            scratch[i2] = packed[k];
            i2 += 1;
            k += 1;
        }
    }

    // --- predict: first-difference over the whole reordered buffer ---
    // C: p = *t1++; while (t1 < stop) { d = *t1 - p + (128+256); p = *t1; *t1 = d; }
    // `p` carries the *previous* (pre-modification) byte; each byte after the
    // first is replaced by its biased delta.  `+(128+256)` matches C exactly
    // (the `+256` is a no-op modulo 256, kept for a literal 1:1 port); `as u8`
    // performs the C `(uint8_t)` truncation.
    let mut p = scratch[0] as i32;
    for cell in scratch.iter_mut().skip(1) {
        let cur = *cell as i32;
        let d = cur - p + (128 + 256);
        p = cur;
        *cell = d as u8;
    }
}

// ---------------------------------------------------------------------------
// unpredict + reorder — port of `unpredict_and_reorder` (internal_rle.c 170-193)
//                       == `reconstruct` + `interleave` (internal_zip.c)
// ---------------------------------------------------------------------------

/// Invert [`reorder_and_predict`]: un-delta `scratch` in place, then
/// re-interleave it into `out`.
///
/// `out` and `scratch` are both exactly `packedbytes` long.
///
/// C reference (`internal_rle.c` / scalar `reconstruct`+`interleave`):
/// ```c
/// /* reconstruct: un-delta */
/// uint8_t* t = buf + 1;
/// while (t < buf + sz) { int d = t[-1] + t[0] - 128; t[0] = (uint8_t)d; ++t; }
/// /* interleave: first half → even bytes, second half → odd bytes */
/// const uint8_t* t1 = source;
/// const uint8_t* t2 = source + (outSize + 1) / 2;
/// uint8_t* s = out;
/// while (s < out + outSize) { *(s++) = *(t1++); if (s < stop) *(s++) = *(t2++); }
/// ```
///
/// Both steps go through `codec-simd`'s safe, SIMD-accelerated API (see the
/// module's SIMD section) — no `unsafe` in this crate.
pub fn unpredict_and_reorder(out: &mut [u8], scratch: &mut [u8]) {
    let n = scratch.len();
    if n == 0 {
        return;
    }
    codec_simd::prefix_sum_bias_u8_inplace(scratch);
    let t2_start = n.div_ceil(2);
    let (t1, t2) = scratch.split_at(t2_start);
    codec_simd::interleave_halves_u8(t1, t2, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// reorder+predict followed by its inverse must reproduce the input exactly.
    fn round_trip(data: &[u8]) {
        let n = data.len();
        let mut scratch = vec![0u8; n];
        reorder_and_predict(&mut scratch, data);
        let mut restored = vec![0u8; n];
        unpredict_and_reorder(&mut restored, &mut scratch);
        assert_eq!(restored, data, "predictor round trip mismatch");
    }

    #[test]
    fn round_trip_empty() {
        round_trip(&[]);
    }

    #[test]
    fn round_trip_single() {
        round_trip(&[0x42]);
    }

    #[test]
    fn round_trip_even_len() {
        let data: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
        round_trip(&data);
    }

    #[test]
    fn round_trip_odd_len() {
        let data: Vec<u8> = (0..=250u8).step_by(3).collect();
        round_trip(&data);
    }

    /// A length spanning many SIMD chunks (16/32 bytes) with a non-multiple
    /// remainder in both directions, plus a length matching the real ZIP
    /// chunk size the SIMD path was added for (16 lines x 16384 width x 3
    /// HALF channels). Correctness of the SIMD tiers THEMSELVES (vs scalar)
    /// is `codec-simd`'s own `tests/cross_tier.rs` contract; this only needs
    /// to prove the round trip through the safe API is correct end to end,
    /// on whatever tier this machine's `codec-simd::Dispatch::detect()`
    /// picks (exercised for real via `CODEC_SIMD_FORCE_LEVEL` in CI/manual
    /// sweeps, not re-implemented here).
    #[test]
    fn round_trip_multi_chunk_lengths() {
        for &n in &[15usize, 16, 17, 31, 32, 33, 47, 48, 49, 1000, 1572864] {
            let data: Vec<u8> = (0..n).map(|i| (i as u32).wrapping_mul(2654435761) as u8).collect();
            round_trip(&data);
        }
    }
}
