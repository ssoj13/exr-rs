//! Port of libdeflate `lib/adler32.c` — scalar Adler-32 checksum.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/adler32.c
//!
//! # What is ported
//! - `adler32_generic` — the portable scalar implementation, ported 1:1.
//! - `libdeflate_adler32` — the public entry point, ported as [`adler32`].
//! - `ADLER32_CHUNK` macro — inlined directly into `adler32_generic` logic.
//! - `DIVISOR` and `MAX_CHUNK_LEN` constants.
//!
//! # N/A in Rust (documented here, not ported)
//! - SIMD dispatch (`arch_select_adler32_func`, `dispatch_adler32`, `adler32_impl`
//!   volatile function pointer) — replaced by the safe runtime dispatcher in
//!   `crate::cpu` and the `adler32_sse2`/`adler32_avx2` modules.  The scalar path
//!   is always the correct fallback and is the source of truth for byte-exact
//!   output.
//! - `x86/adler32_impl.h` and `arm/adler32_impl.h` includes — ported separately
//!   as the SIMD modules above, not in this file.
//! - `adler32_func_t` typedef — not needed; Rust uses trait objects / fn pointers
//!   differently.
//! - `MAYBE_UNUSED` attribute on `adler32_generic` — the function is `pub(crate)`;
//!   Rust does not warn on unused pub items.

/// Adler-32 modulus. `DIVISOR` in the C source.
pub(crate) const DIVISOR: u32 = 65521;

/// Maximum bytes that can be processed in one chunk without `s2` overflowing a
/// `u32`.  Must be a multiple of 4 (the C macro masks with `& ~3`).
/// `MAX_CHUNK_LEN` in the C source.  Also used by the SIMD tiers to bound their
/// per-chunk lengths.
pub(crate) const MAX_CHUNK_LEN: usize = 5552;

/// Scalar Adler-32 over one chunk of at most `MAX_CHUNK_LEN & !3` bytes.
///
/// This is a direct port of the `ADLER32_CHUNK` macro + `adler32_generic` from
/// `adler32.c`.  The 4-byte unrolling with five running accumulators is preserved
/// exactly to match libdeflate's ILP optimisation.
///
/// `s1` = lower 16 bits of running checksum, `s2` = upper 16 bits.
/// Both are reduced mod `DIVISOR` at the end of each chunk.
#[inline]
pub(crate) fn adler32_chunk_scalar(s1: &mut u32, s2: &mut u32, p: &[u8]) {
    let mut n = p.len();
    let mut i = 0usize;

    // Process 4 bytes at a time — mirrors the `if (n >= 4) { do { ... } while (n >= 4); }` block.
    if n >= 4 {
        let mut s1_sum: u32 = 0;
        let mut byte_0_sum: u32 = 0;
        let mut byte_1_sum: u32 = 0;
        let mut byte_2_sum: u32 = 0;
        let mut byte_3_sum: u32 = 0;

        while n >= 4 {
            s1_sum += *s1;
            *s1 = s1.wrapping_add(
                p[i] as u32 + p[i + 1] as u32 + p[i + 2] as u32 + p[i + 3] as u32,
            );
            byte_0_sum += p[i] as u32;
            byte_1_sum += p[i + 1] as u32;
            byte_2_sum += p[i + 2] as u32;
            byte_3_sum += p[i + 3] as u32;
            i += 4;
            n -= 4;
        }

        // Mirrors: s2 += (4*(s1_sum+byte_0_sum)) + (3*byte_1_sum) + (2*byte_2_sum) + byte_3_sum
        *s2 = s2.wrapping_add(
            4u32.wrapping_mul(s1_sum.wrapping_add(byte_0_sum))
                .wrapping_add(3u32.wrapping_mul(byte_1_sum))
                .wrapping_add(2u32.wrapping_mul(byte_2_sum))
                .wrapping_add(byte_3_sum),
        );
    }

    // Handle remaining bytes (0–3). Mirrors `for (; n; n--, p++) { s1 += *p; s2 += s1; }`
    while n > 0 {
        *s1 = s1.wrapping_add(p[i] as u32);
        *s2 = s2.wrapping_add(*s1);
        i += 1;
        n -= 1;
    }

    *s1 %= DIVISOR;
    *s2 %= DIVISOR;
}

/// Scalar Adler-32. Ports `adler32_generic` from `adler32.c`.
///
/// `adler` is the running checksum (pass `1` to start a new checksum).
/// Returns the updated checksum.
pub(crate) fn adler32_generic(adler: u32, data: &[u8]) -> u32 {
    let mut s1 = adler & 0xFFFF;
    let mut s2 = adler >> 16;

    let mut remaining = data.len();
    let mut offset = 0usize;

    while remaining > 0 {
        // Chunk size = MIN(remaining, MAX_CHUNK_LEN & ~3).
        // The `& !3` ensures the chunk is a multiple of 4, matching `MAX_CHUNK_LEN & ~3` in C.
        let chunk_len = remaining.min(MAX_CHUNK_LEN & !3);
        adler32_chunk_scalar(&mut s1, &mut s2, &data[offset..offset + chunk_len]);
        offset += chunk_len;
        remaining -= chunk_len;
    }

    (s2 << 16) | s1
}

/// Public Adler-32 entry point.  Mirrors `libdeflate_adler32`.
///
/// - `adler`: running checksum; pass `1` to start a fresh checksum.
/// - `buffer`: data to checksum; if empty the function returns `adler` unchanged
///   (matches the C `if (buffer == NULL) return 1` initial-value convention —
///   we return `adler` which for a fresh call should be `1`).
///
/// Returns the updated Adler-32 checksum.
pub fn adler32(adler: u32, buffer: &[u8]) -> u32 {
    if buffer.is_empty() {
        // Mirror of `if (buffer == NULL) return 1` — return current running value.
        return adler;
    }
    // Runtime SIMD dispatch (mirrors libdeflate's `adler32_impl` function
    // pointer). Every tier returns the identical checksum as the scalar path
    // (`adler32_generic`), which remains the source of truth and the fallback.
    match crate::cpu::adler32_tier() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::Adler32Tier::Avx2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("avx2")`,
            // satisfying the `#[target_feature(enable = "avx2")]` precondition.
            unsafe { crate::adler32_avx2::adler32_avx2(adler, buffer) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        crate::cpu::Adler32Tier::Sse2 => {
            // SAFETY: tier selected only after `is_x86_feature_detected!("sse2")`,
            // satisfying the `#[target_feature(enable = "sse2")]` precondition.
            unsafe { crate::adler32_sse2::adler32_sse2(adler, buffer) }
        }
        crate::cpu::Adler32Tier::Scalar => adler32_generic(adler, buffer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-vector: empty input with initial value 1 must return 1.
    /// Mirrors `libdeflate_adler32(1, NULL, 0) == 1`.
    #[test]
    fn empty_is_one() {
        assert_eq!(adler32(1, b""), 1);
    }

    /// Known-vector: adler32(1, "Wikipedia") == 0x11E60398.
    /// Standard Adler-32 test vector.
    #[test]
    fn wikipedia_vector() {
        assert_eq!(adler32(1, b"Wikipedia"), 0x11E60398);
    }

    /// Known-vector: adler32(1, "abc") == 0x024d0127.
    #[test]
    fn abc_vector() {
        assert_eq!(adler32(1, b"abc"), 0x024d0127);
    }

    /// Incremental == one-shot: chunking must not affect the result.
    #[test]
    fn incremental_matches_oneshot() {
        let data = b"Hello, world! This is a longer test string for Adler-32.";
        let one_shot = adler32(1, data);

        // Feed in three unequal chunks
        let a = adler32(1, &data[..10]);
        let b = adler32(a, &data[10..30]);
        let c = adler32(b, &data[30..]);
        assert_eq!(c, one_shot);
    }

    /// Stress: data longer than MAX_CHUNK_LEN exercises the chunk-loop path.
    #[test]
    fn long_data_chunk_loop() {
        let data: Vec<u8> = (0u8..=255).cycle().take(MAX_CHUNK_LEN * 3 + 7).collect();
        let result = adler32(1, &data);
        // Must not panic and must equal a naive scalar reference.
        let expected = naive_adler32(&data);
        assert_eq!(result, expected);
    }

    /// Reference implementation (simple, clearly-correct loop) for cross-checking.
    fn naive_adler32(data: &[u8]) -> u32 {
        let mut s1: u32 = 1;
        let mut s2: u32 = 0;
        for &b in data {
            s1 = (s1 + b as u32) % DIVISOR;
            s2 = (s2 + s1) % DIVISOR;
        }
        (s2 << 16) | s1
    }
}
