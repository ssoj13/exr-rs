//! AVX2 vectorized Adler-32 — x86 SIMD tier (VL=32, USE_VNNI=0).
//!
//! Port of libdeflate `lib/x86/adler32_template.h` instantiated with the
//! **VL=32, USE_VNNI=0, USE_AVX512=0** parameters (the `adler32_x86_avx2`
//! variant from `lib/x86/adler32_impl.h`, target `avx2`).
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/adler32_template.h>
//!
//! Same delayed-multiply algorithm as the SSE2 tier (`adler32_sse2.rs`), but
//! processing 2*VL = 64 bytes per inner iteration with 256-bit vectors. The
//! `raw_mults` ordering is interleaved because `vpunpcklbw`/`vpunpckhbw` and
//! `vpmaddwd` operate on the two 128-bit lanes separately.
//!
//! Output is byte-identical to `adler32::adler32` (scalar source of truth) and
//! to libdeflate; verified by `tests/compare_simd.rs`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::adler32::{adler32_chunk_scalar, MAX_CHUNK_LEN};

const VL: usize = 32;

/// 16-bit multipliers for VL=32. The lane-interleaved ordering exactly matches
/// `raw_mults[4][VL/2]` for VL==32 in the C template (required because
/// vpunpck/vpmaddwd act per 128-bit lane).
const RAW_MULTS: [[i16; 16]; 4] = [
    [64, 63, 62, 61, 60, 59, 58, 57, 48, 47, 46, 45, 44, 43, 42, 41],
    [56, 55, 54, 53, 52, 51, 50, 49, 40, 39, 38, 37, 36, 35, 34, 33],
    [32, 31, 30, 29, 28, 27, 26, 25, 16, 15, 14, 13, 12, 11, 10, 9],
    [24, 23, 22, 21, 20, 19, 18, 17, 8, 7, 6, 5, 4, 3, 2, 1],
];

/// Reduce 256-bit `v_s1`/`v_s2` to scalars and add to `s1`/`s2`.
/// Mirrors `reduce_to_32bits` for VL=32, USE_VNNI=0.
///
/// # Safety
/// Requires `avx2`. Pure register ops.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce_to_32bits(v_s1: __m256i, v_s2: __m256i, s1: &mut u32, s2: &mut u32) {
    // No `unsafe` block is needed: these are register-only intrinsics with no
    // memory access, and since Rust 1.87 such intrinsics are *safe* to call
    // inside a fn whose #[target_feature] set implies theirs. The `unsafe fn`
    // marker stays because the caller must still guarantee AVX2 (see `# Safety`).
    // 256 -> 128.
    let mut a = _mm_add_epi32(
        _mm256_extracti128_si256(v_s1, 0),
        _mm256_extracti128_si256(v_s1, 1),
    );
    let mut b = _mm_add_epi32(
        _mm256_extracti128_si256(v_s2, 0),
        _mm256_extracti128_si256(v_s2, 1),
    );
    // 128 -> 32 (USE_VNNI=0 ⇒ no extra v_s1 0x31 shuffle).
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x31));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x02));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x02));
    *s1 = s1.wrapping_add(_mm_cvtsi128_si32(a) as u32);
    *s2 = s2.wrapping_add(_mm_cvtsi128_si32(b) as u32);
}

/// AVX2 Adler-32. Mirrors `adler32_x86` (VL=32, USE_VNNI=0).
///
/// # Safety
/// Requires the `avx2` target feature. Reached only via
/// `cpu::adler32_tier() == Avx2`, which checks `is_x86_feature_detected!`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn adler32_avx2(adler: u32, data: &[u8]) -> u32 {
    // SAFETY: feature-gated by #[target_feature] + runtime dispatch. All vector
    // loads stay within the current chunk; the scalar tail uses safe code.
    unsafe {
        let mults_a = _mm256_loadu_si256(RAW_MULTS[0].as_ptr() as *const __m256i);
        let mults_b = _mm256_loadu_si256(RAW_MULTS[1].as_ptr() as *const __m256i);
        let mults_c = _mm256_loadu_si256(RAW_MULTS[2].as_ptr() as *const __m256i);
        let mults_d = _mm256_loadu_si256(RAW_MULTS[3].as_ptr() as *const __m256i);
        let zeroes = _mm256_setzero_si256();

        let mut s1 = adler & 0xFFFF;
        let mut s2 = adler >> 16;

        let mut off = 0usize;
        let mut remaining = data.len();

        // chunk cap = MIN(2*VL*(INT16_MAX/UINT8_MAX), MAX_CHUNK_LEN) & ~(2*VL-1)
        // 2*VL*128 = 8192 > 5552, so this is MAX_CHUNK_LEN(5552) & ~63 = 5504.
        const CHUNK_CAP: usize = {
            let a = 2 * VL * (32767 / 255);
            let m = if a < MAX_CHUNK_LEN { a } else { MAX_CHUNK_LEN };
            m & !(2 * VL - 1)
        };

        while remaining > 0 {
            let mut n = remaining.min(CHUNK_CAP);
            remaining -= n;

            if n >= 2 * VL {
                let mut v_s1 = zeroes;
                let mut v_s1_sums = zeroes;
                let mut v_byte_sums_a = zeroes;
                let mut v_byte_sums_b = zeroes;
                let mut v_byte_sums_c = zeroes;
                let mut v_byte_sums_d = zeroes;

                s2 = s2.wrapping_add(s1.wrapping_mul((n & !(2 * VL - 1)) as u32));

                while n >= 2 * VL {
                    let p = data.as_ptr().add(off);
                    let data_a = _mm256_loadu_si256(p as *const __m256i);
                    let data_b = _mm256_loadu_si256(p.add(VL) as *const __m256i);

                    v_s1_sums = _mm256_add_epi32(v_s1_sums, v_s1);
                    v_byte_sums_a =
                        _mm256_add_epi16(v_byte_sums_a, _mm256_unpacklo_epi8(data_a, zeroes));
                    v_byte_sums_b =
                        _mm256_add_epi16(v_byte_sums_b, _mm256_unpackhi_epi8(data_a, zeroes));
                    v_byte_sums_c =
                        _mm256_add_epi16(v_byte_sums_c, _mm256_unpacklo_epi8(data_b, zeroes));
                    v_byte_sums_d =
                        _mm256_add_epi16(v_byte_sums_d, _mm256_unpackhi_epi8(data_b, zeroes));
                    v_s1 = _mm256_add_epi32(
                        v_s1,
                        _mm256_add_epi32(
                            _mm256_sad_epu8(data_a, zeroes),
                            _mm256_sad_epu8(data_b, zeroes),
                        ),
                    );
                    off += 2 * VL;
                    n -= 2 * VL;
                }

                // LOG2_VL = 5 ⇒ shift by LOG2_VL+1 = 6.
                let v_s2 = _mm256_add_epi32(
                    _mm256_slli_epi32(v_s1_sums, 6),
                    _mm256_add_epi32(
                        _mm256_add_epi32(
                            _mm256_madd_epi16(v_byte_sums_a, mults_a),
                            _mm256_madd_epi16(v_byte_sums_b, mults_b),
                        ),
                        _mm256_add_epi32(
                            _mm256_madd_epi16(v_byte_sums_c, mults_c),
                            _mm256_madd_epi16(v_byte_sums_d, mults_d),
                        ),
                    ),
                );
                reduce_to_32bits(v_s1, v_s2, &mut s1, &mut s2);
            }

            // Scalar tail (0 <= n < 2*VL) + mod-DIVISOR reduction; mirrors
            // ADLER32_CHUNK(s1, s2, p, n).
            adler32_chunk_scalar(&mut s1, &mut s2, &data[off..off + n]);
            off += n;
        }

        (s2 << 16) | s1
    }
}

#[cfg(test)]
mod tests {
    use crate::adler32::adler32_generic;

    #[test]
    fn avx2_matches_scalar_varied() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skip: avx2 not available");
            return;
        }
        let mut x: u32 = 0x1357_9BDF;
        let mut buf = vec![0u8; 12000]; // > MAX_CHUNK_LEN to hit the chunk loop
        for b in buf.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x & 0xFF) as u8;
        }
        for &init in &[1u32, 0, 0xFFFF_FFFF, 0x0001_0000] {
            for len in 0..=buf.len() {
                let data = &buf[..len];
                // SAFETY: avx2 presence checked above.
                let simd = unsafe { super::adler32_avx2(init, data) };
                assert_eq!(simd, adler32_generic(init, data), "len={len} init={init:#x}");
            }
        }
    }
}
