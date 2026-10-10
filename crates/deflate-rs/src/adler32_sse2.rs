//! SSE2 vectorized Adler-32 — x86 SIMD tier (VL=16, USE_VNNI=0).
//!
//! Port of libdeflate `lib/x86/adler32_template.h` instantiated with the
//! **VL=16, USE_VNNI=0, USE_AVX512=0** parameters (the `adler32_x86_sse2`
//! variant from `lib/x86/adler32_impl.h`, target `sse2`).
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/adler32_template.h>
//!
//! # Algorithm (USE_VNNI=0, the SSE2/AVX2-compatible path)
//! Bytes are horizontally summed with `psadbw + paddd` (for s1). For s2, each
//! byte position `i` of every 2*VL-byte iteration is accumulated into a 16-bit
//! per-position counter via `punpck{l,h}bw + paddw`; after the inner loop the
//! counters are multiplied by `[2*VL, 2*VL-1, ..., 1]` with `pmaddwd` and added
//! to s2 (delayed-multiply trick). The scalar tail (`< 2*VL` bytes of each
//! chunk) reuses the scalar `adler32_chunk`.
//!
//! # Correctness / byte-identity
//! The chunk length here is `MIN(len, MIN(2*VL*(INT16_MAX/UINT8_MAX),
//! MAX_CHUNK_LEN) & ~(2*VL-1))` (≤ 4096 for SSE2). Because every chunk reduces
//! `s1`,`s2` mod DIVISOR, the *final* checksum is independent of chunk size, so
//! this is byte-identical to `adler32::adler32` (the scalar source of truth) and
//! to libdeflate. Verified by `tests/compare_simd.rs`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::adler32::{adler32_chunk_scalar, MAX_CHUNK_LEN};

const VL: usize = 16;

/// 16-bit multipliers `[2*VL, 2*VL-1, ..., 1]` arranged in 4 rows of VL/2.
/// For VL=16 the ordering is simple (matches `vpunpcklbw`/`vpunpckhbw` on one
/// 128-bit lane). Mirrors `raw_mults[4][VL/2]` for VL==16.
const RAW_MULTS: [[i16; 8]; 4] = [
    [32, 31, 30, 29, 28, 27, 26, 25],
    [24, 23, 22, 21, 20, 19, 18, 17],
    [16, 15, 14, 13, 12, 11, 10, 9],
    [8, 7, 6, 5, 4, 3, 2, 1],
];

/// Reduce the four 32-bit lanes of `v_s1`/`v_s2` into scalars and add to `s1`/`s2`.
/// Mirrors `reduce_to_32bits` for VL=16, USE_VNNI=0.
///
/// # Safety
/// Requires `sse2`. Pure register ops.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn reduce_to_32bits(v_s1: __m128i, v_s2: __m128i, s1: &mut u32, s2: &mut u32) {
    // No `unsafe` block is needed: these are register-only intrinsics with no
    // memory access, and since Rust 1.87 such intrinsics are *safe* to call
    // inside a fn whose #[target_feature] set implies theirs. The `unsafe fn`
    // marker stays because the caller must still guarantee SSE2 (see `# Safety`).
    let mut a = v_s1;
    let mut b = v_s2;
    // (USE_VNNI=0 ⇒ no extra v_s1 shuffle-by-0x31 step.)
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x31));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x02));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x02));
    *s1 = s1.wrapping_add(_mm_cvtsi128_si32(a) as u32);
    *s2 = s2.wrapping_add(_mm_cvtsi128_si32(b) as u32);
}

/// SSE2 Adler-32. Mirrors `adler32_x86` (VL=16, USE_VNNI=0).
///
/// `adler` is the running checksum; returns the updated checksum.
///
/// # Safety
/// Requires the `sse2` target feature. `data` is any byte slice. Reached only
/// via `cpu::adler32_tier() == Sse2`, which checks `is_x86_feature_detected!`.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn adler32_sse2(adler: u32, data: &[u8]) -> u32 {
    // SAFETY: feature-gated by #[target_feature] + runtime dispatch. All vector
    // loads stay within the current chunk slice; the scalar tail uses safe code.
    unsafe {
        let mults_a = _mm_loadu_si128(RAW_MULTS[0].as_ptr() as *const __m128i);
        let mults_b = _mm_loadu_si128(RAW_MULTS[1].as_ptr() as *const __m128i);
        let mults_c = _mm_loadu_si128(RAW_MULTS[2].as_ptr() as *const __m128i);
        let mults_d = _mm_loadu_si128(RAW_MULTS[3].as_ptr() as *const __m128i);
        let zeroes = _mm_setzero_si128();

        let mut s1 = adler & 0xFFFF;
        let mut s2 = adler >> 16;

        // We skip the ">65536 && misaligned" head-alignment block: it only
        // affects load alignment (performance), never the result.

        let mut off = 0usize;
        let mut remaining = data.len();

        // chunk cap = MIN(2*VL*(INT16_MAX/UINT8_MAX), MAX_CHUNK_LEN) & ~(2*VL-1)
        // INT16_MAX/UINT8_MAX = 32767/255 = 128 ; 2*VL*128 = 4096.
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

                // s2 += s1 * (n & ~(2*VL-1))
                s2 = s2.wrapping_add(s1.wrapping_mul((n & !(2 * VL - 1)) as u32));

                while n >= 2 * VL {
                    let p = data.as_ptr().add(off);
                    let data_a = _mm_loadu_si128(p as *const __m128i);
                    let data_b = _mm_loadu_si128(p.add(VL) as *const __m128i);

                    v_s1_sums = _mm_add_epi32(v_s1_sums, v_s1);
                    v_byte_sums_a = _mm_add_epi16(v_byte_sums_a, _mm_unpacklo_epi8(data_a, zeroes));
                    v_byte_sums_b = _mm_add_epi16(v_byte_sums_b, _mm_unpackhi_epi8(data_a, zeroes));
                    v_byte_sums_c = _mm_add_epi16(v_byte_sums_c, _mm_unpacklo_epi8(data_b, zeroes));
                    v_byte_sums_d = _mm_add_epi16(v_byte_sums_d, _mm_unpackhi_epi8(data_b, zeroes));
                    v_s1 = _mm_add_epi32(
                        v_s1,
                        _mm_add_epi32(_mm_sad_epu8(data_a, zeroes), _mm_sad_epu8(data_b, zeroes)),
                    );
                    off += 2 * VL;
                    n -= 2 * VL;
                }

                // v_s2 = (2*VL)*v_s1_sums + sum_k mults_k * v_byte_sums_k
                // LOG2_VL = 4 ⇒ shift by LOG2_VL+1 = 5.
                let v_s2 = _mm_add_epi32(
                    _mm_slli_epi32(v_s1_sums, 5),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_madd_epi16(v_byte_sums_a, mults_a),
                            _mm_madd_epi16(v_byte_sums_b, mults_b),
                        ),
                        _mm_add_epi32(
                            _mm_madd_epi16(v_byte_sums_c, mults_c),
                            _mm_madd_epi16(v_byte_sums_d, mults_d),
                        ),
                    ),
                );
                reduce_to_32bits(v_s1, v_s2, &mut s1, &mut s2);
            }

            // Scalar tail for the last 0 <= n < 2*VL bytes of this chunk. This
            // mirrors `ADLER32_CHUNK(s1, s2, p, n)`, which also reduces s1,s2
            // mod DIVISOR at its end (unconditionally, even for n == 0).
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
    fn sse2_matches_scalar_varied() {
        if !std::is_x86_feature_detected!("sse2") {
            eprintln!("skip: sse2 not available");
            return;
        }
        let mut x: u32 = 0xDEAD_BEEF;
        let mut buf = vec![0u8; 9000]; // > MAX_CHUNK_LEN to hit the chunk loop
        for b in buf.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x & 0xFF) as u8;
        }
        for &init in &[1u32, 0, 0xFFFF_FFFF, 0x0001_0000] {
            for len in 0..=buf.len() {
                let data = &buf[..len];
                // SAFETY: sse2 presence checked above.
                let simd = unsafe { super::adler32_sse2(init, data) };
                assert_eq!(simd, adler32_generic(init, data), "len={len} init={init:#x}");
            }
        }
    }
}
