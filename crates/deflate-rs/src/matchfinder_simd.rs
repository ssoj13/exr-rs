//! x86 SIMD tiers for `matchfinder_init` / `matchfinder_rebase`.
//!
//! Port of libdeflate `lib/x86/matchfinder_impl.h` (AVX2 and SSE2 variants).
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/x86/matchfinder_impl.h>
//!
//! `matchfinder_init` fills a position table with `MATCHFINDER_INITVAL`
//! (`_mm*_set1_epi16(INITVAL)`); `matchfinder_rebase` adds `-WINDOW_SIZE` to
//! every entry with **signed saturation** (`_mm*_adds_epi16`). Both are purely
//! elementwise, so they are byte-identical to the scalar paths in
//! `matchfinder_common.rs`, which remain the source of truth and fallback.
//!
//! # Alignment / size note
//! libdeflate guarantees its tables are 32-byte aligned and sized to a multiple
//! of 4 vectors (`MATCHFINDER_MEM_ALIGNMENT` / `MATCHFINDER_SIZE_ALIGNMENT`).
//! In safe Rust the table is a `Vec<i16>` with no such guarantees, so we use
//! **unaligned** loads/stores and fold the `< 1 vector` remainder with the
//! scalar helpers. The numeric result is identical (elementwise op).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::matchfinder_common::{MfPos, MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE};

/// `(u16)-MATCHFINDER_WINDOW_SIZE` as the i16 addend for rebase.
/// For WINDOW_SIZE == 32768 this is `0x8000` == `i16::MIN` == `-32768`.
const REBASE_ADDEND: i16 = (MATCHFINDER_WINDOW_SIZE as u16).wrapping_neg() as i16;

// ===========================================================================
// AVX2 (16 × i16 per vector)
// ===========================================================================

/// AVX2 `matchfinder_init`. Mirrors `matchfinder_init_avx2`.
///
/// # Safety
/// Requires `avx2`. `data` is a valid `&mut [MfPos]`; loads/stores are bounded
/// by `data.len()`. Reached only via `cpu::matchfinder_tier() == Avx2`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn matchfinder_init_avx2(data: &mut [MfPos]) {
    const LANES: usize = 16;
    // SAFETY: feature-gated; the chunk loop stores exactly 16 i16 per step and
    // never writes past a full `[MfPos; LANES]` chunk. Remainder is scalar.
    unsafe {
        let v = _mm256_set1_epi16(MATCHFINDER_INITVAL);
        let (chunks, remainder) = data.as_chunks_mut::<LANES>();
        for c in chunks {
            _mm256_storeu_si256(c.as_mut_ptr() as *mut __m256i, v);
        }
        for slot in remainder {
            *slot = MATCHFINDER_INITVAL;
        }
    }
}

/// AVX2 `matchfinder_rebase`. Mirrors `matchfinder_rebase_avx2` (PADDSW).
///
/// # Safety
/// Requires `avx2`. Same bounds reasoning as `matchfinder_init_avx2`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn matchfinder_rebase_avx2(data: &mut [MfPos]) {
    const LANES: usize = 16;
    // SAFETY: feature-gated; loads then stores 16 i16 per step within bounds.
    unsafe {
        let v = _mm256_set1_epi16(REBASE_ADDEND);
        let (chunks, remainder) = data.as_chunks_mut::<LANES>();
        for c in chunks {
            let p = c.as_mut_ptr() as *mut __m256i;
            let x = _mm256_loadu_si256(p as *const __m256i);
            _mm256_storeu_si256(p, _mm256_adds_epi16(x, v));
        }
        rebase_scalar_remainder(remainder);
    }
}

// ===========================================================================
// SSE2 (8 × i16 per vector)
// ===========================================================================

/// SSE2 `matchfinder_init`. Mirrors `matchfinder_init_sse2`.
///
/// # Safety
/// Requires `sse2`. Bounds as above.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn matchfinder_init_sse2(data: &mut [MfPos]) {
    const LANES: usize = 8;
    // SAFETY: feature-gated; stores 8 i16 per step within bounds.
    unsafe {
        let v = _mm_set1_epi16(MATCHFINDER_INITVAL);
        let (chunks, remainder) = data.as_chunks_mut::<LANES>();
        for c in chunks {
            _mm_storeu_si128(c.as_mut_ptr() as *mut __m128i, v);
        }
        for slot in remainder {
            *slot = MATCHFINDER_INITVAL;
        }
    }
}

/// SSE2 `matchfinder_rebase`. Mirrors `matchfinder_rebase_sse2` (PADDSW).
///
/// # Safety
/// Requires `sse2`. Bounds as above.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn matchfinder_rebase_sse2(data: &mut [MfPos]) {
    const LANES: usize = 8;
    // SAFETY: feature-gated; loads then stores 8 i16 per step within bounds.
    unsafe {
        let v = _mm_set1_epi16(REBASE_ADDEND);
        let (chunks, remainder) = data.as_chunks_mut::<LANES>();
        for c in chunks {
            let p = c.as_mut_ptr() as *mut __m128i;
            let x = _mm_loadu_si128(p as *const __m128i);
            _mm_storeu_si128(p, _mm_adds_epi16(x, v));
        }
        rebase_scalar_remainder(remainder);
    }
}

/// Apply the scalar branchless saturating-subtract-by-WINDOW_SIZE to a short
/// remainder slice, identical to `matchfinder_common::matchfinder_rebase`.
#[inline]
fn rebase_scalar_remainder(rem: &mut [MfPos]) {
    for v in rem.iter_mut() {
        let x = *v;
        let mask = x >> 15;
        *v = (0x8000u16 as i16) | (x & !mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matchfinder_common::{matchfinder_rebase_scalar, rebase_sample as sample};

    #[test]
    fn sse2_init_and_rebase_match_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            eprintln!("skip: sse2 not available");
            return;
        }
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 64, 100] {
            // init
            let mut a = vec![0i16; len];
            let mut b = vec![0i16; len];
            // SAFETY: sse2 checked.
            unsafe { matchfinder_init_sse2(&mut a) };
            b.fill(MATCHFINDER_INITVAL);
            assert_eq!(a, b, "init len={len}");
            // rebase
            let mut a = sample(len);
            let mut b = a.clone();
            // SAFETY: sse2 checked.
            unsafe { matchfinder_rebase_sse2(&mut a) };
            matchfinder_rebase_scalar(&mut b);
            assert_eq!(a, b, "rebase len={len}");
        }
    }

    #[test]
    fn avx2_init_and_rebase_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skip: avx2 not available");
            return;
        }
        for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 48, 100, 257] {
            let mut a = vec![0i16; len];
            let mut b = vec![0i16; len];
            // SAFETY: avx2 checked.
            unsafe { matchfinder_init_avx2(&mut a) };
            b.fill(MATCHFINDER_INITVAL);
            assert_eq!(a, b, "init len={len}");
            let mut a = sample(len);
            let mut b = a.clone();
            // SAFETY: avx2 checked.
            unsafe { matchfinder_rebase_avx2(&mut a) };
            matchfinder_rebase_scalar(&mut b);
            assert_eq!(a, b, "rebase len={len}");
        }
    }
}
