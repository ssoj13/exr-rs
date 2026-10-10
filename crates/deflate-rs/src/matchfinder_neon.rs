//! ARM NEON tiers for `matchfinder_init` / `matchfinder_rebase`.
//!
//! Port of libdeflate `lib/arm/matchfinder_impl.h` (NEON variants).
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/arm/matchfinder_impl.h>
//!
//! `vdupq_n_s16(INITVAL)` for init and `vqaddq_s16` (signed-saturating add of
//! `-WINDOW_SIZE`) for rebase — elementwise, hence byte-identical to the scalar
//! source of truth in `matchfinder_common.rs`.
//!
//! C selects these at compile time whenever `HAVE_NEON_NATIVE` (every aarch64
//! build); `cpu::matchfinder_tier` does the same. The tests below compare them
//! with the scalar path and run on the aarch64 CI runner (macos-aarch64).

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use crate::matchfinder_common::{
    MfPos, MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE, matchfinder_rebase_scalar,
};

/// 8 x i16 per 128-bit register.
const LANES: usize = 8;

/// `(u16)-MATCHFINDER_WINDOW_SIZE` as the i16 addend (0x8000 for 32 KiB window).
const REBASE_ADDEND: i16 = (MATCHFINDER_WINDOW_SIZE as u16).wrapping_neg() as i16;

/// NEON `matchfinder_init`. Mirrors `matchfinder_init_neon`.
///
/// # Safety
/// Requires the `neon` target feature (baseline on aarch64).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn matchfinder_init_neon(data: &mut [MfPos]) {
    let (lanes, rest) = data.as_chunks_mut::<LANES>();
    // SAFETY: feature-gated; each store writes exactly one `[i16; 8]` chunk.
    unsafe {
        let v = vdupq_n_s16(MATCHFINDER_INITVAL);
        for c in lanes {
            vst1q_s16(c.as_mut_ptr(), v);
        }
    }
    rest.fill(MATCHFINDER_INITVAL);
}

/// NEON `matchfinder_rebase`. Mirrors `matchfinder_rebase_neon`.
///
/// # Safety
/// Requires `neon` (baseline on aarch64).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn matchfinder_rebase_neon(data: &mut [MfPos]) {
    let (lanes, rest) = data.as_chunks_mut::<LANES>();
    // SAFETY: feature-gated; each load/store covers exactly one `[i16; 8]` chunk.
    unsafe {
        let v = vdupq_n_s16(REBASE_ADDEND);
        for c in lanes {
            let p = c.as_mut_ptr();
            vst1q_s16(p, vqaddq_s16(vld1q_s16(p), v));
        }
    }
    // C sizes the table to whole registers; the scalar form covers any other length.
    matchfinder_rebase_scalar(rest);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{MatchfinderTier, matchfinder_tier};
    use crate::matchfinder_common::rebase_sample;

    #[test]
    fn neon_init_and_rebase_match_scalar() {
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 64, 100, 257] {
            let mut a = vec![0i16; len];
            // SAFETY: NEON is a baseline feature of every aarch64 target.
            unsafe { matchfinder_init_neon(&mut a) };
            assert_eq!(a, vec![MATCHFINDER_INITVAL; len], "init len={len}");
            let mut a = rebase_sample(len);
            let mut b = a.clone();
            // SAFETY: as above.
            unsafe { matchfinder_rebase_neon(&mut a) };
            matchfinder_rebase_scalar(&mut b);
            assert_eq!(a, b, "rebase len={len}");
        }
    }

    /// The dispatcher really reaches this tier: identical output alone would also
    /// pass on a silent scalar fallback.
    #[test]
    fn dispatcher_selects_neon_unless_capped() {
        let capped = std::env::var("DEFLATE_RS_SIMD")
            .is_ok_and(|v| v.trim().eq_ignore_ascii_case("scalar"));
        let want = if capped { MatchfinderTier::Scalar } else { MatchfinderTier::Neon };
        assert_eq!(matchfinder_tier(), want);
    }
}
