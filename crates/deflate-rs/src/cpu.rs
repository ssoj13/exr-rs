//! Runtime CPU-feature dispatch for the SIMD acceleration tiers.
//!
//! This is the SAFE selector that decides, at runtime, which SIMD tier (if any)
//! to use for `crc32`, `adler32`, and the matchfinder init/rebase helpers. It
//! mirrors libdeflate's `arch_select_*_func` dispatch (see `x86/crc32_impl.h`,
//! `x86/adler32_impl.h`, `x86/matchfinder_impl.h`) but expressed as a small Rust
//! enum chosen with `std::is_x86_feature_detected!` (on aarch64 NEON is baseline and is
//! selected unconditionally, as C's `HAVE_NEON_NATIVE` does).
//!
//! # Why a separate module
//! The scalar path remains the source of truth (see `crc32.rs`, `adler32.rs`,
//! `matchfinder_common.rs`). The SIMD tiers each live in their own module behind
//! `#[target_feature]` and are reached ONLY through the dispatchers here, which
//! fall back to scalar when the feature is unavailable. No `unsafe` lives here.
//!
//! # Env override (mirrors jph-rs's `OJPH_CPU_*` pattern)
//! `DEFLATE_RS_SIMD` caps the maximum tier, for the compare-simd test and for
//! debugging. Recognised values (case-insensitive):
//! - `scalar` — force the portable scalar path (no SIMD at all)
//! - `sse2`   — allow at most SSE2 tiers
//! - `avx2`   — allow at most AVX2 tiers
//! - `pclmul` — allow at most PCLMULQDQ tiers (same cap as the default auto)
//! - unset / anything else — auto-detect the best available tier
//!
//! The override only ever *lowers* the selected tier below what the CPU supports;
//! it can never select a tier the CPU lacks (that would be unsound).

use std::sync::OnceLock;

/// The maximum SIMD tier the env override permits.  Ordered from least to most
/// capable so a numeric comparison expresses "at most this tier".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum SimdCap {
    /// Force the scalar path; no SIMD.
    Scalar,
    /// Permit SSE2-class tiers (and scalar).
    Sse2,
    /// Permit AVX2-class tiers (and below).
    Avx2,
    /// Permit PCLMULQDQ-class tiers (and below). This is also the default cap
    /// when no override is set, since pclmul is the most capable tier ported.
    Pclmul,
}

/// Parse `DEFLATE_RS_SIMD` once.  Absent/unknown ⇒ the highest cap (`Pclmul`),
/// i.e. full auto-detection unrestricted by the override.
fn cap() -> SimdCap {
    static CAP: OnceLock<SimdCap> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("DEFLATE_RS_SIMD") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "scalar" => SimdCap::Scalar,
            "sse2" => SimdCap::Sse2,
            "avx2" => SimdCap::Avx2,
            "pclmul" | "pclmulqdq" => SimdCap::Pclmul,
            _ => SimdCap::Pclmul,
        },
        Err(_) => SimdCap::Pclmul,
    })
}

// ---------------------------------------------------------------------------
// CRC-32 tier selection
// ---------------------------------------------------------------------------

/// Which CRC-32 implementation the dispatcher should use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Crc32Tier {
    /// Portable slice-by-8 scalar path (`crc32.rs`).
    Scalar,
    /// PCLMULQDQ+SSE4.1 carryless-multiply folding (`crc32_pclmul.rs`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Pclmul,
}

/// Pick the CRC-32 tier for this run.  Mirrors `arch_select_crc32_func`, capped
/// to the single PCLMULQDQ tier we port (the AVX2/AVX512 variants in libdeflate
/// are perf-only refinements that produce the identical checksum).
#[inline]
pub(crate) fn crc32_tier() -> Crc32Tier {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cap() >= SimdCap::Pclmul
            && std::is_x86_feature_detected!("pclmulqdq")
            && std::is_x86_feature_detected!("sse4.1")
        {
            return Crc32Tier::Pclmul;
        }
    }
    Crc32Tier::Scalar
}

// ---------------------------------------------------------------------------
// Adler-32 tier selection
// ---------------------------------------------------------------------------

/// Which Adler-32 implementation the dispatcher should use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Adler32Tier {
    /// Portable scalar path (`adler32.rs`).
    Scalar,
    /// SSE2 vectorized (`adler32_sse2.rs`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    /// AVX2 vectorized (`adler32_avx2.rs`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
}

/// Pick the Adler-32 tier for this run.  Mirrors `arch_select_adler32_func`
/// (USE_VNNI=0 variants only — the AVX2/SSE2-compatible algorithm).
#[inline]
pub(crate) fn adler32_tier() -> Adler32Tier {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cap() >= SimdCap::Avx2 && std::is_x86_feature_detected!("avx2") {
            return Adler32Tier::Avx2;
        }
        if cap() >= SimdCap::Sse2 && std::is_x86_feature_detected!("sse2") {
            return Adler32Tier::Sse2;
        }
    }
    Adler32Tier::Scalar
}

// ---------------------------------------------------------------------------
// Matchfinder tier selection
// ---------------------------------------------------------------------------

/// Which matchfinder init/rebase implementation the dispatcher should use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MatchfinderTier {
    /// Portable scalar fill / branchless saturating slide (`matchfinder_common.rs`).
    Scalar,
    /// SSE2 (`_mm_set1_epi16` / `_mm_adds_epi16`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    /// AVX2 (`_mm256_set1_epi16` / `_mm256_adds_epi16`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// ARM NEON (`vdupq_n_s16` / `vqaddq_s16`).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Pick the matchfinder tier for this run.  Mirrors the `#ifdef` cascade in
/// `x86/matchfinder_impl.h` (AVX2 preferred, else SSE2, else scalar) and
/// `arm/matchfinder_impl.h` (NEON whenever `HAVE_NEON_NATIVE`).
#[inline]
pub(crate) fn matchfinder_tier() -> MatchfinderTier {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cap() >= SimdCap::Avx2 && std::is_x86_feature_detected!("avx2") {
            return MatchfinderTier::Avx2;
        }
        if cap() >= SimdCap::Sse2 && std::is_x86_feature_detected!("sse2") {
            return MatchfinderTier::Sse2;
        }
    }
    // NEON is a baseline feature of every aarch64 target, so C selects it at compile
    // time with no runtime check; only `DEFLATE_RS_SIMD=scalar` turns it off.
    #[cfg(target_arch = "aarch64")]
    if cap() > SimdCap::Scalar {
        return MatchfinderTier::Neon;
    }
    MatchfinderTier::Scalar
}

// ---------------------------------------------------------------------------
// Decompressor BMI2 tier selection
// ---------------------------------------------------------------------------

/// Whether the decompressor should use the BMI2-recompiled instantiation.
/// Mirrors `x86/decompress_impl.h`'s `arch_select_decompress_func`.
///
/// Unlike the checksum/matchfinder tiers above, BMI2 has no separate scalar
/// "implementation" of its own to choose between — it's the exact same safe
/// `deflate_decompress_impl`, just recompiled under BMI2 codegen (see
/// `deflate_decompress_bmi2`'s doc comment). So this returns a plain `bool`
/// rather than an enum: `true` selects the BMI2-recompiled path, `false`
/// falls back to the (always-correct) portable path.
#[inline]
pub(crate) fn bmi2_decompress_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cap() != SimdCap::Scalar && std::is_x86_feature_detected!("bmi2") {
            return true;
        }
    }
    false
}

/// Diagnostic: which tier every dispatcher in this module actually selects
/// on THIS machine, right now (honours the live `DEFLATE_RS_SIMD` cap). Not
/// used by any codec path -- exists so a benchmark or bug report can state
/// "the zlib path used adler32 tier X / crc32 tier Y / bmi2=Z" instead of
/// assuming it from reading the dispatch code (verify which SIMD variant
/// actually runs, don't infer it).
pub fn active_tiers_report() -> String {
    format!(
        "adler32={:?} crc32={:?} matchfinder={:?} bmi2_decompress={} (DEFLATE_RS_SIMD cap={:?})",
        adler32_tier(),
        crc32_tier(),
        matchfinder_tier(),
        bmi2_decompress_available(),
        cap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no `DEFLATE_RS_SIMD` override, the BMI2
    /// dispatcher's decision must reduce to EXACTLY the raw CPU feature
    /// check. Without this, a bug in `cap()`'s "unset -> highest cap"
    /// default, or in `bmi2_decompress_available`'s own gating, could
    /// silently disable (or wrongly enable) the BMI2 path on a real
    /// BMI2-capable machine with no test noticing -- every existing
    /// correctness test passes on EITHER path by design (BMI2 is the exact
    /// same safe `deflate_decompress_impl` body, just recompiled under BMI2
    /// codegen), so byte-identity alone proves nothing about which path
    /// actually ran.
    ///
    /// `DEFLATE_RS_SIMD`'s cap is cached in a process-wide `OnceLock`
    /// (`cap()`), read once from the environment on first use -- if the
    /// var happens to be set in THIS process' own environment (e.g. a CI
    /// job or a developer's shell exports it), this test's premise ("when
    /// unset") doesn't hold, so it skips rather than asserting something
    /// the override deliberately makes false.
    ///
    /// Mutation check: temporarily forced `bmi2_decompress_available` to
    /// always return `false` -- this test failed immediately on this
    /// (BMI2-capable) machine (`left: false, right: true`), then reverted.
    #[test]
    fn bmi2_decompress_available_matches_cpu_feature_when_unset() {
        if std::env::var_os("DEFLATE_RS_SIMD").is_some() {
            eprintln!(
                "skipping bmi2_decompress_available_matches_cpu_feature_when_unset: \
                 DEFLATE_RS_SIMD is set in this process' environment"
            );
            return;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert_eq!(
                bmi2_decompress_available(),
                std::is_x86_feature_detected!("bmi2"),
                "bmi2_decompress_available() must exactly track the raw CPU feature check \
                 when DEFLATE_RS_SIMD has no override"
            );
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            assert!(!bmi2_decompress_available(), "non-x86 targets have no BMI2 tier at all");
        }
    }
}
