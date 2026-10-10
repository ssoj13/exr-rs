//! DWA perceptual encoding lookup tables — port of `dwaLookups.cpp`
//! (`generateToLinear` / `generateToNonlinear`) and the precomputed
//! `OpenEXRCore/dwaLookups.h` arrays `dwaCompressorToLinear` /
//! `dwaCompressorToNonlinear`.
//!
//! # What these tables do (1:1 with C++)
//!
//! DWA encodes color channels in a *perceptual* (nonlinear) space before the
//! DCT.  For values `<= 1.0` it uses a gamma-2.2 curve; for `> 1.0` it uses a
//! log curve that blends smoothly into the gamma at 1.0 (gamma blows up above
//! 1, log over-/under-samples below).  The forward map is:
//!
//! ```text
//! nonlinear(linear) = linear^(1/2.2)                          , |linear| <= 1
//!                   = ln(linear)/ln(e^2.2) + 1                 , otherwise
//! ```
//!
//! and `to_linear` is its inverse (`linear^2.2` / `(e^2.2)^(linear-1)`).
//!
//! # The XDR subtlety (why these are NOT plain `f(x)` tables)
//!
//! In C++ the tables are indexed by the **XDR (on-disk, big-endian) half bit
//! pattern**, and `dwaCompressorToLinear` *outputs* an XDR half while
//! `dwaCompressorToNonlinear` *takes* an XDR input and outputs a NATIVE half
//! (the XDR↔NATIVE byte flip is folded into the table — see the
//! `Xdr::write`/`Xdr::read` calls in `dwaLookups.cpp`).
//!
//! Our codec pipeline works entirely in **native little-endian half** (the
//! line buffer is already byte-swapped to native by the seam, and our DCT
//! encoder/decoder read/write native `u16`).  So we must *undo* the XDR folding
//! the C++ generator baked in.  Concretely, for a 16-bit value the XDR form is
//! the byte-swap of the native form (`swap_bytes`).  We therefore build tables
//! that map **native half → native half**:
//!
//! * `to_nonlinear[native]`: C++ computes `toNonlinear[xdr]` where `xdr` is the
//!   XDR encoding of a native value, and the *output* is already native.  So we
//!   evaluate the C++ formula on the native value directly (no swaps).
//! * `to_linear[native]`: C++ computes `toLinear[native]` and stores the XDR of
//!   the result.  We want a native→native table, so we evaluate the formula and
//!   keep the result native (no output swap).
//!
//! The net effect: in *native* space both tables are simply the analytic
//! forward/inverse perceptual maps evaluated through the bit-exact Imath
//! `f32 -> half` rounding ([`imath_rs::Half::from_f32`]), with NaN/Inf mapped to
//! 0.  Because `imath_rs::Half` is byte-exact to C++ `half`, and the `pow`/`log`
//! math is evaluated in `f32`/`f64` exactly as the generator does, the produced
//! half bit patterns match `dwaLookups.h` entry-for-entry.
//!
//! # 3.4 rewrite: single-precision `powf`, not the v3.3.11 double-precision path
//!
//! The v3.3.11 `IlmImf/dwaLookups.cpp` generator (our original port target) computed
//! the `> 1.0` branch in `double`: `pow(logBase, |h|-1)` with `logBase =
//! pow(2.7182818, 2.2)` evaluated in `double`, and the nonlinear branch as
//! `log(|h|)/log(logBase) + 1.0` (also `double`). OpenEXRCore 3.4 (our parity
//! oracle, `internal_dwa_table_init.c::dwa_convertToLinear` /
//! `dwa_convertToNonLinear`) rewrote this to plain `float`: the log-branch uses a
//! precomputed **float** constant `9.02501329156f` (`= pow(2.7182818, 2.2)` cast to
//! `f32`, not `f64`) with `powf` (`f32`), and the nonlinear log-branch divides by
//! the literal `2.2f` directly instead of `log(logBase)` (`ln(f64::powf(e,2.2))`
//! is not bit-exact to `2.2f64` due to the round-trip through `pow`/`ln`). Both
//! rewrites changed rounding for values with `|h| > 1.0`, which the enlarged
//! multi-chunk parity image (unlike the old 7×5 image) actually
//! exercises — the double-precision version silently diverged from the real
//! 3.4.15 oracle at isolated pixels. We must match the ACTUAL oracle, not
//! the historical IlmImf source, so this file now ports the 3.4 table-init math.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_dwa_table_init.c>

use imath_rs::Half;
use std::sync::OnceLock;

/// Lazily-built 64K-entry perceptual conversion tables.
static TABLES: OnceLock<DwaTables> = OnceLock::new();

/// The two 65536-entry half→half lookup tables (native bit patterns).
struct DwaTables {
    /// nonlinear (perceptual) → linear; mirrors `dwaCompressorToLinear`.
    to_linear: Box<[u16; 65536]>,
    /// linear → nonlinear (perceptual); mirrors `dwaCompressorToNonlinear`.
    to_nonlinear: Box<[u16; 65536]>,
}

/// Force the perceptual tables to be built (idempotent).  Called at the top of
/// `compress`/`decompress` so the first hot-path lookup never races the init.
pub fn ensure_tables_initialized() {
    TABLES.get_or_init(init_tables);
}

/// Map a native half bit pattern from nonlinear (perceptual) to linear space.
///
/// Mirrors a single `dwaCompressorToLinear[x]` lookup (in native space).
#[inline]
pub fn to_linear(x: u16) -> u16 {
    let tables = TABLES.get_or_init(init_tables);
    tables.to_linear[x as usize]
}

/// Map a native half bit pattern from linear to nonlinear (perceptual) space.
///
/// Mirrors a single `dwaCompressorToNonlinear[x]` lookup (in native space).
#[inline]
pub fn to_nonlinear(x: u16) -> u16 {
    let tables = TABLES.get_or_init(init_tables);
    tables.to_nonlinear[x as usize]
}

/// Build both 64K tables once.
fn init_tables() -> DwaTables {
    let mut to_linear = Box::new([0u16; 65536]);
    let mut to_nonlinear = Box::new([0u16; 65536]);

    // C++ generator special-cases index 0 → 0 (and the NaN/Inf guard below
    // covers the exponent==0x1F band).
    for i in 0..65536usize {
        to_linear[i] = convert_to_linear(i as u16);
        to_nonlinear[i] = convert_to_nonlinear(i as u16);
    }

    DwaTables {
        to_linear,
        to_nonlinear,
    }
}

/// `pow(2.7182818, 2.2)` cast to `f32` — the exact float literal
/// `dwa_convertToLinear` uses (`internal_dwa_table_init.c:60`: `px =
/// 9.02501329156f; // = pow(2.7182818, 2.2)`). This is a LITERAL float constant
/// in the C++ source, not a value recomputed at runtime, so we spell it the same
/// way (all 12 significant digits, matching the C literal exactly): both
/// `clippy::approx_constant` (which would want `E`, a different `f64`) and
/// `clippy::excessive_precision` (which would want the value truncated to what
/// `f32` can hold) would shift or obscure the exact source literal, so both are
/// deliberately allowed.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const LOG_BASE: f32 = 9.02501329156;

/// Port of `dwa_convertToLinear`'s body for one entry (native in/out).
///
/// `nonlinear → linear`:  `|h| <= 1 → sign·|h|^2.2` ; else `sign·LOG_BASE^(|h|-1)`.
/// NaN/Inf (exponent all-ones) and zero map to 0, exactly as C++. Everything is
/// single-precision `f32` — the 3.4 OpenEXRCore rewrite of the v3.3.11
/// `IlmImf/dwaLookups.cpp` double-precision generator (see module doc).
fn convert_to_linear(x: u16) -> u16 {
    if x == 0 {
        return 0;
    }
    // map NaN and inf to 0 (C++: `if ((i & 0x7c00) == 0x7c00)`).
    if (x & 0x7c00) == 0x7c00 {
        return 0;
    }

    let h = Half::from_bits(x).to_f32();
    let sign = if h < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let abs = h.abs();

    let (px, py) = if abs <= 1.0 { (abs, 2.2_f32) } else { (LOG_BASE, abs - 1.0) };
    let z = px.powf(py);

    Half::from_f32(sign * z).bits()
}

/// Port of `dwa_convertToNonLinear`'s body for one entry (native in/out).
///
/// `linear → nonlinear`:  `|h| <= 1 → sign·|h|^(1/2.2)` ; else
/// `sign·(ln(|h|)/2.2 + 1)`.  NaN/Inf and zero map to 0.  Single-precision `f32`
/// throughout, dividing by the literal `2.2f` (NOT `ln(LOG_BASE)`, which would
/// round differently — see module doc).
fn convert_to_nonlinear(x: u16) -> u16 {
    if x == 0 {
        return 0;
    }
    // map NaN and inf to 0.
    if (x & 0x7c00) == 0x7c00 {
        return 0;
    }

    let h = Half::from_bits(x).to_f32();
    let sign = if h < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let abs = h.abs();

    let z: f32 = if abs <= 1.0 { abs.powf(1.0_f32 / 2.2_f32) } else { abs.ln() / 2.2_f32 + 1.0 };

    Half::from_f32(sign * z).bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(to_linear(0), 0);
        assert_eq!(to_nonlinear(0), 0);
    }

    #[test]
    fn nan_inf_map_to_zero() {
        // +inf = 0x7c00, -inf = 0xfc00, a NaN = 0x7e00.
        for &b in &[0x7c00u16, 0xfc00u16, 0x7e00u16] {
            assert_eq!(to_linear(b), 0);
            assert_eq!(to_nonlinear(b), 0);
        }
    }

    #[test]
    fn roundtrip_near_identity_at_one() {
        // 1.0 -> nonlinear -> linear should be ~1.0 (perceptual fixed point).
        let one = Half::from_f32(1.0).bits();
        let nl = to_nonlinear(one);
        let back = to_linear(nl);
        let r = Half::from_bits(back).to_f32();
        assert!((r - 1.0).abs() < 0.01, "1.0 roundtrip got {r}");
    }

    #[test]
    fn nonlinear_is_monotonic_for_positive_normals() {
        // The perceptual encode must be monotonically non-decreasing across the
        // positive normal range (a sanity check on the curve).
        let mut prev = 0u16;
        for i in 1..0x7c00u16 {
            if (i & 0x7c00) == 0 {
                continue; // skip denormals
            }
            let v = to_nonlinear(i);
            if v != 0 {
                assert!(v >= prev, "non-monotonic at {i}: {v} < {prev}");
                prev = v;
            }
        }
    }
}
