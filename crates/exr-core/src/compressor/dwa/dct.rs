//! 8×8 DCT for DWA — scalar port of `dctForward8x8` / `dctInverse8x8_N` /
//! `dctInverse8x8DcOnly` from `OpenEXRCore/internal_dwa_simd.h`.
//!
//! Forward maps spatial → frequency; inverse maps back.  Both operate in place
//! on a 64-element `f32` block stored row-major.
//!
//! # This ports the SSE2 body, not the `#ifndef IMF_HAVE_SSE2` scalar body
//!
//! `internal_dwa_simd.h` `#define`s `IMF_HAVE_SSE2` unconditionally on
//! `_MSC_VER && (_M_IX86 || _M_X64)` (`:21`), so an MSVC x64 build — our parity
//! oracle — **always** compiles the SSE2 bodies of `dctForward8x8` /
//! `dctInverse8x8_scalar` (the latter name is misleading: the SSE2 build never
//! defines the `#ifndef IMF_HAVE_SSE2` scalar version at all) and the SSE2
//! `dctInverse8x8_N` dispatch table, which `initializeFuncs` picks whenever
//! `check_for_x86_simd` reports `sse2` (true on any x86-64 CPU) and NOT `avx`
//! (false unless the CPU/build target Haswell-or-newer — the oracle's build
//! host apparently doesn't). SSE2 is mathematically equivalent to the "scalar"
//! formulation this file used to port (same DCT, same coefficients) but
//! groups the additions differently — since float addition is not
//! associative, that changes the last bit of some results (confirmed against
//! the source below). Every
//! grouping and pass order here matches the SSE2 body exactly, not the
//! textbook/scalar butterfly a hand-derivation would produce.
//!
//! Forward (`:2109-2118`, comment: "we're always doing a column-wise
//! operation plus transposes"): the same butterfly runs on **columns first**,
//! then (via two `transpose` calls, which cancel) on **rows second** — the
//! reverse of the row-then-column order a plain 2D separable DCT suggests.
//! The per-axis butterfly itself (`a0..a7`, `k0/k1`, `rot_x/rot_y`, the six
//! output stages) is IDENTICAL between the two passes (SSE2 literally reuses
//! the same code on the transposed matrix), so both `forward_8x8` passes below
//! already used one shared expression shape — only the ORDER needed to flip.
//!
//! Inverse (`:1083-1310`): the row pass (`DCT_INVERSE_8x8_SS2_ROW_LOOP`) is a
//! matrix-vector product done by broadcasting each input value across a
//! `__m128` and multiplying by a column of the butterfly matrix, then summing
//! the four broadcasts left-to-right (`evenSum`/`oddSum`) — a DIFFERENT
//! summation grouping than the row pass's `alpha`/`beta` factoring a
//! hand-written scalar port would use. The column pass (`:1265-1310`) keeps
//! the `alpha`/`theta`/`gamma` factoring but sums each `beta` as two paired
//! sub-sums added together (`(A+B)+(C+D)`), not the four-term left fold
//! (`((A+B)+C)+D`) the row-pass-style factoring would give — also not
//! associative-safe, so also ported exactly.
//!
//! Port target (pinned tag v3.4.15 — the actual oracle we cross-check against;
//! v3.3.11 IlmImf predates the OpenEXRCore SIMD rewrite but has the same
//! coefficients under `#ifndef IMF_HAVE_SSE2`, which nothing here compiles):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_dwa_simd.h>

/// DCT block edge length.
pub const BLOCK_SIZE: usize = 8;

/// DCT block element count (8×8).
pub const BLOCK_ELEMENTS: usize = BLOCK_SIZE * BLOCK_SIZE;

// Forward DCT butterfly factors.  These are the EXACT float literals from the C++
// `dctForward8x8` (`internal_dwa_simd.h`): `c4 = cos(pi*4/16)` and the
// pre-multiplied half-cosines `cN/2`.  They MUST be byte-for-byte these values —
// recomputing `0.5 * cos(...)` at run time rounds differently and diverges from
// C++ by 1 ULP in some quantized coefficients (observed as a tiled-DWAA
// pixel-parity miss at the block boundary).
//
// These literals carry more digits than an `f32` can represent, but they are the
// EXACT bytes from the C++ source; `excessive_precision`/`approx_constant` are
// suppressed because truncating to the clippy-suggested shorter form would round
// to a different `f32` and break byte-identical DCT parity vs C++.
//
// `C4` is `cos(pi*4/16)` (~`FRAC_1_SQRT_2`).
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const C4: f32 = 0.70710678; // sqrt(2)/2
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const C4_NEG: f32 = -0.70710678;

#[allow(clippy::excessive_precision)]
const C1_HALF: f32 = 0.490392640;
#[allow(clippy::excessive_precision)]
const C2_HALF: f32 = 0.461939770;
#[allow(clippy::excessive_precision)]
const C3_HALF: f32 = 0.415734810;
#[allow(clippy::excessive_precision)]
const C5_HALF: f32 = 0.277785120;
#[allow(clippy::excessive_precision)]
const C6_HALF: f32 = 0.191341720;
#[allow(clippy::excessive_precision)]
const C7_HALF: f32 = 0.097545161;

/// Forward 8×8 DCT, in place — port of the SSE2 `dctForward8x8`
/// (`internal_dwa_simd.h:2109`).
///
/// **Column pass first, then row pass** — the reverse of the textbook
/// row-then-column order (see module doc: SSE2 transforms columns, transposes
/// twice — which cancels — around a second application of the SAME butterfly,
/// net effect column-then-row). Both passes run the identical butterfly
/// (SSE2 literally reuses the code on the transposed matrix), so only the
/// ORDER differs from a naive port; the per-axis math was already correct.
#[inline]
pub fn forward_8x8(data: &mut [f32; BLOCK_ELEMENTS]) {
    // Column pass (strided) — runs FIRST.
    for col in 0..8 {
        let a0 = data[col] + data[56 + col];
        let a7 = data[col] - data[56 + col];
        let a1 = data[8 + col] + data[16 + col];
        let a2 = data[8 + col] - data[16 + col];
        let a3 = data[24 + col] + data[32 + col];
        let a4 = data[24 + col] - data[32 + col];
        let a5 = data[40 + col] + data[48 + col];
        let a6 = data[40 + col] - data[48 + col];

        let k0 = C4 * (a0 + a3);
        let k1 = C4 * (a1 + a5);

        data[col] = 0.5 * (k0 + k1);
        data[32 + col] = 0.5 * (k0 - k1);

        let rot_x = a2 - a6;
        let rot_y = a0 - a3;

        data[16 + col] = C6_HALF * rot_x + C2_HALF * rot_y;
        data[48 + col] = C6_HALF * rot_y - C2_HALF * rot_x;

        let k0 = C4 * (a1 - a5);
        let k1 = C4_NEG * (a2 + a6);

        let rot_x = a7 - k0;
        let rot_y = a4 + k1;

        data[24 + col] = C3_HALF * rot_x - C5_HALF * rot_y;
        data[40 + col] = C5_HALF * rot_x + C3_HALF * rot_y;

        let rot_x = a7 + k0;
        let rot_y = k1 - a4;

        data[8 + col] = C1_HALF * rot_x - C7_HALF * rot_y;
        data[56 + col] = C7_HALF * rot_x + C1_HALF * rot_y;
    }

    // Row pass — runs SECOND, on the column pass's output. Same butterfly.
    for row in 0..8 {
        let base = row * 8;

        let a0 = data[base] + data[base + 7];
        let a1 = data[base + 1] + data[base + 2];
        let a2 = data[base + 1] - data[base + 2];
        let a3 = data[base + 3] + data[base + 4];
        let a4 = data[base + 3] - data[base + 4];
        let a5 = data[base + 5] + data[base + 6];
        let a6 = data[base + 5] - data[base + 6];
        let a7 = data[base] - data[base + 7];

        let k0 = C4 * (a0 + a3);
        let k1 = C4 * (a1 + a5);

        data[base] = 0.5 * (k0 + k1);
        data[base + 4] = 0.5 * (k0 - k1);

        let rot_x = a2 - a6;
        let rot_y = a0 - a3;

        data[base + 2] = C6_HALF * rot_x + C2_HALF * rot_y;
        data[base + 6] = C6_HALF * rot_y - C2_HALF * rot_x;

        let k0 = C4 * (a1 - a5);
        let k1 = C4_NEG * (a2 + a6);

        let rot_x = a7 - k0;
        let rot_y = a4 + k1;

        data[base + 3] = C3_HALF * rot_x - C5_HALF * rot_y;
        data[base + 5] = C5_HALF * rot_x + C3_HALF * rot_y;

        let rot_x = a7 + k0;
        let rot_y = k1 - a4;

        data[base + 1] = C1_HALF * rot_x - C7_HALF * rot_y;
        data[base + 7] = C7_HALF * rot_x + C1_HALF * rot_y;
    }
}

/// IDCT butterfly factors `0.5 * cos(k·pi/16)`, as the LITERAL `f32` constants
/// the SSE2 `dctInverse8x8_sse2` body hard-codes (`internal_dwa_simd.h:1086-1092`:
/// `{3.535536e-01f, ...}` etc.), not `.5f * cosf(...)` recomputed at runtime —
/// the scalar (`#ifndef IMF_HAVE_SSE2`) body computes them at runtime, but
/// nothing compiles that body on our MSVC x64 oracle (see module doc). Using
/// the exact decimal literals sidesteps any question of whether `cosf` and
/// Rust's `f32::cos` round identically: both toolchains parse the same
/// decimal token to the same nearest `f32` by construction.
#[allow(clippy::excessive_precision)]
const IA: f32 = 3.535536e-01;
#[allow(clippy::excessive_precision)]
const IB: f32 = 4.903927e-01;
#[allow(clippy::excessive_precision)]
const IC: f32 = 4.619398e-01;
#[allow(clippy::excessive_precision)]
const ID: f32 = 4.157349e-01;
#[allow(clippy::excessive_precision)]
const IE: f32 = 2.777855e-01;
#[allow(clippy::excessive_precision)]
const IF: f32 = 1.913422e-01;
#[allow(clippy::excessive_precision)]
const IG: f32 = 9.754573e-02;

/// Inverse 8×8 DCT, in place, skipping the last `zeroed_rows` input rows —
/// port of the SSE2 `dctInverse8x8_sse2` (`internal_dwa_simd.h:1083-1312`),
/// dispatched per `zeroed_rows` exactly like C++'s `dctInverse8x8_N` family.
///
/// When the decoder knows the trailing rows are all zero (from the AC RLE's
/// `last_non_zero`), it passes that count to skip needless row work — exactly
/// the C++ dispatch in `internal_dwa_decoder.h`.
#[inline]
pub fn inverse_8x8(data: &mut [f32; BLOCK_ELEMENTS], zeroed_rows: usize) {
    let (a, b, c, d, e, f, g) = (IA, IB, IC, ID, IE, IF, IG);

    // Row pass — only the non-zeroed rows need transforming. SSE2 computes
    // this as a matrix-vector product: each input is broadcast across a
    // 4-lane register and multiplied by a column of the butterfly matrix,
    // then the four broadcasts are summed LEFT TO RIGHT per lane
    // (`evenSum`/`oddSum` in `DCT_INVERSE_8x8_SS2_ROW_LOOP`) — a different,
    // NOT associativity-equivalent grouping from the `alpha`/`beta` factoring
    // the column pass below uses. `even0..3`/`odd0..3` below are exactly
    // those four `__m128` lanes, unpacked to scalar.
    for row in 0..(8 - zeroed_rows) {
        let base = row * 8;
        let v0 = data[base];
        let v1 = data[base + 1];
        let v2 = data[base + 2];
        let v3 = data[base + 3];
        let v4 = data[base + 4];
        let v5 = data[base + 5];
        let v6 = data[base + 6];
        let v7 = data[base + 7];

        let even0 = ((a * v0 + c * v2) + a * v4) + f * v6;
        let even1 = ((a * v0 + f * v2) - a * v4) - c * v6;
        let even2 = ((a * v0 - f * v2) - a * v4) + c * v6;
        let even3 = ((a * v0 - c * v2) + a * v4) - f * v6;

        let odd0 = ((b * v1 + d * v3) + e * v5) + g * v7;
        let odd1 = ((d * v1 - g * v3) - b * v5) - e * v7;
        let odd2 = ((e * v1 - b * v3) + g * v5) + d * v7;
        let odd3 = ((g * v1 - e * v3) + d * v5) - b * v7;

        data[base] = even0 + odd0;
        data[base + 1] = even1 + odd1;
        data[base + 2] = even2 + odd2;
        data[base + 3] = even3 + odd3;
        data[base + 4] = even3 - odd3;
        data[base + 5] = even2 - odd2;
        data[base + 6] = even1 - odd1;
        data[base + 7] = even0 - odd0;
    }

    // Column pass. `alpha`/`theta`/`gamma` match the row-factored form, but
    // SSE2 sums each `beta` as two PAIRED sub-sums added together
    // (`(A+B)+(C+D)`, `internal_dwa_simd.h:1265-1280`), not the four-term
    // left fold (`((A+B)+C)+D`) — also not associativity-equivalent, so also
    // ported exactly rather than re-derived.
    for col in 0..8 {
        let alpha0 = c * data[16 + col];
        let alpha1 = f * data[16 + col];
        let alpha2 = c * data[48 + col];
        let alpha3 = f * data[48 + col];

        let beta0 = (b * data[8 + col] + d * data[24 + col])
            + (e * data[40 + col] + g * data[56 + col]);
        let beta1 = (d * data[8 + col] - g * data[24 + col])
            - (b * data[40 + col] + e * data[56 + col]);
        let beta2 = (e * data[8 + col] - b * data[24 + col])
            + (g * data[40 + col] + d * data[56 + col]);
        let beta3 = (g * data[8 + col] - e * data[24 + col])
            + (d * data[40 + col] - b * data[56 + col]);

        let theta0 = a * (data[col] + data[32 + col]);
        let theta3 = a * (data[col] - data[32 + col]);

        let theta1 = alpha0 + alpha3;
        let theta2 = alpha1 - alpha2;

        let gamma0 = theta0 + theta1;
        let gamma1 = theta3 + theta2;
        let gamma2 = theta3 - theta2;
        let gamma3 = theta0 - theta1;

        data[col] = gamma0 + beta0;
        data[8 + col] = gamma1 + beta1;
        data[16 + col] = gamma2 + beta2;
        data[24 + col] = gamma3 + beta3;
        data[32 + col] = gamma3 - beta3;
        data[40 + col] = gamma2 - beta2;
        data[48 + col] = gamma1 - beta1;
        data[56 + col] = gamma0 - beta0;
    }
}

/// Inverse DCT for a DC-only block — port of `dctInverse8x8DcOnly`.
///
/// Fills all 64 elements with `data[0] * (1/sqrt(8))^2 == data[0] / 8`.
#[inline]
pub fn inverse_8x8_dc_only(data: &mut [f32; BLOCK_ELEMENTS]) {
    let val = data[0] * 0.3535536 * 0.3535536;
    data.fill(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_block_roundtrips() {
        let mut data = [1.0f32; 64];
        forward_8x8(&mut data);
        assert!(data[0].abs() > 0.1, "DC should be non-zero");
        for v in &data[1..] {
            assert!(v.abs() < 1e-5, "AC should vanish for a constant block");
        }
        inverse_8x8(&mut data, 0);
        for v in data.iter() {
            assert!((v - 1.0).abs() < 1e-4, "reconstruct should be ~1.0, got {v}");
        }
    }

    #[test]
    fn gradient_roundtrips() {
        let mut data = [0.0f32; 64];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32 / 63.0;
        }
        let original = data;
        forward_8x8(&mut data);
        inverse_8x8(&mut data, 0);
        for i in 0..64 {
            assert!((data[i] - original[i]).abs() < 1e-4, "idx {i}");
        }
    }

    #[test]
    fn inverse_dc_only_matches_constant() {
        let mut data = [0.0f32; 64];
        data[0] = 8.0;
        inverse_8x8_dc_only(&mut data);
        let expected = 8.0 * 0.3535536 * 0.3535536;
        for v in data.iter() {
            assert!((v - expected).abs() < 1e-5);
        }
    }
}
