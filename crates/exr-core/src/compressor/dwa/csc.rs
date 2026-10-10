//! Rec.709 RGB ↔ Y′CbCr color-space conversion for DWA — scalar port of
//! `csc709Forward64` / `csc709Inverse64` / `csc709Inverse` from
//! `OpenEXRCore/internal_dwa_simd.h`.
//!
//! DWA decorrelates RGB triplets into luminance + chroma before the DCT so the
//! chroma planes (which the eye is less sensitive to) can be quantised harder.
//! The coefficients are the standard Rec.709 matrices.  There is no SIMD
//! path (exr-core has no `unsafe`); the scalar math is the source of truth
//! and byte-identical to the C++ SIMD path's output.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_simd.h>

use super::dct::BLOCK_ELEMENTS;

/// Forward CSC: RGB → Y′CbCr (Rec.709), three 64-element blocks in place.
///
/// `comp0/comp1/comp2` arrive as R/G/B and leave as Y′/Cb/Cr.
pub fn csc709_forward_64(
    comp0: &mut [f32; BLOCK_ELEMENTS],
    comp1: &mut [f32; BLOCK_ELEMENTS],
    comp2: &mut [f32; BLOCK_ELEMENTS],
) {
    for i in 0..BLOCK_ELEMENTS {
        let r = comp0[i];
        let g = comp1[i];
        let b = comp2[i];

        // Y'  =  0.2126 R + 0.7152 G + 0.0722 B
        comp0[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        // Cb  = -0.1146 R - 0.3854 G + 0.5000 B
        comp1[i] = -0.1146 * r - 0.3854 * g + 0.5000 * b;
        // Cr  =  0.5000 R - 0.4542 G - 0.0458 B
        comp2[i] = 0.5000 * r - 0.4542 * g - 0.0458 * b;
    }
}

/// Inverse CSC: Y′CbCr → RGB (Rec.709), over a `[[f32; 64]; 3]` in place.
///
/// Used for the common non-constant block path (`block_is_constant == false`).
/// `data[0/1/2]` arrive as Y′/Cb/Cr and leave as R/G/B.
pub fn csc709_inverse_64_3(data: &mut [[f32; BLOCK_ELEMENTS]; 3]) {
    // Split into three disjoint mutable channel planes so we can iterate by
    // value rather than re-indexing `data[c][i]` (each plane is independent).
    let (y_plane, rest) = data.split_at_mut(1);
    let (cb_plane, cr_plane) = rest.split_at_mut(1);
    for ((y, cb), cr) in y_plane[0]
        .iter_mut()
        .zip(cb_plane[0].iter_mut())
        .zip(cr_plane[0].iter_mut())
    {
        let (sy, scb, scr) = (*y, *cb, *cr);
        // R = Y' + 1.5747 Cr
        *y = sy + 1.5747 * scr;
        // G = Y' - 0.1873 Cb - 0.4682 Cr
        *cb = sy - 0.1873 * scb - 0.4682 * scr;
        // B = Y' + 1.8556 Cb
        *cr = sy + 1.8556 * scb;
    }
}

/// Inverse CSC for a single (DC-only) pixel of a `[[f32; 64]; 3]` block.
///
/// Constant blocks only need element 0 converted (the rest are copied from it
/// by the decoder), so this avoids transforming all 64 entries.
pub fn csc709_inverse_single_3(data: &mut [[f32; BLOCK_ELEMENTS]; 3]) {
    let y = data[0][0];
    let cb = data[1][0];
    let cr = data[2][0];

    data[0][0] = y + 1.5747 * cr;
    data[1][0] = y - 0.1873 * cb - 0.4682 * cr;
    data[2][0] = y + 1.8556 * cb;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_roundtrips() {
        let mut r = [1.0f32; 64];
        let mut g = [1.0f32; 64];
        let mut b = [1.0f32; 64];
        csc709_forward_64(&mut r, &mut g, &mut b);
        // White → Y'≈1, Cb≈0, Cr≈0.
        assert!((r[0] - 1.0).abs() < 0.01);
        assert!(g[0].abs() < 0.01);
        assert!(b[0].abs() < 0.01);

        let mut data = [r, g, b];
        csc709_inverse_64_3(&mut data);
        assert!((data[0][0] - 1.0).abs() < 0.01);
        assert!((data[1][0] - 1.0).abs() < 0.01);
        assert!((data[2][0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn single_inverse_matches_full_inverse_at_index0() {
        let y = [0.5f32; 64];
        let cb = [0.1f32; 64];
        let cr = [-0.2f32; 64];
        let mut full = [y, cb, cr];
        let mut single = [y, cb, cr];
        csc709_inverse_64_3(&mut full);
        csc709_inverse_single_3(&mut single);
        for c in 0..3 {
            assert!((full[c][0] - single[c][0]).abs() < 1e-6);
        }
    }
}
