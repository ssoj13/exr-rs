//! Zigzag reordering + AC run-length coding for DWA — port of the `remap` /
//! `inv_remap` tables and the `LossyDctEncoder_rleAc` / `LossyDctDecoder_unRleAc`
//! AC codecs from `OpenEXRCore/internal_dwa_encoder.h` / `internal_dwa_decoder.h`.
//!
//! Quantised DCT coefficients are reordered into JPEG zigzag order (low → high
//! frequency) so the trailing high-frequency zeros cluster and RLE well.  The
//! AC stream uses the half-float NaN space as run markers:
//! * `0xff00`        — end of block (all remaining AC are zero);
//! * `0xff00 | n`    — a run of `n` zeros (`1 <= n <= 0xff`);
//! * any other value — a literal coefficient.
//!
//! Port targets (pinned tag v3.3.11):
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_encoder.h>
//! * <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_decoder.h>

use imath_rs::Half;

use super::dct::BLOCK_ELEMENTS;

/// `inv_remap`: linear index → zigzag position (encode placement).
pub const INV_ZIGZAG: [usize; BLOCK_ELEMENTS] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// `remap`: zigzag position → linear index (decode placement).
pub const ZIGZAG: [usize; BLOCK_ELEMENTS] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// RLE end-of-block marker (the `0xff00` NaN-space symbol).
pub const RLE_EOB: u16 = 0xff00;

/// High-byte mask identifying an RLE run/EOB symbol.
pub const RLE_RUN_MASK: u16 = 0xff00;

/// Reorder a zigzag-ordered half block into a natural-order `f32` block —
/// port of the decoder's `fromHalfZigZag` placement (here followed by IDCT).
pub(crate) fn from_zigzag(src: &[u16; BLOCK_ELEMENTS], dst: &mut [f32; BLOCK_ELEMENTS]) {
    for (zig_pos, &natural_idx) in ZIGZAG.iter().enumerate() {
        dst[natural_idx] = Half::from_bits(src[zig_pos]).to_f32();
    }
}

/// RLE-encode the AC coefficients of a zigzag block into `output` — port of
/// `LossyDctEncoder_rleAc`.  Skips the DC at index 0.  Returns the number of
/// AC symbols written (the `_numAcComp` increment in C++).
pub(crate) fn rle_encode_ac(block: &[u16; BLOCK_ELEMENTS], output: &mut Vec<u16>) -> usize {
    let mut count = 0usize;
    let mut i = 1; // skip DC

    while i < BLOCK_ELEMENTS {
        if block[i] != 0 {
            output.push(block[i]);
            count += 1;
            i += 1;
            continue;
        }

        // Count the run of zeros starting at i.
        let mut run_len = 1usize;
        while i + run_len < BLOCK_ELEMENTS && block[i + run_len] == 0 {
            run_len += 1;
        }

        if run_len == 1 {
            // Lone zero → literal 0.
            output.push(0);
            count += 1;
        } else if i + run_len == BLOCK_ELEMENTS {
            // Run extends to end of block → EOB.
            output.push(RLE_EOB);
            count += 1;
        } else {
            // Interior run → 0xff | len.
            output.push(RLE_RUN_MASK | (run_len as u16));
            count += 1;
        }
        i += run_len;
    }

    count
}

/// RLE-decode AC coefficients from `input` into `block` (DC at index 0 left as
/// set by the caller) — port of `LossyDctDecoder_unRleAc`.
///
/// Returns `Ok(last_non_zero)`, the zigzag index of the last non-zero AC
/// coefficient (used to pick the row-skip IDCT variant), or `Err(())` if the
/// input ends mid-block (corrupt stream — the caller maps this to an error,
/// never a panic).
pub(crate) fn rle_decode_ac(
    input: &mut impl Iterator<Item = u16>,
    block: &mut [u16; BLOCK_ELEMENTS],
) -> Result<usize, ()> {
    // Zero the AC region (keep DC at index 0).
    for coef in block[1..].iter_mut() {
        *coef = 0;
    }

    let mut i = 1usize;
    let mut last_non_zero = 0usize;

    while i < BLOCK_ELEMENTS {
        let val = input.next().ok_or(())?;

        if (val & RLE_RUN_MASK) == RLE_RUN_MASK {
            let run = (val & 0xff) as usize;
            if run == 0 {
                // EOB: the rest of the block is already zero.
                break;
            }
            i += run;
        } else {
            // Literal coefficient (the bounds are guaranteed by `i < 64`).
            // C++ sets `lnz = dctComp` unconditionally for every literal,
            // including a literal 0 (`internal_dwa_decoder.h:153-155`) — the
            // encoder emits a literal 0 for a lone isolated zero (a run of
            // length 1; see `rle_encode_ac`'s `run_len == 1` branch), and that
            // position must still count as "last written", or `last_non_zero`
            // under-reports and the IDCT row-skip optimisation drops rows C++
            // does not skip.
            block[i] = val;
            last_non_zero = i;
            i += 1;
        }
    }

    Ok(last_non_zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_tables_are_mutual_inverses() {
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[2], 8);
        for i in 0..64 {
            assert_eq!(ZIGZAG[INV_ZIGZAG[i]], i);
        }
    }

    #[test]
    fn rle_all_zero_ac_is_single_eob() {
        let mut block = [0u16; 64];
        block[0] = 0x3c00; // DC = 1.0
        let mut out = Vec::new();
        let n = rle_encode_ac(&block, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0], RLE_EOB);
    }

    #[test]
    fn rle_roundtrip_sparse() {
        let mut original = [0u16; 64];
        original[0] = 0x3c00;
        original[1] = 0x3800;
        original[5] = 0x3400;
        let mut encoded = Vec::new();
        rle_encode_ac(&original, &mut encoded);

        let mut decoded = [0u16; 64];
        decoded[0] = original[0];
        let mut it = encoded.into_iter();
        let last = rle_decode_ac(&mut it, &mut decoded).unwrap();
        assert_eq!(last, 5);
        assert_eq!(decoded, original);
    }

    #[test]
    fn rle_decode_errors_on_truncation() {
        // A run header that claims more coefficients than the (empty) input has
        // — actually an empty stream mid-block is the corruption case here.
        let mut block = [0u16; 64];
        let mut it = std::iter::empty::<u16>();
        assert!(rle_decode_ac(&mut it, &mut block).is_err());
    }

    #[test]
    fn rle_run_then_value() {
        let mut block = [0u16; 64];
        block[0] = 0x3c00;
        block[1] = 0x3800;
        block[7] = 0x3400; // 5 zeros between idx 1 and 7
        let mut out = Vec::new();
        rle_encode_ac(&block, &mut out);
        assert_eq!(out[0], 0x3800);
        assert_eq!(out[1], RLE_RUN_MASK | 5);
        assert_eq!(out[2], 0x3400);
    }
}
