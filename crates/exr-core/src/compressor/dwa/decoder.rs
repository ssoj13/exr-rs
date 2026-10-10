//! DWA lossy DCT decoder — port of `LossyDctDecoder_execute` (and the CSC
//! variant) from `OpenEXRCore/internal_dwa_decoder.h`.
//!
//! Reverses the encoder: for each block, read the DC, un-RLE the AC, inverse
//! zigzag, inverse DCT (with the row-skip variant chosen from `last_non_zero`),
//! and for CSC groups inverse-transform Y′CbCr → RGB.  Constant (DC-only)
//! blocks take the fast `inverse_8x8_dc_only` path.  Optionally maps results
//! back through the linear LUT.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_decoder.h>

use imath_rs::Half;

use super::csc::{csc709_inverse_64_3, csc709_inverse_single_3};
use super::dct::{inverse_8x8, inverse_8x8_dc_only, BLOCK_ELEMENTS, BLOCK_SIZE};
use super::tables::to_linear;
use super::zigzag::{from_zigzag, rle_decode_ac};
use crate::probe::{self, Probe};

/// Lossy DCT decoder configured for one chunk (width × height pixel grid).
pub struct LossyDctDecoder {
    /// Pixel width of the region.
    width: usize,
    /// Pixel height of the region.
    height: usize,
}

impl LossyDctDecoder {
    /// Build a decoder for the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Number of 8×8 blocks horizontally.
    #[inline]
    pub fn blocks_x(&self) -> usize {
        self.width.div_ceil(BLOCK_SIZE)
    }

    /// Number of 8×8 blocks vertically.
    #[inline]
    pub fn blocks_y(&self) -> usize {
        self.height.div_ceil(BLOCK_SIZE)
    }

    /// Decode a single (non-CSC) channel into native half pixels, writing into
    /// caller-supplied, reused buffers instead of allocating a fresh `output`/
    /// `row_blocks` every call (the encode-side mirror is
    /// `LossyDctEncoder::encode_channel_into`).
    ///
    /// `output` is [`crate::buf::ensure_len`]'d to `width * height` and then
    /// FULLY overwritten -- every `(y, x)` cell in `[0, height) x [0, width)`
    /// is written exactly once by `unblock_row` across all `by` iterations
    /// (the partial-last-row/column cases are handled by `max_y`/`max_x`
    /// inside `unblock_row`, not by skipping cells), so a stale value left
    /// over from a PREVIOUS, differently-sized chunk in the grown tail is
    /// never observed. `row_blocks` is similarly `ensure_len`'d to
    /// `blocks_x * BLOCK_ELEMENTS` and fully overwritten per block before
    /// `unblock_row` reads it.
    ///
    /// `ac` is the chunk's SHARED AC-token iterator: the decoder pulls exactly
    /// `blocks_x * blocks_y` blocks' worth of RLE tokens from it (in block raster
    /// order) and leaves the iterator positioned at the next channel's tokens.
    /// Consuming the shared stream directly avoids the old decode-then-re-encode
    /// `collect_block_ac` round-trip (the tokens / decoded blocks are identical).
    pub fn decode_channel_into(
        &self,
        dc_data: &[u16],
        ac: &mut impl Iterator<Item = u16>,
        use_linear: bool,
        output: &mut Vec<u16>,
        row_blocks: &mut Vec<u16>,
    ) -> Result<(), &'static str> {
        let nbx = self.blocks_x();
        let nby = self.blocks_y();
        if dc_data.len() < nbx * nby {
            return Err("Insufficient DC data");
        }

        crate::buf::ensure_len(output, self.width * self.height);
        crate::buf::ensure_len(row_blocks, nbx * BLOCK_ELEMENTS);

        let mut dc_iter = dc_data.iter().copied();

        for by in 0..nby {
            let max_y = if by == nby - 1 {
                self.height - by * BLOCK_SIZE
            } else {
                BLOCK_SIZE
            };

            for bx in 0..nbx {
                let mut half_zig = [0u16; BLOCK_ELEMENTS];
                let mut dct = [0.0f32; BLOCK_ELEMENTS];

                half_zig[0] = dc_iter.next().ok_or("Missing DC component")?;
                let last_non_zero =
                    rle_decode_ac(ac, &mut half_zig).map_err(|_| "Corrupt AC data")?;
                inverse_block(&half_zig, last_non_zero, &mut dct);

                let block_base = bx * BLOCK_ELEMENTS;
                for i in 0..BLOCK_ELEMENTS {
                    row_blocks[block_base + i] = Half::from_f32(dct[i]).bits();
                }
            }

            self.unblock_row(row_blocks, by, max_y, use_linear, output);
        }

        Ok(())
    }

    /// Decode an interleaved CSC group into native half R/G/B pixels, writing
    /// into caller-supplied, reused buffers (same rationale as [`Self::decode_channel_into`]: `r_out`/`g_out`/`b_out`
    /// and `row_blocks` are each `ensure_len`'d then FULLY overwritten by
    /// `unblock_row`, so growing them in place is sound).
    ///
    /// `dc_data` is plane-grouped (Y, Cb, Cr); `ac` is the chunk's SHARED AC-token
    /// iterator, read per block as (Y, Cb, Cr).  The decoder pulls `3 * blocks_x *
    /// blocks_y` blocks' worth of tokens and leaves the iterator at the next
    /// group/channel — same consumption order as the old `collect_block_ac` pass,
    /// minus its redundant decode→re-encode round-trip.
    #[allow(clippy::too_many_arguments)] // 1:1 with the encode side's *_into shape plus the 3-way RGB split
    pub fn decode_rgb_into(
        &self,
        dc_data: &[u16],
        ac: &mut impl Iterator<Item = u16>,
        use_linear: bool,
        r_out: &mut Vec<u16>,
        g_out: &mut Vec<u16>,
        b_out: &mut Vec<u16>,
        row_blocks: &mut [Vec<u16>; 3],
    ) -> Result<(), &'static str> {
        let nbx = self.blocks_x();
        let nby = self.blocks_y();
        let total = nbx * nby;
        if dc_data.len() < total * 3 {
            return Err("Insufficient DC data for RGB");
        }

        crate::buf::ensure_len(r_out, self.width * self.height);
        crate::buf::ensure_len(g_out, self.width * self.height);
        crate::buf::ensure_len(b_out, self.width * self.height);
        for rb in row_blocks.iter_mut() {
            crate::buf::ensure_len(rb, nbx * BLOCK_ELEMENTS);
        }

        let dc_y = &dc_data[0..total];
        let dc_cb = &dc_data[total..2 * total];
        let dc_cr = &dc_data[2 * total..3 * total];

        for by in 0..nby {
            let max_y = if by == nby - 1 {
                self.height - by * BLOCK_SIZE
            } else {
                BLOCK_SIZE
            };

            for bx in 0..nbx {
                let block_idx = by * nbx + bx;
                let dc_vals = [dc_y[block_idx], dc_cb[block_idx], dc_cr[block_idx]];

                let mut dct: [[f32; BLOCK_ELEMENTS]; 3] =
                    [[0.0; BLOCK_ELEMENTS], [0.0; BLOCK_ELEMENTS], [0.0; BLOCK_ELEMENTS]];
                let mut block_is_constant = true;

                for chan in 0..3 {
                    let mut half_zig = [0u16; BLOCK_ELEMENTS];
                    half_zig[0] = dc_vals[chan];

                    let last_non_zero =
                        rle_decode_ac(ac, &mut half_zig).map_err(|_| "Corrupt AC data")?;

                    if !inverse_block(&half_zig, last_non_zero, &mut dct[chan]) {
                        block_is_constant = false;
                    }
                }

                if !block_is_constant {
                    csc709_inverse_64_3(&mut dct);
                } else {
                    csc709_inverse_single_3(&mut dct);
                    // Replicate the single converted RGB value across the block.
                    let (r, g, b) = (dct[0][0], dct[1][0], dct[2][0]);
                    dct[0][1..].fill(r);
                    dct[1][1..].fill(g);
                    dct[2][1..].fill(b);
                }

                let block_base = bx * BLOCK_ELEMENTS;
                for chan in 0..3 {
                    for i in 0..BLOCK_ELEMENTS {
                        row_blocks[chan][block_base + i] = Half::from_f32(dct[chan][i]).bits();
                    }
                }
            }

            self.unblock_row(&row_blocks[0], by, max_y, use_linear, r_out);
            self.unblock_row(&row_blocks[1], by, max_y, use_linear, g_out);
            self.unblock_row(&row_blocks[2], by, max_y, use_linear, b_out);
        }

        Ok(())
    }

    /// Scatter one row of decoded 8×8 blocks into `output`, applying the linear
    /// LUT when requested — shared by the single-channel and CSC paths.
    fn unblock_row(
        &self,
        row_blocks: &[u16],
        by: usize,
        max_y: usize,
        use_linear: bool,
        output: &mut [u16],
    ) {
        let nbx = self.blocks_x();
        let num_full_blocks = self.width / BLOCK_SIZE;

        for y in 0..max_y {
            let out_y = by * BLOCK_SIZE + y;
            if out_y >= self.height {
                break;
            }
            let row_base = out_y * self.width;

            for bx in 0..num_full_blocks {
                let src_base = bx * BLOCK_ELEMENTS + y * BLOCK_SIZE;
                let dst_base = row_base + bx * BLOCK_SIZE;
                for x in 0..BLOCK_SIZE {
                    let v = row_blocks[src_base + x];
                    output[dst_base + x] = if use_linear { to_linear(v) } else { v };
                }
            }

            if num_full_blocks < nbx {
                let max_x = self.width - num_full_blocks * BLOCK_SIZE;
                let src_base = num_full_blocks * BLOCK_ELEMENTS + y * BLOCK_SIZE;
                let dst_base = row_base + num_full_blocks * BLOCK_SIZE;
                for x in 0..max_x {
                    let v = row_blocks[src_base + x];
                    output[dst_base + x] = if use_linear { to_linear(v) } else { v };
                }
            }
        }
    }
}

/// Map the AC `last_non_zero` index to the number of trailing IDCT rows that
/// are zero — port of the `dctInverse8x8_N` dispatch thresholds.
/// Inverse-transform one block from its zigzag half coefficients into `dct`
/// (which must be zeroed), picking the DC-only IDCT when no AC coefficient is
/// set and the row-skipping full IDCT otherwise — the single dispatch point
/// for both the per-channel and the CSC decode paths, as in C++
/// `internal_dwa_decoder.h`. Returns `true` when the block was DC-only.
#[inline]
fn inverse_block(
    half_zig: &[u16; BLOCK_ELEMENTS],
    last_non_zero: usize,
    dct: &mut [f32; BLOCK_ELEMENTS],
) -> bool {
    if last_non_zero == 0 {
        probe::hit(Probe::DwaDcOnlyBlock);
        dct[0] = Half::from_bits(half_zig[0]).to_f32();
        inverse_8x8_dc_only(dct);
        true
    } else {
        probe::hit(Probe::DwaFullIdctBlock);
        from_zigzag(half_zig, dct);
        inverse_8x8(dct, zeroed_rows_from_last_nonzero(last_non_zero));
        false
    }
}

#[inline]
pub(crate) fn zeroed_rows_from_last_nonzero(last_non_zero: usize) -> usize {
    if last_non_zero < 2 {
        7
    } else if last_non_zero < 3 {
        6
    } else if last_non_zero < 9 {
        5
    } else if last_non_zero < 10 {
        4
    } else if last_non_zero < 20 {
        3
    } else if last_non_zero < 21 {
        2
    } else if last_non_zero < 35 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_counts() {
        let d = LossyDctDecoder::new(100, 50);
        assert_eq!(d.blocks_x(), 13);
        assert_eq!(d.blocks_y(), 7);
    }

    #[test]
    fn dc_only_block_decodes_uniform() {
        let d = LossyDctDecoder::new(8, 8);
        let dc = vec![Half::from_f32(8.0).bits()];
        let ac = [0xff00u16]; // EOB
        let mut out = Vec::new();
        let mut row_blocks = Vec::new();
        d.decode_channel_into(&dc, &mut ac.iter().copied(), false, &mut out, &mut row_blocks)
            .unwrap();
        assert_eq!(out.len(), 64);
        let first = Half::from_bits(out[0]).to_f32();
        for &v in &out {
            assert!((Half::from_bits(v).to_f32() - first).abs() < 0.01);
        }
    }

    /// Row-skip dispatch must equal C++'s `rowStartIdx`/`rowsEmpty` tables
    /// (`internal_dwa_decoder.h`, v3.4.15) for every `last_non_zero`.
    #[test]
    fn zeroed_rows_match_cpp_tables() {
        const ROW_START_IDX: [usize; 7] = [2, 3, 9, 10, 20, 21, 35];
        const ROWS_EMPTY: [usize; 7] = [7, 6, 5, 4, 3, 2, 1];
        for lnz in 0..BLOCK_ELEMENTS {
            let want = ROW_START_IDX
                .iter()
                .position(|&start| lnz < start)
                .map_or(0, |i| ROWS_EMPTY[i]);
            assert_eq!(zeroed_rows_from_last_nonzero(lnz), want, "last_non_zero={lnz}");
        }
    }

    /// The skipped rows really are zero: every zigzag position `0..=lnz`
    /// lands in a natural row below `8 - zeroed_rows`. Guards against a
    /// threshold that skips a row still holding a coefficient.
    #[test]
    fn zeroed_rows_only_skips_empty_rows() {
        use super::super::zigzag::ZIGZAG;
        for lnz in 0..BLOCK_ELEMENTS {
            let live_rows = 8 - zeroed_rows_from_last_nonzero(lnz);
            let max_row = ZIGZAG[..=lnz].iter().map(|&n| n / 8).max().unwrap_or(0);
            assert!(
                max_row < live_rows,
                "last_non_zero={lnz}: coefficient in row {max_row}, only {live_rows} rows transformed"
            );
        }
    }

    /// Behavioural: with nonzeros only at zigzag positions `0..=lnz`, the
    /// row-skipping IDCT is bit-identical to the full IDCT.
    #[test]
    fn row_skip_idct_matches_full_idct() {
        use super::super::zigzag::ZIGZAG;
        let mut seed = 0x9e37_79b9u32;
        for lnz in 0..BLOCK_ELEMENTS {
            for _ in 0..8 {
                let mut block = [0f32; BLOCK_ELEMENTS];
                for &natural in &ZIGZAG[..=lnz] {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    block[natural] = (seed % 2001) as f32 - 1000.0;
                }
                let mut full = block;
                inverse_8x8(&mut full, 0);
                let mut skipped = block;
                inverse_8x8(&mut skipped, zeroed_rows_from_last_nonzero(lnz));
                assert!(
                    full.iter().zip(&skipped).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "last_non_zero={lnz}: row-skip IDCT diverges from full IDCT"
                );
            }
        }
    }

    #[test]
    fn insufficient_dc_errors() {
        let d = LossyDctDecoder::new(16, 16);
        let dc = vec![0u16; 2]; // need 4
        let mut out = Vec::new();
        let mut row_blocks = Vec::new();
        assert!(d
            .decode_channel_into(&dc, &mut std::iter::empty(), false, &mut out, &mut row_blocks)
            .is_err());
    }
}
