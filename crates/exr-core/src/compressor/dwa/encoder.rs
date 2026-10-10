//! DWA lossy DCT encoder — port of `LossyDctEncoder_execute` (and the CSC
//! variant) from `OpenEXRCore/internal_dwa_encoder.h`.
//!
//! Splits each channel into 8×8 blocks (mirroring at the right/bottom edges),
//! optionally perceptual-encodes via the nonlinear LUT, forward-DCTs, then
//! quantises and zigzags, emitting one DC coefficient per block and an RLE'd AC
//! stream.  RGB triplets are first CSC'd to Y′CbCr so chroma can be quantised
//! harder.
//!
//! DC layout for a CSC group is by plane: all Y DC, then all Cb, then all Cr.
//! AC for a CSC group is interleaved per block (Y, Cb, Cr) — matching the
//! decoder's read order.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_encoder.h>

use imath_rs::Half;

use super::csc::csc709_forward_64;
use super::dct::{forward_8x8, BLOCK_ELEMENTS, BLOCK_SIZE};
use super::quantize::{build_quant_table_cbcr, build_quant_table_y, quantize_and_zigzag};
use super::tables::to_nonlinear;
use super::zigzag::rle_encode_ac;
use super::DEFAULT_COMPRESSION_LEVEL;

/// Lossy DCT encoder configured for one chunk (width × height pixel grid).
pub struct LossyDctEncoder {
    /// Pixel width of the region.
    width: usize,
    /// Pixel height of the region.
    height: usize,
    /// Luminance quantisation tolerances `(f32, half)`.
    quant_table_y: ([f32; BLOCK_ELEMENTS], [u16; BLOCK_ELEMENTS]),
    /// Chrominance quantisation tolerances `(f32, half)`.
    quant_table_cbcr: ([f32; BLOCK_ELEMENTS], [u16; BLOCK_ELEMENTS]),
}

impl LossyDctEncoder {
    /// Build an encoder for the given dimensions and DWA level.
    ///
    /// `level` defaults to [`DEFAULT_COMPRESSION_LEVEL`]; the base quantisation
    /// error is `level / 100000` (the C++ `_dwaCompressionLevel / 100000.f`).
    pub fn new(width: usize, height: usize, level: Option<f32>) -> Self {
        let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
        let quant_base_error = (level / 100000.0).max(0.0);
        Self {
            width,
            height,
            quant_table_y: build_quant_table_y(quant_base_error),
            quant_table_cbcr: build_quant_table_cbcr(quant_base_error),
        }
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

    /// Encode a single (non-CSC) channel of native half data, APPENDING its
    /// `dc`/`ac` output onto caller-supplied buffers instead of allocating
    /// and returning a fresh `(Vec<u16>, Vec<u16>)` per call (every production
    /// call site (`compress_dwa`) wanted to `.extend()` its result into a
    /// shared stream immediately, so this writes straight into it instead).
    /// `dc` gets one entry per block (row-major); `ac` is the RLE'd AC
    /// stream for all blocks in order. Matches the `_into` naming convention
    /// already used elsewhere in this codec (e.g. `zlib.rs`'s
    /// `compress_into`/`uncompress_into`). Neither buffer is cleared here --
    /// the caller decides where in its own scratch this call's output
    /// should land (single-channel `dc` is already in final on-wire order,
    /// so it goes straight into `compress_dwa`'s `all_dc`; a CSC group's
    /// `dc` needs plane-grouping first, so `encode_rgb_into` writes to its
    /// own Y/Cb/Cr staging buffers instead -- see its doc).
    pub fn encode_channel_into(&self, data: &[u16], use_nonlinear: bool, dc_out: &mut Vec<u16>, ac_out: &mut Vec<u16>) {
        let nbx = self.blocks_x();
        let nby = self.blocks_y();

        let mut dct = [0.0f32; BLOCK_ELEMENTS];
        let mut half_zig = [0u16; BLOCK_ELEMENTS];

        for by in 0..nby {
            for bx in 0..nbx {
                self.extract_block(data, bx, by, use_nonlinear, &mut dct);
                forward_8x8(&mut dct);
                quantize_and_zigzag(&dct, &self.quant_table_y.0, &self.quant_table_y.1, &mut half_zig);
                dc_out.push(half_zig[0]);
                rle_encode_ac(&half_zig, ac_out);
            }
        }
    }

    /// Encode an RGB triplet as a CSC group (R/G/B → Y′CbCr), filling three
    /// caller-supplied Y/Cb/Cr DC staging buffers (each `.clear()`'d here
    /// first) plus APPENDING `ac` onto `ac_out` instead of returning an owned
    /// `(Vec<u16>, Vec<u16>)` per call; see `encode_channel_into`'s doc for why every production call site wants
    /// the `_into` shape instead). A CSC group's DC must be gathered
    /// plane-by-plane while the per-block loop below interleaves Y/Cb/Cr, so
    /// (unlike `encode_channel_into`'s `dc_out`) it cannot be written
    /// directly in final concatenation order — the caller concatenates
    /// `dc_y`+`dc_cb`+`dc_cr` in that order onto its own `all_dc` afterward,
    /// see `compress_dwa`'s call site. `ac` for a CSC group is interleaved
    /// per block (Y, Cb, Cr) to match the decoder.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rgb_into(
        &self,
        r_data: &[u16],
        g_data: &[u16],
        b_data: &[u16],
        use_nonlinear: bool,
        dc_y: &mut Vec<u16>,
        dc_cb: &mut Vec<u16>,
        dc_cr: &mut Vec<u16>,
        ac_out: &mut Vec<u16>,
    ) {
        let nbx = self.blocks_x();
        let nby = self.blocks_y();
        dc_y.clear();
        dc_cb.clear();
        dc_cr.clear();

        let mut dct_r = [0.0f32; BLOCK_ELEMENTS];
        let mut dct_g = [0.0f32; BLOCK_ELEMENTS];
        let mut dct_b = [0.0f32; BLOCK_ELEMENTS];
        let mut half_zig = [0u16; BLOCK_ELEMENTS];

        for by in 0..nby {
            for bx in 0..nbx {
                self.extract_block(r_data, bx, by, use_nonlinear, &mut dct_r);
                self.extract_block(g_data, bx, by, use_nonlinear, &mut dct_g);
                self.extract_block(b_data, bx, by, use_nonlinear, &mut dct_b);

                csc709_forward_64(&mut dct_r, &mut dct_g, &mut dct_b);

                for (chan_idx, dct) in [&mut dct_r, &mut dct_g, &mut dct_b].iter_mut().enumerate() {
                    forward_8x8(dct);
                    let (tols, htols) = if chan_idx == 0 {
                        &self.quant_table_y
                    } else {
                        &self.quant_table_cbcr
                    };
                    quantize_and_zigzag(dct, tols, htols, &mut half_zig);
                    match chan_idx {
                        0 => dc_y.push(half_zig[0]),
                        1 => dc_cb.push(half_zig[0]),
                        _ => dc_cr.push(half_zig[0]),
                    }
                    rle_encode_ac(&half_zig, ac_out);
                }
            }
        }
    }

    /// Extract an 8×8 block at `(bx, by)` into `output` (`f32`), mirroring at
    /// the right/bottom edges and optionally applying the nonlinear LUT.
    ///
    /// The single-reflection edge handling matches the C++ exactly:
    /// `if (vx >= width) vx = 2*width-1-vx; if (vx < 0) vx = width-1` (and y).
    fn extract_block(
        &self,
        data: &[u16],
        bx: usize,
        by: usize,
        use_nonlinear: bool,
        output: &mut [f32; BLOCK_ELEMENTS],
    ) {
        let w = self.width as i32;
        let h = self.height as i32;
        for y in 0..BLOCK_SIZE {
            for x in 0..BLOCK_SIZE {
                let mut vx = (BLOCK_SIZE * bx + x) as i32;
                let mut vy = (BLOCK_SIZE * by + y) as i32;

                if vx >= w {
                    vx = 2 * w - 1 - vx;
                }
                if vx < 0 {
                    vx = w - 1;
                }
                if vy >= h {
                    vy = 2 * h - 1 - vy;
                }
                if vy < 0 {
                    vy = h - 1;
                }

                let idx = vy as usize * self.width + vx as usize;
                let raw = data.get(idx).copied().unwrap_or(0);
                let h_bits = if use_nonlinear { to_nonlinear(raw) } else { raw };
                output[y * BLOCK_SIZE + x] = Half::from_bits(h_bits).to_f32();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_counts() {
        let e = LossyDctEncoder::new(100, 50, None);
        assert_eq!(e.blocks_x(), 13);
        assert_eq!(e.blocks_y(), 7);
    }

    #[test]
    fn single_channel_constant_has_dc_per_block() {
        let e = LossyDctEncoder::new(16, 16, None);
        let one = Half::from_f32(1.0).bits();
        let data = vec![one; 16 * 16];
        let (mut dc, mut ac) = (Vec::new(), Vec::new());
        e.encode_channel_into(&data, false, &mut dc, &mut ac);
        assert_eq!(dc.len(), 4);
        assert!(dc.iter().all(|&d| d != 0));
    }

    #[test]
    fn rgb_dc_has_three_planes() {
        let e = LossyDctEncoder::new(8, 8, None);
        let one = Half::from_f32(1.0).bits();
        let half = Half::from_f32(0.5).bits();
        let r = vec![one; 64];
        let g = vec![half; 64];
        let b = vec![half; 64];
        let (mut dc_y, mut dc_cb, mut dc_cr, mut ac) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        e.encode_rgb_into(&r, &g, &b, false, &mut dc_y, &mut dc_cb, &mut dc_cr, &mut ac);
        assert_eq!(dc_y.len() + dc_cb.len() + dc_cr.len(), 3); // 1 block × 3 channels
    }

    #[test]
    fn block_extraction_mirrors_small_image() {
        let e = LossyDctEncoder::new(4, 4, None);
        let data: Vec<u16> = (0..16).map(|i| Half::from_f32(i as f32).bits()).collect();
        let mut block = [0.0f32; 64];
        e.extract_block(&data, 0, 0, false, &mut block);
        for y in 0..4 {
            for x in 0..4 {
                let expected = (y * 4 + x) as f32;
                assert!((block[y * 8 + x] - expected).abs() < 0.01, "({x},{y})");
            }
        }
    }
}
