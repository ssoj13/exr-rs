//! Hardware `BC6H_SF16` (`Bc6hRgbFloat`) 4x4 block codec with optional bit-exact lossless rANS residual.
//!
//! # Architecture
//! - **Base layer (`16` bytes per 4x4 block):** Standard Direct3D 11 / Vulkan / WebGPU `BC6H_SF16` Mode 11
//!   (`0b00011`, single-region, 10-bit signed endpoints + 4-bit interpolation weights). Can be uploaded directly to
//!   a `wgpu::TextureFormat::Bc6hRgbFloat` texture for 0.0 ms hardware TMU sampling at 16K x 16K @ 60 fps, or decoded
//!   by the bit-identical CPU/WGSL reference decoder.
//! - **Residual layer (`Bc6hMode::LosslessResidual`):** Immediately follows the contiguous `nbx * nby * 16` hardware
//!   BC6H block array as a `TAG_RANS` (or `TAG_RAW` when incompressible) stream over the stride-2 byte-plane
//!   `split_delta` of `orig_u16.wrapping_sub(bc6h_u16)`. Adding the residual back yields **100% bit-exact** `F16`
//!   reconstruction across all 65,536 `u16` bit patterns (including negative values, subnormals, infinities, and NaNs),
//!   while `extract_bc6h_blocks` still slices the hardware BC6H base blocks in O(1) without touching the residual.

use exv_format::{BlobKind, DecodeError, TAG_RANS, TAG_RAW, classify_blob};
use rans::RansConfig;

use crate::{CodecError, reorder, with_bytes};

/// `BC6H` profile mode stored as a 1-byte `codec_params` (`[0]` or `[1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bc6hMode {
    /// Hardware `BC6H_SF16` 4x4 blocks only (lossy for F16, raw lossless for non-F16).
    Lossy,
    /// Hardware `BC6H_SF16` 4x4 base blocks + rANS-compressed 16-bit residual (100% bit-exact lossless).
    LosslessResidual,
}

impl Bc6hMode {
    /// On-disk mode byte (`0` = Lossy, `1` = LosslessResidual).
    pub const fn byte(self) -> u8 {
        match self {
            Self::Lossy => 0,
            Self::LosslessResidual => 1,
        }
    }

    /// Wire `codec_params` representation.
    pub fn params(self) -> Vec<u8> {
        vec![self.byte()]
    }

    /// Parse `codec_params` for `CodecId::BC6H`.
    pub fn from_params(params: &[u8]) -> Result<Self, &'static str> {
        match params {
            [0] => Ok(Self::Lossy),
            [1] => Ok(Self::LosslessResidual),
            _ => Err("expects [0] (lossy) or [1] (lossless residual)"),
        }
    }

    /// `true` iff this mode is bit-exact lossless (enables temporal `PRED` P-frames and B-frames).
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::LosslessResidual)
    }
}

/// Official BC6H 4-bit interpolation weights (`aWeight4` in the D3D11 / Khronos BC6H spec).
pub const WEIGHT4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// Convert an IEEE 754 `F16` bit pattern to the signed integer space `[-0x7FFF, 0x7FFF]` used by `BC6H_SF16`
/// before endpoint quantization. NaNs/Infinities clamp to the finite signed half dynamic range (`+-0x7BFF`) so
/// endpoint fitting never overflows.
#[inline]
pub fn f16_bits_to_sf16_domain(bits: u16) -> i32 {
    let sign = (bits & 0x8000) != 0;
    let mag = (bits & 0x7FFF).min(0x7BFF) as i32;
    // Inverse of finish_unquantize_sf16: `comp = (interp * 31) >> 5` => `interp ~= (mag << 5) / 31`.
    let unq = ((mag << 5) + 15) / 31;
    if sign { -unq } else { unq }
}

/// Quantize a signed domain value in `[-0x7FFF, 0x7FFF]` to a 10-bit signed `BC6H_SF16` Mode 11 endpoint
/// in `[-511, 511]`.
#[inline]
pub fn quantize_sf16_10(val: i32) -> i32 {
    if val == 0 {
        0
    } else if val >= 0x7FFF {
        511
    } else if val <= -0x7FFF {
        -511
    } else if val > 0 {
        ((val << 9) - 0x4000 + 16384) >> 15
    } else {
        -((((-val) << 9) - 0x4000 + 16384) >> 15)
    }
    .clamp(-511, 511)
}

/// Unquantize a 10-bit signed `BC6H_SF16` Mode 11 endpoint (`q` in `[-512, 511]`) to `[-0x7FFF, 0x7FFF]`
/// per the D3D11 / Khronos BC6H specification.
#[inline]
pub const fn unquantize_sf16_10(q: i32) -> i32 {
    if q == 0 {
        0
    } else if q < 0 {
        if q <= -512 {
            -0x7FFF
        } else {
            -((((-q) << 15) + 0x4000) >> 9)
        }
    } else if q >= 511 {
        0x7FFF
    } else {
        ((q << 15) + 0x4000) >> 9
    }
}

/// Final `BC6H_SF16` unquantize step per the D3D11 / Khronos specification: maps an interpolated value in
/// `[-0x7FFF, 0x7FFF]` to its IEEE 754 `F16` bit pattern (`u16`).
#[inline]
pub const fn finish_unquantize_sf16(interp: i32) -> u16 {
    if interp < 0 {
        let mag = (((-interp) * 31) >> 5) as u16;
        if mag == 0 { 0x0000 } else { (mag & 0x7FFF) | 0x8000 }
    } else {
        ((interp * 31) >> 5) as u16 & 0x7FFF
    }
}

/// Interpolate a single channel between two unquantized `BC6H_SF16` endpoints using 4-bit index `idx` (`0..16`).
#[inline]
pub const fn interpolate_sf16(e0_unq: i32, e1_unq: i32, idx: usize) -> u16 {
    let w = WEIGHT4[idx & 15];
    let interp = (e0_unq * (64 - w) + e1_unq * w + 32) >> 6;
    finish_unquantize_sf16(interp)
}

/// Encode 16 RGB `F16` texels (4x4 row-major, `[R, G, B]` as IEEE 754 `u16` bit patterns) into a 16-byte
/// `BC6H_SF16` Mode 11 (`0b00011`) hardware block.
pub fn encode_bc6h_block(rgb: &[[u16; 3]; 16]) -> [u8; 16] {
    let mut dom = [[0i32; 3]; 16];
    let mut min_v = [i32::MAX; 3];
    let mut max_v = [i32::MIN; 3];
    for (i, px) in rgb.iter().enumerate() {
        for c in 0..3 {
            let v = f16_bits_to_sf16_domain(px[c]);
            dom[i][c] = v;
            if v < min_v[c] {
                min_v[c] = v;
            }
            if v > max_v[c] {
                max_v[c] = v;
            }
        }
    }

    let mut q0 = [0i32; 3];
    let mut q1 = [0i32; 3];
    let mut u0 = [0i32; 3];
    let mut u1 = [0i32; 3];
    for c in 0..3 {
        q0[c] = quantize_sf16_10(min_v[c]);
        q1[c] = quantize_sf16_10(max_v[c]);
        u0[c] = unquantize_sf16_10(q0[c]);
        u1[c] = unquantize_sf16_10(q1[c]);
    }

    // Find the best 4-bit index (0..16) for each of the 16 texels in the unquantized domain.
    let mut indices = [0u8; 16];
    for (i, d) in dom.iter().enumerate() {
        let mut best_idx = 0u8;
        let mut best_err = i64::MAX;
        for (idx, &w) in WEIGHT4.iter().enumerate() {
            let mut err = 0i64;
            for c in 0..3 {
                let interp = (u0[c] * (64 - w) + u1[c] * w + 32) >> 6;
                let diff = i64::from(d[c] - interp);
                err += diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_idx = idx as u8;
            }
        }
        indices[i] = best_idx;
    }

    // Texel 0 is the anchor texel whose MSB (bit 3) must be 0 in Mode 11.
    // Because WEIGHT4 is symmetric (`WEIGHT4[15 - k] == 64 - WEIGHT4[k]`), swapping `(q0, q1)` and replacing
    // every index `idx` with `15 - idx` preserves the exact interpolated values while clearing bit 3 of `indices[0]`.
    if indices[0] >= 8 {
        core::mem::swap(&mut q0, &mut q1);
        for idx in &mut indices {
            *idx = 15 - *idx;
        }
    }

    // Pack into a 128-bit LSB-first bitstream:
    //   bits 0..5   : mode = 0b00011 (3)
    //   bits 5..15  : q0[0] (10-bit two's complement)
    //   bits 15..25 : q0[1]
    //   bits 25..35 : q0[2]
    //   bits 35..45 : q1[0]
    //   bits 45..55 : q1[1]
    //   bits 55..65 : q1[2]
    //   bits 65..68 : indices[0] (3 bits, anchor MSB omitted)
    //   bits 68..128: indices[1..16] (4 bits each)
    let mut bits: u128 = 0b00011;
    let mut shift = 5u32;
    for &q in q0.iter().chain(q1.iter()) {
        let tc = (q as i16 as u16) & 0x3FF;
        bits |= u128::from(tc) << shift;
        shift += 10;
    }
    bits |= u128::from(indices[0] & 0x07) << shift;
    shift += 3;
    for &idx in &indices[1..] {
        bits |= u128::from(idx & 0x0F) << shift;
        shift += 4;
    }
    debug_assert_eq!(shift, 128);
    bits.to_le_bytes()
}

/// Decode a 16-byte `BC6H_SF16` Mode 11 (`0b00011`) block into 16 RGB `F16` texels (`[[u16; 3]; 16]`).
pub fn decode_bc6h_block(block: &[u8; 16]) -> [[u16; 3]; 16] {
    let bits = u128::from_le_bytes(*block);
    let sign_ext_10 = |v: u32| -> i32 {
        let m = (v & 0x3FF) as i32;
        if (m & 0x200) != 0 { m - 1024 } else { m }
    };
    let q0 = [
        sign_ext_10((bits >> 5) as u32),
        sign_ext_10((bits >> 15) as u32),
        sign_ext_10((bits >> 25) as u32),
    ];
    let q1 = [
        sign_ext_10((bits >> 35) as u32),
        sign_ext_10((bits >> 45) as u32),
        sign_ext_10((bits >> 55) as u32),
    ];
    let u0 = [
        unquantize_sf16_10(q0[0]),
        unquantize_sf16_10(q0[1]),
        unquantize_sf16_10(q0[2]),
    ];
    let u1 = [
        unquantize_sf16_10(q1[0]),
        unquantize_sf16_10(q1[1]),
        unquantize_sf16_10(q1[2]),
    ];

    let mut out = [[0u16; 3]; 16];
    let idx0 = ((bits >> 65) as usize) & 0x07;
    out[0] = [
        interpolate_sf16(u0[0], u1[0], idx0),
        interpolate_sf16(u0[1], u1[1], idx0),
        interpolate_sf16(u0[2], u1[2], idx0),
    ];
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let shift = 68 + (i - 1) * 4;
        let idx = ((bits >> shift) as usize) & 0x0F;
        *slot = [
            interpolate_sf16(u0[0], u1[0], idx),
            interpolate_sf16(u0[1], u1[1], idx),
            interpolate_sf16(u0[2], u1[2], idx),
        ];
    }
    out
}

/// Encode a single-channel 4x4 `F16` block into a 16-byte `BC6H_SF16` Mode 11 block (replicated across R, G, B
/// so a hardware `Bc6hRgbFloat` sampler yields the channel in `.r`, `.g`, and `.b`).
pub fn encode_bc6h_mono_block(texels: &[u16; 16]) -> [u8; 16] {
    let mut rgb = [[0u16; 3]; 16];
    for (dst, &v) in rgb.iter_mut().zip(texels.iter()) {
        *dst = [v, v, v];
    }
    encode_bc6h_block(&rgb)
}

/// Decode a single-channel 4x4 `F16` block from a 16-byte `BC6H_SF16` Mode 11 block (reads channel 0).
pub fn decode_bc6h_mono_block(block: &[u8; 16]) -> [u16; 16] {
    let rgb = decode_bc6h_block(block);
    let mut out = [0u16; 16];
    for (dst, src) in out.iter_mut().zip(rgb.iter()) {
        *dst = src[0];
    }
    out
}

/// Byte length of the contiguous hardware `BC6H_SF16` block array for a `w x h` plane (`ceil(w/4) * ceil(h/4) * 16`).
#[inline]
pub fn bc6h_blocks_len(w: u32, h: u32) -> Option<usize> {
    let nbx = (w.div_ceil(4)) as usize;
    let nby = (h.div_ceil(4)) as usize;
    nbx.checked_mul(nby)?.checked_mul(16)
}

/// Mode-byte flag indicating that the `nbx * nby * 16` hardware `BC6H_SF16` block array is compressed with
/// `reorder::split_delta(.., 16)` + `rANS`, prefixed by its `u32` LE byte length (`[mode | 0x80][r_len: u32 LE][r_blocks]`).
pub const BLOCKS_RANS_FLAG: u8 = 0x80;

/// Extract the contiguous hardware `BC6H_SF16` blocks (`ceil(w/4) * ceil(h/4) * 16` bytes) from a `TAG_BC6H`
/// payload in O(1) zero-copy time when stored raw (`(mode & BLOCKS_RANS_FLAG) == 0`). When `BLOCKS_RANS_FLAG` is set,
/// use [`decode_bc6h_blocks_into`] to unpack the hardware block array.
pub fn extract_bc6h_blocks(payload: &[u8], w: u32, h: u32) -> Option<&[u8]> {
    let need = bc6h_blocks_len(w, h)?;
    let mode = *payload.first()?;
    if mode > 1 {
        return None;
    }
    payload.get(1..1usize.checked_add(need)?)
}

/// Unpack the contiguous `nbx * nby * 16` hardware `BC6H_SF16` block array from `payload` into `out_blocks`,
/// returning the byte offset in `payload` immediately following the block section (the residual tail).
pub fn decode_bc6h_blocks_into(
    payload: &[u8],
    w: u32,
    h: u32,
    mode: Bc6hMode,
    out_blocks: &mut [u8],
) -> Result<usize, DecodeError> {
    let blocks_bytes = bc6h_blocks_len(w, h).ok_or(DecodeError::Request("BC6H block count overflows"))?;
    if out_blocks.len() != blocks_bytes {
        return Err(DecodeError::Request("BC6H block output slice length != nbx*nby*16"));
    }
    let raw_mode = *payload.first().ok_or(DecodeError::Corrupt("BC6H empty payload"))?;
    let blocks_rans = (raw_mode & BLOCKS_RANS_FLAG) != 0;
    if (raw_mode & !BLOCKS_RANS_FLAG) != mode.byte() {
        return Err(DecodeError::Corrupt("BC6H blob header mode disagrees with profile"));
    }
    if !blocks_rans {
        let end = 1usize.checked_add(blocks_bytes).ok_or(DecodeError::Corrupt("BC6H blocks overflow"))?;
        let Some(slice) = payload.get(1..end) else {
            return Err(DecodeError::Corrupt("BC6H payload truncated in block array"));
        };
        out_blocks.copy_from_slice(slice);
        Ok(end)
    } else {
        let len_bytes: [u8; 4] = payload
            .get(1..5)
            .ok_or(DecodeError::Corrupt("BC6H rANS block header truncated"))?
            .try_into()
            .map_err(|_| DecodeError::Corrupt("BC6H rANS block header"))?;
        let r_len = u32::from_le_bytes(len_bytes) as usize;
        let end = 5usize.checked_add(r_len).ok_or(DecodeError::Corrupt("BC6H rANS block length overflow"))?;
        let Some(r_slice) = payload.get(5..end) else {
            return Err(DecodeError::Corrupt("BC6H payload truncated in rANS block stream"));
        };
        with_bytes(blocks_bytes, |dec_delta| -> Result<(), DecodeError> {
            rans::decode_into(r_slice, dec_delta).ok_or(DecodeError::Corrupt("BC6H block rANS"))?;
            reorder::join_undelta_into(dec_delta, 16, out_blocks);
            Ok(())
        })
        .map_err(|_| DecodeError::Alloc(blocks_bytes as u64))??;
        Ok(end)
    }
}

/// Encode a `w x h` `F16` plane (`w * h * 2` LE bytes) as a `TAG_BC6H` payload into `out` (after the caller
/// has pushed `TAG_BC6H`).
pub fn encode_plane_into(
    plane: &[u8],
    w: u32,
    h: u32,
    mode: Bc6hMode,
    rans_cfg: RansConfig,
    out: &mut Vec<u8>,
) -> Result<(), CodecError> {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|px| px.checked_mul(2))
        .ok_or(CodecError::Dims("BC6H plane size overflows"))?;
    if plane.len() != expected {
        return Err(CodecError::Dims("BC6H plane length != w*h*2"));
    }
    let blocks_bytes = bc6h_blocks_len(w, h).ok_or(CodecError::Dims("BC6H block count overflows"))?;
    let nbx = w.div_ceil(4) as usize;
    let nby = h.div_ceil(4) as usize;
    let w_us = w as usize;
    let h_us = h as usize;

    let sample_at = |x: usize, y: usize| -> u16 {
        let cx = x.min(w_us.saturating_sub(1));
        let cy = y.min(h_us.saturating_sub(1));
        let off = (cy * w_us + cx) * 2;
        u16::from_le_bytes([plane[off], plane[off + 1]])
    };

    let mut blocks = Vec::with_capacity(blocks_bytes);
    for by in 0..nby {
        for bx in 0..nbx {
            let mut texels = [0u16; 16];
            for dy in 0..4 {
                for dx in 0..4 {
                    texels[dy * 4 + dx] = sample_at(bx * 4 + dx, by * 4 + dy);
                }
            }
            let blk = encode_bc6h_mono_block(&texels);
            blocks.extend_from_slice(&blk);
        }
    }

    let delta_blocks = reorder::split_delta(&blocks, 16);
    let mut r_blocks = Vec::new();
    rans::encode_into(&delta_blocks, rans_cfg, &mut r_blocks)?;
    if 4usize.saturating_add(r_blocks.len()) < blocks.len()
        && let Ok(r_len) = u32::try_from(r_blocks.len())
    {
        out.reserve(5 + r_blocks.len());
        out.push(mode.byte() | BLOCKS_RANS_FLAG);
        out.extend_from_slice(&r_len.to_le_bytes());
        out.extend_from_slice(&r_blocks);
    } else {
        out.reserve(1 + blocks.len());
        out.push(mode.byte());
        out.extend_from_slice(&blocks);
    }

    if mode == Bc6hMode::LosslessResidual {
        with_bytes(expected, |res_plane| -> Result<(), CodecError> {
            for by in 0..nby {
                for bx in 0..nbx {
                    let b_off = (by * nbx + bx) * 16;
                    let mut blk = [0u8; 16];
                    blk.copy_from_slice(&blocks[b_off..b_off + 16]);
                    let dec = decode_bc6h_mono_block(&blk);
                    for dy in 0..4 {
                        let y = by * 4 + dy;
                        if y >= h_us {
                            continue;
                        }
                        for dx in 0..4 {
                            let x = bx * 4 + dx;
                            if x >= w_us {
                                continue;
                            }
                            let off = (y * w_us + x) * 2;
                            let orig = u16::from_le_bytes([plane[off], plane[off + 1]]);
                            let diff = orig.wrapping_sub(dec[dy * 4 + dx]);
                            let le = diff.to_le_bytes();
                            res_plane[off] = le[0];
                            res_plane[off + 1] = le[1];
                        }
                    }
                }
            }
            let delta = reorder::split_delta(res_plane, 2);
            let r_start = out.len();
            out.push(TAG_RANS);
            rans::encode_into(&delta, rans_cfg, out)?;
            if out.len().saturating_sub(r_start + 1) >= res_plane.len() {
                out.truncate(r_start);
                out.push(TAG_RAW);
                out.extend_from_slice(res_plane);
            }
            Ok(())
        })
        .map_err(|_| CodecError::Backend("BC6H residual scratch allocation failed".into()))??;
    }
    Ok(())
}

/// Decode a `TAG_BC6H` payload into `out` (`w * h * 2` LE bytes).
pub fn decode_plane_into(
    payload: &[u8],
    w: u32,
    h: u32,
    mode: Bc6hMode,
    out: &mut [u8],
) -> Result<(), DecodeError> {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|px| px.checked_mul(2))
        .ok_or(DecodeError::Request("BC6H plane size overflows"))?;
    if out.len() != expected {
        return Err(DecodeError::Request("BC6H output slice length != w*h*2"));
    }
    let blocks_bytes = bc6h_blocks_len(w, h).ok_or(DecodeError::Request("BC6H block count overflows"))?;
    let mut blocks_buf = vec![0u8; blocks_bytes];
    let tail_off = decode_bc6h_blocks_into(payload, w, h, mode, &mut blocks_buf)?;
    let tail = &payload[tail_off..];
    if mode == Bc6hMode::Lossy && !tail.is_empty() {
        return Err(DecodeError::Corrupt("BC6H lossy blob has unexpected trailing bytes"));
    }

    let nbx = w.div_ceil(4) as usize;
    let nby = h.div_ceil(4) as usize;
    let w_us = w as usize;
    let h_us = h as usize;

    for by in 0..nby {
        for bx in 0..nbx {
            let b_off = (by * nbx + bx) * 16;
            let mut blk = [0u8; 16];
            blk.copy_from_slice(&blocks_buf[b_off..b_off + 16]);
            let dec = decode_bc6h_mono_block(&blk);
            for dy in 0..4 {
                let y = by * 4 + dy;
                if y >= h_us {
                    continue;
                }
                for dx in 0..4 {
                    let x = bx * 4 + dx;
                    if x >= w_us {
                        continue;
                    }
                    let off = (y * w_us + x) * 2;
                    let le = dec[dy * 4 + dx].to_le_bytes();
                    out[off] = le[0];
                    out[off + 1] = le[1];
                }
            }
        }
    }

    if mode == Bc6hMode::LosslessResidual {
        let kind = classify_blob(tail).ok_or(DecodeError::Corrupt("BC6H residual tag"))?;
        with_bytes(expected, |res_plane| -> Result<(), DecodeError> {
            match kind {
                BlobKind::Rans(r_bytes) => {
                    with_bytes(expected, |dec_delta| -> Result<(), DecodeError> {
                        rans::decode_into(r_bytes, dec_delta).ok_or(DecodeError::Corrupt("BC6H residual rANS"))?;
                        reorder::join_undelta_into(dec_delta, 2, res_plane);
                        Ok(())
                    })
                    .map_err(|_| DecodeError::Alloc(expected as u64))??;
                }
                BlobKind::Raw(raw_bytes) => {
                    if raw_bytes.len() != expected {
                        return Err(DecodeError::Corrupt("BC6H raw residual length"));
                    }
                    res_plane.copy_from_slice(raw_bytes);
                }
                _ => return Err(DecodeError::Corrupt("BC6H residual unexpected blob tag")),
            }
            for (dst, res) in out.as_chunks_mut::<2>().0.iter_mut().zip(res_plane.as_chunks::<2>().0.iter()) {
                let base = u16::from_le_bytes(*dst);
                let delta = u16::from_le_bytes(*res);
                *dst = base.wrapping_add(delta).to_le_bytes();
            }
            Ok(())
        })
        .map_err(|_| DecodeError::Alloc(expected as u64))??;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc6h_block_roundtrip_accuracy() {
        let mut rgb = [[0u16; 3]; 16];
        for (i, px) in rgb.iter_mut().enumerate() {
            // Smooth HDR ramp around 1.0 (0x3C00)
            let base = 0x3C00u16 + (i as u16) * 16;
            *px = [base, base + 8, base + 16];
        }
        let blk = encode_bc6h_block(&rgb);
        assert_eq!(blk[0] & 0x1F, 0b00011, "Mode 11");
        let dec = decode_bc6h_block(&blk);
        for (orig, got) in rgb.iter().zip(dec.iter()) {
            for c in 0..3 {
                let diff = (i32::from(orig[c]) - i32::from(got[c])).abs();
                assert!(diff <= 32, "orig={:#06x} got={:#06x} diff={diff}", orig[c], got[c]);
            }
        }
    }

    #[test]
    fn bc6h_lossless_residual_is_bit_exact_over_all_u16_patterns() {
        // Test 256x256 = all 65,536 possible 16-bit patterns (including negative, subnormals, inf, NaN)!
        let (w, h) = (256u32, 256u32);
        let plane: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let mut encoded = Vec::new();
        encode_plane_into(&plane, w, h, Bc6hMode::LosslessResidual, RansConfig::default(), &mut encoded).expect("encode");
        let mut blocks = vec![0u8; bc6h_blocks_len(w, h).unwrap()];
        decode_bc6h_blocks_into(&encoded, w, h, Bc6hMode::LosslessResidual, &mut blocks).expect("decode blocks");
        let mut decoded = vec![0u8; plane.len()];
        decode_plane_into(&encoded, w, h, Bc6hMode::LosslessResidual, &mut decoded).expect("decode");
        assert_eq!(decoded, plane, "LosslessResidual must reconstruct all 65,536 u16 bit patterns bit-exact");
    }
}
