//! EXV-RT — realtime, embarrassingly-parallel LOSSY block codec (`docs/DESIGN_LOG.md` D-09).
//!
//! # Why
//! The realtime f16-RGBA playback path must decode with NO global serial entropy dependency so the GPU
//! can decode every block independently (Tier-0 proved a per-block decode→RGBA16F sustains ~91 GB/s /
//! 42 fps at 16k — `exv-gpu/tests/bc6h_floor_bench.rs`). This module is the CPU encoder + the matching
//! reference decoder for that block format, plus HDR quality metrics — the Tier-0 quality gate AND the
//! Tier-2 encoder v0. The GPU decoder (WGSL) mirrors `decode_plane` bit-for-bit later.
//!
//! # Format (v0 — endpoint+indices, BC6H-class, per-channel)
//! A channel plane is tiled into `block`×`block` squares (edge blocks clamped). Per block:
//!   `[min:f16][max:f16][ idx_bits-bit index per texel, row-major, LE bit-packed ]`
//! Decode: `value = min + (idx / (2^idx_bits - 1)) * (max - min)`. Each block is INDEPENDENT and at a
//! known byte offset (fixed-rate → offset = block_index × block_bytes, no table needed) — the property
//! that makes GPU decode embarrassingly parallel. `idx_bits` and `block` are the quality/ratio dials.
//!
//! Per-channel (not joint-RGB like real BC6H) keeps it simple and lets each AOV channel pick its own
//! fit; joint-RGB / a per-block transform are later quality refinements if the gate needs them.
//!
//! # Honesty
//! v0 is a linear min/max fit per block — crude on high-variance/highlight blocks (no transform). The
//! quality gate ([`quality`]) reports relative error on the real HDR corpus so we SEE whether 8-bpp-class
//! block compression holds, or whether a transform (DWT) is needed before Tier-2 ships.

// The ONE f16<->f32 implementation (and NaN policy) lives in `hdr`; short local names for this codec.
use crate::hdr::{exp2f, f32_to_half as f2h, half_to_f32 as h2f, log2f};

/// Signed perceptual log map for HDR endpoint fitting: `code = sign(v) * log2(1 + |v|)`. Monotonic,
/// handles 0 and negative AOVs, and COMPRESSES dynamic range so a per-block linear min/max+lerp fit no
/// longer crushes highlights (the v0 failure on HDR colour). Quantization happens in this code space;
/// decode applies [`ilog`]. One `log2`/`exp2` per texel — cheap, keeps the decode embarrassingly parallel.
/// Uses the deterministic [`log2f`]/[`exp2f`] (not libm) so the lossy decode is bit-exact on every platform.
fn flog(v: f32) -> f32 {
    let s = if v < 0.0 { -1.0 } else { 1.0 };
    s * log2f(1.0 + v.abs())
}
/// Inverse of [`flog`]: `v = sign(c) * (2^|c| - 1)`.
fn ilog(c: f32) -> f32 {
    let s = if c < 0.0 { -1.0 } else { 1.0 };
    s * (exp2f(c.abs()) - 1.0)
}

/// Bytes per block for `block`×`block` texels at `idx_bits`: 2 endpoints (f16) + packed indices,
/// rounded up to whole bytes. Fixed per (block, idx_bits) → blocks are directly addressable.
pub fn block_bytes(block: u32, idx_bits: u32) -> usize {
    let texels = (block * block) as usize;
    4 + (texels * idx_bits as usize).div_ceil(8)
}

/// Bits-per-pixel of one channel at (block, idx_bits) (endpoint overhead amortized over the block).
pub fn bpp(block: u32, idx_bits: u32) -> f32 {
    (block_bytes(block, idx_bits) as f32 * 8.0) / (block * block) as f32
}

/// Encode ONE channel plane (`w*h` f16 LE bytes) into fixed-rate blocks. `block` = square edge,
/// `idx_bits` ∈ 1..=8. Output = `n_blocks * block_bytes(block, idx_bits)`, row-major over blocks.
pub fn encode_plane(src_f16le: &[u8], w: u32, h: u32, block: u32, idx_bits: u32) -> Vec<u8> {
    encode_plane_mode(src_f16le, w, h, block, idx_bits, true)
}

/// [`encode_plane`] with explicit `log` mode (log endpoint space when `true` — recommended for HDR).
pub fn encode_plane_mode(src_f16le: &[u8], w: u32, h: u32, block: u32, idx_bits: u32, log: bool) -> Vec<u8> {
    let levels = (1u32 << idx_bits) - 1;
    let bb = block_bytes(block, idx_bits);
    let wb = w.div_ceil(block);
    let hb = h.div_ceil(block);
    let mut out = vec![0u8; (wb * hb) as usize * bb];
    let at = |x: u32, y: u32| -> f32 {
        let i = (y as usize * w as usize + x as usize) * 2;
        let v = h2f(u16::from_le_bytes([src_f16le[i], src_f16le[i + 1]]));
        if log { flog(v) } else { v }
    };
    for by in 0..hb {
        for bx in 0..wb {
            // min/max over the (clamped) block, ignoring non-finite for the fit.
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for ty in 0..block {
                let y = by * block + ty;
                if y >= h { continue; }
                for tx in 0..block {
                    let x = bx * block + tx;
                    if x >= w { continue; }
                    let v = at(x, y);
                    if v.is_finite() {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            if !lo.is_finite() { lo = 0.0; }
            if !hi.is_finite() { hi = lo; }
            let base = (by * wb + bx) as usize * bb;
            // Endpoints stored as f16 IN THE (log) code space; the decoder lerps in the same space
            // then applies the inverse map. (A log code fits f16 fine — its magnitude is ~log2 of v.)
            out[base..base + 2].copy_from_slice(&f2h(lo).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&f2h(hi).to_le_bytes());
            // Quantize each texel to the nearest index; bit-pack row-major into the block's index region.
            let range = (hi - lo).max(f32::MIN_POSITIVE);
            let mut bitpos = 0usize;
            for ty in 0..block {
                let y = (by * block + ty).min(h - 1);
                for tx in 0..block {
                    let x = (bx * block + tx).min(w - 1);
                    let v = at(x, y);
                    let t = if v.is_finite() { ((v - lo) / range).clamp(0.0, 1.0) } else { 0.0 };
                    let idx = (t * levels as f32 + 0.5) as u32; // nearest
                    // LE bit-pack `idx_bits` starting at `bitpos` within the index region (offset +4 bytes).
                    let mut rem = idx_bits;
                    let mut val = idx;
                    let mut bp = bitpos;
                    while rem > 0 {
                        let byte = base + 4 + bp / 8;
                        let off = (bp % 8) as u32;
                        let take = rem.min(8 - off);
                        let m = ((1u32 << take) - 1) as u8;
                        out[byte] |= ((val as u8) & m) << off;
                        val >>= take;
                        rem -= take;
                        bp += take as usize;
                    }
                    bitpos += idx_bits as usize;
                }
            }
        }
    }
    out
}

/// Decode blocks produced by [`encode_plane`] back to a `w*h` f16 LE plane. Reference for the WGSL GPU
/// decoder (must match bit-for-bit).
pub fn decode_plane(blocks: &[u8], w: u32, h: u32, block: u32, idx_bits: u32) -> Vec<u8> {
    decode_plane_mode(blocks, w, h, block, idx_bits, true)
}

/// [`decode_plane`] with explicit `log` mode (must match the encode mode).
pub fn decode_plane_mode(blocks: &[u8], w: u32, h: u32, block: u32, idx_bits: u32, log: bool) -> Vec<u8> {
    let levels = (1u32 << idx_bits) - 1;
    let bb = block_bytes(block, idx_bits);
    let wb = w.div_ceil(block);
    let mut out = vec![0u8; w as usize * h as usize * 2];
    for by in 0..h.div_ceil(block) {
        for bx in 0..wb {
            let base = (by * wb + bx) as usize * bb;
            if base + bb > blocks.len() { continue; }
            let lo = h2f(u16::from_le_bytes([blocks[base], blocks[base + 1]]));
            let hi = h2f(u16::from_le_bytes([blocks[base + 2], blocks[base + 3]]));
            let range = hi - lo;
            let mut bitpos = 0usize;
            for ty in 0..block {
                let y = by * block + ty;
                for tx in 0..block {
                    let x = bx * block + tx;
                    // read idx_bits at bitpos
                    let mut rem = idx_bits;
                    let mut got = 0u32;
                    let mut shift = 0u32;
                    let mut bp = bitpos;
                    while rem > 0 {
                        let byte = base + 4 + bp / 8;
                        let off = (bp % 8) as u32;
                        let take = rem.min(8 - off);
                        let m = ((1u32 << take) - 1) as u8;
                        got |= (((blocks[byte] >> off) & m) as u32) << shift;
                        shift += take;
                        rem -= take;
                        bp += take as usize;
                    }
                    bitpos += idx_bits as usize;
                    if x < w && y < h {
                        let t = got as f32 / levels as f32;
                        let c = lo + t * range;
                        let v = if log { ilog(c) } else { c };
                        let o = (y as usize * w as usize + x as usize) * 2;
                        out[o..o + 2].copy_from_slice(&f2h(v).to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

/// HDR quality of a reconstructed plane vs the original (both f16 LE, same dims). Returns
/// `(rms_rel, max_rel)` — RELATIVE error `|orig-rec| / (|orig| + eps)` (linear HDR makes absolute
/// error meaningless — a highlight at 1000 dwarfs a shadow at 0.01). Non-finite samples are skipped.
pub fn quality(orig_f16le: &[u8], rec_f16le: &[u8]) -> (f32, f32) {
    let n = orig_f16le.len().min(rec_f16le.len()) / 2;
    let eps = 1e-3f32;
    let mut sum = 0.0f64;
    let mut cnt = 0u64;
    let mut maxr = 0.0f32;
    for i in 0..n {
        let a = h2f(u16::from_le_bytes([orig_f16le[i * 2], orig_f16le[i * 2 + 1]]));
        let b = h2f(u16::from_le_bytes([rec_f16le[i * 2], rec_f16le[i * 2 + 1]]));
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let r = (a - b).abs() / (a.abs() + eps);
        sum += (r * r) as f64;
        cnt += 1;
        maxr = maxr.max(r);
    }
    let rms = if cnt > 0 { (sum / cnt as f64).sqrt() as f32 } else { 0.0 };
    (rms, maxr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_plane(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|&v| f2h(v).to_le_bytes()).collect()
    }

    #[test]
    fn half_roundtrip_basic() {
        for &v in &[0.0f32, 1.0, 0.5, 2.0, -3.5, 1024.0, 0.1] {
            let back = h2f(f2h(v));
            assert!((back - v).abs() <= v.abs() * 0.01 + 1e-3, "half rt {v} -> {back}");
        }
    }

    #[test]
    fn block_roundtrip_constant_is_exact() {
        // A constant block: lo==hi, every texel reconstructs to that value (within half precision).
        let (w, h) = (8u32, 8u32);
        let src = f16_plane(&vec![3.25f32; (w * h) as usize]);
        let enc = encode_plane(&src, w, h, 4, 4);
        let dec = decode_plane(&enc, w, h, 4, 4);
        let (rms, maxr) = quality(&src, &dec);
        assert!(rms < 1e-3 && maxr < 1e-2, "constant block rms={rms} max={maxr}");
    }

    #[test]
    fn block_roundtrip_gradient_bounded() {
        // A smooth gradient: error must be bounded and shrink as idx_bits grows (sanity of the dial).
        let (w, h) = (64u32, 64u32);
        let src: Vec<u8> = {
            let mut v = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let val = (x as f32 / w as f32) * 10.0 + (y as f32 / h as f32) * 2.0;
                    v.extend_from_slice(&f2h(val).to_le_bytes());
                }
            }
            v
        };
        let q2 = quality(&src, &decode_plane(&encode_plane(&src, w, h, 8, 2), w, h, 8, 2)).0;
        let q4 = quality(&src, &decode_plane(&encode_plane(&src, w, h, 8, 4), w, h, 8, 4)).0;
        assert!(q4 < q2, "more index bits must reduce error: q2={q2} q4={q4}");
        assert!(q4 < 0.05, "4-bit gradient rms_rel should be small: {q4}");
    }
}
