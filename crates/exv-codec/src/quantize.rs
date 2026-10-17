//! Light-lossy quantization: drop the low `q` bits of each element (with rounding).
//!
//! # Idea (no container/decode changes)
//! Lossy = **lossless encoding of pre-quantized data**: quantization is applied only on
//! encode, the stored values are the quantized ones, so [`crate::decode_channel`] stays the same
//! (it reads already-quantized values, no loss on read). `codec_id` does not change.
//! Compression benefit: zeroed low bits → more repeatable/zero bytes → reorder/wavelet
//! + rANS compress harder.
//!
//! Deterministic (integer operations on the element's bit pattern). Class PXR24/B44
//! (bit-depth reduction). For floats ONLY the mantissa is quantized (see [`quantize_plane`]): rounding
//! the exponent bits is the failure mode `hdr.rs` exists to avoid, and non-finite values pass through.

use exv_format::Dtype;

/// Quantize a plane: zero the low `q` bits of each element with round-to-nearest.
///
/// **Integers:** `q == 0` or `q >= 8*esz` → returns a copy unchanged (meaningless levels); `q` in
/// `1..8*esz` sets the loss strength (max abs error on the integer value ≤ `2^(q-1)`).
///
/// **Floats (F16/BF16/F32):** only the MANTISSA is quantized — `q` saturates at the mantissa width
/// (10/7/23), so a larger `q` can only lose more (drops the whole mantissa) and never rounds the
/// exponent bits (which would turn values into garbage). A mantissa carry into the exponent is a
/// correct round-up to the next power of two, except when it would overflow a finite value to ±inf — then
/// the value is truncated instead and stays finite. NaN/±inf are returned bit-for-bit (payloads and
/// signalling NaNs included), never silently turned into inf.
pub fn quantize_plane(plane: &[u8], dtype: Dtype, q: u32) -> Vec<u8> {
    let esz = dtype.size();
    let bits = 8 * esz as u32;
    // (effective q, exponent-field mask) — the mask is `Some` only for floats.
    let (q, exp_mask) = match dtype {
        Dtype::F16 => (q.min(10), Some(0x7c00u64)),
        Dtype::Bf16 => (q.min(7), Some(0x7f80)),
        Dtype::F32 => (q.min(23), Some(0x7f80_0000)),
        _ => (q, None),
    };
    if q == 0 || (exp_mask.is_none() && q >= bits) {
        return plane.to_vec();
    }
    let half = 1u64 << (q - 1);
    let mask = !((1u64 << q) - 1);
    let width_max = if esz >= 4 { u32::MAX as u64 } else { (1u64 << bits) - 1 };

    // `N` is a const so each element is one fixed-width load/store (a runtime-width byte loop per element).
    fn run<const N: usize>(out: &mut [u8], plane: &[u8], q: Quant) {
        for (o, ch) in out.as_chunks_mut::<N>().0.iter_mut().zip(plane.as_chunks::<N>().0) {
            let v = u64::from(crate::temporal::get::<N>(ch));
            let vq = match q.exp_mask {
                Some(em) if v & em == em => v, // NaN / ±inf: untouched
                _ => {
                    // round to nearest quantum, clamped so the carry does not "wrap" the value to 0
                    let r = (v + q.half).min(q.width_max) & q.mask;
                    match q.exp_mask {
                        Some(em) if r & em == em => v & q.mask, // finite would round up to inf: truncate instead
                        _ => r,
                    }
                }
            };
            // `vq <= width_max <= u32::MAX` (`N <= 4`): its low `N` bytes are those of the `u32` truncation.
            *o = crate::temporal::put::<N>(vq as u32);
        }
    }
    let mut out = vec![0u8; plane.len()];
    let quant = Quant { half, mask, width_max, exp_mask };
    match esz {
        1 => run::<1>(&mut out, plane, quant),
        2 => run::<2>(&mut out, plane, quant),
        _ => run::<4>(&mut out, plane, quant), // dtype widths are 1, 2 or 4
    }
    out
}

/// The per-plane constants of [`quantize_plane`]'s rounding step.
#[derive(Clone, Copy)]
struct Quant {
    half: u64,
    mask: u64,
    width_max: u64,
    exp_mask: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Profile, decode_channel, encode_channel};

    /// Float quantization never touches the exponent, keeps NaN/inf bit-exact and keeps finite values
    /// finite; `q` beyond the mantissa width saturates (monotonic) instead of rounding exponent bits.
    #[test]
    fn float_quant_preserves_exponent_and_nonfinite() {
        let f16 = |bits: &[u16]| bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>();
        let back16 = |v: &[u8]| v.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>();
        // 1.0, signalling NaN, qNaN, +inf, -inf, F16 max finite, -max, a mid value.
        let src = [0x3c00u16, 0x7c01, 0x7e55, 0x7c00, 0xfc00, 0x7bff, 0xfbff, 0x4a37];
        let out = back16(&quantize_plane(&f16(&src), Dtype::F16, 10));
        assert_eq!(&out[1..5], &src[1..5], "NaN/inf untouched bit-for-bit");
        assert_eq!(out[0], 0x3c00, "1.0 keeps its exponent");
        assert_eq!(out[5], 0x7800, "max finite truncates instead of overflowing to inf: {:#06x}", out[5]);
        assert_eq!(out[6], 0xf800);
        // q past the mantissa width saturates: q=15 == q=10 (no exponent rounding).
        assert_eq!(quantize_plane(&f16(&src), Dtype::F16, 15), quantize_plane(&f16(&src), Dtype::F16, 10));
        // F32 / BF16: same rules.
        let f32b: Vec<u8> = [1.5f32, f32::NAN, f32::INFINITY, f32::MAX, -0.0]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let o32: Vec<f32> =
            quantize_plane(&f32b, Dtype::F32, 40).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(o32[0], 2.0, "1.5 rounds to the next power of two (mantissa carry)");
        assert!(o32[1].is_nan() && o32[2] == f32::INFINITY && o32[3].is_finite() && o32[4] == 0.0);
        let bf: Vec<u8> = [0x3f80u16, 0x7f81].iter().flat_map(|b| b.to_le_bytes()).collect();
        assert_eq!(back16(&quantize_plane(&bf, Dtype::Bf16, 12))[1], 0x7f81);
    }

    #[test]
    fn q_zero_and_full_are_noop() {
        let data: Vec<u8> = (0..256u16).flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(quantize_plane(&data, Dtype::U16, 0), data);
        assert_eq!(quantize_plane(&data, Dtype::U16, 16), data); // q>=width → no-op
    }

    #[test]
    fn int_quant_bounds_error() {
        // U16: quantizing q bits → |orig - quantized| <= 2^(q-1) (round to nearest).
        let data: Vec<u8> = (0..4096u16).flat_map(|v| v.to_le_bytes()).collect();
        for q in [1u32, 3, 6, 10] {
            let qd = quantize_plane(&data, Dtype::U16, q);
            let bound = 1i64 << (q - 1);
            for (o, c) in data.as_chunks::<2>().0.iter().zip(qd.as_chunks::<2>().0.iter()) {
                let a = u16::from_le_bytes([o[0], o[1]]) as i64;
                let b = u16::from_le_bytes([c[0], c[1]]) as i64;
                assert!((a - b).abs() <= bound, "q={q}: |{a}-{b}| > {bound}");
            }
            // low q bits cleared
            for c in qd.as_chunks::<2>().0.iter() {
                let b = u16::from_le_bytes([c[0], c[1]]);
                assert_eq!(b & ((1 << q) - 1), 0, "low {q} bits not cleared");
            }
        }
    }

    #[test]
    fn lossy_pipeline_roundtrips_to_quantized() {
        // Lossy = lossless(quantize): decode must return EXACTLY the quantized data (bit-exact).
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n).flat_map(|i| (i as u16).to_le_bytes()).collect();
        let q = 5;
        let quantized = quantize_plane(&plane, Dtype::U16, q);
        for prof in [Profile::lossless_fast(), Profile::lossless_wave()] {
            let blob = encode_channel(&quantized, Dtype::U16, w, h, &prof).expect("encode");
            let back = decode_channel(&blob, Dtype::U16, w, h, &prof).expect("decode");
            assert_eq!(back, quantized, "lossy round-trips to quantized via {:?}", prof.transform);
        }
    }

    #[test]
    fn quant_improves_or_keeps_compression() {
        // On noisy data quantization must not increase the size (usually decreases it).
        let mut seed = 0x1234_5678u64;
        let data: Vec<u8> = (0..8192)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed & 0xff) as u8
            })
            .collect();
        let prof = Profile::lossless_fast();
        let lossless = encode_channel(&data, Dtype::U8, 128, 64, &prof).expect("encode").len();
        let q = quantize_plane(&data, Dtype::U8, 4);
        let lossy = encode_channel(&q, Dtype::U8, 128, 64, &prof).expect("encode").len();
        assert!(lossy <= lossless, "lossy {lossy} > lossless {lossless}");
    }
}
