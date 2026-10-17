//! Transform-domain lossy: **signed** reversible 5/3 wavelet (i64, no wrapping) + uniform
//! quantization of detail subbands + zigzag/varint + rANS.
//!
//! # How it differs from lossless `wavelet.rs`
//! The lossless wavelet computes in the element width with wrapping (mod 2^width) → coefficients
//! are "wrapped" uint values that cannot be quantized. Here lifting runs in **signed i64** without
//! wrapping → coefficients are truly signed (LL ≈ average, details ≈ small signed) and can be quantized.
//!
//! - `q == 0` → lossless (i64-lifting is exactly reversible, clamped to the element range) — this is
//!   the correctness gate of the lifting itself (bit-exact round-trip).
//! - `q > 0` → details are divided by `2^q` (rounding away from zero), LL is preserved exactly. The
//!   loss is bounded; the ratio grows.
//!
//! HDR floats: quantizing the raw float bit pattern corrupts the exponent (garbage). On the lossy
//! path (q>0) float planes are first mapped to the value-domain **log2 codes** of [`crate::hdr`], so
//! detail quantization is perceptually-uniform (relative error). q==0 keeps the exact bit path
//! (lossless contract). INTEGER channels (ids, masks, counters) are NEVER quantized: they ride the same
//! signed wavelet with `q` ignored, so a lossy profile keeps them bit-exact (the project invariant shared
//! with the HTJ2K-lossy and EXV-RT profiles) - a quantized object-id AOV is garbage, not "lower quality".
//!
//! `q` is passed explicitly (the container stores it in `Profile::params`), so the codec stays
//! self-contained and testable.

use std::collections::TryReserveError;

use crate::temporal::{get, put};
use crate::{hdr, wavelet};
use exv_format::{AlignedBuf, Dtype};
use rans::RansConfig;

/// Maximum quantization shift: `1i64 << q` must not overflow i64 (`1 << 63` flips the sign bit),
/// so the exponent is capped at 62.
///
/// # Why clamp instead of skip (monotonicity — BUG #14)
/// The quantization step is `2^q`; every detail coefficient magnitude is bounded (HDR codes ≤
/// `hdr::MAX_CODE`, integers ≤ the element range, each further limited by the two 5/3 lift steps),
/// so once `2^(q-1)` exceeds that bound EVERY detail rounds to 0 — the maximum-loss saturation
/// point. A shift of 62 reaches saturation for every v1 dtype (U32 detail ≤ ~2^34). Clamping the
/// shift here (rather than SILENTLY SKIPPING quantization when `q ≥ 8·esz`, as the old `valid_q`
/// gate did) is what keeps quality MONOTONIC in `q`: a larger `q` can only lose more, never flip the
/// lossy codec back to (near-)lossless. No silent fallback (project invariant). Encode and decode
/// clamp identically, so any `q` — including a forged out-of-range value from the container — stays
/// symmetric.
const QMAX: u32 = crate::WAVE_Q_MAX as u32;

/// Effective quantization step `2^min(q, QMAX)`, shared by [`quantize`]/[`dequantize`] so the two
/// sides always agree.
#[inline]
fn qstep(q: u32) -> i64 {
    1i64 << q.min(QMAX)
}

/// Forward transform + quantize a plane to signed coefficients — the lossy front-end WITHOUT entropy
/// coding. This is the REUSABLE transform stage for a block/parallel codec (Tier-2): the GPU decoder
/// inverts it per block, and an entropy layer (shared-table rANS / fixed-rate) wraps the coefficients.
/// HDR floats (q>0) map through `hdr` log2 codes first (perceptual quant) and their detail subbands are
/// quantized by `2^q`; integers go direct and are never quantized ([`is_quantized`]). Returns `w*h` i64
/// coefficients (LL exact).
pub fn forward_quant(plane: &[u8], dtype: Dtype, w: u32, h: u32, q: u32) -> Vec<i64> {
    let mut g = vec![0i64; (w as usize) * (h as usize)];
    forward_quant_into(plane, dtype, w, h, q, &mut g);
    g
}

/// [`forward_quant`] into `g` (exactly `w*h` coefficients, every one overwritten; `plane` is `w*h*esz` bytes).
/// The EXV-RT encoder transforms thousands of tiny blocks per plane through one reused grid.
pub fn forward_quant_into(plane: &[u8], dtype: Dtype, w: u32, h: u32, q: u32, g: &mut [i64]) {
    let esz = dtype.size();
    let (wu, hu) = (w as usize, h as usize);
    debug_assert_eq!((g.len(), plane.len()), (wu * hu, wu * hu * esz), "forward_quant_into: geometry");
    if is_quantized(dtype, q) {
        hdr::to_codes_into(plane, dtype, hdr::DEFAULT_SCALE, g);
    } else {
        to_i64(plane, esz, g);
    }
    fwd_2d(g, wu, hu);
    // Quantize every FLOAT plane for q>0 (the shift is clamped in `qstep`, never overflows). NO dtype-width
    // skip — that silent lossless fallback was the q8 non-monotonicity (BUG #14); the integer exemption is
    // by dtype CLASS (a documented invariant), not by magnitude.
    if is_quantized(dtype, q) {
        quantize(g, wu, q);
    }
}

/// `true` when a `dtype` plane's detail subbands are quantized at `q`: floats only, and only for `q > 0`.
/// The ONE predicate shared by [`forward_quant`] and the inverse so the two cannot disagree.
fn is_quantized(dtype: Dtype, q: u32) -> bool {
    q > 0 && dtype.is_float()
}

/// Inverse of [`forward_quant`]: dequantize → inverse 5/3 → (HDR) inverse log codes → plane bytes.
/// The reference the GPU per-block inverse-transform kernel mirrors. `coeffs` is `w*h` i64 (shorter is
/// zero-padded, longer truncated — robust to a forged length).
pub fn inverse_dequant(coeffs: &[i64], dtype: Dtype, w: u32, h: u32, q: u32) -> Vec<u8> {
    let (wu, hu) = (w as usize, h as usize);
    let mut g = coeffs.to_vec();
    g.resize(wu * hu, 0);
    let mut out = vec![0u8; wu * hu * dtype.size()];
    inverse_into(&mut g, dtype, w, h, q, &mut out);
    out
}

/// [`inverse_dequant`] on a coefficient buffer of EXACTLY `w*h` values, writing the `w*h*esz` plane bytes into
/// `out` (every byte is overwritten): transforms `g` in place, so the decoder holds ONE `i64` copy of the plane
/// (the working-set figure `LossyWave::work_bytes` accounts), not two, and no output `Vec`.
pub(crate) fn inverse_into(g: &mut [i64], dtype: Dtype, w: u32, h: u32, q: u32, out: &mut [u8]) {
    let esz = dtype.size();
    let (wu, hu) = (w as usize, h as usize);
    // Mirror `forward_quant`: dequantize exactly what was quantized (same clamped shift).
    if is_quantized(dtype, q) {
        dequantize(g, wu, q);
    }
    inv_2d(g, wu, hu);
    if is_quantized(dtype, q) {
        hdr::from_codes_into(g, dtype, hdr::DEFAULT_SCALE, out);
    } else {
        from_i64(g, esz, out);
    }
}

thread_local! {
    /// The `i64` coefficient grid of one decoded plane, kept between calls (taken while in use). Cache-line aligned
    /// ([`AlignedBuf`]): the lifting streams whole rows of it (see `wavelet::GRID`).
    static COEFFS: std::cell::Cell<AlignedBuf> = const { std::cell::Cell::new(AlignedBuf::new()) };
}

/// Largest coefficient grid (bytes) a thread keeps between calls (4 MiB = 512 Ki `i64`).
const COEFFS_KEEP: usize = 4 << 20;

/// Run `f` over a reused per-thread `i64` grid of exactly `n` coefficients with UNSPECIFIED contents (the EXV-RT
/// decoder inverts thousands of tiny blocks per plane; a `Vec` per block was most of its allocation count).
///
/// # Errors
/// The allocator's refusal of an `8·n`-byte grid (reported, never an abort).
pub(crate) fn with_coeffs<T>(n: usize, f: impl FnOnce(&mut [i64]) -> T) -> Result<T, TryReserveError> {
    let mut buf = COEFFS.with(std::cell::Cell::take);
    // A saturated byte count cannot be reserved, so an overflowing `n` is the same typed refusal.
    let res = buf.try_fit(n.saturating_mul(size_of::<i64>())).map(|()| f(buf.cast_mut::<i64>()));
    if buf.capacity() <= COEFFS_KEEP {
        COEFFS.with(|c| c.set(buf));
    }
    res
}

/// Serialize coefficients to the rANS symbol stream (zigzag + LEB128 varint) — the bytes the entropy
/// coder consumes. Public so a block codec can POOL many blocks' coefficients under ONE shared rANS
/// table (the fix for per-block table overhead). Inverse: [`bytes_to_coeffs`].
pub fn coeffs_to_bytes(coeffs: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        put_varint(zigzag(c), &mut bytes);
    }
    bytes
}

/// Parse `n` coefficients from a zigzag+varint byte stream ([`coeffs_to_bytes`] inverse). `None` on a
/// truncated/corrupt stream. Every coefficient needs at least one varint byte, so `n > bytes.len()` is
/// provably corrupt and is rejected BEFORE the `8·n`-byte allocation (a forged tile area must not abort
/// the process).
pub fn bytes_to_coeffs(bytes: &[u8], n: usize) -> Option<Vec<i64>> {
    if n > bytes.len() {
        return None;
    }
    let mut g = vec![0i64; n];
    bytes_to_coeffs_into(bytes, &mut g)?;
    Some(g)
}

/// [`bytes_to_coeffs`] into `g` (`n = g.len()` coefficients; `None` on a truncated/corrupt stream or `n > bytes.len()`).
pub fn bytes_to_coeffs_into(bytes: &[u8], g: &mut [i64]) -> Option<()> {
    if g.len() > bytes.len() {
        return None;
    }
    let mut p = 0usize;
    for c in g {
        *c = unzigzag(get_varint(bytes, &mut p)?);
    }
    Some(())
}

/// Encode a plane lossy: [`forward_quant`] → [`coeffs_to_bytes`] → rANS. (Whole-plane; the block/GPU
/// codec reuses the same stages with a shared-table or fixed-rate entropy layer instead.)
pub fn encode_lossy(plane: &[u8], dtype: Dtype, w: u32, h: u32, q: u32, cfg: RansConfig) -> Result<Vec<u8>, rans::EncodeError> {
    let mut out = Vec::new();
    encode_lossy_into(plane, dtype, w, h, q, cfg, &mut out)?;
    Ok(out)
}

/// [`encode_lossy`] APPENDING the rANS container to `out` (`out` is unchanged on `Err`).
pub fn encode_lossy_into(
    plane: &[u8],
    dtype: Dtype,
    w: u32,
    h: u32,
    q: u32,
    cfg: RansConfig,
    out: &mut Vec<u8>,
) -> Result<(), rans::EncodeError> {
    let n = (w as usize) * (h as usize);
    with_coeffs(n, |g| {
        forward_quant_into(plane, dtype, w, h, q, g);
        crate::with_vec(|bytes| {
            bytes.reserve(g.len());
            for &c in g.iter() {
                put_varint(zigzag(c), bytes);
            }
            rans::encode_into(bytes, cfg, out)
        })
    })
    // No coefficient grid for this plane on this machine: the plane is too large to encode here (the one
    // size-class error this API carries).
    .unwrap_or(Err(rans::EncodeError::TooLarge(n)))
}

/// Decode a lossy plane (requires the same `q`). `None` on a corrupt stream.
pub fn decode_lossy(data: &[u8], dtype: Dtype, w: u32, h: u32, q: u32) -> Option<Vec<u8>> {
    let n = (w as usize).checked_mul(h as usize)?;
    // The plane is allocated only AFTER the stream proved it carries `n` coefficients (see `varints`).
    crate::with_vec(|bytes| {
        varints(data, n, bytes)?;
        let mut out = vec![0u8; n.checked_mul(dtype.size())?];
        finish(bytes, n, dtype, w, h, q, &mut out)?;
        Some(out)
    })
}

/// [`decode_lossy`] into `out`, which must be exactly `w*h*esz` bytes (`None` otherwise). Every byte is
/// overwritten on success.
pub fn decode_lossy_into(data: &[u8], dtype: Dtype, w: u32, h: u32, q: u32, out: &mut [u8]) -> Option<()> {
    // CLASS-A: forged w*h must not overflow usize (capacity/loop) → checked → None.
    let n = (w as usize).checked_mul(h as usize)?;
    if n.checked_mul(dtype.size())? != out.len() {
        return None;
    }
    crate::with_vec(|bytes| {
        varints(data, n, bytes)?;
        finish(bytes, n, dtype, w, h, q, out)
    })
}

/// The zigzag/varint byte stream of an `n`-coefficient lossy plane (rANS decoded into the per-thread scratch `bytes`,
/// no `Vec` per plane: it was 0.65x of the raw tile in churn). `None` unless the container is well formed AND carries
/// at least `n` bytes (every coefficient needs one): the guard that keeps a forged `n` from sizing the 8·n-byte
/// coefficient grid or the output plane.
fn varints(data: &[u8], n: usize, bytes: &mut Vec<u8>) -> Option<()> {
    // CLASS-B: rANS output ceiling = ≤ n coefficients × ≤10 varint bytes (decompression-bomb guard
    // BEFORE the allocation in rANS): a tiny container with a huge `n` is rejected.
    rans::decode_capped_into(data, n.checked_mul(10)?, bytes)?;
    (n <= bytes.len()).then_some(())
}

/// Varint bytes -> coefficients -> inverse transform into `out` (`n*esz` bytes).
fn finish(bytes: &[u8], n: usize, dtype: Dtype, w: u32, h: u32, q: u32, out: &mut [u8]) -> Option<()> {
    with_coeffs(n, |g| {
        bytes_to_coeffs_into(bytes, g)?;
        inverse_into(g, dtype, w, h, q, out);
        Some(())
    })
    .ok()?
}

// --- signed 5/3 lifting: the shared generic body of `wavelet.rs` over i64 ---
// CLASS-E: the decode path lifts UNtrusted coefficients (full i64 from unzigzag/varint); the shared lifting
// wraps, so forged input yields deterministic garbage rather than an overflow panic (DoS). On valid data there is
// no overflow, so the round-trip stays bit-exact.

fn fwd_2d(g: &mut [i64], w: usize, h: usize) {
    wavelet::fwd(g, w, h, ());
}

fn inv_2d(g: &mut [i64], w: usize, h: usize) {
    wavelet::inv(g, w, h, ());
}

// --- quantization of detail subbands (LL is preserved exactly) ---

/// Apply `f` to every DETAIL coefficient (at least one index odd: LH/HL/HH); LL (both even) is untouched.
/// Even rows hold LL at even `x`, so only their odd `x` are visited; odd rows are all detail. Row-wise, no
/// per-pixel branch.
fn for_detail(g: &mut [i64], w: usize, mut f: impl FnMut(&mut i64)) {
    if w == 0 {
        return;
    }
    for (y, row) in g.chunks_exact_mut(w).enumerate() {
        if y & 1 == 0 {
            row.iter_mut().skip(1).step_by(2).for_each(&mut f);
        } else {
            row.iter_mut().for_each(&mut f);
        }
    }
}

fn quantize(g: &mut [i64], w: usize, q: u32) {
    let sh = q.min(QMAX);
    let half = qstep(q) >> 1;
    // Division with rounding away from zero. The step is a power of two and both operands are non-negative,
    // so `/ step` is exactly `>> sh` (a variable-divisor idiv per coefficient was the quantizer's cost).
    for_detail(g, w, |c| *c = if *c >= 0 { (*c + half) >> sh } else { -((-*c + half) >> sh) });
}

fn dequantize(g: &mut [i64], w: usize, q: u32) {
    let step = qstep(q);
    // CLASS-E: on forged coefficients i64×step overflows → wrapping (no panic).
    for_detail(g, w, |c| *c = c.wrapping_mul(step));
}

// --- elements <-> i64 (value/bit pattern) with clamp to range on write ---

/// Widen the LE `esz`-byte elements of `bytes` into `g`.
fn to_i64(bytes: &[u8], esz: usize, g: &mut [i64]) {
    fn load<const N: usize>(bytes: &[u8], g: &mut [i64]) {
        for (e, c) in g.iter_mut().zip(bytes.as_chunks::<N>().0) {
            *e = i64::from(get::<N>(c));
        }
    }
    match esz {
        1 => load::<1>(bytes, g),
        2 => load::<2>(bytes, g),
        _ => load::<4>(bytes, g), // dtype widths are 1, 2 or 4
    }
}

/// Clamp every coefficient to the element range and write its low `esz` bytes into `out` (`g.len() * esz`).
fn from_i64(g: &[i64], esz: usize, out: &mut [u8]) {
    let max = if esz >= 8 { i64::MAX } else { (1i64 << (8 * esz)) - 1 };
    // Clamp to the element range (lossy rarely drifts past the edges), then keep the low `N` bytes.
    fn store<const N: usize>(g: &[i64], max: i64, out: &mut [u8]) {
        // `max < 2^32` for `N <= 4`, so the clamped value's low `N` bytes are those of its `u32` truncation.
        for (c, &v) in out.as_chunks_mut::<N>().0.iter_mut().zip(g) {
            *c = put::<N>(v.clamp(0, max) as u32);
        }
    }
    match esz {
        1 => store::<1>(g, max, out),
        2 => store::<2>(g, max, out),
        _ => store::<4>(g, max, out), // dtype widths are 1, 2 or 4
    }
}

// --- zigzag + varint (LEB128) ---

fn zigzag(c: i64) -> u64 {
    ((c << 1) ^ (c >> 63)) as u64
}
fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}
fn put_varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}
fn get_varint(d: &[u8], p: &mut usize) -> Option<u64> {
    let mut r = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *d.get(*p)?;
        *p += 1;
        r |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return None; // guard against a corrupt varint
        }
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_plane(w: u32, h: u32, f: impl Fn(u32, u32) -> u16) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h) as usize * 2);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y).to_le_bytes());
            }
        }
        v
    }

    /// q=0 → bit-exact lossless (correctness gate for signed lifting + zigzag/varint + rANS).
    #[test]
    fn q0_is_lossless() {
        let cfg = RansConfig::default();
        for &(w, h) in &[(16u32, 16u32), (33, 17), (1, 9), (9, 1), (64, 48)] {
            let plane = u16_plane(w, h, |x, y| ((x * 13 + y * 7) % 4096) as u16);
            let enc = encode_lossy(&plane, Dtype::U16, w, h, 0, cfg).expect("encode");
            let dec = decode_lossy(&enc, Dtype::U16, w, h, 0).expect("decode");
            assert_eq!(dec, plane, "q=0 lossless {w}x{h}");
        }
    }

    /// q>0 → round-trip works, max error is bounded (grows moderately with q).
    #[test]
    fn lossy_error_bounded() {
        let cfg = RansConfig::default();
        let (w, h) = (128u32, 128u32);
        let plane = u16_plane(w, h, |x, y| ((x + y) * 8) as u16);
        for q in [2u32, 4, 6] {
            let enc = encode_lossy(&plane, Dtype::U16, w, h, q, cfg).expect("encode");
            let dec = decode_lossy(&enc, Dtype::U16, w, h, q).expect("decode");
            // max abs error is finite and grows moderately with q
            let max_err = plane
                .as_chunks::<2>().0.iter()
                .zip(dec.as_chunks::<2>().0.iter())
                .map(|(a, b)| {
                    (u16::from_le_bytes([a[0], a[1]]) as i64 - u16::from_le_bytes([b[0], b[1]]) as i64)
                        .abs()
                })
                .max()
                .unwrap_or(0);
            assert!(max_err < (1i64 << (q + 6)), "q={q}: max_err {max_err} too large");
        }
    }

    /// HDR-float lossy via the value-domain log codes (hdr.rs): bounded RELATIVE error across a
    /// 5-decade range, NOT the catastrophic exponent corruption of bit-reinterpret quant (this same
    /// case used to give ~2900x relative error). This is the float-lossy fix.
    #[test]
    fn lossy_float_relative_bounded() {
        let cfg = RansConfig::default();
        let (w, h) = (64u32, 64u32);
        // log-uniform HDR ramp 0.01 .. 1000 (5 decades).
        let plane: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| (0.01f32 * 10f32.powf(5.0 * i as f32 / 4095.0)).to_le_bytes())
            .collect();
        for q in [2u32, 4] {
            let enc = encode_lossy(&plane, Dtype::F32, w, h, q, cfg).expect("encode");
            let dec = decode_lossy(&enc, Dtype::F32, w, h, q).expect("decode");
            let max_rel = plane
                .as_chunks::<4>().0.iter()
                .zip(dec.as_chunks::<4>().0.iter())
                .map(|(a, b)| {
                    let av = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let bv = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (av - bv).abs() / av.abs()
                })
                .fold(0.0f32, f32::max);
            assert!(max_rel < 0.25, "q={q}: float lossy relative error {max_rel} too large");
        }
    }

    /// DIAGNOSTIC (run: `cargo test --release -p exv-codec diag_q_sweep_hdr -- --ignored --nocapture`):
    /// is q=12 a SPIKE (bug) or on a monotonic curve (just heavy loss)? Relative error vs q on HDR F32.
    #[test]
    #[ignore = "diagnostic probe"]
    fn diag_q_sweep_hdr() {
        let cfg = RansConfig::default();
        let (w, h) = (128u32, 128u32);
        let plane: Vec<u8> = (0..(w * h))
            .flat_map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let base = 0.03f32 * 2f32.powf(((x * 0.05).sin() + (y * 0.06).cos()) * 3.0);
                let v = base * (1.0 + 0.25 * ((x * 0.9).sin() * (y * 0.8).cos()));
                v.to_le_bytes()
            })
            .collect();
        let orig: Vec<f32> = plane.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        eprintln!("  q   maxRel%   rmsRel%");
        for q in 6u32..=18 {
            let enc = encode_lossy(&plane, Dtype::F32, w, h, q, cfg).expect("encode");
            let dec = decode_lossy(&enc, Dtype::F32, w, h, q).expect("decode");
            let out: Vec<f32> = dec.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let (mut mx, mut ss) = (0f64, 0f64);
            for (o, d) in orig.iter().zip(&out) {
                let r = if o.abs() > 1e-6 { ((d - o) / o).abs() as f64 } else { 0.0 };
                mx = mx.max(r);
                ss += r * r;
            }
            eprintln!("{q:>3}  {:>7.2}  {:>7.2}", mx * 100.0, (ss / orig.len() as f64).sqrt() * 100.0);
        }
    }

    /// Regression (BUG #14): lossy_wave quality must be MONOTONIC in `q` — a larger `q` may only
    /// lose more, never flip back toward lossless. The old `valid_q` gate SILENTLY SKIPPED
    /// quantization once `q >= 8*esz`, so q=16 on U16 (and q=32 on F32) became (near-)lossless and
    /// q8 measured WORSE than q16. We sweep past each dtype's boundary and assert the error never
    /// collapses below half the worst error seen at any smaller `q` (the bug was a ~40–90× cliff;
    /// sub-percent rounding wiggle near the noise floor is tolerated).
    #[test]
    fn lossy_wave_monotonic_in_q() {
        let cfg = RansConfig::default();
        let (w, h) = (48u32, 48u32);

        // Integer channels (ids, masks) are NEVER quantized: bit-exact at every q (N6 - the root CLAUDE.md
        // promise "integer channels stay LOSSLESS under a lossy profile"), for every integer dtype.
        let iplane = u16_plane(w, h, |x, y| (((x + y) * 37) % 5000 + 100) as u16);
        let u8plane: Vec<u8> = (0..(w * h) as usize).map(|i| (i * 7 % 256) as u8).collect();
        let u32plane: Vec<u8> = (0..(w * h) as usize).flat_map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()).collect();
        for (dt, plane) in [(Dtype::U16, &iplane), (Dtype::U8, &u8plane), (Dtype::U32, &u32plane)] {
            for q in [1u32, 4, 8, 16, 20, 62, 200] {
                let enc = encode_lossy(plane, dt, w, h, q, cfg).expect("encode");
                assert_eq!(decode_lossy(&enc, dt, w, h, q).as_deref(), Some(plane.as_slice()), "{dt:?} q={q} must stay exact");
            }
        }

        // HDR float channel (F32): boundary q = 32. err = max relative.
        let fplane: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                let t = ((i as u32 % w) + (i as u32 / w)) as f32 / (2.0 * w as f32);
                (0.02f32 * 8f32.powf(t * 4.0)).to_le_bytes()
            })
            .collect();
        let ferr = |q: u32| -> f32 {
            let enc = encode_lossy(&fplane, Dtype::F32, w, h, q, cfg).expect("encode");
            let dec = decode_lossy(&enc, Dtype::F32, w, h, q).expect("decode");
            fplane
                .as_chunks::<4>().0.iter()
                .zip(dec.as_chunks::<4>().0.iter())
                .map(|(a, b)| {
                    let av = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let bv = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (av - bv).abs() / av.abs()
                })
                .fold(0.0f32, f32::max)
        };
        let mut worst = 0.0f32;
        for q in 1u32..=36 {
            let e = ferr(q);
            assert!(e * 2.0 >= worst, "F32 q={q}: err {e} collapsed vs worst-so-far {worst} (non-monotonic cliff)");
            worst = worst.max(e);
        }
        assert!(worst > 0.0, "F32 sweep produced no loss at all");
    }

    #[test]
    fn decode_garbage_is_none() {
        assert!(decode_lossy(&[], Dtype::U16, 4, 4, 2).is_none());
    }

    /// A forged tile area must be rejected before any `8·n`-byte coefficient allocation: a valid tiny
    /// rANS stream (1 byte of payload) claiming a 65536² tile is `None`, not an abort.
    #[test]
    fn forged_area_is_rejected_before_alloc() {
        let tiny = rans::encode(&[0u8], RansConfig::default()).expect("encode");
        assert!(decode_lossy(&tiny, Dtype::U16, 65536, 65536, 0).is_none());
        assert!(bytes_to_coeffs(&[0u8; 4], usize::MAX).is_none());
        assert!(bytes_to_coeffs(&[0u8; 4], 5).is_none());
        assert_eq!(bytes_to_coeffs(&[0u8; 4], 4).map(|g| g.len()), Some(4));
    }
}
