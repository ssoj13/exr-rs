//! `hdr.rs` — value-domain (perceptual **log2**) transform for HDR-float lossy coding, plus the ONE
//! set of f16/bf16 ↔ f32 conversions and the deterministic `exp2`/`log2` the lossy paths share.
//!
//! # Why this exists
//! Both lossy backends (`lossy.rs` signed-wavelet and the HTJ2K `htj2k.rs` codes path) work on an
//! **integer** representation of each sample. For true integers that is fine; for HDR `f32`/`f16` the
//! naive "reinterpret the float bits as an integer" representation is **exponential** in value, so
//! uniform quantization corrupts the float exponent → catastrophic error (measured: F32 q=50 →
//! ~1.16e5 max-abs error). The fix is a **value-domain** transform: map each float to a bounded
//! signed integer **code** that is ~linear in `log2(value)`. Quantizing codes then means uniform
//! error in *log space* = uniform **relative** (perceptually uniform) error — the DWA-class idea.
//!
//! # Two ways to spend the loss
//! - **lossy_wave**: codes at the fine [`DEFAULT_SCALE`] (hdr adds negligible loss), and the wavelet's
//!   own detail quantization (`2^q`) is the quality knob.
//! - **HTJ2K codes path**: the codes are coded **losslessly** (reversible HTJ2K), so the ONLY loss is
//!   the hdr quantization itself — controlled by `scale` (codes per stop). Higher scale = finer =
//!   near-lossless + larger; lower scale = coarser/lossier + smaller. Predictable: the max relative
//!   error is ≈ `2^(1/(2·scale)) − 1`.
//!
//! # Determinism contract (CPU on every platform == the GPU kernels)
//! `scale` is an INTEGER (codes per stop, `1..=256`). The platform `libm` (`powf`/`log2`) is NOT
//! bit-exact across OSes (UCRT/glibc/musl/macOS differ by an ULP), which would make lossy files and
//! decodes differ per platform. Everything here uses only IEEE-754 `+ − × ÷` and exact bit ops, which
//! Rust evaluates identically everywhere (no FMA contraction, no libm):
//! - `2^x`, `x∈[0,1)` = [`exp2_frac`]: f64 Horner of `Σ (x·ln2)^k/k!`, 18 terms (error < 1e-19).
//! - **decode**: code magnitude `m` → `k = m − 32·scale`, `a = floor(k/scale)`, `r = k mod scale`;
//!   `value = bitcast<f32>((a+127)<<23) · T[r]` where `T[r] = f32(exp2_frac(r/scale))` (one f64→f32
//!   rounding; the table is [`exp2_table`]). `a ∈ [−32, 32]`; magnitudes above `64·scale` saturate.
//! - **encode**: `a` = the f32's unbiased exponent (bit extraction), `f ∈ [1,2)` its mantissa (exact
//!   in f64); `r` = the number of thresholds `exp2_frac((j+½)/scale) ≤ f` (binary search), `code =
//!   (a+32)·scale + r` — i.e. `round(scale·(log2|v|+32))`, no `log2` call.
//!
//! A WGSL mirror must load `T` from a buffer (this crate's [`exp2_table`]) instead of calling the
//! hardware `exp2` to be bit-exact; then decode is one table fetch + one exact exponent add + one
//! multiply (which is exact: a power-of-two scale of a normal f32).
//!
//! # NaN / non-finite policy (one rule for every float conversion in this crate)
//! - f16/bf16 <-> f32 follow `exv_format::half`'s policy (widen exact incl. NaN payload; narrow RNE, NaN -> canonical qNaN).
//! - The log **codes** cannot represent NaN: `NaN → 0` (= +0.0), `±inf → ±MAX_CODE` (decodes to ±2^32:
//!   finite for F32/BF16, ±inf again once narrowed to F16), `−0.0 → 0` (sign of zero lost),
//!   the zero code is also the rounding result for everything below the first half-step above the floor:
//!   `|v| < 2^(-32 + 1/(2·scale)) → 0` (so exactly `2^-32` encodes to 0 and decodes to 0.0; the smallest
//!   non-zero code is `2^(-32 + 1/scale)`), `|v| > 2^32` saturates. Acceptable for lossy; integer channels never come here.
//! - `scale` out of `1..=256` is clamped (codes always fit a 16-bit signed component).

use std::sync::OnceLock;

use exv_format::Dtype;

/// Stops represented below `1.0` (floor at `2^-EXP_BIAS`). 32 stops ≈ down to 2.3e-10.
const EXP_BIAS: i32 = 32;
/// Total log2 span `[2^-EXP_BIAS, 2^(EXP_SPAN-EXP_BIAS)]`. 64 stops ≈ up to 4.3e9 — covers f16's
/// full range and the practical f32 HDR range.
const EXP_SPAN: i32 = 64;
/// Default codes-per-stop for callers whose downstream codec does the quantizing (lossy_wave):
/// fine enough (~0.135% half-step) that hdr is near-lossless and the codec's quant is the knob.
pub const DEFAULT_SCALE: u32 = 256;
/// Maximum codes-per-stop. Caps code magnitude at `MAX_CODE = EXP_SPAN·MAX_SCALE = 16384`, so codes
/// always fit a 16-bit signed component (the HTJ2K codes path).
const MAX_SCALE: u32 = 256;
/// Maximum code magnitude.
const MAX_CODE: i32 = EXP_SPAN * MAX_SCALE as i32;

// Exact decimal literals of the std constants: parsed identically on every platform.
use std::f64::consts::{LN_2 as LN2, LOG2_E as INV_LN2, SQRT_2 as SQRT2};
/// Taylor terms of `e^y` (`y ≤ ln2`): the truncation error `y^19/19! < 1e-20`, below f64 rounding.
const EXP_TERMS: usize = 18;
/// `1/k!` for `k = 0..=EXP_TERMS`, const-evaluated with IEEE division (identical on every platform).
const INV_FACT: [f64; EXP_TERMS + 1] = {
    let mut t = [1.0f64; EXP_TERMS + 1];
    let mut k = 1;
    while k <= EXP_TERMS {
        t[k] = t[k - 1] / (k as f64);
        k += 1;
    }
    t
};

/// `2^x` for `x ∈ [0, 1)` in f64 — fixed-order Horner over [`INV_FACT`]; only `+`/`×`, so bit-exact on
/// every platform. The single primitive behind the code tables and [`exp2f`].
fn exp2_frac(x: f64) -> f64 {
    let y = x * LN2;
    let mut r = INV_FACT[EXP_TERMS];
    for k in (0..EXP_TERMS).rev() {
        r = r * y + INV_FACT[k];
    }
    r
}

/// Deterministic `2^x` (f32 in/out, evaluated in f64, one final rounding): replaces `powf`/`exp2` from
/// the platform libm in the lossy decode path. NaN → NaN; `x ≥ 128` → +inf; `x < −150` → 0.
pub(crate) fn exp2f(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= 128.0 {
        return f32::INFINITY;
    }
    if x < -150.0 {
        return 0.0;
    }
    let a = x.floor(); // exact
    let p = exp2_frac((x - a) as f64); // x - a is exact in f32
    let pow2 = f64::from_bits(((a as i64 + 1023) as u64) << 52); // exact power of two, a ∈ [-150, 127]
    (p * pow2) as f32 // single correctly-rounded f64→f32 (also handles f32 denormals)
}

/// Deterministic `log2(x)` (f32 in/out, evaluated in f64, one final rounding): replaces the libm
/// `log2` in the lossy encode path. `x<0`/NaN → NaN, `0` → −inf, `+inf` → +inf. Exponent by bit
/// extraction, mantissa by the `atanh` series `ln f = 2z(1 + z²/3 + z⁴/5 + …)`, `z = (f−1)/(f+1)`.
pub(crate) fn log2f(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let b = (x as f64).to_bits(); // every f32 (even denormal) is a normal f64
    let mut e = ((b >> 52) & 0x7ff) as i64 - 1023;
    let mut f = f64::from_bits((b & ((1u64 << 52) - 1)) | (1023u64 << 52)); // [1, 2)
    if f > SQRT2 {
        f *= 0.5; // keep |z| <= 0.172 so 12 series terms reach f64 precision
        e += 1;
    }
    let z = (f - 1.0) / (f + 1.0);
    let z2 = z * z;
    let mut s = 0.0f64;
    for k in (0..12).rev() {
        s = s * z2 + 1.0 / ((2 * k + 1) as f64);
    }
    ((e as f64) + 2.0 * z * s * INV_LN2) as f32
}

/// Mantissa bits indexing [`Lut::bins`] (4096 bins over `[1, 2)`; thresholds are >= 2^-9 apart, so a bin holds at most one).
const BIN_BITS: u32 = 12;

/// Per-`scale` code tables: `tab[r] = f32(2^(r/scale))` (decode) and `thr[j] = 2^((j+½)/scale)`
/// (encode rounding thresholds, f64). Built once per scale and cached ([`lut`]).
///
/// On top of the exact scalar [`Lut::encode`]/[`Lut::decode`] sit whole-domain tables, built lazily by calling
/// THOSE functions on every input (all 2^16 half/bf16 patterns, all `2·MAX_CODE+1` codes), so the table
/// path is bit-identical to the scalar one by construction while a sample costs one load. They replace a
/// branchy f16 widen + binary search per encoded sample and a `div_euclid` + narrow per decoded one.
struct Lut {
    scale: u32,
    tab: Vec<f32>,
    thr: Vec<f64>,
    /// `bins[b]` = how many thresholds are `<= 1 + b/4096`: the binary search starts here and needs <= 1 compare.
    bins: Vec<u16>,
    /// f16 / bf16 bit pattern -> code.
    enc_h: OnceLock<Box<[i16; 65536]>>,
    enc_b: OnceLock<Box<[i16; 65536]>>,
    /// code + MAX_CODE -> f32 value / f16 bits / bf16 bits. Fixed-size arrays: a clamped code indexes them with no
    /// bounds check (the length is a type fact, like the `u16`-indexed encode tables).
    dec_f: OnceLock<Box<[f32; DEC_N]>>,
    dec_h: OnceLock<Box<[u16; DEC_N]>>,
    dec_b: OnceLock<Box<[u16; DEC_N]>>,
}

/// Entries of a whole-domain decode table: every code `-MAX_CODE..=MAX_CODE`.
const DEC_N: usize = 2 * MAX_CODE as usize + 1;

impl Lut {
    fn new(scale: u32) -> Self {
        let s = scale as f64;
        let thr: Vec<f64> = (0..scale).map(|j| exp2_frac((j as f64 + 0.5) / s)).collect();
        let bins = (0..1u32 << BIN_BITS)
            .map(|b| {
                let lo = 1.0 + b as f64 / (1u32 << BIN_BITS) as f64;
                thr.partition_point(|&t| t <= lo) as u16
            })
            .collect();
        Self {
            scale,
            tab: (0..scale).map(|r| exp2_frac(r as f64 / s) as f32).collect(),
            thr,
            bins,
            enc_h: OnceLock::new(),
            enc_b: OnceLock::new(),
            dec_f: OnceLock::new(),
            dec_h: OnceLock::new(),
            dec_b: OnceLock::new(),
        }
    }

    fn encode(&self, v: f32) -> i32 {
        if v.is_nan() {
            return 0;
        }
        let s = self.scale as i32;
        let neg = v.is_sign_negative();
        let signed = |c: i32| if neg { -c } else { c };
        if v.is_infinite() {
            return signed(EXP_SPAN * s);
        }
        let bits = v.abs().to_bits();
        let a = (bits >> 23) as i32 - 127; // unbiased exponent (zero/denormal land far below -32)
        if a < -EXP_BIAS {
            return 0;
        }
        if a >= EXP_SPAN - EXP_BIAS {
            return signed(EXP_SPAN * s); // >= 2^32 saturates at the top code
        }
        let m = bits & 0x7f_ffff;
        let f = 1.0 + m as f64 / 8_388_608.0; // exact mantissa in [1, 2)
        // Count the thresholds `<= f` (0..=scale; scale = carry into the next stop): start from the bin's count
        // (thresholds `<= ` the bin's lower edge, all of which are `<= f` too) and step over the rest.
        let mut r = self.bins.get((m >> (23 - BIN_BITS)) as usize).map_or(0, |&b| b as usize);
        while self.thr.get(r).is_some_and(|&t| t <= f) {
            r += 1;
        }
        signed((a + EXP_BIAS) * s + r as i32)
    }

    fn decode(&self, code: i32) -> f32 {
        if code == 0 {
            return 0.0;
        }
        let s = self.scale as i32;
        // `unsigned_abs`: i32::MIN must not overflow. Saturate forged magnitudes (|k| <= 32·scale).
        let k = code.unsigned_abs().min((EXP_SPAN * s) as u32) as i32 - EXP_BIAS * s;
        let (a, r) = (k.div_euclid(s), k.rem_euclid(s));
        let v = f32::from_bits(((a + 127) as u32) << 23) * self.tab[r as usize]; // exact exponent add
        if code < 0 { -v } else { v }
    }

    /// Whole-domain `bits16 -> code` table of a 16-bit float format (`widen` = its exact f32 conversion).
    fn enc16(&self, bf16: bool) -> &[i16; 65536] {
        let cell = if bf16 { &self.enc_b } else { &self.enc_h };
        cell.get_or_init(|| {
            let widen = if bf16 { bf16_to_f32 } else { half_to_f32 };
            let mut t = Box::new([0i16; 65536]);
            for (e, b) in t.iter_mut().zip(0..=u16::MAX) {
                *e = self.encode(widen(b)) as i16; // |code| <= 16384
            }
            t
        })
    }

    /// Whole-domain `code + MAX_CODE -> value` tables, one per output dtype, so the per-sample decode is a load
    /// (the narrowing to f16/bf16 is baked in).
    fn dec_f32(&self) -> &[f32; DEC_N] {
        self.dec_f.get_or_init(|| {
            let mut t = Box::new([0.0f32; DEC_N]);
            for (e, c) in t.iter_mut().zip(-MAX_CODE..=MAX_CODE) {
                *e = self.decode(c);
            }
            t
        })
    }

    fn dec16(&self, bf16: bool) -> &[u16; DEC_N] {
        let cell = if bf16 { &self.dec_b } else { &self.dec_h };
        cell.get_or_init(|| {
            let narrow = if bf16 { f32_to_bf16 } else { f32_to_half };
            let mut t = Box::new([0u16; DEC_N]);
            for (e, &v) in t.iter_mut().zip(self.dec_f32().iter()) {
                *e = narrow(v);
            }
            t
        })
    }
}

/// The cached [`Lut`] for `scale` (clamped to `1..=256`; at most 256 tables, built lazily).
fn lut(scale: u32) -> &'static Lut {
    static LUTS: [OnceLock<Lut>; MAX_SCALE as usize] = [const { OnceLock::new() }; MAX_SCALE as usize];
    let s = scale.clamp(1, MAX_SCALE);
    LUTS[s as usize - 1].get_or_init(|| Lut::new(s))
}

/// The decode table `T[r] = f32(2^(r/scale))`, `r ∈ 0..scale` — what a GPU decoder uploads instead of
/// calling the hardware `exp2` (see the module's determinism contract).
pub fn exp2_table(scale: u32) -> Vec<f32> {
    lut(scale).tab.clone()
}

/// Float value → signed log2 code at `scale` codes per stop. See the module docs for corner cases.
#[inline]
pub fn encode_value(v: f32, scale: u32) -> i32 {
    lut(scale).encode(v)
}

/// Signed log2 code → float value (inverse of [`encode_value`] at the same `scale`).
#[inline]
pub fn decode_value(code: i32, scale: u32) -> f32 {
    lut(scale).decode(code)
}

// f16 / bf16 <-> f32 live in `exv_format::half` (the SSOT, NaN policy documented there); re-exported so
// `hdr::half_to_f32` & co keep working inside exv-codec.
pub use exv_format::half::{bf16_to_f32, f32_to_bf16, f32_to_half, half_to_f32};

/// Decode one float sample (`esz` bytes at `c`) of `dtype` to `f32`.
#[inline]
fn read_f32(c: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        Dtype::Bf16 => bf16_to_f32(u16::from_le_bytes([c[0], c[1]])),
        Dtype::F16 => half_to_f32(u16::from_le_bytes([c[0], c[1]])),
        _ => 0.0, // non-float dtypes never reach the HDR path
    }
}

// --- plane-level (the codec-facing API) ---

/// Float plane bytes → log2 codes at `scale` codes/stop (`|code| ≤ MAX_CODE` fits the downstream
/// wavelet (`i64`) / HTJ2K (`i32`) integer pipelines — pick `T`). `dtype` must be a float type.
/// F16/BF16 planes are one table load per sample ([`Lut::enc16`]); F32 runs the exact scalar encode.
pub fn to_codes<T: From<i32> + Copy>(plane: &[u8], dtype: Dtype, scale: u32) -> Vec<T> {
    let mut out = vec![T::from(0); plane.len() / dtype.size()];
    to_codes_into(plane, dtype, scale, &mut out);
    out
}

/// [`to_codes`] into `out` (`plane.len() / esz` codes; every one overwritten).
pub fn to_codes_into<T: From<i32>>(plane: &[u8], dtype: Dtype, scale: u32, out: &mut [T]) {
    let (esz, lut) = (dtype.size(), lut(scale));
    match dtype {
        Dtype::F16 | Dtype::Bf16 => {
            let t = lut.enc16(dtype == Dtype::Bf16);
            for (o, c) in out.iter_mut().zip(plane.as_chunks::<2>().0) {
                *o = T::from(i32::from(t[u16::from_le_bytes(*c) as usize]));
            }
        }
        Dtype::F32 => {
            for (o, c) in out.iter_mut().zip(plane.as_chunks::<4>().0) {
                *o = T::from(lut.encode(f32::from_le_bytes(*c)));
            }
        }
        _ => {
            for (o, c) in out.iter_mut().zip(plane.chunks_exact(esz)) {
                *o = T::from(lut.encode(read_f32(c, dtype)));
            }
        }
    }
}

/// Log2 codes (at `scale`) → float plane bytes (inverse of [`to_codes`]). Codes are clamped to the
/// valid range before decoding (forged/quantized inputs saturate rather than wrap). One table load per sample
/// for every float dtype ([`Lut::dec_f32`]/[`Lut::dec16`]); a non-float `dtype` yields zero bytes.
pub fn from_codes<T: Copy + Into<i64>>(codes: &[T], dtype: Dtype, scale: u32) -> Vec<u8> {
    let mut out = vec![0u8; codes.len() * dtype.size()];
    from_codes_into(codes, dtype, scale, &mut out);
    out
}

/// [`from_codes`] into `out` (`codes.len() * esz` bytes; every byte is overwritten, so a recycled buffer is fine).
pub fn from_codes_into<T: Copy + Into<i64>>(codes: &[T], dtype: Dtype, scale: u32, out: &mut [u8]) {
    let lut = lut(scale);
    let idx = |code: T| (code.into().clamp(-(MAX_CODE as i64), MAX_CODE as i64) + MAX_CODE as i64) as usize;
    match dtype {
        Dtype::F32 => {
            let t = lut.dec_f32();
            for (o, &c) in out.as_chunks_mut::<4>().0.iter_mut().zip(codes) {
                *o = t.get(idx(c)).copied().unwrap_or(0.0).to_le_bytes();
            }
        }
        Dtype::F16 | Dtype::Bf16 => {
            let t = lut.dec16(dtype == Dtype::Bf16);
            for (o, &c) in out.as_chunks_mut::<2>().0.iter_mut().zip(codes) {
                *o = t.get(idx(c)).copied().unwrap_or(0).to_le_bytes();
            }
        }
        _ => out.fill(0),
    }
}

#[cfg(test)]
mod tests {
    /// The low-end floor is the ROUNDING boundary of the zero code, not `2^-32` itself: a value exactly at
    /// `2^-32` (and anything up to the first half-step above it) encodes to 0 and decodes to 0.0, the smallest
    /// non-zero code decodes to `2^(-32 + 1/scale)`, and values from the half-step up encode non-zero.
    #[test]
    fn floor_value_rounds_to_the_zero_code() {
        let floor = f32::from_bits((127 - 32) << 23); // exactly 2^-32
        for scale in [1u32, 16, 64, 256] {
            assert_eq!(encode_value(floor, scale), 0, "2^-32 at scale {scale}");
            assert_eq!(decode_value(0, scale), 0.0);
            let first = decode_value(1, scale);
            assert_eq!(encode_value(first, scale), 1, "smallest non-zero code round-trips at scale {scale}");
            assert!(first > floor && (first as f64) <= (floor as f64) * 2f64.powf(1.0 / scale as f64) * 1.0001);
            // Above the half-step the value is non-zero (nearest code is 1, not 0).
            let above = (floor as f64 * 2f64.powf(0.75 / scale as f64)) as f32;
            assert_eq!(encode_value(above, scale), 1, "0.75 step above the floor at scale {scale}");
            // Everything strictly below 2^-32 is the zero code too.
            assert_eq!(encode_value(f32::from_bits((127 - 33) << 23), scale), 0);
        }
    }

    use super::*;

    /// Round-trip of the log transform at DEFAULT_SCALE: bounded relative error across a wide HDR
    /// range, both signs.
    #[test]
    fn log_roundtrip_relative_bounded() {
        let mut v = 1e-9f32;
        while v < 1e8 {
            for s in [1.0f32, -1.0] {
                let x = v * s;
                let y = decode_value(encode_value(x, DEFAULT_SCALE), DEFAULT_SCALE);
                let rel = (y - x).abs() / x.abs();
                assert!(rel < 0.01, "relative error {rel} too large at {x} (got {y})");
                assert_eq!(y.is_sign_negative(), x.is_sign_negative(), "sign preserved at {x}");
            }
            v *= 1.7;
        }
    }

    /// Finer scale ⇒ smaller worst-case round-trip error (the HTJ2K-codes quality knob). Swept over
    /// many HDR magnitudes so the result doesn't hinge on a value that happens to land on a code edge.
    #[test]
    fn scale_controls_precision() {
        let worst = |scale: u32| {
            let mut m = 0.0f32;
            let mut v = 1.3f32;
            for _ in 0..20 {
                let r = decode_value(encode_value(v, scale), scale);
                m = m.max((r - v).abs() / v);
                v *= 1.37;
            }
            m
        };
        let coarse = worst(16);
        let fine = worst(256);
        assert!(fine < coarse, "finer scale must reduce worst error ({fine} !< {coarse})");
        assert!(fine < 0.005, "scale=256 should be near-lossless, got {fine}");
    }

    /// Corner cases must be bounded and never panic.
    #[test]
    fn corner_values() {
        assert_eq!(encode_value(0.0, DEFAULT_SCALE), 0);
        assert_eq!(encode_value(-0.0, DEFAULT_SCALE), 0);
        assert_eq!(decode_value(0, DEFAULT_SCALE), 0.0);
        assert_eq!(encode_value(f32::NAN, DEFAULT_SCALE), 0);
        assert_eq!(encode_value(f32::INFINITY, DEFAULT_SCALE), MAX_CODE);
        assert_eq!(encode_value(f32::NEG_INFINITY, DEFAULT_SCALE), -MAX_CODE);
        assert_eq!(encode_value(1e-20, DEFAULT_SCALE), 0);
        assert_eq!(encode_value(f32::MAX, DEFAULT_SCALE), MAX_CODE);
        let big = decode_value(encode_value(f32::INFINITY, DEFAULT_SCALE), DEFAULT_SCALE);
        assert!(big.is_finite() && big > 1e8);
        // Forged magnitudes and out-of-range scales saturate instead of overflowing the exponent.
        assert_eq!(decode_value(i32::MAX, 1), 2f32.powi(32));
        assert_eq!(decode_value(i32::MIN, 0), -2f32.powi(32));
        assert_eq!(decode_value(5, u32::MAX), decode_value(5, 256));
    }

    /// Codes sitting on a whole stop are EXACT powers of two (the exponent add is exact, T[0] == 1).
    #[test]
    fn whole_stops_are_exact_powers_of_two() {
        for scale in [1u32, 16, 256] {
            for a in -32i32..=32 {
                let code = (a + EXP_BIAS) * scale as i32;
                if code == 0 {
                    continue; // code 0 is the zero value
                }
                assert_eq!(decode_value(code, scale), 2f32.powi(a), "scale {scale} stop {a}");
            }
        }
    }

    /// decode∘encode∘decode is stable for EVERY code at representative scales: encode(decode(c)) == c,
    /// so the table math and the threshold math are mutually consistent (the f32 rounding of `T[r]`
    /// is ~1e-7 relative, a code step is >= 2.7e-3, so no code can flip).
    #[test]
    fn every_code_roundtrips() {
        for scale in [1u32, 16, 64, 256] {
            for c in 1..=EXP_SPAN * scale as i32 {
                for sgn in [1, -1] {
                    let code = c * sgn;
                    assert_eq!(encode_value(decode_value(code, scale), scale), code, "scale {scale} code {code}");
                }
            }
        }
    }

    /// The deterministic primitives agree with the std libm to within 1 f32 ULP (sanity — std is the
    /// loose oracle; our result is the platform-independent one) and hit exact anchors.
    #[test]
    fn exp2f_log2f_match_std() {
        assert_eq!(exp2f(0.5), 2f32.sqrt()); // sqrt is correctly rounded: bit-exact anchor
        assert_eq!(exp2f(3.0), 8.0);
        assert_eq!(exp2f(-1.0), 0.5);
        assert_eq!(exp2f(200.0), f32::INFINITY);
        assert_eq!(exp2f(-200.0), 0.0);
        assert!(exp2f(f32::NAN).is_nan());
        assert_eq!(log2f(8.0), 3.0);
        assert_eq!(log2f(1.0), 0.0);
        assert_eq!(log2f(0.0), f32::NEG_INFINITY);
        assert!(log2f(-1.0).is_nan());
        let ulp = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
        let mut x = -40.0f32;
        while x < 40.0 {
            assert!(ulp(exp2f(x), 2f32.powf(x)) <= 1, "exp2f({x}) = {} vs {}", exp2f(x), 2f32.powf(x));
            x += 0.0371;
        }
        let mut v = 1e-30f32;
        while v < 1e30 {
            let (a, b) = (log2f(v), v.log2());
            assert!((a - b).abs() <= 2e-6 * b.abs().max(1.0), "log2f({v}) = {a} vs {b}");
            v *= 1.9;
        }
    }

    /// The decode table is monotonic, starts at exactly 1, stays below 2 and within 1 ULP of std.
    #[test]
    fn exp2_table_is_sane() {
        for scale in [16u32, 100, 256] {
            let t = exp2_table(scale);
            assert_eq!(t.len(), scale as usize);
            assert_eq!(t[0], 1.0);
            assert!(t.windows(2).all(|w| w[0] < w[1]) && t[t.len() - 1] < 2.0);
            for (r, &v) in t.iter().enumerate() {
                let want = 2f64.powf(r as f64 / scale as f64) as f32;
                assert!((v.to_bits() as i64 - want.to_bits() as i64).abs() <= 1, "scale {scale} r {r}");
            }
        }
    }

    /// Encode an `f32` value into `esz` bytes of `dtype` (test helper).
    fn write_f32(out: &mut [u8], dtype: Dtype, v: f32) {
        match dtype {
            Dtype::F32 => out[..4].copy_from_slice(&v.to_le_bytes()),
            Dtype::Bf16 => out[..2].copy_from_slice(&f32_to_bf16(v).to_le_bytes()),
            Dtype::F16 => out[..2].copy_from_slice(&f32_to_half(v).to_le_bytes()),
            _ => {}
        }
    }

    /// The table paths of the plane API are bit-identical to the scalar `encode_value`/`decode_value`: EVERY
    /// f16/bf16 pattern encodes the same, EVERY code decodes (and narrows) the same, and the binned F32 encode
    /// agrees with the plain threshold count over a dense value sweep (incl. mantissas on bin edges).
    #[test]
    fn table_paths_match_scalar_exhaustively() {
        let all16: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let codes: Vec<i32> = (-MAX_CODE..=MAX_CODE).collect();
        for scale in [7u32, 256] {
            let l = lut(scale);
            for dt in [Dtype::F16, Dtype::Bf16] {
                let widen = if dt == Dtype::F16 { half_to_f32 } else { bf16_to_f32 };
                let got = to_codes::<i32>(&all16, dt, scale);
                for (b, &g) in got.iter().enumerate() {
                    assert_eq!(g, l.encode(widen(b as u16)), "{dt:?} scale {scale} bits {b:#06x}");
                }
                let narrow = if dt == Dtype::F16 { f32_to_half } else { f32_to_bf16 };
                let back = from_codes(&codes, dt, scale);
                for (c, o) in codes.iter().zip(back.as_chunks::<2>().0.iter()) {
                    assert_eq!(u16::from_le_bytes([o[0], o[1]]), narrow(l.decode(*c)), "{dt:?} scale {scale} code {c}");
                }
            }
            let back = from_codes(&codes, Dtype::F32, scale);
            for (c, o) in codes.iter().zip(back.as_chunks::<4>().0.iter()) {
                assert_eq!(
                    f32::from_le_bytes([o[0], o[1], o[2], o[3]]).to_bits(),
                    l.decode(*c).to_bits(),
                    "F32 scale {scale} code {c}"
                );
            }
            // Binned F32 encode vs the unbinned partition_point count.
            let slow = |v: f32| {
                let bits = v.abs().to_bits();
                let f = 1.0 + (bits & 0x7f_ffff) as f64 / 8_388_608.0;
                (((bits >> 23) as i32 - 127 + EXP_BIAS) * scale as i32 + l.thr.partition_point(|&t| t <= f) as i32)
                    * if v < 0.0 { -1 } else { 1 }
            };
            for e in 0..64u32 {
                for m in (0..(1u32 << 23)).step_by(9973).chain((0..(1 << BIN_BITS)).map(|b| b << (23 - BIN_BITS))) {
                    let v = f32::from_bits(((e + 95) << 23) | m);
                    assert_eq!(l.encode(v), slow(v), "scale {scale} bits {:#x}", v.to_bits());
                }
            }
        }
    }

    /// Plane round-trip through the dtype byte conversions for every float dtype.
    #[test]
    fn plane_roundtrip_floats() {
        for dt in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
            let esz = dt.size();
            let vals: Vec<f32> = (1..=64).map(|i| i as f32 * 0.5).collect();
            let mut plane = vec![0u8; vals.len() * esz];
            for (o, &v) in plane.chunks_exact_mut(esz).zip(&vals) {
                write_f32(o, dt, v);
            }
            let codes = to_codes::<i64>(&plane, dt, DEFAULT_SCALE);
            let back = from_codes(&codes, dt, DEFAULT_SCALE);
            for (i, (o, &v)) in back.chunks_exact(esz).zip(&vals).enumerate() {
                let r = read_f32(o, dt);
                let rel = (r - v).abs() / v.abs();
                let tol = if dt == Dtype::F32 { 0.01 } else { 0.03 };
                assert!(rel < tol, "{dt:?} sample {i}: {v} -> {r} (rel {rel})");
            }
        }
    }
}
