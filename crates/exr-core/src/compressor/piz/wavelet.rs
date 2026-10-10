//! 2-D Haar wavelet transform for the PIZ codec — 1:1 port of OpenEXR's
//! `ImfWav.cpp` (`wav2Encode` / `wav2Decode`, `wenc14`/`wdec14`/`wenc16`/
//! `wdec16`).
//!
//! The wavelet is applied **in place** to a `u16` buffer that has already been
//! range-compressed by the PIZ forward LUT, per channel region.  It is an exact
//! integer transform: every `encode` is inverted bit-for-bit by `decode`, so it
//! contributes no rounding and the whole PIZ pipeline stays lossless.
//!
//! # Two basis families (chosen by `max_value`)
//!
//! * **14-bit** (`wenc14`/`wdec14`) — used when the largest transformed
//!   value `mx < (1 << 14)`.  Plain signed arithmetic without modulo; gives the
//!   best Huffman ratios but only valid for 14-bit data.
//! * **16-bit** (`wenc16`/`wdec16`) — used otherwise.  Modulo arithmetic
//!   that wraps within the 16-bit range, valid for the full `u16` domain.
//!
//! The selection mirrors C++ exactly: `w14 = (mx < (1 << 14))`.
//!
//! # Layout (offsets, not pointers)
//!
//! C++ walks the buffer with raw pointers (`px`, `p01`, `p10`, `p11`) advanced
//! by `ox`/`oy` strides.  Here the same traversal is expressed as integer
//! indices into the slice (`pos_x`, `pos_right`, `pos_top`, `pos_top_right`),
//! which keeps every access bounds-checked.  `count = (count_x, count_y)` is the
//! channel region's pixel extent; `offset = (ox, oy)` is the per-axis stride
//! (in `u16` elements) within the interleaved per-channel buffer.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfWav.cpp>
//!
//! The behavioural target (C++ 3.4.15) decodes
//! through OpenEXRCore, not this legacy `Imf` file — see `internal_piz.c:120-461`
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_piz.c#L120-L461>,
//! whose `wdec14_4` decoder [`wdec14_4`] is now ported literally rather
//! than through the legacy `ImfWav.cpp` two-stage `wdec14`+`wdec14` this module
//! otherwise mirrors for the 16-bit basis and the encode direction.

use crate::error::{ExrError, Result};

// ---------------------------------------------------------------------------
// 14-bit basis (no modulo) — port of `wenc14` / `wdec14`
// ---------------------------------------------------------------------------

/// 14-bit forward wavelet step — port of `wenc14`.
///
/// Returns `(l, h)` = (average, difference) of `a` and `b` using plain signed
/// arithmetic.  Wrapping `as i16`/`as u16` casts reproduce C++'s `short`
/// overflow behaviour exactly.
#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let as_ = a as i16;
    let bs = b as i16;

    // C++: short ms = (as + bs) >> 1;  short ds = as - bs;
    let ms = ((as_ as i32 + bs as i32) >> 1) as i16;
    let ds = (as_ as i32 - bs as i32) as i16;

    (ms as u16, ds as u16)
}

/// 14-bit inverse wavelet step — port of `wdec14`.
///
/// Reconstructs `(a, b)` from `(l, h)`.  The `(hi & 1) + (hi >> 1)` term and the
/// signed truncations match C++ bit-for-bit.
#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let ls = l as i16;
    let hs = h as i16;

    let hi = hs as i32;
    // C++: int ai = ls + (hi & 1) + (hi >> 1);
    let ai = ls as i32 + (hi & 1) + (hi >> 1);

    let as_ = ai as i16; // short as = ai;
    let bs = (ai - hi) as i16; // short bs = ai - hi;

    (as_ as u16, bs as u16)
}

/// 14-bit inverse wavelet **2x2 block** step — 1:1 port of `wdec14_4`
/// (`OpenEXRCore/internal_piz.c:139-172`,
/// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_piz.c#L139-L172>).
///
/// The interior (non-boundary) 2-D decode step used to chain two
/// [`wdec14`] calls per axis (the legacy `ImfWav.cpp` two-stage decode this
/// module previously ported), truncating the intermediate sums `ai`/`bi` to
/// `i16` **between** the row and column stages. `wdec14_4` is the actual
/// OpenEXRCore decoder: it keeps every intermediate as a full `int` across
/// both stages and truncates to `u16` only once, at the very end. The two
/// forms disagree whenever an intermediate would overflow `i16` before that
/// final truncation (verified repro: `px == p01 == p10 == p11 == 0x7fff`,
/// `maxValue < 16384`) — unreachable from any of our own valid 14-bit
/// encodes, but reachable from a corrupted/adversarial PIZ chunk, where the
/// old code silently diverged from the C++ decoder instead of matching it
/// byte-for-byte.
///
/// Takes/returns `(px, p01, p10, p11)` in that fixed order (matching the
/// C++ variable names and its final `*px = ai; *p01 = bi; *p10 = ci; *p11 =
/// di;` assignment).
#[inline]
fn wdec14_4(px: u16, p01: u16, p10: u16, p11: u16) -> (u16, u16, u16, u16) {
    // C++ pre-swap comment: px->a, p10->b, p01->c, p11->d.
    let a = px as i16 as i32;
    let b = p10 as i16 as i32;
    let c = p01 as i16 as i32;
    let d = p11 as i16 as i32;

    let i00 = a + (b & 1) + (b >> 1);
    let i10 = i00 - b;
    let i01 = c + (d & 1) + (d >> 1);
    let i11 = i01 - d;

    let ai = i00 + (i01 & 1) + (i01 >> 1);
    let bi = ai - i01;
    let ci = i10 + (i11 & 1) + (i11 >> 1);
    let di = ci - i11;

    (ai as u16, bi as u16, ci as u16, di as u16)
}

// ---------------------------------------------------------------------------
// 16-bit basis (modulo) — port of `wenc16` / `wdec16`
// ---------------------------------------------------------------------------

/// Number of significant bits in the modulo wavelet (`NBITS` in C++).
const NBITS: i32 = 16;
/// Additive offset applied before averaging (`A_OFFSET`).
const A_OFFSET: i32 = 1 << (NBITS - 1);
/// Offset added to the average when the difference is negative (`M_OFFSET`).
const M_OFFSET: i32 = 1 << (NBITS - 1);
/// Mask that wraps values into the 16-bit range (`MOD_MASK`).
const MOD_MASK: i32 = (1 << NBITS) - 1;

/// 16-bit (modulo) forward wavelet step — port of `wenc16`.
///
/// Works for the full 16-bit value range using modulo arithmetic.
#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let a = a as i32;
    let b = b as i32;

    let ao = (a + A_OFFSET) & MOD_MASK;
    let mut m = (ao + b) >> 1;
    let d = ao - b;

    if d < 0 {
        m = (m + M_OFFSET) & MOD_MASK;
    }
    let d = d & MOD_MASK;

    (m as u16, d as u16)
}

/// 16-bit (modulo) inverse wavelet step — port of `wdec16`.
#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let m = l as i32;
    let d = h as i32;

    let bb = (m - (d >> 1)) & MOD_MASK;
    let aa = (d + bb - A_OFFSET) & MOD_MASK;

    (aa as u16, bb as u16)
}

/// `true` when the modulo-free 14-bit basis should be used (`mx < (1 << 14)`).
#[inline]
fn is_14_bit(max_value: u16) -> bool {
    (max_value as i32) < (1 << 14)
}

// ---------------------------------------------------------------------------
// 2-D forward transform — port of `wav2Encode`
// ---------------------------------------------------------------------------

/// Apply the 2-D forward Haar wavelet transform in place — port of `wav2Encode`
/// (`ImfWav.cpp` lines 106-219).
///
/// `buffer` is one channel region; `count = (nx, ny)` is its pixel extent;
/// `offset = (ox, oy)` is the per-axis stride in `u16` elements; `max_value` is
/// the largest value present (selects the 14- vs 16-bit basis).
///
/// # Errors
/// Returns [`ExrError::Logic`] if any computed index would fall outside
/// `buffer` (would indicate a layout mismatch — never a panic / OOB access).
pub fn encode(
    buffer: &mut [u16],
    count: (usize, usize),
    offset: (usize, usize),
    max_value: u16,
) -> Result<()> {
    let (count_x, count_y) = count;
    let (ox, oy) = offset;
    let w14 = is_14_bit(max_value);
    let n = count_x.min(count_y);

    let mut p: usize = 1; // == 1 <<  level
    let mut p2: usize = 2; // == 1 << (level + 1)

    // Hierarchical loop on the smaller dimension n.
    while p2 <= n {
        // C++: py = in; ey = in + oy*(ny - p2); strides scaled by p / p2.
        let mut pos_y: usize = 0;
        // `ny - p2 >= 0` because `p2 <= n <= ny`.
        let end_y = oy * (count_y - p2);
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        // Y loop.
        while pos_y <= end_y {
            let mut pos_x = pos_y;
            let end_x = pos_y + ox * (count_x - p2);

            // X loop.
            while pos_x <= end_x {
                let p01 = pos_x + ox1;
                let p10 = pos_x + oy1;
                let p11 = p10 + ox1;

                idx_ok(buffer, &[pos_x, p01, p10, p11], "wav encode 2D")?;

                if w14 {
                    let (i00, i01) = wenc14(buffer[pos_x], buffer[p01]);
                    let (i10, i11) = wenc14(buffer[p10], buffer[p11]);
                    let (a, b) = wenc14(i00, i10);
                    let (c, d) = wenc14(i01, i11);
                    buffer[pos_x] = a;
                    buffer[p10] = b;
                    buffer[p01] = c;
                    buffer[p11] = d;
                } else {
                    let (i00, i01) = wenc16(buffer[pos_x], buffer[p01]);
                    let (i10, i11) = wenc16(buffer[p10], buffer[p11]);
                    let (a, b) = wenc16(i00, i10);
                    let (c, d) = wenc16(i01, i11);
                    buffer[pos_x] = a;
                    buffer[p10] = b;
                    buffer[p01] = c;
                    buffer[p11] = d;
                }

                pos_x += ox2;
            }

            // Encode (1D) odd column (still in Y loop).
            if count_x & p != 0 {
                let p10 = pos_x + oy1;
                idx_ok(buffer, &[pos_x, p10], "wav encode odd col")?;
                let (i00, h) = if w14 {
                    wenc14(buffer[pos_x], buffer[p10])
                } else {
                    wenc16(buffer[pos_x], buffer[p10])
                };
                buffer[p10] = h;
                buffer[pos_x] = i00;
            }

            pos_y += oy2;
        }

        // Encode (1D) odd line (must loop in X).
        if count_y & p != 0 {
            let mut pos_x = pos_y;
            let end_x = pos_y + ox * (count_x - p2);

            while pos_x <= end_x {
                let p01 = pos_x + ox1;
                idx_ok(buffer, &[pos_x, p01], "wav encode odd line")?;
                let (i00, h) = if w14 {
                    wenc14(buffer[pos_x], buffer[p01])
                } else {
                    wenc16(buffer[pos_x], buffer[p01])
                };
                buffer[p01] = h;
                buffer[pos_x] = i00;

                pos_x += ox2;
            }
        }

        // Next level.
        p = p2;
        p2 <<= 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// 2-D inverse transform — port of `wav2Decode`
// ---------------------------------------------------------------------------

/// Apply the 2-D inverse Haar wavelet transform in place — port of `wav2Decode`
/// (`ImfWav.cpp` lines 225-349).
///
/// Reverses [`encode`] exactly (same parameters, same basis selection).
///
/// # Errors
/// Returns [`ExrError::Logic`] if any computed index would fall outside
/// `buffer` (corrupt layout — never a panic / OOB access).
pub fn decode(
    buffer: &mut [u16],
    count: (usize, usize),
    offset: (usize, usize),
    max_value: u16,
) -> Result<()> {
    let (count_x, count_y) = count;
    let (ox, oy) = offset;
    let w14 = is_14_bit(max_value);
    let n = count_x.min(count_y);

    // Search max level: raise p past n, then back down two halvings.
    let mut p: usize = 1;
    while p <= n {
        p <<= 1;
    }
    p >>= 1;
    let mut p2 = p;
    p >>= 1;

    // Hierarchical loop on the smaller dimension n.
    while p >= 1 {
        let mut pos_y: usize = 0;
        let end_y = oy * (count_y - p2);
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        // Y loop.
        while pos_y <= end_y {
            let mut pos_x = pos_y;
            let end_x = pos_y + ox * (count_x - p2);

            // X loop.
            while pos_x <= end_x {
                let p01 = pos_x + ox1;
                let p10 = pos_x + oy1;
                let p11 = p10 + ox1;

                idx_ok(buffer, &[pos_x, p01, p10, p11], "wav decode 2D")?;

                if w14 {
                    // Single `wdec14_4` call, full-int intermediates —
                    // NOT two chained `wdec14` stages (see its doc comment).
                    let (a, c, b, d) =
                        wdec14_4(buffer[pos_x], buffer[p01], buffer[p10], buffer[p11]);
                    buffer[pos_x] = a;
                    buffer[p01] = c;
                    buffer[p10] = b;
                    buffer[p11] = d;
                } else {
                    let (i00, i10) = wdec16(buffer[pos_x], buffer[p10]);
                    let (i01, i11) = wdec16(buffer[p01], buffer[p11]);
                    let (a, c) = wdec16(i00, i01);
                    let (b, d) = wdec16(i10, i11);
                    buffer[pos_x] = a;
                    buffer[p01] = c;
                    buffer[p10] = b;
                    buffer[p11] = d;
                }

                pos_x += ox2;
            }

            // Decode (1D) odd column (still in Y loop).
            if count_x & p != 0 {
                let p10 = pos_x + oy1;
                idx_ok(buffer, &[pos_x, p10], "wav decode odd col")?;
                let (i00, b) = if w14 {
                    wdec14(buffer[pos_x], buffer[p10])
                } else {
                    wdec16(buffer[pos_x], buffer[p10])
                };
                buffer[p10] = b;
                buffer[pos_x] = i00;
            }

            pos_y += oy2;
        }

        // Decode (1D) odd line (must loop in X).
        if count_y & p != 0 {
            let mut pos_x = pos_y;
            let end_x = pos_y + ox * (count_x - p2);

            while pos_x <= end_x {
                let p01 = pos_x + ox1;
                idx_ok(buffer, &[pos_x, p01], "wav decode odd line")?;
                let (i00, b) = if w14 {
                    wdec14(buffer[pos_x], buffer[p01])
                } else {
                    wdec16(buffer[pos_x], buffer[p01])
                };
                buffer[p01] = b;
                buffer[pos_x] = i00;

                pos_x += ox2;
            }
        }

        // Next level.
        p2 = p;
        p >>= 1;
    }

    Ok(())
}

/// Bounds-check helper: confirm every index in `idxs` is within `buffer`.
///
/// The C++ pointer walk is structurally bounded by the loop conditions; this
/// guard makes the equivalent Rust safe even for a pathological/corrupt layout
/// (it returns an error rather than panicking on an out-of-range index).
#[inline]
fn idx_ok(buffer: &[u16], idxs: &[usize], ctx: &str) -> Result<()> {
    let len = buffer.len();
    for &i in idxs {
        if i >= len {
            return Err(ExrError::Logic(format!(
                "PIZ wavelet index {i} out of range (len {len}) in {ctx}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-step basis functions invert each other (14- and 16-bit).
    #[test]
    fn basis_round_trip() {
        let cases = [
            (13u16, 54u16),
            (0, 0),
            (16374, 16381),
            (1, 23),
            (65535, 0),
            (40000, 25000),
            (3, 0),
        ];
        for &(a, b) in &cases {
            let (l, h) = wenc14(a & 0x3fff, b & 0x3fff);
            assert_eq!(wdec14(l, h), (a & 0x3fff, b & 0x3fff));
            let (l, h) = wenc16(a, b);
            assert_eq!(wdec16(l, h), (a, b));
        }
    }

    /// `wdec14_4`'s repro value — `px == p01 == p10 ==
    /// p11 == 0x7fff` (an out-of-range 14-bit value, only reachable from
    /// corrupt/adversarial input) must decode to the C++ OpenEXRCore result,
    /// which differs from the old two-chained-`wdec14` result
    /// (`0x9fff,0xe000,0x6000,0x2000`) because that path truncates an
    /// intermediate sum to `i16` between stages.
    #[test]
    fn wdec14_4_matches_cpp_on_overflowing_input() {
        let (px, p01, p10, p11) = wdec14_4(0x7fff, 0x7fff, 0x7fff, 0x7fff);
        assert_eq!(
            (px, p01, p10, p11),
            (0x1fff, 0x6000, 0x6000, 0x2000),
            "must match internal_piz.c's wdec14_4, not the truncating two-stage wdec14 chain"
        );
    }

    /// `wdec14_4` inverts the `encode` interior 2x2 step (two chained
    /// [`wenc14`] calls per axis — only the *decode* direction has the
    /// truncation divergence) for every value a real 14-bit encode can
    /// produce.
    #[test]
    fn wdec14_4_inverts_forward_2x2_for_valid_14bit_values() {
        let mut state: u32 = 0xABCD_EF01;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) as u16) & 0x3fff
        };
        for _ in 0..2000 {
            let (px, p01, p10, p11) = (next(), next(), next(), next());

            // Forward, exactly as `encode`'s interior block:
            //   i00,i01 = wenc14(px, p01); i10,i11 = wenc14(p10, p11);
            //   a,b = wenc14(i00, i10);    c,d = wenc14(i01, i11);
            //   new_px=a, new_p10=b, new_p01=c, new_p11=d
            let (i00, i01) = wenc14(px, p01);
            let (i10, i11) = wenc14(p10, p11);
            let (a, b) = wenc14(i00, i10);
            let (c, d) = wenc14(i01, i11);

            let (dpx, dp01, dp10, dp11) = wdec14_4(a, c, b, d);
            assert_eq!(
                (dpx, dp01, dp10, dp11),
                (px, p01, p10, p11),
                "wdec14_4 must invert the forward 2x2 step"
            );
        }
    }

    /// Full 2-D transform round-trips for a spread of shapes and value ranges.
    #[test]
    fn transform_round_trip() {
        fn run(nx: usize, ny: usize, mask: u16) {
            let mut state: u32 = 0x1234_5678 ^ ((nx as u32) << 13) ^ (ny as u32);
            let mut data: Vec<u16> = (0..nx * ny)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((state >> 16) as u16) & mask
                })
                .collect();
            let max = *data.iter().max().unwrap_or(&0);
            let orig = data.clone();
            let offset = (1usize, nx);
            encode(&mut data, (nx, ny), offset, max).unwrap();
            decode(&mut data, (nx, ny), offset, max).unwrap();
            assert_eq!(data, orig, "round-trip failed for {nx}x{ny} mask {mask:#x}");
        }
        for &(nx, ny) in &[
            (1, 1),
            (2, 2),
            (32, 32),
            (7, 13),
            (13, 7),
            (1, 100),
            (100, 1),
            (37, 41),
        ] {
            run(nx, ny, 0x3fff); // 14-bit
            run(nx, ny, 0xffff); // 16-bit
        }
    }
}
