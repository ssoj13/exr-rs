//! Global motion compensation (one MV per P-frame) — the camera panning/shake case.
//!
//! # Why global, not block-based
//! A global integer shift covers the most common VFX case (pan/shake) with minimal risk:
//! no per-block state, no block edges, only edge clamping. Block-based MC is a future
//! improvement. **Lossless for any MV** (reconstruction is exact); MV only reduces the residual → ratio.
//! Deterministic (integer SAD search with a fixed tie-break).

use exv_format::Dtype;

/// Expected byte length of a `w×h` plane of `esz`-byte elements (checked; `None` on usize overflow).
fn plane_bytes(w: u32, h: u32, esz: usize) -> Option<usize> {
    (w as usize).checked_mul(h as usize)?.checked_mul(esz)
}

/// Shift a plane by `(dx,dy)`: `out[y][x] = prev[clamp(y-dy)][clamp(x-dx)]` (edge clamping).
/// The thus-shifted `prev` spatially aligns with `cur` under motion by `(dx,dy)`.
///
/// `None` when `prev.len() != w*h*esz` (inconsistent / hostile plane): the caller must fail, NOT carry on
/// with an unshifted plane — the encoder would then store a non-zero MV for a plane that was never
/// shifted and the decoder would shift it, silently breaking losslessness.
///
/// Row-wise: every output row is one `copy_from_slice` for the in-range middle span plus two edge-pixel
/// fills, all index math in `usize`/`i64` (planes beyond 2^31 pixels are legal).
pub fn shift_plane(prev: &[u8], w: u32, h: u32, dtype: Dtype, dx: i32, dy: i32) -> Option<Vec<u8>> {
    let mut out = vec![0u8; prev.len()];
    shift_into(prev, w, h, dtype, dx, dy, &mut out)?;
    Some(out)
}

/// [`shift_plane`] into the caller's `out` (`prev.len()` bytes, e.g. a pooled plane: every byte is overwritten, so its
/// previous contents do not matter). `None` when `prev` or `out` is not `w*h*esz` bytes.
pub fn shift_into(prev: &[u8], w: u32, h: u32, dtype: Dtype, dx: i32, dy: i32, out: &mut [u8]) -> Option<()> {
    let esz = dtype.size();
    if plane_bytes(w, h, esz) != Some(prev.len()) || out.len() != prev.len() {
        return None;
    }
    if prev.is_empty() {
        return Some(());
    }
    let hu = h as usize;
    let row = w as usize * esz;
    // Output x in [lo, hi) maps to source x-dx inside the row; x < lo clamps to pixel 0, x >= hi to pixel w-1.
    let lo = (dx as i64).clamp(0, w as i64) as usize;
    let hi = (w as i64 + dx as i64).clamp(0, w as i64) as usize;
    let (l_end, m_end) = (lo * esz, hi.max(lo) * esz); // byte ends of the left clamp and the in-range middle
    // `lo - dx` = first mapped source x (in [0, w] by construction of lo/hi).
    let s0 = (lo as i64 - dx as i64).max(0) as usize * esz;
    for (y, dst) in out.chunks_exact_mut(row).enumerate() {
        let sy = (y as i64 - dy as i64).clamp(0, hu as i64 - 1) as usize;
        let src = prev.get(sy * row..(sy + 1) * row)?;
        let (first, last) = (src.get(..esz)?, src.get(row - esz..)?);
        let (left, rest) = dst.split_at_mut_checked(l_end)?;
        let (mid, right) = rest.split_at_mut_checked(m_end - l_end)?;
        left.chunks_exact_mut(esz).for_each(|px| px.copy_from_slice(first));
        if !mid.is_empty() {
            mid.copy_from_slice(src.get(s0..s0 + mid.len())?);
        }
        right.chunks_exact_mut(esz).for_each(|px| px.copy_from_slice(last));
    }
    Some(())
}
/// Little-endian unsigned value of one `N`-byte element (`N <= 8`; `c.len() == N`).
#[inline(always)]
fn elem<const N: usize>(c: &[u8]) -> u64 {
    let mut v = [0u8; 8];
    v[..N].copy_from_slice(c);
    u64::from_le_bytes(v)
}

/// SAD of the sampled pixels (even rows, even columns) of `cur` against `prev` shifted by `(dx, dy)` with edge
/// clamping: `Σ |cur[y][x] - prev[clamp(y-dy)][clamp(x-dx)]|` over even `y`, `x`, values read as unsigned `N`-byte
/// elements. Both planes are exactly `w*h` elements ([`search_global_mv`] validated that once). Row-wise: per row the
/// columns split into the left clamp (all read `prev` pixel 0), the in-range middle (two zipped slices, no index math
/// or bounds checks per pixel) and the right clamp (all read pixel `w-1`).
fn sad<const N: usize>(cur: &[u8], prev: &[u8], w: usize, h: usize, dx: i32, dy: i32) -> u64 {
    let row = w * N;
    // Source column of output x is x - dx; it is in range for x in [lo, hi), clamps to 0 below, to w-1 above.
    let lo = (dx as i64).clamp(0, w as i64) as usize;
    let hi = (w as i64 + dx as i64).clamp(0, w as i64) as usize;
    let hi = hi.max(lo);
    let mut total = 0u64;
    for (y, crow) in cur.chunks_exact(row).enumerate().step_by(2) {
        let sy = (y as i64 - dy as i64).clamp(0, h as i64 - 1) as usize;
        let Some(prow) = prev.get(sy * row..(sy + 1) * row) else { continue };
        let (Some(first), Some(last)) = (prow.get(..N), prow.get(row - N..)) else { continue };
        let (first, last) = (elem::<N>(first), elem::<N>(last));
        // Even columns of each span (the sampling is on the OUTPUT x, so the first even x >= span start).
        let even = |a: usize| a + (a & 1);
        for c in crow.as_chunks::<N>().0.iter().take(lo).step_by(2) {
            total += elem::<N>(c).abs_diff(first);
        }
        let (a, b) = (even(lo), hi);
        if a < b {
            // `a - dx .. b - dx` are source columns, inside [0, w) by the definition of lo/hi.
            let s0 = (a as i64 - dx as i64) as usize;
            if let (Some(cs), Some(ps)) = (crow.get(a * N..b * N), prow.get(s0 * N..(s0 + (b - a)) * N)) {
                total += cs.as_chunks::<N>().0.iter().step_by(2).zip(ps.as_chunks::<N>().0.iter().step_by(2)).map(|(x, y)| elem::<N>(x).abs_diff(elem::<N>(y))).sum::<u64>();
            }
        }
        if let Some(tail) = crow.get(even(hi.max(lo)) * N..) {
            for c in tail.as_chunks::<N>().0.iter().step_by(2) {
                total += elem::<N>(c).abs_diff(last);
            }
        }
    }
    total
}

/// Find the best global MV in the window `[-search, search]²` by minimum SAD (values as uint,
/// stride 2 for speed). Deterministic: tie-break = smaller `|dx|+|dy|`, then traversal order.
/// `search <= 0` → `(0,0)` (= ordinary temporal-delta).
///
/// The candidates' SADs are independent integer sums, so they run in parallel (rayon) and a tuple `min` picks the
/// winner with the serial scan's tie-break: the result does not depend on the thread count, and memory is
/// O(threads) however large `search` is.
pub fn search_global_mv(
    cur: &[u8],
    prev: &[u8],
    w: u32,
    h: u32,
    dtype: Dtype,
    search: i32,
) -> (i32, i32) {
    use rayon::prelude::*;
    let esz = dtype.size();
    // The container stores the MV as a pair of i16 on the wire (see `container.rs`). Clamp the search
    // radius to the i16 range so the chosen MV is ALWAYS representable there: otherwise the encoder
    // would shift `prev` by the full i32 MV while the decoder reads the i16-truncated value and
    // shifts by a different amount — silently breaking the "lossless for any MV" guarantee for
    // |MV| > 32767. This is the single source of MVs, so bounding it here fixes the class. Real search
    // radii are tiny; only a pathological/forged `--search` is affected (CLASS-G).
    let search = search.min(i16::MAX as i32);
    // Both planes must be exactly `w*h` elements: that is the one validation `sad` relies on (it slices rows,
    // never indexes pixels). A mismatch means there is nothing meaningful to search → (0,0); `mc_shift` then
    // rejects the plane.
    let expected = plane_bytes(w, h, esz);
    if search <= 0 || cur.len() != prev.len() || expected != Some(cur.len()) || cur.is_empty() {
        return (0, 0);
    }
    let (wu, hu) = (w as usize, h as usize);
    let side = (2 * search + 1) as usize;
    let sad_of = |i: usize| {
        let (dy, dx) = ((i / side) as i32 - search, (i % side) as i32 - search);
        match esz {
            1 => sad::<1>(cur, prev, wu, hu, dx, dy),
            2 => sad::<2>(cur, prev, wu, hu, dx, dy),
            _ => sad::<4>(cur, prev, wu, hu, dx, dy), // dtype widths are 1, 2 or 4
        }
    };
    // Lexicographic minimum of (SAD, |dx|+|dy|, traversal index) = the serial scan's winner (smaller SAD, then
    // smaller cost, then the candidate met FIRST), in O(threads) memory however large `search` is.
    let best = (0..side * side)
        .into_par_iter()
        .map(|i| {
            let (dy, dx) = ((i / side) as i32 - search, (i % side) as i32 - search);
            (sad_of(i), dx.abs() + dy.abs(), i)
        })
        .min();
    best.map_or((0, 0), |(_, _, i)| ((i % side) as i32 - search, (i / side) as i32 - search))
}

/// Per-element `(a + b) >> 1` of two equally long planes of `esz`-byte elements, computed in `u64` (a U32/F32 sum
/// cannot overflow): the reversible integer bit-pattern MEAN of the B prediction. `None` when the lengths differ
/// or are not whole elements. Const-width loops (a runtime-width byte loop was ~15x slower).
pub fn avg_plane(a: &[u8], b: &[u8], esz: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; a.len()];
    avg_into(&mut out, a, b, esz)?;
    Some(out)
}

/// [`avg_plane`] into the caller's `out` (`a.len()` bytes, every byte overwritten: a pooled plane may be passed).
/// Fixed-size `[u8; N]` chunks (`as_chunks`): no per-element length or bounds checks in the loop.
pub fn avg_into(out: &mut [u8], a: &[u8], b: &[u8], esz: usize) -> Option<()> {
    fn run<const N: usize>(out: &mut [u8], a: &[u8], b: &[u8]) {
        let (o, x, y) = (out.as_chunks_mut::<N>().0, a.as_chunks::<N>().0, b.as_chunks::<N>().0);
        for ((o, x), y) in o.iter_mut().zip(x).zip(y) {
            let m = ((elem::<N>(x) + elem::<N>(y)) >> 1).to_le_bytes();
            o.copy_from_slice(&m[..N]);
        }
    }
    if a.len() != b.len() || out.len() != a.len() || esz == 0 || !a.len().is_multiple_of(esz) {
        return None;
    }
    match esz {
        1 => run::<1>(out, a, b),
        2 => run::<2>(out, a, b),
        4 => run::<4>(out, a, b),
        _ => return None, // dtype widths are 1, 2 or 4
    }
    Some(())
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Per-pixel reference of the documented semantics (what the row-wise fast path must equal).
    fn shift_naive(prev: &[u8], w: usize, h: usize, esz: usize, dx: i32, dy: i32) -> Vec<u8> {
        let mut out = vec![0u8; prev.len()];
        for y in 0..h {
            let sy = (y as i64 - dy as i64).clamp(0, h as i64 - 1) as usize;
            for x in 0..w {
                let sx = (x as i64 - dx as i64).clamp(0, w as i64 - 1) as usize;
                out[(y * w + x) * esz..][..esz].copy_from_slice(&prev[(sy * w + sx) * esz..][..esz]);
            }
        }
        out
    }

    #[test]
    fn shift_roundtrip_zero_is_identity() {
        let p: Vec<u8> = (0..64u16).flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(shift_plane(&p, 8, 8, Dtype::U16, 0, 0), Some(p));
    }

    /// Row-wise copy == per-pixel clamp for every sign / magnitude, including |shift| >= size.
    #[test]
    fn shift_matches_naive_clamp() {
        let (w, h) = (7usize, 5usize);
        for dtype in [Dtype::U8, Dtype::U16, Dtype::F32] {
            let esz = dtype.size();
            let p: Vec<u8> = (0..w * h * esz).map(|i| (i * 31 % 251) as u8).collect();
            for dx in [-20, -7, -3, -1, 0, 1, 3, 6, 7, 20] {
                for dy in [-9, -5, -2, 0, 1, 4, 5, 9] {
                    let got = shift_plane(&p, w as u32, h as u32, dtype, dx, dy).expect("shift");
                    assert_eq!(got, shift_naive(&p, w, h, esz, dx, dy), "{dtype:?} dx={dx} dy={dy}");
                    // A recycled (stale) destination is fully overwritten.
                    let mut dirty = vec![0xab; p.len()];
                    shift_into(&p, w as u32, h as u32, dtype, dx, dy, &mut dirty).expect("shift_into");
                    assert_eq!(dirty, got, "stale destination {dtype:?} dx={dx} dy={dy}");
                }
            }
        }
    }

    /// A plane whose length disagrees with `w*h*esz` is rejected (None), never "shifted" as identity.
    #[test]
    fn short_plane_is_rejected() {
        let short = vec![0u8; 10];
        assert_eq!(shift_plane(&short, 100, 100, Dtype::U16, 1, 1), None);
        let other = vec![0u8; 10];
        assert_eq!(search_global_mv(&short, &other, 100, 100, Dtype::U16, 4), (0, 0));
    }

    /// The per-pixel SAD the row-wise [`sad`] replaced (verbatim semantics: even rows/columns, clamped source).
    fn sad_naive(cur: &[u8], prev: &[u8], w: usize, h: usize, esz: usize, dx: i32, dy: i32) -> u64 {
        let rd = |p: &[u8], i: usize| (0..esz).fold(0i64, |v, k| v | (p[i * esz + k] as i64) << (8 * k));
        let mut s = 0u64;
        for y in (0..h).step_by(2) {
            let sy = (y as i64 - dy as i64).clamp(0, h as i64 - 1) as usize;
            for x in (0..w).step_by(2) {
                let sx = (x as i64 - dx as i64).clamp(0, w as i64 - 1) as usize;
                s += (rd(cur, y * w + x) - rd(prev, sy * w + sx)).unsigned_abs();
            }
        }
        s
    }

    /// Row-wise SAD == per-pixel SAD for every element size, odd/even sizes and shifts beyond the plane.
    #[test]
    fn sad_matches_naive() {
        for (w, h) in [(1usize, 1usize), (2, 3), (7, 5), (8, 8), (13, 6)] {
            for esz in [1usize, 2, 4] {
                let n = w * h * esz;
                let cur: Vec<u8> = (0..n).map(|i| (i * 37 % 251) as u8).collect();
                let prev: Vec<u8> = (0..n).map(|i| (i * 91 % 241) as u8).collect();
                for dx in [-20, -13, -8, -3, -1, 0, 1, 2, 5, 7, 8, 13, 20] {
                    for dy in [-9, -5, -1, 0, 1, 2, 6, 9] {
                        let got = match esz {
                            1 => sad::<1>(&cur, &prev, w, h, dx, dy),
                            2 => sad::<2>(&cur, &prev, w, h, dx, dy),
                            _ => sad::<4>(&cur, &prev, w, h, dx, dy),
                        };
                        assert_eq!(got, sad_naive(&cur, &prev, w, h, esz, dx, dy), "{w}x{h} esz{esz} d({dx},{dy})");
                    }
                }
            }
        }
    }

    /// `avg_plane` == the per-element `(a + b) >> 1` in u64 for every width, and rejects ragged input.
    #[test]
    fn avg_matches_naive() {
        for esz in [1usize, 2, 4] {
            let n = 997 * esz;
            let a: Vec<u8> = (0..n).map(|i| if i % 17 < 6 { 0xff } else { (i * 53 % 256) as u8 }).collect(); // saturating sums too
            let b: Vec<u8> = (0..n).map(|i| (i * 29 % 256) as u8).collect();
            let got = avg_plane(&a, &b, esz).expect("avg");
            for ((o, x), y) in got.chunks_exact(esz).zip(a.chunks_exact(esz)).zip(b.chunks_exact(esz)) {
                let rd = |c: &[u8]| c.iter().enumerate().fold(0u64, |v, (k, &b)| v | (b as u64) << (8 * k));
                assert_eq!(rd(o), (rd(x) + rd(y)) >> 1);
            }
        }
        assert_eq!(avg_plane(&[0u8; 8], &[0u8; 6], 2), None);
        assert_eq!(avg_plane(&[0u8; 7], &[0u8; 7], 2), None);
        assert_eq!(avg_into(&mut [0u8; 6], &[0u8; 8], &[0u8; 8], 2), None, "short output rejected");
        assert_eq!(shift_into(&[0u8; 8], 2, 2, Dtype::U16, 1, 0, &mut [0u8; 7]), None, "short output rejected");
    }

    #[test]
    fn finds_known_shift() {
        // prev — a gradient; cur = prev shifted by (2,1). The search should find (2,1).
        let (w, h) = (32u32, 24u32);
        let prev: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| (((x + 2 * y) & 0xff) as u16).to_le_bytes()))
            .collect();
        let cur = shift_plane(&prev, w, h, Dtype::U16, 2, 1).expect("shift");
        assert_eq!(search_global_mv(&cur, &prev, w, h, Dtype::U16, 4), (2, 1));
    }
}
