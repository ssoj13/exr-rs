//! Reversible integer 5/3 (LeGall) wavelet, **tile-local**, lossless.
//!
//! # Why it is reversible in a fixed width
//! Each lifting step is `a ±= f(neighbors)`, which is exactly undone by `a ∓= f(neighbors)` given
//! the same neighbors. Arithmetic is **wrapping in the element width** (mask = `2^(8·esz)-1`), as in
//! OpenEXR PIZ: values stay in the same byte width (no bit-depth growth → lossless without
//! expansion), while decorrelation is preserved (coefficients cluster near zero modulo). Intermediate
//! sums are computed in `u64` (for esz=4 the sum of two u32 does not fit in u32).
//!
//! # Tile-local (see `docs/archive/PLAN.md` §3.6)
//! One decomposition level per tile with **symmetric extension** at the edges. NO halo and NO
//! dependency between tiles → each tile decodes independently, bit-exact. Bit-exactness = per-tile
//! round-trip (NOT equality across different tilings — that is intentional).

use std::cell::Cell;
use std::collections::TryReserveError;

use crate::CodecError;
use exv_format::{AlignedBuf, DecodeError, Dtype};

/// `true` when `len` is exactly `w*h*esz` (checked: a forged geometry cannot wrap the product).
fn is_plane(len: usize, w: u32, h: u32, esz: usize) -> bool {
    (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(esz)) == Some(len)
}

/// Forward 2D transform: lifting over rows, then over columns. Output length = input length.
///
/// # Errors
/// [`CodecError::Dims`] when `plane` is not exactly `w*h*esz` bytes (the lifting indexes `y*w+x`, so a wrong
/// length is refused, never padded or truncated).
pub fn forward_2d(plane: &[u8], dtype: Dtype, w: u32, h: u32) -> Result<Vec<u8>, CodecError> {
    let mut out = vec![0u8; plane.len()];
    forward_2d_into(plane, dtype, w, h, &mut out)?;
    Ok(out)
}

/// [`forward_2d`] into `out` (`out.len() == plane.len()`; every byte is overwritten). The `u32` lifting grid is a
/// reused per-thread scratch, so a tile transform allocates nothing plane-sized.
///
/// # Errors
/// [`CodecError::Dims`] when `plane` or `out` is not exactly `w*h*esz` bytes.
pub fn forward_2d_into(plane: &[u8], dtype: Dtype, w: u32, h: u32, out: &mut [u8]) -> Result<(), CodecError> {
    let esz = dtype.size();
    if !is_plane(plane.len(), w, h, esz) || out.len() != plane.len() {
        return Err(CodecError::Dims("wavelet plane != w*h*esz"));
    }
    lift_via_grid(plane, esz, out, |g| fwd(g, w as usize, h as usize, width_mask(esz)))
        .map_err(|e| CodecError::Backend(format!("wavelet grid: {e}")))
}

/// Inverse 2D transform (inverse of [`forward_2d`]): columns, then rows.
///
/// # Errors
/// [`DecodeError::Corrupt`] when `coeffs` is not exactly `w*h*esz` bytes (a forged blob must not index out of
/// bounds or decode a zero-padded plane).
pub fn inverse_2d(coeffs: &[u8], dtype: Dtype, w: u32, h: u32) -> Result<Vec<u8>, DecodeError> {
    let mut out = vec![0u8; coeffs.len()];
    inverse_2d_into(coeffs, dtype, w, h, &mut out)?;
    Ok(out)
}

/// [`inverse_2d`] into `out` (`out.len() == coeffs.len()`; every byte is overwritten) — straight into the tile
/// buffer, with the lifting grid in a reused per-thread scratch.
///
/// # Errors
/// [`DecodeError::Corrupt`] when `coeffs` is not exactly `w*h*esz` bytes, [`DecodeError::Request`] when `out`
/// is not.
pub fn inverse_2d_into(coeffs: &[u8], dtype: Dtype, w: u32, h: u32, out: &mut [u8]) -> Result<(), DecodeError> {
    let esz = dtype.size();
    if !is_plane(coeffs.len(), w, h, esz) {
        return Err(DecodeError::Corrupt("wavelet payload length"));
    }
    if out.len() != coeffs.len() {
        return Err(DecodeError::Request("wavelet output plane length"));
    }
    let n = coeffs.len() / esz;
    lift_via_grid(coeffs, esz, out, |g| inv(g, w as usize, h as usize, width_mask(esz)))
        .map_err(|_| DecodeError::Alloc((n * size_of::<u32>()) as u64))
}

thread_local! {
    /// The `u32` lifting grid of one plane, kept between calls (taken while in use). Cache-line aligned
    /// ([`AlignedBuf`]): the vertical lifting streams whole rows of it, and a 16-byte aligned heap block (what the
    /// system allocator returns below ~1 MiB) splits every 64-byte vector access across two lines.
    static GRID: Cell<AlignedBuf> = const { Cell::new(AlignedBuf::new()) };
}

/// Largest grid (bytes) a thread keeps between calls (4 MiB = 1 Mi `u32`).
const GRID_KEEP: usize = 4 << 20;

/// `src` bytes -> `u32` grid (the thread's scratch) -> `lift` in place -> `out` bytes. An allocation failure of the
/// grid is reported, never an abort (`len` derives from the tile geometry).
fn lift_via_grid(src: &[u8], esz: usize, out: &mut [u8], lift: impl FnOnce(&mut [u32])) -> Result<(), TryReserveError> {
    let n = src.len() / esz;
    let mut buf = GRID.with(Cell::take);
    // `try_fit` leaves stale bytes (no zeroing): `to_elems` overwrites all `n` elements.
    let res = buf.try_fit(n * size_of::<u32>());
    if res.is_ok() {
        let g = buf.cast_mut::<u32>();
        to_elems(src, esz, g);
        lift(g);
        from_elems(g, esz, out);
    }
    if buf.capacity() <= GRID_KEEP {
        GRID.with(|c| c.set(buf));
    }
    res
}

/// One lifting sample type. The lossless transform lifts `u32` elements wrapping inside the element width
/// (`Cx` = the width mask); the lossy one lifts signed `i64` (`Cx = ()`, wrapping only so a forged coefficient
/// cannot overflow-panic). ONE generic lifting body ([`rows`]/[`cols`]) serves both, so the two cannot drift.
pub(crate) trait Sample: Copy + Default {
    /// Per-plane context threaded into every step (the width mask, or nothing).
    type Cx: Copy;

    /// `a ± ((l + r + round) >> sh)`; `SUB` picks the direction (forward predict / inverse update subtract the
    /// term, forward update / inverse predict add it), so one body serves both transforms.
    fn lift<const SUB: bool>(a: Self, l: Self, r: Self, round: u32, sh: u32, cx: Self::Cx) -> Self;
}

impl Sample for u32 {
    type Cx = u32;

    /// Sums use `u64`: for `esz = 4` the sum of two `u32` does not fit.
    #[inline(always)]
    fn lift<const SUB: bool>(a: u32, l: u32, r: u32, round: u32, sh: u32, mask: u32) -> u32 {
        let t = ((l as u64 + r as u64 + round as u64) >> sh) as u32;
        (if SUB { a.wrapping_sub(t) } else { a.wrapping_add(t) }) & mask
    }
}

impl Sample for i64 {
    type Cx = ();

    /// Floor via arithmetic shift. Wrapping ops equal the plain ones on every valid plane (no overflow there).
    #[inline(always)]
    fn lift<const SUB: bool>(a: i64, l: i64, r: i64, round: u32, sh: u32, _: ()) -> i64 {
        let t = l.wrapping_add(r).wrapping_add(round as i64) >> sh;
        if SUB { a.wrapping_sub(t) } else { a.wrapping_add(t) }
    }
}

/// Even/odd pairs of a row that [`rows`] keeps on the stack (rows up to 256 samples; `i64` samples = 1 KiB per array).
const ROW_STACK: usize = 128;

/// Forward 2D lifting in place: every row, then every column.
pub(crate) fn fwd<T: Sample>(g: &mut [T], w: usize, h: usize, cx: T::Cx) {
    rows::<T, false>(g, w, cx);
    cols::<T, false>(g, w, h, cx);
}

/// Inverse 2D lifting in place: every column, then every row.
pub(crate) fn inv<T: Sample>(g: &mut [T], w: usize, h: usize, cx: T::Cx) {
    cols::<T, true>(g, w, h, cx);
    rows::<T, true>(g, w, cx);
}

/// 1D lifting of every row, in place. `INV` = the inverse transform (steps in the opposite order and sign).
///
/// Each row is split into its even and odd samples (two contiguous arrays, scratch reused across rows), lifted
/// there, and merged back to the SAME interleaved layout: on the separated arrays every step is a plain
/// element-wise combine of neighbours (`od[i] ± f(ev[i], ev[i+1])`), so it auto-vectorises, where the in-place
/// stride-2 loop with an edge test per sample did not.
///
/// The even/odd scratch lives on the stack for rows up to [`ROW_STACK`] pairs (every EXV-RT block and a default
/// 256-px tile row): EXV-RT decode lifts thousands of tiny blocks per plane, and two heap `Vec`s per call were
/// 6000 allocations per tile (docs/OPT.md R). Wider rows pay one heap pair per call, amortised over all their rows.
fn rows<T: Sample, const INV: bool>(g: &mut [T], w: usize, cx: T::Cx) {
    if w < 2 {
        return;
    }
    let (ne, no) = (w.div_ceil(2), w / 2);
    let (mut sev, mut sod) = ([T::default(); ROW_STACK], [T::default(); ROW_STACK]);
    let (mut hev, mut hod);
    let (ev, od): (&mut [T], &mut [T]) = match (sev.get_mut(..ne), sod.get_mut(..no)) {
        (Some(e), Some(o)) => (e, o),
        _ => {
            (hev, hod) = (vec![T::default(); ne], vec![T::default(); no]);
            (&mut hev, &mut hod)
        }
    };
    for row in g.chunks_exact_mut(w) {
        let (pairs, rem) = row.as_chunks::<2>();
        for ((p, e), o) in pairs.iter().zip(ev.iter_mut()).zip(od.iter_mut()) {
            (*e, *o) = (p[0], p[1]);
        }
        if let (&[last], Some(e)) = (rem, ev.last_mut()) {
            *e = last; // odd width: the trailing sample is an even one
        }
        if INV {
            hupdate::<T, true>(ev, od, cx);
            hpredict::<T, false>(od, ev, cx);
        } else {
            hpredict::<T, true>(od, ev, cx);
            hupdate::<T, false>(ev, od, cx);
        }
        let (pairs, rem) = row.as_chunks_mut::<2>();
        for ((p, &e), &o) in pairs.iter_mut().zip(ev.iter()).zip(od.iter()) {
            (p[0], p[1]) = (e, o);
        }
        if let ([last], Some(&e)) = (rem, ev.last()) {
            *last = e;
        }
    }
}

/// Horizontal predict: `od[i] ± f(ev[i], ev[i+1])`; an even-length row's last odd sample mirrors its left neighbor.
fn hpredict<T: Sample, const SUB: bool>(od: &mut [T], ev: &[T], cx: T::Cx) {
    for ((o, &l), &r) in od.iter_mut().zip(ev).zip(ev.iter().skip(1)) {
        *o = T::lift::<SUB>(*o, l, r, 0, 1, cx);
    }
    if od.len() == ev.len()
        && let (Some(o), Some(&e)) = (od.last_mut(), ev.last())
    {
        *o = T::lift::<SUB>(*o, e, e, 0, 1, cx);
    }
}

/// Horizontal update: `ev[i] ± f(od[i-1], od[i])`; the first even sample (and the last of an odd-length row)
/// mirrors its single odd neighbor.
fn hupdate<T: Sample, const SUB: bool>(ev: &mut [T], od: &[T], cx: T::Cx) {
    if let (Some(e), Some(&o)) = (ev.first_mut(), od.first()) {
        *e = T::lift::<SUB>(*e, o, o, 2, 2, cx);
    }
    for ((e, &l), &r) in ev.iter_mut().skip(1).zip(od).zip(od.iter().skip(1)) {
        *e = T::lift::<SUB>(*e, l, r, 2, 2, cx);
    }
    if ev.len() > od.len()
        && let (Some(e), Some(&o)) = (ev.last_mut(), od.last())
    {
        *e = T::lift::<SUB>(*e, o, o, 2, 2, cx);
    }
}

/// 1D lifting of every column, in place, done ROW-WISE: a vertical lifting step combines whole rows
/// (`row[y] ± f(row[y-1], row[y+1])` across contiguous `x`), so every access is sequential and the loop
/// auto-vectorises. The previous per-column gather/scatter walked the plane with a `w*4`-byte stride (cache
/// associativity thrash on power-of-two widths) and was the transform's main cost. Same arithmetic, same bits.
fn cols<T: Sample, const INV: bool>(g: &mut [T], w: usize, h: usize, cx: T::Cx) {
    if h < 2 || w == 0 {
        return;
    }
    if INV {
        vstep::<T, true>(g, w, h, 0, cx);
        vstep::<T, false>(g, w, h, 1, cx);
    } else {
        vstep::<T, true>(g, w, h, 1, cx);
        vstep::<T, false>(g, w, h, 0, cx);
    }
}

/// One vertical lifting step over the rows `y ≡ parity (mod 2)`: parity 1 = predict (`>> 1`), parity 0 =
/// update (`(.. + 2) >> 2`). The edge rows mirror their single neighbor, exactly like the 1D lifting.
fn vstep<T: Sample, const SUB: bool>(g: &mut [T], w: usize, h: usize, parity: usize, cx: T::Cx) {
    let (round, sh) = if parity == 1 { (0, 1) } else { (2, 2) };
    for y in (parity..h).step_by(2) {
        // `up`/`dn` are the neighbor rows (`y-1`, `y+1`), mirrored at the plane's edges. With `h >= 2` and
        // `g.len() == w*h` both always exist, so the `if let` is a type-level formality, never a skip.
        let (head, tail) = g.split_at_mut(y * w);
        let (cur, rest) = tail.split_at_mut(w);
        let up = if y >= 1 { head.get((y - 1) * w..) } else { rest.get(..w) };
        let dn = if y + 1 < h { rest.get(..w) } else { up };
        if let (Some(up), Some(dn)) = (up, dn) {
            for ((c, &l), &r) in cur.iter_mut().zip(up).zip(dn) {
                *c = T::lift::<SUB>(*c, l, r, round, sh, cx);
            }
        }
    }
}

fn width_mask(esz: usize) -> u32 {
    if esz >= 4 { u32::MAX } else { (1u32 << (8 * esz)) - 1 }
}

/// Little-endian `N`-byte elements widened to `u32` into `elems`. `N` is a const so the load is one mov (a
/// runtime-width byte loop was the transform's second hot spot).
fn load<const N: usize>(bytes: &[u8], elems: &mut [u32]) {
    for (e, c) in elems.iter_mut().zip(bytes.as_chunks::<N>().0) {
        *e = crate::temporal::get::<N>(c);
    }
}

/// Inverse of [`load`]: the low `N` bytes of each element, LE.
fn store<const N: usize>(elems: &[u32], out: &mut [u8]) {
    for (c, &v) in out.as_chunks_mut::<N>().0.iter_mut().zip(elems) {
        *c = crate::temporal::put::<N>(v);
    }
}

fn to_elems(bytes: &[u8], esz: usize, elems: &mut [u32]) {
    match esz {
        1 => load::<1>(bytes, elems),
        2 => load::<2>(bytes, elems),
        _ => load::<4>(bytes, elems),
    }
}

fn from_elems(elems: &[u32], esz: usize, out: &mut [u8]) {
    match esz {
        1 => store::<1>(elems, out),
        2 => store::<2>(elems, out),
        _ => store::<4>(elems, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(plane: &[u8], dtype: Dtype, w: u32, h: u32) {
        let c = forward_2d(plane, dtype, w, h).expect("forward");
        assert_eq!(c.len(), plane.len());
        assert_eq!(inverse_2d(&c, dtype, w, h).expect("inverse"), plane, "wavelet rt {dtype:?} {w}x{h}");
    }

    /// A plane that is not exactly `w*h*esz` bytes is a typed error in both directions (it used to be silently
    /// zero-padded / truncated by a `resize`), including a geometry whose product wraps.
    #[test]
    fn wrong_length_is_a_typed_error() {
        for len in [0usize, 31, 33, 64] {
            let p = vec![7u8; len];
            assert_eq!(forward_2d(&p, Dtype::U16, 4, 4), Err(CodecError::Dims("wavelet plane != w*h*esz")), "len {len}");
            assert_eq!(inverse_2d(&p, Dtype::U16, 4, 4), Err(DecodeError::Corrupt("wavelet payload length")), "len {len}");
        }
        assert!(inverse_2d(&[0u8; 8], Dtype::U32, u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn roundtrip_dtypes_and_shapes() {
        // 256 / 257 / 600 wide straddle `ROW_STACK` (stack scratch vs the heap fallback of `rows`).
        for &(w, h) in &[(16u32, 16u32), (17, 13), (1, 9), (9, 1), (64, 48), (256, 3), (257, 3), (600, 2)] {
            let n = (w * h) as usize;
            let u16d: Vec<u8> = (0..n).flat_map(|i| ((i * 13 % 4096) as u16).to_le_bytes()).collect();
            rt(&u16d, Dtype::U16, w, h);
            rt(&u16d, Dtype::F16, w, h);
            let u8d: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
            rt(&u8d, Dtype::U8, w, h);
            let u32d: Vec<u8> = (0..n).flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()).collect();
            rt(&u32d, Dtype::U32, w, h);
        }
    }
}
