//! Universal reversible byte decorrelation: **stride-aware byte-plane split + byte-wise delta**.
//!
//! # What and why
//! A generalization of the OpenEXR predictor (reorder+delta) to any element size. It groups bytes
//! of the same significance (byte `p` of all elements in a row) → neighboring values are similar →
//! delta produces long runs of zeros/small numbers that rANS compresses close to entropy. It works
//! on bytes (`stride = dtype.size()`), so it is type-agnostic and lossless.
//!
//! The inverse is exact: `join_undelta(split_delta(x, s), s) == x` for any `x`, `s ≥ 1`.

/// Forward transform: for each byte plane `p∈[0,stride)` lay out `data[e*stride+p]` over all
/// elements `e`, applying a per-plane first-difference. The tail (if `len % stride != 0`)
/// is handled as a separate delta block.
pub fn split_delta(data: &[u8], stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    split_delta_into(data, stride, &mut out);
    out
}

/// [`split_delta`] into `out` (`out.len() == data.len()`, every byte is overwritten): the encoder's decorrelated
/// stream is a plane-sized temporary per channel, so it lives in a reused scratch buffer.
pub fn split_delta_into(data: &[u8], stride: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), data.len(), "split_delta_into: output must match the input length");
    if stride <= 1 {
        delta_into(data, out);
        return;
    }
    let n = data.len() / stride;
    let Some((planes, tail)) = out.split_at_mut_checked(n * stride) else { return };
    match stride {
        2 => split_n::<2>(data.as_chunks::<2>().0, planes),
        4 => split_n::<4>(data.as_chunks::<4>().0, planes),
        _ => {
            // Any other stride (never a dtype width): the runtime-width form of `split_n`.
            for (p, plane) in planes.chunks_exact_mut(n.max(1)).enumerate() {
                let mut prev = 0u8;
                for (o, e) in plane.iter_mut().zip(data.chunks_exact(stride)) {
                    let b = e.get(p).copied().unwrap_or(0);
                    *o = b.wrapping_sub(prev);
                    prev = b;
                }
            }
        }
    }
    // tail bytes (len not a multiple of stride)
    delta_into(data.get(n * stride..).unwrap_or_default(), tail);
}

/// Split `N`-byte elements into `N` delta-coded byte planes (`elems.len()` bytes each, back to back in `planes`).
/// One sequential pass per output plane: the strided READ is the only irregular access, and the first difference
/// reads the previous INPUT byte (not the previous output), so there is no loop-carried arithmetic chain. Elements
/// are `[u8; N]` arrays and `p < N` is the loop bound, so `e[p]` has no bounds check.
fn split_n<const N: usize>(elems: &[[u8; N]], planes: &mut [u8]) {
    let n = elems.len();
    for (p, plane) in (0..N).zip(planes.chunks_exact_mut(n.max(1))) {
        let mut prev = 0u8;
        for (o, e) in plane.iter_mut().zip(elems) {
            let b = e[p];
            *o = b.wrapping_sub(prev);
            prev = b;
        }
    }
}

/// Inverse transform of [`split_delta`].
pub fn join_undelta(data: &[u8], stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    join_undelta_into(data, stride, &mut out);
    out
}

/// [`join_undelta`] into `out` (`out.len() == data.len()`, every byte is overwritten): the tile decoder writes the
/// final plane straight into its recycled tile buffer.
pub fn join_undelta_into(data: &[u8], stride: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), data.len(), "join_undelta_into: output must match the input length");
    if stride <= 1 {
        undelta_into(data, out);
        return;
    }
    let n = data.len() / stride;
    let (planes, tail) = data.split_at(n * stride);
    let Some((body, out_tail)) = out.split_at_mut_checked(n * stride) else { return };
    match stride {
        2 => join_n::<2>(planes, n, body),
        4 => join_n::<4>(planes, n, body),
        _ => {
            // Any other stride (never a dtype width): one serial pass per plane, strided write.
            for (p, plane) in planes.chunks_exact(n.max(1)).enumerate() {
                let mut prev = 0u8;
                for (o, &d) in body[p..].iter_mut().step_by(stride).zip(plane) {
                    prev = d.wrapping_add(prev);
                    *o = prev;
                }
            }
        }
    }
    undelta_into(tail, out_tail);
}

/// Undelta + re-interleave of `N` byte planes (`n` bytes each, back to back in `planes`) into `body`
/// (`n*N` bytes). The `N` running sums advance together, element by element: independent dependency chains
/// the CPU overlaps (a lone prefix sum is latency-bound at one add per byte) and one sequential store stream.
/// `N` is a const so the plane cursors live in registers.
///
/// Bounds: every plane is re-sliced to exactly `m` = the element count, so `ps[p][i]` (`p < N` const, `i < m` the
/// loop bound) needs no per-byte check, and the loop has no per-byte branch (the old per-plane `Iterator::next` had one).
fn join_n<const N: usize>(planes: &[u8], n: usize, body: &mut [u8]) {
    let elems = body.as_chunks_mut::<N>().0;
    let m = elems.len().min(n);
    let ps: [&[u8]; N] = std::array::from_fn(|p| planes.get(p * n..p * n + m).unwrap_or_default());
    if ps.iter().any(|s| s.len() != m) {
        return; // `planes` holds `N*n` bytes (the caller's split): unreachable
    }
    let mut acc = [0u8; N];
    for (i, e) in elems.iter_mut().take(m).enumerate() {
        for p in 0..N {
            acc[p] = acc[p].wrapping_add(ps[p][i]);
            e[p] = acc[p];
        }
    }
}

/// First difference of `d` into `o` (equal lengths).
fn delta_into(d: &[u8], o: &mut [u8]) {
    let mut prev = 0u8;
    for (o, &b) in o.iter_mut().zip(d) {
        *o = b.wrapping_sub(prev);
        prev = b;
    }
}

/// Running sum of `d` into `o` (equal lengths).
fn undelta_into(d: &[u8], o: &mut [u8]) {
    let mut prev = 0u8;
    for (o, &b) in o.iter_mut().zip(d) {
        prev = b.wrapping_add(prev);
        *o = prev;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_strides() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 37 % 256) as u8).collect();
        for stride in [1usize, 2, 3, 4] {
            let t = split_delta(&data, stride);
            assert_eq!(t.len(), data.len());
            assert_eq!(join_undelta(&t, stride), data, "stride={stride}");
        }
    }

    #[test]
    fn roundtrip_non_multiple_len() {
        // length not a multiple of stride (tail)
        let data: Vec<u8> = (0..1003u32).map(|i| (i % 251) as u8).collect();
        for stride in [2usize, 4] {
            let t = split_delta(&data, stride);
            assert_eq!(join_undelta(&t, stride), data, "stride={stride}");
        }
    }
}
