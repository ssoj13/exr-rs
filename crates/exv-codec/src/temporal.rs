//! Temporal decorrelation: integer element-wise delta of two frames.
//!
//! # Invariant (see `docs/archive/PLAN.md` §3.7)
//! Delta is taken on the **integer bit-pattern** of the element (for F16/F32 — NOT float subtraction,
//! otherwise bit-precision is lost). Each element is read as an unsigned integer of width `dtype.size()`,
//! subtracted modulo `2^(8·esz)` (wrapping), and written back LE. Exactly reversible.

use exv_format::{DecodeError, Dtype};

use crate::CodecError;

/// `residual = cur - prev` element-wise (wrapping in the dtype width).
///
/// # Errors
/// [`CodecError::Dims`] when the slices differ in length or are not a whole number of elements — never a
/// silently truncated residual (the old `min()` quietly dropped the tail, i.e. produced a "lossless" frame
/// that was not).
pub fn delta(cur: &[u8], prev: &[u8], dtype: Dtype) -> Result<Vec<u8>, CodecError> {
    map_pairs(cur, prev, dtype, |c, p| c.wrapping_sub(p)).ok_or(CodecError::Dims(SHAPE))
}

/// `cur = residual + prev` — inverse of [`delta`].
///
/// # Errors
/// [`DecodeError::Corrupt`] on the same shape violations (a residual blob of the wrong length is a corrupt
/// tile; the cause is no longer collapsed into a bare `None`).
pub fn undelta(residual: &[u8], prev: &[u8], dtype: Dtype) -> Result<Vec<u8>, DecodeError> {
    map_pairs(residual, prev, dtype, |r, p| r.wrapping_add(p)).ok_or(DecodeError::Corrupt(SHAPE))
}

/// `residual += prev` IN PLACE ([`undelta`] without the output `Vec`): the tile decoder adds the reference rows
/// onto the residual it just decoded into its recycled tile buffer.
///
/// # Errors
/// [`DecodeError::Corrupt`] on the shape violations of [`undelta`].
pub fn undelta_assign(residual: &mut [u8], prev: &[u8], dtype: Dtype) -> Result<(), DecodeError> {
    let esz = dtype.size();
    if residual.len() != prev.len() || !residual.len().is_multiple_of(esz) {
        return Err(DecodeError::Corrupt(SHAPE));
    }
    fn run<const N: usize>(dst: &mut [u8], src: &[u8]) {
        for (d, s) in dst.as_chunks_mut::<N>().0.iter_mut().zip(src.as_chunks::<N>().0) {
            *d = put::<N>(get::<N>(d).wrapping_add(get::<N>(s)));
        }
    }
    match esz {
        1 => run::<1>(residual, prev),
        2 => run::<2>(residual, prev),
        _ => run::<4>(residual, prev), // dtype widths are 1, 2 or 4
    }
    Ok(())
}

/// The LE `N`-byte element `c` (`N` = 1, 2 or 4: a dtype width) widened to `u32`. THE element load of the
/// const-width plane loops (`temporal`, `wavelet`, `lossy`, `quantize`): elements arrive as ARRAYS
/// (`as_chunks::<N>`), not `chunks_exact` slices, so the width is a type fact and there is no length check or
/// `copy_from_slice` call per element.
#[inline(always)]
pub(crate) fn get<const N: usize>(c: &[u8; N]) -> u32 {
    let mut v = [0u8; 4];
    for (d, s) in v.iter_mut().zip(c) {
        *d = *s;
    }
    u32::from_le_bytes(v)
}

/// The low `N` bytes of `v`, LE (inverse of [`get`]; the element store of the same loops).
#[inline(always)]
pub(crate) fn put<const N: usize>(v: u32) -> [u8; N] {
    let b = v.to_le_bytes();
    std::array::from_fn(|i| b.get(i).copied().unwrap_or(0))
}

/// Why a temporal operation refused its operands.
const SHAPE: &str = "temporal operands differ in length or hold a partial element";

fn map_pairs(a: &[u8], b: &[u8], dtype: Dtype, op: impl Fn(u32, u32) -> u32) -> Option<Vec<u8>> {
    let esz = dtype.size();
    if a.len() != b.len() || !a.len().is_multiple_of(esz) {
        return None;
    }
    let mut out = vec![0u8; a.len()];
    // `N` is a const so each element is a plain fixed-width load/store and the loop auto-vectorises (the
    // runtime-width byte loop it replaces ran ~2-3 cycles per BYTE).
    fn run<const N: usize>(out: &mut [u8], a: &[u8], b: &[u8], op: impl Fn(u32, u32) -> u32) {
        for ((o, x), y) in out.as_chunks_mut::<N>().0.iter_mut().zip(a.as_chunks::<N>().0).zip(b.as_chunks::<N>().0) {
            *o = put::<N>(op(get::<N>(x), get::<N>(y)));
        }
    }
    match esz {
        1 => run::<1>(&mut out, a, b, op),
        2 => run::<2>(&mut out, a, b, op),
        _ => run::<4>(&mut out, a, b, op), // dtype widths are 1, 2 or 4
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_roundtrip_all_dtypes() {
        for dtype in [Dtype::U8, Dtype::U16, Dtype::F16, Dtype::U32, Dtype::F32, Dtype::Bf16] {
            let esz = dtype.size();
            let cur: Vec<u8> = (0..64 * esz).map(|i| (i * 7 % 256) as u8).collect();
            let prev: Vec<u8> = (0..64 * esz).map(|i| (i * 3 % 256) as u8).collect();
            let res = delta(&cur, &prev, dtype).expect("equal lengths");
            assert_eq!(undelta(&res, &prev, dtype).as_deref(), Ok(cur.as_slice()), "{dtype:?}");
        }
    }

    #[test]
    fn identical_frames_zero_residual() {
        let f: Vec<u8> = (0..256u16).flat_map(|v| v.to_le_bytes()).collect();
        let res = delta(&f, &f, Dtype::F16).expect("equal lengths");
        assert!(res.iter().all(|&b| b == 0), "identical frames → zero residual");
    }

    /// Unequal lengths / a partial trailing element are errors, not a truncated residual.
    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(matches!(delta(&[0u8; 8], &[0u8; 6], Dtype::F16), Err(CodecError::Dims(_))));
        assert!(matches!(undelta(&[0u8; 6], &[0u8; 8], Dtype::F32), Err(DecodeError::Corrupt(_))));
        assert!(delta(&[0u8; 7], &[0u8; 7], Dtype::F16).is_err(), "7 bytes is not a whole F16 plane");
        assert_eq!(delta(&[], &[], Dtype::U32).ok(), Some(Vec::new()));
    }
}
