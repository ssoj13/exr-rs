//! [`AlignedBuf`]: a cache-line (64-byte) aligned byte buffer without `unsafe` (plane buffers, per-thread codec
//! scratch, anything parallel lanes write side by side).
//!
//! # Why
//! `Vec<u8>` only promises 1-byte alignment. On Windows the system heap returns 16-byte aligned blocks below ~1 MiB
//! (measured: `ptr % 64 == 16` for 64-128 KiB buffers) and `page + 64` above; glibc returns `page + 16` for large
//! blocks. A plane whose base is not 64-byte aligned puts the boundary between two tiles of a tile row in the MIDDLE of
//! a cache line on every pixel row, so two decode lanes writing neighbouring tiles false-share that line; SIMD loads and
//! stores of misaligned rows also split cache lines. Rows of `tile_w * esz` bytes (a multiple of 64 for common tile
//! sizes) start on a line boundary only when the base does.
//!
//! # How
//! The storage is a `Vec` of 64-byte, 64-aligned `Line`s; `bytemuck` (a `Pod` cast) views it as `[u8]` or as a typed
//! slice. Lengths are in BYTES; the tail of the last line is padding that is never exposed. Allocation is fallible
//! (`try_reserve_exact`): sizes derived from file data must never abort.

use std::collections::TryReserveError;

use bytemuck::{Pod, Zeroable};

use super::limits::AllocError;

/// Bytes per cache line: the alignment and allocation unit of [`AlignedBuf`] (x86-64 and most ARM cores; Apple
/// M-series use 128-byte lines, for which 64 still avoids splits of 64-byte SIMD accesses).
pub const LINE: usize = 64;

/// One cache line: the allocation unit of [`AlignedBuf`] (its `align` is what the whole buffer inherits).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([u8; LINE]);

/// A growable byte buffer whose data pointer is ALWAYS 64-byte aligned (see the module docs).
///
/// `Deref<Target = [u8]>` exposes exactly `len` bytes. Growth by [`Self::try_resize`] zero-fills the new bytes;
/// shrinking keeps the capacity, so a recycled buffer re-grows without touching the allocator.
///
/// Used by: [`crate::PlanePool`] / [`crate::PlaneBuf`] storage; codec tile lanes and per-thread scratch in hosts.
#[derive(Clone, Default)]
pub struct AlignedBuf {
    lines: Vec<Line>,
    len: usize,
}

impl AlignedBuf {
    /// An empty buffer (no allocation; usable in a `const` thread-local initializer).
    pub const fn new() -> Self {
        Self { lines: Vec::new(), len: 0 }
    }

    /// `len` zero bytes: the aligned twin of [`crate::alloc_zeroed`] for sizes derived from file data.
    ///
    /// # Errors
    /// [`AllocError`] when the allocator refuses (no abort).
    pub fn zeroed(len: usize) -> Result<Self, AllocError> {
        let mut b = Self::new();
        b.try_resize(len).map_err(|_| AllocError { bytes: len })?;
        Ok(b)
    }

    /// Set the length to `len` bytes: bytes beyond the old length read as zero, bytes below it keep their contents.
    /// Never shrinks the allocation.
    ///
    /// # Errors
    /// The allocator's refusal (the buffer is left unchanged).
    pub fn try_resize(&mut self, len: usize) -> Result<(), TryReserveError> {
        let (old, held) = (self.len, self.lines.len() * LINE);
        self.try_fit(len)?;
        // Lines `try_fit` appended are already zero; only bytes an earlier shrink left behind (stale) need clearing.
        if let Some(stale) = self.as_bytes_mut().get_mut(old..len.min(held).max(old)) {
            stale.fill(0);
        }
        Ok(())
    }

    /// Set the length to `len` bytes WITHOUT clearing: bytes beyond the old length are UNSPECIFIED (zero when freshly
    /// allocated, stale bytes of an earlier use otherwise; always initialized). The scratch mode for callers that
    /// overwrite every byte (decoders, lifting grids). Never shrinks the allocation.
    ///
    /// # Errors
    /// The allocator's refusal (the buffer is left unchanged).
    pub fn try_fit(&mut self, len: usize) -> Result<(), TryReserveError> {
        let lines = len.div_ceil(LINE);
        if lines > self.lines.len() {
            self.lines.try_reserve_exact(lines - self.lines.len())?;
            self.lines.resize(lines, Line::zeroed());
        }
        self.len = len;
        Ok(())
    }

    /// Bytes exposed.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when no bytes are exposed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes allocated (a multiple of [`LINE`]): what a pool or a budget charges.
    pub fn capacity(&self) -> usize {
        self.lines.capacity() * LINE
    }

    /// The `len` bytes.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice::<Line, u8>(&self.lines).get(..self.len).unwrap_or(&[])
    }

    /// The `len` bytes, mutable.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len;
        bytemuck::cast_slice_mut::<Line, u8>(&mut self.lines).get_mut(..len).unwrap_or(&mut [])
    }

    /// The bytes as `T`s (`len / size_of::<T>()` of them; the base is 64-aligned, so any `Pod` scalar fits).
    pub fn cast<T: Pod>(&self) -> &[T] {
        let n = self.len / size_of::<T>().max(1);
        bytemuck::cast_slice::<Line, T>(&self.lines).get(..n).unwrap_or(&[])
    }

    /// The bytes as mutable `T`s (`len / size_of::<T>()` of them).
    pub fn cast_mut<T: Pod>(&mut self) -> &mut [T] {
        let n = self.len / size_of::<T>().max(1);
        bytemuck::cast_slice_mut::<Line, T>(&mut self.lines).get_mut(..n).unwrap_or(&mut [])
    }

    /// A plain `Vec<u8>` copy (for APIs that own unaligned buffers: a `Vec<u8>` cannot own 64-aligned storage
    /// without `unsafe`).
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

impl std::ops::Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

/// An aligned copy of bytes already in memory (infallible like `Vec::from(&[u8])`: the size is not file-derived; use
/// [`AlignedBuf::zeroed`] for those).
impl From<&[u8]> for AlignedBuf {
    fn from(src: &[u8]) -> Self {
        let lines = src
            .chunks(LINE)
            .map(|c| {
                let mut l = Line::zeroed();
                l.0.get_mut(..c.len()).unwrap_or(&mut []).copy_from_slice(c);
                l
            })
            .collect();
        Self { lines, len: src.len() }
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl PartialEq for AlignedBuf {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for AlignedBuf {}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf").field("len", &self.len).field("capacity", &self.capacity()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aligned(p: *const u8) -> bool {
        (p as usize).is_multiple_of(LINE)
    }

    /// Every length (incl. sub-line, line-exact and large) yields a 64-aligned base, zeroed growth, kept prefix.
    #[test]
    fn base_is_cache_line_aligned_and_growth_is_zeroed() {
        for len in [1usize, 63, 64, 65, 4096, 128 << 10, 3 << 20] {
            let mut b = AlignedBuf::zeroed(len).expect("alloc");
            assert!(aligned(b.as_ptr()) && b.len() == len && b.iter().all(|&v| v == 0), "len {len}");
            assert!(b.capacity() >= len && b.capacity().is_multiple_of(LINE));
            b.fill(7);
            b.try_resize(len / 2).expect("shrink");
            b.try_resize(len + 100).expect("grow");
            assert!(b[..len / 2].iter().all(|&v| v == 7), "prefix kept");
            assert!(b[len / 2..].iter().all(|&v| v == 0), "regrown bytes are zero, not stale (len {len})");
            assert!(aligned(b.cast::<u32>().as_ptr().cast()) && b.cast_mut::<i64>().len() == (len + 100) / 8);
        }
        assert!(AlignedBuf::new().is_empty() && AlignedBuf::new().as_bytes().is_empty());
        let src: Vec<u8> = (0..200u8).collect();
        let c = AlignedBuf::from(src.as_slice());
        assert!(aligned(c.as_ptr()) && c.to_vec() == src && AlignedBuf::from(&[][..]).is_empty());
    }

    /// `try_fit` keeps stale bytes (scratch mode) and never shrinks the allocation; the base stays aligned after a
    /// reallocating grow.
    #[test]
    fn try_fit_keeps_stale_bytes_and_capacity() {
        let mut b = AlignedBuf::zeroed(256).expect("alloc");
        b.fill(5);
        let cap = b.capacity();
        b.try_fit(10).expect("shrink");
        assert_eq!((b.len(), b.capacity()), (10, cap), "shrink keeps the allocation");
        b.try_fit(256).expect("regrow");
        assert!(b.iter().all(|&v| v == 5), "scratch regrow does not clear");
        b.try_fit(1 << 20).expect("realloc");
        assert!(aligned(b.as_ptr()) && b[..256].iter().all(|&v| v == 5) && b[256..].iter().all(|&v| v == 0));
    }

    #[test]
    fn huge_request_is_an_error_not_an_abort() {
        assert_eq!(AlignedBuf::zeroed(usize::MAX - 1), Err(AllocError { bytes: usize::MAX - 1 }));
        let mut b = AlignedBuf::zeroed(10).expect("alloc");
        assert!(b.try_resize(usize::MAX / 2).is_err());
        assert_eq!(b.len(), 10, "a refused grow leaves the buffer unchanged");
    }
}
