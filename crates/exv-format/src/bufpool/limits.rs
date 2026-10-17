//! Fallible allocation helpers for sizes read from untrusted input.
//!
//! # Why
//! A tiny hostile file can declare huge counts or dimensions; `Vec::with_capacity(n)` / `vec![0; n]` then aborts the
//! process (`handle_alloc_error` cannot be caught). Two complementary pieces:
//! - [`cap`]: bound a CAPACITY HINT by what the remaining input bytes could possibly hold (never fails; the parse loop
//!   itself still fails on truncated data).
//! - [`alloc_zeroed`]: fallible allocation (`try_reserve_exact`): a size that passed the caller's budget but still
//!   cannot be satisfied is an [`AllocError`], never an abort.
//!
//! The caller-owned decode budget itself (how many bytes one call may materialize) is format policy and stays with
//! the format crate.

use std::fmt;

/// The allocator refused `bytes` (a fallible allocation failed instead of aborting the process).
///
/// Returned by [`alloc_zeroed`], [`crate::AlignedBuf::zeroed`] and [`crate::PlanePool::take`]; a format crate maps it
/// into its own decode error with a `From` impl.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocError {
    /// The requested size.
    pub bytes: usize,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot allocate {} bytes", self.bytes)
    }
}

impl std::error::Error for AllocError {}

/// Clamp a capacity hint `n` (read from a file) to what `remaining` input bytes could hold at `min_entry` bytes per
/// entry. Use as `Vec::with_capacity(cap(n, remaining, ENTRY))`.
pub fn cap(n: usize, remaining: usize, min_entry: usize) -> usize {
    n.min(remaining / min_entry.max(1))
}

/// A zero-filled `Vec<u8>` of `len` bytes, or [`AllocError`] when the allocator refuses (no abort). Use for every
/// buffer whose size derives from file-declared dimensions.
///
/// The eager memset is deliberate: a lazy zeroed allocation (fresh OS pages faulted in by the decode lanes) measured
/// no faster for a 4k frame decode on Windows, which serializes page faults whoever takes them, and stable Rust has no
/// safe fallible zeroed allocation. The lever that removes the cost is REUSE ([`crate::PlanePool`]).
///
/// # Errors
/// [`AllocError`] when the allocation cannot be satisfied.
pub fn alloc_zeroed(len: usize) -> Result<Vec<u8>, AllocError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| AllocError { bytes: len })?;
    v.resize(len, 0);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_bounds_hint_by_remaining() {
        assert_eq!(cap(usize::MAX, 100, 9), 11);
        assert_eq!(cap(5, 100, 9), 5);
        assert_eq!(cap(7, 0, 0), 0);
    }

    #[test]
    fn alloc_zeroed_is_fallible() {
        assert_eq!(alloc_zeroed(5), Ok(vec![0; 5]));
        // Larger than the address space can ever serve: an Err, not an abort.
        assert_eq!(alloc_zeroed(usize::MAX), Err(AllocError { bytes: usize::MAX }));
    }
}
