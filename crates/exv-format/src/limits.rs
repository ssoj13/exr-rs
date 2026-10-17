//! Decode-allocation limits for untrusted input — the single source of truth for every parser/decoder
//! that sizes an allocation from a value read out of a file (frame counts, tile counts, plane sizes).
//!
//! # Why
//! A tiny hostile file can declare huge counts/dimensions; `Vec::with_capacity(n)` / `vec![0; n]` then
//! aborts the process (uncatchable `handle_alloc_error`). Three complementary pieces:
//! - [`cap`] (`bufpool`) — bound a *capacity hint* by what the remaining bytes could possibly hold (never
//!   fails; the parse loop itself still fails on truncated data).
//! - [`DecodeLimits`] — an explicit, CALLER-OWNED byte budget for what one decode call may materialize.
//!   Parsing/opening never consults it (a legitimate 20000² all-zero image is a ~1.6 KB file — no
//!   compression-ratio guess can tell it from a bomb); only the places that actually materialize decoded
//!   planes do, and they return [`DecodeError::Limit`] when it is exceeded.
//! - [`alloc_zeroed`] (`bufpool`) — fallible allocation (`try_reserve_exact`): a size that passed the budget but
//!   still cannot be satisfied is an `Err`, never an abort.
//!
//! [`DecodeLimits`] also carries the tile-blob checksum policy ([`Verify`]): it is the one decode-options object
//! every read path already threads (`FrameReader::with_limits`, `VideoIndex::with_limits`, `ExvSource`, hosts), so the
//! policy reaches the CPU decoders and the GPU blob accessors without a parallel options struct.

use crate::error::DecodeError;

/// Default [`DecodeLimits::max_alloc`]: 8 GiB. An absolute number (not RAM-derived: this crate has no
/// system-info dependency), meant for tests and tools that know their inputs. A HOST opening untrusted files
/// should pass `exv_tile::Budget::auto(..)?.limits()` (the one RAM formula) so a 3 KB file declaring 32768² cannot
/// commit gigabytes on a small machine.
pub const DEFAULT_MAX_ALLOC: u64 = 8 << 30;

/// Whether a reader checks each tile blob against the CRC-32C its frame index records (`docs/FORMAT.md` §5) before
/// decoding / uploading it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verify {
    /// Verify every blob a call reads (the default: a mismatch is [`DecodeError::Checksum`], never wrong pixels).
    #[default]
    On,
    /// Trust the bytes (e.g. a host that already verified the file, or a hot GPU playback loop over a verified cache).
    Off,
}

/// Caller-provided decode options: the ceiling on the bytes a single decode call may materialize (decoded planes of a
/// frame / region / GOP / clip, plus one decoded tile of working set) and the blob checksum policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum bytes of decoded pixel data one call may hold at once.
    pub max_alloc: u64,
    /// Tile-blob checksum policy ([`Verify::On`] unless the caller opts out).
    pub verify: Verify,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ALLOC)
    }
}

impl DecodeLimits {
    /// A budget of `max_alloc` bytes, checksums verified.
    pub const fn new(max_alloc: u64) -> Self {
        Self { max_alloc, verify: Verify::On }
    }

    /// The same budget under checksum policy `verify`.
    pub const fn with_verify(mut self, verify: Verify) -> Self {
        self.verify = verify;
        self
    }

    /// `Ok` when materializing `need` bytes fits the budget, else [`DecodeError::Limit`].
    pub fn check(&self, need: u64) -> Result<(), DecodeError> {
        if need <= self.max_alloc { Ok(()) } else { Err(DecodeError::Limit { need, max: self.max_alloc }) }
    }
}

/// `cap` (bound a capacity hint by the remaining input) and `alloc_zeroed` (fallible zeroed `Vec`) are owned by
/// `bufpool::limits`; re-exported so decode paths keep one import. Their `AllocError` becomes [`DecodeError::Alloc`]
/// through `From`.
pub use crate::bufpool::limits::{alloc_zeroed, cap};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_explicit_and_typed() {
        let l = DecodeLimits::new(1000);
        assert_eq!(l.check(1000), Ok(()));
        assert_eq!(l.check(1001), Err(DecodeError::Limit { need: 1001, max: 1000 }));
        assert_eq!(DecodeLimits::default().max_alloc, DEFAULT_MAX_ALLOC);
    }

    #[test]
    fn checksum_policy_is_on_by_default() {
        let on = DecodeLimits::new(1);
        assert_eq!((on.verify, DecodeLimits::default().verify), (Verify::On, Verify::On));
        assert_eq!(on.with_verify(Verify::Off).verify, Verify::Off);
    }

    /// The re-exported allocator refusal maps into the one decode error.
    #[test]
    fn alloc_error_maps_to_decode_alloc() {
        assert_eq!(alloc_zeroed(usize::MAX).map_err(DecodeError::from), Err(DecodeError::Alloc(usize::MAX as u64)));
    }
}
