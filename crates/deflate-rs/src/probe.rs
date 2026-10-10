//! Fast-path engagement probes: a single, shared instrumentation mechanism so
//! a test can assert a fast path was ACTUALLY taken, not just that its result
//! matches a fallback (a fast path can silently degrade to the slow one while
//! every byte-identity test still passes). Same shape as exr-core's own
//! `probe.rs`: an `enum Probe` with `hit`/`get`/
//! `reset` over ONE `thread_local` `[Cell<u64>; N]` array.
//!
//! deflate-rs's fast paths (the inflate fastloop, BMI2 dispatch, the
//! static-Huffman table cache) all run on the CALLING thread — this crate
//! has no rayon/parallel dispatch of its own — so the thread_local-only
//! mechanism (no `AtomicU64`/cross-thread `Probes` struct) is sufficient
//! here; that half of the shared mechanism belongs to exr-core's own
//! `probe.rs` for its rayon `map_init` call sites.
//!
//! Compiled only under `cfg(any(test, feature = "probe"))` — a probe hit is
//! a cheap `Cell` increment, but it should still never appear in a release
//! build that isn't explicitly opted in via the `probe` feature (e.g. for
//! an external benchmark/bug-report tool wanting `active_tiers_report()`-
//! style visibility without a full test harness).

#![cfg(any(test, feature = "probe"))]

use std::cell::Cell;

/// One engagement point this crate instruments. Add a variant + bump [`N`]
/// (and [`Probe::idx`]) for a new fast path; never repurpose an existing
/// variant for a different path (a stale assertion would silently start
/// checking the wrong thing).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Probe {
    /// One hit per `'fastloop` loop ITERATION inside
    /// `deflate_decompress_impl`: proves the fastloop's body actually
    /// ran at least once, not just that `FASTLOOP_ENABLED` is `true`.
    FastloopIters,
    /// One hit per DEFLATE block header parsed (the `'next_block` loop's top)
    /// inside `deflate_decompress_impl`: proves a multi-block stream
    /// really decoded more than one block, not just that the input CONTAINS
    /// more than one (a `fastloop_hit_eob` regression could silently stop
    /// after the first block while still "succeeding" on a truncated read).
    BlocksDecoded,
    /// One hit per `Compressor::new` call that builds the static (fixed)
    /// Huffman code tables: proves the static-codes cache is reused
    /// across multiple blocks in one compress call, not rebuilt per block.
    StaticCodesBuild,
}

/// Number of [`Probe`] variants — keep in lockstep with the enum by hand
/// (a `#[derive]`-based count needs a proc-macro dependency this crate
/// doesn't otherwise have, and the mismatch is caught immediately: an
/// out-of-range [`Probe::idx`] would panic in test builds).
const N: usize = 3;

impl Probe {
    fn idx(self) -> usize {
        match self {
            Probe::FastloopIters => 0,
            Probe::BlocksDecoded => 1,
            Probe::StaticCodesBuild => 2,
        }
    }
}

thread_local! {
    static COUNTS: [Cell<u64>; N] = const { [const { Cell::new(0) }; N] };
}

/// Record one occurrence of `p` on the calling thread.
pub(crate) fn hit(p: Probe) {
    COUNTS.with(|c| {
        let cell = &c[p.idx()];
        cell.set(cell.get() + 1);
    });
}

/// This thread's count for `p` since the last [`reset`]/[`reset_all`].
pub(crate) fn get(p: Probe) -> u64 {
    COUNTS.with(|c| c[p.idx()].get())
}

/// Reset one probe's count to zero on the calling thread.
pub(crate) fn reset(p: Probe) {
    COUNTS.with(|c| c[p.idx()].set(0));
}

/// Reset every probe's count to zero on the calling thread (a fresh test
/// that cares about more than one probe resets once up front instead of
/// calling [`reset`] per variant).
pub(crate) fn reset_all() {
    COUNTS.with(|c| {
        for cell in c.iter() {
            cell.set(0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_get_reset_are_independent_per_probe() {
        reset_all();
        assert_eq!(get(Probe::FastloopIters), 0);
        assert_eq!(get(Probe::BlocksDecoded), 0);

        hit(Probe::FastloopIters);
        hit(Probe::FastloopIters);
        hit(Probe::BlocksDecoded);

        assert_eq!(get(Probe::FastloopIters), 2, "FastloopIters must count its own hits only");
        assert_eq!(get(Probe::BlocksDecoded), 1, "BlocksDecoded must not see FastloopIters' hits");
        assert_eq!(get(Probe::StaticCodesBuild), 0, "an un-hit probe stays at zero");

        reset(Probe::FastloopIters);
        assert_eq!(get(Probe::FastloopIters), 0, "reset(p) must only clear p");
        assert_eq!(get(Probe::BlocksDecoded), 1, "reset(p) must not clear OTHER probes");

        reset_all();
        assert_eq!(get(Probe::BlocksDecoded), 0, "reset_all must clear every probe");
    }
}
