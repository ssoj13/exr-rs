//! Shared scratch-buffer helpers: the grow-or-truncate size policy
//! ([`ensure_len`]) and a thread-safe reusable-buffer pool ([`BufPool`]).
//!
//! A neutral home for both, separate from [`crate::compressor`]: `ensure_len`
//! is used far outside the compressor module (the scanline/tiled read/write
//! pipelines, `exr-gpu`'s upload/readback), and `BufPool` exists specifically
//! for the read/write pipelines' parallel stages, not the compressor itself.
//! `compressor` re-exports `ensure_len` (`pub(crate)`) for the many existing
//! internal call sites that still say `compressor::ensure_len`; new code
//! should use `crate::buf::ensure_len` directly.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::probe::{Probe, SharedProbes};

/// Shared "high-water-mark" scratch-buffer growth policy:
/// grow `buf` to at least `len` bytes, zero-filling
/// only the NEWLY added tail, or `truncate` it to exactly `len` bytes if it
/// is already at least that long — never an unconditional `clear()` +
/// zero-fill of the whole buffer on every call.
///
/// Sound whenever every byte up to `len` is fully overwritten before being
/// read back (a reorder/predict/interleave transform, a per-pixel
/// scatter/gather loop, …) OR is a decoder's own output that never reads
/// past what it has already written this call (an LZ77-style back-reference
/// can never point past the already-decoded prefix) — in both cases stale
/// bytes left over in the tail from a prior, unrelated call are never
/// observed. [`crate::compressor::zlib::uncompress_into`]'s doc has the
/// canonical version of this argument (this helper is the same policy,
/// extracted so it is not hand-rolled at every call site).
///
/// A caller with additional invariants (e.g. exact-fill requirements) checks
/// `buf.len()` itself after the transform that follows, same as before.
///
/// Generic over the element type so the same helper covers byte scratch
/// buffers (`Vec<u8>`) and the codecs that gather into a wider element type
/// instead (PIZ's `Vec<u16>` per-channel scratch).
#[inline]
pub fn ensure_len<T: Clone + Default>(buf: &mut Vec<T>, len: usize) {
    if buf.len() < len {
        buf.resize(len, T::default());
    } else {
        buf.truncate(len);
    }
}

/// Peak windows in flight in a parallel reader: decoding + queued + fetching.
pub const READ_WINDOWS_IN_FLIGHT: usize = 3;
/// Peak "windows" in flight in a parallel writer. Writers run a ring
/// ([`crate::threading::ordered_pipeline`]) of at most ONE window of items
/// outstanding - encoding, finished-awaiting-order, or being written - and a
/// buffer returns to the pool before the item that replaces it is submitted,
/// so one window of buffers is the true peak.
pub const WRITE_WINDOWS_IN_FLIGHT: usize = 1;

/// A thread-safe pool of reusable `Vec<u8>` buffers.
///
/// The scanline/tiled parallel read and write pipelines
/// each hand a buffer across a producer/worker or worker/writer boundary --
/// unlike [`ensure_len`] alone (which only helps when the SAME persistent
/// buffer is reused by the SAME call site, e.g. a codec's own scratch), a
/// buffer that must travel between stages (read on the
/// producer thread, decoded+scattered on a worker thread; or encoded on a
/// worker thread, written on the ordering thread) needs a buffer SOURCE that
/// outlives any one stage and that multiple threads can check buffers in and
/// out of concurrently. `BufPool` is that one mechanism, shared by both
/// pipelines:
/// - Read: [`BufPool::take`] a buffer sized for the next chunk/tile
///   before `stream.read`; after the decoded pixels are scattered into the
///   frame buffer, [`BufPool::give`] the (now-unneeded) raw-bytes buffer
///   back.
/// - Write: [`BufPool::take`] a buffer before compressing a block/tile
///   into it (instead of a fresh `Vec::new()`/`.to_vec()`); after the
///   ordered write flushes it to the stream, [`BufPool::give`] it back.
///
/// Internally a `Mutex<Vec<Vec<u8>>>` (a simple free-list) rather than a
/// lock-free structure: contention is low (one lock per chunk/tile, not per
/// byte) and correctness/auditability matter more here than shaving a lock
/// acquisition off an I/O-bound path.
pub struct BufPool {
    free: Mutex<Vec<Vec<u8>>>,
    /// Hit/miss counts of `take` (test-only; zero-sized otherwise).
    probes: SharedProbes,
}

/// Hands this pool's probe counts to the thread that drops it: the owning
/// read/write call, which creates the pool as a local and only lends it to
/// its workers. A test on that thread then reads worker hits with
/// `probe::get` like any caller-thread probe.
#[cfg(test)]
impl Drop for BufPool {
    fn drop(&mut self) {
        crate::probe::absorb(&self.probes);
    }
}

impl BufPool {
    /// A new, empty pool. The first `take` calls on every worker thread will
    /// allocate fresh; the pool fills up as buffers are returned via `give`.
    pub fn new() -> Self {
        BufPool { free: Mutex::new(Vec::new()), probes: SharedProbes::default() }
    }

    /// Check out a buffer sized to exactly `len` bytes: pops one from the
    /// free list (growing or truncating it via [`ensure_len`]) if one is
    /// available, else allocates a fresh one. The returned buffer's content
    /// is UNSPECIFIED (it may hold a previous user's bytes up to the old
    /// length, or zeros in any newly-grown tail) -- every caller must fully
    /// overwrite (or bounds its reads to) the bytes it actually wrote, same
    /// contract as a freshly `ensure_len`-grown `Vec` reused in place.
    pub fn take(&self, len: usize) -> Vec<u8> {
        let popped = self.free_list().pop();
        self.probes.hit(if popped.is_some() { Probe::BufPoolHit } else { Probe::BufPoolMiss });
        let mut buf = popped.unwrap_or_default();
        ensure_len(&mut buf, len);
        buf
    }

    /// Return a buffer to the pool for a future `take` to reuse. The
    /// buffer's capacity is what matters for reuse; its length is
    /// immaterial (the next `take` calls `ensure_len` on it regardless).
    pub fn give(&self, buf: Vec<u8>) {
        self.free_list().push(buf);
    }

    /// The free list. A poisoned lock (a panic elsewhere while holding it) is
    /// recovered, not re-raised: the list only ever holds whole, idle buffers
    /// (push/pop are single operations), so it is consistent regardless, and
    /// the panic itself already propagates on its own thread.
    fn free_list(&self) -> MutexGuard<'_, Vec<Vec<u8>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// This pool's hit/miss probes; the pipeline owning it
    /// [`absorb`](crate::probe::absorb)s them on its caller thread.
    #[cfg(test)]
    pub(crate) fn probes(&self) -> &SharedProbes {
        &self.probes
    }

    /// Number of buffers currently sitting in the free list (test/debug
    /// visibility into pool behavior; not part of the pool's steady-state
    /// contract, since the "right" free-list size varies with pipeline
    /// depth and thread count).
    #[cfg(test)]
    pub(crate) fn free_count(&self) -> usize {
        self.free_list().len()
    }

    /// Seed the free list with `count` buffers, each already grown to
    /// `len` bytes, in ONE lock acquisition — a bulk-`give` for a caller
    /// that knows its own steady-state depth up front.
    ///
    /// # Why this exists (measured, 2026-09-25)
    /// A read-ahead pipeline with a bounded one-window-deep channel (the
    /// scanline/tiled parallel readers) can have at most TWO windows'
    /// worth of chunks "outstanding" at once: the window the decode
    /// workers are currently draining, plus the next window the producer
    /// is fetching into fresh buffers concurrently. Starting the pool
    /// EMPTY means every buffer in those first ~2 windows is a genuine
    /// cold `Vec::default()` -> [`ensure_len`] growth, which zero-fills
    /// the whole buffer (`Vec::resize`'s documented behavior) -- real,
    /// measured cost, not free. Worse, at HIGHER thread counts each
    /// window is BIGGER (`ThreadCount::max_in_flight` = `2 * num_threads`
    /// chunks) and decode finishes FASTER (more parallelism), so
    /// buffers cycle back to the free list less often relative to how
    /// fast the producer consumes them -- the pool can stay chronically
    /// "cold" for most of a large read instead of warming up after one
    /// window, and MORE of the read's chunks (not just the first window's)
    /// pay the full zero-fill cost. Measured on a 8192x4096 half-RGBA ZIP
    /// scanline read (`fetch_chunk`'s own `BufPool::take` call, temporary
    /// instrumentation, not committed): `ensure_len`'s cost inside `take`
    /// grew from ~8ms (num_threads=2, tiny 4-chunk windows, pool warms up
    /// almost immediately) to ~84ms (num_threads=24, 48-chunk windows) for
    /// the SAME 256 chunks off the SAME in-memory bytes -- the mutex
    /// acquisition itself stayed flat at a fraction of a millisecond both
    /// times, ruling out lock contention; `ensure_len`'s zero-fill work
    /// itself was the growing cost, consistent with far more of the 256
    /// chunks needing a cold/fresh buffer at the higher thread count.
    /// Pre-seeding the pool with enough right-sized buffers up front pays
    /// that same total zero-fill cost exactly ONCE, before the parallel
    /// loop starts, instead of paying it repeatedly as the free list
    /// starves mid-read.
    pub fn prewarm(&self, count: usize, len: usize) {
        if count == 0 {
            return;
        }
        let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = Vec::new();
            ensure_len(&mut buf, len);
            bufs.push(buf);
        }
        self.free_list().extend(bufs);
    }

    /// [`Self::prewarm`], sized to a parallel pipeline's peak number of
    /// outstanding items: `windows_in_flight` windows of `window` items
    /// each, capped at `total_items` so a short read/write never
    /// over-allocates. The depth depends on the pipeline's shape, and each
    /// caller passes its own:
    /// - readers ([`READ_WINDOWS_IN_FLIGHT`]): one window decoding, one
    ///   queued in the one-slot channel, one being fetched by the producer;
    /// - writers ([`WRITE_WINDOWS_IN_FLIGHT`]): a ring of one window of items,
    ///   each replaced only after the oldest was written and given back.
    ///
    /// Measured (U4b, 2048x2048 ZIP tiles, 2/8/24 threads): with the old
    /// shared depth of 2 the readers missed exactly one window of buffers
    /// (32-48 cold zero-filled allocations per read) while the writers
    /// held one window more than they ever used. The `BufPoolMiss` probe
    /// pins both.
    pub fn prewarm_for_pipeline(&self, windows_in_flight: usize, window: usize, item_len: usize, total_items: usize) {
        self.prewarm(window.saturating_mul(windows_in_flight).min(total_items), item_len);
    }
}

impl Default for BufPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_len_grows_and_truncates() {
        let mut buf: Vec<u8> = vec![1, 2, 3];
        ensure_len(&mut buf, 5);
        assert_eq!(buf.len(), 5);
        assert_eq!(&buf[..3], &[1, 2, 3]);
        ensure_len(&mut buf, 2);
        assert_eq!(buf, vec![1, 2]);
    }

    /// A `take`n-then-`give`n buffer is handed back out by a later `take`
    /// instead of a fresh allocation -- the pool's whole reason to exist.
    ///
    /// Mutation check: changed `take` to always `Vec::new()` (ignoring the
    /// free list) -- this test failed (capacity was 0 instead of the
    /// pre-grown value, and `free_count()` after `give` was 0 as expected
    /// but the SECOND `take`'s returned capacity no longer matched the
    /// first's, since a fresh Vec::new() has capacity 0). Reverted after
    /// confirming red.
    #[test]
    fn a_returned_buffer_is_reused_not_reallocated() {
        let pool = BufPool::new();
        let mut buf = pool.take(4096);
        assert!(buf.capacity() >= 4096);
        let cap = buf.capacity();
        buf.fill(0xAA); // simulate use
        pool.give(buf);
        assert_eq!(pool.free_count(), 1);

        let buf2 = pool.take(4096);
        assert_eq!(pool.free_count(), 0, "take must pop from the free list");
        assert_eq!(buf2.capacity(), cap, "the exact same allocation must come back");
    }

    /// `take` still works correctly (right length, right default content in
    /// the grown tail) from an empty pool, and after `give`-ing back a
    /// buffer SMALLER than the next `take` request (must still grow it).
    #[test]
    fn take_from_empty_pool_and_take_larger_than_given() {
        let pool = BufPool::new();
        let buf = pool.take(8);
        assert_eq!(buf.len(), 8);
        pool.give(buf);

        let buf2 = pool.take(16);
        assert_eq!(buf2.len(), 16);
    }

    /// `prewarm` seeds exactly `count` right-sized buffers, so the next
    /// `count` `take` calls all get an already-grown buffer (no growth
    /// needed) and `free_count()` reflects the seed immediately.
    ///
    /// Mutation check: changed `prewarm` to seed buffers of `len - 1`
    /// bytes instead of `len` -- this test failed (`buf.len()` after
    /// `take(len)` was still `len` as expected, since `take` re-runs
    /// `ensure_len`, but `buf.capacity()` right after `prewarm` alone, via
    /// a direct pop, was one byte short of `len`, i.e. `take` was doing
    /// real growth work the prewarm was supposed to have already done).
    /// Reverted after confirming red.
    #[test]
    fn prewarm_seeds_right_sized_buffers_take_needs_no_growth() {
        let pool = BufPool::new();
        pool.prewarm(3, 128);
        assert_eq!(pool.free_count(), 3);

        for _ in 0..3 {
            let buf = pool.take(128);
            assert_eq!(buf.len(), 128);
            assert_eq!(buf.capacity(), 128, "prewarmed buffer must already be exactly the right size");
            // Don't give it back -- confirms each of the 3 seeded buffers
            // is independently usable, not the same one popped 3 times.
        }
        assert_eq!(pool.free_count(), 0, "all 3 prewarmed buffers must have been consumed");
        assert_eq!(pool.probes().get(Probe::BufPoolHit), 3, "every take after prewarm must hit");
        assert_eq!(pool.probes().get(Probe::BufPoolMiss), 0);
        let _cold = pool.take(128);
        assert_eq!(pool.probes().get(Probe::BufPoolMiss), 1, "an empty free list must count a miss");
    }

    /// `prewarm(0, ..)` is a documented no-op (a caller computing a
    /// pipeline depth of 0, e.g. zero chunks to read, must not panic or
    /// allocate).
    #[test]
    fn prewarm_zero_is_a_no_op() {
        let pool = BufPool::new();
        pool.prewarm(0, 4096);
        assert_eq!(pool.free_count(), 0);
    }
}
