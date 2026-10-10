//! Port of `ImfThreading.{h,cpp}` — the global thread-count API — plus the
//! single process-wide `rayon` pool it backs, and a per-file thread-count
//! capture helper.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfThreading.cpp>
//! and `IlmThread/IlmThreadPool.cpp` (behavioural target 3.4.15; unchanged
//! there).
//!
//! # C++ semantics ported here
//! - `globalThreadCount()` / `setGlobalThreadCount(n)` read/write ONE process
//!   -wide pool (`IlmThreadPool.cpp`: `static ThreadPool gThreadPool(0)`,
//!   `IlmThread::ThreadPool::globalThreadPool()`). **The default is 0** —
//!   without TBB, C++ ships single-threaded by default; we match that
//!   default exactly.
//! - A negative count is rejected as `Iex::ArgExc` (`IlmThreadPool.cpp:626-633`)
//!   → [`ExrError::Arg`] here.
//! - With a count of 0 (or 1 — no parallelism gain), C++ runs each task inline
//!   on the calling thread (`handleProcessTask`, `IlmThreadPool.cpp:675-688`).
//!   [`run_on_global_pool`] gets the same net effect with a 1-worker pool:
//!   calling the closure bare would let its `par_iter` spill onto rayon's
//!   ambient all-cores pool.
//! - **Pools are cached per distinct thread count** ([`pool_for_count`]).
//!   Every file that captures the default count shares one pool, which is
//!   C++'s "at most this number of worker threads regardless of how many
//!   files are open" (`ImfThreading.h`) for the common case; a file with an
//!   explicit `numThreads` gets a pool of exactly that size. This matters for
//!   `exr-view`, which can have hundreds of sequence frames open: one pool per
//!   count, never one per file.
//! - Every reader/writer constructor still captures `numThreads` at
//!   construction — an explicit argument if given, else the *current* global
//!   count, read once and then fixed for that file's lifetime.
//!   [`ThreadCount::capture`] is that capture point, for the
//!   `with_num_threads` constructors. The captured count picks the pool
//!   [`run_on_global_pool`] runs on and bounds how many of that file's chunks
//!   are in flight (C++'s `max(1, 2*numThreads)`-buffer ring).

use std::sync::atomic::{AtomicI32, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::error::{ExrError, Result};

// ---------------------------------------------------------------------------
// Global thread count (port of `ImfThreading.{h,cpp}`)
// ---------------------------------------------------------------------------

/// Process-wide default thread count. **0 = serial**, matching C++'s
/// non-TBB default (`IlmThreadPool.cpp:697-714`).
static GLOBAL_THREAD_COUNT: AtomicI32 = AtomicI32::new(0);

/// Port of `Imf::globalThreadCount()`.
///
/// Returns the process-wide default thread count last set by
/// [`set_global_thread_count`], or `0` (serial) if it was never called.
#[inline]
pub fn global_thread_count() -> i32 {
    GLOBAL_THREAD_COUNT.load(Ordering::SeqCst)
}

/// Port of `Imf::setGlobalThreadCount(int)`.
///
/// A negative count is `Iex::ArgExc` in C++ (`IlmThreadPool.cpp:626-633`);
/// ported as [`ExrError::Arg`] rather than a panic (no-panic rule). The pool
/// for the new count is built lazily by the next [`global_pool`] /
/// [`run_on_global_pool`] call that needs it.
pub fn set_global_thread_count(num_threads: i32) -> Result<()> {
    if num_threads < 0 {
        return Err(ExrError::Arg(format!(
            "numThreads must be non-negative (got {num_threads})"
        )));
    }
    GLOBAL_THREAD_COUNT.store(num_threads, Ordering::SeqCst);
    Ok(())
}

// ---------------------------------------------------------------------------
// Pools: one per distinct thread count
// ---------------------------------------------------------------------------

/// Every pool this process uses, keyed by worker count. Files that capture
/// the same count share one pool (the default-everywhere case is therefore
/// C++'s single shared pool, `ImfThreading.h`); a file with an explicit
/// different `numThreads` gets a pool of exactly that size. Pools live for the
/// process, like C++'s global pool; the key set is bounded by the distinct
/// counts callers ask for.
type PoolCache = RwLock<HashMap<i32, Arc<rayon::ThreadPool>>>;
static POOLS: OnceLock<PoolCache> = OnceLock::new();

/// A poisoned lock means some other thread panicked while holding it; this is
/// reported as `ExrError::Base` (no-panic rule: never `.unwrap()` a poisoned
/// lock).
fn lock_poisoned<T>(_: T) -> ExrError {
    ExrError::Base("thread pool cache lock poisoned".to_string())
}

/// The pool with exactly `n >= 1` workers, built on first use and cached.
fn pool_for_count(n: i32) -> Result<Arc<rayon::ThreadPool>> {
    let pools = POOLS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(p) = pools.read().map_err(lock_poisoned)?.get(&n) {
        return Ok(p.clone());
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1) as usize)
        .build()
        .map_err(|e| ExrError::Base(format!("failed to build a {n}-thread pool: {e}")))?;
    // Another thread may have inserted `n` while this one was building; keep
    // the first so every caller shares one pool per count.
    let mut pools = pools.write().map_err(lock_poisoned)?;
    Ok(pools.entry(n).or_insert_with(|| Arc::new(built)).clone())
}

/// Whether the calling thread is a worker of ANY pool this module built.
/// [`ordered_pipeline`] runs inline on such a thread (see its docs): a worker
/// that blocks waiting for another worker's tasks is what deadlocks.
fn on_exr_pool_worker() -> bool {
    // A poisoned lock still holds a consistent map (pools are only ever inserted);
    // answering "not a worker" on poison would send a worker into the blocking path.
    POOLS.get().is_some_and(|pools| {
        let pools = pools.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        pools.values().any(|p| p.current_thread_index().is_some())
    })
}

/// The shared pool for the CURRENT [`global_thread_count`], or `None` when
/// that count is `<= 1` (serial). The returned `Arc` keeps the pool alive for
/// an in-flight caller even if the global count changes meanwhile.
pub fn global_pool() -> Result<Option<Arc<rayon::ThreadPool>>> {
    let n = global_thread_count();
    if n <= 1 { Ok(None) } else { pool_for_count(n).map(Some) }
}

/// Run `f` on a pool of exactly `count` workers (`0` and `1` both mean one
/// worker, i.e. serial).
///
/// `f` always runs inside `install`, never bare: every caller's `f` does
/// `par_iter` work, and a `par_iter` outside any installed pool silently runs
/// on rayon's ambient default pool (one thread per core), so
/// `with_num_threads(Some(n))` would run on all cores whenever the global
/// count was `<= 1`.
pub fn run_on_global_pool<F, R>(count: ThreadCount, f: F) -> Result<R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    Ok(pool_for_count(count.get().max(1))?.install(f))
}

// ---------------------------------------------------------------------------
// Per-file thread-count capture (for the `with_num_threads` ctors)
// ---------------------------------------------------------------------------

/// A validated, already-resolved thread count captured at reader/writer
/// construction time.
///
/// C++ gives every `InputFile`/`OutputFile`/... constructor a
/// `int numThreads = globalThreadCount()` default parameter: the value is
/// read **once**, at construction, not re-read from the global on every
/// chunk. [`ThreadCount::capture`] is that one read, so a later
/// `set_global_thread_count` call cannot change an already-open file's
/// notion of its own concurrency — matching C++.
///
/// The count sizes the pool the file's chunks run on ([`run_on_global_pool`])
/// and bounds how many are in flight at once ([`ThreadCount::max_in_flight`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadCount(i32);

impl ThreadCount {
    /// Resolve the effective thread count for a new reader/writer:
    /// `explicit` if given (validated the same way as
    /// [`set_global_thread_count`]), else the current [`global_thread_count`]
    /// (captured now, fixed for the file's lifetime).
    pub fn capture(explicit: Option<i32>) -> Result<Self> {
        let n = explicit.unwrap_or_else(global_thread_count);
        if n < 0 {
            return Err(ExrError::Arg(format!(
                "numThreads must be non-negative (got {n})"
            )));
        }
        Ok(Self(n))
    }

    /// The raw captured count (may be 0 = serial, or 1 — both run inline;
    /// see [`run_on_global_pool`]).
    #[inline]
    pub fn get(self) -> i32 {
        self.0
    }

    /// Whether this count would run chunks in parallel at all (`> 1`,
    /// matching C++'s `nchunks > 1 && numThreads > 1` gate — the `numThreads`
    /// half of it).
    #[inline]
    pub fn is_parallel(self) -> bool {
        self.0 > 1
    }

    /// How many of this file's chunks the pipelines should keep outstanding on the
    /// shared [`global_pool`] at once: C++'s `max(1, 2*numThreads)`
    /// (`ImfOutputFile.cpp:181-205` and the tiled/deep equivalents) - the
    /// ring [`ordered_pipeline`] runs the writers on, and the floor of
    /// [`Self::window_size_for_bytes`].
    #[inline]
    pub fn max_in_flight(self) -> usize {
        (2 * self.0).max(1) as usize
    }

    /// How many chunks/tiles the read-ahead pipeline should batch into ONE
    /// window, so that parallel is never slower than
    /// serial. [`Self::max_in_flight`] alone sizes a window by CHUNK COUNT,
    /// which is fine for large chunks (ZIP/PIZ/DWA) but pathological for
    /// small ones: `tests/perf_io_vs_decode.rs::many_small_windows_thread_spawn_overhead_65536_height_none`
    /// found NONE compression (1 scanline/chunk) at 65536 height still only
    /// 0.85x of serial even after fixing the one-thread-per-window-transition
    /// bug, because a `max_in_flight()`-sized window of tiny chunks (a few KB
    /// total) does not do enough work to amortize the channel-handoff +
    /// `rayon` dispatch cost of processing it. This grows the window (in
    /// CHUNK COUNT, since callers batch by count) until it covers at least
    /// `min_bytes_per_window` bytes of raw/decoded work — the same idea as
    /// C++'s per-task granularity, sized directly from `bytes_per_chunk`
    /// (an upper bound the caller already knows: `bytes_per_line *
    /// lines_in_buffer` for scanline, `bytes_per_pixel * tile_w * tile_h` for
    /// tiled — never measured per-chunk, since that would need reading the
    /// chunk first).
    #[inline]
    pub fn window_size_for_bytes(self, bytes_per_chunk: usize) -> usize {
        /// Empirically-motivated floor (not derived from a specific machine
        /// number): large enough that `rayon`'s per-task dispatch and this
        /// crate's per-window channel round-trip are a small fraction of the
        /// work, small enough that read-ahead still overlaps I/O with decode
        /// at a fine enough grain to matter for a handful of huge chunks.
        const MIN_BYTES_PER_WINDOW: usize = 256 * 1024;
        let by_count = self.max_in_flight();
        if bytes_per_chunk == 0 {
            return by_count;
        }
        let by_bytes = MIN_BYTES_PER_WINDOW.div_ceil(bytes_per_chunk).max(1);
        by_count.max(by_bytes)
    }
}

// ---------------------------------------------------------------------------
// Ordered produce/emit pipeline (the writers' line-buffer ring)
// ---------------------------------------------------------------------------

/// One `produce` call's outcome crossing the result channel: its `Result`, or
/// the panic payload (re-raised on the calling thread once the pool is idle).
enum Done<T> {
    Ok(Result<T>),
    Panic(Box<dyn std::any::Any + Send>),
}

/// How many items, and how many bytes of them, [`ordered_pipeline`] keeps
/// outstanding (submitted but not yet emitted).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ring {
    /// Item cap (C++'s `2 * numThreads` buffers, or more for tiny items).
    pub items: usize,
    /// Byte cap over the items' `weight`s; at least one item is always
    /// admitted, so a single huge item cannot stall the pipeline.
    pub bytes: usize,
}

#[cfg(test)]
thread_local! {
    /// Test seam: a byte cap for the deep writers' ring on this thread, so the tests
    /// can prove the per-item weights are accounted. Production has no byte cap (see
    /// [`Ring::deep`]).
    static DEEP_RING_BYTES_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Test probe: the largest sum of outstanding item weights this thread's
    /// last pipelines reached (see [`take_peak_bytes`]).
    static PEAK_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn set_deep_ring_bytes(bytes: Option<usize>) {
    DEEP_RING_BYTES_OVERRIDE.with(|c| c.set(bytes));
}

/// The peak outstanding weight since the last call; resets it.
#[cfg(test)]
pub(crate) fn take_peak_bytes() -> usize {
    PEAK_BYTES.with(|c| c.replace(0))
}

impl Ring {
    /// A ring bounded by item count only (items of uniform, known size).
    pub(crate) fn items(items: usize) -> Self {
        Ring { items, bytes: usize::MAX }
    }

    /// The deep writers' ring: C++'s `2 * numThreads` items
    /// (`ImfDeepScanLineOutputFile.cpp:1232-1240`), whatever their sample data
    /// weighs, as in C++. No byte cap is invented here (repo rule: memory budgets
    /// come from the caller's one `Budget`, never from a constant); the items carry
    /// their exact weights, so a caller budget can be plumbed into `bytes` later.
    pub(crate) fn deep(count: ThreadCount) -> Self {
        #[cfg(test)]
        let bytes = DEEP_RING_BYTES_OVERRIDE.with(|c| c.get()).unwrap_or(usize::MAX);
        #[cfg(not(test))]
        let bytes = usize::MAX;
        Ring { items: count.max_in_flight(), bytes }
    }
}

/// Run `produce(0..n_items)` on the shared pool and hand every result to
/// `emit` on the CALLING thread, strictly in index order, as soon as it is
/// ready - while later items are still being produced.
///
/// This is C++'s line-buffer ring (`2 * numThreads` buffers, `Data::Data`,
/// `ImfOutputFile.cpp:181-193`; the same shape in `ImfTiledOutputFile.cpp`,
/// `ImfDeepScanLineOutputFile.cpp`, `ImfDeepTiledOutputFile.cpp`):
/// `writePixels` first queues `min(ring, n)` compression tasks
/// (`ImfOutputFile.cpp:1039-1054`), then loops `wait for the NEXT buffer in
/// order -> writePixelData -> queue one more task`
/// (`:1098-1101`, `:1135`, `:1174-1185`). A window-at-a-time design instead
/// idles the pool during emission/I-O and makes every window wait for its
/// slowest item; here the pool only ever waits for the oldest unemitted item.
///
/// - **Bounded.** Items are admitted while fewer than `ring.items` are
///   outstanding (queued, running, finished-and-waiting-for-order, or being
///   emitted) and their summed `weight(i)` stays within `ring.bytes` (the
///   first outstanding item is always admitted).
/// - **Ordered, off-thread-free emit.** `emit` runs on the caller, so it may
///   own non-`Send` state (the `OStream`); only `produce` crosses threads.
/// - **Deterministic errors.** Items are examined in index order and the first
///   `Err` - from `produce(i)` or `emit(i, _)`, whichever belongs to the lowest
///   index - is returned; nothing is emitted after it and no further item is
///   submitted. A `produce` error at a higher index than a failing `emit` is
///   never reported, whatever order they happened in.
/// - **Joined.** Every submitted task has finished before this returns (the
///   rayon scope waits), so `produce`'s borrows are never outlived.
/// - **Panics** follow `resume_unwind`: a panic in `produce` is caught on the
///   worker (so the loop cannot wait forever for a result that never comes),
///   and once the pool is idle it is re-raised on the calling thread with
///   `std::panic::resume_unwind` - the contract of the `par_iter` this
///   replaces - unless an `Err` of a LOWER index is reported first. Panics
///   that are not re-raised (behind an earlier error or panic) are logged at
///   `error` level, never silently dropped. A panic in `emit` unwinds through
///   the rayon scope, which joins every task first.
/// - **Called from a worker of ANY of this crate's pools** (nested
///   parallelism: a writer invoked inside `run_on_global_pool`'s own
///   `par_iter`/`join`), the items run inline, in order, on that worker. The
///   loop blocks on a channel and cannot steal, so with every worker of a pool
///   blocked like that nobody would run the tasks; C++ likewise runs a task
///   inline when no thread is free (`handleProcessTask`). "Any pool", not
///   just the target one, rules out cross-pool cycles too (pool A's workers
///   blocked on pool B while B's workers are blocked on A): an exr-core
///   worker NEVER blocks here, and `produce` is leaf codec work that never
///   waits on anything, so a pool's tasks always finish. A caller on a
///   FOREIGN rayon pool's worker (the app's own pool) can therefore only wait
///   on exr-core workers that never block - no cycle - but it blocks without
///   stealing its own pool's work, a throughput cost only. The price of the
///   inline rule: a writer started from an exr-core worker is serial.
pub(crate) fn ordered_pipeline<T, P, E>(
    count: ThreadCount,
    n_items: usize,
    ring: Ring,
    weight: impl Fn(usize) -> usize,
    produce: P,
    mut emit: E,
) -> Result<()>
where
    T: Send,
    P: Fn(usize) -> Result<T> + Sync,
    E: FnMut(usize, T) -> Result<()>,
{
    if n_items == 0 {
        return Ok(());
    }
    let pool = pool_for_count(count.get().max(1))?;
    if on_exr_pool_worker() {
        log::debug!("ordered_pipeline: called from a pool worker, running {n_items} items inline");
        for i in 0..n_items {
            emit(i, produce(i)?)?;
        }
        return Ok(());
    }

    type Tx<T> = std::sync::mpsc::Sender<(usize, Done<T>)>;
    let max_items = ring.items.max(1);
    let busy = crate::probe::Busy::default();
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Done<T>)>();
    let mut tx = Some(tx);
    let mut failure: Option<ExrError> = None;
    let mut panicked: Option<Box<dyn std::any::Any + Send>> = None;
    // Finished items that arrived ahead of their turn.
    let mut ready: std::collections::BTreeMap<usize, Done<T>> = std::collections::BTreeMap::new();

    pool.in_place_scope(|scope| {
        let mut submitted = 0usize;
        let mut next = 0usize;
        let mut bytes = 0usize;
        // Submit while the ring has room.
        let fill = |submitted: &mut usize, next: usize, bytes: &mut usize, tx: &mut Option<Tx<T>>| {
            while *submitted < n_items && *submitted - next < max_items {
                let w = weight(*submitted);
                if *submitted > next && bytes.saturating_add(w) > ring.bytes {
                    break;
                }
                let Some(sender) = tx.as_ref() else { return };
                let (i, sender) = (*submitted, sender.clone());
                let (produce, busy) = (&produce, &busy);
                scope.spawn(move |_| {
                    busy.enter();
                    let done = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| produce(i))) {
                        Ok(r) => Done::Ok(r),
                        Err(p) => Done::Panic(p),
                    };
                    busy.exit();
                    // The receiver outlives the scope; a failed send would
                    // mean the consumer is gone and the result moot.
                    if sender.send((i, done)).is_err() {
                        log::trace!("ordered_pipeline: item {i} finished after the consumer stopped");
                    }
                });
                *bytes = bytes.saturating_add(w);
                *submitted += 1;
                #[cfg(test)]
                PEAK_BYTES.with(|c| c.set(c.get().max(*bytes)));
            }
            // After the last submit the original sender goes away, so a lost
            // task shows up as a closed channel (clean error), not a hang.
            if *submitted == n_items {
                *tx = None;
            }
        };

        fill(&mut submitted, next, &mut bytes, &mut tx);
        while next < n_items {
            let done = loop {
                if let Some(d) = ready.remove(&next) {
                    break d;
                }
                match rx.recv() {
                    Ok((j, d)) => {
                        ready.insert(j, d);
                    }
                    Err(_) => {
                        failure = Some(ExrError::Logic("ordered_pipeline: a task ended without a result".into()));
                        return;
                    }
                }
            };
            match done {
                Done::Panic(p) => {
                    panicked = Some(p);
                    return;
                }
                Done::Ok(Err(e)) => {
                    failure = Some(e);
                    return;
                }
                Done::Ok(Ok(item)) => {
                    // Emitting while another item is mid-produce is the whole
                    // point of the ring; a window barrier can never get here.
                    if busy.any() {
                        crate::probe::hit(crate::probe::Probe::WriteOverlap);
                    }
                    if let Err(e) = emit(next, item) {
                        failure = Some(e);
                        return;
                    }
                }
            }
            bytes = bytes.saturating_sub(weight(next));
            next += 1;
            fill(&mut submitted, next, &mut bytes, &mut tx);
        }
    });

    // Whatever is still buffered or queued belongs to items behind the
    // reported outcome; a panic among them must not vanish.
    drop(tx);
    for (j, d) in ready.into_iter().chain(rx.try_iter()) {
        if let Done::Panic(p) = d {
            log::error!("ordered_pipeline: item {j} panicked behind an earlier outcome: {}", panic_msg(&*p));
        }
    }
    if let Some(p) = panicked {
        std::panic::resume_unwind(p);
    }
    failure.map_or(Ok(()), Err)
}

/// Best-effort text of a panic payload, for logging.
fn panic_msg(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// IMPORTANT: no test in this module may call `set_global_thread_count` — it
// mutates process-wide state (both the atomic AND the shared pool) visible
// to every test in this lib test binary, which would make results depend on
// run order / `--test-threads`. Tests that need to mutate the global count
// (including anything that needs an actual sized pool to exist) live in
// their own integration binary, `tests/global_threading.rs`, per the task's
// isolation requirement.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_global_thread_count_is_zero_serial() {
        // Safe to assert unconditionally: nothing else in this lib test
        // binary is allowed to call `set_global_thread_count`.
        assert_eq!(global_thread_count(), 0);
    }

    #[test]
    fn default_global_pool_is_none_serial() {
        // Never having called `set_global_thread_count`, the pool must still
        // be buildable (lazily, for count 0) and absent.
        assert!(global_pool().unwrap().is_none());
    }

    #[test]
    fn thread_count_capture_explicit_overrides_global() {
        // Explicit values are validated independently of the (unmodified,
        // 0) global — no reliance on the global's exact value beyond it
        // being non-negative, which it always is.
        let tc = ThreadCount::capture(Some(4)).unwrap();
        assert_eq!(tc.get(), 4);
        assert!(tc.is_parallel());
    }

    #[test]
    fn thread_count_capture_none_uses_current_global() {
        let tc = ThreadCount::capture(None).unwrap();
        assert_eq!(tc.get(), global_thread_count());
    }

    #[test]
    fn thread_count_capture_negative_explicit_is_rejected() {
        match ThreadCount::capture(Some(-5)) {
            Err(ExrError::Arg(_)) => {}
            other => panic!("expected ExrError::Arg, got {other:?}"),
        }
    }

    #[test]
    fn thread_count_zero_and_one_are_not_parallel() {
        assert!(!ThreadCount::capture(Some(0)).unwrap().is_parallel());
        assert!(!ThreadCount::capture(Some(1)).unwrap().is_parallel());
    }

    #[test]
    fn thread_count_max_in_flight_matches_cpp_ring_formula() {
        assert_eq!(ThreadCount::capture(Some(0)).unwrap().max_in_flight(), 1);
        assert_eq!(ThreadCount::capture(Some(1)).unwrap().max_in_flight(), 2);
        assert_eq!(ThreadCount::capture(Some(4)).unwrap().max_in_flight(), 8);
    }

    #[test]
    fn run_on_global_pool_serial_count_runs() {
        let count = ThreadCount::capture(Some(0)).unwrap();
        let result = run_on_global_pool(count, || 21 * 2).unwrap();
        assert_eq!(result, 42);
    }

    /// The work must run on a pool of exactly `n` workers, never on
    /// rayon's ambient all-cores pool. Byte identity can't show this; the
    /// worker count seen inside the closure can. Mutation (the old
    /// `is_parallel()` gate with a bare `f()` fallback) turns it red.
    #[test]
    fn run_on_global_pool_confines_work_to_exactly_n_threads() {
        for n in [1i32, 2, 4] {
            let count = ThreadCount::capture(Some(n)).unwrap();
            let observed =
                run_on_global_pool(count, rayon::current_num_threads).unwrap();
            assert_eq!(
                observed, n as usize,
                "n={n}: run_on_global_pool must confine work to a pool of exactly n threads, \
                 not rayon's ambient default (this machine's core count)"
            );
        }
    }

    // --- ordered_pipeline -------------------------------------------------

    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::time::{Duration, Instant};

    /// Item-count-only ring (the flat writers' shape).
    fn pipe<T: Send>(
        count: ThreadCount,
        n: usize,
        window: usize,
        produce: impl Fn(usize) -> Result<T> + Sync,
        emit: impl FnMut(usize, T) -> Result<()>,
    ) -> Result<()> {
        ordered_pipeline(count, n, Ring::items(window), |_| 0, produce, emit)
    }

    fn tc(n: i32) -> ThreadCount {
        ThreadCount::capture(Some(n)).unwrap()
    }

    /// Spin until `flag` is set; `Err` after 5 s so a broken pipeline fails
    /// the test instead of hanging it.
    fn wait_for(flag: &AtomicBool) -> Result<()> {
        let t0 = Instant::now();
        while !flag.load(Ordering::SeqCst) {
            if t0.elapsed() > Duration::from_secs(5) {
                return Err(ExrError::Logic("timed out waiting for the pipeline".into()));
            }
            std::thread::yield_now();
        }
        Ok(())
    }

    fn arg_msg(e: ExrError) -> String {
        match e {
            ExrError::Arg(m) => m,
            other => panic!("expected ExrError::Arg, got {other:?}"),
        }
    }

    #[test]
    fn ordered_pipeline_emits_every_item_once_in_index_order() {
        for threads in [2, 4, 9] {
            for window in [1usize, 3, 8, 100] {
                let mut got = Vec::new();
                pipe(
                    tc(threads),
                    50,
                    window,
                    |i| {
                        // Reverse the natural completion order.
                        std::thread::sleep(Duration::from_micros(((50 - i) * 20) as u64));
                        Ok(i * 2)
                    },
                    |i, v| {
                        assert_eq!(v, i * 2);
                        got.push(i);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(got, (0..50).collect::<Vec<_>>(), "threads={threads} window={window}");
            }
        }
    }

    #[test]
    fn ordered_pipeline_zero_items_is_ok() {
        pipe(tc(2), 0, 4, |_| Ok(()), |_, ()| Ok(())).unwrap();
    }

    /// Item 1 can only finish after item 0 was emitted, and item 0 only
    /// finishes once item 1 is mid-produce: so item 0's emission MUST overlap
    /// item 1's production. A window barrier (emit after the whole window is
    /// produced) times item 1 out and fails; so does an overlap probe that
    /// never fires.
    #[test]
    fn ordered_pipeline_emits_while_later_items_are_still_producing() {
        let one_entered = AtomicBool::new(false);
        let zero_emitted = AtomicBool::new(false);
        crate::probe::reset(crate::probe::Probe::WriteOverlap);
        pipe(
            tc(2),
            4,
            4,
            |i| match i {
                0 => wait_for(&one_entered),
                1 => {
                    one_entered.store(true, Ordering::SeqCst);
                    wait_for(&zero_emitted)
                }
                _ => Ok(()),
            },
            |i, ()| {
                if i == 0 {
                    zero_emitted.store(true, Ordering::SeqCst);
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(
            crate::probe::get(crate::probe::Probe::WriteOverlap) > 0,
            "item 0 was emitted with nothing else producing"
        );
    }

    /// With a slow emitter and instant producers the pool would race ahead
    /// without a bound: `live` (entered produce, not yet emitted) stays within
    /// the window, and the window does fill (it is a pipeline, not serial).
    #[test]
    fn ordered_pipeline_never_exceeds_max_in_flight() {
        let (n, window) = (40usize, 4usize);
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        pipe(
            tc(4),
            n,
            window,
            |_| {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                Ok(())
            },
            |_, ()| {
                std::thread::sleep(Duration::from_millis(2));
                live.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= window, "{peak} items in flight, bound is {window}");
        assert!(peak >= 2, "the window never filled (peak {peak}): not pipelined");
    }

    /// The reported error is the one of the LOWEST index, whichever failure
    /// happened first in time, and nothing is emitted past it.
    #[test]
    fn ordered_pipeline_reports_the_lowest_index_error() {
        // A slow produce error at 1 beats a fast one at 5.
        let mut emitted = Vec::new();
        let e = pipe(
            tc(4),
            12,
            12,
            |i| match i {
                1 => {
                    std::thread::sleep(Duration::from_millis(40));
                    Err(ExrError::Arg("p1".into()))
                }
                5 => Err(ExrError::Arg("p5".into())),
                _ => Ok(()),
            },
            |i, ()| {
                emitted.push(i);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(arg_msg(e), "p1");
        assert_eq!(emitted, vec![0]);

        // An emit error at 2 beats a produce error at 4 that is ready first.
        let mut emitted = Vec::new();
        let e = pipe(
            tc(4),
            12,
            12,
            |i| if i == 4 { Err(ExrError::Arg("p4".into())) } else { Ok(()) },
            |i, ()| {
                emitted.push(i);
                if i == 2 { Err(ExrError::Arg("e2".into())) } else { Ok(()) }
            },
        )
        .unwrap_err();
        assert_eq!(arg_msg(e), "e2");
        assert_eq!(emitted, vec![0, 1, 2]);
    }

    #[test]
    fn ordered_pipeline_joins_all_work_and_reraises_a_producer_panic() {
        let finished = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(|| {
            pipe(
                tc(3),
                20,
                20,
                |i| {
                    if i == 3 {
                        panic!("boom");
                    }
                    std::thread::sleep(Duration::from_millis(1));
                    finished.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                |_, ()| Ok(()),
            )
        });
        assert!(r.is_err(), "the producer panic must reach the caller");
        // Everything submitted had returned before the caller saw the panic
        // (items 0-2 are emitted-before-panic; later ones are only joined).
        assert!(finished.load(Ordering::SeqCst) >= 3);

        // A lower-index error still outranks a later panic.
        let r = std::panic::catch_unwind(|| {
            pipe(
                tc(3),
                20,
                20,
                |i| match i {
                    1 => Err(ExrError::Arg("p1".into())),
                    3 => panic!("boom"),
                    _ => Ok(()),
                },
                |_, ()| Ok(()),
            )
        });
        assert_eq!(arg_msg(r.expect("no panic escapes").unwrap_err()), "p1");
    }

    /// Run `f` on a helper thread and fail (not hang) if it takes over 30 s.
    fn watchdog<R: Send + 'static>(what: &str, f: impl FnOnce() -> R + Send + 'static) -> R {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            tx.send(f()).ok();
        });
        rx.recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| panic!("{what}: deadlocked (watchdog)"))
    }

    /// One nested pipeline that must complete in order: `threads` picks the
    /// pool it would submit to.
    fn nested(threads: i32) -> Vec<usize> {
        let mut got = Vec::new();
        pipe(tc(threads), 16, 4, |i| Ok(i * 3), |i, v| {
            assert_eq!(v, i * 3);
            got.push(i);
            Ok(())
        })
        .unwrap();
        got
    }

    fn all_complete(runs: Vec<Vec<usize>>) {
        for got in runs {
            assert_eq!(got, (0..16).collect::<Vec<_>>());
        }
    }

    /// The reported deadlock, shape (E): pool of 8, eight concurrent writers
    /// with num_threads=8 started from that pool's own workers. Every worker
    /// used to block on its pipeline's channel, so nobody ran the tasks. The
    /// watchdog turns a regression into a failure instead of a hung suite.
    #[test]
    fn ordered_pipeline_from_pool_workers_8_on_8_does_not_deadlock() {
        use rayon::prelude::*;
        let runs = watchdog("8 writers on the 8-pool", || {
            run_on_global_pool(tc(8), || (0..8).into_par_iter().map(|_| nested(8)).collect::<Vec<_>>()).unwrap()
        });
        all_complete(runs);
    }

    /// Shape (C): pool of 2, two writers with num_threads=2 via `join`.
    #[test]
    fn ordered_pipeline_from_pool_workers_join_on_2_does_not_deadlock() {
        let (a, b) = watchdog("join on the 2-pool", || {
            run_on_global_pool(tc(2), || rayon::join(|| nested(2), || nested(2))).unwrap()
        });
        all_complete(vec![a, b]);
    }

    /// Shapes that always completed and must stay so: a writer asking for 8
    /// threads from a 2-pool task, and one writer on the 2-pool from a plain
    /// thread.
    #[test]
    fn ordered_pipeline_other_nesting_shapes_complete() {
        use rayon::prelude::*;
        let runs = watchdog("8 from the 2-pool", || {
            run_on_global_pool(tc(2), || (0..4).into_par_iter().map(|_| nested(8)).collect::<Vec<_>>()).unwrap()
        });
        all_complete(runs);
        all_complete(vec![watchdog("plain thread", || nested(2))]);
    }

    /// Cross-pool cycle: the 2-pool's workers start writers on the 3-pool
    /// while the 3-pool's workers start writers on the 2-pool. If a worker
    /// blocked waiting on another pool, both pools could end up with every
    /// worker blocked on the other. Workers of any exr-core pool run nested
    /// pipelines inline, so this completes.
    #[test]
    fn ordered_pipeline_cross_pool_nesting_does_not_deadlock() {
        use rayon::prelude::*;
        let runs = watchdog("cross-pool nesting", || {
            let t1 = std::thread::spawn(|| {
                run_on_global_pool(tc(2), || (0..6).into_par_iter().map(|_| nested(3)).collect::<Vec<_>>()).unwrap()
            });
            let t2 = std::thread::spawn(|| {
                run_on_global_pool(tc(3), || (0..6).into_par_iter().map(|_| nested(2)).collect::<Vec<_>>()).unwrap()
            });
            let mut runs = t1.join().unwrap();
            runs.extend(t2.join().unwrap());
            runs
        });
        all_complete(runs);
    }

    /// The byte cap bounds the outstanding weight (and a single item heavier
    /// than the cap still makes progress).
    #[test]
    fn ordered_pipeline_bounds_outstanding_bytes() {
        take_peak_bytes();
        let mut n_emitted = 0;
        ordered_pipeline(
            tc(4),
            30,
            Ring { items: 100, bytes: 250 },
            |_| 100,
            |_| {
                std::thread::sleep(Duration::from_micros(200));
                Ok(())
            },
            |_, ()| {
                n_emitted += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(n_emitted, 30);
        assert_eq!(take_peak_bytes(), 200, "two 100-byte items fit in 250, a third does not");

        // One item over the cap is still admitted alone.
        let mut seen = 0;
        ordered_pipeline(tc(2), 3, Ring { items: 8, bytes: 10 }, |_| 1000, |_| Ok(()), |_, ()| {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 3);
        assert_eq!(take_peak_bytes(), 1000);
    }

    /// A panic inside `emit` unwinds out of the call after every submitted
    /// task has been joined.
    #[test]
    fn ordered_pipeline_emit_panic_unwinds_after_join() {
        let finished = AtomicUsize::new(0);
        let started = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(|| {
            pipe(
                tc(3),
                30,
                6,
                |_| {
                    started.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(1));
                    finished.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                |i, ()| {
                    if i == 2 {
                        panic!("emit boom");
                    }
                    Ok(())
                },
            )
        });
        assert!(r.is_err());
        assert_eq!(started.load(Ordering::SeqCst), finished.load(Ordering::SeqCst), "a task outlived the call");
    }
}
