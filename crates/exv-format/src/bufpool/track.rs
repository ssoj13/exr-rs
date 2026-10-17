//! [`TrackingAlloc`]: a counting global-allocator wrapper for a deterministic peak-heap signal.
//!
//! # Why
//! Out-of-core work claims "nothing OOMs"; that claim is only credible against a real number. OS RSS sampling races the
//! allocator and Rust returns memory to the OS lazily, so a poll can miss the true high-water mark entirely. A counting
//! [`GlobalAlloc`] wrapper reports the EXACT high-water mark of bytes handed out, independent of any sampling cadence:
//! the measurement behind OOM gates and pool-vs-allocator comparisons.
//!
//! # Usage (binary or integration-test crate)
//! ```standalone_crate
//! use exv_format::bufpool::{TrackingAlloc, track};
//! use std::alloc::System;
//!
//! #[global_allocator]
//! static GLOBAL: TrackingAlloc<System> = TrackingAlloc::new(System);
//!
//! fn main() {
//!     assert!(track::is_installed());
//!     let (_v, peak) = track::measure(|| vec![0u8; 1 << 20]);
//!     assert!(peak >= 1 << 20);
//! }
//! ```
//! Then read the counters with [`current`], [`peak`], [`total_allocated`], reset the window with [`reset_peak`], or
//! wrap a region with [`measure`]. Assert [`is_installed`] first: the counters read 0 without a global allocator.
//!
//! The wrapper is generic over the inner allocator, so swapping `System` for mimalloc is a one-line type change
//! (`TrackingAlloc<MiMalloc>`).
//!
//! # Accuracy & caveats
//! - Counts the [`Layout::size`] requested, not the allocator's rounded-up block size: a stable, allocator-independent
//!   signal (exactly what a before/after comparison needs).
//! - The counters are process-global (two `TrackingAlloc` instances share them). [`measure`]/[`reset_peak`] give a
//!   clean per-region peak only when the region dominates allocation (single-threaded, or the rest of the process is
//!   quiescent). Concurrent allocations from unrelated threads inflate the peak.
//! - Ordering is [`Relaxed`](Ordering::Relaxed): the counters guard no other memory. Cost: four relaxed atomics and one
//!   thread-local bump per
//!   allocation.
//!
//! The `unsafe impl GlobalAlloc` below is the only `unsafe` of this crate (the trait requires it). The library never
//! installs a global allocator itself; its tests live in `tests/track.rs`, a binary of their own, so the process-global
//! counters are not perturbed by the pool tests running in parallel.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live bytes currently handed out (incremented on alloc, decremented on dealloc).
static CURRENT: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of `CURRENT` since process start or the last [`reset_peak`].
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Cumulative bytes ever allocated (allocation churn; never decremented).
static TOTAL: AtomicUsize = AtomicUsize::new(0);
/// Number of allocator calls that handed out memory (alloc, alloc_zeroed, realloc): the other half of churn - many small
/// allocations cost far more than their byte total suggests.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// A [`GlobalAlloc`] that records live/peak/total requested bytes, then forwards to the inner allocator `A`. Install it
/// as the `#[global_allocator]` of a binary or integration-test crate.
#[derive(Debug, Default)]
pub struct TrackingAlloc<A> {
    inner: A,
}

impl<A> TrackingAlloc<A> {
    /// Wrap `inner`. `const` so it can initialise a `static` for `#[global_allocator]`.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

thread_local! {
    // This thread's allocator calls and bytes. `const` init and a `Cell` with
    // no destructor: reading or bumping them never allocates, so it is safe
    // inside the allocator, and `try_with` is a no-op after TLS teardown.
    static THREAD: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Count one allocator call of `size` bytes for the current thread.
#[inline]
fn account_thread(size: usize) {
    // After TLS teardown there is no thread left to attribute to.
    THREAD
        .try_with(|t| {
            let (n, b) = t.get();
            t.set((n + 1, b + size));
        })
        .unwrap_or_default();
}

/// Account `size` live bytes and bump the peak (one `fetch_max`, no CAS loop needed).
#[inline]
fn account_grow(size: usize) {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    account_thread(size);
    TOTAL.fetch_add(size, Ordering::Relaxed);
    let cur = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(cur, Ordering::Relaxed);
}

/// Account `size` freed live bytes (the peak is monotonic, so it is not touched here).
#[inline]
fn account_shrink(size: usize) {
    CURRENT.fetch_sub(size, Ordering::Relaxed);
}

// SAFETY: every method forwards 1:1 to the inner allocator with the same pointer/layout, so the allocator contract is
// upheld by `A`. The accounting is pure atomic bookkeeping that touches no allocator state and runs only on the
// success path (non-null return), so it cannot affect safety.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `layout`; forwarded unchanged.
        let p = unsafe { self.inner.alloc(layout) };
        if !p.is_null() {
            account_grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by this allocator (hence by `inner`) for `layout`; forwarded unchanged.
        unsafe { self.inner.dealloc(ptr, layout) };
        account_shrink(layout.size());
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let p = unsafe { self.inner.alloc_zeroed(layout) };
        if !p.is_null() {
            account_grow(layout.size());
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`/`layout` come from this allocator (hence `inner`) and `new_size` satisfies the caller's
        // `realloc` contract; forwarded unchanged.
        let p = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            // Net live change is `new_size - old_size`, whether the block grew in place or moved (the old block is
            // consumed either way).
            let old = layout.size();
            if new_size >= old {
                account_grow(new_size - old);
            } else {
                ALLOCS.fetch_add(1, Ordering::Relaxed);
                account_thread(0);
                account_shrink(old - new_size);
            }
        }
        p
    }
}

/// `true` when a [`TrackingAlloc`] really is the `#[global_allocator]` of this process. The counters are plain statics
/// that read `0` whether or not a tracker is installed, so a binary that forgot `#[global_allocator]` would see
/// `peak() == 0` and pass any OOM gate vacuously; gates must assert this first.
///
/// Makes one probe allocation first: a process is NOT guaranteed to have allocated before `main` (a minimal Windows
/// binary has not), so "nothing counted yet" alone cannot tell "not installed" from "not used yet".
pub fn is_installed() -> bool {
    drop(std::hint::black_box(Box::new(0u8)));
    total_allocated() > 0
}

/// Live bytes currently allocated (requested sizes not yet freed).
pub fn current() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

/// High-water mark of [`current`] since process start or the last [`reset_peak`].
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Cumulative bytes ever allocated (allocation churn: a proxy for malloc pressure).
pub fn total_allocated() -> usize {
    TOTAL.load(Ordering::Relaxed)
}

/// Reset the peak to the current live bytes: call immediately before a measured region so [`peak`] afterwards reflects
/// only that region's high-water mark.
pub fn reset_peak() {
    PEAK.store(current(), Ordering::Relaxed);
}

/// Run `f` and return `(result, region_peak)`, where `region_peak` is the ADDITIONAL live-byte high-water mark the
/// closure caused above the live bytes already held on entry. See the module-level caveats about concurrency.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = current();
    reset_peak();
    let r = f();
    let region_peak = peak().saturating_sub(before);
    (r, region_peak)
}

/// Number of allocator calls that handed out memory since process start (`alloc`, `alloc_zeroed`, `realloc`).
pub fn allocations() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// Counter values at one instant; [`Snapshot::since`] gives the churn of a region (e.g. one pipeline phase or one
/// request) without resetting anything, so nested and overlapping regions can be measured independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Snapshot {
    /// [`allocations`] at the instant.
    pub allocs: usize,
    /// [`total_allocated`] at the instant.
    pub bytes: usize,
    /// [`current`] at the instant.
    pub live: usize,
}

impl Snapshot {
    /// Read the counters now.
    pub fn now() -> Self {
        Self {
            allocs: allocations(),
            bytes: total_allocated(),
            live: current(),
        }
    }

    /// Allocations and bytes allocated since `self`, and the live-byte change (negative when memory was released).
    pub fn since(&self) -> Churn {
        let now = Self::now();
        Churn {
            allocs: now.allocs.saturating_sub(self.allocs),
            bytes: now.bytes.saturating_sub(self.bytes),
            live_delta: now.live as isize - self.live as isize,
        }
    }
}

/// Allocator traffic of one region, from [`Snapshot::since`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Churn {
    /// Allocator calls in the region.
    pub allocs: usize,
    /// Bytes allocated in the region (freed or not).
    pub bytes: usize,
    /// Live-byte change over the region; negative when it released more than it kept.
    pub live_delta: isize,
}

/// Allocator calls and bytes allocated by the CURRENT thread since it started.
/// Unlike the process counters, unaffected by what other threads do - the
/// measure for work inside a thread pool.
pub fn thread_counts() -> (usize, usize) {
    THREAD.try_with(Cell::get).unwrap_or_default()
}
