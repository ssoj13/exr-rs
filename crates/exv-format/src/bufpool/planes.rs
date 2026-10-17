//! [`PlanePool`] / [`PlaneBuf`]: a recycler of plane-sized byte buffers (whole-frame decode targets for playback,
//! transcode and viewer caches).
//!
//! # Why
//! A fresh buffer costs its first touch: Windows serializes the page faults (measured 1-4.6 GB/s whatever the thread
//! count) while writing a warm, already-faulted buffer runs at 10-14 GB/s, and the eager zero-fill of a new plane was a
//! third of a 4k frame decode. A decoder that overwrites every byte needs neither the zeroing nor a new allocation, so
//! frame N+1 decodes into the buffers frame N-k gave back.
//!
//! # Model
//! - Buffers are matched by EXACT byte length (geometry x sample size); a [`PlaneBuf`] returns itself on drop, so any
//!   holder of a decoded frame (a cache entry, a GOP reference) recycles it without knowing about the pool.
//! - The [`Budget`] caps the IDLE bytes the pool keeps (buffers in use belong to their holder's own budget). Giving back
//!   above the cap evicts the oldest idle buffers first; nothing ever waits.
//! - Buffers are [`AlignedBuf`]s: the base is 64-byte aligned on every platform/allocator.
//! - Contents of a taken buffer are UNSPECIFIED (stale bytes of an earlier frame): only callers that overwrite every
//!   byte may take one. A miss allocates zero-filled ([`AlignedBuf::zeroed`], fallible).
//! - Cheap to clone (an `Arc`); clones share the shelf.

use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::aligned::AlignedBuf;
use super::budget::Budget;
use super::limits::AllocError;
use super::pool::{Weigh, bump};

/// Recycler of plane-sized byte buffers: the owner of whole-frame plane reuse (see the module docs).
#[derive(Clone)]
pub struct PlanePool(Arc<Shelf>);

/// The shared state behind a [`PlanePool`].
struct Shelf {
    budget: Budget,
    /// Idle buffers, oldest first, and their byte total.
    free: Mutex<(VecDeque<AlignedBuf>, u64)>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl std::fmt::Debug for PlanePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanePool").field("cap", &self.0.budget.cap()).field("stats", &self.stats()).finish()
    }
}

/// Counters of a [`PlanePool`] (tests and benches prove reuse with them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneStats {
    /// Takes served by an idle buffer (no allocation, no zeroing).
    pub hits: u64,
    /// Takes that allocated.
    pub misses: u64,
    /// Bytes idle on the shelf now (allocated capacity: whole 64-byte lines).
    pub idle: u64,
}

impl PlanePool {
    /// A pool keeping at most `budget.cap()` idle bytes (live: a [`Budget::set`] applies at the next give-back).
    pub fn new(budget: Budget) -> Self {
        Self(Arc::new(Shelf {
            budget,
            free: Mutex::new((VecDeque::new(), 0)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }))
    }

    /// A buffer of exactly `len` bytes with UNSPECIFIED contents: the most recently given back one of that length, else
    /// a new zero-filled allocation.
    ///
    /// # Errors
    /// [`AllocError`] when the allocator refuses a miss.
    pub fn take(&self, len: usize) -> Result<PlaneBuf, AllocError> {
        let reused = {
            let mut g = self.0.free.lock().unwrap_or_else(PoisonError::into_inner);
            let (free, idle) = &mut *g;
            let hit = free.iter().rposition(|b| b.len() == len).and_then(|i| free.remove(i));
            if let Some(b) = &hit {
                *idle = idle.saturating_sub(b.capacity() as u64);
            }
            hit
        };
        let buf = match reused {
            Some(b) => {
                bump(&self.0.hits);
                b
            }
            None => {
                bump(&self.0.misses);
                AlignedBuf::zeroed(len)?
            }
        };
        Ok(PlaneBuf { buf, home: Some(self.clone()) })
    }

    /// Hit/miss counters and the idle bytes.
    pub fn stats(&self) -> PlaneStats {
        let idle = self.0.free.lock().unwrap_or_else(PoisonError::into_inner).1;
        PlaneStats { hits: self.0.hits.load(Ordering::Relaxed), misses: self.0.misses.load(Ordering::Relaxed), idle }
    }

    /// Shelve `buf` (the give-back of a dropped [`PlaneBuf`]); evicted buffers are freed after the lock is released.
    fn give(&self, buf: AlignedBuf) {
        if buf.capacity() == 0 {
            return;
        }
        let cap = self.0.budget.cap();
        let mut gone = Vec::new();
        {
            let mut g = self.0.free.lock().unwrap_or_else(PoisonError::into_inner);
            let (free, idle) = &mut *g;
            *idle += buf.capacity() as u64;
            free.push_back(buf);
            while *idle > cap {
                let Some(old) = free.pop_front() else { break };
                *idle = idle.saturating_sub(old.capacity() as u64);
                gone.push(old);
            }
        }
        drop(gone);
    }
}

/// One plane's bytes, owned, 64-byte aligned ([`AlignedBuf`]); returns to its [`PlanePool`] on drop (a plain buffer
/// when built by [`PlaneBuf::new`]).
///
/// A panic while a `PlaneBuf` is borrowed mutably (a decoder that unwound halfway) still returns it on drop: that is
/// correct because pooled contents are UNSPECIFIED by contract (every taker overwrites all bytes) and the buffer itself
/// cannot be left in an invalid state (`[u8]` has no invariant); the pool's own mutex is poison-tolerant.
pub struct PlaneBuf {
    buf: AlignedBuf,
    home: Option<PlanePool>,
}

impl PlaneBuf {
    /// An unpooled buffer (dropping it frees it).
    pub fn new(buf: AlignedBuf) -> Self {
        Self { buf, home: None }
    }

    /// The bytes as an owned `Vec` (for `Vec`-based APIs). A COPY: a `Vec<u8>` cannot own 64-aligned storage without
    /// `unsafe`; the aligned buffer itself goes back to its pool on drop, so the copy replaces the fresh allocation +
    /// zero-fill a detached buffer would cost its pool at the next take.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf.to_vec()
    }
}

impl Deref for PlaneBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl DerefMut for PlaneBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

impl std::fmt::Debug for PlaneBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneBuf").field("len", &self.buf.len()).field("pooled", &self.home.is_some()).finish()
    }
}

impl Drop for PlaneBuf {
    fn drop(&mut self) {
        if let Some(pool) = self.home.take() {
            pool.give(std::mem::take(&mut self.buf));
        }
    }
}

/// Charged at its allocated capacity (what the allocator holds), not its length.
impl Weigh for PlaneBuf {
    fn bytes(&self) -> usize {
        self.buf.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dropped `PlaneBuf` comes back for the same length (no allocation), not for another length; the idle cap (in
    /// allocated bytes: whole 64-byte lines) evicts the oldest; `into_vec` copies and the buffer still returns.
    #[test]
    fn plane_pool_recycles_by_length_under_cap() {
        let pool = PlanePool::new(Budget::new(320));
        let mut a = pool.take(128).expect("take");
        a[0] = 9;
        let ptr = a.as_ptr();
        drop(a);
        assert_eq!(pool.stats(), PlaneStats { hits: 0, misses: 1, idle: 128 });
        let b = pool.take(128).expect("take");
        assert_eq!(b.as_ptr(), ptr, "same buffer reused");
        assert_eq!((b[0], pool.stats().hits), (9, 1), "contents are stale, not re-zeroed");
        let c = pool.take(50).expect("take");
        assert_eq!(pool.stats().misses, 2, "other length = miss");
        let (d, e) = (pool.take(128).expect("take"), pool.take(128).expect("take"));
        drop((b, c, d, e)); // 128 + 64 + 128 + 128 > 320: the oldest (b) is evicted
        assert_eq!(pool.stats().idle, 320);
        let held = pool.take(7).expect("take");
        assert!((held.as_ptr() as usize).is_multiple_of(crate::bufpool::LINE), "pooled planes are cache-line aligned");
        assert_eq!(held.into_vec(), vec![0; 7], "into_vec copies the bytes out");
        assert_eq!(pool.stats().idle, 320, "the 64-B buffer went home (the oldest, c, evicted to stay under the cap)");
    }

    /// Give-back on every exit path: a decoder that returns `Err` or PANICS while writing a taken buffer still returns it
    /// (contents are unspecified by contract, the buffer stays valid), and a poisoned shelf mutex keeps working.
    #[test]
    fn plane_buf_returns_home_on_error_and_panic() {
        let pool = PlanePool::new(Budget::new(1 << 20));
        let fail = |p: &PlanePool| -> Result<(), &'static str> {
            let mut b = p.take(4096).map_err(|_| "alloc")?;
            b[0] = 1;
            Err("decode failed")
        };
        assert!(fail(&pool).is_err());
        assert_eq!((pool.stats().misses, pool.stats().idle), (1, 4096), "Err path: buffer shelved");
        let p2 = pool.clone();
        let r = std::panic::catch_unwind(move || {
            let mut b = p2.take(4096).expect("take");
            b[1] = 2;
            panic!("decoder panicked mid-plane");
        });
        assert!(r.is_err());
        assert_eq!(
            pool.stats(),
            PlaneStats { hits: 1, misses: 1, idle: 4096 },
            "panic path: buffer shelved, no new allocation"
        );
        // Poison the shelf mutex itself: give/take must still work (PoisonError::into_inner).
        let p3 = pool.clone();
        let poisoned = std::panic::catch_unwind(move || {
            let _g = p3.0.free.lock();
            panic!("poison the shelf");
        });
        assert!(poisoned.is_err());
        assert!(pool.0.free.is_poisoned());
        drop(pool.take(4096).expect("take after poison"));
        assert_eq!(pool.stats(), PlaneStats { hits: 2, misses: 1, idle: 4096 });
    }

    /// An unpooled buffer is freed on drop; a refused allocation is an `AllocError`; the weight is the capacity.
    #[test]
    fn unpooled_buffer_weight_and_alloc_error() {
        let b = PlaneBuf::new(AlignedBuf::zeroed(65).expect("alloc"));
        assert_eq!((b.len(), b.bytes()), (65, 128));
        let pool = PlanePool::new(Budget::new(0));
        assert_eq!(pool.take(usize::MAX - 1).map(drop), Err(AllocError { bytes: usize::MAX - 1 }));
        drop(pool.take(64).expect("take"));
        assert_eq!(pool.stats().idle, 0, "a zero budget keeps nothing idle");
    }
}
