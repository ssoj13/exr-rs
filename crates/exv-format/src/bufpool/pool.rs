//! [`TilePool`]: a byte-budget LRU cache with pinning, single-flight loads and reserve-before-decode.
//!
//! # Why hand-rolled (no moka / quick_cache)
//! The goal is a HARD memory bound ("decode a 65k^2 image without OOM"). Two properties no off-the-shelf cache gives
//! together: (1) a load RESERVES its bytes before the decode runs, so N workers can never jointly materialize more than
//! the budget (the real OOM source), and (2) `used` is the truth a counting allocator sees, i.e. it includes bytes of
//! evicted-but-still-leased entries and in-flight reservations. Access is usually spatially local with an explicit
//! viewport/playhead order, so plain LRU + pinning (what OIIO and GDAL ship) is enough.
//!
//! # Model
//! - [`Lease`] = the pin (an `Arc`); an entry with an outstanding lease is never evicted. Pinned bytes above the cap
//!   are a hard [`PoolError::OverBudget`] returned to the caller, never a silent over-commit and never an unbounded
//!   wait (the caller may itself hold the leases: waiting would deadlock).
//! - `used` = resident + still-leased-after-removal + reserved-in-flight. A slot subtracts its bytes when its last
//!   owner drops it, so the number cannot drift.
//! - One load per key at a time: the others block on a condvar and receive the SAME lease or the SAME error. Errors
//!   are not cached.
//! - The lock covers map/list operations only: never a load, never the drop of a payload (a GPU texture drop can be
//!   slow): victims are collected under the lock and dropped after it.
//! - The error type is the caller's (`E`, default [`PoolError`]): loaders return their own errors, the pool injects
//!   its failures through `From<PoolError>` and reads them back through [`PoolErr`].
//!
//! # Work-stealing executors (feature `rayon`)
//! A rayon worker must NOT block on another thread's flight: the leader may be an ancestor frame of the SAME stack (a
//! worker waiting inside a nested `join` steals other jobs), which would then never resume. With feature `rayon` a
//! lookup on a rayon worker that finds a flight decodes PRIVATELY instead: the reservation still bounds memory, the
//! result is just not cached.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use super::budget::Budget;

/// A payload a pool can charge: its real heap/VRAM footprint in bytes.
///
/// Implemented here for `Vec<u8>` (its length) and [`crate::PlaneBuf`] (its allocated capacity); hosts implement it
/// for their tile / texture types.
pub trait Weigh {
    /// Bytes this payload holds.
    fn bytes(&self) -> usize;
}

impl Weigh for Vec<u8> {
    fn bytes(&self) -> usize {
        self.len()
    }
}

/// The pool's own failures. `Clone` so a single-flight pool can hand the SAME failure to every waiter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolError {
    /// A load needs `need` bytes but `used` of `cap` are already committed (and pinned): a hard error, never a silent
    /// over-commit.
    OverBudget {
        /// Bytes the load asked for.
        need: usize,
        /// Bytes charged when it was refused.
        used: usize,
        /// The budget cap.
        cap: usize,
    },
    /// The caller cancelled the load ([`Cancel`]).
    Cancelled,
    /// The loader panicked; its reservation was returned and the waiters were released with this error.
    Panicked,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OverBudget { need, used, cap } => {
                write!(
                    f,
                    "load needs {need} B but {used} of {cap} B are committed and pinned (raise the memory budget)"
                )
            }
            Self::Cancelled => f.write_str("load cancelled"),
            Self::Panicked => f.write_str("loader panicked"),
        }
    }
}

impl std::error::Error for PoolError {}

/// An error type a [`TilePool`] can carry: it must hold the pool's own failures (`From<PoolError>`) and expose them
/// again ([`Self::as_pool`]), so single-flight waiters can tell "the leader was cancelled / could not evict" (retry)
/// from a loader failure (share it). `Clone` because every waiter receives the leader's result.
///
/// A host error enum typically has its own `OverBudget` / `Cancelled` variants and maps both ways.
pub trait PoolErr: Clone + From<PoolError> {
    /// The [`PoolError`] this error represents, `None` for a loader's own failure.
    fn as_pool(&self) -> Option<PoolError>;
}

impl PoolErr for PoolError {
    fn as_pool(&self) -> Option<PoolError> {
        Some(self.clone())
    }
}

/// Cooperative cancellation token (cheap clone, shared flag). Loaders poll it between units of work; one unit is not
/// interruptible.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A fresh, not-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel every holder of this token (and its clones).
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// `true` once [`Self::cancel`] was called.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// `Err(PoolError::Cancelled)` once cancelled: `cancel.check()?` in loaders (any `E: From<PoolError>`).
    ///
    /// # Errors
    /// [`PoolError::Cancelled`] after [`Self::cancel`].
    pub fn check(&self) -> Result<(), PoolError> {
        if self.is_cancelled() { Err(PoolError::Cancelled) } else { Ok(()) }
    }
}

/// Live byte counter with a high-water mark.
#[derive(Default)]
struct Used {
    now: AtomicUsize,
    peak: AtomicUsize,
}

impl Used {
    fn add(&self, n: usize) {
        let now = self.now.fetch_add(n, Ordering::AcqRel) + n;
        self.peak.fetch_max(now, Ordering::Relaxed);
    }
    fn sub(&self, n: usize) {
        self.now.fetch_sub(n, Ordering::AcqRel);
    }
    fn get(&self) -> usize {
        self.now.load(Ordering::Acquire)
    }
}

/// A payload plus the bytes it currently charges. Dropping the last owner returns the charge.
struct Slot<V> {
    v: V,
    acct: AtomicUsize,
    used: Arc<Used>,
}

impl<V> Drop for Slot<V> {
    fn drop(&mut self) {
        self.used.sub(self.acct.swap(0, Ordering::AcqRel));
    }
}

/// A pinned, shared view of a cached payload. Holding it keeps the entry resident (and counted); dropping every lease
/// of an evicted entry frees it.
pub struct Lease<V>(Arc<Slot<V>>);

impl<V> Clone for Lease<V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<V> Deref for Lease<V> {
    type Target = V;
    fn deref(&self) -> &V {
        &self.0.v
    }
}

impl<V: fmt::Debug> fmt::Debug for Lease<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Lease").field(&self.0.v).finish()
    }
}

/// One in-flight load; waiters block until the leader publishes.
struct Flight<V, E> {
    res: Mutex<Option<Result<Lease<V>, E>>>,
    cv: Condvar,
    /// The leader may evict (foreground). A non-evicting leader's `OverBudget` says nothing about a waiter that may
    /// evict: such a waiter retries as leader instead of inheriting it.
    evict: bool,
}

impl<V, E: Clone> Flight<V, E> {
    fn new(evict: bool) -> Self {
        Self { res: Mutex::new(None), cv: Condvar::new(), evict }
    }

    fn done(&self, r: Result<Lease<V>, E>) {
        *self.res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
        self.cv.notify_all();
    }

    fn wait(&self) -> Result<Lease<V>, E> {
        let mut g = self.res.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(r) = g.as_ref() {
                return r.clone();
            }
            g = self.cv.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

const NIL: usize = usize::MAX;

struct Node<K, V> {
    key: K,
    /// `None` = free slab cell.
    slot: Option<Arc<Slot<V>>>,
    prev: usize,
    next: usize,
}

enum Ent<V, E> {
    /// Slab index of a resident entry.
    Ready(usize),
    Loading(Arc<Flight<V, E>>),
}

/// Map + intrusive doubly-linked LRU list over a slab: O(1) lookup, bump and evict, no `unsafe`.
struct Inner<K, V, E> {
    map: HashMap<K, Ent<V, E>>,
    nodes: Vec<Node<K, V>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
}

impl<K: Hash + Eq + Clone, V, E> Inner<K, V, E> {
    fn unlink(&mut self, i: usize) {
        let (p, n) = (self.nodes[i].prev, self.nodes[i].next);
        if p == NIL {
            self.head = n
        } else {
            self.nodes[p].next = n
        }
        if n == NIL { self.tail = p } else { self.nodes[n].prev = p }
    }

    fn push_front(&mut self, i: usize) {
        self.nodes[i].prev = NIL;
        self.nodes[i].next = self.head;
        if self.head != NIL {
            let h = self.head;
            self.nodes[h].prev = i;
        }
        self.head = i;
        if self.tail == NIL {
            self.tail = i;
        }
    }

    fn touch(&mut self, i: usize) {
        if self.head != i {
            self.unlink(i);
            self.push_front(i);
        }
    }

    /// Bump resident cell `i` to most recently used and lease it. A `Ready` index always holds a slot (cells are freed
    /// together with their map entry), so `None` never happens.
    fn hit(&mut self, i: usize) -> Option<Lease<V>> {
        self.touch(i);
        self.nodes[i].slot.clone().map(Lease)
    }

    /// Install `slot` as the most recently used entry of `key` (replaces whatever the map held).
    fn install(&mut self, key: K, slot: Arc<Slot<V>>) {
        let node = Node { key: key.clone(), slot: Some(slot), prev: NIL, next: NIL };
        let i = match self.free.pop() {
            Some(i) => {
                self.nodes[i] = node;
                i
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        };
        self.push_front(i);
        self.map.insert(key, Ent::Ready(i));
    }

    /// Unlink + free cell `i`, returning its slot. The caller removes/overwrites the map entry.
    fn take(&mut self, i: usize) -> Option<Arc<Slot<V>>> {
        self.unlink(i);
        self.free.push(i);
        self.nodes[i].slot.take()
    }
}

/// A prefetch loader: produces the payload of a key (polling the batch's [`Cancel`]).
pub type Loader<K, V, E = PoolError> = dyn Fn(&K, &Cancel) -> Result<V, E> + Send + Sync;

/// A unit of prefetch work handed to the caller's executor.
pub type Job = Box<dyn FnOnce() + Send>;

/// Counters for tests, a status bar and OOM gates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Lookups served from the cache.
    pub hits: u64,
    /// Lookups that found nothing resident.
    pub misses: u64,
    /// Loader invocations (== misses that won the flight).
    pub loads: u64,
    /// Requests that joined another thread's load instead of decoding.
    pub waits: u64,
    /// Entries evicted to make room.
    pub evictions: u64,
    /// Reservations refused ([`PoolError::OverBudget`]).
    pub over_budget: u64,
    /// Prefetch jobs that failed (expected: `OverBudget` once the cache is full).
    pub prefetch_err: u64,
    /// High-water mark of `used`.
    pub peak_used: usize,
    /// Bytes charged now (resident + leased-after-removal + in-flight).
    pub used: usize,
    /// Resident entries.
    pub resident: usize,
    /// Resident entries currently pinned by a lease.
    pub pinned: usize,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    loads: AtomicU64,
    waits: AtomicU64,
    evictions: AtomicU64,
    over_budget: AtomicU64,
    prefetch_err: AtomicU64,
}

/// Bump a relaxed statistics counter (shared with [`crate::planes`]).
pub(crate) fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// `true` when this thread must not block on another thread's flight (see the module docs, feature `rayon`).
fn no_wait() -> bool {
    #[cfg(feature = "rayon")]
    {
        rayon::current_thread_index().is_some()
    }
    #[cfg(not(feature = "rayon"))]
    {
        false
    }
}

/// Returns a private load's reservation if its loader panics.
struct Reservation<'a> {
    used: &'a Used,
    cost: usize,
    armed: bool,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.used.sub(self.cost);
        }
    }
}

/// Releases the reservation and unblocks waiters if a loader panics (otherwise they would hang forever).
struct FlightGuard<'a, K: Hash + Eq + Clone, V: Weigh, E: PoolErr> {
    pool: &'a TilePool<K, V, E>,
    key: &'a K,
    flight: Arc<Flight<V, E>>,
    cost: usize,
    armed: bool,
}

impl<K: Hash + Eq + Clone, V: Weigh, E: PoolErr> Drop for FlightGuard<'_, K, V, E> {
    fn drop(&mut self) {
        if self.armed {
            self.pool.used.sub(self.cost);
            self.pool.forget(self.key, &self.flight);
            self.flight.done(Err(PoolError::Panicked.into()));
        }
    }
}

/// Byte-budget LRU of decoded payloads, shared by threads. See the module docs for the contract.
///
/// `K` = cache key (e.g. `(frame, level, chan, tx, ty)`), `V` = payload ([`Weigh`]), `E` = the error loaders return
/// ([`PoolErr`], default [`PoolError`]).
pub struct TilePool<K: Hash + Eq + Clone, V, E = PoolError> {
    inner: Mutex<Inner<K, V, E>>,
    budget: Budget,
    used: Arc<Used>,
    c: Counters,
}

impl<K: Hash + Eq + Clone, V: Weigh, E: PoolErr> TilePool<K, V, E> {
    /// An empty pool charging against `budget` (a clone of a shared [`Budget`] resizes it live).
    pub fn new(budget: Budget) -> Self {
        let inner = Inner { map: HashMap::new(), nodes: Vec::new(), free: Vec::new(), head: NIL, tail: NIL };
        Self { inner: Mutex::new(inner), budget, used: Arc::new(Used::default()), c: Counters::default() }
    }

    fn lock(&self) -> MutexGuard<'_, Inner<K, V, E>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn cap(&self) -> usize {
        usize::try_from(self.budget.cap()).unwrap_or(usize::MAX)
    }

    /// Bytes charged right now (resident + leased-after-eviction + in-flight reservations).
    pub fn used(&self) -> usize {
        self.used.get()
    }

    /// Resident entries.
    pub fn len(&self) -> usize {
        let g = self.lock();
        g.nodes.len().saturating_sub(g.free.len())
    }

    /// `true` when nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if `k` is resident or being loaded. Does not touch the LRU.
    pub fn contains(&self, k: &K) -> bool {
        self.lock().map.contains_key(k)
    }

    /// A snapshot of the counters.
    pub fn stats(&self) -> PoolStats {
        let g = self.lock();
        let live = g.nodes.iter().filter_map(|n| n.slot.as_ref());
        let (resident, pinned) = live.fold((0, 0), |(r, p), s| (r + 1, p + usize::from(Arc::strong_count(s) > 1)));
        let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
        PoolStats {
            hits: ld(&self.c.hits),
            misses: ld(&self.c.misses),
            loads: ld(&self.c.loads),
            waits: ld(&self.c.waits),
            evictions: ld(&self.c.evictions),
            over_budget: ld(&self.c.over_budget),
            prefetch_err: ld(&self.c.prefetch_err),
            peak_used: self.used.peak.load(Ordering::Relaxed),
            used: self.used.get(),
            resident,
            pinned,
        }
    }

    /// Return a removed slot's charge NOW when nobody else holds it (so a reservation made under the same lock sees the
    /// room); a still-leased slot keeps counting until its last lease drops.
    fn retire(&self, slot: Arc<Slot<V>>, victims: &mut Vec<Arc<Slot<V>>>) {
        if Arc::strong_count(&slot) == 1 {
            self.used.sub(slot.acct.swap(0, Ordering::AcqRel));
        }
        victims.push(slot);
    }

    /// Reserve `need` bytes: evict unpinned LRU entries (tail first) until it fits when `evict`, else fail. Called with
    /// the lock held; leases are only created under the lock, so `strong_count == 1` is stable.
    fn reserve(
        &self,
        g: &mut Inner<K, V, E>,
        need: usize,
        evict: bool,
        victims: &mut Vec<Arc<Slot<V>>>,
    ) -> Result<(), E> {
        let cap = self.cap();
        if need <= cap {
            let mut at = g.tail;
            while self.used.get().saturating_add(need) > cap && evict && at != NIL {
                let prev = g.nodes[at].prev;
                if g.nodes[at].slot.as_ref().is_some_and(|s| Arc::strong_count(s) == 1) {
                    let key = g.nodes[at].key.clone();
                    g.map.remove(&key);
                    if let Some(s) = g.take(at) {
                        self.retire(s, victims);
                        bump(&self.c.evictions);
                    }
                }
                at = prev;
            }
            if self.used.get().saturating_add(need) <= cap {
                self.used.add(need);
                return Ok(());
            }
        }
        bump(&self.c.over_budget);
        Err(PoolError::OverBudget { need, used: self.used.get(), cap }.into())
    }

    /// Drop the `Loading` entry of `k` if it is still `flight` (a `retain`/`insert` may have replaced it).
    fn forget(&self, k: &K, flight: &Arc<Flight<V, E>>) {
        let mut g = self.lock();
        if matches!(g.map.get(k), Some(Ent::Loading(f)) if Arc::ptr_eq(f, flight)) {
            g.map.remove(k);
        }
    }

    /// Cache-only lookup (bumps the LRU on a hit).
    pub fn get(&self, k: &K) -> Option<Lease<V>> {
        let mut g = self.lock();
        if let Some(&Ent::Ready(i)) = g.map.get(k) {
            bump(&self.c.hits);
            return g.hit(i);
        }
        bump(&self.c.misses);
        None
    }

    /// The cached payload of `k`, or load it. `cost` is the exact bytes the load will materialize; it is RESERVED
    /// before `load` runs so concurrent loads can never jointly exceed the budget. Other threads asking the same key
    /// wait and share the result.
    ///
    /// # Errors
    /// [`PoolError::OverBudget`] when `cost` (or the loaded size) cannot fit beside the pinned entries,
    /// [`PoolError::Cancelled`], [`PoolError::Panicked`], or whatever `load` returned (not cached).
    pub fn get_or_load(
        &self,
        k: K,
        cost: usize,
        cancel: &Cancel,
        load: impl FnOnce(&Cancel) -> Result<V, E>,
    ) -> Result<Lease<V>, E> {
        self.load_inner(k, cost, cancel, load, true)
    }

    fn load_inner(
        &self,
        k: K,
        cost: usize,
        cancel: &Cancel,
        load: impl FnOnce(&Cancel) -> Result<V, E>,
        evict: bool,
    ) -> Result<Lease<V>, E> {
        let flight = loop {
            cancel.check()?;
            let mut g = self.lock();
            let wait = match g.map.get(&k) {
                Some(&Ent::Ready(i)) => {
                    if let Some(l) = g.hit(i) {
                        bump(&self.c.hits);
                        return Ok(l);
                    }
                    None
                }
                Some(Ent::Loading(f)) => Some(f.clone()),
                None => None,
            };
            if let Some(f) = wait {
                if no_wait() {
                    bump(&self.c.misses);
                    let mut victims = Vec::new();
                    let r = self.reserve(&mut g, cost, evict, &mut victims);
                    drop(g);
                    drop(victims);
                    r?;
                    let mut guard = Reservation { used: &self.used, cost, armed: true };
                    let loaded = load(cancel);
                    guard.armed = false;
                    return self.settle_private(cost, loaded, evict);
                }
                drop(g);
                bump(&self.c.waits);
                match f.wait() {
                    // The leader was cancelled but we were not: take over.
                    Err(e) if e.as_pool() == Some(PoolError::Cancelled) && !cancel.is_cancelled() => continue,
                    // A prefetch leader could not make room without evicting; we may: take over. Its flight already
                    // left the map, so the next pass leads (or joins a newer flight).
                    Err(e) if evict && !f.evict && matches!(e.as_pool(), Some(PoolError::OverBudget { .. })) => {
                        continue;
                    }
                    r => return r,
                }
            }
            bump(&self.c.misses);
            let mut victims = Vec::new();
            let r = self.reserve(&mut g, cost, evict, &mut victims);
            let flight = Arc::new(Flight::new(evict));
            if r.is_ok() {
                g.map.insert(k.clone(), Ent::Loading(flight.clone()));
            }
            drop(g);
            drop(victims);
            r?;
            break flight;
        };

        bump(&self.c.loads);
        let mut guard = FlightGuard { pool: self, key: &k, flight, cost, armed: true };
        let loaded = load(cancel);
        guard.armed = false;
        self.finish(&k, &guard.flight, cost, loaded, evict)
    }

    /// Settle a reservation against the real size of `v`: release the surplus or reserve the shortfall (evicting only
    /// when `evict`). On failure the whole reservation is returned NOW and the payload goes to `victims`, so it is
    /// dropped after the lock like any evicted entry (module rule: never drop a payload under the lock). Called with
    /// the lock held.
    fn settle(
        &self,
        g: &mut Inner<K, V, E>,
        v: V,
        cost: usize,
        evict: bool,
        victims: &mut Vec<Arc<Slot<V>>>,
    ) -> Result<Arc<Slot<V>>, E> {
        let actual = v.bytes();
        // The slot carries the reservation until it is settled: `retire` returns exactly that on failure.
        let slot = Arc::new(Slot { v, acct: AtomicUsize::new(cost), used: self.used.clone() });
        if actual > cost {
            if let Err(e) = self.reserve(g, actual - cost, evict, victims) {
                self.retire(slot, victims);
                return Err(e);
            }
        } else {
            self.used.sub(cost - actual);
        }
        slot.acct.store(actual, Ordering::Release);
        Ok(slot)
    }

    /// Finish an uncached (private) load: a lease charged its real size, or the error with the reservation returned.
    fn settle_private(&self, cost: usize, loaded: Result<V, E>, evict: bool) -> Result<Lease<V>, E> {
        let mut victims = Vec::new();
        let out = match loaded {
            Err(e) => {
                self.used.sub(cost);
                Err(e)
            }
            Ok(v) => {
                let mut g = self.lock();
                self.settle(&mut g, v, cost, evict, &mut victims).map(Lease)
            }
        };
        drop(victims);
        out
    }

    /// Turn a finished load into a cached lease (or an error), settle the reservation, wake the waiters.
    fn finish(
        &self,
        k: &K,
        flight: &Arc<Flight<V, E>>,
        cost: usize,
        loaded: Result<V, E>,
        evict: bool,
    ) -> Result<Lease<V>, E> {
        let mut victims = Vec::new();
        let out = match loaded {
            Err(e) => {
                self.used.sub(cost);
                self.forget(k, flight);
                Err(e)
            }
            Ok(v) => {
                let mut g = self.lock();
                // Cache only if our flight still owns the key (an invalidation may have dropped it).
                let mine = matches!(g.map.get(k), Some(Ent::Loading(f)) if Arc::ptr_eq(f, flight));
                match self.settle(&mut g, v, cost, evict, &mut victims) {
                    Err(e) => {
                        if mine {
                            g.map.remove(k);
                        }
                        Err(e)
                    }
                    Ok(slot) => {
                        if mine {
                            g.install(k.clone(), slot.clone());
                        }
                        Ok(Lease(slot))
                    }
                }
            }
        };
        flight.done(out.clone());
        drop(victims);
        out
    }

    /// Insert an already-computed payload (e.g. GPU planes after a decode), replacing any entry of `k` (the old entry
    /// is dropped even if the new one then cannot fit: the key is invalidated either way).
    ///
    /// # Errors
    /// [`PoolError::OverBudget`] when it cannot fit beside the pinned entries.
    pub fn insert(&self, k: K, v: V) -> Result<Lease<V>, E> {
        let bytes = v.bytes();
        let mut victims = Vec::new();
        let mut g = self.lock();
        if let Some(Ent::Ready(i)) = g.map.remove(&k)
            && let Some(s) = g.take(i)
        {
            self.retire(s, &mut victims);
        }
        let r = self.reserve(&mut g, bytes, true, &mut victims);
        let out = r.map(|()| {
            let slot = Arc::new(Slot { v, acct: AtomicUsize::new(bytes), used: self.used.clone() });
            g.install(k, slot.clone());
            Lease(slot)
        });
        drop(g);
        drop(victims);
        out
    }

    /// Invalidate every entry for which `keep` is false (new content / epoch). Pinned entries leave the map but keep
    /// counting until their leases drop; in-flight loads of dropped keys finish but are not cached.
    pub fn retain(&self, mut keep: impl FnMut(&K) -> bool) {
        let mut victims = Vec::new();
        let mut g = self.lock();
        let drop_keys: Vec<K> = g.map.keys().filter(|k| !keep(k)).cloned().collect();
        for k in drop_keys {
            if let Some(Ent::Ready(i)) = g.map.remove(&k)
                && let Some(s) = g.take(i)
            {
                self.retire(s, &mut victims);
            }
        }
        drop(g);
        drop(victims);
    }

    /// Drop everything ([`Self::retain`] nothing).
    pub fn clear(&self) {
        self.retain(|_| false);
    }
}

impl<K, V, E> TilePool<K, V, E>
where
    K: Hash + Eq + Clone + Send + 'static,
    V: Weigh + Send + Sync + 'static,
    E: PoolErr + Send + 'static,
{
    /// Warm `keys` (`(key, exact cost)`, in the caller's priority order, e.g. nearest to the cursor first) through the
    /// caller's executor (`rayon::spawn`, a thread pool, ...: this crate stays executor-free). Resident or in-flight
    /// keys are skipped; a prefetch NEVER evicts (it only fills free room), so it cannot push out data a viewer needs
    /// now. Returns one token cancelling the whole batch.
    pub fn prefetch(self: &Arc<Self>, keys: Vec<(K, usize)>, load: Arc<Loader<K, V, E>>, exec: &dyn Fn(Job)) -> Cancel {
        let cancel = Cancel::new();
        for (k, cost) in keys {
            if self.contains(&k) {
                continue;
            }
            let (pool, load, c) = (self.clone(), load.clone(), cancel.clone());
            exec(Box::new(move || {
                if pool.load_inner(k.clone(), cost, &c, |c| load(&k, c), false).is_err() {
                    bump(&pool.c.prefetch_err);
                }
            }));
        }
        cancel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// A host-style error: its own variants beside the pool's (exercises the generic `E` path).
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestErr {
        Pool(PoolError),
        Io(&'static str),
    }

    impl From<PoolError> for TestErr {
        fn from(e: PoolError) -> Self {
            Self::Pool(e)
        }
    }

    impl PoolErr for TestErr {
        fn as_pool(&self) -> Option<PoolError> {
            match self {
                Self::Pool(e) => Some(e.clone()),
                Self::Io(_) => None,
            }
        }
    }

    type Pool = TilePool<u32, Vec<u8>, TestErr>;

    /// A payload of `n` bytes.
    fn blob(n: usize) -> Vec<u8> {
        vec![7; n]
    }

    fn pool(cap: u64) -> Pool {
        TilePool::new(Budget::new(cap))
    }

    fn load(p: &Pool, k: u32, n: usize) -> Result<Lease<Vec<u8>>, TestErr> {
        p.get_or_load(k, n, &Cancel::new(), |_| Ok(blob(n)))
    }

    fn over(r: &Result<impl Sized, TestErr>) -> bool {
        matches!(r, Err(TestErr::Pool(PoolError::OverBudget { .. })))
    }

    #[test]
    fn hit_miss_and_lru_evict_order() {
        let p = pool(300);
        for k in 0..3 {
            load(&p, k, 100).expect("fits");
        }
        assert_eq!(p.used(), 300);
        assert!(p.get(&0).is_some(), "touch 0: now 1 is the LRU");
        load(&p, 3, 100).expect("evicts one");
        assert!(p.get(&1).is_none() && p.get(&0).is_some() && p.get(&2).is_some() && p.get(&3).is_some());
        let s = p.stats();
        assert_eq!((s.loads, s.evictions, s.used), (4, 1, 300));
    }

    #[test]
    fn oversize_is_a_hard_error_not_admitted() {
        let p = pool(100);
        assert!(matches!(load(&p, 1, 101), Err(TestErr::Pool(PoolError::OverBudget { need: 101, cap: 100, .. }))));
        assert_eq!(p.used(), 0, "a refused reservation leaves no charge");
    }

    #[test]
    fn failed_load_is_not_cached_and_releases_the_reservation() {
        let p = pool(100);
        let e = p.get_or_load(1, 50, &Cancel::new(), |_| Err(TestErr::Io("boom")));
        assert_eq!(e.err(), Some(TestErr::Io("boom")));
        assert_eq!((p.used(), p.len()), (0, 0));
        assert!(load(&p, 1, 50).is_ok(), "retry works");
    }

    #[test]
    fn leased_entries_are_pinned_and_keep_counting_after_retain() {
        let p = pool(200);
        let a = load(&p, 1, 100).expect("a");
        load(&p, 2, 100).expect("b");
        // Pool is full; 1 is pinned, 2 is evictable.
        load(&p, 3, 100).expect("evicts 2, never the pinned 1");
        assert!(p.get(&1).is_some() && p.get(&2).is_none());
        let b = load(&p, 4, 100);
        assert!(b.is_ok(), "evicts 3");
        let c = p.get(&4).expect("4");
        // Everything pinned: no room for another.
        assert!(over(&load(&p, 5, 100)));
        assert_eq!(p.stats().pinned, 2);
        // Invalidate: entries leave the map but their bytes count until the leases drop.
        p.retain(|_| false);
        assert_eq!(p.len(), 0);
        assert_eq!(p.used(), 200, "leased-after-removal bytes still count");
        drop((a, b, c));
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn replacing_an_entry_returns_its_charge() {
        let p = pool(300);
        p.insert(1, blob(100)).expect("first");
        p.insert(1, blob(40)).expect("replace");
        assert_eq!((p.used(), p.len()), (40, 1));
        p.clear();
        assert!(p.is_empty() && p.used() == 0);
    }

    #[test]
    fn cancelled_load_releases_and_caches_nothing() {
        let p = pool(100);
        let c = Cancel::new();
        c.cancel();
        let r = p.get_or_load(1, 10, &c, |_| Ok(blob(10))).map(drop);
        assert_eq!(r, Err(TestErr::Pool(PoolError::Cancelled)));
        assert_eq!((p.used(), p.len()), (0, 0));
    }

    /// The default error type works without a host enum; `cancel.check()?` converts into it.
    #[test]
    fn default_error_type_and_check_in_loader() {
        let p: TilePool<u8, Vec<u8>> = TilePool::new(Budget::new(10));
        let c = Cancel::new();
        let r = p.get_or_load(0, 4, &c, |c| {
            c.cancel();
            c.check()?;
            Ok(blob(4))
        });
        assert_eq!(r.map(drop), Err(PoolError::Cancelled));
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn single_flight_loads_once_for_many_threads() {
        let p = Arc::new(pool(1000));
        let calls = Arc::new(AtomicUsize::new(0));
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let (p, calls) = (p.clone(), calls.clone());
                thread::spawn(move || {
                    p.get_or_load(9, 100, &Cancel::new(), |_| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(std::time::Duration::from_millis(50));
                        Ok(blob(100))
                    })
                    .map(|l| l.len())
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().expect("thread"), Ok(100));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let s = p.stats();
        assert_eq!((s.loads, s.misses, s.waits + s.hits), (1, 1, 7), "one leader, seven joiners/hits");
    }

    /// A panicking loader returns its reservation and releases its waiters with `Panicked` (no hang).
    #[test]
    fn loader_panic_releases_reservation_and_waiters() {
        let p = Arc::new(pool(1000));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let leader = {
            let p = p.clone();
            thread::spawn(move || {
                p.get_or_load(3, 100, &Cancel::new(), |_| {
                    started_tx.send(()).expect("signal");
                    go_rx.recv().expect("release");
                    panic!("loader panicked");
                })
                .map(drop)
            })
        };
        started_rx.recv().expect("leader started");
        let waiter = {
            let p = p.clone();
            thread::spawn(move || p.get_or_load(3, 100, &Cancel::new(), |_| Ok(blob(100))).map(drop))
        };
        while p.stats().waits == 0 {
            thread::yield_now();
        }
        go_tx.send(()).expect("release leader");
        assert!(leader.join().is_err(), "the leader thread panicked");
        assert_eq!(waiter.join().expect("waiter"), Err(TestErr::Pool(PoolError::Panicked)));
        assert_eq!((p.used(), p.contains(&3)), (0, false));
    }

    /// A rayon worker never blocks on another thread's flight (deadlock when its leader is an ancestor frame of the
    /// same stack): it decodes privately, under its own reservation, uncached.
    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_worker_does_not_wait_on_a_flight() {
        let p = Arc::new(pool(1000));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let leader = {
            let p = p.clone();
            thread::spawn(move || {
                p.get_or_load(5, 100, &Cancel::new(), |_| {
                    started_tx.send(()).expect("signal");
                    go_rx.recv().expect("release");
                    Ok(blob(100))
                })
                .map(drop)
            })
        };
        started_rx.recv().expect("leader started");
        let rp = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let got = rp.install(|| p.get_or_load(5, 100, &Cancel::new(), |_| Ok(blob(100))));
        let private = got.expect("returned while the leader is still loading");
        assert_eq!(p.used(), 200, "leader reservation + the private lease");
        drop(private);
        go_tx.send(()).expect("release leader");
        assert_eq!(leader.join().expect("leader"), Ok(()));
        assert_eq!(p.used(), 100, "private lease dropped, leader's entry cached");
    }

    #[test]
    fn prefetch_loader_growth_does_not_evict() {
        let p = Arc::new(pool(200));
        load(&p, 0, 100).expect("resident");
        load(&p, 1, 100).expect("resident");
        let loader: Arc<Loader<u32, Vec<u8>, TestErr>> = Arc::new(|_, _| Ok(blob(150)));
        p.prefetch(vec![(2u32, 0usize)], loader, &|job| job());
        assert!(
            p.get(&0).is_some() && p.get(&1).is_some(),
            "a prefetch whose payload outgrew its reservation evicts nothing"
        );
        assert!(p.get(&2).is_none());
    }

    #[test]
    fn waiters_share_the_leaders_error() {
        let p = Arc::new(pool(1000));
        let hs: Vec<_> = (0..4)
            .map(|_| {
                let p = p.clone();
                thread::spawn(move || {
                    p.get_or_load(1, 10, &Cancel::new(), |_| {
                        thread::sleep(std::time::Duration::from_millis(40));
                        Err(TestErr::Io("bad"))
                    })
                    .map(drop)
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().expect("thread"), Err(TestErr::Io("bad")));
        }
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn loader_growth_beyond_reservation_is_checked() {
        let p = pool(100);
        let r = p.get_or_load(1, 10, &Cancel::new(), |_| Ok(blob(500)));
        assert!(over(&r));
        assert_eq!(p.used(), 0);
        let ok = p.get_or_load(2, 80, &Cancel::new(), |_| Ok(blob(20))).expect("shrinks");
        assert_eq!((ok.len(), p.used()), (20, 20), "the reservation settles to the real size");
    }

    /// A payload that records, when dropped, whether its pool's lock was free (`try_lock` from the dropping thread).
    struct Probe {
        n: usize,
        pool: Arc<std::sync::OnceLock<std::sync::Weak<TilePool<u32, Probe>>>>,
        locked_at_drop: Arc<AtomicUsize>,
    }

    impl Weigh for Probe {
        fn bytes(&self) -> usize {
            self.n
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            if let Some(p) = self.pool.get().and_then(std::sync::Weak::upgrade)
                && p.inner.try_lock().is_err()
            {
                self.locked_at_drop.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// A load whose payload outgrew its reservation and cannot fit is refused; the refused payload is dropped AFTER the
    /// pool lock is released (a GPU texture drop can be slow), and its reservation is returned.
    #[test]
    fn refused_payload_is_dropped_outside_the_lock() {
        let p = Arc::new(TilePool::<u32, Probe>::new(Budget::new(100)));
        let cell = Arc::new(std::sync::OnceLock::new());
        cell.set(Arc::downgrade(&p)).map_err(drop).expect("set once");
        let locked = Arc::new(AtomicUsize::new(0));
        let mk = |n| Probe { n, pool: cell.clone(), locked_at_drop: locked.clone() };
        let r = p.get_or_load(1, 10, &Cancel::new(), |_| Ok(mk(500)));
        assert!(matches!(r, Err(PoolError::OverBudget { .. })));
        // The private (uncached) route settles the same way; it returns the reservation its caller made.
        p.used.add(10);
        assert!(matches!(p.settle_private(10, Ok(mk(500)), true), Err(PoolError::OverBudget { .. })));
        assert_eq!(locked.load(Ordering::SeqCst), 0, "payload dropped while the pool lock was held");
        assert_eq!(p.used(), 0, "the reservation is returned");
        // Evicted entries obey the same rule.
        p.insert(3, mk(60)).expect("fits");
        p.insert(4, mk(60)).expect("evicts 3");
        assert_eq!((locked.load(Ordering::SeqCst), p.used()), (0, 60));
    }

    /// A prefetch leader (never evicts) failing `OverBudget` must not hand that error to a foreground request that
    /// joined its flight: the foreground request may evict, so it takes over and succeeds.
    #[test]
    fn foreground_waiter_retries_after_prefetch_leader_over_budget() {
        let p = Arc::new(pool(200));
        load(&p, 0, 100).expect("resident, unpinned");
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let leader = {
            let p = p.clone();
            thread::spawn(move || {
                // Reserves 50 (fits), then outgrows to 150: the extra 100 needs an eviction a prefetch may not do.
                p.load_inner(
                    1,
                    50,
                    &Cancel::new(),
                    |_| {
                        started_tx.send(()).expect("signal");
                        go_rx.recv().expect("release");
                        Ok(blob(150))
                    },
                    false,
                )
                .map(drop)
            })
        };
        started_rx.recv().expect("leader started");
        let fg = {
            let p = p.clone();
            thread::spawn(move || p.get_or_load(1, 50, &Cancel::new(), |_| Ok(blob(150))).map(|l| l.len()))
        };
        while p.stats().waits == 0 {
            thread::yield_now();
        }
        go_tx.send(()).expect("release leader");
        assert!(over(&leader.join().expect("leader")));
        assert_eq!(fg.join().expect("foreground"), Ok(150), "the foreground waiter retried as leader and evicted");
        assert!(p.get(&0).is_none() && p.get(&1).is_some());
        assert_eq!(p.used(), 150);
    }

    /// A waiter whose leader was cancelled (but the waiter was not) takes over the load.
    #[test]
    fn waiter_takes_over_a_cancelled_leader() {
        let p = Arc::new(pool(200));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let leader = {
            let p = p.clone();
            thread::spawn(move || {
                p.get_or_load(1, 50, &Cancel::new(), |c| {
                    started_tx.send(()).expect("signal");
                    go_rx.recv().expect("release");
                    c.cancel();
                    c.check()?;
                    Ok(blob(50))
                })
                .map(drop)
            })
        };
        started_rx.recv().expect("leader started");
        let fg = {
            let p = p.clone();
            thread::spawn(move || p.get_or_load(1, 50, &Cancel::new(), |_| Ok(blob(50))).map(|l| l.len()))
        };
        while p.stats().waits == 0 {
            thread::yield_now();
        }
        go_tx.send(()).expect("release leader");
        assert_eq!(leader.join().expect("leader"), Err(TestErr::Pool(PoolError::Cancelled)));
        assert_eq!(fg.join().expect("waiter"), Ok(50));
        assert_eq!((p.used(), p.stats().loads), (50, 2));
    }

    #[test]
    fn prefetch_warms_skips_resident_and_never_evicts() {
        let p = Arc::new(pool(200));
        load(&p, 0, 100).expect("resident");
        let keys = vec![(0u32, 100usize), (1, 100), (2, 100)];
        let loader: Arc<Loader<u32, Vec<u8>, TestErr>> = Arc::new(|_, _| Ok(blob(100)));
        p.prefetch(keys, loader, &|job| job());
        assert!(p.get(&0).is_some() && p.get(&1).is_some(), "0 kept, 1 warmed");
        assert!(p.get(&2).is_none(), "full: prefetch stops instead of evicting");
        assert_eq!(p.stats().prefetch_err, 1);
    }

    /// A live `Budget::set` shrink applies at the next reservation (evicting down to the new cap).
    #[test]
    fn live_budget_shrink_applies_at_next_reservation() {
        let b = Budget::new(300);
        let p: Pool = TilePool::new(b.clone());
        for k in 0..3 {
            load(&p, k, 100).expect("fits");
        }
        b.set(150);
        load(&p, 9, 50).expect("evicts down to the new cap");
        assert!(p.used() <= 150 && p.get(&9).is_some());
    }

    /// Joint overshoot test: 16 threads hammer 20 keys through a 3-entry budget; at every observed instant the charge
    /// (resident + in-flight reservations + leases) stays within the cap, and every miss is one load.
    #[test]
    fn stress_16_threads_tiny_budget_never_exceeds_cap() {
        const CAP: usize = 300;
        let p = Arc::new(pool(CAP as u64));
        let worst = Arc::new(AtomicUsize::new(0));
        let hs: Vec<_> = (0..16u32)
            .map(|t| {
                let (p, worst) = (p.clone(), worst.clone());
                thread::spawn(move || {
                    let mut x = t.wrapping_mul(2_654_435_761) | 1;
                    let mut over = 0usize;
                    for _ in 0..400 {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let k = x % 20;
                        let r = p.get_or_load(k, 100, &Cancel::new(), |_| {
                            worst.fetch_max(p.used(), Ordering::SeqCst);
                            Ok(blob(100))
                        });
                        match r {
                            Ok(l) => {
                                assert_eq!(l.len(), 100);
                                worst.fetch_max(p.used(), Ordering::SeqCst);
                            }
                            Err(TestErr::Pool(PoolError::OverBudget { .. })) => over += 1,
                            Err(e) => panic!("unexpected {e:?}"),
                        }
                    }
                    over
                })
            })
            .collect();
        let over: usize = hs.into_iter().map(|h| h.join().expect("thread")).sum();
        let s = p.stats();
        assert!(worst.load(Ordering::SeqCst) <= CAP, "observed {} > cap", worst.load(Ordering::SeqCst));
        assert!(s.peak_used <= CAP, "peak {} > cap", s.peak_used);
        assert!(s.loads >= 1 && s.loads <= s.misses, "every load is a miss that won its flight");
        assert!(s.used <= CAP && s.pinned == 0);
        assert!(over < 16 * 400, "some requests must succeed");
    }

    /// Model test: a single-threaded pool behaves exactly like a reference LRU (`Vec`, oldest first) under a
    /// pseudo-random get/insert workload with unit-size entries.
    #[test]
    fn matches_a_reference_lru() {
        const CAP: usize = 5;
        let p = pool(CAP as u64);
        let mut model: Vec<u32> = Vec::new(); // front = least recently used
        let mut x = 12345u32;
        for _ in 0..5000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let k = x % 11;
            if x.is_multiple_of(3) {
                let hit = p.get(&k).is_some();
                assert_eq!(hit, model.contains(&k), "get({k})");
                if hit {
                    model.retain(|&m| m != k);
                    model.push(k);
                }
            } else {
                load(&p, k, 1).expect("unit entry always fits");
                if model.contains(&k) {
                    model.retain(|&m| m != k);
                } else if model.len() == CAP {
                    model.remove(0);
                }
                model.push(k);
            }
            assert_eq!(p.used(), model.len());
        }
        for k in 0..11 {
            assert_eq!(p.contains(&k), model.contains(&k), "final state {k}");
        }
    }
}
