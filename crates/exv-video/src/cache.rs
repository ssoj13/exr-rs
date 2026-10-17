//! [`GopCache`]: ONE byte-budgeted, single-flight cache of decoded frames shared by every host that fetches frames of a
//! P/B clip one by one (the C ABI today; any host that does not keep its own frame cache).
//!
//! # Why
//! [`crate::decode_frame`] is stateless: frame `k` of a GOP re-decodes its whole reference chain from the I-frame, so a
//! sequential loop over a P-GOP of length `g` costs `g^2/2` frame decodes (After Effects renders sequentially through
//! the C ABI). The frames of that chain are exactly the frames the next request needs. This cache keeps them and walks
//! the SAME chain walk as the stateless path ([`crate::container::walk`]), asking the cache before it decodes.
//!
//! # How
//! * Pool = `exv_format::bufpool::TilePool` (byte LRU, `Lease` pins, single-flight per key, reserve-before-decode): not a new LRU.
//!   Key = `(clip id, decode index, channel set)`. The channel set is part of the key because a frame holds ONLY the
//!   channels its request decoded; a P reference must carry every channel its dependant decodes, so a chain of
//!   channel set `S` is made of `S` entries (no superset lookups: a reference that lacks a channel would be a decode
//!   error, not a silent hole).
//! * [`GopCache::get`] plans the chain top-down asking the pool for each needed frame ([`crate::container::plan_chain`],
//!   `leaf`): a hit is PINNED and ends that branch of the plan, so a sequential fetch decodes exactly the new frame
//!   and a random fetch decodes only the missing ancestors. The walk then decodes the planned frames ascending, each
//!   through `get_or_load` (so intermediate frames land in the cache and concurrent requests for the same frame decode
//!   it once), and a reference is released right after its last user: a P chain pins at most 2 frames, like the
//!   stateless path. Waits are only ever on a strictly EARLIER frame of the same GOP while holding earlier leases,
//!   so waits cannot cycle.
//! * A frame evicted while an in-flight decode uses it stays alive through its lease (`Arc`): no use-after-evict.
//! * Errors are never cached (pool rule); an ancestor's error is the requested frame's error. References are
//!   validated by the shared planner / `decode_one` exactly as in the stateless path (`gs <= r < di`, else `Corrupt`),
//!   and `Verify` / budget policy is the clip's [`crate::VideoIndex::limits`].
//! * On a rayon worker the pool never blocks on another thread's flight (it decodes privately, uncached): the
//!   result is still correct, only the sharing is lost.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use exv_format::bufpool::{Budget, Cancel, Lease, PoolErr, PoolError, PoolStats, TilePool, Weigh};
use exv_format::DecodeError;
use tracing::info_span;

use crate::container::{Frame, VideoIndex, decode_one, gop_start, plan_chain, walk};
use exv_codec::frame::{Chans, Planes};

/// Why a [`GopCache`] fetch failed: a decode failure, or the pool's own refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// The decode failed (corrupt / checksum / limit / bad request ...): see [`DecodeError`].
    Decode(DecodeError),
    /// The frame (or a pinned ancestor set) cannot fit the cache budget: a hard error, never a silent over-commit.
    OverBudget { need: usize, used: usize, cap: usize },
    /// The caller cancelled the fetch.
    Cancelled,
    /// A decode panicked; the reservation was returned.
    Panicked,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(e) => e.fmt(f),
            Self::OverBudget { need, used, cap } => {
                write!(f, "frame needs {need} B but {used} of {cap} B are committed and pinned (raise the memory budget)")
            }
            Self::Cancelled => f.write_str("frame fetch cancelled"),
            Self::Panicked => f.write_str("frame decode panicked"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<DecodeError> for CacheError {
    fn from(e: DecodeError) -> Self {
        Self::Decode(e)
    }
}

impl From<PoolError> for CacheError {
    fn from(e: PoolError) -> Self {
        match e {
            PoolError::OverBudget { need, used, cap } => Self::OverBudget { need, used, cap },
            PoolError::Cancelled => Self::Cancelled,
            PoolError::Panicked => Self::Panicked,
        }
    }
}

impl PoolErr for CacheError {
    fn as_pool(&self) -> Option<PoolError> {
        match *self {
            Self::OverBudget { need, used, cap } => Some(PoolError::OverBudget { need, used, cap }),
            Self::Cancelled => Some(PoolError::Cancelled),
            Self::Panicked => Some(PoolError::Panicked),
            Self::Decode(_) => None,
        }
    }
}

/// A decoded frame charged by its decoded plane bytes (only the decoded channels).
impl Weigh for Frame {
    fn bytes(&self) -> usize {
        self.planes.bytes()
    }
}

/// The planes of a pinned frame (the `planes_of` of [`walk`]; a named fn so the borrow's lifetime is inferred).
fn lease_planes(l: &Lease<Frame>) -> &Planes {
    &l.planes
}

/// `(clip id, decode index, sorted channel set)`.
type Key = (u64, u32, Arc<[usize]>);

/// The shared decoded-frame cache (module docs). One per process / per host; any number of clips share it, told apart by
/// the id [`Self::open_clip`] hands out.
pub struct GopCache {
    pool: TilePool<Key, Frame, CacheError>,
    next: AtomicU64,
}

impl GopCache {
    /// A cache charged against `budget` (resident + pinned + in-flight bytes; clones of the handle share the cap).
    pub fn new(budget: Budget) -> Self {
        Self { pool: TilePool::new(budget), next: AtomicU64::new(1) }
    }

    /// A fresh clip identity (never reused, so a closed clip's keys cannot alias a new clip's).
    pub fn open_clip(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Drop every cached frame of `clip` now (pinned ones leave when their leases drop).
    pub fn forget(&self, clip: u64) {
        self.pool.retain(|k| k.0 != clip);
    }

    /// Is display frame `k` of `clip` resident for exactly this channel set (tests / status; no LRU bump semantics promised)?
    pub fn has(&self, clip: u64, idx: &VideoIndex, k: usize, chans: Chans<'_>) -> bool {
        let key = idx.decode_idx(k).zip(chan_set(idx, chans).ok()).and_then(|(di, cs)| Some((clip, u32::try_from(di).ok()?, cs)));
        key.is_some_and(|key| self.pool.contains(&key))
    }

    /// Counters (`loads` = frames decoded), bytes charged, resident / pinned entries.
    pub fn stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Display frame `k` of `clip`, decoded for `chans` (the pool's `Lease` pins it while the caller reads it).
    ///
    /// `data` / `idx` are the clip's container and index, the SAME for every call with this `clip` id. A hit costs one
    /// lookup; a miss decodes only the frames the cache does not already hold (module docs). Bit-identical to
    /// [`crate::decode_frame`] for the same arguments.
    ///
    /// # Errors
    /// [`CacheError::Decode`]`(`[`DecodeError::Request`]`)` for an out-of-range `k` or channel, `Decode(Limit)` when the
    /// chain's peak exceeds the clip's budget, [`CacheError::OverBudget`] when the frame cannot fit this cache beside
    /// the pinned entries, [`CacheError::Cancelled`], otherwise the failing frame's decode error.
    pub fn get(
        &self,
        clip: u64,
        data: &[u8],
        idx: &VideoIndex,
        k: usize,
        chans: Chans<'_>,
        cancel: &Cancel,
    ) -> Result<Lease<Frame>, CacheError> {
        let di = idx.decode_idx(k).ok_or(DecodeError::Request("frame index out of range"))?;
        let gs = gop_start(idx, di);
        let set = chan_set(idx, chans)?;
        let cost = cost_of(idx, &set)?;
        let key_of = |f: usize| -> Result<Key, DecodeError> {
            Ok((clip, u32::try_from(f).map_err(|_| DecodeError::Corrupt("frame index exceeds u32"))?, set.clone()))
        };
        // Plan: every frame the pool already holds is pinned and ends its branch (so the chain below it is not decoded).
        let mut pinned: Vec<(usize, Lease<Frame>)> = Vec::new();
        let mut key_err = None;
        let chain = plan_chain(data, idx, gs, di, &mut |f| match key_of(f).map(|key| self.pool.get(&key)) {
            Ok(Some(lease)) => {
                pinned.push((f, lease));
                true
            }
            Ok(None) => false,
            Err(e) => {
                key_err = Some(e);
                false
            }
        })?;
        if let Some(e) = key_err {
            return Err(e.into());
        }
        let _sp = info_span!("gop_get", frame = k, di, gs, steps = chain.steps.len(), hits = pinned.len()).entered();
        idx.check_frames(chain.peak)?;
        walk(
            &chain,
            gs,
            lease_planes,
            |f, held| {
                if let Some(at) = pinned.iter().position(|(pf, _)| *pf == f) {
                    return Ok(pinned.swap_remove(at).1);
                }
                self.pool.get_or_load(key_of(f)?, cost, cancel, |c| {
                    c.check()?;
                    let planes = decode_one(data, idx, f, gs, chans, |r| held.get(r))?;
                    Ok(Frame::new(&idx.header, planes))
                })
            },
        )
    }
}

/// The normalized channel set of a request: sorted, deduplicated, every index inside the header (`All` = all of them,
/// so `All` and an explicit full list share entries).
fn chan_set(idx: &VideoIndex, chans: Chans<'_>) -> Result<Arc<[usize]>, DecodeError> {
    let n = idx.header.channels.len();
    let mut v: Vec<usize> = match chans {
        Chans::All => (0..n).collect(),
        Chans::Only(list) => list.to_vec(),
    };
    v.sort_unstable();
    v.dedup();
    if v.last().is_some_and(|&c| c >= n) {
        return Err(DecodeError::Request("channel index out of range"));
    }
    Ok(v.into())
}

/// Decoded bytes of one frame of `set`: the exact reservation of its load.
fn cost_of(idx: &VideoIndex, set: &[usize]) -> Result<usize, DecodeError> {
    let h = &idx.header;
    set.iter()
        .try_fold(0usize, |acc, &ci| {
            let ch = h.channels.get(ci)?;
            acc.checked_add(h.plane_len(ch.dtype)?)
        })
        .ok_or(DecodeError::Corrupt("frame size overflows"))
}
