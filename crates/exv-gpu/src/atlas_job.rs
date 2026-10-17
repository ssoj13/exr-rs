//! `atlas_job.rs` — the ASYNC half of the resident atlas: plan / worker decode / install (P1-6).
//!
//! ```text
//! UI thread (owns ExvResidentAtlas: TileAtlas LRU, epoch, counters, Flight)       worker thread (ONE per atlas)
//!   plan(clip, frame, want) -> Option<Plan>   pass 1 only, host-cheap
//!   submit(plan, data, idx)  --- Job {Arc bytes, Arc index, Work, Cancel} --->    run_job: TileChains over its own Arcs,
//!                                                                                  decode_* inside GpuContext::scoped,
//!   poll() -> Done  <------------------------ Done {jid, epoch, seen, res} ----    wake()  (viewer repaint)
//!   install(done): jid + wanted-node + merge rules, then atlas.insert
//! ```
//! `ExvResidentAtlas::request` (the synchronous API every gate uses) is `plan` + an INLINE [`run_work`] + `install`, so
//! there is exactly one decode path; the worker only changes WHERE `run_work` runs.
//!
//! # Why a dedicated thread (D-25 in `docs/DESIGN_LOG.md`)
//! wgpu error scopes are thread-local (wgpu 30): every GPU section of a decode must open + pop its scope on ONE thread
//! ([`GpuContext::scoped`]), which a rayon task migrating between helpers (or a scancache scheduler job) cannot
//! promise across the several scoped sections of one batch. One owned thread also owns the jph `ResidentCtx` (`&mut`
//! per decode) with no lock. The CPU-parallel parts (CRC checks, plane decode) still fan out over rayon FROM the worker.
//!
//! # State discipline
//! All atlas state stays UI-thread-only. The worker gets owned inputs (`Job`) and returns owned textures (`Done`); the
//! queue order = data order (the worker submits its GPU work before sending `Done`, the UI's compose submits after
//! the receive), so no fence is needed.

use crate::atlas::{Batch, Clip, ExvDec, Miss, Route, TileTex};
use crate::prelude::*;
use bufpool::Cancel;
use exv_gpu_kernels::GpuContext;
use jph_wgpu_rs::RegionKey;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// "A decode landed" callback the worker fires after sending a [`Done`] (the viewer posts its repaint event; same
/// type as `scancache::Wake`).
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// The clip's container bytes, shared with a job (the viewer's `Arc<fsio::Src>` coerces to it).
pub type Bytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// How long dropping the worker waits for a mid-decode job before abandoning the thread (never block the UI on close).
const JOIN_WAIT: Duration = Duration::from_secs(1);

/// What pass 1 of a request decided to decode: owned (no borrow of the clip), so it can cross to the worker.
pub(crate) struct Work {
    /// Job generation (assigned at registration; `0` before).
    pub jid: u64,
    pub profile: exv_codec::Profile,
    /// Display-frame decode index the tiles were validated for (stamped into the installed [`crate::Planes`]).
    pub seen: u32,
    pub level: Option<u32>,
    pub route: Route,
    pub gpu_htj2k: bool,
    pub dtypes: Vec<Dtype>,
    pub misses: Vec<Miss>,
    /// Atlas epoch at plan time (a clip switch in between drops the whole result).
    pub epoch: u64,
    pub limits: DecodeLimits,
    /// The caller's full wanted channel set (what an in-flight tile promises once installed).
    pub want: Vec<usize>,
}

/// A planned miss batch ([`crate::ExvResidentAtlas::plan`]): hand it to `submit` (async) or let `request` run it inline.
pub struct Plan(pub(crate) Work);

impl Plan {
    /// Number of tiles this plan would decode.
    pub fn tiles(&self) -> usize {
        self.0.misses.len()
    }
}

/// One decode job: a [`Work`] plus the owned clip it reads and its cancel token.
pub(crate) struct Job {
    pub data: Bytes,
    pub idx: Arc<exv_video::VideoIndex>,
    pub work: Work,
    pub cancel: Cancel,
}

/// Why a job produced no tiles.
pub(crate) enum JobErr {
    /// The job's token was cancelled before it finished (never an error for the caller).
    Cancelled,
    /// A decode / device failure (`install` returns it when its job is still live).
    Failed(String),
    /// The job PANICKED on the worker. Like `Failed` for `install`, but the worker thread ends after sending it and the
    /// atlas rebuilds it on the next `submit`: the unwound thread may have left wgpu error scopes pushed (they are
    /// thread-local), and a fresh thread starts with none.
    Panicked(String),
}

/// A finished job: decoded textures per miss, or why not. Opaque; feed it to `install`.
pub struct Done {
    pub(crate) jid: u64,
    pub(crate) epoch: u64,
    pub(crate) seen: u32,
    pub(crate) res: Result<Vec<(Miss, TileTex)>, JobErr>,
}

impl Done {
    /// `true` when the job was cancelled (its result is empty and ignored by `install`).
    pub fn is_cancelled(&self) -> bool {
        matches!(self.res, Err(JobErr::Cancelled))
    }
}

/// Decode `w` over `clip` on the CALLER's thread with `dec`: THE decode body, run inline by `request` and on the worker
/// by [`run_job`]. Pre-opens each distinct miss node (`open_reader` reads `walk.opened`), then runs the clip's route.
/// A cancelled token turns into [`JobErr::Cancelled`] even when the decode itself finished or failed (the result is
/// stale by definition).
pub(crate) fn run_work(dec: &mut ExvDec, clip: Clip<'_>, w: Work, cancel: &Cancel) -> Done {
    let res = run_misses(dec, clip, &w, cancel);
    let res = match res {
        _ if cancel.is_cancelled() => Err(JobErr::Cancelled),
        Ok(texs) => Ok(w.misses.into_iter().zip(texs).collect()),
        Err(e) => Err(JobErr::Failed(e)),
    };
    Done { jid: w.jid, epoch: w.epoch, seen: w.seen, res }
}

/// The route batch of `w` (see [`run_work`]).
fn run_misses(dec: &mut ExvDec, clip: Clip<'_>, w: &Work, cancel: &Cancel) -> Result<Vec<TileTex>, String> {
    let mut walk = exv_video::TileChains::new(clip.data, clip.idx);
    let mut nodes: Vec<u32> = w.misses.iter().map(|m| m.node).collect();
    nodes.sort_unstable();
    nodes.dedup();
    for n in nodes {
        walk.reader(n as usize).map_err(|e| format!("frame {n}: {e}"))?;
    }
    let b = Batch { walk: &walk, profile: &w.profile, dtypes: &w.dtypes, misses: &w.misses, limits: w.limits, cancel };
    match w.route {
        Route::Htj2k => dec.decode_htj2k(&b, w.level, w.gpu_htj2k),
        Route::Wave => dec.decode_wave(&b),
        Route::Rt => dec.decode_rt(&b),
        Route::Bc6h => dec.decode_bc6h(&b),
    }
}

/// [`run_work`] over a job's owned clip.
pub(crate) fn run_job(dec: &mut ExvDec, job: Job) -> Done {
    let data: &[u8] = (*job.data).as_ref();
    run_work(dec, Clip { data, idx: &job.idx }, job.work, &job.cancel)
}

/// What an in-flight job will deliver for one tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pending {
    pub jid: u64,
    /// Frame the job was planned for.
    pub seen: u32,
    /// INTRA node the job decodes.
    pub node: u32,
    /// Full wanted channel set of the plan (the tile holds at least these once installed).
    pub want: Vec<usize>,
    /// Estimated resident bytes of the planes the job adds (charged by `check_visible` before they exist).
    pub est: usize,
    /// A MERGE job (only the channels the resident tile lacked): `est` is a delta ON TOP of the resident tile. A full
    /// decode REPLACES the resident tile (install frees the old planes), so it never adds to them.
    pub merge: bool,
}

/// The pure (device-free) bookkeeping of the async path: which tiles are in flight under which job generation and
/// which tiles the LAST plan wanted resident. Every install rule is a method here so it is unit-testable on the CPU.
#[derive(Default)]
pub(crate) struct Flight {
    jid: u64,
    pending: HashMap<RegionKey, Pending>,
    /// `(node, channels)` each tile of the last plan must hold. A result lands while its tile is still wanted (visible),
    /// whatever node it decoded: a tile of an older frame is valid for the node it carries, and the next plan
    /// re-requests the current one (otherwise a decode slower than the frame period would never display anything).
    wanted: HashMap<RegionKey, (u32, Vec<usize>)>,
}

impl Flight {
    /// A fresh job generation (strictly increasing, never `0`).
    pub fn next_id(&mut self) -> u64 {
        self.jid += 1;
        self.jid
    }

    /// Record `p` as the job delivering `key` (replaces an older job's entry: that job's result is then stale).
    pub fn add(&mut self, key: RegionKey, p: Pending) {
        self.pending.insert(key, p);
    }

    /// Tiles currently in flight.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Resident bytes `key` will occupy once its in-flight job lands, given the `resident` bytes it holds now: a MERGE
    /// adds its delta to the resident planes; a full decode REPLACES them (the old tile is freed at install), so the
    /// footprint is the larger of the two, never the sum. No job in flight = just `resident`.
    pub fn charge(&self, key: &RegionKey, resident: usize) -> usize {
        match self.pending.get(key) {
            None => resident,
            Some(p) if p.merge => resident.saturating_add(p.est),
            Some(p) => resident.max(p.est),
        }
    }

    /// Does an in-flight job already deliver at least `want` for `key` at INTRA `node`?
    pub fn covers(&self, key: &RegionKey, node: u32, want: &[usize]) -> bool {
        self.pending.get(key).is_some_and(|p| p.node == node && want.iter().all(|c| p.want.contains(c)))
    }

    /// [`Self::covers`] for the frame `seen` itself, without knowing the node (the no-chain-walk fast path: a job
    /// planned for the same frame and channels cannot be superseded by a different node).
    pub fn covers_seen(&self, key: &RegionKey, seen: u32, want: &[usize]) -> bool {
        self.pending.get(key).is_some_and(|p| p.seen == seen && want.iter().all(|c| p.want.contains(c)))
    }

    /// The node an in-flight job decodes for `key`.
    pub fn node_of(&self, key: &RegionKey) -> Option<u32> {
        self.pending.get(key).map(|p| p.node)
    }

    /// Replace the wanted set (called by every plan).
    pub fn set_wanted(&mut self, wanted: HashMap<RegionKey, (u32, Vec<usize>)>) {
        self.wanted = wanted;
    }

    /// Does job `jid` still own at least one in-flight entry (i.e. was it not superseded / cancelled)?
    pub fn live(&self, jid: u64) -> bool {
        self.pending.values().any(|p| p.jid == jid)
    }

    /// May job `jid`'s result for `key` be installed? It must still own the in-flight entry (no newer job, no
    /// `cancel_all`) AND the tile must still be wanted by the last plan (visible); the node it decoded may be older than
    /// the wanted one (see [`Self::wanted`]).
    pub fn accepts(&self, jid: u64, key: &RegionKey) -> bool {
        self.pending.get(key).is_some_and(|p| p.jid == jid) && self.wanted.contains_key(key)
    }

    /// Does the last plan still want ANY of `keys`? (A job serving none of them is worth cancelling.)
    pub fn serves(&self, keys: &[RegionKey]) -> bool {
        keys.iter().any(|k| self.wanted.contains_key(k))
    }

    /// Drop every in-flight entry of job `jid` (its result was consumed or discarded).
    pub fn finish(&mut self, jid: u64) {
        self.pending.retain(|_, p| p.jid != jid);
    }

    /// Forget everything in flight and wanted (`cancel_all`): late results then fail [`Self::accepts`].
    pub fn clear(&mut self) {
        self.pending.clear();
        self.wanted.clear();
    }
}

/// Must the running job (atlas epoch `job_epoch`, tiles `keys`) be cancelled after a plan? Only when the clip changed
/// or NONE of its tiles is wanted any more: a job that still has a visible tile keeps delivering (and lands it), so
/// continuous seeking / playback with a decode slower than the frame period still shows something.
pub(crate) fn useless(job_epoch: u64, epoch: u64, flight: &Flight, keys: &[RegionKey]) -> bool {
    job_epoch != epoch || !flight.serves(keys)
}

/// THE install rule: a result lands only when it is `accepted` (job still owns the tile, tile still wanted), the atlas
/// epoch is `fresh` (no clip switch) and, if it is a MERGE job (it carries only the channels the tile was missing), the
/// resident tile still holds the SAME node. A delta over an absent or re-decoded tile would be a partial tile (compose
/// then fails on its missing channels), so it is dropped and the next plan re-requests the full set.
pub(crate) fn lands(accepted: bool, fresh: bool, merge: bool, old_node: Option<u32>, node: u32) -> bool {
    accepted && fresh && (!merge || old_node == Some(node))
}

/// How many tiles one job may take: the decode pool (`budget`) over the largest per-tile estimate `per`, at least one.
/// The textures a job holds before `install` are transient VRAM charged to that pool.
pub(crate) fn chunk_cap(budget: u64, per: usize) -> usize {
    usize::try_from(budget / per.max(1) as u64).unwrap_or(usize::MAX).max(1)
}

/// The one decode worker of an atlas: a dedicated thread running [`run_job`] for each job it receives, one at a time.
pub(crate) struct Worker {
    tx: Option<mpsc::Sender<Job>>,
    rx: mpsc::Receiver<Done>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Worker {
    /// Start the thread. It builds its OWN [`ExvDec`] (so the jph `ResidentCtx`, if the route needs one, is created and
    /// owned on the thread that uses it) over `ctx`, sharing the atlas' `uploaded` counter.
    pub fn spawn(ctx: GpuContext, uploaded: Arc<AtomicU64>, decode_budget: u64, wake: Option<Wake>) -> Result<Self, String> {
        let (tx, jobs) = mpsc::channel::<Job>();
        let (done_tx, rx) = mpsc::channel::<Done>();
        let handle = thread::Builder::new()
            .name("exv-atlas".into())
            .spawn(move || {
                let mut dec = ExvDec::new(ctx, uploaded, decode_budget);
                // Ends when the atlas drops its sender, or right after a panicked job (the atlas rebuilds the thread).
                for job in jobs {
                    let (jid, epoch, seen) = (job.work.jid, job.work.epoch, job.work.seen);
                    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(&mut dec, job)));
                    let panicked = ran.is_err();
                    let done = ran.unwrap_or_else(|p| {
                        let what = p.downcast_ref::<&str>().map(|s| (*s).to_string()).or_else(|| p.downcast_ref::<String>().cloned());
                        Done { jid, epoch, seen, res: Err(JobErr::Panicked(format!("resident decode worker panicked: {}", what.unwrap_or_default()))) }
                    });
                    if done_tx.send(done).is_err() {
                        break; // the atlas is gone
                    }
                    if let Some(w) = &wake {
                        w();
                    }
                    if panicked {
                        break;
                    }
                }
            })
            .map_err(|e| format!("cannot start the resident decode worker ({e})"))?;
        Ok(Self { tx: Some(tx), rx, handle: Some(handle) })
    }

    /// Queue `job` (the atlas sends only when the worker is idle: one job in flight).
    pub fn send(&self, job: Job) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("resident decode worker is closed")?;
        tx.send(job).map_err(|_| "resident decode worker ended".to_string())
    }

    /// The next finished job, `Ok(None)` while the worker is busy / idle, `Err` when the thread is gone.
    pub fn poll(&self) -> Result<Option<Done>, String> {
        match self.rx.try_recv() {
            Ok(d) => Ok(Some(d)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("resident decode worker ended".to_string()),
        }
    }
}

impl Drop for Worker {
    /// Close the job channel (the thread leaves its loop after the job it is running; the atlas cancelled that job
    /// first) and join with a bounded wait, like scancache's pool: a stalled read must never freeze the UI on close.
    fn drop(&mut self) {
        self.tx = None;
        let Some(h) = self.handle.take() else { return };
        let deadline = Instant::now() + JOIN_WAIT;
        while !h.is_finished() {
            if Instant::now() >= deadline {
                log::warn!("exv-atlas: shutdown timeout, abandoning the decode worker");
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        if h.join().is_err() {
            log::warn!("exv-atlas: decode worker panicked on shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(rx: u32) -> RegionKey {
        RegionKey { level: u32::MAX, rx, ry: 0 }
    }

    fn pend(jid: u64, seen: u32, node: u32, want: &[usize]) -> Pending {
        Pending { jid, seen, node, want: want.to_vec(), est: 10, merge: false }
    }

    /// `Done`, `Plan` and what a job owns cross the thread boundary; the device context is shared by clone.
    #[test]
    fn what_crosses_threads_is_send() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync>() {}
        send_sync::<GpuContext>();
        send::<Job>();
        send::<Done>();
        send::<Plan>();
        send::<crate::ExvResidentAtlas>();
    }

    /// A result installs only while its job still owns the tile AND the tile is still wanted (visible); the node it
    /// decoded does not matter (a stale-node tile for a visible key lands, the next plan re-requests the current node).
    #[test]
    fn install_rules_jid_and_wanted_key() {
        let mut f = Flight::default();
        let g1 = f.next_id();
        f.add(key(0), pend(g1, 3, 7, &[0, 1]));
        f.set_wanted(HashMap::from([(key(0), (7, vec![0, 1]))]));
        assert!(f.accepts(g1, &key(0)), "live job, wanted key");
        assert!(!f.accepts(g1, &key(1)), "a key the job never owned");
        // The view moved to another frame: the key is wanted at a NEWER node. The old-node decode still lands...
        f.set_wanted(HashMap::from([(key(0), (8, vec![0, 1]))]));
        assert!(f.accepts(g1, &key(0)), "a stale-node tile for a still-wanted key lands");
        // ...and the next plan sees the resident tile at node 7 != 8 and the in-flight entry gone: a fresh request.
        assert!(!f.covers(&key(0), 8, &[0, 1]), "the in-flight job decodes node 7, not the wanted 8");
        // A newer job takes the key over: the older result is stale.
        let g2 = f.next_id();
        f.add(key(0), pend(g2, 4, 8, &[0, 1]));
        assert!(!f.accepts(g1, &key(0)), "jid mismatch");
        assert!(f.accepts(g2, &key(0)));
        // The key left the wanted set (the view moved off it): drop.
        f.set_wanted(HashMap::new());
        assert!(!f.accepts(g2, &key(0)), "no longer wanted");
    }

    #[test]
    fn finish_and_clear_release_the_flight() {
        let mut f = Flight::default();
        let (g1, g2) = (f.next_id(), f.next_id());
        f.add(key(0), pend(g1, 0, 0, &[0]));
        f.add(key(1), pend(g2, 0, 0, &[0]));
        assert_eq!((f.len(), f.live(g1), f.live(g2)), (2, true, true));
        f.finish(g1);
        assert_eq!((f.len(), f.live(g1), f.live(g2)), (1, false, true));
        f.set_wanted(HashMap::from([(key(1), (0, vec![0]))]));
        f.clear();
        assert_eq!((f.len(), f.live(g2)), (0, false));
        assert!(!f.accepts(g2, &key(1)), "cancel_all: a late result is dropped");
    }

    /// `covers` = same node and a superset of the wanted channels; `covers_seen` = same frame (no node needed).
    #[test]
    fn coverage_needs_node_or_frame_and_channels() {
        let mut f = Flight::default();
        let g = f.next_id();
        f.add(key(0), pend(g, 5, 9, &[0, 1, 2]));
        assert!(f.covers(&key(0), 9, &[0, 2]));
        assert!(!f.covers(&key(0), 9, &[0, 3]), "channel 3 is not on its way");
        assert!(!f.covers(&key(0), 8, &[0]), "another node");
        assert!(f.covers_seen(&key(0), 5, &[1]));
        assert!(!f.covers_seen(&key(0), 6, &[1]), "another frame: the chain walk decides");
        assert_eq!(f.node_of(&key(0)), Some(9));
    }

    /// The install rule: nothing lands unaccepted or under another epoch (a clip switch), a delta-only (merge) result
    /// over an absent or re-decoded tile is dropped, a full result replaces whatever is there.
    #[test]
    fn install_rule_epoch_and_merge_drop() {
        assert!(lands(true, true, true, Some(4), 4), "merge over the tile it was planned against");
        assert!(!lands(true, true, true, None, 4), "merge over an evicted tile would be partial");
        assert!(!lands(true, true, true, Some(3), 4), "merge over a tile re-decoded at another node");
        assert!(lands(true, true, false, None, 4), "a full decode lands anywhere");
        assert!(lands(true, true, false, Some(3), 4), "a full decode replaces an old-node tile");
        assert!(!lands(false, true, false, None, 4), "never an unaccepted result");
        assert!(!lands(true, false, false, None, 4), "a result of another atlas epoch (clip switch) is dropped");
    }

    /// Prune rule: a job keeps running while ANY of its tiles is wanted (playback / scrub with a slow decode must still
    /// show something); it is cancelled only when none is, or when the atlas epoch moved.
    #[test]
    fn prune_only_when_nothing_is_wanted_or_the_epoch_changed() {
        let mut f = Flight::default();
        let keys = [key(0), key(1), key(2)];
        f.set_wanted(HashMap::from([(key(2), (5, vec![0]))]));
        assert!(!useless(1, 1, &f, &keys), "one tile of the job is still wanted (at any node): keep it");
        assert!(useless(1, 2, &f, &keys), "another epoch: the clip changed");
        f.set_wanted(HashMap::from([(key(9), (5, vec![0]))]));
        assert!(useless(1, 1, &f, &keys), "none of its tiles is visible any more");
        f.set_wanted(HashMap::new());
        assert!(useless(1, 1, &f, &[]), "an empty job serves nothing");
    }

    /// `check_visible` arithmetic: a MERGE adds its delta to the resident planes, a full REPLACE is the larger of old and
    /// new (install frees the old tile), a fresh key counts its estimate, no job counts the resident bytes.
    #[test]
    fn charge_is_replace_max_merge_sum() {
        let mut f = Flight::default();
        let g = f.next_id();
        f.add(key(0), Pending { merge: true, est: 40, ..pend(g, 0, 0, &[0]) });
        f.add(key(1), Pending { merge: false, est: 40, ..pend(g, 0, 0, &[0]) });
        f.add(key(2), Pending { merge: false, est: 40, ..pend(g, 0, 0, &[0]) });
        assert_eq!(f.charge(&key(0), 100), 140, "merge: resident + delta");
        assert_eq!(f.charge(&key(1), 100), 100, "replace of a bigger resident tile: max, not the sum");
        assert_eq!(f.charge(&key(1), 10), 40, "replace of a smaller one: the new estimate");
        assert_eq!(f.charge(&key(2), 0), 40, "fresh key: the estimate");
        assert_eq!(f.charge(&key(3), 100), 100, "nothing in flight: resident only");
    }

    /// One job takes `decode pool / tile estimate` tiles (at least one, whatever the estimate).
    #[test]
    fn chunk_cap_is_pool_over_tile_bytes() {
        assert_eq!(chunk_cap(1000, 300), 3);
        assert_eq!(chunk_cap(1000, 5000), 1, "a tile bigger than the pool still decodes, alone");
        assert_eq!(chunk_cap(1000, 0), 1000, "a zero estimate is treated as one byte");
        assert_eq!(chunk_cap(0, 10), 1);
    }

    // ---- GPU gates (`--include-ignored --test-threads=1`): the async path against the synchronous one ----

    use crate::testkit::*;
    use crate::{Chans, ContentKey, DisplayKey, ExvResidentAtlas, Want};

    /// A clip the way the viewer holds it for the worker: shared bytes + the index parsed once.
    struct Shared {
        data: Bytes,
        idx: Arc<exv_video::VideoIndex>,
    }

    fn shared(bytes: Vec<u8>) -> Shared {
        shared_with(bytes, DecodeLimits::default())
    }

    /// [`shared`] under the caller's `limits` (CRC policy, budget): they ride on the index the job reads through.
    fn shared_with(bytes: Vec<u8>, limits: DecodeLimits) -> Shared {
        let idx = Arc::new(exv_video::VideoIndex::parse(&bytes).expect("index").with_limits(limits));
        Shared { data: Arc::new(bytes), idx }
    }

    fn want<'a>(tiles: &'a [(u32, u32)], chans: Chans<'a>) -> Want<'a> {
        Want { tiles, level: None, chans }
    }

    /// Block (bounded) until the worker answers.
    fn wait_done(a: &mut ExvResidentAtlas) -> Done {
        let t = Instant::now();
        loop {
            if let Some(d) = a.poll() {
                return d;
            }
            assert!(t.elapsed() < Duration::from_secs(120), "the decode worker never answered");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// The viewer's present loop for one frame: plan, submit, drain + install, until nothing is left to decode.
    /// Returns the tiles installed.
    fn settle(a: &mut ExvResidentAtlas, s: &Shared, frame: usize, tiles: &[(u32, u32)], chans: Chans<'_>) -> usize {
        let bytes: &[u8] = (*s.data).as_ref();
        let mut installed = 0;
        for _ in 0..64 {
            let clip = Clip { data: bytes, idx: &s.idx };
            match a.plan(clip, frame, want(tiles, chans), ContentKey(1), DisplayKey(0)).expect("plan") {
                Some(p) => a.submit(p, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit"),
                None if !a.busy() => return installed,
                None => {}
            }
            let d = wait_done(a);
            installed += a.install(d).expect("install");
        }
        panic!("settle: still planning after 64 rounds");
    }

    /// **Async == sync gate:** SKIP clips (lossy wave = CPU decode + upload, EXV-RT = GPU block kernel) shown through
    /// plan -> worker -> install are bit-exact vs the CPU `decode_frame` in ascending and seek order, with the SAME
    /// decode / reuse / plane counters as the synchronous `request`; then a channel-set growth decodes ONLY the delta
    /// through the worker (a merge job) and the merged tile is still bit-exact.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_async_path_matches_sync_bit_exact() {
        use exv_codec::Profile;
        use exv_video::{decode_frame, write_video};

        let (w, h) = (600u32, 400u32);
        let a = f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let b = poke_tile(&a, 2, 1);
        let c = poke_tile(&b, 0, 0);
        let frames = vec![a.clone(), a.clone(), b.clone(), b.clone(), c.clone(), a.clone()];
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = jph_wgpu_rs::Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let tiles = tile_grid(3, 2);
        for (name, prof) in [("lossy wave", Profile::lossy_wave(8)), ("EXV-RT", Profile::exv_rt_dwt(16, 0))] {
            let bytes = write_video(&frames, &prof, 8, 0, 0).expect("encode");
            let s = shared(bytes.clone());
            let cpu: Vec<_> = (0..frames.len())
                .map(|k| decode_frame(&bytes, &s.idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu").planes)
                .collect();
            let (mut sync, mut asy) = (new_atlas(&ctx), new_atlas(&ctx));
            for &k in &[0usize, 1, 2, 3, 4, 5, 2, 0, 5] {
                sync.request(Clip { data: &bytes, idx: &s.idx }, k, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("sync request");
                settle(&mut asy, &s, k, &tiles, Chans::All);
                for &t in &tiles {
                    assert_tile(&gpu, &asy, &cpu[k], &s.idx.header, t, 3, &format!("{name} async frame {k}"));
                }
                let (st, at) = (sync.stats(), asy.stats());
                assert_eq!((at.decodes, at.reused, at.planes, at.inflight), (st.decodes, st.reused, st.planes, 0), "{name} frame {k}: async {at} vs sync {st}");
            }

            // Channel growth through the worker: RGB first, then +A decodes only A (a merge job).
            let mut grow = new_atlas(&ctx);
            let n = tiles.len();
            assert_eq!(settle(&mut grow, &s, 0, &tiles, Chans::Only(&[0, 1, 2])), n);
            assert_eq!(grow.stats().planes, 3 * n as u64);
            assert_eq!(settle(&mut grow, &s, 0, &tiles, Chans::Only(&[0, 1, 2, 3])), n, "{name}: the delta lands for every tile");
            assert_eq!((grow.stats().decodes, grow.stats().planes), (2 * n as u64, 4 * n as u64), "{name}: one new plane per tile");
            for &t in &tiles {
                assert_tile(&gpu, &grow, &cpu[0], &s.idx.header, t, 4, &format!("{name} merged"));
            }
        }
    }

    /// **Slow-decode gate (playback / scrub):** while a job runs, `plan` for LATER frames returns at once without touching
    /// the atlas, and the running job is NOT cancelled (its tiles are still visible): it lands its tiles, stamped with the
    /// node they carry, and the next plan re-requests the latest frame, which ends bit-exact. Before the fix a decode slower
    /// than the frame period was stale on arrival, so nothing new was ever displayed.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_slow_decode_lands_stale_tiles_then_replans_to_the_latest_frame() {
        use exv_codec::Profile;
        use exv_video::{decode_frame, write_video};

        let (w, h) = (1024u32, 768u32);
        let names = ["R", "G", "B", "A", "Y"];
        let frames: Vec<_> = (0..12u32).map(|k| f16_image(w, h, 256, 256, &names, k * 7)).collect();
        let bytes = write_video(&frames, &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode all-INTRA clip");
        let s = shared(bytes.clone());
        let last = frames.len() - 1;
        let cpu = decode_frame(&bytes, &s.idx, last, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu").planes;
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = jph_wgpu_rs::Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let tiles = tile_grid(4, 3);
        let n = tiles.len();
        let mut atlas = new_atlas(&ctx);
        let clip = Clip { data: &bytes, idx: &s.idx };

        let first = atlas.plan(clip, 0, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan 0").expect("frame 0 misses");
        assert_eq!(first.tiles(), n);
        atlas.submit(first, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit");
        for k in 1..=last {
            let t = Instant::now();
            let p = atlas.plan(clip, k, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan");
            assert!(p.is_none(), "frame {k}: the worker is busy, the latest viewport is re-planned after its Done");
            assert!(t.elapsed() < Duration::from_secs(5), "plan must never wait for a decode ({:?})", t.elapsed());
            // `plan` inserts nothing: only `install` makes tiles resident.
            assert_eq!(atlas.stats().resident_bytes, 0, "plan must not install anything");
        }
        let slow = wait_done(&mut atlas);
        assert!(!slow.is_cancelled(), "the job still has visible tiles: it must not be cancelled");
        assert_eq!(atlas.install(slow).expect("stale-node install"), n, "visible tiles land even though the view moved on");
        assert_eq!(atlas.get(None, 0, 0).map(|p| p.node), Some(0), "stamped with the node it decoded (frame 0)");
        assert_eq!(settle(&mut atlas, &s, last, &tiles, Chans::All), n, "the next plan re-requests the latest frame");
        assert_eq!(atlas.decode_count(), 2 * n as u64);
        for &t in &tiles {
            assert_eq!(atlas.get(None, t.0, t.1).map(|p| p.node), Some(last as u32));
            assert_tile(&gpu, &atlas, &cpu, &s.idx.header, t, 3, "latest frame after a stale landing");
        }
    }

    /// **Plan-signature gate (P1-24 b/h):** while the worker is busy, re-planning the SAME frame / tiles / channels
    /// (every UI redraw) must not re-open the frame and re-walk the chains; a different signature does, and the result
    /// still lands and re-plans exactly like before.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_busy_replan_with_the_same_signature_skips_pass_1() {
        use crate::atlas::FRAME_OPENS;
        use exv_codec::Profile;
        use exv_video::write_video;
        use std::sync::atomic::Ordering::Relaxed;

        let names = ["R", "G", "B", "A", "Y"];
        let frames: Vec<_> = (0..6u32).map(|k| f16_image(1024, 768, 256, 256, &names, k * 7)).collect();
        let bytes = write_video(&frames, &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode all-INTRA clip");
        let s = shared(bytes.clone());
        let ctx = GpuContext::new().expect("gpu ctx");
        let tiles = tile_grid(4, 3);
        let n = tiles.len();
        let mut atlas = new_atlas(&ctx);
        let clip = Clip { data: &bytes, idx: &s.idx };

        let first = atlas.plan(clip, 0, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan 0").expect("misses");
        atlas.submit(first, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit");
        // `busy()` holds until `poll`: the worker's progress cannot change what these plans see.
        let opens = FRAME_OPENS.load(Relaxed);
        for _ in 0..25 {
            assert!(atlas.plan(clip, 3, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").is_none(), "busy");
        }
        assert_eq!(FRAME_OPENS.load(Relaxed), opens + 1, "25 identical busy plans walk the chains once");
        assert!(atlas.plan(clip, 4, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").is_none());
        assert_eq!(FRAME_OPENS.load(Relaxed), opens + 2, "another frame is another signature");
        let fewer = &tiles[..n - 1];
        assert!(atlas.plan(clip, 4, want(fewer, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").is_none());
        assert_eq!(FRAME_OPENS.load(Relaxed), opens + 3, "another visible set is another signature");

        let done = wait_done(&mut atlas);
        assert!(!done.is_cancelled(), "its tiles are still visible");
        let landed = atlas.install(done).expect("install");
        assert_eq!(landed, n - 1, "every still-visible tile lands (stale node), the unwanted one is dropped");
        assert_eq!(settle(&mut atlas, &s, 4, &tiles, Chans::All), n, "the next plan re-requests frame 4 for every tile");
    }

    /// **Dispatch checkpoint gate (P1-24 a):** the atlas batch decoders refuse to start (typed error, no submit) under
    /// a cancelled token, and run normally under a live one.
    #[test]
    #[ignore = "requires a GPU"]
    fn dispatch_tile_batches_stop_on_a_cancelled_token() {
        use crate::dispatch::{decode_tiles_exvrt_gpu, decode_tiles_wave_gpu};
        let ctx = GpuContext::new().expect("gpu ctx");
        let live = Cancel::new();
        assert!(decode_tiles_wave_gpu(&ctx, &[], &[], &live).expect("empty batch").is_empty());
        assert!(decode_tiles_exvrt_gpu(&ctx, &[], &[], &live).expect("empty batch").is_empty());
        let dead = Cancel::new();
        dead.cancel();
        assert!(decode_tiles_wave_gpu(&ctx, &[], &[], &dead).is_err());
        assert!(decode_tiles_exvrt_gpu(&ctx, &[], &[], &dead).is_err());
    }

    /// **View-change cancel gate:** a job none of whose tiles is visible any more is cancelled by the next plan and
    /// installs nothing (it may have finished before the plan: then the install drops its tiles, so both outcomes are
    /// accepted explicitly); the new view decodes normally afterwards.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_view_change_cancels_a_job_nobody_wants() {
        use exv_codec::Profile;
        use exv_video::write_video;

        let img = f16_image(2048, 2048, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let bytes = write_video(&[img], &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode");
        let s = shared(bytes.clone());
        let ctx = GpuContext::new().expect("gpu ctx");
        let (top, bottom): (Vec<_>, Vec<_>) = tile_grid(8, 8).into_iter().partition(|&(_, ry)| ry < 4);
        let mut atlas = new_atlas(&ctx);
        let clip = Clip { data: &bytes, idx: &s.idx };

        let p = atlas.plan(clip, 0, want(&top, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
        atlas.submit(p, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit");
        assert!(atlas.plan(clip, 0, want(&bottom, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").is_none(), "busy");
        let done = wait_done(&mut atlas);
        let cancelled = done.is_cancelled();
        assert_eq!(atlas.install(done).expect("install"), 0, "no top tile is visible: nothing lands (cancelled = {cancelled})");
        assert_eq!((atlas.decode_count(), atlas.stats().resident_bytes), (0, 0));
        assert_eq!(settle(&mut atlas, &s, 0, &bottom, Chans::All), bottom.len());
        assert!(top.iter().all(|&(rx, ry)| atlas.get(None, rx, ry).is_none()), "the unwanted tiles never became resident");
    }

    /// **Cancel gate:** `cancel_all` right after `submit` drops the job's result (`Done` is a cancellation, or, if the
    /// worker finished first, a result that installs nothing: both accepted) and leaves the atlas reusable; dropping an
    /// atlas with a job in flight returns well inside the worker's bounded join (the job observed the cancel).
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_cancel_all_drops_the_job_and_the_atlas_stays_usable() {
        use exv_codec::Profile;
        use exv_video::write_video;

        let img = f16_image(2048, 2048, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let bytes = write_video(&[img], &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode");
        let s = shared(bytes.clone());
        let ctx = GpuContext::new().expect("gpu ctx");
        let tiles = tile_grid(8, 8);
        let mut atlas = new_atlas(&ctx);
        let clip = Clip { data: &bytes, idx: &s.idx };

        let p = atlas.plan(clip, 0, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
        atlas.submit(p, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit");
        atlas.cancel_all();
        let done = wait_done(&mut atlas);
        let cancelled = done.is_cancelled();
        assert_eq!(atlas.install(done).expect("install"), 0, "cancel_all: nothing lands (cancelled = {cancelled})");
        assert_eq!((atlas.decode_count(), atlas.stats().inflight), (0, 0));
        assert!(atlas.get(None, 0, 0).is_none(), "nothing resident after a cancelled job");
        assert_eq!(settle(&mut atlas, &s, 0, &tiles, Chans::All), tiles.len(), "the atlas decodes normally afterwards (warms the pipelines)");

        // Drop with a job in flight: cancels it; the worker stops at its next checkpoint, far below the join timeout.
        let mut busy = new_atlas(&ctx);
        let p = busy.plan(clip, 0, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
        busy.submit(p, Arc::clone(&s.data), Arc::clone(&s.idx)).expect("submit");
        let t = Instant::now();
        drop(busy);
        assert!(t.elapsed() < JOIN_WAIT / 2, "drop must observe the cancel, not wait out the join timeout ({:?})", t.elapsed());
    }

    /// **Failure gate:** a decode error on the worker (a flipped RAW blob under `Verify::On`) comes out of `install` as
    /// `Err` (the viewer's `present` then falls back to the CPU path), leaves nothing in flight or resident, and the
    /// same atlas decodes the clip normally once the policy allows it.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_worker_error_is_returned_by_install_and_the_atlas_stays_intact() {
        use exv_codec::Profile;
        use exv_format::{DecodeLimits, Dtype, Verify};

        let (w, h) = (256u32, 128u32);
        let n = (w * h) as usize;
        let id: Vec<u8> = (0..n as u32).flat_map(|i| (0xC0DE_0000u32 | i).to_le_bytes()).collect();
        let prof = Profile::exv_rt_dwt(16, 0);
        let img = exv_codec::image::Image::new(
            header(w, h, 128, 128, 4, prof.params(), &[("R", Dtype::F16), ("id", Dtype::U32)]),
            vec![f16_plane(w, h, 0, 0), id],
        )
        .expect("image");
        let good = exv_video::write_still(&img, &prof).expect("encode");
        let needle = (0xC0DE_0000u32 | 5).to_le_bytes();
        let at = good.windows(4).position(|c| c == needle).expect("raw id value in the container");
        let mut bad = good;
        bad[at] ^= 0x77;
        let on = shared_with(bad.clone(), DecodeLimits::default().with_verify(Verify::On));
        let off = shared_with(bad.clone(), DecodeLimits::default().with_verify(Verify::Off));
        let ctx = GpuContext::new().expect("gpu ctx");
        let tiles = [(0u32, 0u32), (1, 0)];
        let mut atlas = new_atlas(&ctx);

        let clip = Clip { data: &bad, idx: &on.idx };
        let p = atlas.plan(clip, 0, want(&tiles, Chans::All), ContentKey(1), DisplayKey(0)).expect("plan").expect("misses");
        atlas.submit(p, Arc::clone(&on.data), Arc::clone(&on.idx)).expect("submit");
        let done = wait_done(&mut atlas);
        assert!(!done.is_cancelled());
        let e = atlas.install(done).expect_err("the worker's checksum failure must surface from install");
        assert!(e.to_lowercase().contains("checksum") || e.contains("CRC"), "{e}");
        assert!(!atlas.busy());
        assert_eq!((atlas.stats().inflight, atlas.decode_count(), atlas.stats().resident_bytes), (0, 0, 0), "nothing in flight or resident");
        assert_eq!(settle(&mut atlas, &off, 0, &tiles, Chans::All), tiles.len(), "the atlas is intact: Verify::Off decodes the clip");
    }

    /// **Wake gate:** the worker fires the repaint callback exactly once per finished job.
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_wake_fires_once_per_done() {
        use exv_codec::Profile;
        use exv_video::write_video;
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

        let img = f16_image(600, 400, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let bytes = write_video(&[img], &Profile::exv_rt_dwt(16, 0), 1, 0, 0).expect("encode");
        let s = shared(bytes);
        let ctx = GpuContext::new().expect("gpu ctx");
        let count = Arc::new(AtomicUsize::new(0));
        let wake: Wake = {
            let count = Arc::clone(&count);
            Arc::new(move || {
                count.fetch_add(1, SeqCst);
            })
        };
        let mut atlas = ExvResidentAtlas::new(&ctx, 512 << 20, crate::DECODE_BUDGET).expect("atlas").with_wake(wake);
        let tiles = tile_grid(3, 2);
        // Two jobs: RGB, then the +A delta.
        settle(&mut atlas, &s, 0, &tiles, Chans::Only(&[0, 1, 2]));
        settle(&mut atlas, &s, 0, &tiles, Chans::Only(&[0, 1, 2, 3]));
        let t = Instant::now();
        while count.load(SeqCst) < 2 && t.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(1));
        }
        thread::sleep(Duration::from_millis(100));
        assert_eq!(count.load(SeqCst), 2, "one wake per Done, no more, no less");
    }
}
