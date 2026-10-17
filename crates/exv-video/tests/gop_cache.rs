//! [`GopCache`]: the shared decoded-frame cache of P/B clips.
//!
//! - bit-identical to the stateless `decode_frame` over gop x B x MV, for every access order and a channel subset;
//! - a sequential fetch decodes each frame ONCE (`decode_one` spans == frames; the stateless loop pays `O(gop^2)`),
//!   a random fetch decodes only the missing ancestors, a tiny budget still decodes each frame once sequentially;
//! - single-flight: 12 threads released by a barrier onto a slow chain decode it once (and really wait on each other);
//! - an ancestor's failure is the frame's failure, is retried (never cached) and leaves healthy GOPs alone;
//! - a frame dropped from the cache map while leased stays readable AND stays charged until its lease drops;
//! - cancel: typed, nothing cached, a cancelled leader never fails its waiters; budgets: typed `OverBudget`, no hang;
//! - clip ids never alias; a rayon pool asking one frame is correct and does not deadlock;
//! - bench (ignored): sequential fetch of a long P-GOP, old path vs cache (`--release --nocapture --ignored`).
//!
//! The span counter is a process-global subscriber (decode spans can be created on rayon workers); every test holds
//! [`serial`] so counts do not leak between tests.

mod common;

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Barrier, Mutex, Once};
use std::time::Instant;

use exv_format::bufpool::{Budget, Cancel};
use common::{clip, frame};
use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header};
use exv_video::{CacheError, Chans, DecodeError, GopCache, VideoIndex, WriteOpts, decode_frame, frame_view, write_clip};
use rayon::prelude::*;
use tracing::span;

static COUNTS: Mutex<Option<HashMap<&'static str, usize>>> = Mutex::new(None);
static LOCK: Mutex<()> = Mutex::new(());

struct Count;

impl tracing::Subscriber for Count {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attrs: &span::Attributes<'_>) -> span::Id {
        if let Ok(mut g) = COUNTS.lock() {
            *g.get_or_insert_with(HashMap::new).entry(attrs.metadata().name()).or_default() += 1;
        }
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}

fn serial() -> std::sync::MutexGuard<'static, ()> {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::subscriber::set_global_default(Count).expect("global subscriber");
    });
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `f`'s value + how many `decode_one` spans (= frames decoded) it created.
fn decodes<R>(f: impl FnOnce() -> R) -> (R, usize) {
    *COUNTS.lock().expect("counts") = Some(HashMap::new());
    let r = f();
    let c = COUNTS.lock().expect("counts").take().unwrap_or_default();
    (r, c.get("decode_one").copied().unwrap_or(0))
}

fn encode(frames: &[Image], gop: u32, b: u32, search: i32) -> (Vec<u8>, VideoIndex) {
    let opts = WriteOpts { gop, b_frames: b, search, ..WriteOpts::default() };
    let (out, _) = write_clip(Cursor::new(Vec::new()), frames, &Profile::lossless_fast(), &opts).expect("encode");
    let bytes = out.into_inner();
    let idx = VideoIndex::parse(&bytes).expect("index");
    (bytes, idx)
}

fn big() -> GopCache {
    GopCache::new(Budget::new(1 << 30))
}

/// Cache vs stateless: bit-identical planes for `chans`, fetched in `order`.
fn same(cache: &GopCache, clip_id: u64, data: &[u8], idx: &VideoIndex, order: &[usize], chans: Chans<'_>, cfg: &str) {
    for &k in order {
        let want = decode_frame(data, idx, k, chans).expect("stateless");
        let got = cache.get(clip_id, data, idx, k, chans, &Cancel::new()).expect("cached");
        for ci in 0..idx.header.channels.len() {
            assert_eq!(got.planes.get(ci), want.planes.get(ci), "{cfg}: frame {k} ch {ci}");
        }
    }
}

#[test]
fn cache_is_bit_identical_to_stateless() {
    let _s = serial();
    for (gop, b, search) in [(1u32, 0u32, 0i32), (3, 0, 0), (8, 0, 0), (8, 0, 4), (8, 2, 0), (5, 1, 4)] {
        let frames = clip(13, 16);
        let (data, idx) = encode(&frames, gop, b, search);
        let n = idx.frame_count();
        let asc: Vec<usize> = (0..n).collect();
        let desc: Vec<usize> = (0..n).rev().collect();
        let shuffled: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect();
        let cfg = format!("gop {gop} b {b} search {search}");
        for (order, name) in [(&asc, "asc"), (&desc, "desc"), (&shuffled, "shuffled")] {
            let cache = big();
            let id = cache.open_clip();
            same(&cache, id, &data, &idx, order, Chans::All, &format!("{cfg} {name} all"));
            same(&cache, id, &data, &idx, order, Chans::Only(&[2, 0]), &format!("{cfg} {name} subset"));
        }
    }
}

/// Sequential fetch of a P-GOP decodes each frame once; the stateless loop decodes `1 + 2 + .. + g` per GOP.
#[test]
fn sequential_fetch_decodes_each_frame_once() {
    let _s = serial();
    let (n, gop) = (24usize, 8u32);
    let (data, idx) = encode(&clip(n as u32, 16), gop, 0, 0);
    let (_, old) = decodes(|| (0..n).for_each(|k| drop(decode_frame(&data, &idx, k, Chans::All).expect("decode"))));
    assert_eq!(old, 3 * (8 * 9 / 2), "stateless loop re-decodes the GOP prefix per frame");
    let cache = big();
    let id = cache.open_clip();
    let (_, new) = decodes(|| {
        for k in 0..n {
            drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"));
        }
    });
    assert_eq!(new, n, "each frame decoded once");
    assert_eq!(cache.stats().loads as usize, n);
    // A second pass in any order is all hits.
    let (_, again) = decodes(|| {
        for k in (0..n).rev() {
            drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"));
        }
    });
    assert_eq!(again, 0, "re-reads are hits");
}

/// A random fetch decodes only the missing ancestors: frame 20 of a gop-24 chain costs 21, then frame 10 is free and
/// frame 22 costs 2 (21, 22; 20 is cached and ends its branch).
#[test]
fn random_fetch_decodes_only_missing_ancestors() {
    let _s = serial();
    let (data, idx) = encode(&clip(24, 16), 24, 0, 0);
    let cache = big();
    let id = cache.open_clip();
    let get = |k: usize| drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"));
    assert_eq!(decodes(|| get(20)).1, 21);
    assert_eq!(decodes(|| get(10)).1, 0);
    assert_eq!(decodes(|| get(22)).1, 2);
    // Another channel set is another entry: its chain decodes again, restricted to its channels.
    let (_, sub) = decodes(|| drop(cache.get(id, &data, &idx, 3, Chans::Only(&[1]), &Cancel::new()).expect("subset")));
    assert_eq!(sub, 4);
}

/// With room for ~3 frames a sequential fetch still decodes each frame once (the previous frame is always resident),
/// and a cold fetch deep in the GOP peaks at 2 pinned frames like the stateless path.
#[test]
fn tiny_budget_still_decodes_once() {
    let _s = serial();
    let n = 16usize;
    let (data, idx) = encode(&clip(n as u32, 16), 16, 0, 0);
    let fb = idx.header.frame_bytes().expect("frame bytes") as usize;
    let cache = GopCache::new(Budget::new(3 * fb as u64));
    let id = cache.open_clip();
    let (_, seq) = decodes(|| {
        for k in 0..n {
            drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"));
        }
    });
    assert_eq!(seq, n);
    assert!(cache.stats().peak_used <= 3 * fb, "stays inside the budget");
    let cache = GopCache::new(Budget::new(3 * fb as u64));
    let id = cache.open_clip();
    let got = cache.get(id, &data, &idx, n - 1, Chans::All, &Cancel::new()).expect("cold deep fetch");
    let want = decode_frame(&data, &idx, n - 1, Chans::All).expect("stateless");
    assert_eq!(got.planes.get(0), want.planes.get(0));
    // A budget below one frame is a typed refusal, not a hang.
    let tiny = GopCache::new(Budget::new(fb as u64 / 2));
    let r = tiny.get(tiny.open_clip(), &data, &idx, 0, Chans::All, &Cancel::new());
    assert!(matches!(r, Err(CacheError::OverBudget { .. })), "{:?}", r.err());
}

/// 12 threads released together onto a slow chain (8 frames of 512 x 512 x 3ch F16, ms per frame) decode it ONCE:
/// without single-flight every thread that planned before the leader finished would decode the frames itself.
/// `waits > 0` proves the threads really overlapped (a joiner blocked on the leader's flight), so the count is not
/// the product of sequential execution.
#[test]
fn concurrent_same_frame_decodes_once() {
    let _s = serial();
    let (n, threads) = (8usize, 12usize);
    let frames: Vec<Image> = (0..n as u32).map(|k| drift(512, k)).collect();
    let (data, idx) = encode(&frames, n as u32, 0, 0);
    let want = decode_frame(&data, &idx, n - 1, Chans::All).expect("stateless");
    let cache = big();
    let id = cache.open_clip();
    let gate = Barrier::new(threads);
    let (_, count) = decodes(|| {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    gate.wait();
                    let f = cache.get(id, &data, &idx, n - 1, Chans::All, &Cancel::new()).expect("cached");
                    for ci in 0..3 {
                        assert_eq!(f.planes.get(ci), want.planes.get(ci), "ch {ci}");
                    }
                });
            }
        });
    });
    let st = cache.stats();
    assert_eq!(count, n, "the chain decoded once for {threads} requests");
    assert_eq!(st.loads as usize, n, "one leader per frame");
    assert!(st.waits > 0, "threads must have joined a flight, else the test did not overlap: {st:?}");
}

/// An out-of-range frame / channel is a typed request error; a damaged frame fails with its own error, is not cached,
/// does not poison its (cached) ancestors, and fails its dependants too.
#[test]
fn errors_are_typed_and_not_cached() {
    let _s = serial();
    let (data, idx) = encode(&clip(8, 16), 8, 0, 0);
    let cache = big();
    let id = cache.open_clip();
    let req = |r: Result<_, CacheError>| matches!(r, Err(CacheError::Decode(DecodeError::Request(_))));
    assert!(req(cache.get(id, &data, &idx, 99, Chans::All, &Cancel::new())));
    assert!(req(cache.get(id, &data, &idx, 0, Chans::Only(&[7]), &Cancel::new())));
    let di = idx.decode_idx(3).expect("frame 3");
    let body = frame_view(&data, &idx, di).expect("frame").body;
    let mut bad = data.clone();
    bad[body.as_ptr() as usize - data.as_ptr() as usize + body.len() / 4] ^= 0x01;
    assert!(cache.get(id, &bad, &idx, 2, Chans::All, &Cancel::new()).is_ok(), "ancestors before the damage decode");
    for k in [3, 5] {
        let r = cache.get(id, &bad, &idx, k, Chans::All, &Cancel::new());
        assert!(matches!(r, Err(CacheError::Decode(_))), "frame {k}: {:?}", r.err());
    }
    assert!(!cache.has(id, &idx, 3, Chans::All), "a failure is not cached");
    assert!(cache.has(id, &idx, 2, Chans::All), "the intact ancestors stay");
    cache.forget(id);
    assert!(!cache.has(id, &idx, 2, Chans::All) && cache.stats().resident == 0, "forget drops the clip");
}

/// A frame dropped from the cache MAP (`forget`) while a caller holds its lease stays readable and bit-identical, is still
/// charged to the budget (an unreachable but live frame is real memory), a re-fetch decodes a fresh entry beside it, and
/// everything is returned once the leases drop. (An LRU eviction cannot take a pinned entry, so `forget` is the way a
/// held frame leaves the map.)
#[test]
fn lease_outlives_eviction() {
    let _s = serial();
    let (data, idx) = encode(&clip(8, 16), 8, 0, 0);
    let cache = big();
    let id = cache.open_clip();
    let want = decode_frame(&data, &idx, 1, Chans::All).expect("stateless");
    let fb = want.planes.bytes();
    let held = cache.get(id, &data, &idx, 1, Chans::All, &Cancel::new()).expect("cached");
    assert_eq!(cache.stats().resident, 2, "frames 0 and 1");
    cache.forget(id);
    let st = cache.stats();
    assert!(!cache.has(id, &idx, 1, Chans::All), "left the map");
    assert_eq!((st.resident, st.used), (0, fb), "only the leased frame is charged (frame 0 had no lease)");
    for ci in 0..3 {
        assert_eq!(held.planes.get(ci), want.planes.get(ci), "leased frame still readable, ch {ci}");
    }
    let (again, n) = decodes(|| cache.get(id, &data, &idx, 1, Chans::All, &Cancel::new()).expect("re-fetch"));
    assert_eq!(n, 2, "forgotten frames decode again (0 and 1)");
    assert!(cache.stats().used >= 3 * fb, "old lease + new frames 0 and 1 are all charged: {:?}", cache.stats());
    assert_eq!(again.planes.get(0), held.planes.get(0));
    drop((held, again));
    cache.forget(id);
    assert_eq!(cache.stats().used, 0, "released with the last lease");
}

/// Damage frame `k` (display) of `data`: one flipped bit inside its body, so its CRC (Verify On) fails.
fn corrupt(data: &[u8], idx: &VideoIndex, k: usize) -> Vec<u8> {
    let di = idx.decode_idx(k).expect("frame");
    let body = frame_view(data, idx, di).expect("frame").body;
    let mut bad = data.to_vec();
    bad[body.as_ptr() as usize - data.as_ptr() as usize + body.len() / 4] ^= 0x01;
    bad
}

/// Frame 2 of a gop-6 clip is damaged: its dependants fail with the SAME typed error on every attempt (the failure is
/// retried, never cached: exactly one more load, the intact ancestors are hits), and the healthy GOPs of the same clip
/// keep decoding bit-identically.
#[test]
fn ancestor_error_propagates_and_is_retried() {
    let _s = serial();
    let (data, idx) = encode(&clip(12, 16), 6, 0, 0);
    let bad = corrupt(&data, &idx, 2);
    let cache = big();
    let id = cache.open_clip();
    let get = |k: usize| cache.get(id, &bad, &idx, k, Chans::All, &Cancel::new());
    assert!(get(1).is_ok(), "ancestors before the damage decode");
    let Err(e1) = get(4) else { panic!("ancestor 2 is damaged") };
    assert!(matches!(e1, CacheError::Decode(_)), "{e1:?}");
    let before = cache.stats();
    let Err(e2) = get(4) else { panic!("still damaged") };
    assert_eq!(e1, e2, "same typed error on retry");
    assert_eq!(cache.stats().loads - before.loads, 1, "only the failed frame is re-attempted; 0 and 1 are hits");
    assert!(!cache.has(id, &idx, 2, Chans::All) && !cache.has(id, &idx, 4, Chans::All), "failures are not cached");
    assert!(cache.has(id, &idx, 1, Chans::All), "healthy ancestors stay");
    same(&cache, id, &bad, &idx, &[6, 11, 7, 8], Chans::All, "healthy gop of a damaged clip");
}

/// A token cancelled before the fetch is a typed `Cancelled`; nothing is decoded, cached or charged, and the cache is
/// fully usable afterwards.
#[test]
fn cancelled_fetch_is_typed_and_not_cached() {
    let _s = serial();
    let (data, idx) = encode(&clip(8, 16), 8, 0, 0);
    let cache = big();
    let id = cache.open_clip();
    let c = Cancel::new();
    c.cancel();
    let (r, n) = decodes(|| cache.get(id, &data, &idx, 5, Chans::All, &c));
    assert!(matches!(r, Err(CacheError::Cancelled)), "{:?}", r.err());
    let st = cache.stats();
    assert_eq!((n, st.loads, st.resident, st.used), (0, 0, 0, 0), "nothing decoded, cached or charged");
    same(&cache, id, &data, &idx, &[5, 7], Chans::All, "after a cancel");
}

/// A leader cancelled MID-chain never fails a waiter that was not cancelled (the pool hands the flight over): per round
/// thread A (cancelled after a varying delay) and B (never cancelled) race for the same frame. B must always succeed
/// bit-identically, A is `Ok` or `Cancelled`, and every frame is decoded exactly once (a cancel loses no completed frame
/// and decodes none twice). The delay sweep makes the cancel land at different points of the walk; the deterministic
/// take-over of a flight whose leader returned `Cancelled` is bufpool's own test (`pool.rs`).
#[test]
fn cancelled_leader_does_not_fail_waiters() {
    let _s = serial();
    let n = 10usize;
    let frames: Vec<Image> = (0..n as u32).map(|k| drift(256, k)).collect();
    let (data, idx) = encode(&frames, n as u32, 0, 0);
    let want = decode_frame(&data, &idx, n - 1, Chans::All).expect("stateless");
    for round in 0..12u64 {
        let cache = big();
        let id = cache.open_clip();
        let (ca, cb) = (Cancel::new(), Cancel::new());
        let gate = Barrier::new(3);
        let (res, count) = decodes(|| {
            std::thread::scope(|s| {
                let a = s.spawn(|| {
                    gate.wait();
                    cache.get(id, &data, &idx, n - 1, Chans::All, &ca).map(|_| ())
                });
                let b = s.spawn(|| {
                    gate.wait();
                    cache.get(id, &data, &idx, n - 1, Chans::All, &cb)
                });
                gate.wait();
                std::thread::sleep(std::time::Duration::from_micros(round * 400));
                ca.cancel();
                (a.join().expect("a"), b.join().expect("b"))
            })
        });
        let (a, b) = res;
        assert!(matches!(a, Ok(()) | Err(CacheError::Cancelled)), "round {round}: A {a:?}");
        let b = b.expect("an uncancelled waiter must succeed");
        for ci in 0..3 {
            assert_eq!(b.planes.get(ci), want.planes.get(ci), "round {round} ch {ci}");
        }
        assert_eq!(count, n, "round {round}: every frame decoded once");
    }
}

/// Budget of exactly one frame, P chain: frame 1 needs its pinned reference 0 beside itself, so it is a typed
/// `OverBudget` (not a hang, not a silent over-commit); the failed fetch leaves no pin behind and the cache keeps
/// serving what fits. Two frames are enough for a P chain of any length (peak 2 pinned), and cold deep fetches too.
#[test]
fn over_budget_pinned_reference_is_typed_and_recoverable() {
    let _s = serial();
    let (data, idx) = encode(&clip(8, 16), 8, 0, 0);
    let fb = decode_frame(&data, &idx, 0, Chans::All).expect("stateless").planes.bytes();
    let cache = GopCache::new(Budget::new(fb as u64));
    let id = cache.open_clip();
    let r = cache.get(id, &data, &idx, 1, Chans::All, &Cancel::new());
    assert!(matches!(r, Err(CacheError::OverBudget { .. })), "{:?}", r.err());
    let st = cache.stats();
    assert_eq!(st.pinned, 0, "a refused fetch leaves nothing pinned");
    assert!(st.used <= fb, "never over-committed: {st:?}");
    same(&cache, id, &data, &idx, &[0, 0], Chans::All, "frame 0 still serves");
    let two = GopCache::new(Budget::new(2 * fb as u64));
    let id = two.open_clip();
    same(&two, id, &data, &idx, &[7, 0, 3, 7], Chans::All, "2-frame budget, cold deep fetch first");
}

/// B-GOP (gop 8, 2 B-frames) under a small budget. Leaves are pinned from planning (before the walk), so a cache of 2-3
/// frames may return a typed `OverBudget` where the stateless walk would pass (D-29 known limits); it must never hang,
/// return another error, return wrong pixels, or break the cache for later fetches. Documented outcome: 3 frames (two
/// anchors + the B) serve every fetch, 2 frames serve the I/P anchors and refuse B-frames with `OverBudget`; with room
/// for the whole clip every frame decodes.
#[test]
fn bgop_small_budget_decodes_or_is_typed_over_budget() {
    let _s = serial();
    let (data, idx) = encode(&clip(13, 16), 8, 2, 0);
    let fb = decode_frame(&data, &idx, 0, Chans::All).expect("stateless").planes.bytes();
    let n = idx.frame_count();
    for frames in [2usize, 3] {
        let cache = GopCache::new(Budget::new((frames * fb) as u64));
        let id = cache.open_clip();
        let mut ok = 0;
        for k in (0..n).chain((0..n).rev()) {
            match cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()) {
                Ok(f) => {
                    let want = decode_frame(&data, &idx, k, Chans::All).expect("stateless");
                    for ci in 0..3 {
                        assert_eq!(f.planes.get(ci), want.planes.get(ci), "{frames} frames: frame {k} ch {ci}");
                    }
                    ok += 1;
                }
                Err(CacheError::OverBudget { .. }) => {}
                Err(e) => panic!("{frames}-frame budget, frame {k}: {e:?}"),
            }
            assert_eq!(cache.stats().pinned, 0, "nothing stays pinned between fetches");
        }
        // Measured: 3 frames (two anchors + the B itself) serve every fetch, 2 frames refuse the B-frames (typed).
        let all = 2 * n;
        assert!(if frames == 3 { ok == all } else { 0 < ok && ok < all }, "{frames}-frame budget: {ok} of {all} decoded");
    }
    let cache = GopCache::new(Budget::new((2 * n * fb) as u64));
    let id = cache.open_clip();
    let all: Vec<usize> = (0..n).chain((0..n).rev()).collect();
    same(&cache, id, &data, &idx, &all, Chans::All, "room for the clip");
}

/// Two clips with DIFFERENT bytes but the same geometry and frame indices never share entries (the clip id is the first
/// half of the key); forgetting one leaves the other intact and still a hit.
#[test]
fn clip_ids_never_share_entries() {
    let _s = serial();
    let (da, ia) = encode(&clip(8, 16), 8, 0, 0);
    let b_frames: Vec<Image> = (0..8).map(|s| frame(s + 40, 16)).collect();
    let (db, ib) = encode(&b_frames, 8, 0, 0);
    let cache = big();
    let (a, b) = (cache.open_clip(), cache.open_clip());
    assert_ne!(a, b);
    for k in 0..8 {
        let fa = cache.get(a, &da, &ia, k, Chans::All, &Cancel::new()).expect("a");
        let fb = cache.get(b, &db, &ib, k, Chans::All, &Cancel::new()).expect("b");
        assert_ne!(fa.planes.get(0), fb.planes.get(0), "frame {k}: the clips really differ");
        let (wa, wb) = (decode_frame(&da, &ia, k, Chans::All).expect("wa"), decode_frame(&db, &ib, k, Chans::All).expect("wb"));
        for ci in 0..3 {
            assert_eq!(fa.planes.get(ci), wa.planes.get(ci), "a frame {k} ch {ci}");
            assert_eq!(fb.planes.get(ci), wb.planes.get(ci), "b frame {k} ch {ci}");
        }
    }
    assert_eq!(cache.stats().loads, 16, "no entry served the other clip");
    cache.forget(a);
    assert!((0..8).all(|k| !cache.has(a, &ia, k, Chans::All) && cache.has(b, &ib, k, Chans::All)), "forget(a) leaves b");
    let (_, n) = decodes(|| drop(cache.get(b, &db, &ib, 7, Chans::All, &Cancel::new()).expect("b hit")));
    assert_eq!(n, 0, "b is still all hits");
}

/// Sixteen rayon tasks on an 8-thread pool fetch the SAME frame of a slow chain. Whether a worker waits on a flight or
/// (bufpool feature `rayon`) decodes privately, the result must be correct and must not deadlock; sharing may only be
/// weaker: every frame is decoded at least once and at most once per task.
#[test]
fn rayon_workers_same_frame_are_correct_and_do_not_deadlock() {
    let _s = serial();
    let (n, tasks) = (8usize, 16usize);
    let frames: Vec<Image> = (0..n as u32).map(|k| drift(512, k)).collect();
    let (data, idx) = encode(&frames, n as u32, 0, 0);
    let want = decode_frame(&data, &idx, n - 1, Chans::All).expect("stateless");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().expect("rayon pool");
    let cache = big();
    let id = cache.open_clip();
    let (same_as_want, count) = decodes(|| {
        pool.install(|| {
            (0..tasks)
                .into_par_iter()
                .map(|_| {
                    let f = cache.get(id, &data, &idx, n - 1, Chans::All, &Cancel::new()).expect("cached");
                    (0..3).all(|ci| f.planes.get(ci) == want.planes.get(ci))
                })
                .collect::<Vec<bool>>()
        })
    });
    assert!(same_as_want.iter().all(|&b| b), "every task read the stateless pixels");
    assert!((n..=tasks * n).contains(&count), "decodes {count} outside {n}..={}", tasks * n);
}

/// 1k x 1k x 3ch F16 whose every pixel drifts per frame (real P residuals, no SKIP).
fn drift(w: u32, k: u32) -> Image {
    let planes: Vec<Vec<u8>> = (0..3u32)
        .map(|c| {
            (0..w * w)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let v = ((x * 3 + y * 5 + c * 11) & 0x3ff) as u16 + (k as u16) * ((x % 5) as u16);
                    (0x3000 + v).to_le_bytes()
                })
                .collect::<Vec<u8>>()
        })
        .collect();
    let header = Header {
        width: w,
        height: w,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: ["R", "G", "B"].iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F16 }).collect(),
    };
    Image::new(header, planes).expect("image")
}

/// Sequential fetch of a gop-24 P clip (48 frames of 1k x 1k x 3ch F16): stateless `decode_frame` loop vs the cache,
/// then random access and a cold deep fetch. Run: `cargo test --release -p exv-video --test gop_cache bench -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn bench_sequential_gop() {
    let _s = serial();
    let n = 48usize;
    let frames: Vec<Image> = (0..n as u32).map(|k| drift(1024, k)).collect();
    let (data, idx) = encode(&frames, 24, 0, 0);
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let (mut t_old, mut t_new, mut c_old, mut c_new) = (Vec::new(), Vec::new(), 0, 0);
    for _ in 0..5 {
        let t0 = Instant::now();
        let (_, c) = decodes(|| (0..n).for_each(|k| drop(decode_frame(&data, &idx, k, Chans::All).expect("decode"))));
        t_old.push(t0.elapsed().as_secs_f64());
        c_old = c;
        let cache = big();
        let id = cache.open_clip();
        let t0 = Instant::now();
        let (_, c) = decodes(|| (0..n).for_each(|k| drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"))));
        t_new.push(t0.elapsed().as_secs_f64());
        c_new = c;
    }
    let (old, new) = (med(t_old), med(t_new));
    println!("sequential {n} frames gop 24: stateless {c_old} decodes {:.0} ms | cache {c_new} decodes {:.0} ms | x{:.1}", old * 1e3, new * 1e3, old / new);
    let cache = big();
    let id = cache.open_clip();
    let order: Vec<usize> = (0..n).map(|i| (i * 17 + 5) % n).collect();
    let t0 = Instant::now();
    let (_, c_rand) = decodes(|| order.iter().for_each(|&k| drop(cache.get(id, &data, &idx, k, Chans::All, &Cancel::new()).expect("cached"))));
    let t_rand = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let (_, c_rand_old) = decodes(|| order.iter().for_each(|&k| drop(decode_frame(&data, &idx, k, Chans::All).expect("decode"))));
    println!("random {n} fetches: stateless {c_rand_old} decodes {:.0} ms | cache {c_rand} decodes {:.0} ms", t0.elapsed().as_secs_f64() * 1e3, t_rand * 1e3);
}
