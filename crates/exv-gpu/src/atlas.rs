//! `atlas.rs` — the resident-tile atlas for out-of-core GPU display of an `.exv` clip.
//!
//! Gated behind `htj2k-backend` (module decl in the crate root). Holds [`ExvResidentAtlas`] (per-tile
//! byte-budgeted resident decode cache), [`Planes`] / [`TileTex`] (its per-channel atlas payload +
//! `TilePayload` impl), and the batched per-route miss decoders (`decode_htj2k`, `decode_wave`, `decode_rt`,
//! `assemble`, `upload_host`): every tile a request misses decodes as ONE batch per route.
//!
//! # Channel subsets
//! A request names the channels the caller displays ([`Chans`], the viewer's routed `LayerMap::chans()`): only those
//! are decoded, uploaded and kept resident (a 5-channel R,G,B + Z + id tile shown as RGB costs 3 planes, not 5). A
//! resident tile that lacks a requested channel is a MISS for exactly the missing channels: they are decoded and MERGED
//! into the tile's payload (the planes already resident are kept, not re-uploaded). The atlas budget charges the planes
//! actually resident ([`TilePayload::bytes`] counts present slots only).
//!
//! # Temporal clips (Tier A: INTRA + SKIP tiles)
//! With a zero MV a tile of a P-frame depends only on the CO-LOCATED tile of its reference, and a SKIP tile IS the
//! reference tile's bytes. So a tile of any frame resolves (`exv_video::TileChains`) to ONE standalone INTRA tile
//! of an ancestor frame (its `node`), and a resident tile stays valid across every frame that resolves to the same
//! node: a static background costs ZERO decodes after the first frame, whatever the GOP depth. Each resident
//! [`Planes`] carries the node it holds, so the stamp is evicted together with the pixels (no side table to go
//! stale). PRED tiles (lossless temporal delta) and MV != 0 / B-frames are typed errors, never a guess.
//!
//! # Sync and async decode
//! [`ExvResidentAtlas::request`] = `plan` (UI thread, pass 1) + the decode run inline + `install`. A viewer uses the
//! split instead: `plan` -> `submit` (one decode worker thread) -> `poll` + `install` on the wake, so a seek or zoom
//! never blocks the UI pass on a batch decode. The job / worker / bookkeeping code is `crate::atlas_job`; the
//! decoders below (`ExvDec`) are the same functions on both paths.

use crate::prelude::*;
use exv_gpu_kernels::*;
use crate::atlas_job::{Bytes, Done, Flight, JobErr, Job, Pending, Plan, Wake, Work, Worker, chunk_cap, lands, run_work, useless};
use crate::dispatch::{TileJob, decode_tiles_bc6h_gpu, decode_tiles_exvrt_gpu, decode_tiles_wave_gpu};
use bufpool::Cancel;
use exv_codec::frame::Chans;
use exv_format::Dtype;
use crate::htj2k::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-tile resident decode cache for **lossless HTJ2K** `.exv` images — the out-of-core viewer
/// path for 16k–65k images. exv stores one HTJ2K codestream PER TILE; this decodes ONLY the
/// requested (visible) tiles at a chosen resolution level into a byte-budgeted [`jph_wgpu_rs::TileAtlas`]
/// of resident textures, so the whole image is NEVER decoded — only what the viewport needs.
///
/// Each tile is a distinct, persistent texture set the atlas owns and LRU-frees under budget pressure, so
/// decode uses [`jph_wgpu_rs::ResidentCtx::decode_*_fresh`] (a fresh owned texture per tile) rather
/// than the frames-in-flight ring (which reuses/overwrites — wrong here, it would clobber a still-
/// resident tile). The expensive GPU objects (pipelines/pool/plane buffers) are reused across tiles.
///
/// Hold one per viewer session. Drive it from the viewport: compute the visible tile grid + LOD from
/// pan/zoom and the routed channel set, call [`ExvResidentAtlas::request`], then sample [`ExvResidentAtlas::get`] per tile.
///
/// One resident tile is a [`Planes`]: one optional single-channel plane texture per header channel ([`TileTex::Planes`],
/// absent = not requested) or, for the GPU-decoded <=4-channel HTJ2K tile, ONE packed RGBA16F texture ([`TileTex::Packed`]).
#[cfg(feature = "htj2k-backend")]
pub struct ExvResidentAtlas {
    atlas: jph_wgpu_rs::TileAtlas<Planes>,
    /// A CLONE of the caller's context (Arc-shared pipelines — no second compile): the device gate (`check_fits`) and
    /// the worker's decoder read it.
    ctx: GpuContext,
    /// The INLINE executor of [`Self::request`] (the synchronous path); the async path runs its own on the worker.
    dec: ExvDec,
    decode_budget: u64,
    decodes: u64,
    reused: u64,
    /// Plane textures made resident since creation ([`AtlasStats::planes`]).
    planes: u64,
    /// Host bytes written into upload staging since creation ([`AtlasStats::uploaded_bytes`]); atomic + shared with the
    /// decoders (inline and worker) because the upload helper runs behind `&self` on whichever thread decodes.
    uploaded: Arc<AtomicU64>,
    /// `(content key, channel set)` that already passed `check_fits` (the device query runs once per clip + channel set, not per frame).
    fits_checked: Option<(u64, Vec<usize>)>,
    /// Force the jph-wgpu GPU tile decode for full-resolution HTJ2K (see [`Self::htj2k_on_gpu`]).
    htj2k_gpu: bool,
    /// In-flight / wanted bookkeeping of the async path (UI thread only, see [`crate::atlas_job`]).
    flight: Flight,
    /// The worker's repaint callback (fired after each finished job).
    wake: Option<Wake>,
    /// The decode worker, started by the first [`Self::submit`].
    worker: Option<Worker>,
    /// The job the worker is running now (one at a time; its [`Done`] is still to be polled).
    running: Option<Running>,
    /// What the last miss-path [`Self::plan`] saw while the worker was busy (see [`PlanSig`]).
    sig: Option<PlanSig>,
}

/// The inputs of a miss-path [`ExvResidentAtlas::plan`] that ended with `Ok(None)` because the worker was busy. While
/// the SAME signature repeats (the viewer redraws on every UI event while a job decodes) pass 1 and the chain walk
/// would recompute the identical answer, so the repeat only re-runs its cheap observable tail. Valid only until the
/// next `submit` / `install` / `cancel_all` or any other plan (they change what pass 1 reads or writes: the in-flight
/// set, the resident tiles, the wanted set), each of which clears it.
#[cfg(feature = "htj2k-backend")]
struct PlanSig {
    seen: u32,
    level: Option<u32>,
    want: Vec<usize>,
    keys: Vec<jph_wgpu_rs::RegionKey>,
    content: jph_wgpu_rs::ContentKey,
    display: jph_wgpu_rs::DisplayKey,
    epoch: u64,
}

/// The job the worker holds: its generation, token and the tiles it decodes (the usefulness test of
/// [`ExvResidentAtlas::prune`]).
#[cfg(feature = "htj2k-backend")]
struct Running {
    jid: u64,
    epoch: u64,
    cancel: Cancel,
    keys: Vec<jph_wgpu_rs::RegionKey>,
}

/// The decoders of the resident path, free of the atlas' UI-thread state: the device context, the upload counter and
/// (lazily) the jph-wgpu decode state. The atlas owns one for the inline `request`; the worker thread owns another.
#[cfg(feature = "htj2k-backend")]
pub(crate) struct ExvDec {
    /// exv-gpu decode context: the **Track A wave/reorder/EXV-RT per-channel GPU path** and every upload.
    ctx: GpuContext,
    /// Shared with the atlas ([`AtlasStats::uploaded_bytes`]).
    uploaded: Arc<AtomicU64>,
    /// jph-wgpu decode state of the forced-GPU / LOD HTJ2K route (`&mut` per decode), built on first use: the default
    /// CPU route never needs it.
    htj: Option<(jph_wgpu_rs::Gpu, jph_wgpu_rs::ResidentCtx)>,
    /// Per-decode jph entropy-pool VRAM bound; also the soft cap of one host-upload staging run.
    decode_budget: u64,
}

/// Test probe: number of times `request` actually opened a frame of the container (the miss path). The resident-hit
/// fast path must leave it untouched ("no re-parse per call").
#[cfg(test)]
pub(crate) static FRAME_OPENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A clip's container bytes + its frame index, as handed to [`ExvResidentAtlas::request`]. The index is parsed ONCE by
/// the caller at open (`VideoIndex::parse(data)`, O(frames)); this borrowed pair is then built per call for free. A viewer
/// calls `request` on every redraw, and the index (plus the frame's tile index) used to be re-parsed each time.
#[cfg(feature = "htj2k-backend")]
#[derive(Clone, Copy)]
pub struct Clip<'a> {
    /// The whole `.exv` container (e.g. the held mmap).
    pub data: &'a [u8],
    /// `VideoIndex::parse(data)` of exactly this `data`.
    pub idx: &'a exv_video::VideoIndex,
}

/// What one [`ExvResidentAtlas::request`] makes resident: the tiles under the viewport, their resolution level and the
/// channels the caller displays.
#[cfg(feature = "htj2k-backend")]
#[derive(Clone, Copy, Debug)]
pub struct Want<'a> {
    /// Tile grid coords `(rx, ry)`.
    pub tiles: &'a [(u32, u32)],
    /// `None` = full-resolution tiles; `Some(l)` = J2K resolution level `l` (lossless HTJ2K only).
    pub level: Option<u32>,
    /// The channels to hold (the viewer's routed `LayerMap::chans()`); [`Chans::All`] = every channel. Only these are
    /// decoded, uploaded and charged to the budget.
    pub chans: Chans<'a>,
}

/// Resident-VRAM and decode counters of an [`ExvResidentAtlas`] (one value for the viewer's status bar).
/// `budget_bytes == 0` means eviction is disabled (unbounded atlas).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasStats {
    /// Bytes of resident tile textures right now (present planes only).
    pub resident_bytes: usize,
    /// The LRU byte budget the atlas evicts down to.
    pub budget_bytes: usize,
    /// Tile decodes performed since creation (cache hits excluded; a tile completed with missing channels counts once).
    pub decodes: u64,
    /// Tiles carried over to a new frame WITHOUT a decode (the resident tile already held the INTRA node the new
    /// frame's SKIP chain resolves to).
    pub reused: u64,
    /// Plane textures made resident since creation (one per decoded channel; a packed RGBA16F tile counts once). With
    /// channel subsets this is the per-plane work counter: a request for one more channel adds exactly one per tile.
    pub planes: u64,
    /// Host-decoded bytes written into upload staging since creation (row pitch padding included): the host copy
    /// traffic of the CPU-routed planes (GPU-decoded planes never cross the bus).
    pub uploaded_bytes: u64,
    /// Tiles an async job is decoding right now (`0` on the synchronous path).
    pub inflight: usize,
    /// Idle VRAM held by the plane-texture pool ([`PlaneTex`]) beside `resident_bytes`: evicted planes kept for the next
    /// tile of the same size, capped at the device's pool budget and freed with the atlas.
    pub pooled_bytes: u64,
}

impl std::fmt::Display for AtlasStats {
    /// `"atlas 412.3 MiB / 1024.0 MiB, 38 decodes, 120 reused, 114 planes, 96.0 MiB up"` (`"unbounded"` when the budget is 0),
    /// plus `", 12 in flight"` while an async job decodes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MIB: f64 = (1u64 << 20) as f64;
        let used = self.resident_bytes as f64 / MIB;
        if self.budget_bytes == 0 {
            write!(f, "atlas {used:.1} MiB / unbounded")?;
        } else {
            write!(f, "atlas {used:.1} MiB / {:.1} MiB", self.budget_bytes as f64 / MIB)?;
        }
        write!(
            f,
            ", {} decodes, {} reused, {} planes, {:.1} MiB up",
            self.decodes,
            self.reused,
            self.planes,
            self.uploaded_bytes as f64 / MIB
        )?;
        if self.inflight > 0 {
            write!(f, ", {} in flight", self.inflight)?;
        }
        if self.pooled_bytes > 0 {
            write!(f, ", {:.1} MiB pooled", self.pooled_bytes as f64 / MIB)?;
        }
        Ok(())
    }
}

/// A resident texture that goes back to the device's plane-texture pool when it dies ([`GpuContext::plane_tex`] /
/// [`Pipelines::recycle_tex`]): THE single place a plane texture's lifetime ends, so every way a tile dies recycles it with
/// no explicit release at each site: LRU eviction and `clear` by the `TileAtlas`, a merge replacing a slot, a MERGE /
/// stale result dropped by `install`, a half-built batch dropped by an error or a cancel. Derefs to the
/// [`ResidentTexture`] (`tex`, `view`, `bytes`) so consumers read it as before.
///
/// **Safe because:** the guard is the texture's only owner (consumers borrow `&PlaneTex`; compose builds its bind groups
/// per call), and a recycled texture is rewritten in full by its next taker before anything samples it, on the same
/// queue (wgpu orders the new writes after every earlier read). A texture from outside the pool (the jph-wgpu planes,
/// created with that crate's own usage flags) is [`Self::plain`] and simply frees.
#[cfg(feature = "htj2k-backend")]
pub struct PlaneTex {
    res: ResidentTexture,
    pool: Option<Arc<Pipelines>>,
}

#[cfg(feature = "htj2k-backend")]
impl PlaneTex {
    /// A plane made by `ctx` (blit or host upload): recycled into `ctx`'s pool on drop.
    pub fn new(ctx: &GpuContext, res: ResidentTexture) -> Self {
        Self { res, pool: Some(Arc::clone(&ctx.pipelines)) }
    }

    /// A texture the pool must not keep (not created through [`GpuContext::plane_tex`]).
    pub fn plain(res: ResidentTexture) -> Self {
        Self { res, pool: None }
    }
}

#[cfg(feature = "htj2k-backend")]
impl std::ops::Deref for PlaneTex {
    type Target = ResidentTexture;
    fn deref(&self) -> &ResidentTexture {
        &self.res
    }
}

#[cfg(feature = "htj2k-backend")]
impl Drop for PlaneTex {
    fn drop(&mut self) {
        if let Some(p) = &self.pool {
            // The pool receives a handle clone while `res` (view first, then its own handle) is dropped right after
            // this returns: nothing samples either, so the texture has no other user.
            p.recycle_tex(self.res.tex.clone());
        }
    }
}

/// The textures of one resident tile.
#[cfg(feature = "htj2k-backend")]
pub enum TileTex {
    /// One slot per header channel (slot index == channel index == `header().channels` order), each an owned
    /// single-channel plane texture (format by dtype). `None` = that channel is NOT resident: it was never requested,
    /// so it was neither decoded nor uploaded (never a placeholder pretending to be data).
    Planes(Vec<Option<PlaneTex>>),
    /// The <=4-channel, no-F32/U32 HTJ2K tile decoded by jph-wgpu (forced GPU route / LOD): ONE `Rgba16Float` texture
    /// whose components ARE all the channels in header order (jph-wgpu decodes the whole codestream, so a packed tile
    /// always holds every channel).
    Packed(PlaneTex),
}

#[cfg(feature = "htj2k-backend")]
impl TileTex {
    /// Is channel `ci` resident?
    pub fn has(&self, ci: usize) -> bool {
        match self {
            TileTex::Planes(s) => s.get(ci).is_some_and(Option::is_some),
            TileTex::Packed(_) => true,
        }
    }

    /// Number of resident textures (a packed tile = 1).
    pub fn count(&self) -> usize {
        match self {
            TileTex::Planes(s) => s.iter().flatten().count(),
            TileTex::Packed(_) => 1,
        }
    }

    /// `self` (the resident tile) completed with `new` (planes decoded for the SAME node): a slot `new` holds
    /// replaces / fills the old one, every other old plane is kept. A packed result holds every channel and wins.
    fn merge(self, new: TileTex) -> TileTex {
        match (self, new) {
            (TileTex::Planes(mut old), TileTex::Planes(new)) => {
                merge_slots(&mut old, new);
                TileTex::Planes(old)
            }
            (_, new @ TileTex::Packed(_)) => new,
            // A packed tile already holds every channel, so it is never a channel MISS (no merge reaches here).
            (old @ TileTex::Packed(_), TileTex::Planes(_)) => old,
        }
    }
}

/// Fill / replace the slots of `old` that `new` holds (growing `old` to `new`'s length); every other old slot is kept.
fn merge_slots<T>(old: &mut Vec<Option<T>>, new: Vec<Option<T>>) {
    if old.len() < new.len() {
        old.resize_with(new.len(), || None);
    }
    for (o, n) in old.iter_mut().zip(new) {
        if n.is_some() {
            *o = n;
        }
    }
}

/// The channels a request wants, as a sorted, de-duplicated list of header channel indices. An out-of-range index
/// or an empty set is a typed error (a caller routing bug, never silently ignored).
fn want_list(chans: Chans<'_>, nch: usize) -> Result<Vec<usize>, String> {
    let mut want: Vec<usize> = match chans {
        Chans::All => (0..nch).collect(),
        Chans::Only(list) => list.to_vec(),
    };
    want.sort_unstable();
    want.dedup();
    if let Some(&ci) = want.iter().find(|&&ci| ci >= nch) {
        return Err(format!("resident atlas: channel {ci} requested, the clip has {nch}"));
    }
    if want.is_empty() {
        return Err("resident atlas: empty channel set requested".to_string());
    }
    Ok(want)
}

/// One resident tile's decoded output ([`TileTex`]) + its temporal stamps. Uniform payload type → the atlas is
/// `TileAtlas<Planes>`.
///
/// [`Planes`] is a [`wgpu_tile_atlas::TilePayload`] whose byte cost is the sum of its PRESENT planes, so the
/// atlas LRU budget reasons about the true (format-aware) VRAM footprint of every resident tile.
#[cfg(feature = "htj2k-backend")]
pub struct Planes {
    /// The resident textures.
    pub tex: TileTex,
    /// Decode index of the INTRA frame these pixels were decoded from (the tile's temporal chain base).
    pub node: u32,
    /// Decode index of the frame this tile was last validated for (the frame it was decoded for, or a later one whose
    /// SKIP chain resolved to the same `node`). A request for the same frame is a hit without opening the container.
    pub seen: u32,
}

#[cfg(feature = "htj2k-backend")]
impl Planes {
    /// Is channel `ci` resident?
    pub fn has(&self, ci: usize) -> bool {
        self.tex.has(ci)
    }

    /// Does the tile hold EVERY channel of `chans`? (The compose gate: a tile that lacks a mapped channel must not be
    /// composed, see `PlaneCompose`.)
    pub fn has_all(&self, chans: &[usize]) -> bool {
        chans.iter().all(|&ci| self.has(ci))
    }

    /// The single-channel plane texture of channel `ci`; `None` when absent or when the tile is [`TileTex::Packed`].
    pub fn plane(&self, ci: usize) -> Option<&ResidentTexture> {
        match &self.tex {
            TileTex::Planes(s) => s.get(ci)?.as_deref(),
            TileTex::Packed(_) => None,
        }
    }
}

#[cfg(feature = "htj2k-backend")]
impl wgpu_tile_atlas::TilePayload for Planes {
    fn bytes(&self) -> usize {
        match &self.tex {
            TileTex::Planes(s) => s.iter().flatten().map(|t| t.bytes).sum(),
            TileTex::Packed(t) => t.bytes,
        }
    }
}

#[cfg(feature = "htj2k-backend")]
impl ExvResidentAtlas {
    /// Build over exv-gpu's shared device. `memory_budget` bounds resident-texture VRAM (the atlas
    /// LRU-evicts non-visible tiles to stay under it; `0` disables eviction); `decode_budget` is the
    /// per-tile entropy-pool VRAM bound forwarded to the decoder (peak VRAM per decode).
    pub fn new(ctx: &GpuContext, memory_budget: usize, decode_budget: u64) -> Result<Self, String> {
        let uploaded = Arc::new(AtomicU64::new(0));
        Ok(Self {
            atlas: jph_wgpu_rs::TileAtlas::new(memory_budget),
            ctx: ctx.clone(),
            dec: ExvDec::new(ctx.clone(), Arc::clone(&uploaded), decode_budget),
            decode_budget,
            decodes: 0,
            reused: 0,
            planes: 0,
            uploaded,
            fits_checked: None,
            htj2k_gpu: false,
            flight: Flight::default(),
            wake: None,
            worker: None,
            running: None,
            sig: None,
        })
    }

    /// Fire `wake` (from the worker thread) whenever an async decode finishes: the viewer posts its repaint event, so
    /// the next frame can `poll` + `install` without any polling timer. Only the async path ([`Self::submit`]) uses it.
    pub fn with_wake(mut self, wake: Wake) -> Self {
        self.wake = Some(wake);
        self
    }

    /// Force the jph-wgpu GPU decode for full-resolution lossless-HTJ2K tiles (parity tests, benchmarks, or keeping the
    /// CPU free). The default is the measured routing rule, [`Self::htj2k_on_gpu`].
    pub fn with_htj2k_gpu(mut self, on: bool) -> Self {
        self.htj2k_gpu = on;
        self
    }

    /// THE routing rule for lossless-HTJ2K tiles: `true` = jph-wgpu on the GPU, `false` = CPU decode + plane upload.
    /// Numbers behind it: [`Self::decode_htj2k`] (the GPU path loses at every measured tile size, so only an explicit
    /// [`Self::with_htj2k_gpu`] or a coarser LOD request takes it). A codestream holding a component jph-wgpu cannot
    /// decode ([`htj2k_gpu_ok`]) always takes the CPU route, whatever channels are requested ([`Self::supports`] refuses
    /// LOD for it, so a LOD request never lands here with such a clip).
    fn htj2k_on_gpu(&self, level: Option<u32>, dtypes: &[Dtype]) -> bool {
        (self.htj2k_gpu || level.is_some()) && htj2k_gpu_ok(dtypes)
    }

    /// THE capability gate of the resident path: can tiles of a clip coded with `profile`, holding the channels
    /// `dtypes` (header order), be made resident at `level` (`None` = full resolution)? [`Self::request`] runs it on every
    /// miss; a host runs it BEFORE starting a resident session, so it never starts one it cannot finish.
    ///
    /// Every dtype has a full-resolution route: a GPU decode where a kernel and an exact plane blit exist
    /// ([`blit_format`]), else a CPU decode + host upload ([`HostFmt`]: BF16/U8/U16 widen to `R32Float` through
    /// `exv_codec::image::sample_f32`, the CPU display value). Refused, with the reason: a codec with no resident route
    /// ([`Route::of`]), a LOD on a wave/EXV-RT clip (no resolution levels: full-res planes under a LOD key = wrong
    /// geometry), a LOD on an HTJ2K clip with a component jph-wgpu cannot decode (the LOD decode IS jph-wgpu).
    pub fn supports(profile: &exv_codec::Profile, dtypes: &[Dtype], level: Option<u32>) -> Result<(), String> {
        let route = Route::of(profile)?;
        if level.is_some() {
            if !matches!(route, Route::Htj2k) {
                return Err("resident atlas: LOD (`level`) is only available for lossless HTJ2K; wave/EXV-RT tiles are full-res".to_string());
            }
            if let Some(d) = dtypes.iter().find(|&&d| !htj2k_gpu_ok(&[d])) {
                return Err(format!("resident atlas: LOD (`level`) is not available for HTJ2K tiles with {d:?} channels (CPU-decode only)"));
            }
        }
        Ok(())
    }

    /// Make the tiles of `want` (grid coords `(rx, ry)`) resident for DISPLAY frame `frame` of a clip ([`Clip`]: its
    /// bytes + the index the caller parsed ONCE; `0` == a 1-frame still), holding at least the channels `want.chans` (the
    /// caller's routed channel set; [`Chans::All`] = every channel). Each tile resolves through its temporal chain
    /// (`exv_video::TileChains`, module docs) to the standalone INTRA tile it equals, decoding it only when the resident
    /// tile does not already hold that node; a resident tile of the right node that lacks some of `chans` decodes ONLY
    /// those channels and merges them in. When every requested tile is already validated for this frame and holds every
    /// requested channel this returns without reading the container at all (the clip is only touched on the miss path).
    ///
    /// `want.level`: `None` = full-resolution tile; `Some(l)` = J2K resolution level `l` (coarser LOD for
    /// zoomed-out views). Decodes ONLY the requested tiles **not already resident**; the atlas's byte budget drives
    /// LRU eviction of tiles that have scrolled off. `content`/`display` are the atlas invalidation keys (a changed
    /// `content` fingerprint — a different clip — clears the atlas); it must be STABLE across the frames of one clip,
    /// the frame identity lives in each tile's `node` stamp. Errors (not a GPU codec, OOB/corrupt tile, a PRED / MV /
    /// B dependency, a channel index out of range, a clip the device cannot hold, a visible set over the atlas budget)
    /// surface as `Err` and leave already-resident tiles intact — never a silent wrong-frame decode.
    ///
    /// SYNCHRONOUS: `plan` + the decode run INLINE on this thread + `install`, the same three steps the async path
    /// ([`Self::submit`] / [`Self::poll`] / [`Self::install`]) splits across threads. It must not be mixed with an
    /// async job in flight (an `Err`).
    pub fn request(
        &mut self,
        clip: Clip<'_>,
        frame: usize,
        want: Want<'_>,
        content: jph_wgpu_rs::ContentKey,
        display: jph_wgpu_rs::DisplayKey,
    ) -> Result<(), String> {
        if self.running.is_some() {
            return Err("resident atlas: request() while an async job is in flight (cancel_all, then poll it out)".to_string());
        }
        let keys = region_keys(want.tiles, want.level);
        let Some(Plan(mut work)) = self.plan(clip, frame, want, content, display)? else {
            return Ok(());
        };
        work.jid = self.flight.next_id();
        self.register(&work);
        let done = run_work(&mut self.dec, clip, work, &Cancel::new());
        self.install(done)?;
        self.check_visible(&keys)
    }

    /// PASS 1 of a request, on the UI thread, host-cheap (no decode, no GPU work): update the viewport, stamp the hits,
    /// resolve every other tile's INTRA node and return the tiles that must be decoded as a [`Plan`] (feed it to
    /// [`Self::submit`], or let [`Self::request`] run it inline). Also records what the frame wants resident (the
    /// `install` rules read it) and cancels a running job that no longer serves it.
    ///
    /// `Ok(None)` = nothing to start: everything wanted is resident, already in flight (planned for this same frame and
    /// channels, so the container is not even opened: the per-redraw case) or the worker is busy (the remaining misses
    /// are re-planned when its [`Done`] arrives; the latest viewport always wins). Errors are those of `request`; a
    /// visible set over the atlas budget counts the estimated bytes of the tiles in flight too.
    pub fn plan(
        &mut self,
        clip: Clip<'_>,
        frame: usize,
        want: Want<'_>,
        content: jph_wgpu_rs::ContentKey,
        display: jph_wgpu_rs::DisplayKey,
    ) -> Result<Option<Plan>, String> {
        use jph_wgpu_rs::{RegionKey, TileKeys};
        let Want { tiles, level, chans } = want;

        let di = clip
            .idx
            .decode_idx(frame)
            .ok_or_else(|| format!("frame {frame} out of range (the clip has {})", clip.idx.frame_count()))?;
        let seen = u32::try_from(di).map_err(|_| "frame index exceeds u32".to_string())?;
        let header = &clip.idx.header;
        let want = want_list(chans, header.channels.len())?;

        // Resident-hit fast path: update the viewport, then return BEFORE touching the container at all when every
        // requested tile is already validated for this frame and holds every wanted channel — or is in flight for
        // exactly this frame and channels (the per-redraw case) — no frame open, no tile-index parse.
        let lvl_key = level.unwrap_or(u32::MAX);
        let keys = TileKeys { content, display };
        let region_keys = region_keys(tiles, level);
        self.atlas.update_viewport(&region_keys, &keys);
        let hit = |p: &Planes| p.seen == seen && p.has_all(&want);
        if region_keys.iter().all(|k| self.atlas.get(k).is_some_and(hit) || self.flight.covers_seen(k, seen, &want)) {
            self.sig = None; // this branch rewrites `wanted` below
            if self.running.is_some() || self.flight.len() > 0 {
                // Only a live job reads `wanted`: the pure-hit redraw allocates nothing.
                let wanted = region_keys
                    .iter()
                    .filter_map(|&k| {
                        let node = self.atlas.get(&k).filter(|p| hit(p)).map(|p| p.node).or_else(|| self.flight.node_of(&k))?;
                        Some((k, (node, want.clone())))
                    })
                    .collect();
                self.flight.set_wanted(wanted);
                self.prune();
            }
            self.check_visible(&region_keys)?;
            return Ok(None);
        }

        // The same signature as the last busy plan: pass 1 would recompute the same answer (nothing it reads or writes
        // changed, see `PlanSig`), so only its observable tail runs: `wanted` is still the one that plan stored.
        let epoch = self.atlas.epoch();
        let again = self.running.is_some()
            && self.sig.as_ref().is_some_and(|s| {
                (s.seen, s.level, s.epoch, s.content, s.display) == (seen, level, epoch, content, keys.display)
                    && s.want == want
                    && s.keys == region_keys
            });
        if again {
            self.prune();
            self.check_visible(&region_keys)?;
            return Ok(None);
        }
        // An error below must not leave a stale signature that would swallow it on the next call.
        self.sig = None;

        // Miss path. The walker opens frames lazily: the frame itself (geometry + codec) plus the ancestors of its SKIP tiles.
        #[cfg(test)]
        FRAME_OPENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut walk = exv_video::TileChains::new(clip.data, clip.idx);
        let profile = walk
            .reader(di)
            .map_err(|e| format!("frame {frame}: {e}"))?
            .profile()
            .map_err(|e| e.to_string())?;
        let dtypes: Vec<Dtype> = header.channels.iter().map(|c| c.dtype).collect();
        // THE capability gate (codec route, LOD rules), shared with the hosts that decide whether to start a session.
        Self::supports(&profile, &dtypes, level)?;
        let route = Route::of(&profile)?;
        let (tiles_x, tiles_y) = (header.tiles_x(), header.tiles_y());
        // The channels a tile decode materializes: jph-wgpu decodes the whole codestream (every component), every other
        // route only the requested channels.
        let gpu_htj2k = matches!(route, Route::Htj2k) && self.htj2k_on_gpu(level, &dtypes);
        let made: Vec<Dtype> = if gpu_htj2k { dtypes.clone() } else { want.iter().map(|&ci| dtypes[ci]).collect() };
        // The device must hold this clip's tiles of the materialized channels as they are RESIDENT (a host-widened plane
        // costs 4 B/px), a typed `Limit` up front, not a mid-decode failure.
        let fits_key = (content.0, want.clone());
        if self.fits_checked.as_ref() != Some(&fits_key) {
            let resident: Vec<Dtype> = made.iter().map(|&d| HostFmt::of(d).resident_dtype(d)).collect();
            self.ctx.check_fits(&header.grid(), &resident).map_err(|e| e.to_string())?;
            self.fits_checked = Some(fits_key);
        }
        let limits = clip.idx.limits();
        let blits = matches!(route, Route::Rt | Route::Bc6h) || profile.gpu_kind() == Some(exv_codec::GpuKind::RansWave);
        let blit_f16 = match self.ctx.blit_fmt(Dtype::F16) {
            Some(f) if blits => f.block_copy_size(None).map_or(4, |b| b as usize),
            _ => 0,
        };
        let est = Est { gpu_htj2k, blit_f16 };

        // Pass 1 (host only, cheap): validate every tile, resolve the INTRA tile it equals in this frame and keep the
        // MISSES. A resident tile holding that node and every wanted channel is a HIT (a SKIP chain keeps it valid across
        // frames); one holding the node but lacking channels is a MERGE miss for exactly those; anything else is
        // stale-by-stamp and is decoded + replaced. A tile listed twice is decoded once; a tile an in-flight job already
        // delivers (same node, wanted channels covered) is not queued again.
        let mut misses: Vec<Miss> = Vec::new();
        let mut queued = std::collections::HashSet::new();
        let mut wanted = std::collections::HashMap::new();
        for &(rx, ry) in tiles {
            let key = RegionKey { level: lvl_key, rx, ry };
            // `rx == tiles_x` would alias tile (0, ry+1) and cache the wrong pixels under (rx, ry).
            if rx >= tiles_x || ry >= tiles_y {
                return Err(format!("tile ({rx},{ry}) outside the {tiles_x}x{tiles_y} tile grid"));
            }
            let t = ry
                .checked_mul(tiles_x)
                .and_then(|v| v.checked_add(rx))
                .ok_or_else(|| "tile index overflow".to_string())?;
            let chain = walk.chain(di, t).map_err(|e| format!("frame {frame} tile ({rx},{ry}): {e}"))?;
            if !chain.preds.is_empty() {
                return Err(format!(
                    "frame {frame} tile ({rx},{ry}): {} PRED residual step(s) over INTRA frame {}: temporal class Pred \
                     has no GPU residual add (decode on the CPU path)",
                    chain.preds.len(),
                    chain.base
                ));
            }
            let node = u32::try_from(chain.base).map_err(|_| "frame index exceeds u32".to_string())?;
            wanted.insert(key, (node, want.clone()));
            let (chans, merge) = match self.atlas.get_mut(&key) {
                Some(p) if p.node == node => {
                    let missing: Vec<usize> = want.iter().copied().filter(|&ci| !p.has(ci)).collect();
                    if missing.is_empty() {
                        if p.seen != seen {
                            p.seen = seen;
                            self.reused += 1;
                        }
                        continue; // resident HIT — do not re-decode.
                    }
                    (missing, true)
                }
                _ => (want.clone(), false),
            };
            if self.flight.covers(&key, node, &want) {
                continue; // already on its way
            }
            if queued.insert(key) {
                let (_, _, tw, th) = header.tile_rect(rx, ry);
                let bytes = est.tile(tw, th, &chans, &dtypes);
                misses.push(Miss { key, t, node, chans, merge, est: bytes });
            }
        }
        self.flight.set_wanted(wanted);
        self.prune();

        if misses.is_empty() || self.running.is_some() {
            if self.running.is_some() {
                self.sig = Some(PlanSig { seen, level, want, keys: region_keys.clone(), content, display: keys.display, epoch });
            }
            self.check_visible(&region_keys)?;
            return Ok(None);
        }
        Ok(Some(Plan(Work { jid: 0, profile, seen, level, route, gpu_htj2k, dtypes, misses, epoch, limits, want })))
    }

    /// Hand a [`Plan`] to the decode worker (started on first use). `data` / `idx` = the clip the plan was made over,
    /// owned so the job outlives the caller's borrow (`Arc<fsio::Src>` and the clip's `Arc<VideoIndex>`). One job runs
    /// at a time; only the first `decode pool / tile bytes` tiles of an oversized plan are taken (the textures a job
    /// holds before `install` are transient VRAM charged to the decode pool) and the rest is re-planned after the
    /// result lands. The worker fires the [`Wake`] when it finishes.
    ///
    /// # Errors
    /// A job already in flight (poll it first) or a worker that cannot start.
    pub fn submit(&mut self, plan: Plan, data: Bytes, idx: Arc<exv_video::VideoIndex>) -> Result<(), String> {
        if self.running.is_some() {
            return Err("resident atlas: submit() while a job is in flight".to_string());
        }
        if self.worker.is_none() {
            self.worker = Some(Worker::spawn(self.ctx.clone(), Arc::clone(&self.uploaded), self.decode_budget, self.wake.clone())?);
        }
        let Plan(mut work) = plan;
        self.sig = None;
        let per = work.misses.iter().map(|m| m.est).max().unwrap_or(0);
        work.misses.truncate(chunk_cap(self.decode_budget, per));
        work.jid = self.flight.next_id();
        self.register(&work);
        let cancel = Cancel::new();
        let running = Running {
            jid: work.jid,
            epoch: work.epoch,
            cancel: cancel.clone(),
            keys: work.misses.iter().map(|m| m.key).collect(),
        };
        let jid = work.jid;
        let sent = match self.worker.as_ref() {
            Some(w) => w.send(Job { data, idx, work, cancel }),
            None => Err("resident decode worker is not running".to_string()),
        };
        if let Err(e) = sent {
            self.flight.finish(jid);
            self.worker = None; // a dead worker is rebuilt by the next submit
            return Err(e);
        }
        self.running = Some(running);
        Ok(())
    }

    /// The worker's finished job, if any (non-blocking; call every frame). Feed it to [`Self::install`]. A worker that
    /// died mid-job yields a failed [`Done`] (and is rebuilt by the next [`Self::submit`]).
    pub fn poll(&mut self) -> Option<Done> {
        let polled = self.worker.as_ref()?.poll();
        match polled {
            Ok(Some(d)) => {
                if self.running.as_ref().is_some_and(|r| r.jid == d.jid) {
                    self.running = None;
                }
                if matches!(d.res, Err(JobErr::Panicked(_))) {
                    self.worker = None; // the thread is ending (see `JobErr::Panicked`); the next `submit` starts a fresh one
                }
                Some(d)
            }
            Ok(None) => None,
            Err(e) => {
                self.worker = None;
                let r = self.running.take()?;
                Some(Done { jid: r.jid, epoch: r.epoch, seen: 0, res: Err(JobErr::Failed(e)) })
            }
        }
    }

    /// Make a finished job's tiles resident (UI thread). Returns how many tiles landed. The result of a job that was
    /// cancelled or superseded is dropped (`Ok(0)`), and so is any tile that:
    /// - no longer belongs to the job (`cancel_all`, or a newer job took the key) or is no longer wanted at all (the
    ///   view moved off it; its textures are freed here). A tile of an OLDER node for a still-visible key lands anyway
    ///   (stamped with the node it carries; the next `plan` re-requests the current one), so a decode slower than the
    ///   frame period of a playing / scrubbed clip still shows something;
    /// - is a MERGE result (only the channels the tile was missing) whose resident tile is absent or holds another node:
    ///   the delta alone would be a partial tile and make `compose` fail, so it is dropped and the next `plan`
    ///   re-requests the full set;
    /// - was decoded under an older atlas epoch (the clip changed).
    ///
    /// Everything else is stamped with the frame the job was planned for and inserted exactly like the synchronous
    /// path does. Eviction down to the budget happens at the next `plan` (`update_viewport`).
    ///
    /// # Errors
    /// The decode / device failure of a job that is still live.
    pub fn install(&mut self, done: Done) -> Result<usize, String> {
        let Done { jid, epoch, seen, res } = done;
        self.sig = None;
        let live = self.flight.live(jid);
        let tiles = match res {
            Ok(t) => t,
            Err(JobErr::Cancelled) => {
                self.flight.finish(jid);
                return Ok(0);
            }
            Err(JobErr::Failed(e) | JobErr::Panicked(e)) => {
                self.flight.finish(jid);
                return if live { Err(e) } else { Ok(0) };
            }
        };
        let fresh = epoch == self.atlas.epoch();
        let mut landed = 0;
        for (m, tex) in tiles {
            let old_node = self.atlas.get(&m.key).map(|p| p.node);
            if !lands(self.flight.accepts(jid, &m.key), fresh, m.merge, old_node, m.node) {
                continue; // `tex` drops here: the decode is stale
            }
            self.decodes += 1;
            self.planes += tex.count() as u64;
            // A merge takes the resident textures OUT (`get_mut` + replace) and re-inserts old + new under the same
            // key at once: `insert` refunds the bytes captured at the previous insert and charges the merged payload,
            // so the budget stays exact (mutating in place would leave the atlas's captured size stale).
            let tex = match self.atlas.get_mut(&m.key) {
                Some(old) if m.merge => std::mem::replace(&mut old.tex, TileTex::Planes(Vec::new())).merge(tex),
                _ => tex,
            };
            self.atlas.insert(m.key, Planes { tex, node: m.node, seen }, epoch);
            landed += 1;
        }
        self.flight.finish(jid);
        Ok(landed)
    }

    /// Cancel the running job and forget everything in flight / wanted: its result (and any late one) is dropped by
    /// [`Self::install`]. For a source change or a viewer close; a frame or view change needs no call (`plan` cancels a
    /// job only when none of its tiles is wanted any more).
    pub fn cancel_all(&mut self) {
        if let Some(r) = &self.running {
            r.cancel.cancel();
        }
        self.flight.clear();
        self.sig = None;
    }

    /// Is a job running on the worker (its [`Done`] not polled yet)?
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Record `w`'s tiles as in flight under its generation.
    fn register(&mut self, w: &Work) {
        for m in &w.misses {
            self.flight.add(m.key, Pending { jid: w.jid, seen: w.seen, node: m.node, want: w.want.clone(), est: m.est, merge: m.merge });
        }
    }

    /// Cancel the running job when it no longer serves the last plan: another atlas epoch, or none of its tiles is still
    /// wanted. A job with ANY wanted tile keeps running and lands (see [`Self::install`]); only a job nobody looks at
    /// any more stops at its next check.
    fn prune(&self) {
        let Some(r) = &self.running else { return };
        if useless(r.epoch, self.atlas.epoch(), &self.flight, &r.keys) {
            r.cancel.cancel();
        }
    }

    /// Hard check of the atlas contract: the visible set is NEVER evicted, so a visible set bigger than the budget
    /// would silently push VRAM past it. Refusing is the viewer's cue to fall back (or the user's to zoom in). Counts
    /// the planes actually resident ([`TilePayload::bytes`](wgpu_tile_atlas::TilePayload::bytes)) plus the estimated
    /// bytes of the tiles an async job is about to add.
    fn check_visible(&self, keys: &[jph_wgpu_rs::RegionKey]) -> Result<(), String> {
        use wgpu_tile_atlas::TilePayload;
        let cap = self.atlas.memory_stats().1;
        if cap == 0 {
            return Ok(()); // unbounded atlas
        }
        let mut uniq = std::collections::HashSet::new();
        let bytes: usize = keys
            .iter()
            .filter(|k| uniq.insert(**k))
            .map(|k| self.flight.charge(k, self.atlas.get(k).map_or(0, Planes::bytes)))
            .sum();
        if bytes > cap {
            const MIB: f64 = (1u64 << 20) as f64;
            return Err(format!(
                "the visible tiles hold {:.1} MiB of resident textures, over the {:.1} MiB atlas budget (zoom in or free VRAM)",
                bytes as f64 / MIB,
                cap as f64 / MIB
            ));
        }
        Ok(())
    }

    /// Borrow the resident textures ([`Planes`]) for tile `(rx, ry)` at `level`
    /// (`None` = full-res), or `None` if it is not currently resident (never requested, or LRU-evicted).
    /// Slot order == component order == `header().channels`; an absent slot was not requested; sample each plane by its
    /// own format.
    pub fn get(&self, level: Option<u32>, rx: u32, ry: u32) -> Option<&Planes> {
        self.atlas.get(&jph_wgpu_rs::RegionKey { level: level.unwrap_or(u32::MAX), rx, ry })
    }
}

/// The atlas keys of a request's tiles at `level` (`None` = full resolution).
#[cfg(feature = "htj2k-backend")]
fn region_keys(tiles: &[(u32, u32)], level: Option<u32>) -> Vec<jph_wgpu_rs::RegionKey> {
    let level = level.unwrap_or(u32::MAX);
    tiles.iter().map(|&(rx, ry)| jph_wgpu_rs::RegionKey { level, rx, ry }).collect()
}

/// Estimated resident bytes of the planes a miss will add, charged by `check_visible` while its job is in flight
/// (before the textures exist). Exact for every route except a GPU-blitted F16 plane vs the CPU upload (`R16Float`,
/// 2 B): the blit is `R32Float` (4 B) on a device without `R16Float` storage writes ([`GpuContext::r16_store`]), which
/// depends on each blob's tag, so the estimate takes the GPU size when the route allows it.
#[cfg(feature = "htj2k-backend")]
#[derive(Clone, Copy)]
struct Est {
    /// jph-wgpu decodes the tiles (a <=4-channel tile is ONE `Rgba16Float` texture).
    gpu_htj2k: bool,
    /// Bytes per texel of a GPU-blitted F16 plane when the wave / EXV-RT GPU kernels may decode it (`0` = no blit route).
    blit_f16: usize,
}

#[cfg(feature = "htj2k-backend")]
impl Est {
    /// Bytes of a `tw` x `th` tile holding `chans` of `dtypes`.
    fn tile(self, tw: u32, th: u32, chans: &[usize], dtypes: &[Dtype]) -> usize {
        let px = tw as usize * th as usize;
        if self.gpu_htj2k && !(dtypes.contains(&Dtype::F32) || dtypes.len() > 4) {
            return px * 8; // the RGBA16F pack holds every channel
        }
        let plane = |ci: &usize| {
            let d = dtypes.get(*ci).copied().unwrap_or(Dtype::F32);
            let host = HostFmt::of(d).resident_dtype(d).size();
            if d == Dtype::F16 { host.max(self.blit_f16) } else { host }
        };
        px * chans.iter().map(plane).sum::<usize>()
    }
}

#[cfg(feature = "htj2k-backend")]
impl ExvDec {
    /// A decoder over `ctx` (the atlas builds one for the inline path, the worker thread builds its own).
    pub(crate) fn new(ctx: GpuContext, uploaded: Arc<AtomicU64>, decode_budget: u64) -> Self {
        Self { ctx, uploaded, htj: None, decode_budget }
    }

    /// The jph-wgpu GPU + resident decode state, built on first use (only the forced-GPU / LOD HTJ2K route needs it).
    fn htj(&mut self) -> Result<(&jph_wgpu_rs::Gpu, &mut jph_wgpu_rs::ResidentCtx), String> {
        if self.htj.is_none() {
            let gpu = jph_wgpu_rs::Gpu::from_parts(self.ctx.device.clone(), self.ctx.queue.clone(), self.ctx.adapter.clone());
            let rctx = jph_wgpu_rs::ResidentCtx::new(&gpu).map_err(|e| format!("resident ctx: {e}"))?;
            self.htj = Some((gpu, rctx));
        }
        let (gpu, rctx) = self.htj.as_mut().ok_or("resident ctx: not built")?;
        Ok((&*gpu, rctx))
    }

    /// **Lossless-HTJ2K misses** (one multi-component codestream per tile).
    ///
    /// # Routing (measured; the ONE owner of the rule is [`ExvResidentAtlas::htj2k_on_gpu`], passed in as `on_gpu`)
    /// jph-wgpu decodes ONE tile per call at ~25 queue submits each (the cost is fixed per tile, not per pixel): on a
    /// 4096x2048 RGB F16 frame it takes 580 / 205 / 65 ms (256 / 512 / 1024 px tiles; 4.5 / 6.4 / 8.2 ms per tile, flat
    /// from 1 to 4 decode contexts: the submits serialize) against 13 / 25 / 33 ms for the CPU `decode_tile` over the same
    /// tiles in parallel (RTX 3080 Ti + the dev CPU). So full-resolution tiles decode on the CPU (parallel over tiles,
    /// `registry` HTJ2K backend: only the wanted components, the pull stops after the last one) + ONE batched host upload
    /// ([`Self::upload_host`]). A coarser LOD stays on jph-wgpu (its level-truncated decode is the only LOD path that does
    /// not decode full resolution first).
    ///
    /// GPU tile routing by dtype (jph-wgpu decodes every component; unwanted planes are dropped, never kept resident):
    ///   * any F32 (the 32-bit float magnitude / NLT path) OR `>4` channels -> jph-wgpu's N-plane resident path
    ///     (`decode_level_planes`): the RGBA16F pack holds <= 4 components and rounds a true f32 to f16;
    ///   * else (<= 4 channels, no F32) -> the RGBA16F resident pack ([`TileTex::Packed`], holds every channel).
    pub(crate) fn decode_htj2k(&mut self, b: &Batch<'_, '_>, level: Option<u32>, on_gpu: bool) -> Result<Vec<TileTex>, String> {
        use exv_codec::{BlobKind, Dst, classify_blob};
        let (walk, profile, dtypes, misses, limits, cancel) = (b.walk, b.profile, b.dtypes, b.misses, &b.limits, b.cancel);
        if !on_gpu {
            let groups = misses
                .iter()
                .map(|m| {
                    let (tw, th) = tile_wh(open_reader(walk, m)?, m.t);
                    Ok(m.chans.iter().map(|&ci| HostPlane { w: tw, h: th, dtype: dtypes[ci] }).collect())
                })
                .collect::<Result<Vec<Vec<HostPlane>>, String>>()?;
            // The decoder's own peak (jph codestream + scratch) sits beside the staged planes: the same `tile_cost`
            // `FrameReader::check_budget` charges, taken over the misses (edge tiles are smaller, so the max is the nominal one).
            let peak = misses.iter().try_fold(0u64, |p, m| Ok::<_, String>(p.max(open_reader(walk, m)?.tile_cost(false).map_err(|e| e.to_string())?)))?;
            let texs = self.upload_host(&groups, limits, peak, cancel, |g, bufs| {
                let m = misses.get(g).ok_or_else(|| corrupt("htj2k upload group"))?;
                let rd = open_reader(walk, m).map_err(GpuError::Corrupt)?;
                let (tw, th) = tile_wh(rd, m.t);
                let blob = rd.tile_blob(m.t).map_err(|e| GpuError::from_decode(e, &format!("tile {}", m.t)))?;
                // Wanted channels decode into the group's scratch planes (in channel order); the rest are `Skip`.
                let mut it = bufs.iter_mut();
                let mut dst: Vec<Dst<'_>> = Vec::with_capacity(dtypes.len());
                for ci in 0..dtypes.len() {
                    dst.push(match m.chans.binary_search(&ci) {
                        Ok(_) => Dst::Plane(it.next().ok_or_else(|| corrupt("htj2k scratch plane"))?.as_mut_slice()),
                        Err(_) => Dst::Skip,
                    });
                }
                exv_codec::decode_tile_dst(blob, dtypes, tw, th, profile, &mut dst)
                    .map_err(|e| GpuError::from_decode(e, &format!("CPU decode of tile {}", m.t)))
            })?;
            return Ok(misses.iter().zip(texs).map(|(m, t)| TileTex::Planes(slots_of(&m.chans, t, dtypes.len()))).collect());
        }
        let needs_planes = dtypes.contains(&Dtype::F32) || dtypes.len() > 4;
        let mut out = Vec::with_capacity(misses.len());
        let ctx = self.ctx.clone();
        for m in misses {
            cancelled(cancel)?;
            let rd = open_reader(walk, m)?;
            let (rx, ry) = (m.t % rd.header().tiles_x(), m.t / rd.header().tiles_x());
            let blob = rd.tile_blob(m.t).map_err(|e| format!("tile {}: {e}", m.t))?;
            let payload = match classify_blob(blob).ok_or_else(|| "bad blob tag".to_string())? {
                BlobKind::Htj2k(p) => p,
                _ => return Err(format!("non-HTJ2K tile {}", m.t)),
            };
            // jph-wgpu records + submits on this thread: its validation / OOM errors surface in OUR error scope (the
            // viewer's device has no panicking handler to fall back on, and scopes are per thread).
            let tex = ctx
                .scoped(|| self.decode_htj2k_tile(payload, level, needs_planes, (rx, ry)).map_err(GpuError::Device))
                .map_err(|e| e.to_string())?;
            out.push(match tex {
                TileTex::Planes(all) if all.len() != dtypes.len() => {
                    return Err(format!("tile {}: {} planes, expected {}", m.t, all.len(), dtypes.len()));
                }
                // jph-wgpu decoded every component: keep the wanted planes only (the budget charges what is resident).
                TileTex::Planes(all) => TileTex::Planes(
                    all.into_iter().enumerate().map(|(ci, p)| p.filter(|_| m.chans.binary_search(&ci).is_ok())).collect(),
                ),
                packed => packed,
            });
        }
        Ok(out)
    }

    /// One GPU HTJ2K tile (see [`Self::decode_htj2k`] for the routing): every component, N planes or one packed texture.
    fn decode_htj2k_tile(
        &mut self,
        payload: &[u8],
        level: Option<u32>,
        needs_planes: bool,
        (rx, ry): (u32, u32),
    ) -> Result<TileTex, String> {
        if needs_planes {
            // Full-res (`level == None`) needs the codestream's full level (`num_decomp`); jph-wgpu's
            // `decode_level_planes` takes an explicit level (there is no full-res planes entry), so open a `Decoder` to
            // read `num_levels()` and (RCT trap) assert the planar path is valid — exv is RCT=off, but a forged RCT-on
            // codestream would corrupt planar output, so hard-reject here.
            let lvl = match level {
                Some(l) => l,
                None => {
                    let budget = self.decode_budget;
                    let (gpu, _) = self.htj()?;
                    let dec = jph_wgpu_rs::Decoder::open(gpu, payload.to_vec(), budget)
                        .map_err(|e| format!("open tile ({rx},{ry}): {e}"))?;
                    if dec.config().color_transform {
                        return Err(format!("tile ({rx},{ry}): RCT-on codestream cannot use the planar AOV path"));
                    }
                    dec.num_levels()
                }
            };
            let budget = self.decode_budget;
            let (gpu, rctx) = self.htj()?;
            let planes = rctx
                .decode_level_planes(gpu, payload, lvl, budget)
                .map_err(|err| format!("decode planes tile ({rx},{ry}): {err}"))?;
            Ok(TileTex::Planes(planes.into_iter().map(|t| Some(PlaneTex::plain(t))).collect()))
        } else {
            let budget = self.decode_budget;
            let (gpu, rctx) = self.htj()?;
            let tex = match level {
                None => rctx.decode_to_texture_fresh(gpu, payload, budget),
                Some(l) => rctx.decode_level_fresh(gpu, payload, l, budget),
            }
            .map_err(|err| format!("decode tile ({rx},{ry}): {err}"))?;
            Ok(TileTex::Packed(PlaneTex::plain(tex)))
        }
    }

    /// **Track A — exv-native wave/reorder misses -> resident per-plane textures**, all tiles in ONE batch.
    ///
    /// A wave/reorder tile stores one self-describing per-channel sub-blob per component
    /// ([`exv_codec::frame::FrameReader::tile_channel_blob`]); only the wanted channels' sub-blobs are read and verified.
    /// Each component becomes its own single-channel [`ResidentTexture`] (format by dtype) so the atlas payload and the
    /// [`crate::compose`] sampler are IDENTICAL to the HTJ2K N-plane path (slot order == channel order). Per-channel routing:
    ///   * **Wavelet53** + `esz<=2` (F16/U16) + a rANS-tagged sub-blob -> the batched GPU 5/3 wave stages
    ///     ([`decode_tiles_wave_gpu`], bit-exact vs CPU `decode_channel`) -> on-device blit. That kernel is
    ///     `rANS -> repack -> inverse-5/3` ONLY — no lossy dequant and no reorder-undelta, so routing a
    ///     `WaveletLossy`/`ReorderDelta` channel through it would silently corrupt: we do NOT.
    ///   * everything else (`WaveletLossy`/`ReorderDelta` transforms, a raw/deflate sub-blob, an `esz>2` component, a
    ///     custom `table_log > 12`) -> CPU `decode_channel_into` + batched host upload ([`Self::assemble`]). EXPLICIT,
    ///     never a silent wrong-GPU decode.
    pub(crate) fn decode_wave(&self, b: &Batch<'_, '_>) -> Result<Vec<TileTex>, String> {
        use exv_codec::{BlobKind, GpuKind, classify_blob};
        let (walk, profile, dtypes, misses) = (b.walk, b.profile, b.dtypes, b.misses);
        let gpu_ok = profile.gpu_kind() == Some(GpuKind::RansWave);
        // GPU only where the kernel takes the element size AND the plane has an exact device blit (U8/U16 do not).
        let (jobs, subs) = tile_jobs(walk, misses, dtypes, b.cancel, |ci, dtype, sub| match classify_blob(sub) {
            Some(BlobKind::Rans(p)) if gpu_ok && dtype.size() <= 2 && self.ctx.blit_fmt(dtype).is_some() => Some((ci, p)),
            _ => None,
        })?;
        let gts = if jobs.iter().all(|j| j.chans.is_empty()) {
            jobs.iter().map(|_| Vec::new()).collect()
        } else {
            cancelled(b.cancel)?;
            decode_tiles_wave_gpu(&self.ctx, dtypes, &jobs, b.cancel).map_err(|e| e.to_string())?
        };
        self.assemble(b, &jobs, &subs, gts)
    }

    /// **EXV-RT misses -> resident per-plane textures**, all tiles in ONE batch. An EXV-RT tile stores one sub-blob per
    /// component: the wanted F16 channels carrying a `BlockRt` blob go to [`decode_tiles_exvrt_gpu`] (bit-exact vs CPU
    /// `decode_plane` at q=0, <= 1 f16 ULP at q>0 -> on-device blit); any other wanted sub-blob (RAW / a non-F16
    /// component) is CPU-decoded and uploaded ([`Self::assemble`]). A device-limit gate surfaces as `Err`, NEVER a silent
    /// wrong-GPU decode.
    pub(crate) fn decode_rt(&self, b: &Batch<'_, '_>) -> Result<Vec<TileTex>, String> {
        use exv_codec::{BlobKind, classify_blob};
        let (walk, dtypes, misses) = (b.walk, b.dtypes, b.misses);
        // The block codec codes F16 only: a BlockRt sub-blob on another dtype goes to the CPU decode, which rejects it.
        let (jobs, subs) = tile_jobs(walk, misses, dtypes, b.cancel, |ci, dtype, sub| match classify_blob(sub) {
            Some(BlobKind::BlockRt(p)) if dtype == Dtype::F16 => Some((ci, p)),
            _ => None,
        })?;
        let gts = if jobs.iter().all(|j| j.chans.is_empty()) {
            jobs.iter().map(|_| Vec::new()).collect()
        } else {
            cancelled(b.cancel)?;
            decode_tiles_exvrt_gpu(&self.ctx, dtypes, &jobs, b.cancel).map_err(|e| e.to_string())?
        };
        self.assemble(b, &jobs, &subs, gts)
    }

    /// **BC6H misses -> resident per-plane textures**. F16 channels carrying a `Bc6h` blob decode on the GPU via
    /// [`decode_tiles_bc6h_gpu`] and blit on-device; any other channel (RAW / non-F16) decodes on the CPU and uploads.
    pub(crate) fn decode_bc6h(&self, b: &Batch<'_, '_>) -> Result<Vec<TileTex>, String> {
        use exv_codec::{BlobKind, classify_blob};
        let (walk, dtypes, misses) = (b.walk, b.dtypes, b.misses);
        let (jobs, subs) = tile_jobs(walk, misses, dtypes, b.cancel, |ci, dtype, sub| match classify_blob(sub) {
            Some(BlobKind::Bc6h(p)) if dtype == Dtype::F16 => Some((ci, p)),
            _ => None,
        })?;
        let gts = if jobs.iter().all(|j| j.chans.is_empty()) {
            jobs.iter().map(|_| Vec::new()).collect()
        } else {
            cancelled(b.cancel)?;
            decode_tiles_bc6h_gpu(&self.ctx, dtypes, &jobs, b.cancel).map_err(|e| e.to_string())?
        };
        self.assemble(b, &jobs, &subs, gts)
    }

    /// Build every miss's [`TileTex::Planes`] (its wanted channels only): each GPU-decoded plane in `gts` is blitted
    /// ON-DEVICE into a resident single-channel texture — ALL blits of the request recorded into ONE encoder and ONE submit
    /// — and every other wanted channel (`subs[mi]` = its verified sub-blobs) is CPU-decoded (tiles/channels in parallel)
    /// straight into the batched upload staging ([`Self::upload_host`]): the explicit correct fallback of every GPU route.
    fn assemble(
        &self,
        b: &Batch<'_, '_>,
        jobs: &[TileJob<'_>],
        subs: &[Vec<(usize, &[u8])>],
        gts: Vec<Vec<(usize, GpuTile)>>,
    ) -> Result<Vec<TileTex>, String> {
        let (profile, dtypes, misses, limits, cancel) = (b.profile, b.dtypes, b.misses, &b.limits, b.cancel);
        let ctx = &self.ctx;
        let nch = dtypes.len();
        let mut planes: Vec<Vec<Option<PlaneTex>>> = misses.iter().map(|_| (0..nch).map(|_| None).collect()).collect();
        if gts.iter().any(|g| !g.is_empty()) {
            cancelled(cancel)?;
            ctx.scoped(|| {
                let mut enc = encoder(&ctx.device, "exv.atlas.blit");
                for ((slots, job), g) in planes.iter_mut().zip(jobs).zip(&gts) {
                    for (ci, gt) in g {
                        let slot = slots.get_mut(*ci).ok_or_else(|| corrupt("atlas: GPU plane channel index"))?;
                        let (tex, view, format) = blit_record(ctx, &mut enc, gt, dtypes[*ci], job.tw, job.th)?;
                        *slot = Some(PlaneTex::new(ctx, ResidentTexture::new_plane(tex, view, job.tw, job.th, format)));
                    }
                }
                ctx.queue.submit([enc.finish()]);
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        }
        // CPU fallbacks (the wanted channels the GPU did not decode): one upload group per (tile, channel), decoded in
        // parallel straight into the staging while the GPU runs the batch above.
        let mut todo: Vec<(usize, usize, &[u8])> = Vec::new();
        for (mi, s) in subs.iter().enumerate() {
            for &(ci, sub) in s {
                if planes[mi].get(ci).is_some_and(Option::is_none) {
                    todo.push((mi, ci, sub));
                }
            }
        }
        if !todo.is_empty() {
            let groups: Vec<Vec<HostPlane>> =
                todo.iter().map(|&(mi, ci, _)| vec![HostPlane { w: jobs[mi].tw, h: jobs[mi].th, dtype: dtypes[ci] }]).collect();
            // Each fallback call decodes ONE channel: charge the largest per-channel peak (the SAME `tile_cost` formula,
            // taken over that channel only) beside the staged planes, like the HTJ2K route does for its whole tile.
            let peak = todo.iter().try_fold(0u64, |p, &(mi, ci, _)| {
                let m = misses.get(mi).ok_or("atlas: fallback tile index")?;
                let c = open_reader(b.walk, m)?.tile_cost_of(false, Chans::Only(&[ci])).map_err(|e| e.to_string())?;
                Ok::<_, String>(p.max(c))
            })?;
            let texs = self.upload_host(&groups, limits, peak, cancel, |g, bufs| {
                let &(mi, ci, sub) = todo.get(g).ok_or_else(|| corrupt("atlas upload group"))?;
                let out = bufs.first_mut().ok_or_else(|| corrupt("atlas scratch plane"))?;
                exv_codec::decode_channel_into(sub, dtypes[ci], jobs[mi].tw, jobs[mi].th, profile, out)
                    .map_err(|e| GpuError::from_decode(e, &format!("CPU decode of tile {} ch {ci}", misses[mi].t)))
            })?;
            for (&(mi, ci, _), t) in todo.iter().zip(texs) {
                planes[mi][ci] = t.into_iter().next();
            }
        }
        misses
            .iter()
            .zip(planes)
            .map(|(m, slots)| {
                if m.chans.iter().any(|&ci| slots.get(ci).is_none_or(Option::is_none)) {
                    return Err(format!("atlas: a wanted channel of tile {} was not decoded", m.t));
                }
                Ok(TileTex::Planes(slots))
            })
            .collect()
    }

    /// Upload host-decoded planes into owned resident single-channel textures in ONE pass per run: `groups[g]` are planes
    /// one `decode(g, scratch)` call produces (`scratch[k]` = plane `k`, sized exactly `w*h*esz` here, decoded over in
    /// place; reused across the groups of a rayon split, so no per-tile allocation once warm). Each group's planes are
    /// copied row by row straight from the cache-hot scratch into the request's upload staging (`write_buffer_with`, the
    /// only host copy) at a `COPY_BYTES_PER_ROW_ALIGNMENT`-multiple pitch — so every plane starts on that alignment too,
    /// above the texel-size offset rule of `copy_buffer_to_texture` — then ONE encoder records every buffer -> texture
    /// copy (one submit). The destination buffer comes from the device's pool ([`GpuContext::pooled`]). Runs are cut below
    /// the device's `max_buffer_size` AND the caller's host budget (`limits.max_alloc`: the staging is host memory)
    /// minus `peak` x the parallel workers, the decoder working set ([`FrameReader::tile_cost`](exv_codec::frame::FrameReader::tile_cost))
    /// charged BEFORE decoding; a run or tile that cannot fit both is a typed `Limit` error.
    /// Format by dtype = [`HostFmt`] (every dtype has one), so a CPU-uploaded plane is interchangeable with a
    /// GPU-decoded one for the per-plane sampler.
    ///
    /// A run is also kept to the decode pool (`decode_budget`, never below the biggest single tile): the staging and
    /// the textures it produces are transient VRAM of the decode, and a cancelled job (`cancel`, checked per run and per
    /// group) stops at the next such boundary instead of finishing a multi-GiB batch nobody wants.
    fn upload_host<F>(&self, groups: &[Vec<HostPlane>], limits: &DecodeLimits, peak: u64, cancel: &Cancel, decode: F) -> Result<Vec<Vec<PlaneTex>>, String>
    where
        F: Fn(usize, &mut [Vec<u8>]) -> GpuResult<()> + Sync + Send,
    {
        let ctx = &self.ctx;
        let dev = usize::try_from(ctx.device.limits().max_buffer_size).unwrap_or(usize::MAX);
        // `fill_parts` decodes the groups of a run in parallel: every concurrent worker owns its own decoder working set.
        let peak = peak.saturating_mul(rayon::current_num_threads().min(groups.len()).max(1) as u64);
        let sizes: Vec<usize> = groups.iter().map(|g| g.iter().map(HostPlane::staged).sum()).collect();
        let pool = usize::try_from(self.decode_budget).unwrap_or(usize::MAX).max(sizes.iter().copied().max().unwrap_or(0));
        let cap = run_cap(dev, limits, peak).min(pool);
        let mut out = Vec::with_capacity(groups.len());
        let mut start = 0;
        while start < groups.len() {
            cancelled(cancel)?;
            let (mut end, mut total) = (start, 0usize);
            while end < groups.len() && total.saturating_add(sizes[end]) <= cap {
                total += sizes[end];
                end += 1;
            }
            if end == start {
                // One tile's planes alone do not fit: the caller's budget (typed limit) or the device buffer cap.
                let need = sizes[start];
                limits.check((need as u64).saturating_add(peak)).map_err(|e| GpuError::from_decode(e, "atlas upload of one tile's planes").to_string())?;
                return Err(format!("atlas upload: one tile's planes ({need} B) exceed max_buffer_size {dev}"));
            }
            out.extend(self.upload_run(&groups[start..end], &sizes[start..end], total, cancel, |g, b| decode(start + g, b))?);
            start = end;
        }
        Ok(out)
    }

    /// One run of [`Self::upload_host`] (`total` = the sum of `sizes`, under the device buffer cap).
    fn upload_run<F>(&self, groups: &[Vec<HostPlane>], sizes: &[usize], total: usize, cancel: &Cancel, decode: F) -> Result<Vec<Vec<PlaneTex>>, String>
    where
        F: Fn(usize, &mut [Vec<u8>]) -> GpuResult<()> + Sync + Send,
    {
        let ctx = &self.ctx;
        if total == 0 {
            return Ok(groups.iter().map(|_| Vec::new()).collect());
        }
        // The staging is the device's pooled upload chunk (warm pages, see `stage_bytes`) or the fresh fallback; the
        // fill runs unlocked in parallel (`fill_parts`), and the texture copies below read straight out of it.
        let staged = stage_bytes(ctx, "exv.atlas.upload", total as u64, |mut view| {
            fill_parts(view.slice(..), sizes, Vec::<Vec<u8>>::new, |bufs, g, mut dst| -> GpuResult<()> {
                cancelled_gpu(cancel)?;
                let planes = groups.get(g).ok_or_else(|| corrupt("atlas upload group"))?;
                if bufs.len() < planes.len() {
                    bufs.resize_with(planes.len(), Vec::new);
                }
                for (b, p) in bufs.iter_mut().zip(planes) {
                    fit(b, p.len())?;
                }
                let bufs = &mut bufs[..planes.len()];
                decode(g, bufs)?;
                for (b, p) in bufs.iter().zip(planes) {
                    let (head, tail) = dst.split_at_checked(p.staged()).map_err(|_| corrupt("atlas upload: plane overruns its part"))?;
                    stage_rows(b, p, head)?;
                    dst = tail;
                }
                Ok(())
            })
        })
        .map_err(|e| e.to_string())?;
        self.uploaded.fetch_add(total as u64, Ordering::Relaxed);
        ctx.scoped(|| {
            let mut enc = encoder(&ctx.device, "exv.atlas.upload");
            let mut off = 0u64;
            let mut out = Vec::with_capacity(groups.len());
            for g in groups {
                let mut texs = Vec::with_capacity(g.len());
                for p in g {
                    let format = HostFmt::of(p.dtype).format();
                    let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
                    // Recycled when the pool has this exact size: the copy below rewrites the whole `w x h` extent.
                    let tex = ctx.plane_tex("exv.atlas.cpu_plane", p.w, p.h, format, usage);
                    let pitch = u32::try_from(p.pitch()).map_err(|_| GpuError::Limit("atlas upload: row pitch exceeds u32".into()))?;
                    enc.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: staged.buf(),
                            layout: wgpu::TexelCopyBufferLayout { offset: off, bytes_per_row: Some(pitch), rows_per_image: Some(p.h) },
                        },
                        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        wgpu::Extent3d { width: p.w, height: p.h, depth_or_array_layers: 1 },
                    );
                    off += p.staged() as u64;
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                    texs.push(PlaneTex::new(ctx, ResidentTexture::new_plane(tex, view, p.w, p.h, format)));
                }
                out.push(texs);
            }
            staged.submit(ctx, enc);
            Ok(out)
        })
        .map_err(|e| e.to_string())
    }
}

#[cfg(feature = "htj2k-backend")]
impl ExvResidentAtlas {
    /// Total tile decodes performed (cache hits excluded) — lets the viewer/tests confirm only
    /// visible, non-resident tiles are decoded.
    pub fn decode_count(&self) -> u64 {
        self.decodes
    }

    /// `(resident_bytes, budget_bytes)` of the backing atlas.
    pub fn memory_stats(&self) -> (usize, usize) {
        self.atlas.memory_stats()
    }

    /// Snapshot of the atlas counters for a status bar / log line ([`AtlasStats`] implements `Display`).
    pub fn stats(&self) -> AtlasStats {
        let (resident_bytes, budget_bytes) = self.memory_stats();
        AtlasStats {
            resident_bytes,
            budget_bytes,
            decodes: self.decodes,
            reused: self.reused,
            planes: self.planes,
            uploaded_bytes: self.uploaded.load(Ordering::Relaxed),
            inflight: self.flight.len(),
            pooled_bytes: self.ctx.tex_pool_stats().1,
        }
    }
}

#[cfg(feature = "htj2k-backend")]
impl Drop for ExvResidentAtlas {
    /// A viewer closing or retiring its source must stop the worker's job: cancel it, then the worker field drops and
    /// joins (bounded) after this returns. The resident planes are released and the idle plane textures freed here, so a
    /// closed clip leaves no pooled VRAM behind (a result the worker still holds recycles into the capped pool when it dies).
    fn drop(&mut self) {
        self.cancel_all();
        self.atlas.clear();
        self.ctx.trim_tex_pool();
    }
}

/// How a host-decoded plane of a dtype becomes a resident texture: THE dtype table of the host upload (total: every
/// dtype has a route, so a valid clip never fails at present time for its channel types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostFmt {
    /// Bytes copy in unchanged (`F16 → R16Float`, `F32 → R32Float`, `U32 → R32Uint`).
    Raw(wgpu::TextureFormat),
    /// Widened on the host to `R32Float` through `exv_codec::image::sample_f32` = the value the CPU viewer displays:
    /// BF16 exactly (`bits << 16`), U8/U16 normalized `v / 255` / `v / 65535` in f32. Why not `R8Unorm`/`R16Unorm`:
    /// `R16Unorm` is not a baseline format (feature `TEXTURE_FORMAT_16BIT_NORM`) and a hardware unorm conversion is
    /// not pinned bit-identical to the CPU path; one exact mechanism for the three costs 4 B/px.
    Widen,
}

impl HostFmt {
    fn of(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F16 => HostFmt::Raw(wgpu::TextureFormat::R16Float),
            Dtype::F32 => HostFmt::Raw(wgpu::TextureFormat::R32Float),
            Dtype::U32 => HostFmt::Raw(wgpu::TextureFormat::R32Uint),
            Dtype::Bf16 | Dtype::U8 | Dtype::U16 => HostFmt::Widen,
        }
    }

    fn format(self) -> wgpu::TextureFormat {
        match self {
            HostFmt::Raw(f) => f,
            HostFmt::Widen => wgpu::TextureFormat::R32Float,
        }
    }

    /// The dtype whose size a resident texel of a `dtype` plane has (VRAM accounting).
    fn resident_dtype(self, dtype: Dtype) -> Dtype {
        match self {
            HostFmt::Raw(_) => dtype,
            HostFmt::Widen => Dtype::F32,
        }
    }
}

/// Every component of an HTJ2K codestream with these channel types is jph-wgpu decodable: F16/F32 only here
/// (`exv_codec::htj2k_gpu_decodable` excludes BF16/U8/U16; a U32 component overflows `dc_from_bitdepth(32)` in jph-wgpu).
fn htj2k_gpu_ok(dtypes: &[Dtype]) -> bool {
    dtypes.iter().all(|&d| d != Dtype::U32 && exv_codec::htj2k_gpu_decodable(d))
}

/// Size `buf` to exactly `len` bytes, growing FALLIBLY (a hostile tile size is a typed `Limit`, not an abort). Contents
/// are unspecified: the decoders overwrite every byte.
fn fit(buf: &mut Vec<u8>, len: usize) -> GpuResult<()> {
    if len > buf.len() {
        buf.try_reserve(len - buf.len()).map_err(|_| GpuError::Limit(format!("atlas upload: cannot allocate a {len} B plane")))?;
    }
    buf.resize(len, 0);
    Ok(())
}

/// Copy one decoded plane (`p.len()` bytes, tight rows of the decoded dtype) into its staging slot (`p.staged()` bytes,
/// texel rows at `p.pitch()`), widening each element to f32 for a [`HostFmt::Widen`] plane.
fn stage_rows(src: &[u8], p: &HostPlane, mut dst: wgpu::WriteOnly<'_, [u8]>) -> GpuResult<()> {
    let (src_row, row, pitch) = (p.src_row(), p.row(), p.pitch());
    if src.len() != p.len() || dst.len() != p.staged() {
        return Err(corrupt("atlas upload: plane / staging size mismatch"));
    }
    let widen = HostFmt::of(p.dtype) == HostFmt::Widen;
    if !widen && row == pitch {
        dst.copy_from_slice(src);
        return Ok(());
    }
    // The staging is write-only (write-combined): a widened row is built in a reused buffer and written once.
    let mut wide: Vec<u8> = if widen { Vec::with_capacity(row) } else { Vec::new() };
    for line in src.chunks_exact(src_row.max(1)) {
        let (head, tail) = dst.split_at_checked(pitch).map_err(|_| corrupt("atlas upload: row overruns the staging"))?;
        if widen {
            wide.clear();
            wide.extend((0..p.w as usize).flat_map(|i| exv_codec::image::sample_f32(line, p.dtype, i).to_le_bytes()));
            head.into_slice(..row).copy_from_slice(&wide);
        } else {
            head.into_slice(..row).copy_from_slice(line);
        }
        dst = tail;
    }
    Ok(())
}

/// The slot vector of a [`TileTex::Planes`]: `texs[k]` lands in slot `chans[k]`, the rest stay absent.
fn slots_of(chans: &[usize], texs: Vec<PlaneTex>, nch: usize) -> Vec<Option<PlaneTex>> {
    let mut slots: Vec<Option<PlaneTex>> = (0..nch).map(|_| None).collect();
    for (&ci, t) in chans.iter().zip(texs) {
        if let Some(s) = slots.get_mut(ci) {
            *s = Some(t);
        }
    }
    slots
}

/// One host-decoded plane of an upload batch (pixel size + dtype; its staging layout derives from them).
#[derive(Clone, Copy, Debug)]
struct HostPlane {
    w: u32,
    h: u32,
    dtype: Dtype,
}

impl HostPlane {
    /// Tight row bytes of the DECODED plane (`w * esz`).
    fn src_row(&self) -> usize {
        self.w as usize * self.dtype.size()
    }

    /// Tight plane bytes (what the decoder writes).
    fn len(&self) -> usize {
        self.src_row() * self.h as usize
    }

    /// Tight row bytes of the resident TEXEL row (`w * texel size`; 4 B/px for a [`HostFmt::Widen`] plane).
    fn row(&self) -> usize {
        self.w as usize * HostFmt::of(self.dtype).resident_dtype(self.dtype).size()
    }

    /// Row pitch in the staging buffer: `copy_buffer_to_texture` requires `bytes_per_row` to be a multiple of
    /// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] (a WebGPU constant, not a device limit). Tile rows of 256/512/1024 px are
    /// already multiples; only edge tiles pad.
    fn pitch(&self) -> usize {
        self.row().next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
    }

    /// Staging bytes of the plane (`pitch * h`, a multiple of the row alignment, so the next plane stays aligned).
    fn staged(&self) -> usize {
        self.pitch() * self.h as usize
    }
}

/// Largest staging run (bytes) of [`ExvResidentAtlas::upload_host`]: under the device buffer cap `dev` AND what the
/// caller's host budget leaves beside the decoder's `peak` working set (the planes + the decoder coexist in RAM).
fn run_cap(dev: usize, limits: &DecodeLimits, peak: u64) -> usize {
    dev.min(usize::try_from(limits.max_alloc.saturating_sub(peak)).unwrap_or(usize::MAX))
}

/// Pixel size `(w, h)` of tile `t` of `reader`'s grid (edge tiles are smaller than the nominal tile).
fn tile_wh(reader: &exv_codec::frame::FrameReader<'_>, t: u32) -> (u32, u32) {
    let tx_n = reader.header().tiles_x();
    let (_, _, tw, th) = reader.header().tile_rect(t % tx_n, t / tx_n);
    (tw, th)
}

/// How a request's missed tiles decode (a clip has ONE profile, so one route per request).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Route {
    /// Lossless HTJ2K (CPU decode + upload by default; jph-wgpu when forced / LOD).
    Htj2k,
    /// exv wave / reorder rANS tiles (batched GPU wave stages; CPU for lossy wave / other blobs).
    Wave,
    /// EXV-RT block tiles (batched GPU block kernel).
    Rt,
    /// Hardware BC6H_SF16 block tiles + optional lossless rANS residual.
    Bc6h,
}

impl Route {
    /// The route of a clip's codec: lossless HTJ2K, the exv wave/reorder rANS family PLUS lossy wavelet (whose tiles
    /// decode on the CPU, `gpu_kind() == None`: the one explicit per-entry-point extension of the registry table),
    /// EXV-RT, or BC6H. Anything else (lossy HTJ2K, unknown) is a hard error, never a silent fallback.
    fn of(profile: &exv_codec::Profile) -> Result<Self, String> {
        use exv_codec::{GpuKind, Transform};
        match (profile.gpu_kind(), &profile.transform) {
            (Some(GpuKind::Htj2k), _) => Ok(Route::Htj2k),
            (Some(GpuKind::RansFast | GpuKind::RansWave), _) | (_, Transform::WaveletLossy { .. }) => Ok(Route::Wave),
            (Some(GpuKind::ExvRt), _) => Ok(Route::Rt),
            (Some(GpuKind::Bc6h), _) => Ok(Route::Bc6h),
            _ => Err("resident atlas needs lossless HTJ2K, an exv wave/reorder, EXV-RT, or BC6H profile".to_string()),
        }
    }
}

/// A tile a request must decode: its atlas key, its grid index `t`, the INTRA frame (decode index) it resolves to, and
/// the channels to decode (sorted). `merge` = the resident tile already holds this node and the other wanted channels:
/// the result completes it instead of replacing it.
pub(crate) struct Miss {
    pub key: jph_wgpu_rs::RegionKey,
    pub t: u32,
    pub node: u32,
    pub chans: Vec<usize>,
    pub merge: bool,
    /// Estimated resident bytes of the planes this miss adds ([`Est`]): what `check_visible` charges while it is in flight.
    pub est: usize,
}

/// One request's miss batch: what every route decoder reads (the clip walker holding the opened frames, the codec, the
/// header dtypes, the misses, the caller's host budget + CRC policy carried by the clip index, and the job's cancel
/// token the decoders poll between units of work).
pub(crate) struct Batch<'b, 'a> {
    pub walk: &'b exv_video::TileChains<'a>,
    pub profile: &'b exv_codec::Profile,
    pub dtypes: &'b [Dtype],
    pub misses: &'b [Miss],
    pub limits: DecodeLimits,
    pub cancel: &'b Cancel,
}

/// `Err` once `c` is cancelled: the decoders' cooperative checkpoint (`run_work` turns any error of a cancelled job
/// into `Done::is_cancelled`). Never placed after a queue submit; the textures of a finished stage are dropped at install.
fn cancelled(c: &Cancel) -> Result<(), String> {
    c.check().map_err(|_| "cancelled".to_string())
}

/// The already-opened tile reader of the frame a [`Miss`] decodes from (`TileChains::chain` opened it in pass 1).
fn open_reader<'w, 'a>(walk: &'w exv_video::TileChains<'a>, m: &Miss) -> Result<&'w exv_codec::frame::FrameReader<'a>, String> {
    walk.opened(m.node as usize).ok_or_else(|| format!("frame {} is not open", m.node))
}

/// One tile's verified `(channel, sub-blob)` of every wanted channel (borrowing the clip bytes).
type ChanSubs<'a> = Vec<(usize, &'a [u8])>;
/// [`ChanSubs`] per miss.
type Subs<'a> = Vec<ChanSubs<'a>>;

/// The per-tile jobs of a per-channel batch: for each miss, the verified sub-blob of every WANTED channel (`subs`, read +
/// CRC-checked for those channels only) and the GPU job of the ones `pick` selects (`(channel, dtype, sub-blob)` ->
/// `(channel, tag-stripped payload)`) plus the tile's pixel size. The payloads borrow the clip bytes.
fn tile_jobs<'a>(
    walk: &exv_video::TileChains<'a>,
    misses: &[Miss],
    dtypes: &[Dtype],
    cancel: &Cancel,
    pick: impl Fn(usize, Dtype, &'a [u8]) -> Option<(usize, &'a [u8])> + Sync,
) -> Result<(Vec<TileJob<'a>>, Subs<'a>), String> {
    use rayon::prelude::*;
    // Parallel over the misses: each wanted channel's CRC check (the CPU-side integrity gate before upload) is the only real work.
    let per: Vec<(TileJob<'a>, ChanSubs<'a>)> = misses
        .par_iter()
        .map(|m| {
            cancelled(cancel)?;
            let rd = open_reader(walk, m)?;
            let (tw, th) = tile_wh(rd, m.t);
            let mut chans = Vec::new();
            let mut subs = Vec::with_capacity(m.chans.len());
            for &ci in &m.chans {
                let dtype = *dtypes.get(ci).ok_or_else(|| format!("tile {}: channel {ci} out of range", m.t))?;
                let sub = rd.tile_channel_blob(m.t, ci).map_err(|e| format!("tile {} ch {ci}: {e}", m.t))?;
                subs.push((ci, sub));
                chans.extend(pick(ci, dtype, sub));
            }
            Ok((TileJob { chans, tw, th }, subs))
        })
        .collect::<Result<_, String>>()?;
    Ok(per.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `merge_slots` fills the absent slots `new` holds, replaces present ones `new` re-decoded, keeps every other old
    /// plane and grows to `new`'s length.
    #[test]
    fn merge_slots_completes_and_keeps() {
        let mut old = vec![Some(10), None, Some(12), None];
        merge_slots(&mut old, vec![None, Some(21), Some(22), None, Some(24)]);
        assert_eq!(old, vec![Some(10), Some(21), Some(22), None, Some(24)]);
        let mut empty: Vec<Option<u8>> = Vec::new();
        merge_slots(&mut empty, vec![None, Some(1)]);
        assert_eq!(empty, vec![None, Some(1)]);
    }

    /// Budget that fits the staged planes but not planes + decoder peak: the run cap shrinks by the peak, and the
    /// one-tile charge (`planes + peak`) is a typed `Limit` error rather than an OOM later.
    #[test]
    fn decoder_peak_is_charged_beside_planes() {
        let (planes, peak) = (1_000u64, 600u64);
        let lim = DecodeLimits::new(1_200);
        assert!(lim.check(planes).is_ok());
        assert!(matches!(lim.check(planes + peak), Err(exv_format::DecodeError::Limit { need: 1_600, max: 1_200 })));
        assert_eq!(run_cap(usize::MAX, &lim, 0), 1_200);
        assert_eq!(run_cap(usize::MAX, &lim, peak), 600);
        assert_eq!(run_cap(512, &lim, peak), 512);
        assert_eq!(run_cap(usize::MAX, &lim, 5_000), 0, "peak over the budget leaves no room: every tile is refused");
    }

    /// `want_list` sorts + de-dups, expands `All`, and refuses an out-of-range or empty set (a routing bug is an error).
    #[test]
    fn want_list_normalizes_and_validates() {
        assert_eq!(want_list(Chans::All, 3), Ok(vec![0, 1, 2]));
        assert_eq!(want_list(Chans::Only(&[2, 0, 2]), 5), Ok(vec![0, 2]));
        assert!(want_list(Chans::Only(&[5]), 5).is_err());
        assert!(want_list(Chans::Only(&[]), 5).is_err());
    }

    /// `Planes::has_all` (the compose gate of the viewer's `ready` set) on a device-free tile whose slots are absent: any
    /// asked channel (resident-looking or out of range) is missing, only the empty ask holds. The positive case needs a
    /// texture, so it is covered by the GPU gates (`atlas_channel_subset_*`, async merge).
    #[test]
    fn has_all_needs_every_asked_channel() {
        let p = Planes { tex: TileTex::Planes(vec![None, None, None]), node: 0, seen: 0 };
        assert!(p.has_all(&[]));
        assert!(!p.has_all(&[1]));
        assert!(!p.has_all(&[0, 5]), "an out-of-range channel is never held");
    }

    /// Staging layout: tight rows for 256-px F16 tiles (512 B, already aligned), padded pitch for an edge tile, every
    /// plane a whole number of aligned rows (so consecutive planes stay aligned).
    #[test]
    fn host_plane_staging_layout() {
        let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let full = HostPlane { w: 256, h: 256, dtype: Dtype::F16 };
        assert_eq!((full.row(), full.pitch(), full.staged()), (512, 512, 512 * 256));
        let edge = HostPlane { w: 100, h: 7, dtype: Dtype::F32 };
        assert_eq!((edge.row(), edge.pitch()), (400, 400usize.next_multiple_of(a)));
        assert_eq!(edge.staged() % a, 0);
        assert_eq!(edge.len(), 400 * 7);
    }
}
