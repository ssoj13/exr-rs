//! `wgpu-tile-atlas` — a standalone, codec-agnostic, wgpu-native VRAM-residency tile cache.
//!
//! This crate holds the *residency logic* of a streaming image viewer with every
//! application-specific concern (egui, OCIO, file I/O, worker threads, paths) stripped out.
//! It answers one question well: **which LOD-aware regions of an image should currently
//! live in GPU memory, and which should be evicted to stay under a byte budget?**
//!
//! Consumers (e.g. a GPU HTJ2K / EXR / OpenEXR region decoder) feed it the set of
//! currently visible [`RegionKey`]s each frame, drain a prioritised work queue with
//! [`TileAtlas::take_next`], and hand finished payloads back via [`TileAtlas::insert`].
//! The cache tracks memory, evicts least-recently-visible regions, and never drops a
//! region that is visible or was loaded this very frame.
//!
//! # Two-key invalidation
//!
//! Each viewport update carries a [`TileKeys`] pair:
//!
//! - **Content key** ([`ContentKey`]) — a *hard* fingerprint (source image, channel
//!   layout, geometry). A change clears the whole atlas and bumps the epoch.
//! - **Display key** ([`DisplayKey`]) — a *soft* fingerprint (exposure, colour transform).
//!   A change keeps the resident pixels but marks every tile stale for a rebake.
//!
//! # Epochs
//!
//! Every [`TileAtlas::clear`] bumps an `epoch` counter. Asynchronous decode results carry
//! the epoch they were issued under; [`TileAtlas::insert`] drops any payload whose epoch no
//! longer matches, so results that were in flight across an invalidation are discarded
//! cleanly instead of corrupting the cache.
//!
//! The cache itself is panic-free: no `unwrap`/`expect`, no indexing that can go out of
//! bounds, and all memory arithmetic is saturating.

#![allow(dead_code)]

use crate::prelude::wgpu;
use std::collections::{HashMap, HashSet, VecDeque};

// ── Core types (codec-agnostic) ───────────────────────────────────────────

/// LOD-aware region key.
///
/// `level` is the resolution level of a multi-resolution image — `0` is the coarsest
/// (e.g. the LL thumbnail in a wavelet codestream), higher levels are finer. `(rx, ry)`
/// is the tile/region grid coordinate *within that level*. This replaces a viewer's
/// file-path-based tile coordinate so the cache stays codec-agnostic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RegionKey {
    /// Resolution level (0 = coarsest).
    pub level: u32,
    /// Region column within the level's grid.
    pub rx: u32,
    /// Region row within the level's grid.
    pub ry: u32,
}

/// Hard-invalidation fingerprint of the *content* being viewed.
///
/// A change (new source image, channel layout, geometry) clears the atlas entirely.
/// The consumer chooses how to compute it (typically a hash of structural parameters).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContentKey(pub u64);

/// Soft-invalidation fingerprint of the *display* parameters.
///
/// A change (exposure, colour transform, channel selection) leaves the resident pixel
/// data in place but flags every tile as stale so the consumer can rebake it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DisplayKey(pub u64);

/// The combined invalidation keys supplied on every [`TileAtlas::update_viewport`].
#[derive(Clone, Copy)]
pub struct TileKeys {
    /// Hard-invalidation fingerprint.
    pub content: ContentKey,
    /// Soft-invalidation fingerprint.
    pub display: DisplayKey,
}

/// Loading state of a region, exposed for streaming/progress overlays.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtlasTileState {
    /// Waiting in the priority queue; not yet handed to a decoder.
    Queued,
    /// Handed out via [`TileAtlas::take_next`]; decode in progress.
    InFlight,
    /// Fully resident in the atlas.
    Done,
}

/// Summary returned by [`TileAtlas::update_viewport`].
pub struct ViewportUpdate {
    /// Number of regions waiting in the priority queue after the update.
    pub queued_count: usize,
    /// Number of regions currently in flight after the update.
    pub in_flight_count: usize,
    /// `true` when the display key changed — the consumer should rebake stale tiles.
    pub needs_rebake: bool,
}

/// Anything the consumer keeps resident in GPU memory.
///
/// The cache only needs the payload's byte size to enforce its budget. When a payload is
/// evicted it is simply dropped, so a wgpu-backed payload frees its texture via `Drop`.
pub trait TilePayload {
    /// Resident size of this payload in bytes (used for the memory budget).
    fn bytes(&self) -> usize;

    /// Host (CPU RAM) bytes this payload also keeps alive, charged against the separate host
    /// budget of [`TileAtlas::with_budgets`]. Defaults to `0` for payloads that live only in
    /// GPU memory.
    ///
    /// Why a second pool: a consumer that keeps CPU-side data per region (canonical pixels, a
    /// CPU display bake) usually holds MORE host bytes than device bytes. Charging both to one
    /// combined budget lets the host side grow past what RAM can afford while the combined
    /// figure still looks fine; with two budgets either pool alone triggers eviction.
    fn host_bytes(&self) -> usize {
        0
    }
}

// ── Internal atlas entry ──────────────────────────────────────────────────

/// One resident region and the bookkeeping the LRU/epoch logic needs.
struct Entry<P> {
    /// Consumer-owned resident data (dropped on eviction).
    payload: P,
    /// Byte size captured at insert time (the budget is charged against this).
    memory_bytes: usize,
    /// Host byte size captured at insert time (charged against the host budget).
    host_bytes: usize,
    /// Monotonic frame counter when this region was last inside the viewport.
    last_visible_frame: u64,
    /// Frame this region was inserted on — protects it from same-frame eviction.
    loaded_frame: u64,
}

// ── TileAtlas ─────────────────────────────────────────────────────────────

/// Persistent, memory-budgeted, LOD-aware VRAM residency cache.
///
/// Generic over the resident payload `P: TilePayload`, so it is fully codec-agnostic.
/// See the crate-level docs for the residency model, two-key invalidation, and epochs.
pub struct TileAtlas<P: TilePayload> {
    /// Resident regions keyed by their LOD-aware coordinate.
    tiles: HashMap<RegionKey, Entry<P>>,
    /// Current hard-invalidation fingerprint.
    content_key: ContentKey,
    /// Current soft-invalidation fingerprint.
    display_key: DisplayKey,
    /// Byte budget; `0` disables eviction entirely.
    max_memory: usize,
    /// Sum of resident `memory_bytes` (kept via saturating arithmetic).
    memory_used: usize,
    /// Host byte budget; `0` disables host-driven eviction (the default for [`Self::new`]).
    max_host_memory: usize,
    /// Sum of resident `host_bytes`.
    host_used: usize,
    /// Monotonic frame counter, incremented once per `update_viewport`.
    frame: u64,
    /// Generation counter; bumped on `clear` so stale async inserts can be dropped.
    epoch: u64,
    /// Priority queue of regions awaiting decode (front = highest priority).
    queue: VecDeque<RegionKey>,
    /// Membership mirror of `queue` for O(1) de-dup checks.
    queue_set: HashSet<RegionKey>,
    /// Regions handed out for decode but not yet inserted.
    in_flight: HashSet<RegionKey>,
    /// Regions inside the current viewport (protected from eviction).
    visible: HashSet<RegionKey>,
    /// Resident regions whose display bake is out of date after a soft invalidation.
    stale_tiles: HashSet<RegionKey>,
    /// Cursor position of the last `sort_by_cursor` call, to skip redundant re-sorts.
    last_sort_tile: Option<(i64, i64)>,
}

impl<P: TilePayload> TileAtlas<P> {
    /// Maximum regions in flight at once (caps decode fan-out on mass-invalidation).
    const MAX_IN_FLIGHT: usize = 64;

    /// Creates an empty atlas with the given byte budget (`0` disables eviction).
    ///
    /// Both invalidation keys start at `0`; the first [`update_viewport`](Self::update_viewport)
    /// with a non-zero content key therefore performs no spurious clear.
    pub fn new(max_memory: usize) -> Self {
        Self::with_budgets(max_memory, 0)
    }

    /// Creates an empty atlas with a device budget (charged with [`TilePayload::bytes`]) and a
    /// separate host budget (charged with [`TilePayload::host_bytes`]). Either pool over its
    /// own budget triggers LRU eviction; `0` disables that pool's limit.
    pub fn with_budgets(max_memory: usize, max_host_memory: usize) -> Self {
        Self {
            tiles: HashMap::new(),
            content_key: ContentKey(0),
            display_key: DisplayKey(0),
            max_memory,
            memory_used: 0,
            max_host_memory,
            host_used: 0,
            frame: 0,
            epoch: 0,
            queue: VecDeque::new(),
            queue_set: HashSet::new(),
            in_flight: HashSet::new(),
            visible: HashSet::new(),
            stale_tiles: HashSet::new(),
            last_sort_tile: None,
        }
    }

    /// Current epoch (generation). Hand this to async decoders so their results can be
    /// validated against the live generation in [`insert`](Self::insert).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Updates the viewport with the regions currently visible and the live keys.
    ///
    /// Order of operations (load-bearing — mirrors the ported residency algorithm):
    /// 1. advance the frame counter;
    /// 2. **hard** invalidation: a changed content key clears everything (bumping the
    ///    epoch) and adopts both new keys;
    /// 3. **soft** invalidation: otherwise a changed display key is adopted, every
    ///    resident tile is marked stale, and `needs_rebake` is flagged;
    /// 4. cancel regions that left the viewport from the queue and in-flight sets;
    /// 5. refresh `last_visible_frame` for still-resident visible regions and enqueue
    ///    newly visible ones at the **front** (highest priority), preserving slice order;
    /// 6. record the new visible set and run eviction.
    pub fn update_viewport(
        &mut self,
        visible: &[RegionKey],
        keys: &TileKeys,
    ) -> ViewportUpdate {
        self.frame += 1;
        let mut needs_rebake = false;

        // 2/3. Two-key invalidation.
        if keys.content != self.content_key {
            // HARD: structural change — drop everything and adopt both keys.
            self.clear();
            self.content_key = keys.content;
            self.display_key = keys.display;
        } else if keys.display != self.display_key {
            // SOFT: presentation change — keep pixels, flag every tile for rebake.
            self.display_key = keys.display;
            self.stale_tiles = self.tiles.keys().copied().collect();
            needs_rebake = true;
        }

        let new_visible: HashSet<RegionKey> = visible.iter().copied().collect();

        // 4. Cancel regions that are no longer visible.
        //    Disjoint-field closure capture (edition 2021) lets `retain` mutate
        //    `queue_set` while borrowing `queue`.
        self.queue.retain(|key| {
            if new_visible.contains(key) {
                true
            } else {
                self.queue_set.remove(key);
                false
            }
        });
        self.in_flight.retain(|key| new_visible.contains(key));

        // 5. Touch resident visible regions; collect genuinely new ones (order preserved).
        let mut new_tiles: Vec<RegionKey> = Vec::new();
        for &key in visible {
            if let Some(entry) = self.tiles.get_mut(&key) {
                entry.last_visible_frame = self.frame;
            } else if !self.queue_set.contains(&key) && !self.in_flight.contains(&key) {
                new_tiles.push(key);
            }
        }

        // New work invalidates any cursor-sort ordering.
        if !new_tiles.is_empty() {
            self.last_sort_tile = None;
        }
        // Push to the FRONT in reverse so the slice's first region ends up first.
        for &key in new_tiles.iter().rev() {
            self.queue.push_front(key);
            self.queue_set.insert(key);
        }

        // 6. Adopt the visible set and rebalance memory.
        self.visible = new_visible;
        self.evict_if_needed();

        ViewportUpdate {
            queued_count: self.queue.len(),
            in_flight_count: self.in_flight.len(),
            needs_rebake,
        }
    }

    /// Evicts least-recently-visible regions until the budget is satisfied.
    ///
    /// **Invariants (load-bearing — do not alter):**
    /// - eviction candidates are `!visible` *and* not loaded this frame
    ///   (`loaded_frame != frame`), so a visible or just-loaded region is **never** evicted,
    ///   even when over budget;
    /// - candidates are sorted **ascending by `last_visible_frame`** (oldest visible first =
    ///   LRU order) and dropped until both pools are within budget (`memory_used <= max_memory`
    ///   and `host_used <= max_host_memory`; a `0` budget never counts as exceeded).
    fn evict_if_needed(&mut self) {
        if !self.over_budget() {
            return;
        }

        // Candidates: neither visible nor freshly loaded this frame.
        let mut candidates: Vec<(RegionKey, u64)> = self
            .tiles
            .iter()
            .filter(|(key, entry)| {
                !self.visible.contains(*key) && entry.loaded_frame != self.frame
            })
            .map(|(key, entry)| (*key, entry.last_visible_frame))
            .collect();

        // LRU: oldest last-visible frame first.
        candidates.sort_by_key(|(_, last_visible_frame)| *last_visible_frame);

        for (key, _) in candidates {
            if !self.over_budget() {
                break;
            }
            if let Some(entry) = self.tiles.remove(&key) {
                self.memory_used = self.memory_used.saturating_sub(entry.memory_bytes);
                self.host_used = self.host_used.saturating_sub(entry.host_bytes);
            }
        }
    }

    /// Whether either pool is over its (non-zero) budget.
    fn over_budget(&self) -> bool {
        (self.max_memory != 0 && self.memory_used > self.max_memory)
            || (self.max_host_memory != 0 && self.host_used > self.max_host_memory)
    }

    /// Whether either pool is at or over its (non-zero) budget (prefetch suppression).
    fn at_budget(&self) -> bool {
        (self.max_memory != 0 && self.memory_used >= self.max_memory)
            || (self.max_host_memory != 0 && self.host_used >= self.max_host_memory)
    }

    /// Inserts a freshly decoded region, charging its bytes against the budget.
    ///
    /// `epoch` must equal the live [`epoch`](Self::epoch); a payload carrying a stale epoch
    /// (issued before an intervening [`clear`](Self::clear)) is **dropped** — it is removed
    /// from the in-flight set and discarded so a late result cannot corrupt a new generation.
    ///
    /// The inserted region is stamped as visible-and-loaded *this frame*, which protects it
    /// from same-frame eviction.
    ///
    /// Equivalent to [`insert_with_display_key`](Self::insert_with_display_key) passing the
    /// atlas's own current display key, so an arrival through this method is never flagged
    /// stale by that check — callers that don't track a per-job display key (e.g.
    /// `jph-wgpu-rs`, `exv-gpu`) keep their exact prior behaviour.
    pub fn insert(&mut self, key: RegionKey, payload: P, epoch: u64) {
        let current = self.display_key;
        self.insert_with_display_key(key, payload, epoch, current);
    }

    /// Inserts a freshly decoded region, stamped with the [`DisplayKey`] the decode job was
    /// issued under.
    ///
    /// Identical to [`insert`](Self::insert) (same epoch check, same budget accounting), plus
    /// one more check: if `display_key` no longer matches the atlas's *current* display key,
    /// the region is inserted as valid data (its content is still current — only the
    /// presentation parameters moved on) but immediately added to the stale set, exactly as if
    /// [`update_viewport`](Self::update_viewport) had just flagged it.
    ///
    /// This closes the window `insert` cannot see: a soft (display-key) invalidation only
    /// marks tiles stale that are *already resident* at the moment it runs. A decode that was
    /// in flight when the display key changed reads the OLD key at job start and, without this
    /// check, would land as if baked under the current one — the exact "the control moved but
    /// the picture didn't" symptom. Ported from `vfx-tile-atlas`'s `TileAtlas::insert`
    /// (`DisplayState::display_key` / `LoadedTile::display_key`), which had this check from the
    /// start; it was missing here because this crate's generic `insert` predates a caller that
    /// stamps jobs with a display key.
    pub fn insert_with_display_key(
        &mut self,
        key: RegionKey,
        payload: P,
        epoch: u64,
        display_key: DisplayKey,
    ) {
        if epoch != self.epoch {
            // Stale generation — discard cleanly.
            self.in_flight.remove(&key);
            return;
        }

        let memory_bytes = payload.bytes();
        let host_bytes = payload.host_bytes();

        // Replacing an existing region: refund its old bytes first.
        if let Some(old) = self.tiles.remove(&key) {
            self.memory_used = self.memory_used.saturating_sub(old.memory_bytes);
            self.host_used = self.host_used.saturating_sub(old.host_bytes);
        }

        self.tiles.insert(
            key,
            Entry {
                payload,
                memory_bytes,
                host_bytes,
                last_visible_frame: self.frame,
                loaded_frame: self.frame,
            },
        );
        self.memory_used = self.memory_used.saturating_add(memory_bytes);
        self.host_used = self.host_used.saturating_add(host_bytes);

        // No longer pending in either pipeline stage.
        self.in_flight.remove(&key);
        self.queue_set.remove(&key);

        // Presentation-stale: valid content, baked under superseded display params.
        if display_key != self.display_key {
            self.stale_tiles.insert(key);
        }
    }

    /// Rejects a region a decode/validation step refused to accept, WITHOUT storing a payload.
    ///
    /// For a consumer that validates a finished payload against something the generic cache
    /// cannot check itself (e.g. a codec-specific dimension/pitch contract) before calling
    /// [`insert`](Self::insert): a straight `return` on that validation failure would leave the
    /// region permanently `in_flight` — [`take_next`](Self::take_next) skips resident AND
    /// in-flight regions, so it would never be queued, decoded, or drawn again for the life of
    /// the atlas. This clears the in-flight marker so a future
    /// [`update_viewport`](Self::update_viewport) can re-queue it like any other missing region.
    pub fn cancel_in_flight(&mut self, key: RegionKey) {
        self.in_flight.remove(&key);
    }

    /// Pops the highest-priority region from the queue and marks it in flight.
    ///
    /// Returns `None` when the queue is empty or the [`MAX_IN_FLIGHT`](Self::MAX_IN_FLIGHT)
    /// cap is reached. Regions that became resident while queued are skipped.
    pub fn take_next(&mut self) -> Option<RegionKey> {
        if self.in_flight.len() >= Self::MAX_IN_FLIGHT {
            return None;
        }
        while let Some(key) = self.queue.pop_front() {
            self.queue_set.remove(&key);
            if self.tiles.contains_key(&key) {
                continue;
            }
            self.in_flight.insert(key);
            return Some(key);
        }
        None
    }

    /// Enqueues prefetch regions at **low priority** (back of the queue).
    ///
    /// Suppressed entirely when at or above budget, to avoid a load→evict thrash loop.
    /// Regions already resident, queued, or in flight are skipped.
    pub fn queue_prefetch(&mut self, keys: &[RegionKey]) {
        if self.at_budget() {
            return;
        }
        for &key in keys {
            if self.tiles.contains_key(&key)
                || self.queue_set.contains(&key)
                || self.in_flight.contains(&key)
            {
                continue;
            }
            self.queue.push_back(key);
            self.queue_set.insert(key);
            // New work invalidates any cursor-sort ordering, as in `update_viewport`:
            // otherwise a prefetch at an unchanged cursor would skip the next sort.
            self.last_sort_tile = None;
        }
    }

    /// Re-sorts the pending queue by squared distance from a cursor region.
    ///
    /// A no-op when fewer than two regions are queued, or when the cursor has not moved
    /// since the last sort (cached in `last_sort_tile`). Distance uses the `(rx, ry)` grid
    /// coordinate; `level` is intentionally ignored so the nearest region in screen-grid
    /// space loads first.
    pub fn sort_by_cursor(&mut self, cx: i64, cy: i64) {
        if self.queue.len() < 2 {
            return;
        }
        if self.last_sort_tile == Some((cx, cy)) {
            return;
        }
        self.last_sort_tile = Some((cx, cy));

        let mut pending: Vec<RegionKey> = self.queue.drain(..).collect();
        pending.sort_by_key(|key| {
            let dx = key.rx as i64 - cx;
            let dy = key.ry as i64 - cy;
            dx * dx + dy * dy
        });
        self.queue.extend(pending);
    }

    /// Clears every resident region and all pipeline state, then bumps the epoch.
    ///
    /// The epoch bump is what lets [`insert`](Self::insert) drop results that were in flight
    /// across this clear.
    pub fn clear(&mut self) {
        self.tiles.clear();
        self.queue.clear();
        self.queue_set.clear();
        self.in_flight.clear();
        self.stale_tiles.clear();
        self.memory_used = 0;
        self.host_used = 0;
        self.epoch += 1;
        self.last_sort_tile = None;
    }

    /// Borrows the resident payload for `key`, or `None` if it is not resident.
    pub fn get(&self, key: &RegionKey) -> Option<&P> {
        self.tiles.get(key).map(|entry| &entry.payload)
    }

    /// Mutably borrows the resident payload for `key`, or `None` if it is not resident.
    ///
    /// For a consumer that re-derives a payload's PRESENTATION in place (e.g. a CPU display
    /// bake re-encoded from data the payload already carries) without changing its residency:
    /// the mutation must not go through [`insert`](Self::insert)/[`clear`](Self::clear), which
    /// would touch `last_visible_frame`/`loaded_frame`/the memory budget for a change that is
    /// none of those things. The atlas does **not** re-measure `bytes()` after this call — if
    /// the mutation changes the payload's resident size, the budget accounting silently drifts.
    /// Only mutate in ways that leave [`TilePayload::bytes`] unchanged.
    pub fn get_mut(&mut self, key: &RegionKey) -> Option<&mut P> {
        self.tiles.get_mut(key).map(|entry| &mut entry.payload)
    }

    /// Iterates the keys of every currently resident region, in unspecified order.
    ///
    /// For a consumer that needs to enumerate what is loaded (e.g. stitching resident tiles
    /// into an analysis sample) without tracking a second copy of the key set itself.
    pub fn keys(&self) -> impl Iterator<Item = RegionKey> + '_ {
        self.tiles.keys().copied()
    }

    /// Reports the loading state of `key`: `Done` (resident), `InFlight`, `Queued`, or
    /// `None` when the region is unknown to the atlas.
    pub fn tile_state(&self, key: &RegionKey) -> Option<AtlasTileState> {
        if self.tiles.contains_key(key) {
            Some(AtlasTileState::Done)
        } else if self.in_flight.contains(key) {
            Some(AtlasTileState::InFlight)
        } else if self.queue_set.contains(key) {
            Some(AtlasTileState::Queued)
        } else {
            None
        }
    }

    /// Returns `(memory_used_bytes, memory_budget_bytes)`.
    pub fn memory_stats(&self) -> (usize, usize) {
        (self.memory_used, self.max_memory)
    }

    /// Returns `(host_used_bytes, host_budget_bytes)` for the host pool (see
    /// [`with_budgets`](Self::with_budgets)).
    pub fn host_memory_stats(&self) -> (usize, usize) {
        (self.host_used, self.max_host_memory)
    }

    /// Returns `(queued, in_flight, loaded)` region counts.
    pub fn atlas_stats(&self) -> (usize, usize, usize) {
        (self.queue.len(), self.in_flight.len(), self.tiles.len())
    }

    /// Drains and returns the regions flagged stale by a soft (display-key) invalidation.
    ///
    /// The consumer rebakes these from already-resident data; the atlas clears the flag so
    /// each stale region is reported exactly once.
    pub fn take_stale(&mut self) -> Vec<RegionKey> {
        self.stale_tiles.drain().collect()
    }
}

// ── wgpu payload helper ────────────────────────────────────────────────────

/// A concrete [`TilePayload`] wrapping a resident wgpu texture and its view.
///
/// This is a convenience for GPU consumers; the cache itself stays generic. The texture is
/// freed automatically when the payload is evicted and dropped. `bytes` is computed for an
/// RGBA16F (8 bytes/texel) region, the common HDR tile format.
#[derive(Clone, Debug)]
pub struct ResidentTexture {
    /// The resident GPU texture (freed on drop).
    pub tex: wgpu::Texture,
    /// A view over [`tex`](Self::tex) for binding/sampling.
    pub view: wgpu::TextureView,
    /// Resident size in bytes, as reported to the cache budget.
    pub bytes: usize,
}

impl ResidentTexture {
    /// Wraps a texture/view pair, computing the RGBA16F resident size as `w * h * 4 * 2`
    /// (four channels × two bytes per half-float channel).
    pub fn new(tex: wgpu::Texture, view: wgpu::TextureView, width: u32, height: u32) -> Self {
        let bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4)
            .saturating_mul(2);
        Self { tex, view, bytes }
    }

    /// Wraps a texture/view pair whose resident size is derived from an arbitrary
    /// [`wgpu::TextureFormat`], so the atlas budget stays **format-aware** for non-RGBA16F payloads
    /// (e.g. Wave 1a's per-plane AOV outputs: single-channel `R16Float` = 2 B/texel, `R32Float` /
    /// `R32Uint` / `R32Sint` = 4 B/texel). The byte cost is `w * h * format.block_copy_size()`,
    /// which reflects the true VRAM footprint the LRU eviction reasons about.
    ///
    /// Codec-agnostic: this takes only wgpu types. The RGBA16F [`Self::new`] constructor is kept as
    /// the default so existing consumers stay byte-identical; this is an *additive* variant.
    /// Falls back to 8 B/texel (the RGBA16F cost) only for the pathological case of a format that
    /// cannot report its block size — none of the formats this crate is used with.
    pub fn new_plane(
        tex: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self { tex, view, bytes: plane_bytes(width, height, format) }
    }

    /// Borrows the texture view for binding.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

impl TilePayload for ResidentTexture {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Resident byte cost of a `width × height` single-plane texture in `format`, as reported to the
/// atlas budget: `width * height * format.block_copy_size()`. Pure (no GPU) so the format-aware
/// budget math is unit-testable without a device. Falls back to 8 B/texel (RGBA16F) only for a
/// format that cannot report a block size — not any format this crate is used with.
pub fn plane_bytes(width: u32, height: u32, format: wgpu::TextureFormat) -> usize {
    let bpp = format.block_copy_size(None).unwrap_or(8) as usize;
    (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(bpp)
}

// ── Tests (GPU-free; exercise the residency logic only) ────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal payload whose only property is its byte size.
    struct Fake(usize);
    impl TilePayload for Fake {
        fn bytes(&self) -> usize {
            self.0
        }
    }

    /// Payload with device and host bytes, for the two-pool budget.
    struct Split {
        device: usize,
        host: usize,
    }
    impl TilePayload for Split {
        fn bytes(&self) -> usize {
            self.device
        }
        fn host_bytes(&self) -> usize {
            self.host
        }
    }

    fn rk(level: u32, rx: u32, ry: u32) -> RegionKey {
        RegionKey { level, rx, ry }
    }

    /// Device and host bytes are charged to separate pools, each reported on its own.
    #[test]
    fn device_and_host_pools_are_accounted_separately() {
        let mut atlas: TileAtlas<Split> = TileAtlas::with_budgets(1 << 20, 1 << 21);
        atlas.update_viewport(&[rk(0, 0, 0)], &keys(1, 1));
        atlas.take_next();
        atlas.insert(rk(0, 0, 0), Split { device: 16, host: 80 }, atlas.epoch());
        assert_eq!(atlas.memory_stats(), (16, 1 << 20));
        assert_eq!(atlas.host_memory_stats(), (80, 1 << 21));
        atlas.clear();
        assert_eq!(atlas.memory_stats().0, 0);
        assert_eq!(atlas.host_memory_stats().0, 0);
    }

    /// The host pool alone forces eviction (device is far under budget), and it takes the LRU
    /// region. With one combined budget the host side could outgrow RAM unnoticed.
    #[test]
    fn the_host_pool_alone_can_force_an_eviction_and_it_takes_the_lru_region() {
        let mut atlas: TileAtlas<Split> = TileAtlas::with_budgets(1 << 30, 150);
        let (a, b) = (rk(0, 0, 0), rk(0, 1, 0));
        atlas.update_viewport(&[a], &keys(1, 1));
        atlas.take_next();
        atlas.insert(a, Split { device: 1, host: 100 }, atlas.epoch());
        atlas.update_viewport(&[b], &keys(1, 1));
        atlas.take_next();
        atlas.insert(b, Split { device: 1, host: 100 }, atlas.epoch());
        assert_eq!(atlas.host_memory_stats().0, 200, "over the host budget until the next tick");
        // Next frame: b still visible; a is the LRU, not visible, not just loaded.
        atlas.update_viewport(&[b], &keys(1, 1));
        assert!(atlas.get(&a).is_none(), "the LRU region went");
        assert!(atlas.get(&b).is_some(), "the visible region stays");
        assert_eq!(atlas.host_memory_stats().0, 100);
        assert_eq!(atlas.memory_stats().0, 1, "device pool was never the reason");
    }

    /// Arrivals within one frame may overshoot (they are protected as just-loaded); the next
    /// frame boundary brings the pool back within budget.
    #[test]
    fn arrivals_inside_one_frame_overshoot_the_budget_and_the_boundary_recovers_it() {
        let mut atlas: TileAtlas<Split> = TileAtlas::with_budgets(0, 100);
        let all = [rk(0, 0, 0), rk(0, 1, 0), rk(0, 0, 1), rk(0, 1, 1)];
        atlas.update_viewport(&all, &keys(1, 1));
        for _ in all {
            atlas.take_next();
        }
        atlas.update_viewport(&[], &keys(1, 1));
        for key in all {
            atlas.insert(key, Split { device: 0, host: 80 }, atlas.epoch());
        }
        let (used, budget) = atlas.host_memory_stats();
        assert!(used > budget, "the overshoot is real: {used} B against {budget} B");
        assert_eq!(atlas.atlas_stats().2, all.len());
        atlas.update_viewport(&[], &keys(1, 1));
        let (used, budget) = atlas.host_memory_stats();
        assert!(used <= budget, "the frame boundary recovers it: {used} <= {budget}");
    }

    /// The cursor sort uses (squared) Euclidean distance, not a product or a sum.
    #[test]
    fn the_cursor_sort_uses_euclidean_distance_not_a_product() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(0);
        let far_x = rk(0, 3, 0); // dist^2 9, product 0
        let near = rk(0, 1, 1); // dist^2 2, product 1
        let mid = rk(0, 0, 2); // dist^2 4, product 0
        atlas.queue_prefetch(&[far_x, near, mid]);
        atlas.sort_by_cursor(0, 0);
        assert_eq!(atlas.take_next(), Some(near));
        assert_eq!(atlas.take_next(), Some(mid));
        assert_eq!(atlas.take_next(), Some(far_x));
    }

    /// New work reopens the cursor sort even when the cursor has not moved, both through the
    /// viewport and through prefetch.
    #[test]
    fn queueing_new_regions_reopens_the_cursor_sort() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(0);
        let (near, mid, far) = (rk(0, 0, 1), rk(0, 2, 2), rk(0, 5, 5));
        atlas.update_viewport(&[near, mid], &keys(1, 1));
        atlas.sort_by_cursor(0, 0);
        // The far region arrives at the FRONT (update_viewport pushes new work first).
        atlas.update_viewport(&[near, mid, far], &keys(1, 1));
        atlas.sort_by_cursor(0, 0);
        assert_eq!(atlas.take_next(), Some(near));
        assert_eq!(atlas.take_next(), Some(mid));
        assert_eq!(atlas.take_next(), Some(far));

        // Same through prefetch at an unchanged cursor.
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(0);
        atlas.queue_prefetch(&[far, mid]);
        atlas.sort_by_cursor(0, 0);
        atlas.queue_prefetch(&[near]);
        atlas.sort_by_cursor(0, 0);
        assert_eq!(atlas.take_next(), Some(near), "the prefetch reopened the sort");
    }

    fn keys(content: u64, display: u64) -> TileKeys {
        TileKeys {
            content: ContentKey(content),
            display: DisplayKey(display),
        }
    }

    /// `plane_bytes` must be format-aware so the budget reflects real VRAM: R16Float = 2 B/texel,
    /// the R32 family = 4 B/texel, and RGBA16Float = 8 B/texel (the pre-existing `new` cost).
    #[test]
    fn plane_bytes_is_format_aware() {
        let (w, h) = (64u32, 32u32);
        let n = (w * h) as usize;
        assert_eq!(plane_bytes(w, h, wgpu::TextureFormat::R16Float), n * 2);
        assert_eq!(plane_bytes(w, h, wgpu::TextureFormat::R32Float), n * 4);
        assert_eq!(plane_bytes(w, h, wgpu::TextureFormat::R32Uint), n * 4);
        assert_eq!(plane_bytes(w, h, wgpu::TextureFormat::R32Sint), n * 4);
        // Matches the RGBA16F `new` byte formula (w*h*4*2), so the default path is unchanged.
        assert_eq!(plane_bytes(w, h, wgpu::TextureFormat::Rgba16Float), n * 8);
    }

    /// Case 1: eviction to budget drops the OLDEST non-visible region first and lands
    /// at or below budget.
    #[test]
    fn evicts_oldest_non_visible_first() {
        // Budget fits two 100-byte regions.
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(200);
        let k = keys(1, 1);

        // Frame 1: a visible, then insert.
        let a = rk(0, 0, 0);
        atlas.update_viewport(&[a], &k);
        atlas.insert(a, Fake(100), atlas.epoch());

        // Frame 2: b visible, then insert (a no longer visible → its last_visible_frame=1).
        let b = rk(0, 1, 0);
        atlas.update_viewport(&[b], &k);
        atlas.insert(b, Fake(100), atlas.epoch());

        // Frame 3: c visible, then insert → memory is now 300, over the 200 budget.
        // (Eviction runs at the START of update_viewport, so the overflow created by an
        //  insert is reconciled on the next viewport tick — faithful to the source.)
        let c = rk(0, 2, 0);
        atlas.update_viewport(&[c], &k);
        atlas.insert(c, Fake(100), atlas.epoch());
        assert_eq!(atlas.memory_stats().0, 300, "over budget until next tick");

        // Frame 4: c still visible. Candidates are a (last-visible frame 1) and b (frame 2);
        // c is protected by visibility. LRU drops the OLDEST, a, to reach budget.
        atlas.update_viewport(&[c], &k);

        let (used, budget) = atlas.memory_stats();
        assert!(used <= budget, "must be within budget: {used} <= {budget}");
        assert!(atlas.get(&a).is_none(), "oldest non-visible (a) evicted first");
        assert!(atlas.get(&b).is_some(), "newer non-visible (b) retained");
        assert!(atlas.get(&c).is_some(), "visible (c) retained");
    }

    /// Case 2: a visible region and a just-loaded region are never evicted, even when the
    /// total far exceeds budget.
    #[test]
    fn never_evicts_visible_or_just_loaded() {
        // Budget of 100 but we will pin two 100-byte regions resident in one frame.
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(100);
        let k = keys(1, 1);

        let a = rk(0, 0, 0);
        let b = rk(0, 1, 0);
        // Both visible this frame.
        atlas.update_viewport(&[a, b], &k);
        // Both inserted this frame → visible AND loaded_frame == frame.
        atlas.insert(a, Fake(100), atlas.epoch());
        atlas.insert(b, Fake(100), atlas.epoch());

        let (used, budget) = atlas.memory_stats();
        assert_eq!(used, 200, "over budget but nothing evictable");
        assert!(used > budget);
        assert!(atlas.get(&a).is_some());
        assert!(atlas.get(&b).is_some());
    }

    /// Case 3: `insert` with a stale epoch is dropped; with the current epoch it is kept.
    #[test]
    fn stale_epoch_insert_dropped() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let k = keys(1, 1);
        let a = rk(0, 0, 0);

        atlas.update_viewport(&[a], &k);
        let live = atlas.epoch();

        // Stale epoch → dropped.
        atlas.insert(a, Fake(100), live.wrapping_sub(1));
        assert!(atlas.get(&a).is_none(), "stale-epoch insert must be dropped");
        assert_eq!(atlas.memory_stats().0, 0);

        // Current epoch → kept.
        atlas.insert(a, Fake(100), live);
        assert!(atlas.get(&a).is_some(), "current-epoch insert must be kept");
        assert_eq!(atlas.memory_stats().0, 100);
    }

    /// Case 4: a content-key change clears all resident regions and bumps the epoch, so a
    /// then-arriving insert carrying the old epoch is dropped.
    #[test]
    fn content_key_change_clears_and_bumps_epoch() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);

        atlas.update_viewport(&[a], &keys(1, 1));
        let old_epoch = atlas.epoch();
        atlas.insert(a, Fake(100), old_epoch);
        assert!(atlas.get(&a).is_some());

        // Hard invalidation: content key changes.
        atlas.update_viewport(&[a], &keys(2, 1));
        assert!(atlas.get(&a).is_none(), "content change clears all tiles");
        assert_eq!(atlas.memory_stats().0, 0);
        assert_ne!(atlas.epoch(), old_epoch, "epoch must be bumped");

        // A late result from the old generation is discarded.
        atlas.insert(a, Fake(100), old_epoch);
        assert!(atlas.get(&a).is_none(), "stale post-clear insert dropped");
    }

    /// Case 5: a display-key change keeps tiles, flags `needs_rebake`, and populates the
    /// stale set returned by `take_stale`.
    #[test]
    fn display_key_change_marks_stale() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);
        let b = rk(0, 1, 0);

        atlas.update_viewport(&[a, b], &keys(1, 1));
        atlas.insert(a, Fake(100), atlas.epoch());
        atlas.insert(b, Fake(100), atlas.epoch());

        // Soft invalidation: only the display key changes.
        let update = atlas.update_viewport(&[a, b], &keys(1, 2));
        assert!(update.needs_rebake, "display change must request rebake");
        assert!(atlas.get(&a).is_some(), "tiles retained on soft invalidation");
        assert!(atlas.get(&b).is_some());

        let mut stale = atlas.take_stale();
        stale.sort_by_key(|k| (k.level, k.rx, k.ry));
        assert_eq!(stale, vec![a, b], "both resident tiles marked stale");
        assert!(atlas.take_stale().is_empty(), "stale set drained once");
    }

    /// Case 5b: a region decoded under display key A, arriving via
    /// [`TileAtlas::insert_with_display_key`] AFTER the atlas's live key has moved to B (it was
    /// in flight across the change), must land stale immediately — not merely "not yet marked",
    /// which is what plain [`TileAtlas::insert`] (and every consumer that ignores this method)
    /// would do.
    #[test]
    fn in_flight_arrival_across_a_display_key_change_lands_stale() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);

        // Job issued while the atlas is on display key A (content=1, display=1).
        atlas.update_viewport(&[a], &keys(1, 1));
        let epoch = atlas.epoch();

        // The display key moves to B while the job is still in flight.
        atlas.update_viewport(&[a], &keys(1, 2));
        assert!(atlas.take_stale().is_empty(), "nothing resident yet to stale");

        // The job lands, stamped with the OLD key (A) it was issued under.
        atlas.insert_with_display_key(a, Fake(100), epoch, DisplayKey(1));

        assert!(atlas.get(&a).is_some(), "content is valid and must still be stored");
        assert_eq!(
            atlas.take_stale(),
            vec![a],
            "an arrival baked under a superseded display key must be stale on arrival"
        );
    }

    /// A region that arrives with the CURRENT display key is not flagged stale.
    #[test]
    fn in_flight_arrival_matching_the_current_display_key_is_not_stale() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);
        atlas.update_viewport(&[a], &keys(1, 5));
        atlas.insert_with_display_key(a, Fake(100), atlas.epoch(), DisplayKey(5));
        assert!(atlas.take_stale().is_empty());
    }

    /// Plain [`TileAtlas::insert`] is unchanged: it stamps the CURRENT display key, so it never
    /// self-flags stale. This pins the exact backward-compatibility guarantee `jph-wgpu-rs` and
    /// `exv-gpu` rely on — neither tracks a per-job display key, and this method must keep
    /// behaving exactly as before the check was added.
    #[test]
    fn plain_insert_never_self_flags_stale() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);
        atlas.update_viewport(&[a], &keys(1, 1));
        atlas.insert(a, Fake(100), atlas.epoch());
        atlas.update_viewport(&[a], &keys(1, 2)); // display key moves on; a is resident already
        atlas.take_stale(); // drain the mass-stale from update_viewport itself
        // A second plain insert (e.g. a re-decode) after the move must not self-stale either.
        atlas.insert(a, Fake(100), atlas.epoch());
        assert!(atlas.take_stale().is_empty(), "plain insert always matches ITS OWN current key");
    }

    /// `cancel_in_flight` clears the marker without storing a payload, and the region can be
    /// queued again afterwards — the exact recovery a consumer-side validation rejection needs.
    #[test]
    fn cancel_in_flight_frees_the_region_for_requeueing() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);

        atlas.update_viewport(&[a], &keys(1, 1));
        assert_eq!(atlas.tile_state(&a), Some(AtlasTileState::Queued));
        atlas.take_next();
        assert_eq!(atlas.tile_state(&a), Some(AtlasTileState::InFlight));

        // A decoder finished, but the consumer's own validation refused the payload.
        atlas.cancel_in_flight(a);
        assert_eq!(
            atlas.tile_state(&a),
            None,
            "no longer in flight, and nothing was inserted"
        );

        // Still visible next frame -> re-queued, not stuck forever.
        atlas.update_viewport(&[a], &keys(1, 1));
        assert_eq!(atlas.tile_state(&a), Some(AtlasTileState::Queued));
    }

    /// `get_mut` lets a consumer re-derive a payload's presentation in place, and the mutation
    /// is visible through `get` afterwards, with the resident byte count untouched (the
    /// contract: only mutate in ways that leave `bytes()` unchanged).
    #[test]
    fn get_mut_mutates_in_place_without_touching_the_budget() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);
        atlas.update_viewport(&[a], &keys(1, 1));
        atlas.insert(a, Fake(100), atlas.epoch());

        let before = atlas.memory_stats().0;
        if let Some(payload) = atlas.get_mut(&a) {
            payload.0 = 999; // mutate the VALUE, not the reported `bytes()` (still 100)
        }
        assert_eq!(atlas.get(&a).map(|p| p.0), Some(999), "mutation is visible");
        assert_eq!(atlas.memory_stats().0, before, "budget accounting is untouched");

        assert!(atlas.get_mut(&rk(0, 9, 9)).is_none(), "a non-resident key is None");
    }

    /// `keys` enumerates exactly the resident regions — not queued or in-flight ones.
    #[test]
    fn keys_lists_only_resident_regions() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(1_000);
        let a = rk(0, 0, 0);
        let b = rk(0, 1, 0);
        let c = rk(0, 2, 0);

        atlas.update_viewport(&[a, b, c], &keys(1, 1));
        atlas.insert(a, Fake(10), atlas.epoch());
        atlas.insert(b, Fake(10), atlas.epoch());
        // c stays queued/in-flight, never inserted.

        let mut got: Vec<RegionKey> = atlas.keys().collect();
        got.sort_by_key(|k| (k.level, k.rx, k.ry));
        assert_eq!(got, vec![a, b], "queued/in-flight regions are not resident");
    }

    /// Case 6: `sort_by_cursor` orders the pending queue by squared distance to the cursor.
    #[test]
    fn sort_by_cursor_orders_by_distance() {
        let mut atlas: TileAtlas<Fake> = TileAtlas::new(0); // no eviction
        let k = keys(1, 1);

        // Three not-yet-loaded regions become visible → all queued.
        let near = rk(0, 5, 5);
        let mid = rk(0, 8, 5);
        let far = rk(0, 20, 20);
        atlas.update_viewport(&[far, mid, near], &k);
        assert_eq!(atlas.atlas_stats().0, 3, "all three queued");

        // Cursor sits on `near`.
        atlas.sort_by_cursor(5, 5);

        // Drain in priority order: nearest first.
        let first = atlas.take_next();
        let second = atlas.take_next();
        let third = atlas.take_next();
        assert_eq!(first, Some(near));
        assert_eq!(second, Some(mid));
        assert_eq!(third, Some(far));
    }
}
