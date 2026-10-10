//! Per-frame tile orchestration: which tiles
//! the view needs, producing them off the UI thread, the RAM cache, the atlas
//! uploads, and the draw list with coarser-level fallback.
//!
//! Sources ([`super::source::TileSource`]): the in-RAM materialized frame and
//! the out-of-core file. The planning ([`plan`]) is pure and unit-tested; the
//! GPU side only follows it.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;

use super::atlas::{Slot, TileAtlas};
use super::cache::Lru;
use super::grid::{Grid, TileId};
use super::reduce::TileData;
use super::render::{DrawList, Instance};
use super::source::{Production, ProductionContext, Readiness, TileSource, Unit};
use crate::image::Selection;

/// A thread-safe "wake the UI loop" callback (the same shape as `scancache::Wake`).
pub type RegionWake = Arc<dyn Fn() + Send + Sync>;

#[cfg(feature = "ofx")]
static ACCEPTED_REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(feature = "ofx")]
fn next_accepted_revision() -> Result<u64, super::region::RegionError> {
    ACCEPTED_REVISION.try_update(
        std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire,
        |revision| revision.checked_add(1),
    ).map_err(|_| "accepted presentation identity exhausted".into())
}

/// Identity of a tile's pixels: the source generation (bumped whenever the
/// shown content changes: frame, layer, overlay, effect) plus the tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// Source generation.
    pub source: u64,
    /// Physical effect output identity (zero for ordinary decoded tiles).
    pub run: u64,
    /// The tile.
    pub id: TileId,
}

/// What the shown content is; a change starts a new source generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDesc {
    /// `DecodedFrame::id`.
    pub frame: u64,
    /// What is shown (layer or solo channel).
    pub view: Selection,
    /// Overlay generation (`ImageBackend::pixels_gen`).
    pub pixels_gen: u64,
    /// Effect-stage key (0 without a stage).
    pub stage: u64,
}

/// Where one drawn tile's pixels come from: the atlas slot of `from`, and the
/// part of that tile covering `id` (texels of `from`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    /// The tile to draw.
    pub id: TileId,
    /// The resident tile used for it (itself or an ancestor).
    pub from: TileId,
    /// Texel rect `[x0, y0, x1, y1]` inside `from`.
    pub texels: [f32; 4],
}

/// For every tile in `want`, the finest resident tile covering it: itself, else
/// the nearest ancestor (the level above covers it with half the texels), so a
/// view is never blank while finer tiles stream in. `resident` answers whether
/// a tile can be drawn now. Tiles with no resident ancestor at all are skipped.
pub fn plan(grid: &Grid, want: &[TileId], mut resident: impl FnMut(TileId) -> bool) -> Vec<Pick> {
    let mut out = Vec::with_capacity(want.len());
    for &id in want {
        let Some(src) = grid.source_rect(id) else { continue };
        for level in id.level..grid.levels() {
            let Some(from) = grid.tile_at(level, src.x, src.y) else { break };
            if !resident(from) {
                continue;
            }
            let Some(texels) = grid.texture_rect(from, src) else { break };
            out.push(Pick {
                id,
                from,
                texels,
            });
            break;
        }
    }
    out
}

/// The CPU half of the tile path for one displayed image: the source, the
/// worker results, the RAM cache.
pub struct TileState {
    desc: Option<SourceDesc>,
    generation: u64,
    /// The generation the GPU atlas was last trimmed to.
    atlas_generation: u64,
    /// Last complete presentation. Its atlas slots remain untouched while the
    /// next frame's overview is being produced.
    presented: Option<Presented>,
    /// Terminal failure/abort allows playback to leave this generation.
    failed: bool,
    context: ProductionContext,
    stopped_units: HashSet<Unit>,
    #[cfg(feature = "ofx")]
    completed_candidate: Option<Unit>,
    waiting: HashMap<Unit, Readiness>,
    source: Option<Arc<dyn TileSource>>,
    grid: Option<Grid>,
    ram: Lru<RamKey, RamEntry>,
    #[cfg(feature = "ofx")]
    budget: super::region::RegionBudget,
    #[cfg(feature = "ofx")]
    region_wake: Arc<std::sync::Mutex<Option<RegionWake>>>,
    /// Units in flight, per generation.
    pending: HashSet<(u64, Unit)>,
    tx: Sender<Produced>,
    rx: Receiver<Produced>,
    /// Called by a worker after it sends a finished unit. During playback the
    /// event loop redraws only at frame deadlines, so without this a frame's
    /// tiles would wait a whole period to be presented (and playback, which
    /// waits for that presentation, would show every second or third frame).
    wake: Option<RegionWake>,
    #[cfg(test)]
    pub scheduled_units: usize,
}

/// Pixel owners use the existing RAM index; display halves and original
/// effect work have distinct entries, never a second cache.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RamKey {
    Display(TileKey),
    #[cfg(feature = "ofx")]
    Processed { source: u64, unit: Unit },
}
impl RamKey {
    fn source(&self) -> u64 {
        match self {
            Self::Display(key) => key.source,
            #[cfg(feature = "ofx")]
            Self::Processed { source, .. } => *source,
        }
    }
}
enum RamEntry {
    Display(Arc<TileData>),
    #[cfg(feature = "ofx")]
    Processed(Arc<super::processed::ProcessedWork>),
}
pub enum CachedTile {
    Display(Arc<TileData>),
    #[cfg(feature = "ofx")]
    Processed(super::processed::ProcessedUpload),
}
impl std::ops::Deref for CachedTile {
    type Target = TileData;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Display(tile) => tile,
            #[cfg(feature = "ofx")]
            Self::Processed(upload) => &upload.tile,
        }
    }
}
/// One finished unit: its generation, the unit, and its tiles (or why not).
type Produced = (u64, Unit, Readiness, ProductionContext, Production);

#[cfg(feature = "ofx")]
#[derive(Clone, Copy, Debug)]
pub struct AcceptedEffect {
    /// Presentation revision covering every accepted physical run, not one tile run.
    pub run: u64,
    pub time: f64,
    pub size: (u32, u32),
    pub aspect: f64,
    pub place: Placement,
}

struct Presented {
    desc: SourceDesc,
    grid: Grid,
    picks: Vec<Pick>,
    place: Placement,
    keys: HashMap<TileId, TileKey>,
    #[cfg(feature = "ofx")]
    pixels: Option<crate::consumers::AcceptedPixels>,
}

impl TileState {
    /// A state whose RAM cache holds at most `ram_budget` bytes.
    pub fn new(ram_budget: usize) -> Self {
        let (tx, rx) = channel();
        #[cfg(feature = "ofx")]
        let region_wake = Arc::new(std::sync::Mutex::new(None::<RegionWake>));
        #[cfg(feature = "ofx")]
        let budget = {
            let signal = region_wake.clone();
            super::region::RegionBudget::with_wake(ram_budget, Some(Arc::new(move || {
                let wake = signal.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
                if let Some(wake) = wake { wake(); }
            })))
        };
        Self {
            desc: None,
            generation: 0,
            atlas_generation: 0,
            presented: None,
            failed: false,
            context: ProductionContext::default(),
            stopped_units: HashSet::new(),
            #[cfg(feature = "ofx")]
            completed_candidate: None,
            waiting: HashMap::new(),
            source: None,
            grid: None,
            ram: Lru::new(ram_budget),
            #[cfg(feature = "ofx")]
            budget,
            #[cfg(feature = "ofx")]
            region_wake,
            pending: HashSet::new(),
            tx,
            rx,
            wake: None,
            #[cfg(test)]
            scheduled_units: 0,
        }
    }

    /// Wake the UI loop whenever a worker finishes a unit (see `wake`).
    pub fn set_wake(&mut self, wake: RegionWake) {
        self.wake = Some(wake);
    }

    /// The current grid.
    pub fn grid(&self) -> Option<&Grid> {
        self.grid.as_ref()
    }

    /// The finest level within `max_pixels` / `max_dim` as `f32` RGBA, for the
    /// scopes: straight from the source when it has it (the in-RAM frame),
    /// else assembled from this generation's cached tiles once all of them are
    /// in (the file source is never decoded whole for a scope; the level is
    /// requested so it arrives).
    pub fn scope_level(&mut self, max_pixels: u64, max_dim: u32) -> Option<(u32, Arc<Vec<f32>>)> {
        let (grid, src) = (self.grid?, Arc::clone(self.source.as_ref()?));
        #[cfg(feature = "ofx")]
        if src.processed() { return None; }
        if let Some(found) = src.scope_level(&grid, max_pixels, max_dim) {
            return Some(found);
        }
        let level = super::source::finest_level(&grid, max_pixels, max_dim);
        let (cols, rows) = grid.tiles(level);
        let ids: Vec<TileId> =
            (0..rows).flat_map(|ty| (0..cols).map(move |tx| TileId { level, tx, ty })).collect();
        let tiles: Vec<CachedTile> = ids.iter().filter_map(|&id| self.cached(id)).collect();
        if tiles.len() != ids.len() {
            self.request(ids);
            return None;
        }
        let (lw, lh) = grid.level_size(level);
        let mut out = vec![0f32; lw as usize * lh as usize * 4];
        for (id, tile) in ids.iter().zip(&tiles) {
            let r = grid.rect(*id)?;
            for y in 0..r.h {
                for x in 0..r.w {
                    let t = (y as usize * tile.w as usize + x as usize) * 4;
                    let o = ((r.y + y) as usize * lw as usize + (r.x + x) as usize) * 4;
                    for c in 0..4 {
                        out[o + c] = f32::from(imath_rs::Half::from_bits(tile.rgba[t + c]));
                    }
                }
            }
        }
        Some((level, Arc::new(out)))
    }

    /// The current source generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether `desc` differs from the shown content (the caller then builds
    /// the source and calls [`Self::set_source`]).
    pub fn is_stale(&self, desc: &SourceDesc) -> bool {
        self.source.is_none() || self.desc.as_ref() != Some(desc)
    }

    /// Retire the obsolete CPU owner before materializing its replacement.
    /// Workers/export holders keep their immutable snapshots alive.
    pub fn retire_cpu_source(&mut self) {
        self.cancel_work();
        self.source = None;
    }

    pub fn description(&self) -> Option<&SourceDesc> {
        self.desc.as_ref()
    }

    /// Prepare CPU units without mutating a GPU presentation (direct -> tiles).
    pub fn prepare_frame(&mut self, place: Placement) -> Option<Grid> {
        self.drain();
        let grid = self.grid?;
        let top = grid.levels() - 1;
        let wanted = grid.visible(grid.pick_level(place.zoom), grid.display_view(place.view()), 1);
        #[cfg(feature = "ofx")]
        if let Some(source) = &self.source && source.processed() {
            let units: HashSet<_> = wanted.iter().map(|&id| source.unit(id)).collect();
            let completed = self.completed_candidate;
            self.ram.unpin_all();
            for unit in units {
                self.ram.set_pinned(&RamKey::Processed { source: self.generation, unit }, true);
            }
            if let Some(unit) = completed {
                self.ram.set_pinned(&RamKey::Processed { source: self.generation, unit }, true);
            }
            self.pin_top();
            // Only obsolete unpinned RAM work retires; accepted presentation
            // owners keep old pixels alive until successful GPU acceptance.
            while let Some(entry) = self.ram.pop_lru() { drop(entry); }
        }
        self.request([TileId {
            level: top,
            tx: 0,
            ty: 0,
        }]);
        self.request(wanted);
        self.pin_top();
        Some(grid)
    }

    pub fn overview_ready(&self) -> bool {
        self.grid.is_some_and(|g| self.has_cached(TileId { level: g.levels() - 1, tx: 0, ty: 0 }))
    }

    /// Show new content: a new generation, the old tiles dropped from RAM
    /// (their results still in flight are ignored when they land).
    pub fn set_source(&mut self, desc: SourceDesc, source: Arc<dyn TileSource>) {
        self.cancel_work();
        self.start_generation();
        // Effect revisions retain their accepted image. Selection/overlay edits
        // explicitly remove it; a frame transition may hold the previous frame.
        if self.presented.as_ref().is_some_and(|old| {
            old.desc.view != desc.view
                || (old.desc.frame == desc.frame && old.desc.pixels_gen != desc.pixels_gen)
        }) {
            self.presented = None;
        }
        let g = self.generation;
        self.grid = Some(source.grid());
        self.source = Some(source);
        self.desc = Some(desc);
        self.pending.clear();
        self.ram.retain(|k| k.source() == g);
    }

    /// Whether this decoded frame has reached the viewport. Playback uses this
    /// to let an asynchronous tile generation finish before replacing it again.
    pub fn is_presented(&self, frame: u64) -> bool {
        self.accepted_description()
            .is_some_and(|desc| desc.frame == frame)
    }

    /// Failed tile production must not stall the rest of a sequence.
    pub fn can_advance(&self, frame: u64) -> bool {
        self.source.is_none()
            || self.is_presented(frame)
            || self.failed
            || self.desc.as_ref().is_none_or(|desc| desc.frame != frame)
    }

    /// A deliberate file change must not hold the previous file's pixels.
    pub fn clear_presentation(&mut self) {
        self.cancel_work();
        self.presented = None;
    }

    fn start_generation(&mut self) {
        self.context = ProductionContext::default();
        self.generation += 1;
        self.failed = false;
    }

    fn cancel_work(&mut self) {
        self.context.cancel();
        self.pending.clear();
        self.waiting.clear();
        self.stopped_units.clear();
        #[cfg(feature = "ofx")]
        { self.completed_candidate = None; }
        self.ram.retain(|_| false);
    }

    /// Validate and pin the next pixel identity before any destructive GPU change.
    fn prepare_presentation(&mut self, grid: Grid, picks: Vec<Pick>, place: Placement)
        -> Result<Presented, super::region::RegionError> {
        let desc = self.desc.clone().ok_or("presentation has no source description")?;
        let mut keys = HashMap::new();
        // The overview owner remains accepted even for an offscreen viewport.
        let top = TileId { level: grid.levels() - 1, tx: 0, ty: 0 };
        for id in picks.iter().map(|pick| pick.from).chain(std::iter::once(top)) {
            keys.insert(id, self.tile_key(id));
        }
        #[cfg(feature = "ofx")]
        let pixels = if self.source.as_ref().is_some_and(|source| source.processed()) {
            // Unchanged coverage reuses its pixel identity and physical owner set.
            if let Some(old) = &self.presented
                && old.grid == grid && old.picks == picks && old.keys == keys
                && old.pixels.as_ref().is_some_and(|pixels| pixels.description() == &desc) {
                old.pixels.clone()
            } else {
                Some(self.freeze_processed(&desc, grid, &picks, &keys)?)
            }
        } else { None };
        Ok(Presented { desc, grid, picks, place, keys,
            #[cfg(feature = "ofx")]
            pixels })
    }

    #[cfg(test)]
    fn accept_frame(&mut self, grid: Grid, picks: Vec<Pick>, place: Placement) {
        if !self.overview_ready() || self.atlas_generation != self.generation { return; }
        match self.prepare_presentation(grid, picks, place) {
            Ok(shown) => self.presented = Some(shown),
            Err(error) => { log::error!("presentation refused: {error}"); self.failed = true; }
        }
    }

    /// Restrict the preflighted owners to successful residency without reading RAM again.
    fn reselect_presentation(&self, mut prepared: Presented, picks: Vec<Pick>)
        -> Result<Presented, super::region::RegionError> {
        let top = TileId { level: prepared.grid.levels() - 1, tx: 0, ty: 0 };
        let mut used: HashSet<_> = picks.iter().map(|pick| pick.from).collect();
        used.insert(top);
        prepared.keys.retain(|id, _| used.contains(id));
        #[cfg(feature = "ofx")]
        if let Some(pixels) = &prepared.pixels {
            if let Some(old) = &self.presented
                && old.desc == prepared.desc && old.grid == prepared.grid
                && old.keys == prepared.keys && old.picks == picks {
                prepared.pixels = old.pixels.clone();
            } else {
                prepared.pixels = Some(pixels.reselect(next_accepted_revision()?, &picks)?);
            }
        }
        prepared.picks = picks;
        Ok(prepared)
    }

    /// Identity of the image actually accepted for presentation.
    pub fn accepted_description(&self) -> Option<&SourceDesc> {
        self.presented.as_ref().map(|shown| &shown.desc)
    }

    #[cfg(feature = "ofx")]
    pub(crate) fn accepted_effect(&self) -> Option<AcceptedEffect> {
        let shown = self.presented.as_ref()?;
        let pixels = shown.pixels.as_ref()?;
        Some(AcceptedEffect { run: pixels.revision(), time: pixels.lineage().time, size: shown.grid.size(),
            aspect: f64::from(shown.grid.display_aspect()), place: shown.place })
    }

    #[cfg(all(test, feature = "ofx"))]
    pub(crate) fn accepted_retirement_credit(&self) -> usize {
        self.presented.as_ref().and_then(|shown| shown.pixels.as_ref())
            .map_or(0, crate::consumers::AcceptedPixels::retirement_credit)
    }

    /// A consumer borrows the exact accepted run set, including the overview fallback.
    #[cfg(feature = "ofx")]
    pub(crate) fn accepted_pixels(&self) -> Option<crate::consumers::AcceptedPixels> {
        self.presented.as_ref()?.pixels.clone()
    }

    #[cfg(feature = "ofx")]
    fn freeze_processed(
        &mut self,
        desc: &SourceDesc,
        grid: Grid,
        picks: &[Pick],
        keys: &HashMap<TileId, TileKey>,
    ) -> Result<crate::consumers::AcceptedPixels, super::region::RegionError> {
        use crate::consumers::{AcceptedCoverage, AcceptedPixels};
        let lineage = self.source.as_ref().and_then(|source| source.processed_lineage())
            .ok_or("processed source has no immutable provenance")?;
        let mut coverage = |pick: Pick| -> Result<AcceptedCoverage, super::region::RegionError> {
            let key = *keys.get(&pick.from).ok_or("accepted pick has no output identity")?;
            let ram_key = self.ram_key(pick.from);
            let Some(RamEntry::Processed(work)) = self.ram.get(&ram_key) else {
                return Err("accepted pick has no original-precision owner".into());
            };
            let (w, h) = work.dimensions();
            let view = work.view(super::grid::PixRect { x: 0, y: 0, w, h })
                .ok_or("accepted owner has invalid checked extent")?;
            Ok(AcceptedCoverage { pick, key, view })
        };
        let incoming = picks.iter().copied().map(&mut coverage).collect::<Result<Vec<_>, _>>()?;
        let top = TileId { level: grid.levels() - 1, tx: 0, ty: 0 };
        let (w, h) = grid.size();
        let texels = grid.texture_rect(top, super::grid::PixRect { x: 0, y: 0, w, h })
            .ok_or("accepted overview does not cover its canvas")?;
        let overview = coverage(Pick { id: top, from: top, texels })?;
        let revision = next_accepted_revision()?;
        AcceptedPixels::new(revision, desc.clone(), grid, lineage, incoming, overview)
    }

    /// Produce the units of `ids` that are neither cached nor in flight, on
    /// the rayon pool (a unit may cover several of the ids: one decode each).
    pub fn request(&mut self, ids: impl IntoIterator<Item = TileId>) {
        let (Some(src), Some(grid)) = (self.source.clone(), self.grid) else {
            return;
        };
        // A direct presentation cancels tile work but retains the immutable
        // source for scopes. Restart only when tiles are actually needed again.
        if self.context.is_cancelled() {
            self.start_generation();
        }
        let g = self.generation;
        let epoch = src.readiness();
        for id in ids {
            if self.has_cached(id) {
                continue;
            }
            let unit = src.unit(id);
            if self.stopped_units.contains(&unit)
                || self.waiting.get(&unit).is_some_and(|old| *old == epoch)
            {
                continue;
            }
            if !self.pending.insert((g, unit)) {
                continue;
            }
            #[cfg(test)]
            {
                self.scheduled_units += 1;
            }
            let tx = self.tx.clone();
            let src = Arc::clone(&src);
            let context = self.context.clone();
            let wake = self.wake.clone();
            rayon::spawn(move || {
                let tiles = if context.is_cancelled() {
                    Production::Cancelled
                } else {
                    src.produce(&grid, unit, &context)
                };
                // The receiver lives as long as the viewer; a send after it is
                // gone just means nobody wants the tiles any more.
                if tx.send((g, unit, epoch, context, tiles)).is_err() {
                    log::trace!("{unit:?} finished after the viewer closed");
                } else if let Some(wake) = wake {
                    wake();
                }
            });
        }
    }

    /// Collect finished units into the RAM cache. Returns how many tiles landed.
    pub fn drain(&mut self) -> usize {
        let mut n = 0;
        while let Ok((g, unit, _epoch, context, tiles)) = self.rx.try_recv() {
            self.pending.remove(&(g, unit));
            if g != self.generation
                || context.is_cancelled()
                || !Arc::ptr_eq(&context.cancelled, &self.context.cancelled)
            {
                continue;
            }
            match tiles {
                Production::Ready(tiles) => {
                    self.waiting.remove(&unit);
                    for (id, t) in tiles {
                        let bytes = t.bytes();
                        self.ram
                            .insert(RamKey::Display(TileKey { source: g, run: 0, id }), RamEntry::Display(Arc::new(t)), bytes);
                        if self.grid.is_some_and(|grid| id.level == grid.levels() - 1) {
                            self.ram.set_pinned(&RamKey::Display(TileKey { source: g, run: 0, id }), true);
                        }
                        n += 1;
                    }
                }
                #[cfg(feature = "ofx")]
                Production::Processed(work) => {
                    self.waiting.remove(&unit);
                    if unit == Unit::Level
                        && let Some(grid) = &mut self.grid
                        && let Err(error) = grid.include_level(work.level) {
                            self.stopped_units.insert(unit);
                            self.failed = true;
                            log::error!("processed proxy geometry refused: {error}");
                            continue;
                    }
                    let key = RamKey::Processed { source: g, unit };
                    let bytes = work.bytes();
                    // Pin the one completed workspace until promotion; Pending
                    // must never rerender a stateful effect.
                    let promoted = match work.budget().promote() {
                        Ok(()) => true,
                        Err(super::region::RegionError::Pending) => false,
                        Err(error) => {
                            self.stopped_units.insert(unit);
                            self.failed = true;
                            log::error!("completed effect owner cannot be retained: {error}");
                            continue;
                        }
                    };
                    if !promoted { self.completed_candidate = Some(unit); }
                    self.ram.insert(key.clone(), RamEntry::Processed(work), bytes);
                    self.ram.set_pinned(&key, !promoted || unit == Unit::Level
                        || self.grid.is_some_and(|grid| unit == Unit::Tile(TileId { level: grid.levels() - 1, tx: 0, ty: 0 })));
                    n += 1;
                }
                #[cfg(feature = "ofx")]
                Production::Refused(error) => {
                    self.stopped_units.insert(unit);
                    self.failed = true;
                    log::error!("effect unit {unit:?} refused: {error}");
                }
                // A decode failure is reported, and the unit is not retried
                // until the content changes (it would fail the same way).
                #[cfg(any(feature = "ofx", test))]
                Production::Pending => {
                    self.waiting.insert(unit, _epoch);
                }
                Production::Cancelled => {
                    self.stopped_units.insert(unit);
                    self.failed = true;
                }
                Production::Failed(e) => {
                    self.stopped_units.insert(unit);
                    self.failed = true;
                    log::error!("tiles {unit:?} could not be produced: {e}");
                }
            }
        }
        n
    }

    fn ram_key(&self, id: TileId) -> RamKey {
        #[cfg(feature = "ofx")]
        if self.source.as_ref().is_some_and(|source| source.processed()) {
            return RamKey::Processed { source: self.generation, unit: self.source.as_ref().unwrap().unit(id) };
        }
        RamKey::Display(TileKey { source: self.generation, run: 0, id })
    }
    fn has_cached(&self, id: TileId) -> bool { self.ram.contains(&self.ram_key(id)) }

    /// A cached tile of the current generation. Processed display halves are
    /// temporary admitted uploads; the RAM entry remains the f32 owner.
    pub fn cached(&mut self, id: TileId) -> Option<CachedTile> {
        let key = self.ram_key(id);
        match self.ram.get(&key)? {
            RamEntry::Display(tile) => Some(CachedTile::Display(tile.clone())),
            #[cfg(feature = "ofx")]
            RamEntry::Processed(work) => {
                let result = work.upload(self.grid.as_ref()?, id);
                match result {
                    Ok(tile) => Some(CachedTile::Processed(tile)),
                    Err(super::region::RegionError::Pending | super::region::RegionError::Unsupported) => None,
                    Err(super::region::RegionError::Budget { .. })
                        if self.grid.is_some_and(|grid| id.level < grid.levels().saturating_sub(1)) => {
                            // Refuse only this optional display view, not its fixed
                            // f32 owner: smaller overview uploads remain usable.
                            None
                        }
                    Err(error) => {
                        log::error!("processed upload refused: {error}");
                        self.failed = true;
                        if let RamKey::Processed { unit, .. } = key {
                            self.stopped_units.insert(unit);
                            if self.completed_candidate == Some(unit) { self.completed_candidate = None; }
                        }
                        self.ram.remove(&key);
                        None
                    }
                }
            },
        }
    }
    fn tile_key(&mut self, id: TileId) -> TileKey {
        let key = self.ram_key(id);
        #[cfg(feature = "ofx")]
        let run = match self.ram.get(&key) {
            Some(RamEntry::Processed(work)) => work.run,
            _ => 0,
        };
        #[cfg(not(feature = "ofx"))]
        let run = { let _ = key; 0 };
        TileKey { source: self.generation, run, id }
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn region_budget(&self) -> super::region::RegionBudget { self.budget.clone() }
    #[cfg(feature = "ofx")]
    pub(crate) fn set_region_wake(&self, wake: Option<RegionWake>) {
        let old = std::mem::replace(&mut *self.region_wake.lock().unwrap_or_else(std::sync::PoisonError::into_inner), wake);
        drop(old);
    }

    /// A completed candidate may replace internal accepted owners only once
    /// they are physically releasable. External views keep promotion Pending.
    #[cfg(feature = "ofx")]
    fn promotion_ready(&mut self, id: TileId) -> bool {
        let key = self.ram_key(id);
        let retiring = self.presented.as_ref()
            .and_then(|old| old.pixels.as_ref()).map_or(0, |pixels| pixels.retirement_credit());
        match self.ram.get(&key) {
            Some(RamEntry::Processed(work)) => work.budget().can_promote_after(retiring).is_ok(),
            _ => true,
        }
    }
    #[cfg(feature = "ofx")]
    fn promote_after_upload(&mut self, id: TileId) -> bool {
        let key = self.ram_key(id);
        let work = match self.ram.get(&key) {
            Some(RamEntry::Processed(work)) => work.clone(),
            _ => return true,
        };
        // Already retained work (including finer uploads of one whole-level run)
        // must not destroy the accepted snapshot on every redraw.
        match work.budget().can_promote_after(0) {
            Ok(()) => {}
            Err(super::region::RegionError::Pending) => {
                // Pre-upload feasibility counted only uniquely releasable old owners.
                self.presented = None;
            }
            Err(error) => {
                log::error!("processed promotion refused: {error}");
                return false;
            }
        }
        match work.budget().promote() {
            Ok(()) => {
                if self.source.as_ref().is_some_and(|source| Some(source.unit(id)) == self.completed_candidate) {
                    self.completed_candidate = None;
                }
                true
            },
            Err(error) => { log::error!("accepted processed promotion lost admission: {error}"); false }
        }
    }

    /// Keep the top level pinned: it is the fallback of every tile.
    pub fn pin_top(&mut self) {
        let Some(grid) = self.grid else { return };
        let top = grid.levels() - 1;
        let (c, r) = grid.tiles(top);
        for ty in 0..r {
            for tx in 0..c {
                let id = TileId { level: top, tx, ty };
                let key = self.ram_key(id);
                self.ram.set_pinned(&key, true);
            }
        }
    }

    /// True once after each new generation: time to drop the old generation's
    /// tiles from the atlas so their slots come back at once.
    pub fn take_new_generation(&mut self) -> bool {
        let fresh = self.atlas_generation != self.generation;
        self.atlas_generation = self.generation;
        fresh
    }

    /// Tiles in flight.
    /// RAM held by the tile cache, and its budget (the status bar).
    pub fn ram(&self) -> (usize, usize) {
        (self.ram.used(), self.ram.budget())
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }
}

impl Drop for TileState {
    fn drop(&mut self) { self.context.cancel(); }
}

/// Screen placement: the image's level-0 pixel `(0, 0)` lands at
/// `origin` (points) and one level-0 pixel is `zoom` points wide; `clip` is the
/// callback rect (NDC `[-1, 1]` of the paint pass).
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Screen position of level-0 pixel `(0, 0)`, points.
    pub origin: egui::Pos2,
    /// Points per level-0 pixel.
    pub zoom: f32,
    /// The paint callback rect, points.
    pub clip: egui::Rect,
}

impl Placement {
    /// The level-0 view rect `[x0, y0, x1, y1)` visible through `clip`.
    pub fn view(&self) -> [f32; 4] {
        let z = self.zoom.max(f32::MIN_POSITIVE);
        [
            (self.clip.min.x - self.origin.x) / z,
            (self.clip.min.y - self.origin.y) / z,
            (self.clip.max.x - self.origin.x) / z,
            (self.clip.max.y - self.origin.y) / z,
        ]
    }

    /// NDC `[x0, y0, x1, y1]` of a level-0 rect.
    pub(super) fn ndc(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> [f32; 4] {
        let sx = |x: u32| {
            let p = self.origin.x + x as f32 * self.zoom;
            (p - self.clip.min.x) / self.clip.width().max(1.0) * 2.0 - 1.0
        };
        let sy = |y: u32| {
            let p = self.origin.y + y as f32 * self.zoom;
            1.0 - (p - self.clip.min.y) / self.clip.height().max(1.0) * 2.0
        };
        [sx(x0), sy(y0), sx(x1), sy(y1)]
    }
}

/// Turn picks into instances grouped by page. `slot` gives each `from` tile's
/// atlas slot; `page_size` normalises texels to UV.
pub fn draw_list(
    grid: &Grid,
    picks: &[Pick],
    place: Placement,
    page_size: u32,
    mut slot: impl FnMut(TileId) -> Option<Slot>,
) -> DrawList {
    let mut by_page: Vec<(u32, Instance)> = Vec::with_capacity(picks.len());
    let p = page_size.max(1) as f32;
    for pick in picks {
        let (Some(src), Some(s)) = (grid.source_rect(pick.id), slot(pick.from)) else { continue };
        let [tx0, ty0, tx1, ty1] = pick.texels;
        let mut dst = place.ndc(src.x, src.y, src.x + src.w, src.y + src.h);
        let origin = place.ndc(0, 0, 0, 0)[0];
        let aspect = grid.display_aspect();
        dst[0] = origin + (dst[0] - origin) * aspect;
        dst[2] = origin + (dst[2] - origin) * aspect;
        by_page.push((
            s.page,
            Instance {
                dst,
                uv: [
                    (s.x as f32 + tx0) / p,
                    (s.y as f32 + ty0) / p,
                    (s.x as f32 + tx1) / p,
                    (s.y as f32 + ty1) / p,
                ],
            },
        ));
    }
    by_page.sort_by_key(|(page, _)| *page);
    let mut list = DrawList::default();
    for (page, inst) in by_page {
        let i = list.instances.len() as u32;
        match list.batches.last_mut() {
            Some((p, r)) if *p == page => r.end = i + 1,
            _ => list.batches.push((page, i..i + 1)),
        }
        list.instances.push(inst);
    }
    list
}

/// One frame of the tile path, GPU side: plan against the atlas, upload what
/// the RAM cache has (up to `upload_budget` bytes), and build the draw list.
/// Requests the visible tiles (plus a prefetch ring) and the top level.
pub fn frame(
    state: &mut TileState,
    atlas: &mut TileAtlas<TileKey>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    page_bgl: &wgpu::BindGroupLayout,
    place: Placement,
    upload_budget: usize,
) -> DrawList {
    atlas.end_frame();
    let Some(grid) = state.prepare_frame(place) else {
        return DrawList::default();
    };
    let generation = state.generation();
    let level = grid.pick_level(place.zoom);
    let view = grid.display_view(place.view());
    let top = grid.levels() - 1;

    let held = |state: &TileState, atlas: &mut TileAtlas<TileKey>| {
        state.presented.as_ref().map_or_else(DrawList::default, |old| {
            let held = Placement { clip: place.clip, ..old.place };
            draw_list(&old.grid, &old.picks, held, atlas.layout.size, |id| {
                atlas.resident(old.keys.get(&id)?)
            })
        })
    };
    let overview = TileId { level: top, tx: 0, ty: 0 };
    let want = grid.visible(level, view, 0);
    let overview_key = state.tile_key(overview);
    // Keep the fallback pinned even when every prospective fine tile is ready.
    if state.atlas_generation != generation || atlas.resident(&overview_key).is_none() {
        #[cfg(feature = "ofx")]
        if !state.promotion_ready(overview) { return held(state, atlas); }
        if !state.has_cached(overview) { return held(state, atlas); }
        // Commit the complete overview separately, even on a one-slot atlas.
        let picks = plan(&grid, &want, |id| id == overview);
        let prepared = match state.prepare_presentation(grid, picks, place) {
            Ok(shown) => shown,
            Err(error) => {
                log::error!("overview preflight refused: {error}");
                state.failed = true;
                return held(state, atlas);
            }
        };
        let key = prepared.keys[&overview];
        if atlas.resident(&key).is_none() {
            let Some(tile) = state.cached(overview) else { return held(state, atlas); };
            if atlas.upload(device, queue, page_bgl, key, &tile).is_none() {
                return held(state, atlas);
            }
        }
        #[cfg(feature = "ofx")]
        if !state.promote_after_upload(overview) { return DrawList::default(); }
        state.take_new_generation();
        state.presented = Some(prepared);
        atlas.retain(|key| key.source == generation);
    }

    // Validate prospective refinement before uploads. Captured owners and keys
    // remain stable even if an upload error changes the RAM cache.
    let mut planned = HashSet::new();
    let mut planned_bytes = 0usize;
    let prospective = plan(&grid, &want, |id| {
        let key = state.tile_key(id);
        if atlas.resident(&key).is_some() || planned.contains(&id) { return true; }
        #[cfg(feature = "ofx")]
        if !state.promotion_ready(id) { return false; }
        if !state.has_cached(id) { return false; }
        let Some(rect) = grid.rect(id) else { return false; };
        let Some(bytes) = u64::from(rect.w).checked_mul(u64::from(rect.h))
            .and_then(|pixels| pixels.checked_mul(8))
            .and_then(|bytes| usize::try_from(bytes).ok()) else { return false; };
        if bytes > upload_budget.saturating_sub(planned_bytes) { return false; }
        planned_bytes += bytes;
        planned.insert(id);
        true
    });
    let mut prepared = match state.prepare_presentation(grid, prospective, place) {
        Ok(shown) => shown,
        Err(error) => {
            log::error!("refinement preflight refused: {error}");
            state.failed = true;
            return held(state, atlas);
        }
    };
    planned.clear();
    for pick in &prepared.picks {
        if !planned.insert(pick.from) { continue; }
        let key = prepared.keys[&pick.from];
        if atlas.resident(&key).is_some() { continue; }
        let Some(tile) = state.cached(pick.from) else { continue; };
        // Optional refusal preserves the accepted overview and successful tiles.
        let _ = atlas.upload(device, queue, page_bgl, key, &tile);
    }
    let actual = plan(&grid, &want, |id| {
        let Some(key) = prepared.keys.get(&id) else { return false; };
        #[cfg(feature = "ofx")]
        if !state.promotion_ready(id) { return false; }
        atlas.resident(key).is_some()
    });
    if prepared.picks != actual {
        match state.reselect_presentation(prepared, actual) {
            Ok(shown) => prepared = shown,
            Err(error) => {
                log::error!("uploaded refinement refused: {error}");
                state.failed = true;
                return held(state, atlas);
            }
        }
    }
    #[cfg(feature = "ofx")]
    {
        planned.clear();
        for pick in &prepared.picks {
            if planned.insert(pick.from) && !state.promote_after_upload(pick.from) {
                return held(state, atlas);
            }
        }
    }
    state.presented = Some(prepared);
    held(state, atlas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::grid::TILE;
    use crate::tiles::reduce::tile_from_rgba;
    use crate::tiles::source::FrameSource;


    #[cfg(feature = "ofx")]
    #[test]
    fn optional_fine_upload_budget_preserves_completed_whole_owner_and_usable_overview() {
        use crate::tiles::processed::{ChargedPixelStorage, ProcessedLineage, ProcessedWork, output_frame};
        use ofx_runtime::ofx_host::OfxRectI;
        struct WholeOutput {
            grid: Grid,
            lineage: Arc<ProcessedLineage>,
        }
        impl TileSource for WholeOutput {
            fn size(&self) -> (u32, u32) { self.grid.size() }
            fn grid(&self) -> Grid { self.grid }
            fn processed(&self) -> bool { true }
            fn processed_lineage(&self) -> Option<Arc<ProcessedLineage>> { Some(self.lineage.clone()) }
            fn unit(&self, _: TileId) -> Unit { Unit::Level }
            fn produce(&self, _: &Grid, _: Unit, _: &ProductionContext) -> Production {
                Production::Cancelled
            }
            fn scope_level(&self, _: &Grid, _: u64, _: u32) -> Option<(u32, Arc<Vec<f32>>)> { None }
        }
        let grid = Grid::on_lattice(513, 33, (0, -33)).unwrap();
        let fine = TileId { level: 0, tx: 0, ty: 0 };
        let top = TileId { level: grid.levels() - 1, tx: 0, ty: 0 };
        let bounds = OfxRectI { x1: 0, y1: 0, x2: 513, y2: 33 };
        let original = ChargedPixelStorage::required_bytes(bounds).unwrap();
        let top_bytes = ProcessedWork::upload_bytes(&grid, top, 0).unwrap();
        let fine_bytes = ProcessedWork::upload_bytes(&grid, fine, 0).unwrap();
        assert!(fine_bytes > top_bytes);
        let workspace = original.checked_add(top_bytes).unwrap();
        let capacity = workspace.checked_mul(2).unwrap();
        let mut state = TileState::new(capacity);
        let budget = state.region_budget();

        // A real older allocation prevents R promotion while the complete new
        // owner still has exactly enough W headroom for the overview upload.
        let old_attempt = budget.begin_attempt().unwrap();
        let old_width = i32::try_from(top_bytes / 16 + 1).unwrap();
        let old_pixels = ChargedPixelStorage::allocate(
            OfxRectI { x1: 0, y1: 0, x2: old_width, y2: 1 }, &old_attempt).unwrap();
        old_attempt.promote().unwrap();
        drop(old_attempt);
        let attempt = budget.begin_attempt().unwrap();
        let pixels = ChargedPixelStorage::allocate(bounds, &attempt).unwrap();
        let frame = output_frame(pixels, bounds, bounds, 1.0).unwrap();
        let work = ProcessedWork::from_output(frame, 0, 0.0).unwrap();
        drop(attempt);
        assert!(matches!(work.budget().promote(), Err(super::super::region::RegionError::Pending)));
        assert!(matches!(work.upload(&grid, fine),
            Err(super::super::region::RegionError::Budget { needed, limit })
                if needed == original + fine_bytes && limit == workspace));
        state.set_source(SourceDesc { frame: 0, view: Selection::default(), pixels_gen: 0, stage: 1 },
            Arc::new(WholeOutput { grid, lineage: work.lineage().clone() }));
        let key = RamKey::Processed { source: state.generation(), unit: Unit::Level };
        state.ram.insert(key.clone(), RamEntry::Processed(work.clone()), work.bytes());
        state.ram.set_pinned(&key, true);
        state.completed_candidate = Some(Unit::Level);
        let baseline = budget.used();
        assert!(state.cached(fine).is_none());
        assert!(!state.failed, "optional view refusal is not a failed physical output");
        assert!(!state.stopped_units.contains(&Unit::Level));
        assert_eq!(state.completed_candidate, Some(Unit::Level));
        let Some(RamEntry::Processed(cached)) = state.ram.get(&key) else {
            panic!("optional upload Budget removed the completed whole owner");
        };
        assert!(Arc::ptr_eq(cached, &work));
        assert_eq!(cached.run, work.run);
        let overview = state.cached(top).expect("the smaller admitted overview remains usable");
        assert_eq!((overview.w, overview.h), grid.level_size(top.level));
        assert!(budget.used() <= capacity);
        drop(overview);
        assert_eq!(budget.used(), baseline);
        assert_eq!(state.scheduled_units, 0, "upload fallback does not invoke a producer");
        drop(work);
        drop(state);
        drop(old_pixels);
        assert_eq!(budget.used(), 0, "last physical allocation releases both R and W");
    }

    struct TestSource {
        mode: std::sync::atomic::AtomicUsize,
        epoch: std::sync::atomic::AtomicU64,
        budget_epoch: std::sync::atomic::AtomicU64,
        calls: std::sync::atomic::AtomicUsize,
        started: Sender<ProductionContext>,
        gate: Option<Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>>,
    }
    impl TileSource for TestSource {
        fn size(&self) -> (u32, u32) {
            (1, 1)
        }
        fn unit(&self, id: TileId) -> Unit {
            Unit::Tile(id)
        }
        fn readiness_epoch(&self) -> u64 {
            self.epoch.load(std::sync::atomic::Ordering::Acquire)
        }
        fn readiness(&self) -> Readiness {
            Readiness { inputs: self.readiness_epoch(), budget: self.budget_epoch.load(std::sync::atomic::Ordering::Acquire) }
        }
        fn produce(&self, grid: &Grid, unit: Unit, context: &ProductionContext) -> Production {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if self.started.send(context.clone()).is_err() {
                log::trace!("fixture observer detached");
            }
            if call == 0
                && let Some(gate) = &self.gate
            {
                let ready = gate.0.lock().unwrap();
                drop(
                    gate.1
                        .wait_timeout_while(ready, std::time::Duration::from_secs(10), |ready| {
                            !*ready
                        })
                        .unwrap(),
                );
            }
            match self.mode.load(std::sync::atomic::Ordering::Acquire) {
                1 => Production::Pending,
                2 => Production::Failed("permanent fixture failure".into()),
                3 => Production::Cancelled,
                _ => {
                    // Deliberately ignore cancellation: scheduler admission must
                    // protect against a source that finishes after retirement.
                    let Unit::Tile(id) = unit else {
                        panic!("single tile fixture");
                    };
                    Production::Ready(vec![(id, tile_from_rgba(&[1.0; 4], grid, id).unwrap())])
                }
            }
        }
        fn scope_level(&self, _: &Grid, _: u64, _: u32) -> Option<(u32, Arc<Vec<f32>>)> {
            None
        }
    }
    struct ReleaseGate(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
    impl Drop for ReleaseGate {
        fn drop(&mut self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
    }
    fn fixture(
        mode: usize,
        gated: bool,
    ) -> (
        Arc<TestSource>,
        Receiver<ProductionContext>,
        Option<ReleaseGate>,
    ) {
        let (started, rx) = channel();
        let gate =
            gated.then(|| Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())));
        let release = gate.as_ref().map(|gate| ReleaseGate(Arc::clone(gate)));
        (
            Arc::new(TestSource {
                mode: mode.into(),
                epoch: 0.into(),
                budget_epoch: 0.into(),
                calls: 0.into(),
                started,
                gate,
            }),
            rx,
            release,
        )
    }
    fn identity(frame: u64, stage: u64) -> SourceDesc {
        SourceDesc {
            frame,
            stage,
            view: Default::default(),
            pixels_gen: 0,
        }
    }
    fn one_tile() -> TileId {
        TileId {
            level: 0,
            tx: 0,
            ty: 0,
        }
    }
    fn complete(state: &mut TileState) -> usize {
        let done = state
            .rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        state.tx.send(done).unwrap();
        state.drain()
    }

    #[test]
    fn healthy_token_abort_is_terminal_without_a_retry_loop_and_playback_advances() {
        let (source, _started, _) = fixture(3, false);
        let mut state = TileState::new(100);
        state.set_source(identity(1, 1), source.clone());
        state.request([one_tile()]);
        assert_eq!(complete(&mut state), 0);
        assert!(
            !state.context.is_cancelled(),
            "source abort is not generation retirement"
        );
        for _ in 0..100 {
            state.request([one_tile()]);
        }
        assert_eq!(state.scheduled_units, 1);
        assert_eq!(state.pending(), 0);
        assert!(state.can_advance(1));
        assert!(state.cached(one_tile()).is_none());
        state.set_source(identity(1, 2), source);
        state.request([one_tile()]);
        complete(&mut state);
        assert_eq!(
            state.scheduled_units, 2,
            "new immutable revision may try again"
        );
    }

    #[test]
    fn permanent_unit_failure_is_scheduled_once_and_playback_advances() {
        let (source, _, _) = fixture(2, false);
        let mut state = TileState::new(100);
        state.set_source(identity(10, 0), source.clone());
        state.request([one_tile(), one_tile()]);
        assert_eq!(complete(&mut state), 0);
        for _ in 0..100 {
            state.request([one_tile()]);
        }
        assert_eq!(state.scheduled_units, 1);
        assert_eq!(source.calls.load(std::sync::atomic::Ordering::Acquire), 1);
        assert_eq!(state.pending(), 0);
        assert!(state.can_advance(10));
        state.set_source(identity(11, 0), source);
        assert!(!state.can_advance(11));
        state.request([one_tile()]);
        complete(&mut state);
        assert_eq!(state.scheduled_units, 2);
        state.set_source(
            identity(12, 0),
            Arc::new(FrameSource {
                rgba: Arc::new(vec![1.0; 4]),
                w: 1,
                h: 1,
            }),
        );
        state.retire_cpu_source();
        assert!(
            state.can_advance(12),
            "failed replacement materialization has no producer"
        );
    }

    #[test]
    fn pending_retries_only_after_readiness_epoch_and_does_not_repaint_forever() {
        let (source, started, _) = fixture(1, false);
        let mut state = TileState::new(100);
        state.set_source(identity(1, 0), source.clone());
        state.request([one_tile()]);
        started
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        // Readiness can advance between admission and arrival. The pending
        // outcome belongs to the captured admission epoch, not the latest one.
        source.epoch.store(1, std::sync::atomic::Ordering::Release);
        complete(&mut state);
        state.request([one_tile()]);
        complete(&mut state);
        for _ in 0..100 {
            state.request([one_tile()]);
        }
        assert_eq!(state.scheduled_units, 2);
        assert_eq!(
            state.pending(),
            0,
            "viewport repaint count excludes deferred input"
        );
        assert_eq!(state.waiting.len(), 1);
        assert!(!state.failed);
        source.mode.store(0, std::sync::atomic::Ordering::Release);
        source.epoch.store(2, std::sync::atomic::Ordering::Release);
        state.request([one_tile(), one_tile()]);
        assert_eq!(complete(&mut state), 1);
        state.request([one_tile()]);
        assert_eq!(state.scheduled_units, 3);
        assert!(state.cached(one_tile()).is_some());
        assert!(state.waiting.is_empty());
    }

    #[test]
    fn simultaneous_budget_and_input_progress_cannot_collide() {
        let (source, _, _) = fixture(1, false);
        let mut state=TileState::new(100);
        state.set_source(identity(1,0),source.clone());
        state.request([one_tile()]);
        complete(&mut state);
        assert_eq!(state.scheduled_units,1);
        source.epoch.store(1,std::sync::atomic::Ordering::Release);
        source.budget_epoch.store(1,std::sync::atomic::Ordering::Release);
        state.request([one_tile()]);
        complete(&mut state);
        assert_eq!(state.scheduled_units,2,"equal changes would collide under XOR");
        for _ in 0..20 {state.request([one_tile()]);}
        assert_eq!(state.scheduled_units,2,"observations do not advance readiness");
    }

    #[test]
    fn replacement_retirement_clear_and_drop_cancel_exact_worker_token() {
        for action in 0..8 {
            let (source, started, release) = fixture(if action >= 4 { 2 } else { 0 }, true);
            let mut state = TileState::new(100);
            state.set_source(identity(1, 0), source.clone());
            state.request([one_tile()]);
            let context = started
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            assert!(Arc::ptr_eq(&context.cancelled, &state.context.cancelled));
            match action % 4 {
                0 => state.set_source(identity(2, 0), source),
                1 => state.retire_cpu_source(),
                2 => state.clear_presentation(),
                _ => {
                    drop(state);
                    assert!(context.is_cancelled());
                    drop(release);
                    continue;
                }
            }
            assert!(context.is_cancelled());
            drop(release);
            assert_eq!(complete(&mut state), 0, "obsolete Ready must not reach RAM");
            assert!(state.cached(one_tile()).is_none());
            assert!(!state.failed);
        }
    }

    #[test]
    fn direct_clear_keeps_resident_identity_and_restart_excludes_old_completion() {
        // Only dispatch runs on this pool: blocking observers stay on the test thread.
        // Two workers let the replacement finish while the obsolete worker remains gated.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("lifecycle fixture pool");
        let source = Arc::new(FrameSource {
            rgba: Arc::new(vec![1.0; 4]),
            w: 1,
            h: 1,
        });
        let mut state = TileState::new(100);
        let desc = identity(1, 0);
        state.set_source(desc.clone(), source);
        let generation = state.generation;
        for _ in 0..20 {
            state.clear_presentation();
            assert_eq!(state.description(), Some(&desc));
            assert_eq!(state.generation, generation);
            assert!(!state.is_stale(&desc));
            assert!(
                state.scope_level(1, 1).is_some(),
                "resident scopes remain usable without a new generation"
            );
        }
        assert_eq!(state.scheduled_units, 0);

        let (source, started, release) = fixture(0, true);
        state.set_source(identity(2, 0), source);
        pool.install(|| state.request([one_tile()]));
        let old_context = started
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let old_generation = state.generation;
        state.clear_presentation();
        pool.install(|| state.request([one_tile()]));
        let new_context = started
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert_eq!(state.generation, old_generation + 1);
        assert!(old_context.is_cancelled());
        assert!(!new_context.is_cancelled());
        assert!(!Arc::ptr_eq(&old_context.cancelled, &new_context.cancelled));
        // The new worker is ungated; only the old worker waits for release.
        assert_eq!(complete(&mut state), 1);
        drop(release);
        assert_eq!(complete(&mut state), 0);
        assert!(state.cached(one_tile()).is_some());
        assert!(!state.failed);
    }

    #[test]
    fn uploaded_overview_allows_offscreen_playback_without_visible_picks() {
        let source = Arc::new(FrameSource {
            rgba: Arc::new(vec![1.0; 4]),
            w: 1,
            h: 1,
        });
        let mut state = TileState::new(100);
        state.set_source(identity(20, 0), source);
        let grid = Grid::new(1, 1);
        let place = Placement {
            origin: egui::pos2(-1024.0, -1024.0),
            zoom: 1.0,
            clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1.0, 1.0)),
        };
        assert!(grid.visible(0, place.view(), 0).is_empty());
        state.request([one_tile()]);
        complete(&mut state);
        assert!(state.overview_ready());
        // frame() calls this after the overview upload, even with no visible tiles.
        assert!(state.take_new_generation());
        state.accept_frame(grid, Vec::new(), place);
        assert_eq!(state.accepted_description(), Some(&identity(20, 0)));
        assert!(
            state.can_advance(20),
            "panning offscreen must not stop the sequence"
        );
    }

    #[test]
    fn effect_revision_retains_accepted_identity_until_usable_overview_and_selection_clears() {
        let source = Arc::new(FrameSource {
            rgba: Arc::new(vec![1.0; 4]),
            w: 1,
            h: 1,
        });
        let mut state = TileState::new(100);
        let grid = Grid::new(1, 1);
        let place = Placement {
            origin: egui::Pos2::ZERO,
            zoom: 1.0,
            clip: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1.0, 1.0)),
        };
        let picks = || plan(&grid, &[one_tile()], |_| true);
        state.set_source(identity(1, 1), source.clone());
        state.request([one_tile()]);
        complete(&mut state);
        assert!(state.take_new_generation());
        state.accept_frame(grid, picks(), place);
        assert_eq!(state.accepted_description(), Some(&identity(1, 1)));
        state.set_source(identity(1, 2), source.clone());
        state.accept_frame(grid, picks(), place);
        assert_eq!(
            state.accepted_description(),
            Some(&identity(1, 1)),
            "not accepted before overview"
        );
        state.request([one_tile()]);
        complete(&mut state);
        state.accept_frame(grid, picks(), place);
        assert_eq!(
            state.accepted_description(),
            Some(&identity(1, 1)),
            "CPU-ready overview is not yet uploaded"
        );
        assert!(state.take_new_generation());
        state.accept_frame(grid, Vec::new(), place);
        assert_eq!(state.accepted_description(), Some(&identity(1, 2)));
        let (failed, _, _) = fixture(2, false);
        state.set_source(identity(1, 3), failed);
        state.request([one_tile()]);
        complete(&mut state);
        assert_eq!(
            state.accepted_description(),
            Some(&identity(1, 2)),
            "failure retains last accepted revision"
        );
        assert!(state.can_advance(1));
        let mut selection = identity(1, 4);
        selection.view.layer = "depth".into();
        state.set_source(selection, source);
        assert!(state.accepted_description().is_none());
        state.clear_presentation();
        assert!(state.accepted_description().is_none());
    }

    #[test]
    fn plan_uses_the_tile_itself_when_resident() {
        let g = Grid::new(2048, 1024);
        let id = TileId { level: 0, tx: 1, ty: 1 };
        let p = plan(&g, &[id], |_| true);
        assert_eq!(p, [Pick { id, from: id, texels: [0.0, 0.0, TILE as f32, TILE as f32] }]);
    }

    #[test]
    fn plan_falls_back_to_the_nearest_resident_ancestor() {
        let g = Grid::new(2048, 1024); // levels: 2048, 1024, 512
        let id = TileId { level: 0, tx: 3, ty: 1 };
        // Only the top level (one 512x256 tile) is resident.
        let p = plan(&g, &[id], |t| t.level == 2);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].from, TileId { level: 2, tx: 0, ty: 0 });
        // Tile (3,1) covers level-0 [1536,2048) x [512,1024): level 2 texels /4.
        assert_eq!(p[0].texels, [384.0, 128.0, 512.0, 256.0]);
        // With level 1 resident, the parent is preferred.
        let p = plan(&g, &[id], |t| t.level >= 1);
        assert_eq!(p[0].from, TileId { level: 1, tx: 1, ty: 0 });
        assert_eq!(p[0].texels, [256.0, 256.0, 512.0, 512.0]);
    }

    #[test]
    fn plan_skips_tiles_with_nothing_resident() {
        let g = Grid::new(2048, 1024);
        assert!(plan(&g, &[TileId { level: 0, tx: 0, ty: 0 }], |_| false).is_empty());
    }

    #[test]
    fn plan_handles_partial_edge_tiles() {
        let g = Grid::new(1100, 600); // level 1 is 550x300 (2 tiles across)
        let id = TileId { level: 0, tx: 2, ty: 1 }; // level-0 [1024,1100) x [512,600)
        let p = plan(&g, &[id], |t| t.level == 1);
        assert_eq!(p[0].from, TileId { level: 1, tx: 1, ty: 0 });
        assert_eq!(p[0].texels, [0.0, 256.0, 38.0, 300.0]);
    }

    #[test]
    fn draw_list_groups_by_page_and_maps_to_ndc_and_uv() {
        let g = Grid::new(1024, 512);
        let place = Placement {
            origin: egui::pos2(0.0, 0.0),
            zoom: 1.0,
            clip: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1024.0, 512.0)),
        };
        let a = TileId { level: 0, tx: 0, ty: 0 };
        let b = TileId { level: 0, tx: 1, ty: 0 };
        let picks = plan(&g, &[b, a], |_| true);
        let slots = |id: TileId| {
            Some(if id == a { Slot { page: 1, x: 0, y: 0 } } else { Slot { page: 0, x: 512, y: 0 } })
        };
        let list = draw_list(&g, &picks, place, 1024, slots);
        assert_eq!(list.batches, [(0, 0..1), (1, 1..2)]);
        // b: level-0 [512,1024) x [0,512) fills the right half of the clip.
        assert_eq!(list.instances[0].dst, [0.0, 1.0, 1.0, -1.0]);
        assert_eq!(list.instances[0].uv, [0.5, 0.0, 1.0, 0.5]);
        assert_eq!(list.instances[1].dst, [-1.0, 1.0, 0.0, -1.0]);
        assert_eq!(list.instances[1].uv, [0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn placement_view_inverts_the_screen_mapping() {
        let place = Placement {
            origin: egui::pos2(100.0, 50.0),
            zoom: 0.5,
            clip: egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(400.0, 300.0)),
        };
        assert_eq!(place.view(), [-200.0, -100.0, 600.0, 500.0]);
    }

    #[test]
    fn tile_state_produces_caches_and_invalidates_by_generation() {
        let (w, h) = (1100u32, 600u32);
        let rgba: Vec<f32> = (0..w * h * 4).map(|i| (i % 7) as f32).collect();
        let src: Arc<dyn TileSource> = Arc::new(FrameSource { rgba: Arc::new(rgba.clone()), w, h });
        let desc = |frame| SourceDesc { frame, view: Selection::default(), pixels_gen: 0, stage: 0 };
        let mut s = TileState::new(usize::MAX);
        assert!(s.is_stale(&desc(1)));
        s.set_source(desc(1), Arc::clone(&src));
        assert!(!s.is_stale(&desc(1)));
        let id = TileId { level: 0, tx: 2, ty: 1 };
        s.request([id, id]);
        assert_eq!(s.pending(), 1, "one request per unit");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while s.cached(id).is_none() && std::time::Instant::now() < deadline {
            s.drain();
            std::thread::yield_now();
        }
        let grid = *s.grid().unwrap();
        assert_eq!(s.cached(id).as_deref(), tile_from_rgba(&rgba, &grid, id).as_ref());
        s.request([id]);
        assert_eq!(s.pending(), 0, "cached tiles are not requested again");

        s.set_source(desc(2), src);
        assert!(s.cached(id).is_none(), "a new generation drops the old tiles");
    }

    #[test]
    fn playback_waits_only_for_the_matching_unfinished_source() {
        let mut state = TileState::new(1);
        assert!(state.can_advance(10), "no source: a materialization failure cannot freeze playback");
        let desc = SourceDesc { frame: 10, view: Default::default(), pixels_gen: 0, stage: 0 };
        let source = Arc::new(FrameSource { rgba: Arc::new(vec![1.0; 4]), w: 1, h: 1 });
        state.set_source(desc, source);
        assert!(!state.can_advance(10), "this source needs a presentation");
        assert!(state.can_advance(11), "a failed source creation leaves an unrelated generation");
        state.failed = true;
        assert!(state.can_advance(10), "decode failure lets the following frame proceed");
    }

    #[test]
    fn overview_survives_later_arrivals_under_a_tiny_ram_budget() {
        let mut state = TileState::new(1);
        let grid = Grid::new(1024, 512);
        let top = TileId { level: 1, tx: 0, ty: 0 };
        let fine = TileId { level: 0, tx: 0, ty: 0 };
        let source = Arc::new(FrameSource { rgba: Arc::new(vec![1.0; 1024 * 512 * 4]), w: 1024, h: 512 });
        state.set_source(SourceDesc { frame: 1, view: Default::default(), pixels_gen: 0, stage: 0 }, source.clone());
        state.tx.send((state.generation(), Unit::Tile(top), Readiness::default(), state.context.clone(), source.produce(&grid, Unit::Tile(top), &state.context))).unwrap();
        state.tx.send((state.generation(), Unit::Tile(fine), Readiness::default(), state.context.clone(), source.produce(&grid, Unit::Tile(fine), &state.context))).unwrap();
        state.drain();
        assert!(state.cached(top).is_some(), "overview pinned before another insertion can evict it");
    }
}
