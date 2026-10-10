//! Tile sources (`.bughunt/design_tiling.md` §3.2): where tile pixels come
//! from. A source produces tiles in WORK UNITS: the in-RAM frame one tile at a
//! time, a file one tile row of a level at a time (one band decode yields the
//! whole row). The scheduler ([`super::view::TileState`]) de-duplicates by
//! unit, so a row is decoded once however many of its tiles are wanted.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::grid::{Grid, TileId};
use super::reduce::{TileData, tile_from_rgba};

/// A unit of tile production.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    /// One tile.
    Tile(TileId),
    /// Every tile of one tile row of one level.
    Row {
        /// Level.
        level: u32,
        /// Tile row.
        ty: u32,
    },
    /// One complete non-tileable effect result, independent of requested zoom.
    #[cfg(feature = "ofx")]
    Level,
    /// Every tile of every level from `from` up: the coarse overview levels,
    /// built by one streaming pass over the file.
    Overview {
        /// The finest level of the overview.
        from: u32,
    },
}

/// Immutable worker context. Cancellation belongs to its generation, never to
/// a reusable source.
#[derive(Clone, Default)]
pub struct ProductionContext {
    pub(crate) cancelled: Arc<AtomicBool>,
}

impl ProductionContext {
    /// Whether the owning generation has retired.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn finish(&self, result: Result<Vec<(TileId, TileData)>, String>) -> Production {
        if self.is_cancelled() {
            Production::Cancelled
        } else {
            match result {
                Ok(tiles) => Production::Ready(tiles),
                Err(error) => Production::Failed(error),
            }
        }
    }
}

/// A unit's outcome, carried on the scheduler's existing result channel.
pub enum Production {
    Ready(Vec<(TileId, TileData)>),
    #[cfg(feature = "ofx")]
    Processed(Arc<super::processed::ProcessedWork>),
    #[cfg(feature = "ofx")]
    Refused(super::region::RegionError),
    /// Input is not ready. Retry only after the source's readiness epoch changes.
    #[cfg(any(feature = "ofx", test))]
    Pending,
    Cancelled,
    /// Permanent for this generation and unit.
    Failed(String),
}

/// Pure observation of independent readiness events, without lossy counter mixing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Readiness {
    pub inputs: u64,
    pub budget: u64,
}

/// Something that can produce the pyramid's tiles of one image.
pub trait TileSource: Send + Sync {
    /// Level-0 extent (the data window).
    fn size(&self) -> (u32, u32);

    /// Geometry belongs to the source; processed output uses a signed lattice.
    fn grid(&self) -> Grid { let (w, h) = self.size(); Grid::new(w, h) }

    #[cfg(feature = "ofx")]
    fn processed(&self) -> bool { false }

    /// Immutable provenance of this generation; metadata never pins decoded pixels.
    #[cfg(feature = "ofx")]
    fn processed_lineage(&self) -> Option<Arc<super::processed::ProcessedLineage>> { None }

    /// The unit that produces `id`.
    fn unit(&self, id: TileId) -> Unit;

    /// Produce `unit`'s tiles (runs on a worker thread).
    fn produce(&self, grid: &Grid, unit: Unit, context: &ProductionContext) -> Production;

    /// Original f32 region under a caller-owned ledger. The ledger is shared
    /// across concurrent renders and retained outputs; consumers attach in module 3.
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        expect(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    fn read_region(
        &self,
        _plan: super::region::RegionPlan,
        _context: &ProductionContext,
        _budget: &super::region::RegionBudget,
    ) -> Result<Arc<super::region::RegionData>, super::region::RegionError> {
        Err(super::region::RegionError::Unsupported)
    }

    /// Changes when a previously pending input may be ready. Its owner must wake the UI.
    fn readiness_epoch(&self) -> u64 {
        0
    }

    /// Captured before a work attempt so a release during production cannot be lost.
    fn readiness(&self) -> Readiness {
        Readiness { inputs: self.readiness_epoch(), budget: 0 }
    }

    /// The finest level whose image this source can hand to the scopes now,
    /// as `(level, w * h * 4 RGBA)`; `None` when nothing is ready yet.
    fn scope_level(
        &self,
        grid: &Grid,
        max_pixels: u64,
        max_dim: u32,
    ) -> Option<(u32, Arc<Vec<f32>>)>;
}

/// The in-RAM materialized frame (phase 1): level-0 RGBA `f32`.
#[derive(Clone)]
pub struct FrameSource {
    /// `w * h * 4` linear RGBA.
    pub rgba: Arc<Vec<f32>>,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl FrameSource {
    /// Checked original f32 region; no half/display texture enters this path.
    pub fn region(
        &self,
        plan: super::region::RegionPlan,
        context: &ProductionContext,
        budget: &super::region::RegionBudget,
    ) -> Result<Arc<super::region::RegionData>, super::region::RegionError> {
        if plan.policy() == super::region::RegionPolicy::MatchingStoredNative && plan.level() > 0 {
            return Err(super::region::RegionError::Unsupported);
        }
        let result = self.region_pixels(plan, context, budget);
        super::region::RegionError::from_result(result, context.is_cancelled())
    }

    fn region_pixels(
        &self,
        plan: super::region::RegionPlan,
        context: &ProductionContext,
        budget: &super::region::RegionBudget,
    ) -> Result<Arc<super::region::RegionData>, super::region::RegionError> {
        if plan.size() != (self.w, self.h) {
            return Err("region source geometry does not match frame".into());
        }
        if context.is_cancelled() {
            return Err("region cancelled".into());
        }
        let bytes = plan
            .output_bytes()
            .checked_add(plan.sums_bytes())
            .and_then(|n| n.checked_add(plan.rect().w as usize * 16))
            .ok_or("region working size overflow")?;
        let permit = budget.reserve(bytes)?;
        let rgba = super::region::resident(&self.rgba, plan, || context.is_cancelled())?;
        super::region::RegionData::finish(plan, rgba, permit)
    }
}

impl TileSource for FrameSource {
    fn read_region(
        &self,
        plan: super::region::RegionPlan,
        context: &ProductionContext,
        budget: &super::region::RegionBudget,
    ) -> Result<Arc<super::region::RegionData>, super::region::RegionError> {
        self.region(plan, context, budget)
    }
    fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    fn unit(&self, id: TileId) -> Unit {
        Unit::Tile(id)
    }

    fn produce(&self, grid: &Grid, unit: Unit, context: &ProductionContext) -> Production {
        if context.is_cancelled() {
            return Production::Cancelled;
        }
        context.finish((|| {
            let Unit::Tile(id) = unit else {
                return Err(format!(
                    "the frame source produces single tiles, not {unit:?}"
                ));
            };
            let tile = tile_from_rgba(&self.rgba, grid, id)
                .ok_or_else(|| format!("tile {id:?} is outside the {}x{} frame", self.w, self.h))?;
            Ok(vec![(id, tile)])
        })())
    }

    fn scope_level(
        &self,
        grid: &Grid,
        max_pixels: u64,
        max_dim: u32,
    ) -> Option<(u32, Arc<Vec<f32>>)> {
        let level = finest_level(grid, max_pixels, max_dim);
        if level == 0 {
            return Some((0, Arc::clone(&self.rgba)));
        }
        super::reduce::level_from_rgba(&self.rgba, grid, level).map(|v| (level, Arc::new(v)))
    }
}

/// The finest level within `max_pixels` and the device's texture limit.
pub fn finest_level(grid: &Grid, max_pixels: u64, max_dim: u32) -> u32 {
    (0..grid.levels())
        .find(|&k| {
            let (w, h) = grid.level_size(k);
            u64::from(w) * u64::from(h) <= max_pixels && w <= max_dim && h <= max_dim
        })
        .unwrap_or(grid.levels() - 1)
}
