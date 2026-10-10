//! Immutable accepted effect pixels shared by readout, scopes and export.
use std::collections::{BTreeMap, BinaryHeap};
use std::cmp::Reverse;
use std::sync::Arc;
use std::mem::size_of;
use crate::tiles::grid::{Grid, PixRect};
use crate::tiles::processed::{CanvasSample, ProcessedLineage, ProcessedView, ProcessedWork};
use crate::tiles::view::{Pick, SourceDesc, TileKey};
use crate::tiles::region::{Permit, RegionAccumulator, RegionBudget, RegionError, RegionPolicy};

pub(crate) struct AcceptedCoverage {
    pub(crate) pick: Pick,
    pub(crate) key: TileKey,
    pub(crate) view: ProcessedView,
}
struct Coverage {
    key: TileKey,
    owner: usize,
    rect: PixRect,
    view_rect: PixRect,
}
struct AcceptedInner {
    revision: u64,
    desc: SourceDesc,
    grid: Grid,
    lineage: Arc<ProcessedLineage>,
    // Exactly one internal Arc per physical work. Coverage stores indices only.
    owners: Vec<Arc<ProcessedWork>>,
    coverage: Vec<Coverage>,
}

/// Cloning pins the accepted mosaic without copying its pixels.
#[derive(Clone)]
pub(crate) struct AcceptedPixels(Arc<AcceptedInner>);

fn intersect(a: PixRect, b: PixRect) -> Option<PixRect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = a.x.checked_add(a.w)?.min(b.x.checked_add(b.w)?);
    let bottom = a.y.checked_add(a.h)?.min(b.y.checked_add(b.h)?);
    (x < right && y < bottom).then_some(PixRect { x, y, w: right - x, h: bottom - y })
}
fn contains(rect: PixRect, x: u32, y: u32) -> bool {
    x >= rect.x && y >= rect.y && x - rect.x < rect.w && y - rect.y < rect.h
}

impl AcceptedPixels {
    pub(crate) fn new(
        revision: u64, desc: SourceDesc, grid: Grid,
        lineage: Arc<ProcessedLineage>, mut incoming: Vec<AcceptedCoverage>,
        overview: AcceptedCoverage,
    ) -> Result<Self, RegionError> {
        if desc.frame != lineage.frame_id || desc.view != lineage.selection {
            return Err("accepted source description does not match its immutable lineage".into());
        }
        crate::tiles::checked_extent(lineage.data_window)?;
        if crate::tiles::checked_extent(lineage.display_window)? != grid.size() {
            return Err("accepted canvas does not match its immutable primary display window".into());
        }
        let (w, h) = grid.size();
        let canvas = PixRect { x: 0, y: 0, w, h };
        let overview_source = overview.key.source;
        let overview_index = incoming.len();
        incoming.push(overview);
        let mut owners: Vec<Arc<ProcessedWork>> = Vec::new();
        let mut owner_indices = BTreeMap::new();
        let mut entries = Vec::with_capacity(incoming.len());
        for (paint, entry) in incoming.into_iter().enumerate() {
            let work = entry.view.work;
            if entry.key.run != work.run || entry.key.id != entry.pick.from
                || entry.key.source != overview_source
                || work.time.to_bits() != lineage.time.to_bits()
                || !Arc::ptr_eq(work.lineage(), &lineage) {
                return Err("accepted coverage does not match its physical output".into());
            }
            let view = ProcessedView { work: work.clone(), rect: entry.view.rect };
            let view_canvas = view.canvas_rect(&grid)
                .ok_or("accepted subview is outside its physical output")?;
            let desired = if paint == overview_index { canvas } else {
                grid.source_rect(entry.pick.id).ok_or("accepted pick is outside canvas")?
            };
            if grid.texture_rect(entry.pick.from, desired) != Some(entry.pick.texels) {
                return Err("accepted pick texels do not match checked display coverage".into());
            }
            let rect = intersect(desired, view_canvas)
                .ok_or("accepted pick misses its physical output")?;
            if rect != desired {
                return Err("accepted pick extends outside its physical subview".into());
            }
            if paint == overview_index && rect != canvas {
                return Err("accepted overview does not cover its canvas".into());
            }
            let owner = match owner_indices.get(&work.run).copied() {
                Some(index) => {
                    if !Arc::ptr_eq(&owners[index], &work) {
                        return Err("accepted run names different physical owners".into());
                    }
                    index
                }
                None => {
                    let index = owners.len();
                    owner_indices.insert(work.run, index);
                    owners.push(work);
                    index
                }
            };
            entries.push(((paint == overview_index, entry.pick.from.level, Reverse(paint)),
                Coverage { key: entry.key, owner, rect, view_rect: entry.view.rect }));
        }
        // Finest accepted source first; last painted pick wins equal-LOD ties.
        // The overview supplies only the uncovered remainder.
        entries.sort_unstable_by_key(|entry| entry.0);
        let coverage = entries.into_iter().map(|entry| entry.1).collect();
        Ok(Self(Arc::new(AcceptedInner { revision, desc, grid, lineage, owners, coverage })))
    }
    /// Reselect only already captured physical runs after opportunistic uploads.
    pub(crate) fn reselect(&self, revision: u64, picks: &[Pick]) -> Result<Self, RegionError> {
        let overview = self.0.coverage.last()
            .ok_or("accepted presentation has no overview")?;
        let capture = |pick: Pick, entry: &Coverage| {
            let work = self.0.owners[entry.owner].clone();
            AcceptedCoverage { pick, key: entry.key,
                view: ProcessedView { work, rect: entry.view_rect } }
        };
        let mut incoming = Vec::new();
        incoming.try_reserve_exact(picks.len()).map_err(|error| error.to_string())?;
        for pick in picks {
            let entry = self.0.coverage.iter().find(|entry| entry.key.id == pick.from)
                .ok_or("reselected pick has no captured physical run")?;
            incoming.push(capture(*pick, entry));
        }
        let (w, h) = self.0.grid.size();
        let rect = PixRect { x: 0, y: 0, w, h };
        let texels = self.0.grid.texture_rect(overview.key.id, rect)
            .ok_or("accepted overview has invalid checked texels")?;
        let pick = Pick { id: overview.key.id, from: overview.key.id, texels };
        Self::new(revision, self.0.desc.clone(), self.0.grid, self.0.lineage.clone(),
            incoming, capture(pick, overview))
    }
    pub(crate) fn revision(&self) -> u64 { self.0.revision }
    pub(crate) fn description(&self) -> &SourceDesc { &self.0.desc }
    pub(crate) fn grid(&self) -> Grid { self.0.grid }
    pub(crate) fn lineage(&self) -> &ProcessedLineage { &self.0.lineage }
    pub(crate) fn is_primary_frame(&self, frame: &crate::image::DecodedFrame) -> bool {
        let lineage = &self.0.lineage;
        frame.id == lineage.frame_id && frame.data_window == lineage.data_window
            && frame.display_window == lineage.display_window
            && f64::from(frame.pixel_aspect).to_bits() == lineage.pixel_aspect.to_bits()
    }
    /// A source point is distinct from the native processed cell covering it.
    pub(crate) fn primary_pixel(&self, x: u32, y: u32) -> Option<(u32, u32)> {
        let (w, h) = self.0.grid.size();
        if x >= w || y >= h { return None; }
        let lineage = &self.0.lineage;
        let absolute_x = i64::from(lineage.display_window.min.x).checked_add(i64::from(x))?;
        let absolute_y = i64::from(lineage.display_window.min.y).checked_add(i64::from(y))?;
        if absolute_x < i64::from(lineage.data_window.min.x)
            || absolute_x > i64::from(lineage.data_window.max.x)
            || absolute_y < i64::from(lineage.data_window.min.y)
            || absolute_y > i64::from(lineage.data_window.max.y) { return None; }
        Some((u32::try_from(absolute_x.checked_sub(i64::from(lineage.data_window.min.x))?).ok()?,
            u32::try_from(absolute_y.checked_sub(i64::from(lineage.data_window.min.y))?).ok()?))
    }
    pub(crate) fn canvas_sample(&self, x: u32, y: u32) -> Option<CanvasSample> {
        let entry = self.0.coverage.iter().find(|entry| contains(entry.rect, x, y))?;
        let sample = self.0.owners[entry.owner].canvas_sample(&self.0.grid, x, y)?;
        (sample.run == entry.key.run).then_some(sample)
    }
    pub(crate) fn retirement_credit(&self) -> usize {
        if Arc::strong_count(&self.0) != 1 { return 0; }
        self.0.owners.iter().fold(0usize, |sum, work| sum.saturating_add(work.retirement_credit()))
    }
    pub(crate) fn budget(&self) -> Result<RegionBudget, RegionError> {
        self.0.owners.first().map(|work| work.budget().physical())
            .ok_or_else(|| RegionError::Failed("accepted presentation has no owner".into()))
    }
    pub(crate) fn level(&self, max_pixels: u64, max_dim: u32) -> u32 {
        let capped = crate::tiles::source::finest_level(&self.0.grid, max_pixels, max_dim);
        let native = self.0.owners.iter().map(|work| work.level).min().unwrap_or(capped);
        capped.max(native)
    }
    fn accumulate_row(&self, level: u32, y: u32, accum: &mut RegionAccumulator,
        sweep: &mut Sweep) -> Result<(), String> {
        let grid = &self.0.grid;
        let (width, _) = grid.level_size(level);
        let first = grid.level_cell(level, 0, y).ok_or("consumer row is outside accepted geometry")?;
        let last = grid.level_cell(level, width.checked_sub(1).ok_or("empty consumer row")?, y)
            .ok_or("consumer row is outside accepted geometry")?;
        let right = last.x.checked_add(last.w).ok_or("consumer row edge overflows")?;
        let bottom = first.y.checked_add(first.h).ok_or("consumer row edge overflows")?;
        let row = PixRect { x: first.x, y: first.y, w: right - first.x, h: first.h };
        sweep.y.clear();
        sweep.y.extend([row.y, bottom]);
        for entry in &self.0.coverage {
            if let Some(rect) = intersect(entry.rect, row) {
                sweep.y.extend([rect.y, rect.y + rect.h]);
            }
        }
        sweep.y.sort_unstable();
        sweep.y.dedup();
        for band in 0..sweep.y.len().saturating_sub(1) {
            let top = sweep.y[band];
            let end = sweep.y[band + 1];
            sweep.events.clear();
            sweep.active.fill(false);
            sweep.heap.clear();
            for (priority, entry) in self.0.coverage.iter().enumerate() {
                if contains(entry.rect, entry.rect.x, top)
                    && let Some(rect) = intersect(entry.rect, row) {
                    sweep.events.push(Event { x: rect.x, priority, enter: true });
                    sweep.events.push(Event { x: rect.x + rect.w, priority, enter: false });
                }
            }
            sweep.events.sort_unstable_by_key(|event| event.x);
            let mut at = 0;
            while at < sweep.events.len() {
                let x = sweep.events[at].x;
                while at < sweep.events.len() && sweep.events[at].x == x {
                    let event = sweep.events[at];
                    sweep.active[event.priority] = event.enter;
                    if event.enter { sweep.heap.push(Reverse(event.priority)); }
                    at += 1;
                }
                while sweep.heap.peek().is_some_and(|Reverse(priority)| !sweep.active[*priority]) {
                    sweep.heap.pop();
                }
                if at == sweep.events.len() { break; }
                let next = sweep.events[at].x;
                if next <= x { continue; }
                let priority = sweep.heap.peek().ok_or("accepted mosaic has a coverage gap")?.0;
                self.add_native_cells(priority, PixRect { x, y: top, w: next - x, h: end - top }, accum)?;
            }
        }
        Ok(())
    }
    fn add_native_cells(&self, priority: usize, part: PixRect, accum: &mut RegionAccumulator)
        -> Result<(), String> {
        let entry = &self.0.coverage[priority];
        let work = &self.0.owners[entry.owner];
        let grid = &self.0.grid;
        let end_x = part.x.checked_add(part.w).ok_or("mosaic edge overflows")?;
        let end_y = part.y.checked_add(part.h).ok_or("mosaic edge overflows")?;
        let mut cy = part.y;
        while cy < end_y {
            let initial = work.canvas_sample(grid, part.x, cy).ok_or("accepted owner misses mosaic cell")?;
            let bottom = initial.cell.y.checked_add(initial.cell.h).ok_or("native cell edge overflows")?.min(end_y);
            if bottom <= cy { return Err("accepted owner has an empty native cell".into()); }
            let mut cx = part.x;
            while cx < end_x {
                let sample = work.canvas_sample(grid, cx, cy).ok_or("accepted owner misses mosaic cell")?;
                let right = sample.cell.x.checked_add(sample.cell.w).ok_or("native cell edge overflows")?.min(end_x);
                let sample_bottom = sample.cell.y.checked_add(sample.cell.h).ok_or("native cell edge overflows")?;
                if right <= cx || sample.run != entry.key.run || !contains(sample.cell, cx, cy)
                    || bottom > sample_bottom {
                    return Err("accepted owner has inconsistent native cells".into());
                }
                accum.add_cell(PixRect { x: cx, y: cy, w: right - cx, h: bottom - cy }, sample.pixel)?;
                cx = right;
            }
            cy = bottom;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Event { x: u32, priority: usize, enter: bool }
struct Sweep {
    y: Vec<u32>,
    events: Vec<Event>,
    active: Vec<bool>,
    heap: BinaryHeap<Reverse<usize>>,
}
impl Sweep {
    fn required_bytes(n: usize) -> Result<usize, RegionError> {
        let edges = n.checked_mul(2).ok_or("coverage scratch overflows")?;
        let y = edges.checked_add(2).and_then(|v| v.checked_mul(size_of::<u32>()))
            .ok_or("coverage scratch overflows")?;
        let events = edges.checked_mul(size_of::<Event>()).ok_or("coverage scratch overflows")?;
        let active = n.checked_mul(size_of::<bool>()).ok_or("coverage scratch overflows")?;
        let heap = n.checked_mul(size_of::<Reverse<usize>>()).ok_or("coverage scratch overflows")?;
        y.checked_add(events).and_then(|v| v.checked_add(active)).and_then(|v| v.checked_add(heap))
            .ok_or_else(|| "coverage scratch overflows".into())
    }
    fn new(n: usize) -> Result<Self, RegionError> {
        let edges = n.checked_mul(2).ok_or("coverage scratch overflows")?;
        let mut y = Vec::new();
        y.try_reserve_exact(edges.checked_add(2).ok_or("coverage scratch overflows")?).map_err(|e| e.to_string())?;
        let mut events = Vec::new();
        events.try_reserve_exact(edges).map_err(|e| e.to_string())?;
        let mut active = Vec::new();
        active.try_reserve_exact(n).map_err(|e| e.to_string())?;
        active.resize(n, false);
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(n).map_err(|e| e.to_string())?;
        Ok(Self { y, events, active, heap })
    }
    fn bytes(&self) -> usize {
        self.y.capacity() * size_of::<u32>() + self.events.capacity() * size_of::<Event>()
            + self.active.capacity() * size_of::<bool>() + self.heap.capacity() * size_of::<Reverse<usize>>()
    }
}

/// Reservation succeeds before a delayed consumer can retain this snapshot.
/// Every physical scratch field is dropped before its final permit.
pub(crate) struct PixelRows {
    snapshot: AcceptedPixels,
    level: u32,
    size: (u32, u32),
    accumulator: RegionAccumulator,
    sweep: Sweep,
    pub(crate) rgba: Vec<f32>,
    pub(crate) alpha: Vec<f32>,
    pub(crate) encoded: Vec<u8>,
    _permit: Permit,
}
impl PixelRows {
    pub(crate) fn new(snapshot: AcceptedPixels, level: u32) -> Result<Self, RegionError> {
        Self::with_scratch(snapshot, level, false)
    }
    pub(crate) fn for_export(snapshot: AcceptedPixels, level: u32) -> Result<Self, RegionError> {
        Self::with_scratch(snapshot, level, true)
    }
    fn with_scratch(snapshot: AcceptedPixels, level: u32, export: bool) -> Result<Self, RegionError> {
        let grid = snapshot.grid();
        let size = grid.level_size(level);
        let plan = grid.region_plan(level, PixRect { x: 0, y: 0, w: size.0, h: 1 }, RegionPolicy::OriginalArea)?;
        let width = usize::try_from(size.0).map_err(|_| "consumer width overflows")?;
        let reducer = RegionAccumulator::cells_required_bytes(plan)?;
        let sweep_bytes = Sweep::required_bytes(snapshot.0.coverage.len())?;
        let row_bytes = width.checked_mul(if export { 26 } else { 16 }).and_then(|n| n.checked_add(reducer))
            .and_then(|n| n.checked_add(sweep_bytes)).ok_or("consumer row storage overflows")?;
        let budget = snapshot.budget()?;
        if row_bytes > budget.candidate_limit() {
            return Err(RegionError::Budget { needed: row_bytes, limit: budget.candidate_limit() });
        }
        let mut permit = budget.reserve(row_bytes)?;
        // Error unwind destroys the allocations before releasing admission.
        let accumulator = RegionAccumulator::for_cells(plan)?;
        let sweep = Sweep::new(snapshot.0.coverage.len())?;
        let rgba_len = width.checked_mul(4).ok_or("consumer row storage overflows")?;
        let alpha_len = if export { width } else { 0 };
        let encoded_len = if export { width.checked_mul(6).ok_or("consumer row storage overflows")? } else { 0 };
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(rgba_len).map_err(|error| error.to_string())?;
        rgba.resize(rgba_len, 0.0);
        let mut alpha = Vec::new();
        alpha.try_reserve_exact(alpha_len).map_err(|error| error.to_string())?;
        alpha.resize(alpha_len, 0.0);
        let mut encoded = Vec::new();
        encoded.try_reserve_exact(encoded_len).map_err(|error| error.to_string())?;
        encoded.resize(encoded_len, 0);
        let actual = accumulator.bytes().checked_add(sweep.bytes())
            .and_then(|n| n.checked_add(rgba.capacity().checked_mul(4)?))
            .and_then(|n| n.checked_add(alpha.capacity().checked_mul(4)?))
            .and_then(|n| n.checked_add(encoded.capacity())).ok_or("consumer capacity overflows")?;
        if actual > budget.candidate_limit() {
            return Err(RegionError::Budget { needed: actual, limit: budget.candidate_limit() });
        }
        permit.resize(actual)?;
        Ok(Self { snapshot, level, size, accumulator, sweep, rgba, alpha, encoded, _permit: permit })
    }
    pub(crate) fn size(&self) -> (u32, u32) { self.size }
    pub(crate) fn level(&self) -> u32 { self.level }
    pub(crate) fn snapshot(&self) -> &AcceptedPixels { &self.snapshot }
    pub(crate) fn bytes(&self) -> usize {
        self.accumulator.bytes() + self.sweep.bytes() + self.rgba.capacity()*4
            + self.alpha.capacity()*4 + self.encoded.capacity()
    }
    pub(crate) fn row(&mut self, y: u32) -> Result<&[f32], RegionError> {
        if y >= self.size.1 { return Err("consumer row is outside frozen output".into()); }
        self.accumulator.restart_cells(y)?;
        self.snapshot.accumulate_row(self.level, y, &mut self.accumulator, &mut self.sweep)?;
        self.accumulator.finish_cells_into(&mut self.rgba)?;
        Ok(&self.rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Selection;
    use crate::tiles::grid::TileId;
    use crate::tiles::processed::{ChargedPixelStorage, output_frame};
    use ofx_runtime::ofx_host::{OfxRectI, PixelStorage};

    fn lineage(w: u32, h: u32) -> Arc<ProcessedLineage> {
        let window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(10, -3),
            imath_rs::V2i::new(10 + w as i32 - 1, -3 + h as i32 - 1));
        Arc::new(ProcessedLineage { primary_index: -7, frame_id: 42,
            data_window: window, display_window: window, pixel_aspect: 1.5,
            selection: Selection::default(), time: -6.75 })
    }
    fn desc(lineage: &ProcessedLineage) -> SourceDesc {
        SourceDesc { frame: lineage.frame_id, view: lineage.selection.clone(), pixels_gen: 3, stage: 8 }
    }
    fn work(grid: Grid, level: u32, budget: &RegionBudget, lineage: Arc<ProcessedLineage>,
        pixel: impl Fn(u32, u32) -> [f32; 4]) -> Arc<ProcessedWork> {
        let (x0, y0, x1, y1) = grid.lattice_bounds(level);
        let bounds = OfxRectI { x1: x0 as i32, y1: -y1 as i32, x2: x1 as i32, y2: -y0 as i32 };
        let mut storage = ChargedPixelStorage::allocate(bounds, budget).unwrap();
        let (w, h) = grid.level_size(level);
        let samples: &mut [f32] = bytemuck::cast_slice_mut(storage.bytes_mut().unwrap());
        for y in 0..h { for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            samples[i..i + 4].copy_from_slice(&pixel(x, y));
        }}
        let frame = output_frame(storage, bounds, bounds, lineage.pixel_aspect).unwrap();
        ProcessedWork::from_rendered(frame, level, lineage.time, budget.clone(), lineage).unwrap()
    }
    fn entry(grid: Grid, id: TileId, from: TileId, work: Arc<ProcessedWork>) -> AcceptedCoverage {
        let desired = grid.source_rect(id).unwrap();
        let pick = Pick { id, from, texels: grid.texture_rect(from, desired).unwrap() };
        let (w, h) = work.dimensions();
        AcceptedCoverage { pick, key: TileKey { source: 7, run: work.run, id: from },
            view: work.view(PixRect { x: 0, y: 0, w, h }).unwrap() }
    }
    fn top(grid: Grid) -> TileId { TileId { level: grid.levels() - 1, tx: 0, ty: 0 } }

    #[test]
    fn constructor_rejects_foreign_identity_texels_and_incomplete_subviews() {
        let budget = RegionBudget::new(8192);
        let mut grid = Grid::on_lattice(5, 7, (0, -7)).unwrap();
        grid.include_level(1).unwrap();
        let lineage = lineage(5, 7);
        let work = work(grid, 1, &budget, lineage.clone(), |_, _| [1.0; 4]);
        let make = || entry(grid, top(grid), top(grid), work.clone());
        let mut wrong = make(); wrong.key.run += 1;
        assert!(AcceptedPixels::new(1, desc(&lineage), grid, lineage.clone(), vec![], wrong).is_err());
        let mut wrong = make(); wrong.pick.texels[2] += 1.0;
        assert!(AcceptedPixels::new(1, desc(&lineage), grid, lineage.clone(), vec![], wrong).is_err());
        let mut partial = make(); partial.view.rect.w -= 1;
        assert!(AcceptedPixels::new(1, desc(&lineage), grid, lineage.clone(), vec![], partial).is_err());
        let wrong_canvas = Grid::on_lattice(5, 8, (0, -8)).unwrap();
        assert!(AcceptedPixels::new(1, desc(&lineage), wrong_canvas, lineage.clone(), vec![], make()).is_err(),
            "accepted canvas must be the frozen primary display extent");
        let foreign_lineage = self::lineage(5, 7);
        assert!(AcceptedPixels::new(1, desc(&lineage), grid, foreign_lineage, vec![], make()).is_err());
        drop(work);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn mixed_fine_and_overview_rows_use_signed_ragged_area_and_captured_runs() {
        let budget = RegionBudget::new(1 << 20);
        let grid = Grid::on_lattice(513, 7, (0, -7)).unwrap();
        let lineage = lineage(513, 7);
        let fine = work(grid, 0, &budget, lineage.clone(), |x, y| [x as f32 + 10.0 * y as f32; 4]);
        let overview = work(grid, 1, &budget, lineage.clone(), |_, _| [1000.0; 4]);
        let id = TileId { level: 0, tx: 0, ty: 0 };
        let pick = entry(grid, id, id, fine.clone()).pick;
        let pixels = AcceptedPixels::new(9, desc(&lineage), grid, lineage.clone(),
            vec![entry(grid, id, id, fine.clone())], entry(grid, top(grid), top(grid), overview.clone())).unwrap();
        assert_eq!(pixels.canvas_sample(511, 6).unwrap().run, fine.run);
        assert_eq!(pixels.canvas_sample(512, 6).unwrap().run, overview.run);
        let selected = pixels.reselect(10, &[pick]).unwrap();
        assert_eq!(selected.canvas_sample(511, 6).unwrap().run, fine.run);
        let fallback = selected.reselect(11, &[]).unwrap();
        assert_eq!(fallback.canvas_sample(0, 0).unwrap().run, overview.run);
        let absent = Pick { from: TileId { level: 0, tx: 1, ty: 0 }, ..pick };
        assert!(pixels.reselect(12, &[absent]).is_err());
        let mut rows = PixelRows::new(pixels, 1).unwrap();
        assert_eq!(rows.size(), (257, 4));
        for y in 0..4 {
            // Independent signed-Y-up bin calculation: first row has one source row.
            let (begin, end) = if y == 0 { (0, 1) } else { (2 * y - 1, 2 * y + 1) };
            let row = rows.row(y).unwrap();
            for x in 0..256 {
                let expected = (2 * x) as f32 + 0.5 + 10.0 * (begin + end - 1) as f32 / 2.0;
                assert_eq!(&row[(x * 4) as usize..(x * 4 + 4) as usize], &[expected; 4]);
            }
            assert_eq!(&row[1024..1028], &[1000.0; 4], "ragged right bin uses accepted overview");
        }
        drop(rows); drop(fallback); drop(selected); drop(fine); drop(overview);
        assert_eq!(budget.used(), 0);
    }


    #[test]
    fn equal_lod_last_paint_wins_but_never_overwrites_finer_coverage() {
        let budget = RegionBudget::new(1 << 20);
        let grid = Grid::on_lattice(513, 3, (0, -3)).unwrap();
        let lineage = lineage(513, 3);
        let first = work(grid, 0, &budget, lineage.clone(), |_, _| [2.0; 4]);
        let last = work(grid, 0, &budget, lineage.clone(), |_, _| [3.0; 4]);
        let overview = work(grid, 1, &budget, lineage.clone(), |_, _| [9.0; 4]);
        let fine = TileId { level: 0, tx: 0, ty: 0 };
        let pixels = AcceptedPixels::new(1, desc(&lineage), grid, lineage,
            vec![entry(grid, fine, fine, first), entry(grid, fine, fine, last),
                entry(grid, fine, top(grid), overview.clone())],
            entry(grid, top(grid), top(grid), overview)).unwrap();
        assert_eq!(pixels.canvas_sample(0, 0).unwrap().pixel, [3.0; 4]);
        assert_eq!(pixels.canvas_sample(512, 2).unwrap().pixel, [9.0; 4]);
        drop(pixels);
        assert_eq!(budget.used(), 0);
    }


    #[test]
    fn reselect_cannot_expand_a_captured_owner_local_subview() {
        let budget = RegionBudget::new(1 << 20);
        let grid = Grid::on_lattice(1025, 3, (0, -3)).unwrap();
        let lineage = lineage(1025, 3);
        let partial_owner = work(grid, 1, &budget, lineage.clone(), |_, _| [2.0; 4]);
        let overview = work(grid, 2, &budget, lineage.clone(), |_, _| [9.0; 4]);
        let from = TileId { level: 1, tx: 0, ty: 0 };
        let id = TileId { level: 0, tx: 0, ty: 0 };
        let mut captured = entry(grid, id, from, partial_owner);
        captured.view.rect.w = 256;
        let pixels = AcceptedPixels::new(1, desc(&lineage), grid, lineage,
            vec![captured], entry(grid, top(grid), top(grid), overview)).unwrap();
        let expanded = Pick { id: from, from,
            texels: grid.texture_rect(from, grid.source_rect(from).unwrap()).unwrap() };
        assert!(pixels.reselect(2, &[expanded]).is_err(),
            "full owner storage is not permission to expand an accepted subview");
        assert_eq!(pixels.canvas_sample(511, 0).unwrap().pixel, [2.0; 4]);
        assert_eq!(pixels.canvas_sample(512, 0).unwrap().pixel, [9.0; 4]);
        drop(pixels);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn same_native_proxy_bits_survive_readout_and_area_rows() {
        let budget = RegionBudget::new(8192);
        let mut grid = Grid::on_lattice(5, 7, (0, -7)).unwrap();
        grid.include_level(1).unwrap();
        let lineage = lineage(5, 7);
        let bits = [0x8000_0000, 0x7fc1_2345, 0x3f80_0347, 0x3f80_0000];
        let work = work(grid, 1, &budget, lineage.clone(), |_, _| bits.map(f32::from_bits));
        let pixels = AcceptedPixels::new(4, desc(&lineage), grid, lineage,
            vec![], entry(grid, top(grid), top(grid), work)).unwrap();
        assert_eq!(pixels.canvas_sample(4, 6).unwrap().pixel.map(f32::to_bits), bits);
        let mut rows = PixelRows::new(pixels, 1).unwrap();
        for y in 0..4 { for pixel in rows.row(y).unwrap().as_chunks::<4>().0 {
            assert_eq!(pixel.iter().copied().map(f32::to_bits).collect::<Vec<_>>(), bits);
        }}
        drop(rows);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn scopes_fit_without_encoder_scratch_and_export_budget_drops_temporary_pins() {
        let budget = RegionBudget::new(24000);
        let grid = Grid::on_lattice(256, 1, (0, -1)).unwrap();
        let lineage = lineage(256, 1);
        let attempt = budget.begin_attempt().unwrap();
        let work = work(grid, 0, &attempt, lineage.clone(), |x, _| [x as f32, 2.0, 3.0, 1.0]);
        drop(attempt);
        let pixels = AcceptedPixels::new(1, desc(&lineage), grid, lineage,
            vec![], entry(grid, top(grid), top(grid), work.clone())).unwrap();
        drop(work);
        let charge = budget.used();
        assert_eq!(pixels.retirement_credit(), charge);
        let mut rows = PixelRows::new(pixels.clone(), 0).unwrap();
        assert_eq!(rows.alpha.capacity(), 0);
        assert_eq!(rows.encoded.capacity(), 0);
        assert!(rows.bytes() <= budget.candidate_limit());
        assert_eq!(&rows.row(0).unwrap()[1020..], &[255.0, 2.0, 3.0, 1.0]);
        assert_eq!(pixels.retirement_credit(), 0);
        drop(rows);
        assert_eq!(budget.used(), charge);
        assert_eq!(pixels.retirement_credit(), charge);
        assert!(matches!(PixelRows::for_export(pixels.clone(), 0),
            Err(RegionError::Budget { needed, limit }) if needed > limit && limit == 12000));
        assert_eq!(budget.used(), charge, "intrinsic export refusal must not reserve scratch");
        assert_eq!(pixels.retirement_credit(), charge, "refused export must release its snapshot alias");
        drop(pixels);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn outer_snapshot_aliases_and_pending_scratch_preserve_real_retirement_accounting() {
        let budget = RegionBudget::new(8192);
        let mut grid = Grid::on_lattice(5, 7, (0, -7)).unwrap();
        grid.include_level(1).unwrap();
        let lineage = lineage(5, 7);
        let attempt = budget.begin_attempt().unwrap();
        let work = work(grid, 1, &attempt, lineage.clone(), |_, _| [1.0; 4]);
        drop(attempt);
        let pixels = AcceptedPixels::new(1, desc(&lineage), grid, lineage,
            vec![], entry(grid, top(grid), top(grid), work.clone())).unwrap();
        drop(work);
        let charge = budget.used();
        assert_eq!(pixels.retirement_credit(), charge);
        let alias = pixels.clone();
        assert_eq!(pixels.retirement_credit(), 0, "an external Inner alias pins every physical run");
        drop(alias);
        assert_eq!(pixels.retirement_credit(), charge);
        let pressure = budget.reserve(8192 - charge).unwrap();
        assert!(matches!(PixelRows::new(pixels.clone(), 1), Err(RegionError::Pending)));
        assert_eq!(pixels.retirement_credit(), charge, "failed scratch must drop its temporary snapshot");
        drop(pressure);
        let rows = PixelRows::new(pixels.clone(), 1).unwrap();
        assert_eq!(pixels.retirement_credit(), 0);
        drop(rows);
        assert_eq!(pixels.retirement_credit(), charge);
        drop(pixels);
        assert_eq!(budget.used(), 0);
    }
}
