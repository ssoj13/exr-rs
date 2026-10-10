//! Charged original-precision effect work retained by TileState's existing RAM map.
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use ofx_runtime::ofx_host::{CpuFrame, OfxRectI, PixelComponents, PixelDepth, PixelLayout, PixelStorage, Premultiplication, StorageKind};
use super::grid::PixRect;
use super::region::{Permit, RegionBudget, RegionError};

static OUTPUT_RUN: AtomicU64 = AtomicU64::new(1);

/// Immutable original-source identity and geometry for one effect generation.
/// This metadata never retains decoded pixels or a demand holding lease.
#[derive(Debug)]
pub struct ProcessedLineage {
    pub(crate) primary_index: i32,
    pub(crate) frame_id: u64,
    pub(crate) data_window: imath_rs::Box2i,
    pub(crate) display_window: imath_rs::Box2i,
    pub(crate) pixel_aspect: f64,
    pub(crate) selection: crate::image::Selection,
    pub(crate) time: f64,
}

/// One original-precision sample and its clipped canvas footprint.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CanvasSample {
    pub(crate) pixel: [f32; 4],
    pub(crate) cell: PixRect,
    pub(crate) run: u64,
    pub(crate) level: u32,
}

/// Original f32 storage. Field order destroys pixels before releasing admission.
pub(crate) struct ChargedPixelStorage {
    pixels: Vec<f32>,
    _permit: Permit,
}
impl ChargedPixelStorage {
    pub(crate) fn required_bytes(bounds: OfxRectI) -> Result<usize, RegionError> {
        let w = usize::try_from(i64::from(bounds.x2) - i64::from(bounds.x1)).map_err(|_|"output width overflows")?;
        let h = usize::try_from(i64::from(bounds.y2) - i64::from(bounds.y1)).map_err(|_|"output height overflows")?;
        w.checked_mul(h).and_then(|n| n.checked_mul(16))
            .filter(|&n| n > 0 && n <= isize::MAX as usize)
            .ok_or_else(|| RegionError::Failed("output extent exceeds addressable storage".into()))
    }
    pub(crate) fn allocate(bounds: OfxRectI, budget: &RegionBudget) -> Result<Self, RegionError> {
        let bytes = Self::required_bytes(bounds)?;
        let mut permit = budget.reserve(bytes)?;
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(bytes / 4).map_err(|e| RegionError::Failed(e.to_string()))?;
        permit.resize(pixels.capacity() * 4)?;
        pixels.resize(bytes / 4, 0.0);
        Ok(Self { pixels, _permit: permit })
    }
}
impl PixelStorage for ChargedPixelStorage {
    fn bytes(&self)->&[u8] { bytemuck::cast_slice(&self.pixels) }
    fn bytes_mut(&mut self)->Option<&mut[u8]> { Some(bytemuck::cast_slice_mut(&mut self.pixels)) }
}

/// One original-precision effect result retained by the existing tile RAM owner map.
///
/// The generic CpuFrame owns the delivered allocation, including negotiated host storage
/// and its admission lease. A physical output has its own run identity; a non-tile
/// whole-level result reuses that run for coarser display, while a tiled presentation may
/// combine several work-unit runs. One run alone does not identify such a presentation.
///
/// Rows are top-first views of checked signed/padded Float RGBA geometry. Aligned storage
/// can be borrowed as f32; unaligned native-endian storage uses pixel reads with original
/// sample bits. Rectangular views retain this exact owner without copying pixels.
/// Accepted consumer snapshots deduplicate these owners and retain their immutable lineage;
/// their aliases keep physical storage charged until the last consumer releases it.
pub struct ProcessedWork {
    pub(crate) run: u64,
    pub(crate) level: u32,
    pub(crate) time: f64,
    pub(crate) bounds: OfxRectI,
    output: CpuFrame,
    budget: RegionBudget,
    lineage: Arc<ProcessedLineage>,
}
impl ProcessedWork {
    #[cfg(test)]
    pub(crate) fn from_output(output: CpuFrame, level: u32, time: f64) -> Result<Arc<Self>, String> {
        let storage = output.clone().into_storage_as::<ChargedPixelStorage>()
            .map_err(|_| "render returned an uncharged output allocation".to_owned())?;
        let budget = storage._permit.budget().clone();
        drop(storage);
        let lineage = Self::fixture_lineage(output.layout().bounds(), time);
        Self::from_rendered(output, level, time, budget, lineage)
    }
    /// One exact layout boundary for pre-dispatch and actual upload admission.
    pub(crate) fn upload_bytes(grid: &super::grid::Grid, id: super::grid::TileId, fine: u32)
        -> Result<usize, RegionError> {
        if id.level < fine { return Err(RegionError::Unsupported); }
        let rect = grid.rect(id).ok_or("upload tile is outside the processed grid")?;
        let half = super::region::pixel_count(rect.w, rect.h)?.checked_mul(8)
            .ok_or("half upload bytes overflow")?;
        let scratch = if id.level > fine {
            (rect.w as usize).checked_mul(48).ok_or("upload row scratch overflows")?
        } else { 0 };
        half.checked_add(scratch).ok_or_else(|| RegionError::Failed("upload peak bytes overflow".into()))
    }
    /// Called only after the entire host request used this attempt's admission.
    pub(crate) fn from_rendered(output: CpuFrame, level: u32, time: f64,
        budget: RegionBudget, lineage: Arc<ProcessedLineage>) -> Result<Arc<Self>, String> {
        if !time.is_finite() || lineage.time.to_bits() != time.to_bits() || level > 40 {
            return Err("processed time or sampling level disagrees with its immutable lineage".into());
        }
        let bounds = output.layout().bounds();
        if output.layout().depth() != PixelDepth::Float
            || output.layout().components() != PixelComponents::Rgba {
            return Err("processed output was not negotiated to Float RGBA".into());
        }
        let run = OUTPUT_RUN.try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| "processed output identity exhausted")?;
        Ok(Arc::new(Self { run, level, time, bounds, output, budget, lineage }))
    }
    pub(crate) fn lineage(&self) -> &Arc<ProcessedLineage> { &self.lineage }
    #[cfg(test)]
    fn fixture_lineage(bounds: OfxRectI, time: f64) -> Arc<ProcessedLineage> {
        let window = imath_rs::Box2i::from_min_max(
            imath_rs::V2i::new(bounds.x1, bounds.y1),
            imath_rs::V2i::new(bounds.x2 - 1, bounds.y2 - 1));
        Arc::new(ProcessedLineage {
            primary_index: 0, frame_id: 0, data_window: window, display_window: window,
            pixel_aspect: 1.0, selection: crate::image::Selection::default(), time,
        })
    }
    /// Location of this checked host output on the reflected, top-first effect lattice.
    fn native_rect(&self, grid: &super::grid::Grid) -> Option<PixRect> {
        if self.level >= grid.levels() || self.level > 40 { return None; }
        let (x0, y0, _, _) = grid.lattice_bounds(self.level);
        let (w, h) = self.dimensions();
        let x = u32::try_from(i64::from(self.bounds.x1).checked_sub(x0)?).ok()?;
        let y = u32::try_from(y0.checked_neg()?.checked_sub(i64::from(self.bounds.y2))?).ok()?;
        let (gw, gh) = grid.level_size(self.level);
        (w > 0 && h > 0 && x.checked_add(w)? <= gw && y.checked_add(h)? <= gh)
            .then_some(PixRect { x, y, w, h })
    }
    /// Read the exact accepted owner sample covering an original top-first canvas pixel.
    pub(crate) fn canvas_sample(&self, grid: &super::grid::Grid, x: u32, y: u32) -> Option<CanvasSample> {
        let (lx, ly) = grid.level_index(self.level, x, y)?;
        let owner = self.native_rect(grid)?;
        let px = lx.checked_sub(owner.x)?;
        let py = ly.checked_sub(owner.y)?;
        Some(CanvasSample { pixel: self.pixel(px, py)?,
            cell: grid.level_cell(self.level, lx, ly)?, run: self.run, level: self.level })
    }
    pub(crate) fn bytes(&self) -> usize { self.budget.attempt_live().unwrap_or(self.output.bytes().len()) }
    pub(crate) fn budget(&self) -> &RegionBudget { &self.budget }
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        ((i64::from(self.bounds.x2) - i64::from(self.bounds.x1)) as u32,
         (i64::from(self.bounds.y2) - i64::from(self.bounds.y1)) as u32)
    }
    fn row_bytes(&self, y: u32) -> Option<&[u8]> {
        let (w, h) = self.dimensions();
        if y >= h { return None; }
        let layout = self.output.layout();
        let logical_y = i32::try_from(i64::from(self.bounds.y2) - 1 - i64::from(y)).ok()?;
        let range = layout.active_row_range(logical_y, self.output.bytes().len()).ok()?;
        let _ = w;
        self.output.bytes().get(range)
    }
    /// Borrow one top-first aligned row after checking the actual host row layout.
    /// Padding and signed stride do not change logical row order. Unaligned storage
    /// returns None here and remains readable through pixel(); no f16 round-trip is used.
    pub(crate) fn row(&self, y: u32) -> Option<&[f32]> {
        bytemuck::try_cast_slice(self.row_bytes(y)?).ok()
    }
    /// Read one top-first native-endian Float RGBA pixel, including valid unaligned storage.
    /// Out-of-bounds coordinates return None; sample bits are not reduced or converted.
    pub(crate) fn pixel(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        if x >= self.dimensions().0 { return None; }
        let bytes = self.row_bytes(y)?.get(x as usize * 16..x as usize * 16 + 16)?;
        Some(std::array::from_fn(|c| f32::from_ne_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap())))
    }
    #[cfg(test)]
    pub(crate) fn copy_row(&self, y: u32, out: &mut Vec<f32>) -> Result<(), RegionError> {
        let bytes = self.row_bytes(y).ok_or("processed row is outside its checked layout")?;
        out.clear();
        out.extend(bytes.as_chunks::<4>().0.iter().map(|v| f32::from_ne_bytes(*v)));
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn pixels(&self) -> &[f32] {
        assert!(self.output.layout().row_bytes().is_some_and(|stride| stride < 0));
        bytemuck::try_cast_slice(self.output.bytes()).unwrap()
    }
    /// Credit only a uniquely held physical output and its sole allocation
    /// membership. Hidden host owners and external views receive no credit.
    pub(crate) fn retirement_credit(self: &Arc<Self>) -> usize {
        if Arc::strong_count(self) == 1 && Arc::strong_count(self.output.storage()) == 1
            && self.budget.attempt_members() == Some(2) {
            self.budget.attempt_live().unwrap_or(0)
        } else { 0 }
    }
    /// Retain this physical owner and run with a checked local top-first rectangle.
    /// Cloning the metadata view keeps the existing charge; it allocates no pixel copy.
    pub(crate) fn view(self: &Arc<Self>, rect: PixRect) -> Option<ProcessedView> {
        let (w, h) = self.dimensions();
        if rect.w == 0 || rect.h == 0 || u64::from(rect.x) + u64::from(rect.w) > u64::from(w)
            || u64::from(rect.y) + u64::from(rect.h) > u64::from(h) { return None; }
        Some(ProcessedView { work: self.clone(), rect })
    }
}
/// Metadata-only rectangular borrow retains the exact allocation and run.
#[derive(Clone)]
pub(crate) struct ProcessedView {
    pub(crate) work: Arc<ProcessedWork>,
    pub(crate) rect: PixRect,
}
impl ProcessedView {
    /// Original canvas span of this owner-local subview, clipped by the shared lattice.
    pub(crate) fn canvas_rect(&self, grid: &super::grid::Grid) -> Option<PixRect> {
        let owner = self.work.native_rect(grid)?;
        if self.rect.w == 0 || self.rect.h == 0
            || self.rect.x.checked_add(self.rect.w)? > owner.w
            || self.rect.y.checked_add(self.rect.h)? > owner.h { return None; }
        let x = owner.x.checked_add(self.rect.x)?;
        let y = owner.y.checked_add(self.rect.y)?;
        let first = grid.level_cell(self.work.level, x, y)?;
        let last = grid.level_cell(self.work.level,
            x.checked_add(self.rect.w.checked_sub(1)?)?,
            y.checked_add(self.rect.h.checked_sub(1)?)?)?;
        Some(PixRect { x: first.x, y: first.y,
            w: last.x.checked_add(last.w)?.checked_sub(first.x)?,
            h: last.y.checked_add(last.h)?.checked_sub(first.y)? })
    }
    pub(crate) fn pixel(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        if x >= self.rect.w || y >= self.rect.h { return None; }
        self.work.pixel(self.rect.x + x, self.rect.y + y)
    }
    pub(crate) fn row(&self, y: u32) -> Option<&[f32]> {
        if y >= self.rect.h { return None; }
        let start = self.rect.x as usize * 4;
        self.work.row(self.rect.y + y)?.get(start..start + self.rect.w as usize * 4)
    }
}
/// A bounded display upload; destruction frees half pixels before its permit.
pub struct ProcessedUpload {
    pub(crate) tile: super::reduce::TileData,
    _permit: Permit,
}
impl ProcessedWork {
    pub(crate) fn upload(self: &Arc<Self>, grid: &super::grid::Grid, id: super::grid::TileId)
        -> Result<ProcessedUpload, RegionError> {
        if id.level < self.level { return Err(RegionError::Unsupported); }
        let rect = grid.rect(id).ok_or("processed upload is outside its canvas")?;
        let (w, h) = self.dimensions();
        let view = self.view(PixRect { x: 0, y: 0, w, h }).ok_or("processed frame has empty pixels")?;
        let half_bytes = super::region::pixel_count(rect.w, rect.h)?.checked_mul(8).ok_or("half upload bytes overflow")?;
        let needed = Self::upload_bytes(grid, id, self.level)?;
        self.budget.can_temporary(needed)?;
        let scratch_budget = self.budget.physical();
        if needed > scratch_budget.candidate_limit() {
            return Err(RegionError::Budget { needed, limit: scratch_budget.candidate_limit() });
        }
        let mut permit = scratch_budget.reserve(needed)?;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(half_bytes / 2).map_err(|e| e.to_string())?;
        if id.level == self.level {
            let (x0, _, _, _) = grid.lattice_bounds(id.level);
            let work_top = i64::from(grid.level_size(id.level).1) - i64::from(self.bounds.y2);
            let work_left = i64::from(self.bounds.x1) - x0;
            for y in rect.y..rect.y + rect.h {
                for x in rect.x..rect.x + rect.w {
                    let px = view.pixel(u32::try_from(i64::from(x) - work_left).map_err(|_|"tile precedes processed output")?,
                        u32::try_from(i64::from(y) - work_top).map_err(|_|"tile precedes processed output")?)
                        .ok_or("tile exceeds processed output")?;
                    rgba.extend(px.map(|v| imath_rs::Half::from(v).bits()));
                }
            }
        } else {
            // Whole-level work can supply all coarser views without rendering again.
            if self.dimensions() != grid.level_size(self.level) {
                return Err(RegionError::Unsupported);
            }
            let mut out = Vec::new();
            out.try_reserve_exact(rect.w as usize * 4).map_err(|e| e.to_string())?;
            for y in rect.y..rect.y + rect.h {
                let plan = grid.region_plan(id.level,
                    PixRect { x: rect.x, y, w: rect.w, h: 1 },
                    super::region::RegionPolicy::OriginalArea)?;
                let mut accum = super::region::RegionAccumulator::new(plan)?;
                let rows = accum.rows();
                let first = grid.level_index(self.level, 0, rows.start)
                    .ok_or("processed reduction starts outside its source grid")?.1;
                let last = grid.level_index(self.level, 0, rows.end.checked_sub(1)
                    .ok_or("processed reduction has no source rows")?)
                    .ok_or("processed reduction ends outside its source grid")?.1;
                for fy in first..=last {
                    if let Some(row) = view.row(fy) {
                        accum.push_level_row(self.level, fy, row)?;
                    } else {
                        accum.push_level_pixels(self.level, fy, |x| view.pixel(x, fy)
                            .ok_or_else(|| "processed pixel exceeds its checked row".to_owned()))?;
                    }
                }
                accum.write_row(&mut out)?;
                rgba.extend(out.iter().map(|&v| imath_rs::Half::from(v).bits()));
                // Retain admission for all scratch until its destructors run.
                permit.resize(rgba.capacity() * 2 + out.capacity() * 4 + accum.bytes())?;
            }
        }
        permit.resize(rgba.capacity() * 2)?;
        Ok(ProcessedUpload { tile: super::reduce::TileData { w: rect.w, h: rect.h, rgba }, _permit: permit })
    }
}
pub(crate) fn output_frame(storage:ChargedPixelStorage,bounds:OfxRectI,full_rod:OfxRectI,par:f64)->Result<CpuFrame,String>{
    let row=(i64::from(bounds.x2)-i64::from(bounds.x1)).checked_mul(16)
        .and_then(|n|i32::try_from(n).ok()).ok_or("output row bytes overflow")?;
    let layout=PixelLayout::new(PixelDepth::Float,PixelComponents::Rgba,bounds,Some(-row),StorageKind::Host)
        .map_err(|_|"output layout is invalid")?;
    CpuFrame::from_storage(layout,full_rod,Arc::new(storage),par,Premultiplication::Premultiplied)
        .map_err(|_|"output frame metadata is invalid".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canvas_cells_preserve_proxy_edges_subview_bounds_and_original_bits() {
        let budget = RegionBudget::new(4096);
        let bounds = OfxRectI { x1: 0, y1: 0, x2: 3, y2: 4 };
        let mut storage = ChargedPixelStorage::allocate(bounds, &budget).unwrap();
        for (i, px) in storage.pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() { px.fill((i + 1) as f32); }
        let special = [0x8000_0000u32, 0x7fc1_2345, 0x3f80_0347, 0x3f80_0000];
        storage.pixels[..4].copy_from_slice(&special.map(f32::from_bits));
        let window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(10, -3), imath_rs::V2i::new(14, 3));
        let lineage = Arc::new(ProcessedLineage { primary_index: -7, frame_id: 42,
            data_window: window, display_window: window, pixel_aspect: 1.5,
            selection: crate::image::Selection::default(), time: -6.75 });
        let frame = output_frame(storage, bounds, bounds, 1.5).unwrap();
        let work = ProcessedWork::from_rendered(frame, 1, -6.75, budget.clone(), lineage.clone()).unwrap();
        let mut grid = super::super::grid::Grid::on_lattice(5, 7, (0, -7)).unwrap();
        grid.include_level(1).unwrap();
        let first = work.canvas_sample(&grid, 0, 0).unwrap();
        assert_eq!(first.cell, PixRect { x: 0, y: 0, w: 2, h: 1 });
        assert_eq!(first.pixel.map(f32::to_bits), special);
        let last = work.canvas_sample(&grid, 4, 6).unwrap();
        assert_eq!(last.cell, PixRect { x: 4, y: 5, w: 1, h: 2 });
        assert_eq!(last.pixel, [12.0; 4]);
        assert_eq!((first.run, first.level), (last.run, 1));
        let middle = work.view(PixRect { x: 1, y: 1, w: 1, h: 2 }).unwrap();
        assert_eq!(middle.canvas_rect(&grid), Some(PixRect { x: 2, y: 1, w: 2, h: 4 }));
        let forged = ProcessedView { work: work.clone(), rect: PixRect { x: 2, y: 0, w: 2, h: 1 } };
        assert!(forged.canvas_rect(&grid).is_none(), "a grid-valid rectangle cannot exceed its physical subview");
        assert!(work.canvas_sample(&grid, 5, 0).is_none());
        assert!(work.canvas_sample(&grid, 0, 7).is_none());
        assert!(Arc::ptr_eq(work.lineage(), &lineage));
        assert_eq!(lineage.primary_index, -7);
        assert_eq!(lineage.time.to_bits(), (-6.75f64).to_bits());
        drop(forged); drop(middle); drop(work);
        assert_eq!(budget.used(), 0, "immutable lineage does not retain decoded or processed pixels");
        assert_eq!(lineage.frame_id, 42);
    }

    #[test]
    fn weak_request_factory_does_not_pin_retirement_but_real_aliases_do() {
        let budget = RegionBudget::new(256);
        let attempt = budget.begin_attempt().unwrap();
        let factory = Arc::new(attempt.admission_view().unwrap());
        let metadata_clone = factory.clone();
        let storage = ChargedPixelStorage::allocate(bounds(), &attempt).unwrap();
        let frame = output_frame(storage, bounds(), bounds(), 1.0).unwrap();
        let work = ProcessedWork::from_rendered(frame, 0, 2.5, attempt.clone(),
            ProcessedWork::fixture_lineage(bounds(), 2.5)).unwrap();
        drop(attempt);
        assert_eq!(work.budget.attempt_members(), Some(2), "only Work and its physical permit retain the attempt");
        assert_eq!(work.retirement_credit(), 32);
        let ram_alias = work.clone();
        assert_eq!(work.retirement_credit(), 0, "another RAM or consumer membership forbids credit");
        drop(ram_alias);
        let cpu_alias = work.output.clone();
        assert_eq!(work.retirement_credit(), 0, "a host storage alias forbids physical retirement");
        drop(cpu_alias);
        assert_eq!(work.retirement_credit(), 32);
        drop(work);
        assert_eq!(budget.used(), 0);
        assert!(matches!(metadata_clone.reserve(16), Err(RegionError::Cancelled)),
            "an expired factory cannot charge a different request's workspace");
        let retry = budget.begin_attempt().unwrap();
        let lease = retry.admission_view().unwrap().reserve(16).unwrap();
        drop(retry);
        assert!(matches!(budget.begin_attempt(), Err(RegionError::Pending)),
            "the surviving physical host lease still owns W");
        drop(lease);
        assert!(budget.begin_attempt().is_ok());
    }

    fn bounds() -> OfxRectI { OfxRectI{x1:0,y1:0,x2:2,y2:1} }
    fn make(budget:&RegionBudget)->Arc<ProcessedWork> {
        let mut storage=ChargedPixelStorage::allocate(bounds(),budget).unwrap();
        storage.pixels.copy_from_slice(&[
            f32::from_bits(0x8000_0000),f32::from_bits(0x7fc1_2345),
            f32::INFINITY,f32::NEG_INFINITY,1.0001,2.0,3.0,4.0]);
        let mut frame=output_frame(storage,bounds(),bounds(),1.0).unwrap();
        assert!(frame.writable_in_place(),"the real Output has its unique writable storage");
        ProcessedWork::from_output(frame,0,2.5).unwrap()
    }
    #[test]
    fn rectangular_views_retain_one_charge_and_original_bits() {
        let budget=RegionBudget::new(32);
        let work=make(&budget);
        let first=work.view(PixRect{x:0,y:0,w:1,h:1}).unwrap();
        let last=work.view(PixRect{x:1,y:0,w:1,h:1}).unwrap();
        assert_eq!(work.bytes(),32);
        assert_eq!(budget.used(),32,"metadata views do not charge the allocation twice");
        assert_eq!(first.pixel(0,0).unwrap().map(f32::to_bits),
            [0x8000_0000,0x7fc1_2345,0x7f80_0000,0xff80_0000]);
        assert_eq!(last.row(0).unwrap(),[1.0001,2.0,3.0,4.0]);
        assert!(last.pixel(1,0).is_none());
        drop(work);
        drop(first);
        assert_eq!(budget.used(),32,"last accepted view retains the allocation");
        drop(last);
        assert_eq!(budget.used(),0);
    }
    #[test]
    fn rerender_in_same_time_and_level_has_a_new_output_run() {
        let budget=RegionBudget::new(64);
        let a=make(&budget);
        let b=make(&budget);
        assert_ne!(a.run,b.run);
        assert_eq!(a.time,b.time);
        assert_eq!(a.level,b.level);
        drop(a);drop(b);
        assert_eq!(budget.used(),0);
    }
    struct OffsetBytes { bytes: Vec<u8>, start: usize, len: usize, _permit: Permit }
    impl PixelStorage for OffsetBytes {
        fn bytes(&self) -> &[u8] { &self.bytes[self.start..self.start + self.len] }
    }
    #[test]
    fn negotiated_padded_signed_rows_and_unaligned_samples_keep_the_physical_owner() {
        let bits = [[0x8000_0000u32, 0x7fc1_2345, 0x3f80_0347, 0x3f80_0000],
                    [0x4000_0000u32, 0xff80_0000, 0x7f80_0000, 0x3f00_0000]];
        for stride in [24, -24] {
            let budget = RegionBudget::new(256);
            let attempt = budget.begin_attempt().unwrap();
            let bounds = OfxRectI { x1: 7, y1: -3, x2: 8, y2: -1 };
            let layout = PixelLayout::new(PixelDepth::Float, PixelComponents::Rgba,
                bounds, Some(stride), StorageKind::Host).unwrap();
            let mut bytes = vec![0xcc; 49];
            let start = if (bytes.as_ptr() as usize).is_multiple_of(4) { 1 } else { 0 };
            for (top, samples) in bits.iter().enumerate() {
                let range = layout.active_row_range(bounds.y2 - 1 - top as i32, 48).unwrap();
                for (c, bits) in samples.iter().enumerate() {
                    bytes[start + range.start + c * 4..start + range.start + c * 4 + 4]
                        .copy_from_slice(&bits.to_ne_bytes());
                }
            }
            let charge = bytes.capacity();
            let storage = Arc::new(OffsetBytes { bytes, start, len: 48,
                _permit: attempt.reserve(charge).unwrap() });
            let frame = CpuFrame::from_storage(layout, bounds, storage.clone(),
                1.25, Premultiplication::Premultiplied).unwrap();
            let work = ProcessedWork::from_rendered(frame, 0, 2.25, attempt.clone(), ProcessedWork::fixture_lineage(bounds, 2.25)).unwrap();
            let view = work.view(PixRect { x: 0, y: 0, w: 1, h: 2 }).unwrap();
            assert!(view.row(0).is_none(), "unaligned row cannot be cast to f32");
            assert_eq!(view.pixel(0, 0).unwrap().map(f32::to_bits), bits[0]);
            assert_eq!(view.pixel(0, 1).unwrap().map(f32::to_bits), bits[1]);
            let mut row = Vec::new();
            work.copy_row(1, &mut row).unwrap();
            assert_eq!(row.into_iter().map(f32::to_bits).collect::<Vec<_>>(), bits[1]);
            assert_eq!(budget.used(), charge);
            drop(work); drop(storage); drop(attempt);
            assert_eq!(budget.used(), charge, "rectangular view is still a real physical owner");
            drop(view);
            assert_eq!(budget.used(), 0);
            assert!(budget.begin_attempt().is_ok(), "last-owner release admits a retry");
        }
    }

    #[test]
    fn a_candidate_upload_must_fit_its_compound_workspace_before_retirement() {
        let budget = RegionBudget::new(128);
        let old = budget.begin_attempt().unwrap();
        let old_pixels = old.reserve(64).unwrap();
        old.promote().unwrap();
        let candidate = budget.begin_attempt().unwrap();
        let pixels = candidate.reserve(48).unwrap();
        assert!(matches!(candidate.can_temporary(24), Err(RegionError::Budget { needed: 72, limit: 64 })));
        drop(pixels); drop(candidate);
        let retry = budget.begin_attempt().unwrap();
        let pixels = retry.reserve(32).unwrap();
        retry.can_temporary(16).unwrap();
        let temporary = retry.physical().reserve(16).unwrap();
        assert_eq!(budget.used(), 112);
        assert!(matches!(retry.promote(), Err(RegionError::Pending)));
        drop(old_pixels); drop(old);
        retry.promote().unwrap();
        assert_eq!(retry.attempt_live(), Some(32), "display temporary is never promoted into retained output");
        assert_eq!(budget.used(), 48);
        drop(temporary); drop(pixels); drop(retry);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn coarser_views_share_one_proxy_and_weight_fractional_original_edges() {
        let budget = RegionBudget::new(4096);
        let bounds = OfxRectI { x1: 0, y1: 0, x2: 3, y2: 4 };
        let mut storage = ChargedPixelStorage::allocate(bounds, &budget).unwrap();
        for (i, pixel) in storage.pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel.fill((i + 1) as f32);
        }
        let output = output_frame(storage, bounds, bounds, 1.0).unwrap();
        let work = ProcessedWork::from_output(output, 1, 0.25).unwrap();
        let mut grid = super::super::grid::Grid::on_lattice(5, 7, (0, -7)).unwrap();
        grid.include_level(2).unwrap();
        assert_eq!(grid.level_size(1), (3, 4));
        let id = super::super::grid::TileId { level: 2, tx: 0, ty: 0 };
        let upload = work.upload(&grid, id).unwrap();
        let got: Vec<_> = upload.tile.rgba.as_chunks::<4>().0.iter()
            .map(|p| imath_rs::Half::from_bits(p[0]).to_f32()).collect();
        assert_eq!(got, [3.5, 5.0, 9.0, 10.5],
            "odd first Y cell and last X cell retain their original area");
        assert_eq!(budget.used(), 192 + 32, "only output and bounded half upload survive scratch destruction");
        drop(work);
        assert_eq!(budget.used(), 32);
        drop(upload);
        assert_eq!(budget.used(), 0);
    }

}
