//! Immutable worker-side OpenFX generation; output ownership stays in TileState.
use std::sync::{Arc, Mutex};
use imath_rs::Box2i;
use ofx_runtime::{Effect, EffectProject, EffectRender, InputSource, RuntimeError, Target};
use ofx_runtime::ofx_host::{ClipDeclaration, CpuFrame, OfxRectI, PixelComponents,
    PixelDepth, Premultiplication, ProjectGeometry, PixelAdmission, PixelLease, PixelRefusal,
    RenderContext};
use crate::image::{DecodedFrame, Selection};
use super::effect_frames::{Inputs, RenderInputs};
use super::effect_geometry::{CanonicalAnchor, PixelAspect, checked_extent, discrete_time};
use super::grid::{Grid, TileId};
use super::processed::{ChargedPixelStorage, ProcessedLineage, ProcessedWork, output_frame};
use super::region::{AttemptAdmission, RegionBudget, RegionError};
use super::source::{ProductionContext, Readiness};

/// The application owns the only accounting ledger. Host leases follow every
/// physical allocation, including persistent ImageMemory and returned guards.
struct HostPixels(AttemptAdmission);
impl std::fmt::Debug for HostPixels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostPixels").field("used", &self.0.used()).finish()
    }
}
struct HostLease { _permit: super::region::Permit }
impl std::fmt::Debug for HostLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("HostLease") }
}
impl PixelLease for HostLease {}
impl PixelAdmission for HostPixels {
    fn reserve(&self, bytes: usize) -> Result<Box<dyn PixelLease>, PixelRefusal> {
        match self.0.reserve(bytes) {
            Ok(permit) => Ok(Box::new(HostLease { _permit: permit })),
            Err(RegionError::Pending) => Err(PixelRefusal::Pending {
                required_bytes: bytes, capacity: self.0.attempt_capacity(),
            }),
            Err(RegionError::Budget { needed, limit }) => Err(PixelRefusal::Budget {
                required_bytes: needed, capacity: limit,
            }),
            Err(_) => Err(PixelRefusal::Unsupported),
        }
    }
}

/// Immutable lazy tile source for one effect/input/time/project snapshot.
/// Demand reads original channels through checked ROI providers; rendering owns charged
/// Float RGBA work in TileState's existing RAM map. Tileable effects plan actual windows;
/// non-tile effects choose one bounded whole-level proxy and reuse its physical result.
/// CPU owner admission and existing readiness/cancellation paths are shared with the host.
/// Consumers freeze accepted physical owners and lineage separately from this pending source.
pub(crate) struct EffectSource {
    effect: Arc<Effect<u32>>,
    inputs: Arc<Inputs>,
    primary_window: Box2i,
    view: Selection,
    anchor: CanonicalAnchor,
    project: EffectProject,
    time: f64,
    budget: RegionBudget,
    canvas: Box2i,
    lineage: Arc<ProcessedLineage>,
    #[cfg(test)]
    trace: super::effect_frames::ProviderTrace,
}
impl EffectSource {
    #[allow(clippy::too_many_arguments, reason = "Keeps the immutable input, timing, and budget snapshot explicit at one boundary.")]
    pub(crate) fn new(effect:Arc<Effect<u32>>,inputs:Arc<Inputs>,primary:Arc<DecodedFrame>,
        view:Selection,time:f64,frame_rate:Option<f64>,frame_range:Option<[f64;2]>,
        budget:RegionBudget)->Result<Self,RegionError> {
        discrete_time(time)?;
        let (primary_index, frame_id) = match inputs.as_ref() {
            Inputs::Static { frame, index } => (*index, frame.id),
            Inputs::Demand { primary_index, primary_id, .. } => (*primary_index, *primary_id),
        };
        if frame_id != primary.id { return Err(RegionError::Cancelled); }
        let aspect=PixelAspect::new(f64::from(primary.pixel_aspect))?;
        let lineage = Arc::new(ProcessedLineage {
            primary_index, frame_id, data_window: primary.data_window,
            display_window: primary.display_window, pixel_aspect: aspect.value(),
            selection: view.clone(), time,
        });
        let canvas=primary.display_window;
        let (w,h)=checked_extent(canvas)?;
        let geometry=ProjectGeometry::new([f64::from(w)*aspect.value(),f64::from(h)],
            [f64::from(w)*aspect.value(),f64::from(h)],[0.,0.],aspect.value())
            .map_err(|_|"effect project geometry is invalid")?;
        if frame_rate.is_some_and(|rate|!rate.is_finite()||rate<=0.) {
            return Err("effect frame rate must be finite and positive".into())
        }
        if frame_range.is_some_and(|range|!range.iter().all(|v|v.is_finite())||range[0]>range[1]) {
            return Err("effect frame range must be finite and ordered".into())
        }
        let anchor=CanonicalAnchor::new(canvas,aspect)?;
        Grid::on_lattice(w,h,(0,-i64::from(h)))?.with_aspect(aspect.value())?;
        Ok(Self{effect,inputs,primary_window:primary.data_window,view,anchor,project:EffectProject{geometry,frame_rate,frame_range},
            time,budget,canvas,lineage,
            #[cfg(test)] trace: Default::default()})
    }
    pub(crate) fn grid(&self)->Result<Grid,RegionError> {
        let (w,h)=checked_extent(self.canvas)?;
        Ok(Grid::on_lattice(w,h,(0,-i64::from(h)))?.with_aspect(self.anchor.aspect().value())?)
    }
    pub(crate) fn readiness(&self)->Readiness {
        Readiness{inputs:self.inputs.readiness_epoch(),budget:self.budget.readiness_epoch()}
    }
    fn declaration(&self,bounds:OfxRectI,rod:OfxRectI)->Result<ClipDeclaration,RegionError> {
        ClipDeclaration::new(PixelDepth::Float,PixelComponents::Rgba,bounds)
            .and_then(|declaration|declaration.with_metadata(rod,self.anchor.aspect().value(),
                Premultiplication::Premultiplied))
            .map_err(|_|RegionError::Failed("effect clip declaration is invalid".into()))
    }
    fn request<'a>(&self,output:CpuFrame,level:u32,window:OfxRectI,frames:&'a RenderInputs<'a>,
        context:&ProductionContext, policy: &Arc<dyn PixelAdmission>)->Result<EffectRender<'a,u32>,RegionError> {
        let input=self.anchor.scaled_bounds(self.primary_window,self.anchor.aspect(),level)?;
        let declaration=self.declaration(input,input)?;
        let scale=2.0_f64.powi(-(level as i32));
        // Every render here is a viewer frame the user is looking at (exports take the
        // accepted snapshot and never re-render), so the plug-in sees an interactive render.
        let mut request=EffectRender::new(self.time,self.project,output,RenderContext::interactive())
            .with_render_scale([scale,scale]).with_frames(frames)
            .with_abort(context.cancelled.clone()).with_window(window)
            .with_pixel_admission(policy.clone());
        for clip in self.effect.described().clips.iter() {
            if clip.primary || !clip.optional {
                request=request.with_input(clip.name.clone(),InputSource::Primary,declaration.clone());
            }
        }
        Ok(request)
    }
    fn error(error:RuntimeError,frames:&RenderInputs<'_>,context:&ProductionContext)->RegionError {
        if context.is_cancelled() || !frames.valid() {return RegionError::Cancelled}
        if let Some(error)=frames.take_error() {return error}
        match error {
            RuntimeError::PixelAdmission(PixelRefusal::Pending { .. }) => RegionError::Pending,
            RuntimeError::PixelAdmission(PixelRefusal::Budget { required_bytes, capacity }) =>
                RegionError::Budget { needed: required_bytes, limit: capacity },
            RuntimeError::PixelAdmission(PixelRefusal::Unsupported) => RegionError::Unsupported,
            RuntimeError::InputPending=>RegionError::Pending,
            RuntimeError::Aborted=>RegionError::Cancelled,
            RuntimeError::SiteUnsupported(_)=>RegionError::Unsupported,
            error=>RegionError::Failed(error.to_string()),
        }
    }
    /// Plan uses metadata-only Output; real pixels are admitted only after its answer.
    pub(crate) fn render_window(&self,grid:&Grid,id:TileId,
        context:&ProductionContext)->Result<Arc<ProcessedWork>,RegionError> {
        let rect=grid.rect(id).ok_or("effect tile is outside its canvas")?;
        self.budget.clear_workspace();
        let attempt = self.budget.begin_attempt()?;
        let policy: Arc<dyn PixelAdmission> = Arc::new(HostPixels(attempt.admission_view()?));
        self.render_bounds(id.level, Some(rect), context, &attempt, &policy, &mut false)
    }
    pub(crate) fn render_level(&self, context:&ProductionContext)->Result<Arc<ProcessedWork>,RegionError> {
        self.budget.clear_workspace();
        for level in 0..=40 {
            if context.is_cancelled() { return Err(RegionError::Cancelled); }
            // A different scale is a fresh candidate. Retired factories must not
            // attribute late host allocations to the next candidate's membership.
            let attempt = self.budget.begin_attempt()?;
            let policy: Arc<dyn PixelAdmission> = Arc::new(HostPixels(attempt.admission_view()?));
            let mut dispatched = false;
            let result = self.render_bounds(level,None,context, &attempt, &policy, &mut dispatched);
            attempt.clear_workspace();
            // A clean attempt drops here before the next scale claims the slot.
            // Escaped storage (even zero bytes) must prevent another candidate.
            match result {
                Err(RegionError::Budget{..}) if !dispatched && level<40
                    && attempt.attempt_live() == Some(0)
                    && attempt.attempt_members() == Some(1) => continue,
                result => return result,
            }
        }
        Err("effect has no bounded proxy level".into())
    }
    fn render_bounds(&self, level:u32, rect:Option<super::grid::PixRect>,
        context:&ProductionContext, budget: &RegionBudget, policy: &Arc<dyn PixelAdmission>,
        dispatched: &mut bool)->Result<Arc<ProcessedWork>,RegionError> {
        if context.is_cancelled() {return Err(RegionError::Cancelled)}
        budget.clear_workspace();
        match self.effect.target() {
            Target::Cpu=>{},
            Target::Site(_)=>return Err(RegionError::Unsupported),
            Target::Unavailable(error)=>return Err(error.into()),
        }
        let full=self.anchor.scaled_bounds(self.canvas,self.anchor.aspect(),level)?;
        let rect=rect.unwrap_or(super::grid::PixRect{x:0,y:0,w:u32::try_from(full.x2-full.x1).map_err(|_|"effect canvas width overflows")?,
            h:u32::try_from(full.y2-full.y1).map_err(|_|"effect canvas height overflows")?});
        let y2=i64::from(full.y2)-i64::from(rect.y);
        let bounds=OfxRectI{x1:i32::try_from(rect.x).map_err(|_|"effect tile X exceeds bounds")?,
            x2:i32::try_from(u64::from(rect.x)+u64::from(rect.w)).map_err(|_|"effect tile X exceeds bounds")?,
            y1:i32::try_from(y2-i64::from(rect.h)).map_err(|_|"effect tile Y exceeds bounds")?,
            y2:i32::try_from(y2).map_err(|_|"effect tile Y exceeds bounds")?};
        let output_bytes = ChargedPixelStorage::required_bytes(bounds)?;
        let mut upload_grid = self.grid()?;
        upload_grid.include_level(level)?;
        let first_upload = if self.effect.described().tiling.tiles {
            TileId { level, tx: rect.x / super::grid::TILE, ty: rect.y / super::grid::TILE }
        } else {
            TileId { level: upload_grid.levels() - 1, tx: 0, ty: 0 }
        };
        let upload_bytes = ProcessedWork::upload_bytes(&upload_grid, first_upload, level)?;
        if output_bytes > budget.retained_limit() {
            return Err(RegionError::Budget { needed: output_bytes, limit: budget.retained_limit() });
        }
        let frames=RenderInputs{inputs:&self.inputs,anchor:self.anchor,view:&self.view,
            context,budget,owners:Mutex::new(Vec::new()),outcome:Mutex::new(None),
            #[cfg(test)] trace: Some(&self.trace)};
        let metadata=CpuFrame::from_declaration(self.declaration(full,full)?);
        let planning=self.request(metadata,level,bounds,&frames,context, policy)?;
        let planned = self.effect.plan(&planning,bounds);
        #[cfg(test)]
        if let Ok(inputs) = &planned { self.trace.planned.lock().unwrap().push(inputs.clone()); }
        budget.clear_workspace();
        planned.map_err(|error|Self::error(error,&frames,context))?;
        if !frames.valid() {return Err(RegionError::Cancelled)}
        if let Some(error)=frames.take_error() {return Err(error)}
        let initial = budget.attempt_live().unwrap_or(0).checked_add(output_bytes).ok_or("initial effect output size overflows")?;
        if initial > budget.retained_limit() {
            return Err(RegionError::Budget { needed: initial, limit: budget.retained_limit() });
        }
        let peak = initial.checked_add(upload_bytes).ok_or("initial effect display peak overflows")?;
        if peak > budget.candidate_limit() {
            return Err(RegionError::Budget { needed: peak, limit: budget.candidate_limit() });
        }
        let output=output_frame(ChargedPixelStorage::allocate(bounds,budget)?,bounds,full,
            self.anchor.aspect().value())?;
        let request=self.request(output,level,bounds,&frames,context, policy)?;
        // The host certifies the native entry boundary. Preparation refusals can
        // try a coarser scale only after all owners retire; Begin/Render refusals
        // are terminal even if the plug-in never completed a Render callback.
        let rendered=self.effect.render_attempt(request);
        // A successful identity/pass-through result is completed work even
        // without native entry; later upload refusal must not choose another scale.
        *dispatched = rendered.lifecycle_started || rendered.result.is_ok();
        let rendered=rendered.result;
        budget.clear_workspace();
        let rendered=rendered.map_err(|error|Self::error(error,&frames,context))?;
        if !frames.valid() {return Err(RegionError::Cancelled)}
        if let Some(error)=frames.take_error() {return Err(error)}
        budget.can_temporary(upload_bytes)?;
        ProcessedWork::from_rendered(rendered,level,self.time,budget.clone(),self.lineage.clone()).map_err(RegionError::from)
    }
}

impl super::source::TileSource for EffectSource {
    fn size(&self) -> (u32, u32) { checked_extent(self.canvas).expect("constructor checked effect canvas") }
    fn grid(&self) -> Grid { self.grid().expect("constructor checked effect lattice") }
    fn processed(&self) -> bool { true }
    fn processed_lineage(&self) -> Option<Arc<ProcessedLineage>> { Some(self.lineage.clone()) }
    fn unit(&self, id: TileId) -> super::source::Unit {
        if self.effect.described().tiling.tiles { super::source::Unit::Tile(id) }
        else { super::source::Unit::Level }
    }
    fn produce(&self, grid: &Grid, unit: super::source::Unit, context: &ProductionContext)
        -> super::source::Production {
        let result = match unit {
            super::source::Unit::Tile(id) => self.render_window(grid, id, context),
            super::source::Unit::Level => self.render_level(context),
            _ => Err(RegionError::Unsupported),
        };
        if context.is_cancelled() { return super::source::Production::Cancelled; }
        match result {
            Ok(work) => super::source::Production::Processed(work),
            Err(RegionError::Pending) => super::source::Production::Pending,
            Err(RegionError::Cancelled) => super::source::Production::Cancelled,
            Err(error) => super::source::Production::Refused(error),
        }
    }
    fn readiness(&self) -> Readiness { self.readiness() }
    fn scope_level(&self, _: &Grid, _: u64, _: u32) -> Option<(u32, Arc<Vec<f32>>)> { None }
}

#[cfg(test)]
#[path = "effect_gaussian_tests.rs"]
mod gaussian_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::tests_support::frame_with;
    use exr_core::api::ChannelData;
    use imath_rs::V2i;
    use ofx_runtime::{Runtime, PluginKey, EffectSetup};
    use ofx_runtime::ofx_host::HostContext;

    #[test]
    fn actual_builtin_worker_request_preserves_top_first_original_precision() {
        let runtime=Runtime::new(crate::ofx::OfxEffects::host_info(),crate::ofx::OfxEffects::bundle_scan());
        let described=runtime.describe(&PluginKey::new("net.ofxrs.basic_gain",1),HostContext::Filter).unwrap();
        let effect=runtime.effect(0,&described,EffectSetup::default());
        let mut primary=frame_with(&["R","G","B","A"]);
        primary.width=2;primary.height=2;
        primary.data_window=Box2i::from_min_max(V2i::new(10,-3),V2i::new(11,-2));
        primary.display_window=primary.data_window;
        let pixels=[[1.0001,0.,0.,1.],[0.,0.25,0.,1.],[0.,0.,0.5,1.],[1.,1.,1.,1.]];
        for (c,(_,channel)) in primary.channels.iter_mut().enumerate() {
            *channel=ChannelData::Float(pixels.iter().map(|p|p[c]).collect());
        }
        let primary=Arc::new(primary);
        let inputs=Arc::new(Inputs::Static{frame:primary.clone(),index:0});
        let budget=RegionBudget::new(4096);
        let source=EffectSource::new(effect,inputs,primary,Selection::default(),0.5,
            Some(24.),Some([0.,0.]),budget.clone()).unwrap();
        let grid=source.grid().unwrap();
        let pool=rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let work=pool.install(||source.render_window(&grid,TileId{level:0,tx:0,ty:0},
            &ProductionContext::default())).unwrap();
        assert_eq!(work.pixels(),pixels.concat());
        assert_eq!(work.time,0.5,"plugin time keeps the requested fraction; input hold uses frame zero");
        assert_eq!(work.dimensions(),(2,2));
        assert_eq!(budget.used(),64,"one accepted f32 output owns the retained physical charge");
        drop(work);
        assert_eq!(budget.used(),0);
    }
    #[test]
    fn actual_gaussian_tiles_stitch_to_one_whole_render_with_roi_halos_and_display_canvas() {
        let runtime = Runtime::new(crate::ofx::OfxEffects::host_info(), crate::ofx::OfxEffects::bundle_scan());
        let described = runtime.describe(&PluginKey::new("net.ofxrs.gaussian_blur", 1), HostContext::Filter).unwrap();
        assert!(described.tiling.tiles);
        let effect = runtime.effect(11, &described, EffectSetup::default());
        let mut primary = frame_with(&["R", "G", "B", "A"]);
        primary.width = 517; primary.height = 7; primary.pixel_aspect = 1.5;
        primary.data_window = Box2i::from_min_max(V2i::new(-13, 5), V2i::new(503, 11));
        primary.display_window = Box2i::from_min_max(V2i::new(-16, 3), V2i::new(503, 13));
        for (c, (_, channel)) in primary.channels.iter_mut().enumerate() {
            *channel = ChannelData::Float((0..primary.width * primary.height).map(|i| {
                if c == 3 { 1.0 } else { ((i * 13 + c as u32 * 7) % 97) as f32 / 97.0 }
            }).collect());
        }
        let primary = Arc::new(primary);
        let inputs = Arc::new(Inputs::Static { frame: primary.clone(), index: 0 });
        let whole_source = EffectSource::new(effect.clone(), inputs.clone(), primary.clone(),
            Selection::default(), 0.75, Some(24.0), Some([0.0, 0.0]), RegionBudget::new(8 << 20)).unwrap();
        let tiled_budget = RegionBudget::new(8 << 20);
        let tiled_source = EffectSource::new(effect, inputs, primary, Selection::default(), 0.75,
            Some(24.0), Some([0.0, 0.0]), tiled_budget.clone()).unwrap();
        let grid = tiled_source.grid().unwrap();
        assert_eq!(grid.size(), (520, 11));
        assert_eq!(grid.display_aspect(), 1.5);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let whole = pool.install(|| whole_source.render_level(&ProductionContext::default())).unwrap();
        assert_eq!(whole.level, 0);
        let (columns, rows) = grid.tiles(0);
        assert_eq!((columns, rows), (2, 1));
        for ty in 0..rows {
            for tx in 0..columns {
                let id = TileId { level: 0, tx, ty };
                let rect = grid.rect(id).unwrap();
                let work = pool.install(|| tiled_source.render_window(&grid, id,
                    &ProductionContext::default())).unwrap();
                for y in 0..rect.h {
                    for x in 0..rect.w {
                        let got = work.pixel(x, y).unwrap();
                        let expected = whole.pixel(rect.x + x, rect.y + y).unwrap();
                        for c in 0..4 {
                            assert!((got[c] - expected[c]).abs() <= 1e-6,
                                "ROI seam ({},{}) channel {}: {} vs {}", rect.x + x, rect.y + y, c, got[c], expected[c]);
                        }
                    }
                }
                drop(work);
                assert_eq!(tiled_budget.used(), 0);
            }
        }
    }

    #[test]
    fn actual_file_input_retires_unpinned_pool_scratch_before_display_headroom_is_needed() {
        let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("module3-file-budget-{}-{unique}.exr", std::process::id()));
        let window = Box2i::from_min_max(V2i::new(-3, -7), V2i::new(-3, -7));
        exr_core::api::Image::new(window)
            .with_channel("Y", ChannelData::Float(vec![1.0001]))
            .write(&path, exr_core::Compression::NoCompression).unwrap();
        let primary = Arc::new(DecodedFrame::open(&path, 0).unwrap());
        assert!(primary.file_types.is_some() && primary.channels.is_empty());
        let inputs = Arc::new(Inputs::Static { frame: primary.clone(), index: 0 });
        let runtime = Runtime::new(crate::ofx::OfxEffects::host_info(), crate::ofx::OfxEffects::bundle_scan());
        let described = runtime.describe(&PluginKey::new("net.ofxrs.basic_gain", 1), HostContext::Filter).unwrap();
        let effect = runtime.effect(12, &described, EffectSetup::default());
        let budget = RegionBudget::new(1024);
        let retained = budget.begin_attempt().unwrap();
        let old = retained.reserve(384).unwrap();
        retained.promote().unwrap();
        let source = EffectSource::new(effect, inputs, primary, Selection::default(),
            0.0, None, Some([0.0, 0.0]), budget.clone()).unwrap();
        let grid = source.grid().unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let work = pool.install(|| source.render_window(&grid, TileId { level: 0, tx: 0, ty: 0 },
            &ProductionContext::default())).unwrap();
        assert_eq!(work.pixel(0, 0).unwrap().map(f32::to_bits),
            [1.0001f32.to_bits(), 1.0001f32.to_bits(), 1.0001f32.to_bits(), 1.0f32.to_bits()]);
        assert_eq!(budget.used(), 384 + work.bytes(), "unowned decode pool is retired at the render boundary");
        let upload = work.upload(&grid, TileId { level: 0, tx: 0, ty: 0 }).unwrap();
        drop(upload);
        drop(work);
        assert_eq!(budget.used(), 384, "failed or completed candidate leaves no hidden scratch owner");
        drop(old); drop(retained);
        assert_eq!(budget.used(), 0);
        std::fs::remove_file(path).unwrap();
    }

}
