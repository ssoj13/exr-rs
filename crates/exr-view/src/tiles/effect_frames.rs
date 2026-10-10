//! Immutable worker-side OpenFX input snapshots, never waiting behind their render.
use std::sync::{Arc, Mutex};
use crate::image::{DecodedFrame, Mapping, Selection};
use ofx_runtime::{FrameInput, InputFrame, InputRequest, InputSource};
use ofx_runtime::ofx_host::{ClipName, CpuFrame, OfxRectD, PixelComponents, PixelDepth, PixelLayout, PixelStorage, Premultiplication, StorageKind};
use super::effect_geometry::{CanonicalAnchor, PixelAspect, discrete_time};
use super::region::{RegionBudget, RegionData, RegionError};
use super::source::ProductionContext;

#[derive(Clone)]
pub(crate) enum FrameOwner {
    Static(Arc<DecodedFrame>),
    Held(scancache::HeldFrame<DecodedFrame>),
}
impl FrameOwner {
    pub(crate) fn frame(&self)->&DecodedFrame {
        match self {Self::Static(frame)=>frame,Self::Held(frame)=>frame}
    }
    pub(crate) fn valid(&self)->bool {
        match self {Self::Static(_)=>true,Self::Held(frame)=>frame.is_valid()}
    }
}
pub(crate) enum Fetch {
    Ready(FrameOwner),
    Pending,
    Blank,
    Failed(String),
    Cancelled,
}
/// A static clip has precisely one known frame; a demand distinguishes exact gaps.
pub(crate) enum Inputs {
    Static { frame:Arc<DecodedFrame>, index:i32 },
    Demand { lease:scancache::DemandLease<DecodedFrame>, primary_index:i32, primary_id:u64 },
}
impl Inputs {
    pub(crate) fn fetch(&self,index:i32)->Fetch {
        match self {
            Self::Static {frame,index:present}=> {
                if index==*present {Fetch::Ready(FrameOwner::Static(frame.clone()))} else {Fetch::Blank}
            }
            Self::Demand{lease,primary_index,primary_id}=>{
                match lease.fetch(index).state {
                    scancache::DemandState::Ready(frame)=>{
                        if !frame.is_valid() || (index==*primary_index && frame.id!=*primary_id) {Fetch::Cancelled}
                        else {Fetch::Ready(FrameOwner::Held(frame))}
                    }
                    scancache::DemandState::Pending(_)=>Fetch::Pending,
                    scancache::DemandState::Missing=>Fetch::Blank,
                    scancache::DemandState::Failed(error)=>Fetch::Failed(error.to_string()),
                    scancache::DemandState::Closed=>Fetch::Failed("frame cache is closed".into()),
                }
            }
        }
    }
    pub(crate) fn readiness_epoch(&self)->u64 {
        match self {Self::Static{..}=>0,Self::Demand{lease,..}=>lease.readiness_epoch()}
    }
}

/// One render/query's bounded original-precision providers and validity witnesses.
pub(crate) const MAX_INPUT_FRAMES: usize = 16;

// Per-source test evidence contains metadata only: no images or admission owners.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct ProviderTrace {
    pub(crate) planned: Mutex<Vec<Vec<(ClipName, OfxRectD)>>>,
    pub(crate) delivered: Mutex<Vec<ProviderObservation>>,
}
#[cfg(test)]
pub(crate) struct ProviderObservation {
    pub(crate) time: f64,
    pub(crate) scale: [f64; 2],
    pub(crate) roi: Option<OfxRectD>,
    pub(crate) plan: super::region::RegionPlan,
    pub(crate) bounds: ofx_runtime::ofx_host::OfxRectI,
    pub(crate) full: ofx_runtime::ofx_host::OfxRectI,
    pub(crate) row: i32,
    pub(crate) bytes: usize,
    pub(crate) par: f64,
}

pub(crate) struct RenderInputs<'a> {
    pub(crate) inputs:&'a Inputs,
    pub(crate) anchor:CanonicalAnchor,
    pub(crate) view:&'a Selection,
    pub(crate) context:&'a ProductionContext,
    pub(crate) budget:&'a RegionBudget,
    pub(crate) owners:Mutex<Vec<FrameOwner>>,
    pub(crate) outcome:Mutex<Option<RegionError>>,
    #[cfg(test)]
    pub(crate) trace: Option<&'a ProviderTrace>,
}
impl RenderInputs<'_> {
    fn retain(&self, owner: FrameOwner) -> Result<(), RegionError> {
        if !owner.valid() || self.context.is_cancelled() {
            return Err(RegionError::Cancelled);
        }
        let mut owners = self.owners.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if owners.iter().any(|held| held.frame().id == owner.frame().id) {
            drop(owners);
            return Ok(());
        }
        if owners.len() == MAX_INPUT_FRAMES {
            drop(owners);
            return Err(RegionError::Failed("effect exceeded its distinct input-frame limit".into()));
        }
        owners.push(owner);
        Ok(())
    }
    fn aspect(&self, frame: &DecodedFrame) -> Result<PixelAspect, RegionError> {
        let par = PixelAspect::new(f64::from(frame.pixel_aspect))?;
        if par != self.anchor.aspect() {
            return Err(RegionError::Unsupported);
        }
        Ok(par)
    }
    fn fail(&self,error:RegionError)->InputFrame {
        let pending=matches!(error,RegionError::Pending|RegionError::Cancelled);
        let text=error.to_string();
        let mut outcome=self.outcome.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Preserve a terminal or cancellation reason over a simultaneous Pending fetch.
        if outcome.is_none() || matches!(outcome.as_ref(),Some(RegionError::Pending)) {*outcome=Some(error)}
        if pending {InputFrame::Pending} else {InputFrame::Failed(text)}
    }
    pub(crate) fn valid(&self)->bool {
        !self.context.is_cancelled() && self.owners.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().all(FrameOwner::valid)
    }
    pub(crate) fn take_error(&self)->Option<RegionError> {
        self.outcome.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    }
}
struct RegionPixels(Arc<RegionData>);
impl PixelStorage for RegionPixels {
    fn bytes(&self)->&[u8] {bytemuck::cast_slice(self.0.rgba())}
}
impl FrameInput<u32> for RenderInputs<'_> {
    fn frame(&self,request:&InputRequest<'_,u32>)->InputFrame {
        if self.context.is_cancelled() {return self.fail(RegionError::Cancelled)}
        if !matches!(request.source, InputSource::Primary) {return self.fail(RegionError::Unsupported)}
        let index=match discrete_time(request.time) {Ok(index)=>index,Err(error)=>return self.fail(error.into())};
        let owner=match self.inputs.fetch(index) {
            Fetch::Ready(owner)=>owner,
            Fetch::Pending=>return self.fail(RegionError::Pending),
            Fetch::Blank=>return InputFrame::Blank,
            Fetch::Failed(error)=>return self.fail(error.into()),
            Fetch::Cancelled=>return self.fail(RegionError::Cancelled),
        };
        let frame=owner.frame();
        let par=match self.aspect(frame) {Ok(par)=>par,Err(error)=>return self.fail(error)};
        if let Err(error)=self.retain(owner.clone()) {return self.fail(error)}
        let level=match sampling_level(request.render_scale) {Ok(level)=>level,Err(error)=>return self.fail(error.into())};
        let (plan,bounds)=match self.anchor.region(frame.data_window,par,level,request.region) {
            Ok(Some(region))=>region,Ok(None)=>return InputFrame::Blank,Err(error)=>return self.fail(error.into()),
        };
        let region=if let Some(types)=&frame.file_types {
            let names:Vec<&str>=frame.channel_names.iter().map(String::as_str).collect();
            let dw=frame.data_window;
            match super::file::FileSource::new(frame.path.clone(),((dw.min.x,dw.min.y),frame.width,frame.height),
                &names,types,Mapping::of(&names,frame.resolve(self.view).as_deref())) {
                Ok(source)=>super::source::TileSource::read_region(&source,plan,self.context,self.budget),
                Err(error)=>Err(error.into()),
            }
        } else {super::effect_input::planar_region(frame,self.view,plan,self.context,self.budget)};
        let region=match region {Ok(region)=>region,Err(error)=>return self.fail(error)};
        if region.plan() != plan {
            return self.fail(RegionError::Failed("original provider changed its checked sampling declaration".into()));
        }
        if !owner.valid() || self.context.is_cancelled() {return self.fail(RegionError::Cancelled)}
        let full=match self.anchor.scaled_bounds(frame.data_window,par,level){Ok(full)=>full,Err(error)=>return self.fail(error.into())};
        let row=match i32::try_from(i64::from(bounds.x2)-i64::from(bounds.x1)).ok().and_then(|w|w.checked_mul(16)) {
            Some(row)=>row,None=>return self.fail("input row bytes overflow".into()),
        };
        let layout=match PixelLayout::new(PixelDepth::Float,PixelComponents::Rgba,bounds,Some(row),StorageKind::Host) {
            Ok(layout)=>layout,Err(_)=>return self.fail("input layout is invalid".into()),
        };
        let picture=match CpuFrame::from_storage(layout,full,Arc::new(RegionPixels(region)),par.value(),Premultiplication::Premultiplied) {
            Ok(picture)=>picture,Err(_)=>return self.fail("input image metadata is invalid".into()),
        };
        #[cfg(test)]
        if let Some(trace) = self.trace {
            trace.delivered.lock().unwrap().push(ProviderObservation {
                time: request.time, scale: request.render_scale, roi: request.region,
                plan, bounds, full, row, bytes: picture.bytes().len(), par: par.value(),
            });
        }
        InputFrame::Frame(picture)
    }
    fn region_of_definition(&self,_clip:&ClipName,source:&InputSource<u32>,time:f64)->Option<OfxRectD>{
        if !matches!(source, InputSource::Primary) {self.fail(RegionError::Unsupported);return None}
        let index=match discrete_time(time){Ok(index)=>index,Err(error)=>{self.fail(error.into());return None}};
        match self.inputs.fetch(index) {
            Fetch::Ready(owner)=>{
                let frame=owner.frame();
                let rod=self.aspect(frame).and_then(|par|self.anchor.rod(frame.data_window,par).map_err(RegionError::from));
                match rod {Ok(rod)=>{
                    if let Err(error)=self.retain(owner) {self.fail(error);return None}
                    Some(rod)
                },Err(error)=>{self.fail(error);None}}
            }
            Fetch::Pending=>{self.fail(RegionError::Pending);None}
            Fetch::Blank=>Some(OfxRectD{x1:0.,y1:0.,x2:0.,y2:0.}),
            Fetch::Failed(error)=>{self.fail(error.into());None}
            Fetch::Cancelled=>{self.fail(RegionError::Cancelled);None}
        }
    }
}
pub(crate) fn sampling_level(scale:[f64;2])->Result<u32,String>{
    if scale[0]!=scale[1] || !scale[0].is_finite() || scale[0]<=0. || scale[0]>1. {
        return Err("viewer inputs require equal positive dyadic render scales".into())
    }
    let level=-scale[0].log2();
    if level.fract()!=0. || !(0. ..=40.).contains(&level) {
        return Err("viewer input render scale is outside the checked sampling pyramid".into())
    }
    Ok(level as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_negative_raw_gap_and_loading_neighbor_remain_distinct_and_par_is_checked() {
        let (release, blocked) = std::sync::mpsc::channel::<()>();
        let blocked = Arc::new(Mutex::new(blocked));
        let (started, entered) = std::sync::mpsc::channel::<()>();
        let cache = scancache::Scancache::new(vec![-2, 0], 1024, 1,
            Arc::new(DecodedFrame::byte_size), Arc::new(move |index, _| {
                if index == 0 {
                    started.send(()).unwrap();
                    // Sender drop also releases this fixture after an assertion unwind.
                    let _ = blocked.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10));
                }
                let mut frame = crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
                frame.id = if index == -2 { 41 } else { 42 };
                frame.pixel_aspect = if index == -2 { 1.5 } else { 1.25 };
                Ok(frame)
            }), Arc::new(|| {}));
        let release_on_exit = release;
        let inputs = Inputs::Demand {
            lease: cache.open_demand(scancache::HoldingBudget::new(1024), MAX_INPUT_FRAMES).unwrap(),
            primary_index: -2, primary_id: 41,
        };
        let primary = ready(&inputs, -2);
        let aspect = PixelAspect::new(1.5).unwrap();
        let anchor = CanonicalAnchor::new(primary.frame().display_window, aspect).unwrap();
        let context = ProductionContext::default();
        let budget = RegionBudget::new(4096);
        let view = Selection::default();
        let frames = RenderInputs { inputs: &inputs, anchor, view: &view,
            context: &context, budget: &budget, owners: Mutex::new(Vec::new()),
            outcome: Mutex::new(None), trace: None };
        let source = InputSource::Primary;
        let clip = ClipName::source();
        let gap = frames.region_of_definition(&clip, &source, -0.1).unwrap();
        assert_eq!((gap.x1, gap.y1, gap.x2, gap.y2), (0.0, 0.0, 0.0, 0.0),
            "floor(-0.1) is the actual absent raw index -1, not a nearest present frame");
        assert!(frames.take_error().is_none());
        assert!(frames.region_of_definition(&clip, &source, 0.25).is_none());
        assert_eq!(frames.take_error(), Some(RegionError::Pending),
            "an exact loading raw frame must never be replaced with transparent Blank");
        entered.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        release_on_exit.send(()).unwrap();
        let neighbor = ready(&inputs, 0);
        assert_eq!(neighbor.frame().id, 42);
        assert!(frames.region_of_definition(&clip, &source, 0.25).is_none());
        assert_eq!(frames.take_error(), Some(RegionError::Unsupported),
            "temporal input PAR cannot silently change the primary canonical lattice");
        drop(neighbor);
        drop(primary);
        drop(frames);
        drop(inputs);
        assert_eq!(budget.used(), 0, "metadata gap/PAR queries did not allocate effect pixels");
    }

    fn ready(inputs: &Inputs, index: i32) -> FrameOwner {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match inputs.fetch(index) {
                Fetch::Ready(owner) => return owner,
                Fetch::Pending => {},
                _ => panic!("exact frame did not remain a ready-or-pending demand"),
            }
            assert!(std::time::Instant::now() < deadline, "exact frame remained pending");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    #[test]
    fn external_old_demand_owner_blocks_new_generation_in_the_same_holding_partition() {
        let holding = scancache::HoldingBudget::new(16);
        let cache = scancache::Scancache::new(vec![0, 1], 1024, 1,
            Arc::new(DecodedFrame::byte_size), Arc::new(|index, _| {
                let mut frame = crate::image::tests_support::frame_with(&["R", "G", "B", "A"]);
                frame.id = index as u64 + 1;
                Ok(frame)
            }), Arc::new(|| {}));
        let old = Inputs::Demand { lease: cache.open_demand(holding.clone(), MAX_INPUT_FRAMES).unwrap(),
            primary_index: 0, primary_id: 1 };
        let external = ready(&old, 0);
        assert_eq!(holding.used_bytes(), 16);
        drop(old);
        let next = Inputs::Demand { lease: cache.open_demand(holding.clone(), MAX_INPUT_FRAMES).unwrap(),
            primary_index: 1, primary_id: 2 };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match next.fetch(1) {
                Fetch::Pending => {},
                Fetch::Ready(_) => panic!("old external owner was bypassed by a new holding budget"),
                _ => panic!("holding pressure became a missing or terminal input"),
            }
            if cache.get(1).is_some() { break; }
            assert!(std::time::Instant::now() < deadline, "new exact frame was not decoded");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(holding.used_bytes(), 16);
        drop(external);
        let admitted = ready(&next, 1);
        assert_eq!(admitted.frame().id, 2);
        assert_eq!(holding.used_bytes(), 16);
        drop(admitted); drop(next);
        assert_eq!(holding.used_bytes(), 0);
    }
}
