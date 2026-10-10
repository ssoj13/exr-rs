//! Canonical OpenFX geometry anchored to the primary display window.
use imath_rs::Box2i;
use ofx_runtime::ofx_host::{OfxRectD, OfxRectI};
use super::grid::PixRect;
use super::region::{RegionPlan, RegionPolicy, SamplingPhase};

/// Validated header pixel aspect ratio; UI-formatted attribute text is never parsed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PixelAspect(f64);
impl PixelAspect {
    pub(crate) fn new(value: f64) -> Result<Self, String> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err("pixel aspect ratio must be finite and positive".into())
        }
    }
    pub(crate) fn value(self) -> f64 { self.0 }
}

/// One immutable primary display-window anchor, shared by every temporal input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CanonicalAnchor {
    x: i64,
    top: i64,
    aspect: PixelAspect,
}
impl CanonicalAnchor {
    pub(crate) fn aspect(self) -> PixelAspect { self.aspect }
    pub(crate) fn new(display: Box2i, aspect: PixelAspect) -> Result<Self, String> {
        checked_extent(display)?;
        Ok(Self {
            x: i64::from(display.min.x),
            top: i64::from(display.max.y) + 1,
            aspect,
        })
    }
    /// Integer Y-up bounds before applying render scale, half open.
    pub(crate) fn original_bounds(self, data: Box2i, aspect: PixelAspect) -> Result<(i64,i64,i64,i64), String> {
        checked_extent(data)?;
        if self.aspect != aspect {
            return Err("temporal frame pixel aspect differs from the primary frame".into());
        }
        Ok((i64::from(data.min.x)-self.x, self.top-i64::from(data.max.y)-1,
            i64::from(data.max.x)+1-self.x, self.top-i64::from(data.min.y)))
    }
    pub(crate) fn rod(self, data: Box2i, aspect: PixelAspect) -> Result<OfxRectD,String> {
        let (x1,y1,x2,y2)=self.original_bounds(data,aspect)?;
        let rod=OfxRectD{x1:x1 as f64*self.aspect.0,y1:y1 as f64,x2:x2 as f64*self.aspect.0,y2:y2 as f64};
        if ![rod.x1,rod.y1,rod.x2,rod.y2].iter().all(|v|v.is_finite()) {return Err("canonical bounds overflow finite coordinates".into())}
        Ok(rod)
    }
    pub(crate) fn scaled_bounds(self, data: Box2i, aspect: PixelAspect, level:u32) -> Result<OfxRectI,String> {
        let (x1,y1,x2,y2)=self.original_bounds(data,aspect)?;
        let step=i64::try_from(1u64.checked_shl(level).ok_or("sampling scale overflows")?).map_err(|_|"sampling scale overflows")?;
        let floor=|n:i64|i32::try_from(n.div_euclid(step)).map_err(|_|"scaled coordinate exceeds OpenFX integer bounds".to_owned());
        let ceil=|n:i64|i32::try_from(-(-n).div_euclid(step)).map_err(|_|"scaled coordinate exceeds OpenFX integer bounds".to_owned());
        Ok(OfxRectI{x1:floor(x1)?,y1:floor(y1)?,x2:ceil(x2)?,y2:ceil(y2)?})
    }
    /// Enclose the requested canonical region and intersect the actual frame.
    pub(crate) fn region(self,data:Box2i,aspect:PixelAspect,level:u32,roi:Option<OfxRectD>)->Result<Option<(RegionPlan,OfxRectI)>,String> {
        let bounds=self.scaled_bounds(data,aspect,level)?;
        let mut wanted=bounds;
        if let Some(r)=roi {
            if ![r.x1,r.y1,r.x2,r.y2].iter().all(|v|v.is_finite()) || r.x1>r.x2 || r.y1>r.y2 {
                return Err("canonical region must have finite ordered bounds".into());
            }
            let scale=2.0_f64.powi(-i32::try_from(level).map_err(|_|"sampling scale overflows")?);
            let coord=|v:f64,up:bool|->Result<i32,String>{
                let n=if up {v.ceil()}else{v.floor()};
                if n<i32::MIN as f64 || n>i32::MAX as f64 {Err("canonical region exceeds OpenFX integer bounds".into())}else{Ok(n as i32)}
            };
            wanted=OfxRectI{x1:coord(r.x1/self.aspect.0*scale,false)?.max(bounds.x1),
                y1:coord(r.y1*scale,false)?.max(bounds.y1),
                x2:coord(r.x2/self.aspect.0*scale,true)?.min(bounds.x2),
                y2:coord(r.y2*scale,true)?.min(bounds.y2)};
        }
        if wanted.x1>=wanted.x2 || wanted.y1>=wanted.y2 {return Ok(None)}
        let (x0,y0,_,_)=self.original_bounds(data,aspect)?;
        let size=checked_extent(data)?;
        let rect=PixRect{x:u32::try_from(i64::from(wanted.x1)-i64::from(bounds.x1)).map_err(|_|"local region overflows")?,
            y:u32::try_from(i64::from(wanted.y1)-i64::from(bounds.y1)).map_err(|_|"local region overflows")?,
            w:u32::try_from(i64::from(wanted.x2)-i64::from(wanted.x1)).map_err(|_|"local region overflows")?,
            h:u32::try_from(i64::from(wanted.y2)-i64::from(wanted.y1)).map_err(|_|"local region overflows")?};
        let plan=RegionPlan::on_lattice(size,level,rect,RegionPolicy::OriginalArea,SamplingPhase::BOTTOM_LEFT,(x0,y0))?;
        Ok(Some((plan,wanted)))
    }
}
pub(crate) fn checked_extent(b:Box2i)->Result<(u32,u32),String>{
    let w=i64::from(b.max.x)-i64::from(b.min.x)+1;
    let h=i64::from(b.max.y)-i64::from(b.min.y)+1;
    if w<=0 || h<=0 {return Err("image window is empty".into())}
    Ok((u32::try_from(w).map_err(|_|"image width overflows")?,u32::try_from(h).map_err(|_|"image height overflows")?))
}
/// Viewer policy: discrete inputs hold the checked floor of the unchanged OFX time.
pub(crate) fn discrete_time(time:f64)->Result<i32,String>{
    let floor=time.floor();
    if !time.is_finite() || floor<i32::MIN as f64 || floor>i32::MAX as f64 {
        Err("input time is not a finite representable frame index".into())
    } else {Ok(floor as i32)}
}

#[cfg(test)]
mod tests {
    use super::*;
    use imath_rs::Vec2;
    fn window(x:i32,y:i32,w:i32,h:i32)->Box2i {Box2i::from_min_max(Vec2::new(x,y),Vec2::new(x+w-1,y+h-1))}
    #[test]
    fn temporal_windows_share_primary_display_anchor_and_par() {
        let par=PixelAspect::new(1.5).unwrap();
        let anchor=CanonicalAnchor::new(window(-7,10,12,9),par).unwrap();
        let data=window(-5,12,4,5);
        let r=anchor.rod(data,par).unwrap();
        assert_eq!([r.x1,r.y1,r.x2,r.y2],[3.0,2.0,9.0,7.0]);
        let b=anchor.scaled_bounds(data,par,1).unwrap();
        assert_eq!([b.x1,b.y1,b.x2,b.y2],[1,1,3,4]);
        let temporal=anchor.rod(window(-8,8,4,5),par).unwrap();
        assert_eq!([temporal.x1,temporal.y1,temporal.x2,temporal.y2],[-1.5,6.0,4.5,11.0]);
        assert!(anchor.rod(data,PixelAspect::new(1.0).unwrap()).is_err());
    }
    #[test]
    fn canonical_roi_encloses_and_clips_without_resetting_odd_bottom_phase() {
        let par=PixelAspect::new(2.0).unwrap();
        let anchor=CanonicalAnchor::new(window(10,-4,7,7),par).unwrap();
        let data=window(11,-3,4,5);
        let (plan,b)=anchor.region(data,par,1,Some(OfxRectD{x1:2.5,y1:1.2,x2:7.1,y2:3.1})).unwrap().unwrap();
        assert_eq!([b.x1,b.y1,b.x2,b.y2],[0,0,2,2]);
        assert_eq!(plan.lattice_origin(),(1,1));
        assert_eq!(plan.phase(),SamplingPhase::BOTTOM_LEFT);
        assert_eq!(plan.rows(),2..5);
        assert!(anchor.region(data,par,0,Some(OfxRectD{x1:100.,y1:100.,x2:101.,y2:101.})).unwrap().is_none());
    }
    #[test]
    fn time_hold_is_checked_floor_not_nearest_or_truncation() {
        assert_eq!(discrete_time(-0.25).unwrap(),-1);
        assert_eq!(discrete_time(-3.0).unwrap(),-3);
        assert_eq!(discrete_time(4.99).unwrap(),4);
        assert_eq!(discrete_time(i32::MIN as f64).unwrap(),i32::MIN);
        assert!(discrete_time(i32::MAX as f64+1.).is_err());
        assert!(discrete_time(i32::MIN as f64-0.1).is_err());
        for bad in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY] {assert!(discrete_time(bad).is_err());}
        for bad in [0.,-1.,f64::NAN,f64::INFINITY] {assert!(PixelAspect::new(bad).is_err());}
    }
}
