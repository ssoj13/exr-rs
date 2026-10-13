//! Borrowed original channel planes through the shared Mapping and region reducer.
use std::sync::Arc;
use crate::image::{DecodedFrame, Mapping, Selection};
use super::region::{RegionAccumulator, RegionBudget, RegionData, RegionError, RegionPlan, RegionPolicy};
use super::source::ProductionContext;
/// `decode` is the frame's stage 1 for the layer (`image::layer_decode`), applied to each row
/// before it is reduced, so region levels average scene_linear light.
pub(crate) fn planar_region(frame:&DecodedFrame,view:&Selection,decode:&crate::ocio::Decode,plan:RegionPlan,context:&ProductionContext,budget:&RegionBudget)->Result<Arc<RegionData>,RegionError>{
    if frame.file_types.is_some() {return Err(RegionError::Unsupported)}
    if plan.size()!=(frame.width,frame.height) {return Err("region source geometry does not match original planes".into())}
    if plan.policy()==RegionPolicy::MatchingStoredNative && plan.level()>0 {return Err(RegionError::Unsupported)}
    if context.is_cancelled() {return Err(RegionError::Cancelled)}
    let names:Vec<&str>=frame.channel_names.iter().map(String::as_str).collect();
    let mapping=Mapping::of(&names,frame.resolve(view).as_deref());
    let bound=mapping.bind(&frame.channels).map_err(|e|RegionError::Failed(e.to_string()))?;
    let row_samples=(frame.width as usize).checked_mul(4).ok_or("original row size overflows")?;
    let accumulator_peak = if plan.level() == 0 { plan.output_bytes() } else {
        plan.output_bytes().checked_add(plan.sums_bytes()).ok_or("original accumulator size overflows")?
    };
    let working=accumulator_peak.checked_add(row_samples.checked_mul(4).ok_or("original row size overflows")?)
        .ok_or("original region working size overflows")?;
    let mut permit=budget.reserve(working)?;
    let mut acc=RegionAccumulator::new(plan)?;
    let mut row=Vec::new();
    row.try_reserve_exact(row_samples).map_err(|e|RegionError::Failed(e.to_string()))?;
    let accumulator_capacity = if plan.level() == 0 { acc.bytes() } else {
        plan.output_bytes().checked_add(acc.bytes()).ok_or("original accumulator capacity overflows")?
    };
    permit.resize(accumulator_capacity.checked_add(row.capacity().checked_mul(4).ok_or("original row capacity overflows")?)
        .ok_or("original row capacity overflows")?)?;
    row.resize(row_samples,0.);
    for y in acc.rows(){
        if context.is_cancelled() {return Err(RegionError::Cancelled)}
        for x in 0..frame.width {
            let i=y as usize*frame.width as usize+x as usize;
            row[x as usize*4..x as usize*4+4].copy_from_slice(&bound.sample(i).map_err(|e|RegionError::Failed(e.to_string()))?);
        }
        decode.apply(&mut row);
        acc.push_row(y,&row)?;
    }
    if context.is_cancelled() {return Err(RegionError::Cancelled)}
    drop(row);
    RegionData::finish(plan,acc.finish()?,permit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::tests_support::frame_with;
    use exr_core::api::ChannelData;
    use crate::tiles::grid::PixRect;
    use crate::tiles::region::SamplingPhase;

    fn full() -> RegionPlan {
        RegionPlan::new((1,1),0,PixRect{x:0,y:0,w:1,h:1},
            RegionPolicy::OriginalArea,SamplingPhase::BOTTOM_LEFT).unwrap()
    }
    #[test]
    fn original_float_bits_survive_without_rgba_materialization() {
        let bits = [0x8000_0000, 0x7fc1_2345, 0x7f80_0000, 0xff80_0000];
        let mut frame = frame_with(&["R","G","B","A"]);
        for ((_,channel),bit) in frame.channels.iter_mut().zip(bits) {
            *channel=ChannelData::Float(vec![f32::from_bits(bit)]);
        }
        let budget = RegionBudget::new(32);
        let region = planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),
            &ProductionContext::default(),&budget).unwrap();
        assert_eq!(region.rgba().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),bits);
        assert_eq!(budget.used(),16,"only retained output remains charged");
        drop(region);
        assert_eq!(budget.used(),0);
    }
    #[test]
    fn typed_uint_mapping_and_cancel_share_the_original_provider() {
        let mut frame=frame_with(&["R","G","B"]);
        frame.channels[0].1=ChannelData::Uint(vec![u32::MAX]);
        frame.channels[1].1=ChannelData::Uint(vec![0]);
        frame.channels[2].1=ChannelData::Uint(vec![u32::MAX/2]);
        let budget=RegionBudget::new(32);
        let context=ProductionContext::default();
        let region=planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),&context,&budget).unwrap();
        assert_eq!(region.rgba(),[1.0,0.0,(u32::MAX/2) as f32/u32::MAX as f32,1.0]);
        drop(region);
        context.cancel();
        assert!(matches!(planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),&context,&budget),
            Err(RegionError::Cancelled)));
        assert_eq!(budget.used(),0);
    }
    #[test]
    fn intrinsic_peak_and_temporary_pressure_are_distinct() {
        let frame=frame_with(&["R","G","B"]);
        let context=ProductionContext::default();
        assert!(matches!(planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),&context,
            &RegionBudget::new(31)),Err(RegionError::Budget{needed:32,limit:31})));
        let budget=RegionBudget::new(32);
        let held=budget.reserve(1).unwrap();
        assert!(matches!(planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),&context,&budget),
            Err(RegionError::Pending)));
        drop(held);
        let region=planar_region(&frame,&Selection::default(),&crate::ocio::Decode::identity(),full(),&context,&budget).unwrap();
        drop(region);
        assert_eq!(budget.used(),0);
    }
    #[test]
    fn original_region_reduces_srgb_in_linear_light() {
        let mut frame = crate::image::tests_support::frame_of(2, 2, vec![
            ("R".into(), ChannelData::Float(vec![0.0, 1.0, 1.0, 0.0])),
            ("G".into(), ChannelData::Float(vec![0.0, 1.0, 1.0, 0.0])),
            ("B".into(), ChannelData::Float(vec![0.0, 1.0, 1.0, 0.0])),
        ]);
        frame.tag = exr_import::ColorTag { space: "srgb_rec709_display".into(), origin: exr_import::TagOrigin::File };
        let decode = crate::ocio::Decoder::builtin_only().decode(&frame).unwrap();
        let plan = RegionPlan::new((2,2),1,PixRect{x:0,y:0,w:1,h:1},
            RegionPolicy::OriginalArea,SamplingPhase::BOTTOM_LEFT).unwrap();
        let region = planar_region(&frame,&Selection::default(),&decode,plan,
            &ProductionContext::default(),&RegionBudget::new(4096)).unwrap();
        assert_eq!(region.rgba(), [0.5, 0.5, 0.5, 1.0]);
    }

}
