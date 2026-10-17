//! Shared float-canvas output used by exr-view and Playa.
pub use soluwell_viewport::display::{CANVAS_FORMAT, Output, PresentPass, Target, HDR_REF_WHITE, SCRGB_NITS, PQ_PEAK, pq, rec2020_nits};
#[cfg(test)]
pub use soluwell_viewport::display::{Encoding, encode, HLG_REF_PEAK};
